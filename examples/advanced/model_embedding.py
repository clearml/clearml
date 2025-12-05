import os
import sys
import argparse
import json
import requests
import numpy as np
from typing import List, Dict, Any

try:
    from PyPDF2 import PdfReader
except ImportError:
    raise ImportError(
        "PyPDF2 is required to extract text from PDF files. Install via `pip install PyPDF2`."
    )


def chunk_text(text: str, chunk_size: int) -> List[str]:
    """
    Split text into chunks of at most chunk_size characters.
    """
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(text[start:end])
        start = end
    return chunks


class EmbeddingService:
    """
    Wrapper around an OpenAI-compatible embedding service.
    """

    def __init__(self, api_url: str, model: str):
        if "CLEARML_AUTH_TOKEN" not in os.environ:
            print("Please set the `CLEARML_AUTH_TOKEN` env var")
            sys.exit(1)
        self.api_key = os.environ["CLEARML_AUTH_TOKEN"]
        self.api_url = api_url
        self.model = model
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Fetch embeddings for a list of texts.
        """
        payload = {"model": self.model, "input": texts}
        resp = requests.post(
            f"{self.api_url}/embeddings", headers=self.headers, json=payload
        )
        resp.raise_for_status()
        data = resp.json()
        return [item["embedding"] for item in data.get("data", [])]


class DocumentStore:
    """
    Persistent store for documents and their embeddings.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        store_path: str = "embeddings_store.json",
        similarity: str = "cosine",
    ):
        self.embedding_service = embedding_service
        self.similarity = similarity
        self.store_path = store_path
        self.docs: List[Dict[str, Any]] = []
        self._load_store()

    def _load_store(self):
        if os.path.exists(self.store_path):
            with open(self.store_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for item in raw:
                self.docs.append(
                    {
                        "id": item["id"],
                        "text": item["text"],
                        "embedding": np.array(item["embedding"], dtype=np.float32),
                    }
                )

    def _save_store(self):
        serializable = []
        for doc in self.docs:
            serializable.append(
                {
                    "id": doc["id"],
                    "text": doc["text"],
                    "embedding": doc["embedding"].tolist(),
                }
            )
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)

    def add_document(self, identifier: str, text: str):
        """
        Add a single document chunk by extracting its embedding.
        """
        embedding = self.embedding_service.embed([text])[0]
        vec = np.array(embedding, dtype=np.float32)
        self.docs.append({"id": identifier, "text": text, "embedding": vec})
        self._save_store()

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query the store for the top_k most similar document chunks.
        """
        q_emb = np.array(
            self.embedding_service.embed([query_text])[0], dtype=np.float32
        )
        results = []
        for doc in self.docs:
            d_emb = doc["embedding"]
            if self.similarity == "cosine":
                sim = np.dot(q_emb, d_emb) / (
                    np.linalg.norm(q_emb) * np.linalg.norm(d_emb)
                )
            elif self.similarity == "dot":
                sim = np.dot(q_emb, d_emb)
            else:
                raise ValueError(f"Unknown similarity metric: {self.similarity}")
            results.append({"id": doc["id"], "score": float(sim), "text": doc["text"]})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


def load_text(path: str) -> str:
    """
    Load plain text or extract text from PDF.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    if path.lower().endswith(".pdf"):
        reader = PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main():
    parser = argparse.ArgumentParser(
        description="Persistent embedding service with chunking"
    )
    parser.add_argument(
        "--api-url",
        required=True,
        help="Embedding service URL endpoint as seen in ClearML UI",
    )
    parser.add_argument(
        "--model", required=True, help="Embedding model name as seen in ClearML UI"
    )
    parser.add_argument(
        "--store-file",
        default="embeddings_store.json",
        help="Path to persistent embeddings store",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Max characters per chunk when ingesting large files",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands: ingest, query")

    ingest_parser = subparsers.add_parser(
        "ingest", help="Ingest documents for embedding"
    )
    ingest_parser.add_argument(
        "files", nargs="+", help="Paths to document files (txt or pdf)"
    )

    query_parser = subparsers.add_parser("query", help="Query nearest document chunks")
    query_parser.add_argument("text", help="Query text")
    query_parser.add_argument(
        "--top-k", type=int, default=5, help="Number of top results to return"
    )
    query_parser.add_argument(
        "--metric",
        choices=["cosine", "dot"],
        default="cosine",
        help="Similarity metric",
    )

    args = parser.parse_args()

    service = EmbeddingService(api_url=args.api_url, model=args.model)
    store = DocumentStore(
        embedding_service=service,
        store_path=args.store_file,
        similarity=getattr(args, "metric", "cosine"),
    )

    if args.command == "ingest":
        for filepath in args.files:
            print(f"Loading {filepath}...")
            txt = load_text(filepath)
            chunks = chunk_text(txt, args.chunk_size)
            base_id = os.path.basename(filepath)
            for i, chunk in enumerate(chunks, start=1):
                chunk_id = f"{base_id}::chunk{i}"
                store.add_document(chunk_id, chunk)
        print(f"Ingested {len(store.docs)} chunks (persisted to {args.store_file}).")

    elif args.command == "query":
        results = store.query(args.text, top_k=args.top_k)
        print(f"Top {len(results)} results:")
        for idx, res in enumerate(results, start=1):
            print(
                f"{idx}. ID: {res['id']}, Score: {res['score']:.4f}\n{res['text'][:200]}...\n"
            )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
