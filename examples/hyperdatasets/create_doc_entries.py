"""Create a HyperDataset populated with Markdown documentation links.

The script downloads a predefined (or user-supplied) list of Markdown files,
captures their content and metadata inside HyperDataset entries, and
optionally generates vector embeddings for semantic search.

Example usage::

    python examples/hyperdatasets/doc_entries_create.py \
        --project "HyperDatasets Examples" \
        --dataset-name "docs-demo" \
        --version-name "version" \
        --embed \
        --vector-field doc_vector
"""

import argparse
import textwrap
from pathlib import Path
from typing import List, Optional, Tuple

# ClearML helpers for storage and HyperDataset primitives
from clearml import StorageManager
from clearml.hyperdatasets import (
    HyperDataset,
    DataEntry,
    DataSubEntry,
)


# TODO: make sure you document on how this is used -> the user needs to make sure the class is in python space when fetching the frames so they are properly casted
# ClearML sub-entry capturing the document blob and metadata
class MarkdownDataSubEntry(DataSubEntry):
    def __init__(self, name: str, text: str, path: str, url: str):
        super().__init__(
            name=name,
            source=url,
            metadata={"text": text, "path": path, "url": url},
        )


def parse_args() -> argparse.Namespace:
    description = textwrap.dedent(__doc__ or "").strip()
    parser = argparse.ArgumentParser(
        description=description or None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--project", default="HyperDatasets Examples", help="ClearML project name")
    parser.add_argument("--dataset-name", required=True, help="HyperDataset collection name")
    parser.add_argument("--version-name", required=True, help="HyperDataset version name")
    parser.add_argument("--description", default="Documentation HyperDataset", help="Dataset description")
    parser.add_argument(
        "--doc-url",
        action="append",
        help="Remote Markdown file to ingest (can be supplied multiple times)",
    )
    parser.add_argument("--limit", type=int, default=100, help="Maximum number of documents to ingest")
    parser.add_argument("--embed", action="store_true", help="Generate embeddings for each document")
    parser.add_argument(
        "--vector-field",
        help="Metadata field used to store the embedding vector",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name or path for embeddings",
    )
    parser.add_argument(
        "--embedding-device",
        default=None,
        help="Optional device string passed to SentenceTransformer (e.g. 'cpu', 'cuda')",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="L2-normalize generated embeddings",
    )
    return parser.parse_args()


DEFAULT_DOC_URLS = [
    "https://github.com/clearml/clearml-docs/blob/main/docs/build_interactive_models.md",
    "https://github.com/clearml/clearml-docs/blob/main/docs/clearml_agent.md",
    "https://github.com/clearml/clearml-docs/blob/main/docs/community.md",
    "https://github.com/clearml/clearml-docs/blob/main/docs/custom_apps.md",
    "https://github.com/clearml/clearml-docs/blob/main/docs/deploying_models.md",
]


def load_markdown(path: Path) -> Tuple[str, str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    title = path.stem
    for line in text.splitlines():
        stripped = line.strip().lstrip("# ")
        if stripped:
            title = stripped
            break
    return title, text


def maybe_encode_embeddings(
    documents: List[dict],
    *,
    model_name: str,
    device: Optional[str],
    normalize: bool,
) -> Optional[List[List[float]]]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("sentence-transformers is required for --embed") from exc

    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode([doc["content"] for doc in documents], normalize_embeddings=normalize)
    return [list(map(float, emb)) for emb in embeddings]


def build_entries(
    documents: List[dict],
    embeddings: Optional[List[List[float]]],
    vector_field: Optional[str],
) -> Tuple[List[DataEntry], Optional[int]]:
    entries: List[DataEntry] = []
    vector_dims: Optional[int] = None
    for idx, doc in enumerate(documents):
        metadata = {
            "title": doc["title"],
            "path": doc["relative_path"],
            "url": doc["url"],
            "size_bytes": len(doc["content"].encode("utf-8")),
            "snippet": doc["snippet"],
        }
        entry = DataEntry(metadata=metadata)
        entry.add_sub_entries(
            [
                MarkdownDataSubEntry(
                    name="document",
                    text=doc["content"],
                    path=doc["relative_path"],
                    url=doc["url"],
                )
            ]
        )

        if embeddings:
            if not vector_field:
                raise ValueError("--vector-field must be provided when --embed is used")
            vector = embeddings[idx]
            entry.set_vector(vector, metadata_field=vector_field)
            if vector_dims is None:
                vector_dims = len(vector)
            elif vector_dims != len(vector):
                raise ValueError("All embedding vectors must share the same dimensionality")

        entries.append(entry)
    return entries, vector_dims


def main() -> None:
    args = parse_args()

    documents: List[dict] = []
    urls = args.doc_url or DEFAULT_DOC_URLS
    for url in urls:
        try:
            local_path = Path(StorageManager.get_local_copy(remote_url=url))
        except Exception as exc:
            raise RuntimeError(f"Failed downloading {url}: {exc}")

        title, content = load_markdown(local_path)
        snippet = textwrap.shorten(content.replace("\n", " "), width=240, placeholder="…")
        documents.append(
            {
                "title": title,
                "content": content,
                "relative_path": local_path.name,
                "snippet": snippet,
                "url": url,
            }
        )
        if args.limit and len(documents) >= args.limit:
            break

    if not documents:
        raise ValueError(f"No Markdown files found under {docs_dir}")

    embeddings = None
    if args.embed:
        if not args.vector_field:
            raise ValueError("--vector-field must be provided when --embed is used")
        embeddings = maybe_encode_embeddings(
            documents,
            model_name=args.embedding_model,
            device=args.embedding_device,
            normalize=args.normalize,
        )

    entries, vector_dims = build_entries(documents, embeddings, args.vector_field)

    field_mappings = None
    if vector_dims:
        field_path = args.vector_field if args.vector_field.startswith("meta.") else f"meta.{args.vector_field}"
        # ClearML vector field mapping so the backend knows about the dense embedding metadata
        field_mappings = {
            field_path: {
                "type": "dense_vector",
                "element_type": "float",
                "dims": vector_dims,
            }
        }

    # ClearML HyperDataset version handle (creates the version if needed)
    dataset = HyperDataset(
        project_name=args.project,
        dataset_name=args.dataset_name,
        version_name=args.version_name,
        description=args.description,
        field_mappings=field_mappings,
    )

    # Upload the assembled entries so ClearML manages storage/indexing
    errors = dataset.add_data_entries(entries, upload_local_files_destination=None, force_upload=True)
    if errors.get("register"):
        raise RuntimeError(f"Failed registering entries: {errors['register']}")

    print(
        "Created HyperDataset version: project={project} dataset={dataset} version={version}".format(
            project=dataset.project_id,
            dataset=dataset.dataset_id,
            version=dataset.version_id,
        )
    )


if __name__ == "__main__":
    main()
