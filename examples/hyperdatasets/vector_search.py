"""
Search HyperDataset entries using vector similarity.

The script embeds either free-form text or an input file (typically an image)
and executes a vector search against a HyperDataset version that already stores
embeddings under the specified metadata field.

Examples::

    # Text query (matching qa_entries_create.py output)
    python examples/hyperdatasets/vector_search.py \
        --project "HyperDatasets Examples" \
        --dataset-name "qa-demo" \
        --version-name "version" \
        --vector-field _vector \
        --query "How do I upload local files?"

    # Image query (matching image_entries_create.py output)
    python examples/hyperdatasets/vector_search.py \
        --project "HyperDatasets Examples" \
        --dataset-name "image-demo" \
        --version-name "version" \
        --vector-field _vector \
        --query examples/hyperdatasets/sample_images/image_11.jpg
"""

import argparse
import textwrap
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

# ClearML helper for resolving dataset versions and running vector search
from clearml.hyperdatasets import HyperDatasetManagement


def parse_args() -> argparse.Namespace:
    description = textwrap.dedent(__doc__ or "").strip()
    parser = argparse.ArgumentParser(
        description=description or None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--project", help="ClearML project name used when resolving the dataset")
    parser.add_argument("--dataset-name", required=True, help="HyperDataset collection name")
    parser.add_argument("--version-name", required=True, help="HyperDataset version name")
    parser.add_argument(
        "--vector-field",
        required=True,
        help="Metadata field that stores embeddings on the dataset entries",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Text query or path to a local file that should be embedded",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=5,
        help="Number of nearest neighbors to retrieve",
    )
    parser.add_argument("--fast", action="store_true", help="Use the fast approximate vector search mode")
    parser.add_argument(
        "--similarity",
        default="cosine",
        choices=["cosine", "l2_norm", "dot_product"],
        help="Similarity function to apply",
    )

    # Text embedding options
    parser.add_argument(
        "--text-embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model used when embedding text queries",
    )
    parser.add_argument(
        "--text-embedding-device",
        default=None,
        help="Optional device string passed to SentenceTransformer (e.g. 'cpu', 'cuda')",
    )
    parser.add_argument(
        "--text-template",
        default="question: {text}\nanswer:",
        help="Template applied to text queries prior to embedding (must contain '{text}')",
    )
    parser.add_argument(
        "--text-normalize",
        action="store_true",
        help="L2-normalize the generated text embeddings",
    )

    # Image embedding options
    parser.add_argument(
        "--image-backend",
        choices=["color", "torch"],
        default="color",
        help="Embedding backend to use when a file path query is supplied",
    )
    parser.add_argument(
        "--image-color-bins",
        type=int,
        default=8,
        help="Number of histogram bins per channel when using the 'color' backend",
    )
    return parser.parse_args()


def embed_text(
    text: str,
    *,
    model_name: str,
    device: Optional[str],
    normalize: bool,
) -> Sequence[float]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("sentence-transformers is required to generate text embeddings") from exc

    model = SentenceTransformer(model_name, device=device)
    embedding = model.encode([text], normalize_embeddings=normalize)[0]
    return [float(v) for v in embedding]


def compute_color_histogram(image: Image.Image, bins: int) -> Sequence[float]:
    arr = np.asarray(image.convert("RGB"))
    hist_components = []
    for channel in range(3):
        channel_data = arr[:, :, channel]
        hist, _ = np.histogram(channel_data, bins=bins, range=(0, 256), density=True)
        hist_components.append(hist.astype(float))
    return np.concatenate(hist_components).tolist()


def compute_torch_embedding(image: Image.Image) -> Sequence[float]:
    try:
        import torch
        from torch import nn
        from torchvision.models import resnet18, ResNet18_Weights
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("torch and torchvision must be installed for the 'torch' image backend") from exc

    weights = ResNet18_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()
    model = resnet18(weights=weights)
    model.eval()

    feature_extractor = nn.Sequential(*(list(model.children())[:-1]))

    with torch.no_grad():
        tensor = preprocess(image).unsqueeze(0)
        features = feature_extractor(tensor).squeeze().numpy().astype(float)
    return [float(v) for v in features]


def embed_image(path: Path, backend: str, color_bins: int) -> Sequence[float]:
    with Image.open(path) as img:
        if backend == "color":
            return compute_color_histogram(img, bins=color_bins)
        return compute_torch_embedding(img)


def build_embedding(args: argparse.Namespace) -> Tuple[Sequence[float], str]:
    query_path = Path(args.query)
    if query_path.exists() and query_path.is_file():
        embedding = embed_image(
            query_path,
            backend=args.image_backend,
            color_bins=args.image_color_bins,
        )
        return embedding, "file"

    template = args.text_template
    if "{text}" not in template:
        raise ValueError("--text-template must contain the placeholder '{text}'")

    payload = template.format(text=args.query)
    embedding = embed_text(
        payload,
        model_name=args.text_embedding_model,
        device=args.text_embedding_device,
        normalize=args.text_normalize,
    )
    return embedding, "text"


def summarise_entry(entry: Any) -> Dict[str, Optional[str]]:
    summary: Dict[str, Optional[str]] = {
        "id": getattr(entry, "id", None),
        "question": None,
        "answer": None,
        "filename": None,
    }

    if hasattr(entry, "question"):
        summary["question"] = getattr(entry, "question")
    if hasattr(entry, "answer"):
        summary["answer"] = getattr(entry, "answer")
    if hasattr(entry, "filename"):
        summary["filename"] = getattr(entry, "filename")

    meta: Dict[str, Any] = {}
    if hasattr(entry, "_metadata"):
        meta = getattr(entry, "_metadata", {}) or {}
    elif hasattr(entry, "metadata"):
        meta = getattr(entry, "metadata", {}) or {}
    elif isinstance(entry, dict):
        meta = entry.get("meta", {}) or {}

    if not summary["question"]:
        summary["question"] = meta.get("question")
    if not summary["answer"]:
        summary["answer"] = meta.get("answer")
    if not summary["filename"]:
        summary["filename"] = meta.get("filename")
    if not summary["id"]:
        summary["id"] = meta.get("id")

    return summary


def main() -> None:
    args = parse_args()

    if args.neighbors <= 0:
        raise ValueError("--neighbors must be a positive integer")

    embedding, query_kind = build_embedding(args)

    # Resolve the HyperDataset version to query within ClearML
    dataset = HyperDatasetManagement.get(
        dataset_name=args.dataset_name,
        version_name=args.version_name,
        project_name=args.project,
    )

    # Execute ClearML's vector search against the stored embeddings
    results = dataset.vector_search(
        reference_vector=embedding,
        vector_field=args.vector_field,
        number_of_neighbors=args.neighbors,
        fast=args.fast,
        similarity_function=args.similarity,
    )

    if not results:
        print("No matching entries were returned")
        return

    descriptor = "file" if query_kind == "file" else "text"
    print(f"Top {len(results)} results for the {descriptor} query:")
    for idx, entry in enumerate(results, start=1):
        data = summarise_entry(entry)
        entry_id = data.get("id") or getattr(entry, "id", "<unknown>")
        print(f"[{idx}] entry_id={entry_id}")
        if data.get("question"):
            print(f"    Question: {data['question']}")
        if data.get("answer"):
            print(f"    Answer  : {data['answer']}")
        if data.get("filename"):
            print(f"    Filename: {data['filename']}")


if __name__ == "__main__":
    main()
