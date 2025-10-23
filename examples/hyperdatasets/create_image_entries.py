"""Create a HyperDataset populated with local image files.

The script demonstrates how to use `DataEntryImage` along with
optional vector embeddings computed from each image. Ten sample images are
shipped under `examples/hyperdatasets/sample_images`, but you can point the
script at any directory of JPEG/PNG assets.

Example:

    python examples/hyperdatasets/image_entries_create.py \
        --project "HyperDatasets Examples" \
        --dataset-name "image-demo" \
        --version-name "version" \
        --image-dir examples/hyperdatasets/sample_images \
        --limit 10 \
        --embed
"""

import argparse
import itertools
import textwrap
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

# ClearML HyperDataset image primitives
from clearml.hyperdatasets import HyperDataset, DataEntryImage, DataSubEntryImage


def parse_args() -> argparse.Namespace:
    description = textwrap.dedent(__doc__ or "").strip()
    parser = argparse.ArgumentParser(
        description=description or None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--project", default="HyperDatasets Examples", help="ClearML project name")
    parser.add_argument("--dataset-name", required=True, help="HyperDataset collection name")
    parser.add_argument("--version-name", required=True, help="HyperDataset version name")
    parser.add_argument("--description", default="Image demo HyperDataset", help="Version description")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("examples/hyperdatasets/sample_images"),
        help="Directory containing image files (JPEG/PNG)",
    )
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of images to ingest")
    parser.add_argument("--embed", action="store_true", help="Compute and store embeddings for each image")
    parser.add_argument(
        "--embedding-backend",
        choices=["color", "torch"],
        default="color",
        help="Embedding backend to use when --embed is passed",
    )
    parser.add_argument(
        "--vector-field",
        help="Metadata field used to store the embedding",
    )
    parser.add_argument(
        "--upload-uri",
        default=None,
        help="Optional ClearML storage URI for uploading the local image files",
    )
    parser.add_argument(
        "--force-upload",
        action="store_true",
        help="Force upload of image files even if hashes already exist on the server",
    )
    return parser.parse_args()


def iter_images(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif"} and path.is_file():
            yield path


def compute_color_histogram(image: Image.Image, bins: int = 8) -> List[float]:
    arr = np.asarray(image.convert("RGB"))
    hist_components = []
    for channel in range(3):
        channel_data = arr[:, :, channel]
        hist, _ = np.histogram(channel_data, bins=bins, range=(0, 256), density=True)
        hist_components.append(hist.astype(float))
    return [float(v) for v in np.concatenate(hist_components)]


def compute_torch_embedding(image: Image.Image) -> List[float]:
    try:
        import torch
        from torch import nn
        from torchvision.models import resnet18, ResNet18_Weights
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("torch and torchvision must be installed for --embedding-backend torch") from exc

    weights = ResNet18_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()
    model = resnet18(weights=weights)
    model.eval()

    feature_extractor = nn.Sequential(*(list(model.children())[:-1]))

    with torch.no_grad():
        tensor = preprocess(image).unsqueeze(0)
        features = feature_extractor(tensor).squeeze().numpy().astype(float)
    return [float(v) for v in features]


def build_entries(
    image_paths: Iterable[Path],
    *,
    embed: bool,
    backend: str,
    vector_field: Optional[str],
) -> Tuple[List[DataEntryImage], Optional[int]]:
    entries: List[DataEntryImage] = []
    vector_dims: Optional[int] = None
    for idx, image_path in enumerate(image_paths, 1):
        with Image.open(image_path) as img:
            width, height = img.size
            # ClearML image entry holding metadata for this frame
            entry = DataEntryImage(metadata={"index": idx, "filename": image_path.name})
            entry.add_sub_entries(
                # ClearML image sub-entry pointing to the local file on disk
                [DataSubEntryImage(name="image", source=str(image_path), width=width, height=height)]
            )

            if embed:
                if not vector_field:
                    raise ValueError("vector_field must be provided when --embed is used")
                if backend == "color":
                    vector = compute_color_histogram(img)
                else:
                    vector = compute_torch_embedding(img)
                entry.set_vector(vector, metadata_field=vector_field)
                if vector_dims is None:
                    vector_dims = len(vector)
                elif vector_dims != len(vector):
                    raise ValueError("All embedding vectors must share the same dimensionality")

        entries.append(entry)
    return entries, vector_dims


def main() -> None:
    args = parse_args()

    if not args.image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")

    image_paths = list(itertools.islice(iter_images(sorted(args.image_dir.iterdir())), args.limit))
    if not image_paths:
        raise ValueError(f"No image files found in {args.image_dir}")

    vector_field = args.vector_field
    if args.embed and not vector_field:
        raise ValueError("--vector-field must be provided when --embed is used")

    entries, vector_dims = build_entries(
        image_paths,
        embed=args.embed,
        backend=args.embedding_backend,
        vector_field=vector_field,
    )

    field_mappings = None
    if vector_dims:
        field_path = vector_field if vector_field.startswith("meta.") else f"meta.{vector_field}"
        # ClearML field mapping describing the embedding metadata schema
        field_mappings = {
            field_path: {
                "type": "dense_vector",
                "element_type": "float",
                "dims": vector_dims,
            }
        }

    # ClearML HyperDataset version handle (creates version if missing)
    dataset = HyperDataset(
        project_name=args.project,
        dataset_name=args.dataset_name,
        version_name=args.version_name,
        description=args.description,
        field_mappings=field_mappings,
    )

    # Upload entries so ClearML manages storage, hashing, and indexing
    errors = dataset.add_data_entries(
        entries,
        upload_local_files_destination=args.upload_uri,
        force_upload=args.force_upload,
    )

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
