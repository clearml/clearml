"""
Download a subset of COCO and convert it into a ClearML HyperDataset.

The script fetches the COCO 2017 annotations JSON and downloads image files on
an as-needed basis (defaulting to the validation split). Each COCO record is
converted into a `DataEntryImage` with per-object bounding boxes, polygon
segmentations, and optional keypoints metadata. Frame-level labels are stored on
the entry metadata so they can be queried later when browsing the HyperDataset.

Example:

    python examples/hyperdatasets/create_coco_hyperdataset.py \
        --dataset-name "coco-2017" \
        --version-name "val2017-top100" \
        --limit 100

"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from zipfile import ZipFile

import requests

from clearml.hyperdatasets import DataEntryImage, DataSubEntryImage, HyperDataset
from clearml.backend_api.session.session import Session


COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_SPLITS = {"train2017", "val2017"}


def parse_args() -> argparse.Namespace:
    description = (globals().get("__doc__") or "").strip()
    parser = argparse.ArgumentParser(
        description=description or None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--project", default="HyperDatasets Examples", help="ClearML project name")
    parser.add_argument("--dataset-name", required=True, help="HyperDataset collection name")
    parser.add_argument("--version-name", required=True, help="HyperDataset version name")
    parser.add_argument(
        "--description",
        default="COCO 2017 import",
        help="Version description stored alongside the HyperDataset version",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("examples/hyperdatasets/outputs/coco"),
        help="Directory where COCO assets will be downloaded",
    )
    parser.add_argument(
        "--split",
        choices=sorted(COCO_SPLITS),
        default="val2017",
        help="COCO split to ingest",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of images to download and register (None for all)",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download annotations and image files even if they already exist locally",
    )
    parser.add_argument(
        "--upload-uri",
        default=None,
        help="Optional ClearML storage URI for uploading downloaded image files",
    )
    parser.add_argument(
        "--force-upload",
        action="store_true",
        help="Force upload of image files even if hashes already exist on the server",
    )
    return parser.parse_args()


def download_file(url: str, destination: Path, *, force: bool = False) -> Path:
    """
    Stream a remote file to disk, skipping when it already exists.
    """
    if destination.exists() and not force:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Downloading %s -> %s", url, destination)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            shutil.copyfileobj(response.raw, handle)
    return destination


def ensure_annotations(workspace: Path, split: str, *, force_download: bool = False) -> Path:
    """
    Ensure the requested COCO annotations file is available locally.
    """
    annotations_dir = workspace / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    json_path = annotations_dir / f"instances_{split}.json"
    if json_path.exists() and not force_download:
        return json_path

    archive_path = annotations_dir / "annotations_trainval2017.zip"
    download_file(COCO_ANNOTATIONS_URL, archive_path, force=force_download)
    target_member = f"annotations/instances_{split}.json"

    logging.info("Extracting %s from %s", target_member, archive_path)
    with ZipFile(archive_path, "r") as archive:
        with archive.open(target_member) as source, json_path.open("wb") as destination:
            shutil.copyfileobj(source, destination)
    return json_path


def iter_selected_images(
    coco_payload: Dict[str, Sequence[dict]],
    split: str,
    limit: Optional[int],
) -> Iterator[dict]:
    """
    Iterate over COCO image records ordered by filename, respecting the limit.
    """
    images = coco_payload.get("images") or []
    sorted_images = sorted(images, key=lambda img: img.get("file_name", ""))
    for idx, image in enumerate(sorted_images):
        if limit is not None and idx >= limit:
            break
        yield image


def group_annotations(coco_payload: Dict[str, Sequence[dict]]) -> Dict[int, List[dict]]:
    """
    Build a mapping of image_id -> annotations list.
    """
    grouped: Dict[int, List[dict]] = defaultdict(list)
    for ann in coco_payload.get("annotations") or []:
        image_id = ann.get("image_id")
        if image_id is not None:
            grouped[int(image_id)].append(ann)
    return grouped


def download_image_if_needed(
    image_info: dict,
    image_dir: Path,
    *,
    force_download: bool = False,
) -> Optional[Path]:
    """
    Download a single COCO image to the workspace unless it already exists.
    """
    file_name = image_info.get("file_name")
    url = image_info.get("coco_url") or image_info.get("flickr_url")
    if not file_name or not url:
        logging.warning("Skipping image %s: missing file name or URL", image_info.get("id"))
        return None

    target = image_dir / file_name
    if target.exists() and not force_download:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        download_file(url, target, force=force_download)
        return target
    except requests.RequestException as exc:
        logging.warning("Failed downloading %s (%s): %s", file_name, url, exc)
        return None


def build_data_entries(
    images: Iterable[dict],
    annotations_by_image: Dict[int, Sequence[dict]],
    categories: Dict[int, str],
    image_root: Path,
    split: str,
) -> List[DataEntryImage]:
    """
    Convert COCO records into `DataEntryImage` objects.
    """
    entries: List[DataEntryImage] = []
    for image_info in images:
        image_id = int(image_info["id"])
        image_path = image_root / image_info["file_name"]
        if not image_path.exists():
            logging.warning("Skipping image %s (%s); file not found", image_id, image_path)
            continue

        entry_metadata = {
            "coco_image_id": image_id,
            "file_name": image_info.get("file_name"),
            "width": image_info.get("width"),
            "height": image_info.get("height"),
            "split": split,
        }
        entry = DataEntryImage(metadata=entry_metadata)
        sub_entry = DataSubEntryImage(
            name="image",
            source=str(image_path.resolve()),
            width=image_info.get("width"),
            height=image_info.get("height"),
            metadata={"split": split},
        )
        entry.add_sub_entries([sub_entry])

        frame_labels: set[str] = set()
        for ann in annotations_by_image.get(image_id, []):
            category_id = ann.get("category_id")
            category_name = categories.get(int(category_id)) if category_id is not None else None
            if category_name:
                frame_labels.add(category_name)

            base_meta = {
                "annotation_id": ann.get("id"),
                "category_id": category_id,
                "category_name": category_name,
                "area": ann.get("area"),
                "iscrowd": bool(ann.get("iscrowd", 0)),
            }

            bbox = ann.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                bbox_meta = dict(base_meta)
                bbox_meta["type"] = "bbox"
                sub_entry.add_annotation(
                    box2d_xywh=bbox[:4],
                    labels=[category_name] if category_name else None,
                    id=f"{ann.get('id')}_bbox",
                    metadata=bbox_meta,
                )

            segmentation = ann.get("segmentation")
            if isinstance(segmentation, list):
                for seg_idx, seg in enumerate(segmentation):
                    if not seg:
                        continue
                    seg_meta = dict(base_meta)
                    seg_meta["type"] = "segmentation"
                    seg_meta["segment_index"] = seg_idx
                    sub_entry.add_annotation(
                        poly2d_xy=seg,
                        labels=[category_name] if category_name else None,
                        id=f"{ann.get('id')}_seg{seg_idx}",
                        metadata=seg_meta,
                    )
            elif segmentation:
                seg_meta = dict(base_meta)
                seg_meta["type"] = "segmentation_rle"
                sub_entry.add_annotation(
                    labels=[category_name] if category_name else None,
                    id=f"{ann.get('id')}_seg_rle",
                    metadata=seg_meta,
                )

            keypoints = ann.get("keypoints")
            if isinstance(keypoints, list) and keypoints:
                points: List[Tuple[float, float]] = []
                visibility: List[int] = []
                for idx in range(0, len(keypoints), 3):
                    x, y, v = keypoints[idx : idx + 3]
                    if v > 0:
                        points.append((float(x), float(y)))
                        visibility.append(int(v))
                if points:
                    kp_meta = dict(base_meta)
                    kp_meta["type"] = "keypoints"
                    kp_meta["visibility"] = visibility
                    sub_entry.add_annotation(
                        points2d_xy=points,
                        labels=[category_name] if category_name else None,
                        id=f"{ann.get('id')}_keypoints",
                        metadata=kp_meta,
                    )

        if frame_labels:
            entry_metadata["frame_labels"] = sorted(frame_labels)
            entry.add_annotation(
                id=f"{image_id}_frame_labels",
                labels=sorted(frame_labels),
                confidence=None,
                metadata={"source": "coco", "image_id": image_id},
            )

        entries.append(entry)
    return entries


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    workspace = args.workspace.expanduser().resolve()
    annotations_path = ensure_annotations(workspace, args.split, force_download=args.force_download)
    logging.info("Loading COCO annotations from %s", annotations_path)

    with annotations_path.open("r", encoding="utf-8") as handle:
        coco_payload = json.load(handle)

    categories = {int(cat["id"]): cat.get("name") for cat in coco_payload.get("categories", [])}
    annotations_by_image = group_annotations(coco_payload)
    selected_images = list(iter_selected_images(coco_payload, args.split, args.limit))

    if not selected_images:
        raise RuntimeError("No COCO images matched the requested limit")

    image_root = workspace / "images" / args.split
    image_root.mkdir(parents=True, exist_ok=True)

    for image_info in selected_images:
        download_image_if_needed(
            image_info,
            image_root,
            force_download=args.force_download,
        )

    entries = build_data_entries(
        selected_images,
        annotations_by_image,
        categories,
        image_root,
        args.split,
    )

    if not entries:
        raise RuntimeError("No entries were created; ensure images were downloaded successfully")

    dataset = HyperDataset(
        project_name=args.project,
        dataset_name=args.dataset_name,
        version_name=args.version_name,
        description=args.description,
        field_mappings={
            "meta.frame_labels": {"type": "keyword"},
        },
    )

    logging.info("Registering %d entries with HyperDataset %s/%s", len(entries), args.dataset_name, args.version_name)
    errors = dataset.add_data_entries(
        entries,
        upload_local_files_destination=args.upload_uri or Session.get_files_server_host(),
        force_upload=args.force_upload,
    )
    print(errors)

    if errors.get("register"):
        raise RuntimeError(f"Failed registering entries: {errors['register']}")

    logging.info(
        "Created HyperDataset version: project=%s dataset=%s version=%s frames=%d",
        dataset.project_id,
        dataset.dataset_id,
        dataset.version_id,
        len(entries),
    )


if __name__ == "__main__":
    main()
