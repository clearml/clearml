import argparse
import os
from typing import Dict, Any, Iterable

import torch

# ClearML utilities for resolving datasets and creating dataview iterators
from clearml.hyperdatasets import (
    HyperDatasetManagement,
    DataView,
)


class HyperDatasetIterable(torch.utils.data.IterableDataset):
    """PyTorch IterableDataset wrapper around a DataView."""

    def __init__(self, query_kwargs: Dict[str, Any], projection: Iterable[str] = None):
        super().__init__()
        self._query_kwargs = dict(query_kwargs)
        self._projection = list(projection) if projection else None

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        # ClearML DataView streaming the requested dataset frames
        dv = DataView(auto_connect_with_task=False)
        dv.add_query(**self._query_kwargs)
        iterator = dv.get_iterator(
            projection=self._projection,
            num_workers=worker_info.num_workers if worker_info else None,
            worker_index=worker_info.id if worker_info else None,
        )
        for entry in iterator:
            if hasattr(entry, "to_api_object"):
                yield entry.to_api_object()
            elif isinstance(entry, dict):
                yield entry
            else:
                yield entry


def parse_args():
    parser = argparse.ArgumentParser(description="Stream HyperDataset frames with PyTorch DataLoader")
    name_group = parser.add_argument_group("Name-based selection")
    name_group.add_argument("--project", help="ClearML project name containing the dataset", default=None)
    name_group.add_argument("--dataset-name", help="Dataset collection name", default=None)
    name_group.add_argument("--version-name", help="Dataset version name", default=None)

    id_group = parser.add_argument_group("ID-based selection")
    id_group.add_argument("--dataset-id", help="Dataset collection id", default=None)
    id_group.add_argument("--version-id", help="Dataset version id", default=None)

    parser.add_argument("--projection", nargs="*", help="Optional projection fields", default=None)
    parser.add_argument("--batch-size", type=int, default=8, help="DataLoader batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader worker processes")
    parser.add_argument("--max-batches", type=int, default=5, help="Maximum number of batches to display")
    return parser.parse_args()


def resolve_dataset(args):
    if args.dataset_name and args.dataset_id:
        raise ValueError("Provide either --dataset-name or --dataset-id, not both")
    if args.version_name and args.version_id:
        raise ValueError("Provide either --version-name or --version-id, not both")

    if args.dataset_id:
        dataset = HyperDatasetManagement.get(dataset_id=args.dataset_id, version_id=args.version_id)
    else:
        if not args.dataset_name:
            raise ValueError("Either --dataset-id or --dataset-name must be supplied")
        # Resolve ClearML dataset/version identifiers
        dataset = HyperDatasetManagement.get(
            dataset_name=args.dataset_name,
            version_name=args.version_name,
            project_name=args.project,
        )
    return dataset


def main():
    args = parse_args()
    dataset = resolve_dataset(args)

    # ClearML query parameters targeting the chosen HyperDataset version
    query_kwargs = {
        "project_id": dataset.project_id or "*",
        "dataset_id": dataset.dataset_id,
        "version_id": dataset.version_id,
    }

    ds = HyperDatasetIterable(query_kwargs=query_kwargs, projection=args.projection)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers)

    print(
        f"Streaming frames from dataset_id={dataset.dataset_id} version_id={dataset.version_id} "
        f"with batch_size={args.batch_size} num_workers={args.num_workers}"
    )

    for batch_index, batch in enumerate(loader):
        if batch_index >= args.max_batches:
            break
        if isinstance(batch, dict):
            summary = {k: v if isinstance(v, (list, tuple)) else type(v).__name__ for k, v in batch.items()}
        else:
            summary = batch
        print(f"Batch {batch_index} -> {summary}")


if __name__ == "__main__":
    os.environ.setdefault("CLEARML_LOG_LEVEL", "INFO")
    main()
