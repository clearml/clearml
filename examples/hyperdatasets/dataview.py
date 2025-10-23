import argparse

from clearml import Task
from tqdm import tqdm
# ClearML HyperDataset helpers for dataset resolution and dataview streaming
from clearml.hyperdatasets import (
    HyperDatasetManagement,
    DataView,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Dataset (collection) name")
    ap.add_argument("--version", required=True, help="Dataset version name")
    ap.add_argument("--limit", type=int, default=None, help="Max frames to print")
    args = ap.parse_args()

    # Track the run inside ClearML
    Task.init(project_name="HyperDatasets Examples", task_name="dataview_iterate")

    # Resolve the dataset/version identifiers from ClearML
    hd = HyperDatasetManagement.get(dataset_name=args.dataset, version_name=args.version)
    project_id = hd.project_id      # ClearML Project id
    dataset_id = hd.dataset_id      # Dataset (collection) id
    version_id = hd.version_id      # Version id

    # Build and register a ClearML DataView query
    dv = DataView(iteration_order="sequential", iteration_infinite=False)
    dv.add_query(project_id=project_id, dataset_id=dataset_id, version_id=version_id)
    print(f"Created dataview: {dv.id}")

    try:
        it = dv.get_iterator()
        total = len(dv)
        count = 0
        for frame in tqdm(it, total=total, desc="Frames"):
            print(frame)
            # Print a small summary of the returned frame structure
            try:
                fid = getattr(frame, "id", None) or frame.get("id")
            except Exception:
                fid = None
            print(f"Frame: id={fid}")
            count += 1
            if args.limit and count >= args.limit:
                break
        print(f"Iterated {count} frame(s)")
    except Exception as e:
        # Friendly message if the backend endpoint is missing in this build
        print("Dataview iteration is not available in this build: ", e)


if __name__ == "__main__":
    main()
