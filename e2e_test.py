"""
End-to-end pytest for external dataset with a parent → child → grandchild lineage.

Each generation uses Dataset.sync_folder to:
  - remove 1 file
  - add    1 file
  - modify 1 file

After each finalize, Dataset.get_local_copy is called and the resulting folder is verified to
contain exactly the expected files with the expected contents.

Requires a running ClearML server (reads credentials from ~/clearml.conf or env vars).
Run with:
    pytest e2e_test.py -s
"""

import os
import pytest
from pathlib import Path

from clearml import Dataset

DATASET_PROJECT = "e2e-tests/immunai-clearml/external-dataset"
OUTPUT_BUCKET = "gs://immunai-test-data/immunai-clearml/external-dataset/"
# OUTPUT_BUCKET = "s3://amzn-s3-clml-production/immunai-clearml/external-dataset/"

os.environ["CLEARML_CACHE_DIR"] = "clearml_cache"
# os.environ.setdefault("AWS_CA_BUNDLE", os.path.expanduser("~/.certs/ca_bundle.pem"))


def is_external_dataset(dataset: Dataset) -> bool:
    return getattr(dataset, "_Dataset__external_files_tag") in dataset.tags and not dataset.file_entries


def _write(folder: Path, name: str, content: str) -> Path:
    p = folder / name
    p.write_text(content)
    return p


def _local_files(local_path: str) -> dict[str, str]:
    """Return {relative_name: content} for every file under local_path."""
    root = Path(local_path)
    return {p.relative_to(root).as_posix(): p.read_text() for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def created_datasets():
    datasets = []
    yield datasets
    for ds in datasets:
        try:
            ds.delete(dataset_id=ds.id, entire_dataset=True, delete_files=True)
        except Exception as exc:
            print(f"Warning: could not delete dataset {ds.id}: {exc}")


def test_external_dataset_lineage(tmp_path, created_datasets):
    dataset_name = "e2e-dataset-ext"

    # ── PARENT ────────────────────────────────────────────────────────────────
    # Folder contains: a.txt, b.txt, c.txt
    parent_folder = tmp_path / "parent"
    parent_folder.mkdir()
    _write(parent_folder, "a.txt", "aaa")
    _write(parent_folder, "b.txt", "bbb")
    _write(parent_folder, "c.txt", "ccc")

    parent_ds = Dataset.create(dataset_name=dataset_name, dataset_project=DATASET_PROJECT)
    created_datasets.append(parent_ds)

    removed, added, modified = parent_ds.sync_folder(parent_folder, verbose=True)
    assert added == 3, f"Expected 3 files added to parent, got {added}"
    assert removed == 0
    assert modified == 0

    parent_ds.upload(output_url=OUTPUT_BUCKET + "parent", upload_as_external_links=True)
    assert parent_ds.finalize(), "should succeed finalize parent"

    parent_ds_ro = Dataset.get(dataset_id=parent_ds.id)
    assert is_external_dataset(parent_ds_ro)
    parent_local = parent_ds_ro.get_local_copy(use_soft_links=False)
    parent_files = _local_files(parent_local)
    assert set(parent_files.keys()) == {"a.txt", "b.txt", "c.txt"}
    assert parent_files["a.txt"] == "aaa"
    assert parent_files["b.txt"] == "bbb"
    assert parent_files["c.txt"] == "ccc"

    parent_subset_local = parent_ds_ro.get_local_copy(use_soft_links=False, files=["a.txt", "c.txt"])
    parent_subset_files = _local_files(parent_subset_local)
    assert set(parent_subset_files.keys()) == {"a.txt", "c.txt"}, (
        f"Unexpected files in parent subset: {set(parent_subset_files.keys())}"
    )
    assert parent_subset_files["a.txt"] == "aaa"
    assert parent_subset_files["c.txt"] == "ccc"
    assert "b.txt" not in parent_subset_files

    # ── CHILD ─────────────────────────────────────────────────────────────────
    # Changes vs parent: remove a.txt, add d.txt, modify b.txt
    child_folder = tmp_path / "child"
    child_folder.mkdir()
    # a.txt intentionally absent → will be removed from dataset
    _write(child_folder, "b.txt", "BBB")  # modified
    _write(child_folder, "c.txt", "ccc")  # unchanged
    _write(child_folder, "d.txt", "ddd")  # new

    child_ds = Dataset.create(
        dataset_name=dataset_name, dataset_project=DATASET_PROJECT, parent_datasets=[parent_ds.id]
    )
    created_datasets.append(child_ds)

    removed, added, modified = child_ds.sync_folder(child_folder, verbose=True)
    assert removed == 1, f"Expected 1 file removed from child, got {removed}"
    assert added == 1, f"Expected 1 file added to child, got {added}"
    assert modified == 1, f"Expected 1 file modified in child, got {modified}"
    child_ds.upload(output_url=OUTPUT_BUCKET + "child", upload_as_external_links=True)
    assert child_ds.finalize(), "should succeed finalize child"

    child_ds_ro = Dataset.get(dataset_id=child_ds.id)
    assert is_external_dataset(child_ds_ro)
    child_local = child_ds_ro.get_local_copy(use_soft_links=False)
    child_files = _local_files(child_local)
    assert set(child_files.keys()) == {"b.txt", "c.txt", "d.txt"}, (
        f"Unexpected files in child local copy: {set(child_files.keys())}"
    )
    assert child_files["b.txt"] == "BBB"
    assert child_files["c.txt"] == "ccc"
    assert child_files["d.txt"] == "ddd"
    assert "a.txt" not in child_files

    child_subset_local = child_ds_ro.get_local_copy(use_soft_links=False, files=["b.txt", "c.txt"])
    child_subset_files = _local_files(child_subset_local)
    assert set(child_subset_files.keys()) == {"b.txt", "c.txt"}, (
        f"Unexpected files in child subset: {set(child_subset_files.keys())}"
    )
    assert child_subset_files["b.txt"] == "BBB"
    assert child_subset_files["c.txt"] == "ccc"
    assert "d.txt" not in child_subset_files
    assert "a.txt" not in child_subset_files

    # ── GRANDCHILD ────────────────────────────────────────────────────────────
    # Changes vs child: remove c.txt, add e.txt, modify d.txt
    grandchild_folder = tmp_path / "grandchild"
    grandchild_folder.mkdir()
    _write(grandchild_folder, "b.txt", "BBB")  # unchanged
    # c.txt intentionally absent → will be removed from dataset
    _write(grandchild_folder, "d.txt", "DDD")  # modified
    _write(grandchild_folder, "e.txt", "eee")  # new

    grandchild_ds = Dataset.create(
        dataset_name=dataset_name, dataset_project=DATASET_PROJECT, parent_datasets=[child_ds.id]
    )
    created_datasets.append(grandchild_ds)

    removed, added, modified = grandchild_ds.sync_folder(grandchild_folder, verbose=True)
    assert removed == 1, f"Expected 1 file removed from grandchild, got {removed}"
    assert added == 1, f"Expected 1 file added to grandchild, got {added}"
    assert modified == 1, f"Expected 1 file modified in grandchild, got {modified}"

    grandchild_ds.upload(output_url=OUTPUT_BUCKET + "grandchild", upload_as_external_links=True)
    assert grandchild_ds.finalize(), "should succeed finalize grandchild"

    grandchild_ds_ro = Dataset.get(dataset_id=grandchild_ds.id)
    assert is_external_dataset(grandchild_ds_ro)
    grandchild_local = grandchild_ds_ro.get_local_copy(use_soft_links=False)
    grandchild_files = _local_files(grandchild_local)
    assert set(grandchild_files.keys()) == {"b.txt", "d.txt", "e.txt"}, (
        f"Unexpected files in grandchild local copy: {set(grandchild_files.keys())}"
    )
    assert grandchild_files["b.txt"] == "BBB"
    assert grandchild_files["d.txt"] == "DDD"
    assert grandchild_files["e.txt"] == "eee"
    assert "a.txt" not in grandchild_files
    assert "c.txt" not in grandchild_files

    grandchild_subset_local = grandchild_ds_ro.get_local_copy(use_soft_links=False, files=["d.txt", "e.txt"])
    grandchild_subset_files = _local_files(grandchild_subset_local)
    assert set(grandchild_subset_files.keys()) == {"d.txt", "e.txt"}, (
        f"Unexpected files in grandchild subset: {set(grandchild_subset_files.keys())}"
    )
    assert grandchild_subset_files["d.txt"] == "DDD"
    assert grandchild_subset_files["e.txt"] == "eee"
    assert "b.txt" not in grandchild_subset_files
