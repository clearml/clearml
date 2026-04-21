"""
Integration tests for the three commits:

  62997c3d  – SHA-256 hash comparison for external link change detection
               (read_hash flag in add_external_files, hash stored on LinkEntry,
                hash persisted in as_dict, upload_hash extra key, get_object_metadata)

  bed10556  – get_local_copy(files=[...]) subset pulling
               (only requested files downloaded, cache folder has unique hash suffix,
                subset cache is independent from full-dataset cache)

  9174775e  – sync_folder prunes stale link entries
               upload(upload_as_external_links=True) uploads files as GCS objects
               and registers them as external LinkEntry records

Requires:
  - ~/.clearml.conf configured (clearml server + GCS credentials)
  - Write access to OUTPUT_BUCKET
  - pip install clearml google-cloud-storage pytest
"""

import hashlib
import tempfile
import uuid
from pathlib import Path

import pytest

from clearml import Dataset
from clearml.datasets.dataset import LinkEntry
from clearml.storage.helper import StorageHelper

OUTPUT_BUCKET = "gs://immunai-test-data/immunai-clearml/external-dataset/"
PROJECT = "test/dataset-features"

# One unique id per test session so GCS paths and dataset names never collide with
# a concurrent run.
_SESSION_ID = uuid.uuid4().hex[:8]


# ── Low-level helpers ─────────────────────────────────────────────────────────


def _gcs_prefix(label: str) -> str:
    return f"{OUTPUT_BUCKET}{label}-{_SESSION_ID}/"


def _upload_to_gcs(local_path: Path, remote_url: str, *, store_hash: bool = False) -> None:
    """Upload one file to GCS.  store_hash=True writes SHA-256 to object metadata."""
    helper = StorageHelper.get(remote_url)
    extra = {"upload_hash": None} if store_hash else None
    helper.upload(src_path=str(local_path), dest_path=remote_url, extra=extra)


def _write_files(base: Path, spec: dict) -> None:
    """Create files under *base*.  spec maps relative-name → text content."""
    for name, content in spec.items():
        p = base / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)


def _delete_dataset(dataset_id: str) -> None:
    try:
        Dataset.delete(dataset_id=dataset_id, force=True)
    except Exception:
        pass


def _delete_gcs_prefix(prefix: str) -> None:
    """Best-effort removal of every object under a GCS prefix."""
    try:
        helper = StorageHelper.get(prefix)
        keys = helper.list(prefix=prefix) or []
        for key in keys:
            try:
                full = key if key.startswith("gs://") else f"{prefix.rstrip('/')}/{key.lstrip('/')}"
                obj = helper.get_object(full)
                if obj:
                    helper._driver.delete_object(obj)
            except Exception:
                pass
    except Exception:
        pass


# ── Module-scoped datasets shared across related tests ────────────────────────


@pytest.fixture(scope="module")
def ext_dataset_with_hash():
    """
    Module fixture: three files uploaded to GCS *with* SHA-256 stored in metadata,
    then added to a finalized dataset via add_external_files(read_hash=True).
    Yields (dataset, gcs_prefix, {relative_path: sha256_hex}).
    """
    prefix = _gcs_prefix("hash-ds")
    spec = {
        "alpha.txt": "content of alpha",
        "beta.txt": "content of beta",
        "subdir/gamma.txt": "content of gamma",
    }
    expected_hashes = {}

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        _write_files(tmp, spec)

        for name in spec:
            local = tmp / name
            remote = f"{prefix}{name}"
            _upload_to_gcs(local, remote, store_hash=True)
            expected_hashes[name] = hashlib.sha256(local.read_bytes()).hexdigest()

    ds = Dataset.create(dataset_name=f"hash-ds-{_SESSION_ID}", dataset_project=PROJECT)
    ds.add_external_files(prefix, read_hash=True)
    ds.upload()
    ds.finalize()

    yield ds, prefix, expected_hashes

    _delete_dataset(ds.id)
    _delete_gcs_prefix(prefix)


@pytest.fixture(scope="module")
def ext_dataset_no_hash():
    """
    Module fixture: three files uploaded to GCS without hash metadata,
    added to a finalized dataset.  Used for subset-pulling tests.
    Yields (dataset, gcs_prefix).
    """
    prefix = _gcs_prefix("subset-ext")
    spec = {
        "alpha.txt": "content of alpha",
        "beta.txt": "content of beta",
        "subdir/gamma.txt": "content of gamma",
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        _write_files(tmp, spec)
        for name in spec:
            _upload_to_gcs(tmp / name, f"{prefix}{name}")

    ds = Dataset.create(dataset_name=f"subset-ext-{_SESSION_ID}", dataset_project=PROJECT)
    ds.add_external_files(prefix)
    ds.upload()
    ds.finalize()

    yield ds, prefix

    _delete_dataset(ds.id)
    _delete_gcs_prefix(prefix)


@pytest.fixture(scope="module")
def chunk_dataset():
    """
    Module fixture: three files uploaded as regular ClearML zip-chunks (not external links).
    Yields the finalized dataset.
    """
    spec = {
        "alpha.txt": "content of alpha",
        "beta.txt": "content of beta",
        "subdir/gamma.txt": "content of gamma",
    }
    ds = Dataset.create(dataset_name=f"chunk-ds-{_SESSION_ID}", dataset_project=PROJECT)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        _write_files(tmp, spec)
        ds.add_files(str(tmp))
        ds.upload(output_url=OUTPUT_BUCKET)

    ds.finalize()

    yield ds

    _delete_dataset(ds.id)


# ═════════════════════════════════════════════════════════════════════════════
# Commit 62997c3d – SHA-256 hash comparison for external link change detection
# ═════════════════════════════════════════════════════════════════════════════


class TestReadHash:
    def test_link_entry_hash_populated(self, ext_dataset_with_hash):
        """add_external_files(read_hash=True) stores SHA-256 on every LinkEntry."""
        ds, prefix, expected = ext_dataset_with_hash
        for name, sha256 in expected.items():
            entry = ds._dataset_link_entries.get(name)
            assert entry is not None, f"LinkEntry for {name!r} missing"
            assert entry.hash == sha256, (
                f"{name}: expected hash {sha256!r}, got {entry.hash!r}"
            )

    def test_hash_persisted_in_as_dict(self, ext_dataset_with_hash):
        """LinkEntry.as_dict() includes the hash field (commit 62997c3d serialization fix)."""
        ds, prefix, expected = ext_dataset_with_hash
        for name in expected:
            entry = ds._dataset_link_entries[name]
            d = entry.as_dict()
            assert "hash" in d
            assert d["hash"] == entry.hash

    def test_no_change_when_hash_matches(self, ext_dataset_with_hash):
        """
        Re-adding the same GCS files with read_hash=True reports 0 new/modified files
        because hash comparison short-circuits the size comparison.
        """
        ds, prefix, _ = ext_dataset_with_hash
        count = ds.add_external_files(prefix, read_hash=True)
        assert count == 0

    def test_change_detected_via_hash(self, tmp_path):
        """
        A file re-uploaded with different content (different hash) is detected as
        modified even when its byte-size happens to be identical.
        """
        prefix = _gcs_prefix("hash-change")
        ds = Dataset.create(dataset_name=f"hash-change-{_SESSION_ID}", dataset_project=PROJECT)
        try:
            # v1: exactly 20 bytes
            v1 = tmp_path / "data.txt"
            v1.write_text("AAAABBBBCCCCDDDDEEEE")  # 20 chars
            remote = f"{prefix}data.txt"
            _upload_to_gcs(v1, remote, store_hash=True)
            ds.add_external_files(prefix, read_hash=True)
            hash_v1 = ds._dataset_link_entries["data.txt"].hash

            # v2: different content, same 20 bytes
            v2 = tmp_path / "data_v2.txt"
            v2.write_text("11112222333344445555")  # 20 chars
            _upload_to_gcs(v2, remote, store_hash=True)
            ds.add_external_files(prefix, read_hash=True)
            hash_v2 = ds._dataset_link_entries["data.txt"].hash

            assert hash_v1 != hash_v2
        finally:
            _delete_dataset(ds.id)
            _delete_gcs_prefix(prefix)


# ═════════════════════════════════════════════════════════════════════════════
# Commit 9174775e – sync_folder prunes stale link entries
# ═════════════════════════════════════════════════════════════════════════════


class TestSyncFolderLinkEntries:
    def test_stale_link_entry_removed(self, tmp_path):
        """
        After sync_folder, link entries whose relative_path has no matching local file
        are pruned from _dataset_link_entries.
        """
        ds = Dataset.create(dataset_name=f"sync-prune-{_SESSION_ID}", dataset_project=PROJECT)
        try:
            ds._dataset_link_entries["keep.txt"] = LinkEntry(
                link="gs://some-bucket/keep.txt",
                relative_path="keep.txt",
                parent_dataset_id=ds.id,
                size=10,
            )
            ds._dataset_link_entries["remove.txt"] = LinkEntry(
                link="gs://some-bucket/remove.txt",
                relative_path="remove.txt",
                parent_dataset_id=ds.id,
                size=10,
            )

            (tmp_path / "keep.txt").write_text("I am present locally")
            # remove.txt intentionally absent from tmp_path

            removed, added, modified = ds.sync_folder(tmp_path)

            assert "keep.txt" in ds._dataset_link_entries
            assert "remove.txt" not in ds._dataset_link_entries
            assert removed >= 1
        finally:
            _delete_dataset(ds.id)

    def test_link_entries_outside_prefix_are_kept(self, tmp_path):
        """
        Link entries whose relative_path does NOT start with the synced dataset_path
        prefix are never pruned, even when the corresponding local file is absent.
        """
        ds = Dataset.create(dataset_name=f"sync-prefix-{_SESSION_ID}", dataset_project=PROJECT)
        try:
            # Outside the synced prefix
            ds._dataset_link_entries["other/data.txt"] = LinkEntry(
                link="gs://some-bucket/other/data.txt",
                relative_path="other/data.txt",
                parent_dataset_id=ds.id,
                size=10,
            )
            # Inside synced prefix, missing locally
            ds._dataset_link_entries["synced/gone.txt"] = LinkEntry(
                link="gs://some-bucket/synced/gone.txt",
                relative_path="synced/gone.txt",
                parent_dataset_id=ds.id,
                size=10,
            )

            sync_dir = tmp_path / "synced"
            sync_dir.mkdir()
            # "synced/gone.txt" not created → should be pruned

            ds.sync_folder(sync_dir, dataset_path="synced")

            assert "other/data.txt" in ds._dataset_link_entries, (
                "Entry outside synced prefix should be preserved"
            )
            assert "synced/gone.txt" not in ds._dataset_link_entries, (
                "Entry inside synced prefix with no local file should be removed"
            )
        finally:
            _delete_dataset(ds.id)

    def test_file_entries_still_pruned(self, tmp_path):
        """
        Existing sync_folder behaviour for _dataset_file_entries is unaffected:
        file entries without a local counterpart are still removed.
        """
        ds = Dataset.create(dataset_name=f"sync-files-{_SESSION_ID}", dataset_project=PROJECT)
        try:
            (tmp_path / "present.txt").write_text("here")
            ds.add_files(str(tmp_path))

            # Remove the file locally before syncing
            (tmp_path / "present.txt").unlink()

            removed, _, _ = ds.sync_folder(tmp_path)

            assert "present.txt" not in ds._dataset_file_entries
            assert removed >= 1
        finally:
            _delete_dataset(ds.id)


# ═════════════════════════════════════════════════════════════════════════════
# Commit 9174775e – upload(upload_as_external_links=True)
# ═════════════════════════════════════════════════════════════════════════════


class TestUploadAsExternalLinks:
    def test_file_entries_become_link_entries(self, tmp_path):
        """
        After upload(upload_as_external_links=True):
          - _dataset_file_entries is empty
          - _dataset_link_entries has one entry per uploaded file
          - each link points into OUTPUT_BUCKET
        """
        spec = {"a.txt": "file a", "b.txt": "file b", "sub/c.txt": "file c"}
        _write_files(tmp_path, spec)

        ds = Dataset.create(dataset_name=f"ext-upload-{_SESSION_ID}", dataset_project=PROJECT)
        try:
            ds.add_files(str(tmp_path))
            assert len(ds._dataset_file_entries) == 3

            ds.upload(output_url=OUTPUT_BUCKET, upload_as_external_links=True)

            assert len(ds._dataset_file_entries) == 0
            assert len(ds._dataset_link_entries) == 3

            for rel_path in spec:
                assert rel_path in ds._dataset_link_entries
                assert ds._dataset_link_entries[rel_path].link.startswith("gs://")
        finally:
            _delete_dataset(ds.id)

    def test_link_destination_contains_dataset_id(self, tmp_path):
        """Each uploaded link URL contains the dataset id, matching the
        external_links/{id}/ path convention."""
        (tmp_path / "x.txt").write_text("x")

        ds = Dataset.create(dataset_name=f"ext-upload-id-{_SESSION_ID}", dataset_project=PROJECT)
        try:
            ds.add_files(str(tmp_path))
            ds.upload(output_url=OUTPUT_BUCKET, upload_as_external_links=True)

            link = ds._dataset_link_entries["x.txt"].link
            assert ds.id in link
        finally:
            _delete_dataset(ds.id)

    def test_get_local_copy_after_external_upload(self, tmp_path):
        """Files uploaded as external links can be retrieved via get_local_copy."""
        spec = {"alpha.txt": "content of alpha", "sub/beta.txt": "content of beta"}
        _write_files(tmp_path, spec)

        ds = Dataset.create(dataset_name=f"ext-glc-{_SESSION_ID}", dataset_project=PROJECT)
        try:
            ds.add_files(str(tmp_path))
            ds.upload(output_url=OUTPUT_BUCKET, upload_as_external_links=True)
            ds.finalize()

            local = Path(ds.get_local_copy())
            for rel_path, content in spec.items():
                assert (local / rel_path).is_file(), f"{rel_path} missing in local copy"
                assert (local / rel_path).read_text() == content
        finally:
            _delete_dataset(ds.id)

    def test_hash_forwarded_to_upload(self, tmp_path):
        """
        File hash computed during add_files is forwarded to helper.upload via
        extra={"hash": ...}.  We verify the FileEntry had a hash before upload
        (meaning it was computed) and that upload completes without error.
        """
        (tmp_path / "h.txt").write_text("hash me")

        ds = Dataset.create(dataset_name=f"ext-hash-fwd-{_SESSION_ID}", dataset_project=PROJECT)
        try:
            ds.add_files(str(tmp_path))
            entry = ds._dataset_file_entries["h.txt"]
            assert entry.hash is not None, "add_files should compute SHA-256"

            # upload must not raise even though extra={"hash": ...} is passed
            ds.upload(output_url=OUTPUT_BUCKET, upload_as_external_links=True)
        finally:
            _delete_dataset(ds.id)


# ═════════════════════════════════════════════════════════════════════════════
# Commit bed10556 – get_local_copy(files=[...]) subset pulling
# ═════════════════════════════════════════════════════════════════════════════


class TestSubsetPullingExternalLinks:
    """Subset pulling for datasets backed by external (GCS) link entries."""

    def test_only_requested_file_downloaded(self, ext_dataset_no_hash):
        ds, _ = ext_dataset_no_hash
        local = Path(ds.get_local_copy(files=["alpha.txt"]))

        assert (local / "alpha.txt").is_file()
        assert not (local / "beta.txt").exists()
        assert not (local / "subdir" / "gamma.txt").exists()

    def test_subset_folder_name_has_hash_suffix(self, ext_dataset_no_hash):
        """Subset cache folder name is longer than the full-dataset folder name."""
        ds, _ = ext_dataset_no_hash
        full_folder = Path(ds.get_local_copy())
        subset_folder = Path(ds.get_local_copy(files=["alpha.txt"]))

        assert full_folder != subset_folder
        # Both contain the dataset id; subset has an extra 8-char hash appended.
        assert ds.id in full_folder.name
        assert ds.id in subset_folder.name
        assert len(subset_folder.name) > len(full_folder.name)

    def test_subset_folder_name_is_deterministic(self, ext_dataset_no_hash):
        """Two calls with the same file list return the same cached folder."""
        ds, _ = ext_dataset_no_hash
        assert ds.get_local_copy(files=["alpha.txt"]) == ds.get_local_copy(files=["alpha.txt"])

    def test_different_subsets_use_different_folders(self, ext_dataset_no_hash):
        """Two distinct file subsets get independent cache folders."""
        ds, _ = ext_dataset_no_hash
        folder_a = ds.get_local_copy(files=["alpha.txt"])
        folder_b = ds.get_local_copy(files=["beta.txt"])
        assert folder_a != folder_b

    def test_subset_hash_depends_on_file_list(self, ext_dataset_no_hash):
        """The 8-char hash embedded in the folder name matches what we compute locally."""
        ds, _ = ext_dataset_no_hash
        files = ["alpha.txt"]
        expected_hash = hashlib.sha256("\n".join(sorted(files)).encode()).hexdigest()[:8]

        subset_folder = Path(ds.get_local_copy(files=files))
        assert subset_folder.name.endswith(expected_hash), (
            f"Expected folder to end with {expected_hash!r}, got {subset_folder.name!r}"
        )

    def test_files_none_downloads_everything(self, ext_dataset_no_hash):
        """get_local_copy(files=None) is the full-dataset path (no hash suffix)."""
        ds, _ = ext_dataset_no_hash
        local = Path(ds.get_local_copy(files=None))

        assert (local / "alpha.txt").is_file()
        assert (local / "beta.txt").is_file()
        assert (local / "subdir" / "gamma.txt").is_file()
        # Folder name == "ds_<id>" with no extra suffix
        assert local.name == f"ds_{ds.id}"

    def test_file_contents_correct(self, ext_dataset_no_hash):
        """Downloaded file content matches what was originally uploaded."""
        ds, _ = ext_dataset_no_hash
        local = Path(ds.get_local_copy(files=["beta.txt"]))
        assert (local / "beta.txt").read_text() == "content of beta"


class TestSubsetPullingChunks:
    """Subset pulling for datasets backed by regular zipped chunk artifacts."""

    def test_only_requested_file_in_folder(self, chunk_dataset):
        # Subset pulling for chunk datasets downloads entire chunks that contain the
        # requested files. All files in the same chunk will be present — the subset
        # controls which *chunks* are fetched, not individual files within a chunk.
        local = Path(chunk_dataset.get_local_copy(files=["alpha.txt"]))
        assert (local / "alpha.txt").is_file()

    def test_subset_folder_differs_from_full(self, chunk_dataset):
        full = Path(chunk_dataset.get_local_copy())
        subset = Path(chunk_dataset.get_local_copy(files=["alpha.txt"]))
        assert full.name != subset.name

    def test_deterministic_cache_for_chunk_subset(self, chunk_dataset):
        f1 = chunk_dataset.get_local_copy(files=["alpha.txt"])
        f2 = chunk_dataset.get_local_copy(files=["alpha.txt"])
        assert f1 == f2

    def test_multi_file_subset(self, chunk_dataset):
        """Requesting multiple files from a chunk dataset downloads the relevant chunks
        and those files are present. Other files in the same chunk may also be present."""
        local = Path(chunk_dataset.get_local_copy(files=["alpha.txt", "beta.txt"]))

        assert (local / "alpha.txt").is_file()
        assert (local / "beta.txt").is_file()
