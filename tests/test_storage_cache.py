import os
import subprocess
import sys
from pathlib import Path

import pytest

from clearml.storage.cache import CacheManager


@pytest.fixture
def cache_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("CLEARML_CACHE_DIR", str(tmp_path))
    cache = CacheManager.CacheContext("test", default_cache_file_limit=1)
    folder = Path(cache.get_cache_folder()) / "dataset"
    folder.mkdir(parents=True)
    (folder / "data").write_text("in use")
    os.utime(folder / "data", (1, 1))
    os.utime(folder, (1, 1))
    (folder.parent / "recent").write_text("keep")
    return cache, folder


def clean_in_subprocess():
    subprocess.run(
        [sys.executable, "-c", "from clearml.storage.cache import CacheManager; "
         "CacheManager.CacheContext('test', default_cache_file_limit=1).clean_cache()"],
        check=True, timeout=30,
    )


@pytest.mark.parametrize("stale_first", [False, True])
def test_cleanup_preserves_locked_folder(cache_folder, stale_first):
    cache, folder = cache_folder
    cache.lock_cache_folder(folder)
    try:
        locks = list(folder.parent.glob("*" + CacheManager._lockfile_suffix))
        assert locks
        if stale_first:
            stale = folder.parent / (CacheManager._lockfile_prefix + "999.dataset" + CacheManager._lockfile_suffix)
            stale.touch()
            os.utime(stale, (2000000000, 2000000000))
        cache.clean_cache()  # Same-process protection remains intact.
        assert (folder / "data").read_text() == "in use"
        clean_in_subprocess()
        assert (folder / "data").read_text() == "in use"
        assert all(lock.exists() for lock in locks)
    finally:
        cache.unlock_cache_folder(folder)


@pytest.mark.parametrize("stale", [False, True])
def test_cleanup_removes_unlocked_folder(cache_folder, stale):
    cache, folder = cache_folder
    if stale:
        lock = folder.parent / (CacheManager._lockfile_prefix + "000.dataset" + CacheManager._lockfile_suffix)
        lock.touch()
    clean_in_subprocess()
    assert not folder.exists()
    assert (folder.parent / "recent").exists()
    if stale:
        assert not lock.exists()
