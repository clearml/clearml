import os
from pathlib import Path
import subprocess
import sys

import pytest

from clearml.storage.cache import CacheManager


@pytest.fixture
def cache(monkeypatch, tmp_path):
    monkeypatch.setenv("CLEARML_CACHE_DIR", str(tmp_path))
    return CacheManager.CacheContext("test-locks", default_cache_file_limit=1)


def populate_cache(cache):
    folder = Path(cache.get_cache_folder())
    dataset = folder / "dataset"
    dataset.mkdir(parents=True)
    data = dataset / "data.txt"
    data.write_text("dataset in use")
    os.utime(data, (1, 1))
    os.utime(dataset, (1, 1))
    recent = folder / "recent"
    recent.write_text("recent file")
    os.utime(recent, (100, 100))
    return dataset, recent


def clean_cache_in_subprocess():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from clearml.storage.cache import CacheManager; "
            "CacheManager.CacheContext('test-locks', default_cache_file_limit=1).clean_cache()",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize("stale_lock", [False, True])
def test_clean_cache_preserves_folder_locked_by_another_process(cache, stale_lock):
    dataset, recent = populate_cache(cache)
    cache.lock_cache_folder(str(dataset))
    active_lock = dataset.parent / ".lock.000.dataset.clearml"
    if stale_lock:
        # Process an abandoned lock before encountering the live lock.
        abandoned_lock = dataset.parent / ".lock.001.dataset.clearml"
        abandoned_lock.touch()
        os.utime(active_lock, (2, 2))
        os.utime(abandoned_lock, (3, 3))
    try:
        clean_cache_in_subprocess()
        assert (dataset / "data.txt").read_text() == "dataset in use"
        assert active_lock.exists()
        assert recent.read_text() == "recent file"
        if stale_lock:
            assert not abandoned_lock.exists()
    finally:
        cache.unlock_cache_folder(str(dataset))

    assert not active_lock.exists()
    clean_cache_in_subprocess()
    assert not dataset.exists()
    assert recent.read_text() == "recent file"


def test_clean_cache_preserves_folder_locked_in_same_process(cache):
    dataset, recent = populate_cache(cache)
    cache.lock_cache_folder(str(dataset))
    try:
        cache.clean_cache()
        assert (dataset / "data.txt").read_text() == "dataset in use"
        assert recent.exists()
    finally:
        cache.unlock_cache_folder(str(dataset))


def test_clean_cache_removes_unlocked_folder_and_abandoned_lock(cache):
    dataset, recent = populate_cache(cache)
    abandoned_lock = dataset.parent / ".lock.000.dataset.clearml"
    abandoned_lock.touch()

    cache.clean_cache()

    assert not dataset.exists()
    assert not abandoned_lock.exists()
    assert recent.read_text() == "recent file"


def test_clean_cache_below_limit_keeps_cached_file(cache):
    folder = Path(cache.get_cache_folder())
    folder.mkdir(parents=True)
    data = folder / "data.txt"
    data.write_text("cached")

    assert cache.clean_cache() is False
    assert data.read_text() == "cached"
