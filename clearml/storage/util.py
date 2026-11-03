import fnmatch
import logging
import sys
from functools import lru_cache
from typing import Optional, Callable, Any
from urllib.parse import quote
from zipfile import ZIP_STORED, ZIP_DEFLATED, ZIP_BZIP2, ZIP_LZMA

from ..config import ConfigSDKWrapper

# Imports backwards compatibility
from .filepaths import get_common_path  # noqa: F401
from .hashing import (  # noqa: F401
    sha256sum,
    md5text,
    crc32text,
    hash_text,
    hash_dict,
)
from .url import quote_url  # noqa: F401
from .size import format_size, parse_size  # noqa: F401
from .archive import extract_tar_archive as safe_extract  # noqa: F401


_ZIP_COMPRESSION_BY_NAME = {
    "stored": ZIP_STORED,
    "deflated": ZIP_DEFLATED,
    "bzip2": ZIP_BZIP2,
    "lzma": ZIP_LZMA,
}


@lru_cache(maxsize=None)
def _warn_invalid_zip_compression(value_repr: str) -> None:
    # lru_cache makes this a warn-once per distinct configured value
    logging.getLogger("clearml.storage").warning(
        "Invalid sdk.storage.zip_compression value %s, expected one of %s. Using default compression",
        value_repr,
        ", ".join(sorted(_ZIP_COMPRESSION_BY_NAME)),
    )


def get_configured_zip_compression(default: int = ZIP_DEFLATED) -> int:
    """
    Return the zip compression algorithm configured in ``sdk.storage.zip_compression``.

    The configuration value is the algorithm name as a string — one of ``"stored"``
    (no compression), ``"deflated"``, ``"bzip2"`` or ``"lzma"`` (case-insensitive,
    an optional ``"zip_"`` prefix is accepted). A raw ``zipfile`` constant integer
    is accepted as well. When the entry is missing or invalid, ``default`` is
    returned (invalid values also emit a warning, once per value).

    :param default: Compression to use when nothing (valid) is configured
    :return: A ``zipfile`` compression constant (e.g. ``ZIP_DEFLATED``)
    """
    # noinspection PyBroadException
    try:
        value = ConfigSDKWrapper.get("storage.zip_compression", None)
    except Exception:
        return default
    if value is None:
        return default
    if isinstance(value, int) and value in _ZIP_COMPRESSION_BY_NAME.values():
        return value
    if isinstance(value, str):
        compression = _ZIP_COMPRESSION_BY_NAME.get(value.strip().lower().replace("zip_", ""))
        if compression is not None:
            return compression
    _warn_invalid_zip_compression(repr(value))
    return default


def get_config_object_matcher(**patterns: Any) -> Callable:
    unsupported = {
        k: v
        for k, v in patterns.items()
        if not isinstance(v, str)
    }
    if unsupported:
        object_matcher_expectation = ", ".join(
            f"{k}={v}"
            for k, v in unsupported.items()
        )
        raise ValueError(f"Unsupported object matcher (expecting string): {object_matcher_expectation}")

    # optimize simple patters
    starts_with = {
        k: v.rstrip("*")
        for k, v in patterns.items()
        if (
            "*" not in v.rstrip("*")
            and "?" not in v
        )
    }
    optimized_patterns = {
        k: v
        for k, v in patterns.items()
        if v not in starts_with
    }

    def _matcher(**kwargs: Any) -> Optional[bool]:
        for key, value in kwargs.items():
            if not value:
                continue
            start = starts_with.get(key)
            if start:
                if value.startswith(start):
                    return True
            else:
                pattern = optimized_patterns.get(key)
                if pattern and fnmatch.fnmatch(value, pattern):
                    return True

    return _matcher


def is_windows() -> bool:
    """
    :return: True if currently running on windows OS
    """
    return sys.platform == "win32"


def encode_string_to_filename(text: str) -> str:
    """
    Encodes a string to be a valid filename.
    return quote(text, safe=" ")
    """
    return quote(text, safe=" ")
