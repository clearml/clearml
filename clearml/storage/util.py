import fnmatch
import sys
from typing import Optional, Callable, Any
from urllib.parse import quote

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
