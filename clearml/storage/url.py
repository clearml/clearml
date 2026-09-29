from typing import Sequence
from urllib.parse import (
    quote,
    urlparse,
    urlunparse,
)


def quote_url(
    url: str,
    valid_schemes: Sequence[str] = ("http", "https"),
) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in valid_schemes:
        return url
    parsed = parsed._replace(path=quote(parsed.path))
    return str(urlunparse(parsed))
