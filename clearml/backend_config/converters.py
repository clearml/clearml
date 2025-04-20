import base64
from typing import Union, Optional, Any, TypeVar, Callable, Tuple

import six

try:
    from typing import Text
except ImportError:
    # windows conda-less hack
    Text = object

ConverterType = TypeVar("ConverterType", bound=Callable[[Any], Any])


def strtobool(val: Text) -> int:
    """Convert a string representation of truth to true (1) or false (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return 1
    elif val in ("n", "no", "f", "false", "off", "0"):
        return 0
    else:
        raise ValueError("invalid truth value %r" % (val,))


def base64_to_text(value: Any) -> Text:
    return base64.b64decode(value).decode("utf-8")


def text_to_bool(value: Text) -> bool:
    return bool(strtobool(value))


def safe_text_to_bool(value: Text) -> bool:
    try:
        return bool(strtobool(value))
    except ValueError:
        return bool(value)


def any_to_bool(value: Optional[Union[int, float, Text]]) -> bool:
    if isinstance(value, six.text_type):
        return text_to_bool(value)
    return bool(value)


def or_(*converters: ConverterType, **kwargs: Tuple[Exception]) -> ConverterType:
    """
    Wrapper that implements an "optional converter" pattern. Allows specifying a converter
    for which a set of exceptions is ignored (and the original value is returned)
    :param converters: A converter callable
    :param exceptions: A tuple of exception types to ignore
    """
    # noinspection PyUnresolvedReferences
    exceptions = kwargs.get("exceptions", (ValueError, TypeError))

    def wrapper(value: Any) -> Any:
        for converter in converters:
            try:
                return converter(value)
            except exceptions:
                pass
        return value

    return wrapper
