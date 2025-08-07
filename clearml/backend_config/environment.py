from os import getenv, environ
from typing import Callable, Union, Dict, Any

from .converters import text_to_bool
from .entry import Entry, NotSet


class EnvEntry(Entry):
    @classmethod
    def default_conversions(cls) -> Dict[Any, Callable[[str], Any]]:
        conversions = super(EnvEntry, cls).default_conversions().copy()
        conversions[bool] = text_to_bool
        return conversions

    def __init__(self, key: str, *more_keys: Any, **kwargs: Any) -> None:
        super(EnvEntry, self).__init__(key, *more_keys, **kwargs)
        self._ignore_errors = kwargs.pop("ignore_errors", False)

    def pop(self) -> None:
        for k in self.keys:
            environ.pop(k, None)

    def _get(self, key: str) -> Union[str, Any]:
        value = getenv(key, "").strip()
        return value or NotSet

    def _set(self, key: str, value: str) -> None:
        environ[key] = value

    def __str__(self) -> str:
        return "env:{}".format(super(EnvEntry, self).__str__())

    def error(self, message: str) -> None:
        if not self._ignore_errors:
            print("Environment configuration: {}".format(message))

    def exists(self) -> bool:
        return any(key for key in self.keys if getenv(key) is not None)
