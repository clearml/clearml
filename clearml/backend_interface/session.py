from abc import ABCMeta, abstractmethod
from typing import Any

import six


class SendError(Exception):
    """A session send() error class"""

    @property
    def result(self) -> Any:
        return self._result

    def __init__(self, result: Any, *args: Any, **kwargs: Any) -> None:
        super(SendError, self).__init__(*args, **kwargs)
        self._result = result


@six.add_metaclass(ABCMeta)
class SessionInterface(object):
    """Session wrapper interface providing a session property and a send convenience method"""

    @property
    @abstractmethod
    def session(self) -> Any:
        pass

    @abstractmethod
    def send(
        self,
        req: Any,
        ignore_errors: bool = False,
        raise_on_errors: bool = True,
        async_enable: bool = False,
    ) -> Any:
        pass
