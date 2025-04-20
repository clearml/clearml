""" Convenience classes supporting python3-like concepts """
import abc
from types import TracebackType
from typing import Optional, Type

import six


@six.add_metaclass(abc.ABCMeta)
class AbstractContextManager(object):
    """An abstract base class for context managers. Supported in contextlib from python 3.6 and up"""

    def __enter__(self) -> "AbstractContextManager":
        """Return `self` upon entering the runtime context."""
        return self

    @abc.abstractmethod
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        """Raise any exception triggered within the runtime context."""
        return None

    @classmethod
    def __subclasshook__(cls, C: type) -> Optional[bool]:
        if cls is AbstractContextManager:
            if any("__enter__" in B.__dict__ for B in C.__mro__) and any("__exit__" in B.__dict__ for B in C.__mro__):
                return True
        return NotImplemented


try:
    from abc import abstractclassmethod

except ImportError:

    class abstractclassmethod(classmethod):
        __isabstractmethod__ = True

        def __init__(self, callable: callable) -> None:
            callable.__isabstractmethod__ = True
            super(abstractclassmethod, self).__init__(callable)
