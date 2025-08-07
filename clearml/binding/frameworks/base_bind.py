from abc import ABCMeta, abstractmethod
from typing import Callable, Union, IO, TYPE_CHECKING, Any

import six

if TYPE_CHECKING:
    from clearml import Task


@six.add_metaclass(ABCMeta)
class PatchBaseModelIO(object):
    """
    Base class for patched models

    :param __main_task: Task to run (Experiment)
    :type __main_task: Task
    :param __patched: True if the model is patched
    :type __patched: bool
    """

    @property
    @abstractmethod
    def __main_task(self) -> "Task":
        pass

    @property
    @abstractmethod
    def __patched(self) -> bool:
        pass

    @staticmethod
    @abstractmethod
    def update_current_task(task: "Task", **kwargs: Any) -> None:
        """
        Update the model task to run
        :param task: the experiment to do
        :type task: Task
        """
        pass

    @staticmethod
    @abstractmethod
    def _patch_model_io() -> None:
        """
        Patching the load and save functions
        """
        pass

    @staticmethod
    @abstractmethod
    def _save(original_fn: Callable, obj: Any, f: Union[str, IO[Any]], *args: Any, **kwargs: Any) -> None:
        pass

    @staticmethod
    @abstractmethod
    def _load(original_fn: Callable, f: Any, *args: Any, **kwargs: Any) -> Any:
        pass
