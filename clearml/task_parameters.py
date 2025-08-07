from typing import (
    Optional,
    Union,
    Iterable,
    List,
    Callable,
    Tuple,
    Type,
    Dict,
    TYPE_CHECKING,
    Any,
)

import six
import attr
from attr import validators

if TYPE_CHECKING:
    from clearml import Task

__all__ = ["range_validator", "param", "percent_param", "TaskParameters"]


def _canonize_validator(current_validator: Union[None, Iterable[Any]]) -> List[Any]:
    """
    Convert current_validator to a new list and return it.

    If current_validator is None return an empty list.
    If current_validator is a list, return a copy of it.
    If current_validator is another type of  iterable, return a list version of it.
    If current_validator is a single value, return a one-list containing it.
    """

    if not current_validator:
        return []

    if isinstance(current_validator, (list, tuple)):
        current_validator = list(current_validator)
    else:
        current_validator = [current_validator]

    return current_validator


def range_validator(
    min_value: Optional[Union[int, float]], max_value: Optional[Union[int, float]]
) -> Callable[[Any, attr.Attribute, Union[int, float]], None]:
    """
    A parameter validator that checks range constraint on a parameter.

    :param min_value: The minimum limit of the range, inclusive. None for no minimum limit.
    :param max_value: The maximum limit of the range, inclusive. None for no maximum limit.
    :return: A new range validator.
    """

    def _range_validator(instance: Any, attribute: attr.Attribute, value: Union[int, float]) -> None:
        if ((min_value is not None) and (value < min_value)) or ((max_value is not None) and (value > max_value)):
            raise ValueError("{} must be in range [{}, {}]".format(attribute.name, min_value, max_value))

    return _range_validator


def param(
    validator: Union[Callable, List[Callable], None] = None,
    range: Union[Tuple[Optional[int], Optional[int]], None] = None,
    type: Union[Type[int], Type[str], Type[float], None] = None,
    desc: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    *args: Any,
    **kwargs: Any,
) -> attr.Attribute:
    """
    A parameter inside a TaskParameters class.

    See TaskParameters for more information.

    :param validator: A validator or validators list.
        Any validator from attr.validators is applicable.

    :param range: The legal values range of the parameter.
        A tuple (min_limit, max_limit). None for no limitation.

    :param type: The type of the parameter.
        Supported types are int, str and float. None to place no limit of the type

    :param desc: A string description of the parameter, for future use.

    :param metadata: A dictionary metadata of the parameter, for future use.

    :param args: Additional arguments to pass to attr.attrib constructor.
    :param kwargs: Additional keyword arguments to pass to attr.attrib constructor.

    :return: An attr.attrib instance to use with TaskParameters class.

    Warning: Do not create an immutable param using args or kwargs. It will cause
    connect method of the TaskParameters class to fail.
    """

    metadata = metadata or {}
    metadata["desc"] = desc

    validator = _canonize_validator(validator)

    if type:
        validator.append(validators.optional(validators.instance_of(type)))

    if range:
        validator.append(range_validator(*range))

    return attr.ib(validator=validator, type=type, metadata=metadata, *args, **kwargs)


def percent_param(*args: Any, **kwargs: Any) -> attr.Attribute:
    """
    A param with type float and range limit (0, 1).
    """
    return param(range=(0, 1), type=float, *args, **kwargs)


class _AttrsMeta(type):
    def __new__(mcs: Type["_AttrsMeta"], name: str, bases: Tuple[Any], dct: Dict[str, Any]) -> type:
        new_class = super(_AttrsMeta, mcs).__new__(mcs, name, bases, dct)
        return attr.s(new_class)


@six.add_metaclass(_AttrsMeta)
class TaskParameters(object):
    """
    Base class for task parameters.

    Inherit this class to create a parameter set to connect to a task.

    Usage Example:
    class MyParams(TaskParameters):
        iterations = param(
            type=int,
            desc="Number of iterations to run",
            range=(0, 100000),
        )

        target_accuracy = percent_param(
            desc="The target accuracy of the model",
        )
    """

    def to_dict(self) -> Dict[str, Any]:
        """
        :return: A new dictionary with keys are the parameters names and values
            are the corresponding values.
        """
        return attr.asdict(self)

    def update_from_dict(self, source_dict: dict) -> None:
        """
        Update the parameters using values from a dictionary.

        :param source_dict: A dictionary with an entry for each parameter to
            update.
        """
        for key, value in source_dict.items():
            if not hasattr(self, key):
                raise ValueError("Unknown key {} in {} object".format(key, type(self).__name__))

            setattr(self, key, value)

    def connect(self, task: "Task") -> None:
        """
        Connect to a task.

        When running locally, the task will save the parameters from self.
        When running with a worker, self will be updated according to the task's
        saved parameters.

        :param task: The task to connect to.
        :type task: .Task
        """

        return task.connect(self)
