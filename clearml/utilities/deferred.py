import threading
from functools import wraps
from typing import Callable, List, Type, Union, Any

import attr
import six


class DeferredExecutionPool(object):
    @attr.s
    class _DeferredAction(object):
        method = attr.ib()
        args = attr.ib()
        kwargs = attr.ib()

    def __init__(self, instance: Any) -> None:
        self._instance = instance
        self._pool = []
        self._lock = threading.Lock()

    def add(self, callable_: Callable, *args: Any, **kwargs: Any) -> None:
        self._pool.append(self._DeferredAction(callable_, args, kwargs))

    def clear(self) -> List["DeferredExecutionPool._DeferredAction"]:
        with self._lock:
            pool = self._pool
            self._pool = []
            return pool

    def apply(self) -> None:
        pool = self.clear()
        for action in pool:
            action.method(self._instance, *action.args, **action.kwargs)

    def copy_from(self, other: "DeferredExecutionPool") -> None:
        if not isinstance(self._instance, type(other._instance)):
            raise ValueError("Copy deferred actions must be with the same instance type")

        self._pool = other._pool[:]


class ParameterizedDefaultDict(dict):
    def __init__(self, factory: Callable[[Any], Any], *args: Any, **kwargs: Any) -> None:
        super(ParameterizedDefaultDict, self).__init__(*args, **kwargs)
        self._factory = factory

    def __missing__(self, key: Any) -> Any:
        self[key] = self._factory(key)
        return self[key]


class DeferredExecution(object):
    def __init__(self, pool_cls: type = DeferredExecutionPool) -> None:
        self._pools = ParameterizedDefaultDict(pool_cls)

    def __get__(self, instance: Any, owner: Type[Any]) -> Union["DeferredExecution", "DeferredExecutionPool"]:
        if not instance:
            return self

        return self._pools[instance]

    def defer_execution(
        self,
        condition_or_attr_name: Union[Callable[[Any], bool], str, bool] = True,
    ) -> Callable[[Callable], Callable]:
        """
        Deferred execution decorator, designed to wrap class functions for classes containing a deferred execution pool.
        :param condition_or_attr_name: Condition controlling whether wrapped function should be deferred.
            True by default. If a callable is provided, it will be called with the class instance (self)
            as first argument. If a string is provided, a class instance (self) attribute by that name is evaluated.
        :return:
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(instance: Any, *args: Any, **kwargs: Any) -> Any:
                if self._resolve_condition(instance, condition_or_attr_name):
                    self._pools[instance].add(func, *args, **kwargs)
                else:
                    return func(instance, *args, **kwargs)

            return wrapper

        return decorator

    @staticmethod
    def _resolve_condition(
        instance: Any,
        condition_or_attr_name: Union[Callable[[Any], Any], str, Any],
    ) -> Any:
        if callable(condition_or_attr_name):
            return condition_or_attr_name(instance)
        elif isinstance(condition_or_attr_name, six.string_types):
            return getattr(instance, condition_or_attr_name)
        return condition_or_attr_name

    def _apply(
        self,
        instance: Any,
        condition_or_attr_name: Union[Callable[[Any], bool], str, bool],
    ) -> None:
        if self._resolve_condition(instance, condition_or_attr_name):
            self._pools[instance].apply()

    def apply_after(
        self,
        condition_or_attr_name: Union[bool, Callable[[Any], bool], str] = True,
    ) -> Callable[[Callable], Callable]:
        """
        Decorator for applying deferred execution pool after wrapped function has completed
        :param condition_or_attr_name: Condition controlling whether deferred pool should be applied. True by default.
            If a callable is provided, it will be called with the class instance (self) as first argument.
            If a string is provided, a class instance (self) attribute by that name is evaluated.
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(instance: Any, *args: Any, **kwargs: Any) -> Any:
                res = func(instance, *args, **kwargs)
                self._apply(instance, condition_or_attr_name)
                return res

            return wrapper

        return decorator

    def apply_before(
        self,
        condition_or_attr_name: Union[bool, Callable[[Any], bool], str] = True,
    ) -> Callable[[Callable], Callable]:
        """
        Decorator for applying deferred execution pool before wrapped function is executed
        :param condition_or_attr_name: Condition controlling whether deferred pool should be applied. True by default.
            If a callable is provided, it will be called with the class instance (self) as first argument.
            If a string is provided, a class instance (self) attribute by that name is evaluated.
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(instance: Any, *args: Any, **kwargs: Any) -> Any:
                self._apply(instance, condition_or_attr_name)
                return func(instance, *args, **kwargs)

            return wrapper

        return decorator
