import logging.config
from copy import deepcopy
from types import TracebackType
from typing import Optional, Dict, Union, Type, Tuple, Any

from pathlib2 import Path


def logger(path: Optional[str] = None) -> logging.Logger:
    """Get a ClearML Python logging.Logger named according to the parent path or stem of the file path provided"""
    name = "clearml"
    if path:
        p = Path(path)
        module = (p.parent if p.stem.startswith("_") else p).stem
        name = "clearml.%s" % module
    return logging.getLogger(name)


def initialize(
    logging_config: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    if extra is not None:
        from logging import Logger

        class _Logger(Logger):
            __extra = extra.copy()

            def _log(
                self,
                level: int,
                msg: str,
                args: tuple,
                exc_info: Optional[Union[bool, Tuple[Type[BaseException], BaseException, TracebackType]]] = None,
                extra: Optional[dict] = None,
                **kwargs: Any
            ) -> None:
                extra = extra or {}
                extra.update(self.__extra)
                super(_Logger, self)._log(level, msg, args, exc_info=exc_info, extra=extra, **kwargs)

        Logger.manager.loggerClass = _Logger

    if logging_config is not None:
        root_logger = logging.getLogger()
        #added a check to avoid calling dictConfig
        #if the root user has already set up their own logging.
        #prevents ClearML from closing their handlers.
        if not root_logger.handlers:
            logging.config.dictConfig(deepcopy(dict(logging_config)))
