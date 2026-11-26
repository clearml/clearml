import os
import sys
import warnings
from functools import partial
from io import IOBase
from typing import Callable, Union, Any

from pathlib2 import Path

from .import_bind import PostImportHookPatching
from ..binding.frameworks import _patched_call, _Empty, WeightsFileHandler
from ..config import running_remotely
from ..debugging.log import LoggerRoot
from ..model import Framework
from ..utilities.lowlevel.file_access import (
    get_filename_from_file_object,
    buffer_writer_close_cb,
)


class PatchedJoblib:
    _patched_joblib = False
    _patched_sk_joblib = False
    _current_task = None

    @staticmethod
    def patch_joblib() -> None:
        # try manually
        PatchedJoblib._patch_joblib()
        # register callback
        PostImportHookPatching.add_on_import("joblib", PatchedJoblib._patch_joblib)
        PostImportHookPatching.add_on_import("sklearn", PatchedJoblib._patch_joblib)

    @staticmethod
    def _patch_joblib() -> bool:
        # noinspection PyBroadException
        try:
            if not PatchedJoblib._patched_joblib and "joblib" in sys.modules:
                PatchedJoblib._patched_joblib = True
                try:
                    import joblib
                except ImportError:
                    joblib = None

                if joblib:
                    # Hook the public dump() / load() as well as the internal functions: joblib >= 1.5 writes
                    # uncompressed files through a plain open() and removed _read_fileobject, so the internal
                    # hooks alone no longer register anything. Inside the public calls the internal hooks are
                    # skipped by _patched_call's recursion guard, so nothing is registered twice.
                    joblib.dump = _patched_call(joblib.dump, PatchedJoblib._dump)
                    joblib.load = _patched_call(joblib.load, PatchedJoblib._load_public)
                    joblib.numpy_pickle._write_fileobject = _patched_call(
                        joblib.numpy_pickle._write_fileobject,
                        partial(PatchedJoblib._write_fileobject, joblib.numpy_pickle),
                    )
                    # removed in joblib 1.5
                    if hasattr(joblib.numpy_pickle, "_read_fileobject"):
                        joblib.numpy_pickle._read_fileobject = _patched_call(
                            joblib.numpy_pickle._read_fileobject, PatchedJoblib._load
                        )
                    joblib.numpy_pickle.NumpyPickler.__init__ = _patched_call(
                        joblib.numpy_pickle.NumpyPickler.__init__,
                        PatchedJoblib._numpypickler,
                    )
                    joblib.memory.MemorizedFunc._cached_call = _patched_call(
                        joblib.memory.MemorizedFunc._cached_call,
                        PatchedJoblib._cached_call_recursion_guard,
                    )

            if not PatchedJoblib._patched_sk_joblib and "sklearn" in sys.modules:
                PatchedJoblib._patched_sk_joblib = True
                try:
                    import sklearn  # noqa: F401

                    # avoid deprecation warning, we must import sklearn before, so we could catch it
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        from sklearn.externals import joblib as sk_joblib
                except ImportError:
                    sk_joblib = None

                if sk_joblib:
                    sk_joblib.numpy_pickle._write_fileobject = _patched_call(
                        sk_joblib.numpy_pickle._write_fileobject,
                        partial(PatchedJoblib._write_fileobject, sk_joblib.numpy_pickle),
                    )
                    sk_joblib.numpy_pickle._read_fileobject = _patched_call(
                        sk_joblib.numpy_pickle._read_fileobject, PatchedJoblib._load
                    )
                    sk_joblib.numpy_pickle.NumpyPickler.__init__ = _patched_call(
                        sk_joblib.numpy_pickle.NumpyPickler.__init__,
                        PatchedJoblib._numpypickler,
                    )

        except Exception:
            return False
        return True

    @staticmethod
    def update_current_task(task: Any) -> None:
        PatchedJoblib._current_task = task
        if not task:
            return
        PatchedJoblib.patch_joblib()

    @staticmethod
    def _dump(original_fn: Callable, *args: Any, **kwargs: Any) -> Any:
        ret = original_fn(*args, **kwargs)
        if not PatchedJoblib._current_task:
            return ret
        # dump(value, filename, ...): either may be passed by keyword, and filename may be a path object
        obj = args[0] if args else kwargs.get("value")
        f = args[1] if len(args) > 1 else kwargs.get("filename")
        if hasattr(f, "__fspath__"):
            f = os.fspath(f)
        if f is not None:
            PatchedJoblib._register_dump(obj, f)
        return ret

    @staticmethod
    def _load_public(original_fn: Callable, *args: Any, **kwargs: Any) -> Any:
        # load(filename, mmap_mode=None, ...)
        f = args[0] if args else kwargs.get("filename")
        if hasattr(f, "__fspath__"):
            f = os.fspath(f)
        if f is None:
            return original_fn(*args, **kwargs)

        def load_with_original_arguments(*_: Any, **__: Any) -> Any:
            # _load() calls this with the file name only; load() keeps its own arguments (mmap_mode, ...)
            return original_fn(*args, **kwargs)

        return PatchedJoblib._load(load_with_original_arguments, f)

    @staticmethod
    def _numpypickler(original_fn: Callable, obj: Any, f: Union[str, IOBase], *args: Any, **kwargs: Any) -> Any:
        ret = original_fn(obj, f, *args, **kwargs)
        if not PatchedJoblib._current_task:
            return ret

        fname = f if isinstance(f, str) else None
        fileobj = ret if isinstance(f, str) else f

        if fileobj and hasattr(fileobj, "close"):

            def callback(*_: Any) -> None:
                PatchedJoblib._register_dump(obj, fname or fileobj)

            if isinstance(fname, str) or hasattr(fileobj, "name"):
                buffer_writer_close_cb(fileobj, callback)
        else:
            PatchedJoblib._register_dump(obj, f)

        return ret

    @staticmethod
    def _write_fileobject(obj: Any, original_fn: Callable, f: Union[str, Any], *args: Any, **kwargs: Any) -> Any:
        ret = original_fn(f, *args, **kwargs)
        if not PatchedJoblib._current_task:
            return ret

        fname = f if isinstance(f, str) else None
        fileobj = ret if isinstance(f, str) else f

        if fileobj and hasattr(fileobj, "close"):

            def callback(*_: Any) -> None:
                PatchedJoblib._register_dump(obj, fname or fileobj)

            if isinstance(fname, str) or hasattr(fileobj, "name"):
                buffer_writer_close_cb(fileobj, callback)
        else:
            PatchedJoblib._register_dump(obj, f)
        return ret

    @staticmethod
    def _register_dump(obj: Any, f: Union[str, IOBase]) -> None:
        filename = get_filename_from_file_object(f, flush=True)
        if not filename:
            return

        # give the model a descriptive name based on the file name
        # noinspection PyBroadException
        try:
            model_name = Path(filename).stem
        except Exception:
            model_name = None
        current_framework = PatchedJoblib.get_model_framework(obj)
        WeightsFileHandler.create_output_model(
            obj,
            filename,
            current_framework,
            PatchedJoblib._current_task,
            singlefile=True,
            model_name=model_name,
        )

    @staticmethod
    def _load(original_fn: Callable, f: Union[str, IOBase], *args: Any, **kwargs: Any) -> Any:
        if not PatchedJoblib._current_task:
            return original_fn(f, *args, **kwargs)

        filename = get_filename_from_file_object(f, flush=False)

        # register input model
        empty = _Empty()
        # Hack: disabled
        if False and running_remotely():
            # we assume scikit-learn, for the time being
            current_framework = Framework.scikitlearn
            filename = WeightsFileHandler.restore_weights_file(
                empty, filename, current_framework, PatchedJoblib._current_task
            )
            model = original_fn(filename or f, *args, **kwargs)
        else:
            # try to load model before registering, in case we fail
            model = original_fn(f, *args, **kwargs)
            current_framework = PatchedJoblib.get_model_framework(model)
            WeightsFileHandler.restore_weights_file(empty, filename, current_framework, PatchedJoblib._current_task)

        if empty.trains_in_model:
            # noinspection PyBroadException
            try:
                model.trains_in_model = empty.trains_in_model
            except Exception:
                pass
        return model

    @staticmethod
    def get_model_framework(obj: Any) -> Framework:
        framework = Framework.scikitlearn
        object_orig_module = None
        # noinspection PyBroadException
        try:
            object_orig_module = obj.__module__ if hasattr(obj, "__module__") else obj.__package__
            model = object_orig_module.partition(".")[0]
            if model == "sklearn":
                framework = Framework.scikitlearn
            elif model == "xgboost":
                framework = Framework.xgboost
            else:
                framework = Framework.scikitlearn
        except Exception:
            LoggerRoot.get_base_logger().debug(
                f"Can't get model framework {object_orig_module}, model framework will be: {framework} "
            )
        return framework

    @staticmethod
    def _cached_call_recursion_guard(original_fn: Callable, *args: Any, **kwargs: Any) -> Any:
        # used just to avoid getting into the `_load` binding in the context of memory caching
        return original_fn(*args, **kwargs)
