import sys
from typing import Callable, Union, IO, Optional, Dict, List, TYPE_CHECKING, Any

import six
from pathlib2 import Path

from ..frameworks import _patched_call, WeightsFileHandler, _Empty
from ..frameworks.base_bind import PatchBaseModelIO
from ..import_bind import PostImportHookPatching

if TYPE_CHECKING:
    from ... import Task
from ...config import running_remotely
from ...model import Framework


class PatchXGBoostModelIO(PatchBaseModelIO):
    _current_task = None
    __patched = None
    __callback_cls = None

    @staticmethod
    def update_current_task(task: Any, **kwargs: Any) -> None:
        PatchXGBoostModelIO._current_task = task
        if not task:
            return
        PatchXGBoostModelIO._patch_model_io()
        PostImportHookPatching.add_on_import("xgboost", PatchXGBoostModelIO._patch_model_io)

    @staticmethod
    def _patch_model_io() -> None:
        if PatchXGBoostModelIO.__patched:
            return

        if "xgboost" not in sys.modules:
            return
        PatchXGBoostModelIO.__patched = True
        # noinspection PyBroadException
        try:
            import xgboost as xgb  # noqa

            bst = xgb.Booster
            bst.save_model = _patched_call(bst.save_model, PatchXGBoostModelIO._save)
            bst.load_model = _patched_call(bst.load_model, PatchXGBoostModelIO._load)
            # noinspection PyBroadException
            try:
                from xgboost.callback import TrainingCallback  # noqa

                PatchXGBoostModelIO.__callback_cls = PatchXGBoostModelIO._generate_training_callback_class()
                xgb.train = _patched_call(xgb.train, PatchXGBoostModelIO._train)
                xgb.training.train = _patched_call(xgb.training.train, PatchXGBoostModelIO._train)
                xgb.sklearn.train = _patched_call(xgb.sklearn.train, PatchXGBoostModelIO._train)
            except ImportError:
                pass
            except Exception:
                pass

        except ImportError:
            pass
        except Exception:
            pass

    @staticmethod
    def _save(
        original_fn: Callable,
        obj: Any,
        f: Union[str, IO],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        ret = original_fn(obj, f, *args, **kwargs)
        if not PatchXGBoostModelIO._current_task:
            return ret

        if isinstance(f, six.string_types):
            filename = f
        elif hasattr(f, "name"):
            filename = f.name
            # noinspection PyBroadException
            try:
                f.flush()
            except Exception:
                pass
        else:
            filename = None

        # give the model a descriptive name based on the file name
        # noinspection PyBroadException
        try:
            model_name = Path(filename).stem
        except Exception:
            model_name = None
        WeightsFileHandler.create_output_model(
            obj,
            filename,
            Framework.xgboost,
            PatchXGBoostModelIO._current_task,
            singlefile=True,
            model_name=model_name,
        )
        return ret

    @staticmethod
    def _load(original_fn: Callable, f: Union[str, IO], *args: Any, **kwargs: Any) -> Any:
        if not PatchXGBoostModelIO._current_task:
            return original_fn(f, *args, **kwargs)

        if isinstance(f, six.string_types):
            filename = f
        elif hasattr(f, "name"):
            filename = f.name
        elif len(args) == 1 and isinstance(args[0], six.string_types):
            filename = args[0]
        else:
            filename = None

        # register input model
        empty = _Empty()
        # Hack: disabled
        if False and running_remotely():
            filename = WeightsFileHandler.restore_weights_file(
                empty, filename, Framework.xgboost, PatchXGBoostModelIO._current_task
            )
            model = original_fn(filename or f, *args, **kwargs)
        else:
            # try to load model before registering, in case we fail
            model = original_fn(f, *args, **kwargs)
            WeightsFileHandler.restore_weights_file(
                empty, filename, Framework.xgboost, PatchXGBoostModelIO._current_task
            )

        if empty.trains_in_model:
            # noinspection PyBroadException
            try:
                model.trains_in_model = empty.trains_in_model
            except Exception:
                pass
        return model

    @staticmethod
    def _train(original_fn: Callable, *args: Any, **kwargs: Any) -> Any:
        if not PatchXGBoostModelIO._current_task:
            return original_fn(*args, **kwargs)
        if PatchXGBoostModelIO.__callback_cls:
            callbacks = kwargs.get("callbacks") or []
            kwargs["callbacks"] = callbacks + [
                PatchXGBoostModelIO.__callback_cls(task=PatchXGBoostModelIO._current_task)
            ]
        return original_fn(*args, **kwargs)

    @classmethod
    def _generate_training_callback_class(cls) -> Optional[Any]:
        try:
            from xgboost.callback import TrainingCallback  # noqa
        except ImportError:
            return None

        class ClearMLCallback(TrainingCallback):
            """
            Log evaluation result at each iteration.
            """

            _scalar_index_counter = 0

            def __init__(self, task: "Task", period: int = 1) -> None:
                self.period = period
                assert period > 0
                self._last_eval = None
                self._last_eval_epoch = None
                self._logger = task.get_logger()
                self._scalar_index = ClearMLCallback._scalar_index_counter
                ClearMLCallback._scalar_index_counter += 1
                super(ClearMLCallback, self).__init__()

            def after_iteration(
                self,
                model: Any,
                epoch: int,
                evals_log: Dict[str, Dict[str, List[float]]],
            ) -> bool:
                """Run after each iteration.  Return True when training should stop."""
                if not evals_log:
                    return False

                if not (self.period == 1 or (epoch % self.period) == 0):
                    self._last_eval = evals_log
                    self._last_eval_epoch = epoch
                    return False

                self._report_eval_log(epoch, evals_log)

                self._last_eval = None
                self._last_eval_epoch = None
                return False

            def after_training(self, model: Any) -> Any:
                """Run after training is finished."""
                if self._last_eval:
                    self._report_eval_log(self._last_eval_epoch, self._last_eval)

                return model

            def _report_eval_log(self, epoch: int, eval_log: Dict[str, Dict[str, List[float]]]) -> None:
                for data, metric in eval_log.items():
                    if self._scalar_index != 0:
                        data = "{} - {}".format(data, self._scalar_index)
                    for metric_name, log in metric.items():
                        value = log[-1]

                        self._logger.report_scalar(title=data, series=metric_name, value=value, iteration=epoch)

        return ClearMLCallback
