from logging import getLogger
from typing import Optional, Any

from ..task import Task

_logger = getLogger("clearml.external.kerastuner")

try:
    import pandas as pd
except ImportError:
    pd = None
    _logger.warning("Pandas is not installed, summary table reporting will be skipped.")

try:
    from kerastuner import Logger
except ImportError:
    _logger.warning("Legacy ClearmlTunerLogger requires 'kerastuner<1.3.0'")
else:

    class ClearmlTunerLogger(Logger):
        # noinspection PyTypeChecker
        def __init__(self, task: Optional[Task] = None) -> ():
            super(ClearmlTunerLogger, self).__init__()
            self.task = task or Task.current_task()
            if not self.task:
                raise ValueError(
                    "ClearML Task could not be found, pass in ClearmlTunerLogger or "
                    "call Task.init before initializing ClearmlTunerLogger"
                )
            self._summary = pd.DataFrame() if pd else None

        def register_tuner(self, tuner_state: dict) -> ():
            """Informs the logger that a new search is starting."""
            pass

        def register_trial(self, trial_id: str, trial_state: dict) -> ():
            """Informs the logger that a new Trial is starting."""
            if not self.task:
                return
            data = {
                "trial_id_{}".format(trial_id): trial_state,
            }
            data.update(self.task.get_model_config_dict())
            self.task.connect_configuration(data)
            self.task.get_logger().tensorboard_single_series_per_graph(True)
            self.task.get_logger()._set_tensorboard_series_prefix(trial_id + " ")
            self.report_trial_state(trial_id, trial_state)

        def report_trial_state(self, trial_id: str, trial_state: dict) -> ():
            if self._summary is None or not self.task:
                return

            trial = {}
            for k, v in trial_state.get("metrics", {}).get("metrics", {}).items():
                m = "metric/{}".format(k)
                observations = trial_state["metrics"]["metrics"][k].get("observations")
                if observations:
                    observations = observations[-1].get("value")
                if observations:
                    trial[m] = observations[-1]
            for k, v in trial_state.get("hyperparameters", {}).get("values", {}).items():
                m = "values/{}".format(k)
                trial[m] = trial_state["hyperparameters"]["values"][k]

            if trial_id in self._summary.index:
                columns = set(list(self._summary) + list(trial.keys()))
                if len(columns) != self._summary.columns.size:
                    self._summary = self._summary.reindex(set(list(self._summary) + list(trial.keys())), axis=1)
                self._summary.loc[trial_id, :] = pd.DataFrame(trial, index=[trial_id]).loc[trial_id, :]
            else:
                self._summary = self._summary.append(pd.DataFrame(trial, index=[trial_id]), sort=False)

            self._summary.index.name = "trial id"
            self._summary = self._summary.reindex(columns=sorted(self._summary.columns))
            self.task.get_logger().report_table("summary", "trial", 0, table_plot=self._summary)

        def exit(self) -> None:
            if not self.task:
                return
            self.task.flush(wait_for_uploads=True)


try:
    from tensorflow.keras.callbacks import Callback
except ImportError:
    _logger.warning(
        "Could not import 'tensorflow.keras.callbacks.Callback'. ClearmlTunerCallback will not be importable"
    )
else:

    class ClearmlTunerCallback(Callback):
        def __init__(
            self,
            tuner: Any,
            best_trials_reported: int = 100,
            task: Optional[Task] = None,
        ) -> None:
            self.task = task or Task.current_task()
            if not self.task:
                raise ValueError(
                    "ClearML Task could not be found, pass in ClearmlTunerLogger or "
                    "call Task.init before initializing ClearmlTunerLogger"
                )
            self.tuner = tuner
            self.best_trials_reported = best_trials_reported
            super(ClearmlTunerCallback, self).__init__()

        def on_train_end(self, *args: Any, **kwargs: Any) -> None:
            summary = pd.DataFrame() if pd else None
            if summary is None:
                return
            best_trials = self.tuner.oracle.get_best_trials(self.best_trials_reported)
            for trial in best_trials:
                trial_dict = {"trial id": trial.trial_id}
                for hparam in trial.hyperparameters.space:
                    trial_dict[hparam.name] = trial.hyperparameters.values.get(hparam.name)
                summary = pd.concat(
                    [summary, pd.DataFrame(trial_dict, index=[trial.trial_id])],
                    ignore_index=True,
                )
            summary.index.name = "trial id"
            summary = summary[["trial id", *sorted(summary.columns[1:])]]
            self.task.get_logger().report_table("summary", "trial", 0, table_plot=summary)
