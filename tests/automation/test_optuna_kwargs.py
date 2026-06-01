# SPDX-License-Identifier: Apache-2.0

"""Tests for OptimizerOptuna verified_optuna_kwargs passthrough.

Verifies that storage and load_if_exists are accepted and forwarded
to optuna.create_study(), while unrecognised kwargs are silently dropped
(consistent with OptimizerBOHB's verified_bohb_kwargs pattern).
"""

from unittest.mock import MagicMock, patch


def _make_optimizer(extra_kwargs=None):
    """Minimal OptimizerOptuna without a live ClearML task."""
    extra_kwargs = extra_kwargs or {}

    mock_objective = MagicMock()
    mock_objective.len = 1
    mock_objective.get_objective_sign.return_value = [-1]

    mock_param = MagicMock()

    with patch("clearml.automation.optuna.optuna.Task"):
        from clearml.automation.optuna.optuna import OptimizerOptuna

        with patch.object(OptimizerOptuna, "__init__", wraps=OptimizerOptuna.__init__):
            optimizer = OptimizerOptuna.__new__(OptimizerOptuna)
            # set required SearchStrategy attrs directly, bypassing super().__init__
            optimizer._objective_metric = mock_objective
            optimizer._optuna_sampler = None
            optimizer._optuna_pruner = None
            optimizer._param_iterator = None
            optimizer._objective = None
            optimizer._study = None
            optimizer.parameter_override_history = []
            optimizer._optimizer_task = None
            optimizer._hyper_parameters = [mock_param]
            optimizer.max_iteration_per_job = 10
            optimizer.min_iteration_per_job = None
            optimizer.pool_period_minutes = 2.0
            optimizer.total_max_jobs = 5
            optimizer._num_concurrent_workers = 1
            optimizer._base_task_id = "base_task_id"
            optimizer._execution_queue = "default"

            verified_optuna_kwargs = ["storage", "load_if_exists"]
            optimizer._optuna_kwargs = dict(
                (k, v) for k, v in extra_kwargs.items() if k in verified_optuna_kwargs
            )
            optimizer._optuna_kwargs.setdefault("load_if_exists", False)

    return optimizer


class TestVerifiedOptunaKwargs:
    def test_storage_is_stored(self):
        opt = _make_optimizer({"storage": "sqlite:///my_study.db"})
        assert opt._optuna_kwargs["storage"] == "sqlite:///my_study.db"

    def test_load_if_exists_true_is_stored(self):
        opt = _make_optimizer({"load_if_exists": True})
        assert opt._optuna_kwargs["load_if_exists"] is True

    def test_load_if_exists_defaults_to_false(self):
        opt = _make_optimizer()
        assert opt._optuna_kwargs["load_if_exists"] is False

    def test_unknown_kwarg_is_dropped(self):
        opt = _make_optimizer({"unknown_param": "value", "storage": "sqlite:///x.db"})
        assert "unknown_param" not in opt._optuna_kwargs
        assert "storage" in opt._optuna_kwargs

    def test_empty_kwargs_only_has_load_if_exists_default(self):
        opt = _make_optimizer()
        assert set(opt._optuna_kwargs.keys()) == {"load_if_exists"}


class TestCreateStudyReceivesKwargs:
    def _run_start_with_kwargs(self, extra_kwargs):
        """Run start() and return the create_study call args."""
        import optuna as real_optuna

        opt = _make_optimizer(extra_kwargs)

        mock_study = MagicMock()
        mock_study.optimize = MagicMock()

        with patch("clearml.automation.optuna.optuna.optuna") as mock_optuna:
            mock_optuna.create_study.return_value = mock_study
            mock_optuna.TrialPruned = real_optuna.TrialPruned

            opt._convert_hyper_parameters_to_optuna = MagicMock(return_value={})
            opt._objective_metric.len = 1
            opt._objective_metric.get_objective_sign.return_value = [-1]

            with patch("clearml.automation.optuna.optuna.OptunaObjective"):
                opt.start()

        return mock_optuna.create_study.call_args

    def test_storage_forwarded_to_create_study(self):
        call_args = self._run_start_with_kwargs({"storage": "sqlite:///test.db"})
        assert call_args.kwargs.get("storage") == "sqlite:///test.db"

    def test_load_if_exists_true_forwarded(self):
        call_args = self._run_start_with_kwargs({"load_if_exists": True})
        assert call_args.kwargs.get("load_if_exists") is True

    def test_load_if_exists_defaults_false_in_create_study(self):
        call_args = self._run_start_with_kwargs({})
        assert call_args.kwargs.get("load_if_exists") is False

    def test_clearml_managed_params_still_present(self):
        call_args = self._run_start_with_kwargs({})
        assert "direction" in call_args.kwargs
        assert "sampler" in call_args.kwargs
        assert "pruner" in call_args.kwargs
