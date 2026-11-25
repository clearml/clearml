from abc import ABC, abstractmethod
import math
import os
import shutil
import zipfile
from tempfile import mkstemp
from typing import (
    List,
    Dict,
    Union,
    Optional,
    Mapping,
    TYPE_CHECKING,
    Sequence,
    Tuple,
    Callable,
    Any,
)
from uuid import uuid4

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

from .backend_api import Session
from .backend_api.services import models, projects
from pathlib2 import Path

from .utilities.config import config_dict_to_text, text_to_config_dict
from .utilities.proxy_object import cast_basic_type
from .utilities.plotly_reporter import SeriesInfo

from .backend_interface.util import (
    validate_dict,
    get_single_result,
    mutually_exclusive,
    exact_match_regex,
    get_or_create_project,
)
from .debugging.log import get_logger
from .errors import UsageError
from .storage.cache import CacheManager
from .storage.helper import StorageHelper
from .storage.filepaths import get_common_path
from .utilities.enum import Options
from .backend_interface import Task as _Task
from .backend_interface.model import create_dummy_model, Model as _Model
from .backend_interface.session import SendError
from .config import running_remotely, get_cache_dir
from .backend_interface.metrics import Reporter, Metrics

if TYPE_CHECKING:
    from .task import Task


class Framework(Options):
    """
    Optional frameworks for output model
    """

    tensorflow = "TensorFlow"
    tensorflowjs = "TensorFlow_js"
    tensorflowlite = "TensorFlow_Lite"
    pytorch = "PyTorch"
    torchscript = "TorchScript"
    caffe = "Caffe"
    caffe2 = "Caffe2"
    onnx = "ONNX"
    keras = "Keras"
    mknet = "MXNet"
    cntk = "CNTK"
    torch = "Torch"
    darknet = "Darknet"
    paddlepaddle = "PaddlePaddle"
    scikitlearn = "ScikitLearn"
    xgboost = "XGBoost"
    lightgbm = "LightGBM"
    parquet = "Parquet"
    megengine = "MegEngine"
    catboost = "CatBoost"
    tensorrt = "TensorRT"
    openvino = "OpenVINO"

    __file_extensions_mapping = {
        ".pb": (
            tensorflow,
            tensorflowjs,
            onnx,
        ),
        ".meta": (tensorflow,),
        ".pbtxt": (
            tensorflow,
            onnx,
        ),
        ".zip": (tensorflow,),
        ".tgz": (tensorflow,),
        ".tar.gz": (tensorflow,),
        "model.json": (tensorflowjs,),
        ".tflite": (tensorflowlite,),
        ".pth": (pytorch,),
        ".pt": (pytorch,),
        ".caffemodel": (caffe,),
        ".prototxt": (caffe,),
        "predict_net.pb": (caffe2,),
        "predict_net.pbtxt": (caffe2,),
        ".onnx": (onnx,),
        ".h5": (keras,),
        ".hdf5": (keras,),
        ".keras": (keras,),
        ".model": (mknet, cntk, xgboost),
        "-symbol.json": (mknet,),
        ".cntk": (cntk,),
        ".t7": (torch,),
        ".cfg": (darknet,),
        "__model__": (paddlepaddle,),
        ".pkl": (scikitlearn, keras, xgboost, megengine),
        ".parquet": (parquet,),
        ".cbm": (catboost,),
        ".plan": (tensorrt,),
    }

    __parent_mapping = {
        "tensorflow": (
            tensorflow,
            tensorflowjs,
            tensorflowlite,
            keras,
        ),
        "pytorch": (pytorch,),
        "xgboost": (xgboost,),
        "lightgbm": (lightgbm,),
        "catboost": (catboost,),
        "joblib": (scikitlearn, xgboost),
    }

    @classmethod
    def get_framework_parents(cls, framework: str) -> List[str]:
        if not framework:
            return []
        parents = []
        for k, v in cls.__parent_mapping.items():
            if framework in v:
                parents.append(k)
        return parents

    @classmethod
    def _get_file_ext(cls, framework: Optional[str], filename: str) -> Tuple[Optional[str], str]:
        mapping = cls.__file_extensions_mapping
        filename = filename.lower()

        def find_framework_by_ext(
            framework_selector: Callable[[List[str]], Optional[str]]
        ) -> Optional[Tuple[str, str]]:
            for ext, frameworks in mapping.items():
                if frameworks and filename.endswith(ext):
                    fw = framework_selector(frameworks)
                    if fw:
                        return fw, ext

        # If no framework, try finding first framework matching the extension, otherwise (or if no match) try matching
        # the given extension to the given framework. If no match return an empty extension
        return (
            (not framework and find_framework_by_ext(lambda frameworks_: frameworks_[0]))
            or find_framework_by_ext(lambda frameworks_: framework if framework in frameworks_ else None)
            or (framework, filename.split(".")[-1] if "." in filename else "")
        )


class BaseModel(ABC):
    # noinspection PyProtectedMember
    _archived_tag = _Task.archived_tag
    _package_tag = "package"

    @property
    def id(self) -> str:
        """
        The ID (system UUID) of the model.

        :return: The model ID.
        """
        return self._get_model_data().id

    @property
    def name(self) -> str:
        """
        The name of the model.

        :return: The model name.
        """
        return self._get_model_data().name

    @name.setter
    def name(self, value: str) -> None:
        """
        Set the model name.

        :param value: The model name.
        """
        self._get_base_model().update(name=value)

    @property
    def project(self) -> str:
        """
        Project ID of the model.

        :return: Project ID
        """
        data = self._get_model_data()
        return data.project

    @project.setter
    def project(self, value: str) -> None:
        """
        Set the project ID of the model.

        :param value: Project ID

        :type value: str
        """
        self._get_base_model().update(project_id=value)

    @property
    def comment(self) -> str:
        """
        A description of the model.

        :return: The model description.
        """
        return self._get_model_data().comment

    @comment.setter
    def comment(self, value: str) -> None:
        """
        Set model description.

        :param value: The model comment/description.
        """
        self._get_base_model().update(comment=value)

    @property
    def tags(self) -> List[str]:
        """
        A list of tags describing the model.

        :return: The list of tags.
        """
        return self._get_model_data().tags

    @tags.setter
    def tags(self, value: List[str]) -> None:
        """
        Set the list of tags describing the model.

        :param value: The tags.

        :type value: list(str)
        """
        self._get_base_model().update(tags=value)

    @property
    def system_tags(self) -> List[str]:
        """
        A list of system tags describing the model.

        :return: The list of tags.
        """
        data = self._get_model_data()
        return data.system_tags if Session.check_min_api_version("2.3") else data.tags

    @system_tags.setter
    def system_tags(self, value: List[str]) -> None:
        """
        Set the list of system tags describing the model.

        :param value: The tags.

        :type value: list(str)
        """
        self._get_base_model().update(system_tags=value)

    @property
    def config_text(self) -> str:
        """
        The configuration as a string. For example, prototxt, a ``.ini`` file, or Python code to evaluate.

        :return: The configuration.
        """
        # noinspection PyProtectedMember
        return _Model._unwrap_design(self._get_model_data().design)

    @property
    def config_dict(self) -> dict:
        """
        The configuration as a dictionary, parsed from the design text. This usually represents the model configuration.
        For example, prototxt, a ``.ini`` file, or Python code to evaluate.

        :return: The configuration.
        """
        return self._text_to_config_dict(self.config_text)

    @property
    def labels(self) -> Dict[str, int]:
        """
        The label enumeration of string (label) to integer (value) pairs.


        :return: A dictionary containing label enumeration, where the keys are labels and the values are integers.
        """
        return self._get_model_data().labels

    @property
    def task(self) -> str:
        """
        The ID of the task connected to this model.  If no task is connected, returns the ID of the task that originally
        created it.

        :return: The Task ID
        """
        return self._task.id if self._task else self.original_task

    @property
    def original_task(self) -> str:
        """
        Return the ID of the Task that created this model.

        :return: The Task ID
        """
        return self._get_base_model().task

    @property
    def url(self) -> str:
        """
        Return the URL of the model file (or archived files)

        :return: The model file URL.
        """
        return self._get_base_model().uri

    @property
    def published(self) -> bool:
        """
        Get the published state of this model.

        :return: ``True`` if the model is published, ``False`` otherwise.

        """
        return self._get_base_model().locked

    @property
    def framework(self) -> str:
        """
        The ML framework of the model (for example: PyTorch, TensorFlow, XGBoost, etc.).

        :return: The model's framework
        """
        return self._get_model_data().framework

    def __init__(self, task: "Task" = None) -> None:
        super(BaseModel, self).__init__()
        self._log = get_logger()
        self._task = None
        self._reload_required = False
        self._reporter = None
        self._floating_data = None
        self._name = None
        self._task_connect_name = None
        self._set_task(task)

    def get_weights(
        self,
        raise_on_error: bool = False,
        force_download: bool = False,
        extract_archive: bool = False,
    ) -> str:
        """
        Download the base model and return the locally stored filename.

        :param raise_on_error: If ``True``, raise ``ValueError`` if the artifact download fails.
        :param force_download: If ``True``, re-download base model even if a cached copy exists.
        :param extract_archive: If ``True``, extract the downloaded weights file if possible.

        :return: The locally stored file.
        """
        # download model (synchronously) and return local file
        return self._get_base_model().download_model_weights(
            raise_on_error=raise_on_error,
            force_download=force_download,
            extract_archive=extract_archive,
        )

    def get_weights_package(
        self,
        return_path: bool = False,
        raise_on_error: bool = False,
        force_download: bool = False,
        extract_archive: bool = True,
    ) -> Optional[Union[str, List[Path]]]:
        """
        Download the base model package into a temporary directory (extract the files), or return a list of the
        locally stored filenames.

        :param return_path: If ``True``, extract weights to a temp directory and return its path.
            If ``False`` (default), return a list of local file paths.
        :param raise_on_error: If ``True``, raise ``ValueError`` if the artifact download fails.
            If ``False``, returns ``None`` and logs a warning.
        :param force_download: If ``True``, re-download the base artifact even if a cached copy exists.
        :param extract_archive: If ``True``, extract the downloaded weights file if possible.

        :return: The model weights, or a list of the locally stored filenames.
            If ``raise_on_error=False``, returns ``None`` on error.
        """
        # check if model was packaged
        if not self._is_package():
            raise ValueError("Model is not packaged")

        # download packaged model
        model_path = self.get_weights(
            raise_on_error=raise_on_error,
            force_download=force_download,
            extract_archive=extract_archive,
        )

        if not model_path:
            if raise_on_error:
                raise ValueError(f"Model package '{self.url}' could not be downloaded")
            return None

        if return_path:
            return model_path

        target_files = list(Path(model_path).glob("*"))
        return target_files

    def report_scalar(self, title: str, series: str, value: float, iteration: int) -> None:
        """
        Plot a scalar series.

        :param title: Plot title (metric). Plot more than one scalar series on the same plot by using
            the same ``title`` for each call to this method.
        :param series: Series name (variant).
        :param value: The value to plot per iteration.
        :param iteration: The reported iteration / step (x-axis of the reported time series)
        """
        self._init_reporter()
        return self._reporter.report_scalar(title=title, series=series, value=float(value), iter=iteration)

    def report_single_value(self, name: str, value: float) -> None:
        """
        Reports a single value metric (for example, total experiment accuracy or mAP)

        :param name: Metric's name
        :param value: Metric's value
        """
        self._init_reporter()
        return self._reporter.report_scalar(title="Summary", series=name, value=float(value), iter=-(2**31))

    def report_histogram(
        self,
        title: str,
        series: str,
        values: Sequence[Union[int, float]],
        iteration: Optional[int] = None,
        labels: Optional[List[str]] = None,
        xlabels: Optional[List[str]] = None,
        xaxis: Optional[str] = None,
        yaxis: Optional[str] = None,
        mode: Optional[str] = None,  # Literal["group", "stack", "relative"]
        data_args: Optional[dict] = None,
        extra_layout: Optional[dict] = None,
    ) -> None:
        """
        Plot a (default grouped) histogram.
        Notice this function will not calculate the histogram,
        it assumes the histogram was already calculated in ``values``.

        For example:

        .. code-block:: py

            vector_series = np.random.randint(10, size=10).reshape(2,5)
            model.report_histogram(
                title='histogram example',
                series='histogram series',
                values=vector_series,
                iteration=0,
                labels=['A','B'],
                xaxis='X axis label',
                yaxis='Y axis label',
            )

        :param title: Plot title (metric).
        :param series: Series name (variant).
        :param values: The series values. A list of floats, or an ``N``-dimensional Numpy array containing
            data for each histogram bar.
        :param iteration: The reported iteration / step. Each ``iteration`` creates another plot.
        :param labels: Labels for each bar group, creating a plot legend labeling each series.
        :param xlabels: Labels per entry in each bucket in the histogram (vector), creating a set of labels
            for each histogram bar on the x-axis.
        :param xaxis: The x-axis title.
        :param yaxis: The y-axis title.
        :param mode: Display mode for multiple histograms. The options are:

          - ``group`` (default)
          - ``stack``
          - ``relative``

        :param data_args: Optional dictionary for data configuration passed directly to ``plotly``.
            See full details on the supported configuration: https://plotly.com/javascript/reference/bar/.
            Example: ``data_args={'orientation': 'h', 'marker': {'color': 'blue'}}``
        :param extra_layout: Optional dictionary for layout configuration, passed directly to ``plotly``.
            See full details on the supported configuration: https://plotly.com/javascript/reference/bar/.
            Example: ``extra_layout={'xaxis': {'type': 'date', 'range': ['2020-01-01', '2020-01-31']}}``
        """
        self._init_reporter()

        if not isinstance(values, np.ndarray):
            values = np.array(values)

        return self._reporter.report_histogram(
            title=title,
            series=series,
            histogram=values,
            iter=iteration or 0,
            labels=labels,
            xlabels=xlabels,
            xtitle=xaxis,
            ytitle=yaxis,
            mode=mode or "group",
            data_args=data_args,
            layout_config=extra_layout,
        )

    def report_vector(
        self,
        title: str,
        series: str,
        values: Sequence[Union[int, float]],
        iteration: Optional[int] = None,
        labels: Optional[List[str]] = None,
        xlabels: Optional[List[str]] = None,
        xaxis: Optional[str] = None,
        yaxis: Optional[str] = None,
        mode: Optional[str] = None,  # Literal["group", "stack", "relative"]
        extra_layout: Optional[dict] = None,
    ) -> None:
        """
        Plot a vector as a (default stacked) histogram.

        For example:

        .. code-block:: py

            vector_series = np.random.randint(10, size=10).reshape(2,5)
            model.report_vector(
                title='vector example',
                series='vector series',
                values=vector_series,
                iteration=0,
                labels=['A','B'],
                xaxis='X axis label',
                yaxis='Y axis label',
            )

        :param title: Plot title (metric).
        :param series: Series name (variant).
        :param values: Vector data as a list of floats or an N-dimensional Numpy array containing data
            for each histogram bar.
        :param iteration: The reported iteration / step. Each ``iteration`` creates another plot.
        :param labels: Labels for each bar group, creating a plot legend labeling each series.
        :param xlabels: Labels per entry in each bucket in the histogram (vector), creating a set of labels
            for each histogram bar on the x-axis.
        :param xaxis: The x-axis title.
        :param yaxis: The y-axis title.
        :param mode: Display mode for multiple histograms. The options are:

          - ``group`` (default)
          - ``stack``
          - ``relative``

        :param extra_layout: Optional dictionary for layout configuration, passed directly to ``plotly``.
            See full details on the supported configuration: https://plotly.com/javascript/reference/layout/.
            Example: ``extra_layout={'showlegend': False, 'plot_bgcolor': 'yellow'}``
        """
        self._init_reporter()
        return self.report_histogram(
            title,
            series,
            values,
            iteration or 0,
            labels=labels,
            xlabels=xlabels,
            xaxis=xaxis,
            yaxis=yaxis,
            mode=mode,
            extra_layout=extra_layout,
        )

    def report_table(
        self,
        title: str,
        series: str,
        iteration: Optional[int] = None,
        table_plot: Optional[Union["pd.DataFrame", Sequence[Sequence]]] = None,
        csv: Optional[str] = None,
        url: Optional[str] = None,
        extra_layout: Optional[Dict] = None,
    ) -> None:
        """
        Report a table plot.

        One and only one of the following parameters must be provided.

        - ``table_plot`` - Pandas DataFrame or Table as list of rows (list)
        - ``csv`` - CSV file
        - ``url`` - URL to CSV file

        For example:

        .. code-block:: py

            df = pd.DataFrame(
                {
                    'num_legs': [2, 4, 8, 0],
                    'num_wings': [2, 0, 0, 0],
                    'num_specimen_seen': [10, 2, 1, 8]
                },
                index=['falcon', 'dog', 'spider', 'fish'],
            )

           model.report_table(title='table example', series='pandas DataFrame', iteration=0, table_plot=df)

        :param title: Table title (metric).
        :param series: Series name (variant).
        :param iteration: The reported iteration / step.
        :param table_plot: The output table plot object.
        :param csv: Path to local CSV file.
        :param url: A URL to the location of CSV file.
        :param extra_layout: Optional dictionary for layout configuration passed directly to ``plotly``.
            See full details on the supported configuration: https://plotly.com/javascript/reference/layout/.
            Example: ``extra_layout={'height': 600}``
        """
        mutually_exclusive(UsageError, _check_none=True, table_plot=table_plot, csv=csv, url=url)
        table = table_plot
        if url or csv:
            if not pd:
                raise UsageError(
                    "pandas is required in order to support reporting tables using CSV or a URL, "
                    "please install the pandas python package"
                )
            if url:
                table = pd.read_csv(url, index_col=[0])
            elif csv:
                table = pd.read_csv(csv, index_col=[0])

        def replace(dst: Any, *srcs: Any) -> None:
            for src in srcs:
                reporter_table.replace(src, dst, inplace=True)

        if isinstance(table, (list, tuple)):
            reporter_table = table
        else:
            reporter_table = table.fillna(str(np.nan))
            replace("NaN", np.nan, math.nan)
            replace("Inf", np.inf, math.inf)
            minus_inf = [-np.inf, -math.inf]
            try:
                minus_inf.append(np.NINF)
            except AttributeError:
                # NINF has been removed in numpy>2.0
                pass
            replace("-Inf", *minus_inf)
        self._init_reporter()
        return self._reporter.report_table(
            title=title,
            series=series,
            table=reporter_table,
            iteration=iteration or 0,
            layout_config=extra_layout,
        )

    def report_line_plot(
        self,
        title: str,
        series: Sequence[SeriesInfo],
        xaxis: str,
        yaxis: str,
        mode: str = "lines",
        iteration: Optional[int] = None,
        reverse_xaxis: bool = False,
        comment: Optional[str] = None,
        extra_layout: Optional[dict] = None,
    ) -> None:
        """
        Plot one or more series as lines.

        :param title: Plot title (metric).
        :param series: All the series data, one list element for each line in the plot.
        :param iteration: The reported iteration / step.
        :param xaxis: The x-axis title.
        :param yaxis: The y-axis title.
        :param mode: The type of line plot. The options are: ``lines`` (default), ``markers``, ``lines+markers``.
        :param reverse_xaxis:  If ``True``, reverse the x-axis (high to low). Defaults to ``False``.
        :param comment: A comment displayed underneath the plot title.
        :param extra_layout: Dictionary for layout configuration, passed directly to ``plotly``.
            See full details on the supported configuration: https://plotly.com/javascript/reference/scatter/.
            Example: ``extra_layout={'xaxis': {'type': 'date', 'range': ['2020-01-01', '2020-01-31']}}``
        """
        self._init_reporter()

        # noinspection PyArgumentList
        series = [SeriesInfo(**s) if isinstance(s, dict) else s for s in series]

        return self._reporter.report_line_plot(
            title=title,
            series=series,
            iter=iteration or 0,
            xtitle=xaxis,
            ytitle=yaxis,
            mode=mode,
            reverse_xaxis=reverse_xaxis,
            comment=comment,
            layout_config=extra_layout,
        )

    def report_scatter2d(
        self,
        title: str,
        series: str,
        scatter: Union[Sequence[Tuple[float, float]], np.ndarray],
        iteration: Optional[int] = None,
        xaxis: Optional[str] = None,
        yaxis: Optional[str] = None,
        labels: Optional[List[str]] = None,
        mode: str = "line",
        comment: Optional[str] = None,
        extra_layout: Optional[dict] = None,
    ) -> None:
        """
        Report a 2D scatter plot.

        For example:

        .. code-block:: py

            scatter2d = np.hstack((
                np.atleast_2d(np.arange(0, 10)).T,
                np.random.randint(10, size=(10, 1))
            ))
            model.report_scatter2d(
                title="example_scatter",
                series="series",
                iteration=0,
                scatter=scatter2d,
                xaxis="title x",
                yaxis="title y",
            )

        Plot multiple 2D scatter series on the same plot by passing the same ``title`` and ``iteration`` values
        to this method:

        .. code-block:: py

            scatter2d_1 = np.hstack((
                np.atleast_2d(np.arange(0, 10)).T,
                np.random.randint(10, size=(10, 1))
            ))
            model.report_scatter2d(
                title="example_scatter",
                series="series_1",
                iteration=1,
                scatter=scatter2d_1,
                xaxis="title x",
                yaxis="title y",
            )

            scatter2d_2 = np.hstack((
                np.atleast_2d(np.arange(0, 10)).T,
                np.random.randint(10, size=(10, 1)),
            ))
            model.report_scatter2d(
                "example_scatter",
                "series_2",
                iteration=1,
                scatter=scatter2d_2,
                xaxis="title x",
                yaxis="title y",
            )

        :param title: Plot title (metric).
        :param series: Series name (variant) of the reported scatter plot.
        :param scatter: The scatter data. ``numpy.ndarray`` or list of (pairs of x,y) scatter.
        :param iteration: The reported iteration / step.
        :param xaxis: The x-axis title.
        :param yaxis: The y-axis title.
        :param labels: Labels per point in the data assigned to the ``scatter`` parameter. The labels must be
            in the same order as the data.
        :param mode: The type of scatter plot. The options are: ``lines`` (default), ``markers``, ``lines+markers``.
        :param comment: A comment displayed with the plot, underneath the title.
        :param extra_layout: Dictionary for layout configuration, passed directly to ``plotly``.
            See full details on the supported configuration: https://plotly.com/javascript/reference/scatter/.
            Example: ``extra_layout={'xaxis': {'type': 'date', 'range': ['2020-01-01', '2020-01-31']}}``
        """
        self._init_reporter()

        if not isinstance(scatter, np.ndarray):
            if not isinstance(scatter, list):
                scatter = list(scatter)
            scatter = np.array(scatter)

        return self._reporter.report_2d_scatter(
            title=title,
            series=series,
            data=scatter,
            iter=iteration or 0,
            mode=mode,
            xtitle=xaxis,
            ytitle=yaxis,
            labels=labels,
            comment=comment,
            layout_config=extra_layout,
        )

    def report_scatter3d(
        self,
        title: str,
        series: str,
        scatter: Union[Sequence[Tuple[float, float, float]], np.ndarray],
        iteration: Optional[int] = None,
        xaxis: Optional[str] = None,
        yaxis: Optional[str] = None,
        zaxis: Optional[str] = None,
        labels: Optional[List[str]] = None,
        mode: str = "markers",
        fill: bool = False,
        comment: Optional[str] = None,
        extra_layout: Optional[dict] = None,
    ) -> None:
        """
        Plot a 3D scatter graph. For example:

        .. code-block:: py

            scatter3d = np.random.randint(10, size=(10, 3))
            model.report_scatter3d(
                title="example_scatter_3d",
                series="series_xyz",
                iteration=1,
                scatter=scatter3d,
                xaxis="title x",
                yaxis="title y",
                zaxis="title z",
            )

        :param title: Plot title (metric)
        :param series: Series name (variant)
        :param scatter: The scatter data as

          - a list of ``(x,y,z)`` tuples
          - a nested list ``[[(x1,y1,z1)...]]``, or
          - a ``numpy.ndarray``.

        :param iteration: The reported iteration / step.
        :param xaxis: The x-axis title.
        :param yaxis: The y-axis title.
        :param zaxis: The z-axis title.
        :param labels: Labels per point in the data assigned to the ``scatter`` parameter. The labels must be
            in the same order as the data.
        :param mode: The type of scatter plot. The options are: ``markers`` (default), ``lines``, ``lines+markers``.
        :param fill: If ``True``, fill the area under the curve. Defaults to ``False``.
        :param comment: A comment displayed underneath the plot title.
        :param extra_layout: Dictionary for layout configuration passed directly to ``plotly``.
            See full details on the supported configuration: https://plotly.com/javascript/reference/scatter3d/.
            Example: ``extra_layout={'xaxis': {'type': 'date', 'range': ['2020-01-01', '2020-01-31']}}``
        """
        self._init_reporter()

        # check if multiple series
        multi_series = isinstance(scatter, list) and (
            isinstance(scatter[0], np.ndarray)
            or (scatter[0] and isinstance(scatter[0], list) and isinstance(scatter[0][0], list))
        )

        if not multi_series:
            if not isinstance(scatter, np.ndarray):
                if not isinstance(scatter, list):
                    scatter = list(scatter)
                scatter = np.array(scatter)
            try:
                scatter = scatter.astype(np.float32)
            except ValueError:
                pass

        return self._reporter.report_3d_scatter(
            title=title,
            series=series,
            data=scatter,
            iter=iteration or 0,
            labels=labels,
            mode=mode,
            fill=fill,
            comment=comment,
            xtitle=xaxis,
            ytitle=yaxis,
            ztitle=zaxis,
            layout_config=extra_layout,
        )

    def report_confusion_matrix(
        self,
        title: str,
        series: str,
        matrix: np.ndarray,
        iteration: Optional[int] = None,
        xaxis: Optional[str] = None,
        yaxis: Optional[str] = None,
        xlabels: Optional[List[str]] = None,
        ylabels: Optional[List[str]] = None,
        yaxis_reversed: bool = False,
        comment: Optional[str] = None,
        extra_layout: Optional[dict] = None,
    ) -> None:
        """
        Plot a heat-map matrix.

        For example:

        .. code-block:: py

            confusion = np.random.randint(10, size=(10, 10))
            model.report_confusion_matrix(
                "example confusion matrix",
                "ignored",
                iteration=1,
                matrix=confusion,
                xaxis="title X",
                yaxis="title Y",
            )

        :param title: Plot title (metric).
        :param series: Series name (variant).
        :param matrix: A heat-map matrix (example: confusion matrix).
        :param iteration: The reported iteration / step.
        :param xaxis: The x-axis title.
        :param yaxis: The y-axis title.
        :param xlabels: Labels for each column of the matrix.
        :param ylabels: Labels for each row of the matrix.
        :param yaxis_reversed: If set to ``False``, the ``(0, 0)`` coordinate is at the bottom left corner.
            If set to ``True``, the ``(0, 0)`` coordinate is at the top left corner.
        :param comment: A comment displayed with the plot, underneath the title.
        :param extra_layout: Optional dictionary for layout configuration, passed directly to ``plotly``.
            See full details on the supported configuration: https://plotly.com/javascript/reference/heatmap/.
            Example: ``extra_layout={'xaxis': {'type': 'date', 'range': ['2020-01-01', '2020-01-31']}}``
        """
        self._init_reporter()

        if not isinstance(matrix, np.ndarray):
            matrix = np.array(matrix)

        return self._reporter.report_value_matrix(
            title=title,
            series=series,
            data=matrix.astype(np.float32),
            iter=iteration or 0,
            xtitle=xaxis,
            ytitle=yaxis,
            xlabels=xlabels,
            ylabels=ylabels,
            yaxis_reversed=yaxis_reversed,
            comment=comment,
            layout_config=extra_layout,
        )

    def report_matrix(
        self,
        title: str,
        series: str,
        matrix: np.ndarray,
        iteration: Optional[int] = None,
        xaxis: Optional[str] = None,
        yaxis: Optional[str] = None,
        xlabels: Optional[List[str]] = None,
        ylabels: Optional[List[str]] = None,
        yaxis_reversed: bool = False,
        extra_layout: Optional[dict] = None,
    ) -> None:
        """
        Plot a confusion matrix.

        .. note::
           This method is the same as ```Model.report_confusion_matrix```.

        :param title: Plot title (metric).
        :param series: Series name (variant).
        :param matrix: A heat-map matrix (example: confusion matrix).
        :param iteration: The reported iteration / step.
        :param xaxis: The x-axis title.
        :param yaxis: The y-axis title.
        :param xlabels: Labels for each column of the matrix.
        :param ylabels: Labels for each row of the matrix.
        :param yaxis_reversed: If set to ``False``, the ``(0, 0)`` coordinate is at the bottom left corner.
            If set to ``True``, the ``(0, 0)`` coordinate is at the top left corner.
        :param extra_layout: Dictionary for layout configuration, passed directly to ``plotly``.
            See full details on the supported configuration: https://plotly.com/javascript/reference/heatmap/.
            Example: ``extra_layout={'xaxis': {'type': 'date', 'range': ['2020-01-01', '2020-01-31']}}``
        """
        self._init_reporter()
        return self.report_confusion_matrix(
            title,
            series,
            matrix,
            iteration or 0,
            xaxis=xaxis,
            yaxis=yaxis,
            xlabels=xlabels,
            ylabels=ylabels,
            yaxis_reversed=yaxis_reversed,
            extra_layout=extra_layout,
        )

    def report_surface(
        self,
        title: str,
        series: str,
        matrix: np.ndarray,
        iteration: Optional[int] = None,
        xaxis: Optional[str] = None,
        yaxis: Optional[str] = None,
        zaxis: Optional[str] = None,
        xlabels: Optional[List[str]] = None,
        ylabels: Optional[List[str]] = None,
        camera: Optional[Sequence[float]] = None,
        comment: Optional[str] = None,
        extra_layout: Optional[dict] = None,
    ) -> None:
        """
        Report a 3D surface plot.

        .. note::
           This method plots the same data as ```Model.report_confusion_matrix```, but presents the
           data as a surface diagram not a confusion matrix.

        .. code-block:: py

            surface_matrix = np.random.randint(10, size=(10, 10))
            model.report_surface(
                "example surface",
                "series",
                iteration=0,
                matrix=surface_matrix,
                xaxis="title X",
                yaxis="title Y",
                zaxis="title Z",
            )

        :param title: Plot title (metric).
        :param series: Series name (variant).
        :param matrix: A heat-map matrix (example: confusion matrix).
        :param iteration: The reported iteration / step.
        :param xaxis: The x-axis title.
        :param yaxis: The y-axis title.
        :param zaxis: The z-axis title.
        :param xlabels: Labels for each column of the matrix (optional).
        :param ylabels: Labels for each row of the matrix (optional).
        :param camera: ``(X,Y,Z)`` coordinates indicating the camera position. The default value is ``(1,1,1)``.
        :param comment: A comment displayed underneath the plot title.
        :param extra_layout: Dictionary for layout configuration passed directly to ``plotly``.
            See full details on the supported configuration: https://plotly.com/javascript/reference/surface/.
            Example: ``extra_layout={'xaxis': {'type': 'date', 'range': ['2020-01-01', '2020-01-31']}}``
        """
        self._init_reporter()

        if not isinstance(matrix, np.ndarray):
            matrix = np.array(matrix)

        return self._reporter.report_value_surface(
            title=title,
            series=series,
            data=matrix.astype(np.float32),
            iter=iteration or 0,
            xlabels=xlabels,
            ylabels=ylabels,
            xtitle=xaxis,
            ytitle=yaxis,
            ztitle=zaxis,
            camera=camera,
            comment=comment,
            layout_config=extra_layout,
        )

    def publish(self) -> None:
        """
        Set the model to the status ``published`` and for public use. If the model's status is already ``published``,
        then this method is a no-op.
        """

        if not self.published:
            self._get_base_model().publish()

    def archive(self) -> None:
        """
        Archive the model. If the model is already archived, this is a no-op
        """
        try:
            self._get_base_model().archive()
        except Exception:
            pass

    def unarchive(self) -> None:
        """
        Unarchive the model. If the model is not archived, this is a no-op
        """
        try:
            self._get_base_model().unarchive()
        except Exception:
            pass

    def get_offline_mode_folder(self) -> str:
        from clearml import Task as OfflineTask

        return OfflineTask.current_task().get_offline_mode_folder()

    def _init_reporter(self) -> None:
        if self._reporter:
            return
        self._base_model = self._get_force_base_model()
        metrics_manager = Metrics(
            session=_Model._get_default_session(),
            storage_uri=None,
            task=self,  # this is fine, the ID of the model will be fetched here
            for_model=True,
        )
        self._reporter = Reporter(metrics=metrics_manager, task=self, for_model=True)

    def _running_remotely(self) -> None:
        return bool(running_remotely() and self._task is not None)

    def _set_task(self, value: _Task) -> None:
        if value is not None and not isinstance(value, _Task):
            raise ValueError("task argument must be of Task type")
        self._task = value

    @abstractmethod
    def _get_model_data(self) -> Any:
        pass

    @abstractmethod
    def _get_base_model(self) -> _Model:
        pass

    def _set_package_tag(self) -> None:
        if self._package_tag not in self.system_tags:
            self.system_tags.append(self._package_tag)
            self._get_base_model().edit(system_tags=self.system_tags)

    def _is_package(self) -> bool:
        return self._package_tag in (self.system_tags or [])

    @staticmethod
    def _config_dict_to_text(config: Union[str, dict]) -> str:
        if not isinstance(config, (str, dict)):
            raise ValueError("Model configuration only supports dictionary or string objects")
        return config_dict_to_text(config)

    @staticmethod
    def _text_to_config_dict(text: str) -> dict:
        if not isinstance(text, str):
            raise ValueError("Model configuration parsing only supports string")
        return text_to_config_dict(text)

    @staticmethod
    def _resolve_config(config_text: Optional[str] = None, config_dict: Optional[dict] = None) -> str:
        mutually_exclusive(
            config_text=config_text,
            config_dict=config_dict,
            _require_at_least_one=False,
        )
        if config_dict:
            return InputModel._config_dict_to_text(config_dict)

        return config_text

    def set_metadata(self, key: str, value: str, v_type: Optional[str] = None) -> bool:
        """
        Set one metadata entry. All parameters must be strings or castable to strings.

        :param key: Key of the metadata entry.
        :param value: Value of the metadata entry.
        :param v_type: Type of the metadata entry.

        :return: ``True`` if the metadata was set, ``False`` otherwise.
        """
        if not self._base_model:
            self._base_model = self._get_force_base_model()
        self._reload_required = (
            _Model._get_default_session()
            .send(
                models.AddOrUpdateMetadataRequest(
                    metadata=[
                        {
                            "key": str(key),
                            "value": str(value),
                            "type": str(v_type)
                            if str(v_type)
                            in (
                                "float",
                                "int",
                                "bool",
                                "str",
                                "basestring",
                                "list",
                                "tuple",
                                "dict",
                            )
                            else str(None),
                        }
                    ],
                    model=self.id,
                    replace_metadata=False,
                )
            )
            .ok()
        )
        return self._reload_required

    def get_metadata(self, key: str) -> Optional[str]:
        """
        Get one metadata entry value (as a string) based on its key. See ``Model.get_metadata_casted``
        if you wish to cast the value to its type (if possible).

        :param key: Key of the metadata entry you want to get.

        :return: String representation of the value of the metadata entry or ``None`` if the entry was not found
        """
        if not self._base_model:
            self._base_model = self._get_force_base_model()
        self._reload_if_required()
        return self.get_all_metadata().get(str(key), {}).get("value")

    def get_metadata_casted(self, key: str) -> Optional[str]:
        """
        Get one metadata entry based on its key, casted to its type if possible.

        :param key: Key of the metadata entry you want to get.

        :return: The value of the metadata entry, casted to its type (if not possible,
            the string representation will be returned) or ``None`` if the entry was not found
        """
        if not self._base_model:
            self._base_model = self._get_force_base_model()
        key = str(key)
        metadata = self.get_all_metadata()
        if key not in metadata:
            return None
        return cast_basic_type(metadata[key].get("value"), metadata[key].get("type"))

    # the "clearml_" prefix keeps these entries apart from metadata a user stores under their own names
    _serving_metadata_prefix = "clearml_serving_"

    def _set_serving_options(self, **options: Any) -> bool:
        """Store each given serving option as a model metadata entry named ``clearml_serving_<option>``"""
        import json

        ok = True
        for name, value in options.items():
            if value is None:
                continue
            if isinstance(value, str):
                text, v_type = value, "str"
            else:
                text, v_type = json.dumps(value), "dict" if isinstance(value, dict) else "list"
            ok = bool(self.set_metadata(self._serving_metadata_prefix + name, text, v_type=v_type)) and ok
        return ok

    def set_serving_engine(self, engine: str) -> bool:
        """
        Store the ClearML Serving engine this model should be served with: the ``--engine`` option of
        ``clearml-serving model add`` (for example ``sklearn``, ``xgboost``, ``lightgbm`` or ``triton``).

        The serving options are stored with the model (as model metadata), so they do not have to be given
        again when the model is deployed. Call the ``set_serving_...`` methods at the end of training, after
        ``update_weights``. They only store values: nothing is deployed and the model file is not changed.

        :param engine: The serving engine name.
        :return: ``True`` if the value was stored.
        """
        if not engine or not isinstance(engine, str):
            raise ValueError("engine must be a non-empty string, for example 'sklearn'")
        return self._set_serving_options(engine=engine)

    def set_serving_endpoint(self, endpoint: str, version: Optional[str] = None) -> bool:
        """
        Store the endpoint this model should be served at: the ``--endpoint`` and ``--version`` options of
        ``clearml-serving model add``. See ``set_serving_engine``.

        :param endpoint: The endpoint name, for example ``"churn"``.
        :param version: The endpoint version, for example ``"1"``.
        :return: ``True`` if the values were stored.
        """
        if not endpoint or not isinstance(endpoint, str):
            raise ValueError("endpoint must be a non-empty string, for example 'churn'")
        return self._set_serving_options(endpoint=endpoint, version=None if version is None else str(version))

    def set_serving_io(
        self,
        input_size: Optional[Sequence] = None,
        input_type: Optional[Union[str, Sequence[str]]] = None,
        input_name: Optional[Union[str, Sequence[str]]] = None,
        output_size: Optional[Sequence] = None,
        output_type: Optional[Union[str, Sequence[str]]] = None,
        output_name: Optional[Union[str, Sequence[str]]] = None,
    ) -> bool:
        """
        Store the model's input and output tensors: the ``--input-size``, ``--input-type``, ``--input-name``,
        ``--output-size``, ``--output-type`` and ``--output-name`` options of ``clearml-serving model add``,
        which the ``triton`` engine needs. See ``set_serving_engine``.

        For a model with several inputs or outputs, pass a list for each: sizes as a list of lists.

        :param input_size: The input size, for example ``[1, 28, 28]``.
        :param input_type: The input type, for example ``"float32"``.
        :param input_name: The name of the input layer, for example ``"INPUT__0"``.
        :param output_size: The output size, for example ``[-1, 10]``.
        :param output_type: The output type, for example ``"float32"``.
        :param output_name: The name of the output layer, for example ``"OUTPUT__0"``.
        :return: ``True`` if the values were stored.
        """
        options = dict(
            input_size=input_size,
            input_type=input_type,
            input_name=input_name,
            output_size=output_size,
            output_type=output_type,
            output_name=output_name,
        )
        options = {k: (v if isinstance(v, str) else list(v)) for k, v in options.items() if v is not None}
        if not options:
            raise ValueError("set_serving_io needs at least one input or output option")
        return self._set_serving_options(**options)

    def set_serving_aux_config(self, aux_config: Dict[str, Any]) -> bool:
        """
        Store engine specific settings: the ``--aux-config`` option of ``clearml-serving model add``, for
        example ``{"platform": "onnxruntime_onnx"}``. See ``set_serving_engine``.

        :param aux_config: The settings, as a dictionary.
        :return: ``True`` if the value was stored.
        """
        if not isinstance(aux_config, dict) or not aux_config:
            raise ValueError("aux_config must be a non-empty dictionary")
        return self._set_serving_options(aux_config=dict(aux_config))

    def set_serving_input_example(self, input_example: Any) -> bool:
        """
        Store an example of the model's input, so that whoever deploys or calls the model can see what it
        takes. See ``set_serving_engine``.

        The example's shape and type are stored, and its column names if it is a pandas DataFrame. Up to five
        rows of its values are stored too, when they are small.

        :param input_example: One or a few rows of valid input: a numpy array, a pandas DataFrame, a torch tensor
            or a list of rows.
        :return: ``True`` if the value was stored.
        """
        import json

        described = {}
        values = input_example
        if hasattr(values, "columns") and hasattr(values, "to_numpy"):
            described["columns"] = [str(c) for c in values.columns]
            values = values.to_numpy()
        if hasattr(values, "detach"):
            values = values.detach().cpu().numpy()
        if hasattr(values, "shape") and hasattr(values, "dtype"):
            described["shape"] = [int(d) for d in values.shape]
            described["dtype"] = str(values.dtype)
            values = values[:5].tolist()
        elif isinstance(values, (list, tuple)) and values:
            values = list(values[:5])
        else:
            raise ValueError(
                "input_example must be a numpy array, a pandas DataFrame, a torch tensor or a list of rows, "
                "not {}".format(type(input_example).__name__)
            )
        if len(json.dumps(values, default=str)) <= 32 * 1024:
            described["data"] = json.loads(json.dumps(values, default=str))
        return self._set_serving_options(input_example=described)

    def set_serving_preprocess(self, preprocess_code: Union[str, Path]) -> bool:
        """
        Store the model's preprocessing code: the ``--preprocess`` option of ``clearml-serving model add``,
        a ``.py`` file (or a folder) with the ``Preprocess`` class ClearML Serving runs around the model.
        See ``set_serving_engine``.

        The file is uploaded as an artifact of the model's task, and its address is stored with the model.
        The code is run by ClearML Serving when the model is served: store only code you would deploy.

        :param preprocess_code: The path of the local ``.py`` file, or of a folder.
        :return: ``True`` if the code was uploaded and its address stored.
        """
        path = Path(preprocess_code)
        if not path.exists():
            raise ValueError("Preprocessing code '{}' could not be found".format(preprocess_code))
        if not self._task or not hasattr(self._task, "upload_artifact"):
            raise ValueError("set_serving_preprocess needs a model connected to a task, such as an OutputModel")
        if not self._base_model:
            self._base_model = self._get_force_base_model()
        name = "{}preprocess_{}".format(self._serving_metadata_prefix, self.id)
        if not self._task.upload_artifact(name=name, artifact_object=path, wait_on_upload=True):
            return False
        return self._set_serving_options(preprocess=self._task.artifacts[name].url)

    def set_serving_requirements(self, requirements: Union[str, Sequence[str]]) -> bool:
        """
        Store the Python packages the model needs when it is served, for example
        ``["scikit-learn==1.5.2"]``. See ``set_serving_engine``.

        :param requirements: A package specification, or a list of them, as given to ``pip install``.
        :return: ``True`` if the value was stored.
        """
        requirements = [requirements] if isinstance(requirements, str) else list(requirements or [])
        if not requirements or not all(isinstance(r, str) and r.strip() for r in requirements):
            raise ValueError("requirements must be a non-empty list of package specifications")
        return self._set_serving_options(requirements=[r.strip() for r in requirements])

    def get_serving_options(self) -> Dict[str, Any]:
        """
        Get the serving options stored with the model by the ``set_serving_...`` methods.

        :return: A dictionary of option name to value, for example
            ``{"engine": "sklearn", "endpoint": "churn"}``. Empty if none were stored.
        """
        prefix = self._serving_metadata_prefix
        return {
            key[len(prefix):]: value
            for key, value in self.get_all_metadata_casted().items()
            if key.startswith(prefix)
        }

    def get_all_metadata(self) -> Dict[str, Dict[str, str]]:
        """
        Returns all metadata as a ``Dict[key, Dict[value, type]]``,
        where ``key``, ``value``, and ``type`` are all strings.
        To get values cast to their original types (if possible), use ``Model.get_all_metadata_casted``.

        :return: All metadata in ``Dict[key, Dict[value, type]]`` format.
        """
        if not self._base_model:
            self._base_model = self._get_force_base_model()
        self._reload_if_required()
        return self._get_model_data().metadata or {}

    def get_all_metadata_casted(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns all metadata as a ``Dict[key, Dict[value, type]]``,
        where ``key`` and ``type`` are strings, and ``value`` is cast to its original type where possible.
        To get all values as strings, use ``Model.get_all_metadata``.

        :return: All metadata in ``Dict[key, Dict[value, type]]`` format.
        """
        if not self._base_model:
            self._base_model = self._get_force_base_model()
        self._reload_if_required()
        result = {}
        metadata = self.get_all_metadata()
        for key, metadata_entry in metadata.items():
            result[key] = cast_basic_type(metadata_entry.get("value"), metadata_entry.get("type"))
        return result

    def set_all_metadata(self, metadata: Dict[str, Dict[str, str]], replace: bool = True) -> bool:
        """
        Set metadata based on the given parameters. Allows replacing all entries or updating the current entries.

        :param metadata: A dictionary of format ``Dict[key, Dict[value, type]]``
            representing the metadata you want to set.
        :param replace: If ``True``, replace all metadata with the entries in the ``metadata`` parameter.
            If ``False``, keep the old metadata and update it with the entries in the ``metadata`` parameter
            (add or change it).

        :return: ``True`` if the metadata was set and ``False`` otherwise
        """
        if not self._base_model:
            self._base_model = self._get_force_base_model()
        metadata_array = [
            {
                "key": str(k),
                "value": str(v_t.get("value")),
                "type": str(v_t.get("type")),
            }
            for k, v_t in metadata.items()
        ]
        self._reload_required = (
            _Model._get_default_session()
            .send(models.AddOrUpdateMetadataRequest(metadata=metadata_array, model=self.id, replace_metadata=replace))
            .ok()
        )
        return self._reload_required

    def _reload_if_required(self) -> None:
        if not self._reload_required:
            return
        self._get_base_model().reload()
        self._reload_required = False

    def _update_base_model(
        self,
        model_name: Optional[str] = None,
        task_model_entry: Optional[str] = None,
    ) -> _Model:
        if not self._task:
            return self._base_model
        # update the model from the task inputs
        labels = self._task.get_labels_enumeration()
        # noinspection PyProtectedMember
        config_text = self._task._get_model_config_text()
        model_name = (
            model_name or self._name or (self._floating_data.name if self._floating_data else None) or self._task.name
        )
        # noinspection PyBroadException
        try:
            task_model_entry = task_model_entry or self._task_connect_name or Path(self._get_model_data().uri).stem
        except Exception:
            pass
        parent = self._task.input_models_id.get(task_model_entry)
        self._base_model.update(
            labels=(self._floating_data.labels if self._floating_data else None) or labels,
            design=(self._floating_data.design if self._floating_data else None) or config_text,
            task_id=self._task.id,
            project_id=self._task.project,
            parent_id=parent,
            name=model_name,
            comment=self._floating_data.comment if self._floating_data else None,
            tags=self._floating_data.tags if self._floating_data else None,
            framework=self._floating_data.framework if self._floating_data else None,
            upload_storage_uri=self._floating_data.upload_storage_uri if self._floating_data else None,
        )

        # remove model floating change set, by now they should have matched the task.
        self._floating_data = None

        # now we have to update the creator task so it points to us
        if str(self._task.status) not in (
            str(self._task.TaskStatusEnum.created),
            str(self._task.TaskStatusEnum.in_progress),
        ):
            self._log.warning(
                f"Could not update last created model in Task {self._task.id}, "
                f"Task status '{self._task.status}' cannot be updated"
            )
        elif task_model_entry:
            self._base_model.update_for_task(
                task_id=self._task.id,
                model_id=self.id,
                type_="output",
                name=task_model_entry,
            )

        return self._base_model

    def _get_force_base_model(
        self,
        model_name: Optional[str] = None,
        task_model_entry: Optional[str] = None,
    ) -> _Model:
        if self._base_model:
            return self._base_model
        if not self._task:
            return None

        # create a new model from the task
        # noinspection PyProtectedMember
        self._base_model = self._task._get_output_model(model_id=None)
        return self._update_base_model(model_name=model_name, task_model_entry=task_model_entry)

    @property
    def archived_tag(self) -> None:
        return self._archived_tag


class Model(BaseModel):
    """
    A read-only representation of an existing model, looked up by ID.
    Can be connected to a Task to pre-initialize a network.
    When running remotely, the model can be overridden via the UI.
    """

    def __init__(self, model_id: str) -> None:
        """
        :param model_id: The ID (system UUID) of the model.
        """
        super(Model, self).__init__()
        self._base_model_id = model_id
        self._base_model = None

    def get_local_copy(
        self,
        extract_archive: Optional[bool] = None,
        raise_on_error: bool = False,
        force_download: bool = False,
    ) -> str:
        """
        Retrieve a valid link to the model file(s).
        If the model URL is a file system link, it will be returned directly.
        If the model URL points to a remote location (``http``, ``s3``, ``gs``, etc.),
        it will download the file(s) and return the temporary location of the downloaded model.

        :param extract_archive: If ``True``, extract the local copy if possible.
            If ``None`` (default), then extract the downloaded file only if the model is a package.
        :param raise_on_error: If ``True``, raise ``ValueError`` if the artifact download fails.
        :param force_download: If ``True``, re-download model artifact even if a cached copy exists.

        :return: A local path to the model (or a downloaded copy of it).
        """
        if self._is_package():
            return self.get_weights_package(
                return_path=True,
                raise_on_error=raise_on_error,
                force_download=force_download,
                extract_archive=True if extract_archive is None else extract_archive,
            )
        return self.get_weights(
            raise_on_error=raise_on_error,
            force_download=force_download,
            extract_archive=False if extract_archive is None else extract_archive,
        )

    def _get_base_model(self) -> _Model:
        if self._base_model:
            return self._base_model

        if not self._base_model_id:
            # this shouldn't actually happen
            raise Exception("Missing model ID, cannot create an empty model")
        self._base_model = _Model(
            upload_storage_uri=None,
            cache_dir=get_cache_dir(),
            model_id=self._base_model_id,
        )
        return self._base_model

    def _get_model_data(self) -> Any:
        return self._get_base_model().data

    @classmethod
    def query_models(
        cls,
        project_name: Optional[str] = None,
        model_name: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        only_published: bool = False,
        include_archived: bool = False,
        max_results: Optional[int] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> List["Model"]:
        """
        Query the model artifactory based on project name / model name / tags.
        Results are sorted by last updated, most recent first.

        :param project_name: Filter by project name string. If not provided, queries across all projects.
        :param model_name: Filter by model name as shown in the artifactory.
        :param tags: Filter by a list of tags (strings).
            To exclude a tag add "-" prefix to the tag. Example: ``["production", "verified", "-qa"]``.
            The default behaviour is to join all tags with a logical "OR" operator.
            To join all tags with a logical "AND" operator instead, use "__$all" as the first string, for example:

            .. code-block:: py

                ["__$all", "best", "model", "ever"]

            To join all tags with AND, but exclude a tag use "__$not" before the excluded tag, for example:

            .. code-block:: py

                ["__$all", "best", "model", "ever", "__$not", "internal", "__$not", "test"]

            The "OR" and "AND" operators apply to all tags that follow them until another operator is specified.
            The NOT operator applies only to the immediately following tag.
            For example:

            .. code-block:: py

                ["__$all", "a", "b", "c", "__$or", "d", "__$not", "e", "__$and", "__$or", "f", "g"]

            This example means ("a" AND "b" AND "c" AND ("d" OR NOT "e") AND ("f" OR "g")).
            See https://clear.ml/docs/latest/docs/clearml_sdk/model_sdk#tag-filters for details.
        :param only_published: If ``True``, return only published models. Defaults to ``False``.
        :param include_archived: If ``True``, include archived models in results. Defaults to ``False``.
        :param max_results: Maximum number of models to return.
        :param metadata: Filter by metadata key-value pairs.

        :return: List of Model objects
        """
        if project_name:
            # noinspection PyProtectedMember
            res = _Model._get_default_session().send(
                projects.GetAllRequest(
                    name=exact_match_regex(project_name),
                    only_fields=["id", "name", "last_update"],
                )
            )
            project = get_single_result(entity="project", query=project_name, results=res.response.projects)
        else:
            project = None

        only_fields = ["id", "created", "system_tags"]

        extra_fields = {f"metadata.{k}.value": v for k, v in (metadata or {}).items()}

        models_fetched = []

        page = 0
        page_size = 500
        results_left = max_results if max_results is not None else float("inf")
        while True:
            # noinspection PyProtectedMember
            res = _Model._get_default_session().send(
                models.GetAllRequest(
                    project=[project.id] if project else None,
                    name=exact_match_regex(model_name) if model_name is not None else None,
                    only_fields=only_fields,
                    tags=tags or None,
                    system_tags=["-" + cls._archived_tag] if not include_archived else None,
                    ready=True if only_published else None,
                    order_by=["-created"],
                    page=page,
                    page_size=page_size if results_left > page_size else results_left,
                    _allow_extra_fields_=True,
                    **extra_fields,
                )
            )
            if not res.response.models:
                break
            models_fetched.extend(res.response.models)
            results_left -= len(res.response.models)
            if results_left <= 0 or len(res.response.models) < page_size:
                break

            page += 1

        return [Model(model_id=m.id) for m in models_fetched]

    @property
    def id(self) -> str:
        return self._base_model_id if self._base_model_id else super(Model, self).id

    @classmethod
    def remove(
        cls,
        model: Union[str, "Model"],
        delete_weights_file: bool = True,
        force: bool = False,
        raise_on_errors: bool = False,
    ) -> bool:
        """
        Remove a model from the model artifactory, and optionally delete its weights file from remote storage.

        :param model: Model ID or Model object to remove.
        :param delete_weights_file: If ``True`` (default), delete the weights file from the remote storage.
        :param force: If ``True``, remove model even if other Tasks are using this model. Defaults to ``False``.
        :param raise_on_errors: If ``True``, raise ``ValueError`` if something went wrong. Defaults to ``False``.

        :return: ``True`` if model was removed successfully.
            Partial removal returns ``False``, i.e. model was deleted but weights file deletion failed.
        """
        if isinstance(model, str):
            model = Model(model_id=model)

        # noinspection PyBroadException
        try:
            weights_url = model.url
        except Exception:
            if raise_on_errors:
                raise ValueError(f"Could not find model id={model.id}")
            return False

        try:
            # noinspection PyProtectedMember
            res = _Model._get_default_session().send(
                models.DeleteRequest(model.id, force=force),
            )
            response = res.wait()
            if not response.ok():
                if raise_on_errors:
                    raise ValueError(f"Could not remove model id={model.id}: {response.meta}")
                return False
        except SendError as ex:
            if raise_on_errors:
                raise ValueError(f"Could not remove model id={model.id}: {ex}")
            return False
        except ValueError:
            if raise_on_errors:
                raise
            return False
        except Exception as ex:
            if raise_on_errors:
                raise ValueError(f"Could not remove model id={model.id}: {ex}")
            return False

        if not delete_weights_file:
            return True

        helper = StorageHelper.get(url=weights_url)
        try:
            if not helper.delete(weights_url):
                if raise_on_errors:
                    raise ValueError(f"Could not remove model id={model.id} weights file: {weights_url}")
                return False
        except Exception as ex:
            if raise_on_errors:
                raise ValueError(f"Could not remove model id={model.id} weights file '{weights_url}': {ex}")
            return False

        return True


class InputModel(Model):
    """
    Load a read-only model from the model artifactory by ``model_id``, or
    by a combination of ``name``, ``project``, and ``tags``.
    The Model can be connected to a task as an input model and overridden remotely via the UI.
    """

    # noinspection PyProtectedMember
    _EMPTY_MODEL_ID = _Model._EMPTY_MODEL_ID
    _WARNING_CONNECTED_NAMES = {}

    @classmethod
    def import_model(
        cls,
        weights_url: str,
        config_text: Optional[str] = None,
        config_dict: Optional[dict] = None,
        label_enumeration: Optional[Mapping[str, int]] = None,
        name: Optional[str] = None,
        project: Optional[str] = None,
        tags: Optional[List[str]] = None,
        comment: Optional[str] = None,
        is_package: bool = False,
        create_as_published: bool = False,
        framework: Optional[str] = None,
    ) -> "InputModel":
        """
        Create a read-only ``InputModel``  from a pre-trained weights file at a specified URL.

        If the URL is already registered in ClearML,
        the existing model is returned and all other parameters are ignored.

        Once imported, the model can be connected to a Task using ``InputModel.connect`` or ``Task.connect``.

        .. note::
           You can switch input models and re-enqueue Tasks for remote executions via the ClearML WebApp (UI).

        :param weights_url: URL of the weights file. If the URL is already registered in ClearML, the existing model
            is returned and all other parameters are ignored. For example:

          - ``https://domain.com/file.bin``
          - ``s3://bucket/file.bin``
          - ``file:///home/user/file.bin``

        :param config_text: Model configuration as a string. This is usually the content of a configuration
            dictionary file. Specify ``config_text`` or ``config_dict``, but not both.
        :param config_dict: Model configuration as a dictionary. Specify ``config_text`` or ``config_dict``,
            but not both.
        :param label_enumeration: Optional label enumeration dictionary of string (label) to integer (value) pairs.

            For example:

            .. code-block:: javascript

               {
                    "background": 0,
                    "person": 1
               }
        :param name: Name of the imported model.
        :param project: Project to add the model to.
        :param tags: List of tags which describe the model.
        :param comment: A comment / description for the model.
        :param is_package: If ``True``, adds a package tag to the model. Defaults to ``False``.
        :param create_as_published: If ``True``, sets the model status to "Published" immediately. Defaults to ``False``,
            the status will be Draft
        :param framework: The framework of the model

        :return: The imported model or existing model if the URL was already registered.
        """
        config_text = cls._resolve_config(config_text=config_text, config_dict=config_dict)
        weights_url = StorageHelper.conform_url(weights_url)
        if not weights_url:
            raise ValueError("Please provide a valid weights_url parameter")
        # convert local to file to remote one
        weights_url = CacheManager.get_remote_url(weights_url)

        extra = (
            {"system_tags": ["-" + cls._archived_tag]}
            if Session.check_min_api_version("2.3")
            else {"tags": ["-" + cls._archived_tag]}
        )
        # noinspection PyProtectedMember
        result = _Model._get_default_session().send(
            models.GetAllRequest(uri=[weights_url], only_fields=["id", "name", "created"], **extra)
        )

        if result.response.models:
            logger = get_logger()

            logger.info(f'A model with uri "{weights_url}" already exists. Selecting it')

            model = get_single_result(
                entity="model",
                query=weights_url,
                results=result.response.models,
                log=logger,
                raise_on_error=False,
            )

            logger.info(f"Selected model id: {model.id}")

            return InputModel(model_id=model.id)

        base_model = _Model(
            upload_storage_uri=None,
            cache_dir=get_cache_dir(),
        )

        from .task import Task

        task = Task.current_task()
        if task:
            comment = f"Imported by task id: {task.id}" + ("\n" + comment if comment else "")
            project_id = task.project
            name = name or f"Imported by {task.name or ''}"
            # do not register the Task, because we do not want it listed after as "output model",
            # the Task never actually created the Model
            task_id = None
        else:
            project_id = None
            task_id = None

        if project:
            project_id = get_or_create_project(
                session=task.session if task else Task._get_default_session(),
                project_name=project,
            )

        if not framework:
            # noinspection PyProtectedMember
            framework, file_ext = Framework._get_file_ext(framework=framework, filename=weights_url)

        base_model.update(
            design=config_text,
            labels=label_enumeration,
            name=name,
            comment=comment,
            tags=tags,
            uri=weights_url,
            framework=framework,
            project_id=project_id,
            task_id=task_id,
        )

        this_model = InputModel(model_id=base_model.id)
        this_model._base_model = base_model

        if is_package:
            this_model._set_package_tag()

        if create_as_published:
            this_model.publish()

        return this_model

    @classmethod
    def load_model(cls, weights_url: str, load_archived: bool = False) -> "InputModel":
        """
        Retrieve a model registered in ClearML by its weights file URL. Returns ``None`` if no matching model is found.

        :param weights_url: URL of the weights file. Examples:

          - ``"https://domain.com/file.bin"``
          - ``"s3://bucket/file.bin"``
          - ``"file:///home/user/file.bin"``

        :param load_archived: If ``True``, include archived models in the search. Defaults to  ``False``.

        :return:  The matching ``InputModel`` object, or ``None`` if no model could be found.
        """
        weights_url = StorageHelper.conform_url(weights_url)
        if not weights_url:
            raise ValueError("Please provide a valid weights_url parameter")

        # convert local to file to remote one
        weights_url = CacheManager.get_remote_url(weights_url)

        if not load_archived:
            # noinspection PyTypeChecker
            extra = (
                {"system_tags": ["-" + _Task.archived_tag]}
                if Session.check_min_api_version("2.3")
                else {"tags": ["-" + cls._archived_tag]}
            )
        else:
            extra = {}

        # noinspection PyProtectedMember
        result = _Model._get_default_session().send(
            models.GetAllRequest(uri=[weights_url], only_fields=["id", "name", "created"], **extra)
        )

        if not result or not result.response or not result.response.models:
            return None

        logger = get_logger()
        model = get_single_result(
            entity="model",
            query=weights_url,
            results=result.response.models,
            log=logger,
            raise_on_error=False,
        )

        return InputModel(model_id=model.id)

    @classmethod
    def empty(
        cls,
        config_text: Optional[str] = None,
        config_dict: Optional[dict] = None,
        label_enumeration: Optional[Mapping[str, int]] = None,
    ) -> "InputModel":
        """
        Create an empty model object. Later, you can assign a model to the empty model object.

        :param config_text: The model configuration as a string.
            This is usually the content of a configuration dictionary file.
            Specify ``config_text`` or ``config_dict``, but not both.
        :param config_dict: The model configuration as a dictionary.
            Specify ``config_text`` or ``config_dict``, but not both.
        :param label_enumeration: The label enumeration dictionary of string (label) to integer (value) pairs.
            For example:

            .. code-block:: javascript

               {
                    "background": 0,
                    "person": 1
               }

        :return: An empty model object.
        """
        design = cls._resolve_config(config_text=config_text, config_dict=config_dict)

        this_model = InputModel(model_id=cls._EMPTY_MODEL_ID)
        this_model._base_model = m = _Model(
            cache_dir=None,
            upload_storage_uri=None,
            model_id=cls._EMPTY_MODEL_ID,
        )
        # noinspection PyProtectedMember
        m._data.design = _Model._wrap_design(design)
        # noinspection PyProtectedMember
        m._data.labels = label_enumeration
        return this_model

    def __init__(
        self,
        model_id: Optional[str] = None,
        name: Optional[str] = None,
        project: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        only_published: bool = False,
    ) -> None:
        """
        Load a model from the Model artifactory by ``model_id`` (UUID),
        or by a combination of ``name``, ``project``, and ``tags``.

        :param model_id: The ClearML ID (system UUID) of the input model whose metadata the **ClearML Server**
            (backend) stores. If provided, all other arguments are ignored.
        :param name: Model name to search and load.
        :param project: Model project name to search model in.
        :param tags: Model tags list to filter by.
        :param only_published: If ``True``, filter out non-published (draft) models.
        """
        if not model_id:
            found_models = self.query_models(
                project_name=project,
                model_name=name,
                tags=tags,
                only_published=only_published,
            )
            if not found_models:
                raise ValueError(
                    f"Could not locate model with project={project} name={name} tags={tags} published={only_published}"
                )
            model_id = found_models[0].id
        super(InputModel, self).__init__(model_id)

    @property
    def id(self) -> str:
        return self._base_model_id

    def connect(
        self,
        task: "Task",
        name: Optional[str] = None,
        ignore_remote_overrides: bool = False,
    ) -> None:
        """
        Connect a preexisting model to a Task object. Preexisting models include:

        - Imported models (``InputModel`` objects created using the ``Logger.import_model`` method).
        - Models already in the ClearML platform, instantiated from the ``InputModel`` class using a ClearML ID.
        - Models from external frameworks (e.g. TensorFlow) used to create an ``InputModel`` object.

        When the experiment is executed remotely in a worker, the input model specified in the experiment UI/backend
        is used, unless ``ignore_remote_overrides`` is set to ``True``.

        .. note::
           You can switch input models and re-enqueue Tasks for remote executions via the ClearML WebApp (UI).

        :param task: The Task object to connect to.
        :param ignore_remote_overrides: If ``True``, UI/backend model changes are ignored during remote execution.
            Defaults to ``False``, meaning that any changes made in the UI/backend will be applied in remote execution.
        :param name: The name under which this model appears on the Task.
            Defaults to the weights filename, or "Input Model" if unavailable.
        """
        self._set_task(task)
        name = name or InputModel._get_connect_name(self)
        InputModel._warn_on_same_name_connect(name)
        ignore_remote_overrides = task._handle_ignore_remote_overrides(
            name + "/_ignore_remote_overrides_input_model_", ignore_remote_overrides
        )

        model_id = None
        # noinspection PyProtectedMember
        if running_remotely() and (task.is_main_task() or task._is_remote_main_task()) and not ignore_remote_overrides:
            input_models = task.input_models_id
            # noinspection PyBroadException
            try:
                # TODO: (temp fix) At the moment, the UI changes the key of the model hparam
                # when modifying its value... There is no way to tell which model was changed
                # so just take the first one in case `name` is not in `input_models`
                model_id = input_models.get(name, next(iter(input_models.values())))
                self._base_model_id = model_id
                self._base_model = InputModel(model_id=model_id)._get_base_model()
            except Exception:
                model_id = None

        if not model_id:
            # we should set the task input model to point to us
            model = self._get_base_model()
            # try to store the input model id, if it is not empty
            # (Empty Should not happen)
            if model.id != self._EMPTY_MODEL_ID:
                task.set_input_model(model_id=model.id, name=name)
            # only copy the model design if the task has no design to begin with
            # noinspection PyProtectedMember
            if not self._task._get_model_config_text() and model.model_design:
                # noinspection PyProtectedMember
                task._set_model_config(config_text=model.model_design)
            if not self._task.get_labels_enumeration() and model.data.labels:
                task.set_model_label_enumeration(model.data.labels)

    @classmethod
    def _warn_on_same_name_connect(cls, name: str) -> None:
        if name not in cls._WARNING_CONNECTED_NAMES:
            cls._WARNING_CONNECTED_NAMES[name] = False
            return
        if cls._WARNING_CONNECTED_NAMES[name]:
            return
        get_logger().warning(
            f"Connecting multiple input models with the same name: `{name}`. This might result in the wrong model being used when executing remotely"
        )
        cls._WARNING_CONNECTED_NAMES[name] = True

    @staticmethod
    def _get_connect_name(model: Optional[Any]) -> str:
        default_name = "Input Model"
        if model is None:
            return default_name
        # noinspection PyBroadException
        try:
            model_uri = getattr(model, "url", getattr(model, "uri", None))
            return Path(model_uri).stem
        except Exception:
            return default_name


class OutputModel(BaseModel):
    """
    Create an output model for a Task (experiment) to store the training results.

    The model is read-write and automatically registered as the Task's output model.

    A common use case is to reuse the OutputModel object, and override the weights after storing a model snapshot.
    Another use case is to create multiple OutputModel objects for a Task, and after a new high score
    is found, store a model snapshot.

    If the model configuration or label enumeration is not provided, values are inherited from the Task's input model.

    .. note::
       When executing a Task remotely with a clearml-agent, you can modify the model configuration and/or model's
       label enumeration using the **ClearML WebApp**.
    """

    _default_output_uri = None
    _offline_folder = "models"

    @property
    def published(self) -> bool:
        """
        Get the published state of this model.

        :return: ``True`` if the model is published, ``False`` otherwise.

        """
        if not self.id:
            return False
        return self._get_base_model().locked

    @property
    def config_text(self) -> str:
        """
        Get the configuration as a string. For example, prototxt, a ``.ini`` file, or Python code to evaluate.

        :return: The configuration.
        """
        # noinspection PyProtectedMember
        return _Model._unwrap_design(self._get_model_data().design)

    @config_text.setter
    def config_text(self, value: str) -> None:
        """
        Set the configuration. Store a blob of text for custom usage.
        """
        self.update_design(config_text=value)

    @property
    def config_dict(self) -> dict:
        """
        Get the configuration as a dictionary parsed from the ``config_text`` text. This usually represents the model
        configuration. For example, from prototxt to ``.ini`` file or Python code to evaluate.

        :return: The configuration.
        """
        return self._text_to_config_dict(self.config_text)

    @config_dict.setter
    def config_dict(self, value: dict) -> None:
        """
        Set the configuration. Saved in the model object.

        :param value: The configuration parameters.
        """
        self.update_design(config_dict=value)

    @property
    def labels(self) -> Dict[str, int]:
        """
        Get the label enumeration as a dictionary of string (label) to integer (value) pairs.

        For example:

        .. code-block:: javascript

           {
                "background": 0,
                "person": 1
           }

        :return: The label enumeration.
        """
        return self._get_model_data().labels

    @labels.setter
    def labels(self, value: Mapping[str, int]) -> None:
        """
        Set the label enumeration.

        :param value: The label enumeration dictionary of string (label) to integer (value) pairs.

            For example:

            .. code-block:: javascript

               {
                    "background": 0,
                    "person": 1
               }

        """
        self.update_labels(labels=value)

    @property
    def upload_storage_uri(self) -> str:
        """
        The URI of the storage destination for uploaded model weight files.

        :return: The URI string
        """
        return self._get_base_model().upload_storage_uri

    @property
    def id(self) -> str:
        from clearml import Task as OfflineTask

        if OfflineTask.is_offline():
            if not self._base_model_id:
                random_uuid = str(uuid4()).replace("-", "")
                self._base_model_id = f"offline-{random_uuid}"
            return self._base_model_id
        return super(OutputModel, self).id

    def __init__(
        self,
        task: Optional["Task"] = None,
        config_text: Optional[str] = None,
        config_dict: Optional[dict] = None,
        label_enumeration: Optional[Mapping[str, int]] = None,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        comment: Optional[str] = None,
        framework: Optional[Union[str, Framework]] = None,
        base_model_id: Optional[str] = None,
    ) -> None:
        """
        Create a new model and immediately connect it to a task.

        We do not allow for Model creation without a task, so we always keep track on how we created the models.
        In remote execution, Model parameters can be overridden by the Task
        (such as model configuration & label enumerator).

        :param task: The Task object with which the OutputModel object is associated.
        :param config_text: The configuration as a string.
            This is usually the content of a configuration dictionary file.
            Specify ``config_text`` or ``config_dict``, but not both.
        :param config_dict: The configuration as a dictionary. Specify ``config_dict`` or ``config_text``, but not both.
        :param label_enumeration: The label enumeration dictionary of string (label) to integer (value)
            pairs.

            For example:

            .. code-block:: javascript

               {
                    "background": 0,
                    "person": 1
               }

        :param name: The name for the newly created model.
        :param tags: A list of strings which are tags for the model.
        :param comment: A comment / description for the model.
        :param framework: The framework of the model or a Framework object.
        :param base_model_id: Model ID to be reused.
        """
        if not task:
            from .task import Task

            task = Task.current_task()
            if not task:
                raise ValueError("task object was not provided, and no current task was found")

        super(OutputModel, self).__init__(task=task)

        config_text = self._resolve_config(config_text=config_text, config_dict=config_dict)

        self._model_local_filename = None
        self._last_uploaded_url = None
        self._base_model = None
        self._base_model_id = None
        self._task_connect_name = None
        self._name = name
        self._label_enumeration = label_enumeration
        # noinspection PyProtectedMember
        action_comment = (
            f"Created by task id: {task.id}"
            if not base_model_id
            else f"Overwritten by task id: {task.id}"
        )
        self._floating_data = create_dummy_model(
            design=_Model._wrap_design(config_text),
            labels=label_enumeration or task.get_labels_enumeration(),
            name=name or self._task.name,
            tags=tags,
            comment=(
                f"{action_comment}\n{comment}"
                if comment
                else action_comment
            ),
            framework=framework,
            upload_storage_uri=task.output_uri,
        )
        # If we have no real model ID, we are done
        if not base_model_id:
            return

        # noinspection PyBroadException
        try:
            # noinspection PyProtectedMember
            _base_model = self._task._get_output_model(model_id=base_model_id)
            _base_model.update(
                labels=self._floating_data.labels,
                design=self._floating_data.design,
                task_id=self._task.id,
                project_id=self._task.project,
                name=self._floating_data.name or self._task.name,
                comment=(
                    f"{_base_model.comment}\n{self._floating_data.comment}"
                    if (
                        _base_model.comment
                        and self._floating_data.comment
                        and self._floating_data.comment not in _base_model.comment
                    )
                    else (_base_model.comment or self._floating_data.comment)
                ),
                tags=self._floating_data.tags,
                framework=self._floating_data.framework,
                upload_storage_uri=self._floating_data.upload_storage_uri,
            )
            self._base_model = _base_model
            self._floating_data = None
            name = self._task_connect_name or Path(_base_model.uri).stem
        except Exception:
            pass
        self.connect(task, name=name)

    def connect(self, task: "Task", name: Optional[str] = None, **kwargs: Any) -> None:
        """
        Connect a preexisting model to a Task object. Preexisting models include:

        - Imported models.
        - Models already in the ClearML platform
        - Models from external frameworks (e.g. TensorFlow)

        :param task: A Task object.
        :param name: The model name as it would appear on the Task object.
            The model's own name can differ, which is useful when a single Task uses multiple models.
        """
        if self._task != task:
            raise ValueError("Can only connect preexisting model to task, but this is a fresh model")

        if name:
            self._task_connect_name = name

        # we should set the task input model to point to us
        model = self._get_base_model()

        # only copy the model design if the task has no design to begin with
        # noinspection PyProtectedMember
        if not self._task._get_model_config_text():
            # noinspection PyProtectedMember
            task._set_model_config(
                config_text=model.model_design if hasattr(model, "model_design") else model.design.get("design", "")
            )
        if not self._task.get_labels_enumeration():
            task.set_model_label_enumeration(model.data.labels if hasattr(model, "data") else model.labels)

        if self._base_model:
            self._base_model.update_for_task(
                task_id=self._task.id,
                model_id=self.id,
                type_="output",
                name=self._task_connect_name,
            )

    def set_upload_destination(self, uri: str) -> None:
        """
        Set the URI of the storage destination for uploaded model weight files.
        Supported storage destinations include S3, Google Cloud Storage, and file locations.

        Using this method, file uploads are separate and then a link to each is stored in the model object.

        .. note::
           For storage requiring credentials, the credentials are stored in the ClearML configuration file,
           ```~/clearml.conf```.

        :param uri: The URI of the upload storage destination.

            For example:

            - ``s3://bucket/directory/``
            - ``file:///tmp/debug/``
        """
        if not uri:
            return

        # Test if we can update the model.
        self._validate_update()

        # Create the storage helper
        storage = StorageHelper.get(uri)

        # Verify that we can upload to this destination
        try:
            uri = storage.verify_upload(folder_uri=uri)
        except Exception:
            raise ValueError(f"Could not set destination uri to: {uri} [Check write permissions]")

        # store default uri
        self._get_base_model().upload_storage_uri = uri

    def update_weights(
        self,
        weights_filename: Optional[str] = None,
        upload_uri: Optional[str] = None,
        target_filename: Optional[str] = None,
        auto_delete_file: bool = True,
        register_uri: Optional[str] = None,
        iteration: Optional[int] = None,
        update_comment: bool = True,
        is_package: bool = False,
        async_enable: bool = True,
    ) -> str:
        """
        Update the model weights from a local file.

        .. note::
           Uploading the model is a background process. This method returns immediately.

        :param weights_filename: The name of the locally stored weights file to upload.
            Specify ``weights_filename`` or ``register_uri``, but not both.
        :param upload_uri: The URI of the storage destination for model weights upload. The default value
            is the previously used URI.
        :param target_filename: The newly created filename in the storage destination location. The default value
            is the ``weights_filename`` value.
        :param auto_delete_file: If ``True`` (default), delete the temporary file after uploading.
        :param register_uri: The URI of an already uploaded weights file. The URI must be valid. Specify
            ``register_uri`` or ``weights_filename``, but not both.
        :param iteration: The iteration number.
        :param update_comment: If ``True`` (default), append the local weights filename to the model comment
            (to maintain provenance).
        :param is_package: If ``True``, mark the weights file as a compressed package, usually a zip file. Defaults to ``False``.
        :param async_enable: If ``True`` (default), upload in the background and return immediately. If ``False``, block
            until the upload completes. Raises an error if the upload fails.
        :return: The uploaded URI.
        """

        def delete_previous_weights_file(filename: str = weights_filename) -> None:
            try:
                if filename:
                    os.remove(filename)
            except OSError:
                self._log.debug(f"Failed removing temporary file {filename}")

        # test if we can update the model
        if self.id and self.published:
            raise ValueError("Model is published and cannot be changed")

        if (not weights_filename and not register_uri) or (weights_filename and register_uri):
            raise ValueError(
                "Model update must have either local weights file to upload, "
                "or pre-uploaded register_uri, never both"
            )

        # only upload if we are connected to a task
        if not self._task:
            raise Exception("Missing a task for this model")

        if self._task.is_offline() and (weights_filename is None or not Path(weights_filename).is_dir()):
            return self._update_weights_offline(
                weights_filename=weights_filename,
                upload_uri=upload_uri,
                target_filename=target_filename,
                register_uri=register_uri,
                iteration=iteration,
                update_comment=update_comment,
                is_package=is_package,
            )

        if weights_filename is not None:
            # Check if weights_filename is a folder, is package upload
            if Path(weights_filename).is_dir():
                return self.update_weights_package(
                    weights_path=weights_filename,
                    upload_uri=upload_uri,
                    target_filename=target_filename or Path(weights_filename).name,
                    auto_delete_file=auto_delete_file,
                    iteration=iteration,
                    async_enable=async_enable,
                )

            # make sure we delete the previous file, if it exists
            if self._model_local_filename != weights_filename:
                delete_previous_weights_file(self._model_local_filename)
            # store temp filename for deletion next time, if needed
            if auto_delete_file:
                self._model_local_filename = weights_filename

        # make sure the created model is updated:
        out_model_file_name = target_filename or weights_filename or register_uri

        # prefer self._task_connect_name if exists
        if self._task_connect_name:
            name = self._task_connect_name
        elif out_model_file_name:
            name = Path(out_model_file_name).stem
        else:
            name = "Output Model"

        if not self._base_model:
            model = self._get_force_base_model(task_model_entry=name)
        else:
            self._update_base_model(task_model_entry=name)
            model = self._base_model
        if not model:
            raise ValueError("Failed creating internal output model")

        # select the correct file extension based on the framework,
        # or update the framework based on the file extension
        # noinspection PyProtectedMember
        framework, file_ext = Framework._get_file_ext(
            framework=self._get_model_data().framework,
            filename=target_filename or weights_filename or register_uri,
        )

        if weights_filename:
            target_filename = target_filename or Path(weights_filename).name
            if not target_filename.lower().endswith(file_ext):
                target_filename += file_ext

        # set target uri for upload (if specified)
        if upload_uri:
            self.set_upload_destination(upload_uri)

        # let us know the iteration number, we put it in the comment section for now.
        iteration_msg = f"snapshot {weights_filename or register_uri} stored"
        line_break = "" if (self.comment or "").startswith("\n") else "\n"
        comment = (
            (
                f"{iteration_msg}{line_break}{self.comment}"
                if self.comment
                else f"{iteration_msg}{line_break}"
            )
            if update_comment
            else None
        )

        # if we have no output destination, just register the local model file
        if weights_filename and not self.upload_storage_uri and not self._task.storage_uri:
            register_uri = weights_filename
            weights_filename = None
            auto_delete_file = False
            self._log.info(f"No output storage destination defined, registering local model {register_uri}")

        # start the upload
        if weights_filename:
            if not model.upload_storage_uri:
                self.set_upload_destination(self.upload_storage_uri or self._task.storage_uri)

            output_uri = model.update_and_upload(
                model_file=weights_filename,
                task_id=self._task.id,
                async_enable=async_enable,
                target_filename=target_filename,
                framework=self.framework or framework,
                comment=comment,
                cb=delete_previous_weights_file if auto_delete_file else None,
                iteration=iteration or self._task.get_last_iteration(),
            )
        elif register_uri:
            register_uri = StorageHelper.conform_url(register_uri)
            output_uri = model.update(
                uri=register_uri,
                task_id=self._task.id,
                framework=framework,
                comment=comment,
            )
        else:
            output_uri = None

        self._last_uploaded_url = output_uri

        if is_package:
            self._set_package_tag()

        return output_uri

    def update_weights_package(
        self,
        weights_filenames: Optional[Sequence[str]] = None,
        weights_path: Optional[str] = None,
        upload_uri: Optional[str] = None,
        target_filename: Optional[str] = None,
        auto_delete_file: bool = True,
        iteration: Optional[int] = None,
        async_enable: bool = True,
    ) -> str:
        """
        Update the model weights from a local file, or from directory containing multiple files.

        .. note::
           Uploading the model is a background process. This method returns immediately.

        :param weights_filenames: The file names of the locally stored model files. Specify ``weights_filenames``,
            or ``weights_path``, but not both.
        :param weights_path: The directory path to a package. All the files in the directory will be uploaded.
            Specify ``weights_path`` or ``weights_filenames``, but not both.
        :param upload_uri: The URI of the storage destination for the model weights upload. The default
            is the previously used URI.
        :param target_filename: The newly created filename in the storage destination URI location. The default
            is the value specified in the ``weights_filename`` parameter.
        :param auto_delete_file: If ``True`` (default), delete temporary file after uploading.
        :param iteration: The iteration number.
        :param async_enable: Whether to upload model in background or to block.
            Will raise an error in the main thread if the weights failed to be uploaded or not.

        :return: The uploaded URI for the weights package.
        """
        # create list of files
        if (not weights_filenames and not weights_path) or (weights_filenames and weights_path):
            raise ValueError("Model update weights package should get either directory path to pack or a list of files")

        if not weights_filenames:
            weights_filenames = list(map(str, Path(weights_path).rglob("*")))
        elif weights_filenames and len(weights_filenames) > 1:
            weights_path = get_common_path(weights_filenames)

        # create packed model from all the files
        fd, zip_file = mkstemp(prefix="model_package.", suffix=".zip")
        try:
            with zipfile.ZipFile(zip_file, "w", allowZip64=True, compression=zipfile.ZIP_STORED) as zf:
                for filename in weights_filenames:
                    relative_file_name = (
                        Path(filename).name
                        if not weights_path
                        else Path(filename).absolute().relative_to(Path(weights_path).absolute()).as_posix()
                    )
                    zf.write(filename, arcname=relative_file_name)
        finally:
            os.close(fd)

        # now we can delete the files (or path if provided)
        if auto_delete_file:

            def safe_remove(path: str, is_dir: bool = False) -> None:
                try:
                    (os.rmdir if is_dir else os.remove)(path)
                except OSError:
                    self._log.info(f"Failed removing temporary {path}")

            for filename in weights_filenames:
                safe_remove(filename)
            if weights_path:
                safe_remove(weights_path, is_dir=True)

        if target_filename and not target_filename.lower().endswith(".zip"):
            target_filename += ".zip"

        # and now we should upload the file, always delete the temporary zip file
        iteration_msg = f"snapshot {weights_filenames} stored"
        line_break = "" if (self.comment or "").startswith("\n") else "\n"
        self.comment = (
            f"{iteration_msg}{line_break}{self.comment}"
            if self.comment
            else f"{iteration_msg}{line_break}"
        )
        uploaded_uri = self.update_weights(
            weights_filename=zip_file,
            auto_delete_file=True,
            upload_uri=upload_uri,
            target_filename=target_filename or "model_package.zip",
            iteration=iteration,
            update_comment=False,
            async_enable=async_enable,
        )
        # set the model tag (by now we should have a model object) so we know we have packaged file
        self._set_package_tag()
        return uploaded_uri

    def update_design(
        self,
        config_text: Optional[str] = None,
        config_dict: Optional[dict] = None,
    ) -> bool:
        """
        Update the model configuration. Store a blob of text for custom usage.

        .. note::
           This method's behavior is lazy. The design update is only forced when the weights
           are updated.

        :param config_text: The configuration as a string.
            This is usually the content of a configuration dictionary file.
            Specify ``config_text`` or ``config_dict``, but not both.
        :param config_dict: The configuration as a dictionary. Specify ``config_text`` or ``config_dict``, but not both.

        :return: ``True`` if the update was successful, ``False`` if it was not.
        """
        if not self._validate_update():
            return False

        config_text = self._resolve_config(config_text=config_text, config_dict=config_dict)

        if self._task and not self._task.get_model_config_text():
            self._task.set_model_config(config_text=config_text)

        if self.id:
            # update the model object (this will happen if we resumed a training task)
            result = self._get_force_base_model().edit(design=config_text)
        else:
            # noinspection PyProtectedMember
            self._floating_data.design = _Model._wrap_design(config_text)
            result = Waitable()

        # you can wait on this object
        return result

    def update_labels(self, labels: Mapping[str, int]) -> Any:
        """
        Update the label enumeration.

        :param labels: The label enumeration dictionary of string (label) to integer (value) pairs.

            For example:

            .. code-block:: javascript

               {
                    "background": 0,
                    "person": 1
               }

        """
        validate_dict(
            labels,
            key_types=(str,),
            value_types=(int,),
            desc="label enumeration",
        )

        if not self._validate_update():
            return

        if self._task:
            self._task.set_model_label_enumeration(labels)

        if self.id:
            # update the model object (this will happen if we resumed a training task)
            result = self._get_force_base_model().edit(labels=labels)
        else:
            self._floating_data.labels = labels
            result = Waitable()

        # you can wait on this object
        return result

    @classmethod
    def wait_for_uploads(
        cls,
        timeout: Optional[float] = None,
        max_num_uploads: Optional[int] = None,
    ) -> None:
        """
        Wait for any pending or in-progress model uploads to complete. If no uploads are pending or in-progress,
        then the ``wait_for_uploads`` returns immediately.

        :param timeout: The timeout interval to wait for uploads (seconds).
        :param max_num_uploads: The maximum number of uploads to wait for.
        """
        _Model.wait_for_results(timeout=timeout, max_num_uploads=max_num_uploads)

    @classmethod
    def set_default_upload_uri(cls, output_uri: Optional[str]) -> None:
        """
        Set the default upload URI for all OutputModels.

        :param output_uri: URL for uploading models. Examples:

          - ``https://demofiles.demo.clear.ml``
          - ``s3://bucket/``
          - ``gs://bucket/``
          - ``azure://bucket/``
          - ``file:///mnt/shared/nfs``
        """
        cls._default_output_uri = str(output_uri) if output_uri else None

    def _update_weights_offline(
        self,
        weights_filename: Optional[str] = None,
        upload_uri: Optional[str] = None,
        target_filename: Optional[str] = None,
        register_uri: Optional[str] = None,
        iteration: Optional[int] = None,
        update_comment: bool = True,
        is_package: bool = False,
    ) -> str:
        if (not weights_filename and not register_uri) or (weights_filename and register_uri):
            raise ValueError(
                "Model update must have either local weights file to upload, "
                "or pre-uploaded register_uri, never both"
            )
        if not self._task:
            raise Exception("Missing a task for this model")
        weights_filename_offline = None
        if weights_filename:
            weights_filename_offline = (
                self._task.get_offline_mode_folder() / self._offline_folder / Path(weights_filename).name
            ).as_posix()
            os.makedirs(os.path.dirname(weights_filename_offline), exist_ok=True)
            shutil.copyfile(weights_filename, weights_filename_offline)
        # noinspection PyProtectedMember
        self._task._offline_output_models.append(
            dict(
                init=dict(
                    config_text=self.config_text,
                    config_dict=self.config_dict,
                    label_enumeration=self._label_enumeration,
                    name=self.name,
                    tags=self.tags,
                    comment=self.comment,
                    framework=self.framework,
                ),
                weights=dict(
                    weights_filename=weights_filename_offline,
                    upload_uri=upload_uri,
                    target_filename=target_filename,
                    register_uri=register_uri,
                    iteration=iteration,
                    update_comment=update_comment,
                    is_package=is_package,
                ),
                output_uri=self._get_base_model().upload_storage_uri or self._default_output_uri,
                id=self.id,
            )
        )
        return weights_filename_offline or register_uri

    def _get_base_model(self) -> Union[_Model, None]:
        if self._floating_data:
            return self._floating_data
        return self._get_force_base_model()

    def _get_model_data(self) -> Any:
        if self._base_model:
            return self._base_model.data
        return self._floating_data

    def _validate_update(self) -> bool:
        # test if we can update the model
        if self.id and self.published:
            raise ValueError("Model is published and cannot be changed")

        return True

    def _get_last_uploaded_filename(self) -> Optional[str]:
        if not self._last_uploaded_url and not self.url:
            return None
        return Path(self._last_uploaded_url or self.url).name


class Waitable:
    def wait(self, *_: Any, **__: Any) -> bool:
        return True
