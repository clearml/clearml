import numpy
import copy
import json
import os
import shutil
import sys
import threading
import time
import warnings
from argparse import ArgumentParser
from logging import getLogger
from tempfile import mkstemp, mkdtemp
from zipfile import ZipFile, ZIP_DEFLATED

try:
    # noinspection PyCompatibility
    from collections.abc import Sequence as CollectionsSequence
except ImportError:
    from collections import Sequence as CollectionsSequence  # noqa

from typing import (
    Optional,
    Union,
    Mapping,
    Sequence,
    Dict,
    Iterable,
    TYPE_CHECKING,
    Callable,
    Tuple,
    List,
    TypeVar,
    Any,
)

import psutil
import six
from pathlib2 import Path

from .backend_config.defs import get_active_config_file, get_config_file
from .backend_api.services import tasks, projects, events, queues
from .backend_api.session.session import (
    Session,
    ENV_ACCESS_KEY,
    ENV_SECRET_KEY,
    ENV_HOST,
    ENV_WEB_HOST,
    ENV_FILES_HOST,
)
from .backend_api.session.defs import (
    ENV_DEFERRED_TASK_INIT,
    ENV_IGNORE_MISSING_CONFIG,
    ENV_OFFLINE_MODE,
    MissingConfigError,
)
from .backend_interface.metrics import Metrics
from .backend_interface.model import Model as BackendModel
from .backend_interface.base import InterfaceBase
from .backend_interface.task import Task as _Task
from .backend_interface.task.log import TaskHandler
from .backend_interface.task.development.worker import DevWorker
from .backend_interface.task.repo import ScriptInfo
from .backend_interface.task.models import TaskModels
from .backend_interface.util import (
    get_single_result,
    exact_match_regex,
    make_message,
    mutually_exclusive,
    get_queue_id,
    get_num_enqueued_tasks,
    get_or_create_project,
)
from .binding.absl_bind import PatchAbsl
from .binding.artifacts import Artifacts, Artifact
from .binding.environ_bind import EnvironmentBind, PatchOsFork
from .binding.frameworks.fastai_bind import PatchFastai
from .binding.frameworks.lightgbm_bind import PatchLIGHTgbmModelIO
from .binding.frameworks.pytorch_bind import PatchPyTorchModelIO
from .binding.frameworks.tensorflow_bind import TensorflowBinding
from .binding.frameworks.xgboost_bind import PatchXGBoostModelIO
from .binding.frameworks.catboost_bind import PatchCatBoostModelIO
from .binding.frameworks.megengine_bind import PatchMegEngineModelIO
from .binding.joblib_bind import PatchedJoblib
from .binding.matplotlib_bind import PatchedMatplotlib
from .binding.hydra_bind import PatchHydra
from .binding.click_bind import PatchClick
from .binding.fire_bind import PatchFire
from .binding.jsonargs_bind import PatchJsonArgParse
from .binding.gradio_bind import PatchGradio
from .binding.frameworks import WeightsFileHandler
from .config import (
    config,
    DEV_TASK_NO_REUSE,
    get_is_master_node,
    DEBUG_SIMULATE_REMOTE_TASK,
    DEV_DEFAULT_OUTPUT_URI,
    deferred_config,
    TASK_SET_ITERATION_OFFSET,
    HOST_MACHINE_IP,
)
from .config import running_remotely, get_remote_task_id
from .config.cache import SessionCache
from .debugging.log import LoggerRoot
from .errors import UsageError
from .logger import Logger
from .model import Model, InputModel, OutputModel, Framework
from .task_parameters import TaskParameters
from .utilities.config import verify_basic_value
from .binding.args import (
    argparser_parseargs_called,
    get_argparser_last_args,
    argparser_update_currenttask,
)
from .utilities.dicts import ReadOnlyDict, merge_dicts, RequirementsDict
from .utilities.proxy_object import (
    ProxyDictPreWrite,
    ProxyDictPostWrite,
    flatten_dictionary,
    nested_from_flat_dictionary,
    naive_nested_from_flat_dictionary,
    StubObject as _TaskStub,
)
from .utilities.resource_monitor import ResourceMonitor
from .utilities.seed import make_deterministic
from .utilities.lowlevel.threads import get_current_thread_id
from .utilities.lowlevel.distributed import (
    get_torch_local_rank,
    get_torch_distributed_anchor_task_id,
    create_torch_distributed_anchor,
)
from .utilities.process.mp import BackgroundMonitor, leave_process
from .utilities.process.exit_hooks import ExitHooks
from .utilities.matching import matches_any_wildcard
from .utilities.parallel import FutureTaskCaller
from .utilities.networking import get_private_ip

# noinspection PyProtectedMember
from .backend_interface.task.args import _Arguments

if TYPE_CHECKING:
    import pandas
    from PIL import Image
    from .router.router import HttpRouter  # noqa: F401

# Forward declaration to help linters
TaskInstance = TypeVar("TaskInstance", bound="Task")


class Task(_Task):
    """
    The ``Task`` class is a code template for a Task object which, together with its connected experiment components,
    represents the current running experiment. These connected components include hyperparameters, loggers,
    configuration, label enumeration, models, and other artifacts.

    The term "main execution Task" refers to the Task context for current running experiment. Python experiment scripts
    can create one, and only one, main execution Task. It is traceable, and after a script runs and ClearML stores
    the Task in the **ClearML Server** (backend), it is modifiable, reproducible, executable by a worker, and you
    can duplicate it for further experimentation.

    The ``Task`` class and its methods allow you to create and manage experiments, as well as perform
    advanced experimentation functions, such as autoML.

    .. warning::
        Do not construct Task objects directly. Use one of the methods listed below to create experiments or
        reference existing experiments.
        Do not define `CLEARML_TASK_*` and `CLEARML_PROC_*` OS environments, they are used internally
        for bookkeeping between processes and agents.

    For detailed information about creating Task objects, see the following methods:

    - Create a new reproducible Task - :meth:`Task.init`

    .. important::
        In some cases, ``Task.init`` may return a Task object which is already stored in **ClearML Server** (already
        initialized), instead of creating a new Task. For a detailed explanation of those cases, see the ``Task.init``
        method.

    - Manually create a new Task (no auto-logging will apply) - :meth:`Task.create`
    - Get the current running Task - :meth:`Task.current_task`
    - Get another (different) Task - :meth:`Task.get_task`

    .. note::
        The **ClearML** documentation often refers to a Task as, "Task (experiment)".

        "Task" refers to the class in the ClearML Python Client Package, the object in your Python experiment script,
        and the entity with which **ClearML Server** and **ClearML Agent** work.

        "Experiment" refers to your deep learning solution, including its connected components, inputs, and outputs,
        and is the experiment you can view, analyze, compare, modify, duplicate, and manage using the ClearML
        **Web-App** (UI).

        Therefore, a "Task" is effectively an "experiment", and "Task (experiment)" encompasses its usage throughout
        the ClearML.

        The exception to this Task behavior is sub-tasks (non-reproducible Tasks), which do not use the main execution
        Task. Creating a sub-task always creates a new Task with a new  Task ID.
    """

    TaskTypes = _Task.TaskTypes

    NotSet = object()

    __create_protection = object()
    __main_task: Optional["Task"] = None
    __exit_hook = None
    __forked_proc_main_pid = None
    __task_id_reuse_time_window_in_hours = deferred_config("development.task_reuse_time_window_in_hours", 24.0, float)
    __detect_repo_async = deferred_config("development.vcs_repo_detect_async", False)
    __default_output_uri = DEV_DEFAULT_OUTPUT_URI.get() or deferred_config("development.default_output_uri", None)

    __hidden_tag = "hidden"

    _launch_multi_node_section = "launch_multi_node"
    _launch_multi_node_instance_tag = "multi_node_instance"

    _external_endpoint_port_map = {"http": "_PORT", "tcp": "external_tcp_port"}
    _external_endpoint_address_map = {"http": "_ADDRESS", "tcp": "external_address"}
    _external_endpoint_service_map = {"http": "EXTERNAL", "tcp": "EXTERNAL_TCP"}
    _external_endpoint_internal_port_map = {
        "http": "_PORT",
        "tcp": "upstream_task_port",
    }
    _external_endpoint_host_tcp_port_mapping = {"tcp_host_mapping": "_external_host_tcp_port_mapping"}

    class _ConnectedParametersType(object):
        argparse = "argument_parser"
        dictionary = "dictionary"
        task_parameters = "task_parameters"

        @classmethod
        def _options(cls):
            return {var for var, val in vars(cls).items() if isinstance(val, six.string_types)}

    def __init__(self, private: Optional[Any] = None, **kwargs: Any) -> None:
        """
        .. warning::
            **Do not construct Task manually!**
            Please use :meth:`Task.init` or :meth:`Task.get_task`
        """
        if private is not Task.__create_protection:
            raise UsageError(
                "Task object cannot be instantiated externally, use Task.current_task() or Task.get_task(...)"
            )
        self._repo_detect_lock = threading.RLock()

        super(Task, self).__init__(**kwargs)
        self._arguments = _Arguments(self)
        self._logger = None
        self._connected_output_model = None
        self._dev_worker = None
        self._connected_parameter_type = None
        self._detect_repo_async_thread = None
        self._resource_monitor = None
        self._calling_filename = None
        self._remote_functions_generated = {}
        self._external_endpoint_ports = {}
        self._http_router = None
        # register atexit, so that we mark the task as stopped
        self._at_exit_called = False

    @classmethod
    def current_task(cls) -> TaskInstance:
        """
        Get the current running Task (experiment). This is the main execution Task (task context) returned as a Task
        object.

        :return: The current running Task (experiment).
        :rtype: Task
        """
        # check if we have no main Task, but the main process created one.
        if not cls.__main_task and cls.__get_master_id_task_id():
            # initialize the Task, connect to stdout
            cls.init()
        # return main Task
        return cls.__main_task

    @classmethod
    def init(
        cls,
        project_name: Optional[str] = None,
        task_name: Optional[str] = None,
        task_type: "Task.TaskTypes" = TaskTypes.training,
        tags: Optional[Sequence[str]] = None,
        reuse_last_task_id: Union[bool, str] = True,
        continue_last_task: Union[bool, str, int] = False,
        output_uri: Optional[Union[str, bool]] = None,
        auto_connect_arg_parser: Union[bool, Mapping[str, bool]] = True,
        auto_connect_frameworks: Union[bool, Mapping[str, Union[bool, str, list]]] = True,
        auto_resource_monitoring: Union[bool, Mapping[str, Any]] = True,
        auto_connect_streams: Union[bool, Mapping[str, bool]] = True,
        deferred_init: bool = False,
    ) -> TaskInstance:
        """
        Creates a new Task (experiment) if:

        - The Task never ran before. No Task with the same ``task_name`` and ``project_name`` is stored in
          **ClearML Server**.
        - The Task has run before (the same ``task_name`` and ``project_name``), and (a) it stored models and / or
          artifacts, or (b) its status is Published , or (c) it is Archived.
        - A new Task is forced by calling ``Task.init`` with ``reuse_last_task_id=False``.

        Otherwise, the already initialized Task object for the same ``task_name`` and ``project_name`` is returned,
        or, when being executed remotely on a clearml-agent, the task returned is the existing task from the backend.

        .. note::
            To reference another Task, instead of initializing the same Task more than once, call
            :meth:`Task.get_task`. For example, to "share" the same experiment in more than one script,
            call ``Task.get_task``. See the ``Task.get_task`` method for an example.

        For example:
        The first time the following code runs, it will create a new Task. The status will be Completed.

        .. code-block:: py

            from clearml import Task
            task = Task.init('myProject', 'myTask')

        If this code runs again, it will not create a new Task. It does not store a model or artifact,
        it is not Published (its status Completed) , it was not Archived, and a new Task is not forced.

        If the Task is Published or Archived, and run again, it will create a new Task with a new Task ID.

        The following code will create a new Task every time it runs, because it stores an artifact.

        .. code-block:: py

            task = Task.init('myProject', 'myOtherTask')

            d = {'a': '1'}
            task.upload_artifact('myArtifact', d)

        :param str project_name: The name of the project in which the experiment will be created. If the project does
            not exist, it is created. If ``project_name`` is ``None``, the repository name is used. (Optional)
        :param str task_name: The name of Task (experiment). If ``task_name`` is ``None``, the Python experiment
            script's file name is used. (Optional)
        :param TaskTypes task_type: The task type. Valid task types:

            - ``TaskTypes.training`` (default)
            - ``TaskTypes.testing``
            - ``TaskTypes.inference``
            - ``TaskTypes.data_processing``
            - ``TaskTypes.application``
            - ``TaskTypes.monitor``
            - ``TaskTypes.controller``
            - ``TaskTypes.optimizer``
            - ``TaskTypes.service``
            - ``TaskTypes.qc``
            - ``TaskTypes.custom``

        :param tags: Add a list of tags (str) to the created Task. For example: tags=['512x512', 'yolov3']
        :param bool reuse_last_task_id: Force a new Task (experiment) with a previously used Task ID,
            and the same project and Task name. If the previously executed Task has artifacts or models, it will not be
            reused (overwritten), and a new Task will be created. When a Task is reused, the previous execution outputs
            are deleted, including console outputs and logs. The values are:

          - ``True`` - Reuse the last  Task ID. (default)
          - ``False`` - Force a new Task (experiment).
          - A string - You can also specify a Task ID (string) to be reused, instead of the cached ID based on the project/name combination.

        :param bool continue_last_task: Continue the execution of a previously executed Task (experiment). When
            continuing the executing of a previously executed Task,
            all previous artifacts / models / logs remain intact.
            New logs will continue iteration/step based on the previous-execution maximum iteration value.
            For example, The last train/loss scalar reported was iteration 100, the next report will be iteration 101.
            The values are:

          - ``True`` - Continue the last Task ID. Specified explicitly by reuse_last_task_id or implicitly with the same logic as reuse_last_task_id
          - ``False`` - Overwrite the execution of previous Task  (default).
          - A string - You can also specify a Task ID (string) to be continued. This is equivalent to `continue_last_task=True` and `reuse_last_task_id=a_task_id_string`.
          - An integer - Specify initial iteration offset (override the auto automatic last_iteration_offset). Pass 0, to disable the automatic last_iteration_offset or specify a different initial offset. You can specify a Task ID to be used with `reuse_last_task_id='task_id_here'`

        :param str output_uri: The default location for output models and other artifacts. If True, the default
            files_server will be used for model storage. In the default location, ClearML creates a subfolder for the
            output. If set to False, local runs will not upload output models and artifacts,
            and remote runs will not use any default values provided using ``default_output_uri``.
            The subfolder structure is the following: \\<output destination name\\> / \\<project name\\> / \\<task name\\>.\\<Task ID\\>.
            Note that for cloud storage, you must install the **ClearML** package for your cloud storage type,
            and then configure your storage credentials. For detailed information, see "Storage" in the ClearML
            Documentation.
            The following are examples of ``output_uri`` values for the supported locations:

          - A shared folder: ``/mnt/share/folder``
          - S3: ``s3://bucket/folder``
          - Google Cloud Storage: ``gs://bucket-name/folder``
          - Azure Storage: ``azure://company.blob.core.windows.net/folder/``
          - Default file server: True

        :param auto_connect_arg_parser: Automatically connect an argparse object to the Task. Supported argument
            parser packages are: argparse, click, python-fire, jsonargparse. The values are:

          - ``True`` - Automatically connect. (default)
          - ``False`` - Do not automatically connect.
          - A dictionary - In addition to a boolean, you can use a dictionary for fined grained control of connected
              arguments. The dictionary keys are argparse variable names and the values are booleans.
              The ``False`` value excludes the specified argument from the Task's parameter section.
              Keys missing from the dictionary default to ``True``, you can change it to be ``False`` by adding
              ``*`` key as ``False`` to the dictionary.
              An empty dictionary defaults to ``False``.

              For example:

              .. code-block:: py

                 auto_connect_arg_parser={"do_not_include_me": False, }

              .. code-block:: py

                 auto_connect_arg_parser={"only_include_me": True, "*": False}

              .. note::
               To manually connect an argparse, use :meth:`Task.connect`.

        :param auto_connect_frameworks: Automatically connect frameworks This includes patching MatplotLib, XGBoost,
            scikit-learn, Keras callbacks, and TensorBoard/X to serialize plots, graphs, and the model location to
            the **ClearML Server** (backend), in addition to original output destination.
            The values are:

          - ``True`` - Automatically connect (default)
          - ``False`` - Do not automatically connect
          - A dictionary - In addition to a boolean, you can use a dictionary for fined grained control of connected
              frameworks. The dictionary keys are frameworks and the values are booleans, other dictionaries used for
              finer control or wildcard strings.
              In case of wildcard strings, the local path of a model file has to match at least one wildcard to be
              saved/loaded by ClearML. Example: ``{'pytorch' : '*.pt', 'tensorflow': ['*.h5', '*']}``
              Keys missing from the dictionary default to ``True``, and an empty dictionary defaults to ``False``.
              Supported keys for finer control: ``{'tensorboard': {'report_hparams': bool}}``  # whether to report TensorBoard hyperparameters

              For example:

              .. code-block:: py

                 auto_connect_frameworks={
                     'matplotlib': True, 'tensorflow': ['*.hdf5, 'something_else*], 'tensorboard': True,
                     'pytorch': ['*.pt'], 'xgboost': True, 'scikit': True, 'fastai': True,
                     'lightgbm': True, 'hydra': True, 'detect_repository': True, 'tfdefines': True,
                     'joblib': True, 'megengine': True, 'catboost': True, 'gradio': True
                 }

              .. code-block:: py

                  auto_connect_frameworks={'tensorboard': {'report_hparams': False}}

        :param bool auto_resource_monitoring: Automatically create machine resource monitoring plots
            These plots appear in the **ClearML Web-App (UI)**, **RESULTS** tab, **SCALARS** sub-tab,
            with a title of **:resource monitor:**.
            The values are:

          - ``True`` - Automatically create resource monitoring plots. (default)
          - ``False`` - Do not automatically create.
          - Class Type - Create ResourceMonitor object of the specified class type.
          - dict - Dictionary of kwargs to be passed to the ResourceMonitor instance. The keys can be:
              - `report_start_sec` OR `first_report_sec` OR `seconds_from_start` - Maximum number of seconds
                to wait for scalar/plot reporting before defaulting
                to machine statistics reporting based on seconds from experiment start time
              - `wait_for_first_iteration_to_start_sec` - Set the initial time (seconds) to wait for iteration
                reporting to be used as x-axis for the resource monitoring,
                if timeout exceeds then reverts to `seconds_from_start`
              - `max_wait_for_first_iteration_to_start_sec` - Set the maximum time (seconds) to allow the resource
                monitoring to revert back to iteration reporting x-axis after starting to report `seconds_from_start`
              - `report_mem_used_per_process` OR `report_global_mem_used` - Compatibility feature,
                report memory usage for the entire machine.
                Default (false), report only on the running process and its sub-processes

        :param auto_connect_streams: Control the automatic logging of stdout and stderr.
            The values are:

          - ``True`` - Automatically connect (default)
          -  ``False`` - Do not automatically connect
          - A dictionary - In addition to a boolean, you can use a dictionary for fined grained control of stdout and
            stderr. The dictionary keys are 'stdout' , 'stderr' and 'logging', the values are booleans.
            Keys missing from the dictionary default to ``False``, and an empty dictionary defaults to ``False``.
            Notice, the default behaviour is logging stdout/stderr. The `logging` module is logged as a by product
            of the stderr logging

            For example:

            .. code-block:: py

               auto_connect_streams={'stdout': True, 'stderr': True, 'logging': False}

        :param deferred_init: (default: False) Wait for Task to be fully initialized (regular behaviour).
            ** BETA feature! use with care **.

            If set to True, `Task.init` function returns immediately and all initialization / communication
            to the clearml-server is running in a background thread. The returned object is
            a full proxy to the regular Task object, hence everything will be working as expected.
            Default behaviour can be controlled with: ``CLEARML_DEFERRED_TASK_INIT=1``. Notes:

          - Any access to the returned proxy `Task` object will essentially wait for the `Task.init` to be completed.
            For example: `print(task.name)` will wait for `Task.init` to complete in the
            background and then return the `name` property of the task original object
          - Before `Task.init` completes in the background, auto-magic logging (console/metric) might be missed
          - If running via an agent, this argument is ignored, and Task init is called synchronously (default)

        :return: The main execution Task (Task context)
        :rtype: Task
        """

        def verify_defaults_match() -> None:
            validate = [
                ("project name", project_name, cls.__main_task.get_project_name()),
                ("task name", task_name, cls.__main_task.name),
                (
                    "task type",
                    str(task_type) if task_type else task_type,
                    str(cls.__main_task.task_type),
                ),
            ]

            for field, default, current in validate:
                if default is not None and default != current:
                    raise UsageError(
                        "Current task already created "
                        "and requested {field} '{default}' does not match current {field} '{current}'. "
                        "If you wish to create additional tasks use `Task.create`, "
                        "or close the current task with `task.close()` before calling `Task.init(...)`".format(
                            field=field,
                            default=default,
                            current=current,
                        )
                    )

        if cls.__main_task is not None and deferred_init != cls.__nested_deferred_init_flag:
            # if this is a subprocess, regardless of what the init was called for,
            # we have to fix the main task hooks and stdout bindings
            if cls.__forked_proc_main_pid != os.getpid() and cls.__is_subprocess():
                if task_type is None:
                    task_type = cls.__main_task.task_type
                # make sure we only do it once per process
                cls.__forked_proc_main_pid = os.getpid()
                # make sure we do not wait for the repo detect thread
                cls.__main_task._detect_repo_async_thread = None
                cls.__main_task._dev_worker = None
                cls.__main_task._resource_monitor = None

                # if we are using threads to send the reports,
                # after forking there are no threads, so we will need to recreate them
                if not getattr(cls, "_report_subprocess_enabled"):
                    # remove the logger from the previous process
                    cls.__main_task.get_logger()
                    # create a new logger (to catch stdout/err)
                    cls.__main_task._logger = None
                    cls.__main_task.__reporter = None
                    # noinspection PyProtectedMember
                    cls.__main_task._get_logger(auto_connect_streams=auto_connect_streams)
                    cls.__main_task._artifacts_manager = Artifacts(cls.__main_task)

                # unregister signal hooks, they cause subprocess to hang
                # noinspection PyProtectedMember
                cls.__main_task.__register_at_exit(cls.__main_task._at_exit)

                # if we are using threads to send the reports,
                # after forking there are no threads, so we will need to recreate them
                if not getattr(cls, "_report_subprocess_enabled"):
                    # start all reporting threads
                    BackgroundMonitor.start_all(task=cls.__main_task)

            if not running_remotely():
                verify_defaults_match()

            return cls.__main_task

        is_sub_process_task_id = None
        # check that we are not a child process, in that case do nothing.
        # we should not get here unless this is Windows/macOS platform, linux support fork
        if cls.__is_subprocess():
            is_sub_process_task_id = cls.__get_master_id_task_id()
            # we could not find a task ID, revert to old stub behaviour
            if not is_sub_process_task_id:
                return _TaskStub()  # noqa

        elif running_remotely() and not get_is_master_node():
            # make sure we only do it once per process
            cls.__forked_proc_main_pid = os.getpid()
            # make sure everyone understands we should act as if we are a subprocess (fake pid 1)
            cls.__update_master_pid_task(pid=1, task=get_remote_task_id())
        else:
            # set us as master process (without task ID)
            cls.__update_master_pid_task()
            is_sub_process_task_id = None

        if task_type is None:
            # Backwards compatibility: if called from Task.current_task and task_type
            # was not specified, keep legacy default value of TaskTypes.training
            task_type = cls.TaskTypes.training
        elif isinstance(task_type, six.string_types):
            if task_type not in Task.TaskTypes.__members__:
                raise ValueError(
                    "Task type '{}' not supported, options are: {}".format(task_type, Task.TaskTypes.__members__.keys())
                )
            task_type = Task.TaskTypes.__members__[str(task_type)]

        def safe_project_default_output_destination(task_, default=None):
            project_ = task_.get_project_object()
            return project_.default_output_destination if project_ else default

        is_deferred = False
        try:
            if not running_remotely():
                # check remote status
                _local_rank = get_torch_local_rank()
                if _local_rank is not None and _local_rank > 0:
                    is_sub_process_task_id = get_torch_distributed_anchor_task_id(timeout=30)

                # only allow if running locally and creating the first Task
                # otherwise we ignore and perform in order
                if ENV_DEFERRED_TASK_INIT.get():
                    deferred_init = True

                if not is_sub_process_task_id and deferred_init and deferred_init != cls.__nested_deferred_init_flag:

                    def completed_cb(x):
                        Task.__forked_proc_main_pid = os.getpid()
                        Task.__main_task = x

                    getLogger().warning("ClearML initializing Task in the background")

                    task = FutureTaskCaller(
                        func=cls.init,
                        func_cb=completed_cb,
                        override_cls=cls,
                        project_name=project_name,
                        task_name=task_name,
                        tags=tags,
                        reuse_last_task_id=reuse_last_task_id,
                        continue_last_task=continue_last_task,
                        output_uri=output_uri,
                        auto_connect_arg_parser=auto_connect_arg_parser,
                        auto_connect_frameworks=auto_connect_frameworks,
                        auto_resource_monitoring=auto_resource_monitoring,
                        auto_connect_streams=auto_connect_streams,
                        deferred_init=cls.__nested_deferred_init_flag,
                    )
                    is_deferred = True
                    # mark as temp master
                    cls.__update_master_pid_task()
                # if this is the main process, create the task
                elif not is_sub_process_task_id:
                    try:
                        task = cls._create_dev_task(
                            default_project_name=project_name,
                            default_task_name=task_name,
                            default_task_type=task_type,
                            tags=tags,
                            reuse_last_task_id=reuse_last_task_id,
                            continue_last_task=continue_last_task,
                            detect_repo=(
                                False
                                if (
                                    isinstance(auto_connect_frameworks, dict)
                                    and not auto_connect_frameworks.get("detect_repository", True)
                                )
                                else True
                            ),
                            auto_connect_streams=auto_connect_streams,
                        )
                        # check if we are local rank 0 (local master),
                        # create an anchor with task ID for the other processes
                        if _local_rank == 0:
                            create_torch_distributed_anchor(task_id=task.id)

                    except MissingConfigError as e:
                        if not ENV_IGNORE_MISSING_CONFIG.get():
                            raise
                        getLogger().warning(str(e))
                        # return a Task-stub instead of the original class
                        # this will make sure users can still call the Stub without code breaking
                        return _TaskStub()  # noqa
                    # set defaults
                    if cls._offline_mode:
                        task.output_uri = None
                        # create target data folder for logger / artifacts
                        # noinspection PyProtectedMember
                        Path(task._get_default_report_storage_uri()).mkdir(parents=True, exist_ok=True)
                    elif output_uri is not None:
                        if output_uri is True:
                            output_uri = safe_project_default_output_destination(task) or True
                        task.output_uri = output_uri
                    elif safe_project_default_output_destination(task):
                        task.output_uri = safe_project_default_output_destination(task)
                    elif cls.__default_output_uri:
                        task.output_uri = str(cls.__default_output_uri)
                    # store new task ID
                    cls.__update_master_pid_task(task=task)
                else:
                    # subprocess should get back the task info
                    task = cls.get_task(task_id=is_sub_process_task_id)
            else:
                # if this is the main process, create the task
                if not is_sub_process_task_id:
                    task = cls(
                        private=cls.__create_protection,
                        task_id=get_remote_task_id(),
                        log_to_backend=False,
                    )
                    if output_uri is False and not task.output_uri:
                        # Setting output_uri=False argument will disable using any default when running remotely
                        pass
                    else:
                        if safe_project_default_output_destination(task) and not task.output_uri:
                            task.output_uri = safe_project_default_output_destination(task)
                        if cls.__default_output_uri and not task.output_uri:
                            task.output_uri = cls.__default_output_uri
                    # store new task ID
                    cls.__update_master_pid_task(task=task)
                    # make sure we are started
                    task.started(ignore_errors=True)
                    # continue last iteration if we had any (or we need to override it)
                    if isinstance(continue_last_task, int) and not isinstance(continue_last_task, bool):
                        task.set_initial_iteration(int(continue_last_task))
                    elif task.data.last_iteration:
                        task.set_initial_iteration(int(task.data.last_iteration) + 1)
                else:
                    # subprocess should get back the task info
                    task = cls.get_task(task_id=is_sub_process_task_id)
        except Exception:
            raise
        else:
            Task.__forked_proc_main_pid = os.getpid()
            Task.__main_task = task

            # register at exist only on the real (none deferred) Task
            if not is_deferred:
                # register the main task for at exit hooks (there should only be one)
                # noinspection PyProtectedMember
                task.__register_at_exit(task._at_exit)
                # noinspection PyProtectedMember
                if cls.__exit_hook:
                    cls.__exit_hook.register_signal_and_exception_hooks()

            # always patch OS forking because of ProcessPool and the alike
            PatchOsFork.patch_fork(task)
            if auto_connect_frameworks:

                def should_connect(*keys):
                    """
                    Evaluates value of auto_connect_frameworks[keys[0]]...[keys[-1]].
                    If at some point in the evaluation, the value of auto_connect_frameworks[keys[0]]...[keys[-1]]
                    is a bool, that value will be returned. If a dictionary is empty, it will be evaluated to False.
                    If a key will not be found in the current dictionary, True will be returned.
                    """
                    should_bind_framework = auto_connect_frameworks
                    for key in keys:
                        if not isinstance(should_bind_framework, dict):
                            return bool(should_bind_framework)
                        if should_bind_framework == {}:
                            return False
                        should_bind_framework = should_bind_framework.get(key, True)
                    return bool(should_bind_framework)

                if not is_deferred and should_connect("hydra"):
                    PatchHydra.update_current_task(task)
                if should_connect("scikit") and should_connect("joblib"):
                    PatchedJoblib.update_current_task(task)
                if should_connect("matplotlib"):
                    PatchedMatplotlib.update_current_task(task)
                if should_connect("tensorflow") or should_connect("tensorboard"):
                    # allow disabling tfdefines
                    if not is_deferred and should_connect("tfdefines"):
                        PatchAbsl.update_current_task(task)
                    TensorflowBinding.update_current_task(
                        task,
                        patch_reporting=should_connect("tensorboard"),
                        patch_model_io=should_connect("tensorflow"),
                        report_hparams=should_connect("tensorboard", "report_hparams"),
                    )
                if should_connect("pytorch"):
                    PatchPyTorchModelIO.update_current_task(task)
                if should_connect("megengine"):
                    PatchMegEngineModelIO.update_current_task(task)
                if should_connect("xgboost"):
                    PatchXGBoostModelIO.update_current_task(task)
                if should_connect("catboost"):
                    PatchCatBoostModelIO.update_current_task(task)
                if should_connect("fastai"):
                    PatchFastai.update_current_task(task)
                if should_connect("lightgbm"):
                    PatchLIGHTgbmModelIO.update_current_task(task)
                if should_connect("gradio"):
                    PatchGradio.update_current_task(task)

                cls.__add_model_wildcards(auto_connect_frameworks)

            # if we are deferred, stop here (the rest we do in the actual init)
            if is_deferred:
                from .backend_interface.logger import StdStreamPatch

                # patch console outputs, we will keep them in memory until we complete the Task init
                # notice we do not load config defaults, as they are not threadsafe
                # we might also need to override them with the vault
                StdStreamPatch.patch_std_streams(
                    task.get_logger(),
                    connect_stdout=(auto_connect_streams is True)
                    or (isinstance(auto_connect_streams, dict) and auto_connect_streams.get("stdout", False)),
                    connect_stderr=(auto_connect_streams is True)
                    or (isinstance(auto_connect_streams, dict) and auto_connect_streams.get("stderr", False)),
                    load_config_defaults=False,
                )
                return task  # noqa

            if auto_resource_monitoring and not is_sub_process_task_id:
                resource_monitor_cls = (
                    auto_resource_monitoring
                    if isinstance(auto_resource_monitoring, six.class_types)
                    else ResourceMonitor
                )
                resource_monitor_kwargs = dict(
                    report_mem_used_per_process=not config.get("development.worker.report_global_mem_used", False),
                    first_report_sec=config.get("development.worker.report_start_sec", None),
                    wait_for_first_iteration_to_start_sec=config.get(
                        "development.worker.wait_for_first_iteration_to_start_sec", None
                    ),
                    max_wait_for_first_iteration_to_start_sec=config.get(
                        "development.worker.max_wait_for_first_iteration_to_start_sec",
                        None,
                    ),
                )
                if isinstance(auto_resource_monitoring, dict):
                    if "report_start_sec" in auto_resource_monitoring:
                        auto_resource_monitoring["first_report_sec"] = auto_resource_monitoring.pop("report_start_sec")
                    if "seconds_from_start" in auto_resource_monitoring:
                        auto_resource_monitoring["first_report_sec"] = auto_resource_monitoring.pop(
                            "seconds_from_start"
                        )
                    if "report_global_mem_used" in auto_resource_monitoring:
                        auto_resource_monitoring["report_mem_used_per_process"] = auto_resource_monitoring.pop(
                            "report_global_mem_used"
                        )
                    resource_monitor_kwargs.update(auto_resource_monitoring)
                task._resource_monitor = resource_monitor_cls(task, **resource_monitor_kwargs)
                task._resource_monitor.start()

            # make sure all random generators are initialized with new seed
            random_seed = task.get_random_seed()
            if random_seed is not None:
                make_deterministic(random_seed)
            task._set_random_seed_used(random_seed)

            if auto_connect_arg_parser:
                EnvironmentBind.update_current_task(task)

                PatchJsonArgParse.update_current_task(task)

                # Patch ArgParser to be aware of the current task
                argparser_update_currenttask(task)

                PatchClick.patch(task)
                PatchFire.patch(task)

                # set excluded arguments
                if isinstance(auto_connect_arg_parser, dict):
                    task._arguments.exclude_parser_args(auto_connect_arg_parser)

                # Check if parse args already called. If so, sync task parameters with parser
                if argparser_parseargs_called():
                    for parser, parsed_args in get_argparser_last_args():
                        task._connect_argparse(parser=parser, parsed_args=parsed_args)

                PatchHydra.delete_overrides()
            elif argparser_parseargs_called():
                # actually we have nothing to do, in remote running, the argparser will ignore
                # all non argparser parameters, only caveat if parameter connected with the same name
                # as the argparser this will be solved once sections are introduced to parameters
                pass

        # Make sure we start the logger, it will patch the main logging object and pipe all output
        # if we are running locally and using development mode worker, we will pipe all stdout to logger.
        # The logger will automatically take care of all patching (we just need to make sure to initialize it)
        logger = task._get_logger(auto_connect_streams=auto_connect_streams)
        # show the debug metrics page in the log, it is very convenient
        if not is_sub_process_task_id:
            if cls._offline_mode:
                logger.report_text(
                    "ClearML running in offline mode, session stored in {}".format(task.get_offline_mode_folder())
                )
            else:
                logger.report_text("ClearML results page: {}".format(task.get_output_log_web_page()))
        # Make sure we start the dev worker if required, otherwise it will only be started when we write
        # something to the log.
        task._dev_mode_setup_worker()

        if (
            (not task._reporter or not task._reporter.is_constructed())
            and is_sub_process_task_id
            and not cls._report_subprocess_enabled
        ):
            task._setup_reporter()

        # start monitoring in background process or background threads
        # monitoring are: Resource monitoring and Dev Worker monitoring classes
        BackgroundMonitor.start_all(task=task)

        # noinspection PyProtectedMember
        task._set_startup_info()
        return task

    def get_http_router(self) -> "HttpRouter":
        """
        Retrieve an instance of `HttpRouter` to manage an external HTTP endpoint and intercept traffic.
        The `HttpRouter` serves as a traffic manager, enabling the creation and configuration of local and external
        routesto redirect, monitor, or manipulate HTTP requests and responses. It is designed to handle routing
        needs such via a proxy setup which handles request/response interception and telemetry reporting for
        applications that require HTTP endpoint management.

        Example usage:

        .. code-block:: py
            def request_callback(request, persistent_state):
                persistent_state["last_request_time"] = time.time()

            def response_callback(response, request, persistent_state):
                print("Latency:", time.time() - persistent_state["last_request_time"])
                if urllib.parse.urlparse(str(request.url).rstrip("/")).path == "/modify":
                    new_content = response.body.replace(b"modify", b"modified")
                    headers = copy.deepcopy(response.headers)
                    headers["Content-Length"] = str(len(new_content))
                    return Response(status_code=response.status_code, headers=headers, content=new_content)

            router = Task.current_task().get_http_router()
            router.set_local_proxy_parameters(incoming_port=9000)
            router.create_local_route(
                source="/",
                target="http://localhost:8000",
                request_callback=request_callback,
                response_callback=response_callback,
                endpoint_telemetry={"model": "MyModel"}
            )
            router.deploy(wait=True)
        """
        try:
            from .router.router import HttpRouter  # noqa
        except ImportError:
            raise UsageError("Could not import `HttpRouter`. Please run `pip install clearml[router]`")

        if self._http_router is None:
            self._http_router = HttpRouter(self)
        return self._http_router

    def request_external_endpoint(
        self,
        port: int,
        protocol: str = "http",
        wait: bool = False,
        wait_interval_seconds: float = 3.0,
        wait_timeout_seconds: float = 90.0,
        static_route: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Request an external endpoint for an application

        :param port: Port the application is listening to
        :param protocol: `http` or `tcp`
        :param wait: If True, wait for the endpoint to be assigned
        :param wait_interval_seconds: The poll frequency when waiting for the endpoint
        :param wait_timeout_seconds: If this timeout is exceeded while waiting for the endpoint,
            the method will no longer wait and None will be returned
        :param static_route: The static route name (not the route path).
            When set, the external endpoint requested will use this route
            instead of generating it based on the task ID. Useful for creating
            persistent, load balanced routes.

        :return: If wait is False, this method will return None.
            If no endpoint could be found while waiting, this method returns None.
            Otherwise, it returns a dictionary containing the following values:
          - endpoint - raw endpoint. One might need to authenticate in order to use this endpoint
          - browser_endpoint - endpoint to be used in browser. Authentication will be handled via the browser
          - port - the port exposed by the application
          - protocol - the protocol used by the endpoint
        """
        Session.verify_feature_set("advanced")
        if protocol not in self._external_endpoint_port_map.keys():
            raise ValueError("Invalid protocol: {}".format(protocol))
        if static_route:
            self._validate_static_route(static_route)

        # sync with router - get data from Task
        if not self._external_endpoint_ports.get(protocol):
            self.reload()
            internal_port = self._get_runtime_properties().get(self._external_endpoint_internal_port_map[protocol])
            if internal_port:
                self._external_endpoint_ports[protocol] = internal_port

            # notice this applies for both raw tcp and http, it is so that we can
            # detect the host machine exposed ports, and register them on the router
            external_host_port_mapping = self._get_runtime_properties().get(
                self._external_endpoint_host_tcp_port_mapping["tcp_host_mapping"]
            )
            self._external_endpoint_ports["tcp_host_mapping"] = external_host_port_mapping

        # check if we need to parse the port mapping, only if running on "bare-metal" host machine.
        if self._external_endpoint_ports.get("tcp_host_mapping"):
            external_host_port_mapping = self._external_endpoint_ports.get("tcp_host_mapping")
            # format is docker standard port mapping format:
            # example: "out:in,out_range100-out_range102:in_range0-in_range2"
            # notice `out` in this context means the host port, the one that
            # the router will route external traffic to
            # noinspection PyBroadException
            out_port = None
            # noinspection PyBroadException
            try:
                for port_range in external_host_port_mapping.split(","):
                    out_range, in_range = port_range.split(":", 1)
                    out_range = out_range.split("-")
                    in_range = in_range.split("-")
                    if int(in_range[0]) <= port <= int(in_range[-1]):
                        # we found a match:
                        out_port = int(out_range[0]) + (port - int(in_range[0]))
                        print(
                            "INFO: Task.request_external_endpoint(...) changed requested external port to {}, "
                            "conforming to mapped external host ports [{} -> {}]".format(out_port, port, port_range)
                        )
                        break

                if not out_port:
                    raise ValueError("match not found defaulting to original port")
            except Exception:
                print(
                    "WARNING: Task.request_external_endpoint(...) failed matching requested port to "
                    "mapped external host port [{} to {}], "
                    "proceeding with original port {}".format(port, external_host_port_mapping, port)
                )

            # change the requested port to the one we have on the machine
            if out_port:
                port = out_port

        # check if we are trying to change the port - currently not allowed
        if self._external_endpoint_ports.get(protocol):
            if self._external_endpoint_ports.get(protocol) == port:
                # we already set this endpoint, but we will set the values again, because maybe IP changed?!
                pass
            else:
                raise ValueError(
                    "Only one endpoint per protocol can be requested at the moment. "
                    "Port already exposed is: {}".format(self._external_endpoint_ports.get(protocol))
                )

        # mark for the router our request
        # noinspection PyProtectedMember
        self._set_runtime_properties(
            {
                "_SERVICE": self._external_endpoint_service_map[protocol],
                self._external_endpoint_address_map[protocol]: HOST_MACHINE_IP.get() or get_private_ip(),
                self._external_endpoint_port_map[protocol]: port,
                **(
                    {"_ROUTER_ENDPOINT_MODE": "path", "_ROUTER_ENDPOINT_MODE_PARAM": static_route}
                    if static_route
                    else {}
                ),
            }
        )
        # required system_tag for the router to catch the routing request
        self.set_system_tags(list(set((self.get_system_tags() or []) + ["external_service"])))
        self._external_endpoint_ports[protocol] = port
        if wait:
            return self.wait_for_external_endpoint(
                wait_interval_seconds=wait_interval_seconds,
                wait_timeout_seconds=wait_timeout_seconds,
                protocol=protocol,
            )
        return None

    def wait_for_external_endpoint(
        self, wait_interval_seconds: float = 3.0, wait_timeout_seconds: float = 90.0, protocol: Optional[str] = "http"
    ) -> Union[Optional[Dict], List[Optional[Dict]]]:
        """
        Wait for an external endpoint to be assigned

        :param wait_interval_seconds: The poll frequency when waiting for the endpoint
        :param wait_timeout_seconds: If this timeout is exceeded while waiting for the endpoint,
            the method will no longer wait
        :param protocol: `http` or `tcp`. Wait for an endpoint to be assigned based on the protocol.
            If None, wait for all supported protocols

        :return: If no endpoint could be found while waiting, this method returns None.
            If a protocol has been specified, it returns a dictionary containing the following values:
          - endpoint - raw endpoint. One might need to authenticate in order to use this endpoint
          - browser_endpoint - endpoint to be used in browser. Authentication will be handled via the browser
          - port - the port exposed by the application
          - protocol - the protocol used by the endpoint
            If not protocol is specified, it returns a list of dictionaries containing the values above,
            for each protocol requested and waited
        """
        Session.verify_feature_set("advanced")
        if protocol:
            return self._wait_for_external_endpoint(
                wait_interval_seconds=wait_interval_seconds,
                wait_timeout_seconds=wait_timeout_seconds,
                protocol=protocol,
                warn=True,
            )
        results = []
        protocols = ["http", "tcp"]
        waited_protocols = []
        for protocol_ in protocols:
            start_time = time.time()
            result = self._wait_for_external_endpoint(
                wait_interval_seconds=wait_interval_seconds,
                wait_timeout_seconds=wait_timeout_seconds,
                protocol=protocol_,
                warn=False,
            )
            elapsed = time.time() - start_time
            if result:
                results.append(result)
            wait_timeout_seconds -= elapsed
            if wait_timeout_seconds > 0 or result:
                waited_protocols.append(protocol_)
        unwaited_protocols = [p for p in protocols if p not in waited_protocols]
        if wait_timeout_seconds <= 0 and unwaited_protocols:
            LoggerRoot.get_base_logger().warning(
                "Timeout exceeded while waiting for {} endpoint(s)".format(",".join(unwaited_protocols))
            )
        return results

    def _wait_for_external_endpoint(
        self,
        wait_interval_seconds=3.0,
        wait_timeout_seconds=90.0,
        protocol="http",
        warn=True,
    ):
        if not self._external_endpoint_ports.get(protocol):
            self.reload()
            internal_port = self._get_runtime_properties().get(self._external_endpoint_internal_port_map[protocol])
            if internal_port:
                self._external_endpoint_ports[protocol] = internal_port
        if not self._external_endpoint_ports.get(protocol):
            if warn:
                LoggerRoot.get_base_logger().warning("No external {} endpoints have been requested".format(protocol))
            return None
        start_time = time.time()
        while True:
            self.reload()
            runtime_props = self._get_runtime_properties()
            endpoint, browser_endpoint = None, None
            if protocol == "http":
                endpoint = runtime_props.get("endpoint")
                browser_endpoint = runtime_props.get("browser_endpoint")
            elif protocol == "tcp":
                health_check = runtime_props.get("upstream_task_port")
                if health_check:
                    endpoint = (
                        runtime_props.get(self._external_endpoint_address_map[protocol])
                        + ":"
                        + str(runtime_props.get(self._external_endpoint_port_map[protocol]))
                    )
            if endpoint or browser_endpoint:
                return {
                    "endpoint": endpoint,
                    "browser_endpoint": browser_endpoint,
                    "port": self._external_endpoint_ports[protocol],
                    "protocol": protocol,
                }
            if time.time() >= start_time + wait_timeout_seconds:
                if warn:
                    LoggerRoot.get_base_logger().warning(
                        "Timeout exceeded while waiting for {} endpoint".format(protocol)
                    )
                return None
            time.sleep(wait_interval_seconds)

    def list_external_endpoints(self, protocol: Optional[str] = None) -> List[Dict]:
        """
        List all external endpoints assigned

        :param protocol: If None, list all external endpoints. Otherwise, only list endpoints
            that use this protocol

        :return: A list of dictionaries. Each dictionary contains the following values:

          - endpoint - raw endpoint. One might need to authenticate in order to use this endpoint
          - browser_endpoint - endpoint to be used in browser. Authentication will be handled via the browser
          - port - the port exposed by the application
          - protocol - the protocol used by the endpoint
        """
        Session.verify_feature_set("advanced")
        runtime_props = self._get_runtime_properties()
        results = []
        protocols = [protocol] if protocol is not None else ["http", "tcp"]
        for protocol in protocols:
            internal_port = runtime_props.get(self._external_endpoint_internal_port_map[protocol])
            if internal_port:
                self._external_endpoint_ports[protocol] = internal_port
            else:
                continue
            endpoint, browser_endpoint = None, None
            if protocol == "http":
                endpoint = runtime_props.get("endpoint")
                browser_endpoint = runtime_props.get("browser_endpoint")
            elif protocol == "tcp":
                health_check = runtime_props.get("upstream_task_port")
                if health_check:
                    endpoint = (
                        runtime_props.get(self._external_endpoint_address_map[protocol])
                        + ":"
                        + str(runtime_props.get(self._external_endpoint_port_map[protocol]))
                    )
            if endpoint or browser_endpoint:
                results.append(
                    {
                        "endpoint": endpoint,
                        "browser_endpoint": browser_endpoint,
                        "port": internal_port,
                        "protocol": protocol,
                    }
                )
        return results

    @classmethod
    def create(
        cls,
        project_name: Optional[str] = None,
        task_name: Optional[str] = None,
        task_type: Optional[str] = None,
        repo: Optional[str] = None,
        branch: Optional[str] = None,
        commit: Optional[str] = None,
        script: Optional[str] = None,
        working_directory: Optional[str] = None,
        packages: Optional[Union[bool, Sequence[str]]] = None,
        requirements_file: Optional[Union[str, Path]] = None,
        docker: Optional[str] = None,
        docker_args: Optional[str] = None,
        docker_bash_setup_script: Optional[str] = None,
        argparse_args: Optional[Sequence[Tuple[str, str]]] = None,
        base_task_id: Optional[str] = None,
        add_task_init_call: bool = True,
        force_single_script_file: bool = False,
        binary: Optional[str] = None,
        module: Optional[str] = None,
        detect_repository: bool = True
    ) -> TaskInstance:
        """
        Manually create and populate a new Task (experiment) in the system.
        If the code does not already contain a call to ``Task.init``, pass add_task_init_call=True,
        and the code will be patched in remote execution (i.e. when executed by `clearml-agent`)

        .. note::
           This method **always** creates a new Task.
           Use :meth:`Task.init` method to automatically create and populate task for the running process.
           To reference an existing Task, call the  :meth:`Task.get_task` method .

        :param project_name: Set the project name for the task. Required if base_task_id is None.
        :param task_name: Set the name of the remote task. Required if base_task_id is None.
        :param task_type: Optional, The task type to be created. Supported values: 'training', 'testing', 'inference',
            'data_processing', 'application', 'monitor', 'controller', 'optimizer', 'service', 'qc', 'custom'
        :param repo: Remote URL for the repository to use, or path to local copy of the git repository.
            Example: 'https://github.com/allegroai/clearml.git' or '~/project/repo'. If ``repo`` is specified, then
            the ``script`` parameter must also be specified
        :param branch: Select specific repository branch/tag (implies the latest commit from the branch)
        :param commit: Select specific commit ID to use (default: latest commit,
            or when used with local repository matching the local commit ID)
        :param script: Specify the entry point script for the remote execution. When used in tandem with
            remote git repository the script should be a relative path inside the repository,
            for example: './source/train.py' . When used with local repository path it supports a
            direct path to a file inside the local repository itself, for example: '~/project/source/train.py'
        :param working_directory: Working directory to launch the script from. Default: repository root folder.
            Relative to repo root or local folder.
        :param packages: Manually specify a list of required packages. Example: ``["tqdm>=2.1", "scikit-learn"]``
            or `True` to automatically create requirements
            based on locally installed packages (repository must be local).
            Pass an empty string to not install any packages (not even from the repository)
        :param requirements_file: Specify requirements.txt file to install when setting the session.
            If not provided, the requirements.txt from the repository will be used.
        :param docker: Select the docker image to be executed in by the remote session
        :param docker_args: Add docker arguments, pass a single string
        :param docker_bash_setup_script: Add bash script to be executed
            inside the docker before setting up the Task's environment
        :param argparse_args: Arguments to pass to the remote execution, list of string pairs (argument, value)
            Notice, only supported if the codebase itself uses argparse.ArgumentParser
        :param base_task_id: Use a pre-existing task in the system, instead of a local repo/script.
            Essentially clones an existing task and overrides arguments/requirements.
        :param add_task_init_call: If True, a 'Task.init()' call is added to the script entry point in remote execution.
        :param force_single_script_file: If True, do not auto-detect local repository
        :param binary: Binary used to launch the entry point
        :param module: If specified instead of executing `script`, a module named `module` is executed.
            Implies script is empty. Module can contain multiple argument for execution,
            for example: module="my.module arg1 arg2"
        :param detect_repository: If True, detect the repository if no repository has been specified.
            If False, don't detect repository under any circumstance. Ignored if `repo` is specified

        :return: The newly created Task (experiment)
        :rtype: Task
        """
        if cls.is_offline():
            raise UsageError("Creating task in offline mode. Use 'Task.init' instead.")
        if not project_name and not base_task_id:
            if not cls.__main_task:
                raise ValueError(
                    "Please provide project_name, no global task context found "
                    "(Task.current_task hasn't been called)"
                )
            project_name = cls.__main_task.get_project_name()
        from .backend_interface.task.populate import CreateAndPopulate

        manual_populate = CreateAndPopulate(
            project_name=project_name,
            task_name=task_name,
            task_type=task_type,
            repo=repo,
            branch=branch,
            commit=commit,
            script=script,
            working_directory=working_directory,
            packages=packages,
            requirements_file=requirements_file,
            docker=docker,
            docker_args=docker_args,
            docker_bash_setup_script=docker_bash_setup_script,
            base_task_id=base_task_id,
            add_task_init_call=add_task_init_call,
            force_single_script_file=force_single_script_file,
            raise_on_missing_entries=False,
            module=module,
            binary=binary,
            detect_repository=detect_repository
        )
        task = manual_populate.create_task()
        if task and argparse_args:
            manual_populate.update_task_args(argparse_args)
            task.reload()

        return task

    @classmethod
    def get_by_name(cls, task_name: str) -> TaskInstance:
        """

        .. note::
            This method is deprecated, use :meth:`Task.get_task` instead.

        Returns the most recent task with the given name from anywhere in the system as a Task object.

        :param str task_name: The name of the task to search for.

        :return: Task object of the most recent task with that name.
        """
        warnings.warn(
            "Warning: 'Task.get_by_name' is deprecated. Use 'Task.get_task' instead",
            DeprecationWarning,
        )
        return cls.get_task(task_name=task_name)

    @classmethod
    def get_task(
        cls,
        task_id: Optional[str] = None,
        project_name: Optional[str] = None,
        task_name: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        allow_archived: bool = True,
        task_filter: Optional[dict] = None,
    ) -> TaskInstance:
        """
        Get a Task by ID, or project name / task name combination.

        For example:

        The following code demonstrates calling ``Task.get_task`` to report a scalar to another Task. The output
        of :meth:`.Logger.report_scalar` from testing is associated with the Task named ``training``. It allows
        training and testing to run concurrently, because they initialized different Tasks (see :meth:`Task.init`
        for information about initializing Tasks).

        The training script:

        .. code-block:: py

            # initialize the training Task
            task = Task.init('myProject', 'training')

            # do some training

        The testing script:

        .. code-block:: py

            # initialize the testing Task
            task = Task.init('myProject', 'testing')

            # get the training Task
            train_task = Task.get_task(project_name='myProject', task_name='training')

            # report metrics in the training Task
            for x in range(10):
                train_task.get_logger().report_scalar('title', 'series', value=x * 2, iteration=x)

        :param str task_id: The ID (system UUID) of the experiment to get.
            If specified, ``project_name`` and ``task_name`` are ignored.
        :param str project_name: The project name of the Task to get.
        :param str task_name: The name of the Task within ``project_name`` to get.
        :param list tags: Filter based on the requested list of tags (strings). To exclude a tag add "-" prefix to the
            tag. Example: ``["best", "-debug"]``.
            The default behaviour is to join all tags with a logical "OR" operator.
            To join all tags with a logical "AND" operator instead, use "__$all" as the first string, for example:

            .. code-block:: py

                ["__$all", "best", "experiment", "ever"]

            To join all tags with AND, but exclude a tag use "__$not" before the excluded tag, for example:

            .. code-block:: py

                ["__$all", "best", "experiment", "ever", "__$not", "internal", "__$not", "test"]

            The "OR" and "AND" operators apply to all tags that follow them until another operator is specified.
            The NOT operator applies only to the immediately following tag.
            For example:

            .. code-block:: py

                ["__$all", "a", "b", "c", "__$or", "d", "__$not", "e", "__$and", "__$or", "f", "g"]

            This example means ("a" AND "b" AND "c" AND ("d" OR NOT "e") AND ("f" OR "g")).
            See https://clear.ml/docs/latest/docs/clearml_sdk/task_sdk/#tag-filters for more information.
        :param bool allow_archived: Only applicable if *not* using specific ``task_id``,
            If True (default), allow to return archived Tasks, if False filter out archived Tasks
        :param bool task_filter: Only applicable if *not* using specific ``task_id``,
            Pass additional query filters, on top of project/name. See details in Task.get_tasks.

        :return: The Task specified by ID, or project name / experiment name combination.
        :rtype: Task
        """
        return cls.__get_task(
            task_id=task_id,
            project_name=project_name,
            task_name=task_name,
            tags=tags,
            include_archived=allow_archived,
            task_filter=task_filter,
        )

    @classmethod
    def get_tasks(
        cls,
        task_ids: Optional[Sequence[str]] = None,
        project_name: Optional[Union[Sequence[str], str]] = None,
        task_name: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        allow_archived: bool = True,
        task_filter: Optional[Dict] = None,
    ) -> List[TaskInstance]:
        """
        Get a list of Tasks objects matching the queries/filters:

        - A list of specific Task IDs.
        - Filter Tasks based on specific fields:
            project name (including partial match), task name (including partial match), tags
            Apply Additional advanced filtering with `task_filter`

        .. note::
            This function returns the most recent 500 tasks. If you wish to retrieve older tasks
            use ``Task.query_tasks()``

        :param list(str) task_ids: The IDs (system UUID) of experiments to get.
            If ``task_ids`` specified, then ``project_name`` and ``task_name`` are ignored.
        :param str project_name: The project name of the Tasks to get. To get the experiment
            in all projects, use the default value of ``None``. (Optional)
            Use a list of strings for multiple optional project names.
        :param str task_name: The full name or partial name of the Tasks to match within the specified
            ``project_name`` (or all projects if ``project_name`` is ``None``).
            This method supports regular expressions for name matching (if you wish to match special characters and
            avoid any regex behaviour, use re.escape()). (Optional)
            To match an exact task name (i.e. not partial matching),
            add ^/$ at the beginning/end of the string, for example: "^exact_task_name_here$"
        :param list tags: Filter based on the requested list of tags (strings). To exclude a tag add "-" prefix to the
            tag. Example: ``["best", "-debug"]``.
            The default behaviour is to join all tags with a logical "OR" operator.
            To join all tags with a logical "AND" operator instead, use "__$all" as the first string, for example:

            .. code-block:: py

                ["__$all", "best", "experiment", "ever"]

            To join all tags with AND, but exclude a tag use "__$not" before the excluded tag, for example:

            .. code-block:: py

                ["__$all", "best", "experiment", "ever", "__$not", "internal", "__$not", "test"]

            The "OR" and "AND" operators apply to all tags that follow them until another operator is specified.
            The NOT operator applies only to the immediately following tag.
            For example:

            .. code-block:: py

                ["__$all", "a", "b", "c", "__$or", "d", "__$not", "e", "__$and", "__$or", "f", "g"]

            This example means ("a" AND "b" AND "c" AND ("d" OR NOT "e") AND ("f" OR "g")).
            See https://clear.ml/docs/latest/docs/clearml_sdk/task_sdk/#tag-filters for more information.
        :param bool allow_archived: If True (default), allow to return archived Tasks, if False filter out archived Tasks
        :param dict task_filter: filter and order Tasks.
            See :class:`.backend_api.service.v?.tasks.GetAllRequest` for details; the ? needs to be replaced by the appropriate version.

          - ``parent`` - (str) filter by parent task-id matching
          - ``search_text`` - (str) free text search (in task fields comment/name/id)
          - ``status`` - List[str] List of valid statuses. Options are: "created", "queued", "in_progress", "stopped", "published", "publishing", "closed", "failed", "completed", "unknown"
          - ``type`` - List[str] List of valid task types. Options are: 'training', 'testing', 'inference', 'data_processing', 'application', 'monitor', 'controller', 'optimizer', 'service', 'qc'. 'custom'
          - ``user`` - List[str] Filter based on Task's user owner, provide list of valid user IDs.
          - ``order_by`` - List[str] List of field names to order by. When ``search_text`` is used. Use '-' prefix to specify descending order. Optional, recommended when using page. Example: ``order_by=['-last_update']``
          - ``_all_`` - dict(fields=[], pattern='')  Match string `pattern` (regular expression) appearing in All `fields`. Example: dict(fields=['script.repository'], pattern='github.com/user')
          - ``_any_`` - dict(fields=[], pattern='')  Match string `pattern` (regular expression) appearing in any of the `fields`. Example: dict(fields=['comment', 'name'], pattern='my comment')
          - Examples - ``{'status': ['stopped'], 'order_by': ["-last_update"]}`` , ``{'order_by'=['-last_update'], '_all_'=dict(fields=['script.repository'], pattern='github.com/user'))``

        :return: The Tasks specified by the parameter combinations (see the parameters).
        :rtype: List[Task]
        """
        task_filter = task_filter or {}
        if not allow_archived:
            task_filter["system_tags"] = (task_filter.get("system_tags") or []) + ["-{}".format(cls.archived_tag)]

        return cls.__get_tasks(
            task_ids=task_ids, project_name=project_name, tags=tags, task_name=task_name, **task_filter
        )

    @classmethod
    def query_tasks(
        cls,
        project_name: Optional[Union[Sequence[str], str]] = None,
        task_name: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        additional_return_fields: Optional[Sequence[str]] = None,
        task_filter: Optional[Dict] = None,
    ) -> Union[List[str], List[Dict[str, str]]]:
        """
        Get a list of Tasks ID matching the specific query/filter.
        Notice, if `additional_return_fields` is specified, returns a list of
        dictionaries with requested fields (dict per Task)

        :param str project_name: The project name of the Tasks to get. To get the experiment
            in all projects, use the default value of ``None``. (Optional)
            Use a list of strings for multiple optional project names.
        :param str task_name: The full name or partial name of the Tasks to match within the specified
            ``project_name`` (or all projects if ``project_name`` is ``None``).
            This method supports regular expressions for name matching (if you wish to match special characters and
            avoid any regex behaviour, use re.escape()). (Optional)
        :param str project_name: project name (str) the task belongs to (use None for all projects)
        :param str task_name: task name (str) within the selected project
            Return any partial match of task_name, regular expressions matching is also supported.
            If None is passed, returns all tasks within the project
        :param list tags: Filter based on the requested list of tags (strings).
            To exclude a tag add "-" prefix to the tag. Example: ``["best", "-debug"]``.
            The default behaviour is to join all tags with a logical "OR" operator.
            To join all tags with a logical "AND" operator instead, use "__$all" as the first string, for example:

            .. code-block:: py

                ["__$all", "best", "experiment", "ever"]

            To join all tags with AND, but exclude a tag use "__$not" before the excluded tag, for example:

            .. code-block:: py

                ["__$all", "best", "experiment", "ever", "__$not", "internal", "__$not", "test"]

            The "OR" and "AND" operators apply to all tags that follow them until another operator is specified.
            The NOT operator applies only to the immediately following tag.
            For example:

            .. code-block:: py

                ["__$all", "a", "b", "c", "__$or", "d", "__$not", "e", "__$and", "__$or", "f", "g"]

            This example means ("a" AND "b" AND "c" AND ("d" OR NOT "e") AND ("f" OR "g")).
            See https://clear.ml/docs/latest/docs/clearml_sdk/task_sdk/#tag-filters for more information.
        :param list additional_return_fields: Optional, if not provided return a list of Task IDs.
            If provided return dict per Task with the additional requested fields.
            Example: ``returned_fields=['last_updated', 'user', 'script.repository']`` will return a list of dict:
            ``[{'id': 'task_id', 'last_update': datetime.datetime(), 'user': 'user_id', 'script.repository': 'https://github.com/user/'}, ]``
        :param dict task_filter: filter and order Tasks.
            See :class:`.backend_api.service.v?.tasks.GetAllRequest` for details; the ? needs to be replaced by the appropriate version.

          - ``parent`` - (str) filter by parent task-id matching
          - ``search_text`` - (str) free text search (in task fields comment/name/id)
          - ``status`` - List[str] List of valid statuses. Options are: "created", "queued", "in_progress", "stopped", "published", "publishing", "closed", "failed", "completed", "unknown"
          - ``type`` - List[Union[str, TaskTypes]] List of valid task types. Options are: 'training', 'testing', 'inference', 'data_processing', 'application', 'monitor', 'controller', 'optimizer', 'service', 'qc'. 'custom'
          - ``user`` - List[str] Filter based on Task's user owner, provide list of valid user IDs.
          - ``order_by`` - List[str] List of field names to order by. When search_text is used. Use '-' prefix to specify descending order. Optional, recommended when using page. Example: ``order_by=['-last_update']``
          - ``_all_`` - dict(fields=[], pattern='')  Match string ``pattern`` (regular expression) appearing in All `fields`. ``dict(fields=['script.repository'], pattern='github.com/user')``
          - ``_any_`` - dict(fields=[], pattern='')  Match string `pattern` (regular expression) appearing in Any of the `fields`. `dict(fields=['comment', 'name'], pattern='my comment')`
          - Examples: ``{'status': ['stopped'], 'order_by': ["-last_update"]}``, ``{'order_by'=['-last_update'], '_all_'=dict(fields=['script.repository'], pattern='github.com/user')}``

        :return: The Tasks specified by the parameter combinations (see the parameters).
        """
        task_filter = task_filter or {}
        if tags:
            task_filter["tags"] = (task_filter.get("tags") or []) + list(tags)
        return_fields = {}
        if additional_return_fields:
            return_fields = set(list(additional_return_fields) + ["id"])
            task_filter["only_fields"] = (task_filter.get("only_fields") or []) + list(return_fields)

        if task_filter.get("type"):
            task_filter["type"] = [str(task_type) for task_type in task_filter["type"]]

        results = cls._query_tasks(project_name=project_name, task_name=task_name, **task_filter)
        return (
            [t.id for t in results]
            if not additional_return_fields
            else [
                {
                    k: cls._get_data_property(prop_path=k, data=r, raise_on_error=False, log_on_error=False)
                    for k in return_fields
                }
                for r in results
            ]
        )

    @property
    def output_uri(self) -> str:
        """
        The storage / output url for this task. This is the default location for output models and other artifacts.

        :return: The url string.
        """
        return self.storage_uri

    @property
    def last_worker(self) -> str:
        """
        ID of last worker that handled the task.

        :return: The worker ID.
        """
        return self._data.last_worker

    @output_uri.setter
    def output_uri(self, value: Union[str, bool]) -> None:
        """
        Set the storage / output url for this task. This is the default location for output models and other artifacts.

        :param str/bool value: The value to set for output URI. Can be either a bucket link, True for default server
            or False. Check Task.init reference docs for more info (output_uri is a parameter).
        """

        # check if this is boolean
        if value is False:
            value = None
        elif value is True:
            value = str(self.__default_output_uri or self._get_default_report_storage_uri())

        # check if we have the correct packages / configuration
        if value and value != self.storage_uri:
            from .storage.helper import StorageHelper

            helper = StorageHelper.get(value)
            if not helper:
                raise ValueError(
                    "Could not get access credentials for '{}' "
                    ", check configuration file ~/clearml.conf".format(value)
                )
            helper.check_write_permissions(value)
        self.storage_uri = value

    @property
    def artifacts(self) -> Dict[str, Artifact]:
        """
        A read-only dictionary of Task artifacts (name, artifact).

        :return: The artifacts.
        """
        if not Session.check_min_api_version("2.3"):
            return ReadOnlyDict()
        artifacts_pairs = []
        if self.data.execution and self.data.execution.artifacts:
            artifacts_pairs = [(a.key, Artifact(a)) for a in self.data.execution.artifacts]
        if self._artifacts_manager:
            artifacts_pairs += list(self._artifacts_manager.registered_artifacts.items())
        return ReadOnlyDict(artifacts_pairs)

    @property
    def models(self) -> Mapping[str, Sequence[Model]]:
        """
        Read-only dictionary of the Task's loaded/stored models.

        :return: A dictionary-like object with "input"/"output" keys and input/output properties, pointing to a
            list-like object containing Model objects. Each list-like object also acts as a dictionary, mapping
            model name to an appropriate model instance.

            Get input/output models:

            .. code-block:: py

                task.models.input
                task.models["input"]

                task.models.output
                task.models["output"]

            Get the last output model:

            .. code-block:: py

                task.models.output[-1]

            Get a model by name:

            .. code-block:: py

                task.models.output["model name"]
        """
        return self.get_models()

    @property
    def logger(self) -> Logger:
        """
        Get a Logger object for reporting, for this task context. You can view all Logger report output associated with
        the Task for which this method is called, including metrics, plots, text, tables, and images, in the
        **ClearML Web-App (UI)**.

        :return: The Logger object for the current Task (experiment).
        """
        return self.get_logger()

    @classmethod
    def clone(
        cls,
        source_task: Optional[Union["Task", str]] = None,
        name: Optional[str] = None,
        comment: Optional[str] = None,
        parent: Optional[str] = None,
        project: Optional[str] = None,
    ) -> TaskInstance:
        """
        Create a duplicate (a clone) of a Task (experiment). The status of the cloned Task is ``Draft``
        and modifiable.

        Use this method to manage experiments and for autoML.

        :param str source_task: The Task to clone. Specify a Task object or a  Task ID. (Optional)
        :param str name: The name of the new cloned Task. (Optional)
        :param str comment: A comment / description for the new cloned Task. (Optional)
        :param str parent: The ID of the parent Task of the new Task.

          - If ``parent`` is not specified, then ``parent`` is set to ``source_task.parent``.
          - If ``parent`` is not specified and ``source_task.parent`` is not available, then ``parent`` set to ``source_task``.

        :param str project: The ID of the project in which to create the new Task.
            If ``None``, the new task inherits the original Task's project. (Optional)

        :return: The new cloned Task (experiment).
        :rtype: Task
        """
        assert isinstance(source_task, (six.string_types, Task))
        if not Session.check_min_api_version("2.4"):
            raise ValueError(
                "ClearML-server does not support DevOps features, upgrade clearml-server to 0.12.0 or above"
            )

        task_id = source_task if isinstance(source_task, six.string_types) else source_task.id
        if not parent:
            if isinstance(source_task, six.string_types):
                source_task = cls.get_task(task_id=source_task)
            parent = source_task.id if not source_task.parent else source_task.parent
        elif isinstance(parent, Task):
            parent = parent.id

        cloned_task_id = cls._clone_task(
            cloned_task_id=task_id,
            name=name,
            comment=comment,
            parent=parent,
            project=project,
        )
        cloned_task = cls.get_task(task_id=cloned_task_id)
        return cloned_task

    @classmethod
    def enqueue(
        cls,
        task: Union["Task", str],
        queue_name: Optional[str] = None,
        queue_id: Optional[str] = None,
        force: bool = False,
    ) -> Any:
        """
        Enqueue a Task for execution, by adding it to an execution queue.

        .. note::
           A worker daemon must be listening at the queue for the worker to fetch the Task and execute it,
           see "ClearML Agent" in the ClearML Documentation.

        :param Task/str task: The Task to enqueue. Specify a Task object or  Task ID.
        :param str queue_name: The name of the queue. If not specified, then ``queue_id`` must be specified.
        :param str queue_id: The ID of the queue. If not specified, then ``queue_name`` must be specified.
        :param bool force: If True, reset the Task if necessary before enqueuing it

        :return: An enqueue JSON response.

            .. code-block:: javascript

               {
                    "queued": 1,
                    "updated": 1,
                    "fields": {
                        "status": "queued",
                        "status_reason": "",
                        "status_message": "",
                        "status_changed": "2020-02-24T15:05:35.426770+00:00",
                        "last_update": "2020-02-24T15:05:35.426770+00:00",
                        "execution.queue": "2bd96ab2d9e54b578cc2fb195e52c7cf"
                        }
                }

            - ``queued``  - The number of Tasks enqueued (an integer or ``null``).
            - ``updated`` - The number of Tasks updated (an integer or ``null``).
            - ``fields``

              - ``status`` - The status of the experiment.
              - ``status_reason`` - The reason for the last status change.
              - ``status_message`` - Information about the status.
              - ``status_changed`` - The last status change date and time (ISO 8601 format).
              - ``last_update`` - The last Task update time, including Task creation, update, change, or events for this task (ISO 8601 format).
              - ``execution.queue`` - The ID of the queue where the Task is enqueued. ``null`` indicates not enqueued.

        """
        assert isinstance(task, (six.string_types, Task))
        if not Session.check_min_api_version("2.4"):
            raise ValueError(
                "ClearML-server does not support DevOps features, upgrade clearml-server to 0.12.0 or above"
            )

        # make sure we have wither name ot id
        mutually_exclusive(queue_name=queue_name, queue_id=queue_id)

        task_id = task if isinstance(task, six.string_types) else task.id
        session = cls._get_default_session()
        if not queue_id:
            queue_id = get_queue_id(session, queue_name)
            if not queue_id:
                raise ValueError('Could not find queue named "{}"'.format(queue_name))

        req = tasks.EnqueueRequest(task=task_id, queue=queue_id)
        exception = None
        res = None
        try:
            res = cls._send(session=session, req=req)
            ok = res.ok()
        except Exception as e:
            exception = e
            ok = False
        if not ok:
            if not force:
                if res:
                    raise ValueError(res.response)
                raise exception
            task = cls.get_task(task_id=task) if isinstance(task, str) else task
            task.reset(set_started_on_success=False, force=True)
            req = tasks.EnqueueRequest(task=task_id, queue=queue_id)
            res = cls._send(session=session, req=req)
            if not res.ok():
                raise ValueError(res.response)
        resp = res.response
        return resp

    @classmethod
    def get_num_enqueued_tasks(cls, queue_name: Optional[str] = None, queue_id: Optional[str] = None) -> int:
        """
        Get the number of tasks enqueued in a given queue.

        :param queue_name: The name of the queue. If not specified, then ``queue_id`` must be specified
        :param queue_id: The ID of the queue. If not specified, then ``queue_name`` must be specified

        :return: The number of tasks enqueued in the given queue
        """
        if not Session.check_min_api_server_version("2.20", raise_error=True):
            raise ValueError("You version of clearml-server does not support the 'queues.get_num_entries' endpoint")
        mutually_exclusive(queue_name=queue_name, queue_id=queue_id)
        session = cls._get_default_session()
        if not queue_id:
            queue_id = get_queue_id(session, queue_name)
            if not queue_id:
                raise ValueError('Could not find queue named "{}"'.format(queue_name))
        result = get_num_enqueued_tasks(session, queue_id)
        if result is None:
            raise ValueError("Could not query the number of enqueued tasks in queue with ID {}".format(queue_id))
        return result

    @classmethod
    def dequeue(cls, task: Union["Task", str]) -> Any:
        """
        Dequeue (remove) a Task from an execution queue.

        :param Task/str task: The Task to dequeue. Specify a Task object or  Task ID.

        :return: A dequeue JSON response.

        .. code-block:: javascript

           {
                "dequeued": 1,
                "updated": 1,
                "fields": {
                    "status": "created",
                    "status_reason": "",
                    "status_message": "",
                    "status_changed": "2020-02-24T16:43:43.057320+00:00",
                    "last_update": "2020-02-24T16:43:43.057320+00:00",
                    "execution.queue": null
                    }
            }

        - ``dequeued``  - The number of Tasks enqueued (an integer or ``null``).
        - ``fields``

          - ``status`` - The status of the experiment.
          - ``status_reason`` - The reason for the last status change.
          - ``status_message`` - Information about the status.
          - ``status_changed`` - The last status change date and time in ISO 8601 format.
          - ``last_update`` - The last time the Task was created, updated,
                changed, or events for this task were reported.
          - ``execution.queue`` - The ID of the queue where the Task is enqueued. ``null`` indicates not enqueued.

        - ``updated`` - The number of Tasks updated (an integer or ``null``).

        """
        assert isinstance(task, (six.string_types, Task))
        if not Session.check_min_api_version("2.4"):
            raise ValueError(
                "ClearML-server does not support DevOps features, upgrade clearml-server to 0.12.0 or above"
            )

        task_id = task if isinstance(task, six.string_types) else task.id
        session = cls._get_default_session()
        req = tasks.DequeueRequest(task=task_id)
        res = cls._send(session=session, req=req)
        resp = res.response
        return resp

    def set_progress(self, progress: int) -> ():
        """
        Sets Task's progress (0 - 100)
        Progress is a field computed and reported by the user.

        :param progress: numeric value (0 - 100)
        """
        if not isinstance(progress, int) or progress < 0 or progress > 100:
            self.log.warning("Can't set progress {} as it is not and int between 0 and 100".format(progress))
            return
        self._set_runtime_properties({"progress": str(progress)})

    def get_progress(self) -> Optional[int]:
        """
        Gets Task's progress (0 - 100)

        :return: Task's progress as an int.
            In case the progress doesn't exist, None will be returned
        """
        progress = self._get_runtime_properties().get("progress")
        if progress is None or not progress.isnumeric():
            return None
        return int(progress)

    def add_tags(self, tags: Union[Sequence[str], str]) -> None:
        """
        Add Tags to this task. Old tags are not deleted. When executing a Task (experiment) remotely,
        this method has no effect.

        :param tags: A list of tags which describe the Task to add.
        """

        if isinstance(tags, six.string_types):
            tags = tags.split(" ")

        self.data.tags = list(set((self.data.tags or []) + tags))
        self._edit(tags=self.data.tags)

    def connect(
        self,
        mutable: Any,
        name: Optional[str] = None,
        ignore_remote_overrides: bool = False,
    ) -> Any:
        """
        Connect an object to a Task object. This connects an experiment component (part of an experiment) to the
        experiment. For example, an experiment component can be a valid object containing some hyperparameters, or a :class:`Model`.
        When running remotely, the value of the connected object is overridden by the corresponding value found
        under the experiment's UI/backend (unless `ignore_remote_overrides` is True).

        :param object mutable: The experiment component to connect. The object must be one of the following types:

          - argparse - An argparse object for parameters.
          - dict - A dictionary for parameters. Note: only keys of type `str` are supported.
          - TaskParameters - A TaskParameters object.
          - :class:`Model` - A model object for initial model warmup, or for model update/snapshot uploading. In practice the model should be either :class:`InputModel` or :class:`OutputModel`.
          - type - A Class type, storing all class properties (excluding '_' prefixed properties).
          - object - A class instance, storing all instance properties (excluding '_' prefixed properties).

        :param str name: A section name associated with the connected object, if 'name' is None defaults to 'General'
            Currently, `name` is only supported for `dict` and `TaskParameter` objects, and should be omitted for the other supported types. (Optional)
            For example, by setting `name='General'` the connected dictionary will be under the General section in the hyperparameters section.
            While by setting `name='Train'` the connected dictionary will be under the Train section in the hyperparameters section.

        :param ignore_remote_overrides: If True, ignore UI/backend overrides when running remotely.
            Default is False, meaning that any changes made in the UI/backend will be applied in remote execution.

        :return: It will return the same object that was passed as the `mutable` argument to the method, except if the type of the object is dict.
                 For dicts the :meth:`Task.connect` will return the dict decorated as a `ProxyDictPostWrite`.
                 This is done to allow propagating the updates from the connected object.

        :raise: Raises an exception if passed an unsupported object.
        """
        # input model connect and task parameters will handle this instead
        if not isinstance(mutable, (InputModel, TaskParameters)):
            ignore_remote_overrides = self._handle_ignore_remote_overrides(
                (name or "General") + "/_ignore_remote_overrides_",
                ignore_remote_overrides,
            )
        # dispatching by match order
        dispatch = (
            (OutputModel, self._connect_output_model),
            (InputModel, self._connect_input_model),
            (ArgumentParser, self._connect_argparse),
            (dict, self._connect_dictionary),
            (TaskParameters, self._connect_task_parameters),
            (type, self._connect_object),
            (object, self._connect_object),
        )

        multi_config_support = Session.check_min_api_version("2.9")
        if multi_config_support and not name and not isinstance(mutable, (OutputModel, InputModel)):
            name = self._default_configuration_section_name

        if not multi_config_support and name and name != self._default_configuration_section_name:
            raise ValueError(
                "Multiple configurations is not supported with the current 'clearml-server', "
                "please upgrade to the latest version"
            )

        for mutable_type, method in dispatch:
            if isinstance(mutable, mutable_type):
                return method(mutable, name=name, ignore_remote_overrides=ignore_remote_overrides)

        raise Exception("Unsupported mutable type %s: no connect function found" % type(mutable).__name__)

    def set_packages(self, packages: Union[str, Path, Sequence[str]]) -> ():
        """
        Manually specify a list of required packages or a local requirements.txt file. Note that this will
        overwrite all existing packages.

        When running remotely this call is ignored

        :param packages: The list of packages or the path to the requirements.txt file.

            Example: ``["tqdm>=2.1", "scikit-learn"]`` or ``"./requirements.txt"`` or ``""``
            Use an empty string (packages="") to clear the requirements section (remote execution will use
                requirements.txt from the git repository if the file exists)
        """
        if running_remotely() or packages is None:
            return
        self._wait_for_repo_detection(timeout=300.0)

        if packages and isinstance(packages, (str, Path)) and Path(packages).is_file():
            with open(Path(packages).as_posix(), "rt") as f:
                # noinspection PyProtectedMember
                self._update_requirements([line.strip() for line in f.readlines()])
            return

        # noinspection PyProtectedMember
        self._update_requirements(packages or "")

    def set_repo(
        self,
        repo: Optional[str] = None,
        branch: Optional[str] = None,
        commit: Optional[str] = None,
    ) -> ():
        """
        Specify a repository to attach to the function.
        Allow users to execute the task inside the specified repository, enabling them to load modules/script
        from the repository. Notice the execution work directory will be the repository root folder.
        Supports both git repo url link, and local repository path (automatically converted into the remote
        git/commit as is currently checkout).
        Example remote url: "https://github.com/user/repo.git".
        Example local repo copy: "./repo" - will automatically store the remote
        repo url and commit ID based on the locally cloned copy.
        When executing remotely, this call will not override the repository data (it is ignored)

        :param repo: Optional, remote URL for the repository to use, OR path to local copy of the git repository.
            Use an empty string to clear the repo.
            Example: "https://github.com/allegroai/clearml.git" or "~/project/repo" or ""
        :param branch: Optional, specify the remote repository branch (Ignored, if local repo path is used).
            Use an empty string to clear the branch.
        :param commit: Optional, specify the repository commit ID (Ignored, if local repo path is used).
            Use an empty string to clear the commit.
        """
        if running_remotely():
            return
        self._wait_for_repo_detection(timeout=300.0)
        with self._edit_lock:
            self.reload()
            if repo is not None:
                # we cannot have None on the value itself
                self.data.script.repository = repo or ""
            if branch is not None:
                # we cannot have None on the value itself
                self.data.script.branch = branch or ""
            if commit is not None:
                # we cannot have None on the value itself
                self.data.script.version_num = commit or ""
            self._edit(script=self.data.script)

    def get_requirements(self) -> RequirementsDict:
        """
        Get the task's requirements

        :return: A `RequirementsDict` object that holds the `pip`, `conda`, `orig_pip` requirements.
        """
        if not running_remotely() and self.is_main_task():
            self._wait_for_repo_detection(timeout=300.0)
        requirements_dict = RequirementsDict()
        requirements_dict.update(self.data.script.requirements)
        return requirements_dict

    def connect_configuration(
        self,
        configuration: Union[Mapping, list, Path, str],
        name: Optional[str] = None,
        description: Optional[str] = None,
        ignore_remote_overrides: bool = False,
    ) -> Union[dict, Path, str]:
        """
        Connect a configuration dictionary or configuration file (pathlib.Path / str) to a Task object.
        This method should be called before reading the configuration file.

        For example, a local file:

        .. code-block:: py

           config_file = task.connect_configuration(config_file)
           my_params = json.load(open(config_file,'rt'))

        A parameter dictionary/list:

        .. code-block:: py

           my_params = task.connect_configuration(my_params)

        When running remotely, the value of the connected configuration is overridden by the corresponding value found
        under the experiment's UI/backend (unless `ignore_remote_overrides` is True).

        :param configuration: The configuration. This is usually the configuration used in the model training process.
            Specify one of the following:

          - A dictionary/list - A dictionary containing the configuration. ClearML stores the configuration in
            the **ClearML Server** (backend), in a HOCON format (JSON-like format) which is editable.
          - A ``pathlib2.Path`` string - A path to the configuration file. ClearML stores the content of the file.
            A local path must be relative path. When executing a Task remotely in a worker, the contents brought
            from the **ClearML Server** (backend) overwrites the contents of the file.

        :param str name: Configuration section name. default: 'General'
            Allowing users to store multiple configuration dicts/files

        :param str description: Configuration section description (text). default: None

        :param bool ignore_remote_overrides: If True, ignore UI/backend overrides when running remotely.
            Default is False, meaning that any changes made in the UI/backend will be applied in remote execution.

        :return: If a dictionary is specified, then a dictionary is returned. If pathlib2.Path / string is
            specified, then a path to a local configuration file is returned. Configuration object.
        """
        ignore_remote_overrides = self._handle_ignore_remote_overrides(
            (name or "General") + "/_ignore_remote_overrides_config_",
            ignore_remote_overrides,
        )
        pathlib_Path = None  # noqa
        cast_Path = Path
        if not isinstance(configuration, (dict, list, Path, six.string_types)):
            try:
                from pathlib import Path as pathlib_Path  # noqa
            except ImportError:
                pass
            if not pathlib_Path or not isinstance(configuration, pathlib_Path):
                raise ValueError(
                    "connect_configuration supports `dict`, `str` and 'Path' types, "
                    "{} is not supported".format(type(configuration))
                )
        if pathlib_Path and isinstance(configuration, pathlib_Path):
            cast_Path = pathlib_Path

        multi_config_support = Session.check_min_api_version("2.9")
        if multi_config_support and not name:
            name = self._default_configuration_section_name

        if not multi_config_support and name and name != self._default_configuration_section_name:
            raise ValueError(
                "Multiple configurations is not supported with the current 'clearml-server', "
                "please upgrade to the latest version"
            )

        # parameter dictionary
        if isinstance(
            configuration,
            (
                dict,
                list,
            ),
        ):

            def _update_config_dict(task, config_dict):
                if multi_config_support:
                    # noinspection PyProtectedMember
                    task._set_configuration(
                        name=name,
                        description=description,
                        config_type="dictionary",
                        config_dict=config_dict,
                    )
                else:
                    # noinspection PyProtectedMember
                    task._set_model_config(config_dict=config_dict)

            def get_dev_config(configuration_: Union[dict, list, Path, str]) -> Union[dict, ProxyDictPostWrite]:
                if multi_config_support:
                    self._set_configuration(
                        name=name,
                        description=description,
                        config_type="dictionary",
                        config_dict=configuration_,
                    )
                else:
                    self._set_model_config(config_dict=configuration)
                if isinstance(configuration_, dict):
                    configuration_ = ProxyDictPostWrite(self, _update_config_dict, configuration_)
                return configuration_

            if (
                not running_remotely()
                or not (self.is_main_task() or self._is_remote_main_task())
                or ignore_remote_overrides
            ):
                configuration = get_dev_config(configuration)
            else:
                # noinspection PyBroadException
                try:
                    remote_configuration = (
                        self._get_configuration_dict(name=name)
                        if multi_config_support
                        else self._get_model_config_dict()
                    )
                except Exception:
                    remote_configuration = None

                if remote_configuration is None:
                    LoggerRoot.get_base_logger().warning(
                        "Could not retrieve remote configuration named '{}'\n"
                        "Using default configuration: {}".format(name, str(configuration))
                    )
                    # update back configuration section
                    if multi_config_support:
                        self._set_configuration(
                            name=name,
                            description=description,
                            config_type="dictionary",
                            config_dict=configuration,
                        )
                    return configuration

                if not remote_configuration:
                    configuration = get_dev_config(configuration)
                elif isinstance(configuration, dict):
                    configuration.clear()
                    configuration.update(remote_configuration)
                    configuration = ProxyDictPreWrite(False, False, **configuration)
                elif isinstance(configuration, list):
                    configuration.clear()
                    configuration.extend(remote_configuration)

            return configuration

        # it is a path to a local file
        if (
            not running_remotely()
            or not (self.is_main_task() or self._is_remote_main_task())
            or ignore_remote_overrides
        ):
            # check if not absolute path
            configuration_path = cast_Path(configuration)
            if not configuration_path.is_file():
                ValueError("Configuration file does not exist")
            try:
                with open(configuration_path.as_posix(), "rt") as f:
                    configuration_text = f.read()
            except Exception:
                raise ValueError(
                    "Could not connect configuration file {}, file could not be read".format(
                        configuration_path.as_posix()
                    )
                )
            if multi_config_support:
                self._set_configuration(
                    name=name,
                    description=description,
                    config_type=(
                        configuration_path.suffixes[-1].lstrip(".")
                        if configuration_path.suffixes and configuration_path.suffixes[-1]
                        else "file"
                    ),
                    config_text=configuration_text,
                )
            else:
                self._set_model_config(config_text=configuration_text)
            return configuration
        else:
            configuration_text = (
                self._get_configuration_text(name=name) if multi_config_support else self._get_model_config_text()
            )
            if configuration_text is None:
                LoggerRoot.get_base_logger().warning(
                    "Could not retrieve remote configuration named '{}'\n"
                    "Using default configuration: {}".format(name, str(configuration))
                )
                # update back configuration section
                if multi_config_support:
                    configuration_path = cast_Path(configuration)
                    if configuration_path.is_file():
                        with open(configuration_path.as_posix(), "rt") as f:
                            configuration_text = f.read()

                        self._set_configuration(
                            name=name,
                            description=description,
                            config_type=(
                                configuration_path.suffixes[-1].lstrip(".")
                                if configuration_path.suffixes and configuration_path.suffixes[-1]
                                else "file"
                            ),
                            config_text=configuration_text,
                        )
                return configuration

            configuration_path = cast_Path(configuration)
            fd, local_filename = mkstemp(
                prefix="clearml_task_config_",
                suffix=(configuration_path.suffixes[-1] if configuration_path.suffixes else ".txt"),
            )
            with open(fd, "w") as f:
                f.write(configuration_text)
            return cast_Path(local_filename) if isinstance(configuration, cast_Path) else local_filename

    def connect_label_enumeration(
        self, enumeration: Dict[str, int], ignore_remote_overrides: bool = False
    ) -> Dict[str, int]:
        """
        Connect a label enumeration dictionary to a Task (experiment) object.

        Later, when creating an output model, the model will include the label enumeration dictionary.

        :param dict enumeration: A label enumeration dictionary of string (label) to integer (value) pairs.

            For example:

            .. code-block:: javascript

               {
                    "background": 0,
                    "person": 1
               }

        :param ignore_remote_overrides: If True, ignore UI/backend overrides when running remotely.
            Default is False, meaning that any changes made in the UI/backend will be applied in remote execution.
        :return: The label enumeration dictionary (JSON).
        """
        ignore_remote_overrides = self._handle_ignore_remote_overrides(
            "General/_ignore_remote_overrides_label_enumeration_",
            ignore_remote_overrides,
        )
        if not isinstance(enumeration, dict):
            raise ValueError(
                "connect_label_enumeration supports only `dict` type, {} is not supported".format(type(enumeration))
            )

        if (
            not running_remotely()
            or not (self.is_main_task() or self._is_remote_main_task())
            or ignore_remote_overrides
        ):
            self.set_model_label_enumeration(enumeration)
        else:
            # pop everything
            enumeration.clear()
            enumeration.update(self.get_labels_enumeration())
        return enumeration

    def get_logger(self) -> Logger:
        """
        Get a Logger object for reporting, for this task context. You can view all Logger report output associated with
        the Task for which this method is called, including metrics, plots, text, tables, and images, in the
        **ClearML Web-App (UI)**.

        :return: The Logger for the Task (experiment).
        """
        return self._get_logger(auto_connect_streams=self._log_to_backend)

    def launch_multi_node(
        self,
        total_num_nodes: int,
        port: Optional[int] = 29500,
        queue: Optional[str] = None,
        wait: bool = False,
        addr: Optional[str] = None,
        devices: Optional[Union[int, Sequence[int]]] = None,
        hide_children=False,  # bool
    ):
        """
        Enqueue multiple clones of the current task to a queue, allowing the task
        to be ran by multiple workers in parallel. Each task running this way is called a node.
        Each node has a rank The node that initialized the execution of the other nodes
        is called the `master node` and it has a rank equal to 0.

        A dictionary named `multi_node_instance` will be connected to the tasks.
        One can use this dictionary to modify the behaviour of this function when running remotely.
        The contents of this dictionary correspond to the parameters of this function, and they are:
        - `total_num_nodes` - the total number of nodes, including the master node
        - `queue` - the queue to enqueue the nodes to

        The following environment variables, will be set:
        - `MASTER_ADDR` - the address of the machine that the master node is running on
        - `MASTER_PORT` - the open port of the machine that the master node is running on
        - `WORLD_SIZE` - the total number of nodes, including the master
        - `RANK` - the rank of the current node (master has rank 0)

        One may use this function in conjuction with PyTorch's distributed communication package.
        Note that `Task.launch_multi_node` should be called before `torch.distributed.init_process_group`.
        For example:

        .. code-block:: py

            from clearml import Task
            import torch
            import torch.distributed as dist

            def run(rank, size):
                print('World size is ', size)
                tensor = torch.zeros(1)
                if rank == 0:
                    for i in range(1, size):
                        tensor += 1
                        dist.send(tensor=tensor, dst=i)
                        print('Sending from rank ', rank, ' to rank ', i, ' data: ', tensor[0])
                else:
                    dist.recv(tensor=tensor, src=0)
                    print('Rank ', rank, ' received data: ', tensor[0])

            if __name__ == '__main__':
                task = Task.init('some_name', 'some_name')
                task.execute_remotely(queue_name='queue')
                config = task.launch_multi_node(4)
                dist.init_process_group('gloo')
                run(config.get('node_rank'), config.get('total_num_nodes'))

        When using the ClearML cloud autoscaler apps, one needs to make sure the nodes can reach eachother.
        The machines need to be in the same security group, the `MASTER_PORT` needs to be exposed and the
        `MASTER_ADDR` needs to be the right private ip of the instance the master is running on.
        For example, to achieve this, one can set the following Docker arguments in the `Additional ClearML Configuration` section:

        .. code-block:: py

            agent.extra_docker_arguments=["--ipc=host", "--network=host", "-p", "29500:29500", "--env", "CLEARML_MULTI_NODE_MASTER_DEF_ADDR=`hostname -I | awk '{print $1}'`"]`

        :param total_num_nodes: The total number of nodes to be enqueued, including the master node,
            which should already be enqueued when running remotely
        :param port: Port opened by the master node. If the environment variable ``CLEARML_MULTI_NODE_MASTER_DEF_PORT``
            is set, the value of this parameter will be set to the one defined in ``CLEARML_MULTI_NODE_MASTER_DEF_PORT``.
            If ``CLEARML_MULTI_NODE_MASTER_DEF_PORT`` doesn't exist, but ``MASTER_PORT`` does, then the value of this
            parameter will be set to the one defined in ``MASTER_PORT``. If neither environment variables exist,
            the value passed to the parameter will be used
        :param queue: The queue to enqueue the nodes to. Can be different from the queue the master
            node is enqueued to. If None, the nodes will be enqueued to the same queue as the master node
        :param wait: If True, the master node will wait for the other nodes to start
        :param addr: The address of the master node's worker. If the environment variable
            ``CLEARML_MULTI_NODE_MASTER_DEF_ADDR`` is set, the value of this parameter will be set to
            the one defined in ``CLEARML_MULTI_NODE_MASTER_DEF_ADDR``.
            If ``CLEARML_MULTI_NODE_MASTER_DEF_ADDR`` doesn't exist, but ``MASTER_ADDR`` does, then the value of this
            parameter will be set to the one defined in ``MASTER_ADDR``. If neither environment variables exist,
            the value passed to the parameter will be used. If this value is None (default), the private IP of
            the machine the master node is running on will be used.
        :param devices: The devices to use. This can be a positive number indicating the number of devices to use,
            a sequence of indices or the value ``-1`` to indicate all available devices should be used.
        :param hide_children: If True, the children tasks will be hidden. Otherwise, they will be visible in the UI

        :return: A dictionary containing relevant information regarding the multi node run. This dictionary has the following entries:

          - `master_addr` - the address of the machine that the master node is running on
          - `master_port` - the open port of the machine that the master node is running on
          - `total_num_nodes` - the total number of nodes, including the master
          - `queue` - the queue the nodes are enqueued to, excluding the master
          - `node_rank` - the rank of the current node (master has rank 0)
          - `wait` - if True, the master node will wait for the other nodes to start
        """

        def set_launch_multi_node_runtime_props(task: Task, conf: Dict[str, Any]) -> None:
            # noinspection PyProtectedMember
            task._set_runtime_properties(
                {"{}/{}".format(self._launch_multi_node_section, k): v for k, v in conf.items()}
            )

        if total_num_nodes < 1:
            raise UsageError("total_num_nodes needs to be at least 1")
        if running_remotely() and not (self.data.execution and self.data.execution.queue) and not queue:
            raise UsageError("Master task is not enqueued to any queue and the queue parameter is None")

        master_conf = {
            "master_addr": os.environ.get(
                "CLEARML_MULTI_NODE_MASTER_DEF_ADDR",
                os.environ.get("MASTER_ADDR", addr or get_private_ip()),
            ),
            "master_port": int(
                os.environ.get(
                    "CLEARML_MULTI_NODE_MASTER_DEF_PORT",
                    os.environ.get("MASTER_PORT", port),
                )
            ),
            "node_rank": 0,
            "wait": wait,
            "devices": devices,
        }
        editable_conf = {"total_num_nodes": total_num_nodes, "queue": queue}
        editable_conf = self.connect(editable_conf, name=self._launch_multi_node_section)
        if not running_remotely():
            return master_conf
        master_conf.update(editable_conf)
        runtime_properties = self._get_runtime_properties()
        remote_node_rank = runtime_properties.get("{}/node_rank".format(self._launch_multi_node_section))

        current_conf = master_conf
        if remote_node_rank:
            # self is a child node, build the conf from the runtime proprerties
            current_conf = {
                entry: runtime_properties.get("{}/{}".format(self._launch_multi_node_section, entry))
                for entry in master_conf.keys()
            }
        elif os.environ.get("CLEARML_MULTI_NODE_MASTER") is None:
            nodes_to_wait = []
            # self is the master node, enqueue the other nodes
            set_launch_multi_node_runtime_props(self, master_conf)
            for node_rank in range(1, master_conf.get("total_num_nodes", total_num_nodes)):
                node = self.clone(source_task=self, parent=self.id)
                node_conf = copy.deepcopy(master_conf)
                node_conf["node_rank"] = node_rank
                set_launch_multi_node_runtime_props(node, node_conf)
                node.set_system_tags(
                    node.get_system_tags()
                    + [self._launch_multi_node_instance_tag]
                    + ([self.__hidden_tag] if hide_children else [])
                )
                if master_conf.get("queue"):
                    Task.enqueue(node, queue_name=master_conf["queue"])
                else:
                    Task.enqueue(node, queue_id=self.data.execution.queue)
                if master_conf.get("wait"):
                    nodes_to_wait.append(node)
            for node_to_wait, rank in zip(
                nodes_to_wait,
                range(1, master_conf.get("total_num_nodes", total_num_nodes)),
            ):
                self.log.info("Waiting for node with task ID {} and rank {}".format(node_to_wait.id, rank))
                node_to_wait.wait_for_status(
                    status=(
                        Task.TaskStatusEnum.completed,
                        Task.TaskStatusEnum.stopped,
                        Task.TaskStatusEnum.closed,
                        Task.TaskStatusEnum.failed,
                        Task.TaskStatusEnum.in_progress,
                    ),
                    check_interval_sec=10,
                )
                self.log.info("Node with task ID {} and rank {} detected".format(node_to_wait.id, rank))
            os.environ["CLEARML_MULTI_NODE_MASTER"] = "1"

        num_devices = 1
        if devices is not None:
            try:
                num_devices = int(devices)
            except TypeError:
                try:
                    num_devices = len(devices)
                except Exception as ex:
                    raise ValueError("Failed parsing number of devices: {}".format(ex))
            except ValueError as ex:
                raise ValueError("Failed parsing number of devices: {}".format(ex))
            if num_devices < 0:
                try:
                    import torch

                    num_devices = torch.cuda.device_count()
                except ImportError:
                    raise ImportError(
                        "Could not import `torch` while finding the number of devices. "
                        "Please install it or set `devices` to a value different than -1"
                    )

        os.environ["MASTER_ADDR"] = current_conf.get("master_addr", "")
        os.environ["MASTER_PORT"] = str(current_conf.get("master_port", ""))
        os.environ["RANK"] = str(
            current_conf.get("node_rank", 0) * num_devices + int(os.environ.get("LOCAL_RANK", "0"))
        )
        os.environ["NODE_RANK"] = str(current_conf.get("node_rank", ""))
        os.environ["WORLD_SIZE"] = str(current_conf.get("total_num_nodes", total_num_nodes) * num_devices)

        return current_conf

    def mark_started(self, force: bool = False) -> ():
        """
        Manually mark a Task as started (happens automatically)

        :param bool force: If True, the task status will be changed to `started` regardless of the current Task state.
        """
        # UI won't let us see metrics if we're not started
        self.started(force=force)
        self.reload()

    def mark_stopped(self, force: bool = False, status_message: Optional[str] = None) -> ():
        """
        Manually mark a Task as stopped (also used in :meth:`_at_exit`)

        :param bool force: If True, the task status will be changed to `stopped` regardless of the current Task state.
        :param str status_message: Optional, add status change message to the stop request.
            This message will be stored as status_message on the Task's info panel
        """
        # flush any outstanding logs
        self.flush(wait_for_uploads=True)
        # mark task as stopped
        self.stopped(force=force, status_message=str(status_message) if status_message else None)

    def mark_stop_request(self, force: bool = False, status_message: Optional[str] = None) -> ():
        """
        Request a task to stop. this will not change the task status
        but mark a request for an agent or SDK to actually stop the Task.
        This will trigger the Task's abort callback, and at the end will
        change the task status to stopped and kill the Task's processes

        Notice: calling this on your own Task, will cause
        the watchdog to call the on_abort callback and kill the process

        :param bool force: If not True, call fails if the task status is not 'in_progress'
        :param str status_message: Optional, add status change message to the stop request.
            This message will be stored as status_message on the Task's info panel
        """
        # flush any outstanding logs
        self.flush(wait_for_uploads=True)
        # request task stop
        return self.stop_request(self, force=force, status_message=status_message)

    def flush(self, wait_for_uploads: bool = False) -> bool:
        """
        Flush any outstanding reports or console logs.

        :param bool wait_for_uploads: Wait for all outstanding uploads to complete

            - ``True`` - Wait
            - ``False`` - Do not wait (default)
        """

        # make sure model upload is done
        if BackendModel.get_num_results() > 0 and wait_for_uploads:
            BackendModel.wait_for_results()

        # flush any outstanding logs
        if self._logger:
            # noinspection PyProtectedMember
            self._logger._flush_stdout_handler()
        if self.__reporter:
            self.__reporter.flush()
            if wait_for_uploads:
                self.__reporter.wait_for_events()

        LoggerRoot.flush()

        return True

    def reset(self, set_started_on_success: bool = False, force: bool = False) -> None:
        """
        Reset a Task. ClearML reloads a Task after a successful reset.
        When a worker executes a Task remotely, the Task does not reset unless
        the ``force`` parameter is set to ``True`` (this avoids accidentally clearing logs and metrics).

        :param bool set_started_on_success: If successful, automatically set the Task to `started`

            - ``True`` - If successful, set to started.
            - ``False`` - If successful, do not set to started. (default)

        :param bool force: Force a Task reset, even when executing the Task (experiment) remotely in a worker

            - ``True`` - Force
            - ``False`` - Do not force (default)
        """
        if not running_remotely() or not self.is_main_task() or force:
            super(Task, self).reset(set_started_on_success=set_started_on_success, force=force)

    def close(self) -> None:
        """
        Closes the current Task and changes its status to "Completed".
        Enables you to manually shut down the task from the process which opened the task.

        This method does not terminate the (current) Python process, in contrast to :meth:`Task.mark_completed`.

        After having :meth:`Task.close` -d a task, the respective object cannot be used anymore and
        methods like :meth:`Task.connect` or :meth:`Task.connect_configuration` will throw a `ValueError`.
        In order to obtain an object representing the task again, use methods like :meth:`Task.get_task`.

        .. warning::
           Only call :meth:`Task.close` if you are certain the Task is not needed.
        """
        if self._at_exit_called:
            return

        # store is main before we call at_exit, because will will Null it
        is_main = self.is_main_task()
        is_sub_process = self.__is_subprocess()

        # wait for repository detection (5 minutes should be reasonable time to detect all packages)
        if self._logger and not self.__is_subprocess():
            self._wait_for_repo_detection(timeout=300.0)

        self.__shutdown()
        # unregister atexit callbacks and signal hooks, if we are the main task
        if is_main:
            self.__register_at_exit(None)
            self._remove_signal_hooks()
            self._remove_exception_hooks()
            if not is_sub_process:
                # make sure we enable multiple Task.init callas with reporting sub-processes
                BackgroundMonitor.clear_main_process(self)
                # noinspection PyProtectedMember
                Logger._remove_std_logger()

                # unbind everything
                PatchHydra.update_current_task(None)
                PatchedJoblib.update_current_task(None)
                PatchedMatplotlib.update_current_task(None)
                PatchAbsl.update_current_task(None)
                TensorflowBinding.update_current_task(None)
                PatchPyTorchModelIO.update_current_task(None)
                PatchMegEngineModelIO.update_current_task(None)
                PatchXGBoostModelIO.update_current_task(None)
                PatchCatBoostModelIO.update_current_task(None)
                PatchFastai.update_current_task(None)
                PatchLIGHTgbmModelIO.update_current_task(None)
                EnvironmentBind.update_current_task(None)
                PatchJsonArgParse.update_current_task(None)
                PatchOsFork.patch_fork(None)

    def delete(
        self,
        delete_artifacts_and_models: bool = True,
        skip_models_used_by_other_tasks: bool = True,
        raise_on_error: bool = False,
        callback: Callable[[str, str], bool] = None,
    ) -> bool:
        """
        Delete the task as well as its output models and artifacts.
        Models and artifacts are deleted from their storage locations, each using its URI.

        Note: in order to delete models and artifacts using their URI, make sure the proper storage credentials are
        configured in your configuration file (e.g. if an artifact is stored in S3, make sure sdk.aws.s3.credentials
        are properly configured and that you have delete permission in the related buckets).

        :param delete_artifacts_and_models: If True, artifacts and models would also be deleted (default True).
                                            If callback is provided, this argument is ignored.
        :param skip_models_used_by_other_tasks: If True, models used by other tasks would not be deleted (default True)
        :param raise_on_error: If True, an exception will be raised when encountering an error.
                               If False an error would be printed and no exception will be raised.
        :param callback: An optional callback accepting a uri type (string) and a uri (string) that will be called
                         for each artifact and model. If provided, the delete_artifacts_and_models is ignored.
                         Return True to indicate the artifact/model should be deleted or False otherwise.
        :return: True if the task was deleted successfully.
        """
        if not running_remotely() or not self.is_main_task():
            return super(Task, self)._delete(
                delete_artifacts_and_models=delete_artifacts_and_models,
                skip_models_used_by_other_tasks=skip_models_used_by_other_tasks,
                raise_on_error=raise_on_error,
                callback=callback,
            )
        return False

    def register_artifact(
        self,
        name: str,
        artifact: "pandas.DataFrame",
        metadata: Dict = None,
        uniqueness_columns: Union[bool, Sequence[str]] = True,
    ) -> None:
        """
        Register (add) an artifact for the current Task. Registered artifacts are dynamically synchronized with the
        **ClearML Server** (backend). If a registered artifact is updated, the update is stored in the
        **ClearML Server** (backend). Registered artifacts are primarily used for Data Auditing.

        The currently supported registered artifact object type is a pandas.DataFrame.

        See also :meth:`Task.unregister_artifact` and :meth:`Task.get_registered_artifacts`.

        .. note::
           ClearML also supports uploaded artifacts which are one-time uploads of static artifacts that are not
           dynamically synchronized with the **ClearML Server** (backend). These static artifacts include
           additional object types. For more information, see :meth:`Task.upload_artifact`.

        :param str name: The name of the artifact.

         .. warning::
            If an artifact with the same name was previously registered, it is overwritten.
        :param object artifact: The artifact object.
        :param dict metadata: A dictionary of key-value pairs for any metadata. This dictionary appears with the
            experiment in the **ClearML Web-App (UI)**, **ARTIFACTS** tab.
        :param uniqueness_columns: A Sequence of columns for artifact uniqueness comparison criteria, or the default
            value of ``True``. If ``True``, the artifact uniqueness comparison criteria is all the columns,
            which is the same as ``artifact.columns``.
        """
        if not isinstance(uniqueness_columns, CollectionsSequence) and uniqueness_columns is not True:
            raise ValueError("uniqueness_columns should be a List (sequence) or True")
        if isinstance(uniqueness_columns, str):
            uniqueness_columns = [uniqueness_columns]
        self._artifacts_manager.register_artifact(
            name=name,
            artifact=artifact,
            metadata=metadata,
            uniqueness_columns=uniqueness_columns,
        )

    def unregister_artifact(self, name: str) -> None:
        """
        Unregister (remove) a registered artifact. This removes the artifact from the watch list that ClearML uses
        to synchronize artifacts with the **ClearML Server** (backend).

        .. important::
           - Calling this method does not remove the artifact from a Task. It only stops ClearML from
             monitoring the artifact.
           - When this method is called, ClearML immediately takes the last snapshot of the artifact.
        """
        self._artifacts_manager.unregister_artifact(name=name)

    def get_registered_artifacts(self) -> Dict[str, Artifact]:
        """
        Get a dictionary containing the Task's registered (dynamically synchronized) artifacts (name, artifact object).

        .. note::
           After calling ``get_registered_artifacts``, you can still modify the registered artifacts.

        :return: The registered (dynamically synchronized) artifacts.
        """
        return self._artifacts_manager.registered_artifacts

    def upload_artifact(
        self,
        name: str,
        artifact_object: Union[str, Mapping, "pandas.DataFrame", numpy.ndarray, "Image.Image", Any],
        metadata: Optional[Mapping] = None,
        delete_after_upload: bool = False,
        auto_pickle: Optional[bool] = None,
        preview: Any = None,
        wait_on_upload: bool = False,
        extension_name: Optional[str] = None,
        serialization_function: Optional[Callable[[Any], Union[bytes, bytearray]]] = None,
        retries: int = 0,
        sort_keys: bool = True,
    ) -> bool:
        """
        Upload (add) a static artifact to a Task object. The artifact is uploaded in the background.

        The currently supported upload (static) artifact types include:

        - string / pathlib2.Path - A path to artifact file. If a wildcard or a folder is specified, then ClearML
          creates and uploads a ZIP file.
        - dict - ClearML stores a dictionary as ``.json`` (or see ``extension_name``) file and uploads it.
        - pandas.DataFrame - ClearML stores a pandas.DataFrame as ``.csv.gz`` (compressed CSV)
          (or see ``extension_name``) file and uploads it.
        - numpy.ndarray - ClearML stores a numpy.ndarray as ``.npz`` (or see ``extension_name``) file and uploads it.
        - PIL.Image - ClearML stores a PIL.Image as ``.png`` (or see ``extension_name``) file and uploads it.
        - Any - If called with auto_pickle=True, the object will be pickled and uploaded.

        :param name: The artifact name.

            .. warning::
               If an artifact with the same name was previously uploaded, then it is overwritten.

        :param artifact_object:  The artifact object.
        :param metadata: A dictionary of key-value pairs for any metadata. This dictionary appears with the
            experiment in the **ClearML Web-App (UI)**, **ARTIFACTS** tab.
        :param bool delete_after_upload: After the upload, delete the local copy of the artifact

            - ``True`` - Delete the local copy of the artifact.
            - ``False`` - Do not delete. (default)

        :param auto_pickle: If True and the artifact_object is not one of the following types:
            pathlib2.Path, dict, pandas.DataFrame, numpy.ndarray, PIL.Image, url (string), local_file (string),
            the artifact_object will be pickled and uploaded as pickle file artifact (with file extension .pkl).
            If set to None (default) the sdk.development.artifacts.auto_pickle configuration value will be used.

        :param preview: The artifact preview

        :param wait_on_upload: Whether the upload should be synchronous, forcing the upload to complete
            before continuing.

        :param extension_name: File extension which indicates the format the artifact should be stored as.
            The following are supported, depending on the artifact type (default value applies when extension_name is None):

          - Any - ``.pkl`` if passed supersedes any other serialization type, and always pickles the object
          - dict - ``.json``, ``.yaml`` (default ``.json``)
          - pandas.DataFrame - ``.csv.gz``, ``.parquet``, ``.feather``, ``.pickle`` (default ``.csv.gz``)
          - numpy.ndarray - ``.npz``, ``.csv.gz`` (default ``.npz``)
          - PIL.Image - whatever extensions PIL supports (default ``.png``)
          - In case the ``serialization_function`` argument is set - any extension is supported

        :param serialization_function: A serialization function that takes one
            parameter of any type which is the object to be serialized. The function should return
            a `bytes` or `bytearray` object, which represents the serialized object. Note that the object will be
            immediately serialized using this function, thus other serialization methods will not be used
            (e.g. `pandas.DataFrame.to_csv`), even if possible. To deserialize this artifact when getting
            it using the `Artifact.get` method, use its `deserialization_function` argument.

        :param retries: Number of retries before failing to upload artifact. If 0, the upload is not retried

        :param sort_keys: If True (default), sort the keys of the artifact if it is yaml/json serializable.
            Otherwise, don't sort the keys. Ignored if the artifact is not yaml/json serializable.

        :return: The status of the upload.

            - ``True`` - Upload succeeded.
            - ``False`` - Upload failed.

        :raise: If the artifact object type is not supported, raise a ``ValueError``.
        """
        exception_to_raise = None
        for retry in range(retries + 1):
            # noinspection PyBroadException
            try:
                if self._artifacts_manager.upload_artifact(
                    name=name,
                    artifact_object=artifact_object,
                    metadata=metadata,
                    delete_after_upload=delete_after_upload,
                    auto_pickle=auto_pickle,
                    preview=preview,
                    wait_on_upload=wait_on_upload,
                    extension_name=extension_name,
                    serialization_function=serialization_function,
                    sort_keys=sort_keys,
                ):
                    return True
            except Exception as e:
                exception_to_raise = e
            if retry < retries:
                getLogger().warning(
                    "Failed uploading artifact '{}'. Retrying... ({}/{})".format(name, retry + 1, retries)
                )
        if exception_to_raise:
            raise exception_to_raise
        return False

    def get_debug_samples(self, title: str, series: str, n_last_iterations: Optional[int] = None) -> List[dict]:
        """
        :param str title: Debug sample's title, also called metric in the UI
        :param str series: Debug sample's series,
            corresponding to debug sample's file name in the UI, also known as variant
        :param int n_last_iterations: How many debug sample iterations to fetch in reverse chronological order.
            Leave empty to get all debug samples.

        :raise: TypeError if `n_last_iterations` is explicitly set to anything other than a positive integer value

        :return: A list of `dict`s, each dictionary containing the debug sample's URL and other metadata.
            The URLs can be passed to StorageManager.get_local_copy to fetch local copies of debug samples.
        """
        from .config.defs import MAX_SERIES_PER_METRIC

        if not n_last_iterations:
            n_last_iterations = MAX_SERIES_PER_METRIC.get()

        if isinstance(n_last_iterations, int) and n_last_iterations >= 0:
            samples = self._get_debug_samples(title=title, series=series, n_last_iterations=n_last_iterations)
        else:
            raise TypeError(
                "Parameter n_last_iterations is expected to be a positive integer value,"
                " but instead got n_last_iterations={}".format(n_last_iterations)
            )

        return samples

    def _send_debug_image_request(self, title, series, n_last_iterations, scroll_id=None):
        return Task._send(
            Task._get_default_session(),
            events.DebugImagesRequest(
                [{"task": self.id, "metric": title, "variants": [series]}],
                iters=n_last_iterations,
                scroll_id=scroll_id,
            ),
        )

    def _get_debug_samples(self, title: str, series: str, n_last_iterations: Optional[int] = None) -> List[dict]:
        response = self._send_debug_image_request(title, series, n_last_iterations)

        debug_samples = []

        while True:
            scroll_id = response.response_data.get("scroll_id", None)

            for metric_resp in response.response_data.get("metrics", []):
                iterations_events: List[List[dict]] = [
                    iteration["events"] for iteration in metric_resp.get("iterations", [])
                ]
                flattened_events = (event for single_iter_events in iterations_events for event in single_iter_events)
                debug_samples.extend(flattened_events)

            response = self._send_debug_image_request(title, series, n_last_iterations, scroll_id=scroll_id)

            if len(debug_samples) == n_last_iterations or all(
                len(metric_resp.get("iterations", [])) == 0 for metric_resp in response.response_data.get("metrics", [])
            ):
                break

        return debug_samples

    def get_models(self) -> Mapping[str, Sequence[Model]]:
        """
        Return a dictionary with ``{'input': [], 'output': []}`` loaded/stored models of the current Task
        Input models are files loaded in the task, either manually or automatically logged
        Output models are files stored in the task, either manually or automatically logged.
        Automatically logged frameworks are for example: TensorFlow, Keras, PyTorch, ScikitLearn(joblib) etc.

        :return: A dictionary-like object with "input"/"output" keys and input/output properties, pointing to a
            list-like object containing Model objects. Each list-like object also acts as a dictionary, mapping
            model name to an appropriate model instance.

            Example:

            .. code-block:: py

                {'input': [clearml.Model()], 'output': [clearml.Model()]}

        """
        return TaskModels(self)

    def is_current_task(self) -> bool:
        """
        .. deprecated:: 0.13.0
           This method is deprecated. Use :meth:`Task.is_main_task` instead.

        Is this Task object the main execution Task (initially returned by :meth:`Task.init`)

        :return: Is this Task object the main execution Task

            - ``True`` - Is the main execution Task.
            - ``False`` - Is not the main execution Task.

        """
        return self.is_main_task()

    def is_main_task(self) -> bool:
        """
        Is this Task object the main execution Task (initially returned by :meth:`Task.init`)

        .. note::
           If :meth:`Task.init` was never called, this method will *not* create
           it, making this test more efficient than:

           .. code-block:: py

              Task.init() == task

        :return: Is this Task object the main execution Task

            - ``True`` - Is the main execution Task.
            - ``False`` - Is not the main execution Task.

        """
        return self is self.__main_task

    def set_model_config(
        self,
        config_text: Optional[str] = None,
        config_dict: Optional[Mapping] = None,
    ) -> None:
        """
        .. deprecated:: 0.14.1
            Use :meth:`Task.connect_configuration` instead.
        """
        self._set_model_config(config_text=config_text, config_dict=config_dict)

    def get_model_config_text(self) -> str:
        """
        .. deprecated:: 0.14.1
            Use :meth:`Task.connect_configuration` instead.
        """
        return self._get_model_config_text()

    def get_model_config_dict(self) -> Dict:
        """
        .. deprecated:: 0.14.1
            Use :meth:`Task.connect_configuration` instead.
        """
        return self._get_model_config_dict()

    def set_model_label_enumeration(self, enumeration: Optional[Mapping[str, int]] = None) -> ():
        """
        Set the label enumeration for the Task object before creating an output model.
        Later, when creating an output model, the model will inherit these properties.

        :param dict enumeration: A label enumeration dictionary of string (label) to integer (value) pairs.

            For example:

            .. code-block:: javascript

               {
                    "background": 0,
                    "person": 1
               }
        """
        super(Task, self).set_model_label_enumeration(enumeration=enumeration)

    def get_last_iteration(self) -> int:
        """
        Get the last reported iteration, which is the last iteration for which the Task reported a metric.

        .. note::
           The maximum reported iteration is not in the local cache. This method
           sends a request to the **ClearML Server** (backend).

        :return: The last reported iteration number.
        """
        self._reload_last_iteration()
        return max(
            self.data.last_iteration or 0,
            self.__reporter.max_iteration if self.__reporter else 0,
        )

    def set_initial_iteration(self, offset: int = 0) -> int:
        """
        Set initial iteration, instead of zero. Useful when continuing training from previous checkpoints

        :param int offset: Initial iteration (at starting point)
        :return: Newly set initial offset.
        """
        return super(Task, self).set_initial_iteration(offset=offset)

    def get_initial_iteration(self) -> int:
        """
        Return the initial iteration offset, default is 0
        Useful when continuing training from previous checkpoints

        :return: Initial iteration offset.
        """
        return super(Task, self).get_initial_iteration()

    def get_last_scalar_metrics(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Get the last scalar metrics which the Task reported. This is a nested dictionary, ordered by title and series.

        For example:

        .. code-block:: javascript

           {
            "title": {
                "series": {
                    "last": 0.5,
                    "min": 0.1,
                    "max": 0.9
                    }
                }
            }

        :return: The last scalar metrics.
        """
        self.reload()
        metrics = self.data.last_metrics
        scalar_metrics = dict()
        for i in metrics.values():
            for j in i.values():
                scalar_metrics.setdefault(j["metric"], {}).setdefault(
                    j["variant"],
                    {"last": j["value"], "min": j["min_value"], "max": j["max_value"]},
                )
        return scalar_metrics

    def get_parameters_as_dict(self, cast: bool = False) -> Dict:
        """
        Get the Task parameters as a raw nested dictionary.

        .. note::
           If `cast` is False (default) The values are not parsed. They are returned as is.

        :param cast: If True, cast the parameter to the original type. Default False,
            values are returned in their string representation

        """
        return naive_nested_from_flat_dictionary(self.get_parameters(cast=cast))

    def set_parameters_as_dict(self, dictionary: Dict) -> None:
        """
        Set the parameters for the Task object from a dictionary. The dictionary can be nested.
        This does not link the dictionary to the Task object. It does a one-time update. This
        is the same behavior as the :meth:`Task.connect` method.
        """
        self._arguments.copy_from_dict(flatten_dictionary(dictionary))

    def get_user_properties(self, value_only: bool = False) -> Dict[str, Union[str, dict]]:
        """
        Get user properties for this task.
        Returns a dictionary mapping user property name to user property details dict.

        :param value_only: If True, returned user property details will be a string representing the property value.
        """
        if not Session.check_min_api_version("2.9"):
            self.log.info("User properties are not supported by the server")
            return {}

        section = "properties"

        params = self._hyper_params_manager.get_hyper_params(
            sections=[section],
            projector=(lambda x: x.get("value")) if value_only else None,
        )

        return dict(params.get(section, {}))

    def set_user_properties(
        self,
        *iterables: Union[Mapping[str, Union[str, dict, None]], Iterable[dict]],
        **properties: Union[str, dict, int, float, None],
    ) -> bool:
        """
        Set user properties for this task.
        A user property can contain the following fields (all of type string):
        name / value / description / type

        Examples:

        .. code-block:: py

            task.set_user_properties(backbone='great', stable=True)
            task.set_user_properties(backbone={"type": int, "description": "network type", "value": "great"}, )
            task.set_user_properties(
                {"name": "backbone", "description": "network type", "value": "great"},
                {"name": "stable", "description": "is stable", "value": True},
            )

        :param iterables: Properties iterables, each can be:

            * A dictionary of string key (name) to either a string value (value) a dict (property details). If the value
                is a dict, it must contain a "value" field. For example:

                .. code-block:: javascript

                    {
                        "property_name": {"description": "This is a user property", "value": "property value"},
                        "another_property_name": {"description": "This is user property", "value": "another value"},
                        "yet_another_property_name": "some value"
                    }


            * An iterable of dicts (each representing property details). Each dict must contain a "name" field and a
                "value" field. For example:

                .. code-block:: javascript

                    [
                        {
                            "name": "property_name",
                            "description": "This is a user property",
                            "value": "property value"
                        },
                        {
                            "name": "another_property_name",
                            "description": "This is another user property",
                            "value": "another value"
                        }
                    ]

        :param properties: Additional properties keyword arguments. Key is the property name, and value can be
            a string (property value) or a dict (property details). If the value is a dict, it must contain a "value"
            field. For example:

            .. code-block:: javascript

                {
                    "property_name": "string as property value",
                    "another_property_name": {
                        "type": "string",
                        "description": "This is user property",
                        "value": "another value"
                    }
                }

        """
        if not Session.check_min_api_version("2.9"):
            self.log.info("User properties are not supported by the server")
            return False

        return self._hyper_params_manager.edit_hyper_params(
            iterables=list(properties.items())
            + (list(iterables.items()) if isinstance(iterables, dict) else list(iterables)),
            replace="none",
            force_section="properties",
        )

    def get_script(self) -> Mapping[str, Optional[str]]:
        """
        Get task's script details.

        Returns a dictionary containing the script details.

        :return: Dictionary with script properties e.g.

        .. code-block:: javascript

           {
                'working_dir': 'examples/reporting',
                'entry_point': 'artifacts.py',
                'branch': 'master',
                'repository': 'https://github.com/allegroai/clearml.git'
           }

        """
        script = self.data.script
        return {
            "working_dir": script.working_dir,
            "entry_point": script.entry_point,
            "branch": script.branch,
            "repository": script.repository,
        }

    def set_script(
        self,
        repository: Optional[str] = None,
        branch: Optional[str] = None,
        commit: Optional[str] = None,
        diff: Optional[str] = None,
        working_dir: Optional[str] = None,
        entry_point: Optional[str] = None,
    ) -> None:
        """
        Set task's script.

        Examples:

        .. code-block:: py

            task.set_script(
                repository='https://github.com/allegroai/clearml.git,
                branch='main',
                working_dir='examples/reporting',
                entry_point='artifacts.py'
            )

        :param repository: Optional, URL of remote repository. use empty string ("") to clear repository entry.
        :param branch: Optional, Select specific repository branch / tag. use empty string ("") to clear branch entry.
        :param commit: Optional, set specific git commit id. use empty string ("") to clear commit ID entry.
        :param diff: Optional, set "git diff" section. use empty string ("") to clear git-diff entry.
        :param working_dir: Optional, Working directory to launch the script from.
        :param entry_point: Optional, Path to execute within the repository.

        """
        self.reload()
        script = self.data.script
        if repository is not None:
            script.repository = str(repository)
        if branch is not None:
            script.branch = str(branch)
            if script.tag:
                script.tag = None
        if commit is not None:
            script.version_num = str(commit)
        if diff is not None:
            script.diff = str(diff)
        if working_dir is not None:
            script.working_dir = str(working_dir)
        if entry_point is not None:
            script.entry_point = str(entry_point)
        # noinspection PyProtectedMember
        self._update_script(script=script)

    def delete_user_properties(self, *iterables: Iterable[Union[dict, Iterable[str]]]) -> bool:
        """
        Delete hyperparameters for this task.

        :param iterables: Hyperparameter key iterables. Each an iterable whose possible values each represent
            a hyperparameter entry to delete, value formats are:

            * A dictionary containing a 'section' and 'name' fields
            * An iterable (e.g. tuple, list etc.) whose first two items denote 'section' and 'name'
        """
        if not Session.check_min_api_version("2.9"):
            self.log.info("User properties are not supported by the server")
            return False

        return self._hyper_params_manager.delete_hyper_params(*iterables)

    def set_base_docker(
        self,
        docker_cmd: Optional[str] = None,
        docker_image: Optional[str] = None,
        docker_arguments: Optional[Union[str, Sequence[str]]] = None,
        docker_setup_bash_script: Optional[Union[str, Sequence[str]]] = None,
    ) -> ():
        """
        Set the base docker image for this experiment
        If provided, this value will be used by clearml-agent to execute this experiment
        inside the provided docker image.
        When running remotely the call is ignored

        :param docker_cmd: Deprecated! compound docker container image + arguments
            (example: 'nvidia/cuda:11.1 -e test=1') Deprecated, use specific arguments.
        :param docker_image: docker container image (example: 'nvidia/cuda:11.1')
        :param docker_arguments: docker execution parameters (example: '-e ENV=1')
        :param docker_setup_bash_script: bash script to run at the
            beginning of the docker before launching the Task itself. example: ['apt update', 'apt-get install -y gcc']
        """
        if not self.running_locally() and self.is_main_task():
            return

        super(Task, self).set_base_docker(
            docker_cmd=docker_cmd or docker_image,
            docker_arguments=docker_arguments,
            docker_setup_bash_script=docker_setup_bash_script,
        )

    @classmethod
    def set_resource_monitor_iteration_timeout(
        cls,
        seconds_from_start: float = 30.0,
        wait_for_first_iteration_to_start_sec: float = 180.0,
        max_wait_for_first_iteration_to_start_sec: float = 1800.0,
    ) -> bool:
        """
        Set the ResourceMonitor maximum duration (in seconds) to wait until first scalar/plot is reported.
        If timeout is reached without any reporting, the ResourceMonitor will start reporting machine statistics based
        on seconds from Task start time (instead of based on iteration).
        Notice! Should be called before `Task.init`.

        :param seconds_from_start: Maximum number of seconds to wait for scalar/plot reporting before defaulting
            to machine statistics reporting based on seconds from experiment start time
        :param wait_for_first_iteration_to_start_sec: Set the initial time (seconds) to wait for iteration reporting
            to be used as x-axis for the resource monitoring, if timeout exceeds then reverts to `seconds_from_start`
        :param max_wait_for_first_iteration_to_start_sec: Set the maximum time (seconds) to allow the resource
            monitoring to revert back to iteration reporting x-axis after starting to report `seconds_from_start`

        :return: True if success
        """
        if ResourceMonitor._resource_monitor_instances:
            getLogger().warning(
                "Task.set_resource_monitor_iteration_timeout called after Task.init."
                " This might not work since the values might not be used in forked processes"
            )
        # noinspection PyProtectedMember
        for instance in ResourceMonitor._resource_monitor_instances:
            # noinspection PyProtectedMember
            instance._first_report_sec = seconds_from_start
            instance.wait_for_first_iteration = wait_for_first_iteration_to_start_sec
            instance.max_check_first_iteration = max_wait_for_first_iteration_to_start_sec

        # noinspection PyProtectedMember
        ResourceMonitor._first_report_sec_default = seconds_from_start
        # noinspection PyProtectedMember
        ResourceMonitor._wait_for_first_iteration_to_start_sec_default = wait_for_first_iteration_to_start_sec
        # noinspection PyProtectedMember
        ResourceMonitor._max_wait_for_first_iteration_to_start_sec_default = max_wait_for_first_iteration_to_start_sec
        return True

    def execute_remotely(
        self,
        queue_name: Optional[str] = None,
        clone: bool = False,
        exit_process: bool = True,
    ) -> Optional["Task"]:
        """
        If task is running locally (i.e., not by ``clearml-agent``), then clone the Task and enqueue it for remote
        execution; or, stop the execution of the current Task, reset its state, and enqueue it. If ``exit==True``,
        *exit* this process.

        .. note::
            If the task is running remotely (i.e., ``clearml-agent`` is executing it), this call is a no-op
            (i.e., does nothing).

        :param queue_name: The queue name used for enqueueing the task. If ``None``, this call exits the process
            without enqueuing the task.
        :param clone: Clone the Task and execute the newly cloned Task
            The values are:

          - ``True`` - A cloned copy of the Task will be created, and enqueued, instead of this Task.
          - ``False`` - The Task will be enqueued.

        :param exit_process: The function call will leave the calling process at the end.

          - ``True`` - Exit the process (exit(0)). Note: if ``clone==False``, then ``exit_process`` must be ``True``.
          - ``False`` - Do not exit the process.

        :return Task: return the task object of the newly generated remotely executing task
        """
        # do nothing, we are running remotely
        if running_remotely() and self.is_main_task():
            return None

        if not self.is_main_task():
            LoggerRoot.get_base_logger().warning(
                "Calling task.execute_remotely is only supported on main Task (created with Task.init)\n"
                "Defaulting to self.enqueue(queue_name={})".format(queue_name)
            )
            if not queue_name:
                raise ValueError("queue_name must be provided")
            enqueue_task = Task.clone(source_task=self) if clone else self
            Task.enqueue(task=enqueue_task, queue_name=queue_name)
            return

        if not clone and not exit_process:
            raise ValueError(
                "clone==False and exit_process==False is not supported. "
                "Task enqueuing itself must exit the process afterwards."
            )

        # make sure we analyze the process
        if self.status in (Task.TaskStatusEnum.in_progress,):
            if clone:
                # wait for repository detection (5 minutes should be reasonable time to detect all packages)
                self.flush(wait_for_uploads=True)
                if self._logger and not self.__is_subprocess():
                    self._wait_for_repo_detection(timeout=300.0)
            else:
                # close ourselves (it will make sure the repo is updated)
                self.close()

        # clone / reset Task
        if clone:
            task = Task.clone(self)
        else:
            task = self
            # check if the server supports enqueueing aborted/stopped Tasks
            if Session.check_min_api_server_version("2.13"):
                self.mark_stopped(force=True)
            else:
                self.reset()

        # enqueue ourselves
        if queue_name:
            Task.enqueue(task, queue_name=queue_name)
            LoggerRoot.get_base_logger().warning(
                "Switching to remote execution, output log page {}".format(task.get_output_log_web_page())
            )
        else:
            # Remove the development system tag
            system_tags = [t for t in task.get_system_tags() if t != self._development_tag]
            self.set_system_tags(system_tags)
            # if we leave the Task out there, it makes sense to make it editable.
            self.reset(force=True)

        # leave this process.
        if exit_process:
            LoggerRoot.get_base_logger().warning(
                "ClearML Terminating local execution process - continuing execution remotely"
            )
            leave_process(0)

        return task

    def create_function_task(
        self,
        func: Callable,
        func_name: Optional[str] = None,
        task_name: Optional[str] = None,
        **kwargs: Optional[Any],
    ) -> Optional["Task"]:
        """
        Create a new task, and call ``func`` with the specified kwargs.
        One can think of this call as remote forking, where the newly created instance is the new Task
        calling the specified func with the appropriate kwargs and leaving once the func terminates.
        Notice that a remote executed function cannot create another child remote executed function.

        .. note::
            - Must be called from the main Task, i.e. the one created by Task.init(...)
            - The remote Tasks inherits the environment from the creating Task
            - In the remote Task, the entrypoint is the same as the creating Task
            - In the remote Task, the execution is the same until reaching this function call

        :param func: A function to execute remotely as a single Task.
            On the remote executed Task the entry-point and the environment are copied from this
            calling process, only this function call redirect the execution flow to the called func,
            alongside the passed arguments
        :param func_name: A unique identifier of the function. Default the function name without the namespace.
            For example Class.foo() becomes 'foo'
        :param task_name: The newly created Task name. Default: the calling Task name + function name
        :param kwargs: name specific arguments for the target function.
            These arguments will appear under the configuration, "Function" section

        :return Task: Return the newly created Task or None if running remotely and execution is skipped
        """
        if not self.is_main_task():
            raise ValueError("Only the main Task object can call create_function_task()")
        if not callable(func):
            raise ValueError("func must be callable")
        if not Session.check_min_api_version("2.9"):
            raise ValueError("Remote function execution is not supported, please upgrade to the latest server version")

        func_name = str(func_name or func.__name__).strip()
        if func_name in self._remote_functions_generated:
            raise ValueError(
                "Function name must be unique, a function by the name '{}' "
                "was already created by this Task.".format(func_name)
            )

        section_name = "Function"
        tag_name = "func"
        func_marker = "__func_readonly__"

        # sanitize the dict, leave only basic types that we might want to override later in the UI
        func_params = {k: v for k, v in kwargs.items() if verify_basic_value(v)}
        func_params[func_marker] = func_name

        # do not query if we are running locally, there is no need.
        task_func_marker = self.running_locally() or self.get_parameter("{}/{}".format(section_name, func_marker))

        # if we are running locally or if we are running remotely but we are not a forked tasks
        # condition explained:
        # (1) running in development mode creates all the forked tasks
        # (2) running remotely but this is not one of the forked tasks (i.e. it is missing the fork tag attribute)
        if self.running_locally() or not task_func_marker:
            self._wait_for_repo_detection(300)
            task = self.clone(
                self,
                name=task_name or "{} <{}>".format(self.name, func_name),
                parent=self.id,
            )
            task.set_system_tags((task.get_system_tags() or []) + [tag_name])
            task.connect(func_params, name=section_name)
            self._remote_functions_generated[func_name] = task.id
            return task

        # check if we are one of the generated functions and if this is us,
        # if we are not the correct function, not do nothing and leave
        if task_func_marker != func_name:
            self._remote_functions_generated[func_name] = len(self._remote_functions_generated) + 1
            return

        # mark this is us:
        self._remote_functions_generated[func_name] = self.id

        # this is us for sure, let's update the arguments and call the function
        self.connect(func_params, name=section_name)
        func_params.pop(func_marker, None)
        kwargs.update(func_params)
        func(**kwargs)
        # This is it, leave the process
        leave_process(0)

    def wait_for_status(
        self,
        status: Iterable["Task.TaskStatusEnum"] = (
            _Task.TaskStatusEnum.completed,
            _Task.TaskStatusEnum.stopped,
            _Task.TaskStatusEnum.closed,
        ),
        raise_on_status: Optional[Iterable["Task.TaskStatusEnum"]] = (_Task.TaskStatusEnum.failed,),
        check_interval_sec: float = 60.0,
    ) -> ():
        """
        Wait for a task to reach a defined status.

        :param status: Status to wait for. Defaults to ('completed', 'stopped', 'closed', )
        :param raise_on_status: Raise RuntimeError if the status of the tasks matches one of these values.
            Defaults to ('failed').
        :param check_interval_sec: Interval in seconds between two checks. Defaults to 60 seconds.

        :raise: RuntimeError if the status is one of ``{raise_on_status}``.
        """
        stopped_status = list(status) + (list(raise_on_status) if raise_on_status else [])
        while self.status not in stopped_status:
            time.sleep(check_interval_sec)

        if raise_on_status and self.status in raise_on_status:
            raise RuntimeError("Task {} has status: {}.".format(self.task_id, self.status))

        # make sure we have the Task object
        self.reload()

    def export_task(self) -> dict:
        """
        Export Task's configuration into a dictionary (for serialization purposes).
        A Task can be copied/modified by calling Task.import_task()
        Notice: Export task does not include the tasks outputs, such as results
        (scalar/plots etc.) or Task artifacts/models

        :return: dictionary of the Task's configuration.
        """
        self.reload()
        export_data = self.data.to_dict()
        export_data.pop("last_metrics", None)
        export_data.pop("last_iteration", None)
        export_data.pop("status_changed", None)
        export_data.pop("status_reason", None)
        export_data.pop("status_message", None)
        export_data.get("execution", {}).pop("artifacts", None)
        export_data.get("execution", {}).pop("model", None)
        export_data["project_name"] = self.get_project_name()
        export_data["session_api_version"] = self.session.api_version
        return export_data

    def update_task(self, task_data: dict) -> bool:
        """
        Update current task with configuration found on the task_data dictionary.
        See also export_task() for retrieving Task configuration.

        :param task_data: dictionary with full Task configuration
        :return: return True if Task update was successful
        """
        return bool(self.import_task(task_data=task_data, target_task=self, update=True))

    def rename(self, new_name: str) -> bool:
        """
        Rename this task

        :param new_name: The new name of this task

        :return: True if the rename was successful and False otherwise
        """
        result = bool(self._edit(name=new_name))
        self.reload()
        return result

    def move_to_project(
        self,
        new_project_id: Optional[str] = None,
        new_project_name: Optional[str] = None,
        system_tags: Optional[Sequence[str]] = None,
    ) -> bool:
        """
        Move this task to another project

        :param new_project_id: The ID of the project the task should be moved to.
            Not required if `new_project_name` is passed.
        :param new_project_name: Name of the new project the task should be moved to.
            Not required if `new_project_id` is passed.
        :param system_tags: System tags for the project the task should be moved to.

        :return: True if the move was successful and False otherwise
        """
        new_project_id = get_or_create_project(
            self.session,
            project_name=new_project_name,
            project_id=new_project_id,
            system_tags=system_tags,
        )
        result = bool(self._edit(project=new_project_id))
        self.reload()
        return result

    def register_abort_callback(
        self,
        callback_function: Optional[Callable],
        callback_execution_timeout: float = 30.0,
    ):  # type (...) -> None
        """
        Register a Task abort callback (single callback function support only).
        Pass a function to be called from a background thread when the Task is **externally** being aborted.
        Users must specify a timeout for the callback function execution (default 30 seconds)
        if the callback execution function exceeds the timeout, the Task's process will be terminated

        Call this register function from the main process only.

        Note: Ctrl-C is Not considered external, only backend induced abort is covered here

        :param callback_function: Callback function to be called via external thread (from the main process).
            pass None to remove existing callback
        :param callback_execution_timeout: Maximum callback execution time in seconds, after which the process
            will be terminated even if the callback did not return
        """
        if self.__is_subprocess():
            raise ValueError("Register abort callback must be called from the main process, this is a subprocess.")

        if callback_function is None:
            if self._dev_worker:
                self._dev_worker.register_abort_callback(callback_function=None, execution_timeout=0, poll_freq=0)
            return

        if float(callback_execution_timeout) <= 0:
            raise ValueError(
                "function_timeout_sec must be positive timeout in seconds, got {}".format(callback_execution_timeout)
            )

        # if we are running remotely we might not have a DevWorker monitoring us, so let's create one
        if not self._dev_worker:
            self._dev_worker = DevWorker()
            self._dev_worker.register(self, stop_signal_support=True)

        poll_freq = 15.0
        self._dev_worker.register_abort_callback(
            callback_function=callback_function,
            execution_timeout=callback_execution_timeout,
            poll_freq=poll_freq,
        )

    @classmethod
    def import_task(
        cls,
        task_data: dict,
        target_task: Optional[Union[str, "Task"]] = None,
        update: bool = False,
    ) -> Optional["Task"]:
        """
        Import (create) Task from previously exported Task configuration (see Task.export_task)
        Can also be used to edit/update an existing Task (by passing `target_task` and `update=True`).

        :param task_data: dictionary of a Task's configuration
        :param target_task: Import task_data into an existing Task. Can be either task_id (str) or Task object.
        :param update: If True, merge task_data with current Task configuration.
        :return: return True if Task was imported/updated
        """

        # restore original API version (otherwise, we might not be able to restore the data correctly)
        force_api_version = task_data.get("session_api_version") or None
        original_api_version = Session.api_version
        original_force_max_api_version = Session.force_max_api_version
        if force_api_version:
            Session.force_max_api_version = str(force_api_version)

        if not target_task:
            project_name = task_data.get("project_name") or Task._get_project_name(task_data.get("project", ""))
            target_task = Task.create(project_name=project_name, task_name=task_data.get("name", None))
        elif isinstance(target_task, six.string_types):
            target_task: Optional[Task] = Task.get_task(task_id=target_task)
        elif not isinstance(target_task, Task):
            raise ValueError(
                "`target_task` must be either Task id (str) or Task object, "
                "received `target_task` type {}".format(type(target_task))
            )
        target_task.reload()
        cur_data = target_task.data.to_dict()
        cur_data = merge_dicts(cur_data, task_data) if update else dict(**task_data)
        cur_data.pop("id", None)
        cur_data.pop("project", None)
        # noinspection PyProtectedMember
        valid_fields = list(tasks.EditRequest._get_data_props().keys())
        cur_data = dict((k, cur_data[k]) for k in valid_fields if k in cur_data)
        res = target_task._edit(**cur_data)
        if res and res.ok():
            target_task.reload()
        else:
            target_task = None

        # restore current api version, and return a new instance if Task with the current version
        if force_api_version:
            Session.force_max_api_version = original_force_max_api_version
            Session.api_version = original_api_version
            if target_task:
                target_task = Task.get_task(task_id=target_task.id)

        return target_task

    @classmethod
    def set_offline(cls, offline_mode: bool = False) -> None:
        """
        Set offline mode, where all data and logs are stored into local folder, for later transmission

        .. note::
            `Task.set_offline` can't move the same task from offline to online, nor can it be applied before `Task.create`.
            See below an example of **incorrect** usage of `Task.set_offline`:

            ```
            from clearml import Task

            Task.set_offline(True)
            task = Task.create(project_name='DEBUG', task_name="offline")
            # ^^^ an error or warning is raised, saying that Task.set_offline(True)
            #     is supported only for `Task.init`
            Task.set_offline(False)
            # ^^^ an error or warning is raised, saying that running Task.set_offline(False)
            #     while the current task is not closed is not supported

            data = task.export_task()

            imported_task = Task.import_task(task_data=data)
            ```

            The correct way to use `Task.set_offline` can be seen in the following example:

            ```
            from clearml import Task

            Task.set_offline(True)
            task = Task.init(project_name='DEBUG', task_name="offline")
            task.upload_artifact("large_artifact", "test_string")
            task.close()
            Task.set_offline(False)

            imported_task = Task.import_offline_session(task.get_offline_mode_folder())
            ```

        :param offline_mode: If True, offline-mode is turned on, and no communication to the backend is enabled.
        :return:
        """
        if running_remotely() or bool(offline_mode) == InterfaceBase._offline_mode:
            return
        if cls.current_task() and cls.current_task().status != cls.TaskStatusEnum.closed and not offline_mode:
            raise UsageError(
                "Switching from offline mode to online mode, but the current task has not been closed. Use `Task.close` to close it."
            )
        ENV_OFFLINE_MODE.set(offline_mode)
        InterfaceBase._offline_mode = bool(offline_mode)
        Session._offline_mode = bool(offline_mode)
        if not offline_mode:
            # noinspection PyProtectedMember
            Session._make_all_sessions_go_online()

    @classmethod
    def is_offline(cls) -> bool:
        """
        Return offline-mode state, If in offline-mode, no communication to the backend is enabled.

        :return: boolean offline-mode state
        """
        return cls._offline_mode

    @classmethod
    def import_offline_session(
        cls, session_folder_zip: str, previous_task_id: Optional[str] = None, iteration_offset: Optional[int] = 0
    ) -> Optional[str]:
        """
        Upload an offline session (execution) of a Task.
        Full Task execution includes repository details, installed packages, artifacts, logs, metric and debug samples.
        This function may also be used to continue a previously executed task with a task executed offline.

        :param session_folder_zip: Path to a folder containing the session, or zip-file of the session folder.
        :param previous_task_id: Task ID of the task you wish to continue with this offline session.
        :param iteration_offset: Reporting of the offline session will be offset with the
            number specified by this parameter. Useful for avoiding overwriting metrics.

        :return: Newly created task ID or the ID of the continued task (previous_task_id)
        """
        print("ClearML: Importing offline session from {}".format(session_folder_zip))

        temp_folder = None
        if Path(session_folder_zip).is_file():
            # unzip the file:
            temp_folder = mkdtemp(prefix="clearml-offline-")
            ZipFile(session_folder_zip).extractall(path=temp_folder)
            session_folder_zip = temp_folder

        session_folder = Path(session_folder_zip)
        if not session_folder.is_dir():
            raise ValueError("Could not find the session folder / zip-file {}".format(session_folder))

        try:
            with open((session_folder / cls._offline_filename).as_posix(), "rt") as f:
                export_data = json.load(f)
        except Exception as ex:
            raise ValueError(
                "Could not read Task object {}: Exception {}".format(session_folder / cls._offline_filename, ex)
            )
        current_task = cls.import_task(export_data)
        if previous_task_id:
            task_holding_reports = cls.get_task(task_id=previous_task_id)
            task_holding_reports.mark_started(force=True)
            task_holding_reports = cls.import_task(export_data, target_task=task_holding_reports, update=True)
        else:
            task_holding_reports = current_task
            task_holding_reports.mark_started(force=True)
        # fix artifacts
        if current_task.data.execution.artifacts:
            from . import StorageManager

            # noinspection PyProtectedMember
            offline_folder = os.path.join(export_data.get("offline_folder", ""), "data/")

            # noinspection PyProtectedMember
            remote_url = current_task._get_default_report_storage_uri()
            if remote_url and remote_url.endswith("/"):
                remote_url = remote_url[:-1]

            for artifact in current_task.data.execution.artifacts:
                local_path = artifact.uri.replace(offline_folder, "", 1)
                local_file = session_folder / "data" / local_path
                if local_file.is_file():
                    remote_path = local_path.replace(
                        ".{}{}".format(export_data["id"], os.sep),
                        ".{}{}".format(current_task.id, os.sep),
                        1,
                    )
                    artifact.uri = "{}/{}".format(remote_url, remote_path)
                    StorageManager.upload_file(local_file=local_file.as_posix(), remote_url=artifact.uri)
            # noinspection PyProtectedMember
            task_holding_reports._edit(execution=current_task.data.execution)
        for output_model in export_data.get("offline_output_models", []):
            model = OutputModel(task=current_task, **output_model["init"])
            if output_model.get("output_uri"):
                model.set_upload_destination(output_model.get("output_uri"))
            model.update_weights(auto_delete_file=False, **output_model["weights"])
            Metrics.report_offline_session(
                model,
                session_folder,
                iteration_offset=iteration_offset,
                remote_url=task_holding_reports._get_default_report_storage_uri(),
                only_with_id=output_model["id"],
                session=task_holding_reports.session,
            )
        # logs
        TaskHandler.report_offline_session(task_holding_reports, session_folder, iteration_offset=iteration_offset)
        # metrics
        Metrics.report_offline_session(
            task_holding_reports,
            session_folder,
            iteration_offset=iteration_offset,
            only_with_id=export_data["id"],
            session=task_holding_reports.session,
        )
        # print imported results page
        print("ClearML results page: {}".format(task_holding_reports.get_output_log_web_page()))
        task_holding_reports.mark_completed()
        # close task
        task_holding_reports.close()

        # cleanup
        if temp_folder:
            # noinspection PyBroadException
            try:
                shutil.rmtree(temp_folder)
            except Exception:
                pass

        return task_holding_reports.id

    @classmethod
    def set_credentials(
        cls,
        api_host: Optional[str] = None,
        web_host: Optional[str] = None,
        files_host: Optional[str] = None,
        key: Optional[str] = None,
        secret: Optional[str] = None,
        store_conf_file: bool = False,
    ) -> None:
        """
        Set new default **ClearML Server** (backend) host and credentials.

        These credentials will be overridden by either OS environment variables, or the ClearML configuration
        file, ``clearml.conf``.

        .. warning::
           Credentials must be set before initializing a Task object.

        For example, to set credentials for a remote computer:

        .. code-block:: py

            Task.set_credentials(
                api_host='http://localhost:8008', web_host='http://localhost:8080', files_host='http://localhost:8081',
                key='optional_credentials',  secret='optional_credentials'
            )
            task = Task.init('project name', 'experiment name')

        :param str api_host: The API server url. For example, ``host='http://localhost:8008'``
        :param str web_host: The Web server url. For example, ``host='http://localhost:8080'``
        :param str files_host: The file server url. For example, ``host='http://localhost:8081'``
        :param str key: The user key (in the key/secret pair). For example, ``key='thisisakey123'``
        :param str secret: The user secret (in the key/secret pair). For example, ``secret='thisisseceret123'``
        :param bool store_conf_file: If True, store the current configuration into the ~/clearml.conf file.
            If the configuration file exists, no change will be made (outputs a warning).
            Not applicable when running remotely (i.e. clearml-agent).
        """
        if api_host:
            Session.default_host = api_host
            if not running_remotely() and not ENV_HOST.get():
                ENV_HOST.set(api_host)
        if web_host:
            Session.default_web = web_host
            if not running_remotely() and not ENV_WEB_HOST.get():
                ENV_WEB_HOST.set(web_host)
        if files_host:
            Session.default_files = files_host
            if not running_remotely() and not ENV_FILES_HOST.get():
                ENV_FILES_HOST.set(files_host)
        if key:
            Session.default_key = key
            if not running_remotely():
                ENV_ACCESS_KEY.set(key)
        if secret:
            Session.default_secret = secret
            if not running_remotely():
                ENV_SECRET_KEY.set(secret)

        if store_conf_file and not running_remotely():
            active_conf_file = get_active_config_file()
            if active_conf_file:
                getLogger().warning(
                    "Could not store credentials in configuration file, '{}' already exists".format(active_conf_file)
                )
            else:
                conf = {
                    "api": dict(
                        api_server=Session.default_host,
                        web_server=Session.default_web,
                        files_server=Session.default_files,
                        credentials=dict(
                            access_key=Session.default_key,
                            secret_key=Session.default_secret,
                        ),
                    )
                }
                with open(get_config_file(), "wt") as f:
                    lines = json.dumps(conf, indent=4).split("\n")
                    f.write("\n".join(lines[1:-1]))

    @classmethod
    def debug_simulate_remote_task(cls, task_id: str, reset_task: bool = False) -> ():
        """
        Simulate remote execution of a specified Task.
        This call will simulate the behaviour of your Task as if executed by the ClearML-Agent
        This means configurations will be coming from the backend server into the code
        (the opposite from manual execution, where the backend logs the code arguments)
        Use with care.

        :param task_id: Task ID to simulate, notice that all configuration will be taken from the specified
            Task, regardless of the code initial values, just like it as if executed by ClearML agent
        :param reset_task: If True, target Task, is automatically cleared / reset.
        """

        # if we are already running remotely, do nothing
        if running_remotely():
            return

        # verify Task ID exists
        task = Task.get_task(task_id=task_id)
        if not task:
            raise ValueError("Task ID '{}' could not be found".format(task_id))

        if reset_task:
            task.reset(set_started_on_success=False, force=True)

        from .config.remote import override_current_task_id
        from .config.defs import LOG_TO_BACKEND_ENV_VAR

        override_current_task_id(task_id)
        LOG_TO_BACKEND_ENV_VAR.set(True)
        DEBUG_SIMULATE_REMOTE_TASK.set(True)

    def get_executed_queue(self, return_name: bool = False) -> Optional[str]:
        """
        Get the queue the task was executed on.

        :param return_name: If True, return the name of the queue. Otherwise, return its ID

        :return: Return the ID or name of the queue the task was executed on.
            If no queue was found, return None
        """
        queue_id = self.data.execution.queue
        if not return_name or not queue_id:
            return queue_id
        try:
            queue_name_result = Task._send(Task._get_default_session(), queues.GetByIdRequest(queue_id))
            return queue_name_result.response.queue.name
        except Exception as e:
            getLogger().warning("Could not get name of queue with ID '{}': {}".format(queue_id, e))
            return None

    @classmethod
    def _create(
        cls,
        project_name: Optional[str] = None,
        task_name: Optional[str] = None,
        task_type: "Task.TaskTypes" = TaskTypes.training,
    ) -> TaskInstance:
        """
        Create a new unpopulated Task (experiment).

        :param str project_name: The name of the project in which the experiment will be created.
            If ``project_name`` is ``None``, and the main execution Task is initialized (see :meth:`Task.init`),
            then the main execution Task's project is used. Otherwise, if the project does
            not exist, it is created. (Optional)
        :param str task_name: The name of Task (experiment).
        :param TaskTypes task_type: The task type.

        :return: The newly created task created.
        :rtype: Task
        """
        if not project_name:
            if not cls.__main_task:
                raise ValueError(
                    "Please provide project_name, no global task context found "
                    "(Task.current_task hasn't been called)"
                )
            project_name = cls.__main_task.get_project_name()

        try:
            task = cls(
                private=cls.__create_protection,
                project_name=project_name,
                task_name=task_name,
                task_type=task_type,
                log_to_backend=False,
                force_create=True,
            )
        except Exception:
            raise
        return task

    def _set_model_config(
        self,
        config_text: Optional[str] = None,
        config_dict: Optional[Mapping] = None,
    ) -> None:
        """
        Set Task model configuration text/dict

        :param config_text: model configuration (unconstrained text string). usually the content
            of a configuration file. If `config_text` is not None, `config_dict` must not be provided.
        :param config_dict: model configuration parameters dictionary.
            If `config_dict` is not None, `config_text` must not be provided.
        """
        # noinspection PyProtectedMember
        design = OutputModel._resolve_config(config_text=config_text, config_dict=config_dict)
        super(Task, self)._set_model_design(design=design)

    def _get_model_config_text(self) -> str:
        """
        Get Task model configuration text (before creating an output model)
        When an output model is created it will inherit these properties

        :return: The model config_text (unconstrained text string).
        """
        return super(Task, self).get_model_design()

    def _get_model_config_dict(self) -> Dict:
        """
        Get Task model configuration dictionary (before creating an output model)
        When an output model is created it will inherit these properties

        :return: config_dict: model configuration parameters dictionary.
        """
        config_text = self._get_model_config_text()
        # noinspection PyProtectedMember
        return OutputModel._text_to_config_dict(config_text)

    def _set_startup_info(self) -> ():
        self._set_runtime_properties(
            runtime_properties={
                "CLEARML VERSION": self.session.client,
                "CLI": sys.argv[0],
                "progress": "0",
            }
        )

    @classmethod
    def _reset_current_task_obj(cls) -> None:
        if not cls.__main_task:
            return
        task = cls.__main_task
        cls.__main_task = None
        cls.__forked_proc_main_pid = None
        if task._dev_worker:
            task._dev_worker.unregister()
            task._dev_worker = None

    @classmethod
    def _has_current_task_obj(cls) -> bool:
        return bool(cls.__main_task)

    @classmethod
    def _create_dev_task(
        cls,
        default_project_name,
        default_task_name,
        default_task_type,
        tags,
        reuse_last_task_id,
        continue_last_task=False,
        detect_repo=True,
        auto_connect_streams=True,
    ):
        if not default_project_name or not default_task_name:
            # get project name and task name from repository name and entry_point
            result, _ = ScriptInfo.get(create_requirements=False, check_uncommitted=False)
            if not default_project_name:
                # noinspection PyBroadException
                try:
                    parts = result.script["repository"].split("/")
                    default_project_name = (parts[-1] or parts[-2]).replace(".git", "") or "Untitled"
                except Exception:
                    default_project_name = "Untitled"
            if not default_task_name:
                # noinspection PyBroadException
                try:
                    default_task_name = os.path.splitext(os.path.basename(result.script["entry_point"]))[0]
                except Exception:
                    pass

        # conform reuse_last_task_id and continue_last_task
        if continue_last_task and isinstance(continue_last_task, str):
            reuse_last_task_id = continue_last_task
            continue_last_task = True
        elif isinstance(continue_last_task, int) and continue_last_task is not True:
            # allow initial offset environment override
            continue_last_task = continue_last_task

        if TASK_SET_ITERATION_OFFSET.get() is not None:
            continue_last_task = TASK_SET_ITERATION_OFFSET.get()

        # if we force no task reuse from os environment
        if DEV_TASK_NO_REUSE.get() or not reuse_last_task_id or isinstance(reuse_last_task_id, str):
            default_task = None
        else:
            # if we have a previous session to use, get the task id from it
            default_task = cls.__get_last_used_task_id(
                default_project_name,
                default_task_name,
                default_task_type.value,
            )

        closed_old_task = False
        default_task_id = None
        task = None
        in_dev_mode = not running_remotely()

        if in_dev_mode:
            if isinstance(reuse_last_task_id, str) and reuse_last_task_id:
                default_task_id = reuse_last_task_id
            elif not reuse_last_task_id or not cls.__task_is_relevant(default_task):
                default_task_id = None
            else:
                default_task_id = default_task.get("id") if default_task else None

            if default_task_id:
                try:
                    task = cls(
                        private=cls.__create_protection,
                        task_id=default_task_id,
                        log_to_backend=True,
                    )

                    # instead of resting the previously used task we are continuing the training with it.
                    if task and (
                        continue_last_task
                        or (isinstance(continue_last_task, int) and not isinstance(continue_last_task, bool))
                    ):
                        task.reload()
                        task.mark_started(force=True)
                        # allow to disable the
                        if continue_last_task is True:
                            task.set_initial_iteration(task.get_last_iteration() + 1)
                        else:
                            task.set_initial_iteration(continue_last_task)

                    else:
                        task_tags = task.data.system_tags if hasattr(task.data, "system_tags") else task.data.tags
                        task_artifacts = (
                            task.data.execution.artifacts if hasattr(task.data.execution, "artifacts") else None
                        )
                        if (
                            (
                                task._status
                                in (
                                    cls.TaskStatusEnum.published,
                                    cls.TaskStatusEnum.closed,
                                )
                            )
                            or task.output_models_id
                            or (cls.archived_tag in task_tags)
                            or (cls._development_tag not in task_tags)
                            or task_artifacts
                        ):
                            # If the task is published or closed, we shouldn't reset it so we can't use it in dev mode
                            # If the task is archived, or already has an output model,
                            #  we shouldn't use it in development mode either
                            default_task_id = None
                            task = None
                        else:
                            with task._edit_lock:
                                # from now on, there is no need to reload, we just clear stuff,
                                # this flag will be cleared off once we actually refresh at the end of the function
                                task._reload_skip_flag = True
                                # reset the task, so we can update it
                                task.reset(set_started_on_success=False, force=False)
                                # clear the heaviest stuff first
                                task._clear_task(
                                    system_tags=[cls._development_tag],
                                    comment=make_message("Auto-generated at %(time)s by %(user)s@%(host)s"),
                                )

                except (Exception, ValueError):
                    # we failed reusing task, create a new one
                    default_task_id = None

        # create a new task
        if not default_task_id:
            task = cls(
                private=cls.__create_protection,
                project_name=default_project_name,
                task_name=default_task_name,
                task_type=default_task_type,
                log_to_backend=True,
            )
            # no need to reload yet, we clear this before the end of the function
            task._reload_skip_flag = True

        if in_dev_mode:
            # update this session, for later use
            cls.__update_last_used_task_id(
                default_project_name,
                default_task_name,
                default_task_type.value,
                task.id,
            )
            # set default docker image from env.
            task._set_default_docker_image()

        # mark us as the main Task, there should only be one dev Task at a time.
        if not Task.__main_task:
            Task.__forked_proc_main_pid = os.getpid()
            Task.__main_task = task

        # mark the task as started
        task.started()
        # reload, making sure we are synced
        task._reload_skip_flag = False
        task.reload()

        # add Task tags
        if tags:
            task.add_tags([tags] if isinstance(tags, str) else tags)

        # force update of base logger to this current task (this is the main logger task)
        logger = task._get_logger(auto_connect_streams=auto_connect_streams)
        if closed_old_task:
            logger.report_text("ClearML Task: Closing old development task id={}".format(default_task.get("id")))
        # print warning, reusing/creating a task
        if default_task_id and not continue_last_task:
            logger.report_text("ClearML Task: overwriting (reusing) task id=%s" % task.id)
        elif default_task_id and continue_last_task:
            logger.report_text(
                "ClearML Task: continuing previous task id=%s Notice this run will not be reproducible!" % task.id
            )
        else:
            logger.report_text("ClearML Task: created new task id=%s" % task.id)

        # update current repository and put warning into logs
        if detect_repo:
            # noinspection PyBroadException
            try:
                import traceback

                stack = traceback.extract_stack(limit=10)
                # NOTICE WE ARE ALWAYS 3 down from caller in stack!
                for i in range(len(stack) - 1, 0, -1):
                    # look for the Task.init call, then the one above it is the callee module
                    if stack[i].name == "init":
                        task._calling_filename = os.path.abspath(stack[i - 1].filename)
                        break
            except Exception:
                pass
            if in_dev_mode and cls.__detect_repo_async:
                task._detect_repo_async_thread = threading.Thread(target=task._update_repository)
                task._detect_repo_async_thread.daemon = True
                task._detect_repo_async_thread.start()
            else:
                task._update_repository()

        # make sure we see something in the UI
        thread = threading.Thread(target=LoggerRoot.flush)
        thread.daemon = True
        thread.start()

        return task

    def _get_logger(
        self,
        flush_period: Optional[float] = NotSet,
        auto_connect_streams: Union[bool, dict] = False,
    ) -> Logger:
        """
        get a logger object for reporting based on the task

        :param flush_period: The period of the logger flush.
            If None of any other False value, will not flush periodically.
            If a logger was created before, this will be the new period and
            the old one will be discarded.

        :return: Logger object
        """

        if not self._logger:
            # do not recreate logger after task was closed/quit
            if self._at_exit_called and self._at_exit_called in (
                True,
                get_current_thread_id(),
            ):
                raise ValueError("Cannot use Task Logger after task was closed")
            # Get a logger object
            self._logger = Logger(
                private_task=self,
                connect_stdout=(auto_connect_streams is True)
                or (isinstance(auto_connect_streams, dict) and auto_connect_streams.get("stdout", False)),
                connect_stderr=(auto_connect_streams is True)
                or (isinstance(auto_connect_streams, dict) and auto_connect_streams.get("stderr", False)),
                connect_logging=isinstance(auto_connect_streams, dict) and auto_connect_streams.get("logging", False),
            )
            # make sure we set our reported to async mode
            # we make sure we flush it in self._at_exit
            self._reporter.async_enable = True
            # if we just created the logger, set default flush period
            if not flush_period or flush_period is self.NotSet:
                flush_period = float(DevWorker.report_period)

        if isinstance(flush_period, (int, float)):
            flush_period = int(abs(flush_period))

        if flush_period is None or isinstance(flush_period, int):
            self._logger.set_flush_period(flush_period)

        return self._logger

    def _connect_output_model(self, model: OutputModel, name: Optional[str] = None, **kwargs: Any) -> OutputModel:
        assert isinstance(model, OutputModel)
        model.connect(self, name=name, ignore_remote_overrides=False)
        return model

    def _save_output_model(self, model: OutputModel) -> None:
        """
        Deprecated: Save a reference to the connected output model.

        :param model: The connected output model
        """
        # deprecated
        self._connected_output_model = model

    def _handle_ignore_remote_overrides(self, overrides_name: str, ignore_remote_overrides: bool) -> bool:
        if self.running_locally() and ignore_remote_overrides:
            self.set_parameter(
                overrides_name,
                True,
                description="If True, ignore UI/backend overrides when running remotely."
                " Set it to False if you would like the overrides to be applied",
                value_type=bool,
            )
        elif not self.running_locally():
            ignore_remote_overrides = self.get_parameter(overrides_name, default=ignore_remote_overrides, cast=True)
        return ignore_remote_overrides

    def _reconnect_output_model(self) -> None:
        """
        Deprecated: If there is a saved connected output model, connect it again.

        This is needed if the input model is connected after the output model
        is connected, an then we will have to get the model design from the
        input model by reconnecting.
        """
        # Deprecated:
        if self._connected_output_model:
            self.connect(self._connected_output_model)

    def _connect_input_model(
        self,
        model: InputModel,
        name: Optional[str] = None,
        ignore_remote_overrides: bool = False,
    ) -> InputModel:
        assert isinstance(model, InputModel)
        # we only allow for an input model to be connected once
        # at least until we support multiple input models
        # notice that we do not check the task's input model because we allow task reuse and overwrite
        # add into comment that we are using this model

        # refresh comment
        comment = self._reload_field("comment") or self.comment or ""

        if not comment.endswith("\n"):
            comment += "\n"
        comment += "Using model id: {}".format(model.id)
        self.set_comment(comment)

        model.connect(self, name, ignore_remote_overrides=ignore_remote_overrides)
        return model

    def _connect_argparse(
        self,
        parser,
        args=None,
        namespace=None,
        parsed_args=None,
        name=None,
        ignore_remote_overrides=False,
    ):
        # do not allow argparser to connect to jupyter notebook
        # noinspection PyBroadException
        try:
            if "IPython" in sys.modules:
                # noinspection PyPackageRequirements
                from IPython import get_ipython  # noqa

                ip = get_ipython()
                if ip is not None and "IPKernelApp" in ip.config:
                    return parser
        except Exception:
            pass

        if self.is_main_task():
            argparser_update_currenttask(self)

        if (parser is None or parsed_args is None) and argparser_parseargs_called():
            # if we have a parser but nor parsed_args, we need to find the parser
            if parser and not parsed_args:
                for _parser, _parsed_args in get_argparser_last_args():
                    if _parser == parser:
                        parsed_args = _parsed_args
                        break
            else:
                # prefer the first argparser (hopefully it is more relevant?!
                for _parser, _parsed_args in get_argparser_last_args():
                    if parser is None:
                        parser = _parser
                    if parsed_args is None and parser == _parser:
                        parsed_args = _parsed_args

        if running_remotely() and (self.is_main_task() or self._is_remote_main_task()) and not ignore_remote_overrides:
            self._arguments.copy_to_parser(parser, parsed_args)
        else:
            self._arguments.copy_defaults_from_argparse(parser, args=args, namespace=namespace, parsed_args=parsed_args)
        return parser

    def _connect_dictionary(
        self,
        dictionary: dict,
        name: Optional[str] = None,
        ignore_remote_overrides: bool = False,
    ) -> dict:
        def _update_args_dict(task: Task, config_dict: Dict) -> None:
            # noinspection PyProtectedMember
            task._arguments.copy_from_dict(flatten_dictionary(config_dict), prefix=name)

        def _refresh_args_dict(task: Task, config_proxy_dict: ProxyDictPostWrite) -> None:
            # reread from task including newly added keys
            # noinspection PyProtectedMember
            a_flat_dict = task._arguments.copy_to_dict(flatten_dictionary(config_proxy_dict), prefix=name)
            # noinspection PyProtectedMember
            nested_dict = config_proxy_dict._to_dict()
            config_proxy_dict.clear()
            config_proxy_dict._do_update(nested_from_flat_dictionary(nested_dict, a_flat_dict))

        def _check_keys(dict_: dict, warning_sent: bool = False) -> None:
            if warning_sent:
                return
            for k, v in dict_.items():
                if warning_sent:
                    return
                if not isinstance(k, str):
                    getLogger().warning(
                        "Unsupported key of type '{}' found when connecting dictionary. It will be converted to str".format(
                            type(k)
                        )
                    )
                    warning_sent = True
                if isinstance(v, dict):
                    _check_keys(v, warning_sent)

        if (
            not running_remotely()
            or not (self.is_main_task() or self._is_remote_main_task())
            or ignore_remote_overrides
        ):
            _check_keys(dictionary)
            flat_dict = {str(k): v for k, v in flatten_dictionary(dictionary).items()}
            self._arguments.copy_from_dict(flat_dict, prefix=name)
            dictionary = ProxyDictPostWrite(self, _update_args_dict, **dictionary)
        else:
            flat_dict = flatten_dictionary(dictionary)
            flat_dict = self._arguments.copy_to_dict(flat_dict, prefix=name)
            dictionary = nested_from_flat_dictionary(dictionary, flat_dict)
            dictionary = ProxyDictPostWrite(self, _refresh_args_dict, **dictionary)

        return dictionary

    def _connect_task_parameters(self, attr_class, name=None, ignore_remote_overrides=False):
        ignore_remote_overrides_section = "_ignore_remote_overrides_"
        if running_remotely():
            ignore_remote_overrides = self.get_parameter(
                (name or "General") + "/" + ignore_remote_overrides_section,
                default=ignore_remote_overrides,
                cast=True,
            )
        if running_remotely() and (self.is_main_task() or self._is_remote_main_task()) and not ignore_remote_overrides:
            parameters = self.get_parameters(cast=True)
            if name:
                parameters = dict(
                    (k[len(name) + 1 :], v) for k, v in parameters.items() if k.startswith("{}/".format(name))
                )
            parameters.pop(ignore_remote_overrides_section, None)
            attr_class.update_from_dict(parameters)
        else:
            parameters_dict = attr_class.to_dict()
            if ignore_remote_overrides:
                parameters_dict[ignore_remote_overrides_section] = True
            self.set_parameters(parameters_dict, __parameters_prefix=name)
        return attr_class

    def _connect_object(self, an_object, name=None, ignore_remote_overrides=False):
        def verify_type(key, value):
            if str(key).startswith("_") or not isinstance(value, self._parameters_allowed_types):
                return False
            # verify everything is json able (i.e. basic types)
            try:
                json.dumps(value)
                return True
            except TypeError:
                return False

        a_dict = {
            k: v
            for cls_ in getattr(an_object, "__mro__", [an_object])
            for k, v in cls_.__dict__.items()
            if verify_type(k, v)
        }
        if running_remotely() and (self.is_main_task() or self._is_remote_main_task()) and not ignore_remote_overrides:
            a_dict = self._connect_dictionary(a_dict, name, ignore_remote_overrides=ignore_remote_overrides)
            for k, v in a_dict.items():
                if getattr(an_object, k, None) != a_dict[k]:
                    setattr(an_object, k, v)

            return an_object
        else:
            self._connect_dictionary(a_dict, name, ignore_remote_overrides=ignore_remote_overrides)
            return an_object

    def _dev_mode_stop_task(self, stop_reason: str, pid: Optional[int] = None) -> None:
        # make sure we do not get called (by a daemon thread) after at_exit
        if self._at_exit_called:
            return

        self.log.warning("### TASK STOPPED - USER ABORTED - {} ###".format(stop_reason.upper().replace("_", " ")))
        self.flush(wait_for_uploads=True)

        # if running remotely, we want the daemon to kill us
        if self.running_locally():
            self.stopped(status_reason="USER ABORTED")

        if self._dev_worker:
            self._dev_worker.unregister()

        # NOTICE! This will end the entire execution tree!
        if self.__exit_hook:
            self.__exit_hook.remote_user_aborted = True
        self._kill_all_child_processes(send_kill=False, pid=pid, allow_kill_calling_pid=False)
        time.sleep(2.0)
        self._kill_all_child_processes(send_kill=True, pid=pid, allow_kill_calling_pid=True)
        os._exit(1)  # noqa

    @staticmethod
    def _kill_all_child_processes(send_kill=False, pid=None, allow_kill_calling_pid=True):
        # get current process if pid not provided
        current_pid = os.getpid()
        kill_ourselves = None
        pid = pid or current_pid
        try:
            parent = psutil.Process(pid)
        except psutil.Error:
            # could not find parent process id
            return
        for child in parent.children(recursive=True):
            # kill ourselves last (if we need to)
            if child.pid == current_pid:
                kill_ourselves = child
                continue
            if send_kill:
                child.kill()
            else:
                child.terminate()

        # parent ourselves
        if allow_kill_calling_pid or parent.pid != current_pid:
            if send_kill:
                parent.kill()
            else:
                parent.terminate()

        # kill ourselves if we need to:
        if allow_kill_calling_pid and kill_ourselves:
            if send_kill:
                kill_ourselves.kill()
            else:
                kill_ourselves.terminate()

    def _dev_mode_setup_worker(self):
        if (
            (running_remotely() and not DEBUG_SIMULATE_REMOTE_TASK.get())
            or not self.is_main_task()
            or self._at_exit_called
            or self._offline_mode
        ):
            return

        if self._dev_worker:
            return self._dev_worker

        self._dev_worker = DevWorker()
        self._dev_worker.register(self)

        logger = self.get_logger()
        flush_period = logger.get_flush_period()
        if not flush_period or flush_period > self._dev_worker.report_period:
            logger.set_flush_period(self._dev_worker.report_period)

    def _wait_for_repo_detection(self, timeout: Optional[float] = None) -> None:
        # wait for detection repo sync
        if not self._detect_repo_async_thread:
            return
        with self._repo_detect_lock:
            if not self._detect_repo_async_thread:
                return
            # noinspection PyBroadException
            try:
                if self._detect_repo_async_thread.is_alive():
                    # if negative timeout, just kill the thread:
                    if timeout is not None and timeout < 0:
                        from .utilities.lowlevel.threads import kill_thread

                        kill_thread(self._detect_repo_async_thread)
                    else:
                        self.log.info("Waiting for repository detection and full package requirement analysis")
                        self._detect_repo_async_thread.join(timeout=timeout)
                        # because join has no return value
                        if self._detect_repo_async_thread.is_alive():
                            self.log.info(
                                "Repository and package analysis timed out ({} sec), giving up".format(timeout)
                            )
                            # done waiting, kill the thread
                            from .utilities.lowlevel.threads import kill_thread

                            kill_thread(self._detect_repo_async_thread)
                        else:
                            self.log.info("Finished repository detection and package analysis")
                self._detect_repo_async_thread = None
            except Exception:
                pass

    def _summary_artifacts(self) -> None:
        # signal artifacts upload, and stop daemon
        self._artifacts_manager.stop(wait=True)
        # print artifacts summary (if not empty)
        if self._artifacts_manager.summary:
            self.get_logger().report_text(self._artifacts_manager.summary)

    def _at_exit(self) -> None:
        # protect sub-process at_exit (should never happen)
        if self._at_exit_called and self._at_exit_called != get_current_thread_id():
            return

        # make sure we do not try to use events, because Python might deadlock itself.
        # https://bugs.python.org/issue41606
        if self.__is_subprocess():
            BackgroundMonitor.set_at_exit_state(True)

        # shutdown will clear the main, so we have to store it before.
        # is_main = self.is_main_task()
        # fix debugger signal in the middle, catch everything
        try:
            self.__shutdown()
        except:  # noqa
            pass
        # In rare cases we might need to forcefully shutdown the process, currently we should avoid it.
        # if is_main:
        #     # we have to forcefully shutdown if we have forked processes, sometimes they will get stuck
        #     os._exit(self.__exit_hook.exit_code if self.__exit_hook and self.__exit_hook.exit_code else 0)

    def __shutdown(self) -> None:
        """
        Will happen automatically once we exit code, i.e. atexit
        :return:
        """
        # protect sub-process at_exit
        if self._at_exit_called:
            is_sub_process = self.__is_subprocess()
            # if we are called twice (signal in the middle of the shutdown),
            _nested_shutdown_call = bool(self._at_exit_called == get_current_thread_id())
            if _nested_shutdown_call and not is_sub_process:
                # if we were called again in the main thread on the main process, let's try again
                # make sure we only do this once
                self._at_exit_called = True
            else:
                # make sure we flush stdout, this is the best we can do.
                if _nested_shutdown_call and self._logger and is_sub_process:
                    # noinspection PyProtectedMember
                    self._logger._close_stdout_handler(wait=True)
                    self._at_exit_called = True
                # if we get here, we should do nothing and leave
                return
        else:
            # from here only a single thread can re-enter
            self._at_exit_called = get_current_thread_id()

        LoggerRoot.clear_logger_handlers()

        # disable lock on signal callbacks, to avoid deadlocks.
        if self.__exit_hook and self.__exit_hook.signal is not None:
            self.__edit_lock = False

        is_sub_process = self.__is_subprocess()

        task_status = None
        # noinspection PyBroadException
        try:
            wait_for_uploads = True
            # first thing mark task as stopped, so we will not end up with "running" on lost tasks
            # if we are running remotely, the daemon will take care of it
            wait_for_std_log = True
            if (
                (not running_remotely() or DEBUG_SIMULATE_REMOTE_TASK.get())
                and self.is_main_task()
                and not is_sub_process
            ):
                # check if we crashed, ot the signal is not interrupt (manual break)
                task_status = ("stopped",)
                if self.__exit_hook:
                    is_exception = self.__exit_hook.exception
                    # check if we are running inside a debugger
                    if not is_exception and sys.modules.get("pydevd"):
                        # noinspection PyBroadException
                        try:
                            is_exception = sys.last_type
                        except Exception:
                            pass

                        # check if this is Jupyter interactive session, do not mark as exception
                        if "IPython" in sys.modules:
                            is_exception = None

                    # only if we have an exception (and not ctrl-break) or signal is not SIGTERM / SIGINT
                    if (
                        is_exception
                        and not isinstance(is_exception, KeyboardInterrupt)
                        and is_exception != KeyboardInterrupt
                    ) or (
                        not self.__exit_hook.remote_user_aborted
                        and (self.__exit_hook.signal not in (None, 2, 15) or self.__exit_hook.exit_code)
                    ):
                        task_status = (
                            "failed",
                            (
                                "Exception {}".format(is_exception)
                                if is_exception
                                else "Signal {}".format(self.__exit_hook.signal)
                            ),
                        )
                        wait_for_uploads = False
                    else:
                        wait_for_uploads = self.__exit_hook.remote_user_aborted or self.__exit_hook.signal is None
                        if (
                            not self.__exit_hook.remote_user_aborted
                            and self.__exit_hook.signal is None
                            and not is_exception
                        ):
                            task_status = ("completed",)
                        else:
                            task_status = ("stopped",)
                            # user aborted. do not bother flushing the stdout logs
                            wait_for_std_log = self.__exit_hook.signal is not None

            # wait for repository detection (if we didn't crash)
            if wait_for_uploads and self._logger:
                # we should print summary here
                self._summary_artifacts()
                # make sure that if we crashed the thread we are not waiting forever
                if not is_sub_process:
                    self._wait_for_repo_detection(timeout=10.0)

            # kill the repo thread (negative timeout, do not wait), if it hasn't finished yet.
            if not is_sub_process:
                self._wait_for_repo_detection(timeout=-1)

            # wait for uploads
            print_done_waiting = False
            if wait_for_uploads and (
                BackendModel.get_num_results() > 0 or (self.__reporter and self.__reporter.events_waiting())
            ):
                self.log.info("Waiting to finish uploads")
                print_done_waiting = True
            # from here, do not send log in background thread
            if wait_for_uploads:
                self.flush(wait_for_uploads=True)
                # wait until the reporter flush everything
                if self.__reporter:
                    self.__reporter.stop()
                    if self.is_main_task():
                        # notice: this will close the reporting for all the Tasks in the system
                        Metrics.close_async_threads()
                        # notice: this will close the jupyter monitoring
                        ScriptInfo.close()
                if self.is_main_task():
                    # noinspection PyBroadException
                    try:
                        from .storage.helper import StorageHelper

                        StorageHelper.close_async_threads()
                    except Exception:
                        pass

                if print_done_waiting:
                    self.log.info("Finished uploading")
            # elif self._logger:
            #     # noinspection PyProtectedMember
            #     self._logger._flush_stdout_handler()

            # from here, do not check worker status
            if self._dev_worker:
                self._dev_worker.unregister()
                self._dev_worker = None

            # stop resource monitoring
            if self._resource_monitor:
                self._resource_monitor.stop()
                self._resource_monitor = None

            if self._logger:
                self._logger.set_flush_period(None)
                # noinspection PyProtectedMember
                self._logger._close_stdout_handler(wait=wait_for_uploads or wait_for_std_log)

            if not is_sub_process:
                # change task status
                if not task_status:
                    pass
                elif task_status[0] == "failed":
                    self.mark_failed(status_reason=task_status[1])
                elif task_status[0] == "completed":
                    self.set_progress(100)
                    self.mark_completed()
                elif task_status[0] == "stopped":
                    self.stopped()

            # this is so in theory we can close a main task and start a new one
            if self.is_main_task():
                Task.__main_task = None
                Task.__forked_proc_main_pid = None
                Task.__update_master_pid_task(task=None)
        except Exception:
            # make sure we do not interrupt the exit process
            pass

        # make sure we store last task state
        if self._offline_mode and not is_sub_process:
            # noinspection PyBroadException
            try:
                # make sure the state of the offline data is saved
                self._edit()
                # create zip file
                offline_folder = self.get_offline_mode_folder()
                zip_file = offline_folder.as_posix() + ".zip"
                with ZipFile(zip_file, "w", allowZip64=True, compression=ZIP_DEFLATED) as zf:
                    for filename in offline_folder.rglob("*"):
                        if filename.is_file():
                            relative_file_name = filename.relative_to(offline_folder).as_posix()
                            zf.write(filename.as_posix(), arcname=relative_file_name)
                print("ClearML Task: Offline session stored in {}".format(zip_file))
            except Exception:
                pass

        # delete locking object (lock file)
        if self._edit_lock:
            # noinspection PyBroadException
            try:
                del self.__edit_lock
            except Exception:
                pass
            self._edit_lock = None

        # make sure no one will re-enter the shutdown method
        self._at_exit_called = True
        if not is_sub_process and BackgroundMonitor.is_subprocess_enabled():
            BackgroundMonitor.wait_for_sub_process(self)

        # we are done
        return

    @classmethod
    def _remove_exception_hooks(cls) -> None:
        if cls.__exit_hook:
            cls.__exit_hook.remove_exception_hooks()

    @classmethod
    def _remove_signal_hooks(cls) -> None:
        if cls.__exit_hook:
            cls.__exit_hook.remove_signal_hooks()

    @classmethod
    def __register_at_exit(cls, exit_callback: Callable) -> None:
        if cls.__exit_hook is None:
            # noinspection PyBroadException
            try:
                cls.__exit_hook = ExitHooks(exit_callback)
                cls.__exit_hook.hook()
            except Exception:
                cls.__exit_hook = None
        else:
            cls.__exit_hook.update_callback(exit_callback)

    @classmethod
    def __get_task(
        cls,
        task_id: Optional[str] = None,
        project_name: Optional[str] = None,
        task_name: Optional[str] = None,
        include_archived: bool = True,
        tags: Optional[Sequence[str]] = None,
        task_filter: Optional[dict] = None,
    ) -> TaskInstance:
        if task_id:
            return cls(private=cls.__create_protection, task_id=task_id, log_to_backend=False)

        if project_name:
            res = cls._send(
                cls._get_default_session(),
                projects.GetAllRequest(name=exact_match_regex(project_name)),
            )
            project = get_single_result(entity="project", query=project_name, results=res.response.projects)
        else:
            project = None

        # get default session, before trying to access tasks.Task so that we do not create two sessions.
        session = cls._get_default_session()
        system_tags = "system_tags" if hasattr(tasks.Task, "system_tags") else "tags"
        task_filter = task_filter or {}
        if not include_archived:
            task_filter["system_tags"] = (task_filter.get("system_tags") or []) + ["-{}".format(cls.archived_tag)]
        if tags:
            task_filter["tags"] = (task_filter.get("tags") or []) + list(tags)
        res = cls._send(
            session,
            tasks.GetAllRequest(
                project=[project.id] if project else None,
                name=exact_match_regex(task_name) if task_name else None,
                only_fields=["id", "name", "last_update", system_tags],
                **task_filter,
            ),
        )
        res_tasks = res.response.tasks
        # if we have more than one result, filter out the 'archived' results
        # notice that if we only have one result we do get the archived one as well.
        if len(res_tasks) > 1:
            filtered_tasks = [
                t
                for t in res_tasks
                if not getattr(t, system_tags, None) or cls.archived_tag not in getattr(t, system_tags, None)
            ]
            # if we did not filter everything (otherwise we have only archived tasks, so we return them)
            if filtered_tasks:
                res_tasks = filtered_tasks

        task = get_single_result(
            entity="task",
            query={
                k: v
                for k, v in dict(
                    project_name=project_name,
                    task_name=task_name,
                    tags=tags,
                    include_archived=include_archived,
                    task_filter=task_filter,
                ).items()
                if v
            },
            results=res_tasks,
            raise_on_error=False,
        )
        if not task:
            # should never happen
            return None  # noqa

        return cls(
            private=cls.__create_protection,
            task_id=task.id,
            log_to_backend=False,
        )

    @classmethod
    def __get_tasks(
        cls,
        task_ids: Optional[Sequence[str]] = None,
        project_name: Optional[Union[Sequence[str], str]] = None,
        task_name: Optional[str] = None,
        **kwargs: Any,
    ) -> List["Task"]:
        if task_ids:
            if isinstance(task_ids, six.string_types):
                task_ids = [task_ids]
            return [
                cls(
                    private=cls.__create_protection,
                    task_id=task_id,
                    log_to_backend=False,
                )
                for task_id in task_ids
            ]

        queried_tasks = cls._query_tasks(
            project_name=project_name, task_name=task_name, fetch_only_first_page=True, **kwargs
        )
        if len(queried_tasks) == 500:
            LoggerRoot.get_base_logger().warning(
                "Too many requests when calling Task.get_tasks()."
                " Returning only the first 500 results."
                " Use Task.query_tasks() to fetch all task IDs"
            )
        return [cls(private=cls.__create_protection, task_id=task.id, log_to_backend=False) for task in queried_tasks]

    @classmethod
    def _query_tasks(
        cls,
        task_ids: Optional[Union[Sequence[str], str]] = None,
        project_name: Optional[Union[Sequence[str], str]] = None,
        task_name: Optional[str] = None,
        fetch_only_first_page: bool = False,
        exact_match_regex_flag: bool = True,
        **kwargs: Any,
    ) -> List["Task"]:
        res = None
        if not task_ids:
            task_ids = None
        elif isinstance(task_ids, six.string_types):
            task_ids = [task_ids]

        if project_name and isinstance(project_name, str):
            project_names = [project_name]
        else:
            project_names = project_name

        project_ids = []
        projects_not_found = []
        if project_names:
            for name in project_names:
                aux_kwargs = {}
                if kwargs.get("_allow_extra_fields_"):
                    aux_kwargs["_allow_extra_fields_"] = True
                    aux_kwargs["search_hidden"] = kwargs.get("search_hidden", False)
                res = cls._send(
                    cls._get_default_session(),
                    projects.GetAllRequest(
                        name=(exact_match_regex(name) if exact_match_regex_flag else name), **aux_kwargs
                    ),
                )
                if res.response and res.response.projects:
                    project_ids.extend([project.id for project in res.response.projects])
                else:
                    projects_not_found.append(name)
            if projects_not_found:
                # If any of the given project names does not exist, fire off a warning
                LoggerRoot.get_base_logger().warning(
                    "No projects were found with name(s): {}".format(", ".join(projects_not_found))
                )
            if not project_ids:
                # If not a single project exists or was found, return empty right away
                return []

        session = cls._get_default_session()
        system_tags = "system_tags" if hasattr(tasks.Task, "system_tags") else "tags"
        only_fields = ["id", "name", "last_update", system_tags]

        if kwargs and kwargs.get("only_fields"):
            only_fields = list(set(kwargs.pop("only_fields")) | set(only_fields))

        # if we have specific page to look for, we should only get the requested one
        if not fetch_only_first_page and kwargs and "page" in kwargs:
            fetch_only_first_page = True

        ret_tasks = []
        page = -1
        page_size = 500
        while page == -1 or (not fetch_only_first_page and res and len(res.response.tasks) == page_size):
            page += 1
            # work on a copy and make sure we override all fields with ours
            request_kwargs = dict(
                id=task_ids,
                project=project_ids if project_ids else kwargs.pop("project", None),
                name=task_name if task_name else kwargs.pop("name", None),
                only_fields=only_fields,
                page=page,
                page_size=page_size,
            )
            # make sure we always override with the kwargs (specifically page selection / page_size)
            request_kwargs.update(kwargs or {})
            res = cls._send(
                session,
                tasks.GetAllRequest(**request_kwargs),
            )
            ret_tasks.extend(res.response.tasks)
        return ret_tasks

    @classmethod
    def _wait_for_deferred(cls, task: Optional["Task"]) -> None:
        """
        Make sure the task object deferred `Task.init` is completed.
        Accessing any of the `task` object's property will ensure the Task.init call was also complete
        This is an internal utility function

        :param task: Optional deferred Task object as returned form Task.init
        """
        if not task:
            return
        # force deferred init to complete
        task.id  # noqa

    @classmethod
    def __get_hash_key(cls, *args: Any) -> str:
        def normalize(x: Any) -> str:
            return "<{}>".format(x) if x is not None else ""

        return ":".join(map(normalize, args))

    @classmethod
    def __get_last_used_task_id(cls, default_project_name, default_task_name, default_task_type):
        hash_key = cls.__get_hash_key(
            cls._get_api_server(),
            default_project_name,
            default_task_name,
            default_task_type,
        )

        # check if we have a cached task_id we can reuse
        # it must be from within the last 24h and with the same project/name/type
        task_sessions = SessionCache.load_dict(str(cls))

        task_data = task_sessions.get(hash_key)
        if task_data is None:
            return None

        try:
            task_data["type"] = cls.TaskTypes(task_data["type"])
        except (ValueError, KeyError):
            LoggerRoot.get_base_logger().warning(
                "Corrupted session cache entry: {}. "
                "Unsupported task type: {}"
                "Creating a new task.".format(hash_key, task_data["type"]),
            )

            return None

        return task_data

    @classmethod
    def __update_last_used_task_id(cls, default_project_name, default_task_name, default_task_type, task_id):
        hash_key = cls.__get_hash_key(
            cls._get_api_server(),
            default_project_name,
            default_task_name,
            default_task_type,
        )

        task_id = str(task_id)
        # update task session cache
        task_sessions = SessionCache.load_dict(str(cls))
        last_task_session = {
            "time": time.time(),
            "project": default_project_name,
            "name": default_task_name,
            "type": default_task_type,
            "id": task_id,
        }

        # remove stale sessions
        for k in list(task_sessions.keys()):
            if (time.time() - task_sessions[k].get("time", 0)) > 60 * 60 * cls.__task_id_reuse_time_window_in_hours:
                task_sessions.pop(k)
        # update current session
        task_sessions[hash_key] = last_task_session
        # store
        SessionCache.store_dict(str(cls), task_sessions)

    @classmethod
    def __task_timed_out(cls, task_data):
        return (
            task_data
            and task_data.get("id")
            and task_data.get("time")
            and (time.time() - task_data.get("time")) > (60 * 60 * cls.__task_id_reuse_time_window_in_hours)
        )

    @classmethod
    def __get_task_api_obj(cls, task_id: str, only_fields: Optional[List[str]] = None) -> Optional["tasks.Task"]:
        if not task_id or cls._offline_mode:
            return None

        all_tasks = cls._send(
            cls._get_default_session(),
            tasks.GetAllRequest(id=[task_id], only_fields=only_fields),
        ).response.tasks

        # The task may not exist in environment changes
        if not all_tasks:
            return None

        return all_tasks[0]

    @classmethod
    def __task_is_relevant(cls, task_data: Mapping[str, Any]) -> bool:
        """
        Check that a cached task is relevant for reuse.

        A task is relevant for reuse if:
            1. It is not timed out i.e it was last use in the previous 24 hours.
            2. It's name, project and type match the data in the server, so not
               to override user changes made by using the UI.

        :param task_data: A mapping from 'id', 'name', 'project', 'type' keys
            to the task's values, as saved in the cache.

        :return: True, if the task is relevant for reuse. False, if not.
        """
        if not task_data:
            return False

        if cls.__task_timed_out(task_data):
            return False

        task_id = task_data.get("id")

        if not task_id:
            return False

        # noinspection PyBroadException
        try:
            task = cls.__get_task_api_obj(task_id, ("id", "name", "project", "type"))
        except Exception:
            task = None

        if task is None:
            return False

        project_name = None
        if task.project:
            # noinspection PyBroadException
            try:
                project = cls._send(
                    cls._get_default_session(),
                    projects.GetByIdRequest(project=task.project),
                ).response.project

                if project:
                    project_name = project.name
            except Exception:
                pass

        if (
            task_data.get("type")
            and task_data.get("type") not in (cls.TaskTypes.training, cls.TaskTypes.testing)
            and not Session.check_min_api_version(2.8)
        ):
            print(
                'WARNING: Changing task type to "{}" : '
                'clearml-server does not support task type "{}", '
                "please upgrade clearml-server.".format(cls.TaskTypes.training, task_data["type"].value)
            )
            task_data["type"] = cls.TaskTypes.training

        compares = (
            (task.name, "name"),
            (project_name, "project"),
            (task.type, "type"),
        )

        # compare after casting to string to avoid enum instance issues
        # remember we might have replaced the api version by now, so enums are different
        return all(
            six.text_type(server_data) == six.text_type(task_data.get(task_data_key))
            for server_data, task_data_key in compares
        )

    @classmethod
    def __close_timed_out_task(cls, task_data: Optional[Dict[str, Any]]) -> bool:
        if not task_data:
            return False

        task = cls.__get_task_api_obj(task_data.get("id"), ("id", "status"))

        if task is None:
            return False

        stopped_statuses = (
            cls.TaskStatusEnum.stopped,
            cls.TaskStatusEnum.published,
            cls.TaskStatusEnum.publishing,
            cls.TaskStatusEnum.closed,
            cls.TaskStatusEnum.failed,
            cls.TaskStatusEnum.completed,
        )

        if task.status not in stopped_statuses:
            cls._send(
                cls._get_default_session(),
                tasks.StoppedRequest(
                    task=task.id,
                    force=True,
                    status_message="Stopped timed out development task",
                ),
            )

            return True
        return False

    @classmethod
    def __add_model_wildcards(
        cls,
        auto_connect_frameworks: Union[Dict[str, Union[str, List[str], Tuple[str]]], Any],
    ) -> None:
        if isinstance(auto_connect_frameworks, dict):
            for k, v in auto_connect_frameworks.items():
                if isinstance(v, str):
                    v = [v]
                if isinstance(v, (list, tuple)):
                    WeightsFileHandler.model_wildcards[k] = [str(i) for i in v]

        def callback(_: Any, model_info: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            if not model_info:
                return None
            parents = Framework.get_framework_parents(model_info.framework)
            wildcards = []
            for parent in parents:
                if WeightsFileHandler.model_wildcards.get(parent):
                    wildcards.extend(WeightsFileHandler.model_wildcards[parent])
            if not wildcards:
                return model_info
            if not matches_any_wildcard(model_info.local_model_path, wildcards):
                return None
            return model_info

        WeightsFileHandler.add_pre_callback(callback)

    def __getstate__(self) -> dict:
        return {
            "main": self.is_main_task(),
            "id": self.id,
            "offline": self.is_offline(),
        }

    def __setstate__(self, state):
        if state["main"] and not self.__main_task:
            Task.__forked_proc_main_pid = None
            Task.__update_master_pid_task(task=state["id"])
        if state["offline"]:
            Task.set_offline(offline_mode=state["offline"])

        task = (
            Task.init(
                continue_last_task=state["id"],
                auto_connect_frameworks={"detect_repository": False},
            )
            if state["main"]
            else Task.get_task(task_id=state["id"])
        )
        self.__dict__ = task.__dict__

    @property
    def resource_monitor(self) -> None:
        return self._resource_monitor
