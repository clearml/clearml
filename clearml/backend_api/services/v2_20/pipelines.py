"""
pipelines service

Provides a management API for pipelines in the system.
"""
from typing import List, Optional, Any
import six
from clearml.backend_api.session import Request, Response, schema_property


class StartPipelineRequest(Request):
    """
    Start a pipeline

    :param task: ID of the task on which the pipeline will be based
    :type task: str
    :param queue: Queue ID in which the created pipeline task will be enqueued
    :type queue: str
    :param args: Task arguments, name/value to be placed in the hyperparameters
        Args section
    :type args: Sequence[dict]
    """

    _service = "pipelines"
    _action = "start_pipeline"
    _version = "2.20"
    _schema = {
        "definitions": {},
        "properties": {
            "args": {
                "description": "Task arguments, name/value to be placed in the hyperparameters Args section",
                "items": {
                    "properties": {
                        "name": {"type": "string"},
                        "value": {"type": ["string", "null"]},
                    },
                    "type": "object",
                },
                "type": "array",
            },
            "queue": {
                "description": "Queue ID in which the created pipeline task will be enqueued",
                "type": "string",
            },
            "task": {
                "description": "ID of the task on which the pipeline will be based",
                "type": "string",
            },
        },
        "required": ["task"],
        "type": "object",
    }

    def __init__(
        self, task: str, queue: Optional[str] = None, args: Optional[List[dict]] = None, **kwargs: Any
    ) -> None:
        super(StartPipelineRequest, self).__init__(**kwargs)
        self.task = task
        self.queue = queue
        self.args = args

    @schema_property("task")
    def task(self) -> str:
        return self._property_task

    @task.setter
    def task(self, value: str) -> None:
        if value is None:
            self._property_task = None
            return
        self.assert_isinstance(value, "task", six.string_types)
        self._property_task = value

    @schema_property("queue")
    def queue(self) -> Optional[str]:
        return self._property_queue

    @queue.setter
    def queue(self, value: Optional[str]) -> None:
        if value is None:
            self._property_queue = None
            return
        self.assert_isinstance(value, "queue", six.string_types)
        self._property_queue = value

    @schema_property("args")
    def args(self) -> Optional[List[dict]]:
        return self._property_args

    @args.setter
    def args(self, value: Optional[List[dict]]) -> None:
        if value is None:
            self._property_args = None
            return
        self.assert_isinstance(value, "args", (list, tuple))
        self.assert_isinstance(value, "args", (dict,), is_array=True)
        self._property_args = value


class StartPipelineResponse(Response):
    """
    Response of pipelines.start_pipeline endpoint.

    :param pipeline: ID of the new pipeline task
    :type pipeline: str
    :param enqueued: True if the task was successfully enqueued
    :type enqueued: bool
    """

    _service = "pipelines"
    _action = "start_pipeline"
    _version = "2.20"
    _schema = {
        "definitions": {},
        "properties": {
            "enqueued": {
                "description": "True if the task was successfully enqueued",
                "type": ["boolean", "null"],
            },
            "pipeline": {
                "description": "ID of the new pipeline task",
                "type": ["string", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, pipeline: Optional[str] = None, enqueued: Optional[bool] = None, **kwargs: Any) -> None:
        super(StartPipelineResponse, self).__init__(**kwargs)
        self.pipeline = pipeline
        self.enqueued = enqueued

    @schema_property("pipeline")
    def pipeline(self) -> Optional[str]:
        return self._property_pipeline

    @pipeline.setter
    def pipeline(self, value: Optional[str]) -> None:
        if value is None:
            self._property_pipeline = None
            return
        self.assert_isinstance(value, "pipeline", six.string_types)
        self._property_pipeline = value

    @schema_property("enqueued")
    def enqueued(self) -> Optional[bool]:
        return self._property_enqueued

    @enqueued.setter
    def enqueued(self, value: Optional[bool]) -> None:
        if value is None:
            self._property_enqueued = None
            return
        self.assert_isinstance(value, "enqueued", (bool,))
        self._property_enqueued = value


response_mapping = {StartPipelineRequest: StartPipelineResponse}
