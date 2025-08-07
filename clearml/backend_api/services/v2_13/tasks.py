"""
tasks service

Provides a management API for tasks in the system.
"""
from typing import List, Optional, Any
import enum
from datetime import datetime
import six
from clearml.backend_api.session import (
    Request,
    BatchRequest,
    Response,
    NonStrictDataModel,
    schema_property,
    StringEnum,
)
from dateutil.parser import parse as parse_datetime


class MultiFieldPatternData(NonStrictDataModel):
    """
    :param pattern: Pattern string (regex)
    :type pattern: str
    :param fields: List of field names
    :type fields: Sequence[str]
    """

    _schema = {
        "properties": {
            "fields": {
                "description": "List of field names",
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
            "pattern": {
                "description": "Pattern string (regex)",
                "type": ["string", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, pattern: Optional[str] = None, fields: Optional[List[str]] = None, **kwargs: Any) -> None:
        super(MultiFieldPatternData, self).__init__(**kwargs)
        self.pattern = pattern
        self.fields = fields

    @schema_property("pattern")
    def pattern(self) -> Optional[str]:
        return self._property_pattern

    @pattern.setter
    def pattern(self, value: Optional[str]) -> None:
        if value is None:
            self._property_pattern = None
            return
        self.assert_isinstance(value, "pattern", six.string_types)
        self._property_pattern = value

    @schema_property("fields")
    def fields(self) -> Optional[List[str]]:
        return self._property_fields

    @fields.setter
    def fields(self, value: Optional[List[str]]) -> None:
        if value is None:
            self._property_fields = None
            return
        self.assert_isinstance(value, "fields", (list, tuple))
        self.assert_isinstance(value, "fields", six.string_types, is_array=True)
        self._property_fields = value


class ModelTypeEnum(StringEnum):
    input = "input"
    output = "output"


class TaskModelItem(NonStrictDataModel):
    """
    :param name: The task model name
    :type name: str
    :param model: The model ID
    :type model: str
    """

    _schema = {
        "properties": {
            "model": {"description": "The model ID", "type": "string"},
            "name": {"description": "The task model name", "type": "string"},
        },
        "required": ["name", "model"],
        "type": "object",
    }

    def __init__(self, name: str, model: str, **kwargs: Any) -> None:
        super(TaskModelItem, self).__init__(**kwargs)
        self.name = name
        self.model = model

    @schema_property("name")
    def name(self) -> str:
        return self._property_name

    @name.setter
    def name(self, value: str) -> None:
        if value is None:
            self._property_name = None
            return
        self.assert_isinstance(value, "name", six.string_types)
        self._property_name = value

    @schema_property("model")
    def model(self) -> str:
        return self._property_model

    @model.setter
    def model(self, value: str) -> None:
        if value is None:
            self._property_model = None
            return
        self.assert_isinstance(value, "model", six.string_types)
        self._property_model = value


class Script(NonStrictDataModel):
    """
    :param binary: Binary to use when running the script
    :type binary: str
    :param repository: Name of the repository where the script is located
    :type repository: str
    :param tag: Repository tag
    :type tag: str
    :param branch: Repository branch ID If not provided and tag not provided,
        default repository branch is used.
    :type branch: str
    :param version_num: Version (changeset) number. Optional (default is head
        version) Unused if tag is provided.
    :type version_num: str
    :param entry_point: Path to execute within the repository
    :type entry_point: str
    :param working_dir: Path to the folder from which to run the script Default -
        root folder of repository
    :type working_dir: str
    :param requirements: A JSON object containing requirements strings by key
    :type requirements: dict
    :param diff: Uncommitted changes found in the repository when task was run
    :type diff: str
    """

    _schema = {
        "properties": {
            "binary": {
                "default": "python",
                "description": "Binary to use when running the script",
                "type": ["string", "null"],
            },
            "branch": {
                "description": "Repository branch id If not provided and tag not provided, default repository branch is used.",
                "type": ["string", "null"],
            },
            "diff": {
                "description": "Uncommitted changes found in the repository when task was run",
                "type": ["string", "null"],
            },
            "entry_point": {
                "description": "Path to execute within the repository",
                "type": ["string", "null"],
            },
            "repository": {
                "description": "Name of the repository where the script is located",
                "type": ["string", "null"],
            },
            "requirements": {
                "description": "A JSON object containing requirements strings by key",
                "type": ["object", "null"],
            },
            "tag": {"description": "Repository tag", "type": ["string", "null"]},
            "version_num": {
                "description": "Version (changeset) number. Optional (default is head version) Unused if tag is provided.",
                "type": ["string", "null"],
            },
            "working_dir": {
                "description": "Path to the folder from which to run the script Default - root folder of repository",
                "type": ["string", "null"],
            },
        },
        "type": "object",
    }

    def __init__(
        self,
        binary: Optional[str] = "python",
        repository: Optional[str] = None,
        tag: Optional[str] = None,
        branch: Optional[str] = None,
        version_num: Optional[str] = None,
        entry_point: Optional[str] = None,
        working_dir: Optional[str] = None,
        requirements: Optional[dict] = None,
        diff: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super(Script, self).__init__(**kwargs)
        self.binary = binary
        self.repository = repository
        self.tag = tag
        self.branch = branch
        self.version_num = version_num
        self.entry_point = entry_point
        self.working_dir = working_dir
        self.requirements = requirements
        self.diff = diff

    @schema_property("binary")
    def binary(self) -> Optional[str]:
        return self._property_binary

    @binary.setter
    def binary(self, value: Optional[str]) -> None:
        if value is None:
            self._property_binary = None
            return
        self.assert_isinstance(value, "binary", six.string_types)
        self._property_binary = value

    @schema_property("repository")
    def repository(self) -> Optional[str]:
        return self._property_repository

    @repository.setter
    def repository(self, value: Optional[str]) -> None:
        if value is None:
            self._property_repository = None
            return
        self.assert_isinstance(value, "repository", six.string_types)
        self._property_repository = value

    @schema_property("tag")
    def tag(self) -> Optional[str]:
        return self._property_tag

    @tag.setter
    def tag(self, value: Optional[str]) -> None:
        if value is None:
            self._property_tag = None
            return
        self.assert_isinstance(value, "tag", six.string_types)
        self._property_tag = value

    @schema_property("branch")
    def branch(self) -> Optional[str]:
        return self._property_branch

    @branch.setter
    def branch(self, value: Optional[str]) -> None:
        if value is None:
            self._property_branch = None
            return
        self.assert_isinstance(value, "branch", six.string_types)
        self._property_branch = value

    @schema_property("version_num")
    def version_num(self) -> Optional[str]:
        return self._property_version_num

    @version_num.setter
    def version_num(self, value: Optional[str]) -> None:
        if value is None:
            self._property_version_num = None
            return
        self.assert_isinstance(value, "version_num", six.string_types)
        self._property_version_num = value

    @schema_property("entry_point")
    def entry_point(self) -> Optional[str]:
        return self._property_entry_point

    @entry_point.setter
    def entry_point(self, value: Optional[str]) -> None:
        if value is None:
            self._property_entry_point = None
            return
        self.assert_isinstance(value, "entry_point", six.string_types)
        self._property_entry_point = value

    @schema_property("working_dir")
    def working_dir(self) -> Optional[str]:
        return self._property_working_dir

    @working_dir.setter
    def working_dir(self, value: Optional[str]) -> None:
        if value is None:
            self._property_working_dir = None
            return
        self.assert_isinstance(value, "working_dir", six.string_types)
        self._property_working_dir = value

    @schema_property("requirements")
    def requirements(self) -> Optional[dict]:
        return self._property_requirements

    @requirements.setter
    def requirements(self, value: Optional[dict]) -> None:
        if value is None:
            self._property_requirements = None
            return
        self.assert_isinstance(value, "requirements", (dict,))
        self._property_requirements = value

    @schema_property("diff")
    def diff(self) -> Optional[str]:
        return self._property_diff

    @diff.setter
    def diff(self, value: Optional[str]) -> None:
        if value is None:
            self._property_diff = None
            return
        self.assert_isinstance(value, "diff", six.string_types)
        self._property_diff = value


class Output(NonStrictDataModel):
    """
    :param destination: Storage id. This is where output files will be stored.
    :type destination: str
    :param model: Model id.
    :type model: str
    :param result: Task result. Values: 'success', 'failure'
    :type result: str
    :param error: Last error text
    :type error: str
    """

    _schema = {
        "properties": {
            "destination": {
                "description": "Storage id. This is where output files will be stored.",
                "type": ["string", "null"],
            },
            "error": {"description": "Last error text", "type": ["string", "null"]},
            "model": {"description": "Model id.", "type": ["string", "null"]},
            "result": {
                "description": "Task result. Values: 'success', 'failure'",
                "type": ["string", "null"],
            },
        },
        "type": "object",
    }

    def __init__(
        self,
        destination: Optional[str] = None,
        model: Optional[str] = None,
        result: Optional[str] = None,
        error: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super(Output, self).__init__(**kwargs)
        self.destination = destination
        self.model = model
        self.result = result
        self.error = error

    @schema_property("destination")
    def destination(self) -> Optional[str]:
        return self._property_destination

    @destination.setter
    def destination(self, value: Optional[str]) -> None:
        if value is None:
            self._property_destination = None
            return
        self.assert_isinstance(value, "destination", six.string_types)
        self._property_destination = value

    @schema_property("model")
    def model(self) -> Optional[str]:
        return self._property_model

    @model.setter
    def model(self, value: Optional[str]) -> None:
        if value is None:
            self._property_model = None
            return
        self.assert_isinstance(value, "model", six.string_types)
        self._property_model = value

    @schema_property("result")
    def result(self) -> Optional[str]:
        return self._property_result

    @result.setter
    def result(self, value: Optional[str]) -> None:
        if value is None:
            self._property_result = None
            return
        self.assert_isinstance(value, "result", six.string_types)
        self._property_result = value

    @schema_property("error")
    def error(self) -> Optional[str]:
        return self._property_error

    @error.setter
    def error(self, value: Optional[str]) -> None:
        if value is None:
            self._property_error = None
            return
        self.assert_isinstance(value, "error", six.string_types)
        self._property_error = value


class ArtifactTypeData(NonStrictDataModel):
    """
    :param preview: Description or textual data
    :type preview: str
    :param content_type: System defined raw data content type
    :type content_type: str
    :param data_hash: Hash of raw data, without any headers or descriptive parts
    :type data_hash: str
    """

    _schema = {
        "properties": {
            "content_type": {
                "description": "System defined raw data content type",
                "type": ["string", "null"],
            },
            "data_hash": {
                "description": "Hash of raw data, without any headers or descriptive parts",
                "type": ["string", "null"],
            },
            "preview": {
                "description": "Description or textual data",
                "type": ["string", "null"],
            },
        },
        "type": "object",
    }

    def __init__(
        self,
        preview: Optional[str] = None,
        content_type: Optional[str] = None,
        data_hash: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super(ArtifactTypeData, self).__init__(**kwargs)
        self.preview = preview
        self.content_type = content_type
        self.data_hash = data_hash

    @schema_property("preview")
    def preview(self) -> Optional[str]:
        return self._property_preview

    @preview.setter
    def preview(self, value: Optional[str]) -> None:
        if value is None:
            self._property_preview = None
            return
        self.assert_isinstance(value, "preview", six.string_types)
        self._property_preview = value

    @schema_property("content_type")
    def content_type(self) -> Optional[str]:
        return self._property_content_type

    @content_type.setter
    def content_type(self, value: Optional[str]) -> None:
        if value is None:
            self._property_content_type = None
            return
        self.assert_isinstance(value, "content_type", six.string_types)
        self._property_content_type = value

    @schema_property("data_hash")
    def data_hash(self) -> Optional[str]:
        return self._property_data_hash

    @data_hash.setter
    def data_hash(self, value: Optional[str]) -> None:
        if value is None:
            self._property_data_hash = None
            return
        self.assert_isinstance(value, "data_hash", six.string_types)
        self._property_data_hash = value


class ArtifactModeEnum(StringEnum):
    input = "input"
    output = "output"


class Artifact(NonStrictDataModel):
    """
    :param key: Entry key
    :type key: str
    :param type: System defined type
    :type type: str
    :param mode: System defined input/output indication
    :type mode: ArtifactModeEnum
    :param uri: Raw data location
    :type uri: str
    :param content_size: Raw data length in bytes
    :type content_size: int
    :param hash: Hash of entire raw data
    :type hash: str
    :param timestamp: Epoch time when artifact was created
    :type timestamp: int
    :param type_data: Additional fields defined by the system
    :type type_data: ArtifactTypeData
    :param display_data: User-defined list of key/value pairs, sorted
    :type display_data: Sequence[Sequence[str]]
    """

    _schema = {
        "properties": {
            "content_size": {
                "description": "Raw data length in bytes",
                "type": "integer",
            },
            "display_data": {
                "description": "User-defined list of key/value pairs, sorted",
                "items": {"items": {"type": "string"}, "type": "array"},
                "type": "array",
            },
            "hash": {"description": "Hash of entire raw data", "type": "string"},
            "key": {"description": "Entry key", "type": "string"},
            "mode": {
                "$ref": "#/definitions/artifact_mode_enum",
                "description": "System defined input/output indication",
            },
            "timestamp": {
                "description": "Epoch time when artifact was created",
                "type": "integer",
            },
            "type": {"description": "System defined type", "type": "string"},
            "type_data": {
                "$ref": "#/definitions/artifact_type_data",
                "description": "Additional fields defined by the system",
            },
            "uri": {"description": "Raw data location", "type": "string"},
        },
        "required": ["key", "type"],
        "type": "object",
    }

    def __init__(
        self,
        key: str,
        type: str,
        mode: Any = None,
        uri: Optional[str] = None,
        content_size: Optional[int] = None,
        hash: Optional[str] = None,
        timestamp: Optional[int] = None,
        type_data: Any = None,
        display_data: Optional[List[List[str]]] = None,
        **kwargs: Any
    ) -> None:
        super(Artifact, self).__init__(**kwargs)
        self.key = key
        self.type = type
        self.mode = mode
        self.uri = uri
        self.content_size = content_size
        self.hash = hash
        self.timestamp = timestamp
        self.type_data = type_data
        self.display_data = display_data

    @schema_property("key")
    def key(self) -> str:
        return self._property_key

    @key.setter
    def key(self, value: str) -> None:
        if value is None:
            self._property_key = None
            return
        self.assert_isinstance(value, "key", six.string_types)
        self._property_key = value

    @schema_property("type")
    def type(self) -> str:
        return self._property_type

    @type.setter
    def type(self, value: str) -> None:
        if value is None:
            self._property_type = None
            return
        self.assert_isinstance(value, "type", six.string_types)
        self._property_type = value

    @schema_property("mode")
    def mode(self) -> Any:
        return self._property_mode

    @mode.setter
    def mode(self, value: Any) -> None:
        if value is None:
            self._property_mode = None
            return
        if isinstance(value, six.string_types):
            try:
                value = ArtifactModeEnum(value)
            except ValueError:
                pass
        else:
            self.assert_isinstance(value, "mode", enum.Enum)
        self._property_mode = value

    @schema_property("uri")
    def uri(self) -> Optional[str]:
        return self._property_uri

    @uri.setter
    def uri(self, value: Optional[str]) -> None:
        if value is None:
            self._property_uri = None
            return
        self.assert_isinstance(value, "uri", six.string_types)
        self._property_uri = value

    @schema_property("content_size")
    def content_size(self) -> Optional[int]:
        return self._property_content_size

    @content_size.setter
    def content_size(self, value: Optional[int]) -> None:
        if value is None:
            self._property_content_size = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "content_size", six.integer_types)
        self._property_content_size = value

    @schema_property("hash")
    def hash(self) -> Optional[str]:
        return self._property_hash

    @hash.setter
    def hash(self, value: Optional[str]) -> None:
        if value is None:
            self._property_hash = None
            return
        self.assert_isinstance(value, "hash", six.string_types)
        self._property_hash = value

    @schema_property("timestamp")
    def timestamp(self) -> Optional[int]:
        return self._property_timestamp

    @timestamp.setter
    def timestamp(self, value: Optional[int]) -> None:
        if value is None:
            self._property_timestamp = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "timestamp", six.integer_types)
        self._property_timestamp = value

    @schema_property("type_data")
    def type_data(self) -> Any:
        return self._property_type_data

    @type_data.setter
    def type_data(self, value: Any) -> None:
        if value is None:
            self._property_type_data = None
            return
        if isinstance(value, dict):
            value = ArtifactTypeData.from_dict(value)
        else:
            self.assert_isinstance(value, "type_data", ArtifactTypeData)
        self._property_type_data = value

    @schema_property("display_data")
    def display_data(self) -> Optional[List[List[str]]]:
        return self._property_display_data

    @display_data.setter
    def display_data(self, value: Optional[List[List[str]]]) -> None:
        if value is None:
            self._property_display_data = None
            return
        self.assert_isinstance(value, "display_data", (list, tuple))
        self.assert_isinstance(value, "display_data", (list, tuple), is_array=True)
        self._property_display_data = value


class ArtifactId(NonStrictDataModel):
    """
    :param key: Entry key
    :type key: str
    :param mode: System defined input/output indication
    :type mode: ArtifactModeEnum
    """

    _schema = {
        "properties": {
            "key": {"description": "Entry key", "type": "string"},
            "mode": {
                "$ref": "#/definitions/artifact_mode_enum",
                "description": "System defined input/output indication",
            },
        },
        "required": ["key"],
        "type": "object",
    }

    def __init__(self, key: str, mode: Any = None, **kwargs: Any) -> None:
        super(ArtifactId, self).__init__(**kwargs)
        self.key = key
        self.mode = mode

    @schema_property("key")
    def key(self) -> str:
        return self._property_key

    @key.setter
    def key(self, value: str) -> None:
        if value is None:
            self._property_key = None
            return
        self.assert_isinstance(value, "key", six.string_types)
        self._property_key = value

    @schema_property("mode")
    def mode(self) -> Any:
        return self._property_mode

    @mode.setter
    def mode(self, value: Any) -> None:
        if value is None:
            self._property_mode = None
            return
        if isinstance(value, six.string_types):
            try:
                value = ArtifactModeEnum(value)
            except ValueError:
                pass
        else:
            self.assert_isinstance(value, "mode", enum.Enum)
        self._property_mode = value


class TaskModels(NonStrictDataModel):
    """
    :param input: The list of task input models
    :type input: Sequence[TaskModelItem]
    :param output: The list of task output models
    :type output: Sequence[TaskModelItem]
    """

    _schema = {
        "properties": {
            "input": {
                "description": "The list of task input models",
                "items": {"$ref": "#/definitions/task_model_item"},
                "type": ["array", "null"],
            },
            "output": {
                "description": "The list of task output models",
                "items": {"$ref": "#/definitions/task_model_item"},
                "type": ["array", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, input: Optional[List[Any]] = None, output: Optional[List[Any]] = None, **kwargs: Any) -> None:
        super(TaskModels, self).__init__(**kwargs)
        self.input = input
        self.output = output

    @schema_property("input")
    def input(self) -> Optional[List[Any]]:
        return self._property_input

    @input.setter
    def input(self, value: Optional[List[Any]]) -> None:
        if value is None:
            self._property_input = None
            return
        self.assert_isinstance(value, "input", (list, tuple))
        if any((isinstance(v, dict) for v in value)):
            value = [TaskModelItem.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "input", TaskModelItem, is_array=True)
        self._property_input = value

    @schema_property("output")
    def output(self) -> Optional[List[Any]]:
        return self._property_output

    @output.setter
    def output(self, value: Optional[List[Any]]) -> None:
        if value is None:
            self._property_output = None
            return
        self.assert_isinstance(value, "output", (list, tuple))
        if any((isinstance(v, dict) for v in value)):
            value = [TaskModelItem.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "output", TaskModelItem, is_array=True)
        self._property_output = value


class Execution(NonStrictDataModel):
    """
    :param queue: Queue ID where task was queued.
    :type queue: str
    :param parameters: Json object containing the Task parameters
    :type parameters: dict
    :param model_desc: Json object representing the Model descriptors
    :type model_desc: dict
    :param model_labels: Json object representing the ids of the labels in the
        model. The keys are the layers' names and the values are the IDs. Not
        applicable for Register (Import) tasks. Mandatory for Training tasks
    :type model_labels: dict
    :param framework: Framework related to the task. Case insensitive. Mandatory
        for Training tasks.
    :type framework: str
    :param artifacts: Task artifacts
    :type artifacts: Sequence[Artifact]
    """

    _schema = {
        "properties": {
            "artifacts": {
                "description": "Task artifacts",
                "items": {"$ref": "#/definitions/artifact"},
                "type": ["array", "null"],
            },
            "framework": {
                "description": "Framework related to the task. Case insensitive. Mandatory for Training tasks. ",
                "type": ["string", "null"],
            },
            "model_desc": {
                "additionalProperties": True,
                "description": "Json object representing the Model descriptors",
                "type": ["object", "null"],
            },
            "model_labels": {
                "additionalProperties": {"type": "integer"},
                "description": "Json object representing the ids of the labels in the model.\n                The keys are the layers' names and the values are the IDs.\n                Not applicable for Register (Import) tasks.\n                Mandatory for Training tasks",
                "type": ["object", "null"],
            },
            "parameters": {
                "additionalProperties": True,
                "description": "Json object containing the Task parameters",
                "type": ["object", "null"],
            },
            "queue": {
                "description": "Queue ID where task was queued.",
                "type": ["string", "null"],
            },
        },
        "type": "object",
    }

    def __init__(
        self,
        queue: Optional[str] = None,
        parameters: Optional[dict] = None,
        model_desc: Optional[dict] = None,
        model_labels: Optional[dict] = None,
        framework: Optional[str] = None,
        artifacts: Optional[List[Any]] = None,
        **kwargs: Any
    ) -> None:
        super(Execution, self).__init__(**kwargs)
        self.queue = queue
        self.parameters = parameters
        self.model_desc = model_desc
        self.model_labels = model_labels
        self.framework = framework
        self.artifacts = artifacts

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

    @schema_property("parameters")
    def parameters(self) -> Optional[dict]:
        return self._property_parameters

    @parameters.setter
    def parameters(self, value: Optional[dict]) -> None:
        if value is None:
            self._property_parameters = None
            return
        self.assert_isinstance(value, "parameters", (dict,))
        self._property_parameters = value

    @schema_property("model_desc")
    def model_desc(self) -> Optional[dict]:
        return self._property_model_desc

    @model_desc.setter
    def model_desc(self, value: Optional[dict]) -> None:
        if value is None:
            self._property_model_desc = None
            return
        self.assert_isinstance(value, "model_desc", (dict,))
        self._property_model_desc = value

    @schema_property("model_labels")
    def model_labels(self) -> Optional[dict]:
        return self._property_model_labels

    @model_labels.setter
    def model_labels(self, value: Optional[dict]) -> None:
        if value is None:
            self._property_model_labels = None
            return
        self.assert_isinstance(value, "model_labels", (dict,))
        self._property_model_labels = value

    @schema_property("framework")
    def framework(self) -> Optional[str]:
        return self._property_framework

    @framework.setter
    def framework(self, value: Optional[str]) -> None:
        if value is None:
            self._property_framework = None
            return
        self.assert_isinstance(value, "framework", six.string_types)
        self._property_framework = value

    @schema_property("artifacts")
    def artifacts(self) -> Optional[List[Any]]:
        return self._property_artifacts

    @artifacts.setter
    def artifacts(self, value: Optional[List[Any]]) -> None:
        if value is None:
            self._property_artifacts = None
            return
        self.assert_isinstance(value, "artifacts", (list, tuple))
        if any((isinstance(v, dict) for v in value)):
            value = [Artifact.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "artifacts", Artifact, is_array=True)
        self._property_artifacts = value


class TaskStatusEnum(StringEnum):
    created = "created"
    queued = "queued"
    in_progress = "in_progress"
    stopped = "stopped"
    published = "published"
    publishing = "publishing"
    closed = "closed"
    failed = "failed"
    completed = "completed"
    unknown = "unknown"


class TaskTypeEnum(StringEnum):
    training = "training"
    testing = "testing"
    inference = "inference"
    data_processing = "data_processing"
    application = "application"
    monitor = "monitor"
    controller = "controller"
    optimizer = "optimizer"
    service = "service"
    qc = "qc"
    custom = "custom"


class LastMetricsEvent(NonStrictDataModel):
    """
    :param metric: Metric name
    :type metric: str
    :param variant: Variant name
    :type variant: str
    :param value: Last value reported
    :type value: float
    :param min_value: Minimum value reported
    :type min_value: float
    :param max_value: Maximum value reported
    :type max_value: float
    """

    _schema = {
        "properties": {
            "max_value": {
                "description": "Maximum value reported",
                "type": ["number", "null"],
            },
            "metric": {"description": "Metric name", "type": ["string", "null"]},
            "min_value": {
                "description": "Minimum value reported",
                "type": ["number", "null"],
            },
            "value": {"description": "Last value reported", "type": ["number", "null"]},
            "variant": {"description": "Variant name", "type": ["string", "null"]},
        },
        "type": "object",
    }

    def __init__(
        self,
        metric: Optional[str] = None,
        variant: Optional[str] = None,
        value: Optional[float] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        super(LastMetricsEvent, self).__init__(**kwargs)
        self.metric = metric
        self.variant = variant
        self.value = value
        self.min_value = min_value
        self.max_value = max_value

    @schema_property("metric")
    def metric(self) -> Optional[str]:
        return self._property_metric

    @metric.setter
    def metric(self, value: Optional[str]) -> None:
        if value is None:
            self._property_metric = None
            return
        self.assert_isinstance(value, "metric", six.string_types)
        self._property_metric = value

    @schema_property("variant")
    def variant(self) -> Optional[str]:
        return self._property_variant

    @variant.setter
    def variant(self, value: Optional[str]) -> None:
        if value is None:
            self._property_variant = None
            return
        self.assert_isinstance(value, "variant", six.string_types)
        self._property_variant = value

    @schema_property("value")
    def value(self) -> Optional[float]:
        return self._property_value

    @value.setter
    def value(self, value: Optional[float]) -> None:
        if value is None:
            self._property_value = None
            return
        self.assert_isinstance(value, "value", six.integer_types + (float,))
        self._property_value = value

    @schema_property("min_value")
    def min_value(self) -> Optional[float]:
        return self._property_min_value

    @min_value.setter
    def min_value(self, value: Optional[float]) -> None:
        if value is None:
            self._property_min_value = None
            return
        self.assert_isinstance(value, "min_value", six.integer_types + (float,))
        self._property_min_value = value

    @schema_property("max_value")
    def max_value(self) -> Optional[float]:
        return self._property_max_value

    @max_value.setter
    def max_value(self, value: Optional[float]) -> None:
        if value is None:
            self._property_max_value = None
            return
        self.assert_isinstance(value, "max_value", six.integer_types + (float,))
        self._property_max_value = value


class LastMetricsVariants(NonStrictDataModel):
    """
    Last metric events, one for each variant hash

    """

    _schema = {
        "additionalProperties": {"$ref": "#/definitions/last_metrics_event"},
        "description": "Last metric events, one for each variant hash",
        "type": "object",
    }


class ParamsItem(NonStrictDataModel):
    """
    :param section: Section that the parameter belongs to
    :type section: str
    :param name: Name of the parameter. The combination of section and name should
        be unique
    :type name: str
    :param value: Value of the parameter
    :type value: str
    :param type: Type of the parameter. Optional
    :type type: str
    :param description: The parameter description. Optional
    :type description: str
    """

    _schema = {
        "properties": {
            "description": {
                "description": "The parameter description. Optional",
                "type": ["string", "null"],
            },
            "name": {
                "description": "Name of the parameter. The combination of section and name should be unique",
                "type": ["string", "null"],
            },
            "section": {
                "description": "Section that the parameter belongs to",
                "type": ["string", "null"],
            },
            "type": {
                "description": "Type of the parameter. Optional",
                "type": ["string", "null"],
            },
            "value": {
                "description": "Value of the parameter",
                "type": ["string", "null"],
            },
        },
        "type": "object",
    }

    def __init__(
        self,
        section: Optional[str] = None,
        name: Optional[str] = None,
        value: Optional[str] = None,
        type: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super(ParamsItem, self).__init__(**kwargs)
        self.section = section
        self.name = name
        self.value = value
        self.type = type
        self.description = description

    @schema_property("section")
    def section(self) -> Optional[str]:
        return self._property_section

    @section.setter
    def section(self, value: Optional[str]) -> None:
        if value is None:
            self._property_section = None
            return
        self.assert_isinstance(value, "section", six.string_types)
        self._property_section = value

    @schema_property("name")
    def name(self) -> Optional[str]:
        return self._property_name

    @name.setter
    def name(self, value: Optional[str]) -> None:
        if value is None:
            self._property_name = None
            return
        self.assert_isinstance(value, "name", six.string_types)
        self._property_name = value

    @schema_property("value")
    def value(self) -> Optional[str]:
        return self._property_value

    @value.setter
    def value(self, value: Optional[str]) -> None:
        if value is None:
            self._property_value = None
            return
        self.assert_isinstance(value, "value", six.string_types)
        self._property_value = value

    @schema_property("type")
    def type(self) -> Optional[str]:
        return self._property_type

    @type.setter
    def type(self, value: Optional[str]) -> None:
        if value is None:
            self._property_type = None
            return
        self.assert_isinstance(value, "type", six.string_types)
        self._property_type = value

    @schema_property("description")
    def description(self) -> Optional[str]:
        return self._property_description

    @description.setter
    def description(self, value: Optional[str]) -> None:
        if value is None:
            self._property_description = None
            return
        self.assert_isinstance(value, "description", six.string_types)
        self._property_description = value


class ConfigurationItem(NonStrictDataModel):
    """
    :param name: Name of the parameter. Should be unique
    :type name: str
    :param value: Value of the parameter
    :type value: str
    :param type: Type of the parameter. Optional
    :type type: str
    :param description: The parameter description. Optional
    :type description: str
    """

    _schema = {
        "properties": {
            "description": {
                "description": "The parameter description. Optional",
                "type": ["string", "null"],
            },
            "name": {
                "description": "Name of the parameter. Should be unique",
                "type": ["string", "null"],
            },
            "type": {
                "description": "Type of the parameter. Optional",
                "type": ["string", "null"],
            },
            "value": {
                "description": "Value of the parameter",
                "type": ["string", "null"],
            },
        },
        "type": "object",
    }

    def __init__(
        self,
        name: Optional[str] = None,
        value: Optional[str] = None,
        type: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super(ConfigurationItem, self).__init__(**kwargs)
        self.name = name
        self.value = value
        self.type = type
        self.description = description

    @schema_property("name")
    def name(self) -> Optional[str]:
        return self._property_name

    @name.setter
    def name(self, value: Optional[str]) -> None:
        if value is None:
            self._property_name = None
            return
        self.assert_isinstance(value, "name", six.string_types)
        self._property_name = value

    @schema_property("value")
    def value(self) -> Optional[str]:
        return self._property_value

    @value.setter
    def value(self, value: Optional[str]) -> None:
        if value is None:
            self._property_value = None
            return
        self.assert_isinstance(value, "value", six.string_types)
        self._property_value = value

    @schema_property("type")
    def type(self) -> Optional[str]:
        return self._property_type

    @type.setter
    def type(self, value: Optional[str]) -> None:
        if value is None:
            self._property_type = None
            return
        self.assert_isinstance(value, "type", six.string_types)
        self._property_type = value

    @schema_property("description")
    def description(self) -> Optional[str]:
        return self._property_description

    @description.setter
    def description(self, value: Optional[str]) -> None:
        if value is None:
            self._property_description = None
            return
        self.assert_isinstance(value, "description", six.string_types)
        self._property_description = value


class ParamKey(NonStrictDataModel):
    """
    :param section: Section that the parameter belongs to
    :type section: str
    :param name: Name of the parameter. If the name is ommitted then the
        corresponding operation is performed on the whole section
    :type name: str
    """

    _schema = {
        "properties": {
            "name": {
                "description": "Name of the parameter. If the name is ommitted then the corresponding operation is performed on the whole section",
                "type": ["string", "null"],
            },
            "section": {
                "description": "Section that the parameter belongs to",
                "type": ["string", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, section: Optional[str] = None, name: Optional[str] = None, **kwargs: Any) -> None:
        super(ParamKey, self).__init__(**kwargs)
        self.section = section
        self.name = name

    @schema_property("section")
    def section(self) -> Optional[str]:
        return self._property_section

    @section.setter
    def section(self, value: Optional[str]) -> None:
        if value is None:
            self._property_section = None
            return
        self.assert_isinstance(value, "section", six.string_types)
        self._property_section = value

    @schema_property("name")
    def name(self) -> Optional[str]:
        return self._property_name

    @name.setter
    def name(self, value: Optional[str]) -> None:
        if value is None:
            self._property_name = None
            return
        self.assert_isinstance(value, "name", six.string_types)
        self._property_name = value


class SectionParams(dict, NonStrictDataModel):
    """
    Task section params
    """

    _schema = {
        "additionalProperties": True,
        "description": "Task section params",
        "type": "object",
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.assert_isinstance(args, "section_params", dict, is_array=True)
        kwargs.update(args)
        self.assert_isinstance(kwargs.values(), "params", (ParamsItem, dict), is_array=True)
        for k, v in kwargs.items():
            if isinstance(v, dict):
                kwargs[k] = ParamsItem(**v)
        super(SectionParams, self).__init__(**kwargs)


class ReplaceHyperparamsEnum(StringEnum):
    none = "none"
    section = "section"
    all = "all"


class Task(NonStrictDataModel):
    """
    :param id: Task id
    :type id: str
    :param name: Task Name
    :type name: str
    :param user: Associated user id
    :type user: str
    :param company: Company ID
    :type company: str
    :param type: Type of task. Values: 'training', 'testing'
    :type type: TaskTypeEnum
    :param status:
    :type status: TaskStatusEnum
    :param comment: Free text comment
    :type comment: str
    :param created: Task creation time (UTC)
    :type created: datetime.datetime
    :param started: Task start time (UTC)
    :type started: datetime.datetime
    :param completed: Task end time (UTC)
    :type completed: datetime.datetime
    :param active_duration: Task duration time (seconds)
    :type active_duration: int
    :param parent: Parent task id
    :type parent: str
    :param project: Project ID of the project to which this task is assigned
    :type project: str
    :param output: Task output params
    :type output: Output
    :param execution: Task execution params
    :type execution: Execution
    :param container: Docker container parameters
    :type container: dict
    :param models: Task models
    :type models: TaskModels
    :param script: Script info
    :type script: Script
    :param tags: User-defined tags list
    :type tags: Sequence[str]
    :param system_tags: System tags list. This field is reserved for system use,
        please don't use it.
    :type system_tags: Sequence[str]
    :param status_changed: Last status change time
    :type status_changed: datetime.datetime
    :param status_message: free text string representing info about the status
    :type status_message: str
    :param status_reason: Reason for last status change
    :type status_reason: str
    :param published: Last status change time
    :type published: datetime.datetime
    :param last_worker: ID of last worker that handled the task
    :type last_worker: str
    :param last_worker_report: Last time a worker reported while working on this
        task
    :type last_worker_report: datetime.datetime
    :param last_update: Last time this task was created, edited, changed or events
        for this task were reported
    :type last_update: datetime.datetime
    :param last_change: Last time any update was done to the task
    :type last_change: datetime.datetime
    :param last_iteration: Last iteration reported for this task
    :type last_iteration: int
    :param last_metrics: Last metric variants (hash to events), one for each metric
        hash
    :type last_metrics: dict
    :param hyperparams: Task hyper params per section
    :type hyperparams: dict
    :param configuration: Task configuration params
    :type configuration: dict
    :param runtime: Task runtime mapping
    :type runtime: dict
    """

    _schema = {
        "properties": {
            "active_duration": {
                "description": "Task duration time (seconds)",
                "type": ["integer", "null"],
            },
            "comment": {"description": "Free text comment", "type": ["string", "null"]},
            "company": {"description": "Company ID", "type": ["string", "null"]},
            "completed": {
                "description": "Task end time (UTC)",
                "format": "date-time",
                "type": ["string", "null"],
            },
            "configuration": {
                "additionalProperties": {"$ref": "#/definitions/configuration_item"},
                "description": "Task configuration params",
                "type": ["object", "null"],
            },
            "container": {
                "type": "object",
                "description": "Docker container parameters",
                "additionalProperties": {"type": ["string", "null"]},
            },
            "created": {
                "description": "Task creation time (UTC) ",
                "format": "date-time",
                "type": ["string", "null"],
            },
            "execution": {
                "description": "Task execution params",
                "oneOf": [{"$ref": "#/definitions/execution"}, {"type": "null"}],
            },
            "hyperparams": {
                "additionalProperties": {"$ref": "#/definitions/section_params"},
                "description": "Task hyper params per section",
                "type": ["object", "null"],
            },
            "id": {"description": "Task id", "type": ["string", "null"]},
            "last_change": {
                "description": "Last time any update was done to the task",
                "format": "date-time",
                "type": ["string", "null"],
            },
            "last_iteration": {
                "description": "Last iteration reported for this task",
                "type": ["integer", "null"],
            },
            "last_metrics": {
                "additionalProperties": {"$ref": "#/definitions/last_metrics_variants"},
                "description": "Last metric variants (hash to events), one for each metric hash",
                "type": ["object", "null"],
            },
            "last_update": {
                "description": "Last time this task was created, edited, changed or events for this task were reported",
                "format": "date-time",
                "type": ["string", "null"],
            },
            "last_worker": {
                "description": "ID of last worker that handled the task",
                "type": ["string", "null"],
            },
            "last_worker_report": {
                "description": "Last time a worker reported while working on this task",
                "format": "date-time",
                "type": ["string", "null"],
            },
            "models": {
                "description": "Task models",
                "oneOf": [{"$ref": "#/definitions/task_models"}, {"type": "null"}],
            },
            "name": {"description": "Task Name", "type": ["string", "null"]},
            "output": {
                "description": "Task output params",
                "oneOf": [{"$ref": "#/definitions/output"}, {"type": "null"}],
            },
            "parent": {"description": "Parent task id", "type": ["string", "null"]},
            "project": {
                "description": "Project ID of the project to which this task is assigned",
                "type": ["string", "null"],
            },
            "published": {
                "description": "Last status change time",
                "format": "date-time",
                "type": ["string", "null"],
            },
            "runtime": {
                "description": "Task runtime mapping",
                "type": ["object", "null"],
                "additionalProperties": True,
            },
            "script": {
                "description": "Script info",
                "oneOf": [{"$ref": "#/definitions/script"}, {"type": "null"}],
            },
            "started": {
                "description": "Task start time (UTC)",
                "format": "date-time",
                "type": ["string", "null"],
            },
            "status": {
                "description": "",
                "oneOf": [{"$ref": "#/definitions/task_status_enum"}, {"type": "null"}],
            },
            "status_changed": {
                "description": "Last status change time",
                "format": "date-time",
                "type": ["string", "null"],
            },
            "status_message": {
                "description": "free text string representing info about the status",
                "type": ["string", "null"],
            },
            "status_reason": {
                "description": "Reason for last status change",
                "type": ["string", "null"],
            },
            "system_tags": {
                "description": "System tags list. This field is reserved for system use, please don't use it.",
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
            "tags": {
                "description": "User-defined tags list",
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
            "type": {
                "description": "Type of task. Values: 'training', 'testing'",
                "oneOf": [{"$ref": "#/definitions/task_type_enum"}, {"type": "null"}],
            },
            "user": {"description": "Associated user id", "type": ["string", "null"]},
        },
        "type": "object",
    }

    def __init__(
        self,
        id: Optional[str] = None,
        name: Optional[str] = None,
        user: Optional[str] = None,
        company: Optional[str] = None,
        type: Any = None,
        status: Any = None,
        comment: Optional[str] = None,
        created: Optional[str] = None,
        started: Optional[str] = None,
        completed: Optional[str] = None,
        active_duration: Optional[int] = None,
        parent: Optional[str] = None,
        project: Optional[str] = None,
        output: Any = None,
        execution: Any = None,
        container: Optional[dict] = None,
        models: Any = None,
        script: Any = None,
        tags: Optional[List[str]] = None,
        system_tags: Optional[List[str]] = None,
        status_changed: Optional[str] = None,
        status_message: Optional[str] = None,
        status_reason: Optional[str] = None,
        published: Optional[str] = None,
        last_worker: Optional[str] = None,
        last_worker_report: Optional[str] = None,
        last_update: Optional[str] = None,
        last_change: Optional[str] = None,
        last_iteration: Optional[int] = None,
        last_metrics: Optional[dict] = None,
        hyperparams: Optional[dict] = None,
        configuration: Optional[dict] = None,
        runtime: Optional[dict] = None,
        **kwargs: Any
    ) -> None:
        super(Task, self).__init__(**kwargs)
        self.id = id
        self.name = name
        self.user = user
        self.company = company
        self.type = type
        self.status = status
        self.comment = comment
        self.created = created
        self.started = started
        self.completed = completed
        self.active_duration = active_duration
        self.parent = parent
        self.project = project
        self.output = output
        self.execution = execution
        self.container = container
        self.models = models
        self.script = script
        self.tags = tags
        self.system_tags = system_tags
        self.status_changed = status_changed
        self.status_message = status_message
        self.status_reason = status_reason
        self.published = published
        self.last_worker = last_worker
        self.last_worker_report = last_worker_report
        self.last_update = last_update
        self.last_change = last_change
        self.last_iteration = last_iteration
        self.last_metrics = last_metrics
        self.hyperparams = hyperparams
        self.configuration = configuration
        self.runtime = runtime

    @schema_property("id")
    def id(self) -> Optional[str]:
        return self._property_id

    @id.setter
    def id(self, value: Optional[str]) -> None:
        if value is None:
            self._property_id = None
            return
        self.assert_isinstance(value, "id", six.string_types)
        self._property_id = value

    @schema_property("name")
    def name(self) -> Optional[str]:
        return self._property_name

    @name.setter
    def name(self, value: Optional[str]) -> None:
        if value is None:
            self._property_name = None
            return
        self.assert_isinstance(value, "name", six.string_types)
        self._property_name = value

    @schema_property("user")
    def user(self) -> Optional[str]:
        return self._property_user

    @user.setter
    def user(self, value: Optional[str]) -> None:
        if value is None:
            self._property_user = None
            return
        self.assert_isinstance(value, "user", six.string_types)
        self._property_user = value

    @schema_property("company")
    def company(self) -> Optional[str]:
        return self._property_company

    @company.setter
    def company(self, value: Optional[str]) -> None:
        if value is None:
            self._property_company = None
            return
        self.assert_isinstance(value, "company", six.string_types)
        self._property_company = value

    @schema_property("type")
    def type(self) -> Any:
        return self._property_type

    @type.setter
    def type(self, value: Any) -> None:
        if value is None:
            self._property_type = None
            return
        if isinstance(value, six.string_types):
            try:
                value = TaskTypeEnum(value)
            except ValueError:
                pass
        else:
            self.assert_isinstance(value, "type", enum.Enum)
        self._property_type = value

    @schema_property("status")
    def status(self) -> Any:
        return self._property_status

    @status.setter
    def status(self, value: Any) -> None:
        if value is None:
            self._property_status = None
            return
        if isinstance(value, six.string_types):
            try:
                value = TaskStatusEnum(value)
            except ValueError:
                pass
        else:
            self.assert_isinstance(value, "status", enum.Enum)
        self._property_status = value

    @schema_property("comment")
    def comment(self) -> Optional[str]:
        return self._property_comment

    @comment.setter
    def comment(self, value: Optional[str]) -> None:
        if value is None:
            self._property_comment = None
            return
        self.assert_isinstance(value, "comment", six.string_types)
        self._property_comment = value

    @schema_property("created")
    def created(self) -> Optional[str]:
        return self._property_created

    @created.setter
    def created(self, value: Optional[str]) -> None:
        if value is None:
            self._property_created = None
            return
        self.assert_isinstance(value, "created", six.string_types + (datetime,))
        if not isinstance(value, datetime):
            value = parse_datetime(value)
        self._property_created = value

    @schema_property("started")
    def started(self) -> Optional[str]:
        return self._property_started

    @started.setter
    def started(self, value: Optional[str]) -> None:
        if value is None:
            self._property_started = None
            return
        self.assert_isinstance(value, "started", six.string_types + (datetime,))
        if not isinstance(value, datetime):
            value = parse_datetime(value)
        self._property_started = value

    @schema_property("completed")
    def completed(self) -> Optional[str]:
        return self._property_completed

    @completed.setter
    def completed(self, value: Optional[str]) -> None:
        if value is None:
            self._property_completed = None
            return
        self.assert_isinstance(value, "completed", six.string_types + (datetime,))
        if not isinstance(value, datetime):
            value = parse_datetime(value)
        self._property_completed = value

    @schema_property("active_duration")
    def active_duration(self) -> Optional[int]:
        return self._property_active_duration

    @active_duration.setter
    def active_duration(self, value: Optional[int]) -> None:
        if value is None:
            self._property_active_duration = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "active_duration", six.integer_types)
        self._property_active_duration = value

    @schema_property("parent")
    def parent(self) -> Optional[str]:
        return self._property_parent

    @parent.setter
    def parent(self, value: Optional[str]) -> None:
        if value is None:
            self._property_parent = None
            return
        self.assert_isinstance(value, "parent", six.string_types)
        self._property_parent = value

    @schema_property("project")
    def project(self) -> Optional[str]:
        return self._property_project

    @project.setter
    def project(self, value: Optional[str]) -> None:
        if value is None:
            self._property_project = None
            return
        self.assert_isinstance(value, "project", six.string_types)
        self._property_project = value

    @schema_property("output")
    def output(self) -> Any:
        return self._property_output

    @output.setter
    def output(self, value: Any) -> None:
        if value is None:
            self._property_output = None
            return
        if isinstance(value, dict):
            value = Output.from_dict(value)
        else:
            self.assert_isinstance(value, "output", Output)
        self._property_output = value

    @schema_property("execution")
    def execution(self) -> Any:
        return self._property_execution

    @execution.setter
    def execution(self, value: Any) -> None:
        if value is None:
            self._property_execution = None
            return
        if isinstance(value, dict):
            value = Execution.from_dict(value)
        else:
            self.assert_isinstance(value, "execution", Execution)
        self._property_execution = value

    @schema_property("container")
    def container(self) -> Optional[dict]:
        return self._property_container

    @container.setter
    def container(self, value: Optional[dict]) -> None:
        if value is None:
            self._property_container = None
            return
        self.assert_isinstance(value, "container", dict)
        self._property_container = value

    @schema_property("models")
    def models(self) -> Any:
        return self._property_models

    @models.setter
    def models(self, value: Any) -> None:
        if value is None:
            self._property_models = None
            return
        if isinstance(value, dict):
            value = TaskModels.from_dict(value)
        else:
            self.assert_isinstance(value, "models", TaskModels)
        self._property_models = value

    @schema_property("script")
    def script(self) -> Any:
        return self._property_script

    @script.setter
    def script(self, value: Any) -> None:
        if value is None:
            self._property_script = None
            return
        if isinstance(value, dict):
            value = Script.from_dict(value)
        else:
            self.assert_isinstance(value, "script", Script)
        self._property_script = value

    @schema_property("tags")
    def tags(self) -> Optional[List[str]]:
        return self._property_tags

    @tags.setter
    def tags(self, value: Optional[List[str]]) -> None:
        if value is None:
            self._property_tags = None
            return
        self.assert_isinstance(value, "tags", (list, tuple))
        self.assert_isinstance(value, "tags", six.string_types, is_array=True)
        self._property_tags = value

    @schema_property("system_tags")
    def system_tags(self) -> Optional[List[str]]:
        return self._property_system_tags

    @system_tags.setter
    def system_tags(self, value: Optional[List[str]]) -> None:
        if value is None:
            self._property_system_tags = None
            return
        self.assert_isinstance(value, "system_tags", (list, tuple))
        self.assert_isinstance(value, "system_tags", six.string_types, is_array=True)
        self._property_system_tags = value

    @schema_property("status_changed")
    def status_changed(self) -> Optional[str]:
        return self._property_status_changed

    @status_changed.setter
    def status_changed(self, value: Optional[str]) -> None:
        if value is None:
            self._property_status_changed = None
            return
        self.assert_isinstance(value, "status_changed", six.string_types + (datetime,))
        if not isinstance(value, datetime):
            value = parse_datetime(value)
        self._property_status_changed = value

    @schema_property("status_message")
    def status_message(self) -> Optional[str]:
        return self._property_status_message

    @status_message.setter
    def status_message(self, value: Optional[str]) -> None:
        if value is None:
            self._property_status_message = None
            return
        self.assert_isinstance(value, "status_message", six.string_types)
        self._property_status_message = value

    @schema_property("status_reason")
    def status_reason(self) -> Optional[str]:
        return self._property_status_reason

    @status_reason.setter
    def status_reason(self, value: Optional[str]) -> None:
        if value is None:
            self._property_status_reason = None
            return
        self.assert_isinstance(value, "status_reason", six.string_types)
        self._property_status_reason = value

    @schema_property("published")
    def published(self) -> Optional[str]:
        return self._property_published

    @published.setter
    def published(self, value: Optional[str]) -> None:
        if value is None:
            self._property_published = None
            return
        self.assert_isinstance(value, "published", six.string_types + (datetime,))
        if not isinstance(value, datetime):
            value = parse_datetime(value)
        self._property_published = value

    @schema_property("last_worker")
    def last_worker(self) -> Optional[str]:
        return self._property_last_worker

    @last_worker.setter
    def last_worker(self, value: Optional[str]) -> None:
        if value is None:
            self._property_last_worker = None
            return
        self.assert_isinstance(value, "last_worker", six.string_types)
        self._property_last_worker = value

    @schema_property("last_worker_report")
    def last_worker_report(self) -> Optional[str]:
        return self._property_last_worker_report

    @last_worker_report.setter
    def last_worker_report(self, value: Optional[str]) -> None:
        if value is None:
            self._property_last_worker_report = None
            return
        self.assert_isinstance(value, "last_worker_report", six.string_types + (datetime,))
        if not isinstance(value, datetime):
            value = parse_datetime(value)
        self._property_last_worker_report = value

    @schema_property("last_update")
    def last_update(self) -> Optional[str]:
        return self._property_last_update

    @last_update.setter
    def last_update(self, value: Optional[str]) -> None:
        if value is None:
            self._property_last_update = None
            return
        self.assert_isinstance(value, "last_update", six.string_types + (datetime,))
        if not isinstance(value, datetime):
            value = parse_datetime(value)
        self._property_last_update = value

    @schema_property("last_change")
    def last_change(self) -> Optional[str]:
        return self._property_last_change

    @last_change.setter
    def last_change(self, value: Optional[str]) -> None:
        if value is None:
            self._property_last_change = None
            return
        self.assert_isinstance(value, "last_change", six.string_types + (datetime,))
        if not isinstance(value, datetime):
            value = parse_datetime(value)
        self._property_last_change = value

    @schema_property("last_iteration")
    def last_iteration(self) -> Optional[int]:
        return self._property_last_iteration

    @last_iteration.setter
    def last_iteration(self, value: Optional[int]) -> None:
        if value is None:
            self._property_last_iteration = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "last_iteration", six.integer_types)
        self._property_last_iteration = value

    @schema_property("last_metrics")
    def last_metrics(self) -> Optional[dict]:
        return self._property_last_metrics

    @last_metrics.setter
    def last_metrics(self, value: Optional[dict]) -> None:
        if value is None:
            self._property_last_metrics = None
            return
        self.assert_isinstance(value, "last_metrics", (dict,))
        self._property_last_metrics = value

    @schema_property("hyperparams")
    def hyperparams(self) -> Optional[dict]:
        return self._property_hyperparams

    @hyperparams.setter
    def hyperparams(self, value: Optional[dict]) -> None:
        if value is None:
            self._property_hyperparams = None
            return
        self.assert_isinstance(value, "hyperparams", dict)
        self.assert_isinstance(value.keys(), "hyperparams_keys", six.string_types, is_array=True)
        self.assert_isinstance(value.values(), "hyperparams_values", (SectionParams, dict), is_array=True)
        value = dict(((k, SectionParams(**v) if isinstance(v, dict) else v) for (k, v) in value.items()))
        self._property_hyperparams = value

    @schema_property("configuration")
    def configuration(self) -> Optional[dict]:
        return self._property_configuration

    @configuration.setter
    def configuration(self, value: Optional[dict]) -> None:
        if value is None:
            self._property_configuration = None
            return
        self.assert_isinstance(value, "configuration", dict)
        self.assert_isinstance(value.keys(), "configuration_keys", six.string_types, is_array=True)
        self.assert_isinstance(
            value.values(),
            "configuration_values",
            (ConfigurationItem, dict),
            is_array=True,
        )
        value = dict(((k, ConfigurationItem(**v) if isinstance(v, dict) else v) for (k, v) in value.items()))
        self._property_configuration = value

    @schema_property("runtime")
    def runtime(self) -> Optional[dict]:
        return self._property_runtime

    @runtime.setter
    def runtime(self, value: Optional[dict]) -> None:
        if value is None:
            self._property_runtime = None
            return
        self.assert_isinstance(value, "runtime", dict)
        self._property_runtime = value


class TaskUrls(NonStrictDataModel):
    """
    :param model_urls:
    :type model_urls: Sequence[str]
    :param event_urls:
    :type event_urls: Sequence[str]
    :param artifact_urls:
    :type artifact_urls: Sequence[str]
    """

    _schema = {
        "properties": {
            "artifact_urls": {"items": {"type": "string"}, "type": ["array", "null"]},
            "event_urls": {"items": {"type": "string"}, "type": ["array", "null"]},
            "model_urls": {"items": {"type": "string"}, "type": ["array", "null"]},
        },
        "type": "object",
    }

    def __init__(
        self,
        model_urls: Optional[List[str]] = None,
        event_urls: Optional[List[str]] = None,
        artifact_urls: Optional[List[str]] = None,
        **kwargs: Any
    ) -> None:
        super(TaskUrls, self).__init__(**kwargs)
        self.model_urls = model_urls
        self.event_urls = event_urls
        self.artifact_urls = artifact_urls

    @schema_property("model_urls")
    def model_urls(self) -> Optional[List[str]]:
        return self._property_model_urls

    @model_urls.setter
    def model_urls(self, value: Optional[List[str]]) -> None:
        if value is None:
            self._property_model_urls = None
            return
        self.assert_isinstance(value, "model_urls", (list, tuple))
        self.assert_isinstance(value, "model_urls", six.string_types, is_array=True)
        self._property_model_urls = value

    @schema_property("event_urls")
    def event_urls(self) -> Optional[List[str]]:
        return self._property_event_urls

    @event_urls.setter
    def event_urls(self, value: Optional[List[str]]) -> None:
        if value is None:
            self._property_event_urls = None
            return
        self.assert_isinstance(value, "event_urls", (list, tuple))
        self.assert_isinstance(value, "event_urls", six.string_types, is_array=True)
        self._property_event_urls = value

    @schema_property("artifact_urls")
    def artifact_urls(self) -> Optional[List[str]]:
        return self._property_artifact_urls

    @artifact_urls.setter
    def artifact_urls(self, value: Optional[List[str]]) -> None:
        if value is None:
            self._property_artifact_urls = None
            return
        self.assert_isinstance(value, "artifact_urls", (list, tuple))
        self.assert_isinstance(value, "artifact_urls", six.string_types, is_array=True)
        self._property_artifact_urls = value


class AddOrUpdateArtifactsRequest(Request):
    """
    Update existing artifacts (search by key/mode) and add new ones

    :param task: Task ID
    :type task: str
    :param artifacts: Artifacts to add or update
    :type artifacts: Sequence[Artifact]
    :param force: If set to True then both new and running task artifacts can be
        edited. Otherwise only the new task ones. Default is False
    :type force: bool
    """

    _service = "tasks"
    _action = "add_or_update_artifacts"
    _version = "2.13"
    _schema = {
        "definitions": {
            "artifact": {
                "properties": {
                    "content_size": {
                        "description": "Raw data length in bytes",
                        "type": "integer",
                    },
                    "display_data": {
                        "description": "User-defined list of key/value pairs, sorted",
                        "items": {"items": {"type": "string"}, "type": "array"},
                        "type": "array",
                    },
                    "hash": {
                        "description": "Hash of entire raw data",
                        "type": "string",
                    },
                    "key": {"description": "Entry key", "type": "string"},
                    "mode": {
                        "$ref": "#/definitions/artifact_mode_enum",
                        "description": "System defined input/output indication",
                    },
                    "timestamp": {
                        "description": "Epoch time when artifact was created",
                        "type": "integer",
                    },
                    "type": {"description": "System defined type", "type": "string"},
                    "type_data": {
                        "$ref": "#/definitions/artifact_type_data",
                        "description": "Additional fields defined by the system",
                    },
                    "uri": {"description": "Raw data location", "type": "string"},
                },
                "required": ["key", "type"],
                "type": "object",
            },
            "artifact_mode_enum": {
                "default": "output",
                "enum": ["input", "output"],
                "type": "string",
            },
            "artifact_type_data": {
                "properties": {
                    "content_type": {
                        "description": "System defined raw data content type",
                        "type": ["string", "null"],
                    },
                    "data_hash": {
                        "description": "Hash of raw data, without any headers or descriptive parts",
                        "type": ["string", "null"],
                    },
                    "preview": {
                        "description": "Description or textual data",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
        },
        "properties": {
            "artifacts": {
                "description": "Artifacts to add or update",
                "items": {"$ref": "#/definitions/artifact"},
                "type": "array",
            },
            "force": {
                "description": "If set to True then both new and running task artifacts can be edited. Otherwise only the new task ones. Default is False",
                "type": "boolean",
            },
            "task": {"description": "Task ID", "type": "string"},
        },
        "required": ["task", "artifacts"],
        "type": "object",
    }

    def __init__(self, task: str, artifacts: List[Any], force: Optional[bool] = None, **kwargs: Any) -> None:
        super(AddOrUpdateArtifactsRequest, self).__init__(**kwargs)
        self.task = task
        self.artifacts = artifacts
        self.force = force

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

    @schema_property("artifacts")
    def artifacts(self) -> List[Any]:
        return self._property_artifacts

    @artifacts.setter
    def artifacts(self, value: List[Any]) -> None:
        if value is None:
            self._property_artifacts = None
            return
        self.assert_isinstance(value, "artifacts", (list, tuple))
        if any((isinstance(v, dict) for v in value)):
            value = [Artifact.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "artifacts", Artifact, is_array=True)
        self._property_artifacts = value

    @schema_property("force")
    def force(self) -> Optional[bool]:
        return self._property_force

    @force.setter
    def force(self, value: Optional[bool]) -> None:
        if value is None:
            self._property_force = None
            return
        self.assert_isinstance(value, "force", (bool,))
        self._property_force = value


class AddOrUpdateArtifactsResponse(Response):
    """
    Response of tasks.add_or_update_artifacts endpoint.

    :param updated: Indicates if the task was updated successfully
    :type updated: int
    """

    _service = "tasks"
    _action = "add_or_update_artifacts"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "updated": {
                "description": "Indicates if the task was updated successfully",
                "type": ["integer", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, updated: Optional[int] = None, **kwargs: Any) -> None:
        super(AddOrUpdateArtifactsResponse, self).__init__(**kwargs)
        self.updated = updated

    @schema_property("updated")
    def updated(self) -> Optional[int]:
        return self._property_updated

    @updated.setter
    def updated(self, value: Optional[int]) -> None:
        if value is None:
            self._property_updated = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "updated", six.integer_types)
        self._property_updated = value


class AddOrUpdateModelRequest(Request):
    """
    Add or update task model

    :param task: ID of the task
    :type task: str
    :param name: The task model name
    :type name: str
    :param model: The model ID
    :type model: str
    :param type: The task model type
    :type type: ModelTypeEnum
    :param iteration: Iteration (used to update task statistics)
    :type iteration: int
    """

    _service = "tasks"
    _action = "add_or_update_model"
    _version = "2.13"
    _schema = {
        "definitions": {"model_type_enum": {"enum": ["input", "output"], "type": "string"}},
        "properties": {
            "iteration": {
                "description": "Iteration (used to update task statistics)",
                "type": "integer",
            },
            "model": {"description": "The model ID", "type": "string"},
            "name": {"description": "The task model name", "type": "string"},
            "task": {"description": "ID of the task", "type": "string"},
            "type": {
                "$ref": "#/definitions/model_type_enum",
                "description": "The task model type",
            },
        },
        "required": ["task", "name", "model", "type"],
        "type": "object",
    }

    def __init__(
        self, task: str, name: str, model: str, type: Any, iteration: Optional[int] = None, **kwargs: Any
    ) -> None:
        super(AddOrUpdateModelRequest, self).__init__(**kwargs)
        self.task = task
        self.name = name
        self.model = model
        self.type = type
        self.iteration = iteration

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

    @schema_property("name")
    def name(self) -> str:
        return self._property_name

    @name.setter
    def name(self, value: str) -> None:
        if value is None:
            self._property_name = None
            return
        self.assert_isinstance(value, "name", six.string_types)
        self._property_name = value

    @schema_property("model")
    def model(self) -> str:
        return self._property_model

    @model.setter
    def model(self, value: str) -> None:
        if value is None:
            self._property_model = None
            return
        self.assert_isinstance(value, "model", six.string_types)
        self._property_model = value

    @schema_property("type")
    def type(self) -> Any:
        return self._property_type

    @type.setter
    def type(self, value: Any) -> None:
        if value is None:
            self._property_type = None
            return
        if isinstance(value, six.string_types):
            try:
                value = ModelTypeEnum(value)
            except ValueError:
                pass
        else:
            self.assert_isinstance(value, "type", enum.Enum)
        self._property_type = value

    @schema_property("iteration")
    def iteration(self) -> Optional[int]:
        return self._property_iteration

    @iteration.setter
    def iteration(self, value: Optional[int]) -> None:
        if value is None:
            self._property_iteration = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "iteration", six.integer_types)
        self._property_iteration = value


class AddOrUpdateModelResponse(Response):
    """
    Response of tasks.add_or_update_model endpoint.

    :param updated: Number of tasks updated (0 or 1)
    :type updated: int
    """

    _service = "tasks"
    _action = "add_or_update_model"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "updated": {
                "description": "Number of tasks updated (0 or 1)",
                "enum": [0, 1],
                "type": ["integer", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, updated: Optional[int] = None, **kwargs: Any) -> None:
        super(AddOrUpdateModelResponse, self).__init__(**kwargs)
        self.updated = updated

    @schema_property("updated")
    def updated(self) -> Optional[int]:
        return self._property_updated

    @updated.setter
    def updated(self, value: Optional[int]) -> None:
        if value is None:
            self._property_updated = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "updated", six.integer_types)
        self._property_updated = value


class ArchiveRequest(Request):
    """
    Archive tasks.
            If a task is queued it will first be dequeued and then archived.


    :param tasks: List of task ids
    :type tasks: Sequence[str]
    :param status_reason: Reason for status change
    :type status_reason: str
    :param status_message: Extra information regarding status change
    :type status_message: str
    """

    _service = "tasks"
    _action = "archive"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "status_message": {
                "description": "Extra information regarding status change",
                "type": "string",
            },
            "status_reason": {
                "description": "Reason for status change",
                "type": "string",
            },
            "tasks": {
                "description": "List of task ids",
                "items": {"type": "string"},
                "type": "array",
            },
        },
        "required": ["tasks"],
        "type": "object",
    }

    def __init__(
        self, tasks: List[str], status_reason: Optional[str] = None, status_message: Optional[str] = None, **kwargs: Any
    ) -> None:
        super(ArchiveRequest, self).__init__(**kwargs)
        self.tasks = tasks
        self.status_reason = status_reason
        self.status_message = status_message

    @schema_property("tasks")
    def tasks(self) -> List[str]:
        return self._property_tasks

    @tasks.setter
    def tasks(self, value: List[str]) -> None:
        if value is None:
            self._property_tasks = None
            return
        self.assert_isinstance(value, "tasks", (list, tuple))
        self.assert_isinstance(value, "tasks", six.string_types, is_array=True)
        self._property_tasks = value

    @schema_property("status_reason")
    def status_reason(self) -> Optional[str]:
        return self._property_status_reason

    @status_reason.setter
    def status_reason(self, value: Optional[str]) -> None:
        if value is None:
            self._property_status_reason = None
            return
        self.assert_isinstance(value, "status_reason", six.string_types)
        self._property_status_reason = value

    @schema_property("status_message")
    def status_message(self) -> Optional[str]:
        return self._property_status_message

    @status_message.setter
    def status_message(self, value: Optional[str]) -> None:
        if value is None:
            self._property_status_message = None
            return
        self.assert_isinstance(value, "status_message", six.string_types)
        self._property_status_message = value


class ArchiveResponse(Response):
    """
    Response of tasks.archive endpoint.

    :param archived: Indicates number of archived tasks
    :type archived: int
    """

    _service = "tasks"
    _action = "archive"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "archived": {
                "description": "Indicates number of archived tasks",
                "type": ["integer", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, archived: Optional[int] = None, **kwargs: Any) -> None:
        super(ArchiveResponse, self).__init__(**kwargs)
        self.archived = archived

    @schema_property("archived")
    def archived(self) -> Optional[int]:
        return self._property_archived

    @archived.setter
    def archived(self, value: Optional[int]) -> None:
        if value is None:
            self._property_archived = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "archived", six.integer_types)
        self._property_archived = value


class ArchiveManyRequest(Request):
    """
    Archive tasks

    :param ids: Entities to move
    :type ids: Sequence[str]
    :param status_reason: Reason for status change
    :type status_reason: str
    :param status_message: Extra information regarding status change
    :type status_message: str
    """

    _service = "tasks"
    _action = "archive_many"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "ids": {
                "description": "Entities to move",
                "items": {"type": "string"},
                "type": "array",
            },
            "status_message": {
                "description": "Extra information regarding status change",
                "type": "string",
            },
            "status_reason": {
                "description": "Reason for status change",
                "type": "string",
            },
        },
        "required": ["ids"],
        "type": "object",
    }

    def __init__(
        self, ids: List[str], status_reason: Optional[str] = None, status_message: Optional[str] = None, **kwargs: Any
    ) -> None:
        super(ArchiveManyRequest, self).__init__(**kwargs)
        self.ids = ids
        self.status_reason = status_reason
        self.status_message = status_message

    @schema_property("ids")
    def ids(self) -> List[str]:
        return self._property_ids

    @ids.setter
    def ids(self, value: List[str]) -> None:
        if value is None:
            self._property_ids = None
            return
        self.assert_isinstance(value, "ids", (list, tuple))
        self.assert_isinstance(value, "ids", six.string_types, is_array=True)
        self._property_ids = value

    @schema_property("status_reason")
    def status_reason(self) -> Optional[str]:
        return self._property_status_reason

    @status_reason.setter
    def status_reason(self, value: Optional[str]) -> None:
        if value is None:
            self._property_status_reason = None
            return
        self.assert_isinstance(value, "status_reason", six.string_types)
        self._property_status_reason = value

    @schema_property("status_message")
    def status_message(self) -> Optional[str]:
        return self._property_status_message

    @status_message.setter
    def status_message(self, value: Optional[str]) -> None:
        if value is None:
            self._property_status_message = None
            return
        self.assert_isinstance(value, "status_message", six.string_types)
        self._property_status_message = value


class ArchiveManyResponse(Response):
    """
    Response of tasks.archive_many endpoint.

    :param archived: Number of tasks archived
    :type archived: int
    """

    _service = "tasks"
    _action = "archive_many"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "failures": {
            "item": {
                "error": {
                    "description": "Error info",
                    "properties": {
                        "codes": {"item": {"type": "integer"}, "type": "array"},
                        "data": {"additionalProperties": True, "type": "object"},
                        "msg": {"type": "string"},
                    },
                    "type": "object",
                },
                "id": {"description": "ID of the failed entity", "type": "string"},
                "type": "object",
            },
            "type": "array",
        },
        "properties": {
            "archived": {
                "description": "Number of tasks archived",
                "type": ["integer", "null"],
            }
        },
    }

    def __init__(self, archived: Optional[int] = None, **kwargs: Any) -> None:
        super(ArchiveManyResponse, self).__init__(**kwargs)
        self.archived = archived

    @schema_property("archived")
    def archived(self) -> Optional[int]:
        return self._property_archived

    @archived.setter
    def archived(self, value: Optional[int]) -> None:
        if value is None:
            self._property_archived = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "archived", six.integer_types)
        self._property_archived = value


class CloneRequest(Request):
    """
    Clone an existing task

    :param task: ID of the task
    :type task: str
    :param new_task_name: The name of the cloned task. If not provided then taken
        from the original task
    :type new_task_name: str
    :param new_task_comment: The comment of the cloned task. If not provided then
        taken from the original task
    :type new_task_comment: str
    :param new_task_tags: The user-defined tags of the cloned task. If not provided
        then taken from the original task
    :type new_task_tags: Sequence[str]
    :param new_task_system_tags: The system tags of the cloned task. If not
        provided then empty
    :type new_task_system_tags: Sequence[str]
    :param new_task_parent: The parent of the cloned task. If not provided then
        taken from the original task
    :type new_task_parent: str
    :param new_task_project: The project of the cloned task. If not provided then
        taken from the original task
    :type new_task_project: str
    :param new_task_hyperparams: The hyper params for the new task. If not provided
        then taken from the original task
    :type new_task_hyperparams: dict
    :param new_task_configuration: The configuration for the new task. If not
        provided then taken from the original task
    :type new_task_configuration: dict
    :param execution_overrides: The execution params for the cloned task. The
        params not specified are taken from the original task
    :type execution_overrides: Execution
    :param validate_references: If set to 'false' then the task fields that are
        copied from the original task are not validated. The default is false.
    :type validate_references: bool
    :param new_project_name: Clone task to a new project by this name (only if
        `new_task_project` is not provided). If a project by this name already exists,
        task will be cloned to existing project.
    :type new_project_name: str
    :param new_task_input_models: The list of input models for the cloned task. If
        not specifed then copied from the original task
    :type new_task_input_models: Sequence[TaskModelItem]
    :param new_task_container: The docker container properties for the new task. If
        not provided then taken from the original task
    :type new_task_container: dict
    """

    _service = "tasks"
    _action = "clone"
    _version = "2.13"
    _schema = {
        "definitions": {
            "artifact": {
                "properties": {
                    "content_size": {
                        "description": "Raw data length in bytes",
                        "type": "integer",
                    },
                    "display_data": {
                        "description": "User-defined list of key/value pairs, sorted",
                        "items": {"items": {"type": "string"}, "type": "array"},
                        "type": "array",
                    },
                    "hash": {
                        "description": "Hash of entire raw data",
                        "type": "string",
                    },
                    "key": {"description": "Entry key", "type": "string"},
                    "mode": {
                        "$ref": "#/definitions/artifact_mode_enum",
                        "description": "System defined input/output indication",
                    },
                    "timestamp": {
                        "description": "Epoch time when artifact was created",
                        "type": "integer",
                    },
                    "type": {"description": "System defined type", "type": "string"},
                    "type_data": {
                        "$ref": "#/definitions/artifact_type_data",
                        "description": "Additional fields defined by the system",
                    },
                    "uri": {"description": "Raw data location", "type": "string"},
                },
                "required": ["key", "type"],
                "type": "object",
            },
            "artifact_mode_enum": {
                "default": "output",
                "enum": ["input", "output"],
                "type": "string",
            },
            "artifact_type_data": {
                "properties": {
                    "content_type": {
                        "description": "System defined raw data content type",
                        "type": ["string", "null"],
                    },
                    "data_hash": {
                        "description": "Hash of raw data, without any headers or descriptive parts",
                        "type": ["string", "null"],
                    },
                    "preview": {
                        "description": "Description or textual data",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "configuration_item": {
                "properties": {
                    "description": {
                        "description": "The parameter description. Optional",
                        "type": ["string", "null"],
                    },
                    "name": {
                        "description": "Name of the parameter. Should be unique",
                        "type": ["string", "null"],
                    },
                    "type": {
                        "description": "Type of the parameter. Optional",
                        "type": ["string", "null"],
                    },
                    "value": {
                        "description": "Value of the parameter",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "execution": {
                "properties": {
                    "artifacts": {
                        "description": "Task artifacts",
                        "items": {"$ref": "#/definitions/artifact"},
                        "type": ["array", "null"],
                    },
                    "framework": {
                        "description": "Framework related to the task. Case insensitive. Mandatory for Training tasks. ",
                        "type": ["string", "null"],
                    },
                    "model_desc": {
                        "additionalProperties": True,
                        "description": "Json object representing the Model descriptors",
                        "type": ["object", "null"],
                    },
                    "model_labels": {
                        "additionalProperties": {"type": "integer"},
                        "description": "Json object representing the ids of the labels in the model.\n                The keys are the layers' names and the values are the IDs.\n                Not applicable for Register (Import) tasks.\n                Mandatory for Training tasks",
                        "type": ["object", "null"],
                    },
                    "parameters": {
                        "additionalProperties": True,
                        "description": "Json object containing the Task parameters",
                        "type": ["object", "null"],
                    },
                    "queue": {
                        "description": "Queue ID where task was queued.",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "params_item": {
                "properties": {
                    "description": {
                        "description": "The parameter description. Optional",
                        "type": ["string", "null"],
                    },
                    "name": {
                        "description": "Name of the parameter. The combination of section and name should be unique",
                        "type": ["string", "null"],
                    },
                    "section": {
                        "description": "Section that the parameter belongs to",
                        "type": ["string", "null"],
                    },
                    "type": {
                        "description": "Type of the parameter. Optional",
                        "type": ["string", "null"],
                    },
                    "value": {
                        "description": "Value of the parameter",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "section_params": {
                "additionalProperties": {"$ref": "#/definitions/params_item"},
                "description": "Task section params",
                "type": "object",
            },
            "task_model_item": {
                "properties": {
                    "model": {"description": "The model ID", "type": "string"},
                    "name": {"description": "The task model name", "type": "string"},
                },
                "required": ["name", "model"],
                "type": "object",
            },
        },
        "properties": {
            "execution_overrides": {
                "$ref": "#/definitions/execution",
                "description": "The execution params for the cloned task. The params not specified are taken from the original task",
            },
            "new_project_name": {
                "description": "Clone task to a new project by this name (only if `new_task_project` is not provided). If a project by this name already exists, task will be cloned to existing project.",
                "type": "string",
            },
            "new_task_comment": {
                "description": "The comment of the cloned task. If not provided then taken from the original task",
                "type": "string",
            },
            "new_task_configuration": {
                "additionalProperties": {"$ref": "#/definitions/configuration_item"},
                "description": "The configuration for the new task. If not provided then taken from the original task",
                "type": "object",
            },
            "new_task_container": {
                "additionalProperties": {"type": ["string", "null"]},
                "description": "The docker container properties for the new task. If not provided then taken from the original task",
                "type": "object",
            },
            "new_task_hyperparams": {
                "additionalProperties": {"$ref": "#/definitions/section_params"},
                "description": "The hyper params for the new task. If not provided then taken from the original task",
                "type": "object",
            },
            "new_task_input_models": {
                "description": "The list of input models for the cloned task. If not specifed then copied from the original task",
                "items": {"$ref": "#/definitions/task_model_item"},
                "type": "array",
            },
            "new_task_name": {
                "description": "The name of the cloned task. If not provided then taken from the original task",
                "type": "string",
            },
            "new_task_parent": {
                "description": "The parent of the cloned task. If not provided then taken from the original task",
                "type": "string",
            },
            "new_task_project": {
                "description": "The project of the cloned task. If not provided then taken from the original task",
                "type": "string",
            },
            "new_task_system_tags": {
                "description": "The system tags of the cloned task. If not provided then empty",
                "items": {"type": "string"},
                "type": "array",
            },
            "new_task_tags": {
                "description": "The user-defined tags of the cloned task. If not provided then taken from the original task",
                "items": {"type": "string"},
                "type": "array",
            },
            "task": {"description": "ID of the task", "type": "string"},
            "validate_references": {
                "description": "If set to 'false' then the task fields that are copied from the original task are not validated. The default is false.",
                "type": "boolean",
            },
        },
        "required": ["task"],
        "type": "object",
    }

    def __init__(
        self,
        task: str,
        new_task_name: Optional[str] = None,
        new_task_comment: Optional[str] = None,
        new_task_tags: Optional[List[str]] = None,
        new_task_system_tags: Optional[List[str]] = None,
        new_task_parent: Optional[str] = None,
        new_task_project: Optional[str] = None,
        new_task_hyperparams: Optional[dict] = None,
        new_task_configuration: Optional[dict] = None,
        execution_overrides: Any = None,
        validate_references: Optional[bool] = None,
        new_project_name: Optional[str] = None,
        new_task_input_models: Optional[List[Any]] = None,
        new_task_container: Optional[dict] = None,
        **kwargs: Any
    ) -> None:
        super(CloneRequest, self).__init__(**kwargs)
        self.task = task
        self.new_task_name = new_task_name
        self.new_task_comment = new_task_comment
        self.new_task_tags = new_task_tags
        self.new_task_system_tags = new_task_system_tags
        self.new_task_parent = new_task_parent
        self.new_task_project = new_task_project
        self.new_task_hyperparams = new_task_hyperparams
        self.new_task_configuration = new_task_configuration
        self.execution_overrides = execution_overrides
        self.validate_references = validate_references
        self.new_project_name = new_project_name
        self.new_task_input_models = new_task_input_models
        self.new_task_container = new_task_container

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

    @schema_property("new_task_name")
    def new_task_name(self) -> Optional[str]:
        return self._property_new_task_name

    @new_task_name.setter
    def new_task_name(self, value: Optional[str]) -> None:
        if value is None:
            self._property_new_task_name = None
            return
        self.assert_isinstance(value, "new_task_name", six.string_types)
        self._property_new_task_name = value

    @schema_property("new_task_comment")
    def new_task_comment(self) -> Optional[str]:
        return self._property_new_task_comment

    @new_task_comment.setter
    def new_task_comment(self, value: Optional[str]) -> None:
        if value is None:
            self._property_new_task_comment = None
            return
        self.assert_isinstance(value, "new_task_comment", six.string_types)
        self._property_new_task_comment = value

    @schema_property("new_task_tags")
    def new_task_tags(self) -> Optional[List[str]]:
        return self._property_new_task_tags

    @new_task_tags.setter
    def new_task_tags(self, value: Optional[List[str]]) -> None:
        if value is None:
            self._property_new_task_tags = None
            return
        self.assert_isinstance(value, "new_task_tags", (list, tuple))
        self.assert_isinstance(value, "new_task_tags", six.string_types, is_array=True)
        self._property_new_task_tags = value

    @schema_property("new_task_system_tags")
    def new_task_system_tags(self) -> Optional[List[str]]:
        return self._property_new_task_system_tags

    @new_task_system_tags.setter
    def new_task_system_tags(self, value: Optional[List[str]]) -> None:
        if value is None:
            self._property_new_task_system_tags = None
            return
        self.assert_isinstance(value, "new_task_system_tags", (list, tuple))
        self.assert_isinstance(value, "new_task_system_tags", six.string_types, is_array=True)
        self._property_new_task_system_tags = value

    @schema_property("new_task_parent")
    def new_task_parent(self) -> Optional[str]:
        return self._property_new_task_parent

    @new_task_parent.setter
    def new_task_parent(self, value: Optional[str]) -> None:
        if value is None:
            self._property_new_task_parent = None
            return
        self.assert_isinstance(value, "new_task_parent", six.string_types)
        self._property_new_task_parent = value

    @schema_property("new_task_project")
    def new_task_project(self) -> Optional[str]:
        return self._property_new_task_project

    @new_task_project.setter
    def new_task_project(self, value: Optional[str]) -> None:
        if value is None:
            self._property_new_task_project = None
            return
        self.assert_isinstance(value, "new_task_project", six.string_types)
        self._property_new_task_project = value

    @schema_property("new_task_hyperparams")
    def new_task_hyperparams(self) -> Optional[dict]:
        return self._property_new_task_hyperparams

    @new_task_hyperparams.setter
    def new_task_hyperparams(self, value: Optional[dict]) -> None:
        if value is None:
            self._property_new_task_hyperparams = None
            return
        self.assert_isinstance(value, "new_task_hyperparams", (dict,))
        self._property_new_task_hyperparams = value

    @schema_property("new_task_configuration")
    def new_task_configuration(self) -> Optional[dict]:
        return self._property_new_task_configuration

    @new_task_configuration.setter
    def new_task_configuration(self, value: Optional[dict]) -> None:
        if value is None:
            self._property_new_task_configuration = None
            return
        self.assert_isinstance(value, "new_task_configuration", (dict,))
        self._property_new_task_configuration = value

    @schema_property("execution_overrides")
    def execution_overrides(self) -> Any:
        return self._property_execution_overrides

    @execution_overrides.setter
    def execution_overrides(self, value: Any) -> None:
        if value is None:
            self._property_execution_overrides = None
            return
        if isinstance(value, dict):
            value = Execution.from_dict(value)
        else:
            self.assert_isinstance(value, "execution_overrides", Execution)
        self._property_execution_overrides = value

    @schema_property("validate_references")
    def validate_references(self) -> Optional[bool]:
        return self._property_validate_references

    @validate_references.setter
    def validate_references(self, value: Optional[bool]) -> None:
        if value is None:
            self._property_validate_references = None
            return
        self.assert_isinstance(value, "validate_references", (bool,))
        self._property_validate_references = value

    @schema_property("new_project_name")
    def new_project_name(self) -> Optional[str]:
        return self._property_new_project_name

    @new_project_name.setter
    def new_project_name(self, value: Optional[str]) -> None:
        if value is None:
            self._property_new_project_name = None
            return
        self.assert_isinstance(value, "new_project_name", six.string_types)
        self._property_new_project_name = value

    @schema_property("new_task_input_models")
    def new_task_input_models(self) -> Optional[List[Any]]:
        return self._property_new_task_input_models

    @new_task_input_models.setter
    def new_task_input_models(self, value: Optional[List[Any]]) -> None:
        if value is None:
            self._property_new_task_input_models = None
            return
        self.assert_isinstance(value, "new_task_input_models", (list, tuple))
        if any((isinstance(v, dict) for v in value)):
            value = [TaskModelItem.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "new_task_input_models", TaskModelItem, is_array=True)
        self._property_new_task_input_models = value

    @schema_property("new_task_container")
    def new_task_container(self) -> Optional[dict]:
        return self._property_new_task_container

    @new_task_container.setter
    def new_task_container(self, value: Optional[dict]) -> None:
        if value is None:
            self._property_new_task_container = None
            return
        self.assert_isinstance(value, "new_task_container", (dict,))
        self._property_new_task_container = value


class CloneResponse(Response):
    """
    Response of tasks.clone endpoint.

    :param id: ID of the new task
    :type id: str
    :param new_project: In case the new_project_name was specified returns the
        target project details
    :type new_project: dict
    """

    _service = "tasks"
    _action = "clone"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "id": {"description": "ID of the new task", "type": ["string", "null"]},
            "new_project": {
                "description": "In case the new_project_name was specified returns the target project details",
                "properties": {
                    "id": {
                        "description": "The ID of the target project",
                        "type": "string",
                    },
                    "name": {
                        "description": "The name of the target project",
                        "type": "string",
                    },
                },
                "type": ["object", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, id: Optional[str] = None, new_project: Optional[dict] = None, **kwargs: Any) -> None:
        super(CloneResponse, self).__init__(**kwargs)
        self.id = id
        self.new_project = new_project

    @schema_property("id")
    def id(self) -> Optional[str]:
        return self._property_id

    @id.setter
    def id(self, value: Optional[str]) -> None:
        if value is None:
            self._property_id = None
            return
        self.assert_isinstance(value, "id", six.string_types)
        self._property_id = value

    @schema_property("new_project")
    def new_project(self) -> Optional[dict]:
        return self._property_new_project

    @new_project.setter
    def new_project(self, value: Optional[dict]) -> None:
        if value is None:
            self._property_new_project = None
            return
        self.assert_isinstance(value, "new_project", (dict,))
        self._property_new_project = value


class CloseRequest(Request):
    """
    Indicates that task is closed

    :param force: Allows forcing state change even if transition is not supported
    :type force: bool
    :param task: Task ID
    :type task: str
    :param status_reason: Reason for status change
    :type status_reason: str
    :param status_message: Extra information regarding status change
    :type status_message: str
    """

    _service = "tasks"
    _action = "close"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "force": {
                "default": False,
                "description": "Allows forcing state change even if transition is not supported",
                "type": ["boolean", "null"],
            },
            "status_message": {
                "description": "Extra information regarding status change",
                "type": "string",
            },
            "status_reason": {
                "description": "Reason for status change",
                "type": "string",
            },
            "task": {"description": "Task ID", "type": "string"},
        },
        "required": ["task"],
        "type": "object",
    }

    def __init__(
        self,
        task: str,
        force: Optional[bool] = False,
        status_reason: Optional[str] = None,
        status_message: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super(CloseRequest, self).__init__(**kwargs)
        self.force = force
        self.task = task
        self.status_reason = status_reason
        self.status_message = status_message

    @schema_property("force")
    def force(self) -> Optional[bool]:
        return self._property_force

    @force.setter
    def force(self, value: Optional[bool]) -> None:
        if value is None:
            self._property_force = None
            return
        self.assert_isinstance(value, "force", (bool,))
        self._property_force = value

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

    @schema_property("status_reason")
    def status_reason(self) -> Optional[str]:
        return self._property_status_reason

    @status_reason.setter
    def status_reason(self, value: Optional[str]) -> None:
        if value is None:
            self._property_status_reason = None
            return
        self.assert_isinstance(value, "status_reason", six.string_types)
        self._property_status_reason = value

    @schema_property("status_message")
    def status_message(self) -> Optional[str]:
        return self._property_status_message

    @status_message.setter
    def status_message(self, value: Optional[str]) -> None:
        if value is None:
            self._property_status_message = None
            return
        self.assert_isinstance(value, "status_message", six.string_types)
        self._property_status_message = value


class CloseResponse(Response):
    """
    Response of tasks.close endpoint.

    :param updated: Number of tasks updated (0 or 1)
    :type updated: int
    :param fields: Updated fields names and values
    :type fields: dict
    """

    _service = "tasks"
    _action = "close"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "fields": {
                "additionalProperties": True,
                "description": "Updated fields names and values",
                "type": ["object", "null"],
            },
            "updated": {
                "description": "Number of tasks updated (0 or 1)",
                "enum": [0, 1],
                "type": ["integer", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, updated: Optional[int] = None, fields: Optional[dict] = None, **kwargs: Any) -> None:
        super(CloseResponse, self).__init__(**kwargs)
        self.updated = updated
        self.fields = fields

    @schema_property("updated")
    def updated(self) -> Optional[int]:
        return self._property_updated

    @updated.setter
    def updated(self, value: Optional[int]) -> None:
        if value is None:
            self._property_updated = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "updated", six.integer_types)
        self._property_updated = value

    @schema_property("fields")
    def fields(self) -> Optional[dict]:
        return self._property_fields

    @fields.setter
    def fields(self, value: Optional[dict]) -> None:
        if value is None:
            self._property_fields = None
            return
        self.assert_isinstance(value, "fields", (dict,))
        self._property_fields = value


class CompletedRequest(Request):
    """
    Signal a task has completed

    :param force: If not true, call fails if the task status is not
        in_progress/stopped
    :type force: bool
    :param task: Task ID
    :type task: str
    :param status_reason: Reason for status change
    :type status_reason: str
    :param status_message: Extra information regarding status change
    :type status_message: str
    """

    _service = "tasks"
    _action = "completed"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "force": {
                "default": False,
                "description": "If not true, call fails if the task status is not in_progress/stopped",
                "type": ["boolean", "null"],
            },
            "status_message": {
                "description": "Extra information regarding status change",
                "type": "string",
            },
            "status_reason": {
                "description": "Reason for status change",
                "type": "string",
            },
            "task": {"description": "Task ID", "type": "string"},
        },
        "required": ["task"],
        "type": "object",
    }

    def __init__(
        self,
        task: str,
        force: Optional[bool] = False,
        status_reason: Optional[str] = None,
        status_message: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super(CompletedRequest, self).__init__(**kwargs)
        self.force = force
        self.task = task
        self.status_reason = status_reason
        self.status_message = status_message

    @schema_property("force")
    def force(self) -> Optional[bool]:
        return self._property_force

    @force.setter
    def force(self, value: Optional[bool]) -> None:
        if value is None:
            self._property_force = None
            return
        self.assert_isinstance(value, "force", (bool,))
        self._property_force = value

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

    @schema_property("status_reason")
    def status_reason(self) -> Optional[str]:
        return self._property_status_reason

    @status_reason.setter
    def status_reason(self, value: Optional[str]) -> None:
        if value is None:
            self._property_status_reason = None
            return
        self.assert_isinstance(value, "status_reason", six.string_types)
        self._property_status_reason = value

    @schema_property("status_message")
    def status_message(self) -> Optional[str]:
        return self._property_status_message

    @status_message.setter
    def status_message(self, value: Optional[str]) -> None:
        if value is None:
            self._property_status_message = None
            return
        self.assert_isinstance(value, "status_message", six.string_types)
        self._property_status_message = value


class CompletedResponse(Response):
    """
    Response of tasks.completed endpoint.

    :param updated: Number of tasks updated (0 or 1)
    :type updated: int
    :param fields: Updated fields names and values
    :type fields: dict
    """

    _service = "tasks"
    _action = "completed"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "fields": {
                "additionalProperties": True,
                "description": "Updated fields names and values",
                "type": ["object", "null"],
            },
            "updated": {
                "description": "Number of tasks updated (0 or 1)",
                "enum": [0, 1],
                "type": ["integer", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, updated: Optional[int] = None, fields: Optional[dict] = None, **kwargs: Any) -> None:
        super(CompletedResponse, self).__init__(**kwargs)
        self.updated = updated
        self.fields = fields

    @schema_property("updated")
    def updated(self) -> Optional[int]:
        return self._property_updated

    @updated.setter
    def updated(self, value: Optional[int]) -> None:
        if value is None:
            self._property_updated = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "updated", six.integer_types)
        self._property_updated = value

    @schema_property("fields")
    def fields(self) -> Optional[dict]:
        return self._property_fields

    @fields.setter
    def fields(self, value: Optional[dict]) -> None:
        if value is None:
            self._property_fields = None
            return
        self.assert_isinstance(value, "fields", (dict,))
        self._property_fields = value


class CreateRequest(Request):
    """
    Create a new task

    :param name: Task name. Unique within the company.
    :type name: str
    :param tags: User-defined tags list
    :type tags: Sequence[str]
    :param system_tags: System tags list. This field is reserved for system use,
        please don't use it.
    :type system_tags: Sequence[str]
    :param type: Type of task
    :type type: TaskTypeEnum
    :param comment: Free text comment
    :type comment: str
    :param parent: Parent task ID Must be a completed task.
    :type parent: str
    :param project: Project ID of the project to which this task is assigned Must
        exist[ab]
    :type project: str
    :param output_dest: Output storage ID Must be a reference to an existing
        storage.
    :type output_dest: str
    :param execution: Task execution params
    :type execution: Execution
    :param script: Script info
    :type script: Script
    :param hyperparams: Task hyper params per section
    :type hyperparams: dict
    :param configuration: Task configuration params
    :type configuration: dict
    :param models: Task models
    :type models: TaskModels
    :param container: Docker container parameters
    :type container: dict
    """

    _service = "tasks"
    _action = "create"
    _version = "2.13"
    _schema = {
        "definitions": {
            "artifact": {
                "properties": {
                    "content_size": {
                        "description": "Raw data length in bytes",
                        "type": "integer",
                    },
                    "display_data": {
                        "description": "User-defined list of key/value pairs, sorted",
                        "items": {"items": {"type": "string"}, "type": "array"},
                        "type": "array",
                    },
                    "hash": {
                        "description": "Hash of entire raw data",
                        "type": "string",
                    },
                    "key": {"description": "Entry key", "type": "string"},
                    "mode": {
                        "$ref": "#/definitions/artifact_mode_enum",
                        "description": "System defined input/output indication",
                    },
                    "timestamp": {
                        "description": "Epoch time when artifact was created",
                        "type": "integer",
                    },
                    "type": {"description": "System defined type", "type": "string"},
                    "type_data": {
                        "$ref": "#/definitions/artifact_type_data",
                        "description": "Additional fields defined by the system",
                    },
                    "uri": {"description": "Raw data location", "type": "string"},
                },
                "required": ["key", "type"],
                "type": "object",
            },
            "artifact_mode_enum": {
                "default": "output",
                "enum": ["input", "output"],
                "type": "string",
            },
            "artifact_type_data": {
                "properties": {
                    "content_type": {
                        "description": "System defined raw data content type",
                        "type": ["string", "null"],
                    },
                    "data_hash": {
                        "description": "Hash of raw data, without any headers or descriptive parts",
                        "type": ["string", "null"],
                    },
                    "preview": {
                        "description": "Description or textual data",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "configuration_item": {
                "properties": {
                    "description": {
                        "description": "The parameter description. Optional",
                        "type": ["string", "null"],
                    },
                    "name": {
                        "description": "Name of the parameter. Should be unique",
                        "type": ["string", "null"],
                    },
                    "type": {
                        "description": "Type of the parameter. Optional",
                        "type": ["string", "null"],
                    },
                    "value": {
                        "description": "Value of the parameter",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "execution": {
                "properties": {
                    "artifacts": {
                        "description": "Task artifacts",
                        "items": {"$ref": "#/definitions/artifact"},
                        "type": ["array", "null"],
                    },
                    "framework": {
                        "description": "Framework related to the task. Case insensitive. Mandatory for Training tasks. ",
                        "type": ["string", "null"],
                    },
                    "model_desc": {
                        "additionalProperties": True,
                        "description": "Json object representing the Model descriptors",
                        "type": ["object", "null"],
                    },
                    "model_labels": {
                        "additionalProperties": {"type": "integer"},
                        "description": "Json object representing the ids of the labels in the model.\n                The keys are the layers' names and the values are the IDs.\n                Not applicable for Register (Import) tasks.\n                Mandatory for Training tasks",
                        "type": ["object", "null"],
                    },
                    "parameters": {
                        "additionalProperties": True,
                        "description": "Json object containing the Task parameters",
                        "type": ["object", "null"],
                    },
                    "queue": {
                        "description": "Queue ID where task was queued.",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "params_item": {
                "properties": {
                    "description": {
                        "description": "The parameter description. Optional",
                        "type": ["string", "null"],
                    },
                    "name": {
                        "description": "Name of the parameter. The combination of section and name should be unique",
                        "type": ["string", "null"],
                    },
                    "section": {
                        "description": "Section that the parameter belongs to",
                        "type": ["string", "null"],
                    },
                    "type": {
                        "description": "Type of the parameter. Optional",
                        "type": ["string", "null"],
                    },
                    "value": {
                        "description": "Value of the parameter",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "script": {
                "properties": {
                    "binary": {
                        "default": "python",
                        "description": "Binary to use when running the script",
                        "type": ["string", "null"],
                    },
                    "branch": {
                        "description": "Repository branch id If not provided and tag not provided, default repository branch is used.",
                        "type": ["string", "null"],
                    },
                    "diff": {
                        "description": "Uncommitted changes found in the repository when task was run",
                        "type": ["string", "null"],
                    },
                    "entry_point": {
                        "description": "Path to execute within the repository",
                        "type": ["string", "null"],
                    },
                    "repository": {
                        "description": "Name of the repository where the script is located",
                        "type": ["string", "null"],
                    },
                    "requirements": {
                        "description": "A JSON object containing requirements strings by key",
                        "type": ["object", "null"],
                    },
                    "tag": {
                        "description": "Repository tag",
                        "type": ["string", "null"],
                    },
                    "version_num": {
                        "description": "Version (changeset) number. Optional (default is head version) Unused if tag is provided.",
                        "type": ["string", "null"],
                    },
                    "working_dir": {
                        "description": "Path to the folder from which to run the script Default - root folder of repository",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "section_params": {
                "additionalProperties": {"$ref": "#/definitions/params_item"},
                "description": "Task section params",
                "type": "object",
            },
            "task_model_item": {
                "properties": {
                    "model": {"description": "The model ID", "type": "string"},
                    "name": {"description": "The task model name", "type": "string"},
                },
                "required": ["name", "model"],
                "type": "object",
            },
            "task_models": {
                "properties": {
                    "input": {
                        "description": "The list of task input models",
                        "items": {"$ref": "#/definitions/task_model_item"},
                        "type": ["array", "null"],
                    },
                    "output": {
                        "description": "The list of task output models",
                        "items": {"$ref": "#/definitions/task_model_item"},
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            },
            "task_type_enum": {
                "enum": [
                    "training",
                    "testing",
                    "inference",
                    "data_processing",
                    "application",
                    "monitor",
                    "controller",
                    "optimizer",
                    "service",
                    "qc",
                    "custom",
                ],
                "type": "string",
            },
        },
        "properties": {
            "comment": {"description": "Free text comment ", "type": "string"},
            "configuration": {
                "additionalProperties": {"$ref": "#/definitions/configuration_item"},
                "description": "Task configuration params",
                "type": "object",
            },
            "container": {
                "type": "object",
                "description": "Docker container parameters",
                "additionalProperties": {"type": ["string", "null"]},
            },
            "execution": {
                "$ref": "#/definitions/execution",
                "description": "Task execution params",
            },
            "hyperparams": {
                "additionalProperties": {"$ref": "#/definitions/section_params"},
                "description": "Task hyper params per section",
                "type": "object",
            },
            "models": {
                "$ref": "#/definitions/task_models",
                "description": "Task models",
            },
            "name": {
                "description": "Task name. Unique within the company.",
                "type": "string",
            },
            "output_dest": {
                "description": "Output storage id Must be a reference to an existing storage.",
                "type": "string",
            },
            "parent": {
                "description": "Parent task id Must be a completed task.",
                "type": "string",
            },
            "project": {
                "description": "Project ID of the project to which this task is assigned Must exist[ab]",
                "type": "string",
            },
            "script": {"$ref": "#/definitions/script", "description": "Script info"},
            "system_tags": {
                "description": "System tags list. This field is reserved for system use, please don't use it.",
                "items": {"type": "string"},
                "type": "array",
            },
            "tags": {
                "description": "User-defined tags list",
                "items": {"type": "string"},
                "type": "array",
            },
            "type": {
                "$ref": "#/definitions/task_type_enum",
                "description": "Type of task",
            },
        },
        "required": ["name", "type"],
        "type": "object",
    }

    def __init__(
        self,
        name: str,
        type: Any,
        tags: Optional[List[str]] = None,
        system_tags: Optional[List[str]] = None,
        comment: Optional[str] = None,
        parent: Optional[str] = None,
        project: Optional[str] = None,
        input: Any = None,
        output_dest: Optional[str] = None,
        execution: Any = None,
        script: Any = None,
        hyperparams: Optional[dict] = None,
        configuration: Optional[dict] = None,
        models: Any = None,
        container: Optional[dict] = None,
        **kwargs: Any
    ) -> None:
        super(CreateRequest, self).__init__(**kwargs)
        self.name = name
        self.tags = tags
        self.system_tags = system_tags
        self.type = type
        self.comment = comment
        self.parent = parent
        self.project = project
        self.input = input
        self.output_dest = output_dest
        self.execution = execution
        self.script = script
        self.hyperparams = hyperparams
        self.configuration = configuration
        self.models = models
        self.container = container

    @schema_property("name")
    def name(self) -> str:
        return self._property_name

    @name.setter
    def name(self, value: str) -> None:
        if value is None:
            self._property_name = None
            return
        self.assert_isinstance(value, "name", six.string_types)
        self._property_name = value

    @schema_property("tags")
    def tags(self) -> Optional[List[str]]:
        return self._property_tags

    @tags.setter
    def tags(self, value: Optional[List[str]]) -> None:
        if value is None:
            self._property_tags = None
            return
        self.assert_isinstance(value, "tags", (list, tuple))
        self.assert_isinstance(value, "tags", six.string_types, is_array=True)
        self._property_tags = value

    @schema_property("system_tags")
    def system_tags(self) -> Optional[List[str]]:
        return self._property_system_tags

    @system_tags.setter
    def system_tags(self, value: Optional[List[str]]) -> None:
        if value is None:
            self._property_system_tags = None
            return
        self.assert_isinstance(value, "system_tags", (list, tuple))
        self.assert_isinstance(value, "system_tags", six.string_types, is_array=True)
        self._property_system_tags = value

    @schema_property("type")
    def type(self) -> Any:
        return self._property_type

    @type.setter
    def type(self, value: Any) -> None:
        if value is None:
            self._property_type = None
            return
        if isinstance(value, six.string_types):
            try:
                value = TaskTypeEnum(value)
            except ValueError:
                pass
        else:
            self.assert_isinstance(value, "type", enum.Enum)
        self._property_type = value

    @schema_property("comment")
    def comment(self) -> Optional[str]:
        return self._property_comment

    @comment.setter
    def comment(self, value: Optional[str]) -> None:
        if value is None:
            self._property_comment = None
            return
        self.assert_isinstance(value, "comment", six.string_types)
        self._property_comment = value

    @schema_property("parent")
    def parent(self) -> Optional[str]:
        return self._property_parent

    @parent.setter
    def parent(self, value: Optional[str]) -> None:
        if value is None:
            self._property_parent = None
            return
        self.assert_isinstance(value, "parent", six.string_types)
        self._property_parent = value

    @schema_property("project")
    def project(self) -> Optional[str]:
        return self._property_project

    @project.setter
    def project(self, value: Optional[str]) -> None:
        if value is None:
            self._property_project = None
            return
        self.assert_isinstance(value, "project", six.string_types)
        self._property_project = value

    @schema_property("input")
    def input(self) -> Any:
        return self._property_input

    @input.setter
    def input(self, value: Any) -> None:
        self._property_input = value

    @schema_property("output_dest")
    def output_dest(self) -> Optional[str]:
        return self._property_output_dest

    @output_dest.setter
    def output_dest(self, value: Optional[str]) -> None:
        if value is None:
            self._property_output_dest = None
            return
        self.assert_isinstance(value, "output_dest", six.string_types)
        self._property_output_dest = value

    @schema_property("execution")
    def execution(self) -> Any:
        return self._property_execution

    @execution.setter
    def execution(self, value: Any) -> None:
        if value is None:
            self._property_execution = None
            return
        if isinstance(value, dict):
            value = Execution.from_dict(value)
        else:
            self.assert_isinstance(value, "execution", Execution)
        self._property_execution = value

    @schema_property("script")
    def script(self) -> Any:
        return self._property_script

    @script.setter
    def script(self, value: Any) -> None:
        if value is None:
            self._property_script = None
            return
        if isinstance(value, dict):
            value = Script.from_dict(value)
        else:
            self.assert_isinstance(value, "script", Script)
        self._property_script = value

    @schema_property("hyperparams")
    def hyperparams(self) -> Optional[dict]:
        return self._property_hyperparams

    @hyperparams.setter
    def hyperparams(self, value: Optional[dict]) -> None:
        if value is None:
            self._property_hyperparams = None
            return
        self.assert_isinstance(value, "hyperparams", dict)
        self.assert_isinstance(value.keys(), "hyperparams_keys", six.string_types, is_array=True)
        self.assert_isinstance(value.values(), "hyperparams_values", (SectionParams, dict), is_array=True)
        value = dict(((k, SectionParams(**v) if isinstance(v, dict) else v) for (k, v) in value.items()))
        self._property_hyperparams = value

    @schema_property("configuration")
    def configuration(self) -> Optional[dict]:
        return self._property_configuration

    @configuration.setter
    def configuration(self, value: Optional[dict]) -> None:
        if value is None:
            self._property_configuration = None
            return
        self.assert_isinstance(value, "configuration", dict)
        self.assert_isinstance(value.keys(), "configuration_keys", six.string_types, is_array=True)
        self.assert_isinstance(
            value.values(),
            "configuration_values",
            (ConfigurationItem, dict),
            is_array=True,
        )
        value = dict(((k, ConfigurationItem(**v) if isinstance(v, dict) else v) for (k, v) in value.items()))
        self._property_configuration = value

    @schema_property("models")
    def models(self) -> Any:
        return self._property_models

    @models.setter
    def models(self, value: Any) -> None:
        if value is None:
            self._property_models = None
            return
        if isinstance(value, dict):
            value = TaskModels.from_dict(value)
        else:
            self.assert_isinstance(value, "models", TaskModels)
        self._property_models = value

    @schema_property("container")
    def container(self) -> Optional[dict]:
        return self._property_container

    @container.setter
    def container(self, value: Optional[dict]) -> None:
        if value is None:
            self._property_container = None
            return
        self.assert_isinstance(value, "container", dict)
        self._property_container = value


class CreateResponse(Response):
    """
    Response of tasks.create endpoint.

    :param id: ID of the task
    :type id: str
    """

    _service = "tasks"
    _action = "create"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {"id": {"description": "ID of the task", "type": ["string", "null"]}},
        "type": "object",
    }

    def __init__(self, id: Optional[str] = None, **kwargs: Any) -> None:
        super(CreateResponse, self).__init__(**kwargs)
        self.id = id

    @schema_property("id")
    def id(self) -> Optional[str]:
        return self._property_id

    @id.setter
    def id(self, value: Optional[str]) -> None:
        if value is None:
            self._property_id = None
            return
        self.assert_isinstance(value, "id", six.string_types)
        self._property_id = value


class DeleteRequest(Request):
    """
    Delete a task along with any information stored for it (statistics, frame updates etc.)
            Unless Force flag is provided, operation will fail if task has objects associated with it - i.e. children tasks and projects.
            Models that refer to the deleted task will be updated with a task ID indicating a deleted task.


    :param move_to_trash: Move task to trash instead of deleting it. For internal
        use only, tasks in the trash are not visible from the API and cannot be
        restored!
    :type move_to_trash: bool
    :param force: If not true, call fails if the task status is 'in_progress'
    :type force: bool
    :param task: Task ID
    :type task: str
    :param status_reason: Reason for status change
    :type status_reason: str
    :param status_message: Extra information regarding status change
    :type status_message: str
    :param return_file_urls: If set to 'true' then return the urls of the files
        that were uploaded by this task. Default value is 'false'
    :type return_file_urls: bool
    :param delete_output_models: If set to 'true' then delete output models of this
        task that are not referenced by other tasks. Default value is 'true'
    :type delete_output_models: bool
    """

    _service = "tasks"
    _action = "delete"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "delete_output_models": {
                "description": "If set to 'true' then delete output models of this task that are not referenced by other tasks. Default value is 'true'",
                "type": "boolean",
            },
            "force": {
                "default": False,
                "description": "If not true, call fails if the task status is 'in_progress'",
                "type": ["boolean", "null"],
            },
            "move_to_trash": {
                "default": False,
                "description": "Move task to trash instead of deleting it. For internal use only, tasks in the trash are not visible from the API and cannot be restored!",
                "type": ["boolean", "null"],
            },
            "return_file_urls": {
                "description": "If set to 'true' then return the urls of the files that were uploaded by this task. Default value is 'false'",
                "type": "boolean",
            },
            "status_message": {
                "description": "Extra information regarding status change",
                "type": "string",
            },
            "status_reason": {
                "description": "Reason for status change",
                "type": "string",
            },
            "task": {"description": "Task ID", "type": "string"},
        },
        "required": ["task"],
        "type": "object",
    }

    def __init__(
        self,
        task: str,
        move_to_trash: Optional[bool] = False,
        force: Optional[bool] = False,
        status_reason: Optional[str] = None,
        status_message: Optional[str] = None,
        return_file_urls: Optional[bool] = None,
        delete_output_models: Optional[bool] = None,
        **kwargs: Any
    ) -> None:
        super(DeleteRequest, self).__init__(**kwargs)
        self.move_to_trash = move_to_trash
        self.force = force
        self.task = task
        self.status_reason = status_reason
        self.status_message = status_message
        self.return_file_urls = return_file_urls
        self.delete_output_models = delete_output_models

    @schema_property("move_to_trash")
    def move_to_trash(self) -> Optional[bool]:
        return self._property_move_to_trash

    @move_to_trash.setter
    def move_to_trash(self, value: Optional[bool]) -> None:
        if value is None:
            self._property_move_to_trash = None
            return
        self.assert_isinstance(value, "move_to_trash", (bool,))
        self._property_move_to_trash = value

    @schema_property("force")
    def force(self) -> Optional[bool]:
        return self._property_force

    @force.setter
    def force(self, value: Optional[bool]) -> None:
        if value is None:
            self._property_force = None
            return
        self.assert_isinstance(value, "force", (bool,))
        self._property_force = value

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

    @schema_property("status_reason")
    def status_reason(self) -> Optional[str]:
        return self._property_status_reason

    @status_reason.setter
    def status_reason(self, value: Optional[str]) -> None:
        if value is None:
            self._property_status_reason = None
            return
        self.assert_isinstance(value, "status_reason", six.string_types)
        self._property_status_reason = value

    @schema_property("status_message")
    def status_message(self) -> Optional[str]:
        return self._property_status_message

    @status_message.setter
    def status_message(self, value: Optional[str]) -> None:
        if value is None:
            self._property_status_message = None
            return
        self.assert_isinstance(value, "status_message", six.string_types)
        self._property_status_message = value

    @schema_property("return_file_urls")
    def return_file_urls(self) -> Optional[bool]:
        return self._property_return_file_urls

    @return_file_urls.setter
    def return_file_urls(self, value: Optional[bool]) -> None:
        if value is None:
            self._property_return_file_urls = None
            return
        self.assert_isinstance(value, "return_file_urls", (bool,))
        self._property_return_file_urls = value

    @schema_property("delete_output_models")
    def delete_output_models(self) -> Optional[bool]:
        return self._property_delete_output_models

    @delete_output_models.setter
    def delete_output_models(self, value: Optional[bool]) -> None:
        if value is None:
            self._property_delete_output_models = None
            return
        self.assert_isinstance(value, "delete_output_models", (bool,))
        self._property_delete_output_models = value


class DeleteResponse(Response):
    """
    Response of tasks.delete endpoint.

    :param deleted: Indicates whether the task was deleted
    :type deleted: bool
    :param updated_children: Number of child tasks whose parent property was
        updated
    :type updated_children: int
    :param updated_models: Number of models whose task property was updated
    :type updated_models: int
    :param frames: Response from frames.rollback
    :type frames: dict
    :param events: Response from events.delete_for_task
    :type events: dict
    :param urls: The urls of the files that were uploaded by this task. Returned if
        the 'return_file_urls' was set to 'true'
    :type urls: TaskUrls
    """

    _service = "tasks"
    _action = "delete"
    _version = "2.13"
    _schema = {
        "definitions": {
            "task_urls": {
                "properties": {
                    "artifact_urls": {
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                    "event_urls": {
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                    "model_urls": {
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            }
        },
        "properties": {
            "deleted": {
                "description": "Indicates whether the task was deleted",
                "type": ["boolean", "null"],
            },
            "events": {
                "additionalProperties": True,
                "description": "Response from events.delete_for_task",
                "type": ["object", "null"],
            },
            "frames": {
                "additionalProperties": True,
                "description": "Response from frames.rollback",
                "type": ["object", "null"],
            },
            "updated_children": {
                "description": "Number of child tasks whose parent property was updated",
                "type": ["integer", "null"],
            },
            "updated_models": {
                "description": "Number of models whose task property was updated",
                "type": ["integer", "null"],
            },
            "urls": {
                "description": "The urls of the files that were uploaded by this task. Returned if the 'return_file_urls' was set to 'true'",
                "oneOf": [{"$ref": "#/definitions/task_urls"}, {"type": "null"}],
            },
        },
        "type": "object",
    }

    def __init__(
        self,
        deleted: Optional[bool] = None,
        updated_children: Optional[int] = None,
        updated_models: Optional[int] = None,
        frames: Optional[dict] = None,
        events: Optional[dict] = None,
        urls: Any = None,
        **kwargs: Any
    ) -> None:
        super(DeleteResponse, self).__init__(**kwargs)
        self.deleted = deleted
        self.updated_children = updated_children
        self.updated_models = updated_models
        self.frames = frames
        self.events = events
        self.urls = urls

    @schema_property("deleted")
    def deleted(self) -> Optional[bool]:
        return self._property_deleted

    @deleted.setter
    def deleted(self, value: Optional[bool]) -> None:
        if value is None:
            self._property_deleted = None
            return
        self.assert_isinstance(value, "deleted", (bool,))
        self._property_deleted = value

    @schema_property("updated_children")
    def updated_children(self) -> Optional[int]:
        return self._property_updated_children

    @updated_children.setter
    def updated_children(self, value: Optional[int]) -> None:
        if value is None:
            self._property_updated_children = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "updated_children", six.integer_types)
        self._property_updated_children = value

    @schema_property("updated_models")
    def updated_models(self) -> Optional[int]:
        return self._property_updated_models

    @updated_models.setter
    def updated_models(self, value: Optional[int]) -> None:
        if value is None:
            self._property_updated_models = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "updated_models", six.integer_types)
        self._property_updated_models = value

    @schema_property("frames")
    def frames(self) -> Optional[dict]:
        return self._property_frames

    @frames.setter
    def frames(self, value: Optional[dict]) -> None:
        if value is None:
            self._property_frames = None
            return
        self.assert_isinstance(value, "frames", (dict,))
        self._property_frames = value

    @schema_property("events")
    def events(self) -> Optional[dict]:
        return self._property_events

    @events.setter
    def events(self, value: Optional[dict]) -> None:
        if value is None:
            self._property_events = None
            return
        self.assert_isinstance(value, "events", (dict,))
        self._property_events = value

    @schema_property("urls")
    def urls(self) -> Any:
        return self._property_urls

    @urls.setter
    def urls(self, value: Any) -> None:
        if value is None:
            self._property_urls = None
            return
        if isinstance(value, dict):
            value = TaskUrls.from_dict(value)
        else:
            self.assert_isinstance(value, "urls", TaskUrls)
        self._property_urls = value


class DeleteArtifactsRequest(Request):
    """
    Delete existing artifacts (search by key/mode)

    :param task: Task ID
    :type task: str
    :param artifacts: Artifacts to delete
    :type artifacts: Sequence[ArtifactId]
    :param force: If set to True then both new and running task artifacts can be
        deleted. Otherwise only the new task ones. Default is False
    :type force: bool
    """

    _service = "tasks"
    _action = "delete_artifacts"
    _version = "2.13"
    _schema = {
        "definitions": {
            "artifact_id": {
                "properties": {
                    "key": {"description": "Entry key", "type": "string"},
                    "mode": {
                        "$ref": "#/definitions/artifact_mode_enum",
                        "description": "System defined input/output indication",
                    },
                },
                "required": ["key"],
                "type": "object",
            },
            "artifact_mode_enum": {
                "default": "output",
                "enum": ["input", "output"],
                "type": "string",
            },
        },
        "properties": {
            "artifacts": {
                "description": "Artifacts to delete",
                "items": {"$ref": "#/definitions/artifact_id"},
                "type": "array",
            },
            "force": {
                "description": "If set to True then both new and running task artifacts can be deleted. Otherwise only the new task ones. Default is False",
                "type": "boolean",
            },
            "task": {"description": "Task ID", "type": "string"},
        },
        "required": ["task", "artifacts"],
        "type": "object",
    }

    def __init__(self, task: str, artifacts: List[Any], force: Optional[bool] = None, **kwargs: Any) -> None:
        super(DeleteArtifactsRequest, self).__init__(**kwargs)
        self.task = task
        self.artifacts = artifacts
        self.force = force

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

    @schema_property("artifacts")
    def artifacts(self) -> List[Any]:
        return self._property_artifacts

    @artifacts.setter
    def artifacts(self, value: List[Any]) -> None:
        if value is None:
            self._property_artifacts = None
            return
        self.assert_isinstance(value, "artifacts", (list, tuple))
        if any((isinstance(v, dict) for v in value)):
            value = [ArtifactId.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "artifacts", ArtifactId, is_array=True)
        self._property_artifacts = value

    @schema_property("force")
    def force(self) -> Optional[bool]:
        return self._property_force

    @force.setter
    def force(self, value: Optional[bool]) -> None:
        if value is None:
            self._property_force = None
            return
        self.assert_isinstance(value, "force", (bool,))
        self._property_force = value


class DeleteArtifactsResponse(Response):
    """
    Response of tasks.delete_artifacts endpoint.

    :param deleted: Indicates if the task was updated successfully
    :type deleted: int
    """

    _service = "tasks"
    _action = "delete_artifacts"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "deleted": {
                "description": "Indicates if the task was updated successfully",
                "type": ["integer", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, deleted: Optional[int] = None, **kwargs: Any) -> None:
        super(DeleteArtifactsResponse, self).__init__(**kwargs)
        self.deleted = deleted

    @schema_property("deleted")
    def deleted(self) -> Optional[int]:
        return self._property_deleted

    @deleted.setter
    def deleted(self, value: Optional[int]) -> None:
        if value is None:
            self._property_deleted = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "deleted", six.integer_types)
        self._property_deleted = value


class DeleteConfigurationRequest(Request):
    """
    Delete task configuration items

    :param task: Task ID
    :type task: str
    :param configuration: List of configuration itemss to delete
    :type configuration: Sequence[str]
    :param force: If set to True then both new and running task configuration can
        be deleted. Otherwise only the new task ones. Default is False
    :type force: bool
    """

    _service = "tasks"
    _action = "delete_configuration"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "configuration": {
                "description": "List of configuration itemss to delete",
                "items": {"type": "string"},
                "type": "array",
            },
            "force": {
                "description": "If set to True then both new and running task configuration can be deleted. Otherwise only the new task ones. Default is False",
                "type": "boolean",
            },
            "task": {"description": "Task ID", "type": "string"},
        },
        "required": ["task", "configuration"],
        "type": "object",
    }

    def __init__(self, task: str, configuration: List[str], force: Optional[bool] = None, **kwargs: Any) -> None:
        super(DeleteConfigurationRequest, self).__init__(**kwargs)
        self.task = task
        self.configuration = configuration
        self.force = force

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

    @schema_property("configuration")
    def configuration(self) -> List[str]:
        return self._property_configuration

    @configuration.setter
    def configuration(self, value: List[str]) -> None:
        if value is None:
            self._property_configuration = None
            return
        self.assert_isinstance(value, "configuration", dict)
        self.assert_isinstance(value.keys(), "configuration_keys", six.string_types, is_array=True)
        self.assert_isinstance(
            value.values(),
            "configuration_values",
            (ConfigurationItem, dict),
            is_array=True,
        )
        value = dict(((k, ConfigurationItem(**v) if isinstance(v, dict) else v) for (k, v) in value.items()))
        self._property_configuration = value

    @schema_property("force")
    def force(self) -> Optional[bool]:
        return self._property_force

    @force.setter
    def force(self, value: Optional[bool]) -> None:
        if value is None:
            self._property_force = None
            return
        self.assert_isinstance(value, "force", (bool,))
        self._property_force = value


class DeleteConfigurationResponse(Response):
    """
    Response of tasks.delete_configuration endpoint.

    :param deleted: Indicates if the task was updated successfully
    :type deleted: int
    """

    _service = "tasks"
    _action = "delete_configuration"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "deleted": {
                "description": "Indicates if the task was updated successfully",
                "type": ["integer", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, deleted: Optional[int] = None, **kwargs: Any) -> None:
        super(DeleteConfigurationResponse, self).__init__(**kwargs)
        self.deleted = deleted

    @schema_property("deleted")
    def deleted(self) -> Optional[int]:
        return self._property_deleted

    @deleted.setter
    def deleted(self, value: Optional[int]) -> None:
        if value is None:
            self._property_deleted = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "deleted", six.integer_types)
        self._property_deleted = value


class DeleteHyperParamsRequest(Request):
    """
    Delete task hyper parameters

    :param task: Task ID
    :type task: str
    :param hyperparams: List of hyper parameters to delete. In case a parameter
        with an empty name is passed all the section will be deleted
    :type hyperparams: Sequence[ParamKey]
    :param force: If set to True then both new and running task hyper params can be
        deleted. Otherwise only the new task ones. Default is False
    :type force: bool
    """

    _service = "tasks"
    _action = "delete_hyper_params"
    _version = "2.13"
    _schema = {
        "definitions": {
            "param_key": {
                "properties": {
                    "name": {
                        "description": "Name of the parameter. If the name is ommitted then the corresponding operation is performed on the whole section",
                        "type": ["string", "null"],
                    },
                    "section": {
                        "description": "Section that the parameter belongs to",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            }
        },
        "properties": {
            "force": {
                "description": "If set to True then both new and running task hyper params can be deleted. Otherwise only the new task ones. Default is False",
                "type": "boolean",
            },
            "hyperparams": {
                "description": "List of hyper parameters to delete. In case a parameter with an empty name is passed all the section will be deleted",
                "items": {"$ref": "#/definitions/param_key"},
                "type": "array",
            },
            "task": {"description": "Task ID", "type": "string"},
        },
        "required": ["task", "hyperparams"],
        "type": "object",
    }

    def __init__(self, task: str, hyperparams: List[Any], force: Optional[bool] = None, **kwargs: Any) -> None:
        super(DeleteHyperParamsRequest, self).__init__(**kwargs)
        self.task = task
        self.hyperparams = hyperparams
        self.force = force

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

    @schema_property("hyperparams")
    def hyperparams(self) -> List[Any]:
        return self._property_hyperparams

    @hyperparams.setter
    def hyperparams(self, value: List[Any]) -> None:
        if value is None:
            self._property_hyperparams = None
            return
        self.assert_isinstance(value, "hyperparams", (ParamKey, dict), is_array=True)
        value = [ParamKey(**v) if isinstance(v, dict) else v for v in value]
        self._property_hyperparams = value

    @schema_property("force")
    def force(self) -> Optional[bool]:
        return self._property_force

    @force.setter
    def force(self, value: Optional[bool]) -> None:
        if value is None:
            self._property_force = None
            return
        self.assert_isinstance(value, "force", (bool,))
        self._property_force = value


class DeleteHyperParamsResponse(Response):
    """
    Response of tasks.delete_hyper_params endpoint.

    :param deleted: Indicates if the task was updated successfully
    :type deleted: int
    """

    _service = "tasks"
    _action = "delete_hyper_params"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "deleted": {
                "description": "Indicates if the task was updated successfully",
                "type": ["integer", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, deleted: Optional[int] = None, **kwargs: Any) -> None:
        super(DeleteHyperParamsResponse, self).__init__(**kwargs)
        self.deleted = deleted

    @schema_property("deleted")
    def deleted(self) -> Optional[int]:
        return self._property_deleted

    @deleted.setter
    def deleted(self, value: Optional[int]) -> None:
        if value is None:
            self._property_deleted = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "deleted", six.integer_types)
        self._property_deleted = value


class DeleteManyRequest(Request):
    """
    Delete tasks

    :param ids: Entities to move
    :type ids: Sequence[str]
    :param move_to_trash: Move task to trash instead of deleting it. For internal
        use only, tasks in the trash are not visible from the API and cannot be
        restored!
    :type move_to_trash: bool
    :param force: If not true, call fails if the task status is 'in_progress'
    :type force: bool
    :param return_file_urls: If set to 'true' then return the urls of the files
        that were uploaded by the tasks. Default value is 'false'
    :type return_file_urls: bool
    :param delete_output_models: If set to 'true' then delete output models of the
        tasks that are not referenced by other tasks. Default value is 'true'
    :type delete_output_models: bool
    """

    _service = "tasks"
    _action = "delete_many"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "delete_output_models": {
                "description": "If set to 'true' then delete output models of the tasks that are not referenced by other tasks. Default value is 'true'",
                "type": "boolean",
            },
            "force": {
                "default": False,
                "description": "If not true, call fails if the task status is 'in_progress'",
                "type": "boolean",
            },
            "ids": {
                "description": "Entities to move",
                "items": {"type": "string"},
                "type": "array",
            },
            "move_to_trash": {
                "default": False,
                "description": "Move task to trash instead of deleting it. For internal use only, tasks in the trash are not visible from the API and cannot be restored!",
                "type": "boolean",
            },
            "return_file_urls": {
                "description": "If set to 'true' then return the urls of the files that were uploaded by the tasks. Default value is 'false'",
                "type": "boolean",
            },
        },
        "required": ["ids"],
        "type": "object",
    }

    def __init__(
        self,
        ids: List[str],
        move_to_trash: Optional[bool] = False,
        force: Optional[bool] = False,
        return_file_urls: Optional[bool] = None,
        delete_output_models: Optional[bool] = None,
        **kwargs: Any
    ) -> None:
        super(DeleteManyRequest, self).__init__(**kwargs)
        self.ids = ids
        self.move_to_trash = move_to_trash
        self.force = force
        self.return_file_urls = return_file_urls
        self.delete_output_models = delete_output_models

    @schema_property("ids")
    def ids(self) -> List[str]:
        return self._property_ids

    @ids.setter
    def ids(self, value: List[str]) -> None:
        if value is None:
            self._property_ids = None
            return
        self.assert_isinstance(value, "ids", (list, tuple))
        self.assert_isinstance(value, "ids", six.string_types, is_array=True)
        self._property_ids = value

    @schema_property("move_to_trash")
    def move_to_trash(self) -> Optional[bool]:
        return self._property_move_to_trash

    @move_to_trash.setter
    def move_to_trash(self, value: Optional[bool]) -> None:
        if value is None:
            self._property_move_to_trash = None
            return
        self.assert_isinstance(value, "move_to_trash", (bool,))
        self._property_move_to_trash = value

    @schema_property("force")
    def force(self) -> Optional[bool]:
        return self._property_force

    @force.setter
    def force(self, value: Optional[bool]) -> None:
        if value is None:
            self._property_force = None
            return
        self.assert_isinstance(value, "force", (bool,))
        self._property_force = value

    @schema_property("return_file_urls")
    def return_file_urls(self) -> Optional[bool]:
        return self._property_return_file_urls

    @return_file_urls.setter
    def return_file_urls(self, value: Optional[bool]) -> None:
        if value is None:
            self._property_return_file_urls = None
            return
        self.assert_isinstance(value, "return_file_urls", (bool,))
        self._property_return_file_urls = value

    @schema_property("delete_output_models")
    def delete_output_models(self) -> Optional[bool]:
        return self._property_delete_output_models

    @delete_output_models.setter
    def delete_output_models(self, value: Optional[bool]) -> None:
        if value is None:
            self._property_delete_output_models = None
            return
        self.assert_isinstance(value, "delete_output_models", (bool,))
        self._property_delete_output_models = value


class DeleteManyResponse(Response):
    """
    Response of tasks.delete_many endpoint.

    :param deleted: Number of tasks deleted
    :type deleted: int
    :param updated_children: Number of child tasks whose parent property was
        updated
    :type updated_children: int
    :param updated_models: Number of models whose task property was updated
    :type updated_models: int
    :param deleted_models: Number of deleted output models
    :type deleted_models: int
    :param urls: The urls of the files that were uploaded by the tasks. Returned if
        the 'return_file_urls' was set to 'true'
    :type urls: TaskUrls
    """

    _service = "tasks"
    _action = "delete_many"
    _version = "2.13"
    _schema = {
        "definitions": {
            "task_urls": {
                "properties": {
                    "artifact_urls": {
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                    "event_urls": {
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                    "model_urls": {
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            }
        },
        "failures": {
            "item": {
                "error": {
                    "description": "Error info",
                    "properties": {
                        "codes": {"item": {"type": "integer"}, "type": "array"},
                        "data": {"additionalProperties": True, "type": "object"},
                        "msg": {"type": "string"},
                    },
                    "type": "object",
                },
                "id": {"description": "ID of the failed entity", "type": "string"},
                "type": "object",
            },
            "type": "array",
        },
        "properties": {
            "deleted": {
                "description": "Number of tasks deleted",
                "type": ["integer", "null"],
            },
            "deleted_models": {
                "description": "Number of deleted output models",
                "type": ["integer", "null"],
            },
            "updated_children": {
                "description": "Number of child tasks whose parent property was updated",
                "type": ["integer", "null"],
            },
            "updated_models": {
                "description": "Number of models whose task property was updated",
                "type": ["integer", "null"],
            },
            "urls": {
                "description": "The urls of the files that were uploaded by the tasks. Returned if the 'return_file_urls' was set to 'true'",
                "oneOf": [{"$ref": "#/definitions/task_urls"}, {"type": "null"}],
            },
        },
    }

    def __init__(
        self,
        deleted: Optional[int] = None,
        updated_children: Optional[int] = None,
        updated_models: Optional[int] = None,
        deleted_models: Optional[int] = None,
        urls: Any = None,
        **kwargs: Any
    ) -> None:
        super(DeleteManyResponse, self).__init__(**kwargs)
        self.deleted = deleted
        self.updated_children = updated_children
        self.updated_models = updated_models
        self.deleted_models = deleted_models
        self.urls = urls

    @schema_property("deleted")
    def deleted(self) -> Optional[int]:
        return self._property_deleted

    @deleted.setter
    def deleted(self, value: Optional[int]) -> None:
        if value is None:
            self._property_deleted = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "deleted", six.integer_types)
        self._property_deleted = value

    @schema_property("updated_children")
    def updated_children(self) -> Optional[int]:
        return self._property_updated_children

    @updated_children.setter
    def updated_children(self, value: Optional[int]) -> None:
        if value is None:
            self._property_updated_children = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "updated_children", six.integer_types)
        self._property_updated_children = value

    @schema_property("updated_models")
    def updated_models(self) -> Optional[int]:
        return self._property_updated_models

    @updated_models.setter
    def updated_models(self, value: Optional[int]) -> None:
        if value is None:
            self._property_updated_models = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "updated_models", six.integer_types)
        self._property_updated_models = value

    @schema_property("deleted_models")
    def deleted_models(self) -> Optional[int]:
        return self._property_deleted_models

    @deleted_models.setter
    def deleted_models(self, value: Optional[int]) -> None:
        if value is None:
            self._property_deleted_models = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "deleted_models", six.integer_types)
        self._property_deleted_models = value

    @schema_property("urls")
    def urls(self) -> Any:
        return self._property_urls

    @urls.setter
    def urls(self, value: Any) -> None:
        if value is None:
            self._property_urls = None
            return
        if isinstance(value, dict):
            value = TaskUrls.from_dict(value)
        else:
            self.assert_isinstance(value, "urls", TaskUrls)
        self._property_urls = value


class DeleteModelsRequest(Request):
    """
    Delete models from task

    :param task: ID of the task
    :type task: str
    :param models: The list of models to delete
    :type models: Sequence[dict]
    """

    _service = "tasks"
    _action = "delete_models"
    _version = "2.13"
    _schema = {
        "definitions": {"model_type_enum": {"enum": ["input", "output"], "type": "string"}},
        "properties": {
            "models": {
                "description": "The list of models to delete",
                "items": {
                    "properties": {
                        "name": {
                            "description": "The task model name",
                            "type": "string",
                        },
                        "type": {
                            "$ref": "#/definitions/model_type_enum",
                            "description": "The task model type",
                        },
                    },
                    "required": ["name", "type"],
                    "type": "object",
                },
                "type": "array",
            },
            "task": {"description": "ID of the task", "type": "string"},
        },
        "required": ["task", "models"],
        "type": "object",
    }

    def __init__(self, task: str, models: List[dict], **kwargs: Any) -> None:
        super(DeleteModelsRequest, self).__init__(**kwargs)
        self.task = task
        self.models = models

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

    @schema_property("models")
    def models(self) -> List[dict]:
        return self._property_models

    @models.setter
    def models(self, value: List[dict]) -> None:
        if value is None:
            self._property_models = None
            return
        self.assert_isinstance(value, "models", (list, tuple))
        self.assert_isinstance(value, "models", (dict,), is_array=True)
        self._property_models = value


class DeleteModelsResponse(Response):
    """
    Response of tasks.delete_models endpoint.

    :param updated: Number of tasks updated (0 or 1)
    :type updated: int
    """

    _service = "tasks"
    _action = "delete_models"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "updated": {
                "description": "Number of tasks updated (0 or 1)",
                "enum": [0, 1],
                "type": ["integer", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, updated: Optional[int] = None, **kwargs: Any) -> None:
        super(DeleteModelsResponse, self).__init__(**kwargs)
        self.updated = updated

    @schema_property("updated")
    def updated(self) -> Optional[int]:
        return self._property_updated

    @updated.setter
    def updated(self, value: Optional[int]) -> None:
        if value is None:
            self._property_updated = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "updated", six.integer_types)
        self._property_updated = value


class DequeueRequest(Request):
    """
    Remove a task from its queue.
            Fails if task status is not queued.

    :param task: Task ID
    :type task: str
    :param status_reason: Reason for status change
    :type status_reason: str
    :param status_message: Extra information regarding status change
    :type status_message: str
    """

    _service = "tasks"
    _action = "dequeue"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "status_message": {
                "description": "Extra information regarding status change",
                "type": "string",
            },
            "status_reason": {
                "description": "Reason for status change",
                "type": "string",
            },
            "task": {"description": "Task ID", "type": "string"},
        },
        "required": ["task"],
        "type": "object",
    }

    def __init__(
        self, task: str, status_reason: Optional[str] = None, status_message: Optional[str] = None, **kwargs: Any
    ) -> None:
        super(DequeueRequest, self).__init__(**kwargs)
        self.task = task
        self.status_reason = status_reason
        self.status_message = status_message

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

    @schema_property("status_reason")
    def status_reason(self) -> Optional[str]:
        return self._property_status_reason

    @status_reason.setter
    def status_reason(self, value: Optional[str]) -> None:
        if value is None:
            self._property_status_reason = None
            return
        self.assert_isinstance(value, "status_reason", six.string_types)
        self._property_status_reason = value

    @schema_property("status_message")
    def status_message(self) -> Optional[str]:
        return self._property_status_message

    @status_message.setter
    def status_message(self, value: Optional[str]) -> None:
        if value is None:
            self._property_status_message = None
            return
        self.assert_isinstance(value, "status_message", six.string_types)
        self._property_status_message = value


class DequeueResponse(Response):
    """
    Response of tasks.dequeue endpoint.

    :param updated: Number of tasks updated (0 or 1)
    :type updated: int
    :param fields: Updated fields names and values
    :type fields: dict
    :param dequeued: Number of tasks dequeued (0 or 1)
    :type dequeued: int
    """

    _service = "tasks"
    _action = "dequeue"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "dequeued": {
                "description": "Number of tasks dequeued (0 or 1)",
                "enum": [0, 1],
                "type": ["integer", "null"],
            },
            "fields": {
                "additionalProperties": True,
                "description": "Updated fields names and values",
                "type": ["object", "null"],
            },
            "updated": {
                "description": "Number of tasks updated (0 or 1)",
                "enum": [0, 1],
                "type": ["integer", "null"],
            },
        },
        "type": "object",
    }

    def __init__(
        self,
        updated: Optional[int] = None,
        fields: Optional[dict] = None,
        dequeued: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        super(DequeueResponse, self).__init__(**kwargs)
        self.updated = updated
        self.fields = fields
        self.dequeued = dequeued

    @schema_property("updated")
    def updated(self) -> Optional[int]:
        return self._property_updated

    @updated.setter
    def updated(self, value: Optional[int]) -> None:
        if value is None:
            self._property_updated = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "updated", six.integer_types)
        self._property_updated = value

    @schema_property("fields")
    def fields(self) -> Optional[dict]:
        return self._property_fields

    @fields.setter
    def fields(self, value: Optional[dict]) -> None:
        if value is None:
            self._property_fields = None
            return
        self.assert_isinstance(value, "fields", (dict,))
        self._property_fields = value

    @schema_property("dequeued")
    def dequeued(self) -> Optional[int]:
        return self._property_dequeued

    @dequeued.setter
    def dequeued(self, value: Optional[int]) -> None:
        if value is None:
            self._property_dequeued = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "dequeued", six.integer_types)
        self._property_dequeued = value


class EditRequest(Request):
    """
    Edit task's details.

    :param task: ID of the task
    :type task: str
    :param force: If not true, call fails if the task status is not 'created'
    :type force: bool
    :param name: Task name Unique within the company.
    :type name: str
    :param tags: User-defined tags list
    :type tags: Sequence[str]
    :param system_tags: System tags list. This field is reserved for system use,
        please don't use it.
    :type system_tags: Sequence[str]
    :param type: Type of task
    :type type: TaskTypeEnum
    :param comment: Free text comment
    :type comment: str
    :param parent: Parent task id Must be a completed task.
    :type parent: str
    :param project: Project ID of the project to which this task is assigned Must
        exist[ab]
    :type project: str
    :param output_dest: Output storage id Must be a reference to an existing
        storage.
    :type output_dest: str
    :param execution: Task execution params
    :type execution: Execution
    :param script: Script info
    :type script: Script
    :param hyperparams: Task hyper params per section
    :type hyperparams: dict
    :param configuration: Task configuration params
    :type configuration: dict
    :param models: Task models
    :type models: TaskModels
    :param container: Docker container parameters
    :type container: dict
    :param runtime: Task runtime mapping
    :type runtime: dict
    """

    _service = "tasks"
    _action = "edit"
    _version = "2.13"
    _schema = {
        "definitions": {
            "artifact": {
                "properties": {
                    "content_size": {
                        "description": "Raw data length in bytes",
                        "type": "integer",
                    },
                    "display_data": {
                        "description": "User-defined list of key/value pairs, sorted",
                        "items": {"items": {"type": "string"}, "type": "array"},
                        "type": "array",
                    },
                    "hash": {
                        "description": "Hash of entire raw data",
                        "type": "string",
                    },
                    "key": {"description": "Entry key", "type": "string"},
                    "mode": {
                        "$ref": "#/definitions/artifact_mode_enum",
                        "description": "System defined input/output indication",
                    },
                    "timestamp": {
                        "description": "Epoch time when artifact was created",
                        "type": "integer",
                    },
                    "type": {"description": "System defined type", "type": "string"},
                    "type_data": {
                        "$ref": "#/definitions/artifact_type_data",
                        "description": "Additional fields defined by the system",
                    },
                    "uri": {"description": "Raw data location", "type": "string"},
                },
                "required": ["key", "type"],
                "type": "object",
            },
            "artifact_mode_enum": {
                "default": "output",
                "enum": ["input", "output"],
                "type": "string",
            },
            "artifact_type_data": {
                "properties": {
                    "content_type": {
                        "description": "System defined raw data content type",
                        "type": ["string", "null"],
                    },
                    "data_hash": {
                        "description": "Hash of raw data, without any headers or descriptive parts",
                        "type": ["string", "null"],
                    },
                    "preview": {
                        "description": "Description or textual data",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "configuration_item": {
                "properties": {
                    "description": {
                        "description": "The parameter description. Optional",
                        "type": ["string", "null"],
                    },
                    "name": {
                        "description": "Name of the parameter. Should be unique",
                        "type": ["string", "null"],
                    },
                    "type": {
                        "description": "Type of the parameter. Optional",
                        "type": ["string", "null"],
                    },
                    "value": {
                        "description": "Value of the parameter",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "execution": {
                "properties": {
                    "artifacts": {
                        "description": "Task artifacts",
                        "items": {"$ref": "#/definitions/artifact"},
                        "type": ["array", "null"],
                    },
                    "framework": {
                        "description": "Framework related to the task. Case insensitive. Mandatory for Training tasks. ",
                        "type": ["string", "null"],
                    },
                    "model_desc": {
                        "additionalProperties": True,
                        "description": "Json object representing the Model descriptors",
                        "type": ["object", "null"],
                    },
                    "model_labels": {
                        "additionalProperties": {"type": "integer"},
                        "description": "Json object representing the ids of the labels in the model.\n                The keys are the layers' names and the values are the IDs.\n                Not applicable for Register (Import) tasks.\n                Mandatory for Training tasks",
                        "type": ["object", "null"],
                    },
                    "parameters": {
                        "additionalProperties": True,
                        "description": "Json object containing the Task parameters",
                        "type": ["object", "null"],
                    },
                    "queue": {
                        "description": "Queue ID where task was queued.",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "params_item": {
                "properties": {
                    "description": {
                        "description": "The parameter description. Optional",
                        "type": ["string", "null"],
                    },
                    "name": {
                        "description": "Name of the parameter. The combination of section and name should be unique",
                        "type": ["string", "null"],
                    },
                    "section": {
                        "description": "Section that the parameter belongs to",
                        "type": ["string", "null"],
                    },
                    "type": {
                        "description": "Type of the parameter. Optional",
                        "type": ["string", "null"],
                    },
                    "value": {
                        "description": "Value of the parameter",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "script": {
                "properties": {
                    "binary": {
                        "default": "python",
                        "description": "Binary to use when running the script",
                        "type": ["string", "null"],
                    },
                    "branch": {
                        "description": "Repository branch id If not provided and tag not provided, default repository branch is used.",
                        "type": ["string", "null"],
                    },
                    "diff": {
                        "description": "Uncommitted changes found in the repository when task was run",
                        "type": ["string", "null"],
                    },
                    "entry_point": {
                        "description": "Path to execute within the repository",
                        "type": ["string", "null"],
                    },
                    "repository": {
                        "description": "Name of the repository where the script is located",
                        "type": ["string", "null"],
                    },
                    "requirements": {
                        "description": "A JSON object containing requirements strings by key",
                        "type": ["object", "null"],
                    },
                    "tag": {
                        "description": "Repository tag",
                        "type": ["string", "null"],
                    },
                    "version_num": {
                        "description": "Version (changeset) number. Optional (default is head version) Unused if tag is provided.",
                        "type": ["string", "null"],
                    },
                    "working_dir": {
                        "description": "Path to the folder from which to run the script Default - root folder of repository",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "section_params": {
                "additionalProperties": {"$ref": "#/definitions/params_item"},
                "description": "Task section params",
                "type": "object",
            },
            "task_model_item": {
                "properties": {
                    "model": {"description": "The model ID", "type": "string"},
                    "name": {"description": "The task model name", "type": "string"},
                },
                "required": ["name", "model"],
                "type": "object",
            },
            "task_models": {
                "properties": {
                    "input": {
                        "description": "The list of task input models",
                        "items": {"$ref": "#/definitions/task_model_item"},
                        "type": ["array", "null"],
                    },
                    "output": {
                        "description": "The list of task output models",
                        "items": {"$ref": "#/definitions/task_model_item"},
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            },
            "task_type_enum": {
                "enum": [
                    "training",
                    "testing",
                    "inference",
                    "data_processing",
                    "application",
                    "monitor",
                    "controller",
                    "optimizer",
                    "service",
                    "qc",
                    "custom",
                ],
                "type": "string",
            },
        },
        "properties": {
            "comment": {"description": "Free text comment ", "type": "string"},
            "configuration": {
                "additionalProperties": {"$ref": "#/definitions/configuration_item"},
                "description": "Task configuration params",
                "type": "object",
            },
            "container": {
                "type": "object",
                "description": "Docker container parameters",
                "additionalProperties": {"type": ["string", "null"]},
            },
            "execution": {
                "$ref": "#/definitions/execution",
                "description": "Task execution params",
            },
            "force": {
                "default": False,
                "description": "If not true, call fails if the task status is not 'created'",
                "type": "boolean",
            },
            "hyperparams": {
                "additionalProperties": {"$ref": "#/definitions/section_params"},
                "description": "Task hyper params per section",
                "type": "object",
            },
            "models": {
                "$ref": "#/definitions/task_models",
                "description": "Task models",
            },
            "name": {
                "description": "Task name Unique within the company.",
                "type": "string",
            },
            "output_dest": {
                "description": "Output storage id Must be a reference to an existing storage.",
                "type": "string",
            },
            "parent": {
                "description": "Parent task id Must be a completed task.",
                "type": "string",
            },
            "project": {
                "description": "Project ID of the project to which this task is assigned Must exist[ab]",
                "type": "string",
            },
            "runtime": {
                "description": "Task runtime mapping",
                "type": ["object", "null"],
                "additionalProperties": True,
            },
            "script": {"$ref": "#/definitions/script", "description": "Script info"},
            "system_tags": {
                "description": "System tags list. This field is reserved for system use, please don't use it.",
                "items": {"type": "string"},
                "type": "array",
            },
            "tags": {
                "description": "User-defined tags list",
                "items": {"type": "string"},
                "type": "array",
            },
            "task": {"description": "ID of the task", "type": "string"},
            "type": {
                "$ref": "#/definitions/task_type_enum",
                "description": "Type of task",
            },
        },
        "required": ["task"],
        "type": "object",
    }

    def __init__(
        self,
        task: str,
        force: Optional[bool] = False,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        system_tags: Optional[List[str]] = None,
        type: Any = None,
        comment: Optional[str] = None,
        parent: Optional[str] = None,
        project: Optional[str] = None,
        output_dest: Optional[str] = None,
        execution: Any = None,
        script: Any = None,
        hyperparams: Optional[dict] = None,
        configuration: Optional[dict] = None,
        models: Any = None,
        container: Optional[dict] = None,
        runtime: Optional[dict] = None,
        **kwargs: Any
    ) -> None:
        super(EditRequest, self).__init__(**kwargs)
        self.task = task
        self.force = force
        self.name = name
        self.tags = tags
        self.system_tags = system_tags
        self.type = type
        self.comment = comment
        self.parent = parent
        self.project = project
        self.output_dest = output_dest
        self.execution = execution
        self.script = script
        self.hyperparams = hyperparams
        self.configuration = configuration
        self.models = models
        self.container = container
        self.runtime = runtime

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

    @schema_property("force")
    def force(self) -> Optional[bool]:
        return self._property_force

    @force.setter
    def force(self, value: Optional[bool]) -> None:
        if value is None:
            self._property_force = None
            return
        self.assert_isinstance(value, "force", (bool,))
        self._property_force = value

    @schema_property("name")
    def name(self) -> Optional[str]:
        return self._property_name

    @name.setter
    def name(self, value: Optional[str]) -> None:
        if value is None:
            self._property_name = None
            return
        self.assert_isinstance(value, "name", six.string_types)
        self._property_name = value

    @schema_property("tags")
    def tags(self) -> Optional[List[str]]:
        return self._property_tags

    @tags.setter
    def tags(self, value: Optional[List[str]]) -> None:
        if value is None:
            self._property_tags = None
            return
        self.assert_isinstance(value, "tags", (list, tuple))
        self.assert_isinstance(value, "tags", six.string_types, is_array=True)
        self._property_tags = value

    @schema_property("system_tags")
    def system_tags(self) -> Optional[List[str]]:
        return self._property_system_tags

    @system_tags.setter
    def system_tags(self, value: Optional[List[str]]) -> None:
        if value is None:
            self._property_system_tags = None
            return
        self.assert_isinstance(value, "system_tags", (list, tuple))
        self.assert_isinstance(value, "system_tags", six.string_types, is_array=True)
        self._property_system_tags = value

    @schema_property("type")
    def type(self) -> Any:
        return self._property_type

    @type.setter
    def type(self, value: Any) -> None:
        if value is None:
            self._property_type = None
            return
        if isinstance(value, six.string_types):
            try:
                value = TaskTypeEnum(value)
            except ValueError:
                pass
        else:
            self.assert_isinstance(value, "type", enum.Enum)
        self._property_type = value

    @schema_property("comment")
    def comment(self) -> Optional[str]:
        return self._property_comment

    @comment.setter
    def comment(self, value: Optional[str]) -> None:
        if value is None:
            self._property_comment = None
            return
        self.assert_isinstance(value, "comment", six.string_types)
        self._property_comment = value

    @schema_property("parent")
    def parent(self) -> Optional[str]:
        return self._property_parent

    @parent.setter
    def parent(self, value: Optional[str]) -> None:
        if value is None:
            self._property_parent = None
            return
        self.assert_isinstance(value, "parent", six.string_types)
        self._property_parent = value

    @schema_property("project")
    def project(self) -> Optional[str]:
        return self._property_project

    @project.setter
    def project(self, value: Optional[str]) -> None:
        if value is None:
            self._property_project = None
            return
        self.assert_isinstance(value, "project", six.string_types)
        self._property_project = value

    @schema_property("output_dest")
    def output_dest(self) -> Optional[str]:
        return self._property_output_dest

    @output_dest.setter
    def output_dest(self, value: Optional[str]) -> None:
        if value is None:
            self._property_output_dest = None
            return
        self.assert_isinstance(value, "output_dest", six.string_types)
        self._property_output_dest = value

    @schema_property("execution")
    def execution(self) -> Any:
        return self._property_execution

    @execution.setter
    def execution(self, value: Any) -> None:
        if value is None:
            self._property_execution = None
            return
        if isinstance(value, dict):
            value = Execution.from_dict(value)
        else:
            self.assert_isinstance(value, "execution", Execution)
        self._property_execution = value

    @schema_property("hyperparams")
    def hyperparams(self) -> Optional[dict]:
        return self._property_hyperparams

    @hyperparams.setter
    def hyperparams(self, value: Optional[dict]) -> None:
        if value is None:
            self._property_hyperparams = None
            return
        self.assert_isinstance(value, "hyperparams", dict)
        self.assert_isinstance(value.keys(), "hyperparams_keys", six.string_types, is_array=True)
        self.assert_isinstance(value.values(), "hyperparams_values", (SectionParams, dict), is_array=True)
        value = dict(((k, SectionParams(**v) if isinstance(v, dict) else v) for (k, v) in value.items()))
        self._property_hyperparams = value

    @schema_property("configuration")
    def configuration(self) -> Optional[dict]:
        return self._property_configuration

    @configuration.setter
    def configuration(self, value: Optional[dict]) -> None:
        if value is None:
            self._property_configuration = None
            return
        self.assert_isinstance(value, "configuration", dict)
        self.assert_isinstance(value.keys(), "configuration_keys", six.string_types, is_array=True)
        self.assert_isinstance(
            value.values(),
            "configuration_values",
            (ConfigurationItem, dict),
            is_array=True,
        )
        value = dict(((k, ConfigurationItem(**v) if isinstance(v, dict) else v) for (k, v) in value.items()))
        self._property_configuration = value

    @schema_property("script")
    def script(self) -> Any:
        return self._property_script

    @script.setter
    def script(self, value: Any) -> None:
        if value is None:
            self._property_script = None
            return
        if isinstance(value, dict):
            value = Script.from_dict(value)
        else:
            self.assert_isinstance(value, "script", Script)
        self._property_script = value

    @schema_property("models")
    def models(self) -> Any:
        return self._property_models

    @models.setter
    def models(self, value: Any) -> None:
        if value is None:
            self._property_models = None
            return
        if isinstance(value, dict):
            value = TaskModels.from_dict(value)
        else:
            self.assert_isinstance(value, "models", TaskModels)
        self._property_models = value

    @schema_property("container")
    def container(self) -> Optional[dict]:
        return self._property_container

    @container.setter
    def container(self, value: Optional[dict]) -> None:
        if value is None:
            self._property_container = None
            return
        self.assert_isinstance(value, "container", dict)
        self._property_container = value

    @schema_property("runtime")
    def runtime(self) -> Optional[dict]:
        return self._property_runtime

    @runtime.setter
    def runtime(self, value: Optional[dict]) -> None:
        if value is None:
            self._property_runtime = None
            return
        self.assert_isinstance(value, "runtime", dict)
        self._property_runtime = value


class EditResponse(Response):
    """
    Response of tasks.edit endpoint.

    :param updated: Number of tasks updated (0 or 1)
    :type updated: int
    :param fields: Updated fields names and values
    :type fields: dict
    """

    _service = "tasks"
    _action = "edit"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "fields": {
                "additionalProperties": True,
                "description": "Updated fields names and values",
                "type": ["object", "null"],
            },
            "updated": {
                "description": "Number of tasks updated (0 or 1)",
                "enum": [0, 1],
                "type": ["integer", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, updated: Optional[int] = None, fields: Optional[dict] = None, **kwargs: Any) -> None:
        super(EditResponse, self).__init__(**kwargs)
        self.updated = updated
        self.fields = fields

    @schema_property("updated")
    def updated(self) -> Optional[int]:
        return self._property_updated

    @updated.setter
    def updated(self, value: Optional[int]) -> None:
        if value is None:
            self._property_updated = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "updated", six.integer_types)
        self._property_updated = value

    @schema_property("fields")
    def fields(self) -> Optional[dict]:
        return self._property_fields

    @fields.setter
    def fields(self, value: Optional[dict]) -> None:
        if value is None:
            self._property_fields = None
            return
        self.assert_isinstance(value, "fields", (dict,))
        self._property_fields = value


class EditConfigurationRequest(Request):
    """
    Add or update task configuration

    :param task: Task ID
    :type task: str
    :param configuration: Task configuration items. The new ones will be added and
        the already existing ones will be updated
    :type configuration: Sequence[ConfigurationItem]
    :param replace_configuration: If set then the all the configuration items will
        be replaced with the provided ones. Otherwise only the provided configuration
        items will be updated or added
    :type replace_configuration: bool
    :param force: If set to True then both new and running task configuration can
        be edited. Otherwise only the new task ones. Default is False
    :type force: bool
    """

    _service = "tasks"
    _action = "edit_configuration"
    _version = "2.13"
    _schema = {
        "definitions": {
            "configuration_item": {
                "properties": {
                    "description": {
                        "description": "The parameter description. Optional",
                        "type": ["string", "null"],
                    },
                    "name": {
                        "description": "Name of the parameter. Should be unique",
                        "type": ["string", "null"],
                    },
                    "type": {
                        "description": "Type of the parameter. Optional",
                        "type": ["string", "null"],
                    },
                    "value": {
                        "description": "Value of the parameter",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            }
        },
        "properties": {
            "configuration": {
                "description": "Task configuration items. The new ones will be added and the already existing ones will be updated",
                "items": {"$ref": "#/definitions/configuration_item"},
                "type": "array",
            },
            "force": {
                "description": "If set to True then both new and running task configuration can be edited. Otherwise only the new task ones. Default is False",
                "type": "boolean",
            },
            "replace_configuration": {
                "description": "If set then the all the configuration items will be replaced with the provided ones. Otherwise only the provided configuration items will be updated or added",
                "type": "boolean",
            },
            "task": {"description": "Task ID", "type": "string"},
        },
        "required": ["task", "configuration"],
        "type": "object",
    }

    def __init__(
        self,
        task: str,
        configuration: List[Any],
        replace_configuration: Optional[bool] = None,
        force: Optional[bool] = None,
        **kwargs: Any
    ) -> None:
        super(EditConfigurationRequest, self).__init__(**kwargs)
        self.task = task
        self.configuration = configuration
        self.replace_configuration = replace_configuration
        self.force = force

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

    @schema_property("configuration")
    def configuration(self) -> List[Any]:
        return self._property_configuration

    @configuration.setter
    def configuration(self, value: List[Any]) -> None:
        if value is None:
            self._property_configuration = None
            return
        self.assert_isinstance(value, "configuration", (dict, ConfigurationItem), is_array=True)
        value = [ConfigurationItem(**v) if isinstance(v, dict) else v for v in value]
        self._property_configuration = value

    @schema_property("replace_configuration")
    def replace_configuration(self) -> Optional[bool]:
        return self._property_replace_configuration

    @replace_configuration.setter
    def replace_configuration(self, value: Optional[bool]) -> None:
        if value is None:
            self._property_replace_configuration = None
            return
        self.assert_isinstance(value, "replace_configuration", (bool,))
        self._property_replace_configuration = value

    @schema_property("force")
    def force(self) -> Optional[bool]:
        return self._property_force

    @force.setter
    def force(self, value: Optional[bool]) -> None:
        if value is None:
            self._property_force = None
            return
        self.assert_isinstance(value, "force", (bool,))
        self._property_force = value


class EditConfigurationResponse(Response):
    """
    Response of tasks.edit_configuration endpoint.

    :param updated: Indicates if the task was updated successfully
    :type updated: int
    """

    _service = "tasks"
    _action = "edit_configuration"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "updated": {
                "description": "Indicates if the task was updated successfully",
                "type": ["integer", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, updated: Optional[int] = None, **kwargs: Any) -> None:
        super(EditConfigurationResponse, self).__init__(**kwargs)
        self.updated = updated

    @schema_property("updated")
    def updated(self) -> Optional[int]:
        return self._property_updated

    @updated.setter
    def updated(self, value: Optional[int]) -> None:
        if value is None:
            self._property_updated = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "updated", six.integer_types)
        self._property_updated = value


class EditHyperParamsRequest(Request):
    """
    Add or update task hyper parameters

    :param task: Task ID
    :type task: str
    :param hyperparams: Task hyper parameters. The new ones will be added and the
        already existing ones will be updated
    :type hyperparams: Sequence[ParamsItem]
    :param replace_hyperparams: Can be set to one of the following: 'all' - all the
        hyper parameters will be replaced with the provided ones 'section' - the
        sections that present in the new parameters will be replaced with the provided
        parameters 'none' (the default value) - only the specific parameters will be
        updated or added
    :type replace_hyperparams: ReplaceHyperparamsEnum
    :param force: If set to True then both new and running task hyper params can be
        edited. Otherwise only the new task ones. Default is False
    :type force: bool
    """

    _service = "tasks"
    _action = "edit_hyper_params"
    _version = "2.13"
    _schema = {
        "definitions": {
            "params_item": {
                "properties": {
                    "description": {
                        "description": "The parameter description. Optional",
                        "type": ["string", "null"],
                    },
                    "name": {
                        "description": "Name of the parameter. The combination of section and name should be unique",
                        "type": ["string", "null"],
                    },
                    "section": {
                        "description": "Section that the parameter belongs to",
                        "type": ["string", "null"],
                    },
                    "type": {
                        "description": "Type of the parameter. Optional",
                        "type": ["string", "null"],
                    },
                    "value": {
                        "description": "Value of the parameter",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "replace_hyperparams_enum": {
                "enum": ["none", "section", "all"],
                "type": "string",
            },
        },
        "properties": {
            "force": {
                "description": "If set to True then both new and running task hyper params can be edited. Otherwise only the new task ones. Default is False",
                "type": "boolean",
            },
            "hyperparams": {
                "description": "Task hyper parameters. The new ones will be added and the already existing ones will be updated",
                "items": {"$ref": "#/definitions/params_item"},
                "type": "array",
            },
            "replace_hyperparams": {
                "$ref": "#/definitions/replace_hyperparams_enum",
                "description": "Can be set to one of the following:\n                     'all' - all the hyper parameters will be replaced with the provided ones\n                     'section' - the sections that present in the new parameters will be replaced with the provided parameters\n                     'none' (the default value) - only the specific parameters will be updated or added",
            },
            "task": {"description": "Task ID", "type": "string"},
        },
        "required": ["task", "hyperparams"],
        "type": "object",
    }

    def __init__(
        self,
        task: str,
        hyperparams: List[Any],
        replace_hyperparams: Any = None,
        force: Optional[bool] = None,
        **kwargs: Any
    ) -> None:
        super(EditHyperParamsRequest, self).__init__(**kwargs)
        self.task = task
        self.hyperparams = hyperparams
        self.replace_hyperparams = replace_hyperparams
        self.force = force

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

    @schema_property("hyperparams")
    def hyperparams(self) -> List[Any]:
        return self._property_hyperparams

    @hyperparams.setter
    def hyperparams(self, value: List[Any]) -> None:
        if value is None:
            self._property_hyperparams = None
            return
        self.assert_isinstance(value, "hyperparams", (dict, ParamsItem), is_array=True)
        value = [ParamsItem(**v) if isinstance(v, dict) else v for v in value]
        self._property_hyperparams = value

    @schema_property("replace_hyperparams")
    def replace_hyperparams(self) -> Any:
        return self._property_replace_hyperparams

    @replace_hyperparams.setter
    def replace_hyperparams(self, value: Any) -> None:
        if value is None:
            self._property_replace_hyperparams = None
            return
        if isinstance(value, six.string_types):
            try:
                value = ReplaceHyperparamsEnum(value)
            except ValueError:
                pass
        else:
            self.assert_isinstance(value, "replace_hyperparams", enum.Enum)
        self._property_replace_hyperparams = value

    @schema_property("force")
    def force(self) -> Optional[bool]:
        return self._property_force

    @force.setter
    def force(self, value: Optional[bool]) -> None:
        if value is None:
            self._property_force = None
            return
        self.assert_isinstance(value, "force", (bool,))
        self._property_force = value


class EditHyperParamsResponse(Response):
    """
    Response of tasks.edit_hyper_params endpoint.

    :param updated: Indicates if the task was updated successfully
    :type updated: int
    """

    _service = "tasks"
    _action = "edit_hyper_params"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "updated": {
                "description": "Indicates if the task was updated successfully",
                "type": ["integer", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, updated: Optional[int] = None, **kwargs: Any) -> None:
        super(EditHyperParamsResponse, self).__init__(**kwargs)
        self.updated = updated

    @schema_property("updated")
    def updated(self) -> Optional[int]:
        return self._property_updated

    @updated.setter
    def updated(self, value: Optional[int]) -> None:
        if value is None:
            self._property_updated = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "updated", six.integer_types)
        self._property_updated = value


class EnqueueRequest(Request):
    """
    Adds a task into a queue.

    Fails if task state is not 'created'.

    Fails if the following parameters in the task were not filled:

    * execution.script.repository

    * execution.script.entrypoint


    :param queue: Queue id. If not provided, task is added to the default queue.
    :type queue: str
    :param task: Task ID
    :type task: str
    :param status_reason: Reason for status change
    :type status_reason: str
    :param status_message: Extra information regarding status change
    :type status_message: str
    """

    _service = "tasks"
    _action = "enqueue"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "queue": {
                "description": "Queue id. If not provided, task is added to the default queue.",
                "type": ["string", "null"],
            },
            "status_message": {
                "description": "Extra information regarding status change",
                "type": "string",
            },
            "status_reason": {
                "description": "Reason for status change",
                "type": "string",
            },
            "task": {"description": "Task ID", "type": "string"},
        },
        "required": ["task"],
        "type": "object",
    }

    def __init__(
        self,
        task: str,
        queue: Optional[str] = None,
        status_reason: Optional[str] = None,
        status_message: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super(EnqueueRequest, self).__init__(**kwargs)
        self.queue = queue
        self.task = task
        self.status_reason = status_reason
        self.status_message = status_message

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

    @schema_property("status_reason")
    def status_reason(self) -> Optional[str]:
        return self._property_status_reason

    @status_reason.setter
    def status_reason(self, value: Optional[str]) -> None:
        if value is None:
            self._property_status_reason = None
            return
        self.assert_isinstance(value, "status_reason", six.string_types)
        self._property_status_reason = value

    @schema_property("status_message")
    def status_message(self) -> Optional[str]:
        return self._property_status_message

    @status_message.setter
    def status_message(self, value: Optional[str]) -> None:
        if value is None:
            self._property_status_message = None
            return
        self.assert_isinstance(value, "status_message", six.string_types)
        self._property_status_message = value


class EnqueueResponse(Response):
    """
    Response of tasks.enqueue endpoint.

    :param updated: Number of tasks updated (0 or 1)
    :type updated: int
    :param fields: Updated fields names and values
    :type fields: dict
    :param queued: Number of tasks queued (0 or 1)
    :type queued: int
    """

    _service = "tasks"
    _action = "enqueue"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "fields": {
                "additionalProperties": True,
                "description": "Updated fields names and values",
                "type": ["object", "null"],
            },
            "queued": {
                "description": "Number of tasks queued (0 or 1)",
                "enum": [0, 1],
                "type": ["integer", "null"],
            },
            "updated": {
                "description": "Number of tasks updated (0 or 1)",
                "enum": [0, 1],
                "type": ["integer", "null"],
            },
        },
        "type": "object",
    }

    def __init__(
        self, updated: Optional[int] = None, fields: Optional[dict] = None, queued: Optional[int] = None, **kwargs: Any
    ) -> None:
        super(EnqueueResponse, self).__init__(**kwargs)
        self.updated = updated
        self.fields = fields
        self.queued = queued

    @schema_property("updated")
    def updated(self) -> Optional[int]:
        return self._property_updated

    @updated.setter
    def updated(self, value: Optional[int]) -> None:
        if value is None:
            self._property_updated = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "updated", six.integer_types)
        self._property_updated = value

    @schema_property("fields")
    def fields(self) -> Optional[dict]:
        return self._property_fields

    @fields.setter
    def fields(self, value: Optional[dict]) -> None:
        if value is None:
            self._property_fields = None
            return
        self.assert_isinstance(value, "fields", (dict,))
        self._property_fields = value

    @schema_property("queued")
    def queued(self) -> Optional[int]:
        return self._property_queued

    @queued.setter
    def queued(self, value: Optional[int]) -> None:
        if value is None:
            self._property_queued = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "queued", six.integer_types)
        self._property_queued = value


class EnqueueManyRequest(Request):
    """
    Enqueue tasks

    :param ids: Entities to move
    :type ids: Sequence[str]
    :param status_reason: Reason for status change
    :type status_reason: str
    :param status_message: Extra information regarding status change
    :type status_message: str
    """

    _service = "tasks"
    _action = "enqueue_many"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "ids": {
                "description": "Entities to move",
                "items": {"type": "string"},
                "type": "array",
            },
            "status_message": {
                "description": "Extra information regarding status change",
                "type": "string",
            },
            "status_reason": {
                "description": "Reason for status change",
                "type": "string",
            },
        },
        "required": ["ids"],
        "type": "object",
    }

    def __init__(
        self, ids: List[str], status_reason: Optional[str] = None, status_message: Optional[str] = None, **kwargs: Any
    ) -> None:
        super(EnqueueManyRequest, self).__init__(**kwargs)
        self.ids = ids
        self.status_reason = status_reason
        self.status_message = status_message

    @schema_property("ids")
    def ids(self) -> List[str]:
        return self._property_ids

    @ids.setter
    def ids(self, value: List[str]) -> None:
        if value is None:
            self._property_ids = None
            return
        self.assert_isinstance(value, "ids", (list, tuple))
        self.assert_isinstance(value, "ids", six.string_types, is_array=True)
        self._property_ids = value

    @schema_property("status_reason")
    def status_reason(self) -> Optional[str]:
        return self._property_status_reason

    @status_reason.setter
    def status_reason(self, value: Optional[str]) -> None:
        if value is None:
            self._property_status_reason = None
            return
        self.assert_isinstance(value, "status_reason", six.string_types)
        self._property_status_reason = value

    @schema_property("status_message")
    def status_message(self) -> Optional[str]:
        return self._property_status_message

    @status_message.setter
    def status_message(self, value: Optional[str]) -> None:
        if value is None:
            self._property_status_message = None
            return
        self.assert_isinstance(value, "status_message", six.string_types)
        self._property_status_message = value


class EnqueueManyResponse(Response):
    """
    Response of tasks.enqueue_many endpoint.

    :param enqueued: Number of tasks enqueued
    :type enqueued: int
    """

    _service = "tasks"
    _action = "enqueue_many"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "failures": {
            "item": {
                "error": {
                    "description": "Error info",
                    "properties": {
                        "codes": {"item": {"type": "integer"}, "type": "array"},
                        "data": {"additionalProperties": True, "type": "object"},
                        "msg": {"type": "string"},
                    },
                    "type": "object",
                },
                "id": {"description": "ID of the failed entity", "type": "string"},
                "type": "object",
            },
            "type": "array",
        },
        "properties": {
            "enqueued": {
                "description": "Number of tasks enqueued",
                "type": ["integer", "null"],
            }
        },
    }

    def __init__(self, enqueued: Optional[int] = None, **kwargs: Any) -> None:
        super(EnqueueManyResponse, self).__init__(**kwargs)
        self.enqueued = enqueued

    @schema_property("enqueued")
    def enqueued(self) -> Optional[int]:
        return self._property_enqueued

    @enqueued.setter
    def enqueued(self, value: Optional[int]) -> None:
        if value is None:
            self._property_enqueued = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "enqueued", six.integer_types)
        self._property_enqueued = value


class FailedRequest(Request):
    """
    Indicates that task has failed

    :param force: Allows forcing state change even if transition is not supported
    :type force: bool
    :param task: Task ID
    :type task: str
    :param status_reason: Reason for status change
    :type status_reason: str
    :param status_message: Extra information regarding status change
    :type status_message: str
    """

    _service = "tasks"
    _action = "failed"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "force": {
                "default": False,
                "description": "Allows forcing state change even if transition is not supported",
                "type": ["boolean", "null"],
            },
            "status_message": {
                "description": "Extra information regarding status change",
                "type": "string",
            },
            "status_reason": {
                "description": "Reason for status change",
                "type": "string",
            },
            "task": {"description": "Task ID", "type": "string"},
        },
        "required": ["task"],
        "type": "object",
    }

    def __init__(
        self,
        task: str,
        force: Optional[bool] = False,
        status_reason: Optional[str] = None,
        status_message: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super(FailedRequest, self).__init__(**kwargs)
        self.force = force
        self.task = task
        self.status_reason = status_reason
        self.status_message = status_message

    @schema_property("force")
    def force(self) -> Optional[bool]:
        return self._property_force

    @force.setter
    def force(self, value: Optional[bool]) -> None:
        if value is None:
            self._property_force = None
            return
        self.assert_isinstance(value, "force", (bool,))
        self._property_force = value

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

    @schema_property("status_reason")
    def status_reason(self) -> Optional[str]:
        return self._property_status_reason

    @status_reason.setter
    def status_reason(self, value: Optional[str]) -> None:
        if value is None:
            self._property_status_reason = None
            return
        self.assert_isinstance(value, "status_reason", six.string_types)
        self._property_status_reason = value

    @schema_property("status_message")
    def status_message(self) -> Optional[str]:
        return self._property_status_message

    @status_message.setter
    def status_message(self, value: Optional[str]) -> None:
        if value is None:
            self._property_status_message = None
            return
        self.assert_isinstance(value, "status_message", six.string_types)
        self._property_status_message = value


class FailedResponse(Response):
    """
    Response of tasks.failed endpoint.

    :param updated: Number of tasks updated (0 or 1)
    :type updated: int
    :param fields: Updated fields names and values
    :type fields: dict
    """

    _service = "tasks"
    _action = "failed"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "fields": {
                "additionalProperties": True,
                "description": "Updated fields names and values",
                "type": ["object", "null"],
            },
            "updated": {
                "description": "Number of tasks updated (0 or 1)",
                "enum": [0, 1],
                "type": ["integer", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, updated: Optional[int] = None, fields: Optional[dict] = None, **kwargs: Any) -> None:
        super(FailedResponse, self).__init__(**kwargs)
        self.updated = updated
        self.fields = fields

    @schema_property("updated")
    def updated(self) -> Optional[int]:
        return self._property_updated

    @updated.setter
    def updated(self, value: Optional[int]) -> None:
        if value is None:
            self._property_updated = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "updated", six.integer_types)
        self._property_updated = value

    @schema_property("fields")
    def fields(self) -> Optional[dict]:
        return self._property_fields

    @fields.setter
    def fields(self, value: Optional[dict]) -> None:
        if value is None:
            self._property_fields = None
            return
        self.assert_isinstance(value, "fields", (dict,))
        self._property_fields = value


class GetAllRequest(Request):
    """
    Get all the company's tasks and all public tasks

    :param id: List of IDs to filter by
    :type id: Sequence[str]
    :param name: Get only tasks whose name matches this pattern (python regular
        expression syntax)
    :type name: str
    :param user: List of user IDs used to filter results by the task's creating
        user
    :type user: Sequence[str]
    :param project: List of project IDs
    :type project: Sequence[str]
    :param page: Page number, returns a specific page out of the resulting list of
        tasks
    :type page: int
    :param page_size: Page size, specifies the number of results returned in each
        page (last page may contain fewer results)
    :type page_size: int
    :param order_by: List of field names to order by. When search_text is used,
        '@text_score' can be used as a field representing the text score of returned
        documents. Use '-' prefix to specify descending order. Optional, recommended
        when using page
    :type order_by: Sequence[str]
    :param type: List of task types. One or more of: 'import', 'annotation',
        'training' or 'testing' (case insensitive)
    :type type: Sequence[str]
    :param tags: List of task user-defined tags. Use '-' prefix to exclude tags
    :type tags: Sequence[str]
    :param system_tags: List of task system tags. Use '-' prefix to exclude system
        tags
    :type system_tags: Sequence[str]
    :param status: List of task status.
    :type status: Sequence[TaskStatusEnum]
    :param only_fields: List of task field names (nesting is supported using '.',
        e.g. execution.model_labels). If provided, this list defines the query's
        projection (only these fields will be returned for each result entry)
    :type only_fields: Sequence[str]
    :param parent: Parent ID
    :type parent: str
    :param status_changed: List of status changed constraint strings (utcformat,
        epoch) with an optional prefix modifier (\\>,\\>=, \\<, \\<=)
    :type status_changed: Sequence[str]
    :param search_text: Free text search query
    :type search_text: str
    :param _all_: Multi-field pattern condition (all fields match pattern)
    :type _all_: MultiFieldPatternData
    :param _any_: Multi-field pattern condition (any field matches pattern)
    :type _any_: MultiFieldPatternData
    """

    _service = "tasks"
    _action = "get_all"
    _version = "2.13"
    _schema = {
        "definitions": {
            "multi_field_pattern_data": {
                "properties": {
                    "fields": {
                        "description": "List of field names",
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                    "pattern": {
                        "description": "Pattern string (regex)",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "task_status_enum": {
                "enum": [
                    "created",
                    "queued",
                    "in_progress",
                    "stopped",
                    "published",
                    "publishing",
                    "closed",
                    "failed",
                    "completed",
                    "unknown",
                ],
                "type": "string",
            },
        },
        "dependencies": {"page": ["page_size"]},
        "properties": {
            "_all_": {
                "description": "Multi-field pattern condition (all fields match pattern)",
                "oneOf": [
                    {"$ref": "#/definitions/multi_field_pattern_data"},
                    {"type": "null"},
                ],
            },
            "_any_": {
                "description": "Multi-field pattern condition (any field matches pattern)",
                "oneOf": [
                    {"$ref": "#/definitions/multi_field_pattern_data"},
                    {"type": "null"},
                ],
            },
            "id": {
                "description": "List of IDs to filter by",
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
            "name": {
                "description": "Get only tasks whose name matches this pattern (python regular expression syntax)",
                "type": ["string", "null"],
            },
            "only_fields": {
                "description": "List of task field names (nesting is supported using '.', e.g. execution.model_labels). If provided, this list defines the query's projection (only these fields will be returned for each result entry)",
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
            "order_by": {
                "description": "List of field names to order by. When search_text is used, '@text_score' can be used as a field representing the text score of returned documents. Use '-' prefix to specify descending order. Optional, recommended when using page",
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
            "page": {
                "description": "Page number, returns a specific page out of the resulting list of tasks",
                "minimum": 0,
                "type": ["integer", "null"],
            },
            "page_size": {
                "description": "Page size, specifies the number of results returned in each page (last page may contain fewer results)",
                "minimum": 1,
                "type": ["integer", "null"],
            },
            "parent": {"description": "Parent ID", "type": ["string", "null"]},
            "project": {
                "description": "List of project IDs",
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
            "search_text": {
                "description": "Free text search query",
                "type": ["string", "null"],
            },
            "status": {
                "description": "List of task status.",
                "items": {"$ref": "#/definitions/task_status_enum"},
                "type": ["array", "null"],
            },
            "status_changed": {
                "description": "List of status changed constraint strings (utcformat, epoch) with an optional prefix modifier (\\>,\\>=, \\<, \\<=)",
                "items": {"pattern": "^(>=|>|<=|<)?.*$", "type": "string"},
                "type": ["array", "null"],
            },
            "system_tags": {
                "description": "List of task system tags. Use '-' prefix to exclude system tags",
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
            "tags": {
                "description": "List of task user-defined tags. Use '-' prefix to exclude tags",
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
            "type": {
                "description": "List of task types. One or more of: 'training', 'testing', 'inference', 'data_processing', 'application', 'monitor', 'controller', 'optimizer', 'service', 'qc' or 'custom' (case insensitive)",
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
            "user": {
                "description": "List of user IDs used to filter results by the task's creating user",
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
        },
        "type": "object",
    }

    def __init__(
        self,
        id: Optional[List[str]] = None,
        name: Optional[str] = None,
        user: Optional[List[str]] = None,
        project: Optional[List[str]] = None,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        order_by: Optional[List[str]] = None,
        type: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        system_tags: Optional[List[str]] = None,
        status: Optional[List[Any]] = None,
        only_fields: Optional[List[str]] = None,
        parent: Optional[str] = None,
        status_changed: Optional[List[str]] = None,
        search_text: Optional[str] = None,
        _all_: Any = None,
        _any_: Any = None,
        **kwargs: Any
    ) -> None:
        super(GetAllRequest, self).__init__(**kwargs)
        self.id = id
        self.name = name
        self.user = user
        self.project = project
        self.page = page
        self.page_size = page_size
        self.order_by = order_by
        self.type = type
        self.tags = tags
        self.system_tags = system_tags
        self.status = status
        self.only_fields = only_fields
        self.parent = parent
        self.status_changed = status_changed
        self.search_text = search_text
        self._all_ = _all_
        self._any_ = _any_

    @schema_property("id")
    def id(self) -> Optional[List[str]]:
        return self._property_id

    @id.setter
    def id(self, value: Optional[List[str]]) -> None:
        if value is None:
            self._property_id = None
            return
        self.assert_isinstance(value, "id", (list, tuple))
        self.assert_isinstance(value, "id", six.string_types, is_array=True)
        self._property_id = value

    @schema_property("name")
    def name(self) -> Optional[str]:
        return self._property_name

    @name.setter
    def name(self, value: Optional[str]) -> None:
        if value is None:
            self._property_name = None
            return
        self.assert_isinstance(value, "name", six.string_types)
        self._property_name = value

    @schema_property("user")
    def user(self) -> Optional[List[str]]:
        return self._property_user

    @user.setter
    def user(self, value: Optional[List[str]]) -> None:
        if value is None:
            self._property_user = None
            return
        self.assert_isinstance(value, "user", (list, tuple))
        self.assert_isinstance(value, "user", six.string_types, is_array=True)
        self._property_user = value

    @schema_property("project")
    def project(self) -> Optional[List[str]]:
        return self._property_project

    @project.setter
    def project(self, value: Optional[List[str]]) -> None:
        if value is None:
            self._property_project = None
            return
        self.assert_isinstance(value, "project", (list, tuple))
        self.assert_isinstance(value, "project", six.string_types, is_array=True)
        self._property_project = value

    @schema_property("page")
    def page(self) -> Optional[int]:
        return self._property_page

    @page.setter
    def page(self, value: Optional[int]) -> None:
        if value is None:
            self._property_page = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "page", six.integer_types)
        self._property_page = value

    @schema_property("page_size")
    def page_size(self) -> Optional[int]:
        return self._property_page_size

    @page_size.setter
    def page_size(self, value: Optional[int]) -> None:
        if value is None:
            self._property_page_size = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "page_size", six.integer_types)
        self._property_page_size = value

    @schema_property("order_by")
    def order_by(self) -> Optional[List[str]]:
        return self._property_order_by

    @order_by.setter
    def order_by(self, value: Optional[List[str]]) -> None:
        if value is None:
            self._property_order_by = None
            return
        self.assert_isinstance(value, "order_by", (list, tuple))
        self.assert_isinstance(value, "order_by", six.string_types, is_array=True)
        self._property_order_by = value

    @schema_property("type")
    def type(self) -> Optional[List[str]]:
        return self._property_type

    @type.setter
    def type(self, value: Optional[List[str]]) -> None:
        if value is None:
            self._property_type = None
            return
        self.assert_isinstance(value, "type", (list, tuple))
        self.assert_isinstance(value, "type", six.string_types, is_array=True)
        self._property_type = value

    @schema_property("tags")
    def tags(self) -> Optional[List[str]]:
        return self._property_tags

    @tags.setter
    def tags(self, value: Optional[List[str]]) -> None:
        if value is None:
            self._property_tags = None
            return
        self.assert_isinstance(value, "tags", (list, tuple))
        self.assert_isinstance(value, "tags", six.string_types, is_array=True)
        self._property_tags = value

    @schema_property("system_tags")
    def system_tags(self) -> Optional[List[str]]:
        return self._property_system_tags

    @system_tags.setter
    def system_tags(self, value: Optional[List[str]]) -> None:
        if value is None:
            self._property_system_tags = None
            return
        self.assert_isinstance(value, "system_tags", (list, tuple))
        self.assert_isinstance(value, "system_tags", six.string_types, is_array=True)
        self._property_system_tags = value

    @schema_property("status")
    def status(self) -> Optional[List[Any]]:
        return self._property_status

    @status.setter
    def status(self, value: Optional[List[Any]]) -> None:
        if value is None:
            self._property_status = None
            return
        self.assert_isinstance(value, "status", (list, tuple))
        if any((isinstance(v, six.string_types) for v in value)):
            value = [TaskStatusEnum(v) if isinstance(v, six.string_types) else v for v in value]
        else:
            self.assert_isinstance(value, "status", TaskStatusEnum, is_array=True)
        self._property_status = value

    @schema_property("only_fields")
    def only_fields(self) -> Optional[List[str]]:
        return self._property_only_fields

    @only_fields.setter
    def only_fields(self, value: Optional[List[str]]) -> None:
        if value is None:
            self._property_only_fields = None
            return
        self.assert_isinstance(value, "only_fields", (list, tuple))
        self.assert_isinstance(value, "only_fields", six.string_types, is_array=True)
        self._property_only_fields = value

    @schema_property("parent")
    def parent(self) -> Optional[str]:
        return self._property_parent

    @parent.setter
    def parent(self, value: Optional[str]) -> None:
        if value is None:
            self._property_parent = None
            return
        self.assert_isinstance(value, "parent", six.string_types)
        self._property_parent = value

    @schema_property("status_changed")
    def status_changed(self) -> Optional[List[str]]:
        return self._property_status_changed

    @status_changed.setter
    def status_changed(self, value: Optional[List[str]]) -> None:
        if value is None:
            self._property_status_changed = None
            return
        self.assert_isinstance(value, "status_changed", (list, tuple))
        self.assert_isinstance(value, "status_changed", six.string_types, is_array=True)
        self._property_status_changed = value

    @schema_property("search_text")
    def search_text(self) -> Optional[str]:
        return self._property_search_text

    @search_text.setter
    def search_text(self, value: Optional[str]) -> None:
        if value is None:
            self._property_search_text = None
            return
        self.assert_isinstance(value, "search_text", six.string_types)
        self._property_search_text = value

    @schema_property("_all_")
    def _all_(self) -> Any:
        return self._property__all_

    @_all_.setter
    def _all_(self, value: Any) -> None:
        if value is None:
            self._property__all_ = None
            return
        if isinstance(value, dict):
            value = MultiFieldPatternData.from_dict(value)
        else:
            self.assert_isinstance(value, "_all_", MultiFieldPatternData)
        self._property__all_ = value

    @schema_property("_any_")
    def _any_(self) -> Any:
        return self._property__any_

    @_any_.setter
    def _any_(self, value: Any) -> None:
        if value is None:
            self._property__any_ = None
            return
        if isinstance(value, dict):
            value = MultiFieldPatternData.from_dict(value)
        else:
            self.assert_isinstance(value, "_any_", MultiFieldPatternData)
        self._property__any_ = value


class GetAllResponse(Response):
    """
    Response of tasks.get_all endpoint.

    :param tasks: List of tasks
    :type tasks: Sequence[Task]
    """

    _service = "tasks"
    _action = "get_all"
    _version = "2.13"
    _schema = {
        "definitions": {
            "artifact": {
                "properties": {
                    "content_size": {
                        "description": "Raw data length in bytes",
                        "type": "integer",
                    },
                    "display_data": {
                        "description": "User-defined list of key/value pairs, sorted",
                        "items": {"items": {"type": "string"}, "type": "array"},
                        "type": "array",
                    },
                    "hash": {
                        "description": "Hash of entire raw data",
                        "type": "string",
                    },
                    "key": {"description": "Entry key", "type": "string"},
                    "mode": {
                        "$ref": "#/definitions/artifact_mode_enum",
                        "description": "System defined input/output indication",
                    },
                    "timestamp": {
                        "description": "Epoch time when artifact was created",
                        "type": "integer",
                    },
                    "type": {"description": "System defined type", "type": "string"},
                    "type_data": {
                        "$ref": "#/definitions/artifact_type_data",
                        "description": "Additional fields defined by the system",
                    },
                    "uri": {"description": "Raw data location", "type": "string"},
                },
                "required": ["key", "type"],
                "type": "object",
            },
            "artifact_mode_enum": {
                "default": "output",
                "enum": ["input", "output"],
                "type": "string",
            },
            "artifact_type_data": {
                "properties": {
                    "content_type": {
                        "description": "System defined raw data content type",
                        "type": ["string", "null"],
                    },
                    "data_hash": {
                        "description": "Hash of raw data, without any headers or descriptive parts",
                        "type": ["string", "null"],
                    },
                    "preview": {
                        "description": "Description or textual data",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "configuration_item": {
                "properties": {
                    "description": {
                        "description": "The parameter description. Optional",
                        "type": ["string", "null"],
                    },
                    "name": {
                        "description": "Name of the parameter. Should be unique",
                        "type": ["string", "null"],
                    },
                    "type": {
                        "description": "Type of the parameter. Optional",
                        "type": ["string", "null"],
                    },
                    "value": {
                        "description": "Value of the parameter",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "execution": {
                "properties": {
                    "artifacts": {
                        "description": "Task artifacts",
                        "items": {"$ref": "#/definitions/artifact"},
                        "type": ["array", "null"],
                    },
                    "framework": {
                        "description": "Framework related to the task. Case insensitive. Mandatory for Training tasks. ",
                        "type": ["string", "null"],
                    },
                    "model_desc": {
                        "additionalProperties": True,
                        "description": "Json object representing the Model descriptors",
                        "type": ["object", "null"],
                    },
                    "model_labels": {
                        "additionalProperties": {"type": "integer"},
                        "description": "Json object representing the ids of the labels in the model.\n                The keys are the layers' names and the values are the IDs.\n                Not applicable for Register (Import) tasks.\n                Mandatory for Training tasks",
                        "type": ["object", "null"],
                    },
                    "parameters": {
                        "additionalProperties": True,
                        "description": "Json object containing the Task parameters",
                        "type": ["object", "null"],
                    },
                    "queue": {
                        "description": "Queue ID where task was queued.",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "last_metrics_event": {
                "properties": {
                    "max_value": {
                        "description": "Maximum value reported",
                        "type": ["number", "null"],
                    },
                    "metric": {
                        "description": "Metric name",
                        "type": ["string", "null"],
                    },
                    "min_value": {
                        "description": "Minimum value reported",
                        "type": ["number", "null"],
                    },
                    "value": {
                        "description": "Last value reported",
                        "type": ["number", "null"],
                    },
                    "variant": {
                        "description": "Variant name",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "last_metrics_variants": {
                "additionalProperties": {"$ref": "#/definitions/last_metrics_event"},
                "description": "Last metric events, one for each variant hash",
                "type": "object",
            },
            "output": {
                "properties": {
                    "destination": {
                        "description": "Storage id. This is where output files will be stored.",
                        "type": ["string", "null"],
                    },
                    "error": {
                        "description": "Last error text",
                        "type": ["string", "null"],
                    },
                    "model": {"description": "Model id.", "type": ["string", "null"]},
                    "result": {
                        "description": "Task result. Values: 'success', 'failure'",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "params_item": {
                "properties": {
                    "description": {
                        "description": "The parameter description. Optional",
                        "type": ["string", "null"],
                    },
                    "name": {
                        "description": "Name of the parameter. The combination of section and name should be unique",
                        "type": ["string", "null"],
                    },
                    "section": {
                        "description": "Section that the parameter belongs to",
                        "type": ["string", "null"],
                    },
                    "type": {
                        "description": "Type of the parameter. Optional",
                        "type": ["string", "null"],
                    },
                    "value": {
                        "description": "Value of the parameter",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "script": {
                "properties": {
                    "binary": {
                        "default": "python",
                        "description": "Binary to use when running the script",
                        "type": ["string", "null"],
                    },
                    "branch": {
                        "description": "Repository branch id If not provided and tag not provided, default repository branch is used.",
                        "type": ["string", "null"],
                    },
                    "diff": {
                        "description": "Uncommitted changes found in the repository when task was run",
                        "type": ["string", "null"],
                    },
                    "entry_point": {
                        "description": "Path to execute within the repository",
                        "type": ["string", "null"],
                    },
                    "repository": {
                        "description": "Name of the repository where the script is located",
                        "type": ["string", "null"],
                    },
                    "requirements": {
                        "description": "A JSON object containing requirements strings by key",
                        "type": ["object", "null"],
                    },
                    "tag": {
                        "description": "Repository tag",
                        "type": ["string", "null"],
                    },
                    "version_num": {
                        "description": "Version (changeset) number. Optional (default is head version) Unused if tag is provided.",
                        "type": ["string", "null"],
                    },
                    "working_dir": {
                        "description": "Path to the folder from which to run the script Default - root folder of repository",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "section_params": {
                "additionalProperties": {"$ref": "#/definitions/params_item"},
                "description": "Task section params",
                "type": "object",
            },
            "task": {
                "properties": {
                    "active_duration": {
                        "description": "Task duration time (seconds)",
                        "type": ["integer", "null"],
                    },
                    "comment": {
                        "description": "Free text comment",
                        "type": ["string", "null"],
                    },
                    "company": {
                        "description": "Company ID",
                        "type": ["string", "null"],
                    },
                    "completed": {
                        "description": "Task end time (UTC)",
                        "format": "date-time",
                        "type": ["string", "null"],
                    },
                    "configuration": {
                        "additionalProperties": {"$ref": "#/definitions/configuration_item"},
                        "description": "Task configuration params",
                        "type": ["object", "null"],
                    },
                    "container": {
                        "type": "object",
                        "description": "Docker container parameters",
                        "additionalProperties": {"type": ["string", "null"]},
                    },
                    "created": {
                        "description": "Task creation time (UTC) ",
                        "format": "date-time",
                        "type": ["string", "null"],
                    },
                    "execution": {
                        "description": "Task execution params",
                        "oneOf": [
                            {"$ref": "#/definitions/execution"},
                            {"type": "null"},
                        ],
                    },
                    "hyperparams": {
                        "additionalProperties": {"$ref": "#/definitions/section_params"},
                        "description": "Task hyper params per section",
                        "type": ["object", "null"],
                    },
                    "id": {"description": "Task id", "type": ["string", "null"]},
                    "last_change": {
                        "description": "Last time any update was done to the task",
                        "format": "date-time",
                        "type": ["string", "null"],
                    },
                    "last_iteration": {
                        "description": "Last iteration reported for this task",
                        "type": ["integer", "null"],
                    },
                    "last_metrics": {
                        "additionalProperties": {"$ref": "#/definitions/last_metrics_variants"},
                        "description": "Last metric variants (hash to events), one for each metric hash",
                        "type": ["object", "null"],
                    },
                    "last_update": {
                        "description": "Last time this task was created, edited, changed or events for this task were reported",
                        "format": "date-time",
                        "type": ["string", "null"],
                    },
                    "last_worker": {
                        "description": "ID of last worker that handled the task",
                        "type": ["string", "null"],
                    },
                    "last_worker_report": {
                        "description": "Last time a worker reported while working on this task",
                        "format": "date-time",
                        "type": ["string", "null"],
                    },
                    "models": {
                        "description": "Task models",
                        "oneOf": [
                            {"$ref": "#/definitions/task_models"},
                            {"type": "null"},
                        ],
                    },
                    "name": {"description": "Task Name", "type": ["string", "null"]},
                    "output": {
                        "description": "Task output params",
                        "oneOf": [{"$ref": "#/definitions/output"}, {"type": "null"}],
                    },
                    "parent": {
                        "description": "Parent task id",
                        "type": ["string", "null"],
                    },
                    "project": {
                        "description": "Project ID of the project to which this task is assigned",
                        "type": ["string", "null"],
                    },
                    "published": {
                        "description": "Last status change time",
                        "format": "date-time",
                        "type": ["string", "null"],
                    },
                    "script": {
                        "description": "Script info",
                        "oneOf": [{"$ref": "#/definitions/script"}, {"type": "null"}],
                    },
                    "started": {
                        "description": "Task start time (UTC)",
                        "format": "date-time",
                        "type": ["string", "null"],
                    },
                    "status": {
                        "description": "",
                        "oneOf": [
                            {"$ref": "#/definitions/task_status_enum"},
                            {"type": "null"},
                        ],
                    },
                    "status_changed": {
                        "description": "Last status change time",
                        "format": "date-time",
                        "type": ["string", "null"],
                    },
                    "status_message": {
                        "description": "free text string representing info about the status",
                        "type": ["string", "null"],
                    },
                    "status_reason": {
                        "description": "Reason for last status change",
                        "type": ["string", "null"],
                    },
                    "system_tags": {
                        "description": "System tags list. This field is reserved for system use, please don't use it.",
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                    "tags": {
                        "description": "User-defined tags list",
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                    "type": {
                        "description": "Type of task. Values: 'training', 'testing'",
                        "oneOf": [
                            {"$ref": "#/definitions/task_type_enum"},
                            {"type": "null"},
                        ],
                    },
                    "user": {
                        "description": "Associated user id",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "task_model_item": {
                "properties": {
                    "model": {"description": "The model ID", "type": "string"},
                    "name": {"description": "The task model name", "type": "string"},
                },
                "required": ["name", "model"],
                "type": "object",
            },
            "task_models": {
                "properties": {
                    "input": {
                        "description": "The list of task input models",
                        "items": {"$ref": "#/definitions/task_model_item"},
                        "type": ["array", "null"],
                    },
                    "output": {
                        "description": "The list of task output models",
                        "items": {"$ref": "#/definitions/task_model_item"},
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            },
            "task_status_enum": {
                "enum": [
                    "created",
                    "queued",
                    "in_progress",
                    "stopped",
                    "published",
                    "publishing",
                    "closed",
                    "failed",
                    "completed",
                    "unknown",
                ],
                "type": "string",
            },
            "task_type_enum": {
                "enum": [
                    "training",
                    "testing",
                    "inference",
                    "data_processing",
                    "application",
                    "monitor",
                    "controller",
                    "optimizer",
                    "service",
                    "qc",
                    "custom",
                ],
                "type": "string",
            },
        },
        "properties": {
            "tasks": {
                "description": "List of tasks",
                "items": {"$ref": "#/definitions/task"},
                "type": ["array", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, tasks: Optional[List[Any]] = None, **kwargs: Any) -> None:
        super(GetAllResponse, self).__init__(**kwargs)
        self.tasks = tasks

    @schema_property("tasks")
    def tasks(self) -> Optional[List[Any]]:
        return self._property_tasks

    @tasks.setter
    def tasks(self, value: Optional[List[Any]]) -> None:
        if value is None:
            self._property_tasks = None
            return
        self.assert_isinstance(value, "tasks", (list, tuple))
        if any((isinstance(v, dict) for v in value)):
            value = [Task.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "tasks", Task, is_array=True)
        self._property_tasks = value


class GetByIdRequest(Request):
    """
    Gets task information

    :param task: Task ID
    :type task: str
    """

    _service = "tasks"
    _action = "get_by_id"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {"task": {"description": "Task ID", "type": "string"}},
        "required": ["task"],
        "type": "object",
    }

    def __init__(self, task: str, **kwargs: Any) -> None:
        super(GetByIdRequest, self).__init__(**kwargs)
        self.task = task

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


class GetByIdResponse(Response):
    """
    Response of tasks.get_by_id endpoint.

    :param task: Task info
    :type task: Task
    """

    _service = "tasks"
    _action = "get_by_id"
    _version = "2.13"
    _schema = {
        "definitions": {
            "artifact": {
                "properties": {
                    "content_size": {
                        "description": "Raw data length in bytes",
                        "type": "integer",
                    },
                    "display_data": {
                        "description": "User-defined list of key/value pairs, sorted",
                        "items": {"items": {"type": "string"}, "type": "array"},
                        "type": "array",
                    },
                    "hash": {
                        "description": "Hash of entire raw data",
                        "type": "string",
                    },
                    "key": {"description": "Entry key", "type": "string"},
                    "mode": {
                        "$ref": "#/definitions/artifact_mode_enum",
                        "description": "System defined input/output indication",
                    },
                    "timestamp": {
                        "description": "Epoch time when artifact was created",
                        "type": "integer",
                    },
                    "type": {"description": "System defined type", "type": "string"},
                    "type_data": {
                        "$ref": "#/definitions/artifact_type_data",
                        "description": "Additional fields defined by the system",
                    },
                    "uri": {"description": "Raw data location", "type": "string"},
                },
                "required": ["key", "type"],
                "type": "object",
            },
            "artifact_mode_enum": {
                "default": "output",
                "enum": ["input", "output"],
                "type": "string",
            },
            "artifact_type_data": {
                "properties": {
                    "content_type": {
                        "description": "System defined raw data content type",
                        "type": ["string", "null"],
                    },
                    "data_hash": {
                        "description": "Hash of raw data, without any headers or descriptive parts",
                        "type": ["string", "null"],
                    },
                    "preview": {
                        "description": "Description or textual data",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "configuration_item": {
                "properties": {
                    "description": {
                        "description": "The parameter description. Optional",
                        "type": ["string", "null"],
                    },
                    "name": {
                        "description": "Name of the parameter. Should be unique",
                        "type": ["string", "null"],
                    },
                    "type": {
                        "description": "Type of the parameter. Optional",
                        "type": ["string", "null"],
                    },
                    "value": {
                        "description": "Value of the parameter",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "execution": {
                "properties": {
                    "artifacts": {
                        "description": "Task artifacts",
                        "items": {"$ref": "#/definitions/artifact"},
                        "type": ["array", "null"],
                    },
                    "framework": {
                        "description": "Framework related to the task. Case insensitive. Mandatory for Training tasks. ",
                        "type": ["string", "null"],
                    },
                    "model_desc": {
                        "additionalProperties": True,
                        "description": "Json object representing the Model descriptors",
                        "type": ["object", "null"],
                    },
                    "model_labels": {
                        "additionalProperties": {"type": "integer"},
                        "description": "Json object representing the ids of the labels in the model.\n                The keys are the layers' names and the values are the IDs.\n                Not applicable for Register (Import) tasks.\n                Mandatory for Training tasks",
                        "type": ["object", "null"],
                    },
                    "parameters": {
                        "additionalProperties": True,
                        "description": "Json object containing the Task parameters",
                        "type": ["object", "null"],
                    },
                    "queue": {
                        "description": "Queue ID where task was queued.",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "last_metrics_event": {
                "properties": {
                    "max_value": {
                        "description": "Maximum value reported",
                        "type": ["number", "null"],
                    },
                    "metric": {
                        "description": "Metric name",
                        "type": ["string", "null"],
                    },
                    "min_value": {
                        "description": "Minimum value reported",
                        "type": ["number", "null"],
                    },
                    "value": {
                        "description": "Last value reported",
                        "type": ["number", "null"],
                    },
                    "variant": {
                        "description": "Variant name",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "last_metrics_variants": {
                "additionalProperties": {"$ref": "#/definitions/last_metrics_event"},
                "description": "Last metric events, one for each variant hash",
                "type": "object",
            },
            "output": {
                "properties": {
                    "destination": {
                        "description": "Storage id. This is where output files will be stored.",
                        "type": ["string", "null"],
                    },
                    "error": {
                        "description": "Last error text",
                        "type": ["string", "null"],
                    },
                    "model": {"description": "Model id.", "type": ["string", "null"]},
                    "result": {
                        "description": "Task result. Values: 'success', 'failure'",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "params_item": {
                "properties": {
                    "description": {
                        "description": "The parameter description. Optional",
                        "type": ["string", "null"],
                    },
                    "name": {
                        "description": "Name of the parameter. The combination of section and name should be unique",
                        "type": ["string", "null"],
                    },
                    "section": {
                        "description": "Section that the parameter belongs to",
                        "type": ["string", "null"],
                    },
                    "type": {
                        "description": "Type of the parameter. Optional",
                        "type": ["string", "null"],
                    },
                    "value": {
                        "description": "Value of the parameter",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "script": {
                "properties": {
                    "binary": {
                        "default": "python",
                        "description": "Binary to use when running the script",
                        "type": ["string", "null"],
                    },
                    "branch": {
                        "description": "Repository branch id If not provided and tag not provided, default repository branch is used.",
                        "type": ["string", "null"],
                    },
                    "diff": {
                        "description": "Uncommitted changes found in the repository when task was run",
                        "type": ["string", "null"],
                    },
                    "entry_point": {
                        "description": "Path to execute within the repository",
                        "type": ["string", "null"],
                    },
                    "repository": {
                        "description": "Name of the repository where the script is located",
                        "type": ["string", "null"],
                    },
                    "requirements": {
                        "description": "A JSON object containing requirements strings by key",
                        "type": ["object", "null"],
                    },
                    "tag": {
                        "description": "Repository tag",
                        "type": ["string", "null"],
                    },
                    "version_num": {
                        "description": "Version (changeset) number. Optional (default is head version) Unused if tag is provided.",
                        "type": ["string", "null"],
                    },
                    "working_dir": {
                        "description": "Path to the folder from which to run the script Default - root folder of repository",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "section_params": {
                "additionalProperties": {"$ref": "#/definitions/params_item"},
                "description": "Task section params",
                "type": "object",
            },
            "task": {
                "properties": {
                    "active_duration": {
                        "description": "Task duration time (seconds)",
                        "type": ["integer", "null"],
                    },
                    "comment": {
                        "description": "Free text comment",
                        "type": ["string", "null"],
                    },
                    "company": {
                        "description": "Company ID",
                        "type": ["string", "null"],
                    },
                    "completed": {
                        "description": "Task end time (UTC)",
                        "format": "date-time",
                        "type": ["string", "null"],
                    },
                    "configuration": {
                        "additionalProperties": {"$ref": "#/definitions/configuration_item"},
                        "description": "Task configuration params",
                        "type": ["object", "null"],
                    },
                    "container": {
                        "type": "object",
                        "description": "Docker container parameters",
                        "additionalProperties": {"type": ["string", "null"]},
                    },
                    "created": {
                        "description": "Task creation time (UTC) ",
                        "format": "date-time",
                        "type": ["string", "null"],
                    },
                    "execution": {
                        "description": "Task execution params",
                        "oneOf": [
                            {"$ref": "#/definitions/execution"},
                            {"type": "null"},
                        ],
                    },
                    "hyperparams": {
                        "additionalProperties": {"$ref": "#/definitions/section_params"},
                        "description": "Task hyper params per section",
                        "type": ["object", "null"],
                    },
                    "id": {"description": "Task id", "type": ["string", "null"]},
                    "last_change": {
                        "description": "Last time any update was done to the task",
                        "format": "date-time",
                        "type": ["string", "null"],
                    },
                    "last_iteration": {
                        "description": "Last iteration reported for this task",
                        "type": ["integer", "null"],
                    },
                    "last_metrics": {
                        "additionalProperties": {"$ref": "#/definitions/last_metrics_variants"},
                        "description": "Last metric variants (hash to events), one for each metric hash",
                        "type": ["object", "null"],
                    },
                    "last_update": {
                        "description": "Last time this task was created, edited, changed or events for this task were reported",
                        "format": "date-time",
                        "type": ["string", "null"],
                    },
                    "last_worker": {
                        "description": "ID of last worker that handled the task",
                        "type": ["string", "null"],
                    },
                    "last_worker_report": {
                        "description": "Last time a worker reported while working on this task",
                        "format": "date-time",
                        "type": ["string", "null"],
                    },
                    "models": {
                        "description": "Task models",
                        "oneOf": [
                            {"$ref": "#/definitions/task_models"},
                            {"type": "null"},
                        ],
                    },
                    "name": {"description": "Task Name", "type": ["string", "null"]},
                    "output": {
                        "description": "Task output params",
                        "oneOf": [{"$ref": "#/definitions/output"}, {"type": "null"}],
                    },
                    "parent": {
                        "description": "Parent task id",
                        "type": ["string", "null"],
                    },
                    "project": {
                        "description": "Project ID of the project to which this task is assigned",
                        "type": ["string", "null"],
                    },
                    "published": {
                        "description": "Last status change time",
                        "format": "date-time",
                        "type": ["string", "null"],
                    },
                    "script": {
                        "description": "Script info",
                        "oneOf": [{"$ref": "#/definitions/script"}, {"type": "null"}],
                    },
                    "started": {
                        "description": "Task start time (UTC)",
                        "format": "date-time",
                        "type": ["string", "null"],
                    },
                    "status": {
                        "description": "",
                        "oneOf": [
                            {"$ref": "#/definitions/task_status_enum"},
                            {"type": "null"},
                        ],
                    },
                    "status_changed": {
                        "description": "Last status change time",
                        "format": "date-time",
                        "type": ["string", "null"],
                    },
                    "status_message": {
                        "description": "free text string representing info about the status",
                        "type": ["string", "null"],
                    },
                    "status_reason": {
                        "description": "Reason for last status change",
                        "type": ["string", "null"],
                    },
                    "system_tags": {
                        "description": "System tags list. This field is reserved for system use, please don't use it.",
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                    "tags": {
                        "description": "User-defined tags list",
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                    "type": {
                        "description": "Type of task. Values: 'training', 'testing'",
                        "oneOf": [
                            {"$ref": "#/definitions/task_type_enum"},
                            {"type": "null"},
                        ],
                    },
                    "user": {
                        "description": "Associated user id",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "task_model_item": {
                "properties": {
                    "model": {"description": "The model ID", "type": "string"},
                    "name": {"description": "The task model name", "type": "string"},
                },
                "required": ["name", "model"],
                "type": "object",
            },
            "task_models": {
                "properties": {
                    "input": {
                        "description": "The list of task input models",
                        "items": {"$ref": "#/definitions/task_model_item"},
                        "type": ["array", "null"],
                    },
                    "output": {
                        "description": "The list of task output models",
                        "items": {"$ref": "#/definitions/task_model_item"},
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            },
            "task_status_enum": {
                "enum": [
                    "created",
                    "queued",
                    "in_progress",
                    "stopped",
                    "published",
                    "publishing",
                    "closed",
                    "failed",
                    "completed",
                    "unknown",
                ],
                "type": "string",
            },
            "task_type_enum": {
                "enum": [
                    "training",
                    "testing",
                    "inference",
                    "data_processing",
                    "application",
                    "monitor",
                    "controller",
                    "optimizer",
                    "service",
                    "qc",
                    "custom",
                ],
                "type": "string",
            },
        },
        "properties": {
            "task": {
                "description": "Task info",
                "oneOf": [{"$ref": "#/definitions/task"}, {"type": "null"}],
            }
        },
        "type": "object",
    }

    def __init__(self, task: Any = None, **kwargs: Any) -> None:
        super(GetByIdResponse, self).__init__(**kwargs)
        self.task = task

    @schema_property("task")
    def task(self) -> Any:
        return self._property_task

    @task.setter
    def task(self, value: Any) -> None:
        if value is None:
            self._property_task = None
            return
        if isinstance(value, dict):
            value = Task.from_dict(value)
        else:
            self.assert_isinstance(value, "task", Task)
        self._property_task = value


class GetConfigurationNamesRequest(Request):
    """
    Get the list of task configuration items names

    :param tasks: Task IDs
    :type tasks: Sequence[str]
    :param skip_empty: If set to 'true' then the names for configurations with
        missing values are not returned
    :type skip_empty: bool
    """

    _service = "tasks"
    _action = "get_configuration_names"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "skip_empty": {
                "default": True,
                "description": "If set to 'true' then the names for configurations with missing values are not returned",
                "type": "boolean",
            },
            "tasks": {
                "description": "Task IDs",
                "items": {"type": "string"},
                "type": "array",
            },
        },
        "required": ["tasks"],
        "type": "object",
    }

    def __init__(self, tasks: List[str], skip_empty: Optional[bool] = True, **kwargs: Any) -> None:
        super(GetConfigurationNamesRequest, self).__init__(**kwargs)
        self.tasks = tasks
        self.skip_empty = skip_empty

    @schema_property("tasks")
    def tasks(self) -> List[str]:
        return self._property_tasks

    @tasks.setter
    def tasks(self, value: List[str]) -> None:
        if value is None:
            self._property_tasks = None
            return
        self.assert_isinstance(value, "tasks", (list, tuple))
        self.assert_isinstance(value, "tasks", six.string_types, is_array=True)
        self._property_tasks = value

    @schema_property("skip_empty")
    def skip_empty(self) -> Optional[bool]:
        return self._property_skip_empty

    @skip_empty.setter
    def skip_empty(self, value: Optional[bool]) -> None:
        if value is None:
            self._property_skip_empty = None
            return
        self.assert_isinstance(value, "skip_empty", (bool,))
        self._property_skip_empty = value


class GetConfigurationNamesResponse(Response):
    """
    Response of tasks.get_configuration_names endpoint.

    :param configurations: Names of task configuration items (keyed by task ID)
    :type configurations: dict
    """

    _service = "tasks"
    _action = "get_configuration_names"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "configurations": {
                "description": "Names of task configuration items (keyed by task ID)",
                "properties": {
                    "names": {
                        "description": "Configuration names",
                        "items": {"type": "string"},
                        "type": "array",
                    },
                    "task": {"description": "Task ID", "type": "string"},
                },
                "type": ["object", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, configurations: Optional[dict] = None, **kwargs: Any) -> None:
        super(GetConfigurationNamesResponse, self).__init__(**kwargs)
        self.configurations = configurations

    @schema_property("configurations")
    def configurations(self) -> Optional[dict]:
        return self._property_configurations

    @configurations.setter
    def configurations(self, value: Optional[dict]) -> None:
        if value is None:
            self._property_configurations = None
            return
        self.assert_isinstance(value, "configurations", (dict,))
        self._property_configurations = value


class GetConfigurationsRequest(Request):
    """
    Get the list of task configurations

    :param tasks: Task IDs
    :type tasks: Sequence[str]
    :param names: Names of the configuration items to retreive. If not passed or
        empty then all the configurations will be retreived.
    :type names: Sequence[str]
    """

    _service = "tasks"
    _action = "get_configurations"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "names": {
                "description": "Names of the configuration items to retreive. If not passed or empty then all the configurations will be retreived.",
                "items": {"type": "string"},
                "type": "array",
            },
            "tasks": {
                "description": "Task IDs",
                "items": {"type": "string"},
                "type": "array",
            },
        },
        "required": ["tasks"],
        "type": "object",
    }

    def __init__(self, tasks: List[str], names: Optional[List[str]] = None, **kwargs: Any) -> None:
        super(GetConfigurationsRequest, self).__init__(**kwargs)
        self.tasks = tasks
        self.names = names

    @schema_property("tasks")
    def tasks(self) -> List[str]:
        return self._property_tasks

    @tasks.setter
    def tasks(self, value: List[str]) -> None:
        if value is None:
            self._property_tasks = None
            return
        self.assert_isinstance(value, "tasks", (list, tuple))
        self.assert_isinstance(value, "tasks", six.string_types, is_array=True)
        self._property_tasks = value

    @schema_property("names")
    def names(self) -> Optional[List[str]]:
        return self._property_names

    @names.setter
    def names(self, value: Optional[List[str]]) -> None:
        if value is None:
            self._property_names = None
            return
        self.assert_isinstance(value, "names", (list, tuple))
        self.assert_isinstance(value, "names", six.string_types, is_array=True)
        self._property_names = value


class GetConfigurationsResponse(Response):
    """
    Response of tasks.get_configurations endpoint.

    :param configurations: Configurations (keyed by task ID)
    :type configurations: Sequence[dict]
    """

    _service = "tasks"
    _action = "get_configurations"
    _version = "2.13"
    _schema = {
        "definitions": {
            "configuration_item": {
                "properties": {
                    "description": {
                        "description": "The parameter description. Optional",
                        "type": ["string", "null"],
                    },
                    "name": {
                        "description": "Name of the parameter. Should be unique",
                        "type": ["string", "null"],
                    },
                    "type": {
                        "description": "Type of the parameter. Optional",
                        "type": ["string", "null"],
                    },
                    "value": {
                        "description": "Value of the parameter",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            }
        },
        "properties": {
            "configurations": {
                "description": "Configurations (keyed by task ID)",
                "items": {
                    "properties": {
                        "configuration": {
                            "description": "Configuration list",
                            "items": {"$ref": "#/definitions/configuration_item"},
                            "type": "array",
                        },
                        "task": {"description": "Task ID", "type": "string"},
                    },
                    "type": "object",
                },
                "type": ["array", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, configurations: Optional[List[dict]] = None, **kwargs: Any) -> None:
        super(GetConfigurationsResponse, self).__init__(**kwargs)
        self.configurations = configurations

    @schema_property("configurations")
    def configurations(self) -> Optional[List[dict]]:
        return self._property_configurations

    @configurations.setter
    def configurations(self, value: Optional[List[dict]]) -> None:
        if value is None:
            self._property_configurations = None
            return
        self.assert_isinstance(value, "configurations", (list, tuple))
        self.assert_isinstance(value, "configurations", (dict,), is_array=True)
        self._property_configurations = value


class GetHyperParamsRequest(Request):
    """
    Get the list of task hyper parameters

    :param tasks: Task IDs
    :type tasks: Sequence[str]
    """

    _service = "tasks"
    _action = "get_hyper_params"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "tasks": {
                "description": "Task IDs",
                "items": {"type": "string"},
                "type": "array",
            }
        },
        "required": ["tasks"],
        "type": "object",
    }

    def __init__(self, tasks: List[str], **kwargs: Any) -> None:
        super(GetHyperParamsRequest, self).__init__(**kwargs)
        self.tasks = tasks

    @schema_property("tasks")
    def tasks(self) -> List[str]:
        return self._property_tasks

    @tasks.setter
    def tasks(self, value: List[str]) -> None:
        if value is None:
            self._property_tasks = None
            return
        self.assert_isinstance(value, "tasks", (list, tuple))
        self.assert_isinstance(value, "tasks", six.string_types, is_array=True)
        self._property_tasks = value


class GetHyperParamsResponse(Response):
    """
    Response of tasks.get_hyper_params endpoint.

    :param params: Hyper parameters (keyed by task ID)
    :type params: Sequence[dict]
    """

    _service = "tasks"
    _action = "get_hyper_params"
    _version = "2.13"
    _schema = {
        "definitions": {
            "params_item": {
                "properties": {
                    "description": {
                        "description": "The parameter description. Optional",
                        "type": ["string", "null"],
                    },
                    "name": {
                        "description": "Name of the parameter. The combination of section and name should be unique",
                        "type": ["string", "null"],
                    },
                    "section": {
                        "description": "Section that the parameter belongs to",
                        "type": ["string", "null"],
                    },
                    "type": {
                        "description": "Type of the parameter. Optional",
                        "type": ["string", "null"],
                    },
                    "value": {
                        "description": "Value of the parameter",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            }
        },
        "properties": {
            "params": {
                "description": "Hyper parameters (keyed by task ID)",
                "items": {
                    "properties": {
                        "hyperparams": {
                            "description": "Hyper parameters",
                            "items": {"$ref": "#/definitions/params_item"},
                            "type": "array",
                        },
                        "task": {"description": "Task ID", "type": "string"},
                    },
                    "type": "object",
                },
                "type": ["array", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, params: Optional[List[dict]] = None, **kwargs: Any) -> None:
        super(GetHyperParamsResponse, self).__init__(**kwargs)
        self.params = params

    @schema_property("params")
    def params(self) -> Optional[List[dict]]:
        return self._property_params

    @params.setter
    def params(self, value: Optional[List[dict]]) -> None:
        if value is None:
            self._property_params = None
            return
        self.assert_isinstance(value, "params", (list, tuple))
        self.assert_isinstance(value, "params", (dict,), is_array=True)
        self._property_params = value


class GetTypesRequest(Request):
    """
    Get the list of task types used in the specified projects

    :param projects: The list of projects which tasks will be analyzed. If not
        passed or empty then all the company and public tasks will be analyzed
    :type projects: Sequence[str]
    """

    _service = "tasks"
    _action = "get_types"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "projects": {
                "description": "The list of projects which tasks will be analyzed. If not passed or empty then all the company and public tasks will be analyzed",
                "items": {"type": "string"},
                "type": ["array", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, projects: Optional[List[str]] = None, **kwargs: Any) -> None:
        super(GetTypesRequest, self).__init__(**kwargs)
        self.projects = projects

    @schema_property("projects")
    def projects(self) -> Optional[List[str]]:
        return self._property_projects

    @projects.setter
    def projects(self, value: Optional[List[str]]) -> None:
        if value is None:
            self._property_projects = None
            return
        self.assert_isinstance(value, "projects", (list, tuple))
        self.assert_isinstance(value, "projects", six.string_types, is_array=True)
        self._property_projects = value


class GetTypesResponse(Response):
    """
    Response of tasks.get_types endpoint.

    :param types: Unique list of the task types used in the requested projects
    :type types: Sequence[str]
    """

    _service = "tasks"
    _action = "get_types"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "types": {
                "description": "Unique list of the task types used in the requested projects",
                "items": {"type": "string"},
                "type": ["array", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, types: Optional[List[str]] = None, **kwargs: Any) -> None:
        super(GetTypesResponse, self).__init__(**kwargs)
        self.types = types

    @schema_property("types")
    def types(self) -> Optional[List[str]]:
        return self._property_types

    @types.setter
    def types(self, value: Optional[List[str]]) -> None:
        if value is None:
            self._property_types = None
            return
        self.assert_isinstance(value, "types", (list, tuple))
        self.assert_isinstance(value, "types", six.string_types, is_array=True)
        self._property_types = value


class MakePrivateRequest(Request):
    """
    Convert public tasks to private

    :param ids: Ids of the tasks to convert. Only the tasks originated by the
        company can be converted
    :type ids: Sequence[str]
    """

    _service = "tasks"
    _action = "make_private"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "ids": {
                "description": "Ids of the tasks to convert. Only the tasks originated by the company can be converted",
                "items": {"type": "string"},
                "type": ["array", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, ids: Optional[List[str]] = None, **kwargs: Any) -> None:
        super(MakePrivateRequest, self).__init__(**kwargs)
        self.ids = ids

    @schema_property("ids")
    def ids(self) -> Optional[List[str]]:
        return self._property_ids

    @ids.setter
    def ids(self, value: Optional[List[str]]) -> None:
        if value is None:
            self._property_ids = None
            return
        self.assert_isinstance(value, "ids", (list, tuple))
        self.assert_isinstance(value, "ids", six.string_types, is_array=True)
        self._property_ids = value


class MakePrivateResponse(Response):
    """
    Response of tasks.make_private endpoint.

    :param updated: Number of tasks updated
    :type updated: int
    """

    _service = "tasks"
    _action = "make_private"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "updated": {
                "description": "Number of tasks updated",
                "type": ["integer", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, updated: Optional[int] = None, **kwargs: Any) -> None:
        super(MakePrivateResponse, self).__init__(**kwargs)
        self.updated = updated

    @schema_property("updated")
    def updated(self) -> Optional[int]:
        return self._property_updated

    @updated.setter
    def updated(self, value: Optional[int]) -> None:
        if value is None:
            self._property_updated = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "updated", six.integer_types)
        self._property_updated = value


class MakePublicRequest(Request):
    """
    Convert company tasks to public

    :param ids: Ids of the tasks to convert
    :type ids: Sequence[str]
    """

    _service = "tasks"
    _action = "make_public"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "ids": {
                "description": "Ids of the tasks to convert",
                "items": {"type": "string"},
                "type": ["array", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, ids: Optional[List[str]] = None, **kwargs: Any) -> None:
        super(MakePublicRequest, self).__init__(**kwargs)
        self.ids = ids

    @schema_property("ids")
    def ids(self) -> Optional[List[str]]:
        return self._property_ids

    @ids.setter
    def ids(self, value: Optional[List[str]]) -> None:
        if value is None:
            self._property_ids = None
            return
        self.assert_isinstance(value, "ids", (list, tuple))
        self.assert_isinstance(value, "ids", six.string_types, is_array=True)
        self._property_ids = value


class MakePublicResponse(Response):
    """
    Response of tasks.make_public endpoint.

    :param updated: Number of tasks updated
    :type updated: int
    """

    _service = "tasks"
    _action = "make_public"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "updated": {
                "description": "Number of tasks updated",
                "type": ["integer", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, updated: Optional[int] = None, **kwargs: Any) -> None:
        super(MakePublicResponse, self).__init__(**kwargs)
        self.updated = updated

    @schema_property("updated")
    def updated(self) -> Optional[int]:
        return self._property_updated

    @updated.setter
    def updated(self, value: Optional[int]) -> None:
        if value is None:
            self._property_updated = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "updated", six.integer_types)
        self._property_updated = value


class MoveRequest(Request):
    """
    Move tasks to a project

    :param ids: Tasks to move
    :type ids: Sequence[str]
    :param project: Target project ID. If not provided, `project_name` must be
        provided.
    :type project: str
    :param project_name: Target project name. If provided and a project with this
        name does not exist, a new project will be created. If not provided, `project`
        must be provided.
    :type project_name: str
    """

    _service = "tasks"
    _action = "move"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "ids": {
                "description": "Tasks to move",
                "items": {"type": "string"},
                "type": "array",
            },
            "project": {
                "description": "Target project ID. If not provided, `project_name` must be provided.",
                "type": "string",
            },
            "project_name": {
                "description": "Target project name. If provided and a project with this name does not exist, a new project will be created. If not provided, `project` must be provided.",
                "type": "string",
            },
        },
        "required": ["ids"],
        "type": "object",
    }

    def __init__(
        self, ids: List[str], project: Optional[str] = None, project_name: Optional[str] = None, **kwargs: Any
    ) -> None:
        super(MoveRequest, self).__init__(**kwargs)
        self.ids = ids
        self.project = project
        self.project_name = project_name

    @schema_property("ids")
    def ids(self) -> List[str]:
        return self._property_ids

    @ids.setter
    def ids(self, value: List[str]) -> None:
        if value is None:
            self._property_ids = None
            return
        self.assert_isinstance(value, "ids", (list, tuple))
        self.assert_isinstance(value, "ids", six.string_types, is_array=True)
        self._property_ids = value

    @schema_property("project")
    def project(self) -> Optional[str]:
        return self._property_project

    @project.setter
    def project(self, value: Optional[str]) -> None:
        if value is None:
            self._property_project = None
            return
        self.assert_isinstance(value, "project", six.string_types)
        self._property_project = value

    @schema_property("project_name")
    def project_name(self) -> Optional[str]:
        return self._property_project_name

    @project_name.setter
    def project_name(self, value: Optional[str]) -> None:
        if value is None:
            self._property_project_name = None
            return
        self.assert_isinstance(value, "project_name", six.string_types)
        self._property_project_name = value


class MoveResponse(Response):
    """
    Response of tasks.move endpoint.

    """

    _service = "tasks"
    _action = "move"
    _version = "2.13"
    _schema = {"additionalProperties": True, "definitions": {}, "type": "object"}


class PingRequest(Request):
    """
     Refresh the task's last update time

    :param task: Task ID
    :type task: str
    """

    _service = "tasks"
    _action = "ping"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {"task": {"description": "Task ID", "type": "string"}},
        "required": ["task"],
        "type": "object",
    }

    def __init__(self, task: str, **kwargs: Any) -> None:
        super(PingRequest, self).__init__(**kwargs)
        self.task = task

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


class PingResponse(Response):
    """
    Response of tasks.ping endpoint.

    """

    _service = "tasks"
    _action = "ping"
    _version = "2.13"
    _schema = {"additionalProperties": False, "definitions": {}, "type": "object"}


class PublishRequest(Request):
    """
    Mark a task status as published. If a model was created, it should be set to ready.

    :param force: If not true, call fails if the task status is not 'stopped'
    :type force: bool
    :param publish_model: Indicates that the task output model (if exists) should
        be published. Optional, the default value is True.
    :type publish_model: bool
    :param task: Task ID
    :type task: str
    :param status_reason: Reason for status change
    :type status_reason: str
    :param status_message: Extra information regarding status change
    :type status_message: str
    """

    _service = "tasks"
    _action = "publish"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "force": {
                "default": False,
                "description": "If not true, call fails if the task status is not 'stopped'",
                "type": ["boolean", "null"],
            },
            "publish_model": {
                "description": "Indicates that the task output model (if exists) should be published. Optional, the default value is True.",
                "type": ["boolean", "null"],
            },
            "status_message": {
                "description": "Extra information regarding status change",
                "type": "string",
            },
            "status_reason": {
                "description": "Reason for status change",
                "type": "string",
            },
            "task": {"description": "Task ID", "type": "string"},
        },
        "required": ["task"],
        "type": "object",
    }

    def __init__(
        self,
        task: str,
        force: Optional[bool] = False,
        publish_model: Optional[bool] = None,
        status_reason: Optional[str] = None,
        status_message: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super(PublishRequest, self).__init__(**kwargs)
        self.force = force
        self.publish_model = publish_model
        self.task = task
        self.status_reason = status_reason
        self.status_message = status_message

    @schema_property("force")
    def force(self) -> Optional[bool]:
        return self._property_force

    @force.setter
    def force(self, value: Optional[bool]) -> None:
        if value is None:
            self._property_force = None
            return
        self.assert_isinstance(value, "force", (bool,))
        self._property_force = value

    @schema_property("publish_model")
    def publish_model(self) -> Optional[bool]:
        return self._property_publish_model

    @publish_model.setter
    def publish_model(self, value: Optional[bool]) -> None:
        if value is None:
            self._property_publish_model = None
            return
        self.assert_isinstance(value, "publish_model", (bool,))
        self._property_publish_model = value

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

    @schema_property("status_reason")
    def status_reason(self) -> Optional[str]:
        return self._property_status_reason

    @status_reason.setter
    def status_reason(self, value: Optional[str]) -> None:
        if value is None:
            self._property_status_reason = None
            return
        self.assert_isinstance(value, "status_reason", six.string_types)
        self._property_status_reason = value

    @schema_property("status_message")
    def status_message(self) -> Optional[str]:
        return self._property_status_message

    @status_message.setter
    def status_message(self, value: Optional[str]) -> None:
        if value is None:
            self._property_status_message = None
            return
        self.assert_isinstance(value, "status_message", six.string_types)
        self._property_status_message = value


class PublishResponse(Response):
    """
    Response of tasks.publish endpoint.

    :param updated: Number of tasks updated (0 or 1)
    :type updated: int
    :param fields: Updated fields names and values
    :type fields: dict
    """

    _service = "tasks"
    _action = "publish"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "fields": {
                "additionalProperties": True,
                "description": "Updated fields names and values",
                "type": ["object", "null"],
            },
            "updated": {
                "description": "Number of tasks updated (0 or 1)",
                "enum": [0, 1],
                "type": ["integer", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, updated: Optional[int] = None, fields: Optional[dict] = None, **kwargs: Any) -> None:
        super(PublishResponse, self).__init__(**kwargs)
        self.updated = updated
        self.fields = fields

    @schema_property("updated")
    def updated(self) -> Optional[int]:
        return self._property_updated

    @updated.setter
    def updated(self, value: Optional[int]) -> None:
        if value is None:
            self._property_updated = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "updated", six.integer_types)
        self._property_updated = value

    @schema_property("fields")
    def fields(self) -> Optional[dict]:
        return self._property_fields

    @fields.setter
    def fields(self, value: Optional[dict]) -> None:
        if value is None:
            self._property_fields = None
            return
        self.assert_isinstance(value, "fields", (dict,))
        self._property_fields = value


class PublishManyRequest(Request):
    """
    Publish tasks

    :param ids: Entities to move
    :type ids: Sequence[str]
    :param status_reason: Reason for status change
    :type status_reason: str
    :param status_message: Extra information regarding status change
    :type status_message: str
    :param force: If not true, call fails if the task status is not 'stopped'
    :type force: bool
    :param publish_model: Indicates that the task output model (if exists) should
        be published. Optional, the default value is True.
    :type publish_model: bool
    """

    _service = "tasks"
    _action = "publish_many"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "force": {
                "default": False,
                "description": "If not true, call fails if the task status is not 'stopped'",
                "type": "boolean",
            },
            "ids": {
                "description": "Entities to move",
                "items": {"type": "string"},
                "type": "array",
            },
            "publish_model": {
                "description": "Indicates that the task output model (if exists) should be published. Optional, the default value is True.",
                "type": "boolean",
            },
            "status_message": {
                "description": "Extra information regarding status change",
                "type": "string",
            },
            "status_reason": {
                "description": "Reason for status change",
                "type": "string",
            },
        },
        "required": ["ids"],
        "type": "object",
    }

    def __init__(
        self,
        ids: List[str],
        status_reason: Optional[str] = None,
        status_message: Optional[str] = None,
        force: Optional[bool] = False,
        publish_model: Optional[bool] = None,
        **kwargs: Any
    ) -> None:
        super(PublishManyRequest, self).__init__(**kwargs)
        self.ids = ids
        self.status_reason = status_reason
        self.status_message = status_message
        self.force = force
        self.publish_model = publish_model

    @schema_property("ids")
    def ids(self) -> List[str]:
        return self._property_ids

    @ids.setter
    def ids(self, value: List[str]) -> None:
        if value is None:
            self._property_ids = None
            return
        self.assert_isinstance(value, "ids", (list, tuple))
        self.assert_isinstance(value, "ids", six.string_types, is_array=True)
        self._property_ids = value

    @schema_property("status_reason")
    def status_reason(self) -> Optional[str]:
        return self._property_status_reason

    @status_reason.setter
    def status_reason(self, value: Optional[str]) -> None:
        if value is None:
            self._property_status_reason = None
            return
        self.assert_isinstance(value, "status_reason", six.string_types)
        self._property_status_reason = value

    @schema_property("status_message")
    def status_message(self) -> Optional[str]:
        return self._property_status_message

    @status_message.setter
    def status_message(self, value: Optional[str]) -> None:
        if value is None:
            self._property_status_message = None
            return
        self.assert_isinstance(value, "status_message", six.string_types)
        self._property_status_message = value

    @schema_property("force")
    def force(self) -> Optional[bool]:
        return self._property_force

    @force.setter
    def force(self, value: Optional[bool]) -> None:
        if value is None:
            self._property_force = None
            return
        self.assert_isinstance(value, "force", (bool,))
        self._property_force = value

    @schema_property("publish_model")
    def publish_model(self) -> Optional[bool]:
        return self._property_publish_model

    @publish_model.setter
    def publish_model(self, value: Optional[bool]) -> None:
        if value is None:
            self._property_publish_model = None
            return
        self.assert_isinstance(value, "publish_model", (bool,))
        self._property_publish_model = value


class PublishManyResponse(Response):
    """
    Response of tasks.publish_many endpoint.

    :param published: Number of tasks published
    :type published: int
    """

    _service = "tasks"
    _action = "publish_many"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "failures": {
            "item": {
                "error": {
                    "description": "Error info",
                    "properties": {
                        "codes": {"item": {"type": "integer"}, "type": "array"},
                        "data": {"additionalProperties": True, "type": "object"},
                        "msg": {"type": "string"},
                    },
                    "type": "object",
                },
                "id": {"description": "ID of the failed entity", "type": "string"},
                "type": "object",
            },
            "type": "array",
        },
        "properties": {
            "published": {
                "description": "Number of tasks published",
                "type": ["integer", "null"],
            }
        },
    }

    def __init__(self, published: Optional[int] = None, **kwargs: Any) -> None:
        super(PublishManyResponse, self).__init__(**kwargs)
        self.published = published

    @schema_property("published")
    def published(self) -> Optional[int]:
        return self._property_published

    @published.setter
    def published(self, value: Optional[int]) -> None:
        if value is None:
            self._property_published = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "published", six.integer_types)
        self._property_published = value


class ResetRequest(Request):
    """
    Reset a task to its initial state, along with any information stored for it (statistics, frame updates etc.).

    :param force: If not true, call fails if the task status is 'completed'
    :type force: bool
    :param clear_all: Clear script and execution sections completely
    :type clear_all: bool
    :param task: Task ID
    :type task: str
    :param status_reason: Reason for status change
    :type status_reason: str
    :param status_message: Extra information regarding status change
    :type status_message: str
    :param return_file_urls: If set to 'true' then return the urls of the files
        that were uploaded by this task. Default value is 'false'
    :type return_file_urls: bool
    :param delete_output_models: If set to 'true' then delete output models of this
        task that are not referenced by other tasks. Default value is 'true'
    :type delete_output_models: bool
    """

    _service = "tasks"
    _action = "reset"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "clear_all": {
                "default": False,
                "description": "Clear script and execution sections completely",
                "type": ["boolean", "null"],
            },
            "delete_output_models": {
                "description": "If set to 'true' then delete output models of this task that are not referenced by other tasks. Default value is 'true'",
                "type": "boolean",
            },
            "force": {
                "default": False,
                "description": "If not true, call fails if the task status is 'completed'",
                "type": ["boolean", "null"],
            },
            "return_file_urls": {
                "description": "If set to 'true' then return the urls of the files that were uploaded by this task. Default value is 'false'",
                "type": "boolean",
            },
            "status_message": {
                "description": "Extra information regarding status change",
                "type": "string",
            },
            "status_reason": {
                "description": "Reason for status change",
                "type": "string",
            },
            "task": {"description": "Task ID", "type": "string"},
        },
        "required": ["task"],
        "type": "object",
    }

    def __init__(
        self,
        task: str,
        force: Optional[bool] = False,
        clear_all: Optional[bool] = False,
        status_reason: Optional[str] = None,
        status_message: Optional[str] = None,
        return_file_urls: Optional[bool] = None,
        delete_output_models: Optional[bool] = None,
        **kwargs: Any
    ) -> None:
        super(ResetRequest, self).__init__(**kwargs)
        self.force = force
        self.clear_all = clear_all
        self.task = task
        self.status_reason = status_reason
        self.status_message = status_message
        self.return_file_urls = return_file_urls
        self.delete_output_models = delete_output_models

    @schema_property("force")
    def force(self) -> Optional[bool]:
        return self._property_force

    @force.setter
    def force(self, value: Optional[bool]) -> None:
        if value is None:
            self._property_force = None
            return
        self.assert_isinstance(value, "force", (bool,))
        self._property_force = value

    @schema_property("clear_all")
    def clear_all(self) -> Optional[bool]:
        return self._property_clear_all

    @clear_all.setter
    def clear_all(self, value: Optional[bool]) -> None:
        if value is None:
            self._property_clear_all = None
            return
        self.assert_isinstance(value, "clear_all", (bool,))
        self._property_clear_all = value

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

    @schema_property("status_reason")
    def status_reason(self) -> Optional[str]:
        return self._property_status_reason

    @status_reason.setter
    def status_reason(self, value: Optional[str]) -> None:
        if value is None:
            self._property_status_reason = None
            return
        self.assert_isinstance(value, "status_reason", six.string_types)
        self._property_status_reason = value

    @schema_property("status_message")
    def status_message(self) -> Optional[str]:
        return self._property_status_message

    @status_message.setter
    def status_message(self, value: Optional[str]) -> None:
        if value is None:
            self._property_status_message = None
            return
        self.assert_isinstance(value, "status_message", six.string_types)
        self._property_status_message = value

    @schema_property("return_file_urls")
    def return_file_urls(self) -> Optional[bool]:
        return self._property_return_file_urls

    @return_file_urls.setter
    def return_file_urls(self, value: Optional[bool]) -> None:
        if value is None:
            self._property_return_file_urls = None
            return
        self.assert_isinstance(value, "return_file_urls", (bool,))
        self._property_return_file_urls = value

    @schema_property("delete_output_models")
    def delete_output_models(self) -> Optional[bool]:
        return self._property_delete_output_models

    @delete_output_models.setter
    def delete_output_models(self, value: Optional[bool]) -> None:
        if value is None:
            self._property_delete_output_models = None
            return
        self.assert_isinstance(value, "delete_output_models", (bool,))
        self._property_delete_output_models = value


class ResetResponse(Response):
    """
    Response of tasks.reset endpoint.

    :param updated: Number of tasks updated (0 or 1)
    :type updated: int
    :param fields: Updated fields names and values
    :type fields: dict
    :param deleted_indices: List of deleted ES indices that were removed as part of
        the reset process
    :type deleted_indices: Sequence[str]
    :param dequeued: Response from queues.remove_task
    :type dequeued: dict
    :param frames: Response from frames.rollback
    :type frames: dict
    :param events: Response from events.delete_for_task
    :type events: dict
    :param deleted_models: Number of output models deleted by the reset
    :type deleted_models: int
    :param urls: The urls of the files that were uploaded by this task. Returned if
        the 'return_file_urls' was set to True
    :type urls: TaskUrls
    """

    _service = "tasks"
    _action = "reset"
    _version = "2.13"
    _schema = {
        "definitions": {
            "task_urls": {
                "properties": {
                    "artifact_urls": {
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                    "event_urls": {
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                    "model_urls": {
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            }
        },
        "properties": {
            "deleted_indices": {
                "description": "List of deleted ES indices that were removed as part of the reset process",
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
            "deleted_models": {
                "description": "Number of output models deleted by the reset",
                "type": ["integer", "null"],
            },
            "dequeued": {
                "additionalProperties": True,
                "description": "Response from queues.remove_task",
                "type": ["object", "null"],
            },
            "events": {
                "additionalProperties": True,
                "description": "Response from events.delete_for_task",
                "type": ["object", "null"],
            },
            "fields": {
                "additionalProperties": True,
                "description": "Updated fields names and values",
                "type": ["object", "null"],
            },
            "frames": {
                "additionalProperties": True,
                "description": "Response from frames.rollback",
                "type": ["object", "null"],
            },
            "updated": {
                "description": "Number of tasks updated (0 or 1)",
                "enum": [0, 1],
                "type": ["integer", "null"],
            },
            "urls": {
                "description": "The urls of the files that were uploaded by this task. Returned if the 'return_file_urls' was set to True",
                "oneOf": [{"$ref": "#/definitions/task_urls"}, {"type": "null"}],
            },
        },
        "type": "object",
    }

    def __init__(
        self,
        updated: Optional[int] = None,
        fields: Optional[dict] = None,
        deleted_indices: Optional[List[str]] = None,
        dequeued: Optional[dict] = None,
        frames: Optional[dict] = None,
        events: Optional[dict] = None,
        deleted_models: Optional[int] = None,
        urls: Any = None,
        **kwargs: Any
    ) -> None:
        super(ResetResponse, self).__init__(**kwargs)
        self.updated = updated
        self.fields = fields
        self.deleted_indices = deleted_indices
        self.dequeued = dequeued
        self.frames = frames
        self.events = events
        self.deleted_models = deleted_models
        self.urls = urls

    @schema_property("updated")
    def updated(self) -> Optional[int]:
        return self._property_updated

    @updated.setter
    def updated(self, value: Optional[int]) -> None:
        if value is None:
            self._property_updated = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "updated", six.integer_types)
        self._property_updated = value

    @schema_property("fields")
    def fields(self) -> Optional[dict]:
        return self._property_fields

    @fields.setter
    def fields(self, value: Optional[dict]) -> None:
        if value is None:
            self._property_fields = None
            return
        self.assert_isinstance(value, "fields", (dict,))
        self._property_fields = value

    @schema_property("deleted_indices")
    def deleted_indices(self) -> Optional[List[str]]:
        return self._property_deleted_indices

    @deleted_indices.setter
    def deleted_indices(self, value: Optional[List[str]]) -> None:
        if value is None:
            self._property_deleted_indices = None
            return
        self.assert_isinstance(value, "deleted_indices", (list, tuple))
        self.assert_isinstance(value, "deleted_indices", six.string_types, is_array=True)
        self._property_deleted_indices = value

    @schema_property("dequeued")
    def dequeued(self) -> Optional[dict]:
        return self._property_dequeued

    @dequeued.setter
    def dequeued(self, value: Optional[dict]) -> None:
        if value is None:
            self._property_dequeued = None
            return
        self.assert_isinstance(value, "dequeued", (dict,))
        self._property_dequeued = value

    @schema_property("frames")
    def frames(self) -> Optional[dict]:
        return self._property_frames

    @frames.setter
    def frames(self, value: Optional[dict]) -> None:
        if value is None:
            self._property_frames = None
            return
        self.assert_isinstance(value, "frames", (dict,))
        self._property_frames = value

    @schema_property("events")
    def events(self) -> Optional[dict]:
        return self._property_events

    @events.setter
    def events(self, value: Optional[dict]) -> None:
        if value is None:
            self._property_events = None
            return
        self.assert_isinstance(value, "events", (dict,))
        self._property_events = value

    @schema_property("deleted_models")
    def deleted_models(self) -> Optional[int]:
        return self._property_deleted_models

    @deleted_models.setter
    def deleted_models(self, value: Optional[int]) -> None:
        if value is None:
            self._property_deleted_models = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "deleted_models", six.integer_types)
        self._property_deleted_models = value

    @schema_property("urls")
    def urls(self) -> Any:
        return self._property_urls

    @urls.setter
    def urls(self, value: Any) -> None:
        if value is None:
            self._property_urls = None
            return
        if isinstance(value, dict):
            value = TaskUrls.from_dict(value)
        else:
            self.assert_isinstance(value, "urls", TaskUrls)
        self._property_urls = value


class ResetManyRequest(Request):
    """
    Reset tasks

    :param ids: Entities to move
    :type ids: Sequence[str]
    :param force: If not true, call fails if the task status is 'completed'
    :type force: bool
    :param clear_all: Clear script and execution sections completely
    :type clear_all: bool
    :param return_file_urls: If set to 'true' then return the urls of the files
        that were uploaded by the tasks. Default value is 'false'
    :type return_file_urls: bool
    :param delete_output_models: If set to 'true' then delete output models of the
        tasks that are not referenced by other tasks. Default value is 'true'
    :type delete_output_models: bool
    """

    _service = "tasks"
    _action = "reset_many"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "clear_all": {
                "default": False,
                "description": "Clear script and execution sections completely",
                "type": "boolean",
            },
            "delete_output_models": {
                "description": "If set to 'true' then delete output models of the tasks that are not referenced by other tasks. Default value is 'true'",
                "type": "boolean",
            },
            "force": {
                "default": False,
                "description": "If not true, call fails if the task status is 'completed'",
                "type": "boolean",
            },
            "ids": {
                "description": "Entities to move",
                "items": {"type": "string"},
                "type": "array",
            },
            "return_file_urls": {
                "description": "If set to 'true' then return the urls of the files that were uploaded by the tasks. Default value is 'false'",
                "type": "boolean",
            },
        },
        "required": ["ids"],
        "type": "object",
    }

    def __init__(
        self,
        ids: List[str],
        force: Optional[bool] = False,
        clear_all: Optional[bool] = False,
        return_file_urls: Optional[bool] = None,
        delete_output_models: Optional[bool] = None,
        **kwargs: Any
    ) -> None:
        super(ResetManyRequest, self).__init__(**kwargs)
        self.ids = ids
        self.force = force
        self.clear_all = clear_all
        self.return_file_urls = return_file_urls
        self.delete_output_models = delete_output_models

    @schema_property("ids")
    def ids(self) -> List[str]:
        return self._property_ids

    @ids.setter
    def ids(self, value: List[str]) -> None:
        if value is None:
            self._property_ids = None
            return
        self.assert_isinstance(value, "ids", (list, tuple))
        self.assert_isinstance(value, "ids", six.string_types, is_array=True)
        self._property_ids = value

    @schema_property("force")
    def force(self) -> Optional[bool]:
        return self._property_force

    @force.setter
    def force(self, value: Optional[bool]) -> None:
        if value is None:
            self._property_force = None
            return
        self.assert_isinstance(value, "force", (bool,))
        self._property_force = value

    @schema_property("clear_all")
    def clear_all(self) -> Optional[bool]:
        return self._property_clear_all

    @clear_all.setter
    def clear_all(self, value: Optional[bool]) -> None:
        if value is None:
            self._property_clear_all = None
            return
        self.assert_isinstance(value, "clear_all", (bool,))
        self._property_clear_all = value

    @schema_property("return_file_urls")
    def return_file_urls(self) -> Optional[bool]:
        return self._property_return_file_urls

    @return_file_urls.setter
    def return_file_urls(self, value: Optional[bool]) -> None:
        if value is None:
            self._property_return_file_urls = None
            return
        self.assert_isinstance(value, "return_file_urls", (bool,))
        self._property_return_file_urls = value

    @schema_property("delete_output_models")
    def delete_output_models(self) -> Optional[bool]:
        return self._property_delete_output_models

    @delete_output_models.setter
    def delete_output_models(self, value: Optional[bool]) -> None:
        if value is None:
            self._property_delete_output_models = None
            return
        self.assert_isinstance(value, "delete_output_models", (bool,))
        self._property_delete_output_models = value


class ResetManyResponse(Response):
    """
    Response of tasks.reset_many endpoint.

    :param reset: Number of tasks reset
    :type reset: int
    :param dequeued: Number of tasks dequeued
    :type dequeued: dict
    :param deleted_models: Number of output models deleted by the reset
    :type deleted_models: int
    :param urls: The urls of the files that were uploaded by the tasks. Returned if
        the 'return_file_urls' was set to 'true'
    :type urls: TaskUrls
    """

    _service = "tasks"
    _action = "reset_many"
    _version = "2.13"
    _schema = {
        "definitions": {
            "task_urls": {
                "properties": {
                    "artifact_urls": {
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                    "event_urls": {
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                    "model_urls": {
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            }
        },
        "failures": {
            "item": {
                "error": {
                    "description": "Error info",
                    "properties": {
                        "codes": {"item": {"type": "integer"}, "type": "array"},
                        "data": {"additionalProperties": True, "type": "object"},
                        "msg": {"type": "string"},
                    },
                    "type": "object",
                },
                "id": {"description": "ID of the failed entity", "type": "string"},
                "type": "object",
            },
            "type": "array",
        },
        "properties": {
            "deleted_models": {
                "description": "Number of output models deleted by the reset",
                "type": ["integer", "null"],
            },
            "dequeued": {
                "additionalProperties": True,
                "description": "Number of tasks dequeued",
                "type": ["object", "null"],
            },
            "reset": {
                "description": "Number of tasks reset",
                "type": ["integer", "null"],
            },
            "urls": {
                "description": "The urls of the files that were uploaded by the tasks. Returned if the 'return_file_urls' was set to 'true'",
                "oneOf": [{"$ref": "#/definitions/task_urls"}, {"type": "null"}],
            },
        },
    }

    def __init__(
        self,
        reset: Optional[int] = None,
        dequeued: Optional[dict] = None,
        deleted_models: Optional[int] = None,
        urls: Any = None,
        **kwargs: Any
    ) -> None:
        super(ResetManyResponse, self).__init__(**kwargs)
        self.reset = reset
        self.dequeued = dequeued
        self.deleted_models = deleted_models
        self.urls = urls

    @schema_property("reset")
    def reset(self) -> Optional[int]:
        return self._property_reset

    @reset.setter
    def reset(self, value: Optional[int]) -> None:
        if value is None:
            self._property_reset = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "reset", six.integer_types)
        self._property_reset = value

    @schema_property("dequeued")
    def dequeued(self) -> Optional[dict]:
        return self._property_dequeued

    @dequeued.setter
    def dequeued(self, value: Optional[dict]) -> None:
        if value is None:
            self._property_dequeued = None
            return
        self.assert_isinstance(value, "dequeued", (dict,))
        self._property_dequeued = value

    @schema_property("deleted_models")
    def deleted_models(self) -> Optional[int]:
        return self._property_deleted_models

    @deleted_models.setter
    def deleted_models(self, value: Optional[int]) -> None:
        if value is None:
            self._property_deleted_models = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "deleted_models", six.integer_types)
        self._property_deleted_models = value

    @schema_property("urls")
    def urls(self) -> Any:
        return self._property_urls

    @urls.setter
    def urls(self, value: Any) -> None:
        if value is None:
            self._property_urls = None
            return
        if isinstance(value, dict):
            value = TaskUrls.from_dict(value)
        else:
            self.assert_isinstance(value, "urls", TaskUrls)
        self._property_urls = value


class SetRequirementsRequest(Request):
    """
    Set the script requirements for a task

    :param task: Task ID
    :type task: str
    :param requirements: A JSON object containing requirements strings by key
    :type requirements: dict
    """

    _service = "tasks"
    _action = "set_requirements"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "requirements": {
                "description": "A JSON object containing requirements strings by key",
                "type": "object",
            },
            "task": {"description": "Task ID", "type": "string"},
        },
        "required": ["task", "requirements"],
        "type": "object",
    }

    def __init__(self, task: str, requirements: dict, **kwargs: Any) -> None:
        super(SetRequirementsRequest, self).__init__(**kwargs)
        self.task = task
        self.requirements = requirements

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

    @schema_property("requirements")
    def requirements(self) -> dict:
        return self._property_requirements

    @requirements.setter
    def requirements(self, value: dict) -> None:
        if value is None:
            self._property_requirements = None
            return
        self.assert_isinstance(value, "requirements", (dict,))
        self._property_requirements = value


class SetRequirementsResponse(Response):
    """
    Response of tasks.set_requirements endpoint.

    :param updated: Number of tasks updated (0 or 1)
    :type updated: int
    :param fields: Updated fields names and values
    :type fields: dict
    """

    _service = "tasks"
    _action = "set_requirements"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "fields": {
                "additionalProperties": True,
                "description": "Updated fields names and values",
                "type": ["object", "null"],
            },
            "updated": {
                "description": "Number of tasks updated (0 or 1)",
                "enum": [0, 1],
                "type": ["integer", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, updated: Optional[int] = None, fields: Optional[dict] = None, **kwargs: Any) -> None:
        super(SetRequirementsResponse, self).__init__(**kwargs)
        self.updated = updated
        self.fields = fields

    @schema_property("updated")
    def updated(self) -> Optional[int]:
        return self._property_updated

    @updated.setter
    def updated(self, value: Optional[int]) -> None:
        if value is None:
            self._property_updated = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "updated", six.integer_types)
        self._property_updated = value

    @schema_property("fields")
    def fields(self) -> Optional[dict]:
        return self._property_fields

    @fields.setter
    def fields(self, value: Optional[dict]) -> None:
        if value is None:
            self._property_fields = None
            return
        self.assert_isinstance(value, "fields", (dict,))
        self._property_fields = value


class StartedRequest(Request):
    """
    Mark a task status as in_progress. Optionally allows to set the task's execution progress.

    :param force: If not true, call fails if the task status is not 'not_started'
    :type force: bool
    :param task: Task ID
    :type task: str
    :param status_reason: Reason for status change
    :type status_reason: str
    :param status_message: Extra information regarding status change
    :type status_message: str
    """

    _service = "tasks"
    _action = "started"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "force": {
                "default": False,
                "description": "If not true, call fails if the task status is not 'not_started'",
                "type": ["boolean", "null"],
            },
            "status_message": {
                "description": "Extra information regarding status change",
                "type": "string",
            },
            "status_reason": {
                "description": "Reason for status change",
                "type": "string",
            },
            "task": {"description": "Task ID", "type": "string"},
        },
        "required": ["task"],
        "type": "object",
    }

    def __init__(
        self,
        task: str,
        force: Optional[bool] = False,
        status_reason: Optional[str] = None,
        status_message: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super(StartedRequest, self).__init__(**kwargs)
        self.force = force
        self.task = task
        self.status_reason = status_reason
        self.status_message = status_message

    @schema_property("force")
    def force(self) -> Optional[bool]:
        return self._property_force

    @force.setter
    def force(self, value: Optional[bool]) -> None:
        if value is None:
            self._property_force = None
            return
        self.assert_isinstance(value, "force", (bool,))
        self._property_force = value

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

    @schema_property("status_reason")
    def status_reason(self) -> Optional[str]:
        return self._property_status_reason

    @status_reason.setter
    def status_reason(self, value: Optional[str]) -> None:
        if value is None:
            self._property_status_reason = None
            return
        self.assert_isinstance(value, "status_reason", six.string_types)
        self._property_status_reason = value

    @schema_property("status_message")
    def status_message(self) -> Optional[str]:
        return self._property_status_message

    @status_message.setter
    def status_message(self, value: Optional[str]) -> None:
        if value is None:
            self._property_status_message = None
            return
        self.assert_isinstance(value, "status_message", six.string_types)
        self._property_status_message = value


class StartedResponse(Response):
    """
    Response of tasks.started endpoint.

    :param updated: Number of tasks updated (0 or 1)
    :type updated: int
    :param fields: Updated fields names and values
    :type fields: dict
    :param started: Number of tasks started (0 or 1)
    :type started: int
    """

    _service = "tasks"
    _action = "started"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "fields": {
                "additionalProperties": True,
                "description": "Updated fields names and values",
                "type": ["object", "null"],
            },
            "started": {
                "description": "Number of tasks started (0 or 1)",
                "enum": [0, 1],
                "type": ["integer", "null"],
            },
            "updated": {
                "description": "Number of tasks updated (0 or 1)",
                "enum": [0, 1],
                "type": ["integer", "null"],
            },
        },
        "type": "object",
    }

    def __init__(
        self, updated: Optional[int] = None, fields: Optional[dict] = None, started: Optional[int] = None, **kwargs: Any
    ) -> None:
        super(StartedResponse, self).__init__(**kwargs)
        self.updated = updated
        self.fields = fields
        self.started = started

    @schema_property("updated")
    def updated(self) -> Optional[int]:
        return self._property_updated

    @updated.setter
    def updated(self, value: Optional[int]) -> None:
        if value is None:
            self._property_updated = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "updated", six.integer_types)
        self._property_updated = value

    @schema_property("fields")
    def fields(self) -> Optional[dict]:
        return self._property_fields

    @fields.setter
    def fields(self, value: Optional[dict]) -> None:
        if value is None:
            self._property_fields = None
            return
        self.assert_isinstance(value, "fields", (dict,))
        self._property_fields = value

    @schema_property("started")
    def started(self) -> Optional[int]:
        return self._property_started

    @started.setter
    def started(self, value: Optional[int]) -> None:
        if value is None:
            self._property_started = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "started", six.integer_types)
        self._property_started = value


class StopRequest(Request):
    """
    Request to stop a running task

    :param force: If not true, call fails if the task status is not 'in_progress'
    :type force: bool
    :param task: Task ID
    :type task: str
    :param status_reason: Reason for status change
    :type status_reason: str
    :param status_message: Extra information regarding status change
    :type status_message: str
    """

    _service = "tasks"
    _action = "stop"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "force": {
                "default": False,
                "description": "If not true, call fails if the task status is not 'in_progress'",
                "type": ["boolean", "null"],
            },
            "status_message": {
                "description": "Extra information regarding status change",
                "type": "string",
            },
            "status_reason": {
                "description": "Reason for status change",
                "type": "string",
            },
            "task": {"description": "Task ID", "type": "string"},
        },
        "required": ["task"],
        "type": "object",
    }

    def __init__(
        self,
        task: str,
        force: Optional[bool] = False,
        status_reason: Optional[str] = None,
        status_message: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super(StopRequest, self).__init__(**kwargs)
        self.force = force
        self.task = task
        self.status_reason = status_reason
        self.status_message = status_message

    @schema_property("force")
    def force(self) -> Optional[bool]:
        return self._property_force

    @force.setter
    def force(self, value: Optional[bool]) -> None:
        if value is None:
            self._property_force = None
            return
        self.assert_isinstance(value, "force", (bool,))
        self._property_force = value

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

    @schema_property("status_reason")
    def status_reason(self) -> Optional[str]:
        return self._property_status_reason

    @status_reason.setter
    def status_reason(self, value: Optional[str]) -> None:
        if value is None:
            self._property_status_reason = None
            return
        self.assert_isinstance(value, "status_reason", six.string_types)
        self._property_status_reason = value

    @schema_property("status_message")
    def status_message(self) -> Optional[str]:
        return self._property_status_message

    @status_message.setter
    def status_message(self, value: Optional[str]) -> None:
        if value is None:
            self._property_status_message = None
            return
        self.assert_isinstance(value, "status_message", six.string_types)
        self._property_status_message = value


class StopResponse(Response):
    """
    Response of tasks.stop endpoint.

    :param updated: Number of tasks updated (0 or 1)
    :type updated: int
    :param fields: Updated fields names and values
    :type fields: dict
    """

    _service = "tasks"
    _action = "stop"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "fields": {
                "additionalProperties": True,
                "description": "Updated fields names and values",
                "type": ["object", "null"],
            },
            "updated": {
                "description": "Number of tasks updated (0 or 1)",
                "enum": [0, 1],
                "type": ["integer", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, updated: Optional[int] = None, fields: Optional[dict] = None, **kwargs: Any) -> None:
        super(StopResponse, self).__init__(**kwargs)
        self.updated = updated
        self.fields = fields

    @schema_property("updated")
    def updated(self) -> Optional[int]:
        return self._property_updated

    @updated.setter
    def updated(self, value: Optional[int]) -> None:
        if value is None:
            self._property_updated = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "updated", six.integer_types)
        self._property_updated = value

    @schema_property("fields")
    def fields(self) -> Optional[dict]:
        return self._property_fields

    @fields.setter
    def fields(self, value: Optional[dict]) -> None:
        if value is None:
            self._property_fields = None
            return
        self.assert_isinstance(value, "fields", (dict,))
        self._property_fields = value


class StopManyRequest(Request):
    """
    Request to stop running tasks

    :param ids: Entities to move
    :type ids: Sequence[str]
    :param status_reason: Reason for status change
    :type status_reason: str
    :param status_message: Extra information regarding status change
    :type status_message: str
    :param force: If not true, call fails if the task status is not 'in_progress'
    :type force: bool
    """

    _service = "tasks"
    _action = "stop_many"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "force": {
                "default": False,
                "description": "If not true, call fails if the task status is not 'in_progress'",
                "type": "boolean",
            },
            "ids": {
                "description": "Entities to move",
                "items": {"type": "string"},
                "type": "array",
            },
            "status_message": {
                "description": "Extra information regarding status change",
                "type": "string",
            },
            "status_reason": {
                "description": "Reason for status change",
                "type": "string",
            },
        },
        "required": ["ids"],
        "type": "object",
    }

    def __init__(
        self,
        ids: List[str],
        status_reason: Optional[str] = None,
        status_message: Optional[str] = None,
        force: Optional[bool] = False,
        **kwargs: Any
    ) -> None:
        super(StopManyRequest, self).__init__(**kwargs)
        self.ids = ids
        self.status_reason = status_reason
        self.status_message = status_message
        self.force = force

    @schema_property("ids")
    def ids(self) -> List[str]:
        return self._property_ids

    @ids.setter
    def ids(self, value: List[str]) -> None:
        if value is None:
            self._property_ids = None
            return
        self.assert_isinstance(value, "ids", (list, tuple))
        self.assert_isinstance(value, "ids", six.string_types, is_array=True)
        self._property_ids = value

    @schema_property("status_reason")
    def status_reason(self) -> Optional[str]:
        return self._property_status_reason

    @status_reason.setter
    def status_reason(self, value: Optional[str]) -> None:
        if value is None:
            self._property_status_reason = None
            return
        self.assert_isinstance(value, "status_reason", six.string_types)
        self._property_status_reason = value

    @schema_property("status_message")
    def status_message(self) -> Optional[str]:
        return self._property_status_message

    @status_message.setter
    def status_message(self, value: Optional[str]) -> None:
        if value is None:
            self._property_status_message = None
            return
        self.assert_isinstance(value, "status_message", six.string_types)
        self._property_status_message = value

    @schema_property("force")
    def force(self) -> Optional[bool]:
        return self._property_force

    @force.setter
    def force(self, value: Optional[bool]) -> None:
        if value is None:
            self._property_force = None
            return
        self.assert_isinstance(value, "force", (bool,))
        self._property_force = value


class StopManyResponse(Response):
    """
    Response of tasks.stop_many endpoint.

    :param stopped: Number of tasks stopped
    :type stopped: int
    """

    _service = "tasks"
    _action = "stop_many"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "failures": {
            "item": {
                "error": {
                    "description": "Error info",
                    "properties": {
                        "codes": {"item": {"type": "integer"}, "type": "array"},
                        "data": {"additionalProperties": True, "type": "object"},
                        "msg": {"type": "string"},
                    },
                    "type": "object",
                },
                "id": {"description": "ID of the failed entity", "type": "string"},
                "type": "object",
            },
            "type": "array",
        },
        "properties": {
            "stopped": {
                "description": "Number of tasks stopped",
                "type": ["integer", "null"],
            }
        },
    }

    def __init__(self, stopped: Optional[int] = None, **kwargs: Any) -> None:
        super(StopManyResponse, self).__init__(**kwargs)
        self.stopped = stopped

    @schema_property("stopped")
    def stopped(self) -> Optional[int]:
        return self._property_stopped

    @stopped.setter
    def stopped(self, value: Optional[int]) -> None:
        if value is None:
            self._property_stopped = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "stopped", six.integer_types)
        self._property_stopped = value


class StoppedRequest(Request):
    """
    Signal a task has stopped

    :param force: If not true, call fails if the task status is not 'stopped'
    :type force: bool
    :param task: Task ID
    :type task: str
    :param status_reason: Reason for status change
    :type status_reason: str
    :param status_message: Extra information regarding status change
    :type status_message: str
    """

    _service = "tasks"
    _action = "stopped"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "force": {
                "default": False,
                "description": "If not true, call fails if the task status is not 'stopped'",
                "type": ["boolean", "null"],
            },
            "status_message": {
                "description": "Extra information regarding status change",
                "type": "string",
            },
            "status_reason": {
                "description": "Reason for status change",
                "type": "string",
            },
            "task": {"description": "Task ID", "type": "string"},
        },
        "required": ["task"],
        "type": "object",
    }

    def __init__(
        self,
        task: str,
        force: Optional[bool] = False,
        status_reason: Optional[str] = None,
        status_message: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super(StoppedRequest, self).__init__(**kwargs)
        self.force = force
        self.task = task
        self.status_reason = status_reason
        self.status_message = status_message

    @schema_property("force")
    def force(self) -> Optional[bool]:
        return self._property_force

    @force.setter
    def force(self, value: Optional[bool]) -> None:
        if value is None:
            self._property_force = None
            return
        self.assert_isinstance(value, "force", (bool,))
        self._property_force = value

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

    @schema_property("status_reason")
    def status_reason(self) -> Optional[str]:
        return self._property_status_reason

    @status_reason.setter
    def status_reason(self, value: Optional[str]) -> None:
        if value is None:
            self._property_status_reason = None
            return
        self.assert_isinstance(value, "status_reason", six.string_types)
        self._property_status_reason = value

    @schema_property("status_message")
    def status_message(self) -> Optional[str]:
        return self._property_status_message

    @status_message.setter
    def status_message(self, value: Optional[str]) -> None:
        if value is None:
            self._property_status_message = None
            return
        self.assert_isinstance(value, "status_message", six.string_types)
        self._property_status_message = value


class StoppedResponse(Response):
    """
    Response of tasks.stopped endpoint.

    :param updated: Number of tasks updated (0 or 1)
    :type updated: int
    :param fields: Updated fields names and values
    :type fields: dict
    """

    _service = "tasks"
    _action = "stopped"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "fields": {
                "additionalProperties": True,
                "description": "Updated fields names and values",
                "type": ["object", "null"],
            },
            "updated": {
                "description": "Number of tasks updated (0 or 1)",
                "enum": [0, 1],
                "type": ["integer", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, updated: Optional[int] = None, fields: Optional[dict] = None, **kwargs: Any) -> None:
        super(StoppedResponse, self).__init__(**kwargs)
        self.updated = updated
        self.fields = fields

    @schema_property("updated")
    def updated(self) -> Optional[int]:
        return self._property_updated

    @updated.setter
    def updated(self, value: Optional[int]) -> None:
        if value is None:
            self._property_updated = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "updated", six.integer_types)
        self._property_updated = value

    @schema_property("fields")
    def fields(self) -> Optional[dict]:
        return self._property_fields

    @fields.setter
    def fields(self, value: Optional[dict]) -> None:
        if value is None:
            self._property_fields = None
            return
        self.assert_isinstance(value, "fields", (dict,))
        self._property_fields = value


class UpdateRequest(Request):
    """
    Update task's runtime parameters

    :param task: ID of the task
    :type task: str
    :param name: Task name Unique within the company.
    :type name: str
    :param tags: User-defined tags list
    :type tags: Sequence[str]
    :param system_tags: System tags list. This field is reserved for system use,
        please don't use it.
    :type system_tags: Sequence[str]
    :param comment: Free text comment
    :type comment: str
    :param project: Project ID of the project to which this task is assigned
    :type project: str
    :param output__error: Free text error
    :type output__error: str
    :param created: Task creation time (UTC)
    :type created: datetime.datetime
    """

    _service = "tasks"
    _action = "update"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "comment": {"description": "Free text comment ", "type": "string"},
            "created": {
                "description": "Task creation time (UTC) ",
                "format": "date-time",
                "type": "string",
            },
            "name": {
                "description": "Task name Unique within the company.",
                "type": "string",
            },
            "output__error": {"description": "Free text error", "type": "string"},
            "project": {
                "description": "Project ID of the project to which this task is assigned",
                "type": "string",
            },
            "system_tags": {
                "description": "System tags list. This field is reserved for system use, please don't use it.",
                "items": {"type": "string"},
                "type": "array",
            },
            "tags": {
                "description": "User-defined tags list",
                "items": {"type": "string"},
                "type": "array",
            },
            "task": {"description": "ID of the task", "type": "string"},
        },
        "required": ["task"],
        "type": "object",
    }

    def __init__(
        self,
        task: str,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        system_tags: Optional[List[str]] = None,
        comment: Optional[str] = None,
        project: Optional[str] = None,
        output__error: Optional[str] = None,
        created: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super(UpdateRequest, self).__init__(**kwargs)
        self.task = task
        self.name = name
        self.tags = tags
        self.system_tags = system_tags
        self.comment = comment
        self.project = project
        self.output__error = output__error
        self.created = created

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

    @schema_property("name")
    def name(self) -> Optional[str]:
        return self._property_name

    @name.setter
    def name(self, value: Optional[str]) -> None:
        if value is None:
            self._property_name = None
            return
        self.assert_isinstance(value, "name", six.string_types)
        self._property_name = value

    @schema_property("tags")
    def tags(self) -> Optional[List[str]]:
        return self._property_tags

    @tags.setter
    def tags(self, value: Optional[List[str]]) -> None:
        if value is None:
            self._property_tags = None
            return
        self.assert_isinstance(value, "tags", (list, tuple))
        self.assert_isinstance(value, "tags", six.string_types, is_array=True)
        self._property_tags = value

    @schema_property("system_tags")
    def system_tags(self) -> Optional[List[str]]:
        return self._property_system_tags

    @system_tags.setter
    def system_tags(self, value: Optional[List[str]]) -> None:
        if value is None:
            self._property_system_tags = None
            return
        self.assert_isinstance(value, "system_tags", (list, tuple))
        self.assert_isinstance(value, "system_tags", six.string_types, is_array=True)
        self._property_system_tags = value

    @schema_property("comment")
    def comment(self) -> Optional[str]:
        return self._property_comment

    @comment.setter
    def comment(self, value: Optional[str]) -> None:
        if value is None:
            self._property_comment = None
            return
        self.assert_isinstance(value, "comment", six.string_types)
        self._property_comment = value

    @schema_property("project")
    def project(self) -> Optional[str]:
        return self._property_project

    @project.setter
    def project(self, value: Optional[str]) -> None:
        if value is None:
            self._property_project = None
            return
        self.assert_isinstance(value, "project", six.string_types)
        self._property_project = value

    @schema_property("output__error")
    def output__error(self) -> Optional[str]:
        return self._property_output__error

    @output__error.setter
    def output__error(self, value: Optional[str]) -> None:
        if value is None:
            self._property_output__error = None
            return
        self.assert_isinstance(value, "output__error", six.string_types)
        self._property_output__error = value

    @schema_property("created")
    def created(self) -> Optional[str]:
        return self._property_created

    @created.setter
    def created(self, value: Optional[str]) -> None:
        if value is None:
            self._property_created = None
            return
        self.assert_isinstance(value, "created", six.string_types + (datetime,))
        if not isinstance(value, datetime):
            value = parse_datetime(value)
        self._property_created = value


class UpdateResponse(Response):
    """
    Response of tasks.update endpoint.

    :param updated: Number of tasks updated (0 or 1)
    :type updated: int
    :param fields: Updated fields names and values
    :type fields: dict
    """

    _service = "tasks"
    _action = "update"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "fields": {
                "additionalProperties": True,
                "description": "Updated fields names and values",
                "type": ["object", "null"],
            },
            "updated": {
                "description": "Number of tasks updated (0 or 1)",
                "enum": [0, 1],
                "type": ["integer", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, updated: Optional[int] = None, fields: Optional[dict] = None, **kwargs: Any) -> None:
        super(UpdateResponse, self).__init__(**kwargs)
        self.updated = updated
        self.fields = fields

    @schema_property("updated")
    def updated(self) -> Optional[int]:
        return self._property_updated

    @updated.setter
    def updated(self, value: Optional[int]) -> None:
        if value is None:
            self._property_updated = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "updated", six.integer_types)
        self._property_updated = value

    @schema_property("fields")
    def fields(self) -> Optional[dict]:
        return self._property_fields

    @fields.setter
    def fields(self, value: Optional[dict]) -> None:
        if value is None:
            self._property_fields = None
            return
        self.assert_isinstance(value, "fields", (dict,))
        self._property_fields = value


class UpdateBatchRequest(BatchRequest):
    """
    Updates a batch of tasks.
            Headers
            Content type should be 'application/json-lines'.

    """

    _service = "tasks"
    _action = "update_batch"
    _version = "2.13"
    _batched_request_cls = UpdateRequest


class UpdateBatchResponse(Response):
    """
    Response of tasks.update_batch endpoint.

    :param updated: Number of tasks updated (0 or 1)
    :type updated: int
    """

    _service = "tasks"
    _action = "update_batch"
    _version = "2.13"
    _schema = {
        "definitions": {},
        "properties": {
            "updated": {
                "description": "Number of tasks updated (0 or 1)",
                "enum": [0, 1],
                "type": ["integer", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, updated: Optional[int] = None, **kwargs: Any) -> None:
        super(UpdateBatchResponse, self).__init__(**kwargs)
        self.updated = updated

    @schema_property("updated")
    def updated(self) -> Optional[int]:
        return self._property_updated

    @updated.setter
    def updated(self, value: Optional[int]) -> None:
        if value is None:
            self._property_updated = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.assert_isinstance(value, "updated", six.integer_types)
        self._property_updated = value


class ValidateRequest(Request):
    """
    Validate task properties (before create)

    :param name: Task name. Unique within the company.
    :type name: str
    :param tags: User-defined tags list
    :type tags: Sequence[str]
    :param system_tags: System tags list. This field is reserved for system use,
        please don't use it.
    :type system_tags: Sequence[str]
    :param type: Type of task
    :type type: TaskTypeEnum
    :param comment: Free text comment
    :type comment: str
    :param parent: Parent task id Must be a completed task.
    :type parent: str
    :param project: Project ID of the project to which this task is assigned Must
        exist[ab]
    :type project: str
    :param output_dest: Output storage id Must be a reference to an existing
        storage.
    :type output_dest: str
    :param execution: Task execution params
    :type execution: Execution
    :param script: Script info
    :type script: Script
    :param hyperparams: Task hyper params per section
    :type hyperparams: dict
    :param configuration: Task configuration params
    :type configuration: dict
    :param models: Task models
    :type models: TaskModels
    :param container: Docker container parameters
    :type container: dict
    """

    _service = "tasks"
    _action = "validate"
    _version = "2.13"
    _schema = {
        "definitions": {
            "artifact": {
                "properties": {
                    "content_size": {
                        "description": "Raw data length in bytes",
                        "type": "integer",
                    },
                    "display_data": {
                        "description": "User-defined list of key/value pairs, sorted",
                        "items": {"items": {"type": "string"}, "type": "array"},
                        "type": "array",
                    },
                    "hash": {
                        "description": "Hash of entire raw data",
                        "type": "string",
                    },
                    "key": {"description": "Entry key", "type": "string"},
                    "mode": {
                        "$ref": "#/definitions/artifact_mode_enum",
                        "description": "System defined input/output indication",
                    },
                    "timestamp": {
                        "description": "Epoch time when artifact was created",
                        "type": "integer",
                    },
                    "type": {"description": "System defined type", "type": "string"},
                    "type_data": {
                        "$ref": "#/definitions/artifact_type_data",
                        "description": "Additional fields defined by the system",
                    },
                    "uri": {"description": "Raw data location", "type": "string"},
                },
                "required": ["key", "type"],
                "type": "object",
            },
            "artifact_mode_enum": {
                "default": "output",
                "enum": ["input", "output"],
                "type": "string",
            },
            "artifact_type_data": {
                "properties": {
                    "content_type": {
                        "description": "System defined raw data content type",
                        "type": ["string", "null"],
                    },
                    "data_hash": {
                        "description": "Hash of raw data, without any headers or descriptive parts",
                        "type": ["string", "null"],
                    },
                    "preview": {
                        "description": "Description or textual data",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "configuration_item": {
                "properties": {
                    "description": {
                        "description": "The parameter description. Optional",
                        "type": ["string", "null"],
                    },
                    "name": {
                        "description": "Name of the parameter. Should be unique",
                        "type": ["string", "null"],
                    },
                    "type": {
                        "description": "Type of the parameter. Optional",
                        "type": ["string", "null"],
                    },
                    "value": {
                        "description": "Value of the parameter",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "execution": {
                "properties": {
                    "artifacts": {
                        "description": "Task artifacts",
                        "items": {"$ref": "#/definitions/artifact"},
                        "type": ["array", "null"],
                    },
                    "framework": {
                        "description": "Framework related to the task. Case insensitive. Mandatory for Training tasks. ",
                        "type": ["string", "null"],
                    },
                    "model_desc": {
                        "additionalProperties": True,
                        "description": "Json object representing the Model descriptors",
                        "type": ["object", "null"],
                    },
                    "model_labels": {
                        "additionalProperties": {"type": "integer"},
                        "description": "Json object representing the ids of the labels in the model.\n                The keys are the layers' names and the values are the IDs.\n                Not applicable for Register (Import) tasks.\n                Mandatory for Training tasks",
                        "type": ["object", "null"],
                    },
                    "parameters": {
                        "additionalProperties": True,
                        "description": "Json object containing the Task parameters",
                        "type": ["object", "null"],
                    },
                    "queue": {
                        "description": "Queue ID where task was queued.",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "params_item": {
                "properties": {
                    "description": {
                        "description": "The parameter description. Optional",
                        "type": ["string", "null"],
                    },
                    "name": {
                        "description": "Name of the parameter. The combination of section and name should be unique",
                        "type": ["string", "null"],
                    },
                    "section": {
                        "description": "Section that the parameter belongs to",
                        "type": ["string", "null"],
                    },
                    "type": {
                        "description": "Type of the parameter. Optional",
                        "type": ["string", "null"],
                    },
                    "value": {
                        "description": "Value of the parameter",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "script": {
                "properties": {
                    "binary": {
                        "default": "python",
                        "description": "Binary to use when running the script",
                        "type": ["string", "null"],
                    },
                    "branch": {
                        "description": "Repository branch id If not provided and tag not provided, default repository branch is used.",
                        "type": ["string", "null"],
                    },
                    "diff": {
                        "description": "Uncommitted changes found in the repository when task was run",
                        "type": ["string", "null"],
                    },
                    "entry_point": {
                        "description": "Path to execute within the repository",
                        "type": ["string", "null"],
                    },
                    "repository": {
                        "description": "Name of the repository where the script is located",
                        "type": ["string", "null"],
                    },
                    "requirements": {
                        "description": "A JSON object containing requirements strings by key",
                        "type": ["object", "null"],
                    },
                    "tag": {
                        "description": "Repository tag",
                        "type": ["string", "null"],
                    },
                    "version_num": {
                        "description": "Version (changeset) number. Optional (default is head version) Unused if tag is provided.",
                        "type": ["string", "null"],
                    },
                    "working_dir": {
                        "description": "Path to the folder from which to run the script Default - root folder of repository",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "section_params": {
                "additionalProperties": {"$ref": "#/definitions/params_item"},
                "description": "Task section params",
                "type": "object",
            },
            "task_model_item": {
                "properties": {
                    "model": {"description": "The model ID", "type": "string"},
                    "name": {"description": "The task model name", "type": "string"},
                },
                "required": ["name", "model"],
                "type": "object",
            },
            "task_models": {
                "properties": {
                    "input": {
                        "description": "The list of task input models",
                        "items": {"$ref": "#/definitions/task_model_item"},
                        "type": ["array", "null"],
                    },
                    "output": {
                        "description": "The list of task output models",
                        "items": {"$ref": "#/definitions/task_model_item"},
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            },
            "task_type_enum": {
                "enum": [
                    "training",
                    "testing",
                    "inference",
                    "data_processing",
                    "application",
                    "monitor",
                    "controller",
                    "optimizer",
                    "service",
                    "qc",
                    "custom",
                ],
                "type": "string",
            },
        },
        "properties": {
            "comment": {"description": "Free text comment ", "type": "string"},
            "configuration": {
                "additionalProperties": {"$ref": "#/definitions/configuration_item"},
                "description": "Task configuration params",
                "type": "object",
            },
            "container": {
                "type": "object",
                "description": "Docker container parameters",
                "additionalProperties": {"type": ["string", "null"]},
            },
            "execution": {
                "$ref": "#/definitions/execution",
                "description": "Task execution params",
            },
            "hyperparams": {
                "additionalProperties": {"$ref": "#/definitions/section_params"},
                "description": "Task hyper params per section",
                "type": "object",
            },
            "models": {
                "$ref": "#/definitions/task_models",
                "description": "Task models",
            },
            "name": {
                "description": "Task name. Unique within the company.",
                "type": "string",
            },
            "output_dest": {
                "description": "Output storage id Must be a reference to an existing storage.",
                "type": "string",
            },
            "parent": {
                "description": "Parent task id Must be a completed task.",
                "type": "string",
            },
            "project": {
                "description": "Project ID of the project to which this task is assigned Must exist[ab]",
                "type": "string",
            },
            "script": {"$ref": "#/definitions/script", "description": "Script info"},
            "system_tags": {
                "description": "System tags list. This field is reserved for system use, please don't use it.",
                "items": {"type": "string"},
                "type": "array",
            },
            "tags": {
                "description": "User-defined tags list",
                "items": {"type": "string"},
                "type": "array",
            },
            "type": {
                "$ref": "#/definitions/task_type_enum",
                "description": "Type of task",
            },
        },
        "required": ["name", "type"],
        "type": "object",
    }

    def __init__(
        self,
        name: str,
        type: Any,
        tags: Optional[List[str]] = None,
        system_tags: Optional[List[str]] = None,
        comment: Optional[str] = None,
        parent: Optional[str] = None,
        project: Optional[str] = None,
        output_dest: Optional[str] = None,
        execution: Any = None,
        script: Any = None,
        hyperparams: Optional[dict] = None,
        configuration: Optional[dict] = None,
        models: Any = None,
        container: Optional[dict] = None,
        **kwargs: Any
    ) -> None:
        super(ValidateRequest, self).__init__(**kwargs)
        self.name = name
        self.tags = tags
        self.system_tags = system_tags
        self.type = type
        self.comment = comment
        self.parent = parent
        self.project = project
        self.output_dest = output_dest
        self.execution = execution
        self.script = script
        self.hyperparams = hyperparams
        self.configuration = configuration
        self.models = models
        self.container = container

    @schema_property("name")
    def name(self) -> str:
        return self._property_name

    @name.setter
    def name(self, value: str) -> None:
        if value is None:
            self._property_name = None
            return
        self.assert_isinstance(value, "name", six.string_types)
        self._property_name = value

    @schema_property("tags")
    def tags(self) -> Optional[List[str]]:
        return self._property_tags

    @tags.setter
    def tags(self, value: Optional[List[str]]) -> None:
        if value is None:
            self._property_tags = None
            return
        self.assert_isinstance(value, "tags", (list, tuple))
        self.assert_isinstance(value, "tags", six.string_types, is_array=True)
        self._property_tags = value

    @schema_property("system_tags")
    def system_tags(self) -> Optional[List[str]]:
        return self._property_system_tags

    @system_tags.setter
    def system_tags(self, value: Optional[List[str]]) -> None:
        if value is None:
            self._property_system_tags = None
            return
        self.assert_isinstance(value, "system_tags", (list, tuple))
        self.assert_isinstance(value, "system_tags", six.string_types, is_array=True)
        self._property_system_tags = value

    @schema_property("type")
    def type(self) -> Any:
        return self._property_type

    @type.setter
    def type(self, value: Any) -> None:
        if value is None:
            self._property_type = None
            return
        if isinstance(value, six.string_types):
            try:
                value = TaskTypeEnum(value)
            except ValueError:
                pass
        else:
            self.assert_isinstance(value, "type", enum.Enum)
        self._property_type = value

    @schema_property("comment")
    def comment(self) -> Optional[str]:
        return self._property_comment

    @comment.setter
    def comment(self, value: Optional[str]) -> None:
        if value is None:
            self._property_comment = None
            return
        self.assert_isinstance(value, "comment", six.string_types)
        self._property_comment = value

    @schema_property("parent")
    def parent(self) -> Optional[str]:
        return self._property_parent

    @parent.setter
    def parent(self, value: Optional[str]) -> None:
        if value is None:
            self._property_parent = None
            return
        self.assert_isinstance(value, "parent", six.string_types)
        self._property_parent = value

    @schema_property("project")
    def project(self) -> Optional[str]:
        return self._property_project

    @project.setter
    def project(self, value: Optional[str]) -> None:
        if value is None:
            self._property_project = None
            return
        self.assert_isinstance(value, "project", six.string_types)
        self._property_project = value

    @schema_property("output_dest")
    def output_dest(self) -> Optional[str]:
        return self._property_output_dest

    @output_dest.setter
    def output_dest(self, value: Optional[str]) -> None:
        if value is None:
            self._property_output_dest = None
            return
        self.assert_isinstance(value, "output_dest", six.string_types)
        self._property_output_dest = value

    @schema_property("execution")
    def execution(self) -> Any:
        return self._property_execution

    @execution.setter
    def execution(self, value: Any) -> None:
        if value is None:
            self._property_execution = None
            return
        if isinstance(value, dict):
            value = Execution.from_dict(value)
        else:
            self.assert_isinstance(value, "execution", Execution)
        self._property_execution = value

    @schema_property("hyperparams")
    def hyperparams(self) -> Optional[dict]:
        return self._property_hyperparams

    @hyperparams.setter
    def hyperparams(self, value: Optional[dict]) -> None:
        if value is None:
            self._property_hyperparams = None
            return
        self.assert_isinstance(value, "hyperparams", dict)
        self.assert_isinstance(value.keys(), "hyperparams_keys", six.string_types, is_array=True)
        self.assert_isinstance(value.values(), "hyperparams_values", (SectionParams, dict), is_array=True)
        value = dict(((k, SectionParams(**v) if isinstance(v, dict) else v) for (k, v) in value.items()))
        self._property_hyperparams = value

    @schema_property("configuration")
    def configuration(self) -> Optional[dict]:
        return self._property_configuration

    @configuration.setter
    def configuration(self, value: Optional[dict]) -> None:
        if value is None:
            self._property_configuration = None
            return
        self.assert_isinstance(value, "configuration", dict)
        self.assert_isinstance(value.keys(), "configuration_keys", six.string_types, is_array=True)
        self.assert_isinstance(
            value.values(),
            "configuration_values",
            (ConfigurationItem, dict),
            is_array=True,
        )
        value = dict(((k, ConfigurationItem(**v) if isinstance(v, dict) else v) for (k, v) in value.items()))
        self._property_configuration = value

    @schema_property("script")
    def script(self) -> Any:
        return self._property_script

    @script.setter
    def script(self, value: Any) -> None:
        if value is None:
            self._property_script = None
            return
        if isinstance(value, dict):
            value = Script.from_dict(value)
        else:
            self.assert_isinstance(value, "script", Script)
        self._property_script = value

    @schema_property("models")
    def models(self) -> Any:
        return self._property_models

    @models.setter
    def models(self, value: Any) -> None:
        if value is None:
            self._property_models = None
            return
        if isinstance(value, dict):
            value = TaskModels.from_dict(value)
        else:
            self.assert_isinstance(value, "models", TaskModels)
        self._property_models = value

    @schema_property("container")
    def container(self) -> Optional[dict]:
        return self._property_container

    @container.setter
    def container(self, value: Optional[dict]) -> None:
        if value is None:
            self._property_container = None
            return
        self.assert_isinstance(value, "container", dict)
        self._property_container = value


class ValidateResponse(Response):
    """
    Response of tasks.validate endpoint.

    """

    _service = "tasks"
    _action = "validate"
    _version = "2.13"
    _schema = {"additionalProperties": False, "definitions": {}, "type": "object"}


response_mapping = {
    GetByIdRequest: GetByIdResponse,
    GetAllRequest: GetAllResponse,
    GetTypesRequest: GetTypesResponse,
    CloneRequest: CloneResponse,
    AddOrUpdateModelRequest: AddOrUpdateModelResponse,
    DeleteModelsRequest: DeleteModelsResponse,
    CreateRequest: CreateResponse,
    ValidateRequest: ValidateResponse,
    UpdateRequest: UpdateResponse,
    UpdateBatchRequest: UpdateBatchResponse,
    EditRequest: EditResponse,
    ResetRequest: ResetResponse,
    ResetManyRequest: ResetManyResponse,
    DeleteManyRequest: DeleteManyResponse,
    DeleteRequest: DeleteResponse,
    ArchiveRequest: ArchiveResponse,
    ArchiveManyRequest: ArchiveManyResponse,
    StartedRequest: StartedResponse,
    StopRequest: StopResponse,
    StopManyRequest: StopManyResponse,
    StoppedRequest: StoppedResponse,
    FailedRequest: FailedResponse,
    CloseRequest: CloseResponse,
    PublishRequest: PublishResponse,
    PublishManyRequest: PublishManyResponse,
    EnqueueRequest: EnqueueResponse,
    EnqueueManyRequest: EnqueueManyResponse,
    DequeueRequest: DequeueResponse,
    SetRequirementsRequest: SetRequirementsResponse,
    CompletedRequest: CompletedResponse,
    PingRequest: PingResponse,
    AddOrUpdateArtifactsRequest: AddOrUpdateArtifactsResponse,
    MakePublicRequest: MakePublicResponse,
    MakePrivateRequest: MakePrivateResponse,
    DeleteArtifactsRequest: DeleteArtifactsResponse,
    GetHyperParamsRequest: GetHyperParamsResponse,
    EditHyperParamsRequest: EditHyperParamsResponse,
    DeleteHyperParamsRequest: DeleteHyperParamsResponse,
    GetConfigurationsRequest: GetConfigurationsResponse,
    GetConfigurationNamesRequest: GetConfigurationNamesResponse,
    EditConfigurationRequest: EditConfigurationResponse,
    DeleteConfigurationRequest: DeleteConfigurationResponse,
    MoveRequest: MoveResponse,
}
