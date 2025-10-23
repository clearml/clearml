"""
dataviews service

Dataview service
"""
import six
from datetime import datetime
import enum

from dateutil.parser import parse as parse_datetime

from clearml.backend_api.session import (
    Request,
    Response,
    NonStrictDataModel,
    schema_property,
    StringEnum,
)


class FilterByRoiEnum(StringEnum):
    disabled = "disabled"
    no_rois = "no_rois"
    label_rules = "label_rules"


class FilterLabelRule(NonStrictDataModel):
    """
    :param label: Lucene format query (see lucene query syntax). Default search
        field is label.keyword and default operator is AND, so searching for:
        'Bus Stop' Blue
        is equivalent to:
        Label.keyword:'Bus Stop' AND label.keyword:'Blue'
    :type label: str
    :param count_range: Range of times ROI appears in the frame (min, max). -1 for
        not applicable. Both integers must be larger than or equal to -1. 2nd integer
        (max) must be either -1 or larger than or equal to the 1st integer (min)
    :type count_range: Sequence[int]
    :param conf_range: Range of ROI confidence level in the frame (min, max). -1
        for not applicable Both min and max can be either -1 or positive. 2nd number
        (max) must be either -1 or larger than or equal to the 1st number (min)
    :type conf_range: Sequence[float]
    :param must_not: If set then the label must not exist or lucene query must not
        be true. The default value is false
    :type must_not: bool
    """

    _schema = {
        "properties": {
            "conf_range": {
                "description": (
                    "Range of ROI confidence level in the frame (min, max). -1 for not applicable\n            Both min"
                    " and max can be either -1 or positive.\n            2nd number (max) must be either -1 or larger"
                    " than or equal to the 1st number (min)"
                ),
                "items": {"type": "number"},
                "maxItems": 2,
                "minItems": 1,
                "type": "array",
            },
            "count_range": {
                "description": (
                    "Range of times ROI appears in the frame (min, max). -1 for not applicable.\n            Both"
                    " integers must be larger than or equal to -1.\n            2nd integer (max) must be either -1 or"
                    " larger than or equal to the 1st integer (min)"
                ),
                "items": {"type": "integer"},
                "maxItems": 2,
                "minItems": 1,
                "type": "array",
            },
            "label": {
                "description": (
                    "Lucene format query (see lucene query syntax).\nDefault search field is label.keyword and default"
                    " operator is AND, so searching for:\n\n'Bus Stop' Blue\n\nis equivalent to:\n\nLabel.keyword:'Bus"
                    " Stop' AND label.keyword:'Blue'"
                ),
                "type": "string",
            },
            "must_not": {
                "default": False,
                "description": (
                    "If set then the label must not exist or lucene query must not be true.\n            The default"
                    " value is false"
                ),
                "type": "boolean",
            },
        },
        "required": ["label"],
        "type": "object",
    }

    def __init__(
        self, label, count_range=None, conf_range=None, must_not=False, **kwargs
    ):
        super(FilterLabelRule, self).__init__(**kwargs)
        self.label = label
        self.count_range = count_range
        self.conf_range = conf_range
        self.must_not = must_not

    @schema_property("label")
    def label(self):
        return self._property_label

    @label.setter
    def label(self, value):
        if value is None:
            self._property_label = None
            return

        self.assert_isinstance(value, "label", six.string_types)
        self._property_label = value

    @schema_property("count_range")
    def count_range(self):
        return self._property_count_range

    @count_range.setter
    def count_range(self, value):
        if value is None:
            self._property_count_range = None
            return

        self.assert_isinstance(value, "count_range", (list, tuple))
        value = [
            int(v) if isinstance(v, float) and v.is_integer() else v for v in value
        ]

        self.assert_isinstance(value, "count_range", six.integer_types, is_array=True)
        self._property_count_range = value

    @schema_property("conf_range")
    def conf_range(self):
        return self._property_conf_range

    @conf_range.setter
    def conf_range(self, value):
        if value is None:
            self._property_conf_range = None
            return

        self.assert_isinstance(value, "conf_range", (list, tuple))

        self.assert_isinstance(
            value, "conf_range", six.integer_types + (float,), is_array=True
        )
        self._property_conf_range = value

    @schema_property("must_not")
    def must_not(self):
        return self._property_must_not

    @must_not.setter
    def must_not(self, value):
        if value is None:
            self._property_must_not = None
            return

        self.assert_isinstance(value, "must_not", (bool,))
        self._property_must_not = value


class FilterRule(NonStrictDataModel):
    """
    :param label_rules: List of FilterLabelRule ('AND' connection)
        disabled - No filtering by ROIs. Select all frames, even if they don't have
        ROIs (all frames)
        no_rois - Select only frames without ROIs (empty frames)
        label_rules - Select frames according to label rules
    :type label_rules: Sequence[FilterLabelRule]
    :param filter_by_roi: Type of filter. Optional, the default value is
        'label_rules'
    :type filter_by_roi: FilterByRoiEnum
    :param frame_query: Frame filter, in Lucene query syntax
    :type frame_query: str
    :param sources_query: Sources filter, in Lucene query syntax. Filters sources
        in each frame.
    :type sources_query: str
    :param dataset: Dataset ID. Must be a dataset which is in the task's view. If
        set to '*' all datasets in View are used.
    :type dataset: str
    :param version: Dataset version to apply rule to. Must belong to the dataset
        and be in the task's view. If set to '*' all version of the datasets in View
        are used.
    :type version: str
    :param weight: Rule weight. Default is 1
    :type weight: float
    """

    _schema = {
        "properties": {
            "dataset": {
                "description": (
                    "Dataset ID. Must be a dataset which is in the task's view. If set to '*' all datasets in View "
                    "are used."
                ),
                "type": "string",
            },
            "filter_by_roi": {
                "description": "Type of filter. Optional, the default value is 'label_rules'",
                "oneOf": [
                    {"$ref": "#/definitions/filter_by_roi_enum"},
                    {"type": "null"},
                ],
            },
            "frame_query": {
                "description": "Frame filter, in Lucene query syntax",
                "type": ["string", "null"],
            },
            "label_rules": {
                "description": (
                    "List of FilterLabelRule ('AND' connection)\n\ndisabled - No filtering by ROIs. Select all frames,"
                    " even if they don't have ROIs (all frames)\n\nno_rois - Select only frames without ROIs (empty"
                    " frames)\n\nlabel_rules - Select frames according to label rules"
                ),
                "items": {"$ref": "#/definitions/filter_label_rule"},
                "type": ["array", "null"],
            },
            "sources_query": {
                "description": "Sources filter, in Lucene query syntax. Filters sources in each frame.",
                "type": ["string", "null"],
            },
            "version": {
                "description": (
                    "Dataset version to apply rule to. Must belong to the dataset and be in the task's view. If set to"
                    " '*' all version of the datasets in View are used."
                ),
                "type": "string",
            },
            "weight": {"description": "Rule weight. Default is 1", "type": "number"},
        },
        "required": ["dataset"],
        "type": "object",
    }

    def __init__(
        self,
        dataset,
        label_rules=None,
        filter_by_roi=None,
        frame_query=None,
        sources_query=None,
        version=None,
        weight=None,
        **kwargs
    ):
        super(FilterRule, self).__init__(**kwargs)
        self.label_rules = label_rules
        self.filter_by_roi = filter_by_roi
        self.frame_query = frame_query
        self.sources_query = sources_query
        self.dataset = dataset
        self.version = version
        self.weight = weight

    @schema_property("label_rules")
    def label_rules(self):
        return self._property_label_rules

    @label_rules.setter
    def label_rules(self, value):
        if value is None:
            self._property_label_rules = None
            return

        self.assert_isinstance(value, "label_rules", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [
                FilterLabelRule.from_dict(v) if isinstance(v, dict) else v
                for v in value
            ]
        else:
            self.assert_isinstance(value, "label_rules", FilterLabelRule, is_array=True)
        self._property_label_rules = value

    @schema_property("filter_by_roi")
    def filter_by_roi(self):
        return self._property_filter_by_roi

    @filter_by_roi.setter
    def filter_by_roi(self, value):
        if value is None:
            self._property_filter_by_roi = None
            return
        if isinstance(value, six.string_types):
            try:
                value = FilterByRoiEnum(value)
            except ValueError:
                pass
        else:
            self.assert_isinstance(value, "filter_by_roi", enum.Enum)
        self._property_filter_by_roi = value

    @schema_property("frame_query")
    def frame_query(self):
        return self._property_frame_query

    @frame_query.setter
    def frame_query(self, value):
        if value is None:
            self._property_frame_query = None
            return

        self.assert_isinstance(value, "frame_query", six.string_types)
        self._property_frame_query = value

    @schema_property("sources_query")
    def sources_query(self):
        return self._property_sources_query

    @sources_query.setter
    def sources_query(self, value):
        if value is None:
            self._property_sources_query = None
            return

        self.assert_isinstance(value, "sources_query", six.string_types)
        self._property_sources_query = value

    @schema_property("dataset")
    def dataset(self):
        return self._property_dataset

    @dataset.setter
    def dataset(self, value):
        if value is None:
            self._property_dataset = None
            return

        self.assert_isinstance(value, "dataset", six.string_types)
        self._property_dataset = value

    @schema_property("version")
    def version(self):
        return self._property_version

    @version.setter
    def version(self, value):
        if value is None:
            self._property_version = None
            return

        self.assert_isinstance(value, "version", six.string_types)
        self._property_version = value

    @schema_property("weight")
    def weight(self):
        return self._property_weight

    @weight.setter
    def weight(self, value):
        if value is None:
            self._property_weight = None
            return

        self.assert_isinstance(value, "weight", six.integer_types + (float,))
        self._property_weight = value


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

    def __init__(self, pattern=None, fields=None, **kwargs):
        super(MultiFieldPatternData, self).__init__(**kwargs)
        self.pattern = pattern
        self.fields = fields

    @schema_property("pattern")
    def pattern(self):
        return self._property_pattern

    @pattern.setter
    def pattern(self, value):
        if value is None:
            self._property_pattern = None
            return

        self.assert_isinstance(value, "pattern", six.string_types)
        self._property_pattern = value

    @schema_property("fields")
    def fields(self):
        return self._property_fields

    @fields.setter
    def fields(self, value):
        if value is None:
            self._property_fields = None
            return

        self.assert_isinstance(value, "fields", (list, tuple))

        self.assert_isinstance(value, "fields", six.string_types, is_array=True)
        self._property_fields = value


class DataviewEntry(NonStrictDataModel):
    """
    :param version: Version id of a version belonging to the dataset
    :type version: str
    :param dataset: Existing Dataset id
    :type dataset: str
    :param merge_with: Version ID to merge with
    :type merge_with: str
    """

    _schema = {
        "properties": {
            "dataset": {"description": "Existing Dataset id", "type": "string"},
            "merge_with": {"description": "Version ID to merge with", "type": "string"},
            "version": {
                "description": "Version id of a version belonging to the dataset",
                "type": "string",
            },
        },
        "required": ["dataset", "version"],
        "type": "object",
    }

    def __init__(self, version, dataset, merge_with=None, **kwargs):
        super(DataviewEntry, self).__init__(**kwargs)
        self.version = version
        self.dataset = dataset
        self.merge_with = merge_with

    @schema_property("version")
    def version(self):
        return self._property_version

    @version.setter
    def version(self, value):
        if value is None:
            self._property_version = None
            return

        self.assert_isinstance(value, "version", six.string_types)
        self._property_version = value

    @schema_property("dataset")
    def dataset(self):
        return self._property_dataset

    @dataset.setter
    def dataset(self, value):
        if value is None:
            self._property_dataset = None
            return

        self.assert_isinstance(value, "dataset", six.string_types)
        self._property_dataset = value

    @schema_property("merge_with")
    def merge_with(self):
        return self._property_merge_with

    @merge_with.setter
    def merge_with(self, value):
        if value is None:
            self._property_merge_with = None
            return

        self.assert_isinstance(value, "merge_with", six.string_types)
        self._property_merge_with = value


class Dataview(NonStrictDataModel):
    """
    :param id: Dataview ID
    :type id: str
    :param name: Dataview name
    :type name: str
    :param created: Dataview creation time (UTC)
    :type created: datetime.datetime
    :param description: Dataview description
    :type description: str
    :param user: Associated user id
    :type user: str
    :param company: Company id
    :type company: str
    :param project: Project ID of the project to which this task is assigned
    :type project: str
    :param tags: User-defined tags list
    :type tags: Sequence[str]
    :param system_tags: System tags list. This field is reserved for system use,
        please don't use it.
    :type system_tags: Sequence[str]
    :param filters: List of FilterRule ('OR' connection)
    :type filters: Sequence[FilterRule]
    :param output_rois: 'all_in_frame' - all rois for a frame are returned
        'only_filtered' - only rois which led this frame to be selected 'frame_per_roi'
        - single roi per frame. Frame can be returned multiple times with a different
        roi each time. Note: this should be used for Training tasks only Note:
        frame_per_roi implies that only filtered rois will be returned
    :type output_rois: OutputRoisEnum
    :param versions: List of dataview entries. All tasks must have at least one
        dataview.
    :type versions: Sequence[DataviewEntry]
    :param iteration: Iteration parameters. Not applicable for register (import)
        tasks.
    :type iteration: Iteration
    :param augmentation: Augmentation parameters. Only for training and testing
        tasks.
    :type augmentation: Augmentation
    :param mapping: Mapping parameters
    :type mapping: Mapping
    :param labels_enumeration: Labels enumerations, specifies numbers to be
        assigned to ROI labels when getting frames
    :type labels_enumeration: dict
    """

    _schema = {
        "properties": {
            "augmentation": {
                "$ref": "#/definitions/augmentation",
                "description": "Augmentation parameters. Only for training and testing tasks.",
            },
            "company": {"description": "Company id", "type": "string"},
            "created": {
                "description": "Dataview creation time (UTC) ",
                "format": "date-time",
                "type": "string",
            },
            "description": {"description": "Dataview description", "type": "string"},
            "filters": {
                "description": "List of FilterRule ('OR' connection)",
                "items": {"$ref": "#/definitions/filter_rule"},
                "type": "array",
            },
            "id": {"description": "Dataview ID", "type": "string"},
            "iteration": {
                "$ref": "#/definitions/iteration",
                "description": "Iteration parameters. Not applicable for register (import) tasks.",
            },
            "labels_enumeration": {
                "additionalProperties": {"type": "integer"},
                "description": (
                    "Labels enumerations, specifies numbers to be assigned to ROI labels when getting frames"
                ),
                "type": "object",
            },
            "mapping": {
                "$ref": "#/definitions/mapping",
                "description": "Mapping parameters",
            },
            "name": {"description": "Dataview name", "type": "string"},
            "output_rois": {
                "$ref": "#/definitions/output_rois_enum",
                "default": "all_in_frame",
                "description": (
                    "'all_in_frame' - all rois for a frame are returned\n                'only_filtered' - only rois"
                    " which led this frame to be selected\n                'frame_per_roi' - single roi per frame."
                    " Frame can be returned multiple times with a different roi each time.\n                Note: this"
                    " should be used for Training tasks only\n                Note: frame_per_roi implies that only"
                    " filtered rois will be returned\n                "
                ),
            },
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
            "user": {"description": "Associated user id", "type": "string"},
            "versions": {
                "description": "List of dataview entries. All tasks must have at least one dataview.",
                "items": {"$ref": "#/definitions/dataview_entry"},
                "type": "array",
            },
            "status": {
                "description": "dataview status",
                "enum": ["draft", "published"],
                "type": "string",
            },
        },
        "required": ["id", "name"],
        "type": "object",
    }

    def __init__(
        self,
        id,
        name,
        created=None,
        description=None,
        user=None,
        company=None,
        project=None,
        tags=None,
        system_tags=None,
        filters=None,
        output_rois="all_in_frame",
        versions=None,
        iteration=None,
        augmentation=None,
        mapping=None,
        labels_enumeration=None,
        status=None,
        **kwargs
    ):
        super(Dataview, self).__init__(**kwargs)
        self.id = id
        self.name = name
        self.created = created
        self.description = description
        self.user = user
        self.company = company
        self.project = project
        self.tags = tags
        self.system_tags = system_tags
        self.filters = filters
        self.output_rois = output_rois
        self.versions = versions
        self.iteration = iteration
        self.augmentation = augmentation
        self.mapping = mapping
        self.labels_enumeration = labels_enumeration
        self.status = status

    @schema_property("id")
    def id(self):
        return self._property_id

    @id.setter
    def id(self, value):
        if value is None:
            self._property_id = None
            return

        self.assert_isinstance(value, "id", six.string_types)
        self._property_id = value

    @schema_property("name")
    def name(self):
        return self._property_name

    @name.setter
    def name(self, value):
        if value is None:
            self._property_name = None
            return

        self.assert_isinstance(value, "name", six.string_types)
        self._property_name = value

    @schema_property("created")
    def created(self):
        return self._property_created

    @created.setter
    def created(self, value):
        if value is None:
            self._property_created = None
            return

        self.assert_isinstance(value, "created", six.string_types + (datetime,))
        if not isinstance(value, datetime):
            value = parse_datetime(value)
        self._property_created = value

    @schema_property("description")
    def description(self):
        return self._property_description

    @description.setter
    def description(self, value):
        if value is None:
            self._property_description = None
            return

        self.assert_isinstance(value, "description", six.string_types)
        self._property_description = value

    @schema_property("user")
    def user(self):
        return self._property_user

    @user.setter
    def user(self, value):
        if value is None:
            self._property_user = None
            return

        self.assert_isinstance(value, "user", six.string_types)
        self._property_user = value

    @schema_property("company")
    def company(self):
        return self._property_company

    @company.setter
    def company(self, value):
        if value is None:
            self._property_company = None
            return

        self.assert_isinstance(value, "company", six.string_types)
        self._property_company = value

    @schema_property("project")
    def project(self):
        return self._property_project

    @project.setter
    def project(self, value):
        if value is None:
            self._property_project = None
            return

        self.assert_isinstance(value, "project", six.string_types)
        self._property_project = value

    @schema_property("tags")
    def tags(self):
        return self._property_tags

    @tags.setter
    def tags(self, value):
        if value is None:
            self._property_tags = None
            return

        self.assert_isinstance(value, "tags", (list, tuple))

        self.assert_isinstance(value, "tags", six.string_types, is_array=True)
        self._property_tags = value

    @schema_property("system_tags")
    def system_tags(self):
        return self._property_system_tags

    @system_tags.setter
    def system_tags(self, value):
        if value is None:
            self._property_system_tags = None
            return

        self.assert_isinstance(value, "system_tags", (list, tuple))

        self.assert_isinstance(value, "system_tags", six.string_types, is_array=True)
        self._property_system_tags = value

    @schema_property("filters")
    def filters(self):
        return self._property_filters

    @filters.setter
    def filters(self, value):
        if value is None:
            self._property_filters = None
            return

        self.assert_isinstance(value, "filters", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [
                FilterRule.from_dict(v) if isinstance(v, dict) else v for v in value
            ]
        else:
            self.assert_isinstance(value, "filters", FilterRule, is_array=True)
        self._property_filters = value

    @schema_property("output_rois")
    def output_rois(self):
        return self._property_output_rois

    @output_rois.setter
    def output_rois(self, value):
        if value is None:
            self._property_output_rois = None
            return
        if isinstance(value, six.string_types):
            try:
                value = OutputRoisEnum(value)
            except ValueError:
                pass
        else:
            self.assert_isinstance(value, "output_rois", enum.Enum)
        self._property_output_rois = value

    @schema_property("versions")
    def versions(self):
        return self._property_versions

    @versions.setter
    def versions(self, value):
        if value is None:
            self._property_versions = None
            return

        self.assert_isinstance(value, "versions", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [
                DataviewEntry.from_dict(v) if isinstance(v, dict) else v for v in value
            ]
        else:
            self.assert_isinstance(value, "versions", DataviewEntry, is_array=True)
        self._property_versions = value

    @schema_property("iteration")
    def iteration(self):
        return self._property_iteration

    @iteration.setter
    def iteration(self, value):
        if value is None:
            self._property_iteration = None
            return
        if isinstance(value, dict):
            value = Iteration.from_dict(value)
        else:
            self.assert_isinstance(value, "iteration", Iteration)
        self._property_iteration = value

    @schema_property("augmentation")
    def augmentation(self):
        return self._property_augmentation

    @augmentation.setter
    def augmentation(self, value):
        if value is None:
            self._property_augmentation = None
            return
        if isinstance(value, dict):
            value = Augmentation.from_dict(value)
        else:
            self.assert_isinstance(value, "augmentation", Augmentation)
        self._property_augmentation = value

    @schema_property("mapping")
    def mapping(self):
        return self._property_mapping

    @mapping.setter
    def mapping(self, value):
        if value is None:
            self._property_mapping = None
            return
        if isinstance(value, dict):
            value = Mapping.from_dict(value)
        else:
            self.assert_isinstance(value, "mapping", Mapping)
        self._property_mapping = value

    @schema_property("labels_enumeration")
    def labels_enumeration(self):
        return self._property_labels_enumeration

    @labels_enumeration.setter
    def labels_enumeration(self, value):
        if value is None:
            self._property_labels_enumeration = None
            return

        self.assert_isinstance(value, "labels_enumeration", (dict,))
        self._property_labels_enumeration = value

    @schema_property("status")
    def status(self):
        return self._property_status

    @status.setter
    def status(self, value):
        if value is None:
            self._property_status = None
            return

        self.assert_isinstance(value, "status", six.string_types)
        self._property_status = value


class OutputRoisEnum(StringEnum):
    all_in_frame = "all_in_frame"
    only_filtered = "only_filtered"
    frame_per_roi = "frame_per_roi"


class Jump(NonStrictDataModel):
    """
    :param time: Max time in milliseconds between frames
    :type time: int
    """

    _schema = {
        "properties": {
            "time": {
                "description": "Max time in milliseconds between frames",
                "type": ["integer", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, time=None, **kwargs):
        super(Jump, self).__init__(**kwargs)
        self.time = time

    @schema_property("time")
    def time(self):
        return self._property_time

    @time.setter
    def time(self, value):
        if value is None:
            self._property_time = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "time", six.integer_types)
        self._property_time = value


class AugmentationSet(NonStrictDataModel):
    """
    :param cls: Augmentation class
    :type cls: str
    :param types: Augmentation type
    :type types: Sequence[str]
    :param strength: Augmentation strength. Range [0,).
    :type strength: float
    :param arguments: Arguments dictionary per custom augmentation type.
    :type arguments: dict
    """

    _schema = {
        "properties": {
            "arguments": {
                "additionalProperties": {
                    "additionalProperties": True,
                    "type": "object",
                },
                "description": "Arguments dictionary per custom augmentation type.",
                "type": ["object", "null"],
            },
            "cls": {"description": "Augmentation class", "type": ["string", "null"]},
            "strength": {
                "description": "Augmentation strength. Range [0,).",
                "minimum": 0,
                "type": ["number", "null"],
            },
            "types": {
                "description": "Augmentation type",
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, cls=None, types=None, strength=None, arguments=None, **kwargs):
        super(AugmentationSet, self).__init__(**kwargs)
        self.cls = cls
        self.types = types
        self.strength = strength
        self.arguments = arguments

    @schema_property("cls")
    def cls(self):
        return self._property_cls

    @cls.setter
    def cls(self, value):
        if value is None:
            self._property_cls = None
            return

        self.assert_isinstance(value, "cls", six.string_types)
        self._property_cls = value

    @schema_property("types")
    def types(self):
        return self._property_types

    @types.setter
    def types(self, value):
        if value is None:
            self._property_types = None
            return

        self.assert_isinstance(value, "types", (list, tuple))

        self.assert_isinstance(value, "types", six.string_types, is_array=True)
        self._property_types = value

    @schema_property("strength")
    def strength(self):
        return self._property_strength

    @strength.setter
    def strength(self, value):
        if value is None:
            self._property_strength = None
            return

        self.assert_isinstance(value, "strength", six.integer_types + (float,))
        self._property_strength = value

    @schema_property("arguments")
    def arguments(self):
        return self._property_arguments

    @arguments.setter
    def arguments(self, value):
        if value is None:
            self._property_arguments = None
            return

        self.assert_isinstance(value, "arguments", (dict,))
        self._property_arguments = value


class Augmentation(NonStrictDataModel):
    """
    :param sets: List of augmentation sets
    :type sets: Sequence[AugmentationSet]
    :param crop_around_rois: Crop image data around all frame ROIs
    :type crop_around_rois: bool
    """

    _schema = {
        "properties": {
            "crop_around_rois": {
                "description": "Crop image data around all frame ROIs",
                "type": ["boolean", "null"],
            },
            "sets": {
                "description": "List of augmentation sets",
                "items": {"$ref": "#/definitions/augmentation_set"},
                "type": ["array", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, sets=None, crop_around_rois=None, **kwargs):
        super(Augmentation, self).__init__(**kwargs)
        self.sets = sets
        self.crop_around_rois = crop_around_rois

    @schema_property("sets")
    def sets(self):
        return self._property_sets

    @sets.setter
    def sets(self, value):
        if value is None:
            self._property_sets = None
            return

        self.assert_isinstance(value, "sets", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [
                AugmentationSet.from_dict(v) if isinstance(v, dict) else v
                for v in value
            ]
        else:
            self.assert_isinstance(value, "sets", AugmentationSet, is_array=True)
        self._property_sets = value

    @schema_property("crop_around_rois")
    def crop_around_rois(self):
        return self._property_crop_around_rois

    @crop_around_rois.setter
    def crop_around_rois(self, value):
        if value is None:
            self._property_crop_around_rois = None
            return

        self.assert_isinstance(value, "crop_around_rois", (bool,))
        self._property_crop_around_rois = value


class IterationOrderEnum(StringEnum):
    sequential = "sequential"
    random = "random"


class Iteration(NonStrictDataModel):
    """
    Sequential Iteration API configuration

    :param order: Input frames order. Values: 'sequential', 'random' In Sequential
        mode frames will be returned according to the order in which the frames were
        added to the dataset.
    :type order: IterationOrderEnum
    :param jump: Jump entry
    :type jump: Jump
    :param min_sequence: Length (in ms) of video clips to return. This is used in
        random order, and in sequential order only if jumping is provided and only for
        video frames
    :type min_sequence: int
    :param infinite: Infinite iteration
    :type infinite: bool
    :param limit: Maximum frames per task. If not passed, frames will end when no
        more matching frames are found, unless infinite is True.
    :type limit: int
    :param random_seed: Random seed used when iterating over the dataview
    :type random_seed: int
    """

    _schema = {
        "description": "Sequential Iteration API configuration",
        "properties": {
            "infinite": {
                "description": "Infinite iteration",
                "type": ["boolean", "null"],
            },
            "jump": {
                "description": "Jump entry",
                "oneOf": [{"$ref": "#/definitions/jump"}, {"type": "null"}],
            },
            "limit": {
                "description": (
                    "Maximum frames per task. If not passed, frames will end when no more matching frames are found,"
                    " unless infinite is True."
                ),
                "type": ["integer", "null"],
            },
            "min_sequence": {
                "description": (
                    "Length (in ms) of video clips to return. This is used in random order, and in sequential order"
                    " only if jumping is provided and only for video frames"
                ),
                "type": ["integer", "null"],
            },
            "order": {
                "description": (
                    "\n                Input frames order. Values: 'sequential', 'random'\n                In"
                    " Sequential mode frames will be returned according to the order in which the frames were added to"
                    " the dataset."
                ),
                "oneOf": [
                    {"$ref": "#/definitions/iteration_order_enum"},
                    {"type": "null"},
                ],
            },
            "random_seed": {
                "description": "Random seed used when iterating over the dataview",
                "type": ["integer", "null"],
            },
        },
        "type": "object",
    }

    def __init__(
        self,
        order=None,
        jump=None,
        min_sequence=None,
        infinite=None,
        limit=None,
        random_seed=None,
        **kwargs
    ):
        super(Iteration, self).__init__(**kwargs)
        self.order = order
        self.jump = jump
        self.min_sequence = min_sequence
        self.infinite = infinite
        self.limit = limit
        self.random_seed = random_seed

    @schema_property("order")
    def order(self):
        return self._property_order

    @order.setter
    def order(self, value):
        if value is None:
            self._property_order = None
            return
        if isinstance(value, six.string_types):
            try:
                value = IterationOrderEnum(value)
            except ValueError:
                pass
        else:
            self.assert_isinstance(value, "order", enum.Enum)
        self._property_order = value

    @schema_property("jump")
    def jump(self):
        return self._property_jump

    @jump.setter
    def jump(self, value):
        if value is None:
            self._property_jump = None
            return
        if isinstance(value, dict):
            value = Jump.from_dict(value)
        else:
            self.assert_isinstance(value, "jump", Jump)
        self._property_jump = value

    @schema_property("min_sequence")
    def min_sequence(self):
        return self._property_min_sequence

    @min_sequence.setter
    def min_sequence(self, value):
        if value is None:
            self._property_min_sequence = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "min_sequence", six.integer_types)
        self._property_min_sequence = value

    @schema_property("infinite")
    def infinite(self):
        return self._property_infinite

    @infinite.setter
    def infinite(self, value):
        if value is None:
            self._property_infinite = None
            return

        self.assert_isinstance(value, "infinite", (bool,))
        self._property_infinite = value

    @schema_property("limit")
    def limit(self):
        return self._property_limit

    @limit.setter
    def limit(self, value):
        if value is None:
            self._property_limit = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "limit", six.integer_types)
        self._property_limit = value

    @schema_property("random_seed")
    def random_seed(self):
        return self._property_random_seed

    @random_seed.setter
    def random_seed(self, value):
        if value is None:
            self._property_random_seed = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "random_seed", six.integer_types)
        self._property_random_seed = value


class LabelSource(NonStrictDataModel):
    """
    :param labels: List of source labels (AND connection). '*' indicates any label.
        Labels must exist in at least one of the dataset versions in the task's view
    :type labels: Sequence[str]
    :param dataset: Source dataset id. '*' for all datasets in view
    :type dataset: str
    :param version: Source dataset version id. Default is '*' (for all versions in
        dataset in the view) Version must belong to the selected dataset, and must be
        in the task's view[i]
    :type version: str
    """

    _schema = {
        "properties": {
            "dataset": {
                "description": "Source dataset id. '*' for all datasets in view",
                "type": ["string", "null"],
            },
            "labels": {
                "description": (
                    "List of source labels (AND connection). '*' indicates any label. Labels must exist in at least one"
                    " of the dataset versions in the task's view"
                ),
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
            "version": {
                "description": (
                    "Source dataset version id. Default is '*' (for all versions in dataset in the view) Version must"
                    " belong to the selected dataset, and must be in the task's view[i]"
                ),
                "type": ["string", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, labels=None, dataset=None, version=None, **kwargs):
        super(LabelSource, self).__init__(**kwargs)
        self.labels = labels
        self.dataset = dataset
        self.version = version

    @schema_property("labels")
    def labels(self):
        return self._property_labels

    @labels.setter
    def labels(self, value):
        if value is None:
            self._property_labels = None
            return

        self.assert_isinstance(value, "labels", (list, tuple))

        self.assert_isinstance(value, "labels", six.string_types, is_array=True)
        self._property_labels = value

    @schema_property("dataset")
    def dataset(self):
        return self._property_dataset

    @dataset.setter
    def dataset(self, value):
        if value is None:
            self._property_dataset = None
            return

        self.assert_isinstance(value, "dataset", six.string_types)
        self._property_dataset = value

    @schema_property("version")
    def version(self):
        return self._property_version

    @version.setter
    def version(self, value):
        if value is None:
            self._property_version = None
            return

        self.assert_isinstance(value, "version", six.string_types)
        self._property_version = value


class MappingRule(NonStrictDataModel):
    """
    :param source: Source label info
    :type source: LabelSource
    :param target: Target label name
    :type target: str
    """

    _schema = {
        "properties": {
            "source": {
                "description": "Source label info",
                "oneOf": [{"$ref": "#/definitions/label_source"}, {"type": "null"}],
            },
            "target": {"description": "Target label name", "type": ["string", "null"]},
        },
        "type": "object",
    }

    def __init__(self, source=None, target=None, **kwargs):
        super(MappingRule, self).__init__(**kwargs)
        self.source = source
        self.target = target

    @schema_property("source")
    def source(self):
        return self._property_source

    @source.setter
    def source(self, value):
        if value is None:
            self._property_source = None
            return
        if isinstance(value, dict):
            value = LabelSource.from_dict(value)
        else:
            self.assert_isinstance(value, "source", LabelSource)
        self._property_source = value

    @schema_property("target")
    def target(self):
        return self._property_target

    @target.setter
    def target(self, value):
        if value is None:
            self._property_target = None
            return

        self.assert_isinstance(value, "target", six.string_types)
        self._property_target = value


class Mapping(NonStrictDataModel):
    """
    :param rules: Rules list
    :type rules: Sequence[MappingRule]
    """

    _schema = {
        "properties": {
            "rules": {
                "description": "Rules list",
                "items": {"$ref": "#/definitions/mapping_rule"},
                "type": ["array", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, rules=None, **kwargs):
        super(Mapping, self).__init__(**kwargs)
        self.rules = rules

    @schema_property("rules")
    def rules(self):
        return self._property_rules

    @rules.setter
    def rules(self, value):
        if value is None:
            self._property_rules = None
            return

        self.assert_isinstance(value, "rules", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [
                MappingRule.from_dict(v) if isinstance(v, dict) else v for v in value
            ]
        else:
            self.assert_isinstance(value, "rules", MappingRule, is_array=True)
        self._property_rules = value


class ArchiveManyRequest(Request):
    """
    Archive dataviews

    :param ids: IDs of the dataviews to archive
    :type ids: Sequence[str]
    """

    _service = "dataviews"
    _action = "archive_many"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {
            "ids": {
                "description": "IDs of the dataviews to archive",
                "items": {"type": "string"},
                "type": "array",
            }
        },
        "required": ["ids"],
        "type": "object",
    }

    def __init__(self, ids, **kwargs):
        super(ArchiveManyRequest, self).__init__(**kwargs)
        self.ids = ids

    @schema_property("ids")
    def ids(self):
        return self._property_ids

    @ids.setter
    def ids(self, value):
        if value is None:
            self._property_ids = None
            return

        self.assert_isinstance(value, "ids", (list, tuple))

        self.assert_isinstance(value, "ids", six.string_types, is_array=True)
        self._property_ids = value


class ArchiveManyResponse(Response):
    """
    Response of dataviews.archive_many endpoint.

    :param succeeded:
    :type succeeded: Sequence[dict]
    :param failed:
    :type failed: Sequence[dict]
    """

    _service = "dataviews"
    _action = "archive_many"
    _version = "2.23"

    _schema = {
        "definitions": {},
        "properties": {
            "failed": {
                "items": {
                    "properties": {
                        "error": {
                            "description": "Error info",
                            "properties": {
                                "codes": {
                                    "items": {"type": "integer"},
                                    "type": "array",
                                },
                                "data": {
                                    "additionalProperties": True,
                                    "type": "object",
                                },
                                "msg": {"type": "string"},
                            },
                            "type": "object",
                        },
                        "id": {
                            "description": "ID of the failed entity",
                            "type": "string",
                        },
                    },
                    "type": "object",
                },
                "type": ["array", "null"],
            },
            "succeeded": {
                "items": {
                    "properties": {
                        "archived": {
                            "description": "Indicates whether the dataview was archived",
                            "type": "boolean",
                        },
                        "id": {
                            "description": "ID of the succeeded entity",
                            "type": "string",
                        },
                    },
                    "type": "object",
                },
                "type": ["array", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, succeeded=None, failed=None, **kwargs):
        super(ArchiveManyResponse, self).__init__(**kwargs)
        self.succeeded = succeeded
        self.failed = failed

    @schema_property("succeeded")
    def succeeded(self):
        return self._property_succeeded

    @succeeded.setter
    def succeeded(self, value):
        if value is None:
            self._property_succeeded = None
            return

        self.assert_isinstance(value, "succeeded", (list, tuple))

        self.assert_isinstance(value, "succeeded", (dict,), is_array=True)
        self._property_succeeded = value

    @schema_property("failed")
    def failed(self):
        return self._property_failed

    @failed.setter
    def failed(self, value):
        if value is None:
            self._property_failed = None
            return

        self.assert_isinstance(value, "failed", (list, tuple))

        self.assert_isinstance(value, "failed", (dict,), is_array=True)
        self._property_failed = value


class CreateRequest(Request):
    """
    Create a new dataview

    :param name: Dataview name
    :type name: str
    :param description: Dataview description
    :type description: str
    :param project: Project ID of the project to which this task is assigned
    :type project: str
    :param filters: List of FilterRule ('OR' connection)
    :type filters: Sequence[FilterRule]
    :param output_rois: 'all_in_frame' - all rois for a frame are returned
        'only_filtered' - only rois which led this frame to be selected 'frame_per_roi'
        - single roi per frame. Frame can be returned multiple times with a different
        roi each time. Note: this should be used for Training tasks only Note:
        frame_per_roi implies that only filtered rois will be returned
    :type output_rois: OutputRoisEnum
    :param versions: List of dataview entries. All tasks must have at least one
        dataview.
    :type versions: Sequence[DataviewEntry]
    :param iteration: Iteration parameters. Not applicable for register (import)
        tasks.
    :type iteration: Iteration
    :param augmentation: Augmentation parameters. Only for training and testing
        tasks.
    :type augmentation: Augmentation
    :param tags: User-defined tags list
    :type tags: Sequence[str]
    :param system_tags: System tags list. This field is reserved for system use,
        please don't use it.
    :type system_tags: Sequence[str]
    :param mapping: Mapping parameters
    :type mapping: Mapping
    :param labels_enumeration: Labels enumerations, specifies numbers to be
        assigned to ROI labels when getting frames
    :type labels_enumeration: dict
    :param status: Dataview status
    :type status: str
    """

    _service = "dataviews"
    _action = "create"
    _version = "2.23"
    _schema = {
        "definitions": {
            "augmentation": {
                "properties": {
                    "crop_around_rois": {
                        "description": "Crop image data around all frame ROIs",
                        "type": ["boolean", "null"],
                    },
                    "sets": {
                        "description": "List of augmentation sets",
                        "items": {"$ref": "#/definitions/augmentation_set"},
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            },
            "augmentation_set": {
                "properties": {
                    "arguments": {
                        "additionalProperties": {
                            "additionalProperties": True,
                            "type": "object",
                        },
                        "description": "Arguments dictionary per custom augmentation type.",
                        "type": ["object", "null"],
                    },
                    "cls": {
                        "description": "Augmentation class",
                        "type": ["string", "null"],
                    },
                    "strength": {
                        "description": "Augmentation strength. Range [0,).",
                        "minimum": 0,
                        "type": ["number", "null"],
                    },
                    "types": {
                        "description": "Augmentation type",
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            },
            "dataview_entry": {
                "properties": {
                    "dataset": {
                        "description": "Existing Dataset id",
                        "type": "string",
                    },
                    "merge_with": {
                        "description": "Version ID to merge with",
                        "type": "string",
                    },
                    "version": {
                        "description": "Version id of a version belonging to the dataset",
                        "type": "string",
                    },
                },
                "required": ["dataset", "version"],
                "type": "object",
            },
            "filter_by_roi_enum": {
                "default": "label_rules",
                "enum": ["disabled", "no_rois", "label_rules"],
                "type": "string",
            },
            "filter_label_rule": {
                "properties": {
                    "conf_range": {
                        "description": (
                            "Range of ROI confidence level in the frame (min, max). -1 for not applicable\n           "
                            " Both min and max can be either -1 or positive.\n            2nd number (max) must be"
                            " either -1 or larger than or equal to the 1st number (min)"
                        ),
                        "items": {"type": "number"},
                        "maxItems": 2,
                        "minItems": 1,
                        "type": "array",
                    },
                    "count_range": {
                        "description": (
                            "Range of times ROI appears in the frame (min, max). -1 for not applicable.\n           "
                            " Both integers must be larger than or equal to -1.\n            2nd integer (max) must be"
                            " either -1 or larger than or equal to the 1st integer (min)"
                        ),
                        "items": {"type": "integer"},
                        "maxItems": 2,
                        "minItems": 1,
                        "type": "array",
                    },
                    "label": {
                        "description": (
                            "Lucene format query (see lucene query syntax).\nDefault search field is label.keyword and"
                            " default operator is AND, so searching for:\n\n'Bus Stop' Blue\n\nis equivalent"
                            " to:\n\nLabel.keyword:'Bus Stop' AND label.keyword:'Blue'"
                        ),
                        "type": "string",
                    },
                    "must_not": {
                        "default": False,
                        "description": (
                            "If set then the label must not exist or lucene query must not be true.\n            The"
                            " default value is false"
                        ),
                        "type": "boolean",
                    },
                },
                "required": ["label"],
                "type": "object",
            },
            "filter_rule": {
                "properties": {
                    "dataset": {
                        "description": (
                            "Dataset ID. Must be a dataset which is in the task's view. If set to '*' all datasets in"
                            " View are used."
                        ),
                        "type": "string",
                    },
                    "filter_by_roi": {
                        "description": "Type of filter. Optional, the default value is 'label_rules'",
                        "oneOf": [
                            {"$ref": "#/definitions/filter_by_roi_enum"},
                            {"type": "null"},
                        ],
                    },
                    "frame_query": {
                        "description": "Frame filter, in Lucene query syntax",
                        "type": ["string", "null"],
                    },
                    "label_rules": {
                        "description": (
                            "List of FilterLabelRule ('AND' connection)\n\ndisabled - No filtering by ROIs. Select all"
                            " frames, even if they don't have ROIs (all frames)\n\nno_rois - Select only frames without"
                            " ROIs (empty frames)\n\nlabel_rules - Select frames according to label rules"
                        ),
                        "items": {"$ref": "#/definitions/filter_label_rule"},
                        "type": ["array", "null"],
                    },
                    "sources_query": {
                        "description": "Sources filter, in Lucene query syntax. Filters sources in each frame.",
                        "type": ["string", "null"],
                    },
                    "version": {
                        "description": (
                            "Dataset version to apply rule to. Must belong to the dataset and be in the task's view. If"
                            " set to '*' all version of the datasets in View are used."
                        ),
                        "type": "string",
                    },
                    "weight": {
                        "description": "Rule weight. Default is 1",
                        "type": "number",
                    },
                },
                "required": ["dataset"],
                "type": "object",
            },
            "iteration": {
                "description": "Sequential Iteration API configuration",
                "properties": {
                    "infinite": {
                        "description": "Infinite iteration",
                        "type": ["boolean", "null"],
                    },
                    "jump": {
                        "description": "Jump entry",
                        "oneOf": [{"$ref": "#/definitions/jump"}, {"type": "null"}],
                    },
                    "limit": {
                        "description": (
                            "Maximum frames per task. If not passed, frames will end when no more matching frames are"
                            " found, unless infinite is True."
                        ),
                        "type": ["integer", "null"],
                    },
                    "min_sequence": {
                        "description": (
                            "Length (in ms) of video clips to return. This is used in random order, and in sequential"
                            " order only if jumping is provided and only for video frames"
                        ),
                        "type": ["integer", "null"],
                    },
                    "order": {
                        "description": (
                            "\n                Input frames order. Values: 'sequential', 'random'\n                In"
                            " Sequential mode frames will be returned according to the order in which the frames were"
                            " added to the dataset."
                        ),
                        "oneOf": [
                            {"$ref": "#/definitions/iteration_order_enum"},
                            {"type": "null"},
                        ],
                    },
                    "random_seed": {
                        "description": "Random seed used when iterating over the dataview",
                        "type": ["integer", "null"],
                    },
                },
                "type": "object",
            },
            "iteration_order_enum": {
                "enum": ["sequential", "random"],
                "type": "string",
            },
            "jump": {
                "properties": {
                    "time": {
                        "description": "Max time in milliseconds between frames",
                        "type": ["integer", "null"],
                    }
                },
                "type": "object",
            },
            "label_source": {
                "properties": {
                    "dataset": {
                        "description": "Source dataset id. '*' for all datasets in view",
                        "type": ["string", "null"],
                    },
                    "labels": {
                        "description": (
                            "List of source labels (AND connection). '*' indicates any label. Labels must exist in at"
                            " least one of the dataset versions in the task's view"
                        ),
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                    "version": {
                        "description": (
                            "Source dataset version id. Default is '*' (for all versions in dataset in the view)"
                            " Version must belong to the selected dataset, and must be in the task's view[i]"
                        ),
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "mapping": {
                "properties": {
                    "rules": {
                        "description": "Rules list",
                        "items": {"$ref": "#/definitions/mapping_rule"},
                        "type": ["array", "null"],
                    }
                },
                "type": "object",
            },
            "mapping_rule": {
                "properties": {
                    "source": {
                        "description": "Source label info",
                        "oneOf": [
                            {"$ref": "#/definitions/label_source"},
                            {"type": "null"},
                        ],
                    },
                    "target": {
                        "description": "Target label name",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "output_rois_enum": {
                "enum": ["all_in_frame", "only_filtered", "frame_per_roi"],
                "type": "string",
            },
        },
        "properties": {
            "augmentation": {
                "$ref": "#/definitions/augmentation",
                "description": "Augmentation parameters. Only for training and testing tasks.",
            },
            "description": {"description": "Dataview description", "type": "string"},
            "filters": {
                "description": "List of FilterRule ('OR' connection)",
                "items": {"$ref": "#/definitions/filter_rule"},
                "type": "array",
            },
            "iteration": {
                "$ref": "#/definitions/iteration",
                "description": "Iteration parameters. Not applicable for register (import) tasks.",
            },
            "labels_enumeration": {
                "additionalProperties": {"type": "integer"},
                "description": (
                    "Labels enumerations, specifies numbers to be assigned to ROI labels when getting frames"
                ),
                "type": "object",
            },
            "mapping": {
                "$ref": "#/definitions/mapping",
                "description": "Mapping parameters",
            },
            "name": {"description": "Dataview name", "type": "string"},
            "output_rois": {
                "$ref": "#/definitions/output_rois_enum",
                "default": "all_in_frame",
                "description": (
                    "'all_in_frame' - all rois for a frame are returned\n                    'only_filtered' - only"
                    " rois which led this frame to be selected\n                    'frame_per_roi' - single roi per"
                    " frame. Frame can be returned multiple times with a different roi each time.\n                   "
                    " Note: this should be used for Training tasks only\n                    Note: frame_per_roi"
                    " implies that only filtered rois will be returned\n                    "
                ),
            },
            "project": {
                "description": "Project ID of the project to which this task is assigned",
                "type": "string",
            },
            "status": {
                "description": "Dataview status",
                "enum": ["draft", "published"],
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
            "versions": {
                "description": "List of dataview entries. All tasks must have at least one dataview.",
                "items": {"$ref": "#/definitions/dataview_entry"},
                "type": "array",
            },
        },
        "required": ["name"],
        "type": "object",
    }

    def __init__(
        self,
        name,
        description=None,
        project=None,
        filters=None,
        output_rois="all_in_frame",
        versions=None,
        iteration=None,
        augmentation=None,
        tags=None,
        system_tags=None,
        mapping=None,
        labels_enumeration=None,
        status=None,
        **kwargs
    ):
        super(CreateRequest, self).__init__(**kwargs)
        self.name = name
        self.description = description
        self.project = project
        self.filters = filters
        self.output_rois = output_rois
        self.versions = versions
        self.iteration = iteration
        self.augmentation = augmentation
        self.tags = tags
        self.system_tags = system_tags
        self.mapping = mapping
        self.labels_enumeration = labels_enumeration
        self.status = status

    @schema_property("name")
    def name(self):
        return self._property_name

    @name.setter
    def name(self, value):
        if value is None:
            self._property_name = None
            return

        self.assert_isinstance(value, "name", six.string_types)
        self._property_name = value

    @schema_property("description")
    def description(self):
        return self._property_description

    @description.setter
    def description(self, value):
        if value is None:
            self._property_description = None
            return

        self.assert_isinstance(value, "description", six.string_types)
        self._property_description = value

    @schema_property("project")
    def project(self):
        return self._property_project

    @project.setter
    def project(self, value):
        if value is None:
            self._property_project = None
            return

        self.assert_isinstance(value, "project", six.string_types)
        self._property_project = value

    @schema_property("filters")
    def filters(self):
        return self._property_filters

    @filters.setter
    def filters(self, value):
        if value is None:
            self._property_filters = None
            return

        self.assert_isinstance(value, "filters", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [
                FilterRule.from_dict(v) if isinstance(v, dict) else v for v in value
            ]
        else:
            self.assert_isinstance(value, "filters", FilterRule, is_array=True)
        self._property_filters = value

    @schema_property("output_rois")
    def output_rois(self):
        return self._property_output_rois

    @output_rois.setter
    def output_rois(self, value):
        if value is None:
            self._property_output_rois = None
            return
        if isinstance(value, six.string_types):
            try:
                value = OutputRoisEnum(value)
            except ValueError:
                pass
        else:
            self.assert_isinstance(value, "output_rois", enum.Enum)
        self._property_output_rois = value

    @schema_property("versions")
    def versions(self):
        return self._property_versions

    @versions.setter
    def versions(self, value):
        if value is None:
            self._property_versions = None
            return

        self.assert_isinstance(value, "versions", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [
                DataviewEntry.from_dict(v) if isinstance(v, dict) else v for v in value
            ]
        else:
            self.assert_isinstance(value, "versions", DataviewEntry, is_array=True)
        self._property_versions = value

    @schema_property("iteration")
    def iteration(self):
        return self._property_iteration

    @iteration.setter
    def iteration(self, value):
        if value is None:
            self._property_iteration = None
            return
        if isinstance(value, dict):
            value = Iteration.from_dict(value)
        else:
            self.assert_isinstance(value, "iteration", Iteration)
        self._property_iteration = value

    @schema_property("augmentation")
    def augmentation(self):
        return self._property_augmentation

    @augmentation.setter
    def augmentation(self, value):
        if value is None:
            self._property_augmentation = None
            return
        if isinstance(value, dict):
            value = Augmentation.from_dict(value)
        else:
            self.assert_isinstance(value, "augmentation", Augmentation)
        self._property_augmentation = value

    @schema_property("tags")
    def tags(self):
        return self._property_tags

    @tags.setter
    def tags(self, value):
        if value is None:
            self._property_tags = None
            return

        self.assert_isinstance(value, "tags", (list, tuple))

        self.assert_isinstance(value, "tags", six.string_types, is_array=True)
        self._property_tags = value

    @schema_property("system_tags")
    def system_tags(self):
        return self._property_system_tags

    @system_tags.setter
    def system_tags(self, value):
        if value is None:
            self._property_system_tags = None
            return

        self.assert_isinstance(value, "system_tags", (list, tuple))

        self.assert_isinstance(value, "system_tags", six.string_types, is_array=True)
        self._property_system_tags = value

    @schema_property("mapping")
    def mapping(self):
        return self._property_mapping

    @mapping.setter
    def mapping(self, value):
        if value is None:
            self._property_mapping = None
            return
        if isinstance(value, dict):
            value = Mapping.from_dict(value)
        else:
            self.assert_isinstance(value, "mapping", Mapping)
        self._property_mapping = value

    @schema_property("labels_enumeration")
    def labels_enumeration(self):
        return self._property_labels_enumeration

    @labels_enumeration.setter
    def labels_enumeration(self, value):
        if value is None:
            self._property_labels_enumeration = None
            return

        self.assert_isinstance(value, "labels_enumeration", (dict,))
        self._property_labels_enumeration = value

    @schema_property("status")
    def status(self):
        return self._property_status

    @status.setter
    def status(self, value):
        if value is None:
            self._property_status = None
            return

        self.assert_isinstance(value, "status", six.string_types)
        self._property_status = value


class CreateResponse(Response):
    """
    Response of dataviews.create endpoint.

    :param id: New dataview's ID
    :type id: str
    """

    _service = "dataviews"
    _action = "create"
    _version = "2.23"

    _schema = {
        "definitions": {},
        "properties": {
            "id": {"description": "New dataview's ID", "type": ["string", "null"]}
        },
        "type": "object",
    }

    def __init__(self, id=None, **kwargs):
        super(CreateResponse, self).__init__(**kwargs)
        self.id = id

    @schema_property("id")
    def id(self):
        return self._property_id

    @id.setter
    def id(self, value):
        if value is None:
            self._property_id = None
            return

        self.assert_isinstance(value, "id", six.string_types)
        self._property_id = value


class DeleteRequest(Request):
    """
    Delete a dataview

    :param dataview: Datatview ID
    :type dataview: str
    :param force: Allow deletion of the published dataview
    :type force: bool
    """

    _service = "dataviews"
    _action = "delete"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {
            "dataview": {"description": "Datatview ID", "type": "string"},
            "force": {
                "default": False,
                "description": "Allow deletion of the published dataview",
                "type": "boolean",
            },
        },
        "required": ["dataview"],
        "type": "object",
    }

    def __init__(self, dataview, force=False, **kwargs):
        super(DeleteRequest, self).__init__(**kwargs)
        self.dataview = dataview
        self.force = force

    @schema_property("dataview")
    def dataview(self):
        return self._property_dataview

    @dataview.setter
    def dataview(self, value):
        if value is None:
            self._property_dataview = None
            return

        self.assert_isinstance(value, "dataview", six.string_types)
        self._property_dataview = value

    @schema_property("force")
    def force(self):
        return self._property_force

    @force.setter
    def force(self, value):
        if value is None:
            self._property_force = None
            return

        self.assert_isinstance(value, "force", (bool,))
        self._property_force = value


class DeleteResponse(Response):
    """
    Response of dataviews.delete endpoint.

    :param deleted: Number of dataviews deleted (0 or 1)
    :type deleted: float
    """

    _service = "dataviews"
    _action = "delete"
    _version = "2.23"

    _schema = {
        "definitions": {},
        "properties": {
            "deleted": {
                "description": "Number of dataviews deleted (0 or 1)",
                "enum": [0, 1],
                "type": ["number", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, deleted=None, **kwargs):
        super(DeleteResponse, self).__init__(**kwargs)
        self.deleted = deleted

    @schema_property("deleted")
    def deleted(self):
        return self._property_deleted

    @deleted.setter
    def deleted(self, value):
        if value is None:
            self._property_deleted = None
            return

        self.assert_isinstance(value, "deleted", six.integer_types + (float,))
        self._property_deleted = value


class DeleteManyRequest(Request):
    """
    Delete dataviews

    :param ids: IDs of the dataviews to delete
    :type ids: Sequence[str]
    :param force: Allow deletion of published dataviews
    :type force: bool
    """

    _service = "dataviews"
    _action = "delete_many"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {
            "force": {
                "default": False,
                "description": "Allow deletion of published dataviews",
                "type": "boolean",
            },
            "ids": {
                "description": "IDs of the dataviews to delete",
                "items": {"type": "string"},
                "type": "array",
            },
        },
        "required": ["ids"],
        "type": "object",
    }

    def __init__(self, ids, force=False, **kwargs):
        super(DeleteManyRequest, self).__init__(**kwargs)
        self.ids = ids
        self.force = force

    @schema_property("ids")
    def ids(self):
        return self._property_ids

    @ids.setter
    def ids(self, value):
        if value is None:
            self._property_ids = None
            return

        self.assert_isinstance(value, "ids", (list, tuple))

        self.assert_isinstance(value, "ids", six.string_types, is_array=True)
        self._property_ids = value

    @schema_property("force")
    def force(self):
        return self._property_force

    @force.setter
    def force(self, value):
        if value is None:
            self._property_force = None
            return

        self.assert_isinstance(value, "force", (bool,))
        self._property_force = value


class DeleteManyResponse(Response):
    """
    Response of dataviews.delete_many endpoint.

    :param succeeded:
    :type succeeded: Sequence[dict]
    :param failed:
    :type failed: Sequence[dict]
    """

    _service = "dataviews"
    _action = "delete_many"
    _version = "2.23"

    _schema = {
        "definitions": {},
        "properties": {
            "failed": {
                "items": {
                    "properties": {
                        "error": {
                            "description": "Error info",
                            "properties": {
                                "codes": {
                                    "items": {"type": "integer"},
                                    "type": "array",
                                },
                                "data": {
                                    "additionalProperties": True,
                                    "type": "object",
                                },
                                "msg": {"type": "string"},
                            },
                            "type": "object",
                        },
                        "id": {
                            "description": "ID of the failed entity",
                            "type": "string",
                        },
                    },
                    "type": "object",
                },
                "type": ["array", "null"],
            },
            "succeeded": {
                "items": {
                    "properties": {
                        "deleted": {
                            "description": "Indicates whether the dataview was deleted",
                            "type": "boolean",
                        },
                        "id": {
                            "description": "ID of the succeeded entity",
                            "type": "string",
                        },
                    },
                    "type": "object",
                },
                "type": ["array", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, succeeded=None, failed=None, **kwargs):
        super(DeleteManyResponse, self).__init__(**kwargs)
        self.succeeded = succeeded
        self.failed = failed

    @schema_property("succeeded")
    def succeeded(self):
        return self._property_succeeded

    @succeeded.setter
    def succeeded(self, value):
        if value is None:
            self._property_succeeded = None
            return

        self.assert_isinstance(value, "succeeded", (list, tuple))

        self.assert_isinstance(value, "succeeded", (dict,), is_array=True)
        self._property_succeeded = value

    @schema_property("failed")
    def failed(self):
        return self._property_failed

    @failed.setter
    def failed(self, value):
        if value is None:
            self._property_failed = None
            return

        self.assert_isinstance(value, "failed", (list, tuple))

        self.assert_isinstance(value, "failed", (dict,), is_array=True)
        self._property_failed = value


class GetAllRequest(Request):
    """
    Get all the company's dataviews and all public dataviews

    :param id: List of IDs to filter by
    :type id: Sequence[str]
    :param name: Get only dataviews whose name matches this pattern (python regular
        expression syntax)
    :type name: str
    :param user: List of user IDs used to filter results by the dataview's creating
        user
    :type user: Sequence[str]
    :param project: List of projects to filter by
    :type project: Sequence[str]
    :param output_rois: List of output ROIS types to filter by
    :type output_rois: Sequence[OutputRoisEnum]
    :param only_fields: List of dataview field names (nesting is supported using
        '.', e.g. execution.model_labels). If provided, this list defines the query's
        projection (only these fields will be returned for each result entry)
    :type only_fields: Sequence[str]
    :param tags: User-defined tags list used to filter results. Prepend '-' to tag
        name to indicate exclusion.
    :type tags: Sequence[str]
    :param system_tags: System tags list used to filter results. Prepend '-' to
        system tag name to indicate exclusion.
    :type system_tags: Sequence[str]
    :param status: Dataview status to filter by
    :type status: str
    :param page: Page number, returns a specific page out of the result list of
        datasets.
    :type page: int
    :param page_size: Page size, specifies the number of results returned in each
        page (last page may contain fewer results)
    :type page_size: int
    :param order_by: List of field names to order by. When search_text is used,
        '@text_score' can be used as a field representing the text score of returned
        documents. Use '-' prefix to specify descending order. Optional, recommended
        when using page
    :type order_by: Sequence[str]
    :param search_text: Free text search query
    :type search_text: str
    :param _all_: Multi-field pattern condition (all fields match pattern)
    :type _all_: MultiFieldPatternData
    :param _any_: Multi-field pattern condition (any field matches pattern)
    :type _any_: MultiFieldPatternData
    :param scroll_id: Scroll ID returned from the previos calls to get_all
    :type scroll_id: str
    :param refresh_scroll: If set then all the data received with this scroll will
        be requeried
    :type refresh_scroll: bool
    :param size: The number of datavievs to retrieve
    :type size: int
    """

    _service = "dataviews"
    _action = "get_all"
    _version = "2.23"
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
            "output_rois_enum": {
                "enum": ["all_in_frame", "only_filtered", "frame_per_roi"],
                "type": "string",
            },
        },
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
                "description": "Get only dataviews whose name matches this pattern (python regular expression syntax)",
                "type": ["string", "null"],
            },
            "only_fields": {
                "description": (
                    "List of dataview field names (nesting is supported using '.', e.g. execution.model_labels). If"
                    " provided, this list defines the query's projection (only these fields will be returned for each"
                    " result entry)"
                ),
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
            "order_by": {
                "description": (
                    "List of field names to order by. When search_text is used, '@text_score' can be used as a field"
                    " representing the text score of returned documents. Use '-' prefix to specify descending order."
                    " Optional, recommended when using page"
                ),
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
            "output_rois": {
                "description": "List of output ROIS types to filter by",
                "items": {"$ref": "#/definitions/output_rois_enum"},
                "type": ["array", "null"],
            },
            "page": {
                "description": "Page number, returns a specific page out of the result list of datasets.",
                "minimum": 0,
                "type": ["integer", "null"],
            },
            "page_size": {
                "description": (
                    "Page size, specifies the number of results returned in each page (last page may contain fewer "
                    "results)"
                ),
                "minimum": 1,
                "type": ["integer", "null"],
            },
            "project": {
                "description": "List of projects to filter by",
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
            "refresh_scroll": {
                "description": "If set then all the data received with this scroll will be requeried",
                "type": ["boolean", "null"],
            },
            "scroll_id": {
                "description": "Scroll ID returned from the previos calls to get_all",
                "type": ["string", "null"],
            },
            "search_text": {
                "description": "Free text search query",
                "type": ["string", "null"],
            },
            "size": {
                "description": "The number of datavievs to retrieve",
                "minimum": 1,
                "type": ["integer", "null"],
            },
            "status": {
                "description": "Dataview status to filter by",
                "enum": ["draft", "published"],
                "type": ["string", "null"],
            },
            "system_tags": {
                "description": (
                    "System tags list used to filter results. Prepend '-' to system tag name to indicate exclusion."
                ),
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
            "tags": {
                "description": (
                    "User-defined tags list used to filter results. Prepend '-' to tag name to indicate exclusion."
                ),
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
            "user": {
                "description": "List of user IDs used to filter results by the dataview's creating user",
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
        },
        "type": "object",
    }

    def __init__(
        self,
        id=None,
        name=None,
        user=None,
        project=None,
        output_rois=None,
        only_fields=None,
        tags=None,
        system_tags=None,
        status=None,
        page=None,
        page_size=None,
        order_by=None,
        search_text=None,
        _all_=None,
        _any_=None,
        scroll_id=None,
        refresh_scroll=None,
        size=None,
        **kwargs
    ):
        super(GetAllRequest, self).__init__(**kwargs)
        self.id = id
        self.name = name
        self.user = user
        self.project = project
        self.output_rois = output_rois
        self.only_fields = only_fields
        self.tags = tags
        self.system_tags = system_tags
        self.status = status
        self.page = page
        self.page_size = page_size
        self.order_by = order_by
        self.search_text = search_text
        self._all_ = _all_
        self._any_ = _any_
        self.scroll_id = scroll_id
        self.refresh_scroll = refresh_scroll
        self.size = size

    @schema_property("id")
    def id(self):
        return self._property_id

    @id.setter
    def id(self, value):
        if value is None:
            self._property_id = None
            return

        self.assert_isinstance(value, "id", (list, tuple))

        self.assert_isinstance(value, "id", six.string_types, is_array=True)
        self._property_id = value

    @schema_property("name")
    def name(self):
        return self._property_name

    @name.setter
    def name(self, value):
        if value is None:
            self._property_name = None
            return

        self.assert_isinstance(value, "name", six.string_types)
        self._property_name = value

    @schema_property("user")
    def user(self):
        return self._property_user

    @user.setter
    def user(self, value):
        if value is None:
            self._property_user = None
            return

        self.assert_isinstance(value, "user", (list, tuple))

        self.assert_isinstance(value, "user", six.string_types, is_array=True)
        self._property_user = value

    @schema_property("project")
    def project(self):
        return self._property_project

    @project.setter
    def project(self, value):
        if value is None:
            self._property_project = None
            return

        self.assert_isinstance(value, "project", (list, tuple))

        self.assert_isinstance(value, "project", six.string_types, is_array=True)
        self._property_project = value

    @schema_property("output_rois")
    def output_rois(self):
        return self._property_output_rois

    @output_rois.setter
    def output_rois(self, value):
        if value is None:
            self._property_output_rois = None
            return

        self.assert_isinstance(value, "output_rois", (list, tuple))
        if any(isinstance(v, six.string_types) for v in value):
            value = [
                OutputRoisEnum(v) if isinstance(v, six.string_types) else v
                for v in value
            ]
        else:
            self.assert_isinstance(value, "output_rois", OutputRoisEnum, is_array=True)
        self._property_output_rois = value

    @schema_property("only_fields")
    def only_fields(self):
        return self._property_only_fields

    @only_fields.setter
    def only_fields(self, value):
        if value is None:
            self._property_only_fields = None
            return

        self.assert_isinstance(value, "only_fields", (list, tuple))

        self.assert_isinstance(value, "only_fields", six.string_types, is_array=True)
        self._property_only_fields = value

    @schema_property("tags")
    def tags(self):
        return self._property_tags

    @tags.setter
    def tags(self, value):
        if value is None:
            self._property_tags = None
            return

        self.assert_isinstance(value, "tags", (list, tuple))

        self.assert_isinstance(value, "tags", six.string_types, is_array=True)
        self._property_tags = value

    @schema_property("system_tags")
    def system_tags(self):
        return self._property_system_tags

    @system_tags.setter
    def system_tags(self, value):
        if value is None:
            self._property_system_tags = None
            return

        self.assert_isinstance(value, "system_tags", (list, tuple))

        self.assert_isinstance(value, "system_tags", six.string_types, is_array=True)
        self._property_system_tags = value

    @schema_property("status")
    def status(self):
        return self._property_status

    @status.setter
    def status(self, value):
        if value is None:
            self._property_status = None
            return

        self.assert_isinstance(value, "status", six.string_types)
        self._property_status = value

    @schema_property("page")
    def page(self):
        return self._property_page

    @page.setter
    def page(self, value):
        if value is None:
            self._property_page = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "page", six.integer_types)
        self._property_page = value

    @schema_property("page_size")
    def page_size(self):
        return self._property_page_size

    @page_size.setter
    def page_size(self, value):
        if value is None:
            self._property_page_size = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "page_size", six.integer_types)
        self._property_page_size = value

    @schema_property("order_by")
    def order_by(self):
        return self._property_order_by

    @order_by.setter
    def order_by(self, value):
        if value is None:
            self._property_order_by = None
            return

        self.assert_isinstance(value, "order_by", (list, tuple))

        self.assert_isinstance(value, "order_by", six.string_types, is_array=True)
        self._property_order_by = value

    @schema_property("search_text")
    def search_text(self):
        return self._property_search_text

    @search_text.setter
    def search_text(self, value):
        if value is None:
            self._property_search_text = None
            return

        self.assert_isinstance(value, "search_text", six.string_types)
        self._property_search_text = value

    @schema_property("_all_")
    def _all_(self):
        return self._property__all_

    @_all_.setter
    def _all_(self, value):
        if value is None:
            self._property__all_ = None
            return
        if isinstance(value, dict):
            value = MultiFieldPatternData.from_dict(value)
        else:
            self.assert_isinstance(value, "_all_", MultiFieldPatternData)
        self._property__all_ = value

    @schema_property("_any_")
    def _any_(self):
        return self._property__any_

    @_any_.setter
    def _any_(self, value):
        if value is None:
            self._property__any_ = None
            return
        if isinstance(value, dict):
            value = MultiFieldPatternData.from_dict(value)
        else:
            self.assert_isinstance(value, "_any_", MultiFieldPatternData)
        self._property__any_ = value

    @schema_property("scroll_id")
    def scroll_id(self):
        return self._property_scroll_id

    @scroll_id.setter
    def scroll_id(self, value):
        if value is None:
            self._property_scroll_id = None
            return

        self.assert_isinstance(value, "scroll_id", six.string_types)
        self._property_scroll_id = value

    @schema_property("refresh_scroll")
    def refresh_scroll(self):
        return self._property_refresh_scroll

    @refresh_scroll.setter
    def refresh_scroll(self, value):
        if value is None:
            self._property_refresh_scroll = None
            return

        self.assert_isinstance(value, "refresh_scroll", (bool,))
        self._property_refresh_scroll = value

    @schema_property("size")
    def size(self):
        return self._property_size

    @size.setter
    def size(self, value):
        if value is None:
            self._property_size = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "size", six.integer_types)
        self._property_size = value


class GetAllResponse(Response):
    """
    Response of dataviews.get_all endpoint.

    :param dataviews: List of dataviews
    :type dataviews: Sequence[Dataview]
    :param scroll_id: Scroll ID that can be used with the next calls to get_all to
        retrieve more data
    :type scroll_id: str
    """

    _service = "dataviews"
    _action = "get_all"
    _version = "2.23"

    _schema = {
        "definitions": {
            "augmentation": {
                "properties": {
                    "crop_around_rois": {
                        "description": "Crop image data around all frame ROIs",
                        "type": ["boolean", "null"],
                    },
                    "sets": {
                        "description": "List of augmentation sets",
                        "items": {"$ref": "#/definitions/augmentation_set"},
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            },
            "augmentation_set": {
                "properties": {
                    "arguments": {
                        "additionalProperties": {
                            "additionalProperties": True,
                            "type": "object",
                        },
                        "description": "Arguments dictionary per custom augmentation type.",
                        "type": ["object", "null"],
                    },
                    "cls": {
                        "description": "Augmentation class",
                        "type": ["string", "null"],
                    },
                    "strength": {
                        "description": "Augmentation strength. Range [0,).",
                        "minimum": 0,
                        "type": ["number", "null"],
                    },
                    "types": {
                        "description": "Augmentation type",
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            },
            "dataview": {
                "properties": {
                    "augmentation": {
                        "$ref": "#/definitions/augmentation",
                        "description": "Augmentation parameters. Only for training and testing tasks.",
                    },
                    "company": {"description": "Company id", "type": "string"},
                    "created": {
                        "description": "Dataview creation time (UTC) ",
                        "format": "date-time",
                        "type": "string",
                    },
                    "description": {
                        "description": "Dataview description",
                        "type": "string",
                    },
                    "filters": {
                        "description": "List of FilterRule ('OR' connection)",
                        "items": {"$ref": "#/definitions/filter_rule"},
                        "type": "array",
                    },
                    "id": {"description": "Dataview ID", "type": "string"},
                    "iteration": {
                        "$ref": "#/definitions/iteration",
                        "description": "Iteration parameters. Not applicable for register (import) tasks.",
                    },
                    "labels_enumeration": {
                        "additionalProperties": {"type": "integer"},
                        "description": (
                            "Labels enumerations, specifies numbers to be assigned to ROI labels when getting frames"
                        ),
                        "type": "object",
                    },
                    "mapping": {
                        "$ref": "#/definitions/mapping",
                        "description": "Mapping parameters",
                    },
                    "name": {"description": "Dataview name", "type": "string"},
                    "output_rois": {
                        "$ref": "#/definitions/output_rois_enum",
                        "default": "all_in_frame",
                        "description": (
                            "'all_in_frame' - all rois for a frame are returned\n                'only_filtered' - only"
                            " rois which led this frame to be selected\n                'frame_per_roi' - single roi"
                            " per frame. Frame can be returned multiple times with a different roi each time.\n        "
                            "        Note: this should be used for Training tasks only\n                Note:"
                            " frame_per_roi implies that only filtered rois will be returned\n                "
                        ),
                    },
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
                    "user": {"description": "Associated user id", "type": "string"},
                    "versions": {
                        "description": "List of dataview entries. All tasks must have at least one dataview.",
                        "items": {"$ref": "#/definitions/dataview_entry"},
                        "type": "array",
                    },
                },
                "required": ["id", "name"],
                "type": "object",
            },
            "dataview_entry": {
                "properties": {
                    "dataset": {
                        "description": "Existing Dataset id",
                        "type": "string",
                    },
                    "merge_with": {
                        "description": "Version ID to merge with",
                        "type": "string",
                    },
                    "version": {
                        "description": "Version id of a version belonging to the dataset",
                        "type": "string",
                    },
                },
                "required": ["dataset", "version"],
                "type": "object",
            },
            "filter_by_roi_enum": {
                "default": "label_rules",
                "enum": ["disabled", "no_rois", "label_rules"],
                "type": "string",
            },
            "filter_label_rule": {
                "properties": {
                    "conf_range": {
                        "description": (
                            "Range of ROI confidence level in the frame (min, max). -1 for not applicable\n           "
                            " Both min and max can be either -1 or positive.\n            2nd number (max) must be"
                            " either -1 or larger than or equal to the 1st number (min)"
                        ),
                        "items": {"type": "number"},
                        "maxItems": 2,
                        "minItems": 1,
                        "type": "array",
                    },
                    "count_range": {
                        "description": (
                            "Range of times ROI appears in the frame (min, max). -1 for not applicable.\n           "
                            " Both integers must be larger than or equal to -1.\n            2nd integer (max) must be"
                            " either -1 or larger than or equal to the 1st integer (min)"
                        ),
                        "items": {"type": "integer"},
                        "maxItems": 2,
                        "minItems": 1,
                        "type": "array",
                    },
                    "label": {
                        "description": (
                            "Lucene format query (see lucene query syntax).\nDefault search field is label.keyword and"
                            " default operator is AND, so searching for:\n\n'Bus Stop' Blue\n\nis equivalent"
                            " to:\n\nLabel.keyword:'Bus Stop' AND label.keyword:'Blue'"
                        ),
                        "type": "string",
                    },
                    "must_not": {
                        "default": False,
                        "description": (
                            "If set then the label must not exist or lucene query must not be true.\n            The"
                            " default value is false"
                        ),
                        "type": "boolean",
                    },
                },
                "required": ["label"],
                "type": "object",
            },
            "filter_rule": {
                "properties": {
                    "dataset": {
                        "description": (
                            "Dataset ID. Must be a dataset which is in the task's view. If set to '*' all datasets in"
                            " View are used."
                        ),
                        "type": "string",
                    },
                    "filter_by_roi": {
                        "description": "Type of filter. Optional, the default value is 'label_rules'",
                        "oneOf": [
                            {"$ref": "#/definitions/filter_by_roi_enum"},
                            {"type": "null"},
                        ],
                    },
                    "frame_query": {
                        "description": "Frame filter, in Lucene query syntax",
                        "type": ["string", "null"],
                    },
                    "label_rules": {
                        "description": (
                            "List of FilterLabelRule ('AND' connection)\n\ndisabled - No filtering by ROIs. Select all"
                            " frames, even if they don't have ROIs (all frames)\n\nno_rois - Select only frames without"
                            " ROIs (empty frames)\n\nlabel_rules - Select frames according to label rules"
                        ),
                        "items": {"$ref": "#/definitions/filter_label_rule"},
                        "type": ["array", "null"],
                    },
                    "sources_query": {
                        "description": "Sources filter, in Lucene query syntax. Filters sources in each frame.",
                        "type": ["string", "null"],
                    },
                    "version": {
                        "description": (
                            "Dataset version to apply rule to. Must belong to the dataset and be in the task's view. If"
                            " set to '*' all version of the datasets in View are used."
                        ),
                        "type": "string",
                    },
                    "weight": {
                        "description": "Rule weight. Default is 1",
                        "type": "number",
                    },
                },
                "required": ["dataset"],
                "type": "object",
            },
            "iteration": {
                "description": "Sequential Iteration API configuration",
                "properties": {
                    "infinite": {
                        "description": "Infinite iteration",
                        "type": ["boolean", "null"],
                    },
                    "jump": {
                        "description": "Jump entry",
                        "oneOf": [{"$ref": "#/definitions/jump"}, {"type": "null"}],
                    },
                    "limit": {
                        "description": (
                            "Maximum frames per task. If not passed, frames will end when no more matching frames are"
                            " found, unless infinite is True."
                        ),
                        "type": ["integer", "null"],
                    },
                    "min_sequence": {
                        "description": (
                            "Length (in ms) of video clips to return. This is used in random order, and in sequential"
                            " order only if jumping is provided and only for video frames"
                        ),
                        "type": ["integer", "null"],
                    },
                    "order": {
                        "description": (
                            "\n                Input frames order. Values: 'sequential', 'random'\n                In"
                            " Sequential mode frames will be returned according to the order in which the frames were"
                            " added to the dataset."
                        ),
                        "oneOf": [
                            {"$ref": "#/definitions/iteration_order_enum"},
                            {"type": "null"},
                        ],
                    },
                    "random_seed": {
                        "description": "Random seed used when iterating over the dataview",
                        "type": ["integer", "null"],
                    },
                },
                "type": "object",
            },
            "iteration_order_enum": {
                "enum": ["sequential", "random"],
                "type": "string",
            },
            "jump": {
                "properties": {
                    "time": {
                        "description": "Max time in milliseconds between frames",
                        "type": ["integer", "null"],
                    }
                },
                "type": "object",
            },
            "label_source": {
                "properties": {
                    "dataset": {
                        "description": "Source dataset id. '*' for all datasets in view",
                        "type": ["string", "null"],
                    },
                    "labels": {
                        "description": (
                            "List of source labels (AND connection). '*' indicates any label. Labels must exist in at"
                            " least one of the dataset versions in the task's view"
                        ),
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                    "version": {
                        "description": (
                            "Source dataset version id. Default is '*' (for all versions in dataset in the view)"
                            " Version must belong to the selected dataset, and must be in the task's view[i]"
                        ),
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "mapping": {
                "properties": {
                    "rules": {
                        "description": "Rules list",
                        "items": {"$ref": "#/definitions/mapping_rule"},
                        "type": ["array", "null"],
                    }
                },
                "type": "object",
            },
            "mapping_rule": {
                "properties": {
                    "source": {
                        "description": "Source label info",
                        "oneOf": [
                            {"$ref": "#/definitions/label_source"},
                            {"type": "null"},
                        ],
                    },
                    "target": {
                        "description": "Target label name",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "output_rois_enum": {
                "enum": ["all_in_frame", "only_filtered", "frame_per_roi"],
                "type": "string",
            },
        },
        "properties": {
            "dataviews": {
                "description": "List of dataviews",
                "items": {"$ref": "#/definitions/dataview"},
                "type": ["array", "null"],
            },
            "scroll_id": {
                "description": "Scroll ID that can be used with the next calls to get_all to retrieve more data",
                "type": ["string", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, dataviews=None, scroll_id=None, **kwargs):
        super(GetAllResponse, self).__init__(**kwargs)
        self.dataviews = dataviews
        self.scroll_id = scroll_id

    @schema_property("dataviews")
    def dataviews(self):
        return self._property_dataviews

    @dataviews.setter
    def dataviews(self, value):
        if value is None:
            self._property_dataviews = None
            return

        self.assert_isinstance(value, "dataviews", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [Dataview.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "dataviews", Dataview, is_array=True)
        self._property_dataviews = value

    @schema_property("scroll_id")
    def scroll_id(self):
        return self._property_scroll_id

    @scroll_id.setter
    def scroll_id(self, value):
        if value is None:
            self._property_scroll_id = None
            return

        self.assert_isinstance(value, "scroll_id", six.string_types)
        self._property_scroll_id = value


class GetByIdRequest(Request):
    """
    Get dataview information

    :param dataview: Datatview ID
    :type dataview: str
    """

    _service = "dataviews"
    _action = "get_by_id"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {"dataview": {"description": "Datatview ID", "type": "string"}},
        "required": ["dataview"],
        "type": "object",
    }

    def __init__(self, dataview, **kwargs):
        super(GetByIdRequest, self).__init__(**kwargs)
        self.dataview = dataview

    @schema_property("dataview")
    def dataview(self):
        return self._property_dataview

    @dataview.setter
    def dataview(self, value):
        if value is None:
            self._property_dataview = None
            return

        self.assert_isinstance(value, "dataview", six.string_types)
        self._property_dataview = value


class GetByIdResponse(Response):
    """
    Response of dataviews.get_by_id endpoint.

    :param dataview: Dataview information
    :type dataview: Dataview
    """

    _service = "dataviews"
    _action = "get_by_id"
    _version = "2.23"

    _schema = {
        "definitions": {
            "augmentation": {
                "properties": {
                    "crop_around_rois": {
                        "description": "Crop image data around all frame ROIs",
                        "type": ["boolean", "null"],
                    },
                    "sets": {
                        "description": "List of augmentation sets",
                        "items": {"$ref": "#/definitions/augmentation_set"},
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            },
            "augmentation_set": {
                "properties": {
                    "arguments": {
                        "additionalProperties": {
                            "additionalProperties": True,
                            "type": "object",
                        },
                        "description": "Arguments dictionary per custom augmentation type.",
                        "type": ["object", "null"],
                    },
                    "cls": {
                        "description": "Augmentation class",
                        "type": ["string", "null"],
                    },
                    "strength": {
                        "description": "Augmentation strength. Range [0,).",
                        "minimum": 0,
                        "type": ["number", "null"],
                    },
                    "types": {
                        "description": "Augmentation type",
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            },
            "dataview": {
                "properties": {
                    "augmentation": {
                        "$ref": "#/definitions/augmentation",
                        "description": "Augmentation parameters. Only for training and testing tasks.",
                    },
                    "company": {"description": "Company id", "type": "string"},
                    "created": {
                        "description": "Dataview creation time (UTC) ",
                        "format": "date-time",
                        "type": "string",
                    },
                    "description": {
                        "description": "Dataview description",
                        "type": "string",
                    },
                    "filters": {
                        "description": "List of FilterRule ('OR' connection)",
                        "items": {"$ref": "#/definitions/filter_rule"},
                        "type": "array",
                    },
                    "id": {"description": "Dataview ID", "type": "string"},
                    "iteration": {
                        "$ref": "#/definitions/iteration",
                        "description": "Iteration parameters. Not applicable for register (import) tasks.",
                    },
                    "labels_enumeration": {
                        "additionalProperties": {"type": "integer"},
                        "description": (
                            "Labels enumerations, specifies numbers to be assigned to ROI labels when getting frames"
                        ),
                        "type": "object",
                    },
                    "mapping": {
                        "$ref": "#/definitions/mapping",
                        "description": "Mapping parameters",
                    },
                    "name": {"description": "Dataview name", "type": "string"},
                    "output_rois": {
                        "$ref": "#/definitions/output_rois_enum",
                        "default": "all_in_frame",
                        "description": (
                            "'all_in_frame' - all rois for a frame are returned\n                'only_filtered' - only"
                            " rois which led this frame to be selected\n                'frame_per_roi' - single roi"
                            " per frame. Frame can be returned multiple times with a different roi each time.\n        "
                            "        Note: this should be used for Training tasks only\n                Note:"
                            " frame_per_roi implies that only filtered rois will be returned\n                "
                        ),
                    },
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
                    "user": {"description": "Associated user id", "type": "string"},
                    "versions": {
                        "description": "List of dataview entries. All tasks must have at least one dataview.",
                        "items": {"$ref": "#/definitions/dataview_entry"},
                        "type": "array",
                    },
                    "status": {
                        "description": "dataview status",
                        "enum": ["draft", "published"],
                        "type": "string",
                    },
                },
                "required": ["id", "name"],
                "type": "object",
            },
            "dataview_entry": {
                "properties": {
                    "dataset": {
                        "description": "Existing Dataset id",
                        "type": "string",
                    },
                    "merge_with": {
                        "description": "Version ID to merge with",
                        "type": "string",
                    },
                    "version": {
                        "description": "Version id of a version belonging to the dataset",
                        "type": "string",
                    },
                },
                "required": ["dataset", "version"],
                "type": "object",
            },
            "filter_by_roi_enum": {
                "default": "label_rules",
                "enum": ["disabled", "no_rois", "label_rules"],
                "type": "string",
            },
            "filter_label_rule": {
                "properties": {
                    "conf_range": {
                        "description": (
                            "Range of ROI confidence level in the frame (min, max). -1 for not applicable\n           "
                            " Both min and max can be either -1 or positive.\n            2nd number (max) must be"
                            " either -1 or larger than or equal to the 1st number (min)"
                        ),
                        "items": {"type": "number"},
                        "maxItems": 2,
                        "minItems": 1,
                        "type": "array",
                    },
                    "count_range": {
                        "description": (
                            "Range of times ROI appears in the frame (min, max). -1 for not applicable.\n           "
                            " Both integers must be larger than or equal to -1.\n            2nd integer (max) must be"
                            " either -1 or larger than or equal to the 1st integer (min)"
                        ),
                        "items": {"type": "integer"},
                        "maxItems": 2,
                        "minItems": 1,
                        "type": "array",
                    },
                    "label": {
                        "description": (
                            "Lucene format query (see lucene query syntax).\nDefault search field is label.keyword and"
                            " default operator is AND, so searching for:\n\n'Bus Stop' Blue\n\nis equivalent"
                            " to:\n\nLabel.keyword:'Bus Stop' AND label.keyword:'Blue'"
                        ),
                        "type": "string",
                    },
                    "must_not": {
                        "default": False,
                        "description": (
                            "If set then the label must not exist or lucene query must not be true.\n            The"
                            " default value is false"
                        ),
                        "type": "boolean",
                    },
                },
                "required": ["label"],
                "type": "object",
            },
            "filter_rule": {
                "properties": {
                    "dataset": {
                        "description": (
                            "Dataset ID. Must be a dataset which is in the task's view. If set to '*' all datasets in"
                            " View are used."
                        ),
                        "type": "string",
                    },
                    "filter_by_roi": {
                        "description": "Type of filter. Optional, the default value is 'label_rules'",
                        "oneOf": [
                            {"$ref": "#/definitions/filter_by_roi_enum"},
                            {"type": "null"},
                        ],
                    },
                    "frame_query": {
                        "description": "Frame filter, in Lucene query syntax",
                        "type": ["string", "null"],
                    },
                    "label_rules": {
                        "description": (
                            "List of FilterLabelRule ('AND' connection)\n\ndisabled - No filtering by ROIs. Select all"
                            " frames, even if they don't have ROIs (all frames)\n\nno_rois - Select only frames without"
                            " ROIs (empty frames)\n\nlabel_rules - Select frames according to label rules"
                        ),
                        "items": {"$ref": "#/definitions/filter_label_rule"},
                        "type": ["array", "null"],
                    },
                    "sources_query": {
                        "description": "Sources filter, in Lucene query syntax. Filters sources in each frame.",
                        "type": ["string", "null"],
                    },
                    "version": {
                        "description": (
                            "Dataset version to apply rule to. Must belong to the dataset and be in the task's view. If"
                            " set to '*' all version of the datasets in View are used."
                        ),
                        "type": "string",
                    },
                    "weight": {
                        "description": "Rule weight. Default is 1",
                        "type": "number",
                    },
                },
                "required": ["dataset"],
                "type": "object",
            },
            "iteration": {
                "description": "Sequential Iteration API configuration",
                "properties": {
                    "infinite": {
                        "description": "Infinite iteration",
                        "type": ["boolean", "null"],
                    },
                    "jump": {
                        "description": "Jump entry",
                        "oneOf": [{"$ref": "#/definitions/jump"}, {"type": "null"}],
                    },
                    "limit": {
                        "description": (
                            "Maximum frames per task. If not passed, frames will end when no more matching frames are"
                            " found, unless infinite is True."
                        ),
                        "type": ["integer", "null"],
                    },
                    "min_sequence": {
                        "description": (
                            "Length (in ms) of video clips to return. This is used in random order, and in sequential"
                            " order only if jumping is provided and only for video frames"
                        ),
                        "type": ["integer", "null"],
                    },
                    "order": {
                        "description": (
                            "\n                Input frames order. Values: 'sequential', 'random'\n                In"
                            " Sequential mode frames will be returned according to the order in which the frames were"
                            " added to the dataset."
                        ),
                        "oneOf": [
                            {"$ref": "#/definitions/iteration_order_enum"},
                            {"type": "null"},
                        ],
                    },
                    "random_seed": {
                        "description": "Random seed used when iterating over the dataview",
                        "type": ["integer", "null"],
                    },
                },
                "type": "object",
            },
            "iteration_order_enum": {
                "enum": ["sequential", "random"],
                "type": "string",
            },
            "jump": {
                "properties": {
                    "time": {
                        "description": "Max time in milliseconds between frames",
                        "type": ["integer", "null"],
                    }
                },
                "type": "object",
            },
            "label_source": {
                "properties": {
                    "dataset": {
                        "description": "Source dataset id. '*' for all datasets in view",
                        "type": ["string", "null"],
                    },
                    "labels": {
                        "description": (
                            "List of source labels (AND connection). '*' indicates any label. Labels must exist in at"
                            " least one of the dataset versions in the task's view"
                        ),
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                    "version": {
                        "description": (
                            "Source dataset version id. Default is '*' (for all versions in dataset in the view)"
                            " Version must belong to the selected dataset, and must be in the task's view[i]"
                        ),
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "mapping": {
                "properties": {
                    "rules": {
                        "description": "Rules list",
                        "items": {"$ref": "#/definitions/mapping_rule"},
                        "type": ["array", "null"],
                    }
                },
                "type": "object",
            },
            "mapping_rule": {
                "properties": {
                    "source": {
                        "description": "Source label info",
                        "oneOf": [
                            {"$ref": "#/definitions/label_source"},
                            {"type": "null"},
                        ],
                    },
                    "target": {
                        "description": "Target label name",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "output_rois_enum": {
                "enum": ["all_in_frame", "only_filtered", "frame_per_roi"],
                "type": "string",
            },
        },
        "properties": {
            "dataview": {
                "description": "Dataview information",
                "oneOf": [{"$ref": "#/definitions/dataview"}, {"type": "null"}],
            }
        },
        "type": "object",
    }

    def __init__(self, dataview=None, **kwargs):
        super(GetByIdResponse, self).__init__(**kwargs)
        self.dataview = dataview

    @schema_property("dataview")
    def dataview(self):
        return self._property_dataview

    @dataview.setter
    def dataview(self, value):
        if value is None:
            self._property_dataview = None
            return
        if isinstance(value, dict):
            value = Dataview.from_dict(value)
        else:
            self.assert_isinstance(value, "dataview", Dataview)
        self._property_dataview = value


class MoveRequest(Request):
    """
    Move dataviews to a project

    :param ids: Dataviews to move
    :type ids: Sequence[str]
    :param project: Target project ID. If not provided, `project_name` must be
        provided. Use null for the root project
    :type project: str
    :param project_name: Target project name. If provided and a project with this
        name does not exist, a new project will be created. If not provided, `project`
        must be provided.
    :type project_name: str
    """

    _service = "dataviews"
    _action = "move"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {
            "ids": {
                "description": "Dataviews to move",
                "items": {"type": "string"},
                "type": "array",
            },
            "project": {
                "description": (
                    "Target project ID. If not provided, `project_name` must be provided. Use null for the root project"
                ),
                "type": "string",
            },
            "project_name": {
                "description": (
                    "Target project name. If provided and a project with this name does not exist, a new project will"
                    " be created. If not provided, `project` must be provided."
                ),
                "type": "string",
            },
        },
        "required": ["ids"],
        "type": "object",
    }

    def __init__(self, ids, project=None, project_name=None, **kwargs):
        super(MoveRequest, self).__init__(**kwargs)
        self.ids = ids
        self.project = project
        self.project_name = project_name

    @schema_property("ids")
    def ids(self):
        return self._property_ids

    @ids.setter
    def ids(self, value):
        if value is None:
            self._property_ids = None
            return

        self.assert_isinstance(value, "ids", (list, tuple))

        self.assert_isinstance(value, "ids", six.string_types, is_array=True)
        self._property_ids = value

    @schema_property("project")
    def project(self):
        return self._property_project

    @project.setter
    def project(self, value):
        if value is None:
            self._property_project = None
            return

        self.assert_isinstance(value, "project", six.string_types)
        self._property_project = value

    @schema_property("project_name")
    def project_name(self):
        return self._property_project_name

    @project_name.setter
    def project_name(self, value):
        if value is None:
            self._property_project_name = None
            return

        self.assert_isinstance(value, "project_name", six.string_types)
        self._property_project_name = value


class MoveResponse(Response):
    """
    Response of dataviews.move endpoint.

    """

    _service = "dataviews"
    _action = "move"
    _version = "2.23"

    _schema = {"additionalProperties": True, "definitions": {}, "type": "object"}


class PublishRequest(Request):
    """
    Publish a dataview

    :param dataview: Datatview ID
    :type dataview: str
    """

    _service = "dataviews"
    _action = "publish"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {"dataview": {"description": "Datatview ID", "type": "string"}},
        "required": ["dataview"],
        "type": "object",
    }

    def __init__(self, dataview, **kwargs):
        super(PublishRequest, self).__init__(**kwargs)
        self.dataview = dataview

    @schema_property("dataview")
    def dataview(self):
        return self._property_dataview

    @dataview.setter
    def dataview(self, value):
        if value is None:
            self._property_dataview = None
            return

        self.assert_isinstance(value, "dataview", six.string_types)
        self._property_dataview = value


class PublishResponse(Response):
    """
    Response of dataviews.publish endpoint.

    :param published: Number of dataviews published (0 or 1)
    :type published: float
    """

    _service = "dataviews"
    _action = "publish"
    _version = "2.23"

    _schema = {
        "definitions": {},
        "properties": {
            "published": {
                "description": "Number of dataviews published (0 or 1)",
                "enum": [0, 1],
                "type": ["number", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, published=None, **kwargs):
        super(PublishResponse, self).__init__(**kwargs)
        self.published = published

    @schema_property("published")
    def published(self):
        return self._property_published

    @published.setter
    def published(self, value):
        if value is None:
            self._property_published = None
            return

        self.assert_isinstance(value, "published", six.integer_types + (float,))
        self._property_published = value


class PublishManyRequest(Request):
    """
    Publish dataviews

    :param ids: IDs of the dataviews to publish
    :type ids: Sequence[str]
    """

    _service = "dataviews"
    _action = "publish_many"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {
            "ids": {
                "description": "IDs of the dataviews to publish",
                "items": {"type": "string"},
                "type": "array",
            }
        },
        "required": ["ids"],
        "type": "object",
    }

    def __init__(self, ids, **kwargs):
        super(PublishManyRequest, self).__init__(**kwargs)
        self.ids = ids

    @schema_property("ids")
    def ids(self):
        return self._property_ids

    @ids.setter
    def ids(self, value):
        if value is None:
            self._property_ids = None
            return

        self.assert_isinstance(value, "ids", (list, tuple))

        self.assert_isinstance(value, "ids", six.string_types, is_array=True)
        self._property_ids = value


class PublishManyResponse(Response):
    """
    Response of dataviews.publish_many endpoint.

    :param succeeded:
    :type succeeded: Sequence[dict]
    :param failed:
    :type failed: Sequence[dict]
    """

    _service = "dataviews"
    _action = "publish_many"
    _version = "2.23"

    _schema = {
        "definitions": {},
        "properties": {
            "failed": {
                "items": {
                    "properties": {
                        "error": {
                            "description": "Error info",
                            "properties": {
                                "codes": {
                                    "items": {"type": "integer"},
                                    "type": "array",
                                },
                                "data": {
                                    "additionalProperties": True,
                                    "type": "object",
                                },
                                "msg": {"type": "string"},
                            },
                            "type": "object",
                        },
                        "id": {
                            "description": "ID of the failed entity",
                            "type": "string",
                        },
                    },
                    "type": "object",
                },
                "type": ["array", "null"],
            },
            "succeeded": {
                "items": {
                    "properties": {
                        "id": {
                            "description": "ID of the succeeded entity",
                            "type": "string",
                        },
                        "published": {
                            "description": "Indicates whether the dataview was published",
                            "type": "boolean",
                        },
                    },
                    "type": "object",
                },
                "type": ["array", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, succeeded=None, failed=None, **kwargs):
        super(PublishManyResponse, self).__init__(**kwargs)
        self.succeeded = succeeded
        self.failed = failed

    @schema_property("succeeded")
    def succeeded(self):
        return self._property_succeeded

    @succeeded.setter
    def succeeded(self, value):
        if value is None:
            self._property_succeeded = None
            return

        self.assert_isinstance(value, "succeeded", (list, tuple))

        self.assert_isinstance(value, "succeeded", (dict,), is_array=True)
        self._property_succeeded = value

    @schema_property("failed")
    def failed(self):
        return self._property_failed

    @failed.setter
    def failed(self, value):
        if value is None:
            self._property_failed = None
            return

        self.assert_isinstance(value, "failed", (list, tuple))

        self.assert_isinstance(value, "failed", (dict,), is_array=True)
        self._property_failed = value


class UnarchiveManyRequest(Request):
    """
    Unarchive dataviews

    :param ids: IDs of the dataviews to unarchive
    :type ids: Sequence[str]
    """

    _service = "dataviews"
    _action = "unarchive_many"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {
            "ids": {
                "description": "IDs of the dataviews to unarchive",
                "items": {"type": "string"},
                "type": "array",
            }
        },
        "required": ["ids"],
        "type": "object",
    }

    def __init__(self, ids, **kwargs):
        super(UnarchiveManyRequest, self).__init__(**kwargs)
        self.ids = ids

    @schema_property("ids")
    def ids(self):
        return self._property_ids

    @ids.setter
    def ids(self, value):
        if value is None:
            self._property_ids = None
            return

        self.assert_isinstance(value, "ids", (list, tuple))

        self.assert_isinstance(value, "ids", six.string_types, is_array=True)
        self._property_ids = value


class UnarchiveManyResponse(Response):
    """
    Response of dataviews.unarchive_many endpoint.

    :param succeeded:
    :type succeeded: Sequence[dict]
    :param failed:
    :type failed: Sequence[dict]
    """

    _service = "dataviews"
    _action = "unarchive_many"
    _version = "2.23"

    _schema = {
        "definitions": {},
        "properties": {
            "failed": {
                "items": {
                    "properties": {
                        "error": {
                            "description": "Error info",
                            "properties": {
                                "codes": {
                                    "items": {"type": "integer"},
                                    "type": "array",
                                },
                                "data": {
                                    "additionalProperties": True,
                                    "type": "object",
                                },
                                "msg": {"type": "string"},
                            },
                            "type": "object",
                        },
                        "id": {
                            "description": "ID of the failed entity",
                            "type": "string",
                        },
                    },
                    "type": "object",
                },
                "type": ["array", "null"],
            },
            "succeeded": {
                "items": {
                    "properties": {
                        "id": {
                            "description": "ID of the succeeded entity",
                            "type": "string",
                        },
                        "unarchived": {
                            "description": "Indicates whether the dataview was unarchived",
                            "type": "boolean",
                        },
                    },
                    "type": "object",
                },
                "type": ["array", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, succeeded=None, failed=None, **kwargs):
        super(UnarchiveManyResponse, self).__init__(**kwargs)
        self.succeeded = succeeded
        self.failed = failed

    @schema_property("succeeded")
    def succeeded(self):
        return self._property_succeeded

    @succeeded.setter
    def succeeded(self, value):
        if value is None:
            self._property_succeeded = None
            return

        self.assert_isinstance(value, "succeeded", (list, tuple))

        self.assert_isinstance(value, "succeeded", (dict,), is_array=True)
        self._property_succeeded = value

    @schema_property("failed")
    def failed(self):
        return self._property_failed

    @failed.setter
    def failed(self, value):
        if value is None:
            self._property_failed = None
            return

        self.assert_isinstance(value, "failed", (list, tuple))

        self.assert_isinstance(value, "failed", (dict,), is_array=True)
        self._property_failed = value


class UpdateRequest(Request):
    """
    Get dataview information

    :param dataview: Datatview ID
    :type dataview: str
    :param name: Dataview name
    :type name: str
    :param description: Dataview description
    :type description: str
    :param project: Project ID of the project to which this task is assigned
    :type project: str
    :param filters: List of FilterRule ('OR' connection)
    :type filters: Sequence[FilterRule]
    :param output_rois: 'all_in_frame' - all rois for a frame are returned
        'only_filtered' - only rois which led this frame to be selected 'frame_per_roi'
        - single roi per frame. Frame can be returned multiple times with a different
        roi each time. Note: this should be used for Training tasks only Note:
        frame_per_roi implies that only filtered rois will be returned
    :type output_rois: OutputRoisEnum
    :param versions: List of dataview entries. All tasks must have at least one
        dataview.
    :type versions: Sequence[DataviewEntry]
    :param iteration: Iteration parameters. Not applicable for register (import)
        tasks.
    :type iteration: Iteration
    :param augmentation: Augmentation parameters. Only for training and testing
        tasks.
    :type augmentation: Augmentation
    :param tags: User-defined tags list
    :type tags: Sequence[str]
    :param system_tags: System tags list. This field is reserved for system use,
        please don't use it.
    :type system_tags: Sequence[str]
    :param mapping: Mapping parameters
    :type mapping: Mapping
    :param labels_enumeration: Labels enumerations, specifies numbers to be
        assigned to ROI labels when getting frames
    :type labels_enumeration: dict
    :param status: Dataview status
    :type status: str
    :param force: Allow update of the published dataview
    :type force: bool
    """

    _service = "dataviews"
    _action = "update"
    _version = "2.23"
    _schema = {
        "definitions": {
            "augmentation": {
                "properties": {
                    "crop_around_rois": {
                        "description": "Crop image data around all frame ROIs",
                        "type": ["boolean", "null"],
                    },
                    "sets": {
                        "description": "List of augmentation sets",
                        "items": {"$ref": "#/definitions/augmentation_set"},
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            },
            "augmentation_set": {
                "properties": {
                    "arguments": {
                        "additionalProperties": {
                            "additionalProperties": True,
                            "type": "object",
                        },
                        "description": "Arguments dictionary per custom augmentation type.",
                        "type": ["object", "null"],
                    },
                    "cls": {
                        "description": "Augmentation class",
                        "type": ["string", "null"],
                    },
                    "strength": {
                        "description": "Augmentation strength. Range [0,).",
                        "minimum": 0,
                        "type": ["number", "null"],
                    },
                    "types": {
                        "description": "Augmentation type",
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            },
            "dataview_entry": {
                "properties": {
                    "dataset": {
                        "description": "Existing Dataset id",
                        "type": "string",
                    },
                    "merge_with": {
                        "description": "Version ID to merge with",
                        "type": "string",
                    },
                    "version": {
                        "description": "Version id of a version belonging to the dataset",
                        "type": "string",
                    },
                },
                "required": ["dataset", "version"],
                "type": "object",
            },
            "filter_by_roi_enum": {
                "default": "label_rules",
                "enum": ["disabled", "no_rois", "label_rules"],
                "type": "string",
            },
            "filter_label_rule": {
                "properties": {
                    "conf_range": {
                        "description": (
                            "Range of ROI confidence level in the frame (min, max). -1 for not applicable\n           "
                            " Both min and max can be either -1 or positive.\n            2nd number (max) must be"
                            " either -1 or larger than or equal to the 1st number (min)"
                        ),
                        "items": {"type": "number"},
                        "maxItems": 2,
                        "minItems": 1,
                        "type": "array",
                    },
                    "count_range": {
                        "description": (
                            "Range of times ROI appears in the frame (min, max). -1 for not applicable.\n           "
                            " Both integers must be larger than or equal to -1.\n            2nd integer (max) must be"
                            " either -1 or larger than or equal to the 1st integer (min)"
                        ),
                        "items": {"type": "integer"},
                        "maxItems": 2,
                        "minItems": 1,
                        "type": "array",
                    },
                    "label": {
                        "description": (
                            "Lucene format query (see lucene query syntax).\nDefault search field is label.keyword and"
                            " default operator is AND, so searching for:\n\n'Bus Stop' Blue\n\nis equivalent"
                            " to:\n\nLabel.keyword:'Bus Stop' AND label.keyword:'Blue'"
                        ),
                        "type": "string",
                    },
                    "must_not": {
                        "default": False,
                        "description": (
                            "If set then the label must not exist or lucene query must not be true.\n            The"
                            " default value is false"
                        ),
                        "type": "boolean",
                    },
                },
                "required": ["label"],
                "type": "object",
            },
            "filter_rule": {
                "properties": {
                    "dataset": {
                        "description": (
                            "Dataset ID. Must be a dataset which is in the task's view. If set to '*' all datasets in"
                            " View are used."
                        ),
                        "type": "string",
                    },
                    "filter_by_roi": {
                        "description": "Type of filter. Optional, the default value is 'label_rules'",
                        "oneOf": [
                            {"$ref": "#/definitions/filter_by_roi_enum"},
                            {"type": "null"},
                        ],
                    },
                    "frame_query": {
                        "description": "Frame filter, in Lucene query syntax",
                        "type": ["string", "null"],
                    },
                    "label_rules": {
                        "description": (
                            "List of FilterLabelRule ('AND' connection)\n\ndisabled - No filtering by ROIs. Select all"
                            " frames, even if they don't have ROIs (all frames)\n\nno_rois - Select only frames without"
                            " ROIs (empty frames)\n\nlabel_rules - Select frames according to label rules"
                        ),
                        "items": {"$ref": "#/definitions/filter_label_rule"},
                        "type": ["array", "null"],
                    },
                    "sources_query": {
                        "description": "Sources filter, in Lucene query syntax. Filters sources in each frame.",
                        "type": ["string", "null"],
                    },
                    "version": {
                        "description": (
                            "Dataset version to apply rule to. Must belong to the dataset and be in the task's view. If"
                            " set to '*' all version of the datasets in View are used."
                        ),
                        "type": "string",
                    },
                    "weight": {
                        "description": "Rule weight. Default is 1",
                        "type": "number",
                    },
                },
                "required": ["dataset"],
                "type": "object",
            },
            "iteration": {
                "description": "Sequential Iteration API configuration",
                "properties": {
                    "infinite": {
                        "description": "Infinite iteration",
                        "type": ["boolean", "null"],
                    },
                    "jump": {
                        "description": "Jump entry",
                        "oneOf": [{"$ref": "#/definitions/jump"}, {"type": "null"}],
                    },
                    "limit": {
                        "description": (
                            "Maximum frames per task. If not passed, frames will end when no more matching frames are"
                            " found, unless infinite is True."
                        ),
                        "type": ["integer", "null"],
                    },
                    "min_sequence": {
                        "description": (
                            "Length (in ms) of video clips to return. This is used in random order, and in sequential"
                            " order only if jumping is provided and only for video frames"
                        ),
                        "type": ["integer", "null"],
                    },
                    "order": {
                        "description": (
                            "\n                Input frames order. Values: 'sequential', 'random'\n                In"
                            " Sequential mode frames will be returned according to the order in which the frames were"
                            " added to the dataset."
                        ),
                        "oneOf": [
                            {"$ref": "#/definitions/iteration_order_enum"},
                            {"type": "null"},
                        ],
                    },
                    "random_seed": {
                        "description": "Random seed used when iterating over the dataview",
                        "type": ["integer", "null"],
                    },
                },
                "type": "object",
            },
            "iteration_order_enum": {
                "enum": ["sequential", "random"],
                "type": "string",
            },
            "jump": {
                "properties": {
                    "time": {
                        "description": "Max time in milliseconds between frames",
                        "type": ["integer", "null"],
                    }
                },
                "type": "object",
            },
            "label_source": {
                "properties": {
                    "dataset": {
                        "description": "Source dataset id. '*' for all datasets in view",
                        "type": ["string", "null"],
                    },
                    "labels": {
                        "description": (
                            "List of source labels (AND connection). '*' indicates any label. Labels must exist in at"
                            " least one of the dataset versions in the task's view"
                        ),
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                    "version": {
                        "description": (
                            "Source dataset version id. Default is '*' (for all versions in dataset in the view)"
                            " Version must belong to the selected dataset, and must be in the task's view[i]"
                        ),
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "mapping": {
                "properties": {
                    "rules": {
                        "description": "Rules list",
                        "items": {"$ref": "#/definitions/mapping_rule"},
                        "type": ["array", "null"],
                    }
                },
                "type": "object",
            },
            "mapping_rule": {
                "properties": {
                    "source": {
                        "description": "Source label info",
                        "oneOf": [
                            {"$ref": "#/definitions/label_source"},
                            {"type": "null"},
                        ],
                    },
                    "target": {
                        "description": "Target label name",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "output_rois_enum": {
                "enum": ["all_in_frame", "only_filtered", "frame_per_roi"],
                "type": "string",
            },
        },
        "properties": {
            "augmentation": {
                "$ref": "#/definitions/augmentation",
                "description": "Augmentation parameters. Only for training and testing tasks.",
            },
            "dataview": {"description": "Datatview ID", "type": "string"},
            "description": {"description": "Dataview description", "type": "string"},
            "filters": {
                "description": "List of FilterRule ('OR' connection)",
                "items": {"$ref": "#/definitions/filter_rule"},
                "type": "array",
            },
            "force": {
                "default": False,
                "description": "Allow update of the published dataview",
                "type": "boolean",
            },
            "iteration": {
                "$ref": "#/definitions/iteration",
                "description": "Iteration parameters. Not applicable for register (import) tasks.",
            },
            "labels_enumeration": {
                "additionalProperties": {"type": "integer"},
                "description": (
                    "Labels enumerations, specifies numbers to be assigned to ROI labels when getting frames"
                ),
                "type": "object",
            },
            "mapping": {
                "$ref": "#/definitions/mapping",
                "description": "Mapping parameters",
            },
            "name": {"description": "Dataview name", "type": "string"},
            "output_rois": {
                "$ref": "#/definitions/output_rois_enum",
                "description": (
                    "'all_in_frame' - all rois for a frame are returned\n                    'only_filtered' - only"
                    " rois which led this frame to be selected\n                    'frame_per_roi' - single roi per"
                    " frame. Frame can be returned multiple times with a different roi each time.\n                   "
                    " Note: this should be used for Training tasks only\n                    Note: frame_per_roi"
                    " implies that only filtered rois will be returned\n                    "
                ),
            },
            "project": {
                "description": "Project ID of the project to which this task is assigned",
                "type": "string",
            },
            "status": {
                "description": "Dataview status",
                "enum": ["draft", "published"],
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
            "versions": {
                "description": "List of dataview entries. All tasks must have at least one dataview.",
                "items": {"$ref": "#/definitions/dataview_entry"},
                "type": "array",
            },
        },
        "required": ["dataview"],
        "type": "object",
    }

    def __init__(
        self,
        dataview,
        name=None,
        description=None,
        project=None,
        filters=None,
        output_rois=None,
        versions=None,
        iteration=None,
        augmentation=None,
        tags=None,
        system_tags=None,
        mapping=None,
        labels_enumeration=None,
        status=None,
        force=False,
        **kwargs
    ):
        super(UpdateRequest, self).__init__(**kwargs)
        self.dataview = dataview
        self.name = name
        self.description = description
        self.project = project
        self.filters = filters
        self.output_rois = output_rois
        self.versions = versions
        self.iteration = iteration
        self.augmentation = augmentation
        self.tags = tags
        self.system_tags = system_tags
        self.mapping = mapping
        self.labels_enumeration = labels_enumeration
        self.status = status
        self.force = force

    @schema_property("dataview")
    def dataview(self):
        return self._property_dataview

    @dataview.setter
    def dataview(self, value):
        if value is None:
            self._property_dataview = None
            return

        self.assert_isinstance(value, "dataview", six.string_types)
        self._property_dataview = value

    @schema_property("name")
    def name(self):
        return self._property_name

    @name.setter
    def name(self, value):
        if value is None:
            self._property_name = None
            return

        self.assert_isinstance(value, "name", six.string_types)
        self._property_name = value

    @schema_property("description")
    def description(self):
        return self._property_description

    @description.setter
    def description(self, value):
        if value is None:
            self._property_description = None
            return

        self.assert_isinstance(value, "description", six.string_types)
        self._property_description = value

    @schema_property("project")
    def project(self):
        return self._property_project

    @project.setter
    def project(self, value):
        if value is None:
            self._property_project = None
            return

        self.assert_isinstance(value, "project", six.string_types)
        self._property_project = value

    @schema_property("filters")
    def filters(self):
        return self._property_filters

    @filters.setter
    def filters(self, value):
        if value is None:
            self._property_filters = None
            return

        self.assert_isinstance(value, "filters", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [
                FilterRule.from_dict(v) if isinstance(v, dict) else v for v in value
            ]
        else:
            self.assert_isinstance(value, "filters", FilterRule, is_array=True)
        self._property_filters = value

    @schema_property("output_rois")
    def output_rois(self):
        return self._property_output_rois

    @output_rois.setter
    def output_rois(self, value):
        if value is None:
            self._property_output_rois = None
            return
        if isinstance(value, six.string_types):
            try:
                value = OutputRoisEnum(value)
            except ValueError:
                pass
        else:
            self.assert_isinstance(value, "output_rois", enum.Enum)
        self._property_output_rois = value

    @schema_property("versions")
    def versions(self):
        return self._property_versions

    @versions.setter
    def versions(self, value):
        if value is None:
            self._property_versions = None
            return

        self.assert_isinstance(value, "versions", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [
                DataviewEntry.from_dict(v) if isinstance(v, dict) else v for v in value
            ]
        else:
            self.assert_isinstance(value, "versions", DataviewEntry, is_array=True)
        self._property_versions = value

    @schema_property("iteration")
    def iteration(self):
        return self._property_iteration

    @iteration.setter
    def iteration(self, value):
        if value is None:
            self._property_iteration = None
            return
        if isinstance(value, dict):
            value = Iteration.from_dict(value)
        else:
            self.assert_isinstance(value, "iteration", Iteration)
        self._property_iteration = value

    @schema_property("augmentation")
    def augmentation(self):
        return self._property_augmentation

    @augmentation.setter
    def augmentation(self, value):
        if value is None:
            self._property_augmentation = None
            return
        if isinstance(value, dict):
            value = Augmentation.from_dict(value)
        else:
            self.assert_isinstance(value, "augmentation", Augmentation)
        self._property_augmentation = value

    @schema_property("tags")
    def tags(self):
        return self._property_tags

    @tags.setter
    def tags(self, value):
        if value is None:
            self._property_tags = None
            return

        self.assert_isinstance(value, "tags", (list, tuple))

        self.assert_isinstance(value, "tags", six.string_types, is_array=True)
        self._property_tags = value

    @schema_property("system_tags")
    def system_tags(self):
        return self._property_system_tags

    @system_tags.setter
    def system_tags(self, value):
        if value is None:
            self._property_system_tags = None
            return

        self.assert_isinstance(value, "system_tags", (list, tuple))

        self.assert_isinstance(value, "system_tags", six.string_types, is_array=True)
        self._property_system_tags = value

    @schema_property("mapping")
    def mapping(self):
        return self._property_mapping

    @mapping.setter
    def mapping(self, value):
        if value is None:
            self._property_mapping = None
            return
        if isinstance(value, dict):
            value = Mapping.from_dict(value)
        else:
            self.assert_isinstance(value, "mapping", Mapping)
        self._property_mapping = value

    @schema_property("labels_enumeration")
    def labels_enumeration(self):
        return self._property_labels_enumeration

    @labels_enumeration.setter
    def labels_enumeration(self, value):
        if value is None:
            self._property_labels_enumeration = None
            return

        self.assert_isinstance(value, "labels_enumeration", (dict,))
        self._property_labels_enumeration = value

    @schema_property("status")
    def status(self):
        return self._property_status

    @status.setter
    def status(self, value):
        if value is None:
            self._property_status = None
            return

        self.assert_isinstance(value, "status", six.string_types)
        self._property_status = value

    @schema_property("force")
    def force(self):
        return self._property_force

    @force.setter
    def force(self, value):
        if value is None:
            self._property_force = None
            return

        self.assert_isinstance(value, "force", (bool,))
        self._property_force = value


class UpdateResponse(Response):
    """
    Response of dataviews.update endpoint.

    :param updated: Number of dataviews updated (0 or 1)
    :type updated: float
    :param fields: Updated fields names and values
    :type fields: dict
    """

    _service = "dataviews"
    _action = "update"
    _version = "2.23"

    _schema = {
        "definitions": {},
        "properties": {
            "fields": {
                "additionalProperties": True,
                "description": "Updated fields names and values",
                "type": ["object", "null"],
            },
            "updated": {
                "description": "Number of dataviews updated (0 or 1)",
                "enum": [0, 1],
                "type": ["number", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, updated=None, fields=None, **kwargs):
        super(UpdateResponse, self).__init__(**kwargs)
        self.updated = updated
        self.fields = fields

    @schema_property("updated")
    def updated(self):
        return self._property_updated

    @updated.setter
    def updated(self, value):
        if value is None:
            self._property_updated = None
            return

        self.assert_isinstance(value, "updated", six.integer_types + (float,))
        self._property_updated = value

    @schema_property("fields")
    def fields(self):
        return self._property_fields

    @fields.setter
    def fields(self, value):
        if value is None:
            self._property_fields = None
            return

        self.assert_isinstance(value, "fields", (dict,))
        self._property_fields = value


response_mapping = {
    CreateRequest: CreateResponse,
    GetAllRequest: GetAllResponse,
    GetByIdRequest: GetByIdResponse,
    ArchiveManyRequest: ArchiveManyResponse,
    UnarchiveManyRequest: UnarchiveManyResponse,
    DeleteManyRequest: DeleteManyResponse,
    DeleteRequest: DeleteResponse,
    PublishManyRequest: PublishManyResponse,
    PublishRequest: PublishResponse,
    UpdateRequest: UpdateResponse,
    MoveRequest: MoveResponse,
}
