"""
frames service

This service provides access to frames metadata stored in the
system's datasets (see Datasets Service). This includes reading metadata, adding new
frames/metadata and updating existing metadata.
"""
import six
import enum


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


class ViewEntry(NonStrictDataModel):
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
            "dataset": {
                "description": "Existing Dataset id",
                "type": ["string", "null"],
            },
            "merge_with": {
                "description": "Version ID to merge with",
                "type": ["string", "null"],
            },
            "version": {
                "description": "Version id of a version belonging to the dataset",
                "type": ["string", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, version=None, dataset=None, merge_with=None, **kwargs):
        super(ViewEntry, self).__init__(**kwargs)
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


class FlowControl(NonStrictDataModel):
    """
    :param bidirectional: If set then frames retreival can go either forward or
        backwards. Otherwise only forward. The default is False. The limitations of
        bidirectional navigation: - Frames are always returned in sequential order -
        The iteration is finite (no support for infinite iteration)
    :type bidirectional: bool
    :param navigate_backwards: When bidirectional is True, settings this to True
        navigates backwards duing frames retreival. Default is False
    :type navigate_backwards: bool
    """

    _schema = {
        "properties": {
            "bidirectional": {
                "description": (
                    "If set then frames retreival can go either forward or backwards. Otherwise only forward.\n        "
                    "        The default is False. The limitations of bidirectional navigation:\n                -"
                    " Frames are always returned in sequential order\n                - The iteration is finite (no"
                    " support for infinite iteration)\n                "
                ),
                "type": ["boolean", "null"],
            },
            "navigate_backwards": {
                "description": (
                    "When bidirectional is True, settings this to True navigates backwards duing frames retreival."
                    " Default is False"
                ),
                "type": ["boolean", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, bidirectional=None, navigate_backwards=None, **kwargs):
        super(FlowControl, self).__init__(**kwargs)
        self.bidirectional = bidirectional
        self.navigate_backwards = navigate_backwards

    @schema_property("bidirectional")
    def bidirectional(self):
        return self._property_bidirectional

    @bidirectional.setter
    def bidirectional(self, value):
        if value is None:
            self._property_bidirectional = None
            return

        self.assert_isinstance(value, "bidirectional", (bool,))
        self._property_bidirectional = value

    @schema_property("navigate_backwards")
    def navigate_backwards(self):
        return self._property_navigate_backwards

    @navigate_backwards.setter
    def navigate_backwards(self, value):
        if value is None:
            self._property_navigate_backwards = None
            return

        self.assert_isinstance(value, "navigate_backwards", (bool,))
        self._property_navigate_backwards = value


class Dataview(NonStrictDataModel):
    """
    :param versions: View dataset versions
    :type versions: Sequence[ViewEntry]
    :param filters: List of FilterRule ('OR' relationship)
    :type filters: Sequence[FilterRule]
    :param output_rois: 'all_in_frame' - all rois for a frame are returned
        'only_filtered' - only rois which led this frame to be selected
        'frame_per_roi' - single roi per frame. Frame can be returned multiple times
        with a different roi each time.
        Note: this should be used for Training tasks only
        Note: frame_per_roi implies that only filtered rois will be returned
    :type output_rois: OutputRoisEnum
    :param iteration: Iteration parameters. Not applicable for register (import)
        tasks.
    :type iteration: Iteration
    :param augmentation: Augmentation parameters. Only for training and testing
        tasks.
    :type augmentation: DvAugmentation
    :param mapping: Mapping parameters
    :type mapping: Mapping
    :param labels_enumeration: Labels enumerations, specifies numbers to be
        assigned to ROI labels when getting frames
    :type labels_enumeration: dict
    """

    _schema = {
        "properties": {
            "augmentation": {
                "description": "Augmentation parameters. Only for training and testing tasks.",
                "oneOf": [{"$ref": "#/definitions/dv_augmentation"}, {"type": "null"}],
            },
            "filters": {
                "description": "List of FilterRule ('OR' relationship)",
                "items": {"$ref": "#/definitions/filter_rule"},
                "type": ["array", "null"],
            },
            "iteration": {
                "description": "Iteration parameters. Not applicable for register (import) tasks.",
                "oneOf": [{"$ref": "#/definitions/iteration"}, {"type": "null"}],
            },
            "labels_enumeration": {
                "additionalProperties": {"type": "integer"},
                "description": (
                    "Labels enumerations, specifies numbers to be assigned to ROI labels when getting frames"
                ),
                "type": ["object", "null"],
            },
            "mapping": {
                "description": "Mapping parameters",
                "oneOf": [{"$ref": "#/definitions/mapping"}, {"type": "null"}],
            },
            "output_rois": {
                "description": (
                    "'all_in_frame' - all rois for a frame are returned\n\n'only_filtered' - only rois which led this"
                    " frame to be selected\n\n'frame_per_roi' - single roi per frame. Frame can be returned multiple"
                    " times with a different roi each time.\n\nNote: this should be used for Training tasks"
                    " only\n\nNote: frame_per_roi implies that only filtered rois will be returned\n                "
                ),
                "oneOf": [{"$ref": "#/definitions/output_rois_enum"}, {"type": "null"}],
            },
            "versions": {
                "description": "View dataset versions",
                "items": {"$ref": "#/definitions/view_entry"},
                "type": ["array", "null"],
            },
        },
        "type": "object",
    }

    def __init__(
        self,
        versions=None,
        filters=None,
        output_rois=None,
        iteration=None,
        augmentation=None,
        mapping=None,
        labels_enumeration=None,
        **kwargs
    ):
        super(Dataview, self).__init__(**kwargs)
        self.versions = versions
        self.filters = filters
        self.output_rois = output_rois
        self.iteration = iteration
        self.augmentation = augmentation
        self.mapping = mapping
        self.labels_enumeration = labels_enumeration

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
                ViewEntry.from_dict(v) if isinstance(v, dict) else v for v in value
            ]
        else:
            self.assert_isinstance(value, "versions", ViewEntry, is_array=True)
        self._property_versions = value

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
            value = DvAugmentation.from_dict(value)
        else:
            self.assert_isinstance(value, "augmentation", DvAugmentation)
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


class OutputRoisEnum(StringEnum):
    all_in_frame = "all_in_frame"
    only_filtered = "only_filtered"
    frame_per_roi = "frame_per_roi"


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


class DvAugmentationSet(NonStrictDataModel):
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
        super(DvAugmentationSet, self).__init__(**kwargs)
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


class DvAugmentation(NonStrictDataModel):
    """
    :param sets: List of augmentation sets
    :type sets: Sequence[DvAugmentationSet]
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
                "items": {"$ref": "#/definitions/dv_augmentation_set"},
                "type": ["array", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, sets=None, crop_around_rois=None, **kwargs):
        super(DvAugmentation, self).__init__(**kwargs)
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
                DvAugmentationSet.from_dict(v) if isinstance(v, dict) else v
                for v in value
            ]
        else:
            self.assert_isinstance(value, "sets", DvAugmentationSet, is_array=True)
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


class Augmentation(NonStrictDataModel):
    """
    :param cls: Augmentation class (see global definitions)
    :type cls: str
    :param type: Augmentation type (see global definitions)
    :type type: str
    :param trans_mat: Transform matrix (list of lists). Required for affine
        transforms.
    :type trans_mat: Sequence[Sequence[float]]
    :param params: Transform parameters, an array ot 3 randomly generated values.
        Fixed values are passed in case of affine reflect augmentation.
    :type params: Sequence[float]
    :param arguments: Arguments dictionary, passed to custom augmentations.
    :type arguments: dict
    :param strength: Transform strength. Required for pixel transforms.
    :type strength: float
    """

    _schema = {
        "properties": {
            "arguments": {
                "additionalProperties": True,
                "description": "Arguments dictionary, passed to custom augmentations.",
                "type": ["object", "null"],
            },
            "cls": {
                "description": "Augmentation class (see global definitions)",
                "type": ["string", "null"],
            },
            "params": {
                "description": (
                    "Transform parameters, an array ot 3 randomly generated values. Fixed values are passed in case of"
                    " affine reflect augmentation."
                ),
                "items": {"type": "number"},
                "type": ["array", "null"],
            },
            "strength": {
                "description": "Transform strength. Required for pixel transforms.",
                "type": ["number", "null"],
            },
            "trans_mat": {
                "description": "Transform matrix (list of lists). Required for affine transforms.",
                "items": {"items": {"type": "number"}, "type": "array"},
                "type": ["array", "null"],
            },
            "type": {
                "description": "Augmentation type (see global definitions)",
                "type": ["string", "null"],
            },
        },
        "type": "object",
    }

    def __init__(
        self,
        cls=None,
        type=None,
        trans_mat=None,
        params=None,
        arguments=None,
        strength=None,
        **kwargs
    ):
        super(Augmentation, self).__init__(**kwargs)
        self.cls = cls
        self.type = type
        self.trans_mat = trans_mat
        self.params = params
        self.arguments = arguments
        self.strength = strength

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

    @schema_property("type")
    def type(self):
        return self._property_type

    @type.setter
    def type(self, value):
        if value is None:
            self._property_type = None
            return

        self.assert_isinstance(value, "type", six.string_types)
        self._property_type = value

    @schema_property("trans_mat")
    def trans_mat(self):
        return self._property_trans_mat

    @trans_mat.setter
    def trans_mat(self, value):
        if value is None:
            self._property_trans_mat = None
            return

        self.assert_isinstance(value, "trans_mat", (list, tuple))

        self.assert_isinstance(value, "trans_mat", (list, tuple), is_array=True)
        self._property_trans_mat = value

    @schema_property("params")
    def params(self):
        return self._property_params

    @params.setter
    def params(self, value):
        if value is None:
            self._property_params = None
            return

        self.assert_isinstance(value, "params", (list, tuple))

        self.assert_isinstance(
            value, "params", six.integer_types + (float,), is_array=True
        )
        self._property_params = value

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


class DatasetVersion(NonStrictDataModel):
    """
    :param id: Dataset id
    :type id: str
    :param version: Dataset version id
    :type version: str
    """

    _schema = {
        "properties": {
            "id": {"description": "Dataset id", "type": ["string", "null"]},
            "version": {
                "description": "Dataset version id",
                "type": ["string", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, id=None, version=None, **kwargs):
        super(DatasetVersion, self).__init__(**kwargs)
        self.id = id
        self.version = version

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


class RoiMask(NonStrictDataModel):
    """
    :param id: Mask ID
    :type id: str
    :param value: Mask value
    :type value: Sequence[int]
    """

    _schema = {
        "properties": {
            "id": {"description": "Mask ID", "type": "string"},
            "value": {
                "description": "Mask value",
                "items": {"type": "integer"},
                "type": "array",
            },
        },
        "required": ["id", "value"],
        "type": "object",
    }

    def __init__(self, id, value, **kwargs):
        super(RoiMask, self).__init__(**kwargs)
        self.id = id
        self.value = value

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

    @schema_property("value")
    def value(self):
        return self._property_value

    @value.setter
    def value(self, value):
        if value is None:
            self._property_value = None
            return

        self.assert_isinstance(value, "value", (list, tuple))
        value = [
            int(v) if isinstance(v, float) and v.is_integer() else v for v in value
        ]

        self.assert_isinstance(value, "value", six.integer_types, is_array=True)
        self._property_value = value


class Roi(NonStrictDataModel):
    """
    :param id: ROI id
    :type id: str
    :param sources: Sources that this ROI belongs to
    :type sources: Sequence[str]
    :param label: ROI labels
    :type label: Sequence[str]
    :param label_num: Label number according to the specified labels mapping Used
        only when ROI is returned as part of a task's frame.
    :type label_num: int
    :param poly: ROI polygon (x0, y0, ..., xn, yn)
    :type poly: Sequence[float]
    :param confidence: ROI confidence
    :type confidence: float
    :param area: ROI area (not used)
    :type area: int
    :param meta: Additional metadata dictionary for the roi
    :type meta: dict
    :param mask: Mask info for this ROI
    :type mask: RoiMask
    """

    _schema = {
        "properties": {
            "area": {"description": "ROI area (not used)", "type": ["integer", "null"]},
            "confidence": {"description": "ROI confidence", "type": ["number", "null"]},
            "id": {"description": "ROI id", "type": ["string", "null"]},
            "label": {
                "description": "ROI labels",
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
            "label_num": {
                "description": (
                    "Label number according to the specified labels mapping Used only when ROI is returned as "
                    "part of a task's frame."
                ),
                "type": ["integer", "null"],
            },
            "mask": {
                "description": "Mask info for this ROI",
                "oneOf": [{"$ref": "#/definitions/roi_mask"}, {"type": "null"}],
            },
            "meta": {
                "additionalProperties": True,
                "description": "Additional metadata dictionary for the roi",
                "type": ["object", "null"],
            },
            "poly": {
                "description": "ROI polygon (x0, y0, ..., xn, yn)",
                "items": {"type": "number"},
                "type": ["array", "null"],
            },
            "sources": {
                "description": "Sources that this ROI belongs to",
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
        },
        "type": "object",
    }

    def __init__(
        self,
        id=None,
        sources=None,
        label=None,
        label_num=None,
        poly=None,
        confidence=None,
        area=None,
        meta=None,
        mask=None,
        **kwargs
    ):
        super(Roi, self).__init__(**kwargs)
        self.id = id
        self.sources = sources
        self.label = label
        self.label_num = label_num
        self.poly = poly
        self.confidence = confidence
        self.area = area
        self.meta = meta
        self.mask = mask

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

    @schema_property("sources")
    def sources(self):
        return self._property_sources

    @sources.setter
    def sources(self, value):
        if value is None:
            self._property_sources = None
            return

        self.assert_isinstance(value, "sources", (list, tuple))

        self.assert_isinstance(value, "sources", six.string_types, is_array=True)
        self._property_sources = value

    @schema_property("label")
    def label(self):
        return self._property_label

    @label.setter
    def label(self, value):
        if value is None:
            self._property_label = None
            return

        self.assert_isinstance(value, "label", (list, tuple))

        self.assert_isinstance(value, "label", six.string_types, is_array=True)
        self._property_label = value

    @schema_property("label_num")
    def label_num(self):
        return self._property_label_num

    @label_num.setter
    def label_num(self, value):
        if value is None:
            self._property_label_num = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "label_num", six.integer_types)
        self._property_label_num = value

    @schema_property("poly")
    def poly(self):
        return self._property_poly

    @poly.setter
    def poly(self, value):
        if value is None:
            self._property_poly = None
            return

        self.assert_isinstance(value, "poly", (list, tuple))

        self.assert_isinstance(
            value, "poly", six.integer_types + (float,), is_array=True
        )
        self._property_poly = value

    @schema_property("confidence")
    def confidence(self):
        return self._property_confidence

    @confidence.setter
    def confidence(self, value):
        if value is None:
            self._property_confidence = None
            return

        self.assert_isinstance(value, "confidence", six.integer_types + (float,))
        self._property_confidence = value

    @schema_property("area")
    def area(self):
        return self._property_area

    @area.setter
    def area(self, value):
        if value is None:
            self._property_area = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "area", six.integer_types)
        self._property_area = value

    @schema_property("meta")
    def meta(self):
        return self._property_meta

    @meta.setter
    def meta(self, value):
        if value is None:
            self._property_meta = None
            return

        self.assert_isinstance(value, "meta", (dict,))
        self._property_meta = value

    @schema_property("mask")
    def mask(self):
        return self._property_mask

    @mask.setter
    def mask(self, value):
        if value is None:
            self._property_mask = None
            return
        if isinstance(value, dict):
            value = RoiMask.from_dict(value)
        else:
            self.assert_isinstance(value, "mask", RoiMask)
        self._property_mask = value


class Mask(NonStrictDataModel):
    """
    :param id: unique ID (in this frame)
    :type id: str
    :param uri: Data URI
    :type uri: str
    :param content_type: Content type (e.g. 'image/jpeg', 'image/png')
    :type content_type: str
    :param width: Width in pixels
    :type width: int
    :param height: Height in pixels
    :type height: int
    :param timestamp: Timestamp in the source data (for video content. for images,
        this value should be 0)
    :type timestamp: int
    """

    _schema = {
        "properties": {
            "content_type": {
                "description": "Content type (e.g. 'image/jpeg', 'image/png')",
                "type": ["string", "null"],
            },
            "height": {"description": "Height in pixels", "type": ["integer", "null"]},
            "id": {
                "description": "unique ID (in this frame)",
                "type": ["string", "null"],
            },
            "timestamp": {
                "default": 0,
                "description": "Timestamp in the source data (for video content. for images, this value should be 0)",
                "type": ["integer", "null"],
            },
            "uri": {"description": "Data URI", "type": ["string", "null"]},
            "width": {"description": "Width in pixels", "type": ["integer", "null"]},
        },
        "type": "object",
    }

    def __init__(
        self,
        id=None,
        uri=None,
        content_type=None,
        width=None,
        height=None,
        timestamp=0,
        **kwargs
    ):
        super(Mask, self).__init__(**kwargs)
        self.id = id
        self.uri = uri
        self.content_type = content_type
        self.width = width
        self.height = height
        self.timestamp = timestamp

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

    @schema_property("uri")
    def uri(self):
        return self._property_uri

    @uri.setter
    def uri(self, value):
        if value is None:
            self._property_uri = None
            return

        self.assert_isinstance(value, "uri", six.string_types)
        self._property_uri = value

    @schema_property("content_type")
    def content_type(self):
        return self._property_content_type

    @content_type.setter
    def content_type(self, value):
        if value is None:
            self._property_content_type = None
            return

        self.assert_isinstance(value, "content_type", six.string_types)
        self._property_content_type = value

    @schema_property("width")
    def width(self):
        return self._property_width

    @width.setter
    def width(self, value):
        if value is None:
            self._property_width = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "width", six.integer_types)
        self._property_width = value

    @schema_property("height")
    def height(self):
        return self._property_height

    @height.setter
    def height(self, value):
        if value is None:
            self._property_height = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "height", six.integer_types)
        self._property_height = value

    @schema_property("timestamp")
    def timestamp(self):
        return self._property_timestamp

    @timestamp.setter
    def timestamp(self, value):
        if value is None:
            self._property_timestamp = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "timestamp", six.integer_types)
        self._property_timestamp = value


class Preview(NonStrictDataModel):
    """
    :param uri: Data URI
    :type uri: str
    :param content_type: Content type (e.g. 'image/jpeg', 'image/png')
    :type content_type: str
    :param width: Width in pixels
    :type width: int
    :param height: Height in pixels
    :type height: int
    :param timestamp: Timestamp in the source data (for video content. for images,
        this value should be 0)
    :type timestamp: int
    """

    _schema = {
        "properties": {
            "content_type": {
                "description": "Content type (e.g. 'image/jpeg', 'image/png')",
                "type": ["string", "null"],
            },
            "height": {"description": "Height in pixels", "type": ["integer", "null"]},
            "timestamp": {
                "default": 0,
                "description": "Timestamp in the source data (for video content. for images, this value should be 0)",
                "type": ["integer", "null"],
            },
            "uri": {"description": "Data URI", "type": ["string", "null"]},
            "width": {"description": "Width in pixels", "type": ["integer", "null"]},
        },
        "type": "object",
    }

    def __init__(
        self,
        uri=None,
        content_type=None,
        width=None,
        height=None,
        timestamp=0,
        **kwargs
    ):
        super(Preview, self).__init__(**kwargs)
        self.uri = uri
        self.content_type = content_type
        self.width = width
        self.height = height
        self.timestamp = timestamp

    @schema_property("uri")
    def uri(self):
        return self._property_uri

    @uri.setter
    def uri(self, value):
        if value is None:
            self._property_uri = None
            return

        self.assert_isinstance(value, "uri", six.string_types)
        self._property_uri = value

    @schema_property("content_type")
    def content_type(self):
        return self._property_content_type

    @content_type.setter
    def content_type(self, value):
        if value is None:
            self._property_content_type = None
            return

        self.assert_isinstance(value, "content_type", six.string_types)
        self._property_content_type = value

    @schema_property("width")
    def width(self):
        return self._property_width

    @width.setter
    def width(self, value):
        if value is None:
            self._property_width = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "width", six.integer_types)
        self._property_width = value

    @schema_property("height")
    def height(self):
        return self._property_height

    @height.setter
    def height(self, value):
        if value is None:
            self._property_height = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "height", six.integer_types)
        self._property_height = value

    @schema_property("timestamp")
    def timestamp(self):
        return self._property_timestamp

    @timestamp.setter
    def timestamp(self, value):
        if value is None:
            self._property_timestamp = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "timestamp", six.integer_types)
        self._property_timestamp = value


class Source(NonStrictDataModel):
    """
    :param id: unique ID (in this frame)
    :type id: str
    :param uri: Data URI
    :type uri: str
    :param content_type: Content type (e.g. 'image/jpeg', 'image/png')
    :type content_type: str
    :param width: Width in pixels
    :type width: int
    :param height: Height in pixels
    :type height: int
    :param timestamp: Timestamp in the source data (for video content. for images,
        this value should be 0)
    :type timestamp: int
    :param masks:
    :type masks: Sequence[Mask]
    :param preview:
    :type preview: Preview
    :param meta: Additional metadata dictionary for the source
    :type meta: dict
    """

    _schema = {
        "properties": {
            "content_type": {
                "description": "Content type (e.g. 'image/jpeg', 'image/png')",
                "type": ["string", "null"],
            },
            "height": {"description": "Height in pixels", "type": ["integer", "null"]},
            "id": {
                "description": "unique ID (in this frame)",
                "type": ["string", "null"],
            },
            "masks": {
                "items": {"$ref": "#/definitions/mask"},
                "type": ["array", "null"],
            },
            "meta": {
                "additionalProperties": True,
                "description": "Additional metadata dictionary for the source",
                "type": ["object", "null"],
            },
            "preview": {"oneOf": [{"$ref": "#/definitions/preview"}, {"type": "null"}]},
            "timestamp": {
                "default": 0,
                "description": "Timestamp in the source data (for video content. for images, this value should be 0)",
                "type": ["integer", "null"],
            },
            "uri": {"description": "Data URI", "type": ["string", "null"]},
            "width": {"description": "Width in pixels", "type": ["integer", "null"]},
        },
        "type": "object",
    }

    def __init__(
        self,
        id=None,
        uri=None,
        content_type=None,
        width=None,
        height=None,
        timestamp=0,
        masks=None,
        preview=None,
        meta=None,
        **kwargs
    ):
        super(Source, self).__init__(**kwargs)
        self.id = id
        self.uri = uri
        self.content_type = content_type
        self.width = width
        self.height = height
        self.timestamp = timestamp
        self.masks = masks
        self.preview = preview
        self.meta = meta

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

    @schema_property("uri")
    def uri(self):
        return self._property_uri

    @uri.setter
    def uri(self, value):
        if value is None:
            self._property_uri = None
            return

        self.assert_isinstance(value, "uri", six.string_types)
        self._property_uri = value

    @schema_property("content_type")
    def content_type(self):
        return self._property_content_type

    @content_type.setter
    def content_type(self, value):
        if value is None:
            self._property_content_type = None
            return

        self.assert_isinstance(value, "content_type", six.string_types)
        self._property_content_type = value

    @schema_property("width")
    def width(self):
        return self._property_width

    @width.setter
    def width(self, value):
        if value is None:
            self._property_width = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "width", six.integer_types)
        self._property_width = value

    @schema_property("height")
    def height(self):
        return self._property_height

    @height.setter
    def height(self, value):
        if value is None:
            self._property_height = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "height", six.integer_types)
        self._property_height = value

    @schema_property("timestamp")
    def timestamp(self):
        return self._property_timestamp

    @timestamp.setter
    def timestamp(self, value):
        if value is None:
            self._property_timestamp = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "timestamp", six.integer_types)
        self._property_timestamp = value

    @schema_property("masks")
    def masks(self):
        return self._property_masks

    @masks.setter
    def masks(self, value):
        if value is None:
            self._property_masks = None
            return

        self.assert_isinstance(value, "masks", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [Mask.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "masks", Mask, is_array=True)
        self._property_masks = value

    @schema_property("preview")
    def preview(self):
        return self._property_preview

    @preview.setter
    def preview(self, value):
        if value is None:
            self._property_preview = None
            return
        if isinstance(value, dict):
            value = Preview.from_dict(value)
        else:
            self.assert_isinstance(value, "preview", Preview)
        self._property_preview = value

    @schema_property("meta")
    def meta(self):
        return self._property_meta

    @meta.setter
    def meta(self, value):
        if value is None:
            self._property_meta = None
            return

        self.assert_isinstance(value, "meta", (dict,))
        self._property_meta = value


class Frame(NonStrictDataModel):
    """
    :param id: Frame id
    :type id: str
    :param augmentation: List of augmentations
    :type augmentation: Sequence[Augmentation]
    :param timestamp: Frame's offset in milliseconds, used primarily for video
        content. Used for the default frames sorting as the secondary key (with the
        primary key being 'context_id'). For images, this value should typically be 0.
        If not set, value is filled from the timestamp of the first source. We
        recommend using this field only in cases concerning the default sorting
        behavior.
    :type timestamp: int
    :param dataset: Frame's dataset version
    :type dataset: DatasetVersion
    :param saved: Last time frame was saved (timestamp)
    :type saved: int
    :param saved_in_version: Last version this frame was saved in (version ID)
    :type saved_in_version: str
    :param updated: Last time frame was saved (timestamp)
    :type updated: int
    :param updated_in_version: Last version this frame was updated in (version ID)
    :type updated_in_version: str
    :param rois: Frame regions of interest
    :type rois: Sequence[Roi]
    :param labels_size: Number of labels returned
    :type labels_size: int
    :param rule_name: Name of the filtering rule according to which this frame was
        provided (if applicable)
    :type rule_name: str
    :param video_gop: Video encoding GOP value for the source of this frame. Only
        valid for video frames
    :type video_gop: float
    :param is_key_frame: Is this a key frame (only applicable in frames who'se src
        is a video)
    :type is_key_frame: bool
    :param key_frame: ID of the key frame that this frame belongs to
    :type key_frame: str
    :param meta: Additional metadata dictionary for the frame. Please note that
        using this field effectively defines a schema (dictionary structure and types
        used as values) - frames within the same dataset cannot use conflicting schemas
        for this field (see documentation for more details).
    :type meta: dict
    :param blob: Raw data (blob) for the frame
    :type blob: str
    :param meta_blob: Non searchable metadata dictionary for the frame. The fields
        in this object cannot be searched by and are not added to the frame schema
    :type meta_blob: dict
    :param new_ver: Newer version of this frame, if asked to merge
    :type new_ver: Frame
    :param label_rule_counts: The number of matched roi per lable rule
    :type label_rule_counts: dict
    :param sources: Sources of this frame
    :type sources: Sequence[Source]
    :param context_id: Context ID. Used for the default frames sorting. If not set
        then it is filled from the uri of the first source.
    :type context_id: str
    """

    _schema = {
        "properties": {
            "augmentation": {
                "description": "List of augmentations",
                "items": {"$ref": "#/definitions/augmentation"},
                "type": ["array", "null"],
            },
            "blob": {
                "description": "Raw data (blob) for the frame",
                "type": ["string", "null"],
            },
            "context_id": {
                "description": (
                    "Context ID. Used for the default frames sorting. If not set then it is filled from the "
                    "uri of the first source."
                ),
                "type": ["string", "null"],
            },
            "dataset": {
                "description": "Frame's dataset version",
                "oneOf": [{"$ref": "#/definitions/dataset_version"}, {"type": "null"}],
            },
            "id": {"description": "Frame id", "type": ["string", "null"]},
            "is_key_frame": {
                "description": "Is this a key frame (only applicable in frames who'se src is a video)",
                "type": ["boolean", "null"],
            },
            "key_frame": {
                "description": "ID of the key frame that this frame belongs to",
                "type": ["string", "null"],
            },
            "label_rule_counts": {
                "additionalProperties": True,
                "description": "The number of matched roi per lable rule",
                "type": ["object", "null"],
            },
            "labels_size": {
                "description": "Number of labels returned",
                "type": ["integer", "null"],
            },
            "meta": {
                "additionalProperties": True,
                "description": (
                    "Additional metadata dictionary for the frame. Please note that using this field effectively"
                    " defines a schema (dictionary structure and types used as values) - frames within the same dataset"
                    " cannot use conflicting schemas for this field (see documentation for more details)."
                ),
                "type": ["object", "null"],
            },
            "meta_blob": {
                "additionalProperties": True,
                "description": (
                    "Non searchable metadata dictionary for the frame. The fields in this object cannot be searched by"
                    " and are not added to the frame schema"
                ),
                "type": ["object", "null"],
            },
            "new_ver": {
                "description": "Newer version of this frame, if asked to merge",
                "oneOf": [{"$ref": "#/definitions/frame"}, {"type": "null"}],
            },
            "rois": {
                "description": "Frame regions of interest",
                "items": {"$ref": "#/definitions/roi"},
                "type": ["array", "null"],
            },
            "rule_name": {
                "description": "Name of the filtering rule according to which this frame was provided (if applicable)",
                "type": ["string", "null"],
            },
            "saved": {
                "description": "Last time frame was saved (timestamp)",
                "type": ["integer", "null"],
            },
            "saved_in_version": {
                "description": "Last version this frame was saved in (version ID)",
                "type": ["string", "null"],
            },
            "sources": {
                "description": "Sources of this frame",
                "items": {"$ref": "#/definitions/source"},
                "type": ["array", "null"],
            },
            "timestamp": {
                "description": (
                    "Frame's offset in milliseconds, used primarily for video content. Used for the default frames"
                    " sorting as the secondary key (with the primary key being 'context_id'). For images, this value"
                    " should typically be 0. If not set, value is filled from the timestamp of the first source. We"
                    " recommend using this field only in cases concerning the default sorting behavior."
                ),
                "type": ["integer", "null"],
            },
            "updated": {
                "description": "Last time frame was saved (timestamp)",
                "type": ["integer", "null"],
            },
            "updated_in_version": {
                "description": "Last version this frame was updated in (version ID)",
                "type": ["string", "null"],
            },
            "video_gop": {
                "description": "Video encoding GOP value for the source of this frame. Only valid for video frames",
                "type": ["number", "null"],
            },
        },
        "type": "object",
    }

    def __init__(
        self,
        id=None,
        augmentation=None,
        timestamp=None,
        dataset=None,
        saved=None,
        saved_in_version=None,
        updated=None,
        updated_in_version=None,
        rois=None,
        labels_size=None,
        rule_name=None,
        video_gop=None,
        is_key_frame=None,
        key_frame=None,
        meta=None,
        blob=None,
        meta_blob=None,
        new_ver=None,
        label_rule_counts=None,
        sources=None,
        context_id=None,
        **kwargs
    ):
        super(Frame, self).__init__(**kwargs)
        self.id = id
        self.augmentation = augmentation
        self.timestamp = timestamp
        self.dataset = dataset
        self.saved = saved
        self.saved_in_version = saved_in_version
        self.updated = updated
        self.updated_in_version = updated_in_version
        self.rois = rois
        self.labels_size = labels_size
        self.rule_name = rule_name
        self.video_gop = video_gop
        self.is_key_frame = is_key_frame
        self.key_frame = key_frame
        self.meta = meta
        self.blob = blob
        self.meta_blob = meta_blob
        self.new_ver = new_ver
        self.label_rule_counts = label_rule_counts
        self.sources = sources
        self.context_id = context_id

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

    @schema_property("augmentation")
    def augmentation(self):
        return self._property_augmentation

    @augmentation.setter
    def augmentation(self, value):
        if value is None:
            self._property_augmentation = None
            return

        self.assert_isinstance(value, "augmentation", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [
                Augmentation.from_dict(v) if isinstance(v, dict) else v for v in value
            ]
        else:
            self.assert_isinstance(value, "augmentation", Augmentation, is_array=True)
        self._property_augmentation = value

    @schema_property("timestamp")
    def timestamp(self):
        return self._property_timestamp

    @timestamp.setter
    def timestamp(self, value):
        if value is None:
            self._property_timestamp = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "timestamp", six.integer_types)
        self._property_timestamp = value

    @schema_property("dataset")
    def dataset(self):
        return self._property_dataset

    @dataset.setter
    def dataset(self, value):
        if value is None:
            self._property_dataset = None
            return
        if isinstance(value, dict):
            value = DatasetVersion.from_dict(value)
        else:
            self.assert_isinstance(value, "dataset", DatasetVersion)
        self._property_dataset = value

    @schema_property("saved")
    def saved(self):
        return self._property_saved

    @saved.setter
    def saved(self, value):
        if value is None:
            self._property_saved = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "saved", six.integer_types)
        self._property_saved = value

    @schema_property("saved_in_version")
    def saved_in_version(self):
        return self._property_saved_in_version

    @saved_in_version.setter
    def saved_in_version(self, value):
        if value is None:
            self._property_saved_in_version = None
            return

        self.assert_isinstance(value, "saved_in_version", six.string_types)
        self._property_saved_in_version = value

    @schema_property("updated")
    def updated(self):
        return self._property_updated

    @updated.setter
    def updated(self, value):
        if value is None:
            self._property_updated = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "updated", six.integer_types)
        self._property_updated = value

    @schema_property("updated_in_version")
    def updated_in_version(self):
        return self._property_updated_in_version

    @updated_in_version.setter
    def updated_in_version(self, value):
        if value is None:
            self._property_updated_in_version = None
            return

        self.assert_isinstance(value, "updated_in_version", six.string_types)
        self._property_updated_in_version = value

    @schema_property("rois")
    def rois(self):
        return self._property_rois

    @rois.setter
    def rois(self, value):
        if value is None:
            self._property_rois = None
            return

        self.assert_isinstance(value, "rois", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [Roi.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "rois", Roi, is_array=True)
        self._property_rois = value

    @schema_property("labels_size")
    def labels_size(self):
        return self._property_labels_size

    @labels_size.setter
    def labels_size(self, value):
        if value is None:
            self._property_labels_size = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "labels_size", six.integer_types)
        self._property_labels_size = value

    @schema_property("rule_name")
    def rule_name(self):
        return self._property_rule_name

    @rule_name.setter
    def rule_name(self, value):
        if value is None:
            self._property_rule_name = None
            return

        self.assert_isinstance(value, "rule_name", six.string_types)
        self._property_rule_name = value

    @schema_property("video_gop")
    def video_gop(self):
        return self._property_video_gop

    @video_gop.setter
    def video_gop(self, value):
        if value is None:
            self._property_video_gop = None
            return

        self.assert_isinstance(value, "video_gop", six.integer_types + (float,))
        self._property_video_gop = value

    @schema_property("is_key_frame")
    def is_key_frame(self):
        return self._property_is_key_frame

    @is_key_frame.setter
    def is_key_frame(self, value):
        if value is None:
            self._property_is_key_frame = None
            return

        self.assert_isinstance(value, "is_key_frame", (bool,))
        self._property_is_key_frame = value

    @schema_property("key_frame")
    def key_frame(self):
        return self._property_key_frame

    @key_frame.setter
    def key_frame(self, value):
        if value is None:
            self._property_key_frame = None
            return

        self.assert_isinstance(value, "key_frame", six.string_types)
        self._property_key_frame = value

    @schema_property("meta")
    def meta(self):
        return self._property_meta

    @meta.setter
    def meta(self, value):
        if value is None:
            self._property_meta = None
            return

        self.assert_isinstance(value, "meta", (dict,))
        self._property_meta = value

    @schema_property("blob")
    def blob(self):
        return self._property_blob

    @blob.setter
    def blob(self, value):
        if value is None:
            self._property_blob = None
            return

        self.assert_isinstance(value, "blob", six.string_types)
        self._property_blob = value

    @schema_property("meta_blob")
    def meta_blob(self):
        return self._property_meta_blob

    @meta_blob.setter
    def meta_blob(self, value):
        if value is None:
            self._property_meta_blob = None
            return

        self.assert_isinstance(value, "meta_blob", (dict,))
        self._property_meta_blob = value

    @schema_property("new_ver")
    def new_ver(self):
        return self._property_new_ver

    @new_ver.setter
    def new_ver(self, value):
        if value is None:
            self._property_new_ver = None
            return
        if isinstance(value, dict):
            value = Frame.from_dict(value)
        else:
            self.assert_isinstance(value, "new_ver", Frame)
        self._property_new_ver = value

    @schema_property("label_rule_counts")
    def label_rule_counts(self):
        return self._property_label_rule_counts

    @label_rule_counts.setter
    def label_rule_counts(self, value):
        if value is None:
            self._property_label_rule_counts = None
            return

        self.assert_isinstance(value, "label_rule_counts", (dict,))
        self._property_label_rule_counts = value

    @schema_property("sources")
    def sources(self):
        return self._property_sources

    @sources.setter
    def sources(self, value):
        if value is None:
            self._property_sources = None
            return

        self.assert_isinstance(value, "sources", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [Source.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "sources", Source, is_array=True)
        self._property_sources = value

    @schema_property("context_id")
    def context_id(self):
        return self._property_context_id

    @context_id.setter
    def context_id(self, value):
        if value is None:
            self._property_context_id = None
            return

        self.assert_isinstance(value, "context_id", six.string_types)
        self._property_context_id = value


class Snippet(NonStrictDataModel):
    """
    :param id: Frame id
    :type id: str
    :param augmentation: List of augmentations
    :type augmentation: Sequence[Augmentation]
    :param timestamp: Frame's offset in milliseconds, used primarily for video
        content. Used for the default frames sorting as the secondary key (with the
        primary key being 'context_id'). For images, this value should typically be 0.
        If not set, value is filled from the timestamp of the first source. We
        recommend using this field only in cases concerning the default sorting
        behavior.
    :type timestamp: int
    :param dataset: Frame's dataset version
    :type dataset: DatasetVersion
    :param saved: Last time frame was saved (timestamp)
    :type saved: int
    :param saved_in_version: Last version this frame was saved in (version ID)
    :type saved_in_version: str
    :param updated: Last time frame was saved (timestamp)
    :type updated: int
    :param updated_in_version: Last version this frame was updated in (version ID)
    :type updated_in_version: str
    :param rois: Frame regions of interest
    :type rois: Sequence[Roi]
    :param labels_size: Number of labels returned
    :type labels_size: int
    :param rule_name: Name of the filtering rule according to which this frame was
        provided (if applicable)
    :type rule_name: str
    :param video_gop: Video encoding GOP value for the source of this frame. Only
        valid for video frames
    :type video_gop: float
    :param is_key_frame: Is this a key frame (only applicable in frames who'se src
        is a video)
    :type is_key_frame: bool
    :param key_frame: ID of the key frame that this frame belongs to
    :type key_frame: str
    :param meta: Additional metadata dictionary for the frame. Please note that
        using this field effectively defines a schema (dictionary structure and types
        used as values) - frames within the same dataset cannot use conflicting schemas
        for this field (see documentation for more details).
    :type meta: dict
    :param blob: Raw data (blob) for the frame
    :type blob: str
    :param meta_blob: Non searchable metadata dictionary for the frame. The fields
        in this object cannot be searched by and are not added to the frame schema
    :type meta_blob: dict
    :param new_ver: Newer version of this frame, if asked to merge
    :type new_ver: Frame
    :param label_rule_counts: The number of matched roi per lable rule
    :type label_rule_counts: dict
    :param sources: Sources of this frame
    :type sources: Sequence[Source]
    :param context_id: Context ID. Used for the default frames sorting. If not set
        then it is filled from the uri of the first source.
    :type context_id: str
    :param num_frames: Number of frames represented by this snippet
    :type num_frames: int
    """

    _schema = {
        "properties": {
            "augmentation": {
                "description": "List of augmentations",
                "items": {"$ref": "#/definitions/augmentation"},
                "type": ["array", "null"],
            },
            "blob": {
                "description": "Raw data (blob) for the frame",
                "type": ["string", "null"],
            },
            "context_id": {
                "description": (
                    "Context ID. Used for the default frames sorting. If not set then it is filled from the "
                    "uri of the first source."
                ),
                "type": ["string", "null"],
            },
            "dataset": {
                "description": "Frame's dataset version",
                "oneOf": [{"$ref": "#/definitions/dataset_version"}, {"type": "null"}],
            },
            "id": {"description": "Frame id", "type": ["string", "null"]},
            "is_key_frame": {
                "description": "Is this a key frame (only applicable in frames who'se src is a video)",
                "type": ["boolean", "null"],
            },
            "key_frame": {
                "description": "ID of the key frame that this frame belongs to",
                "type": ["string", "null"],
            },
            "label_rule_counts": {
                "additionalProperties": True,
                "description": "The number of matched roi per lable rule",
                "type": ["object", "null"],
            },
            "labels_size": {
                "description": "Number of labels returned",
                "type": ["integer", "null"],
            },
            "meta": {
                "additionalProperties": True,
                "description": (
                    "Additional metadata dictionary for the frame. Please note that using this field effectively"
                    " defines a schema (dictionary structure and types used as values) - frames within the same dataset"
                    " cannot use conflicting schemas for this field (see documentation for more details)."
                ),
                "type": ["object", "null"],
            },
            "meta_blob": {
                "additionalProperties": True,
                "description": (
                    "Non searchable metadata dictionary for the frame. The fields in this object cannot be searched by"
                    " and are not added to the frame schema"
                ),
                "type": ["object", "null"],
            },
            "new_ver": {
                "description": "Newer version of this frame, if asked to merge",
                "oneOf": [{"$ref": "#/definitions/frame"}, {"type": "null"}],
            },
            "num_frames": {
                "description": "Number of frames represented by this snippet",
                "type": ["integer", "null"],
            },
            "rois": {
                "description": "Frame regions of interest",
                "items": {"$ref": "#/definitions/roi"},
                "type": ["array", "null"],
            },
            "rule_name": {
                "description": "Name of the filtering rule according to which this frame was provided (if applicable)",
                "type": ["string", "null"],
            },
            "saved": {
                "description": "Last time frame was saved (timestamp)",
                "type": ["integer", "null"],
            },
            "saved_in_version": {
                "description": "Last version this frame was saved in (version ID)",
                "type": ["string", "null"],
            },
            "sources": {
                "description": "Sources of this frame",
                "items": {"$ref": "#/definitions/source"},
                "type": ["array", "null"],
            },
            "timestamp": {
                "description": (
                    "Frame's offset in milliseconds, used primarily for video content. Used for the default frames"
                    " sorting as the secondary key (with the primary key being 'context_id'). For images, this value"
                    " should typically be 0. If not set, value is filled from the timestamp of the first source. We"
                    " recommend using this field only in cases concerning the default sorting behavior."
                ),
                "type": ["integer", "null"],
            },
            "updated": {
                "description": "Last time frame was saved (timestamp)",
                "type": ["integer", "null"],
            },
            "updated_in_version": {
                "description": "Last version this frame was updated in (version ID)",
                "type": ["string", "null"],
            },
            "video_gop": {
                "description": "Video encoding GOP value for the source of this frame. Only valid for video frames",
                "type": ["number", "null"],
            },
        },
        "type": "object",
    }

    def __init__(
        self,
        id=None,
        augmentation=None,
        timestamp=None,
        dataset=None,
        saved=None,
        saved_in_version=None,
        updated=None,
        updated_in_version=None,
        rois=None,
        labels_size=None,
        rule_name=None,
        video_gop=None,
        is_key_frame=None,
        key_frame=None,
        meta=None,
        blob=None,
        meta_blob=None,
        new_ver=None,
        label_rule_counts=None,
        sources=None,
        context_id=None,
        num_frames=None,
        **kwargs
    ):
        super(Snippet, self).__init__(**kwargs)
        self.id = id
        self.augmentation = augmentation
        self.timestamp = timestamp
        self.dataset = dataset
        self.saved = saved
        self.saved_in_version = saved_in_version
        self.updated = updated
        self.updated_in_version = updated_in_version
        self.rois = rois
        self.labels_size = labels_size
        self.rule_name = rule_name
        self.video_gop = video_gop
        self.is_key_frame = is_key_frame
        self.key_frame = key_frame
        self.meta = meta
        self.blob = blob
        self.meta_blob = meta_blob
        self.new_ver = new_ver
        self.label_rule_counts = label_rule_counts
        self.sources = sources
        self.context_id = context_id
        self.num_frames = num_frames

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

    @schema_property("augmentation")
    def augmentation(self):
        return self._property_augmentation

    @augmentation.setter
    def augmentation(self, value):
        if value is None:
            self._property_augmentation = None
            return

        self.assert_isinstance(value, "augmentation", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [
                Augmentation.from_dict(v) if isinstance(v, dict) else v for v in value
            ]
        else:
            self.assert_isinstance(value, "augmentation", Augmentation, is_array=True)
        self._property_augmentation = value

    @schema_property("timestamp")
    def timestamp(self):
        return self._property_timestamp

    @timestamp.setter
    def timestamp(self, value):
        if value is None:
            self._property_timestamp = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "timestamp", six.integer_types)
        self._property_timestamp = value

    @schema_property("dataset")
    def dataset(self):
        return self._property_dataset

    @dataset.setter
    def dataset(self, value):
        if value is None:
            self._property_dataset = None
            return
        if isinstance(value, dict):
            value = DatasetVersion.from_dict(value)
        else:
            self.assert_isinstance(value, "dataset", DatasetVersion)
        self._property_dataset = value

    @schema_property("saved")
    def saved(self):
        return self._property_saved

    @saved.setter
    def saved(self, value):
        if value is None:
            self._property_saved = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "saved", six.integer_types)
        self._property_saved = value

    @schema_property("saved_in_version")
    def saved_in_version(self):
        return self._property_saved_in_version

    @saved_in_version.setter
    def saved_in_version(self, value):
        if value is None:
            self._property_saved_in_version = None
            return

        self.assert_isinstance(value, "saved_in_version", six.string_types)
        self._property_saved_in_version = value

    @schema_property("updated")
    def updated(self):
        return self._property_updated

    @updated.setter
    def updated(self, value):
        if value is None:
            self._property_updated = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "updated", six.integer_types)
        self._property_updated = value

    @schema_property("updated_in_version")
    def updated_in_version(self):
        return self._property_updated_in_version

    @updated_in_version.setter
    def updated_in_version(self, value):
        if value is None:
            self._property_updated_in_version = None
            return

        self.assert_isinstance(value, "updated_in_version", six.string_types)
        self._property_updated_in_version = value

    @schema_property("rois")
    def rois(self):
        return self._property_rois

    @rois.setter
    def rois(self, value):
        if value is None:
            self._property_rois = None
            return

        self.assert_isinstance(value, "rois", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [Roi.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "rois", Roi, is_array=True)
        self._property_rois = value

    @schema_property("labels_size")
    def labels_size(self):
        return self._property_labels_size

    @labels_size.setter
    def labels_size(self, value):
        if value is None:
            self._property_labels_size = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "labels_size", six.integer_types)
        self._property_labels_size = value

    @schema_property("rule_name")
    def rule_name(self):
        return self._property_rule_name

    @rule_name.setter
    def rule_name(self, value):
        if value is None:
            self._property_rule_name = None
            return

        self.assert_isinstance(value, "rule_name", six.string_types)
        self._property_rule_name = value

    @schema_property("video_gop")
    def video_gop(self):
        return self._property_video_gop

    @video_gop.setter
    def video_gop(self, value):
        if value is None:
            self._property_video_gop = None
            return

        self.assert_isinstance(value, "video_gop", six.integer_types + (float,))
        self._property_video_gop = value

    @schema_property("is_key_frame")
    def is_key_frame(self):
        return self._property_is_key_frame

    @is_key_frame.setter
    def is_key_frame(self, value):
        if value is None:
            self._property_is_key_frame = None
            return

        self.assert_isinstance(value, "is_key_frame", (bool,))
        self._property_is_key_frame = value

    @schema_property("key_frame")
    def key_frame(self):
        return self._property_key_frame

    @key_frame.setter
    def key_frame(self, value):
        if value is None:
            self._property_key_frame = None
            return

        self.assert_isinstance(value, "key_frame", six.string_types)
        self._property_key_frame = value

    @schema_property("meta")
    def meta(self):
        return self._property_meta

    @meta.setter
    def meta(self, value):
        if value is None:
            self._property_meta = None
            return

        self.assert_isinstance(value, "meta", (dict,))
        self._property_meta = value

    @schema_property("blob")
    def blob(self):
        return self._property_blob

    @blob.setter
    def blob(self, value):
        if value is None:
            self._property_blob = None
            return

        self.assert_isinstance(value, "blob", six.string_types)
        self._property_blob = value

    @schema_property("meta_blob")
    def meta_blob(self):
        return self._property_meta_blob

    @meta_blob.setter
    def meta_blob(self, value):
        if value is None:
            self._property_meta_blob = None
            return

        self.assert_isinstance(value, "meta_blob", (dict,))
        self._property_meta_blob = value

    @schema_property("new_ver")
    def new_ver(self):
        return self._property_new_ver

    @new_ver.setter
    def new_ver(self, value):
        if value is None:
            self._property_new_ver = None
            return
        if isinstance(value, dict):
            value = Frame.from_dict(value)
        else:
            self.assert_isinstance(value, "new_ver", Frame)
        self._property_new_ver = value

    @schema_property("label_rule_counts")
    def label_rule_counts(self):
        return self._property_label_rule_counts

    @label_rule_counts.setter
    def label_rule_counts(self, value):
        if value is None:
            self._property_label_rule_counts = None
            return

        self.assert_isinstance(value, "label_rule_counts", (dict,))
        self._property_label_rule_counts = value

    @schema_property("sources")
    def sources(self):
        return self._property_sources

    @sources.setter
    def sources(self, value):
        if value is None:
            self._property_sources = None
            return

        self.assert_isinstance(value, "sources", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [Source.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "sources", Source, is_array=True)
        self._property_sources = value

    @schema_property("context_id")
    def context_id(self):
        return self._property_context_id

    @context_id.setter
    def context_id(self, value):
        if value is None:
            self._property_context_id = None
            return

        self.assert_isinstance(value, "context_id", six.string_types)
        self._property_context_id = value

    @schema_property("num_frames")
    def num_frames(self):
        return self._property_num_frames

    @num_frames.setter
    def num_frames(self, value):
        if value is None:
            self._property_num_frames = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "num_frames", six.integer_types)
        self._property_num_frames = value


class RuleCount(NonStrictDataModel):
    """
    :param rule_index: Rule index
    :type rule_index: int
    :param name: Rule name
    :type name: str
    :param count: Number of frames matching this rule
    :type count: int
    :param accurate: True if the provided count is accurate. If False, 'reason'
        will contain the reason why.
    :type accurate: bool
    :param reason: Reason for the count being inaccurate if 'accurate' is True,
        empty otherwise.
    :type reason: str
    """

    _schema = {
        "properties": {
            "accurate": {
                "description": (
                    "True if the provided count is accurate. If False, 'reason' will contain the reason why."
                ),
                "type": ["boolean", "null"],
            },
            "count": {
                "description": "Number of frames matching this rule",
                "type": ["integer", "null"],
            },
            "name": {"description": "Rule name", "type": ["string", "null"]},
            "reason": {
                "description": "Reason for the count being inaccurate if 'accurate' is True, empty otherwise.",
                "type": ["string", "null"],
            },
            "rule_index": {"description": "Rule index", "type": ["integer", "null"]},
        },
        "type": "object",
    }

    def __init__(
        self,
        rule_index=None,
        name=None,
        count=None,
        accurate=None,
        reason=None,
        **kwargs
    ):
        super(RuleCount, self).__init__(**kwargs)
        self.rule_index = rule_index
        self.name = name
        self.count = count
        self.accurate = accurate
        self.reason = reason

    @schema_property("rule_index")
    def rule_index(self):
        return self._property_rule_index

    @rule_index.setter
    def rule_index(self, value):
        if value is None:
            self._property_rule_index = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "rule_index", six.integer_types)
        self._property_rule_index = value

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

    @schema_property("count")
    def count(self):
        return self._property_count

    @count.setter
    def count(self, value):
        if value is None:
            self._property_count = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "count", six.integer_types)
        self._property_count = value

    @schema_property("accurate")
    def accurate(self):
        return self._property_accurate

    @accurate.setter
    def accurate(self, value):
        if value is None:
            self._property_accurate = None
            return

        self.assert_isinstance(value, "accurate", (bool,))
        self._property_accurate = value

    @schema_property("reason")
    def reason(self):
        return self._property_reason

    @reason.setter
    def reason(self, value):
        if value is None:
            self._property_reason = None
            return

        self.assert_isinstance(value, "reason", six.string_types)
        self._property_reason = value


class ClearGetNextStateRequest(Request):
    """
    Clears the scroll state received from the get_next family of functions and releases all of its resources

    :param scroll_id: Scroll session id
    :type scroll_id: str
    """

    _service = "frames"
    _action = "clear_get_next_state"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {
            "scroll_id": {"description": "Scroll session id", "type": "string"}
        },
        "required": ["scroll_id"],
        "type": "object",
    }

    def __init__(self, scroll_id, **kwargs):
        super(ClearGetNextStateRequest, self).__init__(**kwargs)
        self.scroll_id = scroll_id

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


class ClearGetNextStateResponse(Response):
    """
    Response of frames.clear_get_next_state endpoint.

    """

    _service = "frames"
    _action = "clear_get_next_state"
    _version = "2.23"

    _schema = {"definitions": {}, "type": "object"}


class DownloadForDataviewRequest(Request):
    """
    Get an attachment containing the frames returned by the dataview specified in `prepare_download_for_dataview`

    :param prepare_id: Call ID returned by a call to prepare_download_for_dataview
    :type prepare_id: str
    """

    _service = "frames"
    _action = "download_for_dataview"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {
            "prepare_id": {
                "description": "Call ID returned by a call to prepare_download_for_dataview",
                "type": "string",
            }
        },
        "required": ["prepare_id"],
        "type": "object",
    }

    def __init__(self, prepare_id, **kwargs):
        super(DownloadForDataviewRequest, self).__init__(**kwargs)
        self.prepare_id = prepare_id

    @schema_property("prepare_id")
    def prepare_id(self):
        return self._property_prepare_id

    @prepare_id.setter
    def prepare_id(self, value):
        if value is None:
            self._property_prepare_id = None
            return

        self.assert_isinstance(value, "prepare_id", six.string_types)
        self._property_prepare_id = value


class DownloadForDataviewResponse(Response):
    """
    Response of frames.download_for_dataview endpoint.

    """

    _service = "frames"
    _action = "download_for_dataview"
    _version = "2.23"

    _schema = {"definitions": {}, "type": "string"}


class GetByIdRequest(Request):
    """
    Get a specific frame for a dataset version using the frame's id. Random Access API.

    :param dataset: Dataset id
    :type dataset: str
    :param version: Version id
    :type version: str
    :param frame: Frame id
    :type frame: str
    :param projection: Used to select which parts of the frame will be returned.
        Each string represents a field or sub-field (using dot-separated notation). In
        order to specify a specific array element, use array index as a field name. To
        specify all array elements, use '*'.
    :type projection: Sequence[str]
    """

    _service = "frames"
    _action = "get_by_id"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {
            "dataset": {"description": "Dataset id", "type": "string"},
            "frame": {"description": "Frame id", "type": "string"},
            "projection": {
                "description": (
                    "Used to select which parts of the frame will be returned. Each string represents a\n              "
                    "      field or sub-field (using dot-separated notation). In order to specify a specific array"
                    " element,\n                    use array index as a field name. To specify all array elements, use"
                    " '*'."
                ),
                "items": {"type": "string"},
                "type": "array",
            },
            "version": {"description": "Version id", "type": "string"},
        },
        "required": ["dataset", "version", "frame"],
        "type": "object",
    }

    def __init__(self, dataset, version, frame, projection=None, **kwargs):
        super(GetByIdRequest, self).__init__(**kwargs)
        self.dataset = dataset
        self.version = version
        self.frame = frame
        self.projection = projection

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

    @schema_property("frame")
    def frame(self):
        return self._property_frame

    @frame.setter
    def frame(self, value):
        if value is None:
            self._property_frame = None
            return

        self.assert_isinstance(value, "frame", six.string_types)
        self._property_frame = value

    @schema_property("projection")
    def projection(self):
        return self._property_projection

    @projection.setter
    def projection(self, value):
        if value is None:
            self._property_projection = None
            return

        self.assert_isinstance(value, "projection", (list, tuple))

        self.assert_isinstance(value, "projection", six.string_types, is_array=True)
        self._property_projection = value


class GetByIdResponse(Response):
    """
    Response of frames.get_by_id endpoint.

    :param frame: Frame data
    :type frame: Frame
    """

    _service = "frames"
    _action = "get_by_id"
    _version = "2.23"

    _schema = {
        "definitions": {
            "augmentation": {
                "properties": {
                    "arguments": {
                        "additionalProperties": True,
                        "description": "Arguments dictionary, passed to custom augmentations.",
                        "type": ["object", "null"],
                    },
                    "cls": {
                        "description": "Augmentation class (see global definitions)",
                        "type": ["string", "null"],
                    },
                    "params": {
                        "description": (
                            "Transform parameters, an array ot 3 randomly generated values. Fixed values are passed in"
                            " case of affine reflect augmentation."
                        ),
                        "items": {"type": "number"},
                        "type": ["array", "null"],
                    },
                    "strength": {
                        "description": "Transform strength. Required for pixel transforms.",
                        "type": ["number", "null"],
                    },
                    "trans_mat": {
                        "description": "Transform matrix (list of lists). Required for affine transforms.",
                        "items": {"items": {"type": "number"}, "type": "array"},
                        "type": ["array", "null"],
                    },
                    "type": {
                        "description": "Augmentation type (see global definitions)",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "dataset_version": {
                "properties": {
                    "id": {"description": "Dataset id", "type": ["string", "null"]},
                    "version": {
                        "description": "Dataset version id",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "frame": {
                "properties": {
                    "augmentation": {
                        "description": "List of augmentations",
                        "items": {"$ref": "#/definitions/augmentation"},
                        "type": ["array", "null"],
                    },
                    "blob": {
                        "description": "Raw data (blob) for the frame",
                        "type": ["string", "null"],
                    },
                    "context_id": {
                        "description": (
                            "Context ID. Used for the default frames sorting. If not set then it is filled from the uri"
                            " of the first source."
                        ),
                        "type": ["string", "null"],
                    },
                    "dataset": {
                        "description": "Frame's dataset version",
                        "oneOf": [
                            {"$ref": "#/definitions/dataset_version"},
                            {"type": "null"},
                        ],
                    },
                    "id": {"description": "Frame id", "type": ["string", "null"]},
                    "is_key_frame": {
                        "description": "Is this a key frame (only applicable in frames who'se src is a video)",
                        "type": ["boolean", "null"],
                    },
                    "key_frame": {
                        "description": "ID of the key frame that this frame belongs to",
                        "type": ["string", "null"],
                    },
                    "label_rule_counts": {
                        "additionalProperties": True,
                        "description": "The number of matched roi per lable rule",
                        "type": ["object", "null"],
                    },
                    "labels_size": {
                        "description": "Number of labels returned",
                        "type": ["integer", "null"],
                    },
                    "meta": {
                        "additionalProperties": True,
                        "description": (
                            "Additional metadata dictionary for the frame. Please note that using this field"
                            " effectively defines a schema (dictionary structure and types used as values) - frames"
                            " within the same dataset cannot use conflicting schemas for this field (see documentation"
                            " for more details)."
                        ),
                        "type": ["object", "null"],
                    },
                    "meta_blob": {
                        "additionalProperties": True,
                        "description": (
                            "Non searchable metadata dictionary for the frame. The fields in this object cannot be"
                            " searched by and are not added to the frame schema"
                        ),
                        "type": ["object", "null"],
                    },
                    "new_ver": {
                        "description": "Newer version of this frame, if asked to merge",
                        "oneOf": [{"$ref": "#/definitions/frame"}, {"type": "null"}],
                    },
                    "rois": {
                        "description": "Frame regions of interest",
                        "items": {"$ref": "#/definitions/roi"},
                        "type": ["array", "null"],
                    },
                    "rule_name": {
                        "description": (
                            "Name of the filtering rule according to which this frame was provided (if applicable)"
                        ),
                        "type": ["string", "null"],
                    },
                    "saved": {
                        "description": "Last time frame was saved (timestamp)",
                        "type": ["integer", "null"],
                    },
                    "saved_in_version": {
                        "description": "Last version this frame was saved in (version ID)",
                        "type": ["string", "null"],
                    },
                    "sources": {
                        "description": "Sources of this frame",
                        "items": {"$ref": "#/definitions/source"},
                        "type": ["array", "null"],
                    },
                    "timestamp": {
                        "description": (
                            "Frame's offset in milliseconds, used primarily for video content. Used for the default"
                            " frames sorting as the secondary key (with the primary key being 'context_id'). For"
                            " images, this value should typically be 0. If not set, value is filled from the timestamp"
                            " of the first source. We recommend using this field only in cases concerning the default"
                            " sorting behavior."
                        ),
                        "type": ["integer", "null"],
                    },
                    "updated": {
                        "description": "Last time frame was saved (timestamp)",
                        "type": ["integer", "null"],
                    },
                    "updated_in_version": {
                        "description": "Last version this frame was updated in (version ID)",
                        "type": ["string", "null"],
                    },
                    "video_gop": {
                        "description": (
                            "Video encoding GOP value for the source of this frame. Only valid for video frames"
                        ),
                        "type": ["number", "null"],
                    },
                },
                "type": "object",
            },
            "mask": {
                "properties": {
                    "content_type": {
                        "description": "Content type (e.g. 'image/jpeg', 'image/png')",
                        "type": ["string", "null"],
                    },
                    "height": {
                        "description": "Height in pixels",
                        "type": ["integer", "null"],
                    },
                    "id": {
                        "description": "unique ID (in this frame)",
                        "type": ["string", "null"],
                    },
                    "timestamp": {
                        "default": 0,
                        "description": (
                            "Timestamp in the source data (for video content. for images, this value should be 0)"
                        ),
                        "type": ["integer", "null"],
                    },
                    "uri": {"description": "Data URI", "type": ["string", "null"]},
                    "width": {
                        "description": "Width in pixels",
                        "type": ["integer", "null"],
                    },
                },
                "type": "object",
            },
            "preview": {
                "properties": {
                    "content_type": {
                        "description": "Content type (e.g. 'image/jpeg', 'image/png')",
                        "type": ["string", "null"],
                    },
                    "height": {
                        "description": "Height in pixels",
                        "type": ["integer", "null"],
                    },
                    "timestamp": {
                        "default": 0,
                        "description": (
                            "Timestamp in the source data (for video content. for images, this value should be 0)"
                        ),
                        "type": ["integer", "null"],
                    },
                    "uri": {"description": "Data URI", "type": ["string", "null"]},
                    "width": {
                        "description": "Width in pixels",
                        "type": ["integer", "null"],
                    },
                },
                "type": "object",
            },
            "roi": {
                "properties": {
                    "area": {
                        "description": "ROI area (not used)",
                        "type": ["integer", "null"],
                    },
                    "confidence": {
                        "description": "ROI confidence",
                        "type": ["number", "null"],
                    },
                    "id": {"description": "ROI id", "type": ["string", "null"]},
                    "label": {
                        "description": "ROI labels",
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                    "label_num": {
                        "description": (
                            "Label number according to the specified labels mapping Used only when ROI is returned as"
                            " part of a task's frame."
                        ),
                        "type": ["integer", "null"],
                    },
                    "mask": {
                        "description": "Mask info for this ROI",
                        "oneOf": [{"$ref": "#/definitions/roi_mask"}, {"type": "null"}],
                    },
                    "meta": {
                        "additionalProperties": True,
                        "description": "Additional metadata dictionary for the roi",
                        "type": ["object", "null"],
                    },
                    "poly": {
                        "description": "ROI polygon (x0, y0, ..., xn, yn)",
                        "items": {"type": "number"},
                        "type": ["array", "null"],
                    },
                    "sources": {
                        "description": "Sources that this ROI belongs to",
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            },
            "roi_mask": {
                "properties": {
                    "id": {"description": "Mask ID", "type": "string"},
                    "value": {
                        "description": "Mask value",
                        "items": {"type": "integer"},
                        "type": "array",
                    },
                },
                "required": ["id", "value"],
                "type": "object",
            },
            "source": {
                "properties": {
                    "content_type": {
                        "description": "Content type (e.g. 'image/jpeg', 'image/png')",
                        "type": ["string", "null"],
                    },
                    "height": {
                        "description": "Height in pixels",
                        "type": ["integer", "null"],
                    },
                    "id": {
                        "description": "unique ID (in this frame)",
                        "type": ["string", "null"],
                    },
                    "masks": {
                        "items": {"$ref": "#/definitions/mask"},
                        "type": ["array", "null"],
                    },
                    "meta": {
                        "additionalProperties": True,
                        "description": "Additional metadata dictionary for the source",
                        "type": ["object", "null"],
                    },
                    "preview": {
                        "oneOf": [{"$ref": "#/definitions/preview"}, {"type": "null"}]
                    },
                    "timestamp": {
                        "default": 0,
                        "description": (
                            "Timestamp in the source data (for video content. for images, this value should be 0)"
                        ),
                        "type": ["integer", "null"],
                    },
                    "uri": {"description": "Data URI", "type": ["string", "null"]},
                    "width": {
                        "description": "Width in pixels",
                        "type": ["integer", "null"],
                    },
                },
                "type": "object",
            },
        },
        "properties": {
            "frame": {
                "description": "Frame data",
                "oneOf": [{"$ref": "#/definitions/frame"}, {"type": "null"}],
            }
        },
        "type": "object",
    }

    def __init__(self, frame=None, **kwargs):
        super(GetByIdResponse, self).__init__(**kwargs)
        self.frame = frame

    @schema_property("frame")
    def frame(self):
        return self._property_frame

    @frame.setter
    def frame(self, value):
        if value is None:
            self._property_frame = None
            return
        if isinstance(value, dict):
            value = Frame.from_dict(value)
        else:
            self.assert_isinstance(value, "frame", Frame)
        self._property_frame = value


class GetByIdsRequest(Request):
    """
    Get specific frames for a dataset version using the frame ids. Random Access API.

    :param dataset: Dataset ID
    :type dataset: str
    :param version: Version ID
    :type version: str
    :param frame_ids: Frame IDs
    :type frame_ids: Sequence[str]
    :param projection: Used to select which parts of the frame will be returned.
        Each string represents a field or sub-field (using dot-separated notation). In
        order to specify a specific array element, use array index as a field name. To
        specify all array elements, use '*'.
    :type projection: Sequence[str]
    """

    _service = "frames"
    _action = "get_by_ids"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {
            "dataset": {"description": "Dataset ID", "type": "string"},
            "frame_ids": {
                "description": "Frame IDs",
                "items": {"type": "string"},
                "type": "array",
            },
            "projection": {
                "description": (
                    "Used to select which parts of the frame will be returned. Each string represents a\n              "
                    "      field or sub-field (using dot-separated notation). In order to specify a specific array"
                    " element,\n                    use array index as a field name. To specify all array elements, use"
                    " '*'."
                ),
                "items": {"type": "string"},
                "type": "array",
            },
            "version": {"description": "Version ID", "type": "string"},
        },
        "required": ["dataset", "version", "frame_ids"],
        "type": "object",
    }

    def __init__(self, dataset, version, frame_ids, projection=None, **kwargs):
        super(GetByIdsRequest, self).__init__(**kwargs)
        self.dataset = dataset
        self.version = version
        self.frame_ids = frame_ids
        self.projection = projection

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

    @schema_property("frame_ids")
    def frame_ids(self):
        return self._property_frame_ids

    @frame_ids.setter
    def frame_ids(self, value):
        if value is None:
            self._property_frame_ids = None
            return

        self.assert_isinstance(value, "frame_ids", (list, tuple))

        self.assert_isinstance(value, "frame_ids", six.string_types, is_array=True)
        self._property_frame_ids = value

    @schema_property("projection")
    def projection(self):
        return self._property_projection

    @projection.setter
    def projection(self, value):
        if value is None:
            self._property_projection = None
            return

        self.assert_isinstance(value, "projection", (list, tuple))

        self.assert_isinstance(value, "projection", six.string_types, is_array=True)
        self._property_projection = value


class GetByIdsResponse(Response):
    """
    Response of frames.get_by_ids endpoint.

    :param frames: Frames data
    :type frames: Sequence[Frame]
    """

    _service = "frames"
    _action = "get_by_ids"
    _version = "2.23"

    _schema = {
        "definitions": {
            "augmentation": {
                "properties": {
                    "arguments": {
                        "additionalProperties": True,
                        "description": "Arguments dictionary, passed to custom augmentations.",
                        "type": ["object", "null"],
                    },
                    "cls": {
                        "description": "Augmentation class (see global definitions)",
                        "type": ["string", "null"],
                    },
                    "params": {
                        "description": (
                            "Transform parameters, an array ot 3 randomly generated values. Fixed values are passed in"
                            " case of affine reflect augmentation."
                        ),
                        "items": {"type": "number"},
                        "type": ["array", "null"],
                    },
                    "strength": {
                        "description": "Transform strength. Required for pixel transforms.",
                        "type": ["number", "null"],
                    },
                    "trans_mat": {
                        "description": "Transform matrix (list of lists). Required for affine transforms.",
                        "items": {"items": {"type": "number"}, "type": "array"},
                        "type": ["array", "null"],
                    },
                    "type": {
                        "description": "Augmentation type (see global definitions)",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "dataset_version": {
                "properties": {
                    "id": {"description": "Dataset id", "type": ["string", "null"]},
                    "version": {
                        "description": "Dataset version id",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "frame": {
                "properties": {
                    "augmentation": {
                        "description": "List of augmentations",
                        "items": {"$ref": "#/definitions/augmentation"},
                        "type": ["array", "null"],
                    },
                    "blob": {
                        "description": "Raw data (blob) for the frame",
                        "type": ["string", "null"],
                    },
                    "context_id": {
                        "description": (
                            "Context ID. Used for the default frames sorting. If not set then it is filled from the uri"
                            " of the first source."
                        ),
                        "type": ["string", "null"],
                    },
                    "dataset": {
                        "description": "Frame's dataset version",
                        "oneOf": [
                            {"$ref": "#/definitions/dataset_version"},
                            {"type": "null"},
                        ],
                    },
                    "id": {"description": "Frame id", "type": ["string", "null"]},
                    "is_key_frame": {
                        "description": "Is this a key frame (only applicable in frames who'se src is a video)",
                        "type": ["boolean", "null"],
                    },
                    "key_frame": {
                        "description": "ID of the key frame that this frame belongs to",
                        "type": ["string", "null"],
                    },
                    "label_rule_counts": {
                        "additionalProperties": True,
                        "description": "The number of matched roi per lable rule",
                        "type": ["object", "null"],
                    },
                    "labels_size": {
                        "description": "Number of labels returned",
                        "type": ["integer", "null"],
                    },
                    "meta": {
                        "additionalProperties": True,
                        "description": (
                            "Additional metadata dictionary for the frame. Please note that using this field"
                            " effectively defines a schema (dictionary structure and types used as values) - frames"
                            " within the same dataset cannot use conflicting schemas for this field (see documentation"
                            " for more details)."
                        ),
                        "type": ["object", "null"],
                    },
                    "meta_blob": {
                        "additionalProperties": True,
                        "description": (
                            "Non searchable metadata dictionary for the frame. The fields in this object cannot be"
                            " searched by and are not added to the frame schema"
                        ),
                        "type": ["object", "null"],
                    },
                    "new_ver": {
                        "description": "Newer version of this frame, if asked to merge",
                        "oneOf": [{"$ref": "#/definitions/frame"}, {"type": "null"}],
                    },
                    "rois": {
                        "description": "Frame regions of interest",
                        "items": {"$ref": "#/definitions/roi"},
                        "type": ["array", "null"],
                    },
                    "rule_name": {
                        "description": (
                            "Name of the filtering rule according to which this frame was provided (if applicable)"
                        ),
                        "type": ["string", "null"],
                    },
                    "saved": {
                        "description": "Last time frame was saved (timestamp)",
                        "type": ["integer", "null"],
                    },
                    "saved_in_version": {
                        "description": "Last version this frame was saved in (version ID)",
                        "type": ["string", "null"],
                    },
                    "sources": {
                        "description": "Sources of this frame",
                        "items": {"$ref": "#/definitions/source"},
                        "type": ["array", "null"],
                    },
                    "timestamp": {
                        "description": (
                            "Frame's offset in milliseconds, used primarily for video content. Used for the default"
                            " frames sorting as the secondary key (with the primary key being 'context_id'). For"
                            " images, this value should typically be 0. If not set, value is filled from the timestamp"
                            " of the first source. We recommend using this field only in cases concerning the default"
                            " sorting behavior."
                        ),
                        "type": ["integer", "null"],
                    },
                    "updated": {
                        "description": "Last time frame was saved (timestamp)",
                        "type": ["integer", "null"],
                    },
                    "updated_in_version": {
                        "description": "Last version this frame was updated in (version ID)",
                        "type": ["string", "null"],
                    },
                    "video_gop": {
                        "description": (
                            "Video encoding GOP value for the source of this frame. Only valid for video frames"
                        ),
                        "type": ["number", "null"],
                    },
                },
                "type": "object",
            },
            "mask": {
                "properties": {
                    "content_type": {
                        "description": "Content type (e.g. 'image/jpeg', 'image/png')",
                        "type": ["string", "null"],
                    },
                    "height": {
                        "description": "Height in pixels",
                        "type": ["integer", "null"],
                    },
                    "id": {
                        "description": "unique ID (in this frame)",
                        "type": ["string", "null"],
                    },
                    "timestamp": {
                        "default": 0,
                        "description": (
                            "Timestamp in the source data (for video content. for images, this value should be 0)"
                        ),
                        "type": ["integer", "null"],
                    },
                    "uri": {"description": "Data URI", "type": ["string", "null"]},
                    "width": {
                        "description": "Width in pixels",
                        "type": ["integer", "null"],
                    },
                },
                "type": "object",
            },
            "preview": {
                "properties": {
                    "content_type": {
                        "description": "Content type (e.g. 'image/jpeg', 'image/png')",
                        "type": ["string", "null"],
                    },
                    "height": {
                        "description": "Height in pixels",
                        "type": ["integer", "null"],
                    },
                    "timestamp": {
                        "default": 0,
                        "description": (
                            "Timestamp in the source data (for video content. for images, this value should be 0)"
                        ),
                        "type": ["integer", "null"],
                    },
                    "uri": {"description": "Data URI", "type": ["string", "null"]},
                    "width": {
                        "description": "Width in pixels",
                        "type": ["integer", "null"],
                    },
                },
                "type": "object",
            },
            "roi": {
                "properties": {
                    "area": {
                        "description": "ROI area (not used)",
                        "type": ["integer", "null"],
                    },
                    "confidence": {
                        "description": "ROI confidence",
                        "type": ["number", "null"],
                    },
                    "id": {"description": "ROI id", "type": ["string", "null"]},
                    "label": {
                        "description": "ROI labels",
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                    "label_num": {
                        "description": (
                            "Label number according to the specified labels mapping Used only when ROI is returned as"
                            " part of a task's frame."
                        ),
                        "type": ["integer", "null"],
                    },
                    "mask": {
                        "description": "Mask info for this ROI",
                        "oneOf": [{"$ref": "#/definitions/roi_mask"}, {"type": "null"}],
                    },
                    "meta": {
                        "additionalProperties": True,
                        "description": "Additional metadata dictionary for the roi",
                        "type": ["object", "null"],
                    },
                    "poly": {
                        "description": "ROI polygon (x0, y0, ..., xn, yn)",
                        "items": {"type": "number"},
                        "type": ["array", "null"],
                    },
                    "sources": {
                        "description": "Sources that this ROI belongs to",
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            },
            "roi_mask": {
                "properties": {
                    "id": {"description": "Mask ID", "type": "string"},
                    "value": {
                        "description": "Mask value",
                        "items": {"type": "integer"},
                        "type": "array",
                    },
                },
                "required": ["id", "value"],
                "type": "object",
            },
            "source": {
                "properties": {
                    "content_type": {
                        "description": "Content type (e.g. 'image/jpeg', 'image/png')",
                        "type": ["string", "null"],
                    },
                    "height": {
                        "description": "Height in pixels",
                        "type": ["integer", "null"],
                    },
                    "id": {
                        "description": "unique ID (in this frame)",
                        "type": ["string", "null"],
                    },
                    "masks": {
                        "items": {"$ref": "#/definitions/mask"},
                        "type": ["array", "null"],
                    },
                    "meta": {
                        "additionalProperties": True,
                        "description": "Additional metadata dictionary for the source",
                        "type": ["object", "null"],
                    },
                    "preview": {
                        "oneOf": [{"$ref": "#/definitions/preview"}, {"type": "null"}]
                    },
                    "timestamp": {
                        "default": 0,
                        "description": (
                            "Timestamp in the source data (for video content. for images, this value should be 0)"
                        ),
                        "type": ["integer", "null"],
                    },
                    "uri": {"description": "Data URI", "type": ["string", "null"]},
                    "width": {
                        "description": "Width in pixels",
                        "type": ["integer", "null"],
                    },
                },
                "type": "object",
            },
        },
        "properties": {
            "frames": {
                "description": "Frames data",
                "items": {"$ref": "#/definitions/frame"},
                "type": ["array", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, frames=None, **kwargs):
        super(GetByIdsResponse, self).__init__(**kwargs)
        self.frames = frames

    @schema_property("frames")
    def frames(self):
        return self._property_frames

    @frames.setter
    def frames(self, value):
        if value is None:
            self._property_frames = None
            return

        self.assert_isinstance(value, "frames", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [Frame.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "frames", Frame, is_array=True)
        self._property_frames = value


class GetCountRequest(Request):
    """
    Gets the count of frames who's ROIs contain at least one of the given label(s)

    :param dataset: Dataset id
    :type dataset: str
    :param version: Dataset version id
    :type version: str
    :param labels: List of labels. Only frames containing labels from this list
        should be counted. Used only if scroll_id is not provided.
    :type labels: Sequence[str]
    """

    _service = "frames"
    _action = "get_count"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {
            "dataset": {"description": "Dataset id", "type": ["string", "null"]},
            "labels": {
                "description": (
                    "List of labels. Only frames containing labels from this list should be counted. Used only if "
                    "scroll_id is not provided."
                ),
                "items": {"type": "string"},
                "type": "array",
            },
            "version": {"description": "Dataset version id", "type": "string"},
        },
        "required": ["version"],
    }

    def __init__(self, version, dataset=None, labels=None, **kwargs):
        super(GetCountRequest, self).__init__(**kwargs)
        self.dataset = dataset
        self.version = version
        self.labels = labels

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


class GetCountResponse(Response):
    """
    Response of frames.get_count endpoint.

    :param total: Total count of frames for the entire query.
    :type total: int
    """

    _service = "frames"
    _action = "get_count"
    _version = "2.23"

    _schema = {
        "definitions": {},
        "properties": {
            "total": {
                "description": "Total count of frames for the entire query.",
                "type": ["integer", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, total=None, **kwargs):
        super(GetCountResponse, self).__init__(**kwargs)
        self.total = total

    @schema_property("total")
    def total(self):
        return self._property_total

    @total.setter
    def total(self, value):
        if value is None:
            self._property_total = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "total", six.integer_types)
        self._property_total = value


class GetCountForDataviewRequest(Request):
    """
    Gets the count of frames matching the given dataview

    :param dataview: Dataview specification
    :type dataview: Dataview
    """

    _service = "frames"
    _action = "get_count_for_dataview"
    _version = "2.23"
    _schema = {
        "definitions": {
            "dataview": {
                "properties": {
                    "augmentation": {
                        "description": "Augmentation parameters. Only for training and testing tasks.",
                        "oneOf": [
                            {"$ref": "#/definitions/dv_augmentation"},
                            {"type": "null"},
                        ],
                    },
                    "filters": {
                        "description": "List of FilterRule ('OR' relationship)",
                        "items": {"$ref": "#/definitions/filter_rule"},
                        "type": ["array", "null"],
                    },
                    "iteration": {
                        "description": "Iteration parameters. Not applicable for register (import) tasks.",
                        "oneOf": [
                            {"$ref": "#/definitions/iteration"},
                            {"type": "null"},
                        ],
                    },
                    "labels_enumeration": {
                        "additionalProperties": {"type": "integer"},
                        "description": (
                            "Labels enumerations, specifies numbers to be assigned to ROI labels when getting frames"
                        ),
                        "type": ["object", "null"],
                    },
                    "mapping": {
                        "description": "Mapping parameters",
                        "oneOf": [{"$ref": "#/definitions/mapping"}, {"type": "null"}],
                    },
                    "output_rois": {
                        "description": (
                            "'all_in_frame' - all rois for a frame are returned\n\n'only_filtered' - only rois which"
                            " led this frame to be selected\n\n'frame_per_roi' - single roi per frame. Frame can be"
                            " returned multiple times with a different roi each time.\n\nNote: this should be used for"
                            " Training tasks only\n\nNote: frame_per_roi implies that only filtered rois will be"
                            " returned\n                "
                        ),
                        "oneOf": [
                            {"$ref": "#/definitions/output_rois_enum"},
                            {"type": "null"},
                        ],
                    },
                    "versions": {
                        "description": "View dataset versions",
                        "items": {"$ref": "#/definitions/view_entry"},
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            },
            "dv_augmentation": {
                "properties": {
                    "crop_around_rois": {
                        "description": "Crop image data around all frame ROIs",
                        "type": ["boolean", "null"],
                    },
                    "sets": {
                        "description": "List of augmentation sets",
                        "items": {"$ref": "#/definitions/dv_augmentation_set"},
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            },
            "dv_augmentation_set": {
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
            "view_entry": {
                "properties": {
                    "dataset": {
                        "description": "Existing Dataset id",
                        "type": ["string", "null"],
                    },
                    "merge_with": {
                        "description": "Version ID to merge with",
                        "type": ["string", "null"],
                    },
                    "version": {
                        "description": "Version id of a version belonging to the dataset",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
        },
        "properties": {
            "dataview": {
                "$ref": "#/definitions/dataview",
                "description": "Dataview specification",
            }
        },
        "required": ["dataview"],
    }

    def __init__(self, dataview, **kwargs):
        super(GetCountForDataviewRequest, self).__init__(**kwargs)
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


class GetCountForDataviewResponse(Response):
    """
    Response of frames.get_count_for_dataview endpoint.

    :param total: Total count of frames for the entire query.
    :type total: int
    :param rules: Specific information for each rule of this query.
    :type rules: Sequence[RuleCount]
    """

    _service = "frames"
    _action = "get_count_for_dataview"
    _version = "2.23"

    _schema = {
        "definitions": {
            "rule_count": {
                "properties": {
                    "accurate": {
                        "description": (
                            "True if the provided count is accurate. If False, 'reason' will contain the reason why."
                        ),
                        "type": ["boolean", "null"],
                    },
                    "count": {
                        "description": "Number of frames matching this rule",
                        "type": ["integer", "null"],
                    },
                    "name": {"description": "Rule name", "type": ["string", "null"]},
                    "reason": {
                        "description": "Reason for the count being inaccurate if 'accurate' is True, empty otherwise.",
                        "type": ["string", "null"],
                    },
                    "rule_index": {
                        "description": "Rule index",
                        "type": ["integer", "null"],
                    },
                },
                "type": "object",
            }
        },
        "properties": {
            "rules": {
                "description": "Specific information for each rule of this query.",
                "items": {"$ref": "#/definitions/rule_count"},
                "type": ["array", "null"],
            },
            "total": {
                "description": "Total count of frames for the entire query.",
                "type": ["integer", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, total=None, rules=None, **kwargs):
        super(GetCountForDataviewResponse, self).__init__(**kwargs)
        self.total = total
        self.rules = rules

    @schema_property("total")
    def total(self):
        return self._property_total

    @total.setter
    def total(self, value):
        if value is None:
            self._property_total = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "total", six.integer_types)
        self._property_total = value

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
                RuleCount.from_dict(v) if isinstance(v, dict) else v for v in value
            ]
        else:
            self.assert_isinstance(value, "rules", RuleCount, is_array=True)
        self._property_rules = value


class GetCountForDataviewIdRequest(Request):
    """
    Gets the count of frames matching the given dataview

    :param dataview: Dataview ID
    :type dataview: str
    """

    _service = "frames"
    _action = "get_count_for_dataview_id"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {"dataview": {"description": "Dataview ID", "type": "string"}},
        "required": ["dataview"],
    }

    def __init__(self, dataview, **kwargs):
        super(GetCountForDataviewIdRequest, self).__init__(**kwargs)
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


class GetCountForDataviewIdResponse(Response):
    """
    Response of frames.get_count_for_dataview_id endpoint.

    :param total: Total count of frames for the entire query.
    :type total: int
    :param rules: Specific information for each rule of this query.
    :type rules: Sequence[RuleCount]
    """

    _service = "frames"
    _action = "get_count_for_dataview_id"
    _version = "2.23"

    _schema = {
        "definitions": {
            "rule_count": {
                "properties": {
                    "accurate": {
                        "description": (
                            "True if the provided count is accurate. If False, 'reason' will contain the reason why."
                        ),
                        "type": ["boolean", "null"],
                    },
                    "count": {
                        "description": "Number of frames matching this rule",
                        "type": ["integer", "null"],
                    },
                    "name": {"description": "Rule name", "type": ["string", "null"]},
                    "reason": {
                        "description": "Reason for the count being inaccurate if 'accurate' is True, empty otherwise.",
                        "type": ["string", "null"],
                    },
                    "rule_index": {
                        "description": "Rule index",
                        "type": ["integer", "null"],
                    },
                },
                "type": "object",
            }
        },
        "properties": {
            "rules": {
                "description": "Specific information for each rule of this query.",
                "items": {"$ref": "#/definitions/rule_count"},
                "type": ["array", "null"],
            },
            "total": {
                "description": "Total count of frames for the entire query.",
                "type": ["integer", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, total=None, rules=None, **kwargs):
        super(GetCountForDataviewIdResponse, self).__init__(**kwargs)
        self.total = total
        self.rules = rules

    @schema_property("total")
    def total(self):
        return self._property_total

    @total.setter
    def total(self, value):
        if value is None:
            self._property_total = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "total", six.integer_types)
        self._property_total = value

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
                RuleCount.from_dict(v) if isinstance(v, dict) else v for v in value
            ]
        else:
            self.assert_isinstance(value, "rules", RuleCount, is_array=True)
        self._property_rules = value


class GetNextRequest(Request):
    """
    Gets frames for a given dataset version.

    :param dataset: Dataset id
    :type dataset: str
    :param version: Dataset version id
    :type version: str
    :param merge_with: Version ID to merge with
    :type merge_with: str
    :param labels: List of labels. Only frames containing labels from this list
        should be returned. Used only if scroll_id is not provided.
    :type labels: Sequence[str]
    :param batch_size: Max number of images to be returned. Used only if scroll_id
        is not provided.
    :type batch_size: int
    :param scroll_id: Scroll session ID for getting the next batch of images
    :type scroll_id: str
    :param node: Node number. This provides support for multi-node experiments
        running multiple workers executing the same experiment on multiple processes or
        machines
    :type node: int
    :param projection: Used to select which parts of the frame will be returned.
        Each string represents a field or sub-field (using dot-separated notation). In
        order to specify a specific array element, use array index as a field name. To
        specify all array elements, use '*'.
    :type projection: Sequence[str]
    :param remove_none_values: If set to Truethen none values are removed from
        frames (except for metadata)
    :type remove_none_values: bool
    :param clean_subfields: If set to Truethen both frame toplevel fields and
        subfields are cleaned according to the schema. Otherwise only top level fields
    :type clean_subfields: bool
    """

    _service = "frames"
    _action = "get_next"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {
            "batch_size": {
                "default": 100,
                "description": "Max number of images to be returned. Used only if scroll_id is not provided.",
                "type": "integer",
            },
            "clean_subfields": {
                "default": False,
                "description": (
                    "If set to Truethen both frame toplevel fields and subfields are cleaned according to the schema."
                    " Otherwise only top level fields"
                ),
                "type": "boolean",
            },
            "dataset": {"description": "Dataset id", "type": "string"},
            "labels": {
                "description": (
                    "List of labels. Only frames containing labels from this list should be returned. Used only if "
                    "scroll_id is not provided."
                ),
                "items": {"type": "string"},
                "type": "array",
            },
            "merge_with": {"description": "Version ID to merge with", "type": "string"},
            "node": {
                "description": (
                    "Node number. This provides support for multi-node experiments running multiple workers executing"
                    " the same experiment on multiple processes or machines"
                ),
                "type": "integer",
            },
            "projection": {
                "description": (
                    "Used to select which parts of the frame will be returned. Each string represents a\n              "
                    "      field or sub-field (using dot-separated notation). In order to specify a specific array"
                    " element,\n                    use array index as a field name. To specify all array elements, use"
                    " '*'."
                ),
                "items": {"type": "string"},
                "type": "array",
            },
            "remove_none_values": {
                "default": False,
                "description": "If set to Truethen none values are removed from frames (except for metadata)",
                "type": "boolean",
            },
            "scroll_id": {
                "description": "Scroll session ID for getting the next batch of images",
                "type": "string",
            },
            "version": {"description": "Dataset version id", "type": "string"},
        },
        "required": ["dataset", "version"],
        "type": "object",
    }

    def __init__(
        self,
        dataset,
        version,
        merge_with=None,
        labels=None,
        batch_size=100,
        scroll_id=None,
        node=None,
        projection=None,
        remove_none_values=False,
        clean_subfields=False,
        **kwargs
    ):
        super(GetNextRequest, self).__init__(**kwargs)
        self.dataset = dataset
        self.version = version
        self.merge_with = merge_with
        self.labels = labels
        self.batch_size = batch_size
        self.scroll_id = scroll_id
        self.node = node
        self.projection = projection
        self.remove_none_values = remove_none_values
        self.clean_subfields = clean_subfields

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

    @schema_property("batch_size")
    def batch_size(self):
        return self._property_batch_size

    @batch_size.setter
    def batch_size(self, value):
        if value is None:
            self._property_batch_size = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "batch_size", six.integer_types)
        self._property_batch_size = value

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

    @schema_property("node")
    def node(self):
        return self._property_node

    @node.setter
    def node(self, value):
        if value is None:
            self._property_node = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "node", six.integer_types)
        self._property_node = value

    @schema_property("projection")
    def projection(self):
        return self._property_projection

    @projection.setter
    def projection(self, value):
        if value is None:
            self._property_projection = None
            return

        self.assert_isinstance(value, "projection", (list, tuple))

        self.assert_isinstance(value, "projection", six.string_types, is_array=True)
        self._property_projection = value

    @schema_property("remove_none_values")
    def remove_none_values(self):
        return self._property_remove_none_values

    @remove_none_values.setter
    def remove_none_values(self, value):
        if value is None:
            self._property_remove_none_values = None
            return

        self.assert_isinstance(value, "remove_none_values", (bool,))
        self._property_remove_none_values = value

    @schema_property("clean_subfields")
    def clean_subfields(self):
        return self._property_clean_subfields

    @clean_subfields.setter
    def clean_subfields(self, value):
        if value is None:
            self._property_clean_subfields = None
            return

        self.assert_isinstance(value, "clean_subfields", (bool,))
        self._property_clean_subfields = value


class GetNextResponse(Response):
    """
    Response of frames.get_next endpoint.

    :param frames: Frames list
    :type frames: Sequence[Frame]
    :param frames_returned: Number of frames returned
    :type frames_returned: int
    :param scroll_state: JSON object representing the scroll state
    :type scroll_state: dict
    :param scroll_id: Scroll session id to be provided in order to get the next
        batch of images
    :type scroll_id: str
    :param roi_stats: Json object containing the count per labels in frames, e.g. {
        'background': 312, 'boat': 2, 'bus': 4, 'car': 2, }
    :type roi_stats: dict
    :param eof: When 'frames' is empty, represents whether there are no more frames
        left. If "false", client can retry the operation.
    :type eof: bool
    """

    _service = "frames"
    _action = "get_next"
    _version = "2.23"

    _schema = {
        "definitions": {
            "augmentation": {
                "properties": {
                    "arguments": {
                        "additionalProperties": True,
                        "description": "Arguments dictionary, passed to custom augmentations.",
                        "type": ["object", "null"],
                    },
                    "cls": {
                        "description": "Augmentation class (see global definitions)",
                        "type": ["string", "null"],
                    },
                    "params": {
                        "description": (
                            "Transform parameters, an array ot 3 randomly generated values. Fixed values are passed in"
                            " case of affine reflect augmentation."
                        ),
                        "items": {"type": "number"},
                        "type": ["array", "null"],
                    },
                    "strength": {
                        "description": "Transform strength. Required for pixel transforms.",
                        "type": ["number", "null"],
                    },
                    "trans_mat": {
                        "description": "Transform matrix (list of lists). Required for affine transforms.",
                        "items": {"items": {"type": "number"}, "type": "array"},
                        "type": ["array", "null"],
                    },
                    "type": {
                        "description": "Augmentation type (see global definitions)",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "dataset_version": {
                "properties": {
                    "id": {"description": "Dataset id", "type": ["string", "null"]},
                    "version": {
                        "description": "Dataset version id",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "frame": {
                "properties": {
                    "augmentation": {
                        "description": "List of augmentations",
                        "items": {"$ref": "#/definitions/augmentation"},
                        "type": ["array", "null"],
                    },
                    "blob": {
                        "description": "Raw data (blob) for the frame",
                        "type": ["string", "null"],
                    },
                    "context_id": {
                        "description": (
                            "Context ID. Used for the default frames sorting. If not set then it is filled from the uri"
                            " of the first source."
                        ),
                        "type": ["string", "null"],
                    },
                    "dataset": {
                        "description": "Frame's dataset version",
                        "oneOf": [
                            {"$ref": "#/definitions/dataset_version"},
                            {"type": "null"},
                        ],
                    },
                    "id": {"description": "Frame id", "type": ["string", "null"]},
                    "is_key_frame": {
                        "description": "Is this a key frame (only applicable in frames who'se src is a video)",
                        "type": ["boolean", "null"],
                    },
                    "key_frame": {
                        "description": "ID of the key frame that this frame belongs to",
                        "type": ["string", "null"],
                    },
                    "label_rule_counts": {
                        "additionalProperties": True,
                        "description": "The number of matched roi per lable rule",
                        "type": ["object", "null"],
                    },
                    "labels_size": {
                        "description": "Number of labels returned",
                        "type": ["integer", "null"],
                    },
                    "meta": {
                        "additionalProperties": True,
                        "description": (
                            "Additional metadata dictionary for the frame. Please note that using this field"
                            " effectively defines a schema (dictionary structure and types used as values) - frames"
                            " within the same dataset cannot use conflicting schemas for this field (see documentation"
                            " for more details)."
                        ),
                        "type": ["object", "null"],
                    },
                    "meta_blob": {
                        "additionalProperties": True,
                        "description": (
                            "Non searchable metadata dictionary for the frame. The fields in this object cannot be"
                            " searched by and are not added to the frame schema"
                        ),
                        "type": ["object", "null"],
                    },
                    "new_ver": {
                        "description": "Newer version of this frame, if asked to merge",
                        "oneOf": [{"$ref": "#/definitions/frame"}, {"type": "null"}],
                    },
                    "rois": {
                        "description": "Frame regions of interest",
                        "items": {"$ref": "#/definitions/roi"},
                        "type": ["array", "null"],
                    },
                    "rule_name": {
                        "description": (
                            "Name of the filtering rule according to which this frame was provided (if applicable)"
                        ),
                        "type": ["string", "null"],
                    },
                    "saved": {
                        "description": "Last time frame was saved (timestamp)",
                        "type": ["integer", "null"],
                    },
                    "saved_in_version": {
                        "description": "Last version this frame was saved in (version ID)",
                        "type": ["string", "null"],
                    },
                    "sources": {
                        "description": "Sources of this frame",
                        "items": {"$ref": "#/definitions/source"},
                        "type": ["array", "null"],
                    },
                    "timestamp": {
                        "description": (
                            "Frame's offset in milliseconds, used primarily for video content. Used for the default"
                            " frames sorting as the secondary key (with the primary key being 'context_id'). For"
                            " images, this value should typically be 0. If not set, value is filled from the timestamp"
                            " of the first source. We recommend using this field only in cases concerning the default"
                            " sorting behavior."
                        ),
                        "type": ["integer", "null"],
                    },
                    "updated": {
                        "description": "Last time frame was saved (timestamp)",
                        "type": ["integer", "null"],
                    },
                    "updated_in_version": {
                        "description": "Last version this frame was updated in (version ID)",
                        "type": ["string", "null"],
                    },
                    "video_gop": {
                        "description": (
                            "Video encoding GOP value for the source of this frame. Only valid for video frames"
                        ),
                        "type": ["number", "null"],
                    },
                },
                "type": "object",
            },
            "mask": {
                "properties": {
                    "content_type": {
                        "description": "Content type (e.g. 'image/jpeg', 'image/png')",
                        "type": ["string", "null"],
                    },
                    "height": {
                        "description": "Height in pixels",
                        "type": ["integer", "null"],
                    },
                    "id": {
                        "description": "unique ID (in this frame)",
                        "type": ["string", "null"],
                    },
                    "timestamp": {
                        "default": 0,
                        "description": (
                            "Timestamp in the source data (for video content. for images, this value should be 0)"
                        ),
                        "type": ["integer", "null"],
                    },
                    "uri": {"description": "Data URI", "type": ["string", "null"]},
                    "width": {
                        "description": "Width in pixels",
                        "type": ["integer", "null"],
                    },
                },
                "type": "object",
            },
            "preview": {
                "properties": {
                    "content_type": {
                        "description": "Content type (e.g. 'image/jpeg', 'image/png')",
                        "type": ["string", "null"],
                    },
                    "height": {
                        "description": "Height in pixels",
                        "type": ["integer", "null"],
                    },
                    "timestamp": {
                        "default": 0,
                        "description": (
                            "Timestamp in the source data (for video content. for images, this value should be 0)"
                        ),
                        "type": ["integer", "null"],
                    },
                    "uri": {"description": "Data URI", "type": ["string", "null"]},
                    "width": {
                        "description": "Width in pixels",
                        "type": ["integer", "null"],
                    },
                },
                "type": "object",
            },
            "roi": {
                "properties": {
                    "area": {
                        "description": "ROI area (not used)",
                        "type": ["integer", "null"],
                    },
                    "confidence": {
                        "description": "ROI confidence",
                        "type": ["number", "null"],
                    },
                    "id": {"description": "ROI id", "type": ["string", "null"]},
                    "label": {
                        "description": "ROI labels",
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                    "label_num": {
                        "description": (
                            "Label number according to the specified labels mapping Used only when ROI is returned as"
                            " part of a task's frame."
                        ),
                        "type": ["integer", "null"],
                    },
                    "mask": {
                        "description": "Mask info for this ROI",
                        "oneOf": [{"$ref": "#/definitions/roi_mask"}, {"type": "null"}],
                    },
                    "meta": {
                        "additionalProperties": True,
                        "description": "Additional metadata dictionary for the roi",
                        "type": ["object", "null"],
                    },
                    "poly": {
                        "description": "ROI polygon (x0, y0, ..., xn, yn)",
                        "items": {"type": "number"},
                        "type": ["array", "null"],
                    },
                    "sources": {
                        "description": "Sources that this ROI belongs to",
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            },
            "roi_mask": {
                "properties": {
                    "id": {"description": "Mask ID", "type": "string"},
                    "value": {
                        "description": "Mask value",
                        "items": {"type": "integer"},
                        "type": "array",
                    },
                },
                "required": ["id", "value"],
                "type": "object",
            },
            "source": {
                "properties": {
                    "content_type": {
                        "description": "Content type (e.g. 'image/jpeg', 'image/png')",
                        "type": ["string", "null"],
                    },
                    "height": {
                        "description": "Height in pixels",
                        "type": ["integer", "null"],
                    },
                    "id": {
                        "description": "unique ID (in this frame)",
                        "type": ["string", "null"],
                    },
                    "masks": {
                        "items": {"$ref": "#/definitions/mask"},
                        "type": ["array", "null"],
                    },
                    "meta": {
                        "additionalProperties": True,
                        "description": "Additional metadata dictionary for the source",
                        "type": ["object", "null"],
                    },
                    "preview": {
                        "oneOf": [{"$ref": "#/definitions/preview"}, {"type": "null"}]
                    },
                    "timestamp": {
                        "default": 0,
                        "description": (
                            "Timestamp in the source data (for video content. for images, this value should be 0)"
                        ),
                        "type": ["integer", "null"],
                    },
                    "uri": {"description": "Data URI", "type": ["string", "null"]},
                    "width": {
                        "description": "Width in pixels",
                        "type": ["integer", "null"],
                    },
                },
                "type": "object",
            },
        },
        "properties": {
            "eof": {
                "description": (
                    "When 'frames' is empty, represents whether there are no more frames left. "
                    'If "false",\n'
                    "                client can retry the "
                    "operation."
                ),
                "type": ["boolean", "null"],
            },
            "frames": {
                "description": "Frames list",
                "items": {"$ref": "#/definitions/frame"},
                "type": ["array", "null"],
            },
            "frames_returned": {
                "description": "Number of frames returned",
                "type": ["integer", "null"],
            },
            "roi_stats": {
                "additionalProperties": {"type": "integer"},
                "description": (
                    "Json object containing the count per labels in frames, e.g.\n                {\n                  "
                    "  'background': 312,\n                    'boat': 2,\n                    'bus': 4,\n             "
                    "       'car': 2,\n                }"
                ),
                "type": ["object", "null"],
            },
            "scroll_id": {
                "description": "Scroll session id to be provided in order to get the next batch of images",
                "type": ["string", "null"],
            },
            "scroll_state": {
                "additionalProperties": True,
                "description": "JSON object representing the scroll state",
                "type": ["object", "null"],
            },
        },
        "type": "object",
    }

    def __init__(
        self,
        frames=None,
        frames_returned=None,
        scroll_state=None,
        scroll_id=None,
        roi_stats=None,
        eof=None,
        **kwargs
    ):
        super(GetNextResponse, self).__init__(**kwargs)
        self.frames = frames
        self.frames_returned = frames_returned
        self.scroll_state = scroll_state
        self.scroll_id = scroll_id
        self.roi_stats = roi_stats
        self.eof = eof

    @schema_property("frames")
    def frames(self):
        return self._property_frames

    @frames.setter
    def frames(self, value):
        if value is None:
            self._property_frames = None
            return

        self.assert_isinstance(value, "frames", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [Frame.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "frames", Frame, is_array=True)
        self._property_frames = value

    @schema_property("frames_returned")
    def frames_returned(self):
        return self._property_frames_returned

    @frames_returned.setter
    def frames_returned(self, value):
        if value is None:
            self._property_frames_returned = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "frames_returned", six.integer_types)
        self._property_frames_returned = value

    @schema_property("scroll_state")
    def scroll_state(self):
        return self._property_scroll_state

    @scroll_state.setter
    def scroll_state(self, value):
        if value is None:
            self._property_scroll_state = None
            return

        self.assert_isinstance(value, "scroll_state", (dict,))
        self._property_scroll_state = value

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

    @schema_property("roi_stats")
    def roi_stats(self):
        return self._property_roi_stats

    @roi_stats.setter
    def roi_stats(self, value):
        if value is None:
            self._property_roi_stats = None
            return

        self.assert_isinstance(value, "roi_stats", (dict,))
        self._property_roi_stats = value

    @schema_property("eof")
    def eof(self):
        return self._property_eof

    @eof.setter
    def eof(self, value):
        if value is None:
            self._property_eof = None
            return

        self.assert_isinstance(value, "eof", (bool,))
        self._property_eof = value


class GetNextForDataviewRequest(Request):
    """
    Gets frames for a given view specification

    :param dataview: Dataview specification
    :type dataview: Dataview
    :param scroll_id: Scroll session id for getting the next batch of images
    :type scroll_id: str
    :param batch_size: Max number of images to be returned. Used only if scroll_id
        is not provided.
    :type batch_size: int
    :param reset_scroll:
    :type reset_scroll: bool
    :param force_scroll_id:
    :type force_scroll_id: bool
    :param flow_control: Contol if frames retreival always navigate in one
        direction (the default) or can navigate forwards and backwards
    :type flow_control: FlowControl
    :param random_seed: Optional random seed used for frame selection. If not
        provided, one will be generated.
    :type random_seed: int
    :param node: Node number. This provides support for multi-node experiments
        running multiple workers executing the same experiment on multiple processes or
        machines
    :type node: int
    :param projection: Used to select which parts of the frame will be returned.
        Each string represents a field or sub-field (using dot-separated notation). In
        order to specify a specific array element, use array index as a field name. To
        specify all array elements, use '*'.
    :type projection: Sequence[str]
    :param remove_none_values: If set to Truethen none values are removed from
        frames (except for metadata)
    :type remove_none_values: bool
    :param clean_subfields: If set to Truethen both frame toplevel fields and
        subfields are cleaned according to the schema. Otherwise only top level fields
    :type clean_subfields: bool
    """

    _service = "frames"
    _action = "get_next_for_dataview"
    _version = "2.23"
    _schema = {
        "definitions": {
            "dataview": {
                "properties": {
                    "augmentation": {
                        "description": "Augmentation parameters. Only for training and testing tasks.",
                        "oneOf": [
                            {"$ref": "#/definitions/dv_augmentation"},
                            {"type": "null"},
                        ],
                    },
                    "filters": {
                        "description": "List of FilterRule ('OR' relationship)",
                        "items": {"$ref": "#/definitions/filter_rule"},
                        "type": ["array", "null"],
                    },
                    "iteration": {
                        "description": "Iteration parameters. Not applicable for register (import) tasks.",
                        "oneOf": [
                            {"$ref": "#/definitions/iteration"},
                            {"type": "null"},
                        ],
                    },
                    "labels_enumeration": {
                        "additionalProperties": {"type": "integer"},
                        "description": (
                            "Labels enumerations, specifies numbers to be assigned to ROI labels when getting frames"
                        ),
                        "type": ["object", "null"],
                    },
                    "mapping": {
                        "description": "Mapping parameters",
                        "oneOf": [{"$ref": "#/definitions/mapping"}, {"type": "null"}],
                    },
                    "output_rois": {
                        "description": (
                            "'all_in_frame' - all rois for a frame are returned\n\n'only_filtered' - only rois which"
                            " led this frame to be selected\n\n'frame_per_roi' - single roi per frame. Frame can be"
                            " returned multiple times with a different roi each time.\n\nNote: this should be used for"
                            " Training tasks only\n\nNote: frame_per_roi implies that only filtered rois will be"
                            " returned\n                "
                        ),
                        "oneOf": [
                            {"$ref": "#/definitions/output_rois_enum"},
                            {"type": "null"},
                        ],
                    },
                    "versions": {
                        "description": "View dataset versions",
                        "items": {"$ref": "#/definitions/view_entry"},
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            },
            "dv_augmentation": {
                "properties": {
                    "crop_around_rois": {
                        "description": "Crop image data around all frame ROIs",
                        "type": ["boolean", "null"],
                    },
                    "sets": {
                        "description": "List of augmentation sets",
                        "items": {"$ref": "#/definitions/dv_augmentation_set"},
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            },
            "dv_augmentation_set": {
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
            "flow_control": {
                "properties": {
                    "bidirectional": {
                        "description": (
                            "If set then frames retreival can go either forward or backwards. Otherwise only forward.\n"
                            "                The default is False. The limitations of bidirectional navigation:\n      "
                            "          - Frames are always returned in sequential order\n                - The"
                            " iteration is finite (no support for infinite iteration)\n                "
                        ),
                        "type": ["boolean", "null"],
                    },
                    "navigate_backwards": {
                        "description": (
                            "When bidirectional is True, settings this to True navigates backwards duing frames"
                            " retreival. Default is False"
                        ),
                        "type": ["boolean", "null"],
                    },
                },
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
            "view_entry": {
                "properties": {
                    "dataset": {
                        "description": "Existing Dataset id",
                        "type": ["string", "null"],
                    },
                    "merge_with": {
                        "description": "Version ID to merge with",
                        "type": ["string", "null"],
                    },
                    "version": {
                        "description": "Version id of a version belonging to the dataset",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
        },
        "properties": {
            "batch_size": {
                "default": 500,
                "description": "Max number of images to be returned. Used only if scroll_id is not provided.",
                "type": "integer",
            },
            "clean_subfields": {
                "default": False,
                "description": (
                    "If set to Truethen both frame toplevel fields and subfields are cleaned according to the schema."
                    " Otherwise only top level fields"
                ),
                "type": "boolean",
            },
            "dataview": {
                "$ref": "#/definitions/dataview",
                "description": "Dataview specification",
            },
            "flow_control": {
                "$ref": "#/definitions/flow_control",
                "description": (
                    "Contol if frames retreival always navigate in one direction (the default) or can navigate forwards"
                    " and backwards"
                ),
            },
            "force_scroll_id": {"description": "", "type": "boolean"},
            "node": {
                "description": (
                    "Node number. This provides support for multi-node experiments running multiple workers executing"
                    " the same experiment on multiple processes or machines"
                ),
                "type": "integer",
            },
            "projection": {
                "description": (
                    "Used to select which parts of the frame will be returned. Each string represents a\n              "
                    "      field or sub-field (using dot-separated notation). In order to specify a specific array"
                    " element,\n                    use array index as a field name. To specify all array elements, use"
                    " '*'."
                ),
                "items": {"type": "string"},
                "type": "array",
            },
            "random_seed": {
                "description": "Optional random seed used for frame selection. If not provided, one will be generated.",
                "type": "integer",
            },
            "remove_none_values": {
                "default": False,
                "description": "If set to Truethen none values are removed from frames (except for metadata)",
                "type": "boolean",
            },
            "reset_scroll": {"description": "", "type": "boolean"},
            "scroll_id": {
                "description": "Scroll session id for getting the next batch of images",
                "type": "string",
            },
        },
        "required": ["dataview"],
        "type": "object",
    }

    def __init__(
        self,
        dataview,
        scroll_id=None,
        batch_size=500,
        reset_scroll=None,
        force_scroll_id=None,
        flow_control=None,
        random_seed=None,
        node=None,
        projection=None,
        remove_none_values=False,
        clean_subfields=False,
        **kwargs
    ):
        super(GetNextForDataviewRequest, self).__init__(**kwargs)
        self.dataview = dataview
        self.scroll_id = scroll_id
        self.batch_size = batch_size
        self.reset_scroll = reset_scroll
        self.force_scroll_id = force_scroll_id
        self.flow_control = flow_control
        self.random_seed = random_seed
        self.node = node
        self.projection = projection
        self.remove_none_values = remove_none_values
        self.clean_subfields = clean_subfields

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

    @schema_property("batch_size")
    def batch_size(self):
        return self._property_batch_size

    @batch_size.setter
    def batch_size(self, value):
        if value is None:
            self._property_batch_size = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "batch_size", six.integer_types)
        self._property_batch_size = value

    @schema_property("reset_scroll")
    def reset_scroll(self):
        return self._property_reset_scroll

    @reset_scroll.setter
    def reset_scroll(self, value):
        if value is None:
            self._property_reset_scroll = None
            return

        self.assert_isinstance(value, "reset_scroll", (bool,))
        self._property_reset_scroll = value

    @schema_property("force_scroll_id")
    def force_scroll_id(self):
        return self._property_force_scroll_id

    @force_scroll_id.setter
    def force_scroll_id(self, value):
        if value is None:
            self._property_force_scroll_id = None
            return

        self.assert_isinstance(value, "force_scroll_id", (bool,))
        self._property_force_scroll_id = value

    @schema_property("flow_control")
    def flow_control(self):
        return self._property_flow_control

    @flow_control.setter
    def flow_control(self, value):
        if value is None:
            self._property_flow_control = None
            return
        if isinstance(value, dict):
            value = FlowControl.from_dict(value)
        else:
            self.assert_isinstance(value, "flow_control", FlowControl)
        self._property_flow_control = value

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

    @schema_property("node")
    def node(self):
        return self._property_node

    @node.setter
    def node(self, value):
        if value is None:
            self._property_node = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "node", six.integer_types)
        self._property_node = value

    @schema_property("projection")
    def projection(self):
        return self._property_projection

    @projection.setter
    def projection(self, value):
        if value is None:
            self._property_projection = None
            return

        self.assert_isinstance(value, "projection", (list, tuple))

        self.assert_isinstance(value, "projection", six.string_types, is_array=True)
        self._property_projection = value

    @schema_property("remove_none_values")
    def remove_none_values(self):
        return self._property_remove_none_values

    @remove_none_values.setter
    def remove_none_values(self, value):
        if value is None:
            self._property_remove_none_values = None
            return

        self.assert_isinstance(value, "remove_none_values", (bool,))
        self._property_remove_none_values = value

    @schema_property("clean_subfields")
    def clean_subfields(self):
        return self._property_clean_subfields

    @clean_subfields.setter
    def clean_subfields(self, value):
        if value is None:
            self._property_clean_subfields = None
            return

        self.assert_isinstance(value, "clean_subfields", (bool,))
        self._property_clean_subfields = value


class GetNextForDataviewResponse(Response):
    """
    Response of frames.get_next_for_dataview endpoint.

    :param frames: Frames list
    :type frames: Sequence[Frame]
    :param frames_returned: Number of frames returned
    :type frames_returned: int
    :param scroll_state: JSON object representing the scroll state
    :type scroll_state: dict
    :param scroll_id: Scroll session id to be provided in order to get the next
        batch of images
    :type scroll_id: str
    :param roi_stats: Json object containing the count per labels in frames, e.g. {
        'background': 312, 'boat': 2, 'bus': 4, 'car': 2, }
    :type roi_stats: dict
    :param eof: When 'frames' is empty, represents whether there are no more frames
        left. If "false", client can retry the operation.
    :type eof: bool
    :param random_seed: Random seed used for frame selection
    :type random_seed: int
    """

    _service = "frames"
    _action = "get_next_for_dataview"
    _version = "2.23"

    _schema = {
        "definitions": {
            "augmentation": {
                "properties": {
                    "arguments": {
                        "additionalProperties": True,
                        "description": "Arguments dictionary, passed to custom augmentations.",
                        "type": ["object", "null"],
                    },
                    "cls": {
                        "description": "Augmentation class (see global definitions)",
                        "type": ["string", "null"],
                    },
                    "params": {
                        "description": (
                            "Transform parameters, an array ot 3 randomly generated values. Fixed values are passed in"
                            " case of affine reflect augmentation."
                        ),
                        "items": {"type": "number"},
                        "type": ["array", "null"],
                    },
                    "strength": {
                        "description": "Transform strength. Required for pixel transforms.",
                        "type": ["number", "null"],
                    },
                    "trans_mat": {
                        "description": "Transform matrix (list of lists). Required for affine transforms.",
                        "items": {"items": {"type": "number"}, "type": "array"},
                        "type": ["array", "null"],
                    },
                    "type": {
                        "description": "Augmentation type (see global definitions)",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "dataset_version": {
                "properties": {
                    "id": {"description": "Dataset id", "type": ["string", "null"]},
                    "version": {
                        "description": "Dataset version id",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "frame": {
                "properties": {
                    "augmentation": {
                        "description": "List of augmentations",
                        "items": {"$ref": "#/definitions/augmentation"},
                        "type": ["array", "null"],
                    },
                    "blob": {
                        "description": "Raw data (blob) for the frame",
                        "type": ["string", "null"],
                    },
                    "context_id": {
                        "description": (
                            "Context ID. Used for the default frames sorting. If not set then it is filled from the uri"
                            " of the first source."
                        ),
                        "type": ["string", "null"],
                    },
                    "dataset": {
                        "description": "Frame's dataset version",
                        "oneOf": [
                            {"$ref": "#/definitions/dataset_version"},
                            {"type": "null"},
                        ],
                    },
                    "id": {"description": "Frame id", "type": ["string", "null"]},
                    "is_key_frame": {
                        "description": "Is this a key frame (only applicable in frames who'se src is a video)",
                        "type": ["boolean", "null"],
                    },
                    "key_frame": {
                        "description": "ID of the key frame that this frame belongs to",
                        "type": ["string", "null"],
                    },
                    "label_rule_counts": {
                        "additionalProperties": True,
                        "description": "The number of matched roi per lable rule",
                        "type": ["object", "null"],
                    },
                    "labels_size": {
                        "description": "Number of labels returned",
                        "type": ["integer", "null"],
                    },
                    "meta": {
                        "additionalProperties": True,
                        "description": (
                            "Additional metadata dictionary for the frame. Please note that using this field"
                            " effectively defines a schema (dictionary structure and types used as values) - frames"
                            " within the same dataset cannot use conflicting schemas for this field (see documentation"
                            " for more details)."
                        ),
                        "type": ["object", "null"],
                    },
                    "meta_blob": {
                        "additionalProperties": True,
                        "description": (
                            "Non searchable metadata dictionary for the frame. The fields in this object cannot be"
                            " searched by and are not added to the frame schema"
                        ),
                        "type": ["object", "null"],
                    },
                    "new_ver": {
                        "description": "Newer version of this frame, if asked to merge",
                        "oneOf": [{"$ref": "#/definitions/frame"}, {"type": "null"}],
                    },
                    "rois": {
                        "description": "Frame regions of interest",
                        "items": {"$ref": "#/definitions/roi"},
                        "type": ["array", "null"],
                    },
                    "rule_name": {
                        "description": (
                            "Name of the filtering rule according to which this frame was provided (if applicable)"
                        ),
                        "type": ["string", "null"],
                    },
                    "saved": {
                        "description": "Last time frame was saved (timestamp)",
                        "type": ["integer", "null"],
                    },
                    "saved_in_version": {
                        "description": "Last version this frame was saved in (version ID)",
                        "type": ["string", "null"],
                    },
                    "sources": {
                        "description": "Sources of this frame",
                        "items": {"$ref": "#/definitions/source"},
                        "type": ["array", "null"],
                    },
                    "timestamp": {
                        "description": (
                            "Frame's offset in milliseconds, used primarily for video content. Used for the default"
                            " frames sorting as the secondary key (with the primary key being 'context_id'). For"
                            " images, this value should typically be 0. If not set, value is filled from the timestamp"
                            " of the first source. We recommend using this field only in cases concerning the default"
                            " sorting behavior."
                        ),
                        "type": ["integer", "null"],
                    },
                    "updated": {
                        "description": "Last time frame was saved (timestamp)",
                        "type": ["integer", "null"],
                    },
                    "updated_in_version": {
                        "description": "Last version this frame was updated in (version ID)",
                        "type": ["string", "null"],
                    },
                    "video_gop": {
                        "description": (
                            "Video encoding GOP value for the source of this frame. Only valid for video frames"
                        ),
                        "type": ["number", "null"],
                    },
                },
                "type": "object",
            },
            "mask": {
                "properties": {
                    "content_type": {
                        "description": "Content type (e.g. 'image/jpeg', 'image/png')",
                        "type": ["string", "null"],
                    },
                    "height": {
                        "description": "Height in pixels",
                        "type": ["integer", "null"],
                    },
                    "id": {
                        "description": "unique ID (in this frame)",
                        "type": ["string", "null"],
                    },
                    "timestamp": {
                        "default": 0,
                        "description": (
                            "Timestamp in the source data (for video content. for images, this value should be 0)"
                        ),
                        "type": ["integer", "null"],
                    },
                    "uri": {"description": "Data URI", "type": ["string", "null"]},
                    "width": {
                        "description": "Width in pixels",
                        "type": ["integer", "null"],
                    },
                },
                "type": "object",
            },
            "preview": {
                "properties": {
                    "content_type": {
                        "description": "Content type (e.g. 'image/jpeg', 'image/png')",
                        "type": ["string", "null"],
                    },
                    "height": {
                        "description": "Height in pixels",
                        "type": ["integer", "null"],
                    },
                    "timestamp": {
                        "default": 0,
                        "description": (
                            "Timestamp in the source data (for video content. for images, this value should be 0)"
                        ),
                        "type": ["integer", "null"],
                    },
                    "uri": {"description": "Data URI", "type": ["string", "null"]},
                    "width": {
                        "description": "Width in pixels",
                        "type": ["integer", "null"],
                    },
                },
                "type": "object",
            },
            "roi": {
                "properties": {
                    "area": {
                        "description": "ROI area (not used)",
                        "type": ["integer", "null"],
                    },
                    "confidence": {
                        "description": "ROI confidence",
                        "type": ["number", "null"],
                    },
                    "id": {"description": "ROI id", "type": ["string", "null"]},
                    "label": {
                        "description": "ROI labels",
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                    "label_num": {
                        "description": (
                            "Label number according to the specified labels mapping Used only when ROI is returned as"
                            " part of a task's frame."
                        ),
                        "type": ["integer", "null"],
                    },
                    "mask": {
                        "description": "Mask info for this ROI",
                        "oneOf": [{"$ref": "#/definitions/roi_mask"}, {"type": "null"}],
                    },
                    "meta": {
                        "additionalProperties": True,
                        "description": "Additional metadata dictionary for the roi",
                        "type": ["object", "null"],
                    },
                    "poly": {
                        "description": "ROI polygon (x0, y0, ..., xn, yn)",
                        "items": {"type": "number"},
                        "type": ["array", "null"],
                    },
                    "sources": {
                        "description": "Sources that this ROI belongs to",
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            },
            "roi_mask": {
                "properties": {
                    "id": {"description": "Mask ID", "type": "string"},
                    "value": {
                        "description": "Mask value",
                        "items": {"type": "integer"},
                        "type": "array",
                    },
                },
                "required": ["id", "value"],
                "type": "object",
            },
            "source": {
                "properties": {
                    "content_type": {
                        "description": "Content type (e.g. 'image/jpeg', 'image/png')",
                        "type": ["string", "null"],
                    },
                    "height": {
                        "description": "Height in pixels",
                        "type": ["integer", "null"],
                    },
                    "id": {
                        "description": "unique ID (in this frame)",
                        "type": ["string", "null"],
                    },
                    "masks": {
                        "items": {"$ref": "#/definitions/mask"},
                        "type": ["array", "null"],
                    },
                    "meta": {
                        "additionalProperties": True,
                        "description": "Additional metadata dictionary for the source",
                        "type": ["object", "null"],
                    },
                    "preview": {
                        "oneOf": [{"$ref": "#/definitions/preview"}, {"type": "null"}]
                    },
                    "timestamp": {
                        "default": 0,
                        "description": (
                            "Timestamp in the source data (for video content. for images, this value should be 0)"
                        ),
                        "type": ["integer", "null"],
                    },
                    "uri": {"description": "Data URI", "type": ["string", "null"]},
                    "width": {
                        "description": "Width in pixels",
                        "type": ["integer", "null"],
                    },
                },
                "type": "object",
            },
        },
        "properties": {
            "eof": {
                "description": (
                    "When 'frames' is empty, represents whether there are no more frames left. "
                    'If "false",\n'
                    "                client can retry the "
                    "operation."
                ),
                "type": ["boolean", "null"],
            },
            "frames": {
                "description": "Frames list",
                "items": {"$ref": "#/definitions/frame"},
                "type": ["array", "null"],
            },
            "frames_returned": {
                "description": "Number of frames returned",
                "type": ["integer", "null"],
            },
            "random_seed": {
                "description": "Random seed used for frame selection",
                "type": ["integer", "null"],
            },
            "roi_stats": {
                "additionalProperties": {"type": "integer"},
                "description": (
                    "Json object containing the count per labels in frames, e.g.\n                {\n                  "
                    "  'background': 312,\n                    'boat': 2,\n                    'bus': 4,\n             "
                    "       'car': 2,\n                }"
                ),
                "type": ["object", "null"],
            },
            "scroll_id": {
                "description": "Scroll session id to be provided in order to get the next batch of images",
                "type": ["string", "null"],
            },
            "scroll_state": {
                "additionalProperties": True,
                "description": "JSON object representing the scroll state",
                "type": ["object", "null"],
            },
        },
        "type": "object",
    }

    def __init__(
        self,
        frames=None,
        frames_returned=None,
        scroll_state=None,
        scroll_id=None,
        roi_stats=None,
        eof=None,
        random_seed=None,
        **kwargs
    ):
        super(GetNextForDataviewResponse, self).__init__(**kwargs)
        self.frames = frames
        self.frames_returned = frames_returned
        self.scroll_state = scroll_state
        self.scroll_id = scroll_id
        self.roi_stats = roi_stats
        self.eof = eof
        self.random_seed = random_seed

    @schema_property("frames")
    def frames(self):
        return self._property_frames

    @frames.setter
    def frames(self, value):
        if value is None:
            self._property_frames = None
            return

        self.assert_isinstance(value, "frames", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [Frame.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "frames", Frame, is_array=True)
        self._property_frames = value

    @schema_property("frames_returned")
    def frames_returned(self):
        return self._property_frames_returned

    @frames_returned.setter
    def frames_returned(self, value):
        if value is None:
            self._property_frames_returned = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "frames_returned", six.integer_types)
        self._property_frames_returned = value

    @schema_property("scroll_state")
    def scroll_state(self):
        return self._property_scroll_state

    @scroll_state.setter
    def scroll_state(self, value):
        if value is None:
            self._property_scroll_state = None
            return

        self.assert_isinstance(value, "scroll_state", (dict,))
        self._property_scroll_state = value

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

    @schema_property("roi_stats")
    def roi_stats(self):
        return self._property_roi_stats

    @roi_stats.setter
    def roi_stats(self, value):
        if value is None:
            self._property_roi_stats = None
            return

        self.assert_isinstance(value, "roi_stats", (dict,))
        self._property_roi_stats = value

    @schema_property("eof")
    def eof(self):
        return self._property_eof

    @eof.setter
    def eof(self, value):
        if value is None:
            self._property_eof = None
            return

        self.assert_isinstance(value, "eof", (bool,))
        self._property_eof = value

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


class GetNextForDataviewAndContextIdRequest(Request):
    """
    Get frames for the given dataview specification and context id

    :param dataview: Dataview specification
    :type dataview: Dataview
    :param scroll_id: Scroll session id for getting the next batch of images
    :type scroll_id: str
    :param batch_size: Max number of images to be returned. Used only if scroll_id
        is not provided.
    :type batch_size: int
    :param flow_control: Contol if frames retreival always navigate in one
        direction (the default) or can navigate forwards and backwards
    :type flow_control: FlowControl
    :param random_seed: Optional random seed used for frame selection. If not
        provided, one will be generated.
    :type random_seed: int
    :param node: Node number. This provides support for multi-node experiments
        running multiple workers executing the same experiment on multiple processes or
        machines
    :type node: int
    :param projection: Used to select which parts of the frame will be returned.
        Each string represents a field or sub-field (using dot-separated notation). In
        order to specify a specific array element, use array index as a field name. To
        specify all array elements, use '*'.
    :type projection: Sequence[str]
    :param context_id: Retrieve only frames associated with this context_id
    :type context_id: str
    :param timestamp: Start timestamp for frames returned. Optional. Default is 0
    :type timestamp: int
    :param remove_none_values: If set to Truethen none values are removed from
        frames (except for metadata)
    :type remove_none_values: bool
    :param clean_subfields: If set to Truethen both frame toplevel fields and
        subfields are cleaned according to the schema. Otherwise only top level fields
    :type clean_subfields: bool
    """

    _service = "frames"
    _action = "get_next_for_dataview_and_context_id"
    _version = "2.23"
    _schema = {
        "definitions": {
            "dataview": {
                "properties": {
                    "augmentation": {
                        "description": "Augmentation parameters. Only for training and testing tasks.",
                        "oneOf": [
                            {"$ref": "#/definitions/dv_augmentation"},
                            {"type": "null"},
                        ],
                    },
                    "filters": {
                        "description": "List of FilterRule ('OR' relationship)",
                        "items": {"$ref": "#/definitions/filter_rule"},
                        "type": ["array", "null"],
                    },
                    "iteration": {
                        "description": "Iteration parameters. Not applicable for register (import) tasks.",
                        "oneOf": [
                            {"$ref": "#/definitions/iteration"},
                            {"type": "null"},
                        ],
                    },
                    "labels_enumeration": {
                        "additionalProperties": {"type": "integer"},
                        "description": (
                            "Labels enumerations, specifies numbers to be assigned to ROI labels when getting frames"
                        ),
                        "type": ["object", "null"],
                    },
                    "mapping": {
                        "description": "Mapping parameters",
                        "oneOf": [{"$ref": "#/definitions/mapping"}, {"type": "null"}],
                    },
                    "output_rois": {
                        "description": (
                            "'all_in_frame' - all rois for a frame are returned\n\n'only_filtered' - only rois which"
                            " led this frame to be selected\n\n'frame_per_roi' - single roi per frame. Frame can be"
                            " returned multiple times with a different roi each time.\n\nNote: this should be used for"
                            " Training tasks only\n\nNote: frame_per_roi implies that only filtered rois will be"
                            " returned\n                "
                        ),
                        "oneOf": [
                            {"$ref": "#/definitions/output_rois_enum"},
                            {"type": "null"},
                        ],
                    },
                    "versions": {
                        "description": "View dataset versions",
                        "items": {"$ref": "#/definitions/view_entry"},
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            },
            "dv_augmentation": {
                "properties": {
                    "crop_around_rois": {
                        "description": "Crop image data around all frame ROIs",
                        "type": ["boolean", "null"],
                    },
                    "sets": {
                        "description": "List of augmentation sets",
                        "items": {"$ref": "#/definitions/dv_augmentation_set"},
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            },
            "dv_augmentation_set": {
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
            "flow_control": {
                "properties": {
                    "bidirectional": {
                        "description": (
                            "If set then frames retreival can go either forward or backwards. Otherwise only forward.\n"
                            "                The default is False. The limitations of bidirectional navigation:\n      "
                            "          - Frames are always returned in sequential order\n                - The"
                            " iteration is finite (no support for infinite iteration)\n                "
                        ),
                        "type": ["boolean", "null"],
                    },
                    "navigate_backwards": {
                        "description": (
                            "When bidirectional is True, settings this to True navigates backwards duing frames"
                            " retreival. Default is False"
                        ),
                        "type": ["boolean", "null"],
                    },
                },
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
            "view_entry": {
                "properties": {
                    "dataset": {
                        "description": "Existing Dataset id",
                        "type": ["string", "null"],
                    },
                    "merge_with": {
                        "description": "Version ID to merge with",
                        "type": ["string", "null"],
                    },
                    "version": {
                        "description": "Version id of a version belonging to the dataset",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
        },
        "properties": {
            "batch_size": {
                "default": 500,
                "description": "Max number of images to be returned. Used only if scroll_id is not provided.",
                "type": "integer",
            },
            "clean_subfields": {
                "default": False,
                "description": (
                    "If set to Truethen both frame toplevel fields and subfields are cleaned according to the schema."
                    " Otherwise only top level fields"
                ),
                "type": "boolean",
            },
            "context_id": {
                "description": "Retrieve only frames associated with this context_id",
                "type": "string",
            },
            "dataview": {
                "$ref": "#/definitions/dataview",
                "description": "Dataview specification",
            },
            "flow_control": {
                "$ref": "#/definitions/flow_control",
                "description": (
                    "Contol if frames retreival always navigate in one direction (the default) or can navigate forwards"
                    " and backwards"
                ),
            },
            "node": {
                "description": (
                    "Node number. This provides support for multi-node experiments running multiple workers executing"
                    " the same experiment on multiple processes or machines"
                ),
                "type": "integer",
            },
            "projection": {
                "description": (
                    "Used to select which parts of the frame will be returned. Each string represents a\n              "
                    "      field or sub-field (using dot-separated notation). In order to specify a specific array"
                    " element,\n                    use array index as a field name. To specify all array elements, use"
                    " '*'."
                ),
                "items": {"type": "string"},
                "type": "array",
            },
            "random_seed": {
                "description": "Optional random seed used for frame selection. If not provided, one will be generated.",
                "type": "integer",
            },
            "remove_none_values": {
                "default": False,
                "description": "If set to Truethen none values are removed from frames (except for metadata)",
                "type": "boolean",
            },
            "scroll_id": {
                "description": "Scroll session id for getting the next batch of images",
                "type": "string",
            },
            "timestamp": {
                "default": 0,
                "description": "Start timestamp for frames returned. Optional. Default is 0",
                "type": "integer",
            },
        },
        "required": ["dataview", "context_id"],
        "type": "object",
    }

    def __init__(
        self,
        dataview,
        context_id,
        scroll_id=None,
        batch_size=500,
        flow_control=None,
        random_seed=None,
        node=None,
        projection=None,
        timestamp=0,
        remove_none_values=False,
        clean_subfields=False,
        **kwargs
    ):
        super(GetNextForDataviewAndContextIdRequest, self).__init__(**kwargs)
        self.dataview = dataview
        self.scroll_id = scroll_id
        self.batch_size = batch_size
        self.flow_control = flow_control
        self.random_seed = random_seed
        self.node = node
        self.projection = projection
        self.context_id = context_id
        self.timestamp = timestamp
        self.remove_none_values = remove_none_values
        self.clean_subfields = clean_subfields

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

    @schema_property("batch_size")
    def batch_size(self):
        return self._property_batch_size

    @batch_size.setter
    def batch_size(self, value):
        if value is None:
            self._property_batch_size = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "batch_size", six.integer_types)
        self._property_batch_size = value

    @schema_property("flow_control")
    def flow_control(self):
        return self._property_flow_control

    @flow_control.setter
    def flow_control(self, value):
        if value is None:
            self._property_flow_control = None
            return
        if isinstance(value, dict):
            value = FlowControl.from_dict(value)
        else:
            self.assert_isinstance(value, "flow_control", FlowControl)
        self._property_flow_control = value

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

    @schema_property("node")
    def node(self):
        return self._property_node

    @node.setter
    def node(self, value):
        if value is None:
            self._property_node = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "node", six.integer_types)
        self._property_node = value

    @schema_property("projection")
    def projection(self):
        return self._property_projection

    @projection.setter
    def projection(self, value):
        if value is None:
            self._property_projection = None
            return

        self.assert_isinstance(value, "projection", (list, tuple))

        self.assert_isinstance(value, "projection", six.string_types, is_array=True)
        self._property_projection = value

    @schema_property("context_id")
    def context_id(self):
        return self._property_context_id

    @context_id.setter
    def context_id(self, value):
        if value is None:
            self._property_context_id = None
            return

        self.assert_isinstance(value, "context_id", six.string_types)
        self._property_context_id = value

    @schema_property("timestamp")
    def timestamp(self):
        return self._property_timestamp

    @timestamp.setter
    def timestamp(self, value):
        if value is None:
            self._property_timestamp = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "timestamp", six.integer_types)
        self._property_timestamp = value

    @schema_property("remove_none_values")
    def remove_none_values(self):
        return self._property_remove_none_values

    @remove_none_values.setter
    def remove_none_values(self, value):
        if value is None:
            self._property_remove_none_values = None
            return

        self.assert_isinstance(value, "remove_none_values", (bool,))
        self._property_remove_none_values = value

    @schema_property("clean_subfields")
    def clean_subfields(self):
        return self._property_clean_subfields

    @clean_subfields.setter
    def clean_subfields(self, value):
        if value is None:
            self._property_clean_subfields = None
            return

        self.assert_isinstance(value, "clean_subfields", (bool,))
        self._property_clean_subfields = value


class GetNextForDataviewAndContextIdResponse(Response):
    """
    Response of frames.get_next_for_dataview_and_context_id endpoint.

    :param frames: Frames list
    :type frames: Sequence[Frame]
    :param frames_returned: Number of frames returned
    :type frames_returned: int
    :param scroll_state: JSON object representing the scroll state
    :type scroll_state: dict
    :param scroll_id: Scroll session id to be provided in order to get the next
        batch of images
    :type scroll_id: str
    :param roi_stats: Json object containing the count per labels in frames, e.g. {
        'background': 312, 'boat': 2, 'bus': 4, 'car': 2, }
    :type roi_stats: dict
    :param eof: When 'frames' is empty, represents whether there are no more frames
        left. If "false", client can retry the operation.
    :type eof: bool
    :param random_seed: Random seed used for frame selection
    :type random_seed: int
    """

    _service = "frames"
    _action = "get_next_for_dataview_and_context_id"
    _version = "2.23"

    _schema = {
        "definitions": {
            "augmentation": {
                "properties": {
                    "arguments": {
                        "additionalProperties": True,
                        "description": "Arguments dictionary, passed to custom augmentations.",
                        "type": ["object", "null"],
                    },
                    "cls": {
                        "description": "Augmentation class (see global definitions)",
                        "type": ["string", "null"],
                    },
                    "params": {
                        "description": (
                            "Transform parameters, an array ot 3 randomly generated values. Fixed values are passed in"
                            " case of affine reflect augmentation."
                        ),
                        "items": {"type": "number"},
                        "type": ["array", "null"],
                    },
                    "strength": {
                        "description": "Transform strength. Required for pixel transforms.",
                        "type": ["number", "null"],
                    },
                    "trans_mat": {
                        "description": "Transform matrix (list of lists). Required for affine transforms.",
                        "items": {"items": {"type": "number"}, "type": "array"},
                        "type": ["array", "null"],
                    },
                    "type": {
                        "description": "Augmentation type (see global definitions)",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "dataset_version": {
                "properties": {
                    "id": {"description": "Dataset id", "type": ["string", "null"]},
                    "version": {
                        "description": "Dataset version id",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "frame": {
                "properties": {
                    "augmentation": {
                        "description": "List of augmentations",
                        "items": {"$ref": "#/definitions/augmentation"},
                        "type": ["array", "null"],
                    },
                    "blob": {
                        "description": "Raw data (blob) for the frame",
                        "type": ["string", "null"],
                    },
                    "context_id": {
                        "description": (
                            "Context ID. Used for the default frames sorting. If not set then it is filled from the uri"
                            " of the first source."
                        ),
                        "type": ["string", "null"],
                    },
                    "dataset": {
                        "description": "Frame's dataset version",
                        "oneOf": [
                            {"$ref": "#/definitions/dataset_version"},
                            {"type": "null"},
                        ],
                    },
                    "id": {"description": "Frame id", "type": ["string", "null"]},
                    "is_key_frame": {
                        "description": "Is this a key frame (only applicable in frames who'se src is a video)",
                        "type": ["boolean", "null"],
                    },
                    "key_frame": {
                        "description": "ID of the key frame that this frame belongs to",
                        "type": ["string", "null"],
                    },
                    "label_rule_counts": {
                        "additionalProperties": True,
                        "description": "The number of matched roi per lable rule",
                        "type": ["object", "null"],
                    },
                    "labels_size": {
                        "description": "Number of labels returned",
                        "type": ["integer", "null"],
                    },
                    "meta": {
                        "additionalProperties": True,
                        "description": (
                            "Additional metadata dictionary for the frame. Please note that using this field"
                            " effectively defines a schema (dictionary structure and types used as values) - frames"
                            " within the same dataset cannot use conflicting schemas for this field (see documentation"
                            " for more details)."
                        ),
                        "type": ["object", "null"],
                    },
                    "meta_blob": {
                        "additionalProperties": True,
                        "description": (
                            "Non searchable metadata dictionary for the frame. The fields in this object cannot be"
                            " searched by and are not added to the frame schema"
                        ),
                        "type": ["object", "null"],
                    },
                    "new_ver": {
                        "description": "Newer version of this frame, if asked to merge",
                        "oneOf": [{"$ref": "#/definitions/frame"}, {"type": "null"}],
                    },
                    "rois": {
                        "description": "Frame regions of interest",
                        "items": {"$ref": "#/definitions/roi"},
                        "type": ["array", "null"],
                    },
                    "rule_name": {
                        "description": (
                            "Name of the filtering rule according to which this frame was provided (if applicable)"
                        ),
                        "type": ["string", "null"],
                    },
                    "saved": {
                        "description": "Last time frame was saved (timestamp)",
                        "type": ["integer", "null"],
                    },
                    "saved_in_version": {
                        "description": "Last version this frame was saved in (version ID)",
                        "type": ["string", "null"],
                    },
                    "sources": {
                        "description": "Sources of this frame",
                        "items": {"$ref": "#/definitions/source"},
                        "type": ["array", "null"],
                    },
                    "timestamp": {
                        "description": (
                            "Frame's offset in milliseconds, used primarily for video content. Used for the default"
                            " frames sorting as the secondary key (with the primary key being 'context_id'). For"
                            " images, this value should typically be 0. If not set, value is filled from the timestamp"
                            " of the first source. We recommend using this field only in cases concerning the default"
                            " sorting behavior."
                        ),
                        "type": ["integer", "null"],
                    },
                    "updated": {
                        "description": "Last time frame was saved (timestamp)",
                        "type": ["integer", "null"],
                    },
                    "updated_in_version": {
                        "description": "Last version this frame was updated in (version ID)",
                        "type": ["string", "null"],
                    },
                    "video_gop": {
                        "description": (
                            "Video encoding GOP value for the source of this frame. Only valid for video frames"
                        ),
                        "type": ["number", "null"],
                    },
                },
                "type": "object",
            },
            "mask": {
                "properties": {
                    "content_type": {
                        "description": "Content type (e.g. 'image/jpeg', 'image/png')",
                        "type": ["string", "null"],
                    },
                    "height": {
                        "description": "Height in pixels",
                        "type": ["integer", "null"],
                    },
                    "id": {
                        "description": "unique ID (in this frame)",
                        "type": ["string", "null"],
                    },
                    "timestamp": {
                        "default": 0,
                        "description": (
                            "Timestamp in the source data (for video content. for images, this value should be 0)"
                        ),
                        "type": ["integer", "null"],
                    },
                    "uri": {"description": "Data URI", "type": ["string", "null"]},
                    "width": {
                        "description": "Width in pixels",
                        "type": ["integer", "null"],
                    },
                },
                "type": "object",
            },
            "preview": {
                "properties": {
                    "content_type": {
                        "description": "Content type (e.g. 'image/jpeg', 'image/png')",
                        "type": ["string", "null"],
                    },
                    "height": {
                        "description": "Height in pixels",
                        "type": ["integer", "null"],
                    },
                    "timestamp": {
                        "default": 0,
                        "description": (
                            "Timestamp in the source data (for video content. for images, this value should be 0)"
                        ),
                        "type": ["integer", "null"],
                    },
                    "uri": {"description": "Data URI", "type": ["string", "null"]},
                    "width": {
                        "description": "Width in pixels",
                        "type": ["integer", "null"],
                    },
                },
                "type": "object",
            },
            "roi": {
                "properties": {
                    "area": {
                        "description": "ROI area (not used)",
                        "type": ["integer", "null"],
                    },
                    "confidence": {
                        "description": "ROI confidence",
                        "type": ["number", "null"],
                    },
                    "id": {"description": "ROI id", "type": ["string", "null"]},
                    "label": {
                        "description": "ROI labels",
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                    "label_num": {
                        "description": (
                            "Label number according to the specified labels mapping Used only when ROI is returned as"
                            " part of a task's frame."
                        ),
                        "type": ["integer", "null"],
                    },
                    "mask": {
                        "description": "Mask info for this ROI",
                        "oneOf": [{"$ref": "#/definitions/roi_mask"}, {"type": "null"}],
                    },
                    "meta": {
                        "additionalProperties": True,
                        "description": "Additional metadata dictionary for the roi",
                        "type": ["object", "null"],
                    },
                    "poly": {
                        "description": "ROI polygon (x0, y0, ..., xn, yn)",
                        "items": {"type": "number"},
                        "type": ["array", "null"],
                    },
                    "sources": {
                        "description": "Sources that this ROI belongs to",
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            },
            "roi_mask": {
                "properties": {
                    "id": {"description": "Mask ID", "type": "string"},
                    "value": {
                        "description": "Mask value",
                        "items": {"type": "integer"},
                        "type": "array",
                    },
                },
                "required": ["id", "value"],
                "type": "object",
            },
            "source": {
                "properties": {
                    "content_type": {
                        "description": "Content type (e.g. 'image/jpeg', 'image/png')",
                        "type": ["string", "null"],
                    },
                    "height": {
                        "description": "Height in pixels",
                        "type": ["integer", "null"],
                    },
                    "id": {
                        "description": "unique ID (in this frame)",
                        "type": ["string", "null"],
                    },
                    "masks": {
                        "items": {"$ref": "#/definitions/mask"},
                        "type": ["array", "null"],
                    },
                    "meta": {
                        "additionalProperties": True,
                        "description": "Additional metadata dictionary for the source",
                        "type": ["object", "null"],
                    },
                    "preview": {
                        "oneOf": [{"$ref": "#/definitions/preview"}, {"type": "null"}]
                    },
                    "timestamp": {
                        "default": 0,
                        "description": (
                            "Timestamp in the source data (for video content. for images, this value should be 0)"
                        ),
                        "type": ["integer", "null"],
                    },
                    "uri": {"description": "Data URI", "type": ["string", "null"]},
                    "width": {
                        "description": "Width in pixels",
                        "type": ["integer", "null"],
                    },
                },
                "type": "object",
            },
        },
        "properties": {
            "eof": {
                "description": (
                    "When 'frames' is empty, represents whether there are no more frames left. "
                    'If "false",\n'
                    "                client can retry the "
                    "operation."
                ),
                "type": ["boolean", "null"],
            },
            "frames": {
                "description": "Frames list",
                "items": {"$ref": "#/definitions/frame"},
                "type": ["array", "null"],
            },
            "frames_returned": {
                "description": "Number of frames returned",
                "type": ["integer", "null"],
            },
            "random_seed": {
                "description": "Random seed used for frame selection",
                "type": ["integer", "null"],
            },
            "roi_stats": {
                "additionalProperties": {"type": "integer"},
                "description": (
                    "Json object containing the count per labels in frames, e.g.\n                {\n                  "
                    "  'background': 312,\n                    'boat': 2,\n                    'bus': 4,\n             "
                    "       'car': 2,\n                }"
                ),
                "type": ["object", "null"],
            },
            "scroll_id": {
                "description": "Scroll session id to be provided in order to get the next batch of images",
                "type": ["string", "null"],
            },
            "scroll_state": {
                "additionalProperties": True,
                "description": "JSON object representing the scroll state",
                "type": ["object", "null"],
            },
        },
        "type": "object",
    }

    def __init__(
        self,
        frames=None,
        frames_returned=None,
        scroll_state=None,
        scroll_id=None,
        roi_stats=None,
        eof=None,
        random_seed=None,
        **kwargs
    ):
        super(GetNextForDataviewAndContextIdResponse, self).__init__(**kwargs)
        self.frames = frames
        self.frames_returned = frames_returned
        self.scroll_state = scroll_state
        self.scroll_id = scroll_id
        self.roi_stats = roi_stats
        self.eof = eof
        self.random_seed = random_seed

    @schema_property("frames")
    def frames(self):
        return self._property_frames

    @frames.setter
    def frames(self, value):
        if value is None:
            self._property_frames = None
            return

        self.assert_isinstance(value, "frames", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [Frame.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "frames", Frame, is_array=True)
        self._property_frames = value

    @schema_property("frames_returned")
    def frames_returned(self):
        return self._property_frames_returned

    @frames_returned.setter
    def frames_returned(self, value):
        if value is None:
            self._property_frames_returned = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "frames_returned", six.integer_types)
        self._property_frames_returned = value

    @schema_property("scroll_state")
    def scroll_state(self):
        return self._property_scroll_state

    @scroll_state.setter
    def scroll_state(self, value):
        if value is None:
            self._property_scroll_state = None
            return

        self.assert_isinstance(value, "scroll_state", (dict,))
        self._property_scroll_state = value

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

    @schema_property("roi_stats")
    def roi_stats(self):
        return self._property_roi_stats

    @roi_stats.setter
    def roi_stats(self, value):
        if value is None:
            self._property_roi_stats = None
            return

        self.assert_isinstance(value, "roi_stats", (dict,))
        self._property_roi_stats = value

    @schema_property("eof")
    def eof(self):
        return self._property_eof

    @eof.setter
    def eof(self, value):
        if value is None:
            self._property_eof = None
            return

        self.assert_isinstance(value, "eof", (bool,))
        self._property_eof = value

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


class GetNextForDataviewIdRequest(Request):
    """
    Gets frames for a given view specification

    :param dataview: Dataview ID
    :type dataview: str
    :param scroll_id: Scroll session id for getting the next batch of images
    :type scroll_id: str
    :param batch_size: Max number of images to be returned. Used only if scroll_id
        is not provided.
    :type batch_size: int
    :param reset_scroll:
    :type reset_scroll: bool
    :param force_scroll_id:
    :type force_scroll_id: bool
    :param flow_control: Frames retreival that allows eiter one-directional
        navigation (the default) or bidirectional
    :type flow_control: FlowControl
    :param random_seed: Optional random seed used for frame selection. If not
        provided, one will be generated.
    :type random_seed: int
    :param node: Node number. This provides support for multi-node experiments
        running multiple workers executing the same experiment on multiple processes or
        machines
    :type node: int
    :param projection: Used to select which parts of the frame will be returned.
        Each string represents a field or sub-field (using dot-separated notation). In
        order to specify a specific array element, use array index as a field name. To
        specify all array elements, use '*'.
    :type projection: Sequence[str]
    :param remove_none_values: If set to Truethen none values are removed from
        frames (except for metadata)
    :type remove_none_values: bool
    :param clean_subfields: If set to Truethen both frame toplevel fields and
        subfields are cleaned according to the schema. Otherwise only top level fields
    :type clean_subfields: bool
    """

    _service = "frames"
    _action = "get_next_for_dataview_id"
    _version = "2.23"
    _schema = {
        "definitions": {
            "flow_control": {
                "properties": {
                    "bidirectional": {
                        "description": (
                            "If set then frames retreival can go either forward or backwards. Otherwise only forward.\n"
                            "                The default is False. The limitations of bidirectional navigation:\n      "
                            "          - Frames are always returned in sequential order\n                - The"
                            " iteration is finite (no support for infinite iteration)\n                "
                        ),
                        "type": ["boolean", "null"],
                    },
                    "navigate_backwards": {
                        "description": (
                            "When bidirectional is True, settings this to True navigates backwards duing frames"
                            " retreival. Default is False"
                        ),
                        "type": ["boolean", "null"],
                    },
                },
                "type": "object",
            }
        },
        "properties": {
            "batch_size": {
                "default": 500,
                "description": "Max number of images to be returned. Used only if scroll_id is not provided.",
                "type": "integer",
            },
            "clean_subfields": {
                "default": False,
                "description": (
                    "If set to Truethen both frame toplevel fields and subfields are cleaned according to the schema."
                    " Otherwise only top level fields"
                ),
                "type": "boolean",
            },
            "dataview": {"description": "Dataview ID", "type": "string"},
            "flow_control": {
                "$ref": "#/definitions/flow_control",
                "description": (
                    "Frames retreival that allows eiter one-directional navigation (the default) or bidirectional"
                ),
            },
            "force_scroll_id": {"description": "", "type": "boolean"},
            "node": {
                "description": (
                    "Node number. This provides support for multi-node experiments running multiple workers executing"
                    " the same experiment on multiple processes or machines"
                ),
                "type": "integer",
            },
            "projection": {
                "description": (
                    "Used to select which parts of the frame will be returned. Each string represents a\n              "
                    "      field or sub-field (using dot-separated notation). In order to specify a specific array"
                    " element,\n                    use array index as a field name. To specify all array elements, use"
                    " '*'."
                ),
                "items": {"type": "string"},
                "type": "array",
            },
            "random_seed": {
                "description": "Optional random seed used for frame selection. If not provided, one will be generated.",
                "type": "integer",
            },
            "remove_none_values": {
                "default": False,
                "description": "If set to Truethen none values are removed from frames (except for metadata)",
                "type": "boolean",
            },
            "reset_scroll": {"description": "", "type": "boolean"},
            "scroll_id": {
                "description": "Scroll session id for getting the next batch of images",
                "type": "string",
            },
        },
        "required": ["dataview"],
        "type": "object",
    }

    def __init__(
        self,
        dataview,
        scroll_id=None,
        batch_size=500,
        reset_scroll=None,
        force_scroll_id=None,
        flow_control=None,
        random_seed=None,
        node=None,
        projection=None,
        remove_none_values=False,
        clean_subfields=False,
        **kwargs
    ):
        super(GetNextForDataviewIdRequest, self).__init__(**kwargs)
        self.dataview = dataview
        self.scroll_id = scroll_id
        self.batch_size = batch_size
        self.reset_scroll = reset_scroll
        self.force_scroll_id = force_scroll_id
        self.flow_control = flow_control
        self.random_seed = random_seed
        self.node = node
        self.projection = projection
        self.remove_none_values = remove_none_values
        self.clean_subfields = clean_subfields

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

    @schema_property("batch_size")
    def batch_size(self):
        return self._property_batch_size

    @batch_size.setter
    def batch_size(self, value):
        if value is None:
            self._property_batch_size = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "batch_size", six.integer_types)
        self._property_batch_size = value

    @schema_property("reset_scroll")
    def reset_scroll(self):
        return self._property_reset_scroll

    @reset_scroll.setter
    def reset_scroll(self, value):
        if value is None:
            self._property_reset_scroll = None
            return

        self.assert_isinstance(value, "reset_scroll", (bool,))
        self._property_reset_scroll = value

    @schema_property("force_scroll_id")
    def force_scroll_id(self):
        return self._property_force_scroll_id

    @force_scroll_id.setter
    def force_scroll_id(self, value):
        if value is None:
            self._property_force_scroll_id = None
            return

        self.assert_isinstance(value, "force_scroll_id", (bool,))
        self._property_force_scroll_id = value

    @schema_property("flow_control")
    def flow_control(self):
        return self._property_flow_control

    @flow_control.setter
    def flow_control(self, value):
        if value is None:
            self._property_flow_control = None
            return
        if isinstance(value, dict):
            value = FlowControl.from_dict(value)
        else:
            self.assert_isinstance(value, "flow_control", FlowControl)
        self._property_flow_control = value

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

    @schema_property("node")
    def node(self):
        return self._property_node

    @node.setter
    def node(self, value):
        if value is None:
            self._property_node = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "node", six.integer_types)
        self._property_node = value

    @schema_property("projection")
    def projection(self):
        return self._property_projection

    @projection.setter
    def projection(self, value):
        if value is None:
            self._property_projection = None
            return

        self.assert_isinstance(value, "projection", (list, tuple))

        self.assert_isinstance(value, "projection", six.string_types, is_array=True)
        self._property_projection = value

    @schema_property("remove_none_values")
    def remove_none_values(self):
        return self._property_remove_none_values

    @remove_none_values.setter
    def remove_none_values(self, value):
        if value is None:
            self._property_remove_none_values = None
            return

        self.assert_isinstance(value, "remove_none_values", (bool,))
        self._property_remove_none_values = value

    @schema_property("clean_subfields")
    def clean_subfields(self):
        return self._property_clean_subfields

    @clean_subfields.setter
    def clean_subfields(self, value):
        if value is None:
            self._property_clean_subfields = None
            return

        self.assert_isinstance(value, "clean_subfields", (bool,))
        self._property_clean_subfields = value


class GetNextForDataviewIdResponse(Response):
    """
    Response of frames.get_next_for_dataview_id endpoint.

    :param frames: Frames list
    :type frames: Sequence[Frame]
    :param frames_returned: Number of frames returned
    :type frames_returned: int
    :param scroll_state: JSON object representing the scroll state
    :type scroll_state: dict
    :param scroll_id: Scroll session id to be provided in order to get the next
        batch of images
    :type scroll_id: str
    :param roi_stats: Json object containing the count per labels in frames, e.g. {
        'background': 312, 'boat': 2, 'bus': 4, 'car': 2, }
    :type roi_stats: dict
    :param eof: When 'frames' is empty, represents whether there are no more frames
        left. If "false", client can retry the operation.
    :type eof: bool
    :param random_seed: Random seed used for frame selection
    :type random_seed: int
    """

    _service = "frames"
    _action = "get_next_for_dataview_id"
    _version = "2.23"

    _schema = {
        "definitions": {
            "augmentation": {
                "properties": {
                    "arguments": {
                        "additionalProperties": True,
                        "description": "Arguments dictionary, passed to custom augmentations.",
                        "type": ["object", "null"],
                    },
                    "cls": {
                        "description": "Augmentation class (see global definitions)",
                        "type": ["string", "null"],
                    },
                    "params": {
                        "description": (
                            "Transform parameters, an array ot 3 randomly generated values. Fixed values are passed in"
                            " case of affine reflect augmentation."
                        ),
                        "items": {"type": "number"},
                        "type": ["array", "null"],
                    },
                    "strength": {
                        "description": "Transform strength. Required for pixel transforms.",
                        "type": ["number", "null"],
                    },
                    "trans_mat": {
                        "description": "Transform matrix (list of lists). Required for affine transforms.",
                        "items": {"items": {"type": "number"}, "type": "array"},
                        "type": ["array", "null"],
                    },
                    "type": {
                        "description": "Augmentation type (see global definitions)",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "dataset_version": {
                "properties": {
                    "id": {"description": "Dataset id", "type": ["string", "null"]},
                    "version": {
                        "description": "Dataset version id",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "frame": {
                "properties": {
                    "augmentation": {
                        "description": "List of augmentations",
                        "items": {"$ref": "#/definitions/augmentation"},
                        "type": ["array", "null"],
                    },
                    "blob": {
                        "description": "Raw data (blob) for the frame",
                        "type": ["string", "null"],
                    },
                    "context_id": {
                        "description": (
                            "Context ID. Used for the default frames sorting. If not set then it is filled from the uri"
                            " of the first source."
                        ),
                        "type": ["string", "null"],
                    },
                    "dataset": {
                        "description": "Frame's dataset version",
                        "oneOf": [
                            {"$ref": "#/definitions/dataset_version"},
                            {"type": "null"},
                        ],
                    },
                    "id": {"description": "Frame id", "type": ["string", "null"]},
                    "is_key_frame": {
                        "description": "Is this a key frame (only applicable in frames who'se src is a video)",
                        "type": ["boolean", "null"],
                    },
                    "key_frame": {
                        "description": "ID of the key frame that this frame belongs to",
                        "type": ["string", "null"],
                    },
                    "label_rule_counts": {
                        "additionalProperties": True,
                        "description": "The number of matched roi per lable rule",
                        "type": ["object", "null"],
                    },
                    "labels_size": {
                        "description": "Number of labels returned",
                        "type": ["integer", "null"],
                    },
                    "meta": {
                        "additionalProperties": True,
                        "description": (
                            "Additional metadata dictionary for the frame. Please note that using this field"
                            " effectively defines a schema (dictionary structure and types used as values) - frames"
                            " within the same dataset cannot use conflicting schemas for this field (see documentation"
                            " for more details)."
                        ),
                        "type": ["object", "null"],
                    },
                    "meta_blob": {
                        "additionalProperties": True,
                        "description": (
                            "Non searchable metadata dictionary for the frame. The fields in this object cannot be"
                            " searched by and are not added to the frame schema"
                        ),
                        "type": ["object", "null"],
                    },
                    "new_ver": {
                        "description": "Newer version of this frame, if asked to merge",
                        "oneOf": [{"$ref": "#/definitions/frame"}, {"type": "null"}],
                    },
                    "rois": {
                        "description": "Frame regions of interest",
                        "items": {"$ref": "#/definitions/roi"},
                        "type": ["array", "null"],
                    },
                    "rule_name": {
                        "description": (
                            "Name of the filtering rule according to which this frame was provided (if applicable)"
                        ),
                        "type": ["string", "null"],
                    },
                    "saved": {
                        "description": "Last time frame was saved (timestamp)",
                        "type": ["integer", "null"],
                    },
                    "saved_in_version": {
                        "description": "Last version this frame was saved in (version ID)",
                        "type": ["string", "null"],
                    },
                    "sources": {
                        "description": "Sources of this frame",
                        "items": {"$ref": "#/definitions/source"},
                        "type": ["array", "null"],
                    },
                    "timestamp": {
                        "description": (
                            "Frame's offset in milliseconds, used primarily for video content. Used for the default"
                            " frames sorting as the secondary key (with the primary key being 'context_id'). For"
                            " images, this value should typically be 0. If not set, value is filled from the timestamp"
                            " of the first source. We recommend using this field only in cases concerning the default"
                            " sorting behavior."
                        ),
                        "type": ["integer", "null"],
                    },
                    "updated": {
                        "description": "Last time frame was saved (timestamp)",
                        "type": ["integer", "null"],
                    },
                    "updated_in_version": {
                        "description": "Last version this frame was updated in (version ID)",
                        "type": ["string", "null"],
                    },
                    "video_gop": {
                        "description": (
                            "Video encoding GOP value for the source of this frame. Only valid for video frames"
                        ),
                        "type": ["number", "null"],
                    },
                },
                "type": "object",
            },
            "mask": {
                "properties": {
                    "content_type": {
                        "description": "Content type (e.g. 'image/jpeg', 'image/png')",
                        "type": ["string", "null"],
                    },
                    "height": {
                        "description": "Height in pixels",
                        "type": ["integer", "null"],
                    },
                    "id": {
                        "description": "unique ID (in this frame)",
                        "type": ["string", "null"],
                    },
                    "timestamp": {
                        "default": 0,
                        "description": (
                            "Timestamp in the source data (for video content. for images, this value should be 0)"
                        ),
                        "type": ["integer", "null"],
                    },
                    "uri": {"description": "Data URI", "type": ["string", "null"]},
                    "width": {
                        "description": "Width in pixels",
                        "type": ["integer", "null"],
                    },
                },
                "type": "object",
            },
            "preview": {
                "properties": {
                    "content_type": {
                        "description": "Content type (e.g. 'image/jpeg', 'image/png')",
                        "type": ["string", "null"],
                    },
                    "height": {
                        "description": "Height in pixels",
                        "type": ["integer", "null"],
                    },
                    "timestamp": {
                        "default": 0,
                        "description": (
                            "Timestamp in the source data (for video content. for images, this value should be 0)"
                        ),
                        "type": ["integer", "null"],
                    },
                    "uri": {"description": "Data URI", "type": ["string", "null"]},
                    "width": {
                        "description": "Width in pixels",
                        "type": ["integer", "null"],
                    },
                },
                "type": "object",
            },
            "roi": {
                "properties": {
                    "area": {
                        "description": "ROI area (not used)",
                        "type": ["integer", "null"],
                    },
                    "confidence": {
                        "description": "ROI confidence",
                        "type": ["number", "null"],
                    },
                    "id": {"description": "ROI id", "type": ["string", "null"]},
                    "label": {
                        "description": "ROI labels",
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                    "label_num": {
                        "description": (
                            "Label number according to the specified labels mapping Used only when ROI is returned as"
                            " part of a task's frame."
                        ),
                        "type": ["integer", "null"],
                    },
                    "mask": {
                        "description": "Mask info for this ROI",
                        "oneOf": [{"$ref": "#/definitions/roi_mask"}, {"type": "null"}],
                    },
                    "meta": {
                        "additionalProperties": True,
                        "description": "Additional metadata dictionary for the roi",
                        "type": ["object", "null"],
                    },
                    "poly": {
                        "description": "ROI polygon (x0, y0, ..., xn, yn)",
                        "items": {"type": "number"},
                        "type": ["array", "null"],
                    },
                    "sources": {
                        "description": "Sources that this ROI belongs to",
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            },
            "roi_mask": {
                "properties": {
                    "id": {"description": "Mask ID", "type": "string"},
                    "value": {
                        "description": "Mask value",
                        "items": {"type": "integer"},
                        "type": "array",
                    },
                },
                "required": ["id", "value"],
                "type": "object",
            },
            "source": {
                "properties": {
                    "content_type": {
                        "description": "Content type (e.g. 'image/jpeg', 'image/png')",
                        "type": ["string", "null"],
                    },
                    "height": {
                        "description": "Height in pixels",
                        "type": ["integer", "null"],
                    },
                    "id": {
                        "description": "unique ID (in this frame)",
                        "type": ["string", "null"],
                    },
                    "masks": {
                        "items": {"$ref": "#/definitions/mask"},
                        "type": ["array", "null"],
                    },
                    "meta": {
                        "additionalProperties": True,
                        "description": "Additional metadata dictionary for the source",
                        "type": ["object", "null"],
                    },
                    "preview": {
                        "oneOf": [{"$ref": "#/definitions/preview"}, {"type": "null"}]
                    },
                    "timestamp": {
                        "default": 0,
                        "description": (
                            "Timestamp in the source data (for video content. for images, this value should be 0)"
                        ),
                        "type": ["integer", "null"],
                    },
                    "uri": {"description": "Data URI", "type": ["string", "null"]},
                    "width": {
                        "description": "Width in pixels",
                        "type": ["integer", "null"],
                    },
                },
                "type": "object",
            },
        },
        "properties": {
            "eof": {
                "description": (
                    "When 'frames' is empty, represents whether there are no more frames left. "
                    'If "false",\n'
                    "                client can retry the "
                    "operation."
                ),
                "type": ["boolean", "null"],
            },
            "frames": {
                "description": "Frames list",
                "items": {"$ref": "#/definitions/frame"},
                "type": ["array", "null"],
            },
            "frames_returned": {
                "description": "Number of frames returned",
                "type": ["integer", "null"],
            },
            "random_seed": {
                "description": "Random seed used for frame selection",
                "type": ["integer", "null"],
            },
            "roi_stats": {
                "additionalProperties": {"type": "integer"},
                "description": (
                    "Json object containing the count per labels in frames, e.g.\n                {\n                  "
                    "  'background': 312,\n                    'boat': 2,\n                    'bus': 4,\n             "
                    "       'car': 2,\n                }"
                ),
                "type": ["object", "null"],
            },
            "scroll_id": {
                "description": "Scroll session id to be provided in order to get the next batch of images",
                "type": ["string", "null"],
            },
            "scroll_state": {
                "additionalProperties": True,
                "description": "JSON object representing the scroll state",
                "type": ["object", "null"],
            },
        },
        "type": "object",
    }

    def __init__(
        self,
        frames=None,
        frames_returned=None,
        scroll_state=None,
        scroll_id=None,
        roi_stats=None,
        eof=None,
        random_seed=None,
        **kwargs
    ):
        super(GetNextForDataviewIdResponse, self).__init__(**kwargs)
        self.frames = frames
        self.frames_returned = frames_returned
        self.scroll_state = scroll_state
        self.scroll_id = scroll_id
        self.roi_stats = roi_stats
        self.eof = eof
        self.random_seed = random_seed

    @schema_property("frames")
    def frames(self):
        return self._property_frames

    @frames.setter
    def frames(self, value):
        if value is None:
            self._property_frames = None
            return

        self.assert_isinstance(value, "frames", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [Frame.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "frames", Frame, is_array=True)
        self._property_frames = value

    @schema_property("frames_returned")
    def frames_returned(self):
        return self._property_frames_returned

    @frames_returned.setter
    def frames_returned(self, value):
        if value is None:
            self._property_frames_returned = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "frames_returned", six.integer_types)
        self._property_frames_returned = value

    @schema_property("scroll_state")
    def scroll_state(self):
        return self._property_scroll_state

    @scroll_state.setter
    def scroll_state(self, value):
        if value is None:
            self._property_scroll_state = None
            return

        self.assert_isinstance(value, "scroll_state", (dict,))
        self._property_scroll_state = value

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

    @schema_property("roi_stats")
    def roi_stats(self):
        return self._property_roi_stats

    @roi_stats.setter
    def roi_stats(self, value):
        if value is None:
            self._property_roi_stats = None
            return

        self.assert_isinstance(value, "roi_stats", (dict,))
        self._property_roi_stats = value

    @schema_property("eof")
    def eof(self):
        return self._property_eof

    @eof.setter
    def eof(self, value):
        if value is None:
            self._property_eof = None
            return

        self.assert_isinstance(value, "eof", (bool,))
        self._property_eof = value

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


class GetSnippetsForDataviewRequest(Request):
    """
    Return first frame per unique URI for the given dataview specification. Note: 'count_range' option for label rules is not supported and does not affect the returned snippets

    :param dataview: Dataview specification
    :type dataview: Dataview
    :param page_size: The amount of snippets to return for the page. Cannot be
        changed after the first call (after the paging session is created). default=50,
        Optional. To change the page size use 'paging_id'=0 that will reset the paging
        session.
    :type page_size: int
    :param page: The page to return. default=0, Optional
    :type page: int
    :param paging_id: Paging session id for getting the next page of frames
    :type paging_id: str
    :param projection: Used to select which parts of the frame will be returned.
        Each string represents a field or sub-field (using dot-separated notation). In
        order to specify a specific array element, use array index as a field name. To
        specify all array elements, use '*'.
    :type projection: Sequence[str]
    :param aggregate_on_context_id: If set to Truethen one frame is returned per
        unique context_id. If set to Falseall frames are retuned. If not set then
        defaults to the server configured value
    :type aggregate_on_context_id: bool
    """

    _service = "frames"
    _action = "get_snippets_for_dataview"
    _version = "2.23"
    _schema = {
        "definitions": {
            "dataview": {
                "properties": {
                    "augmentation": {
                        "description": "Augmentation parameters. Only for training and testing tasks.",
                        "oneOf": [
                            {"$ref": "#/definitions/dv_augmentation"},
                            {"type": "null"},
                        ],
                    },
                    "filters": {
                        "description": "List of FilterRule ('OR' relationship)",
                        "items": {"$ref": "#/definitions/filter_rule"},
                        "type": ["array", "null"],
                    },
                    "iteration": {
                        "description": "Iteration parameters. Not applicable for register (import) tasks.",
                        "oneOf": [
                            {"$ref": "#/definitions/iteration"},
                            {"type": "null"},
                        ],
                    },
                    "labels_enumeration": {
                        "additionalProperties": {"type": "integer"},
                        "description": (
                            "Labels enumerations, specifies numbers to be assigned to ROI labels when getting frames"
                        ),
                        "type": ["object", "null"],
                    },
                    "mapping": {
                        "description": "Mapping parameters",
                        "oneOf": [{"$ref": "#/definitions/mapping"}, {"type": "null"}],
                    },
                    "output_rois": {
                        "description": (
                            "'all_in_frame' - all rois for a frame are returned\n\n'only_filtered' - only rois which"
                            " led this frame to be selected\n\n'frame_per_roi' - single roi per frame. Frame can be"
                            " returned multiple times with a different roi each time.\n\nNote: this should be used for"
                            " Training tasks only\n\nNote: frame_per_roi implies that only filtered rois will be"
                            " returned\n                "
                        ),
                        "oneOf": [
                            {"$ref": "#/definitions/output_rois_enum"},
                            {"type": "null"},
                        ],
                    },
                    "versions": {
                        "description": "View dataset versions",
                        "items": {"$ref": "#/definitions/view_entry"},
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            },
            "dv_augmentation": {
                "properties": {
                    "crop_around_rois": {
                        "description": "Crop image data around all frame ROIs",
                        "type": ["boolean", "null"],
                    },
                    "sets": {
                        "description": "List of augmentation sets",
                        "items": {"$ref": "#/definitions/dv_augmentation_set"},
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            },
            "dv_augmentation_set": {
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
            "view_entry": {
                "properties": {
                    "dataset": {
                        "description": "Existing Dataset id",
                        "type": ["string", "null"],
                    },
                    "merge_with": {
                        "description": "Version ID to merge with",
                        "type": ["string", "null"],
                    },
                    "version": {
                        "description": "Version id of a version belonging to the dataset",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
        },
        "properties": {
            "aggregate_on_context_id": {
                "description": (
                    "If set to Truethen one frame is returned per unique context_id. If set to Falseall frames are"
                    " retuned. If not set then defaults to the server configured value"
                ),
                "type": "boolean",
            },
            "dataview": {
                "$ref": "#/definitions/dataview",
                "description": "Dataview specification",
            },
            "page": {
                "default": 0,
                "description": "The page to return. default=0, Optional",
                "type": "integer",
            },
            "page_size": {
                "default": 50,
                "description": (
                    "The amount of snippets to return for the page. Cannot be changed after the first call (after the"
                    " paging session is created). default=50, Optional. To change the page size use 'paging_id'=0 that"
                    " will reset the paging session."
                ),
                "type": "integer",
            },
            "paging_id": {
                "description": "Paging session id for getting the next page of frames",
                "type": "string",
            },
            "projection": {
                "description": (
                    "Used to select which parts of the frame will be returned. Each string represents a\n              "
                    "      field or sub-field (using dot-separated notation). In order to specify a specific array"
                    " element,\n                    use array index as a field name. To specify all array elements, use"
                    " '*'."
                ),
                "items": {"type": "string"},
                "type": "array",
            },
        },
        "required": ["dataview"],
        "type": "object",
    }

    def __init__(
        self,
        dataview,
        page_size=50,
        page=0,
        paging_id=None,
        projection=None,
        aggregate_on_context_id=None,
        **kwargs
    ):
        super(GetSnippetsForDataviewRequest, self).__init__(**kwargs)
        self.dataview = dataview
        self.page_size = page_size
        self.page = page
        self.paging_id = paging_id
        self.projection = projection
        self.aggregate_on_context_id = aggregate_on_context_id

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

    @schema_property("paging_id")
    def paging_id(self):
        return self._property_paging_id

    @paging_id.setter
    def paging_id(self, value):
        if value is None:
            self._property_paging_id = None
            return

        self.assert_isinstance(value, "paging_id", six.string_types)
        self._property_paging_id = value

    @schema_property("projection")
    def projection(self):
        return self._property_projection

    @projection.setter
    def projection(self, value):
        if value is None:
            self._property_projection = None
            return

        self.assert_isinstance(value, "projection", (list, tuple))

        self.assert_isinstance(value, "projection", six.string_types, is_array=True)
        self._property_projection = value

    @schema_property("aggregate_on_context_id")
    def aggregate_on_context_id(self):
        return self._property_aggregate_on_context_id

    @aggregate_on_context_id.setter
    def aggregate_on_context_id(self, value):
        if value is None:
            self._property_aggregate_on_context_id = None
            return

        self.assert_isinstance(value, "aggregate_on_context_id", (bool,))
        self._property_aggregate_on_context_id = value


class GetSnippetsForDataviewResponse(Response):
    """
    Response of frames.get_snippets_for_dataview endpoint.

    :param frames: List of frames for the requested page. The amount of frames
        returned is not guaranteed to be equal to the requested page size.
    :type frames: Sequence[Snippet]
    :param frames_total: The total number of first frames per unique URI
    :type frames_total: int
    :param pages_total: The total number of pages
    :type pages_total: int
    :param page: The currently requested page
    :type page: int
    :param paging_id: Paging session id to be provided in order to get the next
        page of frames
    :type paging_id: str
    :param total_in_versions: The total number of snippets for the dataview
        versions (without applying the dataview filters)
    :type total_in_versions: int
    :param versions_updated: The list of versions whose frames were updated since
        the creation of the paging iterator. If a version was updated after the
        iteration was started you may not receive all the updated snippets. To make
        sure that you see all the snippets after the update please reset the paging id
        (this may result in a different total amount of pages for the same page size).
    :type versions_updated: Sequence[str]
    """

    _service = "frames"
    _action = "get_snippets_for_dataview"
    _version = "2.23"

    _schema = {
        "definitions": {
            "augmentation": {
                "properties": {
                    "arguments": {
                        "additionalProperties": True,
                        "description": "Arguments dictionary, passed to custom augmentations.",
                        "type": ["object", "null"],
                    },
                    "cls": {
                        "description": "Augmentation class (see global definitions)",
                        "type": ["string", "null"],
                    },
                    "params": {
                        "description": (
                            "Transform parameters, an array ot 3 randomly generated values. Fixed values are passed in"
                            " case of affine reflect augmentation."
                        ),
                        "items": {"type": "number"},
                        "type": ["array", "null"],
                    },
                    "strength": {
                        "description": "Transform strength. Required for pixel transforms.",
                        "type": ["number", "null"],
                    },
                    "trans_mat": {
                        "description": "Transform matrix (list of lists). Required for affine transforms.",
                        "items": {"items": {"type": "number"}, "type": "array"},
                        "type": ["array", "null"],
                    },
                    "type": {
                        "description": "Augmentation type (see global definitions)",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "dataset_version": {
                "properties": {
                    "id": {"description": "Dataset id", "type": ["string", "null"]},
                    "version": {
                        "description": "Dataset version id",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "frame": {
                "properties": {
                    "augmentation": {
                        "description": "List of augmentations",
                        "items": {"$ref": "#/definitions/augmentation"},
                        "type": ["array", "null"],
                    },
                    "blob": {
                        "description": "Raw data (blob) for the frame",
                        "type": ["string", "null"],
                    },
                    "context_id": {
                        "description": (
                            "Context ID. Used for the default frames sorting. If not set then it is filled from the uri"
                            " of the first source."
                        ),
                        "type": ["string", "null"],
                    },
                    "dataset": {
                        "description": "Frame's dataset version",
                        "oneOf": [
                            {"$ref": "#/definitions/dataset_version"},
                            {"type": "null"},
                        ],
                    },
                    "id": {"description": "Frame id", "type": ["string", "null"]},
                    "is_key_frame": {
                        "description": "Is this a key frame (only applicable in frames who'se src is a video)",
                        "type": ["boolean", "null"],
                    },
                    "key_frame": {
                        "description": "ID of the key frame that this frame belongs to",
                        "type": ["string", "null"],
                    },
                    "label_rule_counts": {
                        "additionalProperties": True,
                        "description": "The number of matched roi per lable rule",
                        "type": ["object", "null"],
                    },
                    "labels_size": {
                        "description": "Number of labels returned",
                        "type": ["integer", "null"],
                    },
                    "meta": {
                        "additionalProperties": True,
                        "description": (
                            "Additional metadata dictionary for the frame. Please note that using this field"
                            " effectively defines a schema (dictionary structure and types used as values) - frames"
                            " within the same dataset cannot use conflicting schemas for this field (see documentation"
                            " for more details)."
                        ),
                        "type": ["object", "null"],
                    },
                    "meta_blob": {
                        "additionalProperties": True,
                        "description": (
                            "Non searchable metadata dictionary for the frame. The fields in this object cannot be"
                            " searched by and are not added to the frame schema"
                        ),
                        "type": ["object", "null"],
                    },
                    "new_ver": {
                        "description": "Newer version of this frame, if asked to merge",
                        "oneOf": [{"$ref": "#/definitions/frame"}, {"type": "null"}],
                    },
                    "rois": {
                        "description": "Frame regions of interest",
                        "items": {"$ref": "#/definitions/roi"},
                        "type": ["array", "null"],
                    },
                    "rule_name": {
                        "description": (
                            "Name of the filtering rule according to which this frame was provided (if applicable)"
                        ),
                        "type": ["string", "null"],
                    },
                    "saved": {
                        "description": "Last time frame was saved (timestamp)",
                        "type": ["integer", "null"],
                    },
                    "saved_in_version": {
                        "description": "Last version this frame was saved in (version ID)",
                        "type": ["string", "null"],
                    },
                    "sources": {
                        "description": "Sources of this frame",
                        "items": {"$ref": "#/definitions/source"},
                        "type": ["array", "null"],
                    },
                    "timestamp": {
                        "description": (
                            "Frame's offset in milliseconds, used primarily for video content. Used for the default"
                            " frames sorting as the secondary key (with the primary key being 'context_id'). For"
                            " images, this value should typically be 0. If not set, value is filled from the timestamp"
                            " of the first source. We recommend using this field only in cases concerning the default"
                            " sorting behavior."
                        ),
                        "type": ["integer", "null"],
                    },
                    "updated": {
                        "description": "Last time frame was saved (timestamp)",
                        "type": ["integer", "null"],
                    },
                    "updated_in_version": {
                        "description": "Last version this frame was updated in (version ID)",
                        "type": ["string", "null"],
                    },
                    "video_gop": {
                        "description": (
                            "Video encoding GOP value for the source of this frame. Only valid for video frames"
                        ),
                        "type": ["number", "null"],
                    },
                },
                "type": "object",
            },
            "mask": {
                "properties": {
                    "content_type": {
                        "description": "Content type (e.g. 'image/jpeg', 'image/png')",
                        "type": ["string", "null"],
                    },
                    "height": {
                        "description": "Height in pixels",
                        "type": ["integer", "null"],
                    },
                    "id": {
                        "description": "unique ID (in this frame)",
                        "type": ["string", "null"],
                    },
                    "timestamp": {
                        "default": 0,
                        "description": (
                            "Timestamp in the source data (for video content. for images, this value should be 0)"
                        ),
                        "type": ["integer", "null"],
                    },
                    "uri": {"description": "Data URI", "type": ["string", "null"]},
                    "width": {
                        "description": "Width in pixels",
                        "type": ["integer", "null"],
                    },
                },
                "type": "object",
            },
            "preview": {
                "properties": {
                    "content_type": {
                        "description": "Content type (e.g. 'image/jpeg', 'image/png')",
                        "type": ["string", "null"],
                    },
                    "height": {
                        "description": "Height in pixels",
                        "type": ["integer", "null"],
                    },
                    "timestamp": {
                        "default": 0,
                        "description": (
                            "Timestamp in the source data (for video content. for images, this value should be 0)"
                        ),
                        "type": ["integer", "null"],
                    },
                    "uri": {"description": "Data URI", "type": ["string", "null"]},
                    "width": {
                        "description": "Width in pixels",
                        "type": ["integer", "null"],
                    },
                },
                "type": "object",
            },
            "roi": {
                "properties": {
                    "area": {
                        "description": "ROI area (not used)",
                        "type": ["integer", "null"],
                    },
                    "confidence": {
                        "description": "ROI confidence",
                        "type": ["number", "null"],
                    },
                    "id": {"description": "ROI id", "type": ["string", "null"]},
                    "label": {
                        "description": "ROI labels",
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                    "label_num": {
                        "description": (
                            "Label number according to the specified labels mapping Used only when ROI is returned as"
                            " part of a task's frame."
                        ),
                        "type": ["integer", "null"],
                    },
                    "mask": {
                        "description": "Mask info for this ROI",
                        "oneOf": [{"$ref": "#/definitions/roi_mask"}, {"type": "null"}],
                    },
                    "meta": {
                        "additionalProperties": True,
                        "description": "Additional metadata dictionary for the roi",
                        "type": ["object", "null"],
                    },
                    "poly": {
                        "description": "ROI polygon (x0, y0, ..., xn, yn)",
                        "items": {"type": "number"},
                        "type": ["array", "null"],
                    },
                    "sources": {
                        "description": "Sources that this ROI belongs to",
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            },
            "roi_mask": {
                "properties": {
                    "id": {"description": "Mask ID", "type": "string"},
                    "value": {
                        "description": "Mask value",
                        "items": {"type": "integer"},
                        "type": "array",
                    },
                },
                "required": ["id", "value"],
                "type": "object",
            },
            "snippet": {
                "properties": {
                    "augmentation": {
                        "description": "List of augmentations",
                        "items": {"$ref": "#/definitions/augmentation"},
                        "type": ["array", "null"],
                    },
                    "blob": {
                        "description": "Raw data (blob) for the frame",
                        "type": ["string", "null"],
                    },
                    "context_id": {
                        "description": (
                            "Context ID. Used for the default frames sorting. If not set then it is filled from the uri"
                            " of the first source."
                        ),
                        "type": ["string", "null"],
                    },
                    "dataset": {
                        "description": "Frame's dataset version",
                        "oneOf": [
                            {"$ref": "#/definitions/dataset_version"},
                            {"type": "null"},
                        ],
                    },
                    "id": {"description": "Frame id", "type": ["string", "null"]},
                    "is_key_frame": {
                        "description": "Is this a key frame (only applicable in frames who'se src is a video)",
                        "type": ["boolean", "null"],
                    },
                    "key_frame": {
                        "description": "ID of the key frame that this frame belongs to",
                        "type": ["string", "null"],
                    },
                    "label_rule_counts": {
                        "additionalProperties": True,
                        "description": "The number of matched roi per lable rule",
                        "type": ["object", "null"],
                    },
                    "labels_size": {
                        "description": "Number of labels returned",
                        "type": ["integer", "null"],
                    },
                    "meta": {
                        "additionalProperties": True,
                        "description": (
                            "Additional metadata dictionary for the frame. Please note that using this field"
                            " effectively defines a schema (dictionary structure and types used as values) - frames"
                            " within the same dataset cannot use conflicting schemas for this field (see documentation"
                            " for more details)."
                        ),
                        "type": ["object", "null"],
                    },
                    "meta_blob": {
                        "additionalProperties": True,
                        "description": (
                            "Non searchable metadata dictionary for the frame. The fields in this object cannot be"
                            " searched by and are not added to the frame schema"
                        ),
                        "type": ["object", "null"],
                    },
                    "new_ver": {
                        "description": "Newer version of this frame, if asked to merge",
                        "oneOf": [{"$ref": "#/definitions/frame"}, {"type": "null"}],
                    },
                    "num_frames": {
                        "description": "Number of frames represented by this snippet",
                        "type": ["integer", "null"],
                    },
                    "rois": {
                        "description": "Frame regions of interest",
                        "items": {"$ref": "#/definitions/roi"},
                        "type": ["array", "null"],
                    },
                    "rule_name": {
                        "description": (
                            "Name of the filtering rule according to which this frame was provided (if applicable)"
                        ),
                        "type": ["string", "null"],
                    },
                    "saved": {
                        "description": "Last time frame was saved (timestamp)",
                        "type": ["integer", "null"],
                    },
                    "saved_in_version": {
                        "description": "Last version this frame was saved in (version ID)",
                        "type": ["string", "null"],
                    },
                    "sources": {
                        "description": "Sources of this frame",
                        "items": {"$ref": "#/definitions/source"},
                        "type": ["array", "null"],
                    },
                    "timestamp": {
                        "description": (
                            "Frame's offset in milliseconds, used primarily for video content. Used for the default"
                            " frames sorting as the secondary key (with the primary key being 'context_id'). For"
                            " images, this value should typically be 0. If not set, value is filled from the timestamp"
                            " of the first source. We recommend using this field only in cases concerning the default"
                            " sorting behavior."
                        ),
                        "type": ["integer", "null"],
                    },
                    "updated": {
                        "description": "Last time frame was saved (timestamp)",
                        "type": ["integer", "null"],
                    },
                    "updated_in_version": {
                        "description": "Last version this frame was updated in (version ID)",
                        "type": ["string", "null"],
                    },
                    "video_gop": {
                        "description": (
                            "Video encoding GOP value for the source of this frame. Only valid for video frames"
                        ),
                        "type": ["number", "null"],
                    },
                },
                "type": "object",
            },
            "source": {
                "properties": {
                    "content_type": {
                        "description": "Content type (e.g. 'image/jpeg', 'image/png')",
                        "type": ["string", "null"],
                    },
                    "height": {
                        "description": "Height in pixels",
                        "type": ["integer", "null"],
                    },
                    "id": {
                        "description": "unique ID (in this frame)",
                        "type": ["string", "null"],
                    },
                    "masks": {
                        "items": {"$ref": "#/definitions/mask"},
                        "type": ["array", "null"],
                    },
                    "meta": {
                        "additionalProperties": True,
                        "description": "Additional metadata dictionary for the source",
                        "type": ["object", "null"],
                    },
                    "preview": {
                        "oneOf": [{"$ref": "#/definitions/preview"}, {"type": "null"}]
                    },
                    "timestamp": {
                        "default": 0,
                        "description": (
                            "Timestamp in the source data (for video content. for images, this value should be 0)"
                        ),
                        "type": ["integer", "null"],
                    },
                    "uri": {"description": "Data URI", "type": ["string", "null"]},
                    "width": {
                        "description": "Width in pixels",
                        "type": ["integer", "null"],
                    },
                },
                "type": "object",
            },
        },
        "properties": {
            "frames": {
                "description": (
                    "List of frames for the requested page. The amount of frames returned is not guaranteed to be equal"
                    " to the requested page size."
                ),
                "items": {"$ref": "#/definitions/snippet"},
                "type": ["array", "null"],
            },
            "frames_total": {
                "description": "The total number of first frames per unique URI",
                "type": ["integer", "null"],
            },
            "page": {
                "description": "The currently requested page",
                "type": ["integer", "null"],
            },
            "pages_total": {
                "description": "The total number of pages",
                "type": ["integer", "null"],
            },
            "paging_id": {
                "description": "Paging session id to be provided in order to get the next page of frames",
                "type": ["string", "null"],
            },
            "total_in_versions": {
                "description": (
                    "The total number of snippets for the dataview versions (without applying the dataview filters)"
                ),
                "type": ["integer", "null"],
            },
            "versions_updated": {
                "description": (
                    "The list of versions whose frames were updated since the creation of the paging iterator. If a"
                    " version was updated after the iteration was started you may not receive all the updated snippets."
                    " To make sure that you see all the snippets after the update please reset the paging id (this may"
                    " result in a different total amount of pages for the same page size)."
                ),
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
        },
        "type": "object",
    }

    def __init__(
        self,
        frames=None,
        frames_total=None,
        pages_total=None,
        page=None,
        paging_id=None,
        total_in_versions=None,
        versions_updated=None,
        **kwargs
    ):
        super(GetSnippetsForDataviewResponse, self).__init__(**kwargs)
        self.frames = frames
        self.frames_total = frames_total
        self.pages_total = pages_total
        self.page = page
        self.paging_id = paging_id
        self.total_in_versions = total_in_versions
        self.versions_updated = versions_updated

    @schema_property("frames")
    def frames(self):
        return self._property_frames

    @frames.setter
    def frames(self, value):
        if value is None:
            self._property_frames = None
            return

        self.assert_isinstance(value, "frames", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [Snippet.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "frames", Snippet, is_array=True)
        self._property_frames = value

    @schema_property("frames_total")
    def frames_total(self):
        return self._property_frames_total

    @frames_total.setter
    def frames_total(self, value):
        if value is None:
            self._property_frames_total = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "frames_total", six.integer_types)
        self._property_frames_total = value

    @schema_property("pages_total")
    def pages_total(self):
        return self._property_pages_total

    @pages_total.setter
    def pages_total(self, value):
        if value is None:
            self._property_pages_total = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "pages_total", six.integer_types)
        self._property_pages_total = value

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

    @schema_property("paging_id")
    def paging_id(self):
        return self._property_paging_id

    @paging_id.setter
    def paging_id(self, value):
        if value is None:
            self._property_paging_id = None
            return

        self.assert_isinstance(value, "paging_id", six.string_types)
        self._property_paging_id = value

    @schema_property("total_in_versions")
    def total_in_versions(self):
        return self._property_total_in_versions

    @total_in_versions.setter
    def total_in_versions(self, value):
        if value is None:
            self._property_total_in_versions = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "total_in_versions", six.integer_types)
        self._property_total_in_versions = value

    @schema_property("versions_updated")
    def versions_updated(self):
        return self._property_versions_updated

    @versions_updated.setter
    def versions_updated(self, value):
        if value is None:
            self._property_versions_updated = None
            return

        self.assert_isinstance(value, "versions_updated", (list, tuple))

        self.assert_isinstance(
            value, "versions_updated", six.string_types, is_array=True
        )
        self._property_versions_updated = value


class GetSnippetsForDataview2Request(Request):
    """
    Return first frame per unique URI for the given dataview specification.
    Note: 'count_range' option for label rules is not supported and does not affect the returned snippets

    :param dataview: Dataview specification
    :type dataview: Dataview
    :param size: The amount of snippets to return.
    :type size: int
    :param search_after: For getting the next portion of snippets should be set to
        the value returned from the previous call. To get snippets from the beginning
        should be set to null or empty string
    :type search_after: str
    :param projection: Used to select which parts of the frame will be returned.
        Each string represents a field or sub-field (using dot-separated notation). In
        order to specify a specific array element, use array index as a field name. To
        specify all array elements, use '*'.
    :type projection: Sequence[str]
    :param aggregation: Specifies whether the returned frames should be aggregated.
        If not passed then 'aggregate_on_context_id' setting is consulted
    :type aggregation: dict
    :param order_by: The list of fields to sort on.
    :type order_by: Sequence[dict]
    """

    _service = "frames"
    _action = "get_snippets_for_dataview2"
    _version = "2.23"
    _schema = {
        "definitions": {
            "dataview": {
                "properties": {
                    "augmentation": {
                        "description": "Augmentation parameters. Only for training and testing tasks.",
                        "oneOf": [
                            {"$ref": "#/definitions/dv_augmentation"},
                            {"type": "null"},
                        ],
                    },
                    "filters": {
                        "description": "List of FilterRule ('OR' relationship)",
                        "items": {"$ref": "#/definitions/filter_rule"},
                        "type": ["array", "null"],
                    },
                    "iteration": {
                        "description": "Iteration parameters. Not applicable for register (import) tasks.",
                        "oneOf": [
                            {"$ref": "#/definitions/iteration"},
                            {"type": "null"},
                        ],
                    },
                    "labels_enumeration": {
                        "additionalProperties": {"type": "integer"},
                        "description": (
                            "Labels enumerations, specifies numbers to be assigned to ROI labels when getting frames"
                        ),
                        "type": ["object", "null"],
                    },
                    "mapping": {
                        "description": "Mapping parameters",
                        "oneOf": [{"$ref": "#/definitions/mapping"}, {"type": "null"}],
                    },
                    "output_rois": {
                        "description": (
                            "'all_in_frame' - all rois for a frame are returned\n\n'only_filtered' - only rois which"
                            " led this frame to be selected\n\n'frame_per_roi' - single roi per frame. Frame can be"
                            " returned multiple times with a different roi each time.\n\nNote: this should be used for"
                            " Training tasks only\n\nNote: frame_per_roi implies that only filtered rois will be"
                            " returned\n                "
                        ),
                        "oneOf": [
                            {"$ref": "#/definitions/output_rois_enum"},
                            {"type": "null"},
                        ],
                    },
                    "versions": {
                        "description": "View dataset versions",
                        "items": {"$ref": "#/definitions/view_entry"},
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            },
            "dv_augmentation": {
                "properties": {
                    "crop_around_rois": {
                        "description": "Crop image data around all frame ROIs",
                        "type": ["boolean", "null"],
                    },
                    "sets": {
                        "description": "List of augmentation sets",
                        "items": {"$ref": "#/definitions/dv_augmentation_set"},
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            },
            "dv_augmentation_set": {
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
            "view_entry": {
                "properties": {
                    "dataset": {
                        "description": "Existing Dataset id",
                        "type": ["string", "null"],
                    },
                    "merge_with": {
                        "description": "Version ID to merge with",
                        "type": ["string", "null"],
                    },
                    "version": {
                        "description": "Version id of a version belonging to the dataset",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
        },
        "properties": {
            "aggregation": {
                "description": (
                    "Specifies whether the returned frames should be aggregated. If not passed then"
                    " 'aggregate_on_context_id' setting is consulted"
                ),
                "properties": {
                    "aggregate": {
                        "description": "If set to Truethen the returned frames are aggregated on the provided fields",
                        "type": "boolean",
                    },
                    "fields": {
                        "description": (
                            "The list of the fields to aggragate on. Only if the aggregate parameter is set to True"
                        ),
                        "items": {"type": "string"},
                        "type": "array",
                    },
                },
                "required": ["aggregate"],
                "type": "object",
            },
            "dataview": {
                "$ref": "#/definitions/dataview",
                "description": "Dataview specification",
            },
            "order_by": {
                "description": "The list of fields to sort on.",
                "items": {
                    "properties": {
                        "field": {
                            "description": "The name of the field",
                            "type": "string",
                        },
                        "order": {
                            "default": "asc",
                            "description": "The order of the sorting",
                            "enum": ["asc", "desc"],
                            "type": "string",
                        },
                    },
                    "type": "object",
                },
                "type": "array",
            },
            "projection": {
                "description": (
                    "Used to select which parts of the frame will be returned. Each string represents a\n              "
                    "      field or sub-field (using dot-separated notation). In order to specify a specific array"
                    " element,\n                    use array index as a field name. To specify all array elements, use"
                    " '*'."
                ),
                "items": {"type": "string"},
                "type": "array",
            },
            "search_after": {
                "description": (
                    "For getting the next portion of snippets should be set to the value returned from the previous"
                    " call. To get snippets from the beginning should be set to null or empty string"
                ),
                "type": "string",
            },
            "size": {
                "default": 50,
                "description": "The amount of snippets to return.",
                "type": "integer",
            },
        },
        "required": ["dataview"],
        "type": "object",
    }

    def __init__(
        self,
        dataview,
        size=50,
        search_after=None,
        projection=None,
        aggregation=None,
        order_by=None,
        **kwargs
    ):
        super(GetSnippetsForDataview2Request, self).__init__(**kwargs)
        self.dataview = dataview
        self.size = size
        self.search_after = search_after
        self.projection = projection
        self.aggregation = aggregation
        self.order_by = order_by

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

    @schema_property("search_after")
    def search_after(self):
        return self._property_search_after

    @search_after.setter
    def search_after(self, value):
        if value is None:
            self._property_search_after = None
            return

        self.assert_isinstance(value, "search_after", six.string_types)
        self._property_search_after = value

    @schema_property("projection")
    def projection(self):
        return self._property_projection

    @projection.setter
    def projection(self, value):
        if value is None:
            self._property_projection = None
            return

        self.assert_isinstance(value, "projection", (list, tuple))

        self.assert_isinstance(value, "projection", six.string_types, is_array=True)
        self._property_projection = value

    @schema_property("aggregation")
    def aggregation(self):
        return self._property_aggregation

    @aggregation.setter
    def aggregation(self, value):
        if value is None:
            self._property_aggregation = None
            return

        self.assert_isinstance(value, "aggregation", (dict,))
        self._property_aggregation = value

    @schema_property("order_by")
    def order_by(self):
        return self._property_order_by

    @order_by.setter
    def order_by(self, value):
        if value is None:
            self._property_order_by = None
            return

        self.assert_isinstance(value, "order_by", (list, tuple))

        self.assert_isinstance(value, "order_by", (dict,), is_array=True)
        self._property_order_by = value


class GetSnippetsForDataview2Response(Response):
    """
    Response of frames.get_snippets_for_dataview2 endpoint.

    :param frames: List of frames for the requested page. The amount of frames
        returned is not guaranteed to be equal to the requested page size.
    :type frames: Sequence[Snippet]
    :param frames_total: The total number of first frames per unique URI
    :type frames_total: int
    :param search_after: The key for querying next batch of frames
    :type search_after: str
    :param total_in_versions: The total number of snippets for the dataview
        versions (without applying the dataview filters)
    :type total_in_versions: int
    """

    _service = "frames"
    _action = "get_snippets_for_dataview2"
    _version = "2.23"

    _schema = {
        "definitions": {
            "augmentation": {
                "properties": {
                    "arguments": {
                        "additionalProperties": True,
                        "description": "Arguments dictionary, passed to custom augmentations.",
                        "type": ["object", "null"],
                    },
                    "cls": {
                        "description": "Augmentation class (see global definitions)",
                        "type": ["string", "null"],
                    },
                    "params": {
                        "description": (
                            "Transform parameters, an array ot 3 randomly generated values. Fixed values are passed in"
                            " case of affine reflect augmentation."
                        ),
                        "items": {"type": "number"},
                        "type": ["array", "null"],
                    },
                    "strength": {
                        "description": "Transform strength. Required for pixel transforms.",
                        "type": ["number", "null"],
                    },
                    "trans_mat": {
                        "description": "Transform matrix (list of lists). Required for affine transforms.",
                        "items": {"items": {"type": "number"}, "type": "array"},
                        "type": ["array", "null"],
                    },
                    "type": {
                        "description": "Augmentation type (see global definitions)",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "dataset_version": {
                "properties": {
                    "id": {"description": "Dataset id", "type": ["string", "null"]},
                    "version": {
                        "description": "Dataset version id",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "frame": {
                "properties": {
                    "augmentation": {
                        "description": "List of augmentations",
                        "items": {"$ref": "#/definitions/augmentation"},
                        "type": ["array", "null"],
                    },
                    "blob": {
                        "description": "Raw data (blob) for the frame",
                        "type": ["string", "null"],
                    },
                    "context_id": {
                        "description": (
                            "Context ID. Used for the default frames sorting. If not set then it is filled from the uri"
                            " of the first source."
                        ),
                        "type": ["string", "null"],
                    },
                    "dataset": {
                        "description": "Frame's dataset version",
                        "oneOf": [
                            {"$ref": "#/definitions/dataset_version"},
                            {"type": "null"},
                        ],
                    },
                    "id": {"description": "Frame id", "type": ["string", "null"]},
                    "is_key_frame": {
                        "description": "Is this a key frame (only applicable in frames who'se src is a video)",
                        "type": ["boolean", "null"],
                    },
                    "key_frame": {
                        "description": "ID of the key frame that this frame belongs to",
                        "type": ["string", "null"],
                    },
                    "label_rule_counts": {
                        "additionalProperties": True,
                        "description": "The number of matched roi per lable rule",
                        "type": ["object", "null"],
                    },
                    "labels_size": {
                        "description": "Number of labels returned",
                        "type": ["integer", "null"],
                    },
                    "meta": {
                        "additionalProperties": True,
                        "description": (
                            "Additional metadata dictionary for the frame. Please note that using this field"
                            " effectively defines a schema (dictionary structure and types used as values) - frames"
                            " within the same dataset cannot use conflicting schemas for this field (see documentation"
                            " for more details)."
                        ),
                        "type": ["object", "null"],
                    },
                    "meta_blob": {
                        "additionalProperties": True,
                        "description": (
                            "Non searchable metadata dictionary for the frame. The fields in this object cannot be"
                            " searched by and are not added to the frame schema"
                        ),
                        "type": ["object", "null"],
                    },
                    "new_ver": {
                        "description": "Newer version of this frame, if asked to merge",
                        "oneOf": [{"$ref": "#/definitions/frame"}, {"type": "null"}],
                    },
                    "rois": {
                        "description": "Frame regions of interest",
                        "items": {"$ref": "#/definitions/roi"},
                        "type": ["array", "null"],
                    },
                    "rule_name": {
                        "description": (
                            "Name of the filtering rule according to which this frame was provided (if applicable)"
                        ),
                        "type": ["string", "null"],
                    },
                    "saved": {
                        "description": "Last time frame was saved (timestamp)",
                        "type": ["integer", "null"],
                    },
                    "saved_in_version": {
                        "description": "Last version this frame was saved in (version ID)",
                        "type": ["string", "null"],
                    },
                    "sources": {
                        "description": "Sources of this frame",
                        "items": {"$ref": "#/definitions/source"},
                        "type": ["array", "null"],
                    },
                    "timestamp": {
                        "description": (
                            "Frame's offset in milliseconds, used primarily for video content. Used for the default"
                            " frames sorting as the secondary key (with the primary key being 'context_id'). For"
                            " images, this value should typically be 0. If not set, value is filled from the timestamp"
                            " of the first source. We recommend using this field only in cases concerning the default"
                            " sorting behavior."
                        ),
                        "type": ["integer", "null"],
                    },
                    "updated": {
                        "description": "Last time frame was saved (timestamp)",
                        "type": ["integer", "null"],
                    },
                    "updated_in_version": {
                        "description": "Last version this frame was updated in (version ID)",
                        "type": ["string", "null"],
                    },
                    "video_gop": {
                        "description": (
                            "Video encoding GOP value for the source of this frame. Only valid for video frames"
                        ),
                        "type": ["number", "null"],
                    },
                },
                "type": "object",
            },
            "mask": {
                "properties": {
                    "content_type": {
                        "description": "Content type (e.g. 'image/jpeg', 'image/png')",
                        "type": ["string", "null"],
                    },
                    "height": {
                        "description": "Height in pixels",
                        "type": ["integer", "null"],
                    },
                    "id": {
                        "description": "unique ID (in this frame)",
                        "type": ["string", "null"],
                    },
                    "timestamp": {
                        "default": 0,
                        "description": (
                            "Timestamp in the source data (for video content. for images, this value should be 0)"
                        ),
                        "type": ["integer", "null"],
                    },
                    "uri": {"description": "Data URI", "type": ["string", "null"]},
                    "width": {
                        "description": "Width in pixels",
                        "type": ["integer", "null"],
                    },
                },
                "type": "object",
            },
            "preview": {
                "properties": {
                    "content_type": {
                        "description": "Content type (e.g. 'image/jpeg', 'image/png')",
                        "type": ["string", "null"],
                    },
                    "height": {
                        "description": "Height in pixels",
                        "type": ["integer", "null"],
                    },
                    "timestamp": {
                        "default": 0,
                        "description": (
                            "Timestamp in the source data (for video content. for images, this value should be 0)"
                        ),
                        "type": ["integer", "null"],
                    },
                    "uri": {"description": "Data URI", "type": ["string", "null"]},
                    "width": {
                        "description": "Width in pixels",
                        "type": ["integer", "null"],
                    },
                },
                "type": "object",
            },
            "roi": {
                "properties": {
                    "area": {
                        "description": "ROI area (not used)",
                        "type": ["integer", "null"],
                    },
                    "confidence": {
                        "description": "ROI confidence",
                        "type": ["number", "null"],
                    },
                    "id": {"description": "ROI id", "type": ["string", "null"]},
                    "label": {
                        "description": "ROI labels",
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                    "label_num": {
                        "description": (
                            "Label number according to the specified labels mapping Used only when ROI is returned as"
                            " part of a task's frame."
                        ),
                        "type": ["integer", "null"],
                    },
                    "mask": {
                        "description": "Mask info for this ROI",
                        "oneOf": [{"$ref": "#/definitions/roi_mask"}, {"type": "null"}],
                    },
                    "meta": {
                        "additionalProperties": True,
                        "description": "Additional metadata dictionary for the roi",
                        "type": ["object", "null"],
                    },
                    "poly": {
                        "description": "ROI polygon (x0, y0, ..., xn, yn)",
                        "items": {"type": "number"},
                        "type": ["array", "null"],
                    },
                    "sources": {
                        "description": "Sources that this ROI belongs to",
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            },
            "roi_mask": {
                "properties": {
                    "id": {"description": "Mask ID", "type": "string"},
                    "value": {
                        "description": "Mask value",
                        "items": {"type": "integer"},
                        "type": "array",
                    },
                },
                "required": ["id", "value"],
                "type": "object",
            },
            "snippet": {
                "properties": {
                    "augmentation": {
                        "description": "List of augmentations",
                        "items": {"$ref": "#/definitions/augmentation"},
                        "type": ["array", "null"],
                    },
                    "blob": {
                        "description": "Raw data (blob) for the frame",
                        "type": ["string", "null"],
                    },
                    "context_id": {
                        "description": (
                            "Context ID. Used for the default frames sorting. If not set then it is filled from the uri"
                            " of the first source."
                        ),
                        "type": ["string", "null"],
                    },
                    "dataset": {
                        "description": "Frame's dataset version",
                        "oneOf": [
                            {"$ref": "#/definitions/dataset_version"},
                            {"type": "null"},
                        ],
                    },
                    "id": {"description": "Frame id", "type": ["string", "null"]},
                    "is_key_frame": {
                        "description": "Is this a key frame (only applicable in frames who'se src is a video)",
                        "type": ["boolean", "null"],
                    },
                    "key_frame": {
                        "description": "ID of the key frame that this frame belongs to",
                        "type": ["string", "null"],
                    },
                    "label_rule_counts": {
                        "additionalProperties": True,
                        "description": "The number of matched roi per lable rule",
                        "type": ["object", "null"],
                    },
                    "labels_size": {
                        "description": "Number of labels returned",
                        "type": ["integer", "null"],
                    },
                    "meta": {
                        "additionalProperties": True,
                        "description": (
                            "Additional metadata dictionary for the frame. Please note that using this field"
                            " effectively defines a schema (dictionary structure and types used as values) - frames"
                            " within the same dataset cannot use conflicting schemas for this field (see documentation"
                            " for more details)."
                        ),
                        "type": ["object", "null"],
                    },
                    "meta_blob": {
                        "additionalProperties": True,
                        "description": (
                            "Non searchable metadata dictionary for the frame. The fields in this object cannot be"
                            " searched by and are not added to the frame schema"
                        ),
                        "type": ["object", "null"],
                    },
                    "new_ver": {
                        "description": "Newer version of this frame, if asked to merge",
                        "oneOf": [{"$ref": "#/definitions/frame"}, {"type": "null"}],
                    },
                    "num_frames": {
                        "description": "Number of frames represented by this snippet",
                        "type": ["integer", "null"],
                    },
                    "rois": {
                        "description": "Frame regions of interest",
                        "items": {"$ref": "#/definitions/roi"},
                        "type": ["array", "null"],
                    },
                    "rule_name": {
                        "description": (
                            "Name of the filtering rule according to which this frame was provided (if applicable)"
                        ),
                        "type": ["string", "null"],
                    },
                    "saved": {
                        "description": "Last time frame was saved (timestamp)",
                        "type": ["integer", "null"],
                    },
                    "saved_in_version": {
                        "description": "Last version this frame was saved in (version ID)",
                        "type": ["string", "null"],
                    },
                    "sources": {
                        "description": "Sources of this frame",
                        "items": {"$ref": "#/definitions/source"},
                        "type": ["array", "null"],
                    },
                    "timestamp": {
                        "description": (
                            "Frame's offset in milliseconds, used primarily for video content. Used for the default"
                            " frames sorting as the secondary key (with the primary key being 'context_id'). For"
                            " images, this value should typically be 0. If not set, value is filled from the timestamp"
                            " of the first source. We recommend using this field only in cases concerning the default"
                            " sorting behavior."
                        ),
                        "type": ["integer", "null"],
                    },
                    "updated": {
                        "description": "Last time frame was saved (timestamp)",
                        "type": ["integer", "null"],
                    },
                    "updated_in_version": {
                        "description": "Last version this frame was updated in (version ID)",
                        "type": ["string", "null"],
                    },
                    "video_gop": {
                        "description": (
                            "Video encoding GOP value for the source of this frame. Only valid for video frames"
                        ),
                        "type": ["number", "null"],
                    },
                },
                "type": "object",
            },
            "source": {
                "properties": {
                    "content_type": {
                        "description": "Content type (e.g. 'image/jpeg', 'image/png')",
                        "type": ["string", "null"],
                    },
                    "height": {
                        "description": "Height in pixels",
                        "type": ["integer", "null"],
                    },
                    "id": {
                        "description": "unique ID (in this frame)",
                        "type": ["string", "null"],
                    },
                    "masks": {
                        "items": {"$ref": "#/definitions/mask"},
                        "type": ["array", "null"],
                    },
                    "meta": {
                        "additionalProperties": True,
                        "description": "Additional metadata dictionary for the source",
                        "type": ["object", "null"],
                    },
                    "preview": {
                        "oneOf": [{"$ref": "#/definitions/preview"}, {"type": "null"}]
                    },
                    "timestamp": {
                        "default": 0,
                        "description": (
                            "Timestamp in the source data (for video content. for images, this value should be 0)"
                        ),
                        "type": ["integer", "null"],
                    },
                    "uri": {"description": "Data URI", "type": ["string", "null"]},
                    "width": {
                        "description": "Width in pixels",
                        "type": ["integer", "null"],
                    },
                },
                "type": "object",
            },
        },
        "properties": {
            "frames": {
                "description": (
                    "List of frames for the requested page. The amount of frames returned is not guaranteed to be equal"
                    " to the requested page size."
                ),
                "items": {"$ref": "#/definitions/snippet"},
                "type": ["array", "null"],
            },
            "frames_total": {
                "description": "The total number of first frames per unique URI",
                "type": ["integer", "null"],
            },
            "search_after": {
                "description": "The key for querying next batch of frames",
                "type": ["string", "null"],
            },
            "total_in_versions": {
                "description": (
                    "The total number of snippets for the dataview versions (without applying the dataview filters)"
                ),
                "type": ["integer", "null"],
            },
        },
        "type": "object",
    }

    def __init__(
        self,
        frames=None,
        frames_total=None,
        search_after=None,
        total_in_versions=None,
        **kwargs
    ):
        super(GetSnippetsForDataview2Response, self).__init__(**kwargs)
        self.frames = frames
        self.frames_total = frames_total
        self.search_after = search_after
        self.total_in_versions = total_in_versions

    @schema_property("frames")
    def frames(self):
        return self._property_frames

    @frames.setter
    def frames(self, value):
        if value is None:
            self._property_frames = None
            return

        self.assert_isinstance(value, "frames", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [Snippet.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "frames", Snippet, is_array=True)
        self._property_frames = value

    @schema_property("frames_total")
    def frames_total(self):
        return self._property_frames_total

    @frames_total.setter
    def frames_total(self, value):
        if value is None:
            self._property_frames_total = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "frames_total", six.integer_types)
        self._property_frames_total = value

    @schema_property("search_after")
    def search_after(self):
        return self._property_search_after

    @search_after.setter
    def search_after(self, value):
        if value is None:
            self._property_search_after = None
            return

        self.assert_isinstance(value, "search_after", six.string_types)
        self._property_search_after = value

    @schema_property("total_in_versions")
    def total_in_versions(self):
        return self._property_total_in_versions

    @total_in_versions.setter
    def total_in_versions(self, value):
        if value is None:
            self._property_total_in_versions = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "total_in_versions", six.integer_types)
        self._property_total_in_versions = value


class GetSnippetsQueryForDataviewRequest(Request):
    """
    Get Elasticsearch query that returns frames with unique URIs for the given dataview specification.
            The query is returned only for the clients that have kibana space set up.
            Note: 'count_range' option for label rules is not supported and not reflected in the query

    :param dataview: Dataview specification
    :type dataview: Dataview
    """

    _service = "frames"
    _action = "get_snippets_query_for_dataview"
    _version = "2.23"
    _schema = {
        "definitions": {
            "dataview": {
                "properties": {
                    "augmentation": {
                        "description": "Augmentation parameters. Only for training and testing tasks.",
                        "oneOf": [
                            {"$ref": "#/definitions/dv_augmentation"},
                            {"type": "null"},
                        ],
                    },
                    "filters": {
                        "description": "List of FilterRule ('OR' relationship)",
                        "items": {"$ref": "#/definitions/filter_rule"},
                        "type": ["array", "null"],
                    },
                    "iteration": {
                        "description": "Iteration parameters. Not applicable for register (import) tasks.",
                        "oneOf": [
                            {"$ref": "#/definitions/iteration"},
                            {"type": "null"},
                        ],
                    },
                    "labels_enumeration": {
                        "additionalProperties": {"type": "integer"},
                        "description": (
                            "Labels enumerations, specifies numbers to be assigned to ROI labels when getting frames"
                        ),
                        "type": ["object", "null"],
                    },
                    "mapping": {
                        "description": "Mapping parameters",
                        "oneOf": [{"$ref": "#/definitions/mapping"}, {"type": "null"}],
                    },
                    "output_rois": {
                        "description": (
                            "'all_in_frame' - all rois for a frame are returned\n\n'only_filtered' - only rois which"
                            " led this frame to be selected\n\n'frame_per_roi' - single roi per frame. Frame can be"
                            " returned multiple times with a different roi each time.\n\nNote: this should be used for"
                            " Training tasks only\n\nNote: frame_per_roi implies that only filtered rois will be"
                            " returned\n                "
                        ),
                        "oneOf": [
                            {"$ref": "#/definitions/output_rois_enum"},
                            {"type": "null"},
                        ],
                    },
                    "versions": {
                        "description": "View dataset versions",
                        "items": {"$ref": "#/definitions/view_entry"},
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            },
            "dv_augmentation": {
                "properties": {
                    "crop_around_rois": {
                        "description": "Crop image data around all frame ROIs",
                        "type": ["boolean", "null"],
                    },
                    "sets": {
                        "description": "List of augmentation sets",
                        "items": {"$ref": "#/definitions/dv_augmentation_set"},
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            },
            "dv_augmentation_set": {
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
            "view_entry": {
                "properties": {
                    "dataset": {
                        "description": "Existing Dataset id",
                        "type": ["string", "null"],
                    },
                    "merge_with": {
                        "description": "Version ID to merge with",
                        "type": ["string", "null"],
                    },
                    "version": {
                        "description": "Version id of a version belonging to the dataset",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
        },
        "properties": {
            "dataview": {
                "$ref": "#/definitions/dataview",
                "description": "Dataview specification",
            }
        },
        "required": ["dataview"],
    }

    def __init__(self, dataview, **kwargs):
        super(GetSnippetsQueryForDataviewRequest, self).__init__(**kwargs)
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


class GetSnippetsQueryForDataviewResponse(Response):
    """
    Response of frames.get_snippets_query_for_dataview endpoint.

    :param query: The Elasticsearch query filter that should bring the snippet
        frames according to the provided dataview
    :type query: dict
    :param kibana_link: The link to the Kibana discovery page with the dataview
        query
    :type kibana_link: str
    """

    _service = "frames"
    _action = "get_snippets_query_for_dataview"
    _version = "2.23"

    _schema = {
        "definitions": {},
        "properties": {
            "kibana_link": {
                "description": "The link to the Kibana discovery page with the dataview query",
                "type": ["string", "null"],
            },
            "query": {
                "description": (
                    "The Elasticsearch query filter that should bring the snippet frames "
                    "according to the provided dataview"
                ),
                "type": ["object", "null"],
            },
        },
    }

    def __init__(self, query=None, kibana_link=None, **kwargs):
        super(GetSnippetsQueryForDataviewResponse, self).__init__(**kwargs)
        self.query = query
        self.kibana_link = kibana_link

    @schema_property("query")
    def query(self):
        return self._property_query

    @query.setter
    def query(self, value):
        if value is None:
            self._property_query = None
            return

        self.assert_isinstance(value, "query", (dict,))
        self._property_query = value

    @schema_property("kibana_link")
    def kibana_link(self):
        return self._property_kibana_link

    @kibana_link.setter
    def kibana_link(self, value):
        if value is None:
            self._property_kibana_link = None
            return

        self.assert_isinstance(value, "kibana_link", six.string_types)
        self._property_kibana_link = value


class GetSourceIdsForDataviewRequest(Request):
    """
    Get unique sorce id that exist in the frames in the given dataview

    :param dataview: Dataview specification
    :type dataview: Dataview
    :param max_count: Number of source IDs to return. default=100, Optional
    :type max_count: int
    """

    _service = "frames"
    _action = "get_source_ids_for_dataview"
    _version = "2.23"
    _schema = {
        "definitions": {
            "dataview": {
                "properties": {
                    "augmentation": {
                        "description": "Augmentation parameters. Only for training and testing tasks.",
                        "oneOf": [
                            {"$ref": "#/definitions/dv_augmentation"},
                            {"type": "null"},
                        ],
                    },
                    "filters": {
                        "description": "List of FilterRule ('OR' relationship)",
                        "items": {"$ref": "#/definitions/filter_rule"},
                        "type": ["array", "null"],
                    },
                    "iteration": {
                        "description": "Iteration parameters. Not applicable for register (import) tasks.",
                        "oneOf": [
                            {"$ref": "#/definitions/iteration"},
                            {"type": "null"},
                        ],
                    },
                    "labels_enumeration": {
                        "additionalProperties": {"type": "integer"},
                        "description": (
                            "Labels enumerations, specifies numbers to be assigned to ROI labels when getting frames"
                        ),
                        "type": ["object", "null"],
                    },
                    "mapping": {
                        "description": "Mapping parameters",
                        "oneOf": [{"$ref": "#/definitions/mapping"}, {"type": "null"}],
                    },
                    "output_rois": {
                        "description": (
                            "'all_in_frame' - all rois for a frame are returned\n\n'only_filtered' - only rois which"
                            " led this frame to be selected\n\n'frame_per_roi' - single roi per frame. Frame can be"
                            " returned multiple times with a different roi each time.\n\nNote: this should be used for"
                            " Training tasks only\n\nNote: frame_per_roi implies that only filtered rois will be"
                            " returned\n                "
                        ),
                        "oneOf": [
                            {"$ref": "#/definitions/output_rois_enum"},
                            {"type": "null"},
                        ],
                    },
                    "versions": {
                        "description": "View dataset versions",
                        "items": {"$ref": "#/definitions/view_entry"},
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            },
            "dv_augmentation": {
                "properties": {
                    "crop_around_rois": {
                        "description": "Crop image data around all frame ROIs",
                        "type": ["boolean", "null"],
                    },
                    "sets": {
                        "description": "List of augmentation sets",
                        "items": {"$ref": "#/definitions/dv_augmentation_set"},
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            },
            "dv_augmentation_set": {
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
            "view_entry": {
                "properties": {
                    "dataset": {
                        "description": "Existing Dataset id",
                        "type": ["string", "null"],
                    },
                    "merge_with": {
                        "description": "Version ID to merge with",
                        "type": ["string", "null"],
                    },
                    "version": {
                        "description": "Version id of a version belonging to the dataset",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
        },
        "properties": {
            "dataview": {
                "$ref": "#/definitions/dataview",
                "description": "Dataview specification",
            },
            "max_count": {
                "default": 100,
                "description": "Number of source IDs to return. default=100, Optional",
                "type": "integer",
            },
        },
        "required": ["dataview"],
    }

    def __init__(self, dataview, max_count=100, **kwargs):
        super(GetSourceIdsForDataviewRequest, self).__init__(**kwargs)
        self.dataview = dataview
        self.max_count = max_count

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

    @schema_property("max_count")
    def max_count(self):
        return self._property_max_count

    @max_count.setter
    def max_count(self, value):
        if value is None:
            self._property_max_count = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "max_count", six.integer_types)
        self._property_max_count = value


class GetSourceIdsForDataviewResponse(Response):
    """
    Response of frames.get_source_ids_for_dataview endpoint.

    :param source_ids: Unique source ids for the dataset version
    :type source_ids: Sequence[str]
    """

    _service = "frames"
    _action = "get_source_ids_for_dataview"
    _version = "2.23"

    _schema = {
        "definitions": {},
        "properties": {
            "source_ids": {
                "description": "Unique source ids for the dataset version",
                "items": {"type": "string"},
                "type": ["array", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, source_ids=None, **kwargs):
        super(GetSourceIdsForDataviewResponse, self).__init__(**kwargs)
        self.source_ids = source_ids

    @schema_property("source_ids")
    def source_ids(self):
        return self._property_source_ids

    @source_ids.setter
    def source_ids(self, value):
        if value is None:
            self._property_source_ids = None
            return

        self.assert_isinstance(value, "source_ids", (list, tuple))

        self.assert_isinstance(value, "source_ids", six.string_types, is_array=True)
        self._property_source_ids = value


class GetWithProjectionRequest(Request):
    """
    For each passed query return projected frames matcing the conditions.
    One frame is returned per unique query field value

    :param versions: Dataset versions
    :type versions: Sequence[ViewEntry]
    :param query: The list of field queries
    :type query: Sequence[dict]
    """

    _service = "frames"
    _action = "get_with_projection"
    _version = "2.23"
    _schema = {
        "definitions": {
            "view_entry": {
                "properties": {
                    "dataset": {
                        "description": "Existing Dataset id",
                        "type": ["string", "null"],
                    },
                    "merge_with": {
                        "description": "Version ID to merge with",
                        "type": ["string", "null"],
                    },
                    "version": {
                        "description": "Version id of a version belonging to the dataset",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            }
        },
        "properties": {
            "query": {
                "description": "The list of field queries",
                "items": {
                    "properties": {
                        "field": {
                            "description": "The field name",
                            "type": "string",
                        },
                        "projection": {
                            "description": "List of projected fields",
                            "items": {"type": "string"},
                            "type": "array",
                        },
                        "values": {
                            "description": "The allowed field values",
                            "items": {"type": "string"},
                            "type": "array",
                        },
                    },
                    "type": "object",
                },
                "type": ["array", "null"],
            },
            "versions": {
                "description": "Dataset versions",
                "items": {"$ref": "#/definitions/view_entry"},
                "type": ["array", "null"],
            },
        },
        "required": ["dataview"],
        "type": "object",
    }

    def __init__(self, versions=None, query=None, **kwargs):
        super(GetWithProjectionRequest, self).__init__(**kwargs)
        self.versions = versions
        self.query = query

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
                ViewEntry.from_dict(v) if isinstance(v, dict) else v for v in value
            ]
        else:
            self.assert_isinstance(value, "versions", ViewEntry, is_array=True)
        self._property_versions = value

    @schema_property("query")
    def query(self):
        return self._property_query

    @query.setter
    def query(self, value):
        if value is None:
            self._property_query = None
            return

        self.assert_isinstance(value, "query", (list, tuple))

        self.assert_isinstance(value, "query", (dict,), is_array=True)
        self._property_query = value


class GetWithProjectionResponse(Response):
    """
    Response of frames.get_with_projection endpoint.

    :param frames: Projected frames per rule indexed by the rule key
    :type frames: dict
    """

    _service = "frames"
    _action = "get_with_projection"
    _version = "2.23"

    _schema = {
        "definitions": {},
        "properties": {
            "frames": {
                "additionalProperties": True,
                "description": "Projected frames per rule indexed by the rule key",
                "type": ["object", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, frames=None, **kwargs):
        super(GetWithProjectionResponse, self).__init__(**kwargs)
        self.frames = frames

    @schema_property("frames")
    def frames(self):
        return self._property_frames

    @frames.setter
    def frames(self, value):
        if value is None:
            self._property_frames = None
            return

        self.assert_isinstance(value, "frames", (dict,))
        self._property_frames = value


class PrepareDownloadForDataviewRequest(Request):
    """
    Prepare request data for frames download. Intended for allowing the user to send the data using
    a POST request (since a query string in a GET request cannot accomodate large generic data structures), and use
    the resulting call's ID as a handle for calling the `download_for_dataview` endpoint using a GET method.


    :param dataview: Dataview specification
    :type dataview: Dataview
    :param random_seed: Optional random seed used for frame selection. If not
        provided, one will be generated.
    :type random_seed: int
    :param node: Node number. This provides support for multi-node experiments
        running multiple workers executing the same experiment on multiple processes or
        machines
    :type node: int
    :param projection: Used to select which parts of the frame will be returned.
        Each string represents a field or sub-field (using dot-separated notation). In
        order to specify a specific array element, use array index as a field name. To
        specify all array elements, use '*'.
    :type projection: Sequence[str]
    :param download_type: Download type. Determines the downloaded file's
        formatting and mime type.
    :type download_type: str
    :param remove_none_values: If set to Truethen none values are removed from
        frames (except for metadata)
    :type remove_none_values: bool
    :param clean_subfields: If set to Truethen both frame toplevel fields and
        subfields are cleaned according to the schema. Otherwise only top level fields
    :type clean_subfields: bool
    """

    _service = "frames"
    _action = "prepare_download_for_dataview"
    _version = "2.23"
    _schema = {
        "definitions": {
            "dataview": {
                "properties": {
                    "augmentation": {
                        "description": "Augmentation parameters. Only for training and testing tasks.",
                        "oneOf": [
                            {"$ref": "#/definitions/dv_augmentation"},
                            {"type": "null"},
                        ],
                    },
                    "filters": {
                        "description": "List of FilterRule ('OR' relationship)",
                        "items": {"$ref": "#/definitions/filter_rule"},
                        "type": ["array", "null"],
                    },
                    "iteration": {
                        "description": "Iteration parameters. Not applicable for register (import) tasks.",
                        "oneOf": [
                            {"$ref": "#/definitions/iteration"},
                            {"type": "null"},
                        ],
                    },
                    "labels_enumeration": {
                        "additionalProperties": {"type": "integer"},
                        "description": (
                            "Labels enumerations, specifies numbers to be assigned to ROI labels when getting frames"
                        ),
                        "type": ["object", "null"],
                    },
                    "mapping": {
                        "description": "Mapping parameters",
                        "oneOf": [{"$ref": "#/definitions/mapping"}, {"type": "null"}],
                    },
                    "output_rois": {
                        "description": (
                            "'all_in_frame' - all rois for a frame are returned\n\n'only_filtered' - only rois which"
                            " led this frame to be selected\n\n'frame_per_roi' - single roi per frame. Frame can be"
                            " returned multiple times with a different roi each time.\n\nNote: this should be used for"
                            " Training tasks only\n\nNote: frame_per_roi implies that only filtered rois will be"
                            " returned\n                "
                        ),
                        "oneOf": [
                            {"$ref": "#/definitions/output_rois_enum"},
                            {"type": "null"},
                        ],
                    },
                    "versions": {
                        "description": "View dataset versions",
                        "items": {"$ref": "#/definitions/view_entry"},
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            },
            "dv_augmentation": {
                "properties": {
                    "crop_around_rois": {
                        "description": "Crop image data around all frame ROIs",
                        "type": ["boolean", "null"],
                    },
                    "sets": {
                        "description": "List of augmentation sets",
                        "items": {"$ref": "#/definitions/dv_augmentation_set"},
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            },
            "dv_augmentation_set": {
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
            "view_entry": {
                "properties": {
                    "dataset": {
                        "description": "Existing Dataset id",
                        "type": ["string", "null"],
                    },
                    "merge_with": {
                        "description": "Version ID to merge with",
                        "type": ["string", "null"],
                    },
                    "version": {
                        "description": "Version id of a version belonging to the dataset",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
        },
        "properties": {
            "clean_subfields": {
                "default": False,
                "description": (
                    "If set to Truethen both frame toplevel fields and subfields are cleaned according to the schema."
                    " Otherwise only top level fields"
                ),
                "type": "boolean",
            },
            "dataview": {
                "$ref": "#/definitions/dataview",
                "description": "Dataview specification",
            },
            "download_type": {
                "default": "json",
                "description": "Download type. Determines the downloaded file's formatting and mime type.",
                "enum": ["json jsonlines"],
                "type": "string",
            },
            "node": {
                "description": (
                    "Node number. This provides support for multi-node experiments running multiple workers executing"
                    " the same experiment on multiple processes or machines"
                ),
                "type": "integer",
            },
            "projection": {
                "description": (
                    "Used to select which parts of the frame will be returned. Each string represents a\n              "
                    "      field or sub-field (using dot-separated notation). In order to specify a specific array"
                    " element,\n                    use array index as a field name. To specify all array elements, use"
                    " '*'."
                ),
                "items": {"type": "string"},
                "type": "array",
            },
            "random_seed": {
                "description": "Optional random seed used for frame selection. If not provided, one will be generated.",
                "type": "integer",
            },
            "remove_none_values": {
                "default": False,
                "description": "If set to Truethen none values are removed from frames (except for metadata)",
                "type": "boolean",
            },
        },
        "required": ["dataview"],
        "type": "object",
    }

    def __init__(
        self,
        dataview,
        random_seed=None,
        node=None,
        projection=None,
        download_type="json",
        remove_none_values=False,
        clean_subfields=False,
        **kwargs
    ):
        super(PrepareDownloadForDataviewRequest, self).__init__(**kwargs)
        self.dataview = dataview
        self.random_seed = random_seed
        self.node = node
        self.projection = projection
        self.download_type = download_type
        self.remove_none_values = remove_none_values
        self.clean_subfields = clean_subfields

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

    @schema_property("node")
    def node(self):
        return self._property_node

    @node.setter
    def node(self, value):
        if value is None:
            self._property_node = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "node", six.integer_types)
        self._property_node = value

    @schema_property("projection")
    def projection(self):
        return self._property_projection

    @projection.setter
    def projection(self, value):
        if value is None:
            self._property_projection = None
            return

        self.assert_isinstance(value, "projection", (list, tuple))

        self.assert_isinstance(value, "projection", six.string_types, is_array=True)
        self._property_projection = value

    @schema_property("download_type")
    def download_type(self):
        return self._property_download_type

    @download_type.setter
    def download_type(self, value):
        if value is None:
            self._property_download_type = None
            return

        self.assert_isinstance(value, "download_type", six.string_types)
        self._property_download_type = value

    @schema_property("remove_none_values")
    def remove_none_values(self):
        return self._property_remove_none_values

    @remove_none_values.setter
    def remove_none_values(self, value):
        if value is None:
            self._property_remove_none_values = None
            return

        self.assert_isinstance(value, "remove_none_values", (bool,))
        self._property_remove_none_values = value

    @schema_property("clean_subfields")
    def clean_subfields(self):
        return self._property_clean_subfields

    @clean_subfields.setter
    def clean_subfields(self, value):
        if value is None:
            self._property_clean_subfields = None
            return

        self.assert_isinstance(value, "clean_subfields", (bool,))
        self._property_clean_subfields = value


class PrepareDownloadForDataviewResponse(Response):
    """
    Response of frames.prepare_download_for_dataview endpoint.

    :param prepare_id: Prepare ID (use when calling `download_for_dataview`)
    :type prepare_id: str
    """

    _service = "frames"
    _action = "prepare_download_for_dataview"
    _version = "2.23"

    _schema = {
        "definitions": {},
        "properties": {
            "prepare_id": {
                "description": "Prepare ID (use when calling `download_for_dataview`)",
                "type": ["string", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, prepare_id=None, **kwargs):
        super(PrepareDownloadForDataviewResponse, self).__init__(**kwargs)
        self.prepare_id = prepare_id

    @schema_property("prepare_id")
    def prepare_id(self):
        return self._property_prepare_id

    @prepare_id.setter
    def prepare_id(self, value):
        if value is None:
            self._property_prepare_id = None
            return

        self.assert_isinstance(value, "prepare_id", six.string_types)
        self._property_prepare_id = value


response_mapping = {
    ClearGetNextStateRequest: ClearGetNextStateResponse,
    GetNextRequest: GetNextResponse,
    GetWithProjectionRequest: GetWithProjectionResponse,
    GetSnippetsForDataviewRequest: GetSnippetsForDataviewResponse,
    GetSnippetsForDataview2Request: GetSnippetsForDataview2Response,
    GetSourceIdsForDataviewRequest: GetSourceIdsForDataviewResponse,
    GetSnippetsQueryForDataviewRequest: GetSnippetsQueryForDataviewResponse,
    GetCountRequest: GetCountResponse,
    GetNextForDataviewAndContextIdRequest: GetNextForDataviewAndContextIdResponse,
    GetNextForDataviewRequest: GetNextForDataviewResponse,
    GetCountForDataviewRequest: GetCountForDataviewResponse,
    GetNextForDataviewIdRequest: GetNextForDataviewIdResponse,
    GetCountForDataviewIdRequest: GetCountForDataviewIdResponse,
    GetByIdRequest: GetByIdResponse,
    GetByIdsRequest: GetByIdsResponse,
    PrepareDownloadForDataviewRequest: PrepareDownloadForDataviewResponse,
    DownloadForDataviewRequest: DownloadForDataviewResponse,
}
