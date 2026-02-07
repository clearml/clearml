"""
datasets service

This service provides a management interface for the datasets managed by the system.
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


class VersionStatusEnum(StringEnum):
    draft = "draft"
    committing = "committing"
    committed = "committed"
    published = "published"


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
    :param label: ROI labels
    :type label: Sequence[str]
    :param poly: ROI polygon (x0, y0, ..., xn, yn)
    :type poly: Sequence[float]
    :param confidence: ROI confidence
    :type confidence: float
    :param meta: Additional metadata dictionary for the roi
    :type meta: dict
    :param sources: Source ID
    :type sources: Sequence[str]
    :param mask: Mask info for this ROI
    :type mask: RoiMask
    """

    _schema = {
        "properties": {
            "confidence": {"description": "ROI confidence", "type": "number"},
            "id": {"description": "ROI id", "type": ["string", "null"]},
            "label": {
                "description": "ROI labels",
                "items": {"type": "string"},
                "type": "array",
            },
            "mask": {
                "$ref": "#/definitions/roi_mask",
                "description": "Mask info for this ROI",
            },
            "meta": {
                "additionalProperties": True,
                "description": "Additional metadata dictionary for the roi",
                "type": "object",
            },
            "poly": {
                "description": "ROI polygon (x0, y0, ..., xn, yn)",
                "items": {"type": "number"},
                "type": "array",
            },
            "sources": {
                "description": "Source ID",
                "items": {"type": "string"},
                "type": "array",
            },
        },
        "required": ["label"],
        "type": "object",
    }

    def __init__(
        self,
        label,
        id=None,
        poly=None,
        confidence=None,
        meta=None,
        sources=None,
        mask=None,
        **kwargs
    ):
        super(Roi, self).__init__(**kwargs)
        self.id = id
        self.label = label
        self.poly = poly
        self.confidence = confidence
        self.meta = meta
        self.sources = sources
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
                "type": "string",
            },
            "height": {"description": "Height in pixels", "type": "integer"},
            "id": {"description": "unique ID (in this frame)", "type": "string"},
            "timestamp": {
                "default": 0,
                "description": "Timestamp in the source data (for video content. for images, this value should be 0)",
                "type": "integer",
            },
            "uri": {"description": "Data URI", "type": "string"},
            "width": {"description": "Width in pixels", "type": "integer"},
        },
        "required": ["id", "uri"],
        "type": "object",
    }

    def __init__(
        self, id, uri, content_type=None, width=None, height=None, timestamp=0, **kwargs
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
                "type": "string",
            },
            "height": {"description": "Height in pixels", "type": "integer"},
            "timestamp": {
                "default": 0,
                "description": "Timestamp in the source data (for video content. for images, this value should be 0)",
                "type": "integer",
            },
            "uri": {"description": "Data URI", "type": "string"},
            "width": {"description": "Width in pixels", "type": "integer"},
        },
        "required": ["uri"],
        "type": "object",
    }

    def __init__(
        self, uri, content_type=None, width=None, height=None, timestamp=0, **kwargs
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
    :param id: Source unique ID within this DatasetVersion
    :type id: str
    :param uri: Source data URI
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
                "type": "string",
            },
            "height": {"description": "Height in pixels", "type": "integer"},
            "id": {
                "description": "Source unique ID within this DatasetVersion",
                "type": "string",
            },
            "masks": {"items": {"$ref": "#/definitions/mask"}, "type": "array"},
            "meta": {
                "additionalProperties": True,
                "description": "Additional metadata dictionary for the source",
                "type": "object",
            },
            "preview": {"$ref": "#/definitions/preview"},
            "timestamp": {
                "default": 0,
                "description": "Timestamp in the source data (for video content. for images, this value should be 0)",
                "type": "integer",
            },
            "uri": {"description": "Source data URI", "type": "string"},
            "width": {"description": "Width in pixels", "type": "integer"},
        },
        "required": ["id", "uri"],
        "type": "object",
    }

    def __init__(
        self,
        id,
        uri,
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
    :param id: Frame id. Must be unique within the dataset's version. If already
        exists, will cause existing frame to be updated
    :type id: str
    :param context_id: Context ID. Used for the default frames sorting. If not set
        then it is filled from the uri of the first source.
    :type context_id: str
    :param timestamp: Frame's offset in milliseconds, used primarily for video
        content. Used for the default frames sorting as the secondary key (with the
        primary key being 'context_id'). For images, this value should typically be 0.
        If not set, value is filled from the timestamp of the first source. We
        recommend using this field only in cases concerning the default sorting
        behavior.
    :type timestamp: int
    :param rois: Frame regions of interest
    :type rois: Sequence[Roi]
    :param meta: Additional metadata dictionary for the frame. Please note that
        using this field effectively defines a schema (dictionary structure and types
        used as values) - frames within the same dataset cannot use conflicting schemas
        for this field (see documentation for more details).
    :type meta: dict
    :param meta_blob: Non searchable metadata dictionary for the frame. The fields
        in this object cannot be searched by and are not added to the frame schema
    :type meta_blob: dict
    :param blob: Raw data (blob) for the frame
    :type blob: str
    :param sources: Sources of this frame
    :type sources: Sequence[Source]
    """

    _schema = {
        "properties": {
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
            "id": {
                "description": (
                    "Frame id. Must be unique within the dataset's version. If already exists, "
                    "will cause existing frame to be updated"
                ),
                "type": ["string", "null"],
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
            "rois": {
                "description": "Frame regions of interest",
                "items": {"$ref": "#/definitions/roi"},
                "type": ["array", "null"],
            },
            "sources": {
                "description": "Sources of this frame",
                "items": {"$ref": "#/definitions/source"},
                "type": "array",
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
        },
        "required": ["sources"],
        "type": "object",
    }

    def __init__(
        self,
        sources,
        id=None,
        context_id=None,
        timestamp=None,
        rois=None,
        meta=None,
        meta_blob=None,
        blob=None,
        **kwargs
    ):
        super(Frame, self).__init__(**kwargs)
        self.id = id
        self.context_id = context_id
        self.timestamp = timestamp
        self.rois = rois
        self.meta = meta
        self.meta_blob = meta_blob
        self.blob = blob
        self.sources = sources

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


class StatCount(NonStrictDataModel):
    """
    :param count: Item name
    :type count: int
    :param name: Number of appearances
    :type name: str
    """

    _schema = {
        "properties": {
            "count": {"description": "Item name", "type": ["integer", "null"]},
            "name": {
                "description": "Number of appearances",
                "type": ["string", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, count=None, name=None, **kwargs):
        super(StatCount, self).__init__(**kwargs)
        self.count = count
        self.name = name

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


class Statistics(NonStrictDataModel):
    """
    :param content_types:
    :type content_types: Sequence[StatCount]
    :param labels:
    :type labels: Sequence[StatCount]
    :param frames:
    :type frames: Sequence[StatCount]
    """

    _schema = {
        "properties": {
            "content_types": {
                "items": {
                    "$ref": "#/definitions/stat_count",
                    "description": (
                        "List of content type counts for the version (e.g.\n                    'image/jpeg',"
                        " 'image/png', 'video/mp4')"
                    ),
                },
                "type": ["array", "null"],
            },
            "frames": {
                "items": {
                    "$ref": "#/definitions/stat_count",
                    "description": (
                        "List of frame counts, indicating the\n                    type of frames included in the "
                        "version (annotated/"
                    ),
                },
                "type": ["array", "null"],
            },
            "labels": {
                "items": {
                    "$ref": "#/definitions/stat_count",
                    "description": (
                        "List of labels' counts,\n                    indicating the categories included in the version"
                    ),
                },
                "type": ["array", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, content_types=None, labels=None, frames=None, **kwargs):
        super(Statistics, self).__init__(**kwargs)
        self.content_types = content_types
        self.labels = labels
        self.frames = frames

    @schema_property("content_types")
    def content_types(self):
        return self._property_content_types

    @content_types.setter
    def content_types(self, value):
        if value is None:
            self._property_content_types = None
            return

        self.assert_isinstance(value, "content_types", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [
                StatCount.from_dict(v) if isinstance(v, dict) else v for v in value
            ]
        else:
            self.assert_isinstance(value, "content_types", StatCount, is_array=True)
        self._property_content_types = value

    @schema_property("labels")
    def labels(self):
        return self._property_labels

    @labels.setter
    def labels(self, value):
        if value is None:
            self._property_labels = None
            return

        self.assert_isinstance(value, "labels", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [
                StatCount.from_dict(v) if isinstance(v, dict) else v for v in value
            ]
        else:
            self.assert_isinstance(value, "labels", StatCount, is_array=True)
        self._property_labels = value

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
            value = [
                StatCount.from_dict(v) if isinstance(v, dict) else v for v in value
            ]
        else:
            self.assert_isinstance(value, "frames", StatCount, is_array=True)
        self._property_frames = value


class SchemaTypeEnum(StringEnum):
    frame = "frame"
    source = "source"
    roi = "roi"


class VersionParadigmEnum(StringEnum):
    single_version = "single_version"
    general = "general"


class Version(NonStrictDataModel):
    """
    :param id: Version ID
    :type id: str
    :param name: Version name
    :type name: str
    :param comment: Version comment
    :type comment: str
    :param parent: Version parent ID
    :type parent: str
    :param task: Task ID of the task which created the version
    :type task: str
    :param status: Version status
    :type status: VersionStatusEnum
    :param company: Company ID
    :type company: str
    :param created: Version creation time (UTC)
    :type created: datetime.datetime
    :param es_index: Name of elasticsearch index
    :type es_index: str
    :param dataset: Datset ID
    :type dataset: str
    :param user: Associated user ID
    :type user: str
    :param committed: Commit time
    :type committed: datetime.datetime
    :param published: Publish time
    :type published: datetime.datetime
    :param committed_rois_ts: Timestamp of last committed ROI
    :type committed_rois_ts: float
    :param committed_frames_ts: Timestamp of last committed frame
    :type committed_frames_ts: float
    :param last_frames_update: Last time version was created, committed or frames
        were updated or saved
    :type last_frames_update: datetime.datetime
    :param metadata: User-provided metadata
    :type metadata: dict
    :param tags: List of user-defined tags
    :type tags: Sequence[str]
    :param system_tags: List of system tags. This field is reserved for system use,
        please don't use it.
    :type system_tags: Sequence[str]
    :param stats: Version statistics
    :type stats: Statistics
    """

    _schema = {
        "properties": {
            "comment": {"description": "Version comment", "type": ["string", "null"]},
            "committed": {
                "description": "Commit time",
                "format": "date-time",
                "type": ["string", "null"],
            },
            "committed_frames_ts": {
                "description": "Timestamp of last committed frame",
                "type": ["number", "null"],
            },
            "committed_rois_ts": {
                "description": "Timestamp of last committed ROI",
                "type": ["number", "null"],
            },
            "company": {"description": "Company ID", "type": ["string", "null"]},
            "created": {
                "description": "Version creation time (UTC) ",
                "format": "date-time",
                "type": ["string", "null"],
            },
            "dataset": {"description": "Datset ID", "type": ["string", "null"]},
            "es_index": {
                "description": "Name of elasticsearch index",
                "type": ["string", "null"],
            },
            "id": {"description": "Version ID", "type": ["string", "null"]},
            "last_frames_update": {
                "description": "Last time version was created, committed or frames were updated or saved",
                "format": "date-time",
                "type": ["string", "null"],
            },
            "metadata": {
                "additionalProperties": True,
                "description": "User-provided metadata",
                "type": ["object", "null"],
            },
            "name": {"description": "Version name", "type": ["string", "null"]},
            "parent": {"description": "Version parent ID", "type": ["string", "null"]},
            "published": {
                "description": "Publish time",
                "format": "date-time",
                "type": ["string", "null"],
            },
            "stats": {
                "description": "Version statistics",
                "oneOf": [{"$ref": "#/definitions/statistics"}, {"type": "null"}],
            },
            "status": {
                "description": "Version status",
                "oneOf": [
                    {"$ref": "#/definitions/version_status_enum"},
                    {"type": "null"},
                ],
            },
            "system_tags": {
                "description": "List of system tags. This field is reserved for system use, please don't use it.",
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
            "tags": {
                "description": "List of user-defined tags",
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
            "task": {
                "description": "Task ID of the task which created the version",
                "type": ["string", "null"],
            },
            "user": {"description": "Associated user ID", "type": ["string", "null"]},
        },
        "type": "object",
    }

    def __init__(
        self,
        id=None,
        name=None,
        comment=None,
        parent=None,
        task=None,
        status=None,
        company=None,
        created=None,
        es_index=None,
        dataset=None,
        user=None,
        committed=None,
        published=None,
        committed_rois_ts=None,
        committed_frames_ts=None,
        last_frames_update=None,
        metadata=None,
        tags=None,
        system_tags=None,
        stats=None,
        **kwargs
    ):
        super(Version, self).__init__(**kwargs)
        self.id = id
        self.name = name
        self.comment = comment
        self.parent = parent
        self.task = task
        self.status = status
        self.company = company
        self.created = created
        self.es_index = es_index
        self.dataset = dataset
        self.user = user
        self.committed = committed
        self.published = published
        self.committed_rois_ts = committed_rois_ts
        self.committed_frames_ts = committed_frames_ts
        self.last_frames_update = last_frames_update
        self.metadata = metadata
        self.tags = tags
        self.system_tags = system_tags
        self.stats = stats

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

    @schema_property("comment")
    def comment(self):
        return self._property_comment

    @comment.setter
    def comment(self, value):
        if value is None:
            self._property_comment = None
            return

        self.assert_isinstance(value, "comment", six.string_types)
        self._property_comment = value

    @schema_property("parent")
    def parent(self):
        return self._property_parent

    @parent.setter
    def parent(self, value):
        if value is None:
            self._property_parent = None
            return

        self.assert_isinstance(value, "parent", six.string_types)
        self._property_parent = value

    @schema_property("task")
    def task(self):
        return self._property_task

    @task.setter
    def task(self, value):
        if value is None:
            self._property_task = None
            return

        self.assert_isinstance(value, "task", six.string_types)
        self._property_task = value

    @schema_property("status")
    def status(self):
        return self._property_status

    @status.setter
    def status(self, value):
        if value is None:
            self._property_status = None
            return
        if isinstance(value, six.string_types):
            try:
                value = VersionStatusEnum(value)
            except ValueError:
                pass
        else:
            self.assert_isinstance(value, "status", enum.Enum)
        self._property_status = value

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

    @schema_property("es_index")
    def es_index(self):
        return self._property_es_index

    @es_index.setter
    def es_index(self, value):
        if value is None:
            self._property_es_index = None
            return

        self.assert_isinstance(value, "es_index", six.string_types)
        self._property_es_index = value

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

    @schema_property("committed")
    def committed(self):
        return self._property_committed

    @committed.setter
    def committed(self, value):
        if value is None:
            self._property_committed = None
            return

        self.assert_isinstance(value, "committed", six.string_types + (datetime,))
        if not isinstance(value, datetime):
            value = parse_datetime(value)
        self._property_committed = value

    @schema_property("published")
    def published(self):
        return self._property_published

    @published.setter
    def published(self, value):
        if value is None:
            self._property_published = None
            return

        self.assert_isinstance(value, "published", six.string_types + (datetime,))
        if not isinstance(value, datetime):
            value = parse_datetime(value)
        self._property_published = value

    @schema_property("committed_rois_ts")
    def committed_rois_ts(self):
        return self._property_committed_rois_ts

    @committed_rois_ts.setter
    def committed_rois_ts(self, value):
        if value is None:
            self._property_committed_rois_ts = None
            return

        self.assert_isinstance(value, "committed_rois_ts", six.integer_types + (float,))
        self._property_committed_rois_ts = value

    @schema_property("committed_frames_ts")
    def committed_frames_ts(self):
        return self._property_committed_frames_ts

    @committed_frames_ts.setter
    def committed_frames_ts(self, value):
        if value is None:
            self._property_committed_frames_ts = None
            return

        self.assert_isinstance(
            value, "committed_frames_ts", six.integer_types + (float,)
        )
        self._property_committed_frames_ts = value

    @schema_property("last_frames_update")
    def last_frames_update(self):
        return self._property_last_frames_update

    @last_frames_update.setter
    def last_frames_update(self, value):
        if value is None:
            self._property_last_frames_update = None
            return

        self.assert_isinstance(
            value, "last_frames_update", six.string_types + (datetime,)
        )
        if not isinstance(value, datetime):
            value = parse_datetime(value)
        self._property_last_frames_update = value

    @schema_property("metadata")
    def metadata(self):
        return self._property_metadata

    @metadata.setter
    def metadata(self, value):
        if value is None:
            self._property_metadata = None
            return

        self.assert_isinstance(value, "metadata", (dict,))
        self._property_metadata = value

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

    @schema_property("stats")
    def stats(self):
        return self._property_stats

    @stats.setter
    def stats(self, value):
        if value is None:
            self._property_stats = None
            return
        if isinstance(value, dict):
            value = Statistics.from_dict(value)
        else:
            self.assert_isinstance(value, "stats", Statistics)
        self._property_stats = value


class Dataset(NonStrictDataModel):
    """
    :param id: Dataset ID
    :type id: str
    :param name: Dataset name
    :type name: str
    :param user: Associated user ID
    :type user: str
    :param company: Company ID
    :type company: str
    :param created: Dataset creation time (UTC)
    :type created: datetime.datetime
    :param last_update: Time of last update (UTC). Updated on dataset update; on
        any version operation: when version is created, modified, committed, published
        or deleted; and on any frame operation: when frames are added, modified or
        deleted.
    :type last_update: datetime.datetime
    :param comment:
    :type comment: str
    :param tags: List of user-defined tags
    :type tags: Sequence[str]
    :param system_tags: List of system tags. This field is reserved for system use,
        please don't use it.
    :type system_tags: Sequence[str]
    :param terms_of_use: Terms of use string
    :type terms_of_use: str
    :param metadata: User-provided metadata
    :type metadata: dict
    :param project: Associated project ID
    :type project: str
    :param display_stats: Calculated statistics for the latest committed or
        published version
    :type display_stats: Statistics
    :param display_version_name: The name of the version from which statistics are
        taken
    :type display_version_name: str
    :param version_count: Amount of versions in dataset. Only supported by
        datasets.get_all.
    :type version_count: int
    :param head_version: The most recent version for write operations. Calculated
        as the non-published version with the longest path to the root.
    :type head_version: Version
    :param paradigm: 'single_version' for datasets whose version tree has only one
        path, 'general' otherwise
    :type paradigm: VersionParadigmEnum
    """

    _schema = {
        "properties": {
            "comment": {"description": "", "type": ["string", "null"]},
            "company": {"description": "Company ID", "type": ["string", "null"]},
            "created": {
                "description": "Dataset creation time (UTC)",
                "format": "date-time",
                "type": ["string", "null"],
            },
            "display_stats": {
                "description": "Calculated statistics for the latest committed or published version",
                "oneOf": [{"$ref": "#/definitions/statistics"}, {"type": "null"}],
            },
            "display_version_name": {
                "description": "The name of the version from which statistics are taken",
                "type": ["string", "null"],
            },
            "head_version": {
                "description": (
                    "The most recent version for write operations. Calculated as the non-published version with the"
                    " longest path to the root."
                ),
                "oneOf": [{"$ref": "#/definitions/version"}, {"type": "null"}],
            },
            "id": {"description": "Dataset ID", "type": ["string", "null"]},
            "last_update": {
                "description": (
                    "Time of last update (UTC). Updated on dataset update; on any version operation:\nwhen version is"
                    " created, modified, committed, published or deleted; and on any frame operation: when frames are"
                    " added,\nmodified or deleted."
                ),
                "format": "date-time",
                "type": ["string", "null"],
            },
            "metadata": {
                "additionalProperties": True,
                "description": "User-provided metadata",
                "type": ["object", "null"],
            },
            "name": {"description": "Dataset name", "type": ["string", "null"]},
            "paradigm": {
                "description": (
                    "'single_version' for datasets whose version tree has only one path, 'general' otherwise"
                ),
                "oneOf": [
                    {"$ref": "#/definitions/version_paradigm_enum"},
                    {"type": "null"},
                ],
            },
            "project": {
                "description": "Associated project ID",
                "type": ["string", "null"],
            },
            "system_tags": {
                "description": "List of system tags. This field is reserved for system use, please don't use it.",
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
            "tags": {
                "description": "List of user-defined tags",
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
            "terms_of_use": {
                "description": "Terms of use string",
                "type": ["string", "null"],
            },
            "user": {"description": "Associated user ID", "type": ["string", "null"]},
            "version_count": {
                "description": "Amount of versions in dataset. Only supported by datasets.get_all.",
                "type": ["integer", "null"],
            },
        },
        "type": "object",
    }

    def __init__(
        self,
        id=None,
        name=None,
        user=None,
        company=None,
        created=None,
        last_update=None,
        comment=None,
        tags=None,
        system_tags=None,
        terms_of_use=None,
        metadata=None,
        project=None,
        display_stats=None,
        display_version_name=None,
        version_count=None,
        head_version=None,
        paradigm=None,
        **kwargs
    ):
        super(Dataset, self).__init__(**kwargs)
        self.id = id
        self.name = name
        self.user = user
        self.company = company
        self.created = created
        self.last_update = last_update
        self.comment = comment
        self.tags = tags
        self.system_tags = system_tags
        self.terms_of_use = terms_of_use
        self.metadata = metadata
        self.project = project
        self.display_stats = display_stats
        self.display_version_name = display_version_name
        self.version_count = version_count
        self.head_version = head_version
        self.paradigm = paradigm

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

    @schema_property("last_update")
    def last_update(self):
        return self._property_last_update

    @last_update.setter
    def last_update(self, value):
        if value is None:
            self._property_last_update = None
            return

        self.assert_isinstance(value, "last_update", six.string_types + (datetime,))
        if not isinstance(value, datetime):
            value = parse_datetime(value)
        self._property_last_update = value

    @schema_property("comment")
    def comment(self):
        return self._property_comment

    @comment.setter
    def comment(self, value):
        if value is None:
            self._property_comment = None
            return

        self.assert_isinstance(value, "comment", six.string_types)
        self._property_comment = value

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

    @schema_property("terms_of_use")
    def terms_of_use(self):
        return self._property_terms_of_use

    @terms_of_use.setter
    def terms_of_use(self, value):
        if value is None:
            self._property_terms_of_use = None
            return

        self.assert_isinstance(value, "terms_of_use", six.string_types)
        self._property_terms_of_use = value

    @schema_property("metadata")
    def metadata(self):
        return self._property_metadata

    @metadata.setter
    def metadata(self, value):
        if value is None:
            self._property_metadata = None
            return

        self.assert_isinstance(value, "metadata", (dict,))
        self._property_metadata = value

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

    @schema_property("display_stats")
    def display_stats(self):
        return self._property_display_stats

    @display_stats.setter
    def display_stats(self, value):
        if value is None:
            self._property_display_stats = None
            return
        if isinstance(value, dict):
            value = Statistics.from_dict(value)
        else:
            self.assert_isinstance(value, "display_stats", Statistics)
        self._property_display_stats = value

    @schema_property("display_version_name")
    def display_version_name(self):
        return self._property_display_version_name

    @display_version_name.setter
    def display_version_name(self, value):
        if value is None:
            self._property_display_version_name = None
            return

        self.assert_isinstance(value, "display_version_name", six.string_types)
        self._property_display_version_name = value

    @schema_property("version_count")
    def version_count(self):
        return self._property_version_count

    @version_count.setter
    def version_count(self, value):
        if value is None:
            self._property_version_count = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "version_count", six.integer_types)
        self._property_version_count = value

    @schema_property("head_version")
    def head_version(self):
        return self._property_head_version

    @head_version.setter
    def head_version(self, value):
        if value is None:
            self._property_head_version = None
            return
        if isinstance(value, dict):
            value = Version.from_dict(value)
        else:
            self.assert_isinstance(value, "head_version", Version)
        self._property_head_version = value

    @schema_property("paradigm")
    def paradigm(self):
        return self._property_paradigm

    @paradigm.setter
    def paradigm(self, value):
        if value is None:
            self._property_paradigm = None
            return
        if isinstance(value, six.string_types):
            try:
                value = VersionParadigmEnum(value)
            except ValueError:
                pass
        else:
            self.assert_isinstance(value, "paradigm", enum.Enum)
        self._property_paradigm = value


class CommitVersionRequest(Request):
    """
    Commit changes to a draft version.

    :param version: Draft version ID
    :type version: str
    :param override_stats: Override version statistics (when provided, these will
        be used instead of computed statistics)
    :type override_stats: Statistics
    :param calculate_stats: If set to false then the version statistics will not be
        calculated on commit (only when version publish not requested). The default is
        true
    :type calculate_stats: bool
    :param publish: If set to true, version will also be published.
    :type publish: bool
    :param force: If publish=true, ignore ongoing annotation tasks with this
        version as input
    :type force: bool
    :param publishing_task: ID of an in-progress annotation task calling this
        endpoint. Versions which are used as input of in-progress annotation tasks can
        only be published if there is only one such task and its ID is sent in this
        field. This is required if one exists.
    :type publishing_task: str
    """

    _service = "datasets"
    _action = "commit_version"
    _version = "2.23"
    _schema = {
        "definitions": {
            "stat_count": {
                "properties": {
                    "count": {
                        "description": "Item name",
                        "type": ["integer", "null"],
                    },
                    "name": {
                        "description": "Number of appearances",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "statistics": {
                "properties": {
                    "content_types": {
                        "items": {
                            "$ref": "#/definitions/stat_count",
                            "description": (
                                "List of content type counts for the version (e.g.\n                    'image/jpeg',"
                                " 'image/png', 'video/mp4')"
                            ),
                        },
                        "type": ["array", "null"],
                    },
                    "frames": {
                        "items": {
                            "$ref": "#/definitions/stat_count",
                            "description": (
                                "List of frame counts, indicating the\n                    type of frames included in"
                                " the version (annotated/"
                            ),
                        },
                        "type": ["array", "null"],
                    },
                    "labels": {
                        "items": {
                            "$ref": "#/definitions/stat_count",
                            "description": (
                                "List of labels' counts,\n                    indicating the categories included in the"
                                " version"
                            ),
                        },
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            },
        },
        "properties": {
            "calculate_stats": {
                "default": True,
                "description": (
                    "If set to false then the version statistics will not be calculated on commit (only when version"
                    " publish not requested). The default is true"
                ),
                "type": "boolean",
            },
            "force": {
                "default": False,
                "description": "If publish=true, ignore ongoing annotation tasks with this version as input",
                "type": "boolean",
            },
            "override_stats": {
                "$ref": "#/definitions/statistics",
                "description": (
                    "Override version statistics (when provided, these will be used instead of computed statistics)"
                ),
            },
            "publish": {
                "default": False,
                "description": "If set to true, version will also be published.",
                "type": "boolean",
            },
            "publishing_task": {
                "description": (
                    "ID of an in-progress annotation task calling this endpoint.\n                    Versions which"
                    " are used as input of in-progress annotation tasks can only be published\n                    if"
                    " there is only one such task and its ID is sent in this field.\n                    This is"
                    " required if one exists."
                ),
                "type": "string",
            },
            "version": {"description": "Draft version ID", "type": "string"},
        },
        "required": ["version"],
        "type": "object",
    }

    def __init__(
        self,
        version,
        override_stats=None,
        calculate_stats=True,
        publish=False,
        force=False,
        publishing_task=None,
        **kwargs
    ):
        super(CommitVersionRequest, self).__init__(**kwargs)
        self.version = version
        self.override_stats = override_stats
        self.calculate_stats = calculate_stats
        self.publish = publish
        self.force = force
        self.publishing_task = publishing_task

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

    @schema_property("override_stats")
    def override_stats(self):
        return self._property_override_stats

    @override_stats.setter
    def override_stats(self, value):
        if value is None:
            self._property_override_stats = None
            return
        if isinstance(value, dict):
            value = Statistics.from_dict(value)
        else:
            self.assert_isinstance(value, "override_stats", Statistics)
        self._property_override_stats = value

    @schema_property("calculate_stats")
    def calculate_stats(self):
        return self._property_calculate_stats

    @calculate_stats.setter
    def calculate_stats(self, value):
        if value is None:
            self._property_calculate_stats = None
            return

        self.assert_isinstance(value, "calculate_stats", (bool,))
        self._property_calculate_stats = value

    @schema_property("publish")
    def publish(self):
        return self._property_publish

    @publish.setter
    def publish(self, value):
        if value is None:
            self._property_publish = None
            return

        self.assert_isinstance(value, "publish", (bool,))
        self._property_publish = value

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

    @schema_property("publishing_task")
    def publishing_task(self):
        return self._property_publishing_task

    @publishing_task.setter
    def publishing_task(self, value):
        if value is None:
            self._property_publishing_task = None
            return

        self.assert_isinstance(value, "publishing_task", six.string_types)
        self._property_publishing_task = value


class CommitVersionResponse(Response):
    """
    Response of datasets.commit_version endpoint.

    :param version: Committed version ID
    :type version: str
    :param parent: Committed version parent version ID
    :type parent: str
    :param dataset: Dataset ID
    :type dataset: str
    :param merged: Number of merged frames
    :type merged: int
    :param saved_and_updated: Number of saved and updated frames
    :type saved_and_updated: int
    :param deleted: Number of deleted frames
    :type deleted: int
    :param total: Total number of processed frames
    :type total: int
    :param failed: Number of failures
    :type failed: int
    :param errors: Failure details
    :type errors: Sequence[dict]
    """

    _service = "datasets"
    _action = "commit_version"
    _version = "2.23"

    _schema = {
        "definitions": {},
        "properties": {
            "dataset": {"description": "Dataset ID", "type": ["string", "null"]},
            "deleted": {
                "description": "Number of deleted frames",
                "type": ["integer", "null"],
            },
            "errors": {
                "description": "Failure details",
                "items": {
                    "additionalProperties": True,
                    "description": "Json object describing an update error",
                    "type": "object",
                },
                "type": ["array", "null"],
            },
            "failed": {
                "description": "Number of failures",
                "type": ["integer", "null"],
            },
            "merged": {
                "description": "Number of merged frames",
                "type": ["integer", "null"],
            },
            "parent": {
                "description": "Committed version parent version ID",
                "type": ["string", "null"],
            },
            "saved_and_updated": {
                "description": "Number of saved and updated frames",
                "type": ["integer", "null"],
            },
            "total": {
                "description": "Total number of processed frames",
                "type": ["integer", "null"],
            },
            "version": {
                "description": "Committed version ID",
                "type": ["string", "null"],
            },
        },
        "type": "object",
    }

    def __init__(
        self,
        version=None,
        parent=None,
        dataset=None,
        merged=None,
        saved_and_updated=None,
        deleted=None,
        total=None,
        failed=None,
        errors=None,
        **kwargs
    ):
        super(CommitVersionResponse, self).__init__(**kwargs)
        self.version = version
        self.parent = parent
        self.dataset = dataset
        self.merged = merged
        self.saved_and_updated = saved_and_updated
        self.deleted = deleted
        self.total = total
        self.failed = failed
        self.errors = errors

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

    @schema_property("parent")
    def parent(self):
        return self._property_parent

    @parent.setter
    def parent(self, value):
        if value is None:
            self._property_parent = None
            return

        self.assert_isinstance(value, "parent", six.string_types)
        self._property_parent = value

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

    @schema_property("merged")
    def merged(self):
        return self._property_merged

    @merged.setter
    def merged(self, value):
        if value is None:
            self._property_merged = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "merged", six.integer_types)
        self._property_merged = value

    @schema_property("saved_and_updated")
    def saved_and_updated(self):
        return self._property_saved_and_updated

    @saved_and_updated.setter
    def saved_and_updated(self, value):
        if value is None:
            self._property_saved_and_updated = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "saved_and_updated", six.integer_types)
        self._property_saved_and_updated = value

    @schema_property("deleted")
    def deleted(self):
        return self._property_deleted

    @deleted.setter
    def deleted(self, value):
        if value is None:
            self._property_deleted = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "deleted", six.integer_types)
        self._property_deleted = value

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

    @schema_property("failed")
    def failed(self):
        return self._property_failed

    @failed.setter
    def failed(self, value):
        if value is None:
            self._property_failed = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "failed", six.integer_types)
        self._property_failed = value

    @schema_property("errors")
    def errors(self):
        return self._property_errors

    @errors.setter
    def errors(self, value):
        if value is None:
            self._property_errors = None
            return

        self.assert_isinstance(value, "errors", (list, tuple))

        self.assert_isinstance(value, "errors", (dict,), is_array=True)
        self._property_errors = value


class CreateRequest(Request):
    """
    Creates a new dataset with an initial (empty) version

    :param name: Dataset name. Unique within the company.
    :type name: str
    :param comment: Dataset comment
    :type comment: str
    :param tags: User-defined tags list
    :type tags: Sequence[str]
    :param system_tags: System tags list. This field is reserved for system use,
        please don't use it.
    :type system_tags: Sequence[str]
    :param terms_of_use: Terms of use string
    :type terms_of_use: str
    :param metadata: User-specified metadata object. Keys must not include '$' and
        '.'.
    :type metadata: dict
    :param public: Create a public dataset Limited to 'root' users.
    :type public: bool
    """

    _service = "datasets"
    _action = "create"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {
            "comment": {"description": "Dataset comment", "type": "string"},
            "metadata": {
                "additionalProperties": True,
                "description": "User-specified metadata object. Keys must not include '$' and '.'.",
                "type": "object",
            },
            "name": {
                "description": "Dataset name. Unique within the company.",
                "type": "string",
            },
            "public": {
                "description": "Create a public dataset Limited to 'root' users.",
                "type": "boolean",
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
            "terms_of_use": {"description": "Terms of use string", "type": "string"},
            "project": {
                "description": "Associated project ID",
                "type": ["string", "null"],
            },
        },
        "required": ["name"],
        "type": "object",
    }

    def __init__(
        self,
        name,
        comment=None,
        tags=None,
        system_tags=None,
        terms_of_use=None,
        metadata=None,
        public=None,
        project=None,
        **kwargs
    ):
        super(CreateRequest, self).__init__(**kwargs)
        self.name = name
        self.comment = comment
        self.tags = tags
        self.system_tags = system_tags
        self.terms_of_use = terms_of_use
        self.metadata = metadata
        self.public = public
        self.project = project

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

    @schema_property("comment")
    def comment(self):
        return self._property_comment

    @comment.setter
    def comment(self, value):
        if value is None:
            self._property_comment = None
            return

        self.assert_isinstance(value, "comment", six.string_types)
        self._property_comment = value

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

    @schema_property("terms_of_use")
    def terms_of_use(self):
        return self._property_terms_of_use

    @terms_of_use.setter
    def terms_of_use(self, value):
        if value is None:
            self._property_terms_of_use = None
            return

        self.assert_isinstance(value, "terms_of_use", six.string_types)
        self._property_terms_of_use = value

    @schema_property("metadata")
    def metadata(self):
        return self._property_metadata

    @metadata.setter
    def metadata(self, value):
        if value is None:
            self._property_metadata = None
            return

        self.assert_isinstance(value, "metadata", (dict,))
        self._property_metadata = value

    @schema_property("public")
    def public(self):
        return self._property_public

    @public.setter
    def public(self, value):
        if value is None:
            self._property_public = None
            return

        self.assert_isinstance(value, "public", (bool,))
        self._property_public = value

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


class CreateResponse(Response):
    """
    Response of datasets.create endpoint.

    :param id: ID of the dataset
    :type id: str
    """

    _service = "datasets"
    _action = "create"
    _version = "2.23"

    _schema = {
        "definitions": {},
        "properties": {
            "id": {"description": "ID of the dataset", "type": ["string", "null"]}
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


class CreateVersionRequest(Request):
    """
    Creates a new dataset version

    :param dataset: Dataset ID
    :type dataset: str
    :param task: ID of the task creating the version
    :type task: str
    :param name: Version name Unique
    :type name: str
    :param comment: Version comment
    :type comment: str
    :param parent: Version parent ID
    :type parent: str
    :param stats: Version statistics
    :type stats: Statistics
    :param metadata: User-specified metadata object. Keys must not include '$' and
        '.'.
    :type metadata: dict
    :param tags: User-defined tags list
    :type tags: Sequence[str]
    :param system_tags: System tags list. This field is reserved for system use,
        please don't use it.
    :type system_tags: Sequence[str]
    """

    _service = "datasets"
    _action = "create_version"
    _version = "2.23"
    _schema = {
        "definitions": {
            "stat_count": {
                "properties": {
                    "count": {
                        "description": "Item name",
                        "type": ["integer", "null"],
                    },
                    "name": {
                        "description": "Number of appearances",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "statistics": {
                "properties": {
                    "content_types": {
                        "items": {
                            "$ref": "#/definitions/stat_count",
                            "description": (
                                "List of content type counts for the version (e.g.\n                    'image/jpeg',"
                                " 'image/png', 'video/mp4')"
                            ),
                        },
                        "type": ["array", "null"],
                    },
                    "frames": {
                        "items": {
                            "$ref": "#/definitions/stat_count",
                            "description": (
                                "List of frame counts, indicating the\n                    type of frames included in"
                                " the version (annotated/"
                            ),
                        },
                        "type": ["array", "null"],
                    },
                    "labels": {
                        "items": {
                            "$ref": "#/definitions/stat_count",
                            "description": (
                                "List of labels' counts,\n                    indicating the categories included in the"
                                " version"
                            ),
                        },
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            },
        },
        "properties": {
            "comment": {"description": "Version comment", "type": "string"},
            "dataset": {"description": "Dataset ID", "type": "string"},
            "metadata": {
                "additionalProperties": True,
                "description": "User-specified metadata object. Keys must not include '$' and '.'.",
                "type": "object",
            },
            "name": {"description": "Version name Unique", "type": "string"},
            "parent": {"description": "Version parent ID", "type": "string"},
            "stats": {
                "$ref": "#/definitions/statistics",
                "description": "Version statistics",
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
            "task": {
                "description": "ID of the task creating the version",
                "type": "string",
            },
        },
        "required": ["dataset", "name"],
        "type": "object",
    }

    def __init__(
        self,
        dataset,
        name,
        task=None,
        comment=None,
        parent=None,
        stats=None,
        metadata=None,
        tags=None,
        system_tags=None,
        **kwargs
    ):
        super(CreateVersionRequest, self).__init__(**kwargs)
        self.dataset = dataset
        self.task = task
        self.name = name
        self.comment = comment
        self.parent = parent
        self.stats = stats
        self.metadata = metadata
        self.tags = tags
        self.system_tags = system_tags

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

    @schema_property("task")
    def task(self):
        return self._property_task

    @task.setter
    def task(self, value):
        if value is None:
            self._property_task = None
            return

        self.assert_isinstance(value, "task", six.string_types)
        self._property_task = value

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

    @schema_property("comment")
    def comment(self):
        return self._property_comment

    @comment.setter
    def comment(self, value):
        if value is None:
            self._property_comment = None
            return

        self.assert_isinstance(value, "comment", six.string_types)
        self._property_comment = value

    @schema_property("parent")
    def parent(self):
        return self._property_parent

    @parent.setter
    def parent(self, value):
        if value is None:
            self._property_parent = None
            return

        self.assert_isinstance(value, "parent", six.string_types)
        self._property_parent = value

    @schema_property("stats")
    def stats(self):
        return self._property_stats

    @stats.setter
    def stats(self, value):
        if value is None:
            self._property_stats = None
            return
        if isinstance(value, dict):
            value = Statistics.from_dict(value)
        else:
            self.assert_isinstance(value, "stats", Statistics)
        self._property_stats = value

    @schema_property("metadata")
    def metadata(self):
        return self._property_metadata

    @metadata.setter
    def metadata(self, value):
        if value is None:
            self._property_metadata = None
            return

        self.assert_isinstance(value, "metadata", (dict,))
        self._property_metadata = value

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


class CreateVersionResponse(Response):
    """
    Response of datasets.create_version endpoint.

    :param id: ID of the version
    :type id: str
    """

    _service = "datasets"
    _action = "create_version"
    _version = "2.23"

    _schema = {
        "definitions": {},
        "properties": {
            "id": {"description": "ID of the version", "type": ["string", "null"]}
        },
        "type": "object",
    }

    def __init__(self, id=None, **kwargs):
        super(CreateVersionResponse, self).__init__(**kwargs)
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
    Delete a dataset

    :param dataset: Dataset ID
    :type dataset: str
    :param delete_all_versions:
    :type delete_all_versions: bool
    :param force:
    :type force: bool
    """

    _service = "datasets"
    _action = "delete"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {
            "dataset": {"description": "Dataset ID", "type": "string"},
            "delete_all_versions": {"description": "", "type": "boolean"},
            "force": {"description": "", "type": "boolean"},
        },
        "required": ["dataset"],
        "type": "object",
    }

    def __init__(self, dataset, delete_all_versions=None, force=None, **kwargs):
        super(DeleteRequest, self).__init__(**kwargs)
        self.dataset = dataset
        self.delete_all_versions = delete_all_versions
        self.force = force

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

    @schema_property("delete_all_versions")
    def delete_all_versions(self):
        return self._property_delete_all_versions

    @delete_all_versions.setter
    def delete_all_versions(self, value):
        if value is None:
            self._property_delete_all_versions = None
            return

        self.assert_isinstance(value, "delete_all_versions", (bool,))
        self._property_delete_all_versions = value

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
    Response of datasets.delete endpoint.

    :param deleted:
    :type deleted: bool
    :param deleted_versions:
    :type deleted_versions: Sequence[str]
    """

    _service = "datasets"
    _action = "delete"
    _version = "2.23"

    _schema = {
        "definitions": {},
        "properties": {
            "deleted": {"type": ["boolean", "null"]},
            "deleted_versions": {
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, deleted=None, deleted_versions=None, **kwargs):
        super(DeleteResponse, self).__init__(**kwargs)
        self.deleted = deleted
        self.deleted_versions = deleted_versions

    @schema_property("deleted")
    def deleted(self):
        return self._property_deleted

    @deleted.setter
    def deleted(self, value):
        if value is None:
            self._property_deleted = None
            return

        self.assert_isinstance(value, "deleted", (bool,))
        self._property_deleted = value

    @schema_property("deleted_versions")
    def deleted_versions(self):
        return self._property_deleted_versions

    @deleted_versions.setter
    def deleted_versions(self, value):
        if value is None:
            self._property_deleted_versions = None
            return

        self.assert_isinstance(value, "deleted_versions", (list, tuple))

        self.assert_isinstance(
            value, "deleted_versions", six.string_types, is_array=True
        )
        self._property_deleted_versions = value


class DeleteFramesRequest(Request):
    """
    Delete frames in a draft version.

    :param version: Draft version ID
    :type version: str
    :param frames: Frame IDs to delete
    :type frames: Sequence[str]
    :param force: Ignore ongoing annotation tasks with this version as input
    :type force: bool
    """

    _service = "datasets"
    _action = "delete_frames"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {
            "force": {
                "default": False,
                "description": "Ignore ongoing annotation tasks with this version as input",
                "type": "boolean",
            },
            "frames": {
                "description": "Frame IDs to delete",
                "items": {"type": "string"},
                "type": "array",
            },
            "version": {"description": "Draft version ID", "type": "string"},
        },
        "required": ["version", "frames"],
        "type": "object",
    }

    def __init__(self, version, frames, force=False, **kwargs):
        super(DeleteFramesRequest, self).__init__(**kwargs)
        self.version = version
        self.frames = frames
        self.force = force

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

    @schema_property("frames")
    def frames(self):
        return self._property_frames

    @frames.setter
    def frames(self, value):
        if value is None:
            self._property_frames = None
            return

        self.assert_isinstance(value, "frames", (list, tuple))

        self.assert_isinstance(value, "frames", six.string_types, is_array=True)
        self._property_frames = value

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


class DeleteFramesResponse(Response):
    """
    Response of datasets.delete_frames endpoint.

    :param deleted: Number of frames deleted
    :type deleted: int
    """

    _service = "datasets"
    _action = "delete_frames"
    _version = "2.23"

    _schema = {
        "definitions": {},
        "properties": {
            "deleted": {
                "description": "Number of frames deleted",
                "type": ["integer", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, deleted=None, **kwargs):
        super(DeleteFramesResponse, self).__init__(**kwargs)
        self.deleted = deleted

    @schema_property("deleted")
    def deleted(self):
        return self._property_deleted

    @deleted.setter
    def deleted(self, value):
        if value is None:
            self._property_deleted = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "deleted", six.integer_types)
        self._property_deleted = value


class DeleteVersionRequest(Request):
    """
    Delete a version of a dataset

    :param version: Dataset version ID
    :type version: str
    :param force:
    :type force: bool
    """

    _service = "datasets"
    _action = "delete_version"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {
            "force": {"description": "", "type": "boolean"},
            "version": {"description": "Dataset version ID", "type": "string"},
        },
        "required": ["version"],
        "type": "object",
    }

    def __init__(self, version, force=None, **kwargs):
        super(DeleteVersionRequest, self).__init__(**kwargs)
        self.version = version
        self.force = force

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


class DeleteVersionResponse(Response):
    """
    Response of datasets.delete_version endpoint.

    :param deleted:
    :type deleted: bool
    """

    _service = "datasets"
    _action = "delete_version"
    _version = "2.23"

    _schema = {
        "definitions": {},
        "properties": {"deleted": {"type": ["boolean", "null"]}},
        "type": "object",
    }

    def __init__(self, deleted=None, **kwargs):
        super(DeleteVersionResponse, self).__init__(**kwargs)
        self.deleted = deleted

    @schema_property("deleted")
    def deleted(self):
        return self._property_deleted

    @deleted.setter
    def deleted(self, value):
        if value is None:
            self._property_deleted = None
            return

        self.assert_isinstance(value, "deleted", (bool,))
        self._property_deleted = value


class GetAllRequest(Request):
    """
    Gets a list of datasets information matching a query

    :param id: List of IDs to filter by
    :type id: Sequence[str]
    :param name: Get only datasets whose name matches this pattern (python regular
        expression syntax)
    :type name: str
    :param tags: User-defined tags filter. Use '-' for exclusion
    :type tags: Sequence[str]
    :param system_tags: System tags filter. Use '-' for exclusion (e.g.
        ['-archived'] for all non-hidden datasets)
    :type system_tags: Sequence[str]
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
    :param only_fields: List of document's field names (nesting is supported using
        '.', e.g. execution.model_labels). If provided, this list defines the query's
        projection (only these fields will be returned for each result entry)
    :type only_fields: Sequence[str]
    :param _all_: Multi-field pattern condition (all fields match pattern)
    :type _all_: MultiFieldPatternData
    :param _any_: Multi-field pattern condition (any field matches pattern)
    :type _any_: MultiFieldPatternData
    :param allow_public: Allow public datasets to be returned in the results
    :type allow_public: bool
    :param resolve_head: If set then dataset paradigm and head version are
        calculated and returned. Note: do not use it with queries that are supposed to
        return multiple datasets.
    :type resolve_head: bool
    :param scroll_id: Scroll ID returned from the previos calls to get_all
    :type scroll_id: str
    :param refresh_scroll: If set then all the data received with this scroll will
        be requeried
    :type refresh_scroll: bool
    :param size: The number of datasets to retrieve
    :type size: int
    """

    _service = "datasets"
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
            }
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
            "allow_public": {
                "default": True,
                "description": "Allow public datasets to be returned in the results",
                "type": ["boolean", "null"],
            },
            "id": {
                "description": "List of IDs to filter by",
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
            "name": {
                "description": "Get only datasets whose name matches this pattern (python regular expression syntax)",
                "type": ["string", "null"],
            },
            "only_fields": {
                "description": (
                    "List of document's field names (nesting is supported using '.', e.g. execution.model_labels). If"
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
            "refresh_scroll": {
                "description": "If set then all the data received with this scroll will be requeried",
                "type": ["boolean", "null"],
            },
            "resolve_head": {
                "default": False,
                "description": (
                    "If set then dataset paradigm and head version are calculated and returned. Note: do not use it"
                    " with queries that are supposed to return multiple datasets."
                ),
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
                "description": "The number of datasets to retrieve",
                "minimum": 1,
                "type": ["integer", "null"],
            },
            "system_tags": {
                "description": (
                    "System tags filter. Use '-' for exclusion (e.g. ['-archived'] for all non-hidden datasets)"
                ),
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
            "tags": {
                "description": "User-defined tags filter. Use '-' for exclusion",
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
            "project": {
                "description": "Associated project ID",
                "type": ["string", "null"],
            },
        },
        "type": "object",
    }

    def __init__(
        self,
        id=None,
        name=None,
        tags=None,
        system_tags=None,
        page=None,
        page_size=None,
        order_by=None,
        search_text=None,
        only_fields=None,
        _all_=None,
        _any_=None,
        allow_public=True,
        resolve_head=False,
        scroll_id=None,
        refresh_scroll=None,
        size=None,
        project=None,
        **kwargs
    ):
        super(GetAllRequest, self).__init__(**kwargs)
        self.id = id
        self.name = name
        self.tags = tags
        self.system_tags = system_tags
        self.page = page
        self.page_size = page_size
        self.order_by = order_by
        self.search_text = search_text
        self.only_fields = only_fields
        self._all_ = _all_
        self._any_ = _any_
        self.allow_public = allow_public
        self.resolve_head = resolve_head
        self.scroll_id = scroll_id
        self.refresh_scroll = refresh_scroll
        self.size = size
        self.project = project

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

    @schema_property("allow_public")
    def allow_public(self):
        return self._property_allow_public

    @allow_public.setter
    def allow_public(self, value):
        if value is None:
            self._property_allow_public = None
            return

        self.assert_isinstance(value, "allow_public", (bool,))
        self._property_allow_public = value

    @schema_property("resolve_head")
    def resolve_head(self):
        return self._property_resolve_head

    @resolve_head.setter
    def resolve_head(self, value):
        if value is None:
            self._property_resolve_head = None
            return

        self.assert_isinstance(value, "resolve_head", (bool,))
        self._property_resolve_head = value

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


class GetAllResponse(Response):
    """
    Response of datasets.get_all endpoint.

    :param datasets: List of datasets
    :type datasets: Sequence[Dataset]
    :param scroll_id: Scroll ID that can be used with the next calls to get_all to
        retrieve more data
    :type scroll_id: str
    """

    _service = "datasets"
    _action = "get_all"
    _version = "2.23"

    _schema = {
        "definitions": {
            "dataset": {
                "properties": {
                    "comment": {"description": "", "type": ["string", "null"]},
                    "company": {
                        "description": "Company ID",
                        "type": ["string", "null"],
                    },
                    "created": {
                        "description": "Dataset creation time (UTC)",
                        "format": "date-time",
                        "type": ["string", "null"],
                    },
                    "display_stats": {
                        "description": "Calculated statistics for the latest committed or published version",
                        "oneOf": [
                            {"$ref": "#/definitions/statistics"},
                            {"type": "null"},
                        ],
                    },
                    "display_version_name": {
                        "description": "The name of the version from which statistics are taken",
                        "type": ["string", "null"],
                    },
                    "head_version": {
                        "description": (
                            "The most recent version for write operations. Calculated as the non-published version with"
                            " the longest path to the root."
                        ),
                        "oneOf": [{"$ref": "#/definitions/version"}, {"type": "null"}],
                    },
                    "id": {"description": "Dataset ID", "type": ["string", "null"]},
                    "last_update": {
                        "description": (
                            "Time of last update (UTC). Updated on dataset update; on any version operation:\nwhen"
                            " version is created, modified, committed, published or deleted; and on any frame"
                            " operation: when frames are added,\nmodified or deleted."
                        ),
                        "format": "date-time",
                        "type": ["string", "null"],
                    },
                    "metadata": {
                        "additionalProperties": True,
                        "description": "User-provided metadata",
                        "type": ["object", "null"],
                    },
                    "name": {
                        "description": "Dataset name",
                        "type": ["string", "null"],
                    },
                    "paradigm": {
                        "description": (
                            "'single_version' for datasets whose version tree has only one path, 'general' otherwise"
                        ),
                        "oneOf": [
                            {"$ref": "#/definitions/version_paradigm_enum"},
                            {"type": "null"},
                        ],
                    },
                    "project": {
                        "description": "Associated project ID",
                        "type": ["string", "null"],
                    },
                    "system_tags": {
                        "description": (
                            "List of system tags. This field is reserved for system use, please don't use it."
                        ),
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                    "tags": {
                        "description": "List of user-defined tags",
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                    "terms_of_use": {
                        "description": "Terms of use string",
                        "type": ["string", "null"],
                    },
                    "user": {
                        "description": "Associated user ID",
                        "type": ["string", "null"],
                    },
                    "version_count": {
                        "description": "Amount of versions in dataset. Only supported by datasets.get_all.",
                        "type": ["integer", "null"],
                    },
                },
                "type": "object",
            },
            "stat_count": {
                "properties": {
                    "count": {
                        "description": "Item name",
                        "type": ["integer", "null"],
                    },
                    "name": {
                        "description": "Number of appearances",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "statistics": {
                "properties": {
                    "content_types": {
                        "items": {
                            "$ref": "#/definitions/stat_count",
                            "description": (
                                "List of content type counts for the version (e.g.\n                    'image/jpeg',"
                                " 'image/png', 'video/mp4')"
                            ),
                        },
                        "type": ["array", "null"],
                    },
                    "frames": {
                        "items": {
                            "$ref": "#/definitions/stat_count",
                            "description": (
                                "List of frame counts, indicating the\n                    type of frames included in"
                                " the version (annotated/"
                            ),
                        },
                        "type": ["array", "null"],
                    },
                    "labels": {
                        "items": {
                            "$ref": "#/definitions/stat_count",
                            "description": (
                                "List of labels' counts,\n                    indicating the categories included in the"
                                " version"
                            ),
                        },
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            },
            "version": {
                "properties": {
                    "comment": {
                        "description": "Version comment",
                        "type": ["string", "null"],
                    },
                    "committed": {
                        "description": "Commit time",
                        "format": "date-time",
                        "type": ["string", "null"],
                    },
                    "committed_frames_ts": {
                        "description": "Timestamp of last committed frame",
                        "type": ["number", "null"],
                    },
                    "committed_rois_ts": {
                        "description": "Timestamp of last committed ROI",
                        "type": ["number", "null"],
                    },
                    "company": {
                        "description": "Company ID",
                        "type": ["string", "null"],
                    },
                    "created": {
                        "description": "Version creation time (UTC) ",
                        "format": "date-time",
                        "type": ["string", "null"],
                    },
                    "dataset": {
                        "description": "Datset ID",
                        "type": ["string", "null"],
                    },
                    "es_index": {
                        "description": "Name of elasticsearch index",
                        "type": ["string", "null"],
                    },
                    "id": {"description": "Version ID", "type": ["string", "null"]},
                    "last_frames_update": {
                        "description": "Last time version was created, committed or frames were updated or saved",
                        "format": "date-time",
                        "type": ["string", "null"],
                    },
                    "metadata": {
                        "additionalProperties": True,
                        "description": "User-provided metadata",
                        "type": ["object", "null"],
                    },
                    "name": {
                        "description": "Version name",
                        "type": ["string", "null"],
                    },
                    "parent": {
                        "description": "Version parent ID",
                        "type": ["string", "null"],
                    },
                    "published": {
                        "description": "Publish time",
                        "format": "date-time",
                        "type": ["string", "null"],
                    },
                    "stats": {
                        "description": "Version statistics",
                        "oneOf": [
                            {"$ref": "#/definitions/statistics"},
                            {"type": "null"},
                        ],
                    },
                    "status": {
                        "description": "Version status",
                        "oneOf": [
                            {"$ref": "#/definitions/version_status_enum"},
                            {"type": "null"},
                        ],
                    },
                    "system_tags": {
                        "description": (
                            "List of system tags. This field is reserved for system use, please don't use it."
                        ),
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                    "tags": {
                        "description": "List of user-defined tags",
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                    "task": {
                        "description": "Task ID of the task which created the version",
                        "type": ["string", "null"],
                    },
                    "user": {
                        "description": "Associated user ID",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "version_paradigm_enum": {
                "enum": ["single_version", "general"],
                "type": "string",
            },
            "version_status_enum": {
                "enum": ["draft", "committing", "committed", "published"],
                "type": "string",
            },
        },
        "properties": {
            "datasets": {
                "description": "List of datasets",
                "items": {"$ref": "#/definitions/dataset"},
                "type": ["array", "null"],
            },
            "scroll_id": {
                "description": "Scroll ID that can be used with the next calls to get_all to retrieve more data",
                "type": ["string", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, datasets=None, scroll_id=None, **kwargs):
        super(GetAllResponse, self).__init__(**kwargs)
        self.datasets = datasets
        self.scroll_id = scroll_id

    @schema_property("datasets")
    def datasets(self):
        return self._property_datasets

    @datasets.setter
    def datasets(self, value):
        if value is None:
            self._property_datasets = None
            return

        self.assert_isinstance(value, "datasets", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [Dataset.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "datasets", Dataset, is_array=True)
        self._property_datasets = value

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
    Gets dataset information

    :param dataset: Dataset ID
    :type dataset: str
    """

    _service = "datasets"
    _action = "get_by_id"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {"dataset": {"description": "Dataset ID", "type": "string"}},
        "required": ["dataset"],
        "type": "object",
    }

    def __init__(self, dataset, **kwargs):
        super(GetByIdRequest, self).__init__(**kwargs)
        self.dataset = dataset

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


class GetByIdResponse(Response):
    """
    Response of datasets.get_by_id endpoint.

    :param dataset: Dataset info
    :type dataset: Dataset
    """

    _service = "datasets"
    _action = "get_by_id"
    _version = "2.23"

    _schema = {
        "definitions": {
            "dataset": {
                "properties": {
                    "comment": {"description": "", "type": ["string", "null"]},
                    "company": {
                        "description": "Company ID",
                        "type": ["string", "null"],
                    },
                    "created": {
                        "description": "Dataset creation time (UTC)",
                        "format": "date-time",
                        "type": ["string", "null"],
                    },
                    "display_stats": {
                        "description": "Calculated statistics for the latest committed or published version",
                        "oneOf": [
                            {"$ref": "#/definitions/statistics"},
                            {"type": "null"},
                        ],
                    },
                    "display_version_name": {
                        "description": "The name of the version from which statistics are taken",
                        "type": ["string", "null"],
                    },
                    "head_version": {
                        "description": (
                            "The most recent version for write operations. Calculated as the non-published version with"
                            " the longest path to the root."
                        ),
                        "oneOf": [{"$ref": "#/definitions/version"}, {"type": "null"}],
                    },
                    "id": {"description": "Dataset ID", "type": ["string", "null"]},
                    "last_update": {
                        "description": (
                            "Time of last update (UTC). Updated on dataset update; on any version operation:\nwhen"
                            " version is created, modified, committed, published or deleted; and on any frame"
                            " operation: when frames are added,\nmodified or deleted."
                        ),
                        "format": "date-time",
                        "type": ["string", "null"],
                    },
                    "metadata": {
                        "additionalProperties": True,
                        "description": "User-provided metadata",
                        "type": ["object", "null"],
                    },
                    "name": {
                        "description": "Dataset name",
                        "type": ["string", "null"],
                    },
                    "paradigm": {
                        "description": (
                            "'single_version' for datasets whose version tree has only one path, 'general' otherwise"
                        ),
                        "oneOf": [
                            {"$ref": "#/definitions/version_paradigm_enum"},
                            {"type": "null"},
                        ],
                    },
                    "project": {
                        "description": "Associated project ID",
                        "type": ["string", "null"],
                    },
                    "system_tags": {
                        "description": (
                            "List of system tags. This field is reserved for system use, please don't use it."
                        ),
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                    "tags": {
                        "description": "List of user-defined tags",
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                    "terms_of_use": {
                        "description": "Terms of use string",
                        "type": ["string", "null"],
                    },
                    "user": {
                        "description": "Associated user ID",
                        "type": ["string", "null"],
                    },
                    "version_count": {
                        "description": "Amount of versions in dataset. Only supported by datasets.get_all.",
                        "type": ["integer", "null"],
                    },
                },
                "type": "object",
            },
            "stat_count": {
                "properties": {
                    "count": {
                        "description": "Item name",
                        "type": ["integer", "null"],
                    },
                    "name": {
                        "description": "Number of appearances",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "statistics": {
                "properties": {
                    "content_types": {
                        "items": {
                            "$ref": "#/definitions/stat_count",
                            "description": (
                                "List of content type counts for the version (e.g.\n                    'image/jpeg',"
                                " 'image/png', 'video/mp4')"
                            ),
                        },
                        "type": ["array", "null"],
                    },
                    "frames": {
                        "items": {
                            "$ref": "#/definitions/stat_count",
                            "description": (
                                "List of frame counts, indicating the\n                    type of frames included in"
                                " the version (annotated/"
                            ),
                        },
                        "type": ["array", "null"],
                    },
                    "labels": {
                        "items": {
                            "$ref": "#/definitions/stat_count",
                            "description": (
                                "List of labels' counts,\n                    indicating the categories included in the"
                                " version"
                            ),
                        },
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            },
            "version": {
                "properties": {
                    "comment": {
                        "description": "Version comment",
                        "type": ["string", "null"],
                    },
                    "committed": {
                        "description": "Commit time",
                        "format": "date-time",
                        "type": ["string", "null"],
                    },
                    "committed_frames_ts": {
                        "description": "Timestamp of last committed frame",
                        "type": ["number", "null"],
                    },
                    "committed_rois_ts": {
                        "description": "Timestamp of last committed ROI",
                        "type": ["number", "null"],
                    },
                    "company": {
                        "description": "Company ID",
                        "type": ["string", "null"],
                    },
                    "created": {
                        "description": "Version creation time (UTC) ",
                        "format": "date-time",
                        "type": ["string", "null"],
                    },
                    "dataset": {
                        "description": "Datset ID",
                        "type": ["string", "null"],
                    },
                    "es_index": {
                        "description": "Name of elasticsearch index",
                        "type": ["string", "null"],
                    },
                    "id": {"description": "Version ID", "type": ["string", "null"]},
                    "last_frames_update": {
                        "description": "Last time version was created, committed or frames were updated or saved",
                        "format": "date-time",
                        "type": ["string", "null"],
                    },
                    "metadata": {
                        "additionalProperties": True,
                        "description": "User-provided metadata",
                        "type": ["object", "null"],
                    },
                    "name": {
                        "description": "Version name",
                        "type": ["string", "null"],
                    },
                    "parent": {
                        "description": "Version parent ID",
                        "type": ["string", "null"],
                    },
                    "published": {
                        "description": "Publish time",
                        "format": "date-time",
                        "type": ["string", "null"],
                    },
                    "stats": {
                        "description": "Version statistics",
                        "oneOf": [
                            {"$ref": "#/definitions/statistics"},
                            {"type": "null"},
                        ],
                    },
                    "status": {
                        "description": "Version status",
                        "oneOf": [
                            {"$ref": "#/definitions/version_status_enum"},
                            {"type": "null"},
                        ],
                    },
                    "system_tags": {
                        "description": (
                            "List of system tags. This field is reserved for system use, please don't use it."
                        ),
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                    "tags": {
                        "description": "List of user-defined tags",
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                    "task": {
                        "description": "Task ID of the task which created the version",
                        "type": ["string", "null"],
                    },
                    "user": {
                        "description": "Associated user ID",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "version_paradigm_enum": {
                "enum": ["single_version", "general"],
                "type": "string",
            },
            "version_status_enum": {
                "enum": ["draft", "committing", "committed", "published"],
                "type": "string",
            },
        },
        "properties": {
            "dataset": {
                "description": "Dataset info",
                "oneOf": [{"$ref": "#/definitions/dataset"}, {"type": "null"}],
            }
        },
        "type": "object",
    }

    def __init__(self, dataset=None, **kwargs):
        super(GetByIdResponse, self).__init__(**kwargs)
        self.dataset = dataset

    @schema_property("dataset")
    def dataset(self):
        return self._property_dataset

    @dataset.setter
    def dataset(self, value):
        if value is None:
            self._property_dataset = None
            return
        if isinstance(value, dict):
            value = Dataset.from_dict(value)
        else:
            self.assert_isinstance(value, "dataset", Dataset)
        self._property_dataset = value


class GetByNameRequest(Request):
    """
    Gets dataset information by dataset name

    :param dataset: Dataset name
    :type dataset: str
    """

    _service = "datasets"
    _action = "get_by_name"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {"dataset": {"description": "Dataset name", "type": "string"}},
        "required": ["dataset"],
        "type": "object",
    }

    def __init__(self, dataset, **kwargs):
        super(GetByNameRequest, self).__init__(**kwargs)
        self.dataset = dataset

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


class GetByNameResponse(Response):
    """
    Response of datasets.get_by_name endpoint.

    :param dataset: Dataset info
    :type dataset: Dataset
    """

    _service = "datasets"
    _action = "get_by_name"
    _version = "2.23"

    _schema = {
        "definitions": {
            "dataset": {
                "properties": {
                    "comment": {"description": "", "type": ["string", "null"]},
                    "company": {
                        "description": "Company ID",
                        "type": ["string", "null"],
                    },
                    "created": {
                        "description": "Dataset creation time (UTC)",
                        "format": "date-time",
                        "type": ["string", "null"],
                    },
                    "display_stats": {
                        "description": "Calculated statistics for the latest committed or published version",
                        "oneOf": [
                            {"$ref": "#/definitions/statistics"},
                            {"type": "null"},
                        ],
                    },
                    "display_version_name": {
                        "description": "The name of the version from which statistics are taken",
                        "type": ["string", "null"],
                    },
                    "head_version": {
                        "description": (
                            "The most recent version for write operations. Calculated as the non-published version with"
                            " the longest path to the root."
                        ),
                        "oneOf": [{"$ref": "#/definitions/version"}, {"type": "null"}],
                    },
                    "id": {"description": "Dataset ID", "type": ["string", "null"]},
                    "last_update": {
                        "description": (
                            "Time of last update (UTC). Updated on dataset update; on any version operation:\nwhen"
                            " version is created, modified, committed, published or deleted; and on any frame"
                            " operation: when frames are added,\nmodified or deleted."
                        ),
                        "format": "date-time",
                        "type": ["string", "null"],
                    },
                    "metadata": {
                        "additionalProperties": True,
                        "description": "User-provided metadata",
                        "type": ["object", "null"],
                    },
                    "name": {
                        "description": "Dataset name",
                        "type": ["string", "null"],
                    },
                    "paradigm": {
                        "description": (
                            "'single_version' for datasets whose version tree has only one path, 'general' otherwise"
                        ),
                        "oneOf": [
                            {"$ref": "#/definitions/version_paradigm_enum"},
                            {"type": "null"},
                        ],
                    },
                    "project": {
                        "description": "Associated project ID",
                        "type": ["string", "null"],
                    },
                    "system_tags": {
                        "description": (
                            "List of system tags. This field is reserved for system use, please don't use it."
                        ),
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                    "tags": {
                        "description": "List of user-defined tags",
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                    "terms_of_use": {
                        "description": "Terms of use string",
                        "type": ["string", "null"],
                    },
                    "user": {
                        "description": "Associated user ID",
                        "type": ["string", "null"],
                    },
                    "version_count": {
                        "description": "Amount of versions in dataset. Only supported by datasets.get_all.",
                        "type": ["integer", "null"],
                    },
                },
                "type": "object",
            },
            "stat_count": {
                "properties": {
                    "count": {
                        "description": "Item name",
                        "type": ["integer", "null"],
                    },
                    "name": {
                        "description": "Number of appearances",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "statistics": {
                "properties": {
                    "content_types": {
                        "items": {
                            "$ref": "#/definitions/stat_count",
                            "description": (
                                "List of content type counts for the version (e.g.\n                    'image/jpeg',"
                                " 'image/png', 'video/mp4')"
                            ),
                        },
                        "type": ["array", "null"],
                    },
                    "frames": {
                        "items": {
                            "$ref": "#/definitions/stat_count",
                            "description": (
                                "List of frame counts, indicating the\n                    type of frames included in"
                                " the version (annotated/"
                            ),
                        },
                        "type": ["array", "null"],
                    },
                    "labels": {
                        "items": {
                            "$ref": "#/definitions/stat_count",
                            "description": (
                                "List of labels' counts,\n                    indicating the categories included in the"
                                " version"
                            ),
                        },
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            },
            "version": {
                "properties": {
                    "comment": {
                        "description": "Version comment",
                        "type": ["string", "null"],
                    },
                    "committed": {
                        "description": "Commit time",
                        "format": "date-time",
                        "type": ["string", "null"],
                    },
                    "committed_frames_ts": {
                        "description": "Timestamp of last committed frame",
                        "type": ["number", "null"],
                    },
                    "committed_rois_ts": {
                        "description": "Timestamp of last committed ROI",
                        "type": ["number", "null"],
                    },
                    "company": {
                        "description": "Company ID",
                        "type": ["string", "null"],
                    },
                    "created": {
                        "description": "Version creation time (UTC) ",
                        "format": "date-time",
                        "type": ["string", "null"],
                    },
                    "dataset": {
                        "description": "Datset ID",
                        "type": ["string", "null"],
                    },
                    "es_index": {
                        "description": "Name of elasticsearch index",
                        "type": ["string", "null"],
                    },
                    "id": {"description": "Version ID", "type": ["string", "null"]},
                    "last_frames_update": {
                        "description": "Last time version was created, committed or frames were updated or saved",
                        "format": "date-time",
                        "type": ["string", "null"],
                    },
                    "metadata": {
                        "additionalProperties": True,
                        "description": "User-provided metadata",
                        "type": ["object", "null"],
                    },
                    "name": {
                        "description": "Version name",
                        "type": ["string", "null"],
                    },
                    "parent": {
                        "description": "Version parent ID",
                        "type": ["string", "null"],
                    },
                    "published": {
                        "description": "Publish time",
                        "format": "date-time",
                        "type": ["string", "null"],
                    },
                    "stats": {
                        "description": "Version statistics",
                        "oneOf": [
                            {"$ref": "#/definitions/statistics"},
                            {"type": "null"},
                        ],
                    },
                    "status": {
                        "description": "Version status",
                        "oneOf": [
                            {"$ref": "#/definitions/version_status_enum"},
                            {"type": "null"},
                        ],
                    },
                    "system_tags": {
                        "description": (
                            "List of system tags. This field is reserved for system use, please don't use it."
                        ),
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                    "tags": {
                        "description": "List of user-defined tags",
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                    "task": {
                        "description": "Task ID of the task which created the version",
                        "type": ["string", "null"],
                    },
                    "user": {
                        "description": "Associated user ID",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "version_paradigm_enum": {
                "enum": ["single_version", "general"],
                "type": "string",
            },
            "version_status_enum": {
                "enum": ["draft", "committing", "committed", "published"],
                "type": "string",
            },
        },
        "properties": {
            "dataset": {
                "description": "Dataset info",
                "oneOf": [{"$ref": "#/definitions/dataset"}, {"type": "null"}],
            }
        },
        "type": "object",
    }

    def __init__(self, dataset=None, **kwargs):
        super(GetByNameResponse, self).__init__(**kwargs)
        self.dataset = dataset

    @schema_property("dataset")
    def dataset(self):
        return self._property_dataset

    @dataset.setter
    def dataset(self, value):
        if value is None:
            self._property_dataset = None
            return
        if isinstance(value, dict):
            value = Dataset.from_dict(value)
        else:
            self.assert_isinstance(value, "dataset", Dataset)
        self._property_dataset = value


class GetLabelKeywordsForRunningTaskRequest(Request):
    """
    Get the joined list of labels for all the datasets' versions in a task.
    Note that the latest committed labels are returned even if task was not completed.
    Fails if the task status is Not Started.

    :param task: Task ID
    :type task: str
    """

    _service = "datasets"
    _action = "get_label_keywords_for_running_task"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {"task": {"description": "Task ID", "type": "string"}},
        "required": ["task"],
        "type": "object",
    }

    def __init__(self, task, **kwargs):
        super(GetLabelKeywordsForRunningTaskRequest, self).__init__(**kwargs)
        self.task = task

    @schema_property("task")
    def task(self):
        return self._property_task

    @task.setter
    def task(self, value):
        if value is None:
            self._property_task = None
            return

        self.assert_isinstance(value, "task", six.string_types)
        self._property_task = value


class GetLabelKeywordsForRunningTaskResponse(Response):
    """
    Response of datasets.get_label_keywords_for_running_task endpoint.

    :param labels: List of objects with properties: name - string - label name
        count - integer - number of occurrences
    :type labels: Sequence[dict]
    """

    _service = "datasets"
    _action = "get_label_keywords_for_running_task"
    _version = "2.23"

    _schema = {
        "definitions": {},
        "properties": {
            "labels": {
                "description": (
                    "List of objects with properties:\n                    name - string - label name\n                "
                    "    count - integer - number of occurrences"
                ),
                "items": {
                    "properties": {
                        "count": {"type": "integer"},
                        "name": {"type": "string"},
                    },
                    "type": "object",
                },
                "type": ["array", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, labels=None, **kwargs):
        super(GetLabelKeywordsForRunningTaskResponse, self).__init__(**kwargs)
        self.labels = labels

    @schema_property("labels")
    def labels(self):
        return self._property_labels

    @labels.setter
    def labels(self, value):
        if value is None:
            self._property_labels = None
            return

        self.assert_isinstance(value, "labels", (list, tuple))

        self.assert_isinstance(value, "labels", (dict,), is_array=True)
        self._property_labels = value


class GetSchemaRequest(Request):
    """
    Get the aggregated fields schema for the provided set of dataset versions.
    The returned schema represents the various fields that can be used to query these versions in a Lucene query.
    In case of conflicting schema types between versions, conflicting items will not be returned.

    :param versions: Dataset version ids. The resulting schema is a merge of the
        common fields from all the specified versions.
    :type versions: Sequence[str]
    :param schema_type: Type of the schema to return (defaults to frame).
        - "frame" - all the fields for query at frame level will be returned
        - "roi" - all the fields for query at frame.rois level will be returned
        - "sources" - all the fields for query at frame.sources level will be returned
    :type schema_type: SchemaTypeEnum
    :param dataset: The ID of the dataset. Either dataset or versions should be
        specified
    :type dataset: str
    """

    _service = "datasets"
    _action = "get_schema"
    _version = "2.23"
    _schema = {
        "definitions": {
            "schema_type_enum": {"enum": ["frame", "source", "roi"], "type": "string"}
        },
        "properties": {
            "dataset": {
                "description": "The ID of the dataset. Either dataset or versions should be specified",
                "type": "string",
            },
            "schema_type": {
                "$ref": "#/definitions/schema_type_enum",
                "default": "frame",
                "description": (
                    "\nType of the schema to return (defaults to frame).\n\n"
                    '- "frame" - all the fields for '
                    "query at frame level will be returned\n\n"
                    '- "roi" - all the fields for '
                    "query at frame.rois level will be returned\n\n"
                    '- "sources" - all the fields '
                    "for query at frame.sources "
                    "level will be returned"
                ),
            },
            "versions": {
                "description": (
                    "Dataset version ids. The resulting schema is a merge of the common fields from all the "
                    "specified versions."
                ),
                "items": {"type": "string"},
                "type": "array",
            },
        },
        "required": ["versions"],
        "type": "object",
    }

    def __init__(self, versions, schema_type="frame", dataset=None, **kwargs):
        super(GetSchemaRequest, self).__init__(**kwargs)
        self.versions = versions
        self.schema_type = schema_type
        self.dataset = dataset

    @schema_property("versions")
    def versions(self):
        return self._property_versions

    @versions.setter
    def versions(self, value):
        if value is None:
            self._property_versions = None
            return

        self.assert_isinstance(value, "versions", (list, tuple))

        self.assert_isinstance(value, "versions", six.string_types, is_array=True)
        self._property_versions = value

    @schema_property("schema_type")
    def schema_type(self):
        return self._property_schema_type

    @schema_type.setter
    def schema_type(self, value):
        if value is None:
            self._property_schema_type = None
            return
        if isinstance(value, six.string_types):
            try:
                value = SchemaTypeEnum(value)
            except ValueError:
                pass
        else:
            self.assert_isinstance(value, "schema_type", enum.Enum)
        self._property_schema_type = value

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


class GetSchemaResponse(Response):
    """
    Response of datasets.get_schema endpoint.

    :param schema: Fields schema dictionary. Contains all the fields (and their
        types) for particular schema type that can be used to query the supplied
        versions in a Lucene query.
    :type schema: dict
    """

    _service = "datasets"
    _action = "get_schema"
    _version = "2.23"

    _schema = {
        "definitions": {},
        "properties": {
            "schema": {
                "description": (
                    "Fields schema dictionary. Contains all the fields (and their types) for particular schema type"
                    " that can be used to query the supplied versions in a Lucene query."
                ),
                "type": ["object", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, schema=None, **kwargs):
        super(GetSchemaResponse, self).__init__(**kwargs)
        self.schema = schema

    @schema_property("schema")
    def schema(self):
        return self._property_schema

    @schema.setter
    def schema(self, value):
        if value is None:
            self._property_schema = None
            return

        self.assert_isinstance(value, "schema", (dict,))
        self._property_schema = value


class GetSchemaKeysRequest(Request):
    """
    Get the field names that can be used in lucene query for the given dataset versions

    :param versions: The IDs of the versions. Either dataset or versions should be
        specified
    :type versions: Sequence[str]
    :param dataset: The ID of the dataset. Either dataset or versions should be
        specified
    :type dataset: str
    """

    _service = "datasets"
    _action = "get_schema_keys"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {
            "dataset": {
                "description": "The ID of the dataset. Either dataset or versions should be specified",
                "type": "string",
            },
            "versions": {
                "description": "The IDs of the versions. Either dataset or versions should be specified",
                "items": {"type": "string"},
                "type": "array",
            },
        },
        "required": ["versions"],
        "type": "object",
    }

    def __init__(self, versions, dataset=None, **kwargs):
        super(GetSchemaKeysRequest, self).__init__(**kwargs)
        self.versions = versions
        self.dataset = dataset

    @schema_property("versions")
    def versions(self):
        return self._property_versions

    @versions.setter
    def versions(self, value):
        if value is None:
            self._property_versions = None
            return

        self.assert_isinstance(value, "versions", (list, tuple))

        self.assert_isinstance(value, "versions", six.string_types, is_array=True)
        self._property_versions = value

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


class GetSchemaKeysResponse(Response):
    """
    Response of datasets.get_schema_keys endpoint.

    :param frame_keys: Frame level fields
    :type frame_keys: Sequence[str]
    :param roi_keys: ROI level fields
    :type roi_keys: Sequence[str]
    :param source_keys: Source level fields
    :type source_keys: Sequence[str]
    """

    _service = "datasets"
    _action = "get_schema_keys"
    _version = "2.23"

    _schema = {
        "definitions": {},
        "properties": {
            "frame_keys": {
                "description": "Frame level fields",
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
            "roi_keys": {
                "description": "ROI level fields",
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
            "source_keys": {
                "description": "Source level fields",
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, frame_keys=None, roi_keys=None, source_keys=None, **kwargs):
        super(GetSchemaKeysResponse, self).__init__(**kwargs)
        self.frame_keys = frame_keys
        self.roi_keys = roi_keys
        self.source_keys = source_keys

    @schema_property("frame_keys")
    def frame_keys(self):
        return self._property_frame_keys

    @frame_keys.setter
    def frame_keys(self, value):
        if value is None:
            self._property_frame_keys = None
            return

        self.assert_isinstance(value, "frame_keys", (list, tuple))

        self.assert_isinstance(value, "frame_keys", six.string_types, is_array=True)
        self._property_frame_keys = value

    @schema_property("roi_keys")
    def roi_keys(self):
        return self._property_roi_keys

    @roi_keys.setter
    def roi_keys(self, value):
        if value is None:
            self._property_roi_keys = None
            return

        self.assert_isinstance(value, "roi_keys", (list, tuple))

        self.assert_isinstance(value, "roi_keys", six.string_types, is_array=True)
        self._property_roi_keys = value

    @schema_property("source_keys")
    def source_keys(self):
        return self._property_source_keys

    @source_keys.setter
    def source_keys(self, value):
        if value is None:
            self._property_source_keys = None
            return

        self.assert_isinstance(value, "source_keys", (list, tuple))

        self.assert_isinstance(value, "source_keys", six.string_types, is_array=True)
        self._property_source_keys = value


class GetSnippetsRequest(Request):
    """
    Return first frame of for unique URIs in the dataset

    :param version: Dataset version ID. If not provided, returns sources used by
        all versions.
    :type version: str
    :param dataset: Dataset ID
    :type dataset: str
    :param max_count: Number of sources to return. default=100, Optional
    :type max_count: int
    """

    _service = "datasets"
    _action = "get_snippets"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {
            "dataset": {"description": "Dataset ID", "type": "string"},
            "max_count": {
                "default": 100,
                "description": "Number of sources to return. default=100, Optional",
                "type": "integer",
            },
            "version": {
                "description": "Dataset version ID. If not provided, returns sources used by all versions.",
                "type": ["string", "null"],
            },
        },
        "required": ["dataset"],
        "type": "object",
    }

    def __init__(self, dataset, version=None, max_count=100, **kwargs):
        super(GetSnippetsRequest, self).__init__(**kwargs)
        self.version = version
        self.dataset = dataset
        self.max_count = max_count

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


class GetSnippetsResponse(Response):
    """
    Response of datasets.get_snippets endpoint.

    :param snippets: list of snippets
    :type snippets: Sequence[Frame]
    """

    _service = "datasets"
    _action = "get_snippets"
    _version = "2.23"

    _schema = {
        "definitions": {
            "frame": {
                "properties": {
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
                    "id": {
                        "description": (
                            "Frame id. Must be unique within the dataset's version. If already exists, will cause"
                            " existing frame to be updated"
                        ),
                        "type": ["string", "null"],
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
                    "rois": {
                        "description": "Frame regions of interest",
                        "items": {"$ref": "#/definitions/roi"},
                        "type": ["array", "null"],
                    },
                    "sources": {
                        "description": "Sources of this frame",
                        "items": {"$ref": "#/definitions/source"},
                        "type": "array",
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
                },
                "required": ["sources"],
                "type": "object",
            },
            "mask": {
                "properties": {
                    "content_type": {
                        "description": "Content type (e.g. 'image/jpeg', 'image/png')",
                        "type": "string",
                    },
                    "height": {"description": "Height in pixels", "type": "integer"},
                    "id": {
                        "description": "unique ID (in this frame)",
                        "type": "string",
                    },
                    "timestamp": {
                        "default": 0,
                        "description": (
                            "Timestamp in the source data (for video content. for images, this value should be 0)"
                        ),
                        "type": "integer",
                    },
                    "uri": {"description": "Data URI", "type": "string"},
                    "width": {"description": "Width in pixels", "type": "integer"},
                },
                "required": ["id", "uri"],
                "type": "object",
            },
            "preview": {
                "properties": {
                    "content_type": {
                        "description": "Content type (e.g. 'image/jpeg', 'image/png')",
                        "type": "string",
                    },
                    "height": {
                        "description": "Height in pixels",
                        "type": "integer",
                    },
                    "timestamp": {
                        "default": 0,
                        "description": (
                            "Timestamp in the source data (for video content. for images, this value should be 0)"
                        ),
                        "type": "integer",
                    },
                    "uri": {"description": "Data URI", "type": "string"},
                    "width": {"description": "Width in pixels", "type": "integer"},
                },
                "required": ["uri"],
                "type": "object",
            },
            "roi": {
                "properties": {
                    "confidence": {
                        "description": "ROI confidence",
                        "type": "number",
                    },
                    "id": {"description": "ROI id", "type": ["string", "null"]},
                    "label": {
                        "description": "ROI labels",
                        "items": {"type": "string"},
                        "type": "array",
                    },
                    "mask": {
                        "$ref": "#/definitions/roi_mask",
                        "description": "Mask info for this ROI",
                    },
                    "meta": {
                        "additionalProperties": True,
                        "description": "Additional metadata dictionary for the roi",
                        "type": "object",
                    },
                    "poly": {
                        "description": "ROI polygon (x0, y0, ..., xn, yn)",
                        "items": {"type": "number"},
                        "type": "array",
                    },
                    "sources": {
                        "description": "Source ID",
                        "items": {"type": "string"},
                        "type": "array",
                    },
                },
                "required": ["label"],
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
                        "type": "string",
                    },
                    "height": {
                        "description": "Height in pixels",
                        "type": "integer",
                    },
                    "id": {
                        "description": "Source unique ID within this DatasetVersion",
                        "type": "string",
                    },
                    "masks": {"items": {"$ref": "#/definitions/mask"}, "type": "array"},
                    "meta": {
                        "additionalProperties": True,
                        "description": "Additional metadata dictionary for the source",
                        "type": "object",
                    },
                    "preview": {"$ref": "#/definitions/preview"},
                    "timestamp": {
                        "default": 0,
                        "description": (
                            "Timestamp in the source data (for video content. for images, this value should be 0)"
                        ),
                        "type": "integer",
                    },
                    "uri": {"description": "Source data URI", "type": "string"},
                    "width": {"description": "Width in pixels", "type": "integer"},
                },
                "required": ["id", "uri"],
                "type": "object",
            },
        },
        "properties": {
            "snippets": {
                "description": "list of snippets",
                "items": {"$ref": "#/definitions/frame"},
                "type": ["array", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, snippets=None, **kwargs):
        super(GetSnippetsResponse, self).__init__(**kwargs)
        self.snippets = snippets

    @schema_property("snippets")
    def snippets(self):
        return self._property_snippets

    @snippets.setter
    def snippets(self, value):
        if value is None:
            self._property_snippets = None
            return

        self.assert_isinstance(value, "snippets", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [Frame.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "snippets", Frame, is_array=True)
        self._property_snippets = value


class GetSourceIdsRequest(Request):
    """
    Get unique source ids from the given dataset version

    :param version: Dataset version ID. If not provided, returns sources used by
        all versions.
    :type version: str
    :param dataset: Dataset ID
    :type dataset: str
    :param max_count: Number of sources to return. default=100, Optional
    :type max_count: int
    """

    _service = "datasets"
    _action = "get_source_ids"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {
            "dataset": {"description": "Dataset ID", "type": "string"},
            "max_count": {
                "default": 100,
                "description": "Number of sources to return. default=100, Optional",
                "type": "integer",
            },
            "version": {
                "description": "Dataset version ID. If not provided, returns sources used by all versions.",
                "type": ["string", "null"],
            },
        },
        "required": ["dataset"],
        "type": "object",
    }

    def __init__(self, dataset, version=None, max_count=100, **kwargs):
        super(GetSourceIdsRequest, self).__init__(**kwargs)
        self.version = version
        self.dataset = dataset
        self.max_count = max_count

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


class GetSourceIdsResponse(Response):
    """
    Response of datasets.get_source_ids endpoint.

    :param source_ids: Unique source ids for the dataset version
    :type source_ids: Sequence[str]
    """

    _service = "datasets"
    _action = "get_source_ids"
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
        super(GetSourceIdsResponse, self).__init__(**kwargs)
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


class GetSourcesRequest(Request):
    """
    Get all sources used by frames in the given dataset version

    :param version: Dataset version ID. If not provided, returns sources used by
        all versions.
    :type version: str
    :param dataset: Dataset ID
    :type dataset: str
    :param max_count: Number of sources to return. default=100, Optional
    :type max_count: int
    """

    _service = "datasets"
    _action = "get_sources"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {
            "dataset": {"description": "Dataset ID", "type": "string"},
            "max_count": {
                "default": 100,
                "description": "Number of sources to return. default=100, Optional",
                "type": "integer",
            },
            "version": {
                "description": "Dataset version ID. If not provided, returns sources used by all versions.",
                "type": ["string", "null"],
            },
        },
        "required": ["dataset"],
        "type": "object",
    }

    def __init__(self, dataset, version=None, max_count=100, **kwargs):
        super(GetSourcesRequest, self).__init__(**kwargs)
        self.version = version
        self.dataset = dataset
        self.max_count = max_count

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


class GetSourcesResponse(Response):
    """
    Response of datasets.get_sources endpoint.

    :param sources: Mapping of source URL to first frame_id of the source
    :type sources: dict
    """

    _service = "datasets"
    _action = "get_sources"
    _version = "2.23"

    _schema = {
        "definitions": {},
        "properties": {
            "sources": {
                "additionalProperties": True,
                "description": "Mapping of source URL to first frame_id of the source",
                "type": ["object", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, sources=None, **kwargs):
        super(GetSourcesResponse, self).__init__(**kwargs)
        self.sources = sources

    @schema_property("sources")
    def sources(self):
        return self._property_sources

    @sources.setter
    def sources(self, value):
        if value is None:
            self._property_sources = None
            return

        self.assert_isinstance(value, "sources", (dict,))
        self._property_sources = value


class GetStatsRequest(Request):
    """
    Get labels' stats for a dataset version

    :param version: Dataset version ID
    :type version: str
    """

    _service = "datasets"
    _action = "get_stats"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {
            "version": {"description": "Dataset version ID", "type": "string"}
        },
        "required": ["version"],
        "type": "object",
    }

    def __init__(self, version, **kwargs):
        super(GetStatsRequest, self).__init__(**kwargs)
        self.version = version

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


class GetStatsResponse(Response):
    """
    Response of datasets.get_stats endpoint.

    :param content_types: List of statistics for each content type
    :type content_types: Sequence[StatCount]
    :param frames: List of statistics for each frame (annotated/unannotated)
    :type frames: Sequence[StatCount]
    :param labels: List of statistics for each label
    :type labels: Sequence[StatCount]
    """

    _service = "datasets"
    _action = "get_stats"
    _version = "2.23"

    _schema = {
        "definitions": {
            "stat_count": {
                "properties": {
                    "count": {
                        "description": "Item name",
                        "type": ["integer", "null"],
                    },
                    "name": {
                        "description": "Number of appearances",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            }
        },
        "properties": {
            "content_types": {
                "description": "List of statistics for each content type",
                "items": {"$ref": "#/definitions/stat_count"},
                "type": ["array", "null"],
            },
            "frames": {
                "description": "List of statistics for each frame (annotated/unannotated)",
                "items": {"$ref": "#/definitions/stat_count"},
                "type": ["array", "null"],
            },
            "labels": {
                "description": "List of statistics for each label",
                "items": {"$ref": "#/definitions/stat_count"},
                "type": ["array", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, content_types=None, frames=None, labels=None, **kwargs):
        super(GetStatsResponse, self).__init__(**kwargs)
        self.content_types = content_types
        self.frames = frames
        self.labels = labels

    @schema_property("content_types")
    def content_types(self):
        return self._property_content_types

    @content_types.setter
    def content_types(self, value):
        if value is None:
            self._property_content_types = None
            return

        self.assert_isinstance(value, "content_types", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [
                StatCount.from_dict(v) if isinstance(v, dict) else v for v in value
            ]
        else:
            self.assert_isinstance(value, "content_types", StatCount, is_array=True)
        self._property_content_types = value

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
            value = [
                StatCount.from_dict(v) if isinstance(v, dict) else v for v in value
            ]
        else:
            self.assert_isinstance(value, "frames", StatCount, is_array=True)
        self._property_frames = value

    @schema_property("labels")
    def labels(self):
        return self._property_labels

    @labels.setter
    def labels(self, value):
        if value is None:
            self._property_labels = None
            return

        self.assert_isinstance(value, "labels", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [
                StatCount.from_dict(v) if isinstance(v, dict) else v for v in value
            ]
        else:
            self.assert_isinstance(value, "labels", StatCount, is_array=True)
        self._property_labels = value


class GetTagsRequest(Request):
    """
    Get user and system tags used for the specified datasets

    :param include_system: If set to 'true' then the list of the system tags is
        also returned. The default value is 'false'
    :type include_system: bool
    :param datasets: The list of datasets for which the tags are collected. If not
        passed or empty then tags from all the datasets collected
    :type datasets: Sequence[str]
    """

    _service = "datasets"
    _action = "get_tags"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {
            "datasets": {
                "description": (
                    "The list of datasets for which the tags are collected. If not passed or empty then tags from "
                    "all the datasets collected"
                ),
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
            "include_system": {
                "default": False,
                "description": (
                    "If set to 'true' then the list of the system tags is also returned. The default value is 'false'"
                ),
                "type": ["boolean", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, include_system=False, datasets=None, **kwargs):
        super(GetTagsRequest, self).__init__(**kwargs)
        self.include_system = include_system
        self.datasets = datasets

    @schema_property("include_system")
    def include_system(self):
        return self._property_include_system

    @include_system.setter
    def include_system(self, value):
        if value is None:
            self._property_include_system = None
            return

        self.assert_isinstance(value, "include_system", (bool,))
        self._property_include_system = value

    @schema_property("datasets")
    def datasets(self):
        return self._property_datasets

    @datasets.setter
    def datasets(self, value):
        if value is None:
            self._property_datasets = None
            return

        self.assert_isinstance(value, "datasets", (list, tuple))

        self.assert_isinstance(value, "datasets", six.string_types, is_array=True)
        self._property_datasets = value


class GetTagsResponse(Response):
    """
    Response of datasets.get_tags endpoint.

    :param tags: The list of unique tag values
    :type tags: Sequence[str]
    :param system_tags: The list of unique system tag values. Returned only if
        'include_system' is set to 'true' in the request
    :type system_tags: Sequence[str]
    """

    _service = "datasets"
    _action = "get_tags"
    _version = "2.23"

    _schema = {
        "definitions": {},
        "properties": {
            "system_tags": {
                "description": (
                    "The list of unique system tag values. Returned only if 'include_system' is set to "
                    "'true' in the request"
                ),
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
            "tags": {
                "description": "The list of unique tag values",
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, tags=None, system_tags=None, **kwargs):
        super(GetTagsResponse, self).__init__(**kwargs)
        self.tags = tags
        self.system_tags = system_tags

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


class GetVersionsRequest(Request):
    """
    Gets the version tree of a dataset.

    :param start_from: Dataset ID
    :type start_from: str
    :param dataset: Get versions starting from this time
    :type dataset: str
    :param only_fields: List of version fields to fetch
    :type only_fields: Sequence[str]
    :param versions: List of version IDs to fetch
    :type versions: Sequence[str]
    :param only_published: Return only published version.
    :type only_published: bool
    :param page: Page number, returns a specific page out of the result list of
        datasets.
    :type page: int
    :param page_size: Page size, specifies the number of results returned in each
        page (last page may contain fewer results)
    :type page_size: int
    :param order_by: List of field names to order by. When search_text is used,
        '@text_score' can be used as a field representing the text score of returned
        documents. Use '-' prefix to specify descending order. Optional, recommended
        when using page. Defaults to [created].
    :type order_by: Sequence[str]
    :param search_text: Free text search query
    :type search_text: str
    :param tags: User-defined tags filter. Use '-' for exclusion
    :type tags: Sequence[str]
    :param system_tags: System tags filter. Use '-' for exclusion
    :type system_tags: Sequence[str]
    """

    _service = "datasets"
    _action = "get_versions"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {
            "dataset": {
                "description": "Get versions starting from this time",
                "type": "string",
            },
            "only_fields": {
                "description": "List of version fields to fetch",
                "items": {"type": "string"},
                "type": "array",
            },
            "only_published": {
                "default": True,
                "description": "Return only published version.",
                "type": "boolean",
            },
            "order_by": {
                "description": (
                    "List of field names to order by. When search_text is used, '@text_score' can be used as a field"
                    " representing the text score of returned documents. Use '-' prefix to specify descending order."
                    " Optional, recommended when using page. Defaults to [created]."
                ),
                "items": {"type": "string"},
                "type": "array",
            },
            "page": {
                "description": "Page number, returns a specific page out of the result list of datasets.",
                "minimum": 0,
                "type": "integer",
            },
            "page_size": {
                "description": (
                    "Page size, specifies the number of results returned in each page (last page may contain fewer "
                    "results)"
                ),
                "minimum": 1,
                "type": "integer",
            },
            "search_text": {"description": "Free text search query", "type": "string"},
            "start_from": {"description": "Dataset ID", "type": ["string", "null"]},
            "system_tags": {
                "description": "System tags filter. Use '-' for exclusion",
                "items": {"type": "string"},
                "type": "array",
            },
            "tags": {
                "description": "User-defined tags filter. Use '-' for exclusion",
                "items": {"type": "string"},
                "type": "array",
            },
            "versions": {
                "description": "List of version IDs to fetch",
                "items": {"type": "string"},
                "type": "array",
            },
        },
        "required": ["dataset"],
        "type": "object",
    }

    def __init__(
        self,
        dataset,
        start_from=None,
        only_fields=None,
        versions=None,
        only_published=True,
        page=None,
        page_size=None,
        order_by=None,
        search_text=None,
        tags=None,
        system_tags=None,
        **kwargs
    ):
        super(GetVersionsRequest, self).__init__(**kwargs)
        self.start_from = start_from
        self.dataset = dataset
        self.only_fields = only_fields
        self.versions = versions
        self.only_published = only_published
        self.page = page
        self.page_size = page_size
        self.order_by = order_by
        self.search_text = search_text
        self.tags = tags
        self.system_tags = system_tags

    @schema_property("start_from")
    def start_from(self):
        return self._property_start_from

    @start_from.setter
    def start_from(self, value):
        if value is None:
            self._property_start_from = None
            return

        self.assert_isinstance(value, "start_from", six.string_types)
        self._property_start_from = value

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

    @schema_property("versions")
    def versions(self):
        return self._property_versions

    @versions.setter
    def versions(self, value):
        if value is None:
            self._property_versions = None
            return

        self.assert_isinstance(value, "versions", (list, tuple))

        self.assert_isinstance(value, "versions", six.string_types, is_array=True)
        self._property_versions = value

    @schema_property("only_published")
    def only_published(self):
        return self._property_only_published

    @only_published.setter
    def only_published(self, value):
        if value is None:
            self._property_only_published = None
            return

        self.assert_isinstance(value, "only_published", (bool,))
        self._property_only_published = value

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


class GetVersionsResponse(Response):
    """
    Response of datasets.get_versions endpoint.

    :param versions: List of versions
    :type versions: Sequence[Version]
    """

    _service = "datasets"
    _action = "get_versions"
    _version = "2.23"

    _schema = {
        "definitions": {
            "stat_count": {
                "properties": {
                    "count": {
                        "description": "Item name",
                        "type": ["integer", "null"],
                    },
                    "name": {
                        "description": "Number of appearances",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "statistics": {
                "properties": {
                    "content_types": {
                        "items": {
                            "$ref": "#/definitions/stat_count",
                            "description": (
                                "List of content type counts for the version (e.g.\n                    'image/jpeg',"
                                " 'image/png', 'video/mp4')"
                            ),
                        },
                        "type": ["array", "null"],
                    },
                    "frames": {
                        "items": {
                            "$ref": "#/definitions/stat_count",
                            "description": (
                                "List of frame counts, indicating the\n                    type of frames included in"
                                " the version (annotated/"
                            ),
                        },
                        "type": ["array", "null"],
                    },
                    "labels": {
                        "items": {
                            "$ref": "#/definitions/stat_count",
                            "description": (
                                "List of labels' counts,\n                    indicating the categories included in the"
                                " version"
                            ),
                        },
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            },
            "version": {
                "properties": {
                    "comment": {
                        "description": "Version comment",
                        "type": ["string", "null"],
                    },
                    "committed": {
                        "description": "Commit time",
                        "format": "date-time",
                        "type": ["string", "null"],
                    },
                    "committed_frames_ts": {
                        "description": "Timestamp of last committed frame",
                        "type": ["number", "null"],
                    },
                    "committed_rois_ts": {
                        "description": "Timestamp of last committed ROI",
                        "type": ["number", "null"],
                    },
                    "company": {
                        "description": "Company ID",
                        "type": ["string", "null"],
                    },
                    "created": {
                        "description": "Version creation time (UTC) ",
                        "format": "date-time",
                        "type": ["string", "null"],
                    },
                    "dataset": {
                        "description": "Datset ID",
                        "type": ["string", "null"],
                    },
                    "es_index": {
                        "description": "Name of elasticsearch index",
                        "type": ["string", "null"],
                    },
                    "id": {"description": "Version ID", "type": ["string", "null"]},
                    "last_frames_update": {
                        "description": "Last time version was created, committed or frames were updated or saved",
                        "format": "date-time",
                        "type": ["string", "null"],
                    },
                    "metadata": {
                        "additionalProperties": True,
                        "description": "User-provided metadata",
                        "type": ["object", "null"],
                    },
                    "name": {
                        "description": "Version name",
                        "type": ["string", "null"],
                    },
                    "parent": {
                        "description": "Version parent ID",
                        "type": ["string", "null"],
                    },
                    "published": {
                        "description": "Publish time",
                        "format": "date-time",
                        "type": ["string", "null"],
                    },
                    "stats": {
                        "description": "Version statistics",
                        "oneOf": [
                            {"$ref": "#/definitions/statistics"},
                            {"type": "null"},
                        ],
                    },
                    "status": {
                        "description": "Version status",
                        "oneOf": [
                            {"$ref": "#/definitions/version_status_enum"},
                            {"type": "null"},
                        ],
                    },
                    "system_tags": {
                        "description": (
                            "List of system tags. This field is reserved for system use, please don't use it."
                        ),
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                    "tags": {
                        "description": "List of user-defined tags",
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                    "task": {
                        "description": "Task ID of the task which created the version",
                        "type": ["string", "null"],
                    },
                    "user": {
                        "description": "Associated user ID",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "version_status_enum": {
                "enum": ["draft", "committing", "committed", "published"],
                "type": "string",
            },
        },
        "properties": {
            "versions": {
                "description": "List of versions",
                "items": {"$ref": "#/definitions/version"},
                "type": ["array", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, versions=None, **kwargs):
        super(GetVersionsResponse, self).__init__(**kwargs)
        self.versions = versions

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
            value = [Version.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "versions", Version, is_array=True)
        self._property_versions = value


class MoveRequest(Request):
    """
    Move datasets to a project

    :param ids: Datasets to move
    :type ids: Sequence[str]
    :param project: Target project ID. If not provided, `project_name` must be
        provided. Use null for the root project
    :type project: str
    :param project_name: Target project name. If provided and a project with this
        name does not exist, a new project will be created. If not provided, `project`
        must be provided.
    :type project_name: str
    """

    _service = "datasets"
    _action = "move"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {
            "ids": {
                "description": "Datasets to move",
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
    Response of datasets.move endpoint.

    """

    _service = "datasets"
    _action = "move"
    _version = "2.23"

    _schema = {"additionalProperties": True, "definitions": {}, "type": "object"}


class PublishAndCreateChildVersionRequest(Request):
    """
    Publishes the specified version and creates a draft child version for it

    :param dataset: Dataset ID
    :type dataset: str
    :param version: Draft version ID
    :type version: str
    :param publish_name: New name for the published version. The default value is
        'snapshot <date-time>'
    :type publish_name: str
    :param publish_comment: New comment for the published version. The default
        value is 'published at <date-time> by <user>'
    :type publish_comment: str
    :param publish_metadata: User-specified metadata object for the published
        version. Keys must not include '$' and '.'.
    :type publish_metadata: dict
    :param child_name: Name for the child version. If not provided then the name of
        the parent version is taken
    :type child_name: str
    :param child_comment: Comment for the child version
    :type child_comment: str
    :param child_metadata: User-specified metadata object for the child version.
        Keys must not include '$' and '.'.
    :type child_metadata: dict
    :param publish_tags: The new user-defined tags for the published version. If
        not passed then the parent version tags are used
    :type publish_tags: Sequence[str]
    :param publish_system_tags: The new system tags for the published version. If
        not passed then the parent version system tags are used
    :type publish_system_tags: Sequence[str]
    :param child_tags: The new user tags for the child version. If not passed then
        the parent version tags are used
    :type child_tags: Sequence[str]
    :param child_system_tags: The new system tags for the child version. If not
        passed then the parent version system tags are used
    :type child_system_tags: Sequence[str]
    """

    _service = "datasets"
    _action = "publish_and_create_child_version"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {
            "child_comment": {
                "description": "Comment for the child version",
                "type": "string",
            },
            "child_metadata": {
                "additionalProperties": True,
                "description": (
                    "User-specified metadata object for the child version. Keys must not include '$' and '.'."
                ),
                "type": "object",
            },
            "child_name": {
                "description": (
                    "Name for the child version. If not provided then the name of the parent version is taken"
                ),
                "type": "string",
            },
            "child_system_tags": {
                "description": (
                    "The new system tags for the child version. If not passed then the parent version system "
                    "tags are used"
                ),
                "items": {"type": "string"},
                "type": "array",
            },
            "child_tags": {
                "description": (
                    "The new user tags for the child version. If not passed then the parent version tags are used"
                ),
                "items": {"type": "string"},
                "type": "array",
            },
            "dataset": {"description": "Dataset ID", "type": "string"},
            "publish_comment": {
                "description": (
                    "New comment for the published version. The default value is 'published at <date-time> by <user>'"
                ),
                "type": "string",
            },
            "publish_metadata": {
                "additionalProperties": True,
                "description": (
                    "User-specified metadata object for the published version. Keys must not include '$' and '.'."
                ),
                "type": "object",
            },
            "publish_name": {
                "description": "New name for the published version. The default value is 'snapshot <date-time>'",
                "type": "string",
            },
            "publish_system_tags": {
                "description": (
                    "The new system tags for the published version. If not passed then the parent version system tags "
                    "are used"
                ),
                "items": {"type": "string"},
                "type": "array",
            },
            "publish_tags": {
                "description": (
                    "The new user-defined tags for the published version. If not passed then the parent "
                    "version tags are used"
                ),
                "items": {"type": "string"},
                "type": "array",
            },
            "version": {"description": "Draft version ID", "type": "string"},
        },
        "required": ["dataset", "version"],
        "type": "object",
    }

    def __init__(
        self,
        dataset,
        version,
        publish_name=None,
        publish_comment=None,
        publish_metadata=None,
        child_name=None,
        child_comment=None,
        child_metadata=None,
        publish_tags=None,
        publish_system_tags=None,
        child_tags=None,
        child_system_tags=None,
        **kwargs
    ):
        super(PublishAndCreateChildVersionRequest, self).__init__(**kwargs)
        self.dataset = dataset
        self.version = version
        self.publish_name = publish_name
        self.publish_comment = publish_comment
        self.publish_metadata = publish_metadata
        self.child_name = child_name
        self.child_comment = child_comment
        self.child_metadata = child_metadata
        self.publish_tags = publish_tags
        self.publish_system_tags = publish_system_tags
        self.child_tags = child_tags
        self.child_system_tags = child_system_tags

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

    @schema_property("publish_name")
    def publish_name(self):
        return self._property_publish_name

    @publish_name.setter
    def publish_name(self, value):
        if value is None:
            self._property_publish_name = None
            return

        self.assert_isinstance(value, "publish_name", six.string_types)
        self._property_publish_name = value

    @schema_property("publish_comment")
    def publish_comment(self):
        return self._property_publish_comment

    @publish_comment.setter
    def publish_comment(self, value):
        if value is None:
            self._property_publish_comment = None
            return

        self.assert_isinstance(value, "publish_comment", six.string_types)
        self._property_publish_comment = value

    @schema_property("publish_metadata")
    def publish_metadata(self):
        return self._property_publish_metadata

    @publish_metadata.setter
    def publish_metadata(self, value):
        if value is None:
            self._property_publish_metadata = None
            return

        self.assert_isinstance(value, "publish_metadata", (dict,))
        self._property_publish_metadata = value

    @schema_property("child_name")
    def child_name(self):
        return self._property_child_name

    @child_name.setter
    def child_name(self, value):
        if value is None:
            self._property_child_name = None
            return

        self.assert_isinstance(value, "child_name", six.string_types)
        self._property_child_name = value

    @schema_property("child_comment")
    def child_comment(self):
        return self._property_child_comment

    @child_comment.setter
    def child_comment(self, value):
        if value is None:
            self._property_child_comment = None
            return

        self.assert_isinstance(value, "child_comment", six.string_types)
        self._property_child_comment = value

    @schema_property("child_metadata")
    def child_metadata(self):
        return self._property_child_metadata

    @child_metadata.setter
    def child_metadata(self, value):
        if value is None:
            self._property_child_metadata = None
            return

        self.assert_isinstance(value, "child_metadata", (dict,))
        self._property_child_metadata = value

    @schema_property("publish_tags")
    def publish_tags(self):
        return self._property_publish_tags

    @publish_tags.setter
    def publish_tags(self, value):
        if value is None:
            self._property_publish_tags = None
            return

        self.assert_isinstance(value, "publish_tags", (list, tuple))

        self.assert_isinstance(value, "publish_tags", six.string_types, is_array=True)
        self._property_publish_tags = value

    @schema_property("publish_system_tags")
    def publish_system_tags(self):
        return self._property_publish_system_tags

    @publish_system_tags.setter
    def publish_system_tags(self, value):
        if value is None:
            self._property_publish_system_tags = None
            return

        self.assert_isinstance(value, "publish_system_tags", (list, tuple))

        self.assert_isinstance(
            value, "publish_system_tags", six.string_types, is_array=True
        )
        self._property_publish_system_tags = value

    @schema_property("child_tags")
    def child_tags(self):
        return self._property_child_tags

    @child_tags.setter
    def child_tags(self, value):
        if value is None:
            self._property_child_tags = None
            return

        self.assert_isinstance(value, "child_tags", (list, tuple))

        self.assert_isinstance(value, "child_tags", six.string_types, is_array=True)
        self._property_child_tags = value

    @schema_property("child_system_tags")
    def child_system_tags(self):
        return self._property_child_system_tags

    @child_system_tags.setter
    def child_system_tags(self, value):
        if value is None:
            self._property_child_system_tags = None
            return

        self.assert_isinstance(value, "child_system_tags", (list, tuple))

        self.assert_isinstance(
            value, "child_system_tags", six.string_types, is_array=True
        )
        self._property_child_system_tags = value


class PublishAndCreateChildVersionResponse(Response):
    """
    Response of datasets.publish_and_create_child_version endpoint.

    :param id: ID of the child version
    :type id: str
    """

    _service = "datasets"
    _action = "publish_and_create_child_version"
    _version = "2.23"

    _schema = {
        "definitions": {},
        "properties": {
            "id": {"description": "ID of the child version", "type": ["string", "null"]}
        },
        "type": "object",
    }

    def __init__(self, id=None, **kwargs):
        super(PublishAndCreateChildVersionResponse, self).__init__(**kwargs)
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


class PublishVersionRequest(Request):
    """
    Publish a draft version.

    :param version: Draft version ID
    :type version: str
    :param force: Ignore ongoing annotation tasks with this version as input
    :type force: bool
    :param publishing_task: ID of an in-progress annotation task calling this
        endpoint. Versions which are used as input of in-progress annotation tasks can
        only be published if there is only one such task and its ID is sent in this
        field. This is required if one exists.
    :type publishing_task: str
    """

    _service = "datasets"
    _action = "publish_version"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {
            "force": {
                "default": False,
                "description": "Ignore ongoing annotation tasks with this version as input",
                "type": "boolean",
            },
            "publishing_task": {
                "description": (
                    "ID of an in-progress annotation task calling this endpoint.\n                    Versions which"
                    " are used as input of in-progress annotation tasks can only be published\n                    if"
                    " there is only one such task and its ID is sent in this field.\n                    This is"
                    " required if one exists."
                ),
                "type": "string",
            },
            "version": {"description": "Draft version ID", "type": "string"},
        },
        "required": ["version"],
        "type": "object",
    }

    def __init__(self, version, force=False, publishing_task=None, **kwargs):
        super(PublishVersionRequest, self).__init__(**kwargs)
        self.version = version
        self.force = force
        self.publishing_task = publishing_task

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

    @schema_property("publishing_task")
    def publishing_task(self):
        return self._property_publishing_task

    @publishing_task.setter
    def publishing_task(self, value):
        if value is None:
            self._property_publishing_task = None
            return

        self.assert_isinstance(value, "publishing_task", six.string_types)
        self._property_publishing_task = value


class PublishVersionResponse(Response):
    """
    Response of datasets.publish_version endpoint.

    """

    _service = "datasets"
    _action = "publish_version"
    _version = "2.23"

    _schema = {"definitions": {}, "properties": {}, "type": "object"}


class SaveFramesRequest(Request):
    """
    Save frames into a draft version. Frame IDs, if sent, will be ignored, and every frame will be assigned a new ID.

    :param version: Draft version ID
    :type version: str
    :param frames: Frames to save
    :type frames: Sequence[Frame]
    """

    _service = "datasets"
    _action = "save_frames"
    _version = "2.23"
    _schema = {
        "definitions": {
            "frame": {
                "properties": {
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
                    "id": {
                        "description": (
                            "Frame id. Must be unique within the dataset's version. If already exists, will cause"
                            " existing frame to be updated"
                        ),
                        "type": ["string", "null"],
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
                    "rois": {
                        "description": "Frame regions of interest",
                        "items": {"$ref": "#/definitions/roi"},
                        "type": ["array", "null"],
                    },
                    "sources": {
                        "description": "Sources of this frame",
                        "items": {"$ref": "#/definitions/source"},
                        "type": "array",
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
                },
                "required": ["sources"],
                "type": "object",
            },
            "mask": {
                "properties": {
                    "content_type": {
                        "description": "Content type (e.g. 'image/jpeg', 'image/png')",
                        "type": "string",
                    },
                    "height": {"description": "Height in pixels", "type": "integer"},
                    "id": {
                        "description": "unique ID (in this frame)",
                        "type": "string",
                    },
                    "timestamp": {
                        "default": 0,
                        "description": (
                            "Timestamp in the source data (for video content. for images, this value should be 0)"
                        ),
                        "type": "integer",
                    },
                    "uri": {"description": "Data URI", "type": "string"},
                    "width": {"description": "Width in pixels", "type": "integer"},
                },
                "required": ["id", "uri"],
                "type": "object",
            },
            "preview": {
                "properties": {
                    "content_type": {
                        "description": "Content type (e.g. 'image/jpeg', 'image/png')",
                        "type": "string",
                    },
                    "height": {
                        "description": "Height in pixels",
                        "type": "integer",
                    },
                    "timestamp": {
                        "default": 0,
                        "description": (
                            "Timestamp in the source data (for video content. for images, this value should be 0)"
                        ),
                        "type": "integer",
                    },
                    "uri": {"description": "Data URI", "type": "string"},
                    "width": {"description": "Width in pixels", "type": "integer"},
                },
                "required": ["uri"],
                "type": "object",
            },
            "roi": {
                "properties": {
                    "confidence": {
                        "description": "ROI confidence",
                        "type": "number",
                    },
                    "id": {"description": "ROI id", "type": ["string", "null"]},
                    "label": {
                        "description": "ROI labels",
                        "items": {"type": "string"},
                        "type": "array",
                    },
                    "mask": {
                        "$ref": "#/definitions/roi_mask",
                        "description": "Mask info for this ROI",
                    },
                    "meta": {
                        "additionalProperties": True,
                        "description": "Additional metadata dictionary for the roi",
                        "type": "object",
                    },
                    "poly": {
                        "description": "ROI polygon (x0, y0, ..., xn, yn)",
                        "items": {"type": "number"},
                        "type": "array",
                    },
                    "sources": {
                        "description": "Source ID",
                        "items": {"type": "string"},
                        "type": "array",
                    },
                },
                "required": ["label"],
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
                        "type": "string",
                    },
                    "height": {
                        "description": "Height in pixels",
                        "type": "integer",
                    },
                    "id": {
                        "description": "Source unique ID within this DatasetVersion",
                        "type": "string",
                    },
                    "masks": {"items": {"$ref": "#/definitions/mask"}, "type": "array"},
                    "meta": {
                        "additionalProperties": True,
                        "description": "Additional metadata dictionary for the source",
                        "type": "object",
                    },
                    "preview": {"$ref": "#/definitions/preview"},
                    "timestamp": {
                        "default": 0,
                        "description": (
                            "Timestamp in the source data (for video content. for images, this value should be 0)"
                        ),
                        "type": "integer",
                    },
                    "uri": {"description": "Source data URI", "type": "string"},
                    "width": {"description": "Width in pixels", "type": "integer"},
                },
                "required": ["id", "uri"],
                "type": "object",
            },
        },
        "properties": {
            "frames": {
                "description": "Frames to save",
                "items": {"$ref": "#/definitions/frame"},
                "type": "array",
            },
            "version": {"description": "Draft version ID", "type": "string"},
        },
        "required": ["version", "frames"],
        "type": "object",
    }

    def __init__(self, version, frames, **kwargs):
        super(SaveFramesRequest, self).__init__(**kwargs)
        self.version = version
        self.frames = frames

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


class SaveFramesResponse(Response):
    """
    Response of datasets.save_frames endpoint.

    :param saved: Number of frames saved
    :type saved: int
    :param failed: Number of frames we failed to save
    :type failed: int
    :param errors: Failure details
    :type errors: Sequence[dict]
    :param ids: Saved frame IDs
    :type ids: Sequence[str]
    :param total_rois: Total number of ROIs saved
    :type total_rois: int
    """

    _service = "datasets"
    _action = "save_frames"
    _version = "2.23"

    _schema = {
        "definitions": {},
        "properties": {
            "errors": {
                "description": "Failure details",
                "items": {
                    "additionalProperties": True,
                    "description": "Json object describing a save error",
                    "type": "object",
                },
                "type": ["array", "null"],
            },
            "failed": {
                "description": "Number of frames we failed to save",
                "type": ["integer", "null"],
            },
            "ids": {
                "description": "Saved frame IDs",
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
            "saved": {
                "description": "Number of frames saved",
                "type": ["integer", "null"],
            },
            "total_rois": {
                "description": "Total number of ROIs saved",
                "type": ["integer", "null"],
            },
        },
        "type": "object",
    }

    def __init__(
        self, saved=None, failed=None, errors=None, ids=None, total_rois=None, **kwargs
    ):
        super(SaveFramesResponse, self).__init__(**kwargs)
        self.saved = saved
        self.failed = failed
        self.errors = errors
        self.ids = ids
        self.total_rois = total_rois

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

    @schema_property("failed")
    def failed(self):
        return self._property_failed

    @failed.setter
    def failed(self, value):
        if value is None:
            self._property_failed = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "failed", six.integer_types)
        self._property_failed = value

    @schema_property("errors")
    def errors(self):
        return self._property_errors

    @errors.setter
    def errors(self, value):
        if value is None:
            self._property_errors = None
            return

        self.assert_isinstance(value, "errors", (list, tuple))

        self.assert_isinstance(value, "errors", (dict,), is_array=True)
        self._property_errors = value

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

    @schema_property("total_rois")
    def total_rois(self):
        return self._property_total_rois

    @total_rois.setter
    def total_rois(self, value):
        if value is None:
            self._property_total_rois = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "total_rois", six.integer_types)
        self._property_total_rois = value


class UpdateRequest(Request):
    """
    Updates an existing dataset object

    :param dataset: Dataset ID
    :type dataset: str
    :param name: Dataset name Unique within the company.
    :type name: str
    :param comment: Dataset comment
    :type comment: str
    :param tags: User-defined tags list
    :type tags: Sequence[str]
    :param system_tags: System tags list. This field is reserved for system use,
        please don't use it.
    :type system_tags: Sequence[str]
    :param terms_of_use: Terms of use string
    :type terms_of_use: str
    :param metadata: User-specified metadata object. Keys must not include '$' and
        '.'.
    :type metadata: dict
    """

    _service = "datasets"
    _action = "update"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {
            "comment": {"description": "Dataset comment", "type": "string"},
            "dataset": {"description": "Dataset ID", "type": "string"},
            "metadata": {
                "additionalProperties": True,
                "description": "User-specified metadata object. Keys must not include '$' and '.'.",
                "type": "object",
            },
            "name": {
                "description": "Dataset name Unique within the company.",
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
            "terms_of_use": {"description": "Terms of use string", "type": "string"},
        },
        "required": ["dataset"],
        "type": "object",
    }

    def __init__(
        self,
        dataset,
        name=None,
        comment=None,
        tags=None,
        system_tags=None,
        terms_of_use=None,
        metadata=None,
        **kwargs
    ):
        super(UpdateRequest, self).__init__(**kwargs)
        self.dataset = dataset
        self.name = name
        self.comment = comment
        self.tags = tags
        self.system_tags = system_tags
        self.terms_of_use = terms_of_use
        self.metadata = metadata

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

    @schema_property("comment")
    def comment(self):
        return self._property_comment

    @comment.setter
    def comment(self, value):
        if value is None:
            self._property_comment = None
            return

        self.assert_isinstance(value, "comment", six.string_types)
        self._property_comment = value

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

    @schema_property("terms_of_use")
    def terms_of_use(self):
        return self._property_terms_of_use

    @terms_of_use.setter
    def terms_of_use(self, value):
        if value is None:
            self._property_terms_of_use = None
            return

        self.assert_isinstance(value, "terms_of_use", six.string_types)
        self._property_terms_of_use = value

    @schema_property("metadata")
    def metadata(self):
        return self._property_metadata

    @metadata.setter
    def metadata(self, value):
        if value is None:
            self._property_metadata = None
            return

        self.assert_isinstance(value, "metadata", (dict,))
        self._property_metadata = value


class UpdateResponse(Response):
    """
    Response of datasets.update endpoint.

    :param updated: Number of datasets updated (0 or 1)
    :type updated: int
    :param fields: Updated fields names names and values
    :type fields: dict
    """

    _service = "datasets"
    _action = "update"
    _version = "2.23"

    _schema = {
        "definitions": {},
        "properties": {
            "fields": {
                "additionalProperties": True,
                "description": "Updated fields names names and values",
                "type": ["object", "null"],
            },
            "updated": {
                "description": "Number of datasets updated (0 or 1)",
                "enum": [0, 1],
                "type": ["integer", "null"],
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
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "updated", six.integer_types)
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


class UpdateFramesRequest(Request):
    """
    Update frames in a draft version. Each frame must contain an ID.

    :param version: Draft version ID
    :type version: str
    :param frames: Frames to update
    :type frames: Sequence[Frame]
    """

    _service = "datasets"
    _action = "update_frames"
    _version = "2.23"
    _schema = {
        "definitions": {
            "frame": {
                "properties": {
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
                    "id": {
                        "description": (
                            "Frame id. Must be unique within the dataset's version. If already exists, will cause"
                            " existing frame to be updated"
                        ),
                        "type": ["string", "null"],
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
                    "rois": {
                        "description": "Frame regions of interest",
                        "items": {"$ref": "#/definitions/roi"},
                        "type": ["array", "null"],
                    },
                    "sources": {
                        "description": "Sources of this frame",
                        "items": {"$ref": "#/definitions/source"},
                        "type": "array",
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
                },
                "required": ["sources"],
                "type": "object",
            },
            "mask": {
                "properties": {
                    "content_type": {
                        "description": "Content type (e.g. 'image/jpeg', 'image/png')",
                        "type": "string",
                    },
                    "height": {"description": "Height in pixels", "type": "integer"},
                    "id": {
                        "description": "unique ID (in this frame)",
                        "type": "string",
                    },
                    "timestamp": {
                        "default": 0,
                        "description": (
                            "Timestamp in the source data (for video content. for images, this value should be 0)"
                        ),
                        "type": "integer",
                    },
                    "uri": {"description": "Data URI", "type": "string"},
                    "width": {"description": "Width in pixels", "type": "integer"},
                },
                "required": ["id", "uri"],
                "type": "object",
            },
            "preview": {
                "properties": {
                    "content_type": {
                        "description": "Content type (e.g. 'image/jpeg', 'image/png')",
                        "type": "string",
                    },
                    "height": {
                        "description": "Height in pixels",
                        "type": "integer",
                    },
                    "timestamp": {
                        "default": 0,
                        "description": (
                            "Timestamp in the source data (for video content. for images, this value should be 0)"
                        ),
                        "type": "integer",
                    },
                    "uri": {"description": "Data URI", "type": "string"},
                    "width": {"description": "Width in pixels", "type": "integer"},
                },
                "required": ["uri"],
                "type": "object",
            },
            "roi": {
                "properties": {
                    "confidence": {
                        "description": "ROI confidence",
                        "type": "number",
                    },
                    "id": {"description": "ROI id", "type": ["string", "null"]},
                    "label": {
                        "description": "ROI labels",
                        "items": {"type": "string"},
                        "type": "array",
                    },
                    "mask": {
                        "$ref": "#/definitions/roi_mask",
                        "description": "Mask info for this ROI",
                    },
                    "meta": {
                        "additionalProperties": True,
                        "description": "Additional metadata dictionary for the roi",
                        "type": "object",
                    },
                    "poly": {
                        "description": "ROI polygon (x0, y0, ..., xn, yn)",
                        "items": {"type": "number"},
                        "type": "array",
                    },
                    "sources": {
                        "description": "Source ID",
                        "items": {"type": "string"},
                        "type": "array",
                    },
                },
                "required": ["label"],
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
                        "type": "string",
                    },
                    "height": {
                        "description": "Height in pixels",
                        "type": "integer",
                    },
                    "id": {
                        "description": "Source unique ID within this DatasetVersion",
                        "type": "string",
                    },
                    "masks": {"items": {"$ref": "#/definitions/mask"}, "type": "array"},
                    "meta": {
                        "additionalProperties": True,
                        "description": "Additional metadata dictionary for the source",
                        "type": "object",
                    },
                    "preview": {"$ref": "#/definitions/preview"},
                    "timestamp": {
                        "default": 0,
                        "description": (
                            "Timestamp in the source data (for video content. for images, this value should be 0)"
                        ),
                        "type": "integer",
                    },
                    "uri": {"description": "Source data URI", "type": "string"},
                    "width": {"description": "Width in pixels", "type": "integer"},
                },
                "required": ["id", "uri"],
                "type": "object",
            },
        },
        "properties": {
            "frames": {
                "description": "Frames to update",
                "items": {"$ref": "#/definitions/frame"},
                "type": "array",
            },
            "version": {"description": "Draft version ID", "type": "string"},
        },
        "required": ["version", "frames"],
        "type": "object",
    }

    def __init__(self, version, frames, **kwargs):
        super(UpdateFramesRequest, self).__init__(**kwargs)
        self.version = version
        self.frames = frames

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


class UpdateFramesResponse(Response):
    """
    Response of datasets.update_frames endpoint.

    :param updated: Number of frames updated
    :type updated: int
    :param merged: Number of frames merged
    :type merged: int
    :param failed: Number of frames we failed to update
    :type failed: int
    :param errors: Failure details
    :type errors: Sequence[dict]
    :param ids: Updated frame IDs
    :type ids: Sequence[str]
    :param total_rois: Total number of ROIs updated
    :type total_rois: int
    """

    _service = "datasets"
    _action = "update_frames"
    _version = "2.23"

    _schema = {
        "definitions": {},
        "properties": {
            "errors": {
                "description": "Failure details",
                "items": {
                    "additionalProperties": True,
                    "description": "Json object describing an update error",
                    "type": "object",
                },
                "type": ["array", "null"],
            },
            "failed": {
                "description": "Number of frames we failed to update",
                "type": ["integer", "null"],
            },
            "ids": {
                "description": "Updated frame IDs",
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
            "merged": {
                "description": "Number of frames merged",
                "type": ["integer", "null"],
            },
            "total_rois": {
                "description": "Total number of ROIs updated",
                "type": ["integer", "null"],
            },
            "updated": {
                "description": "Number of frames updated",
                "type": ["integer", "null"],
            },
        },
        "type": "object",
    }

    def __init__(
        self,
        updated=None,
        merged=None,
        failed=None,
        errors=None,
        ids=None,
        total_rois=None,
        **kwargs
    ):
        super(UpdateFramesResponse, self).__init__(**kwargs)
        self.updated = updated
        self.merged = merged
        self.failed = failed
        self.errors = errors
        self.ids = ids
        self.total_rois = total_rois

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

    @schema_property("merged")
    def merged(self):
        return self._property_merged

    @merged.setter
    def merged(self, value):
        if value is None:
            self._property_merged = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "merged", six.integer_types)
        self._property_merged = value

    @schema_property("failed")
    def failed(self):
        return self._property_failed

    @failed.setter
    def failed(self, value):
        if value is None:
            self._property_failed = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "failed", six.integer_types)
        self._property_failed = value

    @schema_property("errors")
    def errors(self):
        return self._property_errors

    @errors.setter
    def errors(self, value):
        if value is None:
            self._property_errors = None
            return

        self.assert_isinstance(value, "errors", (list, tuple))

        self.assert_isinstance(value, "errors", (dict,), is_array=True)
        self._property_errors = value

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

    @schema_property("total_rois")
    def total_rois(self):
        return self._property_total_rois

    @total_rois.setter
    def total_rois(self, value):
        if value is None:
            self._property_total_rois = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "total_rois", six.integer_types)
        self._property_total_rois = value


class UpdateVersionRequest(Request):
    """
    Updates version information

    :param version: Version ID
    :type version: str
    :param name: New name for the version
    :type name: str
    :param comment: New comment for the version
    :type comment: str
    :param metadata: User-specified metadata object. Keys must not include '$' and
        '.'.
    :type metadata: dict
    :param tags: User-defined tags list
    :type tags: Sequence[str]
    :param system_tags: System tags list. This field is reserved for system use,
        please don't use it.
    :type system_tags: Sequence[str]
    """

    _service = "datasets"
    _action = "update_version"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {
            "comment": {"description": "New comment for the version", "type": "string"},
            "metadata": {
                "additionalProperties": True,
                "description": "User-specified metadata object. Keys must not include '$' and '.'.",
                "type": "object",
            },
            "name": {"description": "New name for the version", "type": "string"},
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
            "version": {"description": "Version ID", "type": "string"},
        },
        "required": ["version"],
        "type": "object",
    }

    def __init__(
        self,
        version,
        name=None,
        comment=None,
        metadata=None,
        tags=None,
        system_tags=None,
        **kwargs
    ):
        super(UpdateVersionRequest, self).__init__(**kwargs)
        self.version = version
        self.name = name
        self.comment = comment
        self.metadata = metadata
        self.tags = tags
        self.system_tags = system_tags

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

    @schema_property("comment")
    def comment(self):
        return self._property_comment

    @comment.setter
    def comment(self, value):
        if value is None:
            self._property_comment = None
            return

        self.assert_isinstance(value, "comment", six.string_types)
        self._property_comment = value

    @schema_property("metadata")
    def metadata(self):
        return self._property_metadata

    @metadata.setter
    def metadata(self, value):
        if value is None:
            self._property_metadata = None
            return

        self.assert_isinstance(value, "metadata", (dict,))
        self._property_metadata = value

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


class UpdateVersionResponse(Response):
    """
    Response of datasets.update_version endpoint.

    :param updated: Number of updated versions
    :type updated: int
    """

    _service = "datasets"
    _action = "update_version"
    _version = "2.23"

    _schema = {
        "definitions": {},
        "properties": {
            "updated": {
                "description": "Number of updated versions",
                "enum": [0, 1],
                "type": ["integer", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, updated=None, **kwargs):
        super(UpdateVersionResponse, self).__init__(**kwargs)
        self.updated = updated

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


response_mapping = {
    GetByIdRequest: GetByIdResponse,
    GetByNameRequest: GetByNameResponse,
    GetAllRequest: GetAllResponse,
    GetVersionsRequest: GetVersionsResponse,
    CreateRequest: CreateResponse,
    UpdateRequest: UpdateResponse,
    PublishAndCreateChildVersionRequest: PublishAndCreateChildVersionResponse,
    CreateVersionRequest: CreateVersionResponse,
    UpdateVersionRequest: UpdateVersionResponse,
    GetStatsRequest: GetStatsResponse,
    GetSourceIdsRequest: GetSourceIdsResponse,
    GetSourcesRequest: GetSourcesResponse,
    GetSnippetsRequest: GetSnippetsResponse,
    GetLabelKeywordsForRunningTaskRequest: GetLabelKeywordsForRunningTaskResponse,
    DeleteVersionRequest: DeleteVersionResponse,
    DeleteRequest: DeleteResponse,
    SaveFramesRequest: SaveFramesResponse,
    UpdateFramesRequest: UpdateFramesResponse,
    DeleteFramesRequest: DeleteFramesResponse,
    CommitVersionRequest: CommitVersionResponse,
    PublishVersionRequest: PublishVersionResponse,
    GetSchemaKeysRequest: GetSchemaKeysResponse,
    GetSchemaRequest: GetSchemaResponse,
    GetTagsRequest: GetTagsResponse,
    MoveRequest: MoveResponse,
}
