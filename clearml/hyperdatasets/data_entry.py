import importlib
import logging
import uuid
from collections import deque

from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Type, TypeVar

from clearml.storage.manager import StorageManagerDiskSpaceFileSizeStrategy


ENTRY_CLASS_KEY = "_clearml_data_entry_class"
SUB_ENTRY_CLASS_KEY = "_clearml_data_sub_entry_class"


_LOGGER = logging.getLogger("HyperDataset")
_T = TypeVar("_T")


def _get_class_identifier(obj: Any) -> str:
    cls = obj if isinstance(obj, type) else obj.__class__
    return cls.__name__


def _copy_without_keys(metadata: Optional[Dict[str, Any]], *keys: str) -> Dict[str, Any]:
    base = dict(metadata or {})
    for key in keys:
        base.pop(key, None)
    return base


def _locate_class(class_path: Optional[str]) -> Optional[type]:
    if not isinstance(class_path, str) or not class_path:
        return None
    try:
        if ":" in class_path:
            module_name, _, qualname = class_path.partition(":")
        else:
            module_name, _, qualname = class_path.rpartition(".")
        if not module_name or not qualname:
            return None
        module = importlib.import_module(module_name)
        attr: Any = module
        for part in qualname.split("."):
            attr = getattr(attr, part)
        if isinstance(attr, type):
            return attr
        return None
    except Exception:
        return None


def _resolve_class(class_path: Optional[str], expected: Type[_T]) -> Optional[Type[_T]]:
    if not class_path:
        return None
    resolved = _find_subclass_by_name(class_path, expected)
    if resolved is not None:
        return resolved
    candidate = _locate_class(class_path)
    if candidate is None:
        _LOGGER.warning("Could not resolve class '%s'; falling back to %s", class_path, expected.__name__)
        return None
    if not issubclass(candidate, expected):
        _LOGGER.warning(
            "Resolved class '%s' is not a subclass of %s; falling back to base type",
            class_path,
            expected.__name__,
        )
        return None
    return candidate


def _find_subclass_by_name(class_name: Optional[str], expected: Type[_T]) -> Optional[Type[_T]]:
    if not isinstance(class_name, str) or not class_name:
        return None
    if expected.__name__ == class_name:
        return expected
    seen: Set[type] = set()
    queue = deque(expected.__subclasses__())
    while queue:
        candidate = queue.popleft()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.__name__ == class_name:
            return candidate
        try:
            queue.extend(candidate.__subclasses__())
        except Exception:
            continue
    return None


class DataEntry:
    def __init__(
        self,
        data_entry_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Container for a logical frame composed of one or more sub-entries.

        :param data_entry_id: Explicit entry identifier to reuse
        :param metadata: Optional metadata dictionary stored on the frame
        """
        self._id = data_entry_id or uuid.uuid4().hex
        self._sub_entries: Dict[str, "DataSubEntry"] = {}
        self._metadata: Dict[str, Any] = metadata or {}
        self._annotations: List[Dict[str, Any]] = []

    def add_sub_entries(self, sub_entries: Iterable["DataSubEntry"]) -> None:
        """
        Attach the provided sub-entry objects, indexed by their name.

        :param sub_entries: Iterable of sub-entry instances to register
        """
        self._sub_entries.update({sub_entry.name: sub_entry for sub_entry in sub_entries})

    def set_vector(
        self,
        vector: Sequence[Any],
        metadata_field: str,
    ) -> None:
        """
        Store a vector representation on the entry metadata.

        :param vector: Sequence of scalar values describing the entry.
        :param metadata_field: Metadata key to populate with the vector (for example ``_vector`` or
            ``vectors.embedding``). The caller must supply the exact metadata path used when
            registering field mappings on the dataset.
        """
        if not isinstance(vector, (list, tuple)):
            raise TypeError("vector must be a list or tuple")
        if not isinstance(metadata_field, str) or not metadata_field:
            raise TypeError("metadata_field must be a non-empty string")
        self._metadata[metadata_field] = list(vector)

    def add_annotation(self, id, labels, confidence, metadata):
        """
        Attach a metadata-only annotation to this entry.

        :param id: Identifier associated with this annotation
        :param labels: Sequence of labels to associate with the annotation
        :param confidence: Optional confidence value
        :param metadata: Additional metadata to store alongside the annotation
        :return: Numeric index of the appended annotation
        """
        roi: Dict[str, Any] = {}
        if labels:
            roi["label"] = list(labels)
        if confidence is not None:
            roi["confidence"] = float(confidence)
        roi_meta: Dict[str, Any] = dict(metadata or {})
        if id is not None:
            roi_meta["_id"] = id
        if roi_meta:
            roi["meta"] = roi_meta
        self._annotations.append(roi)
        return len(self._annotations) - 1

    def remove_annotation(self, index: Optional[int] = None, **kwargs: Any) -> Any:
        """
        Remove a single annotation by numeric index or identifier.

        :param index: Annotation index to remove
        :param kwargs: Alternative filters such as id=...
        :return: Removed annotation payload or None when nothing matched
        """
        if not self._annotations:
            return None
        if index is None:
            ann_id = kwargs.pop("id", None)
            if ann_id is None:
                raise ValueError("index is required (or provide id=...) to remove_annotation")
            removed = self.remove_annotations(id=ann_id)
            return removed[0] if removed else None
        try:
            return self._annotations.pop(index)
        except Exception:
            return None

    def remove_annotations(
        self,
        id: Optional[str] = None,
        label: Optional[str] = None,
        labels: Optional[Sequence[str]] = None,
    ) -> Sequence[Any]:
        """
        Remove annotations that match the provided id or label filters.

        :param id: Annotation identifier to match
        :param label: Single label to match
        :param labels: Sequence of labels to match
        :return: Sequence of removed annotation payloads
        """
        if not self._annotations:
            return []
        removed: List[Dict[str, Any]] = []
        keep: List[Dict[str, Any]] = []
        label_set = set(labels or ([] if label is None else [label])) if (labels or label) else None
        for ann in self._annotations:
            meta = ann.get("meta") or {}
            if id is not None and meta.get("_id") != id:
                keep.append(ann)
                continue
            if label_set is not None:
                ann_labels = set(ann.get("label") or [])
                if not (ann_labels & label_set):
                    keep.append(ann)
                    continue
            removed.append(ann)
        self._annotations = keep
        return removed

    def get_all_annotations(self) -> Sequence[Any]:
        """
        Return all annotations attached to this entry.

        :return: Sequence of annotation payloads
        """
        return list(self._annotations)

    def get_annotations(self, id: Optional[str] = None, index: Optional[int] = None) -> Sequence[Any]:
        """
        Return annotations matching the supplied identifier/index filters.

        :param id: Annotation identifier to filter by
        :param index: Annotation index to fetch
        :return: Sequence of matching annotation payloads
        """
        anns = list(self._annotations)
        if index is not None:
            try:
                return [anns[index]]
            except Exception:
                return []
        if id is not None:
            return [ann for ann in anns if (ann.get("meta") or {}).get("_id") == id]
        return anns

    def to_api_object(self) -> Dict[str, Any]:
        """
        Serialize this entry to the SaveFramesRequest frame schema.

        :return: Frame dictionary ready for backend submission
        """
        meta: Dict[str, Any] = _copy_without_keys(self._metadata, ENTRY_CLASS_KEY)
        meta[ENTRY_CLASS_KEY] = _get_class_identifier(self)
        obj = {"id": self._id, "meta": meta, "sources": []}
        for sub_entry in self._sub_entries.values():
            sub_meta: Dict[str, Any] = {}
            if isinstance(sub_entry._metadata, dict):
                sub_meta = _copy_without_keys(sub_entry._metadata, SUB_ENTRY_CLASS_KEY)
            sub_meta[SUB_ENTRY_CLASS_KEY] = _get_class_identifier(sub_entry)
            obj["meta"][sub_entry.name] = sub_meta
            sub_entry_dict = {"id": sub_entry.name, "uri": sub_entry.source}
            if sub_entry.preview_source:
                sub_entry_dict["preview"] = {"uri": sub_entry.preview_source}
            # Include hashes if available
            if sub_entry.get_hash("source") or sub_entry.get_hash("preview_source"):
                sub_entry_dict["meta"] = {"hash": {}}
                if sub_entry.get_hash("source"):
                    sub_entry_dict["meta"]["hash"]["source"] = sub_entry.get_hash("source")
                if sub_entry.get_hash("preview_source"):
                    sub_entry_dict["meta"]["hash"]["preview_source"] = sub_entry.get_hash("preview_source")
            obj["sources"].append(sub_entry_dict)
        if self._annotations:
            obj["rois"] = list(self._annotations)
        return obj

    def _has_upload_destination(self) -> bool:
        """
        Check whether any sub-entry expects its sources to be uploaded.

        :return: True when at least one sub-entry was given an upload destination
        """
        for sub_entry in self._sub_entries.values():
            if sub_entry._local_sources_upload_destination:
                return True
        return False

    def __iter__(self) -> Iterator["DataSubEntry"]:
        """
        Iterate over the sub-entry objects in insertion order.

        :return: Iterator of sub-entries contained in this entry
        """
        return iter(self._sub_entries.values())

    @property
    def id(self) -> str:
        """
        Return the entry identifier used by the backend.

        :return: Entry identifier string
        """
        return self._id

    @property
    def sub_data_entries(self) -> List["DataSubEntry"]:
        """
        Return the list of attached sub-entries.

        :return: List of sub-entry objects
        """
        return list(self._sub_entries.values())

    def __repr__(self) -> str:
        """
        Return a concise debug representation of the data entry.

        :return: String summarising identifier, metadata keys, and sub-entry names
        """
        meta_keys = sorted(self._metadata.keys())
        sub_names = sorted(self._sub_entries.keys())
        return (
            f"{self.__class__.__name__}(id={self._id!r}, "
            f"metadata_keys={meta_keys}, sub_entries={sub_names})"
        )

    @classmethod
    def from_api_object(cls, data_entry: Any) -> "DataEntry":
        """
        Convert a backend frame object/dict into a generic DataEntry tree.
        """
        def _get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        raw_meta = _get(data_entry, "meta") or {}
        metadata = _copy_without_keys(raw_meta, ENTRY_CLASS_KEY)
        resolved_entry_cls: Optional[Type["DataEntry"]] = None
        if isinstance(raw_meta, dict):
            resolved_entry_cls = _resolve_class(raw_meta.get(ENTRY_CLASS_KEY), DataEntry)
            if resolved_entry_cls and not issubclass(resolved_entry_cls, DataEntry):
                resolved_entry_cls = None
        entry = cls(data_entry_id=_get(data_entry, "id"), metadata=metadata)
        sources = _get(data_entry, "sources") or []
        sub_entries: List[DataSubEntry] = []
        for idx, s in enumerate(sources):
            name = _get(s, "id") or f"source_{idx}"
            # map sub-metadata if exists under frame.meta[name]
            sub_meta_raw = raw_meta.get(name) if isinstance(raw_meta, dict) else None
            sub_meta_clean: Optional[Dict[str, Any]] = None
            resolved_cls: Optional[Type[DataSubEntry]] = None
            if isinstance(sub_meta_raw, dict):
                class_path = sub_meta_raw.get(SUB_ENTRY_CLASS_KEY)
                sub_meta_clean = _copy_without_keys(sub_meta_raw, SUB_ENTRY_CLASS_KEY)
                resolved_cls = _resolve_class(class_path, DataSubEntry)
                metadata[name] = sub_meta_clean
            elif name in metadata:
                metadata[name] = sub_meta_raw

            sub = DataSubEntry.from_api_object(s, name_fallback=name)
            if isinstance(sub_meta_clean, dict):
                sub._metadata = sub_meta_clean
            if resolved_cls and issubclass(resolved_cls, DataSubEntry):
                try:
                    sub.__class__ = resolved_cls
                except TypeError:
                    _LOGGER.warning(
                        "Could not assign sub-entry %s to class '%s'", name, resolved_cls.__name__
                    )
            sub_entries.append(sub)
        if sub_entries:
            entry.add_sub_entries(sub_entries)
        rois = _get(data_entry, "rois") or []
        if rois:
            entry._annotations = list(rois)
        if (
            resolved_entry_cls
            and isinstance(entry, DataEntry)
            and issubclass(resolved_entry_cls, DataEntry)
            and entry.__class__ is not resolved_entry_cls
        ):
            try:
                entry.__class__ = resolved_entry_cls
            except TypeError:
                _LOGGER.warning(
                    "Could not assign data entry %s to class '%s'",
                    entry.id,
                    resolved_entry_cls.__name__,
                )
        return entry


class DataSubEntry:
    def __init__(
        self,
        name: str,
        source: Optional[str],
        preview_source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialise a base sub-entry that stores source URIs and optional metadata.

        :param name: Identifier of the sub-entry (unique per entry)
        :param source: Primary source URI
        :param preview_source: Optional preview URI
        :param local_sources_upload_destination: Upload target used when auto-uploading
        :param metadata: Optional metadata dictionary stored alongside the sub-entry
        """
        self._name = name
        self._source = source
        self._preview_source = preview_source
        self._source_hash: Optional[str] = None
        self._preview_source_hash: Optional[str] = None
        self._local_sources_upload_destination = None
        self._metadata = metadata or {}

    def get_hash(self, source_field: str = "source") -> Optional[str]:
        """
        Return the cached SHA256 hash for the requested source field.

        :param source_field: Either "source" or "preview_source"
        :return: Hex digest string when available, otherwise None
        """
        if source_field == "source":
            return self._source_hash
        if source_field == "preview_source":
            return self._preview_source_hash
        return None

    def get_source(self, source_field: str = "source") -> Optional[str]:
        """
        Return the URI associated with the requested source field.

        :param source_field: Either "source" or "preview_source"
        :return: URI string when set, otherwise None
        """
        if source_field == "source":
            return self._source
        if source_field == "preview_source":
            return self._preview_source
        return None

    def set_source(self, source_field: str, uri: Optional[str]) -> None:
        """
        Update the URI linked to the requested source field.

        :param source_field: Either "source" or "preview_source"
        :param uri: New URI to associate with the field
        """
        if source_field == "source":
            self._source = uri
        elif source_field == "preview_source":
            self._preview_source = uri

    def set_local_sources_upload_destination(self, local_sources_upload_destination):
        """
        Set an upload destination for the local sources. This will be used when uploading the data entry

        :param local_sources_upload_destination: URL to the upload path
        """
        self._local_sources_upload_destination = local_sources_upload_destination

    def get_local_source(
        self,
        raise_on_error: bool = False,
        force_download: bool = False,
    ) -> Optional[str]:
        """
        Retrieve a cached local copy of the primary source URI.

        :param raise_on_error: Raise ValueError when the download fails
        :param force_download: Refresh the cached copy even if it already exists
        :return: Absolute path to the local copy or None on failure/when source missing
        """
        return self._get_local_source_for_field(
            "source",
            raise_on_error=raise_on_error,
            force_download=force_download,
        )

    def get_local_preview_source(
        self,
        raise_on_error: bool = False,
        force_download: bool = False,
    ) -> Optional[str]:
        """
        Retrieve a cached local copy of the preview source URI.

        :param raise_on_error: Raise ValueError when the download fails
        :param force_download: Refresh the cached copy even if it already exists
        :return: Absolute path to the local copy or None on failure/when preview missing
        """
        return self._get_local_source_for_field(
            "preview_source",
            raise_on_error=raise_on_error,
            force_download=force_download,
        )

    def _get_local_source_for_field(
        self,
        source_field: str,
        raise_on_error: bool,
        force_download: bool,
    ) -> Optional[str]:
        uri = self.get_source(source_field)
        if not uri:
            return None
        try:
            local_file = StorageManagerDiskSpaceFileSizeStrategy.get_local_copy(
                uri,
                extract_archive=False,
                force_download=force_download,
            )
        except Exception as ex:
            _LOGGER.warning("Could not fetch local copy for %s: %s", uri, ex)
            local_file = None
        if not local_file and raise_on_error:
            raise ValueError("Failed downloading file: {}".format(uri))
        return local_file

    @property
    def name(self) -> str:
        """
        Return the sub-entry identifier (per entry unique).

        :return: Sub-entry name
        """
        return self._name

    @property
    def source(self) -> Optional[str]:
        """
        Expose the main source URI.

        :return: Source URI string or None
        """
        return self._source

    @property
    def preview_source(self) -> Optional[str]:
        """
        Expose the preview URI when available.

        :return: Preview URI string or None
        """
        return self._preview_source

    def __repr__(self) -> str:
        """
        Return a concise debug representation of the sub-entry.

        :return: String summarising name, source URIs, and metadata keys
        """
        meta_keys = sorted((self._metadata or {}).keys()) if isinstance(self._metadata, dict) else []
        return (
            f"{self.__class__.__name__}(name={self._name!r}, source={self._source!r}, "
            f"preview={self._preview_source!r}, metadata_keys={meta_keys})"
        )

    @classmethod
    def from_api_object(cls, source_obj: Any, name_fallback: str = "source_0") -> "DataSubEntry":
        def _get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        name = _get(source_obj, "id") or name_fallback
        uri = _get(source_obj, "uri")
        preview = None
        p = _get(source_obj, "preview")
        if p:
            preview = _get(p, "uri")
        return cls(name=name, source=uri, preview_source=preview, metadata=None)
