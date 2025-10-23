from typing import Optional, Sequence, Tuple, List, Dict, Any, Union

import logging

from clearml.storage.manager import StorageManagerDiskSpaceFileSizeStrategy

from .data_entry import (
    DataEntry,
    DataSubEntry,
    ENTRY_CLASS_KEY,
    SUB_ENTRY_CLASS_KEY,
    _copy_without_keys,
    _get_class_identifier,
    _resolve_class,
)


Values = Union[
    Sequence[float],
    Sequence[int],
    Sequence[Tuple[float, float]],
    Sequence[Tuple[float, float, float]],
    Sequence[Sequence[Tuple[float, float]]],
    Sequence[Sequence[Tuple[float, float, float]]],
]


class DataSubEntryImage(DataSubEntry):
    def __init__(
        self,
        name: str = "image_entry_0",
        source: Optional[str] = None,
        preview_source: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        timestamp: Optional[int] = None,
        context_id: Optional[str] = None,
        masks_source: Optional[Union[Sequence[str], Dict[str, str]]] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Initialise an image sub-entry with optional dimension, context and mask metadata.

        :param name: Identifier of the sub-entry (defaults to image_entry_0)
        :param source: Primary image URI
        :param preview_source: Optional preview image URI
        :param width: Image width in pixels
        :param height: Image height in pixels
        :param timestamp: Optional timestamp associated with the frame
        :param context_id: Optional context identifier to correlate sources
        :param masks_source: Sequence or mapping of mask URIs
        :param metadata: Optional metadata dictionary stored alongside the sub-entry
        """
        super(DataSubEntryImage, self).__init__(
            name=name,
            source=source,
            preview_source=preview_source,
            metadata=metadata,
        )
        self._width = width
        self._height = height
        self._timestamp = timestamp
        self._context_id = context_id
        self._masks_source: Dict[str, str] = {}
        if masks_source:
            if isinstance(masks_source, dict):
                self._masks_source = {str(k): str(v) for k, v in masks_source.items()}
            else:
                self._masks_source = {f"{i:02d}": str(u) for i, u in enumerate(masks_source)}
        self._annotations: List[Dict[str, Any]] = []

    @property
    def width(self) -> Optional[int]:
        """
        Return cached image width if known.

        :return: Width in pixels or None when unknown
        """
        return self._width

    @width.setter
    def width(self, value: Optional[int]) -> None:
        """
        Update the cached image width.

        :param value: Width in pixels or None to clear the stored value
        """
        self._width = value

    @property
    def height(self) -> Optional[int]:
        """
        Return cached image height if known.

        :return: Height in pixels or None when unknown
        """
        return self._height

    @height.setter
    def height(self, value: Optional[int]) -> None:
        """
        Update the cached image height.

        :param value: Height in pixels or None to clear the stored value
        """
        self._height = value

    @property
    def timestamp(self) -> Optional[int]:
        """
        Return the timestamp associated with this frame, if any.

        :return: Timestamp value or None
        """
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value: Optional[int]) -> None:
        """
        Update the timestamp associated with the sub-entry.

        :param value: Timestamp value or None to clear the stored timestamp
        """
        self._timestamp = value

    @property
    def context_id(self) -> Optional[str]:
        """
        Return the context identifier used to correlate sub-entries.

        :return: Context identifier string or None
        """
        return self._context_id

    @context_id.setter
    def context_id(self, value: Optional[str]) -> None:
        """
        Update the context identifier associated with the sub-entry.

        :param value: Context identifier string or None to clear the stored value
        """
        self._context_id = value

    def set_mask_source(self, uri: Optional[str]) -> Optional[str]:
        """
        Add a single mask URI and auto-number it (00, 01, 02, ...).

        Returns the assigned mask id, or None if uri is falsy.
        """
        if not uri:
            return None
        if self._masks_source is None:
            self._masks_source = {}
        # find next available mask id (00, 01, ...)
        i = 0
        while f"{i:02d}" in self._masks_source:
            i += 1
        mask_id = f"{i:02d}"
        self._masks_source[mask_id] = str(uri)
        return mask_id

    def set_masks_source(self, masks_source: Optional[Union[Sequence[str], Dict[str, str]]] = None) -> None:
        """
        Set multiple mask URIs and auto-number them (00, 01, 02, ...).

        For dict input, the values' iteration order is used and keys are ignored.
        For list/sequence input, order is preserved.
        """
        if masks_source is None:
            self._masks_source = {}
            return
        uris: List[str] = []
        if isinstance(masks_source, dict):
            # use insertion order of values
            uris = [str(v) for v in masks_source.values()]
        else:
            uris = [str(u) for u in masks_source]
        self._masks_source = {f"{i:02d}": u for i, u in enumerate(uris)}

    def get_masks_source_dict(self) -> Dict[str, str]:
        """
        Return a copy of the mask-id to URI mapping.

        :return: Dictionary mapping mask ids to URIs
        """
        return dict(self._masks_source)

    def get_mask_source(self, mask_id: Optional[str] = None) -> Optional[str]:
        """
        Return the URI for the requested mask id (or the first mask if omitted).

        :param mask_id: Mask identifier (e.g. "00")
        :return: Mask URI string or None when unavailable
        """
        if not self._masks_source:
            return None
        if mask_id is None:
            # return the first mask if exists
            return self._masks_source.get(sorted(self._masks_source.keys())[0])
        return self._masks_source.get(mask_id)

    def get_local_mask_source(
        self,
        raise_on_error: bool = False,
        mask_id: Optional[str] = None,
        force_download: bool = False,
    ) -> Optional[str]:
        """
        Retrieve a cached local copy of a specific mask source.

        :param raise_on_error: Raise ValueError when the download fails
        :param mask_id: Mask identifier to fetch; defaults to the first mask
        :param force_download: Refresh an existing cached entry when True
        :return: Absolute path to the local copy or None when unavailable
        """
        uri = self.get_mask_source(mask_id)
        if not uri:
            return None
        try:
            local_file = StorageManagerDiskSpaceFileSizeStrategy.get_local_copy(
                uri,
                extract_archive=False,
                force_download=force_download,
            )
        except Exception as ex:
            logging.getLogger("HyperDataset").warning("Could not fetch local mask copy for %s: %s", uri, ex)
            local_file = None
        if not local_file and raise_on_error:
            raise ValueError("Failed downloading file: {}".format(uri))
        return local_file

    def get_local_masks_source(
        self,
        raise_on_error: bool = False,
        force_download: bool = False,
    ) -> Dict[str, Optional[str]]:
        """
        Retrieve cached local copies for all mask sources on this sub-entry.

        :param raise_on_error: Raise ValueError when any download fails
        :param force_download: Refresh existing cached entries when True
        :return: Mapping of mask id to the local copy path (or None on failure if raise_on_error is False)
        """
        masks: Dict[str, Optional[str]] = {}
        for mid, uri in sorted((self._masks_source or {}).items()):
            if not uri:
                masks[mid] = None
                continue
            try:
                local_file = StorageManagerDiskSpaceFileSizeStrategy.get_local_copy(
                    uri,
                    extract_archive=False,
                    force_download=force_download,
                )
            except Exception as ex:
                logging.getLogger("HyperDataset").warning("Could not fetch local mask copy for %s: %s", uri, ex)
                local_file = None
            if not local_file and raise_on_error:
                raise ValueError("Failed downloading file: {}".format(uri))
            masks[mid] = local_file
        return masks

    def __repr__(self) -> str:
        meta_keys = sorted((self._metadata or {}).keys()) if isinstance(self._metadata, dict) else []
        return (
            f"{self.__class__.__name__}(name={self._name!r}, source={self._source!r}, "
            f"preview={self._preview_source!r}, size=({self._width},{self._height}), "
            f"masks={len(self._masks_source)}, metadata_keys={meta_keys})"
        )

    def add_annotation(
        self,
        poly2d_xy: Optional[Values] = None,
        poly3d_xyz: Optional[Values] = None,
        points2d_xy: Optional[Values] = None,
        points3d_xyz: Optional[Values] = None,
        box2d_xywh: Optional[Values] = None,
        box3d_xyzwhxyzwh: Optional[Values] = None,
        ellipse2d_xyrrt: Optional[Values] = None,
        mask_rgb: Optional[Values] = None,
        frame_class: Optional[Sequence[str]] = None,
        id: Optional[str] = None,
        labels: Optional[Sequence[str]] = None,
        confidence: Optional[float] = None,
        metadata: Optional[dict] = None
    ) -> List[int]:
        """
        Create ROI records for this sub-entry and return their indices.

        :param poly2d_xy: 2D polygon coordinates
        :param poly3d_xyz: 3D polygon coordinates
        :param points2d_xy: 2D keypoint coordinates
        :param points3d_xyz: 3D keypoint coordinates
        :param box2d_xywh: 2D bounding box definition
        :param box3d_xyzwhxyzwh: 3D bounding box definition
        :param ellipse2d_xyrrt: 2D ellipse definition
        :param mask_rgb: RGB mask values
        :param frame_class: Optional frame-level class labels
        :param id: Annotation identifier
        :param labels: Sequence of label names
        :param confidence: Optional confidence value
        :param metadata: Extra metadata mapping to attach to the annotation
        :return: List of annotation indices that were appended
        """
        # Minimal in-memory ROI creation compatible with SaveFramesRequest schema
        anns: List[Dict[str, Any]] = []

        def _flatten_xy(seq: Any) -> Optional[List[float]]:
            if not seq:
                return None
            # Accept [(x,y), ...] or [x0,y0,...]
            if isinstance(seq, (list, tuple)) and seq and isinstance(seq[0], (list, tuple)):
                flat: List[float] = []
                for x, y in seq:  # type: ignore[misc]
                    flat.extend([float(x), float(y)])
                return flat
            try:
                return [float(v) for v in seq]  # type: ignore[return-value]
            except Exception:
                return None

        # helper to append an ROI dict
        def _add_roi(meta_type: Optional[str], poly: Optional[List[float]] = None, mask: Optional[List[int]] = None):
            roi: Dict[str, Any] = {}
            if labels is not None:
                roi["label"] = list(labels)
            if confidence is not None:
                roi["confidence"] = float(confidence)
            roi_meta: Dict[str, Any] = {}
            if metadata:
                roi_meta.update(metadata)
            if meta_type:
                roi_meta["_type"] = meta_type
            if id is not None:
                roi_meta["_id"] = id
            if roi_meta:
                roi["meta"] = roi_meta
            if poly is not None:
                roi["poly"] = poly
            if mask is not None:
                roi["mask"] = {"id": "00", "value": list(mask)}
            # Associate with this subentry's source id (name)
            roi["sources"] = [self.name]
            anns.append(roi)

        # Create ROIs for provided shapes (each becomes its own ROI, sharing id if given)
        if poly2d_xy is not None:
            poly = _flatten_xy(poly2d_xy)
            if poly:
                _add_roi("p2d", poly=poly)
        if box2d_xywh is not None:
            # Represent box as polygon (x,y,w,h[,a]) approximated by rectangle corners without rotation
            try:
                vals = list(box2d_xywh)  # type: ignore
                x, y, w, h = [float(vals[i]) for i in range(4)]
                rect_poly = [x, y, x + w, y, x + w, y + h, x, y + h]
                _add_roi("b2d", poly=rect_poly)
            except Exception:
                pass
        if points2d_xy is not None:
            flat = _flatten_xy(points2d_xy)
            if flat:
                _add_roi("k2d", poly=flat)
        if ellipse2d_xyrrt is not None:
            try:
                vals = [float(v) for v in ellipse2d_xyrrt]  # cx,cy,rx,ry,theta
                _add_roi("elp", poly=vals)
            except Exception:
                pass
        if mask_rgb is not None:
            try:
                mrgb = [int(v) for v in mask_rgb]  # r,g,b
                _add_roi("mask", mask=mrgb)
            except Exception:
                pass
        if poly3d_xyz is not None:
            # Store as flat list in poly, typed
            try:
                flat = [float(v) for v in (sum(poly3d_xyz, []) if isinstance(poly3d_xyz[0], (list, tuple)) else poly3d_xyz)]  # type: ignore[index]
                _add_roi("p3d", poly=flat)
            except Exception:
                pass
        if points3d_xyz is not None:
            try:
                flat = [float(v) for v in (sum(points3d_xyz, []) if isinstance(points3d_xyz[0], (list, tuple)) else points3d_xyz)]  # type: ignore[index]
                _add_roi("k3d", poly=flat)
            except Exception:
                pass
        if box3d_xyzwhxyzwh is not None:
            try:
                flat = [float(v) for v in box3d_xyzwhxyzwh]  # type: ignore
                _add_roi("b3d", poly=flat)
            except Exception:
                pass
        if frame_class is not None and (not labels):
            # Frame-level class as ROI without geometry
            _add_roi(None)

        # Store annotations list
        if not hasattr(self, "_annotations"):
            self._annotations = []  # type: ignore[attr-defined]
        start = len(self._annotations)  # type: ignore[attr-defined]
        # type: ignore[attr-defined]
        self._annotations.extend(anns)
        return list(range(start, start + len(anns)))

    def remove_annotation(self, index: Optional[int] = None, **kwargs: Any) -> Any:
        """
        Remove a single annotation by numeric index or identifier.

        :param index: Annotation index to remove
        :param kwargs: Alternative filters such as id=...
        :return: Removed annotation payload or None when nothing matched
        """
        if not hasattr(self, "_annotations") or not self._annotations:
            return None
        if index is None:
            ann_id = kwargs.pop("id", None)
            if ann_id is None:
                raise ValueError("index is required (or provide id=...) to remove_annotation")
            # delegate to remove_annotations by id and return first removed
            removed = self.remove_annotations(id=ann_id)
            return removed[0] if removed else None
        try:
            return self._annotations.pop(index)
        except Exception:
            return None

    def remove_annotations(
        self, id: Optional[str] = None, label: Optional[str] = None, labels: Optional[Sequence[str]] = None
    ) -> Sequence[Any]:
        """
        Remove annotations that match the provided id or label filters.

        :param id: Annotation identifier to match
        :param label: Single label to match
        :param labels: Sequence of labels to match
        :return: Sequence of removed annotation payloads
        """
        if not hasattr(self, "_annotations") or not self._annotations:
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
        Return all annotations attached to this sub-entry.

        :return: Sequence of annotation payloads
        """
        return list(getattr(self, "_annotations", []) or [])

    def get_annotations(self, id: Optional[str] = None, index: Optional[int] = None) -> Sequence[Any]:
        """
        Return annotations matching the supplied identifier/index filters.

        :param id: Annotation identifier to filter by
        :param index: Annotation index to fetch
        :return: Sequence of matching annotation payloads
        """
        anns = list(getattr(self, "_annotations", []) or [])
        if index is not None:
            try:
                return [anns[index]]
            except Exception:
                return []
        if id is not None:
            return [a for a in anns if (a.get("meta") or {}).get("_id") == id]
        return anns

    @classmethod
    def from_api_object(
        cls,
        source_obj: Any,
        frame_meta: Optional[dict] = None,
        context_id: Optional[str] = None,
        name_fallback: str = "image_0",
    ) -> "DataSubEntryImage":
        def _get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)
        name = _get(source_obj, "id") or name_fallback
        uri = _get(source_obj, "uri")
        width = _get(source_obj, "width")
        height = _get(source_obj, "height")
        ts = _get(source_obj, "timestamp")
        preview = None
        p = _get(source_obj, "preview")
        if p:
            preview = _get(p, "uri")
        masks_list = _get(source_obj, "masks") or []
        masks = {str(_get(m, "id")): _get(m, "uri") for m in masks_list if _get(m, "uri")}
        sub_meta = (frame_meta or {}).get(name) if isinstance(frame_meta, dict) else None
        return cls(
            name=name,
            source=uri,
            preview_source=preview,
            width=width,
            height=height,
            timestamp=ts,
            context_id=context_id,
            masks_source=masks,
            metadata=sub_meta,
        )


class DataEntryImage(DataEntry):
    def __init__(
        self,
        data_entry_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        super(DataEntryImage, self).__init__(data_entry_id=data_entry_id, metadata=metadata)
        # optional global annotations storage for the entry level
        self._global_annotations: List[Dict[str, Any]] = []

    def __repr__(self) -> str:
        meta_keys = sorted((getattr(self, "_metadata", {}) or {}).keys())
        sub_names = [sub.name for sub in (self.sub_data_entries or [])]
        return (
            f"{self.__class__.__name__}(id={self.id!r}, sub_entries={sub_names}, "
            f"metadata_keys={meta_keys})"
        )

    def add_global_annotation(
        self,
        poly2d_xy: Optional[Values] = None,
        poly3d_xyz: Optional[Values] = None,
        points2d_xy: Optional[Values] = None,
        points3d_xyz: Optional[Values] = None,
        box2d_xywh: Optional[Values] = None,
        box3d_xyzwhxyzwh: Optional[Values] = None,
        ellipse2d_xyrrt: Optional[Values] = None,
        mask_rgb: Optional[Values] = None,
        frame_class: Optional[Sequence[str]] = None,
        id: Optional[str] = None,
        labels: Optional[Sequence[str]] = None,
        confidence: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> List[int]:
        """
        Broadcast an annotation request to all sub-entries and aggregate their indices.

        :param poly2d_xy: 2D polygon coordinates
        :param poly3d_xyz: 3D polygon coordinates
        :param points2d_xy: 2D keypoint coordinates
        :param points3d_xyz: 3D keypoint coordinates
        :param box2d_xywh: 2D bounding box definition
        :param box3d_xyzwhxyzwh: 3D bounding box definition
        :param ellipse2d_xyrrt: 2D ellipse definition
        :param mask_rgb: RGB mask values
        :param frame_class: Optional frame-level class labels
        :param id: Annotation identifier
        :param labels: Sequence of label names
        :param confidence: Optional confidence value
        :param metadata: Extra metadata mapping to attach to the annotation
        :return: List of annotation indices returned by the sub-entries
        """
        idxs: List[int] = []
        for sub in (self.sub_data_entries or []):
            if hasattr(sub, "add_annotation"):
                idxs.extend(
                    sub.add_annotation(
                        poly2d_xy=poly2d_xy,
                        poly3d_xyz=poly3d_xyz,
                        points2d_xy=points2d_xy,
                        points3d_xyz=points3d_xyz,
                        box2d_xywh=box2d_xywh,
                        box3d_xyzwhxyzwh=box3d_xyzwhxyzwh,
                        ellipse2d_xyrrt=ellipse2d_xyrrt,
                        mask_rgb=mask_rgb,
                        frame_class=frame_class,
                        id=id,
                        labels=labels,
                        confidence=confidence,
                        metadata=metadata,
                    )
                )
        return idxs

    def remove_global_annotation(self, index: Optional[int] = None, **kwargs: Any) -> Any:
        """
        Remove the first matching annotation across sub-entries.

        :param index: Annotation index to remove
        :param kwargs: Alternative filters such as id=...
        :return: Removed annotation payload or None when nothing matched
        """
        removed = None
        for sub in (self.sub_data_entries or []):
            r = sub.remove_annotation(index=index, **kwargs)
            removed = removed or r
        return removed

    def remove_global_annotations(
        self, id: Optional[str] = None, label: Optional[str] = None, labels: Optional[Sequence[str]] = None
    ) -> Sequence[Any]:
        """
        Remove annotations across sub-entries using the provided filters.

        :param id: Annotation identifier to match
        :param label: Single label to match
        :param labels: Sequence of labels to match
        :return: Sequence of removed annotation payloads
        """
        removed: List[Any] = []
        for sub in (self.sub_data_entries or []):
            if hasattr(sub, "remove_annotations"):
                removed.extend(sub.remove_annotations(id=id, label=label, labels=labels))
        return removed

    def get_all_global_annotations(self) -> Sequence[Any]:
        """
        Return every annotation collected from all sub-entries.

        :return: Sequence of annotation payloads across all sub-entries
        """
        anns: List[Any] = []
        for sub in (self.sub_data_entries or []):
            get_all = getattr(sub, "get_all_annotations", None)
            if callable(get_all):
                anns.extend(get_all())
            else:
                anns.extend(getattr(sub, "_annotations", []) or [])
        return anns

    def get_global_annotations(self, id: Optional[str] = None, index: Optional[int] = None) -> Sequence[Any]:
        """
        Return global annotations filtered by identifier or index.

        :param id: Annotation identifier to filter by
        :param index: Annotation index to fetch
        :return: Sequence of matching annotation payloads
        """
        if id is None and index is None:
            return self.get_all_global_annotations()
        anns: List[Any] = []
        for sub in (self.sub_data_entries or []):
            if hasattr(sub, "get_annotations"):
                anns.extend(sub.get_annotations(id=id, index=index))
        return anns

    def to_api_object(self) -> dict:
        # Build SaveFramesRequest-compatible frame dict
        entry_meta = _copy_without_keys(getattr(self, "_metadata", {}) or {}, ENTRY_CLASS_KEY)
        entry_meta[ENTRY_CLASS_KEY] = _get_class_identifier(self)
        frame: Dict[str, Any] = {"id": self.id, "meta": entry_meta}
        sources: List[Dict[str, Any]] = []
        rois: List[Dict[str, Any]] = []
        context_id: Optional[str] = None
        for sub in (self.sub_data_entries or []):
            # merge per-subentry metadata under its name
            sub_meta: Dict[str, Any] = {}
            if isinstance(getattr(sub, "_metadata", None), dict):
                sub_meta = _copy_without_keys(sub._metadata, SUB_ENTRY_CLASS_KEY)
            sub_meta[SUB_ENTRY_CLASS_KEY] = _get_class_identifier(sub)
            frame["meta"][sub.name] = sub_meta
            s: Dict[str, Any] = {"id": sub.name, "uri": sub.get_source("source")}
            if sub.get_source("preview_source"):
                s["preview"] = {"uri": sub.get_source("preview_source")}
            # dimensions/timestamp
            if getattr(sub, "_width", None) is not None:
                s["width"] = sub._width
            if getattr(sub, "_height", None) is not None:
                s["height"] = sub._height
            if getattr(sub, "_timestamp", None) is not None:
                s["timestamp"] = sub._timestamp
            # hashes metadata
            sh = sub.get_hash("source")
            ph = sub.get_hash("preview_source")
            if sh or ph:
                s_meta: Dict[str, Any] = {"hash": {}}
                if sh:
                    s_meta["hash"]["source"] = sh
                if ph:
                    s_meta["hash"]["preview_source"] = ph
                s["meta"] = s_meta
            # masks
            masks_src: Dict[str, str] = {}
            get_masks = getattr(sub, "get_masks_source_dict", None)
            if callable(get_masks):
                masks_src = get_masks()
            else:
                masks_src = getattr(sub, "_masks_source", {}) or {}
            if masks_src:
                s["masks"] = [{"id": mid, "uri": muri} for mid, muri in sorted(masks_src.items())]
            sources.append(s)
            # aggregate rois
            for ann in getattr(sub, "_annotations", []) or []:
                rois.append(ann)
            # prefer first non-empty context
            if context_id is None and getattr(sub, "_context_id", None):
                context_id = sub._context_id
        frame["sources"] = sources
        if rois:
            frame["rois"] = rois
        if context_id is not None:
            frame["context_id"] = context_id
        return frame

    @classmethod
    def from_api_object(cls, frame: Any) -> "DataEntryImage":
        """
        Convert backend frame (dict/object) to DataEntryImage + DataSubEntryImage tree.
        """
        log = logging.getLogger("DataView")

        def _get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        try:
            raw_meta = _get(frame, "meta") or {}
            metadata = _copy_without_keys(raw_meta, ENTRY_CLASS_KEY)
            resolved_entry_cls = None
            if isinstance(raw_meta, dict):
                resolved_entry_cls = _resolve_class(raw_meta.get(ENTRY_CLASS_KEY), DataEntry)
                if resolved_entry_cls and not issubclass(resolved_entry_cls, DataEntry):
                    resolved_entry_cls = None
            entry = cls(data_entry_id=_get(frame, "id"), metadata=metadata)
            ctx = _get(frame, "context_id")
            sources = _get(frame, "sources") or []
            sub_entries: List[DataSubEntry] = []
            for idx, s in enumerate(sources):
                name = _get(s, "id") or f"image_{idx}"
                sub_meta_raw = raw_meta.get(name) if isinstance(raw_meta, dict) else None
                sub_meta_clean: Optional[Dict[str, Any]] = None
                resolved_cls: Optional[type] = None
                if isinstance(sub_meta_raw, dict):
                    class_path = sub_meta_raw.get(SUB_ENTRY_CLASS_KEY)
                    sub_meta_clean = _copy_without_keys(sub_meta_raw, SUB_ENTRY_CLASS_KEY)
                    resolved_cls = _resolve_class(class_path, DataSubEntry)
                    metadata[name] = sub_meta_clean
                sub = DataSubEntryImage.from_api_object(
                    s,
                    frame_meta=raw_meta if isinstance(raw_meta, dict) else {},
                    context_id=ctx,
                    name_fallback=name,
                )
                if isinstance(sub_meta_clean, dict):
                    sub._metadata = sub_meta_clean
                if resolved_cls and issubclass(resolved_cls, DataSubEntry):
                    try:
                        sub.__class__ = resolved_cls
                    except TypeError:
                        log.warning(
                            "Could not assign image sub-entry %s to class '%s'",
                            name,
                            resolved_cls.__name__,
                        )
                sub_entries.append(sub)
            if sub_entries:
                entry.add_sub_entries(sub_entries)
            # Map ROIs to sub-entries by sources list
            rois = _get(frame, "rois") or []
            by_name = {se.name: se for se in sub_entries}
            for roi in rois:
                srcs = _get(roi, "sources") or []
                targets = srcs or list(by_name.keys())[:1]
                for sid in targets:
                    se = by_name.get(sid)
                    if not se:
                        continue
                    if not hasattr(se, "_annotations"):
                        se._annotations = []  # type: ignore[attr-defined]
                    try:
                        se._annotations.append(roi)
                    except Exception as ex:
                        log.error("Failed attaching ROI to subentry %s: %s", sid, ex)
            if (
                resolved_entry_cls
                and isinstance(entry, DataEntry)
                and issubclass(resolved_entry_cls, DataEntry)
                and entry.__class__ is not resolved_entry_cls
            ):
                try:
                    entry.__class__ = resolved_entry_cls
                except TypeError:
                    log.warning(
                        "Could not assign data entry %s to class '%s'",
                        getattr(entry, "id", None),
                        resolved_entry_cls.__name__,
                    )
            return entry
        except Exception as ex:
            log.exception("from_api_object conversion failed: %s", ex)
            return cls()
