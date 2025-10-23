"""Utilities for HyperDataset DataView interactions used by Task."""
from typing import Any, Dict, Mapping, Sequence, TYPE_CHECKING

from clearml.debugging import get_logger

if TYPE_CHECKING:
    from clearml.backend_interface.task.task import Task


def set_dataview(task: "Task", dataview) -> None:
    """
    Store a HyperDatasets DataView definition into this Task using task properties
    (i.e. under `input.*`), so the DataView appears in the UI, without using
    runtime properties.

    - If `dataview` is a string id, the backend is queried to fetch its full
      definition and it is serialized into the task's `input` section.
    - If `dataview` is a `DataView`, its current state is serialized
      into the task's `input` section.

    :param dataview: DataView instance or dataview id string
    """
    # Avoid top-level import to prevent cycles
    try:
        from .data_view import (
            DataView,
            DataViewManagementBackend,
        )
    except Exception:
        DataView = None  # type: ignore
        DataViewManagementBackend = None  # type: ignore

    payload: Dict[str, Any] = {}

    # Utility: generate unique auxiliary dataview name (dataview_1, dataview_2, ...)
    def _generate_unique_name() -> str:
        try:
            existing_names = set()
            try:
                mapping = task._get_task_property(
                    "input.dataviews", raise_on_error=False, log_on_error=False, default={}
                )
                if isinstance(mapping, dict):
                    existing_names.update(mapping.keys())
            except Exception:
                pass
            try:
                exec_dvs = task._get_task_property(
                    "execution.dataviews", raise_on_error=False, log_on_error=False, default=[]
                ) or []
                for dv in exec_dvs:
                    n = dv.get("name") if isinstance(dv, dict) else getattr(dv, "name", None)
                    if n:
                        existing_names.add(n)
            except Exception:
                pass
            idx = 1
            while True:
                candidate = f"dataview_{idx}"
                if candidate not in existing_names:
                    return candidate
                idx += 1
        except Exception:
            return "dataview_1"

    # Helper: serialize to task.input
    def _serialize_from_parts(
        versions: Sequence[Mapping[str, Any]],
        filters: Sequence[Mapping[str, Any]],
        iteration: Mapping[str, Any],
    ) -> Dict[str, Any]:
        return {
            "input": {
                "view": {"entries": list(versions or [])},
                "frames_filter": {
                    "filtering_rules": list(filters or []),
                    "output_rois": "all_in_frame",
                },
                "augmentation": {},
                "iteration": {
                    "order": iteration.get("order"),
                    "infinite": bool(iteration.get("infinite", False)),
                    "random_seed": iteration.get("random_seed") or 1337,
                },
                # No auxiliary dataviews mapping in this simplified flow
                "dataviews": {},
            },
        }

    try:
        private_map = (
            task._get_task_property(
                "input._private_dataviews", raise_on_error=False, log_on_error=False, default={}
            )
            or {}
        )
    except Exception:
        private_map = {}
    if not isinstance(private_map, dict):
        private_map = {}

    # Case 1: dataview id, fetch from backend
    if isinstance(dataview, (str, bytes)):
        if not DataViewManagementBackend:
            return
        dv_id = dataview.decode() if isinstance(dataview, bytes) else str(dataview)
        # noinspection PyBroadException
        try:
            dv = DataViewManagementBackend.get_by_id(dv_id)
            if not dv:
                return
            # Extract versions
            versions = []
            for e in getattr(dv, "versions", []) or []:
                try:
                    versions.append({"dataset": e.dataset, "version": e.version})
                except Exception:
                    # e might be a dict already
                    if isinstance(e, dict):
                        ds = e.get("dataset")
                        ver = e.get("version")
                        if ds and ver:
                            versions.append({"dataset": ds, "version": ver})
            # Extract filters
            filters = []
            for fr in getattr(dv, "filters", []) or []:
                try:
                    filters.append(fr.to_dict())
                except Exception:
                    # try best-effort
                    if isinstance(fr, dict):
                        filters.append(fr)
            # Extract iteration
            iteration = {}
            it = getattr(dv, "iteration", None)
            if it:
                try:
                    iteration = {
                        "order": getattr(it, "order", None),
                        "infinite": bool(getattr(it, "infinite", False)),
                        "random_seed": getattr(it, "random_seed", None),
                    }
                except Exception:
                    if isinstance(it, dict):
                        iteration = {
                            "order": it.get("order"),
                            "infinite": bool(it.get("infinite", False)),
                            "random_seed": it.get("random_seed"),
                        }
            payload = _serialize_from_parts(versions=versions, filters=filters, iteration=iteration)
            # also store as a named dataview (using backend name or auto-generated)
            dv_name = getattr(dv, "name", None) or _generate_unique_name()
            # input.dataviews mapping
            mapping = (
                task._get_task_property("input.dataviews", raise_on_error=False, log_on_error=False, default={})
                or {}
            )
            if not isinstance(mapping, dict):
                mapping = {}
            mapping = {k: v for k, v in mapping.items() if k != dv_name}
            mapping[dv_name] = dv_id
            payload["input"]["dataviews"] = mapping
            # execution.dataviews list
            exec_dvs = (
                task._get_task_property("execution.dataviews", raise_on_error=False, log_on_error=False, default=[])
                or []
            )
            exec_dvs = [
                x
                for x in exec_dvs
                if (x.get("name") if isinstance(x, dict) else getattr(x, "name", None)) != dv_name
            ]
            dv_dict = {
                "name": dv_name,
                "filters": list(filters or []),
                "versions": list(versions or []),
                "iteration": {
                    "order": iteration.get("order"),
                    "infinite": bool(iteration.get("infinite", False)),
                    "random_seed": iteration.get("random_seed"),
                },
                "augmentation": {},
            }
            existing_meta = private_map.get(dv_name)
            if isinstance(existing_meta, dict) and existing_meta:
                dv_dict["_private_metadata"] = dict(existing_meta)
            # Keep existing execution attributes, only set dataviews
            payload["execution"] = {
                "parameters": getattr(getattr(task.data, "execution", {}), "parameters", {}) or {},
                "model_desc": getattr(getattr(task.data, "execution", {}), "model_desc", {}) or {},
                "dataviews": exec_dvs + [dv_dict],
                "artifacts": getattr(getattr(task.data, "execution", {}), "artifacts", []) or [],
            }
        except Exception as e:
            get_logger("task").exception("Failed fetching dataview by id {}: {}".format(dv_id, e))
            return
    # Case 2: SDK DataView object
    elif DataView and isinstance(dataview, DataView):
        # Build versions and filters from object internals
        try:
            # Ensure the dataview exists to get an id
            try:
                dataview._ensure_created()
            except Exception:
                pass
            dv_id = getattr(dataview, "_id", None)
            dv_name = getattr(dataview, "name", None) or _generate_unique_name()
            # Set back onto the object via its setter (if available)
            try:
                if not getattr(dataview, "name", None):
                    setattr(dataview, "name", dv_name)
            except Exception:
                pass
            # Versions from queries (concrete dataset/version pairs)
            versions = []
            seen = set()
            for q in getattr(dataview, "_queries", []) or []:
                ds = getattr(q, "dataset_id", None)
                ver = getattr(q, "version_id", None)
                if ds and ver and ds != "*" and ver != "*" and (ds, ver) not in seen:
                    versions.append({"dataset": ds, "version": ver})
                    seen.add((ds, ver))
            # Filters
            filters = []
            for fr in getattr(dataview, "_filter_rules", []) or []:
                try:
                    filters.append(fr.to_dict())
                except Exception:
                    filters.append(
                        {
                            "dataset": getattr(fr, "dataset", None),
                            "version": getattr(fr, "version", None),
                            "frame_query": getattr(fr, "frame_query", None),
                            "sources_query": getattr(fr, "sources_query", None),
                            "filter_by_roi": getattr(fr, "filter_by_roi", None),
                            "label_rules": getattr(fr, "label_rules", None),
                            "weight": getattr(fr, "weight", None),
                        }
                    )
            iteration = {
                "order": getattr(dataview, "_iteration_order", None),
                "infinite": bool(getattr(dataview, "_iteration_infinite", False)),
                "random_seed": getattr(dataview, "_iteration_random_seed", None),
            }
            payload = _serialize_from_parts(versions=versions, filters=filters, iteration=iteration)
            # also store as a named dataview
            mapping = (
                task._get_task_property("input.dataviews", raise_on_error=False, log_on_error=False, default={})
                or {}
            )
            if not isinstance(mapping, dict):
                mapping = {}
            mapping = {k: v for k, v in mapping.items() if k != dv_name}
            if dv_id:
                mapping[dv_name] = dv_id
            payload["input"]["dataviews"] = mapping
            exec_dvs = (
                task._get_task_property("execution.dataviews", raise_on_error=False, log_on_error=False, default=[])
                or []
            )
            exec_dvs = [
                x
                for x in exec_dvs
                if (x.get("name") if isinstance(x, dict) else getattr(x, "name", None)) != dv_name
            ]
            dv_dict = {
                "name": dv_name,
                "filters": list(filters or []),
                "versions": list(versions or []),
                "iteration": {
                    "order": iteration.get("order"),
                    "infinite": bool(iteration.get("infinite", False)),
                    "random_seed": iteration.get("random_seed"),
                },
                "augmentation": {},
            }
            private_meta = getattr(dataview, "_private_metadata", None)
            if isinstance(private_meta, dict) and private_meta:
                private_map[dv_name] = dict(private_meta)
                dv_dict["_private_metadata"] = dict(private_meta)
            elif dv_name in private_map:
                private_map.pop(dv_name, None)
            payload["execution"] = {
                "parameters": getattr(getattr(task.data, "execution", {}), "parameters", {}) or {},
                "model_desc": getattr(getattr(task.data, "execution", {}), "model_desc", {}) or {},
                "dataviews": exec_dvs + [dv_dict],
                "artifacts": getattr(getattr(task.data, "execution", {}), "artifacts", []) or [],
            }
        except Exception as e:
            get_logger("task").exception("Failed serializing DataView to task input: {}".format(e))
            return
    else:
        return

    payload["input"]["_private_dataviews"] = private_map

    try:
        task._edit(**payload)
        task.reload()
    except Exception as e:
        get_logger("task").warning("Failed applying dataview payload onto task input: {}".format(e))



def get_dataviews(task: "Task") -> Dict[str, Any]:
    """
    Return a dictionary of HyperDatasets DataView objects reconstructed from this Task
    task properties (primarily from `input.*`). Keys are arbitrary labels.
    """
    dvs: Dict[str, Any] = {}
    # Avoid top-level import to prevent cycles
    try:
        from .data_view import (
            DataView,
            DataViewManagementBackend,
            HyperDatasetQuery,
        )
    except Exception:
        return dvs

    data_input = getattr(task.data, "input", None)
    if not data_input:
        return dvs

    # Prefer named dataviews if available
    private_map: Dict[str, Any] = {}
    try:
        private_map = task._get_task_property(
            "input._private_dataviews", raise_on_error=False, log_on_error=False, default={}
        ) or {}
        if not isinstance(private_map, dict):
            private_map = {}
        mapping = task._get_task_property("input.dataviews", raise_on_error=False, log_on_error=False, default={}) or {}
        exec_dvs = task._get_task_property("execution.dataviews", raise_on_error=False, log_on_error=False, default=[]) or []
        exec_by_name = {}
        for item in exec_dvs:
            nm = item.get("name") if isinstance(item, dict) else getattr(item, "name", None)
            if nm:
                exec_by_name[nm] = item
        for name, dv_id in (mapping.items() if isinstance(mapping, dict) else []):
            item = exec_by_name.get(name) or {}
            # build dv from item fields
            dv = DataView(
                iteration_order="sequential", iteration_infinite=False, auto_connect_with_task=False
            )
            dv._id = dv_id
            it = (item.get("iteration") if isinstance(item, dict) else getattr(item, "iteration", None)) or {}
            if isinstance(it, dict):
                dv._iteration_order = it.get("order", dv._iteration_order)
                dv._iteration_infinite = bool(it.get("infinite", dv._iteration_infinite))
                dv._iteration_random_seed = it.get("random_seed", dv._iteration_random_seed)
            rebuilt = []
            filters = (item.get("filters") if isinstance(item, dict) else getattr(item, "filters", None)) or []
            for fr in filters or []:
                try:
                    rebuilt.append(
                        DataViewManagementBackend.create_filter_rule(
                            dataset=fr.get("dataset"),
                            label_rules=fr.get("label_rules"),
                            filter_by_roi=fr.get("filter_by_roi"),
                            frame_query=fr.get("frame_query"),
                            sources_query=fr.get("sources_query"),
                            version=fr.get("version"),
                            weight=fr.get("weight"),
                        )
                    )
                except Exception:
                    continue
            dv._filter_rules = rebuilt
            versions = (item.get("versions") if isinstance(item, dict) else getattr(item, "versions", None)) or []
            for ve in versions or []:
                ds = ve.get("dataset")
                ver = ve.get("version")
                if ds and ver:
                    dv._queries.append(HyperDatasetQuery(dataset_id=ds, version_id=ver))
            private_meta = {}
            if isinstance(item, dict):
                private_meta = item.get("_private_metadata") or {}
            if not private_meta:
                private_meta = private_map.get(name, {})
            if private_meta:
                dv._apply_private_metadata(private_meta)
            dvs[str(name)] = dv
    except Exception:
        try:
            get_logger("task").exception("Failed reconstructing named DataViews from task input/execution")
        except Exception:
            pass

    # Fallback: construct a single primary entry if none were found
    if not dvs:
        try:
            dv = DataView(
                iteration_order="sequential", iteration_infinite=False, auto_connect_with_task=False
            )
            # iteration
            it = getattr(data_input, "iteration", None) or {}
            try:
                dv._iteration_order = getattr(it, "order", dv._iteration_order)
                dv._iteration_infinite = bool(getattr(it, "infinite", dv._iteration_infinite))
                dv._iteration_random_seed = getattr(it, "random_seed", dv._iteration_random_seed)
            except Exception:
                if isinstance(it, dict):
                    dv._iteration_order = it.get("order", dv._iteration_order)
                    dv._iteration_infinite = bool(it.get("infinite", dv._iteration_infinite))
                    dv._iteration_random_seed = it.get("random_seed", dv._iteration_random_seed)
            # filters
            rebuilt = []
            ff = getattr(data_input, "frames_filter", None) or {}
            filters = getattr(ff, "filtering_rules", None) or (ff.get("filtering_rules") if isinstance(ff, dict) else [])
            for fr in filters or []:
                try:
                    rebuilt.append(
                        DataViewManagementBackend.create_filter_rule(
                            dataset=fr.get("dataset"),
                            label_rules=fr.get("label_rules"),
                            filter_by_roi=fr.get("filter_by_roi"),
                            frame_query=fr.get("frame_query"),
                            sources_query=fr.get("sources_query"),
                            version=fr.get("version"),
                            weight=fr.get("weight"),
                        )
                    )
                except Exception:
                    continue
            dv._filter_rules = rebuilt
            # versions -> queries
            view = getattr(data_input, "view", None) or {}
            entries = getattr(view, "entries", None) or (view.get("entries") if isinstance(view, dict) else [])
            for ve in entries or []:
                ds = getattr(ve, "dataset", None) if not isinstance(ve, dict) else ve.get("dataset")
                ver = getattr(ve, "version", None) if not isinstance(ve, dict) else ve.get("version")
                if ds and ver:
                    dv._queries.append(HyperDatasetQuery(dataset_id=ds, version_id=ver))
            if isinstance(private_map, dict) and private_map:
                first_meta = next(iter(private_map.values()), {})
                if first_meta:
                    dv._apply_private_metadata(first_meta)
            dvs["primary"] = dv
        except Exception:
            try:
                get_logger("task").exception("Failed reconstructing DataView from task input")
            except Exception:
                pass

    return dvs
