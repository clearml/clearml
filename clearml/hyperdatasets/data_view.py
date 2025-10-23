import threading
import queue
from queue import Queue
from math import ceil
from typing import Any, Iterable, List, Optional, Sequence

try:
    from luqum.parser import parser as lucene_parser
except ImportError:
    lucene_parser = None
try:
    from luqum.exceptions import ParseError as LuceneParseError
except ImportError:
    # Backwards compatibility for luqum<=0.9.0
    try:
        from luqum.parser import ParseError as LuceneParseError
    except ImportError:
        pass

from clearml.backend_interface.datasets.hyper_dataset_data_view import DataViewManagementBackend
from clearml.config import running_remotely, get_remote_task_id, get_node_id, get_node_count
from clearml.task import Task
from clearml.storage.manager import StorageManagerDiskSpaceFileSizeStrategy
import logging
from .data_entry import DataEntry, DataSubEntry, ENTRY_CLASS_KEY, _resolve_class
from .data_entry_image import DataEntryImage
from .management import HyperDatasetManagement


_UNSET = object()


class HyperDatasetQuery:
    lucene_parser_warning_sent = False

    @classmethod
    def _validate_lucene(cls, lucene_query):
        """Validate the supplied Lucene query string using `luqum`.

        Empty strings are considered valid. Non-empty values are parsed and raise a
        `LuceneParseError` when the expression is malformed.

        :param lucene_query: Lucene query string to validate
        :return: None
        """
        if not lucene_parser:
            if not cls.lucene_parser_warning_sent:
                logging.getLogger("DataView").warning("Could not validate lucene query because 'luqum' is not installed. "
                    "Run 'pip install luqum' to enable query validation"
                )
                cls.lucene_parser_warning_sent = True
            return
            
        if not lucene_query:
            return
        try:
            lucene_parser.parse(lucene_query)
        except LuceneParseError as e:
            raise type(e)("Failed parsing lucene query '{}': {}".format(lucene_query, e))

    def __init__(
        self,
        project_id="*",   # ClearML datasets: collection id
        dataset_id="*",   # ClearML datasets: version id
        version_id="*",   # Alias for clarity; kept for symmetry
        source_query=None,
        frame_query=None,
        weight=1.0,
        filter_by_roi=None,
        label_rules=None,
    ):
        """Construct a hyper-dataset query filter.

        When concrete dataset/version IDs are supplied the constructor verifies their existence via
        `HyperDatasetManagement`. Optional Lucene queries, ROI filtering, and sampling weights can be
        provided to further refine the query.

        :param project_id: Dataset collection identifier or wildcard
        :param dataset_id: Dataset identifier or wildcard (legacy) used when version is omitted
        :param version_id: Dataset version identifier; defaults to `dataset_id` when empty
        :param source_query: Lucene query applied to frame source metadata
        :param frame_query: Lucene query applied to frame metadata
        :param weight: Relative sampling weight for this query
        :param filter_by_roi: Optional ROI filtering strategy
        :param label_rules: Optional label-rule dictionaries for ROI filtering
        """
        HyperDatasetQuery._validate_lucene(source_query)
        HyperDatasetQuery._validate_lucene(frame_query)
        self._project_id = project_id
        # Prefer explicit version_id if provided, else dataset_id acts as version id
        self._dataset_id = dataset_id
        self._version_id = version_id or dataset_id
        self._validate_dataset_and_version()
        self._source_query = source_query
        self._frame_query = frame_query
        self._weight = weight
        self._filter_by_roi = filter_by_roi
        self._label_rules = label_rules

    @property
    def dataset_id(self):
        """
        Return the dataset identifier targeted by this query.

        :return: Dataset ID string or wildcard marker
        """
        return self._dataset_id

    @property
    def project_id(self):
        """
        Return the dataset collection identifier associated with this query.

        :return: Project ID string or wildcard marker
        """
        return self._project_id

    @property
    def version_id(self):
        """
        Return the dataset version identifier resolved for this query.

        :return: Version ID string or wildcard marker
        """
        return self._version_id

    @property
    def source_query(self):
        """
        Return the Lucene query applied to frame source metadata.

        :return: Lucene query string or None
        """
        return self._source_query

    @property
    def frame_query(self):
        """
        Return the Lucene query applied to frame-level metadata.

        :return: Lucene query string or None
        """
        return self._frame_query

    @property
    def weight(self):
        """
        Return the relative sampling weight assigned to this query.

        :return: Sampling weight as a float
        """
        return self._weight

    @property
    def filter_by_roi(self):
        """
        Return the ROI filtering strategy configured for this query.

        :return: ROI filter identifier or None
        """
        return self._filter_by_roi

    @property
    def label_rules(self):
        """
        Return the label rule definitions used for ROI filtering.

        :return: Sequence or mapping of label rules, or None
        """
        return self._label_rules

    def _validate_dataset_and_version(self):
        """Verify that referenced dataset and version identifiers exist on the backend."""
        if self._dataset_id in (None, "*"):
            return

        version_id = self._version_id if self._version_id not in (None, "*") else None

        exists = HyperDatasetManagement.exists(
            dataset_id=self._dataset_id,
            version_id=version_id,
        )
        if not exists:
            raise ValueError(
                "HyperDataset query references non-existent dataset/version: dataset_id={} version_id={}".format(
                    self._dataset_id, self._version_id
                )
            )

class DataView:
    _MAX_BATCH_SIZE = 10000
    _DEFAULT_LOCAL_BATCH_SIZE = 500

    def __init__(
        self,
        name=None,
        description=None,
        tags=None,
        iteration_order="sequential",
        iteration_infinite=False,
        iteration_random_seed=None,
        iteration_limit=None,
        auto_connect_with_task=True,
    ):
        """
        Instantiate a `DataView` wrapper around backend dataview resources.

        The dataview aggregates query rules and iteration parameters. When running under a ClearML task it
        can optionally auto-connect and restore previously attached definitions.

        :param name: Optional dataview name
        :param description: Optional descriptive text
        :param tags: Optional list of tag strings
        :param iteration_order: Iteration order (`sequential` or `random`)
        :param iteration_infinite: Whether to iterate indefinitely
        :param iteration_random_seed: Seed used for random iteration
        :param iteration_limit: Explicit maximum number of frames to iterate (None means unlimited)
        :param auto_connect_with_task: Auto-attach to the current ClearML task when True
        """
        self._iteration_order = iteration_order
        self._iteration_infinite = iteration_infinite
        self._iteration_limit = iteration_limit if iteration_limit is None else int(iteration_limit)
        # TODO: connect with task in remote execution
        self._auto_connect_with_task = auto_connect_with_task
        self._iteration_random_seed = iteration_random_seed
        self._name = name
        self._description = description
        self._tags = tags
        self._id = None
        self._filter_rules: List[Any] = []
        self._queries: List[HyperDatasetQuery] = []
        self._count_cache = None
        self._synthetic_epoch_limit = None
        self._private_metadata = {}
        self._force_remote_store = False
        # If running remotely under a Task, try to attach using Task helpers
        # Only do this when auto_connect_with_task is enabled to avoid recursion
        try:
            if running_remotely() and self._auto_connect_with_task:
                task = Task.current_task()
                if not task:
                    tid = get_remote_task_id()
                    if tid:
                        task = Task.get_task(task_id=tid)
                if task:
                    self._connected_task = task
                    dv_map = task.get_dataviews() or {}
                    # If a name was provided, prefer a matching dataview by name
                    picked = None
                    if isinstance(dv_map, dict):
                        if self._name:
                            picked = dv_map.get(self._name)
                        else:
                            for _dv in dv_map.values():
                                picked = _dv
                                break
                    if picked:
                        try:
                            self._copy_from_other_dataview(picked)
                            self._force_remote_store = False
                        except Exception:
                            self._force_remote_store = True
                    else:
                        self._force_remote_store = True
        except Exception:
            pass

    @property
    def id(self):
        """
        Return the backend identifier of the materialised DataView.

        :return: DataView ID string or None when not yet created
        """
        return self._id

    @property
    def name(self):
        """
        Return the human-readable name assigned to this DataView.

        :return: DataView name string or None
        """
        return self._name

    @name.setter
    def name(self, value):
        """
        Update the human-readable name associated with this DataView.

        :param value: New DataView name string or None
        """
        self._name = value

    def get_queries(self):
        """Return current HyperDatasetQuery objects attached to this dataview."""
        return list(self._queries)

    def _mutation_allowed(self) -> bool:
        try:
            if running_remotely() and getattr(self, "_auto_connect_with_task", False):
                dv_id = getattr(self, "_id", None)
                if not dv_id:
                    return True
                if not self._queries and not self._filter_rules:
                    return True
                return False
        except Exception:
            pass
        return True

    def _build_filter_rule_from_query(self, query: "HyperDatasetQuery") -> Any:
        return DataViewManagementBackend.create_filter_rule(
            dataset=query.dataset_id,
            label_rules=query.label_rules,
            filter_by_roi=query.filter_by_roi,
            frame_query=query.frame_query,
            sources_query=query.source_query,
            version=query.version_id,
            weight=query.weight,
        )

    def _append_queries(self, queries: Sequence["HyperDatasetQuery"]) -> None:
        if not queries:
            return
        filter_rules: List[Any] = []
        for query in queries:
            if not isinstance(query, HyperDatasetQuery):
                raise ValueError("DataView expects HyperDatasetQuery instances")
            filter_rules.append(self._build_filter_rule_from_query(query))
        self._filter_rules.extend(filter_rules)
        self._queries.extend(queries)
        self._count_cache = None
        self._synthetic_epoch_limit = None
        self._resync_task_attachment()
        if self._id:
            result = DataViewManagementBackend.update_filter_rules(
                dataview_id=self._id, filter_rules=self._filter_rules
            )
            if not result:
                raise ValueError("Failed updating DataView {}".format(self._id))
            self._resync_task_attachment()

    def set_queries(self, queries: Optional[Iterable["HyperDatasetQuery"]]) -> None:
        """
        Replace all existing queries with the supplied collection.

        :param queries: Iterable of `HyperDatasetQuery` objects; pass None or an empty iterable to clear
        """
        if not self._mutation_allowed():
            return
        normalized = list(queries) if queries is not None else []
        self._filter_rules = []
        self._queries = []
        self._count_cache = None
        self._synthetic_epoch_limit = None
        if not normalized:
            if self._id:
                DataViewManagementBackend.update_filter_rules(
                    dataview_id=self._id, filter_rules=[]
                )
            self._resync_task_attachment()
            return
        self._append_queries(normalized)

    def add_query(
        self,
        *,
        project_id: str = "*",
        dataset_id: str = "*",
        version_id: str = "*",
        source_query=None,
        frame_query=None,
        weight: Optional[float] = 1.0,
        filter_by_roi=None,
        label_rules=None,
    ) -> "HyperDatasetQuery":
        """
        Construct and append a single `HyperDatasetQuery` without instantiating it externally.

        :param project_id: Dataset collection identifier or wildcard
        :param dataset_id: Dataset identifier or wildcard
        :param version_id: Dataset version identifier
        :param source_query: Lucene query applied to frame sources
        :param frame_query: Lucene query applied to frame metadata
        :param weight: Sampling weight when combining multiple queries
        :param filter_by_roi: ROI filtering strategy name
        :param label_rules: Optional label rule definitions for ROI filtering
        :return: The created `HyperDatasetQuery` instance
        """
        query = HyperDatasetQuery(
            project_id=project_id,
            dataset_id=dataset_id,
            version_id=version_id,
            source_query=source_query,
            frame_query=frame_query,
            weight=weight,
            filter_by_roi=filter_by_roi,
            label_rules=label_rules,
        )
        self.add_queries(query)
        return query

    def get_iteration_parameters(self):
        """
        :return: The cached iteration configuration for this dataview.
        """
        return {
            "order": self._iteration_order,
            "infinite": self._iteration_infinite,
            "limit": self._iteration_limit,
            "random_seed": self._iteration_random_seed,
        }

    def set_iteration_parameters(self, *, infinite=None, limit=_UNSET):
        """
        Persist iteration settings both locally and on the backend if possible.
        """
        updated = False
        if infinite is not None:
            self._iteration_infinite = bool(infinite)
            updated = True
        if limit is not _UNSET:
            self._iteration_limit = int(limit) if limit is not None else None
            updated = True
        if not updated:
            return
        if self._id:
            DataViewManagementBackend.update_iteration_parameters(
                self._id,
                infinite=self._iteration_infinite,
                limit=self._iteration_limit,
                order=self._iteration_order,
                random_seed=self._iteration_random_seed,
            )

    def add_queries(self, queries: HyperDatasetQuery):
        """
        Append one or more query rules to the dataview.

        If the dataview already exists on the backend the remote filter rules are updated immediately and
        the attached task is re-synchronised.

        :param queries: A `HyperDatasetQuery` instance or iterable of instances to add
        """
        if not self._mutation_allowed():
            return
        if isinstance(queries, HyperDatasetQuery):
            normalized: Sequence[HyperDatasetQuery] = [queries]
        else:
            try:
                normalized = list(queries)
            except TypeError as exc:
                raise ValueError("DataView.add_queries expects a query or an iterable of queries") from exc
        self._append_queries(normalized)

    def _ensure_created(self):
        """
        Ensure a matching backend dataview resource exists.

        The method lazily creates the dataview when first required, reusing the existing resource when it
        already exists. Raises a `ValueError` if no concrete dataset/version pairs can be derived from the
        configured queries.

        :return: None
        """
        if self._id:
            # If running remotely and we already have an id, verify it exists server-side
            try:
                if running_remotely():
                    existing = DataViewManagementBackend.get_by_id(self._id)
                    if existing:
                        return
            except Exception:
                pass
            # If not remote or fetch failed, assume id is valid and return
            if not running_remotely():
                return
        # Build versions from queries; require at least one concrete (dataset, version)
        versions = []
        for q in self._queries:
            ds = getattr(q, "dataset_id", None)
            ver = getattr(q, "version_id", None)
            if ds and ver and ds != "*" and ver != "*":
                versions.append({"dataset": ds, "version": ver})
        if not versions:
            raise ValueError("Cannot create DataView: no concrete (dataset, version) provided in queries")
        self._id = DataViewManagementBackend.create(
            name=self._name,
            description=self._description,
            tags=self._tags,
            infinite=self._iteration_infinite,
            order=self._iteration_order,
            random_seed=self._iteration_random_seed,
            limit=self._iteration_limit,
            versions=versions,
        )
        if self._filter_rules:
            DataViewManagementBackend.update_filter_rules(
                dataview_id=self._id, filter_rules=self._filter_rules
            )
        self._count_cache = None
        self._resync_task_attachment()

    def _store_attachment_on_task(self, *, force_remote: bool = False):
        """
        Persist this dataview definition into the current Task using Task helpers.
        """
        try:
            is_remote = False
            try:
                is_remote = running_remotely()
            except Exception:
                is_remote = False
            if is_remote and not force_remote:
                return
            task = None
            if force_remote:
                task = getattr(self, "_connected_task", None)
                if not task:
                    tid = get_remote_task_id()
                    if tid:
                        task = Task.get_task(task_id=tid)
            if not task:
                task = Task.current_task()
            if not task:
                return
            payload = self.id if (force_remote and self._id) else self
            task.set_dataview(payload)
        except Exception:
            return

    def _resync_task_attachment(self):
        """
        Helper to store current dataview state on the Task when auto-connect is enabled.
        """
        if self._auto_connect_with_task:
            # On remote, avoid modifying the task silently
            try:
                if running_remotely() and not self._force_remote_store:
                    return
            except Exception:
                pass
            self._store_attachment_on_task(force_remote=self._force_remote_store)

    def _calculate_synthetic_epoch_limit(self):
        """
        Compute the synthetic epoch size when allow_repetition is enabled.
        """
        if not self._id:
            return None
        queries = self.get_queries()
        if len(queries) <= 1:
            return None

        weights = [float(q.weight) if q.weight is not None else 1.0 for q in queries]
        if all(q.weight is None for q in queries) and not self._iteration_infinite:
            return None

        total, rule_counts = DataViewManagementBackend.get_count_details_for_id(self._id)
        if total and not self._count_cache:
            self._count_cache = int(total)
        if not rule_counts:
            return None

        if len(rule_counts) < len(queries):
            rule_counts.extend([0] * (len(queries) - len(rule_counts)))

        positive = [(c, w) for c, w in zip(rule_counts, weights) if c > 0]
        if not positive:
            return None

        sum_weights = sum(w for _, w in positive)
        normalized = []
        for count, weight in zip(rule_counts, weights):
            if count <= 0:
                normalized.append(0.0)
            else:
                normalized.append(weight / (sum_weights or 1.0))

        max_count = max(rule_counts)
        if max_count <= 0:
            return None
        largest_idx = next((i for i, count in enumerate(rule_counts) if count == max_count), 0)
        weight_fraction = normalized[largest_idx]
        if weight_fraction <= 0:
            return None

        return int(ceil(max_count / weight_fraction))

    def _auto_connect_task(self):
        """
        Ensure this DataView is connected to the current Task (locally or remotely).

        In local runs, pushes the DataView state into the Task. In remote runs, also
        attempts to pull from Task if already stored.
        """
        try:
            task = Task.current_task()
            if not task and running_remotely():
                tid = get_remote_task_id()
                if tid:
                    task = Task.get_task(task_id=tid)
            if task:
                self._connected_task = task
                # Try to reuse a dataview from the task. If none exists or creation fails
                # (for example when no remote dataview is attached), fallback to a fresh
                # instance without auto-connect.
                try:
                    self._store_attachment_on_task()
                except ValueError:
                    self._auto_connect_with_task = False
                    self._connected_task = None
                    return
        except Exception:
            pass

    def _copy_from_other_dataview(self, other: "DataView") -> None:
        """
        Copy internal state from another DataView instance.
        """
        if not other:
            return
        self._id = getattr(other, "_id", self._id)
        self._iteration_order = getattr(other, "_iteration_order", self._iteration_order)
        self._iteration_infinite = getattr(other, "_iteration_infinite", self._iteration_infinite)
        self._iteration_random_seed = getattr(other, "_iteration_random_seed", self._iteration_random_seed)
        self._iteration_limit = getattr(other, "_iteration_limit", self._iteration_limit)
        self._filter_rules = list(getattr(other, "_filter_rules", []))
        self._queries = list(getattr(other, "_queries", []))
        self._synthetic_epoch_limit = getattr(other, "_synthetic_epoch_limit", self._synthetic_epoch_limit)
        self._private_metadata = dict(getattr(other, "_private_metadata", self._private_metadata) or {})

    def get_iterator(
        self,
        projection=None,
        query_cache_size=None,
        query_queue_depth=5,
        allow_repetition=False,
        auto_synthetic_epoch_limit=None,
        node_id=None,
        worker_index=None,
        num_workers=None,
        cache_in_memory=False,
    ):
        """
        Return an iterator configured to stream frames for this dataview.

        :param projection: Optional projection list selecting frame fields
        :param query_cache_size: Number of frames to request per backend batch
        :param query_queue_depth: Queue depth used by the background fetcher
        :param allow_repetition: Enable synthetic epoch length balancing across queries
        :param auto_synthetic_epoch_limit: Legacy flag equivalent to `allow_repetition`
        :param node_id: Explicit node identifier to send to the backend
        :param worker_index: Worker index when splitting frames across multiple iterators
        :param num_workers: Total number of cooperating workers
        :param cache_in_memory: Reserved flag (currently unused)

        :return: Iterator streaming `DataEntry`-derived objects
        """
        if query_cache_size is None:
            query_cache_size = self._MAX_BATCH_SIZE if running_remotely() else self._DEFAULT_LOCAL_BATCH_SIZE
        # Lazily create dataview on first iteration
        self._ensure_created()

        synthetic_limit = None
        enable_repetition = bool(allow_repetition or auto_synthetic_epoch_limit)
        if enable_repetition:
            synthetic_limit = self._calculate_synthetic_epoch_limit()
            if synthetic_limit:
                iteration_params = self.get_iteration_parameters()
                current_limit = iteration_params.get("limit")
                logger = logging.getLogger("DataView")
                if not iteration_params.get("infinite") or (
                    current_limit and current_limit < synthetic_limit
                ):
                    logger.warning(
                        "DataView is finite without repetition, enabling repetition support infinite=True "
                        "and maximum_number_of_frames=%s",
                        synthetic_limit,
                    )
                else:
                    logger.info(
                        "allow_repetition: Setting DataView iterator maximum_number_of_frames=%s",
                        synthetic_limit,
                    )
                self.set_iteration_parameters(infinite=True, limit=synthetic_limit)
                self._synthetic_epoch_limit = synthetic_limit
            else:
                self._synthetic_epoch_limit = None
        else:
            self._synthetic_epoch_limit = None

        if num_workers is not None and worker_index is None and node_id is not None:
            worker_index = node_id

        if node_id is None:
            try:
                node_id = get_node_id()
            except Exception:
                node_id = None

        iterator = DataView.Iterator(
            dataview=self,
            projection=list(projection) if projection else None,
            query_cache_size=query_cache_size,
            query_queue_depth=query_queue_depth,
            synthetic_limit=synthetic_limit,
            node_id=node_id,
            worker_index=worker_index,
            num_workers=num_workers,
            cache_in_memory=cache_in_memory,
        )

        limit_value = getattr(iterator, "limit", None)
        if enable_repetition:
            self._synthetic_epoch_limit = limit_value

        return iterator

    def get_count(self) -> int:
        """
        Fetch total frames count from backend and cache it.

        Requires the dataview to be created. If backend is unavailable, returns 0.
        """
        self._ensure_created()
        if self._count_cache is not None:
            return self._count_cache
        total = DataViewManagementBackend.get_count_total_for_id(self._id)
        self._count_cache = int(total or 0)
        return self._count_cache

    def __len__(self) -> int:
        if self._iteration_limit is not None:
            return int(self._iteration_limit)
        if self._synthetic_epoch_limit is not None:
            return int(self._synthetic_epoch_limit)
        return self.get_count()

    def prefetch_local_sources(
        self,
        num_workers: int = None,
        wait: bool = True,
        query_cache_size: int = None,
        get_previews: bool = False,
        get_masks: bool = True,
        force_download: bool = False,
    ):
        """
        Prefetch data entry sources (and optionally previews/masks) into the local cache.

        - num_workers: number of worker threads (defaults to cpu count via ThreadPoolExecutor)
        - wait: block until all prefetch tasks complete
        - query_cache_size: data entries per backend fetch batch (defaults to iterator default)
        - get_previews: also prefetch preview URIs if available
        - get_masks: also prefetch mask URIs if available
        - force_download: bypass local cache if True
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        self._ensure_created()

        it = self.get_iterator(query_cache_size=query_cache_size)

        def _extract_uris(data_entry):
            uris = []
            sources = ["source"]
            if get_previews:
                sources += ["preview_source"]
            if get_masks:
                sources += ["mask_source"]
            for source in sources:
                for data_sub_entry in data_entry:
                    url = data_sub_entry.get_source(source)
                    if url:
                        uris.append(url)
            return uris

        # Prefetch using a thread pool
        futures = []
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            for data_entry in it:
                for uri in _extract_uris(data_entry):
                    futures.append(
                        pool.submit(StorageManagerDiskSpaceFileSizeStrategy.get_local_copy, uri, None, True, None, force_download)
                    )
            if wait:
                for f in as_completed(futures):
                    try:
                        _ = f.result()
                    except Exception:
                        continue

    class Iterator:
        def __init__(
            self,
            dataview=None,
            projection=None,
            query_cache_size=None,
            query_queue_depth=None,
            synthetic_limit=None,
            node_id=None,
            worker_index=None,
            num_workers=None,
            cache_in_memory=False,
        ):
            """
            Initialise the iterator wrapper that pulls data_entries from the backend.
            """
            self._dataview = dataview
            self._projection = None
            if projection:
                try:
                    if any(p == "*" for p in projection):
                        self._projection = None
                    else:
                        self._projection = list(projection)
                except Exception:
                    self._projection = None
            self._query_cache_size = int(query_cache_size or DataView._DEFAULT_LOCAL_BATCH_SIZE)
            self._query_queue_depth = int(query_queue_depth or 5)
            capacity = max(1, self._query_queue_depth)
            self._data_entries_queue: Queue = Queue(maxsize=capacity)
            self._stop_event = threading.Event()
            self._started = False
            self._closed = False
            self._error = None
            self._fetch_thread = threading.Thread(target=self._fetcher_daemon, name="HDVFetcher", daemon=True)
            self._logger = logging.getLogger("DataView")
            self._base_limit = int(synthetic_limit) if synthetic_limit is not None else None
            self._limit = self._base_limit
            self._yielded = 0
            self._produced = 0
            self._dispatch_counter = 0
            self._node_id = None
            self._num_workers = None
            self._worker_index = None
            self._cache_in_memory = cache_in_memory
            self._cache = []
            self._full_cache = False
            self._current_items = []
            if node_id is not None:
                self.set_node(node_id)
            if worker_index is not None or num_workers is not None:
                self.set_concurrency(worker_index=worker_index, num_workers=num_workers)
            else:
                # attempt automatic detection (no-op if single worker)
                self.set_concurrency()

        def __iter__(self):
            """
            Return the iterator instance after ensuring the fetch thread is running.
            """
            if self._cache_in_memory and self._full_cache:
                return self._cache.__iter__()
            self._yielded = 0
            if not self._started or self._closed or getattr(self, "_eof_reached", False):
                self._reset_fetch()
                self._started = True
                self._fetch_thread.start()
            return self

        def __next__(self):
            """
            Fetch the next data-entry object, respecting synthetic epoch limits.
            """
            if self._limit is not None and self._yielded >= self._limit:
                self._stop_event.set()
                self._closed = True
                self._eof_reached = True
                if self._error:
                    raise self._error
                self._full_cache = True
                raise StopIteration
            if (
                (self._closed or getattr(self, "_eof_reached", False))
                and not self._current_items
                and self._data_entries_queue.empty()
            ):
                if self._error:
                    raise self._error
                raise StopIteration
            while True:
                if self._error:
                    raise self._error
                try:
                    if not self._current_items:
                        self._current_items = self._data_entries_queue.get(timeout=0.5)
                    item = self._current_items.pop()
                    self._yielded += 1
                    if self._limit is not None and self._yielded >= self._limit:
                        self._stop_event.set()
                        self._closed = True
                        self._eof_reached = True
                    if self._cache_in_memory:
                        self._cache.append(item)
                    return item
                except queue.Empty:
                    if (
                        (self._closed or getattr(self, "_eof_reached", False))
                        or (not self._fetch_thread.is_alive() and self._data_entries_queue.empty())
                    ):
                        if self._error:
                            raise self._error
                        self._full_cache = True
                        raise StopIteration
                    continue

        def __len__(self):
            """
            Return the planned length of the iterator, if available.
            """
            if self._limit is not None:
                return self._limit
            try:
                return int(self._dataview.get_count()) if self._dataview else 0
            except Exception:
                return 0

        @property
        def limit(self):
            """
            Return the effective iteration limit for this iterator instance.

            :return: Maximum number of frames to yield, or None
            """
            return self._limit

        @property
        def node_id(self):
            """
            Resolve the backend node identifier used for fetching frames.

            :return: Node identifier integer or None
            """
            return self._resolve_node_id()

        def set_node(self, node_id=None):
            """
            Force the iterator to use a specific node identifier for backend fetches.
            """
            if self._started and getattr(self, "_fetch_thread", None) and self._fetch_thread.is_alive():
                raise ValueError("Cannot change node id after iterator has started")
            if node_id is None:
                self._node_id = None
                return
            try:
                self._node_id = int(node_id)
            except Exception as exc:
                raise ValueError("node_id must be convertible to int") from exc

        def set_concurrency(self, worker_index=None, num_workers=None):
            """
            Configure worker splitting so multiple iterators can share the same dataview.
            """
            if self._started and getattr(self, "_fetch_thread", None) and self._fetch_thread.is_alive():
                raise ValueError("set_concurrency must be called before the iterator starts")

            resolved_workers = None
            if num_workers is not None:
                try:
                    resolved_workers = int(num_workers)
                except Exception as exc:
                    raise ValueError("num_workers must be an integer") from exc
            else:
                try:
                    detected = get_node_count()
                except Exception:
                    detected = None
                if isinstance(detected, int) and detected > 1:
                    resolved_workers = detected

            if resolved_workers is None or resolved_workers <= 1:
                self._num_workers = None
                self._worker_index = None
                self._adjust_limit_for_concurrency()
                return

            if worker_index is None:
                candidate = self._node_id
                if candidate is None:
                    try:
                        candidate = get_node_id()
                    except Exception:
                        candidate = 0
                worker_index = candidate

            try:
                resolved_index = int(worker_index)
            except Exception as exc:
                raise ValueError("worker_index must be an integer") from exc

            if resolved_index < 0:
                raise ValueError("worker_index must be non-negative")

            resolved_index = resolved_index % resolved_workers

            self._num_workers = resolved_workers
            self._worker_index = resolved_index
            self._adjust_limit_for_concurrency()

        def _adjust_limit_for_concurrency(self):
            """
            Recalculate the iterator limit after concurrency changes.
            """
            if self._base_limit is None:
                self._limit = None
                return
            if self._num_workers and self._num_workers > 1:
                self._limit = int(ceil(self._base_limit / self._num_workers))
            else:
                self._limit = self._base_limit

        def _resolve_node_id(self):
            """
            Resolve and cache the node identifier used for requests.
            """
            if self._node_id is None:
                try:
                    self._node_id = get_node_id()
                except Exception:
                    self._node_id = None
            return self._node_id

        def __del__(self):
            try:
                self._closed = True
                if hasattr(self, "_stop_event"):
                    self._stop_event.set()
                if getattr(self, "_started", False) and getattr(self, "_fetch_thread", None) and self._fetch_thread.is_alive():
                    self._fetch_thread.join(timeout=1)
            except Exception:
                pass

        def _fetcher_daemon(self):
            """
            Background thread that pulls frames from the backend and queues them for consumption.
            """
            eof = False
            scroll_id = None
            last_scroll_id: Optional[str] = None
            reset_scroll = False
            force_scroll = False
            try:
                while not self._stop_event.is_set():
                    if self._limit is not None and self._produced >= self._limit:
                        # EOF reached: stop fetching
                        self._eof_reached = True
                        break
                    if eof:
                        if self._limit is not None and self._produced < self._limit:
                            # restart iteration while reusing the previous scroll id so the backend rebalances rules
                            if last_scroll_id:
                                scroll_id = last_scroll_id
                                force_scroll = True
                            reset_scroll = True
                            eof = False
                            self._dispatch_counter = 0
                        else:
                            self._eof_reached = True
                            break
                    resp = DataViewManagementBackend.get_next_data_entries(
                        dataview=self._dataview.id,
                        scroll_id=scroll_id,
                        batch_size=self._query_cache_size,
                        reset_scroll=reset_scroll or None,
                        force_scroll_id=True if force_scroll else None,
                        node=self._resolve_node_id(),
                        projection=self._projection,
                    )
                    reset_scroll = False
                    force_scroll = False
                    eof = bool(getattr(resp, "eof", False))
                    current_scroll_id = getattr(resp, "scroll_id", None)
                    if current_scroll_id:
                        last_scroll_id = current_scroll_id
                    scroll_id = current_scroll_id
                    frames = getattr(resp, "frames", []) or []
                    items = []
                    for frame in frames:
                        base_cls, resolved_cls = self._determine_entry_classes(frame)
                        if self._num_workers:
                            include = (self._dispatch_counter % self._num_workers) == self._worker_index
                            self._dispatch_counter += 1
                            if not include:
                                continue
                        else:
                            self._dispatch_counter += 1
                        if self._limit is not None and self._produced >= self._limit:
                            eof = True
                            self._eof_reached = True
                            break
                        item = frame
                        # Convert to desired classes if available
                        try:
                            if hasattr(base_cls, "from_api_object"):
                                item = base_cls.from_api_object(frame)
                            elif callable(base_cls):
                                item = base_cls(frame)
                            if (
                                resolved_cls
                                and issubclass(resolved_cls, DataEntry)
                                and isinstance(item, DataEntry)
                                and item.__class__ is not resolved_cls
                            ):
                                try:
                                    item.__class__ = resolved_cls
                                except TypeError:
                                    self._logger.warning(
                                        "Could not assign frame to class '%s'",
                                        resolved_cls.__name__,
                                    )
                        except Exception as ex:
                            self._logger.exception(
                                "Failed converting frame to %s: %s",
                                getattr(base_cls, "__name__", str(base_cls)),
                                ex,
                            )
                            item = frame
                        items.append(item)
                    if items:
                        items.reverse()
                        while not self._stop_event.is_set():
                            try:
                                self._data_entries_queue.put(items, timeout=0.5)
                                self._produced += len(items)
                                break
                            except queue.Full:
                                continue
            except Exception as e:
                self._error = e
            finally:
                # mark thread as finished
                self._started = False

        def _determine_entry_classes(self, frame):
            """Return a tuple of (base_class, resolved_class)."""

            def _get(obj, key, default=None):
                if isinstance(obj, dict):
                    return obj.get(key, default)
                return getattr(obj, key, default)

            meta = _get(frame, "meta")
            resolved_cls = None
            if isinstance(meta, dict):
                class_path = meta.get(ENTRY_CLASS_KEY)
                resolved_cls = _resolve_class(class_path, DataEntry)
                if resolved_cls and not issubclass(resolved_cls, DataEntry):
                    resolved_cls = None

            base_cls: type = DataEntry
            if resolved_cls and issubclass(resolved_cls, DataEntryImage):
                base_cls = DataEntryImage
            elif resolved_cls:
                base_cls = DataEntry
            elif self._frame_is_image(frame):
                base_cls = DataEntryImage
            return base_cls, resolved_cls

        @staticmethod
        def _frame_is_image(frame) -> bool:
            """Return True when frame sources look like images."""

            def _get(obj, key, default=None):
                if isinstance(obj, dict):
                    return obj.get(key, default)
                return getattr(obj, key, default)

            sources = _get(frame, "sources") or []
            if not isinstance(sources, (list, tuple)):
                return False
            for s in sources:
                if _get(s, "width") is not None or _get(s, "height") is not None:
                    return True
                p = _get(s, "preview")
                if p and _get(p, "uri"):
                    return True
                m = _get(s, "masks")
                if m:
                    return True
            return False

        def _reset_fetch(self):
            """
            Reset internal queues and counters so iteration can restart from the beginning.
            """
            # reinitialize iteration state (restart from 0)
            capacity = max(1, self._query_queue_depth)
            self._data_entries_queue = Queue(maxsize=capacity)
            self._stop_event = threading.Event()
            self._closed = False
            self._error = None
            self._eof_reached = False
            self._fetch_thread = threading.Thread(target=self._fetcher_daemon, name="HDVFetcher", daemon=True)
            self._yielded = 0
            self._produced = 0
            self._dispatch_counter = 0
