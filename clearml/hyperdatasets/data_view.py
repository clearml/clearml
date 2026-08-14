import logging
import queue
import threading
from math import ceil
from queue import Queue
from typing import (
    Any,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
    Sequence,
    Iterable,
)

from clearml.backend_api import Session
from clearml.backend_api.services import dataviews as _dataviews
from clearml.backend_interface.datasets.hyper_dataset_data_view import DataViewManagementBackend
from clearml.backend_interface.util import mutually_exclusive
from clearml.config import (
    deferred_config,
    running_remotely,
    get_remote_task_id,
    get_node_id,
    get_node_count,
)
from clearml.storage.manager import StorageManagerDiskSpaceFileSizeStrategy
from clearml.task import Task
from .data_entry import DataEntry, ENTRY_CLASS_KEY, _resolve_class
from .data_entry_image import DataEntryImage
from .management import HyperDatasetManagement

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
                logging.getLogger("DataView").warning(
                    "Could not validate lucene query because 'luqum' is not installed. "
                    "Run 'pip install luqum' to enable query validation"
                )
                cls.lucene_parser_warning_sent = True
            return

        if not lucene_query:
            return
        try:
            lucene_parser.parse(lucene_query)
        except LuceneParseError as e:
            raise type(e)(f"Failed parsing lucene query '{lucene_query}': {e}")

    def __init__(
        self,
        project_id: str = "*",   # ClearML datasets: collection id
        dataset_id: str = "*",   # ClearML datasets: version id
        version_id: str = "*",   # Alias for clarity; kept for symmetry
        source_query: Optional[str] = None,
        frame_query: Optional[str] = None,
        weight: float = 1.0,
        filter_by_roi: Optional[Any] = None,  # Optional[FilterByRoiEnum]
        label_rules: Optional[Any] = None,  # Optional[Sequence[dataviews.FilterLabelRule]]
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
        Session.verify_feature_set("advanced")
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

    @classmethod
    def _from_filter_rule(
        cls,
        rule: Any,
        dataset_id: Optional[str] = None,
        version_id: Optional[str] = None,
    ) -> "HyperDatasetQuery":
        """
        Reconstruct a query from a backend `dataviews.FilterRule` object.

        The rule comes from a stored DataView, so dataset/version existence and
        Lucene syntax are not re-validated (no extra server round-trips).

        :param rule: `dataviews.FilterRule` (or duck-typed object) fetched from the backend
        :param dataset_id: Optional concrete dataset id overriding the rule's (used when
            expanding a wildcard rule against the dataview's version pool)
        :param version_id: Optional concrete version id overriding the rule's
        :return: A `HyperDatasetQuery` instance mirroring the rule
        """
        query = cls.__new__(cls)
        query._project_id = "*"
        query._dataset_id = dataset_id or getattr(rule, "dataset", None) or "*"
        query._version_id = version_id or getattr(rule, "version", None) or "*"
        query._source_query = getattr(rule, "sources_query", None)
        query._frame_query = getattr(rule, "frame_query", None)
        weight = getattr(rule, "weight", None)
        query._weight = float(weight) if weight is not None else 1.0
        query._filter_by_roi = getattr(rule, "filter_by_roi", None)
        query._label_rules = getattr(rule, "label_rules", None)
        return query

    @property
    def dataset_id(self) -> str:
        """
        Return the dataset identifier targeted by this query.

        :return: Dataset ID string or wildcard marker
        """
        return self._dataset_id

    @property
    def project_id(self) -> str:
        """
        Return the dataset collection identifier associated with this query.

        :return: Project ID string or wildcard marker
        """
        return self._project_id

    @property
    def version_id(self) -> str:
        """
        Return the dataset version identifier resolved for this query.

        :return: Version ID string or wildcard marker
        """
        return self._version_id

    @property
    def source_query(self) -> Optional[str]:
        """
        Return the Lucene query applied to frame source metadata.

        :return: Lucene query string or None
        """
        return self._source_query

    @property
    def frame_query(self) -> Optional[str]:
        """
        Return the Lucene query applied to frame-level metadata.

        :return: Lucene query string or None
        """
        return self._frame_query

    @property
    def weight(self) -> float:
        """
        Return the relative sampling weight assigned to this query.

        :return: Sampling weight as a float
        """
        return self._weight

    @property
    def filter_by_roi(self) -> Optional[Any]:  # Optional[FilterByRoiEnum]
        """
        Return the ROI filtering strategy configured for this query.

        :return: ROI filter identifier or None
        """
        return self._filter_by_roi

    @property
    def label_rules(self) -> Optional[Union[Sequence[Any]]]:  # Optional[Sequence[dataviews.FilterLabelRule]]
        """
        Return the label rule definitions used for ROI filtering.

        :return: Sequence of mapping of label rules, or None
        """
        return self._label_rules

    def _validate_dataset_and_version(self):
        """Verify that referenced dataset and version identifiers exist on the backend."""
        if self._dataset_id in (None, "*"):
            return

        version_id = (
            self._version_id
            if self._version_id not in (None, "*")
            else None
        )

        if not HyperDatasetManagement.exists(
            dataset_id=self._dataset_id,
            version_id=version_id,
        ):
            raise ValueError(
                "HyperDataset query references non-existent dataset/version:"
                f" dataset_id={self._dataset_id}"
                f" version_id={self._version_id}"
            )


class DataView:
    _MAX_BATCH_SIZE = 10000
    _DEFAULT_LOCAL_BATCH_SIZE = 500
    _store_dataviews_on_creation = deferred_config("development.store_dataviews_on_creation", True)

    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        iteration_order: str = "sequential",
        iteration_infinite: bool = False,
        iteration_random_seed: Optional[int] = None,
        iteration_limit: Optional[int] = None,
        auto_connect_with_task: bool = True,
        project_name: Optional[str] = None,
    ) -> None:
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
        :param project_name: Optional project name under which the DataView will be persisted
            when stored. If omitted, falls back to the current Task's project. Required on
            servers with project-scoped RBAC for the auto-store to succeed.
        """
        self._iteration_order = iteration_order
        self._iteration_infinite = iteration_infinite
        self._iteration_limit = (
            int(iteration_limit)
            if iteration_limit is not None
            else iteration_limit
        )
        # TODO: connect with task in remote execution
        self._auto_connect_with_task = auto_connect_with_task
        self._iteration_random_seed = iteration_random_seed
        self._name = name
        self._description = description
        self._tags = tags
        self._project_name = project_name
        self._id = None
        # Internal: when False, iteration never auto-stores this DataView on the server
        # (reads go through the inline frames endpoints, which require no stored id)
        self._store_on_iteration = True
        self._filter_rules: List[Any] = []
        self._labels_enumeration: Optional[dict] = None  # Dict[str, int]
        self._mapping_rules: List[Any] = []  # List[dataviews.MappingRule]
        self._queries: List[HyperDatasetQuery] = []
        # Concrete (dataset, version) pairs stored as the backend dataview's version
        # pool; wildcard filter rules select from this pool
        self._version_pool: List[Tuple[str, str]] = []
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
                    task_id = get_remote_task_id()
                    if task_id:
                        task = Task.get_task(task_id=task_id)
                if task:
                    self._connected_task = task
                    data_view_map = task.get_dataviews() or {}
                    # If a name was provided, prefer a matching dataview by name
                    picked_data_view = (
                        (
                            data_view_map.get(self._name)
                            if self._name
                            else next(iter(data_view_map.values()), None)
                        )
                        if isinstance(data_view_map, dict)
                        else None
                    )

                    if picked_data_view:
                        try:
                            self._copy_from_other_dataview(other=picked_data_view)
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
    def name(self, value: Optional[str]):
        """
        Update the human-readable name associated with this DataView.

        :param value: New DataView name string or None
        """
        self._name = value

    @classmethod
    def get(
        cls,
        dataview_id: Optional[str] = None,
        dataview_name: Optional[str] = None,
    ) -> "DataView":
        """
        Get a previously stored DataView from the server.

        The returned DataView is populated with the stored queries and iteration
        parameters, and is not auto-connected to the current Task.

        :param dataview_id: The ID of the DataView.
        :param dataview_name: The name of the DataView. If more than one DataView shares
            the name, the most recently created one is selected (a warning is logged).

        :return: A new DataView object populated from the stored definition

        .. note::
            ``dataview_id`` and ``dataview_name`` are mutually exclusive.
            Exactly one of them must be provided, otherwise a ValueError is raised.
        """
        mutually_exclusive(_exception_cls=ValueError, dataview_id=dataview_id, dataview_name=dataview_name)
        backend_dataview = (
            DataViewManagementBackend.get_by_id(dataview_id)
            if dataview_id
            else DataViewManagementBackend.get_by_name(dataview_name)
        )
        if not backend_dataview:
            raise ValueError(
                'DataView id "{}" was not found'.format(dataview_id)
                if dataview_id
                else 'DataView named "{}" was not found'.format(dataview_name)
            )
        dataview = cls(auto_connect_with_task=False)
        dataview._init_from_backend_object(backend_dataview)
        return dataview

    def _init_from_backend_object(self, backend_dataview: Any) -> None:
        """
        Populate local state from a fetched `dataviews.Dataview` backend object.

        :param backend_dataview: `dataviews.Dataview` object returned by the backend
        """
        self._id = getattr(backend_dataview, "id", None)
        self._name = getattr(backend_dataview, "name", None)
        self._description = getattr(backend_dataview, "description", None)
        self._tags = getattr(backend_dataview, "tags", None)

        iteration = getattr(backend_dataview, "iteration", None)
        if iteration is not None:
            order = getattr(iteration, "order", None)
            if order is not None:
                self._iteration_order = getattr(order, "value", order)
            infinite = getattr(iteration, "infinite", None)
            if infinite is not None:
                self._iteration_infinite = bool(infinite)
            self._iteration_random_seed = getattr(iteration, "random_seed", None)
            limit = getattr(iteration, "limit", None)
            self._iteration_limit = int(limit) if limit is not None else None

        labels_enumeration = getattr(backend_dataview, "labels_enumeration", None)
        self._labels_enumeration = dict(labels_enumeration) if labels_enumeration else None

        mapping = getattr(backend_dataview, "mapping", None)
        self._mapping_rules = list(getattr(mapping, "rules", None) or [])

        version_entries = getattr(backend_dataview, "versions", None) or []
        self._version_pool = [
            (dataset_id, version_id)
            for dataset_id, version_id in (
                (getattr(entry, "dataset", None), getattr(entry, "version", None))
                for entry in version_entries
            )
            if dataset_id and dataset_id != "*" and version_id and version_id != "*"
        ]

        filters = getattr(backend_dataview, "filters", None) or []
        self._filter_rules = list(filters)
        if filters:
            # A rule with a wildcard dataset/version selects from the dataview's version
            # pool — expand such rules against the pool so queries hold concrete pairs
            queries: List[HyperDatasetQuery] = []
            for rule in filters:
                rule_dataset = getattr(rule, "dataset", None) or "*"
                rule_version = getattr(rule, "version", None) or "*"
                matching_pool_pairs = (
                    [
                        (dataset_id, version_id)
                        for dataset_id, version_id in self._version_pool
                        if (rule_dataset == "*" or dataset_id == rule_dataset)
                        and (rule_version == "*" or version_id == rule_version)
                    ]
                    if (rule_dataset == "*" or rule_version == "*")
                    else []
                )
                if matching_pool_pairs:
                    queries.extend(
                        HyperDatasetQuery._from_filter_rule(
                            rule,
                            dataset_id=dataset_id,
                            version_id=version_id,
                        )
                        for dataset_id, version_id in matching_pool_pairs
                    )
                else:
                    queries.append(HyperDatasetQuery._from_filter_rule(rule))
            self._queries = queries
        else:
            # No filter rules stored: the version pool alone defines the selection
            # (`DataviewEntry` also carries `dataset`/`version` attributes)
            self._queries = [
                HyperDatasetQuery._from_filter_rule(entry)
                for entry in version_entries
            ]

        self._count_cache = None
        self._synthetic_epoch_limit = None

    def get_queries(self) -> List[HyperDatasetQuery]:
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

    def _build_filter_rule_from_query(self, query: HyperDatasetQuery) -> Any:  # dataviews.FilterRule
        return DataViewManagementBackend.create_filter_rule(
            dataset=query.dataset_id,
            label_rules=query.label_rules,
            filter_by_roi=query.filter_by_roi,
            frame_query=query.frame_query,
            sources_query=query.source_query,
            version=query.version_id,
            weight=query.weight,
        )

    def _append_queries(self, queries: Sequence[HyperDatasetQuery]) -> None:
        if not queries:
            return

        if any(
            not isinstance(query, HyperDatasetQuery)
            for query in queries
        ):
            raise ValueError("DataView expects HyperDatasetQuery instances")

        self._filter_rules.extend((
            self._build_filter_rule_from_query(query)
            for query in queries
        ))
        self._queries.extend(queries)
        self._count_cache = None
        self._synthetic_epoch_limit = None
        self._resync_task_attachment()
        if self._id:
            # Reuse the stored dataview: update it in place, keeping its version pool
            # consistent with the rules (new queries may reference versions not in it yet)
            result = DataViewManagementBackend.update_filter_rules(
                dataview_id=self._id,
                filter_rules=self._filter_rules,
                versions=self._collect_version_pairs(),
            )
            if not result:
                raise ValueError(f"Failed updating DataView {self._id}")

            self._resync_task_attachment()

    def set_queries(self, queries: Optional[Iterable[HyperDatasetQuery]]) -> None:
        """
        Replace all existing queries with the supplied collection.

        :param queries: Iterable of `HyperDatasetQuery` objects; pass None or an empty iterable to clear
        """
        if not self._mutation_allowed():
            return

        normalized = (
            list(queries)
            if queries is not None
            else []
        )
        self._filter_rules = []
        self._queries = []
        self._version_pool = []
        self._count_cache = None
        self._synthetic_epoch_limit = None
        if not normalized:
            if self._id:
                DataViewManagementBackend.update_filter_rules(
                    dataview_id=self._id,
                    filter_rules=[],
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
        source_query: Optional[str] = None,
        frame_query: Optional[str] = None,
        weight: float = 1.0,
        filter_by_roi: Optional[Any] = None,  # Optional[FilterByRoiEnum]
        label_rules: Optional[Any] = None,  # Optional[Sequence[dataviews.FilterLabelRule]]
    ) -> HyperDatasetQuery:
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

    def set_iteration_parameters(
        self,
        *,
        infinite: Optional[bool] = None,
        limit: Union[Optional[int], object] = _UNSET,  # Union[int, None, _UNSET]
    ):
        """
        Persist iteration settings both locally and on the backend if possible.
        """
        if (infinite is not None) or (limit is not _UNSET):
            if infinite is not None:
                self._iteration_infinite = bool(infinite)
            if limit is not _UNSET:
                self._iteration_limit = (
                    int(limit)
                    if limit is not None
                    else None
                )

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

    def set_labels(self, label_dict: Mapping[str, int]) -> None:
        """
        Set the dataview label enumeration.

        Label enumeration maps label strings to integers, for later use within the
        network. While iterating over the dataview, each ROI whose label appears in
        the enumeration is returned with a matching ``label_num`` value.

        Example: ``{'person': 1, 'pedestrian': 1, 'background': 0}`` maps both
        'person' and 'pedestrian' ROIs to class 1.

        If the dataview already exists on the backend, the stored enumeration is
        updated immediately and the attached task is re-synchronised.

        :param label_dict: Mapping from a label string to its integer representation,
            e.g. ``{'cat': 0, 'dog': 1, 'hound': 1}``. Pass an empty dict to clear
            the enumeration
        """
        if not isinstance(label_dict, dict) or not all(
            isinstance(key, str) and isinstance(value, int) and not isinstance(value, bool)
            for key, value in label_dict.items()
        ):
            raise ValueError("set_labels expects a label string to integer dictionary")

        if not self._mutation_allowed():
            return

        self._labels_enumeration = dict(label_dict) if label_dict else None

        if self._id:
            if not DataViewManagementBackend.update_labels_enumeration(
                dataview_id=self._id,
                labels_enumeration=self._labels_enumeration or {},
            ):
                raise ValueError(f"Failed updating DataView {self._id}")

        self._resync_task_attachment()

    def get_labels(self) -> dict:
        """
        Return the current dataview label enumeration (label string to integer).

        :return: Dictionary of label string to integer, e.g. ``{'cat': 0, 'dog': 1}``.
            Empty when no enumeration was set
        """
        return dict(self._labels_enumeration or {})

    def add_mapping_rule(
        self,
        from_labels: Union[str, Sequence[str]],
        to_label: str,
        dataset_id: Optional[str] = None,
        dataset_name: Optional[str] = None,
        version_id: Optional[str] = None,
        version_name: Optional[str] = None,
        project_name: Optional[str] = None,
    ) -> Optional[Any]:  # Optional[dataviews.MappingRule]
        """
        Add a label mapping rule to the dataview.

        Mapping automatically converts label names to canonical names in the ROIs
        returned for frames while iterating over the dataview. This is used to
        make sure that different naming in different datasets will not produce
        two different classes for the same object.

        Example: If one dataset has ROIs with the label 'pedestrian' and another
        has ROIs with the label 'person', both can be used in a single dataview
        to create a person detector by adding a mapping from 'pedestrian' to 'person'.

        If the dataview already exists on the backend, the stored mapping rules are
        updated immediately and the attached task is re-synchronised.

        .. note::
            Label mapping is performed **after** the frame is matched against
            the dataview's queries. For that reason, the queries must be defined
            according to the dataset's original labels.

        :param from_labels: Label or list of labels to map to ``to_label``. An ROI must
            match *all* of the labels for the mapping to take place
        :param to_label: Label to change ``from_labels`` to
        :param dataset_id: The ID of the dataset to apply the mapping rule to.
            Defaults to '*' (all datasets in the view). Mutually exclusive with ``dataset_name``
        :param dataset_name: The name of the dataset to apply the mapping rule to
        :param version_id: The ID of the dataset version to apply the mapping rule to.
            Defaults to '*' (all versions of the dataset in the view). Mutually
            exclusive with ``version_name``
        :param version_name: The name of the dataset version to apply the mapping rule to.
            Requires ``dataset_id`` or ``dataset_name``
        :param project_name: Optional project filter used when resolving ``dataset_name``
        :return: The created `dataviews.MappingRule` object
        """
        if not to_label or not isinstance(to_label, str):
            raise ValueError("add_mapping_rule expects a non-empty to_label string")
        if not from_labels:
            raise ValueError("add_mapping_rule expects a label or a non-empty list of labels in from_labels")

        if not self._mutation_allowed():
            return None

        dataset_id, version_id = HyperDatasetManagement._resolve_dataset_and_version(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            version_id=version_id,
            version_name=version_name,
            project_name=project_name,
        )

        rule = DataViewManagementBackend.create_mapping_rule(
            from_labels=from_labels,
            to_label=to_label,
            dataset=dataset_id,
            version=version_id,
        )
        self._mapping_rules.append(rule)

        if self._id:
            if not DataViewManagementBackend.update_mapping_rules(
                dataview_id=self._id,
                mapping_rules=self._mapping_rules,
            ):
                raise ValueError(f"Failed updating DataView {self._id}")

        self._resync_task_attachment()
        return rule

    def get_mapping_rules(self) -> List[Any]:  # List[dataviews.MappingRule]
        """
        Return the current label mapping rules attached to this dataview.

        :return: List of `dataviews.MappingRule` objects
        """
        return list(self._mapping_rules)

    def _resolve_project_name(self) -> Optional[str]:
        """Explicit `project_name` wins; otherwise fall back to the current Task's project."""
        if self._project_name:
            return self._project_name
        try:
            task = Task.current_task()
            if task is None:
                tid = get_remote_task_id()
                if tid:
                    task = Task.get_task(task_id=tid)
            if task:
                proj = task.get_project_name()
                if proj:
                    return proj
        except Exception:
            pass
        return None

    def _collect_version_pairs(self) -> List[Tuple[str, str]]:
        """
        Return the ordered, unique, concrete (dataset, version) pairs this DataView spans:
        the stored version pool plus every pair referenced by the current queries.
        Wildcard or incomplete pairs are dropped.

        :return: List of (dataset_id, version_id) tuples
        """
        pairs: List[Tuple[str, str]] = []
        seen = set()
        candidates = list(self._version_pool) + [
            (getattr(query, "dataset_id", None), getattr(query, "version_id", None))
            for query in self._queries
        ]
        for dataset_id, version_id in candidates:
            if (
                not dataset_id
                or dataset_id == "*"
                or not version_id
                or version_id == "*"
                or (dataset_id, version_id) in seen
            ):
                continue
            seen.add((dataset_id, version_id))
            pairs.append((dataset_id, version_id))
        return pairs

    def _build_inline_payload(self) -> Any:
        """
        Build a `frames.Dataview` payload from this DataView's current local state.

        Used by read paths (count, iteration) so they hit the inline `frames.*ForDataview`
        endpoints — no stored DataView id required, no project write permission required.
        """
        version_pairs = self._collect_version_pairs()

        if len(version_pairs) == 0:
            raise ValueError(
                "Cannot fetch DataView contents: no concrete (dataset, version) provided in queries"
            )

        return DataViewManagementBackend.build_inline_dataview(
            versions=[
                _dataviews.DataviewEntry(
                    dataset=dataset_id,
                    version=version_id,
                )
                for dataset_id, version_id in version_pairs
            ],
            filters=(
                self._filter_rules
                or None
            ),
            iteration=_dataviews.Iteration(
                order=self._iteration_order,
                infinite=self._iteration_infinite,
                random_seed=self._iteration_random_seed,
                limit=self._iteration_limit,
            ),
            labels_enumeration=self._labels_enumeration,
            mapping=(
                _dataviews.Mapping(rules=list(self._mapping_rules))
                if self._mapping_rules
                else None
            ),
        )

    def store(self, project_name: Optional[str] = None) -> str:
        """
        Persist this DataView on the server and return its id.

        DataViews are stored automatically the first time they are iterated, unless
        ``sdk.development.store_dataviews_on_creation`` is set to false in clearml.conf —
        in which case this method is the explicit way to persist one.

        :param project_name: Optional project name to attach the DataView to. Overrides
            the value passed at construction. If omitted, the constructor's value
            (or the current Task's project as a fallback) is used.
        :return: The DataView id assigned by the server.
        :raises: The underlying SendError when the server rejects the create call (for
            example, when the user lacks write permission to the resolved project).
        """
        if project_name is not None:
            self._project_name = project_name
        if not self._id:
            self._create_on_server()
        return self._id

    def _ensure_created(self) -> None:
        """
        Auto-store hook called from read paths that need an id (e.g. iteration).

        Honors ``sdk.development.store_dataviews_on_creation``: when disabled, returns
        without contacting the server. Errors during the auto-store are caught and
        logged as a warning so reads can still proceed via the inline frames endpoints.

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
            if not running_remotely():
                return
        if not getattr(self, "_store_on_iteration", True):
            return
        if not bool(self._store_dataviews_on_creation):
            return
        try:
            self._create_on_server()
        except Exception as e:
            # Persisting the DataView is best-effort here — read paths use the inline
            # frames endpoints and don't need an id. Warn and continue.
            logging.getLogger("DataView").warning(
                "Failed to persist DataView on the server (continuing without an id): %s", e
            )
            self._id = None

    def _create_on_server(self) -> None:
        """Build the create payload from local state and persist it. Raises on failure."""
        version_pairs = self._collect_version_pairs()

        if len(version_pairs) == 0:
            raise ValueError(
                "Cannot fetch DataView contents: no concrete (dataset, version) provided in queries"
            )

        versions = [
            {"dataset": dataset_id, "version": version_id}
            for dataset_id, version_id in version_pairs
        ]

        self._id = DataViewManagementBackend.create(
            name=self._name,
            description=self._description,
            tags=self._tags,
            infinite=self._iteration_infinite,
            order=self._iteration_order,
            random_seed=self._iteration_random_seed,
            limit=self._iteration_limit,
            versions=versions,
            project_name=self._resolve_project_name(),
            labels_enumeration=self._labels_enumeration,
            mapping_rules=self._mapping_rules or None,
        )

        if self._filter_rules:
            DataViewManagementBackend.update_filter_rules(
                dataview_id=self._id,
                filter_rules=self._filter_rules,
            )

        self._version_pool = version_pairs
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
                    task_id = get_remote_task_id()
                    if task_id:
                        task = Task.get_task(task_id=task_id)
            if not task:
                task = Task.current_task()

            if not task:
                return

            task.set_dataview(
                self.id
                if (force_remote and self._id)
                else self
            )
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
        queries = self.get_queries()
        if len(queries) <= 1:
            return None

        weights = [
            (
                float(query.weight)
                if query.weight is not None
                else 1.0
            )
            for query in queries
        ]

        if (
            not self._iteration_infinite
            and all(
                query.weight is None
                for query in queries
            )
        ):
            return None

        try:
            payload = self._build_inline_payload()
        except ValueError:
            return None

        total, rule_counts = DataViewManagementBackend.get_count_details(payload)
        if (
            total
            and not self._count_cache
        ):
            self._count_cache = int(total)
        if len(rule_counts) == 0:
            return None

        if len(rule_counts) < len(queries):
            rule_counts.extend([0] * (len(queries) - len(rule_counts)))

        positive_rule_count_weights = [
            weight
            for rule_count, weight in zip(rule_counts, weights)
            if rule_count > 0
        ]
        if len(positive_rule_count_weights) == 0:
            return None

        sum_weights = sum(positive_rule_count_weights)
        normalized_weights = [
            (
                weight / (sum_weights or 1.0)
                if count > 0
                else 0.0
            )
            for count, weight in zip(rule_counts, weights)
        ]

        max_count = max(rule_counts)
        if max_count > 0:
            idx_with_largest_count = next((
                index
                for index, count in enumerate(rule_counts)
                if count == max_count
            ), 0)

            weight_fraction = normalized_weights[idx_with_largest_count]

            return (
                int(ceil(max_count / weight_fraction))
                if weight_fraction > 0
                else None
            )
        else:
            return None

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
        labels_enumeration = getattr(other, "_labels_enumeration", None)
        self._labels_enumeration = dict(labels_enumeration) if labels_enumeration else None
        self._mapping_rules = list(getattr(other, "_mapping_rules", []) or [])
        self._queries = list(getattr(other, "_queries", []))
        self._version_pool = list(getattr(other, "_version_pool", []) or [])
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

        Sends an inline DataView spec — no stored id required.
        """
        if self._count_cache is not None:
            return self._count_cache
        total = DataViewManagementBackend.get_count_total(self._build_inline_payload())
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
                        pool.submit(
                            StorageManagerDiskSpaceFileSizeStrategy.get_local_copy,
                            uri,
                            None,
                            True,
                            None,
                            force_download,
                        )
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
            self._inline_payload = None
            self._query_queue_depth = int(query_queue_depth or 5)
            capacity = max(1, self._query_queue_depth)
            self._data_entries_queue: Queue = Queue(maxsize=capacity)
            self._stop_event = threading.Event()
            self._started = False
            # True once iteration has ever begun on this instance; unlike _started it is
            # never cleared when the fetch thread finishes, so a bare next() can tell a
            # fresh iterator apart from a finished one (and must not restart the latter)
            self._ever_started = False
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
                self._ever_started = True
                self._fetch_thread.start()
            return self

        def __next__(self):
            """
            Fetch the next data-entry object, respecting synthetic epoch limits.
            """
            if not self._ever_started:
                # Calling next() directly on a fresh iterator: start the fetch thread.
                # Never restart a finished iterator here — the fetch thread clears
                # _started when it completes, and restarting would reset the yield
                # counters and iterate forever (opening a new backend scroll per epoch)
                self.__iter__()
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
                if (
                    getattr(self, "_started", False)
                    and getattr(self, "_fetch_thread", None)
                    and self._fetch_thread.is_alive()
                ):
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
                    if self._inline_payload is None:
                        self._inline_payload = self._dataview._build_inline_payload()
                    resp = DataViewManagementBackend.get_next_data_entries(
                        dataview=self._inline_payload,
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
                    # Never downgrade to a bare request: a request without a scroll id
                    # opens a new server-side scroll context on every call
                    scroll_id = current_scroll_id or scroll_id
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
