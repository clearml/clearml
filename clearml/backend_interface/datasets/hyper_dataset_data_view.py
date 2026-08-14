from typing import (
    Any,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from ..base import IdObjectBase
from ..util import exact_match_regex, get_or_create_project, get_single_result, make_message
from ...backend_api.services import dataviews, frames


class DataViewManagementBackend(IdObjectBase):
    """
    Provide backend helpers for creating, updating, and querying HyperDataset DataViews.
    """
    @classmethod
    def create(
        cls,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        infinite: bool = False,
        order: str = "sequential",
        random_seed: Optional[int] = None,
        limit: Optional[int] = None,
        versions: Optional[Iterable[Union[Mapping[str, str], Sequence[str]]]] = None,
        project_name: Optional[str] = None,
        labels_enumeration: Optional[Mapping[str, int]] = None,
    ) -> str:
        """
        Create a DataView on the backend using the structured create request.

        :param name: Optional human-friendly name for the DataView
        :param description: Optional description stored with the DataView
        :param tags: Optional tag list passed to the backend
        :param infinite: Whether iteration loops endlessly when consumed
        :param order: Iteration order to apply when streaming entries
        :param random_seed: Optional seed influencing randomized iteration
        :param limit: Optional upper bound on the number of entries returned
        :param versions: Optional iterable mapping dataset IDs to version IDs
        :param project_name: Optional project name. When set, the project is resolved
            (created on demand) and the DataView is attached to it. Required on servers
            with project-scoped RBAC.
        :param labels_enumeration: Optional label-string to integer enumeration stored
            with the DataView

        :return: Identifier of the created DataView
        """
        session = cls._get_default_session()
        dataview_entries = cls._convert_versions_to_dataview_entries(versions)
        project_id = (
            get_or_create_project(session, project_name)
            if project_name
            else None
        )

        # TODO: check if we need a creation lock
        response = cls._send(
            session=session,
            req=dataviews.CreateRequest(
                name=(
                    name
                    or make_message('Anonymous dataview (%(user)s@%(host)s %(time)s)')
                ),
                description=(
                    description
                    or make_message('Auto-generated on %(time)s by %(user)s@%(host)s')
                ),
                tags=tags,
                filters=[],
                versions=dataview_entries,
                iteration=dataviews.Iteration(
                    order=order,
                    infinite=infinite,
                    random_seed=random_seed,
                    limit=limit,
                ),
                project=project_id,
                **(
                    {"labels_enumeration": dict(labels_enumeration)}
                    if labels_enumeration
                    else {}
                ),
            ),
        )

        return response.response.id

    @classmethod
    def _convert_versions_to_dataview_entries(
        cls,
        versions: Optional[Iterable[Union[Mapping[str, str], Sequence[str]]]],
    ) -> Optional[List[Any]]:  # Optional[List[dataviews.DataviewEntry]]
        """
        Convert `{"dataset": ..., "version": ...}` mappings or (dataset, version) pairs into
        `dataviews.DataviewEntry` objects, dropping wildcard or incomplete entries.

        :param versions: Iterable of version mappings/pairs, or None
        :return: List of `dataviews.DataviewEntry` objects, or None when nothing remains
        """
        def convert_version_to_dataview_entry(
            version: Union[Mapping[str, str], Sequence[str]],
        ) -> Optional[Any]:  # Optional[dataviews.DataviewEntry]
            dataset, dataset_version = (
                (version.get("dataset"), version.get("version"))
                if isinstance(version, dict)
                else (version[0], version[1])
                if isinstance(version, (tuple, list)) and len(version) >= 2
                else (None, None)
            )

            return (
                dataviews.DataviewEntry(
                    dataset=dataset,
                    version=dataset_version,
                )
                if (
                    (dataset and dataset != "*")
                    and (dataset_version and dataset_version != "*")
                )
                else None
            )

        if not versions:
            return None

        dataview_entries = [
            dataview_entry
            for dataview_entry in (
                convert_version_to_dataview_entry(version=version)
                for version in versions
            )
            if dataview_entry is not None
        ]

        return dataview_entries or None

    @classmethod
    def update_filter_rules(
        cls,
        dataview_id: str,
        filter_rules: Sequence[Any],  # Sequence[dataviews.FilterRule]
        versions: Optional[Iterable[Union[Mapping[str, str], Sequence[str]]]] = None,
    ):
        """
        Replace filter rules associated with a DataView.

        :param dataview_id: Identifier of the DataView being updated
        :param filter_rules: Iterable of filter rule objects compatible with the API
        :param versions: Optional version-pool entries (mappings or (dataset, version)
            pairs) to store together with the rules. When None the stored pool is
            left untouched.

        :return: True when the backend confirms a successful update
        """
        dataview_entries = cls._convert_versions_to_dataview_entries(versions)
        response = cls._send(
            session=cls._get_default_session(),
            req=dataviews.UpdateRequest(
                dataview=dataview_id,
                filters=filter_rules,
                **(
                    {"versions": dataview_entries}
                    if dataview_entries
                    else {}
                ),
            ),
        )

        return response.response.updated >= 1

    @classmethod
    def get_by_id(cls, dataview_id: str):
        """
        Fetch a DataView definition using its identifier.

        :param dataview_id: DataView identifier to retrieve

        :return: DataView object from the backend or None when missing
        """
        try:
            response = cls._send(
                session=cls._get_default_session(),
                req=dataviews.GetByIdRequest(dataview=dataview_id),
                raise_on_errors=False,
            )
            return getattr(getattr(response, "response", None), "dataview", None)
        except Exception:
            return None

    @classmethod
    def get_by_name(cls, dataview_name: str):
        """
        Fetch a DataView definition using its name.

        When more than one DataView matches the name, the most recently created one is
        selected (a warning listing the selection is logged).

        :param dataview_name: DataView name to search for (exact match)

        :return: DataView object from the backend or None when missing
        """
        response = cls._send(
            session=cls._get_default_session(),
            req=dataviews.GetAllRequest(
                name=exact_match_regex(dataview_name),
                only_fields=["name", "id", "created"],
            ),
            raise_on_errors=False,
        )
        results = getattr(getattr(response, "response", None), "dataviews", None) or []
        dataview = get_single_result(
            entity="dataview",
            query=dataview_name,
            results=results,
            raise_on_error=False,
            sort_by_date=True,
        )
        if not dataview:
            return None
        return cls.get_by_id(dataview_id=dataview.id)

    @classmethod
    def create_filter_rule(
        cls,
        dataset: str,
        label_rules: Optional[Sequence[Any]] = None,  # Optional[Sequence[dataviews.FilterLabelRule]]
        filter_by_roi: Optional[Any] = None,  # Optional[FilterByRoiEnum]
        frame_query: Optional[str] = None,
        sources_query: Optional[str] = None,
        version: Optional[str] = None,
        weight: Optional[float] = None,
    ) -> Any:  # dataviews.FilterRule
        """
        Build a filter rule structure compatible with DataView update requests.

        :param dataset: Dataset identifier used by the rule
        :param label_rules: Optional label rule configuration
        :param filter_by_roi: Optional ROI filtering parameters
        :param frame_query: Optional query targeting frame metadata
        :param sources_query: Optional query limiting source metadata
        :param version: Optional dataset version identifier
        :param weight: Optional rule weight for sampling decisions

        :return: Dataview filter rule object
        """
        return dataviews.FilterRule(
            dataset=dataset,
            label_rules=label_rules,
            filter_by_roi=filter_by_roi,
            frame_query=frame_query,
            sources_query=sources_query,
            version=version,
            weight=weight,
        )

    @classmethod
    def update_labels_enumeration(
        cls,
        dataview_id: str,
        labels_enumeration: Mapping[str, int],
    ) -> bool:
        """
        Replace the label enumeration associated with a DataView.

        :param dataview_id: Identifier of the DataView being updated
        :param labels_enumeration: Mapping of label strings to integers

        :return: True when the backend confirms a successful update
        """
        response = cls._send(
            session=cls._get_default_session(),
            req=dataviews.UpdateRequest(
                dataview=dataview_id,
                labels_enumeration=dict(labels_enumeration),
            ),
        )

        return response.response.updated >= 1

    @classmethod
    def build_inline_dataview(
        cls,
        versions: Optional[Iterable[Any]] = None,
        filters: Optional[Iterable[Any]] = None,
        iteration: Optional[Any] = None,
        output_rois: Optional[str] = None,
        labels_enumeration: Optional[Mapping[str, int]] = None,
    ) -> Any:
        """
        Build a `frames.Dataview` payload usable with the inline count / next-frame endpoints.

        Accepts SDK-side `dataviews.*` objects (`DataviewEntry`, `FilterRule`, `Iteration`) or
        raw dicts; round-trips them through `frames.Dataview.from_dict()`.
        """
        def convert_value_to_dict(value: Any) -> dict:
            return (
                value.to_dict()
                if hasattr(value, "to_dict")
                else dict(value)
            )

        return frames.Dataview.from_dict({
            **(
                {"versions": [
                    convert_value_to_dict(version)
                    for version in versions
                ]}
                if versions
                else {}
            ),
            **(
                {"filters": [
                    convert_value_to_dict(filter_)
                    for filter_ in filters
                ]}
                if filters
                else {}
            ),
            **(
                {"iteration": convert_value_to_dict(iteration)}
                if iteration is not None
                else {}
            ),
            **(
                {"output_rois": output_rois}
                if output_rois is not None
                else {}
            ),
            **(
                {"labels_enumeration": dict(labels_enumeration)}
                if labels_enumeration
                else {}
            ),
        })

    @classmethod
    def get_next_data_entries(
        cls,
        dataview: Any,  # dataviews.Dataview
        scroll_id: Optional[str] = None,
        batch_size: int = 500,
        reset_scroll: Optional[bool] = None,
        force_scroll_id: Optional[bool] = None,
        flow_control: Optional[Any] = None,  # Optional[frames.FlowControl]
        random_seed: Optional[int] = None,
        node: Optional[int] = None,
        projection: Optional[Sequence[str]] = None,
        remove_none_values: bool = False,
        clean_subfields: bool = False,
    ) -> Optional[Any]:
        """
        Fetch the next batch of entries via the inline `frames.get_next_for_dataview` endpoint.

        :param dataview: Inline `frames.Dataview` payload (build with `build_inline_dataview`).
            Does NOT require a stored DataView id — works for users without project write permission.
        :param scroll_id: Optional server scroll identifier for continuation
        :param batch_size: Maximum number of entries to request per call
        :param reset_scroll: Whether to reset server-side scroll state
        :param force_scroll_id: Optional explicit scroll identifier to reuse
        :param flow_control: Optional flow control configuration for throttling
        :param random_seed: Optional seed to influence randomized retrieval
        :param node: Optional backend node identifier to target
        :param projection: Optional projection definition limiting returned fields
        :param remove_none_values: Whether to strip None values from entries
        :param clean_subfields: Whether to drop nested subfields with empty content

        :return: Backend response object containing frames and continuation metadata
        """
        response = cls._send(
            session=cls._get_default_session(),
            req=frames.GetNextForDataviewRequest(
                dataview=dataview,
                scroll_id=scroll_id,
                batch_size=batch_size,
                reset_scroll=reset_scroll,
                force_scroll_id=force_scroll_id,
                flow_control=flow_control,
                random_seed=random_seed,
                node=node,
                projection=projection,
                remove_none_values=remove_none_values,
                clean_subfields=clean_subfields,
            ),
            raise_on_errors=False,
        )
        return getattr(response, "response", None)

    @classmethod
    def get_count_total(
        cls,
        dataview: Any,  # dataviews.Dataview
    ) -> int:
        """
        Return the total number of frames matching an inline DataView spec.

        :param dataview: Inline `frames.Dataview` payload (build with `build_inline_dataview`).
        :return: Total frame count reported by the backend
        """
        total, _ = cls.get_count_details(dataview)
        return total

    @classmethod
    def get_count_details(
        cls,
        dataview: Any,  # dataviews.Dataview
    ) -> Tuple[int, List[int]]:
        """
        Retrieve overall and per-rule counts for an inline DataView spec.

        :param dataview: Inline `frames.Dataview` payload (build with `build_inline_dataview`).
        :return: Tuple of total frame count and list of per-rule counts. Falls back to (0, []) on error.
        """
        try:
            response = cls._send(
                session=cls._get_default_session(),
                req=frames.GetCountForDataviewRequest(dataview=dataview),
                raise_on_errors=False,
            )
            response_payload = getattr(response, "response", None)
            total = int(getattr(response_payload, "total", 0) or 0)

            def get_rule_count(rule: Any) -> int:
                try:
                    return int(getattr(rule, "count", 0) or 0)
                except Exception:
                    return 0

            rule_counts = [
                get_rule_count(rule)
                for rule in (
                    getattr(response_payload, "rules", [])
                    or []
                )
            ]

            return total, rule_counts
        except Exception:
            return 0, []

    @classmethod
    def update_iteration_parameters(
        cls,
        dataview_id: str,
        *,
        infinite: Optional[bool] = None,
        limit: Optional[int] = None,
        order: Optional[str] = None,  # Optional[dataviews.IterationOrderEnum]
        random_seed: Optional[int] = None,
    ) -> bool:
        """
        Update iteration configuration parameters for a DataView.

        :param dataview_id: DataView identifier to modify
        :param infinite: Optional flag toggling infinite iteration
        :param limit: Optional maximum number of entries per iteration loop
        :param order: Optional iteration order to apply
        :param random_seed: Optional seed affecting randomized iteration

        :return: True when the backend reports that at least one field was updated
        """

        iteration_kwargs = {
            **({"order": order} if order is not None else {}),
            **({"infinite": bool(infinite) if infinite is not None else {}}),
            **({"limit": limit} if limit is not None else {}),
            **({"random_seed": random_seed} if random_seed is not None else {}),
        }

        if not iteration_kwargs:
            return True

        response = cls._send(
            session=cls._get_default_session(),
            req=dataviews.UpdateRequest(
                dataview=dataview_id,
                iteration=dataviews.Iteration(**iteration_kwargs),
            ),
            raise_on_errors=False,
        )

        return bool(
            getattr(
                getattr(response, "response", None),
                "updated",
                0,
            )
        )
