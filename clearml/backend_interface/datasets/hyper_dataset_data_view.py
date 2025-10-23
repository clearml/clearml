from ..base import IdObjectBase
from ..util import make_message
from ...backend_api.services import dataviews, frames


class DataViewManagementBackend(IdObjectBase):
    """
    Provide backend helpers for creating, updating, and querying HyperDataset DataViews.
    """
    @classmethod
    def create(
        cls,
        name=None,
        description=None,
        tags=None,
        infinite=False,
        order="sequential",
        random_seed=None,
        limit=None,
        versions=None,
    ):
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

        :return: Identifier of the created DataView
        """
        dv_entries = None
        if versions:
            dv_entries = []
            for v in versions:
                if isinstance(v, dict):
                    ds = v.get("dataset")
                    ver = v.get("version")
                elif isinstance(v, (tuple, list)) and len(v) >= 2:
                    ds, ver = v[0], v[1]
                else:
                    continue
                if ds and ver and ds != "*" and ver != "*":
                    dv_entries.append(dataviews.DataviewEntry(dataset=ds, version=ver))
        req = dataviews.CreateRequest(
            name=name or make_message('Anonymous dataview (%(user)s@%(host)s %(time)s)'),
            description=description or make_message('Auto-generated on %(time)s by %(user)s@%(host)s'),
            tags=tags,
            filters=[],
            versions=dv_entries,
            iteration=dataviews.Iteration(
                order=order,
                infinite=infinite,
                random_seed=random_seed,
                limit=limit,
            ),
        )
        # TODO: check if we need a creation lock
        res = cls._send(cls._get_default_session(), req)
        return res.response.id

    @classmethod
    def update_filter_rules(cls, dataview_id, filter_rules):
        """
        Replace filter rules associated with a DataView.

        :param dataview_id: Identifier of the DataView being updated
        :param filter_rules: Iterable of filter rule objects compatible with the API

        :return: True when the backend confirms a successful update
        """
        req = dataviews.UpdateRequest(dataview=dataview_id, filters=filter_rules)
        res = cls._send(cls._get_default_session(), req)
        updated = res.response.updated
        if updated >= 1:
            return True
        return False

    @classmethod
    def get_by_id(cls, dataview_id):
        """
        Fetch a DataView definition using its identifier.

        :param dataview_id: DataView identifier to retrieve

        :return: DataView object from the backend or None when missing
        """
        try:
            req = dataviews.GetByIdRequest(dataview=dataview_id)
            res = cls._send(cls._get_default_session(), req, raise_on_errors=False)
            return getattr(getattr(res, "response", None), "dataview", None)
        except Exception:
            return None

    @classmethod
    def create_filter_rule(
        cls,
        dataset,
        label_rules=None,
        filter_by_roi=None,
        frame_query=None,
        sources_query=None,
        version=None,
        weight=None,
    ):
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
    def get_next_data_entries(
        cls,
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
    ):
        """
        Fetch the next batch of entries from a DataView iterator.

        :param dataview: DataView identifier to iterate
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
        # Use frames.get_next_for_dataview_id which takes a dataview id and returns frames
        req = frames.GetNextForDataviewIdRequest(
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
        )
        res = cls._send(cls._get_default_session(), req, raise_on_errors=False)
        return getattr(res, "response", None)

    @classmethod
    def get_count_total_for_id(cls, dataview_id: str) -> int:
        """
        Return the total number of frames matching a DataView identifier.

        :param dataview_id: DataView identifier to query

        :return: Total frame count reported by the backend
        """
        total, _ = cls.get_count_details_for_id(dataview_id)
        return total

    @classmethod
    def get_count_details_for_id(cls, dataview_id: str):
        """
        Retrieve overall and per-rule counts for a DataView.

        :param dataview_id: DataView identifier to query

        :return: Tuple of total frame count and list of per-rule counts
        """
        try:
            req = frames.GetCountForDataviewIdRequest(dataview=dataview_id)
            res = cls._send(cls._get_default_session(), req, raise_on_errors=False)
            response = getattr(res, "response", None)
            total = int(getattr(response, "total", 0) or 0)
            rules = []
            for rule in getattr(response, "rules", []) or []:
                try:
                    rules.append(int(getattr(rule, "count", 0) or 0))
                except Exception:
                    rules.append(0)
            return total, rules
        except Exception:
            return 0, []

    @classmethod
    def update_iteration_parameters(
        cls,
        dataview_id: str,
        *,
        infinite=None,
        limit=None,
        order=None,
        random_seed=None,
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

        iteration_kwargs = {}
        if infinite is not None:
            iteration_kwargs["infinite"] = bool(infinite)
        if limit is not None:
            iteration_kwargs["limit"] = limit
        if order is not None:
            iteration_kwargs["order"] = order
        if random_seed is not None:
            iteration_kwargs["random_seed"] = random_seed

        if not iteration_kwargs:
            return True

        req = dataviews.UpdateRequest(
            dataview=dataview_id,
            iteration=dataviews.Iteration(**iteration_kwargs),
        )
        res = cls._send(cls._get_default_session(), req, raise_on_errors=False)
        return bool(getattr(getattr(res, "response", None), "updated", 0))
