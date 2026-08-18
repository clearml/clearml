from typing import Optional, List, Dict, Any, Sequence, Union

from ...backend_api.services import datasets
from ...backend_api import Session
from ..base import IdObjectBase
from ..util import exact_match_regex
from ..session import SendError
from ...task import Task
from .save_frames_request_no_validate_wrapped import _get_save_frames_request_no_validate


class HyperDatasetManagementBackend(IdObjectBase):
    """
    Provide backend-facing helpers for managing HyperDataset collections, versions, and entries.
    """
    GET_VERSIONS_REQUEST_PAGE_SIZE = 100

    @classmethod
    def create_dataset(
        cls,
        name: str,
        comment: Optional[str] = None,
        tags: Optional[str] = None,
        project=None,
        field_mappings: Optional[Dict[str, Any]] = None,
    ):
        """
        Create or fetch a HyperDataset collection under the specified project.

        :param name: Display name of the dataset collection to create
        :param comment: Optional textual description stored with the dataset
        :param tags: Optional comma-separated tags associated with the dataset
        :param project: Target project identifier or name to contain the dataset
        :param field_mappings: Optional mapping of field names to schema definitions (server >= 3.25)

        :return: Dataset ID for the created or existing collection
        """
        session = cls._get_default_session()
        if field_mappings is not None:
            if not Session.check_min_server_version("3.25"):
                raise ValueError("Minimum ClearML Server version 3.25 is required when specifying field_mappings")
            payload = {"name": name}
            if comment is not None:
                payload["comment"] = comment
            if tags is not None:
                payload["tags"] = tags
            if project is not None:
                payload["project"] = project
            payload["field_mappings"] = field_mappings
            res = session.send_request("datasets", "create", json=payload)
            if res.status_code != 200:
                try:
                    reason = str(res.json()["meta"]["result_msg"])
                except Exception:
                    reason = "unkown reason"
                raise SendError(res, "Failed creating dataset: " + reason)
            return res.json()["data"]["id"]

        try:
            req = datasets.CreateRequest(name=name, comment=comment, project=project, tags=tags)
            res = cls._send(session, req)
            return res.response.id
        except SendError:
            # If already exists, fetch existing dataset ID by exact name (and project when available)
            ga = datasets.GetAllRequest(
                name=exact_match_regex(name),
                project=project,
                only_fields=["id"],
            )
            res = cls._send(session, ga, raise_on_errors=False)
            if res and getattr(res.response, "datasets", None):
                return res.response.datasets[0].id
            raise

    @classmethod
    def create_version(
        cls,
        name: str,
        dataset_id: str,
        comment: Optional[str] = None,
        parent_id: Optional[str] = None,
        **kwargs,
    ):
        """
        Create a new dataset version or return the existing version with the same name.

        :param name: Human-readable name identifying the dataset version
        :param dataset_id: Identifier of the parent dataset collection
        :param comment: Optional description attached to the dataset version
        :param parent_id: Optional list of parent version IDs to record lineage
        :param parent_ids: (Deprecated) Optional list of parent dataset version IDs to link.
            Only one parent ID by hyperdataset is supported. Use `parent_id` instead.

        :return: Newly created version ID or the ID of an existing version with the same name
        """
        current = Task.current_task()
        req = datasets.CreateVersionRequest(
            dataset=dataset_id,
            name=name,
            comment=comment,
            task=(current.id if current else None),
            parent=HyperDatasetManagementBackend._define_parent_id(
                parent_id=parent_id,
                parent_ids=kwargs.get("parent_ids", None),
            ),
        )
        session = cls._get_default_session()
        try:
            res = cls._send(session, req)
            return res.response.id
        except SendError:
            # Version name already exists? Look it up and return existing version id
            ga = datasets.GetVersionsRequest(
                dataset=dataset_id,
                only_fields=["id", "name"],
                only_published=False,
                page_size=100,
            )
            res = cls._send(session, ga, raise_on_errors=False)
            versions = getattr(res.response, "versions", []) if res else []
            for v in versions or []:
                if getattr(v, "name", None) == name:
                    return v.id
            raise

    @classmethod
    def save_data_entries(
        cls,
        dataset_id: str,
        data_entries,
    ):
        """
        Upload and register data entries into the specified dataset version.

        :param dataset_id: Identifier of the dataset version receiving the entries
        :param data_entries: Iterable of API frame objects or `DataEntry` instances to persist

        :return: Backend response object produced by the frames save request
        """
        frames = []
        for de in data_entries:
            if hasattr(de, "to_api_object"):
                frames.append(de.to_api_object())
            else:
                frames.append(de)
        req = _get_save_frames_request_no_validate()(version=dataset_id, frames=frames)
        res = cls._send(cls._get_default_session(), req)
        return res.response

    @classmethod
    def delete_data_entries(
        cls,
        version_id: str,
        entry_ids: Sequence[str],
        force: bool = False,
    ) -> int:
        """
        Delete data entries (frames) from a writable dataset version by their identifiers.

        :param version_id: Dataset version identifier the entries are deleted from
        :param entry_ids: Identifiers of the entries to delete
        :param force: Ignore ongoing annotation tasks using this version as input

        :return: Number of entries the backend reports as deleted
        """
        response = cls._send(
            session=cls._get_default_session(),
            req=datasets.DeleteFramesRequest(
                version=version_id,
                frames=list(entry_ids),
                force=force,
            ),
        )
        return int(getattr(response.response, "deleted", 0) or 0)

    @classmethod
    def commit_version(
        cls,
        version_id: str,
        *,
        publish: bool = False,
        force: bool = False,
        calculate_stats: Optional[bool] = True,
        override_stats: Optional[Any] = None,
        publishing_task: Optional[str] = None,
    ):
        """
        Commit a draft dataset version and refresh its statistics.

        :param version_id: Draft version identifier
        :param publish: Optional flag to publish the version after commit
        :param force: Force publish even with active annotation tasks
        :param calculate_stats: Whether to calculate statistics during commit
        :param override_stats: Optional statistics payload to persist instead of recalculating
        :param publishing_task: Annotation task identifier issuing the commit
        :return: Backend response payload
        """
        req = datasets.CommitVersionRequest(
            version=version_id,
            publish=publish or None,
            force=force or None,
            calculate_stats=calculate_stats,
            override_stats=override_stats,
            publishing_task=publishing_task,
        )
        res = cls._send(cls._get_default_session(), req, raise_on_errors=False)
        return getattr(res, "response", None)

    @classmethod
    def publish_version(
        cls,
        version_id: str,
        force: bool = False,
        publishing_task: Optional[str] = None,
    ):
        """
        Publish a dataset version and refresh its statistics.

        :param version_id: Draft version identifier
        :param force: Force publish even with active annotation tasks
        :param publishing_task: Annotation task identifier issuing the commit
        :return: Backend response payload
        """
        return cls._send(
            session=cls._get_default_session(),
            req=datasets.PublishVersionRequest(
                version=version_id,
                force=force or None,
                publishing_task=publishing_task,
            ),
            raise_on_errors=False,
        )

    @classmethod
    def publish_and_create_child_version(
        cls,
        dataset_id: str,
        version_id: str,
        child_name: Optional[str] = None,
        publish_name: Optional[str] = None,
        publish_comment: Optional[str] = None,
        child_comment: Optional[str] = None,
    ) -> Optional[str]:
        """
        Publish and create a child version of a hyperdataset.

        :param dataset_id: Identifier of the dataset collection owning the version
        :param version_id: Draft/committed version identifier to publish
        :param child_name: Name for the new child version. When omitted,
            the backend reuses the parent version name
        :param publish_name: New name for the published version. When omitted the
            published version keeps the backend default
        :param publish_comment: Optional comment for the published version
        :param child_comment: Optional comment for the new child version

        :return: Identifier of the newly created draft child version, or None
        """
        response = cls._send(
            session=cls._get_default_session(),
            req=datasets.PublishAndCreateChildVersionRequest(
                dataset=dataset_id,
                version=version_id,
                child_name=child_name,
                child_comment=child_comment,
                publish_name=publish_name,
                publish_comment=publish_comment,
            ),
        )

        return getattr(
            getattr(response, "response", None),
            "id",
            None,
        )

    @classmethod
    def get_dataset(
        cls,
        name: str,
        project_id: Optional[str] = None,
    ):
        """
        Fetch a dataset collection by exact name and optional project constraint.

        :param name: Dataset collection name to match exactly
        :param project_id: Optional project identifier limiting the lookup scope

        :return: Dataset object with metadata fields or None when no match is found
        """
        req = datasets.GetAllRequest(
            name=exact_match_regex(name),
            project=project_id,
            size=1
        )
        res = cls._send(cls._get_default_session(), req, raise_on_errors=False)
        ds_list = getattr(getattr(res, "response", None), "datasets", None)
        return ds_list[0] if ds_list else None

    @classmethod
    def get_version(
        cls,
        dataset_id: str,
        version_name: Optional[str] = None,
        only_published: bool = False,
    ) -> Optional[str]:
        """
        Resolve a dataset version identifier according to the provided criteria.

        :param dataset_id: Dataset collection identifier
        :param version_name: Optional version name to resolve explicitly
        :param only_published: Whether to restrict lookups to published versions

        :return: Version ID matching the request or None if no suitable version is found
        """
        if version_name:
            req = datasets.GetVersionsRequest(
                dataset=dataset_id, only_fields=["id", "name"], only_published=only_published
            )
            res = cls._send(cls._get_default_session(), req, raise_on_errors=False)
            versions = getattr(getattr(res, "response", None), "versions", None) or []
            for v in versions:
                if getattr(v, "name", None) == version_name:
                    return getattr(v, "id", None)
            return None

        # Try to get head version via dataset get_all by id
        req = datasets.GetAllRequest(id=[dataset_id], only_fields=["head_version"], resolve_head=True)
        res = cls._send(cls._get_default_session(), req, raise_on_errors=False)
        ds_list = getattr(getattr(res, "response", None), "datasets", None) or []
        if ds_list and getattr(ds_list[0], "head_version", None):
            return getattr(ds_list[0].head_version, "id", None)

        # Fallback to first version
        req = datasets.GetVersionsRequest(dataset=dataset_id, only_fields=["id"], only_published=only_published)
        res = cls._send(cls._get_default_session(), req, raise_on_errors=False)
        versions = getattr(getattr(res, "response", None), "versions", None) or []
        return getattr(versions[0], "id", None) if versions else None

    @classmethod
    def get_dataset_by_id(
        cls,
        dataset_id: str,
    ):
        """
        Retrieve a dataset collection object using its identifier.

        :param dataset_id: Unique dataset collection identifier

        :return: Dataset object retrieved from the backend or None if missing
        """
        try:
            req = datasets.GetByIdRequest(dataset=dataset_id)
            res = cls._send(cls._get_default_session(), req, raise_on_errors=False)
            return getattr(getattr(res, "response", None), "dataset", None)
        except Exception:
            return None

    @classmethod
    def list(
        cls,
        dataset_project: Optional[str] = None,
        partial_name: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        ids: Optional[Sequence[str]] = None,
        recursive_project_search: bool = True,
        include_archived: bool = True,
        page_size: int = 200,
    ) -> List[Dict[str, Any]]:
        """
        List HyperDataset collections that match the provided filters.

        :param dataset_project: Optional ClearML project name used to filter datasets
        :param partial_name: Optional regular expression / partial match for dataset names
        :param tags: Optional list of tags (supports exclusion with '-' prefix)
        :param ids: Optional list of dataset identifiers to fetch explicitly
        :param recursive_project_search: When True, include datasets from sub-projects
        :param include_archived: When False, exclude archived datasets from the results
        :param page_size: Page size used for backend pagination
        :return: List of dictionaries describing the datasets that match the query
        """
        system_tags_filter: Optional[List[str]] = None
        if not include_archived:
            system_tags_filter = ["-archived"]

        tag_filter = list(tags) if tags is not None else None
        id_filter = list(ids) if ids is not None else None
        normalized_project_filter = dataset_project.rstrip("/") if dataset_project else None

        results: List[Dict[str, Any]] = []
        page = 0
        while True:
            request = datasets.GetAllRequest(
                id=id_filter,
                name=partial_name,
                tags=tag_filter,
                system_tags=system_tags_filter,
                page=page if page_size else None,
                page_size=page_size or None,
                only_fields=[
                    "id",
                    "name",
                    "project",
                    "tags",
                    "created",
                    "comment",
                    "version_count",
                    "paradigm",
                    "system_tags",
                ],
            )
            response = cls._send(cls._get_default_session(), request, raise_on_errors=False)
            datasets_page = getattr(getattr(response, "response", None), "datasets", None) or []
            if not datasets_page:
                break

            project_ids = {
                getattr(record, "project", None)
                for record in datasets_page
                if getattr(record, "project", None)
            }
            project_lookup = Task._get_project_names(list(project_ids)) if project_ids else {}

            for record in datasets_page:
                raw_project = project_lookup.get(getattr(record, "project", None))
                if normalized_project_filter:
                    if not cls._project_matches(
                        normalized_project_filter,
                        raw_project,
                        recursive_project_search,
                    ):
                        continue

                display_project = cls._strip_hidden_project(raw_project)
                created_value = getattr(record, "created", None)
                if hasattr(created_value, "isoformat"):
                    created_value = created_value.isoformat()
                paradigm = getattr(record, "paradigm", None)
                if hasattr(paradigm, "value"):
                    paradigm = paradigm.value

                results.append(
                    {
                        "id": getattr(record, "id", None),
                        "name": getattr(record, "name", None),
                        "project": display_project,
                        "tags": list(getattr(record, "tags", []) or []),
                        "system_tags": list(getattr(record, "system_tags", []) or []),
                        "created": created_value,
                        "comment": getattr(record, "comment", None),
                        "version_count": getattr(record, "version_count", None),
                        "paradigm": paradigm,
                    }
                )

            if len(datasets_page) < (page_size or len(datasets_page)) or id_filter:
                break
            page += 1

        return results

    @staticmethod
    def _strip_hidden_project(project_name: Optional[str]) -> Optional[str]:
        if not project_name:
            return project_name
        for marker in ("/.hyperdatasets/", "/.datasets/"):
            if marker in project_name:
                return project_name.split(marker, 1)[0]
        for marker in ("/.hyperdatasets", "/.datasets"):
            if project_name.endswith(marker):
                return project_name[: -len(marker)]
        return project_name

    @classmethod
    def _project_matches(
        cls,
        filter_project: str,
        project_name: Optional[str],
        recursive: bool,
    ) -> bool:
        filter_norm = filter_project.rstrip("/")
        if not filter_norm:
            return True

        if not project_name:
            return False

        candidate = project_name.rstrip("/")
        hidden_suffixes = ("/.hyperdatasets", "/.datasets")

        def _matches_single(name: str) -> bool:
            if recursive:
                return name == filter_norm or name.startswith(filter_norm + "/")
            if name == filter_norm:
                return True
            return any(name.startswith(filter_norm + suffix) for suffix in hidden_suffixes)

        if _matches_single(candidate):
            return True

        stripped = cls._strip_hidden_project(candidate)
        if stripped and _matches_single(stripped.rstrip("/")):
            return True

        return False

    @classmethod
    def get_version_by_id(
        cls,
        dataset_id: str,
        version_id: str,
        only_fields: Optional[List[str]] = None,
    ) -> Optional[Dict]:
        response = cls._send(
            session=cls._get_default_session(),
            req=datasets.GetVersionsRequest(
                dataset=dataset_id,
                versions=[version_id],
                only_published=False,  # Fails to retrieve committed/drafted versions without this option
                **({"only_fields": only_fields} if only_fields is not None else {}),
            ),
            raise_on_errors=False,
        )
        versions = getattr(
            getattr(response, "response", None),
            "versions",
            None,
        ) or []

        if not isinstance(versions, list) or len(versions) == 0:
            return None
        elif len(versions) > 1:
            raise ValueError(f"More than one HyperDataset version found: {versions}")
        else:
            return versions[0]

    @classmethod
    def get_version_metadata(
        cls,
        dataset_id: str,
        version_id: str,
    ) -> Dict[str, Any]:
        """
        Fetch the user-defined metadata stored on a dataset version.

        :param dataset_id: Parent dataset collection identifier
        :param version_id: Dataset version identifier

        :return: Metadata dictionary; empty when none was set
        :raises ValueError: When the version cannot be found
        """
        version = cls.get_version_by_id(
            dataset_id=dataset_id,
            version_id=version_id,
            only_fields=["id", "metadata"],
        )
        if version is None:
            raise ValueError(f"Version not found: {version_id} (dataset={dataset_id})")
        metadata = getattr(version, "metadata", None)
        return dict(metadata) if metadata else {}

    @classmethod
    def set_version_metadata(
        cls,
        dataset_id: str,
        version_id: str,
        metadata: Dict[str, Any],
    ) -> bool:
        """
        Replace the user-defined metadata stored on a dataset version.

        :param dataset_id: Parent dataset collection identifier
        :param version_id: Dataset version identifier
        :param metadata: Metadata dictionary to store (nested dictionaries are supported).
            Keys must not include '$' and '.'

        :return: True when the backend confirms the update (locked/published versions
            cannot change their metadata)
        :raises ValueError: When the version cannot be found
        """
        version = cls.get_version_by_id(
            dataset_id=dataset_id,
            version_id=version_id,
            only_fields=["id", "status"],
        )
        if version is None:
            raise ValueError(f"Version not found: {version_id} (dataset={dataset_id})")
        # Only writable versions can change their metadata (matches the classic SDK,
        # which considered draft and committed versions writable)
        status = str(getattr(version, "status", None) or "")
        if status not in (
            str(datasets.VersionStatusEnum.draft),
            str(datasets.VersionStatusEnum.committed),
        ):
            return False

        response = cls._send(
            session=cls._get_default_session(),
            req=datasets.UpdateVersionRequest(
                version=version_id,
                metadata=metadata,
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

    @classmethod
    def version_exists(
        cls,
        dataset_id: str,
        version_id: str,
    ) -> bool:
        """
        Check whether a dataset version exists under the specified dataset collection.

        :param dataset_id: Parent dataset collection identifier
        :param version_id: Dataset version identifier to verify

        :return: True when the version exists, otherwise False
        """
        try:
            page = 0
            while True:
                req = datasets.GetVersionsRequest(
                    dataset=dataset_id,
                    only_fields=["id"],
                    only_published=False,
                    page=page,
                    page_size=cls.GET_VERSIONS_REQUEST_PAGE_SIZE,
                )
                res = cls._send(cls._get_default_session(), req, raise_on_errors=False)
                versions = getattr(getattr(res, "response", None), "versions", None) or []
                if any(getattr(v, "id", None) == version_id for v in versions):
                    return True
                if not versions or len(versions) < cls.GET_VERSIONS_REQUEST_PAGE_SIZE:
                    return False
                page += 1
        except Exception:
            return False

    @staticmethod
    def _define_parent_id(
        parent_id: Optional[str] = None,
        parent_ids: Optional[Union[str, List[str]]] = None,
    ) -> Optional[str]:
        """
        Allows the deprecation of the `parent_ids` argument by duplicating the attribute but only using
        `parent_id` internally and encouraging the usage of `parent_id` against `parent_ids`.
        We only ever planned to support one parent ID per version, `parent_ids` was a typo in that context.

        :param parent_id: Optional parent dataset version IDs to link
        :param parent_ids: (Deprecated) Optional list of parent dataset version IDs to link.
            Only one parent ID by hyperdataset is supported. Use `parent_id` instead.
        """
        return (
            parent_id
            or (
                parent_ids[0]
                if isinstance(parent_ids, list) and len(parent_ids) > 0
                else parent_ids
                if isinstance(parent_ids, str)
                else None
            )
        )

    @classmethod
    def update_dataset_tags(
        cls,
        dataset_id: str,
        tags: List[str],
    ) -> bool:
        """
        Overwrites the hyperdataset-level tags.

        :param dataset_id: Identifier of the dataset collection on which the tags will be updated.
        :param tags: The list of tags to over-write tags with.

        :return: True if the dataset tags were updated successfully, otherwise False.
        """
        response = cls._send(
            session=cls._get_default_session(),
            req=datasets.UpdateRequest(
                dataset=dataset_id,
                tags=tags,
            ),
            raise_on_errors=False,
        )

        return bool(
            getattr(
                getattr(response, "response", None),
                "updated",
                None,
            )
        )

    @classmethod
    def delete_dataset_version(
        cls,
        version_id: str,
        force: bool = False,
    ) -> bool:
        """
        Delete a dataset version from the backend service.

        :param version_id: Identifier of the dataset version to remove
        :param force: Whether to bypass safety checks and force deletion

        :return: True when the backend confirms the version deletion
        """
        req = datasets.DeleteVersionRequest(version=version_id, force=force or None)
        res = cls._send(cls._get_default_session(), req, raise_on_errors=False)
        return bool(getattr(getattr(res, "response", None), "deleted", False))

    @classmethod
    def delete_dataset(
        cls,
        dataset_id: str,
        delete_all_versions: bool = True,
        force: bool = False,
    ) -> bool:
        """
        Remove an entire dataset collection, optionally deleting all contained versions.

        :param dataset_id: Identifier of the dataset collection to delete
        :param delete_all_versions: Whether to delete every version within the collection
        :param force: Whether to bypass safety checks and force deletion

        :return: True when the backend reports the dataset as deleted
        """
        req = datasets.DeleteRequest(
            dataset=dataset_id,
            delete_all_versions=delete_all_versions or None,
            force=force or None,
        )
        res = cls._send(cls._get_default_session(), req, raise_on_errors=False)
        return bool(getattr(getattr(res, "response", None), "deleted", False))
