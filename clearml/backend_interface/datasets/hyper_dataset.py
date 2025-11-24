from typing import Optional, List, Dict, Any, Sequence

from ...backend_api.services import datasets
from ...backend_api import Session
from ..base import IdObjectBase
from ..util import exact_match_regex
from ..session import SendError
from ...task import Task

# handle import in offline mode
_SaveFramesRequest = datasets.SaveFramesRequest if getattr(datasets, "SaveFramesRequest", None) else object


class _SaveFramesRequestNoValidate(_SaveFramesRequest):
    def validate(self, schema=None):
        pass


class HyperDatasetManagementBackend(IdObjectBase):
    """
    Provide backend-facing helpers for managing HyperDataset collections, versions, and entries.
    """
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
        cls, name: str, dataset_id: str, comment: Optional[str] = None, parent_ids: Optional[List[str]] = None
    ):
        """
        Create a new dataset version or return the existing version with the same name.

        :param name: Human-readable name identifying the dataset version
        :param dataset_id: Identifier of the parent dataset collection
        :param comment: Optional description attached to the dataset version
        :param parent_ids: Optional list of parent version IDs to record lineage

        :return: Newly created version ID or the ID of an existing version with the same name
        """
        current = Task.current_task()
        req = datasets.CreateVersionRequest(
            dataset=dataset_id,
            name=name,
            comment=comment,
            task=(current.id if current else None),
            parent=parent_ids,
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
    def save_data_entries(cls, dataset_id: str, data_entries):
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
        req = _SaveFramesRequestNoValidate(version=dataset_id, frames=frames)
        res = cls._send(cls._get_default_session(), req)
        return res.response

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
    def get_dataset_by_id(cls, dataset_id: str):
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
    def _project_matches(cls, filter_project: str, project_name: Optional[str], recursive: bool) -> bool:
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
    def version_exists(cls, dataset_id: str, version_id: str) -> bool:
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
                    page_size=100,
                )
                res = cls._send(cls._get_default_session(), req, raise_on_errors=False)
                versions = getattr(getattr(res, "response", None), "versions", None) or []
                if any(getattr(v, "id", None) == version_id for v in versions):
                    return True
                if not versions or len(versions) < 100:
                    return False
                page += 1
        except Exception:
            return False

    @classmethod
    def delete_dataset_version(cls, version_id: str, force: bool = False) -> bool:
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
    def delete_dataset(cls, dataset_id: str, delete_all_versions: bool = True, force: bool = False) -> bool:
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
