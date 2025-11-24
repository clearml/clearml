from typing import Optional, Sequence, List, Dict, Any, Type, TypeVar

from clearml.backend_api import Session
from clearml.backend_interface.util import get_existing_project
from clearml.backend_interface.datasets.hyper_dataset import HyperDatasetManagementBackend


HD = TypeVar("HD", bound="HyperDatasetManagement")


class HyperDatasetManagement:
    @classmethod
    def get(
        cls: Type[HD],
        dataset_name: Optional[str] = None,
        version_name: Optional[str] = None,
        project_name: Optional[str] = None,
        *,
        dataset_id: Optional[str] = None,
        version_id: Optional[str] = None,
    ) -> HD:
        """
        Return a `HyperDataset` handle bound to an existing dataset/version.

        :param dataset_name: Dataset collection name. Mutually exclusive with `dataset_id`
        :param version_name: Version name. Mutually exclusive with `version_id`
        :param project_name: Optional ClearML project filter when using `dataset_name`
        :param dataset_id: Dataset identifier. Mutually exclusive with `dataset_name`
        :param version_id: Version identifier. Mutually exclusive with `version_name`

        :return: `HyperDataset` instance pointing at the requested dataset/version
        """

        if dataset_name and dataset_id:
            raise ValueError("Provide either dataset_name or dataset_id, not both")
        if version_name and version_id:
            raise ValueError("Provide either version_name or version_id, not both")

        if not dataset_name and not dataset_id:
            raise ValueError("dataset_name or dataset_id must be provided")

        if dataset_id:
            ds = HyperDatasetManagementBackend.get_dataset_by_id(dataset_id)
            if not ds:
                raise ValueError(f"Dataset not found: {dataset_id}")
        else:
            session = Session()
            project_id = get_existing_project(session, project_name) if project_name else None
            ds = HyperDatasetManagementBackend.get_dataset(name=dataset_name, project_id=project_id)
            if not ds:
                raise ValueError(f"Dataset not found: {dataset_name}")

        dataset_id = getattr(ds, "id", None)
        if not dataset_id:
            raise ValueError("Dataset has no identifier")

        if version_id:
            if not HyperDatasetManagementBackend.version_exists(dataset_id=dataset_id, version_id=version_id):
                raise ValueError(f"Version not found: {version_id}")
            resolved_version_id = version_id
        else:
            resolved_version_id = HyperDatasetManagementBackend.get_version(
                dataset_id=dataset_id, version_name=version_name
            )
            if not resolved_version_id:
                raise ValueError(
                    f"{'Version not found: ' + version_name if version_name else 'No versions found'} (dataset={dataset_name or dataset_id})"  # noqa
                )

        target_cls = cls._result_class()
        obj = target_cls.__new__(target_cls)  # type: ignore[misc]
        obj._project_id = getattr(ds, "project", None)
        obj._dataset_id = dataset_id
        obj._version_id = resolved_version_id
        return obj  # type: ignore[return-value]

    @classmethod
    def exists(
        cls,
        dataset_name: Optional[str] = None,
        version_name: Optional[str] = None,
        project_name: Optional[str] = None,
        *,
        dataset_id: Optional[str] = None,
        version_id: Optional[str] = None,
    ) -> bool:
        """
        Check whether a dataset (and optionally a specific version) exists.

        :param dataset_name: Dataset collection name. Mutually exclusive with `dataset_id`
        :param version_name: Dataset version name. Mutually exclusive with `version_id`
        :param project_name: Optional project filter when searching by name
        :param dataset_id: Dataset identifier to query. Mutually exclusive with `dataset_name`
        :param version_id: Version identifier to query. Mutually exclusive with `version_name`

        :return: True when the dataset (and requested version) can be found
        """
        if dataset_name and dataset_id:
            raise ValueError("Provide either dataset_name or dataset_id, not both")
        if version_name and version_id:
            raise ValueError("Provide either version_name or version_id, not both")

        if not dataset_name and not dataset_id:
            raise ValueError("dataset_name or dataset_id must be provided")

        if dataset_id:
            ds = HyperDatasetManagementBackend.get_dataset_by_id(dataset_id)
            if not ds:
                return False
        else:
            session = Session()
            project_id = get_existing_project(session, project_name) if project_name else None
            ds = HyperDatasetManagementBackend.get_dataset(name=dataset_name, project_id=project_id)
            if not ds:
                return False

        dataset_id = getattr(ds, "id", None)
        if not dataset_id:
            return False

        if version_id in (None, "*") and version_name is None:
            return True

        if version_id not in (None, "*"):
            return HyperDatasetManagementBackend.version_exists(dataset_id=dataset_id, version_id=version_id)

        version = HyperDatasetManagementBackend.get_version(dataset_id=dataset_id, version_name=version_name)
        return bool(version)

    @classmethod
    def list(
        cls,
        project_name: Optional[str] = None,
        partial_name: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        ids: Optional[Sequence[str]] = None,
        recursive_project_search: bool = True,
        include_archived: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        List HyperDataset collections matching the provided filters.

        :param project_name: Optional project filter (matches project hierarchy when recursive)
        :param partial_name: Optional regex / partial dataset name filter
        :param tags: Optional list of tags to filter by
        :param ids: Optional list of dataset identifiers
        :param recursive_project_search: Include subprojects when filtering by project_name
        :param include_archived: Include archived datasets when True
        :return: List of dictionaries describing the matching datasets
        """
        return HyperDatasetManagementBackend.list(
            dataset_project=project_name,
            partial_name=partial_name,
            tags=tags,
            ids=ids,
            recursive_project_search=recursive_project_search,
            include_archived=include_archived,
        )

    @classmethod
    def delete(
        cls,
        dataset_name: str,
        version_name: Optional[str] = None,
        project_name: Optional[str] = None,
        *,
        force: bool = False,
    ) -> bool:
        """
        Delete a dataset or a specific dataset version.

        :param dataset_name: Dataset name to delete (required)
        :param version_name: Version name to delete. When omitted, the entire dataset is removed
        :param project_name: Optional project context when resolving by name
        :param force: Force deletion even when there are protections

        :return: True when deletion completes successfully
        """
        session = Session()
        project_id = get_existing_project(session, project_name) if project_name else None

        ds = HyperDatasetManagementBackend.get_dataset(name=dataset_name, project_id=project_id)
        if not ds:
            return False

        if version_name:
            version_id = HyperDatasetManagementBackend.get_version(dataset_id=ds.id, version_name=version_name)
            if not version_id:
                return False
            return HyperDatasetManagementBackend.delete_dataset_version(version_id=version_id, force=force)

        return HyperDatasetManagementBackend.delete_dataset(dataset_id=ds.id, delete_all_versions=True, force=force)

    @classmethod
    def _result_class(cls) -> Type["HyperDatasetManagement"]:
        if cls is HyperDatasetManagement:
            from .core import HyperDataset  # Local import to avoid circular dependency

            return HyperDataset
        return cls
