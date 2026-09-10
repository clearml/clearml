from typing import (
    Optional,
    Sequence,
    List,
    Dict,
    Any,
    Tuple,
    Type,
    TypeVar,
)

from clearml.backend_api import Session
from clearml.backend_interface.util import get_existing_project, mutually_exclusive
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
        Return a ``HyperDataset`` handle bound to an existing dataset/version.

        :param dataset_name: Dataset collection name. Mutually exclusive with ``dataset_id``.
        :param version_name: Version name. Mutually exclusive with ``version_id``.
        :param project_name: ClearML project filter used when resolving by ``dataset_name``.
        :param dataset_id: Dataset identifier. Mutually exclusive with ``dataset_name``.
        :param version_id: Version identifier. Mutually exclusive with ``version_name``.

        :return: A ``HyperDataset`` instance pointing at the requested dataset/version.
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
                error_message = (
                    f"Version not found: {version_name}"
                    if version_name
                    else "No versions found"
                )
                raise ValueError(f"{error_message} (dataset={dataset_name or dataset_id})")

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

        :param dataset_name: Dataset collection name. Mutually exclusive with ``dataset_id``.
        :param version_name: Dataset version name. Mutually exclusive with ``version_id``.
        :param project_name: Project filter used when resolving by ``dataset_name``.
        :param dataset_id: Dataset identifier to query. Mutually exclusive with ``dataset_name``.
        :param version_id: Version identifier to query. Mutually exclusive with ``version_name``.

        :return: ``True`` if the dataset (and requested version) can be found.
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
    def _resolve_dataset_and_version(
        cls,
        dataset_id: Optional[str] = None,
        dataset_name: Optional[str] = None,
        version_id: Optional[str] = None,
        version_name: Optional[str] = None,
        project_name: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Resolve any mix of dataset/version selectors into a concrete, validated
        ``(dataset_id, version_id)`` pair, where an omitted selector resolves to
        the ``'*'`` wildcard.

        Selector rules:
        - ``dataset_id``/``dataset_name`` are mutually exclusive, as are
          ``version_id``/``version_name``
        - ``version_name`` requires a dataset selector (names are only unique per dataset)
        - names are resolved through ``get()``; explicit IDs are verified with ``exists()``
        - no selectors at all is valid and resolves to ``('*', '*')``

        :param dataset_id: Dataset identifier, or '*' / None for any dataset
        :param dataset_name: Dataset name to resolve into an ID
        :param version_id: Version identifier, or '*' / None for any version
        :param version_name: Version name to resolve into an ID
        :param project_name: Optional project filter used when resolving ``dataset_name``
        :return: Tuple of (dataset ID or '*', version ID or '*')
        :raises ValueError: On conflicting selectors or when the referenced
            dataset/version does not exist
        """
        mutually_exclusive(
            _exception_cls=ValueError, _require_at_least_one=False, dataset_id=dataset_id, dataset_name=dataset_name
        )
        mutually_exclusive(
            _exception_cls=ValueError, _require_at_least_one=False, version_id=version_id, version_name=version_name
        )
        if version_name and not (dataset_name or dataset_id):
            raise ValueError("version_name requires dataset_id or dataset_name to be provided")

        # Name-based selectors: a single get() resolves and validates everything
        if dataset_name or version_name:
            resolved = cls.get(
                dataset_name=dataset_name,
                version_name=version_name,
                project_name=project_name,
                dataset_id=None if dataset_name else dataset_id,
            )
            # get() always resolves a concrete version; only adopt it when one was requested
            resolved_version_id = resolved.version_id if version_name else (version_id or "*")
            return resolved.dataset_id, resolved_version_id

        # Id-based selectors: verify existence of concrete ids, pass wildcards through
        if dataset_id and dataset_id != "*":
            if not cls.exists(
                dataset_id=dataset_id,
                version_id=version_id if version_id not in (None, "*") else None,
            ):
                raise ValueError(
                    f"Dataset/version not found: dataset_id={dataset_id} version_id={version_id}"
                )
        return dataset_id or "*", version_id or "*"

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

        :param project_name: Project filter (matches the project hierarchy when
            ``recursive_project_search`` is ``True``).
        :param partial_name: Regex / partial dataset name filter.
        :param tags: List of tags to filter by.
        :param ids: List of dataset identifiers to filter by.
        :param recursive_project_search: If ``True``, include subprojects when filtering by
            ``project_name`` (default ``True``).
        :param include_archived: If ``True``, include archived datasets (default ``True``).

        :return: A list of dictionaries, each describing a matching dataset.
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

        :param dataset_name: Dataset name to delete (required).
        :param version_name: Version name to delete. When omitted, the entire dataset is removed.
        :param project_name: Project filter used when resolving by ``dataset_name``.
        :param force: If ``True``, force deletion even when there are protections (default ``False``).

        :return: ``True`` if deletion completes successfully.
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

    def commit_version(
        self,
        *,
        publish: bool = False,
        force: bool = False,
        calculate_stats: Optional[bool] = True,
        override_stats: Optional[Any] = None,
        publishing_task: Optional[str] = None,
    ):
        """
        Commit the bound HyperDataset version to refresh backend statistics.

        :param publish: If ``True``, publish the version after committing (default ``False``).
        :param force: If ``True``, force publish even when annotation tasks reference the version
            (default ``False``).
        :param calculate_stats: Whether to calculate statistics during the commit (default ``True``).
        :param override_stats: Statistics payload to persist instead of recalculating them.
        :param publishing_task: Annotation task identifier issuing the commit.

        :return: The backend's response payload.
        """
        if not getattr(self, "_version_id", None):
            raise ValueError("HyperDataset instance is not bound to a dataset version")

        return HyperDatasetManagementBackend.commit_version(
            version_id=self._version_id,
            publish=publish,
            force=force,
            calculate_stats=calculate_stats,
            override_stats=override_stats,
            publishing_task=publishing_task,
        )

    def publish_version(
        self,
        *,
        force: bool = False,
        publishing_task: Optional[str] = None,
    ):
        """
        Publish the bound HyperDataset version.

        :param force: If ``True``, force publish even with active annotation tasks (default ``False``).
        :param publishing_task: Annotation task identifier issuing the commit.

        :return: The backend's response payload.
        """
        if not getattr(self, "_version_id", None):
            raise ValueError("HyperDataset instance is not bound to a dataset version")

        return HyperDatasetManagementBackend.publish_version(
            version_id=self._version_id,
            force=force,
            publishing_task=publishing_task,
        )
