import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Any, Sequence, Dict, Union

import psutil
from requests.compat import json as requests_json

from clearml.backend_api import Session
from clearml.backend_interface.datasets.hyper_dataset import HyperDatasetManagementBackend
from clearml.backend_interface.util import get_or_create_project
from clearml.storage.helper import StorageHelperDiskSpaceFileSizeStrategy
from clearml.storage.manager import StorageManagerDiskSpaceFileSizeStrategy
from clearml.storage.hashing import sha256sum
from .data_entry import DataEntry, ENTRY_CLASS_KEY, _resolve_class
from .data_entry_image import DataEntryImage
from .management import HyperDatasetManagement
from ..backend_interface.datasets.save_frames_request_no_validate_wrapped import _get_save_frames_request_no_validate


COMMIT_ERROR_KEY = "__commit_version_error__"

TagsInputValue = Union[Sequence[str], str]


class HyperDataset(HyperDatasetManagement):
    MAX_HASH_FETCH_BATCH_SIZE = 100
    SOURCE_FIELDS = ["source", "preview_source", "mask_source"]
    SNAPSHOT_VERSION_NAME_REGEX = re.compile(r"snapshot\s+(?P<number>\d+)\s+\((?P<name>.+?)\)")

    def __init__(
        self,
        project_name: str,
        dataset_name: str,
        version_name: str,
        description: Optional[str] = None,
        parent_id: Optional[str] = None,
        # parent_ids: Optional[List[str]] = None,
        field_mappings: Optional[Dict[str, Any]] = None,
        raise_if_exists: bool = False,
        **kwargs,
    ):
        """
        Create a new HyperDataset version within the requested project.

        :param project_name: ClearML project name that will own the dataset.
        :param dataset_name: HyperDataset collection name (top-level dataset).
        :param version_name: Version name to create (or reuse if it already exists).
        :param description: Description for the dataset.
        :param parent_id: Parent dataset version ID to link.
        :param parent_ids: (Deprecated) List of parent dataset version IDs to link.
            Only one parent ID per hyperdataset is supported. Use ``parent_id`` instead.
        :param field_mappings: Mapping that defines vector-capable metadata fields.
            Provide the fully-qualified frame metadata path (e.g. ``meta.my_vector``) and
            the corresponding field settings accepted by the ClearML backend / Elasticsearch
            dense vector type. For example:

            .. code-block:: py

                field_mappings = {
                    "meta.qa_vector": {
                        "type": "dense_vector",
                        "element_type": "float",
                        "dims": 768,
                    }
                }

            When supplied, ClearML Server >= 3.25 is required and vector dimensions are
            validated on every frame ingest or update.
        :param raise_if_exists: Reserved flag for compatibility (currently unused).
        """
        Session.verify_feature_set("advanced")
        self._project_name = project_name
        self._dataset_name = dataset_name
        self._version_name = version_name
        self._description = description
        self._field_mappings = field_mappings

        self._parent_id = HyperDatasetManagementBackend._define_parent_id(
            parent_id=parent_id,
            parent_ids=kwargs.get("parent_ids", None),
        )
        self._parent_ids = self._parent_id

        try:
            self._project_id = get_or_create_project(Session(), project_name)
        except Exception:
            self._project_id = None
        self._dataset_id = HyperDatasetManagementBackend.create_dataset(
            name=dataset_name,
            comment=description,
            project=self._project_id,
            field_mappings=field_mappings,
        )
        self._version_id = HyperDatasetManagementBackend.create_version(
            name=version_name,
            dataset_id=self._dataset_id,
            parent_id=self._parent_id,
        )
        self._tags = None  # Create uninitialized _tags cache

    def add_data_entries(
        self,
        data_entries: Sequence[DataEntry],
        upload_local_files_destination: Optional[str] = None,
        batch_size: int = 1000,
        max_workers: Optional[int] = None,
        show_progress: bool = True,
        upload_retries: int = 5,
        force_upload: bool = False,
        max_request_size_mb: int = 100,
        hash_sources: bool = False,
    ):
        """
        Upload and register a collection of data entries into the HyperDataset version.
        Successful registrations automatically trigger a commit to refresh the version statistics.

        :param data_entries: Sequence of ``DataEntry`` instances to register.
        :param upload_local_files_destination: Storage URI for uploading local sources.
        :param batch_size: Number of entries per backend registration batch.
        :param max_workers: Maximum number of threads for upload work.
        :param show_progress: Reserved for API compatibility (no progress emitted currently).
        :param upload_retries: Number of upload retry attempts per file.
        :param force_upload: If ``True``, upload sources even when their hash indicates they already exist
            on the backend. If ``False`` (default), sources already present on the backend are not
            re-uploaded.
        :param max_request_size_mb: Upper bound for registration request payload size, in MB
            (default 100). Registration batches whose payload exceeds this size are split into
            multiple requests. A single entry larger than this limit is sent in its own request.
            Pass ``None`` (or 0) to disable splitting and send each batch as a single request
            regardless of its payload size.
        :param hash_sources: If ``True``, compute hashes of local sources before upload so they can be
            deduplicated. If ``False`` (default), sources are not hashed.

        :return: A dictionary with ``upload`` and ``register`` keys, each mapping entry IDs to the error
            raised for that entry (empty on success).
        """
        HyperDataset._verify_upload_destination(upload_local_files_destination)
        errors = {"upload": {}, "register": {}}
        should_commit = False
        with ThreadPoolExecutor(max_workers=max_workers or psutil.cpu_count()) as thread_pool:
            for i in range(0, len(data_entries), batch_size):
                batched_data_entries = data_entries[i: i + batch_size]
                upload_errors = self._upload_data_entries(
                    data_entries=batched_data_entries,
                    upload_destination=upload_local_files_destination,
                    retries=upload_retries,
                    force_upload=force_upload,
                    hash_sources=hash_sources,
                    thread_pool=thread_pool,
                )
                register_errors = self._register_data_entries_batched_request_size(
                    data_entries=batched_data_entries,
                    max_request_size_mb=max_request_size_mb,
                ) or {}
                errors["upload"].update(upload_errors)
                errors["register"].update(register_errors)
                if batched_data_entries and len(register_errors) < len(batched_data_entries):
                    should_commit = True
        if should_commit:
            try:
                self.commit_version()
            except Exception as exc:
                errors["register"][COMMIT_ERROR_KEY] = exc
        return errors

    def delete_data_entries(
        self,
        data_entries: Sequence[Union[DataEntry, str]],
        batch_size: int = 1000,
        refresh_version_stats: bool = True,
        force: bool = False,
    ) -> int:
        """
        Delete data entries (frames) from this HyperDataset version.

        Entries may be passed as ``DataEntry`` objects (for example, objects yielded by
        ``get_iterator()``) or as entry ID strings. Entries are deleted by their IDs;
        all other attributes are ignored.

        .. note:: Only writable (non-published) versions can delete their entries.

        :param data_entries: Sequence of ``DataEntry`` instances and/or entry ID strings.
        :param batch_size: Number of entry IDs sent per delete request (default 1000).
            It does not limit the number of entries per call.
        :param refresh_version_stats: If ``True`` (default), commit the version after deleting to refresh
            its statistics. If ``False``, skip the commit.
        :param force: If ``True``, delete the entries even if there are ongoing annotation tasks using this
            version as input. If ``False`` (default), such tasks may prevent the deletion.

        :return: The total number of entries the backend reports as deleted.
        """
        entry_ids = []
        for data_entry in data_entries:
            entry_id = data_entry if isinstance(data_entry, str) else getattr(data_entry, "id", None)
            if not entry_id or not isinstance(entry_id, str):
                raise ValueError(
                    f"delete_data_entries expects DataEntry objects or entry ID strings, got {data_entry!r}"
                )
            entry_ids.append(entry_id)

        if not entry_ids:
            return 0

        batch_size = max(1, int(batch_size))
        deleted = 0
        for i in range(0, len(entry_ids), batch_size):
            deleted += HyperDatasetManagementBackend.delete_data_entries(
                version_id=self.version_id,
                entry_ids=entry_ids[i: i + batch_size],
                force=force,
            )

        if deleted and refresh_version_stats:
            self.commit_version()

        return deleted

    def set_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        Store metadata (dict) of user-defined values on this HyperDataset version.

        The supplied dictionary replaces any previously stored metadata.

        :param metadata: Key/value dictionary (with support for nested dictionaries).
            Keys must not include ``$`` or ``.``.

        :return: ``True`` if successful. Locked or published versions cannot change their version metadata.
        """
        if not isinstance(metadata, dict):
            raise ValueError("set_metadata expects a key/value dictionary")
        if not getattr(self, "_version_id", None) or not getattr(self, "_dataset_id", None):
            raise ValueError("HyperDataset instance is not bound to a dataset version")

        return HyperDatasetManagementBackend.set_version_metadata(
            dataset_id=self._dataset_id,
            version_id=self._version_id,
            metadata=metadata,
        )

    def get_metadata(self) -> Dict[str, Any]:
        """
        Return the metadata (dict) of user-defined values stored on this HyperDataset version.

        The metadata is fetched from the backend on every call.

        :return: Metadata dictionary; empty when none was set.
        """
        if not getattr(self, "_version_id", None) or not getattr(self, "_dataset_id", None):
            raise ValueError("HyperDataset instance is not bound to a dataset version")

        return HyperDatasetManagementBackend.get_version_metadata(
            dataset_id=self._dataset_id,
            version_id=self._version_id,
        )

    def get_iterator(
        self,
        projection: Optional[Sequence[str]] = None,
        batch_size: Optional[int] = None,
    ):
        """
        Return an iterator over all the data entries (frames) of this HyperDataset version.

        This is a convenience wrapper around a single-query ``DataView`` pinned to this
        version: entries stream through the same iteration pipeline as ``DataView``
        iteration (background prefetching, entry-class conversion), but no ``DataView``
        object is persisted on the server and no Task is attached.

        For filtered, weighted, or multi-version iteration, build a ``DataView`` instead.

        :param projection: List of frame fields to return, using dot-separated
            notation (for example ``["id", "sources"]``). When set, entries are
            reconstructed from the projected fields only — non-projected fields are
            missing from the returned objects.
        :param batch_size: Number of entries fetched per backend request.

        :return: Iterator yielding ``DataEntry``-derived objects.
        """
        if not getattr(self, "_version_id", None) or not getattr(self, "_dataset_id", None):
            raise ValueError("HyperDataset instance is not bound to a dataset version")

        version = HyperDatasetManagementBackend.get_version_by_id(
            dataset_id=self._dataset_id,
            version_id=self._version_id,
            only_fields=["id", "status"],
        )
        if version is None:
            raise ValueError(f"Version not found: {self._version_id} (dataset={self._dataset_id})")
        if str(getattr(version, "status", None) or "") != "published":
            logging.getLogger("HyperDataset").warning(
                "Iterating over a non-published dataset version %s", self._version_id
            )

        # Local import to avoid a circular dependency (data_view imports this module)
        from .data_view import DataView

        dataview = DataView(auto_connect_with_task=False)
        # Version iteration must not litter the server with anonymous DataView objects
        dataview._store_on_iteration = False
        dataview.add_query(dataset_id=self._dataset_id, version_id=self._version_id)
        return dataview.get_iterator(
            projection=projection,
            **({"query_cache_size": int(batch_size)} if batch_size else {}),
        )

    @staticmethod
    def _verify_upload_destination(upload_destination: Optional[str] = None):
        """
        Validate that the upload destination is a writable ClearML storage URI.

        :param upload_destination: Storage URI to validate
        """
        if not upload_destination:
            return
        helper = StorageHelperDiskSpaceFileSizeStrategy.get(upload_destination)
        if not helper:
            raise ValueError(
                f"Could not get access credentials for '{upload_destination}' "
                ", check configuration file ~/clearml.conf"
            )
        helper.check_write_permissions(upload_destination)

    def _register_data_entries_batched_request_size(
        self,
        data_entries: Sequence[DataEntry],
        max_request_size_mb: int = None,
    ):
        """
        Register data entries while splitting requests to respect the maximum payload size.

        :param data_entries: Sequence of data entries to register
        :param max_request_size_mb: Maximum request size (MB); `None` disables batching

        :return: Mapping of entry IDs to registration errors
        """
        if max_request_size_mb is not None:
            request_fixed_size = len(
                requests_json.dumps(
                    _get_save_frames_request_no_validate()(version=self._version_id, frames=[]).to_dict()
                ).encode("utf-8")
            )
            max_request_size_bytes = (max_request_size_mb * 1024 * 1024) - request_fixed_size
            batched_data_entries = self._batch_data_entries_by_payload_size(
                data_entries=data_entries,
                max_request_size_bytes=max_request_size_bytes,
            )
        else:
            batched_data_entries = [data_entries]

        errors = {}
        for data_entry_batch in batched_data_entries:
            errors.update(self._register_data_entries(data_entry_batch))

        return errors

    def _batch_data_entries_by_payload_size(
        self,
        data_entries: Sequence[DataEntry],
        max_request_size_bytes: int,
    ) -> List[List[Any]]:
        """
        Aggregate a list of data entries in batches based on the payload size of registering them to the API server.

        :param data_entries: Sequence of data entries to register
        :param max_request_size_mb: Maximum request size (in bytes).

        :return: A list of data entry batches aggregated in lists for individual calls to `self._register_data_entries`.
        """
        data_entry_payload_sizes = [
            len(requests_json.dumps(data_entry.to_api_object()).encode("utf-8"))
            for data_entry in data_entries
        ]

        current_batch = []
        current_batch_size = 0
        batches = []
        for data_entry, data_entry_payload_size in zip(data_entries, data_entry_payload_sizes):
            # Case 1. If the next data entry does not fit in a batch:
            #     - Add the current batch in the list of batches
            #     - Put the current data entry in its own batch
            #     - Reset the batch size calculation
            if data_entry_payload_size > max_request_size_bytes:
                if len(current_batch) > 0:
                    batches.append(current_batch)

                batches.append([data_entry])
                current_batch = []
                current_batch_size = 0
            # Case 2. If the current batch together with the next data entry does not fit in a batch
            #    - Add the current batch into the list of batches
            #    - Start a new batch with the next data entry
            #    - Update the batch size calculation
            elif current_batch_size + data_entry_payload_size > max_request_size_bytes:
                if len(current_batch) > 0:
                    batches.append(current_batch)

                current_batch = [data_entry]
                current_batch_size = data_entry_payload_size
            # Case 3. If the current batch together with the next data entry fits in a batch
            #    - Add the next data entry in the current batch
            #    - Update the batch size calculation
            else:
                current_batch.append(data_entry)
                current_batch_size += data_entry_payload_size

        if len(current_batch) > 0:
            batches.append(current_batch)

        return batches

    def _register_data_entries(
        self,
        data_entries: Sequence[DataEntry],
    ) -> Dict[str, Any]:
        """
        Register a batch of data entries against the current dataset version.

        :param data_entries: Iterable of data entries to register

        :return: Mapping of entry IDs to registration errors
        """
        if not data_entries:
            return {}
        try:
            response = HyperDatasetManagementBackend.save_data_entries(self._version_id, data_entries)
        except Exception as e:
            errors = {data_entry.id: e for data_entry in data_entries}
            return errors

        errors = {}
        for error in response.errors:
            try:
                data_entry_id = (error.get("_id") or error.get("index", {}).get("_id", "")).partition("/")[2]
            except Exception:
                continue
            errors[data_entry_id] = error.get("error") or error.get("index", {}).get("error", {}).get("reason")
        return errors

    def _upload_data_entries(
        self,
        data_entries: Sequence[DataEntry],
        upload_destination: Optional[str] = None,
        retries: int = 5,
        hash_batch_size: int = MAX_HASH_FETCH_BATCH_SIZE,
        force_upload: bool = False,
        hash_sources: bool = False,
        thread_pool=None,
    ):
        """
        Upload local sources for the supplied data entries using the provided thread pool.

        :param data_entries: Sequence of data entries whose sources might require uploading
        :param upload_destination: Explicit storage destination URI, optional per-entry override
        :param retries: Number of retry attempts for failed uploads
        :param hash_batch_size: Batch size used when checking existing hashes
        :param force_upload: Upload even if matching hashes are detected on the backend
        :param hash_sources: Whether to compute hashes before upload for deduplication
        :param thread_pool: Thread pool used for concurrent uploads

        :return: Mapping of entry IDs to upload errors
        """
        hash_batch_size = min(hash_batch_size, HyperDataset.MAX_HASH_FETCH_BATCH_SIZE)

        if not upload_destination:
            data_entries = [d for d in data_entries if d._has_upload_destination()]

        if hash_sources:
            # calculate hashes before potential dedup/upload
            for data_entry in data_entries:
                for sub_data_entry in data_entry:
                    src = sub_data_entry.get_source("source")
                    if src and os.path.isfile(src):
                        try:
                            sub_data_entry._source_hash = sha256sum(src)
                        except Exception:
                            pass
                    psrc = sub_data_entry.get_source("preview_source")
                    if psrc and os.path.isfile(psrc):
                        try:
                            sub_data_entry._preview_source_hash = sha256sum(psrc)
                        except Exception:
                            pass

        if not force_upload:
            # if we are not forced to upload, fetch the already uploaded files based on hashes
            for i in range(0, len(data_entries), hash_batch_size):
                self._set_already_uploaded_files(data_entries[i: i + hash_batch_size])

        upload_errors = {}
        futures = []
        for data_entry in data_entries:
            futures.append(
                thread_pool.submit(
                    self._upload_sources, data_entry, upload_errors=upload_errors, upload_destination=upload_destination
                )
            )
        for future in futures:
            future.result()
        return upload_errors

    def _upload_sources(
        self,
        data_entry: DataEntry,
        upload_errors: dict,
        upload_destination: Optional[str] = None,
    ):
        """
        Upload sources for a single data entry to the target storage destination.

        :param data_entry: Data entry whose sub-sources should be uploaded
        :param upload_errors: Error mapping to populate on failure
        :param upload_destination: Storage destination override

        :return: None
        """
        for sub_data_entry in data_entry:
            # Pick destination: explicit param or per-subentry destination
            dest = upload_destination or sub_data_entry._local_sources_upload_destination
            if not dest:
                continue
            for source_field in HyperDataset.SOURCE_FIELDS:
                source_to_upload = sub_data_entry.get_source(source_field)
                if not source_to_upload:
                    continue
                if not os.path.isfile(source_to_upload):
                    continue
                try:
                    result = StorageManagerDiskSpaceFileSizeStrategy.upload_file(
                        source_to_upload, self._build_source_upload_uri(source_to_upload, dest)
                    )
                    sub_data_entry.set_source(source_field, result)
                except Exception as e:
                    upload_errors[data_entry.id] = str(e)

    def _build_source_upload_uri(
        self,
        source,
        upload_destination,
    ):
        """
        Construct the destination URI for a source file within the dataset hierarchy.

        :param source: Local source path
        :param upload_destination: Base storage URI

        :return: Fully-qualified destination URI for the source
        """
        base = upload_destination.rstrip("/")
        return base + "/" + (self._dataset_id or "") + "/" + (self._version_id or "") + "/" + os.path.basename(source)

    def _set_already_uploaded_files(
        self,
        data_entries: Sequence[DataEntry],
    ):
        """
        Reuse existing remote sources by matching hashes of already-uploaded files.

        :param data_entries: Iterable of data entries to inspect for existing hashes

        :return: None
        """
        hash_to_uploaded_file = {}

        # Import locally to avoid circular dependencies
        from .data_view import DataView
        data_view = DataView(
            iteration_order="sequential",
            iteration_infinite=False,
            auto_connect_with_task=False,
        )

        have_hashes = False
        for source_field in HyperDataset.SOURCE_FIELDS:
            # get hashes present in the frames we want to add
            hashes = []
            for data_entry in data_entries:
                for sub_data_entry in data_entry.sub_data_entries:
                    hash_ = sub_data_entry.get_hash(source_field)
                    if hash_:
                        hashes.append(hash_)

            # add query to fetch the frames with sources that have the same hash as the ones we want to upload
            if hashes:
                lucene = " OR ".join(
                    f'"{hash_}"'
                    for hash_ in hashes
                    if hash_ not in hash_to_uploaded_file
                )
                if lucene:
                    have_hashes = True
                    data_view.add_query(
                        project_id=self._project_id,
                        dataset_id=self._dataset_id,
                        version_id="*",
                        source_query=f"sources.meta.hash.{source_field}:({lucene})",
                    )
        if not have_hashes:
            return

        for found_data_entry in data_view.get_iterator():
            for found_sub_data_entry in found_data_entry:
                hash_to_uploaded_file[
                    found_sub_data_entry.get_hash(source_field)
                ] = found_sub_data_entry.get_source(source_field)

            # set the source as the remote destination based on hash - no need to upload
        for data_entry in data_entries:
            for sub_data_entry in data_entry:
                if sub_data_entry.get_hash(source_field) in hash_to_uploaded_file:
                    sub_data_entry.set_source(
                        source_field, hash_to_uploaded_file[sub_data_entry.get_hash(source_field)]
                    )

    def vector_search(
        self,
        reference_vector: Sequence[float],
        vector_field: str,
        number_of_neighbors: int = 50,
        fast: bool = False,
        similarity_function: str = "cosine",
    ) -> List[Any]:
        if reference_vector is None:
            raise ValueError("reference_vector must be provided")
        if not isinstance(reference_vector, (list, tuple)):
            raise TypeError("reference_vector must be a list or tuple")
        if not reference_vector:
            raise ValueError("reference_vector cannot be empty")

        if number_of_neighbors <= 0:
            raise ValueError("number_of_neighbors must be a positive integer")
        if not isinstance(vector_field, str) or not vector_field.strip():
            raise TypeError("vector_field must be a non-empty string")

        similarity = self._normalize_similarity_function(similarity_function)
        if fast and similarity != "cosine":
            raise ValueError("fast vector search currently supports only cosine similarity")

        field_path = vector_field if vector_field.startswith("meta.") else f"meta.{vector_field}"

        payload = {
            "order_by": [{"field": "context_id", "order": "desc"}],
            "dataview": {
                "versions": [{"dataset": self._dataset_id, "version": self._version_id}],
                "filters": [
                    {
                        "label_rules": [],
                        "filter_by_roi": "label_rules",
                        "frame_query": None,
                        "sources_query": None,
                        "dataset": self._dataset_id,
                        "version": self._version_id,
                        "weight": 1,
                    }
                ],
                "iteration": {"order": "sequential"},
            },
            "disable_aggregation": True,
            "vector_search": {
                "vector": list(reference_vector),
                "field": field_path,
                "method": "fast" if fast else "exact",
                "similarity_func": similarity,
                "neighbors": number_of_neighbors,
            },
            "size": number_of_neighbors,
            "search_after": None,
        }

        session = Session()
        response = session.send_request(
            service="frames",
            action="get_snippets_for_dataview2",
            version="2.34",
            method="post",
            json=payload,
        )
        if not response.ok:
            raise ValueError(
                f"Vector search request failed with status {response.status_code}: {response.text}"
            )

        body = response.json() or {}
        data = body.get("data") or {}
        frames = data.get("frames") or body.get("frames") or []
        return [self._convert_frame_to_entry(frame) for frame in frames]

    @staticmethod
    def _normalize_similarity_function(similarity: str) -> str:
        if not isinstance(similarity, str):
            raise TypeError("similarity_function must be a string")
        normalized = similarity.strip().lower()
        if normalized == "12_norm":
            normalized = "l2_norm"
        valid = {"cosine", "l2_norm", "dot_product"}
        if normalized not in valid:
            raise ValueError(f"Unsupported similarity function: {similarity}")
        return normalized

    @staticmethod
    def _determine_entry_classes(frame):
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
        elif HyperDataset._frame_looks_like_image(frame):
            base_cls = DataEntryImage
        return base_cls, resolved_cls

    @staticmethod
    def _frame_looks_like_image(frame) -> bool:
        def _get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        sources = _get(frame, "sources") or []
        if not isinstance(sources, (list, tuple)):
            return False
        for source in sources:
            if _get(source, "width") is not None or _get(source, "height") is not None:
                return True
            preview = _get(source, "preview")
            if preview and _get(preview, "uri"):
                return True
            masks = _get(source, "masks")
            if masks:
                return True
        return False

    @classmethod
    def _convert_frame_to_entry(cls, frame):
        base_cls, resolved_cls = cls._determine_entry_classes(frame)
        if hasattr(base_cls, "from_api_object"):
            try:
                entry = base_cls.from_api_object(frame)
                if (
                    resolved_cls
                    and issubclass(resolved_cls, DataEntry)
                    and isinstance(entry, DataEntry)
                    and entry.__class__ is not resolved_cls
                ):
                    try:
                        entry.__class__ = resolved_cls
                    except TypeError:
                        pass
                return entry
            except Exception:
                pass
        return frame

    # Public accessors for IDs
    @property
    def project_id(self) -> Optional[str]:
        """
        :return: ClearML Project ID associated with this HyperDataset.
        """
        return self._project_id

    @property
    def dataset_id(self) -> Optional[str]:
        """
        :return: HyperDataset dataset ID.
        """
        return self._dataset_id

    @property
    def version_id(self) -> Optional[str]:
        """
        :return: HyperDataset version ID.
        """
        return self._version_id

    @property
    def tags(self) -> List[str]:
        """
        Pythonic alternative to ``dataset.get_tags()``.

        :return: The list of tags attached to this dataset.
        """
        return self.get_tags()

    @tags.setter
    def tags(self, tags: Optional[TagsInputValue]) -> None:
        """
        Pythonic alternative to ``dataset.set_tags(tags=[...])``.

        :param tags: Tags to set on the dataset. Accepts a single space-separated string, or a sequence
            of tag strings.
        """
        self.set_tags(tags=(tags or []))

    def get_tags(self, reload: bool = False) -> List[str]:
        """
        Return the tags attached to the dataset. Allows invalidating the tags cache via
        ``dataset.get_tags(reload=True)``.

        :param reload: If ``True``, fetch tags from the backend and update the local runtime cache
            (default ``False``).

        :return: The list of tags attached to this dataset.
        """
        if (
            reload  # Invalidate the _tags cache and refetch tags from backend
            or self._tags is None  # If _tags cache is not initialized, initialize it by fetching tags
        ):
            dataset = HyperDatasetManagementBackend.get_dataset_by_id(dataset_id=self.dataset_id)
            self._tags = (
                list(getattr(dataset, "tags", []) or [])
                if dataset
                else []
            )

        return self._tags

    def add_tags(self, tags: TagsInputValue) -> None:
        """
        Add the provided tags to the dataset.

        :param tags: Tags to add to the dataset. Accepts a single space-separated string, or a sequence
            of tag strings.
        """
        tags_to_add = self._normalize_tags(tags=tags)
        new_tags = list(set((
            *self.tags,
            *(tags_to_add or []),
        )))
        HyperDatasetManagementBackend.update_dataset_tags(
            dataset_id=self.dataset_id,
            tags=new_tags,
        )
        self._tags = new_tags

    def remove_tags(self, tags: TagsInputValue) -> None:
        """
        Remove the provided tags from the dataset.

        :param tags: Tags to remove from the dataset. Accepts a single space-separated string, or a sequence
            of tag strings.
        """
        tags_to_remove = self._normalize_tags(tags=tags)
        new_tags = [
            tag
            for tag in (self.tags or [])
            if tag not in tags_to_remove
        ]
        HyperDatasetManagementBackend.update_dataset_tags(
            dataset_id=self.dataset_id,
            tags=new_tags,
        )
        self._tags = new_tags

    def set_tags(self, tags: TagsInputValue) -> None:
        """
        Override the dataset's existing tags with the provided tags.

        :param tags: Tags to set on the dataset. Accepts a single space-separated string, or a sequence
            of tag strings.
        """
        new_tags = self._normalize_tags(tags=tags)
        HyperDatasetManagementBackend.update_dataset_tags(
            dataset_id=self.dataset_id,
            tags=new_tags,
        )
        self._tags = new_tags

    def _normalize_tags(self, tags: TagsInputValue) -> List[str]:
        """
        Converts a user-provided sequence of tags into a backend-admissible list of tags.

        Accepted input formats: space-separated list of tags, any Sequence[str] type.
        Output type: List[str]

        :param tags: The list of tags that will be normalized.

        :return: A list of strings, each representing a tag.
        """
        return (
            tags.split(" ")
            if isinstance(tags, str)
            else list(tags)
        )

    def create_snapshot(self) -> "HyperDataset":
        """
        Publish the current version of this ``HyperDataset`` as an immutable snapshot, then continue
        working on a new child version. This instance is updated in place to point to the new child
        version.

        :return: A new ``HyperDataset`` instance bound to the published snapshot version.
        """
        new_version_id = HyperDatasetManagementBackend.publish_and_create_child_version(
            dataset_id=self.dataset_id,
            version_id=self.version_id,
        )

        # Update the swapped versions' IDs
        # This hyperdataset's version ID has been swapped with the new snapshot's ID in the backend,
        # but not in the SDK, so we do the swap here as well
        self._version_id, snapshot_version_id = new_version_id, self._version_id

        # Now that we updated the version IDs, we can query the backend
        # and update the instances as well
        modified_self_response = HyperDatasetManagementBackend.get_version_by_id(
            dataset_id=self.dataset_id,
            version_id=self.version_id,
        )
        self._version_name = (
            getattr(modified_self_response, "name", None)
            or self._version_name
        )

        # Instantiate the new hyperdataset
        snapshot_response = HyperDatasetManagementBackend.get_version_by_id(
            dataset_id=self.dataset_id,
            version_id=snapshot_version_id,
        )

        snapshot_hyperdataset = HyperDataset(
            project_name=self._project_name,
            dataset_name=self._dataset_name,
            version_name=getattr(snapshot_response, "name", None),
            parent_id=self._version_id,
        )

        return snapshot_hyperdataset
