import psutil
import os
from typing import Optional, List, Any, Sequence, Dict
from concurrent.futures import ThreadPoolExecutor
from requests.compat import json as requests_json

from clearml.backend_interface.datasets.hyper_dataset import HyperDatasetManagementBackend, _SaveFramesRequestNoValidate
from clearml.backend_interface.util import get_or_create_project
from clearml.backend_api import Session
from clearml.storage.helper import StorageHelperDiskSpaceFileSizeStrategy
from clearml.storage.manager import StorageManagerDiskSpaceFileSizeStrategy
from clearml.storage.util import sha256sum

from .data_entry import DataEntry, ENTRY_CLASS_KEY, _resolve_class
from .data_entry_image import DataEntryImage
from .management import HyperDatasetManagement


class HyperDataset(HyperDatasetManagement):
    MAX_HASH_FETCH_BATCH_SIZE = 100
    SOURCE_FIELDS = ["source", "preview_source", "mask_source"]

    def __init__(
        self,
        project_name: str,
        dataset_name: str,
        version_name: str,
        description: Optional[str] = None,
        parent_ids: Optional[List[str]] = None,
        field_mappings: Optional[Dict[str, Any]] = None,
        raise_if_exists: bool = False,
    ):
        """Create a new HyperDataset version within the requested project.

        :param project_name: ClearML project name that will own the dataset
        :param dataset_name: HyperDataset collection name (top-level dataset)
        :param version_name: Version name to create (or reuse if it already exists)
        :param description: Optional dataset description string
        :param parent_ids: Optional list of parent dataset version IDs to link
        :param field_mappings: Optional mapping that defines vector-capable metadata fields.
            Provide the fully-qualified frame metadata path (e.g. ``meta.my_vector``) and
            the corresponding field settings accepted by the ClearML backend / Elasticsearch
            dense vector type. For example::

                field_mappings = {
                    "meta.qa_vector": {
                        "type": "dense_vector",
                        "element_type": "float",
                        "dims": 768,
                    }
                }

            When supplied, ClearML Server >= 3.25 is required and vector dimensions are
            validated on every frame ingest or update.
        :param raise_if_exists: Reserved flag for compatibility (currently unused)
        """
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
            name=version_name, dataset_id=self._dataset_id, parent_ids=parent_ids
        )

    def add_data_entries(
        self,
        data_entries,
        upload_local_files_destination: Optional[str] = None,
        batch_size: int = 1000,
        max_workers: Optional[int] = None,
        show_progress: bool = True,
        upload_retries: int = 5,
        force_upload: bool = False,
        max_request_size_mb: int = None,
        hash_sources: bool = False,
    ):
        """
        Upload and register a collection of data entries into the HyperDataset version.

        :param data_entries: Iterable of `DataEntry` instances to register
        :param upload_local_files_destination: Optional storage URI for uploading local sources
        :param batch_size: Number of entries per backend registration batch
        :param max_workers: Maximum number of threads for upload work
        :param show_progress: Reserved for API compatibility (no progress emitted currently)
        :param upload_retries: Number of upload retry attempts per file
        :param force_upload: Upload even when hashes indicate the source already exists
        :param max_request_size_mb: Optional upper bound for registration request payload size
        :param hash_sources: Whether to hash sources for deduplication prior to upload

        :return: Dictionary containing upload and registration error mappings
        """
        HyperDataset._verify_upload_destination(upload_local_files_destination)
        errors = {"upload": {}, "register": {}}
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
                if max_request_size_mb:
                    register_errors = self._register_data_entries_batched_request_size(
                        data_entries=batched_data_entries, max_request_size_mb=max_request_size_mb
                    )
                else:
                    register_errors = self._register_data_entries(data_entries=batched_data_entries)
                errors["upload"].update(upload_errors)
                errors["register"].update(register_errors)
        return errors

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
                "Could not get access credentials for '{}' "
                ", check configuration file ~/clearml.conf".format(
                    upload_destination
                )
            )
        helper.check_write_permissions(upload_destination)

    def _register_data_entries_batched_request_size(self, data_entries, max_request_size_mb: int = None):
        """
        Register data entries while splitting requests to respect the maximum payload size.

        :param data_entries: Iterable of data entries to register
        :param max_request_size_mb: Maximum request size (MB); `None` disables batching

        :return: Mapping of entry ids to registration errors
        """
        if not max_request_size_mb:
            return self._register_data_entries(data_entries)

        current_batch = []
        current_batch_size = 0
        errors = {}
        request_fixed_size = len(
            requests_json.dumps(_SaveFramesRequestNoValidate(version=self._version_id, frames=[]).to_dict()).encode(
                "utf-8"
            )
        )
        max_request_size_bytes = (max_request_size_mb * 1024 * 1024) - request_fixed_size
        for data_entry in data_entries:
            payload = data_entry.to_api_object()
            data_entry_payload_size = len(requests_json.dumps(payload).encode("utf-8"))
            if data_entry_payload_size > max_request_size_bytes:
                errors[data_entry.id] = "Data entry payload exceeds 'max_request_size_mb'"
                continue
            if current_batch_size + len(current_batch) + data_entry_payload_size > max_request_size_bytes:
                errors.update(self._register_data_entries(current_batch))
                current_batch = []
                current_batch_size = 0
            current_batch.append(data_entry)
            current_batch_size += data_entry_payload_size
        errors.update(self._register_data_entries(current_batch))
        return errors

    def _register_data_entries(self, data_entries):
        """
        Register a batch of data entries against the current dataset version.

        :param data_entries: Iterable of data entries to register

        :return: Mapping of entry ids to registration errors
        """
        errors = {}
        if not data_entries:
            return errors
        try:
            response = HyperDatasetManagementBackend.save_data_entries(self._version_id, data_entries)
        except Exception as e:
            errors = {data_entry.id: e for data_entry in data_entries}
            return errors

        for error in response.errors:
            try:
                data_entry_id = (error.get("_id") or error.get("index", {}).get("_id", "")).partition("/")[2]
            except Exception:
                continue
            errors[data_entry_id] = error.get("error") or error.get("index", {}).get("error", {}).get("reason")
        return errors

    def _upload_data_entries(
        self,
        data_entries,
        upload_destination: Optional[str] = None,
        retries: int = 5,
        hash_batch_size: int = MAX_HASH_FETCH_BATCH_SIZE,
        force_upload: bool = False,
        hash_sources: bool = False,
        thread_pool=None,
    ):
        """
        Upload local sources for the supplied data entries using the provided thread pool.

        :param data_entries: Iterable of data entries whose sources might require uploading
        :param upload_destination: Explicit storage destination URI, optional per-entry override
        :param retries: Number of retry attempts for failed uploads
        :param hash_batch_size: Batch size used when checking existing hashes
        :param force_upload: Upload even if matching hashes are detected on the backend
        :param hash_sources: Whether to compute hashes before upload for deduplication
        :param thread_pool: Thread pool used for concurrent uploads

        :return: Mapping of entry ids to upload errors
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

    def _upload_sources(self, data_entry: Any, upload_errors: dict, upload_destination: Optional[str] = None):
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

    def _build_source_upload_uri(self, source, upload_destination):
        """
        Construct the destination URI for a source file within the dataset hierarchy.

        :param source: Local source path
        :param upload_destination: Base storage URI

        :return: Fully-qualified destination URI for the source
        """
        base = upload_destination.rstrip("/")
        return base + "/" + (self._dataset_id or "") + "/" + (self._version_id or "") + "/" + os.path.basename(source)

    def _set_already_uploaded_files(self, data_entries):
        """
        Reuse existing remote sources by matching hashes of already-uploaded files.

        :param data_entries: Iterable of data entries to inspect for existing hashes

        :return: None
        """
        hash_to_uploaded_file = {}

        # Import locally to avoid circular dependencies
        from .data_view import DataView
        data_view = DataView(
            iteration_order="sequential", iteration_infinite=False, auto_connect_with_task=False
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
                lucene = " OR ".join('"{}"'.format(h) for h in hashes if h not in hash_to_uploaded_file)
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
