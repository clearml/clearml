import fnmatch
import logging
import shutil
from multiprocessing.pool import ThreadPool
from random import random
from time import time
from typing import Iterator, List, Optional, Union
from urllib.parse import quote

from pathlib2 import Path

from .cache import CacheManager
from .callbacks import ProgressReport
from .helper import StorageHelper, StorageHelperDiskSpaceFileSizeStrategy, _StorageHelper
from .archive import extract_tar_archive, extract_zip_archive
from ..config import deferred_config
from ..debugging.log import LoggerRoot


class StorageManager:
    """
    StorageManager provides an interface for uploading and downloading files to and from remote storage.
    Supported remote servers: ``http(s)``, ``s3``, ``gs``, ``azure``, and shared filesystem.
    Caching is enabled by default for all downloaded files.
    """
    _file_upload_retries = deferred_config("network.file_upload_retries", 3)

    storage_helper = StorageHelper
    """
    :meta private:
    """

    @classmethod
    def get_local_copy(
        cls,
        remote_url: str,
        cache_context: Optional[str] = None,
        extract_archive: bool = True,
        name: Optional[str] = None,
        force_download: bool = False,
    ) -> Optional[str]:
        """
        Returns a local path to the given remote file.

        If the remote URL points to a directly accessible local file, it is returned as-is.
        Otherwise, the file is downloaded and stored in the local cache, and the cached path is returned.

        Each cache context holds up to 100 files by default. When the limit is reached, the least recently accessed
        files are deleted to make room. Calling this function on an already-cached file refreshes its last-accessed
        timestamp, preventing its deletion.

        :param remote_url: URL of the remote file to retrieve a local copy of.
        :param cache_context: Cache context identifier. Defaults to ``'global'``.
        :param extract_archive: If ``True``, and the file is a supported archive (currently zip files only), return the
            path to the extracted archive contents instead of the archive file itself.
        :param name: Name of the target file
        :param force_download: If ``True``, re-download even if a cached copy exists. Defaults to ``False``.
        :return: Full path to local copy of the requested URL. Return ``None`` on error.
        """
        if bool(cls.storage_helper.use_disk_space_file_size_strategy):
            cached_file = cls.storage_helper.get_local_copy(remote_url=remote_url, force_download=force_download)
            if cached_file:
                # noinspection PyProtectedMember
                CacheManager._add_remote_url(remote_url=remote_url, local_copy_path=cached_file)

            if not extract_archive or not cached_file:
                return cached_file

            cache = CacheManager.get_cache_manager(cache_context=cache_context)
            cache_path_encoding = Path(cached_file).parent / cache.get_hashed_url_file(remote_url)
            return cls._extract_to_cache(
                cached_file,
                name='cache',
                cache_context=cache_context,
                cache_path_encoding=cache_path_encoding.as_posix()
            )
        cache = CacheManager.get_cache_manager(cache_context=cache_context)
        cached_file = cache.get_local_copy(remote_url=remote_url, force_download=force_download)
        if extract_archive and cached_file:
            # this will get us the actual cache (even with direct access)
            cache_path_encoding = Path(cache.get_cache_folder()) / cache.get_hashed_url_file(remote_url)
            return cls._extract_to_cache(
                cached_file,
                name,
                cache_context,
                cache_path_encoding=cache_path_encoding.as_posix(),
            )
        return cached_file

    @classmethod
    def get_stream(
        cls,
        remote_url: str,
        chunk_size: Optional[int] = None,
    ) -> Optional[Iterator[bytes]]:
        """
        Download a remote file and return its content as an in-memory stream (an iterator of bytes chunks),
        without writing it to disk or storing it in the local cache.

        Example:

        .. code-block:: py

            from io import BytesIO

            buffer = BytesIO()
            for chunk in StorageManager.get_stream("s3://bucket/data/file.bin"):
                buffer.write(chunk)

        :param remote_url: URL of the remote file to stream. Supports ``http(s)``, ``s3``, ``gs``, ``azure``,
            and shared filesystem. Example: ``'s3://bucket/data/file.bin'``
        :param chunk_size: Optional download chunk size in bytes. Best effort: some storage drivers determine
            their own chunk granularity (Google Storage rounds it up to a multiple of 256 KB, Azure and shared
            filesystem ignore it).
        :return: An iterator yielding the file content as bytes chunks. ``None`` on error.
        """
        helper = cls.storage_helper.get(remote_url)
        # stream directly from the remote storage, explicitly bypassing the cached
        # download_as_stream override (it downloads through the disk cache when the
        # disk space file size strategy is enabled)
        stream = _StorageHelper.download_as_stream(helper, remote_url, chunk_size=chunk_size)
        if stream is None or not isinstance(stream, bytes):
            return stream
        # normalize drivers that return the entire payload as a single bytes object
        return iter([stream])

    @classmethod
    def upload_file(
        cls,
        local_file: str,
        remote_url: str,
        wait_for_upload: bool = True,
        retries: Optional[int] = None,
    ) -> str:
        """
        Upload a local file to a remote location. Supports ``http(s)``, ``s3``, ``gs``, ``azure``, and shared
        filesystem.

        Examples:

        .. code-block:: py

            upload_file('/tmp/artifact.yaml', 'http://localhost:8081/manual_artifacts/my_artifact.yaml')
            upload_file('/tmp/artifact.yaml', 's3://a_bucket/artifacts/my_artifact.yaml')
            upload_file('/tmp/artifact.yaml', '/mnt/share/folder/artifacts/my_artifact.yaml')

        :param local_file: Full path of a local file to be uploaded.
        :param remote_url: Full path or remote URL to upload to (including file name).
        :param wait_for_upload: If ``False``, upload in the background and return immediately.  Defaults to ``True``.
        :param retries: Number of retries before failing to upload file.
        :return: Newly uploaded remote URL.
        """
        return CacheManager.get_cache_manager().upload_file(
            local_file=local_file,
            remote_url=remote_url,
            wait_for_upload=wait_for_upload,
            retries=retries if retries else cls._file_upload_retries,
        )

    @classmethod
    def set_cache_file_limit(cls, cache_file_limit: int, cache_context: Optional[str] = None) -> int:
        """
        Set the maximum number of files the cache context can hold. Note: the limit applies to file count only, not
        total storage size.

        :param cache_file_limit: Maximum number of cached files.
        :param cache_context: Optional cache context identifier, default global context.
        :return: The new cache context file limit.
        """
        return CacheManager.get_cache_manager(
            cache_context=cache_context, cache_file_limit=cache_file_limit
        ).set_cache_limit(cache_file_limit)

    @classmethod
    def _extract_to_cache(
        cls,
        cached_file: str,
        name: str,
        cache_context: Optional[str] = None,
        target_folder: Optional[str] = None,
        cache_path_encoding: Optional[str] = None,
        force: bool = False,
    ) -> str:
        """
        Extract cached file to cache folder.
        :param cached_file: Local copy of archive file.
        :param name: Name of the target file.
        :param cache_context: Cache context ID.
        :param target_folder: Specify target path to use for archive extraction.
        :param cache_path_encoding: Specify representation of the local path of the cached files,
            this will always point to local cache folder, even if we have direct access file.
            Used for extracting the cached archived based on ``cache_path_encoding``.
        :param force: Force archive extraction even if target folder exists.
        :return: Cached folder containing the extracted archive content.
        """
        if not cached_file:
            return cached_file

        cached_file = Path(cached_file)
        cache_path_encoding = Path(cache_path_encoding) if cache_path_encoding else None

        # we support zip and tar.gz files auto-extraction
        suffix = cached_file.suffix.lower()
        if suffix == ".gz":
            suffix = "".join(a.lower() for a in cached_file.suffixes[-2:])

        if suffix not in (".zip", ".tgz", ".tar.gz"):
            return str(cached_file)

        cache_folder = Path(cache_path_encoding or cached_file).parent
        archive_suffix = (cache_path_encoding or cached_file).name[: -len(suffix)]
        name = (
            quote(name, safe=" ")  # Encoding string to filename
            if name
            else name
        )
        target_folder = (
            Path(target_folder)
            if target_folder
            else cache_folder / CacheManager.get_context_folder_lookup(cache_context).format(archive_suffix, name)
        )

        if target_folder.is_dir() and not force:
            # noinspection PyBroadException
            try:
                target_folder.touch(exist_ok=True)
                return target_folder.as_posix()
            except Exception:
                pass

        base_logger = LoggerRoot.get_base_logger()
        try:
            # if target folder exists, meaning this is forced ao we extract directly into target folder
            temp_target_folder = (
                target_folder
                if target_folder.is_dir()
                else cache_folder / f"{target_folder.name}_{time() * 1000}_{str(random()).replace('.', '')}"
            )
            temp_target_folder.mkdir(parents=True, exist_ok=True)

            if suffix == ".zip":
                extract_zip_archive(
                    archive_path=cached_file.as_posix(),
                    target=temp_target_folder.as_posix(),
                )
            elif suffix in (".tar.gz", ".tgz"):
                extract_tar_archive(
                    archive_path=cached_file.as_posix(),
                    target=temp_target_folder.as_posix(),
                    suffix=suffix,
                )

            if temp_target_folder != target_folder:
                # we assume we will have such folder if we already extract the file
                # noinspection PyBroadException
                try:
                    # if rename fails, it means that someone else already manged to extract the file, delete the current
                    # folder and return the already existing cached zip folder
                    shutil.move(temp_target_folder.as_posix(), target_folder.as_posix())
                except Exception:
                    if target_folder.exists():
                        target_folder.touch(exist_ok=True)
                    else:
                        base_logger.warning(
                            f"Failed renaming {temp_target_folder.as_posix()} to {target_folder.as_posix()}"
                        )
                    try:
                        shutil.rmtree(temp_target_folder.as_posix())
                    except Exception as ex:
                        base_logger.warning(
                            f"Exception {ex}\nFailed deleting folder {temp_target_folder.as_posix()}"
                        )
        except Exception as ex:
            # failed extracting the file:
            base_logger.warning(f"Exception {ex}\nFailed extracting zip file {cached_file.as_posix()}")
            # noinspection PyBroadException
            try:
                target_folder.rmdir()
            except Exception:
                pass
            return cached_file.as_posix()
        return target_folder.as_posix()

    @classmethod
    def get_files_server(cls) -> str:
        from ..backend_api import Session

        return Session.get_files_server_host()

    @classmethod
    def upload_folder(
        cls,
        local_folder: str,
        remote_url: str,
        match_wildcard: Optional[str] = None,
        retries: Optional[int] = None,
    ) -> Optional[str]:
        """
        Upload a local folder recursively to remote storage, preserving the subfolder structure.

        For example, uploading ``'~/folder/'`` to ``'s3://bucket/'`` using
        ``StorageManager.upload_folder('~/folder/', 's3://bucket/')`` will copy all contents of the local folder to the
        bucket. If the local folder contains ``~/folder/sub/file.ext``, it will be saved remotely as
        ``s3://bucket/sub/file.ext``.

        :param local_folder: Local folder to recursively upload
        :param remote_url: Target remote storage location. The folder structure of ``local_folder`` will
            be recreated under ``remote_url``. Supports ``http(s)``, ``s3``, ``gs``, ``azure``, and shared filesystem.
            Example: ``'s3://bucket/data/'``.
        :param match_wildcard: If specified, only upload files matching this wildcard pattern.
            Example: ``*.json``.
        :param retries: Number of retries before failing to upload a file in the folder.
        :return: Newly uploaded remote URL or ``None`` on error.
        """

        base_logger = LoggerRoot.get_base_logger()

        if not Path(local_folder).is_dir():
            base_logger.error(
                f"Local folder '{local_folder}' does not exist",
                exc_info=base_logger.isEnabledFor(logging.DEBUG),
            )
            return
        local_folder = str(Path(local_folder))
        results = []
        helper = cls.storage_helper.get(remote_url)
        with ThreadPool() as pool:
            for path in Path(local_folder).rglob(match_wildcard or "*"):
                if not path.is_file():
                    continue
                results.append(
                    pool.apply_async(
                        helper.upload,
                        args=(str(path), str(path).replace(local_folder, remote_url)),
                        kwds={"retries": retries if retries else cls._file_upload_retries},
                    )
                )

            success = 0
            failed = 0
            for res in results:
                # noinspection PyBroadException
                try:
                    res.get()  # Reraise the exceptions from remote call (if any)
                    success += 1
                except Exception:
                    failed += 1

            if failed == 0:
                return remote_url

            base_logger.error(
                f"Failed uploading {failed}/{success + failed} files from {local_folder}",
                exc_info=base_logger.isEnabledFor(logging.DEBUG),
            )

    @classmethod
    def download_file(
        cls,
        remote_url: str,
        local_folder: Optional[str] = None,
        overwrite: bool = False,
        skip_zero_size_check: bool = False,
        silence_errors: bool = False,
    ) -> Optional[str]:
        """
        Download a remote file to a local folder, preserving its subfolder structure.

        For example, to download ``s3://bucket/sub/file.ext``, calling
        ``StorageManager.download_file('s3://bucket/sub/file.ext', '~/folder/')`` will save it locally as
        ``~/folder/sub/file.ext``.

        :param remote_url: URL of the remote file to download.  Its path structure will be recreated under the target
            ``local_folder``. Supports ``s3``, ``gs``, ``azure``, and shared filesystem.
            Example: ``'s3://bucket/data/'``
        :param local_folder: Local target folder. If ``None`` (default), uses the cache folder.
        :param overwrite: If ``True``, download remote files even if they exist locally. Defaults to ``False``.
        :param skip_zero_size_check: If ``True``, no error will be raised for files with zero bytes size. Defaults to
            ``False``.
        :param silence_errors: If ``True``, silence errors encountered during download. Defaults to ``False``.

        :return: Path to downloaded file, or ``None`` on error.
        """

        def remove_prefix_from_str(target_str: str, prefix_to_be_removed: str) -> str:
            if target_str.startswith(prefix_to_be_removed):
                return target_str[len(prefix_to_be_removed) :]
            return target_str

        longest_configured_url = cls.storage_helper._resolve_base_url(remote_url)  # noqa
        bucket_path = remove_prefix_from_str(remote_url[len(longest_configured_url) :], "/")

        if not local_folder:
            local_folder = CacheManager.get_cache_manager().get_cache_folder()
        local_path = str(Path(local_folder).expanduser().absolute() / bucket_path)
        helper = cls.storage_helper.get(remote_url)

        return helper.download_to_file(
            remote_url,
            local_path,
            overwrite_existing=overwrite,
            skip_zero_size_check=skip_zero_size_check,
            silence_errors=silence_errors,
        )

    @classmethod
    def exists_file(cls, remote_url: str) -> bool:
        """
        Check if remote file exists. Returns ``False`` for directories.

        :param remote_url: The URL where the file is stored.
             For example: ``'s3://bucket/some_file.txt'``, ``'file://local/file'``

        :return: ``True`` if the ``remote_url`` stores a file. ``False`` otherwise.
        """
        # noinspection PyBroadException
        try:
            if remote_url.endswith("/"):
                return False
            helper = cls.storage_helper.get(remote_url)
            return helper.exists_file(remote_url)
        except Exception:
            return False

    @classmethod
    def get_file_size_bytes(cls, remote_url: str, silence_errors: bool = False) -> Optional[int]:
        """
        Get size of the remote file in bytes.

        :param remote_url: The URL where the file is stored.
            For example: ``'s3://bucket/some_file.txt'``, ``'file://local/file'``
        :param silence_errors: If ``True``, silence errors encountered while fetching the size of the file.
            Default: ``False``

        :return: The size of the file in bytes.
            ``None`` if the file could not be found or an error occurred.
        """
        helper = cls.storage_helper.get(remote_url)
        return helper.get_object_size_bytes(remote_url, silence_errors)

    @classmethod
    def download_folder(
        cls,
        remote_url: str,
        local_folder: Optional[str] = None,
        match_wildcard: Optional[str] = None,
        overwrite: bool = False,
        skip_zero_size_check: bool = False,
        silence_errors: bool = False,
        max_workers: Optional[int] = None,
    ) -> Optional[str]:
        """
        Download a remote folder recursively to the local machine, preserving the subfolder structure.

        For example, downloading ``'s3://bucket/'`` to ``'~/folder/'`` using
        ``StorageManager.download_folder('s3://bucket/', '~/folder/')`` will copy all contents of the bucket into
        ``~/folder/``. If the remote contains: ``s3://bucket/sub/file.ext``, it will be saved locally as:
        ``~/folder/sub/file.ext``.

        :param str remote_url: Source remote storage location, tree structure of ``remote_url`` will
            be created under the target ``local_folder``. Supports ``s3``, ``gs``, ``azure``, and shared filesystem.
            Example: ``'s3://bucket/data/'``
        :param local_folder: Local target folder to create the full tree from ``remote_url``.
            If ``None`` (default), use the cache folder.
        :param match_wildcard: If specified, only download files matching this wildcard pattern.
            Example: ``*.json``
        :param overwrite: If ``True``, download remote files even if they exist locally. Defaults to ``False``.
        :param skip_zero_size_check: If ``True``, no error will be raised for files with zero bytes size. Defaults to
            ``False``.
        :param silence_errors: If ``True``, silence errors encountered during download. Defaults to ``False``.
        :param max_workers: Number of worker threads for parallel downloads. If ``None`` (default), uses the number
             of logical CPU cores in the system (default Python behavior).

        :return: Target local folder
        """

        base_logger = LoggerRoot.get_base_logger()

        if local_folder:
            try:
                Path(local_folder).mkdir(parents=True, exist_ok=True)
            except OSError as ex:
                base_logger.error(
                    f"Failed creating local folder '{local_folder}': {ex}",
                    exc_info=base_logger.isEnabledFor(logging.DEBUG),
                )
                return
        else:
            local_folder = CacheManager.get_cache_manager().get_cache_folder()

        helper = cls.storage_helper.get(remote_url)
        results = []

        with ThreadPool(processes=max_workers) as pool:
            for path in helper.list(prefix=remote_url):
                remote_path = (
                    str(Path(helper.base_url) / Path(path))
                    if helper.get_driver_direct_access(helper.base_url)
                    else f"{helper.base_url.rstrip('/')}/{path.lstrip('/')}"
                )
                if match_wildcard and not fnmatch.fnmatch(remote_path, match_wildcard):
                    continue
                results.append(
                    pool.apply_async(
                        cls.download_file,
                        args=(remote_path, local_folder),
                        kwds={
                            "overwrite": overwrite,
                            "skip_zero_size_check": skip_zero_size_check,
                            "silence_errors": silence_errors,
                        },
                    )
                )
            for res in results:
                res.wait()
        if not results and not silence_errors:
            LoggerRoot.get_base_logger().warning(f"Did not download any files matching {remote_url}")
        return local_folder

    @classmethod
    def list(
        cls,
        remote_url: str,
        return_full_path: bool = False,
        with_metadata: bool = False,
        read_hash: bool = False,
    ) -> Optional[List[Union[str, dict]]]:
        """
        Return a list of object names inside the base path or dictionaries containing the corresponding
        objects' metadata (in case ``with_metadata`` is ``True``).

        :param remote_url: The base path.
            For Google Storage, Azure and S3 it is the bucket of the path, for local files it is the root directory.
            For example: AWS S3: ``s3://bucket/folder_`` will list all the files you have in
            ``s3://bucket-name/folder_*/*``. The same behavior with Google Storage: ``gs://bucket/folder_``,
            Azure blob storage: ``azure://bucket/folder_`` and also file system listing: ``/mnt/share/folder_``
        :param return_full_path: If ``True``, return a list of full object paths instead of relative paths.
            Defaults to
            ``False``.
        :param with_metadata: If ``True``, return a list of dicts containing name and size instead of just names.
            Defaults to ``False``.
        :param read_hash: If ``True`` and ``with_metadata=True``, include SHA-256 hash in each metadata dict
            when the object has it stored in its custom metadata.

        :return: A list of object paths relative to the base path, or a list of the objects' metadata dicts if
            ``with_metadata=True``. Returns ``None`` if the list operation is not supported (e.g. HTTP/HTTPS protocols).
        """
        helper = cls.storage_helper.get(remote_url)
        try:
            helper_list_result = helper.list(
                prefix=remote_url,
                with_metadata=with_metadata,
                read_hash=read_hash,
            )
        except Exception as ex:
            LoggerRoot.get_base_logger().warning(f"Can not list files for '{remote_url}' - {ex}")
            return None

        prefix = remote_url.rstrip("/") if helper.base_url == "file://" else helper.base_url
        if not with_metadata:
            return (
                [f"{prefix}/{name}" for name in helper_list_result]
                if return_full_path
                else helper_list_result
            )
        else:
            if return_full_path:
                for obj in helper_list_result:
                    obj["name"] = f"{prefix}/{obj.get('name')}"
            return helper_list_result

    @classmethod
    def get_metadata(
        cls,
        remote_url: str,
        return_full_path: bool = False,
        read_hash: bool = False,
    ) -> Optional[dict]:
        """
        Get the metadata of the remote object.
        The metadata is a dict containing the following keys: ``name``, ``size``.

        :param remote_url:  URL of the remote object to retrieve metadata for. Supports ``s3``, ``gs``, ``azure``,
            shared filesystem, and ``http(s)``. Example: ``'s3://bucket/data/file.txt'``
        :param return_full_path: If ``True``, the ``name`` field in the returned dict will include the full URL
            including the base. Defaults to ``False``.
        :param read_hash: If ``True``, include SHA-256 hash in the returned dict when the object
            has it stored in its custom metadata.

        :return: A dict containing the metadata of the remote object. ``None``  in case of an error.
        """
        helper = cls.storage_helper.get(remote_url)
        obj = helper.get_object(remote_url)
        if not obj:
            return None
        metadata = helper.get_object_metadata(obj, read_hash=read_hash)
        base_url = helper._resolve_base_url(remote_url)
        if return_full_path and not metadata["name"].startswith(base_url):
            metadata["name"] = base_url + ("/" if not base_url.endswith("/") else "") + metadata["name"]
        return metadata

    @classmethod
    def set_report_upload_chunk_size(cls, chunk_size_mb: int) -> None:
        """
        Set the upload progress report chunk size (in MB). The chunk size
        determines how often the progress reports are logged:
        every time a chunk of data with a size greater than ``chunk_size_mb``
        is uploaded, log the report.
        This function overrides the ``sdk.storage.log.report_upload_chunk_size_mb``
        configuration value.

        :param chunk_size_mb: The chunk size in megabytes
        """
        ProgressReport.report_upload_chunk_size_mb = int(chunk_size_mb)

    @classmethod
    def set_report_download_chunk_size(cls, chunk_size_mb: int) -> None:
        """
        Set the download progress report chunk size (in MB). The chunk size
        determines how often the progress reports are logged:
        every time a chunk of data with a size greater than ``chunk_size_mb``
        is downloaded, log the report.
        This function overwrites the ``sdk.storage.log.report_download_chunk_size_mb``
        config entry

        :param chunk_size_mb: The chunk size in megabytes
        """
        ProgressReport.report_download_chunk_size_mb = int(chunk_size_mb)


class StorageManagerDiskSpaceFileSizeStrategy(StorageManager):
    storage_helper = StorageHelperDiskSpaceFileSizeStrategy
