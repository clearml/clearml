from __future__ import with_statement

import errno
import getpass
import itertools
import json
import logging
import mimetypes
import hashlib
import os
import platform
import shutil
import sys
import threading
import uuid
from _socket import gethostname
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError, CancelledError
from copy import copy
from datetime import datetime, timezone
from multiprocessing.pool import ThreadPool, AsyncResult
from multiprocessing import Lock
from tempfile import mkstemp
from time import time
from types import GeneratorType
from typing import (
    Optional,
    List,
    Dict,
    Iterator,
    Callable,
    Iterable,
    Generator,
    Union,
    Tuple,
    TYPE_CHECKING,
    Any,
)

import numpy
import requests
from urllib.request import url2pathname
from attr import attrs, attrib, asdict

if TYPE_CHECKING:
    from azure.storage.blob import ContentSettings

from furl import furl
from pathlib2 import Path
from requests import codes as requests_codes
from requests.exceptions import ConnectionError
from io import StringIO, BytesIO, FileIO
from queue import Queue, Empty
from urllib.parse import urlparse, quote
from os.path import expandvars, expanduser
from heapq import heapify, heappush, heappop
from psutil import disk_usage

from clearml.utilities.requests_toolbelt import (
    MultipartEncoderMonitor,
    MultipartEncoder,
)
from .callbacks import UploadProgressReport, DownloadProgressReport
from .util import quote_url
from ..backend_api.session import Session
from ..backend_api.utils import get_http_session_with_retry
from ..backend_config.bucket_config import (
    S3BucketConfigurations,
    GSBucketConfigurations,
    AzureContainerConfigurations,
    BucketConfig,
    AzureContainerConfig,
    S3BucketConfig,
)
from .util import format_size
from ..config import config, deferred_config, SESSION_CACHE_FILE, CLEARML_CACHE_DIR, DEFAULT_CACHE_DIR
from ..debugging import get_logger
from ..errors import UsageError
from ..utilities.process.mp import ForkSafeRLock, SafeEvent

from ..storage.util import get_config_object_matcher
from ..utilities.config import get_percentage, get_human_size_default
from ..backend_config.environment import EnvEntry


class StorageError(Exception):
    pass


class DownloadError(Exception):
    pass


class _Driver(metaclass=ABCMeta):
    _certs_cache_context = "certs"
    _file_server_hosts = None

    @classmethod
    def get_logger(cls) -> logging.Logger:
        return get_logger("storage")

    @abstractmethod
    def get_container(
        self,
        container_name: str,
        config: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        pass

    @abstractmethod
    def test_upload(self, test_path: str, config: Any, **kwargs: Any) -> bool:
        pass

    @abstractmethod
    def upload_object_via_stream(
        self,
        iterator: Any,
        container: Any,
        object_name: str,
        extra: dict,
        **kwargs: Any,
    ) -> Any:
        pass

    @abstractmethod
    def list_container_objects(
        self,
        container: Any,
        ex_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        pass

    @abstractmethod
    def get_direct_access(self, remote_path: str, **kwargs: Any) -> Optional[str]:
        pass

    @abstractmethod
    def download_object(
        self,
        obj: Any,
        local_path: str,
        overwrite_existing: bool,
        delete_on_failure: bool,
        callback: Any,
        **kwargs: Any,
    ) -> bool:
        pass

    @abstractmethod
    def download_object_as_stream(self, obj: Any, chunk_size: int, **kwargs: Any) -> GeneratorType:
        pass

    @abstractmethod
    def delete_object(self, obj: Any, **kwargs: Any) -> bool:
        pass

    @abstractmethod
    def upload_object(
        self,
        file_path: str,
        container: Any,
        object_name: str,
        extra: dict,
        **kwargs: Any,
    ) -> Any:
        pass

    @abstractmethod
    def get_object(self, container_name: str, object_name: str, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def exists_file(self, container_name: str, object_name: str) -> bool:
        pass

    @classmethod
    def get_file_server_hosts(cls) -> List[str]:
        if cls._file_server_hosts is None:
            hosts = [Session.get_files_server_host()] + (Session.legacy_file_servers or [])
            for host in hosts[:]:
                substituted = _StorageHelper._apply_url_substitutions(host)
                if substituted not in hosts:
                    hosts.append(substituted)
            cls._file_server_hosts = hosts
        return cls._file_server_hosts

    @classmethod
    def download_cert(cls, cert_url: str) -> str:
        # import here to avoid circular imports
        from .manager import StorageManager

        cls.get_logger().info("Attempting to download remote certificate '{}'".format(cert_url))
        potential_exception = None
        downloaded_verify = None
        try:
            downloaded_verify = StorageManager.get_local_copy(cert_url, cache_context=cls._certs_cache_context)
        except Exception as e:
            potential_exception = e
        if not downloaded_verify:
            cls.get_logger().error(
                "Failed downloading remote certificate '{}'{}".format(
                    cert_url,
                    "Error is: {}".format(potential_exception) if potential_exception else "",
                )
            )
        else:
            cls.get_logger().info("Successfully downloaded remote certificate '{}'".format(cert_url))
        return downloaded_verify


class _HttpDriver(_Driver):
    """LibCloud http/https adapter (simple, enough for now)"""

    timeout_connection = deferred_config("http.timeout.connection", 30)
    timeout_total = deferred_config("http.timeout.total", 30)
    max_retries = deferred_config("http.download.max_retries", 15)
    min_kbps_speed = 50

    schemes = ("http", "https")

    class _Container(object):
        _default_backend_session = None

        def __init__(self, name: str, retries: int = 5, **kwargs: Any) -> None:
            self.name = name
            self.session = get_http_session_with_retry(
                total=retries,
                connect=retries,
                read=retries,
                redirect=retries,
                backoff_factor=0.5,
                backoff_max=120,
                status_forcelist=[
                    requests_codes.request_timeout,
                    requests_codes.timeout,
                    requests_codes.bad_gateway,
                    requests_codes.service_unavailable,
                    requests_codes.bandwidth_limit_exceeded,
                    requests_codes.too_many_requests,
                ],
                config=config,
            )
            self._file_server_hosts = set(_HttpDriver.get_file_server_hosts())

        def _should_attach_auth_header(self) -> bool:
            return any(
                (self.name.rstrip("/") == host.rstrip("/") or self.name.startswith(host.rstrip("/") + "/"))
                for host in self._file_server_hosts
            )

        def get_headers(self, _: Any) -> Dict[str, str]:
            if not self._default_backend_session:
                from ..backend_interface.base import InterfaceBase

                self._default_backend_session = InterfaceBase._get_default_session()

            if self._should_attach_auth_header():
                return self._default_backend_session.add_auth_headers({})

    class _HttpSessionHandle(object):
        def __init__(
            self,
            url: str,
            is_stream: bool,
            container_name: str,
            object_name: str,
        ) -> None:
            self.url, self.is_stream, self.container_name, self.object_name = (
                url,
                is_stream,
                container_name,
                object_name,
            )

    def __init__(self, retries: Optional[int] = None) -> None:
        self._retries = retries or int(self.max_retries)
        self._containers = {}

    def get_container(
        self,
        container_name: str,
        config: Optional[Any] = None,
        **kwargs: Any,
    ) -> _Container:
        if container_name not in self._containers:
            self._containers[container_name] = self._Container(name=container_name, retries=self._retries, **kwargs)
        return self._containers[container_name]

    def upload_object_via_stream(
        self,
        iterator: Any,
        container: Any,
        object_name: str,
        extra: dict = None,
        callback: Any = None,
        **kwargs: Any,
    ) -> requests.Response:
        def monitor_callback(monitor: Any) -> None:
            new_chunk = monitor.bytes_read - monitor.previous_read
            monitor.previous_read = monitor.bytes_read
            try:
                callback(new_chunk)
            except Exception as ex:
                self.get_logger().debug("Exception raised when running callback function: {}".format(ex))

        # when sending data in post, there is no connection timeout, just an entire upload timeout
        timeout = int(self.timeout_total)
        url = container.name
        path = object_name
        if not urlparse(url).netloc:
            host, _, path = object_name.partition("/")
            url += host + "/"

        stream_size = None
        if hasattr(iterator, "tell") and hasattr(iterator, "seek"):
            pos = iterator.tell()
            iterator.seek(0, 2)
            stream_size = iterator.tell() - pos
            iterator.seek(pos, 0)
            timeout = max(timeout, (stream_size / 1024) / float(self.min_kbps_speed))

        m = MultipartEncoder(fields={path: (path, iterator, get_file_mimetype(object_name))})
        if callback and stream_size:
            m = MultipartEncoderMonitor(m, callback=monitor_callback)
            m.previous_read = 0

        headers = {
            "Content-Type": m.content_type,
        }
        headers.update(container.get_headers(url) or {})

        res = container.session.post(url, data=m, timeout=timeout, headers=headers)
        if res.status_code != requests.codes.ok:
            raise ValueError("Failed uploading object {} to {} ({}): {}".format(
                object_name, url, res.status_code, res.text))

        # call back is useless because we are not calling it while uploading...
        return res

    def list_container_objects(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("List is not implemented for http protocol")

    def delete_object(self, obj: _HttpSessionHandle, *args: Any, **kwargs: Any) -> bool:
        assert isinstance(obj, self._HttpSessionHandle)
        container = self._containers[obj.container_name]
        res = container.session.delete(obj.url, headers=container.get_headers(obj.url))
        if res.status_code != requests.codes.ok:
            if not kwargs.get("silent", False):
                self.get_logger().warning(
                    "Failed deleting object %s (%d): %s" % (obj.object_name, res.status_code, res.text)
                )
            return False
        return True

    def get_object(
        self,
        container_name: str,
        object_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> _HttpSessionHandle:
        is_stream = kwargs.get("stream", True)
        url = "/".join(
            (
                container_name[:-1] if container_name.endswith("/") else container_name,
                object_name.lstrip("/"),
            )
        )
        return self._HttpSessionHandle(url, is_stream, container_name, object_name)

    def _get_download_object(self, obj: Any) -> Any:
        # bypass for session result
        if not isinstance(obj, self._HttpSessionHandle):
            return obj

        container = self._containers[obj.container_name]
        # set stream flag before we send the request
        container.session.stream = obj.is_stream
        res = container.session.get(
            obj.url,
            timeout=(int(self.timeout_connection), int(self.timeout_total)),
            headers=container.get_headers(obj.url),
        )
        if res.status_code != requests.codes.ok:
            raise ValueError("Failed getting object %s (%d): %s" % (obj.object_name, res.status_code, res.reason))
        return res

    def download_object_as_stream(self, obj: Any, chunk_size: int = 64 * 1024, **_: Any) -> Iterable[bytes]:
        # return iterable object
        obj = self._get_download_object(obj)
        return obj.iter_content(chunk_size=chunk_size)

    def download_object(
        self,
        obj: Any,
        local_path: str,
        overwrite_existing: bool = True,
        delete_on_failure: bool = True,
        callback: Callable[[int], None] = None,
        **_: Any,
    ) -> int:
        obj = self._get_download_object(obj)
        p = Path(local_path)
        if not overwrite_existing and p.is_file():
            self.get_logger().warning("failed saving after download: overwrite=False and file exists (%s)" % str(p))
            return
        length = 0
        with p.open(mode="wb") as f:
            for chunk in obj.iter_content(chunk_size=5 * 1024 * 1024):
                # filter out keep-alive new chunks
                if not chunk:
                    continue
                chunk_size = len(chunk)
                f.write(chunk)
                length += chunk_size
                if callback:
                    callback(chunk_size)

        return length

    def get_direct_access(self, remote_path: str, **_: Any) -> None:
        return None

    def test_upload(self, test_path: str, config: Any, **kwargs: Any) -> bool:
        return True

    def upload_object(
        self,
        file_path: str,
        container: Any,
        object_name: str,
        extra: dict,
        callback: Optional[Callable] = None,
        **kwargs: Any,
    ) -> Any:
        with open(file_path, "rb") as stream:
            return self.upload_object_via_stream(
                iterator=stream,
                container=container,
                object_name=object_name,
                extra=extra,
                callback=callback,
                **kwargs,
            )

    def exists_file(self, container_name: str, object_name: str) -> bool:
        # noinspection PyBroadException
        try:
            container = self.get_container(container_name)
            url = container_name + object_name
            return container.session.head(url, allow_redirects=True, headers=container.get_headers(url)).ok
        except Exception:
            return False


class _Stream(object):
    encoding = None
    mode = "rw"
    name = ""
    newlines = "\n"
    softspace = False

    def __init__(self, input_iterator: Optional[Iterator[Any]] = None) -> None:
        self.closed = False
        self._buffer = Queue()
        self._input_iterator = input_iterator
        self._leftover = None

    def __iter__(self) -> Any:
        return self

    def __next__(self) -> Any:
        return self.next()

    def close(self) -> None:
        self.closed = True

    def flush(self) -> None:
        pass

    def fileno(self) -> int:
        return 87

    def isatty(self) -> bool:
        return False

    def next(self) -> bytes:
        while not self.closed or not self._buffer.empty():
            # input stream
            if self._input_iterator:
                try:
                    chunck = next(self._input_iterator)
                    # make sure we always return bytes
                    if isinstance(chunck, str):
                        chunck = chunck.encode("utf-8")
                    return chunck
                except StopIteration:
                    self.closed = True
                    raise StopIteration()
                except Exception as ex:
                    _Driver.get_logger().error("Failed downloading: %s" % ex)
            else:
                # in/out stream
                try:
                    return self._buffer.get(block=True, timeout=1.0)
                except Empty:
                    pass

        raise StopIteration()

    def read(self, size: Optional[int] = None) -> bytes:
        try:
            data = self.next() if self._leftover is None else self._leftover
        except StopIteration:
            return b""

        self._leftover = None
        try:
            while size is None or not data or len(data) < size:
                chunk = self.next()
                if chunk is not None:
                    if data is not None:
                        data += chunk
                    else:
                        data = chunk
        except StopIteration:
            pass

        if size is not None and data and len(data) > size:
            self._leftover = data[size:]
            return data[:size]

        return data

    def readline(self, size: Optional[int] = None) -> str:
        return self.read(size)

    def readlines(self, sizehint: Optional[int] = None) -> List[str]:
        pass

    def truncate(self, size: Optional[int] = None) -> None:
        pass

    def write(self, bytes: bytes) -> None:
        self._buffer.put(bytes, block=True)

    def writelines(self, sequence: Iterable[str]) -> None:
        for s in sequence:
            self.write(s)


class _Boto3Driver(_Driver):
    """Boto3 storage adapter (simple, enough for now)"""

    _min_pool_connections = 512
    _max_multipart_concurrency = deferred_config("aws.boto3.max_multipart_concurrency", 16)
    _multipart_threshold = deferred_config("aws.boto3.multipart_threshold", (1024**2) * 8)  # 8 MB
    _multipart_chunksize = deferred_config("aws.boto3.multipart_chunksize", (1024**2) * 8)
    _pool_connections = deferred_config("aws.boto3.pool_connections", 512)
    _connect_timeout = deferred_config("aws.boto3.connect_timeout", 60)
    _read_timeout = deferred_config("aws.boto3.read_timeout", 60)
    _signature_version = deferred_config("aws.boto3.signature_version", None)

    _stream_download_pool_connections = deferred_config("aws.boto3.stream_connections", 128)
    _stream_download_pool = None
    _stream_download_pool_pid = None

    _containers = {}

    scheme = "s3"
    scheme_prefix = str(furl(scheme=scheme, netloc=""))

    _bucket_location_failure_reported = set()

    class _Container(object):
        _creation_lock = ForkSafeRLock()

        def __init__(self, name: str, cfg: S3BucketConfig) -> None:
            try:
                import boto3
            except ImportError:
                raise UsageError(
                    "AWS S3 storage driver (boto3) not found. " 'Please install driver using: pip install "clearml[s3]"'
                )
            # skip 's3://'
            self.name = name[5:]

            # boto3 client creation isn't thread-safe (client itself is)
            with self._creation_lock:
                boto_kwargs = _Boto3Driver._get_boto_resource_kwargs_from_config(cfg)
                boto_session = boto3.Session(
                    profile_name=cfg.profile or None,
                )
                self.resource = boto_session.resource("s3", **boto_kwargs)

                self.config = cfg
                bucket_name = self.name[len(cfg.host) + 1 :] if cfg.host else self.name
                self.bucket = self.resource.Bucket(bucket_name)

    @attrs
    class ListResult(object):
        name = attrib(default=None)
        size = attrib(default=None)

    def __init__(self) -> None:
        pass

    def _get_stream_download_pool(self) -> ThreadPoolExecutor:
        if self._stream_download_pool is None or self._stream_download_pool_pid != os.getpid():
            self._stream_download_pool_pid = os.getpid()
            self._stream_download_pool = ThreadPoolExecutor(max_workers=int(self._stream_download_pool_connections))
        return self._stream_download_pool

    @classmethod
    def _get_boto_resource_kwargs_from_config(cls, cfg: S3BucketConfig) -> Dict[str, Any]:
        try:
            import botocore.client
        except ImportError:
            raise UsageError(
                "AWS S3 storage driver (boto3) not found. " 'Please install driver using: pip install "clearml[s3]"'
            )
        endpoint = (("https://" if cfg.secure else "http://") + cfg.host) if cfg.host else None
        verify = cfg.verify
        if verify is True:
            # True is a non-documented value for boto3, use None instead (which means verify)
            verify = None
        elif isinstance(verify, str) and not os.path.exists(verify) and verify.split("://")[0] in driver_schemes:
            verify = _Boto3Driver.download_cert(verify)
        boto_kwargs = {
            "endpoint_url": endpoint,
            "use_ssl": cfg.secure,
            "verify": verify,
            "region_name": cfg.region or None,  # None in case cfg.region is an empty string
            "config": botocore.client.Config(
                max_pool_connections=max(
                    int(_Boto3Driver._min_pool_connections),
                    int(_Boto3Driver._pool_connections),
                ),
                connect_timeout=int(_Boto3Driver._connect_timeout),
                read_timeout=int(_Boto3Driver._read_timeout),
                signature_version=_Boto3Driver._signature_version,
            ),
        }
        if not cfg.use_credentials_chain:
            boto_kwargs["aws_access_key_id"] = cfg.key or None
            boto_kwargs["aws_secret_access_key"] = cfg.secret or None
            if cfg.token:
                boto_kwargs["aws_session_token"] = cfg.token
        return boto_kwargs

    def get_container(
        self,
        container_name: str,
        config: Optional[Any] = None,
        **kwargs: Any,
    ) -> _Container:
        if container_name not in self._containers:
            self._containers[container_name] = self._Container(name=container_name, cfg=config)
        self._containers[container_name].config.retries = kwargs.get("retries", 5)
        return self._containers[container_name]

    def upload_object_via_stream(
        self,
        iterator: Any,
        container: Any,
        object_name: str,
        callback: Any = None,
        extra: dict = None,
        **kwargs: Any,
    ) -> bool:
        import boto3.s3.transfer

        stream = _Stream(iterator)
        extra_args = {}
        try:
            extra_args = {"ContentType": get_file_mimetype(object_name)}
            extra_args.update(container.config.extra_args or {})
            container.bucket.upload_fileobj(
                stream,
                object_name,
                Config=boto3.s3.transfer.TransferConfig(
                    use_threads=container.config.multipart,
                    max_concurrency=int(self._max_multipart_concurrency) if container.config.multipart else 1,
                    num_download_attempts=container.config.retries,
                    multipart_threshold=int(self._multipart_threshold),
                    multipart_chunksize=int(self._multipart_chunksize),
                ),
                Callback=callback,
                ExtraArgs=extra_args,
            )
        except RuntimeError:
            # one might get an error similar to: "RuntimeError: cannot schedule new futures after interpreter shutdown"
            # In this case, retry the upload without threads
            try:
                container.bucket.upload_fileobj(
                    stream,
                    object_name,
                    Config=boto3.s3.transfer.TransferConfig(
                        use_threads=False,
                        num_download_attempts=container.config.retries,
                        multipart_threshold=int(self._multipart_threshold),
                        multipart_chunksize=int(self._multipart_chunksize),
                    ),
                    Callback=callback,
                    ExtraArgs=extra_args,
                )
            except Exception as ex:
                self.get_logger().error("Failed uploading: %s" % ex)
                return False
        except Exception as ex:
            self.get_logger().error("Failed uploading: %s" % ex)
            return False
        return True

    def upload_object(
        self,
        file_path: str,
        container: Any,
        object_name: str,
        callback: Any = None,
        extra: dict = None,
        **kwargs: Any,
    ) -> bool:
        import boto3.s3.transfer

        extra_args = {}
        try:
            extra_args = {"ContentType": get_file_mimetype(object_name or file_path)}
            extra_args.update(container.config.extra_args or {})
            container.bucket.upload_file(
                file_path,
                object_name,
                Config=boto3.s3.transfer.TransferConfig(
                    use_threads=container.config.multipart,
                    max_concurrency=int(self._max_multipart_concurrency) if container.config.multipart else 1,
                    num_download_attempts=container.config.retries,
                    multipart_threshold=int(self._multipart_threshold),
                    multipart_chunksize=int(self._multipart_chunksize),
                ),
                Callback=callback,
                ExtraArgs=extra_args,
            )
        except RuntimeError:
            # one might get an error similar to: "RuntimeError: cannot schedule new futures after interpreter shutdown"
            # In this case, retry the upload without threads
            try:
                container.bucket.upload_file(
                    file_path,
                    object_name,
                    Config=boto3.s3.transfer.TransferConfig(
                        use_threads=False,
                        num_download_attempts=container.config.retries,
                        multipart_threshold=int(self._multipart_threshold),
                        multipart_chunksize=int(self._multipart_chunksize),
                    ),
                    Callback=callback,
                    ExtraArgs=extra_args,
                )
            except Exception as ex:
                self.get_logger().error("Failed uploading: %s" % ex)
                return False
        except Exception as ex:
            self.get_logger().error("Failed uploading: %s" % ex)
            return False
        return True

    def list_container_objects(
        self,
        container: Any,
        ex_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> Generator[ListResult, None, None]:
        if ex_prefix:
            res = container.bucket.objects.filter(Prefix=ex_prefix)
        else:
            res = container.bucket.objects.all()
        for res in res:
            yield self.ListResult(name=res.key, size=res.size)

    def delete_object(self, object: Any, **kwargs: Any) -> bool:
        from botocore.exceptions import ClientError

        object.delete()
        try:
            # Try loading the file to verify deletion
            object.load()
            return False
        except ClientError as e:
            return int(e.response["Error"]["Code"]) == 404

    def get_object(
        self,
        container_name: str,
        object_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        full_container_name = "s3://" + container_name
        container = self._containers[full_container_name]
        obj = container.resource.Object(container.bucket.name, object_name)
        obj.container_name = full_container_name
        return obj

    def download_object_as_stream(
        self,
        obj: Any,
        chunk_size: int = 64 * 1024,
        verbose: bool = None,
        log: Any = None,
        **_: Any,
    ) -> _Stream:
        def async_download(a_obj: Any, a_stream: Any, cb: Any, cfg: Any) -> None:
            try:
                a_obj.download_fileobj(a_stream, Callback=cb, Config=cfg)
                if cb:
                    cb.close(report_completed=True)
            except Exception as ex:
                if cb:
                    cb.close()
                (log or self.get_logger()).error("Failed downloading: %s" % ex)
            a_stream.close()

        import boto3.s3.transfer

        # return iterable object
        stream = _Stream()
        container = self._containers[obj.container_name]
        config = boto3.s3.transfer.TransferConfig(
            use_threads=container.config.multipart,
            max_concurrency=int(self._max_multipart_concurrency) if container.config.multipart else 1,
            num_download_attempts=container.config.retries,
            multipart_threshold=int(self._multipart_threshold),
            multipart_chunksize=int(self._multipart_chunksize),
        )
        total_size_mb = obj.content_length / (1024.0 * 1024.0)
        remote_path = os.path.join(obj.container_name, obj.key)
        cb = DownloadProgressReport(total_size_mb, verbose, remote_path, log)
        self._get_stream_download_pool().submit(async_download, obj, stream, cb, config)

        return stream

    def download_object(
        self,
        obj: Any,
        local_path: str,
        overwrite_existing: bool = True,
        delete_on_failure: bool = True,
        callback: Any = None,
        **_: Any,
    ) -> None:
        import boto3.s3.transfer

        p = Path(local_path)
        if not overwrite_existing and p.is_file():
            self.get_logger().warning("failed saving after download: overwrite=False and file exists (%s)" % str(p))
            return
        container = self._containers[obj.container_name]
        Config = boto3.s3.transfer.TransferConfig(
            use_threads=container.config.multipart,
            max_concurrency=int(self._max_multipart_concurrency) if container.config.multipart else 1,
            num_download_attempts=container.config.retries,
            multipart_threshold=int(self._multipart_threshold),
            multipart_chunksize=int(self._multipart_chunksize),
        )
        obj.download_file(str(p), Callback=callback, Config=Config)

    @classmethod
    def _test_bucket_config(
        cls,
        conf: S3BucketConfig,
        log: logging.Logger,
        test_path: str = "",
        raise_on_error: bool = True,
        log_on_error: bool = True,
    ) -> bool:
        try:
            import boto3
            from botocore.exceptions import ClientError
        except ImportError:
            return False

        if not conf.bucket:
            return False
        try:
            if not conf.is_valid():
                raise Exception("Missing credentials")

            fullname = furl(conf.bucket).add(path=test_path).add(path="%s-upload_test" % cls.__module__)
            bucket_name = str(fullname.path.segments[0])
            filename = str(furl(path=fullname.path.segments[1:]))
            if conf.subdir:
                filename = "{}/{}".format(conf.subdir, filename)

            data = {
                "user": getpass.getuser(),
                "machine": gethostname(),
                "time": datetime.now(timezone.utc).isoformat(),
            }

            boto_session = boto3.Session(
                profile_name=conf.profile or None,
            )
            boto_kwargs = _Boto3Driver._get_boto_resource_kwargs_from_config(conf)
            boto_resource = boto_session.resource("s3", **boto_kwargs)
            bucket = boto_resource.Bucket(bucket_name)
            bucket.put_object(Key=filename, Body=json.dumps(data).encode("utf-8"))

            region = cls._get_bucket_region(conf=conf, log=log, report_info=True)
            if region and ((conf.region and region != conf.region) or (not conf.region and region != "us-east-1")):
                msg = "incorrect region specified for bucket %s (detected region %s)" % (conf.bucket, region)
            else:
                return True

        except ClientError as ex:
            msg = ex.response["Error"]["Message"]
            if log_on_error and log:
                log.error(msg)

            if raise_on_error:
                raise

        except Exception as ex:
            msg = str(ex)
            if log_on_error and log:
                log.error(msg)

            if raise_on_error:
                raise

        msg = ("Failed testing access to bucket %s: " % conf.bucket) + msg

        if log_on_error and log:
            log.error(msg)

        if raise_on_error:
            raise StorageError(msg)

        return False

    @classmethod
    def _get_bucket_region(
        cls,
        conf: S3BucketConfigurations,
        log: logging.Logger = None,
        report_info: bool = False,
    ) -> str:
        import boto3
        from botocore.exceptions import ClientError

        if not conf.bucket:
            return None

        def report(msg: str) -> None:
            if log and conf.get_bucket_host() not in cls._bucket_location_failure_reported:
                if report_info:
                    log.debug(msg)
                else:
                    log.warning(msg)
                cls._bucket_location_failure_reported.add(conf.get_bucket_host())

        try:
            boto_session = boto3.Session(
                profile_name=conf.profile_name or None,
            )
            boto_kwargs = _Boto3Driver._get_boto_resource_kwargs_from_config(conf)
            boto_kwargs.pop("region_name", None)
            boto_resource = boto_session.resource("s3", **boto_kwargs)
            return boto_resource.meta.client.get_bucket_location(Bucket=conf.bucket)["LocationConstraint"]
        except ClientError as ex:
            report(
                "Failed getting bucket location (region) for bucket "
                "%s: %s (%s, access_key=%s). Default region will be used. "
                "This is normal if you do not have GET_BUCKET_LOCATION permission"
                % (
                    conf.bucket,
                    ex.response["Error"]["Message"],
                    ex.response["Error"]["Code"],
                    conf.key,
                )
            )
        except Exception as ex:
            report(
                "Failed getting bucket location (region) for bucket %s: %s. Default region will be used."
                % (conf.bucket, str(ex))
            )

        return None

    def get_direct_access(self, remote_path: str, **_: Any) -> None:
        return None

    def test_upload(self, test_path: str, config: Any, **_: Any) -> bool:
        return True

    def exists_file(self, container_name: str, object_name: str) -> bool:
        obj = self.get_object(container_name, object_name)
        # noinspection PyBroadException
        try:
            obj.load()
        except Exception:
            return False
        return bool(obj)


class _GoogleCloudStorageDriver(_Driver):
    """Storage driver for google cloud storage"""

    _stream_download_pool_connections = deferred_config("google.storage.stream_connections", 128)
    _stream_download_pool = None
    _stream_download_pool_pid = None

    _containers = {}

    scheme = "gs"
    scheme_prefix = str(furl(scheme=scheme, netloc=""))

    class _Container(object):
        def __init__(self, name: str, cfg: Any) -> None:
            try:
                from google.cloud import storage  # noqa
                from google.oauth2 import service_account  # noqa
            except ImportError:
                raise UsageError(
                    "Google cloud driver not found. "
                    'Please install driver using: pip install "google-cloud-storage>=1.13.2"'
                )

            self.name = name[len(_GoogleCloudStorageDriver.scheme_prefix) :]

            if cfg.credentials_json:
                # noinspection PyBroadException
                try:
                    credentials = service_account.Credentials.from_service_account_file(cfg.credentials_json)
                except Exception:
                    credentials = None

                if not credentials:
                    # noinspection PyBroadException
                    try:
                        # Try parsing this as json to support actual json content and not a file path
                        credentials = service_account.Credentials.from_service_account_info(
                            json.loads(cfg.credentials_json)
                        )
                    except Exception:
                        pass
            else:
                credentials = None

            self.client = storage.Client(project=cfg.project, credentials=credentials)
            for adapter in self.client._http.adapters.values():
                if cfg.pool_connections:
                    adapter._pool_connections = cfg.pool_connections
                if cfg.pool_maxsize:
                    adapter._pool_maxsize = cfg.pool_maxsize

            self.config = cfg
            self.bucket = self.client.bucket(self.name)

    def _get_stream_download_pool(self) -> ThreadPoolExecutor:
        if self._stream_download_pool is None or self._stream_download_pool_pid != os.getpid():
            self._stream_download_pool_pid = os.getpid()
            self._stream_download_pool = ThreadPoolExecutor(max_workers=int(self._stream_download_pool_connections))
        return self._stream_download_pool

    def get_container(
        self,
        container_name: str,
        config: Optional[Any] = None,
        **kwargs: Any,
    ) -> _Container:
        if container_name not in self._containers:
            self._containers[container_name] = self._Container(name=container_name, cfg=config)
        self._containers[container_name].config.retries = kwargs.get("retries", 5)
        return self._containers[container_name]

    def upload_object_via_stream(
        self,
        iterator: Any,
        container: Any,
        object_name: str,
        extra: Optional[dict] = None,
        **kwargs: Any,
    ) -> bool:
        try:
            blob = container.bucket.blob(object_name)
            blob.upload_from_file(iterator)
        except Exception as ex:
            self.get_logger().error("Failed uploading: %s" % ex)
            return False
        return True

    def upload_object(
        self,
        file_path: str,
        container: Any,
        object_name: str,
        extra: Optional[dict] = None,
        **kwargs: Any,
    ) -> bool:
        try:
            blob = container.bucket.blob(object_name)
            blob.upload_from_filename(file_path)
        except Exception as ex:
            self.get_logger().error("Failed uploading: %s" % ex)
            return False
        return True

    def list_container_objects(
        self,
        container: Any,
        ex_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Any]:
        # noinspection PyBroadException
        try:
            return list(container.bucket.list_blobs(prefix=ex_prefix))
        except TypeError:
            # google-cloud-storage < 1.17
            return [blob for blob in container.bucket.list_blobs() if blob.name.startswith(ex_prefix)]

    def delete_object(self, object: Any, **kwargs: Any) -> bool:
        try:
            object.delete()
        except Exception as ex:
            try:
                from google.cloud.exceptions import NotFound  # noqa

                if isinstance(ex, NotFound):
                    return False
            except ImportError:
                pass
            name = getattr(object, "name", "")
            if not kwargs.get("silent", False):
                self.get_logger().warning("Failed deleting object {}: {}".format(name, ex))
            return False

        return not object.exists()

    def get_object(
        self,
        container_name: str,
        object_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        full_container_name = str(furl(scheme=self.scheme, netloc=container_name))
        container = self._containers[full_container_name]
        obj = container.bucket.blob(object_name)
        obj.container_name = full_container_name
        return obj

    def download_object_as_stream(self, obj: Any, chunk_size: int = 256 * 1024, **_: Any) -> _Stream:
        raise NotImplementedError("Unsupported for google storage")

        def async_download(a_obj: Any, a_stream: Any) -> None:
            try:
                a_obj.download_to_file(a_stream)
            except Exception as ex:
                self.get_logger().error("Failed downloading: %s" % ex)
            a_stream.close()

        # return iterable object
        stream = _Stream()
        obj.chunk_size = chunk_size
        self._get_stream_download_pool().submit(async_download, obj, stream)

        return stream

    def download_object(
        self,
        obj: Any,
        local_path: str,
        overwrite_existing: bool = True,
        delete_on_failure: bool = True,
        callback: Any = None,
        **_: Any,
    ) -> None:
        p = Path(local_path)
        if not overwrite_existing and p.is_file():
            self.get_logger().warning("failed saving after download: overwrite=False and file exists (%s)" % str(p))
            return
        obj.download_to_filename(str(p))

    def test_upload(self, test_path: str, config: Any, **_: Any) -> bool:
        bucket_url = str(furl(scheme=self.scheme, netloc=config.bucket))
        bucket = self.get_container(container_name=bucket_url, config=config).bucket

        test_obj = bucket

        if test_path:
            if not test_path.endswith("/"):
                test_path += "/"

            blob = bucket.blob(test_path)

            if blob.exists():
                test_obj = blob

        permissions_to_test = ("storage.objects.get", "storage.objects.update")
        return set(test_obj.test_iam_permissions(permissions_to_test)) == set(permissions_to_test)

    def get_direct_access(self, remote_path: str, **_: Any) -> None:
        return None

    def exists_file(self, container_name: str, object_name: str) -> bool:
        return self.get_object(container_name, object_name).exists()


class _AzureBlobServiceStorageDriver(_Driver):
    scheme = "azure"

    _containers = {}
    _max_connections = deferred_config("azure.storage.max_connections", 0)

    class _Container(object):
        def __init__(
            self,
            name: str,
            config: AzureContainerConfigurations,
            account_url: str,
        ) -> None:
            self.MAX_SINGLE_PUT_SIZE = 4 * 1024 * 1024
            self.SOCKET_TIMEOUT = (300, 2000)
            self.name = name
            self.config = config
            self.account_url = account_url
            try:
                from azure.storage.blob import BlobServiceClient  # noqa

                self.__legacy = False
            except ImportError:
                try:
                    from azure.storage.blob import BlockBlobService  # noqa
                    from azure.common import AzureHttpError  # noqa

                    self.__legacy = True
                except ImportError:
                    raise UsageError(
                        "Azure blob storage driver not found. "
                        "Please install driver using: 'pip install clearml[azure]' or "
                        "pip install '\"azure.storage.blob>=12.0.0\"'"
                    )

            if self.__legacy:
                self.__blob_service = BlockBlobService(
                    account_name=self.config.account_name,
                    account_key=self.config.account_key,
                )
                self.__blob_service.MAX_SINGLE_PUT_SIZE = self.MAX_SINGLE_PUT_SIZE
                self.__blob_service.socket_timeout = self.SOCKET_TIMEOUT
            else:
                credential = {
                    "account_name": self.config.account_name,
                    "account_key": self.config.account_key,
                }
                self.__blob_service = BlobServiceClient(
                    account_url=account_url,
                    credential=credential,
                    max_single_put_size=self.MAX_SINGLE_PUT_SIZE,
                )

        @staticmethod
        def _get_max_connections_dict(
            max_connections: Optional[Union[int, str]] = None,
            key: str = "max_connections",
        ) -> Dict[str, int]:
            # must cast for deferred resolving
            try:
                max_connections = max_connections or int(_AzureBlobServiceStorageDriver._max_connections)
            except (AttributeError, TypeError):
                return {}
            return {key: int(max_connections)} if max_connections else {}

        def create_blob_from_data(
            self,
            container_name: str,
            object_name: str,
            blob_name: str,
            data: bytes,
            max_connections: Optional[int] = None,
            progress_callback: Optional[Callable] = None,
            content_settings: Optional["ContentSettings"] = None,
        ) -> None:
            if self.__legacy:
                self.__blob_service.create_blob_from_bytes(
                    container_name,
                    object_name,
                    data,
                    progress_callback=progress_callback,
                    **self._get_max_connections_dict(max_connections),
                )
            else:
                client = self.__blob_service.get_blob_client(container_name, blob_name)
                client.upload_blob(
                    data,
                    overwrite=True,
                    content_settings=content_settings,
                    **self._get_max_connections_dict(max_connections, key="max_concurrency"),
                )

        def create_blob_from_path(
            self,
            container_name: str,
            blob_name: str,
            path: str,
            max_connections: Optional[int] = None,
            content_settings: Optional["ContentSettings"] = None,
            progress_callback: Optional[Callable[[int, int], None]] = None,
        ) -> None:
            if self.__legacy:
                self.__blob_service.create_blob_from_path(
                    container_name,
                    blob_name,
                    path,
                    content_settings=content_settings,
                    progress_callback=progress_callback,
                    **self._get_max_connections_dict(max_connections),
                )
            else:
                with open(path, "rb") as f:
                    self.create_blob_from_data(
                        container_name,
                        None,
                        blob_name,
                        f,
                        content_settings=content_settings,
                        max_connections=max_connections,
                    )

        def delete_blob(self, container_name: str, blob_name: str) -> None:
            if self.__legacy:
                self.__blob_service.delete_blob(
                    container_name,
                    blob_name,
                )
            else:
                client = self.__blob_service.get_blob_client(container_name, blob_name)
                client.delete_blob()

        def exists(self, container_name: str, blob_name: str) -> bool:
            if self.__legacy:
                return not self.__blob_service.exists(container_name, blob_name)
            else:
                client = self.__blob_service.get_blob_client(container_name, blob_name)
                return client.exists()

        def list_blobs(self, container_name: str, prefix: Optional[str] = None) -> Any:
            if self.__legacy:
                return self.__blob_service.list_blobs(container_name=container_name, prefix=prefix)
            else:
                client = self.__blob_service.get_container_client(container_name)
                return client.list_blobs(name_starts_with=prefix)

        def get_blob_properties(self, container_name: str, blob_name: str) -> Any:
            if self.__legacy:
                return self.__blob_service.get_blob_properties(container_name, blob_name)
            else:
                client = self.__blob_service.get_blob_client(container_name, blob_name)
                return client.get_blob_properties()

        def get_blob_to_bytes(
            self,
            container_name: str,
            blob_name: str,
            progress_callback: Optional[Callable[[int, int], None]] = None,
        ) -> bytes:
            if self.__legacy:
                return self.__blob_service.get_blob_to_bytes(
                    container_name,
                    blob_name,
                    progress_callback=progress_callback,
                )
            else:
                client = self.__blob_service.get_blob_client(container_name, blob_name)
                return client.download_blob().content_as_bytes()

        def get_blob_to_path(
            self,
            container_name: str,
            blob_name: str,
            path: str,
            max_connections: Optional[int] = None,
            progress_callback: Optional[Callable] = None,
        ) -> None:
            if self.__legacy:
                return self.__blob_service.get_blob_to_path(
                    container_name,
                    blob_name,
                    path,
                    progress_callback=progress_callback,
                    **self._get_max_connections_dict(max_connections),
                )
            else:
                client = self.__blob_service.get_blob_client(container_name, blob_name)
                with open(path, "wb") as file:
                    return client.download_blob(
                        **self._get_max_connections_dict(max_connections, key="max_concurrency")
                    ).download_to_stream(file)

        def is_legacy(self) -> bool:
            return self.__legacy

        @property
        def blob_service(self) -> Any:
            return self.__blob_service

    @attrs
    class _Object(object):
        container = attrib()
        blob_name = attrib()
        content_length = attrib()

    def get_container(
        self,
        container_name: Optional[str] = None,
        config: Optional[Any] = None,
        account_url: Optional[str] = None,
        **kwargs: Any,
    ) -> _Container:
        container_name = container_name or config.container_name
        if container_name not in self._containers:
            self._containers[container_name] = self._Container(
                name=container_name, config=config, account_url=account_url
            )
        return self._containers[container_name]

    def upload_object_via_stream(
        self,
        iterator: Any,
        container: Any,
        object_name: str,
        callback: Any = None,
        extra: dict = None,
        max_connections: int = None,
        **kwargs: Any,
    ) -> bool:
        try:
            from azure.common import AzureHttpError  # noqa
        except ImportError:
            from azure.core.exceptions import HttpResponseError  # noqa

            AzureHttpError = HttpResponseError  # noqa

        blob_name = self._blob_name_from_object_path(object_name, container.name)  # noqa: F841
        try:
            container.create_blob_from_data(
                container.name,
                object_name,
                blob_name,
                iterator.read() if hasattr(iterator, "read") else bytes(iterator),
                max_connections=max_connections,
                progress_callback=callback,
            )
            return True
        except AzureHttpError as ex:
            self.get_logger().error("Failed uploading (Azure error): %s" % ex)
        except Exception as ex:
            self.get_logger().error("Failed uploading: %s" % ex)
        return False

    def upload_object(
        self,
        file_path: str,
        container: Any,
        object_name: str,
        callback: Any = None,
        extra: dict = None,
        max_connections: int = None,
        **kwargs: Any,
    ) -> bool:
        try:
            from azure.common import AzureHttpError  # noqa
        except ImportError:
            from azure.core.exceptions import HttpResponseError  # noqa

            AzureHttpError = HttpResponseError  # noqa

        blob_name = self._blob_name_from_object_path(object_name, container.name)
        try:
            from azure.storage.blob import ContentSettings  # noqa

            container.create_blob_from_path(
                container.name,
                blob_name,
                file_path,
                max_connections=max_connections,
                content_settings=ContentSettings(content_type=get_file_mimetype(object_name or file_path)),
                progress_callback=callback,
            )
            return True
        except AzureHttpError as ex:
            self.get_logger().error("Failed uploading (Azure error): %s" % ex)
        except Exception as ex:
            self.get_logger().error("Failed uploading: %s" % ex)

    def list_container_objects(
        self,
        container: Any,
        ex_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Any]:
        return list(container.list_blobs(container_name=container.name, prefix=ex_prefix))

    def delete_object(self, object: Any, **kwargs: Any) -> bool:
        container = object.container
        container.delete_blob(
            container.name,
            object.blob_name,
        )
        return not object.container.exists(container.name, object.blob_name)

    def get_object(
        self,
        container_name: str,
        object_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> _Object:
        container = self._containers.get(container_name)
        if not container:
            raise StorageError("Container `{}` not found for object {}".format(container_name, object_name))

        # blob_name = self._blob_name_from_object_path(object_name, container_name)
        blob = container.get_blob_properties(container.name, object_name)

        if container.is_legacy():
            return self._Object(
                container=container,
                blob_name=blob.name,
                content_length=blob.properties.content_length,
            )
        else:
            return self._Object(container=container, blob_name=blob.name, content_length=blob.size)

    def download_object_as_stream(self, obj: Any, verbose: bool, *_: Any, **__: Any) -> bytes:
        container = obj.container
        total_size_mb = obj.content_length / (1024.0 * 1024.0)
        remote_path = os.path.join(
            "{}://".format(self.scheme),
            container.config.account_name,
            container.name,
            obj.blob_name,
        )
        cb = DownloadProgressReport(total_size_mb, verbose, remote_path, self.get_logger())
        blob = container.get_blob_to_bytes(
            container.name,
            obj.blob_name,
            progress_callback=cb,
        )
        cb.close()
        if container.is_legacy():
            return blob.content
        else:
            return blob

    def download_object(
        self,
        obj: Any,
        local_path: str,
        overwrite_existing: bool = True,
        delete_on_failure: bool = True,
        callback: Callable[[int, int], None] = None,
        max_connections: Optional[int] = None,
        **_: Any,
    ) -> None:
        p = Path(local_path)
        if not overwrite_existing and p.is_file():
            self.get_logger().warning("Failed saving after download: overwrite=False and file exists (%s)" % str(p))
            return

        download_done = SafeEvent()
        download_done.counter = 0

        def callback_func(current: int, total: int) -> None:
            if callback:
                chunk = current - download_done.counter
                download_done.counter += chunk
                callback(chunk)
            if current >= total:
                download_done.set()

        container = obj.container
        container.blob_service.MAX_SINGLE_GET_SIZE = 5 * 1024 * 1024
        _ = container.get_blob_to_path(
            container.name,
            obj.blob_name,
            local_path,
            max_connections=max_connections,
            progress_callback=callback_func,
        )
        if container.is_legacy():
            download_done.wait()

    def test_upload(self, test_path: str, config: Any, **_: Any) -> bool:
        container = self.get_container(config=config)
        try:
            container.blob_service.get_container_properties(container.name)
        except Exception:
            return False
        else:
            # Using the account Key, we can always upload...
            return True

    @classmethod
    def _blob_name_from_object_path(cls, name: str, container_name: str) -> Union[Tuple[str, str], str]:
        scheme = urlparse(name).scheme
        if scheme:
            if scheme != cls.scheme:
                raise StorageError(
                    "When using a URL, only the `{}` scheme is supported for Azure storage: {}",
                    cls.scheme,
                    name,
                )

            f = furl(name)

            if not f.path.segments:
                raise StorageError(
                    "Missing container name in URL {}",
                    name,
                )

            parsed_container_name = f.path.segments[0]

            if parsed_container_name != container_name:
                raise StorageError(
                    "Container name mismatch (expected {}, found {}) in {}",
                    container_name,
                    parsed_container_name,
                    name,
                )

            if len(f.path.segments) == 1:
                raise StorageError(
                    "No path found following container name {} in {}",
                    container_name,
                    name,
                )

            return f.path.segments[0], os.path.join(*f.path.segments[1:])

        return name

    def get_direct_access(self, remote_path: str, **_: Any) -> None:
        return None

    def exists_file(self, container_name: str, object_name: str) -> bool:
        container = self.get_container(container_name)
        return container.exists(container_name, blob_name=object_name)


class _FileStorageDriver(_Driver):
    """
    A base StorageDriver to derive from.
    """

    scheme = "file"
    CHUNK_SIZE = 8096
    IGNORE_FOLDERS = [".lock", ".hash"]
    Object = namedtuple("Object", ["name", "size", "extra", "driver", "container", "hash", "meta_data"])

    class _Container(object):
        def __init__(self, name: str, extra: dict, driver: Any) -> None:
            self.name = name
            self.extra = extra
            self.driver = driver

    def __init__(
        self,
        key: str,
        secret: Optional[str] = None,
        secure: bool = True,
        host: Optional[str] = None,
        port: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        # Use the key as the path to the storage
        self.base_path = key

    def _make_path(self, path: str, ignore_existing: bool = True) -> None:
        """
        Create a path by checking if it already exists
        """

        try:
            os.makedirs(path)
        except OSError:
            exp = sys.exc_info()[1]
            if exp.errno == errno.EEXIST and not ignore_existing:
                raise exp

    def _check_container_name(self, container_name: str) -> None:
        """
        Check if the container name is valid

        :param container_name: Container name
        :type container_name: ``str``
        """

        if "/" in container_name or "\\" in container_name:
            raise ValueError('Container name "{}" cannot contain \\ or / '.format(container_name))

    def _make_container(self, container_name: str) -> _Container:
        """
        Create a container instance

        :param container_name: Container name.
        :type container_name: ``str``

        :return: A Container instance.
        """
        container_name = container_name or "."
        self._check_container_name(container_name)

        full_path = os.path.realpath(os.path.join(self.base_path, container_name))

        try:
            stat = os.stat(full_path)
        except OSError:
            raise OSError('Target path "{}" is not accessible or does not exist'.format(full_path))

        extra = {
            "creation_time": stat.st_ctime,
            "access_time": stat.st_atime,
            "modify_time": stat.st_mtime,
        }

        return self._Container(name=container_name, extra=extra, driver=self)

    def _make_object(self, container: Any, object_name: str) -> Object:
        """
        Create an object instance

        :param container: Container.
        :type container: :class:`Container`

        :param object_name: Object name.
        :type object_name: ``str``

        :return: A Object instance.
        """

        full_path = os.path.realpath(os.path.join(self.base_path, container.name if container else ".", object_name))

        if os.path.isdir(full_path):
            raise ValueError('Target path "{}" already exist'.format(full_path))

        try:
            stat = os.stat(full_path)
        except Exception:
            raise ValueError('Cannot access target path "{}"'.format(full_path))

        extra = {
            "creation_time": stat.st_ctime,
            "access_time": stat.st_atime,
            "modify_time": stat.st_mtime,
        }

        return self.Object(
            name=object_name,
            size=stat.st_size,
            extra=extra,
            driver=self,
            container=container,
            hash=None,
            meta_data=None,
        )

    def iterate_containers(self) -> Generator[Any, None, None]:
        """
        Return a generator of containers.

        :return: A generator of Container instances.
        """

        for container_name in os.listdir(self.base_path):
            full_path = os.path.join(self.base_path, container_name)
            if not os.path.isdir(full_path):
                continue
            yield self._make_container(container_name)

    def _get_objects(self, container: Any, prefix: Optional[str] = None) -> Generator[Object, None, None]:
        """
        Recursively iterate through the file-system and return the object names
        """

        cpath = self.get_container_cdn_url(container, check=True)
        if prefix:
            cpath += "/" + prefix

        for folder, subfolders, files in os.walk(cpath, topdown=True):
            # Remove unwanted subfolders
            for subf in self.IGNORE_FOLDERS:
                if subf in subfolders:
                    subfolders.remove(subf)

            for name in files:
                full_path = os.path.join(folder, name)
                object_name = os.path.relpath(full_path, start=cpath)
                yield self._make_object(container, object_name)

    def iterate_container_objects(self, container: Any, prefix: Optional[str] = None) -> Generator[Object, None, None]:
        """
        Returns a generator of objects for the given container.

        :param container: Container instance
        :type container: :class:`Container`
        :param prefix: The path of a sub directory under the base container path.
            The iterator will only include paths under that subdir.
        :type prefix: Optional[str]

        :return: A generator of Object instances.
        """

        return self._get_objects(container, prefix=prefix)

    def get_container(self, container_name: str, **_: Any) -> Any:
        """
        Return a container instance.

        :param container_name: Container name.
        :type container_name: ``str``

        :return: A Container instance.
        """
        return self._make_container(container_name)

    def get_container_cdn_url(self, container: Any, check: bool = False) -> str:
        """
        Return a container CDN URL.

        :param container: Container instance
        :type  container: :class:`Container`

        :param check: Indicates if the path's existence must be checked
        :type check: ``bool``

        :return: A CDN URL for this container.
        """
        path = os.path.realpath(os.path.join(self.base_path, container.name if container else "."))

        if check and not os.path.isdir(path):
            raise ValueError('Target path "{}" does not exist'.format(path))

        return path

    def get_object(self, container_name: str, object_name: str, **_: Any) -> Object:
        """
        Return an object instance.

        :param container_name: Container name.
        :type  container_name: ``str``

        :param object_name: Object name.
        :type  object_name: ``str``

        :return: An Object instance.
        """
        container = self._make_container(container_name)
        if os.path.isfile(os.path.join(container_name, object_name)):
            return self.Object(
                name=object_name,
                container=container,
                size=Path(object_name).stat().st_size,
                driver=self,
                extra=None,
                hash=None,
                meta_data=None,
            )
        return self._make_object(container, object_name)

    def get_object_cdn_url(self, obj: Object) -> str:
        """
        Return an object CDN URL.

        :param obj: Object instance
        :type  obj: :class:`Object`

        :return: A CDN URL for this object.
        """
        return os.path.realpath(os.path.join(self.base_path, obj.container.name, obj.name))

    def download_object(
        self,
        obj: Object,
        destination_path: str,
        overwrite_existing: bool = False,
        delete_on_failure: bool = True,
        **_: Any,
    ) -> bool:
        """
        Download an object to the specified destination path.

        :param obj: Object instance.
        :type obj: :class:`Object`

        :param destination_path: Full path to a file or a directory where the
                                incoming file will be saved.
        :type destination_path: ``str``

        :param overwrite_existing: True to overwrite an existing file,
            defaults to False.
        :type overwrite_existing: ``bool``

        :param delete_on_failure: True to delete a partially downloaded file if
        the download was not successful (hash mismatch / file size).
        :type delete_on_failure: ``bool``

        :return: True, if an object has been successfully downloaded, False, otherwise.
        """

        obj_path = self.get_object_cdn_url(obj)
        base_name = os.path.basename(destination_path)

        if not base_name and not os.path.exists(destination_path):
            raise ValueError('Path "{}" does not exist'.format(destination_path))

        if not base_name:
            file_path = os.path.join(destination_path, obj.name)
        else:
            file_path = destination_path

        if os.path.exists(file_path) and not overwrite_existing:
            raise ValueError('File "{}" already exists, but overwrite_existing=False'.format(file_path))

        try:
            shutil.copy(obj_path, file_path)
        except IOError:
            if delete_on_failure:
                # noinspection PyBroadException
                try:
                    os.unlink(file_path)
                except Exception:
                    pass
            return False

        return True

    def download_object_as_stream(self, obj: Object, chunk_size: int = None, **_: Any) -> Generator[bytes, None, None]:
        """
        Return a generator which yields object data.

        :param obj: Object instance
        :type obj: :class:`Object`

        :param chunk_size: Optional chunk size (in bytes).
        :type chunk_size: ``int``

        :return: A stream of binary chunks of data.
        """
        path = self.get_object_cdn_url(obj)
        with open(path, "rb") as obj_file:
            for data in self._read_in_chunks(obj_file, chunk_size=chunk_size):
                yield data

    def upload_object(
        self,
        file_path: str,
        container: Any,
        object_name: str,
        extra: dict = None,
        verify_hash: bool = True,
        **_: Any,
    ) -> Object:
        """
        Upload an object currently located on a disk.

        :param file_path: Path to the object on disk.
        :type file_path: ``str``

        :param container: Destination container.
        :type container: :class:`Container`

        :param object_name: Object name.
        :type object_name: ``str``

        :param verify_hash: Verify hast
        :type verify_hash: ``bool``

        :param extra: (optional) Extra attributes (driver specific).
        :type extra: ``dict``
        """
        path = self.get_container_cdn_url(container, check=False)
        obj_path = os.path.join(path, object_name)
        base_path = os.path.dirname(obj_path)

        self._make_path(base_path)

        shutil.copy(file_path, obj_path)

        os.chmod(obj_path, int("664", 8))

        return self._make_object(container, object_name)

    def upload_object_via_stream(
        self,
        iterator: Any,
        container: Any,
        object_name: str,
        extra: dict = None,
        **kwargs: Any,
    ) -> Object:
        """
        Upload an object using an iterator.

        If a provider supports it, chunked transfer encoding is used and you
        don't need to know in advance the amount of data to be uploaded.

        Otherwise if a provider doesn't support it, iterator will be exhausted
        so a total size for data to be uploaded can be determined.

        Note: Exhausting the iterator means that the whole data must be
        buffered in memory which might result in memory exhausting when
        uploading a very large object.

        If a file is located on a disk you are advised to use upload_object
        function which uses fs.stat function to determine the file size and it
        doesn't need to buffer whole object in the memory.

        :type iterator: ``object``
        :param iterator: An object which implements the iterator
                         interface and yields binary chunks of data.

        :type container: :class:`Container`
        :param container: Destination container.

        :type object_name: ``str``
        :param object_name: Object name.

        :type extra: ``dict``
        :param extra: (optional) Extra attributes (driver specific). Note:
            This dictionary must contain a 'content_type' key which represents
            a content type of the stored object.
        """
        path = self.get_container_cdn_url(container, check=True)
        obj_path = os.path.join(path, object_name)
        base_path = os.path.dirname(obj_path)
        self._make_path(base_path)

        obj_path = os.path.realpath(obj_path)
        with open(obj_path, "wb" if not isinstance(iterator, StringIO) else "wt") as obj_file:
            obj_file.write(iterator.read() if hasattr(iterator, "read") else bytes(iterator))

        os.chmod(obj_path, int("664", 8))
        return self._make_object(container, object_name)

    def delete_object(self, obj: Object, **_: Any) -> bool:
        """
        Delete an object.

        :type obj: :class:`Object`
        :param obj: Object instance.

        :return: True on success.
        """
        if not obj:
            return False

        path = self.get_object_cdn_url(obj)

        try:
            os.unlink(path)
        except Exception:  # noqa
            return False

        # # Check and delete all the empty parent folders
        # path = os.path.dirname(path)
        # container_url = obj.container.get_cdn_url()
        #
        # # Delete the empty parent folders till the container's level
        # while path != container_url:
        #     try:
        #         os.rmdir(path)
        #     except OSError:
        #         exp = sys.exc_info()[1]
        #         if exp.errno == errno.ENOTEMPTY:
        #             break
        #         raise exp
        #
        #     path = os.path.dirname(path)

        return True

    def create_container(self, container_name: str) -> Any:
        """
        Create a new container.

        :type container_name: ``str``
        :param container_name: Container name.

        :return: A Container instance on success.
        """
        container_name = container_name or "."
        self._check_container_name(container_name)

        path = os.path.join(self.base_path, container_name)

        try:
            self._make_path(path, ignore_existing=False)
        except OSError:
            exp = sys.exc_info()[1]
            if exp.errno == errno.EEXIST:
                raise ValueError(
                    'Container "{}" with this name already exists. The name '
                    "must be unique among all the containers in the "
                    "system".format(container_name)
                )
            else:
                raise ValueError('Error creating container "{}"'.format(container_name))
        except Exception:
            raise ValueError('Error creating container "{}"'.format(container_name))

        return self._make_container(container_name)

    def delete_container(self, container: Any) -> bool:
        """
        Delete a container.

        :type container: :class:`Container`
        :param container: Container instance

        :return: True on success, False otherwise.
        """

        # Check if there are any objects inside this
        for obj in self._get_objects(container):
            raise ValueError('Container "{}" is not empty'.format(container.name))

        path = self.get_container_cdn_url(container, check=True)

        # noinspection PyBroadException
        try:
            shutil.rmtree(path)
        except Exception:
            return False

        return True

    def list_container_objects(
        self,
        container: Any,
        ex_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Object]:
        return list(self.iterate_container_objects(container, prefix=ex_prefix))

    @staticmethod
    def _read_in_chunks(
        iterator: Any,
        chunk_size: int = None,
        fill_size: bool = False,
        yield_empty: bool = False,
    ) -> Generator[bytes, None, None]:
        """
        Return a generator which yields data in chunks.

        :param iterator: An object which implements an iterator interface
                         or a File like object with read method.
        :type iterator: :class:`object` which implements iterator interface.

        :param chunk_size: Optional chunk size (defaults to CHUNK_SIZE)
        :type chunk_size: ``int``

        :param fill_size: If True, make sure chunks are exactly chunk_size in
                          length (except for last chunk).
        :type fill_size: ``bool``

        :param yield_empty: If true and iterator returned no data, only yield empty
                            bytes object
        :type yield_empty: ``bool``

        TODO: At some point in the future we could use byte arrays here if version
        >= Python 3. This should speed things up a bit and reduce memory usage.
        """
        chunk_size = chunk_size or _FileStorageDriver.CHUNK_SIZE

        if isinstance(iterator, FileIO):
            get_data = iterator.read
            args = (chunk_size,)
        else:
            get_data = next
            args = (iterator,)

        data = b""
        empty = False

        while not empty or len(data) > 0:
            if not empty:
                try:
                    chunk = bytes(get_data(*args))
                    if len(chunk) > 0:
                        data += chunk
                    else:
                        empty = True
                except StopIteration:
                    empty = True

            if len(data) == 0:
                if empty and yield_empty:
                    yield b""

                return

            if fill_size:
                if empty or len(data) >= chunk_size:
                    yield data[:chunk_size]
                    data = data[chunk_size:]
            else:
                yield data
                data = b""

    def get_direct_access(self, remote_path: str, **_: Any) -> Optional[str]:
        if bool(StorageHelper.use_disk_space_file_size_strategy):
            return None
        # this will always make sure we have full path and file:// prefix
        full_url = _StorageHelper.conform_url(remote_path)
        # now get rid of the file:// prefix
        path = Path(full_url[7:])
        if not path.exists():
            raise ValueError("Requested path does not exist: {}".format(path))
        return path.as_posix()

    def test_upload(self, test_path: str, config: Any, **kwargs: Any) -> bool:
        return True

    def exists_file(self, container_name: str, object_name: str) -> bool:
        return os.path.isfile(object_name)


class _StorageHelper(object):
    """Storage helper.
    Used by the entire system to download/upload files.
    Supports both local and remote files (currently local files, network-mapped files, HTTP/S and Amazon S3)
    """

    _temp_download_suffix = ".partially"
    _quotable_uri_schemes = set(_HttpDriver.schemes)

    @classmethod
    def _get_logger(cls) -> logging.Logger:
        return get_logger("storage")

    @attrs
    class _PathSubstitutionRule(object):
        registered_prefix = attrib(type=str)
        local_prefix = attrib(type=str)
        replace_windows_sep = attrib(type=bool)
        replace_linux_sep = attrib(type=bool)

        path_substitution_config = "storage.path_substitution"

        @classmethod
        def load_list_from_config(
            cls,
        ) -> List["_StorageHelper._PathSubstitutionRule"]:
            rules_list = []
            for index, sub_config in enumerate(config.get(cls.path_substitution_config, list())):
                rule = cls(
                    registered_prefix=sub_config.get("registered_prefix", None),
                    local_prefix=sub_config.get("local_prefix", None),
                    replace_windows_sep=sub_config.get("replace_windows_sep", False),
                    replace_linux_sep=sub_config.get("replace_linux_sep", False),
                )

                if any(prefix is None for prefix in (rule.registered_prefix, rule.local_prefix)):
                    cls._get_logger().warning(
                        "Illegal substitution rule configuration '{}[{}]': {}".format(
                            cls.path_substitution_config,
                            index,
                            asdict(rule),
                        )
                    )

                    continue

                if all((rule.replace_windows_sep, rule.replace_linux_sep)):
                    cls._get_logger().warning(
                        "Only one of replace_windows_sep and replace_linux_sep flags may be set."
                        "'{}[{}]': {}".format(
                            cls.path_substitution_config,
                            index,
                            asdict(rule),
                        )
                    )
                    continue

                rules_list.append(rule)

            return rules_list

    class _UploadData(object):
        @property
        def src_path(self) -> str:
            return self._src_path

        @property
        def dest_path(self) -> str:
            return self._dest_path

        @property
        def canonized_dest_path(self) -> str:
            return self._canonized_dest_path

        @property
        def extra(self) -> dict:
            return self._extra

        @property
        def callback(self) -> Any:
            return self._callback

        @property
        def retries(self) -> int:
            return self._retries

        @property
        def return_canonized(self) -> bool:
            return self._return_canonized

        def __init__(
            self,
            src_path: str,
            dest_path: str,
            canonized_dest_path: str,
            extra: dict,
            callback: Any,
            retries: int,
            return_canonized: bool,
        ) -> None:
            self._src_path = src_path
            self._dest_path = dest_path
            self._canonized_dest_path = canonized_dest_path
            self._extra = extra
            self._callback = callback
            self._retries = retries
            self._return_canonized = return_canonized

        def __str__(self) -> str:
            return "src=%s" % self.src_path

    _helpers = {}  # cache of helper instances

    # global terminate event for async upload threads
    # _terminate = threading.Event()
    _async_upload_threads = set()
    _upload_pool = None
    _upload_pool_pid = None

    # collect all bucket credentials that aren't empty (ignore entries with an empty key or secret)
    _s3_configurations = deferred_config("aws.s3", {}, transform=S3BucketConfigurations.from_config)
    _gs_configurations = deferred_config("google.storage", {}, transform=GSBucketConfigurations.from_config)
    _azure_configurations = deferred_config("azure.storage", {}, transform=AzureContainerConfigurations.from_config)
    _path_substitutions = deferred_config(transform=_PathSubstitutionRule.load_list_from_config)

    @property
    def log(self) -> logging.Logger:
        return self._log

    @property
    def scheme(self) -> str:
        return self._scheme

    @property
    def secure(self) -> bool:
        return self._secure

    @property
    def base_url(self) -> str:
        return self._base_url

    @classmethod
    def get(cls, url: str, logger: Optional[logging.Logger] = None, **kwargs: Any) -> Optional["_StorageHelper"]:
        """
        Get a storage helper instance for the given URL

        :return: A _StorageHelper instance.
        """
        # Handle URL substitution etc before locating the correct storage driver
        url = cls._canonize_url(url)

        # Get the credentials we should use for this url
        base_url = cls._resolve_base_url(url)

        instance_key = "%s_%s" % (base_url, threading.current_thread().ident or 0)
        # noinspection PyBroadException
        try:
            configs = kwargs.get("configs")
            if configs:
                instance_key += "_{}".format(configs.cache_name)
        except Exception:
            pass

        force_create = kwargs.pop("__force_create", False)
        if (instance_key in cls._helpers) and (not force_create) and base_url != "file://":
            return cls._helpers[instance_key]

        # Don't canonize URL since we already did it
        try:
            instance = cls(base_url=base_url, url=url, logger=logger, canonize_url=False, **kwargs)
        except (StorageError, UsageError) as ex:
            cls._get_logger().error(str(ex))
            return None
        except Exception as ex:
            cls._get_logger().error("Failed creating storage object {} Reason: {}".format(base_url or url, ex))
            return None

        cls._helpers[instance_key] = instance
        return instance

    @classmethod
    def get_local_copy(cls, remote_url: str, skip_zero_size_check: bool = False) -> str:
        """
        Download a file from remote URL to a local storage, and return path to local copy,

        :param remote_url: Remote URL. Example: https://example.com/file.jpg s3://bucket/folder/file.mp4 etc.
        :param skip_zero_size_check: If True, no error will be raised for files with zero bytes size.
        :return: Path to local copy of the downloaded file. None if error occurred.
        """
        helper = cls.get(remote_url)
        if not helper:
            return None
        # create temp file with the requested file name
        file_name = "." + remote_url.split("/")[-1].split(os.path.sep)[-1]
        _, local_path = mkstemp(suffix=file_name)
        return helper.download_to_file(remote_url, local_path, skip_zero_size_check=skip_zero_size_check)

    def __init__(
        self,
        base_url: str,
        url: str,
        key: Optional[str] = None,
        secret: Optional[str] = None,
        region: Optional[str] = None,
        verbose: bool = False,
        logger: Optional[logging.Logger] = None,
        retries: int = 5,
        token: Optional[str] = None,
        profile: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        level = config.get("storage.log.level", None)

        if level:
            try:
                self._get_logger().setLevel(level)
            except (TypeError, ValueError):
                self._get_logger().error("invalid storage log level in configuration: %s" % level)

        self._log = logger or self._get_logger()
        self._verbose = verbose
        self._retries = retries
        self._extra = {}
        self._base_url = base_url
        self._secure = True
        self._driver = None
        self._container = None
        self._conf = None

        if kwargs.get("canonize_url", True):
            url = self._canonize_url(url)

        parsed = urlparse(url)
        self._scheme = parsed.scheme

        if self._scheme == _AzureBlobServiceStorageDriver.scheme:
            self._conf = copy(self._azure_configurations.get_config_by_uri(url))
            if self._conf is None:
                raise StorageError("Missing Azure Blob Storage configuration for {}".format(url))

            if not self._conf.account_name or not self._conf.account_key:
                raise StorageError("Missing account name or key for Azure Blob Storage access for {}".format(base_url))

            self._driver = _AzureBlobServiceStorageDriver()
            self._container = self._driver.get_container(config=self._conf, account_url=parsed.netloc)

        elif self._scheme == _Boto3Driver.scheme:
            self._conf = copy(self._s3_configurations.get_config_by_uri(url))
            self._secure = self._conf.secure

            final_region = region if region else self._conf.region
            if not final_region:
                final_region = None

            self._conf.update(
                key=key or self._conf.key,
                secret=secret or self._conf.secret,
                multipart=self._conf.multipart,
                region=final_region,
                use_credentials_chain=self._conf.use_credentials_chain,
                token=token or self._conf.token,
                profile=profile or self._conf.profile,
                secure=self._secure,
                extra_args=self._conf.extra_args,
            )

            if not self._conf.use_credentials_chain:
                if not self._conf.key or not self._conf.secret:
                    raise ValueError("Missing key and secret for S3 storage access (%s)" % base_url)

            self._driver = _Boto3Driver()
            self._container = self._driver.get_container(
                container_name=self._base_url, retries=retries, config=self._conf
            )

        elif self._scheme == _GoogleCloudStorageDriver.scheme:
            self._conf = copy(self._gs_configurations.get_config_by_uri(url))
            self._driver = _GoogleCloudStorageDriver()
            self._container = self._driver.get_container(container_name=self._base_url, config=self._conf)

        elif self._scheme in _HttpDriver.schemes:
            self._driver = _HttpDriver(retries=retries)
            self._container = self._driver.get_container(container_name=self._base_url)
        else:  # elif self._scheme == 'file':
            # if this is not a known scheme assume local file
            # url2pathname is specifically intended to operate on (urlparse result).path
            # and returns a cross-platform compatible result
            new_url = normalize_local_path(url[len("file://") :] if url.startswith("file://") else url)
            self._driver = _FileStorageDriver(new_url)
            # noinspection PyBroadException
            try:
                self._container = self._driver.get_container("")
            except Exception:
                self._container = None

    @classmethod
    def terminate_uploads(cls, force: bool = True, timeout: float = 2.0) -> None:
        if force:
            # since async uploaders are daemon threads, we can just return and let them close by themselves
            return
        # signal all threads to terminate and give them a chance for 'timeout' seconds (total, not per-thread)
        # cls._terminate.set()
        remaining_timeout = timeout
        for thread in cls._async_upload_threads:
            t = time()
            # noinspection PyBroadException
            try:
                thread.join(timeout=remaining_timeout)
            except Exception:
                pass
            remaining_timeout -= time() - t

    @classmethod
    def get_aws_storage_uri_from_config(cls, bucket_config: BucketConfig) -> str:
        uri = (
            "s3://{}/{}".format(bucket_config.host, bucket_config.bucket)
            if bucket_config.host
            else "s3://{}".format(bucket_config.bucket)
        )
        if bucket_config.subdir:
            uri += "/" + bucket_config.subdir
        return uri

    @classmethod
    def get_gcp_storage_uri_from_config(cls, bucket_config: BucketConfig) -> str:
        return (
            "gs://{}/{}".format(bucket_config.bucket, bucket_config.subdir)
            if bucket_config.subdir
            else "gs://{}".format(bucket_config.bucket)
        )

    @classmethod
    def get_azure_storage_uri_from_config(cls, bucket_config: BucketConfig) -> str:
        return "azure://{}.blob.core.windows.net/{}".format(bucket_config.account_name, bucket_config.container_name)

    @classmethod
    def get_configuration(cls, bucket_config: BucketConfig) -> S3BucketConfig:
        return cls.get_aws_configuration(bucket_config)

    @classmethod
    def get_aws_configuration(cls, bucket_config: BucketConfig) -> S3BucketConfig:
        return cls._s3_configurations.get_config_by_bucket(bucket_config.bucket, bucket_config.host)

    @classmethod
    def get_gcp_configuration(cls, bucket_config: BucketConfig) -> GSBucketConfigurations:
        return cls._gs_configurations.get_config_by_uri(
            cls.get_gcp_storage_uri_from_config(bucket_config),
            create_if_not_found=False,
        )

    @classmethod
    def get_azure_configuration(cls, bucket_config: AzureContainerConfig) -> AzureContainerConfig:
        return cls._azure_configurations.get_config(bucket_config.account_name, bucket_config.container_name)

    @classmethod
    def add_configuration(
        cls,
        bucket_config: BucketConfig,
        log: Optional[logging.Logger] = None,
        _test_config: bool = True,
    ) -> None:
        return cls.add_aws_configuration(bucket_config, log=log, _test_config=_test_config)

    @classmethod
    def add_aws_configuration(
        cls,
        bucket_config: BucketConfig,
        log: Optional[logging.Logger] = None,
        _test_config: bool = True,
    ) -> None:
        # Try to use existing configuration if we have no key and secret
        use_existing = not bucket_config.is_valid()
        # Get existing config anyway (we'll either try to use it or alert we're replacing it
        existing = cls.get_aws_configuration(bucket_config)
        configs = cls._s3_configurations
        uri = cls.get_aws_storage_uri_from_config(bucket_config)

        if not use_existing:
            # Test bucket config, fails if unsuccessful
            if _test_config:
                _Boto3Driver._test_bucket_config(bucket_config, log)  # noqa
            if existing:
                if log:
                    log.warning("Overriding existing configuration for '{}'".format(uri))
                configs.remove_config(existing)
            configs.add_config(bucket_config)
        else:
            # Try to use existing configuration
            good_config = False
            if existing:
                if log:
                    log.info("Using existing credentials for '{}'".format(uri))
                good_config = _Boto3Driver._test_bucket_config(existing, log, raise_on_error=False)  # noqa

            if not good_config:
                # Try to use global key/secret
                configs.update_config_with_defaults(bucket_config)

                if log:
                    log.info("Using global credentials for '{}'".format(uri))
                if _test_config:
                    _Boto3Driver._test_bucket_config(bucket_config, log)  # noqa
                configs.add_config(bucket_config)

    @classmethod
    def add_gcp_configuration(cls, bucket_config: BucketConfig, log: Optional[logging.Logger] = None) -> None:
        use_existing = not bucket_config.is_valid()
        existing = cls.get_gcp_configuration(bucket_config)
        configs = cls._gs_configurations
        uri = cls.get_gcp_storage_uri_from_config(bucket_config)

        if not use_existing:
            if existing:
                if log:
                    log.warning("Overriding existing configuration for '{}'".format(uri))
                configs.remove_config(existing)
            configs.add_config(bucket_config)
        else:
            good_config = False
            if existing:
                if log:
                    log.info("Using existing config for '{}'".format(uri))
                good_config = _GoogleCloudStorageDriver.test_upload(None, bucket_config)
            if not good_config:
                configs.update_config_with_defaults(bucket_config)
                if log:
                    log.info("Using global credentials for '{}'".format(uri))
                configs.add_config(bucket_config)

    @classmethod
    def add_azure_configuration(cls, bucket_config: BucketConfig, log: Optional[logging.Logger] = None) -> None:
        use_existing = not bucket_config.is_valid()
        existing = cls.get_azure_configuration(bucket_config)
        configs = cls._azure_configurations
        uri = cls.get_azure_storage_uri_from_config(bucket_config)

        if not use_existing:
            if existing:
                if log:
                    log.warning("Overriding existing configuration for '{}'".format(uri))
                configs.remove_config(existing)
            configs.add_config(bucket_config)
        else:
            good_config = False
            if existing:
                if log:
                    log.info("Using existing config for '{}'".format(uri))
                good_config = _AzureBlobServiceStorageDriver.test_upload(None, bucket_config)
            if not good_config:
                configs.update_config_with_defaults(bucket_config)
                if log:
                    log.info("Using global credentials for '{}'".format(uri))
                configs.add_config(bucket_config)

    @classmethod
    def add_path_substitution(
        cls,
        registered_prefix: str,
        local_prefix: str,
        replace_windows_sep: bool = False,
        replace_linux_sep: bool = False,
    ) -> None:
        """
        Add a path substitution rule for storage paths.

        Useful for case where the data was registered under some path, and that
        path was later renamed. This may happen with local storage paths where
        each machine is has different mounts or network drives configurations

        :param registered_prefix: The prefix to search for and replace. This is
            the prefix of the path the data is registered under. This should be the
            exact url prefix, case sensitive, as the data is registered.
        :param local_prefix: The prefix to replace 'registered_prefix' with. This
            is the prefix of the path the data is actually saved under. This should be the
            exact url prefix, case sensitive, as the data is saved under.
        :param replace_windows_sep: If set to True, and the prefix matches, the rest
            of the url has all of the windows path separators (backslash '\') replaced with
            the native os path separator.
        :param replace_linux_sep: If set to True, and the prefix matches, the rest
            of the url has all of the linux/unix path separators (slash '/') replaced with
            the native os path separator.
        """

        if not registered_prefix or not local_prefix:
            raise UsageError("Path substitution prefixes must be non empty strings")

        if replace_windows_sep and replace_linux_sep:
            raise UsageError("Only one of replace_windows_sep and replace_linux_sep may be set.")

        rule = cls._PathSubstitutionRule(
            registered_prefix=registered_prefix,
            local_prefix=local_prefix,
            replace_windows_sep=replace_windows_sep,
            replace_linux_sep=replace_linux_sep,
        )

        cls._path_substitutions.append(rule)

    @classmethod
    def clear_path_substitutions(cls) -> None:
        """
        Removes all path substitution rules, including ones from the configuration file.
        """
        cls._path_substitutions = list()

    def get_object_size_bytes(self, remote_url: str, silence_errors: bool = False) -> Optional[int]:
        """
        Get size of the remote file in bytes.

        :param str remote_url: The url where the file is stored.
            E.g. 's3://bucket/some_file.txt', 'file://local/file'
        :param bool silence_errors: Silence errors that might occur
            when fetching the size of the file. Default: False

        :return: The size of the file in bytes.
            None if the file could not be found or an error occurred.
        """
        obj = self.get_object(remote_url, silence_errors=silence_errors)
        return self._get_object_size_bytes(obj, silence_errors)

    def _get_object_size_bytes(self, obj: Any, silence_errors: bool = False) -> Optional[int]:
        """
        Auxiliary function for `get_object_size_bytes`.
        Get size of the remote object in bytes.

        :param object obj: The remote object
        :param bool silence_errors: Silence errors that might occur
            when fetching the size of the file. Default: False

        :return: The size of the object in bytes.
            None if an error occurred.
        """
        if not obj:
            return None
        size = None
        try:
            if isinstance(self._driver, _HttpDriver) and obj:
                obj = self._driver._get_download_object(obj)  # noqa
                size = int(obj.headers.get("Content-Length", 0))
            elif hasattr(obj, "size"):
                size = obj.size
                # Google storage has the option to reload the object to get the size
                if size is None and hasattr(obj, "reload"):
                    # noinspection PyBroadException
                    try:
                        # To catch google.api_core exceptions
                        obj.reload()
                        size = obj.size
                    except Exception as e:
                        if not silence_errors:
                            self.log.warning(
                                "Failed obtaining object size on reload: {}('{}')".format(e.__class__.__name__, str(e))
                            )
            elif hasattr(obj, "content_length"):
                # noinspection PyBroadException
                try:
                    # To catch botocore exceptions
                    size = obj.content_length  # noqa
                except Exception as e:
                    if not silence_errors:
                        self.log.warning(
                            "Failed obtaining content_length while getting object size: {}('{}')".format(
                                e.__class__.__name__, str(e)
                            )
                        )
        except Exception as e:
            if not silence_errors:
                self.log.warning("Failed getting object size: {}('{}')".format(e.__class__.__name__, str(e)))
        return size

    def get_object_metadata(self, obj: Any) -> dict:
        """
        Get the metadata of the remote object.
        The metadata is a dict containing the following keys: `name`, `size`.

        :param object obj: The remote object

        :return: A dict containing the metadata of the remote object
        """
        name_fields = ("name", "url", "key", "blob_name")
        metadata = {
            "size": self._get_object_size_bytes(obj),
            "name": next(filter(None, (getattr(obj, f, None) for f in name_fields)), None),
        }
        return metadata

    def verify_upload(
        self,
        folder_uri: str = "",
        raise_on_error: bool = True,
        log_on_error: bool = True,
    ) -> str:
        """
        Verify that this helper can upload files to a folder.

        An upload is possible iff:
            1. the destination folder is under the base uri of the url used to create the helper
            2. the helper has credentials to write to the destination folder

        :param folder_uri: The destination folder to test. Must be an absolute
            url that begins with the base uri of the url used to create the helper.
        :param raise_on_error: Raise an exception if an upload is not possible
        :param log_on_error: Log an error if an upload is not possible
        :return: True, if, and only if, an upload to folder_uri is possible.
        """

        folder_uri = self._canonize_url(folder_uri)

        folder_uri = self.conform_url(folder_uri, self._base_url)

        test_path = self._normalize_object_name(folder_uri)

        if self._scheme == _Boto3Driver.scheme:
            _Boto3Driver._test_bucket_config(
                self._conf,
                self._log,
                test_path=test_path,
                raise_on_error=raise_on_error,
                log_on_error=log_on_error,
            )
        elif self._scheme == _GoogleCloudStorageDriver.scheme:
            self._driver.test_upload(test_path, self._conf)

        elif self._scheme == "file":
            # Check path exists
            Path(test_path).mkdir(parents=True, exist_ok=True)
            # check path permissions
            Path(test_path).touch(exist_ok=True)

        return folder_uri

    def upload_from_stream(
        self,
        stream: Any,
        dest_path: str,
        extra: dict = None,
        retries: int = 1,
        return_canonized: bool = True,
    ) -> str:
        canonized_dest_path = self._canonize_url(dest_path)
        object_name = self._normalize_object_name(canonized_dest_path)
        extra = extra.copy() if extra else {}
        extra.update(self._extra)
        last_ex = None
        cb = UploadProgressReport.from_stream(stream, object_name, self._verbose, self._log)
        for i in range(max(1, int(retries))):
            try:
                self._driver.upload_object_via_stream(
                    iterator=stream,
                    container=self._container,
                    object_name=object_name,
                    callback=cb,
                    extra=extra,
                )
                last_ex = None
                break
            except Exception as ex:
                last_ex = ex
                # seek to beginning if possible
                # noinspection PyBroadException
                try:
                    stream.seek(0)
                except Exception:
                    pass

        if cb:
            cb.close(report_completed=not bool(last_ex))

        if last_ex:
            raise last_ex

        result_dest_path = canonized_dest_path if return_canonized else dest_path

        if self.scheme in _StorageHelper._quotable_uri_schemes:  # TODO: fix-driver-schema
            # quote link
            result_dest_path = quote_url(result_dest_path, _StorageHelper._quotable_uri_schemes)

        return result_dest_path

    def upload(
        self,
        src_path: str,
        dest_path: Optional[str] = None,
        extra: Optional[dict] = None,
        async_enable: bool = False,
        cb: Optional[Callable] = None,
        retries: int = 3,
        return_canonized: bool = True,
    ) -> Union[AsyncResult, str]:
        if not dest_path:
            dest_path = os.path.basename(src_path)

        canonized_dest_path = self._canonize_url(dest_path)
        dest_path = dest_path.replace("\\", "/")
        canonized_dest_path = canonized_dest_path.replace("\\", "/")

        result_path = canonized_dest_path if return_canonized else dest_path

        if cb and self.scheme in _StorageHelper._quotable_uri_schemes:  # TODO: fix-driver-schema
            # store original callback
            a_cb = cb

            # quote link
            def callback(result: bool) -> str:
                return a_cb(quote_url(result_path, _StorageHelper._quotable_uri_schemes) if result else result)

            # replace callback with wrapper
            cb = callback

        if async_enable:
            data = self._UploadData(
                src_path=src_path,
                dest_path=dest_path,
                canonized_dest_path=canonized_dest_path,
                extra=extra,
                callback=cb,
                retries=retries,
                return_canonized=return_canonized,
            )
            _StorageHelper._initialize_upload_pool()
            return _StorageHelper._upload_pool.apply_async(self._do_async_upload, args=(data,))
        else:
            res = self._do_upload(
                src_path=src_path,
                dest_path=dest_path,
                canonized_dest_path=canonized_dest_path,
                extra=extra,
                cb=cb,
                verbose=False,
                retries=retries,
                return_canonized=return_canonized,
            )
            if res:
                result_path = quote_url(result_path, _StorageHelper._quotable_uri_schemes)
            return result_path

    def list(self, prefix: Optional[str] = None, with_metadata: bool = False) -> List[Union[str, Dict[str, Any]]]:
        """
        List entries in the helper base path.

        Return a list of names inside this helper base path or a list of dictionaries containing
        the objects' metadata. The base path is determined at creation time and is specific
        for each storage medium.
        For Google Storage and S3 it is the bucket of the path.
        For local files it is the root directory.

        This operation is not supported for http and https protocols.

        :param prefix: If None, return the list as described above. If not, it
            must be a string - the path of a sub directory under the base path.
            the returned list will include only objects under that subdir.

        :param with_metadata: Instead of returning just the names of the objects, return a list of dictionaries
            containing the name and metadata of the remote file. Thus, each dictionary will contain the following
            keys: `name`, `size`.

        :return: The paths of all the objects in the storage base path under prefix or
            a list of dictionaries containing the objects' metadata.
            Listed relative to the base path.
        """
        if prefix:
            prefix = self._canonize_url(prefix)
            if prefix.startswith(self._base_url):
                prefix = prefix[len(self._base_url) :]
                if self._base_url != "file://":
                    prefix = prefix.lstrip("/")
            if self._base_url == "file://":
                prefix = prefix.rstrip("/")
                if prefix.startswith(str(self._driver.base_path)):
                    prefix = prefix[len(str(self._driver.base_path)) :]
            res = self._driver.list_container_objects(self._container, ex_prefix=prefix)
            result = [obj.name if not with_metadata else self.get_object_metadata(obj) for obj in res]

            if self._base_url == "file://":
                if not with_metadata:
                    result = [Path(f).as_posix() for f in result]
                else:
                    for metadata_entry in result:
                        metadata_entry["name"] = Path(metadata_entry["name"]).as_posix()
            return result
        else:
            return [
                obj.name if not with_metadata else self.get_object_metadata(obj)
                for obj in self._driver.list_container_objects(self._container)
            ]

    def download_to_file(
        self,
        remote_path: str,
        local_path: str,
        overwrite_existing: bool = False,
        delete_on_failure: bool = True,
        verbose: Optional[bool] = None,
        skip_zero_size_check: bool = False,
        silence_errors: bool = False,
        direct_access: bool = True,
    ) -> Optional[str]:
        def next_chunk(astream: Union[bytes, Iterable]) -> Tuple[Optional[bytes], Optional[Iterable]]:
            if isinstance(astream, (bytes, bytearray)):
                chunk = astream
                astream = None
            elif astream:
                try:
                    chunk = next(astream)
                except StopIteration:
                    chunk = None
            else:
                chunk = None
            return chunk, astream

        remote_path = self._canonize_url(remote_path)
        verbose = self._verbose if verbose is None else verbose

        tmp_remote_path = remote_path
        # noinspection PyBroadException
        try:
            tmp_remote_path = normalize_local_path(tmp_remote_path)
            if tmp_remote_path.exists():
                remote_path = "file://{}".format(str(tmp_remote_path))
        except Exception:
            pass
        # Check if driver type supports direct access:
        direct_access_path = self.get_driver_direct_access(remote_path)
        if direct_access_path and direct_access:
            return direct_access_path

        temp_local_path = None
        cb = None
        try:
            if verbose:
                self._log.info("Start downloading from {}".format(remote_path))
            # check for 0 sized files as well - we want to override empty files that were created
            # via mkstemp or similar functions
            if not overwrite_existing and Path(local_path).is_file() and Path(local_path).stat().st_size != 0:
                self._log.debug(
                    "File {} already exists, no need to download, thread id = {}".format(
                        local_path,
                        threading.current_thread().ident,
                    ),
                )

                return local_path
            if remote_path.startswith("file://"):
                Path(local_path).parent.mkdir(parents=True, exist_ok=True)
                # use remote_path, because direct_access_path might be None, because of access_rules
                # len("file://") == 7
                shutil.copyfile(remote_path[7:], local_path)
                return local_path
            # we download into temp_local_path so that if we accidentally stop in the middle,
            # we won't think we have the entire file
            temp_local_path = "{}_{}{}".format(local_path, time(), self._temp_download_suffix)
            obj = self.get_object(remote_path, silence_errors=silence_errors)
            if not obj:
                return None

            # object size in bytes
            total_size_mb = -1
            dl_total_mb = 0.0
            download_reported = False
            # chunks size is ignored and always 5Mb
            chunk_size_mb = 5

            # make sure we have the destination folder
            # noinspection PyBroadException
            Path(temp_local_path).parent.mkdir(parents=True, exist_ok=True)

            total_size_bytes = self.get_object_size_bytes(remote_path, silence_errors=silence_errors)
            if total_size_bytes is not None:
                total_size_mb = float(total_size_bytes) / (1024 * 1024)

            # if driver supports download with callback, use it (it might be faster)
            if hasattr(self._driver, "download_object"):
                # callback if verbose we already reported download start, no need to do that again
                cb = DownloadProgressReport(
                    total_size_mb,
                    verbose,
                    remote_path,
                    self._log,
                    report_start=True if verbose else None,
                )
                self._driver.download_object(obj, temp_local_path, callback=cb)
                download_reported = bool(cb.last_reported)
                dl_total_mb = cb.current_status_mb
            else:
                stream = self._driver.download_object_as_stream(obj, chunk_size_mb * 1024 * 1024)
                if stream is None:
                    raise ValueError("Could not download %s" % remote_path)
                with open(temp_local_path, "wb") as fd:
                    data, stream = next_chunk(stream)
                    while data:
                        fd.write(data)
                        data, stream = next_chunk(stream)

            if not skip_zero_size_check and Path(temp_local_path).stat().st_size <= 0:
                raise Exception("downloaded a 0-sized file")

            # if we are on Windows, we need to remove the target file before renaming
            # otherwise posix rename will overwrite the target
            if os.name != "posix":
                # noinspection PyBroadException
                try:
                    os.remove(local_path)
                except Exception:
                    pass

            # rename temp file to local_file
            # noinspection PyBroadException
            try:
                os.rename(temp_local_path, local_path)
            except Exception:
                # noinspection PyBroadException
                try:
                    os.unlink(temp_local_path)
                except Exception:
                    pass
                # file was downloaded by a parallel process, check we have the final output and delete the partial copy
                path_local_path = Path(local_path)
                if not path_local_path.is_file() or (not skip_zero_size_check and path_local_path.stat().st_size <= 0):
                    raise Exception("Failed renaming partial file, downloaded file exists and a 0-sized file")

            # report download if we are on the second chunk
            if cb:
                cb.close(
                    report_completed=True,
                    report_summary=verbose or download_reported,
                    report_prefix="Downloaded",
                    report_suffix="from {} , saved to {}".format(remote_path, local_path),
                )
            elif verbose or download_reported:
                self._log.info(
                    "Downloaded {:.2f} MB successfully from {} , saved to {}".format(
                        dl_total_mb, remote_path, local_path
                    )
                )
            return local_path
        except DownloadError:
            if cb:
                cb.close()
            raise
        except Exception as e:
            if cb:
                cb.close()
            self._log.error("Could not download {} , err: {} ".format(remote_path, e))
            if delete_on_failure and temp_local_path:
                # noinspection PyBroadException
                try:
                    os.remove(temp_local_path)
                except Exception:
                    pass
            return None

    def download_as_stream(
        self, remote_path: str, chunk_size: Optional[int] = None
    ) -> Optional[Generator[bytes, None, None]]:
        remote_path = self._canonize_url(remote_path)
        try:
            obj = self.get_object(remote_path)
            return self._driver.download_object_as_stream(
                obj, chunk_size=chunk_size, verbose=self._verbose, log=self.log
            )
        except DownloadError:
            raise
        except Exception as e:
            self._log.error("Could not download file : %s, err:%s " % (remote_path, str(e)))
            return None

    def download_as_nparray(self, remote_path: str, chunk_size: Optional[int] = None) -> Optional[numpy.ndarray]:
        try:
            stream = self.download_as_stream(remote_path, chunk_size)
            if stream is None:
                return

            # TODO: ugly py3 hack, please remove ASAP
            if not isinstance(stream, GeneratorType):
                import numpy as np

                return np.frombuffer(stream, dtype=np.uint8)
            else:
                import numpy as np

                return np.asarray(bytearray(b"".join(stream)), dtype=np.uint8)

        except Exception as e:
            self._log.error("Could not download file : %s, err:%s " % (remote_path, str(e)))

    def delete(self, path: str, silent: bool = False) -> bool:
        path = self._canonize_url(path)
        return self._driver.delete_object(self.get_object(path), silent=silent)

    def check_write_permissions(self, dest_path: Optional[str] = None) -> bool:
        # create a temporary file, then delete it
        base_url = dest_path or self._base_url
        dest_path = base_url + "/.clearml.{}.test".format(str(uuid.uuid4()))
        # do not check http/s connection permissions
        if dest_path.startswith("http"):
            return True

        try:
            self.upload_from_stream(stream=BytesIO(b"clearml"), dest_path=dest_path)
        except Exception:
            raise ValueError("Insufficient permissions (write failed) for {}".format(base_url))
        try:
            self.delete(path=dest_path)
        except Exception:
            raise ValueError("Insufficient permissions (delete failed) for {}".format(base_url))
        return True

    @classmethod
    def download_from_url(
        cls,
        remote_url: str,
        local_path: str,
        overwrite_existing: bool = False,
        skip_zero_size_check: bool = False,
    ) -> str:
        """
        Download a file from remote URL to a local storage

        :param remote_url: Remote URL. Example: https://example.com/image.jpg or s3://bucket/folder/file.mp4 etc.
        :param local_path: target location for downloaded file. Example: /tmp/image.jpg
        :param overwrite_existing: If True, and local_path exists, it will overwrite it, otherwise print warning
        :param skip_zero_size_check: If True, no error will be raised for files with zero bytes size.
        :return: local_path if download was successful.
        """
        helper = cls.get(remote_url)
        if not helper:
            return None
        return helper.download_to_file(
            remote_url,
            local_path,
            overwrite_existing=overwrite_existing,
            skip_zero_size_check=skip_zero_size_check,
        )

    def get_driver_direct_access(self, path: str) -> Optional[str]:
        """
        Check if the helper's driver has a direct access to the file

        :param str path: file path to check access to
        :return: Return the string representation of the file as path if have access to it, else None
        """
        path = self._canonize_url(path)
        return self._driver.get_direct_access(path)

    @classmethod
    def _canonize_url(cls, url: str) -> str:
        return cls._apply_url_substitutions(url)

    @classmethod
    def _apply_url_substitutions(cls, url: str) -> str:
        def replace_separator(_url: str, where: int, sep: str) -> str:
            return _url[:where] + _url[where:].replace(sep, os.sep)

        for index, rule in enumerate(cls._path_substitutions):
            if url.startswith(rule.registered_prefix):
                url = url.replace(
                    rule.registered_prefix,
                    rule.local_prefix,
                    1,  # count. str.replace() does not support keyword arguments
                )

                if rule.replace_windows_sep:
                    url = replace_separator(url, len(rule.local_prefix), "\\")

                if rule.replace_linux_sep:
                    url = replace_separator(url, len(rule.local_prefix), "/")

                break

        return url

    @classmethod
    def _resolve_base_url(cls, base_url: str) -> str:
        parsed = urlparse(base_url)
        if parsed.scheme == _Boto3Driver.scheme:
            conf = cls._s3_configurations.get_config_by_uri(base_url)
            bucket = conf.bucket
            if not bucket:
                parts = Path(parsed.path.strip("/")).parts
                if parts:
                    bucket = parts[0]
            return "/".join(x for x in ("s3:/", conf.host, bucket) if x)
        elif parsed.scheme == _AzureBlobServiceStorageDriver.scheme:
            conf = cls._azure_configurations.get_config_by_uri(base_url)
            if not conf:
                raise StorageError("Can't find azure configuration for {}".format(base_url))
            return str(furl(base_url).set(path=conf.container_name))
        elif parsed.scheme == _GoogleCloudStorageDriver.scheme:
            conf = cls._gs_configurations.get_config_by_uri(base_url)
            return str(furl(scheme=parsed.scheme, netloc=conf.bucket))
        elif parsed.scheme in _HttpDriver.schemes:
            for files_server in _Driver.get_file_server_hosts():
                if base_url.startswith(files_server):
                    return files_server
            return parsed.scheme + "://"
        else:  # if parsed.scheme == 'file':
            # if we do not know what it is, we assume file
            return "file://"

    @classmethod
    def conform_url(cls, folder_uri: str, base_url: str = None) -> str:
        if not folder_uri:
            return folder_uri
        _base_url = cls._resolve_base_url(folder_uri) if not base_url else base_url

        if not folder_uri.startswith(_base_url):
            prev_folder_uri = folder_uri
            if _base_url == "file://":
                folder_uri = str(Path(folder_uri).absolute())
                if folder_uri.startswith("/"):
                    folder_uri = _base_url + folder_uri
                elif platform.system() == "Windows":
                    folder_uri = "".join((_base_url, folder_uri))
                else:
                    folder_uri = "/".join((_base_url, folder_uri))

                cls._get_logger().debug(
                    "Upload destination {} amended to {} for registration purposes".format(prev_folder_uri, folder_uri)
                )
            else:
                raise ValueError("folder_uri: {} does not start with base url: {}".format(folder_uri, _base_url))

        return folder_uri

    def _absolute_object_name(self, path: str) -> str:
        """Returns absolute remote path, including any prefix that is handled by the container"""
        if not path.startswith(self.base_url):
            return self.base_url.rstrip("/") + "///" + path.lstrip("/")
        return path

    def _normalize_object_name(self, path: str) -> str:
        """Normalize remote path. Remove any prefix that is already handled by the container"""
        if path.startswith(self.base_url):
            path = path[len(self.base_url) :]
            if path.startswith("/") and os.name == "nt":
                path = path[1:]
        if self.scheme in (
            _Boto3Driver.scheme,
            _GoogleCloudStorageDriver.scheme,
            _AzureBlobServiceStorageDriver.scheme,
        ):
            path = path.lstrip("/")
        return path

    def _do_async_upload(self, data: _UploadData) -> str:
        assert isinstance(data, self._UploadData)
        return self._do_upload(
            data.src_path,
            data.dest_path,
            data.canonized_dest_path,
            extra=data.extra,
            cb=data.callback,
            verbose=True,
            retries=data.retries,
            return_canonized=data.return_canonized,
        )

    def _upload_from_file(self, local_path: str, dest_path: str, extra: Optional[dict] = None) -> Any:
        if not hasattr(self._driver, "upload_object"):
            with open(local_path, "rb") as stream:
                res = self.upload_from_stream(stream=stream, dest_path=dest_path, extra=extra)
        else:
            object_name = self._normalize_object_name(dest_path)
            extra = extra.copy() if extra else {}
            extra.update(self._extra)
            cb = UploadProgressReport.from_file(local_path, self._verbose, self._log)
            res = self._driver.upload_object(
                file_path=local_path,
                container=self._container,
                object_name=object_name,
                callback=cb,
                extra=extra,
            )
            if cb:
                cb.close()
        return res

    def _do_upload(
        self,
        src_path: str,
        dest_path: str,
        canonized_dest_path: str,
        extra: Optional[dict] = None,
        cb: Optional[Callable] = None,
        verbose: bool = False,
        retries: int = 1,
        return_canonized: bool = False,
    ) -> str:
        object_name = self._normalize_object_name(canonized_dest_path)
        if cb:
            try:
                cb(None)
            except Exception as e:
                self._log.error("Calling upload callback when starting upload: %s" % str(e))
        if verbose:
            msg = "Starting upload: {} => {}{}".format(
                src_path,
                (self._container.name if self._container.name.endswith("/") else self._container.name + "/")
                if self._container and self._container.name
                else "",
                object_name,
            )
            if object_name.startswith("file://") or object_name.startswith("/"):
                self._log.debug(msg)
            else:
                self._log.info(msg)
        last_ex = None
        for i in range(max(1, int(retries))):
            try:
                if not self._upload_from_file(local_path=src_path, dest_path=canonized_dest_path, extra=extra):
                    # retry if failed
                    last_ex = ValueError("Upload failed")
                    continue
                last_ex = None
                break
            except Exception as e:
                last_ex = e

        if last_ex:
            self._log.error("Exception encountered while uploading %s" % str(last_ex))
            if cb:
                try:
                    cb(False)
                except Exception as e:
                    self._log.warning("Exception on upload callback: %s" % str(e))
            raise last_ex

        if verbose:
            self._log.debug("Finished upload: %s => %s" % (src_path, object_name))
        if cb:
            try:
                cb(canonized_dest_path if return_canonized else dest_path)
            except Exception as e:
                self._log.warning("Exception on upload callback: %s" % str(e))

        return canonized_dest_path if return_canonized else dest_path

    def get_object(self, path: str, silence_errors: bool = False) -> Any:
        """
        Gets the remote object stored at path. The data held by the object
        differs depending on where it is stored.

        :param str path: the path where the remote object is stored
        :param bool silence_errors: Silence errors that might occur
            when fetching the remote object

        :return: The remote object
        """
        path = self._canonize_url(path)
        object_name = self._normalize_object_name(path)
        try:
            return self._driver.get_object(
                container_name=self._container.name if self._container else "",
                object_name=object_name,
            )
        except ConnectionError:
            raise DownloadError
        except Exception as e:
            if not silence_errors:
                self.log.warning("Storage helper problem for {}: {}".format(str(object_name), str(e)))
            return None

    @staticmethod
    def _initialize_upload_pool() -> None:
        if not _StorageHelper._upload_pool or _StorageHelper._upload_pool_pid != os.getpid():
            _StorageHelper._upload_pool_pid = os.getpid()
            _StorageHelper._upload_pool = ThreadPool(processes=1)

    @staticmethod
    def close_async_threads() -> None:
        if _StorageHelper._upload_pool:
            pool = _StorageHelper._upload_pool
            _StorageHelper._upload_pool = None
            # noinspection PyBroadException
            try:
                pool.terminate()
                pool.join()
            except Exception:
                pass

    def exists_file(self, remote_url: str) -> bool:
        remote_url = self._canonize_url(remote_url)
        object_name = self._normalize_object_name(remote_url)
        return self._driver.exists_file(
            container_name=self._container.name if self._container else "",
            object_name=object_name,
        )

    @classmethod
    def sanitize_url(cls, remote_url):
        base_url = cls._resolve_base_url(remote_url)
        if base_url != 'file://' or remote_url.startswith("file://"):
            return remote_url
        absoulte_path = os.path.abspath(remote_url)
        return base_url + absoulte_path


CLEARML_SECONDARY_CACHE_DIR = EnvEntry("CLEARML_SECONDARY_CACHE_DIR", type=str)


# configuration constants
_DISK_STRATEGY_SECTION = "disk_space_file_size_strategy"
_CONFIG_MISSING = object()


# Simplified urlsplit tailored for ClearML usage
def fast_urlsplit(remote_path, *_, **__):
    """Return (scheme, netloc, path, query, fragment) without pulling in urllib."""
    scheme, sep, rest = remote_path.partition('://')
    if not sep:
        scheme, rest = '', remote_path

    netloc, sep, rest = rest.partition('/')
    if not sep:
        netloc, rest = '', netloc

    path_section, sep, fragment = rest.partition('#')
    if not sep:
        fragment = ''

    path, sep, query = path_section.partition('?')
    if not sep:
        query = ''

    return scheme, netloc, path, query, fragment


class StorageHelper(_StorageHelper):
    """ Cached File Storage helper.
        Overloads the standard StorageHelper and provides local caching support for calls to download_to_file().
    """
    use_disk_space_file_size_strategy = deferred_config(
        "storage.cache.{}.enabled".format(_DISK_STRATEGY_SECTION),
        False,
        transform=bool,
    )

    class CacheConfigs(object):
        configs = {}

        class Defaults(object):
            cache_name = None
            cache_base_dir = None
            cleanup_seconds_threshold = None
            max_cleanup_attempts = None
            cache_size_max_used_bytes = None
            cache_size_min_free_bytes = None
            cache_cleanup_margin_pct = None
            cache_zero_file_size_check = None
            cache_file_entry_lookup_size = None
            cache_matchers = None
            direct_access_matchers = None

            @classmethod
            def init(cls):
                cls.cache_name = "local"
                cls.cache_base_dir = _get_cache_dir()
                cls.cleanup_seconds_threshold = _config("cleanup_seconds_threshold", default=20.0)
                cls.max_cleanup_attempts = _config("max_cleanup_attempts", default=5)
                cls.cache_size_max_used_bytes = _disk_config("size", "max_used_bytes", default=-1)
                cls.cache_size_min_free_bytes = _disk_config("size", "min_free_bytes", default="10GB")
                cls.cache_cleanup_margin_pct = _disk_config("size", "cleanup_margin_percent", default=0.05)
                cls.cache_zero_file_size_check = _disk_config("zero_file_size_check", default=False)
                cls.cache_file_entry_lookup_size = _disk_config("file_entry_lookup_size", default=0)
                cls.cache_matchers = _config("enable", default=[{"url": "*"}])
                cls.direct_access_matchers = config.get("storage.direct_access", [{"url": "file://"}])

        def __init__(self, cache_name=None):
            self.cache_name = cache_name if cache_name is not None else self.Defaults.cache_name
            self.cache_base_dir = (_get_cache_dir(cache_name=cache_name) or self.Defaults.cache_base_dir) / "storage"
            self.cleanup_seconds_threshold = _config(
                "cleanup_seconds_threshold", default=self.Defaults.cleanup_seconds_threshold, cache_name=cache_name
            )
            self.max_cleanup_attempts = int(
                max(
                    1,
                    _config("max_cleanup_attempts", default=self.Defaults.max_cleanup_attempts, cache_name=cache_name),
                )
            )
            self.cache_size_max_used_bytes = _disk_config_human_size(
                "size", "max_used_bytes", default=self.Defaults.cache_size_max_used_bytes, cache_name=cache_name
            )
            self.cache_size_min_free_bytes = _disk_config_human_size(
                "size", "min_free_bytes", default=self.Defaults.cache_size_min_free_bytes, cache_name=cache_name
            )
            self.cache_cleanup_margin_pct = _disk_config_percentage(
                "size", "cleanup_margin_percent", default=self.Defaults.cache_cleanup_margin_pct, cache_name=cache_name
            )
            self.cache_zero_file_size_check = _disk_config(
                "zero_file_size_check", default=self.Defaults.cache_zero_file_size_check, cache_name=cache_name
            )
            self.cache_file_entry_lookup_size = _disk_config(
                "file_entry_lookup_size", default=self.Defaults.cache_file_entry_lookup_size, cache_name=cache_name
            )
            self.cache_matchers = [
                get_config_object_matcher(**entry)
                for entry in _config("enable", default=self.Defaults.cache_matchers, cache_name=cache_name)
            ]
            self.direct_access_matchers = [
                get_config_object_matcher(**entry)
                for entry in config.get("storage.direct_access", self.Defaults.direct_access_matchers)
            ]

        @classmethod
        def get(cls, cache_name=None):
            # We consider the absence of the cache_name to refer to the local cache
            return cls.configs.get(cache_name if cache_name is not None else cls.Defaults.cache_name)

        @classmethod
        def create(cls, cache_name=None):
            if cache_name not in cls.configs:
                # We consider the absence of the cache_name to refer to the local cache
                cls.configs[cache_name if cache_name is not None else cls.Defaults.cache_name] = cls(
                    cache_name=cache_name
                )

    _helpers = {}  # cache of helper instances
    _pre_delete_hooks = {}

    _has_secondary = False
    _inited_configs = False

    common_cache_matchers = []

    class _NotDirectOrCached(Exception):
        pass

    @classmethod
    def init_configs(cls):
        if cls._inited_configs:
            return
        cls._inited_configs = True
        cls.CacheConfigs.Defaults.init()
        names = [None, "secondary"]
        for name in names:
            if _get_cache_dir(cache_name=name):
                cls.CacheConfigs.create(cache_name=name)
                if name == "secondary":
                    cls._has_secondary = True

    @property
    def cache_dir(self):
        """
        :return: Base cache folder used for all local copies
        """
        return self.cache_base_dir

    def __init__(self, *args, **kwargs):
        if not bool(self.use_disk_space_file_size_strategy):
            return super(StorageHelper, self).__init__(*args, **kwargs)

        self.__class__.init_configs()

        configs = kwargs.get("configs")
        if not configs:
            configs = self.__class__.CacheConfigs.get()

        super(StorageHelper, self).__init__(*args, **kwargs)

        self._cleanup_seconds_threshold = configs.cleanup_seconds_threshold
        self._cache_clean_in_progress = Lock()
        self._cache_size_lock = ForkSafeRLock()
        self._cache_size = 0
        self._synced_cache_time = None
        self._cache_file_entry_lookup_table = {}

        self._cache_adjustment_lock = ForkSafeRLock()
        self._path_locks = {}
        self._path_locks_cnt = {}
        self._reused_locks = []
        self._reused_min_locks = 32

        self.clean_cache_executor = ThreadPoolExecutor(max_workers=1)
        self.max_cleanup_attempts = configs.max_cleanup_attempts
        self.cache_base_dir = configs.cache_base_dir
        self.cache_size_max_used_bytes = configs.cache_size_max_used_bytes
        self.cache_size_min_free_bytes = configs.cache_size_min_free_bytes
        self.cache_cleanup_margin_pct = configs.cache_cleanup_margin_pct
        self.cache_zero_file_size_check = configs.cache_zero_file_size_check
        self.cache_file_entry_lookup_size = configs.cache_file_entry_lookup_size
        if self.cache_file_entry_lookup_size <= 0:
            self._cache_file_entry_lookup_table = None
        self.direct_access_matchers = configs.direct_access_matchers
        self.cache_matchers = configs.cache_matchers

        try:
            self.cache_base_dir.mkdir(parents=True, exist_ok=True)
        except OSError as ex:
            raise StorageError(
                "Failed creating cache folder {}: {}".format(self.cache_base_dir, ex)
            )

    @classmethod
    def register_pre_delete_hook(cls, client_name, hook, override=False):
        if (client_name in cls._pre_delete_hooks) and not override:
            return

        if not callable(hook):
            raise TypeError("Pre-delete hook must be a callable object")

        cls._pre_delete_hooks[client_name] = hook

    @staticmethod
    def register_cache_matcher(**kwargs):
        """
        Add a matcher for files that should use the cache.
        Files that don't match any matcher will cause an exception.
        The matcher will be used for all instances of StorageHelper.

        :param kwargs: The matcher argument. Currently only content_type and url
            are supported. The values are glob like patterns.
        """
        StorageHelper.common_cache_matchers.append(get_config_object_matcher(**kwargs))

    def register_cache_matcher_for_instance(self, **kwargs):
        """
        Add a matcher for files that should use the cache.
        Files that don't match any matcher will cause an exception.
        The matcher will be used only for the current instance of StorageHelper.

        :param kwargs: The matcher argument. Currently only content_type and url
            are supported. The values are glob like patterns.
        """
        self.cache_matchers.append(get_config_object_matcher(**kwargs))

    def download_as_stream(self, remote_path, chunk_size=None, force_content_type=None, force_cache=False):
        """
        Download remote_path as a stream using the cache. Cache is used if caching is enabled for the remote_path, based
            on the cache configuration. Direct access paths (which are mapped directly to their remote path) are
            returned as they are (assuming they exist, otherwise None is returned)

        :param remote_path: see StorageHelper.download_as_stream()
        :param chunk_size: download chunk size (request size)
        :param force_content_type: force using a specific content type when checking if caching is enabled for this path
            (otherwise content type is guessed based on the path)
        :param force_cache: force caching for this path, regardless of configuration
        :return:
        """
        if not bool(self.use_disk_space_file_size_strategy):
            return super().download_as_stream(remote_path=remote_path, chunk_size=chunk_size)

        try:
            local_file = self._multi_cache_download(
                remote_path,
                force_content_type,
                force_cache=force_cache,
            )

            try:
                result = Path(local_file).open('rb').read()
            except OSError:
                # if the file was just downloaded and we could not read it,
                # we probably deleted it in cache cleanup
                # try one more time to access it
                result = None

            # on the second try, we must raise exception if we still can't access the file
            if result is None:
                local_file = self._multi_cache_download(
                    remote_path,
                    force_content_type,
                    force_cache=force_cache,
                )

                result = Path(local_file).open('rb').read()

            return result
        except self._NotDirectOrCached:
            return super(StorageHelper, self).download_as_stream(remote_path, chunk_size)

    def download_to_file(
        self,
        remote_path,
        local_path=None,
        overwrite_existing=False,
        delete_on_failure=True,
        verbose=None,
        force_content_type=None,
        force_cache=False,
        skip_zero_size_check=False,
        silence_errors=False,
        direct_access=False
    ):
        """
        Download remote_path to a local, cached file. Cache is used if caching is enabled for the remote_path, based on
            the cache configuration. Direct access paths (which are mapped directly to their remote path)
            are returned as they are.

        :param remote_path: see StorageHelper.download_to_file()
        :param local_path: local path where the file should be stored. If the file is not cached, this parameter
            is required. The download location is based on the remote path (for example, s3://bucket/path/to/file
            will be stored in /<cache-dir>/path/to/file)
        :param overwrite_existing: overwrite an existing cached file if exists
        :param delete_on_failure: see StorageHelper.download_to_file()
        :param force_content_type: force using a specific content type when checking if caching is enabled for this path
            (otherwise content type is guessed based on the path)
        :param verbose: Override default verbosity (boolean).
        :param force_cache: force caching for this path, regardless of configuration
        :param skip_zero_size_check: If True will return also files size zero bytes.
        :param silence_errors: If True, don't log erors
        :param direct_access: The parameter is not used, and is present here only for compatibility reasons
        :return:
        """
        if not bool(self.use_disk_space_file_size_strategy):
            return super().download_to_file(
                remote_path=remote_path,
                local_path=local_path,
                overwrite_existing=overwrite_existing,
                delete_on_failure=delete_on_failure,
                verbose=verbose,
                skip_zero_size_check=skip_zero_size_check,
                silence_errors=silence_errors,
                direct_access=direct_access,
            )

        try:
            download_path = self._multi_cache_download(
                remote_path=remote_path,
                force_content_type=force_content_type,
                overwrite_existing=overwrite_existing,
                verbose=verbose,
                force_cache=force_cache,
                delete_on_failure=delete_on_failure,
                skip_size_check=skip_zero_size_check,
                silence_errors=silence_errors,
            )
            if local_path:
                # noinspection PyBroadException
                try:
                    helper = super().get(download_path)
                    helper.download_to_file(
                        download_path,
                        local_path,
                        overwrite_existing=overwrite_existing,
                        delete_on_failure=delete_on_failure,
                        verbose=verbose,
                        skip_zero_size_check=skip_zero_size_check,
                        silence_errors=silence_errors,
                        direct_access=False,
                    )
                except Exception:
                    if not silence_errors:
                        self._log.error(
                            "Could not download {} to local path. Note that the download to the cache succeeded".format(
                                remote_path
                            )
                        )
            return download_path
        except self._NotDirectOrCached:
            if not local_path:
                raise ValueError("Object is not cached, local_path is required (%s)" % remote_path)
            return super(StorageHelper, self).download_to_file(
                remote_path, local_path, overwrite_existing, delete_on_failure
            )

    @classmethod
    def get(cls, url, logger=None, **kwargs):
        """
        Get a cached storage helper instance for the given URL

        :return: StorageHelper instance
        """
        return super(StorageHelper, cls).get(url, logger=logger, **kwargs)

    def get_free_bytes(self):
        stats = disk_usage(str(self.cache_base_dir))
        return stats.free

    @classmethod
    def get_local_copy(cls, remote_url, skip_zero_size_check=False, force_download=False):
        """
        Download a file from remote URL to a local storage, and return path to local copy,

        :param remote_url: Remote URL. Example: https://example.com/file.jpg s3://bucket/folder/file.mp4 etc.
        :return: Path to local copy of the downloaded file. None if error occurred.
        """
        if not bool(cls.use_disk_space_file_size_strategy):
            return super().get_local_copy(remote_url=remote_url, skip_zero_size_check=skip_zero_size_check)
        helper = cls.get(remote_url)
        if not helper:
            return None
        return helper.download_to_file(remote_url, overwrite_existing=force_download)

    def get_direct_access(self, remote_path):
        # type: (str) -> Optional[str]
        """
        Return a direct access path to the requested file.
        If remote_url is not a direct access url, return None
        :param remote_path: URL to local file
        :return: string path to existing direct access file
        """
        direct_path, _, _ = self._get_direct_access_path(remote_path)
        if not direct_path or not os.path.isfile(direct_path):
            return None
        return direct_path

    def _get_direct_access_path(self, remote_path):
        remote_path = self._canonize_url(remote_path)
        absolute_url = self._absolute_object_name(remote_path)
        direct_access = any(True for matcher in self.direct_access_matchers if matcher(url=absolute_url))
        if not direct_access:
            return None, remote_path, absolute_url
        # we do not cache locally stored files (local drive or network mapped)
        scheme, netloc, filename, query, fragment = fast_urlsplit(remote_path, allow_fragments=False)
        # on windows without "file://" prefix, netloc is "C:" make sure we add / at the end to get "C:/"
        return os.path.join('/', (netloc+'/') if netloc else '', filename.lstrip('/')), remote_path, absolute_url

    def _get_cache_path(self, canonized_url, force_content_type=None, base_local_path=None, force_cache=False):
        # canonized_url = self._canonize_url(remote_path)

        scheme, netloc, filename, query, fragment = fast_urlsplit(canonized_url, allow_fragments=False)
        if query:
            filename = filename.split('/')
            filename[-1] = '{}.{}'.format(hashlib.md5(query.encode()).hexdigest(), filename[-1])
            filename = '/'.join(filename if filename[0] else filename[1:])
        else:
            filename = str(Path(filename.lstrip("/")))

        if scheme == 'http':
            scheme = 'https'
        netloc = netloc.replace(":", ".") if netloc else ''
        # notice that os.path will handle '/' in windows as if it was backslash '\\'
        return os.path.join(
            str(self.cache_dir),
            self.__class__._safe_os_join(base_local_path or "", scheme or "", netloc, filename),
        )

    def _multi_cache_download(
        self,
        remote_path,
        force_content_type,
        overwrite_existing=False,
        verbose=None,
        delete_on_failure=True,
        force_cache=False,
        skip_size_check=False,
        silence_errors=False
    ):
        # NOTE: In this function, we mention "downloading" a file. Downloading means:
        # checking if the file exists or if is should be overwritten, then fetching the
        # content of the downloaded file to a temp file, then atomically renaming that file
        # to the target location

        # If we don't have a secondary cache, we will just download it
        if not self.__class__._has_secondary:
            return self._download(
                remote_path,
                force_content_type,
                overwrite_existing=overwrite_existing,
                verbose=verbose,
                delete_on_failure=delete_on_failure,
                force_cache=force_cache,
                skip_size_check=skip_size_check,
                silence_errors=silence_errors
            )

        # We have a secondary cache. Create it
        secondary_cache = self.__class__.get(
            remote_path, configs=self.__class__.CacheConfigs.get("secondary")
        )
        # Check if the file exists in the secondary cache and download it from there
        _, canonized_url, _ = secondary_cache._get_direct_access_path(remote_path)
        secondary_cache_path = secondary_cache._get_cache_path(
            canonized_url, force_content_type, force_cache=force_cache
        )
        if (
            not overwrite_existing
            and os.path.exists(secondary_cache_path)
            and (skip_size_check or os.path.getsize(secondary_cache_path) != 0)
        ):
            if verbose:
                self._log.info("{} found in the secondary cache. Downloading to local cache".format(remote_path))
            secondary_to_local_cache_fetcher = self.__class__.get(secondary_cache_path)
            local_path = os.path.join(
                str(secondary_to_local_cache_fetcher.cache_dir),
                os.path.relpath(secondary_cache_path, str(secondary_cache.cache_dir)),
            )
            return secondary_to_local_cache_fetcher._download(
                secondary_cache_path,
                force_content_type,
                overwrite_existing=overwrite_existing,
                verbose=verbose,
                delete_on_failure=delete_on_failure,
                force_cache=force_cache,
                skip_size_check=skip_size_check,
                allow_direct_access=False,
                cache_path=local_path
            )

        # We got here, which means don't have the file in the secondary cache
        # or we need to overwrite it
        # We first download the file to the local cache
        local_path = self._download(
            remote_path,
            force_content_type,
            overwrite_existing=overwrite_existing,
            verbose=verbose,
            delete_on_failure=delete_on_failure,
            force_cache=force_cache,
            skip_size_check=skip_size_check,
            allow_direct_access=False,
            silence_errors=silence_errors
        )

        # Then download from the local cache to the secondary cache
        # Note that the cache path needs to be calculated manually,
        # to have the same relative paths in both caches
        secondary_cache = self.__class__.get(
            local_path, configs=self.__class__.CacheConfigs.get("secondary")
        )
        secondary_cache_path = os.path.join(
            str(secondary_cache.cache_dir), os.path.relpath(local_path, str(self.cache_dir))
        )
        secondary_cache._download(
            local_path,
            force_content_type,
            overwrite_existing=overwrite_existing,
            verbose=verbose,
            delete_on_failure=delete_on_failure,
            force_cache=force_cache,
            skip_size_check=skip_size_check,
            allow_direct_access=False,
            cache_path=secondary_cache_path,
            silence_errors=silence_errors
        )

        # We return the path to the file in the local cache
        return local_path

    def _download(
        self,
        remote_path,
        force_content_type,
        overwrite_existing=False,
        verbose=None,
        delete_on_failure=True,
        force_cache=False,
        skip_size_check=False,
        allow_direct_access=True,
        cache_path=None,
        silence_errors=False
    ):
        direct_path, canonized_url, absolute_url = self._get_direct_access_path(remote_path)
        if direct_path and allow_direct_access:
            if not os.path.isfile(direct_path) and not silence_errors:
                self._log.error("Direct access path does not exist " +
                                ("%s" % remote_path if remote_path == direct_path else
                                 "(%s => %s)" % (remote_path, direct_path)))
                return None
            return str(direct_path)
        # don't bother checking, look for the entry on the cache lookup table
        if self._cache_file_entry_lookup_table is not None:
            if not cache_path:
                cache_path = self._cache_file_entry_lookup_table.get(canonized_url)
            if cache_path and os.path.isfile(cache_path):
                self._update_cache_file(cache_path, log=self.log)
                return str(cache_path)
        if not cache_path:
            cache_path = self._get_cache_path(canonized_url, force_content_type, force_cache=force_cache)
        if cache_path:
            # update file lookup cache table
            if self._cache_file_entry_lookup_table is not None:
                if len(self._cache_file_entry_lookup_table) >= self.cache_file_entry_lookup_size:
                    self._cache_file_entry_lookup_table.clear()
                self._cache_file_entry_lookup_table[canonized_url] = cache_path
            # return the cached file is it's valid
            if not overwrite_existing and os.path.isfile(cache_path) and (
                    not self.cache_zero_file_size_check or os.path.getsize(cache_path) > 0):
                self._update_cache_file(cache_path, log=self.log)
                return str(cache_path)

            if not force_cache:
                # check if we can actually download the file (i.e. a matcher might say we cannot
                absolute_url = self._absolute_object_name(remote_path)
                content_type = (force_content_type or mimetypes.guess_type(remote_path)[0])
                do_cache = force_cache or any(
                    True
                    for matcher in (self.cache_matchers + self.__class__.common_cache_matchers)
                    if matcher(content_type=content_type, url=absolute_url)
                )
                if not do_cache:
                    raise self._NotDirectOrCached()

            # make needed dirs
            cache_path = Path(cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            # make sure the cache is clean enough
            self._clean_cache()
            # try to download to cache
            key = str(cache_path)

            # prevent multiple downloads of the same file
            with self._cache_adjustment_lock:
                locks = self._path_locks
                if key not in locks:
                    # atomic access
                    try:
                        locks[key] = self._reused_locks.pop(0)
                    except IndexError:
                        locks[key] = ForkSafeRLock()
                lock = locks[key]
                self._path_locks_cnt[key] = 1 + self._path_locks_cnt.get(key, 0)

            try:
                lock.acquire()
                # check if we still need to download the file (someone might already have done that)
                if not overwrite_existing and cache_path.is_file() and cache_path.stat().st_size > 0:
                    self._update_cache_file(cache_path, log=self.log)
                    return str(cache_path)

                # download the file
                # at this point, we no longer allow direct access because
                # the path didn't match the `direct_access` config
                # (checked at the beggining of this function)
                res = super(StorageHelper, self).download_to_file(
                    remote_path=remote_path,
                    local_path=key,
                    overwrite_existing=overwrite_existing,
                    delete_on_failure=delete_on_failure,
                    verbose=verbose,
                    skip_zero_size_check=skip_size_check,
                    direct_access=False,
                )
                if res:
                    # Update cache size according to file size
                    self._update_cache_file(cache_path, downloaded=True, log=self.log)
                    return str(cache_path)
                return None
            finally:
                # remove lock from list
                with self._cache_adjustment_lock:
                    self._path_locks_cnt[key] -= 1
                    if not self._path_locks_cnt[key]:
                        self._path_locks_cnt.pop(key, None)
                        try:
                            locks.pop(key, None)
                        except (KeyError, Exception):
                            pass
                        self._reused_locks.append(lock)

                # now release the lock, if someone was waiting on the lock
                # by the time they get it, the file should be in cache
                lock.release()

        # Not direct access or cached, raise this in order to allow callers to differentiate between this state and
        # an error that returns None (maintains StorageHelper behavior)
        raise self._NotDirectOrCached()

    def _clean_cache(self):
        # if this is the first time we need to clean the cache (first download) we should scan the current cached files
        if self._synced_cache_time is not None:
            # Check if we need to clean the cache
            if ((self.cache_size_max_used_bytes <= 0 or self._cache_size < self.cache_size_max_used_bytes) and
                    (self.cache_size_min_free_bytes <= 0 or self.get_free_bytes() > self.cache_size_min_free_bytes)):
                return

        # check if someone is already cleaning the cache
        if self._cache_clean_in_progress.acquire(block=False):
            try:
                future = self.clean_cache_executor.submit(self._do_clean_cache, self.log)
                future.result(timeout=0)
            except (TimeoutError, CancelledError, Exception):
                pass

    def _update_cache_file(self, file_path, downloaded=False, log=None):
        """ Update cache size with integer value or file size (if a filename is provided) """
        # Update file access

        # it's actually slower to push into the ThreadPool than to touch the file ourselves.
        # cls.cache_touch_executor.submit(cls._do_touch_file, file_path)

        file_path = str(file_path)

        # update the access time of the file
        os.utime(file_path, (time(), os.path.getmtime(file_path)))

        # if downloaded (i.e. added to cache), update total cache size
        # Notice! we need to have the total size and cached files in sync, so we lock both
        if downloaded:
            if log:
                log.debug('Updating cache size for %s' % file_path)
            try:
                file_size = os.path.getsize(file_path)
            except OSError as e:
                file_size = 0
                if log:
                    log.warning('Failed updating file size for %s: %s' % (file_path, e))
            # update the total cache size
            self._update_cache_size(file_size)

    def _do_clean_cache(self, log=None):
        """ Clean cache until size constraints are satisfied. Not error is raised in case constraints are not satisfied,
            we'll just do our very best.
        """
        try:
            used_bytes_limit_enabled = self.cache_size_max_used_bytes > 0
            free_bytes_limit_enabled = self.cache_size_min_free_bytes > 0

            if log:
                log.debug('Checking cache size: size=%s, limit=%s' % (
                    self._cache_size if self._cache_size is not None else "N/A",
                    self.cache_size_max_used_bytes
                ))

            # last cache sync/cleanup
            first_cleanup_scan = self._synced_cache_time is None
            self._synced_cache_time = time()

            # check if we need to run the cache cleanup
            def get_cache_size_str():
                format_size(self._cache_size) if self._cache_size is not None else "N/A"

            if first_cleanup_scan and used_bytes_limit_enabled:
                # if this is the first time we need to clean the cache (first download)
                # we should need to scan the files if we have a max cache size limit
                pass
            elif ((not used_bytes_limit_enabled or self._cache_size < self.cache_size_max_used_bytes)
                    and (not free_bytes_limit_enabled or self.get_free_bytes() > self.cache_size_min_free_bytes)):
                # cache limit not exceeded, leave
                return

            # Calculate cleanup targets
            max_used_bytes_target = self.cache_size_max_used_bytes
            min_free_bytes_target = self.cache_size_min_free_bytes
            if self.cache_cleanup_margin_pct and used_bytes_limit_enabled:
                max_used_bytes_target *= 1.0 - min(0.99, max(0, self.cache_cleanup_margin_pct))
            if self.cache_cleanup_margin_pct and free_bytes_limit_enabled:
                min_free_bytes_target *= 1.0 + min(0.99, max(0, self.cache_cleanup_margin_pct))

            if log:
                log.warning(
                    "Cleaning cache: max_used_bytes=%s, min_free_bytes=%s, free_space=%s"
                    % (
                        format_size(max_used_bytes_target),
                        format_size(min_free_bytes_target),
                        format_size(self.get_free_bytes()),
                    )
                )
                log.warning(
                    "Cache cleanup started (clearing %s)"
                    % (
                        "scan"
                        if first_cleanup_scan
                        else format_size(self._cache_size - max_used_bytes_target)
                        if used_bytes_limit_enabled
                        else format_size(self.cache_size_min_free_bytes - self.get_free_bytes())
                    )
                )

            # Calc and store cache size
            cleanup_session_timestamp = time()
            sorted_cached_files, cache_size = self._update_cache_dir_size()
            with self._cache_size_lock:
                self._cache_size = cache_size

            # Delete files until targets are reached
            while (sorted_cached_files and
                   ((used_bytes_limit_enabled and self._cache_size >= max_used_bytes_target)
                    or (free_bytes_limit_enabled and self.get_free_bytes() <= min_free_bytes_target))):

                # Delete a single file (allow for several failures)
                time_stamp, candidate, size = heappop(sorted_cached_files)

                # check if the last access time is too close to when we started this session,
                # we should quite before we start deleting our own downloads
                if cleanup_session_timestamp - time_stamp < self._cleanup_seconds_threshold:
                    if log:
                        log.warning('Leaving cache cleanup [%d remaining] before starting '
                                    'to clean current downloads, file candidate accessed %.3f sec ago' %
                                    (len(sorted_cached_files), cleanup_session_timestamp - time_stamp))
                    break

                # check if we are currently downloading this file
                # (no need for thread safety, worst case we are not synced)
                if candidate in self._path_locks:
                    continue
                elif candidate.endswith(self._temp_download_suffix):
                    # check if name is in path locks (it will be the non-temp filename)
                    if [f for f in self._path_locks.keys() if f.startswith(candidate)]:
                        continue

                for client_name in self._pre_delete_hooks:
                    try:
                        self._pre_delete_hooks[client_name](candidate)
                    except Exception as exc:
                        if log:
                            log.error("Pre delete hook of client {} failed: {}".format(client_name, exc))

                try:
                    # Remove file and update cache size
                    os.remove(candidate)
                except FileNotFoundError:
                    # probably someone else deleted it.
                    pass
                except Exception as e:
                    if log:
                        log.debug("Failed cleaning file from cache: %s" % str(e))

                # No need to lock cache_size we already locked all cache access
                self._update_cache_size(-size)

            if log:
                if free_bytes_limit_enabled and self.get_free_bytes() <= self.cache_size_min_free_bytes:
                    log.warning("Cache free bytes limit not satisfied after cleaning entire cache (limit is %s)"
                                % format_size(self.cache_size_min_free_bytes))
                log.warning("Cache cleanup done, current size %s" % (format_size(self._cache_size)))
        finally:
            self._cache_clean_in_progress.release()

    def _update_cache_dir_size(self):
        # we should only actually run this update once
        total_size = 0
        cached_files = []
        heapify(cached_files)

        start_path = str(self.cache_base_dir)
        for dirpath, dirnames, filenames in os.walk(start_path):
            for f in filenames:
                # remove session file from cache list, so we do not delete it
                if dirpath == start_path and f == SESSION_CACHE_FILE:
                    continue

                fp = os.path.join(dirpath, f)
                try:
                    atime = os.path.getatime(fp)
                    size = os.path.getsize(fp)
                    heappush(cached_files, (atime, fp, size))
                    total_size += size
                except OSError:
                    continue

        return cached_files, total_size

    def _update_cache_size(self, value):
        """ Update cache size with integer value or file size (if a filename is provided) """
        if not value:
            return
        try:
            self._cache_size_lock.acquire()
            if self._cache_size is not None:
                self._cache_size += value
        finally:
            self._cache_size_lock.release()

    @staticmethod
    def _do_touch_file(file_path):
        Path(file_path).touch()

    @staticmethod
    def _safe_os_join(*paths):
        return os.path.join(*[StorageHelper._limit_folder_name(p) for p in paths])

    @staticmethod
    def _limit_folder_name(folder_path):
        # type (Optional[str]) -> Optional[str]
        # make sure if the folder names are too large we encode them within the file system 255 characters limit
        # assume / separator in folder folder_path

        folder_path = quote(folder_path, safe="/\\=[]!~()#@$&%'+,;+") if folder_path else folder_path

        # optimize, nothing to do if the entire path is short enough
        if not folder_path or len(folder_path) <= 255:
            return folder_path

        sep = '/'
        return sep.join(
            [f if len(f) <= 255 else '{}.{}'.format(hashlib.md5(f.encode()).hexdigest(), f[-200:])
             for f in folder_path.split(sep)])


class _FileStorageDriverDiskSpaceFileSizeStrategy(_FileStorageDriver):
    def get_direct_access(self, remote_path: str, **_: Any) -> Optional[str]:
        return None


class StorageHelperDiskSpaceFileSizeStrategy(StorageHelper):
    use_disk_space_file_size_strategy = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self._driver, _FileStorageDriver):
            self._driver = _FileStorageDriverDiskSpaceFileSizeStrategy(self._driver.base_path)


def _config(*keys, default=None, cache_name=None):
    storage_key = "storage.cache"
    if cache_name is not None:
        storage_key += ".{}".format(cache_name)
    primary_key = ".".join(filter(None, (storage_key, *keys)))
    if cache_name == "secondary":
        nested_key = _secondary_disk_config_key(*keys)
        value = config.get(nested_key, _CONFIG_MISSING)
        if value is not _CONFIG_MISSING:
            return value
    return config.get(primary_key, default)


def _disk_config_key(cache_name, *keys):
    storage_key = "storage.cache"
    if cache_name is not None:
        storage_key += ".{}".format(cache_name)
    return ".".join(filter(None, (storage_key, _DISK_STRATEGY_SECTION, *keys)))


def _secondary_disk_config_key(*keys):
    return ".".join(filter(None, ("storage.cache", _DISK_STRATEGY_SECTION, "secondary", *keys)))


def _disk_config(*keys, default=None, cache_name=None):
    if cache_name == "secondary":
        value = config.get(_secondary_disk_config_key(*keys), _CONFIG_MISSING)
        if value is not _CONFIG_MISSING:
            return default if value is None else value
    return config.get(_disk_config_key(cache_name, *keys), default)


def _disk_config_human_size(*keys, default=None, cache_name=None):
    if cache_name == "secondary":
        value = config.get(_secondary_disk_config_key(*keys), _CONFIG_MISSING)
        if value is not _CONFIG_MISSING:
            if value is None:
                return default
            return get_human_size_default({"value": value}, "value", default)
    return get_human_size_default(config, _disk_config_key(cache_name, *keys), default)


def _disk_config_percentage(*keys, default=None, cache_name=None):
    if cache_name == "secondary":
        value = config.get(_secondary_disk_config_key(*keys), _CONFIG_MISSING)
        if value is not _CONFIG_MISSING:
            if value is None:
                return default
            return get_percentage({"value": value}, "value", required=False, default=default)
    return get_percentage(config, _disk_config_key(cache_name, *keys), required=False, default=default)


def _get_cache_dir(cache_name=None):
    config_key = "storage.cache"
    if cache_name:
        config_key += ".{}".format(cache_name)
    env_var = None
    default_cache_dir = None
    if cache_name is None:
        env_var = CLEARML_CACHE_DIR.get()
        default_cache_dir = DEFAULT_CACHE_DIR
    if cache_name == "secondary":
        env_var = CLEARML_SECONDARY_CACHE_DIR.get()
        nested_default = config.get(
            "{}.{}.secondary.default_base_dir".format("storage.cache", _DISK_STRATEGY_SECTION),
            None,
        )
    else:
        nested_default = None
    cache_base_dir = (
        env_var
        or nested_default
        or config.get("{}.default_base_dir".format(config_key), None)
        or default_cache_dir
    )
    if cache_base_dir is None:
        return None
    cache_base_dir = expandvars(expanduser(cache_base_dir))
    parsed = urlparse(cache_base_dir)
    return Path(os.path.abspath(os.path.join(parsed.netloc, url2pathname(parsed.path))))  # noqa: F405


def normalize_local_path(local_path: str) -> Path:
    """
    Get a normalized local path

    :param local_path: Path of the local file/dir
    :type local_path: str

    :return: Normalized local path
    :rtype: Path
    """
    local_path = os.path.normpath(local_path)
    local_path = os.path.expanduser(local_path)
    local_path = os.path.expandvars(local_path)
    local_path = os.path.realpath(local_path)
    local_path = os.path.abspath(local_path)
    local_path = Path(local_path)
    return local_path


def get_file_mimetype(file_path: str) -> str:
    """
    Get MIME types of a file

    :param file_path: Path of the local file
    :type file_path: str

    :return: File MIME type. Return None if failed to get it
    :rtype: str
    """
    # noinspection PyBroadException
    try:
        file_path = Path(file_path).resolve()
        mimetype, _ = mimetypes.guess_type(file_path.as_posix())
        if mimetype:
            return mimetype
    except Exception:
        pass
    return "binary/octet-stream"


driver_schemes = set(
    filter(
        None,
        itertools.chain(
            (getattr(cls, "scheme", None) for cls in _Driver.__subclasses__()),
            *(getattr(cls, "schemes", []) for cls in _Driver.__subclasses__()),
        ),
    )
)

remote_driver_schemes = driver_schemes - {_FileStorageDriver.scheme}
cloud_driver_schemes = remote_driver_schemes - set(_HttpDriver.schemes)
