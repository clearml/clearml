from abc import abstractproperty
from typing import Optional
import warnings

from ..backend_config.bucket_config import (
    S3BucketConfig,
    AzureContainerConfig,
    GSBucketConfig,
)
from ..storage.helper import StorageHelper


class SetupUploadMixin(object):
    log = abstractproperty()
    storage_uri = abstractproperty()

    def setup_upload(
        self,
        bucket_name: str,
        host: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        multipart: bool = True,
        https: bool = True,
        region: Optional[str] = None,
        verify: bool = True,
    ) -> None:
        """
        (Deprecated) Setup upload options. Only S3 is supported.
        Please note that this function is deprecated. Use `setup_aws_upload`, `setup_gcp_upload` or
        `setup_azure_upload` to setup the upload options for the corresponding cloud.

        :param bucket_name: AWS bucket name
        :param host: Hostname. Only required in case a Non-AWS S3 solution such as a local Minio server is used)
        :param access_key: AWS access key. If not provided, we'll attempt to obtain the key from the
            configuration file (bucket-specific, than global)
        :param secret_key: AWS secret key. If not provided, we'll attempt to obtain the secret from the
            configuration file (bucket-specific, than global)
        :param multipart: Server supports multipart. Only required when using a Non-AWS S3 solution that doesn't support
            multipart.
        :param https: Server supports HTTPS. Only required when using a Non-AWS S3 solution that only supports HTTPS.
        :param region: Bucket region. Required if the bucket doesn't reside in the default region (us-east-1)
        :param verify: Whether or not to verify SSL certificates.
            Only required when using a Non-AWS S3 solution that only supports HTTPS with self-signed certificate.
        """
        warnings.warn(
            "Warning: 'Task.setup_upload' is deprecated. "
            "Use 'setup_aws_upload', 'setup_gcp_upload' or 'setup_azure_upload' instead",
            DeprecationWarning,
        )
        self.setup_aws_upload(
            bucket_name,
            host=host,
            key=access_key,
            secret=secret_key,
            region=region,
            multipart=multipart,
            secure=https,
            verify=verify,
        )

    def setup_aws_upload(
        self,
        bucket: str,  # str
        subdir: Optional[str] = None,  # Optional[str]
        host: Optional[str] = None,  # Optional[str]
        key: Optional[str] = None,  # Optional[str]
        secret: Optional[str] = None,  # Optional[str]
        token: Optional[str] = None,  # Optional[str]
        region: Optional[str] = None,  # Optional[str]
        multipart: bool = True,  # bool
        secure: bool = True,  # bool
        verify: bool = True,  # bool
        profile: Optional[str] = None,  # Optional[str]
    ) -> None:
        """
        Setup S3 upload options.

        :param bucket: AWS bucket name
        :param subdir: Subdirectory in the AWS bucket
        :param host: Hostname. Only required in case a Non-AWS S3 solution such as a local Minio server is used)
        :param key: AWS access key. If not provided, we'll attempt to obtain the key from the
            configuration file (bucket-specific, than global)
        :param secret: AWS secret key. If not provided, we'll attempt to obtain the secret from the
            configuration file (bucket-specific, than global)
        :param token: AWS 2FA token
        :param region: Bucket region. Required if the bucket doesn't reside in the default region (us-east-1)
        :param multipart: Server supports multipart. Only required when using a Non-AWS S3 solution that doesn't support
            multipart.
        :param secure: Server supports HTTPS. Only required when using a Non-AWS S3 solution that only supports HTTPS.
        :param verify: Whether or not to verify SSL certificates.
        :param profile: The AWS profile
            Only required when using a Non-AWS S3 solution that only supports HTTPS with self-signed certificate.
        """
        self._bucket_config = S3BucketConfig(  # noqa
            bucket=bucket,
            subdir=subdir,
            host=host,
            key=key,
            secret=secret,
            token=token,
            region=region,
            multipart=multipart,
            secure=secure,
            verify=verify,
            profile=profile,
        )
        StorageHelper.add_aws_configuration(self._bucket_config, log=self.log)
        self.storage_uri = StorageHelper.get_aws_storage_uri_from_config(self._bucket_config)

    def setup_gcp_upload(
        self,
        bucket: str,
        subdir: str = "",
        project: Optional[str] = None,
        credentials_json: Optional[str] = None,
        pool_connections: Optional[int] = None,
        pool_maxsize: Optional[int] = None,
    ) -> None:
        """
        Setup GCP upload options.

        :param bucket: Bucket to upload to
        :param subdir: Subdir in bucket to upload to
        :param project: Project the bucket belongs to
        :param credentials_json: Path to the JSON file that contains the credentials
        :param pool_connections: The number of urllib3 connection pools to cache
        :param pool_maxsize: The maximum number of connections to save in the pool
        """
        self._bucket_config = GSBucketConfig(  # noqa
            bucket,
            subdir=subdir,
            project=project,
            credentials_json=credentials_json,
            pool_connections=pool_connections,
            pool_maxsize=pool_maxsize,
        )
        StorageHelper.add_gcp_configuration(self._bucket_config, log=self.log)
        self.storage_uri = StorageHelper.get_gcp_storage_uri_from_config(self._bucket_config)

    def setup_azure_upload(
        self,
        account_name: str,
        account_key: str,
        container_name: Optional[str] = None,
    ) -> None:
        """
        Setup Azure upload options.

        :param account_name: Name of the account
        :param account_key: Secret key used to authenticate the account
        :param container_name: The name of the blob container to upload to
        """
        self._bucket_config = AzureContainerConfig(  # noqa
            account_name=account_name,
            account_key=account_key,
            container_name=container_name,
        )
        StorageHelper.add_azure_configuration(self._bucket_config, log=self.log)
        self.storage_uri = StorageHelper.get_azure_storage_uri_from_config(self._bucket_config)
