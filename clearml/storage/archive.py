import os
import tarfile
from typing import Union, Any, Optional
from pathlib import Path
from zipfile import ZipFile

from .filepaths import is_within_directory
from ..debugging.log import LoggerRoot


def create_zip_directories(
    zipfile: Any,
    path: Optional[Union[str, os.PathLike]] = None,
) -> None:
    try:
        path = os.getcwd() if path is None else os.fspath(path)
        for member in zipfile.namelist():
            arcname = member.replace("/", os.path.sep)
            if os.path.altsep:
                arcname = arcname.replace(os.path.altsep, os.path.sep)
            # interpret absolute pathname as relative, remove drive letter or
            # UNC path, redundant separators, "." and ".." components.
            arcname = os.path.splitdrive(arcname)[1]
            invalid_path_parts = ("", os.path.curdir, os.path.pardir)
            arcname = os.path.sep.join(x for x in arcname.split(os.path.sep) if x not in invalid_path_parts)
            if os.path.sep == "\\":
                # noinspection PyBroadException
                try:
                    # filter illegal characters on Windows
                    # noinspection PyProtectedMember
                    arcname = zipfile._sanitize_windows_name(arcname, os.path.sep)
                except Exception:
                    pass

            targetpath = os.path.normpath(os.path.join(path, arcname))

            # Create all upper directories if necessary.
            upperdirs = os.path.dirname(targetpath)
            if upperdirs:
                os.makedirs(upperdirs, exist_ok=True)
    except Exception as e:
        LoggerRoot.get_base_logger().warning("Failed creating zip directories: " + str(e))


def extract_zip_archive(
    archive_path: str,
    target: str = ".",
) -> None:
    """
    Extracts a .zip archive file to a target location.

    :param str archive_path: The path to the .zip archive file.
    :param str target: The location in the filesystem where the contents of the .zip archive should be extracted.
    """
    with ZipFile(file=archive_path, mode='r') as zip_file:
        base_directory = os.path.abspath(target)
        for file_path in zip_file.namelist():
            flag_path_traversal_vulnerability(
                target_folder=base_directory,
                target_file_path=file_path,
            )
            # No need to run flag_symlink_escape_vulnerability
            # zip_file.extractall does not create symlinks

        create_zip_directories(zip_file, base_directory)

        zip_file.extractall(path=target)


def extract_tar_archive(
    archive_path: str,
    suffix: str,  # Literal[".tar.gz", ".tgz"]
    target: str = ".",
    **kwargs: Any,
) -> None:
    """
    Extracts a .tar.gz or .tgz archive file to a target location after sanitization of the archive's contents.
    Tarfile member sanitization (addresses CVE-2007-4559)

    :param str archive_path: The path to the tar archive file.
    :param str suffix: The full extension of the tar file. Either '.tar.gz' or '.tgz'.
    :param str target: The location in the filesystem where the contents of the tar archive should be extracted.
    """
    mode = {
        ".tar.gz": "r",
        ".tgz": "r:gz",
    }[suffix]

    with tarfile.open(name=archive_path, mode=mode) as tar_file:
        base_dir = os.path.abspath(target)
        for member in tar_file.getmembers():
            flag_path_traversal_vulnerability(
                target_folder=base_dir,
                target_file_path=member.name,
            )

            if member.issym() or member.islnk():
                flag_symlink_escape_vulnerability(
                    target_folder=base_dir,
                    target_file_path=member.name,
                    symlink_target=member.linkname,
                )

        tar_file.extractall(path=base_dir, **kwargs)


def flag_path_traversal_vulnerability(
    target_folder: Union[str, Path],
    target_file_path: Union[str, Path],
) -> None:
    """
    Raises a ValueError if the target_file_path is not contained within the target_folder.
    """
    extraction_file_absolute_path = os.path.abspath(os.path.join(
        target_folder,
        target_file_path,
    ))
    if not is_within_directory(
        target_folder,
        extraction_file_absolute_path,
    ):
        raise ValueError(
            "\n".join([
                "Path traversal detected!",
                f"Target folder: '{target_folder}'",
                f"Target path: '{target_file_path}'",
            ])
        )


def flag_symlink_escape_vulnerability(
    target_folder: Union[str, Path],
    target_file_path: Union[str, Path],
    symlink_target: Union[str, Path],
) -> None:
    """
    Raises a ValueError if the symlink_target points outside of the target_folder.
    """
    extraction_link_absolute_path = os.path.abspath(os.path.join(
        target_folder,
        symlink_target,
    ))
    if not is_within_directory(
        target_folder,
        extraction_link_absolute_path,
    ):
        raise ValueError(
            "\n".join([
                f"Link target escapes target_folder '{target_folder}': ",
                f"{target_file_path} -> {symlink_target}",
            ])
        )
