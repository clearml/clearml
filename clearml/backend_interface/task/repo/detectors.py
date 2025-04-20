import abc
import logging
import os
from subprocess import call, CalledProcessError
from typing import Any

import attr
import six
from pathlib2 import Path

from ....config.defs import (
    VCS_REPO_TYPE,
    VCS_DIFF,
    VCS_STATUS,
    VCS_ROOT,
    VCS_BRANCH,
    VCS_COMMIT_ID,
    VCS_REPOSITORY_URL,
)
from ....debugging import get_logger
from .util import get_command_output


class DetectionError(Exception):
    pass


@attr.s
class Result(object):
    """ " Repository information as queried by a detector"""

    url = attr.ib(default="")
    branch = attr.ib(default="")
    commit = attr.ib(default="")
    root = attr.ib(default="")
    status = attr.ib(default="")
    diff = attr.ib(default="")
    modified = attr.ib(default=False, type=bool, converter=bool)

    def is_empty(self) -> bool:
        return not any(attr.asdict(self).values())


@six.add_metaclass(abc.ABCMeta)
class Detector(object):
    """Base class for repository detection"""

    """
    Commands are represented using the result class, where each attribute contains
    the command used to obtain the value of the same attribute in the actual result.
    """

    _fallback = "_fallback"
    _remote = "_remote"

    @classmethod
    def _get_logger(cls) -> logging.Logger:
        return get_logger("Repository Detection")

    @attr.s
    class Commands(object):
        """ " Repository information as queried by a detector"""

        url = attr.ib(default=None, type=list)
        branch = attr.ib(default=None, type=list)
        commit = attr.ib(default=None, type=list)
        root = attr.ib(default=None, type=list)
        status = attr.ib(default=None, type=list)
        diff = attr.ib(default=None, type=list)
        modified = attr.ib(default=None, type=list)
        # alternative commands
        branch_fallback = attr.ib(default=None, type=list)
        diff_fallback = attr.ib(default=None, type=list)
        # remote commands
        commit_remote = attr.ib(default=None, type=list)
        diff_remote = attr.ib(default=None, type=list)
        diff_fallback_remote = attr.ib(default=None, type=list)

    def __init__(self, type_name: str, name: str = None) -> None:
        self.type_name = type_name
        self.name = name or type_name

    def _get_commands(self) -> "Detector.Commands":
        """Returns a RepoInfo instance containing a command for each info attribute"""
        return self.Commands()

    def _get_command_output(
        self,
        path: str,
        name: str,
        command: list,
        commands: "Detector.Commands" = None,
        strip: bool = True,
    ) -> str:
        """Run a command and return its output"""
        try:
            return get_command_output(command, path, strip=strip)

        except (CalledProcessError, UnicodeDecodeError) as ex:
            if not name.endswith(self._fallback):
                fallback_command = attr.asdict(commands or self._get_commands()).get(name + self._fallback)
                if fallback_command:
                    try:
                        return get_command_output(fallback_command, path, strip=strip)
                    except (CalledProcessError, UnicodeDecodeError):
                        pass
            self._get_logger().warning("Can't get {} information for {} repo in {}".format(name, self.type_name, path))
            # full details only in debug
            self._get_logger().debug(
                "Can't get {} information for {} repo in {}: {}".format(name, self.type_name, path, str(ex))
            )
            return ""

    def _get_info(
        self,
        path: str,
        include_diff: bool = False,
        diff_from_remote: bool = False,
    ) -> Result:
        """
        Get repository information.
        :param path: Path to repository
        :param include_diff: Whether to include the diff command's output (if available)
        :param diff_from_remote: Whether to store the remote diff/commit based on the remote commit (not local commit)
        :return: RepoInfo instance
        """
        path = str(path)
        commands = self._get_commands()
        if not include_diff:
            commands.diff = None

        # skip the local commands
        if diff_from_remote and commands:
            for name, command in attr.asdict(commands).items():
                if name.endswith(self._remote) and command:
                    setattr(commands, name[: -len(self._remote)], None)

        info = Result(
            **{
                name: self._get_command_output(path, name, command, commands=commands, strip=bool(name != "diff"))
                for name, command in attr.asdict(commands).items()
                if command and not name.endswith(self._fallback) and not name.endswith(self._remote)
            }
        )

        if diff_from_remote and commands:
            for name, command in attr.asdict(commands).items():
                if name.endswith(self._remote) and command:
                    setattr(commands, name[: -len(self._remote)], command + [info.branch])

            info = attr.assoc(
                info,
                **{
                    name[: -len(self._remote)]: self._get_command_output(
                        path,
                        name[: -len(self._remote)],
                        command + [info.branch],
                        commands=commands,
                        strip=not name.startswith("diff"),
                    )
                    for name, command in attr.asdict(commands).items()
                    if command
                    and (name.endswith(self._remote) and not name[: -len(self._remote)].endswith(self._fallback))
                },
            )
            # make sure we match the modified with the git remote diff state
            info.modified = bool(info.diff)

        return info

    def _post_process_info(self, info: Result) -> Result:
        # check if there are uncommitted changes in the current repository
        return info

    def get_info(
        self,
        path: str,
        include_diff: bool = False,
        diff_from_remote: bool = False,
    ) -> Result:
        """
        Get repository information.
        :param path: Path to repository
        :param include_diff: Whether to include the diff command's output (if available)
        :param diff_from_remote: Whether to store the remote diff/commit based on the remote commit (not local commit)
        :return: RepoInfo instance
        """
        info = self._get_info(path, include_diff, diff_from_remote=diff_from_remote)
        return self._post_process_info(info)

    def _is_repo_type(self, script_path: Path) -> bool:
        try:
            with open(os.devnull, "wb") as devnull:
                return (
                    call(
                        [self.type_name, "status"],
                        stderr=devnull,
                        stdout=devnull,
                        cwd=str(script_path),
                    )
                    == 0
                )
        except CalledProcessError:
            self._get_logger().warning("Can't get {} status".format(self.type_name))
        except (OSError, EnvironmentError, IOError):
            # File not found or can't be executed
            pass
        return False

    def exists(self, script_path: str) -> bool:
        """
        Test whether the given script resides in
        a repository type represented by this plugin.
        """
        return self._is_repo_type(script_path)


class HgDetector(Detector):
    def __init__(self) -> None:
        super(HgDetector, self).__init__("hg")

    def _get_commands(self) -> "HgDetector.Commands":
        return self.Commands(
            url=["hg", "paths", "--verbose"],
            branch=["hg", "--debug", "id", "-b"],
            commit=["hg", "--debug", "id", "-i"],
            root=["hg", "root"],
            status=["hg", "status"],
            diff=["hg", "diff"],
            modified=["hg", "status", "-m"],
        )

    def _post_process_info(self, info: Result) -> Result:
        if info.url:
            info.url = info.url.split(" = ")[1]

        if info.commit:
            info.commit = info.commit.rstrip("+")

        return info


class GitDetector(Detector):
    def __init__(self) -> None:
        super(GitDetector, self).__init__("git")

    def _get_commands(self) -> "GitDetector.Commands":
        return self.Commands(
            url=["git", "ls-remote", "--get-url"],
            branch=["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
            commit=["git", "rev-parse", "HEAD"],
            root=["git", "rev-parse", "--show-toplevel"],
            status=["git", "status", "-s"],
            diff=["git", "diff", "--submodule=diff", "HEAD"],
            modified=["git", "ls-files", "-m"],
            branch_fallback=["git", "rev-parse", "--abbrev-ref", "HEAD"],
            diff_fallback=["git", "diff", "HEAD"],
            diff_remote=[
                "git",
                "diff",
                "--submodule=diff",
            ],
            commit_remote=[
                "git",
                "rev-parse",
            ],
            diff_fallback_remote=[
                "git",
                "diff",
            ],
        )

    def get_info(
        self,
        path: str,
        include_diff: bool = False,
        diff_from_remote: bool = False,
    ) -> Result:
        """
        Get repository information.
        :param path: Path to repository
        :param include_diff: Whether to include the diff command's output (if available)
        :param diff_from_remote: Whether to store the remote diff/commit based on the remote commit (not local commit)
        :return: RepoInfo instance
        """
        info = super(GitDetector, self).get_info(
            path=path, include_diff=include_diff, diff_from_remote=diff_from_remote
        )

        # check if for some reason we ended with the wrong branch name as HEAD (will happen when HEAD is detached)
        if info and info.branch == "HEAD":
            # noinspection PyBroadException
            try:
                # try to reparse the detached HEAD
                commands = self._get_commands()
                name = "branch"
                command = commands.branch_fallback[:]
                command[-1] = "origin"
                info.branch = (
                    self._get_command_output(
                        path,
                        name,
                        command,
                        commands=commands,
                        strip=bool(name != "diff"),
                    )
                    or info.branch
                )
                info = self._post_process_info(info)
            except Exception:
                # not sure what happened, just skip it.
                pass

        # we could not locate the git because of some configuration issue, we need to try a more complex command
        if info and info.url == "origin":
            # noinspection PyBroadException
            try:
                url = get_command_output(["git", "remote", "-v"], path, strip=True)
                url = url.split("\n")[0].split("\t", 1)[1]
                if url.endswith("(fetch)"):
                    url = url[: -len("(fetch)")]
                elif url.endswith("(pull)"):
                    url = url[: -len("(pull)")]
                info.url = url.strip()
            except Exception:
                # not sure what happened, just skip it.
                pass

        return info

    def _post_process_info(self, info: Result) -> Result:
        # Deprecated code: this was intended to make sure git repository names always
        # ended with ".git", but this is not always the case (e.g. Azure Repos)
        # if info.url and not info.url.endswith(".git"):
        #     info.url += ".git"

        if (info.branch or "").startswith("origin/"):
            info.branch = info.branch[len("origin/") :]

        return info


class EnvDetector(Detector):
    def __init__(self, type_name: str) -> None:
        super(EnvDetector, self).__init__(type_name, "{} environment".format(type_name))

    def _is_repo_type(self, script_path: str) -> bool:
        return VCS_REPO_TYPE.get().lower() == self.type_name and bool(VCS_REPOSITORY_URL.get())

    @staticmethod
    def _normalize_root(root: str) -> str:
        """
        Convert to absolute and squash 'path/../folder'
        """
        # noinspection PyBroadException
        try:
            return os.path.abspath((Path.cwd() / root).absolute().as_posix())
        except Exception:
            return Path.cwd()

    def _get_info(
        self,
        _: Any,
        include_diff: bool = False,
        diff_from_remote: bool = None,
    ) -> Result:
        repository_url = VCS_REPOSITORY_URL.get()

        if not repository_url:
            raise DetectionError("No VCS environment data")
        status = VCS_STATUS.get() or ""
        diff = VCS_DIFF.get() or ""
        modified = bool(diff or (status and [s for s in status.split("\n") if s.strip().startswith("M ")]))
        if modified and not diff:
            diff = "# Repository modified, but no git diff could be extracted."
        return Result(
            url=repository_url,
            branch=VCS_BRANCH.get(),
            commit=VCS_COMMIT_ID.get(),
            root=VCS_ROOT.get(converter=self._normalize_root),
            status=status,
            diff=diff,
            modified=modified,
        )


class GitEnvDetector(EnvDetector):
    def __init__(self) -> None:
        super(GitEnvDetector, self).__init__("git")


class HgEnvDetector(EnvDetector):
    def __init__(self) -> None:
        super(HgEnvDetector, self).__init__("hg")
