import logging
import shlex
from abc import ABC, abstractmethod
import os
from typing import List, Tuple, Any

import attr

from ..backend_api import Session
from ..backend_api.session.defs import ENV_AUTH_TOKEN
from .bash_script_template import render_bash_script_template

env_git_user = "CLEARML_AUTOSCALER_GIT_USER"
env_git_pass = "CLEARML_AUTOSCALER_GIT_PASSWORD"

clearml_conf_template = """\
agent.git_user="{git_user}"
agent.git_pass="{git_pass}"
{extra_clearml_conf}
"""


@attr.s
class CloudDriver(ABC):
    # git
    git_user = attr.ib()
    git_pass = attr.ib()

    # clearml
    extra_clearml_conf = attr.ib()
    api_server = attr.ib()
    web_server = attr.ib()
    files_server = attr.ib()
    access_key = attr.ib()
    secret_key = attr.ib()
    auth_token = attr.ib()

    # Other
    extra_vm_bash_script = attr.ib()
    docker_image = attr.ib()
    tags = attr.ib(default="")
    session = attr.ib(default=None)

    def __attrs_post_init__(self) -> None:
        if self.session is None:
            self.session = Session()

    @abstractmethod
    def spin_up_worker(
        self,
        resource: dict,
        worker_prefix: str,
        queue_name: str,
        task_id: str,
    ) -> None:
        """Creates a new worker for clearml.

        First, create an instance in the cloud and install some required packages.
        Then, define clearml-agent environment variables and run clearml-agent for the specified queue.
        NOTE: - Will wait until instance is running
              - This implementation assumes the instance image already has docker installed

        :param dict resource: resource configuration, as defined in BUDGET and QUEUES.
        :param str worker_prefix: worker name without instance_id
        :param str queue_name: clearml queue to listen to
        :param str task_id: Task ID to restart
        """

    @abstractmethod
    def spin_down_worker(self, instance_id: str) -> None:
        """Destroys the cloud instance.

        :param str instance_id: Cloud instance ID to be destroyed (currently, only AWS EC2 is supported)
        """

    @abstractmethod
    def kind(self) -> str:
        """Return driver kind (e.g. 'AWS')"""

    @abstractmethod
    def instance_id_command(self) -> str:
        """Return a shell command to get instance ID"""

    @abstractmethod
    def instance_type_key(self) -> str:
        """Return key in configuration for instance type"""

    def console_log(self, instance_id: str) -> str:
        """Return log for instance"""
        return ""

    def gen_user_data(
        self,
        worker_prefix: str,
        queue_name: str,
        task_id: str,
        cpu_only: bool = False,
    ) -> str:
        return render_bash_script_template(
            queue_name=queue_name,
            worker_prefix=worker_prefix,
            auth_token=self.auth_token,
            access_key=self.access_key,
            api_server=self.api_server,
            clearml_conf=self.clearml_conf(),
            files_server=self.files_server,
            secret_key=self.secret_key,
            web_server=self.web_server,
            extra_vm_bash_script=self.extra_vm_bash_script,
            cpu_only=cpu_only,
            driver_extra=self.driver_bash_extra(task_id),
            docker_image=self.docker_image,
            instance_id_command=self.instance_id_command(),
        )

    def clearml_conf(self) -> str:
        git_user = (
            os.environ.get(env_git_user)  # ruff-format-hint
            or self.git_user
            or ""
        )
        git_pass = (
            os.environ.get(env_git_pass)  # ruff-format-hint
            or self.git_pass
            or ""
        )

        return clearml_conf_template.format(
            git_user=git_user,
            git_pass=git_pass,
            extra_clearml_conf=self.extra_clearml_conf,
        )

    def driver_bash_extra(self, task_id: str) -> str:
        return (
            f"python -m clearml_agent --config-file ~/clearml.conf execute --id {shlex.quote(task_id)}"
            if task_id  # ruff-format-hint
            else ""
        )

    @classmethod
    def from_config(cls, config: dict) -> "CloudDriver":
        session = Session()
        hyper_params, configurations = config["hyper_params"], config["configurations"]
        opts = {
            "git_user": hyper_params["git_user"],
            "git_pass": hyper_params["git_pass"],
            "extra_clearml_conf": configurations["extra_clearml_conf"],
            "api_server": session.get_api_server_host(),
            "web_server": session.get_app_server_host(),
            "files_server": session.get_files_server_host(),
            "access_key": session.access_key,
            "secret_key": session.secret_key,
            "auth_token": ENV_AUTH_TOKEN.get(),
            "extra_vm_bash_script": configurations["extra_vm_bash_script"],
            "docker_image": hyper_params["default_docker_image"],
            "tags": hyper_params.get("tags", ""),
            "session": session,
        }
        return cls(**opts)

    def set_scaler(self, scaler: Any) -> None:
        self.scaler = scaler

    @property
    def logger(self) -> logging.Logger:
        if self.scaler:
            return self.scaler.logger
        return logging.getLogger("AWSDriver")


def parse_tags(s: str) -> List[Tuple[str, str]]:
    """
    >>> parse_tags('k1=v1, k2=v2')
    [('k1', 'v1'), ('k2', 'v2')]
    """
    s = s.strip()
    if not s:
        return []

    tags = []
    for kv in s.split(","):
        if "=" not in kv:
            raise ValueError(kv)
        key, value = [v.strip() for v in kv.split("=", 1)]
        if not key or not value:
            raise ValueError(kv)
        tags.append((key, value))
    return tags
