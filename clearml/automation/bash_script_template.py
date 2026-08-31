import shlex
from typing import Optional

# Placeholders in this template fall into exactly two categories, and the difference is what
# keeps the rendered script injection-free:
#
#   * Shell *data* -- {clearml_conf}, {api_server}, {web_server}, {files_server},
#     {worker_prefix}, {access_key}, {secret_key}, {auth_token}, {queue}, and the image inside
#     {docker}. These carry configuration- and API-supplied text, so
#     `render_bash_script_template` passes each through `shlex.quote` (and neutralises the
#     heredoc body with a quoted delimiter) as it fills the template. The template deliberately
#     holds no quotes of its own around them: `shlex.quote` already emits a complete shell
#     token, and wrapping that token in more quotes would re-open the hole it closes.
#
#   * Shell *code* -- {instance_id_command}, {bash_script} and {driver_extra}. These are
#     command fragments that exist to be executed, so they are inserted verbatim. Whoever
#     hands them to this module is responsible for quoting the values they embed.
#
# A new placeholder belongs to the first category unless it is genuinely a command.
bash_script_template = """\
#!/bin/bash

set -x

apt-get update
apt-get install -y \
        build-essential \
        gcc \
        git \
        python3-dev \
        python3-pip
python3 -m pip install -U pip
python3 -m pip install virtualenv
python3 -m virtualenv clearml_agent_venv
source clearml_agent_venv/bin/activate
python -m pip install clearml-agent
cat << '{clearml_conf_delimiter}' >> ~/clearml.conf
{clearml_conf}
{clearml_conf_delimiter}
export CLEARML_API_HOST={api_server}
export CLEARML_WEB_HOST={web_server}
export CLEARML_FILES_HOST={files_server}
export DYNAMIC_INSTANCE_ID=$({instance_id_command})
export CLEARML_WORKER_ID={worker_prefix}:$DYNAMIC_INSTANCE_ID
export CLEARML_API_ACCESS_KEY={access_key}
export CLEARML_API_SECRET_KEY={secret_key}
export CLEARML_AUTH_TOKEN={auth_token}
source ~/.bashrc
{bash_script}
{driver_extra}
python -m clearml_agent --config-file ~/clearml.conf daemon --queue {queue} {docker}

if [[ $? -ne 0 ]]
then
  exit 1
fi

shutdown
"""

default_clearml_conf_delimiter = "CLEARML_CONF_EOF"


def render_bash_script_template(
    queue_name: str,
    worker_prefix: str,
    auth_token: Optional[str],
    access_key: Optional[str],
    api_server: str,
    clearml_conf: str,
    files_server: str,
    secret_key: Optional[str],
    web_server: str,
    extra_vm_bash_script: str,
    cpu_only: bool,
    driver_extra: str,
    docker_image: Optional[str],
    instance_id_command: str,
) -> str:
    """
    Render the bash script a freshly provisioned instance runs to join a queue as an agent.

    Every argument naming shell data is quoted as the template is filled, so no value can end a
    quote, open a command substitution, or close the configuration heredoc and have the lines
    behind it run as commands. The two arguments naming shell code --
    ``extra_vm_bash_script`` and ``driver_extra`` -- plus ``instance_id_command`` are inserted
    verbatim and must arrive already quoted.

    :param str queue_name: clearml queue the spun-up agent listens to.
    :param str worker_prefix: Worker name the instance id is appended to.
    :param Optional[str] auth_token: clearml authentication token; absent when key/secret are used instead.
    :param Optional[str] access_key: clearml API access key; absent when a token is used instead.
    :param str api_server: clearml API server URL.
    :param str clearml_conf: Body of the ``clearml.conf`` written on the instance.
    :param str files_server: clearml file server URL.
    :param Optional[str] secret_key: clearml API secret key; absent when a token is used instead.
    :param str web_server: clearml web server URL.
    :param str extra_vm_bash_script: Operator-supplied shell code to run before the agent starts.
    :param bool cpu_only: Hide the instance's GPUs from the agent.
    :param str driver_extra: Driver-supplied shell code to run before the agent starts.
    :param Optional[str] docker_image: Image the agent runs tasks in; absent to run without docker.
    :param str instance_id_command: Shell command printing the instance id.
    :returns: The bash script to hand the instance as its user data.
    :rtype: str
    """
    # Lengthen the delimiter until no configuration line can close the heredoc early.
    clearml_conf_delimiter = default_clearml_conf_delimiter
    clearml_conf_lines = frozenset(line.strip() for line in clearml_conf.splitlines())
    while clearml_conf_delimiter in clearml_conf_lines:
        clearml_conf_delimiter += "_"

    return bash_script_template.format(
        queue=shlex.quote(queue_name),
        worker_prefix=shlex.quote(worker_prefix),
        auth_token=shlex.quote(auth_token or ""),
        access_key=shlex.quote(access_key or ""),
        api_server=shlex.quote(api_server),
        clearml_conf=clearml_conf,
        clearml_conf_delimiter=clearml_conf_delimiter,
        files_server=shlex.quote(files_server),
        secret_key=shlex.quote(secret_key or ""),
        web_server=shlex.quote(web_server),
        bash_script=(
            f"export NVIDIA_VISIBLE_DEVICES=none; {extra_vm_bash_script}"  # ruff-format-hint
            if cpu_only
            else extra_vm_bash_script
        ),
        driver_extra=driver_extra,
        docker=(
            f"--docker {shlex.quote(docker_image)}"  # ruff-format-hint
            if docker_image
            else ""
        ),
        instance_id_command=instance_id_command,
    )
