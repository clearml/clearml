import os
import re
from typing import Optional, Tuple

_CONTAINER_MARKERS = re.compile(r"docker|kubepods|containerd|libpod|lxc")


def get_cgroup_cpu_sample(root: str = "/sys/fs/cgroup") -> Optional[Tuple[int, float]]:
    """
    Read this container's CPU accounting from cgroup v2. Inside a container ``/sys/fs/cgroup`` is the
    container's own cgroup, so this matches what ``docker stats`` / ``kubectl top`` report.

    :return: (cumulative CPU microseconds used, number of CPUs the container may use),
        or None when not running inside a container
    :raises OSError, ValueError: inside a container whose cgroup files cannot be read (e.g. cgroup v1)
    """
    if not _in_container():
        return None
    usage = None
    with open(os.path.join(root, "cpu.stat")) as f:
        for line in f:
            if line.startswith("usage_usec"):
                usage = int(line.split()[1])
    if usage is None:
        raise ValueError("usage_usec missing from cpu.stat")
    with open(os.path.join(root, "cpu.max")) as f:
        quota, period = f.read().split()
    cpus = len(os.sched_getaffinity(0)) if quota == "max" else int(quota) / int(period)
    return usage, cpus


def _in_container() -> bool:
    if os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv"):
        return True
    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        return True
    try:
        with open("/proc/1/cgroup") as f:
            init_cgroup = f.read().strip()
    except OSError:
        return False
    # "0::/" means PID 1 sits in the cgroup root, i.e. a private cgroup namespace (a container)
    return bool(_CONTAINER_MARKERS.search(init_cgroup)) or init_cgroup == "0::/"
