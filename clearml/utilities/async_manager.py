import os
import time
from typing import Optional, Callable, Any

import six

from .process.mp import SingletonLock


class AsyncManagerMixin(object):
    _async_results_lock = SingletonLock()
    # per pid (process) list of async jobs (support for sub-processes forking)
    _async_results = {}

    @classmethod
    def _add_async_result(
        cls,
        result: Any,
        wait_on_max_results: Optional[int] = None,
        wait_time: int = 30,
        wait_cb: Optional[Callable[[int], None]] = None,
    ) -> None:
        while True:
            try:
                cls._async_results_lock.acquire()
                # discard completed results
                pid = os.getpid()
                cls._async_results[pid] = [r for r in cls._async_results.get(pid, []) if not r.ready()]
                num_results = len(cls._async_results[pid])
                if wait_on_max_results is not None and num_results >= wait_on_max_results:
                    # At least max_results results are still pending, wait
                    if wait_cb:
                        wait_cb(num_results)
                    if wait_time:
                        time.sleep(wait_time)
                    continue
                # add result
                if result and not result.ready():
                    if not cls._async_results.get(pid):
                        cls._async_results[pid] = []
                    cls._async_results[pid].append(result)
                break
            finally:
                cls._async_results_lock.release()

    @classmethod
    def wait_for_results(cls, timeout: float = None, max_num_uploads: int = None) -> None:
        remaining = timeout
        count = 0
        pid = os.getpid()
        for r in cls._async_results.get(pid, []):
            if r.ready():
                continue
            t = time.time()
            # bugfix for python2.7 threading issues
            if six.PY2 and not remaining:
                while not r.ready():
                    r.wait(timeout=2.0)
            else:
                r.wait(timeout=remaining)
            count += 1
            if max_num_uploads is not None and max_num_uploads - count <= 0:
                break
            if timeout is not None:
                remaining = max(0.0, remaining - max(0.0, time.time() - t))
                if not remaining:
                    break

    @classmethod
    def get_num_results(cls) -> int:
        pid = os.getpid()
        if cls._async_results.get(pid, []):
            return len([r for r in cls._async_results.get(pid, []) if not r.ready()])
        else:
            return 0
