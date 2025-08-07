""" Timing support """
import sys
import time
from typing import Callable, Optional, Dict, List, Any

import six


class Timer(object):
    """A class implementing a simple timer, with a reset option"""

    def __init__(self) -> None:
        self._start_time = 0.0
        self._diff = 0.0
        self._total_time = 0.0
        self._average_time = 0.0
        self._calls = 0
        self.tic()

    def reset(self) -> None:
        self._start_time = 0.0
        self._diff = 0.0
        self.reset_average()

    def reset_average(self) -> None:
        """Reset average counters (does not change current timer)"""
        self._total_time = 0
        self._average_time = 0
        self._calls = 0

    def tic(self) -> None:
        try:
            # using time.time instead of time.clock because time time.clock
            # does not normalize for multi threading
            self._start_time = time.time()
        except Exception:
            pass

    def toc(self, average: bool = True) -> float:
        self._diff = time.time() - self._start_time
        self._total_time += self._diff
        self._calls += 1
        self._average_time = self._total_time / self._calls
        if average:
            return self._average_time
        else:
            return self._diff

    @property
    def average_time(self) -> float:
        return self._average_time

    @property
    def total_time(self) -> float:
        return self._total_time

    def toc_with_reset(self, average: bool = True, reset_if_calls: int = 1000) -> float:
        """Enable toc with reset (slightly inaccurate if reset event occurs)"""
        if self._calls > reset_if_calls:
            last_diff = time.time() - self._start_time
            self._start_time = time.time()
            self._total_time = last_diff
            self._average_time = 0
            self._calls = 0

        return self.toc(average=average)


class TimersMixin(object):
    def __init__(self) -> None:
        self._timers = {}

    def add_timers(self, *names: Any) -> None:
        for name in names:
            self.add_timer(name)

    def add_timer(self, name: str, timer: Timer = None) -> Timer:
        if name in self._timers:
            raise ValueError("timer %s already exists" % name)
        timer = timer or Timer()
        self._timers[name] = timer
        return timer

    def get_timer(self, name: str, default: Optional[Timer] = None) -> Optional[Timer]:
        return self._timers.get(name, default)

    def get_timers(self) -> Dict[str, Timer]:
        return self._timers

    def _call_timer(
        self,
        name: str,
        callable: Callable[[Timer], Any],
        silent_fail: bool = False,
    ) -> Any:
        try:
            return callable(self._timers[name])
        except KeyError:
            if not silent_fail:
                six.reraise(*sys.exc_info())

    def reset_timers(self, *names: Any) -> None:
        for name in names:
            self._call_timer(name, lambda t: t.reset())

    def reset_average_timers(self, *names: Any) -> None:
        for name in names:
            self._call_timer(name, lambda t: t.reset_average())

    def tic_timers(self, *names: Any) -> None:
        for name in names:
            self._call_timer(name, lambda t: t.tic())

    def toc_timers(self, *names: Any) -> List[float]:
        return [self._call_timer(name, lambda t: t.toc()) for name in names]

    def toc_with_reset_timer(self, name: str, average: bool = True, reset_if_calls: int = 1000) -> float:
        return self._call_timer(name, lambda t: t.toc_with_reset(average, reset_if_calls))
