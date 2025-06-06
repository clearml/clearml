import logging
import os
import platform
import sys
import warnings
from math import ceil, log10
from time import time
from typing import Text, Dict, List, Any

import psutil
from pathlib2 import Path

from .process.mp import BackgroundMonitor
from ..backend_api import Session
from ..binding.frameworks.tensorflow_bind import IsTensorboardInit
from ..config import config, ENV_MULTI_NODE_SINGLE_TASK

try:
    from .gpu import gpustat
except ImportError:
    gpustat = None


class ResourceMonitor(BackgroundMonitor):
    _title_machine = ":monitor:machine"
    _title_gpu = ":monitor:gpu"
    _first_report_sec_default = 30.0
    _wait_for_first_iteration_to_start_sec_default = 180.0
    _max_wait_for_first_iteration_to_start_sec_default = 1800.0
    _resource_monitor_instances = []
    _multi_node_single_task = None

    def __init__(
        self,
        task: Session,
        sample_frequency_per_sec: float = 2.0,
        report_frequency_sec: float = 30.0,
        first_report_sec: float = None,
        wait_for_first_iteration_to_start_sec: float = None,
        max_wait_for_first_iteration_to_start_sec: float = None,
        report_mem_used_per_process: bool = True,
    ) -> None:
        super(ResourceMonitor, self).__init__(task=task, wait_period=sample_frequency_per_sec)
        # noinspection PyProtectedMember
        ResourceMonitor._resource_monitor_instances.append(self)
        ResourceMonitor._multi_node_single_task = ENV_MULTI_NODE_SINGLE_TASK.get()
        self._task = task
        self._sample_frequency = sample_frequency_per_sec
        self._report_frequency = report_frequency_sec
        # noinspection PyProtectedMember
        self._first_report_sec = next(
            value
            # noinspection PyProtectedMember
            for value in (
                first_report_sec,
                ResourceMonitor._first_report_sec_default,
                report_frequency_sec,
            )
            if value is not None
        )
        self.wait_for_first_iteration = next(
            value
            for value in (
                wait_for_first_iteration_to_start_sec,
                # noinspection PyProtectedMember
                ResourceMonitor._wait_for_first_iteration_to_start_sec_default,
                0.0,
            )
            if value is not None
        )
        self.max_check_first_iteration = next(
            value
            for value in (
                max_wait_for_first_iteration_to_start_sec,
                # noinspection PyProtectedMember
                ResourceMonitor._max_wait_for_first_iteration_to_start_sec_default,
                0.0,
            )
            if value is not None
        )
        self._num_readouts = 0
        self._readouts = {}
        self._previous_readouts = {}
        self._previous_readouts_ts = time()
        self._gpustat_fail = 0
        self._gpustat = gpustat
        self._active_gpus = None
        self._process_info = psutil.Process() if report_mem_used_per_process else None
        self._last_process_pool = {}
        self._last_process_id_list = []
        self._gpu_memory_per_process = True
        self._default_gpu_utilization = config.get("resource_monitoring.default_gpu_utilization", 100)
        # allow default_gpu_utilization as null in the config, in which case we don't log anything
        if self._default_gpu_utilization is not None:
            self._default_gpu_utilization = int(self._default_gpu_utilization)
        self._gpu_utilization_warning_sent = False

        # noinspection PyBroadException
        try:
            self._debug_mode = bool(os.getenv("CLEARML_RESMON_DEBUG", ""))
        except Exception:
            self._debug_mode = False

        if not self._gpustat:
            self._task.get_logger().report_text("ClearML Monitor: GPU monitoring is not available")
        else:  # if running_remotely():
            # noinspection PyBroadException
            try:
                active_gpus = os.environ.get("NVIDIA_VISIBLE_DEVICES", "") or os.environ.get("CUDA_VISIBLE_DEVICES", "")
                if active_gpus and active_gpus != "all":
                    if os.path.isdir(active_gpus):
                        try:
                            self._active_gpus = os.listdir(active_gpus)
                        except OSError as e:
                            logging.getLogger("clearml.resource_monitor").warning(
                                "Failed listing {}: {}".format(active_gpus, e)
                            )
                    else:
                        self._active_gpus = [g.strip() for g in active_gpus.split(",")]
            except Exception:
                pass

    def daemon(self) -> None:
        if self._is_thread_mode_and_not_main_process():
            return

        multi_node_single_task_reporting = False
        report_node_as_series = False
        rank = 0
        world_size_digits = 0
        # check if we are in multi-node reporting to the same Task
        # noinspection PyBroadException
        try:
            if self._multi_node_single_task:
                # if resource monitoring is disabled, do nothing
                if self._multi_node_single_task < 0:
                    return
                # we are reporting machines stats on a different machine over the same Task
                multi_node_single_task_reporting = True
                if self._multi_node_single_task == 1:
                    # report per machine graph (unique title)
                    report_node_as_series = False
                elif self._multi_node_single_task == 2:
                    # report per machine series (i.e. merge title+series resource and have "node X" as different series)
                    report_node_as_series = True

                # noinspection PyBroadException
                try:
                    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID")) or 0)
                    world_size_digits = ceil(log10(int(os.environ.get("WORLD_SIZE") or 0)))
                except Exception:
                    pass
        except Exception:
            pass

        seconds_since_started = 0
        reported = 0
        last_iteration = 0
        fallback_to_sec_as_iterations = None

        # get max GPU ID, and make sure our active list is within range
        if self._active_gpus:
            # noinspection PyBroadException
            try:
                gpu_stat = self._gpustat.new_query()
                if max(self._active_gpus) > len(gpu_stat.gpus) - 1:
                    self._active_gpus = None
            except Exception:
                pass

        # add Task runtime_properties with the machine spec
        if Session.check_min_api_version("2.13"):
            try:
                machine_spec = self._get_machine_specs()
                if machine_spec:
                    # noinspection PyProtectedMember
                    self._task._set_runtime_properties(runtime_properties=machine_spec)
            except Exception as ex:
                logging.getLogger("clearml.resource_monitor").debug(
                    "Failed logging machine specification: {}".format(ex)
                )

        # last_iteration_interval = None
        # last_iteration_ts = 0
        # repeated_iterations = 0
        while True:
            last_report = time()
            current_report_frequency = self._report_frequency if reported != 0 else self._first_report_sec
            while (time() - last_report) < current_report_frequency:
                # wait for self._sample_frequency seconds, if event set quit
                if self._event.wait(1.0 / self._sample_frequency):
                    return
                # noinspection PyBroadException
                try:
                    self._update_readouts()
                except Exception:
                    pass

            seconds_since_started += int(round(time() - last_report))
            # check if we do not report any metric (so it means the last iteration will not be changed)
            if fallback_to_sec_as_iterations is None:
                if IsTensorboardInit.tensorboard_used():
                    fallback_to_sec_as_iterations = False
                elif seconds_since_started >= self.wait_for_first_iteration:
                    self._task.get_logger().report_text(
                        "ClearML Monitor: Could not detect iteration reporting, "
                        "falling back to iterations as seconds-from-start"
                    )
                    fallback_to_sec_as_iterations = True
            elif fallback_to_sec_as_iterations is True and seconds_since_started <= self.max_check_first_iteration:
                if self._check_logger_reported():
                    fallback_to_sec_as_iterations = False
                    self._task.get_logger().report_text(
                        "ClearML Monitor: Reporting detected, reverting back to iteration based reporting"
                    )

            clear_readouts = True
            # if we do not have last_iteration, we just use seconds as iteration
            if fallback_to_sec_as_iterations:
                iteration = seconds_since_started
            else:
                iteration = self._task.get_last_iteration()
                if iteration < last_iteration:
                    # we started a new session?!
                    # wait out
                    clear_readouts = False
                    iteration = last_iteration
                elif iteration == last_iteration:
                    # repeated_iterations += 1
                    # if last_iteration_interval:
                    #     # to be on the safe side, we don't want to pass the actual next iteration
                    #     iteration += int(0.95*last_iteration_interval[0] * (seconds_since_started - last_iteration_ts)
                    #                      / last_iteration_interval[1])
                    # else:
                    #     iteration += 1
                    clear_readouts = False
                    iteration = last_iteration
                else:
                    # last_iteration_interval = (iteration - last_iteration, seconds_since_started - last_iteration_ts)
                    # repeated_iterations = 0
                    # last_iteration_ts = seconds_since_started
                    last_iteration = iteration
                    fallback_to_sec_as_iterations = False
                    clear_readouts = True

            # start reporting only when we figured out, if this is seconds based, or iterations based
            average_readouts = self._get_average_readouts()
            if fallback_to_sec_as_iterations is not None:
                for k, v in average_readouts.items():
                    # noinspection PyBroadException
                    try:
                        # 3 digits after the dot
                        value = round(v * 1000) / 1000.0
                        title = self._title_gpu if k.startswith("gpu_") else self._title_machine
                        series = k
                        if multi_node_single_task_reporting:
                            if report_node_as_series:
                                # for rank 0 we keep the same original report so that external services
                                # can always check the default cpu/gpu utilization
                                if rank == 0:
                                    self._task.get_logger().report_scalar(
                                        title=title,
                                        series=series,
                                        iteration=iteration,
                                        value=value,
                                    )

                                # now let's create an additional report
                                title = "{}:{}".format(":".join(title.split(":")[:-1]), series)
                                series = "rank {:0{world_size_digits}d}".format(
                                    rank, world_size_digits=world_size_digits
                                )
                            elif rank > 0:
                                title = "{}:rank{:0{world_size_digits}d}".format(
                                    title, rank, world_size_digits=world_size_digits
                                )
                            else:
                                # for rank 0 we keep the same original report so that external services
                                # can always check the default cpu/gpu utilization
                                pass

                        self._task.get_logger().report_scalar(
                            title=title, series=series, iteration=iteration, value=value
                        )

                    except Exception:
                        pass
                # clear readouts if this is update is not averaged
                if clear_readouts:
                    self._clear_readouts()

            # count reported iterations
            reported += 1

    def _update_readouts(self) -> None:
        readouts = self._machine_stats()
        elapsed = time() - self._previous_readouts_ts
        self._previous_readouts_ts = time()
        for k, v in readouts.items():
            # cumulative measurements
            if k.endswith("_mbs"):
                v = (v - self._previous_readouts.get(k, v)) / elapsed

            self._readouts[k] = self._readouts.get(k, 0.0) + v
        self._num_readouts += 1
        self._previous_readouts = readouts

    def _get_num_readouts(self) -> int:
        return self._num_readouts

    def _get_average_readouts(self) -> dict:
        average_readouts = dict((k, v / float(self._num_readouts)) for k, v in self._readouts.items())
        return average_readouts

    def _clear_readouts(self) -> None:
        self._readouts = {}
        self._num_readouts = 0

    def _machine_stats(self) -> Dict[str, float]:
        """
        :return: machine stats dictionary, all values expressed in megabytes
        """
        cpu_usage = [float(v) for v in psutil.cpu_percent(percpu=True)]
        stats = {
            "cpu_usage": sum(cpu_usage) / float(len(cpu_usage)),
        }

        bytes_per_megabyte = 1024**2

        def bytes_to_megabytes(x: float) -> float:
            return x / bytes_per_megabyte

        virtual_memory = psutil.virtual_memory()
        # stats["memory_used_gb"] = bytes_to_megabytes(virtual_memory.used) / 1024
        stats["memory_used_gb"] = (
            bytes_to_megabytes(self._get_process_used_memory() if self._process_info else virtual_memory.used) / 1024
        )
        stats["memory_free_gb"] = bytes_to_megabytes(virtual_memory.available) / 1024
        disk_use_percentage = psutil.disk_usage(Text(Path.home())).percent
        stats["disk_free_percent"] = 100.0 - disk_use_percentage
        with warnings.catch_warnings():
            if logging.root.level > logging.DEBUG:  # If the logging level is bigger than debug, ignore
                # psutil.sensors_temperatures warnings
                warnings.simplefilter("ignore", category=RuntimeWarning)
            sensor_stat = psutil.sensors_temperatures() if hasattr(psutil, "sensors_temperatures") else {}
        if "coretemp" in sensor_stat and len(sensor_stat["coretemp"]):
            stats["cpu_temperature"] = max([float(t.current) for t in sensor_stat["coretemp"]])

        # protect against permission issues
        # update cached measurements
        # noinspection PyBroadException
        try:
            net_stats = psutil.net_io_counters()
            stats["network_tx_mbs"] = bytes_to_megabytes(net_stats.bytes_sent)
            stats["network_rx_mbs"] = bytes_to_megabytes(net_stats.bytes_recv)
        except Exception:
            pass

        # protect against permission issues
        # noinspection PyBroadException
        try:
            io_stats = psutil.disk_io_counters()
            stats["io_read_mbs"] = bytes_to_megabytes(io_stats.read_bytes)
            stats["io_write_mbs"] = bytes_to_megabytes(io_stats.write_bytes)
        except Exception:
            pass

        # check if we can access the gpu statistics
        if self._gpustat:
            # noinspection PyBroadException
            try:
                stats.update(self._get_gpu_stats())
            except Exception:
                # something happened and we can't use gpu stats,
                self._gpustat_fail += 1
                if self._gpustat_fail >= 3:
                    msg = "ClearML Monitor: GPU monitoring failed getting GPU reading, switching off GPU monitoring"
                    if self._debug_mode:
                        import traceback

                        msg += "\n" + traceback.format_exc()
                    self._task.get_logger().report_text(msg)
                    self._gpustat = None

        return stats

    def _check_logger_reported(self) -> bool:
        titles = self.get_logger_reported_titles(self._task)
        return len(titles) > 0

    @classmethod
    def get_logger_reported_titles(cls, task: Session) -> List[str]:
        # noinspection PyProtectedMember
        titles = list(task.get_logger()._get_used_title_series().keys())

        # noinspection PyBroadException
        try:
            multi_node = cls._multi_node_single_task is not None
        except Exception:
            multi_node = False

        if multi_node:
            title_machine = ":".join(cls._title_machine.split(":")[:-1])
            title_gpu = ":".join(cls._title_gpu.split(":")[:-1])
            if not title_machine:
                title_machine = cls._title_machine
            if not title_gpu:
                title_gpu = cls._title_gpu

            try:
                titles = [t for t in titles if not t.startswith(title_machine) and not t.startswith(title_gpu)]
            except ValueError:
                pass
        else:
            try:
                titles.remove(cls._title_machine)
            except ValueError:
                pass
            try:
                titles.remove(cls._title_gpu)
            except ValueError:
                pass

        return titles

    def _get_process_used_memory(self) -> int:
        def mem_usage_children(a_mem_size: int, pr: psutil.Process, parent_mem: psutil.Process = None) -> int:
            self._last_process_id_list.append(pr.pid)
            # add out memory usage
            our_mem = pr.memory_info()
            mem_diff = our_mem.rss - parent_mem.rss if parent_mem else our_mem.rss
            a_mem_size += mem_diff if mem_diff > 0 else 0
            # now we are the parent
            for child in pr.children():
                # get the current memory
                m = pr.memory_info()
                mem_diff = m.rss - our_mem.rss
                a_mem_size += mem_diff if mem_diff > 0 else 0
                a_mem_size = mem_usage_children(a_mem_size, child, parent_mem=m)
            return a_mem_size

        # only run the memory usage query once per reporting period
        # because this memory query is relatively slow, and changes very little.
        if self._last_process_pool.get("cpu") and (time() - self._last_process_pool["cpu"][0]) < self._report_frequency:
            return self._last_process_pool["cpu"][1]

        # if we have no parent process, return 0 (it's an error)
        if not self._process_info:
            return 0

        self._last_process_id_list = []
        mem_size = mem_usage_children(0, self._process_info)
        self._last_process_pool["cpu"] = time(), mem_size

        return mem_size

    def _skip_nonactive_gpu(self, gpu: Any) -> bool:
        if not self._active_gpus:
            return False
        # noinspection PyBroadException
        try:
            uuid = getattr(gpu, "uuid", None)
            mig_uuid = getattr(gpu, "mig_uuid", None)
            return (
                str(gpu.index) not in self._active_gpus
                and (not uuid or uuid not in self._active_gpus)
                and (not mig_uuid or mig_uuid not in self._active_gpus)
            )
        except Exception:
            pass
        return False

    def _get_gpu_stats(self) -> dict:
        if not self._gpustat:
            return {}

        # per process memory query id slow, so we only call it once per reporting period,
        # On the rest of the samples we return the previous memory measurement

        # update mem used by our process and sub processes
        if (
            self._gpu_memory_per_process
            and self._process_info
            and (
                not self._last_process_pool.get("gpu")
                or (time() - self._last_process_pool["gpu"][0]) >= self._report_frequency
            )
        ):
            gpu_mem = {}
            # noinspection PyBroadException
            try:
                gpu_stat = self._gpustat.new_query(per_process_stats=True)
            except Exception:
                gpu_stat = self._gpustat.new_query(per_process_stats=False)

            for i, g in enumerate(gpu_stat.gpus):
                # if processes is None, that means we can't query GPU memory usage per proces, so we can stop
                if g.processes is None:
                    self._gpu_memory_per_process = False
                    break
                # only monitor the active gpu's, if none were selected, monitor everything
                if self._skip_nonactive_gpu(g):
                    continue

                gpu_mem[i] = 0
                for p in g.processes:
                    if p is not None and p["pid"] in self._last_process_id_list:
                        gpu_mem[i] += p.get("gpu_memory_usage", 0)

            self._last_process_pool["gpu"] = time(), gpu_mem
        else:
            # if we do no need to update the memory usage, run global query
            # if we have no parent process (backward compatibility), return global stats
            gpu_stat = self._gpustat.new_query(per_process_stats=False)
            gpu_mem = self._last_process_pool["gpu"][1] if self._last_process_pool.get("gpu") else None

        # generate the statistics dict for actual report
        stats = {}

        for i, g in enumerate(gpu_stat.gpus):
            # only monitor the active gpu's, if none were selected, monitor everything
            if self._skip_nonactive_gpu(g):
                continue
            if g["temperature.gpu"] is not None:
                stats["gpu_%d_temperature" % i] = float(g["temperature.gpu"])
            if g["utilization.gpu"] is not None:
                stats["gpu_%d_utilization" % i] = float(g["utilization.gpu"])
            else:
                stats["gpu_%d_utilization" % i] = self._default_gpu_utilization
                if not self._gpu_utilization_warning_sent:
                    if g.mig_index is not None:
                        self._task.get_logger().report_text(
                            "Running inside MIG, Nvidia driver cannot export utilization, pushing fixed value {}".format(
                                # noqa
                                self._default_gpu_utilization
                            )
                        )
                    else:
                        self._task.get_logger().report_text(
                            "Nvidia driver cannot export utilization, pushing fixed value {}".format(
                                self._default_gpu_utilization
                            )
                        )
                    self._gpu_utilization_warning_sent = True
            stats["gpu_%d_mem_usage" % i] = 100.0 * float(g["memory.used"]) / float(g["memory.total"])
            # already in MBs
            stats["gpu_%d_mem_free_gb" % i] = float(g["memory.total"] - g["memory.used"]) / 1024
            # use previously sampled process gpu memory, or global if it does not exist
            stats["gpu_%d_mem_used_gb" % i] = float(gpu_mem[i] if gpu_mem and i in gpu_mem else g["memory.used"]) / 1024

        return stats

    def _get_machine_specs(self) -> dict:
        specs = {}
        # noinspection PyBroadException
        try:
            specs = {
                "platform": str(sys.platform),
                "python_version": str(platform.python_version()),
                "python_exec": str(sys.executable),
                "OS": str(platform.platform(aliased=True)),
                "processor": str(platform.machine()),
                "cpu_cores": int(psutil.cpu_count()),
                "memory_gb": round(psutil.virtual_memory().total / 1024**3, 1),
                "hostname": str(platform.node()),
                "gpu_count": 0,
            }
            if self._gpustat:
                gpu_stat = self._gpustat.new_query(shutdown=True, get_driver_info=True)
                if gpu_stat.gpus:
                    gpus = [g for i, g in enumerate(gpu_stat.gpus) if not self._skip_nonactive_gpu(g)]
                    specs.update(
                        gpu_count=int(len(gpus)),
                        gpu_type=", ".join(g.name for g in gpus),
                        gpu_memory=", ".join("{}GB".format(round(g.memory_total / 1024.0)) for g in gpus),
                        gpu_driver_version=gpu_stat.driver_version or "",
                        gpu_driver_cuda_version=gpu_stat.driver_cuda_version or "",
                    )

        except Exception:
            pass

        return specs

    @property
    def resource_monitor_instances(self) -> None:
        return self._resource_monitor_instances
