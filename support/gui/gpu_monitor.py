# support/gui/gpu_monitor.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol

try:
    from pynvml import (
        nvmlInit, nvmlShutdown,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetUtilizationRates,
        nvmlDeviceGetMemoryInfo,
        NVMLError,
    )
    _HAS_NVML = True
except Exception:  # pragma: no cover
    _HAS_NVML = False
    NVMLError = Exception  # type: ignore


class AfterScheduler(Protocol):
    def after(self, delay_ms: int, callback: Callable[[], None]) -> Any: ...
    def after_cancel(self, handle: Any) -> None: ...


@dataclass
class GpuSample:
    util: Optional[int] = None      # 0..100
    mem: Optional[int] = None       # 0..100
    err: Optional[str] = None


class GpuMonitor:
    """
    Tk-agnostic GPU utilization poller.

    - Uses NVML if available.
    - Schedules polling using a Tk-like 'after'.
    - Pushes samples to the GUI via 'on_sample'.
    """

    def __init__(
        self,
        scheduler: AfterScheduler,
        on_sample: Callable[[GpuSample], None],
        device_index: int = 0,
        poll_ms: int = 250,
    ) -> None:
        self.scheduler = scheduler
        self.on_sample = on_sample
        self.device_index = int(device_index)
        self.poll_ms = int(poll_ms)

        self._enabled = False
        self._timer_id: Any | None = None
        self._nvml_inited = False
        self._handle = None

    def set_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._enabled:
            return
        self._enabled = enabled
        if self._enabled:
            self.start()
        else:
            self.stop()

    def start(self) -> None:
        if not self._enabled:
            self._enabled = True

        # init once
        if not _HAS_NVML:
            self.on_sample(GpuSample(err="NVML not available (pynvml missing)."))
            return

        if not self._nvml_inited:
            try:
                nvmlInit()
                self._nvml_inited = True
                self._handle = nvmlDeviceGetHandleByIndex(self.device_index)
            except Exception as e:
                self.on_sample(GpuSample(err=f"NVML init failed: {e}"))
                self._nvml_inited = False
                self._handle = None
                return

        self._schedule_next(0)

    def stop(self) -> None:
        self._enabled = False
        if self._timer_id is not None:
            try:
                self.scheduler.after_cancel(self._timer_id)
            except Exception:
                pass
            self._timer_id = None

        # optional: shutdown NVML to be tidy.
        # If you prefer to keep it open for future starts, you can remove this.
        if self._nvml_inited:
            try:
                nvmlShutdown()
            except Exception:
                pass
        self._nvml_inited = False
        self._handle = None

    def _schedule_next(self, delay_ms: int) -> None:
        if not self._enabled:
            return
        self._timer_id = self.scheduler.after(delay_ms, self._poll_once)

    def _poll_once(self) -> None:
        self._timer_id = None
        if not self._enabled:
            return

        if not self._nvml_inited or self._handle is None:
            self.on_sample(GpuSample(err="NVML not initialized."))
            return

        try:
            util = nvmlDeviceGetUtilizationRates(self._handle).gpu  # percent
            meminfo = nvmlDeviceGetMemoryInfo(self._handle)
            mem_pct = int(round((meminfo.used / max(1, meminfo.total)) * 100.0))
            self.on_sample(GpuSample(util=int(util), mem=mem_pct))
        except NVMLError as e:
            self.on_sample(GpuSample(err=f"NVML error: {e}"))
        except Exception as e:
            self.on_sample(GpuSample(err=str(e)))

        self._schedule_next(self.poll_ms)
