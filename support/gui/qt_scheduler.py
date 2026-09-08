"""Qt scheduling helpers used by the PySide camera utilities.

The public ``after``/``after_cancel`` surface intentionally matches the small
scheduler protocol already consumed by ``GpuMonitor``, ``ConfigStore``, and
``CheckerboardLauncher``.  It is not a Tk widget compatibility layer.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
import itertools

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot, Qt


class QtScheduler(QObject):
    """Own one-shot Qt timers and provide safe GUI-thread dispatch."""

    _dispatch_requested = Signal(object, tuple, dict)
    _schedule_requested = Signal(int, int, object, tuple)
    _cancel_requested = Signal(int)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._closed = False
        self._next_handle = itertools.count(1)
        self._timers: dict[int, QTimer] = {}
        self._dispatch_requested.connect(
            self._run_dispatched,
            Qt.ConnectionType.QueuedConnection,
        )
        self._schedule_requested.connect(
            self._start_timer,
            Qt.ConnectionType.QueuedConnection,
        )
        self._cancel_requested.connect(
            self._cancel_on_owner_thread,
            Qt.ConnectionType.QueuedConnection,
        )

    def after(
        self,
        delay_ms: int,
        callback: Callable[..., Any],
        *args: Any,
    ) -> int:
        """Schedule ``callback(*args)`` once and return a cancellable handle."""
        handle = next(self._next_handle)
        if self._closed:
            return handle

        try:
            if QThread.currentThread() == self.thread():
                self._start_timer(handle, max(0, int(delay_ms)), callback, args)
            else:
                self._schedule_requested.emit(
                    handle,
                    max(0, int(delay_ms)),
                    callback,
                    args,
                )
        except RuntimeError:
            # The QObject may already have been destroyed while a Python
            # worker still holds its wrapper during application teardown.
            self._closed = True
        return handle

    @Slot(int, int, object, tuple)
    def _start_timer(
        self,
        handle: int,
        delay_ms: int,
        callback: Callable[..., Any],
        args: tuple[Any, ...],
    ) -> None:
        if self._closed:
            return

        timer = QTimer(self)
        timer.setSingleShot(True)

        def fire() -> None:
            self._timers.pop(handle, None)
            timer.deleteLater()
            if not self._closed:
                callback(*args)

        timer.timeout.connect(fire)
        self._timers[handle] = timer
        timer.start(delay_ms)

    def after_idle(self, callback: Callable[..., Any], *args: Any) -> int:
        return self.after(0, callback, *args)

    def after_cancel(self, handle: Any) -> None:
        """Cancel one handle. Unknown/already-fired handles are harmless."""
        if self._closed:
            return

        try:
            numeric_handle = int(handle)
        except (TypeError, ValueError):
            return
        try:
            if QThread.currentThread() != self.thread():
                self._cancel_requested.emit(numeric_handle)
                return
            self._cancel_on_owner_thread(numeric_handle)
        except RuntimeError:
            self._closed = True

    @Slot(int)
    def _cancel_on_owner_thread(self, numeric_handle: int) -> None:
        timer = self._timers.pop(numeric_handle, None)
        if timer is not None:
            timer.stop()
            timer.deleteLater()

    def cancel_all(self) -> None:
        for handle in tuple(self._timers):
            self.after_cancel(handle)

    def dispatch(
        self,
        callback: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Queue work onto this object's Qt thread from any Python thread."""
        if self._closed:
            return
        try:
            self._dispatch_requested.emit(callback, args, kwargs)
        except RuntimeError:
            self._closed = True

    @Slot(object, tuple, dict)
    def _run_dispatched(
        self,
        callback: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        if not self._closed:
            callback(*args, **kwargs)

    def close(self) -> None:
        if self._closed:
            return

        # close() is called by the owning QWidget on the Qt thread. Mark the
        # scheduler closed first so worker calls racing with timer cleanup are
        # ignored without touching the soon-to-be-deleted QObject.
        self._closed = True
        for handle in tuple(self._timers):
            self._cancel_on_owner_thread(handle)


class QtValue(QObject):
    """Small observable value for non-widget controller state.

    Existing runtime helpers currently expect ``get()``, ``set()`` and
    ``trace_add()``.  Keeping that protocol here avoids coupling those helpers
    to a particular widget while the GUI itself uses ordinary Qt signals.
    """

    changed = Signal(object)

    def __init__(self, value: Any = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._value = value
        self._trace_callbacks: list[Callable[..., Any]] = []
        self.changed.connect(self._notify_traces)

    def get(self) -> Any:
        return self._value

    def set(self, value: Any) -> None:
        if self._value == value:
            return
        self._value = value
        self.changed.emit(value)

    def trace_add(self, _mode: str, callback: Callable[..., Any]) -> Callable[..., Any]:
        self._trace_callbacks.append(callback)
        return callback

    @Slot(object)
    def _notify_traces(self, _value: Any) -> None:
        for callback in tuple(self._trace_callbacks):
            callback(None, None, None)
