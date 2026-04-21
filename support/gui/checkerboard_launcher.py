# support/gui/checkerboard_launcher.py

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class CheckerboardLaunchState:
    proc: Optional[subprocess.Popen] = None


class CheckerboardLauncher:
    """
    Manages launching CalBoardGenerator in a separate process.

    Designed to be GUI-framework-agnostic:
      - You provide an 'after(ms, callback)' scheduler (Tk-compatible)
      - You provide a callback for "state changed" (button text/state)
    """

    def __init__(
        self,
        state: CheckerboardLaunchState,
        after: Callable[[int, Callable[[], None]], Any],
        on_status: Callable[[str, str], None],
        poll_ms: int = 300,
        module_name: str = "support.vision.cal_board_generator",
    ) -> None:
        self.state = state
        self.after = after
        self.on_status = on_status
        self.poll_ms = int(poll_ms)
        self.module_name = module_name

    def is_running(self) -> bool:
        p = self.state.proc
        return p is not None and p.poll() is None

    def stop(self) -> None:
        p = self.state.proc
        if p is not None and p.poll() is None:
            try:
                p.terminate()
            except Exception:
                pass
        self.state.proc = None
        self.on_status("normal", "Checkerboard")

    def toggle(self) -> None:
        """
        If running => stop.
        Else => launch and begin polling.
        """
        if self.is_running():
            self.stop()
            return

        self.on_status("disabled", "Checkerboard (launching.)")

        cmd = [sys.executable, "-m", self.module_name]

        creationflags = 0
        if sys.platform.startswith("win"):
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

        try:
            self.state.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                # stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
        except Exception:
            # If you still want the old "find_spec fallback", you can add it here.
            self.state.proc = None
            self.on_status("normal", "Checkerboard")
            raise

        # mirror your current behavior: re-enable UI immediately and poll to detect exit
        self.on_status("normal", "Checkerboard (running)")
        self.after(self.poll_ms, self._poll)

    def _poll(self) -> None:
        p = self.state.proc
        if p is None:
            self.on_status("normal", "Checkerboard")
            return

        if p.poll() is None:
            self.after(self.poll_ms, self._poll)
            return

        # exited
        self.state.proc = None
        self.on_status("normal", "Checkerboard")
