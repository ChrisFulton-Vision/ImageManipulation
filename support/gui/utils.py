import time
from dataclasses import dataclass
import threading

# Keys that should be treated as edge-triggered (one per distinct press)
_EDGE_KEYS = {ord(' '), ord('f'), ord('w'), ord('s'), ord('e'), ord('p'),
              ord('r'), ord('['), ord(']'), ord('{'), ord('}'),
              ord(';'), ord("'"), ord(':'), ord('"'), ord('b'), ord('n'), 27}

# Keys that should fire on every event (allow repeats within a burst)
_REPEAT_KEYS = {ord('a'), ord('d'), ord('c'), ord('z')}

# Cooldown for edge keys
_EDGE_COOLDOWN_MS = 120


def _is_edge_allowed(key: int, last_ts: dict[int, float]) -> bool:
    now = time.monotonic()
    prev = last_ts.get(key, 0.0)
    if (now - prev) * 1000.0 >= _EDGE_COOLDOWN_MS:
        last_ts[key] = now
        return True
    return False


def fmt_mmss(seconds: float) -> str:
    if seconds is None or seconds != seconds or seconds < 0:  # NaN/neg guard
        return "--:--"
    seconds = int(round(seconds))
    m, s = divmod(seconds, 60)
    return f"{m:02d}:{s:02d}"


@dataclass
class SweepTimer:
    t0: float = None

    def start(self) -> None:
        if self.t0 is None:
            self.t0 = time.monotonic()

    def eta_from_fraction(self, frac_done: float) -> float:
        """ETA seconds given overall progress fraction [0..1]."""
        self.start()
        frac_done = max(0.0, min(1.0, float(frac_done)))
        if frac_done <= 1e-9:
            return float("nan")
        elapsed = time.monotonic() - self.t0
        total_est = elapsed / frac_done
        return max(0.0, total_est - elapsed)


class PausedCache:
    def __init__(self):
        self.idx = None
        self.frame = None

    def set(self, i, f): self.idx, self.frame = i, f

    def get(self, i): return self.frame if self.idx == i else None

    def clear(self): self.idx = self.frame = None


@dataclass
class PlaybackState:
    """Single source of truth for playback state."""
    speed: float = 1.0  # signed: <0 reverse, 0 paused, >0 forward
    last_nonzero_sign: int = 1  # +1 or -1, used when resuming from pause
    stride: int = 1  # cached stride we last told the loader
    curr_idx: int = 0


class ThreadStopper:
    def __init__(self):
        self._ev = threading.Event()

    def set(self):
        self._ev.set()

    def is_set(self) -> bool:
        return self._ev.is_set()
