# buffered_image_loader.py
import threading, queue, time
from typing import Callable, Iterable, Optional, Tuple, List
from cv2 import IMREAD_UNCHANGED, imread
import numpy as np

class BufferedImageLoader:
    """
    Background prefetcher for folders of images.
    - Decodes images off the main/UI thread.
    - Ring buffer smooths over disk/AV jitter.
    - Supports seek/jumps; stale frames auto-dropped.
    """
    def __init__(
        self,
        filepaths: List[str],
        max_buffer: int = 32,
        preprocess: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        start_index: int = 0,
        loop: bool = True,
        read_flags: int = IMREAD_UNCHANGED,
    ):
        assert len(filepaths) > 0, "No images to load"
        self.paths = filepaths
        self.N = len(filepaths)
        self.max_buffer = int(max_buffer)
        self.preprocess = preprocess
        self.read_flags = read_flags

        # public-ish state
        self.loop = bool(loop)
        self.stride = 1  # frames to advance in worker

        # thread & buffers
        self._q: "queue.Queue[Tuple[int, np.ndarray, int]]" = queue.Queue(max_buffer)
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._idx = max(0, min(start_index, self.N - 1))
        self._stop = False
        self._generation = 0  # bump on seek so worker drops stale frames

        self._th = threading.Thread(target=self._worker, name="BufferedImageLoader", daemon=True)

    # ----------------- public API -----------------
    def start(self):
        self._th.start()
        return self

    def stop(self):
        with self._lock:
            self._stop = True
            self._generation += 1  # invalidate everything
            self._cv.notify_all()
        # unblock queue.get waiting users by putting sentinels if needed
        try:
            self._q.put_nowait((-1, None, self._generation))
        except queue.Full:
            pass
        self._th.join(timeout=1.0)

    def seek(self, index: int, clear_buffer: bool = True):
        """Jump to an absolute index; optionally clear buffered frames."""
        index = int(max(0, min(index, self.N - 1)))
        with self._lock:
            self._idx = index
            self._generation += 1  # invalidate frames produced for old timeline
            if clear_buffer:
                self._drain_queue_unlocked()
            self._cv.notify_all()

    def step(self, delta: int):
        """Relative seek (e.g., z/c keys)."""
        with self._lock:
            self._idx = max(0, min(self._idx + int(delta), self.N - 1))
        self.seek(self._idx)  # reuse logic

    def set_stride(self, stride: int):
        """Allow rewind with negative strides; abs(stride) >= 1."""
        s = int(stride)
        if s == 0:
            s = 1
        self.stride = -max(1, abs(s)) if s < 0 else max(1, abs(s))
        with self._lock:
            self._generation += 1
            # Do NOT forcibly drain here; re-align is handled via seek() on direction changes
            self._cv.notify_all()

    def get_next(self, timeout: Optional[float] = 0.2) -> Optional[Tuple[int, np.ndarray]]:
        """
        Pop the next (index, frame) from buffer.
        Returns None if stopped or timed out.
        Drops stale frames produced before the most recent seek/stride change.
        """
        deadline = None if timeout is None else (time.monotonic() + timeout)

        while True:
            # time budget handling
            if deadline is None:
                remaining = None
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None

            try:
                idx, img, gen = self._q.get(timeout=remaining)
            except queue.Empty:
                return None

            # sentinel/shutdown
            if idx < 0 or img is None:
                return None

            # drop stale frames
            with self._lock:
                if gen != self._generation:
                    continue

            return idx, img

    # ----------------- worker logic -----------------
    def _worker(self):
        while True:
            with self._lock:
                # graceful exit?
                if self._stop:
                    return
                idx = self._idx
                gen = self._generation
                stride = self.stride
            # fill buffer unless it’s nearly full
            if self._q.qsize() >= self.max_buffer - 2:
                # light sleep lets UI thread run and queue drain
                time.sleep(0.000)
                continue

            # decode
            path = self.paths[idx]
            img = imread(path, self.read_flags)
            if img is None:
                # broken image: emit blank to keep timeline consistent
                img = np.zeros((480, 640, 3), np.uint8)

            if self.preprocess is not None:
                try:
                    img = self.preprocess(img)
                except Exception:
                    # avoid killing the worker on preprocess error
                    pass

            # publish if still current; otherwise drop
            with self._lock:
                if gen == self._generation and not self._stop:
                    try:
                        self._q.put_nowait((idx, img, gen))
                    except queue.Full:
                        # buffer raced to full; try later
                        pass

                # advance index for next iteration
                if gen == self._generation:
                    next_idx = idx + stride
                    if next_idx >= self.N:
                        next_idx = next_idx % self.N if self.loop else self.N - 1
                    elif next_idx < 0:
                        next_idx = (next_idx + self.N) % self.N if self.loop else 0
                    self._idx = next_idx

            # small cooperative yield
            time.sleep(0.0)

    def _drain_queue_unlocked(self):
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
