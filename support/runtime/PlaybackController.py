from __future__ import annotations

import time
import threading
from collections import deque
from pathlib import Path
from typing import Any

import customtkinter as ctk
import cv2
import numpy as np
from tkinter import TclError

import support.gui.utils as utils
import support.io.data_processing as data
from support.core.enums import PlaybackSpeed
from support.io.my_logging import LOG

SPEED_STEP = pow(2.0, 1.0 / 3.0)  # 3 presses -> 2×
SPEED_STEP_INV = 1.0 / SPEED_STEP


class PlaybackController:
    """Owns folder-playback UI and runtime behavior for CameraGui.

    The controller is intentionally thin in its public API:
      - setup_frame() builds the playback section UI
      - run_folder_reader() runs folder-based playback
      - draw_playback_stats() overlays playback HUD stats
      - populate_ids_times() is reused by the batch-processing path

    Everything else is private playback machinery.
    """

    def __init__(self, owner: Any):
        self.owner = owner

        # Public-ish state used by playback internals
        self.pause_cache = utils.PausedCache()
        self.playback = utils.PlaybackState()
        self.playback_mode_text = ctk.StringVar(value="Playback Mode: FPS")

        self.low_pass_fps = 20.0
        self.curr_fps = 20.0
        self.pause = False
        self.last_nonzero_sign = 1
        self._resume_speed_mag = 1.0
        self.fps_time_log = time.time()

        # Playback slider / command queue state
        self._pb_frame_label = None
        self._pb_frame_text = None
        self._pb_slider = None
        self._pb_num_images = 0
        self._pb_slider_dragging = False
        self._pb_last_sent_idx = None
        self._pb_slider_range_inited = False
        self._pb_cmds = deque()
        self._pb_cmd_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def setup_frame(self) -> None:
        """Construct the playback controls section of the GUI."""
        f = self.owner.playback_frame

        for w in f.winfo_children():
            w.destroy()

        row_id = 0
        self.update_playback_menu()

        playback_label = ctk.CTkLabel(f, textvariable=self.playback_mode_text)
        playback_label.grid(row=row_id, column=0, sticky="w", padx=8, pady=(8, 4))
        row_id += 1

        self._pb_frame_text = ctk.StringVar(value="Frame: — / —")
        self._pb_slider_dragging = False

        self._pb_frame_label = ctk.CTkLabel(f, textvariable=self._pb_frame_text)
        self._pb_frame_label.grid(row=row_id, column=0, sticky="w", padx=8, pady=(4, 2))
        row_id += 1

        self._pb_slider = ctk.CTkSlider(
            f,
            from_=0,
            to=1,
            number_of_steps=1,
            command=self._on_pb_slider_drag,
        )
        self._pb_slider.grid(row=row_id, column=0, sticky="ew", padx=8, pady=(0, 8))
        row_id += 1

        self._pb_slider.bind("<ButtonPress-1>", lambda *_: self._set_pb_slider_dragging(True))
        self._pb_slider.bind("<ButtonRelease-1>", self._on_pb_slider_release)

        f.grid_columnconfigure(0, weight=1)
        row_id += 1

        btn_frame = ctk.CTkFrame(f)
        btn_frame.grid(row=row_id, column=0, padx=5, pady=5, sticky="nsew")

        def mk(text: str, action: str, *args: Any, col: int = 0):
            b = ctk.CTkButton(
                btn_frame,
                text=text,
                command=lambda a=action, ar=args: self._enqueue_playback_cmd(a, *ar),
            )
            b.grid(row=0, column=col, padx=4, pady=4, sticky="nsew")
            return b

        mk("⟲ Rev (r)", "reverse", col=0)
        mk("⟸ Back (z)", "step_back", col=1)
        mk("⏯ Pause (space)", "toggle_pause", col=2)
        mk("Fwd (c) ⟹", "step_forward", col=3)
        mk("Mode (f)", "toggle_fps_mode", col=4)

        row_id += 1
        speed_frame = ctk.CTkFrame(f)
        speed_frame.grid(row=row_id, column=0, padx=5, pady=5, sticky="nsew")

        def mk2(text: str, action: str, *args: Any, col: int = 0):
            ctk.CTkButton(
                speed_frame,
                text=text,
                command=lambda a=action, ar=args: self._enqueue_playback_cmd(a, *ar),
            ).grid(row=0, column=col, padx=4, pady=4, sticky="nsew")

        mk2("Slower (a)", "speed_down", col=0)
        mk2("Faster (d)", "speed_up", col=1)
        mk2("Overlays (w)", "toggle_overlays", col=2)
        mk2("Mark Start (s)", "mark_start", col=3)
        mk2("Mark End (e)", "mark_end", col=4)

    def populate_ids_times(self, directory: str | Path) -> None:
        d = Path(directory)
        log_list = [str(p) for p in d.glob("*.log")]
        if not self.owner.ImageTimeReader.loadLog(log_list):
            self.owner.ImageTimeReader.idsTimes = []
            image_list = list(d.glob("*.bmp")) + list(d.glob("*.png"))
            image_list = data.natural_sort([str(p) for p in image_list])
            for image in image_list:
                self.owner.ImageTimeReader.idsTimes.append([image, None])

    def run_folder_reader(self) -> None:
        """Run the main folder-based playback and analysis loop."""
        loader = None
        try:
            try:
                cv2.destroyWindow(self.owner.windowName)
            except cv2.error:
                pass
            cv2.namedWindow(self.owner.windowName, cv2.WINDOW_NORMAL)

            directory = Path(self.owner.camConfig.imageFilepath).parent
            paths, t = self._build_sequence_and_timebase(directory)
            num_images = len(paths)

            self.playback.speed = 1
            self.pause = False
            self.last_nonzero_sign = 1
            last_speed = self.playback.speed

            if num_images > 0:
                self.playback.curr_idx = int(
                    max(0, min(int(self.playback.curr_idx), num_images - 1))
                )
            else:
                self.playback.curr_idx = 0

            prev_n = int(self._pb_num_images or 0)
            need_range_update = (not self._pb_slider_range_inited) or (prev_n != int(num_images))
            if need_range_update:
                self._pb_slider_range_inited = True
                self._pb_num_images = int(num_images)
                if num_images > 0:
                    self.playback.curr_idx = int(
                        max(0, min(int(self.playback.curr_idx), num_images - 1))
                    )
                else:
                    self.playback.curr_idx = 0
                try:
                    self.owner.after(0, self._pb_ui_set_slider_range, int(num_images))
                    self.owner.after(
                        0,
                        self._pb_ui_set_slider_pos,
                        int(self.playback.curr_idx),
                        int(num_images),
                    )
                except TclError:
                    pass

            self.pause_cache.clear()

            # Load persisted/base offset, but keep UI delta at zero on startup.
            loaded_offset = self.load_time_offset(directory)

            if self.owner.hud_marker is not None:
                self.owner.hud_marker.update_offset(loaded_offset)

            self.owner.camConfig.cam_to_log_time_offset = 0.0

            from support.runtime.buffer_image_loader import BufferedImageLoader as imgBuf

            loader = imgBuf(
                filepaths=[str(p) for p in paths],
                max_buffer=96,
                preprocess=None,
                start_index=self.playback.curr_idx,
                loop=True,
                read_flags=cv2.IMREAD_COLOR,
            ).start()

            pending_keys: list[int] = []
            wall_start = time.monotonic()

            if (
                self.owner.camConfig.playback_mode == PlaybackSpeed.Real_time
                and num_images > 0
                and len(t) > 0
            ):
                rs = float(self.owner.camConfig.rt_speed) or 1e-6
                wall_start = time.monotonic() - (t[self.playback.curr_idx] / rs)

            if self.owner.camConfig.playback_mode == PlaybackSpeed.Fixed_fps and num_images > 0:
                period = 1.0 / max(0.001, float(self.owner.camConfig.target_fps))
                wall_start = time.monotonic() - period * float(self.playback.curr_idx)

            self.update_playback_menu()

            def sleep_until(deadline: float) -> list[int]:
                keys: list[int] = []
                while True:
                    remain = deadline - time.monotonic()
                    if remain <= 0:
                        break
                    slice_ms = min(8, max(1, int(remain * 1000)))
                    keys.extend(self._poll_keys(slice_ms))
                return keys

            while (
                not self.owner.threadStopper.is_set()
                and self.owner.showWindow
                and not self.owner.making_gifOrVid
                and self.window_is_open()
            ):
                curr_time = time.time()
                if (curr_time - self.fps_time_log) > 0.000001:
                    self.curr_fps = 1.0 / abs(curr_time - self.fps_time_log)
                else:
                    self.curr_fps = 0.0
                self.fps_time_log = curr_time

                if self.playback.speed != last_speed:
                    s_abs = self._stride_for_speed(abs(self.playback.speed))

                    if self.playback.speed != 0:
                        self.last_nonzero_sign = 1 if self.playback.speed > 0 else -1

                    if s_abs == 0:
                        self.pause = True
                    else:
                        self.pause = False
                        signed_stride = self.last_nonzero_sign * s_abs
                        if signed_stride != self.playback.stride:
                            loader.set_stride(signed_stride)
                            self.playback.stride = int(signed_stride)

                        self.playback.curr_idx = max(0, min(self.playback.curr_idx, num_images - 1))
                        loader.seek(self.playback.curr_idx, clear_buffer=True)
                        self.pause_cache.clear()

                    last_speed = self.playback.speed

                if not self.pause:
                    got = loader.get_next(timeout=0.02)

                    if abs(self.playback.stride) > 1 and got is not None:
                        latest = got
                        while True:
                            nxt = loader.get_next(timeout=0.0)
                            if nxt is None:
                                break
                            latest = nxt
                        got = latest

                    frame = None
                    if got is not None:
                        got_idx, frame = got
                        self.playback.curr_idx = got_idx

                    self.pause_cache.clear()
                else:
                    target_idx = max(0, min(self.playback.curr_idx, num_images - 1))
                    if self.pause_cache.frame is None or self.pause_cache.idx != target_idx:
                        loader.seek(target_idx, clear_buffer=True)
                        got = loader.get_next(timeout=0.5)
                        if got is not None:
                            got_idx, frame = got
                            self.pause_cache.set(got_idx, frame)
                            self.playback.curr_idx = got_idx
                        else:
                            p = paths[target_idx]
                            if not Path(p).exists():
                                self.pause_cache.clear()
                    frame = self.pause_cache.frame

                if (
                    frame is not None
                    and Path(paths[self.playback.curr_idx]).exists()
                    and len(self.owner.ImageTimeReader.idsTimes) > 0
                ):
                    if (
                        self.owner.camConfig.playback_mode == PlaybackSpeed.Fixed_fps
                        and not self.pause
                    ):
                        period = 1.0 / self.owner.camConfig.target_fps
                        if self.last_nonzero_sign >= 0:
                            img_idx = self.playback.curr_idx
                        else:
                            img_idx = num_images - self.playback.curr_idx
                        target_time = wall_start + period * img_idx
                        if target_time < time.monotonic():
                            wall_start = (
                                time.monotonic()
                                - (1.0 / max(0.001, self.owner.camConfig.target_fps)) * img_idx
                            )
                        pending_keys.extend(sleep_until(target_time))

                    elif (
                        self.owner.camConfig.playback_mode == PlaybackSpeed.Real_time
                        and not self.pause
                    ):
                        rs = float(self.owner.camConfig.rt_speed) or 1e-6
                        elapsed = (time.monotonic() - wall_start) * rs
                        elapsed_ref = (t[-1] - elapsed) if self.last_nonzero_sign < 0 else elapsed

                        if elapsed_ref < t[0]:
                            idx_target = num_images - 1
                            wall_start = time.monotonic() - (
                                (t[idx_target] - t[0]) / rs if self.last_nonzero_sign >= 0
                                else ((t[-1] - t[idx_target]) / rs)
                            )
                        elif elapsed_ref > t[-1]:
                            idx_target = 0
                            wall_start = time.monotonic() - (
                                (t[idx_target] - t[0]) / rs if self.last_nonzero_sign >= 0
                                else ((t[-1] - t[idx_target]) / rs)
                            )
                        else:
                            idx_target = int(np.searchsorted(t, elapsed_ref, side="right") - 1)

                        idx_target = max(0, min(idx_target, num_images - 1))
                        if idx_target != self.playback.curr_idx:
                            loader.seek(idx_target, clear_buffer=True)
                            got = loader.get_next(timeout=0.02)
                            if got is not None:
                                self.playback.curr_idx, frame = got

                        pending_keys.extend(self._poll_keys(1))

                    ts = self.owner.ImageTimeReader.idsTimes[self.playback.curr_idx][1]
                    box_around = (
                        self.owner.camConfig.start_export_idx
                        <= self.playback.curr_idx
                        <= self.owner.camConfig.end_export_idx
                    )
                    if not self.window_is_open():
                        self.owner.threadStopper.set()
                        break
                    name = self.owner.ImageTimeReader.idsTimes[self.playback.curr_idx][0]
                    if ts is None:
                        self.owner.analyze_image(frame, None, name, box_around=box_around)
                    else:
                        self.owner.analyze_image(
                            frame,
                            ts + self.owner.camConfig.cam_to_log_time_offset,
                            name,
                            box_around=box_around,
                        )

                pending_keys.extend(self._poll_keys(1))

                for (action, args) in self._drain_playback_cmds():
                    self.playback.curr_idx, wall_start = self._apply_playback_action(
                        action,
                        args,
                        loader=loader,
                        curr_idx=self.playback.curr_idx,
                        t=t,
                        wall_start=wall_start,
                    )

                while pending_keys:
                    key = pending_keys.pop(0)
                    if key == 27:
                        self.owner.after(0, self.owner.filepath_page.toggle_stream)
                        self.owner.threadStopper.set()
                        break

                    action, args = self._key_to_playback_action(key)
                    if action is not None:
                        self.playback.curr_idx, wall_start = self._apply_playback_action(
                            action,
                            args,
                            loader=loader,
                            curr_idx=self.playback.curr_idx,
                            t=t,
                            wall_start=wall_start,
                        )

                    while self.owner.making_gifOrVid:
                        time.sleep(0.1)

                if self._pb_num_images and self._pb_last_sent_idx != self.playback.curr_idx:
                    self._pb_last_sent_idx = self.playback.curr_idx
                    try:
                        self.owner.after(
                            0,
                            self._pb_ui_set_slider_pos,
                            int(self.playback.curr_idx),
                            int(num_images),
                        )
                    except TclError:
                        pass

                pending_keys.extend(self._poll_keys(1))
                if not self.window_is_open():
                    self.owner.after(0, self.owner.filepath_page.toggle_stream)
                    self.owner.threadStopper.set()
                    break

        finally:
            try:
                cv2.destroyWindow(self.owner.windowName)
            except cv2.error:
                pass
            self.owner.after(0, self.owner.on_worker_exit)
            if loader is not None:
                loader.stop()

    def draw_playback_stats(self, frame, markup_frame, ctx, args) -> None:
        from support.viz.HUD_draw import HUD_Marker

        if self.owner.hud_marker is None:
            self.owner.hud_marker = HUD_Marker()
            self.owner.hud_marker.read_attitude_files(
                self.owner.camConfig.hud_data_filepath
            )

        self.low_pass_fps = 0.925 * self.low_pass_fps + 0.075 * self.curr_fps
        self.owner.hud_marker.draw_playbackStats(
            markup_frame,
            self.low_pass_fps,
            self.owner.camConfig.target_fps,
            self.owner.camConfig.playback_mode,
            self.owner.camConfig.rt_speed,
            self.owner.camConfig.cam_to_log_time_offset,
        )

    def update_playback_menu(self) -> None:
        if self.owner.camConfig.playback_mode == PlaybackSpeed.Fixed_fps:
            def mode(pause_status: bool, playback_speed: float) -> str:
                if pause_status:
                    return "Pause"
                if playback_speed < 0:
                    return "Rewind"
                return "Play"

            self.playback_mode_text.set(
                value=(
                    "Playback Mode: FPS\n"
                    f"Target FPS: {self.owner.camConfig.target_fps:.2f}\n"
                    f"{mode(self.pause, self.playback.speed)}"
                )
            )
        else:
            self.playback_mode_text.set(
                value=(
                    "Playback Mode: Realtime\n"
                    f"Playback Speed: {self.owner.camConfig.rt_speed:.2f}"
                )
            )

    def window_is_open(self) -> bool:
        try:
            return cv2.getWindowProperty(self.owner.windowName, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False

    # ------------------------------------------------------------------
    # Playback slider helpers
    # ------------------------------------------------------------------
    def _set_pb_slider_dragging(self, dragging: bool) -> None:
        self._pb_slider_dragging = bool(dragging)

    def _on_pb_slider_drag(self, value) -> None:
        try:
            v = int(round(float(value)))
        except (TypeError, ValueError):
            return
        n = int(self._pb_num_images or 0)

        if n > 0:
            v = max(0, min(v, n - 1))
            self._pb_frame_text.set(f"Frame: {v} / {n - 1}")
        else:
            self._pb_frame_text.set(f"Frame: {v} / —")

    def _on_pb_slider_release(self, _evt=None) -> None:
        self._set_pb_slider_dragging(False)
        try:
            v = int(round(float(self._pb_slider.get())))
        except (AttributeError, TypeError, ValueError):
            return
        self._enqueue_playback_cmd("seek_idx", v)

    def _pb_ui_set_slider_range(self, n: int) -> None:
        n = int(n)
        if n <= 1:
            self._pb_slider.configure(from_=0, to=1, number_of_steps=1)
            self._pb_frame_text.set("Frame: — / —")
            return

        self._pb_slider.configure(from_=0, to=n - 1, number_of_steps=n - 1)
        self._pb_frame_text.set(f"Frame: 0 / {n - 1}")

    def _pb_ui_set_slider_pos(self, idx: int, n: int) -> None:
        if self._pb_slider_dragging:
            return
        if self._pb_slider is None:
            return

        idx = int(max(0, min(int(idx), int(n) - 1)))
        self._pb_slider.set(idx)
        self._pb_frame_text.set(f"Frame: {idx} / {int(n) - 1}")

    # ------------------------------------------------------------------
    # Playback command queue
    # ------------------------------------------------------------------
    def _enqueue_playback_cmd(self, action: str, *args: Any) -> None:
        with self._pb_cmd_lock:
            self._pb_cmds.append((action, args))

    def _drain_playback_cmds(self) -> list[tuple[str, tuple[Any, ...]]]:
        out: list[tuple[str, tuple[Any, ...]]] = []
        with self._pb_cmd_lock:
            while self._pb_cmds:
                out.append(self._pb_cmds.popleft())
        return out

    # ------------------------------------------------------------------
    # Action mapping and application
    # ------------------------------------------------------------------
    def _apply_playback_action(
            self,
            action: str,
            args,
            *,
            loader,
            curr_idx: int,
            t,
            wall_start: float):
        num_images = len(t)

        if action == "toggle_fps_mode":
            self.pause = False
            self._on_toggle_fps_mode()
            wall_start = self._reanchor_on_mode_change(
                self.owner.camConfig.playback_mode, curr_idx, t
            )
            self.update_playback_menu()
            self.owner.saveToCache()

        elif action == "step_forward":
            curr_idx = self._on_step_forward(curr_idx, num_images)
            self.pause = True
            self.pause_cache.clear()
            self.owner.camConfig.playback_mode = PlaybackSpeed.Fixed_fps
            loader.seek(curr_idx, clear_buffer=True)
            self.update_playback_menu()

        elif action == "step_back":
            curr_idx = self._on_step_back(curr_idx)
            self.pause = True
            self.pause_cache.clear()
            self.owner.camConfig.playback_mode = PlaybackSpeed.Fixed_fps
            loader.seek(curr_idx, clear_buffer=True)
            self.update_playback_menu()

        elif action == "toggle_pause":
            wall_start = self._on_toggle_pause(curr_idx, t, wall_start)
            self.update_playback_menu()

        elif action == "speed_up":
            wall_start = self._on_speed_up(curr_idx=curr_idx, t=t)
            self.update_playback_menu()

        elif action == "speed_down":
            wall_start = self._on_speed_down(curr_idx=curr_idx, t=t)
            self.update_playback_menu()

        elif action == "mark_start":
            self._on_mark_start(curr_idx)

        elif action == "mark_end":
            self._on_mark_end(curr_idx)

        elif action == "reverse":
            wall_start = self._on_reverse(loader=loader, curr_idx=curr_idx, t=t)
            self.update_playback_menu()

        elif action == "seek_idx":
            (target_idx,) = args
            target_idx = int(max(0, min(int(target_idx), len(t) - 1)))
            curr_idx = target_idx
            loader.seek(curr_idx, clear_buffer=True)
            self.pause_cache.clear()
            wall_start = self._reanchor_on_mode_change(
                self.owner.camConfig.playback_mode, curr_idx, t
            )
            self.update_playback_menu()

        elif action == "bank_minus" and self.owner.hud_marker is not None:
            self.owner.hud_marker.cam_bank_offset -= 0.1
        elif action == "bank_plus" and self.owner.hud_marker is not None:
            self.owner.hud_marker.cam_bank_offset += 0.1
        elif action == "offset":
            (delta,) = args
            self._on_adjust_offset(float(delta))
        elif action == "persist_offset":
            self.write_offset_csv()

        return curr_idx, wall_start

    @staticmethod
    def _key_to_playback_action(key: int):
        if key == ord("f"):
            return "toggle_fps_mode", ()
        if key == ord("c"):
            return "step_forward", ()
        if key == ord("z"):
            return "step_back", ()
        if key == ord(" "):
            return "toggle_pause", ()
        if key == ord("d"):
            return "speed_up", ()
        if key == ord("a"):
            return "speed_down", ()
        if key == ord("s"):
            return "mark_start", ()
        if key == ord("e"):
            return "mark_end", ()
        if key == ord("r"):
            return "reverse", ()
        if key == ord("b"):
            return "bank_minus", ()
        if key == ord("n"):
            return "bank_plus", ()
        if key == ord(";"):
            return "offset", (-0.01,)
        if key == ord("'"):
            return "offset", (+0.01,)
        if key == ord(":"):
            return "offset", (-0.10,)
        if key == ord('"'):
            return "offset", (+0.10,)
        if key == ord("["):
            return "offset", (-1.00,)
        if key == ord("]"):
            return "offset", (+1.00,)
        if key == ord("{"):
            return "offset", (-10.00,)
        if key == ord("}"):
            return "offset", (+10.00,)
        if key == ord("p"):
            return "persist_offset", ()
        return None, None

    # ------------------------------------------------------------------
    # Timing / sequence helpers
    # ------------------------------------------------------------------
    def _reanchor_on_mode_change(self, new_mode, curr_idx: int, t) -> float:
        now = time.monotonic()

        if new_mode == PlaybackSpeed.Real_time:
            rs = max(1e-6, float(self.owner.camConfig.rt_speed))
            t0, tN = t[0], t[-1]
            if self.last_nonzero_sign < 0:
                return now - (tN - t[curr_idx]) / rs
            return now - (t[curr_idx] - t0) / rs

        fps = max(0.001, float(self.owner.camConfig.target_fps))
        num_images = len(t)
        phase = (num_images - curr_idx) if self.last_nonzero_sign < 0 else curr_idx
        return now - (phase / fps)

    @staticmethod
    def _rt_reanchor(now: float, curr_idx: int, t, rt_rate: float, sign: int) -> float:
        rt_rate = max(1e-6, float(rt_rate))
        t0, tN = t[0], t[-1]
        if sign < 0:
            return now - (tN - t[curr_idx]) / rt_rate
        return now - (t[curr_idx] - t0) / rt_rate

    def _on_reverse(self, loader, curr_idx: int, t):
        self.last_nonzero_sign = -1 if self.last_nonzero_sign > 0 else 1

        if self.playback.speed != 0:
            self.playback.speed = -self.playback.speed
            s_abs = self._stride_for_speed(abs(self.playback.speed))
            if s_abs > 0:
                loader.set_stride(self.last_nonzero_sign * s_abs)
                loader.seek(curr_idx, clear_buffer=True)

        now = time.monotonic()
        if self.owner.camConfig.playback_mode == PlaybackSpeed.Real_time:
            rs = max(1e-6, float(self.owner.camConfig.rt_speed))
            t0, tN = t[0], t[-1]
            if self.last_nonzero_sign < 0:
                wall_start = now - (tN - t[curr_idx]) / rs
            else:
                wall_start = now - (t[curr_idx] - t0) / rs
        else:
            fps = max(0.001, float(self.owner.camConfig.target_fps))
            num_images = len(t)
            phase = (num_images - curr_idx) if self.last_nonzero_sign < 0 else curr_idx
            wall_start = now - (phase / fps)
        return wall_start

    def _on_toggle_fps_mode(self) -> None:
        self.owner.camConfig.playback_mode = self.owner.camConfig.playback_mode.next()
        if self.owner.camConfig.playback_mode == PlaybackSpeed.Real_time:
            self.owner.camConfig.rt_speed = 1.0

    def _on_step_forward(self, curr_idx: int, num_images: int) -> int:
        curr_idx = min(curr_idx + 1, num_images - 1)
        self.playback.speed = 0.0
        return curr_idx

    def _on_step_back(self, curr_idx: int) -> int:
        curr_idx = max(curr_idx - 1, 0)
        self.playback.speed = 0.0
        return curr_idx

    def _on_toggle_pause(self, curr_idx: int, t, wall_start: float) -> float:
        self.pause = not self.pause
        playing = self.playback.speed != 0
        if playing:
            self._resume_speed_mag = max(1.0, abs(self.playback.speed))
            self.playback.speed = 0.0
            return wall_start

        prev_mag = getattr(self, "_resume_speed_mag", 1.0)
        self.playback.speed = float(self.last_nonzero_sign or 1) * prev_mag

        now = time.monotonic()
        if self.owner.camConfig.playback_mode == PlaybackSpeed.Real_time:
            rs = max(1e-6, float(self.owner.camConfig.rt_speed))
            t0, tN = t[0], t[-1]
            if self.last_nonzero_sign < 0:
                wall_start = now - (tN - t[curr_idx]) / rs
            else:
                wall_start = now - (t[curr_idx] - t0) / rs
        else:
            fps = max(0.001, float(self.owner.camConfig.target_fps))
            phase = (len(t) - curr_idx) if self.last_nonzero_sign < 0 else curr_idx
            wall_start = now - (phase / fps)
        return wall_start

    def _on_speed_up(self, curr_idx: int, t) -> float:
        if self.owner.camConfig.playback_mode == PlaybackSpeed.Real_time:
            prev_rt = self.owner.camConfig.rt_speed
            self.owner.camConfig.rt_speed = min(
                float(self.owner.camConfig.rt_speed) * SPEED_STEP, 128.0
            )
            if prev_rt < 0.99 and self.owner.camConfig.rt_speed > 1.0:
                self.owner.camConfig.rt_speed = 1.0
            now = time.monotonic()
            rs = max(1e-6, float(self.owner.camConfig.rt_speed))
            t0, tN = t[0], t[-1]
            if self.last_nonzero_sign < 0:
                wall_start = now - (tN - t[curr_idx]) / rs
            else:
                wall_start = now - (t[curr_idx] - t0) / rs
        else:
            prev_tgt = self.owner.camConfig.target_fps
            self.owner.camConfig.target_fps = min(
                float(self.owner.camConfig.target_fps) * SPEED_STEP, 320.0
            )
            if prev_tgt < 19.9 and self.owner.camConfig.target_fps > 20.0:
                self.owner.camConfig.target_fps = 20.0
            fps = max(0.001, float(self.owner.camConfig.target_fps))
            phase = (len(t) - curr_idx) if self.last_nonzero_sign < 0 else curr_idx
            wall_start = time.monotonic() - (phase / fps)
        return wall_start

    def _on_speed_down(self, curr_idx: int, t) -> float:
        if self.owner.camConfig.playback_mode == PlaybackSpeed.Real_time:
            prev_rt = self.owner.camConfig.rt_speed
            self.owner.camConfig.rt_speed = max(
                float(self.owner.camConfig.rt_speed) * SPEED_STEP_INV, 0.01
            )
            if prev_rt > 1.01 and self.owner.camConfig.rt_speed < 1.0:
                self.owner.camConfig.rt_speed = 1.0
            now = time.monotonic()
            rs = max(1e-6, float(self.owner.camConfig.rt_speed))
            t0, tN = t[0], t[-1]
            if self.last_nonzero_sign < 0:
                wall_start = now - (tN - t[curr_idx]) / rs
            else:
                wall_start = now - (t[curr_idx] - t0) / rs
        else:
            prev_tgt = self.owner.camConfig.target_fps
            self.owner.camConfig.target_fps = max(
                float(self.owner.camConfig.target_fps) * SPEED_STEP_INV, 0.1
            )
            if prev_tgt > 20.1 and self.owner.camConfig.target_fps < 20.0:
                self.owner.camConfig.target_fps = 20.0
            fps = max(0.001, float(self.owner.camConfig.target_fps))
            phase = (len(t) - curr_idx) if self.last_nonzero_sign < 0 else curr_idx
            wall_start = time.monotonic() - (phase / fps)
        return wall_start

    def _on_mark_start(self, curr_idx: int) -> None:
        self.owner.camConfig.start_export_idx = curr_idx
        if self.owner.camConfig.end_export_idx < self.owner.camConfig.start_export_idx:
            self.owner.camConfig.end_export_idx = self.owner.camConfig.start_export_idx + 1
        self.owner.exportStartFrame.configure(
            text=f"Start Frame: {self.owner.camConfig.start_export_idx}"
        )
        self.owner.exportEndFrame.configure(
            text=f"End Frame: {self.owner.camConfig.end_export_idx}"
        )
        self.owner.saveToCache()

    def _on_mark_end(self, curr_idx: int) -> None:
        self.owner.camConfig.end_export_idx = curr_idx
        if self.owner.camConfig.end_export_idx < self.owner.camConfig.start_export_idx:
            self.owner.camConfig.end_export_idx = max(0, self.owner.camConfig.end_export_idx - 1)
        self.owner.exportStartFrame.configure(
            text=f"Start Frame: {self.owner.camConfig.start_export_idx}"
        )
        self.owner.exportEndFrame.configure(
            text=f"End Frame: {self.owner.camConfig.end_export_idx}"
        )
        self.owner.saveToCache()

    def _on_adjust_offset(self, delta: float) -> None:
        self.owner.camConfig.cam_to_log_time_offset += float(delta)

    def write_offset_csv(self) -> None:
        if self.owner.hud_marker is None:
            return

        self.owner.hud_marker.update_offset(self.owner.camConfig.cam_to_log_time_offset)
        hud_path = Path(self.owner.camConfig.hud_data_filepath)
        if hud_path.is_dir():
            out_csv = hud_path / "__TIME_OFFSET.csv"
        else:
            out_csv = hud_path.parent / "__TIME_OFFSET.csv"

        out_csv.parent.mkdir(parents=True, exist_ok=True)

        import pandas as pd

        pd.DataFrame({"offset": [self.owner.hud_marker.offset]}).to_csv(out_csv, index=False)
        LOG.info(f"Saved offset {self.owner.camConfig.cam_to_log_time_offset:+.3f}s to {out_csv}")
        self.owner.camConfig.cam_to_log_time_offset = 0.0

    @staticmethod
    def _make_timebase(ts_raw, fallback_fps: float, n: int):
        t = np.array([np.nan if v is None else float(v) for v in ts_raw], dtype="float64")
        if n == 0:
            return t
        if np.all(np.isnan(t)):
            step = 1.0 / max(1e-6, float(fallback_fps))
            t = np.arange(n, dtype="float64") * step
        else:
            nans = np.isnan(t)
            if nans.any():
                notn = ~nans
                t[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(notn), t[notn])
        t -= float(t[0])
        return t

    @staticmethod
    def _stride_for_speed(speed_abs: float) -> float:
        s = max(0, int(speed_abs))
        if s == 0:
            return 0
        if s == 1:
            return 1
        return min(8.0, s)

    @staticmethod
    def _poll_keys(max_ms: int = 8) -> list[int]:
        k = cv2.waitKey(max_ms) & 0xFF
        if k not in (0, 0xFF, 255, -1):
            return [k]
        return []

    @staticmethod
    def load_time_offset(directory) -> float:
        try:
            import pandas as pd

            offset_dict = pd.read_csv(Path(directory) / "__TIME_OFFSET.csv")
            return float(offset_dict["offset"][0])
        except FileNotFoundError:
            return 0.0

    def _build_sequence_and_timebase(self, directory):
        self.populate_ids_times(str(directory))

        paths = []
        for rec in self.owner.ImageTimeReader.idsTimes:
            p = Path(rec[0])
            paths.append(p if p.is_absolute() else (Path(directory) / p))

        base_offset = 0.0
        if self.owner.hud_marker is not None:
            base_offset = float(getattr(self.owner.hud_marker, "offset", 0.0) or 0.0)

        delta_offset = float(getattr(self.owner.camConfig, "cam_to_log_time_offset", 0.0) or 0.0)
        total_offset = base_offset + delta_offset

        ts_raw = []
        for _name, ts in self.owner.ImageTimeReader.idsTimes:
            ts_raw.append(None if ts is None else float(ts) + total_offset)

        t = self._make_timebase(ts_raw, self.owner.camConfig.target_fps, len(paths))
        return paths, t
