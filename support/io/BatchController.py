from __future__ import annotations

import os
import threading
from pathlib import Path
from tkinter import filedialog
from typing import Any, Callable

import customtkinter as ctk

import support.gui.utils as utils
import support.io.data_processing as data
import support.viz.colors as clr
from support.gui.gpu_monitor import GpuMonitor, GpuSample
from support.vision.calibration import undistort_points_px


class BatchController:
    """Owns batch-processing UI and offline data-product generation for CameraGui.

    The controller keeps the first migration deliberately low-risk:
      - it composes around the existing CameraGui instance (`owner`)
      - it mirrors key widget/runtime refs back onto `owner` for compatibility
      - it delegates heavy lifting to the existing DataProcessorRunner and Plotter

    Public API:
      - setup_frame(): build the batch-processing page UI
      - sync_from_model(): refresh DP widgets from camConfig
      - run_yolo_batch_start(): launch YOLO CSV generation
      - run_kalman_batch_start(): launch KF sweep
      - run_pnp_qnp_on_folders_threaded(): launch SolvePnP/QnP sweep
      - run_pnp_qnp_from_detection_csv(): bridge CSV -> pose solver path
      - cancel(): request cooperative cancellation
      - close_plots(): close Plotter windows
      - stop(): stop background monitor(s) owned here
    """

    def __init__(self, owner: Any):
        self.owner = owner

        self.gpu_slider = None
        self._runner = None
        self._worker_thread = None
        self._cancel_btn = None
        self._pnp_btn = None
        self._kalman_btn = None
        self._run_btn = None
        self._progress = None
        self._gpu_var = None
        self._progress_label = None
        self._prefetch = None
        self._ckpt_n = None
        self._conf_list = None
        self._img_dir_var = None
        self.gpu_monitor = None

        self._mirror_runtime_refs()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def sync_from_model(self) -> None:
        """Resync batch-processing UI controls from camConfig."""
        cam = self.owner.camConfig

        if self._img_dir_var is not None:
            self._img_dir_var.set(str(getattr(cam, "dp_img_dir", "") or ""))

        if self._conf_list is not None:
            self._conf_list.set(str(getattr(cam, "dp_conf_list", "") or ""))

        if self._ckpt_n is not None:
            self._ckpt_n.set(str(getattr(cam, "dp_ckptN", 200) or 200))

        if self._prefetch is not None:
            self._prefetch.set(str(getattr(cam, "dp_prefetch", 4) or 4))

        if self._gpu_var is not None:
            self._gpu_var.set(bool(getattr(cam, "dp_gpu", False)))

        try:
            if self.gpu_slider is not None:
                enabled = bool(getattr(cam, "dp_gpu", False))
                self.gpu_slider.configure(state="normal" if enabled else "disabled")
                if not enabled:
                    self.gpu_slider.set(0.0)
        except Exception:
            pass

    def setup_frame(self) -> None:
        """Build the batch-processing page for folder-based offline analysis."""
        f = self.owner.data_frame
        for w in f.winfo_children():
            w.destroy()

        f.grid_rowconfigure(99, weight=1)
        f.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(f, text="Batch YOLO over image folder", font=("Segoe UI", 16, "bold")).grid(
            row=0, column=0, columnspan=3, padx=12, pady=(16, 8), sticky="w"
        )

        img_dir_default = (
            getattr(self.owner.camConfig, "dp_img_dir", None)
            or self.owner.camConfig.imageFilepath
            or ""
        )
        self._img_dir_var = ctk.StringVar(value=str(img_dir_default))
        self._mirror_runtime_refs()

        def _choose_dir():
            d = filedialog.askdirectory(title="Select image folder")
            if d:
                self._img_dir_var.set(d)

        ctk.CTkLabel(f, text="Folder:").grid(row=1, column=0, padx=12, pady=6, sticky="w")
        ctk.CTkEntry(f, textvariable=self._img_dir_var).grid(row=1, column=1, padx=12, pady=6, sticky="ew")
        ctk.CTkButton(f, text="Browse…", command=_choose_dir).grid(row=1, column=2, padx=12, pady=6)

        conf_default = getattr(self.owner.camConfig, "dp_conf_list", "0.80")
        self._conf_list = ctk.StringVar(value=str(conf_default))
        self._mirror_runtime_refs()

        ctk.CTkLabel(f, text="YOLO conf values (comma-separated):").grid(
            row=3, column=0, padx=12, pady=6, sticky="w"
        )
        ctk.CTkEntry(f, textvariable=self._conf_list).grid(
            row=3, column=1, padx=12, pady=6, sticky="ew"
        )

        ctk.CTkLabel(
            f,
            text="Example: 0.50, 0.65, 0.80   (defaults to 0.80 on bad input)",
            font=("Segoe UI", 10, "italic"),
        ).grid(
            row=4, column=0, columnspan=3, padx=12, pady=(0, 6), sticky="w"
        )

        ckpt_default = getattr(self.owner.camConfig, "dp_ckptN", 200)
        self._ckpt_n = ctk.StringVar(value=str(ckpt_default))
        self._mirror_runtime_refs()
        ctk.CTkLabel(f, text="Checkpoint every N images:").grid(row=5, column=0, padx=12, pady=6, sticky="w")
        ctk.CTkEntry(f, textvariable=self._ckpt_n, width=100).grid(row=5, column=1, padx=12, pady=6, sticky="w")

        prefetch_default = getattr(self.owner.camConfig, "dp_prefetch", 32)
        self._prefetch = ctk.StringVar(value=str(prefetch_default))
        self._mirror_runtime_refs()
        ctk.CTkLabel(f, text="Prefetch images (count):").grid(row=6, column=0, padx=12, pady=6, sticky="w")
        ctk.CTkEntry(f, textvariable=self._prefetch, width=100).grid(row=6, column=1, padx=12, pady=6, sticky="w")

        self._progress_label = ctk.CTkLabel(f, text="Idle")
        self._progress_label.grid(row=20, column=0, columnspan=3, padx=12, pady=(8, 4), sticky="w")

        self._progress = ctk.CTkProgressBar(f)
        self._progress.grid(row=21, column=0, columnspan=3, padx=12, pady=(0, 8), sticky="ew")
        self._progress.set(0.0)

        gpu_display = bool(getattr(self.owner.camConfig, "dp_gpu", False))
        self._gpu_var = ctk.BooleanVar(value=gpu_display)
        self._mirror_runtime_refs()

        gpu_checkbox = ctk.CTkCheckBox(
            f,
            text="Show GPU Util",
            variable=self._gpu_var,
            command=self.toggle_show_gpu,
        )
        gpu_checkbox.grid(row=25, column=0, columnspan=1, padx=5, pady=5, sticky="ew")

        self.gpu_slider = ctk.CTkSlider(f, from_=0, to=100)
        self.gpu_slider.grid(row=25, column=1, columnspan=2, padx=5, pady=5, sticky="ew")
        self.gpu_slider.configure(state="disabled")
        self.gpu_slider.set(0)
        self._mirror_runtime_refs()

        # Preserve current behavior exactly: the page builder toggles this twice.
        self.toggle_show_gpu()
        self.toggle_show_gpu()

        self._bind_dp_str(self._img_dir_var, "dp_img_dir")
        self._bind_dp_str(self._conf_list, "dp_conf_list")
        self._bind_dp_str(self._ckpt_n, "dp_ckptN")
        self._bind_dp_str(self._prefetch, "dp_prefetch")

        self._run_btn = ctk.CTkButton(
            f,
            text="Run YOLO Batch",
            fg_color=clr.CTK_BUTTON_GREEN,
            command=self.run_yolo_batch_start,
        )
        self._run_btn.grid(row=10, column=0, padx=12, pady=(16, 12), sticky="ew")

        self._kalman_btn = ctk.CTkButton(
            f,
            text="Kalman Batch",
            command=self.run_kalman_batch_start,
        )
        self._kalman_btn.grid(row=10, column=1, padx=12, pady=(16, 12), sticky="ew")

        self._pnp_btn = ctk.CTkButton(
            f,
            text="SolvePnP/QnP Batch",
            command=self.run_pnp_qnp_on_folders_threaded,
        )
        self._pnp_btn.grid(row=10, column=2, padx=12, pady=(16, 12), sticky="ew")

        self._cancel_btn = ctk.CTkButton(
            f,
            text="Cancel",
            command=self.cancel,
            state="disabled",
        )
        self._cancel_btn.grid(row=11, column=0, columnspan=1, padx=12, pady=(0, 12), sticky="ew")

        dp_plotter_btn = ctk.CTkButton(
            f,
            text="Plot",
            command=self.plot_sequential,
        )
        dp_plotter_btn.grid(row=11, column=1, columnspan=1, padx=12, pady=(0, 12), sticky="ew")

        dp_close_plot_btn = ctk.CTkButton(
            f,
            text="Close Plots",
            command=self.close_plots,
        )
        dp_close_plot_btn.grid(row=11, column=2, columnspan=1, padx=12, pady=(0, 12), sticky="ew")

        self._mirror_runtime_refs()

    def on_gpu_sample(self, sample: GpuSample) -> None:
        """Consume a GPU utilization sample and refresh the display widget."""
        if sample.err:
            return
        if sample.util is not None and self.gpu_slider is not None:
            self.gpu_slider.set(sample.util)

    def toggle_show_gpu(self) -> None:
        """Enable or disable GPU utilization monitoring from the UI."""
        enabled = bool(self._gpu_var.get()) if self._gpu_var is not None else False
        self.owner.camConfig.dp_gpu = enabled
        self.owner.saveToCache()

        if self.gpu_slider is not None:
            self.gpu_slider.configure(state="normal" if enabled else "disabled")
            if not enabled:
                self.gpu_slider.set(0.0)

        if self.gpu_monitor is None:
            self.gpu_monitor = GpuMonitor(
                scheduler=self.owner,
                on_sample=self.on_gpu_sample,
                device_index=0,
                poll_ms=250,
            )
            self._mirror_runtime_refs()
        self.gpu_monitor.set_enabled(enabled)

    def run_pnp_qnp_on_folders_threaded(self) -> None:
        """Launch SolvePnP/QnP batch processing on a background thread."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        img_dir = self._current_img_dir()
        self._set_status("Starting SolvePnP/QnP…")

        runner = self._ensure_runner()
        runner.reset_cancel()
        runner.cancel_event.clear()

        conf_list_var = self._conf_list

        def post_status(text: str):
            self._post_to_ui(lambda: self._set_status(text))

        def post_progress(frac: float, text: str):
            self._post_to_ui(lambda: self._set_progress(frac, text))

        def _worker():
            try:
                runner.run_pnp_qnp_conf_sweep(
                    img_dir=img_dir,
                    conf_list_var=conf_list_var,
                    run_pnp_qnp_from_detection_csv=self.run_pnp_qnp_from_detection_csv,
                    post_progress=post_progress,
                    post_status=post_status,
                    sweep_timer=utils.SweepTimer(),
                    fmt_mmss=utils.fmt_mmss,
                )
            except Exception as e:
                post_status(f"SolvePnP/QnP failed: {e}")
            finally:
                self._post_to_ui(self._enable_run_buttons_after_finish)

        self._start_worker(_worker)

    def run_kalman_batch_start(self) -> None:
        """Start a Kalman-filter confidence sweep in a worker thread."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        self._set_progress(0.0, "Starting KF…")
        self._set_run_button_state(running=True)

        runner = self._ensure_runner()
        runner.reset_cancel()
        img_dir = self._current_img_dir()

        def progress_cb(frac: float, text: str) -> None:
            self._post_to_ui(lambda: self._set_progress(frac, text))

        def status_cb(text: str) -> None:
            self._post_to_ui(lambda: self._set_status(text))

        def finish(text: str) -> None:
            self._post_to_ui(lambda: self._finish_run(text))

        def _worker():
            try:
                calibration = self.owner.calibration
                if calibration is None or not getattr(calibration, "validCal", False):
                    finish("No calibration loaded; cannot run KF.")
                    return

                runner.run_kalman_conf_sweep(
                    img_dir=img_dir,
                    conf_list_var=self._conf_list,
                    calibration=calibration,
                    progress_cb=progress_cb,
                    status_cb=status_cb,
                )
                finish("KF sweep done.")
            except Exception as e:
                finish(f"KF sweep failed: {e}")

        self._start_worker(_worker)

    def cancel(self) -> None:
        """Request cancellation of the active batch-processing job."""
        if self._runner is not None:
            try:
                self._runner.cancel_event.set()
            except Exception:
                pass

        self._set_status("Canceling… (finishing current step)")
        if self._cancel_btn is not None:
            self._cancel_btn.configure(state="disabled")
        if self._run_btn is not None:
            self._run_btn.configure(state="disabled")

    def run_yolo_batch_start(self) -> None:
        """Start a YOLO confidence sweep over the selected image folder."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        self._set_progress(0.0, "Starting…")
        self._set_run_button_state(running=True)

        runner = self._ensure_runner()
        runner.reset_cancel()

        self._ensure_yolo_session()

        img_dir = self._current_img_dir()
        self.owner.populate_idsTimes(str(img_dir))
        pairs = list(getattr(self.owner.ImageTimeReader, "idsTimes", []))

        def post_progress(frac: float, text: str) -> None:
            self._post_to_ui(lambda: self._set_progress(frac, text))

        def post_status(text: str) -> None:
            self._post_to_ui(lambda: self._set_status(text))

        def post_finish(text: str) -> None:
            self._post_to_ui(lambda: self._finish_run(text))

        def _worker():
            try:
                out_csv_base = self._processed_data_dir() / "1_yolo_detections.csv"
                params = data.YoloSweepParams(
                    img_dir=img_dir,
                    out_csv_base=out_csv_base,
                    conf_list_var=self.get_conf_text(),
                    ckpt_every_var=self.get_ckpt_n(),
                    prefetch_var=self.get_prefetch_n(),
                    cam_to_log_time_offset=float(getattr(self.owner.camConfig, "cam_to_log_time_offset", 0.0)),
                )

                runner.run_yolo_conf_sweep(
                    yolo_session=self.owner.yoloSession,
                    calibration=self.owner.calibration,
                    ids_times_pairs=pairs,
                    params=params,
                    sweep_timer=utils.SweepTimer(),
                    fmt_mmss=utils.fmt_mmss,
                    post_progress=post_progress,
                    post_status=post_status,
                    post_finish=post_finish,
                    undistort_points_px=undistort_points_px,
                )
            except Exception as e:
                post_finish(f"YOLO batch failed: {e}")

        self._start_worker(_worker)

    def run_pnp_qnp_from_detection_csv(
        self,
        csv_path: str,
        out_pnp: str | None = None,
        out_qnp: str | None = None,
        progress_cb=None,
        cancel_cb: bool = False,
    ) -> None:
        """Run SolvePnP/QnP over an existing detection CSV using GUI state."""
        del cancel_cb  # preserved for signature compatibility

        checkpoint_every = 0
        try:
            if self._ckpt_n is not None:
                v = self._ckpt_n.get()
                if isinstance(v, str):
                    v = v.strip()
                checkpoint_every = int(v) if v else 0
        except Exception:
            checkpoint_every = 0

        if checkpoint_every <= 0:
            try:
                checkpoint_every = int(getattr(self.owner.camConfig, "dp_ckptN", 0) or 0)
            except Exception:
                checkpoint_every = 0

        self._ensure_yolo_session()

        truth_dict = getattr(getattr(self.owner.yoloSession, "reader", None), "idsNamesLocs", None)
        if truth_dict is None:
            raise ValueError("yoloSession.reader.idsNamesLocs is missing (metaYolo not loaded?).")

        cancel_event = None
        if self._runner is not None and getattr(self._runner, "cancel_event", None) is not None:
            cancel_event = self._runner.cancel_event

        self._ensure_runner().run_pnp_qnp_from_detection_csv(
            csv_path=str(csv_path),
            calibration=self.owner.calibration,
            truth_dict=truth_dict,
            checkpoint_every=checkpoint_every,
            out_pnp=out_pnp,
            out_qnp=out_qnp,
            progress_cb=progress_cb,
            cancel_event=cancel_event,
        )

    def plot_sequential(self) -> None:
        """Render saved plots for each configured confidence sweep value."""
        from support.viz.Plotting import Plotter

        if self.owner.plotter is None:
            self.owner.plotter = Plotter()

        parse_vars = data.parse_conf_list(self._conf_list)
        img_dir = self._processed_data_dir()

        for var in parse_vars:
            self.owner.plotter.plot(var, img_dir, False, True)
            try:
                self.owner.winfo_exists()
            except Exception:
                self.close_plots()
                return

    def close_plots(self) -> bool:
        """Close any open plotting windows through the plotting helper."""
        if self.owner.plotter is None:
            return False

        from support.viz.Plotting import Plotter

        self.owner.plotter = Plotter()
        self.owner.plotter.close_plot()
        return True

    def stop(self) -> None:
        """Stop background resources owned by the batch controller."""
        if self.gpu_monitor is not None:
            self.gpu_monitor.stop()

    # ------------------------------------------------------------------
    # Helpers kept public-ish for thin owner wrappers
    # ------------------------------------------------------------------
    def get_conf_text(self) -> str:
        try:
            return str(self._conf_list.get()).strip()
        except Exception:
            return str(getattr(self.owner.camConfig, "dp_conf_list", "0.80") or "0.80")

    def get_ckpt_n(self) -> int:
        try:
            return int(str(self._ckpt_n.get()).strip())
        except Exception:
            try:
                return int(getattr(self.owner.camConfig, "dp_ckptN", 200) or 200)
            except Exception:
                return 200

    def get_prefetch_n(self) -> int:
        try:
            return int(str(self._prefetch.get()).strip())
        except Exception:
            try:
                return int(getattr(self.owner.camConfig, "dp_prefetch", 32) or 32)
            except Exception:
                return 32

    def get_gpu_enabled(self) -> bool:
        try:
            return bool(self._gpu_var.get())
        except Exception:
            return bool(getattr(self.owner.camConfig, "dp_gpu", False))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _mirror_runtime_refs(self) -> None:
        """Mirror controller-owned refs back onto owner for low-risk migration."""
        self.owner.gpu_slider = self.gpu_slider
        self.owner._dp_runner = self._runner
        self.owner._dp_worker = self._worker_thread
        self.owner._dp_cancel_btn = self._cancel_btn
        self.owner._dp_pnp_btn = self._pnp_btn
        self.owner._dp_kalman_btn = self._kalman_btn
        self.owner._dp_run_btn = self._run_btn
        self.owner._dp_progress = self._progress
        self.owner._dp_gpu_var = self._gpu_var
        self.owner._dp_progress_label = self._progress_label
        self.owner._dp_prefetch = self._prefetch
        self.owner._dp_ckptN = self._ckpt_n
        self.owner._dp_conf_list = self._conf_list
        self.owner._dp_img_dir_var = self._img_dir_var
        self.owner.gpu_monitor = self.gpu_monitor

    def _bind_dp_str(self, var: Any, attr_name: str) -> None:
        if var is None:
            return

        def _on_change(*_):
            try:
                setattr(self.owner.camConfig, attr_name, var.get())
                self.owner.saveToCache()
            except Exception:
                pass

        var.trace_add("write", _on_change)

    def _post_to_ui(self, fn: Callable[[], None]) -> None:
        self.owner.after(0, lambda *_: fn(), ())

    def _current_img_dir(self) -> Path:
        p = Path(
            (self._img_dir_var and self._img_dir_var.get().strip())
            or (getattr(self.owner.camConfig, "imageFilepath", "") or "")
        )

        # If a file path sneaks in here, batch work should still land beside it.
        if p.suffix:
            return p.parent
        return p

    def _processed_data_dir(self) -> Path:
        return self._current_img_dir() / "_ProcessedData"

    def _ensure_runner(self):
        if self._runner is None:
            self._runner = data.DataProcessorRunner()
            self._mirror_runtime_refs()
        return self._runner

    def _ensure_yolo_session(self) -> None:
        from support.vision import yolo

        if self.owner.yoloSession is None:
            self.owner.yoloSession = yolo.YOLO()
            self.owner.yoloSession.setNewFolder(self.owner.camConfig.yoloFilepath)
            self.owner.yoloSession.set_calibration(self.owner.calibration)
            self.owner.yoloSession.iou = self.owner.camConfig.yolo_iou

    def _start_worker(self, target: Callable[[], None]) -> None:
        self._worker_thread = threading.Thread(target=target, daemon=True)
        self._mirror_runtime_refs()
        self._worker_thread.start()

    def _set_status(self, text: str) -> None:
        if self._progress_label is not None:
            self._progress_label.configure(text=text)

    def _set_progress(self, frac: float, text: str | None = None) -> None:
        if self._progress is not None:
            self._progress.set(float(frac))
        if text is not None:
            self._set_status(text)

    def _set_run_button_state(self, running: bool) -> None:
        if self._run_btn is not None:
            self._run_btn.configure(state="disabled" if running else "normal")
        if self._cancel_btn is not None:
            self._cancel_btn.configure(state="normal" if running else "normal")

    def _enable_run_buttons_after_finish(self) -> None:
        if self._run_btn is not None:
            self._run_btn.configure(state="normal")
        if self._cancel_btn is not None:
            self._cancel_btn.configure(state="normal")

    def _finish_run(self, text: str) -> None:
        self._set_status(text)
        self._enable_run_buttons_after_finish()
