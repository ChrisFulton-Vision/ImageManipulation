from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

import support.gui.utils as utils
import support.io.data_processing as data
from support.gui.gpu_monitor import GpuMonitor, GpuSample
from support.vision.calibration import undistort_points_px
from support.gui.qt_scheduler import QtValue


class GpuUtilSlider(QSlider):
    """Native Qt GPU indicator with the legacy runtime's narrow value API."""

    def configure(self, *, state: str | None = None, **_ignored: Any) -> None:
        if state is not None:
            self.setEnabled(state != "disabled")

    def set(self, value: float) -> None:
        self.setValue(int(round(float(value))))


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
        self._plot_btn = None
        self._progress = None
        self._gpu_var = None
        self._progress_label = None
        self._prefetch = None
        self._ckpt_n = None
        self._conf_list = None
        self._img_dir_var = None
        self.gpu_monitor = None
        self._plot_thread = None

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
                self.gpu_slider.setEnabled(enabled)
                if not enabled:
                    self.gpu_slider.setValue(0)
        except Exception:
            pass

    def setup_frame(self) -> None:
        """Build the batch-processing page for folder-based offline analysis."""
        f = self.owner.data_frame
        old_layout = f.layout()
        if old_layout is not None:
            while old_layout.count():
                item = old_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
        layout = QGridLayout() if old_layout is None else old_layout
        if old_layout is None:
            f.setLayout(layout)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(8)

        heading = QLabel("Batch YOLO over image folder")
        heading_font = heading.font()
        heading_font.setPointSize(12)
        heading_font.setBold(True)
        heading.setFont(heading_font)
        layout.addWidget(heading, 0, 0, 1, 3)

        img_dir_default = (
            getattr(self.owner.camConfig, "dp_img_dir", None)
            or self.owner.camConfig.imageFilepath
            or ""
        )
        self._img_dir_var = QtValue(str(img_dir_default), self.owner)
        self._mirror_runtime_refs()

        def _choose_dir():
            d = QFileDialog.getExistingDirectory(
                f,
                "Select image folder",
                str(self._current_img_dir()),
            )
            if d:
                self._img_dir_var.set(d)

        folder_edit = QLineEdit(str(self._img_dir_var.get()))
        folder_edit.textChanged.connect(self._img_dir_var.set)
        self._img_dir_var.changed.connect(folder_edit.setText)
        browse_button = QPushButton("Browse…")
        browse_button.clicked.connect(_choose_dir)
        layout.addWidget(QLabel("Folder:"), 1, 0)
        layout.addWidget(folder_edit, 1, 1)
        layout.addWidget(browse_button, 1, 2)

        conf_default = getattr(self.owner.camConfig, "dp_conf_list", "0.80")
        self._conf_list = QtValue(str(conf_default), self.owner)
        self._mirror_runtime_refs()
        conf_edit = QLineEdit(str(self._conf_list.get()))
        conf_edit.textChanged.connect(self._conf_list.set)
        self._conf_list.changed.connect(conf_edit.setText)
        layout.addWidget(QLabel("YOLO conf values (comma-separated):"), 3, 0)
        layout.addWidget(conf_edit, 3, 1, 1, 2)
        hint = QLabel("Example: 0.50, 0.65, 0.80 (defaults to 0.80 on bad input)")
        hint_font = hint.font()
        hint_font.setItalic(True)
        hint.setFont(hint_font)
        layout.addWidget(hint, 4, 0, 1, 3)

        ckpt_default = getattr(self.owner.camConfig, "dp_ckptN", 200)
        self._ckpt_n = QtValue(str(ckpt_default), self.owner)
        self._mirror_runtime_refs()
        checkpoint_edit = QLineEdit(str(self._ckpt_n.get()))
        checkpoint_edit.setMaximumWidth(140)
        checkpoint_edit.textChanged.connect(self._ckpt_n.set)
        self._ckpt_n.changed.connect(checkpoint_edit.setText)
        layout.addWidget(QLabel("Checkpoint every N images:"), 5, 0)
        layout.addWidget(checkpoint_edit, 5, 1)

        prefetch_default = getattr(self.owner.camConfig, "dp_prefetch", 32)
        self._prefetch = QtValue(str(prefetch_default), self.owner)
        self._mirror_runtime_refs()
        prefetch_edit = QLineEdit(str(self._prefetch.get()))
        prefetch_edit.setMaximumWidth(140)
        prefetch_edit.textChanged.connect(self._prefetch.set)
        self._prefetch.changed.connect(prefetch_edit.setText)
        layout.addWidget(QLabel("Prefetch images (count):"), 6, 0)
        layout.addWidget(prefetch_edit, 6, 1)

        self._progress_label = QLabel("Idle")
        layout.addWidget(self._progress_label, 20, 0, 1, 3)
        self._progress = QProgressBar()
        self._progress.setRange(0, 1000)
        self._progress.setValue(0)
        layout.addWidget(self._progress, 21, 0, 1, 3)

        gpu_display = bool(getattr(self.owner.camConfig, "dp_gpu", False))
        self._gpu_var = QtValue(gpu_display, self.owner)
        self._mirror_runtime_refs()
        gpu_checkbox = QCheckBox("Show GPU Util")
        gpu_checkbox.setChecked(gpu_display)
        gpu_checkbox.toggled.connect(self._gpu_var.set)
        self._gpu_var.changed.connect(lambda _value: self.toggle_show_gpu())
        layout.addWidget(gpu_checkbox, 25, 0)

        self.gpu_slider = GpuUtilSlider(Qt.Orientation.Horizontal)
        self.gpu_slider.setRange(0, 100)
        self.gpu_slider.setEnabled(gpu_display)
        self.gpu_slider.setValue(0)
        layout.addWidget(self.gpu_slider, 25, 1, 1, 2)
        self._mirror_runtime_refs()

        self._bind_dp_str(self._img_dir_var, "dp_img_dir")
        self._bind_dp_str(self._conf_list, "dp_conf_list")
        self._bind_dp_str(self._ckpt_n, "dp_ckptN")
        self._bind_dp_str(self._prefetch, "dp_prefetch")

        self._run_btn = QPushButton("Run YOLO Batch")
        self._run_btn.clicked.connect(self.run_yolo_batch_start)
        self._kalman_btn = QPushButton("Kalman Batch")
        self._kalman_btn.clicked.connect(self.run_kalman_batch_start)
        self._pnp_btn = QPushButton("SolvePnP/QnP Batch")
        self._pnp_btn.clicked.connect(self.run_pnp_qnp_on_folders_threaded)
        layout.addWidget(self._run_btn, 10, 0)
        layout.addWidget(self._kalman_btn, 10, 1)
        layout.addWidget(self._pnp_btn, 10, 2)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self.cancel)
        self._plot_btn = QPushButton("Plot")
        self._plot_btn.clicked.connect(self.plot_sequential_threaded)
        close_plot_button = QPushButton("Close Plots")
        close_plot_button.clicked.connect(self.close_plots)
        layout.addWidget(self._cancel_btn, 11, 0)
        layout.addWidget(self._plot_btn, 11, 1)
        layout.addWidget(close_plot_button, 11, 2)
        layout.setColumnStretch(1, 1)
        layout.setRowStretch(99, 1)

        self.toggle_show_gpu()

        self._mirror_runtime_refs()

    def on_gpu_sample(self, sample: GpuSample) -> None:
        """Consume a GPU utilization sample and refresh the display widget."""
        if sample.err:
            return
        if sample.util is not None and self.gpu_slider is not None:
            self.gpu_slider.setValue(int(round(sample.util)))

    def toggle_show_gpu(self) -> None:
        """Enable or disable GPU utilization monitoring from the UI."""
        enabled = bool(self._gpu_var.get()) if self._gpu_var is not None else False
        self.owner.camConfig.dp_gpu = enabled
        self.owner.saveToCache()

        if self.gpu_slider is not None:
            self.gpu_slider.setEnabled(enabled)
            if not enabled:
                self.gpu_slider.setValue(0)

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
            self._cancel_btn.setEnabled(False)
        if self._run_btn is not None:
            self._run_btn.setEnabled(False)

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
                if getattr(self.owner, "shutting_down", False):
                    self.close_plots()
                    return
            except Exception:
                self.close_plots()
                return

    def plot_sequential_threaded(self) -> None:
        """Run plotting on a background thread and keep the button state in sync."""
        if self._plot_thread is not None and self._plot_thread.is_alive():
            return

        self._set_plot_button_state(running=True)

        def _worker():
            try:
                self.plot_sequential()
            finally:
                self._post_to_ui(lambda: self._set_plot_button_state(running=False))

        self._plot_thread = threading.Thread(target=_worker, daemon=True)
        self._plot_thread.start()

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
        self.owner._dp_plot_btn = self._plot_btn
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
        if getattr(self.owner, "shutting_down", False):
            return
        try:
            self.owner.after(0, fn)
        except RuntimeError:
            pass

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
            self._progress_label.setText(text)

    def _set_progress(self, frac: float, text: str | None = None) -> None:
        if self._progress is not None:
            self._progress.setValue(int(round(max(0.0, min(1.0, float(frac))) * 1000)))
        if text is not None:
            self._set_status(text)

    def _set_plot_button_state(self, *, running: bool) -> None:
        if self._plot_btn is not None:
            self._plot_btn.setText("Plotting" if running else "Plot")
            self._plot_btn.setEnabled(not running)

    def _set_run_button_state(self, running: bool) -> None:
        if self._run_btn is not None:
            self._run_btn.setEnabled(not running)
        if self._cancel_btn is not None:
            self._cancel_btn.setEnabled(running)

    def _enable_run_buttons_after_finish(self) -> None:
        if self._run_btn is not None:
            self._run_btn.setEnabled(True)
        if self._cancel_btn is not None:
            self._cancel_btn.setEnabled(False)

    def _finish_run(self, text: str) -> None:
        self._set_status(text)
        self._enable_run_buttons_after_finish()