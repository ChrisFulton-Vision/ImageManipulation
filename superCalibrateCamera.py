import copy
import os
import time
import sys
import threading
import enum
from dataclasses import dataclass

import numpy as np

from numpy.typing import NDArray
from typing import List, Any, Callable
from pathlib import Path

import customtkinter as ctk
import cv2

from support.core.enums import ExportQuality, ImageKernel, ImageSource

import support.gui.CTKCamFilepathPage as Filepath_page
import support.gui.CTKCamImageProcessingPage as Image_processing_page
import support.gui.CTKHotkeyPage as Hotkey_page
import support.gui.utils as utils
from support.gui.checkerboard_launcher import CheckerboardLauncher, CheckerboardLaunchState
import support.gui.UserSelectQueue as GuiQueue

from support.io.camera_config import CameraConfig as CamConfig
from support.io.config_store import ConfigStore
from support.io.image_time_reader import ImageTimeReader
from support.io.my_logging import LOG
from support.io.BatchController import BatchController

from support.vision.calibration import Calibration
from support.vision.vimba_controller import VimbaController as VimbaCam
from support.vision.fisheye_to_cubemap import (
    DEFAULT_CUBEMAP_FACES,
    DEFAULT_CUBEMAP_LAYOUT,
    DEFAULT_CUBEMAP_FACESIZE,
    FisheyeCubemapManager)

from support.runtime.frame_processor import FrameProcessor
from support.runtime.PlaybackController import PlaybackController
from support.runtime.config_runtime import ConfigRuntime
from support.runtime.pose_runtime import PoseRuntime
from support.runtime.stream_runner import StreamRunner

import support.viz.colors as clr
from support.viz.CVFontScaling import small_text, med_text, med_thick, lrg_thick
from support.viz.checkerboard_stats import CheckerboardResiduals as CkR
from support.viz.draw_pnp_qnp import PoseOutput

from copy import deepcopy

# cv2.setNumThreads(0)
cv2.setUseOptimized(True)

#  pip install cv2_enumerate_cameras
#  or
#  pip install git+https://github.com/chinaheyu/cv2_enumerate_cameras.git


CACHE_FILEPATH = str(Path.cwd() / "Caches" / "last_config.pkl")


@dataclass(slots=True)
class YoloOutput:
    """Container for the most recent YOLO-derived measurement products."""
    last_bounding_box_size: tuple[float, float] = None
    last_yolo_center: tuple[float, float] = None
    last_yolo_3d_estimate: NDArray = None
    pose: PoseOutput | None = None


class CameraGui(ctk.CTkFrame):
    """Main CustomTkinter controller for camera calibration and image analysis.

    This widget manages UI pages, persistent configuration, live playback,
    batch processing, and the queued image-processing pipeline used to
    annotate or analyze frames from files, folders, or live cameras.
    """

    def __init__(self, master, *args, **kwargs):
        """Initialize UI state, runtime helpers, and processing pipeline options.

        Builds the controller state, creates child pages and frames, restores
        cached configuration, and wires the queue editor to the underlying
        image-processing model.
        """
        self.own_attitude = None
        self._playback_allowed = None
        self.curr_r_V_d = None
        self.curr_r_T_d = None
        self.last_fg_output = None

        self.gpu_slider = None

        self._dp_runner = None
        self._dp_cancel_btn = None
        self._dp_pnp_btn = None
        self._dp_kalman_btn = None
        self._dp_run_btn = None
        self._dp_progress = None
        self._dp_gpu_var = None
        self._dp_progress_label = None
        self._dp_prefetch = None
        self._dp_ckptN = None
        self._dp_conf_list = None
        self._dp_img_dir_var = None

        self.vimbaCam = VimbaCam()

        self._loading_config = True

        self.func_that_refits = None
        self.list_of_image_process_functors: List[tuple[Callable, dict]] = []

        # Debounced cache writes
        self._save_debounce_id = None

        # Super class init, necessary for customTkinter
        super().__init__(master, *args, **kwargs)
        self._flag_vars: dict[str, ctk.BooleanVar | ctk.DoubleVar] = {}

        self._flags = [
            'yolo_conf', 'yolo_iou'
        ]
        self.threadStopper = utils.ThreadStopper()
        self.camConfig: CamConfig = CamConfig()
        self.windowName = 'Processed Image'
        self.vc = None
        self._thread = None
        self.shutting_down = False
        self.stream_running_var = ctk.BooleanVar(value=False)

        self.recording = False
        self.yoloSession = None
        self.detector = None
        self.arucoDict = None
        self.arucoParams = None

        self.yolo_output_type = YoloOutput
        self.config_runtime = ConfigRuntime(self)
        self.frame_processor = FrameProcessor(self)
        self.pose_runtime = PoseRuntime(self)
        self.stream_runner = StreamRunner(self)

        self._init_flag_vars()

        self.step_options: List[GuiQueue.StepOption] = [
            GuiQueue.StepOption(label="Undistort",
                                fn=self.undistort,
                                arg_specs=GuiQueue.UndistortOpts.ARG_SPECS),
            GuiQueue.StepOption(label="Draw Chessboard",
                                fn=self.draw_chessboard,
                                arg_specs=()),
            GuiQueue.StepOption(label="Resize",
                                fn=self.resize_image,
                                arg_specs=GuiQueue.ResizeOpts.ARG_SPECS),
            GuiQueue.StepOption(
                label="Apply Image Filter",
                fn=self.applyKernel,
                arg_specs_fn=self.image_filter_arg_specs),
            GuiQueue.StepOption(label="Apply YOLO -> Q/PnP",
                                fn=self.run_yolo,
                                arg_specs=GuiQueue.YoloOpts.ARG_SPECS),
            GuiQueue.StepOption(label="Detect Corners in Image",
                                fn=self.detect_corners,
                                arg_specs=()),
            GuiQueue.StepOption(label="Show Phase Correlation",
                                fn=self.phase_correlation,
                                arg_specs=()),
            GuiQueue.StepOption(label="Attempt Horizon Detection",
                                fn=self.detectHorizon,
                                arg_specs=()),
            GuiQueue.StepOption(label="Draw HUD",
                                fn=self.draw_HUD,
                                arg_specs=GuiQueue.HudOpts.ARG_SPECS),
            GuiQueue.StepOption(
                label="Detect AprilTags and Q/PnP",
                fn=self.detectAprilTags,
                arg_specs=GuiQueue.AprilTagDetectOpts.ARG_SPECS,
            ),
        ]
        self.fn_to_label = {opt.fn: opt.label for opt in self.step_options}

        self.calibration = Calibration()
        self.config_store = ConfigStore(CACHE_FILEPATH, configs_dir="Configs", scheduler=self)
        self.detectIDS = None
        self.centers = None
        self.default_filepath = ''

        self.image_processing_page = Image_processing_page.ImageProcessing_page(master, controller=self)
        self.hotkey_page = Hotkey_page.Hotkey_page(master)

        self.export_frame = ctk.CTkFrame(master=master)
        self.playback_frame = ctk.CTkFrame(master=master)
        self.data_frame = ctk.CTkFrame(master=master)
        self.hotkey_frame = ctk.CTkFrame(master=master)

        self.showWindow = False
        self.GaborGUI = None
        self.radius = 800
        self.FG = None
        self.curr_frame = None
        self.curr_frame_gray = None
        self.markup_frame = None
        self.horizon_line = None
        self.hor_last_midpoint = 868 / 2
        self.hor_last_slope = 0.0
        self.ImageTimeReader = ImageTimeReader()
        self.last_time_update = 0.0
        self.min_radius = 200
        self.curr_FG_pixel = (400, 400)
        self.current_var_x = 10.0
        self.current_var_y = 10.0
        self.current_var_z = 10.0
        self.print3DTruthOnce = False
        self.screenshot_impending = False
        self.fisheye_mgr = FisheyeCubemapManager()
        self.hud_marker = None

        self.ThreeDTruthPoints = None
        self.selectCalibLabel = None
        self.pnpResult = None
        self.qnpResult = None
        self.pnpDrawer = None
        # --- Pose results from YOLO detections (multi-feature) ---
        self.plotter = None

        # Checkerboard Handlers
        self._checker_residual = CkR()
        self._checker_state = CheckerboardLaunchState()
        self.btn_checkerboard = None
        self._cb_pattern = [11, 8]
        self._cb_last_ts = 0.0
        self._cb_last_found = False
        self._cb_last_corners = None
        self._cb_throttle_sec = 0.05  # 10 Hz overlay update
        self._cb_status_until = 0.0
        self.checkerboard_launcher = CheckerboardLauncher(
            state=self._checker_state,
            after=self.after,
            on_status=self._on_checker_status,
            poll_ms=300,
            module_name="support.vision.cal_board_generator",
        )

        self.gpu_monitor = None

        # Optimization for undistort
        self.map1, self.map2 = None, None

        self.imageProcessingKernelCombobox = None

        self.last_image = None

        self.available_sources = [source.value for source in ImageSource]

        self.recordButton = ctk.CTkButton(master=self.export_frame, text='Saving Imagery', fg_color='green',
                                          hover_color='navy', command=self.recordOff)
        self.printButton = ctk.CTkButton(master=self.export_frame, text='Print 3D Truth Correlation', fg_color='green',
                                         hover_color='navy', command=self.print3DTruthPointsOnce)
        self.screenshotButton = ctk.CTkButton(master=self.export_frame, text='Screenshot', fg_color='green',
                                              hover_color='navy', command=self.screenshot)

        self.exportQualityCombo = ctk.CTkComboBox(self.export_frame, values=[member.value for member in ExportQuality],
                                                  command=self.updateQuality)

        self.making_gifOrVid = False

        self.exportStartFrame = ctk.CTkLabel(self.export_frame, text=f'Start Frame: {self.camConfig.start_export_idx}')
        self.exportEndFrame = ctk.CTkLabel(self.export_frame, text=f'End Frame: {self.camConfig.end_export_idx}')

        self.btn_checkerboard = ctk.CTkButton(self.export_frame,
                                              state='normal',
                                              text='Checkerboard',
                                              command=self.launch_checkerboard)

        self.img_idx = 0
        self.timeBetweenImgsEntry = None
        self.lastImageTime = 0

        self.lastWidth = 1
        self.lastHeight = 1

        self._ui_active = True
        self._last_ui_tick = 0.0
        self._ui_throttle_sec = 0.10  # refresh UI at most every 100 ms

        self.filepath_page = Filepath_page.Filepath_page(master,
                                                         controller=self)

        self.playback_controller = PlaybackController(self)
        self.batch_controller = BatchController(self)

        self.loadFromCache()

        self.setupFrame()

        self.image_processing_page.grid_columnconfigure(0, weight=1, minsize=400)
        self.image_processing_page.grid_columnconfigure(1, weight=1)

        self.imgProcQueue_editor = GuiQueue.StepSpecQueueEditor(
            master=self.image_processing_page,
            options=self.step_options,
            on_change=self._on_queue_changed,
        )
        self.imgProcQueue_editor.grid(row=20, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        self._loading_config = False
        self.update_post_newCamConfig()

    def on_app_close(self):
        """Shut down background activity and destroy the application cleanly.

        Stops GPU monitoring, cancels scheduled callbacks where possible, closes
        plot windows, and tears down the Tk root without raising shutdown-time
        GUI exceptions.
        """
        import tkinter as tk
        root = self.winfo_toplevel()

        self.shutting_down = True

        if self.gpu_monitor is not None:
            self.gpu_monitor.stop()

        try:
            root.after_cancel("all")
        except tk.TclError:
            pass

        self._plotter_close_plot_alias()

        # Signal live worker to stop before tearing down Tk.
        self.startStreamOff()

        def _finish_close():
            try:
                root.quit()
            except tk.TclError:
                pass
            try:
                root.destroy()
            except tk.TclError:
                pass

        def _poll_worker_then_close():
            t = self._thread
            if t is not None and t.is_alive():
                try:
                    root.after(50, _poll_worker_then_close)  # type: ignore[call-arg]
                except tk.TclError:
                    pass
                return
            _finish_close()

        _poll_worker_then_close()

    def func_to_refit(self, func):
        """Register a callback used to resize or refit the parent layout.

        The callback is invoked after configuration changes that may alter the
        required GUI geometry.
        """
        self.func_that_refits = func

    def _init_flag_vars(self):
        self.config_runtime.init_flag_vars()

    def _on_flag_changed(self, name: str):
        self.config_runtime.on_flag_changed(name)

    def sync_flags_from_model(self):
        self.config_runtime.sync_flags_from_model()

    def _sync_dp_from_model(self):
        self.config_runtime.sync_dp_from_model()

    def _on_queue_changed(self, new_queue):
        self.config_runtime.on_queue_changed(new_queue)

    def _sync_queue_from_model(self):
        self.config_runtime.sync_queue_from_model()

    def _queue_to_config(self, queue):
        return self.config_runtime.queue_to_config(queue)

    def _queue_from_config(self, queue_cfg):
        return self.config_runtime.queue_from_config(queue_cfg)

    @staticmethod
    def _serialize_queue_arg(v):
        return ConfigRuntime.serialize_queue_arg(v)

    @staticmethod
    def _deserialize_queue_arg(spec, raw_val):
        return ConfigRuntime.deserialize_queue_arg(spec, raw_val)

    def loadFromCache(self) -> bool:
        return self.config_runtime.load_from_cache()

    def update_post_newCamConfig(self):
        self.config_runtime.update_post_new_config()

    def saveToCache(self,
                    immediate: bool = False,
                    delay_ms: int = 500):
        self.config_runtime.save_to_cache(immediate=immediate, delay_ms=delay_ms)

    def updateLogFile(self):
        self.config_runtime.update_log_file()

    def updateYOLOModel(self):
        self.config_runtime.update_yolo_model()

    def loadTruthPoints(self):
        self.config_runtime.load_truth_points()

    def updateQuality(self, qualityValue: str):
        self.config_runtime.update_quality(qualityValue)

    def ingestCalibration(self):
        self.config_runtime.ingest_calibration()

    def setupFrame(self):
        """Build the major secondary UI sections for export, data, and playback."""
        self.setup_exportFrame()
        self.setup_dataFrame()
        self.setup_playbackFrame()

    @staticmethod
    def grid_sideBySide(row, *args, col=0):
        """Grid multiple widgets into consecutive columns on the same row.

        Args:
            row: Grid row index.
            *args: Widgets to place.
            col: Starting column index.
        """
        for idx, item in enumerate(args):
            item.grid(row=row, column=col + idx, padx=5, pady=5, sticky='nsew')

    def setup_exportFrame(self):
        """Construct the export tools section of the GUI.

        Adds controls for screenshots, saved imagery, export cadence, quality,
        GIF/video export, export frame bounds, and checkerboard launching.
        """
        rowID = 0
        self.recordOff()
        self.grid_sideBySide(rowID, self.recordButton, self.printButton)
        rowID += 1

        self.screenshotButton.grid(row=rowID, column=0, padx=5, pady=5, sticky='ew')
        rowID += 1

        activeEntryButton = ctk.CTkButton(self.export_frame, text="Time Between Saved Frames",
                                          command=self.getEntryValue)
        self.timeBetweenImgsEntry = ctk.CTkEntry(self.export_frame,
                                                 placeholder_text=str(self.camConfig.secondsBetweenImages))
        self.grid_sideBySide(rowID, activeEntryButton, self.timeBetweenImgsEntry)
        rowID += 1

        qualityLabel = ctk.CTkLabel(self.export_frame, text="GIF Export Quality: ")
        self.grid_sideBySide(rowID, qualityLabel, self.exportQualityCombo)
        rowID += 1

        exportToGifButton = ctk.CTkButton(self.export_frame, text="Export to Gif")
        exportToVidButton = ctk.CTkButton(self.export_frame, text="Export to Vid")
        exportToGifButton.configure(
            command=lambda gif=exportToGifButton, vid=exportToVidButton: self.exportToGif(gif, vid))
        exportToVidButton.configure(
            command=lambda gif=exportToGifButton, vid=exportToVidButton: self.exportToVid(gif, vid))

        self.grid_sideBySide(rowID, exportToGifButton, exportToVidButton)
        rowID += 1

        self.grid_sideBySide(rowID, self.exportStartFrame, self.exportEndFrame)

        rowID += 1

        self.btn_checkerboard.grid(row=rowID, column=1, padx=5, pady=5, sticky='ew')

    def _on_checker_status(self, btn_state: str, btn_text: str) -> None:
        """Update the checkerboard launcher button state and text."""
        self.btn_checkerboard.configure(state=btn_state, text=btn_text)

    def _on_gpu_sample(self, sample):
        self.batch_controller.on_gpu_sample(sample)

    def _on_toggle_show_gpu(self):
        self.batch_controller.toggle_show_gpu()

    def setup_dataFrame(self):
        self.batch_controller.setup_frame()

    def setup_playbackFrame(self):
        self.playback_controller.setup_frame()

    def set_ui_active(self, active: bool):
        """Enable or disable this page's active runtime behavior.

        When deactivated, schedules the live stream to stop so hidden pages do
        not continue consuming CPU or GPU resources.
        """
        import tkinter as tk
        self._ui_active = bool(active)
        # Stop camera stream if page is hidden (don’t burn CPU/GPU off-screen)
        if not self._ui_active:
            if not self.winfo_exists():
                return
            try:
                self.after(100, self.startStreamOff)  # type: ignore[call-arg]
            except tk.TclError:
                pass

    # Optional: react to section changes if you want different behavior
    def on_section_show(self, name: str):
        """Update section-specific behavior when this page becomes visible.

        Currently records whether playback interactions should be allowed and
        requests a parent layout refit if one is registered.
        """
        # Example: only allow OpenCV windows / key polling while in Playback
        self._playback_allowed = (name == "Playback")
        if self.func_that_refits:
            self.func_that_refits()

    def _plotter_close_plot_alias(self):
        return self.batch_controller.close_plots()

    def _get_dp_conf_text(self) -> str:
        return self.batch_controller.get_conf_text()

    def _get_dp_ckpt_n(self) -> int:
        return self.batch_controller.get_ckpt_n()

    def _get_dp_prefetch_n(self) -> int:
        return self.batch_controller.get_prefetch_n()

    def _get_dp_gpu_enabled(self) -> bool:
        return self.batch_controller.get_gpu_enabled()

    def runPnP_QnP_on_folders_threaded(self):
        self.batch_controller.run_pnp_qnp_on_folders_threaded()

    def _run_kalman_batch_start(self):
        self.batch_controller.run_kalman_batch_start()

    def _dp_cancel(self):
        self.batch_controller.cancel()

    def _run_yolo_batch_start(self):
        self.batch_controller.run_yolo_batch_start()

    def run_pnp_qnp_from_detection_csv(
            self,
            csv_path: str,
            out_pnp: str | None = None,
            out_qnp: str | None = None,
            progress_cb=None,
            cancel_cb=False,
    ) -> None:
        self.batch_controller.run_pnp_qnp_from_detection_csv(
            csv_path=csv_path,
            out_pnp=out_pnp,
            out_qnp=out_qnp,
            progress_cb=progress_cb,
            cancel_cb=cancel_cb,
        )

    def launch_checkerboard(self):
        """Toggle the external checkerboard launcher process."""
        self.checkerboard_launcher.toggle()

    # --- Playback slider helpers ---
    def update_playbackMenu(self):
        self.playback_controller.update_playback_menu()

    def populate_idsTimes(self, directory):
        self.playback_controller.populate_ids_times(directory)

    def _window_is_open(self) -> bool:
        return self.playback_controller.window_is_open()

    def reset_runtime_state(self, *, reset_fg: bool = True) -> None:
        """Reset per-run tracking state before starting a new playback/stream/export pass."""
        self.last_image = None
        self.curr_frame_gray = None

        if reset_fg:
            self.FG = None
            self.last_fg_output = None
            self.last_time_update = 0.0
            self.curr_FG_pixel = (400, 400)

            # Optional, but sensible if hyper-focus / FG-adjacent state has been sticky
            self.curr_r_V_d = None
            self.curr_r_T_d = None

    def run_folder_reader(self):
        self.playback_controller.run_folder_reader()

    def startStreamToggle(self):
        if self._thread is None or not self._thread.is_alive():  # thread not running
            self.startStreamOn()
            self.stream_running_var.set(True)
            return True

        self.startStreamOffBool()
        self.stream_running_var.set(False)
        return False

    def startStreamOn(self):
        self.reset_runtime_state(reset_fg=True)
        self.showWindow = True
        self.threadStopper = utils.ThreadStopper()
        self.stream_running_var.set(True)
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def startStreamOffBool(self):
        self.showWindow = False
        self.stream_running_var.set(False)
        try:
            self.threadStopper.set()
        except AttributeError:
            pass

    def startStreamOff(self):
        # UI thread only signals stop. The worker owns stream/window teardown.
        try:
            self.threadStopper.set()
        except AttributeError:
            pass

        self.showWindow = False
        self.stream_running_var.set(False)

        # OpenCV backend can still be nudged here safely.
        if self.vc is not None and self.vc.isOpened():
            try:
                self.vc.release()
            except cv2.error:
                pass
            self.vc = None

    def run_detectSingleImage(self):
        self.stream_runner.run_detect_single_image()

    def run(self):
        self.stream_runner.run()

    def queue_live_vimba_update(self, settings: dict[str, Any]) -> bool:
        """
        Thread-safe: UI calls this to request a live camera change.
        Returns True if the request was queued for a running Vimba stream.
        """
        return self.stream_runner.queue_live_vimba_update(settings)

    def run_vimba_stream(self):
        self.stream_runner.run_vimba_stream()

    def run_video_stream(self):
        self.stream_runner.run_video_stream()

    def on_worker_exit(self):
        self._thread = None
        self.showWindow = False
        self.stream_running_var.set(False)

    def getEntryValue(self):
        """Parse and validate the export frame-spacing entry from the UI."""
        try:
            self.camConfig.secondsBetweenImages = float(self.timeBetweenImgsEntry.get())
        except ValueError:
            self.camConfig.secondsBetweenImages = 1.0
            self.timeBetweenImgsEntry.delete(0, ctk.END)
            self.timeBetweenImgsEntry.configure(placeholder_text='1')
        if self.camConfig.secondsBetweenImages <= 0.0:
            self.timeBetweenImgsEntry.delete(0, ctk.END)
            self.timeBetweenImgsEntry.configure(placeholder_text='1')
            self.camConfig.secondsBetweenImages = 1.0

    def _gather_annotated_frames(self) -> list[NDArray]:
        """Render the selected export frame range into annotated image arrays.

        Loads each source frame, applies the configured analysis pipeline, and
        returns a list of processed images suitable for GIF or video export.
        """
        self.reset_runtime_state(reset_fg=True)
        directory = Path(self.camConfig.imageFilepath).parent
        self.populate_idsTimes(str(directory))

        paths = []
        for rec in self.ImageTimeReader.idsTimes:
            p = Path(rec[0])
            paths.append(p if p.is_absolute() else (directory / p))

        try:
            import pandas as pd
            offset_dict = pd.read_csv(directory / '__TIME_OFFSET.csv')
            loaded_offset = float(offset_dict['offset'][0])

            if self.hud_marker is not None:
                self.hud_marker.update_offset(loaded_offset)

            # UI delta should always start at zero after loading persisted offset.
            self.camConfig.cam_to_log_time_offset = 0.0

        except FileNotFoundError:
            if self.hud_marker is not None:
                self.hud_marker.update_offset(0.0)
            self.camConfig.cam_to_log_time_offset = 0.0

        cv_imgs = []
        start = self.camConfig.start_export_idx
        end = self.camConfig.end_export_idx + 1
        for idx, img_path in zip(range(start, end), paths[start:end]):
            frame = cv2.imread(str(img_path))
            ts = self.ImageTimeReader.idsTimes[idx][1]
            cv_img = self.analyze_image(
                frame,
                img_time=(ts + self.camConfig.cam_to_log_time_offset if ts is not None else None),
                name=self.ImageTimeReader.idsTimes[idx][0],
                display_in_realtime=False
            )
            if cv_img is not None:
                cv_imgs.append(cv_img)

        return cv_imgs

    def exportToGif(self, exportToGifButton, exportToVidButton):
        """Begin asynchronous GIF/APNG export for the current frame range."""
        if self.making_gifOrVid:
            return

        exportToGifButton.configure(text="Making gif...", state='disabled', fg_color=clr.CTK_BLUE)
        exportToVidButton.configure(text="Making gif...", state='disabled', fg_color=clr.CTK_BLUE)
        self.making_gifOrVid = True

        t = threading.Thread(target=self.exportToGif_worker,
                             daemon=True,
                             args=(exportToGifButton, exportToVidButton))
        t.start()

    def exportToVid(self, exportToGifButton, exportToVidButton):
        """Begin asynchronous video export for the current frame range."""
        if self.making_gifOrVid:
            return

        exportToGifButton.configure(text="Making vid...", state='disabled', fg_color=clr.CTK_BLUE)
        exportToVidButton.configure(text="Making vid...", state='disabled', fg_color=clr.CTK_BLUE)
        self.making_gifOrVid = True

        t = threading.Thread(target=self.exportToVid_worker,
                             daemon=True,
                             args=(exportToGifButton, exportToVidButton))
        t.start()

    def exportToGif_worker(self,
                           exportToGifButton, exportToVidButton):
        try:
            frames = self._gather_annotated_frames()
            # from support.io.convert_to_gif import make_gif
            # make_gif(frames, 10, infinite=True, quality=self.camConfig.export_quality)
            from support.io.convert_to_gif import make_apng
            make_apng(frames, 60, infinite=True, quality=self.camConfig.export_quality)
        finally:
            self.after(0, self._exportToGifOrVid_done,
                       exportToGifButton, exportToVidButton)

    def exportToVid_worker(self,
                           exportToGifButton, exportToVidButton):
        try:
            frames = self._gather_annotated_frames()
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
            out = cv2.VideoWriter('output_video.mp4', fourcc, 10, (w, h))
            for f in frames:
                out.write(f)
            out.release()
        finally:
            self.after(0, self._exportToGifOrVid_done,
                       exportToGifButton, exportToVidButton)

    def _exportToGifOrVid_done(self,
                               exportToGifButton, exportToVidButton):
        exportToGifButton.configure(text="Export to GIF", state='normal', fg_color=clr.CTK_BUTTON_GREEN)
        exportToVidButton.configure(text="Export to Vid", state='normal', fg_color=clr.CTK_BUTTON_GREEN)
        self.making_gifOrVid = False

    def screenshot(self):
        self.screenshot_impending = True
        self.screenshotButton.configure(fg_color=clr.CTK_BLACK)
        self.after(500, lambda: self.screenshotButton.configure(fg_color=clr.CTK_GREEN))  # type: ignore[call-arg]

    def recordOn(self):
        self.recordButton.configure(fg_color=clr.CTK_GREEN, text='Saving Imagery', hover_color=clr.CTK_NAVY,
                                    command=self.recordOff)
        self.recording = True

    def recordOff(self):
        self.recordButton.configure(fg_color=clr.CTK_BUTTON_RED, text=f'Saved Imagery: #{self.img_idx}',
                                    hover_color=clr.CTK_BLUE,
                                    command=self.recordOn)
        self.recording = False

    def print3DTruthPointsOnce(self):
        self.print3DTruthOnce = True

    def analyze_image(self,
                      frame,
                      img_time=None,
                      name=None,
                      display_in_realtime=True,
                      box_around=False) -> NDArray | None:
        return self.frame_processor.analyze_image(
            frame,
            img_time=img_time,
            name=name,
            display_in_realtime=display_in_realtime,
            box_around=box_around,
        )

    def cleanup(self, markupFrame, name=None):
        self.frame_processor.cleanup(markupFrame, name=name)

    @staticmethod
    def parse_args(args: dict, obj, *, ignore_unknown=True):
        """
        keymap maps incoming keys -> dataclass field names.
        Mutates obj in-place; returns obj.
        """
        from dataclasses import is_dataclass, fields
        if not is_dataclass(obj) or isinstance(obj, type):
            raise TypeError("Expected a dataclass instance")

        valid_fields = {f.name for f in fields(obj)}

        for in_key, value in args.items():
            if in_key not in obj.KEYMAP:
                if not ignore_unknown:
                    raise KeyError(f"Unknown incoming key: {in_key!r}")
                continue

            field_name = obj.KEYMAP[in_key]
            if field_name not in valid_fields:
                raise KeyError(f"keymap maps {in_key!r} -> {field_name!r}, but that field doesn't exist")

            setattr(obj, field_name, value)

        return obj

    def createDetector(self):
        if self.detector is None:
            self.arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
            self.arucoParams = cv2.aruco.DetectorParameters()
            self.arucoParams.adaptiveThreshWinSizeMin = 5
            self.arucoParams.adaptiveThreshWinSizeMax = 35
            self.arucoParams.adaptiveThreshWinSizeStep = 5
            self.arucoParams.minMarkerPerimeterRate = 0.02  # or higher if tags are big
            self.arucoParams.maxMarkerPerimeterRate = 1.0
            self.arucoParams.cornerRefinementMinAccuracy = 0.1  # or 0.2
            self.arucoParams.cornerRefinementMaxIterations = 20
        self.detector = cv2.aruco.ArucoDetector(self.arucoDict, self.arucoParams)

    def draw_playbackStats(self, frame,
                           markupFrame,
                           ctx: GuiQueue.FrameCtx,
                           args):
        self.playback_controller.draw_playback_stats(frame, markupFrame, ctx, args)

    @staticmethod
    def draw_time(frame, markupFrame, ctx: GuiQueue.FrameCtx, args):
        FrameProcessor.draw_time(frame, markupFrame, ctx, args)

    @staticmethod
    def draw_name(frame,
                  markupFrame,
                  ctx: GuiQueue.FrameCtx,
                  args) -> None:
        FrameProcessor.draw_name(frame, markupFrame, ctx, args)

    def draw_HUD(self, frame: NDArray,
                 markupFrame: NDArray,
                 ctx: GuiQueue.FrameCtx,
                 args) -> None:
        if ctx.img_time is None:
            return

        opts: GuiQueue.HudOpts = self.parse_args(args, GuiQueue.HudOpts())

        if self.calibration.validCal:
            cx_cy = (int(self.calibration.cx), int(self.calibration.cy))
        else:
            cx_cy = (int(markupFrame.shape[0] / 2), int(markupFrame.shape[1] / 2))

        from support.viz.HUD_draw import HUD_Marker
        if self.hud_marker is None:
            self.hud_marker = HUD_Marker(self.camConfig.hud_data_filepath)

        scale = ctx.resize.get_or(1.0)

        attitude = self.hud_marker.draw_HUD(image=markupFrame,
                                            img_time=ctx.img_time,
                                            opts=opts,
                                            cx_cy_ori=cx_cy,
                                            scale=scale)
        if opts.store_attitude:
            self.own_attitude = attitude

    @staticmethod
    def draw_boxAround(frame,
                       markupFrame,
                       ctx: GuiQueue.FrameCtx,
                       args) -> None:
        FrameProcessor.draw_box_around(frame, markupFrame, ctx, args)

    def draw_chessboard(self, frame: NDArray,
                        markupFrame: NDArray,
                        ctx: GuiQueue.FrameCtx,
                        args) -> None:
        if self.curr_frame_gray is None:
            self.curr_frame_gray = cv2.cvtColor(markupFrame, cv2.COLOR_BGR2GRAY)

        self._checker_residual.draw_chessboard(markupFrame, self.curr_frame_gray, self._cb_pattern)

    def _draw_chessboard_state(self, frame):
        height, width, _ = frame.shape
        org1 = (int(width * 0.1), int(height * 0.20))
        org2 = (int(width * 0.1), int(height * 0.25))

        instr_text_a = f'{self._cb_pattern[0]} inner row corners'
        instr_text_b = f'{self._cb_pattern[1]} inner col corners'

        # Draw on A
        cv2.putText(frame, instr_text_a, org1,
                    cv2.FONT_HERSHEY_SIMPLEX, med_text(width), (0, 0, 0), 4)
        cv2.putText(frame, instr_text_a, org1,
                    cv2.FONT_HERSHEY_SIMPLEX, med_text(width), (255, 255, 0), 1)
        cv2.putText(frame, instr_text_b, org2,
                    cv2.FONT_HERSHEY_SIMPLEX, med_text(width), (0, 0, 0), 4)
        cv2.putText(frame, instr_text_b, org2,
                    cv2.FONT_HERSHEY_SIMPLEX, med_text(width), (255, 255, 0), 1)

    def _handle_chessboard_hotkeys(self, key: int):

        # mimic CalBoardGenerator hotkeys: 4/6 adjust cols, 8/2 adjust rows
        changed = False

        if key == ord('4'):  # fewer columns
            if self._cb_pattern[0] > 3:
                self._cb_pattern[0] -= 1
                changed = True

        elif key == ord('6'):  # more columns
            self._cb_pattern[0] += 1
            changed = True

        elif key == ord('8'):  # more rows
            self._cb_pattern[1] += 1
            changed = True

        elif key == ord('2'):  # fewer rows
            if self._cb_pattern[1] > 3:
                self._cb_pattern[1] -= 1
                changed = True

        elif key == ord('r'):  # optional: reset to default
            self._cb_pattern[:] = [11, 8]
            changed = True

        if changed:
            # force an immediate re-detect instead of waiting for throttle
            self._cb_last_ts = 0.0
            # clear cached result so you don't draw stale corners
            self._cb_last_found = False
            self._cb_last_corners = None
            return time.monotonic() + 2

        return None

    @staticmethod
    def plotOnImg(markupFrame, points, names, color):
        for idx, pxPt in enumerate(points):
            offset = int(markupFrame.shape[0] * 0.02)
            cv2.circle(markupFrame, (int(pxPt[0]), int(pxPt[1])), 5, color, 5)
            textLoc = (int(pxPt[0]) - offset, int(pxPt[1] - offset))
            cv2.putText(markupFrame, str(names[idx]), textLoc, cv2.FONT_HERSHEY_SIMPLEX,
                        small_text(markupFrame.shape[0]), (0, 0, 0),
                        4)
            cv2.putText(markupFrame, str(names[idx]), textLoc, cv2.FONT_HERSHEY_SIMPLEX,
                        small_text(markupFrame.shape[0]), color, 2)

    def potentialResize(self, markupFrame):
        self.frame_processor.potential_resize(markupFrame)

    def print_pnp_results(self):
        np.set_printoptions(precision=5, threshold=sys.maxsize, suppress=True)

        if self.ThreeDTruthPoints is None:
            self.loadTruthPoints()

        points = None
        if self.centers is not None and len(self.centers) >= 6:
            truthPoints = copy.copy(self.ThreeDTruthPoints.truthPoints)
            points = []

            removeIDs = []
            for idx, detectID in enumerate(self.detectIDS):
                try:
                    points.append(truthPoints[str(detectID[0])])
                except KeyError:
                    removeIDs.append(idx)

            centers = self.centers.copy()
            for idx in reversed(removeIDs):
                centers = np.delete(centers, idx, axis=0)
            points = np.array(points)

        # probe_pose = np.array([4.89965725, .20014286, -1.55304432])

        self.print3DTruthOnce = False

        if points is None or self.detector is None or self.centers is None:
            return
        if self.qnpResult is None and self.pnpResult is None:
            return

        curr_level = LOG.level
        try:
            import logging
            LOG.setLevel(logging.INFO)
            b1_lne = '\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
            b2_lne = b1_lne + b1_lne + '\n'
            LOG.info(b2_lne)
            LOG.info(f'\nThreeD Correlation Output Requested: Current Time: {time.asctime()}')
            if points is not None and self.detector is not None:
                LOG.info(f'\nObj Points: \n{points}')
            if self.centers is not None:
                LOG.info(f'\nImg Points: \n{self.centers}')
            LOG.info(f'\nCam Matrix: \n{self.calibration.getCameraMatrix()}')
            if self.pnpResult is not None:
                LOG.info(f'\nPnP Result: \ncam_R_tgt:\n{self.pnpResult[0].to_dcm()}\ncam_t_tgt:\n{self.pnpResult[1]}')
            if self.qnpResult is not None:
                LOG.info(f'\nQnP Result: \ncam_R_tgt:\n{self.qnpResult[0].to_dcm()}\ncam_t_tgt:\n{self.qnpResult[1]}')

            LOG.info(b1_lne)
        finally:
            LOG.setLevel(curr_level)

    def undistort(self,
                  frame: NDArray,
                  markupFrame: NDArray,
                  ctx: GuiQueue.FrameCtx,
                  args):

        if not self.calibration.validCal:
            raise ValueError("No calibration loaded!")

        def _mark_undistorted() -> None:
            if not ctx.undistorted.is_set():
                ctx.undistorted.set(True)  # One undistort has been run
            else:
                ctx.undistorted.set(False)  # Multiple undistorts, invalid

        def _paste_into(frame_dst: np.ndarray, img_src: np.ndarray) -> None:
            """Paste src into dst. If same shape, full copy.
            If src is smaller, center it with black padding.
            Otherwise, resize src to dst size.
            """
            h, w = frame_dst.shape[:2]
            hs, ws = img_src.shape[:2]

            if (h, w) == (hs, ws):
                frame_dst[:] = img_src
                return

            # If src fits inside dst, center-blit with padding (no distortion)
            if hs <= h and ws <= w:
                frame_dst[:] = 0
                y0 = (h - hs) // 2
                x0 = (w - ws) // 2
                frame_dst[y0:y0 + hs, x0:x0 + ws] = img_src
                return

            # Otherwise resize to fit (may distort)
            resized = cv2.resize(img_src, (w, h), interpolation=cv2.INTER_LINEAR)
            frame_dst[:] = resized

        opts: GuiQueue.UndistortOpts = self.parse_args(args, GuiQueue.UndistortOpts())

        if self.calibration.fisheye:
            if opts.cubemap:
                stitched = self.fisheye_mgr.render_cubemap(
                    frame=markupFrame,
                    calibration=self.calibration,
                    face_size=DEFAULT_CUBEMAP_FACESIZE,
                    layout=DEFAULT_CUBEMAP_LAYOUT,
                    faces=DEFAULT_CUBEMAP_FACES,
                    cells=3,
                )
                _paste_into(markupFrame, stitched)
            else:
                face_size = min(markupFrame.shape[:2])
                front = self.fisheye_mgr.render_front_face(
                    frame=markupFrame,
                    calibration=self.calibration,
                    face_size=face_size,
                )
                _paste_into(markupFrame, front)

            _mark_undistorted()
            return

        if self.map1 is None or self.map2 is None:
            raise ValueError("No calibration loaded!")

        tmp = cv2.remap(
            markupFrame,
            self.map1,
            self.map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

        self._replace_array_in_place(markupFrame, tmp)
        _mark_undistorted()

    @staticmethod
    def _replace_array_in_place(dst: np.ndarray, src: np.ndarray) -> None:
        if dst.dtype != src.dtype:
            raise TypeError(
                f"Cannot replace ndarray in-place when dtype changes: "
                f"{dst.dtype} -> {src.dtype}"
            )

        if not dst.flags.c_contiguous:
            raise ValueError("markupFrame must be C-contiguous for in-place resize")

        if not dst.flags.owndata:
            raise ValueError("markupFrame must own its data for in-place resize")

        if dst.shape != src.shape:
            dst.resize(src.shape, refcheck=False)

        dst[...] = src

    def resize_image(self,
                     frame: NDArray,
                     markupFrame: NDArray,
                     ctx: GuiQueue.FrameCtx,
                     args):

        opts: GuiQueue.ResizeOpts = self.parse_args(args, GuiQueue.ResizeOpts())
        scale = (10.0 ** opts.scale) / 10.0
        h, w, _ = markupFrame.shape
        newShape = (scale * np.array([h, w])).astype(np.int32)
        if newShape[0] > opts.pixelNum:
            newShape[0] = opts.pixelNum
        if newShape[1] > opts.pixelNum:
            newShape[1] = opts.pixelNum

        tmp = cv2.resize(markupFrame, newShape, interpolation=cv2.INTER_LINEAR)

        self._replace_array_in_place(markupFrame, tmp)

        ctx.resize.set(scale)

    @staticmethod
    def image_filter_arg_specs(args):
        filt = args.get("Filter", ImageKernel.Unfiltered)

        specs = [
            GuiQueue.ArgSpec("Filter", ImageKernel, ImageKernel.Unfiltered),
        ]

        if filt == ImageKernel.Gain:
            specs.append(GuiQueue.ArgSpec("Gain", float, 1.0, 0.0, 3.0))

        if filt == ImageKernel.Brightness:
            specs.append(GuiQueue.ArgSpec("Brightness", float, 0.0, -100.0, 100.0))

        return tuple(specs)

    def applyKernel(self,
                    frame: NDArray,
                    markupFrame: NDArray,
                    ctx: GuiQueue.FrameCtx,
                    args) -> None:
        if 'Filter' not in args:
            return

        processKernel = args['Filter']
        gain = float(args.get('Gain', 1.0))
        brightness = int(args.get('Brightness', 0))

        from support.vision.filter_image import apply_filter
        self.GaborGUI = apply_filter(
            markupFrame,
            processKernel,
            self.GaborGUI,
            gain,
            brightness,
        )

    def detect_corners(self,
                       frame: NDArray,
                       markupFrame: NDArray,
                       ctx: GuiQueue.FrameCtx,
                       args) -> None:
        if self.curr_frame_gray is None:
            self.curr_frame_gray = cv2.cvtColor(markupFrame, cv2.COLOR_BGR2GRAY)
        harris_corners = cv2.cornerHarris(self.curr_frame_gray, 3, 3, 0.05)

        markupFrame[harris_corners > 0.025 * harris_corners.max()] = [0, 255, 255]
        return

    def detectAprilTags(self,
                        frame: NDArray,
                        markupFrame: NDArray,
                        ctx: GuiQueue.FrameCtx,
                        args) -> None:
        self.pose_runtime.detect_april_tags(frame, markupFrame, ctx, args)

    def pnp3DTruthPoints(self,
                         frame: NDArray,
                         markupFrame: NDArray,
                         ctx: GuiQueue.FrameCtx,
                         args) -> None:
        self.pose_runtime.pnp_3d_truth_points(frame, markupFrame, ctx, args)

    def qnp3DTruthPoints(self,
                         frame: NDArray,
                         markupFrame: NDArray,
                         ctx: GuiQueue.FrameCtx,
                         args) -> None:
        self.pose_runtime.qnp_3d_truth_points(frame, markupFrame, ctx, args)

    def detectHorizon(self, frame: NDArray,
                      markupFrame: NDArray,
                      ctx: GuiQueue.FrameCtx,
                      args) -> None:

        if self.curr_frame_gray is None:
            self.curr_frame_gray = cv2.cvtColor(markupFrame, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(self.curr_frame_gray, 100, 200, apertureSize=3)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, 50,
                                minLineLength=np.sum(markupFrame.shape) / 10.0,
                                maxLineGap=20)

        color = clr.RED

        # Find the most horizontal line
        if lines is not None:
            line_lengths = []
            for line in lines:
                for x1, y1, x2, y2 in line:
                    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    line_lengths.append((length, (x1, y1, x2, y2)))

            if line_lengths:
                longest_line = max(line_lengths, key=lambda item: item[0])
                x1, y1, x2, y2 = longest_line[1]
                if np.abs(x2 - x1) > 0.000001:
                    m = (y2 - y1) / (x2 - x1)
                    y1 = y1 - m * x1
                    x2 = markupFrame.shape[1]
                    self.hor_last_midpoint = ((y1 + m * x2 / 2.0) + self.hor_last_midpoint) / 2.0
                    self.hor_last_slope = (m + self.hor_last_slope) / 2.0
                    color = clr.YELLOWGREEN

        x1 = 0
        x2 = int(markupFrame.shape[1])
        y1 = int(self.hor_last_midpoint - self.hor_last_slope * x2 / 2.0)
        y2 = int(self.hor_last_midpoint + self.hor_last_slope * x2 / 2.0)

        cv2.line(markupFrame, (x1, y1), (x2, y2), color, 2)

        self.horizon_line = (x1, y1, x2, y2)
        return

    def check_above_horizon(self, pt):
        return self.pose_runtime.check_above_horizon(pt)

    def hyper_focus(self,
                    markupFrame: NDArray,
                    ctx: GuiQueue.FrameCtx) -> None:
        self.pose_runtime.hyper_focus(markupFrame, ctx)

    @staticmethod
    def _apply_hyper_focus_plan(markupFrame: NDArray, plan) -> None:
        PoseRuntime.apply_hyper_focus_plan(markupFrame, plan)

    def run_yolo(self,
                 frame: NDArray,
                 markupFrame: NDArray,
                 ctx: GuiQueue.FrameCtx,
                 args) -> None:
        """Run YOLO using either the original frame or the current markup frame."""
        self.pose_runtime.run_yolo(frame, markupFrame, ctx, args)

    @staticmethod
    def _draw_factor_graph_overlay(markupFrame: NDArray,
                                   fg_output,
                                   color) -> None:
        PoseRuntime.draw_factor_graph_overlay(markupFrame, fg_output, color)

    def factor_graph(self,
                     frame: NDArray,
                     markupFrame: NDArray,
                     ctx: GuiQueue.FrameCtx,
                     hyper_focus: bool) -> None:
        self.pose_runtime.factor_graph(frame, markupFrame, ctx, hyper_focus)

    def phase_correlation(self,
                          frame: NDArray,
                          markupFrame: NDArray,
                          ctx: GuiQueue.FrameCtx,
                          args) -> None:

        if self.calibration.validCal:
            cx = int(self.calibration.cx)
            cy = int(self.calibration.cy)
        else:
            cx = int(self.curr_frame.shape[0] / 2)
            cy = int(self.curr_frame.shape[1] / 2)

        if self.curr_frame_gray is None:
            self.curr_frame_gray = cv2.cvtColor(markupFrame, cv2.COLOR_BGR2GRAY)

        if self.last_image is not None and self.last_image.shape == self.curr_frame_gray.shape:
            lft_rt, ret = cv2.phaseCorrelate(self.curr_frame_gray.astype(np.float64) / 255.0,
                                             self.last_image.astype(np.float64) / 255.0)
            lft, rt = lft_rt
            cv2.arrowedLine(markupFrame,
                            (cx, cy),
                            (int(cx + 10 * lft),
                             int(cy + 10 * rt)),
                            clr.RED, 3)

        self.last_image = copy.deepcopy(self.curr_frame_gray)

    @staticmethod
    def _cv_pose_to_ours(R_cv: np.ndarray, t_cv: np.ndarray):
        return PoseRuntime.cv_pose_to_ours(R_cv, t_cv)

    def run_folder_reader_profiled(self):
        from support.io.profiler import make_profile, print_stats

        prof = make_profile()
        try:
            prof.enable()
            self.run_folder_reader()
        finally:
            print_stats(prof)
