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

from support.mathHelpers.twoD_to_threeD import solveQnP
from support.mathHelpers.quaternions import Quaternion as q, mat2quat

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
from support.vision.draw_circle_and_mask import dim_except_circle
from support.vision.vimba_controller import VimbaController as VimbaCam, HAVE_VMBPY
from support.vision.fisheye_to_cubemap import (
    DEFAULT_CUBEMAP_FACES,
    DEFAULT_CUBEMAP_LAYOUT,
    DEFAULT_CUBEMAP_FACESIZE,
    FisheyeCubemapManager)

from support.runtime.fg_singleTarget import (
    build_factor_graph_output,
    factor_graph_projection_matrix,
    run_factor_graph_step,
)
from support.runtime.fg_singleTarget import build_hyper_focus_plan
from support.runtime.PlaybackController import PlaybackController

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
                    root.after(50, _poll_worker_then_close)
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
        """Create Tk variable wrappers for config-backed UI flags.

        Initializes tracked variables from the config model and attaches change
        callbacks so user edits immediately propagate back into the config.
        """
        double_vars = ['yolo_conf', 'yolo_iou']
        for name in self._flags:
            if not hasattr(self.camConfig, name):
                setattr(self.camConfig, name, 1.0 if name in double_vars else False)

            if name in double_vars:
                v = ctk.DoubleVar(value=getattr(self.camConfig, name, 1.0))
            else:
                v = ctk.BooleanVar(value=bool(getattr(self.camConfig, name, False)))

            # when UI flips, write to model
            v.trace_add("write", lambda var_name, index, op, n=name: self._on_flag_changed(n))
            self._flag_vars[name] = v

    def _on_flag_changed(self, name: str):
        """Handle a write to a tracked UI flag variable.

        Normalizes the value to the correct scalar type, stores it on the
        camera config, and triggers a cache save.
        """
        # DoubleVars must stay float; everything else is bool
        if name in ("yolo_conf", "yolo_iou"):
            val = float(self._flag_vars[name].get())
        else:
            val = bool(self._flag_vars[name].get())

        setattr(self.camConfig, name, val)
        self.saveToCache()

    def sync_flags_from_model(self):
        """Push config-backed flag values into their Tk variable mirrors."""
        for n in self._flags:
            if n in ("yolo_conf", "yolo_iou"):
                self._flag_vars[n].set(float(getattr(self.camConfig, n, 1.0)))
            else:
                self._flag_vars[n].set(bool(getattr(self.camConfig, n, False)))

    def _sync_dp_from_model(self):
        self.batch_controller.sync_from_model()

    def _on_queue_changed(self, new_queue):
        """Accept a queue edit from the GUI editor and persist it to the model.

        Normalizes queue arguments via deep copy, updates the active processing
        pipeline, writes the serialized queue into the camera config, and saves
        the new state to cache when the signature changes.
        """
        if getattr(self, "_loading_config", False):
            return

        normalized_queue = [
            (fn, copy.deepcopy(args))
            for fn, args in new_queue
        ]

        sig_did_change = self.list_of_image_process_functors != normalized_queue
        self.list_of_image_process_functors = normalized_queue

        if sig_did_change:
            self.camConfig.image_processing_queue = self._queue_to_config(normalized_queue)
            self.saveToCache()

            if self.func_that_refits is not None:
                self.func_that_refits()

    def _sync_queue_from_model(self):
        """Rebuild the runtime queue and queue editor from cached config data.

        Deserializes the saved queue specification into callable steps plus
        arguments, updates the in-memory processing list, and refreshes the
        queue editor without re-emitting change events.
        """
        cfg = getattr(self.camConfig, "image_processing_queue", [])
        rebuilt = self._queue_from_config(cfg)

        self.list_of_image_process_functors = [
            (fn, copy.deepcopy(args)) for fn, args in rebuilt
        ]

        if hasattr(self, "imgProcQueue_editor") and self.imgProcQueue_editor is not None:
            self.imgProcQueue_editor.set_queue(
                [(fn, copy.deepcopy(args)) for fn, args in rebuilt],
                emit_change=False,
            )

    def _queue_to_config(self, queue):
        """Serialize a runtime processing queue into cache-friendly config data.

        Converts each processing function to its user-facing label and serializes
        its argument values into a plain dictionary representation.
        """
        out = []

        for fn, args in queue:
            try:
                label = self.fn_to_label[fn]
            except KeyError:
                raise RuntimeError(f"Queue contains unknown processing function: {fn}")

            clean_args = {
                k: self._serialize_queue_arg(v)
                for k, v in args.items()
            }

            out.append({
                "label": label,
                "args": clean_args
            })

        return out

    def _queue_from_config(self, queue_cfg):
        """Deserialize a saved queue specification into callable pipeline steps.

        Unknown step labels are skipped with a warning. Missing arguments are
        filled from defaults, and cached values are coerced back into their
        expected runtime types where possible.
        """
        if not queue_cfg:
            return []

        option_by_label = {opt.label: opt for opt in self.step_options}
        rebuilt = []

        for row in queue_cfg:
            label = row["label"]

            if label not in option_by_label:
                LOG.warning("Skipping cached queue step '%s' (unknown)", label)
                continue

            opt = option_by_label[label]
            raw_args = copy.deepcopy(row.get("args", {}))

            # Start from defaults so missing fields are filled in automatically
            parsed_args = opt.default_args.copy()

            spec_by_name = {spec.name: spec for spec in opt.get_arg_specs(parsed_args)}
            for arg_name, raw_val in raw_args.items():
                spec = spec_by_name.get(arg_name)
                if spec is None:
                    parsed_args[arg_name] = raw_val
                    continue

                parsed_args[arg_name] = self._deserialize_queue_arg(spec, raw_val)

            # Recompute once more after deserialization in case one arg changes which specs exist
            spec_by_name = {spec.name: spec for spec in opt.get_arg_specs(parsed_args)}
            for spec in spec_by_name.values():
                parsed_args.setdefault(spec.name, spec.default)

            rebuilt.append((opt.fn, parsed_args))

        return rebuilt

    @staticmethod
    def _serialize_queue_arg(v):
        """Convert a queue argument into a cache-safe scalar representation.

        Enum values are stored by value; all other types are passed through
        unchanged.
        """
        if isinstance(v, enum.Enum):
            return v.value
        return v

    @staticmethod
    def _deserialize_queue_arg(spec, raw_val):
        """Reconstruct a typed queue argument from cached data.

        Uses the argument spec's default value to infer the desired runtime type
        and falls back to that default if conversion fails.
        """
        default = spec.default

        # Enum args: rebuild from saved scalar/string value
        if isinstance(default, enum.Enum):
            enum_type = type(default)
            try:
                return enum_type(raw_val)
            except Exception:
                LOG.warning(
                    "Failed to parse enum arg '%s' from cached value %r; using default %r",
                    spec.name, raw_val, default
                )
                return default

        # Optional: coerce basic scalar types back to the default's type
        try:
            if isinstance(default, bool):
                return bool(raw_val)
            if isinstance(default, int) and not isinstance(default, bool):
                return int(raw_val)
            if isinstance(default, float):
                return float(raw_val)
            if isinstance(default, str):
                return str(raw_val)
        except Exception:
            LOG.warning(
                "Failed to parse arg '%s' from cached value %r; using default %r",
                spec.name, raw_val, default
            )
            return default

        return raw_val

    def loadFromCache(self) -> bool:
        """Load cached configuration into the active camera config object.

        Returns:
            True if a YAML-backed config was restored, else False.
        """
        self._loading_config = True
        res = self.config_store.load_from_cache(self.camConfig)
        if not res.loaded_yaml:
            self._loading_config = False
            return False
        return True

    def update_post_newCamConfig(self):
        """Refresh runtime systems after loading or replacing the camera config.

        Synchronizes UI variables, calibration, YOLO state, optional truth data,
        batch-processing controls, queue editor state, and any layout refit
        callback that depends on the new configuration.
        """
        iou = copy.deepcopy(self.camConfig.yolo_iou)
        self._flag_vars["yolo_conf"].set(float(self.camConfig.yolo_conf))
        self._flag_vars["yolo_iou"].set(float(iou))

        self.updateLogFile()
        self.ingestCalibration()
        self.updateYOLOModel()
        if self.ThreeDTruthPoints is not None:
            self.loadTruthPoints()

        self.sync_flags_from_model()
        self._sync_dp_from_model()
        self._sync_queue_from_model()

        try:
            if hasattr(self, "gpu_slider"):
                self.gpu_slider.configure(
                    state="normal" if bool(getattr(self.camConfig, "dp_gpu", False)) else "disabled"
                )
            if not bool(getattr(self.camConfig, "dp_gpu", False)):
                self.gpu_slider.set(0.0)
        except Exception:
            pass

        try:
            if self.exportStartFrame is not None:
                self.exportStartFrame.configure(text=f"Start Frame: {self.camConfig.start_export_idx}")
            if self.exportEndFrame is not None:
                self.exportEndFrame.configure(text=f"End Frame: {self.camConfig.end_export_idx}")
        except Exception:
            pass

        try:
            if hasattr(self, "filepath_page") and self.filepath_page is not None:
                self.filepath_page.sync_labels()
        except Exception:
            pass

        try:
            if hasattr(self, "playback_controller") and self.playback_controller is not None:
                self.playback_controller.update_playback_menu()
        except Exception:
            pass

        try:
            if hasattr(self, "exportQualityCombo") and self.exportQualityCombo is not None:
                self.exportQualityCombo.set(self.camConfig.export_quality.value)
        except Exception:
            pass

        self.saveToCache()

        if self.func_that_refits is not None:
            self.func_that_refits()

    def saveToCache(self,
                    immediate: bool = False,
                    delay_ms: int = 500):
        """Persist the current camera configuration to cache.

        Copies selected live UI values back into the config model before
        delegating to the config store, with optional debounced saving.
        """
        if getattr(self, "_loading_config", False):
            return

        self.camConfig.yolo_conf = float(self._flag_vars["yolo_conf"].get())
        self.camConfig.yolo_iou = float(self._flag_vars["yolo_iou"].get())

        self.config_store.save_to_cache(self.camConfig, immediate=immediate, delay_ms=delay_ms)

    def updateLogFile(self):
        """Reload HUD attitude/log data from the configured source path."""
        if self.hud_marker is not None:
            self.hud_marker.read_attitude_files(self.camConfig.hud_data_filepath)
            self.load_offset_csv(self.camConfig.hud_data_filepath)
        self.saveToCache()

    def updateYOLOModel(self):
        """Point the active YOLO session at the configured model directory."""
        if self.camConfig.yoloFilepath and self.yoloSession is not None:
            self.yoloSession.setNewFolder(self.camConfig.yoloFilepath)

    def loadTruthPoints(self):
        """Load 3D truth points from the configured truth-data file, if present."""
        if not self.camConfig.ThreeDTruthFilepath:
            return

        truth_path = Path(self.camConfig.ThreeDTruthFilepath)

        from support.io.ThreeD_truth import TruthPoints
        self.ThreeDTruthPoints = TruthPoints()
        self.ThreeDTruthPoints.try_load(truth_path)

    def updateQuality(self, qualityValue: str):
        """Update the configured export quality and persist the change."""
        self.camConfig.export_quality = ExportQuality(qualityValue)
        self.saveToCache()

    def ingestCalibration(self):
        """Load calibration data and prepare undistortion maps.

        Attempts to read the configured calibration file, updates related UI,
        propagates the calibration into the YOLO session, and precomputes
        OpenCV remap matrices for fast undistortion during playback.

        Important note: the undistort map does not have the same focal parameters as the original projection!
        """
        if not self.calibration.fromBinFile(self.camConfig.calibFilepath) and not self.calibration.fromFile(
                self.camConfig.calibFilepath):
            if self.selectCalibLabel is not None:
                self.selectCalibLabel.configure(text='No Calibration Found')
                self.after(10, self.update_idletasks)  # type: ignore[call-arg]
            return

        if not self.calibration.validCal:
            return

        if self.selectCalibLabel is not None:
            self.selectCalibLabel.configure(
                text="../" + Path(self.camConfig.calibFilepath).name if self.camConfig.calibFilepath else "../",
                bg_color=self.selectCalibLabel.cget("bg_color"))
            self.filepath_page.update_idletasks()
            self.update_idletasks()
            self.selectCalibLabel.update_idletasks()
            self.filepath_page.update_idletasks()
            self.update_idletasks()

        if self.yoloSession is not None:
            self.yoloSession.set_calibration(self.calibration)

        # New owner for remap caches
        self.fisheye_mgr.clear()

        # Always compute a remapK so downstream code can rely on it existing.
        newK, std_map1, std_map2 = self.fisheye_mgr.ensure_standard_undistort_maps(
            self.calibration,
            alpha=0.0,
        )
        self.calibration.remapK = newK

        # Only the non-fisheye path uses map1/map2 in this class.
        if self.calibration.fisheye:
            self.map1, self.map2 = None, None
        else:
            self.map1, self.map2 = std_map1, std_map2

        self.saveToCache()

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

        try:
            self.pauseCache.clear()
        except Exception:
            pass

        if reset_fg:
            self.FG = None
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
        except Exception:
            pass

    def startStreamOff(self):
        # UI thread only signals stop. The worker owns stream/window teardown.
        try:
            self.threadStopper.set()
        except Exception:
            pass

        self.showWindow = False
        self.stream_running_var.set(False)

        # OpenCV backend can still be nudged here safely.
        if self.vc is not None and self.vc.isOpened():
            try:
                self.vc.release()
            except Exception:
                pass
            self.vc = None

    def run_detectSingleImage(self):
        cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)
        frame = cv2.imread(str(Path(self.camConfig.imageFilepath)))
        while (not self.threadStopper.is_set()
               and self._window_is_open()
               and self.showWindow):

            self.analyze_image(frame)

            key = cv2.waitKey(1)
            if key == 27:
                self.threadStopper.set()
                break

        try:
            cv2.destroyWindow(self.windowName)
        except cv2.error:
            pass
        self.after(0, self._on_worker_exit)  # type: ignore[call-arg]

    def run(self):

        if self.camConfig.imageSource == ImageSource.Camera_Stream:
            self.run_video_stream()
        elif self.camConfig.imageSource == ImageSource.Stream_from_Folder:
            self.run_folder_reader()
        elif self.camConfig.imageSource == ImageSource.Static_Image:
            self.run_detectSingleImage()

    def queue_live_vimba_update(self, settings: dict[str, Any]) -> bool:
        """
        Thread-safe: UI calls this to request a live camera change.
        Returns True if the request was queued for a running Vimba stream.
        """
        if not bool(getattr(self.camConfig, "use_vimba", False)):
            return False
        if not bool(self.stream_running_var.get()):
            return False

        self.vimbaCam.live_update(settings)

        return True

    def run_vimba_stream(self, camConfig):
        if not HAVE_VMBPY:
            LOG.error("VmbPy is not installed or could not be imported.")
            self.after(0, self._on_worker_exit)
            return

        def handler(cam, stream, frame):
            if self.threadStopper.is_set() or not self.showWindow:
                try:
                    cam.queue_frame(frame)
                except Exception:
                    pass
                return

            try:
                self.vimbaCam.update_frame(frame,
                                           self.camConfig.vimba_profile)

            finally:
                try:
                    cam.queue_frame(frame)
                except Exception:
                    pass

        try:
            cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)

            with self.vimbaCam.vmbSystem_getInstance() as vmb:
                cam = self.vimbaCam.select_vimba_camera(vmb,
                                                        self.camConfig.vimba_camera_id)

                with cam:
                    self.vimbaCam.configure_stream(cam, self.camConfig)

                    try:
                        self.vimbaCam.start_stream(
                            handler,
                            self.camConfig.vimba_profile)

                        while (not self.threadStopper.is_set()
                               and self.showWindow
                               and not self.making_gifOrVid):

                            frame, img_time = self.vimbaCam.update_stream(self.camConfig)

                            if frame is not None:
                                self.curr_frame = frame
                                self.analyze_image(frame, img_time=img_time)

                            key = cv2.waitKey(1)
                            if key == 27:
                                self.threadStopper.set()
                                self.showWindow = False
                                break

                            if not self._window_is_open():
                                self.threadStopper.set()
                                self.showWindow = False
                                break

                    finally:
                        self.vimbaCam.stop_stream()

        except Exception as e:
            LOG.exception(f"Vimba stream failed: {e}")

        finally:
            self.vimbaCam.stop_lock()

            try:
                cv2.destroyWindow(self.windowName)
            except cv2.error:
                pass

            self.after(0, self._on_worker_exit)

    def run_video_stream(self):
        if bool(getattr(self.camConfig, "use_vimba", False)):
            self.run_vimba_stream(self.camConfig)
            return

        self.vc = cv2.VideoCapture(self.camConfig.cam_index, cv2.CAP_DSHOW)
        self.vc.set(cv2.CAP_PROP_FPS, 60)

        cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)
        rval, self.curr_frame = self.vc.read()
        if rval:
            cv2.resizeWindow(self.windowName, self.curr_frame.shape[1], self.curr_frame.shape[0])
            self.lastHeight = self.curr_frame.shape[0]
            self.lastWidth = self.curr_frame.shape[1]

        stop_display_time = None

        while (rval and not self.threadStopper.is_set() and
               self.showWindow and not self.making_gifOrVid):
            rval, frame = self.vc.read()

            if stop_display_time is not None:
                self._draw_chessboard_state(frame)

            self.analyze_image(frame)
            key = cv2.waitKey(1)

            if key == 27:
                self.after(0, self.filepath_page.toggle_stream)  # type: ignore[call-arg]
                self.threadStopper.set()
                break

            new_time = self._handle_chessboard_hotkeys(key)
            if new_time is not None:
                stop_display_time = new_time

            if stop_display_time is not None and time.monotonic() > stop_display_time:
                stop_display_time = None

            if not self._window_is_open():
                self.after(0, self.filepath_page.toggle_stream)  # type: ignore[call-arg]
                self.threadStopper.set()
                break

        if self.vc is not None and self.vc.isOpened():
            self.vc.release()
            self.vc = None

        try:
            cv2.destroyWindow(self.windowName)
        except cv2.error:
            pass

        self.after(0, self._on_worker_exit)  # type: ignore[call-arg]

    def _on_worker_exit(self):
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

        """Run the queued processing pipeline on a frame and optionally display it.

        Creates a per-frame context object, applies enabled image-processing
        steps in order, adds fixed overlays, and either displays the result or
        returns a processed image buffer for export.
        """

        if frame is None or frame.size == 0:
            return

        ctx = GuiQueue.FrameCtx(img_time=img_time,
                                name=name,
                                display_in_realtime=display_in_realtime)

        self.pnpResult = None
        self.qnpResult = None
        self.curr_frame_gray = None

        markup_frame = self.markup_frame
        # np.copyto is faster (doesn't reallocate), but requires destination to match shape
        if markup_frame is None or markup_frame.shape != frame.shape:
            markup_frame = frame.copy()
        else:
            np.copyto(markup_frame, frame)

        ######################################################
        for func, args in self.list_of_image_process_functors:
            step_args = args if isinstance(args, dict) else {}
            if not bool(step_args.get("state", True)):
                continue
            func(frame, markup_frame, ctx, step_args)

        if box_around and not self.screenshot_impending:
            self.draw_boxAround(frame, markup_frame, ctx, ())

        if self.camConfig.imageSource == ImageSource.Stream_from_Folder:
            self.draw_name(frame, markup_frame, ctx, ())
            if not self.screenshot_impending and not self.making_gifOrVid:
                self.draw_playbackStats(frame, markup_frame, ctx, ())

        self.draw_time(frame, markup_frame, ctx, ())

        if display_in_realtime:
            self.cleanup(markup_frame, )
        else:
            return np.ascontiguousarray(markup_frame).copy()
        ######################################################

    def cleanup(self, markupFrame, name=None):

        self.potentialResize(markupFrame)

        cv2.imshow(self.windowName if name is None else name,
                   cv2.resize(markupFrame, (self.lastWidth, self.lastHeight)))

        if ((self.recording and time.time() - self.lastImageTime > self.camConfig.secondsBetweenImages) or
                self.screenshot_impending):
            cv2.imwrite(os.path.join(self.camConfig.saveFolder, str(self.img_idx) + '.png'), markupFrame)
            self.img_idx += 1
            self.lastImageTime = time.time()
            self.recordButton.configure(text=f'Saving Imagery: #{self.img_idx}')
            self.screenshot_impending = False

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
        if ctx.img_time is None or ctx.img_time > 1_000_000:  # Alvium
            return

        time_str = f"Flight Time: {ctx.img_time:.2f}"  # + 173.11338 - 11.658461:.2f}"
        from support.viz.HUD_draw import draw_time_on_image
        draw_time_on_image(markupFrame, time_str)

    @staticmethod
    def draw_name(frame,
                  markupFrame,
                  ctx: GuiQueue.FrameCtx,
                  args) -> None:
        from support.viz.HUD_draw import draw_name_on_image
        draw_name_on_image(os.path.basename(ctx.name), markupFrame)

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

    def draw_boxAround(self, frame,
                       markupFrame,
                       ctx: GuiQueue.FrameCtx,
                       args) -> None:
        h, w, _ = markupFrame.shape
        cv2.rectangle(markupFrame, (0, 0), (w - 1, h - 1), clr.HUD_YELLOW, med_thick(h))

    def draw_chessboard(self, frame: NDArray,
                        markupFrame: NDArray,
                        ctx: GuiQueue.FrameCtx,
                        args) -> None:
        if self.curr_frame_gray is None:
            self.curr_frame_gray = cv2.cvtColor(markupFrame, cv2.COLOR_BGR2GRAY)

        self._checker_residual.draw_chessboard(markupFrame, self.curr_frame_gray, self._cb_pattern)

    def _draw_chessboard_state(self, frame):
        width, height, _ = frame.shape
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
        if not self._window_is_open() or markupFrame.shape[0] == 0:
            return
        x, y, width, height = cv2.getWindowImageRect(self.windowName)
        aspectRatio = markupFrame.shape[1] / markupFrame.shape[0]
        if not self._window_is_open():
            return

        if not self.lastHeight == height and height != 0:
            cv2.resizeWindow(self.windowName, int(height * aspectRatio), height)
            self.lastHeight = height
            self.lastWidth = int(height * aspectRatio)
        elif not self.lastWidth == width and width != 0:
            cv2.resizeWindow(self.windowName, width, int(width / aspectRatio))
            self.lastWidth = width
            self.lastHeight = int(width / aspectRatio)

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

    def image_filter_arg_specs(self, args):
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
        from support.vision.aprilTag_detection_and_aligment import (
            detect_apriltags_refined,
            draw_apriltag_detections,
            parse_apriltag_args,
        )

        opts = parse_apriltag_args(args)

        if self.detector is None:
            self.createDetector()

        result = detect_apriltags_refined(
            detector=self.detector,
            markup_frame=markupFrame,
            scale=opts.scale,
        )

        self.centers = result.centers
        self.detectIDS = result.ids

        if not result.ids:
            return

        if opts.inpaint:
            from support.vision.aprilTag_detection_and_aligment import inpaint_apriltags
            inpaint_apriltags(
                markup_frame=markupFrame,
                gray_small=result.small_gray,
                corners_small=result.corners_small,
            )
        else:
            draw_apriltag_detections(
                markup_frame=markupFrame,
                refined_corners_per_marker=result.refined_corners_per_marker,
                ids=result.ids,
            )

        if opts.pnp:
            self.pnp3DTruthPoints(frame, markupFrame, ctx, ())

        if opts.qnp:
            self.qnp3DTruthPoints(frame, markupFrame, ctx, ())

    def pnp3DTruthPoints(self,
                         frame: NDArray,
                         markupFrame: NDArray,
                         ctx: GuiQueue.FrameCtx,
                         args) -> None:

        if self.ThreeDTruthPoints is None:
            self.loadTruthPoints()

        if self.centers is not None and len(self.centers) >= 6:
            truthPoints = deepcopy(self.ThreeDTruthPoints.truthPoints)
            points = []
            distParams = np.zeros((5,))  # use image undistort instead

            removeIDs = []
            for idx, detectID in enumerate(self.detectIDS):
                try:
                    points.append(truthPoints[str(detectID[0])])
                except KeyError:
                    removeIDs.append(idx)

            centers = deepcopy(self.centers)
            for idx in reversed(removeIDs):
                centers = np.delete(centers, idx, axis=0)
            points = np.array(points)

            if len(points) < 6:
                return

            ret, rvec, tvec, *_ = cv2.solvePnPRansac(objectPoints=points,
                                                     imagePoints=centers,
                                                     cameraMatrix=self.calibration.getCameraMatrix(),
                                                     distCoeffs=distParams,
                                                     flags=cv2.SOLVEPNP_ITERATIVE)

            if ret:
                projectedPoints_orig, _ = cv2.projectPoints(self.ThreeDTruthPoints.getTruthPointsNumpy(),
                                                            rvec=rvec,
                                                            tvec=tvec,
                                                            cameraMatrix=self.calibration.getCameraMatrix(),
                                                            distCoeffs=distParams)

                self.plotOnImg(markupFrame, projectedPoints_orig[:, 0, :].astype(int),
                               list(self.ThreeDTruthPoints.getTruthPointsDict().keys()), clr.LIGHTBLUE)

                # quatCV = q.from_rodrigues(rvec)
                # tCV = np.squeeze(tvec)
                quatPnP, vectPnP = q.fromOpenCV_toAftr_rvec(rvec, tvec)

                self.pnpResult = (quatPnP, vectPnP)
                orient_text = 'Orientation (quat) From Truth Points: ' + format(quatPnP, 'ijk.6f')
                (txt_w, txt_h), _ = cv2.getTextSize(orient_text, cv2.FONT_HERSHEY_SIMPLEX,
                                                    small_text(markupFrame.shape[0]),
                                                    4)

                cv2.putText(markupFrame, orient_text,
                            (50, txt_h + 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            small_text(markupFrame.shape[0]),
                            clr.BLACK, 4)
                cv2.putText(markupFrame, orient_text,
                            (50, txt_h + 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            small_text(markupFrame.shape[0]),
                            clr.LIGHTBLUE, 2)
                cv2.putText(markupFrame, 'Location From Truth Frame: ' + np.array2string(vectPnP),
                            (50, 2 * txt_h + 15), cv2.FONT_HERSHEY_SIMPLEX,
                            small_text(markupFrame.shape[0]),
                            clr.BLACK, 4)
                cv2.putText(markupFrame, 'Location From Truth Frame: ' + np.array2string(vectPnP),
                            (50, 2 * txt_h + 15), cv2.FONT_HERSHEY_SIMPLEX,
                            small_text(markupFrame.shape[0]),
                            clr.LIGHTBLUE, 2)
                # LOG.info(f"SE3,Aftr Cam in Truth Frame: \n{quatPnP.T.to_SE3_given_position(quatPnP.T * -vectPnP)}")
        return

    def qnp3DTruthPoints(self,
                         frame: NDArray,
                         markupFrame: NDArray,
                         ctx: GuiQueue.FrameCtx,
                         args) -> None:

        if self.ThreeDTruthPoints is None:
            self.loadTruthPoints()

        if self.centers is not None and len(self.centers) >= 6:
            truthPoints = deepcopy(self.ThreeDTruthPoints.truthPoints)

            points = []
            # distParams = np.zeros((5,))  # use image undistort instead

            removeIDs = []
            for idx, detectID in enumerate(self.detectIDS):
                try:
                    points.append(truthPoints[str(detectID[0])])
                except KeyError:
                    removeIDs.append(idx)

            centers = deepcopy(self.centers)
            for idx in reversed(removeIDs):
                centers = np.delete(centers, idx, axis=0)
            points = np.array(points)

            if len(points) < 6:
                return

            quat, vect, *_ = solveQnP(points, centers, self.calibration, True)
            xyz_proj = quat * self.ThreeDTruthPoints.getTruthPointsNumpy() + vect

            q_aftr_from_cv = mat2quat(np.array([[0., 0., 1.],
                                                [-1., 0., 0.],
                                                [0., -1., 0.]], float))

            vect = q_aftr_from_cv * vect

            quat = q_aftr_from_cv * quat

            us_vs_s_proj = np.zeros((xyz_proj.shape[0], 2))
            us_vs_s_proj[:, 0] = self.calibration.fx * xyz_proj[:, 0] / xyz_proj[:, 2] + self.calibration.cx
            us_vs_s_proj[:, 1] = self.calibration.fy * xyz_proj[:, 1] / xyz_proj[:, 2] + self.calibration.cy

            self.plotOnImg(markupFrame, us_vs_s_proj.astype(int),
                           list(self.ThreeDTruthPoints.getTruthPointsDict().keys()), (255, 255, 255))
            self.qnpResult = (quat, vect)

            orient_text = 'Orientation (quat) From Truth Points: ' + format(quat, 'ijk.6f')
            pos_text = 'Location From Truth Frame: ' + np.array2string(vect)
            (txt_w, txt_h), _ = cv2.getTextSize(orient_text, cv2.FONT_HERSHEY_SIMPLEX,
                                                small_text(markupFrame.shape[0]),
                                                4)
            cv2.putText(markupFrame, orient_text,
                        (50, 3 * txt_h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        small_text(markupFrame.shape[0]),
                        clr.BLACK, 4)
            cv2.putText(markupFrame, orient_text,
                        (50, 3 * txt_h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        small_text(markupFrame.shape[0]),
                        clr.LIGHTBLUE, 2)
            cv2.putText(markupFrame, pos_text, (50, 4 * txt_h + 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        small_text(markupFrame.shape[0]),
                        clr.BLACK, 4)
            cv2.putText(markupFrame, pos_text, (50, 4 * txt_h + 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        small_text(markupFrame.shape[0]),
                        clr.LIGHTBLUE, 2)

        return

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
        if self.horizon_line is None:
            return True

        x1, y1, x2, y2 = self.horizon_line
        return np.cross(np.array([x2 - x1, y2 - y1]), np.array([pt[0] - x1, pt[1] - y1])) < 0

    def hyper_focus(self,
                    markupFrame: NDArray,
                    ctx: GuiQueue.FrameCtx) -> None:

        plan = build_hyper_focus_plan(
            ctx=ctx,
            radius=float(self.radius),
            min_radius=float(self.min_radius),
            frame_shape=markupFrame.shape,
        )

        if plan is None:
            return

        if plan.next_radius is not None:
            self.radius = float(plan.next_radius)

        if plan.next_min_radius is not None:
            self.min_radius = float(plan.next_min_radius)

        if self.yoloSession is not None and plan.desired_yolo_conf is not None:
            self.yoloSession.conf = float(plan.desired_yolo_conf)

        self._apply_hyper_focus_plan(markupFrame, plan)

    @staticmethod
    def _apply_hyper_focus_plan(markupFrame: NDArray, plan) -> None:
        if plan is None or plan.center is None:
            return

        center = (int(plan.center[0]), int(plan.center[1]))

        for p in plan.passes:
            dim_except_circle(markupFrame,
                              center,
                              x_axes=float(p.x_axes),
                              y_axes=float(p.y_axes),
                              dim_factor=float(p.dim_factor))

    def run_yolo(self,
                 frame: NDArray,
                 markupFrame: NDArray,
                 ctx: GuiQueue.FrameCtx,
                 args) -> None:
        """
        Runs YOLO on subsequent images. If the yolo model is single featured, and the object is estimated less than
        100 meters away, then it updates this class's estimation of the solution.
        :return: None, but does adjust
        """

        opts = self.parse_args(args, GuiQueue.YoloOpts())

        from support.vision import yolo
        if self.yoloSession is None:
            self.yoloSession = yolo.YOLO()
            self.yoloSession.setNewFolder(self.camConfig.yoloFilepath)
            self.yoloSession.set_calibration(self.calibration)
            self.yoloSession.iou = self.camConfig.yolo_iou
            self.yoloSession.conf = self.camConfig.yolo_conf

        import support.viz.draw_pnp_qnp as pnpDrw
        if self.pnpDrawer is None:
            self.pnpDrawer = pnpDrw.pnp_qnp_draw()
        # TODO Clean-up bias tracking logic, second input here
        output = self.yoloSession.inferOnImage(frame,
                                               False)

        algos = pnpDrw.twoToThreeSelectedAlgorithms()
        algos.use_pnp = opts.want_pnp
        algos.use_qnp = opts.want_qnp
        algos.use_wqnp = opts.want_wqnp

        scale = ctx.resize.get_or(1.0)

        pose_output = self.pnpDrawer.markUpImage(
            image=markupFrame,
            output=output,
            markup_is_undistorted=ctx.undistorted.get_or(False),
            calibration=self.calibration,
            conf=self.camConfig.yolo_conf,
            iou=self.camConfig.yolo_iou,
            yoloSize=self.yoloSession.yoloSize,
            idsNamesLocs=self.yoloSession.reader.idsNamesLocs,
            usedAlgos=algos,
            originalSize=frame.shape[:2],
            circles_not_features=opts.feature_circles,
            img_scale=scale
        )

        if pose_output is not None:
            self.pnpResult = {
                "rvec": pose_output.pnp_rvec,
                "tvec": pose_output.pnp_tvec,
                "object_points": pose_output.object_points,
                "image_points": pose_output.image_points,
                "class_ids": pose_output.class_ids,
            } if pose_output.pnp_rvec is not None and pose_output.pnp_tvec is not None else None

            self.qnpResult = {
                "q": pose_output.qnp_q,
                "tvec": pose_output.qnp_tvec,
                "object_points": pose_output.object_points,
                "image_points": pose_output.image_points,
                "class_ids": pose_output.class_ids,
            } if pose_output.qnp_q is not None and pose_output.qnp_tvec is not None else None
        else:
            self.pnpResult = None
            self.qnpResult = None

        centers, boxes, scores, class_ids, img_time = output
        last_yolo_center = None
        last_bounding_box_size = None
        last_yolo_3d_estimate = None
        if len(centers) > 0 and self.yoloSession.reader.numClasses == 1:

            best_idx = scores.index(max(scores))
            img_yolo_x_correction = markupFrame.shape[0] / self.yoloSession.reader.imageSize
            img_yolo_y_correction = markupFrame.shape[1] / self.yoloSession.reader.imageSize

            last_bounding_box_size = ((boxes[best_idx][2] - boxes[best_idx][0]) * img_yolo_x_correction,
                                      (boxes[best_idx][3] - boxes[best_idx][1]) * img_yolo_y_correction)
            last_yolo_center = (int(
                centers[best_idx][0] * img_yolo_x_correction), int(
                centers[best_idx][1] * img_yolo_y_correction))

            self.calibration.scaleCalibration(markupFrame.shape[0])
            K = self.calibration.getCameraMatrix()
            # d = self.calibration.getDistortion()  # Presume undistorted image
            twoD_points = np.array([last_yolo_center[0], last_yolo_center[1], 1.0]) * scale
            dist_est = self.calibration.fx * 4.07 / (last_bounding_box_size[0])

            if self.check_above_horizon(last_yolo_center):
                last_yolo_3d_estimate = np.linalg.inv(K).dot(twoD_points) * dist_est
                w, h, _ = markupFrame.shape
                (txt_width, txt_height), base = cv2.getTextSize('I',
                                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                                med_text(w), med_thick(h))
                pad = int(0.3 * txt_height)
                cv2.putText(markupFrame, 'BB-Width Solution',
                            (pad, w - 2 * pad - txt_height),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            med_text(markupFrame.shape[0]), (50, 255, 255), med_thick(h))
                cv2.putText(markupFrame,
                            f'x:{last_yolo_3d_estimate[0]:+.3f}, y:{last_yolo_3d_estimate[1]:+.3f}, ' +
                            f'z:{last_yolo_3d_estimate[2]:+.3f}',
                            (pad, h - pad),
                            cv2.FONT_HERSHEY_SIMPLEX, med_text(markupFrame.shape[0]), (50, 255, 255), med_thick(h))
            else:
                last_yolo_3d_estimate = None

        ctx.yolo.set(YoloOutput(
            last_bounding_box_size=last_bounding_box_size,
            last_yolo_center=last_yolo_center,
            last_yolo_3d_estimate=last_yolo_3d_estimate,
            pose=pose_output
        ))

        if opts.factor_graph:
            self.factor_graph(frame, markupFrame, ctx, opts.hyper_focus)

    def _draw_factor_graph_overlay(self,
                                   markupFrame: NDArray,
                                   fg_output,
                                   color) -> None:
        if fg_output is None or fg_output.curr_FG_pixel is None:
            return

        pixel = (
            int(fg_output.curr_FG_pixel[0]),
            int(fg_output.curr_FG_pixel[1]),
        )

        h, w, _ = markupFrame.shape
        size = int(0.025 * h)
        thickness = lrg_thick(h)
        text = 'Factor Graph Solution'
        (_txt_width, txt_height), _base = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            med_text(w),
            thickness,
        )
        pad = int(0.3 * txt_height)
        txt_height_perRow = txt_height + pad
        loc = (pad, h - 4 * txt_height_perRow - pad)

        cv2.circle(markupFrame, pixel, size, (0, 0, 0), thickness)
        cv2.line(markupFrame,
                 [pixel[0] + size, pixel[1]],
                 [pixel[0] - size, pixel[1]],
                 (0, 0, 0), thickness)
        cv2.line(markupFrame,
                 [pixel[0], pixel[1] + size],
                 [pixel[0], pixel[1] - size],
                 (0, 0, 0), thickness)

        thickness = med_thick(h)
        cv2.circle(markupFrame, pixel, size, color, thickness)
        cv2.line(markupFrame,
                 [pixel[0] + size, pixel[1]],
                 [pixel[0] - size, pixel[1]],
                 color, thickness)
        cv2.line(markupFrame,
                 [pixel[0], pixel[1] + size],
                 [pixel[0], pixel[1] - size],
                 color, thickness)

        cv2.putText(markupFrame, text,
                    loc, cv2.FONT_HERSHEY_SIMPLEX,
                    med_text(markupFrame.shape[0]), (0, 0, 0), lrg_thick(h))
        cv2.putText(markupFrame, text,
                    loc, cv2.FONT_HERSHEY_SIMPLEX,
                    med_text(markupFrame.shape[0]), color, med_thick(h))

    def factor_graph(self,
                     frame: NDArray,
                     markupFrame: NDArray,
                     ctx: GuiQueue.FrameCtx,
                     hyper_focus: bool) -> None:

        yolo = ctx.yolo.get_or()
        color = clr.YELLOWGREEN if yolo is not None else clr.RED

        R_wr = (
            self.own_attitude.rotmat_wr()
            if self.own_attitude is not None and self.own_attitude.valid
            else None
        )

        self.FG, self.last_time_update, has_measurement, pred = run_factor_graph_step(
            fg=self.FG,
            yolo=yolo,
            img_time=ctx.img_time,
            last_time_update=self.last_time_update,
            R_wr=R_wr,
        )

        if not has_measurement:
            color = clr.RED

        if pred is not None and pred.r_T_d is not None:
            K = factor_graph_projection_matrix(
                calibration=self.calibration,
                markup_frame=markupFrame,
                yolo=yolo,
            )

            fg_output = build_factor_graph_output(pred, K)
            if fg_output is not None:
                ctx.fg.set(fg_output)
                self._draw_factor_graph_overlay(markupFrame, fg_output, color)

        if hyper_focus:
            self.hyper_focus(markupFrame, ctx)

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
        """Convert OpenCV camera pose to your convention (proper rotation)."""
        S_MODEL = np.diag([1., -1., 1.])  # det = -1
        C_OURS_TO_CV = np.array([[0., -1., 0.],
                                 [0., 0., 1.],
                                 [1., 0., 0.]], dtype=float)
        C_CV_TO_OURS = C_OURS_TO_CV.T
        R_ours = C_CV_TO_OURS @ R_cv @ S_MODEL
        t_ours = C_CV_TO_OURS @ t_cv
        return mat2quat(R_ours.T), t_ours

    def run_folder_reader_profiled(self):
        from support.io.profiler import make_profile, print_stats

        prof = make_profile()
        try:
            prof.enable()
            self.run_folder_reader()
        finally:
            print_stats(prof)
