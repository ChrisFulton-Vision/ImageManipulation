import copy
import os
import pickle
import time
import sys
import threading
from collections import deque

# import vmbpy.c_binding
# from vmbpy import *

import numpy as np
from pathlib import Path
from tkinter import filedialog
from yaml import safe_load, dump

from customtkinter import (CTkFrame, CTkButton, CTkLabel, CTkSlider, CTkEntry, CTkCheckBox, CTkComboBox, BooleanVar,
                           StringVar, CTkProgressBar, END)
import cv2

from support.mathHelpers.twoD_to_threeD import solveQnP
from support.mathHelpers.quaternions import Quaternion as q, mat2quat
from support.core.enums import ExportQuality, ImageKernel, ImageSource, PlaybackSpeed
import support.gui.utils as utils
from support.io.my_logging import LOG
import support.io.camera_config as camConfig
from support.io.config_store import ConfigStore
from support.vision.calibration import Calibration, undistort_points_px
from support.io.image_time_reader import ImageTimeReader
import support.io.data_processing as data
from support.viz.CVFontScaling import small_text, med_text
from support.gui.checkerboard_launcher import CheckerboardLauncher, CheckerboardLaunchState
from support.gui.gpu_monitor import GpuMonitor, GpuSample
from support.vision.draw_circle_and_mask import dim_except_circle
from support.viz.checkerboard_stats import CheckerboardResiduals as CkR
import support.viz.colors as clr

from copy import deepcopy
from math import pow

SPEED_STEP = pow(2.0, 1.0 / 3.0)  # 3 presses -> 2×
SPEED_STEP_INV = 1.0 / SPEED_STEP

# cv2.setNumThreads(0)
cv2.setUseOptimized(True)

#  pip install cv2_enumerate_cameras
#  or
#  pip install git+https://github.com/chinaheyu/cv2_enumerate_cameras.git



CACHE_FILEPATH = str(Path.cwd() / "Caches" / "last_config.pkl")

class CameraGui(CTkFrame):
    def __init__(self, master, *args, **kwargs):

        self.func_that_refits = None

        # Debounced cache writes
        self._save_debounce_id = None

        # Super class init, necessary for customTkinter
        super().__init__(master, *args, **kwargs)
        self._flag_vars: dict[str, BooleanVar] = {}
        self._checkboxes: dict[str, CTkCheckBox] = {}
        self._flags = [
            "detectTags", "undistort", "pnpLidarPoints", "qnpLidarPoints",
            "yoloInference", "yoloBiasTracking", "detect_corners", "detect_horizon",
            "factor_graph", "hyper_focus", "phase_correlation", "crosshairs",
            "cubemap", "hud", "hideAprilTags", "draw_chessboard",
            # --- Pose from YOLO detections (multi-feature) ---
            # These are separate from the AprilTag/LiDAR toggles above.
            "pnpYoloPoints", "qnpYoloPoints", "qnpKFYoloPoints",
        ]
        self.recording = False
        self.yoloSession = None
        self.camConfig = camConfig.CameraConfig()
        self.detector = None
        self.arucoDict = None
        self.arucoParams = None

        self._init_flag_vars()

        self.calibration = Calibration()
        self.config_store = ConfigStore(CACHE_FILEPATH, configs_dir="Configs", scheduler=self)
        self.detectIDS = None
        self.centers = None
        self.indexDict = {}
        self.scanForCameras()
        self.windowName = 'Processed Image'
        self.default_filepath = ''
        self.cam_frame = CTkFrame(master=master)
        self.config_frame = CTkFrame(master=master)
        self.export_frame = CTkFrame(master=master)
        self.playback_frame = CTkFrame(master=master)
        self.data_frame = CTkFrame(master=master)
        self.hotkey_frame = CTkFrame(master=master)
        self.showWindow = False
        self.GaborGUI = None
        self.radius = 800
        self.last_bounding_box_size = (800, 800)
        self.last_yolo_center = (400, 400)
        self.last_yolo_3d_estimate = (10, 0, 0)
        self.current_center_est = (400, 400)
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
        self.shutting_down = False
        self.printLidar = False
        self.face_size = None
        self.faces_dirs = None
        self.cubemap_faces = None
        self.map_x = None
        self.map_y = None
        self.hud_marker = None
        self.lowPassFPS = 20.0
        self.pnpResult = None
        self.qnpResult = None
        self.pnpDrawer = None
        # --- Pose results from YOLO detections (multi-feature) ---
        self.pnpYoloResult = None
        self.qnpYoloResult = None
        self.qnpKFYoloResult = None
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

        self.vc = None

        self.pauseCache = utils.PausedCache()
        self.playback = utils.PlaybackState()

        # Optimization for undistort
        self.map1, self.map2 = None, None

        self.threadStopper = utils.ThreadStopper()
        self._thread = None

        self._pb_cmds = deque()
        self._pb_cmd_lock = threading.Lock()
        self._pb_num_images = 0
        self._pb_slider_dragging = False
        self._pb_last_sent_idx = None
        self._pb_slider_range_inited = False
        self._pb_slider = None

        self.fps_time_log = time.time()
        self.curr_fps = 20.0
        self.pause = False
        self.last_nonzero_sign = 1

        self.imageProcessingKernelCombobox = None

        self.last_image = None

        self.available_sources = [source.value for source in ImageSource]

        self.streamOrImgCombo = CTkComboBox(self.cam_frame, values=self.available_sources,
                                            command=self.sourceUpdate)
        self.startStreamButton = CTkButton(master=self.cam_frame, text='Start Stream', fg_color=clr.CTK_BUTTON_RED,
                                           hover_color='blue')

        self.configSelectButton = CTkButton(self.cam_frame, text='Select Config File',
                                            command=self.selectConfigFile)
        self.configSelectLabel = CTkLabel(self.cam_frame, text=os.path.basename(self.camConfig.configFilepath))

        self.singleImageFolderSelect = CTkButton(self.cam_frame, text='Select Img',
                                                 command=self.selectImagesFilepath)
        self.singleImageTextButton = CTkButton(self.cam_frame, text='No Image Selected')
        self.multiImageFolderSelect = CTkButton(self.cam_frame, text='Select Img Folder',
                                                command=self.selectImagesFilepath)
        self.multiImageTextButton = CTkButton(self.cam_frame, text='No Folder Selected', command=self.startStreamOn)

        self.recordButton = CTkButton(master=self.export_frame, text='Saving Imagery', fg_color='green',
                                      hover_color='navy', command=self.recordOff)
        self.printButton = CTkButton(master=self.export_frame, text='Print LiDAR', fg_color='green',
                                     hover_color='navy', command=self.printLidarOnce)
        self.selectCameraCombo = CTkComboBox(self.cam_frame, values=list(self.indexDict.keys()),
                                             command=self.selectCamera)
        self.selectFolderLabel = CTkLabel(self.cam_frame,
                                          text="../" + Path(
                                              self.default_filepath).name if self.default_filepath else "../")
        self.selectTruthPointsButton = CTkButton(master=self.cam_frame, text='Select LIDAR Points',
                                                 hover_color='blue', command=self.selectLidarFile)
        self.selectFlightLogButton = CTkButton(master=self.cam_frame, text='Select Flight Log File',
                                               hover_color='blue', command=self.selectLogFile)

        self.playbackModeText = StringVar(value='Playback Mode: FPS')
        self.update_playbackMenu()

        if self.camConfig.lidarFilepath is not None:
            self.selectTruthPointsLabel = CTkLabel(self.cam_frame,
                                                   text="../" + Path(
                                                       self.camConfig.lidarFilepath).name if self.camConfig.lidarFilepath else "../")
        else:
            self.selectTruthPointsLabel = CTkLabel(self.cam_frame, text='No Truth Loaded')

        if self.camConfig.hud_data_filepath is not None:
            self.selectFlightLogLabel = CTkLabel(self.cam_frame, text="../" + Path(
                self.camConfig.hud_data_filepath).name if self.camConfig.hud_data_filepath else "../")
        else:
            self.selectFlightLogLabel = CTkLabel(self.cam_frame, text='No Flight Log Loaded')

        self.lidarTruthPoints = None
        self.selectYOLO_folderButton = CTkButton(self.cam_frame, text='Select YOLO Folder', fg_color=clr.CTK_GREEN,
                                                 command=self.selectYoloFolder)
        self.selectYOLO_folderLabel = CTkLabel(self.cam_frame,
                                               text="../" + Path(
                                                   self.camConfig.yoloFilepath).name if self.camConfig.yoloFilepath else "../"
                                               )
        self.selectCalibLabel = None
        self.drawChessboardButton = CTkCheckBox(self.config_frame, text='Draw Chessboard',
                                                variable=self._flag_vars['draw_chessboard'])
        self.undistortCheckbox = CTkCheckBox(self.config_frame, text='Undistort',
                                             variable=self._flag_vars['undistort'])
        self.detectAprilTagsCheckbox = CTkCheckBox(
            self.config_frame, text="Detect April Tags",
            variable=self._flag_vars["detectTags"]
        )
        self.hideAprilTagsCheckbox = CTkCheckBox(
            self.config_frame, text="Hide April Tags",
            variable=self._flag_vars['hideAprilTags']
        )
        self.detectHorizonCheckbox = CTkCheckBox(self.config_frame, text='Detect Horizon',
                                                 variable=self._flag_vars['detect_horizon'])

        self.yoloInferenceCheckbox = CTkCheckBox(self.config_frame, text='Run YOLO on image',
                                                 variable=self._flag_vars['yoloInference'])

        self.yoloBiasCheckbox = CTkCheckBox(self.config_frame, text='Run YOLO Bias Tracking',
                                            variable=self._flag_vars['yoloBiasTracking'])
        self.factorgraphCheckbox = CTkCheckBox(self.config_frame, text='Factor Graph',
                                               variable=self._flag_vars['factor_graph'])
        self.hyperfocusCheckbox = CTkCheckBox(self.config_frame, text='Hyper Focus',
                                              variable=self._flag_vars['hyper_focus'])
        self.phaseCorrelationCheckbox = CTkCheckBox(self.config_frame, text='PhaseCorrelation',
                                                    variable=self._flag_vars['phase_correlation'])
        self.crosshairsCheckbox = CTkCheckBox(self.config_frame, text='Crosshairs',
                                              variable=self._flag_vars['crosshairs'])
        self.cubemapCheckbox = CTkCheckBox(self.config_frame, text='Cubemap',
                                           variable=self._flag_vars['cubemap'])
        self.hudCheckbox = CTkCheckBox(self.config_frame, text='HUD',
                                       variable=self._flag_vars['hud'])
        self.confSliderLabel = CTkLabel(self.config_frame, text='Conf: 0.75')
        self.confSliderBar = CTkSlider(self.config_frame, command=self.confSlider,
                                       from_=0.15)  # type: ignore[arg-type]  # safe to ignore, ctk accepts float
        self.iouSliderLabel = CTkLabel(self.config_frame, text='IOU: 1.00')
        self.iouSliderBar = CTkSlider(self.config_frame, command=self.iouSlider)

        self.exportQualityCombo = CTkComboBox(self.export_frame, values=[member.value for member in ExportQuality],
                                              command=self.updateQuality)
        self.exportToGifButton = CTkButton(self.export_frame, text="Export to Gif", command=self.exportToGif)
        self.exportToVidButton = CTkButton(self.export_frame, text="Export to Vid", command=self.exportToVid)
        self.making_gifOrVid = False

        self.loadFromCache()

        self._sync_flags_from_model()

        self.exportStartFrame = CTkLabel(self.export_frame, text=f'Start Frame: {self.camConfig.start_export_idx}')
        self.exportEndFrame = CTkLabel(self.export_frame, text=f'End Frame: {self.camConfig.end_export_idx}')

        self.btn_checkerboard = CTkButton(self.export_frame,
                                          state='normal',
                                          text='Checkerboard',
                                          command=self.launch_checkerboard)

        self.confSliderLabel.configure(text=f'Conf: {self.camConfig.yolo_conf:.2f}')
        self.confSliderBar.set(self.camConfig.yolo_conf)
        self.iouSliderLabel.configure(text=f'IOU: {self.camConfig.yolo_iou:.2f}')
        self.iouSliderBar.set(self.camConfig.yolo_iou)

        self.selectCameraCombo.set(list(self.indexDict.keys())[self.camConfig.cam_index])

        self.img_idx = 0
        self.timeBetweenImgsEntry = None
        self.lastImageTime = 0
        self.cam_frame.grid_rowconfigure(list(range(3)), weight=1)  # configure grid system
        self.cam_frame.grid_columnconfigure(list(range(3)), weight=1)
        self.lastWidth = 1
        self.lastHeight = 1
        self.saveToCache(immediate=True)

        self._ui_active = True
        self._last_ui_tick = 0.0
        self._ui_throttle_sec = 0.10  # refresh UI at most every 100 ms

        self.setupFrame()

    def on_app_close(self):

        root = self.winfo_toplevel()

        try:
            if self.gpu_monitor is not None:
                self.gpu_monitor.stop()
        except Exception as e:
            pass

        try:
            # returns a list of after handler IDs
            after_ids = root.tk.call("after", "info")
            for aid in after_ids:
                try:
                    root.after_cancel(aid)
                except Exception:
                    pass
        except Exception:
            pass

        self._plotter_close_plot_alias()

        try:
            root.quit()
        except Exception:
            pass
        try:
            root.destroy()
        except Exception:
            pass


    def func_to_refit(self, func):
        self.func_that_refits = func

    def _init_flag_vars(self):
        for name in self._flags:
            if not hasattr(self.camConfig, name):
                setattr(self.camConfig, name, False)
            v = BooleanVar(value=bool(getattr(self.camConfig, name, False)))

            # when UI flips, write to model
            v.trace_add("write", lambda *_, n=name: self._on_flag_changed(n))
            self._flag_vars[name] = v

    def _on_flag_changed(self, name: str):
        val = bool(self._flag_vars[name].get())
        # guard rails / side-effects
        if name == "undistort" and not self.calibration.validCal:
            # can't enable; snap back off
            self._flag_vars[name].set(False)
            return

        if name == "detectTags":
            if val and self.detector is None:
                self.createDetector()
            if not val:
                self.detector = None

        setattr(self.camConfig, name, val)
        self.saveToCache()

    # keep model -> UI sync helper (if you ever load cache, etc.)
    def _sync_flags_from_model(self):
        for n in self._flags:
            self._flag_vars[n].set(bool(getattr(self.camConfig, n, False)))
        if self.camConfig.detectTags:
            self.createDetector()

    def _sync_dp_from_model(self):
        """Resync batch-processing (DP) UI controls from camConfig.

        This makes DP settings loadable from the YAML config, not just whatever
        was last typed into the UI.
        """

        # strings
        if hasattr(self, "_dp_img_dir_var") and self._dp_img_dir_var is not None:
            self._dp_img_dir_var.set(str(getattr(self.camConfig, "dp_img_dir", "") or ""))

        # floats / ints
        if hasattr(self, "_dp_conf_list") and self._dp_conf_list is not None:
            self._dp_conf_list.set(str(getattr(self.camConfig, "dp_conf_list", "") or ""))

        if hasattr(self, "_dp_ckptN") and self._dp_ckptN is not None:
            self._dp_ckptN.set(str(getattr(self.camConfig, "dp_ckptN", 200) or 200))

        if hasattr(self, "_dp_prefetch") and self._dp_prefetch is not None:
            self._dp_prefetch.set(str(getattr(self.camConfig, "dp_prefetch", 4) or 4))

        # gpu checkbox
        if hasattr(self, "_dp_gpu_var") and self._dp_gpu_var is not None:
            self._dp_gpu_var.set(bool(getattr(self.camConfig, "dp_gpu", False)))

    def set_ui_active(self, active: bool):
        self._ui_active = bool(active)
        # Stop camera stream if page is hidden (don’t burn CPU/GPU off-screen)
        if not self._ui_active:
            try:
                self.after(100, self.startStreamOff)
            except Exception:
                pass
        else:
            # Optionally auto-resume prior state; or keep manual
            pass

    def _ui_should_paint(self, widget=None) -> bool:
        # Throttle + only if page/target is visible
        if not self._ui_active:
            return False
        if widget is not None:
            try:
                if not widget.winfo_viewable():
                    return False
            except Exception:
                return False
        now = time.monotonic()
        if (now - self._last_ui_tick) >= self._ui_throttle_sec:
            self._last_ui_tick = now
            return True
        return False

    # Optional: react to section changes if you want different behavior
    def on_section_show(self, name: str):
        # Example: only allow OpenCV windows / key polling while in Playback
        self._playback_allowed = (name == "Playback")

    # def on_section_hide(self, name: str):
    # if name == "Playback":
    # Close imshow windows / pause playback, etc.
    # try: destroyWindow(self.windowName)
    # except Exception: pass

    def loadFromCache(self):
        res = self.config_store.load_from_cache(self.camConfig)
        if res.yaml_path:
            self.configSelectLabel.configure(text=os.path.basename(res.yaml_path))
        if res.loaded_yaml:
            self.update_post_newCamConfig()

    def update_post_newCamConfig(self):
        self.updateSingleOrStream(rowID=1)
        self.updateLogFile()
        self.ingestCalibration()
        self.updateYOLOLabel()
        self.updateLidarLabel()
        if self.lidarTruthPoints is not None:
            self.loadTruthPoints()

        # --- model -> UI resync on config load (batch DP + flags) ---
        try:
            if hasattr(self, "_sync_flags_from_model"):
                self._sync_flags_from_model()
        except Exception:
            pass

        try:
            if hasattr(self, "_sync_dp_from_model"):
                self._sync_dp_from_model()
        except Exception:
            pass

        # Ensure GPU UI matches dp_gpu setting (avoid cached desync)
        try:
            if hasattr(self, "gpu_slider"):
                self.gpu_slider.configure(
                    state="normal" if bool(
                        getattr(self.camConfig, "dp_gpu", False)) else "disabled")

            if not bool(getattr(self.camConfig, "dp_gpu", False)):
                self.gpu_slider.set(0.0)

        except Exception:
            pass

        self.selectFolderLabel.configure(text=os.path.basename(self.camConfig.saveFolder))

        if self.func_that_refits is not None:
            self.func_that_refits()

    def _flush_cache_now(self):
        """Actually write current config to disk. Called by saveToCache()."""
        cache_path = Path(CACHE_FILEPATH)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Pointer to the most recent config YAML
        with cache_path.open('wb') as f:
            pickle.dump(self.camConfig.configFilepath, f)

        # Full YAML config
        os.makedirs('Configs', exist_ok=True)
        with open(self.camConfig.configFilepath, 'w') as f:
            dump(self.camConfig.toDict, f)

    def saveToCache(self, immediate: bool = False, delay_ms: int = 500):
        self.config_store.save_to_cache(self.camConfig, immediate=immediate, delay_ms=delay_ms)

    def selectFolder(self):
        init_dir = Path(self.default_filepath).parent if self.default_filepath else Path.cwd()
        fp = self.askFilepath(str(init_dir), "Select Imagery Folder")
        if fp:
            self.camConfig.saveFolder = fp
            self.saveToCache(immediate=True)
            self.loadFromCache()
            self.selectFolderLabel.configure(text=os.path.basename(self.camConfig.saveFolder))

    def loadCalibration(self):
        init_dir = Path(self.camConfig.calibFilepath or self.default_filepath or Path.cwd()).parent
        poss_filepath = filedialog.askopenfilename(initialdir=str(init_dir), title='Select Calibration File')
        if poss_filepath:
            self.camConfig.calibFilepath = poss_filepath
            self.ingestCalibration()

    def selectLidarFile(self):
        init_dir = Path(self.camConfig.lidarFilepath or self.default_filepath or Path.cwd()).parent
        poss_filepath = filedialog.askopenfilename(initialdir=str(init_dir), title='Select LIDAR Truth Points')
        if poss_filepath:
            self.camConfig.lidarFilepath = poss_filepath
            self.updateLidarLabel()
            self.loadTruthPoints()
            self.saveToCache()

    def selectLogFile(self):
        init_dir = Path(self.camConfig.hud_data_filepath or self.default_filepath or Path.cwd())
        poss_dir = filedialog.askdirectory(initialdir=str(init_dir), title='Select Flight Log Data')
        if poss_dir:
            self.camConfig.hud_data_filepath = poss_dir
            self.updateLogFile()

    def updateLogFile(self):
        if self.hud_marker is not None:
            self.hud_marker.read_attitude_files(self.camConfig.hud_data_filepath)
        self.updateFlightLogLabel()
        self.saveToCache()

    def selectYoloFolder(self):
        init_dir = Path(self.camConfig.yoloFilepath or Path.cwd())
        poss_dir = filedialog.askdirectory(initialdir=str(init_dir), title='Select YOLO Folder')
        if poss_dir:
            self.camConfig.yoloFilepath = poss_dir
            self.updateYOLOLabel()
            self.saveToCache()

    def updateLidarLabel(self):
        if self.camConfig.lidarFilepath:
            self.selectTruthPointsLabel.configure(text=Path(self.camConfig.lidarFilepath).name)

    def updateFlightLogLabel(self):
        if self.camConfig.hud_data_filepath:
            self.selectFlightLogLabel.configure(text=Path(self.camConfig.hud_data_filepath).name)

    def updateYOLOLabel(self):
        if self.camConfig.yoloFilepath:
            self.selectYOLO_folderLabel.configure(text=Path(self.camConfig.yoloFilepath).name)
            if self.yoloSession is not None:
                self.yoloSession.setNewFolder(self.camConfig.yoloFilepath)

    def loadTruthPoints(self):
        if not self.camConfig.lidarFilepath:
            return

        lidar_path = Path(self.camConfig.lidarFilepath)
        if not lidar_path.exists():
            LOG.error(
                f'Cached LiDAR file not found. Using defaults. Attempted filepath:\n{self.camConfig.lidarFilepath}'
            )
            return

        from support.io.lidar_truth import TruthPoints
        self.lidarTruthPoints = TruthPoints()

        try:
            with lidar_path.open('rb') as f:
                obj = pickle.load(f)
        except AttributeError as e:
            # Old pickle referring to __main__.TruthPoints or otherwise broken:
            LOG.warning("Failed to unpickle LiDAR truth points (%s). Using defaults instead.", e)
            obj = TruthPoints()  # fall back to code-defined truth points
        except Exception as e:
            LOG.error("Error loading LiDAR truth points: %s", e)
            return

        # Accept either a TruthPoints instance or a raw dict
        if isinstance(obj, TruthPoints):
            self.lidarTruthPoints.copy(obj)
        elif isinstance(obj, dict):
            # Existing self.lidarTruthPoints is a TruthPoints()
            self.lidarTruthPoints.truthPoints = deepcopy(obj)
        else:
            LOG.error("Unexpected LiDAR truth data type: %r", type(obj))

    def updateQuality(self, qualityValue: str):
        self.camConfig.export_quality = ExportQuality(qualityValue)
        self.saveToCache()

    def confSlider(self, confValue):
        self.camConfig.yolo_conf = confValue
        if self.yoloSession is not None:
            self.yoloSession.conf = confValue
        self.confSliderLabel.configure(text='Conf: ' + f'{confValue:.2f}')
        self.saveToCache()

    def iouSlider(self, iouValue):
        self.camConfig.yolo_iou = iouValue
        if self.yoloSession is not None:
            self.yoloSession.iou = iouValue
        self.iouSliderLabel.configure(text='IOU: ' + f'{iouValue:.2f}')
        self.saveToCache()

    def ingestCalibration(self):

        if not self.calibration.fromBinFile(self.camConfig.calibFilepath) and not self.calibration.fromFile(
                self.camConfig.calibFilepath):
            if self.selectCalibLabel is not None:
                self.selectCalibLabel.configure(text='No Calibration Found')
                self.after(10, self.update_idletasks())
            return

        if not self.calibration.validCal:
            return

        if self.selectCalibLabel is not None:
            self.selectCalibLabel.configure(
                text="../" + Path(self.camConfig.calibFilepath).name if self.camConfig.calibFilepath else "../",
                bg_color=self.selectCalibLabel.cget("bg_color"))
            self.cam_frame.update_idletasks()
            self.update_idletasks()
            self.selectCalibLabel.update_idletasks()
            self.cam_frame.update_idletasks()
            self.update_idletasks()

        self.undistortCheckbox.configure(state='normal')
        if self.calibration.fisheye:
            cube_state = 'normal'
        else:
            cube_state = 'disabled'
            self.camConfig.cubemap = False
            self.cubemapCheckbox.deselect()
        self.cubemapCheckbox.configure(state=cube_state)

        if self.yoloSession is not None:
            self.yoloSession.set_calibration(self.calibration)

        w, h = self.calibration.width, self.calibration.height
        K = self.calibration.getCameraMatrix()
        D = self.calibration.getDistortion()

        newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0)

        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            K, D, R=None, newCameraMatrix=newK, size=(w, h), m1type=cv2.CV_16SC2
        )

        self.saveToCache()

    def scanForCameras(self):
        self.indexDict = {}
        from cv2_enumerate_cameras import enumerate_cameras
        for camera_info in enumerate_cameras(cv2.CAP_DSHOW):
            self.indexDict[camera_info.name] = camera_info.index

        # with VmbSystem.get_instance() as vmb:
        #     cams = vmb.get_all_cameras()
        #     if cams:
        #         cam = cams[0]
        #         try:
        #             cam._open()
        #         except vmbpy.c_binding.VmbError as e:
        #             LOG.warning(f'Could not open camera: {e}')
        #             return
        #         try:
        #             cam.start_streaming(
        #                 lambda cam, stream, frame: self.display_frame(cam, stream, frame, "Camera Stream"))
        #             time.sleep(5)
        #             cam.stop_streaming()
        #         finally:
        #             cam._close()

    # def display_frame(self, cam, stream, frame, title):
    #     try:
    #         numpy_buffer = frame.as_numpy_ndarray()
    #         if len(numpy_buffer.shape) == 2:
    #             numpy_buffer = cvtColor(numpy_buffer, COLOR_GRAY2BGR)
    #         else:
    #             numpy_buffer = cvtColor(numpy_buffer, COLOR_RGB2BGR)
    #         imshow(title, resize(numpy_buffer, (864, 864)))
    #         waitKey(1)
    #     except vmbpy.c_binding.VmbError as e:
    #         LOG.error("Error processing frame: %s", e)

    def selectCamera(self, key):
        self.camConfig.cam_index = self.indexDict[key]

    def sourceUpdate(self, source):

        self.camConfig.imageSource = ImageSource(source)

        self.updateSingleOrStream(rowID=1)

    def updateSingleOrStream(self, rowID):
        if not self.camConfig.imageSource == ImageSource.Camera_Stream:
            if self.selectCameraCombo.grid_info():
                self.selectCameraCombo.grid_forget()
            if self.startStreamButton.grid_info():
                self.startStreamButton.grid_forget()

        if not self.camConfig.imageSource == ImageSource.Static_Image:
            if self.singleImageFolderSelect.grid_info():
                self.singleImageFolderSelect.grid_forget()
            if self.singleImageTextButton.grid_info():
                self.singleImageTextButton.grid_forget()

        if not self.camConfig.imageSource == ImageSource.Stream_from_Folder:
            if self.multiImageTextButton.grid_info():
                self.multiImageTextButton.grid_forget()
            if self.multiImageFolderSelect.grid_info():
                self.multiImageFolderSelect.grid_forget()

        if self.camConfig.imageSource == ImageSource.Camera_Stream:

            self.startStreamOff()
            if not self.startStreamButton.grid_info():
                self.startStreamButton.grid(row=rowID, column=0, padx=5, pady=5, sticky='nsew')
            if not self.selectCameraCombo.grid_info():
                self.selectCameraCombo.grid(row=rowID, column=1, padx=5, pady=5, sticky='nsew')

        elif self.camConfig.imageSource == ImageSource.Static_Image:

            self.singleImageTextButton.configure(text=Path(self.camConfig.imageFilepath).name)
            if not self.singleImageFolderSelect.grid_info():
                self.singleImageFolderSelect.grid(row=1, column=0, padx=5, pady=5, sticky='nsew')
            if not self.singleImageTextButton.grid_info():
                self.singleImageTextButton.grid(row=1, column=1, padx=5, pady=5, sticky='nsew')

        elif self.camConfig.imageSource == ImageSource.Stream_from_Folder:

            self.multiImageTextButton.configure(text=Path(self.camConfig.imageFilepath).parent.name)
            if not self.multiImageFolderSelect.grid_info():
                self.multiImageFolderSelect.grid(row=1, column=0, padx=5, pady=5, sticky='nsew')
            if not self.multiImageTextButton.grid_info():
                self.multiImageTextButton.grid(row=1, column=1, padx=5, pady=5, sticky='nsew')
        else:
            raise ValueError(f'Unknown Image selection mode: {self.camConfig.imageSource}')

        self.saveToCache()

    def selectConfigFile(self):
        initDir = str(Path.cwd() / 'Configs')

        poss_file = filedialog.asksaveasfilename(
            initialdir=initDir,
            title="Select or create YAML config",
            defaultextension=".yaml",
            filetypes=[("YAML", "*.yaml"), ("All files", "*.*")],
            confirmoverwrite=False,  # <-- key line
        )
        if not poss_file:
            return

        self.camConfig.configFilepath = poss_file
        if os.path.exists(self.camConfig.configFilepath):
            with open(self.camConfig.configFilepath, 'r') as f:
                self.camConfig.fromDict(safe_load(f))
                self.update_post_newCamConfig()
        else:
            with open(self.camConfig.configFilepath, 'w') as f:
                dump(self.camConfig.toDict, f)

        self.configSelectLabel.configure(text=os.path.basename(self.camConfig.configFilepath))
        self._sync_flags_from_model()
        self.saveToCache()

    def selectImagesFilepath(self):
        if self.camConfig.imageFilepath is None:
            initDir = str(Path(self.default_filepath).parent)
        else:
            initDir = self.camConfig.imageFilepath  #os.path.normpath(self.camConfig.imageFilepath)

        poss_file = filedialog.askopenfilename(initialdir=initDir, title="Select Image")
        if poss_file != '':
            self.camConfig.imageFilepath = poss_file
            self.singleImageTextButton.configure(text=Path(self.camConfig.imageFilepath).name)
            self.multiImageTextButton.configure(text=Path(self.camConfig.imageFilepath).parent.name)
            self.saveToCache()

    def setupFrame(self):
        self.setup_camFrame()
        self.setup_configFrame()
        self.setup_exportFrame()
        self.setup_dataFrame()
        self.setup_playbackFrame()

    def setup_camFrame(self):
        rowID = 0

        self.streamOrImgCombo = CTkComboBox(self.cam_frame,
                                            values=['Camera Stream', 'Static Image', 'Stream from Folder'],
                                            command=self.sourceUpdate)
        self.streamOrImgCombo.grid(row=rowID, column=0, padx=5, pady=5, sticky='nsew')
        rowID += 1

        self.startStreamOff()

        self.singleImageTextButton.configure(command=self.startStreamOn)
        if self.camConfig.imageFilepath is not None:
            self.singleImageTextButton.configure(text=Path(self.camConfig.imageFilepath).name)

        self.multiImageTextButton.configure(command=self.startStreamOn)
        if self.camConfig.imageFilepath is not None:
            self.multiImageTextButton.configure(text=Path(self.camConfig.imageFilepath).parent.name)

        self.streamOrImgCombo.set(self.camConfig.imageSource.value)
        self.sourceUpdate(self.camConfig.imageSource.value)

        rowID += 1

        self.configSelectButton.grid(row=rowID, column=0, padx=5, pady=5, sticky='nsew')
        self.configSelectLabel.grid(row=rowID, column=1, padx=5, pady=5, sticky='nsew')

        rowID += 1

        selectFolderButton = CTkButton(self.cam_frame, text='Select Save Folder', command=self.selectFolder)
        selectFolderButton.grid(row=rowID, column=0, padx=5, pady=5, sticky='nsew')

        self.selectFolderLabel.grid(row=rowID, column=1, padx=5, pady=5, sticky='nsew')
        rowID += 1

        selectCalibButton = CTkButton(self.cam_frame, text='Select Calibration', command=self.loadCalibration)
        selectCalibButton.grid(row=rowID, column=0, padx=5, pady=5, sticky='nsew')

        self.selectCalibLabel = CTkLabel(self.cam_frame,
                                         text="../" + os.path.basename(os.path.normpath(self.camConfig.calibFilepath)))
        self.selectCalibLabel.grid(row=rowID, column=1, padx=5, pady=5, sticky='nsew')
        rowID += 1

        self.selectTruthPointsButton.grid(row=rowID, column=0, padx=5, pady=5, sticky='nsew')
        self.selectTruthPointsLabel.grid(row=rowID, column=1, padx=5, pady=5, sticky='nsew')
        rowID += 1

        self.selectYOLO_folderButton.grid(row=rowID, column=0, padx=5, pady=5, sticky='nsew')
        self.selectYOLO_folderLabel.grid(row=rowID, column=1, padx=5, pady=5, sticky='nsew')
        rowID += 1

        self.selectFlightLogButton.grid(row=rowID, column=0, padx=5, pady=5, sticky='nsew')
        self.selectFlightLogLabel.grid(row=rowID, column=1, padx=5, pady=5, sticky='nsew')
        rowID += 1

        aprilTagSizeEntryButton = CTkButton(self.cam_frame, text="Enter Size of April Tag (m)",
                                            command=self.setAprilTagSize)
        aprilTagSizeEntryButton.grid(row=rowID, column=0, padx=5, pady=5, sticky='nsew')

        self.aprilTagSizeEntry = CTkEntry(self.cam_frame, placeholder_text=str(self.camConfig.aprilTagSize))
        self.aprilTagSizeEntry.grid(row=rowID, column=1, padx=5, pady=5, sticky='nsew')

    def setup_configFrame(self):
        rowID = 0

        self.confSliderLabel.grid(row=rowID, column=0, padx=5, pady=5, sticky='nsew')
        self.confSliderBar.grid(row=rowID, column=1, padx=5, pady=5, sticky='nsew')
        rowID += 1

        self.iouSliderLabel.grid(row=rowID, column=0, padx=5, pady=5, sticky='nsew')
        self.iouSliderBar.grid(row=rowID, column=1, padx=5, pady=5, sticky='nsew')
        rowID += 1

        self.drawChessboardButton.grid(row=rowID, column=0, columnspan=1, padx=5, pady=5, sticky='nsew')

        if not self.calibration.validCal:
            self.undistortCheckbox.configure(state='disabled')

        self.undistortCheckbox.grid(row=rowID, column=1, columnspan=2, padx=5, pady=5, sticky='nsew')

        rowID += 1
        self.detectAprilTagsCheckbox.grid(row=rowID, column=0, padx=5, pady=5, sticky='nsew')
        self.hideAprilTagsCheckbox.grid(row=rowID, column=1, padx=5, pady=5, sticky='nsew')

        rowID += 1
        pnpLidarPoints = CTkCheckBox(self.config_frame, text='SolvePnP LiDAR Into Image',
                                     variable=self._flag_vars['pnpLidarPoints'])
        pnpLidarPoints.grid(row=rowID, column=0, columnspan=1, padx=5, pady=5, sticky='ew')

        qnpLidarPoints = CTkCheckBox(self.config_frame, text='SolveQnP LiDAR Into Image',
                                     variable=self._flag_vars['qnpLidarPoints'])
        qnpLidarPoints.grid(row=rowID, column=1, columnspan=1, padx=5, pady=5, sticky='ew')
        rowID += 1

        self.yoloInferenceCheckbox.grid(row=rowID, column=0, columnspan=1, padx=5, pady=5, sticky='ew')

        self.yoloBiasCheckbox.grid(row=rowID, column=1, columnspan=1, padx=5, pady=5, sticky='ew')

        rowID += 1
        # --- Pose from YOLO centers (multi-feature) ---
        # You can enable any combination (PnP / QnP / KF-weighted QnP).
        pnpYoloPoints = CTkCheckBox(self.config_frame, text='SolvePnP from YOLO',
                                    variable=self._flag_vars['pnpYoloPoints'])
        pnpYoloPoints.grid(row=rowID, column=0, columnspan=1, padx=5, pady=5, sticky='ew')

        qnpYoloPoints = CTkCheckBox(self.config_frame, text='SolveQnP from YOLO',
                                    variable=self._flag_vars['qnpYoloPoints'])
        qnpYoloPoints.grid(row=rowID, column=1, columnspan=1, padx=5, pady=5, sticky='ew')

        rowID += 1
        qnpKFYoloPoints = CTkCheckBox(self.config_frame, text='SolveQnP (KF-weighted) from YOLO',
                                      variable=self._flag_vars['qnpKFYoloPoints'])
        qnpKFYoloPoints.grid(row=rowID, column=0, columnspan=2, padx=5, pady=5, sticky='ew')

        rowID += 1

        detectCornersCheckbox = CTkCheckBox(self.config_frame, text='Detect Corners',
                                            variable=self._flag_vars['detect_corners'])
        detectCornersCheckbox.grid(row=rowID, column=0, columnspan=1, padx=5, pady=5, sticky='ew')

        self.detectHorizonCheckbox.grid(row=rowID, column=1, columnspan=1, padx=5, pady=5, sticky='ew')
        rowID += 1

        self.factorgraphCheckbox.grid(row=rowID, column=0, columnspan=1, padx=5, pady=5, sticky='ew')

        self.hyperfocusCheckbox.grid(row=rowID, column=1, columnspan=1, padx=5, pady=5, sticky='ew')

        rowID += 1

        self.phaseCorrelationCheckbox.grid(row=rowID, column=0, columnspan=1, padx=5, pady=5, sticky='ew')

        self.crosshairsCheckbox.grid(row=rowID, column=1, columnspan=1, padx=5, pady=5, sticky='ew')

        rowID += 1

        self.cubemapCheckbox.grid(row=rowID, column=0, columnspan=1, padx=5, pady=5, sticky='ew')

        self.hudCheckbox.grid(row=rowID, column=1, columnspan=1, padx=5, pady=5, sticky='ew')

        rowID += 1

        imageProcessingKernelLabel = CTkLabel(self.config_frame, text='Image Filter: ')
        imageProcessingKernelLabel.grid(row=rowID, column=0, padx=5, pady=5, sticky='ew')
        self.imageProcessingKernelCombobox = CTkComboBox(self.config_frame,
                                                         values=list(ImageKernel.__members__.keys()))
        self.imageProcessingKernelCombobox.set(self.camConfig.processingKernel.name)
        self.imageProcessingKernelCombobox.configure(command=self.updateImageProcessingKernel)
        self.updateImageProcessingKernel(self.camConfig.processingKernel.name)
        self.imageProcessingKernelCombobox.grid(row=rowID, column=1, columnspan=1, padx=5, pady=5, sticky='ew')

        rowID += 1

    def setup_exportFrame(self):
        rowID = 0
        self.recordOff()
        self.recordButton.grid(row=rowID, column=0, columnspan=1, padx=5, pady=5, sticky='ew')
        self.printButton.grid(row=rowID, column=1, columnspan=1, padx=5, pady=5, sticky='ew')
        rowID += 1

        activeEntryButton = CTkButton(self.export_frame, text="Time Between Saved Frames",
                                      command=self.getEntryValue)
        activeEntryButton.grid(row=rowID, column=0, padx=5, pady=5, sticky='nsew')

        self.timeBetweenImgsEntry = CTkEntry(self.export_frame,
                                             placeholder_text=str(self.camConfig.secondsBetweenImages))
        self.timeBetweenImgsEntry.grid(row=rowID, column=1, padx=5, pady=5, sticky='nsew')
        rowID += 1

        qualityLabel = CTkLabel(self.export_frame, text="Export Quality: ")
        qualityLabel.grid(row=rowID, column=0, padx=5, pady=5, sticky='ew')
        self.exportQualityCombo.grid(row=rowID, column=1, padx=5, pady=5, sticky='ew')
        rowID += 1

        self.exportToGifButton.grid(row=rowID, column=0, padx=5, pady=5, sticky='ew')
        self.exportToVidButton.grid(row=rowID, column=1, padx=5, pady=5, sticky='ew')

        rowID += 1

        self.exportStartFrame.grid(row=rowID, column=0, padx=5, pady=5, sticky='ew')
        self.exportEndFrame.grid(row=rowID, column=1, padx=5, pady=5, sticky='ew')

        rowID += 1

        self.btn_checkerboard.grid(row=rowID, column=1, padx=5, pady=5, sticky='ew')

        title = 'Folder Replay Hotkeys'
        items = [
            ("Space", "Pause / resume"),
            ("f", "Toggle Fixed-FPS ↔ Real-time"),
            ("c / z", "Step forward / backward one frame"),
            ("d / a", "Speed up / slow down playback"),
            ("r", "Reverse direction"),
            ("w", "Toggle overlays"),
            ("s / e", "Mark export start / end"),
            ("[ / ] , { / }", "Adjust time offset (small / large)"),
            ("; / ' , : / \"", "Adjust time offset (fine)"),
            ("p", "Persist time offset"),
            ("Esc", "Exit player"),
        ]

        CTkLabel(self.hotkey_frame, text=title, font=("Segoe UI", 16, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=12, pady=(12, 8)
        )

        # headings
        CTkLabel(self.hotkey_frame, text="Key", font=("Segoe UI", 13, "bold")).grid(
            row=1, column=0, sticky="w", padx=12, pady=(6, 2)
        )
        CTkLabel(self.hotkey_frame, text="Action", font=("Segoe UI", 13, "bold")).grid(
            row=1, column=1, sticky="w", padx=12, pady=(6, 2)
        )

        # rows
        for i, (key, desc) in enumerate(items, start=2):
            CTkLabel(self.hotkey_frame, text=key).grid(row=i, column=0, sticky="w", padx=12, pady=2)
            CTkLabel(self.hotkey_frame, text=desc, justify="left", wraplength=520).grid(
                row=i, column=1, sticky="w", padx=12, pady=2
            )

        # let text column expand
        self.hotkey_frame.grid_columnconfigure(0, weight=0)
        self.hotkey_frame.grid_columnconfigure(1, weight=1)



    def _on_checker_status(self, btn_state: str, btn_text: str) -> None:
        self.btn_checkerboard.configure(state=btn_state, text=btn_text)

    def _on_gpu_sample(self, sample: GpuSample) -> None:
        if sample.err:
        #     self.gpuLabel.configure(text=f"GPU: {sample.err}")
            return
        if sample.util is not None:
            self.gpu_slider.set(sample.util)
            # self.gpuLabel.configure(text=f"GPU: {sample.util}%  MEM: {sample.mem}%")

    def _on_toggle_show_gpu(self):
        enabled = bool(self._dp_gpu_var.get())  # authoritative
        self.camConfig.dp_gpu = enabled
        self.saveToCache()

        # UI state
        if hasattr(self, "gpu_slider") and self.gpu_slider is not None:
            self.gpu_slider.configure(state="normal" if enabled else "disabled")
            if not enabled:
                self.gpu_slider.set(0.0)

        # monitor lifecycle
        if self.gpu_monitor is None:
            self.gpu_monitor = GpuMonitor(
                scheduler=self,
                on_sample=self._on_gpu_sample,
                device_index=0,
                poll_ms=250,
            )
        self.gpu_monitor.set_enabled(enabled)

    def setup_dataFrame(self):
        """Build the 'Data Processing' page: folder pick, CSV pick, params, run."""
        f = self.data_frame
        for w in f.winfo_children():
            w.destroy()

        f.grid_rowconfigure(99, weight=1)
        f.grid_columnconfigure(1, weight=1)

        CTkLabel(f, text="Batch YOLO over image folder", font=("Segoe UI", 16, "bold")).grid(
            row=0, column=0, columnspan=3, padx=12, pady=(16, 8), sticky="w"
        )

        # --- Select image folder ---
        # --- Select image folder ---
        img_dir_default = (
                getattr(self.camConfig, "dp_img_dir", None)
                or self.camConfig.imageFilepath
                or ""
        )
        self._dp_img_dir_var = StringVar(value=str(img_dir_default))

        def _choose_dir():
            d = filedialog.askdirectory(title="Select image folder")
            if d:
                self._dp_img_dir_var.set(d)

        CTkLabel(f, text="Folder:").grid(row=1, column=0, padx=12, pady=6, sticky="w")
        CTkEntry(f, textvariable=self._dp_img_dir_var).grid(row=1, column=1, padx=12, pady=6, sticky="ew")
        CTkButton(f, text="Browse…", command=_choose_dir).grid(row=1, column=2, padx=12, pady=6)

        # --- Confidence sweep controls ---
        conf_default = getattr(self.camConfig, "dp_conf_list", "0.80")
        self._dp_conf_list = StringVar(value=str(conf_default))

        CTkLabel(f, text="YOLO conf values (comma-separated):").grid(
            row=3, column=0, padx=12, pady=6, sticky="w"
        )
        CTkEntry(f, textvariable=self._dp_conf_list).grid(
            row=3, column=1, padx=12, pady=6, sticky="ew"
        )

        # Small hint below the entry
        CTkLabel(
            f,
            text="Example: 0.50, 0.65, 0.80   (defaults to 0.80 on bad input)",
            font=("Segoe UI", 10, "italic")
        ).grid(
            row=4, column=0, columnspan=3, padx=12, pady=(0, 6), sticky="w"
        )

        # --- Checkpoint controls ---
        ckpt_default = getattr(self.camConfig, "dp_ckptN", 200)
        self._dp_ckptN = StringVar(value=str(ckpt_default))
        CTkLabel(f, text="Checkpoint every N images:").grid(row=5, column=0, padx=12, pady=6, sticky="w")
        CTkEntry(f, textvariable=self._dp_ckptN, width=100).grid(row=5, column=1, padx=12, pady=6, sticky="w")

        # --- Prefetch controls ---
        prefetch_default = getattr(self.camConfig, "dp_prefetch", 32)
        self._dp_prefetch = StringVar(value=str(prefetch_default))
        CTkLabel(f, text="Prefetch images (count):").grid(row=6, column=0, padx=12, pady=6, sticky="w")
        CTkEntry(f, textvariable=self._dp_prefetch, width=100).grid(row=6, column=1, padx=12, pady=6, sticky="w")

        # --- Progress UI ---
        self._dp_progress_label = CTkLabel(f, text="Idle")
        self._dp_progress_label.grid(row=20, column=0, columnspan=3, padx=12, pady=(8, 4), sticky="w")

        self._dp_progress = CTkProgressBar(f)  # determinate
        self._dp_progress.grid(row=21, column=0, columnspan=3, padx=12, pady=(0, 8), sticky="ew")
        self._dp_progress.set(0.0)

        gpu_display = bool(getattr(self.camConfig, "dp_gpu", False))
        # Keep a handle to the Tk var so config-load can resync the checkbox (no desync).
        self._dp_gpu_var = BooleanVar(value=gpu_display)
        gpu_checkbox = CTkCheckBox(
            f,
            text='Show GPU Util',
            variable=self._dp_gpu_var,
            command=self._on_toggle_show_gpu)
        gpu_checkbox.grid(row=25, column=0, columnspan=1, padx=5, pady=5, sticky='ew')

        self.gpu_slider = CTkSlider(f, from_=0, to=100)
        self.gpu_slider.grid(row=25, column=1, columnspan=2, padx=5, pady=5, sticky='ew')
        self.gpu_slider.configure(state='disabled')
        self.gpu_slider.set(0)

        def _bind_dp_str(var, attr_name):
            if var is None:
                return

            def _on_change(*_):
                try:
                    setattr(self.camConfig, attr_name, var.get())
                    self.saveToCache()
                except Exception:
                    # On parse error etc, just keep the text; _get_dp_conf_values()
                    # will fall back to [0.80] when it’s actually used.
                    pass

            var.trace_add("write", _on_change)

        _bind_dp_str(self._dp_img_dir_var, "dp_img_dir")
        _bind_dp_str(self._dp_conf_list, "dp_conf_list")
        _bind_dp_str(self._dp_ckptN, "dp_ckptN")
        _bind_dp_str(self._dp_prefetch, "dp_prefetch")

        # --- Batch action buttons row ---
        # Left: YOLO batch
        self._dp_run_btn = CTkButton(
            f,
            text="Run YOLO Batch",
            fg_color="#2FA572",
            command=self._run_yolo_batch_start,
        )
        self._dp_run_btn.grid(row=10, column=0, padx=12, pady=(16, 12), sticky="ew")

        self._dp_kalman_btn = CTkButton(
            f,
            text="Kalman Batch",
            command=self._run_kalman_batch_start,
        )
        self._dp_kalman_btn.grid(row=10, column=1, padx=12, pady=(16, 12), sticky="ew")

        # Center: SolvePnP/QnP batch (uses existing threaded worker)
        self._dp_pnp_btn = CTkButton(
            f,
            text="SolvePnP/QnP Batch",
            command=self.runPnP_QnP_on_folders_threaded,
        )
        self._dp_pnp_btn.grid(row=10, column=2, padx=12, pady=(16, 12), sticky="ew")

        # Cancel button in its own full-width row below
        self._dp_cancel_btn = CTkButton(
            f,
            text="Cancel",
            command=self._dp_cancel,
            state="disabled",  # until a run starts
        )
        self._dp_cancel_btn.grid(row=11, column=0, columnspan=1, padx=12, pady=(0, 12), sticky="ew")



        # Cancel button in its own full-width row below
        def plot_sequential():
            from support.viz.Plotting import Plotter
            if self.plotter is None:
                self.plotter = Plotter()
            vars = data.parse_conf_list(getattr(self, "_dp_conf_list", None))
            img_dir = Path(str(os.path.dirname(getattr(self.camConfig, "imageFilepath", "")) or "")) / "_ProcessedData"
            for var in vars:
                self.plotter.plot(var, img_dir, False, True)
                try:
                    self.winfo_exists()
                except:
                    self._plotter_close_plot_alias()
                    return

        dp_plotter_btn = CTkButton(
            f,
            text="Plot",
            command=plot_sequential,
        ).grid(row=11, column=1, columnspan=1, padx=12, pady=(0, 12), sticky="ew")

        dp_close_plot_btn = CTkButton(
            f,
            text="Close Plots",
            command=self._plotter_close_plot_alias,
        ).grid(row=11, column=2, columnspan=1, padx=12, pady=(0, 12), sticky="ew")


    def _plotter_close_plot_alias(self):
        if self.plotter is None:
            return False

        from support.viz.Plotting import Plotter
        self.plotter = Plotter()
        self.plotter.close_plot()
        return True

    def runPnP_QnP_on_folders_threaded(self):
        img_dir_str = (
                (getattr(self, "_dp_img_dir_var", None) and self._dp_img_dir_var.get().strip())
                or (getattr(self.camConfig, "imageFilepath", "") or "")
        )
        img_dir = Path(img_dir_str)

        if hasattr(self, "_dp_progress_label"):
            self._dp_progress_label.configure(text="Starting SolvePnP/QnP…")

        # ensure runner
        if not hasattr(self, "_dp_runner") or self._dp_runner is None:
            self._dp_runner = data.DataProcessorRunner()
        self._dp_runner.reset_cancel()
        self._dp_runner.cancel_event.clear()

        conf_list_var = getattr(self, "_dp_conf_list", None)

        def post_status(text: str):
            def _ui():
                if hasattr(self, "_dp_progress_label"):
                    self._dp_progress_label.configure(text=text)

            self.after(0, _ui)

        def post_progress(frac: float, text: str):
            def _ui():
                if hasattr(self, "_dp_progress"):
                    self._dp_progress.set(float(frac))
                if hasattr(self, "_dp_progress_label"):
                    self._dp_progress_label.configure(text=text)

            self.after(0, _ui)

        def _worker():
            try:
                self._dp_runner.run_pnp_qnp_conf_sweep(
                    img_dir=img_dir,
                    conf_list_var=conf_list_var,
                    run_pnp_qnp_from_detection_csv=self.run_pnp_qnp_from_detection_csv,
                    post_progress=post_progress,
                    post_status=post_status,
                    sweep_timer=utils.SweepTimer(),
                    fmt_mmss=utils._fmt_mmss,
                )
            except Exception as e:
                post_status(f"SolvePnP/QnP failed: {e}")

        threading.Thread(target=_worker, daemon=True).start()

    def _run_kalman_batch_start(self):
        if getattr(self, "_dp_worker", None) and self._dp_worker.is_alive():
            return

        # UI reset
        if hasattr(self, "_dp_progress_label"):
            self._dp_progress_label.configure(text="Starting KF…")
        if hasattr(self, "_dp_progress"):
            self._dp_progress.set(0.0)
        if hasattr(self, "_dp_run_btn"):
            self._dp_run_btn.configure(state="disabled")
        if hasattr(self, "_dp_cancel_btn"):
            self._dp_cancel_btn.configure(state="normal")

        if not hasattr(self, "_dp_runner") or self._dp_runner is None:
            self._dp_runner = data.DataProcessorRunner()
        self._dp_runner.reset_cancel()

        img_dir = Path(
            (getattr(self, "_dp_img_dir_var", None) and self._dp_img_dir_var.get().strip())
            or (getattr(self.camConfig, "imageFilepath", "") or "")
        )

        def progress_cb(frac: float, text: str) -> None:
            def _ui():
                if hasattr(self, "_dp_progress"):
                    self._dp_progress.set(float(frac))
                if hasattr(self, "_dp_progress_label"):
                    self._dp_progress_label.configure(text=text)

            self.after(0, _ui)

        def status_cb(text: str) -> None:
            def _ui():
                if hasattr(self, "_dp_progress_label"):
                    self._dp_progress_label.configure(text=text)

            self.after(0, _ui)

        def _finish(text: str) -> None:
            def _ui():
                if hasattr(self, "_dp_progress_label"):
                    self._dp_progress_label.configure(text=text)
                if hasattr(self, "_dp_run_btn"):
                    self._dp_run_btn.configure(state="normal")
                if hasattr(self, "_dp_cancel_btn"):
                    self._dp_cancel_btn.configure(state="normal")

            self.after(0, _ui)

        def _worker():
            try:
                if self.calibration is None or not getattr(self.calibration, "validCal", False):
                    _finish("No calibration loaded; cannot run KF.")
                    return

                self._dp_runner.run_kalman_conf_sweep(
                    img_dir=img_dir,
                    conf_list_var=getattr(self, "_dp_conf_list", None),
                    calibration=self.calibration,
                    progress_cb=progress_cb,
                    status_cb=status_cb,
                )
                _finish("KF sweep done.")
            except Exception as e:
                _finish(f"KF sweep failed: {e}")

        self._dp_worker = threading.Thread(target=_worker, daemon=True)
        self._dp_worker.start()

    def _dp_cancel(self):
        # 1) Signal cancel
        runner = getattr(self, "_dp_runner", None)
        if runner is not None:
            try:
                self._dp_runner.cancel_event.set()
            except Exception:
                pass

        # 2) UI feedback
        if hasattr(self, "_dp_progress_label"):
            self._dp_progress_label.configure(text="Canceling… (finishing current step)")
        if hasattr(self, "_dp_cancel_btn"):
            self._dp_cancel_btn.configure(state="disabled")  # prevent double-cancel spam
        if hasattr(self, "_dp_run_btn"):
            self._dp_run_btn.configure(state="disabled")  # optional, but usually correct

    def _run_yolo_batch_start(self):
        if getattr(self, "_dp_worker", None) and self._dp_worker.is_alive():
            return

        # UI reset
        if hasattr(self, "_dp_progress_label"):
            self._dp_progress_label.configure(text="Starting…")
        if hasattr(self, "_dp_progress"):
            self._dp_progress.set(0.0)
        if hasattr(self, "_dp_run_btn"):
            self._dp_run_btn.configure(state="disabled")
        if hasattr(self, "_dp_cancel_btn"):
            self._dp_cancel_btn.configure(state="normal")

        # ensure runner
        if not hasattr(self, "_dp_runner") or self._dp_runner is None:
            self._dp_runner = data.DataProcessorRunner()
        self._dp_runner.reset_cancel()

        # lazy import yolo
        from support.vision import yolo
        if self.yoloSession is None:
            self.yoloSession = yolo.YOLO()
            self.yoloSession.setNewFolder(self.camConfig.yoloFilepath)
            self.yoloSession.set_calibration(self.calibration)
            self.yoloSession.iou = self.camConfig.yolo_iou

        # build ids/times
        img_dir = Path(
            (getattr(self, "_dp_img_dir_var", None) and self._dp_img_dir_var.get().strip())
            or (getattr(self.camConfig, "imageFilepath", "") or "")
        )
        self.populate_idsTimes(str(img_dir))
        pairs = list(getattr(self.ImageTimeReader, "idsTimes", []))  # [(path, time), ...]

        # callbacks (UI thread)
        def post_progress(frac: float, text: str) -> None:
            def _ui():
                if hasattr(self, "_dp_progress"):
                    self._dp_progress.set(float(frac))
                if hasattr(self, "_dp_progress_label"):
                    self._dp_progress_label.configure(text=text)

            self.after(0, _ui)

        def post_status(text: str) -> None:
            def _ui():
                if hasattr(self, "_dp_progress_label"):
                    self._dp_progress_label.configure(text=text)

            self.after(0, _ui)

        def post_finish(text: str) -> None:
            def _ui():
                if hasattr(self, "_dp_progress_label"):
                    self._dp_progress_label.configure(text=text)
                if hasattr(self, "_dp_run_btn"):
                    self._dp_run_btn.configure(state="normal")
                if hasattr(self, "_dp_cancel_btn"):
                    self._dp_cancel_btn.configure(state="normal")

            self.after(0, _ui)

        # run worker
        def _worker():
            try:
                out_csv_base = img_dir / "_ProcessedData" / "1_yolo_detections.csv"
                params = data.YoloSweepParams(
                    img_dir=img_dir,
                    out_csv_base=out_csv_base,
                    conf_list_var=getattr(self, "_dp_conf_list", None),
                    ckpt_every_var=getattr(self, "_dp_ckptN", None),
                    prefetch_var=getattr(self, "_dp_prefetch", None),
                    cam_to_log_time_offset=float(getattr(self.camConfig, "cam_to_log_time_offset", 0.0)),
                )

                self._dp_runner.run_yolo_conf_sweep(
                    yolo_session=self.yoloSession,
                    calibration=self.calibration,
                    ids_times_pairs=pairs,
                    params=params,
                    sweep_timer=utils.SweepTimer(),
                    fmt_mmss=utils._fmt_mmss,
                    post_progress=post_progress,
                    post_status=post_status,
                    post_finish=post_finish,
                    undistort_points_px=undistort_points_px,
                )
            except Exception as e:
                post_finish(f"YOLO batch failed: {e}")

        self._dp_worker = threading.Thread(target=_worker, daemon=True)
        self._dp_worker.start()

    def run_pnp_qnp_from_detection_csv(
            self,
            csv_path: str,
            out_pnp: str | None = None,
            out_qnp: str | None = None,
            progress_cb=None,
            cancel_cb=False,
    ) -> None:
        """
        Thin GUI wrapper around support.io.data_processing.run_pnp_qnp_from_detection_csv.

        Keeps:
          - yoloSession/metaYolo linkage (truth_dict source)
          - GUI checkpoint field
          - GUI cancel_event
          - progress_cb forwarding
        """

        # ---- checkpoint cadence ----
        checkpoint_every = 0
        try:
            if hasattr(self, "_dp_ckptN"):
                v = self._dp_ckptN.get()
                if isinstance(v, str):
                    v = v.strip()
                checkpoint_every = int(v) if v else 0
        except Exception:
            checkpoint_every = 0

        if checkpoint_every <= 0:
            try:
                checkpoint_every = int(getattr(self.camConfig, "dp_ckptN", 0) or 0)
            except Exception:
                checkpoint_every = 0

        # ---- truth_dict from metaYolo via yoloSession ----
        # This keeps your “inextricably linked to metaYolo” requirement.
        if getattr(self, "yoloSession", None) is None:
            from support.vision import yolo
            self.yoloSession = yolo.YOLO()
            self.yoloSession.setNewFolder(self.camConfig.yoloFilepath)
            self.yoloSession.set_calibration(self.calibration)
            self.yoloSession.iou = self.camConfig.yolo_iou

        truth_dict = getattr(getattr(self.yoloSession, "reader", None), "idsNamesLocs", None)
        if truth_dict is None:
            raise ValueError("yoloSession.reader.idsNamesLocs is missing (metaYolo not loaded?).")

        # ---- cancel_event (runner owns it) ----
        cancel_event = None
        if hasattr(self, "_dp_runner") and getattr(self._dp_runner, "cancel_event", None) is not None:
            cancel_event = self._dp_runner.cancel_event

        # ---- delegate to module ----
        self._dp_runner.run_pnp_qnp_from_detection_csv(
            csv_path=str(csv_path),
            calibration=self.calibration,
            truth_dict=truth_dict,
            checkpoint_every=checkpoint_every,
            out_pnp=out_pnp,
            out_qnp=out_qnp,
            progress_cb=progress_cb,
            cancel_event=cancel_event,
        )

    def launch_checkerboard(self):
        self.checkerboard_launcher.toggle()

    def setup_playbackFrame(self):
        f = self.playback_frame

        for w in f.winfo_children():
            w.destroy()

        rowID = 0
        self.update_playbackMenu()

        playbackLabel = CTkLabel(f, textvariable=self.playbackModeText)
        playbackLabel.grid(row=rowID, column=0, sticky='w', padx=8, pady=(8, 4))
        rowID += 1

        # --- Frame slider ---
        self._pb_frame_text = StringVar(value="Frame: — / —")
        self._pb_slider_dragging = False

        self._pb_frame_label = CTkLabel(f, textvariable=self._pb_frame_text)
        self._pb_frame_label.grid(row=rowID, column=0, sticky="w", padx=8, pady=(4, 2))
        rowID += 1

        # Start with a safe dummy range; worker will update range once it knows num_images
        self._pb_slider = CTkSlider(
                        f,
                        from_ = 0,
                        to = 1,
                        number_of_steps = 1,
                        command = self._on_pb_slider_drag,  # live label only
                        )
        self._pb_slider.grid(row=rowID, column=0, sticky="ew", padx=8, pady=(0, 8))
        rowID += 1

        # Only seek on release (prevents seek spam while dragging)
        self._pb_slider.bind("<ButtonPress-1>", lambda *_: self._set_pb_slider_dragging(True))
        self._pb_slider.bind("<ButtonRelease-1>", self._on_pb_slider_release)

        f.grid_columnconfigure(0, weight=1)

        rowID += 1

        # --- Primary playback controls (mirror hotkeys) ---
        btn_frame = CTkFrame(self.playback_frame)
        btn_frame.grid(row=rowID, column=0, padx=5, pady=5, sticky="nsew")

        def mk(text, action, *args, col=0):
            b = CTkButton(
                btn_frame,
                text = text,
                command = lambda a=action, ar=args: self._enqueue_playback_cmd(a, *ar),)
            b.grid(row=0, column=col, padx=4, pady=4, sticky="nsew")
            return b

        # Order roughly like a transport bar
        mk("⟲ Rev (r)", "reverse", col=0)
        mk("⟸ Back (z)", "step_back", col=1)
        mk("⏯ Pause (space)", "toggle_pause", col=2)
        mk("Fwd (c) ⟹", "step_forward", col=3)
        mk("Mode (f)", "toggle_fps_mode", col=4)

        rowID += 1
        speed_frame = CTkFrame(self.playback_frame)
        speed_frame.grid(row=rowID, column=0, padx=5, pady=5, sticky="nsew")

        mk2 = lambda text, action, *args, col=0: CTkButton(
                speed_frame,
                text = text,
                command = lambda a=action, ar=args: self._enqueue_playback_cmd(a, *ar),
                ).grid(row=0, column=col, padx=4, pady=4, sticky="nsew")

        mk2("Slower (a)", "speed_down", col=0)
        mk2("Faster (d)", "speed_up", col=1)
        mk2("Overlays (w)", "toggle_overlays", col=2)
        mk2("Mark Start (s)", "mark_start", col=3)
        mk2("Mark End (e)", "mark_end", col=4)

    # --- Playback slider helpers ---
    def _set_pb_slider_dragging(self, dragging: bool):
        self._pb_slider_dragging = bool(dragging)

    def _on_pb_slider_drag(self, value):
        """ UI thread: user is dragging slider.
            We DO NOT seek here; we only update label."""
        try:
            v = int(round(float(value)))
        except Exception:
            return
        n = int(getattr(self, "_pb_num_images", 0) or 0)

        if n > 0:
            v = max(0, min(v, n - 1))
            self._pb_frame_text.set(f"Frame: {v} / {n - 1}")
        else:
            self._pb_frame_text.set(f"Frame: {v} / —")

    def _on_pb_slider_release(self, _evt=None):
        """ UI thread: mouse released -> enqueue ONE seek command. """

        self._set_pb_slider_dragging(False)
        try:
            v = int(round(float(self._pb_slider.get())))
        except Exception:
            return
        self._enqueue_playback_cmd("seek_idx", v)

    def _pb_ui_set_slider_range(self, n: int):
        """ UI thread: update slider bounds when worker learns the dataset length. """

        n = int(n)

        if n <= 1:
            self._pb_slider.configure(from_=0, to=1, number_of_steps=1)
            self._pb_frame_text.set("Frame: — / —")
            return

        self._pb_slider.configure(from_=0, to=n - 1, number_of_steps=n - 1)
        self._pb_frame_text.set(f"Frame: 0 / {n - 1}")

    def _pb_ui_set_slider_pos(self, idx: int, n: int):
        """
        UI thread: update slider position during playback.
        Avoid fighting the user while dragging.
        """

        if getattr(self, "_pb_slider_dragging", False):
            return

        if not hasattr(self, "_pb_slider") or self._pb_slider is None:
            return

        idx = int(max(0, min(int(idx), int(n) - 1)))
        self._pb_slider.set(idx)
        self._pb_frame_text.set(f"Frame: {idx} / {int(n) - 1}")

    # --- Playback command queue (UI thread safe) ---
    def _enqueue_playback_cmd(self, action: str, *args):
        """UI-thread safe: enqueue a playback action for the worker thread to apply."""
        with self._pb_cmd_lock:
            self._pb_cmds.append((action, args))

    def _drain_playback_cmds(self):
        """Worker-thread: pull all queued playback actions."""
        out = []
        with self._pb_cmd_lock:
            while self._pb_cmds:
                out.append(self._pb_cmds.popleft())
        return out

    # --- Single authoritative action dispatcher ---
    def _apply_playback_action(self, action: str, args, *, loader, curr_idx: int, t, wall_start: float):
        """
        Apply ONE playback action. This is the only place that is allowed to mutate:
           - curr_idx
           - wall_start
           - loader seek/stride
           - pauseCache clear
           - playback mode / speed changes
         Returns (curr_idx, wall_start).
         """

        num_images = len(t)

        if action == "toggle_fps_mode":
            self.pause = False
            self._on_toggle_fps_mode()
            wall_start = self._reanchor_on_mode_change(self.camConfig.playback_mode, curr_idx, t)
            self.update_playbackMenu()
            self.saveToCache()

        elif action == "step_forward":
            curr_idx = self._on_step_forward(curr_idx, num_images)
            self.pause = True
            self.pauseCache.clear()
            self.camConfig.playback_mode = PlaybackSpeed.Fixed_fps
            loader.seek(curr_idx, clear_buffer=True)
            self.update_playbackMenu()

        elif action == "step_back":
            curr_idx = self._on_step_back(curr_idx)
            self.pause = True
            self.pauseCache.clear()
            self.camConfig.playback_mode = PlaybackSpeed.Fixed_fps
            loader.seek(curr_idx, clear_buffer=True)
            self.update_playbackMenu()

        elif action == "toggle_pause":
            wall_start = self._on_toggle_pause(curr_idx, t, wall_start)
            self.update_playbackMenu()

        elif action == "speed_up":
            wall_start = self._on_speed_up(curr_idx=curr_idx, t=t)
            self.update_playbackMenu()

        elif action == "speed_down":
            wall_start = self._on_speed_down(curr_idx=curr_idx, t=t)
            self.update_playbackMenu()

        elif action == "toggle_overlays":
            self._on_toggle_overlays()

        elif action == "mark_start":
            self._on_mark_start(curr_idx)

        elif action == "mark_end":
            self._on_mark_end(curr_idx)

        elif action == "reverse":
            wall_start = self._on_reverse(loader=loader, curr_idx=curr_idx, t=t)
            self.update_playbackMenu()

        elif action == "seek_idx":
            # args: (target_idx, )
            (target_idx, ) = args
            target_idx = int(max(0, min(int(target_idx), len(t) - 1)))

            # Seek once, clear pause cache
            curr_idx = target_idx
            loader.seek(curr_idx, clear_buffer=True)
            self.pauseCache.clear()

            wall_start = self._reanchor_on_mode_change(self.camConfig.playback_mode, curr_idx, t)
            self.update_playbackMenu()

        # HUD bank offsets (optional buttons)
        elif action == "bank_minus":
            self.hud_marker.cam_bank_offset -= 0.1
        elif action == "bank_plus":
            self.hud_marker.cam_bank_offset += 0.1

        # Time offset adjustments (optional buttons)
        elif action == "offset":
        # args: (delta,)
            (delta,) = args
            self._on_adjust_offset(float(delta))
        elif action == "persist_offset":
            self._on_persist_offset()

        return curr_idx, wall_start

    def _key_to_playback_action(self, key: int):
        """Map a cv2.waitKey code to an action string + args."""

        if key == ord('f'):
            return ("toggle_fps_mode", ())
        if key == ord('c'):
            return ("step_forward", ())
        if key == ord('z'):
            return ("step_back", ())
        if key == ord(' '):
            return ("toggle_pause", ())
        if key == ord('d'):
            return ("speed_up", ())
        if key == ord('a'):
            return ("speed_down", ())
        if key == ord('w'):
            return ("toggle_overlays", ())
        if key == ord('s'):
            return ("mark_start", ())
        if key == ord('e'):
            return ("mark_end", ())
        if key == ord('r'):
            return ("reverse", ())
        if key == ord('b'):
            return ("bank_minus", ())
        if key == ord('n'):
            return ("bank_plus", ())

        # offset hotkeys
        if key == ord(";"):
            return ("offset", (-0.01,))
        if key == ord("'"):
            return ("offset", (+0.01,))
        if key == ord(':'):
            return ("offset", (-0.10,))
        if key == ord('"'):
            return ("offset", (+0.10,))
        if key == ord('['):
            return ("offset", (-1.00,))
        if key == ord(']'):
            return ("offset", (+1.00,))

        if key == ord('{'):
            return ("offset", (-10.00,))
        if key == ord('}'):
            return ("offset", (+10.00,))
        if key == ord('p'):
            return ("persist_offset", ())
        return (None, None)

    def setAprilTagSize(self):

        try:
            self.camConfig.aprilTagSize = float(self.aprilTagSizeEntry.get())
        except ValueError:
            self.aprilTagSizeEntry.delete(0, END)
            self.aprilTagSizeEntry.configure(placeholder_text=str(self.camConfig.aprilTagSize), )

    def getEntryValue(self):
        try:
            self.camConfig.secondsBetweenImages = float(self.timeBetweenImgsEntry.get())
        except ValueError:
            self.camConfig.secondsBetweenImages = 1.0
            self.timeBetweenImgsEntry.delete(0, END)
            self.timeBetweenImgsEntry.configure(placeholder_text='1')
        if self.camConfig.secondsBetweenImages <= 0.0:
            self.timeBetweenImgsEntry.delete(0, END)
            self.timeBetweenImgsEntry.configure(placeholder_text='1')
            self.camConfig.secondsBetweenImages = 1.0

    def _gather_annotated_frames(self) -> list[np.ndarray]:
        directory = Path(self.camConfig.imageFilepath).parent
        self.populate_idsTimes(str(directory))

        paths = []
        for rec in self.ImageTimeReader.idsTimes:
            p = Path(rec[0])
            paths.append(p if p.is_absolute() else (directory / p))

        try:
            import pandas as pd
            offset_dict = pd.read_csv(directory / '__TIME_OFFSET.csv')
            self.camConfig.cam_to_log_time_offset = float(offset_dict['offset'][0])
        except FileNotFoundError:
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

    def exportToGif(self):
        if self.making_gifOrVid:
            return

        self.exportToGifButton.configure(text="Making gif...", state='disabled', fg_color="blue")
        self.exportToVidButton.configure(text="Making gif...", state='disabled', fg_color="blue")
        self.making_gifOrVid = True

        t = threading.Thread(target=self.exportToGif_worker, daemon=True)
        t.start()

    def exportToVid(self):
        if self.making_gifOrVid:
            return

        self.exportToGifButton.configure(text="Making vid...", state='disabled', fg_color="blue")
        self.exportToVidButton.configure(text="Making vid...", state='disabled', fg_color="blue")
        self.making_gifOrVid = True

        t = threading.Thread(target=self.exportToVid_worker, daemon=True)
        t.start()

    def exportToGif_worker(self):
        try:
            from support.io.convert_to_gif import make_gif
            frames = self._gather_annotated_frames()
            make_gif(frames, 10, infinite=True, quality=self.camConfig.export_quality)
        finally:
            self.after(0, self._exportToGifOrVid_done)

    def exportToVid_worker(self):
        try:
            frames = self._gather_annotated_frames()
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
            out = cv2.VideoWriter('output_video.mp4', fourcc, 10, (w, h))
            for f in frames:
                out.write(f)
            out.release()
        finally:
            self.after(0, self._exportToGifOrVid_done)

    def _exportToGifOrVid_done(self):
        self.exportToGifButton.configure(text="Export to GIF", state='normal', fg_color=clr.CTK_GREEN)
        self.exportToVidButton.configure(text="Export to Vid", state='normal', fg_color=clr.CTK_GREEN)
        self.making_gifOrVid = False

    def startStreamToggle(self):
        if self._thread is None or not self._thread.is_alive():  # thread not running
            self.startStreamOn()
            return True

        self.startStreamOffBool()
        return False

    def startStreamOn(self):
        self.showWindow = True
        self.singleImageTextButton.configure(command=self.startStreamOffBool, text='Stop Displaying',
                                             fg_color=clr.CTK_GREEN,
                                             hover_color='navy')
        self.startStreamButton.configure(command=self.startStreamOffBool, text='Stop Streaming', fg_color=clr.CTK_GREEN,
                                         hover_color='navy')
        self.multiImageTextButton.configure(command=self.startStreamOffBool, fg_color=clr.CTK_GREEN, hover_color='navy')

        self.selectCameraCombo.configure(state='disabled')

        self.streamOrImgCombo.configure(state='disabled')

        self.threadStopper = utils.ThreadStopper()
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def startStreamOffBool(self):
        self.showWindow = False

    def startStreamOff(self):
        cv2.waitKey(1)

        self.threadStopper.set()
        try:
            cv2.destroyWindow(self.windowName)
        except cv2.error as e:
            pass  # Window not yet open

        if self.vc is not None and self.vc.isOpened():
            self.vc.release()
            self.vc = None

        # Only join if we're on a different thread than the worker.
        if self._thread and self._thread.is_alive() and threading.current_thread() != self._thread:
            self._thread.join(timeout=1.0)
        self._thread = None

        if not self.shutting_down:
            if self.camConfig.imageFilepath is not None:
                self.singleImageTextButton.configure(
                    command=self.startStreamOn, fg_color=clr.CTK_BUTTON_RED, hover_color='blue',
                    text=os.path.basename(self.camConfig.imageFilepath)
                )
            self.startStreamButton.configure(command=self.startStreamOn, fg_color=clr.CTK_BUTTON_RED, hover_color='blue')
            self.multiImageTextButton.configure(command=self.startStreamOn, fg_color=clr.CTK_BUTTON_RED, hover_color='blue')

            self.selectCameraCombo.configure(state='normal')
            self.startStreamButton.configure(text='Start Stream')
            self.streamOrImgCombo.configure(state='normal')

            self.showWindow = False

        try:
            cv2.destroyWindow(self.windowName)
        except cv2.error:
            pass  # window not yet open

    def recordOn(self):
        self.recordButton.configure(fg_color='green', text='Saving Imagery', hover_color='navy', command=self.recordOff)
        self.recording = True

    def recordOff(self):
        self.recordButton.configure(fg_color=clr.CTK_BUTTON_RED, text=f'Saved Imagery: #{self.img_idx}', hover_color='blue',
                                    command=self.recordOn)
        self.recording = False

    def printLidarOnce(self):
        self.printLidar = True

    def updateImageProcessingKernel(self, newValue):
        if self.camConfig.processingKernel == ImageKernel.Gabor and self.GaborGUI is not None:
            self.GaborGUI.close()
        self.camConfig.processingKernel = ImageKernel(newValue)
        if self.camConfig.processingKernel == ImageKernel.Unfiltered:
            self.imageProcessingKernelCombobox.configure(fg_color='#343638', text_color='#DCE4EE')
        else:
            self.imageProcessingKernelCombobox.configure(fg_color='yellow', text_color='black')

        self.saveToCache()

    def createDetector(self):
        if self.detector is None:
            self.arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
            self.arucoParams = cv2.aruco.DetectorParameters()
            # self.arucoParams.adaptiveThreshWinSizeMin = 5
            # self.arucoParams.adaptiveThreshWinSizeMax = 35
            # self.arucoParams.adaptiveThreshWinSizeStep = 5
            # self.arucoParams.minMarkerPerimeterRate = 0.02  # or higher if tags are big
            # self.arucoParams.maxMarkerPerimeterRate = 1.0
            # self.arucoParams.cornerRefinementMinAccuracy = 0.1  # or 0.2
            # self.arucoParams.cornerRefinementMaxIterations = 20
        self.detector = cv2.aruco.ArucoDetector(self.arucoDict, self.arucoParams)

    def run_detectSingleImage(self):
        cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)
        frame = cv2.imread(str(Path(self.camConfig.imageFilepath)))
        while (not self.threadStopper.is_set()
               and cv2.getWindowProperty(self.windowName, cv2.WND_PROP_VISIBLE) > 0
               and self.showWindow):

            self.analyze_image(frame)

            key = cv2.waitKey(1)
            if key == 27:
                self.threadStopper.set()
                break

        try:
            cv2.destroyWindow(self.windowName)
        except cv2.error as e:
            pass
        self.after(0, self._on_worker_exit)

    @staticmethod
    def convert_cv_to_pil(img):
        from PIL.Image import fromarray
        return fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def run(self):

        if self.camConfig.imageSource == ImageSource.Camera_Stream:
            self.run_video_stream()
        elif self.camConfig.imageSource == ImageSource.Stream_from_Folder:
            self.run_folder_reader()
        elif self.camConfig.imageSource == ImageSource.Static_Image:
            self.run_detectSingleImage()

    def run_video_stream(self):

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
               cv2.getWindowProperty(self.windowName, cv2.WND_PROP_VISIBLE) > 0 and
               self.showWindow and not self.making_gifOrVid):
            rval, frame = self.vc.read()

            if stop_display_time is not None:
                self._draw_chessboard_state(frame)

            self.analyze_image(frame)
            key = cv2.waitKey(1)

            if key == 27:  # exit on ESC
                self.threadStopper.set()
                break

            if self._flag_vars['draw_chessboard'].get():
                new_time = self._handle_chessboard_hotkeys(key)
                if new_time is not None:
                    stop_display_time = new_time

            if stop_display_time is not None and time.monotonic() > stop_display_time:
                stop_display_time = None
                print('Time out')

        # Minimal teardown in the worker; the UI thread will handle buttons/state.
        if self.vc is not None and self.vc.isOpened():
            self.vc.release()
            self.vc = None

        try:
            cv2.destroyWindow(self.windowName)
        except cv2.error:
            pass

        self.after(0, self._on_worker_exit)
        return

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
    def _stride_for_speed(speed_abs: int) -> int:
        s = max(0, int(speed_abs))
        if s == 0:
            return 0
        if s == 1:
            return 1
        return min(8, s)

    def populate_idsTimes(self, directory):
        d = Path(directory)
        # load logs (keep API: list[str])
        log_list = [str(p) for p in d.glob("*.log")]
        if not self.ImageTimeReader.loadLog(log_list):
            self.ImageTimeReader.idsTimes = []
            image_list = list(d.glob("*.bmp")) + list(d.glob("*.png"))
            # natural_sort expects strings:
            image_list = data.natural_sort([str(p) for p in image_list])
            for image in image_list:
                self.ImageTimeReader.idsTimes.append([image, None])

    @staticmethod
    def _poll_keys(max_ms: int = 8) -> list[int]:
        # One-shot poll: wait up to max_ms for a key
        k = cv2.waitKey(max_ms) & 0xFF
        if k not in (0, 0xFF, 255, -1):
            return [k]
        return []

    @staticmethod
    def load_time_offset(directory):
        try:
            import pandas as pd
            offset_dict = pd.read_csv(directory / "__TIME_OFFSET.csv")
            return float(offset_dict['offset'][0])
        except FileNotFoundError:
            return 0.0

    def _build_sequence_and_timebase(self, directory):
        self.populate_idsTimes(str(directory))

        paths = []
        for rec in self.ImageTimeReader.idsTimes:
            p = Path(rec[0])
            paths.append(p if p.is_absolute() else (directory / p))

        ts_raw = []
        for name, ts in self.ImageTimeReader.idsTimes:
            ts_raw.append(None if ts is None else float(ts) + float(self.camConfig.cam_to_log_time_offset))

        t = self._make_timebase(ts_raw, self.camConfig.target_fps, len(paths))

        return paths, t

    def run_folder_reader_profiled(self):
        import cProfile
        import pstats

        if not self.profile_run_folder:
            # Normal behavior
            return self.run_folder_reader()

        prof = cProfile.Profile()
        try:
            prof.enable()
            self.run_folder_reader()
        finally:
            prof.disable()
            prof.dump_stats("run_folder_reader.prof")

            stats = pstats.Stats(prof).strip_dirs().sort_stats("cumtime")

            print("\n=== Top 40 functions overall (cumtime) ===")
            stats.print_stats(40)

            print("\n=== superCalibrateCamera functions ===")
            stats.print_stats("superCalibrateCamera")

            print("\n=== run_folder_reader / analyze_image ===")
            stats.print_stats("run_folder_reader")
            stats.print_stats("analyze_image")

            print("\n=== Top 40 functions overall (cumtime) ===")
            stats.print_stats(40)

            # Narrow view: only functions from your GUI modules
            print("\n=== GUI-ish functions (superCalibrateCamera) ===")
            stats.print_stats("superCalibrateCamera")

            print("\n=== CustomTkinter / Tk wrappers ===")
            stats.print_stats("customtkinter")
            stats.print_stats("ctk")
            stats.print_stats("tkinter")

    def run_folder_reader(self):
        try:
            cv2.destroyWindow(self.windowName)
        except cv2.error:
            pass
        cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)

        directory = Path(self.camConfig.imageFilepath).parent

        paths, t = self._build_sequence_and_timebase(directory)

        num_images = len(paths)

        self.playback.speed = 1  # negative=rewind, 0=freeze, positive=forward
        self.pause = False
        self.last_nonzero_sign = 1
        last_speed = self.playback.speed
        curr_idx = 0

        # ---- Playback slider: initialize range once we know dataset length ----
        if not getattr(self, "_pb_slider_range_inited", False):
            self._pb_slider_range_inited = True
            self._pb_num_images = int(num_images)

            # UI-thread update
            try:
                self.after(0, self._pb_ui_set_slider_range, int(num_images))
                self.after(0, self._pb_ui_set_slider_pos, int(curr_idx), int(num_images))
            except Exception:
                pass

        # --- PAUSED CACHE: keep 1 frame while paused to avoid refetch spam ---
        self.pauseCache.clear()

        # time offset
        self.camConfig.cam_to_log_time_offset = self.load_time_offset(directory)

        from support.runtime.buffer_image_loader import BufferedImageLoader as imgBuf

        # --- start background loader ---
        loader = imgBuf(
            filepaths=[str(p) for p in paths],
            max_buffer=96,
            preprocess=None,
            start_index=0,
            loop=True,
            read_flags=cv2.IMREAD_COLOR,
        ).start()

        # one-shot key handling
        pending_keys = []
        last_edge_time: dict[int, float] = {}

        wall_start = time.monotonic()

        def sleep_until(deadline) -> list[int]:
            keys = []
            while True:
                remain = deadline - time.monotonic()
                if remain <= 0:
                    break
                slice_ms = int(min(8, max(1, remain * 1000)))
                keys.extend(self._poll_keys(slice_ms))
            return keys

        try:
            while (not self.threadStopper.is_set()
                   and cv2.getWindowProperty(self.windowName, cv2.WND_PROP_VISIBLE)
                   and self.showWindow
                   and not self.making_gifOrVid):

                curr_time = time.time()
                if (curr_time - self.fps_time_log) > 0.000001:
                    self.curr_fps = 1.0 / abs(curr_time - self.fps_time_log)
                else:
                    self.curr_fps = 0.0
                self.fps_time_log = curr_time

                # ===== react to speed changes (incl. direction) =====
                if self.playback.speed != last_speed:
                    s_abs = self._stride_for_speed(abs(self.playback.speed))

                    if self.playback.speed != 0:
                        self.last_nonzero_sign = (1 if self.playback.speed > 0 else -1)

                    if s_abs == 0:
                        self.pause = True
                    else:
                        self.pause = False
                        signed_stride = self.last_nonzero_sign * s_abs
                        if signed_stride != self.playback.stride:
                            loader.set_stride(signed_stride)
                            self.playback.stride = signed_stride

                        curr_idx = max(0, min(curr_idx, num_images - 1))
                        loader.seek(curr_idx, clear_buffer=True)
                        self.pauseCache.clear()

                    last_speed = self.playback.speed

                # ===== fetch a frame =====
                if not self.pause:
                    # streaming mode: loader drives index
                    got = loader.get_next(timeout=0.02)

                    # if skipping (|stride|>1), drain extras so we show freshest
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
                        curr_idx = got_idx

                    # leaving pause → invalidate paused cache
                    self.pauseCache.clear()

                else:
                    # ===== PAUSED MODE with CACHE =====
                    target_idx = max(0, min(curr_idx, num_images - 1))

                    # If cache is invalid or user moved (z/c), fetch once; otherwise reuse cached frame
                    if self.pauseCache.frame is None or self.pauseCache.idx != target_idx:
                        # Seek ONCE; do NOT keep seeking every loop
                        loader.seek(target_idx, clear_buffer=True)

                        got = loader.get_next(timeout=0.5)  # give worker a bit more time while paused
                        if got is not None:
                            got_idx, frame = got
                            self.pauseCache.set(got_idx, frame)
                            curr_idx = got_idx
                        else:
                            # If file truly missing, print once; otherwise keep last cached frame (if any)
                            p = paths[target_idx]
                            if not Path(p).exists():
                                self.pauseCache.clear()
                            # If file exists but frame not ready yet, DON'T print; keep previous cached frame
                    frame = self.pauseCache.frame

                # ===== display / HUD =====
                if frame is not None and Path(paths[curr_idx]).exists() and len(self.ImageTimeReader.idsTimes) > 0:

                    if self.camConfig.playback_mode == PlaybackSpeed.Fixed_fps and not self.pause:
                        period = 1.0 / self.camConfig.target_fps
                        target_time = wall_start + period * (
                            curr_idx if not self.last_nonzero_sign < 0 else num_images - curr_idx)
                        if target_time < time.monotonic():
                            wall_start = time.monotonic() - 1.0 / max(0.001, self.camConfig.target_fps) * (
                                curr_idx if not self.last_nonzero_sign < 0 else num_images - curr_idx)
                        pending_keys.extend(sleep_until(target_time))

                    elif self.camConfig.playback_mode == PlaybackSpeed.Real_time and not self.pause:
                        rs = float(self.camConfig.rt_speed) or 1e-6
                        elapsed = (time.monotonic() - wall_start) * rs
                        elapsed_ref = (t[-1] - elapsed) if self.last_nonzero_sign < 0 else elapsed

                        if elapsed_ref < t[0]:
                            idx_target = num_images - 1
                            wall_start = time.monotonic() - (
                                (t[idx_target] - t[0]) / rs if not self.last_nonzero_sign < 0
                                else ((t[-1] - t[idx_target]) / rs))
                        elif elapsed_ref > t[-1]:
                            idx_target = 0
                            wall_start = time.monotonic() - (
                                (t[idx_target] - t[0]) / rs if not self.last_nonzero_sign < 0
                                else ((t[-1] - t[idx_target]) / rs))
                        else:
                            idx_target = int(np.searchsorted(t, elapsed_ref, side='right') - 1)

                        idx_target = max(0, min(idx_target, num_images - 1))
                        if idx_target != curr_idx:
                            loader.seek(idx_target, clear_buffer=True)
                            got = loader.get_next(timeout=0.02)
                            if got is not None:
                                curr_idx, frame = got

                        pending_keys.extend(self._poll_keys(1))

                    ts = self.ImageTimeReader.idsTimes[curr_idx][1]
                    boxAround = False
                    if self.camConfig.start_export_idx <= curr_idx <= self.camConfig.end_export_idx:
                        boxAround = True
                    if ts is None:
                        self.analyze_image(frame, None, self.ImageTimeReader.idsTimes[curr_idx][0],
                                           box_around=boxAround)
                    else:
                        self.analyze_image(frame, ts + self.camConfig.cam_to_log_time_offset,
                                           self.ImageTimeReader.idsTimes[curr_idx][0], box_around=boxAround)
                elif frame is None:
                    # Nothing to draw this iteration; just keep window responsive
                    pass

                pending_keys.extend(self._poll_keys(1))

                for (action, args) in self._drain_playback_cmds():
                    curr_idx, wall_start = self._apply_playback_action(
                        action, args, loader=loader, curr_idx=curr_idx, t=t, wall_start=wall_start
                    )
                while pending_keys:
                    key = pending_keys.pop(0)
                    if key == 27:  # ESC
                        self.threadStopper.set()
                        break

                    action, args = self._key_to_playback_action(key)
                    if action is not None:
                        curr_idx, wall_start = self._apply_playback_action(
                            action, args, loader=loader, curr_idx=curr_idx, t=t, wall_start=wall_start
                        )

                    while self.making_gifOrVid:
                        time.sleep(0.1)

                # ---- Playback slider: publish position when idx changes ----
                if getattr(self, "_pb_num_images", 0):
                    if getattr(self, "_pb_last_sent_idx", None) != curr_idx:
                        self._pb_last_sent_idx = curr_idx
                        try:
                            self.after(0, self._pb_ui_set_slider_pos, int(curr_idx), int(num_images))
                        except Exception:
                            pass

                pending_keys.extend(self._poll_keys(1))
                if cv2.getWindowProperty(self.windowName, cv2.WND_PROP_VISIBLE) <= 0:
                    self.threadStopper.set()
                    break

        finally:
            try:
                cv2.destroyWindow(self.windowName)
            except cv2.error:
                pass
            self.after(0, self._on_worker_exit)
            loader.stop()

    def update_playbackMenu(self):
        if self.camConfig.playback_mode == PlaybackSpeed.Fixed_fps:
            self.playbackModeText.set(
                value=f"Playback Mode: FPS\nTarget FPS: {self.camConfig.target_fps:.2f}\n{'Pause' if self.pause else 'Rewind' if self.playback.speed < 0 else 'Play'}")
        else:
            self.playbackModeText.set(value=f'Playback Mode: Realtime\nPlayback Speed: {self.camConfig.rt_speed:.2f}')

    def _reanchor_on_mode_change(self, new_mode, curr_idx: int, t) -> float:
        """
        Re-anchor wall_start so the current frame stays fixed when switching modes,
        including while reversing. Uses time.monotonic() to match the main loop.
        """
        now = time.monotonic()

        if new_mode == PlaybackSpeed.Real_time:
            # Map wall clock to log time (direction-aware)
            rs = max(1e-6, float(self.camConfig.rt_speed))
            t0, tN = t[0], t[-1]
            if self.last_nonzero_sign < 0:
                # reverse: tN - (now - wall_start)*rs == t[curr_idx]
                return now - (tN - t[curr_idx]) / rs
            else:
                # forward: (now - wall_start)*rs == t[curr_idx] - t0
                return now - (t[curr_idx] - t0) / rs

        else:
            # Fixed-FPS: anchor to the correct phase for direction
            fps = max(0.001, float(self.camConfig.target_fps))
            num_images = len(t)
            phase = (num_images - curr_idx) if self.last_nonzero_sign < 0 else curr_idx
            return now - (phase / fps)

    @staticmethod
    def _rt_reanchor(now: float, curr_idx: int, t, rt_rate: float, sign: int) -> float:
        """Return a new wall_start so that the effective RT timeline still maps to t[curr_idx]."""
        rt_rate = max(1e-6, float(rt_rate))
        t0, tN = t[0], t[-1]
        if sign < 0:
            # reverse: tN - (now - wall_start)*rt_rate == t[curr_idx]
            return now - (tN - t[curr_idx]) / rt_rate
        else:
            # forward: (now - wall_start)*rt_rate == t[curr_idx] - t0
            return now - (t[curr_idx] - t0) / rt_rate

    def _on_reverse(self, loader, curr_idx: int, t):
        """
        Toggle playback direction without jumping the current frame.
        Returns: (new_wall_start)
        """
        # New direction (+1 forward, -1 reverse)
        self.last_nonzero_sign = -1 if self.last_nonzero_sign > 0 else 1

        # If actively playing, flip speed sign and update stride, then realign buffer at current index.
        if self.playback.speed != 0:
            self.playback.speed = -self.playback.speed
            s_abs = self._stride_for_speed(abs(self.playback.speed))
            if s_abs > 0:
                loader.set_stride(self.last_nonzero_sign * s_abs)
                loader.seek(curr_idx, clear_buffer=True)

        now = time.monotonic()

        if getattr(self.camConfig, "playback_mode", None) == PlaybackSpeed.Real_time:
            # --- Real-time: direction-aware re-anchor on the log timeline ---
            rs = max(1e-6, float(self.camConfig.rt_speed))
            t0, tN = t[0], t[-1]
            if self.last_nonzero_sign < 0:
                # reverse:  tN - (now - wall_start)*rs == t[curr_idx]
                wall_start = now - (tN - t[curr_idx]) / rs
            else:
                # forward:  (now - wall_start)*rs == t[curr_idx] - t0
                wall_start = now - (t[curr_idx] - t0) / rs
        else:
            # --- Fixed-FPS: anchor MUST match the phase used in the loop ---
            # In forward, phase = curr_idx; in reverse, phase = num_images - curr_idx.
            fps = max(0.001, float(self.camConfig.target_fps))
            period = 1.0 / fps
            num_images = len(t)  # or use your existing num_images variable if already in scope
            phase = (num_images - curr_idx) if self.last_nonzero_sign < 0 else curr_idx
            wall_start = now - period * phase

        return wall_start

    @staticmethod
    def _make_timebase(ts_raw, fallback_fps, n):
        t = np.array([np.nan if v is None else float(v) for v in ts_raw], dtype='float64')
        if n == 0:  # <-- guard
            return t
        if np.all(np.isnan(t)):
            step = 1.0 / max(1e-6, float(fallback_fps))
            t = np.arange(n, dtype='float64') * step
        else:
            nans = np.isnan(t)
            if nans.any():
                notn = ~nans
                t[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(notn), t[notn])
        t -= float(t[0])
        return t

    def _on_worker_exit(self):
        # Mark no live worker and reset run-state
        self._thread = None
        self.showWindow = False

        # Rewire buttons back to "start"
        if self.camConfig.imageFilepath is not None:
            self.singleImageTextButton.configure(
                command=self.startStreamOn,
                fg_color=clr.CTK_BUTTON_RED, hover_color='blue',
                text=os.path.basename(self.camConfig.imageFilepath)
            )
        else:
            self.singleImageTextButton.configure(
                command=self.startStreamOn,
                fg_color=clr.CTK_BUTTON_RED, hover_color='blue',
                text='No Image Selected'
            )

        self.startStreamButton.configure(
            command=self.startStreamOn,
            fg_color=clr.CTK_BUTTON_RED, hover_color='blue',
            text='Start Stream'
        )
        self.multiImageTextButton.configure(
            command=self.startStreamOn,
            fg_color=clr.CTK_BUTTON_RED, hover_color='blue'
        )

        # Re-enable selectors
        self.selectCameraCombo.configure(state='normal')
        self.streamOrImgCombo.configure(state='normal')

    # --- Key action helpers (CameraGui) ---

    def _on_toggle_fps_mode(self):
        """Swap Fixed_fps <-> Real_time, preserving perceived position."""
        self.camConfig.playback_mode = self.camConfig.playback_mode.next()
        if self.camConfig.playback_mode == PlaybackSpeed.Real_time:
            self.camConfig.rt_speed = 1.0

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
        playing = (self.playback.speed != 0)
        if playing:
            self._resume_speed_mag = max(1.0, abs(self.playback.speed))
            self.playback.speed = 0.0
            return wall_start

        prev_mag = getattr(self, "_resume_speed_mag", 1.0)
        self.playback.speed = float(self.last_nonzero_sign or 1) * prev_mag

        now = time.monotonic()
        if getattr(self.camConfig, "playback_mode", None) == PlaybackSpeed.Real_time:
            rs = max(1e-6, float(self.camConfig.rt_speed))
            t0, tN = t[0], t[-1]
            if self.last_nonzero_sign < 0:
                wall_start = now - (tN - t[curr_idx]) / rs
            else:
                wall_start = now - (t[curr_idx] - t0) / rs
        else:
            fps = max(0.001, float(self.camConfig.target_fps))
            phase = (len(t) - curr_idx) if self.last_nonzero_sign < 0 else curr_idx
            wall_start = time.monotonic() - (phase / fps)
        return wall_start

    def _on_speed_up(self, curr_idx: int, t) -> float:
        """
        Increase playback speed.
        - RT mode: multiply rt_speed, then re-anchor so current frame stays put.
        - Fixed-FPS: increase target_fps, then re-anchor to current frame index.
        Returns new wall_start.
        """
        if getattr(self.camConfig, "playback_mode", None) == PlaybackSpeed.Real_time:
            # adjust rate
            prev_rt = self.camConfig.rt_speed
            self.camConfig.rt_speed = min(float(self.camConfig.rt_speed) * SPEED_STEP, 128.0)
            if prev_rt < 0.99 and self.camConfig.rt_speed > 1.0:
                self.camConfig.rt_speed = 1.0
            # direction-aware reanchor
            now = time.monotonic()
            rs = max(1e-6, float(self.camConfig.rt_speed))
            t0, tN = t[0], t[-1]
            if self.last_nonzero_sign < 0:
                # reverse: tN - (now - wall_start)*rs == t[curr_idx]
                wall_start = now - (tN - t[curr_idx]) / rs
            else:
                # forward: (now - wall_start)*rs == t[curr_idx] - t0
                wall_start = now - (t[curr_idx] - t0) / rs
        else:
            # Fixed-FPS
            prev_tgt = self.camConfig.target_fps
            self.camConfig.target_fps = min(float(self.camConfig.target_fps) * SPEED_STEP, 320.0)
            if prev_tgt < 19.9 and self.camConfig.target_fps > 20.0:
                self.camConfig.target_fps = 20.0  # Rebaseline for numerical error
            fps = max(0.001, float(self.camConfig.target_fps))
            phase = (len(t) - curr_idx) if self.last_nonzero_sign < 0 else curr_idx
            wall_start = time.monotonic() - (phase / fps)
        return wall_start

    def _on_speed_down(self, curr_idx: int, t) -> float:
        """
        Decrease playback speed.
        - RT mode: divide rt_speed, then re-anchor so current frame stays put.
        - Fixed-FPS: decrease target_fps, then re-anchor to current frame index.
        Returns new wall_start.
        """
        if getattr(self.camConfig, "playback_mode", None) == PlaybackSpeed.Real_time:
            prev_rt = self.camConfig.rt_speed
            self.camConfig.rt_speed = max(float(self.camConfig.rt_speed) * SPEED_STEP_INV, 0.01)
            if prev_rt > 1.01 and self.camConfig.rt_speed < 1.0:
                self.camConfig.rt_speed = 1.0
            now = time.monotonic()
            rs = max(1e-6, float(self.camConfig.rt_speed))
            t0, tN = t[0], t[-1]
            if self.last_nonzero_sign < 0:
                wall_start = now - (tN - t[curr_idx]) / rs
            else:
                wall_start = now - (t[curr_idx] - t0) / rs
        else:
            prev_tgt = self.camConfig.target_fps
            self.camConfig.target_fps = max(float(self.camConfig.target_fps) * SPEED_STEP_INV, 0.1)
            if prev_tgt > 20.1 and self.camConfig.target_fps < 20.0:
                self.camConfig.target_fps = 20.0  # Rebaseline for numerical error
            fps = max(0.001, float(self.camConfig.target_fps))
            phase = (len(t) - curr_idx) if self.last_nonzero_sign < 0 else curr_idx
            wall_start = time.monotonic() - (phase / fps)
        return wall_start

    def _on_toggle_overlays(self):
        self._toggle('undistort')
        self._toggle('yoloInference')
        self._toggle('yoloInference')
        self._toggle('detect_horizon')
        self._toggle('hyper_focus')
        self._toggle('factor_graph')

    def _toggle(self, attribute: str):
        self._flag_vars[attribute].set(not self._flag_vars[attribute].get())
        self.saveToCache()

    def _on_mark_start(self, curr_idx: int):
        self.camConfig.start_export_idx = curr_idx
        if self.camConfig.end_export_idx < self.camConfig.start_export_idx:
            self.camConfig.end_export_idx = self.camConfig.start_export_idx + 1
        self.exportStartFrame.configure(text=f'Start Frame: {self.camConfig.start_export_idx}')
        self.exportEndFrame.configure(text=f'End Frame: {self.camConfig.end_export_idx}')
        self.saveToCache()

    def _on_mark_end(self, curr_idx):
        self.camConfig.end_export_idx = curr_idx
        if self.camConfig.end_export_idx < self.camConfig.start_export_idx:
            self.camConfig.end_export_idx = max(0, self.camConfig.end_export_idx - 1)
        self.exportStartFrame.configure(text=f'Start Frame: {self.camConfig.start_export_idx}')
        self.exportEndFrame.configure(text=f'End Frame: {self.camConfig.end_export_idx}')
        self.saveToCache()

    def _on_adjust_offset(self, delta: float):
        self.camConfig.cam_to_log_time_offset += float(delta)

    def _on_persist_offset(self):
        offset = deepcopy(self.camConfig.cam_to_log_time_offset)
        self.write_offset_csv()
        print(f"Saved offset {offset:+.3f}s to __TIME_OFFSET.csv")

    def write_offset_csv(self):
        self.hud_marker.update_offset(self.camConfig.cam_to_log_time_offset)
        out_csv = Path(self.camConfig.hud_data_filepath)
        import pandas as pd
        pd.DataFrame({"offset": [self.hud_marker.offset]}).to_csv(out_csv, index=False)

        self.camConfig.cam_to_log_time_offset = 0.0

    def analyze_image(self,
                      frame,
                      img_time=None,
                      name=None,
                      display_in_realtime=True,
                      box_around=False):

        if frame is None:
            return

        self.curr_frame_gray = None

        # Sets self.curr_frame to (potentially undistorted) frame, and makes a copy onto self.markup_frame
        if self.calibration.validCal and self.camConfig.undistort:
            self.undistort(frame)
        else:
            self.curr_frame = frame  # explicit reference passed, saves copy if not undistorting

        # np.copyto is faster (doesn't reallocate), but requires destination to match shape
        if self.markup_frame is None or self.markup_frame.shape != self.curr_frame.shape:
            self.markup_frame = self.curr_frame.copy()
        else:
            np.copyto(self.markup_frame, self.curr_frame)

        if self.camConfig.draw_chessboard:
            self.draw_chessboard()

        if self.camConfig.processingKernel != ImageKernel.Unfiltered:
            self.applyKernel()

        if self.camConfig.detect_corners:
            self.corner_detection()

        if self.camConfig.detectTags and self.detector is not None:
            self.detectAprilTags()
            if self.camConfig.hideAprilTags:
                self.inpaint_apriltags()

        if self.camConfig.pnpLidarPoints and self.detector is not None:
            self.pnpLidarPoints()
        else:
            self.pnpResult = None

        if self.camConfig.qnpLidarPoints and self.detector is not None:
            self.qnpLidarPoints()
        else:
            self.qnpResult = None

        if self.camConfig.detect_horizon:
            self.detectHorizon()

        if self.camConfig.hyper_focus:
            self.hyper_focus()

        if self.camConfig.phase_correlation:
            self.phase_correlation()

        if self.camConfig.yoloInference:
            self.run_yolo(frame)  # Takes original frame, not undistort. YOLO presumes original.
            # Optional: pose estimation directly from YOLO centers (PnP / QnP / KF-weighted QnP)
            try:
                self.pose_from_yolo(img_time)
            except Exception:
                pass
        else:
            self.last_bounding_box_size = None
            self.last_yolo_center = None
            self.pnpYoloResult = None
            self.qnpYoloResult = None
            self.qnpKFYoloResult = None

        if self.camConfig.factor_graph:
            self.factor_graph(img_time)
        else:
            self.last_yolo_3d_estimate = None

        if self.camConfig.hud and img_time is not None:
            self.draw_HUD(img_time)

        if box_around:
            x, y, _ = self.markup_frame.shape
            cv2.rectangle(self.markup_frame, (0, 0), (x - 1, y - 1), clr.HUD_YELLOW, 10)

        if self.camConfig.imageSource == ImageSource.Stream_from_Folder:
            self.draw_name(name)

        if img_time is not None:
            self.draw_time(img_time)

        if self.printLidar:
            self.print_pnp_results()

        if display_in_realtime:
            if self.camConfig.imageSource == ImageSource.Stream_from_Folder:
                self.draw_playbackStats()
            self.cleanup()

        if not display_in_realtime:
            return np.ascontiguousarray(self.markup_frame).copy()

    def draw_playbackStats(self):

        (h, w) = self.markup_frame.shape[:2]
        self.lowPassFPS = 0.925 * self.lowPassFPS + 0.075 * self.curr_fps
        (txt_width, txt_height), base = cv2.getTextSize("I", cv2.FONT_HERSHEY_SIMPLEX, med_text(w), 4)
        txt_pix_start_perRow = txt_height + 10
        cv2.putText(self.markup_frame, f"Offset: {self.camConfig.cam_to_log_time_offset:+.2f}s",
                    (10, txt_pix_start_perRow * 2), cv2.FONT_HERSHEY_SIMPLEX,
                    med_text(w), clr.BLACK, 4)
        cv2.putText(self.markup_frame, f"Offset: {self.camConfig.cam_to_log_time_offset:+.2f}s",
                    (10, txt_pix_start_perRow * 2), cv2.FONT_HERSHEY_SIMPLEX,
                    med_text(w), clr.HUD_YELLOW, 2)
        cv2.putText(self.markup_frame,
                    f'Realtime: {self.camConfig.rt_speed:.2f}' if self.camConfig.playback_mode == PlaybackSpeed.Real_time else f'FPS: {self.lowPassFPS:.2f}/{self.camConfig.target_fps:.2f}',
                    (10, txt_pix_start_perRow * 3), cv2.FONT_HERSHEY_SIMPLEX,
                    med_text(w), clr.BLACK, 4)
        cv2.putText(self.markup_frame,
                    f'Realtime: {self.camConfig.rt_speed:.2f}' if self.camConfig.playback_mode == PlaybackSpeed.Real_time else f'FPS: {self.lowPassFPS:.2f}/{self.camConfig.target_fps:.2f}',
                    (10, txt_pix_start_perRow * 3), cv2.FONT_HERSHEY_SIMPLEX,
                    med_text(w), clr.HUD_YELLOW, 2)

    def draw_time(self, img_time):
        time_str = f"Flight Time: {img_time:.2f}"  # + 173.11338 - 11.658461:.2f}"
        (time_width, time_height), base = cv2.getTextSize(time_str, cv2.FONT_HERSHEY_SIMPLEX,
                                                          med_text(self.curr_frame.shape[0]), 4)
        img_w, img_h, *_ = self.curr_frame.shape
        cv2.putText(self.markup_frame, time_str, (img_w - time_width - 10, img_h - time_height * 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, med_text(self.curr_frame.shape[0]), clr.HUD_GREEN, 2)

    def draw_name(self, name):
        (width, height), base = cv2.getTextSize(os.path.basename(name), cv2.FONT_HERSHEY_SIMPLEX,
                                                med_text(self.curr_frame.shape[0]), 4)
        img_w, img_h, *_ = self.curr_frame.shape
        cv2.putText(self.markup_frame, os.path.basename(name), (img_w - width - 10, img_h - height),
                    cv2.FONT_HERSHEY_SIMPLEX, med_text(self.curr_frame.shape[0]), clr.HUD_GREEN, 2)

    def draw_HUD(self, img_time):
        from support.viz.HUD_draw import HUD_Marker
        if self.hud_marker is None:
            self.hud_marker = HUD_Marker()
            self.hud_marker.read_attitude_files(self.camConfig.hud_data_filepath)
        self.hud_marker.draw_HUD(self.markup_frame, img_time)

    def draw_chessboard(self):
        if self.curr_frame_gray is None:
            self.curr_frame_gray = cv2.cvtColor(self.markup_frame, cv2.COLOR_BGR2GRAY)

        self._checker_residual.draw_chessboard(self.markup_frame, self.curr_frame_gray, self._cb_pattern)


    def inpaint_apriltags(self,
                          radius_px: int = 3,
                          dilate_px: int = 2,
                          method: int = cv2.INPAINT_TELEA,
                          feather: bool = True):
        if self.curr_frame_gray is None or self.markup_frame is None:
            return
        if self.detector is None:
            return

        corners, ids, rejected = self.detector.detectMarkers(self.curr_frame_gray)
        if corners is None or len(corners) == 0:
            return

        mh, mw = self.markup_frame.shape[:2]
        gh, gw = self.curr_frame_gray.shape[:2]

        # scale factors from detection image to markup image
        sx = mw / float(gw)
        sy = mh / float(gh)

        # how much to pad each ROI beyond the exact tag corners
        pad = dilate_px + radius_px + 3

        for c in corners:
            # c shape ~ (1, 4, 2) -> (4, 2)
            pts = np.asarray(c).squeeze().reshape(-1, 2).astype(np.float32)

            # scale to markup_frame coords
            pts_scaled = np.empty_like(pts, dtype=np.float32)
            pts_scaled[:, 0] = pts[:, 0] * sx
            pts_scaled[:, 1] = pts[:, 1] * sy

            # tight bounding box around the tag
            x_min = int(np.floor(pts_scaled[:, 0].min())) - pad
            x_max = int(np.ceil(pts_scaled[:, 0].max())) + pad
            y_min = int(np.floor(pts_scaled[:, 1].min())) - pad
            y_max = int(np.ceil(pts_scaled[:, 1].max())) + pad

            # clamp to image
            x_min = max(x_min, 0)
            y_min = max(y_min, 0)
            x_max = min(x_max, mw - 1)
            y_max = min(y_max, mh - 1)
            if x_max <= x_min or y_max <= y_min:
                continue  # degenerate ROI

            roi_w = x_max - x_min + 1
            roi_h = y_max - y_min + 1

            # build local mask for this tag only
            mask_roi = np.zeros((roi_h, roi_w), dtype=np.uint8)

            # shift tag points into ROI coordinates
            pts_roi = pts_scaled.copy()
            pts_roi[:, 0] -= x_min
            pts_roi[:, 1] -= y_min
            pts_int = pts_roi.astype(np.int32)

            cv2.fillConvexPoly(mask_roi, pts_int, 255)

            # optional dilation to cover borders
            if dilate_px > 0:
                k = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)
                )
                mask_roi = cv2.dilate(mask_roi, k)

            # slice out the ROI from the big frame
            frame_roi = self.markup_frame[y_min:y_max + 1, x_min:x_max + 1]

            # inpaint only this small region
            inpainted_roi = cv2.inpaint(frame_roi, mask_roi, radius_px, method)

            if feather:
                blur_ks = max(3, 2 * radius_px + 1)
                soft = cv2.GaussianBlur(mask_roi, (blur_ks, blur_ks), 0).astype(np.float32) / 255.0
                soft = soft[..., None]  # (H,W,1)

                base = frame_roi.astype(np.float32)
                inp = inpainted_roi.astype(np.float32)
                blended_roi = (soft * inp + (1.0 - soft) * base).astype(np.uint8)

                self.markup_frame[y_min:y_max + 1, x_min:x_max + 1] = blended_roi
            else:
                self.markup_frame[y_min:y_max + 1, x_min:x_max + 1] = inpainted_roi

    def print_pnp_results(self):
        np.set_printoptions(precision=5, threshold=sys.maxsize, suppress=True)

        if self.lidarTruthPoints is None:
            self.loadTruthPoints()

        points = None
        if self.centers is not None and len(self.centers) >= 6:
            truthPoints = copy.copy(self.lidarTruthPoints.truthPoints)
            points = []

            removeIDs = []
            for idx, detectID in enumerate(self.detectIDS):
                try:
                    points.append(truthPoints[str(detectID[0])])
                except KeyError as e:
                    removeIDs.append(idx)

            centers = self.centers.copy()
            for id in reversed(removeIDs):
                centers = np.delete(centers, id, axis=0)
            points = np.array(points)

        probe_pose = np.array([4.89965725, .20014286, -1.55304432])

        if points is not None and self.detector is not None:
            print(f'Obj Points: \n{points}')
        if self.centers is not None:
            print(f'Img Points: \n{self.centers}')
        print(f'Cam Matrix: \n{self.calibration.getCameraMatrix()}')
        if self.pnpResult is not None:
            print(f'PnP Result: \ncam_R_tgt:\n{self.pnpResult[0].to_dcm()}\ncam_t_tgt:\n{self.pnpResult[1]}')
        if self.qnpResult is not None:
            print(f'QnP Result: \ncam_R_tgt:\n{self.qnpResult[0].to_dcm()}\ncam_t_tgt:\n{self.qnpResult[1]}')
        print()
        print(f'Diff: {self.pnpResult[1]}')
        print(f'Diff: {probe_pose}')
        print(f'Diff: {self.pnpResult[1] - probe_pose}')

        self.printLidar = False

    def update_cube_map_vectors(self):
        """Compute and store direction vectors for each cube face, shape: (6, H, W, 3)"""
        axes = {
            'right': ([1, 0, 0], [0, -1, 0]),
            'left': ([-1, 0, 0], [0, -1, 0]),
            'top': ([0, -1, 0], [0, 0, -1]),
            'bottom': ([0, 1, 0], [0, 0, 1]),
            'front': ([0, 0, 1], [0, -1, 0]),
            # 'back': ([0, 0, -1], [0, -1, 0]),
        }

        self.faces_dirs = {}
        rng = np.linspace(-1, 1, self.face_size)
        xx, yy = np.meshgrid(rng, -rng)  # Flip Y for image coordinates

        for name, (center, up) in axes.items():
            center = np.array(center)
            up = np.array(up)
            # noinspection PyUnreachableCode
            right = np.cross(center, up)

            dirs = (
                    center[None, None, :]
                    + xx[..., None] * right[None, None, :]
                    + yy[..., None] * up[None, None, :]
            )
            dirs /= np.linalg.norm(dirs, axis=2, keepdims=True)
            self.faces_dirs[name] = dirs.astype(np.float32)

        # return faces
        self.fisheye_to_cubemap_vectorized()

    def update_frontFace_vector(self):
        """Compute and store direction vectors for each cube face, shape: (6, H, W, 3)"""
        axes = {
            'front': ([0, 0, 1], [0, -1, 0])
        }

        self.faces_dirs = {}
        rng = np.linspace(-1, 1, self.face_size)
        xx, yy = np.meshgrid(rng, -rng)  # Flip Y for image coordinates

        for name, (center, up) in axes.items():
            center = np.array(center)
            up = np.array(up)
            # noinspection PyUnreachableCode
            right = np.cross(center, up)

            dirs = (
                    center[None, None, :]
                    + xx[..., None] * right[None, None, :]
                    + yy[..., None] * up[None, None, :]
            )
            dirs /= np.linalg.norm(dirs, axis=2, keepdims=True)
            self.faces_dirs[name] = dirs.astype(np.float32)

        # return faces
        self.fisheye_to_cubemap_vectorized()

    def fisheye_to_cubemap_vectorized(self):

        cube_faces = {}

        self.map_x = {}
        self.map_y = {}

        for face, dirs in self.faces_dirs.items():
            dirs_reshaped = dirs.reshape(-1, 1, 3)

            # Only keep directions roughly facing the front hemisphere
            forward_mask = dirs_reshaped[:, 0, 2] > 0  # Z > 0 means forward
            valid_dirs = dirs_reshaped[forward_mask]

            if valid_dirs.size > 0:
                # Project valid directions
                img_points, _ = cv2.fisheye.projectPoints(
                    valid_dirs, np.zeros(3), np.zeros(3),
                    self.calibration.getCameraMatrix(),
                    self.calibration.getDistortion()
                )
                img_points = img_points.reshape(-1, 2)

                # Prepare remap coordinates
                full_img_points = np.full((self.face_size * self.face_size, 2), -1, dtype=np.float32)
                full_img_points[forward_mask] = img_points

                self.map_x[face] = full_img_points[:, 0].reshape(self.face_size, self.face_size)
                self.map_y[face] = full_img_points[:, 1].reshape(self.face_size, self.face_size)

    def apply_fisheye_faces(self, frame):
        self.cubemap_faces = {}

        for face in self.faces_dirs:
            self.cubemap_faces[face] = self.remap(face, frame)

    def remap(self, face, frame):
        return cv2.remap(
            frame, self.map_x[face], self.map_y[face],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))

    def stitch_cubemap_faces(self, layout, cells=3):
        """
        Arrange the 6 cubemap faces into a 2x3 stitched layout.
        Layout:
            +--------+--------+--------+
            |        |   top  |        |
            +--------+--------+--------+
            |  left   | front |  right |
            +--------+--------+--------+
            |        | bottom |        |
            +--------+--------+--------+
        """
        stitched = np.zeros((cells * self.face_size, cells * self.face_size, 3), dtype=np.uint8)

        for face, (row, col) in layout.items():
            if face in self.cubemap_faces:
                y, x = row * self.face_size, col * self.face_size
                stitched[y:y + self.face_size, x:x + self.face_size] = self.cubemap_faces[face]

        return stitched

    def undistort(self, frame):

        if self.calibration.fisheye:
            if self.camConfig.cubemap:
                if self.face_size is None or self.face_size != 600:
                    self.face_size = 600
                    self.update_cube_map_vectors()

                self.apply_fisheye_faces(frame)
                layout = {
                    'bottom': (2, 1),
                    'left': (1, 0),
                    'front': (1, 1),
                    'right': (1, 2),
                    # 'back': (1, 0),
                    'top': (0, 1)}
                self.curr_frame = self.stitch_cubemap_faces(layout, cells=3)

            else:
                if self.face_size is None or self.face_size != min(frame.shape[:2]):
                    self.face_size = min(frame.shape[:2])
                    self.update_frontFace_vector()

                self.apply_fisheye_faces(frame)

                # self.curr_frame = self.stitch_cubemap_faces(layout, cells=1)
                self.curr_frame = self.cubemap_faces['front']
        else:
            self.curr_frame = cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT)

    def applyKernel(self):
        if self.camConfig.processingKernel != ImageKernel.Gabor and self.GaborGUI is not None:
            self.GaborGUI.close()
            self.GaborGUI = None

        from support.vision.filter_image import applyConvolutionFilter

        if self.camConfig.processingKernel == ImageKernel.Gabor:
            from support.vision.filter_image import GaborGUI
            if self.GaborGUI is None:
                self.GaborGUI = GaborGUI()
            self.markup_frame = applyConvolutionFilter(self.markup_frame,
                                                       self.camConfig.processingKernel,
                                                       self.GaborGUI.gaborFilter)
            return

        self.markup_frame = applyConvolutionFilter(self.markup_frame,
                                                   self.camConfig.processingKernel)

    def corner_detection(self):
        if self.curr_frame_gray is None:
            self.curr_frame_gray = cv2.cvtColor(self.curr_frame, cv2.COLOR_BGR2GRAY)
        harris_corners = cv2.cornerHarris(self.curr_frame_gray, 3, 3, 0.05)

        self.markup_frame[harris_corners > 0.025 * harris_corners.max()] = [0, 255, 255]

    def detectAprilTags(self, scale: float = 0.6):
        """
        Faster AprilTag detection:
          - detect on downscaled image
          - upscale corners
          - refine on full-res gray image with cornerSubPix
        """
        if self.curr_frame_gray is None:
            self.curr_frame_gray = cv2.cvtColor(self.curr_frame, cv2.COLOR_BGR2GRAY)
        if self.detector is None:
            return

        gray_full = self.curr_frame_gray
        h, w = gray_full.shape[:2]

        # 1) Downscale for detection
        if not (0.2 <= scale < 1.0):
            scale = 0.6
        small = cv2.resize(gray_full, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)

        # 2) Detect on smaller image
        corners_small, ids, rejected = self.detector.detectMarkers(small)

        self.centers = None
        self.detectIDS = []

        if corners_small is None or ids is None or len(corners_small) == 0:
            return

        # 3) Upscale corners to full-res and pack into a single array
        all_pts = []
        marker_lengths = []
        for c in corners_small:
            # c: (4,1,2) or (N,1,2)
            pts = c.reshape(-1, 2).astype(np.float32) / scale
            marker_lengths.append(len(pts))
            all_pts.append(pts)

        all_pts = np.concatenate(all_pts, axis=0).reshape(-1, 1, 2)

        # 4) Subpixel refine on full-res gray image
        #    (this is what gives you precise centers back)
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            20,  # max iterations
            0.01  # epsilon
        )
        cv2.cornerSubPix(gray_full, all_pts, (5, 5), (-1, -1), criteria)

        # 5) Split back per marker and draw / accumulate centers
        refined_corners_per_marker = []
        idx0 = 0
        for length in marker_lengths:
            refined_corners_per_marker.append(
                all_pts[idx0:idx0 + length].reshape(-1, 2).copy()
            )
            idx0 += length

        for corners, idx in zip(refined_corners_per_marker, ids):
            # corners: (4,2)
            polyline = [corners.astype(np.int32).reshape((-1, 1, 2))]
            pixCenter = np.mean(corners, axis=0).astype(np.int32)

            if not self.camConfig.hideAprilTags:
                cv2.polylines(self.markup_frame, polyline, True, clr.HUD_GREEN, 4, lineType=cv2.FILLED)
                cv2.putText(self.markup_frame, str(idx[0]), tuple(pixCenter),
                            cv2.FONT_HERSHEY_SIMPLEX, small_text(self.curr_frame.shape[0]), clr.HUD_GREEN, 4)
                cv2.putText(self.markup_frame, str(idx[0]), tuple(pixCenter),
                            cv2.FONT_HERSHEY_SIMPLEX, small_text(self.curr_frame.shape[0]), (0, 0, 0), 1)

            self.detectIDS.append(idx)

            if self.centers is None:
                self.centers = np.array(pixCenter, dtype=np.float32)
            else:
                self.centers = np.vstack((self.centers, pixCenter.astype(np.float32)))

    def pnpLidarPoints(self):

        if self.lidarTruthPoints is None:
            self.loadTruthPoints()

        if self.centers is not None and len(self.centers) >= 6:
            truthPoints = deepcopy(self.lidarTruthPoints.truthPoints)
            points = []
            distParams = np.zeros((5,))  # use image undistort instead

            removeIDs = []
            for idx, detectID in enumerate(self.detectIDS):
                try:
                    points.append(truthPoints[str(detectID[0])])
                except KeyError as e:
                    removeIDs.append(idx)

            centers = deepcopy(self.centers)
            for id in reversed(removeIDs):
                centers = np.delete(centers, id, axis=0)
            points = np.array(points)

            if len(points) < 6:
                return

            ret, rvec, tvec = cv2.solvePnP(objectPoints=points,
                                           imagePoints=centers,
                                           cameraMatrix=self.calibration.getCameraMatrix(),
                                           distCoeffs=distParams,
                                           flags=cv2.SOLVEPNP_ITERATIVE)

            if ret:
                projectedPoints_orig, _ = cv2.projectPoints(self.lidarTruthPoints.getTruthPointsNumpy(),
                                                            rvec=rvec,
                                                            tvec=tvec,
                                                            cameraMatrix=self.calibration.getCameraMatrix(),
                                                            distCoeffs=distParams)

                self.plotOnImg(projectedPoints_orig[:, 0, :].astype(int),
                               list(self.lidarTruthPoints.getTruthPointsDict().keys()), (255, 255, 0))

                quatPnP, vectPnP = q.fromOpenCV_toAftr_rvec(rvec, tvec)

                self.pnpResult = (quatPnP, vectPnP)

                cv2.putText(self.markup_frame, 'Orientation (quat) From LiDAR: ' + format(quatPnP, 'ijk.6f'), (50, 75),
                            cv2.FONT_HERSHEY_DUPLEX, small_text(self.markup_frame.shape[0]),
                            (255, 255, 0), 3,
                            cv2.LINE_AA)
                cv2.putText(self.markup_frame, 'Location From LiDAR: ' + np.array2string(vectPnP),
                            (50, 150), cv2.FONT_HERSHEY_DUPLEX, small_text(self.markup_frame.shape[0]),
                            (255, 255, 0), 3,
                            cv2.LINE_AA)

    def qnpLidarPoints(self):

        if self.lidarTruthPoints is None:
            self.loadTruthPoints()

        if self.centers is not None and len(self.centers) >= 6:
            truthPoints = deepcopy(self.lidarTruthPoints.truthPoints)

            points = []
            # distParams = np.zeros((5,))  # use image undistort instead

            removeIDs = []
            for idx, detectID in enumerate(self.detectIDS):
                try:
                    points.append(truthPoints[str(detectID[0])])
                except KeyError as e:
                    removeIDs.append(idx)

            centers = deepcopy(self.centers)
            for idx in reversed(removeIDs):
                centers = np.delete(centers, idx, axis=0)
            points = np.array(points)

            if len(points) < 6:
                return

            quat, vect, *_ = solveQnP(points, centers, self.calibration, None)
            xyz_proj = quat * self.lidarTruthPoints.getTruthPointsNumpy() + vect

            q_aftr_from_cv = mat2quat(np.array([[0., 0., 1.],
                                                [-1., 0., 0.],
                                                [0., -1., 0.]], float))

            vect = q_aftr_from_cv * vect

            quat = q_aftr_from_cv * quat

            us_vs_s_proj = np.zeros((xyz_proj.shape[0], 2))
            us_vs_s_proj[:, 0] = self.calibration.fx * xyz_proj[:, 0] / xyz_proj[:, 2] + self.calibration.cx
            us_vs_s_proj[:, 1] = self.calibration.fy * xyz_proj[:, 1] / xyz_proj[:, 2] + self.calibration.cy

            self.plotOnImg(us_vs_s_proj.astype(int),
                           list(self.lidarTruthPoints.getTruthPointsDict().keys()), (255, 255, 255))
            self.qnpResult = (quat, vect)
            cv2.putText(self.markup_frame, 'Orientation (quat) From LiDAR: ' + format(quat, 'ijk.6f'), (50, 225),
                        cv2.FONT_HERSHEY_DUPLEX,
                        small_text(self.markup_frame.shape[0]),
                        (255, 255, 0), 3,
                        cv2.LINE_AA)
            cv2.putText(self.markup_frame, 'Location From LiDAR: ' + np.array2string(vect), (50, 300),
                        cv2.FONT_HERSHEY_DUPLEX,
                        small_text(self.markup_frame.shape[0]),
                        (255, 255, 0), 3,
                        cv2.LINE_AA)

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

    def detectHorizon(self):

        if self.curr_frame_gray is None:
            self.curr_frame_gray = cv2.cvtColor(self.curr_frame, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(self.curr_frame_gray, 100, 200, apertureSize=3)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, 50,
                                minLineLength=np.sum(self.curr_frame.shape) / 10.0,
                                maxLineGap=20)

        color = (0, 0, 255)

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
                    x2 = self.curr_frame.shape[1]
                    self.hor_last_midpoint = ((y1 + m * x2 / 2.0) + self.hor_last_midpoint) / 2.0
                    self.hor_last_slope = (m + self.hor_last_slope) / 2.0
                    color = (20, 150, 20)

        x1 = 0
        x2 = int(self.curr_frame.shape[1])
        y1 = int(self.hor_last_midpoint - self.hor_last_slope * x2 / 2.0)
        y2 = int(self.hor_last_midpoint + self.hor_last_slope * x2 / 2.0)

        cv2.line(self.markup_frame, (x1, y1), (x2, y2), color, 2)

        self.horizon_line = (x1, y1, x2, y2)

    def check_above_horizon(self, pt):
        if self.horizon_line is None:
            return True

        x1, y1, x2, y2 = self.horizon_line
        return np.cross(np.array([x2 - x1, y2 - y1]), np.array([pt[0] - x1, pt[1] - y1])) < 0

    def hyper_focus(self):

        if not self.camConfig.factor_graph:
            if self.last_bounding_box_size is not None:
                self.radius = (self.last_bounding_box_size[0] + self.last_bounding_box_size[
                    1] + self.radius * 4.0) / 5.0
            else:
                self.confSlider(self.camConfig.yolo_conf)

            self.markup_frame = dim_except_circle(self.markup_frame, self.current_center_est, 3.0 * self.radius, 0.00)
            self.markup_frame = dim_except_circle(self.markup_frame, self.current_center_est, 1.5 * self.radius, 0.50)

            self.radius = min(800.0, self.radius + 12.0)
            if self.yoloSession is not None:
                self.yoloSession.conf = (0.8 - 0.5) * self.radius / 800.0 + 0.5
                self.confSliderBar.set(self.yoloSession.conf)
        else:
            if self.last_bounding_box_size is not None:
                # 1.0 for single feature, 1.5 for drogue
                self.min_radius = (self.last_bounding_box_size[0] + self.last_bounding_box_size[1]) * 1.0

            # 5.0 for single feature, 50.0 for drogue
            ellipse_width = 5.0 * self.current_var_y + self.min_radius
            ellipse_height = 5.0 * self.current_var_z + self.min_radius

            if self.curr_FG_pixel[0] < 0 or self.curr_FG_pixel[1] < 0 or self.curr_FG_pixel[0] > self.curr_frame.shape[
                1] or \
                    self.curr_FG_pixel[1] > self.curr_frame.shape[0]:
                return

            self.markup_frame = dim_except_circle(self.markup_frame, self.curr_FG_pixel, x_axes=ellipse_width,
                                                  y_axes=ellipse_height, dim_factor=0.10)
            self.markup_frame = dim_except_circle(self.markup_frame, self.curr_FG_pixel, x_axes=ellipse_width * 2.0,
                                                  y_axes=ellipse_height * 2.0, dim_factor=0.00)

    def run_yolo(self, orig_image):
        '''
        Runs YOLO on subsequent images. If the yolo model is single featured, and the object is estimated less than
        100 meters away, then it updates this class's estimation of the solution.
        :return: None, but does adjust
        '''
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
        self.markup_frame, output = self.yoloSession.inferOnImage(orig_image, self.markup_frame,
                                                                               self.camConfig.yoloBiasTracking)

        want_pnp = bool(getattr(self.camConfig, "pnpYoloPoints", False))
        want_qnp = bool(getattr(self.camConfig, "qnpYoloPoints", False))
        want_qnp_kf = bool(getattr(self.camConfig, "qnpKFYoloPoints", False))

        algos = pnpDrw.twoToThreeSelectedAlgorithms()
        algos.use_pnp = want_pnp
        algos.use_qnp = want_qnp
        algos.use_qnp_kf = want_qnp_kf

        self.pnpDrawer.markUpImage(image=self.markup_frame,
                output=output,
                markup_is_undistorted=self.camConfig.undistort,
                calibration=self.calibration,
                conf=self.camConfig.yolo_conf,
                iou=self.camConfig.yolo_iou,
                yoloSize=self.yoloSession.yoloSize,
                idsNamesLocs=self.yoloSession.reader.idsNamesLocs,
                usedAlgos=algos)

        centers, boxes, scores, class_ids, time = output

        if len(centers) > 0 and self.yoloSession.reader.numClasses == 1:
            best_idx = scores.index(max(scores))
            img_yolo_x_correction = self.curr_frame.shape[0] / self.yoloSession.reader.imageSize
            img_yolo_y_correction = self.curr_frame.shape[1] / self.yoloSession.reader.imageSize

            self.last_bounding_box_size = ((boxes[best_idx][2] - boxes[best_idx][0]) * img_yolo_x_correction,
                                           (boxes[best_idx][3] - boxes[best_idx][1]) * img_yolo_y_correction)
            self.last_yolo_center = centers[best_idx]

            self.last_yolo_center = (int(
                self.last_yolo_center[0] * img_yolo_x_correction), int(
                self.last_yolo_center[1] * img_yolo_y_correction))

            K = self.calibration.getCameraMatrix()
            # d = self.calibration.getDistortion()  # Presume undistorted image
            twoD_points = np.array([self.last_yolo_center[0], self.last_yolo_center[1], 1.0])
            dist_est = self.calibration.fx * 4.07 / (self.last_bounding_box_size[0])

            if self.check_above_horizon(self.last_yolo_center):
                self.last_yolo_3d_estimate = np.linalg.inv(K).dot(twoD_points) * dist_est
                w, h, _ = self.curr_frame.shape
                cv2.putText(self.markup_frame, 'BB-Width Solution', (25, w - 75), cv2.FONT_HERSHEY_SIMPLEX,
                            med_text(self.markup_frame.shape[0]), (50, 255, 255), 1)
                cv2.putText(self.markup_frame,
                            f'x:{self.last_yolo_3d_estimate[0]:.3f}, y:{self.last_yolo_3d_estimate[1]:.3f}, z:{self.last_yolo_3d_estimate[2]:.3f}',
                            (25, w - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, med_text(self.markup_frame.shape[0]), (50, 255, 255), 1)
                self.current_center_est = ((self.current_center_est[0] * 2.0 + centers[best_idx][0]) / 3.0,
                                           (self.current_center_est[1] * 2.0 + centers[best_idx][1]) / 3.0)
                return

        self.last_bounding_box_size = None
        self.last_yolo_center = None

    def pose_from_yolo(self, img_time=None):
        """Compute (optional) PnP / QnP / KF-weighted QnP poses from YOLO detections."""
        # Guard: must have calibration
        if not getattr(self.calibration, "validCal", False):
            self.pnpYoloResult = None
            self.qnpYoloResult = None
            self.qnpKFYoloResult = None
            return

        # Quick exit if nothing enabled
        want_pnp = bool(getattr(self.camConfig, "pnpYoloPoints", False))
        want_qnp = bool(getattr(self.camConfig, "qnpYoloPoints", False))
        want_qnp_kf = bool(getattr(self.camConfig, "qnpKFYoloPoints", False))
        if not (want_pnp or want_qnp or want_qnp_kf):
            self.pnpYoloResult = None
            self.qnpYoloResult = None
            self.qnpKFYoloResult = None
            return


    def factor_graph(self, time):
        from support.runtime.fg_drogue_only import FactorGraph
        if self.FG is None:
            self.FG = FactorGraph()
        color = (120, 255, 120)

        if self.last_yolo_3d_estimate is not None:
            if time < self.last_time_update:
                self.FG.reset()
            elif time > self.last_time_update:
                self.FG.newRecvMeas(self.last_yolo_3d_estimate, time)
                self.last_time_update = time
            if self.FG.numMeas > 20:
                self.FG.popOldestMeas()
            if self.FG.numMeas > 2:
                self.FG.opt()
        else:
            color = (0, 0, 255)

        if time is not None and self.FG.numMeas > 2:
            K = self.calibration.getCameraMatrix()
            d = self.calibration.getDistortion()
            self.curr_FG_pixel = K.dot(self.FG.r_T_d[-1] + (time - self.last_time_update) * self.FG.r_V_d[-1])
            self.curr_FG_pixel = (self.curr_FG_pixel / self.curr_FG_pixel[2])[:2]

            h, w, _ = self.markup_frame.shape
            size = 15
            thickness = 2
            cv2.circle(self.markup_frame, (int(self.curr_FG_pixel[0]), int(self.curr_FG_pixel[1])), size, (0, 0, 0),
                       thickness)
            cv2.line(self.markup_frame, [int(self.curr_FG_pixel[0]) + size, int(self.curr_FG_pixel[1])],
                     [int(self.curr_FG_pixel[0]) - size, int(self.curr_FG_pixel[1])], (0, 0, 0), thickness)
            cv2.line(self.markup_frame, [int(self.curr_FG_pixel[0]), int(self.curr_FG_pixel[1]) + size],
                     [int(self.curr_FG_pixel[0]), int(self.curr_FG_pixel[1]) - size], (0, 0, 0), thickness)
            cv2.putText(self.markup_frame, 'Factor Graph Solution', (25, h - 125), cv2.FONT_HERSHEY_SIMPLEX,
                        med_text(self.markup_frame.shape[0]), (0, 0, 0), thickness)

            thickness = 1
            cv2.circle(self.markup_frame, (int(self.curr_FG_pixel[0]), int(self.curr_FG_pixel[1])), size, color,
                       thickness)
            cv2.line(self.markup_frame, [int(self.curr_FG_pixel[0]) + size, int(self.curr_FG_pixel[1])],
                     [int(self.curr_FG_pixel[0]) - size, int(self.curr_FG_pixel[1])], color, thickness)
            cv2.line(self.markup_frame, [int(self.curr_FG_pixel[0]), int(self.curr_FG_pixel[1]) + size],
                     [int(self.curr_FG_pixel[0]), int(self.curr_FG_pixel[1]) - size], color, thickness)
            cv2.putText(self.markup_frame, 'Factor Graph Solution', (25, h - 125), cv2.FONT_HERSHEY_SIMPLEX,
                        med_text(self.markup_frame.shape[0]), color, thickness)

            self.curr_r_T_d, self.curr_r_V_d = self.FG.r_T_d[-1], self.FG.r_V_d[-1]

            var_x, var_y, var_z, var_vx, var_vy, var_vz = self.FG.last_pos_covariance()

            self.current_var_x = var_x + var_vx * (time - self.last_time_update) * np.abs(self.curr_r_V_d[0])
            self.current_var_y = var_y + var_vy * (time - self.last_time_update) * np.abs(self.curr_r_V_d[1])
            self.current_var_z = var_z + var_vz * (time - self.last_time_update) * np.abs(self.curr_r_V_d[2])

        self.last_yolo_3d_estimate = None

    def phase_correlation(self):

        if self.calibration.validCal:
            cx = int(self.calibration.cx)
            cy = int(self.calibration.cy)
        else:
            cx = int(self.curr_frame.shape[0] / 2)
            cy = int(self.curr_frame.shape[1] / 2)

        if self.curr_frame_gray is None:
            self.curr_frame_gray = cv2.cvtColor(self.markup_frame, cv2.COLOR_BGR2GRAY)

        if self.last_image is not None and self.last_image.shape == self.curr_frame_gray.shape:
            lft_rt, ret = cv2.phaseCorrelate(self.curr_frame_gray.astype(np.float64) / 255.0,
                                             self.last_image.astype(np.float64) / 255.0)
            lft, rt = lft_rt
            cv2.arrowedLine(self.markup_frame, (cx, cy), (int(cx + 10 * lft), int(cy + 10 * rt)), (0, 0, 255), 3)

        self.last_image = copy.deepcopy(self.curr_frame_gray)

    def cleanup(self):

        if self.calibration.validCal:
            cx = int(self.calibration.cx)
            cy = int(self.calibration.cy)
        else:
            cx = int(self.curr_frame.shape[0] / 2)
            cy = int(self.curr_frame.shape[1] / 2)

        width = self.curr_frame.shape[0]
        height = self.curr_frame.shape[1]
        thickness = max(int(width / 250), 1)

        if self.camConfig.crosshairs:
            crosshairsH = np.array([[cx + max(int(width / 50), 10), cy], [cx - max(int(width / 50), 10), cy]])
            crosshairsV = np.array([[cx, cy + max(int(height / 50), 10)], [cx, cy - max(int(height / 50), 10)]])

            cv2.polylines(self.markup_frame, [crosshairsH], True, clr.HUD_GREEN, thickness)
            cv2.polylines(self.markup_frame, [crosshairsV], True, clr.HUD_GREEN, thickness)

        self.potentialResize()

        cv2.imshow(self.windowName, cv2.resize(self.markup_frame, (self.lastWidth, self.lastHeight)))

        if self.recording and time.time() - self.lastImageTime > self.camConfig.secondsBetweenImages:
            cv2.imwrite(os.path.join(self.default_filepath, str(self.img_idx) + '.png'), self.markup_frame)
            self.img_idx += 1
            self.lastImageTime = time.time()
            self.recordButton.configure(text=f'Saving Imagery: #{self.img_idx}')

    def plotOnImg(self, points, names, color):
        for idx, pxPt in enumerate(points):
            cv2.circle(self.markup_frame, (int(pxPt[0]), int(pxPt[1])), 5, color, 5)
            textLoc = (int(pxPt[0]) - 30, int(pxPt[1] - 30))
            cv2.putText(self.markup_frame, str(names[idx]), textLoc, cv2.FONT_HERSHEY_SIMPLEX,
                        med_text(self.markup_frame.shape[0]), (0, 0, 0),
                        12,
                        cv2.LINE_AA)
            cv2.putText(self.markup_frame, str(names[idx]), textLoc, cv2.FONT_HERSHEY_SIMPLEX,
                        med_text(self.markup_frame.shape[0]), color, 3,
                        cv2.LINE_AA)

    def potentialResize(self):
        if cv2.getWindowProperty(self.windowName, cv2.WND_PROP_VISIBLE) <= 0:
            return
        x, y, width, height = cv2.getWindowImageRect(self.windowName)
        aspectRatio = self.curr_frame.shape[1] / self.curr_frame.shape[0]
        if not cv2.getWindowProperty(self.windowName, cv2.WND_PROP_VISIBLE):
            return

        if not self.lastHeight == height and height != 0:
            cv2.resizeWindow(self.windowName, int(height * aspectRatio), height)
            self.lastHeight = height
            self.lastWidth = int(height * aspectRatio)
        elif not self.lastWidth == width and width != 0:
            cv2.resizeWindow(self.windowName, width, int(width / aspectRatio))
            self.lastWidth = width
            self.lastHeight = int(width / aspectRatio)

    @staticmethod
    def askFilepath(initDir, text):
        poss_filepath = filedialog.askdirectory(initialdir=initDir, mustexist=True, title=text)
        if poss_filepath == '':
            return None
        return poss_filepath



