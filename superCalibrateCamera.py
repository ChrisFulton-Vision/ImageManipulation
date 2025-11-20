import copy
import os
import pickle
import re
import threading
import time
import sys
import queue

import vmbpy.c_binding
from vmbpy import *

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable
from enum import Enum
from itertools import cycle
from tkinter import filedialog
from yaml import safe_load, dump

from concurrent.futures import ThreadPoolExecutor, wait
from customtkinter import CTkFrame, CTkButton, CTkLabel, CTkSlider, CTkEntry, CTkCheckBox, CTkComboBox, BooleanVar, StringVar, CTkProgressBar, END
from pandas import isna, read_csv, DataFrame

from cv2 import (cvtColor, COLOR_BGR2RGB, COLOR_BGR2GRAY, destroyAllWindows, waitKey, imread, namedWindow,
                 WND_PROP_VISIBLE, rectangle, getWindowProperty, WINDOW_NORMAL, VideoCapture, CAP_DSHOW, CAP_PROP_FPS,
                 resizeWindow, IMREAD_COLOR, getTextSize, FONT_HERSHEY_SIMPLEX, putText, INPAINT_TELEA, fillConvexPoly,
                 getStructuringElement, dilate, MORPH_ELLIPSE, inpaint, GaussianBlur, fisheye, remap, INTER_LINEAR,
                 BORDER_CONSTANT, cornerHarris, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, cornerSubPix, polylines,
                 FILLED, solvePnP, SOLVEPNP_ITERATIVE, projectPoints, LINE_AA, FONT_HERSHEY_DUPLEX, Canny, HoughLinesP,
                 line, circle, phaseCorrelate, arrowedLine, imshow, imwrite, getWindowImageRect, bitwise_and, aruco,
                 setNumThreads, getOptimalNewCameraMatrix, initUndistortRectifyMap, CV_16SC2, COLOR_GRAY2BGR, error,
                 COLOR_RGB2BGR, resize, setUseOptimized, ellipse, bitwise_not, add, undistortPoints, solvePnPRansac,
                 VideoWriter, INTER_AREA)
from PIL.Image import fromarray
from cv2_enumerate_cameras import enumerate_cameras
import numpy as np

from SupportModules import yolo
from SupportModules.Calibration import Calibration
from SupportModules.FG_DrogueOnly import FactorGraph
from SupportModules.ImageTimeReader import ImageTimeReader
from SupportModules.LidarTruth import TruthPoints
from SupportModules.FilterImage import ImageKernel, GaborGUI, applyConvolutionFilter
from SupportModules.HUD_draw import HUD_Marker
from SupportModules.TwoD_to_ThreeD import solveQnP
from SupportModules.bufferImageLoader import BufferedImageLoader as imgBuf
from SupportModules.convertToGif import make_gif, ExportQuality
from SupportModules.quaternions import *
from SupportModules.quaternions import Quaternion as q
from SupportModules.CVFontScaling import small_text, med_text, lrg_text

import logging
import math

SPEED_STEP = math.pow(2.0, 1.0 / 3.0)  # 3 presses -> 2×
SPEED_STEP_INV = 1.0 / SPEED_STEP

LOG = logging.getLogger("superCalibrate")

if not LOG.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(fmt)
    LOG.addHandler(handler)
    # LOG.setLevel(logging.INFO)
    # LOG.setLevel(logging.DEBUG)
    LOG.setLevel(logging.WARNING)

setNumThreads(0)
setUseOptimized(True)

#  pip install cv2_enumerate_cameras
#  or
#  pip install git+https://github.com/chinaheyu/cv2_enumerate_cameras.git

# Keys that should be treated as edge-triggered (one per distinct press)
_EDGE_KEYS = {ord(' '), ord('f'), ord('w'), ord('s'), ord('e'), ord('p'),
              ord('r'), ord('['), ord(']'), ord('{'), ord('}'),
              ord(';'), ord("'"), ord(':'), ord('"'), ord('b'), ord('n'), 27}

# Keys that should fire on every event (allow repeats within a burst)
_REPEAT_KEYS = {ord('a'), ord('d'), ord('c'), ord('z')}

# Cooldown for edge keys (optional, prevents accidental double-hits)
_EDGE_COOLDOWN_MS = 120


def _is_edge_allowed(key: int, last_ts: dict[int, float]) -> bool:
    now = time.monotonic()
    prev = last_ts.get(key, 0.0)
    if (now - prev) * 1000.0 >= _EDGE_COOLDOWN_MS:
        last_ts[key] = now
        return True
    return False


CTK_GREEN = '#2FA572'
HUD_GREEN = (0, 255, 0)
HUD_YELLOW = (0, 255, 255)
BUTTON_RED = 'red3'
CACHE_FILEPATH = str(Path.cwd() / "Caches" / "last_config.pkl")

class PausedCache:
    def __init__(self): self.idx = None; self.frame = None

    def set(self, i, f): self.idx, self.frame = i, f

    def get(self, i): return self.frame if self.idx == i else None

    def clear(self): self.idx = self.frame = None


@dataclass
class PlaybackState:
    """Single source of truth for playback state."""
    speed: float = 1.0  # signed: <0 reverse, 0 paused, >0 forward
    last_nonzero_sign: int = 1  # +1 or -1, used when resuming from pause
    stride: int = 1  # cached stride we last told the loader


def numerical_sort(file_name):
    try:
        return int(file_name.split('.')[0])
    except (ValueError, IndexError):
        return float('inf')


class ThreadStopper:
    def __init__(self):
        self._ev = threading.Event()

    def set(self):
        self._ev.set()

    def is_set(self) -> bool:
        return self._ev.is_set()


class ImageSource(Enum):
    Camera_Stream = 'Camera Stream'
    Static_Image = 'Static Image'
    Stream_from_Folder = 'Stream from Folder'


class PlaybackSpeed(Enum):
    Fixed_fps = 'fixed_fps'
    Real_time = 'realtime'

    def next(self):
        iterator = cycle(self.__class__)
        for member in iterator:
            if member is self:
                return next(iterator)


@dataclass
class CameraConfig:
    configFilepath: str = 'Configs/Default.yaml'
    calibFilepath: str = 'Calibrations/GenericAlvium864.txt'
    imageFilepath: Optional[str] = None
    cam_index: int = 0

    # feature flags
    detectTags: bool = False
    hideAprilTags: bool = True
    undistort: bool = False
    pnpLidarPoints: bool = False
    qnpLidarPoints: bool = False
    yoloInference: bool = False
    yoloBiasTracking: bool = False
    detect_corners: bool = False
    detect_horizon: bool = False
    factor_graph: bool = False
    hyper_focus: bool = False
    phase_correlation: bool = False
    crosshairs: bool = False
    cubemap: bool = False
    hud: bool = False

    # numeric params
    secondsBetweenImages: float = 1.0
    aprilTagSize: float = 0.168
    cam_to_log_time_offset: float = 0.0
    yolo_conf: float = 0.75
    yolo_iou: float = 1.00
    target_fps: float = 20.0
    rt_speed: float = 1.0

    # sources
    imageSource: ImageSource = None  # set default below in __post_init__
    lidarFilepath: str = None
    yoloFilepath: str = ''
    hud_data_filepath: str = ''

    # export range
    export_quality: ExportQuality = ExportQuality.med_quality
    start_export_idx: int = 0
    end_export_idx: int = 1

    # playback / processing
    playback_mode: PlaybackSpeed = None
    processingKernel: ImageKernel = None

    # Data Processing tab defaults
    dp_img_dir: str = ''
    dp_output_csv: str = ''
    dp_conf_list: str = "0.80"
    dp_ckptN: int = 200
    dp_prefetch: int = 32

    def __post_init__(self):
        # Keep existing defaults if not provided
        if self.imageSource is None:
            self.imageSource = ImageSource.Camera_Stream
        if self.playback_mode is None:
            self.playback_mode = PlaybackSpeed.Fixed_fps
        if self.processingKernel is None:
            self.processingKernel = ImageKernel.Unfiltered

    def copy(self, configToCopy):
        for obj in configToCopy.__dict__:
            try:
                self.__dict__[obj] = configToCopy.__dict__[obj]
            except KeyError as e:
                # Allows for versioning issues, changed naming conventions.
                LOG.info(f"Old cache loaded. Observe: {e}")
                pass

    @property
    def toDict(self):
        enum_classes = ['export_quality', 'imageSource', 'playback_mode', 'processingKernel']
        going_out = {}
        for attr in self.__dict__:
            if attr in ['configFilepath']:
                continue
            if attr in enum_classes:
                going_out[attr] = self.__getattribute__(attr).value
            else:
                going_out[attr] = self.__getattribute__(attr)
        return going_out

    def fromDict(self, my_dict: dict):
        enum_dict = {
            'export_quality': ExportQuality,
            'imageSource': ImageSource,
            'playback_mode': PlaybackSpeed,
            'processingKernel': ImageKernel
        }
        for key, value in my_dict.items():
            if hasattr(self, key):
                if key in enum_dict.keys():
                    self.__setattr__(key, enum_dict[key](value))
                else:
                    self.__setattr__(key, value)


class CameraGui(CTkFrame):
    def __init__(self, master, *args, **kwargs):

        self.profile_run_folder = False

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
            "cubemap", "hud", "hideAprilTags"
        ]
        self.recording = False
        self.yoloSession = yolo.YOLO()
        self.camConfig = CameraConfig()
        self.detector = None
        self.arucoDict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36H11)

        self.arucoParams = aruco.DetectorParameters()
        # self.arucoParams.adaptiveThreshWinSizeMin = 5
        # self.arucoParams.adaptiveThreshWinSizeMax = 35
        # self.arucoParams.adaptiveThreshWinSizeStep = 5
        # self.arucoParams.minMarkerPerimeterRate = 0.02  # or higher if tags are big
        # self.arucoParams.maxMarkerPerimeterRate = 1.0
        # self.arucoParams.cornerRefinementMinAccuracy = 0.1  # or 0.2
        # self.arucoParams.cornerRefinementMaxIterations = 20

        self._init_flag_vars()

        self.calibration = Calibration()
        self.detectIDS = None
        self.projectProbe = None
        self.centers = None
        self.indexDict = {}
        self.scanForCameras()
        self.windowName = 'Processed Image'
        self.filepath = ''
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
        self.FG = FactorGraph()
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
        self.hud_marker = HUD_Marker()
        self.lowPassFPS = 20.0
        self.pnpResult = None
        self.qnpResult = None

        self.pauseCache = PausedCache()
        self.playback = PlaybackState()

        # Optimization for undistort
        self.map1, self.map2 = None, None

        self.threadStopper = ThreadStopper()
        self._thread = None

        self.fps_time_log = time.time()
        self.curr_fps = 20.0
        self.pause = False
        self.last_nonzero_sign = 1

        self.imageProcessingKernelCombobox = None

        self.bank_indicator_points = None

        self.last_image = None

        self.available_sources = [source.value for source in ImageSource]

        self.streamOrImgCombo = CTkComboBox(self.cam_frame, values=self.available_sources,
                                                command=self.sourceUpdate)
        self.startStreamButton = CTkButton(master=self.cam_frame, text='Start Stream', fg_color=BUTTON_RED,
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
                                              text="../" + Path(self.filepath).name if self.filepath else "../")
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

        self.lidarTruthPoints = TruthPoints()
        self.selectYOLO_folderButton = CTkButton(self.cam_frame, text='Select YOLO Folder', fg_color=CTK_GREEN,
                                                     command=self.selectYoloFolder)
        self.selectYOLO_folderLabel = CTkLabel(self.cam_frame,
                                                   text="../" + Path(
                                                       self.camConfig.yoloFilepath).name if self.camConfig.yoloFilepath else "../"
                                                   )
        self.selectCalibLabel = None
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

        self.confSliderBar.set(self.camConfig.yolo_conf)
        self.iouSliderBar.set(self.camConfig.yolo_iou)

        self.selectCameraCombo.set(list(self.indexDict.keys())[self.camConfig.cam_index])
        self.vc = None

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

    def func_to_refit(self, func):
        self.func_that_refits = func

    def _init_flag_vars(self):
        for name in self._flags:
            v = BooleanVar(value=bool(getattr(self.camConfig, name)))
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

    def set_ui_active(self, active: bool):
        self._ui_active = bool(active)
        # Stop camera stream if page is hidden (don’t burn CPU/GPU off-screen)
        if not self._ui_active:
            try:
                self.startStreamOff()
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

        cache_path = Path(CACHE_FILEPATH)
        with cache_path.open('rb') as f:
            self.camConfig.configFilepath = pickle.load(f)
            self.configSelectLabel.configure(text=os.path.basename(self.camConfig.configFilepath))

        if os.path.exists(self.camConfig.configFilepath):
            with open(self.camConfig.configFilepath, 'r') as f:
                data = safe_load(f)
                self.camConfig.fromDict(data)
                self.update_post_newCamConfig()


    def update_post_newCamConfig(self):
        self.updateSingleOrStream(rowID=1)
        self.updateLogFile()
        self.ingestCalibration()
        self.updateYOLOLabel()
        self.updateLidarLabel()
        self.loadTruthPoints()

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
        """
        Debounced cache writer.

        - Normal calls:   saveToCache()
            Coalesce many rapid updates into a single write after delay_ms.
        - Immediate save: saveToCache(immediate=True)
            Write to disk right now (used when we need fresh data before
            calling loadFromCache(), etc.).
        """
        # If GUI isn't fully initialized or caller wants sync write, flush now.
        if immediate or not hasattr(self, "after"):
            # cancel any pending debounce
            if getattr(self, "_save_debounce_id", None) is not None and hasattr(self, "after_cancel"):
                try:
                    self.after_cancel(self._save_debounce_id)
                except Exception:
                    pass
                self._save_debounce_id = None

            self._flush_cache_now()
            return

        # Debounced path: cancel any pending save and schedule a new one
        if getattr(self, "_save_debounce_id", None) is not None:
            try:
                self.after_cancel(self._save_debounce_id)
            except Exception:
                pass

        self._save_debounce_id = self.after(delay_ms, self._flush_cache_now)


    def selectFolder(self):
        init_dir = Path(self.filepath).parent if self.filepath else Path.cwd()
        fp = self.askFilepath(str(init_dir), "Select Imagery Folder")
        if fp:
            self.filepath = fp
            self.saveToCache(immediate=True)
            self.loadFromCache()

    def loadCalibration(self):
        init_dir = Path(self.camConfig.calibFilepath or self.filepath or Path.cwd()).parent
        poss_filepath = filedialog.askopenfilename(initialdir=str(init_dir), title='Select Calibration File')
        if poss_filepath:
            self.camConfig.calibFilepath = poss_filepath
            self.ingestCalibration()

    def selectLidarFile(self):
        init_dir = Path(self.camConfig.lidarFilepath or self.filepath or Path.cwd()).parent
        poss_filepath = filedialog.askopenfilename(initialdir=str(init_dir), title='Select LIDAR Truth Points')
        if poss_filepath:
            self.camConfig.lidarFilepath = poss_filepath
            self.updateLidarLabel()
            self.loadTruthPoints()
            self.saveToCache()

    def selectLogFile(self):
        init_dir = Path(self.camConfig.hud_data_filepath or self.filepath or Path.cwd())
        poss_dir = filedialog.askdirectory(initialdir=str(init_dir), title='Select Flight Log Data')
        if poss_dir:
            self.camConfig.hud_data_filepath = poss_dir
            self.updateLogFile()

    def updateLogFile(self):
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

        from SupportModules.LidarTruth import TruthPoints

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
            import copy as _copy
            self.lidarTruthPoints.truthPoints = _copy.deepcopy(obj)
        else:
            LOG.error("Unexpected LiDAR truth data type: %r", type(obj))

    def updateQuality(self, qualityValue: str):
        self.camConfig.export_quality = ExportQuality(qualityValue)
        self.saveToCache()

    def confSlider(self, confValue):
        self.camConfig.yolo_conf = confValue
        self.yoloSession.conf = confValue
        self.confSliderLabel.configure(text='Conf: ' + f'{confValue:.2f}')
        self.saveToCache()

    def iouSlider(self, iouValue):
        self.camConfig.yolo_iou = iouValue
        self.yoloSession.iou = iouValue
        self.iouSliderLabel.configure(text='IOU: ' + f'{iouValue:.2f}')
        self.saveToCache()

    def ingestCalibration(self):

        if not self.calibration.fromBinFile(self.camConfig.calibFilepath) and not self.calibration.fromFile(self.camConfig.calibFilepath):
            if self.selectCalibLabel is not None:
                self.selectCalibLabel.configure(text='No Calibration Found')
                self.after(10, self.update_idletasks())
            return

        if not self.calibration.validCal:
            return

        # Note: this line exists because our aprilTag image was taken at 2848x2848, while calibration images
        # were 1424x1424. Thus, the camera calibration matrix is incorrect for this specific file.
        if self.camConfig.calibFilepath == 'C:/repos/aburn/usr/24WintCalspanFltTest/Alvium_LJ_Calib_2DecSIFTED/calibration.pkl':
            self.calibration.scaleCalibration(2848)

        if self.selectCalibLabel is not None:
            self.selectCalibLabel.configure(text="../" + Path(self.camConfig.calibFilepath).name if self.camConfig.calibFilepath else "../",
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

        self.yoloSession.set_calibration(self.calibration)

        w, h = self.calibration.width, self.calibration.height
        K = self.calibration.getCameraMatrix()
        D = self.calibration.getDistortion()

        newK, _ = getOptimalNewCameraMatrix(K, D, (w, h), alpha=0)

        self.map1, self.map2 = initUndistortRectifyMap(
            K, D, R=None, newCameraMatrix=newK, size=(w, h), m1type=CV_16SC2
        )

        self.saveToCache()

    def scanForCameras(self):
        self.indexDict = {}
        for camera_info in enumerate_cameras(CAP_DSHOW):
            self.indexDict[camera_info.name] = camera_info.index
        with VmbSystem.get_instance() as vmb:
            cams = vmb.get_all_cameras()
            if cams:
                cam = cams[0]
                try:
                    cam._open()
                except vmbpy.c_binding.VmbError as e:
                    LOG.warning(f'Could not open camera: {e}')
                    return
                try:
                    cam.start_streaming(
                        lambda cam, stream, frame: self.display_frame(cam, stream, frame, "Camera Stream"))
                    time.sleep(5)
                    cam.stop_streaming()
                finally:
                    cam._close()

    def display_frame(self, cam, stream, frame, title):
        try:
            numpy_buffer = frame.as_numpy_ndarray()
            if len(numpy_buffer.shape) == 2:
                numpy_buffer = cvtColor(numpy_buffer, COLOR_GRAY2BGR)
            else:
                numpy_buffer = cvtColor(numpy_buffer, COLOR_RGB2BGR)
            imshow(title, resize(numpy_buffer, (864, 864)))
            waitKey(1)
        except vmbpy.c_binding.VmbError as e:
            LOG.error("Error processing frame: %s", e)

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
            initDir = str(Path(self.filepath).parent)
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

        if not self.calibration.validCal:
            self.undistortCheckbox.configure(state='disabled')

        self.undistortCheckbox.grid(row=rowID, column=1, columnspan=2, padx=5, pady=5, sticky='nsew')

        rowID += 1
        self.createDetector()
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

    def _is_completed_row(self, row: dict, n_cls: int) -> bool:
        """Row is 'complete' if it has image_name, image_time and all feat_<cid>_x/y present (even if -1)."""
        if "image_name" not in row or "image_time" not in row:
            return False
        for cid in range(n_cls):
            if f"feat_{cid}_x" not in row or f"feat_{cid}_y" not in row:
                return False
        return True

    def _get_dp_conf_values(self):
        """
        Read the Data Processing confidence list from the GUI and return
        a list of floats.

        Any parse error or out-of-range value => fallback to [0.80].
        """
        default = [0.80]

        raw_var = getattr(self, "_dp_conf_list", None)
        if raw_var is None:
            return default

        raw = (raw_var.get() or "").strip()
        if not raw:
            return default

        try:
            parts = [p.strip() for p in raw.split(",")]
            vals = [float(p) for p in parts if p]

            # no valid numbers?
            if not vals:
                return default

            # ensure all are in [0,1]
            for v in vals:
                if not (0.0 <= v <= 1.0):
                    return default

            return vals

        except Exception:
            return default

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

        # --- Select output CSV ---
        csv_default = getattr(self.camConfig, "dp_output_csv", "")
        self._dp_csv_var = StringVar(value=str(csv_default or ""))

        def _choose_csv():
            p = filedialog.asksaveasfilename(
                title="Select output CSV",
                defaultextension=".csv",
                filetypes=[("CSV", "*.csv")]
            )
            if p:
                self._dp_csv_var.set(p)

        CTkLabel(f, text="Output CSV:").grid(row=2, column=0, padx=12, pady=6, sticky="w")
        CTkEntry(f, textvariable=self._dp_csv_var).grid(row=2, column=1, padx=12, pady=6, sticky="ew")
        CTkButton(f, text="Browse…", command=_choose_csv).grid(row=2, column=2, padx=12, pady=6)

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

        self._dp_cancel_flag = False

        def _bind_dp_str(var, attr_name):
            if var is None:
                return

            def _on_change(*_):
                try:
                    if attr_name == 'dp_conf_list':
                        rounded_list = [round(num, 2) for num in self._get_dp_conf_values()]
                        var.set(str(rounded_list)[1:-1])
                    setattr(self.camConfig, attr_name, var.get())
                    self.saveToCache()
                except Exception:
                    pass

            var.trace_add("write", _on_change)

        _bind_dp_str(self._dp_img_dir_var, "dp_img_dir")
        _bind_dp_str(self._dp_csv_var, "dp_output_csv")
        _bind_dp_str(self._dp_conf_list, "dp_conf_list")
        _bind_dp_str(self._dp_ckptN, "dp_ckptN")
        _bind_dp_str(self._dp_prefetch, "dp_prefetch")

        def runPnP_QnP_on_folders_threaded():
            """
            Kick off SolvePnP/QnP processing in a background thread so the GUI
            stays responsive. Adds progress updates to the status label.
            """

            # --- Read all Tk fields BEFORE launching worker ---
            img_dir_str = (
                    (getattr(self, "_dp_img_dir_var", None) and self._dp_img_dir_var.get().strip())
                    or (getattr(self.camConfig, "imageFilepath", "") or "")
            )
            img_dir = Path(img_dir_str)

            out_csv = (
                    (getattr(self, "_dp_csv_var", None) and self._dp_csv_var.get().strip())
                    or str(img_dir / "yolo_detections.csv")
            )

            conf_list = self._get_dp_conf_values()
            total = len(conf_list)

            # Update UI immediately
            if hasattr(self, "_dp_progress_label"):
                self._dp_progress_label.configure(text=f"Starting SolvePnP/QnP… ({total} files)")

            def update_status(text):
                """Thread-safe UI updater."""
                if hasattr(self, "after") and hasattr(self, "_dp_progress_label"):
                    try:
                        self.after(0, lambda: self._dp_progress_label.configure(text=text))
                    except Exception:
                        pass

            # --- WORKER THREAD ---
            def _worker(out_csv_base, conf_values):
                total = len(conf_values)

                for i, conf in enumerate(conf_values, start=1):
                    if getattr(self, "_dp_cancel_flag", False):
                        update_status("SolvePnP/QnP canceled.")
                        break

                    target = out_csv_base.replace(".csv", f"_conf{conf:.2f}.csv")

                    update_status(f"[{i}/{total}] Checking conf={conf:.2f}…")

                    if not os.path.exists(target):
                        update_status(f"[{i}/{total}] Skipped (missing file)")
                        continue

                    # ------------ NEW: throttled per-row callback ------------
                    last_report = {"t": 0.0, "row": 0}  # small mutable for closure

                    def row_progress(done_rows: int, total_rows: int, img_name: str):
                        now = time.monotonic()

                        # Only update if:
                        #   - first row, or last row, or
                        #   - at least 0.1s has passed, or
                        #   - we've advanced by >= 1% of the file since the last UI update
                        if done_rows == 1 or done_rows == total_rows:
                            do_update = True
                        else:
                            dt = now - last_report["t"]
                            dr = done_rows - last_report["row"]
                            # 1% of file or 0.1s, whichever hits first
                            step_rows = max(1, total_rows // 100)
                            do_update = (dt >= 0.1) or (dr >= step_rows)

                        if not do_update:
                            return

                        # Record the last report
                        last_report["t"] = now
                        last_report["row"] = done_rows

                        def _ui():
                            if hasattr(self, "_dp_progress_label"):
                                self._dp_progress_label.configure(
                                    text=(
                                        f"[{i}/{total}] {os.path.basename(target)} – "
                                        f"{done_rows}/{total_rows} images (last: {img_name})"
                                    )
                                )
                            if hasattr(self, "_dp_progress"):
                                # keep the per-conf progress but smooth within each file
                                base_frac = (i - 1) / max(total, 1)
                                inner_frac = done_rows / max(total_rows, 1)
                                overall = base_frac + inner_frac / max(total, 1)
                                self._dp_progress.set(overall)

                        if hasattr(self, "after"):
                            self.after(0, _ui)

                    # ------------ END throttled callback ------------

                    update_status(f"[{i}/{total}] Running SolvePnP/QnP on {os.path.basename(target)}")
                    try:
                        # your run call, now with a throttled callback
                        self.run_pnp_qnp_from_detection_csv(target, progress_cb=row_progress)
                    except Exception as e:
                        update_status(f"Error on {target}: {e}")
                        continue

                update_status(f"Done! Processed {total} SolvePnP/QnP files.")
                self._dp_cancel_flag = False

            # --- Launch worker ---
            threading.Thread(
                target=_worker,
                args=(out_csv, conf_list),
                daemon=True,
            ).start()

        CTkButton(
            f,
            text="SolvePnP_QnP",
            font=("Segoe UI", 16, "bold"),
            command=runPnP_QnP_on_folders_threaded,
        ).grid(
            row=22, column=0, columnspan=3, padx=12, pady=(16, 8), sticky="w"
        )

        CTkButton(f, text='SolvePnP_QnP', font=("Segoe UI", 16, "bold"),
                      command=lambda: runPnP_QnP_on_folders_threaded()).grid(
            row=22, column=0, columnspan=3, padx=12, pady=(16, 8), sticky="w")

        def _cancel():
            self._dp_cancel_flag = True
            if hasattr(self, "_dp_progress_label"):
                self._dp_progress_label.configure(text="Canceling…")
            if hasattr(self, "_dp_cancel_btn"):
                self._dp_cancel_btn.configure(state="disabled")

        self._dp_run_btn = CTkButton(
            f, text="Run YOLO Batch", fg_color="#2FA572",
            command=self._run_yolo_batch_start
        )
        self._dp_run_btn.grid(row=10, column=0, columnspan=2, padx=12, pady=(16, 12), sticky="ew")

        self._dp_cancel_btn = CTkButton(f, text="Cancel", fg_color="#A52F2F",
                                            command=_cancel)
        self._dp_cancel_btn.grid(row=10, column=2, padx=12, pady=(16, 12), sticky="ew")

    def _write_csv_atomic(self, out_csv: str, columns: list[str], completed_map: dict[str, dict]):
        """Write CSV atomically and sort by numeric portion of image_name."""

        rows = list(completed_map.values())
        df = DataFrame(rows, columns=columns)

        # --- Sort numerically by filename stem (e.g. 1.png, 2.png, 10.png) ---
        def _numeric_key(name: str) -> int:
            try:
                # extract first integer from filename; fall back to 0 if none
                return int(re.search(r"\d+", str(name)).group())
            except Exception:
                return 0

        df = df.sort_values(
            by="image_name",
            key=lambda col: col.map(_numeric_key),
            ignore_index=True,
        )

        tmp = out_csv + ".tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, out_csv)  # atomic replace

    def _run_yolo_batch_start(self):
        if getattr(self, "_dp_worker", None) and self._dp_worker.is_alive():
            return  # already running
        self._dp_cancel_flag = False
        if hasattr(self, "_dp_progress_label"):
            self._dp_progress_label.configure(text="Starting…")
        if hasattr(self, "_dp_progress"):
            self._dp_progress.set(0.0)
        if hasattr(self, "_dp_run_btn"):
            self._dp_run_btn.configure(state="disabled")
        if hasattr(self, "_dp_cancel_btn"):
            self._dp_cancel_btn.configure(state="normal")
        # launch worker
        self._dp_worker = threading.Thread(target=self._run_yolo_batch_worker, daemon=True)
        self._dp_worker.start()

    def _run_yolo_batch_worker(self):
        """
        Worker thread:
          - Resumable, checkpointed, pipelined batch YOLO
          - Producers: disk read + preprocess (CPU), with bounded prefetch
          - Consumer: single GPU session.run
          - UI updates posted via `after(...)`
        """

        # -------------------- helpers (UI-thread posts) --------------------
        def _post_progress(made, total_todo, completed_map, total_all, start, frac_override=None, note=None):
            frac = float(frac_override) if frac_override is not None else (made / float(max(1, total_todo)))
            elapsed = time.monotonic() - start
            eta = (elapsed / max(frac, 1e-9)) * (1.0 - frac)
            pct = int(frac * 100.0 + 0.5)
            mm, ss = int(eta // 60), int(round(eta % 60))
            txt = note or (
                f"conf={current_conf:.2f}  •  "
                f"{made}/{total_todo}  •  {pct}%  •  "
                f"ETA {mm:02d}:{ss:02d}  •  "
                f"(total done: {len(completed_map)}/{all_total})"
            )

            def _ui():
                if hasattr(self, "_dp_progress"):
                    self._dp_progress.set(frac)
                if hasattr(self, "_dp_progress_label"):
                    self._dp_progress_label.configure(text=txt)

            self.after(0, _ui)

        def _post_status(msg):
            def _ui():
                if hasattr(self, "_dp_progress_label"):
                    self._dp_progress_label.configure(text=msg)

            self.after(0, _ui)

        def _post_finish(msg):
            def _ui():
                if hasattr(self, "_dp_progress_label"):
                    self._dp_progress_label.configure(text=msg)
                if hasattr(self, "_dp_run_btn"):
                    self._dp_run_btn.configure(state="normal")
                if hasattr(self, "_dp_cancel_btn"):
                    self._dp_cancel_btn.configure(state="normal")

            self.after(0, _ui)

        # -------------------- config & resume --------------------
        # Paths
        img_dir = Path((getattr(self, "_dp_img_dir_var", None) and self._dp_img_dir_var.get().strip())
                       or (getattr(self.camConfig, "imageFilepath", "") or ""))
        out_csv = (getattr(self, "_dp_csv_var", None) and self._dp_csv_var.get().strip()) or str(
            img_dir / "yolo_detections.csv")

        if not img_dir or not img_dir.exists():
            _post_status("No valid image directory selected.")
            _post_finish("Ready.")
            self._dp_cancel_flag = False
            return

        # Freeze current session thresholds (carry over from main config)

        iou = float(self.yoloSession.iou)
        conf_list = self._get_dp_conf_values()

        _post_status(
            "Running YOLO batch sweep: " +
            ", ".join(f"{c:.2f}" for c in conf_list) +
            f"  (iou={iou:.2f})"
        )

        # Build list of files/times (same as playback)
        self.populate_idsTimes(str(img_dir))
        pairs = list(getattr(self.ImageTimeReader, "idsTimes", []))  # [(path, time), ...]

        if not pairs:
            _post_status("No images found in the selected folder.")
            _post_finish("Ready.")
            self._dp_cancel_flag = False
            return

        # Columns (always complete rows)
        n_cls = self.yoloSession.num_classes

        def _feat_cols(cid: int):
            # If the model has only one class, replace (x,y) with full box (x1,y1,x2,y2)
            if n_cls == 1:
                return [f"feat_{cid}_x1", f"feat_{cid}_y1", f"feat_{cid}_x2", f"feat_{cid}_y2"]
            else:
                return [f"feat_{cid}_x", f"feat_{cid}_y"]

        # Optional undistorted columns — only meaningful for centers, so skip when n_cls==1
        save_ud = bool(getattr(self, "_dp_save_ud", None) and self._dp_save_ud.get()) and (n_cls > 1)

        columns = ["image_name", "image_time"]
        for cid in range(n_cls):
            columns.extend(_feat_cols(cid))
            if save_ud:
                columns.extend([f"feat_{cid}_ud_x", f"feat_{cid}_ud_y"])

        current_conf = None

        for conf in conf_list:
            current_conf = conf
            # each conf gets its own CSV
            out_csv_conf = out_csv.replace(".csv", f"_conf{conf:.2f}.csv")

            # Clear progress for this confidence value
            self.after(0, lambda c=conf: (
                hasattr(self, "_dp_progress_label")
                and self._dp_progress_label.configure(
                    text=f"Preparing batch (conf={c:.2f})…"
                ),
                hasattr(self, "_dp_progress")
                and self._dp_progress.set(0.0)
            ))
            # Resume from existing CSV
            completed_map = {}
            if os.path.exists(out_csv_conf):
                try:
                    prev = read_csv(out_csv_conf)
                    # Normalize any missing columns
                    for col in columns:
                        if col not in prev.columns:
                            prev[col] = (-1.0 if col.startswith("feat_") else None)
                    # Only keep fully-formed rows (all required columns present)
                    need = set(columns)
                    for _, r in prev.iterrows():
                        rd = r.to_dict()
                        if need.issubset(rd.keys()):
                            completed_map[str(rd["image_name"])] = rd
                except Exception as e:
                    _post_status(f"Existing CSV unreadable, starting fresh: {e}")

            # Time map (apply camera->log offset)
            time_offset = float(getattr(self.camConfig, "cam_to_log_time_offset", 0.0))
            time_map = {Path(p).name: (None if t is None else float(t) + time_offset) for (p, t) in pairs}

            # Work list (skip already completed)
            all_total = len(pairs)
            work_items = []
            for p, _t in pairs:
                name = Path(p).name
                if name in completed_map:
                    continue
                work_items.append((p, name))

            total_todo = len(work_items)
            if total_todo == 0:
                # Still rewrite CSV to ensure new columns (e.g., UD) get materialized
                try:
                    self._write_csv_atomic(out_csv_conf, columns, completed_map)
                    self.after(0, lambda: (
                        hasattr(self, "_dp_progress") and self._dp_progress.set(0.0),
                        hasattr(self, "_dp_progress_label") and self._dp_progress_label.configure(
                            text=f"Completed conf={conf:.2f}. Preparing next…"
                        )
                    ))
                    _post_finish(f"Already complete. CSV written: {out_csv_conf}")
                except Exception as e:
                    _post_finish(f"Failed to write CSV: {e}")
                self._dp_cancel_flag = False
                continue

            # Checkpoint config (images per checkpoint)
            try:
                checkpoint_every = max(0, int(str(self._dp_ckptN.get()).strip()))
            except Exception:
                checkpoint_every = 0  # no mid-run checkpoints if not set
            processed_since_ckpt = 0

            # Prefetch / pipeline config
            try:
                prefetch = max(2, int(str(self._dp_prefetch.get()).strip()))
            except Exception:
                prefetch = 32

            import os as _os
            cpu_workers = max(2, min(prefetch, (_os.cpu_count() or 4)))
            q = queue.Queue(maxsize=prefetch)
            producers_done = threading.Event()

            # YOLO dims
            yW, yH = self.yoloSession.yoloSize

            # Optional: matrices for UD points
            if save_ud and hasattr(self, "calibration") and self.calibration is not None:
                import numpy as _np, cv2 as _cv2
                K = self.calibration.getCameraMatrix()
                D = self.calibration.getDistortion()
            else:
                _np = None
                _cv2 = None
                K = D = None

            # -------------------- producer / consumer --------------------
            def _producer_job(path_str: str, name: str):
                if self._dp_cancel_flag:
                    return
                p = Path(path_str)
                if not p.exists():
                    rp = img_dir / p.name
                    if rp.exists():
                        p = rp

                img = imread(str(p), IMREAD_COLOR)
                if img is None:
                    item = (name, None, (0, 0))
                else:
                    H, W = img.shape[:2]
                    try:
                        tensor = self.yoloSession.preprocessImage(img)  # [1,3,h,w] float32
                        item = (name, tensor, (W, H))
                    except Exception:
                        item = (name, None, (W, H))

                # bounded, cancel-aware put
                while not self._dp_cancel_flag:
                    try:
                        q.put(item, timeout=0.05)
                        break
                    except queue.Full:
                        continue

            start = time.monotonic()
            made = 0

            # Start producers
            ex = ThreadPoolExecutor(max_workers=cpu_workers)
            try:
                futures = [ex.submit(_producer_job, p, name) for (p, name) in work_items]

                # watcher that flips when producers finish
                def _watch():
                    wait(futures)
                    producers_done.set()

                threading.Thread(target=_watch, daemon=True).start()

                # single GPU consumer on worker thread
                while True:
                    # stop condition: canceled or producers finished AND queue empty
                    if (self._dp_cancel_flag or producers_done.is_set()) and q.empty():
                        break

                    try:
                        name, tensor, (W, H) = q.get(timeout=0.1)
                    except queue.Empty:
                        # light UI heartbeat
                        _post_progress(made, total_todo, completed_map, all_total, start, note="Working…")
                        continue

                    if tensor is not None:
                        # GPU infer
                        centers, boxes, scores, classes, _dt = self.yoloSession.runOneSession(tensor)

                        # build complete row (raw distorted pixels)
                        rec = {c: -1.0 for cid in range(n_cls) for c in _feat_cols(cid)}
                        sx, sy = (W / float(yW)), (H / float(yH))

                        if n_cls == 1:
                            # For single-class models, store full box geometry (if any)
                            if boxes:
                                x1, y1, x2, y2 = boxes[0]
                                rec["feat_0_x1"] = float(x1) * sx
                                rec["feat_0_y1"] = float(y1) * sy
                                rec["feat_0_x2"] = float(x2) * sx
                                rec["feat_0_y2"] = float(y2) * sy
                            # No UD columns in 1-class mode
                        else:
                            # Multi-class: keep existing center behavior (+ optional UD)
                            found_pts, found_cids = [], []
                            for (cx, cy), cid in zip(centers, classes):
                                cidi = int(cid)
                                x = float(cx) * sx
                                y = float(cy) * sy
                                rec[f"feat_{cidi}_x"] = x
                                rec[f"feat_{cidi}_y"] = y
                                if save_ud:
                                    found_pts.append([x, y])
                                    found_cids.append(cidi)

                            if save_ud and found_pts and _np is not None and _cv2 is not None:
                                pts_np = _np.array(found_pts, dtype=_np.float32).reshape(-1, 1, 2)
                                ud = undistortPoints(pts_np, K, D, P=K).reshape(-1, 2)
                                for (ux, uy), cidi in zip(ud, found_cids):
                                    rec[f"feat_{cidi}_ud_x"] = float(ux)
                                    rec[f"feat_{cidi}_ud_y"] = float(uy)

                        row = {"image_name": name, "image_time": time_map.get(name, None)}
                        row.update(rec)
                        completed_map[name] = row  # overwrite/insert complete row

                    # progress
                    made += 1
                    processed_since_ckpt += 1
                    _post_progress(made, total_todo, completed_map, all_total, start)

                    # checkpoint (atomic + numeric sort) on UI-friendly cadence
                    if checkpoint_every > 0 and not self._dp_cancel_flag and processed_since_ckpt >= checkpoint_every:
                        try:
                            self._write_csv_atomic(out_csv_conf, columns, completed_map)
                            processed_since_ckpt = 0
                            _post_status(f"Checkpoint saved ({len(completed_map)} rows)…")
                        except Exception as e:
                            _post_status(f"Checkpoint save failed: {e}")

            finally:
                # shutdown producers; don't block on cancel
                if self._dp_cancel_flag:
                    ex.shutdown(wait=False, cancel_futures=True)
                else:
                    ex.shutdown(wait=True)

                # final write (always)
                try:
                    self._write_csv_atomic(out_csv_conf, columns, completed_map)
                    msg = ("Partial CSV written (resume later): " + out_csv_conf) if self._dp_cancel_flag else (
                            "Done. CSV written: " + out_csv_conf)
                except Exception as e:
                    msg = f"Failed to write CSV: {e}"

                _post_finish(msg)
                # reset for next run
                self._dp_cancel_flag = False

        _post_finish("All confidence sweeps completed.")
        self._dp_cancel_flag = False
        return

    def run_pnp_qnp_from_detection_csv(
            self,
            csv_path: str,
            out_pnp: str | None = None,
            out_qnp: str | None = None,
            progress_cb: Callable[[int, int, str], None] | None = None
    ) -> None:

        if not getattr(self.calibration, "validCal", False):
            LOG.error("run_pnp_qnp_from_detection_csv: calibration is not valid.")
            return
        if not getattr(self.lidarTruthPoints, "truthPoints", None):
            LOG.error("run_pnp_qnp_from_detection_csv: no LiDAR truth points loaded.")
            return

        csv_path = str(csv_path)
        df = read_csv(csv_path)

        total_rows = int(len(df))
        if total_rows <= 0:
            LOG.warning("run_pnp_qnp_from_detection_csv: %s is empty", csv_path)
            return

        base = Path(csv_path)
        if out_pnp is None:
            out_pnp = str(base.with_name(base.stem + "__pnp.csv"))
        if out_qnp is None:
            out_qnp = str(base.with_name(base.stem + "__qnp.csv"))

        cols = set(df.columns)

        # Discover feature IDs from any feat_* columns
        feat_ids: set[int] = set()
        for col in cols:
            if col.startswith("feat_"):
                parts = col.split("_")
                if len(parts) >= 3:
                    try:
                        feat_ids.add(int(parts[1]))
                    except ValueError:
                        pass
        feat_ids = sorted(feat_ids)

        if not feat_ids:
            LOG.error("run_pnp_qnp_from_detection_csv: no feat_* columns found in %s", csv_path)
            return

        def _valid(v):
            # -1.0 is our "no detection" sentinel
            return (v is not None) and (not isna(v)) and (float(v) > -0.5)

        K = self.calibration.getCameraMatrix()
        D_full = self.calibration.getDistortion()
        truth_dict = self.yoloSession.reader.idsNamesLocs

        pnp_rows = []
        qnp_rows = []

        for idx, (_, row) in enumerate(df.iterrows(), start=1):

            image_name = row.get("image_name", "")
            image_time = row.get("image_time", np.nan)

            # Optional incremental progress callback
            if progress_cb is not None:
                try:
                    progress_cb(idx, total_rows, str(image_name))
                except Exception:
                    # Don't let UI issues kill the batch
                    pass

            # Collect 2D points: prefer undistorted if present
            ud_centers = []
            ud_ids = []
            plain_centers = []
            plain_ids = []

            for fid in feat_ids:
                # 1) Undistorted points (multi-class with save_ud enabled)
                ux = row.get(f"feat_{fid}_ud_x", None)
                uy = row.get(f"feat_{fid}_ud_y", None)
                if _valid(ux) and _valid(uy):
                    ud_centers.append([float(ux), float(uy)])
                    ud_ids.append(fid)
                    continue

                # 2) Raw centers (multi-class normal case)
                x = row.get(f"feat_{fid}_x", None)
                y = row.get(f"feat_{fid}_y", None)
                if _valid(x) and _valid(y):
                    plain_centers.append([float(x), float(y)])
                    plain_ids.append(fid)
                    continue

                # 3) Box geometry -> center (single-class mode)
                x1 = row.get(f"feat_{fid}_x1", None)
                y1 = row.get(f"feat_{fid}_y1", None)
                x2 = row.get(f"feat_{fid}_x2", None)
                y2 = row.get(f"feat_{fid}_y2", None)
                if _valid(x1) and _valid(y1) and _valid(x2) and _valid(y2):
                    cx = 0.5 * (float(x1) + float(x2))
                    cy = 0.5 * (float(y1) + float(y2))
                    plain_centers.append([cx, cy])
                    plain_ids.append(fid)

            # Decide which set to use
            if len(ud_centers) >= 6:
                centers_use = np.asarray(ud_centers, dtype=np.float32)
                ids_use = ud_ids
                use_ud = True
            elif len(plain_centers) >= 6:
                centers_use = np.asarray(plain_centers, dtype=np.float32)
                ids_use = plain_ids
                use_ud = False
            else:
                # Not enough features -> still emit NaNs
                pnp_rows.append({
                    "image_name": image_name,
                    "image_time": image_time,
                    "pnp_qw": np.nan, "pnp_qx": np.nan, "pnp_qy": np.nan, "pnp_qz": np.nan,
                    "pnp_x": np.nan, "pnp_y": np.nan, "pnp_z": np.nan,
                })
                qnp_rows.append({
                    "image_name": image_name,
                    "image_time": image_time,
                    "qnp_qw": np.nan, "qnp_qx": np.nan, "qnp_qy": np.nan, "qnp_qz": np.nan,
                    "qnp_x": np.nan, "qnp_y": np.nan, "qnp_z": np.nan,
                })
                continue

            # Map IDs -> 3D truth points and drop any unknown IDs
            obj_pts = []
            keep_centers = []
            for key, (u, v) in zip(ids_use, centers_use):
                obj_pts.append(truth_dict[key][2:])
                keep_centers.append([u, v])

            if len(obj_pts) < 6:
                pnp_rows.append({
                    "image_name": image_name,
                    "image_time": image_time,
                    "pnp_qw": np.nan, "pnp_qx": np.nan, "pnp_qy": np.nan, "pnp_qz": np.nan,
                    "pnp_x": np.nan, "pnp_y": np.nan, "pnp_z": np.nan,
                })
                qnp_rows.append({
                    "image_name": image_name,
                    "image_time": image_time,
                    "qnp_qw": np.nan, "qnp_qx": np.nan, "qnp_qy": np.nan, "qnp_qz": np.nan,
                    "qnp_x": np.nan, "qnp_y": np.nan, "qnp_z": np.nan,
                })
                continue

            obj_pts = np.asarray(obj_pts, dtype=np.float32)
            img_pts = np.asarray(keep_centers, dtype=np.float32)

            # ----------------- PnP (OpenCV, RANSAC) -----------------
            # If we used undistorted points (via undistortPoints P=K),
            # we should pass zero distortion here.
            distCoeffs = np.zeros((5, 1), dtype=np.float32) if use_ud else D_full

            try:
                ret, rvec, tvec, inliers = solvePnPRansac(
                    objectPoints=obj_pts,
                    imagePoints=img_pts,
                    cameraMatrix=K,
                    distCoeffs=distCoeffs,
                    flags=SOLVEPNP_ITERATIVE
                )
            except error as e:
                LOG.error("solvePnPRansac failed for %s: %s", image_name, e)
                ret = False

            if ret:
                # Same convention as your existing pnpLidarPoints:
                quatPnP, vectPnP = q.fromOpenCV_toAftr_rvec(rvec, tvec)
                pnp_rows.append({
                    "image_name": image_name,
                    "image_time": image_time,
                    "pnp_qw": float(quatPnP.s),
                    "pnp_qx": float(quatPnP.vec[0]),
                    "pnp_qy": float(quatPnP.vec[1]),
                    "pnp_qz": float(quatPnP.vec[2]),
                    "pnp_x": float(vectPnP[0]),
                    "pnp_y": float(vectPnP[1]),
                    "pnp_z": float(vectPnP[2]),
                })
            else:
                pnp_rows.append({
                    "image_name": image_name,
                    "image_time": image_time,
                    "pnp_qw": np.nan, "pnp_qx": np.nan, "pnp_qy": np.nan, "pnp_qz": np.nan,
                    "pnp_x": np.nan, "pnp_y": np.nan, "pnp_z": np.nan,
                })

            # ----------------- QnP (your solveQnP path) -----------------
            try:
                quatQ, vectQ = solveQnP(obj_pts, img_pts, self.calibration, None)

                # Same aircraft-from-CV transform as qnpLidarPoints
                q_aftr_from_cv = mat2quat(np.array([
                    [0., 0., 1.],
                    [-1., 0., 0.],
                    [0., -1., 0.],
                ], float))

                vectQ = q_aftr_from_cv * vectQ
                quatQ = q_aftr_from_cv * quatQ

                qnp_rows.append({
                    "image_name": image_name,
                    "image_time": image_time,
                    "qnp_qw": float(quatQ.s),
                    "qnp_qx": float(quatQ.vec[0]),
                    "qnp_qy": float(quatQ.vec[1]),
                    "qnp_qz": float(quatQ.vec[2]),
                    "qnp_x": float(vectQ[0]),
                    "qnp_y": float(vectQ[1]),
                    "qnp_z": float(vectQ[2]),
                })
            except Exception as e:
                LOG.error("solveQnP failed for %s: %s", image_name, e)
                qnp_rows.append({
                    "image_name": image_name,
                    "image_time": image_time,
                    "qnp_qw": np.nan, "qnp_qx": np.nan, "qnp_qy": np.nan, "qnp_qz": np.nan,
                    "qnp_x": np.nan, "qnp_y": np.nan, "qnp_z": np.nan,
                })

        DataFrame(pnp_rows).to_csv(out_pnp, index=False)
        DataFrame(qnp_rows).to_csv(out_qnp, index=False)
        LOG.info("Offline PnP results written to %s", out_pnp)
        LOG.info("Offline QnP results written to %s", out_qnp)

    def setup_playbackFrame(self):
        rowID = 0
        self.update_playbackMenu()
        playbackLabel = CTkLabel(self.playback_frame, textvariable=self.playbackModeText)
        playbackLabel.grid(row=rowID, column=0, sticky='w', padx=5, pady=5)

    def shutdown(self):
        self.shutting_down = True
        self.recordOff()
        self.startStreamOffBool()
        # Safely wait for window to be gone
        while True:
            try:
                vis = getWindowProperty(self.windowName, WND_PROP_VISIBLE)
                if vis <= 0:
                    break
            except Exception:
                break
            time.sleep(0.1)

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
            offset_dict = read_csv(directory / '__TIME_OFFSET.csv')
            self.camConfig.cam_to_log_time_offset = float(offset_dict['offset'][0])
        except FileNotFoundError:
            self.camConfig.cam_to_log_time_offset = 0.0

        cv_imgs = []
        start = self.camConfig.start_export_idx
        end = self.camConfig.end_export_idx + 1
        for idx, img_path in zip(range(start, end), paths[start:end]):
            frame = imread(str(img_path))
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
            frames = self._gather_annotated_frames()
            make_gif(frames, 10, infinite=True, quality=self.camConfig.export_quality)
        finally:
            self.after(0, self._exportToGifOrVid_done)

    def exportToVid_worker(self):
        try:
            frames = self._gather_annotated_frames()
            h, w = frames[0].shape[:2]
            fourcc = VideoWriter.fourcc(*'mp4v')
            out = VideoWriter('output_video.mp4', fourcc, 10, (w, h))
            for f in frames:
                out.write(f)
            out.release()
        finally:
            self.after(0, self._exportToGifOrVid_done)

    def _exportToGifOrVid_done(self):
        self.exportToGifButton.configure(text="Export to GIF", state='normal', fg_color=CTK_GREEN)
        self.exportToVidButton.configure(text="Export to Vid", state='normal', fg_color=CTK_GREEN)
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
                                             fg_color=CTK_GREEN,
                                             hover_color='navy')
        self.startStreamButton.configure(command=self.startStreamOffBool, text='Stop Streaming', fg_color=CTK_GREEN,
                                         hover_color='navy')
        self.multiImageTextButton.configure(command=self.startStreamOffBool, fg_color=CTK_GREEN, hover_color='navy')

        self.selectCameraCombo.configure(state='disabled')

        self.streamOrImgCombo.configure(state='disabled')

        self.threadStopper = ThreadStopper()
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def startStreamOffBool(self):
        self.showWindow = False

    def startStreamOff(self):
        waitKey(1)

        self.threadStopper.set()
        destroyAllWindows()

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
                    command=self.startStreamOn, fg_color=BUTTON_RED, hover_color='blue',
                    text=os.path.basename(self.camConfig.imageFilepath)
                )
            self.startStreamButton.configure(command=self.startStreamOn, fg_color=BUTTON_RED, hover_color='blue')
            self.multiImageTextButton.configure(command=self.startStreamOn, fg_color=BUTTON_RED, hover_color='blue')

            self.selectCameraCombo.configure(state='normal')
            self.startStreamButton.configure(text='Start Stream')
            self.streamOrImgCombo.configure(state='normal')

            self.showWindow = False

        destroyAllWindows()

    def recordOn(self):
        self.recordButton.configure(fg_color='green', text='Saving Imagery', hover_color='navy', command=self.recordOff)
        self.recording = True

    def recordOff(self):
        self.recordButton.configure(fg_color=BUTTON_RED, text=f'Saved Imagery: #{self.img_idx}', hover_color='blue',
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
        self.detector = aruco.ArucoDetector(self.arucoDict, self.arucoParams)

    def run_detectSingleImage(self):
        namedWindow(self.windowName, WINDOW_NORMAL)
        frame = imread(str(Path(self.camConfig.imageFilepath)))
        while (not self.threadStopper.is_set()
               and getWindowProperty(self.windowName, WND_PROP_VISIBLE) > 0
               and self.showWindow):

            self.analyze_image(frame)

            key = waitKey(1)
            if key == 27:
                self.threadStopper.set()
                break

        destroyAllWindows()
        self.after(0, self._on_worker_exit)

    @staticmethod
    def convert_cv_to_pil(img):
        return fromarray(cvtColor(img, COLOR_BGR2RGB))

    def run(self):

        if self.camConfig.imageSource == ImageSource.Camera_Stream:
            self.run_video_stream()
        elif self.camConfig.imageSource == ImageSource.Stream_from_Folder:
            # self.profile_run_folder = True
            # self.run_folder_reader_profiled()
            self.run_folder_reader()
        elif self.camConfig.imageSource == ImageSource.Static_Image:
            self.run_detectSingleImage()

    def run_video_stream(self):

        self.vc = VideoCapture(self.camConfig.cam_index, CAP_DSHOW)
        self.vc.set(CAP_PROP_FPS, 60)

        namedWindow(self.windowName, WINDOW_NORMAL)
        rval, self.curr_frame = self.vc.read()
        if rval:
            resizeWindow(self.windowName, self.curr_frame.shape[1], self.curr_frame.shape[0])
            self.lastHeight = self.curr_frame.shape[0]
            self.lastWidth = self.curr_frame.shape[1]

        while (rval and not self.threadStopper.is_set() and
               getWindowProperty(self.windowName, WND_PROP_VISIBLE) > 0 and
               self.showWindow and not self.making_gifOrVid):
            rval, frame = self.vc.read()
            self.analyze_image(frame)
            key = waitKey(1)
            if key == 27:  # exit on ESC
                self.threadStopper.set()
                break

        # Minimal teardown in the worker; the UI thread will handle buttons/state.
        if self.vc is not None and self.vc.isOpened():
            self.vc.release()
            self.vc = None
        destroyAllWindows()
        self.after(0, self._on_worker_exit)
        return

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
            image_list = natural_sort([str(p) for p in image_list])
            for image in image_list:
                self.ImageTimeReader.idsTimes.append([image, None])

    @staticmethod
    def _poll_keys(max_ms: int = 8) -> list[int]:
        # One-shot poll: wait up to max_ms for a key
        k = waitKey(max_ms) & 0xFF
        if k not in (0, 0xFF, 255, -1):
            return [k]
        return []

    @staticmethod
    def load_time_offset(directory):
        try:
            offset_dict = read_csv(directory / "__TIME_OFFSET.csv")
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
        destroyAllWindows()
        namedWindow(self.windowName, WINDOW_NORMAL)

        directory = Path(self.camConfig.imageFilepath).parent

        paths, t = self._build_sequence_and_timebase(directory)

        num_images = len(paths)

        self.playback.speed = 1  # negative=rewind, 0=freeze, positive=forward
        self.pause = False
        self.last_nonzero_sign = 1
        last_speed = self.playback.speed
        curr_idx = 0

        # --- PAUSED CACHE: keep 1 frame while paused to avoid refetch spam ---
        self.pauseCache.clear()

        # time offset
        self.camConfig.cam_to_log_time_offset = self.load_time_offset(directory)

        # --- start background loader ---
        loader = imgBuf(
            filepaths=[str(p) for p in paths],
            max_buffer=96,
            preprocess=None,
            start_index=0,
            loop=True,
            read_flags=IMREAD_COLOR,
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
                   and getWindowProperty(self.windowName, WND_PROP_VISIBLE)
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

                # --- Key handling: edge-triggered dispatcher ---
                while pending_keys:
                    key = pending_keys.pop(0)

                    if key in (255, 0xFF, 0, -1):
                        continue

                    # Decide edge vs repeat behavior:
                    if key in _EDGE_KEYS:
                        if not _is_edge_allowed(key, last_edge_time):
                            continue  # skip if within cooldown
                    # if key in _REPEAT_KEYS: let every event through (no gating)

                    # --- dispatch ---
                    if key == ord('f'):
                        self.pause = False
                        self._on_toggle_fps_mode()
                        wall_start = self._reanchor_on_mode_change(
                            new_mode=self.camConfig.playback_mode,
                            curr_idx=curr_idx,
                            t=t
                        )
                        self.update_playbackMenu()
                        self.saveToCache()

                    elif key == ord('c'):
                        curr_idx = self._on_step_forward(curr_idx, num_images)
                        self.pause = True
                        self.pauseCache.clear()
                        self.camConfig.playback_mode = PlaybackSpeed.Fixed_fps
                        loader.seek(curr_idx, clear_buffer=True)
                        self.update_playbackMenu()

                    elif key == ord('z'):
                        curr_idx = self._on_step_back(curr_idx)
                        self.pause = True
                        self.pauseCache.clear()
                        self.camConfig.playback_mode = PlaybackSpeed.Fixed_fps
                        loader.seek(curr_idx, clear_buffer=True)
                        self.update_playbackMenu()

                    elif key == ord(' '):
                        wall_start = self._on_toggle_pause(curr_idx, t, wall_start)
                        self.update_playbackMenu()

                    elif key == ord('d'):
                        wall_start = self._on_speed_up(curr_idx=curr_idx, t=t)
                        self.update_playbackMenu()

                    elif key == ord('a'):
                        wall_start = self._on_speed_down(curr_idx=curr_idx, t=t)
                        self.update_playbackMenu()

                    elif key == ord('w'):
                        self._on_toggle_overlays()

                    elif key == ord('s'):
                        self._on_mark_start(curr_idx)

                    elif key == ord('e'):
                        self._on_mark_end(curr_idx)

                    elif key == ord('r'):
                        wall_start = self._on_reverse(
                            loader=loader, curr_idx=curr_idx, t=t
                        )
                        self.update_playbackMenu()

                    elif key == ord('b'):
                        self.hud_marker.cam_bank_offset -= 0.1
                    elif key == ord('n'):
                        self.hud_marker.cam_bank_offset += 0.1

                    elif key == ord(";"):
                        self._on_adjust_offset(-0.01)
                    elif key == ord("'"):
                        self._on_adjust_offset(+0.01)
                    elif key == ord(':'):
                        self._on_adjust_offset(-0.10)
                    elif key == ord('"'):
                        self._on_adjust_offset(+0.10)
                    elif key == ord('['):
                        self._on_adjust_offset(-1.00)
                    elif key == ord(']'):
                        self._on_adjust_offset(+1.00)
                    elif key == ord('{'):
                        self._on_adjust_offset(-10.00)
                    elif key == ord('}'):
                        self._on_adjust_offset(+10.00)
                    elif key == ord('p'):
                        self._on_persist_offset()

                    elif key == 27:  # ESC
                        self.threadStopper.set()
                        break

                    while self.making_gifOrVid:
                        time.sleep(0.1)

                pending_keys.extend(self._poll_keys(1))
                if getWindowProperty(self.windowName, WND_PROP_VISIBLE) <= 0:
                    self.threadStopper.set()
                    break

        finally:
            destroyAllWindows()
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
                fg_color=BUTTON_RED, hover_color='blue',
                text=os.path.basename(self.camConfig.imageFilepath)
            )
        else:
            self.singleImageTextButton.configure(
                command=self.startStreamOn,
                fg_color=BUTTON_RED, hover_color='blue',
                text='No Image Selected'
            )

        self.startStreamButton.configure(
            command=self.startStreamOn,
            fg_color=BUTTON_RED, hover_color='blue',
            text='Start Stream'
        )
        self.multiImageTextButton.configure(
            command=self.startStreamOn,
            fg_color=BUTTON_RED, hover_color='blue'
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
        out_csv = Path(self.camConfig.hud_data_filepath) / "__TIME_OFFSET.csv"
        DataFrame({"offset": [self.hud_marker.offset]}).to_csv(out_csv, index=False)

        self.camConfig.cam_to_log_time_offset = 0.0

    def analyze_image(self, frame, img_time=None, name=None, display_in_realtime=True, box_around=False):

        if frame is None:
            return

        self.curr_frame_gray = None

        # Sets self.curr_frame to (potentially undistorted) frame, and makes a copy onto self.markup_frame
        if self.calibration.validCal and self.camConfig.undistort:
            self.undistort(frame.copy())
        else:
            self.curr_frame = frame.copy()
        self.markup_frame = self.curr_frame.copy()

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

        if self.camConfig.yoloInference:
            self.run_yolo()
        else:
            self.last_bounding_box_size = None
            self.last_yolo_center = None

        if self.camConfig.factor_graph:
            self.factor_graph(img_time)
        else:
            self.last_yolo_3d_estimate = None

        if self.camConfig.phase_correlation:
            self.phase_correlation()

        if self.camConfig.hud and img_time is not None:
            self.hud_marker.draw_HUD(self.markup_frame, img_time)

        if box_around:
            x, y, _ = self.markup_frame.shape
            rectangle(self.markup_frame, (0, 0), (x - 1, y - 1), HUD_YELLOW, 10)

        height = 0
        if self.camConfig.imageSource == ImageSource.Stream_from_Folder:
            (width, height), base = getTextSize(os.path.basename(name), FONT_HERSHEY_SIMPLEX, med_text(self.curr_frame.shape[0]), 4)
            img_w, img_h, *_ = self.curr_frame.shape
            putText(self.markup_frame, os.path.basename(name), (img_w - width, img_h - height),
                        FONT_HERSHEY_SIMPLEX, med_text(self.curr_frame.shape[0]), HUD_GREEN, 2)
        if img_time is not None:
            time_str = f"Flight Time: {img_time:.2f}"  # + 173.11338 - 11.658461:.2f}"
            (time_width, time_height), base = getTextSize(time_str, FONT_HERSHEY_SIMPLEX, med_text(self.curr_frame.shape[0]), 4)
            img_w, img_h, *_ = self.curr_frame.shape
            putText(self.markup_frame, time_str, (img_w - time_width, img_h - time_height - height - 10),
                        FONT_HERSHEY_SIMPLEX, med_text(self.curr_frame.shape[0]), HUD_GREEN, 2)

        if display_in_realtime:
            if self.camConfig.imageSource == ImageSource.Stream_from_Folder:
                (h, w) = self.markup_frame.shape[:2]
                self.lowPassFPS = 0.925 * self.lowPassFPS + 0.075 * self.curr_fps
                putText(self.markup_frame, f"Offset: {self.camConfig.cam_to_log_time_offset:+.2f}s",
                            (int(0.015 * w), int(0.030 * h)), FONT_HERSHEY_SIMPLEX, med_text(self.curr_frame.shape[0]), HUD_YELLOW, 2)
                putText(self.markup_frame,
                            f'Realtime: {self.camConfig.rt_speed:.2f}' if self.camConfig.playback_mode == PlaybackSpeed.Real_time else f'FPS: {self.lowPassFPS:.2f}/{self.camConfig.target_fps:.2f}',
                            (int(0.015 * w), int(0.060 * h)), FONT_HERSHEY_SIMPLEX, med_text(self.curr_frame.shape[0]), HUD_YELLOW, 2)
            self.cleanup()

        if self.printLidar:
            self.print_pnp_results()

        if not display_in_realtime:
            return self.markup_frame

    def inpaint_apriltags(self,
                          radius_px: int = 3,
                          dilate_px: int = 2,
                          method: int = INPAINT_TELEA,
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

            fillConvexPoly(mask_roi, pts_int, 255)

            # optional dilation to cover borders
            if dilate_px > 0:
                k = getStructuringElement(
                    MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)
                )
                mask_roi = dilate(mask_roi, k)

            # slice out the ROI from the big frame
            frame_roi = self.markup_frame[y_min:y_max + 1, x_min:x_max + 1]

            # inpaint only this small region
            inpainted_roi = inpaint(frame_roi, mask_roi, radius_px, method)

            if feather:
                blur_ks = max(3, 2 * radius_px + 1)
                soft = GaussianBlur(mask_roi, (blur_ks, blur_ks), 0).astype(np.float32) / 255.0
                soft = soft[..., None]  # (H,W,1)

                base = frame_roi.astype(np.float32)
                inp = inpainted_roi.astype(np.float32)
                blended_roi = (soft * inp + (1.0 - soft) * base).astype(np.uint8)

                self.markup_frame[y_min:y_max + 1, x_min:x_max + 1] = blended_roi
            else:
                self.markup_frame[y_min:y_max + 1, x_min:x_max + 1] = inpainted_roi

    def print_pnp_results(self):
        np.set_printoptions(precision=5, threshold=sys.maxsize, suppress=True)

        points = None
        if self.centers is not None and len(self.centers) >= 6:
            truthPoints = copy.copy(self.lidarTruthPoints.truthPoints)
            points = []

            removeIDs = []
            for idx, detectID in enumerate(self.detectIDS):
                try:
                    points.append(truthPoints[str(detectID)])
                except KeyError as e:
                    removeIDs.append(idx)

            centers = self.centers.copy()
            for id in reversed(removeIDs):
                centers = np.delete(centers, id, axis=0)
            points = np.array(points)

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
                img_points, _ = fisheye.projectPoints(
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
        return remap(
            frame, self.map_x[face], self.map_y[face],
            interpolation=INTER_LINEAR,
            borderMode=BORDER_CONSTANT,
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
            self.curr_frame = remap(frame, self.map1, self.map2, interpolation=INTER_LINEAR,
                                        borderMode=BORDER_CONSTANT)

    def applyKernel(self):
        if self.camConfig.processingKernel != ImageKernel.Gabor and self.GaborGUI is not None:
            self.GaborGUI.close()
            self.GaborGUI = None

        if self.camConfig.processingKernel == ImageKernel.Gabor:
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
            self.curr_frame_gray = cvtColor(self.curr_frame, COLOR_BGR2GRAY)
        harris_corners = cornerHarris(self.curr_frame_gray, 3, 3, 0.05)

        self.markup_frame[harris_corners > 0.025 * harris_corners.max()] = [0, 255, 255]

    def detectAprilTags(self, scale: float = 0.6):
        """
        Faster AprilTag detection:
          - detect on downscaled image
          - upscale corners
          - refine on full-res gray image with cornerSubPix
        """
        if self.curr_frame_gray is None:
            self.curr_frame_gray = cvtColor(self.curr_frame, COLOR_BGR2GRAY)
        if self.detector is None:
            return

        gray_full = self.curr_frame_gray
        h, w = gray_full.shape[:2]

        # 1) Downscale for detection
        if not (0.2 <= scale < 1.0):
            scale = 0.6
        small = resize(gray_full, (int(w * scale), int(h * scale)),
                           interpolation=INTER_AREA)

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
            TERM_CRITERIA_EPS + TERM_CRITERIA_MAX_ITER,
            20,  # max iterations
            0.01  # epsilon
        )
        cornerSubPix(gray_full, all_pts, (5, 5), (-1, -1), criteria)

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
                polylines(self.markup_frame, polyline, True, HUD_GREEN, 4, lineType=FILLED)
                putText(self.markup_frame, str(idx[0]), tuple(pixCenter),
                            FONT_HERSHEY_SIMPLEX, small_text(self.curr_frame.shape[0]), HUD_GREEN, 4)
                putText(self.markup_frame, str(idx[0]), tuple(pixCenter),
                            FONT_HERSHEY_SIMPLEX, small_text(self.curr_frame.shape[0]), (0, 0, 0), 1)

            self.detectIDS.append(idx)

            if self.centers is None:
                self.centers = np.array(pixCenter, dtype=np.float32)
            else:
                self.centers = np.vstack((self.centers, pixCenter.astype(np.float32)))

    def pnpLidarPoints(self):

        if self.centers is not None and len(self.centers) >= 6:
            truthPoints = copy.copy(self.lidarTruthPoints.truthPoints)
            points = []
            distParams = np.zeros((5,))  # use image undistort instead

            removeIDs = []
            for idx, detectID in enumerate(self.detectIDS):
                try:
                    points.append(truthPoints[str(detectID)])
                except KeyError as e:
                    removeIDs.append(idx)

            centers = self.centers.copy()
            for id in reversed(removeIDs):
                centers = np.delete(centers, id, axis=0)
            points = np.array(points)

            if len(points) < 6:
                return

            ret, rvec, tvec = solvePnP(objectPoints=points,
                                           imagePoints=centers,
                                           cameraMatrix=self.calibration.getCameraMatrix(),
                                           distCoeffs=distParams,
                                           flags=SOLVEPNP_ITERATIVE)

            if ret:
                projectedPoints_orig, _ = projectPoints(self.lidarTruthPoints.getTruthPointsNumpy(),
                                                            rvec=rvec,
                                                            tvec=tvec,
                                                            cameraMatrix=self.calibration.getCameraMatrix(),
                                                            distCoeffs=distParams)

                self.plotOnImg(projectedPoints_orig[:, 0, :].astype(int),
                               list(self.lidarTruthPoints.getTruthPointsDict().keys()), (255, 255, 0))

                quatPnP, vectPnP = q.fromOpenCV_toAftr_rvec(rvec, tvec)

                self.pnpResult = (quatPnP, vectPnP)

                putText(self.markup_frame, 'Orientation (quat) From LiDAR: ' + format(quatPnP, 'ijk.6f'), (50, 75),
                            FONT_HERSHEY_DUPLEX, small_text(self.markup_frame.shape[0]),
                            (255, 255, 0), 3,
                            LINE_AA)
                putText(self.markup_frame, 'Location From LiDAR: ' + np.array2string(vectPnP),
                            (50, 150), FONT_HERSHEY_DUPLEX, small_text(self.markup_frame.shape[0]),
                            (255, 255, 0), 3,
                            LINE_AA)

    def qnpLidarPoints(self):

        if self.centers is not None and len(self.centers) >= 6:
            truthPoints = copy.copy(self.lidarTruthPoints.truthPoints)

            points = []
            # distParams = np.zeros((5,))  # use image undistort instead

            removeIDs = []
            for idx, detectID in enumerate(self.detectIDS):
                try:
                    points.append(truthPoints[str(detectID)])
                except KeyError as e:
                    removeIDs.append(idx)

            centers = self.centers.copy()
            for idx in reversed(removeIDs):
                centers = np.delete(centers, idx, axis=0)
            points = np.array(points)

            if len(points) < 6:
                return

            quat, vect = solveQnP(points, centers, self.calibration, None)
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
            putText(self.markup_frame, 'Orientation (quat) From LiDAR: ' + format(quat, 'ijk.6f'), (50, 225),
                        FONT_HERSHEY_DUPLEX,
                        small_text(self.markup_frame.shape[0]),
                        (255, 255, 0), 3,
                        LINE_AA)
            putText(self.markup_frame, 'Location From LiDAR: ' + np.array2string(vect), (50, 300),
                        FONT_HERSHEY_DUPLEX,
                        small_text(self.markup_frame.shape[0]),
                        (255, 255, 0), 3,
                        LINE_AA)

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
            self.curr_frame_gray = cvtColor(self.curr_frame, COLOR_BGR2GRAY)

        edges = Canny(self.curr_frame_gray, 100, 200, apertureSize=3)

        lines = HoughLinesP(edges, 1, np.pi / 180.0, 50,
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

        line(self.markup_frame, (x1, y1), (x2, y2), color, 2)

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
                self.confSlider(self.yoloSession.conf)

            self.markup_frame = dim_except_circle(self.markup_frame, self.current_center_est, 3.0 * self.radius, 0.00)
            self.markup_frame = dim_except_circle(self.markup_frame, self.current_center_est, 1.5 * self.radius, 0.50)

            self.radius = min(800.0, self.radius + 12.0)
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

    def run_yolo(self):
        '''
        Runs YOLO on subsequent images. If the yolo model is single featured, and the object is estimated less than
        100 meters away, then it updates this class's estimation of the solution.
        :return: None, but does adjust
        '''
        (self.markup_frame, rvec_tvec), output = self.yoloSession.inferOnImage(self.markup_frame, self.markup_frame,
                                                                               self.camConfig.yoloBiasTracking)

        centers, boxes, scores, class_ids, time = output

        if rvec_tvec is not None:
            self.last_yolo_3d_estimate = np.squeeze(rvec_tvec[1])

        elif len(centers) > 0 and self.yoloSession.reader.numClasses == 1:
            best_idx = scores.index(max(scores))
            img_yolo_x_correction = self.curr_frame.shape[0] / self.yoloSession.reader.imageSize
            img_yolo_y_correction = self.curr_frame.shape[1] / self.yoloSession.reader.imageSize

            self.last_bounding_box_size = ((boxes[best_idx][2] - boxes[best_idx][0]) * img_yolo_x_correction,
                                           (boxes[best_idx][3] - boxes[best_idx][1]) * img_yolo_y_correction)
            self.last_yolo_center = centers[best_idx]

            self.last_yolo_center[0] = int(
                self.last_yolo_center[0] * img_yolo_x_correction)
            self.last_yolo_center[1] = int(
                self.last_yolo_center[1] * img_yolo_y_correction)

            K = self.calibration.getCameraMatrix()
            # d = self.calibration.getDistortion()  # Presume undistorted image
            twoD_points = np.array([self.last_yolo_center[0], self.last_yolo_center[1], 1.0])
            # dist_est = 2.0 / (
            #         self.last_bounding_box_size[0] / self.curr_frame.shape[0] + self.last_bounding_box_size[1] /
            #         self.curr_frame.shape[1])
            # dist_est = 2.0 / (self.last_bounding_box_size[0] + self.last_bounding_box_size[1])
            dist_est = self.calibration.fx * 4.07 / (self.last_bounding_box_size[0])

            if self.check_above_horizon(self.last_yolo_center):
                self.last_yolo_3d_estimate = np.linalg.inv(K).dot(twoD_points) * dist_est
                w, h, _ = self.curr_frame.shape
                putText(self.markup_frame, 'BB-Width Solution', (25, w - 75), FONT_HERSHEY_SIMPLEX,
                            med_text(self.markup_frame.shape[0]), (50, 255, 255), 1)
                putText(self.markup_frame,
                            f'x:{self.last_yolo_3d_estimate[0]:.3f}, y:{self.last_yolo_3d_estimate[1]:.3f}, z:{self.last_yolo_3d_estimate[2]:.3f}',
                            (25, w - 50),
                            FONT_HERSHEY_SIMPLEX, med_text(self.markup_frame.shape[0]), (50, 255, 255), 1)
                # circle(self.markup_frame, (int(self.last_yolo_center[0]), int(self.last_yolo_center[1])),
                #            3, (255, 0, 255), 3)
                self.current_center_est = ((self.current_center_est[0] * 2.0 + centers[best_idx][0]) / 3.0,
                                           (self.current_center_est[1] * 2.0 + centers[best_idx][1]) / 3.0)
                return

        self.last_bounding_box_size = None
        self.last_yolo_center = None

    def factor_graph(self, time):
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
            circle(self.markup_frame, (int(self.curr_FG_pixel[0]), int(self.curr_FG_pixel[1])), size, (0, 0, 0),
                       thickness)
            line(self.markup_frame, [int(self.curr_FG_pixel[0]) + size, int(self.curr_FG_pixel[1])],
                     [int(self.curr_FG_pixel[0]) - size, int(self.curr_FG_pixel[1])], (0, 0, 0), thickness)
            line(self.markup_frame, [int(self.curr_FG_pixel[0]), int(self.curr_FG_pixel[1]) + size],
                     [int(self.curr_FG_pixel[0]), int(self.curr_FG_pixel[1]) - size], (0, 0, 0), thickness)
            putText(self.markup_frame, 'Factor Graph Solution', (25, h - 125), FONT_HERSHEY_SIMPLEX,
                        med_text(self.markup_frame.shape[0]), (0, 0, 0), thickness)

            thickness = 1
            circle(self.markup_frame, (int(self.curr_FG_pixel[0]), int(self.curr_FG_pixel[1])), size, color,
                       thickness)
            line(self.markup_frame, [int(self.curr_FG_pixel[0]) + size, int(self.curr_FG_pixel[1])],
                     [int(self.curr_FG_pixel[0]) - size, int(self.curr_FG_pixel[1])], color, thickness)
            line(self.markup_frame, [int(self.curr_FG_pixel[0]), int(self.curr_FG_pixel[1]) + size],
                     [int(self.curr_FG_pixel[0]), int(self.curr_FG_pixel[1]) - size], color, thickness)
            putText(self.markup_frame, 'Factor Graph Solution', (25, h - 125), FONT_HERSHEY_SIMPLEX,
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
            self.curr_frame_gray = cvtColor(self.markup_frame, COLOR_BGR2GRAY)

        if self.last_image is not None and self.last_image.shape == self.curr_frame_gray.shape:
            lft_rt, ret = phaseCorrelate(self.curr_frame_gray.astype(np.float64) / 255.0,
                                             self.last_image.astype(np.float64) / 255.0)
            lft, rt = lft_rt
            arrowedLine(self.markup_frame, (cx, cy), (int(cx + 10 * lft), int(cy + 10 * rt)), (0, 0, 255), 3)

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

            polylines(self.markup_frame, [crosshairsH], True, HUD_GREEN, thickness)
            polylines(self.markup_frame, [crosshairsV], True, HUD_GREEN, thickness)

        self.potentialResize()

        imshow(self.windowName, resize(self.markup_frame, (self.lastWidth, self.lastHeight)))

        if self.recording and time.time() - self.lastImageTime > self.camConfig.secondsBetweenImages:
            imwrite(os.path.join(self.filepath, str(self.img_idx) + '.png'), self.markup_frame)
            self.img_idx += 1
            self.lastImageTime = time.time()
            self.recordButton.configure(text=f'Saving Imagery: #{self.img_idx}')

    def plotOnImg(self, points, names, color):
        for idx, pxPt in enumerate(points):
            circle(self.markup_frame, (int(pxPt[0]), int(pxPt[1])), 5, color, 5)
            textLoc = (int(pxPt[0]) - 30, int(pxPt[1] - 30))
            putText(self.markup_frame, str(names[idx]), textLoc, FONT_HERSHEY_SIMPLEX, med_text(self.markup_frame.shape[0]), (0, 0, 0),
                        12,
                        LINE_AA)
            putText(self.markup_frame, str(names[idx]), textLoc, FONT_HERSHEY_SIMPLEX, med_text(self.markup_frame.shape[0]), color, 3,
                        LINE_AA)

    def potentialResize(self):
        if getWindowProperty(self.windowName, WND_PROP_VISIBLE) <= 0:
            return
        x, y, width, height = getWindowImageRect(self.windowName)
        aspectRatio = self.curr_frame.shape[1] / self.curr_frame.shape[0]
        if not getWindowProperty(self.windowName, WND_PROP_VISIBLE):
            return

        if not self.lastHeight == height and height != 0:
            resizeWindow(self.windowName, int(height * aspectRatio), height)
            self.lastHeight = height
            self.lastWidth = int(height * aspectRatio)
        elif not self.lastWidth == width and width != 0:
            resizeWindow(self.windowName, width, int(width / aspectRatio))
            self.lastWidth = width
            self.lastHeight = int(width / aspectRatio)

    @staticmethod
    def askFilepath(initDir, text):
        poss_filepath = filedialog.askdirectory(initialdir=initDir, mustexist=True, title=text)
        if poss_filepath == '':
            return None
        return poss_filepath


def dim_except_circle(frame, center, x_axes, y_axes=None, dim_factor=0.5):
    """
    Dims an image everywhere except inside a circle.

    Args:
        frame (np.array): the image
        center (tuple): (x, y) coordinates of the circle's center.
        dim_factor (float): Dimming factor (0 to 1, 0 for black, 1 for no dimming).
    """

    if y_axes is None:
        radius = x_axes
        if dim_factor == 0.0:
            return dim_entirely(frame, center, radius)

        # 1. Create a mask
        mask = np.zeros(frame.shape[:2], dtype="uint8")  # Black mask
        circle(mask, (int(center[0]), int(center[1])), int(radius), (255, 255, 255), -1)  # White circle on mask

    else:
        mask = np.zeros(frame.shape[:2], dtype='uint8')
        # rectangle(mask, (int(center[0]-x_axes),int(center[1]-y_axes)),(int(center[0]+x_axes),int(center[1]+y_axes)),
        #               color=255, thickness=-1)
        ellipse(mask, (int(center[0]), int(center[1])), (int(x_axes), int(y_axes)),
                    angle=0, startAngle=0, endAngle=360, color=(255, 255, 255), thickness=-1)

    # 2. Dim the entire image
    dimmed_img = (frame * dim_factor).astype("uint8")

    # 3. Copy the original circle area back to the dimmed image
    masked_circle = bitwise_and(frame, frame, mask=mask)

    # Invert the mask to select the area outside the circle
    inverted_mask = bitwise_not(mask)

    # Apply the mask to the dimmed image
    masked_dimmed = bitwise_and(dimmed_img, dimmed_img, mask=inverted_mask)

    # Add the original circle back
    frame = add(masked_circle, masked_dimmed)

    return frame


def dim_entirely(frame, center, radius):
    """
    Dims an image everywhere except inside a circle.

    Args:
        frame (np.array): the image
        center (tuple): (x, y) coordinates of the circle's center.
        radius (int): Radius of the circle.
    """

    # 1. Create a mask
    mask = np.zeros(frame.shape[:2], dtype="uint8")  # Black mask
    circle(mask, (int(center[0]), int(center[1])), int(radius), (255, 255, 255), -1)  # White circle on mask

    # 3. Copy the original circle area back to the dimmed image
    return bitwise_and(frame, frame, mask=mask)


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)
