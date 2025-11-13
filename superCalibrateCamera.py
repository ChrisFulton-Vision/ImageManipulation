import copy
import cv2
import os
import pickle
import re
import threading
import time
import sys
import queue

import cProfile
import pstats

import vmbpy.c_binding
from vmbpy import *

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from enum import Enum
from itertools import cycle
from tkinter import filedialog

from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from tkinter import StringVar
import customtkinter as ctk
import pandas as pd
from PIL import Image
from cv2_enumerate_cameras import enumerate_cameras

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
from SupportModules.CVFontScaling import small_text, med_text

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


cv2.setNumThreads(0)
cv2.setUseOptimized(True)

#  import superCalibrate as superCal
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
CAM_CONFIG_CACHE = str(Path.home() / ".superCalibrateCamera" / "cam_config.pkl")

class PausedCache:
    def __init__(self): self.idx = None; self.frame = None
    def set(self, i, f): self.idx, self.frame = i, f
    def get(self, i): return self.frame if self.idx == i else None
    def clear(self): self.idx = self.frame = None

@dataclass
class PlaybackState:
    """Single source of truth for playback state."""
    speed: float = 1.0          # signed: <0 reverse, 0 paused, >0 forward
    last_nonzero_sign: int = 1  # +1 or -1, used when resuming from pause
    stride: int = 1             # cached stride we last told the loader

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
    imageFilepath: Optional[str] = None
    cam_index: int = 0

    # feature flags
    detectTags: bool = False
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
    imageSource: 'ImageSource' = None  # set default below in __post_init__
    lidarFilepath: Optional[str] = None
    yoloFilepath: str = ''
    hud_data_filepath: str = ''

    # export range
    export_quality: ExportQuality = ExportQuality.med_quality
    start_export_idx: int = 0
    end_export_idx: int = 1

    # playback / processing
    playback_mode: 'PlaybackSpeed' = None
    processingKernel: 'ImageKernel' = None

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


class CameraGui(ctk.CTkFrame):
    def __init__(self, master, *args, **kwargs):

        self.profile_run_folder = False

        # Super class init, necessary for customTkinter
        super().__init__(master, *args, **kwargs)
        self._flag_vars: dict[str, ctk.BooleanVar] = {}
        self._checkboxes: dict[str, ctk.CTkCheckBox] = {}
        self._flags = [
            "detectTags", "undistort", "pnpLidarPoints", "qnpLidarPoints",
            "yoloInference", "yoloBiasTracking", "detect_corners", "detect_horizon",
            "factor_graph", "hyper_focus", "phase_correlation", "crosshairs",
            "cubemap", "hud"
        ]
        self.recording = False
        self.yoloSession = yolo.YOLO()
        self.camConfig = CameraConfig()
        self.detector = None
        self.arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
        self.arucoParams = cv2.aruco.DetectorParameters()
        self.arucoParams.cornerRefinementMinAccuracy = 0.15
        self.arucoParams.adaptiveThreshConstant = 1.0
        self._init_flag_vars()

        self.calibration = Calibration()
        self.detectIDS = None
        self.projectProbe = None
        self.centers = None
        self.calibFile = ''
        self.indexDict = {}
        self.scanForCameras()
        self.windowName = 'webcam'
        self.filepath = ''
        self.cam_frame = ctk.CTkFrame(master=master)
        self.config_frame = ctk.CTkFrame(master=master)
        self.export_frame = ctk.CTkFrame(master=master)
        self.playback_frame = ctk.CTkFrame(master=master)
        self.data_frame = ctk.CTkFrame(master=master)
        self.hotkey_frame = ctk.CTkFrame(master=master)
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

        self.streamOrImgCombo = ctk.CTkComboBox(self.cam_frame, values=self.available_sources,
                                                command=self.sourceUpdate)
        self.startStreamButton = ctk.CTkButton(master=self.cam_frame, text='Start Stream', fg_color=BUTTON_RED,
                                               hover_color='blue')

        self.singleImageFolderSelect = ctk.CTkButton(self.cam_frame, text='Select Img',
                                                     command=self.selectImagesFilepath)
        self.singleImageTextButton = ctk.CTkButton(self.cam_frame, text='No Image Selected')
        self.multiImageFolderSelect = ctk.CTkButton(self.cam_frame, text='Select Img Folder',
                                                    command=self.selectImagesFilepath)
        self.multiImageTextButton = ctk.CTkButton(self.cam_frame, text='No Folder Selected', command=self.startStreamOn)

        self.recordButton = ctk.CTkButton(master=self.export_frame, text='Saving Imagery', fg_color='green',
                                          hover_color='navy', command=self.recordOff)
        self.printButton = ctk.CTkButton(master=self.export_frame, text='Print LiDAR', fg_color='green',
                                         hover_color='navy', command=self.printLidarOnce)
        self.selectCameraCombo = ctk.CTkComboBox(self.cam_frame, values=list(self.indexDict.keys()),
                                                 command=self.selectCamera)
        self.selectFolderLabel = ctk.CTkLabel(self.cam_frame,
                                              text="../" + Path(self.filepath).name if self.filepath else "../")
        self.selectTruthPointsButton = ctk.CTkButton(master=self.cam_frame, text='Select LIDAR Points',
                                                     hover_color='blue', command=self.selectLidarFile)
        self.selectFlightLogButton = ctk.CTkButton(master=self.cam_frame, text='Select Flight Log File',
                                                   hover_color='blue', command=self.selectLogFile)

        self.playbackModeText = StringVar(value='Playback Mode: FPS')
        self.update_playbackMenu()

        if self.camConfig.lidarFilepath is not None:
            self.selectTruthPointsLabel = ctk.CTkLabel(self.cam_frame,
                                                       text="../" + Path(
                                                           self.camConfig.lidarFilepath).name if self.camConfig.lidarFilepath else "../")
        else:
            self.selectTruthPointsLabel = ctk.CTkLabel(self.cam_frame, text='No Truth Loaded')

        if self.camConfig.hud_data_filepath is not None:
            self.selectFlightLogLabel = ctk.CTkLabel(self.cam_frame, text="../" + Path(
                self.camConfig.hud_data_filepath).name if self.camConfig.hud_data_filepath else "../")
        else:
            self.selectFlightLogLabel = ctk.CTkLabel(self.cam_frame, text='No Flight Log Loaded')

        self.lidarTruthPoints = TruthPoints()
        self.selectYOLO_folderButton = ctk.CTkButton(self.cam_frame, text='Select YOLO Folder', fg_color=CTK_GREEN,
                                                     command=self.selectYoloFolder)
        self.selectYOLO_folderLabel = ctk.CTkLabel(self.cam_frame,
                                                   text="../" + Path(
                                                       self.camConfig.yoloFilepath).name if self.camConfig.yoloFilepath else "../"
                                                   )
        self.selectCalibLabel = None
        self.undistortCheckbox = ctk.CTkCheckBox(self.config_frame, text='Undistort',
                                                 variable=self._flag_vars['undistort'])
        self.detectAprilTagsCheckbox = ctk.CTkCheckBox(
            self.config_frame, text="Detect April Tags",
            variable=self._flag_vars["detectTags"]
        )
        self.detectHorizonCheckbox = ctk.CTkCheckBox(self.config_frame, text='Detect Horizon',
                                                     variable=self._flag_vars['detect_horizon'])

        self.yoloInferenceCheckbox = ctk.CTkCheckBox(self.config_frame, text='Run YOLO on image',
                                                     variable=self._flag_vars['yoloInference'])

        self.yoloBiasCheckbox = ctk.CTkCheckBox(self.config_frame, text='Run YOLO Bias Tracking',
                                                variable=self._flag_vars['yoloBiasTracking'])
        self.factorgraphCheckbox = ctk.CTkCheckBox(self.config_frame, text='Factor Graph',
                                                variable=self._flag_vars['factor_graph'])
        self.hyperfocusCheckbox = ctk.CTkCheckBox(self.config_frame, text='Hyper Focus',
                                                  variable=self._flag_vars['hyper_focus'])
        self.phaseCorrelationCheckbox = ctk.CTkCheckBox(self.config_frame, text='PhaseCorrelation',
                                                        variable=self._flag_vars['phase_correlation'])
        self.crosshairsCheckbox = ctk.CTkCheckBox(self.config_frame, text='Crosshairs',
                                                  variable=self._flag_vars['crosshairs'])
        self.cubemapCheckbox = ctk.CTkCheckBox(self.config_frame, text='Cubemap',
                                               variable=self._flag_vars['cubemap'])
        self.hudCheckbox = ctk.CTkCheckBox(self.config_frame, text='HUD',
                                           variable=self._flag_vars['hud'])
        self.confSliderLabel = ctk.CTkLabel(self.config_frame, text='Conf: 0.75')
        self.confSliderBar = ctk.CTkSlider(self.config_frame, command=self.confSlider,
                                           from_=0.15)  # type: ignore[arg-type]  # safe to ignore, ctk accepts float
        self.iouSliderLabel = ctk.CTkLabel(self.config_frame, text='IOU: 1.00')
        self.iouSliderBar = ctk.CTkSlider(self.config_frame, command=self.iouSlider)

        self.exportQualityCombo = ctk.CTkComboBox(self.export_frame, values=[member.value for member in ExportQuality],
                                                  command=self.updateQuality)
        self.exportToGifButton = ctk.CTkButton(self.export_frame, text="Export to Gif", command=self.exportToGif)
        self.exportToVidButton = ctk.CTkButton(self.export_frame, text="Export to Vid", command=self.exportToVid)
        self.making_gifOrVid = False

        self.loadFromCache()

        self._sync_flags_from_model()

        self.exportStartFrame = ctk.CTkLabel(self.export_frame, text=f'Start Frame: {self.camConfig.start_export_idx}')
        self.exportEndFrame = ctk.CTkLabel(self.export_frame, text=f'End Frame: {self.camConfig.end_export_idx}')

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
        self.saveToCache()

        self._ui_active = True
        self._last_ui_tick = 0.0
        self._ui_throttle_sec = 0.10  # refresh UI at most every 100 ms

        self.setupFrame()

    def _init_flag_vars(self):
        for name in self._flags:
            v = ctk.BooleanVar(value=bool(getattr(self.camConfig, name)))
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
            try: self.startStreamOff()
            except Exception: pass
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

    def on_section_hide(self, name: str):
        if name == "Playback":
            # Close imshow windows / pause playback, etc.
            try: cv2.destroyWindow(self.windowName)
            except Exception: pass

    def loadFromCache(self):
        """
        Load UI/config cache from CAM_CONFIG_CACHE, supporting:
          1) Legacy 3-pickle format:   [camConfig][filepath][calibFile]
          2) Dict format:               {"version": int, "config": dict|CameraConfig, "filepath": str, "calibFile": str}

        After load:
          - All path-like fields are normalized to *strings* ('' when unset)
          - UI labels are updated if widgets already exist
        """

        def _to_str(p):
            if p is None:
                return ''
            return str(p)

        def _normalize_cached_paths():
            # Top-level
            self.filepath = _to_str(getattr(self, 'filepath', ''))
            self.calibFile = _to_str(getattr(self, 'calibFile', ''))

            # Config paths
            cfg = self.camConfig
            # Some configs may not have all attrs (older caches) -> use getattr defaults
            cfg.imageFilepath = _to_str(getattr(cfg, 'imageFilepath', None))
            cfg.lidarFilepath = _to_str(getattr(cfg, 'lidarFilepath', None))
            cfg.hud_data_filepath = _to_str(getattr(cfg, 'hud_data_filepath', ''))
            cfg.yoloFilepath = _to_str(getattr(cfg, 'yoloFilepath', ''))

        cache_path = Path(CAM_CONFIG_CACHE)

        # Sensible defaults if cache missing
        if not cache_path.exists():
            # Initialize defaults if not already set
            if not hasattr(self, 'camConfig'):
                try:
                    self.camConfig = CameraConfig()  # dataclass path
                except Exception:
                    LOG.exception("Failed to initialize CameraConfig()")
                    raise
            self.filepath = _to_str(getattr(self, 'filepath', Path.cwd()))
            self.calibFile = _to_str(getattr(self, 'calibFile', ''))

            # Update labels if UI is ready
            if hasattr(self, 'selectFolderLabel'):
                try:
                    folder_text = "./" + os.path.basename(os.path.normpath(self.filepath)) if self.filepath else "./"
                    self.selectFolderLabel.configure(text=folder_text)
                except Exception:
                    LOG.exception("Failed to initialize Folder Label")
                    raise
            return

        try:
            with cache_path.open('rb') as f:
                first_obj = pickle.load(f)

                # Case 2: dict format (versioned)
                if isinstance(first_obj, dict) and ('config' in first_obj or 'version' in first_obj):
                    data = first_obj
                    cfg_obj = data.get('config', {})
                    # Accept dict or CameraConfig
                    if isinstance(cfg_obj, dict):
                        try:
                            self.camConfig = CameraConfig(**cfg_obj)
                        except Exception:
                            # Be tolerant of extra keys from older caches
                            self.camConfig = CameraConfig(
                                **{k: v for k, v in cfg_obj.items() if k in CameraConfig().__dict__})
                    else:
                        # Already a CameraConfig (pickled)
                        self.camConfig = cfg_obj

                    self.filepath = _to_str(data.get('filepath', Path.cwd()))
                    self.calibFile = _to_str(data.get('calibFile', ''))

                else:
                    # Case 1: legacy 3-pickle stream
                    # first_obj is camConfig (legacy class or dataclass instance)
                    cam_cfg_loaded = first_obj
                    # If your old class had .copy, keep using it for migration; otherwise assign directly
                    try:
                        # Try dataclass-style construction first
                        if isinstance(cam_cfg_loaded, dict):
                            self.camConfig = CameraConfig(**cam_cfg_loaded)
                        else:
                            # If CameraConfig (or old class), prefer direct assignment
                            self.camConfig = cam_cfg_loaded
                    except Exception:
                        # Fallback for very old caches with a custom copy()
                        try:
                            self.camConfig.copy(cam_cfg_loaded)  # old migration path, if available
                        except Exception:
                            # Last resort: new empty config
                            self.camConfig = CameraConfig()

                    # Next two pickles: filepath, calibFile
                    try:
                        self.filepath = pickle.load(f)
                    except Exception:
                        self.filepath = str(Path.cwd())
                    try:
                        self.calibFile = pickle.load(f)
                    except Exception:
                        self.calibFile = ''

            # Normalize path-like fields to strings for UI code that expects str/''.
            _normalize_cached_paths()

            # --- UI refresh (only if widgets exist already) ---
            # Folder label
            if hasattr(self, 'selectFolderLabel'):
                try:
                    folder_text = "./" + os.path.basename(os.path.normpath(self.filepath)) if self.filepath else "./"
                    self.selectFolderLabel.configure(text=folder_text)
                except Exception:
                    pass

            # Calibration ingest + label (if you have helper)
            try:
                if hasattr(self, 'ingestCalibration') and self.calibFile:
                    self.ingestCalibration()
            except Exception:
                pass

            # Flight log: if set, let the reader ingest
            try:
                if getattr(self.camConfig, 'hud_data_filepath', ''):
                    if hasattr(self, 'hud_marker'):
                        self.hud_marker.read_attitude_files(self.camConfig.hud_data_filepath)
            except Exception:
                pass

            # Per-source labels
            for fn in ('updateLidarLabel', 'updateYOLOLabel', 'updateFlightLogLabel'):
                if hasattr(self, fn):
                    try:
                        getattr(self, fn)()
                    except Exception:
                        pass

        except Exception as e:
            logging.warning("Failed to load cache %s: %s", cache_path, e)
            # Fall back to defaults
            try:
                self.camConfig = CameraConfig()
            except Exception:
                pass
            self.filepath = str(Path.cwd())
            self.calibFile = ''

        self.ingestCalibration()
        try:
            yolo_dir = getattr(self.camConfig, "yoloFilepath", "")
            if yolo_dir and Path(yolo_dir).exists():
                # Actually load the model/meta now (this was missing)
                self.yoloSession.setNewFolder(yolo_dir)
        except Exception as e:
            LOG.warning("Failed to restore YOLO folder from cache: %s", e)

        try:
            self.yoloSession.conf = float(getattr(self.camConfig, "yolo_conf", 0.75))
            self.yoloSession.iou = float(getattr(self.camConfig, "yolo_iou", 1.00))
        except Exception:
            pass
        try:
            if hasattr(self, "confSliderBar"):
                self.confSliderBar.set(self.yoloSession.conf)
            if hasattr(self, "confSliderLabel"):
                self.confSliderLabel.configure(text=f"Conf: {self.yoloSession.conf:.2f}")
            if hasattr(self, "iouSliderBar"):
                self.iouSliderBar.set(self.yoloSession.iou)
            if hasattr(self, "iouSliderLabel"):
                self.iouSliderLabel.configure(text=f"IOU: {self.yoloSession.iou:.2f}")
        except Exception:
            pass
        # --- Restore LiDAR truth points if path cached ---
        try:
            if getattr(self.camConfig, 'lidarFilepath', ''):
                self.loadTruthPoints()
        except Exception:
            pass

    def saveToCache(self):
        cache_path = Path(CAM_CONFIG_CACHE)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # coerce to strings in case fields were set to Path elsewhere
        cam_cfg = self.camConfig
        for attr in ('imageFilepath', 'lidarFilepath', 'hud_data_filepath', 'yoloFilepath'):
            if hasattr(cam_cfg, attr):
                val = getattr(cam_cfg, attr)
                if val is not None and not isinstance(val, str):
                    setattr(cam_cfg, attr, str(val))
        if not isinstance(self.filepath, str):  self.filepath = str(self.filepath)
        if not isinstance(self.calibFile, str): self.calibFile = str(self.calibFile)

        with cache_path.open('wb') as f:
            pickle.dump(self.camConfig, f)
            pickle.dump(self.filepath, f)
            pickle.dump(self.calibFile, f)

    def selectFolder(self):
        init_dir = Path(self.filepath).parent if self.filepath else Path.cwd()
        fp = self.askFilepath(str(init_dir), "Select Imagery Folder")
        if fp:
            self.filepath = fp
            self.camConfig.hud_data_filepath = fp
            self.saveToCache()
            self.loadFromCache()


    def loadCalibration(self):
        init_dir = Path(self.calibFile or self.filepath or Path.cwd()).parent
        poss_filepath = filedialog.askopenfilename(initialdir=str(init_dir), title='Select Calibration File')
        if poss_filepath:
            self.calibFile = poss_filepath
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
            self.hud_marker.read_attitude_files(poss_dir)
            self.updateFlightLogLabel()
            self.saveToCache()

    def selectYoloFolder(self):
        init_dir = Path(self.camConfig.yoloFilepath or Path.cwd())
        poss_dir = filedialog.askdirectory(initialdir=str(init_dir), title='Select YOLO Folder')
        if poss_dir:
            self.camConfig.yoloFilepath = poss_dir
            self.updateYOLOLabel()
            self.yoloSession.setNewFolder(self.camConfig.yoloFilepath)
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

    def loadTruthPoints(self):
        if self.camConfig.lidarFilepath is not None:
            lidar_path = Path(self.camConfig.lidarFilepath)
            if lidar_path.exists():
                with lidar_path.open('rb') as f:
                    test = pickle.load(f)
                    self.lidarTruthPoints.copy(test)
            else:
                LOG.error(
                    f'Cached LiDAR file not found. Using defaults. Attempted filepath:\n{self.camConfig.lidarFilepath}')

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

        if not self.calibration.fromBinFile(self.calibFile) and not self.calibration.fromFile(self.calibFile):
            if self.selectCalibLabel is not None:
                self.selectCalibLabel.configure(text='No Calibration Found')
                self.after(10, self.update_idletasks())
            return

        if not self.calibration.validCal:
            return

        # Note: this line exists because our aprilTag image was taken at 2848x2848, while calibration images
        # were 1424x1424. Thus, the camera calibration matrix is incorrect for this specific file.
        if self.calibFile == 'C:/repos/aburn/usr/24WintCalspanFltTest/Alvium_LJ_Calib_2DecSIFTED/calibration.pkl':
            self.calibration.scaleCalibration(2848)

        if self.selectCalibLabel is not None:
            self.selectCalibLabel.configure(text="../" + Path(self.calibFile).name if self.calibFile else "../",
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
        self.cubemapCheckbox.configure(state=cube_state)

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
        for camera_info in enumerate_cameras(cv2.CAP_DSHOW):
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
                numpy_buffer = cv2.cvtColor(numpy_buffer, cv2.COLOR_GRAY2BGR)
            else:
                numpy_buffer = cv2.cvtColor(numpy_buffer, cv2.COLOR_RGB2BGR)
            cv2.imshow(title, cv2.resize(numpy_buffer, (864, 864)))
            cv2.waitKey(1)
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

            if not self.singleImageFolderSelect.grid_info():
                self.singleImageFolderSelect.grid(row=1, column=0, padx=5, pady=5, sticky='nsew')
            if not self.singleImageTextButton.grid_info():
                self.singleImageTextButton.grid(row=1, column=1, padx=5, pady=5, sticky='nsew')

        elif self.camConfig.imageSource == ImageSource.Stream_from_Folder:

            if not self.multiImageFolderSelect.grid_info():
                self.multiImageFolderSelect.grid(row=1, column=0, padx=5, pady=5, sticky='nsew')
            if not self.multiImageTextButton.grid_info():
                self.multiImageTextButton.grid(row=1, column=1, padx=5, pady=5, sticky='nsew')

        else:
            raise ValueError(f'Unknown Image selection mode: {self.camConfig.imageSource}')

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

        self.streamOrImgCombo = ctk.CTkComboBox(self.cam_frame,
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

        selectFolderButton = ctk.CTkButton(self.cam_frame, text='Select Save Folder', command=self.selectFolder)
        selectFolderButton.grid(row=rowID, column=0, padx=5, pady=5, sticky='nsew')

        self.selectFolderLabel.grid(row=rowID, column=1, padx=5, pady=5, sticky='nsew')
        rowID += 1

        selectCalibButton = ctk.CTkButton(self.cam_frame, text='Select Calibration', command=self.loadCalibration)
        selectCalibButton.grid(row=rowID, column=0, padx=5, pady=5, sticky='nsew')

        self.selectCalibLabel = ctk.CTkLabel(self.cam_frame,
                                             text="../" + os.path.basename(os.path.normpath(self.calibFile)))
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

        aprilTagSizeEntryButton = ctk.CTkButton(self.cam_frame, text="Enter Size of April Tag (m)",
                                                command=self.setAprilTagSize)
        aprilTagSizeEntryButton.grid(row=rowID, column=0, padx=5, pady=5, sticky='nsew')

        self.aprilTagSizeEntry = ctk.CTkEntry(self.cam_frame, placeholder_text=str(self.camConfig.aprilTagSize))
        self.aprilTagSizeEntry.grid(row=rowID, column=1, padx=5, pady=5, sticky='nsew')

    def setup_configFrame(self):
        rowID = 0

        self.confSliderLabel.grid(row=rowID, column=0, padx=5, pady=5, sticky='nsew')
        self.confSliderBar.grid(row=rowID, column=1, padx=5, pady=5, sticky='nsew')
        rowID += 1

        self.iouSliderLabel.grid(row=rowID, column=0, padx=5, pady=5, sticky='nsew')
        self.iouSliderBar.grid(row=rowID, column=1, padx=5, pady=5, sticky='nsew')
        rowID += 1

        self.createDetector()
        self.detectAprilTagsCheckbox.grid(row=rowID, column=0, padx=5, pady=5, sticky='nsew')

        if not self.calibration.validCal:
            self.undistortCheckbox.configure(state='disabled')

        self.undistortCheckbox.grid(row=rowID, column=1, columnspan=2, padx=5, pady=5, sticky='nsew')
        rowID += 1

        pnpLidarPoints = ctk.CTkCheckBox(self.config_frame, text='SolvePnP LiDAR Into Image',
                                         variable=self._flag_vars['pnpLidarPoints'])
        pnpLidarPoints.grid(row=rowID, column=0, columnspan=1, padx=5, pady=5, sticky='ew')

        qnpLidarPoints = ctk.CTkCheckBox(self.config_frame, text='SolveQnP LiDAR Into Image',
                                         variable=self._flag_vars['qnpLidarPoints'])
        qnpLidarPoints.grid(row=rowID, column=1, columnspan=1, padx=5, pady=5, sticky='ew')
        rowID += 1

        self.yoloInferenceCheckbox.grid(row=rowID, column=0, columnspan=1, padx=5, pady=5, sticky='ew')

        self.yoloBiasCheckbox.grid(row=rowID, column=1, columnspan=1, padx=5, pady=5, sticky='ew')

        rowID += 1
        detectCornersCheckbox = ctk.CTkCheckBox(self.config_frame, text='Detect Corners',
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

        imageProcessingKernelLabel = ctk.CTkLabel(self.config_frame, text='Image Filter: ')
        imageProcessingKernelLabel.grid(row=rowID, column=0, padx=5, pady=5, sticky='ew')
        self.imageProcessingKernelCombobox = ctk.CTkComboBox(self.config_frame,
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

        activeEntryButton = ctk.CTkButton(self.export_frame, text="Time Between Saved Frames",
                                          command=self.getEntryValue)
        activeEntryButton.grid(row=rowID, column=0, padx=5, pady=5, sticky='nsew')

        self.timeBetweenImgsEntry = ctk.CTkEntry(self.export_frame,
                                                 placeholder_text=str(self.camConfig.secondsBetweenImages))
        self.timeBetweenImgsEntry.grid(row=rowID, column=1, padx=5, pady=5, sticky='nsew')
        rowID += 1

        qualityLabel = ctk.CTkLabel(self.export_frame, text="Export Quality: ")
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
            ("Space",      "Pause / resume"),
            ("f",          "Toggle Fixed-FPS ↔ Real-time"),
            ("c / z",      "Step forward / backward one frame"),
            ("d / a",      "Speed up / slow down playback"),
            ("r",          "Reverse direction"),
            ("w",          "Toggle overlays"),
            ("s / e",      "Mark export start / end"),
            ("[ / ] , { / }", "Adjust time offset (small / large)"),
            ("; / ' , : / \"", "Adjust time offset (fine)"),
            ("p",          "Persist time offset"),
            ("Esc",        "Exit player"),
        ]

        ctk.CTkLabel(self.hotkey_frame, text=title, font=("Segoe UI", 16, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=12, pady=(12, 8)
        )

        # headings
        ctk.CTkLabel(self.hotkey_frame, text="Key", font=("Segoe UI", 13, "bold")).grid(
            row=1, column=0, sticky="w", padx=12, pady=(6, 2)
        )
        ctk.CTkLabel(self.hotkey_frame, text="Action", font=("Segoe UI", 13, "bold")).grid(
            row=1, column=1, sticky="w", padx=12, pady=(6, 2)
        )

        # rows
        for i, (key, desc) in enumerate(items, start=2):
            ctk.CTkLabel(self.hotkey_frame, text=key).grid(row=i, column=0, sticky="w", padx=12, pady=2)
            ctk.CTkLabel(self.hotkey_frame, text=desc, justify="left", wraplength=520).grid(
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

    def setup_dataFrame(self):
        """Build the 'Data Processing' page: folder pick, CSV pick, params, run."""
        f = self.data_frame
        for w in f.winfo_children():
            w.destroy()

        f.grid_rowconfigure(99, weight=1)
        f.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(f, text="Batch YOLO over image folder", font=("Segoe UI", 16, "bold")).grid(
            row=0, column=0, columnspan=3, padx=12, pady=(16, 8), sticky="w"
        )

        # --- Select image folder ---
        self._dp_img_dir_var = ctk.StringVar(value=str(self.camConfig.imageFilepath or ""))

        def _choose_dir():
            d = filedialog.askdirectory(title="Select image folder")
            if d:
                self._dp_img_dir_var.set(d)

        ctk.CTkLabel(f, text="Folder:").grid(row=1, column=0, padx=12, pady=6, sticky="w")
        ctk.CTkEntry(f, textvariable=self._dp_img_dir_var).grid(row=1, column=1, padx=12, pady=6, sticky="ew")
        ctk.CTkButton(f, text="Browse…", command=_choose_dir).grid(row=1, column=2, padx=12, pady=6)

        # --- Select output CSV ---
        self._dp_csv_var = ctk.StringVar(value="")

        def _choose_csv():
            p = filedialog.asksaveasfilename(
                title="Select output CSV",
                defaultextension=".csv",
                filetypes=[("CSV", "*.csv")]
            )
            if p:
                self._dp_csv_var.set(p)

        ctk.CTkLabel(f, text="Output CSV:").grid(row=2, column=0, padx=12, pady=6, sticky="w")
        ctk.CTkEntry(f, textvariable=self._dp_csv_var).grid(row=2, column=1, padx=12, pady=6, sticky="ew")
        ctk.CTkButton(f, text="Browse…", command=_choose_csv).grid(row=2, column=2, padx=12, pady=6)

        # --- Checkpoint controls ---
        self._dp_ckptN = ctk.StringVar(value="200")  # default every 200 images
        ctk.CTkLabel(f, text="Checkpoint every N images:").grid(row=5, column=0, padx=12, pady=6, sticky="w")
        ctk.CTkEntry(f, textvariable=self._dp_ckptN, width=100).grid(row=5, column=1, padx=12, pady=6, sticky="w")

        # --- Prefetch controls ---
        self._dp_prefetch = ctk.StringVar(value="32")  # how many decoded/preprocessed items to buffer
        ctk.CTkLabel(f, text="Prefetch images (count):").grid(row=6, column=0, padx=12, pady=6, sticky="w")
        ctk.CTkEntry(f, textvariable=self._dp_prefetch, width=100).grid(row=6, column=1, padx=12, pady=6, sticky="w")

        # --- Progress UI ---
        self._dp_progress_label = ctk.CTkLabel(f, text="Idle")
        self._dp_progress_label.grid(row=20, column=0, columnspan=3, padx=12, pady=(8, 4), sticky="w")

        self._dp_progress = ctk.CTkProgressBar(f)  # determinate
        self._dp_progress.grid(row=21, column=0, columnspan=3, padx=12, pady=(0, 8), sticky="ew")
        self._dp_progress.set(0.0)

        self._dp_cancel_flag = False

        def _cancel():
            self._dp_cancel_flag = True
            if hasattr(self, "_dp_progress_label"):
                self._dp_progress_label.configure(text="Canceling…")
            if hasattr(self, "_dp_cancel_btn"):
                self._dp_cancel_btn.configure(state="disabled")

        self._dp_run_btn = ctk.CTkButton(
            f, text="Run YOLO Batch", fg_color="#2FA572",
            command=self._run_yolo_batch_start
        )
        self._dp_run_btn.grid(row=10, column=0, columnspan=2, padx=12, pady=(16, 12), sticky="ew")

        self._dp_cancel_btn = ctk.CTkButton(f, text="Cancel", fg_color="#A52F2F",
                                            command=_cancel)
        self._dp_cancel_btn.grid(row=10, column=2, padx=12, pady=(16, 12), sticky="ew")

    def _write_csv_atomic(self, out_csv: str, columns: list[str], completed_map: dict[str, dict]):
        """Write CSV atomically and sort by numeric portion of image_name."""

        rows = list(completed_map.values())
        df = pd.DataFrame(rows, columns=columns)

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
            txt = note or f"Processed {made}/{total_todo} new  •  {pct}%  •  ETA {mm:02d}:{ss:02d}  •  (total done: {len(completed_map)}/{total_all})"

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
        conf = float(self.yoloSession.conf)
        iou = float(self.yoloSession.iou)
        # Show what we're using
        _post_status(f"Using YOLO conf={conf:.2f}, iou={iou:.2f}…")

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

        # Resume from existing CSV
        completed_map = {}
        if os.path.exists(out_csv):
            try:
                prev = pd.read_csv(out_csv)
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
                self._write_csv_atomic(out_csv, columns, completed_map)
                _post_finish(f"Already complete. CSV written: {out_csv}")
            except Exception as e:
                _post_finish(f"Failed to write CSV: {e}")
            self._dp_cancel_flag = False
            return

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
        import cv2
        def _producer_job(path_str: str, name: str):
            if self._dp_cancel_flag:
                return
            p = Path(path_str)
            if not p.exists():
                rp = img_dir / p.name
                if rp.exists():
                    p = rp

            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
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
        from concurrent.futures import ThreadPoolExecutor, wait, as_completed
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
                            ud = _cv2.undistortPoints(pts_np, K, D, P=K).reshape(-1, 2)
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
                        self._write_csv_atomic(out_csv, columns, completed_map)
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
                self._write_csv_atomic(out_csv, columns, completed_map)
                msg = ("Partial CSV written (resume later): " + out_csv) if self._dp_cancel_flag else (
                            "Done. CSV written: " + out_csv)
            except Exception as e:
                msg = f"Failed to write CSV: {e}"

            _post_finish(msg)
            # reset for next run
            self._dp_cancel_flag = False

    def setup_playbackFrame(self):
        rowID = 0
        self.update_playbackMenu()
        playbackLabel = ctk.CTkLabel(self.playback_frame, textvariable=self.playbackModeText)
        playbackLabel.grid(row=rowID, column=0, sticky='w', padx=5, pady=5)

    def shutdown(self):
        self.shutting_down = True
        self.recordOff()
        self.startStreamOffBool()
        # Safely wait for window to be gone
        while True:
            try:
                vis = cv2.getWindowProperty(self.windowName, cv2.WND_PROP_VISIBLE)
                if vis <= 0:
                    break
            except Exception:
                break
            time.sleep(0.1)

    def setAprilTagSize(self):

        try:
            self.camConfig.aprilTagSize = float(self.aprilTagSizeEntry.get())
        except ValueError:
            self.aprilTagSizeEntry.delete(0, ctk.END)
            self.aprilTagSizeEntry.configure(placeholder_text=str(self.camConfig.aprilTagSize), )

    def getEntryValue(self):
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

    def _gather_annotated_frames(self) -> list[np.ndarray]:
        directory = Path(self.camConfig.imageFilepath).parent
        self.populate_idsTimes(str(directory))

        paths = []
        for rec in self.ImageTimeReader.idsTimes:
            p = Path(rec[0])
            paths.append(p if p.is_absolute() else (directory / p))

        try:
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
                display=False
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
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
            out = cv2.VideoWriter('output_video.mp4', fourcc, 10, (w, h))
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
        cv2.waitKey(1)

        self.threadStopper.set()
        cv2.destroyAllWindows()

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

        cv2.destroyAllWindows()


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

        cv2.destroyAllWindows()
        self.after(0, self._on_worker_exit)

    @staticmethod
    def convert_cv_to_pil(img):
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def run(self):

        if self.camConfig.imageSource == ImageSource.Camera_Stream:
            self.run_video_stream()
        elif self.camConfig.imageSource == ImageSource.Stream_from_Folder:
            self.profile_run_folder = True
            self.run_folder_reader_profiled()
        elif self.camConfig.imageSource == ImageSource.Static_Image:
            self.run_detectSingleImage()

    def run_video_stream(self):

        self.vc = cv2.VideoCapture(self.camConfig.cam_index, cv2.CAP_DSHOW)
        self.vc.set(cv2.CAP_PROP_FPS, 60)

        # cv2.destroyAllWindows()
        cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)
        rval, self.curr_frame = self.vc.read()
        if rval:
            cv2.resizeWindow(self.windowName, self.curr_frame.shape[1], self.curr_frame.shape[0])
            self.lastHeight = self.curr_frame.shape[0]
            self.lastWidth = self.curr_frame.shape[1]

        while (rval and not self.threadStopper.is_set() and
               cv2.getWindowProperty(self.windowName, cv2.WND_PROP_VISIBLE) > 0 and
               self.showWindow and not self.making_gifOrVid):
            rval, frame = self.vc.read()
            self.analyze_image(frame)
            key = cv2.waitKey(1)
            if key == 27:  # exit on ESC
                self.threadStopper.set()
                break

        # Minimal teardown in the worker; the UI thread will handle buttons/state.
        if self.vc is not None and self.vc.isOpened():
            self.vc.release()
            self.vc = None
        cv2.destroyAllWindows()
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
        k = cv2.waitKey(max_ms) & 0xFF
        if k not in (0, 0xFF, 255, -1):
            return [k]
        return []

    @staticmethod
    def load_time_offset(directory):
        try:
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
        cv2.destroyAllWindows()
        cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)

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
                            wall_start = time.monotonic() - ((t[idx_target] - t[0]) / rs if not self.last_nonzero_sign < 0
                                                             else ((t[-1] - t[idx_target]) / rs))
                        elif elapsed_ref > t[-1]:
                            idx_target = 0
                            wall_start = time.monotonic() - ((t[idx_target] - t[0]) / rs if not self.last_nonzero_sign < 0
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
                if cv2.getWindowProperty(self.windowName, cv2.WND_PROP_VISIBLE) <= 0:
                    self.threadStopper.set()
                    break

        finally:
            cv2.destroyAllWindows()
            self.after(0, self._on_worker_exit)
            loader.stop()

    def update_playbackMenu(self):
        if self.camConfig.playback_mode == PlaybackSpeed.Fixed_fps:
            self.playbackModeText.set(value=f'Playback Mode: FPS\nTarget FPS: {self.camConfig.target_fps:.2f}\n{'Pause' if self.pause else 'Rewind' if self.playback.speed < 0 else 'Play'}')
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
        pd.DataFrame({"offset": [self.hud_marker.offset]}).to_csv(out_csv, index=False)

        self.camConfig.cam_to_log_time_offset = 0.0

    def analyze_image(self, frame, img_time=None, name=None, display=True, box_around=False):

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
            self.hud_marker.draw_HUD(self.markup_frame, img_time, box_around)

        height = 0
        if self.camConfig.imageSource == ImageSource.Stream_from_Folder:
            (width, height), base = cv2.getTextSize(os.path.basename(name), cv2.FONT_HERSHEY_SIMPLEX, med_text(), 4)
            img_w, img_h, *_ = self.curr_frame.shape
            cv2.putText(self.markup_frame, os.path.basename(name), (img_w - width, img_h - height),
                        cv2.FONT_HERSHEY_SIMPLEX, med_text(), HUD_GREEN, 2)
        if img_time is not None:
            time_str = f"Flight Time: {img_time:.2f}"  # + 173.11338 - 11.658461:.2f}"
            (time_width, time_height), base = cv2.getTextSize(time_str, cv2.FONT_HERSHEY_SIMPLEX, med_text(), 4)
            img_w, img_h, *_ = self.curr_frame.shape
            cv2.putText(self.markup_frame, time_str, (img_w - time_width, img_h - time_height - height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, med_text(), HUD_GREEN, 2)

        if display:
            (h, w) = self.markup_frame.shape[:2]
            self.lowPassFPS = 0.925 * self.lowPassFPS + 0.075 * self.curr_fps
            cv2.putText(self.markup_frame, f"Offset: {self.camConfig.cam_to_log_time_offset:+.2f}s",
                        (int(0.015 * w), int(0.030 * h)), cv2.FONT_HERSHEY_SIMPLEX, med_text(), HUD_YELLOW, 2)
            cv2.putText(self.markup_frame,
                        f'Realtime: {self.camConfig.rt_speed:.2f}' if self.camConfig.playback_mode == PlaybackSpeed.Real_time else f'FPS: {self.lowPassFPS:.2f}/{self.camConfig.target_fps:.2f}',
                        (int(0.015 * w), int(0.060 * h)), cv2.FONT_HERSHEY_SIMPLEX, med_text(), HUD_YELLOW, 2)
            self.cleanup()

        if self.printLidar:
            self.print_pnp_results()

        if not display:
            return self.markup_frame

    def print_pnp_results(self):
        np.set_printoptions(precision=5, threshold=sys.maxsize, suppress=True)

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
            self.curr_frame_gray = cv2.cvtColor(self.curr_frame, cv2.COLOR_BGR2GRAY)
        harris_corners = cv2.cornerHarris(self.curr_frame_gray, 3, 3, 0.05)

        self.markup_frame[harris_corners > 0.025 * harris_corners.max()] = [0, 255, 255]

    def detectAprilTags(self):

        if self.curr_frame_gray is None:
            self.curr_frame_gray = cv2.cvtColor(self.curr_frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(self.curr_frame_gray)
        self.centers = None
        self.detectIDS = []

        if corners is None or ids is None:
            return

        for corners, idx in zip(corners, ids):
            corners = np.squeeze(np.array(corners))
            polyline = [np.array(corners, np.int32).reshape((-1, 1, 2))]
            pixCenter = np.mean(corners, axis=0).astype(np.int32)
            cv2.polylines(self.markup_frame, polyline, True, HUD_GREEN, 4, lineType=cv2.FILLED)
            cv2.putText(self.markup_frame, str(idx[0]), pixCenter,
                        cv2.FONT_HERSHEY_SIMPLEX, small_text(), HUD_GREEN, 4)
            cv2.putText(self.markup_frame, str(idx[0]), pixCenter,
                        cv2.FONT_HERSHEY_SIMPLEX, small_text(), (0, 0, 0), 1)

            self.detectIDS.append(idx)

            if self.centers is None:
                self.centers = np.array(pixCenter).astype('float32')
            else:
                self.centers = np.vstack((self.centers, np.array(pixCenter).astype('float32')))

    def pnpLidarPoints(self):

        if self.centers is not None and len(self.centers) >= 6:
            truthPoints = copy.copy(self.lidarTruthPoints.truthPoints)
            points = []
            distParams = np.zeros((5,))  # use image undistort instead

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
                            cv2.FONT_HERSHEY_DUPLEX, small_text(),
                            (255, 255, 0), 3,
                            cv2.LINE_AA)
                cv2.putText(self.markup_frame, 'Location From LiDAR: ' + np.array2string(vectPnP),
                            (50, 150), cv2.FONT_HERSHEY_DUPLEX, small_text(),
                            (255, 255, 0), 3,
                            cv2.LINE_AA)

    def qnpLidarPoints(self):

        if self.centers is not None and len(self.centers) >= 6:
            truthPoints = copy.copy(self.lidarTruthPoints.truthPoints)

            points = []
            # distParams = np.zeros((5,))  # use image undistort instead

            removeIDs = []
            for idx, detectID in enumerate(self.detectIDS):
                try:
                    points.append(truthPoints[str(detectID[0])])
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
            cv2.putText(self.markup_frame, 'Orientation (quat) From LiDAR: ' + format(quat, 'ijk.6f'), (50, 225),
                        cv2.FONT_HERSHEY_DUPLEX,
                        small_text(),
                        (255, 255, 0), 3,
                        cv2.LINE_AA)
            cv2.putText(self.markup_frame, 'Location From LiDAR: ' + np.array2string(vect), (50, 300),
                        cv2.FONT_HERSHEY_DUPLEX,
                        small_text(),
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
                cv2.putText(self.markup_frame, 'BB-Width Solution', (25, w - 75), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (50, 255, 255), 1)
                cv2.putText(self.markup_frame,
                            f'x:{self.last_yolo_3d_estimate[0]:.3f}, y:{self.last_yolo_3d_estimate[1]:.3f}, z:{self.last_yolo_3d_estimate[2]:.3f}',
                            (25, w - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 255, 255), 1)
                # cv2.circle(self.markup_frame, (int(self.last_yolo_center[0]), int(self.last_yolo_center[1])),
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
            cv2.circle(self.markup_frame, (int(self.curr_FG_pixel[0]), int(self.curr_FG_pixel[1])), size, (0, 0, 0),
                       thickness)
            cv2.line(self.markup_frame, [int(self.curr_FG_pixel[0]) + size, int(self.curr_FG_pixel[1])],
                     [int(self.curr_FG_pixel[0]) - size, int(self.curr_FG_pixel[1])], (0, 0, 0), thickness)
            cv2.line(self.markup_frame, [int(self.curr_FG_pixel[0]), int(self.curr_FG_pixel[1]) + size],
                     [int(self.curr_FG_pixel[0]), int(self.curr_FG_pixel[1]) - size], (0, 0, 0), thickness)
            cv2.putText(self.markup_frame, 'Factor Graph Solution', (25, h - 125), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 0), thickness)

            thickness = 1
            cv2.circle(self.markup_frame, (int(self.curr_FG_pixel[0]), int(self.curr_FG_pixel[1])), size, color,
                       thickness)
            cv2.line(self.markup_frame, [int(self.curr_FG_pixel[0]) + size, int(self.curr_FG_pixel[1])],
                     [int(self.curr_FG_pixel[0]) - size, int(self.curr_FG_pixel[1])], color, thickness)
            cv2.line(self.markup_frame, [int(self.curr_FG_pixel[0]), int(self.curr_FG_pixel[1]) + size],
                     [int(self.curr_FG_pixel[0]), int(self.curr_FG_pixel[1]) - size], color, thickness)
            cv2.putText(self.markup_frame, 'Factor Graph Solution', (25, h - 125), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, color, thickness)

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

            cv2.polylines(self.markup_frame, [crosshairsH], True, HUD_GREEN, thickness)
            cv2.polylines(self.markup_frame, [crosshairsV], True, HUD_GREEN, thickness)

        self.potentialResize()

        cv2.imshow(self.windowName, cv2.resize(self.markup_frame, (self.lastWidth, self.lastHeight)))

        if self.recording and time.time() - self.lastImageTime > self.camConfig.secondsBetweenImages:
            cv2.imwrite(os.path.join(self.filepath, str(self.img_idx) + '.png'), self.markup_frame)
            self.img_idx += 1
            self.lastImageTime = time.time()
            self.recordButton.configure(text=f'Saving Imagery: #{self.img_idx}')

    def plotOnImg(self, points, names, color):
        for idx, pxPt in enumerate(points):
            cv2.circle(self.markup_frame, (int(pxPt[0]), int(pxPt[1])), 5, color, 5)
            textLoc = (int(pxPt[0]) - 30, int(pxPt[1] - 30))
            cv2.putText(self.markup_frame, str(names[idx]), textLoc, cv2.FONT_HERSHEY_SIMPLEX, med_text(), (0, 0, 0),
                        12,
                        cv2.LINE_AA)
            cv2.putText(self.markup_frame, str(names[idx]), textLoc, cv2.FONT_HERSHEY_SIMPLEX, med_text(), color, 3,
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
        cv2.circle(mask, (int(center[0]), int(center[1])), int(radius), (255, 255, 255), -1)  # White circle on mask

    else:
        mask = np.zeros(frame.shape[:2], dtype='uint8')
        # cv2.rectangle(mask, (int(center[0]-x_axes),int(center[1]-y_axes)),(int(center[0]+x_axes),int(center[1]+y_axes)),
        #               color=255, thickness=-1)
        cv2.ellipse(mask, (int(center[0]), int(center[1])), (int(x_axes), int(y_axes)),
                    angle=0, startAngle=0, endAngle=360, color=(255, 255, 255), thickness=-1)

    # 2. Dim the entire image
    dimmed_img = (frame * dim_factor).astype("uint8")

    # 3. Copy the original circle area back to the dimmed image
    masked_circle = cv2.bitwise_and(frame, frame, mask=mask)

    # Invert the mask to select the area outside the circle
    inverted_mask = cv2.bitwise_not(mask)

    # Apply the mask to the dimmed image
    masked_dimmed = cv2.bitwise_and(dimmed_img, dimmed_img, mask=inverted_mask)

    # Add the original circle back
    frame = cv2.add(masked_circle, masked_dimmed)

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
    cv2.circle(mask, (int(center[0]), int(center[1])), int(radius), (255, 255, 255), -1)  # White circle on mask

    # 3. Copy the original circle area back to the dimmed image
    return cv2.bitwise_and(frame, frame, mask=mask)


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)
