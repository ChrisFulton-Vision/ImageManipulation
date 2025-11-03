import copy
import cv2
import os
import pickle
import re
import threading
import time
import vmbpy.c_binding
from SupportModules import yolo
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from enum import Enum
from itertools import cycle
from tkinter import filedialog

import customtkinter as ctk
import pandas as pd
from PIL import Image
from cv2_enumerate_cameras import enumerate_cameras
from vmbpy import *

from SupportModules.Calibration import Calibration
from SupportModules.FG_DrogueOnly import FactorGraph
from SupportModules.ImageTimeReader import ImageTimeReader
from SupportModules.LidarTruth import TruthPoints
from SupportModules.FilterImage import ImageKernel, Gabor, GaborGUI, applyConvolutionFilter
from SupportModules.HUD_draw import HUD_Marker
from SupportModules.TwoD_to_ThreeD import solveQnP
from SupportModules.bufferImageLoader import BufferedImageLoader as imgBuf
from SupportModules.convertToGif import make_gif, ExportQuality
from SupportModules.quaternions import *
from SupportModules.quaternions import Quaternion as q
from SupportModules.CVFontScaling import small_text, med_text

import logging

LOG = logging.getLogger("superCalibrate")

if not LOG.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(fmt)
    LOG.addHandler(handler)
    LOG.setLevel(logging.INFO)
    # LOG.setLevel(logging.DEBUG)
    # LOG.setLevel(logging.WARNING)

# import superCalibrate as superCal
#pip install cv2_enumerate_cameras
#or
#pip install git+https://github.com/chinaheyu/cv2_enumerate_cameras.git

CTK_GREEN = '#2FA572'
HUD_GREEN = (0, 255, 0)
HUD_YELLOW = (0, 255, 255)
BUTTON_RED = 'red3'
CAM_CONFIG_CACHE = str(Path.home() / ".superCalibrateCamera" / "cam_config.pkl")


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


class ImageSliderBar:
    def __init__(self, num_images: int, refresh_hz: float = 20.0):
        self.alive = True
        self.num_images = int(num_images)
        self.play_speed = 1
        self.curr_img_idx = 0

        # --- Throttle config/state ---
        self.refresh_interval = 1.0 / max(1.0, float(refresh_hz))  # seconds
        self._last_ui_ts = 0.0
        self._ui_pending = False

    def update_img_id(self, new_img_idx):
        self.curr_img_idx = int(new_img_idx)

    def next_id(self):
        """Called from worker thread: advances index AND schedules a throttled UI update."""
        if not self.alive or self.num_images <= 0:
            return self.curr_img_idx

        self.curr_img_idx = (self.curr_img_idx + self.play_speed) % self.num_images
        return self.curr_img_idx

    # ---------------- internal helpers ----------------
    def close(self):
        """Safe shutdown: mark dead, cancel pending UI, then destroy window."""
        self.alive = False

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
class CameraConfig():
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


class CameraGui():
    def __init__(self, gui):
        # self.GifMaker = GifMaker()
        self.gui = gui
        self.yoloSession = yolo.YOLO()
        self.camConfig = CameraConfig()
        self.calibration = Calibration()
        self.detectIDS = None
        self.projectProbe = None
        self.centers = None
        self.calibFile = ''
        self.indexDict = {}
        self.scanForCameras()
        self.windowName = 'webcam'
        self.filepath = ''
        self.cam_frame = ctk.CTkFrame(master=self.gui)
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

        # Optimization for undistort
        self.map1, self.map2 = None, None

        self.threadStopper = ThreadStopper()
        self._thread = None

        self.fps_time_log = time.time()
        self.curr_fps = 20.0

        self.imageProcessingKernelCombobox = None

        self.bank_indicator_points = None

        self.last_image = None

        self.available_sources = [source.value for source in ImageSource]

        self.streamOrImgCombo = ctk.CTkComboBox(self.cam_frame, values=self.available_sources,
                                                command=self.sourceUpdate)
        self.startStreamButton = ctk.CTkButton(master=self.cam_frame, text='Start Stream', fg_color=BUTTON_RED,
                                               hover_color='blue')
        self.recordButton = ctk.CTkButton(master=self.cam_frame, text='Saving Imagery', fg_color='green',
                                          hover_color='navy', command=self.recordOff)
        self.printButton = ctk.CTkButton(master=self.cam_frame, text='Print LiDAR', fg_color='green',
                                         hover_color='navy', command=self.printLidarOnce)
        self.selectCameraCombo = ctk.CTkComboBox(self.cam_frame, values=list(self.indexDict.keys()),
                                                 command=self.selectCamera)
        self.selectFolderLabel = ctk.CTkLabel(self.cam_frame,
                                              text="../" + Path(self.filepath).name if self.filepath else "../")
        self.selectTruthPointsButton = ctk.CTkButton(master=self.cam_frame, text='Select LIDAR Points',
                                                     hover_color='blue', command=self.selectLidarFile)
        self.selectFlightLogButton = ctk.CTkButton(master=self.cam_frame, text='Select Flight Log File',
                                                   hover_color='blue', command=self.selectLogFile)

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
        self.undistortCheckbox = ctk.CTkCheckBox(self.cam_frame, text='Undistort')
        self.detectAprilTagsCheckbox = ctk.CTkCheckBox(self.cam_frame, text='Detect April Tags')
        self.detectHorizonCheckbox = ctk.CTkCheckBox(self.cam_frame, text='Detect Horizon')
        self.yoloInferenceCheckbox = ctk.CTkCheckBox(self.cam_frame, text='Run YOLO on image',
                                                     command=self.toggleYoloInference)
        self.yoloBiasCheckbox = ctk.CTkCheckBox(self.cam_frame, text='Run YOLO Bias Tracking',
                                                command=self.toggleYoloBiasTracking)
        self.factorgraphCheckbox = ctk.CTkCheckBox(self.cam_frame, text='Factor Graph')
        self.hyperfocusCheckbox = ctk.CTkCheckBox(self.cam_frame, text='Hyper Focus')
        self.phaseCorrelationCheckbox = ctk.CTkCheckBox(self.cam_frame, text='PhaseCorrelation')
        self.crosshairsCheckbox = ctk.CTkCheckBox(self.cam_frame, text='Crosshairs')
        self.cubemapCheckbox = ctk.CTkCheckBox(self.cam_frame, text='Cubemap')
        self.hudCheckbox = ctk.CTkCheckBox(self.cam_frame, text='HUD')

        self.singleImageFolderSelect = ctk.CTkButton(self.cam_frame, text='Select Img',
                                                     command=self.selectImagesFilepath)
        self.singleImageTextButton = ctk.CTkButton(self.cam_frame, text='No Image Selected')
        self.multiImageFolderSelect = ctk.CTkButton(self.cam_frame, text='Select Img Folder',
                                                    command=self.selectImagesFilepath)
        self.multiImageTextButton = ctk.CTkButton(self.cam_frame, text='No Folder Selected', command=self.startStreamOn)
        self.confSliderLabel = ctk.CTkLabel(self.cam_frame, text='Conf: 0.75')
        self.confSliderBar = ctk.CTkSlider(self.cam_frame, command=self.confSlider, from_=0.15)
        self.iouSliderLabel = ctk.CTkLabel(self.cam_frame, text='IOU: 1.00')
        self.iouSliderBar = ctk.CTkSlider(self.cam_frame, command=self.iouSlider)

        self.exportQualityCombo = ctk.CTkComboBox(self.cam_frame, values=[member.value for member in ExportQuality],
                                                  command=self.updateQuality)
        self.exportToGifButton = ctk.CTkButton(self.cam_frame, text="Export to Gif", command=self.exportToGif)
        self.exportToVidButton = ctk.CTkButton(self.cam_frame, text="Export to Vid", command=self.exportToVid)
        self.making_gifOrVid = False

        self.loadFromCache()

        self.exportStartFrame = ctk.CTkLabel(self.cam_frame, text=f'Start Frame: {self.camConfig.start_export_idx}')
        self.exportEndFrame = ctk.CTkLabel(self.cam_frame, text=f'End Frame: {self.camConfig.end_export_idx}')

        self.confSliderBar.set(self.camConfig.yolo_conf)
        self.iouSliderBar.set(self.camConfig.yolo_iou)

        self.selectCameraCombo.set(list(self.indexDict.keys())[self.camConfig.cam_index])
        self.vc = None
        # self.vc.setExceptionMode(True)
        # self.detector = Detector(refine_edges=1, decode_sharpening=0.0)

        self.detector = None
        self.arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
        self.arucoParams = cv2.aruco.DetectorParameters()

        self.img_idx = 0
        self.timeBetweenImgsEntry = None
        self.lastImageTime = 0
        self.camFrameGeometry = '445x915'
        self.cam_frame.grid_rowconfigure(list(range(3)), weight=1)  # configure grid system
        self.cam_frame.grid_columnconfigure(list(range(3)), weight=1)
        self.lastWidth = 1
        self.lastHeight = 1
        self.saveToCache()

    def loadFromCache(self):
        """
        Load UI/config cache from CAM_CONFIG_CACHE, supporting:
          1) Legacy 3-pickle format:   [camConfig][filepath][calibFile]
          2) Dict format:               {"version": int, "config": dict|CameraConfig, "filepath": str, "calibFile": str}

        After load:
          - All path-like fields are normalized to *strings* ('' when unset)
          - UI labels are updated if widgets already exist
        """
        from pathlib import Path
        import pickle
        import os

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
                    pass
            self.filepath = _to_str(getattr(self, 'filepath', Path.cwd()))
            self.calibFile = _to_str(getattr(self, 'calibFile', ''))

            # Update labels if UI is ready
            if hasattr(self, 'selectFolderLabel'):
                try:
                    folder_text = "./" + os.path.basename(os.path.normpath(self.filepath)) if self.filepath else "./"
                    self.selectFolderLabel.configure(text=folder_text)
                except Exception:
                    pass
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
            import logging
            logging.warning("Failed to load cache %s: %s", cache_path, e)
            # Fall back to defaults
            try:
                self.camConfig = CameraConfig()
            except Exception:
                pass
            self.filepath = str(Path.cwd())
            self.calibFile = ''

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

        if not self.calibration.fromBinFile(self.calibFile):
            if not self.calibration.fromFile(self.calibFile):
                if self.selectCalibLabel is not None:
                    self.selectCalibLabel.configure(text='No Calibration Found')
                    self.gui.after(100, self.gui.update())
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
            self.cam_frame.update()
            self.gui.update()
            self.selectCalibLabel.update()
            self.cam_frame.update()
            self.gui.update()

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

        newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w,h), alpha=0)

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

        self.confSliderLabel.grid(row=rowID, column=0, padx=5, pady=5, sticky='nsew')
        self.confSliderBar.grid(row=rowID, column=1, padx=5, pady=5, sticky='nsew')
        rowID += 1

        self.iouSliderLabel.grid(row=rowID, column=0, padx=5, pady=5, sticky='nsew')
        self.iouSliderBar.grid(row=rowID, column=1, padx=5, pady=5, sticky='nsew')
        rowID += 1

        # detectAprilTagsCheckbox = ctk.CTkCheckBox(self.cam_frame, text='Detect April Tags')
        if self.camConfig.detectTags is False:
            self.detectAprilTagsCheckbox.deselect()
        else:
            self.detectAprilTagsCheckbox.select()
            self.createDetector()
        self.detectAprilTagsCheckbox.configure(command=self.toggleDetectTags)
        self.detectAprilTagsCheckbox.grid(row=rowID, column=0, padx=5, pady=5, sticky='nsew')

        # self.undistortCheckbox = ctk.CTkCheckBox(self.cam_frame, text='Undistort')
        if not self.calibration.validCal:
            self.undistortCheckbox.configure(state='disabled')

        if self.camConfig.undistort is False:
            self.undistortCheckbox.deselect()
        else:
            self.undistortCheckbox.select()

        self.undistortCheckbox.configure(command=self.toggleUndistort)
        self.undistortCheckbox.grid(row=rowID, column=1, columnspan=2, padx=5, pady=5, sticky='nsew')
        rowID += 1

        pnpLidarPoints = ctk.CTkCheckBox(self.cam_frame, text='SolvePnP LiDAR Into Image')
        if self.camConfig.pnpLidarPoints is False:
            pnpLidarPoints.deselect()
        else:
            pnpLidarPoints.select()
        pnpLidarPoints.configure(command=self.togglePnpLidarPoints)
        pnpLidarPoints.grid(row=rowID, column=0, columnspan=1, padx=5, pady=5, sticky='ew')

        qnpLidarPoints = ctk.CTkCheckBox(self.cam_frame, text='SolveQnP LiDAR Into Image')
        if self.camConfig.qnpLidarPoints is False:
            qnpLidarPoints.deselect()
        else:
            qnpLidarPoints.select()
        qnpLidarPoints.configure(command=self.toggleQnpLidarPoints)
        qnpLidarPoints.grid(row=rowID, column=1, columnspan=1, padx=5, pady=5, sticky='ew')

        rowID += 1

        # yoloInferenceCheckbox = ctk.CTkCheckBox(self.cam_frame, text='Run YOLO on image')
        if self.camConfig.yoloInference is False:
            self.yoloInferenceCheckbox.deselect()
        else:
            self.yoloInferenceCheckbox.select()
        self.yoloInferenceCheckbox.grid(row=rowID, column=0, columnspan=1, padx=5, pady=5, sticky='ew')

        if self.camConfig.yoloBiasTracking is False:
            self.yoloBiasCheckbox.deselect()
        else:
            self.yoloBiasCheckbox.select()
        self.yoloBiasCheckbox.grid(row=rowID, column=1, columnspan=1, padx=5, pady=5, sticky='ew')

        rowID += 1
        detectCornersCheckbox = ctk.CTkCheckBox(self.cam_frame, text='Detect Corners')
        if self.camConfig.detect_corners is False:
            detectCornersCheckbox.deselect()
        else:
            detectCornersCheckbox.select()
        detectCornersCheckbox.configure(command=self.toggleDetectCorners)
        detectCornersCheckbox.grid(row=rowID, column=0, columnspan=1, padx=5, pady=5, sticky='ew')

        # detectHorizonCheckbox = ctk.CTkCheckBox(self.cam_frame, text='Detect Horizon')
        if self.camConfig.detect_horizon is False:
            self.detectHorizonCheckbox.deselect()
        else:
            self.detectHorizonCheckbox.select()
        self.detectHorizonCheckbox.configure(command=self.toggleDetectHorizon)
        self.detectHorizonCheckbox.grid(row=rowID, column=1, columnspan=1, padx=5, pady=5, sticky='ew')
        rowID += 1

        # factorgraphCheckbox = ctk.CTkCheckBox(self.cam_frame, text='Factor Graph')
        if self.camConfig.factor_graph is False:
            self.factorgraphCheckbox.deselect()
        else:
            self.factorgraphCheckbox.select()
        self.factorgraphCheckbox.configure(command=self.toggleFactorgraph)
        self.factorgraphCheckbox.grid(row=rowID, column=0, columnspan=1, padx=5, pady=5, sticky='ew')

        # hyperfocusCheckbox = ctk.CTkCheckBox(self.cam_frame, text='Hyper Focus')
        if self.camConfig.hyper_focus is False:
            self.hyperfocusCheckbox.deselect()
        else:
            self.hyperfocusCheckbox.select()
        self.hyperfocusCheckbox.configure(command=self.toggleHyperFocus)
        self.hyperfocusCheckbox.grid(row=rowID, column=1, columnspan=1, padx=5, pady=5, sticky='ew')

        rowID += 1

        if self.camConfig.phase_correlation is False:
            self.phaseCorrelationCheckbox.deselect()
        else:
            self.phaseCorrelationCheckbox.select()
        self.phaseCorrelationCheckbox.configure(command=self.togglePhaseCorrelation)
        self.phaseCorrelationCheckbox.grid(row=rowID, column=0, columnspan=1, padx=5, pady=5, sticky='ew')

        if self.camConfig.crosshairs is False:
            self.crosshairsCheckbox.deselect()
        else:
            self.crosshairsCheckbox.select()
        self.crosshairsCheckbox.configure(command=self.toggleCrosshairs)
        self.crosshairsCheckbox.grid(row=rowID, column=1, columnspan=1, padx=5, pady=5, sticky='ew')

        rowID += 1

        if not self.camConfig.cubemap:
            self.cubemapCheckbox.deselect()
        else:
            self.cubemapCheckbox.select()
        self.cubemapCheckbox.configure(command=self.toggleCubemap, state='disabled')
        self.cubemapCheckbox.grid(row=rowID, column=0, columnspan=1, padx=5, pady=5, sticky='ew')

        if not self.camConfig.hud:
            self.hudCheckbox.deselect()
        else:
            self.hudCheckbox.select()
        self.hudCheckbox.configure(command=self.toggleHud)
        self.hudCheckbox.grid(row=rowID, column=1, columnspan=1, padx=5, pady=5, sticky='ew')

        rowID += 1

        imageProcessingKernelLabel = ctk.CTkLabel(self.cam_frame, text='Image Filter: ')
        imageProcessingKernelLabel.grid(row=rowID, column=0, padx=5, pady=5, sticky='ew')
        self.imageProcessingKernelCombobox = ctk.CTkComboBox(self.cam_frame,
                                                             values=list(ImageKernel.__members__.keys()))
        self.imageProcessingKernelCombobox.set(self.camConfig.processingKernel.name)
        self.imageProcessingKernelCombobox.configure(command=self.updateImageProcessingKernel)
        self.updateImageProcessingKernel(self.camConfig.processingKernel.name)
        self.imageProcessingKernelCombobox.grid(row=rowID, column=1, columnspan=1, padx=5, pady=5, sticky='ew')
        rowID += 1

        aprilTagSizeEntryButton = ctk.CTkButton(self.cam_frame, text="Enter Size of April Tag (m)",
                                                command=self.setAprilTagSize)
        aprilTagSizeEntryButton.grid(row=rowID, column=0, padx=5, pady=5, sticky='nsew')

        self.aprilTagSizeEntry = ctk.CTkEntry(self.cam_frame, placeholder_text=str(self.camConfig.aprilTagSize))
        self.aprilTagSizeEntry.grid(row=rowID, column=1, padx=5, pady=5, sticky='nsew')
        rowID += 1

        self.recordOff()
        self.recordButton.grid(row=rowID, column=0, columnspan=1, padx=5, pady=5, sticky='ew')
        self.printButton.grid(row=rowID, column=1, columnspan=1, padx=5, pady=5, sticky='ew')
        rowID += 1

        activeEntryButton = ctk.CTkButton(self.cam_frame, text="Time Between Saved Frames",
                                          command=self.getEntryValue)
        activeEntryButton.grid(row=rowID, column=0, padx=5, pady=5, sticky='nsew')

        self.timeBetweenImgsEntry = ctk.CTkEntry(self.cam_frame,
                                                 placeholder_text=str(self.camConfig.secondsBetweenImages))
        self.timeBetweenImgsEntry.grid(row=rowID, column=1, padx=5, pady=5, sticky='nsew')
        rowID += 1

        qualityLabel = ctk.CTkLabel(self.cam_frame, text="Export Quality: ")
        qualityLabel.grid(row=rowID, column=0, padx=5, pady=5, sticky='ew')
        self.exportQualityCombo.grid(row=rowID, column=1, padx=5, pady=5, sticky='ew')
        rowID += 1

        self.exportToGifButton.grid(row=rowID, column=0, padx=5, pady=5, sticky='ew')
        self.exportToVidButton.grid(row=rowID, column=1, padx=5, pady=5, sticky='ew')

        rowID += 1

        self.exportStartFrame.grid(row=rowID, column=0, padx=5, pady=5, sticky='ew')
        self.exportEndFrame.grid(row=rowID, column=1, padx=5, pady=5, sticky='ew')

        rowID += 1

        goBackButton = ctk.CTkButton(self.cam_frame, text="Return to Main", command=self.releaseCamReturnToMain)
        goBackButton.grid(row=rowID, column=0, columnspan=2, padx=5, pady=5, sticky='ew')

        self.cam_frame.pack()

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

    def releaseCamReturnToMain(self):
        self.startStreamOffBool()
        self.gui.returnToMain()

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
            self.gui.after(0, self._exportToGifOrVid_done)

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
            self.gui.after(0, self._exportToGifOrVid_done)

    def _exportToGifOrVid_done(self):
        self.exportToGifButton.configure(text="Export to GIF", state='normal', fg_color=CTK_GREEN)
        self.exportToVidButton.configure(text="Export to Vid", state='normal', fg_color=CTK_GREEN)
        self.making_gifOrVid = False

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

    def _toggle(self, attr: str, checkbox: ctk.CTkCheckBox | None = None):
        val = not getattr(self.camConfig, attr)
        setattr(self.camConfig, attr, val)
        if checkbox is not None:
            (checkbox.select() if val else checkbox.deselect())
        self.saveToCache()

    def togglePnpLidarPoints(self):
        self.camConfig.pnpLidarPoints = not self.camConfig.pnpLidarPoints
        self.saveToCache()

    def toggleQnpLidarPoints(self):
        self.camConfig.qnpLidarPoints = not self.camConfig.qnpLidarPoints
        self.saveToCache()

    def toggleDetectCorners(self):
        self.camConfig.detect_corners = not self.camConfig.detect_corners
        self.saveToCache()

    def toggleYoloInference(self):
        self._toggle('yoloInference', self.yoloInferenceCheckbox)

    def toggleYoloBiasTracking(self):
        self._toggle('yoloBiasTracking', self.yoloBiasCheckbox)

    def toggleDetectHorizon(self):
        self._toggle('detect_horizon', self.detectHorizonCheckbox)

    def togglePhaseCorrelation(self):
        self._toggle('phase_correlation', self.phaseCorrelationCheckbox)

    def toggleCrosshairs(self):
        self._toggle('crosshairs', self.crosshairsCheckbox)

    def toggleCubemap(self):
        self._toggle('cubemap', self.cubemapCheckbox)

    def toggleHud(self):
        self._toggle('hud', self.hudCheckbox)

    def toggleFactorgraph(self):
        self._toggle('factor_graph', self.factorgraphCheckbox)

    def toggleHyperFocus(self):
        self._toggle('hyper_focus', self.hyperfocusCheckbox)

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

    def toggleDetectTags(self):
        if self.detector is None:
            # self.detector = Detector(quad_decimate=1.5, quad_sigma =1.0, decode_sharpening=0.75)
            self.createDetector()
            self.camConfig.detectTags = True
        else:
            self.detector = None
            self.camConfig.detectTags = False
        self.saveToCache()

    def toggleUndistort(self):
        if not self.calibration.validCal:
            self.camConfig.undistort = False
            return

        self.camConfig.undistort = not self.camConfig.undistort
        if self.camConfig.undistort:
            self.undistortCheckbox.select()
        else:
            self.undistortCheckbox.deselect()

        self.saveToCache()

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
        self.gui.after(0, self._on_worker_exit)

    @staticmethod
    def convert_cv_to_pil(img):
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

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
        self.gui.after(0, self._on_worker_exit)
        return

    @staticmethod
    def _stride_for_speed(speed_abs: int) -> int:
        s = max(0, int(speed_abs))
        return 1 if s <= 1 else min(8, s)

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

    def _poll_keys(self, max_ms: int = 1) -> list[int]:
        keys = []
        k = cv2.waitKey(max_ms) & 0xFF
        if k not in (0, 0xFF, 255, -1):
            keys.append(k)
            for _ in range(8):  # drain a short burst
                k2 = cv2.waitKey(1) & 0xFF
                if k2 in (0, 0xFF, 255, -1):
                    break
                keys.append(k2)
        return keys

    def run_folder_reader(self):
        cv2.destroyAllWindows()
        cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)

        reverse_playback = False

        directory = Path(self.camConfig.imageFilepath).parent

        self.populate_idsTimes(str(directory))

        paths = []
        for rec in self.ImageTimeReader.idsTimes:
            p = Path(rec[0])
            paths.append(p if p.is_absolute() else (directory / p))

        num_images = len(paths)

        # timestamps (seconds), None -> infer at fixed spacing later
        ts_raw = []
        for name, ts in self.ImageTimeReader.idsTimes:
            ts_raw.append(None if ts is None else float(ts) + float(self.camConfig.cam_to_log_time_offset))

        t = self._make_timebase(ts_raw, self.camConfig.target_fps, num_images)

        img_slider = ImageSliderBar(num_images, refresh_hz=30.0)
        img_slider.play_speed = 1  # negative=rewind, 0=freeze, positive=forward
        pause = False
        last_nonzero_sign = 1

        last_stride = None
        last_speed = img_slider.play_speed

        # --- PAUSED CACHE: keep 1 frame while paused to avoid refetch spam ---
        paused_cached_idx = None
        paused_cached_frame = None
        printed_missing = set()  # avoid spamming the same missing file message

        # time offset
        try:
            offset_dict = pd.read_csv(directory / "__TIME_OFFSET.csv")
            self.camConfig.cam_to_log_time_offset = float(offset_dict['offset'][0])
        except FileNotFoundError:
            self.camConfig.cam_to_log_time_offset = 0.0

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
        pressed = set()
        pending_keys = []
        curr_idx = 0

        wall_start = time.monotonic()

        def sleep_until(deadline) -> list[int]:
            # Keep GUI responsive and capture bursts
            keys = []
            while True:
                remain = deadline - time.monotonic()
                if remain <= 0:
                    break
                keys.extend(self._poll_keys(int(max(1, remain * 100))))
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
                if img_slider.play_speed != last_speed:
                    s_abs = self._stride_for_speed(abs(img_slider.play_speed))

                    if img_slider.play_speed != 0:
                        last_nonzero_sign = (1 if img_slider.play_speed > 0 else -1)

                    if s_abs == 0:
                        pause = True
                    else:
                        pause = False
                        signed_stride = last_nonzero_sign * s_abs
                        if signed_stride != last_stride:
                            loader.set_stride(signed_stride)
                            last_stride = signed_stride

                        curr_idx = max(0, min(img_slider.curr_img_idx, num_images - 1))
                        loader.seek(curr_idx, clear_buffer=True)
                        reverse_playback = (last_nonzero_sign < 0)

                        # leaving pause → invalidate paused cache
                        paused_cached_idx = None
                        paused_cached_frame = None

                    last_speed = img_slider.play_speed

                # ===== fetch a frame =====
                if not pause:
                    # streaming mode: loader drives index
                    got = loader.get_next(timeout=0.02)

                    # if skipping (|stride|>1), drain extras so we show freshest
                    if last_stride and abs(last_stride) > 1 and got is not None:
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
                        img_slider.curr_img_idx = curr_idx

                    # leaving pause → invalidate paused cache
                    paused_cached_idx = None
                    paused_cached_frame = None

                else:
                    # ===== PAUSED MODE with CACHE =====
                    target_idx = max(0, min(img_slider.curr_img_idx, num_images - 1))

                    # If cache is invalid or user moved (z/c), fetch once; otherwise reuse cached frame
                    if paused_cached_frame is None or paused_cached_idx != target_idx:
                        # Seek ONCE; do NOT keep seeking every loop
                        loader.seek(target_idx, clear_buffer=True)

                        got = loader.get_next(timeout=0.5)  # give worker a bit more time while paused
                        if got is not None:
                            got_idx, frame = got
                            paused_cached_idx = got_idx
                            paused_cached_frame = frame
                            curr_idx = got_idx
                            img_slider.curr_img_idx = curr_idx
                        else:
                            # If file truly missing, print once; otherwise keep last cached frame (if any)
                            p = paths[target_idx]
                            if not Path(p).exists():
                                if p not in printed_missing:
                                    LOG.warning("Log file missing/failed: %s",
                                                self.ImageTimeReader.idsTimes[curr_idx][0])
                                    printed_missing.add(p)
                                paused_cached_frame = None
                                paused_cached_idx = None
                            # If file exists but frame not ready yet, DON'T print; keep previous cached frame
                    frame = paused_cached_frame

                # ===== display / HUD =====
                if frame is not None and Path(paths[curr_idx]).exists() and len(self.ImageTimeReader.idsTimes) > 0:

                    if self.camConfig.playback_mode == PlaybackSpeed.Fixed_fps:
                        period = 1.0 / self.camConfig.target_fps
                        target_time = wall_start + period * (
                            curr_idx if not reverse_playback else num_images - curr_idx)
                        if target_time < time.monotonic():
                            wall_start = time.monotonic() - 1.0 / max(0.001, self.camConfig.target_fps) * (
                                curr_idx if not reverse_playback else num_images - curr_idx)
                        pending_keys = sleep_until(target_time)

                    elif self.camConfig.playback_mode == PlaybackSpeed.Real_time:
                        # elapsed wall time scaled by speed (always non-negative)
                        rs = float(self.camConfig.rt_speed)
                        if rs <= 0:
                            rs = 1e-6
                        elapsed = (time.monotonic() - wall_start) * rs

                        # Map to a target time on the log timeline
                        # Forward: elapsed_ref = elapsed
                        # Reverse: elapsed_ref = (t[-1] - elapsed)
                        if reverse_playback:
                            elapsed_ref = (t[-1] - elapsed)
                        else:
                            elapsed_ref = elapsed

                        # Wrap-around handling with re-anchoring so playback loops continuously
                        if elapsed_ref < t[0]:
                            # Wrapped before start -> show last frame and re-anchor wall_start so we stay continuous
                            idx_target = num_images - 1
                            wall_start = time.monotonic() - ((t[idx_target] - t[0]) / rs if not reverse_playback
                                                             else ((t[-1] - t[idx_target]) / rs))
                        elif elapsed_ref > t[-1]:
                            # Wrapped past end -> show first frame and re-anchor
                            idx_target = 0
                            wall_start = time.monotonic() - ((t[idx_target] - t[0]) / rs if not reverse_playback
                                                             else ((t[-1] - t[idx_target]) / rs))
                        else:
                            # Inside range: pick the frame whose time is just <= elapsed_ref
                            idx_target = int(np.searchsorted(t, elapsed_ref, side='right') - 1)

                        idx_target = max(0, min(idx_target, num_images - 1))

                        if idx_target != curr_idx:
                            loader.seek(idx_target, clear_buffer=True)  # instant re-align
                            got = loader.get_next(timeout=0.02)
                            if got is not None:
                                curr_idx, frame = got

                        # small wait to avoid hot spinning when we're at the correct time
                        pending_keys.extend(self._poll_keys(10))

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
                else:
                    # path doesn’t exist (already printed in paused path; print here for streaming once)
                    p = paths[curr_idx]
                    if p not in printed_missing:
                        LOG.warning("Log file missing/failed: %s", self.ImageTimeReader.idsTimes[curr_idx][0])
                        printed_missing.add(p)

                pending_keys.extend(self._poll_keys(1))

                # --- Key handling: edge-triggered dispatcher ---
                while pending_keys:
                    key = pending_keys.pop(0)

                    # edge-detect helper (same semantics as your current on_key)
                    def on_key(kcode: int) -> bool:
                        if kcode == 255:
                            return False
                        if kcode in pressed:
                            return False
                        pressed.add(kcode)
                        return True

                    # reset edge-state on "no key"
                    if key in (255, 0xFF, 0):
                        pressed.clear()
                        continue

                    elif key == ord('f') and on_key(ord('f')):
                        # Switch between Fixed_fps and Real_time
                        old_mode = self.camConfig.playback_mode
                        maybe_ws = self._on_toggle_fps_mode(t, curr_idx)

                        # --- Re-anchor to keep the current frame fixed ---
                        self.camConfig.playback_mode = self.camConfig.playback_mode  # ensure updated
                        wall_start = self._reanchor_on_mode_change(
                            new_mode=self.camConfig.playback_mode,
                            curr_idx=curr_idx,
                            t=t,
                            last_nonzero_sign=last_nonzero_sign
                        )
                        self.saveToCache()

                    elif key == ord('c') and on_key(ord('c')):
                        self._on_step_forward(img_slider, loader, num_images)
                        pause = True
                        paused_cached_idx = None
                        paused_cached_frame = None
                        self.camConfig.playback_mode = PlaybackSpeed.Fixed_fps
                        loader.seek(img_slider.curr_img_idx, clear_buffer=True)

                    elif key == ord('z') and on_key(ord('z')):
                        self._on_step_back(img_slider, loader, num_images)
                        pause = True
                        paused_cached_idx = None
                        paused_cached_frame = None
                        self.camConfig.playback_mode = PlaybackSpeed.Fixed_fps
                        loader.seek(img_slider.curr_img_idx, clear_buffer=True)

                    elif key == ord(' ') and on_key(ord(' ')):  # space
                        # Toggle pause/play with clean re-anchoring on resume
                        wall_start = self._on_toggle_pause(img_slider, curr_idx, t, wall_start, last_nonzero_sign)

                    elif key == ord('d') and on_key(ord('d')):
                        wall_start = self._on_speed_up(
                            img_slider=img_slider,
                            curr_idx=curr_idx,
                            t=t,
                            wall_start=wall_start,
                            last_nonzero_sign=last_nonzero_sign,
                        )

                    elif key == ord('a') and on_key(ord('a')):
                        wall_start = self._on_speed_down(
                            img_slider=img_slider,
                            curr_idx=curr_idx,
                            t=t,
                            wall_start=wall_start,
                            last_nonzero_sign=last_nonzero_sign,
                        )

                    elif key == ord('w') and on_key(ord('w')):
                        self._on_toggle_overlays()

                    elif key == ord('s') and on_key(ord('s')):
                        self._on_mark_start(img_slider)

                    elif key == ord('e') and on_key(ord('e')):
                        self._on_mark_end(img_slider)

                    elif key == ord('r') and on_key(ord('r')):
                        last_nonzero_sign, wall_start = self._on_reverse(
                            img_slider=img_slider,
                            loader=loader,
                            last_nonzero_sign=last_nonzero_sign,
                            curr_idx=curr_idx,
                            t=t,
                            wall_start=wall_start,
                        )
                        reverse_playback = (last_nonzero_sign < 0)

                    # time offset nudges (small/medium/large)
                    elif key == ord(";") and on_key(ord(";")):
                        self._on_adjust_offset(-0.01)
                    elif key == ord("'") and on_key(ord("'")):
                        self._on_adjust_offset(+0.01)
                    elif key == ord('[') and on_key(ord('[')):
                        self._on_adjust_offset(-0.10)
                    elif key == ord(']') and on_key(ord(']')):
                        self._on_adjust_offset(+0.10)
                    elif key == ord('{') and on_key(ord('{')):
                        self._on_adjust_offset(-1.00)
                    elif key == ord('}') and on_key(ord('}')):
                        self._on_adjust_offset(+1.00)
                    elif key == ord('p') and on_key(ord('p')):
                        self._on_persist_offset()

                    elif key == 27 and on_key(27):  # ESC
                        self.threadStopper.set()
                        break

                    while self.making_gifOrVid:
                        time.sleep(0.1)
                pressed.clear()

        finally:
            cv2.destroyAllWindows()
            self.gui.after(0, self._on_worker_exit)
            if not self.shutting_down:
                img_slider.close()
            loader.stop()

    def _reanchor_on_mode_change(self, new_mode, curr_idx: int, t, last_nonzero_sign: int) -> float:
        """
        Re-anchor wall_start so the current frame stays fixed when switching modes,
        including while reversing. Uses time.monotonic() to match the main loop.
        """
        now = time.monotonic()

        if new_mode == PlaybackSpeed.Real_time:
            # Map wall clock to log time (direction-aware)
            rs = max(1e-6, float(self.camConfig.rt_speed))
            t0, tN = t[0], t[-1]
            if last_nonzero_sign < 0:
                # reverse: tN - (now - wall_start)*rs == t[curr_idx]
                return now - (tN - t[curr_idx]) / rs
            else:
                # forward: (now - wall_start)*rs == t[curr_idx] - t0
                return now - (t[curr_idx] - t0) / rs

        else:
            # Fixed-FPS: anchor to the correct phase for direction
            fps = max(0.001, float(self.camConfig.target_fps))
            num_images = len(t)
            phase = (num_images - curr_idx) if last_nonzero_sign < 0 else curr_idx
            return now - (phase / fps)

    def _rt_reanchor(self, now: float, curr_idx: int, t, rt_rate: float, sign: int) -> float:
        """Return a new wall_start so that the effective RT timeline still maps to t[curr_idx]."""
        rt_rate = max(1e-6, float(rt_rate))
        t0, tN = t[0], t[-1]
        if sign < 0:
            # reverse: tN - (now - wall_start)*rt_rate == t[curr_idx]
            return now - (tN - t[curr_idx]) / rt_rate
        else:
            # forward: (now - wall_start)*rt_rate == t[curr_idx] - t0
            return now - (t[curr_idx] - t0) / rt_rate

    def _on_reverse(self, img_slider, loader, last_nonzero_sign: int, curr_idx: int, t, wall_start: float):
        """
        Toggle playback direction without jumping the current frame.
        Returns: (new_last_nonzero_sign, new_wall_start)
        """
        # New direction (+1 forward, -1 reverse)
        new_sign = -1 if last_nonzero_sign > 0 else 1

        # If actively playing, flip speed sign and update stride, then realign buffer at current index.
        if getattr(img_slider, "play_speed", 0) != 0:
            img_slider.play_speed = -img_slider.play_speed
            try:
                s_abs = self._stride_for_speed(abs(img_slider.play_speed))
            except AttributeError:
                s_abs = max(1, int(round(abs(img_slider.play_speed))))
            if s_abs > 0:
                loader.set_stride(new_sign * s_abs)
                loader.seek(curr_idx, clear_buffer=True)

        now = time.monotonic()

        if getattr(self.camConfig, "playback_mode", None) == PlaybackSpeed.Real_time:
            # --- Real-time: direction-aware re-anchor on the log timeline ---
            rs = max(1e-6, float(self.camConfig.rt_speed))
            t0, tN = t[0], t[-1]
            if new_sign < 0:
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
            phase = (num_images - curr_idx) if new_sign < 0 else curr_idx
            wall_start = now - period * phase

        return new_sign, wall_start

    def _reanchor_wall_start_rt(self, curr_idx: int, t: np.ndarray, wall_start: float) -> float:
        """Keep the current frame stationary when rt_speed changes."""
        rs = float(self.camConfig.rt_speed)
        if abs(rs) < 1e-6:  # avoid div-by-zero; treat as tiny forward speed
            rs = 1e-6
        now = time.monotonic()
        # Align so: (now - wall_start) * rt_speed == t[curr_idx] - t[0]
        return now - ((t[curr_idx] - t[0]) / rs)

    def _make_timebase(self, ts_raw: list[float | None], fallback_fps: float, n: int) -> np.ndarray:
        """
        Build a monotone, normalized timebase (seconds) from possibly-missing timestamps.
        - If all timestamps are None: synthesize from fallback_fps.
        - Else: linearly interpolate gaps; normalize to t[0] == 0.0.
        """
        t = np.array([np.nan if v is None else float(v) for v in ts_raw], dtype='float64')
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

    def _on_toggle_fps_mode(self, t, curr_idx):
        """Swap Fixed_fps <-> Real_time, preserving perceived position."""
        self.camConfig.playback_mode = self.camConfig.playback_mode.next()
        if self.camConfig.playback_mode == PlaybackSpeed.Real_time:
            self.camConfig.rt_speed = 1.0
            # align wall clock to current frame's time
            return time.monotonic() - (t[curr_idx] - t[0]) / max(1e-9, self.camConfig.rt_speed)
        else:
            # fall back to fixed-fps; let caller recompute the synthetic wall_start
            return None

    def _on_step_forward(self, img_slider, loader, num_images):
        img_slider.curr_img_idx = min(img_slider.curr_img_idx + 1, num_images - 1)
        img_slider.play_speed = 0

    def _on_step_back(self, img_slider, loader, num_images):
        img_slider.curr_img_idx = max(img_slider.curr_img_idx - 1, 0)
        img_slider.play_speed = 0

    def _on_toggle_pause(self, img_slider, curr_idx: int, t, wall_start: float, last_nonzero_sign: int) -> float:
        """
        Toggle pause/play.
        - If playing: pause (play_speed -> 0) and keep wall_start (no jump on still frame).
        - If paused: resume with last_nonzero_sign and re-anchor so current frame is preserved.
        Returns new wall_start.
        """
        playing = getattr(img_slider, "play_speed", 0) != 0

        if playing:
            # Pause: freeze on the current frame without touching anchors
            img_slider.play_speed = 0.0
            return wall_start

        # Resume: pick a reasonable magnitude (keep previous abs speed if you store it)
        prev_mag = getattr(self, "_resume_speed_mag", None)
        if prev_mag is None:
            prev_mag = 1.0
        img_slider.play_speed = float(last_nonzero_sign or 1) * prev_mag

        now = time.monotonic()
        if getattr(self.camConfig, "playback_mode", None) == PlaybackSpeed.Real_time:
            rs = max(1e-6, float(self.camConfig.rt_speed))
            t0, tN = t[0], t[-1]
            if last_nonzero_sign < 0:
                wall_start = now - (tN - t[curr_idx]) / rs
            else:
                wall_start = now - (t[curr_idx] - t0) / rs
        else:
            fps = max(0.001, float(self.camConfig.target_fps))
            wall_start = now - (curr_idx / fps)

        return wall_start

    def _on_speed_up(self, img_slider, curr_idx: int, t, wall_start: float, last_nonzero_sign: int) -> float:
        """
        Increase playback speed.
        - RT mode: multiply rt_speed, then re-anchor so current frame stays put.
        - Fixed-FPS: increase target_fps, then re-anchor to current frame index.
        Returns new wall_start.
        """
        if getattr(self.camConfig, "playback_mode", None) == PlaybackSpeed.Real_time:
            # adjust rate
            self.camConfig.rt_speed = min(float(self.camConfig.rt_speed) * 1.25, 128.0)
            # direction-aware reanchor
            now = time.monotonic()
            rs = max(1e-6, float(self.camConfig.rt_speed))
            t0, tN = t[0], t[-1]
            if last_nonzero_sign < 0:
                # reverse: tN - (now - wall_start)*rs == t[curr_idx]
                wall_start = now - (tN - t[curr_idx]) / rs
            else:
                # forward: (now - wall_start)*rs == t[curr_idx] - t0
                wall_start = now - (t[curr_idx] - t0) / rs
        else:
            # Fixed-FPS
            self.camConfig.target_fps = min(float(self.camConfig.target_fps) * 1.25, 240.0)
            fps = max(0.001, float(self.camConfig.target_fps))
            wall_start = time.perf_counter() - (curr_idx / fps)
        return wall_start

    def _on_speed_down(self, img_slider, curr_idx: int, t, wall_start: float, last_nonzero_sign: int) -> float:
        """
        Decrease playback speed.
        - RT mode: divide rt_speed, then re-anchor so current frame stays put.
        - Fixed-FPS: decrease target_fps, then re-anchor to current frame index.
        Returns new wall_start.
        """
        if getattr(self.camConfig, "playback_mode", None) == PlaybackSpeed.Real_time:
            self.camConfig.rt_speed = max(float(self.camConfig.rt_speed) / 1.25, 0.01)
            now = time.monotonic()
            rs = max(1e-6, float(self.camConfig.rt_speed))
            t0, tN = t[0], t[-1]
            if last_nonzero_sign < 0:
                wall_start = now - (tN - t[curr_idx]) / rs
            else:
                wall_start = now - (t[curr_idx] - t0) / rs
        else:
            self.camConfig.target_fps = max(float(self.camConfig.target_fps) / 1.25, 0.1)
            fps = max(0.001, float(self.camConfig.target_fps))
            wall_start = time.perf_counter() - (curr_idx / fps)
        return wall_start

    def _on_toggle_overlays(self):
        self.toggleUndistort()
        self.toggleYoloInference()
        self.toggleDetectHorizon()
        self.toggleHyperFocus()
        self.toggleFactorgraph()

    def _on_mark_start(self, img_slider):
        self.camConfig.start_export_idx = img_slider.curr_img_idx
        if self.camConfig.end_export_idx < self.camConfig.start_export_idx:
            self.camConfig.end_export_idx = self.camConfig.start_export_idx + 1
        self.exportStartFrame.configure(text=f'Start Frame: {self.camConfig.start_export_idx}')
        self.exportEndFrame.configure(text=f'End Frame: {self.camConfig.end_export_idx}')
        self.saveToCache()

    def _on_mark_end(self, img_slider):
        self.camConfig.end_export_idx = img_slider.curr_img_idx
        if self.camConfig.end_export_idx < self.camConfig.start_export_idx:
            self.camConfig.start_export_idx = max(0, self.camConfig.end_export_idx - 1)
        self.exportStartFrame.configure(text=f'Start Frame: {self.camConfig.start_export_idx}')
        self.exportEndFrame.configure(text=f'End Frame: {self.camConfig.end_export_idx}')
        self.saveToCache()

    def _on_adjust_offset(self, delta: float):
        self.camConfig.cam_to_log_time_offset += float(delta)

    def _on_persist_offset(self):
        self.write_offset_csv()
        # if you added logging already, this becomes LOG.info(...)
        print(f"Saved offset {self.camConfig.cam_to_log_time_offset:+.3f}s to __TIME_OFFSET.csv")

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

        if self.detector is not None:
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
            self.run_yolo(img_time)
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
        np.set_printoptions(precision=5, threshold=np.inf, suppress=True)

        points = None
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
        """Return direction vectors for each cube face, shape: (6, H, W, 3)"""
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
        """Return direction vectors for each cube face, shape: (6, H, W, 3)"""
        axes = {
            'front': ([0, 0, 1], [0, -1, 0])
        }

        self.faces_dirs = {}
        rng = np.linspace(-1, 1, self.face_size)
        xx, yy = np.meshgrid(rng, -rng)  # Flip Y for image coordinates

        for name, (center, up) in axes.items():
            center = np.array(center)
            up = np.array(up)
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
            # self.curr_frame = cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_NEAREST,
            #                             borderMode=cv2.BORDER_CONSTANT)

            # self.curr_frame = cv2.undistort(src=frame,
            #                                 cameraMatrix=self.calibration.getCameraMatrix(),
            #                                 distCoeffs=self.calibration.getDistortion())

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
            # probeTip_3d = np.array([[0.0], [0.0], [0.0]])
            # self.projectProbe, _ = cv2.projectPoints(probeTip_3d, rvec=rvec, tvec=tvec,
            #                                          cameraMatrix=self.calibration.getCameraMatrix(),
            #                                          distCoeffs=distParams)

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

        ret = False
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

    def run_yolo(self, img_time):
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
        image_path (np.array): the image
        center (tuple): (x, y) coordinates of the circle's center.
        radius (int): Radius of the circle.
        dim_factor (float): Dimming factor (0 to 1, 0 for black, 1 for no dimming).
    """

    if y_axes is None:
        radius = x_axes
        if dim_factor == 0.0:
            return dim_entirely(frame, center, radius)

        # 1. Create a mask
        mask = np.zeros(frame.shape[:2], dtype="uint8")  # Black mask
        cv2.circle(mask, (int(center[0]), int(center[1])), int(radius), 255, -1)  # White circle on mask

    else:
        mask = np.zeros(frame.shape[:2], dtype='uint8')
        # cv2.rectangle(mask, (int(center[0]-x_axes),int(center[1]-y_axes)),(int(center[0]+x_axes),int(center[1]+y_axes)),
        #               color=255, thickness=-1)
        cv2.ellipse(mask, (int(center[0]), int(center[1])), (int(x_axes), int(y_axes)),
                    angle=0, startAngle=0, endAngle=360, color=255, thickness=-1)

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
        image_path (np.array): the image
        center (tuple): (x, y) coordinates of the circle's center.
        radius (int): Radius of the circle.
        dim_factor (float): Dimming factor (0 to 1, 0 for black, 1 for no dimming).
    """

    # 1. Create a mask
    mask = np.zeros(frame.shape[:2], dtype="uint8")  # Black mask
    cv2.circle(mask, (int(center[0]), int(center[1])), int(radius), 255, -1)  # White circle on mask

    # 3. Copy the original circle area back to the dimmed image
    return cv2.bitwise_and(frame, frame, mask=mask)


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)
