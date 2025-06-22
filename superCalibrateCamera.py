import pickle, copy, os, time, threading, cv2, glob, re, yolo, ctypes, vmbpy.c_binding

import numpy as np
import customtkinter as ctk
from vmbpy import *
from tkinter import filedialog
from threading import Thread
from cv2_enumerate_cameras import enumerate_cameras
from Calibration import Calibration
from LidarTruth import TruthPoints
from enum import Enum
from PIL import Image
from RollInterpreter import RollReader as RollRdr
from numpy import sin, cos, tan, atan2, deg2rad, rad2deg, pi as PI

from FG_DrogueOnly import FactorGraph
from ImageTimeReader import ImageTimeReader

# import superCalibrate as superCal
#pip install cv2_enumerate_cameras
#or
#pip install git+https://github.com/chinaheyu/cv2_enumerate_cameras.git

GREEN = '#2FA572'


# class video_player():
#     def __init__(self, img_filepaths:list):
#         self.img_id = 0
#         self.play_speed = 1
#         self.pause = False
#         self.temp_unpause = False
#         self.quit_now = False
#         self.thread = thread_with_exception(1, self.run_new_menu)
#         self.pop_up = None
#         self.play_pause_button = None
#         self.exit_player_button = None
#         self.imageList = natural_sort(img_filepaths)
#         self.thread.start()
#
#     def run_new_menu(self):
#         if self.pop_up is None:
#             self.pop_up = ctk.CTkToplevel()
#             self.pop_up.title('Video Controls')
#             self.pop_up.geometry('600x250+300+300')
#             self.pop_up.grid_columnconfigure(0, weight=1, uniform='equal')
#             self.pop_up.grid_columnconfigure(1, weight=1, uniform='equal')
#             self.pop_up.grid_columnconfigure(2, weight=1, uniform='equal')
#         if self.play_pause_button is None:
#             self.play_pause_button = ctk.CTkButton(master=self.pop_up, text="Pause", command=self.play_pause)
#             self.play_pause_button.grid(row=0, column=1, sticky='nsew')
#         if self.exit_player_button is None:
#             self.exit_player_button = ctk.CTkButton(master=self.pop_up, text='Quit', command=self.exit_player)
#             self.exit_player_button.grid(row=1, column=1, sticky='nsew')
#
#     def play_pause(self):
#         self.pause = not self.pause
#
#     def exit_player(self):
#         self.quit_now = True
#         self.close()
#
#     def close(self):
#         self.pop_up.destroy()
#         self.thread.raise_exception()
#         self.thread.join()
#         del self
#
#     def next_frame(self)->np.array:
#
#         key = cv2.waitKey(1)
#
#         if key == 99:
#             self.img_id += 1
#             self.play_speed = 0
#             self.temp_unpause = True
#             self.pause = True
#         if key == 122:
#             self.img_id -= 1
#             self.play_speed = 0
#             self.temp_unpause = True
#             self.pause = True
#
#         if key == 32:
#             self.play_speed = 0
#             if not self.pause:
#                 self.pause = not self.pause
#                 self.play_speed = 1
#         if key == 100:
#             self.play_speed += 1
#             self.pause = False
#         if key == 97:
#             self.play_speed -= 1
#             self.pause = False
#
#         if key == 27:
#             self.quit_now = True
#
#         frame = None
#
#         if not self.pause or self.temp_unpause:
#             self.temp_unpause = False
#             self.img_id = (self.img_id + self.play_speed) % len(self.imageList)
#             frame = cv2.imread(self.imageList[self.img_id])
#
#         return frame, self.quit_now

class thread_with_exception(Thread):
    def __init__(self, name, func):
        Thread.__init__(self)
        self.name = name
        self.func = func

    def run(self):
        try:
            while True:
                self.func()
        finally:
            pass

    def get_id(self):
        if hasattr(self, '_thread_id'):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id

    def raise_exception(self):
        thread_id = self.get_id()
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, ctypes.py_object(SystemExit))
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
            print('Exception Raise Failure')


class ImageSliderBar:
    def __init__(self, num_images):
        self.pop_up = ctk.CTkToplevel()
        self.pop_up.focus_force()
        self.pop_up.geometry('400x200')
        self.pop_up.grid_columnconfigure(0, weight=1)
        self.pop_up.grid_rowconfigure([0, 1], weight=1)
        self.num_images = num_images
        self.play_speed = 1
        self.curr_img_idx = 0

        self.slider = ctk.CTkSlider(self.pop_up, from_=0, to=num_images, command=self.update_img_id, height=40)
        self.slider.set(0)
        self.slider.grid(sticky='ew')

    def update_img_id(self, new_img_idx):
        self.curr_img_idx = int(new_img_idx)

    def next_id(self):
        self.curr_img_idx = (self.curr_img_idx + self.play_speed) % self.num_images
        self.slider.set(self.curr_img_idx)
        return self.curr_img_idx

    def close(self):
        self.pop_up.destroy()
        self.pop_up.update()


class GifMaker:
    def __init__(self, gif_name: str = "Output",
                 width_height: tuple[int, int] = (864, 864),
                 img_duration: int = 100,
                 loop: int = 0):
        self.pop_up = ctk.CTkToplevel()
        self.pop_up.lift()
        self.file_path = ''
        self.width, self.height = width_height
        self.output_name = gif_name + ".gif"
        self.img_duration = img_duration
        self.loop = loop
        self.thread = thread_with_exception("GifMaker Thread", self.make_gif)
        self.is_threading = False
        self.execute_thread()

    def execute_thread(self):
        if not self.is_threading:
            self.is_threading = True
            self.thread.start()

    def make_gif(self):
        time.sleep(1)
        self.is_threading = False
        self.thread.raise_exception()

    def numerical_sort(self, file_name):
        try:
            return int(file_name.split('.')[0])
        except (ValueError, IndexError):
            return float('inf')


class Gabor:
    def __init__(self):
        self.pop_up = ctk.CTkToplevel()
        self.pop_up.lift()
        self.ksize = (31, 31)
        self.sigma = 3.0
        self.theta = 0.0
        self.lambd = 10.0
        self.gamma = 0.63
        self.psi = 0.0

        self.sigma_label = None
        self.theta_label = None
        self.lambd_label = None
        self.gamma_label = None

        self.pop_up.geometry('200x300')
        self.pop_up.grid_columnconfigure([0, 1], weight=1)
        self.configure_pop_up()

    def configure_pop_up(self):
        rowID = 0
        self.sigma_label = ctk.CTkLabel(self.pop_up, text=f'Sigma: {self.sigma:.2f}')
        self.sigma_label.grid(row=rowID, column=0, padx=5, pady=5)
        rowID += 1
        sigma_slider = ctk.CTkSlider(self.pop_up, from_=0.01, to=10.0, command=self.update_sigma)
        sigma_slider.set(self.sigma)
        sigma_slider.grid(row=rowID, column=0, columnspan=2, padx=5, pady=5)
        rowID += 1

        self.theta_label = ctk.CTkLabel(self.pop_up, text=f'Theta: {rad2deg(self.theta):.2f}')
        self.theta_label.grid(row=rowID, column=0, padx=5, pady=5)
        rowID += 1
        theta_slider = ctk.CTkSlider(self.pop_up, from_=0.0, to=PI * 2.0, command=self.update_theta)
        theta_slider.set(self.theta)
        theta_slider.grid(row=rowID, column=0, columnspan=2, padx=5, pady=5)
        rowID += 1

        self.lambd_label = ctk.CTkLabel(self.pop_up, text=f'Lambda: {self.lambd:.2f}')
        self.lambd_label.grid(row=rowID, column=0, padx=5, pady=5)
        rowID += 1
        lambd_slider = ctk.CTkSlider(self.pop_up, from_=0.0, to=10.0, command=self.update_lambd)
        lambd_slider.set(self.lambd)
        lambd_slider.grid(row=rowID, column=0, columnspan=2, padx=5, pady=5)
        rowID += 1

        self.gamma_label = ctk.CTkLabel(self.pop_up, text=f'Gamma: {self.gamma:.2f}')
        self.gamma_label.grid(row=rowID, column=0, padx=5, pady=5)
        rowID += 1
        gamma_slider = ctk.CTkSlider(self.pop_up, from_=0.0, to=1.0, command=self.update_gamma)
        gamma_slider.set(self.gamma)
        gamma_slider.grid(row=rowID, column=0, columnspan=2, padx=5, pady=5)
        rowID += 1

    def update_sigma(self, slider_value):
        self.sigma = slider_value
        self.sigma_label.configure(text=f'Sigma: {self.sigma:.2f}')

    def update_theta(self, slider_value):
        self.theta = slider_value
        self.theta_label.configure(text=f'Theta: {rad2deg(self.theta):.2f}')

    def update_lambd(self, slider_value):
        self.lambd = slider_value
        self.lambd_label.configure(text=f'Lambda: {self.lambd:.2f}')

    def update_gamma(self, slider_value):
        self.gamma = slider_value
        self.gamma_label.configure(text=f'Gamma: {self.gamma:.2f}')

    def filter_kernel(self):
        if not self.pop_up.winfo_exists():
            self.pop_up = ctk.CTkToplevel()
            self.pop_up.focus_force()
            self.pop_up.geometry('200x500')
            self.pop_up.grid_columnconfigure([0, 1], weight=1)
            self.configure_pop_up()

        return cv2.getGaborKernel(self.ksize,
                                  self.sigma,
                                  self.theta,
                                  self.lambd,
                                  self.gamma,
                                  self.psi)

    def close(self):
        self.pop_up.destroy()
        self.pop_up.update()


class ImageSource(Enum):
    Camera_Stream = 'Camera Stream'
    Static_Image = 'Static Image'
    Stream_from_Folder = 'Stream from Folder'


class ImageKernels(Enum):
    Unchanged = 'Unchanged'  #None
    Sharpen = 'Sharpen'  #np.array([[0, -1, 0],[-1, 5, -1], [0, -1, 0]])
    GaussBlur = 'GaussBlur'  #np.array([[1, 4, 6, 4, 1],[4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]]) / 256.0
    EdgeDetect = 'EdgeDetect'  #np.array([[-1, -1, -1],[-1, 8, -1], [-1, -1, -1]])
    HorizontalEdgeDetect = 'HorizontalEdgeDetect'
    VerticalEdgeDetect = 'VerticalEdgeDetect'
    BoxBlur = 'BoxBlur'
    SobelEdgeDetectHorizontal = 'SobelEdgeDetectHorizontal'
    SobelEdgeDetectVertical = 'SobelEdgeDetectVertical'
    LaplaceEdgeDetect = 'LaplaceEdgeDetect'
    Gabor = 'Gabor'
    ScharrEdgeDetectHorizontal = 'ScharrEdgeDetectHorizontal'
    ScharrEdgeDetectVertical = 'ScharrEdgeDetectVertical'
    Unsharp = 'Unsharp'


class CameraConfig():
    def __init__(self):
        self.cam_index = 0
        self.detectTags = False
        self.undistort = False
        self.projectLidarPoints = False
        self.yoloInference = False
        self.secondsBetweenImages = 1.0
        self.recording = False
        self.aprilTagSize = 0.168
        self.imageSource = ImageSource.Camera_Stream
        self.imageFilepath = ''
        self.lidarFilepath = None
        self.yoloFilepath = ''
        self.detect_corners = False
        self.detect_horizon = False
        self.factor_graph = False
        self.hyper_focus = False
        self.phase_correlation = False
        self.crosshairs = False
        self.cubemap = False
        self.hud = False
        self.hud_data_filepath = ''

        self.yolo_conf = 0.75
        self.yolo_iou = 1.00

        self.processingKernel = ImageKernels.Unchanged

    def copy(self, configToCopy):
        for obj in configToCopy.__dict__:
            try:
                self.__dict__[obj] = configToCopy.__dict__[obj]
            except KeyError:
                # Allows for versioning issues, changed naming conventions.
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
        self.GaborFilter = None
        self.radius = 800
        self.ellipse_x_axis = 800
        self.ellipse_y_axis = 800
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
        self.face_size = None
        self.faces_dirs = None
        self.cubemap_faces = None
        self.map_x = None
        self.map_y = None
        self.rollReader = RollRdr()

        self.bank_indicator_points = None

        self.last_image = None

        self.available_sources = [source.value for source in ImageSource]

        self.streamOrImgCombo = ctk.CTkComboBox(self.cam_frame, values=self.available_sources,
                                                command=self.sourceUpdate)
        self.startStreamButton = ctk.CTkButton(master=self.cam_frame, text='Start Stream', fg_color='red',
                                               hover_color='blue')
        self.recordButton = ctk.CTkButton(master=self.cam_frame, text='Saving Imagery', fg_color='green',
                                          hover_color='navy', command=self.recordOff)
        self.selectCameraCombo = ctk.CTkComboBox(self.cam_frame, values=list(self.indexDict.keys()),
                                                 command=self.selectCamera)
        self.selectFolderLabel = ctk.CTkLabel(self.cam_frame,
                                              text="../" + os.path.basename(os.path.normpath(self.filepath)))
        self.selectTruthPointsButton = ctk.CTkButton(master=self.cam_frame, text='Select LIDAR Points',
                                                     hover_color='blue', command=self.selectLidarFile)
        self.selectFlightLogButton = ctk.CTkButton(master=self.cam_frame, text='Select Flight Log File',
                                                     hover_color='blue', command=self.selectLogFile)

        if self.camConfig.lidarFilepath is not None:
            self.selectTruthPointsLabel = ctk.CTkLabel(self.cam_frame, text="../" + os.path.basename(
                os.path.normpath(self.camConfig.lidarFilepath)))
        else:
            self.selectTruthPointsLabel = ctk.CTkLabel(self.cam_frame, text='No Truth Loaded')

        if self.camConfig.hud_data_filepath is not None:
            self.selectFlightLogLabel = ctk.CTkLabel(self.cam_frame, text="../" + os.path.basename(
                os.path.normpath(self.camConfig.hud_data_filepath)))
        else:
            self.selectTruthPointsLabel = ctk.CTkLabel(self.cam_frame, text='No Truth Loaded')

        self.lidarTruthPoints = TruthPoints()
        self.selectYOLO_folderButton = ctk.CTkButton(self.cam_frame, text='Select YOLO Folder', fg_color=GREEN,
                                                     command=self.selectYoloFolder)
        self.selectYOLO_folderLabel = ctk.CTkLabel(self.cam_frame,
                                                   text='../' + os.path.basename(
                                                       os.path.normpath(self.camConfig.yoloFilepath)))
        self.selectCalibLabel = None
        self.undistortCheckbox = ctk.CTkCheckBox(self.cam_frame, text='Undistort')
        self.detectAprilTagsCheckbox = ctk.CTkCheckBox(self.cam_frame, text='Detect April Tags')
        self.detectHorizonCheckbox = ctk.CTkCheckBox(self.cam_frame, text='Detect Horizon')
        self.yoloInferenceCheckbox = ctk.CTkCheckBox(self.cam_frame, text='Run YOLO on image',
                                                     command=self.toggleYoloInference)
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

        self.loadFromCache()
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
        self.camFrameGeometry = '455x770'
        self.cam_frame.grid_rowconfigure(list(range(3)), weight=1)  # configure grid system
        self.cam_frame.grid_columnconfigure(list(range(3)), weight=1)
        self.t1 = None
        self.lastWidth = 1
        self.lastHeight = 1
        self.saveToCache()

    def loadFromCache(self):

        if not os.path.exists('camConfig_cache.pkl'):
            self.filepath = os.getcwd()
            return

        with open('camConfig_cache.pkl', 'rb') as camConfigOpen:
            self.camConfig.copy(pickle.load(camConfigOpen))
            self.filepath = copy.copy(pickle.load(camConfigOpen))
            self.calibFile = copy.copy(pickle.load(camConfigOpen))

        self.selectFolderLabel.configure(text="../" + os.path.basename(os.path.normpath(self.filepath)))

        self.ingestCalibration()

        if not self.camConfig.hud_data_filepath == '':
            self.rollReader.read_files(self.camConfig.hud_data_filepath)
        self.updateLidarLabel()
        self.updateYOLOLabel()
        self.updateFlightLogLabel()

        self.yoloSession.setNewFolder(self.camConfig.yoloFilepath)
        self.yoloSession.conf = self.camConfig.yolo_conf
        self.confSliderLabel.configure(text='Conf: ' + f'{self.camConfig.yolo_conf:.2f}')
        self.yoloSession.iou = self.camConfig.yolo_iou
        self.iouSliderLabel.configure(text='IOU: ' + f'{self.camConfig.yolo_iou:.2f}')

        self.loadTruthPoints()

    def saveToCache(self):
        with open('camConfig_cache.pkl', 'wb') as f:
            pickle.dump(self.camConfig, f)
            pickle.dump(self.filepath, f)
            pickle.dump(self.calibFile, f)

    def selectFolder(self):
        fp = self.askFilepath(self.filepath + "/..", "Select Imagery Folder")
        if fp is not None:
            self.filepath = fp
            self.saveToCache()
            self.loadFromCache()

    def loadCalibration(self):
        if self.calibFile != '':
            initial_dir = self.calibFile
        else:
            initial_dir = self.filepath

        poss_filepath = filedialog.askopenfilename(initialdir=initial_dir + '/..',
                                                   title='Select Folder of Calibration')

        if poss_filepath != '':
            self.calibFile = poss_filepath
            self.ingestCalibration()

    def selectLidarFile(self):
        if self.camConfig.lidarFilepath is None:
            poss_filepath = filedialog.askopenfilename(initialdir=self.filepath + '/..',
                                                       title='Select LIDAR Truth Points')
        else:
            poss_filepath = filedialog.askopenfilename(initialdir=self.camConfig.lidarFilepath + '/..',
                                                       title='Select LIDAR Truth Points')
        if poss_filepath != '':
            self.camConfig.lidarFilepath = poss_filepath
            self.updateLidarLabel()
            self.loadTruthPoints()
            self.saveToCache()

    def selectLogFile(self):
        if self.camConfig.hud_data_filepath == '':
            poss_filepath = filedialog.askdirectory(initialdir=self.filepath + '/..',
                                                       title='Select Flight Log Data')
        else:
            poss_filepath = filedialog.askdirectory(initialdir=self.camConfig.hud_data_filepath + '/..',
                                                       title='Select Flight Log Data')
        if poss_filepath != '':
            self.camConfig.hud_data_filepath = poss_filepath
            self.rollReader.read_files(poss_filepath)
            self.updateFlightLogLabel()
            self.saveToCache()


    def selectYoloFolder(self):
        if self.camConfig.yoloFilepath is None:
            poss_filepath = filedialog.askdirectory(initialdir=os.getcwd() + '/..',
                                                    title='Select YOLO Folder')
        else:
            poss_filepath = filedialog.askdirectory(initialdir=self.camConfig.yoloFilepath + '/..',
                                                    title='Select YOLO Folder')
        if poss_filepath != '':
            self.camConfig.yoloFilepath = poss_filepath
            self.updateYOLOLabel()
            self.yoloSession.setNewFolder(self.camConfig.yoloFilepath)
            self.saveToCache()

    def updateLidarLabel(self):
        if self.camConfig.lidarFilepath is not None:
            self.selectTruthPointsLabel.configure(text=os.path.basename(self.camConfig.lidarFilepath))

    def updateFlightLogLabel(self):
        if self.camConfig.hud_data_filepath is not None:
            self.selectFlightLogLabel.configure(text=os.path.basename(self.camConfig.hud_data_filepath))

    def updateYOLOLabel(self):
        if self.camConfig.yoloFilepath is not None:
            self.selectYOLO_folderLabel.configure(text=os.path.basename(self.camConfig.yoloFilepath))

    def loadTruthPoints(self):
        if self.camConfig.lidarFilepath is not None:
            with open(self.camConfig.lidarFilepath, 'rb') as f:
                test = pickle.load(f)
                self.lidarTruthPoints.copy(test)

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
            self.selectCalibLabel.configure(text=os.path.basename(os.path.normpath(self.calibFile)),
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
                    print(f'Could not open camera: {e}')
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
            print(f'Error processing frame: {e}')

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
                self.startStreamButton.grid(row=rowID, column=0, padx=5, pady=5)
            if not self.selectCameraCombo.grid_info():
                self.selectCameraCombo.grid(row=rowID, column=1, padx=5, pady=5)

        elif self.camConfig.imageSource == ImageSource.Static_Image:

            if not self.singleImageFolderSelect.grid_info():
                self.singleImageFolderSelect.grid(row=1, column=0, padx=5, pady=5)
            if not self.singleImageTextButton.grid_info():
                self.singleImageTextButton.grid(row=1, column=1, padx=5, pady=5)

        elif self.camConfig.imageSource == ImageSource.Stream_from_Folder:

            if not self.multiImageFolderSelect.grid_info():
                self.multiImageFolderSelect.grid(row=1, column=0, padx=5, pady=5)
            if not self.multiImageTextButton.grid_info():
                self.multiImageTextButton.grid(row=1, column=1, padx=5, pady=5)

        else:
            raise ValueError(f'Unknown Image selection mode: {self.camConfig.imageSource}')

        self.saveToCache()

    def selectImagesFilepath(self):
        if self.camConfig.imageFilepath is None:
            initDir = self.filepath + '/..'
        else:
            initDir = self.camConfig.imageFilepath  #os.path.normpath(self.camConfig.imageFilepath)

        poss_file = filedialog.askopenfilename(initialdir=initDir, title="Select Image")
        if poss_file != '':
            self.camConfig.imageFilepath = poss_file
            self.singleImageTextButton.configure(text=os.path.basename(self.camConfig.imageFilepath))
            self.multiImageTextButton.configure(text=os.path.basename(os.path.dirname(self.camConfig.imageFilepath)))

            self.saveToCache()

    def setupFrame(self):
        rowID = 0

        self.streamOrImgCombo = ctk.CTkComboBox(self.cam_frame,
                                                values=['Camera Stream', 'Static Image', 'Stream from Folder'],
                                                command=self.sourceUpdate)
        self.streamOrImgCombo.grid(row=rowID, column=0, padx=5, pady=5)
        rowID += 1

        self.startStreamOff()

        self.singleImageTextButton.configure(command=self.startStreamOn)
        if self.camConfig.imageFilepath is not None:
            self.singleImageTextButton.configure(text=os.path.basename(self.camConfig.imageFilepath))

        self.multiImageTextButton.configure(command=self.startStreamOn)
        if self.camConfig.imageFilepath is not None:
            self.multiImageTextButton.configure(text=os.path.basename(os.path.dirname(self.camConfig.imageFilepath)))

        self.streamOrImgCombo.set(self.camConfig.imageSource.value)
        self.sourceUpdate(self.camConfig.imageSource.value)

        rowID += 1

        selectFolderButton = ctk.CTkButton(self.cam_frame, text='Select Save Folder', command=self.selectFolder)
        selectFolderButton.grid(row=rowID, column=0, padx=5, pady=5)

        self.selectFolderLabel.grid(row=rowID, column=1, padx=5, pady=5)
        rowID += 1

        selectCalibButton = ctk.CTkButton(self.cam_frame, text='Select Calibration', command=self.loadCalibration)
        selectCalibButton.grid(row=rowID, column=0, padx=5, pady=5)

        self.selectCalibLabel = ctk.CTkLabel(self.cam_frame,
                                             text="../" + os.path.basename(os.path.normpath(self.calibFile)))
        self.selectCalibLabel.grid(row=rowID, column=1, padx=5, pady=5)
        rowID += 1

        self.selectTruthPointsButton.grid(row=rowID, column=0, padx=5, pady=5)
        self.selectTruthPointsLabel.grid(row=rowID, column=1, padx=5, pady=5)
        rowID += 1

        self.selectYOLO_folderButton.grid(row=rowID, column=0, padx=5, pady=5)
        self.selectYOLO_folderLabel.grid(row=rowID, column=1, padx=5, pady=5)
        rowID += 1

        self.selectFlightLogButton.grid(row=rowID, column=0, padx=5, pady=5)
        self.selectFlightLogLabel.grid(row=rowID, column=1, padx=5, pady=5)
        rowID += 1

        self.confSliderLabel.grid(row=rowID, column=0, padx=5, pady=5)
        self.confSliderBar.grid(row=rowID, column=1, padx=5, pady=5)
        rowID += 1

        self.iouSliderLabel.grid(row=rowID, column=0, padx=5, pady=5)
        self.iouSliderBar.grid(row=rowID, column=1, padx=5, pady=5)
        rowID += 1

        # detectAprilTagsCheckbox = ctk.CTkCheckBox(self.cam_frame, text='Detect April Tags')
        if self.camConfig.detectTags is False:
            self.detectAprilTagsCheckbox.deselect()
        else:
            self.detectAprilTagsCheckbox.select()
            self.createDetector()
        self.detectAprilTagsCheckbox.configure(command=self.toggleDetectTags)
        self.detectAprilTagsCheckbox.grid(row=rowID, column=0, padx=5, pady=5, sticky='ew')

        # self.undistortCheckbox = ctk.CTkCheckBox(self.cam_frame, text='Undistort')
        if not self.calibration.validCal:
            self.undistortCheckbox.configure(state='disabled')

        if self.camConfig.undistort is False:
            self.undistortCheckbox.deselect()
        else:
            self.undistortCheckbox.select()

        self.undistortCheckbox.configure(command=self.toggleUndistort)
        self.undistortCheckbox.grid(row=rowID, column=1, columnspan=2, padx=5, pady=5, sticky='ew')
        rowID += 1

        projectLidarPoints = ctk.CTkCheckBox(self.cam_frame, text='Project Lidar Points into Image')
        if self.camConfig.projectLidarPoints is False:
            projectLidarPoints.deselect()
        else:
            projectLidarPoints.select()
        projectLidarPoints.configure(command=self.toggleLidarPoints)
        projectLidarPoints.grid(row=rowID, column=0, columnspan=1, padx=5, pady=5, sticky='ew')

        # yoloInferenceCheckbox = ctk.CTkCheckBox(self.cam_frame, text='Run YOLO on image')
        if self.camConfig.yoloInference is False:
            self.yoloInferenceCheckbox.deselect()
        else:
            self.yoloInferenceCheckbox.select()
        self.yoloInferenceCheckbox.grid(row=rowID, column=1, columnspan=1, padx=5, pady=5, sticky='ew')

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
        imageProcessingKernelCombobox = ctk.CTkComboBox(self.cam_frame, values=list(ImageKernels.__members__.keys()))
        imageProcessingKernelCombobox.set(self.camConfig.processingKernel.name)
        imageProcessingKernelCombobox.configure(command=self.updateImageProcessingKernel)
        imageProcessingKernelCombobox.grid(row=rowID, column=1, columnspan=1, padx=5, pady=5, sticky='ew')
        rowID += 1

        aprilTagSizeEntryButton = ctk.CTkButton(self.cam_frame, text="Enter Size of April Tag (m)",
                                                command=self.setAprilTagSize)
        aprilTagSizeEntryButton.grid(row=rowID, column=0, padx=5, pady=5)

        self.aprilTagSizeEntry = ctk.CTkEntry(self.cam_frame, placeholder_text=str(self.camConfig.aprilTagSize))
        self.aprilTagSizeEntry.grid(row=rowID, column=1, padx=5, pady=5)
        rowID += 1

        self.recordOff()
        self.recordButton.grid(row=rowID, column=0, columnspan=2, padx=5, pady=5, sticky='ew')
        rowID += 1

        activeEntryButton = ctk.CTkButton(self.cam_frame, text="Enter Time Between Saved Frames",
                                          command=self.getEntryValue)
        activeEntryButton.grid(row=rowID, column=0, padx=5, pady=5)

        self.timeBetweenImgsEntry = ctk.CTkEntry(self.cam_frame,
                                                 placeholder_text=str(self.camConfig.secondsBetweenImages))
        self.timeBetweenImgsEntry.grid(row=rowID, column=1, padx=5, pady=5)
        rowID += 1

        goBackButton = ctk.CTkButton(self.cam_frame, text="Return to Main", command=self.releaseCamReturnToMain)
        goBackButton.grid(row=rowID, column=0, columnspan=2, padx=5, pady=5, sticky='ew')

        self.cam_frame.pack()

    def shutdown(self):
        self.shutting_down = True
        self.recordOff()
        self.startStreamOffBool()

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

    def startStreamOn(self):
        self.showWindow = True
        self.singleImageTextButton.configure(command=self.startStreamOffBool, text='Stop Displaying', fg_color=GREEN,
                                             hover_color='navy')
        self.startStreamButton.configure(command=self.startStreamOffBool, text='Stop Streaming', fg_color=GREEN,
                                         hover_color='navy')
        self.multiImageTextButton.configure(command=self.startStreamOffBool, fg_color=GREEN, hover_color='navy')

        self.selectCameraCombo.configure(state='disabled')

        self.streamOrImgCombo.configure(state='disabled')

        self.t1 = thread_with_exception("CameraGui Thread", self.run)
        self.t1.start()

    def startStreamOffBool(self):
        self.showWindow = False

    def startStreamOff(self):
        cv2.destroyAllWindows()

        if self.vc is not None:
            self.vc.release()

        if not self.shutting_down:
            self.singleImageTextButton.configure(command=self.startStreamOn, fg_color='red', hover_color='blue',
                                                 text=os.path.basename(self.camConfig.imageFilepath))
            self.startStreamButton.configure(command=self.startStreamOn, fg_color='red', hover_color='blue')
            self.multiImageTextButton.configure(command=self.startStreamOn, fg_color='red', hover_color='blue')

            self.selectCameraCombo.configure(state='normal')
            self.startStreamButton.configure(text='Start Stream')
            self.streamOrImgCombo.configure(state='normal')

            self.showWindow = False

        if self.t1 is not None:
            self.t1.raise_exception()
            self.t1.join()

    def recordOn(self):
        self.recordButton.configure(fg_color='green', text='Saving Imagery', hover_color='navy', command=self.recordOff)
        self.recording = True

    def recordOff(self):
        self.recordButton.configure(fg_color='red', text=f'Saved Imagery: #{self.img_idx}', hover_color='blue',
                                    command=self.recordOn)
        self.recording = False

    def toggleLidarPoints(self):
        self.camConfig.projectLidarPoints = not self.camConfig.projectLidarPoints
        self.saveToCache()

    def toggleYoloInference(self):
        self.camConfig.yoloInference = not self.camConfig.yoloInference
        if self.camConfig.yoloInference:
            self.yoloInferenceCheckbox.select()
        else:
            self.yoloInferenceCheckbox.deselect()
        self.saveToCache()

    def toggleDetectCorners(self):
        self.camConfig.detect_corners = not self.camConfig.detect_corners
        self.saveToCache()

    def toggleDetectHorizon(self):
        self.camConfig.detect_horizon = not self.camConfig.detect_horizon
        if self.camConfig.detect_horizon:
            self.detectHorizonCheckbox.select()
        else:
            self.detectHorizonCheckbox.deselect()
        self.saveToCache()

    def toggleFactorgraph(self):
        self.camConfig.factor_graph = not self.camConfig.factor_graph
        if self.camConfig.factor_graph:
            self.factorgraphCheckbox.select()
        else:
            self.factorgraphCheckbox.deselect()
        self.saveToCache()

    def toggleHyperFocus(self):
        self.camConfig.hyper_focus = not self.camConfig.hyper_focus

        if self.camConfig.hyper_focus:
            self.hyperfocusCheckbox.select()
        else:
            self.hyperfocusCheckbox.deselect()

        self.saveToCache()

    def togglePhaseCorrelation(self):
        self.camConfig.phase_correlation = not self.camConfig.phase_correlation
        if self.camConfig.phase_correlation:
            self.phaseCorrelationCheckbox.select()
        else:
            self.phaseCorrelationCheckbox.deselect()

        self.saveToCache()

    def toggleCrosshairs(self):
        self.camConfig.crosshairs = not self.camConfig.crosshairs
        if self.camConfig.crosshairs:
            self.crosshairsCheckbox.select()
        else:
            self.crosshairsCheckbox.deselect()

        self.saveToCache()

    def toggleCubemap(self):
        self.camConfig.cubemap = not self.camConfig.cubemap
        if self.camConfig.cubemap:
            self.cubemapCheckbox.select()
        else:
            self.cubemapCheckbox.deselect()

        self.saveToCache()

    def toggleHud(self):
        self.camConfig.hud = not self.camConfig.hud
        if self.camConfig.hud:
            self.hudCheckbox.select()
        else:
            self.hudCheckbox.deselect()

        self.saveToCache()

    def updateImageProcessingKernel(self, newValue):
        self.camConfig.processingKernel = ImageKernels(newValue)
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
        frame = cv2.imread(self.camConfig.imageFilepath)
        while cv2.getWindowProperty(self.windowName, cv2.WND_PROP_VISIBLE) and self.showWindow:
            self.analyze_image(frame)

            key = cv2.waitKey(1)
            if key == 27:
                cv2.destroyAllWindows()
                break
        self.startStreamOff()

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

        cv2.destroyAllWindows()
        cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)
        rval, self.curr_frame = self.vc.read()
        if rval:
            cv2.resizeWindow(self.windowName, self.curr_frame.shape[1], self.curr_frame.shape[0])
            self.lastHeight = self.curr_frame.shape[0]
            self.lastWidth = self.curr_frame.shape[1]

        while rval and cv2.getWindowProperty(self.windowName, cv2.WND_PROP_VISIBLE) > 0 and self.showWindow:
            rval, frame = self.vc.read()
            self.analyze_image(frame)
            key = cv2.waitKey(1)
            if key == 27:  # exit on ESC
                break
        self.startStreamOff()

    def run_folder_reader(self):
        cv2.destroyAllWindows()
        cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)

        directory = os.path.dirname(self.camConfig.imageFilepath)

        if not self.ImageTimeReader.loadLog(glob.glob(os.path.join(directory, '*.log'))):
            self.ImageTimeReader.idsTimes = []
            imageList = glob.glob(os.path.join(directory, '*.bmp')) + glob.glob(os.path.join(directory, '*.png'))
            imageList = natural_sort(imageList)
            for image in imageList:
                self.ImageTimeReader.idsTimes.append([image, None])

        img_slider = ImageSliderBar(self.ImageTimeReader.numImages)

        img_slider.play_speed = 1
        pause = False
        temp_unpause = False

        while cv2.getWindowProperty(self.windowName, cv2.WND_PROP_VISIBLE) and self.showWindow:

            if not pause or temp_unpause:
                temp_unpause = False
                img_id = img_slider.next_id()

                frame = cv2.imread(os.path.join(directory, self.ImageTimeReader.idsTimes[img_id][0]))
                if frame is not None:
                    self.analyze_image(frame, self.ImageTimeReader.idsTimes[img_id][1],
                                       self.ImageTimeReader.idsTimes[img_id][0])

            key = cv2.waitKey(1)

            if key == 99:  # c
                img_slider.curr_img_idx += 1
                img_slider.play_speed = 0
                temp_unpause = True
                pause = True

            if key == 122:  # z
                img_slider.curr_img_idx -= 1
                img_slider.play_speed = 0
                temp_unpause = True
                pause = True

            if key == 32:  # space
                pause = not pause
                img_slider.play_speed = 0
                if not pause:
                    img_slider.play_speed = 1

            if key == 100:  # d
                img_slider.play_speed += 1
                pause = False

            if key == 97:  # a
                img_slider.play_speed -= 1
                pause = False

            if key == 119:  # w
                self.toggleUndistort()
                self.toggleYoloInference()
                self.toggleDetectHorizon()
                self.toggleHyperFocus()
                self.toggleFactorgraph()

            if key == 27:
                break

        if not self.shutting_down:
            img_slider.close()
        self.startStreamOff()

    def analyze_image(self, frame, img_time=None, name=None):

        self.curr_frame_gray = None

        if self.calibration.validCal and self.camConfig.undistort:
            self.undistort(frame)
        else:
            self.curr_frame = frame.copy()
        self.markup_frame = self.curr_frame.copy()

        if self.camConfig.processingKernel != ImageKernels.Unchanged:
            self.applyKernel()

        if self.camConfig.detect_corners:
            self.corner_detection()

        if self.detector is not None:
            self.detectAprilTags()

        if self.camConfig.projectLidarPoints and self.detector is not None:
            self.projectLidarPoints()

        if self.camConfig.detect_horizon:
            self.detectHorizon()

        if self.camConfig.hyper_focus:
            self.hyper_focus(img_time)

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
            self.draw_hud(img_time)

        if self.camConfig.imageSource == ImageSource.Stream_from_Folder:
            cv2.putText(self.markup_frame, os.path.basename(name), (600, 800),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)

        self.cleanup()

    def draw_hud(self, img_time):
        if self.bank_indicator_points is None:
            self.bank_indicator_points = []
            last_angle = -60
            for new_angle in np.linspace(-50, 60, 12):
                max_rad = 0.10
                if last_angle % 30.0 == 0.0:
                    max_rad = 0.11
                normal_ang = 0.08
                self.bank_indicator_points.append((0.5 + max_rad * sin(deg2rad(last_angle)),
                                                   0.8 + max_rad * cos(deg2rad(last_angle))))
                self.bank_indicator_points.append((0.5 + normal_ang * sin(deg2rad(last_angle)),
                                                   0.8 + normal_ang * cos(deg2rad(last_angle))))
                self.bank_indicator_points.append((0.5 + normal_ang * sin(deg2rad(new_angle)),
                                                   0.8 + normal_ang * cos(deg2rad(new_angle))))
                last_angle = new_angle

            self.bank_indicator_points.append((0.5 + 0.11 * sin(deg2rad(60)),
                                               0.8 + 0.11 * cos(deg2rad(60))))

        x, y, _ = self.markup_frame.shape
        lines = (np.array([x, y]) * np.array(self.bank_indicator_points)).astype(int)
        cv2.polylines(self.markup_frame, [lines],
                      False, (0, 255, 0), 2)

        bank_angle, cmd_bank_angle, mode = self.rollReader.get_roll_at(img_time)

        # bank_angle = 0.0 + 60.0 * sin(img_time)
        bank_pts = []
        bank_pts.append((0.5 + 0.079 * sin(deg2rad(cmd_bank_angle)),
                             0.8 + 0.079 * cos(deg2rad(cmd_bank_angle))))
        bank_pts.append((0.5 + 0.050 * sin(deg2rad(cmd_bank_angle + 15.0)),
                             0.8 + 0.050 * cos(deg2rad(cmd_bank_angle + 15.0))))
        bank_pts.append((0.5 + 0.050 * sin(deg2rad(cmd_bank_angle - 15.0)),
                             0.8 + 0.050 * cos(deg2rad(cmd_bank_angle - 15.0))))

        # cmd_bank_angle = 0.0 + 60.0 * cos(img_time)
        cmd_bank_pts = []
        cmd_bank_pts.append(
            (0.5 + 0.079 * sin(deg2rad(bank_angle)),
             0.8 + 0.079 * cos(deg2rad(bank_angle))))
        cmd_bank_pts.append((0.5 + 0.065 * sin(deg2rad(bank_angle + 10.0)),
                         0.8 + 0.065 * cos(deg2rad(bank_angle + 10.0))))
        cmd_bank_pts.append((0.5 + 0.065 * sin(deg2rad(bank_angle - 10.0)),
                         0.8 + 0.065 * cos(deg2rad(bank_angle - 10.0))))

        lines = (np.array([x, y]) * np.array(bank_pts)).astype(int)
        cmd_lines = (np.array([x, y]) * np.array(cmd_bank_pts)).astype(int)
        cv2.polylines(self.markup_frame, [lines], True, (0, 255, 0))
        cv2.fillPoly(self.markup_frame, [cmd_lines], (0, 255, 0))
        if mode:
            tl = np.array([.40 * x, .80 * y]).astype(int)
            br = np.array([.60 * x, .95 * y]).astype(int)
            cv2.rectangle(self.markup_frame, tl, br, (0, 255, 0))

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

                layout = {'front': (0, 0)}
                # self.curr_frame = self.stitch_cubemap_faces(layout, cells=1)
                self.curr_frame = self.cubemap_faces['front']
        else:
            self.curr_frame = cv2.undistort(src=frame,
                                            cameraMatrix=self.calibration.getCameraMatrix(),
                                            distCoeffs=self.calibration.getDistortion())

    def applyKernel(self):
        if self.camConfig.processingKernel != ImageKernels.Gabor and self.GaborFilter is not None:
            self.GaborFilter.close()
            self.GaborFilter = None

        match self.camConfig.processingKernel:
            case ImageKernels.Sharpen:
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            case ImageKernels.GaussBlur:
                kernel = np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4],
                                   [1, 4, 6, 4, 1]]) / 256.0
            case ImageKernels.EdgeDetect:
                kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            case ImageKernels.HorizontalEdgeDetect:
                kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
            case ImageKernels.VerticalEdgeDetect:
                kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            case ImageKernels.BoxBlur:
                kernel = np.ones((5, 5)) / 25.0
            case ImageKernels.SobelEdgeDetectHorizontal:
                kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            case ImageKernels.SobelEdgeDetectVertical:
                kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            case ImageKernels.LaplaceEdgeDetect:
                kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
            case ImageKernels.Gabor:
                if self.GaborFilter is None:
                    self.GaborFilter = Gabor()
                kernel = self.GaborFilter.filter_kernel()
            case ImageKernels.ScharrEdgeDetectHorizontal:
                kernel = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]])
            case ImageKernels.ScharrEdgeDetectVertical:
                kernel = np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]])
            case ImageKernels.Unsharp:
                gaussian_3 = cv2.GaussianBlur(self.markup_frame, (0, 0), 2.0)
                self.markup_frame = cv2.addWeighted(self.markup_frame, 2.0, gaussian_3, -1.0, 0)
                return
            case _:
                return

        self.markup_frame = cv2.filter2D(self.markup_frame, -1, kernel)

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

        for corners, id in zip(corners, ids):
            corners = np.squeeze(np.array(corners))
            polyline = [np.array(corners, np.int32).reshape((-1, 1, 2))]
            pixCenter = np.mean(corners, axis=0).astype(np.int32)
            cv2.polylines(self.markup_frame, polyline, True, (0, 255, 0), 4, lineType=cv2.FILLED)
            cv2.putText(self.markup_frame, str(id[0]), pixCenter,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
            cv2.putText(self.markup_frame, str(id[0]), pixCenter,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

            self.detectIDS.append(id)

            if self.centers is None:
                self.centers = np.array(pixCenter).astype('float32')
            else:
                self.centers = np.vstack((self.centers, np.array(pixCenter).astype('float32')))

    def projectLidarPoints(self):

        ret = False
        if self.centers is not None and len(self.centers) >= 6:
            truthPoints = copy.copy(self.lidarTruthPoints.truthPoints)
            points = []
            distParams = np.zeros((5,))  # use image undistort instead
            for detectID in self.detectIDS:
                points.append(truthPoints[str(detectID[0])])
            points = np.array(points)

            ret, rvec, tvec = cv2.solvePnP(objectPoints=points,
                                           imagePoints=self.centers,
                                           cameraMatrix=self.calibration.getCameraMatrix(),
                                           distCoeffs=distParams,
                                           flags=cv2.SOLVEPNP_ITERATIVE)
            probeTip_3d = np.array([[0.0], [0.0], [0.0]])
            self.projectProbe, _ = cv2.projectPoints(probeTip_3d, rvec=rvec, tvec=tvec,
                                                     cameraMatrix=self.calibration.getCameraMatrix(),
                                                     distCoeffs=distParams)

            if ret:
                projectedPoints_orig, _ = cv2.projectPoints(self.lidarTruthPoints.getTruthPointsNumpy(),
                                                            rvec=rvec,
                                                            tvec=tvec,
                                                            cameraMatrix=self.calibration.getCameraMatrix(),
                                                            distCoeffs=distParams)

                self.plotOnImg(projectedPoints_orig[:, 0, :].astype(int),
                               list(self.lidarTruthPoints.getTruthPointsDict().keys()), (255, 255, 0))

        if self.projectProbe is not None and self.camConfig.projectLidarPoints:
            cv2.circle(self.markup_frame, self.projectProbe[0, 0, :].astype(int), 6, (255, 0, 0), 6)
            cv2.putText(self.markup_frame, "Probe Tip", self.projectProbe[0, 0, :].astype(int) - [50, 50],
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 6)

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

    def hyper_focus(self, time=None):

        if not self.camConfig.factor_graph:
            if self.last_bounding_box_size is not None:
                self.radius = (self.last_bounding_box_size[0] + self.last_bounding_box_size[
                    1] + self.radius * 4.0) / 5.0
            else:
                self.confSlider(self.yoloSession.conf)

            self.markup_frame = dim_except_circle(self.markup_frame, self.current_center_est, 3.0 * self.radius, 0.00)
            self.markup_frame = dim_except_circle(self.markup_frame, self.current_center_est, 1.5 * self.radius, 0.50)

            self.radius = min(800, self.radius + 12)
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
        self.markup_frame, output = self.yoloSession.inferOnImage(self.markup_frame, self.markup_frame)
        centers, boxes, scores, class_ids, time = output
        if len(centers) > 0:  #and self.yoloSession.reader.numClasses == 1:
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
            d = self.calibration.getDistortion()
            twoD_points = np.array([self.last_yolo_center[0], self.last_yolo_center[1], 1.0])
            dist_est = 2.0 / (
                    self.last_bounding_box_size[0] / self.curr_frame.shape[0] + self.last_bounding_box_size[1] /
                    self.curr_frame.shape[1])
            dist_est = 2.0 / (self.last_bounding_box_size[0] + self.last_bounding_box_size[1])

            if self.check_above_horizon(self.last_yolo_center):
                self.last_yolo_3d_estimate = np.linalg.inv(K).dot(twoD_points) * dist_est
                cv2.circle(self.markup_frame, (int(self.last_yolo_center[0]), int(self.last_yolo_center[1])),
                           3, (255, 0, 255), 3)
                self.current_center_est = ((self.current_center_est[0] * 2.0 + centers[best_idx][0]) / 3.0,
                                           (self.current_center_est[1] * 2.0 + centers[best_idx][1]) / 3.0)
                return

        self.last_bounding_box_size = None
        self.last_yolo_center = None

    def factor_graph(self, time):
        color = (0, 255, 255)
        if self.last_yolo_3d_estimate is not None:
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

            size = 15
            thickness = 1
            cv2.circle(self.markup_frame, (int(self.curr_FG_pixel[0]), int(self.curr_FG_pixel[1])), size, color,
                       thickness)
            cv2.line(self.markup_frame, [int(self.curr_FG_pixel[0]) + size, int(self.curr_FG_pixel[1])],
                     [int(self.curr_FG_pixel[0]) - size, int(self.curr_FG_pixel[1])], color, thickness)
            cv2.line(self.markup_frame, [int(self.curr_FG_pixel[0]), int(self.curr_FG_pixel[1]) + size],
                     [int(self.curr_FG_pixel[0]), int(self.curr_FG_pixel[1]) - size], color, thickness)

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

            cv2.polylines(self.markup_frame, [crosshairsH], True, (0, 255, 0), thickness)
            cv2.polylines(self.markup_frame, [crosshairsV], True, (0, 255, 0), thickness)

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
            cv2.putText(self.markup_frame, str(names[idx]), textLoc, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 12,
                        cv2.LINE_AA)
            cv2.putText(self.markup_frame, str(names[idx]), textLoc, cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3, cv2.LINE_AA)

    def potentialResize(self):
        x, y, width, height = cv2.getWindowImageRect(self.windowName)
        aspectRatio = self.curr_frame.shape[1] / self.curr_frame.shape[0]
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
