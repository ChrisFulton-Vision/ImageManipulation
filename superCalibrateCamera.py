import ctypes

import numpy as np
import customtkinter as ctk
from tkinter import filedialog
from threading import Thread
from cv2_enumerate_cameras import enumerate_cameras
from Calibration import Calibration
from LidarTruth import TruthPoints
import pickle, copy, os, time, threading, cv2, glob, re, yolo
from enum import Enum, auto
from os.path import join
from PIL import Image, ImageTk

# import superCalibrate as superCal
#pip install cv2_enumerate_cameras
#or
#pip install git+https://github.com/chinaheyu/cv2_enumerate_cameras.git

GREEN = '#2FA572'
class video_player():
    def __init__(self, img_filepaths:list):
        self.img_id = 0
        self.play_speed = 1
        self.pause = False
        self.temp_unpause = False
        self.quit_now = False
        self.thread = thread_with_exception(1, self.run_new_menu)
        self.pop_up = None
        self.play_pause_button = None
        self.exit_player_button = None
        self.imageList = natural_sort(img_filepaths)
        self.thread.start()

    def run_new_menu(self):
        if self.pop_up is None:
            self.pop_up = ctk.CTkToplevel()
            self.pop_up.title('Video Controls')
            self.pop_up.geometry('600x250+300+300')
            self.pop_up.grid_columnconfigure(0, weight=1, uniform='equal')
            self.pop_up.grid_columnconfigure(1, weight=1, uniform='equal')
            self.pop_up.grid_columnconfigure(2, weight=1, uniform='equal')
        if self.play_pause_button is None:
            self.play_pause_button = ctk.CTkButton(master=self.pop_up, text="Pause", command=self.play_pause)
            self.play_pause_button.grid(row=0, column=1, sticky='nsew')
        if self.exit_player_button is None:
            self.exit_player_button = ctk.CTkButton(master=self.pop_up, text='Quit', command=self.exit_player)
            self.exit_player_button.grid(row=1, column=1, sticky='nsew')

    def play_pause(self):
        self.pause = not self.pause

    def exit_player(self):
        self.quit_now = True
        self.close()

    def close(self):
        self.pop_up.destroy()
        self.thread.raise_exception()
        self.thread.join()
        del self

    def next_frame(self)->np.array:

        key = cv2.waitKey(1)

        if key == 99:
            self.img_id += 1
            self.play_speed = 0
            self.temp_unpause = True
            self.pause = True
        if key == 122:
            self.img_id -= 1
            self.play_speed = 0
            self.temp_unpause = True
            self.pause = True

        if key == 32:
            self.play_speed = 0
            if not self.pause:
                self.pause = not self.pause
                self.play_speed = 1
        if key == 100:
            self.play_speed += 1
            self.pause = False
        if key == 97:
            self.play_speed -= 1
            self.pause = False

        if key == 27:
            self.quit_now = True

        frame = None

        if not self.pause or self.temp_unpause:
            self.temp_unpause = False
            self.img_id = (self.img_id + self.play_speed) % len(self.imageList)
            frame = cv2.imread(self.imageList[self.img_id])

        return frame, self.quit_now

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
        if hasattr(self,'_thread_id'):
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

class ImageSource(Enum):
    Camera_Stream = 'Camera Stream'
    Static_Image = 'Static Image'
    Stream_from_Folder = 'Stream from Folder'

class ImageKernels(Enum):
    Unchanged = 'Unchanged'  #None
    Sharpen = 'Sharpen'    #np.array([[0, -1, 0],[-1, 5, -1], [0, -1, 0]])
    GaussBlur = 'GaussBlur'  #np.array([[1, 4, 6, 4, 1],[4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]]) / 256.0
    EdgeDetect = 'EdgeDetect' #np.array([[-1, -1, -1],[-1, 8, -1], [-1, -1, -1]])

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
        self.imageFilepath = None
        self.lidarFilepath = None
        self.yoloFilepath = ''
        self.detect_corners = False
        self.processingKernel = ImageKernels.Unchanged

    def copy(self, configToCopy):
        self.__dict__.update(copy.deepcopy(configToCopy.__dict__))


class Camera():
    def __init__(self, gui):
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
        self.windowName =  'webcam'
        self.filepath = ''
        self.cam_frame = ctk.CTkFrame(master=self.gui)
        self.showWindow = False

        self.available_sources = [source.value for source in ImageSource]

        self.streamOrImgCombo = ctk.CTkComboBox(self.cam_frame, values=self.available_sources, command=self.sourceUpdate)
        self.startStreamButton = ctk.CTkButton(master=self.cam_frame, text='Start Stream', fg_color='red', hover_color='blue')
        self.recordButton = ctk.CTkButton(master=self.cam_frame,text='Saving Imagery', fg_color='green', hover_color='navy', command=self.recordOff)
        self.selectCameraCombo = ctk.CTkComboBox(self.cam_frame, values=list(self.indexDict.keys()),
                                                         command=self.selectCamera)
        self.selectFolderLabel = ctk.CTkLabel(self.cam_frame, text="../" + os.path.basename(os.path.normpath(self.filepath)))
        self.selectTruthPointsButton = ctk.CTkButton(master=self.cam_frame, text='Select LIDAR Points', hover_color='blue', command=self.selectLidarFile)

        if self.camConfig.lidarFilepath is not None:
            self.selectTruthPointsLabel = ctk.CTkLabel(self.cam_frame, text="../" + os.path.basename(os.path.normpath(self.camConfig.lidarFilepath)))
        else:
            self.selectTruthPointsLabel = ctk.CTkLabel(self.cam_frame, text='No Truth Loaded')

        self.lidarTruthPoints = TruthPoints()
        self.selectYOLO_folderButton = ctk.CTkButton(self.cam_frame, text='Select YOLO Folder', fg_color=GREEN,command=self.selectYoloFolder)
        self.selectYOLO_folderLabel = ctk.CTkLabel(self.cam_frame,
                                                   text='../' + os.path.basename(os.path.normpath(self.camConfig.yoloFilepath)))
        self.selectCalibLabel = None
        self.undistortCheckbox = None

        self.singleImageFolderSelect = ctk.CTkButton(self.cam_frame, text='Select Img', command=self.selectSingleImage)
        self.singleImageTextButton = ctk.CTkButton(self.cam_frame, text='No Image Selected')
        self.multiImageFolderSelect = ctk.CTkButton(self.cam_frame, text='Select Img Folder', command=self.selectSingleImage)
        self.multiImageTextButton = ctk.CTkButton(self.cam_frame, text='No Folder Selected', command=self.startStreamOn)
        self.confSliderLabel = ctk.CTkLabel(self.cam_frame, text='Conf: 0.75')
        self.confSliderBar = ctk.CTkSlider(self.cam_frame, command=self.confSlider, from_=0.15)
        self.confSliderBar.set(0.75)
        self.iouSliderLabel = ctk.CTkLabel(self.cam_frame, text='IOU: 1.00')
        self.iouSliderBar = ctk.CTkSlider(self.cam_frame, command=self.iouSlider)
        self.iouSliderBar.set(1.00)

        self.loadFromCache()
        self.vc = None
        # self.vc.setExceptionMode(True)
        # self.detector = Detector(refine_edges=1, decode_sharpening=0.0)

        self.detector = None
        self.arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
        self.arucoParams = cv2.aruco.DetectorParameters()

        self.img_idx = 0
        self.timeBetweenImgsEntry = None
        self.lastImageTime = 0
        self.camFrameGeometry = '455x570'
        self.cam_frame.grid_rowconfigure(list(range(3)), weight=1)  # configure grid system
        self.cam_frame.grid_columnconfigure(list(range(3)), weight=1)
        self.t1 = None
        self.aspectRatio = 1.0
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

        self.updateLidarLabel()
        self.updateYOLOLabel()
        self.yoloSession.setNewFolder(self.camConfig.yoloFilepath)
        self.loadTruthPoints()

    def saveToCache(self):
        with open('camConfig_cache.pkl', 'wb') as f:
            pickle.dump(self.camConfig, f)
            pickle.dump(self.filepath, f)
            pickle.dump(self.calibFile, f)

    def selectFolder(self):
        self.filepath = filedialog.askdirectory(initialdir=self.filepath + "/..", mustexist=True, title="Select Imagery Folder")
        self.saveToCache()
        self.loadFromCache()

    def loadCalibration(self):
        self.calibFile = filedialog.askopenfilename(initialdir=self.filepath+'/..', title='Select Folder of Calibration')
        self.ingestCalibration()

    def selectLidarFile(self):
        if self.camConfig.lidarFilepath is None:
            self.camConfig.lidarFilepath = filedialog.askopenfilename(initialdir=self.filepath + '/..',
                                                    title='Select LIDAR Truth Points')
        else:
            self.camConfig.lidarFilepath = filedialog.askopenfilename(initialdir=self.camConfig.lidarFilepath + '/..',
                                                                      title='Select LIDAR Truth Points')
        self.updateLidarLabel()
        self.loadTruthPoints()
        self.saveToCache()

    def selectYoloFolder(self):
        if self.camConfig.yoloFilepath is None:
            self.camConfig.yoloFilepath = filedialog.askdirectory(initialdir=os.getcwd() + '/..',
                                                                  title='Select YOLO Folder')
        else:
            self.camConfig.yoloFilepath = filedialog.askdirectory(initialdir=self.camConfig.yoloFilepath + '/..',
                                                                  title='Select YOLO Folder')
        self.updateYOLOLabel()
        self.yoloSession.setNewFolder(self.camConfig.yoloFilepath)
        self.saveToCache()

    def updateLidarLabel(self):
        if self.camConfig.lidarFilepath is not None:
            self.selectTruthPointsLabel.configure(text=os.path.basename(self.camConfig.lidarFilepath))

    def updateYOLOLabel(self):
        if self.camConfig.yoloFilepath is not None:
            self.selectYOLO_folderLabel.configure(text=os.path.basename(self.camConfig.yoloFilepath))

    def loadTruthPoints(self):
        if self.camConfig.lidarFilepath is not None:
            with open(self.camConfig.lidarFilepath, 'rb') as f:
                test = pickle.load(f)
                self.lidarTruthPoints.copy(test)

    def confSlider(self, confValue):
        self.yoloSession.conf = confValue
        self.confSliderLabel.configure(text='Conf: ' + f'{confValue:.2f}')

    def iouSlider(self, iouValue):
        self.yoloSession.iou = iouValue
        self.iouSliderLabel.configure(text='IOU: ' + f'{iouValue:.2f}')

    def ingestCalibration(self):

        if not self.calibration.fromBinFile(self.calibFile):
            if not self.calibration.fromFile(self.calibFile):
                if self.selectCalibLabel is not None:
                    self.selectCalibLabel.configure(text='No Calibration Found')
                    self.gui.after(100,self.gui.update())
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

        if self.undistortCheckbox is not None:
            self.undistortCheckbox.configure(state='normal')
        self.saveToCache()

    def scanForCameras(self):
        self.indexDict = {}
        for camera_info in enumerate_cameras(cv2.CAP_DSHOW):
            self.indexDict[camera_info.name] = camera_info.index

    def selectCamera(self, key):
        self.cam_index = self.indexDict[key]
        self.vc.release()
        self.vc = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)


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

    def selectSingleImage(self):
        if self.camConfig.imageFilepath is None:
            initDir = self.filepath + '/..'
        else:
            initDir = os.path.normpath(self.camConfig.imageFilepath)

        self.camConfig.imageFilepath = filedialog.askopenfilename(initialdir=initDir, title="Select Image")
        self.singleImageTextButton.configure(text=os.path.basename(self.camConfig.imageFilepath))
        self.multiImageTextButton.configure(text=os.path.basename(os.path.dirname(self.camConfig.imageFilepath)))

    def setupFrame(self):
        rowID = 0

        self.vc = cv2.VideoCapture(self.camConfig.cam_index, cv2.CAP_DSHOW)
        self.vc.set(cv2.CAP_PROP_FPS, 60)

        self.streamOrImgCombo = ctk.CTkComboBox(self.cam_frame, values=['Camera Stream', 'Static Image', 'Stream from Folder'], command=self.sourceUpdate)
        self.streamOrImgCombo.grid(row=rowID, column=0, padx=5, pady=5)
        rowID += 1

        self.startStreamOff()

        self.singleImageTextButton.configure(command=self.detectSingleImage)
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
        rowID +=1


        selectCalibButton = ctk.CTkButton(self.cam_frame, text='Select Calibration', command=self.loadCalibration)
        selectCalibButton.grid(row=rowID, column=0, padx=5, pady=5)

        self.selectCalibLabel = ctk.CTkLabel(self.cam_frame, text="../" + os.path.basename(os.path.normpath(self.calibFile)))
        self.selectCalibLabel.grid(row=rowID, column=1, padx=5, pady=5)
        rowID +=1

        self.selectTruthPointsButton.grid(row=rowID, column=0, padx=5, pady=5)
        self.selectTruthPointsLabel.grid(row=rowID, column=1, padx=5, pady=5)
        rowID += 1

        self.selectYOLO_folderButton.grid(row=rowID, column=0, padx=5, pady=5)
        self.selectYOLO_folderLabel.grid(row=rowID, column=1, padx=5, pady=5)
        rowID += 1

        self.confSliderLabel.grid(row=rowID, column=0, padx=5, pady=5)
        self.confSliderBar.grid(row=rowID, column=1, padx=5, pady=5)
        rowID += 1

        self.iouSliderLabel.grid(row=rowID, column=0, padx=5, pady=5)
        self.iouSliderBar.grid(row=rowID, column=1, padx=5, pady=5)
        rowID += 1

        detectAprilTagsCheckbox = ctk.CTkCheckBox(self.cam_frame, text='Detect April Tags')
        if self.camConfig.detectTags is False:
            detectAprilTagsCheckbox.deselect()
        else:
            detectAprilTagsCheckbox.select()
            self.createDetector()
        detectAprilTagsCheckbox.configure(command=self.toggleDetectTags)
        detectAprilTagsCheckbox.grid(row=rowID, column=0, padx=5, pady=5, sticky='ew')

        self.undistortCheckbox = ctk.CTkCheckBox(self.cam_frame, text='Undistort')
        if not self.calibration.validCal:
            self.undistortCheckbox.configure(state='disabled')

        if self.camConfig.undistort is False:
            self.undistortCheckbox.deselect()
        else:
            self.undistortCheckbox.select()

        self.undistortCheckbox.configure(command=self.toggleUndistort)
        self.undistortCheckbox.grid(row=rowID, column=1,columnspan=2, padx=5, pady=5, sticky='ew')
        rowID += 1

        projectLidarPoints = ctk.CTkCheckBox(self.cam_frame, text='Project Lidar Points into Image')
        if self.camConfig.projectLidarPoints is False:
            projectLidarPoints.deselect()
        else:
            projectLidarPoints.select()
        projectLidarPoints.configure(command=self.toggleLidarPoints)
        projectLidarPoints.grid(row=rowID, column=0, columnspan=2, padx=5, pady=5, sticky='ew')

        yoloInference = ctk.CTkCheckBox(self.cam_frame, text='Run YOLO on image')
        if self.camConfig.yoloInference is False:
            yoloInference.deselect()
        else:
            yoloInference.select()
        yoloInference.configure(command=self.toggleYoloInference)
        yoloInference.grid(row=rowID, column=1, columnspan=2, padx=5, pady=5, sticky='ew')

        rowID += 1
        detectCornersCheckbox = ctk.CTkCheckBox(self.cam_frame, text='Detect Corners')
        if self.camConfig.detect_corners is False:
            detectCornersCheckbox.deselect()
        else:
            detectCornersCheckbox.select()
        detectCornersCheckbox.configure(command=self.toggleDetectCorners)
        detectCornersCheckbox.grid(row=rowID, column=0, columnspan=1, padx=5, pady=5, sticky='ew')

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
        self.recordButton.grid(row=rowID, column=0,columnspan=2, padx=5, pady=5, sticky='ew')
        rowID += 1

        activeEntryButton = ctk.CTkButton(self.cam_frame,text="Enter Time Between Saved Frames", command=self.getEntryValue)
        activeEntryButton.grid(row=rowID, column=0, padx=5, pady=5)

        self.timeBetweenImgsEntry = ctk.CTkEntry(self.cam_frame,placeholder_text=str(self.camConfig.secondsBetweenImages))
        self.timeBetweenImgsEntry.grid(row=rowID, column=1, padx=5, pady=5)
        rowID += 1

        goBackButton = ctk.CTkButton(self.cam_frame, text="Return to Main", command=self.releaseCamReturnToMain)
        goBackButton.grid(row=rowID, column=0, columnspan=2, padx=5, pady=5, sticky='ew')

        self.cam_frame.pack()

    def shutdown(self):
        self.recordOff()
        self.startStreamOff()

    def releaseCamReturnToMain(self):
        self.startStreamOff()
        self.vc.release()
        self.gui.returnToMain()

    def setAprilTagSize(self):

        try:
            self.camConfig.aprilTagSize = float(self.aprilTagSizeEntry.get())
        except ValueError:
            self.aprilTagSizeEntry.delete(0,ctk.END)
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
        self.startStreamButton.configure(command=self.startStreamOff, text='Stop Streaming', fg_color=GREEN, hover_color='navy')
        self.multiImageTextButton.configure(command=self.startStreamOff, fg_color=GREEN, hover_color='navy')

        self.selectCameraCombo.configure(state='disabled')

        self.streamOrImgCombo.configure(state='disabled')

        self.t1 = thread_with_exception(0, self.run)
        self.t1.start()


    def startStreamOff(self):
        self.startStreamButton.configure(command=self.startStreamOn, fg_color='red', hover_color='blue')
        self.multiImageTextButton.configure(command=self.startStreamOn, fg_color='red', hover_color='blue')

        self.selectCameraCombo.configure(state='normal')
        self.startStreamButton.configure(text='Start Stream')

        self.streamOrImgCombo.configure(state='normal')
        if self.t1 is not None:
            self.t1.raise_exception()
            self.t1.join()
        cv2.destroyAllWindows()

    def recordOn(self):
        self.recordButton.configure(fg_color='green', text='Saving Imagery', hover_color='navy', command=self.recordOff)
        self.recording = True

    def recordOff(self):
        self.recordButton.configure(fg_color='red',text=f'Saved Imagery: #{self.img_idx}', hover_color='blue', command=self.recordOn)
        self.recording = False


    def toggleLidarPoints(self):
        self.camConfig.projectLidarPoints = not self.camConfig.projectLidarPoints
        self.saveToCache()

    def toggleYoloInference(self):
        self.camConfig.yoloInference = not self.camConfig.yoloInference
        self.saveToCache()

    def toggleDetectCorners(self):
        self.camConfig.detect_corners = not self.camConfig.detect_corners
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

        if self.undistortCheckbox.get():
            self.camConfig.undistort = True
        else:
            self.camConfig.undistort = False
        self.saveToCache()

    def detectSingleImage(self):
        cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)
        frame = cv2.imread(self.camConfig.imageFilepath)
        self.analyze_image(frame)

        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()

    @staticmethod
    def convert_cv_to_pil(img):
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def run(self):

        if self.camConfig.imageSource == ImageSource.Camera_Stream:
            self.run_video_stream()
        else:
            self.run_folder_reader()

    def run_video_stream(self):
        cv2.destroyAllWindows()
        cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)
        rval, frame = self.vc.read()
        if rval:
            self.aspectRatio = float(frame.shape[1]) / float(frame.shape[0])
            cv2.resizeWindow(self.windowName, frame.shape[1], frame.shape[0])
            self.lastHeight = frame.shape[0]
            self.lastWidth = frame.shape[1]

        while rval and cv2.getWindowProperty(self.windowName, cv2.WND_PROP_VISIBLE) and self.showWindow:
            rval, frame = self.vc.read()
            self.analyze_image(frame)

            key = cv2.waitKey(1)
            if key == 27 or cv2.getWindowProperty(self.windowName, cv2.WND_PROP_VISIBLE) < 1:  # exit on ESC
                self.startStreamOff()
                break


    def run_folder_reader(self):
        cv2.destroyAllWindows()
        cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)
        imageList = glob.glob(os.path.join(os.path.dirname(self.camConfig.imageFilepath), '*.bmp'))
        imageList = natural_sort(imageList)

        img_id = 0
        play_speed = 1
        pause = False
        temp_unpause = False

        while cv2.getWindowProperty(self.windowName, cv2.WND_PROP_VISIBLE) and self.showWindow:

            if not pause or temp_unpause:
                temp_unpause = False
                img_id = (img_id + play_speed) % len(imageList)
                frame = cv2.imread(imageList[img_id])
                self.analyze_image(frame)

            key = cv2.waitKey(1)

            if key == 99:
                img_id += 1
                play_speed = 0
                temp_unpause = True
                pause = True

            if key == 122:
                img_id -= 1
                play_speed = 0
                temp_unpause = True
                pause = True

            if key == 32:
                pause = not pause
                play_speed = 0
                if not pause:
                    play_speed = 1
            if key == 100:
                play_speed += 1
                pause = False
            if key == 97:
                play_speed -= 1
                pause = False

            if key == 27:
                break

        self.startStreamOff()

    def analyze_image(self, frame):
        frame = self.undistort(frame)
        frame = self.applyKernel(frame)
        self.detectAprilTags(frame)
        self.projectLidarPoints(frame)
        self.corner_detection(frame)
        frame = self.run_yolo(frame)
        self.run_yolo_and_cleanup(frame)


    def undistort(self, frame):
        if self.calibration.validCal and self.camConfig.undistort:
            return cv2.undistort(frame, cameraMatrix=self.calibration.getCameraMatrix(),
                                  distCoeffs=self.calibration.getDistortion())
        return frame

    def detectAprilTags(self, frame):
        if self.detector is None:
            return

        webGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(webGray)
        self.centers = None
        self.detectIDS = []

        if corners is None or ids is None:
            return

        for corners, id in zip(corners, ids):
            corners = np.squeeze(np.array(corners))
            polyline = [np.array(corners, np.int32).reshape((-1, 1, 2))]
            pixCenter = np.mean(corners, axis=0).astype(np.int32)
            cv2.polylines(frame, polyline, True, (0, 255, 0), 4, lineType=cv2.FILLED)
            cv2.putText(frame, str(id[0]), pixCenter,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
            cv2.putText(frame, str(id[0]), pixCenter,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

            self.detectIDS.append(id)

            if self.centers is None:
                self.centers = np.array(pixCenter).astype('float32')
            else:
                self.centers = np.vstack((self.centers, np.array(pixCenter).astype('float32')))

    def projectLidarPoints(self, frame):
        if not self.camConfig.projectLidarPoints or self.detector is None:
            return

        ret = False
        if self.centers is not None and len(self.centers) >= 6:
            truthPoints = copy.copy(self.lidarTruthPoints.truthPoints)
            points = []
            distParams = np.zeros((5,)) # use image undistort instead
            for detectID in self.detectIDS:
                points.append(truthPoints[str(detectID[0])])
            points = np.array(points)

            ret, rvec, tvec = cv2.solvePnP(objectPoints=points,
                                       imagePoints=self.centers,
                                       cameraMatrix=self.calibration.getCameraMatrix(),
                                       distCoeffs=distParams,
                                       flags=cv2.SOLVEPNP_ITERATIVE)
            probeTip_3d = np.array([[0.0], [0.0], [0.0]])
            self.projectProbe, _ = cv2.projectPoints(probeTip_3d, rvec=rvec, tvec=tvec, cameraMatrix=self.calibration.getCameraMatrix(), distCoeffs=distParams)

        if self.projectProbe is not None and self.camConfig.projectLidarPoints:
            cv2.circle(frame, self.projectProbe[0,0,:].astype(int), 6, (255, 0, 0), 6)
            cv2.putText(frame, "Probe Tip", self.projectProbe[0,0,:].astype(int) - [50, 50],
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 6)

        if ret:
            projectedPoints_orig, _ = cv2.projectPoints(self.lidarTruthPoints.getTruthPointsNumpy(),
                                                        rvec=rvec,
                                                        tvec=tvec,
                                                        cameraMatrix=self.calibration.getCameraMatrix(),
                                                        distCoeffs=distParams)

            self.plotOnImg(frame, projectedPoints_orig[:, 0, :].astype(int),
                           list(self.lidarTruthPoints.getTruthPointsDict().keys()), (255, 255, 0))

        return frame

    def corner_detection(self, frame):
        if self.camConfig.detect_corners:
            harris_corners = cv2.cornerHarris(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 3, 3, 0.05)

            frame[harris_corners > 0.025 * harris_corners.max()] = [255, 127, 127]

    def applyKernel(self, frame):
        if self.camConfig.processingKernel == ImageKernels.Unchanged:
            return frame
        match self.camConfig.processingKernel:
            case ImageKernels.Sharpen:
                kernel = np.array([[0, -1, 0],[-1, 5, -1], [0, -1, 0]])
            case ImageKernels.GaussBlur:
                kernel = np.array([[1, 4, 6, 4, 1],[4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]]) / 256.0
            case ImageKernels.EdgeDetect:
                kernel = np.array([[-1, -1, -1],[-1, 8, -1], [-1, -1, -1]])
            case _:
                return frame

        return cv2.filter2D(frame, -1, kernel)

    def run_yolo(self, frame):
        if self.camConfig.yoloInference:
            frame, output = self.yoloSession.inferOnImage(frame)
        return frame

    def run_yolo_and_cleanup(self, frame):

        if self.calibration.validCal:
            cx = int(self.calibration.cx)
            cy = int(self.calibration.cy)
        else:
            cx = int(frame.shape[0] / 2)
            cy = int(frame.shape[1] / 2)

        width = frame.shape[0]
        height = frame.shape[1]
        thickness = max(int(width/250),1)

        crosshairsH = np.array([[cx + max(int(width/50),10), cy], [cx - max(int(width/50),10), cy]])
        crosshairsV = np.array([[cx, cy + max(int(height/50),10)], [cx, cy - max(int(height/50),10)]])

        cv2.polylines(frame, [crosshairsH], True, (0, 255, 0), thickness)
        cv2.polylines(frame, [crosshairsV], True, (0, 255, 0), thickness)

        self.potentialResize()

        cv2.imshow(self.windowName, cv2.resize(frame, (self.lastWidth, self.lastHeight)))

        if self.recording and time.time() - self.lastImageTime > self.camConfig.secondsBetweenImages:
            cv2.imwrite(self.filepath + '\\' + str(self.img_idx) + '.png', frame)
            self.img_idx += 1
            self.lastImageTime = time.time()
            self.recordButton.configure(text=f'Saving Imagery: #{self.img_idx}')

    def plotOnImg(self, img, points, names, color):
        for idx, pxPt in enumerate(points):
            cv2.circle(img, (int(pxPt[0]), int(pxPt[1])), 5, color, 5)
            textLoc = (int(pxPt[0]) - 30, int(pxPt[1] - 30))
            cv2.putText(img, str(names[idx]), textLoc, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 12,
                        cv2.LINE_AA)
            cv2.putText(img, str(names[idx]), textLoc, cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3, cv2.LINE_AA)

    def potentialResize(self):
        x, y, width, height = cv2.getWindowImageRect(self.windowName)

        if not self.lastHeight == height and height != 0:
            cv2.resizeWindow(self.windowName, int(height * self.aspectRatio), height)
            self.lastHeight = height
            self.lastWidth = int(height * self.aspectRatio)
        elif not self.lastWidth == width and width != 0:
            cv2.resizeWindow(self.windowName, width, int(width / self.aspectRatio))
            self.lastWidth = width
            self.lastHeight = int(width / self.aspectRatio)

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)