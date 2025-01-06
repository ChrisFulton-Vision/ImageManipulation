import ctypes

import cv2
import numpy as np
from pupil_apriltags import Detector
import customtkinter as ctk
from tkinter import filedialog
import time
import threading
from threading import Thread
from cv2_enumerate_cameras import enumerate_cameras
import os
import pickle
import copy
import superCalibrate as superCal
#pip install cv2_enumerate_cameras
#or
#pip install git+https://github.com/chinaheyu/cv2_enumerate_cameras.git

class TruthPoints():
    def __init__(self):
        self.truthPoints = {}

        self.selectTruthPoints()

        self.saveToCache()

    def selectTruthPoints(self):
        # 0
        self.truthPoints['0'] = np.array([9.95743, -1.17852, -1.30208])
        # 1
        self.truthPoints['1'] = np.array([7.63673, -.732, -.85845])
        # 4
        self.truthPoints['4'] = np.array([5.2572, -2.81513, -1.29637])
        # 5
        self.truthPoints['5'] = np.array([5.84086, -3.35247, -.91818])
        # 6
        self.truthPoints['6'] = np.array([5.23601, -1.07118, -.90861])
        # 7
        self.truthPoints['7'] = np.array([8.4014, -.10626, -1.05846])
        # 8
        self.truthPoints['8'] = np.array([9.03127, -.0561, -.50734])
        # 9
        self.truthPoints['9'] = np.array([3.83315, -2.74696, -.90198])
        # 10
        self.truthPoints['10'] = np.array([5.73997, -2.3464, -.8862])
        # 12
        self.truthPoints['12'] = np.array([7.79781, .81792, -.84915])
        # 17
        self.truthPoints['17'] = np.array([5.37848, -3.76637, -.67129])
        # 18
        self.truthPoints['18'] = np.array([9.05368, .56085, -.17019])
        # 19
        self.truthPoints['19'] = np.array([5.62043, -1.7569, -.98166])
        # 21
        self.truthPoints['21'] = np.array([8.61416, -1.87028, -.84081])
        # 22
        self.truthPoints['22'] = np.array([7.11942, -1.35599, -1.23372])
        # 23
        self.truthPoints['23'] = np.array([7.03449, .00338, -.98724])

    def getTruthPointsDict(self):
        return self.truthPoints

    def getTruthPointsNumpy(self):
        truthPointsArray = None

        for truthPoint in self.truthPoints.values():
            if truthPointsArray is None:
                truthPointsArray = truthPoint
            else:
                truthPointsArray = np.vstack((truthPointsArray, truthPoint))

        return truthPointsArray

    def saveToCache(self):
        with open('LIDAR_Truth_Points.pkl', 'wb') as f:
            pickle.dump(self, f)

    def copy(self, classToCopy):
        self.__dict__.update(copy.deepcopy(classToCopy.__dict__))

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
            print('Window Closed')

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

class CameraConfig():
    def __init__(self):
        self.cam_index = 0
        self.projectLidarPoints = False
        self.secondsBetweenImages = 1.0
        self.recording = False
        self.indexDict = {}
        self.aprilTagSize = 0.168
        self.undistort = False
        self.useCameraAsSource = True
        self.singleImageFilepath = None
        self.lidarFilepath = None

    def copy(self, configToCopy):
        self.__dict__.update(copy.deepcopy(configToCopy.__dict__))


class Camera():
    def __init__(self, gui):
        self.gui = gui
        self.calibration = None
        self.centers = None
        self.calibFile = ''
        self.camConfig = CameraConfig()
        self.scanForCameras()
        self.windowName =  'webcam'
        self.filepath = ''
        self.cam_frame = ctk.CTkFrame(master=self.gui)
        self.showWindow = False
        self.streamOrImgCombo = ctk.CTkComboBox(self.cam_frame, values=['Camera Stream', 'Static Image'], command=self.sourceUpdate)
        self.startStreamButton = ctk.CTkButton(master=self.cam_frame, text='Start Stream', fg_color='red', hover_color='blue')
        self.recordButton = ctk.CTkButton(master=self.cam_frame,text='Saving Imagery', fg_color='green', hover_color='navy', command=self.recordOff)
        self.selectCameraCombo = ctk.CTkComboBox(self.cam_frame, values=list(self.camConfig.indexDict.keys()),
                                                         command=self.selectCamera)
        self.selectFolderLabel = ctk.CTkLabel(self.cam_frame, text="../" + os.path.basename(os.path.normpath(self.filepath)))
        self.selectTruthPointsButton = ctk.CTkButton(master=self.cam_frame, text='Select LIDAR Points', hover_color='blue', command=self.selectLidarFile)
        if self.camConfig.lidarFilepath is not None:
            self.selectTruthPointsLabel = ctk.CTkLabel(self.cam_frame, text="../" + os.path.basename(os.path.normpath(self.camConfig.lidarFilepath)))
        else:
            self.selectTruthPointsLabel = ctk.CTkLabel(self.cam_frame, text='No Truth Loaded')
        self.lidarTruthPoints = TruthPoints()
        self.selectCalibLabel = None
        self.undistortCheckbox = None
        self.singleImageFolderSelect = ctk.CTkButton(self.cam_frame, text='Select AprilTag Img', command=self.selectSingleImage)
        self.singleImageTextButton = ctk.CTkButton(self.cam_frame, text='No Image Selected')
        self.loadFromCache()
        self.vc = cv2.VideoCapture(self.camConfig.cam_index, cv2.CAP_DSHOW)
        self.vc.set(cv2.CAP_PROP_FPS, 60)
        # self.vc.setExceptionMode(True)
        self.detector = Detector()
        self.img_idx = 0
        self.timeBetweenImgsEntry = None
        self.lastImageTime = 0
        self.camFrameGeometry = '455x420'
        self.cam_frame.grid_rowconfigure(list(range(3)), weight=1)  # configure grid system
        self.cam_frame.grid_columnconfigure(list(range(3)), weight=1)
        self.t1 = None
        self.aspectRatio = 1.0
        self.lastWidth = 1
        self.lastHeight = 1
        self.setupFrame()
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

    def updateLidarLabel(self):
        if self.camConfig.lidarFilepath is not None:
            self.selectTruthPointsLabel.configure(text=os.path.basename(self.camConfig.lidarFilepath))

    def loadTruthPoints(self):
        if self.camConfig.lidarFilepath is not None:
            with open(self.camConfig.lidarFilepath, 'rb') as f:
                test = pickle.load(f)
                self.lidarTruthPoints.copy(test)



    def ingestCalibration(self):
        try:
            with open(self.calibFile, 'rb') as f:
                self.calibration = pickle.load(f)
        except FileNotFoundError:
            if self.selectCalibLabel is not None:
                self.selectCalibLabel.configure(text='No Calibration Found')
                return


        scale = 2848.0 / 1424

        self.calibration.fx = scale * self.calibration.fx
        self.calibration.fy = scale * self.calibration.fy
        self.calibration.cx = scale * (self.calibration.cx + 0.5) - 0.5
        self.calibration.cy = scale * (self.calibration.cy + 0.5) - 0.5


        if self.selectCalibLabel is not None:
            self.selectCalibLabel.configure(text=os.path.basename(os.path.normpath(self.calibFile)))

        if self.undistortCheckbox is not None:
            self.undistortCheckbox.configure(state='normal')
        self.saveToCache()

    def scanForCameras(self):
        self.camConfig.indexDict = {}
        for camera_info in enumerate_cameras(cv2.CAP_DSHOW):
            self.camConfig.indexDict[camera_info.name] = camera_info.index

    def selectCamera(self, key):
        self.cam_index = self.camConfig.indexDict[key]
        self.vc.release()
        self.vc = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)


    def sourceUpdate(self, source):
        if source == 'Camera Stream':
            self.camConfig.useCameraAsSource = True
        else:
            self.camConfig.useCameraAsSource = False

        self.updateSingleOrStream(rowID=1)
    def updateSingleOrStream(self, rowID):
        if self.camConfig.useCameraAsSource:

            self.startStreamOff()

            if self.singleImageFolderSelect.grid_info():
                self.singleImageFolderSelect.grid_forget()
            if self.singleImageTextButton.grid_info():
                self.singleImageTextButton.grid_forget()

            self.startStreamButton.grid(row=rowID, column=0, padx=5, pady=5)
            if not self.selectCameraCombo.grid_info():
                self.selectCameraCombo.grid(row=rowID, column=1, padx=5, pady=5)

        else:
            if self.selectCameraCombo.grid_info():
                self.selectCameraCombo.grid_forget()
            if self.startStreamButton.grid_info():
                self.startStreamButton.grid_forget()

            if not self.singleImageFolderSelect.grid_info():
                self.singleImageFolderSelect.grid(row=1, column=0, padx=5, pady=5)
            if not self.singleImageTextButton.grid_info():
                self.singleImageTextButton.grid(row=1, column=1, padx=5, pady=5)

        self.saveToCache()

    def selectSingleImage(self):
        if self.camConfig.singleImageFilepath is None:
            initDir = self.filepath + '/..'
        else:
            initDir = os.path.normpath(self.camConfig.singleImageFilepath)

        self.camConfig.singleImageFilepath = filedialog.askopenfilename(initialdir=initDir, title="Select Image")
        self.singleImageTextButton.configure(text=os.path.basename(self.camConfig.singleImageFilepath))

    def setupFrame(self):
        rowID = 0

        self.streamOrImgCombo = ctk.CTkComboBox(self.cam_frame, values=['Camera Stream', 'Static Image'], command=self.sourceUpdate)
        self.streamOrImgCombo.grid(row=rowID, column=0, padx=5, pady=5)
        rowID += 1

        self.startStreamOff()

        self.singleImageTextButton.configure(command=self.detectSingleImage)
        if self.camConfig.singleImageFilepath is not None:
            self.singleImageTextButton.configure(text = os.path.basename(self.camConfig.singleImageFilepath))

        if self.camConfig.useCameraAsSource:
            self.streamOrImgCombo.set('Camera Stream')
            self.sourceUpdate('Camera Stream')
        else:
            self.streamOrImgCombo.set('Static Image')
            self.sourceUpdate('Static Image')

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

        detectAprilTagsCheckbox = ctk.CTkCheckBox(self.cam_frame, text='Detect April Tags')
        if self.detector is None:
            detectAprilTagsCheckbox.deselect()
        else:
            detectAprilTagsCheckbox.select()
        detectAprilTagsCheckbox.configure(command=self.toggleDetectTags)
        detectAprilTagsCheckbox.grid(row=rowID, column=0, padx=5, pady=5, sticky='ew')

        self.undistortCheckbox = ctk.CTkCheckBox(self.cam_frame, text='Undistort')
        if self.calibration is None:
            self.undistortCheckbox.configure(state='disabled')

        if self.camConfig.undistort is False or self.calibration is None:
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
        projectLidarPoints.grid(row=rowID, column=1, columnspan=2, padx=5, pady=5, sticky='ew')
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

        goBackButton = ctk.CTkButton(self.cam_frame, text="Return to Main", command=self.gui.returnToMain)
        goBackButton.grid(row=rowID, column=0, columnspan=2, padx=5, pady=5, sticky='ew')

        self.cam_frame.pack()

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
        self.startStreamButton.configure(command=self.startStreamOff, text='Stop Streaming', fg_color='green', hover_color='navy')

        self.selectCameraCombo.configure(state='disabled')

        self.streamOrImgCombo.configure(state='disabled')

        self.t1 = thread_with_exception(0, self.run)
        self.t1.start()


    def startStreamOff(self):
        self.startStreamButton.configure(command=self.startStreamOn, fg_color='red', hover_color='blue')

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

    def toggleDetectTags(self):
        if self.detector is None:
            self.detector = Detector()
        else:
            self.detector = None
        self.saveToCache()

    def toggleUndistort(self):
        if self.calibration is None:
            self.camConfig.undistort = False
            return

        if self.undistortCheckbox.get():
            self.camConfig.undistort = True
        else:
            self.camConfig.undistort = False
        self.saveToCache()

    def detectSingleImage(self):
        cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)
        frame = cv2.imread(self.camConfig.singleImageFilepath)
        self.detectAprilTagsAndPrint(frame)

        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()

    def run(self):

        cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)
        rval = False
        if self.camConfig.useCameraAsSource:
            rval, frame = self.vc.read()
            if rval:
                self.aspectRatio = float(frame.shape[1]) / float(frame.shape[0])
                cv2.resizeWindow(self.windowName, frame.shape[1], frame.shape[0])
                self.lastHeight = frame.shape[0]
                self.lastWidth = frame.shape[1]
        else:
            frame = cv2.imread(self.camConfig.singleImageFilepath)
            if frame:
                rval = True

        while rval and cv2.getWindowProperty(self.windowName, cv2.WND_PROP_VISIBLE) and self.showWindow:
            rval, frame = self.vc.read()
            self.detectAprilTagsAndPrint(frame)

            key = cv2.waitKey(20)
            if key == 27:  # exit on ESC
                self.startStreamOff()
                break

        self.startStreamOff()

    def detectAprilTagsAndPrint(self, frame):
        if self.calibration is not None and self.camConfig.undistort:
            frame = cv2.undistort(frame, cameraMatrix=self.calibration.getCameraMatrix(),
                                  distCoeffs=self.calibration.getDistortion())

        webGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.calibration is not None:
            K = self.calibration.getCameraMatrix()
            cx = int(self.calibration.cx)
            cy = int(self.calibration.cy)
        else:
            cx = int(frame.shape[1] / 2)
            cy = int(frame.shape[0] / 2)

        crosshairsH = np.array([[cx + 10, cy], [cx - 10, cy]])
        crosshairsV = np.array([[cx, cy + 10], [cx, cy - 10]])
        cv2.polylines(frame, [crosshairsH], True, (0, 255, 0), 2)
        cv2.polylines(frame, [crosshairsV], True, (0, 255, 0), 2)

        if self.detector is None:
            self.run_cleanup(frame)
            return

        if self.calibration is None:
            detections = self.detector.detect(webGray)
            self.centers = None
            for detection in detections:
                pixCenter = (int(detection.center[0]), int(detection.center[1]))
                cv2.polylines(frame, [detection.corners.astype(int)], True, (0, 255, 0), 2)
                cv2.putText(frame, str(detection.tag_id), pixCenter,
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 6)
                cv2.putText(frame, str(detection.tag_id), pixCenter,
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

                if self.centers is None:
                    self.centers = np.array(pixCenter)
                else:
                    self.centers = np.vstack((self.centers, np.array(pixCenter)))

                self.run_cleanup(frame)
                return


        K = self.calibration.getCameraMatrix()
        detections = self.detector.detect(img=webGray, estimate_tag_pose=True,
                                          camera_params=([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]),
                                          tag_size=self.camConfig.aprilTagSize)
        self.centers = None

        for detection in detections:
            pixCenter = (int(detection.center[0]), int(detection.center[1]))
            cv2.circle(frame, pixCenter, 3, (0,255,0), 3)
            cv2.polylines(frame, [detection.corners.astype(int)], True, (0, 255, 0), 2)
            cv2.putText(frame, str(detection.tag_id), pixCenter,
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 6)
            cv2.putText(frame, str(detection.tag_id), pixCenter,
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            loc = np.round(detection.pose_t, 3)
            loc_x_str = f'x: {loc[0]}'
            loc_y_str = f'y: {loc[1]}'
            loc_z_str = f'z: {loc[2]}'
            cv2.putText(frame, loc_x_str, (int(detection.center[0]), int(detection.center[1] + 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(frame, loc_y_str, (int(detection.center[0]), int(detection.center[1] + 50)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(frame, loc_z_str, (int(detection.center[0]), int(detection.center[1] + 75)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(frame, loc_x_str, (int(detection.center[0]), int(detection.center[1] + 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, loc_y_str, (int(detection.center[0]), int(detection.center[1] + 50)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, loc_z_str, (int(detection.center[0]), int(detection.center[1] + 75)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if self.centers is None:
                self.centers = np.array(pixCenter)
            else:
                self.centers = np.vstack((self.centers, np.array(pixCenter)))

        self.run_cleanup(frame)


    def run_cleanup(self, frame):
        if self.centers is not None:
            self.centers = self.centers.astype('float32')

        if self.camConfig.projectLidarPoints and self.detector is not None:
            frame = self.projectLidarPoints(frame)

        self.potentialResize()
        cv2.imshow(self.windowName, cv2.resize(frame, (self.lastWidth, self.lastHeight)))

        if self.recording and time.time() - self.lastImageTime > self.camConfig.secondsBetweenImages:
            cv2.imwrite(self.filepath + '\\' + str(self.img_idx) + '.png', frame)
            self.img_idx += 1
            self.lastImageTime = time.time()
            self.recordButton.configure(text=f'Saving Imagery: #{self.img_idx}')

    def projectLidarPoints(self, frame):

        if self.camConfig.undistort:
            distParams = np.zeros((5,))
        else:
            distParams = self.calibration.getDistortion()

        if len(self.centers) >= 4:
            ret, rvec, tvec = cv2.solvePnP(objectPoints=self.lidarTruthPoints.getTruthPointsNumpy(),
                                       imagePoints=self.centers,
                                       cameraMatrix=self.calibration.getCameraMatrix(),
                                       distCoeffs=distParams,
                                       flags=cv2.SOLVEPNP_ITERATIVE)
        else:
            ret = False

        if ret:
            projectedPoints_orig, _ = cv2.projectPoints(self.lidarTruthPoints.getTruthPointsNumpy(),
                                                        rvec=rvec,
                                                        tvec=tvec,
                                                        cameraMatrix=self.calibration.getCameraMatrix(),
                                                        distCoeffs=distParams)

            self.plotOnImg(frame, projectedPoints_orig[:, 0, :].astype(int),
                           list(self.lidarTruthPoints.getTruthPointsDict().keys()), (255, 255, 0))

        return frame

    def plotOnImg(self, img, points, names, color):
        for idx, pxPt in enumerate(points):
            cv2.circle(img, (int(pxPt[0]), int(pxPt[1])), 5, color, 5)
            textLoc = (int(pxPt[0]) - 30, int(pxPt[1] - 30))
            cv2.putText(img, str(names[idx]), textLoc, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 12,
                        cv2.LINE_AA)
            cv2.putText(img, str(names[idx]), textLoc, cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3, cv2.LINE_AA)

    def potentialResize(self):
        x, y, width, height = cv2.getWindowImageRect(self.windowName)

        if not self.lastHeight == height:
            cv2.resizeWindow(self.windowName, int(height * self.aspectRatio), height)
            self.lastHeight = height
            self.lastWidth = int(height * self.aspectRatio)
        elif not self.lastWidth == width:
            cv2.resizeWindow(self.windowName, width, int(width / self.aspectRatio))
            self.lastWidth = width
            self.lastHeight = int(width / self.aspectRatio)

