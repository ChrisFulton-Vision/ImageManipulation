import sys
import os

import cv2
import numpy as np
import glob
import time
import beepy
import customtkinter as ctk
from tkinter import filedialog
import pickle
import copy
from threading import Thread
import re
import superCalibrateCamera as cam
from PIL import Image
import colorsys
# import cProfile

sys.path.append(os.getcwd())

class Calib():
    def __init__(self):
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.k1 = None
        self.k2 = None
        self.p1 = None
        self.p2 = None
        self.k3 = None
        self.calTime = None
        self.numCBUsed = None
        self.rmsError = None
        self.width = None
        self.height = None
        self.hfov = None
        self.calStr = None

    def __str__(self):
        if self.calStr is not None:
            return self.calStr
        else:
            return ''

    def setCameraMatrix(self, mtx=None, fx=None, fy=None, cx=None, cy=None):
        if mtx is not None:
            self.fx = mtx[0,0]
            self.fy = mtx[1,1]
            self.cx = mtx[0,2]
            self.cy = mtx[1,2]
        elif fx is not None and fy is not None and cx is not None and cy is not None:
            self.fx = fx
            self.fy = fy
            self.cx = cx
            self.cy = cy
        else:
            self.fx = None
            self.fy = None
            self.cx = None
            self.cy = None
        self.updateCalStr()

    def getCameraMatrix(self):
        if self.fx is not None and self.fy is not None and self.cx is not None and self.cy is not None:
            return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]])
        else:
            return None

    def setDistortion(self, dist=None, k1=None, k2=None, p1=None, p2=None, k3=None):
        if dist is not None:
            self.k1 = dist[0]
            self.k2 = dist[1]
            self.p1 = dist[2]
            self.p2 = dist[3]
            self.k3 = dist[4]
        elif k1 is not None and k2 is not None and p1 is not None and p2 is not None and k3 is not None:
            self.k1 = k1
            self.k2 = k2
            self.p1 = p1
            self.p2 = p2
            self.k3 = k3
        else:
            self.k1 = None
            self.k2 = None
            self.p1 = None
            self.p2 = None
            self.k3 = None
        self.updateCalStr()

    def getDistortion(self):
        if self.k1 is not None and self.k2 is not None and self.p1 is not None and self.p2 is not None and self.k3 is not None:
            return np.array([self.k1, self.k2, self.p1, self.p2, self.k3]).flatten()
        else:
            return None

    def setAccessories(self, calTime, numCBUsed, width, height, hfov, rms):

        self.calTime = calTime
        self.numCBUsed = numCBUsed
        self.width = width
        self.height = height
        self.hfov = hfov
        self.rmsError = rms
        self.updateCalStr()

    def updateCalStr(self):
        mtx = self.getCameraMatrix()
        dist = self.getDistortion()

        if mtx is not None and dist is not None and self.calTime is not None and self.numCBUsed is not None:
            calStr = '#Camera matrix\n'
            calStr += 'fx={:.{}f}'.format(mtx[0, 0], 10) + '\n'
            calStr += 'fy={:.{}f}'.format(mtx[1, 1], 10) + '\n'
            calStr += 'cx={:.{}f}'.format(mtx[0, 2], 10) + '\n'
            calStr += 'cy={:.{}f}'.format(mtx[1, 2], 10) + '\n\n'

            calStr += '#Distortion coefficients\n'
            calStr += 'k1={:.{}f}'.format(dist[0], 10) + '\n'
            calStr += 'k2={:.{}f}'.format(dist[1], 10) + '\n'
            calStr += 'p1={:.{}f}'.format(dist[2], 10) + '\n'
            calStr += 'p2={:.{}f}'.format(dist[3], 10) + '\n'
            calStr += 'k3={:.{}f}'.format(dist[4], 10) + '\n\n'

            calStr += '#Total cal time (sec)\n'
            calStr += 'ct={:.{}f}'.format(self.calTime, 10) + '\n\n'

            calStr += '#Chessboards used\n'
            calStr += 'total=' + str(self.numCBUsed) + '\n'
            calStr += 'valid=' + str(self.numCBUsed) + '\n'
            calStr += 'rmsErr=' + str(self.rmsError) + '\n\n'

            calStr += '#Other\n'
            calStr += 'resolution=' + str(self.width) + 'x' + str(self.height) + '\n'
            calStr += 'hfov=' + str(self.hfov) + "\n"
            self.calStr = calStr

class ImageryConfig():
    def __init__(self):
        self.img_type = 'bmp'
        self.ret = []
        self.mtx = []
        self.dist = []
        self.imgCollection = []
        self.invertImage = False
        self.numInnerCornersW = 5
        self.numInnerCornersH = 5
        self.SUBnumInnerCornersW = 5
        self.SUBnumInnerCornersH = 5
        self.calMode = 'Chessboard'
        self.spacing = 30.0
        self.maxIter = 100
        self.minStepSize = 0.00001
        self.camCal = Calib()
        self.zeroTangentDist = True
        self.fixAspectRatio = True
        self.fixPrincipalPoint = True

    def copy(self, guiToCopy):
        self.__dict__.update(copy.deepcopy(guiToCopy.__dict__))

    def flags(self):
        flags = None
        if self.zeroTangentDist or self.fixAspectRatio or self.fixPrincipalPoint:
            if self.zeroTangentDist:
                flags = cv2.CALIB_ZERO_TANGENT_DIST
            if self.fixAspectRatio:
                if flags is not None:
                    flags += cv2.CALIB_FIX_ASPECT_RATIO
                else:
                    flags = cv2.CALIB_FIX_ASPECT_RATIO
            if self.fixPrincipalPoint:
                if flags is not None:
                    flags += cv2.CALIB_FIX_PRINCIPAL_POINT
                else:
                    flags = cv2.CALIB_FIX_PRINCIPAL_POINT
        return flags

class ImageData():
    def __init__(self, name=''):
        self.imageName = name
        self.include = True
        self.imgPts = None
        self.objPts = None
        self.residual = None
        self.sharpness = None

class FrontEndGui(ctk.CTk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.startTime = time.time()
        self.endTime = time.time()
        self.imageConfig = ImageryConfig()
        self.calculating = False
        self.displayImagePointsButton = None
        self.imageConfigWindowObjects = []
        self.includeCheckboxes = []
        self.mainGeometry = '255x500'
        self.imageWinGeometry = '1250x550'
        self.calGeometry = '200x500'
        self.configGeometry = '500x225'
        self.currImg = None
        self.firstClick = None
        self.camera = None
        self.initImageFrame = False
        self.scale = 1.0

        self.filepath = ''
        self.loadFromCache(True)

        self.t1 = None
        self.t2 = None
        self.t3 = None

        # Custom TKinter Main Window Configuration
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")
        self.title("Camera Calibrater")
        self.geometry(self.mainGeometry)

        self.mainFrame = ctk.CTkFrame(master=self)
        self.mainFrame.grid_rowconfigure([0, 1, 2], weight=1)  # configure grid system
        self.mainFrame.grid_columnconfigure([0, 1, 2], weight=1)
        self.mainFrame.pack()

        self.imageFrame = ctk.CTkFrame(master=self)
        self.imageFrame.grid_rowconfigure(list(range(10)), weight=1)  # configure grid system
        self.imageFrame.grid_columnconfigure(list(range(11)), weight=1)
        self.subImageFrame = ctk.CTkScrollableFrame(master=self.imageFrame)
        self.subImageFrame.grid(row=2, rowspan=5, column=0, columnspan=12, sticky='NSEW')

        self.calFrame = ctk.CTkFrame(master=self)

        self.calFrame.grid_rowconfigure([0, 1], weight=1)
        self.calFrame.grid_columnconfigure([0], weight=1)

        self.configFrame = ctk.CTkFrame(master=self)
        self.stoppingIterationEntry = ctk.CTkEntry(master=self.configFrame)
        self.stoppingIterationButton = ctk.CTkButton(master=self.configFrame,text='Update', command=self.stoppingCritIterUpdate)
        self.stoppingMinStepSizeEntry = ctk.CTkEntry(master=self.configFrame)
        self.stoppingMinStepSizeButton = ctk.CTkButton(master=self.configFrame,text='Update', command=self.stoppingCritMinStepSizeUpdate)

        self.leftArrow = ctk.CTkImage( light_image=Image.open('leftArrow.png'), size=(20,20))
        self.rightArrow = ctk.CTkImage( light_image=Image.open('rightArrow.png'), size=(20,20))

        self.imgInvertProtectedButton = ctk.CTkButton(master=self.imageFrame, text='Invert All', hover_color='navy',
                                               fg_color='blue', width=100, command=self.unprotectInvert)

        self.imgRotateCCWProtectedButton = ctk.CTkButton(master=self.imageFrame, image=self.leftArrow, text='All', hover_color='navy',
                                               fg_color='blue', width=100, command=self.unprotectRotateCCW)

        self.imgRotateCWProtectedButton = ctk.CTkButton(master=self.imageFrame, image=self.rightArrow, text='All', hover_color='navy',
                                               fg_color='blue', width=100, command=self.unprotectRotateCW)

        self.imgGrayProtectedButton = ctk.CTkButton(master=self.imageFrame, text='Grayscale All', hover_color='navy',
                                               fg_color='blue', width=100, command=self.unprotectAllGrayscale)



        # Use rowID to keep track of which row each object is placed. Allows for easy code integration of new objects
        rowID = 0

        self.openCameraButton = ctk.CTkButton(master=self.mainFrame, text='Open Camera', command=self.openCamera, fg_color='navy')
        self.openCameraButton.grid(row=rowID, column=0, padx=5, pady=5, sticky='ew')
        rowID += 1

        # Add button that allows the user to select the folder of the images
        self.selectFolderButton = ctk.CTkButton(master=self.mainFrame, text='Select Folder',command=lambda: self.selectFolder(), fg_color="navy")
        self.selectFolderButton.grid(row=rowID, column=0, padx=5, pady=5, sticky="ew")
        # On same line, add a button that allows the user to interact with the images in the folder
        self.openImagesButton = ctk.CTkButton(master=self.mainFrame, text='Not Selected', fg_color="black", command=self.openImageWindow)
        self.openImagesButton.grid(row=rowID, column=1, padx=5, pady=5, sticky="ew")
        rowID += 1

        # Allow user to select type of calibration
        modes = ['Chessboard', 'Circles']
        self.selectModeLabel = ctk.CTkLabel(master=self.mainFrame, text='Calibration Type')
        self.selectModeLabel.grid(row=rowID, column=0, padx=5, pady=5, sticky='ew')
        self.selectModeCombo = ctk.CTkComboBox(master=self.mainFrame, values=modes,command=self.updateMode )
        self.selectModeCombo.grid(row=rowID, column=1, padx=5, pady=5, sticky='ew')
        rowID += 1

        # Allow user to select type of image
        imgTypes = ['png','bmp', 'img','jpg']
        self.selectImgTypeLabel = ctk.CTkLabel(master=self.mainFrame, text='File Type')
        self.selectImgTypeLabel.grid(row=rowID, column=0, padx=5, pady=5, sticky='ew')
        self.selectImgTypeCombo = ctk.CTkComboBox(master=self.mainFrame, values=imgTypes,command=self.updateImgType )
        self.selectImgTypeCombo.grid(row=rowID, column=1, padx=5, pady=5, sticky='ew')
        rowID += 1

        # Add text label that shows what folder is selected by the user
        self.folderLabel = ctk.CTkLabel(master=self.mainFrame, text="../" + os.path.basename(os.path.normpath(self.filepath)))
        self.folderLabel.grid(row=rowID, column=0, columnspan=2, sticky='ew')
        rowID += 1

        # Add option for the user to invert the images (useful for White-hot images, like from LWIR)
        self.invertImagesCheckbox = ctk.CTkCheckBox(master=self.mainFrame, text='Invert Image? (LWIR)', command=lambda: self.invertImageToggle())
        self.invertImagesCheckbox.grid(row=rowID,columnspan=2, column=0, padx=5, pady=5, sticky='ew')
        rowID += 1

        # Create options for the user to select the number of inner chessboard corners
        values = [str(num) for num in range(5, 21)]
        self.cornerInputLabel = ctk.CTkLabel(self.mainFrame, text='# of Inner CB Corners')
        self.cornerInputLabel.grid(row=rowID, column=0, columnspan=2, padx=5, pady=5)
        rowID += 1
        self.widthLabel = ctk.CTkLabel(self.mainFrame, text='Width')
        self.widthLabel.grid(row=rowID, column=0, padx=5, pady=5)
        self.heightLabel = ctk.CTkLabel(self.mainFrame, text='Height')
        self.heightLabel.grid(row=rowID, column=1, padx=5, pady=5)
        self.widthComboEntry = ctk.CTkComboBox(master=self.mainFrame,
                                               values=values,
                                               command=self.widthInput)
        self.widthComboEntry.grid(row=rowID, column=0, padx=5, pady=5)
        self.heightComboEntry = ctk.CTkComboBox(master=self.mainFrame,
                                                values=values,
                                                command=self.heightInput)
        self.heightComboEntry.grid(row=rowID, column=1, padx=5, pady=5)
        rowID += 1

        # Do the same thing for the sub-pixel search
        self.SUBcornerInputLabel = ctk.CTkLabel(self.mainFrame, text='# for sub-Pixel Search')
        self.SUBcornerInputLabel.grid(row=rowID, column=0, columnspan=2, padx=5, pady=5)
        rowID += 1

        self.SUBwidthLabel = ctk.CTkLabel(self.mainFrame, text='Width')
        self.SUBwidthLabel.grid(row=rowID, column=0, padx=5, pady=5)
        self.SUBheightLabel = ctk.CTkLabel(self.mainFrame, text='Height')
        self.SUBheightLabel.grid(row=rowID, column=1, padx=5, pady=5)

        self.SUBwidthComboEntry = ctk.CTkComboBox(master=self.mainFrame,
                                               values=values,
                                               command=self.SUBwidthInput)
        self.SUBwidthComboEntry.grid(row=rowID, column=0, padx=5, pady=5)

        self.SUBheightComboEntry = ctk.CTkComboBox(master=self.mainFrame,
                                                values=values,
                                                command=self.SUBheightInput)
        self.SUBheightComboEntry.grid(row=rowID, column=1, padx=5, pady=5)
        rowID += 1

        # Initiate the calibration procedure in a separate thread so that the GUI still functions
        self.calibrateButton = ctk.CTkButton(self.mainFrame, text="Calibrate!", state="disabled", command=self.calibrate)
        self.calibrateButton.grid(row=rowID, column=0, columnspan=2, padx=5, pady=5)
        rowID += 1

        self.loadFromCache()
        self.loadImages()
        self.restoreFromImageConfig()

        # Once a calibration is active, allow user to display a window that manages the calibration
        self.displayCal = ctk.CTkButton(self.mainFrame, text='Display Calibration', state='disabled', command=self.openCalWindow)
        self.displayCal.grid(row=rowID, column=0, columnspan=2, padx=5, pady=5)
        if self.imageConfig.camCal.calStr is not None:
            self.displayCal.configure(state='normal')
        rowID += 1

        # Once a calibration is active, allow user to display a window that manages the calibration
        self.configWindowButton = ctk.CTkButton(self.mainFrame, text='Criteria Configuration',command=self.openConfigWindow)
        self.configWindowButton.grid(row=rowID, column=0, columnspan=2, padx=5, pady=5)
        rowID += 1

        self.protectClearCache()

        # self.setupImageFrame()
        self.updateConfigWindow()
        self.updateCalWindow()

    # def __del__(self):
    def on_closing(self):
        if self.camera is not None:
            self.camera.recordOff()
            self.camera.startStreamOff()
        self.destroy()

    def openCamera(self):
        if self.camera is None:
            self.camera = cam.Camera(self)

        self.camera.showWindow = True
        self.unpackAllFrames()
        self.geometry(self.camera.camFrameGeometry)
        self.camera.setupFrame()

    def isFramePacked(self, frame):
        return frame.winfo_ismapped()

    def updateMode(self, newMode):
        self.imageConfig.calMode = newMode
        self.saveToCache()

    def updateImgType(self, newImgType):
        self.imageConfig.img_type = newImgType
        self.imageConfig.imgCollection = []
        self.loadImages()

    def fileName(self, idx):
        return self.filepath + '\\' + self.imageConfig.imgCollection[idx].imageName

    def displayImagePointsThread(self):
        self.displayImagePointsButton.configure(text='Calculating', fg_color='gray', state='disabled')
        self.returnToMain()
        self.openImagesButton.configure(command=None, fg_color='gray')
        self.t3 = Thread(target=self.displayImagePoints, daemon=True)
        self.t3.start()

    def displayImagePoints(self):
        # This function examines all of the active image chessboard results and plots them to a graph.
        # This is useful if the user wants to see what regions have already been included in the image. If a
        # calibration is complete, then it also color-codes the images using HSV to highlight high-performing and
        # low-performing images

        # Must assume same height for each image
        img = cv2.imread(self.fileName(0))
        width = img.shape[1]
        height = img.shape[0]

        for imgClass in self.imageConfig.imgCollection:
            if imgClass.include:
                self.findChessboardCorners(imgClass, showImage=False)

        self.sortBySharpness()
        # Create a blank, black image with 3 color channels (BGR)
        blank_image = np.zeros((height, width, 3), np.uint8)

        # Prepare boolean to confirm at least 1 image produced a chessboard
        gotAtLeastOneImage = False
        minVal = 10.0
        maxVal = 0.0
        for imageClass in self.imageConfig.imgCollection:
            if imageClass.include and imageClass.residual is not None:
                minVal = min([minVal, imageClass.residual])
                maxVal = max([maxVal, imageClass.residual])

        residual = copy.copy(minVal)

        #Examine each image's chessboard solution. If the image has an associated residual, color code the image.
        for idx, imageClass in enumerate(self.imageConfig.imgCollection):

            if imageClass.include and imageClass.imgPts is not None:
                gotAtLeastOneImage = True

                if imageClass.residual is not None:
                    residual = imageClass.residual

                for imgPt in imageClass.imgPts:
                    b,g,r = colorsys.hsv_to_rgb(0.4-0.4*(residual - minVal)/(maxVal-minVal), 1.0, 1.0)
                    cv2.circle(blank_image, (round(imgPt[0][0]),round(imgPt[0][1])), 2, (int(255*b),int(255*g),int(255*r)), 2)

        self.updateImageFrame()

        # If we succeeded at at least one image, then display the image of all of the found corners
        if gotAtLeastOneImage:
            # Draw Legend
            b, g, r = colorsys.hsv_to_rgb(0, 1.0, 1.0)
            cv2.circle(blank_image, (5,20), 2, (int(255*b),int(255*g),int(255*r)), 2)
            cv2.putText(blank_image, 'Residual of: ' + str(round(maxVal,2)), (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255))
            b, g, r = colorsys.hsv_to_rgb(0.4, 1.0, 1.0)
            cv2.circle(blank_image, (5,40), 2, (int(255*b),int(255*g),int(255*r)), 2)
            cv2.putText(blank_image, 'Residual of: ' + str(round(minVal,2)), (10,45), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255))
            cv2.resize(blank_image, (1000, 1000))

            # Convert the openCV object to an Image object, which is ingested by customtkinter
            filledImage = Image.fromarray(blank_image)

            ctkImage = ctk.CTkImage(filledImage, size=(1000,1000))
            imagePointWindow = ctk.CTkToplevel(self)
            imagePointWindow.geometry('1000x1000')
            imagePointWindow.title("Chessboard Corner Locations")
            label = ctk.CTkLabel(master=imagePointWindow, image=ctkImage, text="")
            label.pack()

        self.displayImagePointsButton.configure(text='Display Image Points', fg_color='green', state='normal')
        self.saveToCache()
        self.openImagesButton.configure(command=self.openImageWindow, fg_color='green')

    def saveCal(self, button):
        if self.imageConfig.camCal.calStr is not None:
            saveFile = open(self.filepath + '/calibration.txt','w')
            saveFile.write(self.imageConfig.camCal.calStr)
            with open(self.filepath + '/calibration.pkl', 'wb') as f:
                pickle.dump(self.imageConfig.camCal, f)
            button.configure(fg_color='navy')

    def openImageWindow(self):
        self.unpackAllFrames(exception='imageFrame')
        self.geometry(self.imageWinGeometry)
        self.imageFrame.pack(fill='both', expand=True)

    def openCalWindow(self):
        self.unpackAllFrames(exception='calFrame')
        self.geometry(self.calGeometry)
        self.calFrame.pack(fill='both', expand=True)

    def openConfigWindow(self):
        self.unpackAllFrames(exception='configFrame')
        self.geometry(self.configGeometry)
        self.configFrame.pack()

    def returnToMain(self):
        self.unpackAllFrames()
        self.geometry(self.mainGeometry)
        self.mainFrame.pack()

    def updateCalWindow(self):
        # If we have a previous calibration
        if self.imageConfig.camCal.calStr is not None:

            # Then display the calibration
            saveCalButton = ctk.CTkButton(master=self.calFrame, text='Save Calibration')
            saveCalButton.configure(command=lambda btn = saveCalButton: self.saveCal(btn))
            saveCalButton.grid(row=0, column=0, padx=0, pady=0)

            cal = ctk.CTkLabel(master=self.calFrame, text=self.imageConfig.camCal.calStr, justify='left', anchor='w')
            cal.grid(row=1,column=0,padx=0, pady=0)

            backToMainButton = ctk.CTkButton(master=self.calFrame, text='Go back', command=self.returnToMain)
            backToMainButton.grid(row=3, column=0, padx=5, pady=5)

    def updateConfigWindow(self):
        rowID = 0
        values = [1, 10, 100, 1000]
        stoppingIterationLabel = ctk.CTkLabel(master=self.configFrame,text='Max Iterations: ')
        stoppingIterationLabel.grid(row=rowID, column=0, padx=5, pady=5)
        self.stoppingIterationEntry.configure(placeholder_text=str(self.imageConfig.maxIter))
        self.stoppingIterationEntry.bind('<Return>', self.stoppingCritIterUpdate)

        self.stoppingIterationEntry.grid(row=rowID, column=1, padx=5, pady=5)
        self.stoppingIterationButton.grid(row=rowID, column=2, padx=5, pady=5)
        rowID += 1

        stoppingMinStepSizeLabel = ctk.CTkLabel(master=self.configFrame, text='Stopping Min Step Size: ')
        stoppingMinStepSizeLabel.grid(row=rowID, column=0, padx=5, pady=5)
        self.stoppingMinStepSizeEntry.configure(placeholder_text=str(self.imageConfig.minStepSize))
        self.stoppingMinStepSizeEntry.bind('<Return>', self.stoppingCritMinStepSizeUpdate)
        self.stoppingMinStepSizeEntry.grid(row=rowID, column=1, padx=5, pady=5)
        self.stoppingMinStepSizeButton.grid(row=rowID, column=2, padx=5, pady=5)
        rowID += 1

        fixPrincipalPointCB = ctk.CTkCheckBox(master=self.configFrame, text='Fix Principle Point', checkbox_height=20)
        if self.imageConfig.fixPrincipalPoint:
            fixPrincipalPointCB.select()
        else:
            fixPrincipalPointCB.deselect()
        fixPrincipalPointCB.configure(command=self.toggleFixPrincipalPoint)
        fixPrincipalPointCB.grid(row=rowID, column=0, columnspan=2, padx=0, pady=0, sticky='ew')
        rowID += 1

        fixAspectRatioCB = ctk.CTkCheckBox(master=self.configFrame, text='Fix Aspect Ratio', checkbox_height=20)
        if self.imageConfig.fixAspectRatio:
            fixAspectRatioCB.select()
        else:
            fixAspectRatioCB.deselect()
        fixAspectRatioCB.configure(command=self.toggleFixAspectRatio)
        fixAspectRatioCB.grid(row=rowID, column=0, columnspan=2, padx=0, pady=0, sticky='ew')
        rowID += 1

        zeroTangentDistCB = ctk.CTkCheckBox(master=self.configFrame, text='Zero Tangent Distance', checkbox_height=20)
        if self.imageConfig.zeroTangentDist:
            zeroTangentDistCB.select()
        else:
            zeroTangentDistCB.deselect()
        zeroTangentDistCB.configure(command=self.toggleZeroTangentDist)
        zeroTangentDistCB.grid(row=rowID, column=0, columnspan=2, padx=0, pady=0, sticky='ew')
        rowID += 1

        backToMainButton = ctk.CTkButton(master=self.configFrame, text='Go back', command=self.returnToMain)
        backToMainButton.grid(row=rowID, column=1, padx=5, pady=5)

    def stoppingCritMinStepSizeUpdate(self, entry=None):
        try:
            newStep = float(self.stoppingMinStepSizeEntry.get())
        except ValueError:
            newStep = None
        if isinstance(newStep, float) and newStep > 0:
            self.imageConfig.minStepSize = newStep
        else:
            self.stoppingMinStepSizeEntry.delete(0, ctk.END)
            self.stoppingMinStepSizeEntry.insert(0, str(self.imageConfig.minStepSize))
        self.saveToCache()
        self.stoppingMinStepSizeButton.configure(fg_color='yellow')
        self.stoppingMinStepSizeButton.after(1,self.update())
        self.stoppingMinStepSizeButton.after(500,self.restoreMinSizeButton())

    def restoreMinSizeButton(self):
        self.stoppingMinStepSizeButton.configure(fg_color='green')
    def restoreIterationButton(self):
        self.stoppingIterationButton.configure(fg_color='green')

    def stoppingCritIterUpdate(self, entry=None):
        try:
            newIter = int(self.stoppingIterationEntry.get())
        except ValueError:
            newIter = None
        if isinstance(newIter, int) and newIter > 0:
            self.imageConfig.maxIter = newIter
        else:
            self.stoppingIterationEntry.delete(0, ctk.END)
            self.stoppingIterationEntry.insert(0, str(self.imageConfig.maxIter))
        self.saveToCache()
        self.stoppingIterationButton.configure(fg_color='yellow')
        self.stoppingIterationButton.after(1,self.update())
        self.stoppingIterationButton.after(500,self.restoreIterationButton())

    def toggleZeroTangentDist(self):
        self.imageConfig.zeroTangentDist = not self.imageConfig.zeroTangentDist

    def toggleFixPrincipalPoint(self):
        self.imageConfig.fixPrincipalPoint = not self.imageConfig.fixPrincipalPoint

    def toggleFixAspectRatio(self):
        self.imageConfig.fixAspectRatio = not self.imageConfig.fixAspectRatio

    def unpackAllFrames(self, exception=''):
        if not exception == 'mainFrame' and self.isFramePacked(self.mainFrame):
            self.mainFrame.pack_forget()
        if not exception == 'imageFrame' and self.isFramePacked(self.imageFrame):
            self.imageFrame.pack_forget()
        if not exception == 'configFrame' and self.isFramePacked(self.configFrame):
            self.configFrame.pack_forget()
        if not exception == 'calFrame' and self.isFramePacked(self.calFrame):
            self.calFrame.pack_forget()
        if not exception == 'camFrame' and self.camera is not None and self.isFramePacked(self.camera.cam_frame):
            self.camera.cam_frame.pack_forget()

    def setupImageFrame(self):

        rowID = 0
        selectAll = ctk.CTkButton(master=self.imageFrame, text='Include All Images',
                                           command=self.includeAll)
        selectAll.grid(row=rowID, column=0, padx=5, pady=5)

        removeUnselected = ctk.CTkButton(master=self.imageFrame, text='Remove Unselected', command=self.removeUnused)
        removeUnselected.grid(row=rowID, column=3, padx=5, pady=5, columnspan=2)
        self.displayImagePointsButton = ctk.CTkButton(master=self.imageFrame, text='Display All Chessboard Points', command=self.displayImagePointsThread)
        self.displayImagePointsButton.grid(row=rowID,column=5, columnspan=2, padx=5,pady=5)
        backToMainButton = ctk.CTkButton(master=self.imageFrame, text='Go back', command=self.returnToMain)
        backToMainButton.grid(row=rowID, column=7, padx=5, pady=5)

        self.protectInvert()
        self.protectAllGrayscale()
        self.protectRotateCCW()
        self.protectRotateCW()

        # print(len(self.imageConfigWindowObjects))
        # print(len(self.imageConfig.imgCollection))

        self.imageConfigWindowObjects = []
        rowID = 0

        while len(self.imageConfigWindowObjects) < len(self.imageConfig.imgCollection):
            self.createNewRow(rowID)
            rowID += 1


    def createNewRow(self, rowID):
        imgIncludeCheckbox = ctk.CTkCheckBox(master=self.subImageFrame, text='')
        imgIncludeCheckbox.grid(row=rowID, column=0)

        imgNameButton = ctk.CTkButton(master=self.subImageFrame, text='')
        imgNameButton.grid(row=rowID, column=1, padx=5, pady=5)

        imgRes = ctk.CTkLabel(master=self.subImageFrame, text='')
        imgRes.grid(row=rowID, column=2, padx=5, pady=5)

        imgShp = ctk.CTkLabel(master=self.subImageFrame, text='')
        imgShp.grid(row=rowID, column=3, padx=5, pady=5)

        imgRestoreButton = ctk.CTkButton(master=self.subImageFrame, text='Restore')
        imgRestoreButton.grid(row=rowID, column=4, padx=5, pady=5)

        imgFindCornersButton = ctk.CTkButton(master=self.subImageFrame, text='Find Corners')
        imgFindCornersButton.grid(row=rowID, column=5, padx=5, pady=5)

        imgInvertButton = ctk.CTkButton(master=self.subImageFrame, text='Invert Image')
        imgInvertButton.grid(row=rowID, column=6, padx=5, pady=5)

        imgGrayButton = ctk.CTkButton(master=self.subImageFrame, text='Grayscale Image')
        imgGrayButton.grid(row=rowID, column=7, padx=5, pady=5)

        imgCCWRotateButton = ctk.CTkButton(master=self.subImageFrame, text='', image=self.leftArrow)
        imgCCWRotateButton.grid(row=rowID, column=8, padx=5, pady=5)

        imgCWRotateButton = ctk.CTkButton(master=self.subImageFrame, text='', image=self.rightArrow)
        imgCWRotateButton.grid(row=rowID, column=9, padx=5, pady=5)

        self.imageConfigWindowObjects.append([imgIncludeCheckbox, imgNameButton, imgRes, imgShp, imgRestoreButton,
                                              imgFindCornersButton, imgInvertButton, imgGrayButton,
                                              imgCCWRotateButton, imgCWRotateButton])

    def updateImageFrame(self):

        # print('Rows: ', len(self.imageConfigWindowObjects))
        # print('Imgs: ', len(self.imageConfig.imgCollection), '\n')

        while len(self.imageConfigWindowObjects) > len(self.imageConfig.imgCollection):
            for item in self.imageConfigWindowObjects[-1]:
                item.grid_forget()
                item.destroy()
            self.imageConfigWindowObjects.pop(-1)
        while len(self.imageConfigWindowObjects) < len(self.imageConfig.imgCollection):
            self.createNewRow(len(self.imageConfigWindowObjects))

        self.sortByResidual()

        self.includeCheckboxes = []
        rowID = 0
        for idx, imgClass in enumerate(self.imageConfig.imgCollection):
            (imgIncludeCheckbox,
             imgNameButton,
             imgRes,
             imgShp,
             imgRestoreButton,
             imgFindCornersButton,
             imgInvertButton,
             imgGrayButton,
             imgCCWRotateButton,
             imgCWRotateButton) = self.imageConfigWindowObjects[idx]

            if imgClass.include:
                imgIncludeCheckbox.select()
            else:
                imgIncludeCheckbox.deselect()
            imgIncludeCheckbox.configure(command=lambda ident=idx: self.updateInclusion(ident))

            imgNameButton.configure(text=imgClass.imageName, command=lambda iC=imgClass: self.showBasicImage(iC))

            if imgClass.residual is None:
                currRes = ''
            elif imgClass.residual == 10000.0:
                currRes = 'Disabled'
            else:
                currRes = 'Res: ' + str(round(imgClass.residual,3))

            imgRes.configure(text=currRes)

            if imgClass.sharpness is None:
                currShrp = ''
            else:
                currShrp = 'Shrp: ' + str(round(imgClass.sharpness,3))

            imgShp.configure(text=currShrp)

            imgRestoreButton.configure(command=lambda imgC=imgClass: self.restore(imgC))

            imgFindCornersButton.configure(command=lambda imgC=imgClass: self.findChessboardCorners(imgC, True, True))

            imgInvertButton.configure(command=lambda imgC=imgClass: self.invertIndividualImage(imgC))

            imgGrayButton.configure(command=lambda imgC=imgClass: self.grayscaleIndividualImage(imgC))

            imgCCWRotateButton.configure(command=lambda imgC=imgClass: self.rotateCCWIndividualImage(imgC))

            imgCWRotateButton.configure(command=lambda imgC=imgClass: self.rotateCWIndividualImage(imgC))

            rowID += 1


    def removeUnused(self):
        if not os.path.exists(self.filepath + '/Removed'):
            os.makedirs(self.filepath + '/' + 'Removed')
        removeIds = []
        for idx, imgClass in enumerate(self.imageConfig.imgCollection):

            if not imgClass.include and os.path.exists(self.filepath + '/Removed/' + imgClass.imageName):
                os.replace(self.fileName(idx), self.filepath + '/Removed/' + imgClass.imageName)

            if not imgClass.include:
                removeIds.append(idx)

        for id in reversed(removeIds):
            self.imageConfig.imgCollection.pop(id)
        self.saveToCache()
        self.updateImageFrame()

        if len(self.imageConfig.imgCollection) > 5:
            self.openImagesButton.configure(text=str(len(self.imageConfig.imgCollection)) + ' valid images', fg_color="green")
            self.calibrateButton.configure(state="normal")
        else:
            self.openImagesButton.configure(text=str(len(self.imageConfig.imgCollection)) + ' valid images', fg_color="red")

    def copyToRemovedFolder(self, imgClass):
        if not os.path.exists(self.filepath + '/Removed'):
            os.makedirs(self.filepath + '/' + 'Removed')

        src_path = self.filepath + '/' + imgClass.imageName
        dst_path = self.filepath + '/Removed/' + imgClass.imageName

        if not os.path.exists(dst_path):
            self.writeFile(src_path,dst_path)

    def restore(self, imgClass):
        if os.path.exists(self.filepath + '/Removed/' + imgClass.imageName):

            src_path = self.filepath + '/Removed/' + imgClass.imageName
            dst_path = self.filepath + '/' + imgClass.imageName

            self.writeFile(src_path, dst_path)

    def writeFile(self, src_path, dst_path):
        try:
            # Open the source file in binary read mode
            with open(src_path, 'rb') as src:
                # Open the destination file in binary write mode
                with open(dst_path, 'wb') as dest:
                    # Read and write the file in chunks
                    while True:
                        chunk = src.read(4096)  # Read in chunks of 4 KB
                        if not chunk:
                            break
                        dest.write(chunk)

        except FileNotFoundError as e:
            print(f"Error: {e}")
        except PermissionError as e:
            print(f"Permission error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def updateIncludeCheckboxes(self):
        for idx, checkbox in enumerate(self.includeCheckboxes):
            if self.imageConfig.imgCollection[idx].include:
                checkbox.select()
            else:
                checkbox.deselect()

    def unprotectInvert(self):
        self.imgInvertProtectedButton.configure(command=self.invertAll, fg_color='green', hover_color='dark green')
        self.imgInvertProtectedButton.update()
        self.after(2000, self.protectInvert)

    def unprotectAllGrayscale(self):
        self.imgGrayProtectedButton.configure(command=self.grayscaleAll, fg_color='green', hover_color='dark green')
        self.imgGrayProtectedButton.update()
        self.after(2000, self.protectAllGrayscale)

    def unprotectRotateCCW(self):
        self.imgRotateCCWProtectedButton.configure(command=self.rotateAllCCW, fg_color='green', hover_color='dark green')
        self.imgRotateCCWProtectedButton.update()
        self.after(2000, self.protectRotateCCW)

    def unprotectRotateCW(self):
        self.imgRotateCWProtectedButton.configure(command=self.rotateAllCW, fg_color='green', hover_color='dark green')
        self.imgRotateCWProtectedButton.update()
        self.after(2000, self.protectRotateCW)

    def protectInvert(self, row=1):
        self.imgInvertProtectedButton.configure(fg_color='blue', hover_color='cyan4', command=self.unprotectInvert)
        if not self.imgInvertProtectedButton.winfo_ismapped():
            self.imgInvertProtectedButton.grid(row=row, column=5, padx=5, pady=5, sticky='ew')

    def protectAllGrayscale(self, row=1):
        self.imgGrayProtectedButton.configure(fg_color='blue', hover_color='cyan4', command=self.unprotectAllGrayscale)
        if not self.imgGrayProtectedButton.winfo_ismapped():
            self.imgGrayProtectedButton.grid(row=row, column=6, padx=5, pady=5, sticky='ew')

    def protectRotateCCW(self, row=1):
        self.imgRotateCCWProtectedButton.configure(fg_color='blue', hover_color='cyan4', command=self.unprotectRotateCCW)
        if not self.imgRotateCCWProtectedButton.winfo_ismapped():
            self.imgRotateCCWProtectedButton.grid(row=row, column=7, padx=5, pady=5, sticky='ew')

    def protectRotateCW(self, row=1):
        self.imgRotateCWProtectedButton.configure(fg_color='blue', hover_color='cyan4',
                                                   command=self.unprotectRotateCW)
        if not self.imgRotateCWProtectedButton.winfo_ismapped():
            self.imgRotateCWProtectedButton.grid(row=row, column=8, padx=5, pady=5, sticky='ew')

    def invertAll(self):
        self.imgInvertProtectedButton.configure(fg_color='black')
        self.imgInvertProtectedButton.update()
        for imgClass in self.imageConfig.imgCollection:
            self.invertIndividualImage(imgClass)
        self.protectInvert()

    def grayscaleAll(self):
        self.imgGrayProtectedButton.configure(fg_color='black')
        self.imgGrayProtectedButton.update()
        for imgClass in self.imageConfig.imgCollection:
            self.grayscaleIndividualImage(imgClass)
        self.protectAllGrayscale()

    def rotateAllCW(self):
        self.imgRotateCWProtectedButton.configure(fg_color='black')
        self.imgRotateCWProtectedButton.update()
        for imgClass in self.imageConfig.imgCollection:
            self.rotateCWIndividualImage(imgClass)
        self.protectRotateCW()

    def rotateAllCCW(self):
        self.imgRotateCCWProtectedButton.configure(fg_color='black')
        self.imgRotateCCWProtectedButton.update()
        for imgClass in self.imageConfig.imgCollection:
            self.rotateCCWIndividualImage(imgClass)
        self.protectRotateCCW()

    def invertIndividualImage(self, imgClass):
        filepath = self.filepath + '\\' + imgClass.imageName
        img = cv2.imread(filepath)
        invt_img = cv2.bitwise_not(img)
        cv2.imwrite(filepath, invt_img)

    def grayscaleIndividualImage(self, imgClass):
        self.copyToRemovedFolder(imgClass)
        filepath = self.filepath + '\\' + imgClass.imageName
        img = cv2.imread(filepath)
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cv2.imwrite(filepath, gray_img)

    def rotateCWIndividualImage(self, imgClass):
        filepath = self.filepath + '\\' + imgClass.imageName
        img = cv2.imread(filepath)
        invt_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(filepath, invt_img)

    def rotateCCWIndividualImage(self, imgClass):
        filepath = self.filepath + '\\' + imgClass.imageName
        img = cv2.imread(filepath)
        invt_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(filepath, invt_img)


    def includeAll(self):
        for imgClass in self.imageConfig.imgCollection:
            imgClass.include = True
        self.saveToCache()
        self.updateIncludeCheckboxes()

    def unprotectClearCache(self):
        clearCacheButton = ctk.CTkButton(master=self.mainFrame,text='Really Clear Cache', fg_color='green', command=self.clearCache)
        clearCacheButton.grid(row=13, column=0, columnspan=2, padx=5, pady=5)
        self.after(2000, self.protectClearCache)
        self.after(2000, clearCacheButton.grid_forget)

    def protectClearCache(self):
        clearCacheButton = ctk.CTkButton(master=self.mainFrame,text='Clear Cache', fg_color='blue', hover_color='navy', command=self.unprotectClearCache)
        clearCacheButton.grid(row=13, column=0, columnspan=2, padx=5, pady=5)

    def clearCache(self):
        if os.path.exists(self.filepath + '\\imagery_cache.pkl'):
            filepath = copy.copy(self.filepath)
            os.remove(self.filepath + '\\imagery_cache.pkl')

            clearCacheButton = ctk.CTkButton(master=self.mainFrame, text='Clearing', fg_color='yellow', text_color='black', hover_color='yellow')
            clearCacheButton.grid(row=13, column=0, columnspan=2, padx=5, pady=5)

            self.imageConfig = ImageryConfig()
            self.filepath = filepath
            self.restoreFromWindowState()
            self.loadImages()
            self.updateImageFrame()
        else:
            clearCacheButton = ctk.CTkButton(master=self.mainFrame, text='No cache!', fg_color='red', hover_color='red')
            clearCacheButton.grid(row=13, column=0, columnspan=2, padx=5, pady=5)
        self.after(2000, self.protectClearCache)

    def updateInclusion(self, idx):
        self.imageConfig.imgCollection[idx].include = not self.imageConfig.imgCollection[idx].include
        self.saveToCache()

    def restoreFromImageConfig(self):
        if self.imageConfig.invertImage:
            self.invertImagesCheckbox.select()
        else:
            self.invertImagesCheckbox.deselect()
        self.widthComboEntry.set(str(self.imageConfig.numInnerCornersW))
        self.heightComboEntry.set(str(self.imageConfig.numInnerCornersH))
        self.SUBwidthComboEntry.set(str(self.imageConfig.SUBnumInnerCornersW))
        self.SUBheightComboEntry.set(str(self.imageConfig.SUBnumInnerCornersH))
        self.folderLabel.configure(text="../" + os.path.basename(os.path.normpath(self.filepath)))
        self.selectImgTypeCombo.set(self.imageConfig.img_type)
        self.selectModeCombo.set(self.imageConfig.calMode)

    def restoreFromWindowState(self):
        self.imageConfig.invertImage = self.invertImagesCheckbox.get()
        self.imageConfig.numInnerCornersW = int(self.widthComboEntry.get())
        self.imageConfig.numInnerCornersH = int(self.heightComboEntry.get())
        self.imageConfig.SUBnumInnerCornersW = int(self.SUBwidthComboEntry.get())
        self.imageConfig.SUBnumInnerCornersH = int(self.SUBheightComboEntry.get())
        self.imageConfig.img_type = self.selectImgTypeCombo.get()
        self.imageConfig.calMode = self.selectModeCombo.get()

    def saveToCache(self):

        if len(self.imageConfig.imgCollection) > 0:
            with open(self.filepath + '\\imagery_cache.pkl', 'wb') as f:
                pickle.dump(self.imageConfig, f)

        with open('filepath_cache.pkl', 'wb') as f:
            pickle.dump(self.filepath, f)

    def loadFromCache(self, init=False):
        if init and not os.path.exists('filepath_cache.pkl'):
            self.filepath = os.getcwd()
            return

        if init:
            with open('filepath_cache.pkl', 'rb') as filepathOpen:
                self.filepath = pickle.load(filepathOpen)
                return

        if os.path.exists(self.filepath + '\\imagery_cache.pkl'):
            with open(self.filepath + '\\imagery_cache.pkl', 'rb') as imageConfigOpen:
                self.imageConfig.copy(pickle.load(imageConfigOpen))

        self.loadImages()

    def calibrate(self):
        self.calculating = True
        self.calibrateButton.configure(state='disabled',text='Calculating...', fg_color='gray')
        self.t1 = Thread(target=self.threadedCal, daemon=True)
        self.t1.start()


    def threadedCal(self):
        for imgClass in self.imageConfig.imgCollection:
            if imgClass.include is True:
                self.findChessboardCorners(imgClass, False)
        self.calibrateCamera()
        self.saveToCache()
        self.updateImageFrame()
        self.calibrateButton.configure(state='normal',text='Calibrate', fg_color='green')
        self.calculating = False


    def widthInput(self, newVal):
        self.imageConfig.numInnerCornersW = int(newVal)
        self.saveToCache()

    def heightInput(self, newVal):
        self.imageConfig.numInnerCornersH = int(newVal)
        self.saveToCache()

    def SUBwidthInput(self, newVal):
        self.imageConfig.SUBnumInnerCornersW = int(newVal)
        if self.imageConfig.SUBnumInnerCornersW > self.imageConfig.numInnerCornersW:
            self.imageConfig.SUBnumInnerCornersW = self.imageConfig.numInnerCornersW
        self.saveToCache()

    def SUBheightInput(self, newVal):
        self.imageConfig.SUBnumInnerCornersH = int(newVal)
        if self.imageConfig.SUBnumInnerCornersH > self.imageConfig.numInnerCornersH:
            self.imageConfig.SUBnumInnerCornersH = self.imageConfig.numInnerCornersH
        self.saveToCache()

    def selectFolder(self):
        self.filepath = filedialog.askdirectory(initialdir=self.filepath + "/..", mustexist=True, title="Select Imagery Folder")
        self.imageConfig.imgCollection = []
        self.imageConfig.camCal.calStr = None
        self.loadFromCache(False)
        self.loadImages()
        self.folderLabel.configure(text=os.path.basename(self.filepath))

    def natural_sort(self, l):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)

    def sortBySharpness(self):
        self.imageConfig.imgCollection = sorted(self.imageConfig.imgCollection, key=lambda img: self.sharpnessTest(img.sharpness))

    def sortByResidual(self):
        self.imageConfig.imgCollection = sorted(self.imageConfig.imgCollection, key=lambda img: self.sharpnessTest(img.residual))

    def sharpnessTest(self, sharpValue):
        if sharpValue is None:
            return 50.0
        else:
            return sharpValue

    def loadImages(self):
        imgs = glob.glob(os.path.join(self.filepath, '*.' + self.imageConfig.img_type))

        for img in imgs:
            img = os.path.basename(img)
            isAlreadyPresent = False
            for existingImgClass in self.imageConfig.imgCollection:
                if img == existingImgClass.imageName:
                    isAlreadyPresent = True
            if not isAlreadyPresent:
                self.imageConfig.imgCollection.append(ImageData(os.path.basename(img)))

        if len(self.imageConfig.imgCollection) > 5:
            self.openImagesButton.configure(text=str(len(self.imageConfig.imgCollection)) + ' valid images', fg_color="green")
            self.calibrateButton.configure(state="normal")
        else:
            self.openImagesButton.configure(text=str(len(self.imageConfig.imgCollection)) + ' valid images', fg_color="red")

        self.saveToCache()
        if not self.initImageFrame:
            self.setupImageFrame()
            self.initImageFrame = True
        self.updateImageFrame()

    def invertImageToggle(self):
        self.imageConfig.invertImage = not self.imageConfig.invertImage
        self.saveToCache()

    def showBasicImage(self, imgClass):
        img = cv2.imread(self.filepath + '\\' + imgClass.imageName)

        h, w, toss = img.shape
        if h > 1080 or w > 1080:
            self.scale = max(1080 / h, 1080 / w) * 1.1
            dispImg = cv2.resize(img, (int(w * self.scale), int(h * self.scale)))
        else:
            dispImg = copy.copy(img)

        cv2.namedWindow(imgClass.imageName)

        cv2.imshow(imgClass.imageName, dispImg)

        self.currImgClass = imgClass
        self.currImg = copy.copy(dispImg)
        cv2.setMouseCallback(imgClass.imageName, self.click_event)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        self.firstClick = None

    def click_event(self, event, x, y, flags, param):

        if event == cv2.EVENT_RBUTTONDOWN or flags == cv2.EVENT_FLAG_RBUTTON:
            self.firstClick = None
            cv2.imshow(self.currImgClass.imageName, self.currImg)
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.firstClick = (x,y)
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON and self.firstClick is not None:
            cloned_img = copy.copy(self.currImg)
            cv2.rectangle(cloned_img, self.firstClick, (x,y), (0,255,0), 2)
            cv2.imshow(self.currImgClass.imageName, cloned_img)
        elif event == cv2.EVENT_LBUTTONUP and self.firstClick is not None:
            img = cv2.imread(self.filepath + '\\' + self.currImgClass.imageName)
            self.copyToRemovedFolder(self.currImgClass)
            cv2.destroyAllWindows()

            x = int(x / self.scale)
            y = int(y / self.scale)
            first_x = int(self.firstClick[0] / self.scale)
            first_y = int(self.firstClick[1] / self.scale)

            low_x = min(x, first_x)
            low_y = min(y, first_y)
            high_x = max(x, first_x)
            high_y = max(y, first_y)

            # print(img.shape)
            new_img = copy.copy(img)
            new_img[:,:low_x] = np.zeros(new_img[:,:low_x].shape)
            new_img[:low_y] = np.zeros(new_img[:low_y].shape)
            new_img[:,high_x:] = np.zeros(new_img[:,high_x:].shape)
            new_img[high_y:] = np.zeros(new_img[high_y:].shape)

            cv2.imwrite(self.filepath + '/' + self.currImgClass.imageName, new_img)

            # if h > 1080 or w > 1080:
            #     new_img = cv2.resize(new_img, (int(w / scale), int(h / scale)))

            h, w, toss = new_img.shape
            dispImg = cv2.resize(new_img, (int(w * self.scale), int(h * self.scale)))

            cv2.imshow("New", dispImg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    def findChessboardCorners(self, imgClass, showImage = True, updateImageFrame = False):
        img = cv2.imread(self.filepath + '\\' + imgClass.imageName)

        if img is None:
            imgClass.include = False
            return


        if imgClass.imgPts is None:
            objp = np.zeros((self.imageConfig.numInnerCornersW * self.imageConfig.numInnerCornersH, 3), np.float32)
            objp[:, :2] = np.mgrid[0:self.imageConfig.numInnerCornersW, 0:self.imageConfig.numInnerCornersH].T.reshape(-1, 2) * self.imageConfig.spacing


            if self.imageConfig.invertImage:
                temp = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                inv_img = cv2.bitwise_not(temp)
                gray = inv_img
            else:
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            if self.imageConfig.calMode == 'Chessboard':
                ret, corners = cv2.findChessboardCorners(gray,
                                                         (self.imageConfig.numInnerCornersW,
                                                          self.imageConfig.numInnerCornersH),
                                                         flags=cv2.ADAPTIVE_THRESH_GAUSSIAN_C)

            elif self.imageConfig.calMode == 'Circles':
                
                # params = cv2.SimpleBlobDetector_Params()
                #
                # params.filterByArea = True
                # params.minArea = 50
                # blob = cv2.SimpleBlobDetector_create(params)


                ret, corners = cv2.findCirclesGrid(gray,
                                             (self.imageConfig.numInnerCornersW,
                                              self.imageConfig.numInnerCornersH),
                                             flags=cv2.ADAPTIVE_THRESH_GAUSSIAN_C)#, blobDetector=blob)
            else:
                ret = False
                print('Unknown Cal Mode')

            if ret == True:
                imgClass.objPts = objp

                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, self.imageConfig.maxIter, self.imageConfig.minStepSize)
                corners2 = cv2.cornerSubPix(gray, np.float32(corners),
                                            (self.imageConfig.SUBnumInnerCornersH, self.imageConfig.SUBnumInnerCornersW),
                                            (-1, -1), criteria)
                imgClass.imgPts = corners2

                sharpness = cv2.estimateChessboardSharpness(gray, (self.imageConfig.numInnerCornersW, self.imageConfig.numInnerCornersH), np.float32(corners2))
                imgClass.sharpness = sharpness[0][0]
            else:
                imgClass.include = False
                self.openImagesButton.configure(text=str(len(self.imageConfig.imgCollection)) + ' valid images')


        if updateImageFrame:
            self.updateImageFrame()

        self.saveToCache()

        if showImage:
            if imgClass.imgPts is not None:
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img,
                                            (self.imageConfig.numInnerCornersW, self.imageConfig.numInnerCornersH),
                                            imgClass.imgPts, True)

                x = int(np.average(imgClass.imgPts[:, 0, 0]))
                y = int(np.average(imgClass.imgPts[:, 0, 1]))
                min_X = max(int(np.min(imgClass.imgPts[:, 0, 0])), 0)
                max_X = min(int(np.max(imgClass.imgPts[:, 0, 0])), img.shape[0])
                min_Y = max(int(np.min(imgClass.imgPts[:, 0, 1])), 0)
                max_Y = min(int(np.max(imgClass.imgPts[:, 0,  1])), img.shape[1])
                dist_x = int(np.max(np.array([[max_X - x], [x - min_X]])))
                dist_y = int(np.max(np.array([[max_Y - y], [y - min_Y]])))

                roi = cv2.getRectSubPix(img, (2*dist_x, 2*dist_y), (x, y))

                h, w, chan = roi.shape
                dispImg = cv2.resize(roi, (int(w * self.scale), int(h * self.scale)))

                cv2.imshow('Chessboard Corners Detected', dispImg)
                cv2.waitKey(0)
            else:
                imgClass.imgPts = None
                imgClass.objPts = None
                imgClass.include = False

                if showImage:
                    h, w = gray.shape
                    # if h > 1080 or w > 1080:
                    #     scale = max(h / 1080, w / 1080) * 1.1
                    #     gray = cv2.resize(gray, (int(w / scale), int(h / scale)))
                    dispImg = cv2.resize(gray, (int(w * self.scale), int(h * self.scale)))
                    cv2.imshow('NO CHESSBOARD CORNERS FOUND', dispImg)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()


    def findIndexGivenImageName(self, name):
        sol_idx = None
        for idx, imgClass in enumerate(self.imageConfig.imgCollection):
            if imgClass.imageName == name:
                sol_idx = idx
        return sol_idx


    def sortByResdiualHandleNone(self, residual):
        if residual is None:
            return 1000.0
        else:
            return residual

    def calibrateCamera(self):
        '''
        Calibrate a camera and output calibration parameters and residuals.

        Inputs:
        folder - relative path to folder of images (this path should be relative to the working directory). Images MUST be .jpg
        corners - tuple of the number of corners (horizontal, vertical) to look for (not number of squares!)
        spacing - the spacing between the corners
        print_results - Boolean on whether the results should be printed to the screen
        show_results = Boolean on whether to display images on the screen
        exclude - list of images that should be excluded, e.g. [0,1,2,3,13,22,27]

        Outputs: Tuple of
        retval - RSS residual error, first term returned from cv.calibrateCamera
        K - instrinsic calibration matrix (3,3) numpy ndarray
        R - distortion coefficient (5,) ndarray
        res - list of residuals for each input image
        '''

        self.startTime = time.time()

        images = []
        objPoints = []
        imgPoints = []
        for idx,imgClass in enumerate(self.imageConfig.imgCollection):
            if imgClass.include:
                images.append(self.fileName(idx))
                objPoints.append(imgClass.objPts)
                imgPoints.append(imgClass.imgPts)


        initialVector = np.eye(3, 3)
        initialVector[2, 2] = 0

        gray = cv2.cvtColor(cv2.imread(self.fileName(0)), cv2.COLOR_BGR2GRAY)

        calValues = cv2.calibrateCameraROExtended(
            objPoints,
            imgPoints,
            gray.shape[::-1],
            1,
            initialVector,
            None,
            flags=(self.imageConfig.flags()))
        ret = calValues[0]
        mtx = calValues[1]
        dist = calValues[2]
        rvecs = calValues[3]
        tvecs = calValues[4]
        newObjPoints = calValues[5]
        stdDevIntrinsics = calValues[6]
        stdDevExtrinsics = calValues[7]
        stdDevObjPoints = calValues[8]
        residuals = calValues[9]

        for img_idx, img in enumerate(images):
            self.imageConfig.imgCollection[self.findIndexGivenImageName(os.path.basename(img))].residual = residuals[img_idx][0]

        # Sort the images by their residual Values
        self.sortByResidual()
        # self.imageConfig.imgCollection.reverse()

        self.endTime = time.time()

        fovx, fovy, focalLength, principalPoint, aspectRatio = cv2.calibrationMatrixValues(mtx, gray.shape[::-1], 25., 25.)

        self.imageConfig.camCal.setCameraMatrix(mtx=mtx)
        self.imageConfig.camCal.setDistortion(dist=dist.T)
        self.imageConfig.camCal.setAccessories(calTime=self.endTime - self.startTime, numCBUsed=len(images),
                                               width=gray.shape[::-1][0],height=gray.shape[::-1][0],hfov=fovx, rms=ret)

        self.saveToCache()
        self.updateCalWindow()
        self.displayCal.configure(state='normal')

        self.t2 = Thread(target=self.beep_beep, daemon=True)
        self.t2.start()

        self.imageConfig.ret = ret
        self.imageConfig.mtx = mtx
        self.imageConfig.dist = dist

    def beep_beep(self):
        beepy.beep(sound=6)
        # 1:'coin', 2:'robot_error', 3:'error', 4:'ping', 5:'ready', 6:'success', 7:'wilhelm'

if __name__ == '__main__':

    gui = FrontEndGui()
    # cProfile.run('gui.mainloop()')
    gui.mainloop()

