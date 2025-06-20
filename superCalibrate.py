import datetime
import sys, os, cv2, glob, time, copy, colorsys, pickle
from os.path import join
import numpy as np
import customtkinter as ctk
from tkinter import filedialog
from threading import Thread
from PIL import Image
from enum import Enum

from Calibration import Calibration
import superCalibrateCamera as cam

sys.path.append(os.getcwd())
GREEN = '#2FA572'


class CalibrationType(Enum):
    Chessboard = 'Chessboard'
    Circles = 'Circles'
    chArUco = 'chArUco'


class ImageryCalibrationConfig():
    '''
    This class stores information cleanly about any calibration that has occurred or is intended to occur. Because
    it stores all the information and settings for the calibration, this class is neatly packaged in a cache for
    easy reloading.
    '''

    def __init__(self):
        self.img_type = 'bmp'
        self.img_collection = []
        self.invert_image = False
        self.num_inner_corners_W = 8
        self.num_inner_corners_H = 11
        self.SUB_num_inner_corners_W = 5
        self.SUB_num_inner_corners_H = 5
        self.calMode = CalibrationType.Chessboard
        self.spacing = 30.0
        self.maxIter = 100
        self.minStepSize = 0.00001
        self.camCal = Calibration()
        self.zeroTangentDist = True
        self.fixAspectRatio = True
        self.fixPrincipalPoint = True
        self.fisheye = False

    @property
    def num_valid_imgs(self):
        '''
        Property method for CalConfig. Call as:
        calConfig = ImageryCalibrationConfig()
        num_imgs = calConfig.num_valid_imgs
        :return: Number of valid images, excluding those specifically not included in submenu or previous cache.
        Images that aren't included but are in the list are visible in submenus, but not included in calibration.
        '''
        num_valid = 0
        for img in self.img_collection:
            if img.include:
                num_valid += 1
        return num_valid

    def copy(self, configToCopy):
        '''
        Caching helper function. When reading from binary, copy all named dictionary items in the CalibrationConfig.
        Changes to this class will cause version errors when reading in old configs IFF names are changed or removed.
        Adding NEW parameters does not create a version error, but the parameter will not change through the load.
        :param configToCopy: Loaded value, typically from cache.
        :return:
        '''
        for obj in configToCopy.__dict__:
            try:
                self.__dict__[obj] = configToCopy.__dict__[obj]
            except KeyError:
                # Allows for versioning issues, changed naming conventions.
                pass

    @property
    def flags(self):
        '''
        Helper function that returns the composite flag value for a calibration based on own settings.
        :return: cv2-style flags for calibration.
        '''
        flags = None
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
    '''
    This class stores information maintained by a single image. The image MUST have a name which is its filename.
    Include sets whether the image is part of the calibration.
    imgPts stores 2d identified features (such as chessboard corners)
    objPts stores 3d expected features (such as 3d coords for the chessboard)
    Residual characterizes the performance of the calibration. This is a good estimate for image quality.
    Sharpness characterizes the blurriness of the image. This is a rough estimate for image quality.
    '''

    def __init__(self, name=''):
        self.imageName = name
        self.include = True
        self.imgPts = None
        self.objPts = None
        self.residual = None
        self.sharpness = None


class FrontEndGui(ctk.CTk):
    '''
    This is the main GUI that the user interacts with when running the program. This class manages the main loop,
    displays the main buttons, creates the submenus (but waits to show them until asked), and generally maintains
    system state.
    '''

    def __init__(self, *args, **kwargs):

        # Super class init, necessary for customTkinter
        super().__init__(*args, **kwargs)

        # "Nicely" closes camera, if it is active
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Stores calibration times
        self.startTime = time.time()
        self.endTime = time.time()

        # Stores calibration configuration states
        self.imageConfig = ImageryCalibrationConfig()

        # Bool for calibration state
        self.calculating = False

        # Default Geometries for window and subwindows
        self.mainGeometry = '255x560'
        self.imageWinGeometry = '1250x550'
        self.calGeometry = '250x650'
        self.configGeometry = '500x225'

        # Various helper variable NONE-initialization
        self.displayImagePointsButton = None
        self.currImg = None
        self.firstClick = None
        self.camera = None
        self.initImageFrame = False
        self.imageConfigWindowObjects = []
        self.scale = 1.0

        self.filepath = ''
        self.loadFromCache(True)

        # Empty thread-holding objects
        self.t1 = None
        self.t2 = None
        self.t3 = None

        ##########################################################################
        # Custom TKinter Main Window Configuration
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")
        self.title("Camera Calibrater")
        self.geometry(self.mainGeometry)

        self.mainFrame = ctk.CTkFrame(master=self)
        self.mainFrame.grid_rowconfigure([0, 1, 2], weight=1)  # configure grid system
        self.mainFrame.grid_columnconfigure([0, 1, 2], weight=1)
        self.mainFrame.pack()

        ##########################################################################
        # Image Management Frame Setup
        self.imageFrame = ctk.CTkFrame(master=self)
        self.imageFrame.grid_rowconfigure(list(range(10)), weight=1)  # configure grid system
        self.imageFrame.grid_columnconfigure(list(range(11)), weight=1)
        self.subImageFrame = ctk.CTkScrollableFrame(master=self.imageFrame)
        self.subImageFrame.grid(row=2, rowspan=5, column=0, columnspan=12, sticky='NSEW')

        self.leftArrow = ctk.CTkImage(light_image=Image.open('leftArrow.png'), size=(20, 20))
        self.rightArrow = ctk.CTkImage(light_image=Image.open('rightArrow.png'), size=(20, 20))

        self.imgInvertProtectedButton = ctk.CTkButton(master=self.imageFrame, text='Invert All', hover_color='navy',
                                                      fg_color='blue', width=100, command=self.unprotectInvert)

        self.imgRotateCCWProtectedButton = ctk.CTkButton(master=self.imageFrame, image=self.leftArrow, text='All',
                                                         hover_color='navy',
                                                         fg_color='blue', width=100, command=self.unprotectRotateCCW)

        self.imgRotateCWProtectedButton = ctk.CTkButton(master=self.imageFrame, image=self.rightArrow, text='All',
                                                        hover_color='navy',
                                                        fg_color='blue', width=100, command=self.unprotectRotateCW)

        self.imgGrayProtectedButton = ctk.CTkButton(master=self.imageFrame, text='Grayscale All', hover_color='navy',
                                                    fg_color='blue', width=100, command=self.unprotectAllGrayscale)

        ##########################################################################
        # Calibration Frame Setup
        self.calFrame = ctk.CTkFrame(master=self)

        self.calFrame.grid_rowconfigure([0, 1], weight=1)
        self.calFrame.grid_columnconfigure([0], weight=1)

        ##########################################################################
        # Custom TKinter Configuration Setup
        self.configFrame = ctk.CTkFrame(master=self)
        self.stoppingIterationEntry = ctk.CTkEntry(master=self.configFrame)
        self.stoppingIterationButton = ctk.CTkButton(master=self.configFrame, text='Update',
                                                     command=self.stoppingCritIterUpdate)
        self.stoppingMinStepSizeEntry = ctk.CTkEntry(master=self.configFrame)
        self.stoppingMinStepSizeButton = ctk.CTkButton(master=self.configFrame, text='Update',
                                                       command=self.stoppingCritMinStepSizeUpdate)
        ##########################################################################
        # Now Initialize the buttons on the main frame
        # Use rowID to keep track of which row each object is placed. Allows for easy code integration of new objects
        rowID = 0

        self.openCameraButton = ctk.CTkButton(master=self.mainFrame, text='Open Camera', command=self.openCamera,
                                              fg_color='navy')
        self.openCameraButton.grid(row=rowID, column=0, padx=5, pady=5, sticky='ew')
        rowID += 1

        # Add button that allows the user to select the folder of the images
        self.selectFolderButton = ctk.CTkButton(master=self.mainFrame, text='Select Folder', command=self.selectFolder,
                                                fg_color="navy")
        self.selectFolderButton.grid(row=rowID, column=0, padx=5, pady=5, sticky="ew")

        # On same line, add a button that allows the user to interact with the images in the folder
        self.openImagesButton = ctk.CTkButton(master=self.mainFrame, text='Not Selected', fg_color="black",
                                              command=self.openImageWindow)
        self.openImagesButton.grid(row=rowID, column=1, padx=5, pady=5, sticky="ew")
        rowID += 1

        # Allow user to select type of calibration
        modes = [type.value for type in CalibrationType]
        self.selectModeLabel = ctk.CTkLabel(master=self.mainFrame, text='Calibration Type')
        self.selectModeLabel.grid(row=rowID, column=0, padx=5, pady=5, sticky='ew')
        self.selectModeCombo = ctk.CTkComboBox(master=self.mainFrame, values=modes, command=self.updateMode)
        self.selectModeCombo.grid(row=rowID, column=1, padx=5, pady=5, sticky='ew')
        rowID += 1

        # Allow user to select type of image
        imgTypes = ['png', 'bmp', 'img', 'jpg']
        self.selectImgTypeLabel = ctk.CTkLabel(master=self.mainFrame, text='File Type')
        self.selectImgTypeLabel.grid(row=rowID, column=0, padx=5, pady=5, sticky='ew')
        self.selectImgTypeCombo = ctk.CTkComboBox(master=self.mainFrame, values=imgTypes, command=self.updateImgType)
        self.selectImgTypeCombo.grid(row=rowID, column=1, padx=5, pady=5, sticky='ew')
        rowID += 1

        # Add text label that shows what folder is selected by the user
        self.folderLabel = ctk.CTkLabel(master=self.mainFrame,
                                        text="../" + os.path.basename(os.path.normpath(self.filepath)))
        self.folderLabel.grid(row=rowID, column=0, columnspan=2, sticky='ew')
        rowID += 1

        # Add option for the user to invert the images (useful for White-hot images, like from LWIR)
        self.invertImagesCheckbox = ctk.CTkCheckBox(master=self.mainFrame, text='Invert Image? (LWIR)',
                                                    command=self.invertImageToggle)
        if self.imageConfig.invert_image:
            self.invertImagesCheckbox.select()
        self.invertImagesCheckbox.grid(row=rowID, columnspan=2, column=0, padx=5, pady=5, sticky='ew')
        rowID += 1

        self.fisheyeLensCheckbox = ctk.CTkCheckBox(master=self.mainFrame, text='Fisheye Lens?',
                                                   command=self.toggleFisheye)
        if self.imageConfig.fisheye:
            self.fisheyeLensCheckbox.select()
        self.fisheyeLensCheckbox.grid(row=rowID, columnspan=2, column=0, padx=5, pady=5, sticky='ew')
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
        self.calibrateButton = ctk.CTkButton(self.mainFrame, text="Calibrate!", state="disabled",
                                             command=self.calibrate)
        self.calibrateButton.grid(row=rowID, column=0, columnspan=2, padx=5, pady=5)
        rowID += 1

        self.loadFromCache()

        # Once a calibration is active, allow user to display a window that manages the calibration
        self.displayCal = None
        self.createCalibrationDisplay(rowID)
        rowID += 1

        # Once a calibration is active, allow user to display a window that manages the calibration
        self.configWindowButton = None
        self.createConfigWindowButton(rowID)
        rowID += 1

        # Create a button that allows a user to clear the cache, but they must select it twice
        self.clearCacheRow = rowID
        self.protectClearCache()

        # Now that necessary starting variables are created, load states from cache

        self.restoreFromImageConfig()
        self.updateConfigWindow()
        self.updateCalWindow()

    def createCalibrationDisplay(self, rowID):
        self.displayCal = ctk.CTkButton(self.mainFrame, text='Display Calibration', state='disabled',
                                        command=self.openCalWindow)
        self.displayCal.grid(row=rowID, column=0, columnspan=2, padx=5, pady=5)
        if self.imageConfig.camCal.validCal:
            self.displayCal.configure(state='normal')

    def createCalibrationWindowButton(self, rowID):
        self.displayCal = ctk.CTkButton(self.mainFrame, text='Display Calibration', state='disabled',
                                        command=self.openCalWindow)
        self.displayCal.grid(row=rowID, column=0, columnspan=2, padx=5, pady=5)
        if self.imageConfig.camCal.validCal:
            self.displayCal.configure(state='normal')

    def createConfigWindowButton(self, rowID):
        self.configWindowButton = ctk.CTkButton(self.mainFrame, text='Criteria Configuration',
                                                command=self.openConfigWindow)
        self.configWindowButton.grid(row=rowID, column=0, columnspan=2, padx=5, pady=5)

    def on_closing(self):
        '''
        Closes camera streaming down, if it's still active, ensuring all destructors called in sequence
        '''
        if self.camera is not None:
            self.camera.shutdown()
        self.after(1000, self.destroy())

    def openCamera(self):
        '''
        Opens Camera class and camera class GUI overrides main GUI while active
        :return:
        '''
        if self.camera is None:
            self.camera = cam.CameraGui(self)

        self.camera.showWindow = True
        self.unpackAllFrames()
        self.geometry(self.camera.camFrameGeometry)
        self.camera.setupFrame()

    def isFramePacked(self, frame):
        return frame.winfo_ismapped()

    def updateMode(self, newMode):
        '''
        Setter from button click
        '''
        self.imageConfig.calMode = CalibrationType(newMode)
        self.saveToCache()

    def updateImgType(self, newImgType):
        '''
        Setter from button click
        '''
        self.imageConfig.img_type = newImgType
        self.imageConfig.img_collection = []
        self.loadImages()

    def fileName(self, idx):
        '''
        :param idx: the number of the image in the collection
        :return: the filename with root filepath prepended
        '''
        return join(self.filepath, self.imageConfig.img_collection[idx].imageName)

    def displayImagePointsThread(self):
        '''
        This button spawns a thread that looks at all the existing chessboard corners (see displayImagePoints()).
        While the thread is spawned, the image menu is unavailable (as it is being consistently updated).
        '''
        self.displayImagePointsButton.configure(text='Calculating', fg_color='gray', state='disabled')
        self.returnToMain()
        self.openImagesButton.configure(command=None, fg_color='gray', state='disabled')
        self.t3 = Thread(target=self.displayImagePoints, daemon=True)
        self.t3.start()

    def displayImagePoints(self):
        '''
        This function examines all of the active image chessboard results and plots them to a graph.
        This is useful if the user wants to see what regions have already been included in the image. If a
        calibration is complete, then it also color-codes the images using HSV to highlight high-performing and
        low-performing images
        '''

        # Must assume same height for each image
        img = cv2.imread(self.fileName(0))
        width = img.shape[1]
        height = img.shape[0]

        for imgClass in self.imageConfig.img_collection:
            if imgClass.include:
                self.findChessboardCorners(imgClass, showImage=False)

        self.sortBySharpness()
        # Create a blank, black image with 3 color channels (BGR)
        blank_image = np.zeros((height, width, 3), np.uint8)

        # Prepare boolean to confirm at least 1 image produced a chessboard
        gotAtLeastOneImage = False
        minVal = 10.0
        maxVal = 0.0
        for imageClass in self.imageConfig.img_collection:
            if imageClass.include and imageClass.residual is not None:
                minVal = min([minVal, imageClass.residual])
                maxVal = max([maxVal, imageClass.residual])

        residual = copy.copy(minVal)

        #Examine each image's chessboard solution. If the image has an associated residual, color code the image.
        for idx, imageClass in enumerate(self.imageConfig.img_collection):

            if imageClass.include and imageClass.imgPts is not None:
                gotAtLeastOneImage = True

                if imageClass.residual is not None:
                    residual = imageClass.residual

                for imgPt in imageClass.imgPts:
                    b, g, r = colorsys.hsv_to_rgb(0.4 - 0.4 * (residual - minVal) / (maxVal - minVal), 1.0, 1.0)
                    cv2.circle(blank_image, (round(imgPt[0][0]), round(imgPt[0][1])), 2,
                               (int(255 * b), int(255 * g), int(255 * r)), 2)

        self.updateImageFrame()

        # If we succeeded at at least one image, then display the image of all of the found corners
        if gotAtLeastOneImage:
            # Draw Legend
            b, g, r = colorsys.hsv_to_rgb(0, 1.0, 1.0)
            cv2.circle(blank_image, (5, 20), 2, (int(255 * b), int(255 * g), int(255 * r)), 2)
            cv2.putText(blank_image, 'Residual of: ' + str(round(maxVal, 2)), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255))
            b, g, r = colorsys.hsv_to_rgb(0.4, 1.0, 1.0)
            cv2.circle(blank_image, (5, 40), 2, (int(255 * b), int(255 * g), int(255 * r)), 2)
            cv2.putText(blank_image, 'Residual of: ' + str(round(minVal, 2)), (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255))
            cv2.resize(blank_image, (1000, int(1000 / width * height)))

            # Convert the openCV object to an Image object, which is ingested by customtkinter
            filledImage = Image.fromarray(blank_image)

            ctkImage = ctk.CTkImage(filledImage, size=(1000, int(1000 / width * height)))
            imagePointWindow = ctk.CTkToplevel(self)
            imagePointWindow.geometry('1000x' + str(int(1000 / width * height)))
            imagePointWindow.title("Chessboard Corner Locations")
            label = ctk.CTkLabel(master=imagePointWindow, image=ctkImage, text="")
            label.pack()

        self.displayImagePointsButton.configure(text='Display Image Points', fg_color=GREEN, state='normal')
        self.saveToCache()
        self.openImagesButton.configure(command=self.openImageWindow, fg_color=GREEN, state='normal')

    def saveCal(self, button):
        if self.imageConfig.camCal.validCal:
            self.imageConfig.camCal.toFile(self.filepath)
            self.imageConfig.camCal.toBinFile(self.filepath)
            button.configure(fg_color='navy')
            return
        button.configure(fg_color='red')

    def openImageWindow(self):
        self.unpackAllFrames()
        self.geometry(self.imageWinGeometry)
        self.updateImageFrame()
        self.after(100, self.imageFrame.pack(fill='both', expand=True))

    def openCalWindow(self):
        self.unpackAllFrames()
        self.geometry(self.calGeometry)
        self.updateCalWindow()
        self.calFrame.pack(fill='both', expand=True)

    def openConfigWindow(self):
        self.unpackAllFrames()
        self.geometry(self.configGeometry)
        self.configFrame.pack()

    def returnToMain(self):
        self.unpackAllFrames()
        self.geometry(self.mainGeometry)
        self.mainFrame.pack()

    def updateCalWindow(self):
        # If we have a previous calibration
        if self.imageConfig.camCal.validCal:

            # Then display the calibration
            saveCalButton = ctk.CTkButton(master=self.calFrame, text='Save Calibration')
            saveCalButton.configure(command=lambda btn=saveCalButton: self.saveCal(btn))
            saveCalButton.grid(row=0, column=0, padx=0, pady=0)

            cal = ctk.CTkLabel(master=self.calFrame, text=self.imageConfig.camCal.calStr, justify='left', anchor='w')
            cal.grid(row=1, column=0, padx=0, pady=0)

            scale864Button = ctk.CTkButton(master=self.calFrame, text='Scale to 864x864', command=self.scaleTo864)
            scale864Button.grid(row=3, column=0, padx=5, pady=5)
            scale2848Button = ctk.CTkButton(master=self.calFrame, text='Scale to 2848x2848', command=self.scaleTo2848)
            scale2848Button.grid(row=4, column=0, padx=5, pady=5)
            scaleAnyButton = ctk.CTkButton(master=self.calFrame, text='Scale to Input Size', command=self.scaleToInput)
            scaleAnyButton.grid(row=5, column=0, padx=5, pady=5)

        backToMainButton = ctk.CTkButton(master=self.calFrame, text='Go back', command=self.returnToMain)
        backToMainButton.grid(row=6, column=0, padx=5, pady=5)

    def scaleTo864(self):
        self.imageConfig.camCal.scaleCalibration(864)
        self.saveToCache()
        self.updateCalWindow()

    def scaleTo2848(self):
        self.imageConfig.camCal.scaleCalibration(2848)
        self.saveToCache()
        self.updateCalWindow()

    def scaleToInput(self):
        dialog = ctk.CTkInputDialog(
            text='Input an integer value. The updated calibration width will be this value.',
            title='Calibration Scale Selection')
        self.imageConfig.camCal.scaleCalibration(int(dialog.get_input()))
        self.saveToCache()
        self.updateCalWindow()

    def updateConfigWindow(self):
        rowID = 0
        values = [1, 10, 100, 1000]
        stoppingIterationLabel = ctk.CTkLabel(master=self.configFrame, text='Max Iterations: ')
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
        self.stoppingMinStepSizeButton.after(1, self.update())
        self.stoppingMinStepSizeButton.after(500, self.restoreMinSizeButton())

    def restoreMinSizeButton(self):
        self.stoppingMinStepSizeButton.configure(fg_color=GREEN)

    def restoreIterationButton(self):
        self.stoppingIterationButton.configure(fg_color=GREEN)

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
        self.stoppingIterationButton.after(1, self.update())
        self.stoppingIterationButton.after(500, self.restoreIterationButton())

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
        selectAllButton = ctk.CTkButton(master=self.imageFrame, text='Include All Images',
                                        command=self.includeAll)
        selectAllButton.grid(row=rowID, column=0, padx=5, pady=5)

        removeUnselected = ctk.CTkButton(master=self.imageFrame, text='Remove Unselected', command=self.removeUnused)
        removeUnselected.grid(row=rowID, column=3, padx=5, pady=5, columnspan=2)

        self.displayImagePointsButton = ctk.CTkButton(master=self.imageFrame, text='Display All Chessboard Points',
                                                      command=self.displayImagePointsThread)
        self.displayImagePointsButton.grid(row=rowID, column=5, columnspan=2, padx=5, pady=5)

        backToMainButton = ctk.CTkButton(master=self.imageFrame, text='Go back', command=self.returnToMain)
        backToMainButton.grid(row=rowID, column=7, padx=5, pady=5)

        self.protectInvert()
        self.protectAllGrayscale()
        self.protectRotateCCW()
        self.protectRotateCW()

        self.imageConfigWindowObjects = []
        rowID = 0

        while len(self.imageConfigWindowObjects) < self.imageConfig.num_valid_imgs:
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

        while len(self.imageConfigWindowObjects) > len(self.imageConfig.img_collection):
            for item in self.imageConfigWindowObjects[-1]:
                item.grid_forget()
                item.destroy()
            self.imageConfigWindowObjects.pop(-1)
        while len(self.imageConfigWindowObjects) < len(self.imageConfig.img_collection):
            self.createNewRow(len(self.imageConfigWindowObjects))

        self.sortByResidual()

        rowID = 0
        for idx, imgClass in enumerate(self.imageConfig.img_collection):
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
                currRes = 'Res: ' + str(round(imgClass.residual, 3))

            imgRes.configure(text=currRes)

            if imgClass.sharpness is None:
                currShrp = ''
            else:
                currShrp = 'Shrp: ' + str(round(imgClass.sharpness, 3))

            imgShp.configure(text=currShrp)

            imgRestoreButton.configure(command=lambda imgC=imgClass: self.restore(imgC))

            imgFindCornersButton.configure(command=lambda imgC=imgClass: self.findChessboardCorners(imgC, True, True))

            imgInvertButton.configure(command=lambda imgC=imgClass: self.invertIndividualImage(imgC))

            imgGrayButton.configure(command=lambda imgC=imgClass: self.grayscaleIndividualImage(imgC))

            imgCCWRotateButton.configure(command=lambda imgC=imgClass: self.rotateCCWIndividualImage(imgC))

            imgCWRotateButton.configure(command=lambda imgC=imgClass: self.rotateCWIndividualImage(imgC))

            rowID += 1

    def removeUnused(self):
        if not os.path.exists(join(self.filepath, 'Removed')):
            os.makedirs(join(self.filepath, 'Removed'))
        removeIds = []
        for idx, imgClass in enumerate(self.imageConfig.img_collection):

            if not imgClass.include and os.path.exists(join(self.filepath, imgClass.imageName)):
                os.replace(join(self.filepath, imgClass.imageName), join(self.filepath, 'Removed', imgClass.imageName))

            if not imgClass.include:
                removeIds.append(idx)

        for id in reversed(removeIds):
            self.imageConfig.img_collection.pop(id)
        self.saveToCache()
        self.updateImageFrame()

        if len(self.imageConfig.img_collection) > 5:
            self.openImagesButton.configure(text=str(self.imageConfig.num_valid_imgs) + ' valid images', fg_color=GREEN)
            self.calibrateButton.configure(state="normal")
        else:
            self.openImagesButton.configure(text=str(self.imageConfig.num_valid_imgs) + ' valid images', fg_color="red")

    def copyToRemovedFolder(self, imgClass):
        if not os.path.exists(join(self.filepath, 'Removed')):
            os.makedirs(join(self.filepath, 'Removed'))

        src_path = join(self.filepath, imgClass.imageName)
        dst_path = join(self.filepath, 'Removed', imgClass.imageName)

        if not os.path.exists(dst_path):
            self.writeFile(src_path, dst_path)

    def restore(self, imgClass):
        if os.path.exists(join(self.filepath, 'Removed', imgClass.imageName)):
            src_path = join(self.filepath, 'Removed', imgClass.imageName)
            dst_path = join(self.filepath, imgClass.imageName)

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
        for idx, allGuiItems in enumerate(self.imageConfigWindowObjects):
            checkbox, *_ = allGuiItems
            if self.imageConfig.img_collection[idx].include:
                checkbox.select()
            else:
                checkbox.deselect()

    def unprotectInvert(self):
        self.imgInvertProtectedButton.configure(command=self.invertAll, fg_color=GREEN, hover_color='dark green')
        self.imgInvertProtectedButton.update()
        self.after(2000, self.protectInvert)

    def unprotectAllGrayscale(self):
        self.imgGrayProtectedButton.configure(command=self.grayscaleAll, fg_color=GREEN, hover_color='dark green')
        self.imgGrayProtectedButton.update()
        self.after(2000, self.protectAllGrayscale)

    def unprotectRotateCCW(self):
        self.imgRotateCCWProtectedButton.configure(command=self.rotateAllCCW, fg_color=GREEN, hover_color='dark green')
        self.imgRotateCCWProtectedButton.update()
        self.after(2000, self.protectRotateCCW)

    def unprotectRotateCW(self):
        self.imgRotateCWProtectedButton.configure(command=self.rotateAllCW, fg_color=GREEN, hover_color='dark green')
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
        self.imgRotateCCWProtectedButton.configure(fg_color='blue', hover_color='cyan4',
                                                   command=self.unprotectRotateCCW)
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
        for imgClass in self.imageConfig.img_collection:
            self.invertIndividualImage(imgClass)
        self.protectInvert()

    def grayscaleAll(self):
        self.imgGrayProtectedButton.configure(fg_color='black')
        self.imgGrayProtectedButton.update()
        for imgClass in self.imageConfig.img_collection:
            self.grayscaleIndividualImage(imgClass)
        self.protectAllGrayscale()

    def rotateAllCW(self):
        self.imgRotateCWProtectedButton.configure(fg_color='black')
        self.imgRotateCWProtectedButton.update()
        for imgClass in self.imageConfig.img_collection:
            self.rotateCWIndividualImage(imgClass)
        self.protectRotateCW()

    def rotateAllCCW(self):
        self.imgRotateCCWProtectedButton.configure(fg_color='black')
        self.imgRotateCCWProtectedButton.update()
        for imgClass in self.imageConfig.img_collection:
            self.rotateCCWIndividualImage(imgClass)
        self.protectRotateCCW()

    def invertIndividualImage(self, imgClass):
        filepath = join(self.filepath, imgClass.imageName)
        img = cv2.imread(filepath)
        invt_img = cv2.bitwise_not(img)
        cv2.imwrite(filepath, invt_img)

    def grayscaleIndividualImage(self, imgClass):
        self.copyToRemovedFolder(imgClass)
        filepath = join(self.filepath, imgClass.imageName)
        img = cv2.imread(filepath)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(filepath, gray_img)

    def rotateCWIndividualImage(self, imgClass):
        filepath = join(self.filepath, imgClass.imageName)
        img = cv2.imread(filepath)
        invt_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(filepath, invt_img)

    def rotateCCWIndividualImage(self, imgClass):
        filepath = join(self.filepath, imgClass.imageName)
        img = cv2.imread(filepath)
        invt_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(filepath, invt_img)

    def includeAll(self):
        for imgClass in self.imageConfig.img_collection:
            imgClass.include = True
        self.saveToCache()
        self.updateIncludeCheckboxes()

    def unprotectClearCache(self, button: ctk.CTkButton):
        button.grid_forget()
        clearCacheButton = ctk.CTkButton(master=self.mainFrame, text='Really Clear Cache', fg_color='green',
                                         command=self.clearCache)
        clearCacheButton.grid(row=self.clearCacheRow, column=0, columnspan=2, padx=5, pady=5)
        self.after(2000, self.protectClearCache)
        self.after(2000, clearCacheButton.grid_forget)

    def protectClearCache(self):
        clearCacheButton = ctk.CTkButton(master=self.mainFrame, text='Clear Cache', fg_color='blue', hover_color='navy')
        clearCacheButton.configure(command=lambda button=clearCacheButton: self.unprotectClearCache(button))
        clearCacheButton.grid(row=self.clearCacheRow, column=0, columnspan=2, padx=5, pady=5)

    def clearCache(self):
        if os.path.exists(join(self.filepath, 'imagery_cache.pkl')):
            filepath = copy.copy(self.filepath)
            os.remove(join(self.filepath, 'imagery_cache.pkl'))

            clearCacheButton = ctk.CTkButton(master=self.mainFrame, text='Clearing', fg_color='yellow',
                                             text_color='black', hover_color='yellow')
            clearCacheButton.grid(row=self.clearCacheRow, column=0, columnspan=2, padx=5, pady=5)

            self.imageConfig = ImageryCalibrationConfig()
            self.filepath = filepath
            self.restoreFromWindowState()
            self.loadImages()
            self.updateImageFrame()
        else:
            clearCacheButton = ctk.CTkButton(master=self.mainFrame, text='No cache!', fg_color='red', hover_color='red')
            clearCacheButton.grid(row=self.clearCacheRow, column=0, columnspan=2, padx=5, pady=5)
        self.after(2000, self.protectClearCache)

    def updateInclusion(self, idx):
        self.imageConfig.img_collection[idx].include = not self.imageConfig.img_collection[idx].include
        self.saveToCache()

    def restoreFromImageConfig(self):
        if self.imageConfig.invert_image:
            self.invertImagesCheckbox.select()
        else:
            self.invertImagesCheckbox.deselect()
        if self.imageConfig.fisheye:
            self.fisheyeLensCheckbox.select()
        else:
            self.fisheyeLensCheckbox.deselect()
        self.widthComboEntry.set(str(self.imageConfig.num_inner_corners_W))
        self.heightComboEntry.set(str(self.imageConfig.num_inner_corners_H))
        self.SUBwidthComboEntry.set(str(self.imageConfig.SUB_num_inner_corners_W))
        self.SUBheightComboEntry.set(str(self.imageConfig.SUB_num_inner_corners_H))
        self.folderLabel.configure(text="../" + os.path.basename(os.path.normpath(self.filepath)))
        self.selectImgTypeCombo.set(self.imageConfig.img_type)
        self.selectModeCombo.set(self.imageConfig.calMode.value)

    def restoreFromWindowState(self):
        self.imageConfig.invert_image = self.invertImagesCheckbox.get()
        self.imageConfig.num_inner_corners_W = int(self.widthComboEntry.get())
        self.imageConfig.num_inner_corners_H = int(self.heightComboEntry.get())
        self.imageConfig.SUB_num_inner_corners_W = int(self.SUBwidthComboEntry.get())
        self.imageConfig.SUB_num_inner_corners_H = int(self.SUBheightComboEntry.get())
        self.imageConfig.img_type = self.selectImgTypeCombo.get()
        self.imageConfig.calMode = CalibrationType(self.selectModeCombo.get())

    def saveToCache(self):

        if len(self.imageConfig.img_collection) > 0:
            with open(join(self.filepath, 'imagery_cache.pkl'), 'wb') as f:
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

        if os.path.exists(join(self.filepath, 'imagery_cache.pkl')):
            with open(join(self.filepath, 'imagery_cache.pkl'), 'rb') as imageConfigOpen:
                self.imageConfig.copy(pickle.load(imageConfigOpen))
        else:
            self.imageConfig = ImageryCalibrationConfig()
            if not self.imageConfig.camCal.fromFile(self.filepath):
                success = self.imageConfig.camCal.fromBinFile(self.filepath)

        self.restoreFromImageConfig()
        self.loadImages()

    def calibrate(self):
        self.calculating = True
        self.calibrateButton.configure(state='disabled', text='Calculating...', fg_color='gray')
        self.t1 = Thread(target=self.threadedCal)
        self.t1.start()

    def threadedCal(self):
        for imgClass in self.imageConfig.img_collection:
            if imgClass.include is True:
                self.findChessboardCorners(imgClass, False)
        self.calibrateCamera()
        self.saveToCache()
        self.updateImageFrame()
        self.calibrateButton.configure(state='normal', text='Calibrate', fg_color='green')
        self.calculating = False

    def widthInput(self, newVal):
        self.imageConfig.num_inner_corners_W = int(newVal)
        self.saveToCache()

    def heightInput(self, newVal):
        self.imageConfig.num_inner_corners_H = int(newVal)
        self.saveToCache()

    def SUBwidthInput(self, newVal):
        self.imageConfig.SUB_num_inner_corners_W = int(newVal)
        if self.imageConfig.SUB_num_inner_corners_W > self.imageConfig.num_inner_corners_W:
            self.imageConfig.SUB_num_inner_corners_W = self.imageConfig.num_inner_corners_W
        self.saveToCache()

    def SUBheightInput(self, newVal):
        self.imageConfig.SUB_num_inner_corners_H = int(newVal)
        if self.imageConfig.SUB_num_inner_corners_H > self.imageConfig.num_inner_corners_H:
            self.imageConfig.SUB_num_inner_corners_H = self.imageConfig.num_inner_corners_H
        self.saveToCache()

    def selectFolder(self):
        poss_filepath = filedialog.askopenfilename(initialdir=self.filepath + "/..", title="Select Imagery Folder")
        if poss_filepath == '':
            return

        self.filepath = os.path.dirname(poss_filepath)
        self.folderLabel.configure(text=os.path.basename(self.filepath))
        self.imageConfig.img_collection = []
        self.loadFromCache(False)
        self.saveToCache()

    def sortBySharpness(self):
        self.imageConfig.img_collection = sorted(self.imageConfig.img_collection,
                                                key=lambda img: self.sharpnessTest(img.sharpness))

    def sortByResidual(self):
        self.imageConfig.img_collection = sorted(self.imageConfig.img_collection,
                                                key=lambda img: self.sharpnessTest(img.residual))

    def sharpnessTest(self, sharpValue):
        if sharpValue is None:
            return 50.0
        else:
            return sharpValue

    def loadImages(self):
        imgs = glob.glob(join(self.filepath, '*.' + self.imageConfig.img_type))

        for img in imgs:
            img = os.path.basename(img)
            isAlreadyPresent = False
            for existingImgClass in self.imageConfig.img_collection:
                if img == existingImgClass.imageName:
                    isAlreadyPresent = True
            if not isAlreadyPresent:
                self.imageConfig.img_collection.append(ImageData(os.path.basename(img)))

        if len(self.imageConfig.img_collection) > 5:
            self.openImagesButton.configure(text=str(self.imageConfig.num_valid_imgs) + ' valid images', fg_color=GREEN)
            self.calibrateButton.configure(state="normal")
        else:
            self.openImagesButton.configure(text=str(self.imageConfig.num_valid_imgs) + ' valid images', fg_color="red")
            self.calibrateButton.configure(state="disabled")

        self.saveToCache()
        if not self.initImageFrame:
            self.setupImageFrame()
            self.initImageFrame = True
        self.updateImageFrame()

    def invertImageToggle(self):
        self.imageConfig.invert_image = not self.imageConfig.invert_image
        self.saveToCache()

    def toggleFisheye(self):
        self.imageConfig.fisheye = not self.imageConfig.fisheye
        self.saveToCache()

    def showBasicImage(self, imgClass):
        img = cv2.imread(join(self.filepath, imgClass.imageName))

        h, w, toss = img.shape
        if h > 1080 or w > 1080:
            self.scale = max(1080 / h, 1080 / w) * 1.1
            dispImg = cv2.resize(img, (int(w * self.scale), int(h * self.scale)))
        else:
            dispImg = copy.copy(img)

        cv2.namedWindow(imgClass.imageName)

        # kernel = np.array([[0, -1, 0],[-1, 5, -1], [0, -1, 0]])
        # sharpened_image = cv2.filter2D(dispImg, -1, kernel)
        #
        # gray = cv2.cvtColor(dispImg, cv2.COLOR_BGR2GRAY)
        # _laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        # _sharpness = _laplacian.var()
        #
        # _contrast = gray.std()
        #
        # _clarity = _sharpness * _contrast
        #
        # _sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        # _sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        # _sobel = np.sqrt(_sobel_x ** 2 + _sobel_y ** 2)
        # _resolution = np.mean(_sobel)

        # print(f'Laplacian: {_laplacian}')
        # print(f'Sharpness: {_sharpness}')
        # print(f'Contrast: {_contrast}')
        # print(f'Clarity: {_clarity}')
        # print(f'Resolution: {_resolution}')

        cv2.imshow(imgClass.imageName, dispImg)
        # cv2.imshow('Sharper? ', sharpened_image)

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
            self.firstClick = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON and self.firstClick is not None:
            cloned_img = copy.copy(self.currImg)
            cv2.rectangle(cloned_img, self.firstClick, (x, y), (0, 255, 0), 2)
            cv2.imshow(self.currImgClass.imageName, cloned_img)
        elif event == cv2.EVENT_LBUTTONUP and self.firstClick is not None:
            img = cv2.imread(join(self.filepath, self.currImgClass.imageName))
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

            new_img = copy.copy(img)
            new_img[:, :low_x] = np.zeros(new_img[:, :low_x].shape)
            new_img[:low_y] = np.zeros(new_img[:low_y].shape)
            new_img[:, high_x:] = np.zeros(new_img[:, high_x:].shape)
            new_img[high_y:] = np.zeros(new_img[high_y:].shape)

            cv2.imwrite(join(self.filepath, self.currImgClass.imageName), new_img)

            h, w, toss = new_img.shape
            dispImg = cv2.resize(new_img, (int(w * self.scale), int(h * self.scale)))

            cv2.imshow("New", dispImg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def findChessboardCorners(self, imgClass, showImage=True, updateImageFrame=False):

        # if imgClass.imgPts is not None:
        #     return  # Already have points for this image

        img = cv2.imread(join(self.filepath, imgClass.imageName))

        if img is None:
            imgClass.include = False
            return

        # arucoDict = cv2.aruco.DICT_5X5_1000
        # squaresVertically, squaresHorizontally = 12, 9
        # square_length = 30
        # marker_length = 15
        # margin = 20
        # params = cv2.aruco.DetectorParameters()
        # dict = cv2.aruco.getPredefinedDictionary(arucoDict)
        # detector = cv2.aruco.ArucoDetector(dict, params)
        # board = cv2.aruco.CharucoBoard((squaresVertically, squaresHorizontally), square_length, marker_length, dict)
        # charucodetector = cv2.aruco.CharucoDetector(board)
        # charuco_corners, charuco_ids, marker_corners, marker_ids = charucodetector.detectBoard(img)
        #
        # newImg = copy.copy(img)
        # newImg = cv2.aruco.drawDetectedCornersCharuco(newImg, charuco_corners, charuco_ids, (0,0,120))
        # cv2.imshow("hey hey hey!", newImg)
        # cv2.waitKey(0)

        objp = np.zeros((self.imageConfig.num_inner_corners_W * self.imageConfig.num_inner_corners_H, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.imageConfig.num_inner_corners_W, 0:self.imageConfig.num_inner_corners_H].T.reshape(-1,
                                                                                                                   2) * self.imageConfig.spacing

        if self.imageConfig.invert_image:
            temp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            inv_img = cv2.bitwise_not(temp)
            gray = inv_img
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if self.imageConfig.calMode == CalibrationType.Chessboard:
            ret, corners = cv2.findChessboardCorners(gray,
                                                     (self.imageConfig.num_inner_corners_W,
                                                      self.imageConfig.num_inner_corners_H),
                                                     flags=cv2.ADAPTIVE_THRESH_GAUSSIAN_C)

        elif self.imageConfig.calMode == CalibrationType.Circles:

            ret, corners = cv2.findCirclesGrid(gray,
                                               (self.imageConfig.num_inner_corners_W,
                                                self.imageConfig.num_inner_corners_H),
                                               flags=cv2.ADAPTIVE_THRESH_GAUSSIAN_C)  #, blobDetector=blob)
        else:
            ret = False
            print('Unknown Cal Mode')

        if ret == True:
            imgClass.objPts = objp

            criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, self.imageConfig.maxIter, self.imageConfig.minStepSize)
            corners2 = cv2.cornerSubPix(gray, np.float32(corners),
                                        (self.imageConfig.SUB_num_inner_corners_H, self.imageConfig.SUB_num_inner_corners_W),
                                        (-1, -1), criteria)
            imgClass.imgPts = corners2

            sharpness = cv2.estimateChessboardSharpness(gray, (
            self.imageConfig.num_inner_corners_W, self.imageConfig.num_inner_corners_H), np.float32(corners2))
            imgClass.sharpness = sharpness[0][0]
        else:
            imgClass.include = False
            self.openImagesButton.configure(text=str(len(self.imageConfig.img_collection)) + ' valid images')

        if updateImageFrame:
            self.updateImageFrame()

        self.saveToCache()

        if showImage:
            if imgClass.imgPts is not None:
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img,
                                                (self.imageConfig.num_inner_corners_W, self.imageConfig.num_inner_corners_H),
                                                imgClass.imgPts, True)

                min_X = max(int(np.min(imgClass.imgPts[:, 0, 0]) - 100), 0)
                max_X = min(int(np.max(imgClass.imgPts[:, 0, 0] + 100)), img.shape[1])
                min_Y = max(int(np.min(imgClass.imgPts[:, :, 1] - 100)), 0)
                max_Y = min(int(np.max(imgClass.imgPts[:, :, 1] + 100)), img.shape[0])

                roi = img[min_Y:max_Y, min_X:max_X, :]

                cv2.imshow('Chessboard Corners Detected', roi)
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
        for idx, imgClass in enumerate(self.imageConfig.img_collection):
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
        for idx, imgClass in enumerate(self.imageConfig.img_collection):
            if imgClass.include:
                images.append(self.fileName(idx))
                objPoints.append(imgClass.objPts)
                imgPoints.append(imgClass.imgPts)

        initialVector = np.eye(3, 3)
        initialVector[2, 2] = 0

        gray = cv2.cvtColor(cv2.imread(self.fileName(0)), cv2.COLOR_BGR2GRAY)

        if self.imageConfig.fisheye:
            K = np.array([[400.0, 0.0, 400.0],[0.0, 400.0, 400.0],[0.0, 0.0, 1.0]])
            D = np.zeros((4, 1))
            rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(objPoints))]
            tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(objPoints))]
            calValues = cv2.fisheye.calibrate(
                objectPoints=np.expand_dims(np.asarray(objPoints), -2),
                imagePoints=imgPoints,
                image_size=gray.shape[::-1],
                K=K,
                D=D,
                rvecs=rvecs,
                tvecs=tvecs,
                flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW,
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, self.imageConfig.maxIter,
                          self.imageConfig.minStepSize))
            ret = calValues[0]
            mtx = calValues[1]
            dist = np.squeeze(calValues[2])
        else:
            calValues = cv2.calibrateCameraROExtended(
                objPoints,
                imgPoints,
                gray.shape[::-1],
                1,
                initialVector,
                None,
                flags=(self.imageConfig.flags))
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
                self.imageConfig.img_collection[self.findIndexGivenImageName(os.path.basename(img))].residual = \
                residuals[img_idx][0]

            # Sort the images by their residual Values
            self.sortByResidual()

        self.endTime = time.time()

        fovx, fovy, focalLength, principalPoint, aspectRatio = cv2.calibrationMatrixValues(mtx, gray.shape[::-1], 25.,
                                                                                           25.)

        self.imageConfig.camCal.fisheye = self.imageConfig.fisheye
        self.imageConfig.camCal.setCameraMatrix(mtx=mtx)
        self.imageConfig.camCal.setDistortion(dist=dist.T)
        self.imageConfig.camCal.setAccessories(calTime=self.endTime - self.startTime, numCBUsed=len(images),
                                               width=gray.shape[::-1][0], height=gray.shape[::-1][1], hfov=fovx,
                                               rms=ret, timeOfCompute=datetime.datetime.now())


        self.updateCalWindow()
        self.displayCal.configure(state='normal')


if __name__ == '__main__':
    (gui := FrontEndGui()).mainloop()
