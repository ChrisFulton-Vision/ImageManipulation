import colorsys
import copy
import datetime
import glob
import os
import pickle
import sys
import time
import tkinter
from enum import Enum
from functools import partial
from os.path import join
from threading import Thread
from tkinter import filedialog

from customtkinter import (CTkFrame, CTkImage, CTkEntry, CTkButton, CTkLabel, CTkComboBox, CTkCheckBox, CTkInputDialog,
                           END, CTkToplevel)
import cv2
import numpy as np
from PIL.Image import open as pilOpen, fromarray

from support.vision.calibration import Calibration

sys.path.append(os.getcwd())
GREEN = '#2FA572'
FILEPATH_CACHE = 'Caches/filepath_cache.pkl'
IMAGE_CACHE = 'imagery_cache.pkl'  # Local to each folder structure, stored with imagery


class CalibrationType(Enum):
    Chessboard = 'Chessboard'
    Circles = 'Circles'
    chArUco = 'chArUco'


class ImageryCalibrationConfig:
    """
    This class stores information cleanly about any calibration that has occurred or is intended to occur. Because
    it stores all the information and settings for the calibration, this class is neatly packaged in a cache for
    easy reloading.
    """

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
        self.screen_based_checkerboard = False

    @property
    def num_valid_imgs(self):
        """
        Property method for CalConfig. Call as:
        calConfig = ImageryCalibrationConfig()
        num_imgs = calConfig.num_valid_imgs
        :return: Number of valid images, excluding those specifically not included in submenu or previous cache.
        Images that aren't included but are in the list are visible in submenus, but not included in calibration.
        """
        num_valid = 0
        for img in self.img_collection:
            if img.include:
                num_valid += 1
        return num_valid

    def copy(self, configToCopy):
        """
        Caching helper function. When reading from binary, copy all named dictionary items in the CalibrationConfig.
        Changes to this class will cause version errors when reading in old configs IFF names are changed or removed.
        Adding NEW parameters does not create a version error, but the parameter will not change through the load.
        :param configToCopy: Loaded value, typically from cache.
        :return:
        """
        for obj in configToCopy.__dict__:
            try:
                self.__dict__[obj] = configToCopy.__dict__[obj]
            except KeyError:
                # Allows for versioning issues, changed naming conventions.
                pass

    @property
    def flags(self):
        """
        Helper function that returns the composite flag value for a calibration based on own settings.
        :return: cv2-style flags for calibration.
        """
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


class ImageData:
    """
    This class stores information maintained by a single image. The image MUST have a name which is its filename.
    Include sets whether the image is part of the calibration.
    imgPts stores 2d identified features (such as chessboard corners)
    objPts stores 3d expected features (such as 3d coords for the chessboard)
    Residual characterizes the performance of the calibration. This is a good estimate for image quality.
    Sharpness characterizes the blurriness of the image. This is a rough estimate for image quality.
    """

    def __init__(self, name=''):
        self.imageName = name
        self.include = True
        self.imgPts = None
        self.objPts = None
        self.residual = None
        self.sharpness = None


class CalibrateGui(CTkFrame):
    """
    This is the main GUI that the user interacts with when running the program. This class manages the main loop,
    displays the main buttons, creates the submenus (but waits to show them until asked), and generally maintains
    system state.
    """

    def __init__(self, master, *args, **kwargs):
        # Super class init, necessary for customTkinter
        super().__init__(master, *args, **kwargs)

        # Stores calibration configuration states
        self.currImgClass = None
        self.imageConfig = ImageryCalibrationConfig()

        # Bool for calibration state
        self.calculating = False

        # Various helper variable NONE-initialization
        self.displayImagePointsButton = None
        self.currImg = None
        self.firstClick = None
        self.camera = None
        self.initImageFrame = False
        self.imageConfigWindowObjects = []
        self.scale = 1.0

        self.func_that_refits = None

        self.filepath = ''
        self.loadFromCache(True)

        # Empty thread-holding objects
        self.t1 = None
        self.t2 = None
        self.t3 = None

        ##########################################################################
        # Image Management Frame Setup

        self.imageFrame = None
        self.leftArrow = CTkImage(light_image=pilOpen('leftArrow.png'), size=(20, 20))
        self.rightArrow = CTkImage(light_image=pilOpen('rightArrow.png'), size=(20, 20))

        self.imgInvertProtectedButton = None
        self.imgRotateCCWProtectedButton = None
        self.imgRotateCWProtectedButton = None
        self.imgGrayProtectedButton = None

        self.imageFrame = None
        self._rows_holder = None
        self._page_start = 0
        self._page_size = 10
        self._page_label = None

        self.firstPageBtn = None
        self.prevPageBtn = None
        self.nextPageBtn = None
        self.lastPageBtn = None

        self.saveCalButton = None
        self.scale864Button = None
        self.scale2848Button = None
        self.scaleAnyButton = None
        self.calLabel = None

        ##########################################################################
        # Now Initialize the buttons on the main frame
        self.selectFolderButton = None
        self.availImagesLabel = None
        self.selectModeLabel = None
        self.selectModeCombo = None
        self.selectImgTypeLabel = None
        self.selectImgTypeCombo = None
        self.folderLabel = None
        self.invertImagesCheckbox = None
        self.fisheyeLensCheckbox = None
        self.cornerInputLabel = None
        self.widthLabel = self.heightLabel = self.widthComboEntry = self.heightComboEntry = None
        self.SUBcornerInputLabel = None
        self.SUBwidthLabel = self.SUBheightLabel = self.SUBwidthComboEntry = self.SUBheightComboEntry = None
        self.calibrateButton = None
        self.displayCal = None

        # # Once a calibration is active, allow user to display a window that manages the calibration
        self.displayCal = None

        # # Once a calibration is active, allow user to display a window that manages the calibration
        self.configWindowButton = None

        self._ui_active = True
        self._last_ui_tick = 0.0
        self._ui_throttle_sec = 0.10  # repaint at most every 100ms

    def set_ui_active(self, active: bool):
        self._ui_active = bool(active)
        # Stop/avoid background refreshers when hidden (threads/after loops).
        # If you have repeating after() callbacks, guard their reschedule on this flag.

    def func_to_refit(self, func):
        self.func_that_refits = func

    def on_section_show(self, name: str):
        if self.func_that_refits:
            self.func_that_refits()

    def setup_configFrame(self, master_frame):
        f = CTkFrame(master_frame)
        rowID = 0

        f.grid_rowconfigure([0, 1, 2], weight=1)  # configure grid system
        f.grid_columnconfigure([0, 1, 2], weight=1)

        self.selectFolderButton = CTkButton(master=f, text='Select Folder', command=self.selectFolder,
                                            fg_color="navy")
        self.selectFolderButton.grid(row=rowID, column=0, padx=5, pady=5, sticky="ew")

        self.availImagesLabel = CTkLabel(master=f, text='Not Selected', fg_color="black")
        self.availImagesLabel.grid(row=rowID, column=1, padx=5, pady=5, sticky="ew")

        rowID += 1
        self.selectModeLabel = CTkLabel(master=f, text='Calibration Type')
        self.selectModeLabel.grid(row=rowID, column=0, padx=5, pady=5, sticky='nsew')
        modes = [calType.value for calType in CalibrationType]
        self.selectModeCombo = CTkComboBox(master=f, values=modes, command=self.updateMode)
        self.selectModeCombo.grid(row=rowID, column=1, padx=5, pady=5, sticky='ew')

        rowID += 1
        imgTypes = ['png', 'bmp', 'img', 'jpg']
        self.selectImgTypeLabel = CTkLabel(master=f, text='File Type')
        self.selectImgTypeLabel.grid(row=rowID, column=0, padx=5, pady=5, sticky='nsew')
        self.selectImgTypeCombo = CTkComboBox(master=f, values=imgTypes, command=self.updateImgType)
        self.selectImgTypeCombo.grid(row=rowID, column=1, padx=5, pady=5, sticky='ew')

        rowID += 1
        folderPathLabel = CTkLabel(master=f,
                                   text="Filepath:")
        folderPathLabel.grid(row=rowID, column=0, sticky='nsw')
        self.folderLabel = CTkLabel(master=f,
                                    text="../" + os.path.basename(os.path.normpath(self.filepath)))
        self.folderLabel.grid(row=rowID, column=1, columnspan=2, sticky='nsw')

        rowID += 1
        self.invertImagesCheckbox = CTkCheckBox(master=f, text='Invert Image? (LWIR)',
                                                command=self.invertImageToggle)
        if self.imageConfig.invert_image:
            self.invertImagesCheckbox.select()
        self.invertImagesCheckbox.grid(row=rowID, columnspan=1, column=0, padx=5, pady=5, sticky='nsw')

        self.fisheyeLensCheckbox = CTkCheckBox(master=f, text='Fisheye Lens?',
                                               command=self.toggleFisheye)
        if self.imageConfig.fisheye:
            self.fisheyeLensCheckbox.select()
        self.fisheyeLensCheckbox.grid(row=rowID, columnspan=1, column=1, padx=5, pady=5, sticky='nsw')
        rowID += 1

        values = [str(num) for num in range(5, 21)]
        self.cornerInputLabel = CTkLabel(f, text='# of Inner CB Corners')
        self.cornerInputLabel.grid(row=rowID, column=0, columnspan=2, padx=5, pady=5)
        rowID += 1

        self.widthLabel = CTkLabel(f, text='Width')
        self.widthLabel.grid(row=rowID, column=0, padx=5, pady=5)
        self.heightLabel = CTkLabel(f, text='Height')
        self.heightLabel.grid(row=rowID, column=1, padx=5, pady=5)
        self.widthComboEntry = CTkComboBox(master=f,
                                           values=values,
                                           command=self.widthInput)
        self.widthComboEntry.grid(row=rowID, column=0, padx=5, pady=5)
        self.heightComboEntry = CTkComboBox(master=f,
                                            values=values,
                                            command=self.heightInput)
        self.heightComboEntry.grid(row=rowID, column=1, padx=5, pady=5)
        rowID += 1

        self.SUBcornerInputLabel = CTkLabel(f, text='# for sub-Pixel Search')
        self.SUBcornerInputLabel.grid(row=rowID, column=0, columnspan=2, padx=5, pady=5)
        rowID += 1

        self.SUBwidthLabel = CTkLabel(f, text='Width')
        self.SUBwidthLabel.grid(row=rowID, column=0, padx=5, pady=5)
        self.SUBheightLabel = CTkLabel(f, text='Height')
        self.SUBheightLabel.grid(row=rowID, column=1, padx=5, pady=5)

        self.SUBwidthComboEntry = CTkComboBox(master=f,
                                              values=values,
                                              command=self.SUBwidthInput)
        self.SUBwidthComboEntry.grid(row=rowID, column=0, padx=5, pady=5)

        self.SUBheightComboEntry = CTkComboBox(master=f,
                                               values=values,
                                               command=self.SUBheightInput)
        self.SUBheightComboEntry.grid(row=rowID, column=1, padx=5, pady=5)
        rowID += 1

        self.calibrateButton = CTkButton(f, text="Calibrate!", state="disabled",
                                         command=lambda fr=master_frame: self.calibrate(fr))
        self.calibrateButton.grid(row=rowID, column=0, columnspan=1, padx=5, pady=5)

        self.protectClearCache(f, rowID)

        self.loadFromCache()

        rowID += 1

        return f

    def updateMode(self, newMode):
        """
        Setter from button click
        """
        self.imageConfig.calMode = CalibrationType(newMode)
        self.saveToCache()

    def updateImgType(self, newImgType):
        """
        Setter from button click
        """
        self.imageConfig.img_type = newImgType
        self.imageConfig.img_collection = []
        self.loadImages()

    def fileName(self, idx):
        """
        :param idx: the number of the image in the collection
        :return: the filename with root filepath prepended
        """
        return join(self.filepath, self.imageConfig.img_collection[idx].imageName)

    def displayImagePointsThread(self, master_frame):
        """
        This button spawns a thread that looks at all the existing chessboard corners (see displayImagePoints()).
        While the thread is spawned, the image menu is unavailable (as it is being consistently updated).
        """
        self.displayImagePointsButton.configure(text='Calculating', fg_color='gray', state='disabled')
        self.availImagesLabel.configure(fg_color='gray', state='disabled')
        self.t3 = Thread(target=lambda f=master_frame: self.displayImagePoints(f), daemon=True)
        self.t3.start()

    def displayImagePoints(self, master_frame):
        """
        This function examines all of the active image chessboard results and plots them to a graph.
        This is useful if the user wants to see what regions have already been included in the image. If a
        calibration is complete, then it also color-codes the images using HSV to highlight high-performing and
        low-performing images
        """

        # Must assume same height for each image
        img = cv2.imread(self.fileName(0))
        width = img.shape[1]
        height = img.shape[0]

        for imgClass in self.imageConfig.img_collection:
            if imgClass.include:
                self.findChessboardCorners(master_frame, imgClass, showImage=False)

        self.sortForCuration()
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

        #  Examine each image's chessboard solution. If the image has an associated residual, color code the image.
        for idx, imageClass in enumerate(self.imageConfig.img_collection):

            if imageClass.include and imageClass.imgPts is not None:
                gotAtLeastOneImage = True

                if imageClass.residual is not None:
                    residual = imageClass.residual

                for imgPt in imageClass.imgPts:
                    b, g, r = colorsys.hsv_to_rgb(0.4 - 0.4 * (residual - minVal) / (maxVal - minVal), 1.0, 1.0)
                    cv2.circle(blank_image,
                               (round(imgPt[0][0]), round(imgPt[0][1])), 2,
                               (int(255 * b), int(255 * g), int(255 * r)), 2)

        self.updateImageFrame(master_frame)

        # If we succeeded at at least one image, then display the image of all of the found corners
        if gotAtLeastOneImage:
            # Draw Legend
            b, g, r = colorsys.hsv_to_rgb(0, 1.0, 1.0)
            cv2.circle(blank_image, (5, 20), 2, (int(255 * b), int(255 * g), int(255 * r)), 2)
            cv2.putText(blank_image, 'Residual of: ' + str(round(maxVal, 2)), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255))
            b, g, r = colorsys.hsv_to_rgb(0.4, 1.0, 1.0)
            cv2.circle(blank_image, (5, 40), 2, (int(255 * b), int(255 * g), int(255 * r)), 2)
            cv2.putText(blank_image,
                        'Residual of: ' + str(round(minVal, 2)),
                        (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255))
            cv2.resize(blank_image, (1000, int(1000 / width * height)))

            # Convert the openCV object to an Image object, which is ingested by customtkinter
            filledImage = fromarray(blank_image)

            ctkImage = CTkImage(filledImage, size=(1000, int(1000 / width * height)))
            imagePointWindow = CTkToplevel(self)
            imagePointWindow.geometry('1000x' + str(int(1000 / width * height)))
            imagePointWindow.title("Chessboard Corner Locations")
            label = CTkLabel(master=imagePointWindow, image=ctkImage, text="")
            label.pack()

        self.displayImagePointsButton.configure(text='Display Image Points', fg_color=GREEN, state='normal')
        self.saveToCache()
        self.availImagesLabel.configure(fg_color=GREEN, state='normal')

    def saveCal(self, button):
        if self.imageConfig.camCal.validCal:
            self.imageConfig.camCal.toFile(self.filepath)
            self.imageConfig.camCal.toBinFile(self.filepath)
            button.configure(fg_color='navy')
            return
        button.configure(fg_color='red')

    def setup_CalFrame(self, master_frame):

        f = CTkFrame(master_frame)
        # If we have a previous calibration
        if self.saveCalButton is None:
            self.saveCalButton = CTkButton(master=f, text='Save Calibration')
            self.saveCalButton.configure(command=lambda btn=self.saveCalButton: self.saveCal(btn))
            self.saveCalButton.grid(row=0, column=0, padx=5, pady=5)

        if self.scale864Button is None:
            self.scale864Button = CTkButton(master=f, text='Scale to 864x864', command=lambda: self.scaleTo864(f))
            self.scale864Button.grid(row=3, column=0, padx=5, pady=5)

        if self.scale2848Button is None:
            self.scale2848Button = CTkButton(master=f, text='Scale to 2848x2848', command=lambda: self.scaleTo2848(f))
            self.scale2848Button.grid(row=4, column=0, padx=5, pady=5)

        if self.scaleAnyButton is None:
            self.scaleAnyButton = CTkButton(master=f, text='Scale to Input Size', command=lambda: self.scaleToInput(f))
            self.scaleAnyButton.grid(row=5, column=0, padx=5, pady=5)

        if self.imageConfig.camCal.validCal:
            # Then display the calibration
            if self.calLabel is None:
                self.calLabel = CTkLabel(master=f, text=self.imageConfig.camCal.calStr, justify='center', anchor='w')
                self.calLabel.grid(row=1, column=0, padx=5, pady=5)
            else:
                self.calLabel.configure(text=self.imageConfig.camCal.calStr)
        else:
            if self.calLabel is None:
                self.calLabel = CTkLabel(f, text="No calibration calculated yet.", justify='center')
                self.calLabel.grid(row=1, column=0, padx=5, pady=5)
            else:
                self.calLabel.configure(text="No calibration available.")

            self.saveCalButton.configure(state='disabled')
            self.scale864Button.configure(state='disabled')
            self.scale2848Button.configure(state='disabled')
            self.scaleAnyButton.configure(state='disabled')

        return f

    def scaleTo864(self, master_frame):
        self.imageConfig.camCal.scaleCalibration(864)
        self.saveToCache()
        self.updateConfigWindow(master_frame)
        self.setup_CalFrame(master_frame)

    def scaleTo2848(self, master_frame):
        self.imageConfig.camCal.scaleCalibration(2848)
        self.saveToCache()
        self.updateConfigWindow(master_frame)
        self.setup_CalFrame(master_frame)

    def scaleToInput(self, master_frame):
        dialog = CTkInputDialog(
            text='Input an integer value. The updated calibration width will be this value.',
            title='Calibration Scale Selection')
        try:
            self.imageConfig.camCal.scaleCalibration(int(dialog.get_input()))
            self.saveToCache()
            self.updateConfigWindow(master_frame)
            self.setup_CalFrame(master_frame)
        except ValueError:
            from support.io.my_logging import LOG
            LOG.warning('Invalid input. Please input only an integer.')

    def updateConfigWindow(self, master_frame):
        rowID = 0
        f = CTkFrame(master_frame)

        stoppingIterationLabel = CTkLabel(master=f, text='Max Iterations: ')
        stoppingIterationLabel.grid(row=rowID, column=0, padx=5, pady=5)
        stoppingIterationEntry = CTkEntry(master=f, placeholder_text=str(self.imageConfig.maxIter))
        stoppingIterationEntry.bind('<Return>', lambda event, x=stoppingIterationEntry: self.stoppingCritIterUpdate(x))

        stoppingIterationEntry.grid(row=rowID, column=1, padx=5, pady=5)
        rowID += 1

        stoppingMinStepSizeLabel = CTkLabel(master=f, text='Stopping Min Step Size: ')
        stoppingMinStepSizeLabel.grid(row=rowID, column=0, padx=5, pady=5)

        stoppingMinStepSizeEntry = CTkEntry(master=f, placeholder_text=str(self.imageConfig.minStepSize))
        stoppingMinStepSizeEntry.bind('<Return>',
                                      lambda event, x=stoppingMinStepSizeEntry:
                                      self.stoppingCritMinStepSizeUpdate(x))
        stoppingMinStepSizeEntry.grid(row=rowID, column=1, padx=5, pady=5)
        rowID += 1

        fixPrincipalPointCB = CTkCheckBox(master=f, text='Fix Principle Point', checkbox_height=20)
        if self.imageConfig.fixPrincipalPoint:
            fixPrincipalPointCB.select()
        else:
            fixPrincipalPointCB.deselect()
        fixPrincipalPointCB.configure(command=self.toggleFixPrincipalPoint)
        fixPrincipalPointCB.grid(row=rowID, column=0, columnspan=2, padx=0, pady=0, sticky='nsw')
        rowID += 1

        fixAspectRatioCB = CTkCheckBox(master=f, text='Fix Aspect Ratio', checkbox_height=20)
        if self.imageConfig.fixAspectRatio:
            fixAspectRatioCB.select()
        else:
            fixAspectRatioCB.deselect()
        fixAspectRatioCB.configure(command=self.toggleFixAspectRatio)
        fixAspectRatioCB.grid(row=rowID, column=0, columnspan=2, padx=0, pady=0, sticky='nsw')
        rowID += 1

        zeroTangentDistCB = CTkCheckBox(master=f, text='Zero Tangent Distance', checkbox_height=20)
        if self.imageConfig.zeroTangentDist:
            zeroTangentDistCB.select()
        else:
            zeroTangentDistCB.deselect()
        zeroTangentDistCB.configure(command=self.toggleZeroTangentDist)
        zeroTangentDistCB.grid(row=rowID, column=0, columnspan=2, padx=0, pady=0, sticky='nsw')
        rowID += 1

        screen_based_checkerboardCB = CTkCheckBox(master=f, text="Screen Based Checkerboard", checkbox_height=20)
        if self.imageConfig.screen_based_checkerboard:
            screen_based_checkerboardCB.select()
        else:
            screen_based_checkerboardCB.deselect()
        screen_based_checkerboardCB.configure(command=self.toggleScreenbasedCheckerboard)
        screen_based_checkerboardCB.grid(row=rowID, column=0, columnspan=2, padx=0, pady=0, sticky='nsw')

        return f

    def stoppingCritMinStepSizeUpdate(self, entry=None):
        try:
            newStep = float(entry.get())
        except ValueError:
            entry.delete(0, END)
            entry.insert(0, str(self.imageConfig.maxIter))
            return

        if isinstance(newStep, float) and newStep > 0:
            self.imageConfig.minStepSize = newStep
        else:
            entry.delete(0, END)
            entry.insert(0, str(self.imageConfig.minStepSize))
        self.saveToCache()
        entry.configure(fg_color='yellow')
        entry.after(1, self.update_idletasks())
        entry.after(500, entry.configure(fg_color='green'))

    def stoppingCritIterUpdate(self, entry=None):
        try:
            newIter = int(entry.get())
        except ValueError:
            entry.delete(0, END)
            entry.insert(0, str(self.imageConfig.maxIter))
            return
        self.imageConfig.maxIter = newIter
        self.saveToCache()
        entry.configure(fg_color='yellow')
        entry.after(1, self.update_idletasks())
        entry.after(500, entry.configure(fg_color=GREEN))

    def toggleZeroTangentDist(self):
        self.imageConfig.zeroTangentDist = not self.imageConfig.zeroTangentDist

    def toggleScreenbasedCheckerboard(self):
        self.imageConfig.screen_based_checkerboard = not self.imageConfig.screen_based_checkerboard

    def toggleFixPrincipalPoint(self):
        self.imageConfig.fixPrincipalPoint = not self.imageConfig.fixPrincipalPoint

    def toggleFixAspectRatio(self):
        self.imageConfig.fixAspectRatio = not self.imageConfig.fixAspectRatio

    def _render_row(self, idx, widgets, f):
        """Retitle + rebind one row from item idx (no new widgets)."""
        (imgIncludeCheckbox,
         imgNameButton,
         imgRes,
         imgShp,
         imgRestoreButton,
         imgFindCornersButton,
         imgInvertButton,
         imgGrayButton,
         imgCCWRotateButton,
         imgCWRotateButton) = widgets

        imgClass = self.imageConfig.img_collection[idx]

        # checkbox state
        (imgIncludeCheckbox.select() if imgClass.include else imgIncludeCheckbox.deselect())
        imgIncludeCheckbox.configure(command=partial(self.updateInclusion, idx),
                                     state="normal")

        # labels
        if imgClass.residual is None:
            currRes = ''
        elif imgClass.residual == 10000.0:
            currRes = 'Disabled'
        else:
            currRes = f'Res: {round(imgClass.residual, 3)}'
        imgRes.configure(text=currRes)

        currShrp = '' if imgClass.sharpness is None else f'Shrp: {round(imgClass.sharpness, 3)}'
        imgShp.configure(text=currShrp)

        # commands
        imgNameButton.configure(text=imgClass.imageName,
                                command=partial(self.showBasicImage, imgClass), state="normal")
        imgRestoreButton.configure(command=partial(self.restore, imgClass), state="normal")
        imgFindCornersButton.configure(command=partial(self.findChessboardCorners, f, imgClass, True, True),
                                       state="normal")
        imgInvertButton.configure(command=partial(self.invertIndividualImage, imgClass), state="normal")
        imgGrayButton.configure(command=partial(self.grayscaleIndividualImage, imgClass), state="normal")
        imgCCWRotateButton.configure(command=partial(self.rotateCCWIndividualImage, imgClass), state="normal")
        imgCWRotateButton.configure(command=partial(self.rotateCWIndividualImage, imgClass), state="normal")

    def _page_bounds(self):
        total = len(self.imageConfig.img_collection)
        start = max(0, min(self._page_start, max(0, total - 1)))
        end = min(total, start + self._page_size)
        return start, end, total

    def _refresh_all_rows(self, f):
        start, end, total = self._page_bounds()
        needed = end - start
        while len(self.imageConfigWindowObjects) < needed:
            self.createNewRow(self._rows_holder, len(self.imageConfigWindowObjects))

        for i in range(needed):
            idx = start + i
            widgets = self.imageConfigWindowObjects[i]
            self._render_row(idx, widgets, f)
            for w in widgets:
                try:
                    w.grid()
                except tkinter.TclError:
                    pass

        for i in range(needed, len(self.imageConfigWindowObjects)):
            for w in self.imageConfigWindowObjects[i]:
                try:
                    w.grid_remove()
                except tkinter.TclError:
                    pass

        self._update_page_label_and_buttons()

    def setup_imageFrame(self, master_frame):

        f = CTkFrame(master_frame)  # <— plain frame
        f.grid_rowconfigure(0, weight=0)  # header
        f.grid_rowconfigure(1, weight=1)  # rows
        f.grid_columnconfigure(0, weight=1)

        # Header container (row 0)
        header = CTkFrame(f, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(list(range(12)), weight=0)  # plenty of columns

        # Rows container (row 1)
        self._rows_holder = CTkFrame(f, fg_color="transparent")
        self._rows_holder.grid(row=1, column=0, sticky="nsew")
        self._rows_holder.grid_columnconfigure(list(range(10)), weight=1)

        # --- put all header buttons in `header` (NOT in f) ---
        rowID = 0
        selectAllButton = CTkButton(master=header, text='Include All', command=self.includeAll)
        selectAllButton.grid(row=rowID, column=0, padx=5, pady=5)

        removeUnselected = CTkButton(master=header, text='Remove Unselected',
                                     command=lambda fr=master_frame: self.removeUnused(fr))
        removeUnselected.grid(row=rowID, column=1, padx=5, pady=5, columnspan=2)

        self.displayImagePointsButton = CTkButton(master=header, text='Display All Chessboard Points',
                                                  command=lambda fr=master_frame: self.displayImagePointsThread(fr))
        self.displayImagePointsButton.grid(row=rowID, column=3, columnspan=2, padx=5, pady=5)

        self.imgInvertProtectedButton = CTkButton(master=header, text='Invert All', hover_color='navy',
                                                  fg_color='blue', width=100, command=self.unprotectInvert)
        self.imgRotateCCWProtectedButton = CTkButton(master=header, image=self.leftArrow, text='All',
                                                     hover_color='navy', fg_color='blue', width=100,
                                                     command=self.unprotectRotateCCW)
        self.imgRotateCWProtectedButton = CTkButton(master=header, image=self.rightArrow, text='All',
                                                    hover_color='navy', fg_color='blue', width=100,
                                                    command=self.unprotectRotateCW)
        self.imgGrayProtectedButton = CTkButton(master=header, text='Grayscale All', hover_color='navy',
                                                fg_color='blue', width=100, command=self.unprotectAllGrayscale)

        self.protectInvert(row=0)
        self.protectRotateCCW(row=0)
        self.protectRotateCW(row=0)
        self.protectAllGrayscale(row=0)

        # Pager controls in the header
        self._page_label = CTkLabel(header, text="1/1")
        self._page_label.grid(row=rowID, column=12, padx=6, pady=6, sticky="e")

        self.firstPageBtn = CTkButton(header, text="◀◀", command=self._first_page, width=70)
        self.prevPageBtn = CTkButton(header, text="◀ Prev", command=self._page_prev, width=70)
        self.nextPageBtn = CTkButton(header, text="Next ▶", command=self._page_next, width=70)
        self.lastPageBtn = CTkButton(header, text="▶▶", command=self._last_page, width=70)
        self.firstPageBtn.grid(row=rowID, column=10, padx=6, pady=6, sticky='w')
        self.prevPageBtn.grid(row=rowID, column=11, padx=6, pady=6, sticky="w")
        self.nextPageBtn.grid(row=rowID, column=13, padx=6, pady=6, sticky="w")
        self.lastPageBtn.grid(row=rowID, column=14, padx=6, pady=6, sticky='w')

        # fresh paging state
        self._page_start = 0
        self.imageConfigWindowObjects = []
        for child in list(self._rows_holder.winfo_children()):
            child.destroy()

        self.imageFrame = f
        self._refresh_all_rows(f)  # render only current page
        self.updateImageFrame(f)
        return f

    def createNewRow(self, f, rowID):
        imgIncludeCheckbox = CTkCheckBox(master=f, text='')
        imgIncludeCheckbox.grid(row=rowID, column=0)

        imgNameButton = CTkButton(master=f, text='')
        imgNameButton.grid(row=rowID, column=1, padx=5, pady=5)

        imgRes = CTkLabel(master=f, text='')
        imgRes.grid(row=rowID, column=2, padx=5, pady=5)

        imgShp = CTkLabel(master=f, text='')
        imgShp.grid(row=rowID, column=3, padx=5, pady=5)

        imgRestoreButton = CTkButton(master=f, text='Restore')
        imgRestoreButton.grid(row=rowID, column=4, padx=5, pady=5)

        imgFindCornersButton = CTkButton(master=f, text='Find Corners')
        imgFindCornersButton.grid(row=rowID, column=5, padx=5, pady=5)

        imgInvertButton = CTkButton(master=f, text='Invert Image')
        imgInvertButton.grid(row=rowID, column=6, padx=5, pady=5)

        imgGrayButton = CTkButton(master=f, text='Grayscale Image')
        imgGrayButton.grid(row=rowID, column=7, padx=5, pady=5)

        imgCCWRotateButton = CTkButton(master=f, text='', image=self.leftArrow)
        imgCCWRotateButton.grid(row=rowID, column=8, padx=5, pady=5)

        imgCWRotateButton = CTkButton(master=f, text='', image=self.rightArrow)
        imgCWRotateButton.grid(row=rowID, column=9, padx=5, pady=5)

        self.imageConfigWindowObjects.append([imgIncludeCheckbox, imgNameButton, imgRes, imgShp, imgRestoreButton,
                                              imgFindCornersButton, imgInvertButton, imgGrayButton,
                                              imgCCWRotateButton, imgCWRotateButton])

    def _update_page_label_and_buttons(self):
        total = len(self.imageConfig.img_collection)
        pages = max(1, (total + self._page_size - 1) // self._page_size)
        curr = min(pages, (self._page_start // self._page_size) + 1)
        if self._page_label:
            self._page_label.configure(text=f"{curr}/{pages}")

        # enable/disable pager buttons safely
        # self.firstPageBtn.configure(state=("normal" if curr > 1 else "disabled"))
        # self.prevPageBtn.configure(state=("normal" if curr > 1 else "disabled"))
        # self.nextPageBtn.configure(state=("normal" if curr < pages else "disabled"))
        # self.lastPageBtn.configure(state=("normal" if curr < pages else "disabled"))

    def _first_page(self):
        if self._page_start == 0:
            self._last_page()
            return
        self._page_start = 0
        self.updateImageFrame()

    def _page_prev(self):
        if self._page_start == 0:
            self._last_page()
            return
        self._page_start = max(0, self._page_start - self._page_size)
        self.updateImageFrame()

    def _page_next(self):
        _, end, total = self._page_bounds()
        if end < total:
            self._page_start += self._page_size
        else:
            self._page_start = 0
        self.updateImageFrame()

    def _last_page(self):
        total = len(self.imageConfig.img_collection)
        if total <= 0:
            self._page_start = 0
            self.updateImageFrame()
            return

        pages = max(1, (total + self._page_size - 1) // self._page_size)
        last_start = (pages - 1) * self._page_size  # start index of the last page

        if self._page_start == last_start:
            self._page_start = 0
        else:
            self._page_start = last_start

        self.updateImageFrame()

    def updateImageFrame(self, f=None):
        f = f or self.imageFrame
        if not f:
            return
        self._refresh_all_rows(f)

    def removeUnused(self, master_frame):
        if not os.path.exists(join(self.filepath, 'Removed')):
            os.makedirs(join(self.filepath, 'Removed'))
        removeIds = []
        for idx, imgClass in enumerate(self.imageConfig.img_collection):

            if not imgClass.include and os.path.exists(join(self.filepath, imgClass.imageName)):
                os.replace(join(self.filepath, imgClass.imageName), join(self.filepath, 'Removed', imgClass.imageName))

            if not imgClass.include:
                removeIds.append(idx)

        for idx in reversed(removeIds):
            self.imageConfig.img_collection.pop(idx)
        self.saveToCache()
        self.updateImageFrame(master_frame)

        if len(self.imageConfig.img_collection) > 5:
            self.availImagesLabel.configure(text=f'{self.imageConfig.num_valid_imgs} valid images', fg_color='blue')
            self.calibrateButton.configure(state="normal")
        else:
            self.availImagesLabel.configure(text=f'{self.imageConfig.num_valid_imgs} valid images', fg_color="red")

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

    @staticmethod
    def writeFile(src_path, dst_path):
        if os.path.exists(src_path):
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
        self.imgInvertProtectedButton.update_idletasks()
        self.after(2000, self.protectInvert, 1)

    def unprotectAllGrayscale(self):
        self.imgGrayProtectedButton.configure(command=self.grayscaleAll, fg_color=GREEN, hover_color='dark green')
        self.imgGrayProtectedButton.update_idletasks()
        self.after(2000, self.protectAllGrayscale, 1)

    def unprotectRotateCCW(self):
        self.imgRotateCCWProtectedButton.configure(command=self.rotateAllCCW, fg_color=GREEN, hover_color='dark green')
        self.imgRotateCCWProtectedButton.update_idletasks()
        self.after(2000, self.protectRotateCCW, 1)

    def unprotectRotateCW(self):
        self.imgRotateCWProtectedButton.configure(command=self.rotateAllCW, fg_color=GREEN, hover_color='dark green')
        self.imgRotateCWProtectedButton.update_idletasks()
        self.after(2000, self.protectRotateCW, 1)

    def protectInvert(self, row=1):
        self.imgInvertProtectedButton.configure(fg_color='blue', hover_color='cyan4', command=self.unprotectInvert)
        if not self.imgInvertProtectedButton.winfo_ismapped():
            self.imgInvertProtectedButton.grid(row=row, column=6, padx=5, pady=5, sticky='nsew')

    def protectAllGrayscale(self, row=1):
        self.imgGrayProtectedButton.configure(fg_color='blue', hover_color='cyan4', command=self.unprotectAllGrayscale)
        if not self.imgGrayProtectedButton.winfo_ismapped():
            self.imgGrayProtectedButton.grid(row=row, column=7, padx=5, pady=5, sticky='nsew')

    def protectRotateCCW(self, row=1):
        self.imgRotateCCWProtectedButton.configure(fg_color='blue', hover_color='cyan4',
                                                   command=self.unprotectRotateCCW)
        if not self.imgRotateCCWProtectedButton.winfo_ismapped():
            self.imgRotateCCWProtectedButton.grid(row=row, column=8, padx=5, pady=5, sticky='nsew')

    def protectRotateCW(self, row=1):
        self.imgRotateCWProtectedButton.configure(fg_color='blue', hover_color='cyan4',
                                                  command=self.unprotectRotateCW)
        if not self.imgRotateCWProtectedButton.winfo_ismapped():
            self.imgRotateCWProtectedButton.grid(row=row, column=9, padx=5, pady=5, sticky='nsew')

    def invertAll(self):
        self.imgInvertProtectedButton.configure(fg_color='black')
        self.imgInvertProtectedButton.update_idletasks()
        for imgClass in self.imageConfig.img_collection:
            self.invertIndividualImage(imgClass)
        self.protectInvert()

    def grayscaleAll(self):
        self.imgGrayProtectedButton.configure(fg_color='black')
        self.imgGrayProtectedButton.update_idletasks()
        for imgClass in self.imageConfig.img_collection:
            self.grayscaleIndividualImage(imgClass)
        self.protectAllGrayscale()

    def rotateAllCW(self):
        self.imgRotateCWProtectedButton.configure(fg_color='black')
        self.imgRotateCWProtectedButton.update_idletasks()
        for imgClass in self.imageConfig.img_collection:
            self.rotateCWIndividualImage(imgClass)
        self.protectRotateCW()

    def rotateAllCCW(self):
        self.imgRotateCCWProtectedButton.configure(fg_color='black')
        self.imgRotateCCWProtectedButton.update_idletasks()
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

    def unprotectClearCache(self, master_frame, button: CTkButton, rowID):
        button.grid_forget()
        clearCacheButton = CTkButton(master=master_frame, text='Really Clear Cache', fg_color='green',
                                     command=lambda f=master_frame: self.clearCache(f, rowID))
        clearCacheButton.grid(row=rowID, column=1, columnspan=1, padx=5, pady=5)
        self.after(2000, self.protectClearCache, master_frame, rowID)
        self.after(2000, clearCacheButton.grid_forget)

    def protectClearCache(self, master_frame, rowID):
        clearCacheButton = CTkButton(master=master_frame, text='Clear Cache', fg_color='blue', hover_color='navy')
        clearCacheButton.configure(
            command=lambda button=clearCacheButton, f=master_frame, rid=rowID: self.unprotectClearCache(f, button, rid))
        clearCacheButton.grid(row=rowID, column=1, columnspan=1, padx=5, pady=5)

    def clearCache(self, master_frame, rowID):
        if os.path.exists(join(self.filepath, IMAGE_CACHE)):
            filepath = copy.copy(self.filepath)
            os.remove(join(self.filepath, IMAGE_CACHE))

            clearCacheButton = CTkButton(master=master_frame, text='Clearing', fg_color='yellow',
                                         text_color='black', hover_color='yellow')
            clearCacheButton.grid(row=rowID, column=0, columnspan=2, padx=5, pady=5)
            self.imageConfig.camCal = Calibration()
            self.imageConfig = ImageryCalibrationConfig()
            self.filepath = filepath
            self.restoreFromWindowState()
            self.loadImages()
            self.updateImageFrame(master_frame)
            self.calLabel.configure(text=self.imageConfig.camCal.calStr)
        else:
            clearCacheButton = CTkButton(master=master_frame, text='No cache!', fg_color='red', hover_color='red')
            clearCacheButton.grid(row=rowID, column=0, columnspan=2, padx=5, pady=5)
        self.after(2000, self.protectClearCache, master_frame, rowID)

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
            with open(join(self.filepath, IMAGE_CACHE), 'wb') as f:
                pickle.dump(self.imageConfig, f)

        if not os.path.exists(os.path.dirname(FILEPATH_CACHE)):
            os.makedirs(os.path.dirname(FILEPATH_CACHE))

        with open(FILEPATH_CACHE, 'wb') as f:
            pickle.dump(self.filepath, f)

    def loadFromCache(self, init=False):
        if init:
            if not os.path.exists(FILEPATH_CACHE):
                os.makedirs(os.path.dirname(FILEPATH_CACHE))
                self.filepath = os.getcwd()
                return

            with open(FILEPATH_CACHE, 'rb') as filepathOpen:
                self.filepath = pickle.load(filepathOpen)
                return

        if os.path.exists(join(self.filepath, IMAGE_CACHE)):
            try:
                with open(join(self.filepath, IMAGE_CACHE), 'rb') as imageConfigOpen:
                    self.imageConfig.copy(pickle.load(imageConfigOpen))
            except ModuleNotFoundError:
                pass
        else:
            self.imageConfig = ImageryCalibrationConfig()
            if not self.imageConfig.camCal.fromFile(self.filepath):
                success = self.imageConfig.camCal.fromBinFile(self.filepath)

        self.restoreFromImageConfig()
        self.loadImages()

    def calibrate_buttonCallback(self, master_frame, btn: CTkButton):
        self.calculating = True
        self.calibrateButton.configure(state='disabled', text='Calibrating...', fg_color='gray')
        self.t1 = Thread(target=lambda f=master_frame: self.threadedCalWithButtonCallback(f, btn))
        self.t1.start()

    def threadedCalWithButtonCallback(self, master_frame, btn: CTkButton):
        for imgClass in self.imageConfig.img_collection:
            if imgClass.include is True:
                self.findChessboardCorners(master_frame, imgClass, False)
        self.calibrateCamera()
        self.saveToCache()
        self.updateImageFrame(master_frame)
        self.calibrateButton.configure(state='normal', text='Calibrate', fg_color=GREEN)
        self.calculating = False
        btn.configure(text='Start Calibration', state='normal')
        self.update()

    def calibrate(self, master_frame):
        self.calculating = True
        self.calibrateButton.configure(state='disabled', text='Calibrating...', fg_color='gray')
        self.t1 = Thread(target=lambda f=master_frame: self.threadedCal(f))
        self.t1.start()

    def threadedCal(self, master_frame):
        for imgClass in self.imageConfig.img_collection:
            if imgClass.include is True:
                self.findChessboardCorners(master_frame, imgClass, False)
        self.calibrateCamera()
        self.saveToCache()
        self.updateImageFrame(master_frame)
        self.calibrateButton.configure(state='normal', text='Calibrate', fg_color=GREEN)
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
        self.updateImageFrame()

    def sortForCuration(self):
        """
        Sort priority:
          1) residual (if any exist)
          2) sharpness (if any exist)
          3) imageName (fallback)
        """
        imgs = self.imageConfig.img_collection

        has_residual = any((img.include and img.residual is not None) for img in imgs)
        has_sharpness = any((img.include and img.sharpness is not None) for img in imgs)

        if has_residual:
            # smaller residuals first; None goes to the bottom via sharpnessTest(None)->50.0
            self.imageConfig.img_collection = sorted(
                imgs,
                key=lambda img: (self.sharpnessTest(img.residual), img.imageName.lower())
            )
        elif has_sharpness:
            # keep your existing behavior (whatever “sharpness” direction you intended)
            self.imageConfig.img_collection = sorted(
                imgs,
                key=lambda img: (self.sharpnessTest(img.sharpness), img.imageName.lower())
            )
        else:
            self.imageConfig.img_collection = sorted(imgs, key=lambda img: img.imageName.lower())

    def sortBySharpness(self):
        self.imageConfig.img_collection = sorted(self.imageConfig.img_collection,
                                                 key=lambda img: self.sharpnessTest(img.sharpness))

    def sortByResidual(self):
        self.imageConfig.img_collection = sorted(self.imageConfig.img_collection,
                                                 key=lambda img: self.sharpnessTest(img.residual))

    @staticmethod
    def sharpnessTest(sharpValue):
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
            self.availImagesLabel.configure(text=f'{self.imageConfig.num_valid_imgs} valid images', fg_color='blue')
            self.calibrateButton.configure(state="normal")
        else:
            self.availImagesLabel.configure(text=f'{self.imageConfig.num_valid_imgs} valid images', fg_color="red")
            self.calibrateButton.configure(state="disabled")

        self.saveToCache()
        if not self.initImageFrame:
            self.initImageFrame = True

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

        cv2.namedWindow(imgClass.imageName, cv2.WINDOW_NORMAL)

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

    def findChessboardCorners(self, master_frame, imgClass, showImage=True, updateImageFrame=False):

        if imgClass.imgPts is not None and not showImage:
            return  # Already have points for this image

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
        objp[:, :2] = np.mgrid[0:self.imageConfig.num_inner_corners_W,
        0:self.imageConfig.num_inner_corners_H].T.reshape(-1, 2) * self.imageConfig.spacing

        if self.imageConfig.invert_image:
            temp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            inv_img = cv2.bitwise_not(temp)
            gray = inv_img
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if imgClass.imgPts is not None and showImage:
            self.drawImagePoints(imgClass, img, gray)
            return

        if self.imageConfig.calMode == CalibrationType.Chessboard:
            if self.imageConfig.screen_based_checkerboard:
                flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
                ret, corners = cv2.findChessboardCornersSB(gray,
                                                           (self.imageConfig.num_inner_corners_W,
                                                            self.imageConfig.num_inner_corners_H),
                                                           flags)
            else:
                ret, corners = cv2.findChessboardCorners(gray,
                                                         (self.imageConfig.num_inner_corners_W,
                                                          self.imageConfig.num_inner_corners_H))

        elif self.imageConfig.calMode == CalibrationType.Circles:

            ret, corners = cv2.findCirclesGrid(gray,
                                               (self.imageConfig.num_inner_corners_W,
                                                self.imageConfig.num_inner_corners_H),
                                               flags=cv2.ADAPTIVE_THRESH_GAUSSIAN_C)  #, blobDetector=blob)
        else:
            ret = False
            print('Unknown Cal Mode')

        if ret:
            imgClass.objPts = objp

            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, self.imageConfig.maxIter,
                self.imageConfig.minStepSize)
            corners2 = cv2.cornerSubPix(gray, np.float32(corners),
                                        (self.imageConfig.SUB_num_inner_corners_H,
                                         self.imageConfig.SUB_num_inner_corners_W),
                                        (-1, -1), criteria)
            imgClass.imgPts = corners2

            sharpness = cv2.estimateChessboardSharpness(gray, (
                self.imageConfig.num_inner_corners_W, self.imageConfig.num_inner_corners_H), np.float32(corners2))
            imgClass.sharpness = sharpness[0][0]
        else:
            imgClass.include = False
            self.availImagesLabel.configure(text=str(len(self.imageConfig.img_collection)) + ' valid images')

        if updateImageFrame:
            self.updateImageFrame(master_frame)

        self.saveToCache()

        if showImage:
            self.drawImagePoints(imgClass, img, gray)

    def drawImagePoints(self, imgClass, img, gray):
        if imgClass.imgPts is None:
            imgClass.include = False
            h, w = gray.shape
            dispImg = cv2.resize(gray, (int(w * self.scale), int(h * self.scale)))
            cv2.imshow('NO CHESSBOARD CORNERS FOUND', dispImg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return

        # Compute ROI in original image coordinates
        min_X = max(int(np.min(imgClass.imgPts[:, 0, 0]) - 100), 0)
        max_X = min(int(np.max(imgClass.imgPts[:, 0, 0]) + 100), img.shape[1])
        min_Y = max(int(np.min(imgClass.imgPts[:, 0, 1]) - 100), 0)
        max_Y = min(int(np.max(imgClass.imgPts[:, 0, 1]) + 100), img.shape[0])

        roi_base = img[min_Y:max_Y, min_X:max_X].copy()

        # Shift points into ROI coordinates
        pts_roi = imgClass.imgPts.copy()
        pts_roi[:, 0, 0] -= min_X
        pts_roi[:, 0, 1] -= min_Y

        pattern_size = (self.imageConfig.num_inner_corners_W,
                        self.imageConfig.num_inner_corners_H)

        resid_roi, proj_roi, Hmat = self.chessboard_point_residuals_homography(pts_roi, pattern_size)

        self.show_with_locked_aspect_redraw(
            'Chessboard Corners Detected',
            roi_base,
            pts_roi,
            pattern_size,
            resid_roi=resid_roi,  # <-- new
            proj_roi=proj_roi  # <-- optional (can draw predicted too)
        )

    @staticmethod
    def show_with_locked_aspect_redraw(name, base_img, pts_roi, pattern_size, *, resid_roi=None, proj_roi=None):
        if base_img.ndim == 2:
            base_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)

        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        last_w_win, last_h_win = None, None

        # Precompute some stats for coloring/thresholding
        if resid_roi is not None and len(resid_roi) > 0:
            resid_roi = np.asarray(resid_roi, dtype=np.float32).reshape(-1)
            # Use a robust max so 1 crazy point doesn't wash out the colormap
            r_med = float(np.median(resid_roi))
            r_mad = float(np.median(np.abs(resid_roi - r_med)) + 1e-12)
            r_lo = 0.0
            r_hi = max(float(np.percentile(resid_roi, 95)), r_med + 6.0 * 1.4826 * r_mad)
        else:
            r_lo, r_hi = 0.0, 1.0

        while True:
            if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
                return

            try:
                x, y, w_win, h_win = cv2.getWindowImageRect(name)
            except cv2.error:
                break

            if w_win <= 0 or h_win <= 0:
                key = cv2.waitKey(30)
                if key == 27:
                    break
                continue

            if last_w_win == w_win and last_h_win == h_win:
                key = cv2.waitKey(30)
                if key == 27:
                    break
                continue
            last_w_win, last_h_win = w_win, h_win

            h_img, w_img = base_img.shape[:2]
            s = min(w_win / w_img, h_win / h_img)
            new_w = max(1, int(round(w_img * s)))
            new_h = max(1, int(round(h_img * s)))

            resized = cv2.resize(base_img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            # Scale observed points
            pts_scaled = pts_roi.copy().astype(np.float32)
            pts_scaled[:, 0, 0] *= s
            pts_scaled[:, 0, 1] *= s
            pts2 = pts_scaled.reshape(-1, 2)

            # Optionally scale projected points too (for debug)
            if proj_roi is not None:
                proj2 = (np.asarray(proj_roi, dtype=np.float32) * s).reshape(-1, 2)
            else:
                proj2 = None

            # Draw points colored by residual
            if resid_roi is None:
                # fallback: just draw green points
                for (u, v) in pts2:
                    cv2.circle(resized, (int(round(u)), int(round(v))), 3, (0, 255, 0), 2)
            else:
                # Identify worst K
                K = min(10, len(resid_roi))
                worst_idx = np.argsort(-resid_roi)[:K]

                for k, (u, v) in enumerate(pts2):
                    e = float(resid_roi[k])
                    t = (e - r_lo) / (r_hi - r_lo + 1e-12)
                    t = float(np.clip(t, 0.0, 1.0))

                    # Hue: green -> red (0.33 -> 0.0)
                    r, g, b = colorsys.hsv_to_rgb(0.33 * (1.0 - t), 1.0, 1.0)
                    color = (int(255 * b), int(255 * g), int(255 * r))

                    # Slightly bigger marker for bad points
                    rad = 3 + int(round(4 * t))
                    thick = 2 + int(round(2 * t))
                    cv2.circle(resized, (int(round(u)), int(round(v))), rad, color, thick)

                # Annotate worst points with index + error
                for k in worst_idx:
                    u, v = pts2[k]
                    txt = f"{k}:{resid_roi[k]:.1f}px"
                    cv2.putText(resized, txt, (int(round(u)) + 6, int(round(v)) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(resized, txt, (int(round(u)) + 6, int(round(v)) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

                # Optional: draw predicted points as tiny crosses
                if proj2 is not None:
                    for (u, v) in proj2:
                        u = int(round(u));
                        v = int(round(v))
                        cv2.line(resized, (u - 3, v), (u + 3, v), (255, 255, 255), 1)
                        cv2.line(resized, (u, v - 3), (u, v + 3), (255, 255, 255), 1)

                # Summary text
                mean_e = float(np.mean(resid_roi))
                max_e = float(np.max(resid_roi))
                cv2.putText(resized, f"per-point residuals: mean {mean_e:.2f}px  max {max_e:.2f}px",
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(resized, f"per-point residuals: mean {mean_e:.2f}px  max {max_e:.2f}px",
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

            # Letterbox canvas
            canvas = np.zeros((h_win, w_win, 3), dtype=np.uint8)
            x_off = (w_win - new_w) // 2
            y_off = (h_win - new_h) // 2
            canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

            cv2.imshow(name, canvas)

            key = cv2.waitKey(30)
            if key == 27:
                break

        if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
            return
        cv2.destroyWindow(name)

    @staticmethod
    def chessboard_point_residuals_homography(imgPts, pattern_size):
        """
        imgPts: (N,1,2) float32/64 in ROI coordinates
        pattern_size: (W, H) inner corners
        Returns:
            resid_px: (N,) reprojection residual in pixels
            proj: (N,2) predicted points from homography
            H: 3x3 homography
        """
        W, H = pattern_size
        N = W * H
        pts = np.asarray(imgPts, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] != N:
            return None, None, None

        # Ideal grid coordinates in chessboard index space
        # (0..W-1, 0..H-1)
        obj2d = np.array([(i, j) for j in range(H) for i in range(W)], dtype=np.float32)

        # Robust homography (helps if a few corners are bad)
        Hmat, inliers = cv2.findHomography(obj2d, pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if Hmat is None:
            # fall back to least squares
            Hmat, _ = cv2.findHomography(obj2d, pts, method=0)

        # Project ideal grid through H -> predicted pixel locations
        proj = cv2.perspectiveTransform(obj2d.reshape(-1, 1, 2), Hmat).reshape(-1, 2)

        resid = np.linalg.norm(pts - proj, axis=1)  # pixels
        return resid, proj, Hmat

    def findIndexGivenImageName(self, name):
        sol_idx = None
        for idx, imgClass in enumerate(self.imageConfig.img_collection):
            if imgClass.imageName == name:
                sol_idx = idx
        return sol_idx

    def calibrateCamera(self):
        """
        Calibrate a camera and output calibration parameters and residuals.

        Inputs:
        folder - relative path to folder of images (this path should be relative to the working directory).
        Images MUST be .jpg
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
        """

        startTime = time.time()

        images = []
        objPoints = []
        imgPoints = []
        for idx, imgClass in enumerate(self.imageConfig.img_collection):
            if imgClass.include:
                images.append(self.fileName(idx))
                objPoints.append(imgClass.objPts)
                imgPoints.append(imgClass.imgPts)

        gray = cv2.cvtColor(cv2.imread(self.fileName(0)), cv2.COLOR_BGR2GRAY)

        if self.imageConfig.fisheye:
            def _as_fisheye_object_points(objPts):
                out = []
                for P in objPts:
                    P = np.asarray(P, dtype=np.float64)
                    # accept (N,3) or (N,1,3)
                    if P.ndim == 2 and P.shape[1] == 3:
                        P = P.reshape(-1, 1, 3)
                    elif P.ndim == 3 and P.shape[1:] == (1, 3):
                        pass
                    else:
                        raise ValueError(f"objectPoints must be (N,3) or (N,1,3), got {P.shape}")
                    out.append(P)
                return out

            def _as_fisheye_image_points(imgPts):
                out = []
                for p in imgPts:
                    p = np.asarray(p, dtype=np.float64)
                    # accept (N,2) or (N,1,2)
                    if p.ndim == 2 and p.shape[1] == 2:
                        p = p.reshape(-1, 1, 2)
                    elif p.ndim == 3 and p.shape[1:] == (1, 2):
                        pass
                    else:
                        raise ValueError(f"imagePoints must be (N,2) or (N,1,2), got {p.shape}")
                    out.append(p)
                return out

            K = np.array([[400.0, 0.0, 400.0],
                          [0.0, 400.0, 400.0],
                          [0.0, 0.0, 1.0]], dtype=np.float64)
            D = np.zeros((4, 1), dtype=np.float64)

            objp = _as_fisheye_object_points(objPoints)
            imgp = _as_fisheye_image_points(imgPoints)

            # You can pass empty lists; OpenCV will fill them.
            rvecs, tvecs = [], []

            rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                objectPoints=objp,
                imagePoints=imgp,
                image_size=gray.shape[::-1],
                K=K,
                D=D,
                rvecs=rvecs,
                tvecs=tvecs,
                flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW,
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                          self.imageConfig.maxIter,
                          self.imageConfig.minStepSize)
            )

            ret = rms
            mtx = K
            dist = D.reshape(-1)  # 4,
        else:
            K0 = cv2.initCameraMatrix2D(objPoints, imgPoints, gray.shape[::-1], 0)

            flags = (self.imageConfig.flags or 0) | cv2.CALIB_USE_INTRINSIC_GUESS

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        self.imageConfig.maxIter,  # e.g. 30–50 is usually enough
                        self.imageConfig.minStepSize)  # e.g. 1e-6..1e-5

            calValues = cv2.calibrateCameraROExtended(
                objectPoints=objPoints,
                imagePoints=imgPoints,
                imageSize=gray.shape[::-1],
                iFixedPoint=1,
                cameraMatrix=K0,
                distCoeffs=None,
                flags=flags,
                criteria=criteria
            )
            ret = calValues[0]
            mtx = calValues[1]
            dist = calValues[2]
            # rvecs = calValues[3]
            # tvecs = calValues[4]
            # newObjPoints = calValues[5]
            # stdDevIntrinsics = calValues[6]
            # stdDevExtrinsics = calValues[7]
            # stdDevObjPoints = calValues[8]
            residuals = calValues[9]

            for img_idx, img in enumerate(images):
                self.imageConfig.img_collection[self.findIndexGivenImageName(os.path.basename(img))].residual = \
                    residuals[img_idx][0]

            # Sort the images by their residual Values
            self.sortByResidual()

        endTime = time.time()

        fovx, fovy, focalLength, principalPoint, aspectRatio = cv2.calibrationMatrixValues(mtx, gray.shape[::-1], 25.,
                                                                                           25.)

        self.imageConfig.camCal.fisheye = self.imageConfig.fisheye
        self.imageConfig.camCal.setCameraMatrix(mtx=mtx)
        self.imageConfig.camCal.setDistortion(dist=dist.T)
        self.imageConfig.camCal.setAccessories(calTime=endTime - startTime, numCBUsed=len(images),
                                               width=gray.shape[::-1][0], height=gray.shape[::-1][1], hfov=fovx,
                                               rms=ret, timeOfCompute=datetime.datetime.now())

        self.calLabel.configure(text=self.imageConfig.camCal.calStr)

        self.saveCalButton.configure(state='normal')
        self.scale864Button.configure(state='normal')
        self.scale2848Button.configure(state='normal')
        self.scaleAnyButton.configure(state='normal')

        self.notify()

    @staticmethod
    def notify():

        # Try winsound
        try:
            import winsound
            def play_beep():
                winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS)
                time.sleep(0.2)  # allow sound to fully complete before closing thread

            t1 = Thread(target=play_beep, daemon=True)
            t1.start()
            return

        except ImportError:
            print("\a", end="", flush=True)
