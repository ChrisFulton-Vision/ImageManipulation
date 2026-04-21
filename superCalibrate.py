import colorsys
import copy
import datetime
import glob
import os
import pickle
import random
import sys
import time
import tkinter
from enum import Enum
from functools import partial
from os.path import join
from threading import Thread
from tkinter import filedialog, messagebox

from customtkinter import (CTkFrame, CTkImage, CTkEntry, CTkButton, CTkLabel, CTkComboBox, CTkCheckBox, CTkInputDialog,
                           END, CTkToplevel, CTkRadioButton)
import cv2
import numpy as np
from PIL.Image import open as pil_open, fromarray

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
        self.num_inner_corners_w = 8
        self.num_inner_corners_h = 11
        self.sub_num_inner_corners_w = 5
        self.sub_num_inner_corners_h = 5
        self.cal_mode = CalibrationType.Chessboard
        self.spacing = 30.0
        self.max_iter = 100
        self.min_step_size = 0.00001
        self.cam_cal = Calibration()
        self.zero_tangent_dist = True
        self.fix_aspect_ratio = True
        self.fix_principal_point = True
        self.fisheye = False
        self.screen_based_checkerboard = False

    LEGACY_ATTRS = {
        'num_inner_corners_W': 'num_inner_corners_w',
        'num_inner_corners_H': 'num_inner_corners_h',
        'SUB_num_inner_corners_W': 'sub_num_inner_corners_w',
        'SUB_num_inner_corners_H': 'sub_num_inner_corners_h',
        'calMode': 'cal_mode',
        'maxIter': 'max_iter',
        'minStepSize': 'min_step_size',
        'camCal': 'cam_cal',
        'zeroTangentDist': 'zero_tangent_dist',
        'fixAspectRatio': 'fix_aspect_ratio',
        'fixPrincipalPoint': 'fix_principal_point',
    }

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

    def copy(self, config_to_copy):
        """
        Caching helper function. When reading from binary, copy all named dictionary items in the CalibrationConfig.
        Changes to this class will cause version errors when reading in old configs IFF names are changed or removed.
        Adding NEW parameters does not create a version error, but the parameter will not change through the load.
        :param config_to_copy: Loaded value, typically from cache.
        :return:
        """
        for obj, value in config_to_copy.__dict__.items():
            self.__dict__[self.LEGACY_ATTRS.get(obj, obj)] = value
        self.normalize_legacy_attrs()

    def normalize_legacy_attrs(self):
        for old_name, new_name in self.LEGACY_ATTRS.items():
            if hasattr(self, old_name) and not hasattr(self, new_name):
                setattr(self, new_name, getattr(self, old_name))
        for img in self.img_collection:
            if hasattr(img, 'normalize_legacy_attrs'):
                img.normalize_legacy_attrs()
            else:
                ImageData.normalize_object_legacy_attrs(img)

    @property
    def flags(self):
        """
        Helper function that returns the composite flag value for a calibration based on own settings.
        :return: cv2-style flags for calibration.
        """
        flags = None
        if self.zero_tangent_dist:
            flags = cv2.CALIB_ZERO_TANGENT_DIST

        if self.fix_aspect_ratio:
            if flags is not None:
                flags += cv2.CALIB_FIX_ASPECT_RATIO
            else:
                flags = cv2.CALIB_FIX_ASPECT_RATIO

        if self.fix_principal_point:
            if flags is not None:
                flags += cv2.CALIB_FIX_PRINCIPAL_POINT
            else:
                flags = cv2.CALIB_FIX_PRINCIPAL_POINT

        return flags


class ImageData:
    """
    This class stores information maintained by a single image. The image MUST have a name which is its filename.
    Include sets whether the image is part of the calibration.
    img_pts stores 2d identified features (such as chessboard corners)
    obj_pts stores 3d expected features (such as 3d coords for the chessboard)
    Residual characterizes the performance of the calibration. This is a good estimate for image quality.
    Sharpness characterizes the blurriness of the image. This is a rough estimate for image quality.
    """

    def __init__(self, name=''):
        self.image_name = name
        self.include = True
        self.img_pts = None
        self.obj_pts = None
        self.residual = None
        self.sharpness = None
        self.points_edited = False

    def normalize_legacy_attrs(self):
        self.normalize_object_legacy_attrs(self)

    @staticmethod
    def normalize_object_legacy_attrs(obj):
        legacy_attrs = {
            'imageName': 'image_name',
            'imgPts': 'img_pts',
            'objPts': 'obj_pts',
        }
        for old_name, new_name in legacy_attrs.items():
            if hasattr(obj, old_name) and not hasattr(obj, new_name):
                setattr(obj, new_name, getattr(obj, old_name))


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
        self.curr_img_class = None
        self.image_config = ImageryCalibrationConfig()

        # Bool for calibration state
        self.calculating = False

        # Various helper variable NONE-initialization
        self.display_image_points_button = None
        self.curr_img = None
        self.first_click = None
        self.camera = None
        self.init_image_frame = False
        self.image_config_window_objects = []
        self.scale = 1.0

        self.func_that_refits = None
        self.on_calibration_complete = None

        self.filepath = ''
        self.loadFromCache(True)

        # Empty thread-holding objects
        self.t1 = None
        self.t2 = None
        self.t3 = None

        ##########################################################################
        # Image Management Frame Setup

        self.image_frame = None
        self.left_arrow = CTkImage(light_image=pil_open('leftArrow.png'), size=(20, 20))
        self.right_arrow = CTkImage(light_image=pil_open('rightArrow.png'), size=(20, 20))
        self.arrow_button_size = 32
        self.header_button_min_width = 32
        self.header_button_padding = 24
        self.header_button_char_width = 7
        self.header_button_image_width = 20
        self.display_image_points_label = 'Display All'

        self.img_invert_protected_button = None
        self.img_rotate_ccw_protected_button = None
        self.img_rotate_cw_protected_button = None
        self.img_gray_protected_button = None

        self.image_frame = None
        self._rows_holder = None
        self._page_start = 0
        self._page_size = 10
        self._page_label = None

        self.first_page_btn = None
        self.prev_page_btn = None
        self.next_page_btn = None
        self.last_page_btn = None

        self.cal_frame = None
        self.save_cal_button = None
        self.open_folder_button = None
        self.scale864_button = None
        self.scale2848_button = None
        self.scale_any_button = None
        self.cal_label = None

        ##########################################################################
        # Now Initialize the buttons on the main frame
        self.select_folder_button = None
        self.avail_images_label = None
        self.select_mode_label = None
        self.select_mode_combo = None
        self.select_img_type_label = None
        self.select_img_type_combo = None
        self.folder_label = None
        self.invert_images_checkbox = None
        self.fisheye_lens_checkbox = None
        self.corner_input_label = None
        self.width_label = self.height_label = self.width_combo_entry = self.height_combo_entry = None
        self.sub_corner_input_label = None
        self.sub_width_label = self.sub_height_label = self.sub_width_combo_entry = self.sub_height_combo_entry = None
        self.calibrate_button = None
        self.clear_cache_button = None
        self.display_cal = None

        self.display_cal = None

        self.config_window_button = None

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

    def header_button_width(self, text, has_image=False):
        width = self.header_button_padding + len(text) * self.header_button_char_width
        if has_image:
            width += self.header_button_image_width
        return max(self.header_button_min_width, width)

    def setup_configFrame(self, master_frame):
        f = CTkFrame(master_frame)
        row_id = 0

        f.grid_rowconfigure([0, 1, 2], weight=1)  # configure grid system
        f.grid_columnconfigure([0, 1, 2], weight=1)

        self.select_folder_button = CTkButton(master=f, text='Select Folder', command=self.selectFolder,
                                              fg_color="navy")
        self.select_folder_button.grid(row=row_id, column=0, padx=5, pady=5, sticky="ew")

        self.avail_images_label = CTkLabel(master=f, text='Not Selected', fg_color="black")
        self.avail_images_label.grid(row=row_id, column=1, padx=5, pady=5, sticky="ew")

        row_id += 1
        self.select_mode_label = CTkLabel(master=f, text='Calibration Type')
        self.select_mode_label.grid(row=row_id, column=0, padx=5, pady=5, sticky='nsew')
        modes = [cal_type.value for cal_type in CalibrationType]
        self.select_mode_combo = CTkComboBox(master=f, values=modes, command=self.updateMode)
        self.select_mode_combo.grid(row=row_id, column=1, padx=5, pady=5, sticky='ew')

        row_id += 1
        img_types = ['png', 'bmp', 'img', 'jpg']
        self.select_img_type_label = CTkLabel(master=f, text='File Type')
        self.select_img_type_label.grid(row=row_id, column=0, padx=5, pady=5, sticky='nsew')
        self.select_img_type_combo = CTkComboBox(master=f, values=img_types, command=self.updateImgType)
        self.select_img_type_combo.grid(row=row_id, column=1, padx=5, pady=5, sticky='ew')

        row_id += 1
        folder_path_label = CTkLabel(master=f,
                                     text="Filepath:")
        folder_path_label.grid(row=row_id, column=0, sticky='nsw')
        self.folder_label = CTkLabel(master=f,
                                     text="../" + os.path.basename(os.path.normpath(self.filepath)))
        self.folder_label.grid(row=row_id, column=1, columnspan=2, sticky='nsw')

        row_id += 1
        self.invert_images_checkbox = CTkCheckBox(master=f, text='Invert Image? (LWIR)',
                                                  command=self.invertImageToggle)
        if self.image_config.invert_image:
            self.invert_images_checkbox.select()
        self.invert_images_checkbox.grid(row=row_id, columnspan=1, column=0, padx=5, pady=5, sticky='nsw')

        self.fisheye_lens_checkbox = CTkCheckBox(master=f, text='Fisheye Lens?',
                                                 command=self.toggleFisheye)
        if self.image_config.fisheye:
            self.fisheye_lens_checkbox.select()
        self.fisheye_lens_checkbox.grid(row=row_id, columnspan=1, column=1, padx=5, pady=5, sticky='nsw')
        row_id += 1

        values = [str(num) for num in range(5, 21)]
        self.corner_input_label = CTkLabel(f, text='# of Inner CB Corners')
        self.corner_input_label.grid(row=row_id, column=0, columnspan=2, padx=5, pady=5)
        row_id += 1

        self.width_label = CTkLabel(f, text='Width')
        self.width_label.grid(row=row_id, column=0, padx=5, pady=5)
        self.height_label = CTkLabel(f, text='Height')
        self.height_label.grid(row=row_id, column=1, padx=5, pady=5)
        self.width_combo_entry = CTkComboBox(master=f,
                                             values=values,
                                             command=self.widthInput)
        self.width_combo_entry.grid(row=row_id, column=0, padx=5, pady=5)
        self.height_combo_entry = CTkComboBox(master=f,
                                              values=values,
                                              command=self.heightInput)
        self.height_combo_entry.grid(row=row_id, column=1, padx=5, pady=5)
        row_id += 1

        self.sub_corner_input_label = CTkLabel(f, text='# for sub-Pixel Search')
        self.sub_corner_input_label.grid(row=row_id, column=0, columnspan=2, padx=5, pady=5)
        row_id += 1

        self.sub_width_label = CTkLabel(f, text='Width')
        self.sub_width_label.grid(row=row_id, column=0, padx=5, pady=5)
        self.sub_height_label = CTkLabel(f, text='Height')
        self.sub_height_label.grid(row=row_id, column=1, padx=5, pady=5)

        self.sub_width_combo_entry = CTkComboBox(master=f,
                                                 values=values,
                                                 command=self.SUBwidthInput)
        self.sub_width_combo_entry.grid(row=row_id, column=0, padx=5, pady=5)

        self.sub_height_combo_entry = CTkComboBox(master=f,
                                                  values=values,
                                                  command=self.SUBheightInput)
        self.sub_height_combo_entry.grid(row=row_id, column=1, padx=5, pady=5)
        row_id += 1

        self.calibrate_button = CTkButton(f, text="Calibrate!", state="disabled",
                                          command=lambda fr=master_frame: self.calibrate(fr))
        self.calibrate_button.grid(row=row_id, column=0, columnspan=1, padx=5, pady=5)

        self.protectClearCache(f, row_id)

        self.loadFromCache()

        row_id += 1

        return f

    def updateMode(self, new_mode):
        """
        Setter from button click
        """
        self.image_config.cal_mode = CalibrationType(new_mode)
        self.saveToCache()

    def updateImgType(self, new_img_type):
        """
        Setter from button click
        """
        self.image_config.img_type = new_img_type
        self.image_config.img_collection = []
        self.loadImages()

    def fileName(self, idx):
        """
        :param idx: the number of the image in the collection
        :return: the filename with root filepath prepended
        """
        return join(self.filepath, self.image_config.img_collection[idx].image_name)

    def displayImagePointsThread(self, master_frame):
        """
        This button spawns a thread that looks at all the existing chessboard corners (see displayImagePoints()).
        While the thread is spawned, the image menu is unavailable (as it is being consistently updated).
        """
        self.display_image_points_button.configure(text='Calculating', fg_color='gray', state='disabled')
        self.avail_images_label.configure(fg_color='gray', state='disabled')
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

        for img_class in self.image_config.img_collection:
            if img_class.include:
                self.findChessboardCorners(master_frame, img_class, show_image=False)

        self.sortForCuration()
        # Create a blank, black image with 3 color channels (BGR)
        blank_image = np.zeros((height, width, 3), np.uint8)

        # Prepare boolean to confirm at least 1 image produced a chessboard
        got_at_least_one_image = False
        min_val = 10.0
        max_val = 0.0
        for image_class in self.image_config.img_collection:
            if image_class.include and image_class.residual is not None:
                min_val = min([min_val, image_class.residual])
                max_val = max([max_val, image_class.residual])

        residual = copy.copy(min_val)

        #  Examine each image's chessboard solution. If the image has an associated residual, color code the image.
        for idx, image_class in enumerate(self.image_config.img_collection):

            if image_class.include and image_class.img_pts is not None:
                got_at_least_one_image = True

                if image_class.residual is not None:
                    residual = image_class.residual

                for img_pt in image_class.img_pts:
                    b, g, r = colorsys.hsv_to_rgb(0.4 - 0.4 * (residual - min_val) / (max_val - min_val), 1.0, 1.0)
                    cv2.circle(blank_image,
                               (round(img_pt[0][0]), round(img_pt[0][1])), 2,
                               (int(255 * b), int(255 * g), int(255 * r)), 2)

        self.updateImageFrame(master_frame)

        # If we succeeded at at least one image, then display the image of all of the found corners
        if got_at_least_one_image:
            # Draw Legend
            b, g, r = colorsys.hsv_to_rgb(0, 1.0, 1.0)
            cv2.circle(blank_image, (5, 20), 2, (int(255 * b), int(255 * g), int(255 * r)), 2)
            cv2.putText(blank_image, 'Residual of: ' + str(round(max_val, 2)), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255))
            b, g, r = colorsys.hsv_to_rgb(0.4, 1.0, 1.0)
            cv2.circle(blank_image, (5, 40), 2, (int(255 * b), int(255 * g), int(255 * r)), 2)
            cv2.putText(blank_image,
                        'Residual of: ' + str(round(min_val, 2)),
                        (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255))
            cv2.resize(blank_image, (1000, int(1000 / width * height)))

            # Convert the openCV object to an Image object, which is ingested by customtkinter
            filled_image = fromarray(blank_image)

            ctk_image = CTkImage(filled_image, size=(1000, int(1000 / width * height)))
            image_point_window = CTkToplevel(self)
            image_point_window.geometry('1000x' + str(int(1000 / width * height)))
            image_point_window.title("Chessboard Corner Locations")
            label = CTkLabel(master=image_point_window, image=ctk_image, text="")
            label.pack()

        self.display_image_points_button.configure(text=self.display_image_points_label, fg_color=GREEN, state='normal')
        self.saveToCache()
        self.avail_images_label.configure(fg_color=GREEN, state='normal')

    def saveCal(self, button):
        if self.image_config.cam_cal.validCal:
            self.image_config.cam_cal.toFile(self.filepath)
            self.image_config.cam_cal.toBinFile(self.filepath)
            button.configure(fg_color='navy')
            return
        button.configure(fg_color='red')

    def openCalibrationFolder(self):
        if not self.filepath or not os.path.isdir(self.filepath):
            messagebox.showerror('Folder Not Found', 'The current calibration folder is not available.')
            return
        os.startfile(self.filepath)

    def setup_CalFrame(self, master_frame):

        if self.cal_frame is None:
            self.cal_frame = CTkFrame(master_frame)

            self.save_cal_button = CTkButton(master=self.cal_frame, text='Save Calibration')
            self.save_cal_button.configure(command=lambda btn=self.save_cal_button: self.saveCal(btn))
            self.save_cal_button.grid(row=0, column=0, padx=5, pady=5)

            self.open_folder_button = CTkButton(master=self.cal_frame, text='Open Folder',
                                                command=self.openCalibrationFolder)
            self.open_folder_button.grid(row=1, column=0, padx=5, pady=5)

            self.cal_label = CTkLabel(master=self.cal_frame, justify='center', anchor='w')
            self.cal_label.grid(row=2, column=0, padx=5, pady=5)

            self.scale864_button = CTkButton(master=self.cal_frame, text='Scale to 864x864',
                                             command=lambda: self.scaleTo864(self.cal_frame))
            self.scale864_button.grid(row=3, column=0, padx=5, pady=5)

            self.scale2848_button = CTkButton(master=self.cal_frame, text='Scale to 2848x2848',
                                              command=lambda: self.scaleTo2848(self.cal_frame))
            self.scale2848_button.grid(row=4, column=0, padx=5, pady=5)

            self.scale_any_button = CTkButton(master=self.cal_frame, text='Scale to Input Size',
                                              command=lambda: self.scaleToInput(self.cal_frame))
            self.scale_any_button.grid(row=5, column=0, padx=5, pady=5)

        self.updateCalFrameState()
        return self.cal_frame

    def updateCalFrameState(self):
        if self.cal_frame is None:
            return

        folder_button_state = 'normal' if self.filepath and os.path.isdir(self.filepath) else 'disabled'
        self.open_folder_button.configure(state=folder_button_state)

        if self.image_config.cam_cal.validCal:
            self.cal_label.configure(text=self.image_config.cam_cal.calStr)
            self.save_cal_button.configure(state='normal')
            self.scale864_button.configure(state='normal')
            self.scale2848_button.configure(state='normal')
            self.scale_any_button.configure(state='normal')
        else:
            self.cal_label.configure(text="No calibration calculated yet.")
            self.save_cal_button.configure(state='disabled')
            self.scale864_button.configure(state='disabled')
            self.scale2848_button.configure(state='disabled')
            self.scale_any_button.configure(state='disabled')

    def scaleTo864(self, master_frame):
        self.scaleCalibrationTo(864)

    def scaleTo2848(self, master_frame):
        self.scaleCalibrationTo(2848)

    def scaleToInput(self, master_frame):
        dialog = CTkInputDialog(
            text='Input an integer value. The updated calibration width will be this value.',
            title='Calibration Scale Selection')
        try:
            self.scaleCalibrationTo(int(dialog.get_input()))
        except (TypeError, ValueError):
            from support.io.my_logging import LOG
            LOG.warning('Invalid input. Please input only an integer.')

    def scaleCalibrationTo(self, width):
        self.image_config.cam_cal.scaleCalibration(width)
        self.saveToCache()
        self.updateCalFrameState()

    def updateConfigWindow(self, master_frame):
        row_id = 0
        f = CTkFrame(master_frame)

        stopping_iteration_label = CTkLabel(master=f, text='Max Iterations: ')
        stopping_iteration_label.grid(row=row_id, column=0, padx=5, pady=5)
        stopping_iteration_entry = CTkEntry(master=f, placeholder_text=str(self.image_config.max_iter))
        stopping_iteration_entry.bind('<Return>',
                                      lambda event, x=stopping_iteration_entry: self.stoppingCritIterUpdate(x))

        stopping_iteration_entry.grid(row=row_id, column=1, padx=5, pady=5)
        row_id += 1

        stopping_min_step_size_label = CTkLabel(master=f, text='Stopping Min Step Size: ')
        stopping_min_step_size_label.grid(row=row_id, column=0, padx=5, pady=5)

        stopping_min_step_size_entry = CTkEntry(master=f, placeholder_text=str(self.image_config.min_step_size))
        stopping_min_step_size_entry.bind('<Return>',
                                          lambda event, x=stopping_min_step_size_entry:
                                          self.stoppingCritMinStepSizeUpdate(x))
        stopping_min_step_size_entry.grid(row=row_id, column=1, padx=5, pady=5)
        row_id += 1

        fix_principal_point_cb = CTkCheckBox(master=f, text='Fix Principal Point', checkbox_height=20)
        if self.image_config.fix_principal_point:
            fix_principal_point_cb.select()
        else:
            fix_principal_point_cb.deselect()
        fix_principal_point_cb.configure(command=self.toggleFixPrincipalPoint)
        fix_principal_point_cb.grid(row=row_id, column=0, columnspan=2, padx=0, pady=0, sticky='nsw')
        row_id += 1

        fix_aspect_ratio_cb = CTkCheckBox(master=f, text='Fix Aspect Ratio', checkbox_height=20)
        if self.image_config.fix_aspect_ratio:
            fix_aspect_ratio_cb.select()
        else:
            fix_aspect_ratio_cb.deselect()
        fix_aspect_ratio_cb.configure(command=self.toggleFixAspectRatio)
        fix_aspect_ratio_cb.grid(row=row_id, column=0, columnspan=2, padx=0, pady=0, sticky='nsw')
        row_id += 1

        zero_tangent_dist_cb = CTkCheckBox(master=f, text='Zero Tangent Distance', checkbox_height=20)
        if self.image_config.zero_tangent_dist:
            zero_tangent_dist_cb.select()
        else:
            zero_tangent_dist_cb.deselect()
        zero_tangent_dist_cb.configure(command=self.toggleZeroTangentDist)
        zero_tangent_dist_cb.grid(row=row_id, column=0, columnspan=2, padx=0, pady=0, sticky='nsw')
        row_id += 1

        screen_based_checkerboard_cb = CTkCheckBox(master=f, text="Screen Based Checkerboard", checkbox_height=20)
        if self.image_config.screen_based_checkerboard:
            screen_based_checkerboard_cb.select()
        else:
            screen_based_checkerboard_cb.deselect()
        screen_based_checkerboard_cb.configure(command=self.toggleScreenbasedCheckerboard)
        screen_based_checkerboard_cb.grid(row=row_id, column=0, columnspan=2, padx=0, pady=0, sticky='nsw')

        return f

    def stoppingCritMinStepSizeUpdate(self, entry=None):
        try:
            new_step = float(entry.get())
        except ValueError:
            entry.delete(0, END)
            entry.insert(0, str(self.image_config.max_iter))
            return

        if isinstance(new_step, float) and new_step > 0:
            self.image_config.min_step_size = new_step
        else:
            entry.delete(0, END)
            entry.insert(0, str(self.image_config.min_step_size))
        self.saveToCache()
        self.flashEntrySaved(entry)

    def stoppingCritIterUpdate(self, entry=None):
        try:
            new_iter = int(entry.get())
        except ValueError:
            entry.delete(0, END)
            entry.insert(0, str(self.image_config.max_iter))
            return
        self.image_config.max_iter = new_iter
        self.saveToCache()
        self.flashEntrySaved(entry)

    def flashEntrySaved(self, entry):
        entry.configure(fg_color='yellow')
        entry.after(1, self.update_idletasks)
        entry.after(500, lambda: entry.configure(fg_color=GREEN))

    def toggleZeroTangentDist(self):
        self.image_config.zero_tangent_dist = not self.image_config.zero_tangent_dist

    def toggleScreenbasedCheckerboard(self):
        self.image_config.screen_based_checkerboard = not self.image_config.screen_based_checkerboard

    def toggleFixPrincipalPoint(self):
        self.image_config.fix_principal_point = not self.image_config.fix_principal_point

    def toggleFixAspectRatio(self):
        self.image_config.fix_aspect_ratio = not self.image_config.fix_aspect_ratio

    def _render_row(self, idx, widgets, f):
        """Retitle + rebind one row from item idx (no new widgets)."""
        (img_include_checkbox,
         img_name_button,
         img_res,
         img_shp,
         img_restore_button,
         img_find_corners_button,
         img_edit_points_button,
         img_invert_button,
         img_gray_button,
         img_ccw_rotate_button,
         img_cw_rotate_button) = widgets

        img_class = self.image_config.img_collection[idx]

        # checkbox state
        (img_include_checkbox.select() if img_class.include else img_include_checkbox.deselect())
        img_include_checkbox.configure(command=partial(self.updateInclusion, idx),
                                       state="normal")

        # labels
        if getattr(img_class, 'points_edited', False) and img_class.residual is None:
            curr_res = 'Edited'
        elif img_class.residual is None:
            curr_res = ''
        elif img_class.residual == 10000.0:
            curr_res = 'Disabled'
        else:
            curr_res = f'Res: {round(img_class.residual, 3)}'
        img_res.configure(text=curr_res)

        curr_shrp = '' if img_class.sharpness is None else f'Shrp: {round(img_class.sharpness, 3)}'
        img_shp.configure(text=curr_shrp)

        # commands
        img_name_button.configure(text=img_class.image_name,
                                  command=partial(self.showBasicImage, img_class), state="normal")
        img_restore_button.configure(command=partial(self.restore, img_class), state="normal")
        img_find_corners_button.configure(command=partial(self.findChessboardCorners, f, img_class, True, True),
                                          state="normal")
        img_edit_points_button.configure(command=partial(self.editImagePoints, img_class), state="normal")
        img_invert_button.configure(command=partial(self.invertIndividualImage, img_class), state="normal")
        img_gray_button.configure(command=partial(self.grayscaleIndividualImage, img_class), state="normal")
        img_ccw_rotate_button.configure(command=partial(self.rotateCCWIndividualImage, img_class), state="normal")
        img_cw_rotate_button.configure(command=partial(self.rotateCWIndividualImage, img_class), state="normal")

    def _page_bounds(self):
        total = len(self.image_config.img_collection)
        start = max(0, min(self._page_start, max(0, total - 1)))
        end = min(total, start + self._page_size)
        return start, end, total

    def _refresh_all_rows(self, f):
        start, end, total = self._page_bounds()
        needed = end - start
        while len(self.image_config_window_objects) < needed:
            self.createNewRow(self._rows_holder, len(self.image_config_window_objects))

        for i in range(needed):
            idx = start + i
            widgets = self.image_config_window_objects[i]
            self._render_row(idx, widgets, f)
            for w in widgets:
                try:
                    w.grid()
                except tkinter.TclError:
                    pass

        for i in range(needed, len(self.image_config_window_objects)):
            for w in self.image_config_window_objects[i]:
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
        header.grid_columnconfigure(list(range(17)), weight=0)  # plenty of columns

        # Rows container (row 1)
        self._rows_holder = CTkFrame(f, fg_color="transparent")
        self._rows_holder.grid(row=1, column=0, sticky="nsew")
        self._rows_holder.grid_columnconfigure(list(range(11)), weight=0)

        # --- put all header buttons in `header` (NOT in f) ---
        row_id = 0
        select_all_button = CTkButton(master=header, text='Include All',
                                      width=self.header_button_width('Include All'),
                                      command=self.includeAll)
        select_all_button.grid(row=row_id, column=0, padx=5, pady=5)

        remove_unselected = CTkButton(master=header, text='Hide Unchecked',
                                      width=self.header_button_width('Hide Unchecked'),
                                      command=lambda fr=master_frame: self.removeUnused(fr))
        remove_unselected.grid(row=row_id, column=1, padx=5, pady=5, columnspan=2)

        decimate_button = CTkButton(master=header, text='Decimate',
                                    width=self.header_button_width('Decimate'),
                                    command=self._open_decimate_dialog)
        decimate_button.grid(row=row_id, column=3, padx=5, pady=5)

        self.display_image_points_button = CTkButton(master=header, text=self.display_image_points_label,
                                                     width=self.header_button_width(self.display_image_points_label),
                                                     command=lambda fr=master_frame: self.displayImagePointsThread(fr))
        self.display_image_points_button.grid(row=row_id, column=4, columnspan=2, padx=5, pady=5)

        self.img_invert_protected_button = CTkButton(master=header, text='Invert All', hover_color='navy',
                                                     fg_color='blue',
                                                     width=self.header_button_width('Invert All'),
                                                     command=self.unprotectInvert)
        self.img_rotate_ccw_protected_button = CTkButton(master=header, image=self.left_arrow, text='All',
                                                         hover_color='navy', fg_color='blue',
                                                         width=self.header_button_width('All', has_image=True),
                                                         height=self.arrow_button_size,
                                                         command=self.unprotectRotateCCW)
        self.img_rotate_cw_protected_button = CTkButton(master=header, image=self.right_arrow, text='All',
                                                        hover_color='navy', fg_color='blue',
                                                        width=self.header_button_width('All', has_image=True),
                                                        height=self.arrow_button_size,
                                                        command=self.unprotectRotateCW)
        self.img_gray_protected_button = CTkButton(master=header, text='Grayscale All', hover_color='navy',
                                                   fg_color='blue',
                                                   width=self.header_button_width('Grayscale All'),
                                                   command=self.unprotectAllGrayscale)

        self.protectInvert(row=0)
        self.protectRotateCCW(row=0)
        self.protectRotateCW(row=0)
        self.protectAllGrayscale(row=0)

        # Pager controls in the header
        self._page_label = CTkLabel(header, text="1/1")
        self._page_label.grid(row=row_id, column=12, padx=6, pady=6, sticky="e")

        self.first_page_btn = CTkButton(header, text="◀◀", command=self._first_page, width=70)
        self.prev_page_btn = CTkButton(header, text="◀ Prev", command=self._page_prev, width=70)
        self.next_page_btn = CTkButton(header, text="Next ▶", command=self._page_next, width=70)
        self.last_page_btn = CTkButton(header, text="▶▶", command=self._last_page, width=70)
        for page_button in (self.first_page_btn, self.prev_page_btn, self.next_page_btn, self.last_page_btn):
            page_button.configure(width=self.header_button_width(page_button.cget("text")))
        self.first_page_btn.grid(row=row_id, column=13, padx=6, pady=6, sticky='w')
        self.prev_page_btn.grid(row=row_id, column=14, padx=6, pady=6, sticky="w")
        self.next_page_btn.grid(row=row_id, column=15, padx=6, pady=6, sticky="w")
        self.last_page_btn.grid(row=row_id, column=16, padx=6, pady=6, sticky='w')

        # fresh paging state
        self._page_start = 0
        self.image_config_window_objects = []
        for child in list(self._rows_holder.winfo_children()):
            child.destroy()

        self.image_frame = f
        self._refresh_all_rows(f)  # render only current page
        self.updateImageFrame(f)
        return f

    def createNewRow(self, f, row_id):
        img_include_checkbox = CTkCheckBox(master=f, text='', width=24)
        img_include_checkbox.grid(row=row_id, column=0)

        img_name_button = CTkButton(master=f, text='')
        img_name_button.grid(row=row_id, column=1, padx=5, pady=5)

        img_res = CTkLabel(master=f, text='')
        img_res.grid(row=row_id, column=2, padx=5, pady=5)

        img_shp = CTkLabel(master=f, text='')
        img_shp.grid(row=row_id, column=3, padx=5, pady=5)

        img_restore_button = CTkButton(master=f, text='Restore')
        img_restore_button.grid(row=row_id, column=4, padx=5, pady=5)

        img_find_corners_button = CTkButton(master=f, text='Find Corners')
        img_find_corners_button.grid(row=row_id, column=5, padx=5, pady=5)

        img_edit_points_button = CTkButton(master=f, text='Edit Points')
        img_edit_points_button.grid(row=row_id, column=6, padx=5, pady=5)

        img_invert_button = CTkButton(master=f, text='Invert Image')
        img_invert_button.grid(row=row_id, column=7, padx=5, pady=5)

        img_gray_button = CTkButton(master=f, text='Grayscale Image')
        img_gray_button.grid(row=row_id, column=8, padx=5, pady=5)

        img_ccw_rotate_button = CTkButton(master=f, text='', image=self.left_arrow,
                                          width=self.arrow_button_size, height=self.arrow_button_size)
        img_ccw_rotate_button.grid(row=row_id, column=9, padx=5, pady=5)

        img_cw_rotate_button = CTkButton(master=f, text='', image=self.right_arrow,
                                         width=self.arrow_button_size, height=self.arrow_button_size)
        img_cw_rotate_button.grid(row=row_id, column=10, padx=5, pady=5)

        self.image_config_window_objects.append(
            [img_include_checkbox, img_name_button, img_res, img_shp, img_restore_button,
             img_find_corners_button, img_edit_points_button, img_invert_button, img_gray_button,
             img_ccw_rotate_button, img_cw_rotate_button])

    def _update_page_label_and_buttons(self):
        total = len(self.image_config.img_collection)
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
        total = len(self.image_config.img_collection)
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
        f = f or self.image_frame
        if not f:
            return
        self._refresh_all_rows(f)

    def removeUnused(self, master_frame):
        if not os.path.exists(join(self.filepath, 'Removed')):
            os.makedirs(join(self.filepath, 'Removed'))
        remove_ids = []
        for idx, img_class in enumerate(self.image_config.img_collection):

            if not img_class.include and os.path.exists(join(self.filepath, img_class.image_name)):
                os.replace(join(self.filepath, img_class.image_name),
                           join(self.filepath, 'Removed', img_class.image_name))

            if not img_class.include:
                remove_ids.append(idx)

        for idx in reversed(remove_ids):
            self.image_config.img_collection.pop(idx)
        self.saveToCache()
        self.updateImageFrame(master_frame)
        self.updateImageAvailabilityState()

    def copyToRemovedFolder(self, img_class):
        if not os.path.exists(join(self.filepath, 'Removed')):
            os.makedirs(join(self.filepath, 'Removed'))

        src_path = join(self.filepath, img_class.image_name)
        dst_path = join(self.filepath, 'Removed', img_class.image_name)

        if not os.path.exists(dst_path):
            self.writeFile(src_path, dst_path)

    def restore(self, img_class):
        if os.path.exists(join(self.filepath, 'Removed', img_class.image_name)):
            src_path = join(self.filepath, 'Removed', img_class.image_name)
            dst_path = join(self.filepath, img_class.image_name)

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
        for idx, all_gui_items in enumerate(self.image_config_window_objects):
            checkbox, *_ = all_gui_items
            if self.image_config.img_collection[idx].include:
                checkbox.select()
            else:
                checkbox.deselect()

    def unprotectInvert(self):
        self.unprotectImageActionButton(self.img_invert_protected_button, self.invertAll, self.protectInvert)

    def unprotectAllGrayscale(self):
        self.unprotectImageActionButton(self.img_gray_protected_button, self.grayscaleAll, self.protectAllGrayscale)

    def unprotectRotateCCW(self):
        self.unprotectImageActionButton(self.img_rotate_ccw_protected_button, self.rotateAllCCW, self.protectRotateCCW)

    def unprotectRotateCW(self):
        self.unprotectImageActionButton(self.img_rotate_cw_protected_button, self.rotateAllCW, self.protectRotateCW)

    def unprotectImageActionButton(self, button, action, protect):
        button.configure(command=action, fg_color=GREEN, hover_color='dark green')
        button.update_idletasks()
        self.after(2000, protect, 1)

    def protectInvert(self, row=1):
        self.protectImageActionButton(self.img_invert_protected_button, self.unprotectInvert, row, 7)

    def protectAllGrayscale(self, row=1):
        self.protectImageActionButton(self.img_gray_protected_button, self.unprotectAllGrayscale, row, 8)

    def protectRotateCCW(self, row=1):
        self.protectImageActionButton(self.img_rotate_ccw_protected_button, self.unprotectRotateCCW, row, 9)

    def protectRotateCW(self, row=1):
        self.protectImageActionButton(self.img_rotate_cw_protected_button, self.unprotectRotateCW, row, 10)

    @staticmethod
    def protectImageActionButton(button, command, row, column):
        button.configure(fg_color='blue', hover_color='cyan4', command=command)
        if not button.winfo_ismapped():
            button.grid(row=row, column=column, padx=5, pady=5, sticky='nsew')

    def invertAll(self):
        self.applyToAllImages(self.img_invert_protected_button, self.invertIndividualImage, self.protectInvert)

    def grayscaleAll(self):
        self.applyToAllImages(self.img_gray_protected_button, self.grayscaleIndividualImage, self.protectAllGrayscale)

    def rotateAllCW(self):
        self.applyToAllImages(self.img_rotate_cw_protected_button, self.rotateCWIndividualImage, self.protectRotateCW)

    def rotateAllCCW(self):
        self.applyToAllImages(self.img_rotate_ccw_protected_button, self.rotateCCWIndividualImage,
                              self.protectRotateCCW)

    def applyToAllImages(self, button, action, protect):
        button.configure(fg_color='black')
        button.update_idletasks()
        for img_class in self.image_config.img_collection:
            action(img_class)
        protect()

    def invertIndividualImage(self, img_class):
        self.transformIndividualImage(img_class, cv2.bitwise_not)

    def grayscaleIndividualImage(self, img_class):
        self.copyToRemovedFolder(img_class)
        self.transformIndividualImage(img_class, lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    def rotateCWIndividualImage(self, img_class):
        self.transformIndividualImage(img_class, lambda img: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))

    def rotateCCWIndividualImage(self, img_class):
        self.transformIndividualImage(img_class, lambda img: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))

    def transformIndividualImage(self, img_class, transform):
        filepath = join(self.filepath, img_class.image_name)
        img = cv2.imread(filepath)
        cv2.imwrite(filepath, transform(img))

    def includeAll(self):
        for img_class in self.image_config.img_collection:
            img_class.include = True
        self.saveToCache()
        self.updateIncludeCheckboxes()
        self.updateImageAvailabilityState()

    # ------------------------------------------------------------------
    # Decimate: thin the selected image set to a target count
    # ------------------------------------------------------------------
    def _open_decimate_dialog(self):
        """Open a popup that lets the user choose target count and selection mode, then deselect excess images."""
        total = len(self.image_config.img_collection)

        dialog = CTkToplevel(self)
        dialog.title("Decimate Images")
        dialog.geometry("340x220")
        dialog.resizable(False, False)
        dialog.grab_set()  # modal

        CTkLabel(dialog, text=f"Total images: {total}").grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 2))

        CTkLabel(dialog, text="Target # to keep:").grid(row=1, column=0, padx=10, pady=5, sticky="e")
        target_entry = CTkEntry(dialog, width=80)
        target_entry.insert(0, str(total))
        target_entry.grid(row=1, column=1, padx=10, pady=5, sticky="w")

        mode_var = tkinter.IntVar(value=0)  # 0 = uniform, 1 = random
        CTkLabel(dialog, text="Selection mode:").grid(row=2, column=0, columnspan=2, padx=10, pady=(8, 0))
        CTkRadioButton(dialog, text="Uniform (evenly spaced)", variable=mode_var, value=0
                       ).grid(row=3, column=0, columnspan=2, padx=20, pady=2, sticky="w")
        CTkRadioButton(dialog, text="Random", variable=mode_var, value=1
                       ).grid(row=4, column=0, columnspan=2, padx=20, pady=2, sticky="w")

        def on_apply():
            try:
                target = int(target_entry.get())
            except ValueError:
                return
            target = max(0, min(target, total))
            self._apply_decimation(target, mode_var.get())
            dialog.destroy()

        CTkButton(dialog, text="Apply", command=on_apply).grid(row=5, column=0, columnspan=2, padx=10, pady=12)

    def _apply_decimation(self, target: int, mode: int):
        """Deselect images so that only *target* remain selected.

        Parameters
        ----------
        target : int
            Number of images that should stay selected (``include=True``).
        mode : int
            0 – uniform (evenly spaced indices kept),
            1 – random subset kept.
        """
        collection = self.image_config.img_collection
        total = len(collection)

        if target >= total:
            # Nothing to deselect – select all
            for img in collection:
                img.include = True
        elif target <= 0:
            for img in collection:
                img.include = False
        else:
            if mode == 0:  # uniform
                # Evenly spaced indices across the collection
                keep_indices = set(
                    int(round(i * (total - 1) / (target - 1))) for i in range(target)
                )
            else:  # random
                keep_indices = set(random.sample(range(total), target))

            for idx, img in enumerate(collection):
                img.include = idx in keep_indices

        self.saveToCache()
        self.updateIncludeCheckboxes()
        self.updateImageAvailabilityState()

    def unprotectClearCache(self, master_frame, row_id):
        self.clear_cache_button.configure(text='Really Clear Cache', fg_color='green', hover_color='dark green',
                                          text_color='white',
                                          command=lambda f=master_frame: self.clearCache(f, row_id))
        self.clear_cache_button.grid(row=row_id, column=1, columnspan=1, padx=5, pady=5)
        self.after(2000, self.protectClearCache, master_frame, row_id)

    def protectClearCache(self, master_frame, row_id):
        if self.clear_cache_button is None:
            self.clear_cache_button = CTkButton(master=master_frame, text='Clear Cache')
        self.clear_cache_button.configure(text='Clear Cache', fg_color='blue', hover_color='navy', text_color='white',
                                          command=lambda f=master_frame, rid=row_id: self.unprotectClearCache(f, rid))
        self.clear_cache_button.grid(row=row_id, column=1, columnspan=1, padx=5, pady=5)

    def clearCache(self, master_frame, row_id):
        if os.path.exists(join(self.filepath, IMAGE_CACHE)):
            filepath = copy.copy(self.filepath)
            os.remove(join(self.filepath, IMAGE_CACHE))

            self.clear_cache_button.configure(text='Clearing', fg_color='yellow', text_color='black',
                                              hover_color='yellow')
            self.clear_cache_button.grid(row=row_id, column=1, columnspan=1, padx=5, pady=5)
            self.image_config.cam_cal = Calibration()
            self.image_config = ImageryCalibrationConfig()
            self.filepath = filepath
            self.restoreFromWindowState()
            self.loadImages()
            self.updateImageFrame(master_frame)
            self.updateCalFrameState()
        else:
            self.clear_cache_button.configure(text='No cache!', fg_color='red', text_color='white', hover_color='red')
            self.clear_cache_button.grid(row=row_id, column=0, columnspan=2, padx=5, pady=5)
        self.after(2000, self.protectClearCache, master_frame, row_id)

    def updateInclusion(self, idx):
        self.image_config.img_collection[idx].include = not self.image_config.img_collection[idx].include
        self.saveToCache()
        self.updateImageAvailabilityState()

    def restoreFromImageConfig(self):
        if self.image_config.invert_image:
            self.invert_images_checkbox.select()
        else:
            self.invert_images_checkbox.deselect()
        if self.image_config.fisheye:
            self.fisheye_lens_checkbox.select()
        else:
            self.fisheye_lens_checkbox.deselect()
        self.width_combo_entry.set(str(self.image_config.num_inner_corners_w))
        self.height_combo_entry.set(str(self.image_config.num_inner_corners_h))
        self.sub_width_combo_entry.set(str(self.image_config.sub_num_inner_corners_w))
        self.sub_height_combo_entry.set(str(self.image_config.sub_num_inner_corners_h))
        self.folder_label.configure(text="../" + os.path.basename(os.path.normpath(self.filepath)))
        self.select_img_type_combo.set(self.image_config.img_type)
        self.select_mode_combo.set(self.image_config.cal_mode.value)

    def restoreFromWindowState(self):
        self.image_config.invert_image = self.invert_images_checkbox.get()
        self.image_config.num_inner_corners_w = int(self.width_combo_entry.get())
        self.image_config.num_inner_corners_h = int(self.height_combo_entry.get())
        self.image_config.sub_num_inner_corners_w = int(self.sub_width_combo_entry.get())
        self.image_config.sub_num_inner_corners_h = int(self.sub_height_combo_entry.get())
        self.image_config.img_type = self.select_img_type_combo.get()
        self.image_config.cal_mode = CalibrationType(self.select_mode_combo.get())

    def saveToCache(self):

        if len(self.image_config.img_collection) > 0:
            with open(join(self.filepath, IMAGE_CACHE), 'wb') as f:
                pickle.dump(self.image_config, f)

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

            with open(FILEPATH_CACHE, 'rb') as filepath_open:
                self.filepath = pickle.load(filepath_open)
                return

        if os.path.exists(join(self.filepath, IMAGE_CACHE)):
            try:
                with open(join(self.filepath, IMAGE_CACHE), 'rb') as image_config_open:
                    self.image_config.copy(pickle.load(image_config_open))
            except ModuleNotFoundError:
                pass
        else:
            self.image_config = ImageryCalibrationConfig()
            if not self.image_config.cam_cal.fromFile(self.filepath):
                success = self.image_config.cam_cal.fromBinFile(self.filepath)

        self.restoreFromImageConfig()
        self.loadImages()

    def calibrate_buttonCallback(self, master_frame, btn: CTkButton):
        self.calculating = True
        self.calibrate_button.configure(state='disabled', text='Calibrating...', fg_color='gray')
        self.t1 = Thread(target=lambda f=master_frame: self.threadedCalWithButtonCallback(f, btn))
        self.t1.start()

    def threadedCalWithButtonCallback(self, master_frame, btn: CTkButton):
        for img_class in self.image_config.img_collection:
            if img_class.include is True:
                self.findChessboardCorners(master_frame, img_class, False)
        self.calibrateCamera()
        self.saveToCache()
        self.updateImageFrame(master_frame)
        self.finishCalibration(btn)
        self.update()

    def calibrate(self, master_frame):
        self.calculating = True
        self.calibrate_button.configure(state='disabled', text='Calibrating...', fg_color='gray')
        self.t1 = Thread(target=lambda f=master_frame: self.threadedCal(f))
        self.t1.start()

    def threadedCal(self, master_frame):
        for img_class in self.image_config.img_collection:
            if img_class.include is True:
                self.findChessboardCorners(master_frame, img_class, False)
        self.calibrateCamera()
        self.saveToCache()
        self.updateImageFrame(master_frame)
        self.finishCalibration()

    def finishCalibration(self, btn: CTkButton = None):
        self.calibrate_button.configure(state='normal', text='Calibrate', fg_color=GREEN)
        self.calculating = False
        if btn is not None:
            btn.configure(text='Start Calibration', state='normal', fg_color=GREEN, hover_color='dark green')
        if callable(self.on_calibration_complete):
            self.on_calibration_complete()

    def widthInput(self, new_val):
        self.image_config.num_inner_corners_w = int(new_val)
        self.saveToCache()

    def heightInput(self, new_val):
        self.image_config.num_inner_corners_h = int(new_val)
        self.saveToCache()

    def SUBwidthInput(self, new_val):
        self.image_config.sub_num_inner_corners_w = int(new_val)
        if self.image_config.sub_num_inner_corners_w > self.image_config.num_inner_corners_w:
            self.image_config.sub_num_inner_corners_w = self.image_config.num_inner_corners_w
        self.saveToCache()

    def SUBheightInput(self, new_val):
        self.image_config.sub_num_inner_corners_h = int(new_val)
        if self.image_config.sub_num_inner_corners_h > self.image_config.num_inner_corners_h:
            self.image_config.sub_num_inner_corners_h = self.image_config.num_inner_corners_h
        self.saveToCache()

    def selectFolder(self):
        poss_filepath = filedialog.askopenfilename(initialdir=self.filepath + "/..", title="Select Imagery Folder")
        if poss_filepath == '':
            return

        self.filepath = os.path.dirname(poss_filepath)
        self.folder_label.configure(text=os.path.basename(self.filepath))
        self.image_config.img_collection = []

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
        imgs = self.image_config.img_collection

        has_residual = any((img.include and img.residual is not None) for img in imgs)
        has_sharpness = any((img.include and img.sharpness is not None) for img in imgs)

        if has_residual:
            # smaller residuals first; None goes to the bottom via sharpnessTest(None)->50.0
            self.image_config.img_collection = sorted(
                imgs,
                key=lambda img: (self.sharpnessTest(img.residual), img.image_name.lower())
            )
        elif has_sharpness:
            # keep your existing behavior (whatever “sharpness” direction you intended)
            self.image_config.img_collection = sorted(
                imgs,
                key=lambda img: (self.sharpnessTest(img.sharpness), img.image_name.lower())
            )
        else:
            self.image_config.img_collection = sorted(imgs, key=lambda img: img.image_name.lower())

    def sortBySharpness(self):
        self.image_config.img_collection = sorted(self.image_config.img_collection,
                                                  key=lambda img: self.sharpnessTest(img.sharpness))

    def sortByResidual(self):
        self.image_config.img_collection = sorted(self.image_config.img_collection,
                                                  key=lambda img: self.sharpnessTest(img.residual))

    @staticmethod
    def sharpnessTest(sharp_value):
        return 50.0 if sharp_value is None else sharp_value

    def loadImages(self):
        imgs = glob.glob(join(self.filepath, '*.' + self.image_config.img_type))
        existing_names = {img.image_name for img in self.image_config.img_collection}

        for img in imgs:
            img_name = os.path.basename(img)
            if img_name not in existing_names:
                self.image_config.img_collection.append(ImageData(img_name))
                existing_names.add(img_name)

        self.updateImageAvailabilityState()

        self.saveToCache()
        if not self.init_image_frame:
            self.init_image_frame = True

    def updateImageAvailabilityState(self):
        if self.avail_images_label is None or self.calibrate_button is None:
            return

        has_enough_images = self.image_config.num_valid_imgs > 5
        self.avail_images_label.configure(
            text=f'{self.image_config.num_valid_imgs} valid images',
            fg_color='blue' if has_enough_images else 'red'
        )
        self.calibrate_button.configure(state="normal" if has_enough_images else "disabled")

    def invertImageToggle(self):
        self.image_config.invert_image = not self.image_config.invert_image
        self.saveToCache()

    def toggleFisheye(self):
        self.image_config.fisheye = not self.image_config.fisheye
        self.saveToCache()

    def showBasicImage(self, img_class):
        img = cv2.imread(join(self.filepath, img_class.image_name))

        h, w, toss = img.shape
        if h > 1080 or w > 1080:
            self.scale = max(1080 / h, 1080 / w) * 1.1
            disp_img = cv2.resize(img, (int(w * self.scale), int(h * self.scale)))
        else:
            disp_img = copy.copy(img)

        cv2.namedWindow(img_class.image_name, cv2.WINDOW_NORMAL)

        cv2.imshow(img_class.image_name, disp_img)

        self.curr_img_class = img_class
        self.curr_img = copy.copy(disp_img)
        cv2.setMouseCallback(img_class.image_name, self.click_event)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        self.first_click = None

    def click_event(self, event, x, y, flags, param):

        if event == cv2.EVENT_RBUTTONDOWN or flags == cv2.EVENT_FLAG_RBUTTON:
            self.first_click = None
            cv2.imshow(self.curr_img_class.image_name, self.curr_img)
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.first_click = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON and self.first_click is not None:
            cloned_img = copy.copy(self.curr_img)
            cv2.rectangle(cloned_img, self.first_click, (x, y), (0, 255, 0), 2)
            cv2.imshow(self.curr_img_class.image_name, cloned_img)
        elif event == cv2.EVENT_LBUTTONUP and self.first_click is not None:
            img = cv2.imread(join(self.filepath, self.curr_img_class.image_name))
            self.copyToRemovedFolder(self.curr_img_class)
            cv2.destroyAllWindows()

            x = int(x / self.scale)
            y = int(y / self.scale)
            first_x = int(self.first_click[0] / self.scale)
            first_y = int(self.first_click[1] / self.scale)

            low_x = min(x, first_x)
            low_y = min(y, first_y)
            high_x = max(x, first_x)
            high_y = max(y, first_y)

            new_img = copy.copy(img)
            new_img[:, :low_x] = np.zeros(new_img[:, :low_x].shape)
            new_img[:low_y] = np.zeros(new_img[:low_y].shape)
            new_img[:, high_x:] = np.zeros(new_img[:, high_x:].shape)
            new_img[high_y:] = np.zeros(new_img[high_y:].shape)

            cv2.imwrite(join(self.filepath, self.curr_img_class.image_name), new_img)

            h, w, toss = new_img.shape
            disp_img = cv2.resize(new_img, (int(w * self.scale), int(h * self.scale)))

            cv2.imshow("New", disp_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def editImagePoints(self, img_class):
        img = cv2.imread(join(self.filepath, img_class.image_name))
        if img is None:
            messagebox.showwarning("Edit points", "Image could not be read.")
            return
        if img_class.img_pts is None:
            messagebox.showwarning("Edit points", "Find corners before editing points.")
            return

        h, w = img.shape[:2]
        display_scale = min(1.0, 1200 / w, 900 / h)
        window_name = f"Edit Corners - {img_class.image_name}"
        zoom_window_name = f"Corner Zoom - {img_class.image_name}"
        state = {
            'img': img,
            'img_class': img_class,
            'display_scale': display_scale,
            'window_name': window_name,
            'zoom_window_name': zoom_window_name,
            'selected_idx': None,
            'dragging': False,
            'dirty': False,
            'last_orig_xy': None,
            'drag_start_point': None,
            'dragging_in_zoom': False,
            'zoom_origin': (0, 0),
            'zoom_drag_origin': None,
            'select_radius_px': 18,
            'zoom_half_size': 20,
            'zoom_scale': 18,
            'nudge_step': 0.1,
            'fine_nudge_step': 0.01,
            'hotkey_help': [
                "Left drag: move nearest corner",
                "Zoom: drag/click selected corner precisely",
                "I/J/K/L: nudge 0.1 px",
                "W/A/S/D: nudge 0.01 px",
                "Right click: cancel current drag",
                "Esc or Q: close editor",
            ],
        }

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.namedWindow(zoom_window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.edit_points_mouse_event, state)
        cv2.setMouseCallback(zoom_window_name, self.edit_points_zoom_mouse_event, state)
        self.redraw_edit_points(state)

        while True:
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord('q')):
                break
            if self.handle_edit_points_key(key, state):
                self.redraw_edit_points(state)

        cv2.destroyWindow(window_name)
        try:
            cv2.destroyWindow(zoom_window_name)
        except cv2.error:
            pass

        if state['dirty']:
            self.mark_points_edited(img_class)
            self.invalidate_calibration_results()
            self.saveToCache()
            self.updateImageFrame()
            self.show_edited_point_residuals(img_class)

    def handle_edit_points_key(self, key, state):
        selected_idx = state['selected_idx']
        if selected_idx is None:
            return False

        step = state['fine_nudge_step'] if key in (ord('a'), ord('d'), ord('w'), ord('s')) else state['nudge_step']
        deltas = {
            ord('j'): (-step, 0.0),
            ord('l'): (step, 0.0),
            ord('i'): (0.0, -step),
            ord('k'): (0.0, step),
            ord('a'): (-step, 0.0),
            ord('d'): (step, 0.0),
            ord('w'): (0.0, -step),
            ord('s'): (0.0, step),
        }
        if key not in deltas:
            return False

        img_class = state['img_class']
        dx, dy = deltas[key]
        x = img_class.img_pts[selected_idx, 0, 0] + dx
        y = img_class.img_pts[selected_idx, 0, 1] + dy
        self.move_corner_to(img_class, selected_idx, (x, y))
        state['last_orig_xy'] = (x, y)
        state['dirty'] = True
        self.mark_points_edited(img_class)
        self.saveToCache()
        return True

    def edit_points_mouse_event(self, event, x, y, flags, state):
        img_class = state['img_class']
        scale = state['display_scale']
        orig_xy = self.display_to_original_point(x, y, scale, state['img'].shape)
        state['last_orig_xy'] = orig_xy

        if event == cv2.EVENT_RBUTTONDOWN or flags == cv2.EVENT_FLAG_RBUTTON:
            if state['dragging'] and state['selected_idx'] is not None and state['drag_start_point'] is not None:
                idx = state['selected_idx']
                img_class.img_pts[idx, 0, :] = state['drag_start_point']
            state['dragging'] = False
            state['dragging_in_zoom'] = False
            state['selected_idx'] = None
            state['drag_start_point'] = None
            state['zoom_drag_origin'] = None
            self.redraw_edit_points(state)
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            idx = self.nearest_corner_idx(img_class.img_pts, x, y, scale)
            if idx is not None:
                state['selected_idx'] = idx
                state['dragging'] = True
                state['drag_start_point'] = img_class.img_pts[idx, 0, :].copy()
                self.move_corner_to(img_class, idx, orig_xy)
                state['dirty'] = True
                self.redraw_edit_points(state)
            return

        if event == cv2.EVENT_MOUSEMOVE:
            if state['dragging'] and state['selected_idx'] is not None:
                self.move_corner_to(img_class, state['selected_idx'], orig_xy)
                state['dirty'] = True
            self.redraw_edit_points(state)
            return

        if event == cv2.EVENT_LBUTTONUP and state['dragging'] and state['selected_idx'] is not None:
            self.move_corner_to(img_class, state['selected_idx'], orig_xy)
            state['dirty'] = True
            self.mark_points_edited(img_class)
            self.saveToCache()
            state['dragging'] = False
            state['drag_start_point'] = None
            state['zoom_drag_origin'] = None
            self.redraw_edit_points(state)

    def edit_points_zoom_mouse_event(self, event, x, y, flags, state):
        img_class = state['img_class']
        orig_xy = self.zoom_to_original_point(x, y, state)
        state['last_orig_xy'] = orig_xy

        if event == cv2.EVENT_RBUTTONDOWN or flags == cv2.EVENT_FLAG_RBUTTON:
            if state['dragging'] and state['selected_idx'] is not None and state['drag_start_point'] is not None:
                idx = state['selected_idx']
                img_class.img_pts[idx, 0, :] = state['drag_start_point']
            state['dragging'] = False
            state['dragging_in_zoom'] = False
            state['selected_idx'] = None
            state['drag_start_point'] = None
            state['zoom_drag_origin'] = None
            self.redraw_edit_points(state)
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            idx = self.nearest_zoom_corner_idx(img_class.img_pts, x, y, state)
            if idx is not None:
                state['selected_idx'] = idx
                state['drag_start_point'] = img_class.img_pts[idx, 0, :].copy()
            if state['selected_idx'] is not None:
                state['dragging'] = True
                state['dragging_in_zoom'] = True
                state['zoom_drag_origin'] = state['zoom_origin']
                self.move_corner_to(img_class, state['selected_idx'], orig_xy)
                state['dirty'] = True
                self.redraw_edit_points(state)
            return

        if event == cv2.EVENT_MOUSEMOVE and state['dragging_in_zoom'] and state['selected_idx'] is not None:
            self.move_corner_to(img_class, state['selected_idx'], orig_xy)
            state['dirty'] = True
            self.redraw_edit_points(state)
            return

        if event == cv2.EVENT_LBUTTONUP and state['dragging_in_zoom'] and state['selected_idx'] is not None:
            self.move_corner_to(img_class, state['selected_idx'], orig_xy)
            state['dirty'] = True
            self.mark_points_edited(img_class)
            self.saveToCache()
            state['dragging'] = False
            state['dragging_in_zoom'] = False
            state['drag_start_point'] = None
            state['zoom_drag_origin'] = None
            self.redraw_edit_points(state)

    @staticmethod
    def display_to_original_point(x, y, scale, img_shape):
        h, w = img_shape[:2]
        orig_x = float(np.clip(x / scale, 0, w - 1))
        orig_y = float(np.clip(y / scale, 0, h - 1))
        return orig_x, orig_y

    @staticmethod
    def nearest_corner_idx(img_pts, x, y, scale, select_radius_px=18):
        pts = img_pts.reshape(-1, 2).astype(np.float32)
        display_pts = pts * scale
        distances = np.linalg.norm(display_pts - np.array([x, y], dtype=np.float32), axis=1)
        idx = int(np.argmin(distances))
        if distances[idx] > select_radius_px:
            return None
        return idx

    @staticmethod
    def zoom_to_original_point(x, y, state):
        origin = state['zoom_drag_origin'] or state['zoom_origin']
        zoom_scale = state['zoom_scale']
        h, w = state['img'].shape[:2]
        orig_x = float(np.clip(origin[0] + x / zoom_scale, 0, w - 1))
        orig_y = float(np.clip(origin[1] + y / zoom_scale, 0, h - 1))
        return orig_x, orig_y

    @staticmethod
    def nearest_zoom_corner_idx(img_pts, x, y, state):
        origin = state['zoom_drag_origin'] or state['zoom_origin']
        zoom_scale = state['zoom_scale']
        pts = img_pts.reshape(-1, 2).astype(np.float32)
        zoom_pts = (pts - np.array(origin, dtype=np.float32)) * zoom_scale
        distances = np.linalg.norm(zoom_pts - np.array([x, y], dtype=np.float32), axis=1)
        idx = int(np.argmin(distances))
        if distances[idx] > state['select_radius_px']:
            return None
        return idx

    @staticmethod
    def move_corner_to(img_class, idx, orig_xy):
        img_class.img_pts[idx, 0, 0] = float(orig_xy[0])
        img_class.img_pts[idx, 0, 1] = float(orig_xy[1])

    @staticmethod
    def mark_points_edited(img_class):
        img_class.residual = None
        img_class.points_edited = True

    def invalidate_calibration_results(self):
        self.image_config.cam_cal = Calibration()
        for image_class in self.image_config.img_collection:
            image_class.residual = None
        self.sortForCuration()
        self.updateCalFrameState()

    def redraw_edit_points(self, state):
        img = state['img']
        img_class = state['img_class']
        scale = state['display_scale']
        display_img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)),
                                 interpolation=cv2.INTER_NEAREST)
        pts = img_class.img_pts.reshape(-1, 2)

        for idx, (px, py) in enumerate(pts):
            center = (int(round(px * scale)), int(round(py * scale)))
            color = (0, 255, 255) if idx == state['selected_idx'] else (0, 255, 0)
            radius = 6 if idx == state['selected_idx'] else 4
            cv2.circle(display_img, center, radius, color, 2)

        self.draw_hotkey_help(display_img, state['hotkey_help'])
        cv2.imshow(state['window_name'], display_img)
        self.redraw_edit_points_zoom(state)

    @staticmethod
    def draw_hotkey_help(img, lines, origin=(10, 22)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        line_height = 18
        thickness = 1
        x, y = origin
        text_width = max(cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in lines)
        box_w = text_width + 14
        box_h = line_height * len(lines) + 10
        overlay = img.copy()
        cv2.rectangle(overlay, (x - 6, y - 16), (x - 6 + box_w, y - 16 + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, dst=img)
        for idx, line in enumerate(lines):
            baseline_y = y + idx * line_height
            cv2.putText(img, line, (x, baseline_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    @staticmethod
    def redraw_edit_points_zoom(state):
        img = state['img']
        img_class = state['img_class']
        selected_idx = state['selected_idx']
        half_size = state['zoom_half_size']
        zoom_scale = state['zoom_scale']
        h, w = img.shape[:2]

        if state['dragging_in_zoom'] and state['zoom_drag_origin'] is not None:
            x0, y0 = state['zoom_drag_origin']
            x1 = min(w, x0 + half_size * 2 + 1)
            y1 = min(h, y0 + half_size * 2 + 1)
            if selected_idx is not None:
                center_x, center_y = img_class.img_pts[selected_idx, 0, :]
                center_x, center_y = int(round(center_x)), int(round(center_y))
            else:
                center_x, center_y = state['last_orig_xy'] or (x0 + half_size, y0 + half_size)
        elif selected_idx is not None:
            center_x, center_y = img_class.img_pts[selected_idx, 0, :]
            center_x, center_y = int(round(center_x)), int(round(center_y))
            x0 = max(0, center_x - half_size)
            x1 = min(w, center_x + half_size + 1)
            y0 = max(0, center_y - half_size)
            y1 = min(h, center_y + half_size + 1)
        else:
            if state['last_orig_xy'] is not None:
                center_x, center_y = state['last_orig_xy']
            else:
                pts = img_class.img_pts.reshape(-1, 2)
                center_x, center_y = [int(round(v)) for v in pts[0]]
            x0 = max(0, center_x - half_size)
            x1 = min(w, center_x + half_size + 1)
            y0 = max(0, center_y - half_size)
            y1 = min(h, center_y + half_size + 1)

        x0 = int(np.clip(np.floor(x0), 0, w - 1))
        y0 = int(np.clip(np.floor(y0), 0, h - 1))
        x1 = int(np.clip(np.ceil(x1), x0 + 1, w))
        y1 = int(np.clip(np.ceil(y1), y0 + 1, h))
        state['zoom_origin'] = (x0, y0)
        roi = img[y0:y1, x0:x1].copy()
        zoom = cv2.resize(roi, (roi.shape[1] * zoom_scale, roi.shape[0] * zoom_scale),
                          interpolation=cv2.INTER_NEAREST)

        pts = img_class.img_pts.reshape(-1, 2)
        for idx, (px, py) in enumerate(pts):
            if x0 <= px < x1 and y0 <= py < y1:
                point = (int(round((px - x0) * zoom_scale)), int(round((py - y0) * zoom_scale)))
                color = (0, 255, 255) if idx == selected_idx else (0, 255, 0)
                cv2.circle(zoom, point, 7, color, 2)

        cross_x = int(round((center_x - x0) * zoom_scale))
        cross_y = int(round((center_y - y0) * zoom_scale))
        cv2.line(zoom, (cross_x, 0), (cross_x, zoom.shape[0] - 1), (255, 255, 255), 1)
        cv2.line(zoom, (0, cross_y), (zoom.shape[1] - 1, cross_y), (255, 255, 255), 1)
        cv2.putText(zoom, "Drag/click here for precision", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3,
                    cv2.LINE_AA)
        cv2.putText(zoom, "Drag/click here for precision", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        CalibrateGui.draw_hotkey_help(zoom, state['hotkey_help'], origin=(8, 48))
        cv2.imshow(state['zoom_window_name'], zoom)

    def show_edited_point_residuals(self, img_class):
        img = cv2.imread(join(self.filepath, img_class.image_name))
        if img is None or img_class.img_pts is None:
            return
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.drawImagePoints(
            img_class,
            img,
            gray,
            window_name='Edited Chessboard Corners',
            show_homography_residuals=False,
            show_projected_points=False,
            status_text='Edited points shown. Recalibrate to compute calibration residuals.'
        )

    def findChessboardCorners(self, master_frame, img_class, show_image=True, update_image_frame=False):

        if img_class.img_pts is not None and not show_image:
            return  # Already have points for this image

        img = cv2.imread(join(self.filepath, img_class.image_name))

        if img is None:
            img_class.include = False
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

        objp = np.zeros((self.image_config.num_inner_corners_w * self.image_config.num_inner_corners_h, 3),
                        np.float32)
        objp[:, :2] = np.mgrid[0:self.image_config.num_inner_corners_w,
                               0:self.image_config.num_inner_corners_h].T.reshape(-1, 2) * self.image_config.spacing

        if self.image_config.invert_image:
            temp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            inv_img = cv2.bitwise_not(temp)
            gray = inv_img
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if img_class.img_pts is not None and show_image:
            self.drawImagePoints(img_class, img, gray)
            return

        if self.image_config.cal_mode == CalibrationType.Chessboard:
            if self.image_config.screen_based_checkerboard:
                flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
                ret, corners = cv2.findChessboardCornersSB(gray,
                                                           (self.image_config.num_inner_corners_w,
                                                            self.image_config.num_inner_corners_h),
                                                           flags)
            else:
                ret, corners = cv2.findChessboardCorners(gray,
                                                         (self.image_config.num_inner_corners_w,
                                                          self.image_config.num_inner_corners_h))

        elif self.image_config.cal_mode == CalibrationType.Circles:

            ret, corners = cv2.findCirclesGrid(gray,
                                               (self.image_config.num_inner_corners_w,
                                                self.image_config.num_inner_corners_h),
                                               flags=cv2.ADAPTIVE_THRESH_GAUSSIAN_C)  #, blobDetector=blob)
        else:
            ret = False
            corners = None
            print('Unknown Cal Mode')

        if ret:
            img_class.obj_pts = objp

            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, self.image_config.max_iter,
                self.image_config.min_step_size)
            corners2 = cv2.cornerSubPix(gray, np.float32(corners),
                                        (self.image_config.sub_num_inner_corners_h,
                                         self.image_config.sub_num_inner_corners_w),
                                        (-1, -1), criteria)
            img_class.img_pts = corners2
            img_class.residual = None
            img_class.points_edited = False

            sharpness = cv2.estimateChessboardSharpness(gray, (
                self.image_config.num_inner_corners_w, self.image_config.num_inner_corners_h), np.float32(corners2))
            img_class.sharpness = sharpness[0][0]

            if not self.calculating:
                self.invalidate_calibration_results()
        else:
            img_class.include = False
            self.updateImageAvailabilityState()

        if update_image_frame:
            self.updateImageFrame(master_frame)

        self.saveToCache()

        if show_image:
            self.drawImagePoints(img_class, img, gray)

    def drawImagePoints(self, img_class, img, gray, window_name='Chessboard Corners Detected',
                        show_homography_residuals=True, show_projected_points=True, status_text=None):
        if img_class.img_pts is None:
            img_class.include = False
            h, w = gray.shape
            disp_img = cv2.resize(gray, (int(w * self.scale), int(h * self.scale)))
            cv2.imshow('NO CHESSBOARD CORNERS FOUND', disp_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return

        # Compute ROI in original image coordinates
        min_x = max(int(np.min(img_class.img_pts[:, 0, 0]) - 100), 0)
        max_x = min(int(np.max(img_class.img_pts[:, 0, 0]) + 100), img.shape[1])
        min_y = max(int(np.min(img_class.img_pts[:, 0, 1]) - 100), 0)
        max_y = min(int(np.max(img_class.img_pts[:, 0, 1]) + 100), img.shape[0])

        roi_base = img[min_y:max_y, min_x:max_x].copy()

        # Shift points into ROI coordinates
        pts_roi = img_class.img_pts.copy()
        pts_roi[:, 0, 0] -= min_x
        pts_roi[:, 0, 1] -= min_y

        pattern_size = (self.image_config.num_inner_corners_w,
                        self.image_config.num_inner_corners_h)

        if show_homography_residuals or show_projected_points:
            resid_roi, proj_roi, hmat = self.chessboard_point_residuals_homography(pts_roi, pattern_size)
        else:
            resid_roi, proj_roi = None, None
        if not show_projected_points:
            proj_roi = None

        self.show_with_locked_aspect_redraw(
            window_name,
            roi_base,
            pts_roi,
            pattern_size,
            resid_roi=resid_roi,  # <-- new
            proj_roi=proj_roi,  # <-- optional (can draw predicted too)
            on_click=lambda: self.editImagePoints(img_class),
            status_text=status_text
        )

    @staticmethod
    def show_with_locked_aspect_redraw(name, base_img, pts_roi, pattern_size, *, resid_roi=None, proj_roi=None,
                                       on_click=None, status_text=None):
        if base_img.ndim == 2:
            base_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)

        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        click_state = {'clicked': False}
        if on_click is not None:
            def open_editor_on_click(event, _x, _y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    click_state['clicked'] = True

            cv2.setMouseCallback(name, open_editor_on_click)
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
                if click_state['clicked']:
                    cv2.destroyWindow(name)
                    on_click()
                    return
                continue

            if last_w_win == w_win and last_h_win == h_win:
                key = cv2.waitKey(30)
                if key == 27:
                    break
                if click_state['clicked']:
                    cv2.destroyWindow(name)
                    on_click()
                    return
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
                worst_count = min(10, len(resid_roi))
                worst_idx = np.argsort(-resid_roi)[:worst_count]

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
                        u = int(round(u))
                        v = int(round(v))
                        cv2.line(resized, (u - 3, v), (u + 3, v), (255, 255, 255), 1)
                        cv2.line(resized, (u, v - 3), (u, v + 3), (255, 255, 255), 1)

                # Summary text
                mean_e = float(np.mean(resid_roi))
                max_e = float(np.max(resid_roi))
                cv2.putText(resized, f"homography residuals: mean {mean_e:.2f}px  max {max_e:.2f}px",
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(resized, f"homography residuals: mean {mean_e:.2f}px  max {max_e:.2f}px",
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

            if on_click is not None:
                edit_text = "Click anywhere to edit corners"
                cv2.putText(resized, edit_text, (10, max(45, resized.shape[0] - 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(resized, edit_text, (10, max(45, resized.shape[0] - 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            if status_text is not None:
                cv2.putText(resized, status_text, (10, max(45, resized.shape[0] - 40)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(resized, status_text, (10, max(45, resized.shape[0] - 40)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            # Letterbox canvas
            canvas = np.zeros((h_win, w_win, 3), dtype=np.uint8)
            x_off = (w_win - new_w) // 2
            y_off = (h_win - new_h) // 2
            canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

            cv2.imshow(name, canvas)

            key = cv2.waitKey(30)
            if key == 27:
                break
            if click_state['clicked']:
                cv2.destroyWindow(name)
                on_click()
                return

        if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
            return
        cv2.destroyWindow(name)

    @staticmethod
    def chessboard_point_residuals_homography(img_pts, pattern_size):
        """
        imgPts: (N,1,2) float32/64 in ROI coordinates
        pattern_size: (W, H) inner corners
        Returns:
            resid_px: (N,) reprojection residual in pixels
            proj: (N,2) predicted points from homography
            H: 3x3 homography
        """
        pattern_width, pattern_height = pattern_size
        expected_points = pattern_width * pattern_height
        pts = np.asarray(img_pts, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] != expected_points:
            return None, None, None

        # Ideal grid coordinates in chessboard index space
        # (0..W-1, 0..H-1)
        obj2d = np.array([(i, j) for j in range(pattern_height) for i in range(pattern_width)], dtype=np.float32)

        # Robust homography (helps if a few corners are bad)
        hmat, inliers = cv2.findHomography(obj2d, pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if hmat is None:
            # fall back to least squares
            hmat, _ = cv2.findHomography(obj2d, pts, method=0)

        # Project ideal grid through H -> predicted pixel locations
        proj = cv2.perspectiveTransform(obj2d.reshape(-1, 1, 2), hmat).reshape(-1, 2)

        resid = np.linalg.norm(pts - proj, axis=1)  # pixels
        return resid, proj, hmat

    def findIndexGivenImageName(self, name):
        sol_idx = None
        for idx, img_class in enumerate(self.image_config.img_collection):
            if img_class.image_name == name:
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

        start_time = time.time()

        images = []
        obj_points = []
        img_points = []
        for idx, img_class in enumerate(self.image_config.img_collection):
            if img_class.include and img_class.obj_pts is not None and img_class.img_pts is not None:
                images.append(self.fileName(idx))
                obj_points.append(img_class.obj_pts)
                img_points.append(img_class.img_pts)
            elif img_class.include:
                img_class.include = False

        if not images:
            self.rejectCalibration(
                "No usable calibration images were found.\n\n"
                "The calibration pattern was not detected in any selected image. Check the calibration type, "
                "inner-corner width/height, inversion setting, file type, and image quality, then try again."
            )
            return False

        if len(images) < 2:
            self.rejectCalibration(
                "Calibration needs at least two usable images.\n\n"
                f"Only {len(images)} selected image had a detected calibration pattern. Add more images with "
                "successful pattern detections, or check the calibration type, inner-corner width/height, "
                "inversion setting, and image quality."
            )
            return False

        first_img = cv2.imread(images[0])
        if first_img is None:
            self.rejectCalibration(
                "Calibration could not start because the first usable image could not be read."
            )
            return False

        gray = cv2.cvtColor(first_img, cv2.COLOR_BGR2GRAY)

        if self.image_config.fisheye:
            def _as_fisheye_object_points(obj_pts):
                out = []
                for points in obj_pts:
                    points = np.asarray(points, dtype=np.float64)
                    # accept (N,3) or (N,1,3)
                    if points.ndim == 2 and points.shape[1] == 3:
                        points = points.reshape(-1, 1, 3)
                    elif points.ndim == 3 and points.shape[1:] == (1, 3):
                        pass
                    else:
                        raise ValueError(f"objectPoints must be (N,3) or (N,1,3), got {points.shape}")
                    out.append(points)
                return out

            def _as_fisheye_image_points(img_pts):
                out = []
                for p in img_pts:
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

            camera_matrix = np.array([[400.0, 0.0, 400.0],
                                      [0.0, 400.0, 400.0],
                                      [0.0, 0.0, 1.0]], dtype=np.float64)
            distortion_coeffs = np.zeros((4, 1), dtype=np.float64)

            objp = _as_fisheye_object_points(obj_points)
            imgp = _as_fisheye_image_points(img_points)

            # You can pass empty lists; OpenCV will fill them.
            rvecs, tvecs = [], []

            try:
                rms, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.fisheye.calibrate(
                    objectPoints=objp,
                    imagePoints=imgp,
                    image_size=gray.shape[::-1],
                    K=camera_matrix,
                    D=distortion_coeffs,
                    rvecs=rvecs,
                    tvecs=tvecs,
                    flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW,
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                              self.image_config.max_iter,
                              self.image_config.min_step_size)
                )
            except (cv2.error, ValueError) as err:
                self.rejectCalibrationFromOpenCv(err)
                return False

            ret = rms
            mtx = camera_matrix
            dist = distortion_coeffs.reshape(-1)  # 4,
        else:
            try:
                initial_camera_matrix = cv2.initCameraMatrix2D(obj_points, img_points, gray.shape[::-1], 0)
            except cv2.error as err:
                self.rejectCalibrationFromOpenCv(err)
                return False

            flags = (self.image_config.flags or 0) | cv2.CALIB_USE_INTRINSIC_GUESS

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        self.image_config.max_iter,  # e.g. 30–50 is usually enough
                        self.image_config.min_step_size)  # e.g. 1e-6..1e-5

            try:
                cal_values = cv2.calibrateCameraROExtended(
                    objectPoints=obj_points,
                    imagePoints=img_points,
                    imageSize=gray.shape[::-1],
                    iFixedPoint=1,
                    cameraMatrix=initial_camera_matrix,
                    distCoeffs=None,
                    flags=flags,
                    criteria=criteria
                )
            except cv2.error as err:
                self.rejectCalibrationFromOpenCv(err)
                return False
            ret = cal_values[0]
            mtx = cal_values[1]
            dist = cal_values[2]
            # rvecs = calValues[3]
            # tvecs = calValues[4]
            # newObjPoints = calValues[5]
            # stdDevIntrinsics = calValues[6]
            # stdDevExtrinsics = calValues[7]
            # stdDevObjPoints = calValues[8]
            residuals = cal_values[9]

            for img_idx, img in enumerate(images):
                img_class = self.image_config.img_collection[self.findIndexGivenImageName(os.path.basename(img))]
                img_class.residual = residuals[img_idx][0]
                img_class.points_edited = False

            # Sort the images by their residual Values
            self.sortByResidual()

        end_time = time.time()

        fovx, fovy, focal_length, principal_point, aspect_ratio = cv2.calibrationMatrixValues(mtx, gray.shape[::-1],
                                                                                              25.,
                                                                                              25.)

        self.image_config.cam_cal.fisheye = self.image_config.fisheye
        self.image_config.cam_cal.setCameraMatrix(mtx=mtx)
        self.image_config.cam_cal.setDistortion(dist=dist.T)
        self.image_config.cam_cal.setAccessories(calTime=end_time - start_time, numCBUsed=len(images),
                                                 width=gray.shape[::-1][0], height=gray.shape[::-1][1], hfov=fovx,
                                                 rms=ret, timeOfCompute=datetime.datetime.now())

        self.updateCalFrameState()

        self.notify()
        return True

    def rejectCalibration(self, message):
        self.updateImageAvailabilityState()
        print(message)

        def show_warning():
            messagebox.showwarning("Calibration not run", message)

        self.after(0, show_warning)  # type: ignore[call-arg]

    def rejectCalibrationFromOpenCv(self, err):
        self.rejectCalibration(
            "Calibration could not be solved from the selected images.\n\n"
            "Add more usable images with varied camera positions, or check the calibration type, "
            "inner-corner width/height, inversion setting, and image quality.\n\n"
            f"OpenCV reported: {err}"
        )

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
