from __future__ import annotations
from typing import Callable, Optional
from support.io.camera_config import CameraConfig
from support.core.enums import ImageSource
import support.viz.colors as clr
import customtkinter as ctk
from tkinter import filedialog
from pathlib import Path
from yaml import safe_load, dump
import os
import cv2
import cv2_enumerate_cameras


class Filepath_page(ctk.CTkFrame):
    def __init__(
            self,
            master,
            *args,
            get_super_config: Callable[[], 'CameraConfig'],
            start_stream_func: Callable,
            stop_stream_func: Callable,
            save_to_cache: Callable,
            load_from_cache: Callable,
            update_post_newCamConfig: Callable,
            sync_flags_from_model: Callable,
            ingestCalibration: Callable,
            loadTruthPoints: Callable,
            updateYOLOLabel: Callable,
            updateLogFile: Callable,
            default_filepath: str = '',
            **kwargs,
    ):
        self.startStreamOff = stop_stream_func
        self.startStreamOn = start_stream_func
        self.saveToCache = save_to_cache
        self.loadFromCache = load_from_cache
        self.update_post_newCamConfig = update_post_newCamConfig
        self._sync_flags_from_model = sync_flags_from_model
        self.ingestCalibration = ingestCalibration
        self.default_filepath = default_filepath
        self.loadTruthPoints = loadTruthPoints
        self.updateYOLOLabel = updateYOLOLabel
        self.updateLogFile = updateLogFile

        self.indexDict = {}
        self.scanForCameras()

        # remove your custom arg from anything CTk sees
        self._get_super_config = get_super_config

        super().__init__(master, *args, **kwargs)
        self.grid_rowconfigure(list(range(3)), weight=1)  # configure grid system
        self.grid_columnconfigure(list(range(3)), weight=1)

        ###### BUTTON CREATION #######
        self.streamOrImgCombo = ctk.CTkComboBox(self,
                                                values=['Camera Stream', 'Static Image', 'Stream from Folder'],
                                                command=self.sourceUpdate)
        self.selectCameraCombo = ctk.CTkComboBox(self, values=list(self.indexDict.keys()),
                                                 command=self.selectCamera)

        self.startStreamButton = ctk.CTkButton(master=self, text='Start Stream', fg_color=clr.CTK_BUTTON_RED,
                                               hover_color='blue')

        self.singleImageFolderSelect = ctk.CTkButton(self, text='Select Img',
                                                     command=self.selectImagesFilepath)
        self.singleImageTextButton = ctk.CTkButton(self, text='No Image Selected',
                                                   command=self.start_stream)
        if self._get_super_config().imageFilepath is not None:
            self.singleImageTextButton.configure(text=Path(self._get_super_config().imageFilepath).name)

        self.multiImageFolderSelect = ctk.CTkButton(self, text='Select Img Folder',
                                                    command=self.selectImagesFilepath)
        self.multiImageTextButton = ctk.CTkButton(self, text='No Folder Selected',
                                                  command=self.start_stream)

        self.configSelectButton = ctk.CTkButton(self, text='Select Config File',
                                                command=self.selectConfigFile)
        self.configSelectText = ctk.StringVar(value=os.path.basename(self._get_super_config().configFilepath))
        self.configSelectLabel = ctk.CTkLabel(self, textvariable=self.configSelectText)

        selectSaveFolderButton = ctk.CTkButton(self, text='Select Save Folder', command=self.selectFolder)
        self.saveFolderText = ctk.StringVar(value="../" + Path(
            self.default_filepath).name if self.default_filepath else "../")
        selectSaveFolderLabel = ctk.CTkLabel(self,
                                             textvariable=self.saveFolderText)

        if self._get_super_config().imageFilepath is not None:
            self.multiImageTextButton.configure(text=Path(self._get_super_config().imageFilepath).parent.name)

        self.streamOrImgCombo.set(self._get_super_config().imageSource.value)
        self.sourceUpdate(self._get_super_config().imageSource.value)

        # Calibration Selector
        selectCalibButton = ctk.CTkButton(self, text='Select Calibration', command=self.loadCalibration)
        self.selectCalibLabelText = ctk.StringVar(value="../" + os.path.basename(
            os.path.normpath(self._get_super_config().calibFilepath)))
        selectCalibLabel = ctk.CTkLabel(self, textvariable=self.selectCalibLabelText)

        # 3D Truth Points Selector
        selectTruthPointsButton = ctk.CTkButton(master=self, text='Select 3D Truth Points',
                                                hover_color='blue', command=self.select3DTruthFile)
        if self._get_super_config().ThreeDTruthFilepath is None:
            self.selectTruthPointsLabelText = ctk.StringVar(value='No Truth Loaded')
        elif self._get_super_config().ThreeDTruthFilepath:
            self.selectTruthPointsLabelText = ctk.StringVar(value="../" + Path(
                self._get_super_config().ThreeDTruthFilepath).name)
        else:
            self.selectTruthPointsLabelText = ctk.StringVar(value="../")
        selectTruthPointsLabel = ctk.CTkLabel(self, textvariable=self.selectTruthPointsLabelText)

        # YOLO Selector
        selectYOLO_folderButton = ctk.CTkButton(self, text='Select YOLO Folder', fg_color=clr.CTK_GREEN,
                                                command=self.selectYoloFolder)
        self.yoloFolderText = ctk.StringVar(value="../" + Path(
            self._get_super_config().yoloFilepath).name if self._get_super_config().yoloFilepath else "../")
        selectYOLO_folderLabel = ctk.CTkLabel(self,
                                              textvariable=self.yoloFolderText)

        # Flight Log Loader
        selectFlightLogButton = ctk.CTkButton(master=self, text='Select Flight Log File',
                                              hover_color='blue', command=self.selectLogFile)
        if self._get_super_config().hud_data_filepath is None:
            self.FlightLogLabelText = ctk.StringVar(value='No Flight Log Loaded')
        else:
            self.FlightLogLabelText = ctk.StringVar(value="../" + Path(
                self._get_super_config().hud_data_filepath).name if self._get_super_config().hud_data_filepath else "../")
        selectFlightLogLabel = ctk.CTkLabel(self, textvariable=self.FlightLogLabelText)

        # April Tag Size Entry
        aprilTagSizeEntry = ctk.CTkEntry(self, placeholder_text=str(self._get_super_config().aprilTagSize))
        aprilTagSizeEntryButton = ctk.CTkButton(self, text="Enter Size of April Tag (m)",
                                                command=lambda entry=aprilTagSizeEntry: self.setAprilTagSize(entry))

        ###### BUTTON GRIDDING #######
        rowID = 0
        self.streamOrImgCombo.grid(row=rowID, column=0, padx=5, pady=5, sticky='nsew')
        rowID += 2

        self.grid_sideBySide(rowID, self.configSelectButton, self.configSelectLabel)
        rowID += 1

        self.grid_sideBySide(rowID, selectSaveFolderButton, selectSaveFolderLabel)
        rowID += 1

        self.grid_sideBySide(rowID, selectCalibButton, selectCalibLabel)
        rowID += 1

        self.grid_sideBySide(rowID, selectTruthPointsButton, selectTruthPointsLabel)
        rowID += 1

        self.grid_sideBySide(rowID, selectYOLO_folderButton, selectYOLO_folderLabel)
        rowID += 1

        self.grid_sideBySide(rowID, selectFlightLogButton, selectFlightLogLabel)
        rowID += 1

        self.grid_sideBySide(rowID, aprilTagSizeEntryButton, aprilTagSizeEntry)

    def start_stream(self):
        self.selectCameraCombo.configure(state='disabled')
        self.streamOrImgCombo.configure(state='disabled')
        self.singleImageTextButton.configure(command=self.stop_stream)
        self.multiImageTextButton.configure(command=self.stop_stream)
        self.startStreamOn()

    def stop_stream(self):
        self.selectCameraCombo.configure(state='normal')
        self.streamOrImgCombo.configure(state='normal')
        self.singleImageTextButton.configure(command=self.start_stream)
        self.multiImageTextButton.configure(command=self.start_stream)
        self.startStreamOff()

    def selectFolder(self):
        init_dir = Path(self.default_filepath).parent if self.default_filepath else Path.cwd()
        fp = self.askFilepath(str(init_dir), "Select Imagery Folder")
        if fp:
            self._get_super_config().saveFolder = fp
            self.saveToCache(immediate=True)
            self.loadFromCache()
            self.saveFolderText.set(os.path.basename(self._get_super_config().saveFolder))

    def sourceUpdate(self, source):

        self._get_super_config().imageSource = ImageSource(source)

        self.updateSingleOrStream(rowID=1)

    def updateSingleOrStream(self, rowID):
        if not self._get_super_config().imageSource == ImageSource.Camera_Stream:
            if self.selectCameraCombo.grid_info():
                self.selectCameraCombo.grid_forget()
            if self.startStreamButton.grid_info():
                self.startStreamButton.grid_forget()

        if not self._get_super_config().imageSource == ImageSource.Static_Image:
            if self.singleImageFolderSelect.grid_info():
                self.singleImageFolderSelect.grid_forget()
            if self.singleImageTextButton.grid_info():
                self.singleImageTextButton.grid_forget()

        if not self._get_super_config().imageSource == ImageSource.Stream_from_Folder:
            if self.multiImageTextButton.grid_info():
                self.multiImageTextButton.grid_forget()
            if self.multiImageFolderSelect.grid_info():
                self.multiImageFolderSelect.grid_forget()

        if self._get_super_config().imageSource == ImageSource.Camera_Stream:

            self.startStreamOff()
            if not self.startStreamButton.grid_info():
                self.startStreamButton.grid(row=rowID, column=0, padx=5, pady=5, sticky='nsew')
            if not self.selectCameraCombo.grid_info():
                self.selectCameraCombo.grid(row=rowID, column=1, padx=5, pady=5, sticky='nsew')

        elif self._get_super_config().imageSource == ImageSource.Static_Image:

            if self._get_super_config().imageFilepath is None:
                fp = self.default_filepath
            else:
                fp = self._get_super_config().imageFilepath

            self.singleImageTextButton.configure(text=Path(fp).name)
            if not self.singleImageFolderSelect.grid_info():
                self.singleImageFolderSelect.grid(row=1, column=0, padx=5, pady=5, sticky='nsew')
            if not self.singleImageTextButton.grid_info():
                self.singleImageTextButton.grid(row=1, column=1, padx=5, pady=5, sticky='nsew')

        elif self._get_super_config().imageSource == ImageSource.Stream_from_Folder:

            if self._get_super_config().imageFilepath is not None:
                fp = self._get_super_config().imageFilepath
            else:
                fp = self.default_filepath

            self.multiImageTextButton.configure(text=Path(fp).parent.name)
            if not self.multiImageFolderSelect.grid_info():
                self.multiImageFolderSelect.grid(row=1, column=0, padx=5, pady=5, sticky='nsew')
            if not self.multiImageTextButton.grid_info():
                self.multiImageTextButton.grid(row=1, column=1, padx=5, pady=5, sticky='nsew')
        else:
            raise ValueError(f'Unknown Image selection mode: {self._get_super_config().imageSource}')

        self.saveToCache()

    ### HELPERS ########################

    def scanForCameras(self):
        self.indexDict = {}
        from cv2_enumerate_cameras import enumerate_cameras
        for camera_info in enumerate_cameras(cv2.CAP_DSHOW):
            self.indexDict[camera_info.name] = camera_info.index

    @staticmethod
    def askFilepath(initDir, text):
        poss_filepath = filedialog.askdirectory(initialdir=initDir, mustexist=True, title=text)
        if poss_filepath == '':
            return None
        return poss_filepath

    @staticmethod
    def grid_sideBySide(row, *args, col=0):
        for idx, item in enumerate(args):
            item.grid(row=row, column=col + idx, padx=5, pady=5, sticky='nsew')

    ### BUTTON ACTIONS #################
    def selectCamera(self, key):
        self._get_super_config().cam_index = self.indexDict[key]

    def selectImagesFilepath(self):
        if self._get_super_config().imageFilepath is None:
            initDir = str(Path(self.default_filepath).parent)
        else:
            initDir = self._get_super_config().imageFilepath  #os.path.normpath(self.camConfig.imageFilepath)

        poss_file = filedialog.askopenfilename(initialdir=initDir, title="Select Image")
        if poss_file != '':
            self._get_super_config().imageFilepath = poss_file
            self.singleImageTextButton.configure(text=Path(self._get_super_config().imageFilepath).name)
            self.multiImageTextButton.configure(text=Path(self._get_super_config().imageFilepath).parent.name)
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

        self._get_super_config().configFilepath = poss_file
        if os.path.exists(self._get_super_config().configFilepath):
            with open(self._get_super_config().configFilepath, 'r') as f:
                self._get_super_config().fromDict(safe_load(f))
                self.update_post_newCamConfig()
        else:
            with open(self._get_super_config().configFilepath, 'w') as f:
                dump(self._get_super_config().toDict, f)

        self.configSelectLabel.configure(text=os.path.basename(self._get_super_config().configFilepath))
        self._sync_flags_from_model()
        self.saveToCache()

    def setAprilTagSize(self, aprilTagSizeEntry):
        try:
            self._get_super_config().aprilTagSize = float(aprilTagSizeEntry.get())
        except ValueError:
            aprilTagSizeEntry.delete(0, ctk.END)
            aprilTagSizeEntry.configure(placeholder_text=str(self._get_super_config().aprilTagSize), )
        self.saveToCache()

    def loadCalibration(self):
        init_dir = Path(self._get_super_config().calibFilepath or self.default_filepath or Path.cwd()).parent
        poss_filepath = filedialog.askopenfilename(initialdir=str(init_dir), title='Select Calibration File')
        if poss_filepath:
            self._get_super_config().calibFilepath = poss_filepath
            self.ingestCalibration()
            text = "../" + Path(
                self._get_super_config().calibFilepath).name if self._get_super_config().calibFilepath else "../"
            self.selectCalibLabelText.set(text)
        else:
            self.selectCalibLabelText.set("No Calibration Loaded")

    def select3DTruthFile(self):
        init_dir = Path(self._get_super_config().ThreeDTruthFilepath or self.default_filepath or Path.cwd()).parent
        poss_filepath = filedialog.askopenfilename(initialdir=str(init_dir), title='Select 3D Truth Points')
        if poss_filepath:
            self._get_super_config().ThreeDTruthFilepath = poss_filepath
            self.update3DTruthLabel()
            self.loadTruthPoints()
            self.saveToCache()

    def update3DTruthLabel(self):
        if self._get_super_config().ThreeDTruthFilepath:
            self.selectTruthPointsLabelText.set(Path(self._get_super_config().ThreeDTruthFilepath).name)

    def selectYoloFolder(self):
        init_dir = Path(self._get_super_config().yoloFilepath or Path.cwd())
        poss_dir = filedialog.askdirectory(initialdir=str(init_dir), title='Select YOLO Folder')
        if poss_dir:
            self._get_super_config().yoloFilepath = poss_dir
            self.yoloFolderText.set(Path(self._get_super_config().yoloFilepath).name)
            self.updateYOLOLabel()
            self.saveToCache()

    def selectLogFile(self):
        init_dir = Path(self._get_super_config().hud_data_filepath or self.default_filepath or Path.cwd())
        poss_dir = filedialog.askdirectory(initialdir=str(init_dir), title='Select Flight Log Data')
        if poss_dir:
            self._get_super_config().hud_data_filepath = poss_dir
            self.updateLogFile()
