from __future__ import annotations
from typing import Callable, Optional, Protocol
from support.io.camera_config import CameraConfig
from support.viz.draw_pnp_qnp import pnp_qnp_draw
from support.core.enums import ImageSource
import support.viz.colors as clr
import customtkinter as ctk
from tkinter import filedialog
from pathlib import Path
from yaml import safe_load, dump
import os
import cv2


# This protocol class enforces typesafe appropriate usage of super-class functions
# This list of functions is for things this page GUI does not control, but does update
# with interaction on these GUI buttons. Example, select new YOLO model forces the
# super to reload its assigned YOLO model.
class FilepathController(Protocol):
    camConfig: CameraConfig
    pnpDrawer: pnp_qnp_draw

    def saveToCache(self, immediate: bool = False) -> None: ...

    def loadFromCache(self) -> None: ...

    def update_post_newCamConfig(self) -> None: ...

    def sync_flags_from_model(self) -> None: ...

    def ingestCalibration(self) -> None: ...

    def loadTruthPoints(self) -> None: ...

    def updateYOLOModel(self) -> None: ...

    def updateLogFile(self) -> None: ...

    def startStreamToggle(self) -> bool: ...


class Filepath_page(ctk.CTkFrame):
    def __init__(
            self,
            master,
            *args,
            controller: FilepathController,
            **kwargs,
    ):
        self.ctrl = controller
        self.default_filepath = '/..'

        self.indexDict = {}
        self.scanForCameras()

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
                                               hover_color='blue',
                                               command=self.toggle_stream)

        self.singleImageFolderSelect = ctk.CTkButton(self, text='Select Img',
                                                     command=self.selectImagesFilepath)
        self.singleImageTextButton = ctk.CTkButton(self, text='No Image Selected',
                                                   command=self.toggle_stream)
        if self.ctrl.camConfig.imageFilepath is not None:
            self.singleImageTextButton.configure(text=Path(self.ctrl.camConfig.imageFilepath).name)

        self.multiImageFolderSelect = ctk.CTkButton(self, text='Select Img Folder',
                                                    command=self.selectImagesFilepath)
        self.multiImageTextButton = ctk.CTkButton(self, text='No Folder Selected',
                                                  command=self.toggle_stream)

        self.configSelectButton = ctk.CTkButton(self, text='Select Config File',
                                                command=self.selectConfigFile)
        self.configSelectText = ctk.StringVar(value=os.path.basename(self.ctrl.camConfig.configFilepath))
        configSelectLabel = ctk.CTkLabel(self, textvariable=self.configSelectText)

        selectSaveFolderButton = ctk.CTkButton(self, text='Select Save Folder', command=self.selectSaveFolder)
        self.saveFolderText = ctk.StringVar(value="../" + Path(
            self.default_filepath).name if self.default_filepath else "../")
        selectSaveFolderLabel = ctk.CTkLabel(self,
                                             textvariable=self.saveFolderText)

        if self.ctrl.camConfig.imageFilepath is not None:
            self.multiImageTextButton.configure(text=Path(self.ctrl.camConfig.imageFilepath).parent.name)

        self.streamOrImgCombo.set(self.ctrl.camConfig.imageSource.value)
        self.sourceUpdate(self.ctrl.camConfig.imageSource.value)

        # Calibration Selector
        selectCalibButton = ctk.CTkButton(self, text='Select Calibration', command=self.loadCalibration)
        self.selectCalibLabelText = ctk.StringVar(value="../" + os.path.basename(
            os.path.normpath(self.ctrl.camConfig.calibFilepath)))
        selectCalibLabel = ctk.CTkLabel(self, textvariable=self.selectCalibLabelText)

        # 3D Truth Points Selector
        selectTruthPointsButton = ctk.CTkButton(master=self, text='Select 3D Truth Points',
                                                hover_color='blue', command=self.select3DTruthFile)
        if self.ctrl.camConfig.ThreeDTruthFilepath is None:
            self.selectTruthPointsLabelText = ctk.StringVar(value='No Truth Loaded')
        elif self.ctrl.camConfig.ThreeDTruthFilepath:
            self.selectTruthPointsLabelText = ctk.StringVar(value="../" + Path(
                self.ctrl.camConfig.ThreeDTruthFilepath).name)
        else:
            self.selectTruthPointsLabelText = ctk.StringVar(value="../")
        selectTruthPointsLabel = ctk.CTkLabel(self, textvariable=self.selectTruthPointsLabelText)

        # YOLO Selector
        selectYOLO_folderButton = ctk.CTkButton(self, text='Select YOLO Folder',
                                                command=self.selectYoloFolder)
        self.yoloFolderText = ctk.StringVar(value="../" + Path(
            self.ctrl.camConfig.yoloFilepath).name if self.ctrl.camConfig.yoloFilepath else "../")
        selectYOLO_folderLabel = ctk.CTkLabel(self,
                                              textvariable=self.yoloFolderText)

        # Flight Log Loader
        selectFlightLogButton = ctk.CTkButton(master=self, text='Select Flight Log File',
                                              hover_color='blue', command=self.selectLogFile)
        if self.ctrl.camConfig.hud_data_filepath is None:
            self.FlightLogLabelText = ctk.StringVar(value='No Flight Log Loaded')
        else:
            self.FlightLogLabelText = ctk.StringVar(value="../" + Path(
                self.ctrl.camConfig.hud_data_filepath).name if self.ctrl.camConfig.hud_data_filepath else "../")
        selectFlightLogLabel = ctk.CTkLabel(self, textvariable=self.FlightLogLabelText)

        # April Tag Size Entry
        aprilTagSizeEntry = ctk.CTkEntry(self, placeholder_text=str(self.ctrl.camConfig.aprilTagSize))
        aprilTagSizeEntryButton = ctk.CTkButton(self, text="Enter Size of April Tag (m)",
                                                command=lambda entry=aprilTagSizeEntry: self.setAprilTagSize(entry))

        ###### BUTTON GRIDDING #######
        rowID = 0
        self.streamOrImgCombo.grid(row=rowID, column=0, padx=5, pady=5, sticky='nsew')
        rowID += 2

        self.grid_sideBySide(rowID, self.configSelectButton, configSelectLabel)
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

        self.sync_labels()

    def toggle_stream(self):
        running = self.ctrl.startStreamToggle()
        self.update_buttonsForStream(running)

    def update_buttonsForStream(self, running: bool):
        if running:
            self.multiImageTextButton.configure(fg_color="royalblue4")
            self.startStreamButton.configure(text='Stop Stream', fg_color="royalblue4")
            self.selectCameraCombo.configure(state='disabled')
            self.streamOrImgCombo.configure(state='disabled')
        else:
            self.multiImageTextButton.configure(fg_color=clr.CTK_BUTTON_RED)
            self.startStreamButton.configure(text='Start Stream', fg_color=clr.CTK_BUTTON_RED)
            self.selectCameraCombo.configure(state='normal')
            self.streamOrImgCombo.configure(state='normal')

    def selectSaveFolder(self):
        init_dir = Path(self.ctrl.camConfig.saveFolder or Path(self.default_filepath) or Path.cwd())
        fp = self.askFilepath(str(init_dir), "Select Folder For Saving")
        if fp:
            self.ctrl.camConfig.saveFolder = fp
            self.ctrl.saveToCache(immediate=True)
            self.ctrl.loadFromCache()
            self.updateSaveFolderLabel()

    def updateSaveFolderLabel(self):
        self.saveFolderText.set(Path(self.ctrl.camConfig.saveFolder).name)

    def sourceUpdate(self, source):
        self.ctrl.camConfig.imageSource = ImageSource(source)
        self.updateSingleOrStream(rowID=1)

    def updateSingleOrStream(self, rowID):
        if not self.ctrl.camConfig.imageSource == ImageSource.Camera_Stream:
            if self.selectCameraCombo.grid_info():
                self.selectCameraCombo.grid_forget()
            if self.startStreamButton.grid_info():
                self.startStreamButton.grid_forget()

        if not self.ctrl.camConfig.imageSource == ImageSource.Static_Image:
            if self.singleImageFolderSelect.grid_info():
                self.singleImageFolderSelect.grid_forget()
            if self.singleImageTextButton.grid_info():
                self.singleImageTextButton.grid_forget()

        if not self.ctrl.camConfig.imageSource == ImageSource.Stream_from_Folder:
            if self.multiImageTextButton.grid_info():
                self.multiImageTextButton.grid_forget()
            if self.multiImageFolderSelect.grid_info():
                self.multiImageFolderSelect.grid_forget()

        if self.ctrl.camConfig.imageSource == ImageSource.Camera_Stream:

            if not self.startStreamButton.grid_info():
                self.startStreamButton.grid(row=rowID, column=0, padx=5, pady=5, sticky='nsew')
            if not self.selectCameraCombo.grid_info():
                self.selectCameraCombo.grid(row=rowID, column=1, padx=5, pady=5, sticky='nsew')

        elif self.ctrl.camConfig.imageSource == ImageSource.Static_Image:

            if self.ctrl.camConfig.imageFilepath is None:
                fp = self.default_filepath
            else:
                fp = self.ctrl.camConfig.imageFilepath

            self.singleImageTextButton.configure(text=Path(fp).name)
            if not self.singleImageFolderSelect.grid_info():
                self.singleImageFolderSelect.grid(row=1, column=0, padx=5, pady=5, sticky='nsew')
            if not self.singleImageTextButton.grid_info():
                self.singleImageTextButton.grid(row=1, column=1, padx=5, pady=5, sticky='nsew')

        elif self.ctrl.camConfig.imageSource == ImageSource.Stream_from_Folder:

            if self.ctrl.camConfig.imageFilepath is not None:
                fp = self.ctrl.camConfig.imageFilepath
            else:
                fp = self.default_filepath

            self.multiImageTextButton.configure(text=Path(fp).parent.name)
            if not self.multiImageFolderSelect.grid_info():
                self.multiImageFolderSelect.grid(row=1, column=0, padx=5, pady=5, sticky='nsew')
            if not self.multiImageTextButton.grid_info():
                self.multiImageTextButton.grid(row=1, column=1, padx=5, pady=5, sticky='nsew')
        else:
            raise ValueError(f'Unknown Image selection mode: {self.ctrl.camConfig.imageSource}')

        self.ctrl.saveToCache()

    ### HELPERS ########################

    def scanForCameras(self):
        self.indexDict = {}
        from cv2_enumerate_cameras import enumerate_cameras
        for camera_info in enumerate_cameras(cv2.CAP_DSHOW):
            self.indexDict[camera_info.name] = camera_info.index

        # with VmbSystem.get_instance() as vmb:
        #     cams = vmb.get_all_cameras()
        #     if cams:
        #         cam = cams[0]
        #         try:
        #             cam._open()
        #         except vmbpy.c_binding.VmbError as e:
        #             LOG.warning(f'Could not open camera: {e}')
        #             return
        #         try:
        #             cam.start_streaming(
        #                 lambda cam, stream, frame: self.display_frame(cam, stream, frame, "Camera Stream"))
        #             time.sleep(5)
        #             cam.stop_streaming()
        #         finally:
        #             cam._close()

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
        self.ctrl.camConfig.cam_index = self.indexDict[key]

    def selectImagesFilepath(self):
        if self.ctrl.camConfig.imageFilepath is None:
            initDir = str(Path(self.default_filepath).parent)
        else:
            initDir = self.ctrl.camConfig.imageFilepath  #os.path.normpath(self.camConfig.imageFilepath)

        poss_file = filedialog.askopenfilename(initialdir=initDir, title="Select Image")
        if poss_file != '':
            self.ctrl.camConfig.imageFilepath = poss_file
            self.singleImageTextButton.configure(text=Path(self.ctrl.camConfig.imageFilepath).name)
            self.multiImageTextButton.configure(text=Path(self.ctrl.camConfig.imageFilepath).parent.name)
            self.ctrl.saveToCache()

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

        self.ctrl.camConfig.configFilepath = poss_file
        if os.path.exists(self.ctrl.camConfig.configFilepath):
            with open(self.ctrl.camConfig.configFilepath, 'r') as f:
                self.ctrl.camConfig.fromDict(safe_load(f))
                self.ctrl.update_post_newCamConfig()
        else:
            with open(self.ctrl.camConfig.configFilepath, 'w') as f:
                dump(self.ctrl.camConfig.toDict, f)

        self.sync_labels()

    def updateConfigLabel(self):
        self.configSelectText.set(Path(self.ctrl.camConfig.configFilepath).name)

    def sync_labels(self):
        self.updateSingleOrStream(rowID=1)
        self.updateConfigLabel()
        self.updateSaveFolderLabel()
        self.updateCalLabel()
        self.update3DTruthLabel()
        self.updateYoloLabel()
        self.updateLogLabel()

    def setAprilTagSize(self, aprilTagSizeEntry):
        try:
            self.ctrl.camConfig.aprilTagSize = float(aprilTagSizeEntry.get())
        except ValueError:
            aprilTagSizeEntry.delete(0, ctk.END)
            aprilTagSizeEntry.configure(placeholder_text=str(self.ctrl.camConfig.aprilTagSize), )
        self.ctrl.saveToCache()

    def loadCalibration(self):
        init_dir = Path(self.ctrl.camConfig.calibFilepath or self.default_filepath or Path.cwd())
        poss_filepath = filedialog.askopenfilename(initialdir=str(init_dir), title='Select Calibration File')
        if poss_filepath:
            self.ctrl.camConfig.calibFilepath = poss_filepath
            self.ctrl.ingestCalibration()
            self.updateCalLabel()
        else:
            self.selectCalibLabelText.set("No Calibration Loaded")

    def updateCalLabel(self):
        name = Path(self.ctrl.camConfig.calibFilepath).name
        self.selectCalibLabelText.set(name)

    def select3DTruthFile(self):
        init_dir = Path(self.ctrl.camConfig.ThreeDTruthFilepath or self.default_filepath or Path.cwd())
        poss_filepath = filedialog.askopenfilename(initialdir=str(init_dir), title='Select 3D Truth Points')
        if poss_filepath:
            self.ctrl.camConfig.ThreeDTruthFilepath = poss_filepath
            self.update3DTruthLabel()
            self.ctrl.loadTruthPoints()
            self.ctrl.saveToCache()

    def update3DTruthLabel(self):
        if self.ctrl.camConfig.ThreeDTruthFilepath:
            self.selectTruthPointsLabelText.set(Path(self.ctrl.camConfig.ThreeDTruthFilepath).name)

    def selectYoloFolder(self):
        init_dir = Path(self.ctrl.camConfig.yoloFilepath or Path.cwd())
        poss_dir = filedialog.askdirectory(initialdir=str(init_dir), title='Select YOLO Folder')

        # If have an old solution, destroy when changing objects
        if self.ctrl.pnpDrawer is not None:
            self.ctrl.pnpDrawer.last_q_vec = None
            self.ctrl.pnpDrawer.last_t_vec = None

        if poss_dir:
            self.ctrl.camConfig.yoloFilepath = poss_dir
            self.ctrl.updateYOLOModel()
            self.updateYoloLabel()
            self.ctrl.saveToCache()

    def updateYoloLabel(self):
        self.yoloFolderText.set(Path(self.ctrl.camConfig.yoloFilepath).name)

    def selectLogFile(self):
        init_dir = Path(self.ctrl.camConfig.hud_data_filepath or self.default_filepath or Path.cwd())
        poss_dir = filedialog.askdirectory(initialdir=str(init_dir), title='Select Flight Log Data')
        if poss_dir:
            self.ctrl.camConfig.hud_data_filepath = poss_dir
            self.ctrl.updateLogFile()

    def updateLogLabel(self):
        self.FlightLogLabelText.set(Path(self.ctrl.camConfig.hud_data_filepath).name)
