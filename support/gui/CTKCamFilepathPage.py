from __future__ import annotations
from typing import Protocol, Any
from support.io.camera_config import CameraConfig
from support.viz.draw_pnp_qnp import pnp_qnp_draw
from support.core.enums import ImageSource
import support.viz.colors as clr
import customtkinter as ctk
from tkinter import filedialog, messagebox, TclError
from pathlib import Path
from yaml import safe_load, dump
import os
import cv2

from support.io.my_logging import LOG

try:
    from vmbpy import VmbSystem

    HAVE_VMBPY = True
except ImportError:
    VmbSystem = None
    HAVE_VMBPY = False


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

    def queue_live_vimba_update(self, settings: dict[str, Any]) -> bool: ...


class Filepath_page(ctk.CTkFrame):
    def __init__(
            self,
            master,
            *args,
            controller,
            **kwargs,
    ):
        self.ctrl = controller
        self.default_filepath = '/..'

        self.cameraChoices: dict[str, dict[str, Any]] = {}
        self._last_valid_camera_key: str | None = None
        self._suppress_camera_callback = False

        self.scanForCameras()

        super().__init__(master, *args, **kwargs)
        self.grid_rowconfigure(list(range(3)), weight=1)
        self.grid_columnconfigure(list(range(3)), weight=1)

        ###### BUTTON CREATION #######
        self.streamOrImgCombo = ctk.CTkComboBox(
            self,
            values=['Camera Stream', 'Static Image', 'Stream from Folder'],
            command=self.sourceUpdate
        )

        camera_values = list(self.cameraChoices.keys()) if self.cameraChoices else ['No Cameras Found']
        self.selectCameraCombo = ctk.CTkComboBox(
            self,
            values=camera_values,
            command=self.selectCamera,
            state='normal' if self.cameraChoices else 'disabled',
        )

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

        # Alvium / Vimba camera controls
        self._ensure_vimba_control_defaults()
        self.vimba_gain_auto_var = ctk.StringVar(value=str(getattr(self.ctrl.camConfig, "vimba_gain_auto", "Off")))
        self.vimba_gain_var = ctk.StringVar(value=str(getattr(self.ctrl.camConfig, "vimba_gain", 0.0)))
        self.vimba_exposure_auto_var = ctk.StringVar(
            value=str(getattr(self.ctrl.camConfig, "vimba_exposure_auto", "Off")))
        self.vimba_exposure_var = ctk.StringVar(value=str(getattr(self.ctrl.camConfig, "vimba_exposure_us", 10000.0)))
        self.vimba_status_text = ctk.StringVar(value="Alvium controls apply when the Vimba stream starts.")

        self.vimba_controls_frame = ctk.CTkFrame(self)
        self.vimba_controls_frame.grid_columnconfigure(0, weight=0)
        self.vimba_controls_frame.grid_columnconfigure(1, weight=1)
        self.vimba_controls_frame.grid_columnconfigure(2, weight=0)
        self.vimba_controls_frame.grid_columnconfigure(3, weight=1)

        ctk.CTkLabel(self.vimba_controls_frame, text="Alvium Controls", font=("Segoe UI", 13, "bold")).grid(
            row=0, column=0, columnspan=4, padx=5, pady=(5, 2), sticky='w')

        ctk.CTkLabel(self.vimba_controls_frame, text="Gain Auto").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.vimba_gain_auto_combo = ctk.CTkComboBox(
            self.vimba_controls_frame,
            values=["Off", "Once", "Continuous"],
            variable=self.vimba_gain_auto_var,
            command=lambda *_: self._on_vimba_mode_changed(),
        )
        self.vimba_gain_auto_combo.grid(row=1, column=1, padx=5, pady=5, sticky='ew')

        ctk.CTkLabel(self.vimba_controls_frame, text="Gain").grid(row=1, column=2, padx=5, pady=5, sticky='w')
        self.vimba_gain_entry = ctk.CTkEntry(self.vimba_controls_frame, textvariable=self.vimba_gain_var)
        self.vimba_gain_entry.grid(row=1, column=3, padx=5, pady=5, sticky='ew')

        ctk.CTkLabel(self.vimba_controls_frame, text="Exposure Auto").grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.vimba_exposure_auto_combo = ctk.CTkComboBox(
            self.vimba_controls_frame,
            values=["Off", "Once", "Continuous"],
            variable=self.vimba_exposure_auto_var,
            command=lambda *_: self._on_vimba_mode_changed(),
        )
        self.vimba_exposure_auto_combo.grid(row=2, column=1, padx=5, pady=5, sticky='ew')

        ctk.CTkLabel(self.vimba_controls_frame, text="Exposure (us)").grid(row=2, column=2, padx=5, pady=5, sticky='w')
        self.vimba_exposure_entry = ctk.CTkEntry(self.vimba_controls_frame, textvariable=self.vimba_exposure_var)
        self.vimba_exposure_entry.grid(row=2, column=3, padx=5, pady=5, sticky='ew')

        self.vimba_gain_entry.bind("<Return>", lambda *_: self.save_vimba_controls())
        self.vimba_gain_entry.bind("<FocusOut>", lambda *_: self.save_vimba_controls())

        self.vimba_exposure_entry.bind("<Return>", lambda *_: self.save_vimba_controls())
        self.vimba_exposure_entry.bind("<FocusOut>", lambda *_: self.save_vimba_controls())

        self.vimba_profile_var = ctk.StringVar(
            value=self._normalize_vimba_profile_name(
                getattr(self.ctrl.camConfig, "vimba_profile", "Full Res")
            )
        )

        ctk.CTkLabel(self.vimba_controls_frame, text="Profile").grid(
            row=3, column=0, padx=5, pady=5, sticky='w'
        )
        self.vimba_profile_combo = ctk.CTkComboBox(
            self.vimba_controls_frame,
            values=[
                "Full Res",
                "Zoom 1440",
                "BinSum To 1440",
                "BinAvg To 1440",
                "Zoom 864",
                "BinSum To 864",
                "BinAvg To 864",
            ],
            variable=self.vimba_profile_var,
            command=lambda *_: self._on_vimba_profile_changed(),
        )
        self.vimba_profile_combo.grid(row=3, column=1, padx=5, pady=5, sticky='ew')

        self.vimba_read_button = ctk.CTkButton(self.vimba_controls_frame, text="Read Camera",
                                               command=self.read_vimba_controls)
        self.vimba_read_button.grid(row=4, column=0, padx=5, pady=5, sticky='ew')

        self.vimba_save_button = ctk.CTkButton(self.vimba_controls_frame, text="Save Settings",
                                               command=self.save_vimba_controls)
        self.vimba_save_button.grid(row=4, column=1, padx=5, pady=5, sticky='ew')

        ctk.CTkLabel(self.vimba_controls_frame, textvariable=self.vimba_status_text, justify='left').grid(
            row=4, column=2, columnspan=2, padx=5, pady=5, sticky='w')
        self._refresh_vimba_manual_widgets()

        self.streamOrImgCombo.set(self.ctrl.camConfig.imageSource.value)
        self.sourceUpdate(self.ctrl.camConfig.imageSource.value)
        self._restore_selected_camera_combo()

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
        # If we're about to START a stream, first push any UI-edited Vimba values
        # into camConfig so the stream thread uses the latest settings.
        stream_var = getattr(self.ctrl, "stream_running_var", None)
        currently_running = bool(stream_var.get()) if stream_var is not None else False

        if not currently_running and self._should_show_vimba_controls():
            self.save_vimba_controls()

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

        self._set_vimba_controls_enabled(not running)

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

        if self.ctrl.camConfig.imageSource == ImageSource.Camera_Stream:
            self.scanForCameras()
            self._refresh_camera_combo_values()
            self._restore_selected_camera_combo()

        self.updateSingleOrStream(rowID=1)

    def updateSingleOrStream(self, rowID):
        if not self.ctrl.camConfig.imageSource == ImageSource.Camera_Stream:
            if self.selectCameraCombo.grid_info():
                self.selectCameraCombo.grid_forget()
            if self.startStreamButton.grid_info():
                self.startStreamButton.grid_forget()
            if self.vimba_controls_frame.grid_info():
                self.vimba_controls_frame.grid_forget()

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

            if self._should_show_vimba_controls():
                self._sync_vimba_controls_from_model()
                if not self.vimba_controls_frame.grid_info():
                    self.vimba_controls_frame.grid(row=rowID + 1, column=0, columnspan=2, padx=5, pady=5, sticky='nsew')

                stream_var = getattr(self.ctrl, "stream_running_var", None)
                running = bool(stream_var.get()) if stream_var is not None else False
                self._set_vimba_controls_enabled(not running)
            elif self.vimba_controls_frame.grid_info():
                self.vimba_controls_frame.grid_forget()

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
    def _on_vimba_profile_changed(self):
        profile = str(self.vimba_profile_var.get() or "Full Res")
        blurbs = {
            "Full Res": "Full sensor, no preview downscale.",
            "Zoom 1440": "Centered 1440 ROI. Applies on next stream start.",
            "BinSum To 1440": "2x2 digital sum binning for a 1440-class preview. Applies on next stream start.",
            "BinAvg To 1440": "2x2 digital average binning for a 1440-class preview. Applies on next stream start.",
            "Zoom 864": "Centered 864 ROI. Applies on next stream start.",
            "BinSum To 864": "2x2 digital sum binning for a fast 864-class preview. Applies on next stream start.",
            "BinAvg To 864": "2x2 digital average binning for a fast 864-class preview. Applies on next stream start.",
        }
        self.vimba_status_text.set(blurbs.get(profile, "Vimba profile updated."))
        self.save_vimba_controls()

    def _ensure_vimba_control_defaults(self):
        defaults = {
            "vimba_profile": "Full Res",
            "vimba_gain_auto": "Off",
            "vimba_gain": 0.0,
            "vimba_exposure_auto": "Off",
            "vimba_exposure_us": 10000.0,
        }
        for name, value in defaults.items():
            if not hasattr(self.ctrl.camConfig, name):
                setattr(self.ctrl.camConfig, name, value)

    @staticmethod
    def _normalize_vimba_auto_mode(value: Any) -> str:
        text = str(value or "Off").strip().lower()
        mapping = {
            "off": "Off",
            "once": "Once",
            "continuous": "Continuous",
            "manual": "Off",
            "false": "Off",
            "true": "Continuous",
        }
        return mapping.get(text, "Off")

    @staticmethod
    def _normalize_vimba_profile_name(value: Any) -> str:
        text = str(value or "Full Res").strip()
        mapping = {
            "Bin To 1440": "BinSum To 1440",
            "Bin To 864": "BinSum To 864",
            "Bin To 712": "BinSum To 712",
        }
        return mapping.get(text, text)

    def _should_show_vimba_controls(self) -> bool:
        return (
                self.ctrl.camConfig.imageSource == ImageSource.Camera_Stream
                and bool(getattr(self.ctrl.camConfig, "use_vimba", False))
        )

    def _set_vimba_controls_enabled(self, enabled: bool):
        # Always allow hot tuning widgets
        hot_state = 'normal'

        for widget_name in (
                'vimba_gain_auto_combo',
                'vimba_gain_entry',
                'vimba_exposure_auto_combo',
                'vimba_exposure_entry',
                'vimba_save_button',
        ):
            widget = getattr(self, widget_name, None)
            if widget is None:
                continue
            try:
                widget.configure(state=hot_state)
            except TclError:
                pass

        # Keep camera-read disabled while running if you want to avoid contention
        cold_state = 'normal' if enabled else 'disabled'
        for widget_name in (
                'vimba_read_button',
        ):
            widget = getattr(self, widget_name, None)
            if widget is None:
                continue
            try:
                widget.configure(state=cold_state)
            except TclError:
                pass

        self._refresh_vimba_manual_widgets()

    def _refresh_vimba_manual_widgets(self):
        gain_state = 'normal' if self._normalize_vimba_auto_mode(
            self.vimba_gain_auto_var.get()) == 'Off' else 'disabled'
        exp_state = 'normal' if self._normalize_vimba_auto_mode(
            self.vimba_exposure_auto_var.get()) == 'Off' else 'disabled'
        try:
            self.vimba_gain_entry.configure(state=gain_state)
        except TclError:
            pass
        try:
            self.vimba_exposure_entry.configure(state=exp_state)
        except TclError:
            pass

    def _on_vimba_mode_changed(self):
        self._refresh_vimba_manual_widgets()
        self.save_vimba_controls()

    def _sync_vimba_controls_from_model(self):
        self._ensure_vimba_control_defaults()

        if not hasattr(self, "vimba_gain_auto_var"):
            return

        self.vimba_profile_var.set(
            self._normalize_vimba_profile_name(
                getattr(self.ctrl.camConfig, "vimba_profile", "Full Res")
            )
        )
        self.vimba_gain_auto_var.set(
            self._normalize_vimba_auto_mode(
                getattr(self.ctrl.camConfig, "vimba_gain_auto", "Off")
            )
        )
        self.vimba_gain_var.set(str(getattr(self.ctrl.camConfig, "vimba_gain", 0.0)))
        self.vimba_exposure_auto_var.set(
            self._normalize_vimba_auto_mode(
                getattr(self.ctrl.camConfig, "vimba_exposure_auto", "Off")
            )
        )
        self.vimba_exposure_var.set(str(getattr(self.ctrl.camConfig, "vimba_exposure_us", 10000.0)))
        self._refresh_vimba_manual_widgets()

    def save_vimba_controls(self):
        self._ensure_vimba_control_defaults()

        self.ctrl.camConfig.vimba_profile = self._normalize_vimba_profile_name(
            self.vimba_profile_var.get() or "Full Res"
        )
        self.ctrl.camConfig.vimba_gain_auto = self._normalize_vimba_auto_mode(self.vimba_gain_auto_var.get())
        self.ctrl.camConfig.vimba_exposure_auto = self._normalize_vimba_auto_mode(self.vimba_exposure_auto_var.get())

        try:
            self.ctrl.camConfig.vimba_gain = float(self.vimba_gain_var.get())
        except ValueError:
            self.vimba_gain_var.set(str(getattr(self.ctrl.camConfig, "vimba_gain", 0.0)))

        try:
            self.ctrl.camConfig.vimba_exposure_us = float(self.vimba_exposure_var.get())
        except ValueError:
            self.vimba_exposure_var.set(str(getattr(self.ctrl.camConfig, "vimba_exposure_us", 10000.0)))

        self.ctrl.saveToCache(immediate=True)

        live_applied = self.ctrl.queue_live_vimba_update({
            "vimba_gain_auto": self.ctrl.camConfig.vimba_gain_auto,
            "vimba_gain": self.ctrl.camConfig.vimba_gain,
            "vimba_exposure_auto": self.ctrl.camConfig.vimba_exposure_auto,
            "vimba_exposure_us": self.ctrl.camConfig.vimba_exposure_us,
        })

        if live_applied:
            self.vimba_status_text.set(
                "Gain/exposure sent live. "
                f"Profile '{self.ctrl.camConfig.vimba_profile}' "
                "will apply on next stream start."
            )
        else:
            self.vimba_status_text.set(
                "Alvium settings saved. "
                f"Profile '{self.ctrl.camConfig.vimba_profile}' "
                "will apply on the next Vimba stream start."
            )

        self._refresh_vimba_manual_widgets()

    @staticmethod
    def _try_get_camera_feature(cam, *feature_names: str):
        for feature_name in feature_names:
            try:
                feature = getattr(cam, feature_name)
            except AttributeError:
                feature = None
            if feature is not None:
                return feature, feature_name
        return None, None

    def read_vimba_controls(self):
        if not HAVE_VMBPY:
            messagebox.showerror("VmbPy Missing", "VmbPy is not installed, so Alvium controls cannot be read.")
            return

        meta = self.cameraChoices.get(self.selectCameraCombo.get())
        if not meta or meta.get("backend") != "vimba":
            messagebox.showinfo("No Alvium Selected", "Select a Vimba/Alvium camera first.")
            return

        cam_id = str(meta.get("camera_id") or getattr(self.ctrl.camConfig, "vimba_camera_id", "") or "").strip()
        if not cam_id:
            messagebox.showerror("Camera ID Missing", "Could not determine the selected Alvium camera id.")
            return

        try:
            with VmbSystem.get_instance() as vmb:
                cam = vmb.get_camera_by_id(cam_id)
                with cam:
                    gain_auto_feat, _ = self._try_get_camera_feature(cam, "GainAuto")
                    gain_feat, _ = self._try_get_camera_feature(cam, "Gain")
                    exposure_auto_feat, _ = self._try_get_camera_feature(cam, "ExposureAuto")
                    exposure_feat, _ = self._try_get_camera_feature(cam, "ExposureTime", "ExposureTimeAbs")

                    if gain_auto_feat is not None:
                        self.vimba_gain_auto_var.set(self._normalize_vimba_auto_mode(gain_auto_feat.get()))
                    if gain_feat is not None:
                        self.vimba_gain_var.set(str(gain_feat.get()))
                    if exposure_auto_feat is not None:
                        self.vimba_exposure_auto_var.set(self._normalize_vimba_auto_mode(exposure_auto_feat.get()))
                    if exposure_feat is not None:
                        self.vimba_exposure_var.set(str(exposure_feat.get()))
        except (AttributeError, RuntimeError) as e:
            LOG.warning(f"Could not read Vimba camera controls: {e}")
            messagebox.showerror("Read Failed", f"Could not read Alvium controls.\n\n{e}")
            return

        self.save_vimba_controls()
        self.vimba_status_text.set("Alvium settings loaded from the selected camera.")

    def _refresh_camera_combo_values(self):
        values = list(self.cameraChoices.keys()) if self.cameraChoices else ['No Cameras Found']
        self.selectCameraCombo.configure(
            values=values,
            state='normal' if self.cameraChoices else 'disabled',
        )

    def _preferred_camera_key_from_model(self) -> str | None:
        use_vimba = bool(getattr(self.ctrl.camConfig, "use_vimba", False))
        vimba_camera_id = str(getattr(self.ctrl.camConfig, "vimba_camera_id", "") or "")
        cam_index = getattr(self.ctrl.camConfig, "cam_index", None)

        if use_vimba and vimba_camera_id:
            for key, meta in self.cameraChoices.items():
                if meta.get("backend") == "vimba" and meta.get("camera_id") == vimba_camera_id:
                    return key

        for key, meta in self.cameraChoices.items():
            if meta.get("backend") == "opencv" and meta.get("cam_index") == cam_index:
                return key

        return next(iter(self.cameraChoices), None)

    def _restore_selected_camera_combo(self):
        key = self._preferred_camera_key_from_model()
        if key is None:
            self._last_valid_camera_key = None
            if hasattr(self, "selectCameraCombo"):
                self._suppress_camera_callback = True
                try:
                    self.selectCameraCombo.set("No Cameras Found")
                finally:
                    self._suppress_camera_callback = False
            return

        self._last_valid_camera_key = key
        if hasattr(self, "selectCameraCombo"):
            self._suppress_camera_callback = True
            try:
                self.selectCameraCombo.set(key)
            finally:
                self._suppress_camera_callback = False

    @staticmethod
    def _probe_camera_choice(meta: dict[str, Any]) -> tuple[bool, str | None]:
        backend = meta.get("backend")

        if backend == "opencv":
            idx = int(meta["cam_index"])
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            try:
                if not cap.isOpened():
                    return False, f"OpenCV camera index {idx} could not be opened."
                ok, _ = cap.read()
                if not ok:
                    return False, f"OpenCV camera index {idx} opened but did not return a frame."
                return True, None
            finally:
                cap.release()

        if backend == "vimba":
            if not HAVE_VMBPY:
                return False, "VmbPy is not installed."

            cam_id = meta["camera_id"]
            try:
                with VmbSystem.get_instance() as vmb:
                    cam = vmb.get_camera_by_id(cam_id)
                    with cam:
                        pass
                return True, None
            except (AttributeError, RuntimeError) as e:
                return False, f"Vimba camera could not be opened: {e}"

        return False, f"Unknown camera backend: {backend}"

    def scanForCameras(self):
        self.cameraChoices = {}

        # --- OpenCV / DirectShow cameras ---
        try:
            from cv2_enumerate_cameras import enumerate_cameras
            for camera_info in enumerate_cameras(cv2.CAP_DSHOW):
                label = f"OpenCV: {camera_info.name} [{camera_info.index}]"
                self.cameraChoices[label] = {
                    "backend": "opencv",
                    "cam_index": int(camera_info.index),
                    "display_name": camera_info.name,
                }
        except (ImportError, cv2.error, OSError) as e:
            LOG.warning(f"Could not enumerate OpenCV cameras: {e}")

        # --- Vimba cameras ---
        if HAVE_VMBPY:
            try:
                with VmbSystem.get_instance() as vmb:
                    for cam in vmb.get_all_cameras():
                        try:
                            cam_id = cam.get_id()
                        except (AttributeError, RuntimeError):
                            cam_id = ""

                        try:
                            serial = cam.get_serial()
                        except (AttributeError, RuntimeError):
                            serial = ""

                        try:
                            name = cam.get_name()
                        except (AttributeError, RuntimeError):
                            name = cam_id or "Unknown Vimba Camera"

                        suffix = serial if serial else cam_id
                        label = f"Vimba: {name} [{suffix}]"

                        self.cameraChoices[label] = {
                            "backend": "vimba",
                            "camera_id": cam_id,
                            "camera_serial": serial,
                            "display_name": name,
                        }
            except (AttributeError, RuntimeError) as e:
                LOG.warning(f"Could not enumerate Vimba cameras: {e}")

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
        if self._suppress_camera_callback:
            return

        meta = self.cameraChoices.get(key)
        if meta is None:
            self._restore_selected_camera_combo()
            return

        ok, err = self._probe_camera_choice(meta)
        if not ok:
            LOG.warning(f"Rejected camera selection '{key}': {err}")
            messagebox.showerror("Camera Connection Failed", err or f"Could not open camera:\n{key}")

            # Snap back to previous valid choice
            self._suppress_camera_callback = True
            try:
                if self._last_valid_camera_key is not None:
                    self.selectCameraCombo.set(self._last_valid_camera_key)
                else:
                    fallback = self._preferred_camera_key_from_model()
                    if fallback is not None:
                        self.selectCameraCombo.set(fallback)
            finally:
                self._suppress_camera_callback = False
            return

        # Accept selection
        if meta["backend"] == "opencv":
            self.ctrl.camConfig.cam_index = int(meta["cam_index"])
            self.ctrl.camConfig.use_vimba = False
            self.ctrl.camConfig.vimba_camera_id = ""
            self.vimba_status_text.set("OpenCV camera selected.")
        elif meta["backend"] == "vimba":
            self.ctrl.camConfig.use_vimba = True
            self.ctrl.camConfig.vimba_camera_id = str(meta["camera_id"])
            self.vimba_status_text.set("Alvium selected. Settings apply when the Vimba stream starts.")

        self._last_valid_camera_key = key
        self._sync_vimba_controls_from_model()
        self.updateSingleOrStream(rowID=1)
        self.ctrl.saveToCache(immediate=True)

    def selectImagesFilepath(self):
        if self.ctrl.camConfig.imageFilepath is None:
            initDir = str(Path(self.default_filepath).parent)
        else:
            initDir = str(
                Path(self.ctrl.camConfig.imageFilepath).parent)  #os.path.normpath(self.camConfig.imageFilepath)

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
        self._sync_vimba_controls_from_model()
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
        if self.ctrl.camConfig.calibFilepath is not None:
            init_dir = Path(self.ctrl.camConfig.calibFilepath).parent
        else:
            init_dir = Path(self.default_filepath or Path.cwd())
        poss_filepath = filedialog.askopenfilename(initialdir=str(init_dir), title='Select Calibration File')
        if poss_filepath != '':
            self.ctrl.camConfig.calibFilepath = poss_filepath
            self.ctrl.ingestCalibration()
            self.updateCalLabel()

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
