from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol
import os

import cv2
from yaml import YAMLError, dump, safe_load

from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from support.core.enums import ImageSource
from support.io.camera_config import CameraConfig
from support.io.my_logging import LOG
from support.viz.draw_pnp_qnp import pnp_qnp_draw

try:
    from vmbpy import VmbSystem

    HAVE_VMBPY = True
except ImportError:
    VmbSystem = None
    HAVE_VMBPY = False


class FilepathController(Protocol):
    camConfig: CameraConfig
    pnpDrawer: pnp_qnp_draw | None
    stream_running_var: Any

    def saveToCache(self, immediate: bool = False) -> None: ...
    def after_cancel(self, handle: Any) -> None: ...
    def loadFromCache(self) -> None: ...
    def update_post_newCamConfig(self) -> None: ...
    def ingestCalibration(self) -> None: ...
    def loadTruthPoints(self) -> None: ...
    def updateYOLOModel(self) -> None: ...
    def updateLogFile(self) -> bool: ...
    def startStreamToggle(self) -> bool: ...
    def queue_live_vimba_update(self, settings: dict[str, Any]) -> bool: ...


class PathRow(QWidget):
    def __init__(self, button_text: str, callback, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.button = QPushButton(button_text)
        self.value = QLabel()
        self.value.setWordWrap(True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.button)
        layout.addWidget(self.value, 1)
        self.button.clicked.connect(callback)

    def set_path(self, path: str | os.PathLike | None, empty: str = "Not selected") -> None:
        if path:
            path_obj = Path(path)
            self.value.setText(path_obj.name or str(path_obj))
            self.value.setToolTip(str(path_obj))
        else:
            self.value.setText(empty)
            self.value.setToolTip("")


class FilepathPage(QWidget):
    """Native Qt source, file, calibration, and Alvium configuration page."""

    VIMBA_PROFILES = (
        "Full Res",
        "Zoom 1440",
        "BinSum To 1440",
        "BinAvg To 1440",
        "Zoom 864",
        "BinSum To 864",
        "BinAvg To 864",
    )

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        controller: FilepathController,
    ) -> None:
        super().__init__(parent)
        self.ctrl = controller
        self.default_filepath = str(Path.cwd())
        self.cameraChoices: dict[str, dict[str, Any]] = {}
        self._last_valid_camera_key: str | None = None
        self._suppress_camera_callback = False
        self._ensure_vimba_control_defaults()

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        self.streamOrImgCombo = QComboBox()
        self.streamOrImgCombo.addItems(
            ["Camera Stream", "Static Image", "Stream from Folder"]
        )
        root.addWidget(self.streamOrImgCombo)

        self.source_stack = QStackedWidget()
        self.source_stack.addWidget(self._build_camera_source())
        self.source_stack.addWidget(self._build_static_source())
        self.source_stack.addWidget(self._build_folder_source())
        root.addWidget(self.source_stack)

        self.config_row = PathRow("Select Config File", self.selectConfigFile)
        self.save_row = PathRow("Select Save Folder", self.selectSaveFolder)
        self.calibration_row = PathRow("Select Calibration", self.loadCalibration)
        self.truth_row = PathRow("Select 3D Truth Points", self.select3DTruthFile)
        self.yolo_row = PathRow("Select YOLO Folder", self.selectYoloFolder)
        self.log_row = PathRow("Select Flight Log Folder", self.selectLogFile)
        for row in (
            self.config_row,
            self.save_row,
            self.calibration_row,
            self.truth_row,
            self.yolo_row,
            self.log_row,
        ):
            root.addWidget(row)

        april_row = QWidget()
        april_layout = QHBoxLayout(april_row)
        april_layout.setContentsMargins(0, 0, 0, 0)
        april_layout.addWidget(QLabel("April tag size (m)"))
        self.april_tag_size = QDoubleSpinBox()
        self.april_tag_size.setDecimals(6)
        self.april_tag_size.setRange(1.0e-6, 1.0e6)
        self.april_tag_size.setValue(float(self.ctrl.camConfig.aprilTagSize))
        april_layout.addWidget(self.april_tag_size, 1)
        root.addWidget(april_row)
        root.addStretch(1)

        self.streamOrImgCombo.currentTextChanged.connect(self.sourceUpdate)
        self.april_tag_size.valueChanged.connect(self.setAprilTagSize)

        self.scanForCameras()
        self._refresh_camera_combo_values()
        self.sync_labels()

    def _build_camera_source(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        self.startStreamButton = QPushButton("Start Stream")
        self.selectCameraCombo = QComboBox()
        top.addWidget(self.startStreamButton)
        top.addWidget(self.selectCameraCombo, 1)
        layout.addLayout(top)
        self.vimba_controls_frame = self._build_vimba_controls()
        layout.addWidget(self.vimba_controls_frame)
        self.startStreamButton.clicked.connect(self.toggle_stream)
        self.selectCameraCombo.currentTextChanged.connect(self.selectCamera)
        return page

    def _build_static_source(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        self.singleImageFolderSelect = QPushButton("Select Image")
        self.singleImageTextButton = QPushButton("No Image Selected")
        layout.addWidget(self.singleImageFolderSelect)
        layout.addWidget(self.singleImageTextButton, 1)
        self.singleImageFolderSelect.clicked.connect(self.selectImagesFilepath)
        self.singleImageTextButton.clicked.connect(self.toggle_stream)
        return page

    def _build_folder_source(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        self.multiImageFolderSelect = QPushButton("Select Image Folder")
        self.multiImageTextButton = QPushButton("No Folder Selected")
        layout.addWidget(self.multiImageFolderSelect)
        layout.addWidget(self.multiImageTextButton, 1)
        self.multiImageFolderSelect.clicked.connect(self.selectImagesFilepath)
        self.multiImageTextButton.clicked.connect(self.toggle_stream)
        return page

    def _build_vimba_controls(self) -> QGroupBox:
        group = QGroupBox("Alvium Controls")
        form = QFormLayout(group)
        self.vimba_profile_combo = QComboBox()
        self.vimba_profile_combo.addItems(self.VIMBA_PROFILES)
        self.vimba_gain_auto_combo = QComboBox()
        self.vimba_gain_auto_combo.addItems(["Off", "Once", "Continuous"])
        self.vimba_gain_entry = QLineEdit()
        self.vimba_exposure_auto_combo = QComboBox()
        self.vimba_exposure_auto_combo.addItems(["Off", "Once", "Continuous"])
        self.vimba_exposure_entry = QLineEdit()
        form.addRow("Profile", self.vimba_profile_combo)
        form.addRow("Gain auto", self.vimba_gain_auto_combo)
        form.addRow("Gain", self.vimba_gain_entry)
        form.addRow("Exposure auto", self.vimba_exposure_auto_combo)
        form.addRow("Exposure (µs)", self.vimba_exposure_entry)

        buttons = QWidget()
        button_layout = QHBoxLayout(buttons)
        button_layout.setContentsMargins(0, 0, 0, 0)
        self.vimba_read_button = QPushButton("Read Camera")
        self.vimba_save_button = QPushButton("Save Settings")
        button_layout.addWidget(self.vimba_read_button)
        button_layout.addWidget(self.vimba_save_button)
        form.addRow(buttons)
        self.vimba_status_label = QLabel("Alvium controls apply when the Vimba stream starts.")
        self.vimba_status_label.setWordWrap(True)
        form.addRow(self.vimba_status_label)

        self.vimba_profile_combo.currentTextChanged.connect(self._on_vimba_profile_changed)
        self.vimba_gain_auto_combo.currentTextChanged.connect(self._on_vimba_mode_changed)
        self.vimba_exposure_auto_combo.currentTextChanged.connect(self._on_vimba_mode_changed)
        self.vimba_gain_entry.editingFinished.connect(self.save_vimba_controls)
        self.vimba_exposure_entry.editingFinished.connect(self.save_vimba_controls)
        self.vimba_read_button.clicked.connect(self.read_vimba_controls)
        self.vimba_save_button.clicked.connect(self.save_vimba_controls)
        return group

    @staticmethod
    def _running_value(controller: FilepathController) -> bool:
        value = getattr(controller, "stream_running_var", False)
        return bool(value.get() if hasattr(value, "get") else value)

    def toggle_stream(self) -> None:
        if not self._running_value(self.ctrl) and self._should_show_vimba_controls():
            self.save_vimba_controls()
        self.update_buttonsForStream(bool(self.ctrl.startStreamToggle()))

    def update_buttonsForStream(self, running: bool) -> None:
        self.startStreamButton.setText("Stop Stream" if running else "Start Stream")
        self.startStreamButton.setProperty("running", running)
        self.startStreamButton.style().unpolish(self.startStreamButton)
        self.startStreamButton.style().polish(self.startStreamButton)
        self.selectCameraCombo.setEnabled(not running)
        self.streamOrImgCombo.setEnabled(not running)
        self._set_vimba_controls_enabled(not running)

    def sourceUpdate(self, source: str) -> None:
        self.ctrl.camConfig.imageSource = ImageSource(source)
        self._sync_source_controls()
        self.ctrl.saveToCache()

    def _sync_source_controls(self) -> None:
        """Refresh source widgets from the model without persisting it."""
        source_to_index = {
            ImageSource.Camera_Stream: 0,
            ImageSource.Static_Image: 1,
            ImageSource.Stream_from_Folder: 2,
        }
        self.source_stack.setCurrentIndex(source_to_index[self.ctrl.camConfig.imageSource])
        if self.ctrl.camConfig.imageSource == ImageSource.Camera_Stream:
            self.scanForCameras()
            self._refresh_camera_combo_values()
            self._restore_selected_camera_combo()
        self.vimba_controls_frame.setVisible(self._should_show_vimba_controls())
        self._sync_source_labels()

    def _sync_source_labels(self) -> None:
        image_path = getattr(self.ctrl.camConfig, "imageFilepath", None)
        self.singleImageTextButton.setText(Path(image_path).name if image_path else "No Image Selected")
        self.multiImageTextButton.setText(Path(image_path).parent.name if image_path else "No Folder Selected")

    def selectImagesFilepath(self) -> None:
        current = getattr(self.ctrl.camConfig, "imageFilepath", None)
        initial = str(Path(current).parent if current else Path(self.default_filepath))
        if self.ctrl.camConfig.imageSource == ImageSource.Stream_from_Folder:
            selected = QFileDialog.getExistingDirectory(self, "Select Image Folder", initial)
            if selected:
                candidates = sorted(
                    path for path in Path(selected).iterdir()
                    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
                )
                self.ctrl.camConfig.imageFilepath = str(candidates[0]) if candidates else selected
        else:
            selected, _ = QFileDialog.getOpenFileName(self, "Select Image", initial)
            if selected:
                self.ctrl.camConfig.imageFilepath = selected
        if selected:
            self._sync_source_labels()
            self.ctrl.saveToCache()

    def selectSaveFolder(self) -> None:
        initial = str(Path(self.ctrl.camConfig.saveFolder or self.default_filepath))
        selected = QFileDialog.getExistingDirectory(self, "Select Folder For Saving", initial)
        if selected:
            self.ctrl.camConfig.saveFolder = selected
            self.ctrl.saveToCache(immediate=True)
            self.ctrl.loadFromCache()
            self.save_row.set_path(selected)

    def selectConfigFile(self) -> None:
        initial = str(Path.cwd() / "Configs")
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Select or create YAML config",
            initial,
            "YAML (*.yaml *.yml);;All files (*)",
            options=QFileDialog.Option.DontConfirmOverwrite,
        )
        if not selected:
            return
        if not Path(selected).suffix:
            selected += ".yaml"
        selected_path = Path(selected)

        # Parse an existing file before changing the active path.  A malformed
        # config must not partially replace the currently active model.
        payload = None
        if selected_path.exists():
            try:
                with selected_path.open("r", encoding="utf-8") as config_file:
                    payload = safe_load(config_file) or {}
                if not isinstance(payload, dict):
                    raise ValueError("The YAML root must be a mapping.")
            except (OSError, YAMLError, ValueError) as error:
                QMessageBox.critical(
                    self,
                    "Could Not Load Config",
                    f"The selected configuration was not changed.\n\n{error}",
                )
                return

        # Prevent an already scheduled save for the previous model from firing
        # after configFilepath changes and overwriting the selected file.
        pending_save = getattr(self.ctrl, "_save_debounce_id", None)
        if pending_save is not None:
            try:
                self.ctrl.after_cancel(pending_save)
            except (AttributeError, RuntimeError, TypeError):
                pass
            self.ctrl._save_debounce_id = None

            current_path_text = str(
                getattr(self.ctrl.camConfig, "configFilepath", "") or ""
            )
            if current_path_text:
                current_path = Path(current_path_text).resolve(strict=False)
                if current_path != selected_path.resolve(strict=False):
                    # Complete the previous config's pending auto-save while
                    # its filepath is still active, then switch models.
                    self.ctrl.saveToCache(immediate=True)

        was_loading = bool(getattr(self.ctrl, "_loading_config", False))
        self.ctrl._loading_config = True
        try:
            if payload is not None:
                self.ctrl.camConfig.fromDict(payload)
                # Config files copied or renamed on disk may contain their old
                # path.  The file selected by the user is authoritative.
                self.ctrl.camConfig.configFilepath = str(selected_path)
                self.ctrl.update_post_newCamConfig()
            else:
                self.ctrl.camConfig.configFilepath = str(selected_path)
                selected_path.parent.mkdir(parents=True, exist_ok=True)
                with selected_path.open("w", encoding="utf-8") as config_file:
                    dump(self.ctrl.camConfig.toDict, config_file)
            self.sync_labels()
        finally:
            self.ctrl._loading_config = was_loading

    def loadCalibration(self) -> None:
        current = getattr(self.ctrl.camConfig, "calibFilepath", None)
        initial = str(Path(current).parent if current else Path(self.default_filepath))
        selected, _ = QFileDialog.getOpenFileName(self, "Select Calibration File", initial)
        if selected:
            self.ctrl.camConfig.calibFilepath = selected
            self.ctrl.ingestCalibration()
            self.calibration_row.set_path(selected)

    def select3DTruthFile(self) -> None:
        initial = str(Path(self.ctrl.camConfig.ThreeDTruthFilepath or self.default_filepath))
        selected, _ = QFileDialog.getOpenFileName(self, "Select 3D Truth Points", initial)
        if selected:
            self.ctrl.camConfig.ThreeDTruthFilepath = selected
            self.ctrl.loadTruthPoints()
            self.ctrl.saveToCache()
            self.truth_row.set_path(selected)

    def selectYoloFolder(self) -> None:
        initial = str(Path(self.ctrl.camConfig.yoloFilepath or Path.cwd()))
        selected = QFileDialog.getExistingDirectory(self, "Select YOLO Folder", initial)
        if self.ctrl.pnpDrawer is not None:
            self.ctrl.pnpDrawer.last_q_vec = None
            self.ctrl.pnpDrawer.last_t_vec = None
        if selected:
            self.ctrl.camConfig.yoloFilepath = selected
            self.ctrl.updateYOLOModel()
            self.ctrl.saveToCache()
            self.yolo_row.set_path(selected)

    def selectLogFile(self) -> None:
        initial = str(Path(self.ctrl.camConfig.hud_data_filepath or self.default_filepath))
        selected = QFileDialog.getExistingDirectory(self, "Select Flight Log Data", initial)
        if selected:
            self.ctrl.camConfig.hud_data_filepath = selected
            self.ctrl.updateLogFile()
            self.log_row.set_path(selected)

    def setAprilTagSize(self, value: float) -> None:
        self.ctrl.camConfig.aprilTagSize = float(value)
        self.ctrl.saveToCache()

    def sync_labels(self) -> None:
        source = self.ctrl.camConfig.imageSource.value
        with QSignalBlocker(self.streamOrImgCombo):
            self.streamOrImgCombo.setCurrentText(source)
        self._sync_source_controls()
        self.config_row.set_path(self.ctrl.camConfig.configFilepath)
        self.save_row.set_path(self.ctrl.camConfig.saveFolder)
        self.calibration_row.set_path(self.ctrl.camConfig.calibFilepath)
        self.truth_row.set_path(self.ctrl.camConfig.ThreeDTruthFilepath, "No Truth Loaded")
        self.yolo_row.set_path(self.ctrl.camConfig.yoloFilepath)
        self.log_row.set_path(self.ctrl.camConfig.hud_data_filepath, "No Flight Log Loaded")
        with QSignalBlocker(self.april_tag_size):
            self.april_tag_size.setValue(float(self.ctrl.camConfig.aprilTagSize))
        self._sync_vimba_controls_from_model()

    def _ensure_vimba_control_defaults(self) -> None:
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
        return {
            "off": "Off",
            "once": "Once",
            "continuous": "Continuous",
            "manual": "Off",
            "false": "Off",
            "true": "Continuous",
        }.get(str(value or "Off").strip().lower(), "Off")

    @staticmethod
    def _normalize_vimba_profile_name(value: Any) -> str:
        text = str(value or "Full Res").strip()
        return {"Bin To 1440": "BinSum To 1440", "Bin To 864": "BinSum To 864"}.get(text, text)

    def _should_show_vimba_controls(self) -> bool:
        return (
            self.ctrl.camConfig.imageSource == ImageSource.Camera_Stream
            and bool(getattr(self.ctrl.camConfig, "use_vimba", False))
        )

    def _set_vimba_controls_enabled(self, not_running: bool) -> None:
        self.vimba_read_button.setEnabled(not_running)
        self.vimba_profile_combo.setEnabled(not_running)
        self.vimba_save_button.setEnabled(True)
        self._refresh_vimba_manual_widgets()

    def _refresh_vimba_manual_widgets(self) -> None:
        self.vimba_gain_entry.setEnabled(self.vimba_gain_auto_combo.currentText() == "Off")
        self.vimba_exposure_entry.setEnabled(self.vimba_exposure_auto_combo.currentText() == "Off")

    def _sync_vimba_controls_from_model(self) -> None:
        self._ensure_vimba_control_defaults()
        widgets = (
            self.vimba_profile_combo,
            self.vimba_gain_auto_combo,
            self.vimba_exposure_auto_combo,
            self.vimba_gain_entry,
            self.vimba_exposure_entry,
        )
        blockers = [QSignalBlocker(widget) for widget in widgets]
        self.vimba_profile_combo.setCurrentText(
            self._normalize_vimba_profile_name(self.ctrl.camConfig.vimba_profile)
        )
        self.vimba_gain_auto_combo.setCurrentText(
            self._normalize_vimba_auto_mode(self.ctrl.camConfig.vimba_gain_auto)
        )
        self.vimba_exposure_auto_combo.setCurrentText(
            self._normalize_vimba_auto_mode(self.ctrl.camConfig.vimba_exposure_auto)
        )
        self.vimba_gain_entry.setText(str(self.ctrl.camConfig.vimba_gain))
        self.vimba_exposure_entry.setText(str(self.ctrl.camConfig.vimba_exposure_us))
        del blockers
        self._refresh_vimba_manual_widgets()
        self.vimba_controls_frame.setVisible(self._should_show_vimba_controls())

    def _on_vimba_mode_changed(self, _text: str = "") -> None:
        self._refresh_vimba_manual_widgets()
        self.save_vimba_controls()

    def _on_vimba_profile_changed(self, profile: str) -> None:
        descriptions = {
            "Full Res": "Full sensor, no preview downscale.",
            "Zoom 1440": "Centered 1440 ROI. Applies on next stream start.",
            "BinSum To 1440": "2×2 digital sum binning. Applies on next stream start.",
            "BinAvg To 1440": "2×2 digital average binning. Applies on next stream start.",
            "Zoom 864": "Centered 864 ROI. Applies on next stream start.",
            "BinSum To 864": "2×2 digital sum binning. Applies on next stream start.",
            "BinAvg To 864": "2×2 digital average binning. Applies on next stream start.",
        }
        self.vimba_status_label.setText(descriptions.get(profile, "Vimba profile updated."))
        self.save_vimba_controls()

    def save_vimba_controls(self) -> None:
        self.ctrl.camConfig.vimba_profile = self._normalize_vimba_profile_name(
            self.vimba_profile_combo.currentText()
        )
        self.ctrl.camConfig.vimba_gain_auto = self._normalize_vimba_auto_mode(
            self.vimba_gain_auto_combo.currentText()
        )
        self.ctrl.camConfig.vimba_exposure_auto = self._normalize_vimba_auto_mode(
            self.vimba_exposure_auto_combo.currentText()
        )
        try:
            self.ctrl.camConfig.vimba_gain = float(self.vimba_gain_entry.text())
        except ValueError:
            self.vimba_gain_entry.setText(str(self.ctrl.camConfig.vimba_gain))
        try:
            self.ctrl.camConfig.vimba_exposure_us = float(self.vimba_exposure_entry.text())
        except ValueError:
            self.vimba_exposure_entry.setText(str(self.ctrl.camConfig.vimba_exposure_us))
        self.ctrl.saveToCache(immediate=True)
        live = self.ctrl.queue_live_vimba_update({
            "vimba_gain_auto": self.ctrl.camConfig.vimba_gain_auto,
            "vimba_gain": self.ctrl.camConfig.vimba_gain,
            "vimba_exposure_auto": self.ctrl.camConfig.vimba_exposure_auto,
            "vimba_exposure_us": self.ctrl.camConfig.vimba_exposure_us,
        })
        prefix = "Gain/exposure sent live." if live else "Alvium settings saved."
        self.vimba_status_label.setText(
            f"{prefix} Profile '{self.ctrl.camConfig.vimba_profile}' applies on the next stream start."
        )
        self._refresh_vimba_manual_widgets()

    @staticmethod
    def _try_get_camera_feature(camera, *names: str):
        for name in names:
            feature = getattr(camera, name, None)
            if feature is not None:
                return feature
        return None

    def read_vimba_controls(self) -> None:
        if not HAVE_VMBPY:
            QMessageBox.critical(self, "VmbPy Missing", "VmbPy is not installed.")
            return
        meta = self.cameraChoices.get(self.selectCameraCombo.currentText())
        if not meta or meta.get("backend") != "vimba":
            QMessageBox.information(self, "No Alvium Selected", "Select a Vimba/Alvium camera first.")
            return
        try:
            with VmbSystem.get_instance() as vmb:
                camera = vmb.get_camera_by_id(str(meta["camera_id"]))
                with camera:
                    gain_auto = self._try_get_camera_feature(camera, "GainAuto")
                    gain = self._try_get_camera_feature(camera, "Gain")
                    exposure_auto = self._try_get_camera_feature(camera, "ExposureAuto")
                    exposure = self._try_get_camera_feature(camera, "ExposureTime", "ExposureTimeAbs")
                    if gain_auto is not None:
                        self.vimba_gain_auto_combo.setCurrentText(self._normalize_vimba_auto_mode(gain_auto.get()))
                    if gain is not None:
                        self.vimba_gain_entry.setText(str(gain.get()))
                    if exposure_auto is not None:
                        self.vimba_exposure_auto_combo.setCurrentText(
                            self._normalize_vimba_auto_mode(exposure_auto.get())
                        )
                    if exposure is not None:
                        self.vimba_exposure_entry.setText(str(exposure.get()))
        except (AttributeError, RuntimeError) as error:
            LOG.warning(f"Could not read Vimba camera controls: {error}")
            QMessageBox.critical(self, "Read Failed", str(error))
            return
        self.save_vimba_controls()

    def scanForCameras(self) -> None:
        self.cameraChoices = {}
        try:
            from cv2_enumerate_cameras import enumerate_cameras

            for info in enumerate_cameras(cv2.CAP_DSHOW):
                self.cameraChoices[f"OpenCV: {info.name} [{info.index}]"] = {
                    "backend": "opencv",
                    "cam_index": int(info.index),
                    "display_name": info.name,
                }
        except (ImportError, cv2.error, OSError) as error:
            LOG.warning(f"Could not enumerate OpenCV cameras: {error}")
        if HAVE_VMBPY:
            try:
                with VmbSystem.get_instance() as vmb:
                    for camera in vmb.get_all_cameras():
                        camera_id = camera.get_id()
                        serial = camera.get_serial()
                        name = camera.get_name()
                        self.cameraChoices[f"Vimba: {name} [{serial or camera_id}]"] = {
                            "backend": "vimba",
                            "camera_id": camera_id,
                            "camera_serial": serial,
                            "display_name": name,
                        }
            except (AttributeError, RuntimeError) as error:
                LOG.warning(f"Could not enumerate Vimba cameras: {error}")

    def _refresh_camera_combo_values(self) -> None:
        selected = self.selectCameraCombo.currentText()
        with QSignalBlocker(self.selectCameraCombo):
            self.selectCameraCombo.clear()
            self.selectCameraCombo.addItems(
                list(self.cameraChoices) if self.cameraChoices else ["No Cameras Found"]
            )
            self.selectCameraCombo.setEnabled(bool(self.cameraChoices))
            if selected in self.cameraChoices:
                self.selectCameraCombo.setCurrentText(selected)

    def _preferred_camera_key_from_model(self) -> str | None:
        if bool(getattr(self.ctrl.camConfig, "use_vimba", False)):
            wanted = str(getattr(self.ctrl.camConfig, "vimba_camera_id", "") or "")
            for key, meta in self.cameraChoices.items():
                if meta.get("backend") == "vimba" and meta.get("camera_id") == wanted:
                    return key
        wanted_index = getattr(self.ctrl.camConfig, "cam_index", None)
        for key, meta in self.cameraChoices.items():
            if meta.get("backend") == "opencv" and meta.get("cam_index") == wanted_index:
                return key
        return next(iter(self.cameraChoices), None)

    def _restore_selected_camera_combo(self) -> None:
        key = self._preferred_camera_key_from_model()
        self._last_valid_camera_key = key
        if key is not None:
            with QSignalBlocker(self.selectCameraCombo):
                self.selectCameraCombo.setCurrentText(key)

    @staticmethod
    def _probe_camera_choice(meta: dict[str, Any]) -> tuple[bool, str | None]:
        if meta.get("backend") == "opencv":
            index = int(meta["cam_index"])
            capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            try:
                if not capture.isOpened():
                    return False, f"OpenCV camera index {index} could not be opened."
                ok, _ = capture.read()
                return (True, None) if ok else (False, f"Camera {index} returned no frame.")
            finally:
                capture.release()
        if meta.get("backend") == "vimba":
            if not HAVE_VMBPY:
                return False, "VmbPy is not installed."
            try:
                with VmbSystem.get_instance() as vmb:
                    with vmb.get_camera_by_id(meta["camera_id"]):
                        pass
                return True, None
            except (AttributeError, RuntimeError) as error:
                return False, str(error)
        return False, f"Unknown camera backend: {meta.get('backend')}"

    def selectCamera(self, key: str) -> None:
        if self._suppress_camera_callback or key not in self.cameraChoices:
            return
        meta = self.cameraChoices[key]
        ok, error = self._probe_camera_choice(meta)
        if not ok:
            QMessageBox.critical(self, "Camera Connection Failed", error or "Could not open camera.")
            if self._last_valid_camera_key:
                with QSignalBlocker(self.selectCameraCombo):
                    self.selectCameraCombo.setCurrentText(self._last_valid_camera_key)
            return
        if meta["backend"] == "opencv":
            self.ctrl.camConfig.cam_index = int(meta["cam_index"])
            self.ctrl.camConfig.use_vimba = False
            self.ctrl.camConfig.vimba_camera_id = ""
        else:
            self.ctrl.camConfig.use_vimba = True
            self.ctrl.camConfig.vimba_camera_id = str(meta["camera_id"])
        self._last_valid_camera_key = key
        self._sync_vimba_controls_from_model()
        self.ctrl.saveToCache(immediate=True)


Filepath_page = FilepathPage
