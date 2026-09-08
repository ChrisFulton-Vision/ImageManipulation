from __future__ import annotations

from typing import Protocol, Any

from PySide6.QtCore import Qt, QSignalBlocker
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QSlider,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class ImageProcessingController(Protocol):
    camConfig: Any

    def saveToCache(self, immediate: bool = False) -> None: ...


class FloatSlider(QWidget):
    """Integer-backed slider with an exact floating-point spin box."""

    def __init__(
        self,
        minimum: float,
        maximum: float,
        value: float,
        *,
        decimals: int = 2,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._scale = 10**decimals
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(round(minimum * self._scale), round(maximum * self._scale))
        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(decimals)
        self.spin.setRange(minimum, maximum)
        self.spin.setSingleStep(1 / self._scale)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        layout.addWidget(self.slider)
        layout.addWidget(self.spin)

        self.slider.valueChanged.connect(self._slider_changed)
        self.spin.valueChanged.connect(self._spin_changed)
        self.setValue(value)

    @property
    def valueChanged(self):
        return self.spin.valueChanged

    def value(self) -> float:
        return float(self.spin.value())

    def setValue(self, value: float) -> None:
        value = max(self.spin.minimum(), min(self.spin.maximum(), float(value)))
        with QSignalBlocker(self.spin), QSignalBlocker(self.slider):
            self.spin.setValue(value)
            self.slider.setValue(round(value * self._scale))

    def _slider_changed(self, value: int) -> None:
        with QSignalBlocker(self.spin):
            self.spin.setValue(value / self._scale)
        self.spin.valueChanged.emit(self.spin.value())

    def _spin_changed(self, value: float) -> None:
        with QSignalBlocker(self.slider):
            self.slider.setValue(round(value * self._scale))


class ImageProcessingPage(QWidget):
    """Native Qt controls for YOLO confidence and IOU thresholds."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        controller: ImageProcessingController,
    ) -> None:
        super().__init__(parent)
        self.ctrl = controller

        self.conf_control = FloatSlider(
            0.15,
            1.0,
            float(controller.camConfig.yolo_conf),
            decimals=2,
        )
        self.iou_control = FloatSlider(
            0.0,
            1.0,
            float(controller.camConfig.yolo_iou),
            decimals=2,
        )

        self._root_layout = QVBoxLayout(self)
        self._root_layout.setContentsMargins(8, 8, 8, 8)
        self._root_layout.setSpacing(8)

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.addRow(QLabel("Confidence"), self.conf_control)
        form.addRow(QLabel("IOU"), self.iou_control)
        self._root_layout.addLayout(form)

        self._queue_editor: QWidget | None = None

        self.conf_control.valueChanged.connect(self._on_conf_changed)
        self.iou_control.valueChanged.connect(self._on_iou_changed)

        # Compatibility names used by the existing controller/runtime code.
        self.confSliderBar = self.conf_control
        self.iouSliderBar = self.iou_control

    def _on_conf_changed(self, value: float) -> None:
        self.ctrl.camConfig.yolo_conf = float(value)
        self.ctrl.saveToCache()

    def _on_iou_changed(self, value: float) -> None:
        self.ctrl.camConfig.yolo_iou = float(value)
        self.ctrl.saveToCache()

    def sync_from_model(self) -> None:
        self.conf_control.setValue(float(self.ctrl.camConfig.yolo_conf))
        self.iou_control.setValue(float(self.ctrl.camConfig.yolo_iou))

    def set_queue_editor(self, editor: QWidget) -> None:
        """Install the queue/options splitter as the page's expanding region."""
        if self._queue_editor is editor:
            return
        if self._queue_editor is not None:
            self._root_layout.removeWidget(self._queue_editor)
            self._queue_editor.setParent(None)

        self._queue_editor = editor
        editor.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self._root_layout.addWidget(editor, 1)


ImageProcessing_page = ImageProcessingPage
