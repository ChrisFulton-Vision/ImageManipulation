from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Any, Tuple, Union, Type, ClassVar
from support.core.enums import YoloInferenceSource, check_if_enum, Type_enum

from PySide6.QtCore import Qt, QSignalBlocker, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

DEFAULT_CHOICE = "(select)"

ROW_H = 36  # fixed row height
DROPDOWN_W = 120
DROPDOWN_H = 32  # fixed optionmenu height
ICON_BTN_W = 36
ARGS_VALUE_W = 220

Args = Dict[str, Any]
ArgType = Union[Type[bool], Type[int], Type[float], Type[str], Type_enum]
StepFn = Callable[..., None]
StepSpec = Tuple[StepFn, Args]


@dataclass(frozen=True)
class ArgSpec:
    name: str
    typ: ArgType | tuple[ArgType, ...]
    default: Any
    min: float | None = None
    max: float | None = None
    path_kind: str | None = None
    editor: str = "auto"
    decimals: int | None = None


@dataclass(frozen=True, slots=True)
class ArgBinding:
    label: str  # GUI key / display name
    field: str  # dataclass attribute name
    object_type: type
    default: object
    min: float | None = None
    max: float | None = None


@dataclass(slots=True)
class UndistortOpts:
    cubemap: bool = False
    BINDINGS: ClassVar[tuple[ArgBinding, ...]] = (
        ArgBinding("Cubemap from Fisheye", "cubemap", bool, False),
    )

    # Derived, guaranteed consistent
    ARG_SPECS: ClassVar[tuple["ArgSpec", ...]] = tuple(
        ArgSpec(b.label, b.object_type, b.default, b.min, b.max) for b in BINDINGS
    )
    KEYMAP: ClassVar[dict[str, str]] = {b.label: b.field for b in BINDINGS}


@dataclass(slots=True)
class CalibrationAdjustOpts:
    """Additive changes relative to the most recently loaded calibration."""

    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0

    COMMON_BINDINGS: ClassVar[tuple[ArgBinding, ...]] = (
        ArgBinding("Δfx from loaded (px)", "fx", float, 0.0),
        ArgBinding("Δfy from loaded (px)", "fy", float, 0.0),
        ArgBinding("Δcx from loaded (px)", "cx", float, 0.0),
        ArgBinding("Δcy from loaded (px)", "cy", float, 0.0),
    )
    BROWN_CONRADY_BINDINGS: ClassVar[tuple[ArgBinding, ...]] = (
        ArgBinding("Δk1 (Brown-Conrady)", "k1", float, 0.0),
        ArgBinding("Δk2 (Brown-Conrady)", "k2", float, 0.0),
        ArgBinding("Δp1 (Brown-Conrady)", "p1", float, 0.0),
        ArgBinding("Δp2 (Brown-Conrady)", "p2", float, 0.0),
        ArgBinding("Δk3 (Brown-Conrady)", "k3", float, 0.0),
    )
    FISHEYE_BINDINGS: ClassVar[tuple[ArgBinding, ...]] = (
        ArgBinding("Δk1 (fisheye)", "k1", float, 0.0),
        ArgBinding("Δk2 (fisheye)", "k2", float, 0.0),
        ArgBinding("Δk3 (fisheye)", "k3", float, 0.0),
        ArgBinding("Δk4 (fisheye)", "k4", float, 0.0),
    )

    ALL_BINDINGS: ClassVar[tuple[ArgBinding, ...]] = (
        *COMMON_BINDINGS,
        *BROWN_CONRADY_BINDINGS,
        *FISHEYE_BINDINGS,
    )
    KEYMAP: ClassVar[dict[str, str]] = {b.label: b.field for b in ALL_BINDINGS}

    @classmethod
    def arg_specs(cls, *, fisheye: bool) -> tuple["ArgSpec", ...]:
        distortion = cls.FISHEYE_BINDINGS if fisheye else cls.BROWN_CONRADY_BINDINGS
        return tuple(
            ArgSpec(
                b.label,
                b.object_type,
                b.default,
                b.min,
                b.max,
                editor="entry",
                decimals=10 if b.field.startswith(("k", "p")) else 4,
            )
            for b in (*cls.COMMON_BINDINGS, *distortion)
        )


@dataclass(slots=True)
class ResizeOpts:
    scale: float = 1.0
    pixelNum: int = 864
    BINDINGS: ClassVar[tuple[ArgBinding, ...]] = (
        ArgBinding("Exp Scale Image", "scale", float, 1.0, 0.05, 1.5),
        ArgBinding("Maximum Pixel Number", "pixelNum", int, 864, 1, 2848),
    )

    # Derived, guaranteed consistent
    ARG_SPECS: ClassVar[tuple["ArgSpec", ...]] = tuple(
        ArgSpec(b.label, b.object_type, b.default, b.min, b.max) for b in BINDINGS
    )
    KEYMAP: ClassVar[dict[str, str]] = {b.label: b.field for b in BINDINGS}


@dataclass(slots=True)
class AprilTagDetectOpts:
    scale: float = 1.0
    inpaint: bool = False
    pnp: bool = False
    qnp: bool = False
    BINDINGS: ClassVar[tuple[ArgBinding, ...]] = (
        ArgBinding("Scale", "scale", float, 1.0),
        ArgBinding("Hide April Tags", "inpaint", bool, True),
        ArgBinding("PnP from Truth", "pnp", bool, True),
        ArgBinding("QnP from Truth", "qnp", bool, True),
    )

    # Derived, guaranteed consistent
    ARG_SPECS: ClassVar[tuple["ArgSpec", ...]] = tuple(
        ArgSpec(b.label, b.object_type, b.default, b.min, b.max) for b in BINDINGS
    )
    KEYMAP: ClassVar[dict[str, str]] = {b.label: b.field for b in BINDINGS}


@dataclass(slots=True)
class YoloOpts:
    want_pnp: bool = False
    want_qnp: bool = False
    want_wqnp_yolo: bool = False
    want_wqnp_kfest: bool = False
    factor_graph: bool = False
    hyper_focus: bool = False
    feature_circles: bool = False
    inference_source: YoloInferenceSource = YoloInferenceSource.ORIGINAL
    display_feature_ids: str = ""
    model_folder: str = ""
    sigma_proc: float = 0.5
    BINDINGS: ClassVar[tuple[ArgBinding, ...]] = (
        ArgBinding("PnP", "want_pnp", bool, False),
        ArgBinding("QnP", "want_qnp", bool, False),
        ArgBinding("wQnP_yolo", "want_wqnp_yolo", bool, False),
        ArgBinding("wQnP_KFest", "want_wqnp_kfest", bool, False),
        ArgBinding("Factor Graph", "factor_graph", bool, False),
        ArgBinding("Hyper Attention", "hyper_focus", bool, False),
        ArgBinding("Feature Circles", "feature_circles", bool, False),
        ArgBinding("Inference Source", "inference_source", YoloInferenceSource, YoloInferenceSource.ORIGINAL),
        ArgBinding("Sigma Proc", "sigma_proc", float, 0.5, 0.00001, 10.0),
    )

    # Derived, guaranteed consistent
    ARG_SPECS: ClassVar[tuple["ArgSpec", ...]] = (
        *(ArgSpec(b.label, b.object_type, b.default, b.min, b.max) for b in BINDINGS),
        ArgSpec("Display Features", str, ""),
        ArgSpec("YOLO Folder", str, "", path_kind="directory"),
    )
    KEYMAP: ClassVar[dict[str, str]] = {b.label: b.field for b in BINDINGS}
    KEYMAP["Display Features"] = "display_feature_ids"
    KEYMAP["YOLO Folder"] = "model_folder"


@dataclass(slots=True)
class HudOpts:
    store_attitude: bool = True
    mode_channel: int = 10
    reverse_throttle_pwm: bool = False
    map_transparency: float = 0.35
    draw_attitude: bool = True
    draw_as_alt: bool = True
    draw_title: bool = True
    draw_crosshairs: bool = True
    draw_mode: bool = True
    BINDINGS: ClassVar[tuple[ArgBinding, ...]] = (
        ArgBinding("Store Attitude for FG", "store_attitude", bool, True),
        ArgBinding("Mode Channel", "mode_channel", int, 10),
        ArgBinding("Reverse Throttle PWM", "reverse_throttle_pwm", bool, False),
        ArgBinding("Map Alpha", "map_transparency", float, 0.35, 0.0, 1.0),
        ArgBinding("Attitude", "draw_attitude", bool, True),
        ArgBinding("Airspeed/Alt", "draw_as_alt", bool, True),
        ArgBinding("Image Name", "draw_title", bool, True),
        ArgBinding("Crosshairs", "draw_crosshairs", bool, True),
        ArgBinding("Control Mode", "draw_mode", bool, True),
    )

    ARG_SPECS: ClassVar[tuple["ArgSpec", ...]] = tuple(
        ArgSpec(b.label, b.object_type, b.default, b.min, b.max) for b in BINDINGS
    )
    KEYMAP: ClassVar[dict[str, str]] = {b.label: b.field for b in BINDINGS}


_UNSET = object()


class Slot:
    def __init__(self):
        self._v = _UNSET

    def set(self, v):
        self._v = v

    def clear(self):
        self._v = _UNSET

    def is_set(self) -> bool:
        return self._v is not _UNSET

    def get(self):
        if self._v is _UNSET:
            raise KeyError("Slot is unset")
        return self._v

    def get_or(self, default=None):
        return default if self._v is _UNSET else self._v


@dataclass(slots=True)
class FrameCtx:
    img_time: Optional[float] = None
    name: Optional[str] = None
    display_in_realtime: bool = True
    undistorted: Slot = field(default_factory=Slot)
    yolo: Slot = field(default_factory=Slot)
    fg: Slot = field(default_factory=Slot)
    resize: Slot = field(default_factory=Slot)


@dataclass
class _QueueRow:
    frame: QFrame
    dropdown: QComboBox
    enabled_chk: QCheckBox
    args_btn: QPushButton
    up_btn: QPushButton
    down_btn: QPushButton
    remove_btn: QPushButton
    args: Args

    @property
    def label(self) -> str:
        return self.dropdown.currentText()


@dataclass(frozen=True)
class StepOption:
    label: str
    fn: StepFn
    arg_specs: tuple[ArgSpec, ...] = ()
    arg_specs_fn: Optional[Callable[[Args], tuple[ArgSpec, ...]]] = None
    keymap: Optional[Dict[str, str]] = None

    def get_arg_specs(self, args: Optional[Args] = None) -> tuple[ArgSpec, ...]:
        if self.arg_specs_fn is not None:
            return self.arg_specs_fn(args or {})
        return self.arg_specs

    @property
    def default_args(self) -> Args:
        return {spec.name: spec.default for spec in self.get_arg_specs({})}


class StepSpecQueueEditor(QWidget):
    """Native Qt editor for ``StepSpec = (callable, args)`` queues."""

    queueChanged = Signal(list)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        options: List[StepOption],
        on_change: Optional[Callable[[List[StepSpec]], None]] = None,
        initial: Optional[List[StepSpec]] = None,
        func_that_refits: Callable[..., Any] | None = None,
        **_kwargs: Any,
    ) -> None:
        super().__init__(parent)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.setMinimumHeight(300)
        if any(option.label == DEFAULT_CHOICE for option in options):
            raise ValueError(f"'{DEFAULT_CHOICE}' is reserved.")

        self._step_options = list(options)
        self._labels = [DEFAULT_CHOICE, *(option.label for option in options)]
        self._label_to_opt = {option.label: option for option in options}
        self._fn_to_opt = {option.fn: option for option in options}
        self._on_change = on_change
        self.func_that_refits = func_that_refits or (lambda: None)
        self._rows: List[_QueueRow] = []
        self._active_idx: int | None = None

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        root.addWidget(splitter)

        queue_widget = QWidget()
        queue_layout = QVBoxLayout(queue_widget)
        queue_layout.setContentsMargins(0, 0, 8, 0)
        queue_layout.addWidget(self._heading("Image Processing Queue"))
        self._rows_layout = QVBoxLayout()
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(4)
        queue_layout.addLayout(self._rows_layout)
        queue_layout.addStretch(1)

        queue_scroll = QScrollArea()
        queue_scroll.setWidgetResizable(True)
        queue_scroll.setFrameShape(QFrame.Shape.NoFrame)
        queue_scroll.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        queue_scroll.setWidget(queue_widget)

        args_widget = QWidget()
        args_layout = QVBoxLayout(args_widget)
        args_layout.setContentsMargins(8, 0, 0, 0)
        args_layout.addWidget(self._heading("Step Arguments"))
        self._args_hint = QLabel("Select a step to edit its arguments.")
        self._args_hint.setWordWrap(True)
        args_layout.addWidget(self._args_hint)
        self._args_body = QWidget()
        self._args_form = QFormLayout(self._args_body)
        self._args_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        args_layout.addWidget(self._args_body)
        args_layout.addStretch(1)

        args_scroll = QScrollArea()
        args_scroll.setWidgetResizable(True)
        args_scroll.setFrameShape(QFrame.Shape.NoFrame)
        args_scroll.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        args_scroll.setWidget(args_widget)

        splitter.addWidget(queue_scroll)
        splitter.addWidget(args_scroll)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([520, 360])

        if on_change is not None:
            self.queueChanged.connect(on_change)

        self.set_queue(initial or [], emit=False)

    @staticmethod
    def _heading(text: str) -> QLabel:
        label = QLabel(text)
        font = label.font()
        font.setBold(True)
        label.setFont(font)
        return label

    def get_queue(self) -> List[StepSpec]:
        queue: List[StepSpec] = []
        for row in self._rows:
            option = self._label_to_opt.get(row.label)
            if option is not None:
                args = dict(row.args)
                args["state"] = row.enabled_chk.isChecked()
                queue.append((option.fn, args))
        return queue

    def refresh_dynamic_args(self) -> None:
        """Rebuild the active argument form when an external schema changes."""
        self._render_args_panel()

    def set_queue(
        self,
        queue: Optional[List[StepSpec]],
        *,
        emit: bool = False,
        emit_change: bool | None = None,
    ) -> None:
        """Replace the displayed queue.

        ``emit_change`` preserves the public keyword used by ConfigRuntime and
        older callers. ``emit`` is retained as the shorter Qt-side spelling.
        When both are supplied, the compatibility keyword wins.
        """
        if emit_change is not None:
            emit = bool(emit_change)
        self._clear_rows()
        for fn, args in queue or []:
            option = self._fn_to_opt.get(fn)
            if option is None:
                continue
            self._add_row(option.label, dict(args))
        self._add_row(DEFAULT_CHOICE, {})
        self._active_idx = self._first_real_row_index()
        self._refresh_row_controls()
        self._render_args_panel()
        if emit:
            self._emit_change()

    def _clear_rows(self) -> None:
        for row in self._rows:
            self._rows_layout.removeWidget(row.frame)
            row.frame.deleteLater()
        self._rows.clear()

    def _add_row(self, selected: str, args: Args) -> _QueueRow:
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setMinimumHeight(ROW_H)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(5, 3, 5, 3)
        layout.setSpacing(5)

        dropdown = QComboBox()
        dropdown.addItems(self._labels)
        dropdown.setCurrentText(selected if selected in self._labels else DEFAULT_CHOICE)
        dropdown.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        enabled = QCheckBox("On")
        enabled.setChecked(bool(args.get("state", True)))
        args_button = QPushButton("⚙")
        up_button = QPushButton("↑")
        down_button = QPushButton("↓")
        remove_button = QPushButton("✕")
        for button in (args_button, up_button, down_button, remove_button):
            button.setFixedWidth(ICON_BTN_W)

        layout.addWidget(dropdown, 1)
        layout.addWidget(enabled)
        layout.addWidget(args_button)
        layout.addWidget(up_button)
        layout.addWidget(down_button)
        layout.addWidget(remove_button)
        self._rows_layout.addWidget(frame)

        option = self._label_to_opt.get(selected)
        row_args = dict(option.default_args) if option is not None else {}
        row_args.update(args)
        row = _QueueRow(
            frame,
            dropdown,
            enabled,
            args_button,
            up_button,
            down_button,
            remove_button,
            row_args,
        )
        self._rows.append(row)

        dropdown.currentTextChanged.connect(lambda _text, r=row: self._on_row_changed(r))
        enabled.toggled.connect(lambda checked, r=row: self._on_enabled_toggled(r, checked))
        args_button.clicked.connect(lambda _=False, r=row: self._set_active(r))
        up_button.clicked.connect(lambda _=False, r=row: self._move_row(r, -1))
        down_button.clicked.connect(lambda _=False, r=row: self._move_row(r, 1))
        remove_button.clicked.connect(lambda _=False, r=row: self._remove_row(r))
        return row

    def _on_enabled_toggled(self, row: _QueueRow, checked: bool) -> None:
        row.args["state"] = bool(checked)
        self._set_active(row, render=True)
        self._emit_change()

    def _on_row_changed(self, row: _QueueRow) -> None:
        option = self._label_to_opt.get(row.label)
        state = row.enabled_chk.isChecked()
        if option is None:
            row.args = {}
        else:
            expected = {spec.name for spec in option.get_arg_specs(row.args)}
            if (set(row.args) - {"state"}) != expected:
                row.args = dict(option.default_args)
            row.args["state"] = state

        self._set_active(row, render=False)
        self._ensure_placeholder()
        self._refresh_row_controls()
        self._render_args_panel()
        self._emit_change()

    def _set_active(self, row: _QueueRow, *, render: bool = True) -> None:
        try:
            self._active_idx = self._rows.index(row)
        except ValueError:
            return
        if render:
            self._render_args_panel()

    def _move_row(self, row: _QueueRow, direction: int) -> None:
        try:
            old_index = self._rows.index(row)
        except ValueError:
            return
        new_index = old_index + direction
        if not 0 <= new_index < len(self._rows):
            return
        if row.label == DEFAULT_CHOICE or self._rows[new_index].label == DEFAULT_CHOICE:
            return

        self._rows[old_index], self._rows[new_index] = self._rows[new_index], self._rows[old_index]
        self._rows_layout.removeWidget(row.frame)
        self._rows_layout.insertWidget(new_index, row.frame)
        self._active_idx = new_index
        self._refresh_row_controls()
        self._render_args_panel()
        self._emit_change()

    def _remove_row(self, row: _QueueRow) -> None:
        try:
            index = self._rows.index(row)
        except ValueError:
            return
        if row.label == DEFAULT_CHOICE:
            return
        self._rows.pop(index)
        self._rows_layout.removeWidget(row.frame)
        row.frame.deleteLater()
        self._ensure_placeholder()
        self._active_idx = self._first_real_row_index()
        self._refresh_row_controls()
        self._render_args_panel()
        self._emit_change()

    def _ensure_placeholder(self) -> None:
        if not self._rows or self._rows[-1].label != DEFAULT_CHOICE:
            self._add_row(DEFAULT_CHOICE, {})

    def _refresh_row_controls(self) -> None:
        placeholder_index = len(self._rows) - 1
        for index, row in enumerate(self._rows):
            placeholder = index == placeholder_index and row.label == DEFAULT_CHOICE
            row.enabled_chk.setEnabled(not placeholder)
            row.args_btn.setEnabled(not placeholder)
            row.remove_btn.setEnabled(not placeholder)
            row.up_btn.setEnabled(not placeholder and index > 0)
            row.down_btn.setEnabled(not placeholder and index < placeholder_index - 1)

    def _first_real_row_index(self) -> int | None:
        return next((i for i, row in enumerate(self._rows) if row.label != DEFAULT_CHOICE), None)

    def _emit_change(self) -> None:
        self.queueChanged.emit(self.get_queue())

    def _clear_form(self) -> None:
        while self._args_form.rowCount():
            self._args_form.removeRow(0)

    def _render_args_panel(self) -> None:
        self._clear_form()
        if self._active_idx is None or not 0 <= self._active_idx < len(self._rows):
            self._args_hint.setText("Select a step to edit its arguments.")
            return

        row = self._rows[self._active_idx]
        option = self._label_to_opt.get(row.label)
        if option is None:
            self._args_hint.setText("Select a step to edit its arguments.")
            return

        self._args_hint.setText(option.label)
        specs = option.get_arg_specs(row.args)
        if not specs:
            self._args_form.addRow(QLabel("(no args)"))
            return

        for spec in specs:
            value = row.args.get(spec.name, spec.default)
            self._args_form.addRow(spec.name, self._editor_for(spec, value))

    def _editor_for(self, spec: ArgSpec, value: Any) -> QWidget:
        if check_if_enum(value):
            combo = QComboBox()
            enum_type = type(value)
            combo.addItems(member.name for member in enum_type)
            combo.setCurrentText(value.name)
            combo.currentTextChanged.connect(
                lambda text, n=spec.name, et=enum_type: self._set_arg(n, et[text])
            )
            return combo

        if isinstance(value, bool):
            checkbox = QCheckBox()
            checkbox.setChecked(value)
            checkbox.toggled.connect(lambda checked, n=spec.name: self._set_arg(n, bool(checked)))
            return checkbox

        if isinstance(value, int) and not isinstance(value, bool):
            spin = QSpinBox()
            spin.setRange(
                int(spec.min) if spec.min is not None else -(2**31),
                int(spec.max) if spec.max is not None else 2**31 - 1,
            )
            spin.setValue(value)
            spin.valueChanged.connect(lambda new_value, n=spec.name: self._set_arg(n, int(new_value)))
            return spin

        if isinstance(value, float):
            spin = QDoubleSpinBox()
            spin.setDecimals(spec.decimals if spec.decimals is not None else 6)
            spin.setRange(
                float(spec.min) if spec.min is not None else -1.0e12,
                float(spec.max) if spec.max is not None else 1.0e12,
            )
            span = spin.maximum() - spin.minimum()
            spin.setSingleStep(max(10.0 ** -spin.decimals(), min(0.1, span / 100.0)))
            spin.setValue(value)
            spin.valueChanged.connect(lambda new_value, n=spec.name: self._set_arg(n, float(new_value)))
            return spin

        line_edit = QLineEdit(str(value))
        if spec.path_kind is None:
            line_edit.editingFinished.connect(
                lambda n=spec.name, edit=line_edit: self._set_arg(n, edit.text())
            )
            return line_edit

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(line_edit, 1)
        browse = QPushButton("Browse")
        layout.addWidget(browse)

        def choose_path() -> None:
            current = line_edit.text() or "."
            if spec.path_kind == "directory":
                selected = QFileDialog.getExistingDirectory(self, f"Select {spec.name}", current)
            else:
                selected, _ = QFileDialog.getOpenFileName(self, f"Select {spec.name}", current)
            if selected:
                line_edit.setText(selected)
                self._set_arg(spec.name, selected)

        line_edit.editingFinished.connect(lambda: self._set_arg(spec.name, line_edit.text()))
        browse.clicked.connect(choose_path)
        return container

    def _set_arg(self, name: str, value: Any) -> None:
        if self._active_idx is None or not 0 <= self._active_idx < len(self._rows):
            return
        row = self._rows[self._active_idx]
        option = self._label_to_opt.get(row.label)
        if option is None:
            return
        old_specs = option.get_arg_specs(row.args)
        spec = next((item for item in old_specs if item.name == name), None)
        if spec is None:
            return

        allowed = spec.typ if isinstance(spec.typ, tuple) else (spec.typ,)
        if not any(isinstance(value, candidate) for candidate in allowed if isinstance(candidate, type)):
            return
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if spec.min is not None:
                value = max(value, spec.min)
            if spec.max is not None:
                value = min(value, spec.max)
        row.args[name] = value

        # Some steps change their argument schema based on an enum selection.
        # Reconcile defaults and rebuild the form immediately when that schema
        # changes (for example Unfiltered -> Gabor).
        new_specs = option.get_arg_specs(row.args)
        old_names = tuple(item.name for item in old_specs)
        new_names = tuple(item.name for item in new_specs)
        if new_names != old_names:
            state = bool(row.args.get("state", True))
            row.args = {
                item.name: row.args.get(item.name, item.default)
                for item in new_specs
            }
            row.args["state"] = state
            self._render_args_panel()
        self._emit_change()
