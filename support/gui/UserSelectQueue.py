import customtkinter as ctk
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Any, Tuple, Union, Type, ClassVar
from enum import Enum

DEFAULT_CHOICE = "(select)"

ROW_H = 36  # fixed row height
DROPDOWN_H = 32  # fixed optionmenu height
ICON_BTN_W = 36

Args = Dict[str, Any]
ArgType = Union[Type[bool], Type[int], Type[float], Type[str], Type[Enum]]
StepFn = Callable[..., None]
StepSpec = Tuple[StepFn, Args]

@dataclass(frozen=True)
class ArgSpec:
    name: str
    typ: ArgType | tuple[ArgType, ...]
    default: Any
    min: float | None = None
    max: float | None = None

@dataclass(frozen=True, slots=True)
class ArgBinding:
    label: str          # GUI key / display name
    field: str          # dataclass attribute name
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


class YoloInferenceSource(Enum):
    ORIGINAL = "Original Image"
    MARKUP = "Markup Frame"


@dataclass(slots=True)
class YoloOpts:
    want_pnp: bool = False
    want_qnp: bool = False
    want_wqnp: bool = False
    factor_graph: bool = False
    hyper_focus: bool = False
    feature_circles: bool = False
    inference_source: YoloInferenceSource = YoloInferenceSource.ORIGINAL
    BINDINGS: ClassVar[tuple[ArgBinding, ...]] = (
        ArgBinding("PnP", "want_pnp", bool, False),
        ArgBinding("QnP", "want_qnp", bool, False),
        ArgBinding("wQnP", "want_wqnp", bool, False),
        ArgBinding("Factor Graph", "factor_graph", bool, False),
        ArgBinding("Hyper Attention", "hyper_focus", bool, False),
        ArgBinding("Feature Circles", "feature_circles", bool, False),
        ArgBinding("Inference Source", "inference_source", YoloInferenceSource, YoloInferenceSource.ORIGINAL),
    )

    # Derived, guaranteed consistent
    ARG_SPECS: ClassVar[tuple["ArgSpec", ...]] = tuple(
        ArgSpec(b.label, b.object_type, b.default, b.min, b.max) for b in BINDINGS
    )
    KEYMAP: ClassVar[dict[str, str]] = {b.label: b.field for b in BINDINGS}
    

@dataclass(slots=True)
class HudOpts:
    store_attitude: bool = True
    map_transparency: float = 0.35
    draw_attitude: bool = True
    draw_as_alt: bool = True
    draw_title: bool = True
    draw_crosshairs: bool = True
    draw_mode: bool = True
    BINDINGS: ClassVar[tuple[ArgBinding, ...]] = (
        ArgBinding("Store Attitude for FG", "store_attitude", bool, True),
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
    frame: ctk.CTkFrame
    var: ctk.StringVar
    dropdown: ctk.CTkOptionMenu
    enabled_var: ctk.BooleanVar
    enabled_chk: ctk.CTkCheckBox
    args_btn: ctk.CTkButton
    up_btn: ctk.CTkButton
    down_btn: ctk.CTkButton
    remove_btn: ctk.CTkButton
    args: Args


@dataclass(frozen=True)
class StepOption:
    label: str
    fn: StepFn
    arg_specs: tuple[ArgSpec, ...] = ()
    arg_specs_fn: Optional[Callable[[Args], tuple[ArgSpec, ...]]] = None

    def get_arg_specs(self, args: Optional[Args] = None) -> tuple[ArgSpec, ...]:
        if self.arg_specs_fn is not None:
            return self.arg_specs_fn(args or {})
        return self.arg_specs

    @property
    def default_args(self) -> Args:
        return {spec.name: spec.default for spec in self.get_arg_specs({})}

class StepSpecQueueEditor(ctk.CTkFrame):
    """
    Dynamic queue editor for StepSpec = (StepFn, Args)
    - last row is placeholder
    - selecting a real step in the last row adds a new placeholder row
    - ✕ removes a row
    """

    def __init__(
            self,
            master,
            *,
            options: List["StepOption"],
            on_change: Optional[Callable[[List[StepSpec]], None]] = None,
            initial: Optional[List[StepSpec]] = None,
            **kwargs,
    ):
        super().__init__(master, **kwargs)

        # Build label->StepOption lookup
        if any(o.label == DEFAULT_CHOICE for o in options):
            raise ValueError(f"'{DEFAULT_CHOICE}' is reserved.")
        self._step_options: List["StepOption"] = options
        self._labels: List[str] = [DEFAULT_CHOICE] + [o.label for o in options]
        self._label_to_opt: Dict[str, "StepOption"] = {o.label: o for o in options}

        self._on_change = on_change
        self._rows: List[_QueueRow] = []

        # --- two-panel layout: queue left, args right ---
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(0, weight=1)

        self._queue_panel = ctk.CTkFrame(self, fg_color="transparent")
        self._queue_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=0)
        self._queue_panel.grid_columnconfigure(0, weight=1)

        self._args_panel = ctk.CTkFrame(self)
        self._args_panel.grid(row=0, column=1, sticky="nsew", padx=(8, 0), pady=0)
        self._args_panel.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(self._queue_panel, text="Image Processing Queue", anchor="w")
        title.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))

        args_title = ctk.CTkLabel(self._args_panel, text="Step Arguments", anchor="w")
        args_title.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))

        self._args_hint = ctk.StringVar(value="Select a step to edit its arguments.")
        self._args_hint_label = ctk.CTkLabel(self._args_panel,
                                             textvariable=self._args_hint,
                                             anchor="w")
        self._args_hint_label.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))

        # NOTE: CTkFrame has a fairly large default requested height when it has no children.
        # That makes the parent page look "too tall" until the first dropdown selection
        # causes widgets to be created inside this body.
        self._args_body = ctk.CTkFrame(self._args_panel, fg_color="transparent", height=1, width=1)
        self._args_body.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 8))
        self._args_body.grid_columnconfigure(1, weight=1)

        # Ensure the frame size follows children (and stays tiny when empty).
        self._args_body.grid_propagate(True)

        self._active_idx: Optional[int] = None

        # Seed from initial queue if provided
        if initial:
            for fn, args in initial:
                # try to match an existing option by fn+args; else just show fn name
                label = self._find_label_for(fn, args)
                self._add_row(selected=label, args_override=args)

        self._ensure_trailing_placeholder()

        # auto-select first real row if present
        self._active_idx = self._first_real_row_index()
        self._render_args_panel()
        self._emit_change()

    def _on_first_map(self, _evt=None) -> None:
        try:
            self.unbind("<Map>")
        except Exception:
            pass

        self._ensure_trailing_placeholder()
        self._refresh_remove_buttons()

        if self._active_idx is None:
            self._active_idx = self._first_real_row_index()

        # Let CTk finish internal measurement once, then render the args panel.
        self.after_idle(self._render_args_panel)

    def _find_label_for(self, fn: StepFn, args: Args) -> str:
        for o in self._step_options:
            if o.fn is fn:
                return o.label

        name = getattr(fn, "__name__", "step")
        return f"{name}{args}"

    # -------- public API --------

    def get_queue(self) -> List[StepSpec]:
        out: List[StepSpec] = []
        for r in self._rows:
            lab = r.var.get()
            if lab and lab != DEFAULT_CHOICE and lab in self._label_to_opt:
                opt = self._label_to_opt[lab]
                out.append((opt.fn, dict(r.args)))  # <-- copy args
        return out

    def set_queue(self, queue: List[StepSpec], *, emit_change: bool = False) -> None:
        for r in self._rows:
            r.frame.destroy()
        self._rows.clear()

        for fn, args in queue:
            self._add_row(selected=self._find_label_for(fn, args), args_override=dict(args))

        self._ensure_trailing_placeholder()
        self._refresh_remove_buttons()
        self._refresh_move_buttons()
        self._active_idx = self._first_real_row_index()
        self._render_args_panel()

        if emit_change:
            self._emit_change()

    # -------- internals --------

    def _add_row(self, *, selected: str = DEFAULT_CHOICE, args_override: Optional[Args] = None) -> None:
        row_frame = ctk.CTkFrame(self._queue_panel, fg_color="transparent", height=ROW_H)
        row_frame.grid(row=len(self._rows) + 1, column=0, sticky="ew", padx=8, pady=4)
        row_frame.grid_columnconfigure(0, weight=1)

        # IMPORTANT: fixed-height row; prevents CTkOptionMenu initial reqheight inflation
        row_frame.grid_propagate(False)

        var = ctk.StringVar(value=selected if selected in self._labels else DEFAULT_CHOICE)

        # Seed args:
        init_args: Args = {}
        if var.get() in self._label_to_opt:
            init_args = dict(self._label_to_opt[var.get()].default_args)
        if args_override is not None:
            init_args = dict(args_override)

        init_args.setdefault("state", True)

        dropdown = ctk.CTkOptionMenu(
            row_frame,
            values=self._labels,
            variable=var,
            command=lambda _=None, rf=row_frame: self._on_row_changed(rf),
            anchor="w",
            height=DROPDOWN_H,
        )
        dropdown.grid(row=0, column=0, sticky="ew", padx=(0, 6), pady=0)
        dropdown.bind("<Button-1>", lambda _evt, rf=row_frame: self._set_active_by_frame(rf))

        enabled_var = ctk.BooleanVar(value=bool(init_args.get("state", True)))
        enabled_chk = ctk.CTkCheckBox(
            row_frame,
            text="On",
            width=60,
            height=DROPDOWN_H,
            variable=enabled_var,
            command=lambda rf=row_frame: self._on_enabled_toggled(rf),
        )
        enabled_chk.grid(row=0, column=1, sticky="e", padx=(0, 6), pady=0)
        enabled_chk.bind("<Button-1>", lambda _evt, rf=row_frame: self._set_active_by_frame(rf))

        args_btn = ctk.CTkButton(
            row_frame,
            text="⚙",
            width=ICON_BTN_W,
            height=DROPDOWN_H,
            command=lambda rf=row_frame: self._edit_args_for_frame(rf),
        )
        args_btn.grid(row=0, column=2, sticky="e", padx=(0, 6), pady=0)

        up_btn = ctk.CTkButton(
            row_frame,
            text="↑",
            width=ICON_BTN_W,
            height=DROPDOWN_H,
            command=lambda rf=row_frame: self._move_row_by_frame(rf, -1),
        )
        up_btn.grid(row=0, column=3, sticky="e", padx=(0, 6), pady=0)

        down_btn = ctk.CTkButton(
            row_frame,
            text="↓",
            width=ICON_BTN_W,
            height=DROPDOWN_H,
            command=lambda rf=row_frame: self._move_row_by_frame(rf, +1),
        )
        down_btn.grid(row=0, column=4, sticky="e", padx=(0, 6), pady=0)

        remove_btn = ctk.CTkButton(
            row_frame,
            text="✕",
            width=ICON_BTN_W,
            height=DROPDOWN_H,
            command=lambda rf=row_frame: self._remove_row_by_frame(rf),
        )
        remove_btn.grid(row=0, column=5, sticky="e", pady=0)

        self._rows.append(
            _QueueRow(
                row_frame,
                var,
                dropdown,
                enabled_var,
                enabled_chk,
                args_btn,
                up_btn,
                down_btn,
                remove_btn,
                init_args,
            )
        )
        self._refresh_remove_buttons()
        self._refresh_move_buttons()

    def _on_enabled_toggled(self, frame: ctk.CTkFrame) -> None:
        idx = next((i for i, r in enumerate(self._rows) if r.frame == frame), None)
        if idx is None:
            return

        row = self._rows[idx]
        row.args["state"] = bool(row.enabled_var.get())
        self._active_idx = idx
        self._render_args_panel()
        self._emit_change()

    def _move_row_by_frame(self, frame: ctk.CTkFrame, direction: int) -> None:
        idx = next((i for i, r in enumerate(self._rows) if r.frame == frame), None)
        if idx is None:
            return

        new_idx = idx + direction
        if new_idx < 0 or new_idx >= len(self._rows):
            return

        # Never move into/out of the trailing placeholder row
        if self._rows[idx].var.get() == DEFAULT_CHOICE:
            return
        if self._rows[new_idx].var.get() == DEFAULT_CHOICE:
            return

        self._rows[idx], self._rows[new_idx] = self._rows[new_idx], self._rows[idx]

        if self._active_idx == idx:
            self._active_idx = new_idx
        elif self._active_idx == new_idx:
            self._active_idx = idx

        self._regrid_rows()
        self._refresh_remove_buttons()
        self._refresh_move_buttons()
        self._render_args_panel()
        self._emit_change()

    def _refresh_move_buttons(self) -> None:
        last_idx = len(self._rows) - 1

        for i, r in enumerate(self._rows):
            is_placeholder = (i == last_idx and r.var.get() == DEFAULT_CHOICE)

            if is_placeholder:
                r.up_btn.configure(state="disabled")
                r.down_btn.configure(state="disabled")
                continue

            can_move_up = i > 0
            can_move_down = i < last_idx - 1  # cannot move into trailing placeholder

            r.up_btn.configure(state="normal" if can_move_up else "disabled")
            r.down_btn.configure(state="normal" if can_move_down else "disabled")

    def _remove_row_by_frame(self, frame: ctk.CTkFrame) -> None:
        idx = next((i for i, r in enumerate(self._rows) if r.frame == frame), None)
        if idx is None:
            return

        self._rows[idx].frame.destroy()
        del self._rows[idx]

        self._regrid_rows()
        self._ensure_trailing_placeholder()
        self._refresh_remove_buttons()
        self._refresh_move_buttons()

        # active row bookkeeping

        if self._active_idx is not None:
            if idx == self._active_idx:
                self._active_idx = self._first_real_row_index()
            elif idx < self._active_idx:
                self._active_idx -= 1
        self._render_args_panel()
        self._emit_change()

    def _regrid_rows(self) -> None:
        for i, r in enumerate(self._rows):
            r.frame.grid_configure(row=i + 1)

    def _ensure_trailing_placeholder(self) -> None:
        if not self._rows or self._rows[-1].var.get() != DEFAULT_CHOICE:
            self._add_row(selected=DEFAULT_CHOICE)

    def _refresh_remove_buttons(self) -> None:
        for i, r in enumerate(self._rows):
            is_trailing_placeholder = (i == len(self._rows) - 1) and (r.var.get() == DEFAULT_CHOICE)
            widget_state = "disabled" if is_trailing_placeholder else "normal"
            r.remove_btn.configure(state=widget_state)
            r.args_btn.configure(state=widget_state)
            r.enabled_chk.configure(state=widget_state)

    def _on_row_changed(self, frame: ctk.CTkFrame) -> None:
        """
        Called when a row's dropdown selection changes.
        - Ensures row.args matches the selected option
        - Keeps existing edited args if selection didn't change
        - Preserves row.args["state"]
        - Updates active row + args panel
        """
        idx = next((i for i, r in enumerate(self._rows) if r.frame == frame), None)
        if idx is not None:
            row = self._rows[idx]
            lab = row.var.get()

            if lab in self._label_to_opt:
                opt = self._label_to_opt[lab]
                expected_keys = {spec.name for spec in opt.get_arg_specs(row.args)}
                state_val = bool(row.args.get("state", row.enabled_var.get()))

                arg_keys_without_state = set(row.args.keys()) - {"state"}
                if arg_keys_without_state != expected_keys:
                    row.args = dict(opt.default_args)

                row.args["state"] = state_val
                row.enabled_var.set(state_val)
            else:
                # placeholder or invalid selection
                row.args = {}
                row.enabled_var.set(True)

            self._active_idx = idx

        self._ensure_trailing_placeholder()
        self._refresh_remove_buttons()
        self._refresh_move_buttons()
        self._render_args_panel()
        self._emit_change()

    def _emit_change(self) -> None:
        if self._on_change:
            self._on_change(self.get_queue())

    # -------- args panel helpers --------
    def _first_real_row_index(self) -> Optional[int]:
        for i, r in enumerate(self._rows):
            if r.var.get() != DEFAULT_CHOICE:
                return i
        return None

    def _set_active_by_frame(self, frame: ctk.CTkFrame) -> None:
        idx = next((i for i, r in enumerate(self._rows) if r.frame == frame), None)
        if idx is None:
            return
        self._active_idx = idx
        self._render_args_panel()

    def _edit_args_for_frame(self, frame: ctk.CTkFrame) -> None:
        # right panel is live; clicking ⚙ just forces focus to that row
        self._set_active_by_frame(frame)

    def _clear_args_body(self) -> None:
        for w in self._args_body.winfo_children():
            try:
                w.destroy()
            except Exception:
                pass

    def _render_args_panel(self) -> None:
        self._clear_args_body()

        if self._active_idx is None or self._active_idx >= len(self._rows):
            self._args_hint.set("Select a step to edit its arguments.")
            return

        row = self._rows[self._active_idx]
        lab = row.var.get()
        if lab == DEFAULT_CHOICE or lab not in self._label_to_opt:
            self._args_hint.set("Select a step to edit its arguments.")
            return

        opt = self._label_to_opt[lab]
        self._args_hint.set(opt.label)

        # If step has no real args, say so.
        if len(opt.get_arg_specs(row.args)) == 0:
            ctk.CTkLabel(self._args_body, text="(no args)", anchor="w").grid(
                row=0, column=0, columnspan=2, sticky="w", pady=4
            )
            return

        for i, spec in enumerate(opt.get_arg_specs(row.args)):
            name = spec.name
            val = row.args.get(name, spec.default)

            lbl = ctk.CTkLabel(self._args_body, text=name, anchor="w")
            lbl.grid(row=i, column=0, sticky="w", padx=(0, 8), pady=4)

            if isinstance(val, Enum):

                enum_t = type(val)
                var = ctk.StringVar(value=val.name)

                def _on_enum_change(choice, n=spec.name, et=enum_t):
                    self._set_arg(n, et[choice])
                    self._render_args_panel()

                dd = ctk.CTkOptionMenu(
                    self._args_body,
                    values=[e.name for e in enum_t],
                    variable=var,
                    command=_on_enum_change,
                    anchor="w",
                )
                dd.grid(row=i, column=1, sticky="ew", pady=4)
                continue

            if isinstance(val, bool):
                bvar = ctk.BooleanVar(value=bool(val))
                sw = ctk.CTkSwitch(
                    self._args_body,
                    text="",
                    variable=bvar,
                    command=lambda n=spec.name, v=bvar: self._set_arg(n, bool(v.get())),
                )
                sw.grid(row=i, column=1, sticky="w", pady=4)
                continue

            # float -> slider (default 0..1 unless bounds provided)
            if isinstance(val, float):
                min_v = spec.min if getattr(spec, "min", None) is not None else 0.0
                max_v = spec.max if getattr(spec, "max", None) is not None else 1.0

                lbl_var = ctk.StringVar(value=f"{name}: {float(val):8.2f}")
                lbl.configure(textvariable=lbl_var)

                fvar = ctk.DoubleVar(value=float(val))

                def _on_slider(v, n=spec.name, lv=lbl_var, label=name):
                    fv = float(v)
                    lv.set(f"{label}: {fv:8.2f}")
                    self._set_arg(n, fv)

                slider = ctk.CTkSlider(
                    self._args_body,
                    from_=float(min_v),
                    to=float(max_v),
                    variable=fvar,
                    command=_on_slider,
                    width=75
                )
                slider.grid(row=i, column=1, sticky="ew", pady=4)
                continue

            # int/str -> entry with cast on commit
            svar = ctk.StringVar(value=str(val))
            ent = ctk.CTkEntry(self._args_body, textvariable=svar)
            ent.grid(row=i, column=1, sticky="ew", pady=4)

            spec_name = spec.name

            def _commit(_evt=None, n=spec_name, sv=svar, old=val):
                txt = sv.get()
                try:
                    if isinstance(old, int) and not isinstance(old, bool):
                        newv = int(txt)
                    else:
                        newv = txt
                except Exception:
                    sv.set(str(old))
                    return
                self._set_arg(n, newv)

            ent.bind("<Return>", _commit)
            ent.bind("<FocusOut>", _commit)

    def _set_arg(self, name: str, value: Any) -> None:
        if self._active_idx is None or self._active_idx >= len(self._rows):
            return

        row = self._rows[self._active_idx]
        lab = row.var.get()
        if lab not in self._label_to_opt:
            return

        opt = self._label_to_opt[lab]

        spec = next((s for s in opt.get_arg_specs(row.args) if s.name == name), None)
        if spec is None:
            return

        # type enforcement
        allowed = spec.typ if isinstance(spec.typ, tuple) else (spec.typ,)
        if not any(isinstance(value, t) for t in allowed if isinstance(t, type)):
            return

        # bounds
        if isinstance(value, (int, float)):
            if spec.min is not None and value < spec.min:
                value = spec.min
            if spec.max is not None and value > spec.max:
                value = spec.max

        row.args[name] = value
        self._emit_change()
