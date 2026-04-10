import copy
import enum
from pathlib import Path
from typing import Any

import customtkinter as ctk

from support.core.enums import ExportQuality
from support.io.my_logging import LOG


class ConfigRuntime:
    """Owns GUI-model sync, queue serialization, and config-backed refreshes."""

    def __init__(self, owner: Any):
        self.owner = owner

    def init_flag_vars(self) -> None:
        double_vars = ["yolo_conf", "yolo_iou"]
        for name in self.owner._flags:
            if not hasattr(self.owner.camConfig, name):
                setattr(self.owner.camConfig, name, 1.0 if name in double_vars else False)

            if name in double_vars:
                var = ctk.DoubleVar(value=getattr(self.owner.camConfig, name, 1.0))
            else:
                var = ctk.BooleanVar(value=bool(getattr(self.owner.camConfig, name, False)))

            var.trace_add("write", lambda var_name, index, op, n=name: self.on_flag_changed(n))
            self.owner._flag_vars[name] = var

    def on_flag_changed(self, name: str) -> None:
        if name in ("yolo_conf", "yolo_iou"):
            value = float(self.owner._flag_vars[name].get())
        else:
            value = bool(self.owner._flag_vars[name].get())

        setattr(self.owner.camConfig, name, value)
        self.save_to_cache()

    def sync_flags_from_model(self) -> None:
        for name in self.owner._flags:
            if name in ("yolo_conf", "yolo_iou"):
                self.owner._flag_vars[name].set(float(getattr(self.owner.camConfig, name, 1.0)))
            else:
                self.owner._flag_vars[name].set(bool(getattr(self.owner.camConfig, name, False)))

    def sync_dp_from_model(self) -> None:
        self.owner.batch_controller.sync_from_model()

    def on_queue_changed(self, new_queue) -> None:
        if getattr(self.owner, "_loading_config", False):
            return

        normalized_queue = [(fn, copy.deepcopy(args)) for fn, args in new_queue]
        sig_did_change = self.owner.list_of_image_process_functors != normalized_queue
        self.owner.list_of_image_process_functors = normalized_queue

        if sig_did_change:
            self.owner.camConfig.image_processing_queue = self.queue_to_config(normalized_queue)
            self.save_to_cache()

            if self.owner.func_that_refits is not None:
                self.owner.func_that_refits()

    def sync_queue_from_model(self) -> None:
        cfg = getattr(self.owner.camConfig, "image_processing_queue", [])
        rebuilt = self.queue_from_config(cfg)

        self.owner.list_of_image_process_functors = [
            (fn, copy.deepcopy(args)) for fn, args in rebuilt
        ]

        if hasattr(self.owner, "imgProcQueue_editor") and self.owner.imgProcQueue_editor is not None:
            self.owner.imgProcQueue_editor.set_queue(
                [(fn, copy.deepcopy(args)) for fn, args in rebuilt],
                emit_change=False,
            )

    def queue_to_config(self, queue):
        out = []

        for fn, args in queue:
            try:
                label = self.owner.fn_to_label[fn]
            except KeyError:
                raise RuntimeError(f"Queue contains unknown processing function: {fn}")

            clean_args = {key: self.serialize_queue_arg(value) for key, value in args.items()}
            out.append({"label": label, "args": clean_args})

        return out

    def queue_from_config(self, queue_cfg):
        if not queue_cfg:
            return []

        option_by_label = {opt.label: opt for opt in self.owner.step_options}
        rebuilt = []

        for row in queue_cfg:
            label = row["label"]

            if label not in option_by_label:
                LOG.warning("Skipping cached queue step '%s' (unknown)", label)
                continue

            opt = option_by_label[label]
            raw_args = copy.deepcopy(row.get("args", {}))
            parsed_args = opt.default_args.copy()

            spec_by_name = {spec.name: spec for spec in opt.get_arg_specs(parsed_args)}
            for arg_name, raw_val in raw_args.items():
                spec = spec_by_name.get(arg_name)
                if spec is None:
                    parsed_args[arg_name] = raw_val
                    continue

                parsed_args[arg_name] = self.deserialize_queue_arg(spec, raw_val)

            spec_by_name = {spec.name: spec for spec in opt.get_arg_specs(parsed_args)}
            for spec in spec_by_name.values():
                parsed_args.setdefault(spec.name, spec.default)

            rebuilt.append((opt.fn, parsed_args))

        return rebuilt

    @staticmethod
    def serialize_queue_arg(value):
        if isinstance(value, enum.Enum):
            return value.value
        return value

    @staticmethod
    def deserialize_queue_arg(spec, raw_val):
        default = spec.default

        if isinstance(default, enum.Enum):
            enum_type = type(default)
            try:
                return enum_type(raw_val)
            except (TypeError, ValueError):
                LOG.warning(
                    "Failed to parse enum arg '%s' from cached value %r; using default %r",
                    spec.name, raw_val, default,
                )
                return default

        try:
            if isinstance(default, bool):
                return bool(raw_val)
            if isinstance(default, int) and not isinstance(default, bool):
                return int(raw_val)
            if isinstance(default, float):
                return float(raw_val)
            if isinstance(default, str):
                return str(raw_val)
        except TypeError as e:
            LOG.warning(
                "Failed to parse arg '%s' from cached value %r; using default %r. \nError: %s",
                spec.name, raw_val, default, e,
            )
            return default

        return raw_val

    def load_from_cache(self) -> bool:
        self.owner._loading_config = True
        res = self.owner.config_store.load_from_cache(self.owner.camConfig)
        if not res.loaded_yaml:
            self.owner._loading_config = False
            return False
        return True

    def update_post_new_config(self) -> None:
        iou = copy.deepcopy(self.owner.camConfig.yolo_iou)
        self.owner._flag_vars["yolo_conf"].set(float(self.owner.camConfig.yolo_conf))
        self.owner._flag_vars["yolo_iou"].set(float(iou))

        self.update_log_file()
        self.ingest_calibration()
        self.update_yolo_model()
        if self.owner.ThreeDTruthPoints is not None:
            self.load_truth_points()

        self.sync_flags_from_model()
        self.sync_dp_from_model()
        self.sync_queue_from_model()

        try:
            if hasattr(self.owner, "gpu_slider"):
                self.owner.gpu_slider.configure(
                    state="normal" if bool(getattr(self.owner.camConfig, "dp_gpu", False)) else "disabled"
                )
            if not bool(getattr(self.owner.camConfig, "dp_gpu", False)):
                self.owner.gpu_slider.set(0.0)
        except TypeError:
            pass

        if self.owner.exportStartFrame is not None:
            self.owner.exportStartFrame.configure(text=f"Start Frame: {self.owner.camConfig.start_export_idx}")
        if self.owner.exportEndFrame is not None:
            self.owner.exportEndFrame.configure(text=f"End Frame: {self.owner.camConfig.end_export_idx}")

        if hasattr(self.owner, "filepath_page") and self.owner.filepath_page is not None:
            self.owner.filepath_page.sync_labels()

        if hasattr(self.owner, "playback_controller") and self.owner.playback_controller is not None:
            self.owner.playback_controller.update_playback_menu()

        if hasattr(self.owner, "exportQualityCombo") and self.owner.exportQualityCombo is not None:
            self.owner.exportQualityCombo.set(self.owner.camConfig.export_quality.value)

        self.save_to_cache()

        if self.owner.func_that_refits is not None:
            self.owner.func_that_refits()

    def save_to_cache(self, immediate: bool = False, delay_ms: int = 500) -> None:
        if getattr(self.owner, "_loading_config", False):
            return

        self.owner.camConfig.yolo_conf = float(self.owner._flag_vars["yolo_conf"].get())
        self.owner.camConfig.yolo_iou = float(self.owner._flag_vars["yolo_iou"].get())
        self.owner.config_store.save_to_cache(
            self.owner.camConfig,
            immediate=immediate,
            delay_ms=delay_ms,
        )

    def update_log_file(self) -> None:
        if self.owner.hud_marker is not None:
            self.owner.hud_marker.read_attitude_files(self.owner.camConfig.hud_data_filepath)
            self.owner.playback_controller.load_time_offset(self.owner.camConfig.hud_data_filepath)
        self.save_to_cache()

    def update_yolo_model(self) -> None:
        if self.owner.camConfig.yoloFilepath and self.owner.yoloSession is not None:
            self.owner.yoloSession.setNewFolder(self.owner.camConfig.yoloFilepath)

    def load_truth_points(self) -> None:
        if not self.owner.camConfig.ThreeDTruthFilepath:
            return

        truth_path = Path(self.owner.camConfig.ThreeDTruthFilepath)

        from support.io.ThreeD_truth import TruthPoints

        self.owner.ThreeDTruthPoints = TruthPoints()
        self.owner.ThreeDTruthPoints.try_load(truth_path)

    def update_quality(self, quality_value: str) -> None:
        self.owner.camConfig.export_quality = ExportQuality(quality_value)
        self.save_to_cache()

    def ingest_calibration(self) -> None:
        if (
            not self.owner.calibration.fromBinFile(self.owner.camConfig.calibFilepath)
            and not self.owner.calibration.fromFile(self.owner.camConfig.calibFilepath)
        ):
            if self.owner.selectCalibLabel is not None:
                self.owner.selectCalibLabel.configure(text="No Calibration Found")
                self.owner.after(10, self.owner.update_idletasks)  # type: ignore[call-arg]
            return

        if not self.owner.calibration.validCal:
            return

        if self.owner.selectCalibLabel is not None:
            self.owner.selectCalibLabel.configure(
                text="../" + Path(self.owner.camConfig.calibFilepath).name if self.owner.camConfig.calibFilepath else "../",
                bg_color=self.owner.selectCalibLabel.cget("bg_color"),
            )
            self.owner.filepath_page.update_idletasks()
            self.owner.update_idletasks()
            self.owner.selectCalibLabel.update_idletasks()
            self.owner.filepath_page.update_idletasks()
            self.owner.update_idletasks()

        if self.owner.yoloSession is not None:
            self.owner.yoloSession.set_calibration(self.owner.calibration)

        self.owner.fisheye_mgr.clear()

        newK, std_map1, std_map2 = self.owner.fisheye_mgr.ensure_standard_undistort_maps(
            self.owner.calibration,
            alpha=0.0,
        )
        self.owner.calibration.remapK = newK

        if self.owner.calibration.fisheye:
            self.owner.map1, self.owner.map2 = None, None
        else:
            self.owner.map1, self.owner.map2 = std_map1, std_map2

        self.save_to_cache()
