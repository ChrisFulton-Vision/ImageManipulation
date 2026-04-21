import threading
from collections import deque
from typing import Any
from support.io.my_logging import LOG
import time
import cv2
import numpy as np
from numpy.typing import NDArray

try:
    from vmbpy import VmbSystem, PixelFormat, VmbTimeout, AllocationMode
    HAVE_VMBPY = True
except Exception:
    VmbSystem = None
    PixelFormat = None
    VmbTimeout = Exception
    HAVE_VMBPY = False

class VimbaController:
    def __init__(self):
        self._live_vimba_cmds = deque()
        self._live_vimba_cmd_lock = threading.Lock()
        self._live_vimba_cam = None
        self._live_vimba_enabled = False

        self._latest_raw_frame = None
        self._latest_raw_time = None
        self._latest_lock = threading.Lock()

    def _clear_deque(self):
        self._live_vimba_cmds.clear()

    def _append_deque(self, profile: dict[str, Any]):
        self._live_vimba_cmds.append(profile)

    def live_update(self, settings: dict[str, Any]):
        with self._live_vimba_cmd_lock:
            self._live_vimba_cmds.clear()
            self._live_vimba_cmds.append(dict(settings))

    def _drain_live_vimba_updates(self, camConfig):
        cam = self._live_vimba_cam
        if cam is None or not self._live_vimba_enabled:
            return

        pending = None
        with self._live_vimba_cmd_lock:
            if self._live_vimba_cmds:
                pending = self._live_vimba_cmds.pop()
                self._live_vimba_cmds.clear()

        if pending is None:
            return

        gain_auto = self._normalize_vimba_auto_mode(
            pending.get("vimba_gain_auto", getattr(camConfig, "vimba_gain_auto", "Off"))
        )
        exposure_auto = self._normalize_vimba_auto_mode(
            pending.get("vimba_exposure_auto", getattr(camConfig, "vimba_exposure_auto", "Off"))
        )

        try:
            gain_value = float(pending.get("vimba_gain", getattr(camConfig, "vimba_gain", 0.0)))
        except Exception:
            gain_value = float(getattr(camConfig, "vimba_gain", 0.0))

        try:
            exposure_value = float(pending.get("vimba_exposure_us", getattr(camConfig, "vimba_exposure_us", 10000.0)))
        except Exception:
            exposure_value = float(getattr(camConfig, "vimba_exposure_us", 10000.0))

        self._set_vimba_enum_feature(cam, ("GainAuto",), gain_auto)
        if gain_auto == "Off":
            self._set_vimba_float_feature(cam, ("Gain",), gain_value)

        self._set_vimba_enum_feature(cam, ("ExposureAuto",), exposure_auto)
        if exposure_auto == "Off":
            self._set_vimba_float_feature(cam, ("ExposureTime", "ExposureTimeAbs"), exposure_value)


        LOG.info("Applied live Vimba tuning update.")

    def select_vimba_camera(self, vmb, cam_id):
        cams = list(vmb.get_all_cameras())
        if not cams:
            raise RuntimeError("No Vimba cameras detected.")

        wanted = str(cam_id or "").strip()
        if not wanted:
            return cams[0]

        for cam in cams:
            try:
                if wanted in {cam.get_id(), cam.get_serial(), cam.get_name()}:
                    return cam
            except Exception:
                pass

        available = []
        for cam in cams:
            try:
                available.append(f"{cam.get_id()} / {cam.get_serial()} / {cam.get_name()}")
            except Exception:
                pass

        raise RuntimeError(
            "Requested Vimba camera was not found.\nAvailable cameras:\n" + "\n".join(available)
        )

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
    def _try_get_vimba_feature(cam, *feature_names: str):
        for feature_name in feature_names:
            try:
                feature = getattr(cam, feature_name)
            except Exception:
                feature = None
            if feature is not None:
                return feature, feature_name
        return None, None

    def _set_vimba_enum_feature(self, cam, feature_names: tuple[str, ...], value: str) -> bool:
        for feature_name in feature_names:
            feature, actual_name = self._try_get_vimba_feature(cam, feature_name)
            if feature is None:
                continue
            try:
                feature.set(value)
                return True
            except Exception as e:
                LOG.warning(f"Could not set Vimba {actual_name} to {value}: {e}")
        return False

    def _set_vimba_float_feature(self, cam, feature_names: tuple[str, ...], value: float) -> bool:
        for feature_name in feature_names:
            feature, actual_name = self._try_get_vimba_feature(cam, feature_name)
            if feature is None:
                continue
            try:
                value_to_set = float(value)
                try:
                    lo, hi = feature.get_range()
                    value_to_set = min(max(value_to_set, float(lo)), float(hi))
                except Exception:
                    pass
                feature.set(value_to_set)
                return True
            except Exception as e:
                LOG.warning(f"Could not set Vimba {actual_name} to {value}: {e}")
        return False

    def _set_vimba_int_feature(self, cam, feature_names: tuple[str, ...], value: int) -> bool:
        for feature_name in feature_names:
            feature, actual_name = self._try_get_vimba_feature(cam, feature_name)
            if feature is None:
                continue
            try:
                value_to_set = int(round(value))
                try:
                    lo, hi = feature.get_range()
                    value_to_set = min(max(value_to_set, int(lo)), int(hi))
                except Exception:
                    lo = 0

                try:
                    inc = int(feature.get_increment())
                    if inc > 1:
                        value_to_set = lo + ((value_to_set - lo) // inc) * inc
                except Exception:
                    pass

                feature.set(value_to_set)
                return True
            except Exception as e:
                LOG.warning(f"Could not set Vimba {actual_name} to {value}: {e}")
        return False

    def get_vimba_profile_spec(self, vimba_profile) -> dict[str, int | str]:
        profile = str(vimba_profile or "Full Res")

        profiles = {
            "Full Res": {
                "name": "Full Res",
                "bin_x": 1,
                "bin_y": 1,
                "bin_mode": "",
                "roi_w": 0,      # 0 => full sensor
                "roi_h": 0,
                "preview_max_dim": 0,   # 0 => no forced preview downscale
                "buffer_count": 2,
            },
            "Zoom 1440": {
                "name": "Zoom 1440",
                "bin_x": 0,
                "bin_y": 0,
                "bin_mode": "",
                "roi_w": 1440,
                "roi_h": 1440,
                "preview_max_dim": 1440,
                "buffer_count": 2,
            },
            "BinSum To 1440": {
                "name": "BinSum To 1440",
                "bin_x": 2,
                "bin_y": 2,
                "bin_mode": "Sum",
                "roi_w": 0, # example: 864, captures center 864 columns of image
                "roi_h": 0,
                "preview_max_dim": 1440,
                "buffer_count": 2,
            },
            "BinAvg To 1440": {
                "name": "BinAvg To 1440",
                "bin_x": 2,
                "bin_y": 2,
                "bin_mode": "Average",
                "roi_w": 0,
                "roi_h": 0,
                "preview_max_dim": 1440,
                "buffer_count": 2,
            },
            "Zoom 864": {
                "name": "Zoom 864",
                "bin_x": 0,
                "bin_y": 0,
                "bin_mode": "",
                "roi_w": 864, # example: 864, captures center 864 columns of image
                "roi_h": 864,
                "preview_max_dim": 864,
                "buffer_count": 2,
            },
            "BinSum To 864": {
                "name": "BinSum To 864",
                "bin_x": 2,
                "bin_y": 2,
                "bin_mode": "Sum",
                "roi_w": 0, # example: 864, captures center 864 columns of image
                "roi_h": 0,
                "preview_max_dim": 864,
                "buffer_count": 2,
            },
            "BinAvg To 864": {
                "name": "BinAvg To 864",
                "bin_x": 2,
                "bin_y": 2,
                "bin_mode": "Average",
                "roi_w": 0,
                "roi_h": 0,
                "preview_max_dim": 864,
                "buffer_count": 2,
            },
            "BinSum To 712": {
                "name": "BinSum To 712",
                "bin_x": 4,
                "bin_y": 4,
                "bin_mode": "Sum",
                "roi_w": 0, # example: 864, captures center 864 columns of image
                "roi_h": 0,
                "preview_max_dim": 864,
                "buffer_count": 2,
            },
            "BinAvg To 712": {
                "name": "BinAvg To 712",
                "bin_x": 4,
                "bin_y": 4,
                "bin_mode": "Average",
                "roi_w": 0,
                "roi_h": 0,
                "preview_max_dim": 864,
                "buffer_count": 2,
            },
        }

        legacy_profiles = {
            "Bin To 1440": "BinSum To 1440",
            "Bin To 864": "BinSum To 864",
            "Bin To 712": "BinSum To 712",
        }
        profile = legacy_profiles.get(profile, profile)
        return profiles.get(profile, profiles["Full Res"])

    def _apply_vimba_profile(self, cam, vimba_profile):
        spec = self.get_vimba_profile_spec(vimba_profile)

        bin_mode = str(spec.get("bin_mode", "") or "").strip()
        if bin_mode:
            self._set_vimba_enum_feature(
                cam,
                ("BinningSelector",),
                "Digital",
            )
            self._set_vimba_enum_feature(
                cam,
                ("BinningHorizontalMode", "BinningVerticalMode"),
                bin_mode,
            )

        # 1) Binning first
        self._set_vimba_int_feature(cam, ("BinningHorizontal",), int(spec["bin_x"]))
        self._set_vimba_int_feature(cam, ("BinningVertical",), int(spec["bin_y"]))

        width_feature, _ = self._try_get_vimba_feature(cam, "Width")
        height_feature, _ = self._try_get_vimba_feature(cam, "Height")
        offset_x_feature, _ = self._try_get_vimba_feature(cam, "OffsetX")
        offset_y_feature, _ = self._try_get_vimba_feature(cam, "OffsetY")

        if width_feature is None or height_feature is None:
            return

        # 2) Reset offsets before changing ROI size
        if offset_x_feature is not None:
            self._set_vimba_int_feature(cam, ("OffsetX",), 0)
        if offset_y_feature is not None:
            self._set_vimba_int_feature(cam, ("OffsetY",), 0)

        full_w = int(width_feature.get_range()[1])
        full_h = int(height_feature.get_range()[1])

        desired_w = int(spec["roi_w"]) if int(spec["roi_w"]) > 0 else full_w
        desired_h = int(spec["roi_h"]) if int(spec["roi_h"]) > 0 else full_h

        self._set_vimba_int_feature(cam, ("Width",), desired_w)
        self._set_vimba_int_feature(cam, ("Height",), desired_h)

        actual_w = int(width_feature.get())
        actual_h = int(height_feature.get())

        # 3) Center the ROI if we are not using full sensor
        if actual_w < full_w and offset_x_feature is not None:
            self._set_vimba_int_feature(cam, ("OffsetX",), (full_w - actual_w) // 2)
        if actual_h < full_h and offset_y_feature is not None:
            self._set_vimba_int_feature(cam, ("OffsetY",), (full_h - actual_h) // 2)

    def configure_vimba_camera(self, cam, camConfig):
        settings_xml = str(getattr(camConfig, "vimba_settings_xml", "") or "").strip()
        if settings_xml:
            cam.load_settings(settings_xml)

        # Force free-run
        for feat_name, val in (
                ("TriggerSelector", "FrameStart"),
                ("TriggerMode", "Off"),
                ("AcquisitionMode", "Continuous"),
                ("ExposureMode", "Timed"),
        ):
            try:
                getattr(cam, feat_name).set(val)
            except Exception:
                pass

        # Apply profile-driven binning / ROI first
        self._apply_vimba_profile(cam, camConfig.vimba_profile)

        gain_auto = self._normalize_vimba_auto_mode(getattr(camConfig, "vimba_gain_auto", "Off"))
        exposure_auto = self._normalize_vimba_auto_mode(getattr(camConfig, "vimba_exposure_auto", "Off"))

        try:
            gain_value = float(getattr(camConfig, "vimba_gain", 0.0) or 0.0)
        except Exception:
            gain_value = 0.0

        try:
            exposure_value = float(getattr(camConfig, "vimba_exposure_us", 10000.0) or 10000.0)
        except Exception:
            exposure_value = 10000.0

        self._set_vimba_enum_feature(cam, ("GainAuto",), gain_auto)
        if gain_auto == "Off":
            self._set_vimba_float_feature(cam, ("Gain",), gain_value)

        self._set_vimba_enum_feature(cam, ("ExposureAuto",), exposure_auto)
        if exposure_auto == "Off":
            self._set_vimba_float_feature(cam, ("ExposureTime", "ExposureTimeAbs"), exposure_value)

        try:
            if hasattr(cam, "GVSPAdjustPacketSize"):
                cam.GVSPAdjustPacketSize.run()

                for _ in range(50):
                    try:
                        if cam.GVSPAdjustPacketSize.is_done():
                            break
                    except Exception:
                        break
                    time.sleep(0.1)
        except Exception as e:
            LOG.warning(f"GVSPAdjustPacketSize failed: {e}")

        try:
            fmts = set(cam.get_pixel_formats())
            for name in ("Bgr8", "Mono8"):
                pf = getattr(PixelFormat, name, None)
                if pf is not None and pf in fmts:
                    cam.set_pixel_format(pf)
                    break
        except Exception as e:
            LOG.warning(f"Could not set pixel format: {e}")

    def get_PixelFormats(self):
        bgr8 = getattr(PixelFormat, "Bgr8", None)
        mono8 = getattr(PixelFormat, "Mono8", None)
        return bgr8, mono8

    def vmbSystem_getInstance(self):
        return VmbSystem.get_instance()

    def configure_stream(self, cam, camConfig):
        self.configure_vimba_camera(cam, camConfig)
        self._live_vimba_cam = cam
        self._live_vimba_enabled = True

    def start_stream(self, handler, vimba_profile):

        profile_spec = self.get_vimba_profile_spec(vimba_profile)
        stream_buffer_count = int(profile_spec["buffer_count"])
        self._live_vimba_cam.start_streaming(handler,
                                             buffer_count=stream_buffer_count,
                                             allocation_mode=AllocationMode.AllocAndAnnounceFrame,)

    def update_stream(self, camConfig):
        frame = None
        img_time = None

        with self._latest_lock:
            if self._latest_raw_frame is not None:
                frame = self._latest_raw_frame
                img_time = self._latest_raw_time
                self._latest_raw_frame = None
                self._latest_raw_time = None

        self._drain_live_vimba_updates(camConfig)

        if frame is not None:
            return frame, img_time
        return None, None

    def _vmbpy_frame_to_bgr(self, frame) -> NDArray:
        # If the camera isn't already producing Bgr8/Mono8, try to convert.
        bgr8, mono8 = self.get_PixelFormats()

        if bgr8 is not None:
            try:
                frame.convert_pixel_format(bgr8)
            except Exception:
                if mono8 is not None:
                    try:
                        frame.convert_pixel_format(mono8)
                    except Exception:
                        pass

        img = frame.as_opencv_image()
        if img is None:
            raise RuntimeError("Failed to export Vimba frame as OpenCV image.")

        img = np.ascontiguousarray(img)

        # Your pipeline generally behaves best with a 3-channel image.
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img = img.copy()

        return img

    def update_frame(self, frame, vimba_profile):
        profile_spec = self.get_vimba_profile_spec(vimba_profile)
        preview_max_dim = int(profile_spec["preview_max_dim"])

        img = self._vmbpy_frame_to_bgr(frame)

        if preview_max_dim > 0:
            h, w = img.shape[:2]
            max_dim = max(h, w)
            if max_dim > preview_max_dim:
                s = preview_max_dim / float(max_dim)
                new_w = min(w, preview_max_dim)
                new_h = min(h, preview_max_dim)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        ts = time.time()
        with self._latest_lock:
            self._latest_raw_frame = img
            self._latest_raw_time = ts

    def stop_stream(self):
        try:
            self._live_vimba_cam.stop_streaming()
        except Exception as e:
            LOG.warning(f"cam.stop_streaming() during teardown raised: {e}")

        self._live_vimba_enabled = False
        self._live_vimba_cam = None
        with self._live_vimba_cmd_lock:
            self._live_vimba_cmds.clear()

    def stop_lock(self):
        with self._latest_lock:
            self._latest_raw_frame = None
            self._latest_raw_time = None
