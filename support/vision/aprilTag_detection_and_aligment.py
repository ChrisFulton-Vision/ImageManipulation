import cv2
import numpy as np
from dataclasses import dataclass
from typing import Any

from numpy.typing import NDArray

import support.gui.UserSelectQueue as GuiQueue
import support.viz.colors as clr
from support.viz.CVFontScaling import small_text


@dataclass(slots=True)
class AprilTagDetectionResult:
    ids: list[np.ndarray]
    centers: NDArray | None
    corners_small: list[np.ndarray] | None
    refined_corners_per_marker: list[np.ndarray]
    gray_full: NDArray
    small_gray: NDArray

def inpaint_apriltags(markupFrame,
                      gray,
                      corners,
                      radius_px: int = 3,
                      dilate_px: int = 2,
                      method: int = cv2.INPAINT_TELEA,
                      feather: bool = True) -> None:
    mh, mw = markupFrame.shape[:2]
    gh, gw = gray.shape[:2]

    # scale factors from detection image to markup image
    sx = mw / float(gw)
    sy = mh / float(gh)

    # how much to pad each ROI beyond the exact tag corners
    pad = dilate_px + radius_px + 3

    for c in corners:
        pts = c.reshape(-1, 2).astype(np.float32)

        # scale detected corners from gray-space to markup-space
        pts_scaled = pts.copy()
        pts_scaled[:, 0] *= sx
        pts_scaled[:, 1] *= sy

        x_min = int(np.floor(pts_scaled[:, 0].min())) - pad
        x_max = int(np.ceil(pts_scaled[:, 0].max())) + pad
        y_min = int(np.floor(pts_scaled[:, 1].min())) - pad
        y_max = int(np.ceil(pts_scaled[:, 1].max())) + pad

        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(x_max, mw - 1)
        y_max = min(y_max, mh - 1)

        if x_max <= x_min or y_max <= y_min:
            continue

        roi_w = x_max - x_min + 1
        roi_h = y_max - y_min + 1
        mask_roi = np.zeros((roi_h, roi_w), dtype=np.uint8)

        pts_roi = pts_scaled.copy()
        pts_roi[:, 0] -= x_min
        pts_roi[:, 1] -= y_min
        pts_int = pts_roi.astype(np.int32)

        cv2.fillConvexPoly(mask_roi, pts_int, 255)

        if dilate_px > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)
            )
            mask_roi = cv2.dilate(mask_roi, k)

        frame_roi = markupFrame[y_min:y_max + 1, x_min:x_max + 1]
        inpainted_roi = cv2.inpaint(frame_roi, mask_roi, radius_px, method)

        if feather:
            blur_ks = max(3, 2 * radius_px + 1)
            soft = cv2.GaussianBlur(mask_roi, (blur_ks, blur_ks), 0).astype(np.float32) / 255.0
            soft = soft[..., None]

            base = frame_roi.astype(np.float32)
            inp = inpainted_roi.astype(np.float32)
            blended_roi = (soft * inp + (1.0 - soft) * base).astype(np.uint8)

            markupFrame[y_min:y_max + 1, x_min:x_max + 1] = blended_roi
        else:
            markupFrame[y_min:y_max + 1, x_min:x_max + 1] = inpainted_roi

def parse_apriltag_args(args: Any) -> GuiQueue.AprilTagDetectOpts:
    opts = GuiQueue.AprilTagDetectOpts()

    if args is None:
        return opts

    def set_scale(v: Any) -> None:
        if isinstance(v, bool):
            raise TypeError("AprilTag args: 'scale' must be numeric, not bool")
        if not isinstance(v, (int, float)):
            raise TypeError(f"AprilTag args: 'scale' must be int/float, got {type(v)}")
        opts.scale = float(v)

    def set_bool(name: str, v: Any) -> None:
        if isinstance(v, bool):
            setattr(opts, name, v)
            return
        if isinstance(v, int) and v in (0, 1):
            setattr(opts, name, bool(v))
            return
        raise TypeError(f"AprilTag args: '{name}' must be bool (or 0/1), got {type(v)}")

    if isinstance(args, dict):
        # Accept either GUI labels or internal field names.
        normalized = {}
        for k, v in args.items():
            key = GuiQueue.AprilTagDetectOpts.KEYMAP.get(str(k), str(k))
            normalized[key] = v

        if "scale" in normalized:
            set_scale(normalized["scale"])
        if "inpaint" in normalized:
            set_bool("inpaint", normalized["inpaint"])
        if "pnp" in normalized:
            set_bool("pnp", normalized["pnp"])
        if "qnp" in normalized:
            set_bool("qnp", normalized["qnp"])

        return opts

    raise TypeError(f"AprilTag args: unsupported args type {type(args)}")


def detect_apriltags_refined(
    detector,
    markup_frame: NDArray,
    scale: float,
) -> AprilTagDetectionResult:
    """
    Detect AprilTags on a downscaled grayscale image, then refine corners
    at full resolution using cornerSubPix.
    """
    if detector is None:
        raise ValueError("detector must not be None")

    gray_full = cv2.cvtColor(markup_frame, cv2.COLOR_BGR2GRAY)
    h, w = gray_full.shape[:2]

    if not (0.05 <= float(scale) <= 1.0):
        scale = 0.6

    sw = max(1, int(w * scale))
    sh = max(1, int(h * scale))
    small_gray = cv2.resize(gray_full, (sw, sh), interpolation=cv2.INTER_AREA)

    corners_small, ids, _rejected = detector.detectMarkers(small_gray)

    if corners_small is None or ids is None or len(corners_small) == 0:
        return AprilTagDetectionResult(
            ids=[],
            centers=None,
            corners_small=None,
            refined_corners_per_marker=[],
            gray_full=gray_full,
            small_gray=small_gray,
        )

    all_pts = []
    marker_lengths = []

    for c in corners_small:
        pts = c.reshape(-1, 2).astype(np.float32) / float(scale)
        marker_lengths.append(len(pts))
        all_pts.append(pts)

    all_pts = np.concatenate(all_pts, axis=0).reshape(-1, 1, 2)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        20,
        0.01,
    )
    cv2.cornerSubPix(gray_full, all_pts, (5, 5), (-1, -1), criteria)

    refined_corners_per_marker = []
    idx0 = 0
    for length in marker_lengths:
        refined_corners_per_marker.append(
            all_pts[idx0:idx0 + length].reshape(-1, 2).copy()
        )
        idx0 += length

    centers = None
    ids_out: list[np.ndarray] = []

    for corners, idx in zip(refined_corners_per_marker, ids):
        pix_center = np.mean(corners, axis=0).astype(np.int32)
        ids_out.append(idx)

        if centers is None:
            centers = np.array(pix_center, dtype=np.float32)
        else:
            centers = np.vstack((centers, pix_center.astype(np.float32)))

    return AprilTagDetectionResult(
        ids=ids_out,
        centers=centers,
        corners_small=list(corners_small),
        refined_corners_per_marker=refined_corners_per_marker,
        gray_full=gray_full,
        small_gray=small_gray,
    )


def draw_apriltag_detections(
    markup_frame: NDArray,
    refined_corners_per_marker: list[np.ndarray],
    ids: list[np.ndarray],
) -> None:
    for corners, idx in zip(refined_corners_per_marker, ids):
        polyline = [corners.astype(np.int32).reshape((-1, 1, 2))]
        pix_center = np.mean(corners, axis=0).astype(np.int32)

        cv2.polylines(markup_frame, polyline, True, clr.HUD_GREEN, 4, lineType=cv2.FILLED)
        cv2.putText(markup_frame, str(idx[0]), tuple(pix_center),
                    cv2.FONT_HERSHEY_SIMPLEX, small_text(markup_frame.shape[0]), clr.HUD_GREEN, 4)
        cv2.putText(markup_frame, str(idx[0]), tuple(pix_center),
                    cv2.FONT_HERSHEY_SIMPLEX, small_text(markup_frame.shape[0]), (0, 0, 0), 1)


def inpaint_apriltags(markup_frame: NDArray,
                      gray_small: NDArray,
                      corners_small,
                      radius_px: int = 3,
                      dilate_px: int = 2,
                      method: int = cv2.INPAINT_TELEA,
                      feather: bool = True) -> None:
    mh, mw = markup_frame.shape[:2]
    gh, gw = gray_small.shape[:2]

    sx = mw / float(gw)
    sy = mh / float(gh)

    pad = dilate_px + radius_px + 3

    for c in corners_small:
        pts = c.reshape(-1, 2).astype(np.float32)

        pts_scaled = pts.copy()
        pts_scaled[:, 0] *= sx
        pts_scaled[:, 1] *= sy

        x_min = int(np.floor(pts_scaled[:, 0].min())) - pad
        x_max = int(np.ceil(pts_scaled[:, 0].max())) + pad
        y_min = int(np.floor(pts_scaled[:, 1].min())) - pad
        y_max = int(np.ceil(pts_scaled[:, 1].max())) + pad

        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(x_max, mw - 1)
        y_max = min(y_max, mh - 1)
        if x_max <= x_min or y_max <= y_min:
            continue

        roi_w = x_max - x_min + 1
        roi_h = y_max - y_min + 1
        mask_roi = np.zeros((roi_h, roi_w), dtype=np.uint8)

        pts_roi = pts_scaled.copy()
        pts_roi[:, 0] -= x_min
        pts_roi[:, 1] -= y_min
        pts_int = pts_roi.astype(np.int32)

        cv2.fillConvexPoly(mask_roi, pts_int, 255)

        if dilate_px > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)
            )
            mask_roi = cv2.dilate(mask_roi, k)

        frame_roi = markup_frame[y_min:y_max + 1, x_min:x_max + 1]
        inpainted_roi = cv2.inpaint(frame_roi, mask_roi, radius_px, method)

        if feather:
            blur_ks = max(3, 2 * radius_px + 1)
            soft = cv2.GaussianBlur(mask_roi, (blur_ks, blur_ks), 0).astype(np.float32) / 255.0
            soft = soft[..., None]

            base = frame_roi.astype(np.float32)
            inp = inpainted_roi.astype(np.float32)
            blended_roi = (soft * inp + (1.0 - soft) * base).astype(np.uint8)

            markup_frame[y_min:y_max + 1, x_min:x_max + 1] = blended_roi
        else:
            markup_frame[y_min:y_max + 1, x_min:x_max + 1] = inpainted_roi