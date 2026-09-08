import argparse
import math
from support.vision.calibration import Calibration, default_864_cam, distort_points_px, undistort_points_px
import numpy as np
import cv2
from functools import partial
import time
from pathlib import Path

from PySide6.QtCore import QSignalBlocker, QTimer, Qt, Signal
from PySide6.QtGui import QCloseEvent, QImage, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

DIST_TIME = 5
FPS = 30
PAUSE_TIME = 5.0

GUI_PARAM_NAMES = ("k1", "k2", "p1", "p2", "k3")
GUI_PARAM_RANGES = {
    "k1": (-5.0, 5.0),
    "k2": (-5.0, 5.0),
    "p1": (-0.10, 0.10),
    "p2": (-0.10, 0.10),
    "k3": (-2.0, 2.0),
}
GUI_RANDOMIZER_PERIODS_S = {
    "k1": (9.0, 11.0),
    "k2": (8.0, 13.0),
    "p1": (7.0, 15.0),
    "p2": (6.0, 17.0),
    "k3": (5.0, 19.0),
}
GUI_RANDOMIZER_MIX = {
    "k1": (0.72, 0.23),
    "k2": (0.58, 0.31),
    "p1": (0.70, 0.22),
    "p2": (0.62, 0.26),
    "k3": (0.66, 0.28),
}
GUI_RANDOMIZER_PHASES = {
    "k1": (0.0, 0.8),
    "k2": (1.3, 2.6),
    "p1": (2.1, 4.2),
    "p2": (3.4, 1.1),
    "k3": (4.5, 3.0),
}
GUI_DEFAULT_NUM_COL_LINES = 21
GUI_DEFAULT_NUM_ROW_LINES = 21
GUI_MIN_GRID_LINES = 2
GUI_MAX_GRID_LINES = 101


def _mesh_grid_positions(
        *,
        width: int,
        height: int,
        num_col_lines: int,
        num_row_lines: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_positions = np.linspace(0.0, width - 1.0, num_col_lines, dtype=np.float64)
    y_positions = np.linspace(0.0, height - 1.0, num_row_lines, dtype=np.float64)
    return x_positions, y_positions


def build_mesh_grid(
        *,
        width: int,
        height: int,
        num_col_lines: int,
        num_row_lines: int,
        num_line_samples: int = 200,
        margin_frac: float = 0.15,
) -> list[np.ndarray]:
    """Build a regular mesh in image space with modest endpoint overrun.

    The line locations remain inside the image so the undistort comparison stays
    interpretable. Only the sampled endpoints are extended slightly past the
    image bounds to keep compressed distorted lines visible near the edges.
    """
    margin_x = float(width) * float(margin_frac)
    margin_y = float(height) * float(margin_frac)

    x_positions, y_positions = _mesh_grid_positions(
        width=width,
        height=height,
        num_col_lines=num_col_lines,
        num_row_lines=num_row_lines,
    )
    sample_x = np.linspace(-margin_x, (width - 1.0) + margin_x, num_line_samples, dtype=np.float64)
    sample_y = np.linspace(-margin_y, (height - 1.0) + margin_y, num_line_samples, dtype=np.float64)

    mesh_lines: list[np.ndarray] = []
    for x in x_positions:
        mesh_lines.append(np.column_stack((np.full_like(sample_y, x), sample_y)))
    for y in y_positions:
        mesh_lines.append(np.column_stack((sample_x, np.full_like(sample_x, y))))
    return mesh_lines


def build_mesh_intersections(
        *,
        width: int,
        height: int,
        num_col_lines: int,
        num_row_lines: int,
) -> np.ndarray:
    """Build the exact mesh intersection points in image space."""
    x_positions, y_positions = _mesh_grid_positions(
        width=width,
        height=height,
        num_col_lines=num_col_lines,
        num_row_lines=num_row_lines,
    )
    grid_x, grid_y = np.meshgrid(x_positions, y_positions, indexing="xy")
    return np.column_stack((grid_x.ravel(), grid_y.ravel()))


def distort_mesh_grid(calibration: Calibration, mesh_lines: list[np.ndarray]) -> list[np.ndarray]:
    """Apply forward distortion to each sampled mesh polyline."""
    return [distort_points_px(calibration, line_points) for line_points in mesh_lines]


def undistort_mesh_grid(calibration: Calibration,
                        mesh_lines: list[np.ndarray],
                        num_fp_steps: int = 5,
                        num_newton_steps: int | None = None) -> list[np.ndarray]:
    """Apply inverse distortion to each sampled mesh polyline.

    ``num_fp_steps`` is forwarded to :func:`undistort_points_px` as
    ``fp_steps`` so the demo can trade speed vs convergence explicitly.
    """
    return [undistort_points_px(calibration,
                                line_points,
                                fp_steps=num_fp_steps,
                                newton_steps=num_newton_steps) for line_points in mesh_lines]


def undistort_mesh_grid_opencv(calibration: Calibration, mesh_lines: list[np.ndarray]) -> list[np.ndarray]:
    """Apply OpenCV's inverse distortion to each sampled mesh polyline."""
    K = calibration.getCameraMatrix().astype(np.float64, copy=False)
    D = calibration.getDistortion().astype(np.float64, copy=False)

    undistorted_lines: list[np.ndarray] = []
    for line_points in mesh_lines:
        pts_cv = np.asarray(line_points, dtype=np.float64).reshape(-1, 1, 2)
        if calibration.fisheye:
            undistorted = cv2.fisheye.undistortPoints(pts_cv, K, D, R=None, P=K)
        else:
            undistorted = cv2.undistortPoints(pts_cv, K, D, R=None, P=K)
        undistorted_lines.append(undistorted.reshape(-1, 2))
    return undistorted_lines


def distort_mesh_points(calibration: Calibration, points: np.ndarray) -> np.ndarray:
    return distort_points_px(calibration, np.asarray(points, dtype=np.float64))


def build_preview_calibration(*, preview_size: int = 900) -> Calibration:
    cal = default_864_cam().copy()
    dist = cal.getDistortion().astype(np.float64, copy=True)
    # dist[:2] *= 8.0
    # dist[4] *= 8.0
    cal.setDistortion(dist)

    scale = float(preview_size) / float(cal.width)
    cal.width = int(round(cal.width * scale))
    cal.height = int(round(cal.height * scale))
    cal.fx *= scale
    cal.fy *= scale
    cal.cx = 0.5 * (cal.width - 1.0)
    cal.cy = 0.5 * (cal.height - 1.0)
    return cal


def undistort_mesh_points(calibration: Calibration,
                          points: np.ndarray,
                          num_fp_steps: int = 5,
                          num_newton_steps: int | None = None) -> np.ndarray:
    return undistort_points_px(
        calibration,
        np.asarray(points, dtype=np.float64),
        fp_steps=num_fp_steps,
        newton_steps=num_newton_steps,
    )


def undistort_mesh_points_opencv(calibration: Calibration, points: np.ndarray) -> np.ndarray:
    K = calibration.getCameraMatrix().astype(np.float64, copy=False)
    D = calibration.getDistortion().astype(np.float64, copy=False)
    pts_cv = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    if calibration.fisheye:
        undistorted = cv2.fisheye.undistortPoints(pts_cv, K, D, R=None, P=K)
    else:
        undistorted = cv2.undistortPoints(pts_cv, K, D, R=None, P=K)
    return undistorted.reshape(-1, 2)


def interpolate_mesh_grid(
        clean_lines: list[np.ndarray],
        distorted_lines: list[np.ndarray],
        t: float,
) -> list[np.ndarray]:
    if len(clean_lines) != len(distorted_lines):
        raise ValueError("clean_lines and distorted_lines must have the same number of lines")

    t = float(np.clip(t, 0.0, 1.0))
    interpolated_lines: list[np.ndarray] = []
    for clean_line, distorted_line in zip(clean_lines, distorted_lines, strict=True):
        if clean_line.shape != distorted_line.shape:
            raise ValueError("clean and distorted mesh lines must have matching shapes")
        interpolated_lines.append((1.0 - t) * clean_line + t * distorted_line)
    return interpolated_lines


def interpolate_points(start_points: np.ndarray, end_points: np.ndarray, t: float) -> np.ndarray:
    if start_points.shape != end_points.shape:
        raise ValueError("start_points and end_points must have matching shapes")
    t = float(np.clip(t, 0.0, 1.0))
    return (1.0 - t) * start_points + t * end_points


def hsv_to_bgr(h: float, s: int = 255, v: int = 255) -> tuple[int, int, int]:
    hsv = np.array([[[h % 180.0, s, v]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_hue_mesh(
        image: np.ndarray,
        *,
        clean_lines: list[np.ndarray],
        mesh_lines: list[np.ndarray],
        line_thickness: int = 3,
        interpolation: float = 1.0,
) -> np.ndarray:
    mesh = image.copy()
    used_lines = interpolate_mesh_grid(clean_lines, mesh_lines, interpolation)

    total_lines = max(1, len(used_lines) - 1)
    hue_step = 80.0 / total_lines

    for line_index, points in enumerate(used_lines):
        color = hsv_to_bgr(line_index * hue_step + 60.0)
        points_i32 = np.round(points).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(mesh, [points_i32], isClosed=False, color=color, thickness=line_thickness, lineType=cv2.LINE_AA)

    return mesh


def residual_color_bgr(residual_px: float, max_residual_px: float) -> tuple[int, int, int]:
    frac = float(np.clip(residual_px / max_residual_px, 0.0, 1.0)) ** 0.25
    hue = (1.0 - frac) * 60.0
    return hsv_to_bgr(hue)


def draw_residual_points(
        image: np.ndarray,
        *,
        start_points: np.ndarray,
        end_points: np.ndarray,
        reference_points: np.ndarray,
        interpolation: float,
        max_residual_px: float,
        radius: int = 3,
) -> np.ndarray:
    overlay = image.copy()
    current_points = interpolate_points(start_points, end_points, interpolation)
    residuals_px = np.linalg.norm(reference_points - current_points, axis=1)

    for point, residual_px in zip(current_points, residuals_px, strict=True):
        color = residual_color_bgr(float(residual_px), max_residual_px)
        point_i32 = tuple(np.round(point).astype(np.int32))
        cv2.circle(overlay, point_i32, radius, color, thickness=-1, lineType=cv2.LINE_AA)

    return overlay


def draw_frame(img,
               text,
               text_func,
               title,
               inter):
    if text is not None and text_func is not None:
        draw_text = text + f': {inter * 100.0:.2f}%'
        text_func(img=img,
                  text=draw_text)
    cv2.imshow("Distort Demo", img)


def make_side_by_side(left_img: np.ndarray, right_img: np.ndarray) -> np.ndarray:
    return np.hstack((left_img, right_img))


def add_panel_label(img: np.ndarray, text: str) -> np.ndarray:
    labeled = img.copy()
    cv2.putText(
        labeled,
        text,
        (int(0.04 * labeled.shape[1]), int(0.08 * labeled.shape[0])),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )
    return labeled


def build_pixel_grid(*, width: int, height: int) -> np.ndarray:
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
        indexing="xy",
    )
    return np.column_stack((grid_x.ravel(), grid_y.ravel()))


def remap_image(image: np.ndarray, sample_points: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    map_x = sample_points[:, 0].reshape(height, width).astype(np.float32)
    map_y = sample_points[:, 1].reshape(height, width).astype(np.float32)
    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def interpolate_distortion_calibration(calibration: Calibration, t: float) -> Calibration:
    interpolated = calibration.copy()
    interpolated.setDistortion(calibration.getDistortion() * float(np.clip(t, 0.0, 1.0)))
    return interpolated


def distort_demo_image(
        calibration: Calibration,
        image: np.ndarray,
        *,
        pixel_grid: np.ndarray | None = None,
) -> np.ndarray:
    if pixel_grid is None:
        pixel_grid = build_pixel_grid(width=calibration.width, height=calibration.height)
    source_points = undistort_points_px(calibration, pixel_grid, fp_steps=11, newton_steps=3)
    return remap_image(image, source_points)


def undistort_demo_image(
        calibration: Calibration,
        image: np.ndarray,
        *,
        pixel_grid: np.ndarray | None = None,
) -> np.ndarray:
    if pixel_grid is None:
        pixel_grid = build_pixel_grid(width=calibration.width, height=calibration.height)
    source_points = distort_points_px(calibration, pixel_grid)
    return remap_image(image, source_points)


def load_demo_image(image_path: Path, *, width: int, height: int) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    if image.shape[1] != width or image.shape[0] != height:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image


def render_comparison_frame(
        base_img: np.ndarray,
        *,
        left_start: list[np.ndarray],
        left_end: list[np.ndarray],
        right_start: list[np.ndarray],
        right_end: list[np.ndarray],
        left_start_points: np.ndarray,
        left_end_points: np.ndarray,
        right_start_points: np.ndarray,
        right_end_points: np.ndarray,
        left_reference_points: np.ndarray,
        right_reference_points: np.ndarray,
        interpolation: float,
        left_label: str,
        right_label: str,
) -> np.ndarray:
    max_residual_px = 0.10 * float(base_img.shape[1])
    left_img = add_panel_label(
        draw_residual_points(
            draw_hue_mesh(base_img, clean_lines=left_start, mesh_lines=left_end, interpolation=interpolation),
            start_points=left_start_points,
            end_points=left_end_points,
            reference_points=left_reference_points,
            interpolation=interpolation,
            max_residual_px=max_residual_px,
        ),
        left_label,
    )
    right_img = add_panel_label(
        draw_residual_points(
            draw_hue_mesh(base_img, clean_lines=right_start, mesh_lines=right_end, interpolation=interpolation),
            start_points=right_start_points,
            end_points=right_end_points,
            reference_points=right_reference_points,
            interpolation=interpolation,
            max_residual_px=max_residual_px,
        ),
        right_label,
    )
    return make_side_by_side(left_img, right_img)


def animate_comparison_phase(
        base_img: np.ndarray,
        *,
        left_start: list[np.ndarray],
        left_end: list[np.ndarray],
        right_start: list[np.ndarray],
        right_end: list[np.ndarray],
        left_label: str,
        right_label: str,
        duration_s: float,
        overlay_text: str,
        text_func,
) -> None:
    t0 = time.perf_counter()
    while True:
        elapsed_s = time.perf_counter() - t0
        interpolation = min(elapsed_s / duration_s, 1.0)
        frame = render_comparison_frame(
            base_img,
            left_start=left_start,
            left_end=left_end,
            right_start=right_start,
            right_end=right_end,
            interpolation=interpolation,
            left_label=left_label,
            right_label=right_label,
        )
        draw_frame(
            frame,
            text=overlay_text,
            text_func=text_func,
            title="Distort Demo",
            inter=interpolation,
        )
        if cv2.waitKey(1) == 27 or interpolation >= 1.0:
            break


def write_demo_video(
        output_path: Path,
        *,
        base_img: np.ndarray,
        clean_mesh: list[np.ndarray],
        distorted_mesh: list[np.ndarray],
        clean_points: np.ndarray,
        distorted_points: np.ndarray,
        left_label: str,
        right_label: str,
        phases: list[dict],
        fps: int = FPS,
        pause_time_s: float = PAUSE_TIME,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_size = (base_img.shape[1] * 2, base_img.shape[0])
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        frame_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    add_text = partial(
        cv2.putText,
        org=(int(0.1 * base_img.shape[1]), int(0.9 * base_img.shape[0])),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1.0,
        color=(255, 255, 255),
    )

    def render_frame(
            *,
            left_start: list[np.ndarray],
            left_end: list[np.ndarray],
            right_start: list[np.ndarray],
            right_end: list[np.ndarray],
            left_start_points: np.ndarray,
            left_end_points: np.ndarray,
            right_start_points: np.ndarray,
            right_end_points: np.ndarray,
            left_reference_points: np.ndarray,
            right_reference_points: np.ndarray,
            interpolation: float,
            overlay_text: str,
    ) -> np.ndarray:
        frame = render_comparison_frame(
            base_img,
            left_start=left_start,
            left_end=left_end,
            right_start=right_start,
            right_end=right_end,
            left_start_points=left_start_points,
            left_end_points=left_end_points,
            right_start_points=right_start_points,
            right_end_points=right_end_points,
            left_reference_points=left_reference_points,
            right_reference_points=right_reference_points,
            interpolation=interpolation,
            left_label=left_label,
            right_label=right_label,
        )
        draw_text = f"{overlay_text}: {interpolation * 100.0:.2f}%"
        add_text(img=frame, text=draw_text)
        return frame

    def write_phase(
            *,
            left_start: list[np.ndarray],
            left_end: list[np.ndarray],
            right_start: list[np.ndarray],
            right_end: list[np.ndarray],
            left_start_points: np.ndarray,
            left_end_points: np.ndarray,
            right_start_points: np.ndarray,
            right_end_points: np.ndarray,
            left_reference_points: np.ndarray,
            right_reference_points: np.ndarray,
            duration_s: float,
            overlay_text: str,
    ) -> None:
        frame_count = max(1, int(round(duration_s * fps)))
        for idx in range(frame_count):
            interpolation = 1.0 if frame_count == 1 else idx / (frame_count - 1)
            frame = render_frame(
                left_start=left_start,
                left_end=left_end,
                right_start=right_start,
                right_end=right_end,
                left_start_points=left_start_points,
                left_end_points=left_end_points,
                right_start_points=right_start_points,
                right_end_points=right_end_points,
                left_reference_points=left_reference_points,
                right_reference_points=right_reference_points,
                interpolation=interpolation,
                overlay_text=overlay_text,
            )
            writer.write(frame)

    try:
        write_phase(
            left_start=clean_mesh,
            left_end=distorted_mesh,
            right_start=clean_mesh,
            right_end=distorted_mesh,
            left_start_points=clean_points,
            left_end_points=distorted_points,
            right_start_points=clean_points,
            right_end_points=distorted_points,
            left_reference_points=clean_points,
            right_reference_points=clean_points,
            duration_s=pause_time_s,
            overlay_text="Undistort Compare",
        )
        for phase in phases:
            write_phase(
                left_start=phase["left_start"],
                left_end=phase["left_end"],
                right_start=phase["right_start"],
                right_end=phase["right_end"],
                left_start_points=phase["left_start_points"],
                left_end_points=phase["left_end_points"],
                right_start_points=phase["right_start_points"],
                right_end_points=phase["right_end_points"],
                left_reference_points=phase["left_reference_points"],
                right_reference_points=phase["right_reference_points"],
                duration_s=phase["duration_s"],
                overlay_text=phase["overlay_text"],
            )
        final_phase = phases[-1]
        write_phase(
            left_start=final_phase["left_end"],
            left_end=final_phase["left_end"],
            right_start=final_phase["right_end"],
            right_end=final_phase["right_end"],
            left_start_points=final_phase["left_end_points"],
            left_end_points=final_phase["left_end_points"],
            right_start_points=final_phase["right_end_points"],
            right_end_points=final_phase["right_end_points"],
            left_reference_points=final_phase["left_reference_points"],
            right_reference_points=final_phase["right_reference_points"],
            duration_s=pause_time_s,
            overlay_text="Final Undistorted",
        )
    finally:
        writer.release()

    return output_path


def write_image_demo_video(
        output_path: Path,
        *,
        calibration: Calibration,
        clean_image: np.ndarray,
        fps: int = FPS,
        pause_time_s: float = PAUSE_TIME,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_size = (clean_image.shape[1], clean_image.shape[0])
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        frame_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    add_text = partial(
        cv2.putText,
        org=(int(0.06 * clean_image.shape[1]), int(0.9 * clean_image.shape[0])),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1.0,
        color=(255, 255, 255),
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    pixel_grid = build_pixel_grid(width=calibration.width, height=calibration.height)
    distorted_image = distort_demo_image(calibration, clean_image, pixel_grid=pixel_grid)
    restored_image = undistort_demo_image(calibration, distorted_image, pixel_grid=pixel_grid)

    def labeled_frame(image: np.ndarray, overlay_text: str, interpolation: float, label: str) -> np.ndarray:
        frame = add_panel_label(image, label)
        add_text(img=frame, text=f"{overlay_text}: {interpolation * 100.0:.2f}%")
        return frame

    def write_distortion_transition(
            *,
            duration_s: float,
            overlay_text: str,
            frame_builder,
            right_label: str,
    ) -> None:
        frame_count = max(1, int(round(duration_s * fps)))
        for idx in range(frame_count):
            interpolation = 1.0 if frame_count == 1 else idx / (frame_count - 1)
            writer.write(labeled_frame(frame_builder(interpolation), overlay_text, interpolation, right_label))

    try:
        write_distortion_transition(
            duration_s=DIST_TIME,
            overlay_text="Forward Distortion",
            frame_builder=lambda interpolation: distort_demo_image(
                interpolate_distortion_calibration(calibration, interpolation),
                clean_image,
                pixel_grid=pixel_grid,
            ),
            right_label="Distorted",
        )
        write_distortion_transition(
            duration_s=DIST_TIME,
            overlay_text="Inverse Distortion",
            frame_builder=lambda interpolation: undistort_demo_image(
                interpolate_distortion_calibration(calibration, interpolation),
                distorted_image,
                pixel_grid=pixel_grid,
            ),
            right_label="Undistorted",
        )
        write_distortion_transition(
            duration_s=DIST_TIME,
            overlay_text="Undo Undistortion",
            frame_builder=lambda interpolation: undistort_demo_image(
                interpolate_distortion_calibration(calibration, 1.0 - interpolation),
                distorted_image,
                pixel_grid=pixel_grid,
            ),
            right_label="Distorted",
        )
        write_distortion_transition(
            duration_s=DIST_TIME,
            overlay_text="Undo Distortion",
            frame_builder=lambda interpolation: distort_demo_image(
                interpolate_distortion_calibration(calibration, 1.0 - interpolation),
                clean_image,
                pixel_grid=pixel_grid,
            ),
            right_label="Clean",
        )

    finally:
        writer.release()

    return output_path


def render_gui_preview(
        calibration: Calibration,
        *,
        num_col_lines: int = GUI_DEFAULT_NUM_COL_LINES,
        num_row_lines: int = GUI_DEFAULT_NUM_ROW_LINES,
        num_line_samples: int = 160,
) -> np.ndarray:
    base = np.zeros((calibration.height, calibration.width, 3), dtype=np.uint8)
    clean_mesh = build_mesh_grid(
        width=calibration.width,
        height=calibration.height,
        num_col_lines=num_col_lines,
        num_row_lines=num_row_lines,
        num_line_samples=num_line_samples,
    )
    clean_points = build_mesh_intersections(
        width=calibration.width,
        height=calibration.height,
        num_col_lines=num_col_lines,
        num_row_lines=num_row_lines,
    )
    distorted_mesh = distort_mesh_grid(calibration, clean_mesh)
    distorted_points = distort_mesh_points(calibration, clean_points)
    preview = draw_residual_points(
        draw_hue_mesh(base, clean_lines=clean_mesh, mesh_lines=distorted_mesh, line_thickness=2),
        start_points=clean_points,
        end_points=distorted_points,
        reference_points=clean_points,
        interpolation=1.0,
        max_residual_px=0.10 * float(calibration.width),
        radius=3,
    )
    subtitle = "Brown-Conrady Distortion"
    values = "  ".join(f"{name}={getattr(calibration, name):+.5f}" for name in GUI_PARAM_NAMES)
    cv2.putText(
        preview,
        subtitle,
        (24, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        values,
        (24, calibration.height - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        lineType=cv2.LINE_AA,
    )
    return preview


class _ParameterControl(QWidget):
    """A precise float editor backed by a slider and a spin box."""

    valueChanged = Signal(float)
    _SLIDER_SCALE = 1_000_000

    def __init__(self, name: str, minimum: float, maximum: float, parent: QWidget | None = None):
        super().__init__(parent)
        self.name = name

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(4)

        self.name_label = QLabel(name)
        self.name_label.setStyleSheet("font-weight: 600;")
        layout.addWidget(self.name_label, 0, 0)

        self.spin_box = QDoubleSpinBox()
        self.spin_box.setDecimals(6)
        self.spin_box.setRange(minimum, maximum)
        self.spin_box.setSingleStep(0.0001 if max(abs(minimum), abs(maximum)) <= 0.1 else 0.01)
        self.spin_box.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.spin_box.setFixedWidth(130)
        layout.addWidget(self.spin_box, 0, 1)
        layout.setColumnStretch(0, 1)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(
            round(minimum * self._SLIDER_SCALE),
            round(maximum * self._SLIDER_SCALE),
        )
        layout.addWidget(self.slider, 1, 0, 1, 2)

        self.slider.valueChanged.connect(self._on_slider_changed)
        self.spin_box.valueChanged.connect(self._on_spin_changed)

    def value(self) -> float:
        return float(self.spin_box.value())

    def setValue(self, value: float) -> None:
        value = float(value)
        with QSignalBlocker(self.slider), QSignalBlocker(self.spin_box):
            self.slider.setValue(round(value * self._SLIDER_SCALE))
            self.spin_box.setValue(value)

    def _on_slider_changed(self, slider_value: int) -> None:
        value = slider_value / self._SLIDER_SCALE
        with QSignalBlocker(self.spin_box):
            self.spin_box.setValue(value)
        self.valueChanged.emit(value)

    def _on_spin_changed(self, value: float) -> None:
        with QSignalBlocker(self.slider):
            self.slider.setValue(round(value * self._SLIDER_SCALE))
        self.valueChanged.emit(float(value))


class DistortionGui(QWidget):
    """Native PySide6 viewer for the Brown-Conrady distortion model."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Brown-Conrady Distortion Viewer")
        self.resize(1320, 980)
        self.setMinimumSize(1180, 900)

        self.preview_cal = build_preview_calibration()
        self.parameter_controls: dict[str, _ParameterControl] = {}
        self.num_col_lines = GUI_DEFAULT_NUM_COL_LINES
        self.num_row_lines = GUI_DEFAULT_NUM_ROW_LINES
        self.preview_pixmap: QPixmap | None = None
        self._randomizer_active = False
        self._randomizer_t0 = 0.0

        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self.render_preview)

        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._fit_preview)

        self._randomizer_timer = QTimer(self)
        self._randomizer_timer.setInterval(33)
        self._randomizer_timer.timeout.connect(self.step_randomizer)

        self._build_ui()
        self.reset_defaults()

    def _build_ui(self) -> None:
        self.setStyleSheet(
            """
            DistortionGui { background: #16191d; color: #f1f3f5; }
            QFrame#panel { background: #20242a; border: 1px solid #343a42; border-radius: 8px; }
            QLabel { color: #f1f3f5; }
            QPushButton { background: #2f6fed; color: white; border: 0; border-radius: 5px;
                          min-height: 30px; padding: 3px 10px; font-weight: 600; }
            QPushButton:hover { background: #397af5; }
            QPushButton:pressed { background: #255dcc; }
            QSpinBox, QDoubleSpinBox { background: #171a1f; color: #f1f3f5;
                                      border: 1px solid #4a515b; border-radius: 4px;
                                      padding: 3px; padding-right: 22px; }
            QSlider::groove:horizontal { background: #3c424b; height: 5px; border-radius: 2px; }
            QSlider::sub-page:horizontal { background: #2f6fed; border-radius: 2px; }
            QSlider::handle:horizontal { background: #f1f3f5; width: 14px; margin: -5px 0;
                                         border-radius: 7px; }
            """
        )

        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(16)

        controls = QFrame()
        controls.setObjectName("panel")
        controls.setFixedWidth(310)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(16, 16, 16, 16)
        controls_layout.setSpacing(12)

        title = QLabel("Distortion Parameters")
        title.setStyleSheet("font-size: 20px; font-weight: 700;")
        controls_layout.addWidget(title)

        for name in GUI_PARAM_NAMES:
            self._build_slider_row(controls_layout, name=name)

        controls_layout.addLayout(self._build_grid_controls())

        reset_button = QPushButton("Reset Defaults")
        reset_button.clicked.connect(self.reset_defaults)
        controls_layout.addWidget(reset_button)

        zero_button = QPushButton("Set Zeros")
        zero_button.clicked.connect(self.set_zeros)
        controls_layout.addWidget(zero_button)

        self.randomizer_button = QPushButton("Start Randomizer")
        self.randomizer_button.clicked.connect(self.toggle_randomizer)
        controls_layout.addWidget(self.randomizer_button)

        description = QLabel(
            "The preview uses the forward Brown-Conrady model with a live distorted mesh."
        )
        description.setWordWrap(True)
        description.setStyleSheet("color: #b7bec8;")
        controls_layout.addWidget(description)
        controls_layout.addStretch(1)
        root_layout.addWidget(controls)

        preview_frame = QFrame()
        preview_frame.setObjectName("panel")
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(12, 12, 12, 12)
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview_label.setMinimumSize(640, 480)
        self.preview_label.setStyleSheet("background: #050607; border-radius: 4px;")
        preview_layout.addWidget(self.preview_label)
        root_layout.addWidget(preview_frame, 1)

    def _build_slider_row(self, parent_layout: QVBoxLayout, *, name: str) -> None:
        minimum, maximum = GUI_PARAM_RANGES[name]
        control = _ParameterControl(name, minimum, maximum)
        control.valueChanged.connect(lambda value, param_name=name: self.on_slider_change(param_name, value))
        self.parameter_controls[name] = control
        parent_layout.addWidget(control)

    def _build_grid_controls(self) -> QGridLayout:
        layout = QGridLayout()
        layout.setContentsMargins(0, 8, 0, 4)
        layout.setHorizontalSpacing(10)
        layout.addWidget(QLabel("Mesh Grid"), 0, 0, 1, 2)
        layout.addWidget(QLabel("Columns"), 1, 0)
        layout.addWidget(QLabel("Rows"), 1, 1)

        self.num_col_lines_spin = QSpinBox()
        self.num_col_lines_spin.setRange(GUI_MIN_GRID_LINES, GUI_MAX_GRID_LINES)
        self.num_col_lines_spin.setValue(GUI_DEFAULT_NUM_COL_LINES)
        self.num_col_lines_spin.setMinimumWidth(100)
        layout.addWidget(self.num_col_lines_spin, 2, 0)

        self.num_row_lines_spin = QSpinBox()
        self.num_row_lines_spin.setRange(GUI_MIN_GRID_LINES, GUI_MAX_GRID_LINES)
        self.num_row_lines_spin.setValue(GUI_DEFAULT_NUM_ROW_LINES)
        self.num_row_lines_spin.setMinimumWidth(100)
        layout.addWidget(self.num_row_lines_spin, 2, 1)

        self.num_col_lines_spin.valueChanged.connect(self.apply_grid_dimensions)
        self.num_row_lines_spin.valueChanged.connect(self.apply_grid_dimensions)
        return layout

    def on_slider_change(self, name: str, value: float) -> None:
        self.stop_randomizer()
        setattr(self.preview_cal, name, float(value))
        self.schedule_render()

    def apply_grid_dimensions(self, _value: int | None = None) -> None:
        num_col_lines = self.num_col_lines_spin.value()
        num_row_lines = self.num_row_lines_spin.value()
        changed = (num_col_lines != self.num_col_lines) or (num_row_lines != self.num_row_lines)
        self.num_col_lines = num_col_lines
        self.num_row_lines = num_row_lines
        if changed:
            self.schedule_render()

    def reset_defaults(self) -> None:
        self.stop_randomizer()
        default_cal = build_preview_calibration()
        self.apply_coefficients({name: float(getattr(default_cal, name)) for name in GUI_PARAM_NAMES})

    def set_zeros(self) -> None:
        self.stop_randomizer()
        self.apply_coefficients({name: 0.0 for name in GUI_PARAM_NAMES})

    def apply_coefficients(self, coefficients: dict[str, float]) -> None:
        for name in GUI_PARAM_NAMES:
            value = float(coefficients[name])
            self.parameter_controls[name].setValue(value)
            setattr(self.preview_cal, name, value)
        self.schedule_render()

    def toggle_randomizer(self) -> None:
        if self._randomizer_active:
            self.stop_randomizer()
            return
        self._randomizer_active = True
        self._randomizer_t0 = time.perf_counter()
        self.randomizer_button.setText("Stop Randomizer")
        self.step_randomizer()
        self._randomizer_timer.start()

    def stop_randomizer(self) -> None:
        self._randomizer_timer.stop()
        if self._randomizer_active:
            self._randomizer_active = False
            self.randomizer_button.setText("Start Randomizer")

    def step_randomizer(self) -> None:
        if not self._randomizer_active:
            return

        elapsed_s = time.perf_counter() - self._randomizer_t0
        coefficients: dict[str, float] = {}
        for name in GUI_PARAM_NAMES:
            lo, hi = GUI_PARAM_RANGES[name]
            amp_primary, amp_secondary = GUI_RANDOMIZER_MIX[name]
            period_primary, period_secondary = GUI_RANDOMIZER_PERIODS_S[name]
            phase_primary, phase_secondary = GUI_RANDOMIZER_PHASES[name]

            primary = math.sin((2.0 * math.pi * elapsed_s / period_primary) + phase_primary)
            secondary = math.sin((2.0 * math.pi * elapsed_s / period_secondary) + phase_secondary)
            combined = amp_primary * primary + amp_secondary * secondary
            normalized = float(np.clip(combined, -0.98, 0.98))
            center = 0.5 * (lo + hi)
            half_span = 0.5 * (hi - lo)
            coefficients[name] = center + normalized * half_span

        self.apply_coefficients(coefficients)

    def schedule_render(self) -> None:
        self._render_timer.start(1)

    def render_preview(self) -> None:
        preview_bgr = render_gui_preview(
            self.preview_cal,
            num_col_lines=self.num_col_lines,
            num_row_lines=self.num_row_lines,
        )
        preview_rgb = np.ascontiguousarray(cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB))
        height, width, channels = preview_rgb.shape
        image = QImage(
            preview_rgb.data,
            width,
            height,
            channels * width,
            QImage.Format.Format_RGB888,
        ).copy()
        self.preview_pixmap = QPixmap.fromImage(image)
        self._fit_preview()

    def _fit_preview(self) -> None:
        if self.preview_pixmap is None or self.preview_label.width() <= 1 or self.preview_label.height() <= 1:
            return
        self.preview_label.setPixmap(
            self.preview_pixmap.scaled(
                self.preview_label.contentsRect().size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._resize_timer.start(0)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.stop_randomizer()
        self._render_timer.stop()
        self._resize_timer.stop()
        super().closeEvent(event)


def run_gui() -> int:
    app = QApplication.instance()
    owns_application = app is None
    if app is None:
        app = QApplication([])
    window = DistortionGui()
    window.show()
    return app.exec() if owns_application else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Brown-Conrady distortion demo.")
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Open a live Brown-Conrady slider GUI instead of rendering demo videos.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Render a picture-based distortion demo instead of the mesh comparison videos.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    cal = default_864_cam()
    cal.setDistortion(8.0 * cal.getDistortion())
    # cal.p1 = cal.p2 = 0.0

    if args.image is not None:
        clean_image = load_demo_image(args.image, width=cal.width, height=cal.height)

        products_dir = Path(__file__).with_name("Products")
        output_path = products_dir / f"distortion_image_demo_{args.image.stem}.mp4"
        write_image_demo_video(
            output_path,
            calibration=cal,
            clean_image=clean_image,
        )
        print(f"Wrote image demo video: {output_path}")
        return

    NUM_COL_LINES = 41
    NUM_ROW_LINES = 41

    img = np.zeros((cal.height, cal.width, 3), dtype=np.uint8)
    clean_mesh = build_mesh_grid(
        width=cal.width,
        height=cal.height,
        num_col_lines=NUM_COL_LINES,
        num_row_lines=NUM_ROW_LINES,
        num_line_samples=300,
    )
    clean_points = build_mesh_intersections(
        width=cal.width,
        height=cal.height,
        num_col_lines=NUM_COL_LINES,
        num_row_lines=NUM_ROW_LINES,
    )
    distorted_mesh = distort_mesh_grid(cal, clean_mesh)
    distorted_points = distort_mesh_points(cal, clean_points)
    undistorted_fp5 = undistort_mesh_grid(cal, distorted_mesh, num_fp_steps=5, num_newton_steps=0)
    undistorted_fp5_points = undistort_mesh_points(cal, distorted_points, num_fp_steps=5, num_newton_steps=0)
    undistorted_fp11 = undistort_mesh_grid(cal, distorted_mesh, num_fp_steps=11, num_newton_steps=0)
    undistorted_fp11_points = undistort_mesh_points(cal, distorted_points, num_fp_steps=11, num_newton_steps=0)
    undistorted_opencv = undistort_mesh_grid_opencv(cal, distorted_mesh)
    undistorted_opencv_points = undistort_mesh_points_opencv(cal, distorted_points)
    undistorted_fp2 = undistort_mesh_grid(cal, distorted_mesh, num_fp_steps=2, num_newton_steps=0)
    undistorted_fp2_points = undistort_mesh_points(cal, distorted_points, num_fp_steps=2, num_newton_steps=0)
    undistorted_hybrid = undistort_mesh_grid(cal, distorted_mesh, num_fp_steps=2, num_newton_steps=1)
    undistorted_hybrid_points = undistort_mesh_points(cal, distorted_points, num_fp_steps=2, num_newton_steps=1)
    undistorted_hybrid3 = undistort_mesh_grid(cal, distorted_mesh, num_fp_steps=2, num_newton_steps=3)
    undistorted_hybrid3_points = undistort_mesh_points(cal, distorted_points, num_fp_steps=2, num_newton_steps=3)

    products_dir = Path(__file__).with_name("Products")
    output_path_opencv = products_dir / "distortion_compare_fp5_vs_opencv.mp4"
    write_demo_video(
        output_path_opencv,
        base_img=img,
        clean_mesh=clean_mesh,
        distorted_mesh=distorted_mesh,
        clean_points=clean_points,
        distorted_points=distorted_points,
        left_label="FP=5",
        right_label="OpenCV",
        phases=[
            {
                "left_start": distorted_mesh,
                "left_end": undistorted_fp5,
                "right_start": distorted_mesh,
                "right_end": undistorted_opencv,
                "left_start_points": distorted_points,
                "left_end_points": undistorted_fp5_points,
                "right_start_points": distorted_points,
                "right_end_points": undistorted_opencv_points,
                "left_reference_points": clean_points,
                "right_reference_points": clean_points,
                "duration_s": DIST_TIME,
                "overlay_text": "Undistort: FP(5) vs OpenCV",
            }
        ],
    )
    print(f"Wrote demo video: {output_path_opencv}")

    output_path_hybrid = products_dir / "distortion_compare_fp5_vs_hybrid.mp4"
    write_demo_video(
        output_path_hybrid,
        base_img=img,
        clean_mesh=clean_mesh,
        distorted_mesh=distorted_mesh,
        clean_points=clean_points,
        distorted_points=distorted_points,
        left_label="FP=5",
        right_label="FP=2, GN=1",
        phases=[
            {
                "left_start": distorted_mesh,
                "left_end": undistorted_fp2,
                "right_start": distorted_mesh,
                "right_end": undistorted_fp2,
                "left_start_points": distorted_points,
                "left_end_points": undistorted_fp2_points,
                "right_start_points": distorted_points,
                "right_end_points": undistorted_fp2_points,
                "left_reference_points": clean_points,
                "right_reference_points": clean_points,
                "duration_s": 0.5 * DIST_TIME,
                "overlay_text": "Phase 1: first two FP steps",
            },
            {
                "left_start": undistorted_fp2,
                "left_end": undistorted_fp5,
                "right_start": undistorted_fp2,
                "right_end": undistorted_hybrid,
                "left_start_points": undistorted_fp2_points,
                "left_end_points": undistorted_fp5_points,
                "right_start_points": undistorted_fp2_points,
                "right_end_points": undistorted_hybrid_points,
                "left_reference_points": clean_points,
                "right_reference_points": clean_points,
                "duration_s": 0.5 * DIST_TIME,
                "overlay_text": "Phase 2: FP(3) vs GN(1)",
            },
        ],
    )
    print(f"Wrote demo video: {output_path_hybrid}")

    output_path_hybrid = products_dir / "distortion_compare_fp11_vs_FP2GN3.mp4"
    write_demo_video(
        output_path_hybrid,
        base_img=img,
        clean_mesh=clean_mesh,
        distorted_mesh=distorted_mesh,
        clean_points=clean_points,
        distorted_points=distorted_points,
        left_label="FP=11",
        right_label="FP=2, GN=3",
        phases=[
            {
                "left_start": distorted_mesh,
                "left_end": undistorted_fp2,
                "right_start": distorted_mesh,
                "right_end": undistorted_fp2,
                "left_start_points": distorted_points,
                "left_end_points": undistorted_fp2_points,
                "right_start_points": distorted_points,
                "right_end_points": undistorted_fp2_points,
                "left_reference_points": clean_points,
                "right_reference_points": clean_points,
                "duration_s": 0.5 * DIST_TIME,
                "overlay_text": "Phase 1: first two FP steps",
            },
            {
                "left_start": undistorted_fp2,
                "left_end": undistorted_fp5,
                "right_start": undistorted_fp2,
                "right_end": undistorted_hybrid,
                "left_start_points": undistorted_fp2_points,
                "left_end_points": undistorted_fp5_points,
                "right_start_points": undistorted_fp2_points,
                "right_end_points": undistorted_hybrid_points,
                "left_reference_points": clean_points,
                "right_reference_points": clean_points,
                "duration_s": 0.5 * DIST_TIME,
                "overlay_text": "Phase 2: FP(3) vs GN(1)",
            },
            {
                "left_start": undistorted_fp5,
                "left_end": undistorted_fp11,
                "right_start": undistorted_hybrid,
                "right_end": undistorted_hybrid3,
                "left_start_points": undistorted_fp5_points,
                "left_end_points": undistorted_fp11_points,
                "right_start_points": undistorted_hybrid_points,
                "right_end_points": undistorted_hybrid3_points,
                "left_reference_points": clean_points,
                "right_reference_points": clean_points,
                "duration_s": 0.5 * DIST_TIME,
                "overlay_text": "Phase 3: FP(10) vs FB(2)GN(3)",
            },
        ],
    )
    print(f"Wrote demo video: {output_path_hybrid}")


def cli() -> None:
    args = parse_args()
    if args.gui:
        run_gui()
        return
    main(args)


if __name__ == '__main__':
    cli()