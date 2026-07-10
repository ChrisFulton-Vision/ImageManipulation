import argparse
from support.vision.calibration import Calibration, default_2848_cam, distort_points_px, undistort_points_px
import numpy as np
import cv2
from functools import partial
import time
from pathlib import Path

import customtkinter as ctk
from PIL import Image

DIST_TIME = 5
FPS = 30
PAUSE_TIME = 1.0

GUI_PARAM_NAMES = ("k1", "k2", "p1", "p2", "k3")
GUI_PARAM_RANGES = {
    "k1": (-2.0, 2.0),
    "k2": (-2.0, 2.0),
    "p1": (-0.05, 0.05),
    "p2": (-0.05, 0.05),
    "k3": (-2.0, 2.0),
}


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
    cal = default_2848_cam().copy()
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


def render_gui_preview(
        calibration: Calibration,
        *,
        num_col_lines: int = 21,
        num_row_lines: int = 21,
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


class DistortionGui(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Brown-Conrady Distortion Viewer")
        self.geometry("1320x980")
        self.minsize(1180, 900)

        self.preview_cal = build_preview_calibration()
        self.slider_vars: dict[str, ctk.DoubleVar] = {}
        self.preview_image = None
        self._render_job = None

        ctk.set_appearance_mode("dark")
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        controls = ctk.CTkFrame(self, corner_radius=8)
        controls.grid(row=0, column=0, padx=16, pady=16, sticky="ns")
        controls.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(controls, text="Distortion Parameters", font=("Segoe UI", 20, "bold")).grid(
            row=0, column=0, padx=16, pady=(16, 12), sticky="w"
        )

        for row_index, name in enumerate(GUI_PARAM_NAMES, start=1):
            self._build_slider_row(controls, row=row_index, name=name)

        ctk.CTkButton(controls, text="Reset Defaults", command=self.reset_defaults).grid(
            row=len(GUI_PARAM_NAMES) + 1, column=0, padx=16, pady=(16, 10), sticky="ew"
        )
        ctk.CTkButton(controls, text="Set Zeros", command=self.set_zeros).grid(
            row=len(GUI_PARAM_NAMES) + 2, column=0, padx=16, pady=(16, 10), sticky="ew"
        )

        ctk.CTkLabel(
            controls,
            text="The preview uses the forward Brown-Conrady model with a live distorted mesh.",
            justify="left",
            wraplength=260,
        ).grid(row=len(GUI_PARAM_NAMES) + 3, column=0, padx=16, pady=(0, 16), sticky="w")

        preview_frame = ctk.CTkFrame(self, corner_radius=8)
        preview_frame.grid(row=0, column=1, padx=(0, 16), pady=16, sticky="nsew")
        preview_frame.grid_rowconfigure(0, weight=1)
        preview_frame.grid_columnconfigure(0, weight=1)

        self.preview_label = ctk.CTkLabel(preview_frame, text="")
        self.preview_label.grid(row=0, column=0, padx=12, pady=12, sticky="nsew")

        self.reset_defaults()

    def _build_slider_row(self, parent, *, row: int, name: str) -> None:
        current = float(getattr(self.preview_cal, name))
        var = ctk.DoubleVar(value=current)
        self.slider_vars[name] = var

        row_frame = ctk.CTkFrame(parent, fg_color="transparent")
        row_frame.grid(row=row, column=0, padx=16, pady=8, sticky="ew")
        row_frame.grid_columnconfigure(0, weight=1)

        label_var = ctk.StringVar(value=self._format_label(name, current))
        setattr(self, f"{name}_label_var", label_var)

        ctk.CTkLabel(row_frame, textvariable=label_var, anchor="w").grid(row=0, column=0, sticky="ew")
        ctk.CTkSlider(
            row_frame,
            from_=GUI_PARAM_RANGES[name][0],
            to=GUI_PARAM_RANGES[name][1],
            variable=var,
            command=lambda _value, param_name=name: self.on_slider_change(param_name),
        ).grid(row=1, column=0, pady=(6, 0), sticky="ew")

    @staticmethod
    def _format_label(name: str, value: float) -> str:
        return f"{name}: {value:+.6f}"

    def on_slider_change(self, name: str) -> None:
        value = float(self.slider_vars[name].get())
        getattr(self, f"{name}_label_var").set(self._format_label(name, value))
        setattr(self.preview_cal, name, value)
        self.schedule_render()

    def reset_defaults(self) -> None:
        default_cal = build_preview_calibration()
        self.preview_cal = default_cal
        for name in GUI_PARAM_NAMES:
            value = float(getattr(default_cal, name))
            self.slider_vars[name].set(value)
            getattr(self, f"{name}_label_var").set(self._format_label(name, value))
        self.schedule_render()

    def set_zeros(self):
        default_cal = build_preview_calibration()
        default_cal.setDistortion(default_cal.getDistortion() * 0.0)
        self.preview_cal = default_cal
        for name in GUI_PARAM_NAMES:
            value = float(getattr(default_cal, name))
            self.slider_vars[name].set(value)
            getattr(self, f"{name}_label_var").set(self._format_label(name, value))
        self.schedule_render()

    def schedule_render(self) -> None:
        if self._render_job is not None:
            self.after_cancel(self._render_job)
        self._render_job = self.after(1, self.render_preview)

    def render_preview(self) -> None:
        self._render_job = None
        preview_bgr = render_gui_preview(self.preview_cal)
        preview_rgb = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(preview_rgb)
        self.preview_image = ctk.CTkImage(light_image=image, dark_image=image, size=image.size)
        self.preview_label.configure(image=self.preview_image)


def run_gui() -> None:
    app = DistortionGui()
    app.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Brown-Conrady distortion demo.")
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Open a live Brown-Conrady slider GUI instead of rendering demo videos.",
    )
    return parser.parse_args()


def main():
    cal = default_2848_cam()
    cal.setDistortion(8.0 * cal.getDistortion())
    cal.p1 = cal.p2 = 0.0
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
    undistorted_fp10 = undistort_mesh_grid(cal, distorted_mesh, num_fp_steps=10, num_newton_steps=0)
    undistorted_fp10_points = undistort_mesh_points(cal, distorted_points, num_fp_steps=10, num_newton_steps=0)
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

    output_path_hybrid = products_dir / "distortion_compare_fp10_vs_FP2GN3.mp4"
    write_demo_video(
        output_path_hybrid,
        base_img=img,
        clean_mesh=clean_mesh,
        distorted_mesh=distorted_mesh,
        clean_points=clean_points,
        distorted_points=distorted_points,
        left_label="FP=10",
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
                "left_end": undistorted_fp10,
                "right_start": undistorted_hybrid,
                "right_end": undistorted_hybrid3,
                "left_start_points": undistorted_fp5_points,
                "left_end_points": undistorted_fp10_points,
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
    if args.gui or True:
        run_gui()
        return
    main()


if __name__ == '__main__':
    cli()
