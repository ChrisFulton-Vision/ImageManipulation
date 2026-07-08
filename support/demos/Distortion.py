from support.vision.calibration import Calibration, default_864_cam, distort_points_px, undistort_points_px
import numpy as np
import cv2
from functools import partial
import time
from pathlib import Path

DIST_TIME = 5
FPS = 30
PAUSE_TIME = 1.0


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

    x_positions = np.linspace(0.0, width - 1.0, num_col_lines, dtype=np.float64)
    y_positions = np.linspace(0.0, height - 1.0, num_row_lines, dtype=np.float64)
    sample_x = np.linspace(-margin_x, (width - 1.0) + margin_x, num_line_samples, dtype=np.float64)
    sample_y = np.linspace(-margin_y, (height - 1.0) + margin_y, num_line_samples, dtype=np.float64)

    mesh_lines: list[np.ndarray] = []
    for x in x_positions:
        mesh_lines.append(np.column_stack((np.full_like(sample_y, x), sample_y)))
    for y in y_positions:
        mesh_lines.append(np.column_stack((sample_x, np.full_like(sample_x, y))))
    return mesh_lines


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


def hsv_to_bgr(h: float, s: int = 255, v: int = 255) -> tuple[int, int, int]:
    hsv = np.array([[[h % 180.0, s, v]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_hue_mesh(
        image: np.ndarray,
        *,
        clean_lines: list[np.ndarray],
        mesh_lines: list[np.ndarray],
        line_thickness: int = 1,
        interpolation: float = 1.0,
) -> np.ndarray:
    mesh = image.copy()
    used_lines = interpolate_mesh_grid(clean_lines, mesh_lines, interpolation)

    total_lines = max(1, len(used_lines) - 1)
    hue_step = 179.0 / total_lines

    for line_index, points in enumerate(used_lines):
        color = hsv_to_bgr(line_index * hue_step)
        points_i32 = np.round(points).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(mesh, [points_i32], isClosed=False, color=color, thickness=line_thickness, lineType=cv2.LINE_AA)

    return mesh


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
        interpolation: float,
        left_label: str,
        right_label: str,
) -> np.ndarray:
    left_img = add_panel_label(
        draw_hue_mesh(base_img, clean_lines=left_start, mesh_lines=left_end, interpolation=interpolation),
        left_label,
    )
    right_img = add_panel_label(
        draw_hue_mesh(base_img, clean_lines=right_start, mesh_lines=right_end, interpolation=interpolation),
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
            interpolation: float,
            overlay_text: str,
    ) -> np.ndarray:
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
        draw_text = f"{overlay_text}: {interpolation * 100.0:.2f}%"
        add_text(img=frame, text=draw_text)
        return frame

    def write_phase(
            *,
            left_start: list[np.ndarray],
            left_end: list[np.ndarray],
            right_start: list[np.ndarray],
            right_end: list[np.ndarray],
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
            duration_s=pause_time_s,
            overlay_text="Undistort Compare",
        )
        for phase in phases:
            write_phase(
                left_start=phase["left_start"],
                left_end=phase["left_end"],
                right_start=phase["right_start"],
                right_end=phase["right_end"],
                duration_s=phase["duration_s"],
                overlay_text=phase["overlay_text"],
            )
        final_phase = phases[-1]
        write_phase(
            left_start=final_phase["left_end"],
            left_end=final_phase["left_end"],
            right_start=final_phase["right_end"],
            right_end=final_phase["right_end"],
            duration_s=pause_time_s,
            overlay_text="Final Undistorted",
        )
    finally:
        writer.release()

    return output_path


def main():
    cal = default_864_cam()
    # cal.setDistortion(1.0 * cal.getDistortion())
    # cal.p1 = cal.p2 = 0.0
    NUM_COL_LINES = 21
    NUM_ROW_LINES = 21

    img = np.zeros((cal.height, cal.width, 3), dtype=np.uint8)
    clean_mesh = build_mesh_grid(
        width=cal.width,
        height=cal.height,
        num_col_lines=NUM_COL_LINES,
        num_row_lines=NUM_ROW_LINES,
        num_line_samples=300,
    )
    distorted_mesh = distort_mesh_grid(cal, clean_mesh)
    undistorted_fp5 = undistort_mesh_grid(cal, distorted_mesh, num_fp_steps=5, num_newton_steps=0)
    undistorted_fp10 = undistort_mesh_grid(cal, distorted_mesh, num_fp_steps=10, num_newton_steps=0)
    undistorted_opencv = undistort_mesh_grid_opencv(cal, distorted_mesh)
    undistorted_fp2 = undistort_mesh_grid(cal, distorted_mesh, num_fp_steps=2, num_newton_steps=0)
    undistorted_hybrid = undistort_mesh_grid(cal, distorted_mesh, num_fp_steps=2, num_newton_steps=1)
    undistorted_hybrid3 = undistort_mesh_grid(cal, distorted_mesh, num_fp_steps=2, num_newton_steps=3)

    products_dir = Path(__file__).with_name("Products")
    output_path_opencv = products_dir / "distortion_compare_fp5_vs_opencv.mp4"
    write_demo_video(
        output_path_opencv,
        base_img=img,
        clean_mesh=clean_mesh,
        distorted_mesh=distorted_mesh,
        left_label="FP=5",
        right_label="OpenCV",
        phases=[
            {
                "left_start": distorted_mesh,
                "left_end": undistorted_fp5,
                "right_start": distorted_mesh,
                "right_end": undistorted_opencv,
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
        left_label="FP=5",
        right_label="FP=2, GN=1",
        phases=[
            {
                "left_start": distorted_mesh,
                "left_end": undistorted_fp2,
                "right_start": distorted_mesh,
                "right_end": undistorted_fp2,
                "duration_s": 0.5 * DIST_TIME,
                "overlay_text": "Phase 1: first two FP steps",
            },
            {
                "left_start": undistorted_fp2,
                "left_end": undistorted_fp5,
                "right_start": undistorted_fp2,
                "right_end": undistorted_hybrid,
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
        left_label="FP=10",
        right_label="FP=2, GN=3",
        phases=[
            {
                "left_start": distorted_mesh,
                "left_end": undistorted_fp2,
                "right_start": distorted_mesh,
                "right_end": undistorted_fp2,
                "duration_s": 0.5 * DIST_TIME,
                "overlay_text": "Phase 1: first two FP steps",
            },
            {
                "left_start": undistorted_fp2,
                "left_end": undistorted_fp5,
                "right_start": undistorted_fp2,
                "right_end": undistorted_hybrid,
                "duration_s": 0.5 * DIST_TIME,
                "overlay_text": "Phase 2: FP(3) vs GN(1)",
            },
            {
                "left_start": undistorted_fp5,
                "left_end": undistorted_fp10,
                "right_start": undistorted_hybrid,
                "right_end": undistorted_hybrid3,
                "duration_s": 0.5 * DIST_TIME,
                "overlay_text": "Phase 3: FP(10) vs FB(2)GN(3)",
            },
        ],
    )
    print(f"Wrote demo video: {output_path_hybrid}")

if __name__ == '__main__':
    main()
