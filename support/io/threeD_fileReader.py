"""
threeD_fileReader.py

I/O, plotting, demo, and command-line helpers for the 3-D to 3-D SE(3) solver.

The math layer lives in:

    support.mathHelpers.threeD_to_threeD

This module intentionally holds the operational pieces that should not be
imported every time the solver/data-model class is imported.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

import support.mathHelpers.quaternions as q
from support.mathHelpers.threeD_to_threeD import (
    SE3,
    ThreeDToThreeD,
    ThreeD_to_ThreeD,
    as_points3,
    quat_sxyz_to_rotation_matrix,
)

FloatArray = NDArray[np.float64]
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_CSV = REPO_ROOT / "Data" / "lidar_data.csv"
DEFAULT_TARGET_CSV = REPO_ROOT / "Data" / "mocap_data.csv"


def load_points_csv(path: str | Path, *, points_are_columns: Optional[bool] = None) -> FloatArray:
    """Load 3-D points from a CSV file and return an N x 3 array."""

    points = np.loadtxt(Path(path), delimiter=",")
    return as_points3(points, points_are_columns=points_are_columns)


def resolve_input_csv(path: str | Path) -> Path:
    """Resolve an input CSV against common project locations."""

    candidate = Path(path).expanduser()
    if candidate.exists():
        return candidate

    script_relative = REPO_ROOT / candidate
    if script_relative.exists():
        return script_relative

    data_relative = REPO_ROOT / "Data" / candidate.name
    if data_relative.exists():
        return data_relative

    return candidate


def quaternion_to_array(quat) -> FloatArray:
    """Return a scalar-first quaternion-like object as a flat numpy array."""

    if hasattr(quat, "ndarray"):
        return np.asarray(quat.ndarray, dtype=float).reshape(-1)
    return np.asarray(quat, dtype=float).reshape(-1)


def _format_csv_value(value: object) -> str:
    """Format one value for human-readable CSV output."""

    if isinstance(value, (float, np.floating)):
        return f"{float(value):.12g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _write_csv_row(file_obj, *values: object) -> None:
    """Write one CSV row with a space after each comma for readability."""

    file_obj.write(", ".join(_format_csv_value(v) for v in values) + "\n")


def _write_aligned_table(file_obj, rows: list[list[object]]) -> None:
    """Write rows with padded columns so values line up vertically."""

    if not rows:
        file_obj.write("\n")
        return

    formatted_rows = [[_format_csv_value(value) for value in row] for row in rows]
    num_cols = max(len(row) for row in formatted_rows)
    padded_rows = [row + [""] * (num_cols - len(row)) for row in formatted_rows]
    widths = [
        max(len(row[col_idx]) for row in padded_rows)
        for col_idx in range(num_cols)
    ]

    for row in padded_rows:
        file_obj.write(", ".join(value.rjust(width) for value, width in zip(row, widths)) + "\n")


def _write_labeled_matrix(file_obj, title: str, row_labels: list[str], matrix: ArrayLike) -> None:
    """Write a small labeled matrix as CSV."""

    arr = np.asarray(matrix, dtype=float)
    _write_csv_row(file_obj, f"# {title}")
    rows: list[list[object]] = [["label", *[f"c{n}" for n in range(arr.shape[1])]]]
    for label, row in zip(row_labels, arr):
        rows.append([label, *row])
    _write_aligned_table(file_obj, rows)
    _write_csv_row(file_obj)


def _write_points_transposed(file_obj, title: str, points: ArrayLike) -> None:
    """Write N x 3 points as transposed x/y/z rows for spreadsheet inspection."""

    pts = as_points3(points)
    _write_csv_row(file_obj, f"# {title}")
    rows: list[list[object]] = [["axis", *[f"p{n}" for n in range(pts.shape[0])]]]
    for axis_name, axis_values in zip(("x", "y", "z"), pts.T):
        rows.append([axis_name, *axis_values])
    _write_aligned_table(file_obj, rows)
    _write_csv_row(file_obj)


def save_solution_csv(fit: ThreeDToThreeD, path: str | Path = "solution.csv") -> None:
    """Save an estimated transform and residual diagnostics to a text CSV file."""

    residuals = fit.create_y(flatten=False)
    source = as_points3(fit.source_points)
    target = as_points3(fit.target_points)
    transformed = fit.transform_points(fit.source_points)
    inverse = fit.transform.inverse()
    target_in_source = inverse.transform_points(fit.target_points)
    point_headers = [f"p{n}" for n in range(source.shape[0])]

    with Path(path).open("w", encoding="utf-8") as f:
        _write_csv_row(f, "# Transform convention: target ~= R @ source + t")
        _write_csv_row(f)

        _write_csv_row(f, "# Summary")
        _write_aligned_table(
            f,
            [
                ["metric", "value"],
                ["rmse", fit.diagnostics.rmse],
                ["weighted_rmse", fit.diagnostics.weighted_rmse],
                ["max_error", fit.diagnostics.max_error],
            ],
        )
        _write_csv_row(f)

        _write_labeled_matrix(f, "Rotation R", ["r0", "r1", "r2"], fit.R)
        _write_labeled_matrix(f, "Homogeneous SE3 matrix", ["r0", "r1", "r2", "r3"], fit.transform.matrix)

        _write_csv_row(f, "# Translation t")
        _write_aligned_table(f, [["component", "x", "y", "z"], ["t", *fit.t]])
        _write_csv_row(f)

        quat = quaternion_to_array(fit.q_sxyz)
        _write_csv_row(f, "# Quaternion [s, x, y, z]")
        _write_aligned_table(f, [["component", "s", "x", "y", "z"], ["q", *quat]])
        _write_csv_row(f)

        _write_points_transposed(f, "Source points in source frame", source)
        _write_points_transposed(f, "Target points in target frame", target)
        _write_points_transposed(f, "Estimated source points transformed into target frame", transformed)
        _write_points_transposed(f, "Estimated target points transformed into source frame", target_in_source)
        _write_points_transposed(f, "Residual vectors: target - (R @ source + t)", residuals)

        _write_csv_row(f, "# Target-frame comparison: measured target vs estimated transformed source")
        comparison_rows: list[list[object]] = [[
            "field",
            *sum(([f"{p}_measured", f"{p}_estimated", f"{p}_delta"] for p in point_headers), []),
        ]]
        for axis_name, target_axis, transformed_axis, residual_axis in zip(("x", "y", "z"), target.T, transformed.T, residuals.T):
            row: list[object] = [axis_name]
            for measured, estimated, delta in zip(target_axis, transformed_axis, residual_axis):
                row.extend((measured, estimated, delta))
            comparison_rows.append(row)
        comparison_rows.append(["norm", *sum((["", norm, ""] for norm in fit.diagnostics.residual_norms), [])])
        _write_aligned_table(f, comparison_rows)
        _write_csv_row(f)

        source_recovery_error = source - target_in_source
        _write_csv_row(f, "# Source-frame comparison: measured source vs estimated target mapped into source")
        recovery_rows: list[list[object]] = [[
            "field",
            *sum(([f"{p}_measured", f"{p}_estimated", f"{p}_delta"] for p in point_headers), []),
        ]]
        for axis_name, source_axis, estimated_axis, delta_axis in zip(
            ("x", "y", "z"),
            source.T,
            target_in_source.T,
            source_recovery_error.T,
        ):
            row = [axis_name]
            for measured, estimated, delta in zip(source_axis, estimated_axis, delta_axis):
                row.extend((measured, estimated, delta))
            recovery_rows.append(row)
        _write_aligned_table(f, recovery_rows)
        _write_csv_row(f)


def fit_from_csv(
    source_csv: str | Path,
    target_csv: str | Path,
    *,
    points_are_columns: Optional[bool] = None,
    source_frame: Optional[str] = None,
    target_frame: Optional[str] = None,
) -> ThreeDToThreeD:
    """Load two point CSV files and estimate the source-to-target SE(3)."""

    source_path = Path(source_csv)
    target_path = Path(target_csv)
    source = load_points_csv(source_path, points_are_columns=points_are_columns)
    target = load_points_csv(target_path, points_are_columns=points_are_columns)

    return ThreeDToThreeD(
        source,
        target,
        source_frame=source_frame or source_path.stem,
        target_frame=target_frame or target_path.stem,
    )


def random_rotation(rng: np.random.Generator) -> FloatArray:
    """Generate a random SO(3) rotation matrix for tests/demo."""

    quat = q.randomQuat()
    return quat_sxyz_to_rotation_matrix(quat)


def quat_slerp(q0: q.Quaternion, q1: q.Quaternion, alpha: float):
    """Spherical linear interpolation between scalar-first quaternions."""

    return q0.slerp_with(q1, alpha)


def set_axes_equal_3d(ax, points: ArrayLike | list[ArrayLike], *, pad_fraction: float = 0.0) -> None:
    """Make a 3D axis use equal scaling in x, y, z for one or more point clouds."""

    if isinstance(points, list):
        pts = np.vstack([np.asarray(cloud, dtype=float) for cloud in points])
    else:
        pts = np.asarray(points, dtype=float)
    mins = np.min(pts, axis=0)
    maxs = np.max(pts, axis=0)
    center = 0.5 * (mins + maxs)
    span = np.max(maxs - mins)
    radius = max(0.5 * span * (1.0 + pad_fraction), 1.0e-6)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def visualize_alignment(
    source_points: ArrayLike,
    target_points: ArrayLike,
    transform: SE3,
    *,
    source_label: str = "source",
    target_label: str = "target",
    frames: int = 150,
    interval_ms: int = 16,
    hold_frames: int = 100,
    ghost_stride: int = 8,
    max_ghosts: int = 12,
    save_gif_path: Optional[str | Path] = None,
) -> None:
    """
    Show:
      - left: animation of source points moving under the estimated transform,
              with ghost clouds and live correspondence lines
      - right: final aligned 3D result

    If save_gif_path is provided, also save the animation as a GIF.
    """

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    source = as_points3(source_points)
    target = as_points3(target_points)
    source_final = transform.transform_points(source)

    source_marker = "o"
    target_marker = "^"
    aligned_marker = "s"
    source_color = "tab:blue"
    target_color = "tab:orange"
    aligned_color = "tab:green"
    correspondence_color = "0.45"

    # Interpolate from identity transform to the estimated transform.
    q_identity = q.identity()
    q_final = transform.q_sxyz
    t_final = transform.t

    alphas = np.linspace(0.0, 1.0, frames)
    animated_clouds = []

    for alpha in alphas:
        q_alpha = quat_slerp(q_identity, q_final, float(alpha))
        R_alpha = quat_sxyz_to_rotation_matrix(q_alpha)
        t_alpha = alpha * t_final
        pts_alpha = source @ R_alpha.T + t_alpha
        animated_clouds.append(pts_alpha)

    total_anim_frames = frames + hold_frames

    fig = plt.figure(figsize=(14, 6))
    ax_anim = fig.add_subplot(1, 2, 1, projection="3d")
    ax_final = fig.add_subplot(1, 2, 2, projection="3d")

    ax_anim.scatter(
        source[:, 0], source[:, 1], source[:, 2],
        label=f"{source_label} (start)",
        alpha=0.22,
        marker=source_marker,
        color=source_color,
    )

    ax_anim.scatter(
        target[:, 0], target[:, 1], target[:, 2],
        label=f"{target_label}",
        marker=target_marker,
        color=target_color,
    )

    moving = ax_anim.scatter(
        animated_clouds[0][:, 0],
        animated_clouds[0][:, 1],
        animated_clouds[0][:, 2],
        label=f"{source_label} transformed",
        marker=aligned_marker,
        color=aligned_color,
    )

    ghost_artists = []
    for _ in range(max_ghosts):
        ghost = ax_anim.scatter([], [], [], alpha=0.08, marker=aligned_marker, color=aligned_color)
        ghost_artists.append(ghost)

    centroid_line, = ax_anim.plot([], [], [], linewidth=1.5, alpha=0.5, color=aligned_color)
    centroids = np.array([pts.mean(axis=0) for pts in animated_clouds])

    anim_corr_lines = []
    for _ in range(source.shape[0]):
        line, = ax_anim.plot([], [], [], alpha=0.20, linewidth=1.0, color=correspondence_color)
        anim_corr_lines.append(line)

    ax_anim.set_title("Estimated transform animation")
    ax_anim.set_xlabel("X")
    ax_anim.set_ylabel("Y")
    ax_anim.set_zlabel("Z")
    ax_anim.legend(loc="upper left")
    set_axes_equal_3d(ax_anim, [source, target, animated_clouds[0]], pad_fraction=0.20)

    ax_final.scatter(
        target[:, 0], target[:, 1], target[:, 2],
        label=f"{target_label}",
        marker=target_marker,
        color=target_color,
    )
    ax_final.scatter(
        source_final[:, 0], source_final[:, 1], source_final[:, 2],
        label=f"{source_label} aligned",
        marker=aligned_marker,
        color=aligned_color,
    )

    for p_src, p_tgt in zip(source_final, target):
        ax_final.plot(
            [p_src[0], p_tgt[0]],
            [p_src[1], p_tgt[1]],
            [p_src[2], p_tgt[2]],
            alpha=0.35,
            color=correspondence_color,
        )

    rmse = np.sqrt(np.mean(np.sum((target - source_final) ** 2, axis=1)))

    ax_final.set_title(f"Final alignment\nRMSE = {rmse:.6f}")
    ax_final.set_xlabel("X")
    ax_final.set_ylabel("Y")
    ax_final.set_zlabel("Z")
    ax_final.legend(loc="upper left")
    set_axes_equal_3d(ax_final, [target, source_final], pad_fraction=0.10)

    def update(frame_idx: int):
        shown_idx = min(frame_idx, frames - 1)
        pts = animated_clouds[shown_idx]

        moving._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])

        ghost_indices = list(range(shown_idx - ghost_stride, -1, -ghost_stride))
        ghost_indices = ghost_indices[:max_ghosts]

        for i, ghost in enumerate(ghost_artists):
            if i < len(ghost_indices):
                gpts = animated_clouds[ghost_indices[i]]
                ghost._offsets3d = (gpts[:, 0], gpts[:, 1], gpts[:, 2])

                strength = 1.0 - (i / max(max_ghosts - 1, 1))
                ghost.set_alpha(0.05 + 0.20 * strength)
            else:
                ghost._offsets3d = ([], [], [])
                ghost.set_alpha(0.0)

        trail = centroids[:shown_idx + 1]
        centroid_line.set_data(trail[:, 0], trail[:, 1])
        centroid_line.set_3d_properties(trail[:, 2])

        for line, p_src, p_tgt in zip(anim_corr_lines, pts, target):
            line.set_data([p_src[0], p_tgt[0]], [p_src[1], p_tgt[1]])
            line.set_3d_properties([p_src[2], p_tgt[2]])

        clouds_for_scale = [target, pts]
        if shown_idx < max(5, frames // 8):
            clouds_for_scale.append(source)
        set_axes_equal_3d(ax_anim, clouds_for_scale, pad_fraction=0.20)

        if frame_idx < frames:
            ax_anim.set_title(
                f"Estimated transform animation (alpha = {alphas[shown_idx]:.2f})"
            )
        else:
            ax_anim.set_title("Estimated transform animation (final alignment hold)")

        return (moving, centroid_line, *ghost_artists, *anim_corr_lines)

    ani = FuncAnimation(
        fig,
        update,
        frames=total_anim_frames,
        interval=interval_ms,
        blit=False,
        repeat=True,
    )

    fig._ani = ani
    plt.tight_layout()

    if save_gif_path is not None:
        save_gif_path = Path(save_gif_path)
        fps = min(60, max(1, int(round(1000 / interval_ms))))
        print(f"Saving GIF to: {save_gif_path}")
        ani.save(save_gif_path, writer=PillowWriter(fps=fps))
        print("GIF save complete.")

    plt.show()


def print_3dPts(threeD_proj: ArrayLike) -> None:
    """Legacy helper: print 3-D points in a readable form."""

    pts = as_points3(threeD_proj)
    print(f"Norm: {np.linalg.norm(pts)}")
    for n, point in enumerate(pts):
        print(f"Feature: {n:3d}, x: {point[0]: .5f}, y: {point[1]: .5f}, z: {point[2]: .5f}")


def run_demo(*, show_plot: bool = True, save_gif_path: Optional[str | Path] = "alignment_demo.gif") -> int:
    """Small deterministic smoke test with tunable measurement noise."""

    rng = np.random.default_rng(7)

    source_true = rng.normal(0.0, 1.0, size=(20, 3))
    R_true = random_rotation(rng)
    t_true = np.array([10.0, -2.0, 0.5])

    target_true = source_true @ R_true.T + t_true

    sigma_source = 0.02
    sigma_target = 0.02

    source_measured = source_true + rng.normal(0.0, sigma_source, size=source_true.shape)
    target_measured = target_true + rng.normal(0.0, sigma_target, size=target_true.shape)

    fit = ThreeDToThreeD(
        source_measured,
        target_measured,
        source_frame="lidar",
        target_frame="camera",
    )

    R_error = fit.R @ R_true.T
    angle_error_rad = np.arccos(
        np.clip((np.trace(R_error) - 1.0) / 2.0, -1.0, 1.0)
    )
    t_error = fit.t - t_true

    sigma_residual_axis = np.sqrt(sigma_source**2 + sigma_target**2)
    expected_residual_norm_scale = np.sqrt(3.0) * sigma_residual_axis

    print("=== 3D-to-3D SE(3) noisy demo ===")
    print(f"Number of points: {source_true.shape[0]}")
    print(f"Source noise sigma: {sigma_source}")
    print(f"Target noise sigma: {sigma_target}")
    print(f"Approx residual norm scale: {expected_residual_norm_scale:.6f}")

    print("\nTrue R:\n", R_true)
    print("Estimated R:\n", fit.R)
    print("Rotation error Frobenius norm:", np.linalg.norm(fit.R - R_true))
    print("Rotation error angle [deg]:", np.rad2deg(angle_error_rad))

    print("\nTrue t:", t_true)
    print("Estimated t:", fit.t)
    print("Translation error:", t_error)
    print("Translation error norm:", np.linalg.norm(t_error))

    print("\nQuaternion [s, x, y, z]:", fit.q_sxyz)
    print("SE3:\n", fit.to_SE3().matrix)

    print("\nResidual diagnostics:")
    print("RMSE:", fit.diagnostics.rmse)
    print("Weighted RMSE:", fit.diagnostics.weighted_rmse)
    print("Max error:", fit.diagnostics.max_error)
    print("Residual norms:", fit.diagnostics.residual_norms)

    if show_plot:
        visualize_alignment(
            source_measured,
            target_measured,
            fit.transform,
            source_label="lidar",
            target_label="camera",
            save_gif_path=save_gif_path,
        )

    return 0


def run_original_random_test() -> int:
    """Restore the original ``__main__`` synthetic test harness."""

    try:
        from support.mathHelpers.quaternions import randomQuat
    except Exception as exc:  # pragma: no cover - depends on local project package
        raise ImportError(
            "run_original_random_test requires support.mathHelpers.quaternions.randomQuat"
        ) from exc

    source = np.random.normal(0.0, 1.0, (10, 3))
    noise = np.random.normal(0.0, 0.1, source.shape)

    print(source)
    q_true = randomQuat()
    t_true = np.array([10.0, 0.0, 0.0]) + np.random.normal(1.0, 1.0, (3,))
    target = (q_true * source + t_true) + noise

    print(f"Targets: \n{q_true}\n{t_true}\n")
    try:
        print(f"Targets: \n{q_true.to_SE3_given_position(t_true)}")
    except Exception:
        pass

    fit = ThreeD_to_ThreeD(source, target)

    print(f"Estimates: \nR =\n{fit.R}\nt = {fit.t}\nq [s, x, y, z] = {fit.q_sxyz}\n")
    try:
        print(f"Estimates(SE3):\n{fit.to_project_SE3()}\n")
    except Exception:
        print(f"Estimates(SE3):\n{fit.to_SE3().matrix}\n")

    resolved_residual = fit.create_y()
    true_residual = (target - (q_true * source + t_true)).reshape(-1)
    print(f"Resolved Residual: {np.linalg.norm(resolved_residual)}\n{resolved_residual}")
    print(f"True Residual: {np.linalg.norm(true_residual)}\n{true_residual}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Estimate the SE(3) transform mapping source 3-D points into target 3-D points."
    )
    parser.add_argument(
        "source_csv",
        nargs="?",
        default=str(DEFAULT_SOURCE_CSV),
        help="CSV of source-frame points to transform. Default: <repo>/Data/lidar_data.csv",
    )
    parser.add_argument(
        "target_csv",
        nargs="?",
        default=str(DEFAULT_TARGET_CSV),
        help="CSV of corresponding target-frame points. Default: <repo>/Data/mocap_data.csv",
    )
    parser.add_argument(
        "--output",
        default="solution.csv",
        help="Output CSV path for transform and residual diagnostics.",
    )
    parser.add_argument(
        "--points-are-columns",
        action="store_true",
        help="Force legacy 3 x N input interpretation instead of automatic N x 3 / 3 x N detection.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a deterministic synthetic test instead of reading CSV files.",
    )
    parser.add_argument(
        "--original-test",
        action="store_true",
        help="Run the original random quaternion test harness from the old __main__ block.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not display or save the visualization.",
    )
    parser.add_argument(
        "--gif",
        default="alignment_from_csv.gif",
        help="GIF path for CSV visualization. Use an empty string to skip saving.",
    )
    args = parser.parse_args(argv)

    if args.original_test:
        return run_original_random_test()

    if args.demo:
        return run_demo(show_plot=not args.no_plot)

    source_path = resolve_input_csv(args.source_csv)
    target_path = resolve_input_csv(args.target_csv)
    if not source_path.exists() or not target_path.exists():
        print(
            f"Input CSV files were not found.\n"
            f"  source: {source_path}\n"
            f"  target: {target_path}\n"
            "Provide source_csv and target_csv, "
            "or run with --demo for a synthetic test."
        )
        return 2

    points_are_columns = True if args.points_are_columns else None
    fit = fit_from_csv(
        source_path,
        target_path,
        points_are_columns=points_are_columns,
    )
    save_solution_csv(fit, args.output)

    np.set_printoptions(suppress=True, threshold=np.inf, precision=5)
    print("Transform convention: target ~= R @ source + t")
    print("R:\n", fit.R)
    print("t:", fit.t)
    print("q [s, x, y, z]:", fit.q_sxyz)
    print("RMSE:      ", fit.diagnostics.rmse)
    print("Max error: ", fit.diagnostics.max_error)
    print(f"Wrote {args.output}")

    if not args.no_plot:
        save_gif_path = args.gif if args.gif else None
        visualize_alignment(
            fit.source_points,
            fit.target_points,
            fit.transform,
            source_label=source_path.stem,
            target_label=target_path.stem,
            save_gif_path=save_gif_path,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
