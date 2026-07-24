"""
threeD_fileReader.py

CSV-oriented I/O helpers for the 3-D to 3-D SE(3) solver.

This module is the operational path for loading point files, fitting the rigid
transform, and writing a human-readable solution report. Visualization, test
harnesses, and CLI entry points live in:

    support.demos.threeD_alignment_demo
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from support.mathHelpers.threeD_to_threeD import (
    ThreeDToThreeD,
    as_points3,
)

FloatArray = NDArray[np.float64]
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_CSV = REPO_ROOT / "Data" / "mia_offset.csv"
DEFAULT_TARGET_CSV = REPO_ROOT / "Data" / "mia_mocap.csv"


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
