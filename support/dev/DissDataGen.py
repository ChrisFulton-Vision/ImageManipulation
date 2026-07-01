"""
Analyze dissertation navigation comparison CSVs.

This script compares multi-feature pose-estimation methods and, optionally, an
additional single-feature CSV. It produces summary CSVs and dissertation-friendly plots
for:

  * method availability
  * frame-to-frame translation and attitude smoothness
  * distance tracking across methods, including single-feature estimates when provided
  * attitude tracking across quaternion pose methods
  * reprojection residuals, when present
  * feature-health / detector operating-envelope diagnostics, when present
  * RANSAC outlier vs. KF innovation-rejection behavior, when present
  * frame sequencing diagnostics from image names and timestamps

Typical CLI usage:
    python DissDataGen.py navcalcs_GIII_Pseudo.csv --out results_giii

With a single-feature CSV:
    python DissDataGen.py navcalcs_GIII_Pseudo.csv \
        --single-csv single_feature_nav.csv \
        --single-label "Single-feature YOLO" \
        --out results_giii_with_single

Optional artifact exclusions for temporal smoothness:
    python DissDataGen.py navcalcs_Aligned_Drogue_65_BothContext.csv \
        --out results_drogue \
        --exclude-window 27738.bmp 10 \
        --exclude-window 28166.bmp 3

GUI usage:
    Run without positional arguments. The script opens a file chooser for the primary
    navcalcs CSV and then an optional file chooser for a single-feature CSV.

Notes:
    - Smoothness metrics are sensitive to frame order. By default, non-unit image-number
      jumps and timestamp outliers are excluded from smoothness summaries.
    - Reprojection residuals for wQnP using KF centers may be evaluated against raw YOLO
      centers depending on how the CSV was generated. Treat that residual column with care.
    - Single-feature CSVs are position-only unless quaternion columns are supplied. They
      appear in distance/translation plots but are omitted from attitude plots when no
      quaternion is available.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Method specifications
# -----------------------------

@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    valid_col: str | None
    t_cols: tuple[str, str, str]
    q_cols: tuple[str, str, str, str] | None = None
    rmse_col: str | None = None
    source: str = "primary"

    @property
    def has_attitude(self) -> bool:
        return self.q_cols is not None


PRIMARY_METHODS: tuple[MethodSpec, ...] = (
    MethodSpec(
        key="pnp",
        label="OpenCV SolvePnP",
        valid_col="pnp_valid",
        q_cols=("pnp_qw", "pnp_qx", "pnp_qy", "pnp_qz"),
        t_cols=("pnp_tvec_x", "pnp_tvec_y", "pnp_tvec_z"),
        rmse_col="pnp_rmse_px",
    ),
    MethodSpec(
        key="qnp",
        label="SolveQnP",
        valid_col="qnp_valid",
        q_cols=("qnp_qw", "qnp_qx", "qnp_qy", "qnp_qz"),
        t_cols=("qnp_tvec_x", "qnp_tvec_y", "qnp_tvec_z"),
        rmse_col="qnp_rmse_px",
    ),
    MethodSpec(
        key="wqnp_yolo",
        label="wQnP, YOLO centers + KF covariance",
        valid_col="wqnp_yolo_valid",
        q_cols=("wqnp_yolo_qw", "wqnp_yolo_qx", "wqnp_yolo_qy", "wqnp_yolo_qz"),
        t_cols=("wqnp_yolo_tvec_x", "wqnp_yolo_tvec_y", "wqnp_yolo_tvec_z"),
        rmse_col="wqnp_yolo_rmse_px",
    ),
    MethodSpec(
        key="wqnp_kfest",
        label="wQnP, KF centers + KF covariance",
        valid_col="wqnp_kfest_valid",
        q_cols=("wqnp_kfest_qw", "wqnp_kfest_qx", "wqnp_kfest_qy", "wqnp_kfest_qz"),
        t_cols=("wqnp_kfest_tvec_x", "wqnp_kfest_tvec_y", "wqnp_kfest_tvec_z"),
        rmse_col="wqnp_kfest_rmse_px",
    ),
)

SINGLE_POSITION_CANDIDATES: tuple[tuple[str, tuple[str, str, str], str | None], ...] = (
    # Current solo-network navcalcs format.
    ("single_feature", ("single_feature_estimated_tvec_x", "single_feature_estimated_tvec_y", "single_feature_estimated_tvec_z"), None),
    # Older / generic single-feature formats.
    ("single_feature", ("single_feature_tvec_x", "single_feature_tvec_y", "single_feature_tvec_z"), "single_feature_valid"),
    ("single", ("single_tvec_x", "single_tvec_y", "single_tvec_z"), "single_valid"),
    ("sf", ("sf_tvec_x", "sf_tvec_y", "sf_tvec_z"), "sf_valid"),
    ("single_feature", ("single_feature_x", "single_feature_y", "single_feature_z"), "single_feature_valid"),
    ("single", ("single_x", "single_y", "single_z"), "single_valid"),
    ("sf", ("sf_x", "sf_y", "sf_z"), "sf_valid"),
    ("los_range", ("los_range_x", "los_range_y", "los_range_z"), "los_range_valid"),
    ("measurement", ("measurement_x", "measurement_y", "measurement_z"), "measurement_valid"),
    ("estimate", ("estimate_x", "estimate_y", "estimate_z"), "estimate_valid"),
    ("est", ("est_x", "est_y", "est_z"), "est_valid"),
    ("relative", ("rel_x", "rel_y", "rel_z"), "valid"),
    ("position", ("pos_x", "pos_y", "pos_z"), "valid"),
    ("xyz", ("x", "y", "z"), "valid"),
)

SINGLE_ATTITUDE_CANDIDATES: tuple[tuple[str, tuple[str, str, str, str]], ...] = (
    ("single_feature", ("single_feature_qw", "single_feature_qx", "single_feature_qy", "single_feature_qz")),
    ("single", ("single_qw", "single_qx", "single_qy", "single_qz")),
    ("sf", ("sf_qw", "sf_qx", "sf_qy", "sf_qz")),
    ("estimate", ("estimate_qw", "estimate_qx", "estimate_qy", "estimate_qz")),
    ("est", ("est_qw", "est_qx", "est_qy", "est_qz")),
    ("quat", ("qw", "qx", "qy", "qz")),
)


# -----------------------------
# Basic utilities
# -----------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def available_columns(df: pd.DataFrame, cols: Iterable[str]) -> bool:
    return all(c in df.columns for c in cols)


def as_bool_series(series: pd.Series) -> pd.Series:
    """Robustly convert CSV validity columns to boolean."""
    if series.dtype == bool:
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) != 0.0

    true_values = {"true", "t", "1", "yes", "y", "valid"}
    return series.fillna("false").astype(str).str.strip().str.lower().isin(true_values)


def parse_image_number(name: object) -> float:
    """Extract the last integer from an image filename, e.g. 27738 from 27738.bmp."""
    if pd.isna(name):
        return np.nan
    matches = re.findall(r"\d+", str(name))
    return float(matches[-1]) if matches else np.nan


def parse_id_set(value: object) -> set[int]:
    """Parse class-id strings such as '1;2;3', '[1, 2, 3]', or NaN."""
    if value is None or pd.isna(value):
        return set()
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return set()
    return {int(tok) for tok in re.findall(r"[-+]?\d+", text)}


def count_id_string(value: object) -> int:
    return len(parse_id_set(value))


def numeric_matrix(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    return df.loc[:, cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def finite_stats(values: np.ndarray | pd.Series) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "p05": np.nan,
            "p95": np.nan,
            "max": np.nan,
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def write_markdown_table(df: pd.DataFrame, path: Path, float_fmt: str = ".4g") -> None:
    """Write a lightweight markdown table without requiring tabulate."""
    def fmt(x: object) -> str:
        if isinstance(x, (float, np.floating)):
            if math.isnan(float(x)):
                return ""
            return format(float(x), float_fmt)
        return str(x)

    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------
# Quaternion utilities
# -----------------------------

def quat_normalize(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    norm[norm == 0.0] = np.nan
    return q / norm


def quat_conj(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[:, 1:] *= -1.0
    return out


def quat_mul(q2: np.ndarray, q1: np.ndarray) -> np.ndarray:
    """Hamilton product, scalar-first, vectorized row-wise: q = q2 * q1."""
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    return np.column_stack(
        [
            w2 * w1 - x2 * x1 - y2 * y1 - z2 * z1,
            w2 * x1 + x2 * w1 + y2 * z1 - z2 * y1,
            w2 * y1 - x2 * z1 + y2 * w1 + z2 * x1,
            w2 * z1 + x2 * y1 - y2 * x1 + z2 * w1,
        ]
    )


def quat_delta_angle_deg(q_prev: np.ndarray, q_cur: np.ndarray) -> np.ndarray:
    """Shortest attitude change angle between row-wise scalar-first quaternions."""
    q_prev = quat_normalize(q_prev)
    q_cur = quat_normalize(q_cur)
    q_rel = quat_mul(q_cur, quat_conj(q_prev))
    q_rel = quat_normalize(q_rel)
    scalar = np.clip(np.abs(q_rel[:, 0]), 0.0, 1.0)
    angle = 2.0 * np.arccos(scalar)
    angle = np.minimum(angle, 2.0 * np.pi - angle)
    return np.rad2deg(angle)


def quat_angle_from_reference_deg(q: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Angle from a single reference quaternion to each quaternion row."""
    q = quat_normalize(q)
    ref = np.asarray(ref, dtype=float).reshape(1, 4)
    ref = quat_normalize(ref)
    ref_tile = np.repeat(ref, q.shape[0], axis=0)
    return quat_delta_angle_deg(ref_tile, q)


# -----------------------------
# Derived columns and diagnostics
# -----------------------------

def add_sequence_diagnostics(df: pd.DataFrame, dt_outlier_factor: float = 4.0) -> pd.DataFrame:
    df = df.copy()

    if "image_name" in df.columns:
        df["image_number"] = df["image_name"].map(parse_image_number)
        df["frame_gap"] = df["image_number"].diff()
    elif "image_number" not in df.columns:
        df["image_number"] = np.nan
        df["frame_gap"] = np.nan
    else:
        df["frame_gap"] = pd.to_numeric(df["image_number"], errors="coerce").diff()

    if "image_time" in df.columns:
        df["dt"] = pd.to_numeric(df["image_time"], errors="coerce").diff()
    elif "time" in df.columns:
        df["dt"] = pd.to_numeric(df["time"], errors="coerce").diff()
    elif "dt" not in df.columns:
        df["dt"] = np.nan

    positive_dt = pd.to_numeric(df["dt"], errors="coerce").to_numpy(dtype=float)
    positive_dt = positive_dt[np.isfinite(positive_dt) & (positive_dt > 0.0)]
    nominal_dt = float(np.median(positive_dt)) if positive_dt.size else np.nan
    df["nominal_dt"] = nominal_dt

    gap = pd.to_numeric(df["frame_gap"], errors="coerce").to_numpy(dtype=float)
    gap_bad = np.isfinite(gap) & (gap != 1.0)

    dt = pd.to_numeric(df["dt"], errors="coerce").to_numpy(dtype=float)
    if np.isfinite(nominal_dt) and nominal_dt > 0:
        dt_bad = np.isfinite(dt) & ((dt <= 0.0) | (dt > dt_outlier_factor * nominal_dt))
    else:
        dt_bad = np.zeros(len(df), dtype=bool)

    if len(df):
        gap_bad[0] = False
        dt_bad[0] = False

    df["sequence_artifact"] = gap_bad | dt_bad
    return df


def apply_manual_exclusion_windows(df: pd.DataFrame, windows: list[tuple[str, int]]) -> pd.DataFrame:
    df = df.copy()
    manual = np.zeros(len(df), dtype=bool)
    if not windows or "image_number" not in df.columns:
        df["manual_exclusion"] = manual
        return df

    nums = pd.to_numeric(df["image_number"], errors="coerce").to_numpy(dtype=float)
    for image_name, half_width in windows:
        center = parse_image_number(image_name)
        if not np.isfinite(center):
            continue
        manual |= np.isfinite(nums) & (nums >= center - half_width) & (nums <= center + half_width)
    df["manual_exclusion"] = manual
    return df


def add_count_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    count_specs = {
        "pnp_inlier_count": "pnp_inlier_class_ids",
        "pnp_outlier_count": "pnp_outlier_class_ids",
        "kf_rejected_measurement_count": "kf_rejected_measurement_class_ids",
        "kf_track_count": "kf_track_class_ids",
        "pose_feature_count": "pose_class_ids",
    }

    for count_col, ids_col in count_specs.items():
        if count_col not in df.columns and ids_col in df.columns:
            df[count_col] = df[ids_col].map(count_id_string)

    if "kf_accepted_measurement_count" not in df.columns:
        if "kf_track_count" in df.columns and "kf_rejected_measurement_count" in df.columns:
            df["kf_accepted_measurement_count"] = (
                pd.to_numeric(df["kf_track_count"], errors="coerce")
                - pd.to_numeric(df["kf_rejected_measurement_count"], errors="coerce")
            ).clip(lower=0)

    return df


def compute_rejection_overlap(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "pnp_outlier_class_ids" not in df.columns or "kf_rejected_measurement_class_ids" not in df.columns:
        df["ransac_kf_reject_overlap_count"] = np.nan
        return df

    overlaps = []
    for pnp_out, kf_rej in zip(df["pnp_outlier_class_ids"], df["kf_rejected_measurement_class_ids"]):
        overlaps.append(len(parse_id_set(pnp_out) & parse_id_set(kf_rej)))
    df["ransac_kf_reject_overlap_count"] = overlaps
    return df


def add_range_from_angle_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, MethodSpec | None]:
    """Create a standardized single-feature xyz estimate if az/el/range columns exist."""
    df = df.copy()

    range_candidates = ["range_m", "range", "rho", "distance", "distance_m", "sf_range_m", "single_range_m"]
    az_rad_candidates = ["azimuth_rad", "az_rad", "bearing_rad", "theta_az_rad", "sf_azimuth_rad"]
    el_rad_candidates = ["elevation_rad", "el_rad", "theta_el_rad", "sf_elevation_rad"]
    az_deg_candidates = ["azimuth_deg", "az_deg", "bearing_deg", "theta_az_deg", "sf_azimuth_deg"]
    el_deg_candidates = ["elevation_deg", "el_deg", "theta_el_deg", "sf_elevation_deg"]

    def first_existing(candidates: Sequence[str]) -> str | None:
        return next((c for c in candidates if c in df.columns), None)

    r_col = first_existing(range_candidates)
    az_col = first_existing(az_rad_candidates)
    el_col = first_existing(el_rad_candidates)
    az_deg_col = first_existing(az_deg_candidates)
    el_deg_col = first_existing(el_deg_candidates)

    if r_col is None:
        return df, None

    if az_col is None and az_deg_col is not None:
        az = np.deg2rad(pd.to_numeric(df[az_deg_col], errors="coerce").to_numpy(dtype=float))
    elif az_col is not None:
        az = pd.to_numeric(df[az_col], errors="coerce").to_numpy(dtype=float)
    else:
        return df, None

    if el_col is None and el_deg_col is not None:
        el = np.deg2rad(pd.to_numeric(df[el_deg_col], errors="coerce").to_numpy(dtype=float))
    elif el_col is not None:
        el = pd.to_numeric(df[el_col], errors="coerce").to_numpy(dtype=float)
    else:
        return df, None

    rho = pd.to_numeric(df[r_col], errors="coerce").to_numpy(dtype=float)

    # Camera-frame convention: x right, y down/up depending source sign convention, z forward.
    # This uses the common azimuth/elevation spherical convention and is intended as a
    # practical comparison proxy. If your source uses y-up elevation, flip the source column
    # before export or replace this conversion.
    df["single_feature_from_az_el_range_tvec_x"] = rho * np.sin(az) * np.cos(el)
    df["single_feature_from_az_el_range_tvec_y"] = rho * np.sin(el)
    df["single_feature_from_az_el_range_tvec_z"] = rho * np.cos(az) * np.cos(el)

    valid_col = next((c for c in ["valid", "single_valid", "sf_valid"] if c in df.columns), None)
    if valid_col is None:
        valid_col = "single_feature_from_az_el_range_valid"
        df[valid_col] = np.isfinite(rho) & np.isfinite(az) & np.isfinite(el) & (rho > 0.0)

    return df, MethodSpec(
        key="single_feature",
        label="Single-feature",
        valid_col=valid_col,
        t_cols=(
            "single_feature_from_az_el_range_tvec_x",
            "single_feature_from_az_el_range_tvec_y",
            "single_feature_from_az_el_range_tvec_z",
        ),
        source="single",
    )


def available_primary_methods(df: pd.DataFrame) -> list[MethodSpec]:
    """Return primary pose methods that have the required columns and at least one valid row."""
    found = []
    for m in PRIMARY_METHODS:
        required = [m.valid_col, *m.t_cols]
        if m.q_cols is not None:
            required.extend(m.q_cols)
        if not available_columns(df, required):
            continue
        if m.valid_col is not None and not bool(as_bool_series(df[m.valid_col]).any()):
            continue
        found.append(m)
    return found


def discover_single_feature_method(df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, MethodSpec | None]:
    """Find or create a single-feature position method from a secondary CSV."""
    df = df.copy()

    # Direct xyz/tvec columns first.
    for key, t_cols, valid_col in SINGLE_POSITION_CANDIDATES:
        if available_columns(df, t_cols):
            q_cols = None
            for _, qc in SINGLE_ATTITUDE_CANDIDATES:
                if available_columns(df, qc):
                    q_cols = qc
                    break

            if valid_col not in df.columns:
                valid_col = f"{key}_auto_valid"
                mat = numeric_matrix(df, t_cols)
                df[valid_col] = np.all(np.isfinite(mat), axis=1)

            return df, MethodSpec(
                key="single_feature",
                label=label,
                valid_col=valid_col,
                t_cols=t_cols,
                q_cols=q_cols,
                rmse_col=next((c for c in ["single_rmse_px", "sf_rmse_px", "rmse_px"] if c in df.columns), None),
                source="single",
            )

    # If no direct xyz columns exist, try azimuth/elevation/range.
    df, m = add_range_from_angle_columns(df)
    if m is not None:
        return df, MethodSpec(
            key=m.key,
            label=label,
            valid_col=m.valid_col,
            t_cols=m.t_cols,
            q_cols=m.q_cols,
            rmse_col=m.rmse_col,
            source="single",
        )

    return df, None


# -----------------------------
# Loading / merging
# -----------------------------

def preprocess_common(df: pd.DataFrame, dt_outlier_factor: float) -> pd.DataFrame:
    df = add_sequence_diagnostics(df, dt_outlier_factor=dt_outlier_factor)
    df = add_count_columns(df)
    df = compute_rejection_overlap(df)
    return df


def merge_single_feature_csv(
    primary: pd.DataFrame,
    primary_methods: list[MethodSpec],
    single_csv: Path,
    *,
    label: str,
    dt_outlier_factor: float,
    time_tolerance: float | None,
) -> tuple[pd.DataFrame, list[MethodSpec], str]:
    """Merge a secondary single-feature CSV onto the primary frame table.

    The solo-network files generated by the dissertation pipeline contain columns named

        single_feature_estimated_tvec_x
        single_feature_estimated_tvec_y
        single_feature_estimated_tvec_z
        single_feature_estimated_range

    plus image/frame/time identifiers.  This merger keeps only the discovered
    single-feature method columns, renames them with a ``single_feature__`` prefix, and
    joins them to the primary multi-feature table.  It tries several merge keys and keeps
    the one that produces the most valid single-feature rows, which avoids a common
    trap where two files have image names from different folders but matching frame
    indices or timestamps.
    """
    single_raw = pd.read_csv(single_csv)
    single_raw = preprocess_common(single_raw, dt_outlier_factor=dt_outlier_factor)
    single_raw, single_method = discover_single_feature_method(single_raw, label=label)
    if single_method is None:
        raise RuntimeError(
            "Could not find single-feature position columns in the secondary CSV. "
            "Expected xyz/tvec columns such as single_feature_estimated_tvec_x/y/z, "
            "single_feature_tvec_x/y/z, sf_x/y/z, x/y/z, or azimuth/elevation/range columns."
        )

    # Keep only merge keys and the standardized method columns. Rename method columns so
    # they cannot collide with primary columns.
    rename: dict[str, str] = {}
    single_cols_needed = set(single_method.t_cols)
    if single_method.q_cols:
        single_cols_needed.update(single_method.q_cols)
    if single_method.valid_col:
        single_cols_needed.add(single_method.valid_col)
    if single_method.rmse_col:
        single_cols_needed.add(single_method.rmse_col)

    for c in single_cols_needed:
        rename[c] = f"single_feature__{c}"

    key_cols = [c for c in ["image_name", "image_number", "frame_idx", "image_time", "time"] if c in single_raw.columns]
    single_keep = single_raw.loc[:, sorted(set(key_cols) | single_cols_needed)].rename(columns=rename)

    new_t_cols = tuple(rename[c] for c in single_method.t_cols)
    new_q_cols = tuple(rename[c] for c in single_method.q_cols) if single_method.q_cols else None
    new_valid_col = rename[single_method.valid_col] if single_method.valid_col else None
    new_rmse_col = rename[single_method.rmse_col] if single_method.rmse_col else None

    method = MethodSpec(
        key="single_feature",
        label=label,
        valid_col=new_valid_col,
        t_cols=new_t_cols,
        q_cols=new_q_cols,
        rmse_col=new_rmse_col,
        source="single",
    )

    def valid_coverage(candidate: pd.DataFrame) -> int:
        try:
            return int(np.sum(method_valid_mask(candidate, method)))
        except Exception:
            return 0

    candidates: list[tuple[int, str, pd.DataFrame]] = []

    # Prefer exact image names when they actually overlap, but do not blindly accept a
    # zero-coverage merge just because the columns exist.
    for key in ["image_name", "image_number", "frame_idx"]:
        if key in primary.columns and key in single_keep.columns:
            merged = primary.merge(single_keep, on=key, how="left", suffixes=("", "_single"))
            candidates.append((valid_coverage(merged), f"Merged single-feature CSV by {key}.", merged))

    # Last resort: merge-asof by time.
    primary_time_col = "image_time" if "image_time" in primary.columns else ("time" if "time" in primary.columns else None)
    single_time_col = "image_time" if "image_time" in single_keep.columns else ("time" if "time" in single_keep.columns else None)
    if primary_time_col is not None and single_time_col is not None:
        left = primary.sort_values(primary_time_col).copy()
        right = single_keep.sort_values(single_time_col).copy()
        tol = time_tolerance
        if tol is None:
            dt = pd.to_numeric(primary.get("dt", pd.Series(dtype=float)), errors="coerce")
            pos_dt = dt[(dt > 0) & np.isfinite(dt)]
            tol = float(0.5 * pos_dt.median()) if not pos_dt.empty else None
        merged_time = pd.merge_asof(
            left,
            right,
            left_on=primary_time_col,
            right_on=single_time_col,
            direction="nearest",
            tolerance=tol,
            suffixes=("", "_single"),
        ).sort_index()
        candidates.append((valid_coverage(merged_time), f"Merged single-feature CSV by nearest time, tolerance={tol}.", merged_time))

    if not candidates:
        raise RuntimeError("Could not merge single-feature CSV: no shared image, frame, or time key found.")

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_coverage, merge_note, merged = candidates[0]
    merge_note = f"{merge_note} Valid merged single-feature rows: {best_coverage}/{len(primary)}."

    methods = [*primary_methods, method]
    return merged, methods, merge_note


def append_primary_single_feature_method(
    df: pd.DataFrame,
    methods: list[MethodSpec],
    *,
    label: str,
) -> tuple[pd.DataFrame, list[MethodSpec], str]:
    """Add a single-feature method from the primary CSV when solo columns are present."""
    df2, single_method = discover_single_feature_method(df, label=label)
    if single_method is None:
        return df, methods, ""

    # Avoid appending a duplicate if a future primary method with the same t columns exists.
    if any(tuple(m.t_cols) == tuple(single_method.t_cols) for m in methods):
        return df2, methods, ""

    single_method = MethodSpec(
        key="single_feature",
        label=label,
        valid_col=single_method.valid_col,
        t_cols=single_method.t_cols,
        q_cols=single_method.q_cols,
        rmse_col=single_method.rmse_col,
        source="primary_single",
    )
    coverage = int(np.sum(method_valid_mask(df2, single_method)))
    note = f"Detected single-feature estimate columns in the primary CSV. Valid rows: {coverage}/{len(df2)}."
    return df2, [*methods, single_method], note


# -----------------------------
# Analysis products
# -----------------------------

def method_valid_mask(df: pd.DataFrame, m: MethodSpec) -> np.ndarray:
    if m.valid_col is not None and m.valid_col in df.columns:
        return as_bool_series(df[m.valid_col]).to_numpy(dtype=bool)
    mat = numeric_matrix(df, m.t_cols)
    ok = np.all(np.isfinite(mat), axis=1)
    if m.q_cols is not None:
        ok &= np.all(np.isfinite(numeric_matrix(df, m.q_cols)), axis=1)
    return ok


def method_position_norm(df: pd.DataFrame, m: MethodSpec) -> np.ndarray:
    t = numeric_matrix(df, m.t_cols)
    valid = method_valid_mask(df, m)
    dist = np.linalg.norm(t, axis=1)
    return np.where(valid, dist, np.nan)


def compute_method_deltas(df: pd.DataFrame, m: MethodSpec) -> pd.DataFrame:
    valid = method_valid_mask(df, m)
    t = numeric_matrix(df, m.t_cols)

    trans_delta = np.full(len(df), np.nan)
    attitude_delta = np.full(len(df), np.nan)

    if len(df) >= 2:
        pair_valid = valid[1:] & valid[:-1]
        trans_step = np.linalg.norm(t[1:] - t[:-1], axis=1)
        trans_delta[1:] = np.where(pair_valid, trans_step, np.nan)

        if m.q_cols is not None:
            q = numeric_matrix(df, m.q_cols)
            att_step = quat_delta_angle_deg(q[:-1], q[1:])
            attitude_delta[1:] = np.where(pair_valid, att_step, np.nan)

    return pd.DataFrame(
        {
            "frame_idx": df["frame_idx"] if "frame_idx" in df.columns else np.arange(len(df)),
            "image_name": df["image_name"] if "image_name" in df.columns else "",
            "image_number": df["image_number"] if "image_number" in df.columns else np.nan,
            "sequence_artifact": df["sequence_artifact"] if "sequence_artifact" in df.columns else False,
            "manual_exclusion": df["manual_exclusion"] if "manual_exclusion" in df.columns else False,
            "valid": valid,
            f"{m.key}_delta_tvec_norm": trans_delta,
            f"{m.key}_delta_attitude_deg": attitude_delta,
        }
    )


def summarize_availability(df: pd.DataFrame, methods: list[MethodSpec]) -> pd.DataFrame:
    rows = []
    n = len(df)
    for m in methods:
        valid = method_valid_mask(df, m)
        rows.append(
            {
                "method": m.label,
                "key": m.key,
                "source": m.source,
                "has_attitude": bool(m.has_attitude),
                "valid_frames": int(np.sum(valid)),
                "total_frames": int(n),
                "valid_percent": float(100.0 * np.mean(valid)) if n else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_base_pair_mask(
    df: pd.DataFrame,
    *,
    exclude_sequence_artifacts: bool,
    skip_initial: int,
) -> np.ndarray:
    base_pair_ok = np.ones(len(df), dtype=bool)
    if len(df):
        base_pair_ok[0] = False
    if exclude_sequence_artifacts:
        seq_bad = df.get("sequence_artifact", pd.Series(False, index=df.index)).to_numpy(dtype=bool)
        man_bad = df.get("manual_exclusion", pd.Series(False, index=df.index)).to_numpy(dtype=bool)
        endpoint_bad = seq_bad | man_bad
        prev_bad = np.r_[False, endpoint_bad[:-1]]
        base_pair_ok &= ~(endpoint_bad | prev_bad)
    if skip_initial > 0:
        base_pair_ok[: min(skip_initial, len(base_pair_ok))] = False
    return base_pair_ok


def summarize_smoothness(
    df: pd.DataFrame,
    methods: list[MethodSpec],
    *,
    common_valid: bool,
    exclude_sequence_artifacts: bool,
    skip_initial: int,
) -> pd.DataFrame:
    rows = []
    base_pair_ok = build_base_pair_mask(
        df,
        exclude_sequence_artifacts=exclude_sequence_artifacts,
        skip_initial=skip_initial,
    )

    if common_valid:
        common = np.ones(len(df), dtype=bool)
        for m in methods:
            common &= method_valid_mask(df, m)
        common_pair = common & np.r_[False, common[:-1]]
        base_pair_ok &= common_pair

    for m in methods:
        d = compute_method_deltas(df, m)
        method_pair_ok = base_pair_ok.copy()
        if not common_valid:
            valid = method_valid_mask(df, m)
            method_pair_ok &= valid & np.r_[False, valid[:-1]]

        trans = d[f"{m.key}_delta_tvec_norm"].to_numpy(dtype=float)
        att = d[f"{m.key}_delta_attitude_deg"].to_numpy(dtype=float)
        trans = np.where(method_pair_ok, trans, np.nan)
        att = np.where(method_pair_ok, att, np.nan)
        ts = finite_stats(trans)
        a_s = finite_stats(att)

        rows.append(
            {
                "method": m.label,
                "key": m.key,
                "source": m.source,
                "has_attitude": bool(m.has_attitude),
                "pair_count": ts["count"],
                "trans_mean": ts["mean"],
                "trans_median": ts["median"],
                "trans_p95": ts["p95"],
                "trans_max": ts["max"],
                "att_count": a_s["count"],
                "att_mean_deg": a_s["mean"],
                "att_median_deg": a_s["median"],
                "att_p95_deg": a_s["p95"],
                "att_max_deg": a_s["max"],
                "common_valid": common_valid,
                "sequence_artifacts_excluded": exclude_sequence_artifacts,
                "skip_initial": skip_initial,
            }
        )
    return pd.DataFrame(rows)


def summarize_rmse(df: pd.DataFrame, methods: list[MethodSpec]) -> pd.DataFrame:
    rows = []
    for m in methods:
        if m.rmse_col and m.rmse_col in df.columns:
            stats = finite_stats(pd.to_numeric(df[m.rmse_col], errors="coerce"))
            rows.append(
                {
                    "method": m.label,
                    "key": m.key,
                    "source": m.source,
                    "rmse_count": stats["count"],
                    "rmse_mean_px": stats["mean"],
                    "rmse_median_px": stats["median"],
                    "rmse_p95_px": stats["p95"],
                    "rmse_max_px": stats["max"],
                }
            )
    return pd.DataFrame(rows)


def summarize_health(df: pd.DataFrame) -> pd.DataFrame:
    health_cols = [
        "pose_feature_count",
        "pnp_inlier_count",
        "pnp_outlier_count",
        "kf_track_count",
        "kf_accepted_measurement_count",
        "kf_rejected_measurement_count",
        "ransac_kf_reject_overlap_count",
        "feature_spread_rms_px",
        "feature_spread_mean_radius_px",
        "apparent_target_bbox_width_px",
        "apparent_target_bbox_height_px",
        "apparent_target_bbox_diag_px",
        "apparent_target_bbox_area_px",
    ]
    rows = []
    for col in health_cols:
        if col in df.columns:
            stats = finite_stats(pd.to_numeric(df[col], errors="coerce"))
            rows.append({"quantity": col, **stats})
    return pd.DataFrame(rows)


# -----------------------------
# Plots
# -----------------------------

def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def x_axis(df: pd.DataFrame) -> tuple[pd.Series | np.ndarray, str]:
    if "image_number" in df.columns and df["image_number"].notna().any():
        return df["image_number"], "Image number"
    if "image_time" in df.columns and df["image_time"].notna().any():
        return df["image_time"], "Image time"
    if "frame_idx" in df.columns:
        return df["frame_idx"], "Frame index"
    return np.arange(len(df)), "Frame"


def rotate_xticks() -> None:
    plt.xticks(rotation=25, ha="right")


def plot_availability(summary: pd.DataFrame, out: Path) -> None:
    if summary.empty:
        return
    plt.figure(figsize=(8.5, 4.8))
    plt.bar(summary["method"], summary["valid_percent"])
    plt.ylabel("Valid frames (%)")
    plt.title("Method availability")
    rotate_xticks()
    plt.ylim(0, max(105, float(summary["valid_percent"].max()) * 1.1))
    savefig(out / "availability_percent.png")


def plot_smoothness_bars(summary: pd.DataFrame, out: Path, suffix: str) -> None:
    if summary.empty:
        return

    plot_specs = [
        ("trans_median", "Median translation jump", "smoothness_translation_median", "Median frame-to-frame translation jump", None),
        ("trans_p95", "95th percentile translation jump", "smoothness_translation_p95", "95th percentile frame-to-frame translation jump", None),
        ("att_median_deg", "Median attitude jump (deg)", "smoothness_attitude_median", "Median frame-to-frame attitude jump", "att_count"),
        ("att_p95_deg", "95th percentile attitude jump (deg)", "smoothness_attitude_p95", "95th percentile frame-to-frame attitude jump", "att_count"),
    ]

    for metric, ylabel, fname, title, count_col in plot_specs:
        if metric not in summary.columns:
            continue
        plot_df = summary.copy()
        if count_col is not None and count_col in plot_df.columns:
            plot_df = plot_df[plot_df[count_col] > 0]
        plot_df = plot_df[np.isfinite(pd.to_numeric(plot_df[metric], errors="coerce"))]
        if plot_df.empty:
            continue
        plt.figure(figsize=(8.5, 4.8))
        plt.bar(plot_df["method"], pd.to_numeric(plot_df[metric], errors="coerce"))
        plt.ylabel(ylabel)
        plt.title(title)
        rotate_xticks()
        savefig(out / f"{fname}_{suffix}.png")


def plot_time_series_deltas(df: pd.DataFrame, methods: list[MethodSpec], out: Path, skip_initial: int) -> None:
    x, xlabel = x_axis(df)

    plt.figure(figsize=(10.5, 5.2))
    for m in methods:
        d = compute_method_deltas(df, m)
        y = d[f"{m.key}_delta_tvec_norm"].to_numpy(dtype=float)
        if skip_initial:
            y[:skip_initial] = np.nan
        plt.plot(x, y, linewidth=1.0, label=m.label)
    plt.xlabel(xlabel)
    plt.ylabel("Translation jump")
    plt.title("Frame-to-frame translation jump")
    plt.legend(fontsize="small")
    savefig(out / "timeseries_translation_jump.png")

    attitude_methods = [m for m in methods if m.has_attitude]
    if attitude_methods:
        plt.figure(figsize=(10.5, 5.2))
        for m in attitude_methods:
            d = compute_method_deltas(df, m)
            y = d[f"{m.key}_delta_attitude_deg"].to_numpy(dtype=float)
            if skip_initial:
                y[:skip_initial] = np.nan
            plt.plot(x, y, linewidth=1.0, label=m.label)
        plt.xlabel(xlabel)
        plt.ylabel("Attitude jump (deg)")
        plt.title("Frame-to-frame attitude jump")
        plt.legend(fontsize="small")
        savefig(out / "timeseries_attitude_jump.png")


def plot_tracking_comparisons(df: pd.DataFrame, methods: list[MethodSpec], out: Path) -> None:
    """Plot distance and attitude tracking comparisons across methods."""
    x, xlabel = x_axis(df)

    # Distance/range tracking: all methods with position estimates, including single-feature.
    plt.figure(figsize=(10.5, 5.2))
    for m in methods:
        y = method_position_norm(df, m)
        plt.plot(x, y, linewidth=1.0, label=m.label)
    plt.xlabel(xlabel)
    plt.ylabel("Distance / translation norm")
    plt.title("Distance tracking comparison")
    plt.legend(fontsize="small")
    savefig(out / "timeseries_distance_tracking.png")

    # Distance difference relative to OpenCV SolvePnP if present, otherwise first method.
    if len(methods) >= 2:
        ref = next((m for m in methods if m.key == "pnp"), methods[0])
        ref_dist = method_position_norm(df, ref)
        plt.figure(figsize=(10.5, 5.2))
        for m in methods:
            if m.key == ref.key:
                continue
            y = method_position_norm(df, m) - ref_dist
            plt.plot(x, y, linewidth=1.0, label=f"{m.label} - {ref.label}")
        plt.axhline(0.0, linewidth=0.8)
        plt.xlabel(xlabel)
        plt.ylabel("Distance difference")
        plt.title(f"Distance difference relative to {ref.label}")
        plt.legend(fontsize="small")
        savefig(out / "timeseries_distance_difference_vs_reference.png")

    # Attitude tracking: angle from each method's first valid attitude.
    attitude_methods = [m for m in methods if m.has_attitude]
    if attitude_methods:
        plt.figure(figsize=(10.5, 5.2))
        for m in attitude_methods:
            q = numeric_matrix(df, m.q_cols)  # type: ignore[arg-type]
            valid = method_valid_mask(df, m) & np.all(np.isfinite(q), axis=1)
            y = np.full(len(df), np.nan)
            if np.any(valid):
                ref_q = q[np.argmax(valid)]
                y = quat_angle_from_reference_deg(q, ref_q)
                y = np.where(valid, y, np.nan)
            plt.plot(x, y, linewidth=1.0, label=m.label)
        plt.xlabel(xlabel)
        plt.ylabel("Attitude change from first valid pose (deg)")
        plt.title("Attitude tracking comparison")
        plt.legend(fontsize="small")
        savefig(out / "timeseries_attitude_tracking_relative_initial.png")

        # Attitude difference relative to OpenCV SolvePnP if present, otherwise first attitude method.
        if len(attitude_methods) >= 2:
            ref = next((m for m in attitude_methods if m.key == "pnp"), attitude_methods[0])
            ref_q = numeric_matrix(df, ref.q_cols)  # type: ignore[arg-type]
            ref_valid = method_valid_mask(df, ref)
            plt.figure(figsize=(10.5, 5.2))
            for m in attitude_methods:
                if m.key == ref.key:
                    continue
                q = numeric_matrix(df, m.q_cols)  # type: ignore[arg-type]
                valid = method_valid_mask(df, m) & ref_valid
                diff = quat_delta_angle_deg(ref_q, q)
                diff = np.where(valid, diff, np.nan)
                plt.plot(x, diff, linewidth=1.0, label=f"{m.label} vs. {ref.label}")
            plt.xlabel(xlabel)
            plt.ylabel("Attitude difference (deg)")
            plt.title(f"Attitude difference relative to {ref.label}")
            plt.legend(fontsize="small")
            savefig(out / "timeseries_attitude_difference_vs_reference.png")


def plot_rmse(df: pd.DataFrame, methods: list[MethodSpec], out: Path) -> None:
    rmse_methods = [m for m in methods if m.rmse_col and m.rmse_col in df.columns]
    if not rmse_methods:
        return
    x, xlabel = x_axis(df)
    plt.figure(figsize=(10.5, 5.2))
    for m in rmse_methods:
        plt.plot(x, pd.to_numeric(df[m.rmse_col], errors="coerce"), linewidth=1.0, label=m.label)
    plt.xlabel(xlabel)
    plt.ylabel("Reprojection RMSE (px)")
    plt.title("Reprojection residuals")
    plt.legend(fontsize="small")
    savefig(out / "timeseries_reprojection_rmse.png")


def plot_feature_health(df: pd.DataFrame, out: Path) -> None:
    x, xlabel = x_axis(df)

    count_cols = [
        ("pose_feature_count", "Pose features"),
        ("pnp_inlier_count", "PnP/RANSAC inliers"),
        ("pnp_outlier_count", "PnP/RANSAC outliers"),
        ("kf_accepted_measurement_count", "KF accepted"),
        ("kf_rejected_measurement_count", "KF rejected"),
        ("ransac_kf_reject_overlap_count", "RANSAC/KF rejection overlap"),
    ]
    if any(c in df.columns for c, _ in count_cols):
        plt.figure(figsize=(10.5, 5.2))
        for col, label in count_cols:
            if col in df.columns:
                plt.plot(x, pd.to_numeric(df[col], errors="coerce"), linewidth=1.0, label=label)
        plt.xlabel(xlabel)
        plt.ylabel("Count")
        plt.title("Feature-health and rejection diagnostics")
        plt.legend(fontsize="small")
        savefig(out / "timeseries_feature_health_counts.png")

    spread_cols = [
        ("feature_spread_rms_px", "Feature spread RMS"),
        ("feature_spread_mean_radius_px", "Feature mean radius"),
        ("apparent_target_bbox_diag_px", "Apparent bbox diagonal"),
    ]
    if any(c in df.columns for c, _ in spread_cols):
        plt.figure(figsize=(10.5, 5.2))
        for col, label in spread_cols:
            if col in df.columns:
                plt.plot(x, pd.to_numeric(df[col], errors="coerce"), linewidth=1.0, label=label)
        plt.xlabel(xlabel)
        plt.ylabel("Pixels")
        plt.title("Apparent target size and feature spread")
        plt.legend(fontsize="small")
        savefig(out / "timeseries_feature_spread_size.png")


def plot_sequence_artifacts(df: pd.DataFrame, out: Path) -> None:
    if "dt" not in df.columns and "frame_gap" not in df.columns:
        return
    x, xlabel = x_axis(df)

    if "dt" in df.columns and df["dt"].notna().any():
        plt.figure(figsize=(10.5, 4.2))
        plt.plot(x, df["dt"], linewidth=1.0)
        plt.xlabel(xlabel)
        plt.ylabel("dt")
        plt.title("Frame timestamp spacing")
        savefig(out / "sequence_dt.png")

    if "frame_gap" in df.columns and df["frame_gap"].notna().any():
        plt.figure(figsize=(10.5, 4.2))
        plt.plot(x, df["frame_gap"], linewidth=1.0)
        plt.xlabel(xlabel)
        plt.ylabel("Image-number gap")
        plt.title("Image filename sequence gaps")
        savefig(out / "sequence_frame_gap.png")


def plot_operating_envelope_scatter(df: pd.DataFrame, methods: list[MethodSpec], out: Path) -> None:
    if not methods:
        return
    chosen = next((m for m in methods if m.key == "wqnp_kfest"), methods[-1])
    d = compute_method_deltas(df, chosen)
    y = d[f"{chosen.key}_delta_tvec_norm"].to_numpy(dtype=float)

    for xcol, xlabel, fname in [
        ("feature_spread_rms_px", "Feature spread RMS (px)", "scatter_spread_vs_translation_jump.png"),
        ("apparent_target_bbox_diag_px", "Apparent bbox diagonal (px)", "scatter_bbox_diag_vs_translation_jump.png"),
        ("pose_feature_count", "Pose feature count", "scatter_feature_count_vs_translation_jump.png"),
    ]:
        if xcol not in df.columns:
            continue
        x = pd.to_numeric(df[xcol], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if np.sum(mask) < 3:
            continue
        plt.figure(figsize=(6.8, 5.2))
        plt.scatter(x[mask], y[mask], s=10, alpha=0.6)
        plt.xlabel(xlabel)
        plt.ylabel(f"{chosen.label} translation jump")
        plt.title("Operating-envelope diagnostic")
        savefig(out / fname)


# -----------------------------
# Narrative summary
# -----------------------------

def build_text_summary(
    csv_path: Path,
    single_csv_path: Path | None,
    merge_note: str,
    df: pd.DataFrame,
    availability: pd.DataFrame,
    smooth_common: pd.DataFrame,
    smooth_per_method: pd.DataFrame,
    rmse: pd.DataFrame,
    health: pd.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append(f"Navigation comparison summary for {csv_path.name}")
    lines.append("=" * len(lines[-1]))
    lines.append("")
    if single_csv_path is not None:
        lines.append(f"Single-feature CSV: {single_csv_path.name}")
        lines.append(merge_note)
        lines.append("")
    elif merge_note:
        lines.append(merge_note)
        lines.append("")
    lines.append(f"Frames: {len(df)}")
    if "image_time" in df.columns and "dt" in df.columns:
        dt = pd.to_numeric(df["dt"], errors="coerce")
        pos_dt = dt[(dt > 0) & np.isfinite(dt)]
        if not pos_dt.empty:
            lines.append(f"Nominal dt: {pos_dt.median():.6g} s  ({1.0 / pos_dt.median():.3g} Hz)")
    if "sequence_artifact" in df.columns:
        lines.append(f"Sequence artifacts marked: {int(df['sequence_artifact'].sum())}")
    if "manual_exclusion" in df.columns:
        lines.append(f"Manual-exclusion frames marked: {int(df['manual_exclusion'].sum())}")
    lines.append("")

    lines.append("Availability")
    lines.append("------------")
    for _, row in availability.iterrows():
        attitude_note = "attitude" if row.get("has_attitude", False) else "position-only"
        lines.append(
            f"{row['method']} ({attitude_note}): {int(row['valid_frames'])}/{int(row['total_frames'])} "
            f"({row['valid_percent']:.2f}%)"
        )
    lines.append("")

    if not smooth_common.empty:
        lines.append("Common-valid smoothness, sequence artifacts excluded")
        lines.append("---------------------------------------------------")
        best_trans = smooth_common.sort_values("trans_median").iloc[0]
        att_rows = smooth_common[smooth_common["att_count"] > 0]
        best_att = att_rows.sort_values("att_median_deg").iloc[0] if not att_rows.empty else None
        for _, row in smooth_common.iterrows():
            att_text = ""
            if row["att_count"] > 0 and np.isfinite(row["att_median_deg"]):
                att_text = f", median Δθ={row['att_median_deg']:.6g} deg, 95% Δθ={row['att_p95_deg']:.6g} deg"
            lines.append(
                f"{row['method']}: median Δt={row['trans_median']:.6g}, "
                f"95% Δt={row['trans_p95']:.6g}{att_text}"
            )
        lines.append(f"Smoothest median translation jump: {best_trans['method']}")
        if best_att is not None:
            lines.append(f"Smoothest median attitude jump: {best_att['method']}")
        lines.append("")

    if not rmse.empty:
        lines.append("Reprojection RMSE")
        lines.append("-----------------")
        for _, row in rmse.iterrows():
            lines.append(
                f"{row['method']}: median={row['rmse_median_px']:.6g} px, "
                f"95%={row['rmse_p95_px']:.6g} px"
            )
        lines.append(
            "Note: KF-center wQnP residuals may be evaluated against raw YOLO centers "
            "depending on CSV generation; compare with care."
        )
        lines.append("")

    if not health.empty:
        lines.append("Feature-health highlights")
        lines.append("-------------------------")
        for q in [
            "pose_feature_count",
            "pnp_inlier_count",
            "pnp_outlier_count",
            "kf_accepted_measurement_count",
            "kf_rejected_measurement_count",
            "feature_spread_rms_px",
            "apparent_target_bbox_diag_px",
        ]:
            match = health[health["quantity"] == q]
            if not match.empty:
                row = match.iloc[0]
                lines.append(
                    f"{q}: median={row['median']:.6g}, mean={row['mean']:.6g}, "
                    f"range={row['p05']:.6g}--{row['p95']:.6g} (5--95%)"
                )
        lines.append("")

    if {"pnp_outlier_count", "kf_rejected_measurement_count", "ransac_kf_reject_overlap_count"}.issubset(df.columns):
        lines.append("RANSAC/KF rejection comparison")
        lines.append("-----------------------------")
        for c, label in [
            ("pnp_outlier_count", "OpenCV/RANSAC outliers"),
            ("kf_rejected_measurement_count", "KF rejected measurements"),
            ("ransac_kf_reject_overlap_count", "Overlap"),
        ]:
            s = pd.to_numeric(df[c], errors="coerce")
            lines.append(f"{label}: mean={s.mean():.6g}, median={s.median():.6g}")
        lines.append("Interpretation: geometric RANSAC and temporal KF gating are complementary diagnostics.")
        lines.append("")

    return "\n".join(lines)


# -----------------------------
# Main / CLI / GUI
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze navcalcs pose-comparison CSVs, optionally with single-feature CSV.")
    parser.add_argument("csv", type=Path, nargs="?", help="Primary navcalcs CSV file.")
    parser.add_argument("--single-csv", type=Path, default=None, help="Optional single-feature CSV to merge and compare.")
    parser.add_argument("--single-label", default="Single-feature", help="Legend label for the single-feature method.")
    parser.add_argument("--out", type=Path, default=None, help="Output directory. Defaults to <csv stem>_analysis.")
    parser.add_argument(
        "--skip-initial",
        type=int,
        default=5,
        help="Initial frames to skip for smoothness summaries, useful for KF warm-up. Default: 5.",
    )
    parser.add_argument(
        "--include-sequence-artifacts",
        action="store_true",
        help="Include sequence-artifact steps in smoothness summaries. Default excludes them.",
    )
    parser.add_argument(
        "--dt-outlier-factor",
        type=float,
        default=4.0,
        help="Mark dt as suspicious when dt > factor * nominal dt. Default: 4.0.",
    )
    parser.add_argument(
        "--time-merge-tolerance",
        type=float,
        default=None,
        help="Tolerance for merge-asof by time if no image/frame key matches. Default: half nominal dt.",
    )
    parser.add_argument(
        "--exclude-window",
        nargs=2,
        action="append",
        metavar=("IMAGE_NAME", "HALF_WIDTH"),
        default=[],
        help="Manually exclude +/- HALF_WIDTH frames around IMAGE_NAME for smoothness metrics. Can repeat.",
    )
    parser.add_argument(
        "--no-gui-single",
        action="store_true",
        help="When using GUI mode, do not ask for an optional single-feature CSV.",
    )
    return parser.parse_args()


def select_file_gui(title: str, initial_dir: Path | None = None) -> Path | None:
    try:
        from tkinter import Tk, filedialog
    except Exception:
        return None

    root = Tk()
    root.withdraw()
    filename = filedialog.askopenfilename(
        title=title,
        initialdir=str(initial_dir) if initial_dir else None,
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    root.destroy()
    return Path(filename) if filename else None


def main() -> None:
    args = parse_args()

    default_dir = Path("C:/repos/aburn/usr/jarvis_submodules/camera_calibration_python/Data/Dissertation")

    csv_path = args.csv
    single_csv_path = args.single_csv

    if csv_path is None:
        csv_path = select_file_gui("Select primary navcalcs CSV", default_dir)
        if csv_path is None:
            raise RuntimeError("No primary CSV selected.")
        if not args.no_gui_single and single_csv_path is None:
            single_csv_path = select_file_gui("Optional: select single-feature CSV, or cancel", csv_path.parent)

    out = args.out
    if out is None:
        if single_csv_path is None:
            out = csv_path.with_suffix("").with_name(csv_path.stem + "_analysis")
        else:
            out = csv_path.with_suffix("").with_name(csv_path.stem + "_with_single_analysis")

    ensure_dir(out)
    ensure_dir(out / "plots")
    ensure_dir(out / "tables")

    manual_windows = [(name, int(width)) for name, width in args.exclude_window]

    df = pd.read_csv(csv_path)
    df = preprocess_common(df, dt_outlier_factor=args.dt_outlier_factor)
    df = apply_manual_exclusion_windows(df, manual_windows)

    methods = available_primary_methods(df)

    merge_note = ""
    if single_csv_path is not None:
        df, methods, merge_note = merge_single_feature_csv(
            df,
            methods,
            single_csv_path,
            label=args.single_label,
            dt_outlier_factor=args.dt_outlier_factor,
            time_tolerance=args.time_merge_tolerance,
        )
    else:
        df, methods, merge_note = append_primary_single_feature_method(
            df,
            methods,
            label=args.single_label,
        )

    if not methods:
        raise RuntimeError(
            "No valid pose or single-feature methods found. "
            "Expected primary pose columns or single_feature_estimated_tvec_x/y/z."
        )

    exclude_sequence_artifacts = not args.include_sequence_artifacts

    availability = summarize_availability(df, methods)
    smooth_per_method = summarize_smoothness(
        df,
        methods,
        common_valid=False,
        exclude_sequence_artifacts=exclude_sequence_artifacts,
        skip_initial=args.skip_initial,
    )
    smooth_common = summarize_smoothness(
        df,
        methods,
        common_valid=True,
        exclude_sequence_artifacts=exclude_sequence_artifacts,
        skip_initial=args.skip_initial,
    )
    rmse = summarize_rmse(df, methods)
    health = summarize_health(df)

    # Write derived data and tables.
    df.to_csv(out / "tables" / "derived_navcalcs.csv", index=False)
    availability.to_csv(out / "tables" / "availability_summary.csv", index=False)
    smooth_per_method.to_csv(out / "tables" / "smoothness_per_method_summary.csv", index=False)
    smooth_common.to_csv(out / "tables" / "smoothness_common_valid_summary.csv", index=False)
    rmse.to_csv(out / "tables" / "rmse_summary.csv", index=False)
    health.to_csv(out / "tables" / "feature_health_summary.csv", index=False)

    write_markdown_table(availability, out / "tables" / "availability_summary.md")
    write_markdown_table(smooth_common, out / "tables" / "smoothness_common_valid_summary.md")
    write_markdown_table(smooth_per_method, out / "tables" / "smoothness_per_method_summary.md")
    if not rmse.empty:
        write_markdown_table(rmse, out / "tables" / "rmse_summary.md")
    if not health.empty:
        write_markdown_table(health, out / "tables" / "feature_health_summary.md")

    # Plots.
    plot_availability(availability, out / "plots")
    plot_smoothness_bars(smooth_common, out / "plots", suffix="common_valid")
    plot_smoothness_bars(smooth_per_method, out / "plots", suffix="per_method")
    plot_time_series_deltas(df, methods, out / "plots", skip_initial=args.skip_initial)
    plot_tracking_comparisons(df, methods, out / "plots")
    plot_rmse(df, methods, out / "plots")
    plot_feature_health(df, out / "plots")
    plot_sequence_artifacts(df, out / "plots")
    plot_operating_envelope_scatter(df, methods, out / "plots")

    # Text summary.
    summary = build_text_summary(
        csv_path,
        single_csv_path,
        merge_note,
        df,
        availability,
        smooth_common,
        smooth_per_method,
        rmse,
        health,
    )
    (out / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"\nWrote analysis to: {out}")


if __name__ == "__main__":
    main()
