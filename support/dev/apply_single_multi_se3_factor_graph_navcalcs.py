#!/usr/bin/env python3
"""
Fuse single-feature range/position measurements with multi-feature SE(3) pose
measurements using independent SE(3)-style factor-graph smoothers.

This script is intended for navcalcs CSVs that contain both:

  * single-feature target estimates, typically from a bounding-box / apparent-size
    range model:

        single_feature_estimated_tvec_x
        single_feature_estimated_tvec_y
        single_feature_estimated_tvec_z
        single_feature_estimated_range

  * multi-feature pose estimates, such as:

        pnp_*, qnp_*, wqnp_yolo_*, wqnp_kfest_*

For each requested multi-feature method, the script builds one fused graph using:

    state at frame k:
        pose_k = [t_x, t_y, t_z, phi_x, phi_y, phi_z]
        rate_k = [v_x, v_y, v_z, omega_x, omega_y, omega_z]

    factors:
        multi-feature pose_k ~= measured SE(3) pose_k       full 6-DOF factor
        single-feature t_k ~= measured position/range        3-DOF position factor
        pose_{k+1} ~= pose_k + rate_k * dt                  dynamics factor
        rate_{k+1} ~= rate_k                                rate smoothness factor

The single-feature measurement is intentionally anisotropic.  Its direction is
allowed to help the graph, but its radial/range axis can be devalued heavily:

    P_single = sigma_tangent^2 (I - u u^T) + sigma_range^2 u u^T

where u is the unit vector from camera to target.  Tune sigma_range_* larger than
sigma_tangent_* when the single-feature apparent-size range estimate should be
trusted much less than the image bearing.

By default this script does NOT overwrite the input CSV.  It writes:

    <input_stem>_sfmf_fg.csv
    <input_stem>_sfmf_fg_summary.csv
    <input_stem>_sfmf_fg_summary.txt

Example:

    python apply_single_multi_se3_factor_graph_navcalcs.py \
        --csv navcalcs_Cub_021325_1_50M.csv \
        --methods wqnp_kfest qnp \
        --single-sigma-range-base 8 \
        --single-sigma-range-frac 0.15 \
        --single-sigma-tangent-base 0.50 \
        --single-sigma-tangent-frac 0.015

For four files:

    python apply_single_multi_se3_factor_graph_navcalcs.py --csv file1.csv file2.csv file3.csv file4.csv
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "This script requires scipy for sparse factor-graph solves. "
        "Install scipy in the active environment and rerun."
    ) from exc


# -----------------------------------------------------------------------------
# Method definitions
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    valid_col: str
    q_cols: tuple[str, str, str, str]
    t_cols: tuple[str, str, str]
    cov_col: str | None = None
    rmse_col: str | None = None


METHODS: tuple[MethodSpec, ...] = (
    MethodSpec(
        key="pnp",
        label="OpenCV SolvePnP",
        valid_col="pnp_valid",
        q_cols=("pnp_qw", "pnp_qx", "pnp_qy", "pnp_qz"),
        t_cols=("pnp_tvec_x", "pnp_tvec_y", "pnp_tvec_z"),
        cov_col=None,
        rmse_col="pnp_rmse_px",
    ),
    MethodSpec(
        key="qnp",
        label="SolveQnP",
        valid_col="qnp_valid",
        q_cols=("qnp_qw", "qnp_qx", "qnp_qy", "qnp_qz"),
        t_cols=("qnp_tvec_x", "qnp_tvec_y", "qnp_tvec_z"),
        cov_col="qnp_cov6",
        rmse_col="qnp_rmse_px",
    ),
    MethodSpec(
        key="wqnp_yolo",
        label="wQnP, YOLO centers + KF covariance",
        valid_col="wqnp_yolo_valid",
        q_cols=("wqnp_yolo_qw", "wqnp_yolo_qx", "wqnp_yolo_qy", "wqnp_yolo_qz"),
        t_cols=("wqnp_yolo_tvec_x", "wqnp_yolo_tvec_y", "wqnp_yolo_tvec_z"),
        cov_col="wqnp_yolo_cov6",
        rmse_col="wqnp_yolo_rmse_px",
    ),
    MethodSpec(
        key="wqnp_kfest",
        label="wQnP, KF centers + KF covariance",
        valid_col="wqnp_kfest_valid",
        q_cols=("wqnp_kfest_qw", "wqnp_kfest_qx", "wqnp_kfest_qy", "wqnp_kfest_qz"),
        t_cols=("wqnp_kfest_tvec_x", "wqnp_kfest_tvec_y", "wqnp_kfest_tvec_z"),
        cov_col="wqnp_kfest_cov6",
        rmse_col=None,  # Important: do not px-gate using weighted chi residual.
    ),
)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class FusionConfig:
    # Dynamics / smoothness model.  These are process-noise-style sigmas in the
    # linear graph, not truth claims.  Increase process_sigma_vel to allow more
    # acceleration; decrease it for stronger constant-velocity smoothing.
    process_sigma_pos: float = 0.5
    process_sigma_rot: float = math.radians(0.50)
    process_sigma_vel: float = 2.0
    process_sigma_omega: float = math.radians(5.0)

    velocity_prior_sigma: float = 20.0
    omega_prior_sigma: float = math.radians(90.0)
    pose_prior_trans_sigma: float = 50.0
    pose_prior_rot_sigma: float = math.radians(90.0)

    # Multi-feature covariance model.  If a method has a cov6 column, it is used.
    # The variance multiplier and floors can tune trust in those measurements.
    multi_cov_alpha_trans: float = 1.0
    multi_cov_alpha_rot: float = 1.0
    multi_cov_floor_trans: float = 0.05
    multi_cov_floor_rot: float = math.radians(0.25)
    multi_cov_range_sigma_frac: float = 0.002

    # PnP / fallback covariance when no cov6 exists.
    pnp_sigma_pos_base: float = 0.35
    pnp_sigma_pos_range_frac: float = 0.003
    pnp_sigma_pos_rmse_scale: float = 0.006
    pnp_sigma_rot_deg_base: float = 0.75
    pnp_sigma_rot_deg_rmse_scale: float = 0.025

    # Single-feature anisotropic covariance.  The radial/range sigma should usually
    # be substantially larger than the tangent sigma.
    single_sigma_tangent_base: float = 0.75
    single_sigma_tangent_frac: float = 0.015
    single_sigma_range_base: float = 5.0
    single_sigma_range_frac: float = 0.05
    single_cov_alpha: float = 1.0

    # Single-feature gating.  Set max/min to NaN to disable a bound.
    single_range_min: float = 0.0
    single_range_max: float = math.nan
    single_mad_gate: float = 8.0
    single_mad_window: int = 11
    single_mad_min_scale: float = 5.0

    # Multi-feature gating.
    multi_rmse_gate_px: float = 50.0
    min_multi_feature_count: int = 0

    # Covariance regularization.
    min_trans_sigma: float = 0.003
    min_rot_sigma: float = 1e-4
    max_trans_sigma: float = 100.0
    max_rot_sigma: float = math.radians(90.0)
    cov_eig_floor: float = 1e-12
    hessian_damping: float = 1e-9
    covariance_mode: str = "block"


@dataclass
class FusionResult:
    method: MethodSpec
    x: np.ndarray
    H: sp.csc_matrix
    cov_pose_blocks: np.ndarray
    q0: np.ndarray
    valid_rows: np.ndarray
    multi_valid: np.ndarray
    single_valid: np.ndarray
    single_reject_reason: list[str]
    solve_info: dict[str, object]


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def have_cols(df: pd.DataFrame, cols: Iterable[str]) -> bool:
    return all(c in df.columns for c in cols)


def as_bool(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.fillna(False).to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(series):
        return (series.fillna(0).astype(float) != 0.0).to_numpy(dtype=bool)
    true_values = {"true", "t", "1", "yes", "y", "valid"}
    return series.fillna("false").astype(str).str.strip().str.lower().isin(true_values).to_numpy(dtype=bool)


def numeric(df: pd.DataFrame, cols: Iterable[str]) -> np.ndarray:
    return df.loc[:, list(cols)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def numeric_series(df: pd.DataFrame, col: str, default: float = np.nan) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), default, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def finite_rows(arr: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(arr), axis=1)


def vector_norm(arr: np.ndarray) -> np.ndarray:
    return np.linalg.norm(arr, axis=1)


def matrix_to_string(M: np.ndarray, precision: int = 8) -> str:
    M = np.asarray(M, dtype=float)
    return ";".join(",".join(f"{float(v):.{precision}g}" for v in row) for row in M)


def parse_cov6(value: object) -> np.ndarray | None:
    """Parse either 'a,b;...' 6x6 strings or flat/bracketed numeric strings."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None

    rows = [r.strip() for r in text.split(";") if r.strip()]
    if len(rows) == 6:
        parsed_rows = []
        ok = True
        for row in rows:
            vals = [v for v in re.split(r"[,\s]+", row.strip()) if v]
            if len(vals) != 6:
                ok = False
                break
            try:
                parsed_rows.append([float(v) for v in vals])
            except ValueError:
                ok = False
                break
        if ok:
            return np.asarray(parsed_rows, dtype=float)

    vals = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", text)
    if len(vals) == 36:
        return np.asarray([float(v) for v in vals], dtype=float).reshape(6, 6)
    if len(vals) == 6:
        return np.diag([float(v) for v in vals])
    return None


def nearest_positive_definite(
    C: np.ndarray,
    *,
    min_trans_sigma: float,
    min_rot_sigma: float,
    max_trans_sigma: float,
    max_rot_sigma: float,
    eig_floor: float,
) -> np.ndarray:
    C = np.asarray(C, dtype=float)
    if C.shape != (6, 6) or not np.all(np.isfinite(C)):
        raise ValueError("Covariance must be finite 6x6")

    C = 0.5 * (C + C.T)
    diag = np.diag(C).copy()
    mins = np.array([min_trans_sigma**2] * 3 + [min_rot_sigma**2] * 3, dtype=float)
    maxs = np.array([max_trans_sigma**2] * 3 + [max_rot_sigma**2] * 3, dtype=float)
    diag = np.clip(diag, mins, maxs)
    np.fill_diagonal(C, diag)

    s = np.sqrt(np.clip(np.diag(C), mins, maxs))
    S_inv = np.diag(1.0 / s)
    R = S_inv @ C @ S_inv
    R = np.nan_to_num(0.5 * (R + R.T), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(R, 1.0)

    w, V = np.linalg.eigh(R)
    w = np.clip(w, max(float(eig_floor), 1e-12), None)
    R_pd = (V * w) @ V.T
    R_pd = 0.5 * (R_pd + R_pd.T)
    S = np.diag(s)
    C_pd = S @ R_pd @ S
    C_pd = 0.5 * (C_pd + C_pd.T)

    jitter = max(float(eig_floor), 1e-12)
    for _ in range(8):
        try:
            np.linalg.cholesky(C_pd)
            return C_pd
        except np.linalg.LinAlgError:
            C_pd = C_pd + jitter * np.eye(6)
            jitter *= 10.0
    return np.diag(np.clip(np.diag(C_pd), mins, maxs))


def regularize_cov6(C: np.ndarray, cfg: FusionConfig) -> np.ndarray:
    return nearest_positive_definite(
        C,
        min_trans_sigma=cfg.min_trans_sigma,
        min_rot_sigma=cfg.min_rot_sigma,
        max_trans_sigma=cfg.max_trans_sigma,
        max_rot_sigma=cfg.max_rot_sigma,
        eig_floor=cfg.cov_eig_floor,
    )


def regularize_cov3(C: np.ndarray, cfg: FusionConfig) -> np.ndarray:
    C = np.asarray(C, dtype=float)
    C = 0.5 * (C + C.T)
    if C.shape != (3, 3) or not np.all(np.isfinite(C)):
        raise ValueError("Covariance must be finite 3x3")
    minv = cfg.min_trans_sigma**2
    maxv = cfg.max_trans_sigma**2
    diag = np.clip(np.diag(C), minv, maxv)
    np.fill_diagonal(C, diag)
    w, V = np.linalg.eigh(C)
    w = np.clip(w, max(cfg.cov_eig_floor, minv * 1e-6, 1e-15), maxv)
    C = (V * w) @ V.T
    C = 0.5 * (C + C.T)
    return C


def write_text_table(df: pd.DataFrame, path: Path) -> None:
    lines = [df.to_string(index=False)] if not df.empty else ["No rows."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# Quaternion helpers, scalar-first
# -----------------------------------------------------------------------------

def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.where(n <= 1e-15, np.nan, n)
    return q / n


def quat_conj(q: np.ndarray) -> np.ndarray:
    out = np.array(q, dtype=float, copy=True)
    out[..., 1:] *= -1.0
    return out


def quat_mul(q2: np.ndarray, q1: np.ndarray) -> np.ndarray:
    q2 = np.asarray(q2, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    return np.stack(
        [
            w2 * w1 - x2 * x1 - y2 * y1 - z2 * z1,
            w2 * x1 + x2 * w1 + y2 * z1 - z2 * y1,
            w2 * y1 - x2 * z1 + y2 * w1 + z2 * x1,
            w2 * z1 + x2 * y1 - y2 * x1 + z2 * w1,
        ],
        axis=-1,
    )


def quat_inv(q: np.ndarray) -> np.ndarray:
    return quat_conj(quat_normalize(q))


def quat_exp(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=float)
    theta = np.linalg.norm(phi, axis=-1, keepdims=True)
    half = 0.5 * theta
    small = theta < 1e-12
    axis_scale = np.where(small, 0.5 - theta**2 / 48.0, np.sin(half) / theta)
    w = np.where(small, 1.0 - theta**2 / 8.0, np.cos(half))
    v = axis_scale * phi
    return quat_normalize(np.concatenate([w, v], axis=-1))


def quat_log(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(np.asarray(q, dtype=float))
    if q.ndim == 1:
        q = q[None, :]
        squeeze = True
    else:
        squeeze = False

    q = q.copy()
    flip = q[:, 0] < 0.0
    q[flip] *= -1.0

    w = np.clip(q[:, 0], -1.0, 1.0)
    v = q[:, 1:]
    nv = np.linalg.norm(v, axis=1)
    theta = 2.0 * np.arctan2(nv, w)

    scale = np.zeros_like(nv)
    small = nv < 1e-12
    scale[~small] = theta[~small] / nv[~small]
    scale[small] = 2.0
    out = v * scale[:, None]
    return out[0] if squeeze else out


def make_quaternion_sequence_continuous(qs: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = qs.astype(float).copy()
    prev = None
    for i in range(len(out)):
        if not valid[i] or not np.all(np.isfinite(out[i])):
            continue
        out[i] = quat_normalize(out[i])
        if prev is not None and float(np.dot(out[i], prev)) < 0.0:
            out[i] *= -1.0
        prev = out[i].copy()
    return out


def tangent_to_quaternion(phi: np.ndarray, q0: np.ndarray) -> np.ndarray:
    return quat_normalize(quat_mul(quat_exp(phi), q0))


# -----------------------------------------------------------------------------
# Measurement preparation
# -----------------------------------------------------------------------------

def available_methods(df: pd.DataFrame) -> list[MethodSpec]:
    found: list[MethodSpec] = []
    for m in METHODS:
        if have_cols(df, [m.valid_col, *m.q_cols, *m.t_cols]):
            found.append(m)
    return found


def method_by_key(key: str) -> MethodSpec | None:
    for m in METHODS:
        if m.key == key:
            return m
    return None


def local_mad_filter(values: np.ndarray, valid: np.ndarray, window: int, gate: float, min_scale: float) -> tuple[np.ndarray, list[str]]:
    """Reject obvious local range outliers using a rolling median/MAD rule."""
    N = len(values)
    reasons = ["" for _ in range(N)]
    if not np.isfinite(gate) or gate <= 0.0 or window < 3:
        return valid.copy(), reasons

    window = int(window)
    if window % 2 == 0:
        window += 1
    half = window // 2
    out = valid.copy()
    for k in range(N):
        if not valid[k] or not np.isfinite(values[k]):
            continue
        lo = max(0, k - half)
        hi = min(N, k + half + 1)
        neigh = values[lo:hi][valid[lo:hi] & np.isfinite(values[lo:hi])]
        if neigh.size < max(5, window // 2):
            continue
        med = float(np.median(neigh))
        mad = float(np.median(np.abs(neigh - med)))
        robust_sigma = 1.4826 * mad
        scale = max(robust_sigma, float(min_scale))
        if abs(float(values[k]) - med) > float(gate) * scale:
            out[k] = False
            reasons[k] = "local_range_mad_gate"
    return out, reasons


def prepare_single_measurements(
    df: pd.DataFrame,
    *,
    t_cols: tuple[str, str, str],
    range_col: str,
    valid_col: str | None,
    cfg: FusionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if not have_cols(df, t_cols):
        raise ValueError(f"Missing single-feature tvec columns: {t_cols}")
    t = numeric(df, t_cols)
    rng = numeric_series(df, range_col, default=np.nan) if range_col in df.columns else vector_norm(t)

    valid = finite_rows(t) & np.isfinite(rng) & (vector_norm(t) > 1e-9)
    reasons = ["" for _ in range(len(df))]

    if valid_col is not None and valid_col in df.columns:
        valid &= as_bool(df[valid_col])

    if np.isfinite(cfg.single_range_min):
        bad = valid & (rng < cfg.single_range_min)
        for k in np.flatnonzero(bad):
            reasons[int(k)] = "single_range_min"
        valid &= ~bad
    if np.isfinite(cfg.single_range_max):
        bad = valid & (rng > cfg.single_range_max)
        for k in np.flatnonzero(bad):
            reasons[int(k)] = "single_range_max"
        valid &= ~bad

    valid_mad, mad_reasons = local_mad_filter(
        rng,
        valid,
        window=cfg.single_mad_window,
        gate=cfg.single_mad_gate,
        min_scale=cfg.single_mad_min_scale,
    )
    for k, reason in enumerate(mad_reasons):
        if reason and not reasons[k]:
            reasons[k] = reason
    valid = valid_mad

    return t, rng, valid, reasons


def prepare_multi_measurements(
    df: pd.DataFrame,
    method: MethodSpec,
    cfg: FusionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    q = numeric(df, method.q_cols)
    t = numeric(df, method.t_cols)
    valid = as_bool(df[method.valid_col]) & finite_rows(q) & finite_rows(t)
    reasons = ["" for _ in range(len(df))]

    if method.rmse_col is not None and method.rmse_col in df.columns:
        rmse = numeric_series(df, method.rmse_col)
        if np.isfinite(cfg.multi_rmse_gate_px) and cfg.multi_rmse_gate_px > 0.0:
            bad = valid & np.isfinite(rmse) & (rmse > cfg.multi_rmse_gate_px)
            for k in np.flatnonzero(bad):
                reasons[int(k)] = "multi_rmse_gate"
            valid &= ~bad

    if cfg.min_multi_feature_count > 0 and "pose_feature_count" in df.columns:
        nfeat = numeric_series(df, "pose_feature_count", default=0.0)
        bad = valid & np.isfinite(nfeat) & (nfeat < cfg.min_multi_feature_count)
        for k in np.flatnonzero(bad):
            reasons[int(k)] = "min_multi_feature_count"
        valid &= ~bad

    return t, q, valid, reasons


def choose_reference_quaternion(q_multi: np.ndarray, valid_multi: np.ndarray) -> np.ndarray:
    q_cont = make_quaternion_sequence_continuous(q_multi, valid_multi)
    if int(np.sum(valid_multi)) >= 1:
        return q_cont[int(np.flatnonzero(valid_multi)[0])].copy()
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


def build_pose_measurement_tangent(
    q_multi: np.ndarray,
    t_multi: np.ndarray,
    valid_multi: np.ndarray,
    q0: np.ndarray,
) -> np.ndarray:
    N = len(q_multi)
    z = np.full((N, 6), np.nan, dtype=float)
    if int(np.sum(valid_multi)) == 0:
        return z
    q_cont = make_quaternion_sequence_continuous(q_multi, valid_multi)
    q0_inv = quat_inv(q0)
    z[valid_multi, :3] = t_multi[valid_multi]
    z[valid_multi, 3:] = quat_log(quat_mul(q_cont[valid_multi], q0_inv))
    return z


def build_initial_pose_trajectory(
    times: np.ndarray,
    z_multi: np.ndarray,
    valid_multi: np.ndarray,
    single_t: np.ndarray,
    valid_single: np.ndarray,
) -> np.ndarray:
    N = len(times)
    z = np.full((N, 6), np.nan, dtype=float)

    # Prefer multi-feature translation, fill gaps with single-feature translation.
    z[valid_single, :3] = single_t[valid_single]
    z[valid_multi, :3] = z_multi[valid_multi, :3]

    # Only multi-feature measurements provide attitude.
    z[valid_multi, 3:] = z_multi[valid_multi, 3:]

    if not np.any(np.isfinite(z[:, :3])):
        z[:, :3] = 0.0
    if not np.any(np.isfinite(z[:, 3:])):
        z[:, 3:] = 0.0

    filled = pd.DataFrame(z, columns=[f"d{i}" for i in range(6)])
    filled["time"] = times
    filled = filled.set_index("time").sort_index()
    filled = filled.interpolate(method="index", limit_direction="both")
    z_filled = filled.to_numpy(dtype=float)
    z_filled = np.nan_to_num(z_filled, nan=0.0, posinf=0.0, neginf=0.0)
    return z_filled


def build_initial_rates(times: np.ndarray, z_filled: np.ndarray) -> np.ndarray:
    N = len(times)
    rates = np.zeros((N, 6), dtype=float)
    if N < 2:
        return rates
    for k in range(N):
        if k == 0:
            dt = max(float(times[1] - times[0]), 1e-9)
            rates[k] = (z_filled[1] - z_filled[0]) / dt
        elif k == N - 1:
            dt = max(float(times[-1] - times[-2]), 1e-9)
            rates[k] = (z_filled[-1] - z_filled[-2]) / dt
        else:
            dt = max(float(times[k + 1] - times[k - 1]), 1e-9)
            rates[k] = (z_filled[k + 1] - z_filled[k - 1]) / dt
    return rates


# -----------------------------------------------------------------------------
# Covariance models
# -----------------------------------------------------------------------------

def apply_alpha_floor_cov6(C: np.ndarray, z_pose: np.ndarray, method: MethodSpec, cfg: FusionConfig) -> np.ndarray:
    D = np.diag(
        [math.sqrt(max(cfg.multi_cov_alpha_trans, 0.0))] * 3
        + [math.sqrt(max(cfg.multi_cov_alpha_rot, 0.0))] * 3
    )
    rng = float(np.linalg.norm(z_pose[:3])) if np.all(np.isfinite(z_pose[:3])) else 0.0
    sig_t2 = cfg.multi_cov_floor_trans**2 + (cfg.multi_cov_range_sigma_frac * rng) ** 2
    sig_r2 = cfg.multi_cov_floor_rot**2
    C_eff = D @ C @ D.T + np.diag([sig_t2] * 3 + [sig_r2] * 3)
    return regularize_cov6(C_eff, cfg)


def nominal_multi_covariance(row: pd.Series, method: MethodSpec, z_pose: np.ndarray, cfg: FusionConfig) -> np.ndarray:
    if method.cov_col is not None and method.cov_col in row.index:
        C = parse_cov6(row[method.cov_col])
        if C is not None:
            return regularize_cov6(C, cfg)

    # Conservative fallback, mainly for PnP.
    rng = float(np.linalg.norm(z_pose[:3]))
    rmse = np.nan
    if method.rmse_col is not None and method.rmse_col in row.index:
        try:
            rmse = float(row[method.rmse_col])
        except Exception:
            rmse = np.nan
    if not np.isfinite(rmse):
        rmse = 10.0

    sig_t = cfg.pnp_sigma_pos_base + cfg.pnp_sigma_pos_range_frac * rng + cfg.pnp_sigma_pos_rmse_scale * rmse
    sig_r_deg = cfg.pnp_sigma_rot_deg_base + cfg.pnp_sigma_rot_deg_rmse_scale * rmse
    sig_t = float(np.clip(sig_t, cfg.min_trans_sigma, cfg.max_trans_sigma))
    sig_r = float(np.clip(np.deg2rad(sig_r_deg), cfg.min_rot_sigma, cfg.max_rot_sigma))
    return np.diag([sig_t**2] * 3 + [sig_r**2] * 3)


def multi_covariance(row: pd.Series, method: MethodSpec, z_pose: np.ndarray, cfg: FusionConfig) -> np.ndarray:
    C = nominal_multi_covariance(row, method, z_pose, cfg)
    return apply_alpha_floor_cov6(C, z_pose, method, cfg)


def single_position_covariance(t: np.ndarray, cfg: FusionConfig) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    rng = float(np.linalg.norm(t))
    if not np.isfinite(rng) or rng <= 1e-12:
        rng = 0.0
        u = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        u = t / rng

    sig_tan = cfg.single_sigma_tangent_base + cfg.single_sigma_tangent_frac * rng
    sig_rng = cfg.single_sigma_range_base + cfg.single_sigma_range_frac * rng
    sig_tan *= math.sqrt(max(cfg.single_cov_alpha, 0.0))
    sig_rng *= math.sqrt(max(cfg.single_cov_alpha, 0.0))

    I = np.eye(3, dtype=float)
    uu = np.outer(u, u)
    C = (sig_tan**2) * (I - uu) + (sig_rng**2) * uu
    return regularize_cov3(C, cfg)


# -----------------------------------------------------------------------------
# Factor-graph assembly
# -----------------------------------------------------------------------------

def pose_idx(k: int) -> np.ndarray:
    return np.arange(6 * k, 6 * k + 6, dtype=int)


def rate_idx(N: int, k: int) -> np.ndarray:
    off = 6 * N
    return np.arange(off + 6 * k, off + 6 * k + 6, dtype=int)


def add_block(H: sp.lil_matrix, rows: np.ndarray, cols: np.ndarray, block: np.ndarray) -> None:
    H[np.ix_(rows, cols)] = H[np.ix_(rows, cols)] + block


def add_factor(
    H: sp.lil_matrix,
    b: np.ndarray,
    idx_blocks: list[np.ndarray],
    A_blocks: list[np.ndarray],
    W: np.ndarray,
    z: np.ndarray | None = None,
) -> None:
    if z is None:
        z = np.zeros(W.shape[0], dtype=float)
    WA = [W @ A for A in A_blocks]
    Wz = W @ z
    for idx_i, A_i in zip(idx_blocks, A_blocks):
        b[idx_i] += A_i.T @ Wz
        for idx_j, A_j, WA_j in zip(idx_blocks, A_blocks, WA):
            add_block(H, idx_i, idx_j, A_i.T @ WA_j)

def precision_from_cov(C: np.ndarray, *, eig_floor: float = 1e-12) -> np.ndarray:
    """Return a symmetric precision matrix from a covariance, with PSD-safe flooring."""
    C = np.asarray(C, dtype=float)
    C = 0.5 * (C + C.T)

    try:
        return np.linalg.inv(C)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(C)
        finite_pos = w[np.isfinite(w) & (w > 0.0)]
        scale_floor = float(np.max(finite_pos) * 1e-12) if finite_pos.size else float(eig_floor)
        floor = max(float(eig_floor), scale_floor)

        w = np.clip(w, floor, None)
        W = (V * (1.0 / w)) @ V.T
        return 0.5 * (W + W.T)

def solve_fused_graph_for_method(
    df: pd.DataFrame,
    times: np.ndarray,
    method: MethodSpec,
    cfg: FusionConfig,
    *,
    single_t_cols: tuple[str, str, str],
    single_range_col: str,
    single_valid_col: str | None,
) -> FusionResult:
    single_t, single_range, valid_single, single_reasons = prepare_single_measurements(
        df,
        t_cols=single_t_cols,
        range_col=single_range_col,
        valid_col=single_valid_col,
        cfg=cfg,
    )
    multi_t, multi_q, valid_multi, multi_reasons = prepare_multi_measurements(df, method, cfg)

    if int(np.sum(valid_single)) < 2 and int(np.sum(valid_multi)) < 2:
        raise ValueError("Need at least two usable single or multi measurements.")
    if int(np.sum(valid_multi)) < 1:
        raise ValueError("Need at least one usable multi-feature pose to anchor attitude.")

    q0 = choose_reference_quaternion(multi_q, valid_multi)
    z_multi = build_pose_measurement_tangent(multi_q, multi_t, valid_multi, q0)
    z_filled = build_initial_pose_trajectory(times, z_multi, valid_multi, single_t, valid_single)
    rates0 = build_initial_rates(times, z_filled)

    N = len(df)
    n_state = 12 * N
    H = sp.lil_matrix((n_state, n_state), dtype=float)
    b = np.zeros(n_state, dtype=float)
    I6 = np.eye(6, dtype=float)
    A_single = np.hstack([np.eye(3), np.zeros((3, 3), dtype=float)])

    # Weak pose prior on the first state to keep all-single early segments anchored.
    C_pose_prior = np.diag([cfg.pose_prior_trans_sigma**2] * 3 + [cfg.pose_prior_rot_sigma**2] * 3)
    add_factor(H, b, [pose_idx(0)], [I6], np.linalg.inv(C_pose_prior), z_filled[0])

    # Multi-feature full SE(3) factors.
    for k in np.flatnonzero(valid_multi):
        kk = int(k)
        C = multi_covariance(df.iloc[kk], method, z_multi[kk], cfg)
        W = precision_from_cov(C, eig_floor=cfg.cov_eig_floor)
        add_factor(H, b, [pose_idx(kk)], [I6], W, z_multi[kk])

    # Single-feature position factors.  Radial/range uncertainty can be much larger.
    for k in np.flatnonzero(valid_single):
        kk = int(k)
        C = single_position_covariance(single_t[kk], cfg)
        W = np.linalg.inv(C)
        add_factor(H, b, [pose_idx(kk)], [A_single], W, single_t[kk])

    # Dynamics factors.
    for k in range(N - 1):
        dt = max(float(times[k + 1] - times[k]), 1e-9)
        sig_pose = np.array([cfg.process_sigma_pos] * 3 + [cfg.process_sigma_rot] * 3, dtype=float)
        C_pose = np.diag((sig_pose**2) * max(dt, 1e-6))
        add_factor(
            H,
            b,
            [pose_idx(k + 1), pose_idx(k), rate_idx(N, k)],
            [I6, -I6, -dt * I6],
            np.linalg.inv(C_pose),
            None,
        )

        sig_rate = np.array([cfg.process_sigma_vel] * 3 + [cfg.process_sigma_omega] * 3, dtype=float)
        C_rate = np.diag((sig_rate**2) * max(dt, 1e-6))
        add_factor(
            H,
            b,
            [rate_idx(N, k + 1), rate_idx(N, k)],
            [I6, -I6],
            np.linalg.inv(C_rate),
            None,
        )

    # Weak prior on initial rate.
    C_rate_prior = np.diag([cfg.velocity_prior_sigma**2] * 3 + [cfg.omega_prior_sigma**2] * 3)
    add_factor(H, b, [rate_idx(N, 0)], [I6], np.linalg.inv(C_rate_prior), rates0[0])

    if cfg.hessian_damping > 0.0:
        H.setdiag(H.diagonal() + float(cfg.hessian_damping))

    H_csc = H.tocsc()
    try:
        x = spla.spsolve(H_csc, b)
        solver_status = "spsolve"
    except Exception:
        x, *_ = spla.lsqr(H_csc, b)[:4]
        solver_status = "lsqr"

    cov_blocks = covariance_blocks_from_hessian(H_csc, N, cfg)
    valid_rows = valid_single | valid_multi

    info: dict[str, object] = {
        "solver_status": solver_status,
        "rows": int(N),
        "n_state": int(n_state),
        "single_valid_count": int(np.sum(valid_single)),
        "multi_valid_count": int(np.sum(valid_multi)),
        "single_raw_finite_count": int(np.sum(finite_rows(single_t) & np.isfinite(single_range))),
        "multi_raw_valid_count": int(np.sum(as_bool(df[method.valid_col]))),
        "single_sigma_tangent_base": cfg.single_sigma_tangent_base,
        "single_sigma_tangent_frac": cfg.single_sigma_tangent_frac,
        "single_sigma_range_base": cfg.single_sigma_range_base,
        "single_sigma_range_frac": cfg.single_sigma_range_frac,
        "single_cov_alpha": cfg.single_cov_alpha,
        "multi_cov_alpha_trans": cfg.multi_cov_alpha_trans,
        "multi_cov_alpha_rot": cfg.multi_cov_alpha_rot,
        "multi_cov_floor_trans": cfg.multi_cov_floor_trans,
        "multi_cov_floor_rot": cfg.multi_cov_floor_rot,
        "multi_cov_range_sigma_frac": cfg.multi_cov_range_sigma_frac,
        "multi_rmse_gate_px": cfg.multi_rmse_gate_px,
        "min_multi_feature_count": cfg.min_multi_feature_count,
        "process_sigma_pos": cfg.process_sigma_pos,
        "process_sigma_rot": cfg.process_sigma_rot,
        "process_sigma_vel": cfg.process_sigma_vel,
        "process_sigma_omega": cfg.process_sigma_omega,
        "covariance_mode": cfg.covariance_mode,
    }

    # Blend multi reasons into a combined single_reject_reason-like list for easy CSV export.
    reject_reason = []
    for s_reason, m_reason in zip(single_reasons, multi_reasons):
        if s_reason and m_reason:
            reject_reason.append(f"single:{s_reason};multi:{m_reason}")
        elif s_reason:
            reject_reason.append(f"single:{s_reason}")
        elif m_reason:
            reject_reason.append(f"multi:{m_reason}")
        else:
            reject_reason.append("")

    return FusionResult(
        method=method,
        x=np.asarray(x, dtype=float),
        H=H_csc,
        cov_pose_blocks=cov_blocks,
        q0=q0,
        valid_rows=valid_rows,
        multi_valid=valid_multi,
        single_valid=valid_single,
        single_reject_reason=reject_reason,
        solve_info=info,
    )


def covariance_blocks_from_hessian(H: sp.csc_matrix, N: int, cfg: FusionConfig) -> np.ndarray:
    blocks = np.zeros((N, 6, 6), dtype=float)
    if cfg.covariance_mode == "none":
        blocks[:] = np.nan
        return blocks

    if cfg.covariance_mode == "selected":
        try:
            lu = spla.splu(H)
            n_state = H.shape[0]
            for k in range(N):
                idx = pose_idx(k)
                E = np.zeros((n_state, 6), dtype=float)
                E[idx, np.arange(6)] = 1.0
                X = lu.solve(E)
                C = X[idx, :]
                blocks[k] = 0.5 * (C + C.T)
            return blocks
        except Exception as exc:
            print(f"Selected-inverse covariance failed ({exc}); falling back to block mode.")

    for k in range(N):
        idx = pose_idx(k)
        Hkk = H[np.ix_(idx, idx)].toarray()
        try:
            C = np.linalg.inv(Hkk)
        except np.linalg.LinAlgError:
            C = np.linalg.pinv(Hkk)
        blocks[k] = 0.5 * (C + C.T)
    return blocks


# -----------------------------------------------------------------------------
# Output integration
# -----------------------------------------------------------------------------

def append_fusion_columns(df: pd.DataFrame, result: FusionResult) -> pd.DataFrame:
    out = df.copy()
    N = len(out)
    poses = result.x[: 6 * N].reshape(N, 6)
    rates = result.x[6 * N : 12 * N].reshape(N, 6)
    t_s = poses[:, :3]
    phi_s = poses[:, 3:]
    q_s = tangent_to_quaternion(phi_s, result.q0)
    covs = result.cov_pose_blocks
    sig = np.sqrt(np.clip(np.diagonal(covs, axis1=1, axis2=2), 0.0, np.inf))

    prefix = f"{result.method.key}_sfmf_fg"
    out[f"{prefix}_valid"] = result.valid_rows
    out[f"{prefix}_single_factor_valid"] = result.single_valid
    out[f"{prefix}_multi_factor_valid"] = result.multi_valid
    out[f"{prefix}_reject_reason"] = result.single_reject_reason

    out[f"{prefix}_tvec_x"] = t_s[:, 0]
    out[f"{prefix}_tvec_y"] = t_s[:, 1]
    out[f"{prefix}_tvec_z"] = t_s[:, 2]
    out[f"{prefix}_range"] = np.linalg.norm(t_s, axis=1)

    out[f"{prefix}_qw"] = q_s[:, 0]
    out[f"{prefix}_qx"] = q_s[:, 1]
    out[f"{prefix}_qy"] = q_s[:, 2]
    out[f"{prefix}_qz"] = q_s[:, 3]

    out[f"{prefix}_vel_x"] = rates[:, 0]
    out[f"{prefix}_vel_y"] = rates[:, 1]
    out[f"{prefix}_vel_z"] = rates[:, 2]
    out[f"{prefix}_omega_x"] = rates[:, 3]
    out[f"{prefix}_omega_y"] = rates[:, 4]
    out[f"{prefix}_omega_z"] = rates[:, 5]

    names = ["x", "y", "z", "rx", "ry", "rz"]
    for i, name in enumerate(names):
        out[f"{prefix}_sigma_{name}"] = sig[:, i]
    out[f"{prefix}_cov6"] = [matrix_to_string(covs[k]) for k in range(N)]

    for key, value in result.solve_info.items():
        if isinstance(value, (str, int, float, bool, np.floating, np.integer, np.bool_)):
            out[f"{prefix}_{key}"] = value

    return out


def make_summary_row(csv_path: Path, result: FusionResult) -> dict[str, object]:
    row: dict[str, object] = {
        "csv": csv_path.name,
        "method": result.method.label,
        "key": result.method.key,
    }
    row.update(result.solve_info)
    return row


def make_plots(df: pd.DataFrame, csv_path: Path, methods: list[MethodSpec], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    if "image_time" in df.columns:
        x = pd.to_numeric(df["image_time"], errors="coerce").to_numpy(dtype=float)
        x = x - np.nanmin(x)
        xlabel = "Elapsed image time (s)"
    elif "frame_idx" in df.columns:
        x = pd.to_numeric(df["frame_idx"], errors="coerce").to_numpy(dtype=float)
        xlabel = "Frame index"
    else:
        x = np.arange(len(df), dtype=float)
        xlabel = "Row index"

    single_range = numeric_series(df, "single_feature_estimated_range")
    for m in methods:
        prefix = f"{m.key}_sfmf_fg"
        if f"{prefix}_range" not in df.columns:
            continue
        plt.figure(figsize=(11, 5))
        plt.plot(x, single_range, linewidth=0.8, alpha=0.65, label="single-feature range")
        if have_cols(df, m.t_cols):
            multi_range = vector_norm(numeric(df, m.t_cols))
            valid = as_bool(df[m.valid_col]) if m.valid_col in df.columns else np.ones(len(df), dtype=bool)
            plt.plot(x, np.where(valid, multi_range, np.nan), linewidth=0.9, alpha=0.8, label=f"{m.label} raw range")
        plt.plot(x, df[f"{prefix}_range"], linewidth=1.8, label=f"{m.label} fused SFMF FG")
        plt.xlabel(xlabel)
        plt.ylabel("Range / translation norm")
        plt.title(f"Single + multi-feature SE(3) fusion: {m.label}")
        plt.legend(fontsize="small")
        plt.tight_layout()
        plt.savefig(out_dir / f"{csv_path.stem}_{prefix}_range.png", dpi=180)
        plt.close()


# -----------------------------------------------------------------------------
# CLI and main
# -----------------------------------------------------------------------------

def choose_csvs_with_dialog() -> list[Path]:
    try:
        from tkinter import Tk, filedialog

        root = Tk()
        root.withdraw()
        filenames = filedialog.askopenfilenames(
            title="Select navcalcs CSV file(s)",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        root.destroy()
        return [Path(f) for f in filenames]
    except Exception:
        return []


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fuse single-feature and multi-feature navcalcs measurements with an SE(3) factor graph.")
    p.add_argument("--csv", type=Path, nargs="*", default=None, help="One or more input navcalcs CSVs. If omitted, a file dialog opens.")
    p.add_argument("--output", type=Path, default=None, help="Output CSV path. Only valid with one input CSV.")
    p.add_argument("--output-dir", type=Path, default=None, help="Directory for output CSVs. Defaults to each input file's directory.")
    p.add_argument("--suffix", default="_sfmf_fg", help="Suffix appended to output CSV stem when not overwriting. Default: _sfmf_fg.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite input CSVs instead of writing suffixed outputs.")
    p.add_argument("--no-backup", action="store_true", help="Do not create a timestamped backup when overwriting.")
    p.add_argument("--methods", nargs="*", default=["wqnp_kfest", "qnp", "wqnp_yolo", "pnp"], help="Multi-feature methods to fuse, or 'all'.")

    p.add_argument("--single-tvec-cols", nargs=3, default=["single_feature_estimated_tvec_x", "single_feature_estimated_tvec_y", "single_feature_estimated_tvec_z"], help="Single-feature tvec columns.")
    p.add_argument("--single-range-col", default="single_feature_estimated_range", help="Single-feature range column.")
    p.add_argument("--single-valid-col", default=None, help="Optional single-feature validity column. If absent, finite tvec/range defines validity.")

    p.add_argument("--process-sigma-pos", type=float, default=0.50, help="Pose dynamics position sigma [m].")
    p.add_argument("--process-sigma-rot-deg", type=float, default=0.50, help="Pose dynamics rotation sigma [deg].")
    p.add_argument("--process-sigma-vel", type=float, default=2.0, help="Rate random-walk translational sigma [m/s]. Larger values allow more acceleration.")
    p.add_argument("--process-sigma-omega-deg", type=float, default=5.0, help="Rate random-walk angular sigma [deg/s].")
    p.add_argument("--velocity-prior-sigma", type=float, default=20.0, help="Weak prior sigma for initial translational rate [m/s].")
    p.add_argument("--omega-prior-sigma-deg", type=float, default=90.0, help="Weak prior sigma for initial angular rate [deg/s].")
    p.add_argument("--pose-prior-trans-sigma", type=float, default=50.0, help="Weak prior sigma for first pose translation [m].")
    p.add_argument("--pose-prior-rot-deg", type=float, default=90.0, help="Weak prior sigma for first pose attitude [deg].")

    p.add_argument("--multi-cov-alpha-trans", type=float, default=1.0, help="Multi-feature translation covariance variance multiplier.")
    p.add_argument("--multi-cov-alpha-rot", type=float, default=1.0, help="Multi-feature rotation covariance variance multiplier.")
    p.add_argument("--multi-cov-floor-trans", type=float, default=0.05, help="Multi-feature translation sigma floor [m].")
    p.add_argument("--multi-cov-floor-rot-deg", type=float, default=0.25, help="Multi-feature rotation sigma floor [deg].")
    p.add_argument("--multi-cov-range-sigma-frac", type=float, default=0.002, help="Additional multi-feature translation sigma fraction times range.")
    p.add_argument("--multi-rmse-gate-px", type=float, default=50.0, help="Reject multi-feature factors with RMSE above this value. Set <=0 to disable.")
    p.add_argument("--min-multi-feature-count", type=int, default=0, help="Reject multi-feature factors below this pose_feature_count. Default disabled.")

    p.add_argument("--single-sigma-tangent-base", type=float, default=0.75, help="Single-feature tangent-plane sigma base [m].")
    p.add_argument("--single-sigma-tangent-frac", type=float, default=0.015, help="Single-feature tangent-plane sigma fraction times range.")
    p.add_argument("--single-sigma-range-base", type=float, default=5.0, help="Single-feature radial/range sigma base [m].")
    p.add_argument("--single-sigma-range-frac", type=float, default=0.05, help="Single-feature radial/range sigma fraction times range. Larger devalues range more.")
    p.add_argument("--single-cov-alpha", type=float, default=1.0, help="Single-feature covariance variance multiplier.")
    p.add_argument("--single-range-min", type=float, default=0.0, help="Minimum accepted single-feature range [m].")
    p.add_argument("--single-range-max", type=float, default=math.nan, help="Maximum accepted single-feature range [m]. NaN disables.")
    p.add_argument("--single-mad-gate", type=float, default=8.0, help="Local MAD gate multiplier for single-feature range outliers. <=0 disables.")
    p.add_argument("--single-mad-window", type=int, default=11, help="Rolling window for single-feature range MAD gate.")
    p.add_argument("--single-mad-min-scale", type=float, default=5.0, help="Minimum robust scale [m] used by MAD gate.")

    p.add_argument("--pnp-sigma-pos-base", type=float, default=0.35)
    p.add_argument("--pnp-sigma-pos-range-frac", type=float, default=0.003)
    p.add_argument("--pnp-sigma-pos-rmse-scale", type=float, default=0.006)
    p.add_argument("--pnp-sigma-rot-deg-base", type=float, default=0.75)
    p.add_argument("--pnp-sigma-rot-deg-rmse-scale", type=float, default=0.025)

    p.add_argument("--min-trans-sigma", type=float, default=0.003)
    p.add_argument("--min-rot-sigma", type=float, default=1e-4)
    p.add_argument("--max-trans-sigma", type=float, default=100.0)
    p.add_argument("--max-rot-deg", type=float, default=90.0)
    p.add_argument("--cov-eig-floor", type=float, default=1e-12)
    p.add_argument("--hessian-damping", type=float, default=1e-9)
    p.add_argument("--covariance-mode", choices=["block", "selected", "none"], default="block")

    p.add_argument("--make-plots", action="store_true", help="Write quick range overlay plots next to the output summary.")
    return p.parse_args()


def make_config(args: argparse.Namespace) -> FusionConfig:
    return FusionConfig(
        process_sigma_pos=float(args.process_sigma_pos),
        process_sigma_rot=math.radians(float(args.process_sigma_rot_deg)),
        process_sigma_vel=float(args.process_sigma_vel),
        process_sigma_omega=math.radians(float(args.process_sigma_omega_deg)),
        velocity_prior_sigma=float(args.velocity_prior_sigma),
        omega_prior_sigma=math.radians(float(args.omega_prior_sigma_deg)),
        pose_prior_trans_sigma=float(args.pose_prior_trans_sigma),
        pose_prior_rot_sigma=math.radians(float(args.pose_prior_rot_deg)),
        multi_cov_alpha_trans=float(args.multi_cov_alpha_trans),
        multi_cov_alpha_rot=float(args.multi_cov_alpha_rot),
        multi_cov_floor_trans=float(args.multi_cov_floor_trans),
        multi_cov_floor_rot=math.radians(float(args.multi_cov_floor_rot_deg)),
        multi_cov_range_sigma_frac=float(args.multi_cov_range_sigma_frac),
        pnp_sigma_pos_base=float(args.pnp_sigma_pos_base),
        pnp_sigma_pos_range_frac=float(args.pnp_sigma_pos_range_frac),
        pnp_sigma_pos_rmse_scale=float(args.pnp_sigma_pos_rmse_scale),
        pnp_sigma_rot_deg_base=float(args.pnp_sigma_rot_deg_base),
        pnp_sigma_rot_deg_rmse_scale=float(args.pnp_sigma_rot_deg_rmse_scale),
        single_sigma_tangent_base=float(args.single_sigma_tangent_base),
        single_sigma_tangent_frac=float(args.single_sigma_tangent_frac),
        single_sigma_range_base=float(args.single_sigma_range_base),
        single_sigma_range_frac=float(args.single_sigma_range_frac),
        single_cov_alpha=float(args.single_cov_alpha),
        single_range_min=float(args.single_range_min),
        single_range_max=float(args.single_range_max),
        single_mad_gate=float(args.single_mad_gate),
        single_mad_window=int(args.single_mad_window),
        single_mad_min_scale=float(args.single_mad_min_scale),
        multi_rmse_gate_px=float(args.multi_rmse_gate_px),
        min_multi_feature_count=int(args.min_multi_feature_count),
        min_trans_sigma=float(args.min_trans_sigma),
        min_rot_sigma=float(args.min_rot_sigma),
        max_trans_sigma=float(args.max_trans_sigma),
        max_rot_sigma=math.radians(float(args.max_rot_deg)),
        cov_eig_floor=float(args.cov_eig_floor),
        hessian_damping=float(args.hessian_damping),
        covariance_mode=str(args.covariance_mode),
    )


def output_path_for(csv_path: Path, args: argparse.Namespace) -> Path:
    if args.output is not None:
        return Path(args.output)
    if args.overwrite:
        return csv_path
    out_dir = Path(args.output_dir) if args.output_dir is not None else csv_path.parent
    return out_dir / f"{csv_path.stem}{args.suffix}{csv_path.suffix}"


def process_one_csv(csv_path: Path, args: argparse.Namespace, cfg: FusionConfig) -> tuple[Path, list[dict[str, object]]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "image_time" in df.columns:
        times_abs = pd.to_numeric(df["image_time"], errors="coerce").to_numpy(dtype=float)
    elif "frame_idx" in df.columns:
        times_abs = pd.to_numeric(df["frame_idx"], errors="coerce").to_numpy(dtype=float)
    else:
        times_abs = np.arange(len(df), dtype=float)
    if not np.all(np.isfinite(times_abs)):
        bad = ~np.isfinite(times_abs)
        times_abs[bad] = np.arange(len(df), dtype=float)[bad]
    times = times_abs - times_abs[0]

    requested_keys = {m.key for m in METHODS} if any(str(x).lower() == "all" for x in args.methods) else set(args.methods)
    methods = [m for m in available_methods(df) if m.key in requested_keys]
    if not methods:
        raise RuntimeError(f"No requested multi-feature methods found in {csv_path.name}.")

    out_df = df.copy()
    summary_rows: list[dict[str, object]] = []
    print(f"\nInput CSV: {csv_path}")
    print(f"Rows: {len(df)}")
    print(f"Fusing methods: {', '.join(m.key for m in methods)}")

    for method in methods:
        print(f"  === {method.label} ({method.key}) ===")
        try:
            result = solve_fused_graph_for_method(
                out_df,
                times,
                method,
                cfg,
                single_t_cols=tuple(args.single_tvec_cols),
                single_range_col=str(args.single_range_col),
                single_valid_col=args.single_valid_col,
            )
        except Exception as exc:
            print(f"    Skipping {method.key}: {exc}")
            continue
        out_df = append_fusion_columns(out_df, result)
        summary_rows.append(make_summary_row(csv_path, result))
        print(
            f"    single factors: {result.solve_info['single_valid_count']}; "
            f"multi factors: {result.solve_info['multi_valid_count']}; "
            f"solver={result.solve_info['solver_status']}"
        )

    if not summary_rows:
        raise RuntimeError(f"No fused graph outputs were generated for {csv_path.name}.")

    out_path = output_path_for(csv_path, args)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.resolve() == csv_path.resolve() and not args.no_backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = csv_path.with_name(f"{csv_path.stem}_before_sfmf_fg_{stamp}{csv_path.suffix}")
        shutil.copy2(csv_path, backup)
        print(f"  Backup written: {backup}")

    out_df.to_csv(out_path, index=False)
    summary = pd.DataFrame(summary_rows)
    summary_base = out_path.with_name(f"{out_path.stem}_summary")
    summary.to_csv(summary_base.with_suffix(".csv"), index=False)
    write_text_table(summary, summary_base.with_suffix(".txt"))

    if args.make_plots:
        make_plots(out_df, csv_path, methods, out_path.with_name(f"{out_path.stem}_plots"))

    print(f"  Updated CSV written: {out_path}")
    print(f"  Summary written: {summary_base.with_suffix('.txt')}")
    return out_path, summary_rows


def main() -> None:
    args = parse_args()
    csvs = args.csv if args.csv else choose_csvs_with_dialog()
    if not csvs:
        raise SystemExit("No CSV selected.")
    if args.output is not None and len(csvs) != 1:
        raise SystemExit("--output can only be used with exactly one input CSV. Use --output-dir for multiple files.")

    cfg = make_config(args)
    all_rows: list[dict[str, object]] = []
    outputs: list[Path] = []
    for csv_path in csvs:
        out_path, rows = process_one_csv(Path(csv_path), args, cfg)
        outputs.append(out_path)
        all_rows.extend(rows)

    if len(outputs) > 1:
        combined_dir = Path(args.output_dir) if args.output_dir is not None else outputs[0].parent
        combined = pd.DataFrame(all_rows)
        combined_path = combined_dir / f"sfmf_fg_combined_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\nCombined summary written: {combined_path}")

    print("\nNew fused columns use the pattern <method>_sfmf_fg_*; e.g., wqnp_kfest_sfmf_fg_tvec_x.")
    print("Single-feature range trust is mainly controlled by --single-sigma-range-base and --single-sigma-range-frac.")


if __name__ == "__main__":
    main()
