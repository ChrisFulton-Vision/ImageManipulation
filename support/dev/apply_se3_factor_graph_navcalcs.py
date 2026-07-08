#!/usr/bin/env python3
"""
Apply independent SE(3)-style factor-graph smoothers to navcalcs CSV files.

This script is designed for dissertation flight-test CSVs containing pose estimates
from:

    OpenCV SolvePnP
    SolveQnP
    weighted SolveQnP using YOLO centers + KF covariance
    weighted SolveQnP using KF centers + KF covariance

For each method, it builds a separate multi-state factor graph in local SE(3)
coordinates.  The graph uses:

    state at frame k:
        pose_k = [t_x, t_y, t_z, phi_x, phi_y, phi_z]
        rate_k = [v_x, v_y, v_z, omega_x, omega_y, omega_z]

    measurement factors:
        pose_k ~= measured pose_k

    dynamics factors:
        pose_{k+1} ~= pose_k + rate_k * dt
        rate_{k+1} ~= rate_k

The position is represented directly.  The attitude is represented in a continuous
local tangent coordinate built from the measured quaternion sequence; the smoothed
attitude is then mapped back to scalar-first quaternions.

The QnP / wQnP measurement factors use their CSV covariance columns when present:
    qnp_cov6
    wqnp_yolo_cov6
    wqnp_kfest_cov6

OpenCV SolvePnP does not usually have an exported pose covariance, so it uses a
conservative heuristic covariance based on range and reprojection RMSE.

An empirical covariance-inflation model is also available.  The reported pose
covariance can be inflated as

    C_eff = D C D + diag(s_floor_t^2, s_floor_t^2, s_floor_t^2,
                         s_floor_r^2, s_floor_r^2, s_floor_r^2)

where D applies alpha-style variance scaling and

    s_floor_t^2 = floor_trans^2 + (range_sigma_frac * ||t||)^2 .

The range term intentionally grows uncertainty for more distant camera-to-object
solutions.  This inflation is meant to represent an effective truth-calibrated
uncertainty model, not a physical decomposition of individual error sources.

The script appends columns such as:
    pnp_fg_tvec_x, pnp_fg_tvec_y, pnp_fg_tvec_z
    pnp_fg_qw, pnp_fg_qx, pnp_fg_qy, pnp_fg_qz
    pnp_fg_cov6
    pnp_fg_sigma_x, ..., pnp_fg_sigma_rz
    pnp_fg_vel_x, ..., pnp_fg_omega_z

and analogous columns for qnp, wqnp_yolo, and wqnp_kfest.

By default, the original CSV is backed up and overwritten in place.

Example:
    python apply_se3_factor_graph_navcalcs.py --csv navcalcs_probe.csv

No arguments:
    Opens a file-selection dialog.
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
        rmse_col="wqnp_kfest_rmse_px",
    ),
)


# -----------------------------------------------------------------------------
# Small utilities
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


def finite_rows(arr: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(arr), axis=1)


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
    """Symmetrize and regularize a 6x6 covariance matrix."""
    C = np.asarray(C, dtype=float)
    if C.shape != (6, 6) or not np.all(np.isfinite(C)):
        raise ValueError("Covariance must be finite 6x6")

    C = 0.5 * (C + C.T)

    # Clip diagonal variances first, preserving off-diagonals as much as possible.
    diag = np.diag(C).copy()
    mins = np.array([min_trans_sigma ** 2] * 3 + [min_rot_sigma ** 2] * 3, dtype=float)
    maxs = np.array([max_trans_sigma ** 2] * 3 + [max_rot_sigma ** 2] * 3, dtype=float)
    diag = np.clip(diag, mins, maxs)
    np.fill_diagonal(C, diag)

    # Convert to a correlation-like matrix, clip eigs, then restore scale.  This
    # avoids one monster off-diagonal ruining the whole little covariance picnic.
    s = np.sqrt(np.clip(np.diag(C), mins, maxs))
    S = np.diag(s)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.linalg.solve(S, C) @ np.linalg.inv(S)
    R = np.nan_to_num(0.5 * (R + R.T), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(R, 1.0)

    w, V = np.linalg.eigh(R)
    w = np.clip(w, max(float(eig_floor), 1e-12), None)
    R_pd = (V * w) @ V.T
    R_pd = 0.5 * (R_pd + R_pd.T)
    C_pd = S @ R_pd @ S
    C_pd = 0.5 * (C_pd + C_pd.T)

    # Last tiny jitter to guarantee Cholesky.
    jitter = max(float(eig_floor), 1e-12)
    for _ in range(8):
        try:
            np.linalg.cholesky(C_pd)
            return C_pd
        except np.linalg.LinAlgError:
            C_pd = C_pd + jitter * np.eye(6)
            jitter *= 10.0

    # Last-resort diagonal covariance.
    return np.diag(np.clip(np.diag(C_pd), mins, maxs))


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
    """Hamilton product q = q2 * q1, scalar-first."""
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
    """SO(3) exponential map as scalar-first quaternion."""
    phi = np.asarray(phi, dtype=float)
    theta = np.linalg.norm(phi, axis=-1, keepdims=True)
    half = 0.5 * theta
    small = theta < 1e-12
    axis_scale = np.where(small, 0.5 - theta ** 2 / 48.0, np.sin(half) / theta)
    w = np.where(small, 1.0 - theta ** 2 / 8.0, np.cos(half))
    v = axis_scale * phi
    return quat_normalize(np.concatenate([w, v], axis=-1))


def quat_log(q: np.ndarray) -> np.ndarray:
    """SO(3) logarithm map from scalar-first quaternion to rotation vector."""
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


def pose_measurements_to_tangent(
    times: np.ndarray,
    t_meas: np.ndarray,
    q_meas: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return z_pose (N,6), q0, valid, and filled initial pose trajectory."""
    N = len(times)
    valid = valid & finite_rows(t_meas) & finite_rows(q_meas)
    if int(np.sum(valid)) < 2:
        raise ValueError("Need at least two valid measurements for smoothing.")

    q_cont = make_quaternion_sequence_continuous(q_meas, valid)
    first_idx = int(np.flatnonzero(valid)[0])
    q0 = q_cont[first_idx].copy()
    q0_inv = quat_inv(q0)

    phi = np.full((N, 3), np.nan, dtype=float)
    phi[valid] = quat_log(quat_mul(q_cont[valid], q0_inv))

    z = np.full((N, 6), np.nan, dtype=float)
    z[:, :3] = t_meas
    z[:, 3:] = phi

    # Fill an initial pose trajectory for every row by interpolation.  Measurement
    # factors will still only be applied where valid==True.
    filled = pd.DataFrame(z, columns=[f"d{i}" for i in range(6)])
    filled["time"] = times
    filled = filled.set_index("time").sort_index()
    filled = filled.interpolate(method="index", limit_direction="both")
    z_filled = filled.to_numpy(dtype=float)

    return z, q0, valid, z_filled


def tangent_to_quaternion(phi: np.ndarray, q0: np.ndarray) -> np.ndarray:
    return quat_normalize(quat_mul(quat_exp(phi), q0))


# -----------------------------------------------------------------------------
# Factor-graph normal-equation assembly
# -----------------------------------------------------------------------------

@dataclass
class SmootherConfig:
    process_sigma_pos: float = 0.08
    process_sigma_rot: float = math.radians(0.10)
    process_sigma_vel: float = 0.02
    process_sigma_omega: float = math.radians(0.75)

    velocity_prior_sigma: float = 5.0
    omega_prior_sigma: float = math.radians(5.0)

    pnp_sigma_pos_base: float = 0.35
    pnp_sigma_pos_range_frac: float = 0.003
    pnp_sigma_pos_rmse_scale: float = 0.006
    pnp_sigma_rot_deg_base: float = 0.75
    pnp_sigma_rot_deg_rmse_scale: float = 0.025

    # Empirical covariance inflation.  The alpha terms are variance multipliers.
    # A value of 9.0 means the corresponding 1-sigma width triples.
    #
    # Translation floor is added in variance form as
    #     sigma_floor_t(r)^2 = floor_trans^2 + (range_sigma_frac * ||t||)^2
    # so distant pose estimates naturally receive larger range uncertainty.
    #
    # raw_* tunes exported <method>_cov6_effective columns for the original pose
    # methods. fg_* tunes exported <method>_fg_cov6 uncertainty columns.
    #
    # meas_* tunes the actual graph measurement weights.  This is intentionally
    # separate from fg_* because the FG covariance envelope may need to be broad
    # for dissertation coverage plots, while the smoother itself usually only
    # needs a moderate de-gospeling of QnP Hessian covariances.
    raw_cov_alpha_trans: float = 16.0
    raw_cov_alpha_rot: float = 4.0
    raw_cov_floor_trans: float = 0.22
    raw_cov_floor_rot: float = math.radians(0.35)
    raw_cov_range_sigma_frac: float = 0.0025

    fg_cov_alpha_trans: float = 1.5
    fg_cov_alpha_rot: float = 2.0
    fg_cov_floor_trans: float = 0.25
    fg_cov_floor_rot: float = math.radians(0.25)
    fg_cov_range_sigma_frac: float = 0.0015

    # Measurement-weight inflation is applied by default only to methods that
    # supply solver covariance columns, i.e., the QnP / weighted-QnP family.
    # OpenCV SolvePnP already uses the conservative heuristic covariance below.
    meas_cov_alpha_trans: float = 9.0
    meas_cov_alpha_rot: float = 2.0
    meas_cov_floor_trans: float = 0.30
    meas_cov_floor_rot: float = math.radians(0.25)
    meas_cov_range_sigma_frac: float = 0.0025

    inflate_measurement_covariances: bool = True
    inflate_pnp_measurement_covariance: bool = False
    inflate_reported_covariances: bool = True
    write_inflated_raw_covariances: bool = True

    min_trans_sigma: float = 0.003
    min_rot_sigma: float = 1e-4
    max_trans_sigma: float = 30.0
    max_rot_sigma: float = math.radians(60.0)
    cov_eig_floor: float = 1e-12

    hessian_damping: float = 1e-9
    covariance_mode: str = "block"


@dataclass
class SmootherResult:
    x: np.ndarray
    H: sp.csc_matrix
    cov_pose_blocks: np.ndarray
    valid: np.ndarray
    q0: np.ndarray
    measurement_valid_count: int
    solve_info: dict[str, float | int | str]


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
    """Add factor ||A x - z||_W^2 to normal equations H x = b."""
    if z is None:
        z = np.zeros(W.shape[0], dtype=float)

    WA = [W @ A for A in A_blocks]
    Wz = W @ z
    for i, (idx_i, A_i) in enumerate(zip(idx_blocks, A_blocks)):
        b[idx_i] += A_i.T @ Wz
        for idx_j, A_j, WA_j in zip(idx_blocks, A_blocks, WA):
            add_block(H, idx_i, idx_j, A_i.T @ WA_j)


def regularize_covariance_for_config(C: np.ndarray, cfg: SmootherConfig) -> np.ndarray:
    return nearest_positive_definite(
        C,
        min_trans_sigma=cfg.min_trans_sigma,
        min_rot_sigma=cfg.min_rot_sigma,
        max_trans_sigma=cfg.max_trans_sigma,
        max_rot_sigma=cfg.max_rot_sigma,
        eig_floor=cfg.cov_eig_floor,
    )


def covariance_inflation_matrix(alpha_trans: float, alpha_rot: float) -> np.ndarray:
    """Return diagonal square-root variance scaling for alpha inflation."""
    alpha_t = max(float(alpha_trans), 0.0)
    alpha_r = max(float(alpha_rot), 0.0)
    return np.diag([math.sqrt(alpha_t)] * 3 + [math.sqrt(alpha_r)] * 3)


def covariance_floor_matrix(
    z_pose: np.ndarray,
    floor_trans: float,
    floor_rot: float,
    range_sigma_frac: float,
) -> np.ndarray:
    """Return additive covariance floor for one 6-DOF pose measurement/state.

    The translation floor increases with range.  This is intentionally isotropic
    in translation coordinates so downstream range-sigma diagnostics that only
    read covariance diagonals still see the added uncertainty.
    """
    z_pose = np.asarray(z_pose, dtype=float)
    rng = float(np.linalg.norm(z_pose[:3])) if z_pose.size >= 3 else 0.0
    if not np.isfinite(rng):
        rng = 0.0

    sig_t2 = float(floor_trans) ** 2 + (float(range_sigma_frac) * rng) ** 2
    sig_r2 = float(floor_rot) ** 2
    sig_t2 = max(sig_t2, 0.0)
    sig_r2 = max(sig_r2, 0.0)
    return np.diag([sig_t2] * 3 + [sig_r2] * 3)


def inflate_pose_covariance(
    C: np.ndarray,
    z_pose: np.ndarray,
    cfg: SmootherConfig,
    *,
    alpha_trans: float,
    alpha_rot: float,
    floor_trans: float,
    floor_rot: float,
    range_sigma_frac: float,
) -> np.ndarray:
    """Apply alpha + additive floor empirical covariance inflation."""
    C = np.asarray(C, dtype=float)
    D = covariance_inflation_matrix(alpha_trans, alpha_rot)
    C_eff = D @ C @ D.T + covariance_floor_matrix(z_pose, floor_trans, floor_rot, range_sigma_frac)
    return regularize_covariance_for_config(C_eff, cfg)


def nominal_measurement_covariance_for_row(
    row: pd.Series,
    method: MethodSpec,
    z_pose: np.ndarray,
    cfg: SmootherConfig,
) -> np.ndarray:
    if method.cov_col is not None and method.cov_col in row.index:
        C = parse_cov6(row[method.cov_col])
        if C is not None:
            return regularize_covariance_for_config(C, cfg)

    # Conservative PnP / fallback covariance.
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
    return np.diag([sig_t ** 2, sig_t ** 2, sig_t ** 2, sig_r ** 2, sig_r ** 2, sig_r ** 2])


def measurement_covariance_for_row(
    row: pd.Series,
    method: MethodSpec,
    z_pose: np.ndarray,
    cfg: SmootherConfig,
) -> np.ndarray:
    C = nominal_measurement_covariance_for_row(row, method, z_pose, cfg)

    # The QnP-family covariance columns are often Hessian/local-fit covariances.
    # They can be much too confident as truth-level measurement weights, which
    # pins the FG state to the raw estimate and makes the smoother visually do
    # nothing.  PnP has no exported covariance and already uses the conservative
    # heuristic fallback, so leave PnP alone unless explicitly requested.
    should_inflate = bool(cfg.inflate_measurement_covariances)
    if method.cov_col is None and not bool(cfg.inflate_pnp_measurement_covariance):
        should_inflate = False

    if should_inflate:
        C = inflate_pose_covariance(
            C,
            z_pose,
            cfg,
            alpha_trans=cfg.meas_cov_alpha_trans,
            alpha_rot=cfg.meas_cov_alpha_rot,
            floor_trans=cfg.meas_cov_floor_trans,
            floor_rot=cfg.meas_cov_floor_rot,
            range_sigma_frac=cfg.meas_cov_range_sigma_frac,
        )
    return C


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


def solve_factor_graph_for_method(
    df: pd.DataFrame,
    times: np.ndarray,
    method: MethodSpec,
    cfg: SmootherConfig,
) -> SmootherResult:
    q_raw = numeric(df, method.q_cols)
    t_raw = numeric(df, method.t_cols)
    valid = as_bool(df[method.valid_col]) & finite_rows(q_raw) & finite_rows(t_raw)

    z_pose, q0, valid, z_filled = pose_measurements_to_tangent(times, t_raw, q_raw, valid)
    rates0 = build_initial_rates(times, z_filled)

    N = len(times)
    n_state = 12 * N
    H = sp.lil_matrix((n_state, n_state), dtype=float)
    b = np.zeros(n_state, dtype=float)

    I6 = np.eye(6, dtype=float)

    # Measurement factors.
    for k in np.flatnonzero(valid):
        C = measurement_covariance_for_row(df.iloc[int(k)], method, z_pose[int(k)], cfg)
        W = np.linalg.inv(C)
        add_factor(H, b, [pose_idx(int(k))], [I6], W, z_pose[int(k)])

    # Dynamics factors: pose_{k+1} - pose_k - rate_k dt = 0.
    for k in range(N - 1):
        dt = max(float(times[k + 1] - times[k]), 1e-9)

        # Pose propagation residual.
        sig_pose = np.array(
            [cfg.process_sigma_pos] * 3 + [cfg.process_sigma_rot] * 3,
            dtype=float,
        )
        # Gentle dt scaling: larger dt allows larger pose model mismatch.
        C_pose = np.diag((sig_pose ** 2) * max(dt, 1e-6))
        W_pose = np.linalg.inv(C_pose)

        add_factor(
            H,
            b,
            [pose_idx(k + 1), pose_idx(k), rate_idx(N, k)],
            [I6, -I6, -dt * I6],
            W_pose,
            None,
        )

        # Rate random-walk residual.
        sig_rate = np.array(
            [cfg.process_sigma_vel] * 3 + [cfg.process_sigma_omega] * 3,
            dtype=float,
        )
        C_rate = np.diag((sig_rate ** 2) * max(dt, 1e-6))
        W_rate = np.linalg.inv(C_rate)
        add_factor(
            H,
            b,
            [rate_idx(N, k + 1), rate_idx(N, k)],
            [I6, -I6],
            W_rate,
            None,
        )

    # Weak prior on first rate keeps the endpoint goblin from juggling the nullspace.
    C_prior = np.diag([cfg.velocity_prior_sigma ** 2] * 3 + [cfg.omega_prior_sigma ** 2] * 3)
    W_prior = np.linalg.inv(C_prior)
    add_factor(H, b, [rate_idx(N, 0)], [I6], W_prior, rates0[0])

    # Tiny damping for numerical stability.
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
    if cfg.inflate_reported_covariances:
        pose_states = np.asarray(x, dtype=float)[: 6 * N].reshape(N, 6)
        cov_blocks = inflate_covariance_blocks(cov_blocks, pose_states, cfg)

    info = {
        "solver_status": solver_status,
        "n_rows": int(N),
        "n_state": int(n_state),
        "measurement_valid_count": int(np.sum(valid)),
        "covariance_mode": cfg.covariance_mode,
        "cov_alpha_trans": float(cfg.fg_cov_alpha_trans),
        "cov_alpha_rot": float(cfg.fg_cov_alpha_rot),
        "cov_floor_trans": float(cfg.fg_cov_floor_trans),
        "cov_floor_rot": float(cfg.fg_cov_floor_rot),
        "cov_range_sigma_frac": float(cfg.fg_cov_range_sigma_frac),
        "meas_cov_alpha_trans": float(cfg.meas_cov_alpha_trans),
        "meas_cov_alpha_rot": float(cfg.meas_cov_alpha_rot),
        "meas_cov_floor_trans": float(cfg.meas_cov_floor_trans),
        "meas_cov_floor_rot": float(cfg.meas_cov_floor_rot),
        "meas_cov_range_sigma_frac": float(cfg.meas_cov_range_sigma_frac),
        "inflate_measurement_covariances": bool(cfg.inflate_measurement_covariances),
        "inflate_pnp_measurement_covariance": bool(cfg.inflate_pnp_measurement_covariance),
        "inflate_reported_covariances": bool(cfg.inflate_reported_covariances),
    }

    return SmootherResult(
        x=np.asarray(x, dtype=float),
        H=H_csc,
        cov_pose_blocks=cov_blocks,
        valid=np.ones(N, dtype=bool),
        q0=q0,
        measurement_valid_count=int(np.sum(valid)),
        solve_info=info,
    )


def inflate_covariance_blocks(covs: np.ndarray, pose_states: np.ndarray, cfg: SmootherConfig) -> np.ndarray:
    """Apply empirical inflation to a stack of pose covariance blocks."""
    covs = np.asarray(covs, dtype=float)
    pose_states = np.asarray(pose_states, dtype=float)
    out = np.array(covs, dtype=float, copy=True)
    for k in range(len(out)):
        if out[k].shape != (6, 6) or not np.all(np.isfinite(out[k])):
            continue
        z_pose = pose_states[k] if k < len(pose_states) else np.zeros(6, dtype=float)
        out[k] = inflate_pose_covariance(
            out[k],
            z_pose,
            cfg,
            alpha_trans=cfg.fg_cov_alpha_trans,
            alpha_rot=cfg.fg_cov_alpha_rot,
            floor_trans=cfg.fg_cov_floor_trans,
            floor_rot=cfg.fg_cov_floor_rot,
            range_sigma_frac=cfg.fg_cov_range_sigma_frac,
        )
    return out


def covariance_blocks_from_hessian(H: sp.csc_matrix, N: int, cfg: SmootherConfig) -> np.ndarray:
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

    # Default: local Hessian block inverse. This is fast and useful for diagnostics,
    # but it does not include all off-diagonal graph correlations.
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
# CSV integration
# -----------------------------------------------------------------------------

def available_methods(df: pd.DataFrame) -> list[MethodSpec]:
    found: list[MethodSpec] = []
    for m in METHODS:
        required = [m.valid_col, *m.q_cols, *m.t_cols]
        if have_cols(df, required):
            found.append(m)
    return found


def append_inflated_raw_covariance_columns(
    df: pd.DataFrame,
    methods: list[MethodSpec],
    cfg: SmootherConfig,
) -> pd.DataFrame:
    """Append <method>_cov6_effective and sigma columns for raw pose methods.

    These columns preserve the original Hessian/heuristic covariance columns and
    provide a separate truth-diagnostic covariance that includes empirical
    alpha/floor/range inflation.
    """
    if not cfg.write_inflated_raw_covariances:
        return df

    out = df.copy()
    sigma_names = ["x", "y", "z", "rx", "ry", "rz"]

    for method in methods:
        if not have_cols(out, [method.valid_col, *method.t_cols]):
            continue

        t_raw = numeric(out, method.t_cols)
        valid = as_bool(out[method.valid_col]) & finite_rows(t_raw)
        if have_cols(out, method.q_cols):
            valid &= finite_rows(numeric(out, method.q_cols))

        cov_strings: list[str] = []
        sigmas = np.full((len(out), 6), np.nan, dtype=float)
        range_floor_sigma = np.full(len(out), np.nan, dtype=float)

        for k in range(len(out)):
            if not valid[k]:
                cov_strings.append("")
                continue

            z_pose = np.zeros(6, dtype=float)
            z_pose[:3] = t_raw[k]
            C_nom = nominal_measurement_covariance_for_row(out.iloc[int(k)], method, z_pose, cfg)
            C_eff = (
                inflate_pose_covariance(
                    C_nom,
                    z_pose,
                    cfg,
                    alpha_trans=cfg.raw_cov_alpha_trans,
                    alpha_rot=cfg.raw_cov_alpha_rot,
                    floor_trans=cfg.raw_cov_floor_trans,
                    floor_rot=cfg.raw_cov_floor_rot,
                    range_sigma_frac=cfg.raw_cov_range_sigma_frac,
                )
                if cfg.inflate_reported_covariances
                else C_nom
            )
            cov_strings.append(matrix_to_string(C_eff))
            sigmas[k] = np.sqrt(np.clip(np.diag(C_eff), 0.0, np.inf))

            rng = float(np.linalg.norm(z_pose[:3]))
            range_floor_sigma[k] = math.sqrt(
                max(float(cfg.raw_cov_floor_trans) ** 2 + (float(cfg.raw_cov_range_sigma_frac) * rng) ** 2, 0.0)
            )

        prefix = method.key
        out[f"{prefix}_cov6_effective"] = cov_strings
        for i, name in enumerate(sigma_names):
            out[f"{prefix}_sigma_{name}_effective"] = sigmas[:, i]
        out[f"{prefix}_range_sigma_floor_effective"] = range_floor_sigma
        out[f"{prefix}_cov_alpha_trans_effective"] = float(cfg.raw_cov_alpha_trans)
        out[f"{prefix}_cov_alpha_rot_effective"] = float(cfg.raw_cov_alpha_rot)
        out[f"{prefix}_cov_floor_trans_effective"] = float(cfg.raw_cov_floor_trans)
        out[f"{prefix}_cov_floor_rot_effective"] = float(cfg.raw_cov_floor_rot)
        out[f"{prefix}_cov_range_sigma_frac_effective"] = float(cfg.raw_cov_range_sigma_frac)

    return out


def append_result_columns(df: pd.DataFrame, method: MethodSpec, result: SmootherResult) -> pd.DataFrame:
    out = df.copy()
    N = len(out)
    poses = result.x[: 6 * N].reshape(N, 6)
    rates = result.x[6 * N : 12 * N].reshape(N, 6)

    t_s = poses[:, :3]
    phi_s = poses[:, 3:]
    q_s = tangent_to_quaternion(phi_s, result.q0)

    prefix = f"{method.key}_fg"

    out[f"{prefix}_valid"] = result.valid
    out[f"{prefix}_tvec_x"] = t_s[:, 0]
    out[f"{prefix}_tvec_y"] = t_s[:, 1]
    out[f"{prefix}_tvec_z"] = t_s[:, 2]
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

    out[f"{prefix}_range"] = np.linalg.norm(t_s, axis=1)

    covs = result.cov_pose_blocks
    sig = np.sqrt(np.clip(np.diagonal(covs, axis1=1, axis2=2), 0.0, np.inf))
    sigma_names = ["x", "y", "z", "rx", "ry", "rz"]
    for i, name in enumerate(sigma_names):
        out[f"{prefix}_sigma_{name}"] = sig[:, i]

    out[f"{prefix}_cov6"] = [matrix_to_string(covs[k]) for k in range(N)]
    out[f"{prefix}_cov_alpha_trans"] = float(result.solve_info.get("cov_alpha_trans", np.nan))
    out[f"{prefix}_cov_alpha_rot"] = float(result.solve_info.get("cov_alpha_rot", np.nan))
    out[f"{prefix}_cov_floor_trans"] = float(result.solve_info.get("cov_floor_trans", np.nan))
    out[f"{prefix}_cov_floor_rot"] = float(result.solve_info.get("cov_floor_rot", np.nan))
    out[f"{prefix}_cov_range_sigma_frac"] = float(result.solve_info.get("cov_range_sigma_frac", np.nan))
    out[f"{prefix}_cov_inflated"] = bool(result.solve_info.get("inflate_reported_covariances", False))
    out[f"{prefix}_measurement_covariances_inflated"] = bool(result.solve_info.get("inflate_measurement_covariances", False))
    out[f"{prefix}_measurement_cov_alpha_trans"] = float(result.solve_info.get("meas_cov_alpha_trans", np.nan))
    out[f"{prefix}_measurement_cov_alpha_rot"] = float(result.solve_info.get("meas_cov_alpha_rot", np.nan))
    out[f"{prefix}_measurement_cov_floor_trans"] = float(result.solve_info.get("meas_cov_floor_trans", np.nan))
    out[f"{prefix}_measurement_cov_floor_rot"] = float(result.solve_info.get("meas_cov_floor_rot", np.nan))
    out[f"{prefix}_measurement_cov_range_sigma_frac"] = float(result.solve_info.get("meas_cov_range_sigma_frac", np.nan))
    out[f"{prefix}_measurement_count"] = result.measurement_valid_count
    out[f"{prefix}_covariance_mode"] = result.solve_info["covariance_mode"]
    out[f"{prefix}_solver_status"] = result.solve_info["solver_status"]

    return out


def build_summary_rows(method: MethodSpec, result: SmootherResult) -> dict[str, object]:
    N = result.solve_info["n_rows"]
    return {
        "method": method.label,
        "key": method.key,
        "rows": int(N),
        "measurement_valid_count": int(result.measurement_valid_count),
        "fg_valid_count": int(np.sum(result.valid)),
        "solver_status": result.solve_info["solver_status"],
        "covariance_mode": result.solve_info["covariance_mode"],
        "cov_alpha_trans": result.solve_info.get("cov_alpha_trans", np.nan),
        "cov_alpha_rot": result.solve_info.get("cov_alpha_rot", np.nan),
        "cov_floor_trans": result.solve_info.get("cov_floor_trans", np.nan),
        "cov_floor_rot_rad": result.solve_info.get("cov_floor_rot", np.nan),
        "cov_range_sigma_frac": result.solve_info.get("cov_range_sigma_frac", np.nan),
        "meas_cov_alpha_trans": result.solve_info.get("meas_cov_alpha_trans", np.nan),
        "meas_cov_alpha_rot": result.solve_info.get("meas_cov_alpha_rot", np.nan),
        "meas_cov_floor_trans": result.solve_info.get("meas_cov_floor_trans", np.nan),
        "meas_cov_floor_rot_rad": result.solve_info.get("meas_cov_floor_rot", np.nan),
        "meas_cov_range_sigma_frac": result.solve_info.get("meas_cov_range_sigma_frac", np.nan),
        "inflate_measurement_covariances": result.solve_info.get("inflate_measurement_covariances", False),
        "inflate_pnp_measurement_covariance": result.solve_info.get("inflate_pnp_measurement_covariance", False),
        "inflate_reported_covariances": result.solve_info.get("inflate_reported_covariances", False),
    }


def save_summary(summary_rows: list[dict[str, object]], path: Path) -> None:
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(path.with_suffix(".csv"), index=False)

    lines = ["# SE(3) factor-graph smoothing summary", ""]
    if summary.empty:
        lines.append("No methods were smoothed.")
    else:
        lines.append(summary.to_string(index=False))
    path.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def choose_csv_with_dialog() -> Path | None:
    try:
        from tkinter import Tk, filedialog
        root = Tk()
        root.withdraw()
        filename = filedialog.askopenfilename(
            title="Select navcalcs CSV to smooth",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        root.destroy()
        if not filename:
            return None
        return Path(filename)
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    cfg = SmootherConfig()
    p = argparse.ArgumentParser(description="Append independent SE(3) factor-graph smoothing columns to a navcalcs CSV.")
    p.add_argument("--csv", type=Path, default=None, help="Input navcalcs CSV. If omitted, a file dialog opens.")
    p.add_argument("--output", type=Path, default=None, help="Optional output CSV. Default overwrites input.")
    p.add_argument("--no-backup", action="store_true", help="Do not create a timestamped backup before overwriting input.")
    p.add_argument("--methods", nargs="*", default=["pnp", "qnp", "wqnp_yolo", "wqnp_kfest"], help="Methods to smooth.")

    p.add_argument("--process-sigma-pos", type=float, default=cfg.process_sigma_pos, help="Position process sigma for pose dynamics residual [m].")
    p.add_argument("--process-sigma-rot-deg", type=float, default=math.degrees(cfg.process_sigma_rot), help="Rotation process sigma for pose dynamics residual [deg].")
    p.add_argument("--process-sigma-vel", type=float, default=cfg.process_sigma_vel, help="Velocity random-walk sigma [m/s].")
    p.add_argument("--process-sigma-omega-deg", type=float, default=math.degrees(cfg.process_sigma_omega), help="Angular-rate random-walk sigma [deg/s].")

    p.add_argument("--velocity-prior-sigma", type=float, default=cfg.velocity_prior_sigma, help="Weak prior sigma for initial translational rate [m/s].")
    p.add_argument("--omega-prior-sigma-deg", type=float, default=math.degrees(cfg.omega_prior_sigma), help="Weak prior sigma for initial angular rate [deg/s].")

    p.add_argument("--min-trans-sigma", type=float, default=cfg.min_trans_sigma, help="Minimum translation sigma accepted from cov6 [m].")
    p.add_argument("--min-rot-sigma", type=float, default=cfg.min_rot_sigma, help="Minimum rotation sigma accepted from cov6 [rad].")
    p.add_argument("--max-trans-sigma", type=float, default=cfg.max_trans_sigma, help="Maximum translation sigma accepted from cov6 [m].")
    p.add_argument("--max-rot-deg", type=float, default=math.degrees(cfg.max_rot_sigma), help="Maximum rotation sigma accepted from cov6 [deg].")
    p.add_argument("--cov-eig-floor", type=float, default=cfg.cov_eig_floor, help="Eigenvalue floor used while regularizing cov6.")

    p.add_argument("--hessian-damping", type=float, default=cfg.hessian_damping, help="Small diagonal damping added to the normal equations.")
    p.add_argument(
        "--covariance-mode",
        choices=["block", "selected", "none"],
        default=cfg.covariance_mode,
        help=(
            "Covariance estimate mode. 'block' is fast local inverse of each pose Hessian block. "
            "'selected' computes selected diagonal inverse blocks of the full Hessian and is slower."
        ),
    )

    p.add_argument("--raw-cov-alpha-trans", type=float, default=cfg.raw_cov_alpha_trans, help="Raw effective covariance translation variance multiplier. 9.0 triples 1-sigma.")
    p.add_argument("--raw-cov-alpha-rot", type=float, default=cfg.raw_cov_alpha_rot, help="Raw effective covariance rotation variance multiplier. 4.0 doubles 1-sigma.")
    p.add_argument("--raw-cov-floor-trans", type=float, default=cfg.raw_cov_floor_trans, help="Raw effective covariance translation sigma floor [m].")
    p.add_argument("--raw-cov-floor-rot-deg", type=float, default=math.degrees(cfg.raw_cov_floor_rot), help="Raw effective covariance rotation sigma floor [deg].")
    p.add_argument("--raw-cov-range-sigma-frac", type=float, default=cfg.raw_cov_range_sigma_frac, help="Raw effective covariance extra translation sigma fraction times range, e.g. 0.005 gives 0.5 m at 100 m.")
    p.add_argument("--fg-cov-alpha-trans", type=float, default=cfg.fg_cov_alpha_trans, help="Exported FG covariance translation variance multiplier. 9.0 triples 1-sigma.")
    p.add_argument("--fg-cov-alpha-rot", type=float, default=cfg.fg_cov_alpha_rot, help="Exported FG covariance rotation variance multiplier. 4.0 doubles 1-sigma.")
    p.add_argument("--fg-cov-floor-trans", type=float, default=cfg.fg_cov_floor_trans, help="Exported FG covariance translation sigma floor [m].")
    p.add_argument("--fg-cov-floor-rot-deg", type=float, default=math.degrees(cfg.fg_cov_floor_rot), help="Exported FG covariance rotation sigma floor [deg].")
    p.add_argument("--fg-cov-range-sigma-frac", type=float, default=cfg.fg_cov_range_sigma_frac, help="Exported FG covariance extra translation sigma fraction times range, e.g. 0.005 gives 0.5 m at 100 m.")

    meas_group = p.add_mutually_exclusive_group()
    meas_group.add_argument("--inflate-measurement-covariances", dest="inflate_measurement_covariances", action="store_true", default=cfg.inflate_measurement_covariances, help="Use meas-cov inflation as FG measurement weights for covariance-exporting methods. Enabled by default.")
    meas_group.add_argument("--no-inflate-measurement-covariances", dest="inflate_measurement_covariances", action="store_false", help="Use nominal exported QnP/wQnP covariance columns directly as FG measurement weights.")
    p.add_argument("--inflate-pnp-measurement-covariance", action="store_true", default=cfg.inflate_pnp_measurement_covariance, help="Also apply meas-cov inflation to OpenCV SolvePnP's heuristic measurement covariance.")
    p.add_argument("--meas-cov-alpha-trans", type=float, default=cfg.meas_cov_alpha_trans, help="Measurement-weight covariance translation variance multiplier for covariance-exporting methods.")
    p.add_argument("--meas-cov-alpha-rot", type=float, default=cfg.meas_cov_alpha_rot, help="Measurement-weight covariance rotation variance multiplier for covariance-exporting methods.")
    p.add_argument("--meas-cov-floor-trans", type=float, default=cfg.meas_cov_floor_trans, help="Measurement-weight covariance translation sigma floor [m].")
    p.add_argument("--meas-cov-floor-rot-deg", type=float, default=math.degrees(cfg.meas_cov_floor_rot), help="Measurement-weight covariance rotation sigma floor [deg].")
    p.add_argument("--meas-cov-range-sigma-frac", type=float, default=cfg.meas_cov_range_sigma_frac, help="Measurement-weight covariance extra translation sigma fraction times range.")
    p.add_argument("--no-inflate-reported-covariances", action="store_true", help="Disable raw and FG alpha/floor/range inflation on exported covariance columns.")
    p.add_argument("--no-write-inflated-raw-covariances", action="store_true", help="Do not append <method>_cov6_effective raw covariance columns.")

    p.add_argument("--pnp-sigma-pos-base", type=float, default=cfg.pnp_sigma_pos_base)
    p.add_argument("--pnp-sigma-pos-range-frac", type=float, default=cfg.pnp_sigma_pos_range_frac)
    p.add_argument("--pnp-sigma-pos-rmse-scale", type=float, default=cfg.pnp_sigma_pos_rmse_scale)
    p.add_argument("--pnp-sigma-rot-deg-base", type=float, default=cfg.pnp_sigma_rot_deg_base)
    p.add_argument("--pnp-sigma-rot-deg-rmse-scale", type=float, default=cfg.pnp_sigma_rot_deg_rmse_scale)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.csv or choose_csv_with_dialog()
    if csv_path is None:
        raise SystemExit("No CSV selected.")

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    output_path = Path(args.output) if args.output is not None else csv_path

    df = pd.read_csv(csv_path)
    if "image_time" in df.columns:
        times_abs = pd.to_numeric(df["image_time"], errors="coerce").to_numpy(dtype=float)
    elif "frame_idx" in df.columns:
        times_abs = pd.to_numeric(df["frame_idx"], errors="coerce").to_numpy(dtype=float)
    else:
        times_abs = np.arange(len(df), dtype=float)

    if not np.all(np.isfinite(times_abs)):
        # Use row index for any irredeemable time entries.
        bad = ~np.isfinite(times_abs)
        times_abs[bad] = np.arange(len(df), dtype=float)[bad]

    # Relative times improve numerical conditioning without changing dt.
    times = times_abs - times_abs[0]

    cfg = SmootherConfig(
        process_sigma_pos=float(args.process_sigma_pos),
        process_sigma_rot=math.radians(float(args.process_sigma_rot_deg)),
        process_sigma_vel=float(args.process_sigma_vel),
        process_sigma_omega=math.radians(float(args.process_sigma_omega_deg)),
        velocity_prior_sigma=float(args.velocity_prior_sigma),
        omega_prior_sigma=math.radians(float(args.omega_prior_sigma_deg)),
        pnp_sigma_pos_base=float(args.pnp_sigma_pos_base),
        pnp_sigma_pos_range_frac=float(args.pnp_sigma_pos_range_frac),
        pnp_sigma_pos_rmse_scale=float(args.pnp_sigma_pos_rmse_scale),
        pnp_sigma_rot_deg_base=float(args.pnp_sigma_rot_deg_base),
        pnp_sigma_rot_deg_rmse_scale=float(args.pnp_sigma_rot_deg_rmse_scale),
        raw_cov_alpha_trans=float(args.raw_cov_alpha_trans),
        raw_cov_alpha_rot=float(args.raw_cov_alpha_rot),
        raw_cov_floor_trans=float(args.raw_cov_floor_trans),
        raw_cov_floor_rot=math.radians(float(args.raw_cov_floor_rot_deg)),
        raw_cov_range_sigma_frac=float(args.raw_cov_range_sigma_frac),
        fg_cov_alpha_trans=float(args.fg_cov_alpha_trans),
        fg_cov_alpha_rot=float(args.fg_cov_alpha_rot),
        fg_cov_floor_trans=float(args.fg_cov_floor_trans),
        fg_cov_floor_rot=math.radians(float(args.fg_cov_floor_rot_deg)),
        fg_cov_range_sigma_frac=float(args.fg_cov_range_sigma_frac),
        meas_cov_alpha_trans=float(args.meas_cov_alpha_trans),
        meas_cov_alpha_rot=float(args.meas_cov_alpha_rot),
        meas_cov_floor_trans=float(args.meas_cov_floor_trans),
        meas_cov_floor_rot=math.radians(float(args.meas_cov_floor_rot_deg)),
        meas_cov_range_sigma_frac=float(args.meas_cov_range_sigma_frac),
        inflate_measurement_covariances=bool(args.inflate_measurement_covariances),
        inflate_pnp_measurement_covariance=bool(args.inflate_pnp_measurement_covariance),
        inflate_reported_covariances=not bool(args.no_inflate_reported_covariances),
        write_inflated_raw_covariances=not bool(args.no_write_inflated_raw_covariances),
        min_trans_sigma=float(args.min_trans_sigma),
        min_rot_sigma=float(args.min_rot_sigma),
        max_trans_sigma=float(args.max_trans_sigma),
        max_rot_sigma=math.radians(float(args.max_rot_deg)),
        cov_eig_floor=float(args.cov_eig_floor),
        hessian_damping=float(args.hessian_damping),
        covariance_mode=str(args.covariance_mode),
    )

    requested = set(args.methods)
    methods = [m for m in available_methods(df) if m.key in requested]
    if not methods:
        raise SystemExit("No requested pose methods with complete columns were found in the CSV.")

    summary_rows: list[dict[str, object]] = []
    out_df = df.copy()
    out_df = append_inflated_raw_covariance_columns(out_df, methods, cfg)

    print(f"Input CSV: {csv_path}")
    print(f"Rows: {len(df)}")
    print(f"Methods: {', '.join(m.key for m in methods)}")
    print("")

    for m in methods:
        print(f"=== {m.label} ({m.key}) ===")
        try:
            result = solve_factor_graph_for_method(out_df, times, m, cfg)
        except Exception as exc:
            print(f"  Skipping {m.key}: {exc}")
            continue

        out_df = append_result_columns(out_df, m, result)
        summary_rows.append(build_summary_rows(m, result))
        print(
            f"  measurements: {result.measurement_valid_count}/{len(df)}; "
            f"solver={result.solve_info['solver_status']}; "
            f"covariance={result.solve_info['covariance_mode']}"
        )

    if not summary_rows:
        raise SystemExit("No factor-graph results were generated.")

    # if output_path.resolve() == csv_path.resolve() and not args.no_backup:
    #     stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     backup = csv_path.with_name(f"{csv_path.stem}_before_fg_{stamp}{csv_path.suffix}")
        # shutil.copy2(csv_path, backup)
        # print(f"\nBackup written: {backup}")

    out_df.to_csv(output_path, index=False)

    summary_base = output_path.with_name(f"{output_path.stem}_fg_summary")
    save_summary(summary_rows, summary_base)

    print(f"Updated CSV written: {output_path}")
    print(f"Summary written: {summary_base.with_suffix('.txt')}")
    print("")
    print("New columns use the pattern <method>_fg_*; e.g., qnp_fg_tvec_x, qnp_fg_qw, qnp_fg_cov6.")


if __name__ == "__main__":
    main()
