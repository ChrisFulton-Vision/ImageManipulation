from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

# Uses your existing Quaternion helper for synthetic data generation only.
# If you don't have support.mathHelpers.quaternions in a given environment,
# you can still use the fusion core by providing real SE(3) measurements.
# Uses your existing Quaternion helper for synthetic data generation only.
# If you don't have support.mathHelpers.quaternions available, you can still use the fusion core
# by providing real SE(3) measurements and avoiding the synthetic test harness.
try:
    from support.mathHelpers.quaternions import Quaternion as q, random_quat_within_deg, from_SE3, mats2quats, \
        se3s2quats
except Exception:  # pragma: no cover
    q = None


    def random_quat_within_deg(*args, **kwargs):
        raise ImportError("support.mathHelpers.quaternions not found; synthetic test harness is unavailable.")

np.set_printoptions(suppress=True, precision=6)


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def merge_times(times: List[float], tol: float) -> np.ndarray:
    """
    Merge timestamps within `tol` into a single representative time.
    This keeps the solver robust to tiny float rounding differences while still allowing
    genuinely asynchronous sensor updates (e.g., 0.1s apart).
    """
    if not times:
        return np.array([], dtype=float)
    times_sorted = np.array(sorted(float(t) for t in times), dtype=float)
    merged = [float(times_sorted[0])]
    for t in times_sorted[1:]:
        if abs(float(t) - merged[-1]) <= tol:
            continue
        merged.append(float(t))
    return np.array(merged, dtype=float)


def quat_and_t_from_SE3(T: np.ndarray) -> Tuple["q", np.ndarray]:
    """
    Your q().from_SE3() may return either:
        - quat
        - (quat, t)
    This helper supports both patterns.
    """
    if q is None:
        raise RuntimeError("Quaternion helper not available in this environment.")
    res = from_SE3(T)
    if isinstance(res, tuple) and len(res) == 2:
        qq, tt = res
        return qq, np.asarray(tt, dtype=float).reshape(3)
    return res, np.asarray(T[:3, 3], dtype=float).reshape(3)


# =============================================================================
# MultiStateFusionFGMeasurementMelding.py
#
# Purpose
# -------
# A lightweight, dependency-free, factor-graph-style *multi-state* fusion example for
# estimating a time-series of object poses in SE(3), using:
#   - timestamped pose measurements from heterogeneous sensors, and
#   - a simple motion model:
#       nearly constant linear velocity (world-frame), and
#       nearly constant angular rate (body/world-frame, see notes).
#
# State per time k
# ---------------
#   X_k = { T_w_obj_k, v_w_k, w_k }
# where
#   - T_w_obj_k : 4x4 SE(3) pose (obj -> world)
#   - v_w_k     : (3,) linear velocity in world frame
#   - w_k       : (3,) angular rate used in propagation
#
# Motion model (between consecutive state times)
# ----------------------------------------------
#   t_{k+1} = t_k + v_k * dt
#   R_{k+1} = Exp(w_k * dt) * R_k
#   v_{k+1} = v_k
#   w_{k+1} = w_k
#
# Measurements (factors)
# ----------------------
# Each measurement at timestamp t_k provides:
#   - sensor extrinsic: T_w_s (sensor -> world)
#   - relative pose measurement: T_s_obj_meas (obj -> sensor)
#   - 6x6 covariance in measurement residual coordinates
#
# Predicted measurement:
#   T_s_obj_pred = inv(T_w_s) * T_w_obj_k
#
# Residual convention
# -------------------
# We reuse the exact pose residual convention from StaticFGMeasurementMelding:
#   r_t = t_pred - t_meas
#   r_R = log( R_meas^T * R_pred )
# so "prediction minus reference".
#
# Optimization
# ------------
# - LM / Gauss-Newton on the stacked residuals
# - Numeric Jacobian (explicit, easy to tweak / debug)
# - Pose is updated via a left perturbation:
#       T <- Exp(delta) * T
#   while v and w are updated additively.
#
# Diagnostics / plotting
# ----------------------
# - 3D world trajectory (truth, estimate, and per-measurement world "lifts")
# - per-axis position vs time plots
# - simple orientation error vs time plot
#
# =============================================================================


# -----------------------------------------------------------------------------
# SE(3) / SO(3) helpers (same frame convention as your static file)
# -----------------------------------------------------------------------------
def se3_mul(T_ab: NDArray[np.float64], T_bc: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compose SE(3) transforms: T_ac = T_ab * T_bc (x_a = T_ab @ x_b)."""
    return T_ab @ T_bc


def se3_inv(T_ab: NDArray[np.float64]) -> NDArray[np.float64]:
    """Invert an SE(3) transform (b->a) => (a->b)."""
    R = T_ab[:3, :3]
    t = T_ab[:3, 3]
    T_ba = np.eye(4, dtype=float)
    T_ba[:3, :3] = R.T
    T_ba[:3, 3] = -R.T @ t
    return T_ba


def _skew(w: NDArray[np.float64]) -> NDArray[np.float64]:
    wx, wy, wz = float(w[0]), float(w[1]), float(w[2])
    return np.array([[0.0, -wz, wy],
                     [wz, 0.0, -wx],
                     [-wy, wx, 0.0]], dtype=float)


def so3_exp(w: NDArray[np.float64]) -> NDArray[np.float64]:
    """SO(3) Exp via Rodrigues."""
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3, dtype=float) + _skew(w)
    k = w / theta
    K = _skew(k)
    s = math.sin(theta)
    c = math.cos(theta)
    return np.eye(3, dtype=float) + s * K + (1.0 - c) * (K @ K)


def so3_log(R: NDArray[np.float64]) -> NDArray[np.float64]:
    """SO(3) Log with trace clamp."""
    tr = float(np.trace(R))
    cos_theta = (tr - 1.0) * 0.5
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = float(math.acos(cos_theta))

    if theta < 1e-12:
        W = 0.5 * (R - R.T)
        return np.array([W[2, 1], W[0, 2], W[1, 0]], dtype=float)

    W = (R - R.T) * (0.5 / math.sin(theta))
    w = np.array([W[2, 1], W[0, 2], W[1, 0]], dtype=float)
    return theta * w


def se3_from_Rt(R: NDArray[np.float64], t: NDArray[np.float64]) -> NDArray[np.float64]:
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def se3_perturb_left(T: NDArray[np.float64], delta6: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Left perturbation:
        T <- Exp(delta) * T
    using small-step retraction:
        R <- Exp(dtheta) @ R
        t <- t + dt
    where delta6 = [dt(3), dtheta(3)].
    """
    dt = np.asarray(delta6[:3], dtype=float)
    dth = np.asarray(delta6[3:6], dtype=float)
    R = T[:3, :3]
    t = T[:3, 3]
    Rn = so3_exp(dth) @ R
    tn = t + dt
    return se3_from_Rt(Rn, tn)


def se3_residual(T_pred: NDArray[np.float64], T_ref: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    6D residual between predicted and reference SE(3):
      r_t = t_pred - t_ref
      r_R = log( R_ref^T * R_pred )
    """
    R_pred = T_pred[:3, :3]
    t_pred = T_pred[:3, 3]
    R_ref = T_ref[:3, :3]
    t_ref = T_ref[:3, 3]

    r_t = t_pred - t_ref
    r_R = so3_log(R_ref.T @ R_pred)
    return np.hstack([r_t, r_R]).astype(float)


def chol_whitener(C: NDArray[np.float64]) -> NDArray[np.float64]:
    L = np.linalg.cholesky(C)
    return np.linalg.inv(L)


def _angle_from_R(R: NDArray[np.float64]) -> float:
    c = (float(np.trace(R)) - 1.0) * 0.5
    c = float(np.clip(c, -1.0, 1.0))
    return float(np.arccos(c))


# -----------------------------------------------------------------------------
# Sensor / Measurement definitions (timestamped)
# -----------------------------------------------------------------------------
CovModel = Callable[["TimedMeasurement"], np.ndarray]  # returns (6,6)


@dataclass(frozen=True)
class Sensor:
    name: str
    T_w_s: np.ndarray  # (4,4) sensor pose in world (sensor -> world)
    cov_model: CovModel


@dataclass
class TimedMeasurement:
    sensor: Sensor
    t: float  # seconds (or any monotonically increasing timebase)
    T_s_obj: np.ndarray  # (4,4) measurement: obj -> sensor
    meta: Dict[str, Any] = field(default_factory=dict)

    def covariance(self) -> np.ndarray:
        C = np.asarray(self.sensor.cov_model(self), dtype=float)
        if C.shape == (6,):
            C = np.diag(C)
        if C.shape != (6, 6):
            raise ValueError("Covariance must be (6,) or (6,6)")
        np.linalg.cholesky(C)  # PD check
        return C


def rle_cov(sig_r_lat_el, sig_rot):
    """
    Same R/L/El translation covariance model used in your static FG file,
    expressed in the sensor frame for translation and in Rodrigues coords for rotation.
    """
    sig_r = float(sig_r_lat_el[0])
    sig_lat = float(sig_r_lat_el[1])
    sig_el = float(sig_r_lat_el[2])
    sig_rot = np.array(sig_rot, dtype=float).reshape(-1)
    if sig_rot.size == 1:
        sig_rot = np.repeat(sig_rot, 3)

    def model(meas: TimedMeasurement) -> np.ndarray:
        t = np.asarray(meas.T_s_obj[:3, 3], dtype=float)
        d = float(np.linalg.norm(t))
        if d < 1e-9:
            C_tt_s = np.diag([sig_r ** 2, sig_lat ** 2, sig_el ** 2])
        else:
            u_r = t / d

            z_ref = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(float(u_r @ z_ref)) > 0.95:
                z_ref = np.array([1.0, 0.0, 0.0], dtype=float)

            u_lat = np.cross(z_ref, u_r)
            u_lat /= max(np.linalg.norm(u_lat), 1e-12)

            u_el = np.cross(u_r, u_lat)
            u_el /= max(np.linalg.norm(u_el), 1e-12)

            R_s_rle = np.column_stack([u_r, u_lat, u_el])
            C_rle = np.diag([sig_r ** 2, sig_lat ** 2, sig_el ** 2])
            C_tt_s = R_s_rle @ C_rle @ R_s_rle.T

        C = np.zeros((6, 6), dtype=float)
        C[:3, :3] = C_tt_s
        C[3:, 3:] = np.diag(sig_rot ** 2)
        return C

    return model


# -----------------------------------------------------------------------------
# Multi-state factor graph
# -----------------------------------------------------------------------------
@dataclass
class ProcessNoise:
    """
    Simple diagonal process-noise model in the *residual coordinates* of the dynamics factor.

    Dynamics residual is 12D:
      [pose(6), vel(3), omega(3)]

    Default scaling:
      - pose translation/rotation residuals treated as position/angle errors
      - vel/omega residuals treated as delta-v/delta-omega over the step

    You can treat these as tuning knobs; keep them conservative at first.
    """
    sig_pos: float = 0.25  # meters
    sig_rot: float = 0.02  # radians
    sig_vel: float = 0.05  # m/s
    sig_omega: float = 0.05  # rad/s

    def cov12(self, dt: float) -> np.ndarray:
        # Mild dt-scaling. For CV / constant-rate, you can choose other scalings later.
        # Here: pose residual ~ (position, angle), vel residual ~ (m/s), omega residual ~ (rad/s)
        sp = float(self.sig_pos)
        sr = float(self.sig_rot)
        sv = float(self.sig_vel)
        so = float(self.sig_omega)

        diag = np.array([sp ** 2, sp ** 2, sp ** 2,
                         sr ** 2, sr ** 2, sr ** 2,
                         sv ** 2, sv ** 2, sv ** 2,
                         so ** 2, so ** 2, so ** 2], dtype=float)

        # Optional gentle dt influence (helps when dt varies)
        # - larger dt => slightly larger expected model mismatch in pose
        # - smaller dt => slightly tighter pose constraint
        scale_pose = max(dt, 1e-6)
        diag[:6] *= scale_pose

        return np.diag(diag)


@dataclass
class TrajectoryState:
    T_w_obj: np.ndarray  # (4,4)
    v_w: np.ndarray  # (3,)
    w: np.ndarray  # (3,)


class MultiStateFusionFGMeasurementMelding:
    """
    Multi-state SE(3) trajectory fusion with CV / constant-rate dynamics.
    """

    def __init__(
            self,
            measurements: List[TimedMeasurement],
            *,
            process_noise: ProcessNoise | None = None,
            time_merge_tol: float = 1e-6):
        if not measurements:
            raise ValueError("Need at least 1 measurement")
        self.measurements = sorted(measurements, key=lambda m: float(m.t))
        self.process_noise = process_noise or ProcessNoise()
        self._diagnostics: Dict[str, Any] = {}

        # Unique state times (robust to tiny rounding; still supports async updates)
        raw_times = [float(m.t) for m in self.measurements]
        self.times = merge_times(raw_times, tol=float(time_merge_tol))
        if self.times.size < 1:
            raise ValueError("No timestamps found")

        # Map each measurement to the nearest state time bin (within tol)
        self._meas_to_k: List[int] = []
        for m in self.measurements:
            t = float(m.t)
            k = int(np.searchsorted(self.times, t, side="left"))
            cand = []
            if 0 <= k < self.times.size:
                cand.append(k)
            if k - 1 >= 0:
                cand.append(k - 1)
            if not cand:
                raise RuntimeError("No state times available")
            k_best = min(cand, key=lambda kk: abs(float(self.times[kk]) - t))
            if abs(float(self.times[k_best]) - t) > float(time_merge_tol):
                raise ValueError(
                    f"Measurement time {t} not within tol={time_merge_tol} of any state time"
                )
            self._meas_to_k.append(int(k_best))

        # Precompute measurement whiteners
        self._W_meas = [chol_whitener(m.covariance()) for m in self.measurements]

    # --------------------------
    # Initial guess construction
    # --------------------------
    def initial_guess(self) -> List[TrajectoryState]:
        """
        Build an initial guess by:
          1) Lifting measurements into world at each timestamp
          2) Information-weighted average translation + chordal rotation mean (per time)
          3) Finite-difference to seed v and w
        """
        # group measurements by state index
        by_k: Dict[int, List[int]] = {}
        for i, _m in enumerate(self.measurements):
            k = int(self._meas_to_k[i])
            by_k.setdefault(k, []).append(i)

        T_list: List[np.ndarray] = []

        for k in range(int(self.times.size)):
            idxs = by_k.get(k, [])
            if not idxs:
                # No measurements at this time bin: carry forward last pose (will be shaped by dynamics)
                if T_list:
                    T_list.append(T_list[-1].copy())
                    continue
                raise ValueError("First state time has no measurements; provide at least one measurement at start")

            lifts = [se3_mul(self.measurements[i].sensor.T_w_s, self.measurements[i].T_s_obj) for i in idxs]

            # translation info-weighted mean
            Wsum = np.zeros((3, 3), dtype=float)
            bsum = np.zeros(3, dtype=float)
            for i, T_w in zip(idxs, lifts):
                C = self.measurements[i].covariance()
                Wi = np.linalg.inv(C[:3, :3])
                ti = T_w[:3, 3]
                Wsum += Wi
                bsum += Wi @ ti
            t0 = np.linalg.solve(Wsum, bsum)

            # chordal rotation mean + projection
            M = np.zeros((3, 3), dtype=float)
            for i, T_w in zip(idxs, lifts):
                C = self.measurements[i].covariance()
                Ri = T_w[:3, :3]
                wi = 1.0 / float(np.clip(np.trace(C[3:6, 3:6]), 1e-12, np.inf))
                M += wi * Ri
            U, _, Vt = np.linalg.svd(M)
            R0 = U @ Vt
            if np.linalg.det(R0) < 0:
                U[:, -1] *= -1
                R0 = U @ Vt

            T_list.append(se3_from_Rt(R0, t0))

        N = len(T_list)
        v_list = [np.zeros(3, dtype=float) for _ in range(N)]
        w_list = [np.zeros(3, dtype=float) for _ in range(N)]

        # Seed v and w from pose diffs
        if N >= 2:
            for k in range(N - 1):
                dt = float(self.times[k + 1] - self.times[k])
                dt = max(dt, 1e-9)
                t0 = T_list[k][:3, 3]
                t1 = T_list[k + 1][:3, 3]
                v_list[k] = (t1 - t0) / dt

                R0 = T_list[k][:3, :3]
                R1 = T_list[k + 1][:3, :3]
                dR = R1 @ R0.T
                w_list[k] = so3_log(dR) / dt

            v_list[-1] = v_list[-2].copy()
            w_list[-1] = w_list[-2].copy()

        states = [TrajectoryState(T_w_obj=T_list[k], v_w=v_list[k], w=w_list[k]) for k in range(N)]
        self._diagnostics["T_w_obj_init_list"] = [s.T_w_obj.copy() for s in states]
        self._diagnostics["v_init_list"] = [s.v_w.copy() for s in states]
        self._diagnostics["w_init_list"] = [s.w.copy() for s in states]
        return states

    # --------------------------
    # Packing / unpacking
    # --------------------------
    def _pack(self, states: List[TrajectoryState]) -> np.ndarray:
        """
        Pack state variables into a single vector x.
        Note: poses are *not* stored in x; x stores v and w only, and we keep poses in list form.
        For numeric Jacobian, we still need a consistent "delta ordering", so we treat x as:
            [pose_deltas(6N), v(3N), w(3N)]
        but the baseline (non-perturbed) x is zero for pose deltas.
        """
        N = len(states)
        x = np.zeros(6 * N + 3 * N + 3 * N, dtype=float)
        # fill v and w
        off_v = 6 * N
        off_w = off_v + 3 * N
        for k, s in enumerate(states):
            x[off_v + 3 * k: off_v + 3 * k + 3] = s.v_w
            x[off_w + 3 * k: off_w + 3 * k + 3] = s.w
        return x

    def _unpack(self, x: np.ndarray, T_list: List[np.ndarray]) -> List[TrajectoryState]:
        N = len(T_list)
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != 12 * N:
            raise ValueError(f"x size {x.size} != 12N ({12 * N})")
        off_v = 6 * N
        off_w = off_v + 3 * N
        states: List[TrajectoryState] = []
        for k in range(N):
            v = x[off_v + 3 * k: off_v + 3 * k + 3].copy()
            w = x[off_w + 3 * k: off_w + 3 * k + 3].copy()
            states.append(TrajectoryState(T_w_obj=T_list[k], v_w=v, w=w))
        return states

    def _apply_delta(self, base_states: List[TrajectoryState], delta: np.ndarray) -> List[TrajectoryState]:
        """
        Apply a stacked delta = [dpose(6N), dv(3N), dw(3N)] to a list of states.
        """
        N = len(base_states)
        delta = np.asarray(delta, dtype=float).reshape(-1)
        if delta.size != 12 * N:
            raise ValueError(f"delta size {delta.size} != 12N ({12 * N})")

        T_new = []
        for k in range(N):
            dpose = delta[6 * k: 6 * k + 6]
            T_new.append(se3_perturb_left(base_states[k].T_w_obj, dpose))

        off_v = 6 * N
        off_w = off_v + 3 * N
        out: List[TrajectoryState] = []
        for k in range(N):
            dv = delta[off_v + 3 * k: off_v + 3 * k + 3]
            dw = delta[off_w + 3 * k: off_w + 3 * k + 3]
            out.append(TrajectoryState(
                T_w_obj=T_new[k],
                v_w=base_states[k].v_w + dv,
                w=base_states[k].w + dw,
            ))
        return out

    # --------------------------
    # Residual stacking
    # --------------------------
    def _stacked_whitened_residual(self, states: List[TrajectoryState]) -> np.ndarray:
        """
        Residual vector r(x) stacked as:
          - all measurement factors (6 each)
          - all dynamics factors between consecutive states (12 each)
        """
        N = len(states)
        # measurement factors
        chunks: List[np.ndarray] = []
        for mi, (m, W) in enumerate(zip(self.measurements, self._W_meas)):
            k = int(self._meas_to_k[mi])
            T_w = states[k].T_w_obj
            T_s_pred = se3_mul(se3_inv(m.sensor.T_w_s), T_w)
            r6 = se3_residual(T_s_pred, m.T_s_obj)
            chunks.append(W @ r6)

        # dynamics factors
        for k in range(N - 1):
            dt = float(self.times[k + 1] - self.times[k])
            dt = max(dt, 1e-9)
            s0 = states[k]
            s1 = states[k + 1]

            # propagate pose using (v_k, w_k)
            t_pred = s0.T_w_obj[:3, 3] + s0.v_w * dt
            R_pred = so3_exp(s0.w * dt) @ s0.T_w_obj[:3, :3]
            T_pred = se3_from_Rt(R_pred, t_pred)

            r_pose = se3_residual(T_pred, s1.T_w_obj)  # 6
            r_v = (s1.v_w - s0.v_w).astype(float)  # 3
            r_w = (s1.w - s0.w).astype(float)  # 3
            r12 = np.hstack([r_pose, r_v, r_w])

            Q = self.process_noise.cov12(dt)
            Wq = chol_whitener(Q)
            chunks.append(Wq @ r12)

        return np.hstack(chunks)

    @staticmethod
    def _cost(r_w: np.ndarray) -> float:
        return 0.5 * float(r_w @ r_w)

    # --------------------------
    # Numeric Jacobian
    # --------------------------
    def numeric_jacobian(self,
                         states: List[TrajectoryState],
                         *,
                         eps_pose_t: float = 1e-4,
                         eps_pose_r: float = 1e-5,
                         eps_v: float = 1e-4,
                         eps_w: float = 1e-5) -> np.ndarray:
        """
        Numeric Jacobian dr/d(delta) where delta has length 12N.
        """
        N = len(states)
        r0 = self._stacked_whitened_residual(states)
        m = r0.size
        J = np.zeros((m, 12 * N), dtype=float)

        # pose translation (3)
        for k in range(N):
            for i in range(3):
                d = np.zeros(12 * N, dtype=float)
                d[6 * k + i] = eps_pose_t
                sp = self._apply_delta(states, d)
                rp = self._stacked_whitened_residual(sp)
                J[:, 6 * k + i] = (rp - r0) / eps_pose_t

        # pose rotation (3)
        for k in range(N):
            for i in range(3):
                d = np.zeros(12 * N, dtype=float)
                d[6 * k + 3 + i] = eps_pose_r
                sp = self._apply_delta(states, d)
                rp = self._stacked_whitened_residual(sp)
                J[:, 6 * k + 3 + i] = (rp - r0) / eps_pose_r

        off_v = 6 * N
        off_w = off_v + 3 * N

        for k in range(N):
            for i in range(3):
                d = np.zeros(12 * N, dtype=float)
                d[off_v + 3 * k + i] = eps_v
                sp = self._apply_delta(states, d)
                rp = self._stacked_whitened_residual(sp)
                J[:, off_v + 3 * k + i] = (rp - r0) / eps_v

        for k in range(N):
            for i in range(3):
                d = np.zeros(12 * N, dtype=float)
                d[off_w + 3 * k + i] = eps_w
                sp = self._apply_delta(states, d)
                rp = self._stacked_whitened_residual(sp)
                J[:, off_w + 3 * k + i] = (rp - r0) / eps_w

        return J

    # --------------------------
    # LM optimizer
    # --------------------------
    def solve(self,
              *,
              max_iters: int = 20,
              lambda0: float = 1e-3,
              lambda_up: float = 10.0,
              lambda_down: float = 3.0,
              step_tol: float = 1e-6,
              eps_pose_t: float = 1e-4,
              eps_pose_r: float = 1e-5,
              eps_v: float = 1e-4,
              eps_w: float = 1e-5,
              verbose: bool = True) -> Tuple[List[TrajectoryState], Dict[str, Any]]:
        """
        Run LM on the full trajectory.
        """
        states = self.initial_guess()
        lam = float(lambda0)

        r = self._stacked_whitened_residual(states)
        cost = self._cost(r)

        it = 0
        converged = False
        step_norm = 1.0

        for it in range(1, max_iters + 1):
            J = self.numeric_jacobian(states,
                                      eps_pose_t=eps_pose_t,
                                      eps_pose_r=eps_pose_r,
                                      eps_v=eps_v,
                                      eps_w=eps_w)
            H = J.T @ J
            g = J.T @ r
            H_lm = H + lam * np.eye(H.shape[0], dtype=float)

            try:
                delta = -np.linalg.solve(H_lm, g)
            except np.linalg.LinAlgError:
                delta, *_ = np.linalg.lstsq(H_lm, -g, rcond=None)

            step_norm = float(np.linalg.norm(delta))

            trial = self._apply_delta(states, delta)
            r_trial = self._stacked_whitened_residual(trial)
            cost_trial = self._cost(r_trial)
            improve = cost - cost_trial

            if verbose:
                print(f"[LM] Iter {it:02d}: cost {cost:.6e}  step {step_norm:.3e}  "
                      f"improve {improve:.3e}  lambda {lam:.3e}")

            if improve > 0.0:
                states = trial
                r = r_trial
                cost = cost_trial
                lam = max(lam / lambda_down, 1e-12)
            else:
                lam = lam * lambda_up

            if step_norm < step_tol:
                converged = True
                break

        info = {
            "converged": converged,
            "iters": it,
            "final_cost": cost,
            "lambda_final": lam,
            "step_norm": step_norm,
            "eps_pose_t": eps_pose_t,
            "eps_pose_r": eps_pose_r,
            "eps_v": eps_v,
            "eps_w": eps_w,
        }

        self._diagnostics["states_opt"] = [TrajectoryState(
            T_w_obj=s.T_w_obj.copy(), v_w=s.v_w.copy(), w=s.w.copy()
        ) for s in states]
        self._diagnostics["lm_info"] = info

        return states, info

    # --------------------------
    # Diagnostics helpers
    # --------------------------
    def sensor_positions_world(self) -> Tuple[List[str], np.ndarray]:
        """
        Returns:
          - names: list[str] unique sensor names
          - pos_w: (S,3) world positions of each sensor (from T_w_s)
        """
        sensors_by_name = {}
        for m in self.measurements:
            sensors_by_name[m.sensor.name] = m.sensor

        names = sorted(sensors_by_name.keys())
        pos_w = np.stack([sensors_by_name[nm].T_w_s[:3, 3] for nm in names], axis=0)
        return names, pos_w

    def get_estimated_trajectory(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          - positions: (N,3)
          - rotations: (N,3,3)
        Requires solve() called.
        """
        states = self._diagnostics.get("states_opt", None)
        if states is None:
            raise RuntimeError("Call solve() first.")
        pos = np.stack([s.T_w_obj[:3, 3] for s in states], axis=0)
        rot = np.stack([s.T_w_obj[:3, :3] for s in states], axis=0)
        return pos, rot

    def lifted_measurements_world(self) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        Returns:
          - t_meas: (M,) timestamps
          - names: list[str] sensor name per measurement
          - pos_w: (M,3) world positions from "lift": T_w_s * T_s_obj
        """
        t_meas = np.array([float(m.t) for m in self.measurements], dtype=float)
        names = [m.sensor.name for m in self.measurements]
        pos_w = np.stack([se3_mul(m.sensor.T_w_s, m.T_s_obj)[:3, 3] for m in self.measurements], axis=0)
        return t_meas, names, pos_w

    def lifted_measurements_world_pose(self):
        """
        Lift each measurement pose into world frame.

        For each measurement m with:
            T_s_obj  : object pose expressed in sensor frame
            T_w_s    : sensor pose expressed in world frame

        We compute:
            T_w_obj = T_w_s @ T_s_obj

        Returns
        -------
        t_meas : (M,) np.ndarray
            Measurement timestamps.
        meas_names : list[str]
            Sensor name for each measurement.
        T_w_obj_meas : (M, 4, 4) np.ndarray
            Lifted world-frame pose matrices.
        """
        if len(self.measurements) == 0:
            return (
                np.zeros((0,), dtype=float),
                [],
                np.zeros((0, 4, 4), dtype=float),
            )

        t_meas = np.array([float(m.t) for m in self.measurements], dtype=float)
        meas_names = [m.sensor.name for m in self.measurements]

        # Use your existing SE(3) multiplication helper
        T_w_obj_meas = np.stack(
            [se3_mul(m.sensor.T_w_s, m.T_s_obj) for m in self.measurements],
            axis=0
        )

        return t_meas, meas_names, T_w_obj_meas


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def animate_trajectory_3d_loop(
        fg: MultiStateFusionFGMeasurementMelding,
        *,
        T_true_list: List[np.ndarray] | None = None,
        interval_ms: int = 50,
        trail: int | None = None,
        meas_history_s: float | None = 0.5,
        # --- Mahalanobis ellipsoid controls ---
        show_mahalanobis_ellipsoid: bool = True,
        conf_3d: float = 0.95,  # confidence for 3D ellipsoid
        cov_damping: float = 1e-9,  # added to (J^T J) diagonal for stability
        cov_clip: float = 1e6,  # clip eigenvalues to avoid blowups
        ellipsoid_u: int = 18,  # mesh resolution (latitude)
        ellipsoid_v: int = 24,  # mesh resolution (longitude)
        elev: float = 30.0,
        azim: float = -60.0,
        show: bool = True) -> None:
    """
    Animate the 3D world trajectory across time after optimization and automatically replay.

    Adds an optional Mahalanobis confidence ellipsoid around the fused position
    using the approximate covariance Sigma ≈ (J^T J)^-1 (translation block only).

    Notes
    -----
    - This is a local (linearized) covariance at the optimum.
    - Ellipsoid is drawn from the 3x3 translation covariance of pose-delta at each state k.
    """
    from matplotlib.animation import FuncAnimation

    # ----------------------------
    # Helper: chi-square quantiles without scipy
    # ----------------------------
    # df=3 approximation: use Wilson–Hilferty transform to invert Chi-square CDF
    # Good accuracy for practical confidence levels.
    # If you later want exact, we can swap to scipy.stats.chi2.ppf when available.
    def chi2_quantile_df3(p: float) -> float:
        p = float(np.clip(p, 1e-12, 1.0 - 1e-12))

        # Approx via inverse normal (Acklam) + Wilson-Hilferty
        # 1) inverse normal
        def inv_norm(u: float) -> float:
            # Peter John Acklam's approximation
            a = [-3.969683028665376e+01, 2.209460984245205e+02,
                 -2.759285104469687e+02, 1.383577518672690e+02,
                 -3.066479806614716e+01, 2.506628277459239e+00]
            b = [-5.447609879822406e+01, 1.615858368580409e+02,
                 -1.556989798598866e+02, 6.680131188771972e+01,
                 -1.328068155288572e+01]
            c = [-7.784894002430293e-03, -3.223964580411365e-01,
                 -2.400758277161838e+00, -2.549732539343734e+00,
                 4.374664141464968e+00, 2.938163982698783e+00]
            d = [7.784695709041462e-03, 3.224671290700398e-01,
                 2.445134137142996e+00, 3.754408661907416e+00]
            plow = 0.02425
            phigh = 1.0 - plow
            if u < plow:
                q = np.sqrt(-2.0 * np.log(u))
                return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                    ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
            if u > phigh:
                q = np.sqrt(-2.0 * np.log(1.0 - u))
                return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                    ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
            q = u - 0.5
            r = q * q
            return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
                (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)

        z = float(inv_norm(p))
        k = 3.0
        # Wilson–Hilferty: X ≈ k * (1 - 2/(9k) + z*sqrt(2/(9k)))^3
        return float(k * (1.0 - 2.0 / (9.0 * k) + z * np.sqrt(2.0 / (9.0 * k))) ** 3)

    chi2_val = chi2_quantile_df3(conf_3d)
    k_ell = float(np.sqrt(chi2_val))  # scale factor on std-dev axes

    # ----------------------------
    # Data
    # ----------------------------
    t_state = fg.times
    pos_est, _ = fg.get_estimated_trajectory()
    t_meas, meas_names, pos_meas = fg.lifted_measurements_world()

    uniq = sorted(set(meas_names))
    markers = ["o", "^", "s", "d", "x", "+", "v", "<", ">"]

    # ----------------------------
    # Covariance per state (translation only)
    # ----------------------------
    cov_xyz_list = None
    if show_mahalanobis_ellipsoid:
        states_opt = fg._diagnostics.get("states_opt", None)
        if states_opt is None:
            raise RuntimeError("Call fg.solve() before animating with covariance.")
        J = fg.numeric_jacobian(states_opt)
        H = J.T @ J
        H = H + float(cov_damping) * np.eye(H.shape[0], dtype=float)
        Sigma = np.linalg.pinv(H)

        N = len(t_state)
        cov_xyz_list = []
        for k in range(N):
            i0 = 6 * k  # pose delta start; translation is i0:i0+3
            cov_xyz = Sigma[i0:i0 + 3, i0:i0 + 3].copy()

            # clip eigenvalues for sanity
            w, V = np.linalg.eigh(cov_xyz)
            w = np.clip(w, 0.0, float(cov_clip))
            cov_xyz = (V * w) @ V.T
            cov_xyz_list.append(cov_xyz)

    # ----------------------------
    # Figure / axes
    # ----------------------------
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Trajectory replay (world)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=elev, azim=azim)

    # --- Sensor locations (static) ---
    sensor_names, sensor_pos = fg.sensor_positions_world()
    sensor_sc = ax.scatter(sensor_pos[:, 0], sensor_pos[:, 1], sensor_pos[:, 2],
                           marker="*", s=180, alpha=0.9, label="sensors")
    for nm, p in zip(sensor_names, sensor_pos):
        ax.text(p[0], p[1], p[2], f" {nm}")

    # Stable axes bounds
    pts = [pos_est, pos_meas, sensor_pos]  # <-- add sensor_pos here

    if T_true_list is not None:
        pos_true = np.stack([T[:3, 3] for T in T_true_list], axis=0)
        pts.append(pos_true)

    allp = np.vstack(pts)
    mins = allp.min(axis=0)
    maxs = allp.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    pad = 0.08 * span

    ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
    ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])
    ax.set_zlim(mins[2] - pad[2], maxs[2] + pad[2])
    ax.set_xlim([32.5, 55.0])
    ax.set_ylim([7.5, 13.0])
    ax.set_zlim([12.0, 35.0])

    # Artists
    (line_est,) = ax.plot([], [], [], linewidth=2, label="estimate")

    meas_scatters = {}
    for i, nm in enumerate(uniq):
        sc = ax.scatter([], [], [], marker=markers[i % len(markers)], label=f"{nm} lifts")
        meas_scatters[nm] = sc

    if T_true_list is not None:
        (line_true,) = ax.plot([], [], [], linewidth=2, linestyle="--", label="truth")
    else:
        line_true = None

    time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)
    ax.legend(loc='lower left')

    # --- Ellipsoid surface artist (we'll recreate each frame and delete the old one) ---
    ell_surf = None

    def _slice_window(k: int) -> slice:
        if trail is None:
            return slice(0, k + 1)
        k0 = max(0, k + 1 - int(trail))
        return slice(k0, k + 1)

    # Precompute unit sphere mesh once
    u = np.linspace(0.0, np.pi, int(ellipsoid_u))
    v = np.linspace(0.0, 2.0 * np.pi, int(ellipsoid_v))
    uu, vv = np.meshgrid(u, v, indexing="ij")
    xs = np.sin(uu) * np.cos(vv)
    ys = np.sin(uu) * np.sin(vv)
    zs = np.cos(uu)
    S = np.stack([xs, ys, zs], axis=0).reshape(3, -1)  # (3, M)

    def init():
        nonlocal ell_surf
        line_est.set_data([], [])
        line_est.set_3d_properties([])
        if line_true is not None:
            line_true.set_data([], [])
            line_true.set_3d_properties([])
        for sc in meas_scatters.values():
            sc._offsets3d = ([], [], [])
        time_text.set_text("")
        if ell_surf is not None:
            ell_surf.remove()
            ell_surf = None
        return (line_est, *(meas_scatters.values()),
                *([line_true] if line_true is not None else []),
                time_text)

    def update(frame_k: int):
        nonlocal ell_surf

        k = int(frame_k)
        win = _slice_window(k)

        # Estimate line
        pe = pos_est[win, :]
        line_est.set_data(pe[:, 0], pe[:, 1])
        line_est.set_3d_properties(pe[:, 2])

        # Truth line
        if line_true is not None and T_true_list is not None:
            pt = pos_true[win, :]
            line_true.set_data(pt[:, 0], pt[:, 1])
            line_true.set_3d_properties(pt[:, 2])

        tcur = float(t_state[k])

        # Sliding measurement window
        if meas_history_s is None:
            t0 = -np.inf
        else:
            t0 = tcur - float(meas_history_s)

        names_arr = np.array(meas_names)
        for nm in uniq:
            idx = np.where((names_arr == nm) & (t_meas > t0) & (t_meas <= tcur + 1e-12))[0]
            if idx.size == 0:
                meas_scatters[nm]._offsets3d = ([], [], [])
            else:
                pm = pos_meas[idx, :]
                meas_scatters[nm]._offsets3d = (pm[:, 0], pm[:, 1], pm[:, 2])

        # Mahalanobis ellipsoid update
        if show_mahalanobis_ellipsoid and cov_xyz_list is not None:
            mu = pos_est[k, :].copy()
            cov_xyz = cov_xyz_list[k]

            # Eigen-decomposition: cov = V diag(w) V^T
            w, V = np.linalg.eigh(cov_xyz)
            w = np.clip(w, 0.0, float(cov_clip))

            # axis lengths at chosen confidence (k_ell * sqrt(eigvals))
            axes = k_ell * np.sqrt(np.maximum(w, 0.0))  # (3,)

            # Transform unit sphere to ellipsoid:
            # E = V diag(axes) * S + mu
            A = V @ np.diag(axes)
            E = (A @ S).reshape(3, xs.shape[0], xs.shape[1])
            X = E[0] + mu[0]
            Y = E[1] + mu[1]
            Z = E[2] + mu[2]

            # delete old and draw new
            if ell_surf is not None:
                ell_surf.remove()
            ell_surf = ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True, alpha=0.18)

        # HUD text
        if meas_history_s is None:
            time_text.set_text(f"t = {tcur:.3f} s  (k={k + 1}/{len(t_state)})  meas: all<=t")
        else:
            time_text.set_text(f"t = {tcur:.3f} s  (k={k + 1}/{len(t_state)})  meas: last {meas_history_s:.2f}s")

        artists = [line_est, *meas_scatters.values(), time_text, sensor_sc]
        if line_true is not None:
            artists.append(line_true)
        if ell_surf is not None:
            artists.append(ell_surf)
        return tuple(artists)

    fig.tight_layout()
    anim = FuncAnimation(
        fig,
        update,
        frames=range(len(t_state)),
        init_func=init,
        interval=int(interval_ms),
        blit=False,
        repeat=True
    )

    from matplotlib.animation import PillowWriter

    # fps is usually 1000/interval_ms
    fps = max(1, int(round(1000.0 / interval_ms)))

    anim.save(
        "trajectory.gif",
        writer=PillowWriter(fps=fps),
        dpi=300,  # bump to 150–200 for higher quality
        savefig_kwargs={"pad_inches": 0}
    )

    # Keep a reference to avoid garbage collection
    fg._diagnostics["_trajectory_anim"] = anim

    plt.tight_layout()
    if show:
        plt.show()


def plot_trajectory_3d(fg: MultiStateFusionFGMeasurementMelding,
                       *,
                       T_true_list: List[np.ndarray] | None = None,
                       show: bool = True) -> None:
    """
    3D plot:
      - fused trajectory (line)
      - measurement world lifts (scatter)
      - optional truth trajectory (line)
    """
    t_meas, names, pos_meas = fg.lifted_measurements_world()
    pos_est, _ = fg.get_estimated_trajectory()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Trajectory (world)")

    ax.plot(pos_est[:, 0], pos_est[:, 1], pos_est[:, 2], linewidth=2, label="estimate")

    # --- Sensor locations (static) ---
    sensor_names, sensor_pos = fg.sensor_positions_world()
    ax.scatter(sensor_pos[:, 0], sensor_pos[:, 1], sensor_pos[:, 2],
               marker="*", s=180, alpha=0.9, label="sensors")

    # Optional: label each one
    for nm, p in zip(sensor_names, sensor_pos):
        ax.text(p[0], p[1], p[2], f" {nm}")

    # Separate per-sensor without hardcoding colors: different markers.
    unique = sorted(set(names))
    markers = ["o", "^", "s", "d", "x", "+", "v", "<", ">"]
    for si, nm in enumerate(unique):
        idx = [i for i, n in enumerate(names) if n == nm]
        p = pos_meas[idx, :]
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], marker=markers[si % len(markers)], label=f"{nm} lifts")

    if T_true_list is not None:
        pos_true = np.stack([T[:3, 3] for T in T_true_list], axis=0)
        ax.plot(pos_true[:, 0], pos_true[:, 1], pos_true[:, 2], linewidth=2, linestyle="--", label="truth")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    plt.tight_layout()
    if show:
        plt.show()


def plot_position_vs_time(
        fg: MultiStateFusionFGMeasurementMelding,
        *,
        T_true_list: List[np.ndarray] | None = None,
        show: bool = True) -> None:
    """
    Position components vs time for estimate, optional truth,
    and all lifted measurements (world frame).
    """

    # Estimated trajectory
    pos_est, _ = fg.get_estimated_trajectory()
    t_state = fg.times

    # Measurements lifted to world
    t_meas, meas_names, pos_meas = fg.lifted_measurements_world()
    uniq = sorted(set(meas_names))
    markers = ["o", "^", "s", "d", "x", "+", "v", "<", ">"]

    fig, axes = plt.subplots(3, 1, sharex=True)
    fig.suptitle("Position vs time")

    labels = ["x", "y", "z"]

    for i in range(3):
        ax = axes[i]

        # --- Estimate line ---
        ax.plot(t_state, pos_est[:, i], label=f"{labels[i]} est")

        # --- Truth (optional) ---
        if T_true_list is not None:
            pos_true = np.stack([T[:3, 3] for T in T_true_list], axis=0)
            ax.plot(t_state, pos_true[:, i], linestyle="--", label=f"{labels[i]} true")

        # --- Measurements (scatter by sensor) ---
        for si, nm in enumerate(uniq):
            idx = [k for k, name in enumerate(meas_names) if name == nm]
            if not idx:
                continue
            ax.scatter(
                t_meas[idx],
                pos_meas[idx, i],
                marker=markers[si % len(markers)],
                s=18,
                alpha=0.7,
                label=f"{labels[i]} {nm} meas"
            )

        ax.set_ylabel(labels[i])
        ax.grid(True)

    axes[-1].set_xlabel("t")

    # Avoid legend explosion: only show one combined legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    plt.tight_layout()
    if show:
        plt.show()


def plot_orientation_error_vs_time(fg: MultiStateFusionFGMeasurementMelding,
                                   *,
                                   T_true_list: List[np.ndarray] | None = None,
                                   show: bool = True) -> None:
    """
    Orientation error angle (rad) vs time (if truth provided).
    """
    if T_true_list is None:
        return
    _, R_est = fg.get_estimated_trajectory()
    t = fg.times

    ang = []
    for k in range(len(T_true_list)):
        R_true = T_true_list[k][:3, :3]
        dR = R_est[k] @ R_true.T
        ang.append(_angle_from_R(dR))
    ang = np.array(ang, dtype=float)

    plt.figure()
    plt.title("Orientation error vs time")
    plt.plot(t, ang)
    plt.xlabel("t")
    plt.ylabel("angle error (rad)")
    plt.tight_layout()
    if show:
        plt.show()


def _as_wxyz(q: np.ndarray) -> np.ndarray:
    """
    Normalize + coerce to (w,x,y,z).
    If you're already wxyz, set this to return q.
    If you're xyzw, swap accordingly.
    """
    q = np.asarray(q, dtype=float).reshape(4)
    # CHANGE THIS LINE IF YOUR CONVENTION IS (x,y,z,w):
    # q = q[[3, 0, 1, 2]]  # xyzw -> wxyz
    q = q / max(1e-12, np.linalg.norm(q))
    return q


def _make_continuous(qs_wxyz: np.ndarray, ref_wxyz: np.ndarray | None = None) -> np.ndarray:
    """
    Flip signs so the sequence is continuous (dot >= 0 with previous).
    Also optionally align overall sign to a reference quaternion (e.g., truth at k=0).
    """
    qs = qs_wxyz.copy()
    if ref_wxyz is not None:
        if np.dot(qs[0], ref_wxyz) < 0:
            qs[0] *= -1.0

    for k in range(1, len(qs)):
        if np.dot(qs[k], qs[k - 1]) < 0:
            qs[k] *= -1.0
    return qs


def plot_quat_s3_projections(q_est_list, q_true_list=None, title="Quaternion projections"):
    """
    q_*_list: iterable of quaternions (either wxyz or xyzw; adjust _as_wxyz)
    Plots:
      - (w,x) in Real–i disk
      - (y,z) in j–k disk
    """
    q_est = np.stack([_as_wxyz(q) for q in q_est_list], axis=0)
    q_true = None
    if q_true_list is not None:
        q_true = np.stack([_as_wxyz(q) for q in q_true_list], axis=0)

    # Make sign continuous (and align est to truth if provided)
    if q_true is not None:
        q_true = _make_continuous(q_true)
        q_est = _make_continuous(q_est, ref_wxyz=q_true[0])
    else:
        q_est = _make_continuous(q_est)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle(title)

    # unit circle for each disk
    th = np.linspace(0, 2 * np.pi, 400)
    cx, cy = np.cos(th), np.sin(th)

    # --- Real–i: (w,x)
    ax1.plot(cx, cy, linewidth=1)
    ax1.set_aspect("equal", "box")
    ax1.set_title("Projection onto Real–i: (w, x)")
    ax1.set_xlabel("w (real)")
    ax1.set_ylabel("x (i)")

    ax1.plot(q_est[:, 0], q_est[:, 1], label="est")
    ax1.scatter(q_est[0, 0], q_est[0, 1], marker="o")  # start
    ax1.scatter(q_est[-1, 0], q_est[-1, 1], marker="x")  # end

    if q_true is not None:
        ax1.plot(q_true[:, 0], q_true[:, 1], linestyle="--", label="true")
        ax1.scatter(q_true[0, 0], q_true[0, 1], marker="o")
        ax1.scatter(q_true[-1, 0], q_true[-1, 1], marker="x")
    ax1.scatter(0, 0, marker='+')
    ax1.legend(loc="lower left")

    # --- j–k: (y,z)
    ax2.plot(cx, cy, linewidth=1)
    ax2.set_aspect("equal", "box")
    ax2.set_title("Projection onto j–k: (y, z)")
    ax2.set_xlabel("y (j)")
    ax2.set_ylabel("z (k)")

    ax2.plot(q_est[:, 2], q_est[:, 3], label="est")
    ax2.scatter(q_est[0, 2], q_est[0, 3], marker="o")
    ax2.scatter(q_est[-1, 2], q_est[-1, 3], marker="x")
    ax2.scatter(0, 0, marker='+')
    if q_true is not None:
        ax2.plot(q_true[:, 2], q_true[:, 3], linestyle="--", label="true")
        ax2.scatter(q_true[0, 2], q_true[0, 3], marker="o")
        ax2.scatter(q_true[-1, 2], q_true[-1, 3], marker="x")

    ax2.legend(loc="lower left")

    plt.tight_layout()
    plt.show()


def hopf_projection_wxyz(q_wxyz: np.ndarray) -> np.ndarray:
    """
    Hopf map S^3 -> S^2 for unit quaternion q=(w,x,y,z).
    Returns a 3D unit vector on S^2.
    """
    w, x, y, z = q_wxyz
    return np.array([
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        w * w + z * z - x * x - y * y
    ], dtype=float)


def _to_wxyz(q) -> np.ndarray:
    for name in ("wxyz", "as_wxyz", "to_wxyz"):
        attr = getattr(q, name, None)
        if callable(attr):
            arr = np.asarray(attr(), dtype=float).reshape(4)
            break
        if attr is not None and not callable(attr):
            arr = np.asarray(attr, dtype=float).reshape(4)
            break
    else:
        arr = np.asarray(q, dtype=float).reshape(4)

    n = float(np.linalg.norm(arr))
    if n <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return arr / n


def quat_to_R_wxyz(q_wxyz: np.ndarray) -> np.ndarray:
    """Rotation matrix from unit quaternion q=(w,x,y,z)."""
    w, x, y, z = q_wxyz
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    return np.array([
        [ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz],
    ], dtype=float)


def lifted_measurements_world_pose(self) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Returns:
      - t_meas: (M,) timestamps
      - names: list[str] sensor name per measurement
      - T_w_obj_meas: (M,4,4) lifted world poses from "lift": T_w_s * T_s_obj
    """
    t_meas = np.array([float(m.t) for m in self.measurements], dtype=float)
    names = [m.sensor.name for m in self.measurements]
    T_w_obj_meas = np.stack(
        [se3_mul(m.sensor.T_w_s, m.T_s_obj) for m in self.measurements],
        axis=0
    )
    return t_meas, names, T_w_obj_meas


def animate_quat_triad_on_s2(
        q_est_list,
        *,
        q_true_list=None,
        interval_ms: int = 50,
        trail: int | None = 80,  # number of frames of tip trail for each axis
        axis_len: float = 1.0,
        elev: float = 20.0,
        azim: float = -60.0,
        save_path: str | None = None,  # "triad.gif" or "triad.mp4"
        dpi: int = 140,
        show: bool = True,
        # ---- NEW: measurement overlay ----
        fg=None,
        meas_history_s: float | None = 0.5,  # seconds; None => show all <= t
) -> None:
    """
    Animate rotated body triad on S² (axis tips on unit sphere).

    Optional: overlay measurement-implied triad tips by lifting each measurement pose
    into world (T_w_obj_meas = T_w_s * T_s_obj_meas), extracting R, and plotting
    R*e_x, R*e_y, R*e_z tips within a sliding time window.
    """

    # ---- precompute rotated axes (estimate / truth) ----
    q_est = np.stack([_to_wxyz(q) for q in q_est_list], axis=0)
    R_est = np.stack([quat_to_R_wxyz(q) for q in q_est], axis=0)  # (N,3,3)
    # columns of R are world-frame images of body axes
    ex_est = axis_len * R_est[:, :, 0]
    ey_est = axis_len * R_est[:, :, 1]
    ez_est = axis_len * R_est[:, :, 2]

    ex_true = ey_true = ez_true = None
    if q_true_list is not None:
        q_true = np.stack([_to_wxyz(q) for q in q_true_list], axis=0)
        R_true = np.stack([quat_to_R_wxyz(q) for q in q_true], axis=0)
        ex_true = axis_len * R_true[:, :, 0]
        ey_true = axis_len * R_true[:, :, 1]
        ez_true = axis_len * R_true[:, :, 2]

    N = ex_est.shape[0]

    def _slice(k: int) -> slice:
        if trail is None:
            return slice(0, k + 1)
        k0 = max(0, k + 1 - int(trail))
        return slice(k0, k + 1)

    # ---- measurement pose lifting (optional) ----
    have_meas = False
    if fg is not None:
        if not hasattr(fg, "times"):
            raise AttributeError("fg must expose fg.times to time-align measurements.")
        if not hasattr(fg, "lifted_measurements_world_pose"):
            raise AttributeError(
                "fg must implement lifted_measurements_world_pose() -> (t_meas, names, T_w_obj_meas). "
                "Add it (T_w_obj = T_w_s @ T_s_obj) to enable measurement triad overlays."
            )

        t_state = np.asarray(fg.times, dtype=float).reshape(-1)
        if len(t_state) != N:
            raise ValueError(f"fg.times length ({len(t_state)}) must match q_est_list length ({N}).")

        t_meas, meas_names, T_w_obj_meas = fg.lifted_measurements_world_pose()
        t_meas = np.asarray(t_meas, dtype=float).reshape(-1)
        T_w_obj_meas = np.asarray(T_w_obj_meas, dtype=float)
        if T_w_obj_meas.ndim != 3 or T_w_obj_meas.shape[1:] != (4, 4):
            raise ValueError("T_w_obj_meas must have shape (M,4,4).")

        R_meas = T_w_obj_meas[:, :3, :3]  # (M,3,3)
        have_meas = True
    else:
        t_state = np.arange(N, dtype=float)

    # ---- figure ----
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Quaternion triad on S² (rotated body axes)")
    ax.view_init(elev=elev, azim=azim)

    # # unit sphere
    # uu = np.linspace(0.0, np.pi, 36)
    # vv = np.linspace(0.0, 2.0*np.pi, 72)
    # U, V = np.meshgrid(uu, vv, indexing="ij")
    # X = np.sin(U) * np.cos(V)
    # Y = np.sin(U) * np.sin(V)
    # Z = np.cos(U)
    # ax.plot_surface(X, Y, Z, linewidth=0, alpha=0.10, antialiased=True)

    ax.set_xlim(-1.05 * axis_len, 1.05 * axis_len)
    ax.set_ylim(-1.05 * axis_len, 1.05 * axis_len)
    ax.set_zlim(-1.05 * axis_len, 1.05 * axis_len)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # triad lines (origin -> axis tip)
    (lx_e,) = ax.plot([], [], [], linewidth=2, label="est x-axis")
    (ly_e,) = ax.plot([], [], [], linewidth=2, label="est y-axis")
    (lz_e,) = ax.plot([], [], [], linewidth=2, label="est z-axis")

    # trails of axis tips
    (tx_e,) = ax.plot([], [], [], linewidth=1, alpha=0.7)
    (ty_e,) = ax.plot([], [], [], linewidth=1, alpha=0.7)
    (tz_e,) = ax.plot([], [], [], linewidth=1, alpha=0.7)

    if ex_true is not None:
        (lx_t,) = ax.plot([], [], [], linewidth=2, linestyle="--", label="true x-axis")
        (ly_t,) = ax.plot([], [], [], linewidth=2, linestyle="--", label="true y-axis")
        (lz_t,) = ax.plot([], [], [], linewidth=2, linestyle="--", label="true z-axis")
        (tx_t,) = ax.plot([], [], [], linewidth=1, linestyle="--", alpha=0.6)
        (ty_t,) = ax.plot([], [], [], linewidth=1, linestyle="--", alpha=0.6)
        (tz_t,) = ax.plot([], [], [], linewidth=1, linestyle="--", alpha=0.6)
    else:
        lx_t = ly_t = lz_t = tx_t = ty_t = tz_t = None

    # ---- NEW: measurement axis-tip scatters (optional) ----
    if have_meas:
        # One cloud per axis; keep it light so it doesn't dominate
        sc_mx = ax.scatter([], [], [], s=12, alpha=0.25, marker="o", label="meas x-tip")
        sc_my = ax.scatter([], [], [], s=12, alpha=0.25, marker="^", label="meas y-tip")
        sc_mz = ax.scatter([], [], [], s=12, alpha=0.25, marker="s", label="meas z-tip")
    else:
        sc_mx = sc_my = sc_mz = None

    hud = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)
    ax.legend(loc="lower left")

    def _set_line_from_origin(line, v):
        line.set_data([0.0, v[0]], [0.0, v[1]])
        line.set_3d_properties([0.0, v[2]])

    def init():
        for ln in [lx_e, ly_e, lz_e, tx_e, ty_e, tz_e, lx_t, ly_t, lz_t, tx_t, ty_t, tz_t]:
            if ln is None:
                continue
            ln.set_data([], [])
            ln.set_3d_properties([])

        if sc_mx is not None:
            sc_mx._offsets3d = ([], [], [])
            sc_my._offsets3d = ([], [], [])
            sc_mz._offsets3d = ([], [], [])

        hud.set_text("")
        return tuple(
            a for a in [
                lx_e, ly_e, lz_e, tx_e, ty_e, tz_e,
                lx_t, ly_t, lz_t, tx_t, ty_t, tz_t,
                sc_mx, sc_my, sc_mz,
                hud
            ] if a is not None
        )

    def update(k: int):
        k = int(k)
        win = _slice(k)

        # estimated triad
        _set_line_from_origin(lx_e, ex_est[k])
        _set_line_from_origin(ly_e, ey_est[k])
        _set_line_from_origin(lz_e, ez_est[k])

        tx = ex_est[win]
        ty = ey_est[win]
        tz = ez_est[win]
        tx_e.set_data(tx[:, 0], tx[:, 1]);
        tx_e.set_3d_properties(tx[:, 2])
        ty_e.set_data(ty[:, 0], ty[:, 1]);
        ty_e.set_3d_properties(ty[:, 2])
        tz_e.set_data(tz[:, 0], tz[:, 1]);
        tz_e.set_3d_properties(tz[:, 2])

        # true triad
        if ex_true is not None:
            _set_line_from_origin(lx_t, ex_true[k])
            _set_line_from_origin(ly_t, ey_true[k])
            _set_line_from_origin(lz_t, ez_true[k])

            tx2 = ex_true[win]
            ty2 = ey_true[win]
            tz2 = ez_true[win]
            tx_t.set_data(tx2[:, 0], tx2[:, 1])
            tx_t.set_3d_properties(tx2[:, 2])
            ty_t.set_data(ty2[:, 0], ty2[:, 1])
            ty_t.set_3d_properties(ty2[:, 2])
            tz_t.set_data(tz2[:, 0], tz2[:, 1])
            tz_t.set_3d_properties(tz2[:, 2])

        # ---- NEW: measurement axis-tip overlay ----
        if sc_mx is not None:
            tcur = float(t_state[k])
            if meas_history_s is None:
                mask = (t_meas <= tcur + 1e-12)
            else:
                t0 = tcur - float(meas_history_s)
                mask = (t_meas > t0) & (t_meas <= tcur + 1e-12)

            idx = np.where(mask)[0]
            if idx.size == 0:
                sc_mx._offsets3d = ([], [], [])
                sc_my._offsets3d = ([], [], [])
                sc_mz._offsets3d = ([], [], [])
            else:
                # Compute tips: axis_len * (R_meas @ e_i)
                Rw = R_meas[idx, :, :]  # (K,3,3)

                exm = axis_len * Rw[:, :, 0]  # (K,3)
                eym = axis_len * Rw[:, :, 1]
                ezm = axis_len * Rw[:, :, 2]

                sc_mx._offsets3d = (exm[:, 0], exm[:, 1], exm[:, 2])
                sc_my._offsets3d = (eym[:, 0], eym[:, 1], eym[:, 2])
                sc_mz._offsets3d = (ezm[:, 0], ezm[:, 1], ezm[:, 2])

        hud.set_text(f"k={k + 1}/{N}")
        return tuple(
            a for a in [
                lx_e, ly_e, lz_e, tx_e, ty_e, tz_e,
                lx_t, ly_t, lz_t, tx_t, ty_t, tz_t,
                sc_mx, sc_my, sc_mz,
                hud
            ] if a is not None
        )

    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(fig, update, frames=range(N), init_func=init,
                         interval=int(interval_ms), blit=False, repeat=True)

    if save_path is not None:
        fps = max(1, int(round(1000.0 / interval_ms)))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        if save_path.lower().endswith(".gif"):
            from matplotlib.animation import PillowWriter
            anim.save(save_path, writer=PillowWriter(fps=fps), dpi=dpi,
                      savefig_kwargs={"pad_inches": 0.02})
        elif save_path.lower().endswith(".mp4"):
            from matplotlib.animation import FFMpegWriter
            anim.save(save_path, writer=FFMpegWriter(fps=fps, bitrate=4000), dpi=dpi,
                      savefig_kwargs={"pad_inches": 0.02})
        else:
            raise ValueError("save_path must end with .gif or .mp4")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_all_time_diagnostics(fg: MultiStateFusionFGMeasurementMelding,
                              *,
                              T_true_list: List[np.ndarray] | None = None,
                              show: bool = True) -> None:
    plot_trajectory_3d(fg, T_true_list=T_true_list, show=False)
    plot_position_vs_time(fg, T_true_list=T_true_list, show=False)
    plot_orientation_error_vs_time(fg, T_true_list=T_true_list, show=False)
    if show:
        plt.show()


# -----------------------------------------------------------------------------
# Synthetic test harness (mirrors the feel of StaticFGMeasurementMelding)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class SensorSpec:
    name: str
    mean_loc_w: np.ndarray
    yaw_deg: float = 0.0
    pos_std_s: np.ndarray = None
    rod_std: np.ndarray = None
    loc_jitter_std: float = 0.0


def _make_sensor_quat(yaw_deg: float, randomize: bool, rot_jitter_deg: float = 3.0) -> q:
    yaw_q = q().from_eulerD_rpy(np.array([0.0, 0.0, yaw_deg]))
    if not randomize:
        return yaw_q
    return yaw_q * random_quat_within_deg(rot_jitter_deg)


def _true_pose_in_sensor(obj_quat: q, w_t_obj: np.ndarray, s_quat: q, w_t_s: np.ndarray) -> np.ndarray:
    """Return true T_s_obj."""
    q_s_obj = s_quat.T * obj_quat
    t_s_obj = s_quat.T * (w_t_obj - w_t_s)
    return q_s_obj.to_SE3_given_position(t_s_obj)


def _noisy_measurement_from_true(T_s_obj_true: np.ndarray, pos_std: np.ndarray, rod_std: np.ndarray) -> np.ndarray:
    true_q, true_t = quat_and_t_from_SE3(T_s_obj_true)
    est_q = true_q.perturb_from_rodrigues_std(rod_std)
    est_t = true_t + pos_std * np.random.randn(3)
    return est_q.to_SE3_given_position(est_t)


def generate_synthetic_trajectory(*,
                                  randomize: bool = True,
                                  N: int = 25,
                                  dt: float = 0.2) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Returns:
      - times: (N,)
      - T_true_list: list[SE3] length N
      - v_true: (3,)
      - w_true: (3,)
    """
    if q is None:
        raise ImportError('Synthetic helpers require support.mathHelpers.quaternions; q is None in this environment.')

    times = np.arange(N, dtype=float) * float(dt)

    accel = np.array([0.05, 0.00, 0.00])

    if randomize:
        t0 = np.array([10.0, 0.0, 0.0], dtype=float) + 2.0 * np.random.randn(3)
        v_true = np.array([0.0, 0.6, 0.0], dtype=float) + 0.05 * np.random.randn(3)
        w_true = np.array([0.0, 0.0, 0.18], dtype=float) + 0.02 * np.random.randn(3)
        q0 = random_quat_within_deg(5.0)
    else:
        t0 = np.array([10.0, 0.0, 0.0], dtype=float)
        v_true = np.array([0.0, 0.6, 0.0], dtype=float)
        w_true = np.array([0.0, 0.0, 0.18], dtype=float)
        q0 = q()

    T_true_list: List[np.ndarray] = []
    R = q0.to_dcm()
    t = t0.copy()
    for k in range(N):
        T_true_list.append(se3_from_Rt(R, t))
        # propagate
        t = t + v_true * dt + accel * dt * dt / 2.0
        v_true = v_true + accel * dt
        R = so3_exp(w_true * dt) @ R

    return times, T_true_list, v_true, w_true


def generate_synthetic_measurements(times: np.ndarray,
                                    T_true_list: List[np.ndarray],
                                    *,
                                    randomize: bool = True) -> Tuple[List[TimedMeasurement], Dict[str, Sensor]]:
    """
    Build several static sensors and create timestamped relative pose measurements.
    """
    if q is None:
        raise ImportError('Synthetic helpers require support.mathHelpers.quaternions; q is None in this environment.')

    sensors: List[SensorSpec] = [
        SensorSpec("EO", mean_loc_w=np.array([0.0, 0.0, 3.0]), yaw_deg=0.0,
                   pos_std_s=np.array([0.5, 0.1, 0.1]), rod_std=np.array([0.1, 0.05, 0.05]), loc_jitter_std=0.5),
        SensorSpec("LR", mean_loc_w=np.array([0.0, -3.0, 3.0]), yaw_deg=40.0,
                   pos_std_s=np.array([0.1, 0.5, 0.5]), rod_std=np.array([0.05, 0.1, 0.1]), loc_jitter_std=0.5),
        SensorSpec("MW", mean_loc_w=np.array([0.0, 1.0, 3.0]), yaw_deg=-10.0,
                   pos_std_s=np.array([0.5, 0.5, 0.5]), rod_std=np.array([0.50, 0.50, 0.50]), loc_jitter_std=0.5),
    ]

    sensor_objs: Dict[str, Sensor] = {}
    for spec in sensors:
        w_t_s = spec.mean_loc_w + (spec.loc_jitter_std * np.random.randn(3) if randomize else 0.0)
        s_quat = _make_sensor_quat(spec.yaw_deg, randomize=randomize)
        T_w_s = s_quat.to_SE3_given_position(w_t_s)
        sensor_objs[spec.name] = Sensor(spec.name, T_w_s=T_w_s, cov_model=rle_cov(spec.pos_std_s, spec.rod_std))

    measurements: List[TimedMeasurement] = []
    # Correct measurement generation using SensorSpecs (keeps per-sensor noise)
    for k, tk in enumerate(times):
        T_w_obj_true = T_true_list[k]
        obj_quat, w_t_obj = quat_and_t_from_SE3(T_w_obj_true)

        for spec in sensors:
            s = sensor_objs[spec.name]
            s_quat, w_t_s = quat_and_t_from_SE3(s.T_w_s)

            T_s_obj_true = _true_pose_in_sensor(obj_quat=obj_quat, w_t_obj=w_t_obj, s_quat=s_quat, w_t_s=w_t_s)
            if randomize:
                T_s_obj_meas = _noisy_measurement_from_true(T_s_obj_true, spec.pos_std_s, spec.rod_std)
            else:
                T_s_obj_meas = T_s_obj_true

            measurements.append(
                TimedMeasurement(sensor=s, t=float(tk), T_s_obj=T_s_obj_meas, meta={"sensor": spec.name}))

    return measurements, sensor_objs

class SE3:
    def __init__(self, R, t):
        self.R = R
        self.t = t

    def __str__(self):
        return str(self.array)

    def __mul__(self, other):
        return SE3(self.R @ other.R, self.R @ other.t + self.t)

    @property
    def array(self):
        array = np.identity(4)
        array[:3, :3] = self.R
        array[:3, 3] = self.t
        return array

    @property
    def inv(self):
        return SE3(self.R.T, self.R.T @ -self.t)

    @staticmethod
    def from_array(array: np.typing.NDArray):
        return SE3(array[:3, :3], array[:3, 3])

def run_test(randomize: bool = True) -> None:
    np.random.seed(123)
    times, T_true_list, v_true, w_true = generate_synthetic_trajectory(randomize=randomize, N=20, dt=0.2)
    measurements, _ = generate_synthetic_measurements(times, T_true_list, randomize=randomize)
    print(measurements)
    fg = MultiStateFusionFGMeasurementMelding(
        measurements,
        process_noise=ProcessNoise(sig_pos=0.05, sig_rot=0.03, sig_vel=0.05, sig_omega=0.08),
    )

    states_opt, info = fg.solve(max_iters=15, verbose=True, eps_pose_t=1e-4, eps_pose_r=1e-5, eps_v=1e-4, eps_w=1e-5)
    print(info)

    pos_est, dcm_est = fg.get_estimated_trajectory()
    q_true = se3s2quats(T_true_list)
    q_est = mats2quats(dcm_est)
    animate_quat_triad_on_s2(q_est, q_true_list=q_true,
                             save_path="Axis.gif",
                             fg=fg,
                             trail=120, elev=25, azim=-60)
    plot_quat_s3_projections(q_est, q_true_list=q_true)

    animate_trajectory_3d_loop(fg, T_true_list=T_true_list,
                               interval_ms=50,
                               trail=30)
    plot_all_time_diagnostics(fg, T_true_list=T_true_list, show=True)


def kai_generate_cameras():
    pos_std_s = np.array([1.5, 0.5, 0.5])
    rod_std_s = np.array([0.05, 0.05, 0.05])

    cam1_SE3 = np.array([[-0.258819044, -0.000000084, -0.965925813, 40.137699127],
                         [ 0.000000000, -1.000000000,  0.000000087, 10.356100082],
                         [-0.965925694,  0.000000023,  0.258819044, 30.799999237],
                         [ 0.000000000,  0.000000000,  0.000000000,  1.000000000]])

    cam1_SE3 = SE3.from_array(cam1_SE3)

    cam2_SE3 = np.array([[-0.906307757, -0.000000087, -0.422618270, 50.030502319],
                         [ 0.000000079, -1.000000000,  0.000000037,  9.970000267],
                         [-0.422618270,  0.000000000,  0.906307757, 20.803199768],
                         [ 0.000000000,  0.000000000,  0.000000000,  1.000000000]])
    cam2_SE3 = SE3.from_array(cam2_SE3)

    cam1 = Sensor('cam1', T_w_s=cam1_SE3.array, cov_model=rle_cov(pos_std_s, rod_std_s))
    cam2 = Sensor('cam2', T_w_s=cam2_SE3.array, cov_model=rle_cov(pos_std_s, rod_std_s))
    from support.dev.makeCSV import parse_data
    times, is_tanker, estimated, truth, error = parse_data()

    start_time = times[0]
    times = [(time - start_time) / 1000.0 for time in times]
    measurements: List[TimedMeasurement] = []

    max_num = 200

    for idx, (time, is_tank, est, tru) in enumerate(zip(times, is_tanker, estimated, truth)):
        if idx > max_num:
            break
        if not is_tank:
            est_SE3 = SE3.from_array(est)
            est_camFrame = cam2_SE3.inv * est_SE3
            measurements.append(TimedMeasurement(sensor=cam2,
                                                 t=time,
                                                 T_s_obj=est_camFrame.array,
                                                 meta={"sensor" : "cam2"}))
        else:
            est_SE3 = SE3.from_array(est)
            est_camFrame = cam1_SE3.inv * est_SE3
            measurements.append(TimedMeasurement(sensor=cam1,
                                                 t=time,
                                                 T_s_obj=est_camFrame.array,
                                                 meta={"sensor" : "cam1"}))


    fg = MultiStateFusionFGMeasurementMelding(
        measurements,
        process_noise=ProcessNoise(sig_pos=0.1, sig_rot=0.03, sig_vel=0.5, sig_omega=0.08),
    )

    states_opt, info = fg.solve(max_iters=15, verbose=True, eps_pose_t=1e-4, eps_pose_r=1e-5, eps_v=1e-4, eps_w=1e-5)
    print(info)

    pos_est, dcm_est = fg.get_estimated_trajectory()
    q_true = se3s2quats(truth)
    q_est = mats2quats(dcm_est)
    animate_quat_triad_on_s2(q_est, q_true_list=q_true,
                             save_path="Axis.gif",
                             fg=fg,
                             trail=120, elev=25, azim=-60)
    plot_quat_s3_projections(q_est, q_true_list=q_true)

    animate_trajectory_3d_loop(fg, T_true_list=truth,
                               interval_ms=50,
                               trail=30)
    truth_win = truth[:len(fg.times)]
    plot_all_time_diagnostics(fg, T_true_list=truth_win, show=True)


def main(test: bool = True) -> None:
    if test:
        run_test(randomize=True)

    #  Kai Data
    else:
        sensors = kai_generate_cameras()


if __name__ == "__main__":
    main(test=False)
