from __future__ import annotations
import numpy as np
import math
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Callable, Any, Dict, Tuple, List
import matplotlib.pyplot as plt
from support.mathHelpers.quaternions import Quaternion as q, random_quat_within_deg

pg_brk = '\n' + '~' * 95 + '\n'
np.random.seed(123)
np.set_printoptions(suppress=True, precision=5)


# =============================================================================
# StaticFGMeasurementMelding.py
#
# Purpose
# -------
# A lightweight, dependency-free, factor-graph-style sensor fusion example for
# estimating a *static* drogue pose in SE(3).
#
# Multiple heterogeneous sensors (e.g., EO, LIDAR) each provide:
#   - an extrinsic calibration (sensor pose in world), and
#   - a relative drogue pose measurement in their own sensor frame,
#   - with an associated 6x6 covariance model.
#
# The solver estimates a single world-frame drogue pose T_w_drg by minimizing
# the sum of squared, *whitened* pose residuals using a Levenberg–Marquardt loop.
#
# Design goals
# ------------
# - Explicit math (no black-box solvers)
# - Easy experimentation with covariance models
# - Clear frame conventions to avoid sign / order bugs
# - Useful diagnostics for debugging sensor weighting and geometry
#
# Key conventions (read once; prevents frame bugs)
# -----------------------------------------------
# - T_ab maps coordinates from frame b into frame a:
#
#       x_a = T_ab @ x_b
#
# - Camera extrinsics are "sensor pose in world":
#
#       eo_camPose == T_w_eo   (EO -> world)
#       lr_camPose == T_w_lr
#
# - Measurements are "drogue pose in sensor frame":
#
#       eo_drgPose == T_eo_drg (DRG -> EO)
#       lr_drgPose == T_lr_drg
#
# - The estimated state is:
#
#       T_w_drg  (DRG -> world)
#
# Typical usage
# -------------
#   eo = Sensor("EO", T_w_s=eo_camPose, cov_model=...)
#   lr = Sensor("LR", T_w_s=lr_camPose, cov_model=...)
#
#   meas = [
#       Measurement(eo, eo_drgPose),
#       Measurement(lr, lr_drgPose),
#   ]
#
#   fg = StaticFGMeasurementMelding(meas)
#   fg.solve(max_iters=25)
#   fg.print_summary()
#   plot_all_fg_diagnostics(fg)
#
# =============================================================================

# -----------------------------------------------------------------------------
# ALL Plotting Functions for post-meld display
# -----------------------------------------------------------------------------

def _require_diag(fg):
    d = getattr(fg, "_diagnostics", {}) or {}
    T_opt = d.get("T_w_obj_opt", None)
    if T_opt is None:
        T_opt = d.get("T_w_drg_opt", None)
    if T_opt is None:
        raise RuntimeError("No optimized pose found. Call fg.solve(...) before plotting.")
    return d, T_opt


def _factor_pred_and_residual(m, T_w_obj):
    # Predicted measurement in sensor frame: T_s_obj_pred = inv(T_w_s) * T_w_obj
    T_s_obj_pred = se3_mul(se3_inv(m.sensor.T_w_s), T_w_obj)
    r = se3_residual(T_s_obj_pred, m.T_s_obj)  # (6,)
    return T_s_obj_pred, r


def plot_sigma_residuals_per_sensor(fg, *, use_opt: bool = True, show: bool = True):
    """
    Bar plot of z = W r (sigma units) for each sensor.
    One figure per sensor.

    Requires:
      - fg.solve() already called (for T_opt)
      - fg.W exists and aligned with fg.measurements
    """
    d, T_opt = _require_diag(fg)
    T_use = T_opt if use_opt else d.get("T_w_obj_init", None) or d.get("T_w_drg_init", None) or T_opt

    labels = ["dx", "dy", "dz", "dθx", "dθy", "dθz"]

    for i, (m, W) in enumerate(zip(fg.measurements, fg.W)):
        _, r = _factor_pred_and_residual(m, T_use)
        z = np.abs(W @ r)

        plt.figure()
        plt.title(f"{m.sensor.name}: sigma-unit residual z = W r")
        plt.bar(np.arange(6), z)
        plt.xticks(np.arange(6), labels)
        plt.axhline(0.0)
        plt.ylabel("sigma units")
        plt.tight_layout()

    if show:
        plt.show()


def plot_chi2_per_sensor(fg, *, use_opt: bool = True, normalize_by_dof: bool = True, show: bool = True):
    """
    Bar plot of per-sensor chi^2 contribution:
        chi2_i = ||W_i r_i||^2

    If normalize_by_dof=True, divides by 6 (each pose factor has 6 residual components).
    """
    d, T_opt = _require_diag(fg)
    T_use = T_opt if use_opt else d.get("T_w_obj_init", None) or d.get("T_w_drg_init", None) or T_opt

    names = []
    chi2s = []

    for i, (m, W) in enumerate(zip(fg.measurements, fg.W)):
        _, r = _factor_pred_and_residual(m, T_use)
        z = W @ r
        chi2 = float(z @ z)
        if normalize_by_dof:
            chi2 /= 6.0
        names.append(m.sensor.name)
        chi2s.append(chi2)

    plt.figure()
    plt.title("Per-sensor chi² contribution" + (" (normalized by 6)" if normalize_by_dof else ""))
    plt.bar(np.arange(len(names)), chi2s)
    plt.xticks(np.arange(len(names)), names)
    plt.ylabel("chi²" + ("/6" if normalize_by_dof else ""))
    plt.tight_layout()

    if show:
        plt.show()


def _plot_cov_ellipse_2d(C2, mean2, *, title: str, xlabel: str, ylabel: str, nsig: float = 2.0):
    """
    Plot a 2D covariance ellipse for a 2x2 covariance matrix.
    """
    C2 = np.asarray(C2, dtype=float)
    mean2 = np.asarray(mean2, dtype=float).reshape(2)

    # Eigen-decomposition
    vals, vecs = np.linalg.eigh(C2)
    vals = np.maximum(vals, 0.0)

    # Parametric ellipse in eigenbasis
    t = np.linspace(0, 2 * np.pi, 200)
    circle = np.vstack([np.cos(t), np.sin(t)])  # (2,N)
    radii = nsig * np.sqrt(vals)  # (2,)
    ellipse = (vecs @ (radii[:, None] * circle)) + mean2[:, None]  # (2,N)

    plt.figure()
    plt.title(title)
    plt.plot(mean2[0], mean2[1], marker="o")  # mean
    plt.plot(ellipse[0, :], ellipse[1, :])  # ellipse
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axis("equal")
    plt.tight_layout()


def _ellipsoid_mesh_from_cov(C3, center, nsig=2.0, n_u=40, n_v=20):
    """
    Build an ellipsoid mesh for the 3x3 covariance C3 centered at `center`.

    Ellipsoid is the nsig-sigma surface:
        (x-center)^T C^{-1} (x-center) = nsig^2
    """
    C3 = np.asarray(C3, dtype=float)
    center = np.asarray(center, dtype=float).reshape(3)

    # Eigen-decomposition (C3 should be SPD; clamp tiny negatives from numeric noise)
    vals, vecs = np.linalg.eigh(C3)
    vals = np.maximum(vals, 0.0)

    # Radii along principal axes
    radii = nsig * np.sqrt(vals)  # (3,)

    # Parametric unit sphere
    u = np.linspace(0.0, 2.0 * np.pi, n_u)
    v = np.linspace(0.0, np.pi, n_v)
    uu, vv = np.meshgrid(u, v)

    xs = np.cos(uu) * np.sin(vv)
    ys = np.sin(uu) * np.sin(vv)
    zs = np.cos(vv)

    # Scale sphere -> axis-aligned ellipsoid
    E = np.stack([xs * radii[0], ys * radii[1], zs * radii[2]], axis=0)  # (3, n_v, n_u)

    # Rotate into world basis and translate
    # vecs maps eigenbasis -> world basis
    E_flat = E.reshape(3, -1)
    P_flat = (vecs @ E_flat) + center[:, None]
    P = P_flat.reshape(3, *xs.shape)

    X, Y, Z = P[0], P[1], P[2]
    return X, Y, Z


def plot_translation_ellipsoids_3d(
        fg,
        *,
        nsig: float = 2.0,
        show_sensor_ellipsoids: bool = True,
        show_fused_ellipsoid: bool = True,
        show_sensor_means: bool = True,
        show_fused_mean: bool = True,
        show: bool = True,
):
    """
    Single 3D plot with translation uncertainty ellipsoids:
      - One ellipsoid per sensor "lift" (T_w_drg_i = T_w_s_i * T_s_drg_meas_i)
        using that sensor's translation covariance rotated into world.
      - One ellipsoid for the fused estimate using fg._diagnostics["cov6"][:3,:3].

    Notes / assumptions
    -------------------
    - Uses translation covariance only (3x3). Rotation uncertainty ellipsoids are not shown.
    - Assumes each measurement covariance translation block is expressed in the *sensor frame*
      for dt, so we rotate into world with R_w_s:
          C_tt_world = R_w_s * C_tt_sensor * R_w_s^T
      This is consistent with many SE(3) residual conventions where translation residual is in
      the measurement frame.

    Requirements
    ------------
    - fg.measurements : list of Measurement (with sensor.T_w_s and measurement.T_s_obj)
    - fg._diagnostics contains "T_w_obj_opt" (or "T_w_drg_opt") and "cov6"
    - se3_mul(...) exists (same module) and returns 4x4
    """
    d = getattr(fg, "_diagnostics", {}) or {}

    T_opt = d.get("T_w_obj_opt", None)
    if T_opt is None:
        T_opt = d.get("T_w_drg_opt", None)
    if T_opt is None:
        raise RuntimeError("No optimized pose found. Call fg.solve(...) before plotting.")

    cov6 = d.get("cov6", None)
    if cov6 is None:
        raise RuntimeError("No posterior covariance found. Ensure fg.solve() computed cov6.")

    cov6 = np.asarray(cov6, dtype=float)
    C_fused = cov6[:3, :3]
    mu_fused = np.asarray(T_opt[:3, 3], dtype=float)

    # Optional truth (support either name)
    T_true = getattr(fg, "T_w_obj_true", None)
    if T_true is None:
        T_true = getattr(fg, "T_w_drg_true", None)
    if T_true is None:
        # Also allow it to be in diagnostics
        T_true = d.get("T_w_obj_true", None) or d.get("T_w_drg_true", None)

    mu_true = None
    if T_true is not None:
        mu_true = np.asarray(T_true[:3, 3], dtype=float)

    # Build per-sensor lifts and world-frame translation covariances
    lifts = d.get("T_w_obj_lifts", None)
    if lifts is None:
        lifts = [se3_mul(m.sensor.T_w_s, m.T_s_obj) for m in fg.measurements]

    sensor_mus = []
    sensor_covs_world = []
    sensor_names = []

    for m, T_w_drg_i in zip(fg.measurements, lifts):
        mu_i = np.asarray(T_w_drg_i[:3, 3], dtype=float)

        # Translation covariance from measurement model (sensor frame)
        C_i = np.asarray(m.covariance(), dtype=float)
        C_tt_s = C_i[:3, :3]

        # Rotate translation covariance into world
        R_w_s = np.asarray(m.sensor.T_w_s[:3, :3], dtype=float)
        C_tt_w = R_w_s @ C_tt_s @ R_w_s.T

        sensor_mus.append(mu_i)
        sensor_covs_world.append(C_tt_w)
        sensor_names.append(m.sensor.name)

    # --- Plot ---
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"Translation uncertainty ellipsoids (±{nsig}σ)")

    # Avoid specifying colors; instead differentiate via linestyle/linewidth and labels
    sensor_colors = {
        "EO": "#1f77b4",  # blue
        "LR": "#ff7f0e",  # orange
        "MW": "#2ca02c",  # green
    }
    fused_color = "#d62728"  # red

    if show_sensor_ellipsoids:
        for name, mu_i, Cw_i in zip(sensor_names, sensor_mus, sensor_covs_world):
            X, Y, Z = _ellipsoid_mesh_from_cov(Cw_i, mu_i, nsig=nsig)
            ax.plot_wireframe(
                X, Y, Z,
                rstride=2, cstride=2,
                linewidth=1.2,
                color=sensor_colors.get(name, "gray"),
                label=f"{name} lift ±{nsig}σ"
            )

    if show_fused_ellipsoid:
        X, Y, Z = _ellipsoid_mesh_from_cov(C_fused, mu_fused, nsig=nsig)
        ax.plot_wireframe(
            X, Y, Z,
            rstride=2, cstride=2,
            linewidth=2.5,
            color=fused_color,
            label=f"Fused ±{nsig}σ"
        )

    if show_sensor_means:
        for name, mu_i in zip(sensor_names, sensor_mus):
            ax.scatter(*mu_i, color=sensor_colors.get(name, "gray"), marker="o")
            ax.text(*mu_i, f" {name} lift")

    if show_fused_mean:
        ax.scatter(*mu_fused, color=fused_color, marker="x", s=60)
        ax.text(*mu_fused, " fused")

    if mu_true is not None:
        truth_label = "Truth"
        ax.scatter(mu_true[0], mu_true[1], mu_true[2], marker="*", s=80, label=truth_label)
        ax.text(mu_true[0], mu_true[1], mu_true[2], truth_label)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    plt.tight_layout()

    if show:
        plt.show()


def plot_hessian_eigs(fg, *, eps_t: float | None = None, eps_r: float | None = None,
                      use_opt: bool = True, show: bool = True):
    """
    Plot the eigenvalue spectrum of H = J^T J (semilog y).

    Uses fg._numeric_jacobian(...) and the same eps_t/eps_r you used for solve(),
    unless overridden.
    """
    d, T_opt = _require_diag(fg)
    T_use = T_opt if use_opt else d.get("T_w_obj_init", None) or d.get("T_w_drg_init", None) or T_opt

    # Pull eps from lm_info if available
    lm_info = d.get("lm_info", {}) if isinstance(d.get("lm_info", None), dict) else {}
    if eps_t is None:
        eps_t = float(lm_info.get("eps_t", 1e-4))
    if eps_r is None:
        eps_r = float(lm_info.get("eps_r", 1e-5))

    J = fg.numeric_jacobian(T_use, eps_t=eps_t, eps_r=eps_r)
    H = J.T @ J
    vals = np.linalg.eigvalsh(H)
    vals = np.sort(vals)

    plt.figure()
    plt.title("Hessian spectrum (eigvals of H = JᵀJ)")
    plt.semilogy(np.arange(1, vals.size + 1), np.maximum(vals, 1e-300), marker="o")
    plt.xlabel("eigenvalue index")
    plt.ylabel("eigenvalue (log scale)")
    plt.tight_layout()

    if show:
        plt.show()


def plot_world_geometry_topdown(fg, *, show_truth: bool = True, show_lifts: bool = True,
                                use_opt: bool = True, show: bool = True):
    """
    Simple XY top-down geometry sketch:
      - sensor positions (world)
      - estimated drogue position (world)
      - optional true drogue position (world)
      - optional per-measurement world lifts (world)
      - rays from sensor -> (est) and sensor -> (lift)
    """
    d, T_opt = _require_diag(fg)
    T_use = T_opt if use_opt else d.get("T_w_obj_init", None) or d.get("T_w_drg_init", None) or T_opt

    # Extract sensor positions and names
    s_xy = []
    names = []
    for m in fg.measurements:
        t = np.asarray(m.sensor.T_w_s[:3, 3], dtype=float)
        s_xy.append(t[:2])
        names.append(m.sensor.name)
    s_xy = np.asarray(s_xy)

    # Estimated drogue position
    t_est = np.asarray(T_use[:3, 3], dtype=float)
    est_xy = t_est[:2]

    # True drogue position (optional)
    T_true = getattr(fg, "T_w_obj_true", None)
    true_xy = None
    if show_truth and T_true is not None:
        true_xy = np.asarray(T_true[:3, 3], dtype=float)[:2]

    # Lifts (optional)
    lifts = d.get("T_w_obj_lifts", None)
    if lifts is None:
        lifts = [se3_mul(m.sensor.T_w_s, m.T_s_obj) for m in fg.measurements]
    lift_xy = np.asarray([np.asarray(T[:3, 3], dtype=float)[:2] for T in lifts])

    plt.figure()
    plt.title("Top-down geometry (XY)")

    # Sensors
    plt.plot(s_xy[:, 0], s_xy[:, 1], marker="o", linestyle="None", label="Sensors")
    for (x, y), nm in zip(s_xy, names):
        plt.text(x, y, f" {nm}")

    # Estimated
    plt.plot(est_xy[0], est_xy[1], marker="x", label=("Estimate (opt)" if use_opt else "Estimate (init)"))
    plt.text(est_xy[0], est_xy[1], " EST")

    # Truth
    if true_xy is not None:
        plt.plot(true_xy[0], true_xy[1], marker="*", label="Truth")
        plt.text(true_xy[0], true_xy[1], " TRUE")

    # Lifts
    if show_lifts:
        plt.plot(lift_xy[:, 0], lift_xy[:, 1], marker=".", linestyle="None", label="Per-sensor lifts")
        for (x, y), nm in zip(lift_xy, names):
            plt.text(x, y, f" lift({nm})")

    # Rays to estimate
    for (sx, sy) in s_xy:
        plt.plot([sx, est_xy[0]], [sy, est_xy[1]], linewidth=1)

    # Rays to lifts
    if show_lifts:
        for (sx, sy), (lx, ly) in zip(s_xy, lift_xy):
            plt.plot([sx, lx], [sy, ly], linewidth=1)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()

    if show:
        plt.show()


# -------------
# Convenience runner: call this after fg.solve(...)
# -------------
def plot_all_fg_diagnostics(fg, *, show: bool = False):
    """
    Runs the full plotting suite discussed:
      1) sigma-unit residuals per sensor
      2) per-sensor chi² bar plot
      3) posterior translation covariance ellipses (XY, XZ, YZ)
      4) Hessian eigenvalue spectrum
      5) top-down world geometry (XY)
    """
    plot_sigma_residuals_per_sensor(fg, show=show)
    plot_chi2_per_sensor(fg, show=show)
    plot_translation_ellipsoids_3d(fg, show=show)
    plot_hessian_eigs(fg, show=show)
    plot_world_geometry_topdown(fg, show=show)
    if not show:
        plt.show()


# -----------------------------------------------------------------------------
# SE(3) / SO(3) helpers (T_ab maps b -> a, i.e., x_a = T_ab @ x_b)
# -----------------------------------------------------------------------------
def se3_mul(T_ab: NDArray[np.float64], T_bc: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compose two SE(3) transforms using the project's frame convention.

    Convention
    ----------
    T_ab maps coordinates from frame b into frame a:

        x_a = T_ab @ x_b

    Therefore, the composition rule is:

        T_ac = T_ab * T_bc

    Parameters
    ----------
    T_ab : (4,4) ndarray
        Homogeneous transform mapping b -> a.
    T_bc : (4,4) ndarray
        Homogeneous transform mapping c -> b.

    Returns
    -------
    T_ac : (4,4) ndarray
        Homogeneous transform mapping c -> a.

    Notes
    -----
    Inputs are expected to be valid homogeneous transforms with bottom row
    [0, 0, 0, 1]."""
    return T_ab @ T_bc


def se3_inv(T_ab: NDArray[np.float64]) -> NDArray[np.float64]:
    """Invert an SE(3) transform.

    Given T_ab (mapping b -> a), return T_ba (mapping a -> b).

    Parameters
    ----------
    T_ab : (4,4) ndarray
        Homogeneous transform mapping b -> a.

    Returns
    -------
    T_ba : (4,4) ndarray
        Homogeneous transform mapping a -> b.

    Notes
    -----
    Uses the SE(3) inverse:

    R_ba = R_ab^T
    t_ba = -R_ab^T * t_ab"""
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
    """Exponential map for SO(3): so(3) -> SO(3).

    Parameters
    ----------
    w : (3,) ndarray
        Rotation vector in axis-angle (Lie algebra) form, in radians.

    Returns
    -------
    R : (3,3) ndarray
        Rotation matrix corresponding to Exp(w).

    Notes
    -----
    Uses Rodrigues' formula. For very small ||w||, a first-order approximation
    is used to avoid numerical issues."""
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        # First-order approximation
        return np.eye(3, dtype=float) + _skew(w)

    k = w / theta
    K = _skew(k)
    s = math.sin(theta)
    c = math.cos(theta)
    return np.eye(3, dtype=float) + s * K + (1.0 - c) * (K @ K)


def so3_log(R: NDArray[np.float64]) -> NDArray[np.float64]:
    """Logarithm map for SO(3): SO(3) -> so(3).

    Parameters
    ----------
    R : (3,3) ndarray
        Rotation matrix.

    Returns
    -------
    w : (3,) ndarray
        Rotation vector (axis-angle) in radians such that Exp(w) ~= R.

    Notes
    -----
    - The trace is clamped to [-1, 1] before acos to avoid domain errors.
    - Near identity, a first-order approximation is used based on (R - R^T)."""
    # Clamp trace for numerical safety
    tr = float(np.trace(R))
    cos_theta = (tr - 1.0) * 0.5
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = math.acos(cos_theta)

    if theta < 1e-12:
        # Near identity: use first-order approximation from skew(R - R^T)
        W = 0.5 * (R - R.T)
        return np.array([W[2, 1], W[0, 2], W[1, 0]], dtype=float)

    # General case
    W = (R - R.T) * (0.5 / math.sin(theta))
    w = np.array([W[2, 1], W[0, 2], W[1, 0]], dtype=float)
    return theta * w


def se3_from_Rt(R: NDArray[np.float64], t: NDArray[np.float64]) -> NDArray[np.float64]:
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def se3_perturb_left(T: NDArray[np.float64], delta: NDArray[np.float64]) -> NDArray[np.float64]:
    """Left-multiply a small perturbation on SE(3).

    Update rule
    -----------
    This code uses a *left* perturbation:

        T <- Exp(delta) * T

    where delta is a 6-vector arranged as:

        delta = [dt_x, dt_y, dt_z, dtheta_x, dtheta_y, dtheta_z]

    with dt in meters (or your translation unit) and dtheta in radians.

    Implementation
    --------------
    A practical small-step retraction is used:

        R <- Exp(dtheta) @ R
        t <- t + dt

    This is suitable for LM/GN when delta remains small (as enforced by damping).

    Parameters
    ----------
    T : (4,4) ndarray
        Current transform.
    delta : (6,) ndarray
        Tangent-space perturbation.

    Returns
    -------
    T_new : (4,4) ndarray
    Updated transform after applying the perturbation."""
    dt = delta[:3]
    dth = delta[3:6]
    R = T[:3, :3]
    t = T[:3, 3]
    Rn = so3_exp(dth) @ R
    tn = t + dt
    return se3_from_Rt(Rn, tn)


def se3_residual(T_cam_drg_pred: NDArray[np.float64],
                 T_cam_drg_meas: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute the 6D pose residual between a predicted and measured camera-frame pose.

    Residual definition
    -------------------
    Translation residual:
        r_t = t_pred - t_meas

    Rotation residual (minimal 3-vector):
        r_R = log( R_meas^T @ R_pred )

    The rotation residual is in the Lie algebra so(3), representing the small
    rotation that takes the measurement into the prediction.

    Parameters
    ----------
    T_cam_drg_pred : (4,4) ndarray
        Predicted drogue pose in the camera frame.
    T_cam_drg_meas : (4,4) ndarray
        Measured drogue pose in the camera frame.

    Returns
    -------
    r : (6,) ndarray
    Stacked residual [dx, dy, dz, dtheta_x, dtheta_y, dtheta_z]."""
    R_pred = T_cam_drg_pred[:3, :3]
    t_pred = T_cam_drg_pred[:3, 3]
    R_meas = T_cam_drg_meas[:3, :3]
    t_meas = T_cam_drg_meas[:3, 3]

    r_t = t_pred - t_meas
    r_R = so3_log(R_meas.T @ R_pred)
    return np.hstack([r_t, r_R]).astype(float)


def chol_whitener(C: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute a whitening matrix from a 6x6 covariance.

    We want W such that for a residual r:

        ||W r||^2 = r^T C^{-1} r

    Using Cholesky factorization:

        C = L L^T  (L lower triangular)

    a valid whitener is:

        W = L^{-1}

    Parameters
    ----------
    C : (6,6) ndarray
        Positive definite covariance matrix.

    Returns
    -------
    W : (6,6) ndarray
    Whitening matrix."""
    L = np.linalg.cholesky(C)
    return np.linalg.inv(L)


# -----------------------------------------------------------------------------
# Sensor and Measurement Definitions
# -----------------------------------------------------------------------------
def rle_cov(sig_r_lat_el, sig_rot):
    """
    Build covariance where translation is specified in a Radial/Lateral/Elevation basis.

    All sig_* are std-devs (meters for translation, radians for rotation).
    sig_rot can be scalar or 3-vector (Rodrigues stds).

    Basis (in sensor frame):
      u_r  : radial unit vector along measured translation t_s
      u_lat: lateral (perp to u_r, roughly horizontal)
      u_el : elevation (completes right-handed triad)

    Returns a Measurement -> (6,6) covariance.
    """
    sig_r   = float(sig_r_lat_el[0])
    sig_lat = float(sig_r_lat_el[1])
    sig_el  = float(sig_r_lat_el[2])
    sig_rot = np.array(sig_rot, dtype=float).reshape(-1)
    if sig_rot.size == 1:
        sig_rot = np.repeat(sig_rot, 3)

    def model(meas: Measurement) -> np.ndarray:
        t = np.asarray(meas.T_s_obj[:3, 3], dtype=float)
        d = float(np.linalg.norm(t))
        if d < 1e-9:
            # Degenerate: fall back to sensor axes
            C_tt_s = np.diag([sig_r**2, sig_lat**2, sig_el**2])
        else:
            u_r = t / d

            # Choose an "up-ish" reference to define lateral/elevation robustly
            z_ref = np.array([0.0, 0.0, 1.0], dtype=float)
            # If u_r is too close to z, use x instead
            if abs(float(u_r @ z_ref)) > 0.95:
                z_ref = np.array([1.0, 0.0, 0.0], dtype=float)

            u_lat = np.cross(z_ref, u_r)
            u_lat /= max(np.linalg.norm(u_lat), 1e-12)

            u_el = np.cross(u_r, u_lat)
            u_el /= max(np.linalg.norm(u_el), 1e-12)

            R_s_rle = np.column_stack([u_r, u_lat, u_el])  # maps RLE coords -> sensor coords

            C_rle = np.diag([sig_r**2, sig_lat**2, sig_el**2])
            C_tt_s = R_s_rle @ C_rle @ R_s_rle.T

        C = np.zeros((6, 6), dtype=float)
        C[:3, :3] = C_tt_s
        C[3:, 3:] = np.diag(sig_rot**2)
        return C

    return model


CovModel = Callable[["Measurement"], np.ndarray]  # returns (6,6)

@dataclass(frozen=True)
class Sensor:
    """
    Represents a physical sensor with a fixed extrinsic calibration and a
    measurement covariance model.

    Attributes
    ----------
    name : str
        Human-readable sensor name (e.g., "EO", "LR").
    T_w_s : (4,4) ndarray
        Sensor pose in world coordinates (maps sensor -> world).
    cov_model : Callable[[Measurement], ndarray]
        Function that returns a 6x6 covariance for a given measurement.

    Notes
    -----
    - The covariance model may depend on measurement content (e.g., distance
      to target, detector confidence, pixel geometry).
    - Keeping this as a callable makes it easy to experiment with sensor models
      without changing the optimizer.
    """
    name: str
    T_w_s: np.ndarray  # (4,4) sensor pose in world
    cov_model: CovModel  # Measurement -> (6,6)


@dataclass
class Measurement:
    """
    A single relative drogue pose measurement produced by a sensor.

    Attributes
    ----------
    sensor : Sensor
        The sensor that produced this measurement.
    T_s_obj : (4,4) ndarray
        Measured drogue pose in the sensor frame (maps drg -> sensor).
    meta : dict
        Optional metadata (confidence, image size, etc.) usable by cov_model.

    Methods
    -------
    covariance()
        Returns the 6x6 covariance matrix associated with this measurement.

    Notes
    -----
    - The covariance is defined over the 6D residual:
          [dx, dy, dz, dtheta_x, dtheta_y, dtheta_z]
      where translation is in meters and rotation is in radians.
    """
    sensor: Sensor
    T_s_obj: np.ndarray  # (4,4) measurement: obj -> sensor
    meta: Dict[str, Any] = field(default_factory=dict)

    def covariance(self) -> np.ndarray:
        C = self.sensor.cov_model(self)
        C = np.array(C, dtype=float)
        if C.shape == (6,):
            C = np.diag(C)
        if C.shape != (6, 6):
            raise ValueError("Covariance must be (6,) or (6,6)")
        # PD check
        np.linalg.cholesky(C)
        return C


@dataclass(frozen=True)
class SensorSpec:
    name: str
    mean_loc_w: np.ndarray  # (3,)
    yaw_deg: float = 0.0  # about world Z, degrees
    pos_std_s: np.ndarray = None  # (3,) std on translation measurement in sensor frame
    rod_std: np.ndarray = None  # (3,) std on Rodrigues perturbation
    loc_jitter_std: float = 1.0  # std used by rand_position() for sensor placement


def _std6_to_cov6(pos_std: np.ndarray, rod_std: np.ndarray) -> np.ndarray:
    std = np.diag(np.hstack([pos_std, rod_std]))
    return std @ std


def _make_sensor_quat(yaw_deg: float, randomize: bool,
                      rot_jitter_deg: float) -> q:
    """
    Returns a quaternion for the sensor in world.
    - If randomize: base yaw rotation * random small rotation
    - Else: just base yaw rotation (or identity if yaw=0)
    """
    yaw_q = q().from_eulerD_rpy(np.array([0.0, 0.0, yaw_deg]))
    if not randomize:
        return yaw_q
    jitter = random_quat_within_deg(rot_jitter_deg)
    return yaw_q * jitter


def _true_pose_in_sensor(drg_quat: q,
                         w_t_drg: np.ndarray,
                         s_quat: q,
                         w_t_s: np.ndarray) -> Tuple[q, np.ndarray, np.ndarray]:
    """
    Compute true drogue pose in sensor frame:
      q_s_drg = s^T * drg
      t_s_drg = s^T * (w_t_drg - w_t_s)
      T_s_drg = SE3(q_s_drg, t_s_drg)
    """
    true_q = s_quat.T * drg_quat
    true_t = s_quat.T * (w_t_drg - w_t_s)
    true_T = true_q.to_SE3_given_position(true_t)
    return true_q, true_t, true_T


def _noisy_measurement_from_true(true_q: q,
                                 true_t: np.ndarray,
                                 pos_std: np.ndarray,
                                 rod_std: np.ndarray) -> np.ndarray:
    """
    Apply noise in the same style as your current code:
      - rotation: perturb_from_rodrigues_std(rod_std)
      - translation: add pos_std * randn
    Returns T_s_drg_meas (4x4).
    """
    est_q = true_q.perturb_from_rodrigues_std(rod_std)
    est_t = true_t + pos_std * np.random.randn(3)
    return est_q.to_SE3_given_position(est_t)


# -----------------------------------------------------------------------------
# Factor-graph-style static fusion for drogue pose (single SE(3) state)
# -----------------------------------------------------------------------------
@dataclass
class LMInfo:
    converged: bool
    iters: int
    final_cost: float
    lambda_final: float
    step_norm: float


class StaticFGMeasurementMelding:
    """
    Static factor-graph-style fusion for a single drogue pose in SE(3).

    Problem
    -------
    Estimate a constant world-frame drogue pose T_w_drg given multiple
    independent sensor-relative pose measurements and their covariances.

    State
    -----
    - T_w_drg : (4,4) ndarray
        World-frame drogue pose (maps drg -> world).

    Measurements (factors)
    ----------------------
    Each sensor contributes one factor defined by:
      - T_w_s        : sensor pose in world (extrinsic calibration)
      - T_s_drg_meas : measured drogue pose in sensor frame
      - C            : 6x6 covariance of the pose residual

    Forward model
    -------------
    Given a candidate world pose T_w_drg, the predicted measurement is:

        T_s_drg_pred = inv(T_w_s) * T_w_drg

    Residual definition
    -------------------
    Translation:
        r_t = t_pred - t_meas

    Rotation (minimal):
        r_R = log( R_meas^T * R_pred )

    The full residual is:
        r = [r_t, r_R]  ∈ R^6

    Objective
    ---------
    Minimize the sum of squared, whitened residuals:

        minimize 0.5 * sum_i || W_i r_i ||^2

    where W_i is a Cholesky-based whitener derived from the measurement covariance.

    Optimization
    ------------
    Uses Levenberg–Marquardt with a left perturbation on SE(3):

        T <- Exp(delta) * T

    Diagnostics
    -----------
    After calling ``solve()``, detailed diagnostics are stored internally and
    can be inspected via ``print_summary()`` without re-running optimization.
    """

    def __init__(self, measurements: list[Measurement]):
        self._diagnostics = {}
        self.measurements = measurements
        self._validate_input()
        self.W = [chol_whitener(m.covariance()) for m in self.measurements]

    def _set_diag(self, **kwargs):
        self._diagnostics.update(kwargs)

    def _validate_input(self):
        if not self.measurements:
            raise ValueError("Need at least 1 measurement")

        for i, m in enumerate(self.measurements):
            if m.sensor.T_w_s.shape != (4, 4):
                raise ValueError(f"measurement[{i}].sensor.T_w_s must be (4,4)")
            if m.T_s_obj.shape != (4, 4):
                raise ValueError(f"measurement[{i}].T_s_obj must be (4,4)")
            if not np.isfinite(m.sensor.T_w_s).all():
                raise ValueError(f"measurement[{i}].sensor.T_w_s has non-finite values")
            if not np.isfinite(m.T_s_obj).all():
                raise ValueError(f"measurement[{i}].T_s_obj has non-finite values")

            C = m.covariance()  # already checks shape + PD via chol
            if C.shape != (6, 6):
                raise ValueError(f"measurement[{i}] covariance not (6,6)")

    def solve(self,
              max_iters: int = 25,
              verbose_lm: bool = True,
              **lm_kwargs):
        """
        Run the full fusion pipeline and cache diagnostics.

        Steps
        -----
        1) Lift each sensor measurement into the world frame:
               T_w_drg_i = T_w_s_i * T_s_drg_meas

        2) Construct an initial guess for T_w_drg using an information-weighted
           average of translations and a chordal mean of rotations.

        3) Refine the estimate using Levenberg–Marquardt.

        4) Approximate the posterior covariance of the solution in tangent space.

        Returns
        -------
        T_opt : (4,4) ndarray
            Optimized world-frame drogue pose.
        info : dict
            LM convergence and tuning information.
        """
        # Compute EO/LR world lifts (useful for debugging + init guess sanity)
        T_w_obj_lifts = [se3_mul(m.sensor.T_w_s, m.T_s_obj) for m in self.measurements]

        # Initial guess
        T_init = self.initial_guess_world()

        # Optimize
        T_opt, info = self.optimize_lm(
            T_init,
            max_iters=max_iters,
            verbose=verbose_lm,
            **lm_kwargs
        )
        # --- Posterior covariance at optimum (6x6 in tangent space) ---
        cov6, cov_info = self.estimate_posterior_covariance(
            T_opt,
            eps_t=lm_kwargs.get("eps_t", 1e-4),
            eps_r=lm_kwargs.get("eps_r", 1e-5),
            scale_by_reduced_chi2=True,
            damping=0.0,
        )

        std6 = np.sqrt(np.maximum(np.diag(cov6), 0.0))

        # Stash for print_summary()
        self._diagnostics["cov6"] = cov6
        self._diagnostics["std6"] = std6
        self._diagnostics["cov_info"] = cov_info

        # Stash everything needed for reporting
        self._set_diag(
            T_w_obj_lifts=T_w_obj_lifts,
            T_w_obj_init=T_init,
            T_w_obj_opt=T_opt,
            lm_info=info,
        )

        return T_opt, info

    def estimate_posterior_covariance(
            self,
            T_w_drg_opt: np.ndarray,
            *,
            eps_t: float = 1e-4,
            eps_r: float = 1e-5,
            scale_by_reduced_chi2: bool = True,
            damping: float = 0.0,
    ) -> tuple[np.ndarray, dict]:
        """
        Approximate the posterior covariance of the optimized pose.

        Returns a 6x6 covariance in the local tangent space:

            delta = [dt_x, dt_y, dt_z, dtheta_x, dtheta_y, dtheta_z]

        Notes
        -----
        - This is a local approximation around the optimum.
        - Because residuals are whitened, sensor weighting is already embedded.
        - Optionally scales by reduced chi-square for conservative uncertainty.
        """
        r = self._stacked_whitened_residual(T_w_drg_opt)  # (m,)
        m = int(r.size)

        J = self.numeric_jacobian(T_w_drg_opt, eps_t=eps_t, eps_r=eps_r)  # (m,6)
        H = J.T @ J

        if damping > 0.0:
            H = H + float(damping) * np.eye(6)

        # Invert with a stable fallback
        try:
            cov6 = np.linalg.inv(H)
            inv_method = "inv"
        except np.linalg.LinAlgError:
            cov6 = np.linalg.pinv(H)
            inv_method = "pinv"

        chi2 = float(r @ r)
        dof = max(m - 6, 1)
        scale = 1.0

        if scale_by_reduced_chi2:
            scale = chi2 / float(dof)
            cov6 = cov6 * scale

        # Useful diagnostics
        try:
            cond = float(np.linalg.cond(H))
        except np.linalg.LinAlgError:
            cond = float("nan")

        info = {
            "m": m,
            "dof": dof,
            "chi2": chi2,
            "scale": scale,
            "inv_method": inv_method,
            "cond_H": cond,
            "eps_t": eps_t,
            "eps_r": eps_r,
            "damping": damping,
            "scale_by_reduced_chi2": scale_by_reduced_chi2,
        }
        return cov6, info

    def print_summary(self):
        """
        Print a comprehensive diagnostic summary of the fusion result.

        Includes
        --------
        - Initial and optimized world-frame drogue pose
        - Per-sensor measurements, predictions, and truth (if provided)
        - Raw and whitened residuals in sensor frames
        - Per-factor residual norms (init vs opt)
        - World-frame truth comparison (if provided)
        - Posterior covariance and standard deviations

        This function is intentionally verbose and designed for debugging,
        validation, and experimentation.
        """

        sep = "=" * 96
        sub = "-" * 96

        d = getattr(self, "_diagnostics", {}) or {}

        # Pull useful cached values (if present)
        T_init = d.get("T_w_obj_init", None)
        if T_init is None:
            T_init = d.get("T_w_drg_init", None)

        T_opt = d.get("T_w_obj_opt", None)
        if T_opt is None:
            T_opt = d.get("T_w_drg_opt", None)

        # Build lifts (world estimates from each measurement) if not cached
        lifts = d.get("T_w_obj_lifts", None) or d.get("T_w_drg_lifts", None)
        if lifts is None:
            lifts = [se3_mul(m.sensor.T_w_s, m.T_s_obj) for m in self.measurements]

        # Optional truth
        T_w_obj_true = getattr(self, "T_w_obj_true", None)
        T_s_obj_true_by_sensor = getattr(self, "T_s_obj_true_by_sensor", None)

        # Local helpers
        def _fmt(x, nd=6):
            if x is None:
                return "None"
            if isinstance(x, (float, int, np.floating, np.integer)):
                return f"{float(x):.{nd}f}"
            return str(x)

        def _norm(vect):
            return float(np.linalg.norm(vect))

        def _angle_from_R(R):
            # angle in [0, pi]
            # robust clamp against numeric creep
            c = (np.trace(R) - 1.0) * 0.5
            c = float(np.clip(c, -1.0, 1.0))
            return float(np.arccos(c))

        def _pose_error(T_a, T_b):
            """
            Pose difference a->b in a simple, readable form:
              - translational error norm
              - rotational angle error (rad)
            """
            if T_a is None or T_b is None:
                return None, None
            T_err = se3_mul(se3_inv(T_a), T_b)
            d_pos = T_err[:3, 3]
            d_ang = _angle_from_R(T_err[:3, :3])
            return _norm(d_pos), d_ang

        def _factor_residual(m, T_w_obj):
            # predicts measurement in sensor frame given world object pose:
            #   pred: T_s_obj_pred = inv(T_w_s) * T_w_obj
            pred = se3_mul(se3_inv(m.sensor.T_w_s), T_w_obj)
            return se3_residual(pred, m.T_s_obj)  # (6,)

        # -------------------- Header --------------------
        print("\n" + sep)
        print("StaticFGMeasurementMelding — Summary (Generic N-sensor)")
        print(sep)

        print(f"\n# Measurements: {len(self.measurements)}")
        if T_init is not None:
            print("\nInitial guess (T_w_obj_init):")
            print(T_init)
        if T_opt is not None:
            print("\nOptimized (T_w_obj_opt):")
            print(T_opt)

        # If available, print LM info
        lm_info = d.get("lm_info", None)
        if isinstance(lm_info, dict) and lm_info:
            print("\n" + sep)
            print("LM info:")
            print(sep)
            for k in ["iters", "converged", "final_cost", "final_lambda", "final_step_norm", "final_improve"]:
                if k in lm_info:
                    print(f"  {k}: {lm_info[k]}")
            # Print any other keys (stable-ish order)
            other = [k for k in lm_info.keys() if
                     k not in {"iters", "converged", "final_cost", "final_lambda", "final_step_norm", "final_improve"}]
            for k in sorted(other):
                v = lm_info[k]
                if isinstance(v, (float, int, np.floating, np.integer, str, bool)):
                    print(f"  {k}: {v}")

        # -------------------- Measurements --------------------
        print("\n" + sep)
        print("Measurements")
        print(sep)

        for i, m in enumerate(self.measurements):
            print("\n" + sub)
            print(f"[{i + 1}] Sensor: {m.sensor.name}")
            print(sub)

            print("T_w_s (sensor extrinsic):")
            print(m.sensor.T_w_s)

            print("\nT_s_obj (measured pose):")
            print(m.T_s_obj)

            # --- Predicted + truth projections into sensor frame ---
            if T_opt is not None:
                # Predicted measurement from optimized world pose
                #   T_s_obj_pred = inv(T_w_s) * T_w_obj_opt
                T_s_obj_pred = se3_mul(se3_inv(m.sensor.T_w_s), T_opt)

                print("\nT_s_obj (predicted from T_w_obj_opt):")
                print(T_s_obj_pred)

                # Residual: predicted vs measured (same convention as solver)
                r_pm = se3_residual(T_s_obj_pred, m.T_s_obj)

                dt_pm = float(np.linalg.norm(r_pm[:3]))
                dth_pm = float(np.linalg.norm(r_pm[3:]))

                print("\nResidual (pred - meas):")
                print(f"  ||dt||     = {dt_pm:.6f}")
                print(f"  ||dtheta|| = {dth_pm:.6f} rad")

                try:
                    W = self.W[i]
                    rw_pm = W @ r_pm
                    print(f"  ||W*r||    = {float(np.linalg.norm(rw_pm)):.6f}")
                except (IndexError, ValueError, AttributeError):
                    pass

                print("  r = [dt_x, dt_y, dt_z, dth_x, dth_y, dth_z]:")
                print(r_pm)

                # ---- Truth projected into sensor frame (if available) ----
                if isinstance(T_s_obj_true_by_sensor, dict) and m.sensor.name in T_s_obj_true_by_sensor:
                    T_s_obj_true = T_s_obj_true_by_sensor[m.sensor.name]

                    print("\nT_s_obj (truth):")
                    print(T_s_obj_true)

                    # Residual: predicted vs truth
                    r_pt = se3_residual(T_s_obj_pred, T_s_obj_true)

                    dt_pt = float(np.linalg.norm(r_pt[:3]))
                    dth_pt = float(np.linalg.norm(r_pt[3:]))

                    print("\nResidual (pred - truth):")
                    print(f"  ||dt||     = {dt_pt:.6f}")
                    print(f"  ||dtheta|| = {dth_pt:.6f} rad")

                    try:
                        rw_pt = W @ r_pt
                        print(f"  ||W*r||    = {float(np.linalg.norm(rw_pt)):.6f}")
                        print("Weighted Residual")
                        print("  z = W*r (sigma units):")
                        print(rw_pt)
                        print(f"  max|z| = {float(np.max(np.abs(rw_pt))):.3f}")
                    except (NameError, ValueError, AttributeError):
                        pass
                    print("Unweighted Residual")
                    print("  r = [dt_x, dt_y, dt_z, dth_x, dth_y, dth_z]:")
                    print(r_pt)
                    print(f"  max|r| = {float(np.max(np.abs(r_pt))):.3f}")

            # Covariance + whitener sanity
            C = m.covariance()
            print("\nCovariance C (6x6):")
            print(C)

            # World lift
            print("\nLifted world pose from this measurement (T_w_obj_i = T_w_s * T_s_obj):")
            print(lifts[i])

            # Optional truth in sensor frame
            if isinstance(T_s_obj_true_by_sensor, dict) and m.sensor.name in T_s_obj_true_by_sensor:
                T_true_s = T_s_obj_true_by_sensor[m.sensor.name]
                dt, dtheta = _pose_error(T_true_s, m.T_s_obj)
                print("\nTruth comparison in sensor frame (if provided):")
                print(f"  ||dt||  = {_fmt(dt)}")
                print(f"  dtheta  = {_fmt(dtheta)} rad")

        # -------------------- Per-factor residuals (init vs opt) --------------------
        if T_init is not None or T_opt is not None:
            print("\n" + sep)
            print("Per-factor residual norms (raw + whitened)")
            print(sep)

            for i, (m, W) in enumerate(zip(self.measurements, self.W)):
                print("\n" + sub)
                print(f"[{i + 1}] {m.sensor.name}")
                print(sub)

                if T_init is not None:
                    r0 = _factor_residual(m, T_init)
                    rw0 = W @ r0
                    print("Init:")
                    print(
                        f"  raw  : ||dt||={_fmt(_norm(r0[:3]))}   ||dtheta||={_fmt(_norm(r0[3:]))}   ||r||={_fmt(_norm(r0))}")
                    print(f"  white: ||W*r||={_fmt(_norm(rw0))}")

                if T_opt is not None:
                    r1 = _factor_residual(m, T_opt)
                    rw1 = W @ r1
                    print("Opt:")
                    print(
                        f"  raw  : ||dt||={_fmt(_norm(r1[:3]))}   ||dtheta||={_fmt(_norm(r1[3:]))}   ||r||={_fmt(_norm(r1))}")
                    print(f"  white: ||W*r||={_fmt(_norm(rw1))}")

        # -------------------- Truth comparison in world --------------------
        if T_w_obj_true is not None:
            print("\n" + sep)
            print("Truth comparison (world)")
            print(sep)
            print("True World Pose")
            print(T_w_obj_true)
            print(sep)

            if T_init is not None:
                dt, dtheta = _pose_error(T_w_obj_true, T_init)
                print("Init vs truth:")
                print(f"  ||dt|| = {_fmt(dt)}")
                print(f"  dtheta = {_fmt(dtheta)} rad")

            if T_opt is not None:
                dt, dtheta = _pose_error(T_w_obj_true, T_opt)
                print("Opt vs truth:")
                print(f"  ||dt|| = {_fmt(dt)}")
                print(f"  dtheta = {_fmt(dtheta)} rad")

        cov6 = d.get("cov6", None)
        std6 = d.get("std6", None)
        cov_info = d.get("cov_info", None)

        if cov6 is not None:
            print("\nPosterior covariance (tangent space, 6x6):")
            print("  delta = [dt_x, dt_y, dt_z, dtheta_x, dtheta_y, dtheta_z]")
            print(cov6)
            if std6 is not None:
                print("\nPosterior stddev (sqrt(diag(cov6))):")
                print(std6)
            if isinstance(cov_info, dict):
                print("\nPosterior cov diagnostics:")
                for k in ["m", "dof", "chi2", "scale", "inv_method", "cond_H", "eps_t", "eps_r", "damping"]:
                    if k in cov_info:
                        print(f"  {k}: {cov_info[k]}")

        print("\n" + sep)
        print("End summary")
        print(sep)

    # -----------------------
    # Factor graph pieces
    # -----------------------
    @staticmethod
    def _predict_cam_measurement(T_w_cam: NDArray[np.float64],
                                 T_w_drg: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Predicted camera-frame drogue pose:
            T_cam_drg = inv(T_w_cam) * T_w_drg
        """
        return se3_mul(se3_inv(T_w_cam), T_w_drg)

    def _stacked_whitened_residual(self, T_w_obj):
        chunks = []
        for m, W in zip(self.measurements, self.W):
            T_w_s = m.sensor.T_w_s
            pred = se3_mul(se3_inv(T_w_s), T_w_obj)  # T_s_obj_pred
            r = se3_residual(pred, m.T_s_obj)
            chunks.append(W @ r)
        return np.hstack(chunks)

    @staticmethod
    def _cost(r_w: NDArray[np.float64]) -> float:
        return 0.5 * float(r_w @ r_w)

    def numeric_jacobian(self, T_w_drg: np.ndarray, eps_t: float = 1e-4, eps_r: float = 1e-5) -> np.ndarray:
        """
        Numerical Jacobian of the stacked, whitened residual r(T) w.r.t. a 6D se(3) perturbation.

        delta = [dt_x, dt_y, dt_z, dtheta_x, dtheta_y, dtheta_z]
          - translation perturbed with eps_t (meters)
          - rotation perturbed with eps_r (radians)

        Returns:
            J: (m x 6) where m is total residual length (e.g., 12 for 2 factors).
        """
        r0 = self._stacked_whitened_residual(T_w_drg)  # shape (m,)
        m = r0.size
        J = np.zeros((m, 6), dtype=float)

        # Translation columns
        for i in range(3):
            d = np.zeros(6, dtype=float)
            d[i] = eps_t
            Tp = se3_perturb_left(T_w_drg, d)
            rp = self._stacked_whitened_residual(Tp)
            J[:, i] = (rp - r0) / eps_t

        # Rotation columns
        for i in range(3):
            d = np.zeros(6, dtype=float)
            d[3 + i] = eps_r
            Tp = se3_perturb_left(T_w_drg, d)
            rp = self._stacked_whitened_residual(Tp)
            J[:, 3 + i] = (rp - r0) / eps_r

        return J

    # -----------------------
    # Initialization
    # -----------------------
    def initial_guess_world(self) -> NDArray[np.float64]:
        lifts = [se3_mul(m.sensor.T_w_s, m.T_s_obj) for m in self.measurements]

        # ---- translation info-weighted mean ----
        Wsum = np.zeros((3, 3), dtype=float)
        bsum = np.zeros(3, dtype=float)
        for m, T_w_obj in zip(self.measurements, lifts):
            C = m.covariance()
            Wi = np.linalg.inv(C[:3, :3])  # translation information
            ti = T_w_obj[:3, 3]
            Wsum += Wi
            bsum += Wi @ ti
        t0 = np.linalg.solve(Wsum, bsum)

        # ---- rotation chordal mean + projection ----
        M = np.zeros((3, 3), dtype=float)
        for m, T_w_obj in zip(self.measurements, lifts):
            C = m.covariance()
            Ri = T_w_obj[:3, :3]
            wi = 1.0 / float(np.clip(np.trace(C[3:6, 3:6]), 1e-12, np.inf))
            M += wi * Ri

        U, _, Vt = np.linalg.svd(M)
        R0 = U @ Vt
        if np.linalg.det(R0) < 0:
            U[:, -1] *= -1
            R0 = U @ Vt

        return se3_from_Rt(R0, t0)

    # -----------------------
    # LM optimizer
    # -----------------------
    def optimize_lm(self,
                    T_init: np.ndarray,
                    max_iters: int = 25,
                    eps_t: float = 1e-4,
                    eps_r: float = 1e-5,
                    lambda0: float = 3e-4,
                    lambda_up: float = 10.0,
                    lambda_down: float = 3.0,
                    step_tol: float = 1e-7,
                    verbose: bool = True):
        """
        Levenberg–Marquardt optimization on SE(3) for a static drogue pose.

        Notes
        -----
        - Residuals are already whitened before entering the solver.
        - Translation and rotation use separate finite-difference step sizes.
        - The update uses a left perturbation, consistent with Lie-group practice.
        """
        T = T_init.copy()
        lam = float(lambda0)

        def cost_from_r(res):
            return 0.5 * float(res @ res)

        r = self._stacked_whitened_residual(T)
        cost = cost_from_r(r)

        it = 0
        converged = False
        step_norm = 1.0

        for it in range(1, max_iters + 1):
            # --- Jacobian with split eps ---
            J = self.numeric_jacobian(T, eps_t=eps_t, eps_r=eps_r)

            # Normal equations
            H = J.T @ J
            g = J.T @ r

            # LM damping
            H_lm = H + lam * np.eye(6)

            # Solve for step
            try:
                delta = -np.linalg.solve(H_lm, g)
            except np.linalg.LinAlgError:
                # Fall back to least squares if needed
                delta, *_ = np.linalg.lstsq(H_lm, -g, rcond=None)

            step_norm = float(np.linalg.norm(delta))

            # Trial step
            T_trial = se3_perturb_left(T, delta)
            r_trial = self._stacked_whitened_residual(T_trial)
            cost_trial = cost_from_r(r_trial)

            improve = cost - cost_trial

            if verbose:
                print(f"[LM] Iter {it:02d}: cost {cost:.6e}  step {step_norm:.3e}  "
                      f"improve {improve:.3e}  lambda {lam:.3e}")

            # Accept / reject
            if improve > 0:
                T = T_trial
                r = r_trial
                cost = cost_trial
                lam = max(lam / lambda_down, 1e-12)
            else:
                lam = lam * lambda_up

            # --- Convergence test uses step_tol, not eps ---
            if step_norm < step_tol:
                converged = True
                break

        # You can return your existing info struct; here’s a generic dict
        info = {
            "converged": converged,
            "iters": it,
            "final_cost": cost,
            "lambda_final": lam,
            "step_norm": step_norm,
            "eps_t": eps_t,
            "eps_r": eps_r,
            "step_tol": step_tol,
        }
        return T, info


# -----------------------------------------------------------------------------
# Synthetic data generation / test harness
# -----------------------------------------------------------------------------
def rand_position(loc: NDArray, mu: float) -> NDArray:
    return loc + mu * np.random.randn(3)


def test_values(randomize: bool = True):
    # ----------------------------
    # "Data entry" (compact)
    # ----------------------------
    drogue_meanLoc = np.array([10.0, 0.0, 0.0])

    sensors: List[SensorSpec] = [
        SensorSpec(
            name="EO",
            mean_loc_w=np.array([0.0, 0.0, 3.0]),
            yaw_deg=0.0,
            pos_std_s=np.array([5.0, 1.0, 1.0]),
            rod_std=np.array([0.05, 0.02, 0.02]),
            loc_jitter_std=1.0,
        ),
        SensorSpec(
            name="LR",
            mean_loc_w=np.array([0.0, -3.0, 3.0]),
            yaw_deg=40.0,
            pos_std_s=np.array([1.0, 2.0, 2.0]),
            rod_std=np.array([0.03, 0.02, 0.02]),
            loc_jitter_std=1.0,
        ),
        SensorSpec(
            name="MW",
            mean_loc_w=np.array([0.0, 1.0, 3.0]),
            yaw_deg=-10.0,
            pos_std_s=np.array([1.0, 5.0, 5.0]),
            rod_std=np.array([0.1, 0.1, 0.1]),
            loc_jitter_std=1.0,
        ),
    ]

    # ----------------------------
    # World truth pose
    # ----------------------------
    if randomize:
        w_t_drg = rand_position(drogue_meanLoc, 5.0)
        drg_quat = random_quat_within_deg(10.0)
    else:
        w_t_drg = drogue_meanLoc
        drg_quat = q()

    true_drg_pose = drg_quat.to_SE3_given_position(w_t_drg)

    # ----------------------------
    # Per-sensor generation loop
    # ----------------------------
    measurements: List[Measurement] = []
    true_in_sensor: Dict[str, np.ndarray] = {}  # name -> true T_s_drg (4x4)

    for spec in sensors:
        # sensor placement
        if randomize:
            w_t_s = rand_position(spec.mean_loc_w, spec.loc_jitter_std)
        else:
            w_t_s = spec.mean_loc_w

        # sensor orientation (world)
        s_quat = _make_sensor_quat(spec.yaw_deg, randomize=randomize, rot_jitter_deg=10.0)

        # extrinsic
        T_w_s = s_quat.to_SE3_given_position(w_t_s)

        # true drogue in sensor frame
        true_q, true_t, T_s_drg_true = _true_pose_in_sensor(
            drg_quat=drg_quat,
            w_t_drg=w_t_drg,
            s_quat=s_quat,
            w_t_s=w_t_s,
        )
        true_in_sensor[spec.name] = T_s_drg_true

        # noisy measurement
        if randomize:
            T_s_drg_meas = _noisy_measurement_from_true(
                true_q=true_q,
                true_t=true_t,
                pos_std=spec.pos_std_s,
                rod_std=spec.rod_std,
            )
        else:
            T_s_drg_meas = T_s_drg_true

        # covariance model (constant)
        sensor = Sensor(spec.name, T_w_s=T_w_s,
                        cov_model=rle_cov(sig_r_lat_el=spec.pos_std_s, sig_rot=spec.rod_std))
        measurements.append(Measurement(sensor=sensor, T_s_obj=T_s_drg_meas))

    true_list = tuple(true_in_sensor[nm] for nm in [s.name for s in sensors])
    return true_drg_pose, measurements, true_list


def run_test(randomize: bool = False):
    true_drg_pose, meas, truth = test_values(randomize=randomize)
    true_eo_drgPose, true_lr_drgPose, true_mw_drgPose = truth

    fg = StaticFGMeasurementMelding(meas)

    fg.T_w_obj_true = true_drg_pose
    fg.T_s_obj_true_by_sensor = {
        "EO": true_eo_drgPose,
        "LR": true_lr_drgPose,
        "MW": true_mw_drgPose
    }

    fg.solve(max_iters=25, verbose_lm=True, eps_t=1e-5, eps_r=1e-5)
    fg.print_summary()
    plot_all_fg_diagnostics(fg)


def const_cov(C6x6: np.ndarray):
    def model(meas: Measurement) -> np.ndarray:
        return C6x6

    return model


def inv_distance_cov(a_t, b_t, a_r, b_r, eps=1e-6):
    # a_* and b_* can be scalars or 3-vectors for anisotropic noise
    a_t = np.array(a_t, dtype=float).reshape(-1)
    b_t = np.array(b_t, dtype=float).reshape(-1)
    a_r = np.array(a_r, dtype=float).reshape(-1)
    b_r = np.array(b_r, dtype=float).reshape(-1)

    def model(meas: Measurement) -> np.ndarray:
        t = meas.T_s_obj[:3, 3]
        d = float(np.linalg.norm(t))
        d = max(d, eps)

        sig_t = a_t + b_t / d
        sig_r = a_r + b_r / d

        if sig_t.size == 1:
            sig_t = np.repeat(sig_t, 3)
        if sig_r.size == 1:
            sig_r = np.repeat(sig_r, 3)

        std = np.diag(np.hstack([sig_t, sig_r]))
        return std @ std

    return model


def main(test: bool = True) -> None:
    if test:
        run_test(randomize=True)
        return

    # Example usage for your own data:
    eo_camPose = np.eye(4)
    eo_camPose[:3, 3] = np.array([0.0, 0.0, 3.0])

    eo_drgPose = np.eye(4)
    eo_drgPose[:3, 3] = np.array([10.0, 0.0, -3.0])

    # inv distance model: a + b/r for r=distance to object
    eo = Sensor("EO", T_w_s=eo_camPose,
                cov_model=inv_distance_cov(a_t=np.array([4.0, 0.75, 0.75]),
                                           b_t=np.array([3.0, 2.0, 2.0]),
                                           a_r=np.array([0.05, 0.01, 0.01]),
                                           b_r=np.array([0.01, 0.01, 0.01])))

    lr_camPose = np.eye(4)
    lr_camPose[:3, 3] = np.array([0.0, -3.0, 3.0])

    lr_cov = np.diag([1.0, 2.0, 2.0, 0.05, 0.02, 0.02])

    lr_drgPose = np.eye(4)
    lr_drgPose[:3, 3] = np.array([10.0, 0.0, -3.0])
    lr = Sensor("LR", T_w_s=lr_camPose, cov_model=const_cov(lr_cov))

    meas = [
        Measurement(sensor=eo, T_s_obj=eo_drgPose),
        Measurement(sensor=lr, T_s_obj=lr_drgPose),
    ]

    fg = StaticFGMeasurementMelding(meas)

    # Only if you have them. Say, from MOCAP
    # fg.T_w_drg_true = true_drg_pose       # SE3 style numpy array
    # fg.eo_drgPose_true = true_eo_drgPose  # SE3 style numpy array
    # fg.lr_drgPose_true = true_lr_drgPose  # SE3 style numpy array

    fg.solve(max_iters=25, verbose_lm=True, eps_t=1e-4, eps_r=1e-5)
    fg.print_summary()
    plot_all_fg_diagnostics(fg)


if __name__ == '__main__':
    main(test=True)
