"""

TODO UPDATE NOTES
Custom Perspective-n-Point (PnP) Demo
-------------------------------------

This module demonstrates a simple pipeline that estimates the **orientation** (unit
quaternion) and **position** (3D translation) of a rigid object/landmark set from
its detected feature points in a single camera image.

Key components:
- Synthetic feature generation in the object's local/model frame (`feature_points`).
- A pinhole camera projection model implemented in `h(q, t)` using the author's
  frame conventions: **x forward, y left (image u decreases to the left sign),
  z up**.
- An ultra-fast closed-form initializer `init_pose_wahba` that seeds the pose by
  (1) estimating rotation with a Wahba/Kabsch alignment on unit bearing vectors,
  then (2) estimating translation by linear least-squares given that rotation.
- A Gauss-Newton style optimizer `opt` that iteratively refines pose by solving a
  normal-equation step using the analytic Jacobian provided by `deriv`.

Notes on conventions and signs:
- The projection used here differs from typical OpenCV conventions by a sign flip
  for the horizontal (u) coordinate. Carefully track the mapping in `h()` and in
  the Jacobian `deriv()`.
- The quaternion class is expected to expose `.s` (scalar part), `.vec` (vector part),
  `.T` (rotate points into the camera frame), `from_eulerD_rpy`, `to_rodrigues`,
  and `angle_betweenD` methods, plus a `vect_deriv(p, right_project=True)`
  derivative helper for analytic Jacobians.

The code is written as an end-to-end script. Run directly to see a synthetic test
with noisy measurements, the initializer results, and the final optimized pose.
"""

import numpy as np
from sys import maxsize
from support.mathHelpers.quaternions import Quaternion as q, mat2quat
from support.vision.calibration import Calibration
from support.mathHelpers.include_numba import _njit as njit, prange
from support.core.enums import robust_cost
from numpy.typing import NDArray
from dataclasses import dataclass

# Pretty-printing controls for numpy (purely cosmetic; does not affect math)
np.set_printoptions(suppress=True, precision=4, threshold=maxsize)


@dataclass
class QnPStats:
    N: int
    dof: int
    sse_w: float
    s2: float
    cov6: np.ndarray

@dataclass
class SeedConfig:
    """
    Controls robust unseeded initialization for solveQnP().

    Goals:
      - Keep per-frame cost ~baseline (opt called once)
      - Improve robustness vs occasional DLT catastrophes
      - Make tradeoff configurable for precision vs speed
    """
    enabled: bool = True

    # Fast-path: try weighted DLT once (using sigma) and accept if "good enough".
    try_weighted_dlt_first: bool = True
    dlt_sigma_floor_px: float = 1.0
    accept_median_err_px: float = 12.0      # accept seed if median reproj error <= this
    accept_front_frac: float = 0.90        # accept seed if >= this fraction is in front

    # If fast-path fails: multi-hypothesis DLT + cheap scoring (NO full opt in loop).
    ransac_enabled: bool = True
    ransac_iters: int = 32                 # good starting point for ~60 points and low outlier rate
    subset_size: int = 6                   # 6 is fastest; bump to 8 if needed
    early_exit_median_err_px: float = 10.0  # stop early if we find a great seed

    # PROSAC-like sampling based on sigma (prefer most-certain points first)
    prosac_enabled: bool = True
    prosac_min_pool: int = 12              # initial pool size for PROSAC growth
    rng_seed: int = 0

    # Optional: do a tiny refine only on top-K seeds (still cheap)
    refine_top_k: int = 1                  # 0 disables, 1 is usually enough
    refine_max_iters: int = 4              # <=4 keeps it cheap

    # Scoring: score on quick subset first (avoid full scoring every hypothesis)
    score_quick_M: int = 2                # 0 => score all points always
    z_eps: float = 1e-3

############################################ REGULAR OPT PATH, NO ONLINE CAL ##########################################

@njit(cache=True, fastmath=False)
def _project_accum_irls_numba(object_pts, meas_2N,
                              qw, qx, qy, qz, tx, ty, tz,
                              fx, fy, cx, cy,
                              kind_int, c,
                              inv_sigma_2N,  # either None or (2N,) float64
                              z_eps=1e-3):
    N = object_pts.shape[0]
    R = _quat_to_R_numba(qw, qx, qy, qz)

    LtL = np.zeros((6, 6), dtype=np.float64)
    Lty = np.zeros(6, dtype=np.float64)
    y2 = 0.0

    front = 0

    eps = 1e-12
    if c <= 0.0:
        c = 1.0

    for i in range(N):
        X = object_pts[i, 0]
        Y = object_pts[i, 1]
        Z = object_pts[i, 2]

        rx = R[0, 0] * X + R[0, 1] * Y + R[0, 2] * Z
        ry = R[1, 0] * X + R[1, 1] * Y + R[1, 2] * Z
        rz = R[2, 0] * X + R[2, 1] * Y + R[2, 2] * Z

        x = rx + tx
        y = ry + ty
        z = rz + tz

        if z > z_eps:
            front += 1

        invz = 1.0 / z
        invz2 = invz * invz

        u = fx * (x * invz) + cx
        v = fy * (y * invz) + cy

        r2 = 2 * i
        du = meas_2N[r2 + 0] - u
        dv = meas_2N[r2 + 1] - v

        # base whitening from KF sigmas (2N)
        if inv_sigma_2N is None:
            iu = 1.0
            iv = 1.0
        else:
            iu = inv_sigma_2N[r2 + 0]
            iv = inv_sigma_2N[r2 + 1]

        # robust sw per feature from whitened magnitude (same as your robust kernel)
        if kind_int == 0:
            sw = 1.0
        else:
            rwu = du * iu
            rwv = dv * iv
            rmag = np.sqrt(rwu * rwu + rwv * rwv)

            if rmag < eps:
                w = 1.0
            elif kind_int == 1:  # huber
                w = 1.0 if rmag <= c else (c / rmag)
            elif kind_int == 2:  # cauchy
                t = rmag / c
                w = 1.0 / (1.0 + t * t)
            else:  # tukey
                t = rmag / c
                if t >= 1.0:
                    w = 0.0
                else:
                    a = 1.0 - t * t
                    w = a * a

            sw = np.sqrt(w)

        # final sqrt-weights for each residual row
        wu = iu * sw
        wv = iv * sw

        # weighted residuals
        ru = wu * du
        rv = wv * dv

        y2 += ru * ru + rv * rv

        # Jacobian row algebra (same as your _project_and_jacobian_numba)
        uX = fx * invz
        uZ = -fx * x * invz2
        vY = fy * invz
        vZ = -fy * y * invz2

        a = rx
        b = ry
        cR = rz

        # rot partials
        Lurx = uZ * b
        Lury = uX * cR - uZ * a
        Lurz = -uX * b

        Lvrx = -vY * cR + vZ * b
        Lvry = -vZ * a
        Lvrz = vY * a

        # trans partials
        Ju = (uX, 0.0, uZ)  # (tx,ty,tz) cols for u row
        Jv = (0.0, vY, vZ)  # (tx,ty,tz) cols for v row

        # build weighted Jacobian rows (6 elements each)
        ju0 = wu * Lurx
        ju1 = wu * Lury
        ju2 = wu * Lurz
        ju3 = wu * Ju[0]
        ju4 = wu * Ju[1]
        ju5 = wu * Ju[2]

        jv0 = wv * Lvrx
        jv1 = wv * Lvry
        jv2 = wv * Lvrz
        jv3 = wv * Jv[0]
        jv4 = wv * Jv[1]
        jv5 = wv * Jv[2]

        # Accumulate Lty += J^T r
        Lty[0] += ju0 * ru + jv0 * rv
        Lty[1] += ju1 * ru + jv1 * rv
        Lty[2] += ju2 * ru + jv2 * rv
        Lty[3] += ju3 * ru + jv3 * rv
        Lty[4] += ju4 * ru + jv4 * rv
        Lty[5] += ju5 * ru + jv5 * rv

        # Accumulate LtL += J^T J (two rows)
        # (manual outer-products, still tiny and fast at N<=100)
        # u-row
        LtL[0, 0] += ju0 * ju0
        LtL[0, 1] += ju0 * ju1
        LtL[0, 2] += ju0 * ju2
        LtL[0, 3] += ju0 * ju3
        LtL[0, 4] += ju0 * ju4
        LtL[0, 5] += ju0 * ju5
        LtL[1, 1] += ju1 * ju1
        LtL[1, 2] += ju1 * ju2
        LtL[1, 3] += ju1 * ju3
        LtL[1, 4] += ju1 * ju4
        LtL[1, 5] += ju1 * ju5
        LtL[2, 2] += ju2 * ju2
        LtL[2, 3] += ju2 * ju3
        LtL[2, 4] += ju2 * ju4
        LtL[2, 5] += ju2 * ju5
        LtL[3, 3] += ju3 * ju3
        LtL[3, 4] += ju3 * ju4
        LtL[3, 5] += ju3 * ju5
        LtL[4, 4] += ju4 * ju4
        LtL[4, 5] += ju4 * ju5
        LtL[5, 5] += ju5 * ju5

        # v-row
        LtL[0, 0] += jv0 * jv0
        LtL[0, 1] += jv0 * jv1
        LtL[0, 2] += jv0 * jv2
        LtL[0, 3] += jv0 * jv3
        LtL[0, 4] += jv0 * jv4
        LtL[0, 5] += jv0 * jv5
        LtL[1, 1] += jv1 * jv1
        LtL[1, 2] += jv1 * jv2
        LtL[1, 3] += jv1 * jv3
        LtL[1, 4] += jv1 * jv4
        LtL[1, 5] += jv1 * jv5
        LtL[2, 2] += jv2 * jv2
        LtL[2, 3] += jv2 * jv3
        LtL[2, 4] += jv2 * jv4
        LtL[2, 5] += jv2 * jv5
        LtL[3, 3] += jv3 * jv3
        LtL[3, 4] += jv3 * jv4
        LtL[3, 5] += jv3 * jv5
        LtL[4, 4] += jv4 * jv4
        LtL[4, 5] += jv4 * jv5
        LtL[5, 5] += jv5 * jv5

    # symmetrize LtL
    LtL[1, 0] = LtL[0, 1]
    LtL[2, 0] = LtL[0, 2]
    LtL[3, 0] = LtL[0, 3]
    LtL[4, 0] = LtL[0, 4]
    LtL[5, 0] = LtL[0, 5]
    LtL[2, 1] = LtL[1, 2]
    LtL[3, 1] = LtL[1, 3]
    LtL[4, 1] = LtL[1, 4]
    LtL[5, 1] = LtL[1, 5]
    LtL[3, 2] = LtL[2, 3]
    LtL[4, 2] = LtL[2, 4]
    LtL[5, 2] = LtL[2, 5]
    LtL[4, 3] = LtL[3, 4]
    LtL[5, 3] = LtL[3, 5]
    LtL[5, 4] = LtL[4, 5]

    front_frac = front / float(max(1, N))
    return LtL, Lty, y2, front_frac


# --- Small helpers --------------------------------------------------------------
import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def _quat_mul(qw1, qx1, qy1, qz1, qw2, qx2, qy2, qz2):
    # (q1 ⊗ q2)
    return (
        qw1 * qw2 - qx1 * qx2 - qy1 * qy2 - qz1 * qz2,
        qw1 * qx2 + qx1 * qw2 + qy1 * qz2 - qz1 * qy2,
        qw1 * qy2 - qx1 * qz2 + qy1 * qw2 + qz1 * qx2,
        qw1 * qz2 + qx1 * qy2 - qy1 * qx2 + qz1 * qw2,
    )


@njit(cache=True, fastmath=False)
def _quat_normalize_posw(qw, qx, qy, qz):
    n2 = qw * qw + qx * qx + qy * qy + qz * qz
    if n2 <= 0.0:
        return 1.0, 0.0, 0.0, 0.0
    invn = 1.0 / np.sqrt(n2)
    qw *= invn
    qx *= invn
    qy *= invn
    qz *= invn
    # enforce positive scalar (your force_s_pos)
    if qw < 0.0:
        qw = -qw
        qx = -qx
        qy = -qy
        qz = -qz
    return qw, qx, qy, qz


@njit(cache=True, fastmath=False)
def _rotvec_to_quat(drx, dry, drz):
    # Rodrigues/rotation-vector to quaternion
    th2 = drx * drx + dry * dry + drz * drz
    if th2 < 1e-24:
        # small-angle: sin(th/2)/th ~ 0.5
        return 1.0, 0.5 * drx, 0.5 * dry, 0.5 * drz
    th = np.sqrt(th2)
    half = 0.5 * th
    s = np.sin(half) / th
    return np.cos(half), drx * s, dry * s, drz * s


@njit(cache=True, fastmath=False)
def _apply_delta_q(qw, qx, qy, qz, drx, dry, drz):
    dqw, dqx, dqy, dqz = _rotvec_to_quat(drx, dry, drz)
    # left-multiply: q_new = dq ⊗ q
    nw, nx, ny, nz = _quat_mul(dqw, dqx, dqy, dqz, qw, qx, qy, qz)
    return _quat_normalize_posw(nw, nx, ny, nz)


@njit(parallel=True, fastmath=False, cache=True)
def _deriv_kernel_numba(RX: np.ndarray, xyz_cam: np.ndarray, fx: float, fy: float) -> np.ndarray:
    """
    Build L (2N x 6) for state [drx,dry,drz, tx,ty,tz].
    RX is R(q)X (N x 3), xyz_cam is RX + t (N x 3).
    """
    N = RX.shape[0]
    L = np.empty((2 * N, 6), dtype=np.float64)

    for i in prange(N):
        x = xyz_cam[i, 0]
        y = xyz_cam[i, 1]
        z = xyz_cam[i, 2]

        # projection partials
        invz = 1.0 / z
        invz2 = invz * invz

        uX = fx * invz
        uZ = -fx * x * invz2
        vY = fy * invz
        vZ = -fy * y * invz2

        a = RX[i, 0]  # RXx
        b = RX[i, 1]  # RXy
        c = RX[i, 2]  # RXz

        # L_rot = dUV_dXYZ @ (-skew(RX))
        # derived closed-form to avoid per-point matrix alloc:
        # row u:
        Lurx = uZ * b
        Lury = uX * c - uZ * a
        Lurz = -uX * b

        # row v:
        Lvrx = -vY * c + vZ * b
        Lvry = -vZ * a
        Lvrz = vY * a

        r = 2 * i

        # rotation cols
        L[r, 0] = Lurx
        L[r, 1] = Lury
        L[r, 2] = Lurz
        L[r + 1, 0] = Lvrx
        L[r + 1, 1] = Lvry
        L[r + 1, 2] = Lvrz

        # translation cols: dUV_dXYZ @ I
        L[r, 3] = uX
        L[r, 4] = 0.0
        L[r, 5] = uZ

        L[r + 1, 3] = 0.0
        L[r + 1, 4] = vY
        L[r + 1, 5] = vZ

    return L


@njit(cache=True, fastmath=False)
def _quat_to_R_numba(qw, qx, qy, qz):
    # Assumes q is unit-ish; still works if slightly off.
    # Returns 3x3 rotation matrix.
    ww = qw * qw
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz

    R = np.empty((3, 3), dtype=np.float64)
    R[0, 0] = ww + xx - yy - zz
    R[0, 1] = 2.0 * (xy - wz)
    R[0, 2] = 2.0 * (xz + wy)

    R[1, 0] = 2.0 * (xy + wz)
    R[1, 1] = ww - xx + yy - zz
    R[1, 2] = 2.0 * (yz - wx)

    R[2, 0] = 2.0 * (xz - wy)
    R[2, 1] = 2.0 * (yz + wx)
    R[2, 2] = ww - xx - yy + zz
    return R


@njit(parallel=True, cache=True, fastmath=False)
def _project_and_jacobian_numba(object_pts, qw, qx, qy, qz, tx, ty, tz, fx, fy, cx, cy,
                                proj_2N_out, RX_out, xyz_cam_out, L_out):
    """
    Fills:
      proj_2N_out: (2N,)
      RX_out      : (N,3)   = R*X
      xyz_cam_out : (N,3)   = R*X + t
      L_out       : (2N,6)  Jacobian wrt [drx,dry,drz, tx,ty,tz]
    """
    N = object_pts.shape[0]
    R = _quat_to_R_numba(qw, qx, qy, qz)

    for i in prange(N):
        X = object_pts[i, 0]
        Y = object_pts[i, 1]
        Z = object_pts[i, 2]

        rx = R[0, 0] * X + R[0, 1] * Y + R[0, 2] * Z
        ry = R[1, 0] * X + R[1, 1] * Y + R[1, 2] * Z
        rz = R[2, 0] * X + R[2, 1] * Y + R[2, 2] * Z

        RX_out[i, 0] = rx
        RX_out[i, 1] = ry
        RX_out[i, 2] = rz

        x = rx + tx
        y = ry + ty
        z = rz + tz

        xyz_cam_out[i, 0] = x
        xyz_cam_out[i, 1] = y
        xyz_cam_out[i, 2] = z

        # projection
        invz = 1.0 / z
        invz2 = invz * invz

        u = fx * (x * invz) + cx
        v = fy * (y * invz) + cy

        r2 = 2 * i
        proj_2N_out[r2] = u
        proj_2N_out[r2 + 1] = v

        # Jacobian (same algebra as your _deriv_kernel_numba) :contentReference[oaicite:3]{index=3}
        uX = fx * invz
        uZ = -fx * x * invz2
        vY = fy * invz
        vZ = -fy * y * invz2

        a = rx
        b = ry
        c = rz

        Lurx = uZ * b
        Lury = uX * c - uZ * a
        Lurz = -uX * b

        Lvrx = -vY * c + vZ * b
        Lvry = -vZ * a
        Lvrz = vY * a

        # rot cols
        L_out[r2, 0] = Lurx
        L_out[r2, 1] = Lury
        L_out[r2, 2] = Lurz
        L_out[r2 + 1, 0] = Lvrx
        L_out[r2 + 1, 1] = Lvry
        L_out[r2 + 1, 2] = Lvrz

        # trans cols
        L_out[r2, 3] = uX
        L_out[r2, 4] = 0.0
        L_out[r2, 5] = uZ

        L_out[r2 + 1, 3] = 0.0
        L_out[r2 + 1, 4] = vY
        L_out[r2 + 1, 5] = vZ


@njit(fastmath=False, cache=True)
def _accum_LtL_Lty_numba(L: np.ndarray, y: np.ndarray, sqrtw: np.ndarray):
    """
    Accumulate LtL and Lty for weighted least squares without forming Q or
    modifying L/y in-place.

    Inputs:
      L     : (2N,6)
      y     : (2N,)
      sqrtw : (2N,)  left-multipliers (sqrt weights)

    Returns:
      LtL : (6,6)
      Lty : (6,)
      y2  : scalar sum of squares of weighted residuals (||Q y||^2)
    """
    # Per-thread partials to reduce contention (numba supports this pattern)
    # Shape: (nthreads, 6, 6) etc would be ideal, but numba doesn't expose nthreads
    # reliably in all configs. We'll do a manual reduction via prange over rows and
    # use local accumulators + atomic add pattern on a small array.

    LtL = np.zeros((6, 6), dtype=np.float64)
    Lty = np.zeros(6, dtype=np.float64)
    y2 = 0.0

    M = L.shape[0]

    for i in range(M):
        wi = sqrtw[i]
        yi = wi * y[i]

        # weighted Jacobian row
        r0 = wi * L[i, 0]
        r1 = wi * L[i, 1]
        r2 = wi * L[i, 2]
        r3 = wi * L[i, 3]
        r4 = wi * L[i, 4]
        r5 = wi * L[i, 5]

        # Accumulate Lty
        # (J^T r) where r == yi (scalar residual for this row)
        Lty[0] += r0 * yi
        Lty[1] += r1 * yi
        Lty[2] += r2 * yi
        Lty[3] += r3 * yi
        Lty[4] += r4 * yi
        Lty[5] += r5 * yi

        # Accumulate LtL (outer product of weighted row)
        # Fill upper triangle then mirror (cheaper)
        LtL[0, 0] += r0 * r0
        LtL[0, 1] += r0 * r1
        LtL[0, 2] += r0 * r2
        LtL[0, 3] += r0 * r3
        LtL[0, 4] += r0 * r4
        LtL[0, 5] += r0 * r5

        LtL[1, 1] += r1 * r1
        LtL[1, 2] += r1 * r2
        LtL[1, 3] += r1 * r3
        LtL[1, 4] += r1 * r4
        LtL[1, 5] += r1 * r5

        LtL[2, 2] += r2 * r2
        LtL[2, 3] += r2 * r3
        LtL[2, 4] += r2 * r4
        LtL[2, 5] += r2 * r5

        LtL[3, 3] += r3 * r3
        LtL[3, 4] += r3 * r4
        LtL[3, 5] += r3 * r5

        LtL[4, 4] += r4 * r4
        LtL[4, 5] += r4 * r5

        LtL[5, 5] += r5 * r5

        y2 += yi * yi

    # Mirror upper -> lower
    for r in range(6):
        for c in range(r + 1, 6):
            LtL[c, r] = LtL[r, c]

    return LtL, Lty, y2


def _row_normed(A, eps=1e-12):
    """Row-normalize a 2D array.

    Each row is divided by its L2 norm; very small norms are clamped by *eps* to
    avoid division by ~0.
    """
    n = np.linalg.norm(A, axis=1, keepdims=True)
    return A / np.clip(n, eps, None)


# --- Camera projection ----------------------------------------------------------

def _get_cal_distortion(cal: Calibration) -> tuple[float, float, float, float, float]:
    """
    Returns Brown-Conrady distortion in the internal order:
      (k1, k2, k3, p1, p2)

    Missing attributes default to zero so old Calibration objects still work.
    """

    def safe_float_attr(obj, name, default=0.0):
        v = getattr(obj, name, default)
        return float(default if v is None else v)
    return (
        safe_float_attr(cal, "k1", 0.0),
        safe_float_attr(cal, "k2", 0.0),
        safe_float_attr(cal, "k3", 0.0),
        safe_float_attr(cal, "p1", 0.0),
        safe_float_attr(cal, "p2", 0.0),
    )


def _opencv_dist_coeffs_from_cal(cal: Calibration) -> np.ndarray:
    """
    OpenCV distortion order is [k1, k2, p1, p2, k3].
    """
    k1, k2, k3, p1, p2 = _get_cal_distortion(cal)
    return np.array([k1, k2, p1, p2, k3], dtype=np.float64).reshape(-1, 1)


def _project_points_brown_from_xyz(XYZ: np.ndarray, cal: Calibration) -> np.ndarray:
    """
    Brown-Conrady projection from camera-frame XYZ -> pixels.

    Distortion model:
      x_d = x_n * radial + 2*p1*x_n*y_n + p2*(r^2 + 2*x_n^2)
      y_d = y_n * radial + p1*(r^2 + 2*y_n^2) + 2*p2*x_n*y_n
      radial = 1 + k1*r^2 + k2*r^4 + k3*r^6
    """
    XYZ = np.asarray(XYZ, dtype=np.float64)
    X = XYZ[:, 0]
    Y = XYZ[:, 1]
    Z = XYZ[:, 2]

    # avoid blowups at/near z=0
    Zs = np.where(np.abs(Z) < 1e-12, np.where(Z >= 0.0, 1e-12, -1e-12), Z)

    xn = X / Zs
    yn = Y / Zs

    k1, k2, k3, p1, p2 = _get_cal_distortion(cal)

    r2 = xn * xn + yn * yn
    r4 = r2 * r2
    r6 = r4 * r2

    radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6

    xd = xn * radial + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn)
    yd = yn * radial + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn

    uv = np.empty((XYZ.shape[0], 2), dtype=np.float64)
    uv[:, 0] = float(cal.fx) * xd + float(cal.cx)
    uv[:, 1] = float(cal.fy) * yd + float(cal.cy)
    return uv


# --- Camera projection ----------------------------------------------------------

def h(est_q: q, est_t: NDArray, feature_points, cal: Calibration):
    """
    Project all model points into pixel coordinates using Brown-Conrady distortion.

    Returns a flattened length-2N vector [u0, v0, u1, v1, ...].
    """
    XYZ_proj = est_q * feature_points + est_t
    return _project_points_brown_from_xyz(XYZ_proj, cal).reshape(-1)


def _skew(v: np.ndarray) -> np.ndarray:
    """Return 3x3 skew-symmetric matrix [v]_x such that [v]_x w = v × w."""
    vx, vy, vz = v
    return np.array([
        [0.0, -vz, vy],
        [vz, 0.0, -vx],
        [-vy, vx, 0.0]
    ], dtype=float)


# --- Analytic Jacobian of h w.r.t. (q, t) -------------------------------------

def deriv(est_q: q, est_t: np.ndarray, feature_points: np.ndarray, cal: Calibration):
    """
    Jacobian for 2D reprojection residuals wrt [drx,dry,drz, tx,ty,tz].

    Uses Numba kernel when available. Critical: avoid per-call astype() copies.
    """
    # Compute camera-frame points using your existing quaternion/vector plumbing.
    # xyz_cam: (N,3) ; RX: (N,3) == R(q)X
    xyz_cam = est_q * feature_points + est_t
    RX = xyz_cam - est_t

    # Ensure float64 + C-contiguous only if necessary (avoid unconditional copies)
    if xyz_cam.dtype != np.float64 or not xyz_cam.flags["C_CONTIGUOUS"]:
        xyz_cam = np.ascontiguousarray(xyz_cam, dtype=np.float64)
    if RX.dtype != np.float64 or not RX.flags["C_CONTIGUOUS"]:
        RX = np.ascontiguousarray(RX, dtype=np.float64)

    return _deriv_kernel_numba(RX, xyz_cam, float(cal.fx), float(cal.fy))


def print_rayPts(ray_proj: NDArray):
    """Nicely print a flattened [u0, v0, u1, v1, ...] vector (debug helper)."""
    ray_proj = ray_proj.reshape(-1, 2)
    print(f"Norm: {np.linalg.norm(ray_proj)}")
    for n, ray in enumerate(ray_proj):
        print(f"Feature: {n:3d}, px: {ray[0]: .5f}, py: {ray[1]: .5f}")


def _expand_to_2N_weights(w, N):
    """Accept per-point (N,) or per-residual (2N,) weights; return (2N,)."""
    w = np.asarray(w, dtype=float).reshape(-1)
    if w.size == N:
        return np.repeat(w, 2)
    if w.size == 2 * N:
        return w
    raise ValueError(f"weight length must be N or 2N; got {w.size}, N={N}")


@njit(cache=True, fastmath=True)
def _robust_sqrt_weights_inplace_numba(
        y_2N, N, kind_int, c, inv_sigma_2N, out_sqrtw_2N
):
    """
    Fill out_sqrtw_2N (2N,) with per-residual sqrt-weights for robust IRLS.
    Robust is computed per-feature from 2D residual magnitude.

    kind_int: 0=none, 1=huber, 2=cauchy, 3=tukey
    If inv_sigma_2N is provided, robust operates on whitened residuals:
        r = sqrt((du*iu)^2 + (dv*iv)^2)
    """
    eps = 1e-12
    if c <= 0.0:
        c = 1.0

    for i in range(N):
        du = y_2N[2 * i + 0]
        dv = y_2N[2 * i + 1]

        # Robust decision variable: whitened residual magnitude if available
        if inv_sigma_2N is not None:
            iu = inv_sigma_2N[2 * i + 0]
            iv = inv_sigma_2N[2 * i + 1]
            ru = du * iu
            rv = dv * iv
            r = (ru * ru + rv * rv) ** 0.5
        else:
            r = (du * du + dv * dv) ** 0.5

        # Weight function w(r) (NOT sqrt yet)
        if kind_int == 0 or r < eps:
            w = 1.0

        elif kind_int == 1:  # huber
            if r <= c:
                w = 1.0
            else:
                w = c / r

        elif kind_int == 2:  # cauchy
            t = r / c
            w = 1.0 / (1.0 + t * t)

        else:  # kind_int == 3: tukey
            t = r / c
            if t >= 1.0:
                w = 0.0
            else:
                a = 1.0 - t * t
                w = a * a

        sw = w ** 0.5
        out_sqrtw_2N[2 * i + 0] = sw
        out_sqrtw_2N[2 * i + 1] = sw


def _inv_sigma_2N_from_sigma(sigma_2N,
                             N: int,
                             eps: float = 1e-6,
                             big: float = 1e6):
    """
    Returns inv_sigma_2N (2N,) where inv_sigma[i] = 1/sigma[i].
    Accepts sigma length N or 2N. Missing/invalid -> big sigma -> tiny inv weight.
    """
    if sigma_2N is None:
        return None

    s = np.asarray(sigma_2N, dtype=np.float64).ravel()
    if s.size == N:
        s = np.repeat(s, 2)
    if s.size != 2 * N:
        raise ValueError(f"sigma_2N must be N or 2N; got {s.size}, N={N}")

    s = s.copy()
    bad = (~np.isfinite(s)) | (s <= 0.0)
    s[bad] = big
    s = np.maximum(s, eps)
    return 1.0 / s


def opt(
        img_pts: NDArray,
        object_pts: NDArray,
        cal: Calibration,
        return_stats: bool,
        seed_q: q = None,
        seed_t: NDArray = None,
        robust_kind: robust_cost = robust_cost.none,
        robust_param: float = 2.0,
        sigma_2N=None,
        sigma_floor_px: float = 1.0,
        max_iters: int = 20):
    """
    Refine pose to minimize ||meas_pix - h(q,t)|| using weighted GN/LM.
    State: [δr, δt] (6 DOF), minimal tangent update.
    """

    # ----------------- Initialization -----------------
    if seed_q is None or seed_t is None:
        est_q, est_t = DLT(object_pts, img_pts, cal)
    else:
        est_q = seed_q.copy()
        est_t = seed_t.copy()

    # Ensure float64, contiguous
    meas_pix = np.ascontiguousarray(img_pts.reshape(-1), dtype=np.float64)
    N = int(img_pts.shape[0])
    object_pts64 = np.ascontiguousarray(object_pts, dtype=np.float64)

    # ----------- Sigma whitening (once) -----------
    inv_sigma_2N = _inv_sigma_2N_from_sigma(sigma_2N, N)
    if inv_sigma_2N is not None:
        sf = float(sigma_floor_px)
        if (not np.isfinite(sf)) or (sf <= 0.0):
            sf = 1.0
        inv_sigma_2N = np.minimum(inv_sigma_2N, 1.0 / sf)

    # Robust enum → int (matches your existing mapping) :contentReference[oaicite:1]{index=1}
    kind_int = 0
    if robust_kind == robust_cost.huber:
        kind_int = 1
    elif robust_kind == robust_cost.cauchy:
        kind_int = 2
    elif robust_kind == robust_cost.tukey:
        kind_int = 3

    # Float-state (no quaternion objects in-loop)
    qw, qx, qy, qz = float(est_q.s), float(est_q.vec[0]), float(est_q.vec[1]), float(est_q.vec[2])
    tx, ty, tz = float(est_t[0]), float(est_t[1]), float(est_t[2])

    lam = 1e-1

    # ================= GN/LM LOOP =================
    for iter_num in range(1, max_iters + 1):
        LtL, Lty, old_y2, front_frac = _project_accum_irls_numba(
            object_pts64, meas_pix,
            qw, qx, qy, qz, tx, ty, tz,
            float(cal.fx), float(cal.fy), float(cal.cx), float(cal.cy),
            int(kind_int), float(robust_param),
            inv_sigma_2N
        )

        # chirality check (same intent as before, but fused)
        if front_frac < 0.9:
            lam *= 3.0
            continue

        # LM solve: (LtL + lam I) dx = Lty
        A = LtL.copy()
        A[0, 0] += lam
        A[1, 1] += lam
        A[2, 2] += lam
        A[3, 3] += lam
        A[4, 4] += lam
        A[5, 5] += lam
        dx = np.linalg.solve(A, Lty)

        drx, dry, drz = float(dx[0]), float(dx[1]), float(dx[2])
        dtx, dty, dtz = float(dx[3]), float(dx[4]), float(dx[5])

        tqw, tqx, tqy, tqz = _apply_delta_q(qw, qx, qy, qz, drx, dry, drz)
        ttx, tty, ttz = tx + dtx, ty + dty, tz + dtz

        _LtL2, _Lty2, new_y2, front2 = _project_accum_irls_numba(
            object_pts64, meas_pix,
            tqw, tqx, tqy, tqz, ttx, tty, ttz,
            float(cal.fx), float(cal.fy), float(cal.cx), float(cal.cy),
            int(kind_int), float(robust_param),
            inv_sigma_2N
        )

        if front2 < 0.9:
            lam *= 3.0
            continue

        # Predicted reduction (standard LM; avoids needing L.dot(dx))
        pred = 0.5 * float(dx.dot(lam * dx + Lty))
        if pred <= 0.0 or (not np.isfinite(pred)):
            lam *= 3.0
            continue

        rho = float((old_y2 - new_y2) / pred)

        if rho > 0.25 and new_y2 < old_y2:
            # accept
            qw, qx, qy, qz = tqw, tqx, tqy, tqz
            tx, ty, tz = ttx, tty, ttz
            lam *= 0.3
        else:
            lam *= 3.0

        # stop: small step or small improvement
        if np.linalg.norm(dx) < 1e-7:
            break
        if abs(old_y2 - new_y2) / max(1e-12, old_y2) < 1e-6:
            break

    # Wrap back into quaternion object ONCE
    est_q = q(qw, np.array([qx, qy, qz], dtype=np.float64))
    est_t = np.array([tx, ty, tz], dtype=np.float64)
    est_q.force_s_pos()  # safe; should already be positive-w

    if not return_stats:
        return est_q, est_t

    # Final stats at converged pose
    LtL, _Lty, y2, _front = _project_accum_irls_numba(
        object_pts64, meas_pix,
        float(est_q.s), float(est_q.vec[0]), float(est_q.vec[1]), float(est_q.vec[2]),
        float(est_t[0]), float(est_t[1]), float(est_t[2]),
        float(cal.fx), float(cal.fy), float(cal.cx), float(cal.cy),
        int(kind_int), float(robust_param),
        inv_sigma_2N
    )

    dof = 2 * N - 6
    if dof < 1:
        dof = 1
    s2 = float(y2) / float(dof)

    I = np.eye(6, dtype=np.float64)
    cov6 = s2 * np.linalg.solve(LtL, I)

    return est_q, est_t, QnPStats(
        N=int(N),
        dof=int(dof),
        sse_w=float(y2),
        s2=float(s2),
        cov6=cov6
    )

############################################ REGULAR OPT PATH, NO ONLINE CAL END ######################################


############################################ SUPER OPT PATH WITH ONLINE CAL ###########################################
# =========================
# Joint pose + intrinsics + Brown-Conrady distortion
# =========================

@dataclass
class JointPnPStats:
    N: int
    dof: int
    sse_meas_w: float
    sse_prior: float
    sse_total: float
    s2: float
    cov15: np.ndarray               # full solver covariance in state [pose6, cal9]
    cov9: np.ndarray                # calibration block covariance in solver state
    cal_state: np.ndarray           # [log_fx, log_fy, cx, cy, k1, k2, k3, p1, p2]

    # Backward-ish compatibility fields
    cov10: np.ndarray | None = None
    cov4: np.ndarray | None = None
    intr_state: np.ndarray | None = None


def _intr_dist_state_from_cal(cal: Calibration) -> np.ndarray:
    fx = float(cal.fx)
    fy = float(cal.fy)
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"Calibration focal lengths must be positive, got fx={fx}, fy={fy}")

    k1 = float(getattr(cal, "k1", 0.0))
    k2 = float(getattr(cal, "k2", 0.0))
    k3 = float(getattr(cal, "k3", 0.0))
    p1 = float(getattr(cal, "p1", 0.0))
    p2 = float(getattr(cal, "p2", 0.0))

    return np.array(
        [np.log(fx), np.log(fy), float(cal.cx), float(cal.cy), k1, k2, k3, p1, p2],
        dtype=np.float64
    )


def _intr_dist_from_state(theta9: np.ndarray) -> tuple[float, float, float, float, float, float, float, float, float]:
    log_fx, log_fy, cx, cy, k1, k2, k3, p1, p2 = [float(v) for v in np.asarray(theta9, dtype=np.float64).reshape(9)]
    fx = float(np.exp(log_fx))
    fy = float(np.exp(log_fy))
    return fx, fy, cx, cy, k1, k2, k3, p1, p2


def _clone_cal_with_intrinsics_and_distortion(
    cal: Calibration,
    fx: float, fy: float, cx: float, cy: float,
    k1: float, k2: float, k3: float, p1: float, p2: float
) -> Calibration:
    import copy
    cal_out = copy.deepcopy(cal)
    cal_out.fx = float(fx)
    cal_out.fy = float(fy)
    cal_out.cx = float(cx)
    cal_out.cy = float(cy)
    cal_out.k1 = float(k1)
    cal_out.k2 = float(k2)
    cal_out.k3 = float(k3)
    cal_out.p1 = float(p1)
    cal_out.p2 = float(p2)
    return cal_out


def _prior_info_from_cov_or_info(
    prior_cov: np.ndarray | None = None,
    prior_info: np.ndarray | None = None,
    state_dim: int = 9,
) -> np.ndarray:
    if prior_info is not None:
        I = np.asarray(prior_info, dtype=np.float64)
        if I.shape != (state_dim, state_dim):
            raise ValueError(f"prior_info must be {state_dim}x{state_dim}, got {I.shape}")
        return I

    if prior_cov is not None:
        P = np.asarray(prior_cov, dtype=np.float64)
        if P.shape != (state_dim, state_dim):
            raise ValueError(f"prior_cov must be {state_dim}x{state_dim}, got {P.shape}")
        return np.linalg.pinv(P)

    return np.zeros((state_dim, state_dim), dtype=np.float64)


def _prior_cost_and_rhs(theta: np.ndarray, prior_mean: np.ndarray | None, prior_info: np.ndarray | None):
    """
    Prior residual is:
        r_prior = prior_mean - theta
    with information matrix prior_info.
    """
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    n = theta.size

    if prior_mean is None or prior_info is None:
        return np.zeros((n, n), dtype=np.float64), np.zeros(n, dtype=np.float64), 0.0

    d = np.asarray(prior_mean, dtype=np.float64).reshape(n) - theta
    H = np.asarray(prior_info, dtype=np.float64).reshape(n, n)
    g = H @ d
    c = float(d @ H @ d)
    return H, g, c


def _default_raw_dist_cov5() -> np.ndarray:
    """
    Default raw-state prior covariance for [k1, k2, k3, p1, p2].

    These are intentionally loose enough to move,
    but not so loose that single-view optimization explodes into interpretive dance.
    """
    stds = np.array([0.10, 0.05, 0.02, 0.005, 0.005], dtype=np.float64)
    return np.diag(stds * stds)


def _build_prior_mean_cov9_from_cal(cal: Calibration) -> tuple[np.ndarray, np.ndarray]:
    """
    Builds solver-space prior for:
      theta9 = [log_fx, log_fy, cx, cy, k1, k2, k3, p1, p2]

    Expected raw-space covariance ordering if present:
      [fx, fy, cx, cy, k1, k2, k3, p1, p2]
    """
    fx = float(cal.fx)
    fy = float(cal.fy)
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"Calibration focal lengths must be positive, got fx={fx}, fy={fy}")

    k1 = float(getattr(cal, "k1", 0.0))
    k2 = float(getattr(cal, "k2", 0.0))
    k3 = float(getattr(cal, "k3", 0.0))
    p1 = float(getattr(cal, "p1", 0.0))
    p2 = float(getattr(cal, "p2", 0.0))

    prior_mean9 = np.array(
        [np.log(fx), np.log(fy), float(cal.cx), float(cal.cy), k1, k2, k3, p1, p2],
        dtype=np.float64
    )

    raw_cov9 = np.zeros((9, 9), dtype=np.float64)
    cov_intr_dist = getattr(cal, "cov_intr_dist", None)

    if cov_intr_dist is not None:
        cov_intr_dist = np.asarray(cov_intr_dist, dtype=np.float64)
        if cov_intr_dist.shape != (9, 9):
            raise ValueError(f"cal.cov_intr_dist must be 9x9, got {cov_intr_dist.shape}")
        raw_cov9[:, :] = cov_intr_dist
    else:
        cov4 = getattr(cal, "cov", None)
        if cov4 is not None:
            cov4 = np.asarray(cov4, dtype=np.float64)
            if cov4.shape != (4, 4):
                raise ValueError(f"cal.cov must be 4x4 when used as intrinsics prior, got {cov4.shape}")
            raw_cov9[:4, :4] = cov4
        else:
            raw_cov9[:4, :4] = np.diag(np.array([100.0, 100.0, 50.0, 50.0], dtype=np.float64) ** 2)

        raw_cov9[4:, 4:] = _default_raw_dist_cov5()

    # raw -> solver state transform
    # raw:    [fx, fy, cx, cy, k1, k2, k3, p1, p2]
    # solver: [log_fx, log_fy, cx, cy, k1, k2, k3, p1, p2]
    G = np.eye(9, dtype=np.float64)
    G[0, 0] = 1.0 / fx
    G[1, 1] = 1.0 / fy

    prior_cov9 = G @ raw_cov9 @ G.T
    return prior_mean9, prior_cov9


@njit(cache=True, fastmath=False)
def _brown_conrady_norm_and_partials(xn, yn, k1, k2, k3, p1, p2):
    """
    Returns:
      xd, yd,
      dxd_dxn, dxd_dyn,
      dyd_dxn, dyd_dyn,
      dxd_dk1, dxd_dk2, dxd_dk3, dxd_dp1, dxd_dp2,
      dyd_dk1, dyd_dk2, dyd_dk3, dyd_dp1, dyd_dp2
    """
    r2 = xn * xn + yn * yn
    r4 = r2 * r2
    r6 = r4 * r2

    radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6

    xd = xn * radial + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn)
    yd = yn * radial + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn

    common = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4
    drad_dxn = 2.0 * xn * common
    drad_dyn = 2.0 * yn * common

    dxd_dxn = radial + xn * drad_dxn + 2.0 * p1 * yn + 6.0 * p2 * xn
    dxd_dyn = xn * drad_dyn + 2.0 * p1 * xn + 2.0 * p2 * yn

    dyd_dxn = yn * drad_dxn + 2.0 * p1 * xn + 2.0 * p2 * yn
    dyd_dyn = radial + yn * drad_dyn + 6.0 * p1 * yn + 2.0 * p2 * xn

    dxd_dk1 = xn * r2
    dxd_dk2 = xn * r4
    dxd_dk3 = xn * r6
    dxd_dp1 = 2.0 * xn * yn
    dxd_dp2 = r2 + 2.0 * xn * xn

    dyd_dk1 = yn * r2
    dyd_dk2 = yn * r4
    dyd_dk3 = yn * r6
    dyd_dp1 = r2 + 2.0 * yn * yn
    dyd_dp2 = 2.0 * xn * yn

    return (
        xd, yd,
        dxd_dxn, dxd_dyn,
        dyd_dxn, dyd_dyn,
        dxd_dk1, dxd_dk2, dxd_dk3, dxd_dp1, dxd_dp2,
        dyd_dk1, dyd_dk2, dyd_dk3, dyd_dp1, dyd_dp2
    )


@njit(cache=True, fastmath=False)
def _project_accum_joint_irls_dist_numba(
        object_pts, meas_2N,
        qw, qx, qy, qz, tx, ty, tz,
        log_fx, log_fy, cx, cy, k1, k2, k3, p1, p2,
        kind_int, c,
        inv_sigma_2N,
        z_eps=1e-3):
    """
    Joint accumulator for state:
      [drx, dry, drz, tx, ty, tz,
       dlog_fx, dlog_fy, dcx, dcy,
       dk1, dk2, dk3, dp1, dp2]
    """
    N = object_pts.shape[0]
    R = _quat_to_R_numba(qw, qx, qy, qz)

    fx = np.exp(log_fx)
    fy = np.exp(log_fy)

    LtL = np.zeros((15, 15), dtype=np.float64)
    Lty = np.zeros(15, dtype=np.float64)
    y2 = 0.0
    front = 0

    eps = 1e-12
    if c <= 0.0:
        c = 1.0

    for i in range(N):
        X = object_pts[i, 0]
        Y = object_pts[i, 1]
        Z = object_pts[i, 2]

        rx = R[0, 0] * X + R[0, 1] * Y + R[0, 2] * Z
        ry = R[1, 0] * X + R[1, 1] * Y + R[1, 2] * Z
        rz = R[2, 0] * X + R[2, 1] * Y + R[2, 2] * Z

        x = rx + tx
        y = ry + ty
        z = rz + tz

        if z > z_eps:
            front += 1

        invz = 1.0 / z
        xn = x * invz
        yn = y * invz

        (xd, yd,
         dxd_dxn, dxd_dyn,
         dyd_dxn, dyd_dyn,
         dxd_dk1, dxd_dk2, dxd_dk3, dxd_dp1, dxd_dp2,
         dyd_dk1, dyd_dk2, dyd_dk3, dyd_dp1, dyd_dp2) = _brown_conrady_norm_and_partials(
            xn, yn, k1, k2, k3, p1, p2
        )

        u = fx * xd + cx
        v = fy * yd + cy

        r2i = 2 * i
        du = meas_2N[r2i + 0] - u
        dv = meas_2N[r2i + 1] - v

        if inv_sigma_2N is None:
            iu = 1.0
            iv = 1.0
        else:
            iu = inv_sigma_2N[r2i + 0]
            iv = inv_sigma_2N[r2i + 1]

        if kind_int == 0:
            sw = 1.0
        else:
            rwu = du * iu
            rwv = dv * iv
            rmag = np.sqrt(rwu * rwu + rwv * rwv)

            if rmag < eps:
                w = 1.0
            elif kind_int == 1:  # huber
                w = 1.0 if rmag <= c else (c / rmag)
            elif kind_int == 2:  # cauchy
                t = rmag / c
                w = 1.0 / (1.0 + t * t)
            else:  # tukey
                t = rmag / c
                if t >= 1.0:
                    w = 0.0
                else:
                    a = 1.0 - t * t
                    w = a * a

            sw = np.sqrt(w)

        wu = iu * sw
        wv = iv * sw

        ru = wu * du
        rv = wv * dv

        y2 += ru * ru + rv * rv

        # normalized-coordinate derivatives
        dxn_dx = invz
        dxn_dy = 0.0
        dxn_dz = -xn * invz

        dyn_dx = 0.0
        dyn_dy = invz
        dyn_dz = -yn * invz

        # chain rule: XYZ -> normalized -> distorted normalized -> pixel
        du_dX = fx * (dxd_dxn * dxn_dx + dxd_dyn * dyn_dx)
        du_dY = fx * (dxd_dxn * dxn_dy + dxd_dyn * dyn_dy)
        du_dZ = fx * (dxd_dxn * dxn_dz + dxd_dyn * dyn_dz)

        dv_dX = fy * (dyd_dxn * dxn_dx + dyd_dyn * dyn_dx)
        dv_dY = fy * (dyd_dxn * dxn_dy + dyd_dyn * dyn_dy)
        dv_dZ = fy * (dyd_dxn * dxn_dz + dyd_dyn * dyn_dz)

        a = rx
        b = ry
        cR = rz

        # row gradient g dotted into (-skew(RX))
        Lurx = b * du_dZ - cR * du_dY
        Lury = cR * du_dX - a * du_dZ
        Lurz = a * du_dY - b * du_dX

        Lvrx = b * dv_dZ - cR * dv_dY
        Lvry = cR * dv_dX - a * dv_dZ
        Lvrz = a * dv_dY - b * dv_dX

        ju = np.empty(15, dtype=np.float64)
        jv = np.empty(15, dtype=np.float64)

        # pose cols
        ju[0] = wu * Lurx
        ju[1] = wu * Lury
        ju[2] = wu * Lurz
        ju[3] = wu * du_dX
        ju[4] = wu * du_dY
        ju[5] = wu * du_dZ

        jv[0] = wv * Lvrx
        jv[1] = wv * Lvry
        jv[2] = wv * Lvrz
        jv[3] = wv * dv_dX
        jv[4] = wv * dv_dY
        jv[5] = wv * dv_dZ

        # intrinsics cols
        ju[6] = wu * (fx * xd)   # du / d log_fx
        ju[7] = 0.0
        ju[8] = wu * 1.0         # du / d cx
        ju[9] = 0.0

        jv[6] = 0.0
        jv[7] = wv * (fy * yd)   # dv / d log_fy
        jv[8] = 0.0
        jv[9] = wv * 1.0         # dv / d cy

        # distortion cols
        ju[10] = wu * (fx * dxd_dk1)
        ju[11] = wu * (fx * dxd_dk2)
        ju[12] = wu * (fx * dxd_dk3)
        ju[13] = wu * (fx * dxd_dp1)
        ju[14] = wu * (fx * dxd_dp2)

        jv[10] = wv * (fy * dyd_dk1)
        jv[11] = wv * (fy * dyd_dk2)
        jv[12] = wv * (fy * dyd_dk3)
        jv[13] = wv * (fy * dyd_dp1)
        jv[14] = wv * (fy * dyd_dp2)

        # accumulate g = J^T r
        for cidx in range(15):
            Lty[cidx] += ju[cidx] * ru + jv[cidx] * rv

        # accumulate H = J^T J
        for rix in range(15):
            jru = ju[rix]
            jrv = jv[rix]
            for cix in range(rix, 15):
                LtL[rix, cix] += jru * ju[cix] + jrv * jv[cix]

    # symmetrize
    for rix in range(15):
        for cix in range(rix + 1, 15):
            LtL[cix, rix] = LtL[rix, cix]

    front_frac = front / float(max(1, N))
    return LtL, Lty, y2, front_frac


def opt_pose_and_intrinsics(
        img_pts: NDArray,
        object_pts: NDArray,
        cal: Calibration,
        return_stats: bool,
        seed_q: q = None,
        seed_t: NDArray = None,
        prior_mean9: np.ndarray | None = None,   # [log_fx, log_fy, cx, cy, k1, k2, k3, p1, p2]
        prior_cov9: np.ndarray | None = None,
        prior_info9: np.ndarray | None = None,
        robust_kind: robust_cost = robust_cost.none,
        robust_param: float = 2.0,
        sigma_2N=None,
        sigma_floor_px: float = 1.0,
        max_iters: int = 20,
        pose_only_warmstart_iters: int = 3):
    """
    Joint LM refinement of pose + intrinsics + Brown-Conrady distortion.

    Solver state:
      [drx, dry, drz, tx, ty, tz,
       dlog_fx, dlog_fy, dcx, dcy,
       dk1, dk2, dk3, dp1, dp2]
    """

    # ----------------- Initialization -----------------
    if seed_q is None or seed_t is None:
        est_q, est_t = DLT(object_pts, img_pts, cal)
    else:
        est_q = seed_q.copy()
        est_t = np.asarray(seed_t, dtype=np.float64).copy()

    meas_pix = np.ascontiguousarray(img_pts.reshape(-1), dtype=np.float64)
    object_pts64 = np.ascontiguousarray(object_pts, dtype=np.float64)
    N = int(img_pts.shape[0])

    inv_sigma_2N = _inv_sigma_2N_from_sigma(sigma_2N, N)
    if inv_sigma_2N is not None:
        sf = float(sigma_floor_px)
        if (not np.isfinite(sf)) or (sf <= 0.0):
            sf = 1.0
        inv_sigma_2N = np.minimum(inv_sigma_2N, 1.0 / sf)

    kind_int = 0
    if robust_kind == robust_cost.huber:
        kind_int = 1
    elif robust_kind == robust_cost.cauchy:
        kind_int = 2
    elif robust_kind == robust_cost.tukey:
        kind_int = 3

    qw, qx, qy, qz = float(est_q.s), float(est_q.vec[0]), float(est_q.vec[1]), float(est_q.vec[2])
    tx, ty, tz = float(est_t[0]), float(est_t[1]), float(est_t[2])

    theta9 = _intr_dist_state_from_cal(cal) if prior_mean9 is None else np.asarray(prior_mean9, dtype=np.float64).reshape(9).copy()
    prior_mean9_use = theta9.copy() if prior_mean9 is None else np.asarray(prior_mean9, dtype=np.float64).reshape(9).copy()
    prior_info9_use = _prior_info_from_cov_or_info(prior_cov=prior_cov9, prior_info=prior_info9, state_dim=9)

    lam = 1e-1

    # ----------------- LM Loop -----------------
    for iter_num in range(1, max_iters + 1):
        LtL, Lty, meas_old_y2, front_frac = _project_accum_joint_irls_dist_numba(
            object_pts64, meas_pix,
            qw, qx, qy, qz, tx, ty, tz,
            float(theta9[0]), float(theta9[1]), float(theta9[2]), float(theta9[3]),
            float(theta9[4]), float(theta9[5]), float(theta9[6]), float(theta9[7]), float(theta9[8]),
            int(kind_int), float(robust_param),
            inv_sigma_2N
        )

        if front_frac < 0.9:
            lam *= 3.0
            continue

        H_prior9, g_prior9, prior_old_y2 = _prior_cost_and_rhs(theta9, prior_mean9_use, prior_info9_use)

        H = LtL.copy()
        g = Lty.copy()
        H[6:15, 6:15] += H_prior9
        g[6:15] += g_prior9

        total_old_y2 = float(meas_old_y2 + prior_old_y2)

        dx = np.zeros(15, dtype=np.float64)

        # warmstart the basin with pose-only updates for a few iterations
        if iter_num <= int(max(0, pose_only_warmstart_iters)):
            A6 = H[:6, :6].copy()
            for k in range(6):
                A6[k, k] += lam
            g6 = g[:6]
            dx6 = np.linalg.solve(A6, g6)
            dx[:6] = dx6
            pred = 0.5 * float(dx6.dot(lam * dx6 + g6))
        else:
            A = H.copy()
            for k in range(15):
                A[k, k] += lam
            dx = np.linalg.solve(A, g)
            pred = 0.5 * float(dx.dot(lam * dx + g))

        if pred <= 0.0 or (not np.isfinite(pred)):
            lam *= 3.0
            continue

        drx, dry, drz = float(dx[0]), float(dx[1]), float(dx[2])
        dtx, dty, dtz = float(dx[3]), float(dx[4]), float(dx[5])

        tqw, tqx, tqy, tqz = _apply_delta_q(qw, qx, qy, qz, drx, dry, drz)
        ttx, tty, ttz = tx + dtx, ty + dty, tz + dtz

        ttheta9 = theta9.copy()
        ttheta9 += dx[6:15]

        # keep focal lengths sane in log space
        ttheta9[0] = np.clip(ttheta9[0], np.log(1.0), np.log(1e6))
        ttheta9[1] = np.clip(ttheta9[1], np.log(1.0), np.log(1e6))

        _LtL2, _Lty2, meas_new_y2, front2 = _project_accum_joint_irls_dist_numba(
            object_pts64, meas_pix,
            tqw, tqx, tqy, tqz, ttx, tty, ttz,
            float(ttheta9[0]), float(ttheta9[1]), float(ttheta9[2]), float(ttheta9[3]),
            float(ttheta9[4]), float(ttheta9[5]), float(ttheta9[6]), float(ttheta9[7]), float(ttheta9[8]),
            int(kind_int), float(robust_param),
            inv_sigma_2N
        )

        if front2 < 0.9:
            lam *= 3.0
            continue

        _, _, prior_new_y2 = _prior_cost_and_rhs(ttheta9, prior_mean9_use, prior_info9_use)
        total_new_y2 = float(meas_new_y2 + prior_new_y2)

        rho = float((total_old_y2 - total_new_y2) / pred)

        if rho > 0.25 and total_new_y2 < total_old_y2:
            qw, qx, qy, qz = tqw, tqx, tqy, tqz
            tx, ty, tz = ttx, tty, ttz
            theta9 = ttheta9
            lam *= 0.3
        else:
            lam *= 3.0

        if np.linalg.norm(dx) < 1e-7:
            break
        if abs(total_old_y2 - total_new_y2) / max(1e-12, total_old_y2) < 1e-6:
            break

    # ----------------- Finalize -----------------
    est_q = q(qw, np.array([qx, qy, qz], dtype=np.float64))
    est_q.force_s_pos()
    est_t = np.array([tx, ty, tz], dtype=np.float64)

    fx, fy, cx, cy, k1, k2, k3, p1, p2 = _intr_dist_from_state(theta9)
    cal_out = _clone_cal_with_intrinsics_and_distortion(cal, fx, fy, cx, cy, k1, k2, k3, p1, p2)

    if not return_stats:
        return est_q, est_t, cal_out

    LtL_meas, _, meas_y2, _ = _project_accum_joint_irls_dist_numba(
        object_pts64, meas_pix,
        float(est_q.s), float(est_q.vec[0]), float(est_q.vec[1]), float(est_q.vec[2]),
        float(est_t[0]), float(est_t[1]), float(est_t[2]),
        float(theta9[0]), float(theta9[1]), float(theta9[2]), float(theta9[3]),
        float(theta9[4]), float(theta9[5]), float(theta9[6]), float(theta9[7]), float(theta9[8]),
        int(kind_int), float(robust_param),
        inv_sigma_2N
    )

    H_prior9, _, prior_y2 = _prior_cost_and_rhs(theta9, prior_mean9_use, prior_info9_use)

    Htot = LtL_meas.copy()
    Htot[6:15, 6:15] += H_prior9

    dof = 2 * N - 15
    if dof < 1:
        dof = 1

    s2 = float(meas_y2) / float(dof)

    I15 = np.eye(15, dtype=np.float64)
    cov15 = s2 * np.linalg.solve(Htot, I15)
    cov9 = cov15[6:15, 6:15].copy()

    return est_q, est_t, cal_out, JointPnPStats(
        N=int(N),
        dof=int(dof),
        sse_meas_w=float(meas_y2),
        sse_prior=float(prior_y2),
        sse_total=float(meas_y2 + prior_y2),
        s2=float(s2),
        cov15=cov15,
        cov9=cov9,
        cal_state=theta9.copy(),
        cov10=None,
        cov4=cov9[:4, :4].copy(),
        intr_state=theta9[:4].copy(),
    )

############################################ SUPER OPT PATH WITH ONLINE CAL END #######################################


def enforce_chirality(q_est, t_est, object_pts):
    # Camera-frame points
    XYZ = q_est * object_pts + t_est
    Z = XYZ[:, 2]

    front_fraction = np.mean(Z > 0.0)

    # If most points are behind the camera, flip
    if front_fraction < 0.5:
        t_est = -t_est
        return True, q_est, t_est,
    return False, q_est, t_est


def _score_pose(q_est, t_est, object_pts, img_pts, cal, sigma_2N=None):
    """Robust score: median reprojection error in pixels (lower is better)."""
    proj = h(q_est, t_est, object_pts, cal).reshape(-1, 2)
    err = img_pts - proj
    e = np.sqrt(np.sum(err ** 2, axis=1))

    # Optional: whiten before scoring if you have sigmas
    if sigma_2N is not None:
        s = np.asarray(sigma_2N, dtype=float).ravel()
        if s.size == 2 * img_pts.shape[0]:
            sx = s[0::2]
            sy = s[1::2]
            # avoid divide-by-zero
            e = np.sqrt((err[:, 0] / np.maximum(sx, 1e-6)) ** 2 + (err[:, 1] / np.maximum(sy, 1e-6)) ** 2)

    return np.median(e)

def robust_seed_ransac_dlt(
    object_pts, img_pts, cal,
    sigma_2N=None,
    iters=64,
    subset=8,
    refine_iters=5,
    robust_kind=robust_cost.huber,
    robust_param=2.0,
    rng_seed=0
):
    """
    Multi-hypothesis initializer:
      sample subset -> DLT -> chirality -> short refine -> robust score
    Returns (best_q, best_t) or (None, None) if it fails.
    """
    N = img_pts.shape[0]
    if N < 6:
        return None, None

    rng = np.random.default_rng(rng_seed)

    best = None
    best_score = np.inf

    # Precompute per-point trust weights if you want (optional)
    # Example: if sigma_2N provided, weight points by 1/sigma (roughly)
    trust_w = None
    if sigma_2N is not None:
        s = np.asarray(sigma_2N, dtype=float).ravel()
        if s.size == 2*N:
            sx = s[0::2]; sy = s[1::2]
            sp = np.sqrt(np.maximum(sx, 1e-6) * np.maximum(sy, 1e-6))
            trust_w = 1.0 / np.maximum(sp, 1e-6)  # (N,)

    for _ in range(iters):
        idx = rng.choice(N, size=min(subset, N), replace=False)
        obj_s = object_pts[idx]
        img_s = img_pts[idx]

        # Subset weights (optional)
        tw_s = trust_w[idx] if trust_w is not None else None

        try:
            q0, t0 = DLT(obj_s, img_s, cal, trust_weighting=tw_s)
        except Exception:
            continue

        _, q0, t0 = enforce_chirality(q0, t0, object_pts)

        # quick local refine (few iterations) to stabilize basin selection
        try:
            q1, t1 = opt(
                img_pts, object_pts, cal,
                return_stats=False,
                seed_q=q0, seed_t=t0,
                sigma_2N=sigma_2N,
                robust_kind=robust_kind,
                robust_param=robust_param
            )
        except Exception:
            continue

        score = _score_pose(q1, t1, object_pts, img_pts, cal, sigma_2N=sigma_2N)
        if np.isfinite(score) and score < best_score:
            best_score = score
            best = (q1, t1)

    if best is None:
        return None, None
    return best

def DLT(object_pts: NDArray,
        img_pts: NDArray,
        cal: Calibration,
        trust_weighting: np.ndarray = None):
    """
    DLT initializer using *normalized* image coordinates (x~, y~),
    with correct handling of trust_weighting.

    - img_pts: (N,2) pixels (u,v)
    - object_pts: (N,3)
    - trust_weighting:
        * None
        * length N  (one weight per point)  -> expanded to 2N
        * length 2N (one weight per residual row) used directly
      (weights are assumed to be sqrt-weights, i.e., multiply rows by w)
    """

    object_pts = np.asarray(object_pts, dtype=np.float64)
    img_pts = np.asarray(img_pts, dtype=np.float64)

    num_points = img_pts.shape[0]
    if num_points < 6:
        raise ValueError(f"DLT needs >= 6 points, got {num_points}")

    # --- normalized coordinates ---
    xtil = (img_pts[:, 0] - cal.cx) / cal.fx
    ytil = (img_pts[:, 1] - cal.cy) / cal.fy

    # --- build A in normalized space ---
    # Same structure as your original, but x,y are replaced with xtil,ytil.
    A = np.zeros((2 * num_points, 12), dtype=np.float64)

    for i in range(num_points):
        X, Y, Z = object_pts[i]
        x = xtil[i]
        y = ytil[i]

        A[2 * i] = [-X, -Y, -Z, -1, 0, 0, 0, 0, x * X, x * Y, x * Z, x]
        A[2 * i + 1] = [0, 0, 0, 0, -X, -Y, -Z, -1, y * X, y * Y, y * Z, y]

    # --- apply trust weighting correctly ---
    if trust_weighting is not None:
        w = np.asarray(trust_weighting, dtype=np.float64).ravel()

        # Allow N weights (per point) or 2N weights (per row)
        if w.size == num_points:
            w = np.repeat(w, 2)
        elif w.size != 2 * num_points:
            raise ValueError(
                f"trust_weighting must have length N={num_points} or 2N={2 * num_points}, got {w.size}"
            )

        # Row-scale A by w (equivalent to diag(w) @ A, but faster/safer)
        A = (w[:, None] * A)

    # --- solve Ap=0 via SVD ---
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    p = Vt[-1, :]
    P = p.reshape((3, 4))

    # In normalized form, K = I, so M == P.
    # We still extract R_init, t_init for completeness.
    R_init, t_init = P[:, :3], P[:, 3]

    # --- enforce orthogonality on R ---
    U, _, Vt_r = np.linalg.svd(R_init)
    R = U @ Vt_r
    if np.linalg.det(R) < 0:
        # keep proper rotation
        U[:, -1] *= -1.0
        R = U @ Vt_r
        t_init *= -1.0  # keep projective sign consistent

    # --- translation solve (your existing method) ---
    t = _solve_t_given_R(object_pts, xtil, ytil, R)

    # --- return quaternion + translation ---
    q_init = mat2quat(R)
    return q_init, t


def _solve_t_given_R(Xw, x_tilde, y_tilde, R, w=None):
    """Solve translation t linearly given rotation R and image ratios (y/x, z/x).

    Projection used by h():
        u = fx * (  x / z ) + cx  ->  x/z = (u - cx)/fx = x_tilde
        v = fy * (  y / z ) + cy  ->  y/z = (v - cy)/fy = y_tilde

    For X_cam = R Xw + t with x = a1 + tx, y = a2 + ty, z = a3 + tz:
        x_tilde * (a3 + tz) = a1 + tx  ->  -tx     + (x_tilde)*tz =  a1 - x_tilde*a3
        y_tilde * (a3 + tz) = a2 + ty  ->      -ty + (y_tilde)*tz =  a2 - y_tilde*a3
    """
    R = np.asarray(R, float)
    Xw = np.asarray(Xw, float)
    x_tilde = np.asarray(x_tilde, float).reshape(-1)
    y_tilde = np.asarray(y_tilde, float).reshape(-1)

    N = Xw.shape[0]
    a1, a2, a3 = (R @ Xw.T)

    A = np.zeros((2 * N, 3), dtype=float)
    b = np.zeros(2 * N, dtype=float)

    # x/z equation rows
    A[0::2, 0] = -1.0
    A[0::2, 2] = x_tilde
    b[0::2] = a1 - x_tilde * a3

    # y/z equation rows
    A[1::2, 1] = -1.0
    A[1::2, 2] = y_tilde
    b[1::2] = a2 - y_tilde * a3

    if w is not None:
        ww = np.repeat(np.asarray(w, float).reshape(-1), 2)
        A = ww[:, None] * A
        b = ww * b

    t, *_ = np.linalg.lstsq(A, b, rcond=None)
    return t

def _sigma_to_point_sqrtw(sigma_2N, N: int, sigma_floor_px: float) -> np.ndarray | None:
    """
    Convert (2N,) sigma (px) into per-point sqrt-weights (N,).
    Weight model: w_i = 1 / max( sqrt(sx_i * sy_i), floor )
    Returns None if sigma_2N is None or malformed.
    """
    if sigma_2N is None:
        return None
    s = np.asarray(sigma_2N, dtype=float).reshape(-1)
    if s.size != 2 * N:
        return None
    sx = np.maximum(s[0::2], 1e-9)
    sy = np.maximum(s[1::2], 1e-9)
    sp = np.sqrt(sx * sy)
    floor = float(sigma_floor_px) if (sigma_floor_px is not None and sigma_floor_px > 0) else 1.0
    sp = np.maximum(sp, floor)
    return 1.0 / sp


def _front_fraction(q_est: q, t_est: np.ndarray, object_pts: np.ndarray, z_eps: float = 1e-3) -> float:
    """Fraction of points with z > z_eps in camera frame."""
    XYZ = q_est * object_pts + t_est
    return float(np.mean(XYZ[:, 2] > float(z_eps)))


def _median_reproj_err_px(
    q_est: q,
    t_est: np.ndarray,
    object_pts: np.ndarray,
    img_pts: np.ndarray,
    cal: Calibration,
    idx: np.ndarray | None = None
) -> float:
    """Median L2 reprojection error in pixels (cheap; no Jacobians)."""
    if idx is not None:
        obj = object_pts[idx]
        img = img_pts[idx]
    else:
        obj = object_pts
        img = img_pts
    proj = h(q_est, t_est, obj, cal).reshape(-1, 2)
    e = img - proj
    r = np.sqrt(np.sum(e * e, axis=1))
    return float(np.median(r))


def _choose_subset_indices_prosac(
    rng: np.random.Generator,
    order: np.ndarray,
    it: int,
    iters: int,
    subset_size: int,
    min_pool: int,
    N: int
) -> np.ndarray:
    """
    Simple PROSAC growth:
      - Start sampling from top min_pool most-certain points
      - Grow pool size towards N across iterations
    """
    if N <= subset_size:
        return order.copy()

    mp = max(int(min_pool), subset_size)
    mp = min(mp, N)

    if iters <= 1:
        pool = mp
    else:
        pool = int(mp + (N - mp) * (it / float(iters - 1)))
        pool = min(max(pool, mp), N)

    pool_idx = order[:pool]
    return rng.choice(pool_idx, size=subset_size, replace=False)


def _robust_seed_fast(
    object_pts: np.ndarray,
    img_pts: np.ndarray,
    cal: Calibration,
    sigma_2N=None,
    cfg: SeedConfig | None = None,
    robust_kind_for_refine: robust_cost = robust_cost.huber,
    robust_param_for_refine: float = 2.0,
):
    """
    Returns (seed_q, seed_t, info_dict).

    Key performance rule:
      - NEVER calls full opt() inside the hypothesis loop
      - Optional tiny refine only on top-K seeds (K small)
    """
    if cfg is None:
        cfg = SeedConfig()

    N = int(img_pts.shape[0])
    info = {"path": "dlt", "accepted_fast": False, "ransac_used": False}

    # ----------------------------
    # Fast path: weighted DLT once
    # ----------------------------
    if cfg.try_weighted_dlt_first:
        wN = _sigma_to_point_sqrtw(sigma_2N, N, cfg.dlt_sigma_floor_px)

        try:
            q0, t0 = DLT(object_pts, img_pts, cal, trust_weighting=wN)
        except Exception:
            q0, t0 = None, None

        if q0 is not None:
            _, q0, t0 = enforce_chirality(q0, t0, object_pts)
            ff = _front_fraction(q0, t0, object_pts, z_eps=cfg.z_eps)

            if cfg.score_quick_M and cfg.score_quick_M > 0:
                quick_idx = np.arange(min(int(cfg.score_quick_M), N))
                med = _median_reproj_err_px(q0, t0, object_pts, img_pts, cal, idx=quick_idx)
            else:
                med = _median_reproj_err_px(q0, t0, object_pts, img_pts, cal, idx=None)

            if (ff >= cfg.accept_front_frac) and (med <= cfg.accept_median_err_px):
                info["path"] = "weighted_dlt_fast_accept"
                info["accepted_fast"] = True
                return q0, t0, info

    # ----------------------------
    # Fallback: multi-hypothesis DLT + cheap scoring
    # ----------------------------
    if (not cfg.ransac_enabled) or (cfg.ransac_iters <= 0):
        info["path"] = "plain_dlt_fallback"
        q0, t0 = DLT(object_pts, img_pts, cal)
        _, q0, t0 = enforce_chirality(q0, t0, object_pts)
        return q0, t0, info

    info["ransac_used"] = True
    rng = np.random.default_rng(int(cfg.rng_seed))

    # certainty ordering for PROSAC (lowest sigma => highest weight => earlier)
    if cfg.prosac_enabled:
        wN = _sigma_to_point_sqrtw(sigma_2N, N, cfg.dlt_sigma_floor_px)
        if wN is None:
            order = np.arange(N, dtype=int)
        else:
            order = np.argsort(-wN)  # descending weight => most certain first
    else:
        order = np.arange(N, dtype=int)

    # quick scoring subset indices (fixed)
    if cfg.score_quick_M and cfg.score_quick_M > 0:
        M = min(int(cfg.score_quick_M), N)
        quick_idx = order[:M].copy()
    else:
        quick_idx = None

    best = None
    best_med = np.inf

    top = []  # (med, q, t)
    K = max(int(cfg.refine_top_k), 0)

    for it in range(int(cfg.ransac_iters)):
        if cfg.prosac_enabled:
            idx = _choose_subset_indices_prosac(
                rng, order, it, int(cfg.ransac_iters),
                subset_size=int(cfg.subset_size),
                min_pool=int(cfg.prosac_min_pool),
                N=N
            )
        else:
            idx = rng.choice(N, size=min(int(cfg.subset_size), N), replace=False)

        # optional weights for subset DLT
        w_sub = None
        if cfg.try_weighted_dlt_first:
            wN = _sigma_to_point_sqrtw(sigma_2N, N, cfg.dlt_sigma_floor_px)
            if wN is not None:
                w_sub = wN[idx]

        try:
            qh, th = DLT(object_pts[idx], img_pts[idx], cal, trust_weighting=w_sub)
        except Exception:
            continue

        _, qh, th = enforce_chirality(qh, th, object_pts)

        ff = _front_fraction(qh, th, object_pts, z_eps=cfg.z_eps)
        if ff < cfg.accept_front_frac:
            continue

        med = _median_reproj_err_px(qh, th, object_pts, img_pts, cal, idx=quick_idx)

        if med < best_med:
            best_med = med
            best = (qh, th)

        if K > 0:
            top.append((med, qh, th))

        if med <= float(cfg.early_exit_median_err_px):
            break

    if best is None:
        info["path"] = "plain_dlt_after_ransac_fail"
        q0, t0 = DLT(object_pts, img_pts, cal)
        _, q0, t0 = enforce_chirality(q0, t0, object_pts)
        return q0, t0, info

    # Optional tiny refine on top-K (still cheap)
    if K > 0 and len(top) > 0 and cfg.refine_max_iters > 0:
        top.sort(key=lambda x: x[0])
        top = top[:K]

        best_ref = None
        best_ref_med = np.inf

        for (_med, qh, th) in top:
            try:
                qr, tr = opt(
                    img_pts, object_pts, cal,
                    return_stats=False,
                    seed_q=qh, seed_t=th,
                    sigma_2N=sigma_2N,
                    robust_kind=robust_kind_for_refine,
                    robust_param=robust_param_for_refine,
                    sigma_floor_px=1.0,
                    max_iters=int(cfg.refine_max_iters),
                )
            except Exception:
                continue

            med_full = _median_reproj_err_px(qr, tr, object_pts, img_pts, cal, idx=None)
            if med_full < best_ref_med:
                best_ref_med = med_full
                best_ref = (qr, tr)

        if best_ref is not None:
            info["path"] = "ransac_topk_refined"
            return best_ref[0], best_ref[1], info

    info["path"] = "ransac_best_unrefined"
    return best[0], best[1], info


def solveQnP(
    object_pts: NDArray,
    img_pts: NDArray,
    cal: Calibration,
    return_stats: bool = False,
    sigma_2N=None,
    user_seed_q=None,
    user_seed_t=None,
    robust_kind: robust_cost = robust_cost.huber,
    robust_param: float = 2.0,
    seed_cfg: SeedConfig | None = None,
    use_solvePnP_as_seed: bool = False,
    online_calibration: bool = False,
):
    """
    QnP solver:
      1) seed selection (user seed OR robust initializer)
      2) enforce chirality
      3) nonlinear refinement
      4) optional online calibration of intrinsics + Brown-Conrady distortion
    """

    # ----------------------------
    # 1) Choose a good initial seed
    # ----------------------------
    if (user_seed_q is not None) and (user_seed_t is not None):
        seed_q = user_seed_q.copy()
        seed_t = np.asarray(user_seed_t, dtype=np.float64).copy()

    elif use_solvePnP_as_seed:
        import cv2

        ret, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=object_pts,
            imagePoints=img_pts,
            cameraMatrix=cal.getCameraMatrix(),
            distCoeffs=_opencv_dist_coeffs_from_cal(cal),
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        seed_q = q().from_rodrigues(rvec)
        seed_t = np.squeeze(tvec)

    else:
        if seed_cfg is None:
            seed_cfg = SeedConfig()

        if seed_cfg.enabled:
            seed_q, seed_t, _seed_info = _robust_seed_fast(
                object_pts=np.asarray(object_pts, dtype=float),
                img_pts=np.asarray(img_pts, dtype=float),
                cal=cal,
                sigma_2N=sigma_2N,
                cfg=seed_cfg,
                robust_kind_for_refine=robust_kind,
                robust_param_for_refine=robust_param,
            )
        else:
            seed_q, seed_t = DLT(object_pts, img_pts, cal)

    _, seed_q, seed_t = enforce_chirality(seed_q, seed_t, object_pts)

    # ----------------------------
    # 2) Refinement
    # ----------------------------
    if not online_calibration:
        return opt(
            img_pts,
            object_pts,
            cal,
            return_stats=return_stats,
            seed_q=seed_q,
            seed_t=seed_t,
            sigma_2N=sigma_2N,
            robust_kind=robust_kind,
            robust_param=robust_param,
            sigma_floor_px=1.0,
            max_iters=20,
        )

    # Build solver-space prior for [log_fx, log_fy, cx, cy, k1, k2, k3, p1, p2]
    prior_mean9, prior_cov9 = _build_prior_mean_cov9_from_cal(cal)

    est_q, est_t, cal_est, stats = opt_pose_and_intrinsics(
        img_pts,
        object_pts,
        cal,
        return_stats=True,
        seed_q=seed_q,
        seed_t=seed_t,
        sigma_2N=sigma_2N,
        prior_mean9=prior_mean9,
        prior_cov9=prior_cov9,
        robust_kind=robust_kind,
        robust_param=robust_param,
        sigma_floor_px=1.0,
        max_iters=20,
    )

    # IMPORTANT:
    # opt_pose_and_intrinsics already used the prior internally, so cal_est/stats
    # are already posterior-ish results. Do NOT apply a second KF update here.

    # Copy estimated calibration back into the live object.
    cal.fx = float(cal_est.fx)
    cal.fy = float(cal_est.fy)
    cal.cx = float(cal_est.cx)
    cal.cy = float(cal_est.cy)
    cal.k1 = float(getattr(cal_est, "k1", 0.0))
    cal.k2 = float(getattr(cal_est, "k2", 0.0))
    cal.k3 = float(getattr(cal_est, "k3", 0.0))
    cal.p1 = float(getattr(cal_est, "p1", 0.0))
    cal.p2 = float(getattr(cal_est, "p2", 0.0))

    # Convert posterior covariance from solver state
    #   [log_fx, log_fy, cx, cy, k1, k2, k3, p1, p2]
    # back to raw state
    #   [fx, fy, cx, cy, k1, k2, k3, p1, p2]
    J9 = np.eye(9, dtype=np.float64)
    J9[0, 0] = float(cal.fx)  # d fx / d log_fx = fx
    J9[1, 1] = float(cal.fy)  # d fy / d log_fy = fy

    raw_cov9 = J9 @ stats.cov9 @ J9.T

    # Keep old field for compatibility with existing code that expects 4x4 intrinsics covariance
    cal.cov = raw_cov9[:4, :4].copy()

    # New fields
    cal.cov_intr_dist = raw_cov9.copy()
    cal.dist_cov = raw_cov9[4:, 4:].copy()

    from support.io.my_logging import LOG
    LOG.info(f'Old Cal Matrix Prior-ish: \n{cal.getCameraMatrix()}')
    LOG.info(f'Est Cal Matrix: \n{cal_est.getCameraMatrix()}\n')
    LOG.info(
        f'Est Distortion [k1, k2, k3, p1, p2]: '
        f'[{cal.k1:.6g}, {cal.k2:.6g}, {cal.k3:.6g}, {cal.p1:.6g}, {cal.p2:.6g}]'
    )
    LOG.info(f'Posterior P raw [fx, fy, cx, cy, k1, k2, k3, p1, p2]: \n{raw_cov9}\n\n')

    if return_stats:
        return est_q, est_t, stats
    return est_q, est_t



if __name__ == '__main__':
    pass
