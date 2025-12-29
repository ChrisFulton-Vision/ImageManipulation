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
from support.core.quaternions import Quaternion as q, mat2quat
from support.io.calibration import Calibration
from support.include_numba import _njit as njit, prange
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

# --- Small helpers --------------------------------------------------------------
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
    ww = qw*qw; xx = qx*qx; yy = qy*qy; zz = qz*qz
    wx = qw*qx; wy = qw*qy; wz = qw*qz
    xy = qx*qy; xz = qx*qz; yz = qy*qz

    R = np.empty((3, 3), dtype=np.float64)
    R[0, 0] = ww + xx - yy - zz
    R[0, 1] = 2.0*(xy - wz)
    R[0, 2] = 2.0*(xz + wy)

    R[1, 0] = 2.0*(xy + wz)
    R[1, 1] = ww - xx + yy - zz
    R[1, 2] = 2.0*(yz - wx)

    R[2, 0] = 2.0*(xz - wy)
    R[2, 1] = 2.0*(yz + wx)
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

        rx = R[0, 0]*X + R[0, 1]*Y + R[0, 2]*Z
        ry = R[1, 0]*X + R[1, 1]*Y + R[1, 2]*Z
        rz = R[2, 0]*X + R[2, 1]*Y + R[2, 2]*Z

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
        proj_2N_out[r2]     = u
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
    y2  = 0.0

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

def h(est_q: q, est_t: np.array, feature_points, cal: Calibration):
    """Project all `FEATURE_OFFSETS` into pixel coordinates given pose (q, t).

    The pose maps model points into the camera frame as:  X_cam = q.T * X + t
    Then the custom pinhole projection computes pixels [u, v]:
        u = FX * (x / z) + CX
        v = FY * (y / z) + CY

    Returns a flattened length-2N vector [u0, v0, u1, v1, ...].
    """

    XYZ_proj = est_q * feature_points + est_t
    X, Y, Z = XYZ_proj[:, 0], XYZ_proj[:, 1], XYZ_proj[:, 2]

    us_vs_s_proj = np.zeros((XYZ_proj.shape[0], 2), dtype=float)
    us_vs_s_proj[:, 0] = cal.fx * (X / Z) + cal.cx
    us_vs_s_proj[:, 1] = cal.fy * (Y / Z) + cal.cy

    return us_vs_s_proj.flatten()


def _skew(v: np.ndarray) -> np.ndarray:
    """Return 3x3 skew-symmetric matrix [v]_x such that [v]_x w = v × w."""
    vx, vy, vz = v
    return np.array([
        [0.0, -vz,  vy],
        [vz,  0.0, -vx],
        [-vy, vx,  0.0]
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

def print_rayPts(ray_proj: np.array):
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
    sigma_floor_px: float = 1.0):
    """
    Refine pose to minimize ||meas_pix - h(q,t)|| using weighted GN.
    State: [δr, δt] (6 DOF), minimal tangent update.
    """

    # ----------------- Initialization -----------------
    if seed_q is None or seed_t is None:
        est_q, est_t = DLT(object_pts, img_pts, cal)
    else:
        est_q = seed_q.copy()
        est_t = seed_t.copy()

    meas_pix = img_pts.reshape(-1).astype(np.float64)
    N = img_pts.shape[0]

    # Work buffers (allocated once)
    proj = np.empty(2 * N, dtype=np.float64)
    y = np.empty(2 * N, dtype=np.float64)
    RX = np.empty((N, 3), dtype=np.float64)
    xyz = np.empty((N, 3), dtype=np.float64)
    L = np.empty((2 * N, 6), dtype=np.float64)

    object_pts64 = np.ascontiguousarray(object_pts, dtype=np.float64)

    # ----------- Sigma whitening (once) -----------
    inv_sigma_2N = _inv_sigma_2N_from_sigma(sigma_2N, N)
    if inv_sigma_2N is not None:
        # Prevent absurdly tiny sigmas from dominating the solve.
        # We cap inv_sigma <= 1/sigma_floor.
        sf = float(sigma_floor_px)
        if (not np.isfinite(sf)) or (sf <= 0.0):
            sf = 1.0
        inv_sigma_2N = np.minimum(inv_sigma_2N, 1.0 / sf)

    sqrtw = np.empty(2 * N, dtype=np.float64)
    rw = np.empty(2 * N, dtype=np.float64)

    # Robust enum → int
    kind_int = 0
    if robust_kind == robust_cost.huber:
        kind_int = 1
    elif robust_kind == robust_cost.cauchy:
        kind_int = 2
    elif robust_kind == robust_cost.tukey:
        kind_int = 3

    lam = 1e-1
    keep_going = True
    iter_num = 0

    # ================= GN LOOP =================
    while keep_going:
        iter_num += 1

        # ---- Projection + Jacobian ----
        qw, qx, qy, qz = est_q.s, *est_q.vec
        tx, ty, tz = est_t

        _project_and_jacobian_numba(
            object_pts64,
            float(qw), float(qx), float(qy), float(qz),
            float(tx), float(ty), float(tz),
            float(cal.fx), float(cal.fy), float(cal.cx), float(cal.cy),
            proj, RX, xyz, L
        )

        y[:] = meas_pix - proj

        # ---- Build sqrtw = R^{-1/2} * robust ----
        if inv_sigma_2N is None:
            sqrtw[:] = 1.0
        else:
            sqrtw[:] = inv_sigma_2N

        if kind_int != 0:
            _robust_sqrt_weights_inplace_numba(
                y, N, kind_int, float(robust_param), inv_sigma_2N, rw
            )
            sqrtw *= rw

        LtL, Lty, y2 = _accum_LtL_Lty_numba(L, y, sqrtw)
        old_y_mag = np.sqrt(y2)

        # ---- Damped solve ----
        delta_x = np.linalg.solve(LtL + lam * np.eye(6), Lty)

        # ---- Line search ----
        scale = 1.0
        accepted = False

        while not accepted:
            delta_r = scale * delta_x[:3]
            delta_t = scale * delta_x[3:]

            trial_q = q.from_rodrigues(delta_r) * est_q
            trial_t = est_t + delta_t

            XYZ = trial_q * object_pts + trial_t
            if np.mean(XYZ[:, 2] > 1e-3) < 0.9:
                lam *= 3.0
                scale *= 0.5
                if scale < 1e-4:
                    accepted = True
                continue

            qw, qx, qy, qz = trial_q.s, *trial_q.vec
            tx, ty, tz = trial_t

            _project_and_jacobian_numba(
                object_pts64,
                float(qw), float(qx), float(qy), float(qz),
                float(tx), float(ty), float(tz),
                float(cal.fx), float(cal.fy), float(cal.cx), float(cal.cy),
                proj, RX, xyz, L
            )

            y[:] = meas_pix - proj

            if inv_sigma_2N is None:
                sqrtw[:] = 1.0
            else:
                sqrtw[:] = inv_sigma_2N

            if kind_int != 0:
                _robust_sqrt_weights_inplace_numba(
                    y, N, kind_int, float(robust_param), inv_sigma_2N, rw
                )
                sqrtw *= rw

            new_y_mag = np.linalg.norm(sqrtw * y)

            # GN ratio test
            y_pred = y - L.dot(scale * delta_x)
            y_pred_mag = np.linalg.norm(sqrtw * y_pred)

            denom = old_y_mag - y_pred_mag
            ratio = 0.0 if denom == 0 else (old_y_mag - new_y_mag) / denom

            if 0.25 < ratio < 4.0:
                lam *= 0.3
                est_q = trial_q
                est_t = trial_t
                accepted = True
            else:
                lam *= 3.0
                scale *= 0.5
                if scale < 1e-4:
                    accepted = True

        if np.linalg.norm(scale * delta_x) < 1e-7 or iter_num > 20:
            keep_going = False

    est_q.force_s_pos()
    if not return_stats:
        return est_q, est_t

    LtL, _Lty, y2 = _accum_LtL_Lty_numba(L, y, sqrtw)

    dof = 2 * N - 6
    if dof < 1:
        dof = 1
    s2 = float(y2) / float(dof)

    # Cov = s2 * inv(LtL)
    I = np.eye(6, dtype=np.float64)
    cov6 = s2 * np.linalg.solve(LtL, I)

    return est_q, est_t, QnPStats(N=int(N),
                                  dof=int(dof),
                                  sse_w=float(y2),
                                  s2=float(s2),
                                  cov6=cov6)

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
    img_pts    = np.asarray(img_pts,    dtype=np.float64)

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

        A[2 * i]     = [-X, -Y, -Z, -1, 0, 0, 0, 0, x * X, x * Y, x * Z, x]
        A[2 * i + 1] = [0, 0, 0, 0, -X, -Y, -Z, -1, y * X, y * Y, y * Z, y]

    # --- apply trust weighting correctly ---
    if trust_weighting is not None:
        w = np.asarray(trust_weighting, dtype=np.float64).ravel()

        # Allow N weights (per point) or 2N weights (per row)
        if w.size == num_points:
            w = np.repeat(w, 2)
        elif w.size != 2 * num_points:
            raise ValueError(
                f"trust_weighting must have length N={num_points} or 2N={2*num_points}, got {w.size}"
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


def solveQnP(object_pts: np.array,
             img_pts: np.array,
             cal: Calibration,
             return_stats:bool = False,
             sigma_2N=None,
             user_seed_q=None,
             user_seed_t=None,
             robust_kind:robust_cost = robust_cost.huber,
             robust_param:float = 2.0):
    """
    Quaternion-based PnP solver.

    - If user_seed_q / user_seed_t are provided, they are used as the initial pose.
    - Otherwise, we initialize with a DLT pose, then refine with Gauss–Newton (opt).
    - Optional 'trust_weighting' (length 2N) down-weights residuals in pixel space
      during the nonlinear refinement (but not in DLT).
    """

    # ------------------------------------------------------------------
    # 1) Choose a good initial seed
    # ------------------------------------------------------------------
    if (user_seed_q is not None) and (user_seed_t is not None):
        # Use caller-provided seed (e.g., previous frame, or PnP pose)
        seed_q = user_seed_q.copy()
        seed_t = user_seed_t.copy()
    else:
        # Use DLT initializer to mirror PnP-style behavior
        seed_q, seed_t = DLT(object_pts, img_pts, cal)

    flipped, seed_q, seed_t = enforce_chirality(seed_q, seed_t, object_pts)
    # ------------------------------------------------------------------
    # 2) Nonlinear refinement around the seed (Gauss–Newton / LM-like)
    #    This is where trust_weighting gives us an advantage over PnP.
    # ------------------------------------------------------------------

    # If caller provided sigma, use a robust loss by default (outlier safety).
    if sigma_2N is not None and robust_kind == robust_cost.none:
        robust_kind = robust_cost.huber
        robust_param = 2.0  # 2-sigma in whitened units when sigma_2N is provided

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
        sigma_floor_px=1.0
    )

if __name__ == '__main__':
    pass
