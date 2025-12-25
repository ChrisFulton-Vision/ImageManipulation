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

from sys import maxsize
from SupportModules.quaternions import Quaternion as q
from SupportModules.quaternions import *
from SupportModules.Calibration import Calibration
from SupportModules.include_numba import njit, prange
from numpy.linalg import norm
from numpy.typing import NDArray
from enum import Enum
from itertools import cycle

class robust_cost(Enum):
    none = None,
    huber = 'huber',
    cauchy = 'cauchy',
    tukey = 'tukey'

    def next(self):
        iterator = cycle(self.__class__)
        for member in iterator:
            if member is self:
                return next(iterator)

# Pretty-printing controls for numpy (purely cosmetic; does not affect math)
np.set_printoptions(suppress=True, precision=4, threshold=maxsize)

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

def _robust_sqrt_weights_from_residual(y_2N, N,
                                       kind: robust_cost = robust_cost.huber,
                                       param=2.0,
                                       sigma_2N=None, eps=1e-12):
    """
    Return per-residual sqrt-weights (2N,) for IRLS.

    We robustify per *feature* using the 2D residual norm:
        r_i = sqrt(du^2 + dv^2)
    Optionally normalize by sigma (pixel std-dev) before applying robust loss.

    kind:
      - "huber": param = delta  (in sigma units if sigma provided, else pixels)
      - "cauchy": param = c
      - "tukey": param = c (Tukey biweight)
    """
    y = np.asarray(y_2N, float).reshape(-1)
    y2 = y.reshape(N, 2)

    if sigma_2N is not None:
        s = np.asarray(sigma_2N, float).reshape(-1)
        s = np.clip(s, eps, None)
        s2 = s.reshape(N, 2)
        # scalar sigma per feature (RMS of u/v sigmas)
        sigma_i = np.sqrt(0.5 * (s2[:, 0]**2 + s2[:, 1]**2))
        r = np.linalg.norm(y2, axis=1) / np.clip(sigma_i, eps, None)
    else:
        r = np.linalg.norm(y2, axis=1)

    r = np.clip(r, eps, None)

    c = float(param)

    if kind == robust_cost.huber:
        # w = 1                    if r <= c
        # w = c / r                if r >  c
        w = np.ones_like(r)
        mask = r > c
        w[mask] = c / r[mask]

    elif kind == robust_cost.cauchy:
        # rho = (c^2/2) log(1 + (r/c)^2)  ->  w = 1 / (1 + (r/c)^2)
        t = (r / c)
        w = 1.0 / (1.0 + t * t)

    elif kind == robust_cost.tukey:
        # Tukey biweight: w = (1 - (r/c)^2)^2 for r < c else 0
        t = r / c
        w = np.zeros_like(r)
        mask = t < 1.0
        w[mask] = (1.0 - t[mask]**2)**2

    else:
        raise ValueError(f"Unknown robust kind '{kind}'")

    # IRLS uses sqrt(w) as left-multipliers of residual/J
    sqrtw_2N = np.repeat(np.sqrt(np.clip(w, 0.0, None)), 2)
    return sqrtw_2N

def opt(img_pts: NDArray, object_pts: NDArray,
        cal: Calibration, seed_q: q = None, seed_t: NDArray = None,
        trust_weighting: NDArray = None,
        robust_kind: robust_cost = robust_cost.none,          # e.g. "huber", "cauchy", "tukey"
        robust_param=2.0,          # delta (Huber) or c (Cauchy/Tukey)
        sigma_2N=None):            # optional per-residual sigma (2N,) in pixels

    """Refine pose to minimize ||meas_pix - h(q, t)|| using a GN-like loop.

    Uses a *minimal* 6D state:
        x = [δr_x, δr_y, δr_z, δt_x, δt_y, δt_z]^T

    where δr is a small Rodrigues vector in the camera frame, applied via:
        q_new = Quaternion.from_rodrigues(δr) * q_old

    Translation is updated additively: t_new = t_old + δt.
    """

    # ----------------- Initialization -----------------
    if np.linalg.norm(seed_t) > 500.0 or seed_q.norm > 500.0:
        seed_q = seed_t = None

    if seed_q is None or seed_t is None:
        est_q, est_t = DLT(object_pts, img_pts, cal)
    else:
        est_q = seed_q.copy()
        est_t = seed_t.copy()

    meas_pix = img_pts.flatten()
    N = img_pts.shape[0]

    proj = np.empty(2 * N, dtype=np.float64)
    y = np.empty(2 * N, dtype=np.float64)
    RX = np.empty((N, 3), dtype=np.float64)
    xyz = np.empty((N, 3), dtype=np.float64)
    L = np.empty((2 * N, 6), dtype=np.float64)

    object_pts64 = np.ascontiguousarray(object_pts, dtype=np.float64)

    keep_going = True
    iter_num = 0
    while keep_going:
        iter_num += 1

        # Residual and Jacobian at current pose
        # Convert current quaternion object -> 4 scalars (cheap)
        qw = float(est_q.s)
        qx = float(est_q.vec[0])
        qy = float(est_q.vec[1])
        qz = float(est_q.vec[2])
        tx = float(est_t[0])
        ty = float(est_t[1])
        tz = float(est_t[2])

        _project_and_jacobian_numba(
            object_pts64,
            qw, qx, qy, qz, tx, ty, tz,
            float(cal.fx), float(cal.fy), float(cal.cx), float(cal.cy),
            proj, RX, xyz, L
        )

        # y := meas - proj (in-place)
        y[:] = meas_pix - proj

        # ---- Build combined sqrt-weights (2N,) ----
        sqrtw = np.ones(2 * N, dtype=float)

        # (A) trust weighting (accept N or 2N)
        if trust_weighting is not None:
            tw = _expand_to_2N_weights(trust_weighting, N)
            tw = np.clip(tw / np.mean(tw), 0.25, 4.0)

            # if your trust_weighting is intended as *cost* weights, use sqrt:
            sqrtw *= np.sqrt(tw)

            # if you intentionally tuned with your old behavior (w applied directly),
            # comment out the sqrt line above and instead do:
            # sqrtw *= tw

        # (B) robust weighting (depends on current residual)
        if robust_kind is not robust_kind.none:
            rw = _robust_sqrt_weights_from_residual(
                y_2N=y, N=N, kind=robust_kind, param=robust_param, sigma_2N=sigma_2N
            )
            sqrtw *= rw

        # ------------- Damped normal equations -------------
        LtL, Lty, y2 = _accum_LtL_Lty_numba(L, y, sqrtw)
        old_y_mag = np.sqrt(y2)

        lam = 1e-1

        LAMBDA = lam * np.eye(LtL.shape[0])

        LtL_damped = LtL + LAMBDA

        delta_x = np.linalg.solve(LtL_damped, Lty)     # (6,)

        # ------------- Backtracking line search -------------
        scale = 1.0
        scale_is_good = False
        while not scale_is_good:
            delta_r = scale * delta_x[0:3]
            delta_t = scale * delta_x[3:6]

            trial_q = q.from_rodrigues(delta_r) * est_q
            trial_t = est_t + delta_t

            # Reject poses with points too close/behind camera (prevents 1/Z blowups)
            XYZ_trial = trial_q * object_pts + trial_t
            if np.mean(XYZ_trial[:, 2]> 1e-3) < 0.9:
                scale *= 0.5
                lam *= 3.0
                if scale < 1e-4:
                    break
                continue

            # compute weighted residual norm without Q
            qw = float(trial_q.s)
            qx = float(trial_q.vec[0])
            qy = float(trial_q.vec[1])
            qz = float(trial_q.vec[2])
            tx = float(trial_t[0])
            ty = float(trial_t[1])
            tz = float(trial_t[2])

            # We only need proj here; L_out can be reused (still filled, harmless)
            _project_and_jacobian_numba(
                object_pts64,
                qw, qx, qy, qz, tx, ty, tz,
                float(cal.fx), float(cal.fy), float(cal.cx), float(cal.cy),
                proj, RX, xyz, L
            )

            y[:] = meas_pix - proj
            new_y_mag = float(np.linalg.norm(sqrtw * y))

            # Linear prediction of residual magnitude
            y_pred = y - L.dot(scale * delta_x)
            y_pred_mag = float(np.linalg.norm(sqrtw * y_pred))

            # If perfect agreement between nonlinear and linear prediction, stop
            if np.abs(old_y_mag - y_pred_mag) < 1e-5:
                scale_is_good = True
                keep_going = False
            else:
                # Accept step if the ratio is in a reasonable trust range
                denom = (old_y_mag - y_pred_mag)
                if denom == 0:
                    ratio = 0.0
                else:
                    ratio = (old_y_mag - new_y_mag) / denom

                if 0.25 < ratio < 4.0:
                    lam *= 0.3
                    scale_is_good = True
                    # Commit step
                    est_q = q.from_rodrigues(delta_r) * est_q
                    est_t = est_t + delta_t
                else:
                    # Backtrack
                    lam *= 3.0
                    scale /= 2.0
                    if scale < 1e-4:
                        # Don't get stuck forever
                        scale_is_good = True

        # ------------- Termination -------------
        if norm(scale * delta_x) < 1e-7 or iter_num > 20:
            keep_going = False

    # Enforce sign convention once at the end
    est_q.force_s_pos()
    return est_q, est_t


def enforce_chirality(q_est, t_est, object_pts, cal):
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
             trust_weighting=None,
             user_seed_q=None,
             user_seed_t=None):
    """
    Quaternion-based PnP solver.

    - If user_seed_q / user_seed_t are provided, they are used as the initial pose.
    - Otherwise, we initialize with a DLT pose, then refine with Gauss–Newton (opt).
    - Optional 'trust_weighting' (length 2N) down-weights residuals in pixel space
      during the nonlinear refinement (but not in DLT).
    """
    # trust_weighting = np.ones(img_pts.size)
    # trust_weighting[0] = trust_weighting[1] = 0.1

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

    flipped, seed_q, seed_t = enforce_chirality(seed_q, seed_t, object_pts, cal)
    # ------------------------------------------------------------------
    # 2) Nonlinear refinement around the seed (Gauss–Newton / LM-like)
    #    This is where trust_weighting gives us an advantage over PnP.
    # ------------------------------------------------------------------
    est_q, est_t = opt(
        img_pts,
        object_pts,
        cal,
        seed_q=seed_q,
        seed_t=seed_t,
        trust_weighting=trust_weighting,
        robust_kind=robust_cost.cauchy
    )

    # ------------------------------------------------------------------
    # 3) Enforce chirality (points in front of camera), then optionally
    #    re-optimize from the corrected pose.
    # ------------------------------------------------------------------
    # flipped, est_q, est_t = enforce_chirality(est_q, est_t, object_pts, cal)
    # if flipped:
    #     # Small re-optimization starting from the chirality-corrected pose
    #     est_q, est_t = opt(
    #         img_pts,
    #         object_pts,
    #         cal,
    #         seed_q=est_q,
    #         seed_t=est_t,
    #         trust_weighting=trust_weighting,
    #     )

    # ------------------------------------------------------------------
    # 4) Clean up quaternion sign convention (avoid random sign flips)
    # ------------------------------------------------------------------
    est_q.force_s_pos()

    return est_q, est_t



if __name__ == '__main__':
    pass
