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
import numpy as np
from SupportModules.quaternions import Quaternion as q
from SupportModules.quaternions import *
from SupportModules.Calibration import Calibration
from SupportModules.include_numba import njit, prange
from numpy.linalg import norm
from numpy.typing import NDArray
from copy import deepcopy

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


def opt(img_pts: NDArray, object_pts: NDArray,
        cal: Calibration, seed_q: q = None, seed_t: NDArray = None,
        trust_weighting: NDArray = None):
    """Refine pose to minimize ||meas_pix - h(q, t)|| using a GN-like loop.

    Uses a *minimal* 6D state:
        x = [δr_x, δr_y, δr_z, δt_x, δt_y, δt_z]^T

    where δr is a small Rodrigues vector in the camera frame, applied via:
        q_new = Quaternion.from_rodrigues(δr) * q_old

    Translation is updated additively: t_new = t_old + δt.
    """

    # ----------------- Initialization -----------------
    if seed_q is None or seed_t is None:
        est_q, est_t = DLT(object_pts, img_pts, cal, trust_weighting)
    else:
        est_q = deepcopy(seed_q)
        est_t = deepcopy(seed_t)

    meas_pix = img_pts.flatten()

    keep_going = True
    iter_num = 0
    while keep_going:
        iter_num += 1

        # Residual and Jacobian at current pose
        y = meas_pix - h(est_q, est_t, object_pts, cal)
        old_y_mag = norm(y)
        L = deriv(est_q, est_t, object_pts, cal)      # (2N, 6)

        if trust_weighting is not None:
            Q = np.diag(trust_weighting)
            y = Q.dot(y)
            L = Q.dot(L)
        else:
            Q = None

        # ------------- Damped normal equations -------------
        lam = 1e-1  # tune; 1e-4 to 1e-1 is a reasonable range
        LtL = L.T @ L                                  # (6, 6)
        Lty = L.T @ y                                  # (6,)

        LAMBDA = lam * np.eye(LtL.shape[0])

        LtL_damped = LtL + LAMBDA

        delta_x = np.linalg.solve(LtL, Lty)     # (6,)

        # ------------- Backtracking line search -------------
        scale = 1.0
        scale_is_good = False
        while not scale_is_good:
            delta_r = scale * delta_x[0:3]
            delta_t = scale * delta_x[3:6]

            trial_q = q.from_rodrigues(delta_r) * est_q
            trial_t = est_t + delta_t

            if Q is not None:
                new_y_mag = norm(Q.dot(
                    meas_pix - h(trial_q, trial_t, object_pts, cal)
                ))
            else:
                new_y_mag = norm(
                    meas_pix - h(trial_q, trial_t, object_pts, cal)
                )

            # Linear prediction of residual magnitude
            y_pred_mag = norm(y - L.dot(scale * delta_x))

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
                    scale_is_good = True
                    # Commit step
                    est_q = q.from_rodrigues(delta_r) * est_q
                    est_t = est_t + delta_t
                else:
                    # Backtrack
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
        q_est = q(-q_est.s, -q_est.vec)
        t_est = -t_est
        return True, q_est, t_est,
    return False, q_est, t_est

def DLT(object_pts: NDArray, img_pts: NDArray, cal: Calibration, trust_weighting: np.array = None):

    num_points = len(img_pts)

    A = np.zeros((2 * num_points, 12))

    for i in range(num_points):
        X, Y, Z = object_pts[i]
        x, y = img_pts[i]

        A[2 * i] = [-X, -Y, -Z, -1, 0, 0, 0, 0, x * X, x * Y, x * Z, x]
        A[2 * i + 1] = [0, 0, 0, 0, -X, -Y, -Z, -1, y * X, y * Y, y * Z, y]

    if trust_weighting is not None:
        Q = np.diag( trust_weighting)
        A = Q @ A

    # 2. Solve the linear system Ap = 0 using SVD
    _, _, Vt = np.linalg.svd(A)

    # The solution is the last column of V (or last row of Vt)
    p = Vt[-1, :]
    P = p.reshape((3, 4))

    # 3. Extract K, R, and t from the projection matrix P
    # P = K[R|t] => M = inv(K) * P
    M = cal.inv @ P
    R_init, t_init = M[:, :3], M[:, 3]

    # 4. Enforce orthogonality on R
    U, _, Vt_r = np.linalg.svd(R_init)
    R = U @ Vt_r
    if np.linalg.det(R) < 0:
        R *= -1.0


    xtil = (img_pts[:, 0] - cal.cx) / cal.fx
    ytil = (img_pts[:, 1] - cal.cy) / cal.fy
    t = _solve_t_given_R(object_pts, xtil, ytil, R)

    # 5. Get the final rvec
    # rvec, _ = cv2.Rodrigues(R)
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
        seed_q = deepcopy(user_seed_q)
        seed_t = deepcopy(user_seed_t)
    else:
        # Use DLT initializer to mirror PnP-style behavior
        seed_q, seed_t = DLT(object_pts, img_pts, cal, trust_weighting=trust_weighting)


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
