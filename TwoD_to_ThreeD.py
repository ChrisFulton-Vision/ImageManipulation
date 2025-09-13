"""
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
from quaternions import Quaternion as q
from quaternions import *
from Calibration import Calibration
import numpy as np
from numpy import square as sq
from numpy.linalg import norm
from numpy.typing import NDArray
from copy import deepcopy

# Pretty-printing controls for numpy (purely cosmetic; does not affect math)
np.set_printoptions(suppress=True, precision=4, threshold=maxsize)


def _ensure_shapes(Xw, uv):
    Xw = np.asarray(Xw, float)
    uv = np.asarray(uv, float)
    if Xw.ndim == 1:
        Xw = Xw.reshape(-1, 3)
    if uv.ndim == 1:
        uv = uv.reshape(-1, 2)
    return Xw, uv


# def robust_init_pose(Xw, uv, cal):
#     """
#     EPnP-only seeding.
#     - Uses cv2.SOLVEPNP_EPNP to get an initial rotation.
#     - Converts to your quaternion convention (R = q.T).
#     - Computes t with your closed-form _solve_t_given_R, so it's consistent with h().
#     - Optional tiny axis-permutation pick (P @ R) for your display convention.
#     """
#     Xw, uv = _ensure_shapes(Xw, uv)
#
#     obj = Xw.astype(np.float32)
#     img = uv.astype(np.float32)
#     K = np.array([[cal.fx, 0, cal.cx],
#                   [0, cal.fy, cal.cy],
#                   [0, 0, 1]], dtype=np.float32)
#
#     ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_SQPNP)
#     if not ok:
#         # Fallback (very rare): Wahba seed
#         q0, t0 = init_pose_wahba(Xw, uv, cal)
#         return q0, t0
#
#     Rcv, _ = cv2.Rodrigues(rvec)
#
#     # Minimal, robust pick between Rcv and your display-permuted P @ Rcv
#     P = np.array([[0., 0., 1.],
#                   [1., 0., 0.],
#                   [0., 1., 0.]], dtype=float)
#
#     q_seed = mat2quat( (P @ Rcv).T)
#     ytil = (uv[:, 0] - cal.cx) / cal.fx
#     ztil = (uv[:, 1] - cal.cy) / cal.fy
#     t_seed = _solve_t_given_R(Xw, ytil, ztil, P @ Rcv)
#     # print("GOOD:")
#     # print(Xw)
#     # print(ytil)
#     # print(ztil)
#     # print(P @ Rcv)
#     # print(t_seed)
#     # print()
#     return q_seed, t_seed


# --- Small helpers --------------------------------------------------------------

def _row_normed(A, eps=1e-12):
    """Row-normalize a 2D array.

    Each row is divided by its L2 norm; very small norms are clamped by *eps* to
    avoid division by ~0.
    """
    n = np.linalg.norm(A, axis=1, keepdims=True)
    return A / np.clip(n, eps, None)


def _solve_t_given_R(Xw, y_tilde, z_tilde, R, w=None):
    """Solve translation t linearly given rotation R and image ratios (y/x, z/x).

    Projection used by h():
        u = fx * ( +y / x ) + cx  ->  y/x = (u - cx)/fx = y_tilde
        v = fy * (  z / x ) + cy  ->  z/x = (v - cy)/fy = z_tilde

    For X_cam = R Xw + t with x = a1 + tx, y = a2 + ty, z = a3 + tz:
        y_tilde * (a1 + tx) = a2 + ty  ->  ( y_tilde)*tx  - ty      =  a2 - y_tilde*a1
        z_tilde * (a1 + tx) = a3 + tz  ->  ( z_tilde)*tx       - tz =  a3 - z_tilde*a1
    """
    R = np.asarray(R, float)
    Xw = np.asarray(Xw, float)
    y_tilde = np.asarray(y_tilde, float).reshape(-1)
    z_tilde = np.asarray(z_tilde, float).reshape(-1)

    r1, r2, r3 = R
    a1 = Xw @ r1
    a2 = Xw @ r2
    a3 = Xw @ r3
    N = Xw.shape[0]

    A = np.zeros((2 * N, 3), dtype=float)
    b = np.zeros(2 * N, dtype=float)

    # y/x equation rows
    A[0::2, 0] = y_tilde  # *tx
    A[0::2, 1] = -1.0  # -ty
    b[0::2] = a2 - y_tilde * a1

    # z/x equation rows
    A[1::2, 0] = z_tilde  # *tx
    A[1::2, 2] = -1.0  # -tz
    b[1::2] = a3 - z_tilde * a1

    if w is not None:
        ww = np.repeat(np.asarray(w, float).reshape(-1), 2)
        A = ww[:, None] * A
        b = ww * b

    t, *_ = np.linalg.lstsq(A, b, rcond=None)
    return t


# # --- Fast initializer: Wahba/Kabsch rotation + linear t ------------------------
#
# def init_pose_wahba(Xw, meas_pix, cal: Calibration):
#     """
#     Robust seed that matches your projection:
#         u = FX * ( +y / x ) + CX
#         v = FY * (  z / x ) + CY
#
#     Strategy:
#       1) Build unit bearing rays B from pixels (two variants: nominal y/x and flipped).
#       2) Object-centered Wahba/Kabsch using weighted directions of centered 3D points.
#       3) Solve t linearly given R to match y/x, z/x (your algebra).
#       4) Enforce cheirality by majority (x_cam > 0).
#       5) Evaluate BOTH branches against actual pixel residual via h(), pick lower.
#
#     Returns:
#       (q_init, t_init) where q_init.T == R used by h().
#     """
#     # --- inputs to numpy arrays / shapes ---------------------------------------
#     Xw = np.asarray(Xw, float).reshape(-1, 3)
#     uv = np.asarray(meas_pix, float).reshape(-1, 2)
#     N = Xw.shape[0]
#
#     # Pixel -> slopes consistent with h(): +y/x and +z/x
#     ytil_nom = (uv[:, 0] - cal.cx) / cal.fx
#     ztil = (uv[:, 1] - cal.cy) / cal.fy
#
#     # Center and build weighted directions of object points
#     Xc = Xw - Xw.mean(axis=0, keepdims=True)
#     Xdir = _row_normed(Xc)
#
#     # Gentle weight clamp to avoid hinging on a single far tag
#     w0 = np.linalg.norm(Xc, axis=1)
#     if np.all(np.isfinite(w0)) and np.any(w0 > 0):
#         p20, p80 = np.percentile(w0, [20.0, 80.0])
#         w0 = np.clip(w0, p20, p80)
#         w0 /= (w0.mean() + 1e-12)
#     else:
#         w0 = np.ones(N, dtype=float)
#
#     def solve_branch(ytil_current):
#         """Compute (R, t, q, residual) for one y/x sign branch."""
#         # Bearings B = unit([1, y/x, z/x]) to align with camera-frame axes used by h()
#         B = _row_normed(np.column_stack([np.ones(N), ytil_current, ztil]))
#
#         # Wahba/Kabsch on weighted directions: H = (B * w)^T @ Xdir
#         H = (B * w0[:, None]).T @ Xdir
#         U, S, Vt = np.linalg.svd(H, full_matrices=False)
#         R = U @ Vt
#         if np.linalg.det(R) < 0:  # enforce det +1 with minimal change
#             U[:, -1] *= -1
#             R = U @ Vt
#
#         # Linear t that matches the y/x and z/x equations (your helper)
#         t = _solve_t_given_R(Xw, ytil_current, ztil, R, w=w0)
#
#         # Majority cheirality: x_cam must be positive for most points
#         X_cam = Xw @ R.T + t
#         if np.count_nonzero(X_cam[:, 0] > 0.0) < 0.8 * N:
#             # Minimal reflection that flips x while preserving right-handedness
#             R = R @ np.diag([-1.0, -1.0, 1.0])
#             t[:2] *= -1.0
#
#         # Map to your quaternion convention (q.T == R used by h())
#         q_corr = q(quat=np.array([0.0259804137, -0.3295700285, 0.3014442269, 0.8943377396]))
#         qk = q_corr * mat2quat(R.T)
#
#         # Pixel residual under the actual projection model
#         pred = h(qk, t, Xw, cal)
#         res = float(np.linalg.norm(uv.flatten() - pred))
#         return res, qk, t
#
#     # Evaluate both branches: nominal y/x and flipped y/x
#     candidates = [
#         solve_branch(ytil_nom),
#         solve_branch(-ytil_nom),
#     ]
#
#     # Choose the candidate with the lower pixel-space residual
#     candidates.sort(key=lambda c: c[0])
#     _, q_best, t_best = candidates[0]
#     return q_best, t_best
#
#
# def _cheirality_frac_Rt(R, t, Xw, eps=1e-9):
#     Xc = (R @ Xw.T).T + t
#     return float(np.count_nonzero(Xc[:, 0] > eps)) / Xw.shape[0]
#
#
# def _post_refine_flip_biside(est_q, est_t, meas_pix, object_pts, cal,
#                              min_ch=0.90, margin_px=100.0):
#     """
#     Final sanity check that tries 180° flips on BOTH sides of R:
#       candidates:  R,
#                    F @ R, R @ F for F in {Rx, Ry, Rz(=xy)}
#     For each candidate:
#       - re-solve t with _solve_t_given_R
#       - compute residual under h
#       - require cheirality >= min_ch
#     Switching rule:
#       • If current pose is cheirality-INVALID and some candidate is VALID,
#         take the VALID one with lowest residual (no margin).
#       • Else (current VALID), only switch if residual improves by >= margin_px.
#     """
#     import numpy as np
#
#     uv = np.asarray(meas_pix, float).reshape(-1, 2)
#     Xw = np.asarray(object_pts, float).reshape(-1, 3)
#
#     # Current state
#     R0 = quat2mat(est_q.ndarray).T
#     res0 = float(np.linalg.norm(uv.flatten() - h(est_q, est_t, Xw, cal)))
#     ch0 = _cheirality_frac_Rt(R0, est_t, Xw)
#
#     # 180° proper rotations about camera axes
#     Fx = np.diag([1.0, -1.0, -1.0])  # about cam-x
#     Fy = np.diag([-1.0, 1.0, -1.0])  # about cam-y
#     Fz = np.diag([-1.0, -1.0, 1.0])  # about cam-z
#
#     ytil = (uv[:, 0] - cal.cx) / cal.fx
#     ztil = (uv[:, 1] - cal.cy) / cal.fy
#
#     def score_R(Rcand):
#         t = _solve_t_given_R(Xw, ytil, ztil, Rcand)
#         q_new = mat2quat(Rcand.T)  # returns your Quaternion
#         res = float(np.linalg.norm(uv.flatten() - h(q_new, t, Xw, cal)))
#         ch = _cheirality_frac_Rt(Rcand, t, Xw)
#         return res, ch, q_new, t
#
#     cands = []
#     # identity first (so cands[0] is "current")
#     cands.append(score_R(R0))
#
#     # LEFT (camera-frame) flips
#     for F in (Fx, Fy, Fz):
#         cands.append(score_R(F @ R0))
#     # RIGHT (world-frame) flips
#     for F in (Fx, Fy, Fz):
#         cands.append(score_R(R0 @ F))
#
#     # Partition by cheirality validity
#     valid = [(r, c, qn, tn) for (r, c, qn, tn) in cands if c >= min_ch]
#     resI, chI, _, _ = cands[0]
#
#     # If current is invalid but some candidate is valid → take best valid
#     if chI < min_ch and valid:
#         best_r, _, best_q, best_t = min(valid, key=lambda x: x[0])
#         return best_q, best_t
#
#     # If current is valid → only switch on a big residual gain
#     if valid:
#         best_r, _, best_q, best_t = min(valid, key=lambda x: x[0])
#         if (resI - best_r) >= margin_px:
#             return best_q, best_t
#
#     return est_q, est_t


# --- Camera projection ----------------------------------------------------------

def h(est_q: q, est_t: np.array, feature_points, cal: Calibration):
    """Project all `FEATURE_OFFSETS` into pixel coordinates given pose (q, t).

    The pose maps model points into the camera frame as:  X_cam = q.T * X + t
    Then the custom pinhole projection computes pixels [u, v]:
        u = FX * (-y / x) + CX
        v = FY * ( z / x) + CY

    Returns a flattened length-2N vector [u0, v0, u1, v1, ...].
    """

    xyz_proj = est_q.T * feature_points + est_t
    us_vs_s_proj = np.zeros((xyz_proj.shape[0], 2))
    us_vs_s_proj[:, 0] = cal.fx * xyz_proj[:, 1] / xyz_proj[:, 0] + cal.cx
    us_vs_s_proj[:, 1] = cal.fy * xyz_proj[:, 2] / xyz_proj[:, 0] + cal.cy

    return us_vs_s_proj.flatten()


# --- Analytic Jacobian of h w.r.t. (q, t) -------------------------------------

def deriv(est_q: q, est_t: np.array, feature_points, cal: Calibration):
    """Return analytic Jacobian L = dh/dx evaluated at (est_q, est_t).

    State ordering: x = [qs, qx, qy, qz, tx, ty, tz]^T  (7 parameters)
    Output ordering matches `h()`: [u0, v0, u1, v1, ...] (2N measurements).

    This leverages the quaternion helper `est_q.transpose_vec_deriv(p)` which is
    expected to produce d(X_cam)/dq for a model point `p`, already projected to
    the tangent space of unit quaternions (right-projected onto the constraint).

    Returns
    -------
    np.ndarray, shape (2N, 7)
        Jacobian matrix.
    """

    num_points = len(feature_points)
    L = np.zeros((2 * num_points, 7))

    # Current camera-frame coordinates of each feature
    xyz_proj = est_q.T * feature_points + est_t

    # Unpack for compact per-point derivatives of the projection
    x_hat, y_hat, z_hat = xyz_proj[:, 0], xyz_proj[:, 1], xyz_proj[:, 2]

    # Placeholders for partials of each (u,v) w.r.t. state components
    dfeature_dqs = np.zeros((num_points, 2))
    dfeature_dqy = np.zeros((num_points, 2))
    dfeature_dqz = np.zeros((num_points, 2))
    dfeature_dqx = np.zeros((num_points, 2))

    dfeature_dvx = np.zeros((num_points, 2))  # w.r.t. tx
    dfeature_dvy = np.zeros((num_points, 2))  # w.r.t. ty
    dfeature_dvz = np.zeros((num_points, 2))  # w.r.t. tz

    # Loop over features to accumulate per-point analytic derivatives
    for idx, feature in enumerate(feature_points):
        # new_deriv is the 3x4 Jacobian d(X_cam)/d[q s qx qy qz] for this point
        new_deriv = est_q.transpose_vec_deriv(feature)

        dx_dqs, dx_dqx, dx_dqy, dx_dqz = new_deriv[0, :]
        dy_dqs, dy_dqx, dy_dqy, dy_dqz = new_deriv[1, :]
        dz_dqs, dz_dqx, dz_dqy, dz_dqz = new_deriv[2, :]

        # Translation effect on camera-frame coords is identity
        dx_dtx, dy_dty, dz_dtz = 1.0, 1.0, 1.0

        # Projection partials for u, v with respect to x, y, z at this point
        #   u = FX * ( -y / x ) + CX =>  du/dy = -FX / x, du/dx = FX * y / x^2
        #   v = FY * (  z / x ) + CY =>  dv/dz =  FY / x, dv/dx = -FY * z / x^2
        du_dy = cal.fx / x_hat[idx]
        du_dx = -cal.fx * y_hat[idx] / sq(x_hat[idx])
        dv_dz = cal.fy / x_hat[idx]
        dv_dx = -cal.fy * z_hat[idx] / sq(x_hat[idx])

        # Chain rule: d(u,v)/dq = d(u,v)/d(x,y,z) * d(x,y,z)/dq
        dfeature_dqs[idx, 0] = du_dy * dy_dqs + du_dx * dx_dqs
        dfeature_dqs[idx, 1] = dv_dz * dz_dqs + dv_dx * dx_dqs

        dfeature_dqx[idx, 0] = du_dy * dy_dqx + du_dx * dx_dqx
        dfeature_dqx[idx, 1] = dv_dz * dz_dqx + dv_dx * dx_dqx

        dfeature_dqy[idx, 0] = du_dy * dy_dqy + du_dx * dx_dqy
        dfeature_dqy[idx, 1] = dv_dz * dz_dqy + dv_dx * dx_dqy

        dfeature_dqz[idx, 0] = du_dy * dy_dqz + du_dx * dx_dqz
        dfeature_dqz[idx, 1] = dv_dz * dz_dqz + dv_dx * dx_dqz

        # Translation columns (tx, ty, tz)
        dfeature_dvx[idx, 0] = du_dx * dx_dtx
        dfeature_dvx[idx, 1] = dv_dx * dx_dtx

        dfeature_dvy[idx, 0] = du_dy * dy_dty
        dfeature_dvy[idx, 1] = 0.0  # v unchanged directly by y when holding x,z

        dfeature_dvz[idx, 0] = 0.0  # u unchanged directly by z when holding x,y
        dfeature_dvz[idx, 1] = dv_dz * dz_dtz

    # Stack columns in state order: [qs, qx, qy, qz, tx, ty, tz]
    L[:, 0] = dfeature_dqs.flatten()
    L[:, 1] = dfeature_dqx.flatten()
    L[:, 2] = dfeature_dqy.flatten()
    L[:, 3] = dfeature_dqz.flatten()

    L[:, 4] = dfeature_dvx.flatten()
    L[:, 5] = dfeature_dvy.flatten()
    L[:, 6] = dfeature_dvz.flatten()

    # Finite-difference debug code retained (commented) for validation
    # delt = 1e-5
    # h_1 = h(est_q, est_t)
    # h_2 = h(q(s=est_q.s + delt, vec=est_q.vec + np.array([0.0, 0.0, 0.0])), est_t)
    # print_rayPts((h_2 - h_1) / delt)

    return L


def print_rayPts(ray_proj: np.array):
    """Nicely print a flattened [u0, v0, u1, v1, ...] vector (debug helper)."""
    ray_proj = ray_proj.reshape(-1, 2)
    print(f"Norm: {np.linalg.norm(ray_proj)}")
    for n, ray in enumerate(ray_proj):
        print(f"Feature: {n:3d}, px: {ray[0]: .5f}, py: {ray[1]: .5f}")


def opt(img_pts: NDArray, object_pts: NDArray,
        cal: Calibration, seed_q: q = None, seed_t: NDArray = None, sigma_squared: NDArray = None):
    """Refine pose to minimize ||meas_pix - h(q, t)|| using a GN-like loop.

    Uses the analytic Jacobian `deriv`, a pseudoinverse step `delta_x`, and a
    simple backtracking line search on a scalar `scale` to accept/reject the step
    based on the agreement between linear prediction and actual residual change.

    Termination conditions:
      - Small step (norm(scale * delta_x) < 1e-7)
      - Iteration limit reached (iter > 10)

    Returns
    -------
    (est_q, est_t)
        The refined quaternion and translation.
    """

    if seed_q is None or seed_t is None:
        est_q, est_t = DLT(object_pts, img_pts, cal, sigma_squared)
    else:
        est_q = deepcopy(seed_q)
        est_t = deepcopy(seed_t)

    meas_pix = img_pts.flatten()

    keep_going = True
    iter_num = 0
    while keep_going:
        iter_num += 1

        y = meas_pix - h(est_q, est_t, object_pts, cal)
        old_y_mag = norm(y)
        L = deriv(est_q, est_t, object_pts, cal)

        if sigma_squared is not None:
            Q = np.diag(1.0 / sigma_squared)
            y = Q.dot(y)
            L = Q.dot(L)
        else:
            Q = None

        delta_x = np.linalg.pinv(L).dot(y)

        scale = 1.0
        scale_is_good = False
        while not scale_is_good:
            # Trial step
            if Q is not None:
                new_y_mag = norm(Q.dot(
                    meas_pix - h(q(s=float(est_q.s + scale * delta_x[0]), vec=est_q.vec + scale * delta_x[1:4]),
                                 est_t + scale * delta_x[4:], object_pts, cal)))
            else:
                new_y_mag = norm(
                    meas_pix - h(q(s=float(est_q.s + scale * delta_x[0]), vec=est_q.vec + scale * delta_x[1:4]),
                                 est_t + scale * delta_x[4:], object_pts, cal))

            # Linear prediction of residual magnitude
            y_pred_mag = norm(y - L.dot(scale * delta_x))

            # If perfect agreement between nonlinear and linear prediction, stop
            if np.abs(old_y_mag - y_pred_mag) < 1e-5:
                scale_is_good = True
                keep_going = False
            else:
                # Accept step if the ratio is in a reasonable trust range
                ratio = (old_y_mag - new_y_mag) / (old_y_mag - y_pred_mag)
                if 0.25 < ratio < 4.0:
                    scale_is_good = True
                    est_q = q(
                        s=float(est_q.s + scale * delta_x[0]),
                        vec=est_q.vec + scale * delta_x[1:4],
                    )
                    est_t += scale * delta_x[4:]
                else:
                    # Backtrack
                    scale /= 2.0

        if norm(scale * delta_x) < 1e-7 or iter_num > 10:
            keep_going = False

    return est_q, est_t


def DLT(object_pts: NDArray, img_pts: NDArray, cal: Calibration, sigma_squared: np.array = None):

    num_points = len(img_pts)

    A = np.zeros((2 * num_points, 12))

    for i in range(num_points):
        X, Y, Z = object_pts[i]
        x, y = img_pts[i]

        A[2 * i] = [-X, -Y, -Z, -1, 0, 0, 0, 0, x * X, x * Y, x * Z, x]
        A[2 * i + 1] = [0, 0, 0, 0, -X, -Y, -Z, -1, y * X, y * Y, y * Z, y]

    if sigma_squared is not None:
        Q = np.diag(1.0 / sigma_squared)
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

    P = np.array([[0, 0, 1],
                  [1, 0, 0],
                  [0, 1, 0]])
    R = P @ R
    ytil = (img_pts[:, 0] - cal.cx) / cal.fx
    ztil = (img_pts[:, 1] - cal.cy) / cal.fy
    t = _solve_t_given_R(object_pts, ytil, ztil, R)

    # 5. Get the final rvec
    # rvec, _ = cv2.Rodrigues(R)
    q_init = mat2quat(R.T)

    return q_init, t


def solveQnP(object_pts: np.array, img_pts: np.array, cal: Calibration, sigma_squared=None):
    """
    :param object_pts: Truth Object Points
    :param img_pts: Detected Feature points in image
    :param cal: Camera calibration from Calibration.py
    :param sigma_squared:
    :return:
    """

    sigma_squared = np.ones(2 * len(object_pts))
    sigma_squared[0] = 100.0
    sigma_squared[1] = 100.0

    # q_init, t_init = DLT(object_pts, img_pts, cal, sigma_squared)
    est_q, est_t = opt(img_pts, object_pts, cal, sigma_squared=sigma_squared)
    # est_q, est_t = q_init, t_init

    est_q.force_s_pos()

    # est_q, est_t = _post_refine_flip_biside(est_q, est_t, img_pts, object_pts, cal, min_ch=0.90, margin_px=100.0)

    # print(f'Residual: {norm(img_pts - h(est_q, est_t, object_pts, cal))}')
    # print(f'Init: {init_q}, {init_t}')
    # print(f'InitM: \n{quat2mat(init_q)}')
    # print(f'Est: {est_q}, {est_t}')
    # print(f'EstM: \n{quat2mat(est_q)}')
    # print(f'Trans: \n{quat2mat(init_q) @ quat2mat(est_q.T)}')
    # print(f'Ang Between (deg): {est_q.angle_betweenD(init_q)}\n\n')
    return est_q, est_t


if __name__ == '__main__':
    pass
