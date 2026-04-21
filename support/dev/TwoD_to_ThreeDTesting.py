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

from support.mathHelpers.quaternions import Quaternion as q, mat2quat
import numpy as np
from numpy import square as sq
from numpy.linalg import norm
from copy import deepcopy, copy
from cv2 import (solvePnP, Rodrigues, solvePnPGeneric, solvePnPRansac,
                 SOLVEPNP_AP3P, SOLVEPNP_ITERATIVE, SOLVEPNP_EPNP)
import datetime

# Pretty-printing controls for numpy (purely cosmetic; does not affect math)
np.set_printoptions(suppress=True, precision=4, threshold=np.inf)

NUM_OF_POINTS = 20


def feature_points():
    """Return model-frame feature points as a (N, 3) ndarray.

    Originally, this function created a small cross pattern. For robustness in
    testing, we generate 20 points from a zero-mean Gaussian distribution with a
    standard deviation of 3 in each axis. These are interpreted as the object's
    landmark locations in its own local/model coordinate frame. The *pose*
    (q, t) maps these to the camera frame before projection.

    Returns
    -------
    np.ndarray, shape (N, 3)
        Model-frame 3D feature coordinates.
    """
    # Example deterministic pattern kept for reference:
    # going_out = np.array([0.0, 0.0, 0.0])
    # going_out = np.vstack((going_out, np.array([1.0, 0.0, 0.0])))
    # going_out = np.vstack((going_out, np.array([-1.0, 0.0, 0.0])))
    # going_out = np.vstack((going_out, np.array([0.0, 1.0, 0.0])))
    # going_out = np.vstack((going_out, np.array([0.0, -1.0, 0.0])))

    going_out = np.random.normal(0.0, 3.0, (NUM_OF_POINTS, 3))
    return going_out


def pixel_point_covariances():
    sigma_squared = sq(np.random.normal(3.0, 2.0, (2 * NUM_OF_POINTS)))
    return sigma_squared


# --- Global synthetic scene & camera intrinsics --------------------------------
FEATURE_OFFSETS = feature_points()
FX, FY = 800.0, 800.0  # focal lengths (pixels)
CX, CY = 432.0, 432.0  # principal point (pixels)
# Intrinsic matrix written in author's custom arrangement; the projection uses
# explicit FX, FY, CX, CY in h(); INTRINSIC is kept for reference/debug.
INTRINSIC = np.array([[0.0, 0.0, 1.0], [FX, 0.0, CX], [0.0, FY, CY]])

# Camera mapping between your projection and OpenCV's pinhole.
# Your pixels: u = FX*(-y/x)+CX , v = FY*( z/x)+CY
# OpenCV:      u = FX*(Xc/Zc)+CX, v = FY*(Yc/Zc)+CY
# The axis map that matches the pixel equations is:
#   [Xc, Yc, Zc]^T = C * [x, y, z]^T with C below (det = -1).
C_OURS_TO_CV = np.array([[0., 1., 0.],
                         [0., 0., 1.],
                         [1., 0., 0.]], dtype=float)
C_CV_TO_OURS = C_OURS_TO_CV.T


K_CV = np.array([[FX, 0., CX],
                 [0., FY, CY],
                 [0., 0., 1.]], dtype=float)
DIST_COEFFS = np.zeros(5, dtype=float)


def _cv_pose_to_ours(R_cv: np.ndarray, t_cv: np.ndarray):
    """Convert OpenCV camera pose to your convention (proper rotation)."""
    R_ours = C_CV_TO_OURS @ R_cv
    t_ours = C_CV_TO_OURS @ t_cv
    return mat2quat(R_ours.T), t_ours


def _rms_h(q_, t_, meas_flat):
    r = (h(q_, t_) - meas_flat).reshape(-1, 2)
    return float(np.sqrt(np.mean(r[:, 0] ** 2 + r[:, 1] ** 2)))


def opencv_pnp_iterative_baseline(object_points: np.ndarray,
                                  image_points_flat: np.ndarray,
                                  cheirality_thresh: float = 0.8):
    """
    Pure OpenCV baseline with cheirality enforcement.

    Step 1: SOLVEPNP_ITERATIVE (no seed).
    Step 2: Cheirality check in OpenCV frame (fraction of points with Zc > 0).
    Step 3: If low, recover via solvePnPGeneric(AP3P) to get multiple candidates,
            pick the one with the highest positive-depth fraction, then polish
            with ITERATIVE (useExtrinsicGuess=True). No seeding from 'our' method.
    """
    # Prep data in OpenCV's model/camera convention (handedness-safe bridge)
    img = image_points_flat.reshape(-1, 2).astype(np.float64)
    obj_cv = object_points

    def pos_depth_frac(R_cv: np.ndarray, t_cv: np.ndarray) -> float:
        # Z in OpenCV camera frame
        Z = (obj_cv @ R_cv.T)[:, 2] + t_cv[2]
        return float(np.mean(Z > 0.0))

    # --- 1) Plain ITERATIVE (no seed) ---
    ok, rvec, tvec = solvePnP(
        objectPoints=obj_cv,
        imagePoints=img,
        cameraMatrix=K_CV,
        distCoeffs=DIST_COEFFS,
        flags=SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("OpenCV SOLVEPNP_ITERATIVE failed (no seed).")

    R_cv, _ = Rodrigues(rvec)
    t_cv = tvec.reshape(3)
    pf = pos_depth_frac(R_cv, t_cv)

    # --- 2) If cheirality is good, convert & return ---
    if pf >= cheirality_thresh:
        return _cv_pose_to_ours(R_cv, t_cv)

    # --- 3) Recover via multi-hypothesis AP3P, pick front-most, then polish ---
    # solvePnPGeneric returns multiple (rvecs, tvecs); we choose by positive-depth fraction
    retval, rvecs, tvecs, reprojErrs = solvePnPGeneric(
        objectPoints=obj_cv,
        imagePoints=img,
        cameraMatrix=K_CV,
        distCoeffs=DIST_COEFFS,
        flags=SOLVEPNP_AP3P,  # multi-solution minimal solver
    )
    if not retval or len(rvecs) == 0:
        # Fallback to the original even if cheirality is poor
        return _cv_pose_to_ours(R_cv, t_cv)

    # Rank candidates: highest positive-depth fraction, break ties by lower reprojection error
    best = None
    best_key = None
    for i in range(len(rvecs)):
        R_i, _ = Rodrigues(rvecs[i]);
        t_i = tvecs[i].reshape(3)
        pf_i = pos_depth_frac(R_i, t_i)
        err_i = float(reprojErrs[i]) if reprojErrs is not None and len(reprojErrs) > i else np.inf
        key = (-pf_i, err_i)  # maximize pf_i, then minimize error
        if best is None or key < best_key:
            best = (R_i, t_i)
            best_key = key

    R_seed, t_seed = best

    # Final polish on ALL points with ITERATIVE using the chosen candidate as an initial guess
    ok_polish, rvec_pol, tvec_pol = solvePnP(
        objectPoints=obj_cv,
        imagePoints=img,
        cameraMatrix=K_CV,
        distCoeffs=DIST_COEFFS,
        rvec=Rodrigues(R_seed)[0],
        tvec=t_seed.reshape(3, 1),
        useExtrinsicGuess=True,
        flags=SOLVEPNP_ITERATIVE,
    )
    if ok_polish:
        R_cv, _ = Rodrigues(rvec_pol)
        t_cv = tvec_pol.reshape(3)
    else:
        R_cv, t_cv = R_seed, t_seed  # use the AP3P candidate directly

    # Convert back to your convention (proper rotation) and return
    return _cv_pose_to_ours(R_cv, t_cv)


def opencv_pnp_ransac_baseline(object_points: np.ndarray,
                               image_points_flat: np.ndarray,
                               ransac_thresh_px: float = 10.0):
    """
    OpenCV RANSAC baseline: AP3P minimal → (optional) inlier refit → polish.
    No seeding from your pipeline.
    """
    img = image_points_flat.reshape(-1, 2).astype(np.float64)
    obj_cv = object_points

    ok, rvec, tvec, inliers = solvePnPRansac(
        objectPoints=obj_cv,
        imagePoints=img,
        cameraMatrix=K_CV,
        distCoeffs=DIST_COEFFS,
        reprojectionError=ransac_thresh_px,  # try 8–12 px for ~5 px per-axis noise
        confidence=0.999,
        iterationsCount=3000,
        flags=SOLVEPNP_AP3P,
    )
    if not ok or inliers is None or len(inliers) < 4:
        raise RuntimeError(f"RANSAC failed or too few inliers ({0 if inliers is None else len(inliers)}).")

    # Refit on inliers (ITERATIVE if ≥6, else EPNP), then polish on ALL
    inliers = inliers.reshape(-1)
    obj_in, img_in = obj_cv[inliers], img[inliers]

    if len(inliers) >= 6:
        ok2, rvec_refit, tvec_refit = solvePnP(obj_in, img_in, K_CV, DIST_COEFFS, flags=SOLVEPNP_ITERATIVE)
    else:
        ok2, rvec_refit, tvec_refit = solvePnP(obj_in, img_in, K_CV, DIST_COEFFS, flags=SOLVEPNP_EPNP)
    if not ok2:
        rvec_refit, tvec_refit = rvec, tvec

    ok3, rvec_final, tvec_final = solvePnP(
        objectPoints=obj_in,
        imagePoints=img_in,
        cameraMatrix=K_CV,
        distCoeffs=DIST_COEFFS,
        rvec=rvec_refit,
        tvec=tvec_refit,
        useExtrinsicGuess=True,
        flags=SOLVEPNP_ITERATIVE,
    )
    if not ok3:
        rvec_final, tvec_final = rvec_refit, tvec_refit

    R_cv, _ = Rodrigues(rvec_final)
    t_cv = tvec_final.reshape(3)
    q_ours, t_ours = _cv_pose_to_ours(R_cv, t_cv)
    return (q_ours, t_ours), inliers


# --- Small helpers --------------------------------------------------------------

def _row_normed(A, eps=1e-12):
    """Row-normalize a 2D array.

    Each row is divided by its L2 norm; very small norms are clamped by *eps* to
    avoid division by ~0.
    """
    n = np.linalg.norm(A, axis=1, keepdims=True)
    return A / np.clip(n, eps, None)


def _solve_t_given_R(Xw, y_tilde, z_tilde, R, w=None):
    """Solve translation **t** linearly given a rotation **R** and image rays.

    Given the specific projection equations used in `h()`:
      u = fx * ( -y / x ) + cx  -> y/x = - (u - cx) / fx = y_tilde
      v = fy * (  z / x ) + cy  -> z/x =   (v - cy) / fy = z_tilde

    For a world point Xw transformed into the camera frame by R and t,
    [x, y, z]^T = R Xw + t. Rearranging the two scalar constraints per point
    yields a linear system A t = b that we solve in least squares.

    Parameters
    ----------
    Xw : (N, 3) ndarray
        World/model points.
    y_tilde, z_tilde : (N,) ndarray
        Normalized image coordinate ratios per point.
    R : (3, 3) ndarray
        Rotation taking world points into the camera frame (see `init_pose_wahba`).
    w : (N,) ndarray or None
        Optional weights per correspondence; if provided, they are duplicated for
        the 2 equations per point.

    Returns
    -------
    t : (3,) ndarray
        Least-squares translation estimate.
    """
    r1, r2, r3 = R
    a1, a2, a3 = Xw @ r1, Xw @ r2, Xw @ r3
    N = Xw.shape[0]

    # Build 2N equations: one from u (y/x) and one from v (z/x)
    A = np.zeros((2 * N, 3))
    b = np.zeros(2 * N)

    # From  -y_tilde * (a1 + tx) = a2 + ty  ->  (-y_tilde)*tx - ty = a2 + y_tilde*a1
    A[0::2, 0], A[0::2, 1], b[0::2] = -y_tilde, -1.0, a2 + y_tilde * a1

    # From   z_tilde * (a1 + tx) = a3 + tz  ->   (z_tilde)*tx - tz = a3 - z_tilde*a1
    A[1::2, 0], A[1::2, 2], b[1::2] = z_tilde, -1.0, a3 - z_tilde * a1

    if w is not None:
        # Apply weights to both the rows (u and v) per correspondence
        ww = np.repeat(w, 2)
        A = ww[:, None] * A
        b = ww * b

    # Solve in least squares (minimum-norm if underdetermined)
    t, *_ = np.linalg.lstsq(A, b, rcond=None)
    return t


# --- Fast initializer: Wahba/Kabsch rotation + linear t ------------------------

def init_pose_wahba(Xw, meas_pix, fx, fy, cx, cy):
    """Compute a fast pose seed from image measurements.

    Steps:
    1) Build unit *bearing* vectors in the camera frame from pixel coordinates
       using the custom projection used by `h()`.
    2) Compute a rotation R (world->camera) aligning normalized directions of
       centered 3D points to those bearings via a weighted Wahba/Kabsch SVD.
    3) Solve translation t from linear least squares holding R fixed.

    Returns
    -------
    (q_init, t_init)
        Quaternion whose transpose acts as R in `h()` (i.e., `q_init.T == R`),
        and the translation vector.
    """
    Xw = np.asarray(Xw, dtype=float).reshape(-1, 3)
    uv = np.asarray(meas_pix, dtype=float).reshape(-1, 2)

    # From h():  u = fx*(-y/x)+cx,  v = fy*(z/x)+cy
    y_tilde = -(uv[:, 0] - cx) / fx
    z_tilde = (uv[:, 1] - cy) / fy

    # Unit camera-frame bearings proportional to [1, -y_tilde, z_tilde]
    B = _row_normed(np.column_stack([np.ones(len(uv)), -y_tilde, z_tilde]))

    # Remove translation by centering, then normalize direction of each point
    Xc = Xw - Xw.mean(axis=0, keepdims=True)
    Xdir = _row_normed(Xc)

    # (Optional) weight by baseline length to emphasize well-separated points
    w = np.linalg.norm(Xc, axis=1) + 1e-9

    # Wahba/Kabsch: find R maximizing sum w_i <R Xdir_i, B_i>
    H = (Xdir * w[:, None]).T @ B
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Enforce a proper rotation (det=+1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # One-shot translation from the linearized projection constraints
    t = _solve_t_given_R(Xw, y_tilde, z_tilde, R, w=w)

    # Cheirality/handedness guard: ensure positive median depth; keep det=+1
    depths = Xw @ R[0] + t[0]
    if np.median(depths) <= 0:
        # Flip first two columns (equivalent to 180 deg around z) and adapt t
        R = R @ np.diag([-1, -1, 1])
        t[:2] *= -1.0

    # IMPORTANT: h() uses est_q.T * X + t, so est_q.T must equal R
    return mat2quat(R.T), t


# --- Camera projection ----------------------------------------------------------

def h(est_q: q, est_t: np.array, feature_points, fx, fy, cx, cy):
    """Project all `FEATURE_OFFSETS` into pixel coordinates given pose (q, t).

    The pose maps model points into the camera frame as:  X_cam = q.T * X + t
    Then the custom pinhole projection computes pixels [u, v]:
        u = FX * (-y / x) + CX
        v = FY * ( z / x) + CY

    Returns a flattened length-2N vector [u0, v0, u1, v1, ...].
    """
    xyz_proj = est_q.T * feature_points + est_t
    us_vs_s_proj = np.zeros((xyz_proj.shape[0], 2))
    us_vs_s_proj[:, 0] = fx * xyz_proj[:, 1] / xyz_proj[:, 0] + cx
    us_vs_s_proj[:, 1] = fy * xyz_proj[:, 2] / xyz_proj[:, 0] + cy

    return us_vs_s_proj.flatten()


# --- Analytic Jacobian of h w.r.t. (q, t) -------------------------------------

def deriv(est_q: q, est_t: np.array, feature_points=FEATURE_OFFSETS,
          fx=FX, fy=FY, cx=CX, cy=CY):
    """Return analytic Jacobian L = dh/dx evaluated at (est_q, est_t).

    State ordering: x = [qs, qx, qy, qz, tx, ty, tz]^T  (7 parameters)
    Output ordering matches `h()`: [u0, v0, u1, v1, ...] (2N measurements).

    This leverages the quaternion helper `est_q.transpose_vect_deriv(p)` which is
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
        new_deriv = est_q.transpose_vect_deriv(feature)

        dx_dqs, dx_dqx, dx_dqy, dx_dqz = new_deriv[0, :]
        dy_dqs, dy_dqx, dy_dqy, dy_dqz = new_deriv[1, :]
        dz_dqs, dz_dqx, dz_dqy, dz_dqz = new_deriv[2, :]

        # Translation effect on camera-frame coords is identity
        dx_dtx, dy_dty, dz_dtz = 1.0, 1.0, 1.0

        # Projection partials for u, v with respect to x, y, z at this point
        #   u = FX * ( -y / x ) + CX =>  du/dy = -FX / x, du/dx = FX * y / x^2
        #   v = FY * (  z / x ) + CY =>  dv/dz =  FY / x, dv/dx = -FY * z / x^2
        du_dy = fx / x_hat[idx]
        du_dx = -fx * y_hat[idx] / sq(x_hat[idx])
        dv_dz = fy / x_hat[idx]
        dv_dx = -fy * z_hat[idx] / sq(x_hat[idx])

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


def opt(est_q: q, est_t: np.array, meas_pix: np.array, sigma_squared: np.array = None, feature_points=FEATURE_OFFSETS,
        fx=FX, fy=FY, cx=CX, cy=CY):
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
    keep_going = True
    iter = 0
    while keep_going:
        iter += 1

        y = meas_pix - h(est_q, est_t, feature_points, fx, fy, cx, cy)
        old_y_mag = norm(y)
        L = deriv(est_q, est_t, feature_points, fx, fy, cx, cy)

        if sigma_squared is not None:
            Q = np.diag(1.0 / sigma_squared)
            y = Q.dot(y)
            L = Q.dot(L)

        delta_x = np.linalg.pinv(L).dot(y)

        scale = 1.0
        scale_is_good = False
        while not scale_is_good:
            # Trial step
            if sigma_squared is not None:
                new_y_mag = norm(Q.dot(
                    # new_y_mag=norm(
                    meas_pix - h(q(s=est_q.s + scale * delta_x[0], vec=est_q.vec + scale * delta_x[1:4]),
                                 est_t + scale * delta_x[4:],
                                 feature_points,
                                 fx, fy, cx, cy)))
            else:
                new_y_mag = norm(
                    meas_pix - h(q(s=est_q.s + scale * delta_x[0], vec=est_q.vec + scale * delta_x[1:4]),
                                 est_t + scale * delta_x[4:],
                                 feature_points, fx, fy, cx, cy))

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
                        s=est_q.s + scale * delta_x[0],
                        vec=est_q.vec + scale * delta_x[1:4],
                    )
                    est_t += scale * delta_x[4:]
                else:
                    # Backtrack
                    scale /= 2.0

        if norm(scale * delta_x) < 1e-7 or iter > 10:
            keep_going = False

    return est_q, est_t


def solveQnP(object_pts, img_pts, fx, fy, cx, cy, sigma_squared=None):
    img_pts = deepcopy(img_pts).flatten()
    init_q, init_t = init_pose_wahba(object_pts, img_pts, fx, fy, cx, cy)
    est_q, est_t = opt(init_q, init_t, img_pts, sigma_squared, object_pts, fx, fy, cx, cy)
    est_q.force_s_pos()
    return est_q, est_t


# --- Demo / entry point --------------------------------------------------------

def main():
    """Run a synthetic pose-estimation demo using the functions above."""
    # Convention: x forward, y left, z up

    # Create a ground-truth orientation by Euler roll-pitch-yaw in degrees
    true_q = q()
    true_q.from_eulerD_rpy(np.array([0.0, 10.0, 45.0]))
    print(true_q.to_rodrigues())

    # Ground-truth translation in camera frame
    true_t = np.array([20.0, 0.0, 0.0])

    # Ideal projections of model features
    orig_meas_pix = h(true_q, true_t, FEATURE_OFFSETS, FX, FY, CX, CY)

    print("True Location in pixel space")
    print_rayPts(orig_meas_pix)

    sigma_squared = pixel_point_covariances()
    sigma_squared = None

    meas_pix = np.zeros_like(orig_meas_pix)
    if sigma_squared is not None:
        # Add Gaussian pixel noise based on covariances created per pixel
        for idx, (m_pix, std_sq) in enumerate(zip(orig_meas_pix, sigma_squared)):
            meas_pix[idx] = np.random.normal(m_pix, np.sqrt(std_sq))
    else:
        # Add i.i.d. Gaussian pixel noise (sigma=5 px) to simulate detections
        meas_pix = orig_meas_pix + np.random.normal(0.0, 2.0, orig_meas_pix.shape)

    print("\nNoisy \"Measured\" Location in pixel space")
    print_rayPts(meas_pix)

    print(f"Covariances: \n{sigma_squared}")

    # Fast seed via Wahba + linear t
    seed_start_time = datetime.datetime.now()
    init_q, init_t = init_pose_wahba(FEATURE_OFFSETS, meas_pix, FX, FY, CX, CY)
    seed_end_time = datetime.datetime.now()
    est_q = copy(init_q)
    est_t = copy(init_t)
    print(f"Init:")
    print(est_q, est_t)

    # Nonlinear refinement
    opt_start_time = datetime.datetime.now()
    est_q, est_t = opt(est_q, est_t, meas_pix, sigma_squared)
    est_q.force_s_pos()
    opt_end_time = datetime.datetime.now()

    # Diagnostics: compare residuals
    print("Final Residual: ")
    print_rayPts(h(est_q, est_t, FEATURE_OFFSETS, FX, FY, CX, CY) - meas_pix)
    print("Optimal Residual: ")
    print_rayPts(h(true_q, true_t, FEATURE_OFFSETS, FX, FY, CX, CY) - meas_pix)

    # Summary
    print("\n\nTrue:")
    print(true_q, true_t, '')

    print("\n\nInitial guess:")
    print(init_q, init_t)
    print("Est:")
    print(est_q, est_t)
    print(f"Angle Between True and Init Quat(deg): \n{true_q.angle_betweenD(init_q):.4f}")
    print(f"Distance Between True and Init tvec: \n{norm(true_t - init_t):.4f}")
    print(f"Angle Between True and Est Quat(deg): \n{true_q.angle_betweenD(est_q):.4f}")
    print(f"Distance Between True and Est tvec: \n{norm(true_t - est_t):.4f}")
    # print(f'Seed Time taken: {seed_end_time - seed_start_time}')
    # print(f'Opt Time taken: {opt_end_time - opt_start_time}')
    # print(f'Total Time taken: {seed_end_time - seed_start_time + opt_end_time - opt_start_time}')

    iter_start_time = datetime.datetime.now()
    cv_q, cv_t = opencv_pnp_iterative_baseline(FEATURE_OFFSETS, meas_pix)
    iter_end_time = datetime.datetime.now()

    rans_start_time = datetime.datetime.now()
    (cvR_q, cvR_t), inliers = opencv_pnp_ransac_baseline(FEATURE_OFFSETS, meas_pix)
    rans_end_time = datetime.datetime.now()

    print("\nOpenCV SolvePnP Iterative:")
    print(cv_q, cv_t)
    print(f"Angle Between True and Est Quat(deg): \n{true_q.angle_betweenD(cv_q):.4f}")
    print(f"Distance Between True and Est tvec: \n{norm(true_t - cv_t):.4f}")
    # print(f'Total Time taken: {iter_end_time - iter_start_time}')
    print("\nOpenCV SolvePnP RANSAC:")
    print(cvR_q, cvR_t)
    print(f"Angle Between True and Est Quat(deg): \n{true_q.angle_betweenD(cvR_q):.4f}")
    print(f"Distance Between True and Est tvec: \n{norm(true_t - cvR_t):.4f}")
    # print(f'Total Time taken: {rans_end_time - rans_start_time}')
    print(f'Inliers: {len(inliers)} / {len(FEATURE_OFFSETS)}')
    print(inliers)


if __name__ == '__main__':
    main()
