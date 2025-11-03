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
import numpy as np
from numpy import square as sq
from numpy.linalg import norm
from numpy.typing import NDArray
from copy import deepcopy

# Pretty-printing controls for numpy (purely cosmetic; does not affect math)
np.set_printoptions(suppress=True, precision=4, threshold=maxsize)

# --- Small helpers --------------------------------------------------------------

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


# --- Analytic Jacobian of h w.r.t. (q, t) -------------------------------------

def deriv(est_q: q, est_t: np.array, feature_points, cal: Calibration):
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
    xyz_proj = est_q * feature_points + est_t

    # Unpack for compact per-point derivatives of the projection
    X_hat, Y_hat, Z_hat = xyz_proj[:, 0], xyz_proj[:, 1], xyz_proj[:, 2]

    # Loop over features to accumulate per-point analytic derivatives
    for idx, feature in enumerate(feature_points):
        # new_deriv is the 3x4 Jacobian d(X_cam)/d[q s qx qy qz] for this point
        new_deriv = est_q.vect_deriv(feature, False)

        # Translation effect on camera-frame coords is identity
        # dx_dtx, dy_dty, dz_dtz = 1.0, 1.0, 1.0

        # Projection partials for u, v with respect to x, y, z at this point
        #   u = FX * (  x / z ) + CX =>  du/dx =  FX / z, du/dz = -FX * x / z^2
        #   v = FY * (  y / z ) + CY =>  dv/dy =  FY / z, dv/dz = -FY * y / z^2
        du_dX = cal.fx / Z_hat[idx]
        du_dZ = -cal.fx * X_hat[idx] / sq(Z_hat[idx])

        dv_dY = cal.fy / Z_hat[idx]
        dv_dZ = -cal.fy * Y_hat[idx] / sq(Z_hat[idx])

        dUV_dXYZ = np.array([[du_dX, 0.0, du_dZ],
                           [0.0, dv_dY, dv_dZ]])

        L[2*idx:2*idx+2, 0:4] = dUV_dXYZ @ new_deriv

        # Translation columns (∂(X,Y,Z)/∂t = I)
        L[2*idx:2*idx+2, 4:7] = dUV_dXYZ

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
    # print(f'EstT: \n{est_t}')
    # print(f'Trans: \n{quat2mat(init_q) @ quat2mat(est_q.T)}')
    # print(f'Ang Between (deg): {est_q.angle_betweenD(init_q)}\n\n')
    return est_q, est_t


if __name__ == '__main__':
    pass
