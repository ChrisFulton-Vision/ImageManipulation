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

from quaternions import Quaternion as q
from quaternions import *
import numpy as np
from numpy import square as sq, abs
from numpy.linalg import norm
from typing_extensions import Tuple
from copy import deepcopy
import datetime

# Pretty-printing controls for numpy (purely cosmetic; does not affect math)
np.set_printoptions(suppress=True, precision=4, threshold=np.inf)
EPS = 0.000001

class ThreeD_to_ThreeD:
    def __init__(self, points1: np.array, points2: np.array) -> None:
        self.points1 = deepcopy(points1)
        self.points2 = deepcopy(points2)
        self.num_points = self.points1.shape[0]
        self.q, self.t = self.init_pose(points1, points2)
        self.opt()


    @staticmethod
    def gramSchmidtAxis(points: np.array) -> q | None:
        x_axis = points[1] - points[0]
        x_nm = norm(x_axis)
        if x_nm < EPS:
            # X axis degenerate
            return
        x_axis /= x_nm

        yp_axis = points[2] - points[0]
        yp_nm = norm(yp_axis)
        if yp_nm < EPS:
            # Y axis degenerate
            return

        z_axis = np.cross(x_axis, yp_axis)
        z_nm = norm(z_axis)
        if z_nm < EPS:
            # Z axis degenerate
            return
        z_axis /= z_nm

        y_axis = np.cross(z_axis, x_axis)

        mat = np.hstack([x_axis.T, y_axis.T, z_axis.T])
        return mat2quat(mat)

    @staticmethod
    def init_pose(points1: np.array, points2: np.array) -> Tuple[q, np.array]:
        quat1 = ThreeD_to_ThreeD.gramSchmidtAxis(points1)
        quat2 = ThreeD_to_ThreeD.gramSchmidtAxis(points2)

        if quat1 is None or quat2 is None:
            return None

        new_q = quat2.T * quat1
        if new_q.s < 0.0:
            new_q *= -1.0

        tvec = np.mean(points2.reshape(-1, 3), axis=0) - np.mean(new_q * (points1.reshape(-1, 3)), axis=0)

        return new_q, tvec

    def create_y(self, new_q: q = None, new_t: np.array = None) -> np.array:
        if new_q is None:
            new_q = deepcopy(self.q)
        if new_t is None:
            new_t = deepcopy(self.t)

        return (self.points2 - new_q * self.points1 - new_t).flatten()


    def create_L(self):
        L = np.zeros((3 * self.num_points, 7))


        for idx, pt1 in enumerate(self.points1):
            x_row_idx = idx * 3

            analy_deriv = self.q.vect_deriv(pt1, False)

            L[x_row_idx:x_row_idx + 3, :4] = analy_deriv
            L[x_row_idx:x_row_idx + 3, 4:] = np.eye(3)

        return L

    def opt(self) -> None:
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

            y = self.create_y()
            old_y_mag = norm(y)
            L = self.create_L()

            delta_x = np.linalg.pinv(L).dot(y)

            scale = 1.0
            scale_is_good = False
            while not scale_is_good:
                new_q = q(quat=self.q.ndarray + scale * delta_x[:4], makeUnitVec=True)
                new_t = self.t + scale * delta_x[4:]
                new_y_mag = norm(self.create_y(new_q, new_t))

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
                        self.q = q(quat=self.q.ndarray + scale * delta_x[:4], makeUnitVec=True)
                        self.t += scale * delta_x[4:]
                    else:
                        # Backtrack
                        scale /= 2.0

            if norm(scale * delta_x) < 1e-7 or iter > 10:
                keep_going = False

        print("Estimated:")
        print(self.q, self.t)

def print_3dPts(threeD_proj: np.array):
    """Nicely print a flattened [x0, y0, z0, x1, ...] vector (debug helper)."""
    threeD_proj = copy.deepcopy(threeD_proj).reshape(-1, 3)
    print(f"Norm: {np.linalg.norm(threeD_proj)}")
    for n, point in enumerate(threeD_proj):
        print(f"Feature: {n:3d}, x: {point[0]: .5f}, y: {point[1]: .5f}, z: {point[2]: .5f}")


# --- Demo / entry point --------------------------------------------------------

def main():

    test1 = np.random.normal(0.0, 1.0, (10,3))
    noise = np.random.normal(0.0, 0.1, test1.shape)
    q = randomQuat()
    t = np.array([10.0, 0.0, 0.0]) + np.random.normal(1.0, 1.0, (3,))
    test2 = (q * test1 + t) + noise

    print(f'Targets: \n{q}\n{t}')

    optClass = ThreeD_to_ThreeD(test1, test2)

    print(f'Resolved Residual: {norm(optClass.create_y())}\n{optClass.create_y()}')
    print(f'True Residual: {norm(optClass.create_y(q, t))}\n{optClass.create_y(q, t)}')



if __name__ == '__main__':
    main()
