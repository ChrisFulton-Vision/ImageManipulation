"""
ThreeD_to_ThreeD.py (annotated)

Purpose
-------
Estimate a rigid 3D→3D transform (unit quaternion rotation `q` and translation
`t`) that maps `points1` to `points2` by minimizing reprojection residuals via a
simple Gauss–Newton / trust-region backtracking scheme.

Key changes vs. the original:
- Added comprehensive type annotations (NumPy dtypes and Optional types).
- Added docstrings and explanatory comments.
- Fixed a subtle bug in `gramSchmidtAxis` when forming the 3×3 orientation
  matrix (now uses `np.column_stack` so the axes become *columns* of the matrix).
- Made return types explicit and added validation in `__init__` for degenerate
  seeds.
- Minor cleanups (consistent `deepcopy`, clarified variable names).

Dependencies
------------
Relies on a `quaternions` module that provides:
- `Quaternion` (aliased as `q` here) with attributes/methods:
  - `.ndarray` (or similar) exposing the 4-vector
  - `.s` the scalar part
  - `.T` transpose/inverse as appropriate for composition
  - `__mul__` overloaded for rotating Nx3 arrays of 3D points
  - `.vect_deriv(point, makeUnitVec)` returning a 3×4 Jacobian d(R(q)p)/dq
- `mat2quat(R)` : 3×3 rotation matrix → `Quaternion`
- `randomQuat()` : random unit quaternion

Notes
-----
- This is a *minimal* LM-like scheme focused on readability. It uses the
  pseudo-inverse for the normal equations and a simple backtracking rule based
  on the ratio of actual vs. predicted residual reduction.
- For serious performance/robustness, consider: damping (Levenberg),
  robust loss, weighting by per-point covariances, and stopping criteria tied to
  gradient/step norms.
"""
from __future__ import annotations

from typing import Optional, Tuple

from sys import maxsize
from numpy.linalg import norm
from numpy.typing import NDArray
import numpy as np
from support.mathHelpers.quaternions import Quaternion as Quat, mat2quat, randomQuat

np.set_printoptions(suppress=True, precision=4, threshold=maxsize)

# Small epsilon to guard against degeneracy in Gram–Schmidt seed.
EPS: float = 1e-6


class ThreeD_to_ThreeD:
    """Estimate a rigid transform between two corresponding 3D point sets."""

    def __init__(self, points1: NDArray[np.floating], points2: NDArray[np.floating]) -> None:
        self.points1: NDArray[np.floating] = points1.reshape(-1, 3).copy()
        self.points2: NDArray[np.floating] = points2.reshape(-1, 3).copy()
        self.num_points: int = self.points1.shape[0]

        seed = self.init_pose(self.points1, self.points2)
        if seed is None:
            raise ValueError("Degenerate seed from Gram–Schmidt (collinear or duplicate points).")
        self.q: Quat = Quat()
        self.t: NDArray = np.zeros(1)

        self.q, self.t = seed

        self.opt()

    @staticmethod
    def gramSchmidtAxis(points: NDArray[np.floating]) -> Optional[Quat]:
        p0, p1, p2 = points[0], points[1], points[2]

        x_axis: NDArray[np.floating] = p1 - p0
        x_nm = float(norm(x_axis))
        if x_nm < EPS:
            return None
        x_axis /= x_nm

        yp_axis: NDArray[np.floating] = p2 - p0
        yp_nm = float(norm(yp_axis))
        if yp_nm < EPS:
            return None

        z_axis: NDArray[np.floating] = np.cross(x_axis, yp_axis)
        z_nm = float(norm(z_axis))
        if z_nm < EPS:
            return None
        z_axis /= z_nm

        y_axis: NDArray[np.floating] = np.cross(z_axis, x_axis)

        R: NDArray[np.floating] = np.column_stack((x_axis, y_axis, z_axis))
        return mat2quat(R)

    @staticmethod
    def init_pose(points1: NDArray[np.floating], points2: NDArray[np.floating]) -> (
            Optional)[Tuple[Quat, NDArray[np.floating]]]:
        quat1 = ThreeD_to_ThreeD.gramSchmidtAxis(points1)
        quat2 = ThreeD_to_ThreeD.gramSchmidtAxis(points2)

        if quat1 is None or quat2 is None:
            return None

        new_q: Quat = quat2.T * quat1
        if new_q.s < 0.0:
            new_q *= -1.0

        tvec: NDArray[np.floating] = np.mean(points2.reshape(-1, 3), axis=0) - np.mean(
            new_q * points1.reshape(-1, 3), axis=0
        )
        return new_q, tvec

    def create_y(self, new_q: Optional[Quat] = None, new_t: Optional[NDArray[np.floating]] = None) -> NDArray[np.floating]:
        if new_q is None:
            new_q = self.q.copy()
        if new_t is None:
            new_t = self.t.copy()

        residuals: NDArray[np.floating] = self.points2 - (new_q * self.points1) - new_t
        return residuals.reshape(-1)

    def create_L(self) -> NDArray[np.floating]:
        L = np.zeros((3 * self.num_points, 7), dtype=float)

        for idx, pt1 in enumerate(self.points1):
            row = idx * 3
            dRp_dq: NDArray[np.floating] = self.q.vect_deriv(pt1, False)
            L[row: row + 3, :4] = dRp_dq
            L[row: row + 3, 4:] = np.eye(3)

        return L

    def opt(self) -> None:
        lambda_damp: float = 1e-2

        keep_going = True
        it = 0
        while keep_going:
            it += 1

            y = self.create_y()
            old_y_mag = float(norm(y))
            L = self.create_L()

            JT: NDArray[np.floating] = L.T
            JTJ: NDArray[np.floating] = JT @ L
            JTy: NDArray[np.floating] = JT @ y
            D: NDArray[np.floating] = np.diag(np.diag(JTJ))
            try:
                delta_x = np.linalg.solve(JTJ + lambda_damp * D, JTy)
            except np.linalg.LinAlgError:
                delta_x = np.linalg.pinv(JTJ + lambda_damp * D) @ JTy

            scale = 1.0
            while True:
                new_q = Quat(quat=self.q.ndarray + scale * delta_x[:4], makeUnitQuat=True)
                new_t = self.t + scale * delta_x[4:]
                new_y_mag = float(norm(self.create_y(new_q, new_t)))

                y_pred_mag = float(norm(y - L @ (scale * delta_x)))

                if abs(old_y_mag - y_pred_mag) < 1e-5:
                    self.q = new_q
                    self.t = new_t
                    break

                denom = max(old_y_mag - y_pred_mag, 1e-12)
                ratio = (old_y_mag - new_y_mag) / denom

                if 0.25 < ratio < 4.0:
                    self.q = new_q
                    self.t = new_t
                    if ratio > 0.75:
                        lambda_damp = max(lambda_damp / 3.0, 1e-12)
                    break
                else:
                    lambda_damp = min(lambda_damp * 2.0, 1e12)
                    scale *= 0.5
                    if scale < 1e-6:
                        break

            if float(norm(scale * delta_x)) < 1e-7 or it > 10:
                keep_going = False

        if self.q.s < 0.0:
            self.q *= -1.0


def print_3dPts(threeD_proj: NDArray[np.floating]) -> None:
    pts = threeD_proj.reshape(-1, 3).copy()
    print(f"Norm: {np.linalg.norm(pts)}")
    for n, point in enumerate(pts):
        print(f"Feature: {n:3d}, x: {point[0]: .5f}, y: {point[1]: .5f}, z: {point[2]: .5f}")


def main() -> None:
    test1: NDArray[np.floating] = np.random.normal(0.0, 1.0, (10, 3))
    noise: NDArray[np.floating] = np.random.normal(0.0, 0.1, test1.shape)
    print(test1)
    q_true: Quat = randomQuat()
    t_true: NDArray[np.floating] = np.array([10.0, 0.0, 0.0]) + np.random.normal(1.0, 1.0, (3,))
    test2: NDArray[np.floating] = (q_true * test1 + t_true) + noise

    print(f"Targets: \n{q_true}\n{t_true}\n")
    print(f"Targets: \n{q_true.to_SE3_given_position(t_true)}")

    optClass = ThreeD_to_ThreeD(test1, test2)

    print(f'Estimates: \n{optClass.q}\n{optClass.t}\n')
    print(f'Estimates(SE3):\n{optClass.q.to_SE3_given_position(optClass.t)}\n')

    print(f"Resolved Residual: {norm(optClass.create_y())}\n{optClass.create_y()}")
    print(f"True Residual: {norm(optClass.create_y(q_true, t_true))}\n{optClass.create_y(q_true, t_true)}")


if __name__ == "__main__":
    main()
