import numpy as np
from dataclasses import dataclass

import support.mathHelpers.quaternions as q
from support.mathHelpers.SE3 import SE3_q
from support.vision.calibration import Calibration


@dataclass
class SE3ThreeDtoTwoD_noModel:
    points_cam1: np.ndarray
    observed_pixels_cam2: np.ndarray
    cam2_cal: Calibration

    def __post_init__(self) -> None:
        self.points_cam1 = np.asarray(self.points_cam1, dtype=np.float64)
        self.observed_pixels_cam2 = np.asarray(self.observed_pixels_cam2, dtype=np.float64)

        if self.points_cam1.ndim != 2 or self.points_cam1.shape[1] != 3:
            raise ValueError("points_cam1 must be shaped (N, 3)")
        if self.observed_pixels_cam2.ndim != 2 or self.observed_pixels_cam2.shape[1] != 2:
            raise ValueError("observed_pixels_cam2 must be shaped (N, 2)")
        if self.points_cam1.shape[0] != self.observed_pixels_cam2.shape[0]:
            raise ValueError("points_cam1 and observed_pixels_cam2 must have the same length")

    def residual(self, state: SE3_q) -> np.ndarray:
        predicted_points_cam2 = state * self.points_cam1
        z = predicted_points_cam2[:, 2]

        if np.any(z <= 0):
            return np.full((2 * len(self.points_cam1),), 1.0e9, dtype=np.float64)

        predicted_pixels = np.empty((len(self.points_cam1), 2), dtype=np.float64)
        predicted_pixels[:, 0] = self.cam2_cal.fx * predicted_points_cam2[:, 0] / z + self.cam2_cal.cx
        predicted_pixels[:, 1] = self.cam2_cal.fy * predicted_points_cam2[:, 1] / z + self.cam2_cal.cy
        return (predicted_pixels - self.observed_pixels_cam2).reshape(-1)

    def jacobian(self, state: SE3_q) -> np.ndarray:
        predicted_points_cam2 = state * self.points_cam1
        num_points = predicted_points_cam2.shape[0]
        J = np.zeros((2 * num_points, 6), dtype=np.float64)

        for idx, point_cam2 in enumerate(predicted_points_cam2):
            x, y, z = point_cam2
            if z <= 0:
                continue

            projection_jacobian = np.array([
                [self.cam2_cal.fx / z, 0.0, -self.cam2_cal.fx * x / (z * z)],
                [0.0, self.cam2_cal.fy / z, -self.cam2_cal.fy * y / (z * z)],
            ], dtype=np.float64)

            point_jacobian = np.hstack((-q.skew(point_cam2), np.eye(3, dtype=np.float64)))
            J[2 * idx:2 * idx + 2, :] = projection_jacobian @ point_jacobian

        return J

    @staticmethod
    def retract(state: SE3_q, dx: np.ndarray) -> SE3_q:
        dx = np.asarray(dx, dtype=np.float64).reshape(6)

        perturb = SE3_q(
            quat=q.Quaternion.exp_so3(dx[:3]),
            tvec=dx[3:],
        )
        new_state = perturb * state
        new_state.quat.force_s_pos()
        return new_state
