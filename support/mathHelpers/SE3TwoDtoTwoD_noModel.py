import numpy as np
from dataclasses import dataclass, field

import support.mathHelpers.quaternions as q
from support.mathHelpers.SE3 import SE3_q
from support.vision.calibration import Calibration, undistort_points_px


_EPS = np.finfo(np.float64).eps


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= _EPS):
        raise ValueError("All bearing vectors must have non-zero norm.")
    return vectors / norms


def _pixels_to_bearings(pixels: np.ndarray, cal: Calibration, *, mode: str) -> np.ndarray:
    undistorted_px = undistort_points_px(cal, np.asarray(pixels, dtype=np.float64), mode=mode)
    norm_xy = np.empty_like(undistorted_px, dtype=np.float64)
    norm_xy[:, 0] = (undistorted_px[:, 0] - cal.cx) / cal.fx
    norm_xy[:, 1] = (undistorted_px[:, 1] - cal.cy) / cal.fy
    return _normalize_rows(np.column_stack((norm_xy, np.ones(len(norm_xy), dtype=np.float64))))


def _unit_translation(tvec: np.ndarray) -> np.ndarray:
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(tvec)
    if norm <= _EPS:
        raise ValueError("Translation vector must be non-zero for 2D-to-2D epipolar optimization.")
    return tvec / norm


@dataclass
class SE3TwoDtoTwoD_noModel:
    observed_pixels_cam1: np.ndarray
    observed_pixels_cam2: np.ndarray
    cam1_cal: Calibration
    cam2_cal: Calibration
    undistort_mode: str = "precise"
    finite_difference_step: float = 1.0e-6
    _bearings_cam1: np.ndarray = field(init=False, repr=False)
    _bearings_cam2: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.observed_pixels_cam1 = np.asarray(self.observed_pixels_cam1, dtype=np.float64)
        self.observed_pixels_cam2 = np.asarray(self.observed_pixels_cam2, dtype=np.float64)

        if self.observed_pixels_cam1.ndim != 2 or self.observed_pixels_cam1.shape[1] != 2:
            raise ValueError("observed_pixels_cam1 must be shaped (N, 2)")
        if self.observed_pixels_cam2.ndim != 2 or self.observed_pixels_cam2.shape[1] != 2:
            raise ValueError("observed_pixels_cam2 must be shaped (N, 2)")
        if self.observed_pixels_cam1.shape[0] != self.observed_pixels_cam2.shape[0]:
            raise ValueError("observed_pixels_cam1 and observed_pixels_cam2 must have the same length")
        if self.observed_pixels_cam1.shape[0] < 5:
            raise ValueError("At least 5 correspondences are required for 2D-to-2D pose estimation.")

        self._bearings_cam1 = _pixels_to_bearings(
            self.observed_pixels_cam1,
            self.cam1_cal,
            mode=self.undistort_mode,
        )
        self._bearings_cam2 = _pixels_to_bearings(
            self.observed_pixels_cam2,
            self.cam2_cal,
            mode=self.undistort_mode,
        )

    def residual(self, state: SE3_q) -> np.ndarray:
        try:
            t_hat = _unit_translation(state.tvec)
        except ValueError:
            return np.full((len(self._bearings_cam1),), 1.0e9, dtype=np.float64)

        R = state.quat.to_dcm()
        essential = q.skew(t_hat) @ R

        Ex1 = (essential @ self._bearings_cam1.T).T
        Etx2 = (essential.T @ self._bearings_cam2.T).T
        numerators = np.einsum("ij,ij->i", self._bearings_cam2, Ex1)
        denominators = (
            Ex1[:, 0] * Ex1[:, 0]
            + Ex1[:, 1] * Ex1[:, 1]
            + Etx2[:, 0] * Etx2[:, 0]
            + Etx2[:, 1] * Etx2[:, 1]
        )

        invalid = denominators <= _EPS

        residuals = np.empty((len(self._bearings_cam1),), dtype=np.float64)
        residuals[~invalid] = numerators[~invalid] / np.sqrt(denominators[~invalid])
        residuals[invalid] = 1.0e9
        return residuals

    def jacobian(self, state: SE3_q) -> np.ndarray:
        step = float(self.finite_difference_step)
        if step <= 0.0:
            raise ValueError("finite_difference_step must be positive")

        residual_0 = self.residual(state)
        J = np.zeros((residual_0.size, 6), dtype=np.float64)

        for col in range(6):
            dx = np.zeros((6,), dtype=np.float64)
            dx[col] = step
            residual_plus = self.residual(self.retract(state, dx))
            residual_minus = self.residual(self.retract(state, -dx))
            J[:, col] = (residual_plus - residual_minus) / (2.0 * step)

        return J

    @staticmethod
    def retract(state: SE3_q, dx: np.ndarray) -> SE3_q:
        dx = np.asarray(dx, dtype=np.float64).reshape(6)

        perturb = SE3_q(
            quat=q.Quaternion.exp_so3(dx[:3]),
            tvec=dx[3:],
        )
        new_state = perturb * state

        t_norm = np.linalg.norm(new_state.tvec)
        if t_norm <= _EPS:
            new_state.tvec = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            new_state.tvec = new_state.tvec / t_norm

        new_state.quat.force_s_pos()
        return new_state
