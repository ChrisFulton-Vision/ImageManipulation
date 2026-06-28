import numpy as np
from dataclasses import dataclass
from typing import overload

@dataclass
class SE2:
    """A tiny SE(2) class for 2D rigid transforms.

    The transform maps model points into world points as

        p_world = R(theta_rad) @ p_model + t.

    Composition is left-to-right in the same style as the 3D demo:

        world_point = pose * model_point
        combined_pose = left_pose * right_pose
    """

    theta_rad: float
    tvec: np.ndarray

    def __post_init__(self) -> None:
        self.theta_rad = float(wrap_angle_signed(self.theta_rad))
        self.tvec = np.asarray(self.tvec, dtype=np.float64).reshape(2)

    @property
    def R(self) -> np.ndarray:
        c = np.cos(self.theta_rad)
        s = np.sin(self.theta_rad)
        return np.array([[c, -s],
                         [s, c]], dtype=np.float64)

    @classmethod
    def identity(cls) -> "SE2":
        return cls(0.0, np.zeros(2, dtype=np.float64))

    @classmethod
    def random(cls, max_translation: float = 5.0, max_angle_deg: float = 180.0) -> "SE2":
        theta_rad = float(np.deg2rad(np.random.uniform(-max_angle_deg, max_angle_deg)))
        tvec = np.asarray(np.random.uniform(-max_translation, max_translation, size=2))
        return cls(theta_rad, tvec)

    @classmethod
    def exp(cls, dx: np.ndarray) -> "SE2":
        """SE(2) exponential map for a left perturbation dx = [dtheta, vx, vy]."""
        dx = np.asarray(dx, dtype=np.float64).reshape(3)
        theta_rad = float(dx[0])
        v = dx[1:3]

        if abs(theta_rad) < 1.0e-9:
            V = np.eye(2, dtype=np.float64)
        else:
            sin_theta = np.sin(theta_rad)
            cos_theta = np.cos(theta_rad)
            A = sin_theta / theta_rad
            B = (1.0 - cos_theta) / theta_rad
            V = np.array([[A, -B],
                          [B, A]], dtype=np.float64)

        return cls(theta_rad, V @ v)

    def copy(self) -> "SE2":
        return SE2(self.theta_rad, self.tvec.copy())

    def inverse(self) -> "SE2":
        R_T = self.R.T
        return SE2(-self.theta_rad, -(R_T @ self.tvec))

    @property
    def inv(self) -> "SE2":
        """Return the inverse transform. Named to match the SE(3) demo style."""
        return self.inverse()

    def interpolate(self, other: "SE2", alpha: float) -> "SE2":
        alpha = float(np.clip(alpha, 0.0, 1.0))
        dtheta_rad = wrap_angle_signed(other.theta_rad - self.theta_rad)
        theta_rad = wrap_angle_signed(self.theta_rad + alpha * dtheta_rad)
        tvec = (1.0 - alpha) * self.tvec + alpha * other.tvec
        return SE2(theta_rad, tvec)

    @overload
    def __mul__(self, other: "SE2") -> "SE2":
        ...

    @overload
    def __mul__(self, other: np.ndarray) -> np.ndarray:
        ...

    def __mul__(self, other):
        if isinstance(other, SE2):
            theta_rad = wrap_angle_signed(self.theta_rad + other.theta_rad)
            tvec = self.R @ other.tvec + self.tvec
            return SE2(theta_rad, tvec)

        points = np.asarray(other, dtype=np.float64)
        if points.shape == (2,):
            return self.R @ points + self.tvec
        if points.ndim == 2 and points.shape[1] == 2:
            return points @ self.R.T + self.tvec
        raise TypeError(f"SE2 can transform a 2-vector, an Nx2 array, or compose with SE2; got shape {points.shape}")

    def __repr__(self) -> str:
        return f"SE2(theta_rad_deg={np.rad2deg(self.theta_rad): .3f}, tvec=[{self.tvec[0]: .3f}, {self.tvec[1]: .3f}])"

def wrap_angle_signed(theta_rad: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (theta_rad + np.pi) % (2.0 * np.pi) - np.pi