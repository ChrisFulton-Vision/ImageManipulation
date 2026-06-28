import numpy as np
from dataclasses import dataclass
from support.mathHelpers.SE2 import SE2

@dataclass
class SE2PointAlignmentProblem:
    model_verts: np.ndarray
    measured_verts: np.ndarray

    def __post_init__(self) -> None:
        if self.model_verts.shape != self.measured_verts.shape:
            raise ValueError(
                "model_verts and measured_verts must have the same shape"
            )
        if self.model_verts.ndim != 2 or self.model_verts.shape[1] != 2:
            raise ValueError(
                "model_verts and measured_verts must be shaped (N, 2)"
            )

    def residual(self, state: SE2) -> np.ndarray:
        estimated_verts = state * self.model_verts
        return (self.measured_verts - estimated_verts).reshape(-1)

    def jacobian(self, state: SE2) -> np.ndarray:
        estimated_verts = state * self.model_verts

        num_elements = int(np.prod(self.measured_verts.shape))
        num_states = 3
        L = np.zeros((num_elements, num_states), dtype=np.float64)

        for idx, p_world in enumerate(estimated_verts):
            px, py = p_world

            # Residual:
            #
            #   y_i = measured_i - estimated_i
            #
            # Left perturbation:
            #
            #   T_new = Exp(dx) T
            #
            # Point update:
            #
            #   p_new ~= p + dtheta * J p + dt
            #
            # Therefore residual update:
            #
            #   y_new ~= y - dtheta * J p - dt
            #
            # With J p = [-py, px]^T:
            #
            #   d y / d dtheta = [py, -px]^T
            #   d y / d dt     = -I
            #
            L[2 * idx:2 * idx + 2, 0] = [py, -px]
            L[2 * idx:2 * idx + 2, 1:3] = -np.eye(2)

        return L

    @staticmethod
    def retract(state: SE2, dx: np.ndarray) -> SE2:
        return SE2.exp(dx) * state
