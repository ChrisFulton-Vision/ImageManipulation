import numpy as np
from dataclasses import dataclass
from support.mathHelpers.SE3 import SE3_q
import support.mathHelpers.quaternions as q

@dataclass
class SE3PointAlignmentProblem:
    body_points: np.ndarray
    measured_points: np.ndarray

    def residual(self, state: "SE3_q") -> np.ndarray:
        predicted_points = state * self.body_points
        return (predicted_points - self.measured_points).reshape(-1)

    def jacobian(self, state: "SE3_q") -> np.ndarray:
        predicted_points = state * self.body_points
        num_points = predicted_points.shape[0]

        L = np.zeros((3 * num_points, 6), dtype=np.float64)

        for point_idx, predicted_point in enumerate(predicted_points):
            row = 3 * point_idx

            # Left perturbation:
            #
            #   T_new = Exp(dx) T
            #
            # Therefore:
            #
            #   p_new ~= p + dtheta x p + dt
            #         = p - [p]_x dtheta + dt
            #
            L[row:row + 3, 0:3] = -q.skew(predicted_point)
            L[row:row + 3, 3:6] = np.eye(3)

        return L

    @staticmethod
    def retract(state: "SE3_q", dx: np.ndarray) -> "SE3_q":
        dx = np.asarray(dx, dtype=np.float64).reshape(6)

        dtheta = dx[0:3]
        dt = dx[3:6]

        perturb = SE3_q(
            quat=q.Quaternion.exp_so3(dtheta),
            tvec=dt,
        )

        # Left SE(3) perturbation:
        #
        #   T_new = Exp(dx) * T
        #
        new_state = perturb * state
        new_state.quat.force_s_pos()
        return new_state