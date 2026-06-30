from typing import Protocol, TypeVar, Generic, Literal
import numpy as np
from dataclasses import dataclass
from copy import deepcopy

StateT = TypeVar("StateT")


class LeastSquaresProblem(Protocol[StateT]):
    def residual(self, state: StateT) -> np.ndarray:
        ...

    def jacobian(self, state: StateT) -> np.ndarray:
        ...

    def retract(self, state: StateT, dx: np.ndarray) -> StateT:
        ...


@dataclass
class LevenbergMarquardt(Generic[StateT]):
    state: StateT
    problem: LeastSquaresProblem[StateT]
    robust_loss: Literal["none", "huber", "cauchy", "tukey"] = "none"
    robust_scale: float = 1.0
    damping_enabled: bool = False
    damping: float = 1e-1
    adaptive: bool = True
    damping_up: float = 10.0
    damping_down: float = 0.3
    min_damping: float = 1e-10
    use_diagonal_damping: bool = True
    tolerance: float = 1e-9
    max_steps: int = 10
    max_iter: int = 10
    accept_rho_min: float = 1.0e-3
    good_rho_min: float = 0.75
    bad_rho_max: float = 0.25
    numerical_check: bool = False
    store_y_mags: bool = True
    store_states: bool = False

    def __post_init__(self):
        if self.robust_scale <= 0.0:
            raise ValueError(f"robust_scale must be positive, got {self.robust_scale}")

        self.idx: int = 0
        self.reject_count: int = 0
        self.converged: bool = False
        self.last_step_norm: float = np.inf
        self.final_cost: float = np.inf

        self.y_mag_hist = []
        self.states_hist = []

        if self.numerical_check:
            self.check_jacobian()
        self.state = self.optimize()

    def check_jacobian(
            self,
            eps: float = 1e-6,
            atol: float = 1e-6,
            rtol: float = 1e-4,
            verbose: bool = True,
    ) -> bool:
        create_y = self.problem.residual
        create_L = self.problem.jacobian
        update = self.problem.retract

        x = self.state

        base_y = np.asarray(create_y(x), dtype=np.float64).reshape(-1)
        base_L = np.asarray(create_L(x), dtype=np.float64)

        if base_L.ndim != 2:
            raise ValueError(f"Jacobian must be 2D, got shape {base_L.shape}")

        residual_dim, tangent_dim = base_L.shape

        if base_y.shape[0] != residual_dim:
            raise ValueError(
                "Residual/Jacobian shape mismatch: "
                f"residual has length {base_y.shape[0]}, "
                f"Jacobian has {residual_dim} rows."
            )

        numerical_L = np.zeros_like(base_L, dtype=np.float64)

        for col_idx in range(tangent_dim):
            dx_plus = np.zeros(tangent_dim, dtype=np.float64)
            dx_minus = np.zeros(tangent_dim, dtype=np.float64)

            dx_plus[col_idx] = eps
            dx_minus[col_idx] = -eps

            y_plus = np.asarray(create_y(update(x, dx_plus)), dtype=np.float64).reshape(-1)
            y_minus = np.asarray(create_y(update(x, dx_minus)), dtype=np.float64).reshape(-1)

            numerical_L[:, col_idx] = (y_plus - y_minus) / (2.0 * eps)

        error = numerical_L - base_L

        col_abs_errors = np.linalg.norm(error, axis=0)
        col_scales = np.maximum(
            np.linalg.norm(base_L, axis=0),
            np.linalg.norm(numerical_L, axis=0),
        )
        col_rel_errors = col_abs_errors / np.maximum(col_scales, 1.0)

        max_abs_error = float(np.max(np.abs(error)))
        max_rel_error = float(np.max(col_rel_errors))

        passed = bool(
            np.all(col_abs_errors <= atol + rtol * np.maximum(col_scales, 1.0))
        )

        if verbose:
            print("=" * 80)
            print("Jacobian numerical check")
            print("=" * 80)
            print(f"Residual dimension: {residual_dim}")
            print(f"Tangent dimension:  {tangent_dim}")
            print(f"Step size eps:      {eps:.3e}")
            print(f"Max abs error:      {max_abs_error:.3e}")
            print(f"Max rel error:      {max_rel_error:.3e}")
            print(f"Passed:             {passed}")
            print()

            for col_idx in range(tangent_dim):
                print(
                    f"Column {col_idx:02d}: "
                    f"abs error = {col_abs_errors[col_idx]:.3e}, "
                    f"rel error = {col_rel_errors[col_idx]:.3e}"
                )

            if not passed:
                worst_col = int(np.argmax(col_rel_errors))
                print()
                print(f"Worst column: {worst_col}")
                print("Analytic column:")
                print(base_L[:, worst_col])
                print("Numerical column:")
                print(numerical_L[:, worst_col])
                print("Difference:")
                print(error[:, worst_col])

        print("=" * 80)
        return passed

    @property
    def _is_done(self):
        return self.reject_count > self.max_iter

    def _copy_state(self, state: StateT) -> StateT:
        copy_method = getattr(state, "copy", None)
        if callable(copy_method):
            return copy_method()
        return deepcopy(state)

    @staticmethod
    def _y_mag_from_y(y: np.ndarray) -> float:
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        return float(np.linalg.norm(y))

    def _robust_weight_sqrt_and_cost(self, y: np.ndarray) -> tuple[np.ndarray, float]:
        y = np.asarray(y, dtype=np.float64).reshape(-1)

        if self.robust_loss == "none":
            return np.ones_like(y), 0.5 * float(y.T @ y)

        abs_y = np.abs(y)
        scale = self.robust_scale

        if self.robust_loss == "huber":
            weights = np.ones_like(y)
            mask = abs_y > scale
            weights[mask] = scale / abs_y[mask]

            cost_terms = np.empty_like(y)
            cost_terms[~mask] = 0.5 * y[~mask] ** 2
            cost_terms[mask] = scale * (abs_y[mask] - 0.5 * scale)
            return np.sqrt(weights), float(np.sum(cost_terms))

        scaled_sq = (y / scale) ** 2

        if self.robust_loss == "cauchy":
            weights = 1.0 / (1.0 + scaled_sq)
            cost = 0.5 * (scale ** 2) * float(np.sum(np.log1p(scaled_sq)))
            return np.sqrt(weights), cost

        if self.robust_loss == "tukey":
            inside = scaled_sq < 1.0
            weights = np.zeros_like(y)
            weights[inside] = (1.0 - scaled_sq[inside]) ** 2

            cost_terms = np.full_like(y, (scale ** 2) / 6.0)
            cost_terms[inside] = (scale ** 2 / 6.0) * (1.0 - (1.0 - scaled_sq[inside]) ** 3)
            return np.sqrt(weights), float(np.sum(cost_terms))

        raise ValueError(f"Unsupported robust_loss: {self.robust_loss}")

    def _weighted_system(self, L: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        L = np.asarray(L, dtype=np.float64)

        sqrt_w, cost = self._robust_weight_sqrt_and_cost(y)
        weighted_y = sqrt_w * y
        weighted_L = sqrt_w[:, None] * L
        return weighted_L, weighted_y, cost

    def _record_history(self, state: StateT, y: np.ndarray | None = None) -> None:
        if self.store_y_mags:
            if y is None:
                y = self.problem.residual(state)
            _, weighted_y, _ = self._weighted_system(self.problem.jacobian(state), y)
            self.y_mag_hist.append(self._y_mag_from_y(weighted_y))

        if self.store_states:
            self.states_hist.append(self._copy_state(state))

    def _damp_down(self):
        self.damping *= self.damping_down
        if self.damping < self.min_damping:
            self.damping = self.min_damping
        self.reject_count = 0

    def _damp_up(self):
        self.damping *= self.damping_up
        self.reject_count += 1

    def optimize(self):
        create_y = self.problem.residual
        create_L = self.problem.jacobian
        update = self.problem.retract

        x = self._copy_state(self.state)

        def form_trial_dx(L, y):
            H = L.T @ L
            g = L.T @ y

            if self.damping_enabled:
                if self.use_diagonal_damping:
                    D = np.diag(np.diag(H))
                    diag = np.diag(D).copy()
                    diag[diag == 0.0] = 1.0
                    D = np.diag(diag)
                else:
                    D = np.eye(H.shape[0], dtype=np.float64)

                H += self.damping * D
                dx = -np.linalg.solve(H, g)
                pred_improve = 0.5 * float(dx.T @ (self.damping * D @ dx - g))

                return dx, pred_improve

            dx = -np.linalg.solve(H, g)
            pred_improve = -0.5 * float(dx.T @ g)
            return dx, pred_improve

        initial_y = create_y(x)
        self._record_history(x, initial_y)

        converged = False

        for step_idx in range(self.max_steps):
            orig_step_y = create_y(x)
            raw_L = create_L(x)
            L, y, orig_step_cost = self._weighted_system(raw_L, orig_step_y)

            keep_lm_going = True

            while keep_lm_going:
                dx, pred_improve = form_trial_dx(L, y)
                self.last_step_norm = float(np.linalg.norm(dx))

                if self.last_step_norm < self.tolerance:
                    converged = True
                    keep_lm_going = False
                    break

                trial_x = update(x, dx)
                pert_y = create_y(trial_x)
                _, _, pert_cost = self._weighted_system(create_L(trial_x), pert_y)

                actual_improve = orig_step_cost - pert_cost
                rho = actual_improve / pred_improve if pred_improve > 0.0 else -np.inf

                if rho > self.accept_rho_min or self._is_done:
                    x = trial_x

                    if rho >= self.good_rho_min:
                        self._damp_down()
                    elif rho < self.bad_rho_max:
                        self._damp_up()
                    else:
                        self.reject_count = 0

                    self.idx += 1
                    self._record_history(x, pert_y)
                    self.final_cost = pert_cost

                    if abs(actual_improve) < self.tolerance:
                        converged = True

                    keep_lm_going = False

                else:
                    self._damp_up()

            if converged:
                break

        if not np.isfinite(self.final_cost):
            final_y = create_y(x)
            _, _, self.final_cost = self._weighted_system(create_L(x), final_y)

        self.converged = converged
        return x


def main() -> None:
    from support.mathHelpers.SE3 import SE3_q
    from support.mathHelpers.quaternions import Quaternion, skew
    from support.mathHelpers.SE3PointAlignmentProblem import SE3PointAlignmentProblem

    from support.io.my_logging import LOG

    np.set_printoptions(precision=6, suppress=True)

    body_points = np.array([
        [-1.0, -1.0, -0.5],
        [ 1.0, -1.0, -0.5],
        [ 1.0,  1.0, -0.5],
        [-1.0,  1.0, -0.5],
        [-0.8, -0.6,  0.7],
        [ 0.9, -0.5,  0.8],
        [ 0.7,  0.8,  0.9],
        [-0.6,  0.7,  0.6],
        [ 0.2, -0.1,  1.4],
    ], dtype=np.float64)

    true_state = SE3_q(
        quat=Quaternion.exp_so3(np.deg2rad(np.array([18.0, -11.0, 27.0]))),
        tvec=np.array([1.2, -0.7, 2.4], dtype=np.float64),
    )

    initial_state = SE3_q(
        quat=Quaternion.exp_so3(np.deg2rad(np.array([-22.0, 16.0, -18.0]))),
        tvec=np.array([-1.0, 0.9, 0.4], dtype=np.float64),
    )

    np.random.seed(42)
    measured_points = true_state * body_points + 0.05 * np.random.normal(size=body_points.shape)

    problem = SE3PointAlignmentProblem(
        body_points=body_points,
        measured_points=measured_points,
    )

    initial_residual = problem.residual(initial_state)

    LOG.info(f"Initial residual norm: {np.linalg.norm(initial_residual)}")
    LOG.info(f"Initial rotation error [deg]: {initial_state.quat.angle_betweenD(true_state.quat)}")
    LOG.info(f"Initial translation error: {np.linalg.norm(initial_state.tvec - true_state.tvec)}\n")

    solver = LevenbergMarquardt(
        state=initial_state,
        problem=problem,
        damping_enabled=True,
        damping=1e-2,
        use_diagonal_damping=True,
        numerical_check=True,
        max_steps=30,
        max_iter=20,
        tolerance=1e-12,
        store_y_mags=True,
        store_states=True,
    )

    final_state = solver.state
    final_residual = problem.residual(final_state)

    LOG.info(f"\nFinal residual norm: {np.linalg.norm(final_residual)}")
    LOG.info(f"Final rotation error [deg]: {final_state.quat.angle_betweenD(true_state.quat)}")
    LOG.info(f"Final translation error: {np.linalg.norm(final_state.tvec - true_state.tvec)}\n")

    LOG.info(f"True q: {true_state.quat}")
    LOG.info(f"Final q: {final_state.quat}")
    LOG.info(f"True t: {true_state.tvec}")
    LOG.info(f"Final t: {final_state.tvec}\n")

    LOG.info("Residual history:")
    LOG.info(solver.y_mag_hist)

    LOG.info("\nNumber of stored states:")
    LOG.info(len(solver.states_hist))
    for state in solver.states_hist:
        LOG.info(state)


if __name__ == "__main__":
    main()
