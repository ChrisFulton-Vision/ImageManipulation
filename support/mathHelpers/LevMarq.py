from typing import Protocol, TypeVar, Generic, Literal
import numpy as np
from dataclasses import dataclass
from copy import deepcopy
from time import perf_counter

from support.mathHelpers.Linalg import call_backend as linalg_call_backend
from support.mathHelpers.Linalg import NUMBA_AVAILABLE as LINALG_NUMBA_AVAILABLE

StateT = TypeVar("StateT")
LinearSolverBackend = Literal["numpy", "linalg_numpy", "linalg_numba", "linalg_spd_numpy", "linalg_spd_numba"]


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
    linear_solver_backend: LinearSolverBackend = "linalg_spd_numpy"

    def __post_init__(self):
        if self.robust_scale <= 0.0:
            raise ValueError(f"robust_scale must be positive, got {self.robust_scale}")
        if self.linear_solver_backend in {"linalg_numba", "linalg_spd_numba"} and not LINALG_NUMBA_AVAILABLE:
            raise ValueError(
                f"linear_solver_backend={self.linear_solver_backend!r} requires numba-backed Linalg support."
            )

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

    def _solve_step_system(self, h: np.ndarray, g: np.ndarray) -> np.ndarray:
        rhs = -np.asarray(g, dtype=np.float64).reshape(-1, 1)

        if self.linear_solver_backend == "numpy":
            return np.linalg.solve(h, rhs).reshape(-1)

        if self.linear_solver_backend == "linalg_numpy":
            return np.asarray(linalg_call_backend("solve", "numpy", h, rhs)).reshape(-1)

        if self.linear_solver_backend == "linalg_numba":
            return np.asarray(linalg_call_backend("solve", "numba", h, rhs)).reshape(-1)

        if self.linear_solver_backend == "linalg_spd_numpy":
            return np.asarray(linalg_call_backend("solve_spd", "numpy", h, rhs)).reshape(-1)

        if self.linear_solver_backend == "linalg_spd_numba":
            return np.asarray(linalg_call_backend("solve_spd", "numba", h, rhs)).reshape(-1)

        raise ValueError(f"Unknown linear_solver_backend {self.linear_solver_backend!r}")

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
                dx = self._solve_step_system(H, g)
                pred_improve = 0.5 * float(dx.T @ (self.damping * D @ dx - g))

                return dx, pred_improve

            dx = self._solve_step_system(H, g)
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


@dataclass
class LinearRegressionProblem:
    x_data: np.ndarray
    y_data: np.ndarray

    def residual(self, state: np.ndarray) -> np.ndarray:
        slope, intercept = np.asarray(state, dtype=np.float64).reshape(2)
        return slope * self.x_data + intercept - self.y_data

    def jacobian(self, state: np.ndarray) -> np.ndarray:
        _ = state
        return np.column_stack((self.x_data, np.ones_like(self.x_data)))

    @staticmethod
    def retract(state: np.ndarray, dx: np.ndarray) -> np.ndarray:
        return np.asarray(state, dtype=np.float64).reshape(2) + np.asarray(dx, dtype=np.float64).reshape(2)


@dataclass
class ExponentialFitProblem:
    x_data: np.ndarray
    y_data: np.ndarray

    def residual(self, state: np.ndarray) -> np.ndarray:
        amplitude, decay, bias = np.asarray(state, dtype=np.float64).reshape(3)
        model = amplitude * np.exp(decay * self.x_data) + bias
        return model - self.y_data

    def jacobian(self, state: np.ndarray) -> np.ndarray:
        amplitude, decay, _bias = np.asarray(state, dtype=np.float64).reshape(3)
        exp_term = np.exp(decay * self.x_data)
        return np.column_stack((exp_term, amplitude * self.x_data * exp_term, np.ones_like(self.x_data)))

    @staticmethod
    def retract(state: np.ndarray, dx: np.ndarray) -> np.ndarray:
        return np.asarray(state, dtype=np.float64).reshape(3) + np.asarray(dx, dtype=np.float64).reshape(3)


@dataclass
class CircleFitProblem:
    points: np.ndarray

    def residual(self, state: np.ndarray) -> np.ndarray:
        cx, cy, radius = np.asarray(state, dtype=np.float64).reshape(3)
        dx = self.points[:, 0] - cx
        dy = self.points[:, 1] - cy
        return np.sqrt(dx * dx + dy * dy) - radius

    def jacobian(self, state: np.ndarray) -> np.ndarray:
        cx, cy, _radius = np.asarray(state, dtype=np.float64).reshape(3)
        dx = self.points[:, 0] - cx
        dy = self.points[:, 1] - cy
        dist = np.sqrt(dx * dx + dy * dy)
        dist = np.maximum(dist, 1e-12)
        return np.column_stack((-dx / dist, -dy / dist, -np.ones_like(dist)))

    @staticmethod
    def retract(state: np.ndarray, dx: np.ndarray) -> np.ndarray:
        return np.asarray(state, dtype=np.float64).reshape(3) + np.asarray(dx, dtype=np.float64).reshape(3)


@dataclass
class DenseNonlinearLeastSquaresProblem:
    design_matrix: np.ndarray
    coupling_matrix: np.ndarray
    observations: np.ndarray
    nonlinearity_scale: float = 0.05

    def residual(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64).reshape(-1)
        linear_term = self.design_matrix @ state
        coupled_arg = self.coupling_matrix @ state
        nonlinear_term = self.nonlinearity_scale * np.sin(coupled_arg)
        return linear_term + nonlinear_term - self.observations

    def jacobian(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64).reshape(-1)
        coupled_arg = self.coupling_matrix @ state
        row_scale = self.nonlinearity_scale * np.cos(coupled_arg)
        return self.design_matrix + row_scale[:, None] * self.coupling_matrix

    @staticmethod
    def retract(state: np.ndarray, dx: np.ndarray) -> np.ndarray:
        return np.asarray(state, dtype=np.float64).reshape(-1) + np.asarray(dx, dtype=np.float64).reshape(-1)


@dataclass
class BenchmarkCase:
    name: str
    initial_state: StateT
    problem: LeastSquaresProblem[StateT]
    solver_kwargs: dict
    error_fn: callable
    repeats: int = 10


def _copy_benchmark_state(state):
    copy_method = getattr(state, "copy", None)
    if callable(copy_method):
        return copy_method()
    return deepcopy(state)


def make_benchmark_cases():
    from support.mathHelpers.SE3 import SE3_q
    from support.mathHelpers.quaternions import Quaternion
    from support.mathHelpers.SE3PointAlignmentProblem import SE3PointAlignmentProblem
    rng = np.random.default_rng(42)

    x_line = np.linspace(-3.0, 3.0, 200)
    true_line = np.array([2.5, -0.8], dtype=np.float64)
    y_line = true_line[0] * x_line + true_line[1] + 0.05 * rng.standard_normal(x_line.shape[0])
    line_problem = LinearRegressionProblem(x_data=x_line, y_data=y_line)
    line_initial = np.array([-1.5, 2.0], dtype=np.float64)

    x_exp = np.linspace(0.0, 2.0, 160)
    true_exp = np.array([1.8, -1.4, 0.35], dtype=np.float64)
    y_exp = true_exp[0] * np.exp(true_exp[1] * x_exp) + true_exp[2] + 0.02 * rng.standard_normal(x_exp.shape[0])
    exp_problem = ExponentialFitProblem(x_data=x_exp, y_data=y_exp)
    exp_initial = np.array([0.8, -0.2, 0.0], dtype=np.float64)

    angles = np.linspace(0.0, 2.0 * np.pi, 180, endpoint=False)
    true_circle = np.array([1.25, -0.65, 2.1], dtype=np.float64)
    circle_points = np.column_stack(
        (
            true_circle[0] + true_circle[2] * np.cos(angles),
            true_circle[1] + true_circle[2] * np.sin(angles),
        )
    )
    circle_points += 0.03 * rng.standard_normal(circle_points.shape)
    circle_problem = CircleFitProblem(points=circle_points)
    circle_initial = np.array([0.3, 0.2, 1.3], dtype=np.float64)

    body_points = np.array(
        [
            [-1.0, -1.0, -0.5],
            [1.0, -1.0, -0.5],
            [1.0, 1.0, -0.5],
            [-1.0, 1.0, -0.5],
            [-0.8, -0.6, 0.7],
            [0.9, -0.5, 0.8],
            [0.7, 0.8, 0.9],
            [-0.6, 0.7, 0.6],
            [0.2, -0.1, 1.4],
        ],
        dtype=np.float64,
    )
    true_se3 = SE3_q(
        quat=Quaternion.exp_so3(np.deg2rad(np.array([18.0, -11.0, 27.0]))),
        tvec=np.array([1.2, -0.7, 2.4], dtype=np.float64),
    )
    initial_se3 = SE3_q(
        quat=Quaternion.exp_so3(np.deg2rad(np.array([-22.0, 16.0, -18.0]))),
        tvec=np.array([-1.0, 0.9, 0.4], dtype=np.float64),
    )
    measured_points = true_se3 * body_points + 0.05 * rng.standard_normal(size=body_points.shape)
    se3_problem = SE3PointAlignmentProblem(body_points=body_points, measured_points=measured_points)

    medium_param_dim = 48
    medium_residual_dim = 2400
    medium_design = rng.standard_normal((medium_residual_dim, medium_param_dim))
    medium_design += 0.25 * rng.standard_normal((medium_residual_dim, 1)) @ np.ones((1, medium_param_dim))
    medium_coupling = 0.15 * rng.standard_normal((medium_residual_dim, medium_param_dim))
    medium_true = rng.standard_normal(medium_param_dim) * 0.2
    medium_obs = (
        medium_design @ medium_true
        + 0.05 * np.sin(medium_coupling @ medium_true)
        + 0.01 * rng.standard_normal(medium_residual_dim)
    )
    medium_initial = medium_true + 0.15 * rng.standard_normal(medium_param_dim)
    medium_problem = DenseNonlinearLeastSquaresProblem(
        design_matrix=medium_design,
        coupling_matrix=medium_coupling,
        observations=medium_obs,
        nonlinearity_scale=0.05,
    )

    large_param_dim = 96
    large_residual_dim = 6000
    large_design = rng.standard_normal((large_residual_dim, large_param_dim))
    large_design += 0.2 * rng.standard_normal((large_residual_dim, 1)) @ np.ones((1, large_param_dim))
    large_coupling = 0.12 * rng.standard_normal((large_residual_dim, large_param_dim))
    large_true = rng.standard_normal(large_param_dim) * 0.15
    large_obs = (
        large_design @ large_true
        + 0.04 * np.sin(large_coupling @ large_true)
        + 0.01 * rng.standard_normal(large_residual_dim)
    )
    large_initial = large_true + 0.12 * rng.standard_normal(large_param_dim)
    large_problem = DenseNonlinearLeastSquaresProblem(
        design_matrix=large_design,
        coupling_matrix=large_coupling,
        observations=large_obs,
        nonlinearity_scale=0.04,
    )

    return [
        BenchmarkCase(
            name="linear_regression",
            initial_state=line_initial,
            problem=line_problem,
            solver_kwargs=dict(
                damping_enabled=True,
                damping=1e-2,
                use_diagonal_damping=True,
                max_steps=12,
                max_iter=10,
                tolerance=1e-12,
                store_y_mags=False,
                store_states=False,
            ),
            error_fn=lambda state: float(np.linalg.norm(np.asarray(state) - true_line)),
        ),
        BenchmarkCase(
            name="exponential_fit",
            initial_state=exp_initial,
            problem=exp_problem,
            solver_kwargs=dict(
                damping_enabled=True,
                damping=1e-2,
                use_diagonal_damping=True,
                max_steps=20,
                max_iter=15,
                tolerance=1e-12,
                store_y_mags=False,
                store_states=False,
            ),
            error_fn=lambda state: float(np.linalg.norm(np.asarray(state) - true_exp)),
        ),
        BenchmarkCase(
            name="circle_fit",
            initial_state=circle_initial,
            problem=circle_problem,
            solver_kwargs=dict(
                damping_enabled=True,
                damping=1e-2,
                use_diagonal_damping=True,
                max_steps=20,
                max_iter=15,
                tolerance=1e-12,
                store_y_mags=False,
                store_states=False,
            ),
            error_fn=lambda state: float(np.linalg.norm(np.asarray(state) - true_circle)),
        ),
        BenchmarkCase(
            name="se3_point_alignment",
            initial_state=initial_se3,
            problem=se3_problem,
            solver_kwargs=dict(
                damping_enabled=True,
                damping=1e-2,
                use_diagonal_damping=True,
                max_steps=30,
                max_iter=20,
                tolerance=1e-12,
                store_y_mags=False,
                store_states=False,
            ),
            error_fn=lambda state: float(
                state.quat.angle_betweenD(true_se3.quat) + np.linalg.norm(state.tvec - true_se3.tvec)
            ),
        ),
        BenchmarkCase(
            name="dense_nonlinear_medium_48x2400",
            initial_state=medium_initial,
            problem=medium_problem,
            solver_kwargs=dict(
                damping_enabled=True,
                damping=1e-2,
                use_diagonal_damping=True,
                max_steps=12,
                max_iter=10,
                tolerance=1e-12,
                store_y_mags=False,
                store_states=False,
            ),
            error_fn=lambda state: float(np.linalg.norm(np.asarray(state) - medium_true)),
            repeats=5,
        ),
        BenchmarkCase(
            name="dense_nonlinear_large_96x6000",
            initial_state=large_initial,
            problem=large_problem,
            solver_kwargs=dict(
                damping_enabled=True,
                damping=1e-2,
                use_diagonal_damping=True,
                max_steps=10,
                max_iter=8,
                tolerance=1e-12,
                store_y_mags=False,
                store_states=False,
            ),
            error_fn=lambda state: float(np.linalg.norm(np.asarray(state) - large_true)),
            repeats=3,
        ),
    ]


def run_solver_benchmarks() -> None:
    np.set_printoptions(precision=6, suppress=True)

    backend_order: list[LinearSolverBackend] = [
        "numpy",
        "linalg_numpy",
        "linalg_spd_numpy",
    ]
    if LINALG_NUMBA_AVAILABLE:
        backend_order.extend(["linalg_numba", "linalg_spd_numba"])

    cases = make_benchmark_cases()
    print("Levenberg-Marquardt backend comparisons")
    print(f"Numba-backed Linalg available: {LINALG_NUMBA_AVAILABLE}")

    for case in cases:
        print(f"\n{case.name}")
        repeats = case.repeats
        reference_cost = None
        reference_error = None

        for backend in backend_order:
            elapsed_total = 0.0
            final_cost = None
            final_error = None
            final_iterations = None
            converged = None

            for _ in range(repeats):
                start = perf_counter()
                solver = LevenbergMarquardt(
                    state=_copy_benchmark_state(case.initial_state),
                    problem=case.problem,
                    linear_solver_backend=backend,
                    **case.solver_kwargs,
                )
                elapsed_total += perf_counter() - start
                final_cost = solver.final_cost
                final_error = case.error_fn(solver.state)
                final_iterations = solver.idx
                converged = solver.converged

            avg_ms = elapsed_total * 1000.0 / repeats
            if reference_cost is None:
                reference_cost = final_cost
                reference_error = final_error

            cost_delta = abs(float(final_cost) - float(reference_cost))
            error_delta = abs(float(final_error) - float(reference_error))
            print(
                f"  {backend}: {avg_ms:.3f} ms avg over {repeats} run(s), "
                f"iters={final_iterations}, converged={converged}, "
                f"final_cost={float(final_cost):.6e}, cost_delta={cost_delta:.3e}, "
                f"state_error={float(final_error):.6e}, error_delta={error_delta:.3e}"
            )


def main() -> None:
    run_solver_benchmarks()


if __name__ == "__main__":
    main()
