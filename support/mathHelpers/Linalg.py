import numpy as np
from time import perf_counter

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


np.set_printoptions(precision=4, suppress=True)


def print_title(title):
    print(f"\n{'=' * 12} {title} {'=' * 12}")


def print_matrix(name, matrix):
    print(f"{name} =")
    print(matrix)


def adjoint(a):
    return a.conj().T


def complex_phase(z):
    return 1.0 if z == 0 else z / np.abs(z)


def working_dtype(*arrays):
    return np.result_type(*arrays, 1.0)


def as_column(x):
    array = np.asarray(x)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


@njit(cache=True)
def lu_factorize_numba(a):
    n = a.shape[0]
    lu = a.copy()
    pivots = np.arange(n)

    for pivot in range(n):
        pivot_row = pivot
        pivot_abs = np.abs(lu[pivot, pivot])
        for row in range(pivot + 1, n):
            value = np.abs(lu[row, pivot])
            if value > pivot_abs:
                pivot_abs = value
                pivot_row = row

        if np.isclose(lu[pivot_row, pivot], 0.0):
            raise ValueError("Matrix is singular to working precision.")

        if pivot_row != pivot:
            for col in range(n):
                temp = lu[pivot, col]
                lu[pivot, col] = lu[pivot_row, col]
                lu[pivot_row, col] = temp
            temp_index = pivots[pivot]
            pivots[pivot] = pivots[pivot_row]
            pivots[pivot_row] = temp_index

        for row in range(pivot + 1, n):
            lu[row, pivot] /= lu[pivot, pivot]
            factor = lu[row, pivot]
            for col in range(pivot + 1, n):
                lu[row, col] -= factor * lu[pivot, col]

    return lu, pivots


@njit(cache=True)
def solve_lu_numba(lu, pivots, b):
    n = lu.shape[0]
    pb = np.empty(n, dtype=lu.dtype)
    y = np.empty(n, dtype=lu.dtype)
    x = np.empty(n, dtype=lu.dtype)

    for i in range(n):
        pb[i] = b[pivots[i]]

    for i in range(n):
        rhs = pb[i]
        for j in range(i):
            rhs -= lu[i, j] * y[j]
        y[i] = rhs

    for i in range(n - 1, -1, -1):
        rhs = y[i]
        for j in range(i + 1, n):
            rhs -= lu[i, j] * x[j]
        x[i] = rhs / lu[i, i]

    return x


@njit(cache=True)
def solve_linear_system_numba(a, b):
    lu, pivots = lu_factorize_numba(a)
    return lu, pivots, solve_lu_numba(lu, pivots, b)


@njit(cache=True)
def cholesky_manual_numba(a):
    n = a.shape[0]
    l = np.zeros_like(a)

    for i in range(n):
        for j in range(i + 1):
            if i == j:
                diagonal_term = a[i, i]
                for k in range(j):
                    diagonal_term -= l[i, k] * l[i, k]
                l[i, j] = np.sqrt(diagonal_term)
            else:
                numerator = a[i, j]
                for k in range(j):
                    numerator -= l[i, k] * l[j, k]
                l[i, j] = numerator / l[j, j]

    return l


@njit(cache=True)
def matrix_exponential_series_numba(a, terms):
    n = a.shape[0]
    result = np.eye(n, dtype=a.dtype)
    current = np.eye(n, dtype=a.dtype)

    for k in range(1, terms):
        current = current @ a / k
        result = result + current

    return result


@njit(cache=True)
def power_iteration_numba(a, steps):
    n = a.shape[0]
    v = np.ones(n, dtype=a.dtype)
    v /= np.sqrt(np.sum(v * v))
    eigenvalue = 0.0

    for _ in range(steps):
        w = a @ v
        numerator = np.sum(v * w)
        denominator = np.sum(v * v)
        eigenvalue = numerator / denominator
        v = w / np.sqrt(np.sum(w * w))

    return eigenvalue, v


def demo_basic_operations():
    print_title("Basic Operations")
    a = np.array([[2.0, -1.0], [0.0, 3.0]])
    b = np.array([[1.0, 4.0], [2.0, -2.0]])
    x = np.array([[3.0], [1.0]])

    print_matrix("A", a)
    print_matrix("B", b)
    print_matrix("x", x)
    print_matrix("A + B", a + b)
    print_matrix("A - B", a - b)
    print_matrix("A @ B", a @ b)
    print_matrix("A.T", a.T)
    print_matrix("A @ x", a @ x)


def demo_gaussian_elimination():
    print_title("Gaussian Elimination")
    a = np.array(
        [
            [2.0, 1.0, -1.0],
            [-3.0, -1.0, 2.0],
            [-2.0, 1.0, 2.0],
        ]
    )
    b = np.array([[8.0], [-11.0], [-3.0]])
    augmented = np.array(
        [
            [2.0, 1.0, -1.0, 8.0],
            [-3.0, -1.0, 2.0, -11.0],
            [-2.0, 1.0, 2.0, -3.0],
        ]
    )

    print_matrix("[A|b] start", augmented)

    augmented[1] = augmented[1] + 1.5 * augmented[0]
    augmented[2] = augmented[2] + augmented[0]
    print_matrix("After eliminating below row 1", augmented)

    augmented[2] = augmented[2] - 4.0 * augmented[1]
    print_matrix("After eliminating below row 2", augmented)

    solution = np.zeros(3)
    solution[2] = augmented[2, 3] / augmented[2, 2]
    solution[1] = (augmented[1, 3] - augmented[1, 2] * solution[2]) / augmented[1, 1]
    solution[0] = (
        augmented[0, 3] - augmented[0, 1] * solution[1] - augmented[0, 2] * solution[2]
    ) / augmented[0, 0]
    print_matrix("solution", solution.reshape(-1, 1))
    print_matrix("A @ solution", a @ solution.reshape(-1, 1))
    print_matrix("b", b)


def lu_factorize_manual(a):
    lu = np.array(a, dtype=working_dtype(a), copy=True)
    n = lu.shape[0]
    pivots = np.arange(n)

    for pivot in range(n):
        pivot_row = pivot + np.argmax(np.abs(lu[pivot:, pivot]))
        if np.isclose(lu[pivot_row, pivot], 0.0):
            raise ValueError("Matrix is singular to working precision.")

        if pivot_row != pivot:
            lu[[pivot, pivot_row]] = lu[[pivot_row, pivot]]
            pivots[[pivot, pivot_row]] = pivots[[pivot_row, pivot]]

        for row in range(pivot + 1, n):
            lu[row, pivot] = lu[row, pivot] / lu[pivot, pivot]
            lu[row, pivot + 1 :] = lu[row, pivot + 1 :] - lu[row, pivot] * lu[pivot, pivot + 1 :]

    return lu, pivots


def solve_lu_manual(lu, pivots, b):
    b = as_column(b).astype(working_dtype(lu, b))
    n = lu.shape[0]
    pb = b[pivots]
    y = np.zeros((n, 1), dtype=working_dtype(lu, b))
    x = np.zeros((n, 1), dtype=working_dtype(lu, b))

    for i in range(n):
        y[i, 0] = pb[i, 0] - np.dot(lu[i, :i], y[:i, 0])

    for i in range(n - 1, -1, -1):
        x[i, 0] = (y[i, 0] - np.dot(lu[i, i + 1 :], x[i + 1 :, 0])) / lu[i, i]

    return x


def solve_linear_system_manual(a, b):
    lu, pivots = lu_factorize_manual(a)
    return lu, pivots, solve_lu_manual(lu, pivots, b)


def least_squares_normal_equations_manual(a, b):
    b = as_column(b)
    normal_matrix = adjoint(a) @ a
    normal_rhs = adjoint(a) @ b
    lu, pivots, x = solve_linear_system_manual(normal_matrix, normal_rhs)
    return normal_matrix, normal_rhs, lu, pivots, x


def least_squares_normal_equations_numba(a, b):
    b = as_column(b)
    normal_matrix = a.T @ a
    normal_rhs = a.T @ b
    lu, pivots, x = solve_linear_system_numba(normal_matrix, normal_rhs[:, 0])
    return normal_matrix, normal_rhs, lu, pivots, x.reshape(-1, 1)


def solve_linear_system_numpy(a, b):
    return np.linalg.solve(a, as_column(b))


def least_squares_numpy(a, b):
    return np.linalg.lstsq(a, as_column(b), rcond=None)[0]


def cholesky_manual(a):
    n = a.shape[0]
    l = np.zeros_like(a, dtype=working_dtype(a))

    for i in range(n):
        for j in range(i + 1):
            if i == j:
                diagonal_term = a[i, i] - np.sum(l[i, :j] * np.conj(l[i, :j]))
                l[i, j] = np.sqrt(diagonal_term)
            else:
                numerator = a[i, j] - np.sum(l[i, :j] * np.conj(l[j, :j]))
                l[i, j] = numerator / l[j, j]
    return l


def demo_cholesky():
    print_title("Cholesky Decomposition")
    a = np.array(
        [
            [25.0, 15.0, -5.0],
            [15.0, 18.0, 0.0],
            [-5.0, 0.0, 11.0],
        ]
    )

    print_matrix("A", a)
    l = cholesky_manual(a)
    print_matrix("L", l)
    print_matrix("L @ L*", l @ adjoint(l))


def matrix_exponential_series(a, terms=10):
    dtype = working_dtype(a)
    result = np.eye(a.shape[0], dtype=dtype)
    current = np.eye(a.shape[0], dtype=dtype)

    for k in range(1, terms):
        current = current @ a / k
        print_matrix(f"Term {k}: A^{k}/{k}!", current)
        result = result + current

    return result


def matrix_exponential_series_no_print(a, terms=10):
    dtype = working_dtype(a)
    result = np.eye(a.shape[0], dtype=dtype)
    current = np.eye(a.shape[0], dtype=dtype)

    for k in range(1, terms):
        current = current @ a / k
        result = result + current

    return result


def demo_power_series():
    print_title("Power Series / Matrix Exponential")
    a = np.array([[0.0, 1.0], [-1.0, 0.0]])
    print_matrix("A", a)
    approx = matrix_exponential_series(a, terms=8)
    print_matrix("exp(A) approx", approx)


def power_iteration(a, steps=8):
    v = np.ones(a.shape[0], dtype=working_dtype(a))
    v = v / np.linalg.norm(v)

    for step in range(steps):
        w = a @ v
        eigenvalue = np.vdot(v, w) / np.vdot(v, v)
        v = w / np.linalg.norm(w)
        print_matrix(f"v_{step + 1}", v.reshape(-1, 1))
        print(f"Rayleigh quotient {step + 1}: {eigenvalue}")

    return eigenvalue, v


def power_iteration_no_print(a, steps=8):
    v = np.ones(a.shape[0], dtype=working_dtype(a))
    v = v / np.linalg.norm(v)
    eigenvalue = 0.0

    for _ in range(steps):
        w = a @ v
        eigenvalue = np.vdot(v, w) / np.vdot(v, v)
        v = w / np.linalg.norm(w)

    return eigenvalue, v


def demo_power_iteration():
    print_title("Power Iteration")
    a = np.array(
        [
            [4.0, 1.0, 0.0],
            [1.0, 3.0, 1.0],
            [0.0, 1.0, 2.0],
        ]
    )
    print_matrix("A", a)
    eigenvalue, eigenvector = power_iteration(a)
    print(f"Dominant eigenvalue approx: {eigenvalue}")
    print_matrix("Dominant eigenvector approx", eigenvector.reshape(-1, 1))


def householder_qr(a):
    dtype = working_dtype(a)
    r = a.astype(dtype).copy()
    m, n = r.shape
    q = np.eye(m, dtype=dtype)

    for k in range(min(m - 1, n)):
        x = r[k:, k]
        alpha = -complex_phase(x[0]) * np.linalg.norm(x)
        if np.abs(alpha) == 0.0:
            continue

        e1 = np.zeros_like(x)
        e1[0] = 1.0
        v = x - alpha * e1
        v = v / np.linalg.norm(v)

        h_small = np.eye(len(x), dtype=dtype) - 2.0 * np.outer(v, np.conj(v))
        h = np.eye(m, dtype=dtype)
        h[k:, k:] = h_small

        print_matrix(f"Householder H_{k + 1}", h)
        r = h @ r
        q = q @ h
        print_matrix(f"R after H_{k + 1}", r)

    return q, r


def demo_householder():
    print_title("Householder QR")
    a = np.array(
        [
            [12.0, -51.0, 4.0],
            [6.0, 167.0, -68.0],
            [-4.0, 24.0, -41.0],
        ]
    )
    print_matrix("A", a)
    q, r = householder_qr(a)
    print_matrix("Q", q)
    print_matrix("R", r)
    print_matrix("Q @ R", q @ r)


def givens_rotation(a, i, j, col):
    x = a[i, col]
    y = a[j, col]
    r = np.sqrt(np.abs(x) ** 2 + np.abs(y) ** 2)
    if r == 0:
        return np.eye(a.shape[0], dtype=working_dtype(a))

    c = x / r
    s = y / r
    g = np.eye(a.shape[0], dtype=working_dtype(a))
    g[i, i] = np.conj(c)
    g[i, j] = np.conj(s)
    g[j, i] = -s
    g[j, j] = c
    return g


def demo_givens():
    print_title("Givens Rotations")
    a = np.array(
        [
            [6.0, 5.0, 0.0],
            [5.0, 1.0, 4.0],
            [0.0, 4.0, 3.0],
        ]
    )
    print_matrix("A", a)

    g1 = givens_rotation(a, 0, 1, 0)
    a1 = g1 @ a
    print_matrix("G_1", g1)
    print_matrix("G_1 @ A", a1)

    g2 = givens_rotation(a1, 1, 2, 1)
    a2 = g2 @ a1
    print_matrix("G_2", g2)
    print_matrix("G_2 @ G_1 @ A", a2)


def classical_gram_schmidt(a):
    m, n = a.shape
    dtype = working_dtype(a)
    q = np.zeros((m, n), dtype=dtype)
    r = np.zeros((n, n), dtype=dtype)

    for j in range(n):
        v = a[:, j].copy()
        for i in range(j):
            r[i, j] = np.vdot(q[:, i], a[:, j])
            v = v - r[i, j] * q[:, i]
        r[j, j] = np.linalg.norm(v)
        q[:, j] = v / r[j, j]
        print_matrix(f"q_{j + 1}", q[:, : j + 1])

    return q, r


def demo_gram_schmidt():
    print_title("Gram-Schmidt")
    a = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    print_matrix("A", a)
    q, r = classical_gram_schmidt(a)
    print_matrix("Q", q)
    print_matrix("R", r)
    print_matrix("Q* @ Q", adjoint(q) @ q)


def demo_least_squares():
    print_title("Least Squares")
    a = np.array(
        [
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 4.0],
        ]
    )
    b = np.array([[6.0], [5.0], [7.0], [10.0]])

    print_matrix("A", a)
    print_matrix("b", b)
    normal_matrix, normal_rhs, lu, pivots, x = least_squares_normal_equations_manual(a, b)
    print_matrix("A* @ A", normal_matrix)
    print_matrix("A* @ b", normal_rhs)
    print_matrix("LU-packed normal-equation factorization", lu)
    print_matrix("Pivot vector", pivots.reshape(-1, 1))
    print_matrix("Least squares solution", x)
    print_matrix("A @ x", a @ x)


def demo_projection():
    print_title("Vector Projection")
    u = np.array([[2.0], [1.0], [2.0]])
    v = np.array([[3.0], [0.0], [1.0]])

    print_matrix("u", u)
    print_matrix("v", v)
    projection_scale = np.vdot(u[:, 0], v[:, 0]) / np.vdot(u[:, 0], u[:, 0])
    projection = projection_scale * u
    print_matrix("proj_u(v)", projection)
    print_matrix("orthogonal component", v - projection)


def benchmark(label, fn, repeat=5):
    times = []
    result = None

    for _ in range(repeat):
        start = perf_counter()
        result = fn()
        times.append(perf_counter() - start)

    return min(times), result


def dominant_eigenpair_numpy(a, steps=None):
    eigenvalues, eigenvectors = np.linalg.eigh(a)
    idx = np.argmax(np.abs(eigenvalues))
    return eigenvalues[idx], eigenvectors[:, idx]


def matrix_exponential_via_eig_numpy(a, terms=None):
    eigenvalues, eigenvectors = np.linalg.eig(a)
    inverse = np.linalg.inv(eigenvectors)
    return eigenvectors @ np.diag(np.exp(eigenvalues)) @ inverse


BACKEND_IMPLEMENTATIONS = {
    "python": {
        "solve": lambda a, b: solve_linear_system_manual(a, b)[2],
        "least_squares": lambda a, b: least_squares_normal_equations_manual(a, b)[4],
        "cholesky": cholesky_manual,
        "power_iteration": power_iteration_no_print,
        "matrix_exponential_series": matrix_exponential_series_no_print,
    },
    "numpy": {
        "solve": solve_linear_system_numpy,
        "least_squares": least_squares_numpy,
        "cholesky": np.linalg.cholesky,
        "power_iteration": dominant_eigenpair_numpy,
        "matrix_exponential_series": matrix_exponential_via_eig_numpy,
    },
}

if NUMBA_AVAILABLE:
    BACKEND_IMPLEMENTATIONS["numba"] = {
        "solve": lambda a, b: solve_linear_system_numba(a, as_column(b)[:, 0])[2].reshape(-1, 1),
        "least_squares": lambda a, b: least_squares_normal_equations_numba(a, b)[4],
        "cholesky": cholesky_manual_numba,
        "power_iteration": power_iteration_numba,
        "matrix_exponential_series": matrix_exponential_series_numba,
    }


def available_backends():
    return tuple(BACKEND_IMPLEMENTATIONS.keys())


def call_backend(operation, backend, *args, **kwargs):
    if backend not in BACKEND_IMPLEMENTATIONS:
        raise ValueError(f"Unknown backend '{backend}'. Available: {available_backends()}")

    operations = BACKEND_IMPLEMENTATIONS[backend]
    if operation not in operations:
        raise ValueError(f"Backend '{backend}' does not implement '{operation}'.")

    return operations[operation](*args, **kwargs)


def warm_up_numba_backend(a_solve, b_solve, m_ls, b_ls, spd, symmetric, matrix_exp):
    if not NUMBA_AVAILABLE:
        return

    call_backend("solve", "numba", a_solve, b_solve)
    call_backend("least_squares", "numba", m_ls, b_ls)
    call_backend("cholesky", "numba", spd)
    call_backend("power_iteration", "numba", symmetric, 20)
    call_backend("matrix_exponential_series", "numba", matrix_exp, 20)


def demo_backend_comparison():
    print_title("Backend Comparison")
    rng = np.random.default_rng(7)

    a_solve = rng.standard_normal((120, 120))
    a_solve += 5.0 * np.eye(120)
    b_solve = rng.standard_normal((120, 1))

    m_ls = rng.standard_normal((240, 12))
    b_ls = rng.standard_normal((240, 1))

    m_chol = rng.standard_normal((140, 140))
    spd = m_chol.T @ m_chol + 1e-3 * np.eye(140)

    m_power = rng.standard_normal((160, 160))
    symmetric = 0.5 * (m_power + m_power.T)

    matrix_exp = rng.standard_normal((20, 20)) * 0.05

    warm_up_numba_backend(a_solve, b_solve, m_ls, b_ls, spd, symmetric, matrix_exp)

    comparison_backends = ["python"]
    if NUMBA_AVAILABLE:
        comparison_backends.append("numba")
    comparison_backends.append("numpy")

    solve_results = {}
    ls_results = {}
    chol_results = {}
    power_results = {}
    exp_results = {}

    for backend in comparison_backends:
        solve_results[backend] = benchmark(
            f"{backend} solve", lambda backend=backend: call_backend("solve", backend, a_solve, b_solve)
        )
        ls_results[backend] = benchmark(
            f"{backend} least squares",
            lambda backend=backend: call_backend("least_squares", backend, m_ls, b_ls),
        )
        chol_results[backend] = benchmark(
            f"{backend} cholesky", lambda backend=backend: call_backend("cholesky", backend, spd)
        )
        power_results[backend] = benchmark(
            f"{backend} power iteration",
            lambda backend=backend: call_backend("power_iteration", backend, symmetric, 20),
        )
        exp_results[backend] = benchmark(
            f"{backend} matrix exponential series",
            lambda backend=backend: call_backend("matrix_exponential_series", backend, matrix_exp, 20),
        )

    solve_reference = solve_results["python"][1]
    ls_reference = ls_results["python"][1]
    chol_reference = chol_results["python"][1]
    power_reference_lambda, power_reference_v = power_results["python"][1]
    exp_reference = exp_results["python"][1]

    print("Available backends:", ", ".join(comparison_backends))
    print("Solve:", " ".join(f"{backend}={solve_results[backend][0]:.6f}s" for backend in comparison_backends))
    print(
        "Solve diffs:",
        " ".join(
            f"{backend}-vs-python={np.linalg.norm(as_column(solve_results[backend][1]) - solve_reference):.3e}"
            for backend in comparison_backends
            if backend != "python"
        ),
    )

    print(
        "Least squares:",
        " ".join(f"{backend}={ls_results[backend][0]:.6f}s" for backend in comparison_backends),
    )
    print(
        "Least-squares diffs:",
        " ".join(
            f"{backend}-vs-python={np.linalg.norm(as_column(ls_results[backend][1]) - ls_reference):.3e}"
            for backend in comparison_backends
            if backend != "python"
        ),
    )

    print("Cholesky:", " ".join(f"{backend}={chol_results[backend][0]:.6f}s" for backend in comparison_backends))
    print(
        "Cholesky diffs:",
        " ".join(
            f"{backend}-vs-python={np.linalg.norm(chol_results[backend][1] - chol_reference):.3e}"
            for backend in comparison_backends
            if backend != "python"
        ),
    )

    print(
        "Power iteration:",
        " ".join(f"{backend}={power_results[backend][0]:.6f}s" for backend in comparison_backends),
    )
    print(
        "Power lambda diffs:",
        " ".join(
            f"{backend}-vs-python={abs(power_results[backend][1][0] - power_reference_lambda):.3e}"
            for backend in comparison_backends
            if backend != "python"
        ),
    )
    print(
        "Power vector diffs:",
        " ".join(
            f"{backend}-vs-python={min(np.linalg.norm(power_results[backend][1][1] - power_reference_v), np.linalg.norm(power_results[backend][1][1] + power_reference_v)):.3e}"
            for backend in comparison_backends
            if backend != "python"
        ),
    )

    print(
        "Matrix exponential series:",
        " ".join(f"{backend}={exp_results[backend][0]:.6f}s" for backend in comparison_backends),
    )
    print(
        "Series diffs:",
        " ".join(
            f"{backend}-vs-python={np.linalg.norm(exp_results[backend][1] - exp_reference):.3e}"
            for backend in comparison_backends
            if backend != "python"
        ),
    )


def run_comparisons():
    from support.runtime.timing_tools import timed

    RNG_SEED = 7
    ATOL = 1e-8
    RTOL = 1e-6
    POWER_ITERATION_STEPS = 200

    def make_benchmark_inputs():
        rng = np.random.default_rng(RNG_SEED)

        a_solve = rng.standard_normal((120, 120))
        a_solve += 5.0 * np.eye(120)
        b_solve = rng.standard_normal((120, 1))

        m_ls = rng.standard_normal((240, 12))
        b_ls = rng.standard_normal((240, 1))

        m_chol = rng.standard_normal((140, 140))
        spd = m_chol.T @ m_chol + 1e-3 * np.eye(140)

        m_power = rng.standard_normal((160, 160))
        symmetric = 0.5 * (m_power + m_power.T)

        matrix_exp = rng.standard_normal((20, 20)) * 0.05

        return {
            "solve": (a_solve, b_solve),
            "least_squares": (m_ls, b_ls),
            "cholesky": (spd,),
            "power_iteration": (symmetric, POWER_ITERATION_STEPS),
            "matrix_exponential_series": (matrix_exp, 20),
        }

    def warm_up_numba(inputs):
        if not NUMBA_AVAILABLE:
            return

        a_solve, b_solve = inputs["solve"]
        m_ls, b_ls = inputs["least_squares"]
        (spd,) = inputs["cholesky"]
        symmetric, _ = inputs["power_iteration"]
        matrix_exp, _ = inputs["matrix_exponential_series"]

        warm_up_numba_backend(a_solve, b_solve, m_ls, b_ls, spd, symmetric, matrix_exp)

    def assert_allclose(name, numpy_result, linalg_result):
        np.testing.assert_allclose(
            np.asarray(linalg_result),
            np.asarray(numpy_result),
            atol=ATOL,
            rtol=RTOL,
            err_msg=f"{name} results diverged",
        )

    def assert_eigenpair_close(name, numpy_result, linalg_result):
        numpy_eigenvalue, numpy_eigenvector = numpy_result
        linalg_eigenvalue, linalg_eigenvector = linalg_result

        np.testing.assert_allclose(
            linalg_eigenvalue,
            numpy_eigenvalue,
            atol=ATOL,
            rtol=RTOL,
            err_msg=f"{name} eigenvalues diverged",
        )

        numpy_vector = np.asarray(numpy_eigenvector).reshape(-1)
        linalg_vector = np.asarray(linalg_eigenvector).reshape(-1)
        vector_error = min(
            np.linalg.norm(linalg_vector - numpy_vector),
            np.linalg.norm(linalg_vector + numpy_vector),
        )
        if vector_error > 1e-4:
            raise AssertionError(f"{name} eigenvectors diverged: error={vector_error:.3e}")

    def benchmark_operation(name, numpy_fn, linalg_fn, args, repeats, validator):
        numpy_result = None
        linalg_result = None

        @timed(repeats=repeats)
        def run_numpy():
            nonlocal numpy_result
            numpy_result = numpy_fn(*args)
            return numpy_result

        @timed(repeats=repeats)
        def run_linalg():
            nonlocal linalg_result
            linalg_result = linalg_fn(*args)
            return linalg_result

        print(f"\n{name}")
        print(f"  numpy:")
        run_numpy()
        print(f"  linalg ({'numba' if NUMBA_AVAILABLE else 'python'}):")
        run_linalg()
        validator(name, numpy_result, linalg_result)
        print("  result check: OK")

    inputs = make_benchmark_inputs()
    warm_up_numba(inputs)

    backend = "numba" if NUMBA_AVAILABLE else "python"

    benchmark_operation(
        name="solve",
        numpy_fn=lambda a, b: solve_linear_system_numpy(a, b),
        linalg_fn=lambda a, b: call_backend("solve", backend, a, b),
        args=inputs["solve"],
        repeats=50,
        validator=assert_allclose,
    )

    benchmark_operation(
        name="least_squares",
        numpy_fn=lambda a, b: least_squares_numpy(a, b),
        linalg_fn=lambda a, b: call_backend("least_squares", backend, a, b),
        args=inputs["least_squares"],
        repeats=50,
        validator=assert_allclose,
    )

    benchmark_operation(
        name="cholesky",
        numpy_fn=np.linalg.cholesky,
        linalg_fn=lambda a: call_backend("cholesky", backend, a),
        args=inputs["cholesky"],
        repeats=100,
        validator=assert_allclose,
    )

    benchmark_operation(
        name="power_iteration",
        numpy_fn=lambda a, steps: dominant_eigenpair_numpy(a),
        linalg_fn=lambda a, steps: call_backend("power_iteration", backend, a, steps),
        args=inputs["power_iteration"],
        repeats=100,
        validator=assert_eigenpair_close,
    )

    benchmark_operation(
        name="matrix_exponential_series",
        numpy_fn=lambda a, terms: matrix_exponential_via_eig_numpy(a),
        linalg_fn=lambda a, terms: call_backend("matrix_exponential_series", backend, a, terms),
        args=inputs["matrix_exponential_series"],
        repeats=100,
        validator=assert_allclose,
    )


def main():
    # demo_basic_operations()
    # demo_projection()
    # demo_gaussian_elimination()
    # demo_cholesky()
    # demo_power_series()
    # demo_power_iteration()
    # demo_gram_schmidt()
    # demo_householder()
    # demo_givens()
    # demo_least_squares()
    # demo_backend_comparison()

    run_comparisons()


if __name__ == "__main__":
    main()
