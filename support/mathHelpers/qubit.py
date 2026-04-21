# qubit.py
"""
Minimal single-qubit playground.

Key ideas:
- A qubit state is a normalized complex 2-vector |ψ> = [α, β]^T.
- Single-qubit gates are 2x2 unitary matrices U (often with det(U)=1 for SU(2)).
- On the Bloch sphere, SU(2) acts as 3D rotations (double cover of SO(3)),
  analogous to unit quaternions acting on R^3.

This module is intentionally small and "math-forward" so you can explore.

Doctests:
>>> import numpy as np
>>> q = Qubit.zero()
>>> np.allclose(q.state, np.array([1+0j, 0+0j]))
True
>>> q2 = q.apply(H)
>>> probs = q2.probabilities()
>>> np.allclose(probs, [0.5, 0.5])
True
>>> # Rx(pi) takes |0> -> -i|1>
>>> q3 = Qubit.zero().apply(Rx(np.pi))
>>> np.allclose(np.abs(q3.state), np.array([0.0, 1.0]))
True
>>> # Bloch vector of |0> is +Z, |1> is -Z
>>> np.allclose(Qubit.zero().bloch(), [0.0, 0.0, 1.0])
True
>>> np.allclose(Qubit.one().bloch(), [0.0, 0.0, -1.0])
True
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

C = np.complex128
R = np.float64

# --- Pauli matrices ---
I2: NDArray[C] = np.array([[1, 0], [0, 1]], dtype=C)
X:  NDArray[C] = np.array([[0, 1], [1, 0]], dtype=C)
Y:  NDArray[C] = np.array([[0, -1j], [1j, 0]], dtype=C)
Z:  NDArray[C] = np.array([[1, 0], [0, -1]], dtype=C)

# --- Common named gates ---
H: NDArray[C] = (1.0 / np.sqrt(2.0)) * np.array([[1, 1], [1, -1]], dtype=C)
S: NDArray[C] = np.array([[1, 0], [0, 1j]], dtype=C)
T: NDArray[C] = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=C)


def _as_col2(x: NDArray) -> NDArray[C]:
    x = np.asarray(x, dtype=C).reshape(2)
    return x


def _normalize_state(state: NDArray[C]) -> NDArray[C]:
    n = np.linalg.norm(state)
    if n < 1e-15:
        raise ValueError("State has ~zero norm; cannot normalize.")
    return state / n


def is_unitary(U: NDArray, atol: float = 1e-10) -> bool:
    U = np.asarray(U, dtype=C).reshape(2, 2)
    return np.allclose(U.conj().T @ U, I2, atol=atol)


def global_phase_fix(state: NDArray[C]) -> NDArray[C]:
    """
    Make a canonical representative by forcing the first nonzero component real+positive.
    Useful for printing/comparing states where global phase doesn't matter.
    """
    s = np.asarray(state, dtype=C).reshape(2)
    for k in range(2):
        if abs(s[k]) > 1e-12:
            ph = s[k] / abs(s[k])
            return s / ph
    return s


@dataclass(frozen=True)
class Qubit:
    """
    A single qubit pure state |ψ> = [α, β]^T with ||ψ||=1.

    Note: physical states are equivalent up to global phase. This class stores
    one representative; use global_phase_fix() if you want a canonical view.
    """
    state: NDArray[C]

    def __post_init__(self) -> None:
        s = _normalize_state(_as_col2(self.state))
        object.__setattr__(self, "state", s)

    @staticmethod
    def zero() -> "Qubit":
        return Qubit(np.array([1.0 + 0j, 0.0 + 0j], dtype=C))

    @staticmethod
    def one() -> "Qubit":
        return Qubit(np.array([0.0 + 0j, 1.0 + 0j], dtype=C))

    @staticmethod
    def from_angles(theta: float, phi: float) -> "Qubit":
        """
        Bloch-sphere parameterization:
          |ψ> = cos(theta/2)|0> + exp(i phi) sin(theta/2)|1>

        theta in [0, pi], phi in [0, 2pi).
        """
        a = np.cos(theta / 2.0)
        b = np.exp(1j * phi) * np.sin(theta / 2.0)
        return Qubit(np.array([a, b], dtype=C))

    def apply(self, U: NDArray) -> "Qubit":
        """
        Apply a 2x2 gate U to this qubit: |ψ'> = U|ψ>.
        """
        U = np.asarray(U, dtype=C).reshape(2, 2)
        # Allow non-unitary exploration if you want, but default expectation is unitary.
        out = U @ self.state
        return Qubit(out)

    def density(self) -> NDArray[C]:
        """ρ = |ψ><ψ|"""
        ket = self.state.reshape(2, 1)
        return ket @ ket.conj().T

    def probabilities(self) -> NDArray[np.float64]:
        """
        Measurement probabilities in computational basis:
          P(0)=|α|^2, P(1)=|β|^2
        """
        a, b = self.state
        return np.array([float((a.conj() * a).real), float((b.conj() * b).real)], dtype=R)

    def measure(self, rng: np.random.Generator | None = None) -> int:
        """
        Projective measurement in computational basis. Returns 0 or 1.
        """
        if rng is None:
            rng = np.random.default_rng()
        p0, p1 = self.probabilities()
        return int(rng.random() >= p0)

    def expectation(self, A: NDArray) -> complex:
        """
        <A> = <ψ|A|ψ>
        """
        A = np.asarray(A, dtype=C).reshape(2, 2)
        bra = self.state.conj().reshape(1, 2)
        ket = self.state.reshape(2, 1)
        return complex((bra @ A @ ket)[0, 0])

    def bloch(self) -> NDArray[np.float64]:
        """
        Bloch vector r = ( <X>, <Y>, <Z> ) for a pure state.
        """
        rx = self.expectation(X).real
        ry = self.expectation(Y).real
        rz = self.expectation(Z).real
        return np.array([float(rx), float(ry), float(rz)], dtype=R)

    def pretty(self, canonical_phase: bool = True) -> str:
        s = global_phase_fix(self.state) if canonical_phase else self.state
        a, b = s
        return f"|ψ> = [{a:.6g}, {b:.6g}]^T"


# --- Rotation gates: U = exp(-i θ/2 * (n·σ)) ---
def _rot(n: NDArray[np.float64], theta: float) -> NDArray[C]:
    n = np.asarray(n, dtype=R).reshape(3)
    nn = np.linalg.norm(n)
    if nn < 1e-15:
        raise ValueError("Rotation axis has ~zero norm.")
    n = n / nn
    nx, ny, nz = n
    sigma = nx * X + ny * Y + nz * Z
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return (c * I2 - 1j * s * sigma).astype(C)


def Rx(theta: float) -> NDArray[C]:
    return _rot(np.array([1.0, 0.0, 0.0], dtype=R), theta)


def Ry(theta: float) -> NDArray[C]:
    return _rot(np.array([0.0, 1.0, 0.0], dtype=R), theta)


def Rz(theta: float) -> NDArray[C]:
    return _rot(np.array([0.0, 0.0, 1.0], dtype=R), theta)


def R_axis_angle(axis: NDArray[np.float64], theta: float) -> NDArray[C]:
    """
    Generic axis-angle single-qubit rotation (SU(2)).
    """
    return _rot(axis, theta)


# --- Optional: quaternion <-> SU(2) mapping ---
def su2_from_quaternion(w: float, x: float, y: float, z: float) -> NDArray[C]:
    """
    Map a (unit) quaternion q = w + x i + y j + z k to an SU(2) matrix U.

    A standard identification between unit quaternions and SU(2) is:
        U = [[ w - i z,  -y - i x],
             [ y - i x,   w + i z]]

    If (w,x,y,z) is unit-length, U is unitary with det(U)=1.
    """
    return np.array(
        [[w - 1j * z, -y - 1j * x],
         [y - 1j * x,  w + 1j * z]],
        dtype=C
    )


def quaternion_from_su2(U: NDArray) -> tuple[float, float, float, float]:
    """
    Invert su2_from_quaternion for SU(2) matrices (up to numerical tolerance).

    Returns (w, x, y, z).
    """
    U = np.asarray(U, dtype=C).reshape(2, 2)
    a = U[0, 0]
    b = U[0, 1]
    # From mapping:
    # a = w - i z  => w = Re(a), z = -Im(a)
    # b = -y - i x => y = -Re(b), x = -Im(b)
    w = float(a.real)
    z = float(-a.imag)
    y = float(-b.real)
    x = float(-b.imag)
    # Normalize to be safe (global sign corresponds to same rotation on Bloch sphere)
    n = np.sqrt(w*w + x*x + y*y + z*z)
    if n < 1e-15:
        raise ValueError("Degenerate SU(2)->quat conversion (norm ~0).")
    return (w / n, x / n, y / n, z / n)

if __name__ == "__main__":
    q = Qubit.zero()
    print(q.pretty(), q.bloch())  # ~ [0,0,1]

    U = R_axis_angle([0, 1, 0], np.pi / 2)  # rotate about +Y by 90 deg
    q2 = q.apply(U)
    print(q2.pretty(), q2.bloch())  # ~ [1,0,0] (moved from +Z to +X)

    # Quaternion analogy: use your unit quaternion (w,x,y,z) -> SU(2) matrix -> act on qubit
    w, x, y, z = 0.9238795, 0.0, 0.3826834, 0.0  # ~ +Y, 45deg (half-angle in w)
    Uq = su2_from_quaternion(w, x, y, z)
    q3 = Qubit.zero().apply(Uq)
    print(q3.bloch())