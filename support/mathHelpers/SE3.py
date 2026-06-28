import numpy as np
from numpy.typing import NDArray
from typing import overload
import support.mathHelpers.quaternions as q

_FLOAT_EPS = np.finfo(np.float64).eps

class SE3_q:
    def __init__(self,
                 quat: q.Quaternion | None = None,
                 tvec: NDArray | None = None) -> None:
        self.quat: q.Quaternion = q.identity() if quat is None else q.Quaternion(quat=quat, makeUnitQuat=False)
        self.tvec: NDArray = np.zeros((3,), dtype=float) if tvec is None else np.asarray(tvec, dtype=float).reshape(3)

    def __repr__(self) -> str:
        return f"SE3_q(quat={self.quat!r}, tvec={self.tvec!r})"

    def copy(self) -> "SE3_q":
        return SE3_q(quat=self.quat.copy(), tvec=self.tvec.copy())

    def compose(self, other: "SE3_q") -> "SE3_q":
        if not isinstance(other, SE3_q):
            raise ValueError(f"Can only compose SE3_q with SE3_q, got {type(other)}")
        return SE3_q(
            quat=(self.quat * other.quat).normalize(),
            tvec=self.quat * other.tvec + self.tvec,
        )

    @overload
    def __mul__(self, other: "SE3_q") -> "SE3_q":
        ...

    @overload
    def __mul__(self, other: np.ndarray) -> np.ndarray:
        ...

    def __mul__(self, other: object) -> "SE3_q | np.ndarray | NotImplementedType":
        if isinstance(other, SE3_q):
            return self.compose(other)
        if isinstance(other, np.ndarray):
            arr = np.asarray(other, dtype=float)
            if arr.shape == (3,):
                return self.quat * arr + self.tvec
            if arr.ndim == 2 and arr.shape == (3, 3):
                return (self.quat * arr) + self.tvec.reshape(3, 1)
            if arr.ndim == 2 and arr.shape[1] == 3:
                return self.quat * arr + self.tvec
            if arr.ndim == 2 and arr.shape[0] == 3:
                return (self.quat * arr) + self.tvec.reshape(3, 1)
            raise ValueError(f"SE3_q can only transform arrays with one dimension equal to 3, got shape {arr.shape}")
        return NotImplemented

    @property
    def inv(self) -> "SE3_q":
        q_inv = self.quat.T
        return SE3_q(quat=q_inv, tvec=-(q_inv * self.tvec))

    def inverse(self) -> "SE3_q":
        return self.inv

    @property
    def array(self) -> NDArray:
        return self.quat.to_SE3_given_position(self.tvec)

    @property
    def matrix(self) -> NDArray:
        return self.array

    @property
    def minimal(self) -> NDArray:
        return np.concatenate((self.quat.ln_so3.vec, self.tvec))

    def jacobian(self) -> NDArray:
        """
        Jacobian of the local minimal pose coordinates under a left perturbation.

        Let the pose state be parameterized as x = [phi, t], where
            phi = Log_SO3(q)
        and let the optimizer apply a left perturbation
            T <- Exp(delta) * T
        with delta = [dtheta, dt].

        This returns dx / ddelta evaluated at the current pose:
            [ J_l(phi)^(-T)   0 ]
            [   -[t]_x        I ]

        The transpose on the rotational block matches the row-major 3-vector
        conventions used by this module's quaternion log map helpers.
        """
        J = np.eye(6, dtype=float)
        phi = self.quat.ln_so3.vec
        J[:3, :3] = np.linalg.inv(q.so3_left_jacobian(phi)).T
        J[3:, :3] = -q.skew(self.tvec)
        return J

    @staticmethod
    def random(tvec_length: float = 1.0) -> "SE3_q":
        tvec_length = float(tvec_length)
        if tvec_length < 0.0:
            raise ValueError("tvec_length must be non-negative")

        if tvec_length == 0.0:
            tvec = np.zeros(3, dtype=float)
        else:
            from typing import cast
            tvec: NDArray = cast(NDArray, np.random.randn(3))
            norm = np.linalg.norm(tvec)
            if norm < _FLOAT_EPS:
                tvec = np.array([tvec_length, 0.0, 0.0], dtype=float)
            else:
                tvec = (tvec / norm) * tvec_length

        return SE3_q(quat=q.randomQuat(), tvec=tvec)

    @staticmethod
    def from_SE3_DCM(Mat4: NDArray) -> "SE3_q":
        quat, tvec = q.from_SE3(Mat4)
        return SE3_q(quat=quat, tvec=tvec)
