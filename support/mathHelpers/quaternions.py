"""Functions to operate on, or return, quaternions.

Quaternions here consist of 4 values ``w, x, y, z``, where ``w`` is the
real (scalar) part, and ``x, y, z`` are the complex (vector) part.

Note - rotation matrices here apply to column vectors, that is,
they are applied on the left of the vector.  For example:

Terms used in function names:

* *mat* : array shape (3, 3) (3D non-homogenous coordinates)
* *aff* : affine array shape (4, 4) (3D homogenous coordinates)
* *quat* : quaternion shape (4,)
* *axangle* : rotations encoded by axis vector and angle scalar
"""

import numpy as np
from numbers import Real
from types import NotImplementedType
from typing import Any, Optional
from numpy import cos, arccos, sin, arcsin, arctan2, rad2deg, deg2rad, sqrt, abs
from numpy.typing import NDArray
from typing_extensions import Self

# Import overwritten Numba decorator
# if user has Numba, allows for njit decorator and prange
# if user doesn't have Numba, njit decorator does nothing and prange is aliased of range
from support.mathHelpers.include_numba import _njit as njit, prange

_FLOAT_EPS = np.finfo(np.float64).eps


@njit(cache=True, fastmath=False)
def _cross3(a0, a1, a2, b0, b1, b2):
    return (a1 * b2 - a2 * b1,
            a2 * b0 - a0 * b2,
            a0 * b1 - a1 * b0)


@njit(parallel=True, cache=True, fastmath=False)
def rotate_vecs_quat_numba(w: float, x: float, y: float, z: float, vecs: np.ndarray) -> np.ndarray:
    """
    Rotate Nx3 vectors by unit quaternion q = (w, x, y, z).
    Uses the 't' formulation: t = 2 * (q_vec x v); v' = v + w*t + (q_vec x t)
    """
    N = vecs.shape[0]
    out = np.empty((N, 3), dtype=np.float64)

    for i in prange(N):
        vx = vecs[i, 0]
        vy = vecs[i, 1]
        vz = vecs[i, 2]

        # t = 2 * cross(qvec, v)
        cx0, cx1, cx2 = _cross3(x, y, z, vx, vy, vz)
        tx = 2.0 * cx0
        ty = 2.0 * cx1
        tz = 2.0 * cx2

        # cross(qvec, t)
        c2x0, c2x1, c2x2 = _cross3(x, y, z, tx, ty, tz)

        out[i, 0] = vx + w * tx + c2x0
        out[i, 1] = vy + w * ty + c2x1
        out[i, 2] = vz + w * tz + c2x2

    return out


@njit(cache=True, fastmath=False)
def qmul_numba(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a,b are shape (4,) = [w,x,y,z]
    aw, ax, ay, az = a[0], a[1], a[2], a[3]
    bw, bx, by, bz = b[0], b[1], b[2], b[3]
    out = np.empty(4, dtype=np.float64)
    out[0] = aw * bw - ax * bx - ay * by - az * bz
    out[1] = aw * bx + ax * bw + ay * bz - az * by
    out[2] = aw * by - ax * bz + ay * bw + az * bx
    out[3] = aw * bz + ax * by - ay * bx + az * bw
    return out


@njit(parallel=True, cache=True, fastmath=False)
def qmul_batch_left_numba(a: np.ndarray, Bs: np.ndarray) -> np.ndarray:
    # a shape (4,), Bs shape (N,4) => out (N,4) = a ⊗ Bs[i]
    N = Bs.shape[0]
    out = np.empty((N, 4), dtype=np.float64)
    aw, ax, ay, az = a[0], a[1], a[2], a[3]
    for i in prange(N):
        bw, bx, by, bz = Bs[i, 0], Bs[i, 1], Bs[i, 2], Bs[i, 3]
        out[i, 0] = aw * bw - ax * bx - ay * by - az * bz
        out[i, 1] = aw * bx + ax * bw + ay * bz - az * by
        out[i, 2] = aw * by - ax * bz + ay * bw + az * bx
        out[i, 3] = aw * bz + ax * by - ay * bx + az * bw
    return out


class Quaternion:
    __slots__ = ("s", "vec", "_cache4")
    __array_priority__ = 10_000  # overrides numpy priority for right mult

    def __init__(
        self,
        s: Optional[float] = None,
        vec: Optional[NDArray] = None,
        quat: Optional[NDArray | Self] = None,
        makeUnitQuat: bool = True,
    ) -> None:
        # These two parameters form the definition of the quaternion. self.s is a scalar associated with the
        # real component of the quaternion, while self.vec is the vector, associated with i, j, k / x, y, z components
        self.s: float = 1.0
        self.vec: NDArray = np.zeros((3,))
        self._cache4: Optional[np.ndarray] = None

        # Included for redundancy, if a quaternion is passed in, make a copy of its values
        if isinstance(quat, Quaternion):
            self.s = float(quat.s)
            self.vec = np.array(quat.vec, copy=True)
            # raise ValueError(f'Gave me a quaternion already, with {s = } and {vec = }')
            return

        #
        if s is None and vec is None and quat is not None:
            self.s = float(quat[0])
            self.vec = np.array(quat[1:4]).flatten()
            self.checkUnit(makeUnitQuat)
            return

        if s is None and vec is None:
            self.s = 1.0
            self.vec = np.zeros((3,))
            self.checkUnit(makeUnitQuat)
            return

        if s is None:
            self.vec = self._coerce_vec3(vec)
            self.s = sqrt(1.0 - self.vec.dot(self.vec))
            self.checkUnit(makeUnitQuat)
            return

        if vec is None:
            if not isinstance(s, Real):
                raise ValueError('s should be a single float')
            self.s = float(s)
            self.vec = np.zeros((3,))
        else:
            self.s = float(s)
            self.vec = self._coerce_vec3(vec)
        self.checkUnit(makeUnitQuat)

    @staticmethod
    def _coerce_vec3(vec: NDArray) -> np.ndarray:
        if np.shape(vec) == (3,):
            return np.array(vec, copy=True)
        if np.shape(vec) == (3, 1) or np.shape(vec) == (1, 3):
            return np.array(vec, copy=True).flatten()
        raise ValueError('vec should be a (3,) or (3,1) or (1,3) numpy array')

    def _array_copy(self) -> np.ndarray:
        return np.array([self.s, self.vec[0], self.vec[1], self.vec[2]], dtype=np.float64)

    def checkUnit(self, makeUnitQuat: bool) -> None:
        if makeUnitQuat:
            norm: float = self.norm
            if abs(norm) < _FLOAT_EPS:
                raise ValueError('Cannot make zero-quaternion a unit.')
            self.s /= norm
            self.vec /= norm

    def __str__(self) -> str:
        strg: str = ''
        if self.s < 0:
            strg += f'[{self.s:.10f}, <'
        else:
            strg += f'[ {abs(self.s):.10f}, <'
        if self.vec[0] < 0:
            strg += f'{self.vec[0]:.10f}, '
        else:
            strg += f' {abs(self.vec[0]):.10f}, '
        if self.vec[1] < 0:
            strg += f'{self.vec[1]:.10f}, '
        else:
            strg += f' {abs(self.vec[1]):.10f}, '
        if self.vec[2] < 0:
            strg += f'{self.vec[2]:.10f}>]'
        else:
            strg += f' {abs(self.vec[2]):.10f}>]'
        return strg

    def __format__(self, format_spec: str) -> str:
        def plus_or_minus(val: float, ijk_spec: str = '.3f') -> str:
            return f' - {-val:{ijk_spec}}' if val < 0.0 else f' + {val:{ijk_spec}}'

        if format_spec[:3] == 'ijk':

            if len(format_spec) == 3:
                return f'{self.s:.3f}{plus_or_minus(self.vec[0])}i{plus_or_minus(self.vec[1])}j{plus_or_minus(self.vec[2])}k'

            ijk_format = format_spec[3:]
            return f'{self.s:{ijk_format}}{plus_or_minus(self.vec[0], ijk_format)}i{plus_or_minus(self.vec[1], ijk_format)}j{plus_or_minus(self.vec[2], ijk_format)}k'
        else:
            return self.__str__()

    def __repr__(self) -> str:
        return self.__str__()

    def __xor__(self, scalar: float):
        return self.power(scalar)

    # def __format__(self, format_spec):
    #     if format_spec.startswith("."):
    #         precision = int(format_spec[1:])
    #         np.set_printoptions(precision=precision)
    #     return self.__str__

    def __truediv__(self, divisor: object) -> Self | NotImplementedType:
        if isinstance(divisor, Real):
            return Quaternion(quat=self.ndarray / float(divisor), makeUnitQuat=False)
        return NotImplemented

    def __sub__(self, subtractor: object) -> Self | NotImplementedType:
        if isinstance(subtractor, np.ndarray) and subtractor.shape == (4,):
            return Quaternion(quat=self.ndarray - subtractor, makeUnitQuat=False)

        if isinstance(subtractor, Quaternion):
            return Quaternion(quat=self.ndarray - subtractor.ndarray, makeUnitQuat=False)
        return NotImplemented

    def __add__(self, other: Self) -> Self:
        return Quaternion(quat=np.array([self.s + other.s,
                                         self.vec[0] + other.vec[0],
                                         self.vec[1] + other.vec[1],
                                         self.vec[2] + other.vec[2]]), makeUnitQuat=False)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Quaternion):
            return False

        if abs(self.s - other.s) < _FLOAT_EPS and np.linalg.norm(self.vec - other.vec) < _FLOAT_EPS:
            return True

        # A negative quaternion is equivalent to it's positive: -q = q
        if abs(self.s + other.s) < _FLOAT_EPS and np.linalg.norm(self.vec + other.vec) < _FLOAT_EPS:
            return True

        return False

    # NEP-18 hook: intercept np.matmul(A, q) when q is a Quaternion
    @staticmethod
    def __array_function__(func: Any, types: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        if func is np.matmul:
            A, B = args
            # cases: A @ q   or   q @ B (you can support both if you like)
            if isinstance(B, Quaternion):
                return B.__rmatmul__(A)  # defer to our rmatmul
            # (Optional) support q @ A if you want:
            if isinstance(A, Quaternion):
                # define your left-matmul here if desired
                return A @ B
        return NotImplemented

    def __rmul__(self, other: object) -> Self | NotImplementedType:
        if isinstance(other, Real):
            return self * other
        return NotImplemented

    def __rmatmul__(self, other: object) -> np.ndarray | NotImplementedType:
        if isinstance(other, np.ndarray):
            if other.shape[0] == 4:
                going_out = np.zeros(other.T.shape)
                for idx, quat in enumerate(other):
                    going_out[idx] = (Quaternion(quat=quat, makeUnitQuat=False).__mul__(self)).ndarray
                return going_out
        return NotImplemented

    def __matmul__(self, multiplier: object) -> np.ndarray | Self:
        return self * multiplier

    def __mul__(self, multiplier: object) -> np.ndarray | Self:
        """
        "*" Operator override:
        If multiplier is a 3x1 np.array, treat it like a quat-vect multiplication
            return 3x1 np.array vector
        If multiplier is a 4x1 np.array, treat it like a quat-quat multiplication,
            return 4x1 np.array
        If multiplier is a 3x3 np.array, treat it like multiple quat-vect multiplication,
            return 3x3 np.array
        If multiplier is a 3xN np.array, treat it like a series of vectors, each to be
            rotated by the quaternion and then returned as a 3xN np.array
        If multiplier is a Nx3 np.array, treat it like a series of vectors, each to be
        rotated by the quaternion and then returned as a Nx3 np.array
        If multiplier is another Quaternion object, treat it like quat-quat,
            return 4x1 np.array
        """
        if isinstance(multiplier, np.ndarray):
            if multiplier.ndim == 1 and multiplier.shape == (3,):
                return self.qv_mult(multiplier)
            if multiplier.ndim == 1 and multiplier.shape == (4,):
                return self.qn_mult(multiplier)
            if multiplier.ndim == 2 and multiplier.shape[1] == 4:
                return self.qQs_mult(multiplier)
            if multiplier.ndim == 2 and multiplier.shape == (3, 3):
                return self.qM_mult(multiplier)
            if multiplier.ndim == 2 and multiplier.shape[1] == 3:
                return self.qVECS_mult(multiplier)
            if multiplier.ndim == 2 and multiplier.shape[0] == 3:
                return self.qVECS_mult(multiplier.T).T
            raise ValueError(f'Bad multiplier, unknown object: {multiplier}')
        if isinstance(multiplier, Quaternion):
            return Quaternion(quat=self.qq_mult(multiplier), makeUnitQuat=False)
        if isinstance(multiplier, Real):
            scalar = float(multiplier)
            return Quaternion(s=scalar * self.s, vec=scalar * self.vec, makeUnitQuat=False)
        raise ValueError(f'Bad multiplier, unknown object: {multiplier}')

    def specializedQuatDiff(self, quat: Self) -> np.ndarray | Self:
        return self.T * quat

    def normalize(self) -> Self:
        norm: float = self.norm
        self.s /= norm
        self.vec /= norm
        return self

    def qVECS_mult(self, vecs: NDArray) -> np.ndarray:
        """
        Rotate an Nx3 array of vectors by this quaternion.
        Uses a Numba JIT kernel when available; otherwise falls back to Python loop.
        """
        vecs = np.asarray(vecs)

        # Common case: Nx3
        if vecs.ndim == 2 and vecs.shape[1] == 3:
            # Ensure dtype/contiguous once (avoid per-call copies elsewhere)
            v = vecs
            if v.dtype != np.float64 or not v.flags["C_CONTIGUOUS"]:
                v = np.ascontiguousarray(v, dtype=np.float64)

            return rotate_vecs_quat_numba(float(self.s),
                                          float(self.vec[0]),
                                          float(self.vec[1]),
                                          float(self.vec[2]),
                                          v)

        # Fallback: original behavior
        sol = np.zeros(vecs.shape, dtype=float)
        for idx, vec in enumerate(vecs):
            sol[idx] = self.qv_mult(vec)
        return sol

    def vect_deriv(self, vect: NDArray, isQuatConjugated: bool) -> np.ndarray:
        """
        Important note! Finding the quaternion partial derivatives with respect to a quaternion that is transposed is
        an entirely different operation!! Be careful when using this function.
        The derivative quaternion MUST NOT be transposed, and MUST BE handled with the boolean entry.
        Example: partial of q1.T * v1 with respect to q1 should be input as:
        q1.vect_deriv(v1, True)

        A BETTER implementation would be to use this pure method exclusively for non-transposed quaternions
        For transposed quaternions, use:
        q1.transpose_vect_deriv(v1)

        >>> for _ in range(100):
        ...     q = randomQuat()
        ...     delt = 0.0000001
        ...     qs = Quaternion(s=q.s+delt, vec=q.vec)
        ...     qx = Quaternion(s=q.s, vec=q.vec + np.array([delt, 0.0, 0.0]))
        ...     qy = Quaternion(s=q.s, vec=q.vec + np.array([0.0, delt, 0.0]))
        ...     qz = Quaternion(s=q.s, vec=q.vec + np.array([0.0, 0.0, delt]))
        ...     vecs = np.random.random((50,3))
        ...     for vec in vecs:
        ...         analy_deriv = q.vect_deriv(vec, True)
        ...         h0 = q.T * vec
        ...         hs = qs.T * vec
        ...         hx = qx.T * vec
        ...         hy = qy.T * vec
        ...         hz = qz.T * vec
        ...         np.testing.assert_allclose(analy_deriv, np.column_stack([(hs-h0)/delt, (hx-h0)/delt, (hy-h0)/delt, (hz-h0)/delt]),
        ...                                    atol=0.0001, rtol=0.0001)
        ...     q = randomQuat()
        ...     delt = 0.0000001
        ...     qs = Quaternion(s=q.s+delt, vec=q.vec)
        ...     qx = Quaternion(s=q.s, vec=q.vec + np.array([delt, 0.0, 0.0]))
        ...     qy = Quaternion(s=q.s, vec=q.vec + np.array([0.0, delt, 0.0]))
        ...     qz = Quaternion(s=q.s, vec=q.vec + np.array([0.0, 0.0, delt]))
        ...     vecs = np.random.random((50,3))
        ...     for vec in vecs:
        ...         analy_deriv = q.vect_deriv(vec, False)
        ...         h0 = q * vec
        ...         hs = qs * vec
        ...         hx = qx * vec
        ...         hy = qy * vec
        ...         hz = qz * vec
        ...         np.testing.assert_allclose(analy_deriv, np.column_stack([(hs-h0)/delt, (hx-h0)/delt, (hy-h0)/delt, (hz-h0)/delt]),
        ...                                    atol=0.0001, rtol=0.0001)
        """
        # This one can be a little tricky. These produce four different answers:
        # q.vectDeriv(vec, False)
        # q.T.vectDeriv(vec, False) <= Invalid!
        # q.T.vectDeriv(vec, True) <= Invalid!
        # q.vectDeriv(vec, True)  ... Valid, but better is:
        # q.transpose_vect_deriv(vec)

        '''
        This function returns the Jacobian in 3x4, presuming you are taking the partial derivative
        of composition q * v, with respect to q. 
        Columns are (left to right) qs, qx, qy, qz. 
        Rows are (top to bottom) x, y, z
        '''

        if not isQuatConjugated:
            quat = self.copy()
        else:
            quat = self.T.copy()

        deriv = np.zeros((3, 4))

        # d_q0
        deriv[:, 0] = 2.0 * (quat.s * vect + np.cross(quat.vec, vect))

        # d_qx
        deriv[:, 1] = 2.0 * (np.array([quat.vec[0] * vect[0] + np.dot(quat.vec, vect),
                                       quat.vec[1] * vect[0],
                                       quat.vec[2] * vect[0]]) +
                             -quat.vec[0] * vect +
                             quat.s * np.array([0.0, -vect[2], vect[1]]))

        # d_qy
        deriv[:, 2] = 2.0 * (np.array([quat.vec[0] * vect[1],
                                       (quat.vec[1] * vect[1] + np.dot(quat.vec, vect)),
                                       quat.vec[2] * vect[1]]) +
                             -quat.vec[1] * vect +
                             quat.s * np.array([vect[2], 0.0, -vect[0]]))
        # d_qz
        deriv[:, 3] = 2.0 * (np.array([quat.vec[0] * vect[2],
                                       quat.vec[1] * vect[2],
                                       (quat.vec[2] * vect[2] + np.dot(quat.vec, vect))]) +
                             -quat.vec[2] * vect +
                             quat.s * np.array([-vect[1], vect[0], 0.0]))

        if isQuatConjugated:
            deriv[:, 1:] = -deriv[:, 1:]
            P = quat.T.normal_plane_projection
        else:
            P = quat.normal_plane_projection
        return deriv @ P

    def transpose_vect_deriv(self, vect: NDArray) -> np.ndarray:
        '''
        Tiny helper, that helps perform the transpose derivative without mistakes.
        partial ( q.T * vec) / partial (q) may now be written:
                q.transpose_vect_deriv(vec)
        instead of
                q.T.vect_deriv(vec, True)
        '''
        return self.vect_deriv(vect, True)

    def to_dcm(self) -> np.ndarray:
        return quat2mat(self.ndarray)

    def qq_mult(self, multQuat: Self) -> np.ndarray:
        a = self._ndarray_view()
        b = multQuat._ndarray_view()
        return qmul_numba(a, b)

    def qn_mult(self, multNdarray: NDArray) -> np.ndarray:
        a = self._ndarray_view()
        b = np.asarray(multNdarray, dtype=np.float64).reshape(4, )
        return qmul_numba(a, b)

    def qQs_mult(self, QsNdarray: NDArray) -> np.ndarray:
        a = self._ndarray_view()
        Bs = np.asarray(QsNdarray, dtype=np.float64)
        if not Bs.flags["C_CONTIGUOUS"]:
            Bs = np.ascontiguousarray(Bs)
        return qmul_batch_left_numba(a, Bs)

    def qv_mult(self, multVec: NDArray) -> np.ndarray:
        return (2 * np.dot(self.vec, multVec) * self.vec +
                (self.s ** 2 - np.dot(self.vec, self.vec)) * multVec +
                2 * self.s * np.cross(self.vec, multVec))

    def qM_mult(self, multMat: NDArray) -> np.ndarray:
        solution = np.zeros((3, 3))
        solution[:, 0] = self.qv_mult(multMat[:, 0])
        solution[:, 1] = self.qv_mult(multMat[:, 1])
        solution[:, 2] = self.qv_mult(multMat[:, 2])
        return solution

    def qv_mult_alt(self, multVec: NDArray) -> np.ndarray:
        t = 2.0 * np.cross(self.vec, multVec)
        return multVec + self.s * t + np.cross(self.vec, t)

    def copy(self) -> Self:
        return Quaternion(s=float(self.s), vec=self.vec.copy(), makeUnitQuat=False)

    def force_s_pos(self) -> Self:
        if self.s < 0:
            self.s *= -1.0
            self.vec *= -1.0
        return self

    @property
    def T(self) -> Self:
        return Quaternion(self.s, -self.vec, makeUnitQuat=False)

    @property
    def inv(self) -> Self:
        mag_sq = self.mag ** 2
        return Quaternion(self.s / mag_sq, -self.vec / mag_sq, makeUnitQuat=False)

    @property
    def ndarray(self) -> np.ndarray:
        return self._array_copy()

    @property
    def nparray(self) -> np.ndarray:
        return self._array_copy()

    @property
    def array(self) -> np.ndarray:
        return self._array_copy()

    def _ndarray_view(self) -> np.ndarray:
        # DO NOT USE OUTSIDE CLASS
        # MUTABLE SCRATCH BUFFER
        # RUNS CODE 50-100% FASTER, BUT USER COULD MODIFY QUAT UNINTENTIONALLY
        if self._cache4 is None:
            self._cache4 = np.empty(4, dtype=np.float64)
        self._cache4[0] = self.s
        self._cache4[1:] = self.vec
        return self._cache4

    @property
    def conj(self) -> Self:
        return self.T

    @property
    def norm(self) -> float:
        return sqrt(self.s ** 2.0 + self.vec.dot(self.vec))

    @property
    def mag(self) -> float:
        return sqrt(self.s ** 2 + self.vec.dot(self.vec))

    @property
    def x(self) -> float:
        return self.vec[0]

    @property
    def y(self) -> float:
        return self.vec[1]

    @property
    def z(self) -> float:
        return self.vec[2]

    @property
    def rollR(self) -> float:
        return arctan2(2 * (self.s * self.x + self.y * self.z), 1 - 2 * (self.x * self.x + self.y * self.y))

    @property
    def rollD(self) -> float:
        return rad2deg(self.rollR)

    @property
    def pitchR(self) -> float:
        return arcsin(2 * (self.s * self.y - self.z * self.x))

    @property
    def pitchD(self) -> float:
        return rad2deg(self.pitchR)

    @property
    def yawR(self) -> float:
        return arctan2(2 * (self.s * self.z + self.x * self.y), 1 - 2 * (self.y * self.y + self.z * self.z))

    @property
    def yawD(self) -> float:
        return rad2deg(self.yawR)

    def from_eulerD_rpy(self, rpy: NDArray) -> "Quaternion":
        self.from_eulerR_rpy(deg2rad(rpy))
        return self

    def from_eulerR_rpy(self, rpy: NDArray) -> Self:
        # half angles
        rol = rpy[0] / 2.0
        ptc = rpy[1] / 2.0
        yaw = rpy[2] / 2.0
        self.s = cos(rol) * cos(ptc) * cos(yaw) + sin(rol) * sin(ptc) * sin(yaw)
        self.vec[0] = sin(rol) * cos(ptc) * cos(yaw) - cos(rol) * sin(ptc) * sin(yaw)
        self.vec[1] = cos(rol) * sin(ptc) * cos(yaw) + sin(rol) * cos(ptc) * sin(yaw)
        self.vec[2] = cos(rol) * cos(ptc) * sin(yaw) - sin(rol) * sin(ptc) * cos(yaw)
        return self

    def eulerR(self, order: str = 'rpy') -> NDArray:
        going_out = []
        for char in order:
            match char:
                case 'r':
                    going_out.append(self.rollR)
                case 'p':
                    going_out.append(self.pitchR)
                case 'y':
                    going_out.append(self.yawR)
                case _:
                    raise ValueError("EulerR function may only take 'r', 'p', or 'y' as inputs for order.")
        return np.array(going_out)

    def eulerD(self, order: str = 'rpy') -> NDArray:
        return rad2deg(self.eulerR(order))

    def angle_betweenR(self, otherQuat: Self) -> float:
        cosVal = (self.s * otherQuat.s + self.vec.dot(otherQuat.vec)) / (self.norm * otherQuat.norm)
        if 1.0 < cosVal < 1.00001:
            return 0.0
        else:
            acosVal = 2.0 * arccos(cosVal)
            if acosVal > np.pi:
                return 2.0 * np.pi - acosVal
            return acosVal

    def angle_betweenD(self, otherQuat: Self) -> float:
        return 180.0 / np.pi * self.angle_betweenR(otherQuat)

    @property
    def exp(self) -> Self:
        vec_norm = np.linalg.norm(self.vec)
        if vec_norm > 0.00000001:
            return np.exp(self.s) * Quaternion(s=cos(vec_norm), vec=self.vec / vec_norm * sin(vec_norm),
                                               makeUnitQuat=False)
        return Quaternion(s=1.0, vec=np.zeros((3,)), makeUnitQuat=False)

    @property
    def ln(self) -> Self:
        if np.linalg.norm(self.vec) < 0.000001:
            return Quaternion(s=0.0, vec=np.zeros((3,)), makeUnitQuat=False)
        return Quaternion(s=np.log(self.norm), vec=self.vec / np.linalg.norm(self.vec) * np.acos(self.s / self.norm),
                          makeUnitQuat=False)

    def power(self, power: Real) -> Self:
        if not isinstance(power, float):
            power = float(power)
        return (power * self.ln).exp.normalize()

    @staticmethod
    def from_axis_angle(axis: np.ndarray, angle: float) -> Self:
        axis = np.asarray(axis, dtype=float)
        axis /= np.linalg.norm(axis)
        half = 0.5 * angle
        return Quaternion(s=np.cos(half), vec=axis * np.sin(half))

    def to_axis_angle(self) -> tuple[np.ndarray, float]:
        if np.linalg.norm(self.vec) < 1e-12:
            return np.array([1.0, 0.0, 0.0]), 0.0
        axis = self.vec / np.linalg.norm(self.vec)
        angle = 2.0 * np.arctan2(np.linalg.norm(self.vec), self.s)
        return axis, angle

    @staticmethod
    def from_rodrigues(rod_vec: np.ndarray) -> Self:
        r = np.asarray(rod_vec, dtype=float).reshape(3)
        theta = float(np.linalg.norm(r))
        if theta < 1e-12:
            # Zero rotation
            return Quaternion(s=1.0, vec=np.zeros(3))
        axis = r / theta
        half = 0.5 * theta
        return Quaternion(s=np.cos(half), vec=axis * np.sin(half))

    def to_rodrigues(self) -> np.ndarray:
        # Ensure unit (or close)
        w = float(self.s)
        v = np.asarray(self.vec, dtype=float).reshape(3)
        vnorm = float(np.linalg.norm(v))
        # angle = 2 * atan2(||v||, w); axis = v/||v||
        if vnorm < 1e-12:
            return np.zeros(3)
        angle = 2.0 * np.arctan2(vnorm, max(1e-16, w))
        axis = v / vnorm
        return axis * angle

    @staticmethod
    def fromOpenCV_toAftr_rvec(rvec: NDArray, tvec: NDArray) -> tuple[Self, np.ndarray]:

        q_CV_TO_AFTR = mat2quat(np.array([[0., 0., 1.],
                                          [-1., 0., 0.],
                                          [0., -1., 0.]], float))

        rod_quat = q_CV_TO_AFTR * Quaternion.from_rodrigues(rvec)
        new_t = q_CV_TO_AFTR * tvec
        return rod_quat, np.squeeze(new_t)

    def slerp(self, q2: Self, t: Real) -> Self:
        return self * (self.inv * q2).power(t)

    @property
    def normal_plane_projection(self) -> np.ndarray:
        return np.eye(4) - np.outer(self.ndarray, self.ndarray)

    @property
    def inplace_deriv(self) -> np.ndarray:
        return self.normal_plane_projection

    @property
    def T_inplace_deriv(self) -> np.ndarray:
        P = self.T.normal_plane_projection
        P[1:] = -P[1:]
        return P

    def to_SE3_given_position(self, position: NDArray) -> NDArray:
        going_out = np.eye(4)
        going_out[:3, :3] = self.to_dcm()
        going_out[:3, 3] = position
        return going_out

    def perturb_from_rodrigues_std(self, std: NDArray) -> "Quaternion":
        dtheta = std * np.random.randn(3)
        perturb_q = Quaternion().from_rodrigues(dtheta)
        return perturb_q * self

    def perturb_from_rodrigues_var(self, var: NDArray) -> "Quaternion":
        return self.perturb_from_rodrigues_std(np.sqrt(var))


pure_qs = Quaternion(s=1.0, vec=np.zeros((3,)))
pure_qx = Quaternion(s=0.0, vec=np.array([1.0, 0.0, 0.0]))
pure_qy = Quaternion(s=0.0, vec=np.array([0.0, 1.0, 0.0]))
pure_qz = Quaternion(s=0.0, vec=np.array([0.0, 0.0, 1.0]))


def from_SE3(Mat4: NDArray):
    if Mat4.shape != (4, 4):
        raise ValueError(f"Mat4 must be of shape (4,4), but is {Mat4.shape}")
    dcm = Mat4[:3, :3]
    pos = Mat4[:3, 3]
    return mat2quat(dcm), pos


def skew(v: np.ndarray) -> np.ndarray:
    """Return [v]_x such that [v]_x @ a = v x a."""
    v = np.asarray(v, dtype=float).reshape(3)
    x, y, z = v
    return np.array([[0.0, -z, y],
                     [z, 0.0, -x],
                     [-y, x, 0.0]], dtype=float)


def so3_left_jacobian(phi: np.ndarray) -> np.ndarray:
    """
    Left Jacobian J_l(phi) for SO(3), phi is 3-vector rotation vector.
      J = I - (1-cosθ)/θ^2 [φ]_x + (θ - sinθ)/θ^3 [φ]_x^2
    with stable small-angle series.
    """
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = float(np.linalg.norm(phi))
    I3 = np.eye(3)
    if theta < 1e-8:
        # Series: I - 1/2 Φ + 1/6 Φ^2 + O(θ^3)
        Phi = skew(phi)
        return I3 - 0.5 * Phi + (1.0 / 6.0) * (Phi @ Phi)

    Phi = skew(phi)
    a = (1.0 - np.cos(theta)) / (theta * theta)
    b = (theta - np.sin(theta)) / (theta * theta * theta)
    return I3 - a * Phi + b * (Phi @ Phi)


def interpolate(q1: Quaternion, q2: Quaternion, t: float):
    interp: Quaternion = q1 * (q1.inv * q2).power(t)
    return interp


def randomQuat():
    u1, u2, u3 = np.random.rand(3)
    q = Quaternion(quat=np.array([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3),
    ]))
    return q


def random_quat_within_deg(theta_max_deg: float):
    theta_max = np.deg2rad(theta_max_deg)

    # random axis uniformly on S^2
    v = np.random.normal(size=3)
    v /= np.linalg.norm(v)

    # angle distributed for uniform measure in SO(3) restricted to theta <= theta_max
    r = np.random.rand()
    theta = 2.0 * np.arcsin((r ** (1.0 / 3.0)) * np.sin(theta_max / 2.0))

    s = np.sin(theta / 2.0)
    c = np.cos(theta / 2.0)

    # Your class appears to be (x, y, z, w) from the code shown
    return Quaternion(quat=np.array([c, v[0] * s, v[1] * s, v[2] * s]))


def left_quat_productDeriv(quatL, quatR, isTargetConjugated):
    """
    Important note! Finding the quaternion partial derivatives with respect to a quaternion that is transposed is
    an entirely different operation!! Be careful when using this function.
    The derivative quaternion should NOT be transposed, and instead must be handled with the boolean entry.
    Other quaternions MUST be input as their transpose, if they are transposed in the original equation.
    Example: partial of q1.T * q2.T with respect to q1 should be input as:
    left_quat_productDeriv(q1, q2.T, True)
    >>> for i in range(100):
    ...     q1 = randomQuat()
    ...     q2 = randomQuat()
    ...     q0 = q1 * q2
    ...     delt = 0.00000001
    ...     qs = Quaternion(s=q1.s + delt, vec=q1.vec) * q2
    ...     qx = Quaternion(s=q1.s, vec=q1.vec + np.array([delt, 0, 0])) * q2
    ...     qy = Quaternion(s=q1.s, vec=q1.vec + np.array([0, delt, 0])) * q2
    ...     qz = Quaternion(s=q1.s, vec=q1.vec + np.array([0, 0, delt])) * q2
    ...     analy_deriv = left_quat_productDeriv(q1, q2, False)
    ...     np.testing.assert_allclose(analy_deriv, np.array([(qs-q0).ndarray/delt,(qx-q0).ndarray/delt,(qy-q0).ndarray/delt,(qz-q0).ndarray/delt]).T, rtol=1e-05, atol=1e-05)
    >>> for i in range(100):
    ...     q1 = randomQuat()
    ...     q2 = randomQuat()
    ...     q0 = q1.T * q2
    ...     delt = 0.00000001
    ...     qs = Quaternion(s=q1.s + delt, vec=q1.vec).T * q2
    ...     qx = Quaternion(s=q1.s, vec=q1.vec + np.array([delt, 0, 0])).T * q2
    ...     qy = Quaternion(s=q1.s, vec=q1.vec + np.array([0, delt, 0])).T * q2
    ...     qz = Quaternion(s=q1.s, vec=q1.vec + np.array([0, 0, delt])).T * q2
    ...     analy_deriv = left_quat_productDeriv(q1, q2, True)
    ...     np.testing.assert_allclose(analy_deriv, np.array([(qs-q0).ndarray/delt,(qx-q0).ndarray/delt,(qy-q0).ndarray/delt,(qz-q0).ndarray/delt]).T, rtol=1e-05, atol=1e-05)

    :param quatL:
    :param quatR:
    :param isTargetConjugated:
    :return:
    """
    if isTargetConjugated:
        return (quatL.T_inplace_deriv @ quatR).T
    else:
        return (quatL.inplace_deriv @ quatR).T


def right_quat_productDeriv(quatL, quatR, isTargetConjugated):
    """
    Important note! Finding the quaternion partial derivatives with respect to a quaternion that is transposed is
    an entirely different operation!! Be careful when using this function.
    The derivative quaternion should NOT be transposed, and instead must be handled with the boolean entry.
    Other quaternions MUST be input as their transpose, if they are transposed in the original equation.
    Example: partial of q1.T * q2.T with respect to q2 should be input as:
    right_quat_productDeriv(q1.T, q2, True)

    >>> for i in range(100):
    ...     q1 = randomQuat()
    ...     q2 = randomQuat()
    ...     q0 = q1 * q2
    ...     delt = 0.00000001
    ...     qs = q1 * Quaternion(s=q2.s + delt, vec=q2.vec)
    ...     qx = q1 * Quaternion(s=q2.s, vec=q2.vec + np.array([delt, 0, 0]))
    ...     qy = q1 * Quaternion(s=q2.s, vec=q2.vec + np.array([0, delt, 0]))
    ...     qz = q1 * Quaternion(s=q2.s, vec=q2.vec + np.array([0, 0, delt]))
    ...     analy_deriv = right_quat_productDeriv(q1, q2, False)
    ...     np.testing.assert_allclose(analy_deriv, np.array([(qs-q0).ndarray/delt,(qx-q0).ndarray/delt,(qy-q0).ndarray/delt,(qz-q0).ndarray/delt]).T, rtol=1e-05, atol=1e-05)
    >>> for i in range(100):
    ...     q1 = randomQuat()
    ...     q2 = randomQuat()
    ...     q0 = q1 * q2.T
    ...     delt = 0.00000001
    ...     qs = q1 * Quaternion(s=q2.s + delt, vec=q2.vec).T
    ...     qx = q1 * Quaternion(s=q2.s, vec=q2.vec + np.array([delt, 0, 0])).T
    ...     qy = q1 * Quaternion(s=q2.s, vec=q2.vec + np.array([0, delt, 0])).T
    ...     qz = q1 * Quaternion(s=q2.s, vec=q2.vec + np.array([0, 0, delt])).T
    ...     analy_deriv = right_quat_productDeriv(q1, q2, True)
    ...     np.testing.assert_allclose(analy_deriv, np.array([(qs-q0).ndarray/delt,(qx-q0).ndarray/delt,(qy-q0).ndarray/delt,(qz-q0).ndarray/delt]).T, rtol=1e-05, atol=1e-05)

    :param quatL:
    :param quatR:
    :param isTargetConjugated:
    :return:
    """
    if isTargetConjugated:
        return (quatL @ quatR.T_inplace_deriv).T
    else:
        return (quatL @ quatR.inplace_deriv).T


def tri_quat_productDeriv(quat1, quat2, quat3, idx, isTargetConjugated):
    """
    Important note! Finding the quaternion partial derivatives with respect to a quaternion that is transposed is
    an entirely different operation!! Be careful when using this function.
    The derivative quaternion should NOT be transposed, and instead must be handled with the boolean entry.
    Other quaternions MUST be input as their transpose, if they are transposed in the original equation.
    Example: partial of q1 * q2.T * q3.T with respect to q2 should be input as:
    tri_quat_productDeriv(q1, q2, q3.T, 'm', True)

    >>> for i in range(100):
    ...     q0 = randomQuat()
    ...     q1 = randomQuat()
    ...     q2 = randomQuat()
    ...     delt = 0.0000001
    ...     mult0 = q0 * q1 * q2.T
    ...     mults = q0 * Quaternion(s=q1.s + delt, vec=q1.vec) * q2.T
    ...     multx = q0 * Quaternion(s=q1.s, vec=q1.vec + np.array([delt, 0, 0])) * q2.T
    ...     multy = q0 * Quaternion(s=q1.s, vec=q1.vec + np.array([0, delt, 0])) * q2.T
    ...     multz = q0 * Quaternion(s=q1.s, vec=q1.vec + np.array([0, 0, delt])) * q2.T
    ...
    ...     numDeriv = np.zeros((4,4))
    ...
    ...     numDeriv[:,0] = ((mults - mult0)/delt).ndarray
    ...     numDeriv[:,1] = ((multx - mult0)/delt).ndarray
    ...     numDeriv[:,2] = ((multy - mult0)/delt).ndarray
    ...     numDeriv[:,3] = ((multz - mult0)/delt).ndarray
    ...
    ...     np.testing.assert_allclose(tri_quat_productDeriv(q0, q1, q2.T, 'm', False), numDeriv, rtol=1e-05, atol=1e-05)
    >>> for i in range(100):
    ...     q0 = randomQuat()
    ...     q1 = randomQuat()
    ...     q2 = randomQuat()
    ...     delt = 0.0000001
    ...     mult0 = q0 * q1.T * q2
    ...     mults = q0 * Quaternion(s=q1.s + delt, vec=q1.vec).T * q2
    ...     multx = q0 * Quaternion(s=q1.s, vec=q1.vec + np.array([delt, 0, 0])).T * q2
    ...     multy = q0 * Quaternion(s=q1.s, vec=q1.vec + np.array([0, delt, 0])).T * q2
    ...     multz = q0 * Quaternion(s=q1.s, vec=q1.vec + np.array([0, 0, delt])).T * q2
    ...
    ...     numDeriv = np.zeros((4,4))
    ...
    ...     numDeriv[:,0] = ((mults - mult0)/delt).ndarray
    ...     numDeriv[:,1] = ((multx - mult0)/delt).ndarray
    ...     numDeriv[:,2] = ((multy - mult0)/delt).ndarray
    ...     numDeriv[:,3] = ((multz - mult0)/delt).ndarray
    ...
    ...     np.testing.assert_allclose(tri_quat_productDeriv(q0, q1, q2, 'm', True), numDeriv, rtol=1e-05, atol=1e-05)
    >>> for i in range(100):
    ...     q0 = randomQuat()
    ...     q1 = randomQuat()
    ...     q2 = randomQuat()
    ...     delt = 0.0000001
    ...     mult0 = q0 * q1 * q2
    ...     mults = Quaternion(s=q0.s + delt, vec=q0.vec) * q1 * q2
    ...     multx = Quaternion(s=q0.s, vec=q0.vec + np.array([delt, 0, 0])) * q1 * q2
    ...     multy = Quaternion(s=q0.s, vec=q0.vec + np.array([0, delt, 0])) * q1 * q2
    ...     multz = Quaternion(s=q0.s, vec=q0.vec + np.array([0, 0, delt])) * q1 * q2
    ...
    ...     numDeriv = np.zeros((4,4))
    ...
    ...     numDeriv[:,0] = ((mults - mult0)/delt).ndarray
    ...     numDeriv[:,1] = ((multx - mult0)/delt).ndarray
    ...     numDeriv[:,2] = ((multy - mult0)/delt).ndarray
    ...     numDeriv[:,3] = ((multz - mult0)/delt).ndarray
    ...
    ...     np.testing.assert_allclose(tri_quat_productDeriv(q0, q1, q2, 'l', False), numDeriv, rtol=1e-05, atol=1e-05)
    >>> for i in range(100):
    ...     q0 = randomQuat()
    ...     q1 = randomQuat()
    ...     q2 = randomQuat()
    ...     delt = 0.0000001
    ...     mult0 = q0.T * q1 * q2
    ...     mults = Quaternion(s=q0.s + delt, vec=q0.vec).T * q1 * q2
    ...     multx = Quaternion(s=q0.s, vec=q0.vec + np.array([delt, 0, 0])).T * q1 * q2
    ...     multy = Quaternion(s=q0.s, vec=q0.vec + np.array([0, delt, 0])).T * q1 * q2
    ...     multz = Quaternion(s=q0.s, vec=q0.vec + np.array([0, 0, delt])).T * q1 * q2
    ...
    ...     numDeriv = np.zeros((4,4))
    ...
    ...     numDeriv[:,0] = ((mults - mult0)/delt).ndarray
    ...     numDeriv[:,1] = ((multx - mult0)/delt).ndarray
    ...     numDeriv[:,2] = ((multy - mult0)/delt).ndarray
    ...     numDeriv[:,3] = ((multz - mult0)/delt).ndarray
    ...
    ...     np.testing.assert_allclose(tri_quat_productDeriv(q0, q1, q2, 'l', True), numDeriv, rtol=1e-05, atol=1e-05)
    >>> for i in range(100):
    ...     q0 = randomQuat()
    ...     q1 = randomQuat()
    ...     q2 = randomQuat()
    ...     delt = 0.0000001
    ...     mult0 = q0 * q1 * q2
    ...     mults = q0 * q1 * Quaternion(s=q2.s + delt, vec=q2.vec)
    ...     multx = q0 * q1 * Quaternion(s=q2.s, vec=q2.vec + np.array([delt, 0, 0]))
    ...     multy = q0 * q1 * Quaternion(s=q2.s, vec=q2.vec + np.array([0, delt, 0]))
    ...     multz = q0 * q1 * Quaternion(s=q2.s, vec=q2.vec + np.array([0, 0, delt]))
    ...
    ...     numDeriv = np.zeros((4,4))
    ...
    ...     numDeriv[:,0] = ((mults - mult0)/delt).ndarray
    ...     numDeriv[:,1] = ((multx - mult0)/delt).ndarray
    ...     numDeriv[:,2] = ((multy - mult0)/delt).ndarray
    ...     numDeriv[:,3] = ((multz - mult0)/delt).ndarray
    ...
    ...     np.testing.assert_allclose(tri_quat_productDeriv(q0, q1, q2, 'r', False), numDeriv, rtol=1e-05, atol=1e-05)

    :param quat1: q0 in quaternion multiplication q0 * q1 * q2
    :param quat2: q1 in quaternion multiplication q0 * q1 * q2
    :param quat3: q2 in quaternion multiplication q0 * q1 * q2
    :return: Partial derivative of multiplication with respect to idx, 'l' for left, 'm' for middle', 'r' for right
    """
    deriv = np.zeros((4, 4))
    if idx == 'l':
        return left_quat_productDeriv(quat1, quat2 * quat3, isTargetConjugated)

    if idx == 'm':
        if isTargetConjugated:
            P = quat2.T.normal_plane_projection
        else:
            P = quat2.normal_plane_projection

        deriv = (quat1 @ P @ quat3).T
        if isTargetConjugated:
            deriv[:, 1:] = -deriv[:, 1:]
        return deriv

    if idx == 'r':
        return right_quat_productDeriv(quat1 * quat2, quat3, isTargetConjugated)

    raise ValueError('tri_quat_productDeriv idx should be \'l\', \'m\', or \'r\'')


def fillpositive(xyz, w2_thresh=None):
    """ Compute unit quaternion from last 3 values

    Parameters
    ----------
    xyz : iterable
       iterable containing 3 values, corresponding to quaternion x, y, z
    w2_thresh : None or float, optional
       threshold to determine if w squared is really negative.
       If None (default) then w2_thresh set equal to
       ``-np.finfo(xyz.dtype).eps``, if possible, otherwise
       ``-np.finfo(np.float64).eps``

    Returns
    -------
    wxyz : array shape (4,)
         Full 4 values of quaternion

    Notes
    -----
    If w, x, y, z are the values in the full quaternion, assumes w is
    positive.

    Gives error if w*w is estimated to be negative

    w = 0 corresponds to a 180 degree rotation

    The unit quaternion specifies that np.dot(wxyz, wxyz) == 1.

    If w is positive (assumed here), w is given by:

    w = sqrt(1.0-(x*x+y*y+z*z))

    w2 = 1.0-(x*x+y*y+z*z) can be near zero, which will lead to
    numerical instability in sqrt.  Here we use the system maximum
    float type to reduce numerical instability

    Examples
    --------
    >>> wxyz = fillpositive([0,0,0])
    >>> assert np.all(wxyz == [1, 0, 0, 0])
    >>> wxyz = fillpositive([1,0,0]) # Corner case; w is 0
    >>> assert np.all(wxyz == [0, 1, 0, 0])
    >>> assert np.dot(wxyz, wxyz) == 1
    """
    # Check inputs (force error if < 3 values)
    if len(xyz) != 3:
        raise ValueError('xyz should have length 3')
    # If necessary, guess precision of input
    if w2_thresh is None:
        try:  # trap errors for non-array, integer array
            w2_thresh = -np.finfo(xyz.dtype).eps * 3
        except (AttributeError, ValueError):
            w2_thresh = -_FLOAT_EPS * 3
    # Use maximum precision
    xyz = np.asarray(xyz, dtype=np.float64)
    # Calculate w
    w2 = 1.0 - np.dot(xyz, xyz)
    if w2 < 0:
        if w2 < w2_thresh:
            raise ValueError('w2 should be positive, but is %e' % w2)
        w = 0
    else:
        w = sqrt(w2)
    return np.r_[w, xyz]


def quat2mat(q):
    """ Calculate rotation matrix corresponding to quaternion

    Parameters
    ----------
    q : 4 element array-like

    Returns
    -------
    M : (3,3) array
      Rotation matrix corresponding to input quaternion *q*

    Notes
    -----
    Rotation matrix applies to column vectors, and is applied to the
    left of coordinate vectors.  The algorithm here allows quaternions that
    have not been normalized.

    References
    ----------
    Algorithm from http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion

    Examples
    --------
    >>> M = quat2mat([1, 0, 0, 0]) # Identity quaternion
    >>> np.allclose(M, np.eye(3))
    True
    >>> M = quat2mat([0, 1, 0, 0]) # 180 degree rotn around axis 0
    >>> np.allclose(M, np.diag([1, -1, -1]))
    True
    """
    if isinstance(q, Quaternion):
        w = q.s
        x, y, z = q.vec
    else:
        w, x, y, z = q
    Nq = w * w + x * x + y * y + z * z
    if Nq < _FLOAT_EPS:
        return np.eye(3)
    s = 2.0 / Nq
    X = x * s
    Y = y * s
    Z = z * s
    wX = w * X
    wY = w * Y
    wZ = w * Z
    xX = x * X
    xY = x * Y
    xZ = x * Z
    yY = y * Y
    yZ = y * Z
    zZ = z * Z
    return np.array(
        [[1.0 - (yY + zZ), xY - wZ, xZ + wY],
         [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
         [xZ - wY, yZ + wX, 1.0 - (xX + yY)]])


def mat2quat(M):
    """Quaternion [w, x, y, z] from 3x3 rotation matrix M (robust Bar-Itzhack).
       Returns a unit quaternion with non-negative w.
    """

    M = np.asarray(M, dtype=float)
    assert M.shape == (3, 3)
    U, _, Vt = np.linalg.svd(M)
    M = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt

    m00, m01, m02 = M[0]
    m10, m11, m12 = M[1]
    m20, m21, m22 = M[2]
    tr = m00 + m11 + m22

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 - m00 + m11 - m22) * 2.0
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 - m00 - m11 + m22) * 2.0
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S

    q = np.array([w, x, y, z], dtype=float)
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return Quaternion(s=float(q[0]), vec=q[1:])


def se3s2quats(SE3s: list[NDArray]) -> np.ndarray:
    return np.array([mat2quat(M[:3, :3]).ndarray for M in SE3s])


def mats2quats(mats: NDArray) -> np.ndarray:
    """Convert array of Nx3x3 matrices to Nx4 quaternions."""
    return np.array([mat2quat(M).ndarray for M in mats])


def quats2mats(quats: np.ndarray) -> np.ndarray:
    """Convert array of Nx4 quaternions to Nx3x3 rotation matrices."""
    return np.array([quat2mat(q) for q in quats])


def qmult(q1, q2):
    """ Multiply two quaternions

    Parameters
    ----------
    q1 : 4 element sequence
    q2 : 4 element sequence

    Returns
    -------
    q12 : shape (4,) array

    Notes
    -----
    See : http://en.wikipedia.org/wiki/Quaternions#Hamilton_product
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    # if w < 0.0 and not allowNegative:
    #     return -np.array([w, x, y, z])
    # else:
    return np.array([w, x, y, z])


def qconjugate(q):
    """ Conjugate of quaternion

    Parameters
    ----------
    q : 4 element sequence
       w, i, j, k of quaternion

    >>> test = np.random.rand(4)-0.5
    >>> newtest = np.array([test[0], -test[1], -test[2], -test[3]])
    >>> np.allclose(newtest, qconjugate(test))
    True

    Returns
    -------
    conjq : array shape (4,)
       w, i, j, k of conjugate of `q`
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def qnorm(q):
    """ Return norm of quaternion

    Parameters
    ----------
    q : 4 element sequence
       w, i, j, k of quaternion

    Returns
    -------
    n : scalar
       quaternion norm

    Notes
    -----
    http://mathworld.wolfram.com/QuaternionNorm.html
    """
    return sqrt(q.dot(q))


def qisunit(q):
    """ Return True is this is very nearly a unit quaternion """
    return np.allclose(qnorm(q), 1)


def qinverse(q):
    """ Return multiplicative inverse of quaternion `q`

    Parameters
    ----------
    q : 4 element sequence
       w, i, j, k of quaternion

    Returns
    -------
    invq : array shape (4,)
       w, i, j, k of quaternion inverse
    """
    return qconjugate(q) / qnorm(q)


def qeye(dtype=np.float64):
    """ Return identity quaternion """
    return np.array([1.0, 0, 0, 0], dtype=dtype)
