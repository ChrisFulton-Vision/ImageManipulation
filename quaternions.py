"""Functions to operate on, or return, quaternions.

Quaternions here consist of 4 values ``w, x, y, z``, where ``w`` is the
real (scalar) part, and ``x, y, z`` are the complex (vector) part.

Note - rotation matrices here apply to column vectors, that is,
they are applied on the left of the vector.  For example:

>>> import numpy as np
>>> q = [0, 1, 0, 0] # 180 degree rotation around axis 0
>>> M = quat2mat(q) # from this module
>>> vec = np.array([1, 2, 3]).reshape((3,1)) # column vector
>>> tvec = np.dot(M, vec)

Terms used in function names:

* *mat* : array shape (3, 3) (3D non-homogenous coordinates)
* *aff* : affine array shape (4, 4) (3D homogenous coordinates)
* *quat* : quaternion shape (4,)
* *axangle* : rotations encoded by axis vector and angle scalar
"""

import numpy as np
from numpy import cos, arccos, sin, arcsin, arctan2, rad2deg, deg2rad, sqrt, abs
from typing_extensions import Self, Union
from copy import deepcopy

_FLOAT_EPS = np.finfo(np.float64).eps


class Quaternion:
    __array_priority__ = 10_000  # overrides numpy priority for right mult

    def __init__(self, s: float = None, vec: np.array = None, quat: np.array = None, makeUnitQuat: bool = True) -> None:
        # These two parameters form the definition of the quaternion. self.s is a scalar asscoiated with the
        # real component of the quaternion, while self.vec is the vector, associated with i, j, k / x, y, z components
        self.s: float = 1.0
        self.vec: np.array = np.zeros((3,))

        # Included for redundancy, if a quaternion is passed in, make a copy of its values
        if isinstance(quat, Quaternion):
            self.s = deepcopy(quat.s)
            self.vec = deepcopy(quat.vec)
            # raise ValueError(f'Gave me a quaternion already, with {s = } and {vec = }')
            return

        #
        if s is None and vec is None and quat is not None:
            self.s = deepcopy(quat[0])
            self.vec = deepcopy(quat[1:4]).flatten()
            self.checkUnit(makeUnitQuat)
            return

        if s is None and vec is None:
            self.s = 1.0
            self.vec = np.zeros((3,))
            self.checkUnit(makeUnitQuat)
            return

        if s is None:
            if np.shape(vec) == (3,):
                self.vec = deepcopy(vec)
            elif np.shape(vec) == (3, 1) or np.shape(vec) == (1, 3):
                self.vec = vec.flatten()
            else:
                raise ValueError('vec should be a (3,) or (3,1) or (1,3) numpy array')
            self.s = sqrt(1.0 - vec.dot(vec))
            self.checkUnit(makeUnitQuat)
            return

        if vec is None:
            if not type(s, float):
                raise ValueError('s should be a single float')
            self.s = deepcopy(s)
            self.vec = np.zeros((3,))
        else:
            self.s = s
            if np.shape(vec) == (3,):
                self.vec = deepcopy(vec)
            elif np.shape(vec) == (3, 1) or np.shape(vec) == (1, 3):
                self.vec = deepcopy(vec).flatten()
            else:
                raise ValueError('vec should be a (3,) or (3,1) or (1,3) numpy array')
        self.checkUnit(makeUnitQuat)

    def checkUnit(self, makeUnitQuat: bool):
        if makeUnitQuat:
            norm: float = self.norm
            if abs(norm) < 0.000001:
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

    def __xor__(self, scalar: float):
        print(scalar, self)
        return self.power(scalar)

    # def __format__(self, format_spec):
    #     if format_spec.startswith("."):
    #         precision = int(format_spec[1:])
    #         np.set_printoptions(precision=precision)
    #     return self.__str__

    def __truediv__(self, divisor: float) -> Self:
        if isinstance(divisor, float):
            return Quaternion(quat=self.ndarray / divisor, makeUnitQuat=False)

    def __sub__(self, subtractor: Union[np.ndarray, Self]) -> Self:
        if isinstance(subtractor, np.ndarray) and subtractor.shape == (4,):
            return Quaternion(quat=self.ndarray - subtractor, makeUnitQuat=False)

        elif isinstance(subtractor, Quaternion):
            return Quaternion(quat=self.ndarray - subtractor.ndarray, makeUnitQuat=False)

    def __add__(self, other: Self) -> Self:
        return Quaternion(quat=np.array([self.s + other.s,
                                         self.vec[0] + other.vec[0],
                                         self.vec[1] + other.vec[1],
                                         self.vec[2] + other.vec[2]]), makeUnitQuat=False)

    def __eq__(self, other: Self):
        if not isinstance(other, Quaternion):
            raise TypeError(f'Comparing two unlike objects. Self (Quaternion) and {type(other)}')

        if abs(self.s - other.s) < _FLOAT_EPS and np.linalg.norm(self.vec - other.vec) < _FLOAT_EPS:
            return True

        # A negative quaternion is equivalent to it's positive: -q = q
        if abs(self.s + other.s) < _FLOAT_EPS and np.linalg.norm(self.vec + other.vec) < _FLOAT_EPS:
            return True

        return False

    # NEP-18 hook: intercept np.matmul(A, q) when q is a Quaternion
    def __array_function__(self, func, types, args, kwargs):
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

    def __rmul__(self, other):
        if isinstance(other, float):
            return self * other

    def __rmatmul__(self, other):
        if isinstance(other, np.ndarray):
            if other.shape[0] == 4:
                going_out = np.zeros((other.T.shape))
                for idx, quat in enumerate(other):
                    going_out[idx] = (Quaternion(quat=quat,makeUnitQuat=False).__mul__(self)).ndarray
                return going_out


    def __matmul__(self, multiplier):
        return self * multiplier

    def __mul__(self, multiplier):
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
            if multiplier.shape == (3,):
                return self.qv_mult(multiplier)
            if multiplier.shape == (4,):
                return self.qn_mult(multiplier)
            if multiplier.shape[1] == 4:
                return self.qQs_mult(multiplier)
            if multiplier.shape == (3, 3):
                return self.qM_mult(multiplier)
            if multiplier.shape[1] == 3:
                return self.qVECS_mult(multiplier)
            if multiplier.shape[0] == 3:
                return self.qVECS_mult(multiplier.T).T
            else:
                raise ValueError(f'Bad multiplier, unknown object: {multiplier}')
        elif isinstance(multiplier, Quaternion):
            return Quaternion(quat=self.qq_mult(multiplier), makeUnitQuat=False)
        elif isinstance(multiplier, float):
            return Quaternion(s=multiplier * self.s, vec=multiplier * self.vec, makeUnitQuat=False)
        else:
            raise ValueError(f'Bad multiplier, unknown object: {multiplier}')

    def specializedQuatDiff(self, quat):
        return self.T * quat

    def normalize(self):
        norm: float = self.norm
        self.s /= norm
        self.vec /= norm
        return self

    def qVECS_mult(self, vecs: np.array):
        sol: np.array = np.zeros(vecs.shape)
        for idx, vec in enumerate(vecs):
            sol[idx] = self.qv_mult(vec)
        return sol

    def vect_deriv(self, vect: np.array, isQuatConjugated: bool):
        '''
        Important note! Finding the quaternion partial derivatives with respect to a quaternion that is transposed is
        an entirely different operation!! Be careful when using this function.
        The derivative quaternion MUST be transposed, and must be ALSO be handled with the boolean entry.
        Example: partial of q1.T * v1 with respect to q1 should be input as:
        q1.T.vect_deriv(v1, True)

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
        ...         analy_deriv = q.T.vect_deriv(vec, True)
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
        '''
        # This one can be a little tricky. These produce four different answers:
        # q.vectDeriv(vec, False)
        # q.T.vectDeriv(vec, False) <= Invalid!
        # q.vectDeriv(vec, True) <= Invalid!
        # q.T.vectDeriv(vec, True) ... Valid, but better is:
        # q.transpose_vect_deriv(vec)

        '''
        This function returns the Jacobian in 3x4, presuming you are taking the partial derivative
        of composition q * v, with respect to q. 
        Columns are (left to right) qs, qx, qy, qz. 
        Rows are (top to bottom) x, y, z
        '''

        deriv = np.zeros((3, 4))

        # d_q0
        deriv[:, 0] = 2.0 * (self.s * vect + np.cross(self.vec, vect))

        # d_qx
        deriv[:, 1] = 2.0 * (np.array([self.vec[0] * vect[0] + np.dot(self.vec, vect),
                                       self.vec[1] * vect[0],
                                       self.vec[2] * vect[0]]) +
                             -self.vec[0] * vect +
                             self.s * np.array([0.0, -vect[2], vect[1]]))

        # d_qy
        deriv[:, 2] = 2.0 * (np.array([self.vec[0] * vect[1],
                                       (self.vec[1] * vect[1] + np.dot(self.vec, vect)),
                                       self.vec[2] * vect[1]]) +
                             -self.vec[1] * vect +
                             self.s * np.array([vect[2], 0.0, -vect[0]]))
        # d_qz
        deriv[:, 3] = 2.0 * (np.array([self.vec[0] * vect[2],
                                       self.vec[1] * vect[2],
                                       (self.vec[2] * vect[2] + np.dot(self.vec, vect))]) +
                             -self.vec[2] * vect +
                             self.s * np.array([-vect[1], vect[0], 0.0]))

        if isQuatConjugated:
            deriv[:, 1:] = -deriv[:, 1:]
            P = self.T.normal_plane_projection
        else:
            P = self.normal_plane_projection
        return deriv @ P

    def transpose_vect_deriv(self, vect: np.array):
        '''
        Tiny helper, that helps perform the transpose derivative without mistakes.
        partial ( q.T * vec) / partial (q) may now be written:
                q.transpose_vect_deriv(vec)
        instead of
                q.T.vect_deriv(vec, True)
        '''
        return self.T.vect_deriv(vect, True)

    def to_dcm(self):
        return quat2mat(self.ndarray)

    def qq_mult(self, multQuat):
        return qmult(self.ndarray, multQuat.ndarray)

    def qn_mult(self, multNdarray):
        return qmult(self.ndarray, multNdarray)

    def qQs_mult(self, QsNdarray):
        going_out = np.zeros((QsNdarray.shape))
        for idx, quat_as_ndarray in enumerate(QsNdarray):
            going_out[idx] = qmult(self.ndarray, quat_as_ndarray)
        return going_out

    def qv_mult(self, multVec):
        return (2 * np.dot(self.vec, multVec) * self.vec +
                (self.s ** 2 - np.dot(self.vec, self.vec)) * multVec +
                2 * self.s * np.cross(self.vec, multVec))

    def qM_mult(self, multMat):
        solution = np.zeros((3, 3))
        solution[:, 0] = self.qv_mult(multMat[:, 0])
        solution[:, 1] = self.qv_mult(multMat[:, 1])
        solution[:, 2] = self.qv_mult(multMat[:, 2])
        return solution

    def qv_mult_alt(self, multVec):
        t = 2.0 * np.cross(self.vec, multVec)
        return multVec + self.s * t + np.cross(self.vec, t)

    def copy(self):
        return deepcopy(self)

    def force_s_pos(self):
        if self.s < 0:
            self.s *= -1.0
            self.vec *= -1.0
        return self

    @property
    def T(self):
        return Quaternion(self.s, -self.vec, makeUnitQuat=False)

    @property
    def inv(self):
        return Quaternion(self.s, -self.vec, makeUnitQuat=False) / self.mag ** 2

    @property
    def ndarray(self):
        return np.append(self.s, self.vec)

    @property
    def conj(self):
        return Quaternion(np.append(self.s, -self.vec))

    @property
    def norm(self):
        return sqrt(self.s ** 2.0 + self.vec.dot(self.vec))

    @property
    def mag(self):
        return sqrt(self.s ** 2 + self.vec.dot(self.vec))

    @property
    def x(self):
        return self.vec[0]

    @property
    def y(self):
        return self.vec[1]

    @property
    def z(self):
        return self.vec[2]

    @property
    def rollR(self):
        return arctan2(2 * (self.s * self.x + self.y * self.z), 1 - 2 * (self.x * self.x + self.y * self.y))

    @property
    def rollD(self):
        return rad2deg(self.rollR)

    @property
    def pitchR(self):
        return arcsin(2 * (self.s * self.y - self.z * self.x))

    @property
    def pitchD(self):
        return rad2deg(self.pitchR)

    @property
    def yawR(self):
        return arctan2(2 * (self.s * self.z + self.x * self.y), 1 - 2 * (self.y * self.y + self.z * self.z))

    @property
    def yawD(self):
        return rad2deg(self.yawR)

    def from_eulerD_rpy(self, rpy: np.array) -> None:
        self.from_eulerR_rpy(deg2rad(rpy))

    def from_eulerR_rpy(self, rpy: np.array) -> None:
        # half angles
        rol = rpy[0] / 2.0
        ptc = rpy[1] / 2.0
        yaw = rpy[2] / 2.0
        self.s = cos(rol) * cos(ptc) * cos(yaw) + sin(rol) * sin(ptc) * sin(yaw)
        self.vec[0] = sin(rol) * cos(ptc) * cos(yaw) - cos(rol) * sin(ptc) * sin(yaw)
        self.vec[1] = cos(rol) * sin(ptc) * cos(yaw) + sin(rol) * cos(ptc) * sin(yaw)
        self.vec[2] = cos(rol) * cos(ptc) * sin(yaw) - sin(rol) * sin(ptc) * cos(yaw)

    def eulerR(self, order: str = 'rpy') -> np.array:
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

    def eulerD(self, order: str = 'rpy') -> np.array:
        return rad2deg(self.eulerR(order))

    def angle_betweenR(self, otherQuat):
        cosVal = (self.s * otherQuat.s + self.vec.dot(otherQuat.vec)) / (self.norm * otherQuat.norm)
        if 1.0 < cosVal < 1.00001:
            return 0.0
        else:
            acosVal = 2.0 * arccos(cosVal)
            if acosVal > np.pi:
                return 2.0 * np.pi - acosVal
            return acosVal

    def angle_betweenD(self, otherQuat):
        return 180.0 / np.pi * self.angle_betweenR(otherQuat)

    @property
    def exp(self):
        if np.linalg.norm(self.vec) > 0.00000001:
            return np.exp(self.s) * Quaternion(s=cos(np.linalg.norm(self.vec)),
                                               vec=self.vec / np.linalg.norm(self.vec) * sin(np.linalg.norm(self.vec)))
        return Quaternion(s=1.0, vec=np.zeros((3,)))

    @property
    def ln(self):
        if np.linalg.norm(self.vec) < 0.000001:
            return Quaternion(s=0.0, vec=np.zeros((3,)))
        return Quaternion(s=np.log(self.norm), vec=self.vec / np.linalg.norm(self.vec) * np.acos(self.s / self.norm))

    def power(self, power: float):
        if not isinstance(power, float):
            power = float(power)
        return (power * self.ln).exp.normalize()

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
    def from_openCV_rvec(rvec: np.array, tvec: np.array):

        rod_quat = Quaternion.from_rodrigues(rvec)
        new_t = rod_quat.T * -tvec
        return rod_quat, np.squeeze(new_t)

    def slerp(self, q2: Self, t) -> Self:
        return self * (self.inv * q2).power(t)

    @property
    def normal_plane_projection(self):
        return np.eye(4) - np.outer(self.ndarray, self.ndarray)

    @property
    def inplace_deriv(self):
        return self.normal_plane_projection

    @property
    def T_inplace_deriv(self):
        P = self.T.normal_plane_projection
        P[1:] = -P[1:]
        return P


pure_qs = Quaternion(s=1.0, vec=np.zeros((3,)))
pure_qx = Quaternion(s=0.0, vec=np.array([1.0, 0.0, 0.0]))
pure_qy = Quaternion(s=0.0, vec=np.array([0.0, 1.0, 0.0]))
pure_qz = Quaternion(s=0.0, vec=np.array([0.0, 0.0, 1.0]))


def interpolate(q1: Quaternion, q2: Quaternion, t: float):
    interp: Quaternion = q1 * (q1.inv * q2).power(t)
    return interp


def randomQuat(unit=True):
    return Quaternion(quat=np.random.rand(4, ), makeUnitQuat=unit)


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
    ''' Compute unit quaternion from last 3 values

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
    >>> import numpy as np
    >>> wxyz = fillpositive([0,0,0])
    >>> assert np.all(wxyz == [1, 0, 0, 0])
    >>> wxyz = fillpositive([1,0,0]) # Corner case; w is 0
    >>> assert np.all(wxyz == [0, 1, 0, 0])
    >>> assert np.dot(wxyz, wxyz) == 1
    '''
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
    ''' Calculate rotation matrix corresponding to quaternion

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
    >>> import numpy as np
    >>> M = quat2mat([1, 0, 0, 0]) # Identity quaternion
    >>> np.allclose(M, np.eye(3))
    True
    >>> M = quat2mat([0, 1, 0, 0]) # 180 degree rotn around axis 0
    >>> np.allclose(M, np.diag([1, -1, -1]))
    True
    '''
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


def mat2quat_jumbled(M):
    ''' Calculate quaternion corresponding to given rotation matrix

    Method claimed to be robust to numerical errors in `M`.

    Constructs quaternion by calculating maximum eigenvector for matrix
    ``K`` (constructed from input `M`).  Although this is not tested, a maximum
    eigenvalue of 1 corresponds to a valid rotation.

    A quaternion ``q*-1`` corresponds to the same rotation as ``q``; thus the
    sign of the reconstructed quaternion is arbitrary, and we return
    quaternions with positive w (q[0]).

    See notes.

    Parameters
    ----------
    M : array-like
      3x3 rotation matrix

    Returns
    -------
    q : (4,) array
      closest quaternion to input matrix, having positive q[0]

    References
    ----------
    * http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    * Bar-Itzhack, Itzhack Y. (2000), "New method for extracting the
      quaternion from a rotation matrix", AIAA Journal of Guidance,
      Control and Dynamics 23(6):1085-1087 (Engineering Note), ISSN
      0731-5090

    Examples
    --------
    >>> import numpy as np
    >>> q = mat2quat(np.eye(3)) # Identity rotation
    >>> np.allclose(q.ndarray, [1, 0, 0, 0])
    True
    >>> q = mat2quat(np.diag([1, -1, -1]))
    >>> np.allclose(q.ndarray, [0, 1, 0, 0]) # 180 degree rotn around axis 0
    True

    Notes
    -----
    http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion

    Bar-Itzhack, Itzhack Y. (2000), "New method for extracting the
    quaternion from a rotation matrix", AIAA Journal of Guidance,
    Control and Dynamics 23(6):1085-1087 (Engineering Note), ISSN
    0731-5090

    '''
    # Qyx refers to the contribution of the y input vector component to
    # the x output vector component.  Qyx is therefore the same as
    # M[0,1].  The notation is from the Wikipedia article.
    Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz = M.flat
    # Fill only lower half of symmetric matrix
    K = np.array([
        [Qxx + Qyy + Qzz, 0, 0, 0],
        [Qyz - Qzy, Qxx - Qyy - Qzz, 0, 0],
        [Qzx - Qxz, Qyx + Qxy, Qyy - Qxx - Qzz, 0],
        [Qxy - Qyx, Qzx + Qxz, Qzy + Qyz, Qzz - Qxx - Qyy],
    ]
    ) / 3.0
    # Use Hermitian eigenvectors, values for speed
    vals, vecs = np.linalg.eigh(K)
    # Select largest eigenvector, reorder to x,y,z,w quaternion
    q = vecs[:, np.argmax(vals)]
    # Prefer quaternion with positive w
    # (q * -1 corresponds to same rotation as q)
    if q[3] < 0:
        q *= -1
    return Quaternion(s=q[3], vec=np.diag(np.array([1.0, 1.0, -1.0])) @ q[0:3][::-1], makeUnitQuat=True)


def qmult(q1, q2):
    ''' Multiply two quaternions

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
    '''
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
    ''' Conjugate of quaternion

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
    '''
    return np.array([q[0], -q[1], -q[2], -q[3]])


def qnorm(q):
    ''' Return norm of quaternion

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
    '''
    return sqrt(q.dot(q))


def qisunit(q):
    ''' Return True is this is very nearly a unit quaternion '''
    return np.allclose(qnorm(q), 1)


def qinverse(q):
    ''' Return multiplicative inverse of quaternion `q`

    Parameters
    ----------
    q : 4 element sequence
       w, i, j, k of quaternion

    Returns
    -------
    invq : array shape (4,)
       w, i, j, k of quaternion inverse
    '''
    return qconjugate(q) / qnorm(q)


def qeye(dtype=np.float64):
    ''' Return identity quaternion '''
    return np.array([1.0, 0, 0, 0], dtype=dtype)
