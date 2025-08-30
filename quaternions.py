'''Functions to operate on, or return, quaternions.

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
'''

import numpy as np
from typing_extensions import Self, Union
import copy
_FLOAT_EPS = np.finfo(np.float64).eps

class Quaternion:
    def __init__(self, s: float = None, vec: np.array = None, quat: np.array = None, makeUnitVec: bool = True) -> None:
        # These two parameters form the definition of the quaternion. self.s is a scalar asscoiated with the
        # real component of the quaternion, while self.vec is the vector, associated with i, j, k / x, y, z components
        self.s = 1.0
        self.vec = np.zeros((3,))

        # Included for redundancy, if a quaternion is passed in, make a copy of its values
        if isinstance(quat, Quaternion):
            self.s = copy.deepcopy(quat.s)
            self.vec = copy.deepcopy(quat.vec)
            # raise ValueError(f'Gave me a quaternion already, with {s = } and {vec = }')
            return

        #
        if s is None and vec is None and quat is not None:
            self.s = quat[0]
            self.vec = quat[1:4].flatten()
            self.checkUnit(makeUnitVec)
            return

        if s is None and vec is None:
            self.s = 1.0
            self.vec = np.zeros((3,))
            self.checkUnit(makeUnitVec)
            return

        if s is None:
            if np.shape(vec) == (3,):
                self.vec = vec
            elif np.shape(vec) == (3, 1) or np.shape(vec) == (1, 3):
                self.vec = vec.flatten()
            else:
                raise ValueError('vec should be a (3,) or (3,1) or (1,3) numpy array')
            self.s = np.sqrt(1.0 - vec.dot(vec))
            self.checkUnit(makeUnitVec)
            return

        if vec is None:
            if len(s) > 1:
                raise ValueError('s should be a single float')
            self.s = s
            self.vec = np.zeros((3,))
        else:
            self.s = s
            if np.shape(vec) == (3,):
                self.vec = vec
            elif np.shape(vec) == (3, 1) or np.shape(vec) == (1, 3):
                self.vec = vec.flatten()
            else:
                raise ValueError('vec should be a (3,) or (3,1) or (1,3) numpy array')
        self.checkUnit(makeUnitVec)

    def checkUnit(self,makeUnitVec):
        if makeUnitVec:
            norm = self.norm
            if np.abs(norm) > 0.000001:
                self.s /= norm
                self.vec /= norm

    def __str__(self) -> str:
        if self.s < 0:
            strg = f'[{self.s:.10f}, <'
        else:
            strg = f'[ {np.abs(self.s):.10f}, <'
        if self.vec[0] < 0:
            strg += f'{self.vec[0]:.10f}, '
        else:
            strg += f' {np.abs(self.vec[0]):.10f}, '
        if self.vec[1] < 0:
            strg += f'{self.vec[1]:.10f}, '
        else:
            strg += f' {np.abs(self.vec[1]):.10f}, '
        if self.vec[2] < 0:
            strg += f'{self.vec[2]:.10f}>]'
        else:
            strg += f' {np.abs(self.vec[2]):.10f}>]'
        return strg

    def __format__(self, format_spec) -> str:
        if format_spec == 'ijk':
            return f'{self.s:.3f} + {self.vec[0]:.3f}i + {self.vec[1]:.3f}j + {self.vec[2]:.3f}k'
        else:
            return self.__str__()

    def __xor__(self, scalar):
        print(scalar, self)
        return self.power(scalar)

    # def __format__(self, format_spec):
    #     if format_spec.startswith("."):
    #         precision = int(format_spec[1:])
    #         np.set_printoptions(precision=precision)
    #     return self.__str__

    def __truediv__(self, divisor: float) -> Self:
        if isinstance(divisor, float):
            return Quaternion(quat=self.ndarray / divisor, makeUnitVec=False)

    def __sub__(self, subtractor: Union[np.ndarray, Self]) -> Self:
        if isinstance(subtractor, np.ndarray) and subtractor.shape == (4,):
            return Quaternion(quat=self.ndarray - subtractor, makeUnitVec=False)

        elif isinstance(subtractor, Quaternion):
            return Quaternion(quat=self.ndarray - subtractor.ndarray, makeUnitVec=False)

    def __add__(self, other: Self) -> Self:
        return Quaternion(quat=np.array([self.s + other.s,
                                         self.vec[0] + other.vec[0],
                                         self.vec[1] + other.vec[1],
                                         self.vec[2] + other.vec[2]]),makeUnitVec=False)

    def __eq__(self, other):
        if not isinstance(other, Quaternion):
            raise TypeError(f'Comparing two unlike objects. Self (Quaternion) and {type(other)}')

        if np.abs(self.s - other.s) < _FLOAT_EPS and np.linalg.norm(self.vec - other.vec) < _FLOAT_EPS:
            return True

        # A negative quaternion is equivalent to it's positive: -q = q
        if np.abs(self.s + other.s) < _FLOAT_EPS and np.linalg.norm(self.vec + other.vec) < _FLOAT_EPS:
            return True

        return False


    def __rmul__(self, other):
        if isinstance(other, float):
            return self * other
    def __mul__(self, multiplier, severalColumnVecs=False):
        '''
        "*" Operator override:
        If multiplier is a 3x1 np.array, treat it like a quat-vect multiplication
            return 3x1 np.array vector
        If multiplier is a 4x1 np.array, treat it like a quat-quat multiplication,
            return 4x1 np.array
        If multiplier is a 3x3 np.array, treat it like multiple quat-vect multiplication,
            return 3x3 np.array
        If multiplier is another Quaternion object, treat it like quat-quat,
            return 4x1 np.array
        '''
        if isinstance(multiplier, np.ndarray):
            if multiplier.shape == (3,):
                return self.qv_mult(multiplier)
            if multiplier.shape == (4,):
                return self.qn_mult(multiplier)
            if multiplier.shape == (3, 3):
                return self.qM_mult(multiplier)
            if multiplier.shape[0] == 3:
                return self.qVECS_mult(multiplier)
            else:
                raise ValueError(f'Bad multiplier, unknown object: {multiplier}')
        elif isinstance(multiplier, Quaternion):
            return Quaternion(quat=self.qq_mult(multiplier), makeUnitVec=False)
        elif isinstance(multiplier, float):
            return Quaternion(s=multiplier * self.s, vec=multiplier * self.vec, makeUnitVec=False)
        else:
            raise ValueError(f'Bad multiplier, unknown object: {multiplier}')

    def specializedQuatDiff(self, quat):
        return self.T * quat

    def normalize(self):
        norm = self.norm
        self.s /= norm
        self.vec /= norm
        return self

    def qVECS_mult(self, vecs):
        sol = np.zeros(vecs.shape)
        for idx, vec in enumerate(vecs.T):
            sol[:, idx] = self.qv_mult(vec)
        return sol

    def vect_deriv(self, vect, isQuatConjugated):
        deriv = np.zeros((3, 4))

        # d_q0
        deriv[:, 0] = 2 * self.s * vect + 2 * np.cross(self.vec, vect)

        # d_qx
        deriv[:, 1] = np.array([2 * (self.vec[0] * vect[0] + np.dot(self.vec, vect)),
                                2 * self.vec[1] * vect[0],
                                2 * self.vec[2] * vect[0]]) + \
                      -2 * self.vec[0] * vect + \
                      2 * self.s * np.array([0.0, -vect[2], vect[1]])

        # d_qy
        deriv[:, 2] = np.array([2 * self.vec[0] * vect[1],
                                2 * (self.vec[1] * vect[1] + np.dot(self.vec, vect)),
                                2 * self.vec[2] * vect[1]]) + \
                      -2 * self.vec[1] * vect + \
                      2 * self.s * np.array([vect[2], 0.0, -vect[0]])
        # d_qz
        deriv[:, 3] = np.array([2 * self.vec[0] * vect[2],
                                2 * self.vec[1] * vect[2],
                                2 * (self.vec[2] * vect[2] + np.dot(self.vec, vect))]) + \
                      -2 * self.vec[2] * vect + \
                      2 * self.s * np.array([-vect[1], vect[0], 0.0])
        if isQuatConjugated:
            deriv[:, 1] *= -1.0
            deriv[:, 2] *= -1.0
            deriv[:, 3] *= -1.0

        return deriv

    def to_dcm(self):
        return quat2mat(self.ndarray)

    def qq_mult(self, multQuat):
        return qmult(self.ndarray, multQuat.ndarray)

    def qn_mult(self, multNdarray):
        return qmult(self.ndarray, multNdarray)

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

    @property
    def T(self):
        return Quaternion(self.s, -self.vec, makeUnitVec=False)

    @property
    def inv(self):
        return Quaternion(self.s, -self.vec, makeUnitVec=False)/self.mag ** 2

    @property
    def ndarray(self):
        return np.append(self.s, self.vec)

    @property
    def conj(self):
        return Quaternion(np.append(self.s, -self.vec))

    @property
    def norm(self):
        return np.sqrt(self.s ** 2.0 + self.vec.dot(self.vec))

    @property
    def mag(self):
        return np.sqrt(self.s ** 2 + self.vec.dot(self.vec))

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
        return np.arctan2(2 * (self.s * self.x + self.y * self.z), 1 - 2 * (self.x * self.x + self.y * self.y))

    @property
    def rollD(self):
        return np.rad2deg(self.rollR)

    @property
    def pitchR(self):
        return np.arcsin(2 * (self.s * self.y - self.z * self.x))

    @property
    def pitchD(self):
        return np.rad2deg(self.pitchR)

    @property
    def yawR(self):
        return np.arctan2(2 * (self.s * self.z + self.x * self.y), 1 - 2 * (self.y * self.y + self.z * self.z))

    @property
    def yawD(self):
        return np.rad2deg(self.yawR)

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
        return np.rad2deg(self.eulerR(order))

    def angle_betweenR(self, otherQuat):
        cosVal = (self.s * otherQuat.s + self.vec.dot(otherQuat.vec)) / (self.norm * otherQuat.norm)
        if 1.0 < cosVal < 1.00001:
            return 0.0
        else:
            acosVal = 2.0 * np.arccos(cosVal)
            if acosVal > np.pi:
                return 2.0 * np.pi - acosVal
            return acosVal

    def angle_betweenD(self, otherQuat):
        return 180.0 / np.pi * self.angle_betweenR(otherQuat)

    @property
    def exp(self):
        if np.linalg.norm(self.vec) > 0.00000001:
            return np.exp(self.s) * Quaternion(s=np.cos(np.linalg.norm(self.vec)), vec=self.vec/np.linalg.norm(self.vec)*np.sin(np.linalg.norm(self.vec)))
        return Quaternion(s=1.0, vec=np.zeros((3,)))

    @property
    def ln(self):
        if np.linalg.norm(self.vec) < 0.000001:
            return Quaternion(s=0.0, vec= np.zeros((3,)))
        return Quaternion(s=np.log(self.norm), vec=self.vec/np.linalg.norm(self.vec)*np.acos(self.s/self.norm))

    def power(self, power:float):
        if not isinstance(power, float):
            power = float(power)
        return (power * self.ln).exp.normalize()


    @staticmethod
    def from_rodrigues(rod_vec: np.array):
        norm = np.linalg.norm(rod_vec)
        cosRod = np.cos(norm / 2.0)
        sinRod = np.sin(norm / 2.0)
        return Quaternion(s=cosRod, vec=np.array([rod_vec[0]/norm * sinRod,
                                                            rod_vec[1]/norm * sinRod,
                                                            rod_vec[2]/norm * sinRod]))

    def slerp(self, q2:Self, t) -> Self:
        return self * (self.inv * q2).power(t)

class DualQuat:
    def __init__(self, q_real : Quaternion = Quaternion(), q_dual : Quaternion = Quaternion(quat=np.array([0.0,0.0,0.0,0.0])),
                 r : Quaternion = None, t : Quaternion = None, t_vec : np.array = None) -> None:

        if r is not None:
            if not 1-10*_FLOAT_EPS < r.mag < 1+10*_FLOAT_EPS:
                raise TypeError(f'Input r should be a unit quaternion (r.mag == 1.0):  r.mag={r.mag}')

            if t is not None:
                if t.s < -_FLOAT_EPS or t.s > _FLOAT_EPS:
                    raise TypeError(f'Input t should be a pure quaternion (t.s == 0):  t.s={t.s}')

                self.q_real = r
                self.q_dual = 0.5 * t * r
                return

            if t_vec is not None:
                t = Quaternion(s=0, vec=t_vec, makeUnitVec=False)
                self.q_real = r
                self.q_dual = 0.5 * t * r
                return

            raise TypeError(f'Rotation Quaterion input, {r}, but neither t nor t_vec were supplied.')

        if not isinstance(q_real,Quaternion) or not isinstance(q_dual, Quaternion):
            raise TypeError(f'Bad input types. Required Quaternions: q_real: {type(q_real)}, q_dual: {type(q_dual)}')

        self.q_real = q_real
        self.q_dual = q_dual

    def __str__(self) -> str:
        return f'{self.q_real.__str__()} + {self.q_dual.__str__()}\u03B5'

    def __mul__(self, other) -> Self:
        going_out = copy.deepcopy(self)
        otherCopy = copy.deepcopy(other)

        if isinstance(otherCopy, float) or isinstance(otherCopy, int):
            going_out.q_real *= otherCopy
            going_out.q_dual *= otherCopy
            return going_out
        if isinstance(otherCopy, DualQuat):
            going_out.q_dual = going_out.q_real * otherCopy.q_dual + going_out.q_dual * otherCopy.q_real
            going_out.q_real *= otherCopy.q_real
            return going_out
        if isinstance(otherCopy, np.ndarray) and otherCopy.shape == (3,):
            multiplier = Quaternion(s=0.0, vec=otherCopy)
            going_out.q_dual = going_out.q_real * multiplier + going_out.q_dual * Quaternion()
            return going_out
        raise ValueError('Object Type Unknown: {type(other)} is not a valid object.')

    def __rmul__(self, other) -> Self:
        if isinstance(other, DualQuat):
            raise TypeError("It shouldn't be able to get here, but it would multiple wrong now.")
        return copy.deepcopy(self) * other

    def __sub__(self, other : Self) -> Self:
        going_out = copy.deepcopy(self)
        if isinstance(other, DualQuat):
            # going_out.q_real += other.q_real
            going_out.q_dual -= other.q_dual
        return going_out

    def __add__(self, other : Self) -> Self:
        going_out = copy.deepcopy(self)
        if isinstance(other, DualQuat):
            # going_out.q_real += other.q_real
            going_out.q_dual += other.q_dual
        return going_out

    def __neg__(self):
        return self * -1.0

    def __eq__(self, other):
        if not isinstance(other, DualQuat):
            raise TypeError(f'Comparing two unlike objects. Self (DualQuat) and {type(other)}')

        if self.q_real == other.q_real and self.q_dual == other.q_dual:
            return True

        return False

    def translateInFrame(self, vec):
        new_t_vec = self.t_vec + vec
        return DualQuat(r=self.r, t_vec=new_t_vec)

    @property
    def T(self):
        '''
        >>> test_Q = DualQuat(q_real=q.Quaternion(s=1.0,vec=np.array([0.0,0.0,0.0])),
        ... q_dual=q.Quaternion(s=0.0,vec=np.array([-1.0,1.0,1.0]), makeUnitVec=False))
        >>> sol_Q =  DualQuat(q_real=q.Quaternion(s=1.0,vec=np.array([0.0,0.0,0.0])),
        ... q_dual=q.Quaternion(s=0.0,vec=np.array([1.0,-1.0,-1.0]), makeUnitVec=False))
        >>> assert test_Q == sol_Q
        '''
        return DualQuat(q_real=copy.deepcopy(self.q_real).T, q_dual=copy.deepcopy(self.q_dual).T)

    @property
    def mag(self):
        return self.T * self

    @property
    def norm(self):
        return self.T * self
    @property
    def inv(self):
        if self.q_real.norm > _FLOAT_EPS:
            return DualQuat(q_real = copy.deepcopy(self.q_real).inv, q_dual= copy.deepcopy(self.q_real).inv * -0.5 * copy.deepcopy(self.q_dual) * copy.deepcopy(self.q_real).inv)
        raise ValueError('No Inverse, Magnitude too close to 0')

    @property
    def r(self):
        return self.q_real

    @property
    def t(self):
        return 2.0 * self.q_dual * self.q_real.T

    @property
    def t_vec(self):
        return (2.0 * self.q_dual * self.q_real.T).ndarray[1:4]

    @property
    def isUnit(self):
        return (1.0 - 10.0*_FLOAT_EPS < self.q_real.mag < 1.0 + 10.0*_FLOAT_EPS and
                -_FLOAT_EPS < self.q_real.ndarray.dot(self.q_dual.ndarray) < _FLOAT_EPS)

pure_qs = Quaternion(s=1.0, vec=np.zeros((3,)))
pure_qx = Quaternion(s=0.0, vec=np.array([1.0,0.0,0.0]))
pure_qy = Quaternion(s=0.0, vec=np.array([0.0,1.0,0.0]))
pure_qz = Quaternion(s=0.0, vec=np.array([0.0,0.0,1.0]))

def interpolate(q1:Quaternion, q2:Quaternion, t:float):
    interp : Quaternion = q1 * (q1.inv * q2).power(t)
    return interp

def randomQuat(unit=True):
    return Quaternion(quat=np.random.rand(4,), makeUnitVec=unit)

def left_quat_productDeriv(quatL, quatR, isTargetConjugated):
    deriv = np.zeros((4,4))
    deriv[:, 0] = (pure_qs * quatR).ndarray
    deriv[:, 1] = (pure_qx * quatR).ndarray
    deriv[:, 2] = (pure_qy * quatR).ndarray
    deriv[:, 3] = (pure_qz * quatR).ndarray
    if isTargetConjugated:
        deriv[:, 1:] = -deriv[:,1:]
    return deriv

def right_quat_productDeriv(quatL, quatR, isTargetConjugated):
    deriv = np.zeros((4,4))
    deriv[:, 0] = (quatL * pure_qs).ndarray
    deriv[:, 1] = (quatL * pure_qx).ndarray
    deriv[:, 2] = (quatL * pure_qy).ndarray
    deriv[:, 3] = (quatL * pure_qz).ndarray
    if isTargetConjugated:
        deriv[:, 1:] = -deriv[:,1:]
    return deriv

def tri_quat_productDeriv(quat1, quat2, quat3, idx, isTargetConjugated):
    '''
    >>> for i in range(1000):
    ...     q0 = randomQuat()
    ...     q1 = randomQuat()
    ...     q2 = randomQuat()
    ...     delt = 0.00000001
    ...     mult = q0 * q1 * q2
    ...     q1.s += delt
    ...     sMult = q0 * q1 * q2
    ...     q1.s -= delt
    ...
    ...     q1.vec[0] += delt
    ...     xMult = q0 * q1 * q2
    ...     q1.vec[0] -= delt
    ...
    ...     q1.vec[1] += delt
    ...     yMult = q0 * q1 * q2
    ...     q1.vec[1] -= delt
    ...
    ...     q1.vec[2] += delt
    ...     zMult = q0 * q1 * q2
    ...     q1.vec[2] -= delt
    ...
    ...     numDeriv = np.zeros((4,4))
    ...
    ...     numDeriv[:,0] = ((sMult - mult)/delt).ndarray
    ...     numDeriv[:,1] = ((xMult - mult)/delt).ndarray
    ...     numDeriv[:,2] = ((yMult - mult)/delt).ndarray
    ...     numDeriv[:,3] = ((zMult - mult)/delt).ndarray
    ...
    ...     np.testing.assert_allclose(tri_quat_productDeriv(q0, q1, q2, 'm'), numDeriv, rtol=1e-05, atol=1e-05)

    :param quat1: q0 in quaternion multiplication q0 * q1 * q2
    :param quat2: q1 in quaternion multiplication q0 * q1 * q2
    :param quat3: q2 in quaternion multiplication q0 * q1 * q2
    :return: Partial derivative of multiplication with respect to idx, 'l' for left, 'm' for middle', 'r' for right
    '''
    deriv = np.zeros((4,4))
    if idx == 'l':
        deriv[:, 0] = (pure_qs * quat2 * quat3).ndarray
        deriv[:, 1] = (pure_qx * quat2 * quat3).ndarray
        deriv[:, 2] = (pure_qy * quat2 * quat3).ndarray
        deriv[:, 3] = (pure_qz * quat2 * quat3).ndarray
        if isTargetConjugated:
            deriv[:, 1:] = -deriv[:,1:]
        return deriv

    if idx == 'm':
        deriv[:, 0] = (quat1 * pure_qs * quat3).ndarray
        deriv[:, 1] = (quat1 * pure_qx * quat3).ndarray
        deriv[:, 2] = (quat1 * pure_qy * quat3).ndarray
        deriv[:, 3] = (quat1 * pure_qz * quat3).ndarray
        if isTargetConjugated:
            deriv[:, 1:] = -deriv[:,1:]
        return deriv

    if idx == 'r':
        deriv[:, 0] = (quat1 * quat2 * pure_qs).ndarray
        deriv[:, 1] = (quat1 * quat2 * pure_qx).ndarray
        deriv[:, 2] = (quat1 * quat2 * pure_qy).ndarray
        deriv[:, 3] = (quat1 * quat2 * pure_qz).ndarray
        if isTargetConjugated:
            deriv[:, 1:] = -deriv[:,1:]
        return deriv

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

    w = np.sqrt(1.0-(x*x+y*y+z*z))

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
        w = np.sqrt(w2)
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
    >>> np.allclose(q, [1, 0, 0, 0])
    True
    >>> q = mat2quat(np.diag([1, -1, -1]))
    >>> np.allclose(q, [0, 1, 0, 0]) # 180 degree rotn around axis 0
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
    # Select largest eigenvector, reorder to w,x,y,z quaternion
    q = vecs[:, np.argmax(vals)]
    # Prefer quaternion with positive w
    # (q * -1 corresponds to same rotation as q)
    if q[0] < 0:
        q *= -1
    return Quaternion(s=q[0], vec=q[1:], makeUnitVec=True)


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
    return np.sqrt(q.dot(q))


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

