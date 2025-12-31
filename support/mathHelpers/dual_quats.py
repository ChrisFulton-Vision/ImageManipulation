import numpy as np
import quaternions as q
from typing_extensions import Self, Union
import copy

_FLOAT_EPS = np.finfo(np.float64).eps

class DualQuat:
    def __init__(self, q_real : q.Quaternion = q.Quaternion(), q_dual : q.Quaternion = q.Quaternion(quat=np.array([0.0,0.0,0.0,0.0]), makeUnitQuat=False),
                 r : q.Quaternion = None, t : q.Quaternion = None, t_vec : np.array = None) -> None:

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
                t = q.Quaternion(s=0, vec=t_vec, makeUnitQuat=False)
                self.q_real = r
                self.q_dual = 0.5 * t * r
                return

            raise TypeError(f'Rotation Quaterion input, {r}, but neither t nor t_vec were supplied.')

        if not isinstance(q_real,q.Quaternion) or not isinstance(q_dual, q.Quaternion):
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
            multiplier = q.Quaternion(s=0.0, vec=otherCopy)
            going_out.q_dual = going_out.q_real * multiplier + going_out.q_dual * q.Quaternion()
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

if __name__ == '__main__':
    test_dualA = DualQuat()
    test_dualB = DualQuat(q_real=q.Quaternion(s=0.5, vec=np.array([0.5, 0.0, 0.0])),
                          q_dual=q.Quaternion(s=0.5, vec=np.array([-0.5, 0.0, 0.0])))
    test_dualC = DualQuat(r=q.Quaternion(s=1.0,vec=np.array([0.0,0.0,0.0])),
                          t_vec=np.array([3.0,3.0,1.0]))

    print('Creation')
    print(f'Dual A: {test_dualA}')
    print(f'Dual B: {test_dualB}')
    print(f'Dual C: {test_dualC}')

    print('\nEncodings')
    print(f'A.r: {test_dualA.r}, A.t: {test_dualA.t}')
    print(f'B.r: {test_dualB.r}, B.t: {test_dualB.t}')
    print(f'C.r: {test_dualC.r}, B.t: {test_dualC.t}')
    print(f'A.tvec: {test_dualA.t_vec}, B.tvec: {test_dualB.t_vec}')

    print('\nCreation from Encoding')
    print(f'A: {DualQuat(r=test_dualA.r, t_vec=test_dualA.t_vec)}')
    print(f'B: {DualQuat(r=test_dualB.r, t_vec=test_dualB.t_vec)}')

    print('\nMag')
    print(f'A.mag:   {test_dualA.mag}')
    print(f'A.norm:  {test_dualA.norm}')
    print(f'B.mag:   {test_dualB.mag}')
    print(f'||A*B||: {(test_dualA * test_dualB).mag}')
    print(f'||C||:   {test_dualC.mag}')

    print('\nIs Unit?')
    print(f'A^-1*A: {(test_dualA.inv * test_dualA).isUnit}')
    print(f'A*A^-1: {(test_dualA * test_dualA.inv).isUnit}')
    print(f'A*A^*:  {(test_dualA * test_dualA.T).isUnit}')
    print(f'B:      {(test_dualB).isUnit}')
    print(f'B*B^-1: {(test_dualB * test_dualB.inv).isUnit}')
    print(f'B*B^*:  {(test_dualB * test_dualB.T).isUnit}')
    print(f'B^-1*B: {(test_dualB.inv * test_dualB).isUnit}')
    print(f'(A*B) * (A*B)^T: {((test_dualA * test_dualB) * (test_dualA * test_dualB).T).isUnit}')

    print('\nAddition')
    print(f'A + B: {test_dualA + test_dualB}')
    print(f'B + C: {test_dualB + test_dualC}')
    print(f'(B + C).r: {(test_dualB + test_dualC).r}')
    print(f'(B + C).t: {(test_dualB + test_dualC).t}')
    print(f'||(B + C)||: {(test_dualB + test_dualC).mag}')
    print(f'||(B + C)||=1: {(test_dualB + test_dualC).isUnit}')

    print('\nMultiplication')
    print(f'A * B:  {test_dualA * test_dualB}')
    print(f'A * -1: {test_dualA * -1.0}')
    print(f'B * -1: {test_dualB * -1.0}')
    print(f'(A * B).t_vec: {(test_dualA * test_dualB).t_vec}')
    print(f'(B * B).t_vec: {(test_dualB * test_dualB).t_vec}')

    print('\nTranspose')
    print(f'A^*:     {test_dualA.T}')
    print(f'B^*:     {test_dualB.T}')
    print(f'A*A^*:   {test_dualA * test_dualA.T}')
    print(f'(A*B)^*: {(test_dualA * test_dualB).T}')
    print(f'C^*:     {test_dualC.T}')

    print('\nInv')
    print(f'A^-1:       {test_dualA.inv}')
    print(f'B^-1:       {test_dualB.inv}')
    print(f'B^-1*B:     {test_dualB.inv * test_dualB}')
    print(f'||B^-1*B||: {(test_dualB.inv * test_dualB).mag}')
    print(f'(A*B)^-1:   {(test_dualA * test_dualB).inv}')
    print(f'C^-1:       {test_dualC.inv}')

    print('\nIdent')
    print(f'A^-1*A: {test_dualA.inv * test_dualA}')
    print(f'A*A^-1: {test_dualA * test_dualA.inv}')
    print(f'B*B^-1: {test_dualB * test_dualB.inv}')
    print(f'B^-1*B: {test_dualB.inv * test_dualB}')
    print(f'A*B * (A*B)^-1: {(test_dualA * test_dualB) * (test_dualA * test_dualB).T}')


    print()
    r = q.Quaternion(quat=np.random.rand(4), makeUnitQuat=True)
    t = q.Quaternion(s=0.0, vec=np.random.rand(3)*20.0-10.0, makeUnitQuat=False)

    print('Random Creation')
    print(f'r: {r}, t: {t}')
    dQ = DualQuat(q_real=r, q_dual= 0.5 * t * r)
    dQ1 = DualQuat(r=r, t=t)
    print(f'dQ:  {dQ}')
    print(f'dQ1: {dQ1}')
    print(f'dQ.T:    {dQ.T}')
    print(f'dQ.inv:  {dQ.inv}')
    print(f'dQ.T * dQ:    {dQ.T * dQ}')
    print(f'dQ.inv * dQ:  {dQ.inv * dQ}')
    print(f'(dQ.inv * dQ).t: {(dQ.inv * dQ).t}')
    print(f'Unit: {dQ * dQ1.T}')
    print(f'Is Unit?: {dQ.isUnit}')
    print(f'r: {dQ.r}')
    print(f't: {dQ1.t}')
    print()

    t_Q_tAnt = q.Quaternion()
    r_Q_t = q.Quaternion(s=.99752, vec=np.array([0.02973,0.01113,-0.0969]))

    t_T_tAnt = np.array([8.40012, -.15817, -.09577])
    r_T_t = np.array([16.59222, -4.20523, 5.03376])
    r_T_a = np.array([-9.27027,  0.45487, 0.67881])

    t_dQ_tAnt = DualQuat(r=t_Q_tAnt, t_vec=t_T_tAnt)
    r_dQ_t = DualQuat(r=r_Q_t, t_vec=r_T_t)
    r_dQ_rAnt = DualQuat(r=q.Quaternion(), t_vec=-r_T_a)

    print((r_dQ_t * r_T_a).t)
    r_dQ_t = r_dQ_t.translateInFrame(-r_T_a)
    print(f'Translated: {r_dQ_t}')

    dQc = r_dQ_t * t_dQ_tAnt
    print(f'r_dQ_t: {r_dQ_t}\nr_dQ_t.r: {r_dQ_t.r}, r_dQ_t.t: {r_dQ_t.t}')
    print(f'r_dQ_t: {r_dQ_t}\nr_dQ_t.r: {r_dQ_t.r}, r_dQ_t.t: {r_dQ_t.t}')
    # print(f'r_dQ_rAnt: {r_dQ_rAnt}\nr_dQ_rAnt.r: {r_dQ_rAnt.r}, r_dQ_rAnt.t: {r_dQ_rAnt.t}')
    print(f'dQc: {dQc}')
    print(f'dQc.r: {dQc.r}\ndQc.t: {dQc.t_vec}')
    print((r_dQ_t * t_dQ_tAnt).t_vec)

    t_Q_tAnt = q.Quaternion()
    r_Q_t = q.Quaternion(s=.99752, vec=np.array([0.02973,0.01113,-0.0969]))

    t_T_tAnt = np.array([8.40012, -.15817, -.09577])
    r_T_t = np.array([16.59222, -4.20523, 5.03376])
    r_T_a = np.array([-9.27027,  0.45487, 0.67881])

    t_dQ_tAnt = DualQuat(r=q.Quaternion(), t_vec=t_T_tAnt)
    print(f'1: {t_dQ_tAnt}')
    print((t_dQ_tAnt).t)
    r_dQ_t = DualQuat(r=r_Q_t, t_vec=r_T_t)
    print((r_dQ_t).t)
    r_dQ_rAnt = DualQuat(r=q.Quaternion(), t_vec=-r_T_a)

    print(f'2: {r_dQ_t * t_dQ_tAnt}')
    print((r_dQ_rAnt * r_dQ_t * t_dQ_tAnt).t)
