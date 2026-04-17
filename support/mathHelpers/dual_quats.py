"""Dual quaternion helpers for rigid transforms.

The class stores a dual quaternion as ``q_real + epsilon * q_dual`` where
``q_real`` is the rotation quaternion and ``q_dual`` encodes translation.
"""

from __future__ import annotations

import copy

import numpy as np
from typing_extensions import Self

import quaternions as q

_FLOAT_EPS = np.finfo(np.float64).eps
_UNIT_QUAT_TOL = 10.0 * _FLOAT_EPS


def _zero_quaternion() -> q.Quaternion:
    return q.Quaternion(quat=np.zeros(4, dtype=float), makeUnitQuat=False)


class DualQuat:
    """Minimal dual quaternion representation built on the local Quaternion class."""

    def __init__(
        self,
        q_real: q.Quaternion | None = None,
        q_dual: q.Quaternion | None = None,
        r: q.Quaternion | None = None,
        t: q.Quaternion | None = None,
        t_vec: np.ndarray | None = None,
    ) -> None:
        """Construct from explicit dual-quaternion parts or from pose terms.

        Examples
        --------
        >>> import numpy as np
        >>> pose = DualQuat(r=q.Quaternion(), t_vec=np.array([1.0, 2.0, 3.0]))
        >>> np.allclose(pose.t_vec, np.array([1.0, 2.0, 3.0]))
        True
        >>> pure_t = q.Quaternion(s=0.0, vec=np.array([1.0, 2.0, 3.0]), makeUnitQuat=False)
        >>> via_t = DualQuat(r=q.Quaternion(), t=pure_t)
        >>> np.allclose(via_t.t_vec, np.array([1.0, 2.0, 3.0]))
        True
        >>> DualQuat(q_real="bad")
        Traceback (most recent call last):
        ...
        TypeError: Bad input types. Required Quaternions: q_real=<class 'str'>, q_dual=<class 'quaternions.Quaternion'>
        >>> DualQuat(r=q.Quaternion(), t=q.Quaternion())
        Traceback (most recent call last):
        ...
        TypeError: Input t should be a pure quaternion (t.s == 0): t.s=1.0
        """
        if r is not None:
            self.q_real, self.q_dual = self._from_pose(r=r, t=t, t_vec=t_vec)
            return

        q_real = q.Quaternion() if q_real is None else q_real
        q_dual = _zero_quaternion() if q_dual is None else q_dual

        if not isinstance(q_real, q.Quaternion) or not isinstance(q_dual, q.Quaternion):
            raise TypeError(
                f"Bad input types. Required Quaternions: q_real={type(q_real)}, q_dual={type(q_dual)}"
            )

        self.q_real = q_real
        self.q_dual = q_dual

    @staticmethod
    def _from_pose(
        r: q.Quaternion,
        t: q.Quaternion | None = None,
        t_vec: np.ndarray | None = None,
    ) -> tuple[q.Quaternion, q.Quaternion]:
        if not isinstance(r, q.Quaternion):
            raise TypeError(f"Input r should be a Quaternion: {type(r)}")

        if not 1.0 - _UNIT_QUAT_TOL < r.mag < 1.0 + _UNIT_QUAT_TOL:
            raise TypeError(f"Input r should be a unit quaternion (r.mag == 1.0): r.mag={r.mag}")

        if t is not None and t_vec is not None:
            raise TypeError("Supply either t or t_vec, not both.")

        if t_vec is not None:
            t = q.Quaternion(s=0.0, vec=np.asarray(t_vec, dtype=float), makeUnitQuat=False)

        if t is None:
            raise TypeError(f"Rotation quaternion input, {r}, but neither t nor t_vec were supplied.")

        if not isinstance(t, q.Quaternion):
            raise TypeError(f"Input t should be a Quaternion: {type(t)}")

        if not -_FLOAT_EPS < t.s < _FLOAT_EPS:
            raise TypeError(f"Input t should be a pure quaternion (t.s == 0): t.s={t.s}")

        return r, 0.5 * t * r

    def __str__(self) -> str:
        return f"{self.q_real} + {self.q_dual}ε"

    def __repr__(self) -> str:
        return f"DualQuat(q_real={self.q_real}, q_dual={self.q_dual})"

    def __mul__(self, other) -> Self:
        """Multiply by a scalar, another dual quaternion, or a pure 3-vector.

        Examples
        --------
        >>> import numpy as np
        >>> dq = DualQuat(r=q.Quaternion(), t_vec=np.array([3.0, 4.0, 5.0]))
        >>> scaled = 2.0 * dq
        >>> np.allclose(scaled.q_dual.ndarray, 2.0 * dq.q_dual.ndarray)
        True
        >>> point = dq * np.array([1.0, 0.0, 0.0])
        >>> point.r == dq.r
        True
        >>> expected = dq.q_real * q.Quaternion(s=0.0, vec=np.array([1.0, 0.0, 0.0]), makeUnitQuat=False) + dq.q_dual
        >>> np.allclose(point.q_dual.ndarray, expected.ndarray)
        True
        >>> composed = DualQuat(r=q.Quaternion(), t_vec=np.array([1.0, 0.0, 0.0])) * DualQuat(r=q.Quaternion(), t_vec=np.array([0.0, 2.0, 0.0]))
        >>> np.allclose(composed.t_vec, np.array([1.0, 2.0, 0.0]))
        True
        """
        if isinstance(other, (float, int)):
            return DualQuat(q_real=self.q_real * other, q_dual=self.q_dual * other)

        if isinstance(other, DualQuat):
            return DualQuat(
                q_real=self.q_real * other.q_real,
                q_dual=self.q_real * other.q_dual + self.q_dual * other.q_real,
            )

        if isinstance(other, np.ndarray) and other.shape == (3,):
            multiplier = q.Quaternion(s=0.0, vec=np.asarray(other, dtype=float), makeUnitQuat=False)
            return DualQuat(
                q_real=copy.deepcopy(self.q_real),
                q_dual=self.q_real * multiplier + self.q_dual * q.Quaternion(),
            )

        raise TypeError(f"Object type unknown: {type(other)} is not a valid multiplier.")

    def __rmul__(self, other) -> Self:
        if isinstance(other, DualQuat):
            raise TypeError("Dual quaternion right-multiplication should resolve through __mul__.")
        return self * other

    def __sub__(self, other: Self) -> Self:
        """Subtract component-wise.

        Examples
        --------
        >>> import numpy as np
        >>> a = DualQuat(r=q.Quaternion(), t_vec=np.array([3.0, 0.0, 0.0]))
        >>> b = DualQuat(r=q.Quaternion(), t_vec=np.array([1.0, 0.0, 0.0]))
        >>> diff = a - b
        >>> np.allclose(diff.q_dual.ndarray, a.q_dual.ndarray - b.q_dual.ndarray)
        True
        """
        if not isinstance(other, DualQuat):
            return NotImplemented
        return DualQuat(q_real=self.q_real - other.q_real, q_dual=self.q_dual - other.q_dual)

    def __add__(self, other: Self) -> Self:
        """Add component-wise.

        Examples
        --------
        >>> import numpy as np
        >>> a = DualQuat(r=q.Quaternion(), t_vec=np.array([3.0, 4.0, 5.0]))
        >>> b = DualQuat(r=q.Quaternion(), t_vec=np.array([1.0, 0.0, 0.0]))
        >>> summed = a + b
        >>> np.allclose(summed.q_dual.ndarray, a.q_dual.ndarray + b.q_dual.ndarray)
        True
        """
        if not isinstance(other, DualQuat):
            return NotImplemented
        return DualQuat(q_real=self.q_real + other.q_real, q_dual=self.q_dual + other.q_dual)

    def __neg__(self) -> Self:
        """Negate both components.

        Examples
        --------
        >>> import numpy as np
        >>> dq = DualQuat(r=q.Quaternion(), t_vec=np.array([3.0, 4.0, 5.0]))
        >>> -dq == (-1.0 * dq)
        True
        """
        return self * -1.0

    def __eq__(self, other) -> bool:
        """Compare both components.

        Examples
        --------
        >>> import numpy as np
        >>> dq = DualQuat(r=q.Quaternion(), t_vec=np.array([1.0, 2.0, 3.0]))
        >>> dq == dq.copy()
        True
        >>> dq == object()
        False
        """
        if not isinstance(other, DualQuat):
            return NotImplemented
        return self.q_real == other.q_real and self.q_dual == other.q_dual

    def translateInFrame(self, vec: np.ndarray) -> Self:
        """Translate by adding a frame-relative offset to the encoded translation.

        Examples
        --------
        >>> import numpy as np
        >>> pose = DualQuat(r=q.Quaternion(), t_vec=np.array([1.0, 2.0, 3.0]))
        >>> moved = pose.translateInFrame(np.array([-1.0, 0.5, 1.0]))
        >>> np.allclose(moved.t_vec, np.array([0.0, 2.5, 4.0]))
        True
        """
        return DualQuat(r=self.r, t_vec=self.t_vec + np.asarray(vec, dtype=float))

    def copy(self) -> Self:
        """Return a deep copy of the dual quaternion.

        Examples
        --------
        >>> import numpy as np
        >>> dq = DualQuat(r=q.Quaternion(), t_vec=np.array([3.0, 4.0, 5.0]))
        >>> copied = dq.copy()
        >>> copied == dq and copied is not dq
        True
        """
        return DualQuat(q_real=self.q_real.copy(), q_dual=self.q_dual.copy())

    @property
    def T(self) -> Self:
        """Quaternion conjugate of both real and dual parts.

        Examples
        --------
        >>> import numpy as np
        >>> dq = DualQuat(r=q.Quaternion(), t_vec=np.array([3.0, 4.0, 5.0]))
        >>> conj = dq.T
        >>> np.allclose(conj.q_real.ndarray, dq.q_real.T.ndarray)
        True
        >>> np.allclose(conj.q_dual.ndarray, dq.q_dual.T.ndarray)
        True
        """
        return DualQuat(q_real=self.q_real.T, q_dual=self.q_dual.T)

    @property
    def mag(self) -> Self:
        return self.T * self

    @property
    def norm(self) -> Self:
        return self.T * self

    @property
    def inv(self) -> Self:
        """Multiplicative inverse.

        Examples
        --------
        >>> import numpy as np
        >>> pose = DualQuat(r=q.randomQuat(), t_vec=np.array([1.0, 2.0, 3.0]))
        >>> identity = pose.inv * pose
        >>> np.allclose(identity.r.ndarray, q.Quaternion().ndarray)
        True
        >>> np.allclose(identity.t_vec, np.zeros(3))
        True
        """
        if self.q_real.norm <= _FLOAT_EPS:
            raise ValueError("No inverse, magnitude too close to 0.")

        q_real_inv = self.q_real.inv
        return DualQuat(
            q_real=q_real_inv,
            q_dual=-1.0 * (q_real_inv * self.q_dual * q_real_inv),
        )

    @property
    def r(self) -> q.Quaternion:
        return self.q_real

    @property
    def t(self) -> q.Quaternion:
        return 2.0 * self.q_dual * self.q_real.T

    @property
    def t_vec(self) -> np.ndarray:
        """Translation vector encoded by the dual quaternion."""
        return self.t.ndarray[1:4]

    @property
    def isUnit(self) -> bool:
        """Return whether the dual quaternion satisfies the unit constraints.

        Examples
        --------
        >>> import numpy as np
        >>> dq = DualQuat(r=q.Quaternion(), t_vec=np.array([3.0, 4.0, 5.0]))
        >>> dq.isUnit
        True
        """
        return bool(
            1.0 - _UNIT_QUAT_TOL < self.q_real.mag < 1.0 + _UNIT_QUAT_TOL
            and -_FLOAT_EPS < self.q_real.ndarray.dot(self.q_dual.ndarray) < _FLOAT_EPS
        )
