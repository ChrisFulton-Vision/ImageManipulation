"""
threeD_to_threeD.py

Estimate the rigid SE(3) transform between two corresponding 3-D point sets.

The estimated transform maps source-frame points into the target frame:

    p_target ~= R @ p_source + t

where R is constrained to SO(3) and t is a 3-vector. The implementation uses
an SVD/Kabsch/orthogonal-Procrustes solution, which is the closed-form least-
squares optimum for isotropic 3-D point residuals.

This module intentionally contains only the math/data-model layer. CSV I/O,
demos, plotting, animations, and command-line entry points live in:

    support.io.threeD_fileReader

Input point arrays may be either N x 3 or 3 x N. By default, N x 3 is assumed,
except for unambiguous 3 x N arrays where N != 3. Use points_are_columns=True
to force legacy 3 x N interpretation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

import support.mathHelpers.quaternions as q

FloatArray = NDArray[np.float64]

_EPS = 1.0e-12


@dataclass(frozen=True)
class SE3:
    """Rigid transform from ``source_frame`` into ``target_frame``.

    The transform convention is

        p_target = R @ p_source + t

    for column-vector mathematics. For an N x 3 point array, this is evaluated
    efficiently as ``points @ R.T + t``.
    """

    R: FloatArray
    t: FloatArray
    source_frame: str = "source"
    target_frame: str = "target"

    def __post_init__(self) -> None:
        R = np.asarray(self.R, dtype=float).reshape(3, 3)
        t = np.asarray(self.t, dtype=float).reshape(3)
        if not np.all(np.isfinite(R)) or not np.all(np.isfinite(t)):
            raise ValueError("SE3 contains non-finite values.")
        object.__setattr__(self, "R", R)
        object.__setattr__(self, "t", t)

    @property
    def matrix(self) -> FloatArray:
        """Return the 4 x 4 homogeneous matrix representation."""

        T = np.eye(4, dtype=float)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T

    @property
    def q_sxyz(self):
        """Return the scalar-first project quaternion corresponding to ``R``."""

        return q.mat2quat(self.R)

    def inverse(self) -> "SE3":
        """Return the inverse rigid transform."""

        R_inv = self.R.T
        t_inv = -R_inv @ self.t
        return SE3(
            R=R_inv,
            t=t_inv,
            source_frame=self.target_frame,
            target_frame=self.source_frame,
        )

    def transform_points(
        self,
        points: ArrayLike,
        *,
        points_are_columns: Optional[bool] = None,
    ) -> FloatArray:
        """Apply the transform to points and return an N x 3 array."""

        pts = as_points3(points, points_are_columns=points_are_columns)
        return pts @ self.R.T + self.t

    def as_project_quaternion(self):  # pragma: no cover - depends on local project package
        """Return the project Quaternion object when support.mathHelpers is available."""

        return q.mat2quat(self.R)

    def as_project_SE3(self):  # pragma: no cover - depends on local project package
        """Return the project SE3 object when the project Quaternion class supports it."""

        return self.as_project_quaternion().to_SE3_given_position(self.t)


@dataclass(frozen=True)
class AlignmentDiagnostics:
    """Fit quality information for a 3-D point alignment."""

    residuals: FloatArray
    residual_norms: FloatArray
    rmse: float
    weighted_rmse: float
    max_error: float
    singular_values: FloatArray
    source_rank: int
    target_rank: int


class ThreeDToThreeD:
    """Estimate an SE(3) transform from corresponding 3-D points.

    Parameters
    ----------
    source_points:
        Points expressed in the source frame. These are transformed.
    target_points:
        Corresponding points expressed in the target frame.
    weights:
        Optional nonnegative per-point weights. If omitted, all points are
        weighted equally.
    points_are_columns:
        ``True`` for legacy 3 x N point matrices, ``False`` for N x 3 point
        matrices, or ``None`` for automatic detection.
    source_frame, target_frame:
        Names stored in the resulting :class:`SE3` object.
    """

    def __init__(
        self,
        source_points: ArrayLike,
        target_points: ArrayLike,
        *,
        weights: Optional[ArrayLike] = None,
        points_are_columns: Optional[bool] = None,
        source_frame: str = "source",
        target_frame: str = "target",
    ) -> None:
        self.source_points = as_points3(source_points, points_are_columns=points_are_columns)
        self.target_points = as_points3(target_points, points_are_columns=points_are_columns)
        self.weights = validate_weights(weights, self.source_points.shape[0])
        self.num_points = self.source_points.shape[0]

        self.transform, self.diagnostics = fit_se3_kabsch(
            self.source_points,
            self.target_points,
            weights=self.weights,
            source_frame=source_frame,
            target_frame=target_frame,
        )

        # Compatibility aliases for older scripts.
        self.points1 = self.source_points
        self.points2 = self.target_points
        self.R = self.transform.R
        self.t = self.transform.t
        self.q_sxyz = self.transform.q_sxyz
        self.q = self.q_sxyz
        self.ans_rot = self.R
        self.ans_trans = self.t.reshape(3, 1)
        self.residual = self.diagnostics.residual_norms

    def transform_points(self, points: ArrayLike, *, points_are_columns: Optional[bool] = None) -> FloatArray:
        """Apply the estimated source-to-target transform to points."""

        return self.transform.transform_points(points, points_are_columns=points_are_columns)

    def create_y(
        self,
        transform: Optional[SE3] = None,
        *,
        flatten: bool = True,
    ) -> FloatArray:
        """Return target minus transformed-source residuals."""

        if transform is None:
            transform = self.transform
        residuals = self.target_points - transform.transform_points(self.source_points)
        return residuals.reshape(-1) if flatten else residuals

    def to_SE3(self) -> SE3:
        """Return the estimated SE3 object."""

        return self.transform

    def to_project_SE3(self):  # pragma: no cover - depends on local project package
        """Return the project-native SE3 object when project helpers are available."""

        return self.transform.as_project_SE3()


# Backward-compatible class name used by the older file.
class ThreeD_to_ThreeD(ThreeDToThreeD):
    pass


def as_points3(points: ArrayLike, *, points_are_columns: Optional[bool] = None) -> FloatArray:
    """Return points as an N x 3 floating-point array.

    Legacy alignment scripts often used 3 x N arrays. Newer numerical code is
    generally easier to read as N x 3. This helper accepts both.
    """

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        raise ValueError(f"Expected a 2-D point array, got shape {pts.shape}.")

    if points_are_columns is True:
        if pts.shape[0] != 3:
            raise ValueError(f"Expected a 3 x N array, got shape {pts.shape}.")
        pts = pts.T
    elif points_are_columns is False:
        if pts.shape[1] != 3:
            raise ValueError(f"Expected an N x 3 array, got shape {pts.shape}.")
    else:
        if pts.shape[1] == 3:
            pass
        elif pts.shape[0] == 3:
            pts = pts.T
        else:
            raise ValueError(f"Expected N x 3 or 3 x N points, got shape {pts.shape}.")

    pts = np.ascontiguousarray(pts, dtype=float)
    if pts.shape[0] < 3:
        raise ValueError("At least three corresponding points are required.")
    if not np.all(np.isfinite(pts)):
        raise ValueError("Point array contains NaN or infinite values.")
    return pts


def validate_weights(weights: Optional[ArrayLike], num_points: int) -> FloatArray:
    """Return normalized positive weights of length ``num_points``."""

    if weights is None:
        return np.full(num_points, 1.0 / num_points, dtype=float)

    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.size != num_points:
        raise ValueError(f"Expected {num_points} weights, got {w.size}.")
    if not np.all(np.isfinite(w)):
        raise ValueError("Weights contain NaN or infinite values.")
    if np.any(w < 0.0):
        raise ValueError("Weights must be nonnegative.")
    total = float(np.sum(w))
    if total <= _EPS:
        raise ValueError("At least one weight must be positive.")
    return w / total


def fit_se3_kabsch(
    source_points: ArrayLike,
    target_points: ArrayLike,
    *,
    weights: Optional[ArrayLike] = None,
    points_are_columns: Optional[bool] = None,
    source_frame: str = "source",
    target_frame: str = "target",
) -> tuple[SE3, AlignmentDiagnostics]:
    """Fit the weighted least-squares rigid transform from source to target."""

    source = as_points3(source_points, points_are_columns=points_are_columns)
    target = as_points3(target_points, points_are_columns=points_are_columns)
    if source.shape != target.shape:
        raise ValueError(f"Point arrays must have the same shape, got {source.shape} and {target.shape}.")

    w = validate_weights(weights, source.shape[0])

    source_centroid = np.sum(source * w[:, None], axis=0)
    target_centroid = np.sum(target * w[:, None], axis=0)
    source_centered = source - source_centroid
    target_centered = target - target_centroid

    source_rank = centered_rank(source_centered, w)
    target_rank = centered_rank(target_centered, w)
    if min(source_rank, target_rank) < 2:
        raise ValueError(
            "Degenerate point geometry: at least three non-collinear corresponding "
            "points are required to determine a unique 3-D rigid rotation."
        )

    # Cross-covariance for column-vector convention p_target ~= R @ p_source + t.
    H = source_centered.T @ (target_centered * w[:, None])
    U, singular_values, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T
    # Reflection correction: coordinate-frame alignment must live in SO(3), not O(3).
    if np.linalg.det(R) < 0.0:
        D = np.eye(3)
        D[-1, -1] = -1.0
        R = Vt.T @ D @ U.T

    # Numerical cleanup: project back onto SO(3) after determinant correction.
    R = project_to_so3(R)
    t = target_centroid - R @ source_centroid

    transform = SE3(R=R, t=t, source_frame=source_frame, target_frame=target_frame)
    residuals = target - transform.transform_points(source)
    residual_norms = np.linalg.norm(residuals, axis=1)
    rmse = float(np.sqrt(np.mean(np.sum(residuals * residuals, axis=1))))
    weighted_rmse = float(np.sqrt(np.sum(w * np.sum(residuals * residuals, axis=1))))
    max_error = float(np.max(residual_norms))

    diagnostics = AlignmentDiagnostics(
        residuals=residuals,
        residual_norms=residual_norms,
        rmse=rmse,
        weighted_rmse=weighted_rmse,
        max_error=max_error,
        singular_values=singular_values,
        source_rank=source_rank,
        target_rank=target_rank,
    )
    return transform, diagnostics


def centered_rank(centered_points: FloatArray, weights: FloatArray) -> int:
    """Return the numerical rank of the weighted centered point cloud."""

    weighted = centered_points * np.sqrt(weights[:, None])
    singular_values = np.linalg.svd(weighted, compute_uv=False)
    tol = max(centered_points.shape) * np.finfo(float).eps * max(float(singular_values[0]), 1.0)
    return int(np.sum(singular_values > tol))


def project_to_so3(R: ArrayLike) -> FloatArray:
    """Project a nearly valid rotation matrix onto SO(3)."""

    R = np.asarray(R, dtype=float).reshape(3, 3)
    U, _, Vt = np.linalg.svd(R)
    R_so3 = U @ Vt
    if np.linalg.det(R_so3) < 0.0:
        U[:, -1] *= -1.0
        R_so3 = U @ Vt
    return R_so3


def rotation_matrix_to_quat_sxyz(R: ArrayLike):
    """Convert a rotation matrix to a scalar-first project quaternion [s, x, y, z]."""

    R = np.asarray(R, dtype=float).reshape(3, 3)
    return q.mat2quat(R)


def quat_sxyz_to_rotation_matrix(quat: q.Quaternion) -> FloatArray:
    """Convert a scalar-first unit quaternion [s, x, y, z] to a rotation matrix."""

    quat = quat.normalize()
    s = quat.s
    x, y, z = quat.vec
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - s * z), 2.0 * (x * z + s * y)],
            [2.0 * (x * y + s * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - s * x)],
            [2.0 * (x * z - s * y), 2.0 * (y * z + s * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )
