"""Single-feature aircraft range/position geometry.

This module is meant to be the single source of truth for the simple
whole-aircraft bounding-box range model used by both live runtime display and
offline/batch data processing.

Coordinate convention
---------------------
The returned position is in the camera coordinate frame implied by the supplied
camera matrix K. For the usual OpenCV camera model, this is:

    x: image right
    y: image down
    z: camera forward

Range model
-----------
For a detected whole-aircraft bounding box with pixel width ``bbox_w_px`` and a
known physical aircraft width ``aircraft_width_m``, the forward range estimate is

    range_m = fx_px * aircraft_width_m / bbox_w_px

The center pixel is then back-projected through K and scaled by ``range_m``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


_FLOAT_EPS = 1.0e-12
_DEFAULT_MIN_BBOX_WIDTH_PX = 1.0e-6


@dataclass(frozen=True, slots=True)
class TargetProfile:
    """Physical target metadata needed by the single-feature range model."""

    name: str
    aircraft_width_m: float


# Keep target widths here, not in runtime or batch-processing code.
CUB_TARGET = TargetProfile(name="Cub", aircraft_width_m=4.07)
OSPREY_TARGET = TargetProfile(name="Osprey", aircraft_width_m=3.52636931926423)

TARGET_PROFILES: Mapping[str, TargetProfile] = {
    "cub": CUB_TARGET,
    "osprey": OSPREY_TARGET,
}

# Common control: change this one line, or pass a target_profile explicitly.
DEFAULT_SINGLE_FEATURE_TARGET = OSPREY_TARGET


@dataclass(frozen=True, slots=True)
class SingleFeatureEstimate:
    """Single-feature range and camera-frame position estimate."""

    range_m: float
    xyz_cam_m: NDArray[np.float64]
    center_px: tuple[float, float]
    bbox_w_px: float
    bbox_h_px: float | None
    aircraft_width_m: float
    target_name: str | None = None

    @property
    def norm_m(self) -> float:
        """Euclidean norm of ``xyz_cam_m`` in meters."""
        return float(np.linalg.norm(self.xyz_cam_m))


def resolve_target_profile(target_profile: str | TargetProfile | None = None) -> TargetProfile:
    """Resolve a target-profile name or object to a :class:`TargetProfile`.

    Parameters
    ----------
    target_profile:
        ``None`` uses :data:`DEFAULT_SINGLE_FEATURE_TARGET`. Strings are matched
        case-insensitively against :data:`TARGET_PROFILES`.
    """
    if target_profile is None:
        return DEFAULT_SINGLE_FEATURE_TARGET

    if isinstance(target_profile, TargetProfile):
        return target_profile

    key = str(target_profile).strip().lower()
    try:
        return TARGET_PROFILES[key]
    except KeyError as exc:
        valid = ", ".join(sorted(TARGET_PROFILES.keys()))
        raise ValueError(f"Unknown target_profile={target_profile!r}. Valid keys: {valid}") from exc


def resolve_aircraft_width_m(
    *,
    aircraft_width_m: float | None = None,
    target_profile: str | TargetProfile | None = None,
) -> tuple[float, str | None]:
    """Return the aircraft width and the profile name used to obtain it.

    ``aircraft_width_m`` is allowed for one-off studies. When omitted, the width
    comes from ``target_profile``. If both are supplied, the explicit width wins,
    but must be positive.
    """
    if aircraft_width_m is not None:
        width = float(aircraft_width_m)
        if not np.isfinite(width) or width <= 0.0:
            raise ValueError(f"aircraft_width_m must be positive and finite, got {aircraft_width_m!r}")
        profile_name = None if target_profile is None else resolve_target_profile(target_profile).name
        return width, profile_name

    profile = resolve_target_profile(target_profile)
    width = float(profile.aircraft_width_m)
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError(f"Target profile {profile.name!r} has invalid width {width!r}")
    return width, profile.name


def camera_matrix_from_calibration(
    calibration: Any,
    *,
    image_size_px: tuple[float, float] | None = None,
    scale_to_image: bool = True,
) -> NDArray[np.float64]:
    """Return a copy of a calibration object's camera matrix.

    This helper intentionally does not mutate the calibration object. If the
    image being processed differs from the calibration image size, set
    ``image_size_px=(width_px, height_px)`` and leave ``scale_to_image=True``.
    """
    if calibration is None:
        raise ValueError("calibration is required")

    K = np.asarray(calibration.getCameraMatrix(), dtype=np.float64).copy()
    if K.shape != (3, 3):
        raise ValueError(f"camera matrix must have shape (3, 3), got {K.shape}")

    if not scale_to_image or image_size_px is None:
        return K

    calib_w = float(getattr(calibration, "width", image_size_px[0]) or image_size_px[0])
    calib_h = float(getattr(calibration, "height", image_size_px[1]) or image_size_px[1])
    return scale_camera_matrix_to_image(
        K,
        calibration_size_px=(calib_w, calib_h),
        image_size_px=image_size_px,
    )


def scale_camera_matrix_to_image(
    K: ArrayLike,
    *,
    calibration_size_px: tuple[float, float],
    image_size_px: tuple[float, float],
) -> NDArray[np.float64]:
    """Scale a camera matrix from calibration image size to current image size.

    Parameters
    ----------
    K:
        Original 3x3 camera matrix.
    calibration_size_px:
        ``(width_px, height_px)`` associated with ``K``.
    image_size_px:
        ``(width_px, height_px)`` for the image currently being processed.
    """
    K_scaled = np.asarray(K, dtype=np.float64).copy()
    if K_scaled.shape != (3, 3):
        raise ValueError(f"camera matrix must have shape (3, 3), got {K_scaled.shape}")

    calib_w, calib_h = (float(calibration_size_px[0]), float(calibration_size_px[1]))
    image_w, image_h = (float(image_size_px[0]), float(image_size_px[1]))

    if calib_w <= 0.0 or calib_h <= 0.0 or image_w <= 0.0 or image_h <= 0.0:
        raise ValueError(
            "calibration_size_px and image_size_px must contain positive values; "
            f"got calibration_size_px={calibration_size_px!r}, image_size_px={image_size_px!r}"
        )

    K_scaled[0, :] *= image_w / calib_w
    K_scaled[1, :] *= image_h / calib_h
    return K_scaled


def range_from_bbox_width(
    *,
    bbox_w_px: float,
    focal_length_px: float,
    aircraft_width_m: float | None = None,
    target_profile: str | TargetProfile | None = None,
    min_bbox_width_px: float = _DEFAULT_MIN_BBOX_WIDTH_PX,
) -> float | None:
    """Estimate forward range from bounding-box width.

    Returns ``None`` for invalid or too-small bounding boxes.
    """
    bbox_w = float(bbox_w_px)
    fx = float(focal_length_px)
    if not np.isfinite(bbox_w) or bbox_w <= float(min_bbox_width_px):
        return None
    if not np.isfinite(fx) or abs(fx) <= _FLOAT_EPS:
        raise ValueError(f"focal_length_px must be finite and nonzero, got {focal_length_px!r}")

    width_m, _profile_name = resolve_aircraft_width_m(
        aircraft_width_m=aircraft_width_m,
        target_profile=target_profile,
    )
    return float(fx * width_m / bbox_w)


def camera_xyz_from_pixel_and_range(
    *,
    center_px: tuple[float, float],
    range_m: float,
    K: ArrayLike,
) -> NDArray[np.float64]:
    """Back-project a pixel through ``K`` and scale it by range."""
    K_arr = np.asarray(K, dtype=np.float64)
    if K_arr.shape != (3, 3):
        raise ValueError(f"camera matrix must have shape (3, 3), got {K_arr.shape}")

    r = float(range_m)
    if not np.isfinite(r):
        raise ValueError(f"range_m must be finite, got {range_m!r}")

    u = float(center_px[0])
    v = float(center_px[1])
    if not np.isfinite(u) or not np.isfinite(v):
        raise ValueError(f"center_px must be finite, got {center_px!r}")

    pix_h = np.array([u, v, 1.0], dtype=np.float64)
    return np.linalg.solve(K_arr, pix_h) * r


def estimate_single_feature_from_center_width(
    *,
    center_px: tuple[float, float],
    bbox_w_px: float,
    K: ArrayLike,
    bbox_h_px: float | None = None,
    aircraft_width_m: float | None = None,
    target_profile: str | TargetProfile | None = None,
    min_bbox_width_px: float = _DEFAULT_MIN_BBOX_WIDTH_PX,
) -> SingleFeatureEstimate | None:
    """Estimate range and camera-frame position from center pixel and box width.

    Returns ``None`` if the bounding box is invalid. Raises ``ValueError`` for
    invalid calibration or target configuration.
    """
    K_arr = np.asarray(K, dtype=np.float64)
    if K_arr.shape != (3, 3):
        raise ValueError(f"camera matrix must have shape (3, 3), got {K_arr.shape}")

    width_m, profile_name = resolve_aircraft_width_m(
        aircraft_width_m=aircraft_width_m,
        target_profile=target_profile,
    )

    range_m = range_from_bbox_width(
        bbox_w_px=bbox_w_px,
        focal_length_px=float(K_arr[0, 0]),
        aircraft_width_m=width_m,
        min_bbox_width_px=min_bbox_width_px,
    )
    if range_m is None:
        return None

    xyz_cam_m = camera_xyz_from_pixel_and_range(
        center_px=(float(center_px[0]), float(center_px[1])),
        range_m=range_m,
        K=K_arr,
    )

    bbox_h = None if bbox_h_px is None else float(bbox_h_px)
    return SingleFeatureEstimate(
        range_m=float(range_m),
        xyz_cam_m=np.asarray(xyz_cam_m, dtype=np.float64),
        center_px=(float(center_px[0]), float(center_px[1])),
        bbox_w_px=float(bbox_w_px),
        bbox_h_px=bbox_h,
        aircraft_width_m=float(width_m),
        target_name=profile_name,
    )


def estimate_single_feature_from_bbox_xyxy(
    *,
    bbox_xyxy_px: tuple[float, float, float, float],
    K: ArrayLike,
    aircraft_width_m: float | None = None,
    target_profile: str | TargetProfile | None = None,
    min_bbox_width_px: float = _DEFAULT_MIN_BBOX_WIDTH_PX,
) -> SingleFeatureEstimate | None:
    """Estimate range and camera-frame position from ``(x1, y1, x2, y2)``."""
    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy_px)
    bbox_w_px = x2 - x1
    bbox_h_px = y2 - y1
    center_px = (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    return estimate_single_feature_from_center_width(
        center_px=center_px,
        bbox_w_px=bbox_w_px,
        bbox_h_px=bbox_h_px,
        K=K,
        aircraft_width_m=aircraft_width_m,
        target_profile=target_profile,
        min_bbox_width_px=min_bbox_width_px,
    )


__all__ = [
    "CUB_TARGET",
    "DEFAULT_SINGLE_FEATURE_TARGET",
    "OSPREY_TARGET",
    "SingleFeatureEstimate",
    "TARGET_PROFILES",
    "TargetProfile",
    "camera_matrix_from_calibration",
    "camera_xyz_from_pixel_and_range",
    "estimate_single_feature_from_bbox_xyxy",
    "estimate_single_feature_from_center_width",
    "range_from_bbox_width",
    "resolve_aircraft_width_m",
    "resolve_target_profile",
    "scale_camera_matrix_to_image",
]
