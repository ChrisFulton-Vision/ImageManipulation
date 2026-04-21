# CalPix_v_OpenCV_fisheyeTesting.py

import time
import numpy as np
import cv2
from numpy.typing import NDArray

from support.vision.calibration import (
    Calibration,
    distort_points_px,
    undistort_points_px,
    default_fisheye_cam,
)
from support.runtime.pixel_handler import Pixel


# ============================================================
# OpenCV fisheye helpers
# ============================================================

def make_opencv_fisheye_mats(cal: Calibration):
    """
    Build OpenCV fisheye camera matrices from Calibration.
    """
    assert cal.fisheye
    K = np.array(
        [[cal.fx, 0.0, cal.cx],
         [0.0, cal.fy, cal.cy],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    D = np.array([cal.k1, cal.k2, cal.k3, cal.k4], dtype=np.float64)
    return K, D


def opencv_fisheye_undistort_batch(K, D, pts_px_dist):
    """
    OpenCV fisheye undistortPoints wrapper.
    """
    pts = np.asarray(pts_px_dist, dtype=np.float64).reshape(-1, 1, 2)
    und = cv2.fisheye.undistortPoints(pts, K, D, P=K)
    return und.reshape(-1, 2)


def opencv_fisheye_distort_batch(K, D, pts_px_und):
    """
    Correct OpenCV fisheye distortion:
    input: UNDISTORTED PIXELS
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # pixels → normalized
    xn = (pts_px_und[:, 0] - cx) / fx
    yn = (pts_px_und[:, 1] - cy) / fy

    pts_norm = np.column_stack([xn, yn]).reshape(-1, 1, 2).astype(np.float64)

    pts_dist = cv2.fisheye.distortPoints(pts_norm, K, D)
    return pts_dist.reshape(-1, 2)



# ============================================================
# Scalar (Pixel-based) references
# ============================================================

def ours_undistort_batch_scalar(cal: Calibration, pts_px_dist: NDArray):
    out = np.empty_like(pts_px_dist, dtype=np.float64)
    for i in range(pts_px_dist.shape[0]):
        p = Pixel(pix_coords=pts_px_dist[i], already_undistorted=False)
        cal.undistort_point(p)
        out[i] = p.pix_coords
    return out


def ours_distort_batch_scalar(cal: Calibration, pts_px_und: NDArray):
    out = np.empty_like(pts_px_und, dtype=np.float64)
    for i in range(pts_px_und.shape[0]):
        p = Pixel(pix_coords=pts_px_und[i], already_undistorted=True)
        cal.havePix_needNorm(p)
        cal.distort_point(p)
        out[i] = p.pix_coords
    return out


# ============================================================
# Metrics & timing
# ============================================================

def bench(fn, iters=50, warmup=5):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def err_stats(est: np.ndarray, truth: np.ndarray, mask=None):
    e = est - truth
    en = np.sqrt(e[:, 0] ** 2 + e[:, 1] ** 2)
    if mask is not None:
        en = en[mask]
    if en.size == 0:
        return dict(rms=np.nan, max=np.nan, p99=np.nan)
    return dict(
        rms=float(np.sqrt(np.mean(en * en))),
        max=float(np.max(en)),
        p99=float(np.percentile(en, 99)),
    )


def in_bounds_mask(pts_px, width, height):
    return (
        (pts_px[:, 0] >= 0.0) & (pts_px[:, 0] <= width - 1) &
        (pts_px[:, 1] >= 0.0) & (pts_px[:, 1] <= height - 1)
    )


# ============================================================
# Main test harness
# ============================================================

def run_fisheye_point_tests(
    *,
    cal: Calibration | None = None,
    n_pts: int = 200_000,
    seed: int = 0,
    iters: int = 50,
    undistort_modes: tuple[str, ...] = ("newton", "precise"),
):
    """
    Compare OpenCV fisheye vs our fisheye calibration.
    """
    if cal is None:
        cal = default_fisheye_cam()
        cal.fx = cal.fy = 250.0
        cal.cx = 960
        cal.cy = 540
        cal.k1 = -.35
        cal.k2 = 0.11
        cal.k3 = -0.04
        cal.k4 = 0.008

    assert cal.validCal and cal.fisheye

    rng = np.random.default_rng(seed)

    # Random UNDISTORTED points
    pts_und = np.column_stack([
        rng.uniform(0.0, cal.width - 1.0, n_pts),
        rng.uniform(0.0, cal.height - 1.0, n_pts),
    ]).astype(np.float64)

    # Generate distorted "measurements" using OpenCV
    K, D = make_opencv_fisheye_mats(cal)
    pts_dist = opencv_fisheye_distort_batch(K, D, pts_und)

    mask_dist = in_bounds_mask(pts_dist, cal.width, cal.height)

    # ---------- UNDISTORT ----------
    cv_und = opencv_fisheye_undistort_batch(K, D, pts_dist)
    t_cv_und = bench(lambda: opencv_fisheye_undistort_batch(K, D, pts_dist), iters)

    und_results = {}
    for mode in undistort_modes:
        ours_und = undistort_points_px(cal, pts_dist, mode=mode)
        und_stats = err_stats(cv_und, ours_und, mask_dist)
        t_ours_und = bench(lambda m=mode: undistort_points_px(cal, pts_dist, mode=m), iters)
        und_results[str(mode)] = {
            "opencv_time": t_cv_und,
            "ours_time": t_ours_und,
            **und_stats,
        }

    # ---------- DISTORT ----------
    cv_dist = opencv_fisheye_distort_batch(K, D, pts_und)
    ours_dist = distort_points_px(cal, pts_und)

    mask_und = in_bounds_mask(pts_und, cal.width, cal.height)
    dist_stats = err_stats(cv_dist, ours_dist, mask_und)

    t_cv_dist = bench(lambda: opencv_fisheye_distort_batch(K, D, pts_und), iters)
    t_ours_dist = bench(lambda: distort_points_px(cal, pts_und), iters)

    return {
        "UNDISTORT": und_results,
        "DISTORT": {
            "opencv_time": t_cv_dist,
            "ours_time": t_ours_dist,
            **dist_stats,
        },
    }


# ============================================================
# CLI hook
# ============================================================

if __name__ == "__main__":
    results = run_fisheye_point_tests()
    for task, stats in results.items():
        print(f"\n[{task}]")
        if task == "UNDISTORT":
            for mode, mstats in stats.items():
                print(f"  (mode={mode})")
                for k, v in mstats.items():
                    print(f"    {k:>12s}: {v}")
        else:
            for k, v in stats.items():
                print(f"  {k:>12s}: {v}")
