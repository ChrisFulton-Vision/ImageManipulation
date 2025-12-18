import time
import numpy as np
import cv2

from Calibration import default_864_cam, distort_points_px, undistort_points_px
from PixelHandler import Pixel


def make_opencv_mats(cal):
    K = np.array([[cal.fx, 0.0, cal.cx],
                  [0.0, cal.fy, cal.cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.array([cal.k1, cal.k2, cal.p1, cal.p2, cal.k3], dtype=np.float64)
    return K, dist


# ---------- Scalar (Pixel-loop) implementations ----------

def ours_undistort_batch_scalar(cal, pts_px_dist):
    """Scalar reference: per-point Pixel + Calibration.undistort_point."""
    out = np.empty_like(pts_px_dist, dtype=np.float64)
    for i in range(pts_px_dist.shape[0]):
        p = Pixel(pix_coords=pts_px_dist[i], already_undistorted=False)
        cal.undistort_point(p)
        out[i] = p.pix_coords
    return out


def ours_distort_batch_scalar(cal, pts_px_und):
    """Scalar reference: per-point Pixel + havePix_needNorm + distort_point."""
    out = np.empty_like(pts_px_und, dtype=np.float64)
    for i in range(pts_px_und.shape[0]):
        p = Pixel(pix_coords=pts_px_und[i], already_undistorted=True)
        cal.havePix_needNorm(p)
        cal.distort_point(p)
        out[i] = p.pix_coords
    return out


# ---------- OpenCV implementations ----------

def opencv_undistort_batch(K, dist, pts_px_dist):
    pts = pts_px_dist.reshape(-1, 1, 2).astype(np.float64)
    und = cv2.undistortPoints(pts, K, dist, P=K)
    return und.reshape(-1, 2)


def opencv_distort_batch(K, dist, pts_px_und):
    """
    Distort by projecting (x,y,1) with rvec=tvec=0.
    This maps undistorted pixel coordinates into distorted pixels using the same
    distortion model OpenCV uses in projectPoints.
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (pts_px_und[:, 0] - cx) / fx
    y = (pts_px_und[:, 1] - cy) / fy
    obj = np.column_stack([x, y, np.ones_like(x)]).astype(np.float64)

    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    img, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    return img.reshape(-1, 2)


# ---------- Metrics & timing ----------

def rms_err(a, b):
    d = a - b
    return np.sqrt(np.mean(np.sum(d * d, axis=1)))


def max_err(a, b):
    return np.max(np.abs(a - b))


def bench(fn, iters=50, warmup=5):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def main():
    cv2.setUseOptimized(True)

    cal = default_864_cam()
    assert cal.validCal, "Calibration not valid"
    K, dist = make_opencv_mats(cal)

    # ------------------------------------------------------------------
    # TRUTHY UNDISTORT TEST:
    # Start from known UNDISTORTED pixel points, distort them, then recover.
    # This produces "ground truth" undistorted points for error reporting.
    # ------------------------------------------------------------------
    N_truth = 50_000
    rng = np.random.default_rng(0)

    # Make undistorted pixels in-frame. (These are "truth" targets.)
    pts_px_und_truth = np.column_stack([
        rng.uniform(0, cal.width, size=N_truth),
        rng.uniform(0, cal.height, size=N_truth)
    ]).astype(np.float64)

    # Distort them using OpenCV and our forward model.
    pts_px_dist_cv = opencv_distort_batch(K, dist, pts_px_und_truth)
    pts_px_dist_ours = distort_points_px(cal, pts_px_und_truth)

    # Confirm forward distortion agreement (should be ~1e-13 px)
    print("FORWARD DISTORTION agreement (our distort_points_px vs OpenCV projectPoints)")
    print("  RMS px err:", rms_err(pts_px_dist_ours, pts_px_dist_cv))
    print("  Max px err:", max_err(pts_px_dist_ours, pts_px_dist_cv))

    # Undistort the SAME distorted set using each method and compare to truth
    # Use OpenCV-distorted points as the common input for fairness.
    pts_px_dist = pts_px_dist_cv

    ours_und_scalar = ours_undistort_batch_scalar(cal, pts_px_dist)
    ours_und_vec_prec = undistort_points_px(cal, pts_px_dist, mode="precise")
    ours_und_vec_cv = undistort_points_px(cal, pts_px_dist, mode="opencv")
    cv_und = opencv_undistort_batch(K, dist, pts_px_dist)

    print("\nUNDISTORT accuracy vs TRUTH (starting from known undistorted -> distort -> undistort)")
    print("  scalar (2fp+newton):      RMS px err:", rms_err(ours_und_scalar, pts_px_und_truth))
    print("                            Max px err:", max_err(ours_und_scalar, pts_px_und_truth))
    print("  vectorized (precise):     RMS px err:", rms_err(ours_und_vec_prec, pts_px_und_truth))
    print("                            Max px err:", max_err(ours_und_vec_prec, pts_px_und_truth))
    print("  vectorized (opencv-5fp):  RMS px err:", rms_err(ours_und_vec_cv, pts_px_und_truth))
    print("                            Max px err:", max_err(ours_und_vec_cv, pts_px_und_truth))
    print("  cv   undistortPoints:     RMS px err:", rms_err(cv_und, pts_px_und_truth))
    print("                            Max px err:", max_err(cv_und, pts_px_und_truth))

    # Also report how far each method is from OpenCV (helps confirm compatibility mode)
    print("\nUNDISTORT agreement vs OpenCV (same distorted input)")
    print("  scalar (2fp+newton):      RMS px err:", rms_err(ours_und_scalar, cv_und))
    print("                            Max px err:", max_err(ours_und_scalar, cv_und))
    print("  vectorized (precise):     RMS px err:", rms_err(ours_und_vec_prec, cv_und))
    print("                            Max px err:", max_err(ours_und_vec_prec, cv_und))
    print("  vectorized (opencv-5fp):  RMS px err:", rms_err(ours_und_vec_cv, cv_und))
    print("                            Max px err:", max_err(ours_und_vec_cv, cv_und))

    # ------------------------------------------------------------------
    # Timing: use a smaller subset so Python scalar loop is tolerable
    # ------------------------------------------------------------------
    pts_px_dist_small = pts_px_dist[:20_000]
    pts_px_und_small_truth = pts_px_und_truth[:20_000]

    t_scalar_und = bench(lambda: ours_undistort_batch_scalar(cal, pts_px_dist_small), iters=10)
    t_vec_und_prec = bench(lambda: undistort_points_px(cal, pts_px_dist_small, mode="precise"), iters=300)
    t_vec_und_cv = bench(lambda: undistort_points_px(cal, pts_px_dist_small, mode="opencv"), iters=300)
    t_cv_und = bench(lambda: opencv_undistort_batch(K, dist, pts_px_dist_small), iters=800)

    t_scalar_dist = bench(lambda: ours_distort_batch_scalar(cal, pts_px_und_small_truth), iters=10)
    t_vec_dist = bench(lambda: distort_points_px(cal, pts_px_und_small_truth), iters=800)
    t_cv_dist = bench(lambda: opencv_distort_batch(K, dist, pts_px_und_small_truth), iters=800)

    N = pts_px_dist_small.shape[0]
    print("\nTIMING (avg seconds per call)   N=", N)
    print(f"  ours undistort scalar:         {t_scalar_und:.6f}")
    print(f"  ours undistort vec (precise):  {t_vec_und_prec:.6f}")
    print(f"  ours undistort vec (opencv):   {t_vec_und_cv:.6f}")
    print(f"  cv   undistort:                {t_cv_und:.6f}")
    print(f"  ours distort scalar:           {t_scalar_dist:.6f}")
    print(f"  ours distort vec:              {t_vec_dist:.6f}")
    print(f"  cv   distort:                  {t_cv_dist:.6f}")

    print("\nTHROUGHPUT (points/sec)")
    print(f"  ours undistort scalar:         {N / t_scalar_und:,.0f}")
    print(f"  ours undistort vec (precise):  {N / t_vec_und_prec:,.0f}")
    print(f"  ours undistort vec (opencv):   {N / t_vec_und_cv:,.0f}")
    print(f"  cv   undistort:                {N / t_cv_und:,.0f}")
    print(f"  ours distort scalar:           {N / t_scalar_dist:,.0f}")
    print(f"  ours distort vec:              {N / t_vec_dist:,.0f}")
    print(f"  cv   distort:                  {N / t_cv_dist:,.0f}")


if __name__ == "__main__":
    main()
