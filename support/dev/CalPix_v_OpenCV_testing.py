import time
import numpy as np
from numpy.typing import NDArray
import cv2
from collections import deque

from support.io.Calibration import default_864_cam, distort_points_px, undistort_points_px, Calibration
from support.runtime.PixelHandler import Pixel

def make_opencv_mats(cal: Calibration):
    K = np.array([[cal.fx, 0.0, cal.cx],
                  [0.0, cal.fy, cal.cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.array([cal.k1, cal.k2, cal.p1, cal.p2, cal.k3], dtype=np.float64)
    return K, dist


# ---------- Scalar (Pixel-loop) implementations ----------

def ours_undistort_batch_scalar(cal: Calibration, pts_px_dist: NDArray):
    """Scalar reference: per-point Pixel + Calibration.undistort_point."""
    out = np.empty_like(pts_px_dist, dtype=np.float64)
    for i in range(pts_px_dist.shape[0]):
        p = Pixel(pix_coords=pts_px_dist[i], already_undistorted=False)
        cal.undistort_point(p)
        out[i] = p.pix_coords
    return out


def ours_distort_batch_scalar(cal: Calibration, pts_px_und: NDArray):
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

def build_undistort_maps(K, dist, w, h):
    """
    Build OpenCV undistort rectify maps ONCE.
    map1/map2: for each UNDISTORTED output pixel (u), gives DISTORTED source pixel (d) to sample.
    """
    map1, map2 = cv2.initUndistortRectifyMap(
        K, dist, None, K, (w, h), cv2.CV_32FC1
    )
    return map1, map2


def undistort_image_via_remap(img_dist, map1, map2):
    """
    Proper usage: undistort an IMAGE via remap.
    """
    return cv2.remap(img_dist, map1, map2, interpolation=cv2.INTER_LINEAR)


def build_inverse_point_lut_from_maps(map1, map2):
    """
    Build an inverse LUT for POINT undistortion:
        given a distorted integer pixel (dx,dy), return an undistorted pixel (ux,uy).

    We "splat" each undistorted pixel (ux,uy) into its mapped distorted location (dx,dy),
    then fill holes with a nearest-neighbor fill (multi-source BFS).
    """
    h, w = map1.shape[:2]

    inv_u_x = np.full((h, w), np.nan, dtype=np.float32)
    inv_u_y = np.full((h, w), np.nan, dtype=np.float32)

    # Forward map: u -> d. Build inverse bins: d -> u by splatting.
    # For each undistorted pixel u=(ux,uy), map gives distorted sample d=(dx,dy).
    ux_grid, uy_grid = np.meshgrid(np.arange(w, dtype=np.float32),
                                   np.arange(h, dtype=np.float32))

    dx = np.rint(map1).astype(np.int32)
    dy = np.rint(map2).astype(np.int32)

    valid = (dx >= 0) & (dx < w) & (dy >= 0) & (dy < h) & np.isfinite(map1) & np.isfinite(map2)

    dxv = dx[valid]
    dyv = dy[valid]
    uxv = ux_grid[valid]
    uyv = uy_grid[valid]

    inv_u_x[dyv, dxv] = uxv
    inv_u_y[dyv, dxv] = uyv

    # --- Fill holes (nearest-neighbor) ---
    # Multi-source BFS from known pixels over 4-neighborhood.
    known = np.isfinite(inv_u_x)

    q = deque()
    # Seed queue with all known pixels
    ys, xs = np.nonzero(known)
    for y, x in zip(ys.tolist(), xs.tolist()):
        q.append((y, x))

    # Visited is same as known; we will "grow" it.
    visited = known.copy()

    # 4-connected neighbors
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while q:
        y, x = q.popleft()
        ux = inv_u_x[y, x]
        uy = inv_u_y[y, x]
        for dy_, dx_ in nbrs:
            yn = y + dy_
            xn = x + dx_
            if 0 <= yn < h and 0 <= xn < w and not visited[yn, xn]:
                inv_u_x[yn, xn] = ux
                inv_u_y[yn, xn] = uy
                visited[yn, xn] = True
                q.append((yn, xn))

    return inv_u_x, inv_u_y


def opencv_point_undistort_via_inverse_lut(pts_px_dist, inv_u_x, inv_u_y):
    """
    Point-wise distorted -> undistorted using the precomputed inverse LUT (nearest pixel lookup).
    """
    h, w = inv_u_x.shape[:2]
    x = np.clip(np.rint(pts_px_dist[:, 0]).astype(np.int32), 0, w - 1)
    y = np.clip(np.rint(pts_px_dist[:, 1]).astype(np.int32), 0, h - 1)
    und_x = inv_u_x[y, x].astype(np.float64)
    und_y = inv_u_y[y, x].astype(np.float64)
    return np.column_stack([und_x, und_y])


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


def test_cal(cal: Calibration):
    cv2.setUseOptimized(True)

    K, dist = make_opencv_mats(cal)

    # ------------------------------------------------------------------
    # TRUTHY UNDISTORT TEST:
    # Start from known UNDISTORTED pixel points, distort them, then recover.
    # This produces "ground truth" undistorted points for error reporting.
    # ------------------------------------------------------------------
    N_truth = 50_000
    rng = np.random.default_rng(0)

    # --- Build OpenCV maps ONCE and an inverse point LUT ONCE ---
    t0 = time.perf_counter()
    map1, map2 = build_undistort_maps(K, dist, cal.width, cal.height)
    inv_u_x, inv_u_y = build_inverse_point_lut_from_maps(map1, map2)
    t_build_lut = time.perf_counter() - t0

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
    cv_lut_und = opencv_point_undistort_via_inverse_lut(pts_px_dist, inv_u_x, inv_u_y)

    print("\nUNDISTORT accuracy vs TRUTH (starting from known undistorted -> distort -> undistort)")
    print(" scalar (2fp+newton):        RMS px err:", rms_err(ours_und_scalar, pts_px_und_truth))
    print("                             Max px err:", max_err(ours_und_scalar, pts_px_und_truth))
    print(" vectorized (precise):       RMS px err:", rms_err(ours_und_vec_prec, pts_px_und_truth))
    print("                             Max px err:", max_err(ours_und_vec_prec, pts_px_und_truth))
    print(" vectorized (opencv-5fp):    RMS px err:", rms_err(ours_und_vec_cv, pts_px_und_truth))
    print("                             Max px err:", max_err(ours_und_vec_cv, pts_px_und_truth))
    print(" cv   undistortPoints:       RMS px err:", rms_err(cv_und, pts_px_und_truth))
    print("                             Max px err:", max_err(cv_und, pts_px_und_truth))
    print(" cv undistort (inverse LUT): RMS px err:", rms_err(cv_lut_und, pts_px_und_truth))
    print("                             Max px err:", max_err(cv_lut_und, pts_px_und_truth))

    # Also report how far each method is from OpenCV undistortPoints
    print("\nUNDISTORT agreement vs OpenCV (same distorted input)")
    print("  scalar (2fp+newton):       RMS px err:", rms_err(ours_und_scalar, cv_und))
    print("                             Max px err:", max_err(ours_und_scalar, cv_und))
    print("  vectorized (precise):      RMS px err:", rms_err(ours_und_vec_prec, cv_und))
    print("                             Max px err:", max_err(ours_und_vec_prec, cv_und))
    print("  vectorized (opencv-5fp):   RMS px err:", rms_err(ours_und_vec_cv, cv_und))
    print("                             Max px err:", max_err(ours_und_vec_cv, cv_und))
    print(" cv undistort (inverse LUT): RMS px err:", rms_err(cv_lut_und, cv_und))
    print("                             Max px err:", max_err(cv_lut_und, cv_und))

    # ------------------------------------------------------------------
    # Timing: use a smaller subset so Python scalar loop is tolerable
    # ------------------------------------------------------------------
    pts_px_dist_small = pts_px_dist[:20_000]
    pts_px_und_small_truth = pts_px_und_truth[:20_000]

    t_scalar_und = bench(lambda: ours_undistort_batch_scalar(cal, pts_px_dist_small), iters=10)
    t_vec_und_prec = bench(lambda: undistort_points_px(cal, pts_px_dist_small, mode="precise"), iters=800)
    t_vec_und_cv = bench(lambda: undistort_points_px(cal, pts_px_dist_small, mode="opencv"), iters=800)
    t_cv_und = bench(lambda: opencv_undistort_batch(K, dist, pts_px_dist_small), iters=800)

    t_scalar_dist = bench(lambda: ours_distort_batch_scalar(cal, pts_px_und_small_truth), iters=10)
    t_vec_dist = bench(lambda: distort_points_px(cal, pts_px_und_small_truth), iters=800)
    t_cv_dist = bench(lambda: opencv_distort_batch(K, dist, pts_px_und_small_truth), iters=800)
    # --- Timing: point-wise inverse LUT (dist->und), map built once ---
    t_cv_lut_und = bench(lambda: opencv_point_undistort_via_inverse_lut(pts_px_dist_small, inv_u_x, inv_u_y),iters=800)

    N = pts_px_dist_small.shape[0]
    print("\nTIMING (avg seconds per call)   N=", N)
    print(f"  ours undistort scalar:         {t_scalar_und:.6f}")
    print(f"  ours undistort vec (precise):  {t_vec_und_prec:.6f}")
    print(f"  ours undistort vec (opencv):   {t_vec_und_cv:.6f}")
    print(f"  cv   undistort:                {t_cv_und:.6f}")
    print(f"  ours distort scalar:           {t_scalar_dist:.6f}")
    print(f"  ours distort vec:              {t_vec_dist:.6f}")
    print(f"  cv   distort:                  {t_cv_dist:.6f}")
    print(f"  cv   inverse LUT build (one-time): {t_build_lut:.6f} s")
    print(f"  cv   undistort (inverse LUT):  {t_cv_lut_und:.6f}")

    print("\nTHROUGHPUT (points/sec)")
    print(f"  ours undistort scalar:         {N / t_scalar_und:,.0f}")
    print(f"  ours undistort vec (precise):  {N / t_vec_und_prec:,.0f}")
    print(f"  ours undistort vec (opencv):   {N / t_vec_und_cv:,.0f}")
    print(f"  cv   undistort:                {N / t_cv_und:,.0f}")
    print(f"  ours distort scalar:           {N / t_scalar_dist:,.0f}")
    print(f"  ours distort vec:              {N / t_vec_dist:,.0f}")
    print(f"  cv   distort:                  {N / t_cv_dist:,.0f}")
    print(f"  cv   undistort (inverse LUT):  {N / t_cv_lut_und:,.0f}")
    print("======================================================\n")

def main():
    cal = default_864_cam()
    np.set_printoptions(suppress=True)
    # sweep
    for strength in ["low", "medium", "high"]:
        for profile in ["radial_only", "tangential_only", "mixed"]:
            cal.randomize(keep_intrinsics=False, strength=strength, profile=profile)
            assert cal.validCal

            print("Testing Calibration: ")
            print(f"Strength: {strength}, Profile: {profile}")
            print(f"Projection Matrix:\n{cal.getCameraMatrix()}")
            print(f"Distortion Coefficients:\n{cal.getDistortion()}")
            print(f"Image Size, Width: {cal.width}px x Height: {cal.height}px")
            print(f"Maximum Distortion: {cal.max_corner_distortion_px()[0]}")
            min_detJ, min_abs_detJ, frac_neg_detJ, max_kappa = cal.orientation_preservation_metrics()
            print(f"Orientation Preservation:")
            print(f"  Min Jacobian Det:         {min_detJ}")
            print(f"  Closest Jacobian to Zero: {min_abs_detJ}")
            print(f"  Image Flip Fraction:      {frac_neg_detJ}")
            print(f"  Max Kappa:                {max_kappa}")
            print()
            test_cal(cal)


if __name__ == "__main__":
    main()
