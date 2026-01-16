import os
import time
import numpy as np
from numpy.typing import NDArray
import cv2
from collections import deque
from support.mathHelpers.include_numba import _njit as njit, prange

from support.vision.calibration import default_864_cam, distort_points_px, undistort_points_px, Calibration
from support.runtime.pixel_handler import Pixel

METHOD_COLORS = {
    # UNDISTORT
    "cv undistort (inverse LUT)":     "#1f77b4",  # blue
    "cv undistortPoints":             "#ff7f0e",  # orange
    "ours undistort scalar":          "#2ca02c",  # green
    "ours undistort vec (5fp)":    "#d62728",  # red
    "ours undistort vec (2fp+N)":   "#9467bd",  # purple

    # DISTORT
    "cv projectPoints":               "#8c564b",  # brown
    "ours distort scalar":            "#17becf",  # cyan
    "ours distort vec":               "#bcbd22",  # olive

    "fwd LUT nearest":                "#7f7f7f",  # gray
    "fwd LUT bilinear":               "#e377c2",  # pink
    "fwd LUT bilinear (numba)":       "#ff9896",  # light red
    "fwd LUT bilinear (numba ss)":    "#c5b0d5",  # light purple
}

def prejit_undistort_kernels():
    cal = Calibration().randomize(
        rng=np.random.default_rng(0),
        width=864, height=864,
        keep_intrinsics=False,
        profile="mixed",
        strength="high",
        ensure_invertible=True,
    )
    K, dist = make_opencv_mats(cal)

    # representative points
    pts_und = np.array([[100.0, 200.0],
                        [400.0, 500.0],
                        [800.0, 100.0]], dtype=np.float64)
    pts_dist = opencv_distort_batch(K, dist, pts_und).astype(np.float64)

    # Force-compile both code paths
    _ = undistort_points_px(cal, pts_dist, mode="opencv")
    _ = undistort_points_px(cal, pts_dist, mode="precise")


def plot_tradeoff_pairs(
    paired_rows,
    *,
    title="UNDISTORT: 5FP vs 2FP+N across calibrations",
    connect_every=10,
    sort_by="severity",          # "severity" or None
    noise_floor=1e-13,           # equivalence floor for RMS
    cmap_name="turbo",
):
    """
    paired_rows: list of dicts, each corresponds to ONE calibration:
      {
        "cal_id": int,
        "severity": float,        # e.g. max_corner_distortion_px
        "run_id": str,            # optional
        "m5": {"time_s":..., "rms":..., "method": "ours undistort vec (5fp)"},
        "mH": {"time_s":..., "rms":..., "method": "ours undistort vec (2fp+N)"},
      }

    Produces:
      - scatter of (time, RMS) for both methods, colored by severity
      - connecting line between paired points every Nth calibration
      - textbox: % faster, % more-precise-or-equivalent, % both
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    m5_name = "ours undistort vec (5fp)"
    mh_name = "ours undistort vec (2fp+N)"

    rows = list(paired_rows)
    if sort_by == "severity":
        rows.sort(key=lambda r: r["severity"])

    # ---- arrays ----
    x5 = np.asarray([r["m5"]["time_s"] for r in rows], dtype=float)
    y5 = np.asarray([r["m5"]["rms"]    for r in rows], dtype=float)

    xh = np.asarray([r["mH"]["time_s"] for r in rows], dtype=float)
    yh = np.asarray([r["mH"]["rms"]    for r in rows], dtype=float)

    sev = np.asarray([r["severity"] for r in rows], dtype=float)

    # ---- dominance / stats ----
    faster = xh < x5

    # "more precise OR essentially equivalent"
    # - counts as better if yh <= y5
    # - counts as equivalent if both are at/below noise floor
    precise_or_eq = (yh <= y5) | ((y5 <= noise_floor) & (yh <= noise_floor))

    both = faster & precise_or_eq

    n = max(1, len(rows))
    pct_faster = 100.0 * float(np.sum(faster)) / n
    pct_prec   = 100.0 * float(np.sum(precise_or_eq)) / n
    pct_both   = 100.0 * float(np.sum(both)) / n

    # ---- severity colormap ----
    # Robust norm so a few extremes don't blow out the scale
    vmin = float(np.percentile(sev, 2.0))
    vmax = float(np.percentile(sev, 98.0))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.min(sev)), float(np.max(sev))
        if vmax <= vmin:
            vmax = vmin + 1.0

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl.cm.get_cmap(cmap_name)

    plt.figure(figsize=(14, 6))
    ax = plt.gca()

    # ---- scatters (same colors, different markers) ----
    sc5 = ax.scatter(
        x5, y5,
        c=sev, cmap=cmap, norm=norm,
        marker='o', alpha=0.70, s=60,
        label=m5_name,
        linewidths=0.0,
    )
    sch = ax.scatter(
        xh, yh,
        c=sev, cmap=cmap, norm=norm,
        marker='s', alpha=0.70, s=60,
        label=mh_name,
        linewidths=0.0,
    )

    # ---- connecting lines every Nth calibration ----
    step = max(1, int(connect_every))
    for i in range(0, len(rows), step):
        # use same severity color for the connector, very faint
        col = cmap(norm(sev[i]))
        ax.plot([x5[i], xh[i]], [y5[i], yh[i]], color=col, alpha=0.4, linewidth=1.0)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Avg time per call (seconds)")
    ax.set_ylabel("RMS px error")
    ax.set_title(title)
    xmax = np.percentile(np.r_[x5, xh], 99.5)
    ax.set_xlim(left=None, right=xmax * 1.05)
    ax.grid(True, which="both")

    # Legend
    leg = ax.legend(loc="upper right")
    ax.add_artist(leg)

    # Colorbar for severity
    cbar = plt.colorbar(sc5, ax=ax, pad=0.02)
    cbar.set_label("Distortion severity (max corner displacement, px)")

    # ---- textbox under legend ----
    stats_txt = (
        f"2FP+N faster: {pct_faster:5.1f}%\n"
        f"2FP+N ≥ precision (≤ {noise_floor:.0e} floor): {pct_prec:5.1f}%\n"
        f"Both: {pct_both:5.1f}%"
    )

    # Place under legend in axes coords (top-right-ish)
    ax.text(
        0.985, 0.86, stats_txt,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="0.7"),
    )

    plt.tight_layout()
    plt.show()


def err_stats(est: np.ndarray, truth: np.ndarray, mask: np.ndarray | None = None):
    """
    Returns RMS, Max, p99, p99.9 of pointwise Euclidean error.
    If mask is provided, uses only masked points.
    """
    e = est - truth
    en = np.sqrt(e[:, 0] * e[:, 0] + e[:, 1] * e[:, 1])

    if mask is not None:
        en = en[mask]

    if en.size == 0:
        return dict(rms=np.nan, max=np.nan, p99=np.nan, p999=np.nan)

    return dict(
        rms=float(np.sqrt(np.mean(en * en))),
        max=float(np.max(en)),
        p99=float(np.percentile(en, 99)),
        p999=float(np.percentile(en, 99.9)),
    )


def plot_tradeoff(all_results, title_prefix="speed vs precision (across all runs)"):
    """
    all_results: list[dict] with keys:
      task: "UNDISTORT" or "DISTORT"
      method: str
      time_s: float
      rms: float
      run_id: str   # e.g. "high|mixed"
      (optional) source: str  # if you add it; otherwise method drives legend
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # stable color assignment per method
    all_methods = sorted(set(r["method"] for r in all_results))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    method_to_color = {
        m: color_cycle[i % len(color_cycle)]
        for i, m in enumerate(all_methods)
    }

    # ---- marker mapping for run_id ----
    run_ids = sorted(set(r["run_id"] for r in all_results if "run_id" in r))
    markers = ['o', 's', '^', 'D', 'P', 'X', 'v', '<', '>', '*', 'h', '8']
    run_to_marker = {rid: markers[i % len(markers)] for i, rid in enumerate(run_ids)}

    # ---- method ordering (keeps legend stable) ----
    def _methods_for_task(rows):
        return sorted(set(r["method"] for r in rows))

    # ---- helper: robust-ish "outlier" rule within each method ----
    # Labels points that are much worse than that method's typical RMS.
    def _outlier_mask(method_rms):
        med = np.median(method_rms)
        # label anything > 10x median (tweak if you want)
        return method_rms > (10.0 * med)

    for task in ["UNDISTORT", "DISTORT"]:
        rows = [r for r in all_results if r["task"] == task]
        if not rows:
            continue
        missing = {r["method"] for r in rows} - METHOD_COLORS.keys()
        if missing:
            raise KeyError(f"Missing color definitions for methods: {missing}")

        plt.figure(figsize=(14, 6))

        # plot each method in its own color; within that, marker shape indicates run_id
        for method in _methods_for_task(rows):
            #  OpenCV uses exact method down to numerical precision
            if task == "DISTORT" and method == "cv projectPoints":
                continue

            rr = [r for r in rows if r["method"] == method]
            xs = np.array([r["time_s"] for r in rr])
            ys = np.array([r["rms"] for r in rr])
            rids = [r["run_id"] for r in rr]

            color = METHOD_COLORS[method]

            first = True
            for rid in sorted(set(rids)):
                sel = [i for i, x in enumerate(rids) if x == rid]
                label = method if first else None

                plt.scatter(
                    xs[sel],
                    ys[sel],
                    color=color,
                    marker=run_to_marker[rid],
                    alpha=0.7,
                    label=label,
                )
                first = False

            # # annotate outliers with (run_id + method)
            # mask = _outlier_mask(ys)
            # for i in np.where(mask)[0]:
            #     plt.annotate(
            #         f"{rr[i]['run_id']} | {method}",
            #         (xs[i], ys[i]),
            #         textcoords="offset points",
            #         xytext=(6, 6),
            #         fontsize=8,
            #     )

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Avg time per call (seconds)")
        plt.ylabel("RMS px error")
        plt.title(f"{task}: {title_prefix}")
        plt.grid(True, which="both")

        # Legend 1: methods (colors)
        leg_methods = plt.legend(
            title="Method (color)",
            fontsize=8,
            title_fontsize=9,
            loc="upper right",
            bbox_to_anchor=(1.0, 1.0),  # top-right corner
        )
        plt.gca().add_artist(leg_methods)

        # Legend 2: run markers
        # Make a compact marker legend with dummy handles:
        handles = []
        labels = []
        for rid in run_ids:
            h = plt.Line2D(
                [],
                [],
                color="black",
                marker=run_to_marker[rid],
                linestyle="None",
                markersize=7,
            )
            handles.append(h)
            labels.append(rid)

        leg_runs = plt.legend(
            handles,
            labels,
            title="Run (marker)",
            fontsize=8,
            title_fontsize=9,
            loc="upper right",
            bbox_to_anchor=(1.0, 0.72),  # ↓ shift down; tweak if needed
        )

        plt.tight_layout()
        plt.show()



def make_opencv_mats(cal: Calibration):
    K = np.array([[cal.fx, 0.0, cal.cx],
                  [0.0, cal.fy, cal.cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.array([cal.k1, cal.k2, cal.p1, cal.p2, cal.k3], dtype=np.float64)
    return K, dist

def build_forward_maps_supersampled_opencv(K, dist, w, h, s):
    """
    Build supersampled forward maps using OpenCV's initUndistortRectifyMap.

    Returns:
      fwd_d_x_ss, fwd_d_y_ss: float32 arrays of shape (h*s, w*s)
      t_build: seconds
    """
    Kss = K.copy()
    Kss[0, 0] *= s  # fx
    Kss[1, 1] *= s  # fy
    Kss[0, 2] *= s  # cx
    Kss[1, 2] *= s  # cy

    t0 = time.perf_counter()
    map1_ss, map2_ss = cv2.initUndistortRectifyMap(
        Kss, dist, None, Kss, (w * s, h * s), cv2.CV_32FC1
    )
    t1 = time.perf_counter()
    return map1_ss, map2_ss, (t1 - t0)

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

@njit(parallel=True, fastmath=True, cache=True)
def forward_lut_bilinear_numba(pts_px_und, fwd_d_x, fwd_d_y):
    """
    pts_px_und: (N,2) float64
    fwd_d_x/y:  (H,W) float32 (map from undistorted pixel -> distorted pixel)
    returns:    (N,2) float64
    """
    h, w = fwd_d_x.shape
    out = np.empty((pts_px_und.shape[0], 2), dtype=np.float64)

    for i in prange(pts_px_und.shape[0]):
        x = pts_px_und[i, 0]
        y = pts_px_und[i, 1]

        # clamp for safe x1/y1 indexing
        if x < 0.0:
            x = 0.0
        elif x > w - 1.000001:
            x = w - 1.000001

        if y < 0.0:
            y = 0.0
        elif y > h - 1.000001:
            y = h - 1.000001

        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        x1 = x0 + 1
        y1 = y0 + 1
        if x1 >= w: x1 = w - 1
        if y1 >= h: y1 = h - 1

        wx = x - x0
        wy = y - y0

        f00x = float(fwd_d_x[y0, x0])
        f10x = float(fwd_d_x[y0, x1])
        f01x = float(fwd_d_x[y1, x0])
        f11x = float(fwd_d_x[y1, x1])

        f00y = float(fwd_d_y[y0, x0])
        f10y = float(fwd_d_y[y0, x1])
        f01y = float(fwd_d_y[y1, x0])
        f11y = float(fwd_d_y[y1, x1])

        dx = (1.0 - wx) * (1.0 - wy) * f00x + wx * (1.0 - wy) * f10x + (1.0 - wx) * wy * f01x + wx * wy * f11x
        dy = (1.0 - wx) * (1.0 - wy) * f00y + wx * (1.0 - wy) * f10y + (1.0 - wx) * wy * f01y + wx * wy * f11y

        out[i, 0] = dx
        out[i, 1] = dy

    return out

@njit(parallel=True, fastmath=True, cache=True)
def forward_lut_bilinear_numba_supersampled(pts_px_und, fwd_d_x_ss, fwd_d_y_ss, s):
    """
    pts_px_und: (N,2) float64 undistorted pixel coords
    fwd_d_x_ss/y_ss: (H*s, W*s) float32 forward maps
    s: int supersample factor
    returns: (N,2) float64 distorted pixel coords
    """
    h, w = fwd_d_x_ss.shape
    out = np.empty((pts_px_und.shape[0], 2), dtype=np.float64)

    for i in prange(pts_px_und.shape[0]):
        # scale into supersampled grid coordinates
        x = pts_px_und[i, 0] * s
        y = pts_px_und[i, 1] * s

        # clamp
        if x < 0.0:
            x = 0.0
        elif x > w - 1.000001:
            x = w - 1.000001

        if y < 0.0:
            y = 0.0
        elif y > h - 1.000001:
            y = h - 1.000001

        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        x1 = x0 + 1
        y1 = y0 + 1
        if x1 >= w: x1 = w - 1
        if y1 >= h: y1 = h - 1

        wx = x - x0
        wy = y - y0

        f00x = float(fwd_d_x_ss[y0, x0])
        f10x = float(fwd_d_x_ss[y0, x1])
        f01x = float(fwd_d_x_ss[y1, x0])
        f11x = float(fwd_d_x_ss[y1, x1])

        f00y = float(fwd_d_y_ss[y0, x0])
        f10y = float(fwd_d_y_ss[y0, x1])
        f01y = float(fwd_d_y_ss[y1, x0])
        f11y = float(fwd_d_y_ss[y1, x1])

        dx = (1.0 - wx) * (1.0 - wy) * f00x + wx * (1.0 - wy) * f10x + (1.0 - wx) * wy * f01x + wx * wy * f11x
        dy = (1.0 - wx) * (1.0 - wy) * f00y + wx * (1.0 - wy) * f10y + (1.0 - wx) * wy * f01y + wx * wy * f11y

        out[i, 0] = dx / s
        out[i, 1] = dy / s

    return out
# ---------- OpenCV implementations ----------
def make_opencv_undistort_runner(K, dist, pts_px_dist_f64):
    """
    Returns a closure that runs OpenCV undistortPoints with *no per-call reshapes/copies*.
    pts_px_dist_f64 must be float64 (N,2).
    """
    pts_cv = pts_px_dist_f64.reshape(-1, 1, 2)  # view, no copy

    def run():
        und = cv2.undistortPoints(pts_cv, K, dist, P=K)
        return und.reshape(-1, 2)

    return run

def opencv_undistort_batch(K, dist, pts_px_dist):
    # Convenience wrapper (NOT ideal for timing)
    pts_px_dist = np.asarray(pts_px_dist, dtype=np.float64)
    runner = make_opencv_undistort_runner(K, dist, pts_px_dist)
    return runner()

def make_opencv_distort_runner(K, dist, pts_px_und_f64):
    """
    Returns a closure that runs OpenCV projectPoints with *no per-call obj/rvec/tvec allocation*.
    pts_px_und_f64 must be float64 (N,2) and interpreted as UNDISTORTED pixels.
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    x = (pts_px_und_f64[:, 0] - cx) / fx
    y = (pts_px_und_f64[:, 1] - cy) / fy

    obj = np.empty((pts_px_und_f64.shape[0], 3), dtype=np.float64)
    obj[:, 0] = x
    obj[:, 1] = y
    obj[:, 2] = 1.0

    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)

    def run():
        img, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        return img.reshape(-1, 2)

    return run


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

def build_forward_point_lut_from_maps(map1, map2):
    """
    Build a forward LUT for POINT distortion:
        given an undistorted integer pixel (ux,uy), return distorted pixel (dx,dy).

    This is essentially just the undistortRectify maps themselves, but named explicitly
    as a "forward point LUT" for symmetry with the inverse LUT.
    """
    # map1/map2 are float32 (H,W) already: u -> d
    fwd_d_x = map1.astype(np.float32, copy=False)
    fwd_d_y = map2.astype(np.float32, copy=False)
    return fwd_d_x, fwd_d_y


def opencv_point_distort_via_forward_lut_nearest(pts_px_und, fwd_d_x, fwd_d_y):
    """
    Point-wise undistorted -> distorted using a precomputed forward LUT
    with nearest-neighbor lookup at integer pixel.
    """
    h, w = fwd_d_x.shape[:2]
    x = np.clip(np.rint(pts_px_und[:, 0]).astype(np.int32), 0, w - 1)
    y = np.clip(np.rint(pts_px_und[:, 1]).astype(np.int32), 0, h - 1)
    dx = fwd_d_x[y, x].astype(np.float64)
    dy = fwd_d_y[y, x].astype(np.float64)
    return np.column_stack([dx, dy])


def opencv_point_distort_via_forward_lut_bilinear(pts_px_und, fwd_d_x, fwd_d_y):
    """
    Point-wise undistorted -> distorted using a precomputed forward LUT
    with bilinear interpolation (subpixel lookup).
    """
    h, w = fwd_d_x.shape[:2]

    # Clamp to [0, w-1] / [0, h-1] but keep room for x1/y1 indexing.
    x = np.clip(pts_px_und[:, 0], 0.0, w - 1.000001).astype(np.float64)
    y = np.clip(pts_px_und[:, 1], 0.0, h - 1.000001).astype(np.float64)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    x1 = np.clip(x1, 0, w - 1)
    y1 = np.clip(y1, 0, h - 1)

    wx = x - x0
    wy = y - y0

    # Sample four corners for each map
    f00x = fwd_d_x[y0, x0].astype(np.float64)
    f10x = fwd_d_x[y0, x1].astype(np.float64)
    f01x = fwd_d_x[y1, x0].astype(np.float64)
    f11x = fwd_d_x[y1, x1].astype(np.float64)

    f00y = fwd_d_y[y0, x0].astype(np.float64)
    f10y = fwd_d_y[y0, x1].astype(np.float64)
    f01y = fwd_d_y[y1, x0].astype(np.float64)
    f11y = fwd_d_y[y1, x1].astype(np.float64)

    # Bilinear blend
    dx = (1.0 - wx) * (1.0 - wy) * f00x + wx * (1.0 - wy) * f10x + (1.0 - wx) * wy * f01x + wx * wy * f11x
    dy = (1.0 - wx) * (1.0 - wy) * f00y + wx * (1.0 - wy) * f10y + (1.0 - wx) * wy * f01y + wx * wy * f11y

    return np.column_stack([dx, dy])



# ---------- Metrics & timing ----------

def bench(fn, iters=50, warmup=5):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    return (t1 - t0) / iters

def in_bounds_mask(pts_px, width, height):
    """
    Boolean mask: True if point lies within image bounds [0,w-1] x [0,h-1].
    pts_px: (N,2)
    """
    return (
        (pts_px[:, 0] >= 0.0) & (pts_px[:, 0] <= (width  - 1)) &
        (pts_px[:, 1] >= 0.0) & (pts_px[:, 1] <= (height - 1))
    )


def rms_err(a, b):
    d = a - b
    return np.sqrt(np.mean(d[:, 0]**2 + d[:, 1]**2))


def max_err(a, b):
    d = a - b
    return np.sqrt(np.max(d[:, 0]**2 + d[:, 1]**2))


def test_cal(cal: Calibration, run_id: str):


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
    map1, map2 = build_undistort_maps(K, dist, cal.width,  cal.height)

    fwd_d_x, fwd_d_y = build_forward_point_lut_from_maps(map1, map2)
    W, H = cal.width, cal.height
    s = 2  # LUT supersample factor

    # --- Regular maps (for inverse LUT + baseline forward LUT at s=1) ---
    t0 = time.perf_counter()
    map1, map2 = build_undistort_maps(K, dist, W, H)
    t_build_maps = time.perf_counter() - t0

    fwd_d_x, fwd_d_y = build_forward_point_lut_from_maps(map1, map2)

    t0 = time.perf_counter()
    inv_u_x, inv_u_y = build_inverse_point_lut_from_maps(map1, map2)
    t_build_inv_lut = time.perf_counter() - t0

    # --- Supersampled maps (for forward_lut_bilinear_numba_supersampled) ---
    fwd_d_x_ss, fwd_d_y_ss, t_build_fwd_ss = build_forward_maps_supersampled_opencv(K, dist, W, H, s)

    # Make undistorted pixels in-frame. (These are "truth" targets.)
    pts_px_und_truth = np.column_stack([
        rng.uniform(0, cal.width, size=N_truth),
        rng.uniform(0, cal.height, size=N_truth)
    ]).astype(np.float64)

    # Distort them using OpenCV and our forward model.
    pts_px_dist_cv = opencv_distort_batch(K, dist, pts_px_und_truth)
    pts_px_dist_ours_scalar = ours_distort_batch_scalar(cal, pts_px_und_truth)
    pts_px_dist_ours = distort_points_px(cal, pts_px_und_truth)

    # Confirm forward distortion agreement (should be ~1e-13 px)
    print_title("FORWARD DISTORTION agreement ")
    row = [("Direct Comparison (scalar):",
            rms_err(pts_px_dist_ours_scalar, pts_px_dist_cv),
            max_err(pts_px_dist_ours_scalar, pts_px_dist_cv)),

           ("Direct Comparison (vec):",
            rms_err(pts_px_dist_ours, pts_px_dist_cv),
            max_err(pts_px_dist_ours, pts_px_dist_cv))]
    print_rmsSection('(our distort_points_px vs OpenCV projectPoints)', row)

    # Forward LUT distortion
    pts_px_dist_fwd_nn = opencv_point_distort_via_forward_lut_nearest(pts_px_und_truth, fwd_d_x, fwd_d_y)
    pts_px_dist_fwd_bl = opencv_point_distort_via_forward_lut_bilinear(pts_px_und_truth, fwd_d_x, fwd_d_y)
    pts_px_dist_fwd_bl_numba = forward_lut_bilinear_numba(pts_px_und_truth, fwd_d_x, fwd_d_y)
    pts_px_dist_fwd_bl_numba_ss = forward_lut_bilinear_numba_supersampled(pts_px_und_truth, fwd_d_x_ss, fwd_d_y_ss, s)

    print_title("FORWARD LUT distortion agreement vs OpenCV projectPoints")
    row = [('fwd LUT (nearest):',
            rms_err(pts_px_dist_fwd_nn, pts_px_dist_cv),
            max_err(pts_px_dist_fwd_nn, pts_px_dist_cv)),

           ('fwd LUT (bilinear):',
           rms_err(pts_px_dist_fwd_bl, pts_px_dist_cv),
           max_err(pts_px_dist_fwd_bl, pts_px_dist_cv)),

           ('fwd LUT (bilinear - numba):',
           rms_err(pts_px_dist_fwd_bl_numba, pts_px_dist_cv),
           max_err(pts_px_dist_fwd_bl_numba, pts_px_dist_cv)),

           ('fwd LUT (BL/numba/supersampled):',
           rms_err(pts_px_dist_fwd_bl_numba_ss, pts_px_dist_cv),
           max_err(pts_px_dist_fwd_bl_numba_ss, pts_px_dist_cv)),]
    print_rmsSection('(same undistorted input)',
                     row)

    # Undistort the SAME distorted set using each method and compare to truth
    # Use OpenCV-distorted points as the common input for fairness.
    pts_px_dist = pts_px_dist_cv

    ours_und_scalar = ours_undistort_batch_scalar(cal, pts_px_dist)
    ours_und_vec_prec = undistort_points_px(cal, pts_px_dist, mode="precise")
    ours_und_vec_cv = undistort_points_px(cal, pts_px_dist, mode="opencv")
    cv_und = opencv_undistort_batch(K, dist, pts_px_dist)
    cv_lut_und = opencv_point_undistort_via_inverse_lut(pts_px_dist, inv_u_x, inv_u_y)


    print_title('UNDISTORT accuracy vs TRUTH')
    row = [('scalar (2fp+newton):',
            rms_err(ours_und_scalar, pts_px_und_truth),
            max_err(ours_und_scalar, pts_px_und_truth)),

           ('vectorized (2fp+N):',
            rms_err(ours_und_vec_prec, pts_px_und_truth),
            max_err(ours_und_vec_prec, pts_px_und_truth)),

           ('vectorized (5fp):',
            rms_err(ours_und_vec_cv, pts_px_und_truth),
            max_err(ours_und_vec_cv, pts_px_und_truth)),

           ('cv   undistortPoints:',
            rms_err(cv_und, pts_px_und_truth),
            max_err(cv_und, pts_px_und_truth)),

           ('cv undistort (inverse LUT):',
            rms_err(cv_lut_und, pts_px_und_truth),
            max_err(cv_lut_und, pts_px_und_truth)),
           ]
    print_rmsSection('(starting from known undistorted -> distort -> undistort)', row)

    # Also report how far each method is from OpenCV undistortPoints
    row = [("  scalar (2fp+N):",
            rms_err(ours_und_scalar, cv_und),
            max_err(ours_und_scalar, cv_und)),

           ("  vectorized (2fp+N):",
            rms_err(ours_und_vec_prec, cv_und),
            max_err(ours_und_vec_prec, cv_und)),

           ("  vectorized (5fp):",
            rms_err(ours_und_vec_cv, cv_und),
            max_err(ours_und_vec_cv, cv_und)),

           (" cv undistort (inverse LUT):",
            rms_err(cv_lut_und, cv_und),
            max_err(cv_lut_und, cv_und))]
    print_rmsSection("UNDISTORT agreement vs OpenCV (same distorted input)", row)

    W = cal.width  # 864
    H = cal.height  # 864

    # Truth distorted pixels (analytic or OpenCV)
    d_truth = pts_px_dist_cv  # or pts_px_dist_ours

    # LUT result you want to analyze
    d_lut = pts_px_dist_fwd_bl_numba  # or nn / non-numba bilinear

    # In-bounds mask based on TRUTH mapping
    mask_in = in_bounds_mask(d_truth, W, H)
    mask_out = ~mask_in

    print_title("FORWARD LUT bilinear (Numba) error vs TRUTH")
    forward_luts = [
        ("FORWARD LUT nearest", pts_px_dist_fwd_nn),
        ("FORWARD LUT bilinear (numpy)", pts_px_dist_fwd_bl),
        ("FORWARD LUT bilinear (numba)", pts_px_dist_fwd_bl_numba),
        ("FORWARD LUT bilinear (numba ss)", pts_px_dist_fwd_bl_numba_ss),
    ]
    for name, pts_est in forward_luts:
        print_lut_error_block(
            title=f"{name} error vs TRUTH",
            pts_est=pts_est,
            pts_truth=d_truth,
            W=W,
            H=H,
        )

    # ------------------------------------------------------------------
    # Timing: use a smaller subset so Python scalar loop is tolerable
    # ------------------------------------------------------------------
    pts_px_dist_small = pts_px_dist[:20_000]
    pts_px_und_small_truth = pts_px_und_truth[:20_000]

    # Ensure OpenCV inputs are float64 once (so no per-call astype copies)
    pts_px_dist_small_f64 = np.asarray(pts_px_dist_small, dtype=np.float64)
    pts_px_und_small_f64 = np.asarray(pts_px_und_small_truth, dtype=np.float64)

    cv_und_runner = make_opencv_undistort_runner(K, dist, pts_px_dist_small_f64)
    cv_dist_runner = make_opencv_distort_runner(K, dist, pts_px_und_small_f64)
    # Ensure OpenCV inputs are float64 once (so no per-call astype copies)

    t_scalar_und = bench(lambda: ours_undistort_batch_scalar(cal, pts_px_dist_small), iters=10)
    t_vec_und_prec = bench(lambda: undistort_points_px(cal, pts_px_dist_small, mode="precise"), iters=800)
    t_vec_und_cv = bench(lambda: undistort_points_px(cal, pts_px_dist_small, mode="opencv"), iters=800)
    t_cv_und = bench(cv_und_runner, iters=800)
    t_cv_dist = bench(cv_dist_runner, iters=800)

    t_scalar_dist = bench(lambda: ours_distort_batch_scalar(cal, pts_px_und_small_truth), iters=10)
    t_vec_dist = bench(lambda: distort_points_px(cal, pts_px_und_small_truth), iters=800)
    t_cv_lut_dist_nn = bench(
        lambda: opencv_point_distort_via_forward_lut_nearest(pts_px_und_small_truth, fwd_d_x, fwd_d_y), iters=800)
    t_cv_lut_dist_bl = bench(
        lambda: opencv_point_distort_via_forward_lut_bilinear(pts_px_und_small_truth, fwd_d_x, fwd_d_y), iters=800)

    t_cv_lut_dist_bl_numba = bench(lambda: forward_lut_bilinear_numba_supersampled(
        pts_px_und_small_truth, fwd_d_x_ss, fwd_d_y_ss, s),
                                   iters=800)

    # --- Timing: point-wise inverse LUT (dist->und), map built once ---
    t_cv_lut_und = bench(lambda: opencv_point_undistort_via_inverse_lut(pts_px_dist_small, inv_u_x, inv_u_y),iters=800)

    N = pts_px_dist_small.shape[0]

    print("\nTIMING (avg seconds per call)   N=", N)

    row = [('ours undistort scalar:', t_scalar_und),
           ('ours undistort vec (2fp+N):', t_vec_und_prec),
           ('ours undistort vec (5fp):', t_vec_und_cv),
           ('cv   undistort:', t_cv_und),
           ('cv   undistort (inverse LUT):', t_cv_lut_und),
           ]
    print_floatSection('UNDISTORT', row)

    row = [('ours distort scalar:', t_scalar_dist),
           ('ours distort vec:', t_vec_dist),
           ('cv   distort:', t_cv_dist),
           ('cv   distort (fwd LUT nn):', t_cv_lut_dist_nn),
           ('cv   distort (fwd LUT bilinear):', t_cv_lut_dist_bl),
           ('cv   distort (fwd LUT bilinear numba):', t_cv_lut_dist_bl_numba),
           ]
    print_floatSection('DISTORT', row)


    row = [('cv   undistort maps build (one-time):',t_build_maps),
           (f'cv   fwd LUT maps build s={s} (one-time):',t_build_fwd_ss),
           ('cv   inverse LUT build (one-time):',t_build_inv_lut),
           ]
    print_floatSection('ONE TIME BUILD COSTS', row)

    print("\nTHROUGHPUT (points/sec)")

    row = [('ours undistort scalar:', N / t_scalar_und),
           ('ours undistort vec (2fp+N):', N / t_vec_und_prec),
           ('ours undistort vec (5fp):', N / t_vec_und_cv),
           ('cv  undistort:', N / t_cv_und),
           ('cv  undistort (inverse LUT):', N / t_cv_lut_und),
           ]
    print_intSection('UNDISTORT', row)

    row = [('ours distort scalar:', N / t_scalar_dist),
           ('ours distort vec:', N / t_vec_dist),
           ('cv   distort:', N / t_cv_dist),
           ('cv   distort (fwd LUT nn):', N / t_cv_lut_dist_nn),
           ('cv   distort (fwd LUT bilinear):', N / t_cv_lut_dist_bl),
           ('cv   distort (fwd LUT bilinear numba):', N / t_cv_lut_dist_bl_numba),
           ]
    print_intSection('DISTORT', row)
    print_break(rows=3)

    # ------------------------------------------------------------
    # Collect speed + precision (for plotting across ALL main() runs)
    # ------------------------------------------------------------
    results = []

    # UNDISTORT precision vs truth (all points; truth is in-frame by construction)
    und_methods = [
        ("ours undistort scalar", ours_und_scalar, t_scalar_und),
        ("ours undistort vec (2fp+N)", ours_und_vec_prec, t_vec_und_prec),
        ("ours undistort vec (5fp)", ours_und_vec_cv, t_vec_und_cv),
        ("cv undistortPoints", cv_und, t_cv_und),
        ("cv undistort (inverse LUT)", cv_lut_und, t_cv_lut_und),
    ]
    for name, est, t_s in und_methods:
        st = err_stats(est, pts_px_und_truth)  # no mask
        results.append(dict(
            run_id=run_id,
            task="UNDISTORT",
            method=name,
            time_s=float(t_s),
            **st
        ))

    # DISTORT precision vs truth (use in-bounds mask based on truth distorted pixel)
    # truth distorted pixels:
    d_truth = pts_px_dist_cv  # you already define this :contentReference[oaicite:4]{index=4}
    mask_in = in_bounds_mask(d_truth, W, H)

    # build the missing forward-LUT numba outputs on the FULL set (for precision),
    # matching your earlier NN/bilinear outputs:
    pts_px_dist_fwd_bl_numba = forward_lut_bilinear_numba(pts_px_und_truth, fwd_d_x, fwd_d_y)
    pts_px_dist_fwd_bl_numba_ss = forward_lut_bilinear_numba_supersampled(
        pts_px_und_truth, fwd_d_x_ss, fwd_d_y_ss, s
    )

    dist_methods = [
        ("ours distort scalar", pts_px_dist_ours_scalar, t_scalar_dist),
        ("ours distort vec", pts_px_dist_ours, t_vec_dist),
        ("cv projectPoints", pts_px_dist_cv, t_cv_dist),

        ("fwd LUT nearest", pts_px_dist_fwd_nn, t_cv_lut_dist_nn),
        ("fwd LUT bilinear", pts_px_dist_fwd_bl, t_cv_lut_dist_bl),
        ("fwd LUT bilinear (numba)", pts_px_dist_fwd_bl_numba, t_cv_lut_dist_bl),  # or bench separately
        ("fwd LUT bilinear (numba ss)", pts_px_dist_fwd_bl_numba_ss, t_cv_lut_dist_bl_numba),
    ]

    for name, est, t_s in dist_methods:
        st = err_stats(est, d_truth, mask=mask_in)
        results.append(dict(
            run_id=run_id,
            task="DISTORT",
            method=name,
            time_s=float(t_s),
            **st
        ))

    return results

def show_tradeoff_pairs_sweep(
    *,
    num_cals=150,
    width=864,
    height=864,
    # sweep controls
    strengths=("high","high"),
    profiles=("radial_only", "tangential_only", "mixed"),
    # evaluation sizes
    N_err=50_000,        # precision evaluation set
    N_time=1_000,       # timing evaluation set (keeps things fast)
    # bench controls
    iters=400,
    warmup=30,
    connect_every=10,
    seed=123,
):
    """
    Generates a paired speed-vs-precision plot comparing:
      - ours undistort vec (5fp):   undistort_points_px(..., mode="opencv")
      - ours undistort vec (2fp+N): undistort_points_px(..., mode="precise")

    across randomized calibrations. Points are paired per calibration and
    optionally connected every Nth calibration.
    """

    # Forces warm-up so first iteration isn't significantly slower unfairly
    prejit_undistort_kernels()

    paired_rows = []
    rng = np.random.default_rng(seed)

    # Pre-generate a single truth point set in-frame for repeatability across cals.
    # (You *can* regenerate per-cal if you want; keeping it fixed reduces variance.)
    pts_px_und_truth = np.column_stack([
        rng.uniform(0.0, float(width),  size=N_err),
        rng.uniform(0.0, float(height), size=N_err),
    ]).astype(np.float64)

    # Subset used for timing
    pts_px_und_time_truth = pts_px_und_truth[:N_time].copy()

    cal = Calibration()  # reuse object to reduce alloc noise

    cal_id = 0
    total = len(strengths) * len(profiles) * num_cals

    for strength in strengths:
        for profile in profiles:
            for _ in range(num_cals):
                # --- randomize calibration ---
                cal.randomize(
                    rng=rng,
                    width=width,
                    height=height,
                    keep_intrinsics=False,
                    profile=profile,
                    strength=strength,
                    ensure_invertible=True,
                )
                assert cal.validCal

                # distortion severity: max corner displacement in pixels
                severity, _ = cal.max_corner_distortion_px()

                # OpenCV mats
                K, dist = make_opencv_mats(cal)

                # --- build distorted inputs from truth (OpenCV is "ground truth" forward) ---
                pts_px_dist = opencv_distort_batch(K, dist, pts_px_und_truth)
                pts_px_dist_time = np.ascontiguousarray(pts_px_dist[:N_time], dtype=np.float64)

                # --- precision (full N_err set, not timed) ---
                und5 = undistort_points_px(cal, pts_px_dist, mode="opencv")   # 5FP
                undH = undistort_points_px(cal, pts_px_dist, mode="precise")  # 2FP+N

                st5 = err_stats(und5, pts_px_und_truth)  # truth is in-frame by construction
                stH = err_stats(undH, pts_px_und_truth)

                # optional extra diagnostics (helps interpret convergence/stability)
                oob5 = float(np.mean(~in_bounds_mask(und5, width, height)))
                oobH = float(np.mean(~in_bounds_mask(undH, width, height)))

                # --- timing (smaller N_time set) ---
                run5 = (lambda: undistort_points_px(cal, pts_px_dist_time, mode="opencv"))
                runH = (lambda: undistort_points_px(cal, pts_px_dist_time, mode="precise"))

                t5 = bench(run5, iters=iters, warmup=warmup)
                tH = bench(runH, iters=iters, warmup=warmup)

                paired_rows.append({
                    "cal_id": int(cal_id),
                    "severity": float(severity),
                    "run_id": f"{strength}|{profile}",
                    "m5": {
                        "time_s": float(t5),
                        "rms": float(st5["rms"]),
                        "method": "ours undistort vec (5fp)",
                        "oob": oob5,
                    },
                    "mH": {
                        "time_s": float(tH),
                        "rms": float(stH["rms"]),
                        "method": "ours undistort vec (2fp+N)",
                        "oob": oobH,
                    },
                })

                cal_id += 1
                if cal_id % 10 == 0:
                    print(f"[pairs sweep] {cal_id}/{total} done...")

    plot_tradeoff_pairs(
        paired_rows,
        connect_every=connect_every,
        sort_by="severity",
        title="UNDISTORT: 5FP vs 2FP+N (paired per calibration; sorted by corner distortion)",
    )

    # If you also want the paired data for later analysis / saving:
    return paired_rows


def main():
    show_tradeoff_pairs_sweep(
        num_cals=20,          # per (strength,profile) bucket; start small
        iters=10,
        warmup=25,
        connect_every=1,
    )

    cv2.setUseOptimized(True)
    cv2.setNumThreads(os.cpu_count())  # or a fixed N for fairness
    cv2.ocl.setUseOpenCL(True)  # avoid surprise GPU/OpenCL paths

    print("BUILD INFORMATION:")
    print(cv2.getBuildInformation())

    cal = default_864_cam()
    np.set_printoptions(suppress=True)
    # sweep

    all_results = []

    for strength in ["low", "medium", "high"]:
        for profile in ["radial_only", "tangential_only", "mixed"]:
            cal.randomize(keep_intrinsics=False, strength=strength, profile=profile)
            assert cal.validCal

            print_break()
            print_title("Testing Calibration:")
            print_title(f"Strength: {strength}, Profile: {profile}")
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
            run_id = f'{strength}|{profile}'
            all_results.extend(test_cal(cal, run_id))
    plot_tradeoff(all_results)


def print_lut_error_block(
    title: str,
    pts_est: np.ndarray,
    pts_truth: np.ndarray,
    W: int,
    H: int,
):
    err = pts_est - pts_truth
    err_norm = np.linalg.norm(err, axis=1)

    mask_in = in_bounds_mask(pts_truth, W, H)
    mask_out = ~mask_in

    print_title(title)

    print("  in-bounds:")
    if np.any(err_norm[mask_in]):
        print(f"    RMS px err: {np.sqrt(np.mean(err_norm[mask_in] ** 2))}")
        print(f"    Max px err: {np.max(err_norm[mask_in])}")
    else:
        print(f'     No points remain in bounds.')

    print("  out-of-bounds:")
    if np.any(err_norm[mask_out]):
        print(f"    RMS px err: {np.sqrt(np.mean(err_norm[mask_out] ** 2))}")
        print(f"    Max px err: {np.max(err_norm[mask_out])}")
    else:
        print(f'     No points exceeded boundary.')

    print(f"  fraction out-of-bounds: {np.mean(mask_out)}")

    print(f"  p99:   {np.percentile(err_norm[mask_in], 99)}")
    print(f"  p99.9: {np.percentile(err_norm[mask_in], 99.9)}")

def print_intSection(title: str, row: list, size: int = 60):
    print_title(title, size)
    print_intRow(row)
    print_break()

def print_floatSection(title: str, row: list, size: int = 60):
    print_title(title, size)
    print_floatRow(row)
    print_break()

def print_rmsSection(title: str, row: list, size: int = 60):
    print_title(title, size)
    print_rmsRow(row)
    print_break()

def print_rmsRow(row: list, size: int = 60):
    rms_label = 'RMS px err: '
    max_label = 'Max px err: '
    for name, rms, max in row:
        rms_txt = f'{rms:.6e}'
        max_txt = f'{max:.6e}'
        empty_space = size - len(name) - len(rms_label) - len(rms_txt) - 1
        print(f' {name}' + empty_space * ' ' + rms_label + rms_txt)
        empty_space = size - len(max_label) - len(max_txt)
        print(empty_space * ' ' + max_label + max_txt)

def print_floatRow(row: list, size: int = 60):
    for text, num in row:
        num_txt = f'{num:>10,.7f}'
        empty_space = size - len(text) - len(num_txt) - 1
        print(f' {text}' + ' ' * empty_space + num_txt)

def print_intRow(row: list, size: int = 60):
    # text_width = max(len(text) for text, _ in row)
    numb_width = max(len(f'{num:,.0f}') for _, num in row)
    for text, num in row:
        empty_space = size - len(text) - numb_width - 1
        print(f' {text}' + ' ' * empty_space + f'{num:>{numb_width},.0f}')

def print_title(title: str, size: int = 60):
    title_len = len(title)
    corr_odd = 0 if (size - title_len) % 2 == 0 else 1
    size_sides = int((size - title_len)/2) - 1
    print(size_sides * '=' + ' ' + title + ' ' + (corr_odd + size_sides) * '=')

def print_break(size: int = 60, rows: int = 1):
    for _ in range(max(rows,1)):
        print(size * '=')
    print()

if __name__ == "__main__":
    main()
