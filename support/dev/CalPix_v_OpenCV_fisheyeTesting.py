# CalPix_v_OpenCV_fisheyeTesting.py

import time
import numpy as np
import cv2
from numpy.typing import NDArray
import os

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

# ============================================================
# Fisheye sweep helpers
# ============================================================

def fisheye_corner_severity_px(cal: Calibration) -> tuple[float, np.ndarray]:
    """
    Return maximum forward fisheye distortion displacement at image corners.

    This mirrors the Brown-Conrady max_corner_distortion_px metric, but uses
    the generic distort_points_px path so it works for fisheye calibrations.
    """
    w = int(cal.width)
    h = int(cal.height)
    corners_px_und = np.array(
        [[0.0, 0.0],
         [float(w - 1), 0.0],
         [float(w - 1), float(h - 1)],
         [0.0, float(h - 1)]],
        dtype=np.float64,
    )
    corners_px_dist = distort_points_px(cal, corners_px_und)
    d = corners_px_dist - corners_px_und
    mags = np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1])
    return float(np.max(mags)), mags


def fisheye_min_dtheta_dtheta(cal: Calibration, samples: int = 256) -> float:
    """
    Minimum derivative of theta_d(theta) over the image FOV.

    theta_d = theta * (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8)

    dtheta_d/dtheta =
        1 + 3 k1 theta^2 + 5 k2 theta^4 + 7 k3 theta^6 + 9 k4 theta^8

    A positive derivative over the FOV is a practical local monotonicity check.
    """
    xs = np.array([0.0, cal.width - 1.0], dtype=np.float64)
    ys = np.array([0.0, cal.height - 1.0], dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)

    xn = (X.ravel() - cal.cx) / cal.fx
    yn = (Y.ravel() - cal.cy) / cal.fy
    rmax = float(np.sqrt((xn * xn + yn * yn).max()))
    theta_max = float(np.arctan(rmax))

    theta = np.linspace(0.0, theta_max, samples, dtype=np.float64)
    t2 = theta * theta
    t4 = t2 * t2
    t6 = t4 * t2
    t8 = t4 * t4

    deriv = (
        1.0
        + 3.0 * cal.k1 * t2
        + 5.0 * cal.k2 * t4
        + 7.0 * cal.k3 * t6
        + 9.0 * cal.k4 * t8
    )
    return float(np.min(deriv))


def randomize_fisheye_cal(
    rng: np.random.Generator,
    *,
    width: int = 1920,
    height: int = 1080,
    strength: str = "medium",
    profile: str = "mixed",
    max_attempts: int = 500,
) -> Calibration:
    """
    Random fisheye calibration generator for benchmark sweeps.

    Profiles:
      barrel_like      alternating signs beginning with negative k1
      pincushion_like  alternating signs beginning with positive k1
      mixed            random signs

    Strength controls both coefficient magnitude and focal length range.
    Smaller focal length means a wider FOV and a harder fisheye inversion.
    """
    strength = str(strength).lower()
    profile = str(profile).lower()

    coeff_ranges = {
        "low":    (0.05, 0.015, 0.004, 0.0015),
        "medium": (0.16, 0.055, 0.018, 0.0060),
        "high":   (0.36, 0.130, 0.045, 0.0140),
    }

    fx_ranges = {
        "low":    (900.0, 1600.0),
        "medium": (500.0, 1000.0),
        "high":   (250.0, 700.0),
    }

    if strength not in coeff_ranges:
        raise ValueError(f"Unknown strength: {strength}")
    if profile not in ("barrel_like", "pincushion_like", "mixed"):
        raise ValueError(f"Unknown profile: {profile}")

    a1, a2, a3, a4 = coeff_ranges[strength]
    fx_lo, fx_hi = fx_ranges[strength]

    for _ in range(max_attempts):
        cal = default_fisheye_cam()
        cal.fisheye = True

        cal.width = int(width)
        cal.height = int(height)

        cal.fx = float(rng.uniform(fx_lo, fx_hi))
        cal.fy = float(cal.fx * rng.uniform(0.97, 1.03))
        cal.cx = float((width - 1) * 0.5 + rng.uniform(-5.0, 5.0))
        cal.cy = float((height - 1) * 0.5 + rng.uniform(-5.0, 5.0))

        if profile == "barrel_like":
            signs = np.array([-1.0, 1.0, -1.0, 1.0])
        elif profile == "pincushion_like":
            signs = np.array([1.0, -1.0, 1.0, -1.0])
        else:
            signs = rng.choice(np.array([-1.0, 1.0]), size=4)

        mags = np.array([
            rng.uniform(0.25 * a1, a1),
            rng.uniform(0.10 * a2, a2),
            rng.uniform(0.05 * a3, a3),
            rng.uniform(0.02 * a4, a4),
        ])

        cal.k1, cal.k2, cal.k3, cal.k4 = (signs * mags).astype(float)

        # Keep accessories valid in case default_fisheye_cam changes later.
        cal.calTime = 0.0
        cal.numCBUsed = 0
        cal.rmsError = 0.0
        cal.hfov = float(2.0 * np.degrees(np.arctan((cal.width * 0.5) / cal.fx)))

        if not cal.validCal:
            continue

        # Practical monotonicity guard over the image FOV.
        if fisheye_min_dtheta_dtheta(cal) <= 0.05:
            continue

        # Reject pathological cases where almost all distorted points leave the image.
        test_pts = np.array([
            [0.0, 0.0],
            [width - 1.0, 0.0],
            [width - 1.0, height - 1.0],
            [0.0, height - 1.0],
            [(width - 1.0) * 0.5, (height - 1.0) * 0.5],
        ], dtype=np.float64)
        K, D = make_opencv_fisheye_mats(cal)
        test_dist = opencv_fisheye_distort_batch(K, D, test_pts)
        if np.all(np.isfinite(test_dist)):
            return cal

    raise RuntimeError("Failed to sample a stable fisheye calibration.")


def prejit_fisheye_kernels():
    """
    Compile Numba fisheye paths before benchmark timing.
    """
    cal = default_fisheye_cam()
    cal.fx = cal.fy = 500.0
    cal.cx = (cal.width - 1) * 0.5
    cal.cy = (cal.height - 1) * 0.5
    cal.k1 = -0.18
    cal.k2 = 0.04
    cal.k3 = -0.01
    cal.k4 = 0.002

    K, D = make_opencv_fisheye_mats(cal)
    pts_und = np.array(
        [[100.0, 200.0],
         [400.0, 500.0],
         [800.0, 100.0]],
        dtype=np.float64,
    )
    pts_dist = opencv_fisheye_distort_batch(K, D, pts_und)

    _ = distort_points_px(cal, pts_und)
    _ = undistort_points_px(cal, pts_dist, mode="newton")
    _ = undistort_points_px(cal, pts_dist, mode="precise")


def ratio_with_floor(num: float, den: float, floor: float = 1e-13) -> float:
    """
    Ratio helper for precision values near numerical floor.
    """
    return max(float(num), floor) / max(float(den), floor)


def format_ratio(x: float) -> str:
    if not np.isfinite(x):
        return "--"
    if 0.995 <= x <= 1.005:
        return r"\(\approx 1.0\times\)"
    return rf"\({x:.2f}\times\)"


def profile_label(profile: str) -> str:
    return str(profile).replace("_", "-").title()


# ============================================================
# Main fisheye test harness
# ============================================================

def run_fisheye_point_tests(
    *,
    cal: Calibration | None = None,
    run_id: str = "single",
    strength: str = "custom",
    profile: str = "custom",
    n_pts: int = 100_000,
    n_time: int = 20_000,
    seed: int = 0,
    iters: int = 100,
    warmup: int = 10,
    undistort_modes: tuple[str, ...] = ("newton", "precise"),
):
    """
    Compare OpenCV fisheye undistortion against our fisheye modes.

    This returns both:
      - accuracy vs known undistorted truth
      - agreement vs OpenCV fisheye.undistortPoints

    Precision statistics are computed only for distorted points that remain
    in image bounds, which represents valid distorted measurements.
    """
    if cal is None:
        cal = default_fisheye_cam()
        cal.fx = cal.fy = 250.0
        cal.cx = 960.0
        cal.cy = 540.0
        cal.k1 = -0.35
        cal.k2 = 0.11
        cal.k3 = -0.04
        cal.k4 = 0.008

    assert cal.validCal and cal.fisheye

    rng = np.random.default_rng(seed)

    pts_und = np.column_stack([
        rng.uniform(0.0, cal.width - 1.0, n_pts),
        rng.uniform(0.0, cal.height - 1.0, n_pts),
    ]).astype(np.float64)

    K, D = make_opencv_fisheye_mats(cal)

    # OpenCV forward fisheye is the common source of distorted measurements.
    pts_dist = opencv_fisheye_distort_batch(K, D, pts_und)
    mask_dist = in_bounds_mask(pts_dist, cal.width, cal.height)
    valid_fraction = float(np.mean(mask_dist))

    if not np.any(mask_dist):
        raise RuntimeError(f"No distorted points remained in bounds for run {run_id}.")

    pts_dist_time = np.ascontiguousarray(pts_dist[:min(n_time, n_pts)], dtype=np.float64)

    cv_und = opencv_fisheye_undistort_batch(K, D, pts_dist)
    cv_truth_stats = err_stats(cv_und, pts_und, mask=mask_dist)
    t_cv = bench(
        lambda: opencv_fisheye_undistort_batch(K, D, pts_dist_time),
        iters=iters,
        warmup=warmup,
    )

    und_results = {
        "opencv": {
            "method": "cv2.fisheye.undistortPoints",
            "time_s": float(t_cv),
            "throughput": float(pts_dist_time.shape[0] / t_cv),
            "truth": cv_truth_stats,
            "agreement_cv": dict(rms=0.0, max=0.0, p99=0.0),
        }
    }

    for mode in undistort_modes:
        ours_und = undistort_points_px(cal, pts_dist, mode=mode)

        truth_stats = err_stats(ours_und, pts_und, mask=mask_dist)
        agreement_stats = err_stats(ours_und, cv_und, mask=mask_dist)

        t_ours = bench(
            lambda m=mode: undistort_points_px(cal, pts_dist_time, mode=m),
            iters=iters,
            warmup=warmup,
        )

        und_results[str(mode)] = {
            "method": f"ours fisheye {mode}",
            "time_s": float(t_ours),
            "throughput": float(pts_dist_time.shape[0] / t_ours),
            "truth": truth_stats,
            "agreement_cv": agreement_stats,
        }

    severity, _ = fisheye_corner_severity_px(cal)

    return {
        "run_id": run_id,
        "strength": strength,
        "profile": profile,
        "severity_px": float(severity),
        "min_dtheta_dtheta": float(fisheye_min_dtheta_dtheta(cal)),
        "valid_fraction": valid_fraction,
        "cal": cal,
        "UNDISTORT": und_results,
    }


def make_fisheye_pair_row(result: dict) -> dict:
    """
    Convert one detailed fisheye result into paired-row form.

    Baseline: mode="newton"
    Hybrid:   mode="precise"
    """
    base = result["UNDISTORT"]["newton"]
    hyb = result["UNDISTORT"]["precise"]

    return {
        "run_id": result["run_id"],
        "strength": result["strength"],
        "profile": result["profile"],
        "severity": result["severity_px"],
        "valid_fraction": result["valid_fraction"],
        "min_dtheta_dtheta": result["min_dtheta_dtheta"],
        "newton": {
            "time_s": float(base["time_s"]),
            "rms": float(base["truth"]["rms"]),
            "max": float(base["truth"]["max"]),
            "p99": float(base["truth"]["p99"]),
        },
        "precise": {
            "time_s": float(hyb["time_s"]),
            "rms": float(hyb["truth"]["rms"]),
            "max": float(hyb["truth"]["max"]),
            "p99": float(hyb["truth"]["p99"]),
        },
        "opencv": {
            "time_s": float(result["UNDISTORT"]["opencv"]["time_s"]),
            "rms": float(result["UNDISTORT"]["opencv"]["truth"]["rms"]),
            "max": float(result["UNDISTORT"]["opencv"]["truth"]["max"]),
            "p99": float(result["UNDISTORT"]["opencv"]["truth"]["p99"]),
        },
    }


def plot_fisheye_tradeoff_pairs(
    paired_rows: list[dict],
    *,
    out_pdf: str | None = "FisheyeUndistortSpeedPrecision.pdf",
    noise_floor: float = 1e-13,
    title: str = "FISHEYE UNDISTORT: Newton vs FP+Newton",
):
    """
    Paired speed/precision plot for fisheye undistortion.
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    rows = list(paired_rows)
    if not rows:
        return

    x_newton = np.asarray([r["newton"]["time_s"] for r in rows], dtype=float)
    y_newton = np.asarray([max(r["newton"]["rms"], noise_floor) for r in rows], dtype=float)

    x_precise = np.asarray([r["precise"]["time_s"] for r in rows], dtype=float)
    y_precise = np.asarray([max(r["precise"]["rms"], noise_floor) for r in rows], dtype=float)

    sev = np.asarray([r["severity"] for r in rows], dtype=float)

    faster = x_precise < x_newton
    precise_or_eq = (
        (y_precise <= y_newton)
        | ((y_precise <= noise_floor) & (y_newton <= noise_floor))
    )

    n = len(rows)
    pct_faster = 100.0 * np.sum(faster) / n
    pct_precise = 100.0 * np.sum(precise_or_eq) / n
    pct_both = 100.0 * np.sum(faster & precise_or_eq) / n

    vmin = float(np.percentile(sev, 2.0))
    vmax = float(np.percentile(sev, 98.0))
    if vmax <= vmin:
        vmax = vmin + 1.0

    cmap = mpl.colormaps.get_cmap("turbo")
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(14, 6))
    ax = plt.gca()

    sc0 = ax.scatter(
        x_newton,
        y_newton,
        c=sev,
        cmap=cmap,
        norm=norm,
        marker="o",
        s=60,
        alpha=0.75,
        label="ours fisheye newton",
    )

    ax.scatter(
        x_precise,
        y_precise,
        c=sev,
        cmap=cmap,
        norm=norm,
        marker="s",
        s=60,
        alpha=0.75,
        label="ours fisheye precise",
    )

    for r in rows:
        col = cmap(norm(r["severity"]))
        ax.plot(
            [r["newton"]["time_s"], r["precise"]["time_s"]],
            [max(r["newton"]["rms"], noise_floor), max(r["precise"]["rms"], noise_floor)],
            color=col,
            alpha=0.35,
            linewidth=1.0,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Avg time per call (seconds)")
    ax.set_ylabel("RMS px error vs truth")
    ax.set_title(title)
    ax.grid(True, which="both")

    ax.legend(loc="upper right")

    cbar = plt.colorbar(sc0, ax=ax, pad=0.02)
    cbar.set_label("Distortion severity (max corner displacement, px)")

    stats_txt = (
        f"precise ≥ accuracy: {pct_precise:5.1f}%\n"
        f"precise faster:     {pct_faster:5.1f}%\n"
        f"both:              {pct_both:5.1f}%"
    )
    ax.text(
        0.985,
        0.86,
        stats_txt,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="0.7"),
    )

    plt.tight_layout()

    if out_pdf is not None:
        plt.savefig(out_pdf, bbox_inches="tight")

    plt.show()


def print_fisheye_pair_summary(paired_rows: list[dict], *, noise_floor: float = 1e-13):
    """
    Console summary of each paired fisheye run.
    """
    print("\nFISHEYE UNDISTORT PAIRED SUMMARY")
    print("=" * 90)
    print(f"{'Run':30s} {'PrecRatio':>12s} {'TimeRatio':>12s} {'Valid':>8s} {'Severity(px)':>14s}")
    print("-" * 90)

    for r in paired_rows:
        precision_ratio = ratio_with_floor(r["newton"]["rms"], r["precise"]["rms"], noise_floor)
        runtime_ratio = r["newton"]["time_s"] / r["precise"]["time_s"]
        label = f"{r['strength']}|{r['profile']}"
        print(
            f"{label:30s} "
            f"{precision_ratio:12.3g} "
            f"{runtime_ratio:12.3g} "
            f"{100.0 * r['valid_fraction']:7.1f}% "
            f"{r['severity']:14.3f}"
        )


def print_latex_fisheye_pair_table(paired_rows: list[dict], *, noise_floor: float = 1e-13):
    """
    Print a page-friendly LaTeX table summarized by strength/profile.

    If multiple random calibrations exist for each strength/profile bucket,
    the displayed ratios are medians and the count columns show paired dominance.
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in paired_rows:
        key = (r["strength"], r["profile"])
        groups.setdefault(key, []).append(r)

    strength_order = {"low": 0, "medium": 1, "high": 2}
    profile_order = {"barrel_like": 0, "pincushion_like": 1, "mixed": 2}

    def sort_key(item):
        strength, profile = item[0]
        return (strength_order.get(strength, 99), profile_order.get(profile, 99), strength, profile)

    print()
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Paired comparison of fisheye Newton and fixed-point-initialized Newton undistortion performance. Precision ratio is \(e_{\mathrm{RMS},N}/e_{\mathrm{RMS},FP+N}\), so values greater than one indicate lower RMS error for the fixed-point-initialized Newton method. Runtime ratio is \(t_N/t_{FP+N}\), so values greater than one indicate faster runtime for the fixed-point-initialized Newton method.}")
    print(r"\label{tab:app_fisheye_undistort_paired_comparison}")
    print(r"\begin{tabular}{llcccc}")
    print(r"\toprule")
    print(r"Strength & Profile & Precision Ratio & Runtime Ratio & Precise & Faster \\")
    print(r"\midrule")

    total_prec = 0
    total_fast = 0
    total_n = 0

    for (strength, profile), rows in sorted(groups.items(), key=sort_key):
        prec_ratios = []
        time_ratios = []
        n_prec = 0
        n_fast = 0

        for r in rows:
            pr = ratio_with_floor(r["newton"]["rms"], r["precise"]["rms"], noise_floor)
            tr = r["newton"]["time_s"] / r["precise"]["time_s"]
            prec_ratios.append(pr)
            time_ratios.append(tr)

            precise_or_eq = (
                (r["precise"]["rms"] <= r["newton"]["rms"])
                or ((r["precise"]["rms"] <= noise_floor) and (r["newton"]["rms"] <= noise_floor))
            )
            faster = r["precise"]["time_s"] < r["newton"]["time_s"]

            n_prec += int(precise_or_eq)
            n_fast += int(faster)

        n = len(rows)
        total_prec += n_prec
        total_fast += n_fast
        total_n += n

        print(
            rf"{strength.title()} & {profile_label(profile)} & "
            rf"{format_ratio(float(np.median(prec_ratios)))} & "
            rf"{format_ratio(float(np.median(time_ratios)))} & "
            rf"{n_prec}/{n} & {n_fast}/{n} \\"
        )

    print(r"\midrule")
    print(rf"\multicolumn{{2}}{{l}}{{Summary}} & -- & -- & {total_prec}/{total_n} & {total_fast}/{total_n} \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()


def run_fisheye_sweep(
    *,
    num_cals: int = 5,
    strengths: tuple[str, ...] = ("low", "medium", "high"),
    profiles: tuple[str, ...] = ("barrel_like", "pincushion_like", "mixed"),
    width: int = 1920,
    height: int = 1080,
    n_pts: int = 100_000,
    n_time: int = 20_000,
    iters: int = 100,
    warmup: int = 10,
    seed: int = 123,
    out_pdf: str | None = "FisheyeUndistortSpeedPrecision.pdf",
):
    """
    Run fisheye paired tests over strength/profile buckets.

    Returns paired rows suitable for plotting or table generation.
    """
    prejit_fisheye_kernels()

    rng = np.random.default_rng(seed)
    paired_rows = []

    total = len(strengths) * len(profiles) * num_cals
    k = 0

    for strength in strengths:
        for profile in profiles:
            for j in range(num_cals):
                cal = randomize_fisheye_cal(
                    rng,
                    width=width,
                    height=height,
                    strength=strength,
                    profile=profile,
                )

                run_id = f"{strength}|{profile}|{j:03d}"

                result = run_fisheye_point_tests(
                    cal=cal,
                    run_id=run_id,
                    strength=strength,
                    profile=profile,
                    n_pts=n_pts,
                    n_time=n_time,
                    seed=seed + k,
                    iters=iters,
                    warmup=warmup,
                )

                paired_rows.append(make_fisheye_pair_row(result))

                k += 1
                if k % 5 == 0 or k == total:
                    print(f"[fisheye sweep] {k}/{total} done...")

    print_fisheye_pair_summary(paired_rows)
    print_latex_fisheye_pair_table(paired_rows)
    plot_fisheye_tradeoff_pairs(paired_rows, out_pdf=out_pdf)

    return paired_rows


# ============================================================
# CLI hook
# ============================================================

if __name__ == "__main__":
    cv2.setUseOptimized(True)
    cv2.setNumThreads(os.cpu_count() or 1)
    cv2.ocl.setUseOpenCL(False)

    rows = run_fisheye_sweep(
        num_cals=5,       # increase for the final dissertation plot
        n_pts=100_000,
        n_time=20_000,
        iters=100,
        warmup=10,
        out_pdf="FisheyeUndistortSpeedPrecision.pdf",
    )