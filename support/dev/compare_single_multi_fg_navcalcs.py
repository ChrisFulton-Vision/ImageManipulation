#!/usr/bin/env python3
"""
Generate comparison plots and summary tables for single-feature / multi-feature
SE(3) fusion navcalcs CSV files.

Expected input:
    A navcalcs CSV that contains:
      - single-feature range columns, e.g. single_feature_estimated_range
      - raw multi-feature pose columns, e.g. qnp_tvec_x/y/z and qnp_valid
      - fused SFMF factor-graph columns, e.g. qnp_sfmf_fg_tvec_x/y/z and
        qnp_sfmf_fg_valid

The script creates:
  * range overview plots across all methods
  * per-method raw vs fused vs single-feature range plots
  * per-method range-difference plots
  * per-method x/y/z raw-vs-fused plots
  * per-method stacked x/y/z dissertation plots
  * optional time/range-based truncation for close-range-only replays
  * per-method fused sigma plots
  * per-method factor-validity strip plots
  * summary CSV / Markdown / text files
  * pairwise range-agreement tables between all available series

By default the script processes one CSV, but multiple CSVs may be supplied.
Plots are written as both PNG and vector PDF.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Method definitions
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    raw_valid_col: str
    raw_t_cols: tuple[str, str, str]
    raw_rmse_col: str | None
    fg_valid_col: str
    fg_t_cols: tuple[str, str, str]
    fg_sigma_cols: tuple[str, str, str]
    fg_factor_single_valid_col: str
    fg_factor_multi_valid_col: str
    fg_reject_reason_col: str


METHODS: tuple[MethodSpec, ...] = (
    MethodSpec(
        key="pnp",
        label="OpenCV SolvePnP",
        raw_valid_col="pnp_valid",
        raw_t_cols=("pnp_tvec_x", "pnp_tvec_y", "pnp_tvec_z"),
        raw_rmse_col="pnp_rmse_px",
        fg_valid_col="pnp_sfmf_fg_valid",
        fg_t_cols=("pnp_sfmf_fg_tvec_x", "pnp_sfmf_fg_tvec_y", "pnp_sfmf_fg_tvec_z"),
        fg_sigma_cols=("pnp_sfmf_fg_sigma_x", "pnp_sfmf_fg_sigma_y", "pnp_sfmf_fg_sigma_z"),
        fg_factor_single_valid_col="pnp_sfmf_fg_single_factor_valid",
        fg_factor_multi_valid_col="pnp_sfmf_fg_multi_factor_valid",
        fg_reject_reason_col="pnp_sfmf_fg_reject_reason",
    ),
    MethodSpec(
        key="qnp",
        label="SolveQnP",
        raw_valid_col="qnp_valid",
        raw_t_cols=("qnp_tvec_x", "qnp_tvec_y", "qnp_tvec_z"),
        raw_rmse_col="qnp_rmse_px",
        fg_valid_col="qnp_sfmf_fg_valid",
        fg_t_cols=("qnp_sfmf_fg_tvec_x", "qnp_sfmf_fg_tvec_y", "qnp_sfmf_fg_tvec_z"),
        fg_sigma_cols=("qnp_sfmf_fg_sigma_x", "qnp_sfmf_fg_sigma_y", "qnp_sfmf_fg_sigma_z"),
        fg_factor_single_valid_col="qnp_sfmf_fg_single_factor_valid",
        fg_factor_multi_valid_col="qnp_sfmf_fg_multi_factor_valid",
        fg_reject_reason_col="qnp_sfmf_fg_reject_reason",
    ),
    MethodSpec(
        key="wqnp_yolo",
        label="wQnP, YOLO centers + KF covariance",
        raw_valid_col="wqnp_yolo_valid",
        raw_t_cols=("wqnp_yolo_tvec_x", "wqnp_yolo_tvec_y", "wqnp_yolo_tvec_z"),
        raw_rmse_col="wqnp_yolo_rmse_px",
        fg_valid_col="wqnp_yolo_sfmf_fg_valid",
        fg_t_cols=("wqnp_yolo_sfmf_fg_tvec_x", "wqnp_yolo_sfmf_fg_tvec_y", "wqnp_yolo_sfmf_fg_tvec_z"),
        fg_sigma_cols=("wqnp_yolo_sfmf_fg_sigma_x", "wqnp_yolo_sfmf_fg_sigma_y", "wqnp_yolo_sfmf_fg_sigma_z"),
        fg_factor_single_valid_col="wqnp_yolo_sfmf_fg_single_factor_valid",
        fg_factor_multi_valid_col="wqnp_yolo_sfmf_fg_multi_factor_valid",
        fg_reject_reason_col="wqnp_yolo_sfmf_fg_reject_reason",
    ),
    MethodSpec(
        key="wqnp_kfest",
        label="wQnP, KF centers + KF covariance",
        raw_valid_col="wqnp_kfest_valid",
        raw_t_cols=("wqnp_kfest_tvec_x", "wqnp_kfest_tvec_y", "wqnp_kfest_tvec_z"),
        raw_rmse_col="wqnp_kfest_rmse_px",
        fg_valid_col="wqnp_kfest_sfmf_fg_valid",
        fg_t_cols=("wqnp_kfest_sfmf_fg_tvec_x", "wqnp_kfest_sfmf_fg_tvec_y", "wqnp_kfest_sfmf_fg_tvec_z"),
        fg_sigma_cols=("wqnp_kfest_sfmf_fg_sigma_x", "wqnp_kfest_sfmf_fg_sigma_y", "wqnp_kfest_sfmf_fg_sigma_z"),
        fg_factor_single_valid_col="wqnp_kfest_sfmf_fg_single_factor_valid",
        fg_factor_multi_valid_col="wqnp_kfest_sfmf_fg_multi_factor_valid",
        fg_reject_reason_col="wqnp_kfest_sfmf_fg_reject_reason",
    ),
)

SINGLE_RANGE_COL = "single_feature_estimated_range"
SINGLE_T_COLS = (
    "single_feature_estimated_tvec_x",
    "single_feature_estimated_tvec_y",
    "single_feature_estimated_tvec_z",
)
TIME_CANDIDATES = ("_plot_time", "image_time", "frame_idx")
LITERATURE_STYLE = {
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.titleweight": "semibold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.6,
    "grid.linestyle": "--",
    "legend.frameon": False,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
BASE_METHOD_COLORS: dict[str, str] = {
    "pnp": "#1f77b4",
    "qnp": "#2ca02c",
    "wqnp_yolo": "#ff7f0e",
    "wqnp_kfest": "#d62728",
}
FAMILY_SHADE_OFFSETS: dict[str, float] = {
    "raw": -0.08,
    "fg": 0.28,
}
FAMILY_LINESTYLES: dict[str, str] = {
    "raw": "-",
    "fg": "--",
}
SINGLE_FEATURE_COLOR = "#eeeeee"
REFERENCE_COLOR = "#6e6e6e"
AXIS_COLORS: dict[str, str] = {
    "x": "#4c78a8",
    "y": "#59a14f",
    "z": "#e15759",
}


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def have_cols(df: pd.DataFrame, cols: Iterable[str]) -> bool:
    return all(c in df.columns for c in cols)


def as_bool(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.fillna(False).to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(series):
        return (series.fillna(0).astype(float) != 0.0).to_numpy(dtype=bool)
    true_values = {"true", "t", "1", "yes", "y", "valid"}
    return series.fillna("false").astype(str).str.strip().str.lower().isin(true_values).to_numpy(dtype=bool)


def to_numeric_series(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def numeric(df: pd.DataFrame, cols: Iterable[str]) -> np.ndarray:
    return df.loc[:, list(cols)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def finite_rows(arr: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(arr), axis=1)


def vector_norm(arr: np.ndarray) -> np.ndarray:
    return np.linalg.norm(arr, axis=1)


def finite_stats(values: np.ndarray | pd.Series) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "p05": np.nan,
            "p95": np.nan,
            "max_abs": np.nan,
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "rmse": float(np.sqrt(np.mean(arr ** 2))),
        "mae": float(np.mean(np.abs(arr))),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "max_abs": float(np.max(np.abs(arr))),
    }


def write_markdown_table(df: pd.DataFrame, path: Path, float_fmt: str = ".5g") -> None:
    def fmt(x: object) -> str:
        if isinstance(x, (float, np.floating)):
            if math.isnan(float(x)):
                return ""
            return format(float(x), float_fmt)
        return str(x)

    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def setup_matplotlib_style() -> None:
    plt.rcParams.update(LITERATURE_STYLE)


def tint_color(color: str, amount: float) -> str:
    rgb = np.array(mcolors.to_rgb(color), dtype=float)
    amount = float(np.clip(amount, -1.0, 1.0))
    if amount >= 0.0:
        mixed = rgb + (1.0 - rgb) * amount
    else:
        mixed = rgb * (1.0 + amount)
    return mcolors.to_hex(np.clip(mixed, 0.0, 1.0))


def method_style(method: MethodSpec, family: str) -> dict[str, object]:
    base = BASE_METHOD_COLORS.get(method.key, "#4c4c4c")
    color = tint_color(base, FAMILY_SHADE_OFFSETS.get(family, 0.0))
    return {
        "color": color,
        "linestyle": FAMILY_LINESTYLES.get(family, "-"),
    }


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, which="major", axis="both")
    ax.set_axisbelow(True)


def savefig(path: Path) -> None:
    plt.tight_layout()
    if path.suffix.lower() != ".pdf":
        plt.savefig(path, dpi=220)
        plt.savefig(path.with_suffix(".pdf"))
    else:
        plt.savefig(path)
    plt.close()


def get_time_axis(df: pd.DataFrame) -> tuple[np.ndarray, str]:
    for col in TIME_CANDIDATES:
        if col in df.columns:
            if col == "_plot_time":
                label = "Time from retained start (s)"
            elif col == "image_time":
                label = "Image time (s)"
            else:
                label = "Frame index"
            return to_numeric_series(df[col]), label
    return np.arange(len(df), dtype=float), "Row index"


def get_filter_axis(df: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Return the native axis used for trimming before any relative-time column is added."""
    for col in ("image_time", "frame_idx"):
        if col in df.columns:
            return to_numeric_series(df[col]), col
    return np.arange(len(df), dtype=float), "row_index"


def apply_replay_truncation(
        df: pd.DataFrame,
        *,
        axis_min: float | None = None,
        axis_max: float | None = None,
        start_after_single_range_below: float | None = None,
        zero_time: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply optional leading/trailing truncation for close-range replay plots.

    Truncation is intentionally performed before adding _plot_time so that --time-min
    and --time-max refer to the native image_time/frame_idx axis in the CSV.
    """
    info: dict[str, object] = {"original_rows": int(len(df))}
    if df.empty:
        info.update({"retained_rows": 0, "axis_name": "row_index"})
        return df.copy(), info

    axis, axis_name = get_filter_axis(df)
    keep = np.isfinite(axis)
    if axis_min is not None:
        keep &= axis >= float(axis_min)
    if axis_max is not None:
        keep &= axis <= float(axis_max)

    if start_after_single_range_below is not None and SINGLE_RANGE_COL in df.columns:
        single_range = to_numeric_series(df[SINGLE_RANGE_COL])
        candidate = keep & np.isfinite(single_range) & (single_range <= float(start_after_single_range_below))
        idx = np.flatnonzero(candidate)
        if idx.size:
            keep &= np.arange(len(df)) >= idx[0]
            info["range_start_row"] = int(idx[0])
            info["range_start_axis_value"] = float(axis[idx[0]])
        else:
            info["range_start_row"] = None
            info["range_start_axis_value"] = None

    out = df.loc[keep].copy()
    info.update({
        "retained_rows": int(len(out)),
        "axis_name": axis_name,
        "axis_min_requested": axis_min,
        "axis_max_requested": axis_max,
        "start_after_single_range_below": start_after_single_range_below,
    })

    if zero_time and not out.empty:
        kept_axis, kept_axis_name = get_filter_axis(out)
        out["_plot_time"] = kept_axis - kept_axis[0]
        info["plot_time_zeroed_from"] = kept_axis_name
        info["plot_time_origin"] = float(kept_axis[0])

    return out, info


def method_available(df: pd.DataFrame, method: MethodSpec) -> bool:
    needed = [method.raw_valid_col, *method.raw_t_cols, method.fg_valid_col, *method.fg_t_cols]
    return have_cols(df, needed)


def compute_range_from_cols(df: pd.DataFrame, cols: tuple[str, str, str]) -> np.ndarray:
    vec = numeric(df, cols)
    rng = vector_norm(vec)
    rng[~finite_rows(vec)] = np.nan
    return rng


def mask_with_valid(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    out[~valid] = np.nan
    return out


def sigma_range_from_diag(tvec: np.ndarray, sigmas_xyz: np.ndarray) -> np.ndarray:
    out = np.full(len(tvec), np.nan, dtype=float)
    finite = finite_rows(tvec) & finite_rows(sigmas_xyz)
    if not np.any(finite):
        return out
    r = np.linalg.norm(tvec[finite], axis=1)
    u = np.zeros_like(tvec[finite])
    nz = r > 1e-12
    u[nz] = tvec[finite][nz] / r[nz, None]
    var = np.sum((u ** 2) * (sigmas_xyz[finite] ** 2), axis=1)
    out[finite] = np.sqrt(np.clip(var, 0.0, np.inf))
    return out


# -----------------------------------------------------------------------------
# Summaries
# -----------------------------------------------------------------------------

def build_method_summary(df: pd.DataFrame, methods: list[MethodSpec]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    single_range = to_numeric_series(df[SINGLE_RANGE_COL]) if SINGLE_RANGE_COL in df.columns else np.full(len(df),
                                                                                                          np.nan)
    single_valid = np.isfinite(single_range)

    for m in methods:
        raw_valid = as_bool(df[m.raw_valid_col]) & finite_rows(numeric(df, m.raw_t_cols))
        fg_valid = as_bool(df[m.fg_valid_col]) & finite_rows(numeric(df, m.fg_t_cols))
        single_factor_valid = as_bool(
            df[m.fg_factor_single_valid_col]) if m.fg_factor_single_valid_col in df.columns else np.zeros(len(df),
                                                                                                          dtype=bool)
        multi_factor_valid = as_bool(
            df[m.fg_factor_multi_valid_col]) if m.fg_factor_multi_valid_col in df.columns else np.zeros(len(df),
                                                                                                        dtype=bool)

        raw_range = mask_with_valid(compute_range_from_cols(df, m.raw_t_cols), raw_valid)
        fg_range = mask_with_valid(compute_range_from_cols(df, m.fg_t_cols), fg_valid)

        rmse_px = to_numeric_series(df[m.raw_rmse_col]) if m.raw_rmse_col and m.raw_rmse_col in df.columns else np.full(
            len(df), np.nan)
        raw_rmse_stats = finite_stats(rmse_px[raw_valid])

        delta_fg_raw = fg_range - raw_range
        delta_fg_single = fg_range - single_range
        delta_raw_single = raw_range - single_range

        reject_counts: dict[str, int] = {}
        if m.fg_reject_reason_col in df.columns:
            vc = df[m.fg_reject_reason_col].fillna("").astype(str).value_counts()
            reject_counts = {k: int(v) for k, v in vc.items() if k}

        row: dict[str, object] = {
            "method": m.label,
            "key": m.key,
            "rows": int(len(df)),
            "single_feature_range_finite": int(np.sum(single_valid)),
            "raw_valid": int(np.sum(raw_valid)),
            "fg_valid": int(np.sum(fg_valid)),
            "fg_single_factor_valid": int(np.sum(single_factor_valid)),
            "fg_multi_factor_valid": int(np.sum(multi_factor_valid)),
            "raw_range_mean": finite_stats(raw_range)["mean"],
            "raw_range_std": finite_stats(raw_range)["std"],
            "fg_range_mean": finite_stats(fg_range)["mean"],
            "fg_range_std": finite_stats(fg_range)["std"],
            "raw_rmse_px_mean": raw_rmse_stats["mean"],
            "raw_rmse_px_p95": raw_rmse_stats["p95"],
        }

        for prefix, arr in [
            ("fg_minus_raw", delta_fg_raw),
            ("fg_minus_single", delta_fg_single),
            ("raw_minus_single", delta_raw_single),
        ]:
            stats = finite_stats(arr)
            for stat_name in ["count", "mean", "median", "std", "rmse", "mae", "p05", "p95", "max_abs"]:
                row[f"{prefix}_{stat_name}"] = stats[stat_name]

        if reject_counts:
            row["reject_reason_counts"] = "; ".join(f"{k}:{v}" for k, v in reject_counts.items())
        else:
            row["reject_reason_counts"] = ""

        rows.append(row)

    return pd.DataFrame(rows)


def build_pairwise_range_summary(df: pd.DataFrame, methods: list[MethodSpec]) -> pd.DataFrame:
    series_map: dict[str, tuple[str, np.ndarray]] = {}
    if SINGLE_RANGE_COL in df.columns:
        single = to_numeric_series(df[SINGLE_RANGE_COL])
        series_map["single_feature"] = ("Single-feature range", single)

    for m in methods:
        raw_valid = as_bool(df[m.raw_valid_col]) & finite_rows(numeric(df, m.raw_t_cols))
        fg_valid = as_bool(df[m.fg_valid_col]) & finite_rows(numeric(df, m.fg_t_cols))
        raw_range = mask_with_valid(compute_range_from_cols(df, m.raw_t_cols), raw_valid)
        fg_range = mask_with_valid(compute_range_from_cols(df, m.fg_t_cols), fg_valid)
        series_map[f"{m.key}_raw"] = (f"{m.label} raw", raw_range)
        series_map[f"{m.key}_sfmf_fg"] = (f"{m.label} fused FG", fg_range)

    keys = list(series_map.keys())
    rows: list[dict[str, object]] = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ka, kb = keys[i], keys[j]
            la, a = series_map[ka]
            lb, b = series_map[kb]
            valid = np.isfinite(a) & np.isfinite(b)
            diff = a[valid] - b[valid]
            stats = finite_stats(diff)
            rows.append({
                "series_a": la,
                "series_b": lb,
                "common_count": stats["count"],
                "difference_mean": stats["mean"],
                "difference_median": stats["median"],
                "difference_std": stats["std"],
                "difference_rmse": stats["rmse"],
                "difference_mae": stats["mae"],
                "difference_p05": stats["p05"],
                "difference_p95": stats["p95"],
                "difference_max_abs": stats["max_abs"],
            })
    return pd.DataFrame(rows)


def build_summary_text(csv_path: Path, methods: list[MethodSpec], method_summary: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("Single-feature / multi-feature fusion comparison")
    lines.append("===============================================")
    lines.append("")
    lines.append(f"Input CSV: {csv_path.name}")
    lines.append(f"Rows: {int(method_summary['rows'].iloc[0]) if not method_summary.empty else 'n/a'}")
    lines.append(f"Methods analyzed: {', '.join(m.key for m in methods)}")
    lines.append("")
    if method_summary.empty:
        lines.append("No methods were available.")
        return "\n".join(lines)

    for _, row in method_summary.iterrows():
        lines.append(f"{row['method']} ({row['key']})")
        lines.append("-" * (len(str(row['method'])) + len(str(row['key'])) + 3))
        lines.append(
            f"raw valid={int(row['raw_valid'])}, fused valid={int(row['fg_valid'])}, "
            f"single factor used={int(row['fg_single_factor_valid'])}, multi factor used={int(row['fg_multi_factor_valid'])}"
        )
        lines.append(
            f"fused-vs-raw range: mean={row['fg_minus_raw_mean']:.5g}, rmse={row['fg_minus_raw_rmse']:.5g}, "
            f"mae={row['fg_minus_raw_mae']:.5g}"
        )
        lines.append(
            f"fused-vs-single range: mean={row['fg_minus_single_mean']:.5g}, rmse={row['fg_minus_single_rmse']:.5g}, "
            f"mae={row['fg_minus_single_mae']:.5g}"
        )
        if isinstance(row.get("reject_reason_counts", ""), str) and row["reject_reason_counts"]:
            lines.append(f"reject reasons: {row['reject_reason_counts']}")
        lines.append("")

    best = method_summary[np.isfinite(method_summary["fg_minus_raw_rmse"])]
    if not best.empty:
        best = best.sort_values("fg_minus_raw_rmse")
        lines.append(f"Smallest fused-vs-raw range RMSE: {best.iloc[0]['method']}")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_range_overview(df: pd.DataFrame, methods: list[MethodSpec], out_dir: Path, stem: str) -> None:
    from support.viz.plots import finish_figure, line

    x, xlabel = get_time_axis(df)
    ylim_arrays: list[np.ndarray] = []
    series: list[np.ndarray] = []
    labels: list[str] = []
    colors: list[str] = []
    series_style: list[dict[str, object]] = []

    if SINGLE_RANGE_COL in df.columns:
        single_range = to_numeric_series(df[SINGLE_RANGE_COL])
        ylim_arrays.append(single_range)
        series.append(single_range)
        labels.append("Single-feature range")
        colors.append(SINGLE_FEATURE_COLOR)
        series_style.append({"linewidth": 1.8, "linestyle": "-"})

    for m in methods:
        raw_valid = as_bool(df[m.raw_valid_col]) & finite_rows(numeric(df, m.raw_t_cols))
        fg_valid = as_bool(df[m.fg_valid_col]) & finite_rows(numeric(df, m.fg_t_cols))
        raw_range = mask_with_valid(compute_range_from_cols(df, m.raw_t_cols), raw_valid)
        fg_range = mask_with_valid(compute_range_from_cols(df, m.fg_t_cols), fg_valid)
        ylim_arrays.extend((raw_range, fg_range))
        series.extend((raw_range, fg_range))
        labels.extend((f"{m.key} raw", f"{m.key} sfmf-fg"))
        colors.extend((method_style(m, "raw")["color"], method_style(m, "fg")["color"]))
        series_style.extend((
            {"linewidth": 1.3, "linestyle": method_style(m, "raw")["linestyle"], "alpha": 0.9},
            {"linewidth": 1.7, "linestyle": method_style(m, "fg")["linestyle"]},
        ))

    save_path = out_dir / f"{stem}_range_overview.png"
    fig, ax, _ = line(
        x=x,
        y=np.column_stack(series),
        label=labels,
        color=colors,
        series_kwargs=series_style,
        xlabel=xlabel,
        ylabel="Range",
        title="Range overview: single-feature, raw multi-feature, and fused SFMF-FG",
        legend=True,
        legend_kwargs={"ncol": 2},
        figsize=(12, 6),
        show=False,
        save=None,
    )
    set_robust_ylim(ax, ylim_arrays, lo_pct=1.0, hi_pct=99.0, pad_frac=0.15)
    finish_figure(
        fig,
        save=[save_path, save_path.with_suffix(".pdf")],
        show=False,
    )
    plt.close(fig)


def plot_per_method_range(df: pd.DataFrame, method: MethodSpec, out_dir: Path, stem: str) -> None:
    from support.viz.plots import finish_figure, line

    x, xlabel = get_time_axis(df)
    raw_valid = as_bool(df[method.raw_valid_col]) & finite_rows(numeric(df, method.raw_t_cols))
    fg_valid = as_bool(df[method.fg_valid_col]) & finite_rows(numeric(df, method.fg_t_cols))
    raw_range = mask_with_valid(compute_range_from_cols(df, method.raw_t_cols), raw_valid)
    fg_t = numeric(df, method.fg_t_cols)
    fg_range = mask_with_valid(vector_norm(fg_t), fg_valid)
    single_range = to_numeric_series(df[SINGLE_RANGE_COL]) if SINGLE_RANGE_COL in df.columns else np.full(len(df),
                                                                                                          np.nan)
    y = np.column_stack((single_range, raw_range, fg_range))
    save_path = out_dir / f"{stem}_{method.key}_range_compare.png"

    fig, ax, _ = line(
        x=x,
        y=y,
        label=["Single-feature range", f"{method.key} raw multi-feature", f"{method.key} fused SFMF-FG"],
        color=[
            SINGLE_FEATURE_COLOR,
            method_style(method, "raw")["color"],
            method_style(method, "fg")["color"],
        ],
        series_kwargs=[
            {"linewidth": 1.8, "linestyle": "-"},
            {"linewidth": 1.4, "linestyle": method_style(method, "raw")["linestyle"]},
            {"linewidth": 1.8, "linestyle": method_style(method, "fg")["linestyle"]},
        ],
        xlabel=xlabel,
        ylabel="Range",
        title=f"Range comparison: {method.label}",
        legend=True,
        figsize=(11.5, 5.5),
        show=False,
        save=None,
    )

    if have_cols(df, method.fg_sigma_cols):
        sig_xyz = numeric(df, method.fg_sigma_cols)
        sigma_r = sigma_range_from_diag(fg_t, sig_xyz)
        upper = fg_range + 3.0 * sigma_r
        lower = fg_range - 3.0 * sigma_r
        ax.fill_between(
            x,
            lower,
            upper,
            alpha=0.14,
            color=method_style(method, "fg")["color"],
            label=f"{method.key} fused +/-3sigma(range)",
        )
        leg = ax.legend()
        leg.get_frame().set_facecolor(fig.get_facecolor())
        leg.get_frame().set_edgecolor(ax.xaxis.label.get_color())
        for text in leg.get_texts():
            text.set_color(ax.xaxis.label.get_color())

    finish_figure(
        fig,
        save=[save_path, save_path.with_suffix(".pdf")],
        show=False,
    )
    plt.close(fig)


def plot_per_method_range_deltas(df: pd.DataFrame, method: MethodSpec, out_dir: Path, stem: str) -> None:
    from support.viz.plots import finish_figure, line

    x, xlabel = get_time_axis(df)
    raw_valid = as_bool(df[method.raw_valid_col]) & finite_rows(numeric(df, method.raw_t_cols))
    fg_valid = as_bool(df[method.fg_valid_col]) & finite_rows(numeric(df, method.fg_t_cols))
    raw_range = mask_with_valid(compute_range_from_cols(df, method.raw_t_cols), raw_valid)
    fg_range = mask_with_valid(compute_range_from_cols(df, method.fg_t_cols), fg_valid)
    single_range = to_numeric_series(df[SINGLE_RANGE_COL]) if SINGLE_RANGE_COL in df.columns else np.full(len(df),
                                                                                                          np.nan)
    y = np.column_stack((fg_range - raw_range, fg_range - single_range, raw_range - single_range))
    save_path = out_dir / f"{stem}_{method.key}_range_deltas.png"

    fig, ax, _ = line(
        x=x,
        y=y,
        label=["fused - raw multi", "fused - single", "raw multi - single"],
        series_kwargs=[
            {"linewidth": 1.5, "linestyle": "-"},
            {"linewidth": 1.5, "linestyle": ":"},
            {"linewidth": 1.3, "linestyle": "-."},
        ],
        xlabel=xlabel,
        ylabel="Range difference",
        title=f"Range differences: {method.label}",
        legend=True,
        figsize=(11.5, 5.2),
        show=False,
        save=None,
    )
    ax.axhline(0.0, linewidth=0.9, color=REFERENCE_COLOR)
    finish_figure(
        fig,
        save=[save_path, save_path.with_suffix(".pdf")],
        show=False,
    )
    plt.close(fig)


def plot_xyz_comparison(df: pd.DataFrame, method: MethodSpec, out_dir: Path, stem: str) -> None:
    from support.viz.plots import finish_figure, line

    x, xlabel = get_time_axis(df)
    raw_valid = as_bool(df[method.raw_valid_col]) & finite_rows(numeric(df, method.raw_t_cols))
    fg_valid = as_bool(df[method.fg_valid_col]) & finite_rows(numeric(df, method.fg_t_cols))
    raw_t = numeric(df, method.raw_t_cols)
    fg_t = numeric(df, method.fg_t_cols)
    raw_t[~raw_valid] = np.nan
    fg_t[~fg_valid] = np.nan
    single_t = numeric(df, SINGLE_T_COLS) if have_cols(df, SINGLE_T_COLS) else None

    for i, axis in enumerate("xyz"):
        series = [raw_t[:, i], fg_t[:, i]]
        labels = [f"{method.key} raw {axis}", f"{method.key} fused {axis}"]
        colors = [method_style(method, "raw")["color"], method_style(method, "fg")["color"]]
        series_style = [
            {"linewidth": 1.4, "linestyle": "-"},
            {"linewidth": 1.7, "linestyle": "--"},
        ]
        if single_t is not None:
            series.insert(0, single_t[:, i])
            labels.insert(0, f"Single-feature {axis}")
            colors.insert(0, SINGLE_FEATURE_COLOR)
            series_style.insert(0, {"linewidth": 1.2, "linestyle": ":"})

        save_path = out_dir / f"{stem}_{method.key}_{axis}_compare.png"
        fig, _, _ = line(
            x=x,
            y=np.column_stack(series),
            label=labels,
            color=colors,
            series_kwargs=series_style,
            xlabel=xlabel,
            ylabel=f"{axis.upper()} component",
            title=f"{axis.upper()} component comparison: {method.label}",
            legend=True,
            figsize=(11.5, 5.2),
            show=False,
            save=None,
        )
        finish_figure(
            fig,
            save=[save_path, save_path.with_suffix(".pdf")],
            show=False,
        )
        plt.close(fig)


def set_robust_ylim(
        ax: plt.Axes,
        arrays: list[np.ndarray],
        lo_pct: float = 1.0,
        hi_pct: float = 99.0,
        pad_frac: float = 0.12,
        min_span: float = 1.0,
) -> None:
    vals = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return

    lo = float(np.percentile(vals, lo_pct))
    hi = float(np.percentile(vals, hi_pct))

    if not np.isfinite(lo) or not np.isfinite(hi):
        return

    span = max(hi - lo, min_span)
    center = 0.5 * (lo + hi)
    half = 0.5 * span * (1.0 + 2.0 * pad_frac)

    ax.set_ylim(center - half, center + half)


def plot_xyz_stacked_comparison(df: pd.DataFrame, method: MethodSpec, out_dir: Path, stem: str) -> None:
    """Create a compact three-panel component plot for dissertation-body use."""
    from support.viz.plots import finish_figure, line, setup_subplots

    x, xlabel = get_time_axis(df)
    raw_valid = as_bool(df[method.raw_valid_col]) & finite_rows(numeric(df, method.raw_t_cols))
    fg_valid = as_bool(df[method.fg_valid_col]) & finite_rows(numeric(df, method.fg_t_cols))
    raw_t = numeric(df, method.raw_t_cols)
    fg_t = numeric(df, method.fg_t_cols)
    raw_t[~raw_valid] = np.nan
    fg_t[~fg_valid] = np.nan
    single_t = numeric(df, SINGLE_T_COLS) if have_cols(df, SINGLE_T_COLS) else None

    fig, axes = setup_subplots(
        nrows=3,
        ncols=1,
        figsize=(11.5, 8.2),
        sharex=True,
        suptitle=f"Camera-frame component comparison: {method.label}",
        xlabels=["", "", xlabel],
        ylabels=["X [m]", "Y [m]", "Z [m]"],
        titles=["x lateral", "y vertical", "z range"],
    )
    axis_names = ["x lateral", "y vertical", "z range"]
    for i, (axis, axis_name) in enumerate(zip("xyz", axis_names, strict=True)):
        ax = axes[i]
        series = [raw_t[:, i], fg_t[:, i]]
        labels = [f"{method.key} raw", f"{method.key} SFMF-FG"]
        colors = [method_style(method, "raw")["color"], method_style(method, "fg")["color"]]
        series_style = [
            {"linewidth": 1.25, "linestyle": "-"},
            {"linewidth": 1.65, "linestyle": "--"},
        ]

        if single_t is not None:
            series.insert(0, single_t[:, i])
            labels.insert(0, "Single-feature")
            colors.insert(0, SINGLE_FEATURE_COLOR)
            series_style.insert(0, {"linewidth": 1.1, "linestyle": ":"})

        line(
            x=x,
            y=np.column_stack(series),
            ax=ax,
            label=labels,
            color=colors,
            series_kwargs=series_style,
            legend=(i == 0),
            legend_kwargs={"ncol": 3, "loc": "best"},
            show=False,
            save=None,
        )

        # Use robust scaling so a few single-feature spikes do not flatten the useful trace.
        # Include all plotted series, but ignore the most extreme 1% tails.
        ylim_arrays = [raw_t[:, i], fg_t[:, i]]
        if single_t is not None:
            ylim_arrays.append(single_t[:, i])
        set_robust_ylim(ax, ylim_arrays, lo_pct=1.0, hi_pct=99.0, pad_frac=0.15)

        style_axes(ax)
    save_path = out_dir / f"{stem}_{method.key}_xyz_compare.png"
    finish_figure(
        fig,
        tight_layout=True,
        tight_layout_rect=(0, 0, 1, 0.97),
        save=[save_path, save_path.with_suffix(".pdf")],
        show=False,
    )
    plt.close(fig)


def plot_fused_sigmas(df: pd.DataFrame, method: MethodSpec, out_dir: Path, stem: str) -> None:
    if not have_cols(df, method.fg_sigma_cols):
        return
    from support.viz.plots import line

    x, xlabel = get_time_axis(df)
    sig = numeric(df, method.fg_sigma_cols)
    t_fg = numeric(df, method.fg_t_cols)
    sigma_r = sigma_range_from_diag(t_fg, sig)
    y = np.column_stack((sig[:, 0], sig[:, 1], sig[:, 2], sigma_r))
    save_path = out_dir / f"{stem}_{method.key}_sigma_translation.png"

    fig, _, _ = line(
        x=x,
        y=y,
        label=["sigma_x", "sigma_y", "sigma_z", "sigma_range (diag projection)"],
        color=[AXIS_COLORS["x"], AXIS_COLORS["y"], AXIS_COLORS["z"], method_style(method, "fg")["color"]],
        series_kwargs=[
            {"linewidth": 1.4, "linestyle": "-"},
            {"linewidth": 1.4, "linestyle": "-"},
            {"linewidth": 1.4, "linestyle": "-"},
            {"linewidth": 1.9, "linestyle": "--"},
        ],
        xlabel=xlabel,
        ylabel="1-sigma",
        title=f"Fused translation sigma diagnostics: {method.label}",
        legend=True,
        figsize=(11.5, 5.2),
        save=[save_path, save_path.with_suffix(".pdf")],
        show=False,
    )
    plt.close(fig)


def plot_factor_validity(df: pd.DataFrame, method: MethodSpec, out_dir: Path, stem: str) -> None:
    from support.viz.plots import line

    x, xlabel = get_time_axis(df)
    overall = as_bool(df[method.fg_valid_col]).astype(float)
    single_valid = as_bool(df[method.fg_factor_single_valid_col]).astype(
        float) if method.fg_factor_single_valid_col in df.columns else np.zeros(len(df))
    multi_valid = as_bool(df[method.fg_factor_multi_valid_col]).astype(
        float) if method.fg_factor_multi_valid_col in df.columns else np.zeros(len(df))
    y = np.column_stack((overall, single_valid, multi_valid))
    save_path = out_dir / f"{stem}_{method.key}_factor_validity.png"

    fig, _, _ = line(
        x=x,
        y=y,
        label=["overall fused valid", "single-feature factor valid", "multi-feature factor valid"],
        color=[method_style(method, "fg")["color"], SINGLE_FEATURE_COLOR, method_style(method, "raw")["color"]],
        series_kwargs=[
            {"linestyle": "--", "linewidth": 1.8},
            {"linestyle": ":", "linewidth": 1.3},
            {"linestyle": "-", "linewidth": 1.3},
        ],
        xlabel=xlabel,
        ylabel="Validity",
        title=f"Factor usage / validity: {method.label}",
        legend=True,
        yticks=[0.0, 1.0],
        yticklabels=["False", "True"],
        ylim=(-0.1, 1.1),
        figsize=(11.5, 4.6),
        save=[save_path, save_path.with_suffix(".pdf")],
        show=False,
    )
    plt.close(fig)


def plot_raw_rmse(df: pd.DataFrame,
                  method: MethodSpec,
                  out_dir: Path,
                  stem: str) -> None:
    if not method.raw_rmse_col or method.raw_rmse_col not in df.columns:
        return
    from support.viz.plots import line

    x, xlabel = get_time_axis(df)
    rmse = to_numeric_series(df[method.raw_rmse_col])
    save_path = out_dir / f"{stem}_{method.key}_raw_rmse.png"
    line(x=x,
         y=rmse,
         label="raw multi-feature RMSE [px]",
         xlabel=xlabel,
         ylabel="RMSE [px]",
         title=f"Raw multi-feature reprojection RMSE: {method.label}",
         legend=True,
         save=[save_path, save_path.with_suffix('.pdf')],
         linewidth=1.5,
         figsize=(11.5, 5.0),
         show=False)


def plot_reject_reasons(df: pd.DataFrame,
                        method: MethodSpec,
                        out_dir: Path,
                        stem: str) -> None:
    if method.fg_reject_reason_col not in df.columns:
        return

    vc = df[method.fg_reject_reason_col].fillna("").astype(str).value_counts()
    vc = vc[vc.index != ""]
    if vc.empty:
        return

    save_path = out_dir / f"{stem}_{method.key}_reject_reasons.png"
    from support.viz.plots import bar
    bar(
        x=vc.index.astype(str),
        height=vc.to_numpy(dtype=float),
        ylabel="Count",
        title=f"Fused reject reasons: {method.label}",
        xticklabel_kwargs={"rotation": 25, "ha": "right"},
        save=[save_path, save_path.with_suffix('.pdf')],
        show=False
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate lovely comparison plots for SFMF fusion navcalcs CSV files.")
    p.add_argument("--csv", nargs="+", type=Path, default=None, help="One or more SFMF fusion navcalcs CSV files.")
    p.add_argument("--out-root", type=Path, default=None,
                   help="Optional output root directory. Default creates <csvstem>_comparison_plots beside each CSV.")
    p.add_argument("--methods", nargs="*", default=None, help="Subset of methods to plot, e.g. pnp qnp wqnp_kfest")
    p.add_argument("--main-method", default="pnp", choices=[m.key for m in METHODS],
                   help="Representative method used for dissertation-body stacked plots.")
    p.add_argument("--suffix", default="",
                   help="Optional suffix appended to output directory and file stems, e.g. _truncated.")
    p.add_argument("--time-min", type=float, default=None,
                   help="Retain rows with native image_time/frame_idx >= this value.")
    p.add_argument("--time-max", type=float, default=None,
                   help="Retain rows with native image_time/frame_idx <= this value.")
    p.add_argument("--start-after-single-range-below", type=float, default=None,
                   help="Drop leading rows until single_feature_estimated_range is at or below this value.")
    p.add_argument("--zero-time", action="store_true",
                   help="Plot retained rows with time reset to zero at the first retained row.")
    p.add_argument("--plot-set", choices=("all", "main"), default="all",
                   help="all: every diagnostic; main: range overview plus main-method plots only.")
    return p.parse_args()


def maybe_dialog_csv_paths(args: argparse.Namespace) -> argparse.Namespace:
    if args.csv:
        return args
    try:
        from tkinter import Tk, filedialog

        root = Tk()
        root.withdraw()
        selected = filedialog.askopenfilenames(
            title="Select one or more SFMF fusion navcalcs CSV files",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        root.destroy()
        args.csv = [Path(path) for path in selected] if selected else None
    except Exception:
        return args
    return args


def process_csv(csv_path: Path, out_root: Path | None, requested_methods: set[str] | None,
                args: argparse.Namespace) -> None:
    df_full = pd.read_csv(csv_path)
    df, trunc_info = apply_replay_truncation(
        df_full,
        axis_min=args.time_min,
        axis_max=args.time_max,
        start_after_single_range_below=args.start_after_single_range_below,
        zero_time=args.zero_time,
    )
    if df.empty:
        raise RuntimeError(f"No rows remain after truncation for {csv_path}")

    suffix = args.suffix.strip()
    if suffix and not suffix.startswith("_"):
        suffix = "_" + suffix
    stem = f"{csv_path.stem}{suffix}"
    out_name = f"{stem}_comparison_plots"
    out_dir = (out_root / out_name) if out_root is not None else csv_path.with_name(out_name)
    tables_dir = out_dir / "tables"
    plots_dir = out_dir / "plots"
    ensure_dir(plots_dir)
    ensure_dir(tables_dir)

    methods = [m for m in METHODS if
               method_available(df, m) and (requested_methods is None or m.key in requested_methods)]
    if not methods:
        raise RuntimeError(f"No supported methods found in {csv_path}")

    method_summary = build_method_summary(df, methods)
    pairwise_summary = build_pairwise_range_summary(df, methods)
    summary_text = build_summary_text(csv_path, methods, method_summary)
    trunc_lines = ["", "Truncation / plotting window", "----------------------------"]
    for k, v in trunc_info.items():
        trunc_lines.append(f"{k}: {v}")
    summary_text = summary_text + "\n" + "\n".join(trunc_lines)

    method_summary.to_csv(tables_dir / f"{stem}_method_summary.csv", index=False)
    pairwise_summary.to_csv(tables_dir / f"{stem}_pairwise_range_summary.csv", index=False)
    write_markdown_table(method_summary, tables_dir / f"{stem}_method_summary.md")
    write_markdown_table(pairwise_summary, tables_dir / f"{stem}_pairwise_range_summary.md")
    (tables_dir / f"{stem}_summary.txt").write_text(summary_text, encoding="utf-8")
    pd.DataFrame([trunc_info]).to_csv(tables_dir / f"{stem}_truncation_info.csv", index=False)

    main_method = next((m for m in methods if m.key == args.main_method), methods[0])

    plot_range_overview(df, methods, plots_dir, stem)
    if args.plot_set == "main":
        plot_per_method_range(df, main_method, plots_dir, stem)
        plot_xyz_stacked_comparison(df, main_method, plots_dir, stem)
        plot_raw_rmse(df, main_method, plots_dir, stem)
        plot_reject_reasons(df, main_method, plots_dir, stem)
    else:
        for m in methods:
            plot_per_method_range(df, m, plots_dir, stem)
            plot_per_method_range_deltas(df, m, plots_dir, stem)
            plot_xyz_comparison(df, m, plots_dir, stem)
            plot_xyz_stacked_comparison(df, m, plots_dir, stem)
            plot_fused_sigmas(df, m, plots_dir, stem)
            plot_factor_validity(df, m, plots_dir, stem)
            plot_raw_rmse(df, m, plots_dir, stem)
            plot_reject_reasons(df, m, plots_dir, stem)

    print(summary_text)
    print(f"\nWrote comparison plots and tables to: {out_dir}")


def main() -> None:
    args = maybe_dialog_csv_paths(parse_args())
    if not args.csv:
        raise SystemExit("Provide --csv or select one or more CSV files in the file dialog.")
    setup_matplotlib_style()
    requested_methods = set(args.methods) if args.methods else ["pnp", "qnp", "wqnp_yolo", "wqnp_kfest"]
    for csv_path in args.csv:
        process_csv(Path(csv_path), args.out_root, requested_methods, args)


if __name__ == "__main__":
    main()
