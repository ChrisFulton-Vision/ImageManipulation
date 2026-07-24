"""Small plotting/theme helpers built around Matplotlib.

The default theme is a dark, gold-accented "space clock" style inspired by
black-hole / astronomical imagery.  The module is intentionally lightweight:
all public plotting functions accept generic Matplotlib overrides and return
``(fig, ax, artist)`` so callers can keep customizing after the helper runs.

Run this file directly to generate a demo gallery:

    python plots.py

"""

from __future__ import annotations

from cycler import cycler
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence
from collections.abc import Callable

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, LinearSegmentedColormap, Normalize, to_hex
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
except Exception:  # pragma: no cover - only matters on unusual Matplotlib installs
    Axes3D = None  # type: ignore[assignment]

# -----------------------------------------------------------------------------
# Colormaps
# -----------------------------------------------------------------------------

SPACE_CLOCK_COLORS = [
    # "#030712",
    "#123f73",
    "#1f8ec7",
    "#ffd27a",
    "#b43b35",
    "#5a224e",
    # "#030712",  # close the loop for phase / angle plots
]

SPACE_CLOCK_SEQUENTIAL_COLORS = [
    # "#030712",
    # "#0b1f3a",
    "#123f73",
    "#1f8ec7",
    "#5a224e",
    "#b43b35",
    "#f08a3c",
    "#ffd27a",
]

AURORA_COLORS = [
    "#020617",
    "#073b4c",
    "#118ab2",
    "#06d6a0",
    "#c7f9cc",
]

FIRE_ICE_COLORS = [
    "#031926",
    "#1f7a8c",
    "#bfdbf7",
    "#f4f1de",
    "#f9844a",
    "#bc3908",
]


def make_colormap(
        colors: Sequence[str | tuple[float, float, float] | tuple[float, float, float, float]],
        *,
        name: str = "custom",
        n: int = 256,
) -> LinearSegmentedColormap:
    """Create a Matplotlib colormap from ordered color stops."""
    return LinearSegmentedColormap.from_list(name, list(colors), N=n)


space_clock = make_colormap(SPACE_CLOCK_COLORS, name="space_clock")
space_clock_sequential = make_colormap(
    SPACE_CLOCK_SEQUENTIAL_COLORS,
    name="space_clock_sequential",
)
aurora = make_colormap(AURORA_COLORS, name="aurora")
fire_ice = make_colormap(FIRE_ICE_COLORS, name="fire_ice")

COLORMAPS: dict[str, Colormap] = {
    "space_clock": space_clock,
    "space_clock_sequential": space_clock_sequential,
    "aurora": aurora,
    "fire_ice": fire_ice,
    "magma": plt.get_cmap("magma"),
    "inferno": plt.get_cmap("inferno"),
    "twilight_shifted": plt.get_cmap("twilight_shifted"),
    "viridis": plt.get_cmap("viridis"),
}


def get_cmap(cmap: str | Colormap | None, default: Colormap | None = None) -> Colormap:
    """Resolve a colormap name/object into a Colormap."""
    if cmap is None:
        if default is None:
            return space_clock
        return default
    if isinstance(cmap, str):
        if cmap in COLORMAPS:
            return COLORMAPS[cmap]
        return plt.get_cmap(cmap)
    return cmap


def sample_cmap(cmap: str | Colormap = space_clock, n: int = 8) -> list[str]:
    """Return ``n`` representative hex colors sampled from a colormap."""
    resolved = get_cmap(cmap)
    return [to_hex(resolved(t)) for t in np.linspace(0.0, 1.0, n)]


# -----------------------------------------------------------------------------
# Themes
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Theme:
    name: str = "space_clock"
    background: str = "#030712"
    foreground: str = "#ffd27a"
    accent: str = "#1f8ec7"
    grid: tuple[float, float, float, float] = (1.0, 0.82, 0.48, 0.18)
    pane: tuple[float, float, float, float] = (0.01, 0.03, 0.07, 1.0)
    cmap: str | Colormap = "space_clock"
    color_cycle: tuple[str, ...] = (
        "#1f8ec7",
        "#f08a3c",
        "#b43b35",
        "#5a224e",
        "#ffd27a",
    )
    figsize_2d: tuple[float, float] = (8.5, 5.2)
    figsize_3d: tuple[float, float] = (9.0, 7.0)
    elev: float = 32.0
    azim: float = -55.0

    def with_overrides(self, **kwargs: Any) -> "Theme":
        return replace(self, **{k: v for k, v in kwargs.items() if v is not None})


SPACE_CLOCK = Theme()
PRESENTATION = SPACE_CLOCK
INTERACTIVE = SPACE_CLOCK.with_overrides(name="interactive", figsize_3d=(8.0, 6.0))
PAPER = Theme(
    name="paper",
    background="white",
    foreground="black",
    accent="#1f77b4",
    grid=(0.0, 0.0, 0.0, 0.16),
    pane=(1.0, 1.0, 1.0, 1.0),
    cmap="viridis",
    color_cycle=(
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
    ),
)

THEMES: dict[str, Theme] = {
    "space_clock": SPACE_CLOCK,
    "presentation": PRESENTATION,
    "interactive": INTERACTIVE,
    "paper": PAPER,
}


def get_theme(theme: str | Theme | None = None, **overrides: Any) -> Theme:
    """Resolve a theme name/object and apply optional dataclass-field overrides."""
    if theme is None:
        base = SPACE_CLOCK
    elif isinstance(theme, str):
        base = THEMES[theme]
    else:
        base = theme
    return base.with_overrides(**overrides)


# -----------------------------------------------------------------------------
# Axis / figure styling helpers
# -----------------------------------------------------------------------------

AxesKind = Literal["2d", "3d"]
SavePath = str | Path
SaveTarget = SavePath | Sequence[SavePath]


def _expand_axis_value(
        value: str | Sequence[str] | None,
        count: int,
        *,
        default: str,
) -> list[str]:
    if value is None:
        return [default] * count
    if isinstance(value, str):
        return [value] * count
    values = list(value)
    if len(values) != count:
        raise ValueError(f"Expected {count} axis values, got {len(values)}")
    return values


def apply_theme(ax: Axes, theme: str | Theme | None = None, *, is_3d: bool | None = None) -> Axes:
    """Apply theme colors to an existing Matplotlib axis."""
    th = get_theme(theme)
    ax.set_facecolor(th.background)
    ax.set_prop_cycle(cycler(color=th.color_cycle))
    ax.tick_params(colors=th.foreground)

    ax.title.set_color(th.foreground)
    ax.xaxis.label.set_color(th.foreground)
    ax.yaxis.label.set_color(th.foreground)

    for spine in ax.spines.values():
        spine.set_color(th.foreground)
        spine.set_alpha(0.35)

    ax.grid(True, color=th.grid)

    if is_3d is None:
        is_3d = hasattr(ax, "zaxis")

    if is_3d:
        ax.zaxis.label.set_color(th.foreground)  # type: ignore[attr-defined]
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:  # type: ignore[attr-defined]
            axis.set_pane_color(th.pane)
            axis._axinfo["grid"]["color"] = th.grid

    return ax


def setup_axes(
        *,
        kind: AxesKind = "2d",
        theme: str | Theme | None = None,
        title: str = "",
        xlabel: str = "x",
        ylabel: str = "y",
        zlabel: str = "z",
        figsize: tuple[float, float] | None = None,
        elev: float | None = None,
        azim: float | None = None,
        fig: Figure | None = None,
        ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Create or style a Matplotlib axis."""
    th = get_theme(theme)

    if ax is not None:
        fig = ax.figure
    elif fig is not None:
        ax = fig.add_subplot(111, projection="3d" if kind == "3d" else None)
    else:
        figsize = figsize or (th.figsize_3d if kind == "3d" else th.figsize_2d)
        fig = plt.figure(figsize=figsize, facecolor=th.background)
        ax = fig.add_subplot(111, projection="3d" if kind == "3d" else None)

    fig.patch.set_facecolor(th.background)
    apply_theme(ax, th, is_3d=(kind == "3d"))

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if kind == "3d":
        ax.set_zlabel(zlabel)  # type: ignore[attr-defined]
        ax.view_init(elev=th.elev if elev is None else elev,
                     azim=th.azim if azim is None else azim)  # type: ignore[attr-defined]

    return fig, ax


def setup_subplots(
        *,
        nrows: int = 1,
        ncols: int = 1,
        kind: AxesKind = "2d",
        theme: str | Theme | None = None,
        figsize: tuple[float, float] | None = None,
        sharex: bool = False,
        sharey: bool = False,
        squeeze: bool = True,
        suptitle: str | None = None,
        titles: str | Sequence[str] | None = None,
        xlabels: str | Sequence[str] | None = None,
        ylabels: str | Sequence[str] | None = None,
        zlabels: str | Sequence[str] | None = None,
        elev: float | None = None,
        azim: float | None = None,
) -> tuple[Figure, Any]:
    """Create a themed subplot grid and apply per-axis labels/titles."""
    th = get_theme(theme)
    count = nrows * ncols

    if figsize is None:
        base_w, base_h = th.figsize_3d if kind == "3d" else th.figsize_2d
        figsize = (base_w * ncols, base_h * nrows)

    subplot_kw = {"projection": "3d"} if kind == "3d" else None
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        squeeze=squeeze,
        subplot_kw=subplot_kw,
    )
    fig.patch.set_facecolor(th.background)

    axes_array = np.atleast_1d(axes).ravel()
    resolved_titles = _expand_axis_value(titles, count, default="")
    resolved_xlabels = _expand_axis_value(xlabels, count, default="x")
    resolved_ylabels = _expand_axis_value(ylabels, count, default="y")
    resolved_zlabels = _expand_axis_value(zlabels, count, default="z")

    for i, ax in enumerate(axes_array):
        apply_theme(ax, th, is_3d=(kind == "3d"))
        ax.set_title(resolved_titles[i])
        ax.set_xlabel(resolved_xlabels[i])
        ax.set_ylabel(resolved_ylabels[i])
        if kind == "3d":
            ax.set_zlabel(resolved_zlabels[i])  # type: ignore[attr-defined]
            ax.view_init(
                elev=th.elev if elev is None else elev,
                azim=th.azim if azim is None else azim,
            )  # type: ignore[attr-defined]

    if suptitle:
        fig.suptitle(suptitle, color=th.foreground)

    return fig, axes


def add_colorbar(
        fig: Figure,
        ax: Axes,
        mappable: Any,
        *,
        label: str = "value",
        theme: str | Theme | None = None,
        shrink: float = 0.8,
        pad: float = 0.05,
        **kwargs: Any,
):
    """Add a theme-colored colorbar."""
    th = get_theme(theme)
    cbar = fig.colorbar(mappable, ax=ax, shrink=shrink, pad=pad, **kwargs)
    cbar.set_label(label, color=th.foreground)
    cbar.ax.yaxis.set_tick_params(color=th.foreground)
    plt.setp(cbar.ax.get_yticklabels(), color=th.foreground)
    cbar.outline.set_edgecolor(th.foreground)
    cbar.outline.set_alpha(0.35)
    return cbar


def _finish(
    fig: Figure,
    *,
    tight_layout: bool = True,
    show: bool = True,
    save: SaveTarget | None = None,
) -> None:
    if tight_layout:
        fig.tight_layout()

    if save is not None:
        save_targets = [save] if isinstance(save, (str, Path)) else list(save)
        for save_target in save_targets:
            fig.savefig(
                save_target,
                dpi=180,
                bbox_inches="tight",
                facecolor=fig.get_facecolor(),
                edgecolor=fig.get_edgecolor(),
            )

    if show:
        plt.show()


def finish_figure(
    fig: Figure,
    *,
    tight_layout: bool = True,
    tight_layout_rect: tuple[float, float, float, float] | None = None,
    show: bool = True,
    save: SaveTarget | None = None,
) -> None:
    """Finalize and optionally save an already-constructed figure."""
    if tight_layout:
        if tight_layout_rect is None:
            fig.tight_layout()
        else:
            fig.tight_layout(rect=tight_layout_rect)

    if save is not None:
        save_targets = [save] if isinstance(save, (str, Path)) else list(save)
        for save_target in save_targets:
            fig.savefig(
                save_target,
                dpi=180,
                bbox_inches="tight",
                facecolor=fig.get_facecolor(),
                edgecolor=fig.get_edgecolor(),
            )

    if show:
        plt.show()


# -----------------------------------------------------------------------------
# Plot builders
# -----------------------------------------------------------------------------

StyleName = Literal["presentation", "interactive", "paper"]


def surface(
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        *,
        theme: str | Theme | None = None,
        style: StyleName = "presentation",
        cmap: str | Colormap | None = None,
        title: str = "3D surface",
        xlabel: str = "x",
        ylabel: str = "y",
        zlabel: str = "z",
        show_colorbar: bool = True,
        colorbar_label: str = "value",
        elev: float | None = None,
        azim: float | None = None,
        rstride: int | None = None,
        cstride: int | None = None,
        antialiased: bool | None = None,
        shade: bool = False,
        show: bool = True,
        save: SaveTarget | None = None,
        ax: Axes | None = None,
        **surface_kwargs: Any,
) -> tuple[Figure, Axes, Any]:
    """Make a styled 3D surface plot."""
    th = get_theme(theme or style)
    stride_default = 3 if style == "interactive" else 1
    aa_default = False if style == "interactive" else True

    fig, ax = setup_axes(kind="3d", theme=th, title=title, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, elev=elev,
                         azim=azim, ax=ax)

    surf = ax.plot_surface(
        X,
        Y,
        Z,
        cmap=get_cmap(cmap, get_cmap(th.cmap)),
        linewidth=0,
        antialiased=aa_default if antialiased is None else antialiased,
        shade=shade,
        rstride=stride_default if rstride is None else rstride,
        cstride=stride_default if cstride is None else cstride,
        **surface_kwargs,
    )

    if show_colorbar:
        add_colorbar(fig, ax, surf, label=colorbar_label, theme=th, shrink=0.65, pad=0.1)

    _finish(fig, show=show, save=save)
    return fig, ax, surf


# Backward-compatible name from the earlier file.
def plot_space_clock_surface(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, **kwargs: Any) -> tuple[Figure, Axes, Any]:
    return surface(X, Y, Z, theme="space_clock", title=kwargs.pop("title", "3D Surface with space_clock Colormap"),
                   **kwargs)


def wireframe(
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        *,
        theme: str | Theme | None = None,
        title: str = "3D wireframe",
        color: str | None = None,
        show: bool = True,
        save: SaveTarget | None = None,
        ax: Axes | None = None,
        **kwargs: Any,
) -> tuple[Figure, Axes, Any]:
    th = get_theme(theme)
    fig, ax = setup_axes(kind="3d", theme=th, title=title, ax=ax)
    artist = ax.plot_wireframe(X, Y, Z, color=color or th.accent, **kwargs)
    _finish(fig, show=show, save=save)
    return fig, ax, artist


def heatmap(
        Z: np.ndarray,
        *,
        x: Sequence[float] | None = None,
        y: Sequence[float] | None = None,
        theme: str | Theme | None = None,
        cmap: str | Colormap | None = None,
        title: str = "Heatmap",
        xlabel: str = "x",
        ylabel: str = "y",
        show_colorbar: bool = True,
        colorbar_label: str = "value",
        origin: str = "lower",
        aspect: str = "auto",
        show: bool = True,
        save: SaveTarget | None = None,
        ax: Axes | None = None,
        **kwargs: Any,
) -> tuple[Figure, Axes, Any]:
    th = get_theme(theme)
    fig, ax = setup_axes(kind="2d", theme=th, title=title, xlabel=xlabel, ylabel=ylabel, ax=ax)
    extent = None
    if x is not None and y is not None:
        extent = [min(x), max(x), min(y), max(y)]
    artist = ax.imshow(Z, cmap=get_cmap(cmap, get_cmap(th.cmap)), origin=origin, aspect=aspect, extent=extent, **kwargs)
    if show_colorbar:
        add_colorbar(fig, ax, artist, label=colorbar_label, theme=th)
    _finish(fig, show=show, save=save)
    return fig, ax, artist


def contour(
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        *,
        theme: str | Theme | None = None,
        cmap: str | Colormap | None = None,
        title: str = "Contour",
        xlabel: str = "x",
        ylabel: str = "y",
        levels: int | Sequence[float] = 24,
        filled: bool = True,
        show_colorbar: bool = True,
        colorbar_label: str = "value",
        show: bool = True,
        save: SaveTarget | None = None,
        ax: Axes | None = None,
        **kwargs: Any,
) -> tuple[Figure, Axes, Any]:
    th = get_theme(theme)
    fig, ax = setup_axes(kind="2d", theme=th, title=title, xlabel=xlabel, ylabel=ylabel, ax=ax)
    fn = ax.contourf if filled else ax.contour
    artist = fn(X, Y, Z, levels=levels, cmap=get_cmap(cmap, get_cmap(th.cmap)), **kwargs)
    if show_colorbar:
        add_colorbar(fig, ax, artist, label=colorbar_label, theme=th)
    _finish(fig, show=show, save=save)
    return fig, ax, artist


def line(
        x: Sequence[float],
        y: Sequence[float] | np.ndarray,
        *,
        theme: str | Theme | None = None,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        color: str | Sequence[str] | None = None,
        label: str | Sequence[str] | None = None,
        series_kwargs: Sequence[dict[str, Any]] | None = None,
        legend: bool = False,
        legend_kwargs: dict[str, Any] | None = None,
        xticks: Sequence[float] | None = None,
        xticklabels: Sequence[str] | None = None,
        xticklabel_kwargs: dict[str, Any] | None = None,
        yticks: Sequence[float] | None = None,
        yticklabels: Sequence[str] | None = None,
        yticklabel_kwargs: dict[str, Any] | None = None,
        ylim: tuple[float, float] | None = None,
        show: bool = True,
        save: SaveTarget | None = None,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        **kwargs: Any,
) -> tuple[Figure, Axes, Any]:
    th = get_theme(theme)
    resolved_title = title
    resolved_xlabel = xlabel
    resolved_ylabel = ylabel

    if ax is not None:
        if resolved_title is None:
            resolved_title = ax.get_title()
        if resolved_xlabel is None:
            resolved_xlabel = ax.get_xlabel()
        if resolved_ylabel is None:
            resolved_ylabel = ax.get_ylabel()
    else:
        if resolved_title is None:
            resolved_title = "Line plot"
        if resolved_xlabel is None:
            resolved_xlabel = "x"
        if resolved_ylabel is None:
            resolved_ylabel = "y"

    fig, ax = setup_axes(
        kind="2d",
        theme=th,
        title=resolved_title,
        xlabel=resolved_xlabel,
        ylabel=resolved_ylabel,
        ax=ax,
        figsize=figsize,
    )
    plot_kwargs = dict(kwargs)

    y_arr = np.asarray(y)
    is_multi_series = y_arr.ndim == 2 and y_arr.shape[1] > 1

    if is_multi_series:
        artists = []
        labels = list(label) if isinstance(label, Sequence) and not isinstance(label, str) else None
        colors = list(color) if isinstance(color, Sequence) and not isinstance(color, str) else None

        for i in range(y_arr.shape[1]):
            current_series_kwargs = dict(plot_kwargs)
            if series_kwargs is not None:
                if i >= len(series_kwargs):
                    raise ValueError("series_kwargs must match the number of y series")
                current_series_kwargs.update(series_kwargs[i])
            if colors is not None:
                current_series_kwargs["color"] = colors[i]
            elif isinstance(color, str):
                current_series_kwargs["color"] = color

            series_label = labels[i] if labels is not None else (label if isinstance(label, str) else None)
            artists.extend(ax.plot(x, y_arr[:, i], label=series_label, **current_series_kwargs))

        artist = artists
    else:
        if color is not None:
            plot_kwargs["color"] = color
        artist = ax.plot(x, y, label=label, **plot_kwargs)

    if xticks is not None:
        if xticklabels is None:
            ax.set_xticks(xticks)
        else:
            ax.set_xticks(xticks, labels=xticklabels)
    elif xticklabels is not None:
        ax.set_xticklabels(xticklabels)

    if yticks is not None:
        if yticklabels is None:
            ax.set_yticks(yticks)
        else:
            ax.set_yticks(yticks, labels=yticklabels)
    elif yticklabels is not None:
        ax.set_yticklabels(yticklabels)

    if xticklabel_kwargs:
        plt.setp(ax.get_xticklabels(), **xticklabel_kwargs)
    if yticklabel_kwargs:
        plt.setp(ax.get_yticklabels(), **yticklabel_kwargs)
    if ylim is not None:
        ax.set_ylim(*ylim)

    if legend or label:
        leg = ax.legend(**(legend_kwargs or {}))
        leg.get_frame().set_facecolor(th.background)
        leg.get_frame().set_edgecolor(th.foreground)
        for text in leg.get_texts():
            text.set_color(th.foreground)
    _finish(fig, show=show, save=save)
    return fig, ax, artist


def scatter(
        x: Sequence[float],
        y: Sequence[float],
        *,
        c: Sequence[float] | None = None,
        theme: str | Theme | None = None,
        cmap: str | Colormap | None = None,
        title: str = "Scatter",
        xlabel: str = "x",
        ylabel: str = "y",
        show_colorbar: bool = True,
        colorbar_label: str = "value",
        show: bool = True,
        save: SaveTarget | None = None,
        ax: Axes | None = None,
        **kwargs: Any,
) -> tuple[Figure, Axes, Any]:
    th = get_theme(theme)
    fig, ax = setup_axes(kind="2d", theme=th, title=title, xlabel=xlabel, ylabel=ylabel, ax=ax)
    artist = ax.scatter(x, y, c=c, cmap=get_cmap(cmap, get_cmap(th.cmap)) if c is not None else None,
                        color=None if c is not None else th.accent, **kwargs)
    if show_colorbar and c is not None:
        add_colorbar(fig, ax, artist, label=colorbar_label, theme=th)
    _finish(fig, show=show, save=save)
    return fig, ax, artist


def scatter3d(
        x: Sequence[float],
        y: Sequence[float],
        z: Sequence[float],
        *,
        c: Sequence[float] | None = None,
        theme: str | Theme | None = None,
        cmap: str | Colormap | None = None,
        title: str = "3D scatter",
        show_colorbar: bool = True,
        colorbar_label: str = "value",
        show: bool = True,
        save: SaveTarget | None = None,
        ax: Axes | None = None,
        **kwargs: Any,
) -> tuple[Figure, Axes, Any]:
    th = get_theme(theme)
    fig, ax = setup_axes(kind="3d", theme=th, title=title, ax=ax)
    artist = ax.scatter(x, y, z, c=c, cmap=get_cmap(cmap, get_cmap(th.cmap)) if c is not None else None,
                        color=None if c is not None else th.accent, **kwargs)
    if show_colorbar and c is not None:
        add_colorbar(fig, ax, artist, label=colorbar_label, theme=th, shrink=0.65, pad=0.1)
    _finish(fig, show=show, save=save)
    return fig, ax, artist


def quiver(
        X: np.ndarray,
        Y: np.ndarray,
        U: np.ndarray,
        V: np.ndarray,
        *,
        C: np.ndarray | None = None,
        theme: str | Theme | None = None,
        cmap: str | Colormap | None = None,
        title: str = "Quiver",
        xlabel: str = "x",
        ylabel: str = "y",
        show_colorbar: bool = True,
        colorbar_label: str = "value",
        show: bool = True,
        save: SaveTarget | None = None,
        ax: Axes | None = None,
        **kwargs: Any,
) -> tuple[Figure, Axes, Any]:
    th = get_theme(theme)
    fig, ax = setup_axes(kind="2d", theme=th, title=title, xlabel=xlabel, ylabel=ylabel, ax=ax)
    if C is None:
        artist = ax.quiver(X, Y, U, V, color=th.accent, **kwargs)
    else:
        artist = ax.quiver(X, Y, U, V, C, cmap=get_cmap(cmap, get_cmap(th.cmap)), **kwargs)
        if show_colorbar:
            add_colorbar(fig, ax, artist, label=colorbar_label, theme=th)
    ax.set_aspect("equal", adjustable="box")
    _finish(fig, show=show, save=save)
    return fig, ax, artist


def streamplot(
        x: Sequence[float],
        y: Sequence[float],
        U: np.ndarray,
        V: np.ndarray,
        *,
        color: np.ndarray | str | None = None,
        theme: str | Theme | None = None,
        cmap: str | Colormap | None = None,
        title: str = "Streamplot",
        xlabel: str = "x",
        ylabel: str = "y",
        show_colorbar: bool = True,
        colorbar_label: str = "value",
        show: bool = True,
        save: SaveTarget | None = None,
        ax: Axes | None = None,
        **kwargs: Any,
) -> tuple[Figure, Axes, Any]:
    th = get_theme(theme)
    fig, ax = setup_axes(kind="2d", theme=th, title=title, xlabel=xlabel, ylabel=ylabel, ax=ax)
    if color is None:
        color = np.sqrt(U ** 2 + V ** 2)
    artist = ax.streamplot(x, y, U, V, color=color, cmap=get_cmap(cmap, get_cmap(th.cmap)), **kwargs)
    if show_colorbar and not isinstance(color, str):
        add_colorbar(fig, ax, artist.lines, label=colorbar_label, theme=th)
    ax.set_aspect("equal", adjustable="box")
    _finish(fig, show=show, save=save)
    return fig, ax, artist


def histogram(
        data: Sequence[float] | Sequence[Sequence[float]],
        *,
        bins: int | Sequence[float] = 40,
        theme: str | Theme | None = None,
        title: str = "Histogram",
        xlabel: str = "value",
        ylabel: str = "count",
        color: str | Sequence[str] | None = None,
        label: str | Sequence[str] | None = None,
        legend: bool = False,
        stacked: bool = False,
        alpha: float = 0.82,
        show: bool = True,
        save: SaveTarget | None = None,
        ax: Axes | None = None,
        **kwargs: Any,
) -> tuple[Figure, Axes, Any]:
    th = get_theme(theme)
    fig, ax = setup_axes(
        kind="2d",
        theme=th,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        ax=ax,
    )

    is_multi = (
            isinstance(data, (list, tuple))
            and len(data) > 0
            and np.ndim(data[0]) > 0
    )

    if color is None:
        if is_multi:
            n_series = len(data)
            color = [
                th.color_cycle[i % len(th.color_cycle)]
                for i in range(n_series)
            ]
        else:
            color = th.color_cycle[0]

    hist_kwargs = dict(kwargs)
    hist_kwargs.setdefault("color", color)
    hist_kwargs.setdefault("edgecolor", th.foreground)
    hist_kwargs.setdefault("alpha", alpha)
    hist_kwargs.setdefault("label", label)
    hist_kwargs.setdefault("stacked", stacked)

    artist = ax.hist(data, bins=bins, **hist_kwargs)

    if legend or label is not None:
        leg = ax.legend()
        leg.get_frame().set_facecolor(th.background)
        leg.get_frame().set_edgecolor(th.foreground)
        for text in leg.get_texts():
            text.set_color(th.foreground)

    _finish(fig, show=show, save=save)
    return fig, ax, artist

def bar(
        x: Sequence[Any],
        height: Sequence[float],
        *,
        theme: str | Theme | None = None,
        title: str = "Bar chart",
        xlabel: str = "x",
        ylabel: str = "value",
        color: str | Sequence[str] | None = None,
        label: str | None = None,
        legend: bool = False,
        width: float = 0.8,
        xticks: Sequence[float] | None = None,
        xticklabels: Sequence[str] | None = None,
        xticklabel_kwargs: dict[str, Any] | None = None,
        show: bool = True,
        save: SaveTarget | None = None,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        **kwargs: Any,
) -> tuple[Figure, Axes, Any]:
    """
    Plot a bar chart.

    Parameters
    ----------
    x
        Bar locations or category labels.
    height
        Height of each bar.
    color
        Either a single color or one color per bar. If omitted, the theme's
        color cycle is used.
    xticks
        Explicit x-axis tick locations to apply after plotting.
    xticklabels
        Optional tick labels to use with ``xticks``.
    xticklabel_kwargs
        Text properties applied to the x-axis tick labels, for example
        ``{"rotation": 25, "ha": "right"}``.
    """
    th = get_theme(theme)

    fig, ax = setup_axes(
        kind="2d",
        theme=th,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        ax=ax,
        figsize=figsize,
    )

    if color is None:
        color = [
            th.color_cycle[i % len(th.color_cycle)]
            for i in range(len(height))
        ]

    bar_kwargs = dict(kwargs)
    bar_kwargs.setdefault("color", color)
    bar_kwargs.setdefault("edgecolor", th.foreground)
    bar_kwargs.setdefault("linewidth", 1.0)

    artist = ax.bar(
        x,
        height,
        width=width,
        label=label,
        **bar_kwargs,
    )

    if xticks is not None:
        if xticklabels is None:
            ax.set_xticks(xticks)
        else:
            ax.set_xticks(xticks, labels=xticklabels)
    elif xticklabels is not None:
        ax.set_xticklabels(xticklabels)

    if xticklabel_kwargs:
        plt.setp(ax.get_xticklabels(), **xticklabel_kwargs)

    if legend or label is not None:
        leg = ax.legend()
        leg.get_frame().set_facecolor(th.background)
        leg.get_frame().set_edgecolor(th.foreground)
        for text in leg.get_texts():
            text.set_color(th.foreground)

    _finish(fig, show=show, save=save)
    return fig, ax, artist

# -----------------------------------------------------------------------------
# Composable add-ons
# -----------------------------------------------------------------------------

def animate_map(
        *,
        init_plot: Callable[[], tuple[Figure, Axes, Any]],
        update_frame: Callable[[int, Figure, Axes, Any], Sequence[Any] | Any | None],
        frames: int,
        interval: int = 40,
        blit: bool = False,
        repeat: bool = True,
        save: SaveTarget | None = None,
        fps: int = 30,
        dpi: int = 180,
        writer: Literal["ffmpeg", "pillow"] = "ffmpeg",
        show: bool = True,
) -> tuple[Figure, Axes, Any, FuncAnimation]:
    """
    Generic animation wrapper.

    init_plot:
        Creates the initial plot and returns (fig, ax, artist).

    update_frame:
        Called as update_frame(frame_index, fig, ax, artist).
        Should mutate the artist/axis and return changed artists if blitting.

    frames:
        Number of animation frames.
    """
    fig, ax, artist = init_plot()

    current_artist = artist

    def _update(i: int):
        nonlocal current_artist

        changed = update_frame(i, fig, ax, current_artist)

        if changed is not None:
            current_artist = changed

        if changed is None:
            return []
        if isinstance(changed, Sequence):
            return changed
        return [changed]

    anim = FuncAnimation(
        fig,
        _update,
        frames=frames,
        interval=interval,
        blit=blit,
        repeat=repeat,
    )

    if save is not None:
        save_path = Path(save)
        if writer == "pillow" or save_path.suffix.lower() == ".gif":
            anim.save(save_path, writer=PillowWriter(fps=fps), dpi=dpi)
        else:
            anim.save(save_path, writer=FFMpegWriter(fps=fps), dpi=dpi)

    if show:
        plt.show()

    return fig, ax, artist, anim


def add_zero_plane(ax: Axes, X: np.ndarray, Y: np.ndarray, *, alpha: float = 0.12, color: str = "#ffd27a",
                   **kwargs: Any) -> Any:
    """Add a translucent z=0 reference plane to a 3D axis."""
    Z0 = np.zeros_like(X)
    return ax.plot_surface(X, Y, Z0, color=color, alpha=alpha, linewidth=0, shade=False, **kwargs)


def add_wireframe(ax: Axes, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, *, color: str = "#ffd27a", alpha: float = 0.35,
                  **kwargs: Any) -> Any:
    """Overlay a wireframe on an existing 3D surface axis."""
    return ax.plot_wireframe(X, Y, Z, color=color, alpha=alpha, **kwargs)


def camera_orbit_frames(ax: Axes, *, frames: int = 120, elev: float = 32.0, start_azim: float = -180.0,
                        stop_azim: float = 180.0) -> Iterable[None]:
    """Yield camera positions for a simple orbit animation loop."""
    for azim in np.linspace(start_azim, stop_azim, frames):
        ax.view_init(elev=elev, azim=float(azim))  # type: ignore[attr-defined]
        yield None


# -----------------------------------------------------------------------------
# Demo gallery
# -----------------------------------------------------------------------------


def _demo_fields(n: int = 180) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    x = np.linspace(-3, 3, n)
    y = np.linspace(-3, 3, n)
    X, Y = np.meshgrid(x, y)

    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X)

    # Integer multiple of Theta prevents a branch-cut seam:
    # sin(... - k*Theta) is continuous across Theta = +/-pi when k is integer.
    k = 1
    Z = np.exp(-0.18 * R**2) * np.sin(5.5 * R - k * Theta)

    U = -Y / (R + 0.25) + 0.25 * np.sin(2 * X)
    V = X / (R + 0.25) + 0.25 * np.cos(2 * Y)

    return x, y, X, Y, Z, U, V


def demo_animated_heatmap(save: str | Path | None = None):
    x, y, X, Y, Z0, U, V = _demo_fields(n=160)
    R = np.sqrt(X ** 2 + Y ** 2)
    Theta = np.arctan2(Y, X)

    def init_plot():
        return heatmap(
            Z0,
            x=x,
            y=y,
            title="Animated heatmap",
            show_colorbar=True,
            show=False,
        )

    def update_frame(i, fig, ax, artist):
        phase = 2 * np.pi * i / 120
        Z = np.exp(-0.18 * R ** 2) * np.sin(5.5 * R - Theta + phase)

        artist.set_data(Z)
        artist.set_clim(Z.min(), Z.max())
        ax.set_title(f"Animated heatmap | frame {i:03d}")

        return artist

    return animate_map(
        init_plot=init_plot,
        update_frame=update_frame,
        frames=120,
        interval=33,
        save=save,
        show=True,
    )

def demo_animated_surface(save: str | Path | None = None):
    x, y, X, Y, Z, U, V = _demo_fields(n=360)
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X)

    def make_Z(i: int):
        phase = 2 * np.pi * i / 120
        return np.exp(-0.18 * R**2) * np.sin(5.5 * R - Theta + phase)

    def init_plot():
        return surface(
            X,
            Y,
            make_Z(0),
            title="Animated Doom Clock Surface",
            elev=25,
            azim=-35,
            show_colorbar=False,
            antialiased=False,
            show=False,
            save=None
        )

    def update_frame(i, fig, ax, artist):
        artist.remove()

        Z_new = make_Z(i)

        new_artist = ax.plot_surface(
            X,
            Y,
            Z_new,
            cmap=get_cmap(get_theme("space_clock").cmap),
            linewidth=0,
            antialiased=False,
            shade=False,
            rstride=3,
            cstride=3
        )

        ax.set_title(f"Animated Doom Clock Surface | frame {i:03d}")

        return new_artist

    return animate_map(
        init_plot=init_plot,
        update_frame=update_frame,
        frames=120,
        interval=33,
        save=save,
        show=True,
        dpi=600
    )



def main(*, output_dir: str | Path | None = None, show: bool = True) -> list[Path]:
    """Generate a small demo gallery showing the plotting helpers."""
    paths: list[Path] = []
    out = Path(output_dir) if output_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)

    def save_path(name: str) -> Path | None:
        if out is None:
            return None
        path = out / name
        paths.append(path)
        return path

    x, y, X, Y, Z, U, V = _demo_fields(n=160)

    surface(
        X,
        Y,
        Z,
        title="Doom Clock Surface",
        style="interactive",
        elev=25,
        azim=-35,
        show_colorbar=True,
        antialiased=False,
        show=show,
        save=save_path("01_surface.png"),
    )

    heatmap(
        Z,
        x=x,
        y=y,
        title="Heatmap",
        show=show,
        save=save_path("02_heatmap.png"),
    )

    contour(
        X,
        Y,
        Z,
        title="Filled contour",
        levels=32,
        show=show,
        save=save_path("03_contour.png"),
    )

    t = np.linspace(0, 8 * np.pi, 800)
    line(
        t,
        np.sin(t) * np.exp(-0.03 * t),
        title="Line plot",
        xlabel="time",
        ylabel="amplitude",
        linewidth=2.0,
        show=show,
        save=save_path("04_line.png"),
    )

    rng = np.random.default_rng(7)
    xs = rng.normal(size=500)
    ys = 0.55 * xs + rng.normal(scale=0.55, size=500)
    scatter(
        xs,
        ys,
        c=np.hypot(xs, ys),
        title="Scatter",
        s=18,
        alpha=0.85,
        show=show,
        save=save_path("05_scatter.png"),
    )

    scatter3d(
        np.cos(t[::8]) * (1 + 0.06 * t[::8]),
        np.sin(t[::8]) * (1 + 0.06 * t[::8]),
        0.08 * t[::8],
        c=t[::8],
        title="3D scatter spiral",
        s=18,
        show=show,
        save=save_path("06_scatter3d.png"),
    )

    step = 8
    speed = np.sqrt(U[::step, ::step] ** 2 + V[::step, ::step] ** 2)
    quiver(
        X[::step, ::step],
        Y[::step, ::step],
        U[::step, ::step],
        V[::step, ::step],
        C=speed,
        title="Quiver vector field",
        scale=35,
        width=0.004,
        show=show,
        save=save_path("07_quiver.png"),
    )

    streamplot(
        x,
        y,
        U,
        V,
        title="Streamplot vector field",
        density=1.5,
        linewidth=1.1,
        arrowsize=0.8,
        show=show,
        save=save_path("08_streamplot.png"),
    )

    histogram(
        [rng.normal(loc=0.0, scale=1.0, size=2_000), rng.normal(loc=0.0, scale=1.0, size=2_000)],
        title="Histogram",
        bins=45,
        show=show,
        save=save_path("09_histogram.png"),
    )

    bar(
        ["PnP", "QnP", "FG", "Truth"],
        [1.28, 0.74, 0.51, 0.42],
        title="Bar chart",
        ylabel="RMSE [m]",
        show=show,
        save=save_path("10_bar.png"),
    )

    fig, ax, surf = surface(
        X,
        Y,
        Z,
        title="Composable helpers",
        style="interactive",
        show_colorbar=False,
        show=False,
        save=None,
    )
    add_zero_plane(ax, X[::4, ::4], Y[::4, ::4])
    add_wireframe(ax, X[::8, ::8], Y[::8, ::8], Z[::8, ::8], rstride=1, cstride=1)
    final_save = save_path("11_composable_surface.png")
    _finish(fig, show=show, save=final_save)

    demo_animated_surface("animate_surface.gif")
    demo_animated_heatmap("animated_heatmap.gif")

    return paths


if __name__ == "__main__":
    main(output_dir=Path(__file__).with_name("plot_demos"), show=True)
