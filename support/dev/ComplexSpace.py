"""Utilities for visualizing mappings on the complex plane."""

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib import colors
from matplotlib.animation import FuncAnimation

ANIMATION = None


class InterpolationMode(str, Enum):
    """Supported interpolation spaces for morphing line samples."""

    CARTESIAN = "cartesian"
    POLAR = "polar"


class AngleDirection(str, Enum):
    """Supported angular interpolation paths in polar coordinates."""

    SHORTEST = "shortest"
    COUNTERCLOCKWISE = "counterclockwise"


@dataclass(frozen=True)
class GridLine:
    """Sampled source and target geometry for one grid line."""

    source: np.ndarray
    target: np.ndarray
    valid: np.ndarray
    color: tuple
    label: str


@dataclass(frozen=True)
class ComplexMapping:
    """A complex-valued mapping paired with a display label."""

    func: Callable[[np.ndarray], np.ndarray]
    label: str


def make_complex_function():
    """Return the example mapping used by the module's script entrypoint."""

    return ComplexMapping(func=lambda z: z ** 2, label="z^2")


def ease_in_out(t):
    """Smoothly interpolate between 0 and 1 with zero slope at the ends."""

    return 0.5 - 0.5 * np.cos(np.pi * t)


def _coerce_interpolation_mode(mode):
    """Normalize interpolation mode inputs to an enum value."""

    if isinstance(mode, InterpolationMode):
        return mode
    return InterpolationMode(mode)


def _coerce_angle_direction(direction):
    """Normalize angle direction inputs to an enum value."""

    if isinstance(direction, AngleDirection):
        return direction
    return AngleDirection(direction)


def _coerce_mapping(mapping):
    """Normalize mapping inputs to a labeled ComplexMapping."""

    if isinstance(mapping, ComplexMapping):
        return mapping
    return ComplexMapping(
        func=mapping,
        label=getattr(mapping, "__name__", mapping.__class__.__name__),
    )


def interpolate_complex_points(
    source_xy,
    target_xy,
    blend,
    mode=InterpolationMode.POLAR,
    angle_direction=AngleDirection.SHORTEST,
):
    """Interpolate sampled points between input and output complex geometry."""

    mode = _coerce_interpolation_mode(mode)
    angle_direction = _coerce_angle_direction(angle_direction)

    if mode is InterpolationMode.CARTESIAN:
        return (1.0 - blend) * source_xy + blend * target_xy

    source_z = source_xy[:, 0] + 1j * source_xy[:, 1]
    target_z = target_xy[:, 0] + 1j * target_xy[:, 1]

    source_r = np.abs(source_z)
    target_r = np.abs(target_z)
    source_theta = np.angle(source_z)
    target_theta = np.angle(target_z)

    raw_delta_theta = target_theta - source_theta
    if angle_direction is AngleDirection.COUNTERCLOCKWISE:
        delta_theta = raw_delta_theta % (2 * np.pi)
    else:
        delta_theta = (raw_delta_theta + np.pi) % (2 * np.pi) - np.pi
    interp_theta = source_theta + blend * delta_theta

    interp_r = np.empty_like(source_r)
    positive_mask = (source_r > 0.0) & (target_r > 0.0)
    interp_r[positive_mask] = np.exp(
        (1.0 - blend) * np.log(source_r[positive_mask])
        + blend * np.log(target_r[positive_mask])
    )
    interp_r[~positive_mask] = (
        (1.0 - blend) * source_r[~positive_mask] + blend * target_r[~positive_mask]
    )

    interp_z = interp_r * np.exp(1j * interp_theta)
    coords = np.column_stack((interp_z.real, interp_z.imag))

    zero_zero_mask = (source_r == 0.0) & (target_r == 0.0)
    if np.any(zero_zero_mask):
        coords[zero_zero_mask] = 0.0

    return coords


def build_grid_lines(mapping, xlim, ylim, grid_count, samples_per_line):
    """Sample a rectangular input grid and map it through a complex function."""

    mapping = _coerce_mapping(mapping)

    x_values = np.linspace(xlim[0], xlim[1], grid_count)
    y_values = np.linspace(ylim[0], ylim[1], grid_count)
    x_line = np.linspace(xlim[0], xlim[1], samples_per_line)
    y_line = np.linspace(ylim[0], ylim[1], samples_per_line)

    x_norm = colors.Normalize(vmin=xlim[0], vmax=xlim[1])
    y_norm = colors.Normalize(vmin=ylim[0], vmax=ylim[1])
    x_cmap = colormaps["viridis"]
    y_cmap = colormaps["plasma"]

    lines = []
    output_points = []

    for x0 in x_values:
        z_line = x0 + 1j * y_line
        w_line = mapping.func(z_line)
        valid = np.isfinite(w_line.real) & np.isfinite(w_line.imag)
        source_xy = np.column_stack((np.full_like(y_line, x0), y_line))
        target_xy = np.column_stack((w_line.real, w_line.imag))
        lines.append(
            GridLine(
                source=source_xy,
                target=target_xy,
                valid=valid,
                color=x_cmap(x_norm(x0)),
                label=f"Re(z)={x0:.2f}",
            )
        )
        output_points.append(target_xy[valid])

    for y0 in y_values:
        z_line = x_line + 1j * y0
        w_line = mapping.func(z_line)
        valid = np.isfinite(w_line.real) & np.isfinite(w_line.imag)
        source_xy = np.column_stack((x_line, np.full_like(x_line, y0)))
        target_xy = np.column_stack((w_line.real, w_line.imag))
        lines.append(
            GridLine(
                source=source_xy,
                target=target_xy,
                valid=valid,
                color=y_cmap(y_norm(y0)),
                label=f"Im(z)={y0:.2f}",
            )
        )
        output_points.append(target_xy[valid])

    finite_segments = [pts for pts in output_points if pts.size > 0]
    finite_output = (
        np.concatenate(finite_segments)
        if finite_segments
        else np.empty((0, 2), dtype=float)
    )
    return lines, finite_output, x_norm, y_norm, x_cmap, y_cmap


def animate_complex_mapping(
    mapping,
    xlim=(-3.0, 3.0),
    ylim=(-3.0, 3.0),
    grid_count=13,
    samples_per_line=1200,
    frames=250,
    interval_ms=20,
    interpolation_mode=InterpolationMode.POLAR,
    angle_direction=AngleDirection.SHORTEST,
    show=True,
):
    """Animate a complex-plane grid morphing under a supplied mapping."""

    interpolation_mode = _coerce_interpolation_mode(interpolation_mode)
    angle_direction = _coerce_angle_direction(angle_direction)
    mapping = _coerce_mapping(mapping)

    xlim = np.asarray(xlim, dtype=float)
    ylim = np.asarray(ylim, dtype=float)

    lines, finite_output, x_norm, y_norm, x_cmap, y_cmap = build_grid_lines(
        mapping=mapping,
        xlim=2 * xlim,
        ylim=2 * ylim,
        grid_count=grid_count,
        samples_per_line=samples_per_line,
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    artists = []

    input_xlim = np.array(xlim, dtype=float)
    input_ylim = np.array(ylim, dtype=float)

    if finite_output.size > 0:
        output_xlim = np.percentile(finite_output[:, 0], [1, 99])
        output_ylim = np.percentile(finite_output[:, 1], [1, 99])
    else:
        output_xlim = input_xlim.copy()
        output_ylim = input_ylim.copy()

    for line_data in lines:
        artist, = ax.plot(
            [],
            [],
            color=line_data.color,
            alpha=0.85,
            linewidth=1.2,
        )
        artists.append(artist)

    legend_handles = [
        plt.Line2D([0], [0], color=x_cmap(0.8), label="Re(z) = constant"),
        plt.Line2D([0], [0], color=y_cmap(0.8), label="Im(z) = constant"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False)

    sm_x = plt.cm.ScalarMappable(norm=x_norm, cmap=x_cmap)
    sm_y = plt.cm.ScalarMappable(norm=y_norm, cmap=y_cmap)
    sm_x.set_array([])
    sm_y.set_array([])
    fig.colorbar(sm_x, ax=ax, pad=0.02, fraction=0.046, label="Re(z) line value")
    fig.colorbar(sm_y, ax=ax, pad=0.10, fraction=0.046, label="Im(z) line value")

    def update(frame_idx):
        cycle_position = frame_idx / (frames - 1)
        if cycle_position <= 0.5:
            t = cycle_position * 2.0
        else:
            t = (1.0 - cycle_position) * 2.0

        blend = ease_in_out(t)

        for artist, line_data in zip(artists, lines):
            coords = interpolate_complex_points(
                line_data.source,
                line_data.target,
                blend,
                mode=interpolation_mode,
                angle_direction=angle_direction,
            )
            valid = line_data.valid
            artist.set_data(coords[valid, 0], coords[valid, 1])

        current_xlim = (1.0 - blend) * input_xlim + blend * output_xlim
        current_ylim = (1.0 - blend) * input_ylim + blend * output_ylim
        ax.set_xlim(current_xlim)
        ax.set_ylim(current_ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("Real")
        ax.set_ylabel("Imaginary")
        ax.set_title(
            "Complex Mapping Warp "
            f"({interpolation_mode.value}, {angle_direction.value}) "
            f"for f(z) = {mapping.label}   t={blend:0.2f}"
        )

        if blend < 0.5:
            ax.set_xlabel("Re(z) -> Re(f(z))")
            ax.set_ylabel("Im(z) -> Im(f(z))")
        else:
            ax.set_xlabel("Re(f(z))")
            ax.set_ylabel("Im(f(z))")

        return artists

    anim = FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=interval_ms,
        blit=False,
        repeat=True,
    )

    # th_circ = np.linspace(0, 2 * np.pi, 100)
    # x_circ = np.cos(th_circ)
    # y_circ = np.sin(th_circ)
    # ax.plot(x_circ, y_circ, color="red", label="Re(z) = constant")

    fig.suptitle("Animated Complex Grid Warp", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if show:
        plt.show()
    else:
        anim._draw_was_started = True
        plt.close(fig)
    return anim


if __name__ == "__main__":
    ANIMATION = animate_complex_mapping(make_complex_function())
