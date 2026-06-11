from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.lines import Line2D

LENGTH = 4.0
HEIGHT = 2.0
TAPER_SCALE = 0.95

N_FRAMES = 180
FRAME_INTERVAL_MS = 40

NUM_ITERATIONS = 5
NUM_STATES = 3

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Rectangle vertex order: lower-right, lower-left, upper-left, upper-right.
# Keeping a consistent order is what makes the red correspondence lines meaningful.
RECT_VERTS = np.array([
    [ LENGTH / 2.0, -HEIGHT / 2.0],
    [-LENGTH / 2.0, -HEIGHT / 2.0],
    [-LENGTH / 2.0,  HEIGHT / 2.0],
    [ LENGTH / 2.0,  HEIGHT / 2.0],
], dtype=np.float64)

# A trapezoid that is close to the rectangle, but not exactly the same shape.
# The optimizer can only use SE(2), so the final residual will generally not be zero.
TRAPEZOID_VERTS = np.array([
    [ LENGTH / (2.0 * TAPER_SCALE),
      -HEIGHT * TAPER_SCALE / 2.0],

    [-LENGTH / (2.0 * TAPER_SCALE),
     -HEIGHT * TAPER_SCALE / 2.0],

    [-LENGTH * TAPER_SCALE / 2.0,
     HEIGHT * TAPER_SCALE / 2.0],

    [ LENGTH * TAPER_SCALE / 2.0,
      HEIGHT * TAPER_SCALE / 2.0],
], dtype=np.float64)

EDGES = np.array([
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
])

NUM_VERTS = len(RECT_VERTS)
NUM_ELEMENTS = int(np.prod(RECT_VERTS.shape))


def wrap_angle(theta: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class SE2:
    """A tiny SE(2) class for 2D rigid transforms.

    The transform maps model points into world points as

        p_world = R(theta) @ p_model + t.

    Composition is left-to-right in the same style as the 3D demo:

        world_point = pose * model_point
        combined_pose = left_pose * right_pose
    """

    theta: float
    tvec: np.ndarray

    def __post_init__(self) -> None:
        self.theta = float(wrap_angle(self.theta))
        self.tvec = np.asarray(self.tvec, dtype=np.float64).reshape(2)

    @property
    def R(self) -> np.ndarray:
        c = np.cos(self.theta)
        s = np.sin(self.theta)
        return np.array([[c, -s],
                         [s,  c]], dtype=np.float64)

    @classmethod
    def identity(cls) -> SE2:
        return cls(0.0, np.zeros(2, dtype=np.float64))

    @classmethod
    def random(cls, max_translation: float = 5.0, max_angle_deg: float = 180.0) -> SE2:
        theta = np.deg2rad(np.random.uniform(-max_angle_deg, max_angle_deg))
        tvec = np.random.uniform(-max_translation, max_translation, size=2)
        return cls(theta, tvec)

    @classmethod
    def exp(cls, dx: np.ndarray) -> SE2:
        """Small left perturbation used by the Gauss-Newton update.

        This intentionally uses the first-order-friendly parameterization
        dx = [dtheta, dtx, dty]. For this visualization, that keeps the
        Jacobian easy to inspect and mirrors the lightweight style of the demo.
        """
        dx = np.asarray(dx, dtype=np.float64).reshape(3)
        return cls(dx[0], dx[1:3])

    def copy(self) -> SE2:
        return SE2(self.theta, self.tvec.copy())

    def inverse(self) -> SE2:
        R_T = self.R.T
        return SE2(-self.theta, -(R_T @ self.tvec))

    @property
    def inv(self) -> SE2:
        """Return the inverse transform. Named to match the SE(3) demo style."""
        return self.inverse()

    def interpolate(self, other: SE2, alpha: float) -> SE2:
        alpha = float(np.clip(alpha, 0.0, 1.0))
        dtheta = wrap_angle(other.theta - self.theta)
        theta = wrap_angle(self.theta + alpha * dtheta)
        tvec = (1.0 - alpha) * self.tvec + alpha * other.tvec
        return SE2(theta, tvec)

    def __mul__(self, other):
        if isinstance(other, SE2):
            theta = wrap_angle(self.theta + other.theta)
            tvec = self.R @ other.tvec + self.tvec
            return SE2(theta, tvec)

        points = np.asarray(other, dtype=np.float64)
        if points.shape == (2,):
            return self.R @ points + self.tvec
        if points.ndim == 2 and points.shape[1] == 2:
            return points @ self.R.T + self.tvec
        raise TypeError(f"SE2 can transform a 2-vector, an Nx2 array, or compose with SE2; got shape {points.shape}")

    def __repr__(self) -> str:
        return f"SE2(theta_deg={np.rad2deg(self.theta): .3f}, tvec=[{self.tvec[0]: .3f}, {self.tvec[1]: .3f}])"


MEAS_SE2 = SE2.random(max_translation=4.0, max_angle_deg=70.0)
MEAS_VERTS = MEAS_SE2 * RECT_VERTS


def factor_graph() -> tuple[list[tuple[SE2, float]], Callable[[SE2 | None], np.ndarray]]:
    """Fit the trapezoid pose to the measured rectangle correspondences.

    Residual for vertex i:

        y_i = measured_i - estimated_pose * trapezoid_i

    The state is a left perturbation dx = [dtheta, dtx, dty]. For the current
    world point p = estimated_pose * trapezoid_i,

        d y_i / d dtheta = -J p = [p_y, -p_x]^T
        d y_i / d dt     = -I

    where J is the 2D generator [[0, -1], [1, 0]].
    """
    initial_perturb = SE2.random(max_translation=3.0, max_angle_deg=90.0)
    est_SE2 = initial_perturb * MEAS_SE2

    stored_SE2: list[tuple[SE2, float]] = []

    def create_y(SE2_input: SE2 | None = None) -> np.ndarray:
        y = np.zeros(NUM_ELEMENTS, dtype=np.float64)

        if SE2_input is None:
            SE2_input = est_SE2

        est_verts = SE2_input * TRAPEZOID_VERTS
        for idx, (m_vert, s_vert) in enumerate(zip(MEAS_VERTS, est_verts)):
            y[2 * idx:2 * idx + 2] = m_vert - s_vert
        return y

    def create_L() -> np.ndarray:
        L = np.zeros((NUM_ELEMENTS, NUM_STATES), dtype=np.float64)
        est_verts = est_SE2 * TRAPEZOID_VERTS
        for idx, p_world in enumerate(est_verts):
            x, y = p_world
            L[2 * idx:2 * idx + 2, 0] = [y, -x]
            L[2 * idx:2 * idx + 2, 1:3] = -np.eye(2)
        return L
    
    y = create_y()
    L = create_L()
    y_mag = float(y.T @ y)
    stored_SE2.append((est_SE2.copy(), y_mag))

    for _ in range(NUM_ITERATIONS):
        dx = -np.linalg.pinv(L) @ y
        pert_SE2 = SE2.exp(dx)
        est_SE2 = pert_SE2 * est_SE2

        y = create_y()
        L = create_L()
        y_mag = float(y.T @ y)
        stored_SE2.append((est_SE2.copy(), y_mag))

    return stored_SE2, create_y


def compute_axis_length(verts: np.ndarray) -> float:
    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    return 0.25 * float(np.min(np.maximum(maxs - mins, 1.0e-6)))


def create_object_artists(ax, verts, edges, pose: SE2, color: str, alpha: float = 1.0,
                          show_axes: bool = True, point_size: int = 45) -> dict:
    world_verts = pose * verts
    translation = pose.tvec
    axis_length = compute_axis_length(world_verts)
    local_basis = np.eye(2, dtype=np.float64) * axis_length
    world_basis = pose * local_basis

    scatter = ax.scatter(world_verts[:, 0], world_verts[:, 1], color=color, s=point_size, alpha=alpha, zorder=3)

    axis_colors = ("r", "g")
    axis_labels = ("X", "Y")
    axis_lines: list[Line2D] = []
    axis_texts = []
    if show_axes:
        for axis_vec, axis_color, axis_label in zip(world_basis, axis_colors, axis_labels):
            line, = ax.plot([translation[0], axis_vec[0]],
                            [translation[1], axis_vec[1]],
                            color=axis_color,
                            linewidth=2,
                            alpha=alpha,
                            zorder=4)
            text = ax.text(axis_vec[0],
                           axis_vec[1],
                           axis_label,
                           color=axis_color,
                           alpha=alpha,
                           ha="center",
                           va="center",
                           zorder=5)
            axis_lines.append(line)
            axis_texts.append(text)

    edge_lines: list[Line2D] = []
    for start_idx, end_idx in edges:
        edge = world_verts[[start_idx, end_idx]]
        line, = ax.plot(edge[:, 0], edge[:, 1], color=color, alpha=alpha, linewidth=2, zorder=2)
        edge_lines.append(line)

    return {
        "ax": ax,
        "verts_model": verts,
        "edges": edges,
        "scatter": scatter,
        "axis_lines": axis_lines,
        "axis_texts": axis_texts,
        "edge_lines": edge_lines,
    }


def update_object_artists(artists: dict, pose: SE2) -> np.ndarray:
    world_verts = pose * artists["verts_model"]
    translation = pose.tvec
    axis_length = compute_axis_length(world_verts)
    local_basis = np.eye(2, dtype=np.float64) * axis_length
    world_basis = pose * local_basis

    artists["scatter"].set_offsets(world_verts)

    for axis_line, axis_text, axis_vec in zip(artists["axis_lines"], artists["axis_texts"], world_basis):
        axis_line.set_data([translation[0], axis_vec[0]], [translation[1], axis_vec[1]])
        axis_text.set_position((axis_vec[0], axis_vec[1]))

    for edge_line, (start_idx, end_idx) in zip(artists["edge_lines"], artists["edges"]):
        edge = world_verts[[start_idx, end_idx]]
        edge_line.set_data(edge[:, 0], edge[:, 1])

    return world_verts


def create_connection_lines(ax, verts_a: np.ndarray, verts_b: np.ndarray) -> list[Line2D]:
    lines: list[Line2D] = []
    for vert_a, vert_b in zip(verts_a, verts_b):
        line, = ax.plot([vert_a[0], vert_b[0]],
                        [vert_a[1], vert_b[1]],
                        color="darkred",
                        linewidth=1.5,
                        alpha=0.75,
                        zorder=1)
        lines.append(line)
    return lines


def update_connection_lines(lines: list[Line2D], verts_a: np.ndarray, verts_b: np.ndarray) -> None:
    for line, vert_a, vert_b in zip(lines, verts_a, verts_b):
        line.set_data([vert_a[0], vert_b[0]], [vert_a[1], vert_b[1]])


def set_artist_group_visible(artists: dict, visible: bool) -> None:
    artists["scatter"].set_visible(visible)
    for line in artists["axis_lines"]:
        line.set_visible(visible)
    for text in artists["axis_texts"]:
        text.set_visible(visible)
    for line in artists["edge_lines"]:
        line.set_visible(visible)


def set_axes_equal(ax, mins: np.ndarray, maxs: np.ndarray) -> None:
    center = (mins + maxs) / 2.0
    radius = 0.55 * float(np.max(maxs - mins))
    radius = max(radius, 1.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_aspect("equal", adjustable="box")


def frame_to_alpha(frame_idx: int) -> tuple[int, int, float]:
    phase_length = max(int(N_FRAMES * 0.9 / NUM_ITERATIONS), 1)
    pos = frame_idx / phase_length
    first_idx = int(pos)
    second_idx = first_idx + 1
    alpha = pos - first_idx

    if second_idx > NUM_ITERATIONS:
        first_idx = NUM_ITERATIONS - 1
        second_idx = NUM_ITERATIONS
        alpha = 1.0

    return first_idx, second_idx, alpha


def save_demo_snapshot(fig: plt.Figure, stem_suffix: str = "_final") -> Path:
    output_path = Path(__file__).with_name(f"{Path(__file__).stem}{stem_suffix}.pdf")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return output_path


def save_demo_video(anim: FuncAnimation) -> Path:
    output_path = Path(__file__).with_suffix(".mp4")
    fps = max(1, round(1000.0 / FRAME_INTERVAL_MS))
    anim.save(output_path, writer=FFMpegWriter(fps=fps, bitrate=4000), dpi=150)
    return output_path


def main() -> None:
    fig, ax = plt.subplots()
    fig.subplots_adjust(right=0.72)

    stored_SE2, create_y_func = factor_graph()
    _, opt_res_mag = stored_SE2[-1]

    meas_pose_0 = MEAS_SE2
    state_pose_0, _ = stored_SE2[0]

    meas_artists = create_object_artists(ax, RECT_VERTS, EDGES, meas_pose_0, "tab:blue")
    state_artists = create_object_artists(ax, TRAPEZOID_VERTS, EDGES, state_pose_0, "tab:orange")

    meas_verts_0 = meas_pose_0 * RECT_VERTS
    state_verts_0 = state_pose_0 * TRAPEZOID_VERTS
    connection_lines = create_connection_lines(ax, meas_verts_0, state_verts_0)

    all_initial_verts = np.vstack([meas_verts_0, state_verts_0])
    mins = all_initial_verts.min(axis=0)
    maxs = all_initial_verts.max(axis=0)

    for pose, _ in stored_SE2:
        ghost_verts = pose * TRAPEZOID_VERTS
        mins = np.minimum(mins, ghost_verts.min(axis=0))
        maxs = np.maximum(maxs, ghost_verts.max(axis=0))

    set_axes_equal(ax, mins, maxs)

    status_text = ax.text(
        1.02,
        0.98,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        family="monospace",
        clip_on=False,
    )

    ghost_artists = []
    ghost_labels = []
    for ghost_idx, (ghost_SE2, _) in enumerate(stored_SE2):
        ghost_artist = create_object_artists(
            ax,
            TRAPEZOID_VERTS,
            EDGES,
            ghost_SE2,
            "tab:orange",
            alpha=0.18,
            show_axes=False,
            point_size=25,
        )
        set_artist_group_visible(ghost_artist, False)
        ghost_artists.append(ghost_artist)

        ghost_verts = ghost_SE2 * TRAPEZOID_VERTS
        label_pos = ghost_verts[0]
        label_text = "Start" if ghost_idx == 0 else f"Iteration {ghost_idx}"
        ghost_label = ax.text(
            label_pos[0],
            label_pos[1],
            label_text,
            color="tab:orange",
            alpha=0.55,
        )
        ghost_label.set_visible(False)
        ghost_labels.append(ghost_label)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("2D Object Alignment Demo: Fit a Trapezoid to a Rectangle")
    ax.grid(True, alpha=0.25)

    legend_handles = [
        Line2D([0], [0], color="tab:blue", lw=2, marker="o", label="Measured rectangle"),
        Line2D([0], [0], color="tab:orange", lw=2, marker="o", label="Estimated trapezoid"),
        Line2D([0], [0], color="darkred", lw=1.5, label="Correspondence residuals"),
    ]
    ax.legend(handles=legend_handles, loc="lower left")

    def update(frame_idx: int):
        first_idx, second_idx, alpha = frame_to_alpha(frame_idx)
        first_SE2, first_res = stored_SE2[first_idx]
        second_SE2, second_res = stored_SE2[second_idx]

        state_pose = first_SE2.interpolate(second_SE2, alpha)
        y = create_y_func(state_pose)
        residual = float(y.T @ y)
        error_SE2 = state_pose.inv * MEAS_SE2
        angle_error_deg = abs(np.rad2deg(wrap_angle(error_SE2.theta)))
        trans_error = float(np.linalg.norm(error_SE2.tvec))

        for ghost_idx, (ghost_artist, ghost_label) in enumerate(zip(ghost_artists, ghost_labels)):
            visible = ghost_idx <= first_idx
            set_artist_group_visible(ghost_artist, visible)
            ghost_label.set_visible(visible)

        meas_verts = update_object_artists(meas_artists, MEAS_SE2)
        state_verts = update_object_artists(state_artists, state_pose)
        update_connection_lines(connection_lines, meas_verts, state_verts)

        status_text.set_text(
            f"           Estimated | Solution\n"
            f"Theta:       {np.rad2deg(state_pose.theta):8.3f}|{np.rad2deg(meas_pose_0.theta):8.3f} deg\n"
            f"t_x:         {state_pose.tvec[0]:8.3f}|{meas_pose_0.tvec[0]:8.3f}\n"
            f"t_y:         {state_pose.tvec[1]:8.3f}|{meas_pose_0.tvec[1]:8.3f}\n\n"
            f"Iteration:   {second_idx:8d}\n"
            f"Alpha:       {alpha:8.3f}\n\n"
            f"Residual:    {residual:8.3f}|{opt_res_mag:8.3f}\n"
            f"Angle-Error: {angle_error_deg:8.3f} deg\n"
            f"Trans-Error: {trans_error:8.3f}"
        )

        return [
            meas_artists["scatter"],
            state_artists["scatter"],
            status_text,
            *meas_artists["axis_lines"],
            *state_artists["axis_lines"],
            *meas_artists["axis_texts"],
            *state_artists["axis_texts"],
            *meas_artists["edge_lines"],
            *state_artists["edge_lines"],
            *connection_lines,
            *(artist for ghost_artist in ghost_artists for artist in [ghost_artist["scatter"], *ghost_artist["edge_lines"]]),
            *ghost_labels,
        ]

    update(0)
    fig.canvas.draw()
    snapshot_path = save_demo_snapshot(fig, "_initial")
    print(f"Saved initial snapshot to: {snapshot_path}")

    update(N_FRAMES - 1)
    fig.canvas.draw()
    snapshot_path = save_demo_snapshot(fig)
    print(f"Saved final snapshot to: {snapshot_path}")

    anim = FuncAnimation(fig, update, frames=N_FRAMES, interval=FRAME_INTERVAL_MS, blit=False, repeat=True)
    fig._object_alignment_anim = anim

    video_path = save_demo_video(anim)
    print(f"Saved video to: {video_path}")

    plt.show()


if __name__ == "__main__":
    main()
