from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import proj3d

from support.mathHelpers.LevMarq import LevenbergMarquardt
from support.mathHelpers.SE2PointAlignmentProblem import SE2PointAlignmentProblem
from support.mathHelpers.SE2 import SE2

LENGTH = 4.0
HEIGHT = 2.0
TAPER_SCALE = 0.95

N_FRAMES = 180
FRAME_INTERVAL_MS = 40
MESH_GRID_SIZE = 90
MESH_ELEV_DEG = 34.0
MESH_AZIM_START_DEG = -95.0
MESH_AZIM_END_DEG = -135.0
MESH_Z_MIN = 0.0
MESH_Z_MAX = 30.0
MESH_LABEL_MIN_THETA_DEG = 6.0
MESH_LABEL_MIN_RADIUS = 0.08
MESH_LABEL_MIN_RESIDUAL = 0.35
MESH_LABEL_Z_OFFSET = 0.6

NUM_ITERATIONS = 15
NUM_STATES = 3

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Rectangle vertex order: lower-right, lower-left, upper-left, upper-right.
# Keeping a consistent order is what makes the red correspondence lines meaningful.
RECT_VERTS = np.array([
    [LENGTH / 2.0, -HEIGHT / 2.0],
    [-LENGTH / 2.0, -HEIGHT / 2.0],
    [-LENGTH / 2.0, HEIGHT / 2.0],
    [LENGTH / 2.0, HEIGHT / 2.0],
], dtype=np.float64)

# A trapezoid that is close to the rectangle, but not exactly the same shape.
# The optimizer can only use SE(2), so the final residual will generally not be zero.
TRAPEZOID_VERTS = np.array([
    [LENGTH / (2.0 * TAPER_SCALE),
     -HEIGHT * TAPER_SCALE / 2.0],

    [-LENGTH / (2.0 * TAPER_SCALE),
     -HEIGHT * TAPER_SCALE / 2.0],

    [-LENGTH * TAPER_SCALE / 2.0,
     HEIGHT * TAPER_SCALE / 2.0],

    [LENGTH * TAPER_SCALE / 2.0,
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


def wrap_angle_signed(theta_rad: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (theta_rad + np.pi) % (2.0 * np.pi) - np.pi


def wrap_angle_2pi(theta_rad: float) -> float:
    """Wrap an angle to [0, 2pi)."""
    return theta_rad % (2.0 * np.pi)


MEAS_SE2 = SE2.random(max_translation=4.0, max_angle_deg=0.0)
MEAS_VERTS = MEAS_SE2 * RECT_VERTS

def factor_graph() -> tuple[list[tuple[SE2, float]], Callable[[SE2 | None], np.ndarray]]:
    """Fit the trapezoid pose to the measured rectangle correspondences.

    This version delegates the nonlinear least-squares solve to the generic
    LevenbergMarquardt optimizer. The object-alignment problem only provides:

        residual(state)
        jacobian(state)
        retract(state, dx)

    The optimizer no longer knows anything special about SE(2).
    """

    initial_perturb = SE2.random(max_translation=3.0, max_angle_deg=0.0)

    # Keep the intentionally difficult near-180-degree initial condition.
    EPSILON = np.deg2rad(0.1)
    initial_perturb = SE2(np.pi - EPSILON, initial_perturb.tvec)

    est_SE2 = initial_perturb * MEAS_SE2

    problem = SE2PointAlignmentProblem(
        model_verts=TRAPEZOID_VERTS,
        measured_verts=MEAS_VERTS,
    )

    solver = LevenbergMarquardt(
        state=est_SE2,
        problem=problem,
        damping_enabled=True,
        damping=1e1,
        adaptive=True,
        damping_up=10.0,
        damping_down=0.3,
        min_damping=1e-10,
        use_diagonal_damping=False,
        tolerance=1e-9,
        max_steps=NUM_ITERATIONS,
        max_iter=10,
        accept_rho_min=1.0e-3,
        good_rho_min=0.75,
        bad_rho_max=0.25,
        numerical_check=False,
        store_y_mags=True,
        store_states=True,
    )

    stored_SE2 = [
        (pose.copy(), residual_mag)
        for pose, residual_mag in zip(solver.states_hist, solver.y_mag_hist)
    ]
    target_len = NUM_ITERATIONS + 1
    if stored_SE2:
        final_pose, final_residual_mag = stored_SE2[-1]
        while len(stored_SE2) < target_len:
            stored_SE2.append((final_pose.copy(), float(final_residual_mag)))

    def create_y(SE2_input: SE2 | None = None) -> np.ndarray:
        if SE2_input is None:
            SE2_input = solver.state
        return problem.residual(SE2_input)

    return stored_SE2, create_y


def create_mesh_grid(
    stored_SE2: list[tuple[SE2, float]],
    create_y_func: Callable[[SE2 | None], np.ndarray],
    ref_pose: SE2 | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a residual surface over theta and translation distance from the final pose.

    The translation dimension is now:

        ||t - t_ref||

    where t_ref is the final optimized translation. This fixes the previous
    behavior where the surface used:

        | ||t|| - ||t_ref|| |

    which could be near zero even when the current translation was visibly far
    from the solution.
    """
    if ref_pose is None:
        ref_pose = stored_SE2[-1][0]

    ref_tvec = np.asarray(ref_pose.tvec, dtype=np.float64).reshape(2)

    # Choose a fixed translation-slice direction. Use the direction from the
    # final pose back toward the starting pose, because that makes the surface
    # slice pass through the visually important initial condition.
    start_tvec = np.asarray(stored_SE2[0][0].tvec, dtype=np.float64).reshape(2)
    start_delta = start_tvec - ref_tvec
    start_delta_norm = float(np.linalg.norm(start_delta))

    if start_delta_norm < 1.0e-9:
        # Fallback: use the largest stored translation displacement from the
        # final pose. If every stored pose has the same translation, use +x.
        deltas = np.array([
            np.asarray(pose.tvec, dtype=np.float64).reshape(2) - ref_tvec
            for pose, _ in stored_SE2
        ])
        delta_norms = np.linalg.norm(deltas, axis=1)
        max_idx = int(np.argmax(delta_norms))

        if float(delta_norms[max_idx]) < 1.0e-9:
            trans_dir = np.array([1.0, 0.0], dtype=np.float64)
        else:
            trans_dir = deltas[max_idx] / float(delta_norms[max_idx])
    else:
        trans_dir = start_delta / start_delta_norm

    translation_delta_vals = np.array([
        float(np.linalg.norm(np.asarray(pose.tvec, dtype=np.float64).reshape(2) - ref_tvec))
        for pose, _ in stored_SE2
    ], dtype=np.float64)

    max_translation_delta = float(np.max(translation_delta_vals)) if translation_delta_vals.size else 1.0
    max_translation_delta = max(max_translation_delta, 1.0e-9)

    translation_pad = max(0.5, 0.25 * max_translation_delta)

    theta_grid, translation_delta_grid = np.meshgrid(
        np.linspace(np.deg2rad(-90.0), np.deg2rad(450.0), MESH_GRID_SIZE),
        np.linspace(0.0, max_translation_delta + translation_pad, MESH_GRID_SIZE),
    )

    residual_grid = np.zeros_like(theta_grid)
    for row_idx in range(theta_grid.shape[0]):
        for col_idx in range(theta_grid.shape[1]):
            pose = create_slice_pose(
                theta_grid[row_idx, col_idx],
                translation_delta_grid[row_idx, col_idx],
                trans_dir,
                ref_tvec,
            )
            y = create_y_func(pose)
            residual_grid[row_idx, col_idx] = float(np.linalg.norm(y))

    return theta_grid, translation_delta_grid, residual_grid, trans_dir, ref_tvec


def create_slice_pose(theta_rad: float, translation_delta: float, trans_dir: np.ndarray, ref_tvec: np.ndarray) -> SE2:
    """Create a pose on the 1D translation slice through the final pose.

    translation_delta = 0 means the final/ideal translation.

    Positive translation_delta moves away from the final translation along the
    fixed slice direction trans_dir.
    """
    trans_dir = np.asarray(trans_dir, dtype=np.float64).reshape(2)
    ref_tvec = np.asarray(ref_tvec, dtype=np.float64).reshape(2)

    translation_delta = max(0.0, float(translation_delta))
    return SE2(theta_rad, ref_tvec + translation_delta * trans_dir)


def select_mesh_label_indices(
        path_theta_deg: np.ndarray,
        path_radius: np.ndarray,
        path_residual: np.ndarray,
) -> list[int]:
    keep: list[int] = []
    for idx, (theta_deg, radius_val, residual_val) in enumerate(zip(path_theta_deg, path_radius, path_residual)):
        if not keep:
            keep.append(idx)
            continue

        last_idx = keep[-1]
        if (
            abs(float(theta_deg - path_theta_deg[last_idx])) < MESH_LABEL_MIN_THETA_DEG
            and abs(float(radius_val - path_radius[last_idx])) < MESH_LABEL_MIN_RADIUS
            and abs(float(residual_val - path_residual[last_idx])) < MESH_LABEL_MIN_RESIDUAL
        ):
            continue

        keep.append(idx)

    if keep[-1] != len(path_theta_deg) - 1:
        keep.append(len(path_theta_deg) - 1)
    return keep


def create_projected_3d_label(ax, x: float, y: float, z: float, text: str):
    x_proj, y_proj, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
    return ax.annotate(
        text,
        xy=(x_proj, y_proj),
        xytext=(0, 0),
        textcoords="offset points",
        ha="center",
        va="center",
        color="black",
        fontsize=8,
        bbox={
            "boxstyle": "round,pad=0.15",
            "facecolor": "white",
            "alpha": 0.9,
            "edgecolor": "0.75",
        },
        zorder=1000,
    )


def update_projected_3d_label(ax, label, x: float, y: float, z: float) -> None:
    x_proj, y_proj, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
    label.xy = (x_proj, y_proj)


def compute_axis_length(verts: np.ndarray) -> float:
    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    return 0.25 * float(np.min(np.maximum(maxs - mins, 1.0e-6)))


def create_object_artists(ax, verts, edges, pose: SE2, color: str, alpha: float = 1.0,
                          show_axes: bool = True, point_size: int = 45) -> dict:
    world_verts = pose * verts
    translation = pose.tvec
    axis_length = compute_axis_length(verts)
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
    axis_length = compute_axis_length(artists["verts_model"])
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


def frame_to_alpha(frame_idx: int, num_segments: int) -> tuple[int, int, float]:
    num_segments = max(int(num_segments), 1)

    phase_length = max(int(N_FRAMES * 0.9 / num_segments), 1)
    pos = frame_idx / phase_length

    first_idx = int(pos)
    second_idx = first_idx + 1
    alpha = pos - first_idx

    if second_idx > num_segments:
        first_idx = num_segments - 1
        second_idx = num_segments
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


def configure_figure_layout(fig: plt.Figure) -> None:
    fig.set_size_inches(16.0, 7.5, forward=True)
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.10, top=0.90, wspace=0.28)


def main() -> None:
    fig = plt.figure(figsize=(16.0, 7.5))
    ax = fig.add_subplot(1, 2, 1)
    ax_res = fig.add_subplot(1, 2, 2, projection="3d")
    configure_figure_layout(fig)

    stored_SE2, create_y_func = factor_graph()
    num_path_segments = max(len(stored_SE2) - 1, 1)
    _, opt_res_mag = stored_SE2[-1]

    theta_grid, translation_delta_grid, residual_grid, trans_dir, ref_tvec = create_mesh_grid(
        stored_SE2,
        create_y_func,
        ref_pose=stored_SE2[-1][0],
    )

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
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        family="monospace",
        clip_on=False,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "alpha": 0.85,
            "edgecolor": "0.8",
        },
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

    surface = ax_res.plot_surface(
        np.rad2deg(theta_grid),
        translation_delta_grid,
        residual_grid,
        cmap=cm.viridis,
        linewidth=0,
        antialiased=True,
        alpha=0.88,
    )
    fig.colorbar(surface, ax=ax_res, fraction=0.046, pad=0.08, shrink=0.78, label="Residual ||y||_2")

    path_theta_deg = np.array([
        np.rad2deg(wrap_angle_2pi(pose.theta_rad))
        for pose, _ in stored_SE2
    ], dtype=np.float64)

    # This is the actual translation distance from the final/ideal pose.
    # Unlike | ||t|| - ||t_ref|| |, this is only zero at the final translation.
    path_translation_delta = np.array([
        float(np.linalg.norm(np.asarray(pose.tvec, dtype=np.float64).reshape(2) - ref_tvec))
        for pose, _ in stored_SE2
    ], dtype=np.float64)

    # Use the actual residual of the actual optimizer state.
    # This keeps the black path consistent with the left-hand pose display.
    path_residual = np.array([
        float(np.linalg.norm(create_y_func(pose)))
        for pose, _ in stored_SE2
    ], dtype=np.float64)

    ax_res.plot(
        path_theta_deg,
        path_translation_delta,
        path_residual,
        color="black",
        linewidth=2.0,
        marker="o",
        markersize=4,
    )
    ax_res.scatter(path_theta_deg[0], path_translation_delta[0], path_residual[0], color="tab:red", s=60,
                   depthshade=False)
    ax_res.scatter(path_theta_deg[-1], path_translation_delta[-1], path_residual[-1], color="tab:green", s=60,
                   depthshade=False)

    labeled_indices = set(select_mesh_label_indices(path_theta_deg, path_translation_delta, path_residual))
    path_labels = []
    path_label_positions = []
    for idx, (theta_deg, translation_delta_val, residual_val) in enumerate(
            zip(path_theta_deg, path_translation_delta, path_residual)
    ):
        if idx not in labeled_indices:
            continue

        label_text = "Start" if idx == 0 else ("Final" if idx == len(path_theta_deg) - 1 else str(idx))
        label_x = float(theta_deg)
        label_y = float(translation_delta_val)
        label_z = float(residual_val + MESH_LABEL_Z_OFFSET)
        label = create_projected_3d_label(
            ax_res,
            label_x,
            label_y,
            label_z,
            label_text,
        )
        path_labels.append(label)
        path_label_positions.append((label_x, label_y, label_z))

    interp_point = ax_res.scatter(
        [path_theta_deg[0]],
        [path_translation_delta[0]],
        [path_residual[0]],
        color="white",
        edgecolors="black",
        s=90,
        linewidths=1.2,
        depthshade=False,
        zorder=10,
    )

    ax_res.set_xlabel("theta [deg]")
    ax_res.set_ylabel("translation error ||t - t*||")
    ax_res.set_zlabel("Residual ||y||_2")
    ax_res.set_title("Residual Surface Over (theta, translation error)")
    ax_res.set_ylim(0.0, float(translation_delta_grid.max()))
    ax_res.set_zlim(MESH_Z_MIN, MESH_Z_MAX)
    ax_res.view_init(elev=MESH_ELEV_DEG, azim=MESH_AZIM_START_DEG)
    for label, (label_x, label_y, label_z) in zip(path_labels, path_label_positions):
        update_projected_3d_label(ax_res, label, label_x, label_y, label_z)

    dir_angle_deg = np.rad2deg(np.arctan2(trans_dir[1], trans_dir[0]))
    ax_res.text2D(
        0.03,
        0.97,
        f"translation slice dir: {dir_angle_deg:6.2f} deg",
        transform=ax_res.transAxes,
        va="top",
    )

    def update(frame_idx: int):
        first_idx, second_idx, alpha = frame_to_alpha(frame_idx, num_path_segments)
        first_SE2, first_res = stored_SE2[first_idx]
        second_SE2, second_res = stored_SE2[second_idx]
        cam_progress = float(np.clip(frame_idx / max(N_FRAMES - 1, 1), 0.0, 1.0))
        cam_azim_deg = (1.0 - cam_progress) * MESH_AZIM_START_DEG + cam_progress * MESH_AZIM_END_DEG

        state_pose = first_SE2.interpolate(second_SE2, alpha)
        y = create_y_func(state_pose)
        residual = float(np.linalg.norm(y))

        error_SE2 = state_pose.inv * MEAS_SE2
        angle_error_deg = abs(np.rad2deg(wrap_angle_signed(error_SE2.theta_rad)))
        trans_error = float(np.linalg.norm(error_SE2.tvec))

        interp_theta_deg = float(np.rad2deg(wrap_angle_2pi(state_pose.theta_rad)))
        interp_translation_delta = float(
            np.linalg.norm(np.asarray(state_pose.tvec, dtype=np.float64).reshape(2) - ref_tvec)
        )

        # Use the actual interpolated-state residual so the moving marker agrees
        # with the left-hand pose view.
        interp_residual = residual

        for g_idx, (g_artist, g_label) in enumerate(zip(ghost_artists, ghost_labels)):
            visible = g_idx <= first_idx
            set_artist_group_visible(g_artist, visible)
            g_label.set_visible(visible)

        meas_verts = update_object_artists(meas_artists, MEAS_SE2)
        state_verts = update_object_artists(state_artists, state_pose)
        update_connection_lines(connection_lines, meas_verts, state_verts)

        interp_point._offsets3d = (
            [interp_theta_deg],
            [interp_translation_delta],
            [interp_residual],
        )

        ax_res.view_init(elev=MESH_ELEV_DEG, azim=cam_azim_deg)
        for label, (label_x, label_y, label_z) in zip(path_labels, path_label_positions):
            update_projected_3d_label(ax_res, label, label_x, label_y, label_z)

        status_text.set_text(
            f"           Estimated | Solution\n"
            f"theta:       {np.rad2deg(state_pose.theta_rad):8.3f}|{np.rad2deg(meas_pose_0.theta_rad):8.3f} deg\n"
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
            interp_point,
            *path_labels,
            *(artist for g_artist in ghost_artists for artist in
              [ghost_artist["scatter"], *ghost_artist["edge_lines"]]),
            *ghost_labels,
        ]

    from support.io.my_logging import LOG

    update(0)
    configure_figure_layout(fig)
    fig.canvas.draw()
    snapshot_path = save_demo_snapshot(fig, "_initial")
    LOG.info(f"Saved initial snapshot to: {snapshot_path}")

    update(N_FRAMES - 1)
    fig.canvas.draw()
    snapshot_path = save_demo_snapshot(fig)
    LOG.info(f"Saved final snapshot to: {snapshot_path}")

    anim = FuncAnimation(fig, update, frames=N_FRAMES, interval=FRAME_INTERVAL_MS, blit=False, repeat=True)
    fig._object_alignment_anim = anim

    video_path = save_demo_video(anim)
    LOG.info(f"Saved video to: {video_path}")

    plt.show()


if __name__ == "__main__":
    main()
