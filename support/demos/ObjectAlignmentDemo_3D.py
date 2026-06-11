import numpy as np
from pathlib import Path
import support.mathHelpers.quaternions as q
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from mpl_toolkits.mplot3d import proj3d

LENGTH = 2
WIDTH = 2
HEIGHT = 5
TAPER_SCALE = 1.2

N_FRAMES = 180
FRAME_INTERVAL_MS = 40
CAMERA_ELEV_DEG = 25.0
CAMERA_AZIM_RATE_DEG = 1.0

RECT_PRISM_VERTS = np.array([[LENGTH / 2, WIDTH / 2, -HEIGHT / 2],
                             [-LENGTH / 2, WIDTH / 2, -HEIGHT / 2],
                             [-LENGTH / 2, -WIDTH / 2, -HEIGHT / 2],
                             [LENGTH / 2, -WIDTH / 2, -HEIGHT / 2],
                             [LENGTH / 2, WIDTH / 2, HEIGHT / 2],
                             [-LENGTH / 2, WIDTH / 2, HEIGHT / 2],
                             [-LENGTH / 2, -WIDTH / 2, HEIGHT / 2],
                             [LENGTH / 2, -WIDTH / 2, HEIGHT / 2]]).astype(np.float32)

TAPERED_PRISM_VERTS = np.array([[LENGTH / (2 * TAPER_SCALE), WIDTH / (2 * TAPER_SCALE),
                                 -HEIGHT * TAPER_SCALE / 2],

                                [-LENGTH / (2 * TAPER_SCALE), WIDTH / (2 * TAPER_SCALE),
                                 -HEIGHT * TAPER_SCALE / 2],

                                [-LENGTH / (2 * TAPER_SCALE), -WIDTH / (2 * TAPER_SCALE),
                                 -HEIGHT * TAPER_SCALE / 2],

                                [LENGTH / (2 * TAPER_SCALE), -WIDTH / (2 * TAPER_SCALE),
                                 -HEIGHT * TAPER_SCALE / 2],

                                [LENGTH * TAPER_SCALE / 2, WIDTH * TAPER_SCALE / 2,
                                 HEIGHT * TAPER_SCALE / 2],

                                [-LENGTH * TAPER_SCALE / 2, WIDTH * TAPER_SCALE / 2,
                                 HEIGHT * TAPER_SCALE / 2],

                                [-LENGTH * TAPER_SCALE / 2, -WIDTH * TAPER_SCALE / 2,
                                 HEIGHT * TAPER_SCALE / 2],

                                [LENGTH * TAPER_SCALE / 2, -WIDTH * TAPER_SCALE / 2,
                                 HEIGHT * TAPER_SCALE / 2]]
                               ).astype(np.float32)

EDGES = np.array([[0, 1],
                  [1, 2],
                  [2, 3],
                  [3, 0],
                  [0, 4],
                  [1, 5],
                  [2, 6],
                  [3, 7],
                  [4, 5],
                  [5, 6],
                  [6, 7],
                  [7, 4]])
NUM_VERTS = len(RECT_PRISM_VERTS)
NUM_ELEMENTS = np.prod(RECT_PRISM_VERTS.shape)
NUM_STATES = 6

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

MEAS_SE3 = q.SE3_q.random(5.0)
MEAS_VERTS = MEAS_SE3 * RECT_PRISM_VERTS

NUM_ITERATIONS = 5

def factor_graph():
    pert_q = q.random_quat_within_deg(90.0)
    pert_t = np.random.rand(3,)*10.0
    est_SE3 = q.SE3_q(pert_q * MEAS_SE3.quat, MEAS_SE3.tvec + pert_t)

    stored_SE3 = []

    def create_y(SE3_input=None):
        y = np.zeros((NUM_ELEMENTS,))

        if SE3_input is None:
            SE3_input = est_SE3

        for idx, verts in enumerate(zip(MEAS_VERTS, TAPERED_PRISM_VERTS)):
            m_verts, s_verts = verts
            y[3*idx:3*idx+3] = m_verts - SE3_input * s_verts

        return y

    def create_L():
        L = np.zeros((NUM_ELEMENTS, NUM_STATES))
        I3 = np.eye(3)
        for idx, verts in enumerate(TAPERED_PRISM_VERTS):
            L[3 * idx:3 * idx + 3, 0:3] = q.skew(est_SE3.quat * verts)
            L[3 * idx:3 * idx + 3, 3:6] = -I3
        return L

    y = create_y()
    L = create_L()
    y_mag = y.T @ y
    stored_SE3.append([est_SE3.copy(), y_mag])

    for idx in range(NUM_ITERATIONS):
        dx = -np.linalg.pinv(L) @ y
        pert_q = q.Quaternion.exp_so3(dx[:3])
        pert_t = dx[3:]
        est_SE3.quat = pert_q * est_SE3.quat
        est_SE3.tvec += pert_t

        y = create_y()
        L = create_L()
        y_mag = y.T @ y
        stored_SE3.append([est_SE3.copy(), y_mag])

    return stored_SE3, create_y

def compute_axis_length(verts: np.ndarray) -> float:
    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    return 0.35 * np.min(np.maximum(maxs - mins, 1.0e-6))


def project_world_to_axes(ax, point: np.ndarray) -> tuple[float, float]:
    x_proj, y_proj, _ = proj3d.proj_transform(point[0], point[1], point[2], ax.get_proj())
    x_disp, y_disp = ax.transData.transform((x_proj, y_proj))
    return ax.transAxes.inverted().transform((x_disp, y_disp))


def create_object_artists(ax, verts, edges, SE3, color, alpha=1.0, show_axes=True, point_size=40):
    world_verts = SE3 * verts
    translation = SE3.tvec
    axis_length = compute_axis_length(world_verts)
    basis = np.eye(3, dtype=np.float32) * axis_length
    world_basis = SE3 * basis

    scatter = ax.scatter(world_verts[:, 0], world_verts[:, 1], world_verts[:, 2], color=color, s=point_size, alpha=alpha)

    axis_colors = ("r", "g", "b")
    axis_labels = ("X", "Y", "Z")
    axis_lines = []
    axis_texts = []
    if show_axes:
        for axis_vec, axis_color, axis_label in zip(world_basis.T, axis_colors, axis_labels):
            line, = ax.plot([translation[0], axis_vec[0]],
                            [translation[1], axis_vec[1]],
                            [translation[2], axis_vec[2]],
                            color=axis_color,
                            linewidth=2,
                            alpha=alpha)
            text_x, text_y = project_world_to_axes(ax, axis_vec)
            text = ax.text2D(text_x,
                             text_y,
                             axis_label,
                             transform=ax.transAxes,
                             color=axis_color,
                             alpha=alpha,
                             rotation=0,
                             ha="center",
                             va="center")
            axis_lines.append(line)
            axis_texts.append(text)

    edge_lines = []
    for start_idx, end_idx in edges:
        edge = world_verts[[start_idx, end_idx]]
        line, = ax.plot(edge[:, 0], edge[:, 1], edge[:, 2], color=color, alpha=alpha)
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


def update_object_artists(artists: dict, SE3: q.SE3_q) -> np.ndarray:
    ax = artists["ax"]
    world_verts = SE3 * artists["verts_model"]
    translation = SE3.tvec
    axis_length = compute_axis_length(world_verts)
    basis = np.eye(3, dtype=np.float32) * axis_length
    world_basis = SE3 * basis

    artists["scatter"]._offsets3d = (world_verts[:, 0], world_verts[:, 1], world_verts[:, 2])

    for axis_line, axis_text, axis_vec in zip(artists["axis_lines"], artists["axis_texts"], world_basis.T):
        axis_line.set_data_3d([translation[0], axis_vec[0]],
                              [translation[1], axis_vec[1]],
                              [translation[2], axis_vec[2]])
        text_x, text_y = project_world_to_axes(ax, axis_vec)
        axis_text.set_position((text_x, text_y))
        axis_text.set_rotation(0)

    for edge_line, (start_idx, end_idx) in zip(artists["edge_lines"], artists["edges"]):
        edge = world_verts[[start_idx, end_idx]]
        edge_line.set_data_3d(edge[:, 0], edge[:, 1], edge[:, 2])

    return world_verts


def create_connection_lines(ax, verts_a: np.ndarray, verts_b: np.ndarray):
    lines = []
    for vert_a, vert_b in zip(verts_a, verts_b):
        line, = ax.plot([vert_a[0], vert_b[0]],
                        [vert_a[1], vert_b[1]],
                        [vert_a[2], vert_b[2]],
                        color="darkred",
                        linewidth=1.5)
        lines.append(line)
    return lines


def update_connection_lines(lines, verts_a: np.ndarray, verts_b: np.ndarray) -> None:
    for line, vert_a, vert_b in zip(lines, verts_a, verts_b):
        line.set_data_3d([vert_a[0], vert_b[0]],
                         [vert_a[1], vert_b[1]],
                         [vert_a[2], vert_b[2]])


def set_artist_group_visible(artists: dict, visible: bool) -> None:
    artists["scatter"].set_visible(visible)
    for line in artists["axis_lines"]:
        line.set_visible(visible)
    for text in artists["axis_texts"]:
        text.set_visible(visible)
    for line in artists["edge_lines"]:
        line.set_visible(visible)


def set_axes_equal(ax, mins: np.ndarray, maxs: np.ndarray) -> tuple[np.ndarray, float]:
    center = (mins + maxs) / 2.0
    radius = np.max(maxs - mins) / 2.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    return center, radius


def measurement_pose_at_frame(frame_idx: int, base_pose: q.SE3_q) -> q.SE3_q:
    del frame_idx
    return base_pose


def state_pose_at_frame(frame_idx: int, meas_pose: q.SE3_q, initial_perturb: q.SE3_q) -> q.SE3_q:
    alpha = frame_idx / max(N_FRAMES - 1, 1)

    rot_vec = initial_perturb.quat.ln_so3.vec
    interp_quat = q.Quaternion.exp_so3((1.0 - alpha) * rot_vec)
    interp_tvec = (1.0 - alpha) * initial_perturb.tvec
    interp_perturb = q.SE3_q(interp_quat, interp_tvec)

    return interp_perturb * meas_pose


def camera_view_at_frame(frame_idx: int) -> tuple[float, float]:
    elev = CAMERA_ELEV_DEG
    azim = 45.0 + CAMERA_AZIM_RATE_DEG * frame_idx / 2.0
    return elev, azim


def save_demo_snapshot(fig: plt.Figure, stem_suffix: str = "_final") -> Path:
    output_path = Path(__file__).with_name(f"{Path(__file__).stem}{stem_suffix}.pdf")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return output_path


def save_demo_video(anim: FuncAnimation) -> Path:
    output_path = Path(__file__).with_suffix(".mp4")
    fps = max(1, round(1000.0 / FRAME_INTERVAL_MS))
    anim.save(output_path, writer=FFMpegWriter(fps=fps, bitrate=4000), dpi=150)
    return output_path


def main():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")



    pert_q = q.random_quat_within_deg(1.0)
    pert_t = np.random.rand(3)/100.0
    pert_SE3 = q.SE3_q(pert_q, pert_t)

    meas_pose_0 = measurement_pose_at_frame(0, MEAS_SE3)
    state_pose_0 = state_pose_at_frame(0, meas_pose_0, pert_SE3)
    meas_rpy = meas_pose_0.quat.eulerD()
    meas_trans = meas_pose_0.tvec

    meas_artists = create_object_artists(ax, RECT_PRISM_VERTS, EDGES, meas_pose_0, "tab:blue")
    state_artists = create_object_artists(ax, TAPERED_PRISM_VERTS, EDGES, state_pose_0, "tab:orange")

    state_verts_0 = state_pose_0 * TAPERED_PRISM_VERTS
    connection_lines = create_connection_lines(ax, MEAS_VERTS, state_verts_0)

    mins = np.minimum(MEAS_VERTS.min(axis=0), state_verts_0.min(axis=0))
    maxs = np.maximum(MEAS_VERTS.max(axis=0), state_verts_0.max(axis=0))
    set_axes_equal(ax, mins, maxs)

    status_text = ax.text2D(
        1.02,
        0.98,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        family="monospace",
        clip_on=False,
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Object Alignment Demo")

    elev_0, azim_0 = camera_view_at_frame(0)
    ax.view_init(elev=elev_0, azim=azim_0)

    stored_SE3, create_y_func = factor_graph()

    opt_res = create_y_func(meas_pose_0)
    opt_res_mag = opt_res.T @ opt_res

    ghost_artists = []
    ghost_labels = []
    for ghost_idx, (ghost_SE3, _) in enumerate(stored_SE3):
        ghost_artist = create_object_artists(
            ax,
            TAPERED_PRISM_VERTS,
            EDGES,
            ghost_SE3,
            "tab:orange",
            alpha=0.18,
            show_axes=False,
            point_size=20,
        )
        set_artist_group_visible(ghost_artist, False)
        ghost_artists.append(ghost_artist)

        ghost_verts = ghost_SE3 * TAPERED_PRISM_VERTS
        label_pos = ghost_verts[0]
        label_text = "Start" if ghost_idx == 0 else f"Iteration {ghost_idx}"
        ghost_label = ax.text(
            label_pos[0],
            label_pos[1],
            label_pos[2],
            label_text,
            color="tab:orange",
            alpha=0.55,
        )
        ghost_label.set_visible(False)
        ghost_labels.append(ghost_label)

    phase_length = int(N_FRAMES*0.9/5)
    def frame_to_alpha(frame_idx):
        pos = frame_idx / phase_length
        first_idx = int(pos)
        second_idx = first_idx + 1
        alpha = pos - first_idx

        if second_idx > NUM_ITERATIONS:
            first_idx = NUM_ITERATIONS - 1
            second_idx = NUM_ITERATIONS
            alpha = 1.0

        return first_idx, second_idx, alpha

    def update(frame_idx: int):
        first_idx, sec_idx, alpha = frame_to_alpha(frame_idx)
        first_SE3, first_res = stored_SE3[first_idx]
        second_SE3, second_res = stored_SE3[sec_idx]

        quat = first_SE3.quat.slerp(second_SE3.quat, alpha)
        tvec = second_SE3.tvec * alpha + first_SE3.tvec * (1.0 - alpha)
        state_SE3 = q.SE3_q(quat, tvec)
        state_rpy = state_SE3.quat.eulerD()
        y = create_y_func(state_SE3)
        residual = y.T @ y
        error_SE3 = state_SE3.inv * MEAS_SE3

        if alpha == 0.0:
            LAST_IDX = first_idx

        meas_pose = measurement_pose_at_frame(frame_idx, MEAS_SE3)
        state_pose = state_SE3

        for ghost_idx, (ghost_artist, ghost_label) in enumerate(zip(ghost_artists, ghost_labels)):
            visible = ghost_idx <= first_idx
            set_artist_group_visible(ghost_artist, visible)
            ghost_label.set_visible(visible)

        set_axes_equal(ax, mins, maxs)

        elev, azim = camera_view_at_frame(frame_idx)
        ax.view_init(elev=elev, azim=azim)

        meas_verts = update_object_artists(meas_artists, meas_pose)
        state_verts = update_object_artists(state_artists, state_pose)
        update_connection_lines(connection_lines, meas_verts, state_verts)

        meas_rpy = meas_pose_0.quat.eulerD()

        status_text.set_text(
            f"           Estimated | Solution\n"
            f"Roll:        {state_rpy[0]:8.3f}|{meas_rpy[0]:8.3f} deg\n"
            f"Pitch:       {state_rpy[1]:8.3f}|{meas_rpy[1]:8.3f} deg\n"
            f"Yaw:         {state_rpy[2]:8.3f}|{meas_rpy[2]:8.3f} deg\n"
            f"Cx:          {tvec[0]:8.3f}|{meas_trans[0]:8.3f}\n"
            f"Cy:          {tvec[1]:8.3f}|{meas_trans[1]:8.3f}\n"
            f"Cz:          {tvec[2]:8.3f}|{meas_trans[2]:8.3f}\n\n"
            f"Iteration:   {sec_idx:8d}\n"
            f"Alpha:       {alpha:8.3f}\n\n"
            f"Residual:    {residual:8.3f}|{opt_res_mag:8.3f}\n"
            f"Angle-Error: {error_SE3.quat.angle_betweenD(q.identity()):8.3f} deg\n"
            f"Trans-Error: {np.linalg.norm(error_SE3.tvec):8.3f}"
        )

        plt.tight_layout()

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

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
