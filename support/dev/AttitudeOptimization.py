import numpy as np
import support.mathHelpers.quaternions as q
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

np.set_printoptions(precision=6, suppress=True)

# -----------------------------------------------------------------------------
# Quaternion batch least-squares playground
#
# Goal:
#   Estimate one unknown quaternion, q2, from many noisy measurements of
#
#       z_i ~= q2 * q1_i
#
#   where q1_i is known for each measurement and z_i is a noisy quaternion
#   measurement of the composite attitude.
#
# The perturbation model used here is right-multiplicative on q2:
#
#       q2_updated = q2_current * exp_so3(dx)
#
# With the residual definition
#
#       r_i = Log( z_i^* * (q2_current * q1_i) )
#
# the local linearized residual is
#
#       r_i(dx) ~= r_i(0) + J_i dx
#       J_i      = R(q1_i)^T
#
# so stacking N measurements produces a tall 3N x 3 system:
#
#       J_stack dx = -r_stack
#
# and the pseudo-inverse gives the least-squares correction.
# -----------------------------------------------------------------------------

N_MEASUREMENTS = 12
MEAS_NOISE_SIGMA_RAD = np.deg2rad(10.)
INITIAL_ERROR_RAD = np.deg2rad(np.array([50.0, -30.0, 20.0]))
MAX_ITERS = 5
CONVERGENCE_TOL_RAD = 1.0e-12

# Visualization toggles
SHOW_PLOT = True
SAVE_ANIMATION = False
ANIMATION_FILENAME = "quaternion_batch_airplane.gif"
SUBFRAMES_PER_ITER = 20
AIRPLANE_SCALE = 1.0

rng = np.random.default_rng(7)


def small_noise_vec(sigma_rad: float) -> NDArray:
    """Draw a small 3-vector perturbation in radians."""
    return rng.normal(loc=0.0, scale=sigma_rad, size=3)


def calc_single_residual(meas: q.Quaternion, state: q.Quaternion) -> NDArray:
    """
    Residual from measured quaternion to predicted/state quaternion.

    This is intentionally the same sign convention as the single-measurement
    playground: dx = -pinv(J) @ r.
    """
    return (meas.T * state).ln_so3.vec


def make_noisy_measurements(
    q2_true: q.Quaternion,
    q1_samples: list[q.Quaternion],
    noise_sigma_rad: float,
) -> tuple[list[q.Quaternion], list[q.Quaternion]]:
    """
    Build clean and noisy measurements of q2_true * q1_i.

    Noise is applied as a small right-multiplicative quaternion perturbation
    on each composite measurement.
    """
    clean_measurements: list[q.Quaternion] = []
    noisy_measurements: list[q.Quaternion] = []

    for q1_i in q1_samples:
        clean_i = (q2_true * q1_i).force_s_pos()
        noise_i = q.Quaternion.exp_so3(small_noise_vec(noise_sigma_rad))
        noisy_i = (clean_i * noise_i).force_s_pos()

        clean_measurements.append(clean_i)
        noisy_measurements.append(noisy_i)

    return clean_measurements, noisy_measurements


def build_tall_system(
    q2_current: q.Quaternion,
    q1_samples: list[q.Quaternion],
    measurements: list[q.Quaternion],
) -> tuple[NDArray, NDArray]:
    """
    Stack all 3-vector residuals and 3x3 Jacobian blocks.

    Returns:
        J_stack: shape (3*N, 3)
        r_stack: shape (3*N,)
    """
    J_blocks: list[NDArray] = []
    r_blocks: list[NDArray] = []

    for q1_i, meas_i in zip(q1_samples, measurements):
        predicted_i = (q2_current * q1_i).force_s_pos()

        r_i = calc_single_residual(meas_i, predicted_i)
        J_i = q1_i.to_dcm().T

        r_blocks.append(r_i)
        J_blocks.append(J_i)

    J_stack = np.vstack(J_blocks)
    r_stack = np.concatenate(r_blocks)
    return J_stack, r_stack


def apply_batch_update(
    q2_current: q.Quaternion,
    q1_samples: list[q.Quaternion],
    measurements: list[q.Quaternion],
) -> tuple[q.Quaternion, NDArray, NDArray, NDArray]:
    """
    Solve the tall least-squares system and right-apply the correction to q2.
    """
    J_stack, r_stack = build_tall_system(q2_current, q1_samples, measurements)

    # Equivalent options:
    #   dx = -np.linalg.pinv(J_stack) @ r_stack
    #   dx = np.linalg.lstsq(J_stack, -r_stack, rcond=None)[0]
    # Use pinv here because the playground is explicitly demonstrating it.
    dx = -np.linalg.pinv(J_stack) @ r_stack

    dq_update = q.Quaternion.exp_so3(dx)
    q2_updated = (q2_current * dq_update).force_s_pos()

    return q2_updated, dx, J_stack, r_stack


def quat_error_vec(reference: q.Quaternion, estimate: q.Quaternion) -> NDArray:
    """Small-angle error vector from reference quaternion to estimate quaternion."""
    return calc_single_residual(reference.force_s_pos(), estimate.force_s_pos())


def print_error_summary(label: str, q2_true: q.Quaternion, q2_estimate: q.Quaternion) -> None:
    err_vec = quat_error_vec(q2_true, q2_estimate)
    print(f"{label} error vector [rad]: {err_vec}")
    print(f"{label} error norm   [deg]: {np.rad2deg(np.linalg.norm(err_vec)):.6f}")


# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------

def quat_geodesic_interp(q_start: q.Quaternion, q_end: q.Quaternion, alpha: float) -> q.Quaternion:
    """Interpolate from q_start to q_end using the local log/exp map."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    delta = (q_start.T * q_end).ln_so3.vec
    return (q_start * q.Quaternion.exp_so3(alpha * delta)).force_s_pos()


def make_airplane_segments(scale: float = 1.0) -> list[NDArray]:
    """Return a tiny wireframe airplane defined in body coordinates."""
    nose = scale * np.array([1.3, 0.0, 0.0])
    tail = scale * np.array([-1.0, 0.0, 0.0])
    wing_l = scale * np.array([0.0, 0.95, 0.0])
    wing_r = scale * np.array([0.0, -0.95, 0.0])
    tail_l = scale * np.array([-0.75, 0.35, 0.0])
    tail_r = scale * np.array([-0.75, -0.35, 0.0])
    fin_top = scale * np.array([-0.75, 0.0, 0.35])
    fin_base = scale * np.array([-0.75, 0.0, 0.0])

    return [
        np.vstack([tail, nose]),       # fuselage
        np.vstack([wing_l, wing_r]),   # main wing
        np.vstack([tail_l, tail_r]),   # tail plane
        np.vstack([fin_base, fin_top]),  # vertical stabilizer
    ]


def rotate_segments(segments: list[NDArray], quat_obj: q.Quaternion) -> list[NDArray]:
    """Rotate airplane segments into world coordinates."""
    R = quat_obj.to_dcm()
    rotated: list[NDArray] = []
    for seg in segments:
        rotated.append((R @ seg.T).T)
    return rotated


def set_axes_equal_3d(ax) -> None:
    """Force a 3D axis to use equal scaling."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    radius = 0.5 * max([x_range, y_range, z_range])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - radius, x_middle + radius])
    ax.set_ylim3d([y_middle - radius, y_middle + radius])
    ax.set_zlim3d([z_middle - radius, z_middle + radius])


def draw_world_axes(ax, axis_len: float = 1.6) -> None:
    ax.plot([0.0, axis_len], [0.0, 0.0], [0.0, 0.0], lw=2, color="tab:red")
    ax.plot([0.0, 0.0], [0.0, axis_len], [0.0, 0.0], lw=2, color="tab:green")
    ax.plot([0.0, 0.0], [0.0, 0.0], [0.0, axis_len], lw=2, color="tab:blue")
    ax.text(axis_len + 0.05, 0.0, 0.0, "X", color="tab:red")
    ax.text(0.0, axis_len + 0.05, 0.0, "Y", color="tab:green")
    ax.text(0.0, 0.0, axis_len + 0.05, "Z", color="tab:blue")


def build_animation(
    q2_true: q.Quaternion,
    q2_initial: q.Quaternion,
    q_history: list[q.Quaternion],
    residual_rms_deg_history: list[float],
    dx_deg_history: list[float],
    error_deg_history: list[float],
    J_final: NDArray,
) -> None:
    """Build a Matplotlib animation with a tiny airplane and convergence plots."""
    segments = make_airplane_segments(AIRPLANE_SCALE)
    n_steps = len(q_history) - 1
    total_frames = max(1, n_steps * SUBFRAMES_PER_ITER + 1)

    fig = plt.figure(figsize=(13, 8))
    gs = GridSpec(2, 2, width_ratios=[1.55, 1.0], height_ratios=[1.0, 1.0], figure=fig)

    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_err = fig.add_subplot(gs[0, 1])
    ax_conv = fig.add_subplot(gs[1, 1])

    # ----- 3D airplane axis -----
    draw_world_axes(ax3d)
    ax3d.set_title("Quaternion Batch LS: silly airplane convergence")
    ax3d.set_xlim(-1.8, 1.8)
    ax3d.set_ylim(-1.8, 1.8)
    ax3d.set_zlim(-1.8, 1.8)
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    set_axes_equal_3d(ax3d)
    ax3d.view_init(elev=22, azim=40)

    true_segments = rotate_segments(segments, q2_true)
    init_segments = rotate_segments(segments, q2_initial)
    est_segments = rotate_segments(segments, q_history[0])

    true_lines = []
    init_lines = []
    est_lines = []

    for seg in true_segments:
        line, = ax3d.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="tab:green", lw=3, alpha=0.9)
        true_lines.append(line)
    for seg in init_segments:
        line, = ax3d.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="tab:red", lw=2, alpha=0.25, ls="--")
        init_lines.append(line)
    for seg in est_segments:
        line, = ax3d.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="tab:orange", lw=3, alpha=1.0)
        est_lines.append(line)

    ax3d.text2D(0.02, 0.97, "Green = true attitude", transform=ax3d.transAxes, color="tab:green")
    ax3d.text2D(0.02, 0.93, "Red dashed = initial guess", transform=ax3d.transAxes, color="tab:red")
    ax3d.text2D(0.02, 0.89, "Orange = current estimate", transform=ax3d.transAxes, color="tab:orange")

    status_text = ax3d.text2D(0.02, 0.02, "", transform=ax3d.transAxes, fontsize=10)
    info_text = ax3d.text2D(
        0.60,
        0.02,
        f"N = {N_MEASUREMENTS}\nJ shape = ({J_final.shape[0]}, {J_final.shape[1]})\nrank(J) = {np.linalg.matrix_rank(J_final)}\ncond(J) = {np.linalg.cond(J_final):.3f}",
        transform=ax3d.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    # ----- Error plot -----
    iter_error_x = np.arange(len(error_deg_history))
    ax_err.plot(iter_error_x, error_deg_history, marker="o", lw=2, color="tab:purple")
    current_err_marker, = ax_err.plot([0], [error_deg_history[0]], marker="o", markersize=10, color="gold")
    ax_err.set_title("Attitude error norm")
    ax_err.set_xlabel("Iteration")
    ax_err.set_ylabel("Error norm [deg]")
    ax_err.grid(True, alpha=0.3)
    ax_err.set_xlim(-0.2, max(1.2, len(error_deg_history) - 0.8))
    ax_err.set_ylim(0.0, max(error_deg_history) * 1.1 + 1e-9)

    # ----- Convergence plot -----
    iter_conv_x = np.arange(len(residual_rms_deg_history))
    residual_plot, = ax_conv.semilogy(iter_conv_x, residual_rms_deg_history, marker="o", lw=2, color="tab:blue", label="Residual RMS")
    dx_plot, = ax_conv.semilogy(iter_conv_x, dx_deg_history, marker="s", lw=2, color="tab:orange", label="|dx|")
    current_resid_marker, = ax_conv.semilogy([0], [residual_rms_deg_history[0]], marker="o", markersize=10, color="gold")
    current_dx_marker, = ax_conv.semilogy([0], [dx_deg_history[0]], marker="s", markersize=10, color="gold")
    ax_conv.set_title("Convergence history")
    ax_conv.set_xlabel("Iteration")
    ax_conv.set_ylabel("Magnitude [deg]")
    ax_conv.grid(True, alpha=0.3, which="both")
    ax_conv.legend(loc="best")
    ax_conv.set_xlim(-0.2, max(1.2, len(iter_conv_x) - 0.8))

    fig.tight_layout()

    def update(frame_idx: int):
        if n_steps == 0:
            seg_idx = 0
            alpha = 1.0
        else:
            seg_idx = min(frame_idx // SUBFRAMES_PER_ITER, n_steps - 1)
            sub_idx = frame_idx % SUBFRAMES_PER_ITER
            alpha = sub_idx / float(SUBFRAMES_PER_ITER)
            if frame_idx >= total_frames - 1:
                seg_idx = n_steps - 1
                alpha = 1.0

        q_start = q_history[seg_idx]
        q_end = q_history[min(seg_idx + 1, len(q_history) - 1)]
        q_anim = quat_geodesic_interp(q_start, q_end, alpha)
        est_now = rotate_segments(segments, q_anim)

        for line, seg in zip(est_lines, est_now):
            line.set_data(seg[:, 0], seg[:, 1])
            line.set_3d_properties(seg[:, 2])

        err_interp = (1.0 - alpha) * error_deg_history[seg_idx] + alpha * error_deg_history[min(seg_idx + 1, len(error_deg_history) - 1)]
        resid_interp = residual_rms_deg_history[min(seg_idx, len(residual_rms_deg_history) - 1)]
        dx_interp = dx_deg_history[min(seg_idx, len(dx_deg_history) - 1)]

        current_err_marker.set_data([seg_idx + alpha], [err_interp])
        current_resid_marker.set_data([seg_idx], [resid_interp])
        current_dx_marker.set_data([seg_idx], [dx_interp])

        status_text.set_text(
            f"Iter {seg_idx}  →  {min(seg_idx + 1, n_steps)}\n"
            f"blend = {alpha:0.2f}\n"
            f"error = {err_interp:0.4f} deg\n"
            f"resid RMS = {resid_interp:0.4f} deg\n"
            f"|dx| = {dx_interp:0.6f} deg"
        )

        artists = est_lines + [current_err_marker, current_resid_marker, current_dx_marker, status_text, info_text]
        return artists

    anim = FuncAnimation(fig, update, frames=total_frames, interval=90, blit=False, repeat=True)

    if SAVE_ANIMATION:
        print(f"Saving animation to {ANIMATION_FILENAME} ...")
        anim.save(ANIMATION_FILENAME, writer="pillow", fps=12)

    plt.show()


# -----------------------------------------------------------------------------
# Simulation setup
# -----------------------------------------------------------------------------

q2_true = q.randomQuat().force_s_pos()
q1_samples = [q.randomQuat().force_s_pos() for _ in range(N_MEASUREMENTS)]

clean_measurements, noisy_measurements = make_noisy_measurements(
    q2_true=q2_true,
    q1_samples=q1_samples,
    noise_sigma_rad=MEAS_NOISE_SIGMA_RAD,
)

# Start from a deliberately wrong q2 estimate.
q2_initial = (q2_true * q.Quaternion.exp_so3(INITIAL_ERROR_RAD)).force_s_pos()
q2_initial = q.identity()
q2_current = q2_initial

print("Quaternion batch least-squares playground")
print("-----------------------------------------")
print(f"Number of measurements: {N_MEASUREMENTS}")
print(f"Measurement noise sigma [deg]: {np.rad2deg(MEAS_NOISE_SIGMA_RAD):.3f}")
print(f"Expected stacked Jacobian shape: ({3 * N_MEASUREMENTS}, 3)\n")

print(f"True q2:    {q2_true}")
print(f"Initial q2: {q2_current}\n")
print_error_summary("Initial", q2_true, q2_current)
print()


# -----------------------------------------------------------------------------
# Batch solve
# -----------------------------------------------------------------------------

q_history: list[q.Quaternion] = [q2_current]
residual_rms_deg_history: list[float] = []
dx_deg_history: list[float] = []
error_deg_history: list[float] = [np.rad2deg(np.linalg.norm(quat_error_vec(q2_true, q2_current)))]

for iter_idx in range(MAX_ITERS):
    q2_next, dx, J_stack, r_stack = apply_batch_update(
        q2_current=q2_current,
        q1_samples=q1_samples,
        measurements=noisy_measurements,
    )

    residual_rms_rad = np.sqrt(np.mean(r_stack**2))
    dx_deg = np.rad2deg(np.linalg.norm(dx))

    print(f"Iteration {iter_idx}")
    print(f"  J_stack shape: {J_stack.shape}")
    print(f"  r_stack shape: {r_stack.shape}")
    print(f"  residual RMS [deg]: {np.rad2deg(residual_rms_rad):.6f}")
    print(f"  dx [rad]: {dx}")
    print(f"  |dx| [deg]: {dx_deg:.6f}")

    q2_current = q2_next
    q_history.append(q2_current)
    residual_rms_deg_history.append(np.rad2deg(residual_rms_rad))
    dx_deg_history.append(dx_deg)
    error_deg_history.append(np.rad2deg(np.linalg.norm(quat_error_vec(q2_true, q2_current))))

    print(f"  current q: {q2_current}")

    if np.linalg.norm(dx) < CONVERGENCE_TOL_RAD:
        print("  Converged: update is below tolerance.")
        break

print()
print(f"Final q2: {q2_current}")
print_error_summary("Final", q2_true, q2_current)


# -----------------------------------------------------------------------------
# Optional teaching diagnostics
# -----------------------------------------------------------------------------

J_final, r_final = build_tall_system(q2_current, q1_samples, noisy_measurements)
print("\nFinal stacked system diagnostics")
print("--------------------------------")
print(f"J_final shape: {J_final.shape}")
print(f"r_final shape: {r_final.shape}")
print(f"rank(J_final): {np.linalg.matrix_rank(J_final)}")
print(f"cond(J_final): {np.linalg.cond(J_final):.6f}")
print("\nFirst 3x3 Jacobian block, J_0 = R(q1_0)^T:")
print(J_final[:3, :])
print("\nFirst residual block, r_0:")
print(r_final[:3])


# -----------------------------------------------------------------------------
# Matplotlib animation
# -----------------------------------------------------------------------------

if SHOW_PLOT:
    build_animation(
        q2_true=q2_true,
        q2_initial=q2_initial,
        q_history=q_history,
        residual_rms_deg_history=residual_rms_deg_history,
        dx_deg_history=dx_deg_history,
        error_deg_history=error_deg_history,
        J_final=J_final,
    )
