"""
chirality_vector_field_demo.py

Demonstrates a chirality failure mode for planar PnP-style pose estimation.

Convention:
    OpenCV-style camera frame:
        +X right
        +Y down
        +Z forward

Pose convention:
    X_cam = R_obj_to_cam @ X_obj + t_obj_in_cam

For purely planar object points with z_obj = 0, the image depends only on the first
two rotation columns and on translation:

    X_cam = r1 * x_obj + r2 * y_obj + t

This gives an exact anti-chiral twin for any proper rotation R:

    R_bad = R_true @ Rz(pi)
    t_bad = -t_true

because the planar camera-frame points become X_cam_bad = -X_cam_true. Image
projections match exactly, but the twin places the entire object behind the
camera.

This demo now uses a slightly non-planar target: an inner ring is recessed
behind the outer ring. That breaks the exact degeneracy, but the mirrored
pose remains a useful anti-chiral seed and typically converges to a nearby
back-side local minimum.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from support.mathHelpers.quaternions import Quaternion


@dataclass(frozen=True)
class Camera:
    fx: float = 900.0
    fy: float = 900.0
    cx: float = 432.0
    cy: float = 432.0


@dataclass(frozen=True)
class Drogue:
    radius_m: float = 0.45
    n_points: int = 12
    inner_radius_m: float = 0.22
    n_inner_points: int = 8
    inner_phase_deg: float = 22.5
    inner_z_m: float = -0.08


@dataclass(frozen=True)
class Pose:
    R_obj_to_cam: np.ndarray
    t_obj_in_cam: np.ndarray


def rotz(theta_rad: float) -> np.ndarray:
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)

    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def make_drogue_points(drogue: Drogue) -> np.ndarray:
    theta_outer = np.linspace(0.0, 2.0 * np.pi, drogue.n_points, endpoint=False)
    outer_points = np.column_stack(
        [
            drogue.radius_m * np.cos(theta_outer),
            drogue.radius_m * np.sin(theta_outer),
            np.zeros_like(theta_outer),
        ]
    )

    theta_inner = np.linspace(0.0, 2.0 * np.pi, drogue.n_inner_points, endpoint=False)
    theta_inner = theta_inner + np.deg2rad(drogue.inner_phase_deg)
    inner_points = np.column_stack(
        [
            drogue.inner_radius_m * np.cos(theta_inner),
            drogue.inner_radius_m * np.sin(theta_inner),
            np.full_like(theta_inner, drogue.inner_z_m),
        ]
    )

    return np.vstack([outer_points, inner_points])


def drogue_ring_slices(drogue: Drogue) -> tuple[slice, slice]:
    outer = slice(0, drogue.n_points)
    inner = slice(drogue.n_points, drogue.n_points + drogue.n_inner_points)
    return outer, inner


def rotation_from_rpy_deg(rpy_deg: tuple[float, float, float] | np.ndarray) -> np.ndarray:
    quat = Quaternion().from_eulerD_rpy(np.asarray(rpy_deg, dtype=float).reshape(3))
    return quat.to_dcm()


def transform_points(points_obj: np.ndarray, R_obj_to_cam: np.ndarray, t_obj_in_cam: np.ndarray) -> np.ndarray:
    return (R_obj_to_cam @ points_obj.T).T + t_obj_in_cam.reshape(1, 3)


def project_points(
    points_obj: np.ndarray,
    R_obj_to_cam: np.ndarray,
    t_obj_in_cam: np.ndarray,
    camera: Camera,
) -> tuple[np.ndarray, np.ndarray]:
    points_cam = transform_points(points_obj, R_obj_to_cam, t_obj_in_cam)

    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]

    pixels = np.column_stack(
        [
            camera.fx * x / z + camera.cx,
            camera.fy * y / z + camera.cy,
        ]
    )

    return pixels, points_cam


def residual_vector(
    points_obj: np.ndarray,
    observed_pixels: np.ndarray,
    R_obj_to_cam: np.ndarray,
    t_obj_in_cam: np.ndarray,
    camera: Camera,
) -> np.ndarray:
    predicted_pixels, _ = project_points(points_obj, R_obj_to_cam, t_obj_in_cam, camera)
    return (predicted_pixels - observed_pixels).reshape(-1)


def translation_jacobian(
    points_obj: np.ndarray,
    R_obj_to_cam: np.ndarray,
    t_obj_in_cam: np.ndarray,
    camera: Camera,
) -> np.ndarray:
    points_cam = transform_points(points_obj, R_obj_to_cam, t_obj_in_cam)

    X = points_cam[:, 0]
    Y = points_cam[:, 1]
    Z = points_cam[:, 2]

    J = np.zeros((2 * len(points_obj), 3), dtype=float)

    J[0::2, 0] = camera.fx / Z
    J[0::2, 2] = -camera.fx * X / (Z * Z)

    J[1::2, 1] = camera.fy / Z
    J[1::2, 2] = -camera.fy * Y / (Z * Z)

    return J


def pose_jacobian(
    points_obj: np.ndarray,
    R_obj_to_cam: np.ndarray,
    t_obj_in_cam: np.ndarray,
    camera: Camera,
) -> np.ndarray:
    points_cam = transform_points(points_obj, R_obj_to_cam, t_obj_in_cam)
    RX = points_cam - t_obj_in_cam.reshape(1, 3)

    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]

    invz = 1.0 / z
    invz2 = invz * invz

    uX = camera.fx * invz
    uZ = -camera.fx * x * invz2
    vY = camera.fy * invz
    vZ = -camera.fy * y * invz2

    a = RX[:, 0]
    b = RX[:, 1]
    c = RX[:, 2]

    J = np.zeros((2 * len(points_obj), 6), dtype=float)

    J[0::2, 0] = uZ * b
    J[0::2, 1] = uX * c - uZ * a
    J[0::2, 2] = -uX * b
    J[0::2, 3] = uX
    J[0::2, 5] = uZ

    J[1::2, 0] = -vY * c + vZ * b
    J[1::2, 1] = -vZ * a
    J[1::2, 2] = vY * a
    J[1::2, 4] = vY
    J[1::2, 5] = vZ

    return J


def descent_direction_translation(
    points_obj: np.ndarray,
    observed_pixels: np.ndarray,
    R_obj_to_cam: np.ndarray,
    t_obj_in_cam: np.ndarray,
    camera: Camera,
    method: str = "gauss_newton",
    damping: float = 1.0e-6,
) -> tuple[np.ndarray, float, float]:
    r = residual_vector(points_obj, observed_pixels, R_obj_to_cam, t_obj_in_cam, camera)
    J = translation_jacobian(points_obj, R_obj_to_cam, t_obj_in_cam, camera)
    cost = 0.5 * float(r @ r)

    _, points_cam = project_points(points_obj, R_obj_to_cam, t_obj_in_cam, camera)
    min_depth = float(np.min(points_cam[:, 2]))

    if method == "gradient":
        direction = -J.T @ r
    elif method == "gauss_newton":
        H = J.T @ J + damping * np.eye(3)
        direction = -np.linalg.solve(H, J.T @ r)
    else:
        raise ValueError(f"Unknown method: {method}")

    return direction, cost, min_depth


def apply_pose_delta(
    R_obj_to_cam: np.ndarray,
    t_obj_in_cam: np.ndarray,
    delta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dtheta = np.asarray(delta[:3], dtype=float).reshape(3)
    dt = np.asarray(delta[3:], dtype=float).reshape(3)
    dR = Quaternion.exp_so3(dtheta).to_dcm()
    return dR @ R_obj_to_cam, t_obj_in_cam + dt


def solve_pose_lm(
    points_obj: np.ndarray,
    observed_pixels: np.ndarray,
    camera: Camera,
    seed_pose: Pose,
    *,
    max_iters: int = 60,
    damping: float = 1.0e-2,
    step_tol: float = 1.0e-10,
    cost_tol: float = 1.0e-12,
) -> tuple[Pose, dict[str, float | int | bool]]:
    R_est = np.asarray(seed_pose.R_obj_to_cam, dtype=float).copy()
    t_est = np.asarray(seed_pose.t_obj_in_cam, dtype=float).reshape(3).copy()

    lam = float(damping)
    prev_cost = np.inf
    converged = False
    last_step_norm = np.inf

    for it in range(1, max_iters + 1):
        residual = residual_vector(points_obj, observed_pixels, R_est, t_est, camera)
        cost = 0.5 * float(residual @ residual)
        J = pose_jacobian(points_obj, R_est, t_est, camera)

        H = J.T @ J
        g = J.T @ residual

        step = -np.linalg.solve(H + lam * np.eye(6), g)
        R_trial, t_trial = apply_pose_delta(R_est, t_est, step)
        residual_trial = residual_vector(points_obj, observed_pixels, R_trial, t_trial, camera)
        cost_trial = 0.5 * float(residual_trial @ residual_trial)

        if np.isfinite(cost_trial) and cost_trial < cost:
            R_est = R_trial
            t_est = t_trial
            last_step_norm = float(np.linalg.norm(step))
            rel_cost = abs(cost - cost_trial) / max(1.0, cost)
            lam = max(1.0e-10, lam * 0.3)

            if last_step_norm < step_tol or rel_cost < cost_tol:
                converged = True
                prev_cost = cost_trial
                break

            prev_cost = cost_trial
        else:
            lam = min(1.0e8, lam * 4.0)
            prev_cost = cost

    pose = Pose(R_obj_to_cam=R_est, t_obj_in_cam=t_est)
    _, points_cam = project_points(points_obj, pose.R_obj_to_cam, pose.t_obj_in_cam, camera)
    info = {
        "iterations": it,
        "converged": converged,
        "cost": float(prev_cost),
        "step_norm": float(last_step_norm),
        "min_depth": float(np.min(points_cam[:, 2])),
        "max_depth": float(np.max(points_cam[:, 2])),
    }
    return pose, info


def make_mirrored_antichiral_seed(true_pose: Pose) -> Pose:
    return Pose(
        R_obj_to_cam=true_pose.R_obj_to_cam @ rotz(np.pi),
        t_obj_in_cam=-np.asarray(true_pose.t_obj_in_cam, dtype=float),
    )


def converge_antichiral_pose(
    points_obj: np.ndarray,
    observed_pixels: np.ndarray,
    camera: Camera,
    true_pose: Pose,
    *,
    seed_rotation_perturb_deg: tuple[float, float, float] = (8.0, -6.0, 12.0),
    seed_translation_perturb_m: tuple[float, float, float] = (0.12, -0.08, 0.35),
) -> tuple[Pose, dict[str, float | int | bool], Pose]:
    mirrored_seed = make_mirrored_antichiral_seed(true_pose)
    seed_R = rotation_from_rpy_deg(seed_rotation_perturb_deg) @ mirrored_seed.R_obj_to_cam
    seed_t = mirrored_seed.t_obj_in_cam + np.asarray(seed_translation_perturb_m, dtype=float)

    solved_pose, solve_info = solve_pose_lm(
        points_obj=points_obj,
        observed_pixels=observed_pixels,
        camera=camera,
        seed_pose=Pose(R_obj_to_cam=seed_R, t_obj_in_cam=seed_t),
    )
    return solved_pose, solve_info, mirrored_seed


def make_vector_field(
    points_obj: np.ndarray,
    observed_pixels: np.ndarray,
    R_branch: np.ndarray,
    camera: Camera,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_values: np.ndarray,
    n_xy: int = 7,
    method: str = "gauss_newton",
    max_arrow_length: float = 0.35,
) -> dict[str, np.ndarray]:
    xs = np.linspace(*x_range, n_xy)
    ys = np.linspace(*y_range, n_xy)

    positions = []
    directions = []
    costs = []
    min_depths = []

    for z in z_values:
        for y in ys:
            for x in xs:
                t = np.array([x, y, z], dtype=float)
                _, pts_cam = project_points(points_obj, R_branch, t, camera)

                if np.min(np.abs(pts_cam[:, 2])) < 0.25:
                    continue

                direction, cost, min_depth = descent_direction_translation(
                    points_obj=points_obj,
                    observed_pixels=observed_pixels,
                    R_obj_to_cam=R_branch,
                    t_obj_in_cam=t,
                    camera=camera,
                    method=method,
                )

                norm = np.linalg.norm(direction)
                if not np.isfinite(norm) or norm < 1.0e-12:
                    direction_scaled = np.zeros(3)
                else:
                    direction_scaled = direction / norm * min(norm, max_arrow_length)

                positions.append(t)
                directions.append(direction_scaled)
                costs.append(cost)
                min_depths.append(min_depth)

    return {
        "positions": np.asarray(positions),
        "directions": np.asarray(directions),
        "costs": np.asarray(costs),
        "min_depths": np.asarray(min_depths),
    }


def branch_ranges(center: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], np.ndarray]:
    center = np.asarray(center, dtype=float).reshape(3)
    x_range = (center[0] - 0.8, center[0] + 0.8)
    y_range = (center[1] - 0.7, center[1] + 0.7)
    z_values = np.linspace(center[2] - 3.0, center[2] + 3.0, 6)
    return x_range, y_range, z_values


def draw_drogue_pose(
    ax,
    drogue: Drogue,
    points_obj: np.ndarray,
    R_obj_to_cam: np.ndarray,
    t_obj_in_cam: np.ndarray,
    label: str,
    color: str,
) -> None:
    points_cam = transform_points(points_obj, R_obj_to_cam, t_obj_in_cam)

    ax.scatter(
        points_cam[:, 0],
        points_cam[:, 1],
        points_cam[:, 2],
        s=35,
        color=color,
        label=label,
    )

    outer_slice, inner_slice = drogue_ring_slices(drogue)
    for ring_points in (points_cam[outer_slice], points_cam[inner_slice]):
        ring = np.vstack([ring_points, ring_points[0]])
        ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], color=color, linewidth=1.5)


def plot_projection_overlay(
    drogue: Drogue,
    points_obj: np.ndarray,
    camera: Camera,
    true_pose: Pose,
    bad_pose: Pose,
    output_path: str | None = "projection_overlay.png",
    show: bool = True,
) -> None:
    pixels_true, cam_true = project_points(points_obj, true_pose.R_obj_to_cam, true_pose.t_obj_in_cam, camera)
    pixels_bad, cam_bad = project_points(points_obj, bad_pose.R_obj_to_cam, bad_pose.t_obj_in_cam, camera)

    print("Projection overlay check")
    print("------------------------")
    print(f"max pixel difference: {np.max(np.linalg.norm(pixels_true - pixels_bad, axis=1)):.6e} px")
    print(f"true min depth:       {np.min(cam_true[:, 2]):.6f} m")
    print(f"bad max depth:        {np.max(cam_bad[:, 2]):.6f} m")
    print()

    fig, ax = plt.subplots(figsize=(7, 7))
    outer_slice, inner_slice = drogue_ring_slices(drogue)
    ax.scatter(pixels_true[outer_slice, 0], pixels_true[outer_slice, 1], s=90, label="true valid pose")
    ax.scatter(pixels_true[inner_slice, 0], pixels_true[inner_slice, 1], s=55, label="true inner ring")
    ax.scatter(pixels_bad[:, 0], pixels_bad[:, 1], s=35, marker="x", label="invalid chirality twin")

    for i, p in enumerate(pixels_true[outer_slice]):
        ax.text(p[0] + 4.0, p[1] + 4.0, str(i), fontsize=9)
    for i, p in enumerate(pixels_true[inner_slice], start=drogue.n_points):
        ax.text(p[0] + 4.0, p[1] + 4.0, str(i), fontsize=8)

    for ring_pixels in (pixels_true[outer_slice], pixels_true[inner_slice]):
        ring = np.vstack([ring_pixels, ring_pixels[0]])
        ax.plot(ring[:, 0], ring[:, 1], linewidth=1.0, alpha=0.8)

    ax.set_title("Same image residual, opposite chirality")
    ax.set_xlabel("u [px]")
    ax.set_ylabel("v [px]")
    ax.set_xlim([0, 864])
    ax.set_ylim([0, 864])
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.grid(True)
    ax.legend()

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_chirality_vector_field(
    output_path: str | None = "chirality_vector_field.png",
    method: str = "gauss_newton",
    *,
    true_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    true_translation_m: tuple[float, float, float] = (-0.35, -0.20, 6.0),
    show: bool = True,
) -> dict[str, Pose | dict[str, float | int | bool]]:
    camera = Camera()
    drogue = Drogue(radius_m=0.45, n_points=12)
    points_obj = make_drogue_points(drogue)

    true_pose = Pose(
        R_obj_to_cam=rotation_from_rpy_deg(true_rpy_deg),
        t_obj_in_cam=np.asarray(true_translation_m, dtype=float),
    )

    observed_pixels, _ = project_points(points_obj, true_pose.R_obj_to_cam, true_pose.t_obj_in_cam, camera)
    bad_pose, bad_info, mirrored_seed = converge_antichiral_pose(
        points_obj=points_obj,
        observed_pixels=observed_pixels,
        camera=camera,
        true_pose=true_pose,
    )

    true_x_range, true_y_range, true_z_values = branch_ranges(true_pose.t_obj_in_cam)
    bad_x_range, bad_y_range, bad_z_values = branch_ranges(bad_pose.t_obj_in_cam)

    field_true = make_vector_field(
        points_obj=points_obj,
        observed_pixels=observed_pixels,
        R_branch=true_pose.R_obj_to_cam,
        camera=camera,
        x_range=true_x_range,
        y_range=true_y_range,
        z_values=true_z_values,
        n_xy=5,
        method=method,
        max_arrow_length=0.22,
    )

    field_bad = make_vector_field(
        points_obj=points_obj,
        observed_pixels=observed_pixels,
        R_branch=bad_pose.R_obj_to_cam,
        camera=camera,
        x_range=bad_x_range,
        y_range=bad_y_range,
        z_values=bad_z_values,
        n_xy=5,
        method=method,
        max_arrow_length=0.22,
    )

    print("True pose")
    print("---------")
    print(f"rpy_deg:   {np.asarray(true_rpy_deg)}")
    print(f"t_true:    {true_pose.t_obj_in_cam}")
    print()

    print("Anti-chiral branch")
    print("------------------")
    print(f"mirrored seed t:{mirrored_seed.t_obj_in_cam}")
    print(f"LM twin t:     {bad_pose.t_obj_in_cam}")
    print(f"LM converged:  {bad_info['converged']}")
    print(f"LM iterations: {bad_info['iterations']}")
    print(f"LM cost:       {bad_info['cost']:.6e}")
    print(f"bad max depth: {bad_info['max_depth']:.6f} m")
    print()

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    P1 = field_true["positions"]
    D1 = field_true["directions"]
    C1 = np.log10(field_true["costs"] + 1.0)
    ax.scatter(P1[:, 0], P1[:, 1], P1[:, 2], c=C1, cmap="Greens", s=14, alpha=0.38, label="valid-branch samples")
    ax.quiver(
        P1[:, 0], P1[:, 1], P1[:, 2],
        D1[:, 0], D1[:, 1], D1[:, 2],
        length=1.0,
        normalize=False,
        color="tab:green",
        linewidth=0.75,
        alpha=0.75,
    )
    draw_drogue_pose(ax, drogue, points_obj, true_pose.R_obj_to_cam, true_pose.t_obj_in_cam, "valid pose", "tab:green")

    P2 = field_bad["positions"]
    D2 = field_bad["directions"]
    C2 = np.log10(field_bad["costs"] + 1.0)
    ax.scatter(P2[:, 0], P2[:, 1], P2[:, 2], c=C2, cmap="Reds", s=14, alpha=0.38, label="invalid-branch samples")
    ax.quiver(
        P2[:, 0], P2[:, 1], P2[:, 2],
        D2[:, 0], D2[:, 1], D2[:, 2],
        length=1.0,
        normalize=False,
        color="tab:red",
        linewidth=0.75,
        alpha=0.75,
    )
    draw_drogue_pose(ax, drogue, points_obj, bad_pose.R_obj_to_cam, bad_pose.t_obj_in_cam, "invalid chirality twin", "tab:red")

    ax.scatter([0.0], [0.0], [0.0], s=100, marker="^", color="tab:blue", label="camera")
    ax.plot([0.0, 0.0], [0.0, 0.0], [0.0, 8.0], linestyle="--", linewidth=2.0, color="tab:blue", alpha=0.8)
    ax.text(0.0, 0.0, 8.3, "+Z optical axis", color="tab:blue")

    ax.set_title("PnP / QnP chirality demo with arbitrary true rotation")
    ax.set_xlabel("candidate $t_x$ [m]")
    ax.set_ylabel("candidate $t_y$ [m]")
    ax.set_zlabel("candidate $t_z$ [m]")

    stacked_positions = np.vstack([P1, P2, true_pose.t_obj_in_cam.reshape(1, 3), bad_pose.t_obj_in_cam.reshape(1, 3)])
    mins = stacked_positions.min(axis=0)
    maxs = stacked_positions.max(axis=0)
    pad = np.array([0.45, 0.45, 0.8])
    ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
    ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])
    ax.set_zlim(mins[2] - pad[2], maxs[2] + pad[2])
    ax.view_init(elev=12, azim=-100)
    ax.legend(loc="upper left")
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    plot_projection_overlay(
        drogue=drogue,
        points_obj=points_obj,
        camera=camera,
        true_pose=true_pose,
        bad_pose=bad_pose,
        output_path="projection_overlay.png" if output_path is not None else None,
        show=show,
    )

    return {
        "true_pose": true_pose,
        "bad_pose": bad_pose,
        "bad_info": bad_info,
        "mirrored_seed": mirrored_seed,
    }


if __name__ == "__main__":
    plot_chirality_vector_field(
        output_path="chirality_vector_field.png",
        method="gauss_newton",
        true_rpy_deg=(0.0, -45.0, 35.0),
    )
