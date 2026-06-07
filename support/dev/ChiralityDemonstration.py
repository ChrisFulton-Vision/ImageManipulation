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

The demo uses a planar drogue-like ring of labeled feature points.
For a true front-facing pose:

    R_true = I
    t_true = [0, 0, +Z]

there is an invalid chirality twin:

    R_bad = Rz(pi)
    t_bad = [0, 0, -Z]

For planar points with z=0, both poses produce identical image projections,
but the bad solution places every feature behind the camera.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


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


def rotz(theta_rad: float) -> np.ndarray:
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)

    return np.array(
        [
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def make_drogue_points(drogue: Drogue) -> np.ndarray:
    """
    Returns Nx3 planar feature points arranged in a ring.
    The points are intentionally labeled by index.

    Object frame:
        x-y plane is the drogue face.
        z=0 for all features.
    """

    theta = np.linspace(0.0, 2.0 * np.pi, drogue.n_points, endpoint=False)

    points = np.column_stack(
        [
            drogue.radius_m * np.cos(theta),
            drogue.radius_m * np.sin(theta),
            np.zeros_like(theta),
        ]
    )

    return points


def transform_points(points_obj: np.ndarray, R_obj_to_cam: np.ndarray, t_obj_in_cam: np.ndarray) -> np.ndarray:
    return (R_obj_to_cam @ points_obj.T).T + t_obj_in_cam.reshape(1, 3)


def project_points(
    points_obj: np.ndarray,
    R_obj_to_cam: np.ndarray,
    t_obj_in_cam: np.ndarray,
    camera: Camera,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Projects object-frame points into pixels.

    Returns:
        pixels: Nx2 array
        points_cam: Nx3 array
    """

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
    """
    Jacobian of pixel residuals with respect to translation only.

    For one point:

        u = fx X/Z + cx
        v = fy Y/Z + cy

    Therefore:

        du/dtx = fx/Z
        du/dty = 0
        du/dtz = -fx X/Z^2

        dv/dtx = 0
        dv/dty = fy/Z
        dv/dtz = -fy Y/Z^2
    """

    points_cam = transform_points(points_obj, R_obj_to_cam, t_obj_in_cam)

    X = points_cam[:, 0]
    Y = points_cam[:, 1]
    Z = points_cam[:, 2]

    J = np.zeros((2 * len(points_obj), 3), dtype=float)

    J[0::2, 0] = camera.fx / Z
    J[0::2, 1] = 0.0
    J[0::2, 2] = -camera.fx * X / (Z * Z)

    J[1::2, 0] = 0.0
    J[1::2, 1] = camera.fy / Z
    J[1::2, 2] = -camera.fy * Y / (Z * Z)

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
    """
    Computes a translation-space descent direction from the residual Jacobian.

    method:
        "gradient":
            direction = -J.T r

        "gauss_newton":
            direction = -(J.T J + lambda I)^-1 J.T r

    Returns:
        direction: 3-vector
        cost: 0.5 * r.T r
        min_depth: minimum feature depth in camera frame
    """

    r = residual_vector(points_obj, observed_pixels, R_obj_to_cam, t_obj_in_cam, camera)
    J = translation_jacobian(points_obj, R_obj_to_cam, t_obj_in_cam, camera)

    cost = 0.5 * float(r @ r)

    _, points_cam = project_points(points_obj, R_obj_to_cam, t_obj_in_cam, camera)
    min_depth = float(np.min(points_cam[:, 2]))

    if method == "gradient":
        direction = -J.T @ r

    elif method == "gauss_newton":
        H = J.T @ J + damping * np.eye(3)
        g = J.T @ r
        direction = -np.linalg.solve(H, g)

    else:
        raise ValueError(f"Unknown method: {method}")

    return direction, cost, min_depth


def make_vector_field(
    points_obj: np.ndarray,
    observed_pixels: np.ndarray,
    R_branch: np.ndarray,
    camera: Camera,
    x_range: tuple[float, float] = (-1.25, 1.25),
    y_range: tuple[float, float] = (-1.25, 1.25),
    z_values: np.ndarray | None = None,
    n_xy: int = 7,
    method: str = "gauss_newton",
    max_arrow_length: float = 0.35,
) -> dict[str, np.ndarray]:
    """
    Builds a 3D vector field over candidate translations.

    The rotation is held fixed to one branch. This lets us compare:

        valid branch:   R = I
        invalid branch: R = Rz(pi)

    The plotted arrows show the translation update direction recommended
    by the residual Jacobian.
    """

    if z_values is None:
        z_values = np.array([-9.0, -7.5, -6.0, -4.5, -3.0, 3.0, 4.5, 6.0, 7.5, 9.0])

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

                # Skip poses too close to the projection singularity.
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


def draw_drogue_pose(
    ax,
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

    # Draw ring connection.
    ring = np.vstack([points_cam, points_cam[0]])
    ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], color=color, linewidth=1.5)


def plot_projection_overlay(
    points_obj: np.ndarray,
    camera: Camera,
    R_true: np.ndarray,
    t_true: np.ndarray,
    R_bad: np.ndarray,
    t_bad: np.ndarray,
    output_path: str | None = "projection_overlay.png",
) -> None:
    pixels_true, cam_true = project_points(points_obj, R_true, t_true, camera)
    pixels_bad, cam_bad = project_points(points_obj, R_bad, t_bad, camera)

    print("Projection overlay check")
    print("------------------------")
    print(f"max pixel difference: {np.max(np.linalg.norm(pixels_true - pixels_bad, axis=1)):.6e} px")
    print(f"true min depth:       {np.min(cam_true[:, 2]):.6f} m")
    print(f"bad min depth:        {np.min(cam_bad[:, 2]):.6f} m")
    print()

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(pixels_true[:, 0], pixels_true[:, 1], s=90, label="true valid pose")
    ax.scatter(
        pixels_bad[:, 0],
        pixels_bad[:, 1],
        s=35,
        marker="x",
        label="invalid chirality twin",
    )

    for i, p in enumerate(pixels_true):
        ax.text(p[0] + 4.0, p[1] + 4.0, str(i), fontsize=9)

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

    plt.show()


def plot_chirality_vector_field(
    output_path: str | None = "chirality_vector_field.png",
    method: str = "gauss_newton",
) -> None:
    camera = Camera()
    drogue = Drogue(radius_m=0.45, n_points=12)
    points_obj = make_drogue_points(drogue)

    # Slightly off-center true pose so the geometry is not perfectly symmetric.
    # Chosen so the valid pose appears on the left side of the figure.
    tx_true = -0.35
    ty_true = -0.20
    z_true = 6.0

    R_true = np.eye(3)
    t_true = np.array([tx_true, ty_true, z_true])

    # Planar chirality twin:
    # same image projections, but negative depth for all points
    R_bad = rotz(np.pi)
    t_bad = np.array([-tx_true, -ty_true, -z_true])

    observed_pixels, _ = project_points(points_obj, R_true, t_true, camera)

    # Sample each branch only near its own basin.
    field_true = make_vector_field(
        points_obj=points_obj,
        observed_pixels=observed_pixels,
        R_branch=R_true,
        camera=camera,
        x_range=(-1.2, 0.3),
        y_range=(-0.9, 0.4),
        z_values=np.linspace(2.0, 9.0, 6),
        n_xy=5,
        method=method,
        max_arrow_length=0.22,
    )

    field_bad = make_vector_field(
        points_obj=points_obj,
        observed_pixels=observed_pixels,
        R_branch=R_bad,
        camera=camera,
        x_range=(-0.3, 1.2),
        y_range=(-0.4, 0.9),
        z_values=np.linspace(-9.0, -2.0, 6),
        n_xy=5,
        method=method,
        max_arrow_length=0.22,
    )

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    # -----------------------------
    # Valid branch
    # -----------------------------
    P1 = field_true["positions"]
    D1 = field_true["directions"]
    C1 = np.log10(field_true["costs"] + 1.0)

    scatter1 = ax.scatter(
        P1[:, 0],
        P1[:, 1],
        P1[:, 2],
        c=C1,
        cmap="Greens",
        s=14,
        alpha=0.38,
        label="valid-branch samples",
    )

    ax.quiver(
        P1[:, 0],
        P1[:, 1],
        P1[:, 2],
        D1[:, 0],
        D1[:, 1],
        D1[:, 2],
        length=1.0,
        normalize=False,
        color="tab:green",
        linewidth=0.75,
        alpha=0.75,
    )

    draw_drogue_pose(
        ax=ax,
        points_obj=points_obj,
        R_obj_to_cam=R_true,
        t_obj_in_cam=t_true,
        label="valid pose",
        color="tab:green",
    )

    # -----------------------------
    # Invalid chirality branch
    # -----------------------------
    P2 = field_bad["positions"]
    D2 = field_bad["directions"]
    C2 = np.log10(field_bad["costs"] + 1.0)

    scatter2 = ax.scatter(
        P2[:, 0],
        P2[:, 1],
        P2[:, 2],
        c=C2,
        cmap="Reds",
        s=14,
        alpha=0.38,
        label="invalid-branch samples",
    )

    ax.quiver(
        P2[:, 0],
        P2[:, 1],
        P2[:, 2],
        D2[:, 0],
        D2[:, 1],
        D2[:, 2],
        length=1.0,
        normalize=False,
        color="tab:red",
        linewidth=0.75,
        alpha=0.75,
    )

    draw_drogue_pose(
        ax=ax,
        points_obj=points_obj,
        R_obj_to_cam=R_bad,
        t_obj_in_cam=t_bad,
        label="invalid chirality twin",
        color="tab:red",
    )

    # -----------------------------
    # Camera and optical axis
    # -----------------------------
    ax.scatter(
        [0.0], [0.0], [0.0],
        s=100,
        marker="^",
        color="tab:blue",
        label="camera",
    )

    # Draw the camera optical axis (+Z in the camera frame)
    ax.plot(
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 8.0],
        linestyle="--",
        linewidth=2.0,
        color="tab:blue",
        alpha=0.8,
    )
    ax.text(0.0, 0.0, 8.3, "+Z optical axis", color="tab:blue")

    # -----------------------------
    # Formatting
    # -----------------------------
    ax.set_title(
        "PnP / QnP chirality demo: valid and invalid residual basins on one plot"
    )
    ax.set_xlabel("candidate $t_x$ [m]")
    ax.set_ylabel("candidate $t_y$ [m]")
    ax.set_zlabel("candidate $t_z$ [m]")

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-10.0, 10.0)

    # This view usually makes the camera appear central and the geometry read left/right.
    # If the “left/right” appearance is reversed on your machine, just tweak azim slightly.
    ax.view_init(elev=12, azim=-100)

    ax.legend(loc="upper left")

    # Optional separate colorbars for each branch
    # cbar1 = fig.colorbar(scatter1, ax=ax, shrink=0.55, pad=0.02)
    # cbar1.set_label(r"valid branch: $\log_{10}(1 + \frac{1}{2} r^T r)$")

    # cbar2 = fig.colorbar(scatter2, ax=ax, shrink=0.55, pad=0.10)
    # cbar2.set_label(r"invalid branch: $\log_{10}(1 + \frac{1}{2} r^T r)$")

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    plt.show()

    plot_projection_overlay(
        points_obj=points_obj,
        camera=camera,
        R_true=R_true,
        t_true=t_true,
        R_bad=R_bad,
        t_bad=t_bad,
        output_path="projection_overlay.png",
    )


if __name__ == "__main__":
    plot_chirality_vector_field(
        output_path="chirality_vector_field.png",
        method="gauss_newton",
    )