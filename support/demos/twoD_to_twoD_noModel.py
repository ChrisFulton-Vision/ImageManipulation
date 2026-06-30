import cv2
import numpy as np
from support.vision.calibration import default_864_cam, Calibration
import support.mathHelpers.quaternions as quat
import support.mathHelpers.quaternions as q
from support.mathHelpers.SE3 import SE3_q
from support.mathHelpers.LevMarq import LevenbergMarquardt
from support.mathHelpers.SE3TwoDtoTwoD_noModel import SE3TwoDtoTwoD_noModel
from support.runtime.pixel_handler import Pixel as pxl
from dataclasses import dataclass, field

np.random.seed(42)


@dataclass
class Sensor:
    camCal: Calibration = field(default_factory=lambda: default_864_cam().randomize())
    SE3: SE3_q = field(default_factory=SE3_q)

    def __post_init__(self):
        """Randomize the sensor pose after default construction."""
        self.SE3.quat = quat.random_quat_within_deg(20.0)
        self.SE3.tvec = np.random.normal(loc=0, scale=1, size=3) + np.array([0, 0, -5.0])


@dataclass
class Feature:
    tvec: np.ndarray = field(default_factory=lambda: np.random.rand(3,))

    def __repr__(self):
        """Return a compact xyz string for quick debugging output."""
        return f'x:{self.tvec[0]:.3f}, y:{self.tvec[1]:.3f}, z:{self.tvec[2]:.3f}\n'


def generate_sensors() -> tuple[Sensor, Sensor]:
    """Create a pair of randomized sensors with independent calibration and pose."""
    cam1 = Sensor()
    cam2 = Sensor()
    return cam1, cam2


def generate_features(num_features: int = 50) -> tuple[Feature, ...]:
    """Create synthetic world points used by both cameras."""
    all_features: tuple[Feature, ...] = tuple(Feature() for _ in range(num_features))
    return all_features


def create_image(cam: Sensor, features: tuple[Feature, ...]) -> np.ndarray:
    """Render visible world features into an undistorted image for one camera."""
    shape = (cam.camCal.height, cam.camCal.width, 3)
    image = np.zeros(shape, dtype=np.uint8)

    for feat in features:
        point_cam = cam.SE3.inv * feat.tvec
        if point_cam[2] <= 0:
            continue

        norm_xy = point_cam[:2] / point_cam[2]
        px_point = pxl(norm_coords=norm_xy, already_undistorted=True)
        cam.camCal.haveNorm_needPix(px_point)

        x, y = px_point.pix_coords
        xi = int(round(x))
        yi = int(round(y))

        if 0 <= xi < cam.camCal.width and 0 <= yi < cam.camCal.height:
            image[yi-2:yi+2, xi-2:xi+2, 1:] = 255

    return image


def draw_square(image: np.ndarray, x: int, y: int, color: tuple[int, int, int], half_size: int = 2) -> None:
    """Draw a filled square centered on a pixel, clipped to image bounds."""
    h, w = image.shape[:2]
    x0 = max(0, x - half_size)
    x1 = min(w, x + half_size + 1)
    y0 = max(0, y - half_size)
    y1 = min(h, y + half_size + 1)
    image[y0:y1, x0:x1, :] = color


def project_point(cam: Sensor, point_world: np.ndarray) -> np.ndarray | None:
    """Project a world point into undistorted pixel coordinates for one camera."""
    point_cam = cam.SE3.inv * point_world
    if point_cam[2] <= 0:
        return None

    norm_xy = point_cam[:2] / point_cam[2]
    px_point = pxl(norm_coords=norm_xy, already_undistorted=True)
    cam.camCal.haveNorm_needPix(px_point)
    return np.asarray(px_point.pix_coords, dtype=float)


def build_measurements(
        cam1: Sensor,
        cam2: Sensor,
        features: tuple[Feature, ...],
        *,
        pixel_noise_std: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build noisy pixel correspondences for features visible in both cameras."""
    pixel_noise_std = float(pixel_noise_std)
    if pixel_noise_std < 0.0:
        raise ValueError("pixel_noise_std must be non-negative")

    pixels_cam1: list[np.ndarray] = []
    pixels_cam2: list[np.ndarray] = []

    for feat in features:
        pixel_cam1 = project_point(cam1, feat.tvec)
        pixel_cam2 = project_point(cam2, feat.tvec)

        if pixel_cam1 is None or pixel_cam2 is None:
            continue

        if pixel_noise_std > 0.0:
            pixel_cam1 = pixel_cam1 + np.random.normal(loc=0.0, scale=pixel_noise_std, size=2)
            pixel_cam2 = pixel_cam2 + np.random.normal(loc=0.0, scale=pixel_noise_std, size=2)

        x1, y1 = pixel_cam1
        x2, y2 = pixel_cam2
        if 0 <= x1 < cam1.camCal.width and 0 <= y1 < cam1.camCal.height and 0 <= x2 < cam2.camCal.width and 0 <= y2 < cam2.camCal.height:
            pixels_cam1.append(pixel_cam1)
            pixels_cam2.append(pixel_cam2)

    if not pixels_cam1:
        raise ValueError("No shared visible features between cam1 and cam2.")

    return np.asarray(pixels_cam1, dtype=float), np.asarray(pixels_cam2, dtype=float)


def pixel_to_bearing(cal: Calibration, pixel_xy: np.ndarray) -> np.ndarray:
    px_point = pxl(pix_coords=np.asarray(pixel_xy, dtype=float))
    cal.havePix_needNorm(px_point)
    bearing = np.array([px_point.norm_coords[0], px_point.norm_coords[1], 1.0], dtype=float)
    return bearing / np.linalg.norm(bearing)


def line_norm_to_pixel_endpoints(cam_cal: Calibration, line_norm: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]] | None:
    a, b, c = [float(v) for v in line_norm]
    candidates: list[tuple[float, float]] = []
    x_bounds = (0.0, float(cam_cal.width - 1))
    y_bounds = (0.0, float(cam_cal.height - 1))

    if abs(b) > 1.0e-12:
        for x in x_bounds:
            y = -(a * ((x - cam_cal.cx) / cam_cal.fx) + c) * cam_cal.fy / b + cam_cal.cy
            if 0.0 <= y <= y_bounds[1]:
                candidates.append((x, y))
    if abs(a) > 1.0e-12:
        for y in y_bounds:
            x = -(b * ((y - cam_cal.cy) / cam_cal.fy) + c) * cam_cal.fx / a + cam_cal.cx
            if 0.0 <= x <= x_bounds[1]:
                candidates.append((x, y))

    unique: list[tuple[int, int]] = []
    for x, y in candidates:
        point = (int(round(x)), int(round(y)))
        if point not in unique:
            unique.append(point)

    if len(unique) < 2:
        return None
    return unique[0], unique[1]


def point_to_epiline_distance_px(cam_cal: Calibration, pixel_xy: np.ndarray, line_norm: np.ndarray) -> float:
    x_norm = (float(pixel_xy[0]) - cam_cal.cx) / cam_cal.fx
    y_norm = (float(pixel_xy[1]) - cam_cal.cy) / cam_cal.fy
    a, b, c = [float(v) for v in line_norm]
    denom = np.hypot(a / cam_cal.fx, b / cam_cal.fy)
    if denom <= 1.0e-12:
        return float("inf")
    return abs(a * x_norm + b * y_norm + c) / denom


def project_cam_point_to_pixel(cam_cal: Calibration, point_cam: np.ndarray) -> np.ndarray | None:
    point_cam = np.asarray(point_cam, dtype=float).reshape(3)
    if point_cam[2] <= 1.0e-12:
        return None

    px_point = pxl(norm_coords=point_cam[:2] / point_cam[2], already_undistorted=True)
    cam_cal.haveNorm_needPix(px_point)
    return np.asarray(px_point.pix_coords, dtype=float)


def triangulate_midpoint_cam1(
        observed_pixels_cam1: np.ndarray,
        observed_pixels_cam2: np.ndarray,
        cam1_cal: Calibration,
        cam2_cal: Calibration,
        est_cam2_from_cam1: SE3_q,
) -> np.ndarray:
    bearings_cam1 = np.asarray([pixel_to_bearing(cam1_cal, px) for px in observed_pixels_cam1], dtype=float)
    bearings_cam2 = np.asarray([pixel_to_bearing(cam2_cal, px) for px in observed_pixels_cam2], dtype=float)

    R_21 = est_cam2_from_cam1.quat.to_dcm()
    t_21 = est_cam2_from_cam1.tvec / np.linalg.norm(est_cam2_from_cam1.tvec)
    R_12 = R_21.T
    cam2_center_in_cam1 = -(R_12 @ t_21)
    rays_cam2_in_cam1 = (R_12 @ bearings_cam2.T).T

    points_cam1 = np.empty((len(observed_pixels_cam1), 3), dtype=float)
    for idx, (ray1, ray2) in enumerate(zip(bearings_cam1, rays_cam2_in_cam1)):
        rhs = cam2_center_in_cam1
        A = np.column_stack((ray1, -ray2))
        lambdas, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
        closest_cam1 = lambdas[0] * ray1
        closest_cam2 = cam2_center_in_cam1 + lambdas[1] * ray2
        points_cam1[idx] = 0.5 * (closest_cam1 + closest_cam2)

    return points_cam1


def summarize_triangulation(
        triangulated_points_cam1: np.ndarray,
        observed_pixels_cam1: np.ndarray,
        observed_pixels_cam2: np.ndarray,
        cam1_cal: Calibration,
        cam2_cal: Calibration,
        est_cam2_from_cam1: SE3_q,
) -> dict[str, np.ndarray | float]:
    predicted_pixels_cam1 = np.empty_like(observed_pixels_cam1)
    predicted_pixels_cam2 = np.empty_like(observed_pixels_cam2)

    valid_mask = np.ones((len(triangulated_points_cam1),), dtype=bool)
    points_cam2 = est_cam2_from_cam1 * triangulated_points_cam1

    for idx, (point_cam1, point_cam2) in enumerate(zip(triangulated_points_cam1, points_cam2)):
        pred_px1 = project_cam_point_to_pixel(cam1_cal, point_cam1)
        pred_px2 = project_cam_point_to_pixel(cam2_cal, point_cam2)
        if pred_px1 is None or pred_px2 is None:
            valid_mask[idx] = False
            continue
        predicted_pixels_cam1[idx] = pred_px1
        predicted_pixels_cam2[idx] = pred_px2

    if not np.any(valid_mask):
        raise ValueError("No valid triangulated points remained in front of both cameras.")

    residuals_cam1 = predicted_pixels_cam1[valid_mask] - observed_pixels_cam1[valid_mask]
    residuals_cam2 = predicted_pixels_cam2[valid_mask] - observed_pixels_cam2[valid_mask]
    l2_cam1 = np.linalg.norm(residuals_cam1, axis=1)
    l2_cam2 = np.linalg.norm(residuals_cam2, axis=1)
    depth_cam1 = triangulated_points_cam1[valid_mask, 2]
    depth_cam2 = points_cam2[valid_mask, 2]

    return {
        "valid_mask": valid_mask,
        "points_cam2": points_cam2,
        "predicted_pixels_cam1": predicted_pixels_cam1,
        "predicted_pixels_cam2": predicted_pixels_cam2,
        "residuals_cam1": residuals_cam1,
        "residuals_cam2": residuals_cam2,
        "l2_cam1": l2_cam1,
        "l2_cam2": l2_cam2,
        "depth_cam1_values": depth_cam1,
        "depth_cam2_values": depth_cam2,
        "num_valid_points": float(np.count_nonzero(valid_mask)),
        "mean_reproj_cam1_px": float(np.mean(l2_cam1)),
        "rms_reproj_cam1_px": float(np.sqrt(np.mean(l2_cam1 * l2_cam1))),
        "max_reproj_cam1_px": float(np.max(l2_cam1)),
        "total_reproj_cam1_px": float(np.sum(l2_cam1)),
        "mean_reproj_cam2_px": float(np.mean(l2_cam2)),
        "rms_reproj_cam2_px": float(np.sqrt(np.mean(l2_cam2 * l2_cam2))),
        "max_reproj_cam2_px": float(np.max(l2_cam2)),
        "total_reproj_cam2_px": float(np.sum(l2_cam2)),
        "mean_depth_cam1": float(np.mean(depth_cam1)),
        "min_depth_cam1": float(np.min(depth_cam1)),
        "max_depth_cam1": float(np.max(depth_cam1)),
        "mean_depth_cam2": float(np.mean(depth_cam2)),
        "min_depth_cam2": float(np.min(depth_cam2)),
        "max_depth_cam2": float(np.max(depth_cam2)),
    }


def summarize_matching(
        observed_pixels_cam1: np.ndarray,
        observed_pixels_cam2: np.ndarray,
        cam1_cal: Calibration,
        cam2_cal: Calibration,
        est_cam2_from_cam1: SE3_q,
) -> dict[str, np.ndarray | float]:
    problem = SE3TwoDtoTwoD_noModel(
        observed_pixels_cam1=observed_pixels_cam1,
        observed_pixels_cam2=observed_pixels_cam2,
        cam1_cal=cam1_cal,
        cam2_cal=cam2_cal,
    )
    sampson_residuals = problem.residual(est_cam2_from_cam1)
    essential = q.skew(est_cam2_from_cam1.tvec / np.linalg.norm(est_cam2_from_cam1.tvec)) @ est_cam2_from_cam1.quat.to_dcm()

    line_distances_px = np.empty((len(observed_pixels_cam1),), dtype=float)
    for idx, (observed_px_cam1, observed_px_cam2) in enumerate(zip(observed_pixels_cam1, observed_pixels_cam2)):
        line_cam2 = essential @ pixel_to_bearing(cam1_cal, observed_px_cam1)
        line_distances_px[idx] = point_to_epiline_distance_px(cam2_cal, observed_px_cam2, line_cam2)

    squared_distances = line_distances_px * line_distances_px
    return {
        "sampson_residuals": sampson_residuals,
        "line_distances_px": line_distances_px,
        "mean_point_to_line_px": float(np.mean(line_distances_px)),
        "median_point_to_line_px": float(np.median(line_distances_px)),
        "rms_point_to_line_px": float(np.sqrt(np.mean(squared_distances))),
        "max_point_to_line_px": float(np.max(line_distances_px)),
        "total_point_to_line_px": float(np.sum(line_distances_px)),
        "total_squared_point_to_line_px": float(np.sum(squared_distances)),
        "mean_abs_sampson": float(np.mean(np.abs(sampson_residuals))),
        "rms_sampson": float(np.sqrt(np.mean(sampson_residuals * sampson_residuals))),
        "max_abs_sampson": float(np.max(np.abs(sampson_residuals))),
        "total_abs_sampson": float(np.sum(np.abs(sampson_residuals))),
    }


def optimize(
        observed_pixels_cam1: np.ndarray,
        observed_pixels_cam2: np.ndarray,
        cam1_cal: Calibration,
        cam2_cal: Calibration,
        seed_cam2_from_cam1: SE3_q | None = None,
        *,
        max_iters: int = 60,
        damping: float = 1.0e-2,
        step_tol: float = 1.0e-10,
        cost_tol: float = 1.0e-12,
) -> tuple[SE3_q, dict[str, float | int | bool]]:
    """Solve for cam2-from-cam1 from 2D-2D correspondences."""
    est = SE3_q() if seed_cam2_from_cam1 is None else seed_cam2_from_cam1.copy()
    t_norm = np.linalg.norm(est.tvec)
    est.tvec = np.array([1.0, 0.0, 0.0], dtype=float) if t_norm <= 1.0e-12 else est.tvec / t_norm
    problem = SE3TwoDtoTwoD_noModel(
        observed_pixels_cam1=np.asarray(observed_pixels_cam1, dtype=float),
        observed_pixels_cam2=np.asarray(observed_pixels_cam2, dtype=float),
        cam1_cal=cam1_cal,
        cam2_cal=cam2_cal,
    )

    solver = LevenbergMarquardt(
        state=est,
        problem=problem,
        damping_enabled=True,
        damping=float(damping),
        adaptive=True,
        damping_up=4.0,
        damping_down=0.3,
        min_damping=1.0e-10,
        use_diagonal_damping=False,
        tolerance=min(float(step_tol), float(cost_tol)),
        max_steps=int(max_iters),
        max_iter=int(max_iters),
        accept_rho_min=1.0e-12,
        good_rho_min=0.75,
        bad_rho_max=0.25,
        numerical_check=False,
        store_y_mags=True,
        store_states=False,
    )

    final_cost = float(solver.final_cost)

    info = {
        "iterations": int(solver.idx),
        "converged": bool(solver.converged),
        "cost": float(final_cost),
        "step_norm": float(solver.last_step_norm),
        "lambda": float(solver.damping),
        "num_points": int(len(observed_pixels_cam1)),
    }
    return solver.state, info


def create_solution_image(
        cam2: Sensor,
        observed_pixels_cam1: np.ndarray,
        observed_pixels_cam2: np.ndarray,
        est_cam2_from_cam1: SE3_q,
        cam1_cal: Calibration,
) -> np.ndarray:
    """Visualize observed cam2 points and estimated epipolar lines from cam1 observations."""
    image = np.zeros((cam2.camCal.height, cam2.camCal.width, 3), dtype=np.uint8)
    essential = q.skew(est_cam2_from_cam1.tvec / np.linalg.norm(est_cam2_from_cam1.tvec)) @ est_cam2_from_cam1.quat.to_dcm()

    for observed_px_cam1, observed_px in zip(observed_pixels_cam1, observed_pixels_cam2):
        line_cam2 = essential @ pixel_to_bearing(cam1_cal, observed_px_cam1)
        endpoints = line_norm_to_pixel_endpoints(cam2.camCal, line_cam2)
        ox, oy = int(round(observed_px[0])), int(round(observed_px[1]))

        if 0 <= ox < cam2.camCal.width and 0 <= oy < cam2.camCal.height:
            draw_square(image, ox, oy, (0, 255, 0))
        if endpoints is not None:
            cv2.line(image, endpoints[0], endpoints[1], (0, 0, 255), 1)

    return image


def main():
    """Run the synthetic two-view example, optimize the relative pose, and display results."""
    cam1, cam2 = generate_sensors()
    features = generate_features()

    w, h = cam1.camCal.width, cam1.camCal.height
    composite = np.zeros((h, 2 * w + 1, 3), dtype=np.uint8)
    composite[:, :w] = create_image(cam1, features)
    composite[:, w, :] = 128
    composite[:, w + 1:] = create_image(cam2, features)
    cv2.imshow('img', composite)
    cv2.waitKey(0)

    observed_pixels_cam1, observed_pixels_cam2 = build_measurements(cam1,
                                                                    cam2,
                                                                    features,
                                                                    pixel_noise_std=1.0)
    true_SE3 = cam2.SE3.inv * cam1.SE3
    true_SE3.tvec = true_SE3.tvec / np.linalg.norm(true_SE3.tvec)
    seed_SE3 = SE3_q(
        quat=quat.random_quat_within_deg(25.0) * true_SE3.quat,
        tvec=true_SE3.tvec + np.array([0.1, -0.05, 0.15]),
    )
    est_SE3, info = optimize(observed_pixels_cam1, observed_pixels_cam2, cam1.camCal, cam2.camCal, seed_SE3)
    solution_image = create_solution_image(cam2, observed_pixels_cam1, observed_pixels_cam2, est_SE3, cam1.camCal)
    matching_stats = summarize_matching(observed_pixels_cam1, observed_pixels_cam2, cam1.camCal, cam2.camCal, est_SE3)
    triangulated_points_cam1 = triangulate_midpoint_cam1(observed_pixels_cam1,
                                                         observed_pixels_cam2,
                                                         cam1.camCal,
                                                         cam2.camCal,
                                                         est_SE3)
    triangulation_stats = summarize_triangulation(triangulated_points_cam1,
                                                  observed_pixels_cam1,
                                                  observed_pixels_cam2,
                                                  cam1.camCal,
                                                  cam2.camCal,
                                                  est_SE3)

    err_SE3 = est_SE3.inv * true_SE3
    trans_dir_err_deg = np.degrees(np.arccos(np.clip(np.dot(est_SE3.tvec, true_SE3.tvec), -1.0, 1.0)))

    from support.io.my_logging import LOG
    LOG.info(f"LM info: {info}")
    LOG.info(f"True cam2_from_cam1:\n{true_SE3}")
    LOG.info(f"Estimated cam2_from_cam1:\n{est_SE3}")
    LOG.info(f"Rotation error deg: {err_SE3.quat.angle_betweenD(quat.identity()):.6f}")
    LOG.info(f"Translation direction error deg: {trans_dir_err_deg:.6f}")
    LOG.info(
        "Match stats | "
        f"mean point->epiline px: {matching_stats['mean_point_to_line_px']:.6f}, "
        f"median: {matching_stats['median_point_to_line_px']:.6f}, "
        f"rms: {matching_stats['rms_point_to_line_px']:.6f}, "
        f"max: {matching_stats['max_point_to_line_px']:.6f}"
    )
    LOG.info(
        "Match totals | "
        f"sum point->epiline px: {matching_stats['total_point_to_line_px']:.6f}, "
        f"sum squared point->epiline px: {matching_stats['total_squared_point_to_line_px']:.6f}, "
        f"sum abs Sampson: {matching_stats['total_abs_sampson']:.6f}"
    )
    LOG.info(
        "Sampson stats | "
        f"mean abs: {matching_stats['mean_abs_sampson']:.6f}, "
        f"rms: {matching_stats['rms_sampson']:.6f}, "
        f"max abs: {matching_stats['max_abs_sampson']:.6f}"
    )
    LOG.info(
        "Triangulation reprojection | "
        f"valid points: {int(triangulation_stats['num_valid_points'])}, "
        f"cam1 mean/rms/max px: "
        f"{triangulation_stats['mean_reproj_cam1_px']:.6f}/"
        f"{triangulation_stats['rms_reproj_cam1_px']:.6f}/"
        f"{triangulation_stats['max_reproj_cam1_px']:.6f}, "
        f"cam2 mean/rms/max px: "
        f"{triangulation_stats['mean_reproj_cam2_px']:.6f}/"
        f"{triangulation_stats['rms_reproj_cam2_px']:.6f}/"
        f"{triangulation_stats['max_reproj_cam2_px']:.6f}"
    )
    LOG.info(
        "Triangulation totals | "
        f"cam1 total L2 px: {triangulation_stats['total_reproj_cam1_px']:.6f}, "
        f"cam2 total L2 px: {triangulation_stats['total_reproj_cam2_px']:.6f}"
    )
    LOG.info(
        "Triangulated depth scale | "
        f"cam1 mean/min/max: "
        f"{triangulation_stats['mean_depth_cam1']:.6f}/"
        f"{triangulation_stats['min_depth_cam1']:.6f}/"
        f"{triangulation_stats['max_depth_cam1']:.6f}, "
        f"cam2 mean/min/max: "
        f"{triangulation_stats['mean_depth_cam2']:.6f}/"
        f"{triangulation_stats['min_depth_cam2']:.6f}/"
        f"{triangulation_stats['max_depth_cam2']:.6f}"
    )
    valid_mask = np.asarray(triangulation_stats["valid_mask"], dtype=bool)
    predicted_pixels_cam1 = np.asarray(triangulation_stats["predicted_pixels_cam1"], dtype=float)
    predicted_pixels_cam2 = np.asarray(triangulation_stats["predicted_pixels_cam2"], dtype=float)
    points_cam2 = np.asarray(triangulation_stats["points_cam2"], dtype=float)
    line_distances_px = np.asarray(matching_stats["line_distances_px"], dtype=float)
    sampson_residuals = np.asarray(matching_stats["sampson_residuals"], dtype=float)

    LOG.info("Per-feature diagnostics follow.")
    for idx, (obs1, obs2, pt1, pt2, valid) in enumerate(
            zip(observed_pixels_cam1,
                observed_pixels_cam2,
                triangulated_points_cam1,
                points_cam2,
                valid_mask)
    ):
        if valid:
            pred1 = predicted_pixels_cam1[idx]
            pred2 = predicted_pixels_cam2[idx]
            reproj1 = pred1 - obs1
            reproj2 = pred2 - obs2
            reproj1_l2 = float(np.linalg.norm(reproj1))
            reproj2_l2 = float(np.linalg.norm(reproj2))
            cam1_depth = float(pt1[2])
            cam2_depth = float(pt2[2])
            valid_str = "valid"
        else:
            pred1 = np.array([np.nan, np.nan], dtype=float)
            pred2 = np.array([np.nan, np.nan], dtype=float)
            reproj1 = np.array([np.nan, np.nan], dtype=float)
            reproj2 = np.array([np.nan, np.nan], dtype=float)
            reproj1_l2 = float("nan")
            reproj2_l2 = float("nan")
            cam1_depth = float("nan")
            cam2_depth = float("nan")
            valid_str = "invalid"

        LOG.info(
            f"Feature {idx:03d} | {valid_str} | "
            f"obs1=({obs1[0]:.3f},{obs1[1]:.3f}) "
            f"obs2=({obs2[0]:.3f},{obs2[1]:.3f}) | "
            f"tri_cam1=({pt1[0]:.6f},{pt1[1]:.6f},{pt1[2]:.6f}) "
            f"tri_cam2=({pt2[0]:.6f},{pt2[1]:.6f},{pt2[2]:.6f}) | "
            f"depths=({cam1_depth:.6f},{cam2_depth:.6f}) | "
            f"pred1=({pred1[0]:.3f},{pred1[1]:.3f}) "
            f"pred2=({pred2[0]:.3f},{pred2[1]:.3f}) | "
            f"reproj1=({reproj1[0]:.3f},{reproj1[1]:.3f}) l2={reproj1_l2:.6f} | "
            f"reproj2=({reproj2[0]:.3f},{reproj2[1]:.3f}) l2={reproj2_l2:.6f} | "
            f"epiline_px={line_distances_px[idx]:.6f} | "
            f"sampson={sampson_residuals[idx]:.9f}"
        )
    LOG.info("solution image: green=observed cam2 pixels, red=estimated epipolar lines from cam1 pixels")
    cv2.imshow('solution', solution_image)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
