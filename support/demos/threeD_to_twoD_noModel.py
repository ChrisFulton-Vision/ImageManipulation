import cv2
import numpy as np
from dataclasses import dataclass, field

import support.mathHelpers.quaternions as quat
from support.mathHelpers.SE3 import SE3_q
from support.mathHelpers.LevMarq import LevenbergMarquardt
from support.mathHelpers.SE3ThreeDtoTwoD_noModel import SE3ThreeDtoTwoD_noModel
from support.runtime.pixel_handler import Pixel as pxl
from support.vision.calibration import Calibration, default_864_cam

np.random.seed(42)


@dataclass
class Sensor:
    camCal: Calibration = field(default_factory=lambda: default_864_cam().randomize())
    SE3: SE3_q = field(default_factory=SE3_q)

    def __post_init__(self):
        self.SE3.quat = quat.random_quat_within_deg(20.0)
        self.SE3.tvec = np.random.normal(loc=0, scale=1, size=3) + np.array([0, 0, -5.0])


@dataclass
class Feature:
    tvec: np.ndarray = field(default_factory=lambda: np.random.rand(3,))

    def __repr__(self):
        return f"x:{self.tvec[0]:.3f}, y:{self.tvec[1]:.3f}, z:{self.tvec[2]:.3f}\n"


def generate_sensors() -> tuple[Sensor, Sensor]:
    return Sensor(), Sensor()


def generate_features(num_features: int = 50) -> tuple[Feature, ...]:
    return tuple(Feature() for _ in range(num_features))


def draw_square(image: np.ndarray, x: int, y: int, color: tuple[int, int, int], half_size: int = 2) -> None:
    h, w = image.shape[:2]
    x0 = max(0, x - half_size)
    x1 = min(w, x + half_size + 1)
    y0 = max(0, y - half_size)
    y1 = min(h, y + half_size + 1)
    image[y0:y1, x0:x1, :] = color


def project_point(cam: Sensor, point_world: np.ndarray) -> np.ndarray | None:
    point_cam = cam.SE3.inv * point_world
    if point_cam[2] <= 0:
        return None

    norm_xy = point_cam[:2] / point_cam[2]
    px_point = pxl(norm_coords=norm_xy, already_undistorted=True)
    cam.camCal.haveNorm_needPix(px_point)
    return np.asarray(px_point.pix_coords, dtype=float)


def create_image(cam: Sensor, features: tuple[Feature, ...]) -> np.ndarray:
    image = np.zeros((cam.camCal.height, cam.camCal.width, 3), dtype=np.uint8)

    for feat in features:
        pixel_xy = project_point(cam, feat.tvec)
        if pixel_xy is None:
            continue

        xi = int(round(pixel_xy[0]))
        yi = int(round(pixel_xy[1]))
        if 0 <= xi < cam.camCal.width and 0 <= yi < cam.camCal.height:
            image[yi - 2:yi + 2, xi - 2:xi + 2, 1:] = 255

    return image


def build_measurements(
        cam1: Sensor,
        cam2: Sensor,
        features: tuple[Feature, ...],
        *,
        pixel_noise_std: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    pixel_noise_std = float(pixel_noise_std)
    if pixel_noise_std < 0.0:
        raise ValueError("pixel_noise_std must be non-negative")

    points_cam1: list[np.ndarray] = []
    pixels_cam2: list[np.ndarray] = []

    for feat in features:
        point_cam1 = cam1.SE3.inv * feat.tvec
        pixel_cam2 = project_point(cam2, feat.tvec)

        if point_cam1[2] <= 0 or pixel_cam2 is None:
            continue

        if pixel_noise_std > 0.0:
            pixel_cam2 = pixel_cam2 + np.random.normal(loc=0.0, scale=pixel_noise_std, size=2)

        x2, y2 = pixel_cam2
        if 0 <= x2 < cam2.camCal.width and 0 <= y2 < cam2.camCal.height:
            points_cam1.append(np.asarray(point_cam1, dtype=float))
            pixels_cam2.append(pixel_cam2)

    if not points_cam1:
        raise ValueError("No shared visible features between cam1 and cam2.")

    return np.asarray(points_cam1, dtype=float), np.asarray(pixels_cam2, dtype=float)


def predict_pixels(
        est_cam2_from_cam1: SE3_q,
        points_cam1: np.ndarray,
        cam2_cal: Calibration,
) -> np.ndarray:
    pred_points_cam2 = est_cam2_from_cam1 * points_cam1
    z = pred_points_cam2[:, 2]
    pred_pixels = np.empty((len(points_cam1), 2), dtype=float)
    pred_pixels[:, 0] = cam2_cal.fx * pred_points_cam2[:, 0] / z + cam2_cal.cx
    pred_pixels[:, 1] = cam2_cal.fy * pred_points_cam2[:, 1] / z + cam2_cal.cy
    return pred_pixels


def summarize_reprojection(
        points_cam1: np.ndarray,
        observed_pixels_cam2: np.ndarray,
        predicted_pixels_cam2: np.ndarray,
        est_cam2_from_cam1: SE3_q,
) -> dict[str, np.ndarray | float]:
    points_cam1 = np.asarray(points_cam1, dtype=float)
    observed_pixels_cam2 = np.asarray(observed_pixels_cam2, dtype=float)
    predicted_pixels_cam2 = np.asarray(predicted_pixels_cam2, dtype=float)
    predicted_points_cam2 = est_cam2_from_cam1 * points_cam1
    residuals = predicted_pixels_cam2 - observed_pixels_cam2
    l2_errors = np.linalg.norm(residuals, axis=1)

    return {
        "predicted_points_cam2": predicted_points_cam2,
        "residuals": residuals,
        "l2_errors": l2_errors,
        "mean_l2_px": float(np.mean(l2_errors)),
        "median_l2_px": float(np.median(l2_errors)),
        "rms_l2_px": float(np.sqrt(np.mean(l2_errors * l2_errors))),
        "max_l2_px": float(np.max(l2_errors)),
        "total_l2_px": float(np.sum(l2_errors)),
        "total_squared_l2_px": float(np.sum(l2_errors * l2_errors)),
        "mean_depth_cam1": float(np.mean(points_cam1[:, 2])),
        "min_depth_cam1": float(np.min(points_cam1[:, 2])),
        "max_depth_cam1": float(np.max(points_cam1[:, 2])),
        "mean_depth_cam2": float(np.mean(predicted_points_cam2[:, 2])),
        "min_depth_cam2": float(np.min(predicted_points_cam2[:, 2])),
        "max_depth_cam2": float(np.max(predicted_points_cam2[:, 2])),
    }


def optimize(
        points_cam1: np.ndarray,
        observed_pixels_cam2: np.ndarray,
        cam2_cal: Calibration,
        seed_cam2_from_cam1: SE3_q | None = None,
        *,
        max_iters: int = 60,
        damping: float = 1.0e-2,
        step_tol: float = 1.0e-10,
        cost_tol: float = 1.0e-12,
) -> tuple[SE3_q, dict[str, float | int | bool]]:
    est = SE3_q() if seed_cam2_from_cam1 is None else seed_cam2_from_cam1.copy()
    problem = SE3ThreeDtoTwoD_noModel(
        points_cam1=np.asarray(points_cam1, dtype=float),
        observed_pixels_cam2=np.asarray(observed_pixels_cam2, dtype=float),
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
        robust_loss='huber'
    )

    info = {
        "iterations": int(solver.idx),
        "converged": bool(solver.converged),
        "cost": float(solver.final_cost),
        "step_norm": float(solver.last_step_norm),
        "lambda": float(solver.damping),
        "num_points": int(len(points_cam1)),
    }
    return solver.state, info


def create_solution_image(
        cam2: Sensor,
        observed_pixels_cam2: np.ndarray,
        predicted_pixels_cam2: np.ndarray,
) -> np.ndarray:
    image = np.zeros((cam2.camCal.height, cam2.camCal.width, 3), dtype=np.uint8)

    for observed_px, predicted_px in zip(observed_pixels_cam2, predicted_pixels_cam2):
        ox, oy = int(round(observed_px[0])), int(round(observed_px[1]))
        px, py = int(round(predicted_px[0])), int(round(predicted_px[1]))

        if 0 <= ox < cam2.camCal.width and 0 <= oy < cam2.camCal.height:
            draw_square(image, ox, oy, (0, 255, 0))
        if 0 <= px < cam2.camCal.width and 0 <= py < cam2.camCal.height:
            draw_square(image, px, py, (0, 0, 255))
        if (
                0 <= ox < cam2.camCal.width and 0 <= oy < cam2.camCal.height
                and 0 <= px < cam2.camCal.width and 0 <= py < cam2.camCal.height
        ):
            cv2.line(image, (ox, oy), (px, py), (255, 255, 255), 1)

    return image


def main():
    cam1, cam2 = generate_sensors()
    features = generate_features()

    w, h = cam1.camCal.width, cam1.camCal.height
    composite = np.zeros((h, 2 * w + 1, 3), dtype=np.uint8)
    composite[:, :w] = create_image(cam1, features)
    composite[:, w, :] = 128
    composite[:, w + 1:] = create_image(cam2, features)
    cv2.imshow("img", composite)
    cv2.waitKey(0)

    points_cam1, observed_pixels_cam2 = build_measurements(cam1, cam2, features, pixel_noise_std=1.0)
    true_SE3 = cam2.SE3.inv * cam1.SE3
    seed_SE3 = SE3_q(
        quat=quat.random_quat_within_deg(55.0) * true_SE3.quat,
        tvec=true_SE3.tvec + np.array([0.1, -0.05, 0.15]),
    )
    est_SE3, info = optimize(points_cam1, observed_pixels_cam2, cam2.camCal, seed_SE3)
    predicted_pixels_cam2 = predict_pixels(est_SE3, points_cam1, cam2.camCal)
    reprojection_stats = summarize_reprojection(points_cam1, observed_pixels_cam2, predicted_pixels_cam2, est_SE3)
    solution_image = create_solution_image(cam2, observed_pixels_cam2, predicted_pixels_cam2)

    err_SE3 = est_SE3.inv * true_SE3

    from support.io.my_logging import LOG
    LOG.info(f"LM info: {info}")
    LOG.info(f"True cam2_from_cam1:\n{true_SE3}")
    LOG.info(f"Estimated cam2_from_cam1:\n{est_SE3}")
    LOG.info(f"Rotation error deg: {err_SE3.quat.angle_betweenD(quat.identity()):.6f}")
    LOG.info(f"Translation error: {np.linalg.norm(err_SE3.tvec):.6e}")
    LOG.info(
        "Reprojection stats | "
        f"mean/median/rms/max px: "
        f"{reprojection_stats['mean_l2_px']:.6f}/"
        f"{reprojection_stats['median_l2_px']:.6f}/"
        f"{reprojection_stats['rms_l2_px']:.6f}/"
        f"{reprojection_stats['max_l2_px']:.6f}"
    )
    LOG.info(
        "Reprojection totals | "
        f"sum L2 px: {reprojection_stats['total_l2_px']:.6f}, "
        f"sum squared L2 px: {reprojection_stats['total_squared_l2_px']:.6f}"
    )
    LOG.info(
        "Depth stats | "
        f"cam1 mean/min/max: "
        f"{reprojection_stats['mean_depth_cam1']:.6f}/"
        f"{reprojection_stats['min_depth_cam1']:.6f}/"
        f"{reprojection_stats['max_depth_cam1']:.6f}, "
        f"cam2 mean/min/max: "
        f"{reprojection_stats['mean_depth_cam2']:.6f}/"
        f"{reprojection_stats['min_depth_cam2']:.6f}/"
        f"{reprojection_stats['max_depth_cam2']:.6f}"
    )
    predicted_points_cam2 = np.asarray(reprojection_stats["predicted_points_cam2"], dtype=float)
    residuals = np.asarray(reprojection_stats["residuals"], dtype=float)
    l2_errors = np.asarray(reprojection_stats["l2_errors"], dtype=float)
    LOG.info("Per-feature diagnostics follow.")
    for idx, (point_cam1, point_cam2, observed_px, predicted_px, residual, l2_error) in enumerate(
            zip(points_cam1,
                predicted_points_cam2,
                observed_pixels_cam2,
                predicted_pixels_cam2,
                residuals,
                l2_errors)
    ):
        LOG.info(
            f"Feature {idx:03d} | "
            f"cam1_pt=({point_cam1[0]:.6f},{point_cam1[1]:.6f},{point_cam1[2]:.6f}) | "
            f"cam2_pt=({point_cam2[0]:.6f},{point_cam2[1]:.6f},{point_cam2[2]:.6f}) | "
            f"obs2=({observed_px[0]:.3f},{observed_px[1]:.3f}) "
            f"pred2=({predicted_px[0]:.3f},{predicted_px[1]:.3f}) | "
            f"residual=({residual[0]:.3f},{residual[1]:.3f}) | "
            f"l2_px={l2_error:.6f}"
        )
    LOG.info("solution image: green=observed, red=predicted, white=residual")
    cv2.imshow("solution", solution_image)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
