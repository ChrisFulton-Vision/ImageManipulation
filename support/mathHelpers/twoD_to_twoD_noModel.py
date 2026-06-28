import cv2
import numpy as np
from support.vision.calibration import default_864_cam, Calibration
import support.mathHelpers.quaternions as quat
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
    """Build cam1-frame 3D points and noisy cam2 pixel measurements for shared visible features."""
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
    """Project cam1-frame 3D points into cam2 pixel coordinates under the current estimate."""
    pred_points_cam2 = est_cam2_from_cam1 * points_cam1
    z = pred_points_cam2[:, 2]
    pred_pixels = np.empty((len(points_cam1), 2), dtype=float)
    pred_pixels[:, 0] = cam2_cal.fx * pred_points_cam2[:, 0] / z + cam2_cal.cx
    pred_pixels[:, 1] = cam2_cal.fy * pred_points_cam2[:, 1] / z + cam2_cal.cy
    return pred_pixels


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
    """Solve for cam2-from-cam1 with a Levenberg-Marquardt reprojection fit."""
    est = SE3_q() if seed_cam2_from_cam1 is None else seed_cam2_from_cam1.copy()
    problem = SE3TwoDtoTwoD_noModel(
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
    )

    final_cost = float(solver.final_cost)

    info = {
        "iterations": int(solver.idx),
        "converged": bool(solver.converged),
        "cost": float(final_cost),
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
    """Visualize measured versus predicted cam2 pixels and their residual segments."""
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

    points_cam1, observed_pixels_cam2 = build_measurements(cam1, cam2, features, pixel_noise_std=1.0)
    true_SE3 = cam2.SE3.inv * cam1.SE3
    seed_SE3 = SE3_q(
        quat=quat.random_quat_within_deg(5.0) * true_SE3.quat,
        tvec=true_SE3.tvec + np.array([0.1, -0.05, 0.15]),
    )
    est_SE3, info = optimize(points_cam1, observed_pixels_cam2, cam2.camCal, seed_SE3)
    predicted_pixels_cam2 = predict_pixels(est_SE3, points_cam1, cam2.camCal)
    solution_image = create_solution_image(cam2, observed_pixels_cam2, predicted_pixels_cam2)

    err_SE3 = est_SE3.inv * true_SE3

    from support.io.my_logging import LOG
    LOG.info(f"LM info: {info}")
    LOG.info(f"True cam2_from_cam1:\n{true_SE3}")
    LOG.info(f"Estimated cam2_from_cam1:\n{est_SE3}")
    LOG.info(f"Rotation error deg: {err_SE3.quat.angle_betweenD(quat.identity()):.6f}")
    LOG.info(f"Translation error: {np.linalg.norm(err_SE3.tvec):.6e}")
    LOG.info("solution image: green=observed, red=predicted, white=residual")
    cv2.imshow('solution', solution_image)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
