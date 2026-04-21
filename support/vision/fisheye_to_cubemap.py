import cv2
import numpy as np


DEFAULT_CUBEMAP_FACES = ("right", "left", "top", "bottom", "front")
DEFAULT_FRONT_FACE = ("front",)
DEFAULT_CUBEMAP_FACESIZE = 600

DEFAULT_CUBEMAP_LAYOUT = {
    "bottom": (2, 1),
    "left": (1, 0),
    "front": (1, 1),
    "right": (1, 2),
    "top": (0, 1),
}

_FACE_AXES = {
    "right": ([1, 0, 0], [0, -1, 0]),
    "left": ([-1, 0, 0], [0, -1, 0]),
    "top": ([0, -1, 0], [0, 0, -1]),
    "bottom": ([0, 1, 0], [0, 0, 1]),
    "front": ([0, 0, 1], [0, -1, 0]),
    # "back": ([0, 0, -1], [0, -1, 0]),
}


def _normalize_faces(faces) -> tuple[str, ...]:
    faces = tuple(faces)
    for face in faces:
        if face not in _FACE_AXES:
            raise KeyError(f"Unknown cubemap face '{face}'")
    return faces


def _build_face_dirs(face_size: int, faces: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Compute per-pixel unit direction vectors for the requested cube faces."""
    if face_size <= 0:
        raise ValueError(f"face_size must be > 0, got {face_size}")

    faces = _normalize_faces(faces)

    rng = np.linspace(-1.0, 1.0, face_size, dtype=np.float32)
    xx, yy = np.meshgrid(rng, -rng)  # Flip Y for image coordinates

    faces_dirs: dict[str, np.ndarray] = {}

    for name in faces:
        center, up = _FACE_AXES[name]
        center = np.asarray(center, dtype=np.float32)
        up = np.asarray(up, dtype=np.float32)
        right = np.cross(center, up).astype(np.float32)

        dirs = (
            center[None, None, :]
            + xx[..., None] * right[None, None, :]
            + yy[..., None] * up[None, None, :]
        )
        dirs /= np.linalg.norm(dirs, axis=2, keepdims=True)
        faces_dirs[name] = dirs.astype(np.float32)

    return faces_dirs


class FisheyeCubemapManager:
    """
    Owns reusable caches for:
      - cube-face direction vectors
      - cube-face remap grids
      - standard undistortion maps

    This lets the GUI/controller stop carrying around cubemap_x/y/faces_dirs state.
    """

    def __init__(self):
        self._faces_dirs_cache: dict[tuple[int, tuple[str, ...]], dict[str, np.ndarray]] = {}
        self._cubemap_map_cache: dict[
            tuple[tuple, int, tuple[str, ...]],
            tuple[dict[str, np.ndarray], dict[str, np.ndarray]],
        ] = {}
        self._standard_map_cache: dict[tuple[tuple, float], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def clear(self) -> None:
        self._faces_dirs_cache.clear()
        self._cubemap_map_cache.clear()
        self._standard_map_cache.clear()

    @staticmethod
    def _calibration_signature(calibration) -> tuple:
        if calibration is None or not getattr(calibration, "validCal", False):
            raise ValueError("A valid calibration is required.")

        K = np.asarray(calibration.getCameraMatrix(), dtype=np.float64)
        D = np.asarray(calibration.getDistortion(), dtype=np.float64).reshape(-1)

        return (
            bool(getattr(calibration, "fisheye", False)),
            int(getattr(calibration, "width", 0) or 0),
            int(getattr(calibration, "height", 0) or 0),
            K.shape,
            K.tobytes(),
            D.shape,
            D.tobytes(),
        )

    def ensure_standard_undistort_maps(self, calibration, alpha: float = 0.0):
        """
        Returns (newK, map1, map2), cached by calibration signature.

        For fisheye calibrations, uses cv2.fisheye undistort-map generation.
        For non-fisheye calibrations, uses the regular OpenCV path.
        """
        sig = (self._calibration_signature(calibration), float(alpha))

        if sig in self._standard_map_cache:
            return self._standard_map_cache[sig]

        w = int(calibration.width)
        h = int(calibration.height)
        K = np.asarray(calibration.getCameraMatrix(), dtype=np.float64)
        D = np.asarray(calibration.getDistortion(), dtype=np.float64)

        if bool(getattr(calibration, "fisheye", False)):
            R = np.eye(3, dtype=np.float64)
            newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D, (w, h), R, balance=float(alpha)
            )
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K, D, R, newK, (w, h), cv2.CV_32FC1
            )
        else:
            newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=float(alpha))
            map1, map2 = cv2.initUndistortRectifyMap(
                K, D, None, newK, (w, h), cv2.CV_32FC1
            )

        self._standard_map_cache[sig] = (newK, map1, map2)
        return self._standard_map_cache[sig]

    def get_face_dirs(self, face_size: int, faces=DEFAULT_CUBEMAP_FACES) -> dict[str, np.ndarray]:
        faces = _normalize_faces(faces)
        key = (int(face_size), faces)

        if key not in self._faces_dirs_cache:
            self._faces_dirs_cache[key] = _build_face_dirs(int(face_size), faces)

        return self._faces_dirs_cache[key]

    @staticmethod
    def _build_cubemap_maps_from_dirs(
        calibration,
        face_size: int,
        faces_dirs: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Build remap grids (cubemap_x, cubemap_y) for the requested faces.

        Unlike the old implementation, this builds *all* faces and never returns
        early from inside the loop.
        """
        if calibration is None or not getattr(calibration, "validCal", False):
            raise ValueError("A valid calibration is required.")

        K = np.asarray(calibration.getCameraMatrix(), dtype=np.float64)
        D = np.asarray(calibration.getDistortion(), dtype=np.float64)

        cubemap_x: dict[str, np.ndarray] = {}
        cubemap_y: dict[str, np.ndarray] = {}

        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)

        for face, dirs in faces_dirs.items():
            dirs_reshaped = np.asarray(dirs, dtype=np.float64).reshape(-1, 1, 3)

            # Only keep directions roughly facing the front hemisphere.
            forward_mask = dirs_reshaped[:, 0, 2] > 0.0

            full_img_points = np.full((face_size * face_size, 2), -1.0, dtype=np.float32)

            if np.any(forward_mask):
                valid_dirs = dirs_reshaped[forward_mask]
                img_points, _ = cv2.fisheye.projectPoints(
                    valid_dirs,
                    rvec,
                    tvec,
                    K,
                    D,
                )
                full_img_points[forward_mask] = img_points.reshape(-1, 2).astype(np.float32)

            cubemap_x[face] = full_img_points[:, 0].reshape(face_size, face_size)
            cubemap_y[face] = full_img_points[:, 1].reshape(face_size, face_size)

        return cubemap_x, cubemap_y

    def ensure_cubemap_maps(self, calibration, face_size: int, faces=DEFAULT_CUBEMAP_FACES):
        faces = _normalize_faces(faces)
        key = (self._calibration_signature(calibration), int(face_size), faces)

        if key not in self._cubemap_map_cache:
            faces_dirs = self.get_face_dirs(int(face_size), faces)
            self._cubemap_map_cache[key] = self._build_cubemap_maps_from_dirs(
                calibration=calibration,
                face_size=int(face_size),
                faces_dirs=faces_dirs,
            )

        return self._cubemap_map_cache[key]

    @staticmethod
    def remap_face(frame: np.ndarray, cubemap_x: dict[str, np.ndarray], cubemap_y: dict[str, np.ndarray], face: str):
        return cv2.remap(
            frame,
            cubemap_x[face],
            cubemap_y[face],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    def render_faces(self, frame: np.ndarray, calibration, face_size: int, faces=DEFAULT_CUBEMAP_FACES):
        cubemap_x, cubemap_y = self.ensure_cubemap_maps(calibration, int(face_size), faces)
        return {
            face: self.remap_face(frame, cubemap_x, cubemap_y, face)
            for face in _normalize_faces(faces)
        }

    @staticmethod
    def stitch_faces(
        cubemap_faces: dict[str, np.ndarray],
        face_size: int,
        layout: dict[str, tuple[int, int]],
        cells: int = 3,
    ) -> np.ndarray:
        """
        Arrange cubemap faces into a stitched layout.

        Example layout:
            +--------+--------+--------+
            |        |   top  |        |
            +--------+--------+--------+
            |  left  | front  | right  |
            +--------+--------+--------+
            |        | bottom |        |
            +--------+--------+--------+
        """
        if not cubemap_faces:
            raise ValueError("cubemap_faces is empty")

        sample = next(iter(cubemap_faces.values()))
        channels = 1 if sample.ndim == 2 else sample.shape[2]
        out_shape = (
            cells * face_size,
            cells * face_size,
        ) if channels == 1 else (
            cells * face_size,
            cells * face_size,
            channels,
        )

        stitched = np.zeros(out_shape, dtype=sample.dtype)

        for face, (row, col) in layout.items():
            if face not in cubemap_faces:
                continue
            y0 = row * face_size
            x0 = col * face_size
            stitched[y0:y0 + face_size, x0:x0 + face_size] = cubemap_faces[face]

        return stitched

    def render_cubemap(
        self,
        frame: np.ndarray,
        calibration,
        face_size: int,
        layout: dict[str, tuple[int, int]] = DEFAULT_CUBEMAP_LAYOUT,
        faces=DEFAULT_CUBEMAP_FACES,
        cells: int = 3,
    ) -> np.ndarray:
        cubemap_faces = self.render_faces(
            frame=frame,
            calibration=calibration,
            face_size=int(face_size),
            faces=faces,
        )
        return self.stitch_faces(
            cubemap_faces=cubemap_faces,
            face_size=int(face_size),
            layout=layout,
            cells=cells,
        )

    def render_front_face(self, frame: np.ndarray, calibration, face_size: int) -> np.ndarray:
        faces = self.render_faces(
            frame=frame,
            calibration=calibration,
            face_size=int(face_size),
            faces=DEFAULT_FRONT_FACE,
        )
        return faces["front"]
