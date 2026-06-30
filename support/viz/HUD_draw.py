import numpy as np
from numpy import sin, cos, deg2rad
from support.io.attitude_interpreter import (
    AttitudeReader as AttRdr,
    ControlMode,
    CAMERA_LEVER_ARM_BODY_M,
    CAMERA_RPY_OFFSET_DEG,
)
from support.viz.CVFontScaling import med_text, small_thick, med_thick, lrg_thick
from numpy.typing import NDArray
import cv2
from math import sin, cos, radians
import support.viz.colors as clr
from support.core.enums import PlaybackSpeed
from support.gui.UserSelectQueue import HudOpts

MINIMAP_TRI = np.array([[0, 5, -5, 0], [0, 12, 12, 0]])


class HUD_Marker:
    def __init__(self, filepath=None, img_folder_path=None):
        self.cam_bank_offset = 0.0  # deg
        self.attRdr = AttRdr()
        self.bank_indicator_points = self.create_bank_indicator()
        self.last_xy = (864, 864)
        self.bank_indicator_lines = None
        self.controlMode_text_loc = None
        self.throttle_loc = None
        self.throttle_circle_points = None

        # --- minimap state ---
        self.minimap_rect = None           # (x0, y0, x1, y1)
        self.minimap_inner_rect = None     # (x0, y0, x1, y1)
        self.minimap_start_px = None       # (x, y) once GPS is available
        self.minimap_trail_px = []         # recent [(x, y), ...]
        self.minimap_trail_maxlen = 40
        self.minimap_last_time_s = None

        self.update_storage(864, 864)

        if filepath is not None:
            self.read_attitude_files(filepath, img_folder_path)

    def update_storage(self, x, y):
        self.last_xy = (x, y)

        self.bank_indicator_lines = [
            (np.array([self.last_xy[0], self.last_xy[1]]) * np.array(self.bank_indicator_points)).astype(int)]

        self.throttle_loc = (int(x * 0.8), int(y * 0.50))
        num = 20

        thetas = np.linspace(0.0, 245.0, num)
        points = np.zeros((num, 2), int)
        r = 0.06
        points[:, 0] = self.throttle_loc[0] + (np.sin(np.deg2rad(thetas)) * x * r).astype(int)
        points[:, 1] = self.throttle_loc[1] - (np.cos(np.deg2rad(thetas)) * x * r).astype(int)
        self.throttle_circle_points = [points]

        self.controlMode_text_loc = np.array([.65 * x, .90 * y]).astype(int)

        # --- minimap geometry ---
        map_w = int(0.18 * x)
        map_h = int(0.18 * y)
        margin = int(0.025 * x)
        pad = max(4, int(0.008 * x))

        x1 = x - margin
        y0 = margin
        x0 = x1 - map_w
        y1 = y0 + map_h

        self.minimap_rect = (x0, y0, x1, y1)
        self.minimap_inner_rect = (x0 + pad, y0 + pad, x1 - pad, y1 - pad)

        # image resized -> force recompute next GPS frame
        self.minimap_start_px = None
        self.minimap_trail_px.clear()

    @staticmethod
    def create_bank_indicator():
        bank_indicator_points = []
        last_angle = -60
        for new_angle in np.linspace(-50, 60, 12):
            max_rad = 0.10
            if last_angle % 30.0 == 0.0:
                max_rad = 0.11
            normal_ang = 0.08
            bank_indicator_points.append((0.5 + max_rad * sin(deg2rad(last_angle)),
                                          0.8 + max_rad * cos(deg2rad(last_angle))))
            bank_indicator_points.append((0.5 + normal_ang * sin(deg2rad(last_angle)),
                                          0.8 + normal_ang * cos(deg2rad(last_angle))))
            bank_indicator_points.append((0.5 + normal_ang * sin(deg2rad(new_angle)),
                                          0.8 + normal_ang * cos(deg2rad(new_angle))))
            last_angle = new_angle

        bank_indicator_points.append((0.5 + 0.11 * sin(deg2rad(60)),
                                      0.8 + 0.11 * cos(deg2rad(60))))
        return bank_indicator_points

    def read_attitude_files(self,
                            filepath,
                            img_folder_path,
                            update_time_offset_func=None):

        self.attRdr = AttRdr()
        loaded = self.attRdr.read_files(filepath,
                                        img_folder_path,
                                        update_time_offset_func)

        # New data set -> reset minimap state
        self.minimap_start_px = None
        self.minimap_trail_px.clear()
        self.minimap_last_time_s = None
        return loaded

    def update_offset(self, delta_offset):
        self.attRdr.offset += delta_offset

    @property
    def offset(self):
        return self.attRdr.offset

    def _map_to_minimap_px(self, map_x: float, map_y: float) -> tuple[int, int] | None:
        if not getattr(self.attRdr, 'has_gps', False):
            return None

        x_min = float(self.attRdr.map_x_min)
        x_max = float(self.attRdr.map_x_max)
        y_min = float(self.attRdr.map_y_min)
        y_max = float(self.attRdr.map_y_max)

        dx = x_max - x_min
        dy = y_max - y_min
        if dx <= 1e-12 or dy <= 1e-12:
            return None

        ix0, iy0, ix1, iy1 = self.minimap_inner_rect
        iw = max(1, ix1 - ix0)
        ih = max(1, iy1 - iy0)

        # preserve aspect ratio
        scale = min(iw / dx, ih / dy)
        draw_w = dx * scale
        draw_h = dy * scale

        off_x = 0.5 * (iw - draw_w)
        off_y = 0.5 * (ih - draw_h)

        px = ix0 + off_x + (map_x - x_min) * scale
        py = iy0 + off_y + (y_max - map_y) * scale  # invert y for screen coords

        return int(round(px)), int(round(py))

    def draw_minimap(self, image, att, map_transparency):
        if map_transparency < 0.01:
            return

        rect = self.minimap_rect
        inner = self.minimap_inner_rect
        x0, y0, x1, y1 = rect
        ix0, iy0, ix1, iy1 = inner
        h, w = image.shape[:2]

        # background box
        alpha = map_transparency
        roi = image[y0:y1, x0:x1]
        overlay = roi.copy()
        cv2.rectangle(overlay, (0, 0), (x1 - x0, y1 - y0), clr.BLACK, -1)
        cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, dst=roi)

        cv2.rectangle(image, (x0, y0), (x1, y1), clr.HUD_GREEN, med_thick(h))
        self.draw_minimap_runway(image, att)

        cur_px = self._map_to_minimap_px(att.map_x, att.map_y)
        if cur_px is None:
            return

        # establish start point lazily from first valid GPS draw
        if self.minimap_start_px is None:
            self.minimap_start_px = self._map_to_minimap_px(att.home_x, att.home_y)

        # reset trail if playback jumps around hard
        if self.minimap_last_time_s is not None:
            if abs(att.time_s - self.minimap_last_time_s) > 2.0:
                self.minimap_trail_px.clear()
        self.minimap_last_time_s = att.time_s

        # append only if moved enough to matter
        if len(self.minimap_trail_px) == 0:
            self.minimap_trail_px.append(cur_px)
        else:
            last_px = self.minimap_trail_px[-1]
            if (abs(cur_px[0] - last_px[0]) >= 1) or (abs(cur_px[1] - last_px[1]) >= 1):
                self.minimap_trail_px.append(cur_px)
                if len(self.minimap_trail_px) > self.minimap_trail_maxlen:
                    self.minimap_trail_px.pop(0)

        # trail
        if len(self.minimap_trail_px) >= 2:
            pts = np.array(self.minimap_trail_px, dtype=np.int32)
            cv2.polylines(image, [pts], False, clr.HUD_GREEN, small_thick(h))

        # start marker "X"
        if self.minimap_start_px is not None:
            sx, sy = self.minimap_start_px
            s = 4
            cv2.line(image, (sx - s, sy - s), (sx + s, sy + s), clr.DARKBLUE, 2)
            cv2.line(image, (sx - s, sy + s), (sx + s, sy - s), clr.DARKBLUE, 2)

        # current marker "O"
        hdg_rad = np.deg2rad(att.yaw_deg)
        rotmat_2d = np.array([[np.cos(hdg_rad), -np.sin(hdg_rad)],
                              [np.sin(hdg_rad), np.cos(hdg_rad)]])
        cx, cy = cur_px
        triangle = (rotmat_2d @ MINIMAP_TRI).astype(np.int32) + np.array([[cx], [cy]])
        cv2.polylines(image, [triangle.T], False, clr.HUD_GREEN, med_thick(h))

    def draw_minimap_runway(self, image, att):
        runway_rect_map_m = getattr(att, 'runway_rect_map_m', None)
        if not runway_rect_map_m:
            return

        x_min, y_min, x_max, y_max = runway_rect_map_m
        p0 = self._map_to_minimap_px(x_min, y_min)
        p1 = self._map_to_minimap_px(x_max, y_max)
        if p0 is None or p1 is None:
            return

        h, _w = image.shape[:2]
        x0, y0 = p0
        x1, y1 = p1
        cv2.rectangle(
            image,
            (min(x0, x1), min(y0, y1)),
            (max(x0, x1), max(y0, y1)),
            clr.HUD_YELLOW,
            small_thick(h),
        )

    def draw_HUD(self,
                 image: NDArray,
                 img_time: float,
                 opts: HudOpts,
                 calibration=None,
                 cx_cy_ori: tuple[float, float] = None,
                 scale: float = 1.0):
        h, w = image.shape[:2]

        cx_cy = (np.array(cx_cy_ori) * scale).astype(np.int32)

        draw_attitude = opts.draw_attitude
        draw_as_alt = opts.draw_as_alt
        draw_crosshairs = opts.draw_crosshairs
        draw_mode = opts.draw_mode
        map_transparency = opts.map_transparency

        # If image size changes
        if h != self.last_xy[0] or w != self.last_xy[1]:
            self.update_storage(h, w)

        att = self.attRdr.get_attitude_at(img_time)

        if draw_crosshairs:
            self.draw_crosshairs(image, cx_cy)

        if draw_as_alt:
            cv2.putText(image, f'AS: {att.speed_mps:.0f}', (int(h * 0.20), int(w * 0.5)),
                        cv2.FONT_HERSHEY_SIMPLEX, med_text(h), clr.HUD_GREEN, med_thick(h))
            self.draw_altitude(image, att.altitude_m)

        if draw_attitude and att.valid:
            self.draw_bankAngle(image,
                                att.roll_deg,
                                att.cmd_roll_deg,
                                att.pitch_deg,
                                att.cmd_pitch_deg)
            self.draw_pitchAngle(image, att.pitch_deg, att.roll_deg)
            self.draw_throttleResponse(image, att.throttle_pct)

        if draw_mode:
            self.draw_controlMode(image, att.mode)

        self.draw_runway(image, att, calibration, cx_cy, scale)

        # --- minimap ---
        self.draw_minimap(image, att, map_transparency)

        return att

    @staticmethod
    def _camera_offset_rotmat() -> np.ndarray:
        roll_deg, pitch_deg, yaw_deg = CAMERA_RPY_OFFSET_DEG
        rr = np.deg2rad(roll_deg)
        rp = np.deg2rad(pitch_deg)
        ry = np.deg2rad(yaw_deg)

        cr, sr = np.cos(rr), np.sin(rr)
        cp, sp = np.cos(rp), np.sin(rp)
        cy, sy = np.cos(ry), np.sin(ry)

        rx = np.array([
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr, cr],
        ], dtype=float)
        ry_m = np.array([
            [cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ], dtype=float)
        rz = np.array([
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=float)
        return rz @ ry_m @ rx

    @classmethod
    def _body_to_camera_points(cls, body_points: np.ndarray) -> np.ndarray:
        # Body frame is treated as [forward, right, down].
        # Camera frame for projection is [right, down, forward].
        lever_arm_body = np.asarray(CAMERA_LEVER_ARM_BODY_M, dtype=float).reshape(1, 3)
        camera_relative_body = body_points - lever_arm_body
        r_cb = cls._camera_offset_rotmat().T
        camera_aligned_body = (r_cb @ camera_relative_body.T).T
        return np.column_stack((
            camera_aligned_body[:, 1],
            camera_aligned_body[:, 2],
            camera_aligned_body[:, 0],
        ))

    @staticmethod
    def _fallback_camera_matrix(image: NDArray, cx_cy: tuple[int, int] | None) -> np.ndarray:
        h, w = image.shape[:2]
        cx = float(w * 0.5 if cx_cy is None else cx_cy[0])
        cy = float(h * 0.5 if cx_cy is None else cx_cy[1])
        fx = float(w)
        fy = float(w)
        return np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ], dtype=float)

    @staticmethod
    def _clip_polygon_to_near_plane(cam_points: np.ndarray, near_z: float = 1e-3) -> np.ndarray:
        if cam_points.shape[0] == 0:
            return cam_points

        clipped = []
        prev = cam_points[-1]
        prev_inside = prev[2] >= near_z

        for curr in cam_points:
            curr_inside = curr[2] >= near_z

            if curr_inside != prev_inside:
                dz = curr[2] - prev[2]
                if abs(dz) > 1e-12:
                    t = (near_z - prev[2]) / dz
                    clipped.append(prev + t * (curr - prev))

            if curr_inside:
                clipped.append(curr)

            prev = curr
            prev_inside = curr_inside

        if not clipped:
            return np.empty((0, 3), dtype=float)

        return np.asarray(clipped, dtype=float)

    def _project_body_points(
        self,
        image: NDArray,
        body_points: np.ndarray,
        calibration,
        cx_cy: tuple[int, int] | None,
        scale: float,
    ) -> np.ndarray | None:
        if body_points.shape[0] == 0:
            return None

        cam_points = self._body_to_camera_points(body_points)
        cam_points = self._clip_polygon_to_near_plane(cam_points)
        if cam_points.shape[0] < 2:
            return None

        if calibration is not None and getattr(calibration, 'validCal', False):
            k = calibration.getCameraMatrix()
        else:
            k = None
        if k is None:
            k = self._fallback_camera_matrix(image, cx_cy)
        elif abs(scale - 1.0) > 1e-9:
            k = np.asarray(k, dtype=float).copy()
            k[0, 0] *= scale
            k[1, 1] *= scale
            k[0, 2] = scale * (k[0, 2] + 0.5) - 0.5
            k[1, 2] = scale * (k[1, 2] + 0.5) - 0.5

        homog = (k @ cam_points.T).T
        return homog[:, :2] / homog[:, 2:3]

    def draw_runway(self, image, att, calibration, cx_cy, scale: float):
        corners_body = getattr(att, 'runway_corners_body_m', None)
        if not corners_body:
            return

        body_points = np.asarray(corners_body, dtype=float).reshape(-1, 3)
        pts_2d = self._project_body_points(image, body_points, calibration, cx_cy, scale)
        if pts_2d is None or pts_2d.shape[0] < 2:
            return

        h, w = image.shape[:2]
        pts_px = np.rint(pts_2d).astype(np.int32)
        clip_rect = (0, 0, w, h)
        thickness = med_thick(h)

        for idx in range(pts_px.shape[0]):
            p0 = tuple(int(v) for v in pts_px[idx])
            p1 = tuple(int(v) for v in pts_px[(idx + 1) % pts_px.shape[0]])
            ok, q0, q1 = cv2.clipLine(clip_rect, p0, p1)
            if ok:
                cv2.line(image, q0, q1, clr.HUD_YELLOW, thickness)

    @staticmethod
    def draw_crosshairs(image, cx_cy):
        if cx_cy is None:
            return
        cx, cy = cx_cy
        width = image.shape[1]
        height = image.shape[0]
        thickness = med_thick(height)

        crosshairsH = np.array([[cx + max(int(width / 50), 10), cy], [cx - max(int(width / 50), 10), cy]])
        crosshairsV = np.array([[cx, cy + max(int(height / 50), 10)], [cx, cy - max(int(height / 50), 10)]])

        cv2.polylines(image, [crosshairsH], True, clr.HUD_GREEN, thickness)
        cv2.polylines(image, [crosshairsV], True, clr.HUD_GREEN, thickness)

    def draw_bankAngle(self, image, bank_angle, cmd_bank_angle, pitch_angle, cmd_pitch_angle):
        x, y = self.last_xy
        green = clr.HUD_GREEN
        h, w = image.shape[:2]

        # --- Static bank indicator (prebuilt in self.bank_indicator_lines) ---
        cv2.polylines(image, self.bank_indicator_lines, False, green, med_thick(h))

        # --- Precompute trig once ---
        # Bank for "response"
        th_b = radians(bank_angle)
        s_b = sin(th_b)
        c_b = cos(th_b)
        # Commanded bank for "cmd"
        th_cb = radians(cmd_bank_angle)
        s_cb = sin(th_cb)
        c_cb = cos(th_cb)

        # Angle deltas as constants
        c15, s15 = cos(radians(15.0)), sin(radians(15.0))
        c10, s10 = cos(radians(10.0)), sin(radians(10.0))

        # --- Helper to make normalized HUD points (centered at 0.5,0.8) ---
        def bank_pts_base(s0, c0, r_main, r_wing, c_delta, s_delta):
            # theta, theta±delta via angle-addition (no extra trig calls)
            sx_p = s0 * c_delta + c0 * s_delta  # sin(theta+delta)
            cx_p = c0 * c_delta - s0 * s_delta  # cos(theta+delta)
            sx_m = s0 * c_delta - c0 * s_delta  # sin(theta-delta)
            cx_m = c0 * c_delta + s0 * s_delta  # cos(theta-delta)
            return [
                (0.5 + r_main * s0, 0.8 + r_main * c0),
                (0.5 + r_wing * sx_p, 0.8 + r_wing * cx_p),
                (0.5 + r_wing * sx_m, 0.8 + r_wing * cx_m),
            ]

        bank_pts = bank_pts_base(s_b, c_b, r_main=0.079, r_wing=0.050, c_delta=c15, s_delta=s15)
        cmd_bank_pts = bank_pts_base(s_cb, c_cb, r_main=0.079, r_wing=0.065, c_delta=c10, s_delta=s10)

        # Scale to pixels (avoid tiny broadcasting arrays)
        lines = np.array([(int(px * x), int(py * y)) for (px, py) in bank_pts], dtype=np.int32)
        cmd_lines = np.array([(int(px * x), int(py * y)) for (px, py) in cmd_bank_pts], dtype=np.int32)

        # --- Draw bank shapes ---
        cv2.polylines(image, [lines], True, green, med_thick(h))  # "Bank Cmd" in your comment
        cv2.fillPoly(image, [cmd_lines], green)  # "Bank Response" in your comment

        # --- Pitch command triangles ---
        # Use cos(-θ)=cos θ and sin(-θ)=-sin θ
        c_neg = c_b  # cos(-bank) ==  cos(bank)
        s_neg = -s_b  # sin(-bank) == -sin(bank)

        cx = 0.5 * x
        cy = 0.5 * y
        delta = (cmd_pitch_angle - pitch_angle) / 200.0

        # General rotation/translation helper for triangle vertices.
        # Given coefficients (alpha, beta) that were used as:
        #   X: alpha * cos_neg + beta * sin_neg
        #   Y: alpha * sin_neg - beta * cos_neg
        # convert to pixel offsets and add (cx, cy).
        def hv(alpha: float, beta: float):
            X = cx + x * (alpha * c_neg + beta * s_neg)
            Y = cy + y * (alpha * s_neg - beta * c_neg)
            return int(X), int(Y)

        # Left triangle vertices
        left_tri = np.array([
            hv(-0.01, delta),
            hv(-0.03, 0.01 + delta),
            hv(-0.03, -0.01 + delta),
        ], dtype=np.int32)

        # Right triangle vertices
        right_tri = np.array([
            hv(+0.01, delta),
            hv(+0.03, 0.01 + delta),
            hv(+0.03, -0.01 + delta),
        ], dtype=np.int32)

        cv2.polylines(image, [left_tri], True, green, med_thick(h))
        cv2.polylines(image, [right_tri], True, green, med_thick(h))

    def draw_pitchAngle(self, image, pitch_angle, bank_angle):
        x, y = self.last_xy
        cx = 0.5 * x
        cy = 0.5 * y

        img_height = image.shape[1]
        pitch_spacing = 16.0 * img_height / 864.0
        inner = 0.04 * x
        outer = 0.15 * x
        s_b = sin(radians(bank_angle + self.cam_bank_offset))
        c_b = cos(radians(bank_angle + self.cam_bank_offset))

        to_int = int
        green = clr.HUD_GREEN
        h, w = image.shape[:2]
        txt_scale = med_text(h)
        txt_thick = med_thick(h)

        k = int(round(pitch_angle / 10.0))
        candidates = []
        for dk in (-2, -1, 0, 1, 2):
            val = 10 * (k + dk)
            if -30 <= val <= 30 and abs(pitch_angle - val) < 25.0:
                candidates.append(val)

        for i in candidates:
            dy = -(pitch_angle - i) * pitch_spacing

            x_off = dy * s_b
            y_off = dy * c_b

            x1 = cx - outer * c_b - x_off
            y1 = cy + outer * s_b - y_off
            x2 = cx - inner * c_b - x_off
            y2 = cy + inner * s_b - y_off

            cv2.line(image,
                     (to_int(x1), to_int(y1)),
                     (to_int(x2), to_int(y2)),
                     green, txt_thick)

            x3 = cx + inner * c_b - x_off
            y3 = cy - inner * s_b - y_off
            x4 = cx + outer * c_b - x_off
            y4 = cy - outer * s_b - y_off

            cv2.line(image,
                     (to_int(x3), to_int(y3)),
                     (to_int(x4), to_int(y4)),
                     green, txt_thick)

            cv2.putText(
                image, str(int(i)),
                (to_int(x4 + x * 0.02), to_int(y4)),
                cv2.FONT_HERSHEY_SIMPLEX, txt_scale, green, txt_thick
            )

        cv2.circle(image, (int(cx), int(cy)), 5, green, txt_thick)

    def draw_altitude(self, image, alt):
        x, y = self.last_xy
        h, w = image.shape[:2]

        alt_text = f'ALT: {alt:.0f}'

        (width, height), baseline = cv2.getTextSize(alt_text,
                                                    cv2.FONT_HERSHEY_SIMPLEX,
                                                    med_text(h), med_thick(h))

        cv2.putText(image, alt_text,
                    (int(0.775 * x - width / 2.0), int(0.4 * y - height / 2.0)),
                    cv2.FONT_HERSHEY_SIMPLEX, med_text(h), clr.HUD_GREEN, med_thick(h))

    def draw_throttleResponse(self, image, cmd_throttle):
        x, y = self.last_xy
        r = 0.06
        theta = cmd_throttle * 2.450
        h, w = image.shape[:2]

        tri = np.array([[self.throttle_loc[0] + (np.sin(np.deg2rad(theta)) * x * (r * 0.95)),
                         self.throttle_loc[1] - (np.cos(np.deg2rad(theta)) * x * (r * 0.95))],
                        [self.throttle_loc[0] + (np.sin(np.deg2rad(theta + 5.0)) * x * (r * 0.8)),
                         self.throttle_loc[1] - (np.cos(np.deg2rad(theta + 5.0)) * x * (r * 0.6))],
                        [self.throttle_loc[0] + (np.sin(np.deg2rad(theta - 5.0)) * x * (r * 0.8)),
                         self.throttle_loc[1] - (np.cos(np.deg2rad(theta - 5.0)) * x * (r * 0.6))]], np.int32)

        cv2.polylines(image, self.throttle_circle_points, False, clr.HUD_GREEN, 2)

        cv2.fillPoly(image, [tri], clr.HUD_GREEN)
        (width, height), baseline = cv2.getTextSize(f'{cmd_throttle:.1f}%',
                                                    cv2.FONT_HERSHEY_SIMPLEX,
                                                    med_text(h), med_thick(h))
        cv2.putText(image, f'{cmd_throttle:.1f}%',
                    (int(self.throttle_loc[0] - width / 2),
                     int(self.throttle_loc[1] - height / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, med_text(h), clr.HUD_GREEN, med_thick(h))

    def draw_controlMode(self, image, mode):
        h, w = image.shape[:2]
        match mode:
            case ControlMode.controller:
                txt = "MODE: CNTL"
                display_clr = (255, 150, 0)
            case ControlMode.manual:
                txt = "MODE: MAN"
                display_clr = (255, 255, 0)
            case ControlMode.auto:
                txt = "MODE: AUTO"
                display_clr = clr.HUD_GREEN
            case _:
                txt = "MODE: ERR"
                display_clr = (0, 0, 255)

        cv2.putText(image, txt, self.controlMode_text_loc,
                    cv2.FONT_HERSHEY_SIMPLEX, med_text(h), display_clr, med_thick(h))

    @staticmethod
    def draw_playbackStats(image, lowPassFPS, target_fps, playback_mode, rt_speed, cam_to_log_time_offset, last_nonzero_sign):
        (h, w) = image.shape[:2]

        (txt_width, txt_height), base = cv2.getTextSize("I", cv2.FONT_HERSHEY_SIMPLEX, med_text(w), 4)
        pad = int(0.3 * txt_height)
        txt_pix_start_perRow = txt_height + pad
        cv2.putText(image, f"Offset: {cam_to_log_time_offset:+.2f}s",
                    (pad, txt_pix_start_perRow * 2 + pad), cv2.FONT_HERSHEY_SIMPLEX,
                    med_text(h), clr.BLACK, lrg_thick(h))
        cv2.putText(image, f"Offset: {cam_to_log_time_offset:+.2f}s",
                    (pad, txt_pix_start_perRow * 2 + pad), cv2.FONT_HERSHEY_SIMPLEX,
                    med_text(h), clr.HUD_YELLOW, med_thick(h))

        sign = '+' if last_nonzero_sign > 0 else '-'
        if playback_mode == PlaybackSpeed.Real_time:
            playspeed_text = 'Realtime: ' + sign + f'{rt_speed:.2f}'
        else:
            playspeed_text = 'FPS: ' + sign + f'{lowPassFPS:.2f}/{target_fps:.2f}'
        cv2.putText(image,
                    playspeed_text,
                    (pad, txt_pix_start_perRow * 3 + pad), cv2.FONT_HERSHEY_SIMPLEX,
                    med_text(h), clr.BLACK, lrg_thick(h))
        cv2.putText(image,
                    playspeed_text,
                    (pad, txt_pix_start_perRow * 3 + pad), cv2.FONT_HERSHEY_SIMPLEX,
                    med_text(h), clr.HUD_YELLOW, med_thick(h))


def draw_name_on_image(name, frame):
    h, w = frame.shape[:2]
    (width, height), base = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX,
                                            med_text(h), lrg_thick(h))
    pad = int(0.3 * height)
    cv2.putText(frame, name, (w - width - pad, h - height),
                cv2.FONT_HERSHEY_SIMPLEX, med_text(h), clr.BLACK, lrg_thick(h))
    cv2.putText(frame, name, (w - width - pad, h - height),
                cv2.FONT_HERSHEY_SIMPLEX, med_text(h), clr.HUD_GREEN, med_thick(h))


def draw_time_on_image(frame, time_str):
    h, w = frame.shape[:2]
    (time_width, _time_height), _base = cv2.getTextSize(time_str, cv2.FONT_HERSHEY_SIMPLEX,
                                                        med_text(h), lrg_thick(h))
    (_ref_width, ref_height), _ref_base = cv2.getTextSize("I", cv2.FONT_HERSHEY_SIMPLEX,
                                                          med_text(h), lrg_thick(h))

    pad = int(0.3 * ref_height)
    row_height = ref_height + pad
    x = w - time_width - pad
    y = h - row_height - ref_height

    cv2.putText(frame, time_str, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, med_text(h), clr.BLACK, lrg_thick(h))
    cv2.putText(frame, time_str, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, med_text(h), clr.HUD_GREEN, med_thick(h))
