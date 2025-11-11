import cv2
import numpy as np
from numpy import sin, cos, deg2rad
from SupportModules.AttitudeInterpreter import AttitudeReader as AttRdr
from SupportModules.AttitudeInterpreter import ControlMode
from SupportModules.CVFontScaling import med_text
from numpy.typing import NDArray
from math import sin, cos, radians

HUD_GREEN = (0, 255, 0)
HUD_YELLOW = (0, 255, 255)


class HUD_Marker:
    def __init__(self):
        self.cam_bank_offset = 0.0  # deg
        self.attRdr = AttRdr()
        self.bank_indicator_points = self.create_bank_indicator()
        self.last_xy = (864, 864)
        self.bank_indicator_lines = None
        self.controlMode_text_loc = None
        self.throttle_loc = None
        self.throttle_circle_points = None

        self.update_storage(864, 864)

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

    def read_attitude_files(self, filepath):
        self.attRdr = AttRdr()
        self.attRdr.read_files(filepath)

    def update_offset(self, delta_offset):
        self.attRdr.offset += delta_offset

    @property
    def offset(self):
        return self.attRdr.offset

    def draw_HUD(self, image: NDArray, img_time: float, box_around: bool):
        x, y, _ = image.shape

        # If image size changes
        if x != self.last_xy[0] or y != self.last_xy[1]:
            self.update_storage(x, y)

        speed, alt, bank_angle, cmd_bank_angle, pitch_angle, cmd_pitch_angle, cmd_throttle, mode = self.attRdr.get_attitude_at(
            img_time)  # + 173.11338 - 11.658461)

        # Speed
        cv2.putText(image, f'AS: {speed:.0f}', (int(x * 0.20), int(y * 0.5)),
                    cv2.FONT_HERSHEY_SIMPLEX, med_text(), HUD_GREEN, 2)

        self.draw_bankAngle(image, bank_angle, cmd_bank_angle, pitch_angle, cmd_pitch_angle)

        self.draw_pitchAngle(image, pitch_angle, bank_angle)

        self.draw_altitude(image, alt)

        self.draw_throttleResponse(image, cmd_throttle)

        self.draw_controlMode(image, mode)

        # cv2.putText(image, f'BnkOffset: {self.cam_bank_offset:.1f}', (100,100), cv2.FONT_HERSHEY_SIMPLEX, med_text(), HUD_YELLOW, 2)

        if box_around:
            cv2.rectangle(image, (0, 0), (x - 1, y - 1), HUD_YELLOW, 10)


    def draw_bankAngle(self, image, bank_angle, cmd_bank_angle, pitch_angle, cmd_pitch_angle):
        x, y = self.last_xy
        green = HUD_GREEN
        draw_lines = cv2.polylines
        fill_poly = cv2.fillPoly

        # --- Static bank indicator (prebuilt in self.bank_indicator_lines) ---
        draw_lines(image, self.bank_indicator_lines, False, green, 2)

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
        draw_lines(image, [lines], True, green, 2)  # "Bank Cmd" in your comment
        fill_poly(image, [cmd_lines], green)  # "Bank Response" in your comment

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
            return (int(X), int(Y))

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

        draw_lines(image, [left_tri], True, green, 2)
        draw_lines(image, [right_tri], True, green, 2)

    def draw_pitchAngle(self, image, pitch_angle, bank_angle):
        x, y = self.last_xy
        cx = 0.5 * x
        cy = 0.5 * y

        # Hoist constants & lookups
        pitch_spacing = 16.0
        inner = 0.04 * x
        outer = 0.15 * x
        s_b = sin(radians(bank_angle + self.cam_bank_offset))
        c_b = cos(radians(bank_angle + self.cam_bank_offset))
        putText = cv2.putText
        drawLine = cv2.line
        to_int = int  # local alias is slightly faster than global lookup
        green = HUD_GREEN
        # Cache the scale once per call (your no-arg cached version)
        txt_scale = med_text()

        # Compute only the ticks that can possibly render (±25° window, 10° spacing)
        # Center tick index
        k = int(round(pitch_angle / 10.0))
        # Candidate tick values in degrees, clamped to [-30, 30]
        candidates = []
        for dk in (-2, -1, 0, 1, 2):  # at most 5 ticks
            val = 10 * (k + dk)
            if -30 <= val <= 30 and abs(pitch_angle - val) < 25.0:
                candidates.append(val)

        for i in candidates:
            # vertical spacing; dy < 0 means higher on screen
            dy = -(pitch_angle - i) * pitch_spacing

            # offset rotated by bank
            x_off = dy * s_b
            y_off = dy * c_b

            # left line: outer -> inner
            x1 = cx - outer * c_b - x_off
            y1 = cy + outer * s_b - y_off
            x2 = cx - inner * c_b - x_off
            y2 = cy + inner * s_b - y_off

            drawLine(image, (to_int(x1), to_int(y1)), (to_int(x2), to_int(y2)), green, 2)

            # right line: inner -> outer
            x3 = cx + inner * c_b - x_off
            y3 = cy - inner * s_b - y_off
            x4 = cx + outer * c_b - x_off
            y4 = cy - outer * s_b - y_off

            drawLine(image, (to_int(x3), to_int(y3)), (to_int(x4), to_int(y4)), green, 2)

            # label; avoid f-string format cost by int()
            putText(
                image, str(int(i)),
                (to_int(x4 + x * 0.02), to_int(y4)),
                cv2.FONT_HERSHEY_SIMPLEX, txt_scale, green, 2
            )

        cv2.circle(image, (int(cx), int(cy)), 5, green, 2)

    def draw_altitude(self, image, alt):
        x, y = self.last_xy

        alt_text = f'ALT: {alt:.0f}'

        (width, height), baseline = cv2.getTextSize(alt_text, cv2.FONT_HERSHEY_SIMPLEX,
                                                    med_text(), 2)

        cv2.putText(image, alt_text,
                    (int(0.775 * x - width / 2.0), int(0.4 * y - height / 2.0)),
                    cv2.FONT_HERSHEY_SIMPLEX, med_text(), HUD_GREEN, 2)

        # cv2.rectangle(image,
        #               (int(0.773 * x - width / 2.0 ), int(0.4 * y - height * 2.0 )),
        #               (int(0.777 * x + width / 2.0 ), int(0.4 * y + height / 2.0)),
        #               HUD_GREEN,
        #               2)

    def draw_throttleResponse(self, image, cmd_throttle):
        # Throttle response
        x, y = self.last_xy
        r = 0.06
        theta = cmd_throttle * 2.450

        tri = np.array([[self.throttle_loc[0] + (np.sin(np.deg2rad(theta)) * x * (r * 0.95)),
                         self.throttle_loc[1] - (np.cos(np.deg2rad(theta)) * x * (r * 0.95))],
                        [self.throttle_loc[0] + (np.sin(np.deg2rad(theta + 5.0)) * x * (r * 0.8)),
                         self.throttle_loc[1] - (np.cos(np.deg2rad(theta + 5.0)) * x * (r * 0.6))],
                        [self.throttle_loc[0] + (np.sin(np.deg2rad(theta - 5.0)) * x * (r * 0.8)),
                         self.throttle_loc[1] - (np.cos(np.deg2rad(theta - 5.0)) * x * (r * 0.6))]], np.int32)

        cv2.polylines(image, self.throttle_circle_points, False, HUD_GREEN, 2)  # Arc

        cv2.fillPoly(image, [tri], HUD_GREEN)  # Triangle Pointer
        (width, height), baseline = cv2.getTextSize(f'{cmd_throttle:.1f}%', cv2.FONT_HERSHEY_SIMPLEX,
                                                    med_text(), 2)
        cv2.putText(image, f'{cmd_throttle:.1f}%',
                    (int(self.throttle_loc[0] - width / 2), int(self.throttle_loc[1] - height / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, med_text(), HUD_GREEN, 2)

    def draw_controlMode(self, image, mode):
        if mode == ControlMode.controller:
            cv2.putText(image, "MODE: CNTL", self.controlMode_text_loc,
                        cv2.FONT_HERSHEY_SIMPLEX, med_text(), (255, 150, 0), 2)
        if mode == ControlMode.manual:
            cv2.putText(image, "MODE: MAN", self.controlMode_text_loc,
                        cv2.FONT_HERSHEY_SIMPLEX, med_text(), (255, 255, 0), 2)
        if mode == ControlMode.auto:
            cv2.putText(image, "MODE: AUTO", self.controlMode_text_loc,
                        cv2.FONT_HERSHEY_SIMPLEX, med_text(), HUD_GREEN, 2)
        if mode == ControlMode.error:
            cv2.putText(image, "MODE: ERR", self.controlMode_text_loc,
                        cv2.FONT_HERSHEY_SIMPLEX, med_text(), (0, 0, 255), 2)
