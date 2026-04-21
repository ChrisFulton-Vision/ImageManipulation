import cv2
import numpy as np
import time
from support.viz.CVFontScaling import med_text
import support.viz.colors as clr

class CheckerboardResiduals:
    def __init__(self):

        self._cb_last_found = False
        self._cb_last_resid = 0.0
        self._cb_last_point_resid = 0.0
        self._cb_last_corners = None
        self._cb_last_ts = 0.0
        self._cb_throttle_sec = 0.05

    @staticmethod
    def _line_fit_residual_px(pts_xy: np.ndarray) -> float:
        """
        pts_xy: (N,2) float array
        Returns RMS perpendicular distance (pixels) to best-fit line.
        """
        if pts_xy.shape[0] < 2:
            return float("nan")

        # cv2.fitLine returns normalized direction (vx,vy) and a point (x0,y0) on the line
        vx, vy, x0, y0 = cv2.fitLine(pts_xy.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).flatten()

        # Perpendicular distance from point p to line through x0 with direction v:
        # dist = |(p-x0) x v| / ||v||, but ||v||≈1 from fitLine
        dx = pts_xy[:, 0] - x0
        dy = pts_xy[:, 1] - y0
        dist = np.abs(dx * vy - dy * vx)  # since ||v|| ~ 1
        return float(np.sqrt(np.mean(dist * dist)))

    @staticmethod
    def _line_fit_point_dists_px(pts_xy: np.ndarray) -> np.ndarray:
        """
        pts_xy: (N,2) float array
        Returns per-point perpendicular distance (pixels) to best-fit line.
        """
        if pts_xy.shape[0] < 2:
            return np.full((pts_xy.shape[0],), np.nan, dtype=np.float32)

        vx, vy, x0, y0 = cv2.fitLine(pts_xy.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        dx = pts_xy[:, 0] - x0
        dy = pts_xy[:, 1] - y0
        dist = np.abs(dx * vy - dy * vx)  # since ||v|| ~ 1
        return dist.astype(np.float32)


    def _chessboard_point_residuals(self, corners: np.ndarray, pattern: tuple[int, int]) -> np.ndarray:
        """
        corners: (N,1,2) from OpenCV, pattern=(cols,rows) inner corners.

        Returns per-point residual (pixels). For each corner we compute:
            r_i = max( dist_to_its_row_line , dist_to_its_col_line )

        This tends to highlight local warps / glare / bad detections better than a single RMS.
        """
        cols, rows = pattern
        pts = corners.reshape(-1, 2).astype(np.float32)  # (N,2)
        if pts.shape[0] != cols * rows:
            return np.full((pts.shape[0],), np.nan, dtype=np.float32)

        grid = pts.reshape(rows, cols, 2)  # [r,c,(x,y)]

        # Accumulate distances from row and col fits
        row_d = np.zeros((rows, cols), dtype=np.float32)
        col_d = np.zeros((rows, cols), dtype=np.float32)

        for r in range(rows):
            row_d[r, :] = self._line_fit_point_dists_px(grid[r, :, :])

        for c in range(cols):
            col_d[:, c] = self._line_fit_point_dists_px(grid[:, c, :])

        per = np.maximum(row_d, col_d).reshape(-1)
        return per


    def _chessboard_straightness_residual(self, corners: np.ndarray, pattern: tuple[int, int]) -> tuple[
        float, float, float]:
        """
        corners: (N,1,2) from OpenCV, pattern=(cols,rows) inner corners.
        Returns (rms_rows, rms_cols, rms_all) in pixels.
        """
        cols, rows = pattern
        pts = corners.reshape(-1, 2).astype(np.float32)  # (N,2)
        if pts.shape[0] != cols * rows:
            return float("nan"), float("nan"), float("nan")

        grid = pts.reshape(rows, cols, 2)  # row-major: [r,c,(x,y)]

        row_rms = []
        for r in range(rows):
            row_rms.append(self._line_fit_residual_px(grid[r, :, :]))

        col_rms = []
        for c in range(cols):
            col_rms.append(self._line_fit_residual_px(grid[:, c, :]))

        rms_rows = float(np.nanmean(row_rms))
        rms_cols = float(np.nanmean(col_rms))
        rms_all = float(np.sqrt((rms_rows * rms_rows + rms_cols * rms_cols) / 2.0))
        return rms_rows, rms_cols, rms_all


    def draw_chessboard(self, markup_frame, gray_frame, cb_pattern):

        now = time.monotonic()
        if (now - self._cb_last_ts) >= self._cb_throttle_sec:
            _cb_last_ts = now

            flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
            found, corners = cv2.findChessboardCornersSB(gray_frame,
                                                         cb_pattern,
                                                         flags)

            self._cb_last_found = bool(found)
            self._cb_last_corners = corners if found else None

            # NEW: compute residual on update ticks
            if self._cb_last_found and self._cb_last_corners is not None:
                rr, cc, allr = self._chessboard_straightness_residual(self._cb_last_corners,
                                                                 cb_pattern)
                self._cb_last_resid = (rr, cc, allr)
                self._cb_last_point_resid = self._chessboard_point_residuals(self._cb_last_corners, cb_pattern)
            else:
                self._cb_last_resid = None
                self._cb_last_point_resid = None

        # Draw from cache (smooth display)
        if self._cb_last_found and self._cb_last_corners is not None:
            # drawChessboardCorners(self.markup_frame,
            #                       self._cb_pattern,
            #                       self._cb_last_corners, True)

            # NEW: highlight bad corners (larger residual => warmer color)
            per = self._cb_last_point_resid
            if per is not None:
                pts = self._cb_last_corners.reshape(-1, 2)
                # Absolute scaling: choose a pixel residual that counts as "hot"
                HOT_PX = 1.0

                for (x, y), r in zip(pts, per):
                    if not np.isfinite(r):
                        continue

                    bgr = self.residual_to_bgr(r)

                    cx, cy = int(round(x)), int(round(y))
                    cv2.circle(markup_frame, (cx, cy), 4,
                               (0, 0, 0), -1, cv2.LINE_AA)  # black underlay
                    cv2.circle(markup_frame, (cx, cy), 3,
                               bgr, -1, cv2.LINE_AA)

            # NEW: overlay residual
            if self._cb_last_resid is not None:
                rr, cc, allr = self._cb_last_resid
                txt = f"CB resid (px): row={rr:.2f} col={cc:.2f} rms={allr:.2f}"
            else:
                txt = "CB resid: ---"

        else:
            txt = "CB: NOT FOUND"

        h, w, _ = markup_frame.shape
        # Put text on the image (top-right)
        (txt_width, txt_height), base = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, med_text(w), 4)
        org = (w - txt_width - 10, 10 + txt_height)

        cv2.putText(markup_frame,
                    txt, org, cv2.FONT_HERSHEY_SIMPLEX,
                    med_text(w), clr.BLACK,
                    4, cv2.LINE_AA)
        cv2.putText(markup_frame,
                    txt, org, cv2.FONT_HERSHEY_SIMPLEX,
                    med_text(w), clr.LIGHTBLUE,
                    2, cv2.LINE_AA)


    @staticmethod
    def residual_to_bgr(r_px, hot_px=1.0):
        """
        Map absolute residual (pixels) to BGR color using HSV.
        0 px -> green
        hot_px -> red
        """
        t = np.clip(r_px / hot_px, 0.0, 1.0)

        # Hue: green (60) -> red (0)
        h = int((1.0 - t) * 60)
        s = 255
        v = 255

        hsv = np.uint8([[[h, s, v]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        return int(bgr[0]), int(bgr[1]), int(bgr[2])