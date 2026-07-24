from typing import Any
from pathlib import Path
import time

import cv2
import numpy as np
from numpy.typing import NDArray

from support.core.pixel_kalmanFilter import KalmanFilter as PixelKalmanFilter
from support.core.enums import ImageSource
from support.io.attitude_interpreter import CAMERA_RPY_OFFSET_DEG
import support.gui.UserSelectQueue as GuiQueue
from support.mathHelpers.quaternions import Quaternion as q, mat2quat
from support.mathHelpers.single_feature_geometry import (
    camera_matrix_from_calibration,
    estimate_single_feature_from_center_width,
)
from support.mathHelpers.twoD_to_threeD import solveQnP
from support.runtime.fg_singleTarget import (
    build_factor_graph_output,
    build_hyper_focus_plan,
    factor_graph_projection_matrix,
    run_factor_graph_step,
)
from support.vision.draw_circle_and_mask import dim_except_circle
import support.viz.colors as clr
from support.viz.CVFontScaling import lrg_thick, med_text, med_thick, small_text


class PoseRuntime:
    """Owns detector/session-driven pose estimation and factor-graph runtime."""

    def __init__(self, owner: Any):
        self.owner = owner
        self._truth_lookup_source = None
        self._truth_lookup = None
        self._yolo_sessions_by_dir: dict[str, Any] = {}
        self._active_yolo_dir: str | None = None
        self._feature_kfs: dict[int, PixelKalmanFilter] = {}
        self._feature_detection_timeout_s = 0.5
        self._feature_last_detection_time_s: dict[int, float] = {}
        self._last_feature_kf_time_s: float | None = None
        self._feature_kf_stabilization_ref_R_wc: np.ndarray | None = None
        self._feature_kf_sigma_proc: float = 0.5

    @staticmethod
    def _frame_time_s(ctx: GuiQueue.FrameCtx) -> float:
        if ctx.img_time is not None:
            return float(ctx.img_time)
        return time.monotonic()

    @staticmethod
    def _parse_display_feature_ids(raw_value: str) -> set[int] | None:
        if raw_value is None:
            return None
        text = str(raw_value).strip()
        if text == "":
            return None
        feature_ids: set[int] = set()
        for token in text.split(","):
            item = token.strip()
            if item == "":
                continue
            try:
                feature_ids.add(int(item))
            except ValueError:
                continue
        return feature_ids if len(feature_ids) > 0 else None

    def _configure_feature_kf(self, kf: PixelKalmanFilter, width_px: float, height_px: float) -> None:
        kf.set_image_size(width_px, height_px)
        kf.set_sigma_proc(self._feature_kf_sigma_proc)

    def _get_feature_kf(self, class_id: int, width_px: float, height_px: float) -> PixelKalmanFilter:
        kf = self._feature_kfs.get(int(class_id))
        if kf is None:
            kf = PixelKalmanFilter(width_px=width_px, height_px=height_px)
            self._configure_feature_kf(kf, width_px, height_px)
            self._feature_kfs[int(class_id)] = kf
        else:
            self._configure_feature_kf(kf, width_px, height_px)
        return kf

    def _set_feature_kf_sigma_proc(self, sigma_proc: float, width_px: float, height_px: float) -> None:
        sigma_proc = float(sigma_proc)
        if self._feature_kf_sigma_proc == sigma_proc and not self._feature_kfs:
            return
        self._feature_kf_sigma_proc = sigma_proc
        for kf in self._feature_kfs.values():
            self._configure_feature_kf(kf, width_px, height_px)

    def _reset_feature_kf_bank(self) -> None:
        self._feature_kfs.clear()
        self._feature_last_detection_time_s.clear()
        self._feature_kf_stabilization_ref_R_wc = None

    def _disable_feature_kf_bank(self) -> None:
        self._reset_feature_kf_bank()
        self._last_feature_kf_time_s = None

    def _reset_yolo_runtime_state(self) -> None:
        self._disable_feature_kf_bank()
        if self.owner.pnpDrawer is not None:
            self.owner.pnpDrawer.last_q_vec = None
            self.owner.pnpDrawer.last_t_vec = None
            self.owner.pnpDrawer.last_wq_vec = None
            self.owner.pnpDrawer.last_wt_vec = None
            self.owner.pnpDrawer.last_pnp_rvec = None
            self.owner.pnpDrawer.last_pnp_tvec = None

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
    def _camera_cv_from_body_rotmat(cls) -> np.ndarray:
        r_cb = cls._camera_offset_rotmat().T
        perm = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ], dtype=float)
        return perm @ r_cb

    def _current_feature_stabilization_homographies(
        self,
        width_px: float,
        height_px: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        attitude = getattr(self.owner, "own_attitude", None)
        if attitude is None or not getattr(attitude, "valid", False):
            return None

        R_wr = attitude.rotmat_wr()
        if R_wr is None:
            return None

        R_cam_from_body = self._camera_cv_from_body_rotmat()
        R_body_from_cam = R_cam_from_body.T
        R_wc = np.asarray(R_wr, dtype=float) @ R_body_from_cam

        if self._feature_kf_stabilization_ref_R_wc is None:
            self._feature_kf_stabilization_ref_R_wc = np.asarray(R_wc, dtype=float)

        if self.owner.calibration is not None and self.owner.calibration.validCal:
            K = np.asarray(self.owner.calibration.getCameraMatrix(), dtype=np.float64)
        else:
            cx = float(width_px) * 0.5
            cy = float(height_px) * 0.5
            fx = float(width_px)
            fy = float(width_px)
            K = np.array([
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)

        try:
            K_inv = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            return None

        H_stab_from_curr = K @ (self._feature_kf_stabilization_ref_R_wc.T @ R_wc) @ K_inv
        H_curr_from_stab = K @ (R_wc.T @ self._feature_kf_stabilization_ref_R_wc) @ K_inv
        return (
            np.asarray(H_stab_from_curr, dtype=np.float64),
            np.asarray(H_curr_from_stab, dtype=np.float64),
        )

    @staticmethod
    def _apply_homography_point(H: np.ndarray | None, point_xy: np.ndarray) -> np.ndarray | None:
        if H is None:
            return np.asarray(point_xy, dtype=np.float64)
        x = float(point_xy[0])
        y = float(point_xy[1])
        mapped = np.asarray(H, dtype=np.float64) @ np.array([x, y, 1.0], dtype=np.float64)
        if abs(float(mapped[2])) < 1e-12:
            return None
        return np.asarray(mapped[:2] / mapped[2], dtype=np.float64)

    def _feature_track_metadata_from_kf(
        self,
        kf: PixelKalmanFilter,
        H_curr_from_stab: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float, bool, np.ndarray] | None:
        pos_cov_px = kf.position_covariance_px()
        gate_cov_px = kf.gate_ellipse_covariance_px()
        if (
            pos_cov_px is None
            or pos_cov_px.shape != (2, 2)
            or not np.all(np.isfinite(pos_cov_px))
            or gate_cov_px is None
            or gate_cov_px.shape != (2, 2)
            or not np.all(np.isfinite(gate_cov_px))
            or kf.x is None
        ):
            return None
        center_px = np.array(
            [
                float(kf.x[0]) * float(kf.width_px),
                float(kf.x[1]) * float(kf.height_px),
            ],
            dtype=np.float64,
        )
        if H_curr_from_stab is not None:
            center_curr = self._apply_homography_point(H_curr_from_stab, center_px)
            if center_curr is None or not np.all(np.isfinite(center_curr)):
                return None
            center_px = center_curr
        return pos_cov_px, gate_cov_px, float(kf.max_mahalanobis_sq), bool(kf.last_used_measurement), center_px

    def _build_feature_kf_metadata(self, prepared, frame_time_s: float, width_px: float, height_px: float, idsNamesLocs):
        if prepared is None:
            prepared_class_ids = []
            prepared_centers = np.empty((0, 2), dtype=np.float64)
            prepared_pose_ids = []
        else:
            prepared_class_ids = list(prepared.class_ids)
            prepared_centers = np.asarray(prepared.centers_for_pnp, dtype=np.float64)
            prepared_pose_ids = list(prepared.pose_class_ids)

        freeze_only = False
        if (
            self._last_feature_kf_time_s is not None
            and frame_time_s < float(self._last_feature_kf_time_s)
        ):
            self._reset_feature_kf_bank()
        elif (
            self._last_feature_kf_time_s is not None
            and frame_time_s == float(self._last_feature_kf_time_s)
        ):
            freeze_only = True
        self._last_feature_kf_time_s = float(frame_time_s)
        stab_h = self._current_feature_stabilization_homographies(width_px, height_px)
        H_stab_from_curr = None if stab_h is None else stab_h[0]
        H_curr_from_stab = None if stab_h is None else stab_h[1]

        seen_ids = set()
        sigma_yolo_2N = np.full((2 * len(prepared_pose_ids),), 1e6, dtype=np.float64)
        feature_gate_covs = np.full((len(prepared_class_ids), 2, 2), np.nan, dtype=np.float64)
        feature_gate_mahal_sq = np.full((len(prepared_class_ids),), np.nan, dtype=np.float64)
        feature_used = np.zeros((len(prepared_class_ids),), dtype=bool)
        tracker_stats: dict[int, tuple[np.ndarray, np.ndarray, float, bool, np.ndarray]] = {}

        meas_by_cid: dict[int, np.ndarray] = {}
        for class_id, center_px in zip(prepared_class_ids, prepared_centers):
            meas_by_cid[int(class_id)] = np.asarray(center_px, dtype=np.float64)

        active_ids = sorted(set(self._feature_kfs.keys()) | set(meas_by_cid.keys()))
        for cid in active_ids:
            seen_ids.add(cid)
            has_measurement = cid in meas_by_cid
            if not has_measurement and cid not in self._feature_kfs:
                continue
            kf = self._get_feature_kf(cid, width_px, height_px)
            last_detection_time_s = self._feature_last_detection_time_s.get(cid)

            if (
                not freeze_only
                and last_detection_time_s is not None
                and (frame_time_s - float(last_detection_time_s)) > self._feature_detection_timeout_s
            ):
                kf.reset()
                self._feature_last_detection_time_s.pop(cid, None)
                last_detection_time_s = None

            if not freeze_only:
                if has_measurement:
                    center_px = meas_by_cid[cid]
                    if H_stab_from_curr is not None:
                        center_px_stab = self._apply_homography_point(H_stab_from_curr, center_px)
                        if center_px_stab is None or not np.all(np.isfinite(center_px_stab)):
                            continue
                        center_px = center_px_stab
                    z = np.array([
                        float(center_px[0]) / max(width_px, 1.0),
                        float(center_px[1]) / max(height_px, 1.0),
                    ], dtype=np.float64)
                    self._feature_last_detection_time_s[cid] = float(frame_time_s)
                else:
                    z = None
                kf.update_KF(new_time=frame_time_s, z=z)

            meta = self._feature_track_metadata_from_kf(kf, H_curr_from_stab=H_curr_from_stab)
            if meta is None:
                continue
            pos_cov_px, gate_cov_px, gate_mahal_sq, used_now, center_px_now = meta
            tracker_stats[cid] = (
                pos_cov_px,
                gate_cov_px,
                gate_mahal_sq,
                used_now,
                center_px_now,
            )

        for idx, cid in enumerate(prepared_class_ids):
            stat = tracker_stats.get(int(cid))
            if stat is None:
                continue
            _pos_cov_px, gate_cov_px, gate_mahal_sq, used, _center_px = stat
            feature_gate_covs[idx] = gate_cov_px
            feature_gate_mahal_sq[idx] = gate_mahal_sq
            feature_used[idx] = used

        for idx, cid in enumerate(prepared_pose_ids):
            stat = tracker_stats.get(int(cid))
            if stat is None:
                continue
            _pos_cov_px, gate_cov_px, _gate_mahal_sq, used, _center_px = stat
            sigmas = np.sqrt(np.maximum(np.diag(gate_cov_px), 1e-6))
            if not used:
                sigmas *= 2.0
            sigma_yolo_2N[2 * idx: 2 * idx + 2] = sigmas

        kfest_object_points: list[list[float]] = []
        kfest_image_points: list[list[float]] = []
        kfest_class_ids: list[int] = []
        kfest_sigma_2N: list[float] = []
        kfest_position_covs: list[np.ndarray] = []
        kfest_gate_covs: list[np.ndarray] = []
        kfest_gate_mahal_sq: list[float] = []
        kfest_used: list[bool] = []
        for cid in sorted(tracker_stats.keys()):
            if cid < 0 or cid >= len(idsNamesLocs):
                continue
            last_detection_time_s = self._feature_last_detection_time_s.get(cid)
            if last_detection_time_s is None:
                continue
            if (frame_time_s - float(last_detection_time_s)) > self._feature_detection_timeout_s:
                continue
            x, y, z = idsNamesLocs[cid][2:]
            pos_cov_px, gate_cov_px, _gate_mahal_sq, used, center_px = tracker_stats[cid]
            kfest_object_points.append([x, y, z])
            kfest_image_points.append([float(center_px[0]), float(center_px[1])])
            kfest_class_ids.append(int(cid))
            kfest_position_covs.append(np.asarray(pos_cov_px, dtype=np.float64))
            kfest_gate_covs.append(np.asarray(gate_cov_px, dtype=np.float64))
            kfest_gate_mahal_sq.append(float(_gate_mahal_sq))
            kfest_used.append(bool(used))
            sigmas = np.sqrt(np.maximum(np.diag(gate_cov_px), 1e-6))
            if not used:
                sigmas *= 2.0
            kfest_sigma_2N.extend([float(sigmas[0]), float(sigmas[1])])

        stale_ids = []
        for cid, kf in self._feature_kfs.items():
            if freeze_only or cid in seen_ids or kf.lastStateTime is None:
                continue
            last_detection_time_s = self._feature_last_detection_time_s.get(cid)
            if last_detection_time_s is None:
                stale_ids.append(cid)
                continue
            if (frame_time_s - float(last_detection_time_s)) > self._feature_detection_timeout_s:
                stale_ids.append(cid)
        for cid in stale_ids:
            self._feature_kfs.pop(cid, None)
            self._feature_last_detection_time_s.pop(cid, None)

        return (
            sigma_yolo_2N,
            feature_gate_covs,
            feature_gate_mahal_sq,
            feature_used,
            np.asarray(kfest_object_points, dtype=np.float64),
            np.asarray(kfest_image_points, dtype=np.float64),
            kfest_class_ids,
            np.asarray(kfest_sigma_2N, dtype=np.float64),
            np.asarray(kfest_position_covs, dtype=np.float64),
            np.asarray(kfest_gate_covs, dtype=np.float64),
            np.asarray(kfest_gate_mahal_sq, dtype=np.float64),
            np.asarray(kfest_used, dtype=bool),
        )

    @staticmethod
    def _draw_bottom_left_text(
        markup_frame: NDArray,
        text: str,
        row_idx: int,
        color,
    ) -> None:
        h, w, _ = markup_frame.shape
        (_txt_width, txt_height), _base = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            med_text(w),
            med_thick(h),
        )
        pad = int(0.3 * txt_height)
        txt_height_per_row = txt_height + pad
        loc = (pad, h - (row_idx + 1) * txt_height_per_row - pad)

        cv2.putText(
            markup_frame,
            text,
            loc,
            cv2.FONT_HERSHEY_SIMPLEX,
            med_text(markup_frame.shape[0]),
            clr.BLACK,
            lrg_thick(h),
        )
        cv2.putText(
            markup_frame,
            text,
            loc,
            cv2.FONT_HERSHEY_SIMPLEX,
            med_text(markup_frame.shape[0]),
            color,
            med_thick(h),
        )

    def _get_truth_lookup(self) -> dict[int, np.ndarray]:
        if self.owner.ThreeDTruthPoints is None:
            self.owner.loadTruthPoints()

        truth_points = self.owner.ThreeDTruthPoints.truthPoints
        if truth_points is not self._truth_lookup_source:
            self._truth_lookup_source = truth_points
            self._truth_lookup = {
                int(k): np.asarray(v, dtype=np.float64)
                for k, v in truth_points.items()
            }
        return self._truth_lookup

    def _matched_truth_correspondences(self) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        centers = self.owner.centers
        detect_ids = self.owner.detectIDS
        if centers is None or detect_ids is None or len(centers) < 6:
            return None, None

        lookup = self._get_truth_lookup()
        object_points = []
        image_points = []

        for detect_id, center in zip(detect_ids, centers):
            key = int(detect_id[0])
            obj_pt = lookup.get(key)
            if obj_pt is None:
                continue
            object_points.append(obj_pt)
            image_points.append(center)

        if len(object_points) < 6:
            return None, None

        return (
            np.asarray(object_points, dtype=np.float64),
            np.asarray(image_points, dtype=np.float64),
        )

    def detect_april_tags(
        self,
        frame: NDArray,
        markup_frame: NDArray,
        ctx: GuiQueue.FrameCtx,
        args,
    ) -> None:
        from support.vision.aprilTag_detection_and_aligment import (
            detect_apriltags_refined,
            draw_apriltag_detections,
            parse_apriltag_args,
        )

        opts = parse_apriltag_args(args)

        if self.owner.detector is None:
            self.owner.createDetector()

        result = detect_apriltags_refined(
            detector=self.owner.detector,
            markup_frame=markup_frame,
            scale=opts.scale,
        )

        self.owner.centers = result.centers
        self.owner.detectIDS = result.ids

        if not result.ids:
            return

        if opts.inpaint:
            from support.vision.aprilTag_detection_and_aligment import inpaint_apriltags

            inpaint_apriltags(
                markup_frame=markup_frame,
                gray_small=result.small_gray,
                corners_small=result.corners_small,
            )
        else:
            draw_apriltag_detections(
                markup_frame=markup_frame,
                refined_corners_per_marker=result.refined_corners_per_marker,
                ids=result.ids,
            )

        if opts.pnp:
            self.pnp_3d_truth_points(frame, markup_frame, ctx, ())

        if opts.qnp:
            self.qnp_3d_truth_points(frame, markup_frame, ctx, ())

    def pnp_3d_truth_points(
        self,
        frame: NDArray,
        markup_frame: NDArray,
        ctx: GuiQueue.FrameCtx,
        args,
    ) -> None:
        points, centers = self._matched_truth_correspondences()
        if points is not None:
            dist_params = np.zeros((5,))

            ret, rvec, tvec, *_ = cv2.solvePnPRansac(
                objectPoints=points,
                imagePoints=centers,
                cameraMatrix=self.owner.calibration.getCameraMatrix(),
                distCoeffs=dist_params,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

            if ret:
                projected_points_orig, _ = cv2.projectPoints(
                    self.owner.ThreeDTruthPoints.getTruthPointsNumpy(),
                    rvec=rvec,
                    tvec=tvec,
                    cameraMatrix=self.owner.calibration.getCameraMatrix(),
                    distCoeffs=dist_params,
                )

                self.owner.plotOnImg(
                    markup_frame,
                    projected_points_orig[:, 0, :].astype(int),
                    list(self.owner.ThreeDTruthPoints.getTruthPointsDict().keys()),
                    clr.LIGHTBLUE,
                )

                quat_pnp, vect_pnp = q.fromOpenCV_toAftr_rvec(rvec, tvec)

                self.owner.pnpResult = (quat_pnp, vect_pnp)
                orient_text = "Orientation (quat) From Truth Points: " + format(quat_pnp, "ijk.6f")
                (_txt_w, txt_h), _ = cv2.getTextSize(
                    orient_text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    small_text(markup_frame.shape[0]),
                    4,
                )

                cv2.putText(
                    markup_frame,
                    orient_text,
                    (50, txt_h + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    small_text(markup_frame.shape[0]),
                    clr.BLACK,
                    4,
                )
                cv2.putText(
                    markup_frame,
                    orient_text,
                    (50, txt_h + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    small_text(markup_frame.shape[0]),
                    clr.LIGHTBLUE,
                    2,
                )
                location_text = "Location From Truth Frame: " + np.array2string(vect_pnp)
                cv2.putText(
                    markup_frame,
                    location_text,
                    (50, 2 * txt_h + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    small_text(markup_frame.shape[0]),
                    clr.BLACK,
                    4,
                )
                cv2.putText(
                    markup_frame,
                    location_text,
                    (50, 2 * txt_h + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    small_text(markup_frame.shape[0]),
                    clr.LIGHTBLUE,
                    2,
                )

    def qnp_3d_truth_points(
        self,
        frame: NDArray,
        markup_frame: NDArray,
        ctx: GuiQueue.FrameCtx,
        args,
    ) -> None:
        points, centers = self._matched_truth_correspondences()
        if points is not None:
            quat, vect, *_ = solveQnP(points, centers, self.owner.calibration, True)
            xyz_proj = quat * self.owner.ThreeDTruthPoints.getTruthPointsNumpy() + vect

            q_aftr_from_cv = mat2quat(
                np.array(
                    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
                    float,
                )
            )

            vect = q_aftr_from_cv * vect
            quat = q_aftr_from_cv * quat

            us_vs_s_proj = np.zeros((xyz_proj.shape[0], 2))
            us_vs_s_proj[:, 0] = (
                self.owner.calibration.fx * xyz_proj[:, 0] / xyz_proj[:, 2] + self.owner.calibration.cx
            )
            us_vs_s_proj[:, 1] = (
                self.owner.calibration.fy * xyz_proj[:, 1] / xyz_proj[:, 2] + self.owner.calibration.cy
            )

            self.owner.plotOnImg(
                markup_frame,
                us_vs_s_proj.astype(int),
                list(self.owner.ThreeDTruthPoints.getTruthPointsDict().keys()),
                (255, 255, 255),
            )
            self.owner.qnpResult = (quat, vect)

            orient_text = "Orientation (quat) From Truth Points: " + format(quat, "ijk.6f")
            pos_text = "Location From Truth Frame: " + np.array2string(vect)
            (_txt_w, txt_h), _ = cv2.getTextSize(
                orient_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                small_text(markup_frame.shape[0]),
                4,
            )
            cv2.putText(
                markup_frame,
                orient_text,
                (50, 3 * txt_h + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                small_text(markup_frame.shape[0]),
                clr.BLACK,
                4,
            )
            cv2.putText(
                markup_frame,
                orient_text,
                (50, 3 * txt_h + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                small_text(markup_frame.shape[0]),
                clr.LIGHTBLUE,
                2,
            )
            cv2.putText(
                markup_frame,
                pos_text,
                (50, 4 * txt_h + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                small_text(markup_frame.shape[0]),
                clr.BLACK,
                4,
            )
            cv2.putText(
                markup_frame,
                pos_text,
                (50, 4 * txt_h + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                small_text(markup_frame.shape[0]),
                clr.LIGHTBLUE,
                2,
            )

    def check_above_horizon(self, pt) -> bool:
        if self.owner.horizon_line is None:
            return True

        x1, y1, x2, y2 = self.owner.horizon_line
        return np.cross(np.array([x2 - x1, y2 - y1]), np.array([pt[0] - x1, pt[1] - y1])) < 0

    def hyper_focus(self, markup_frame: NDArray, ctx: GuiQueue.FrameCtx) -> None:
        plan = build_hyper_focus_plan(
            ctx=ctx,
            radius=float(self.owner.radius),
            min_radius=float(self.owner.min_radius),
            frame_shape=markup_frame.shape,
        )

        if plan is None:
            return

        if plan.next_radius is not None:
            self.owner.radius = float(plan.next_radius)

        if plan.next_min_radius is not None:
            self.owner.min_radius = float(plan.next_min_radius)

        if self.owner.yoloSession is not None and plan.desired_yolo_conf is not None:
            self.owner.yoloSession.conf = float(plan.desired_yolo_conf)

        self.apply_hyper_focus_plan(markup_frame, plan)

    @staticmethod
    def apply_hyper_focus_plan(markup_frame: NDArray, plan) -> None:
        if plan is None or plan.center is None:
            return

        center = (int(plan.center[0]), int(plan.center[1]))
        for p in plan.passes:
            dim_except_circle(
                markup_frame,
                center,
                x_axes=float(p.x_axes),
                y_axes=float(p.y_axes),
                dim_factor=float(p.dim_factor),
            )

    def _resolve_yolo_folder(self, opts: GuiQueue.YoloOpts) -> str:
        queue_path = (opts.model_folder or "").strip()
        if queue_path:
            return queue_path
        return str(getattr(self.owner.camConfig, "yoloFilepath", "") or "").strip()

    @staticmethod
    def _normalize_dir(path: str) -> str:
        return str(Path(path).expanduser().resolve())

    @staticmethod
    def _validate_yolo_folder(yolo_folder: str) -> None:
        folder = Path(yolo_folder)
        if not folder.is_dir():
            raise ValueError(f"YOLO folder does not exist: {yolo_folder}")
        if not any(folder.glob("*.onnx")) or not any(folder.glob("*.csv")):
            raise ValueError(
                "YOLO folder must contain at least one .onnx model and one .csv metadata file: "
                f"{yolo_folder}"
            )

    def _ensure_yolo_session(self, yolo_folder: str, *, force_reload: bool = False) -> None:
        from support.vision import yolo

        if not yolo_folder:
            raise ValueError("No YOLO folder selected. Set one in the YOLO queue step or the filepath page.")

        requested = self._normalize_dir(yolo_folder.strip())
        try:
            self._validate_yolo_folder(requested)
        except ValueError as e:
            from support.io.my_logging import LOG
            LOG.warning(f"Invalid YOLO folder: {yolo_folder} resulting in error:\n{e}")
            return

        switched_model = requested != self._active_yolo_dir
        if force_reload:
            self._yolo_sessions_by_dir.pop(requested, None)

        session = self._yolo_sessions_by_dir.get(requested)
        if session is None:
            session = yolo.YOLO()
            session.setNewFolder(requested)
            session.set_calibration(self.owner.calibration)
            self._yolo_sessions_by_dir[requested] = session
        elif switched_model or force_reload:
            session.setNewFolder(requested)
            session.set_calibration(self.owner.calibration)

        session.iou = self.owner.camConfig.yolo_iou
        session.conf = self.owner.camConfig.yolo_conf
        session.set_calibration(self.owner.calibration)

        if switched_model or force_reload:
            self._reset_yolo_runtime_state()
            self._active_yolo_dir = requested

        self.owner.yoloSession = session

    def run_yolo(self, frame: NDArray, markup_frame: NDArray, ctx: GuiQueue.FrameCtx, args) -> None:
        opts = self.owner.parse_args(args, GuiQueue.YoloOpts())
        yolo_folder = self._resolve_yolo_folder(opts)
        self._ensure_yolo_session(yolo_folder)

        import support.viz.draw_pnp_qnp as pnp_drw

        if self.owner.pnpDrawer is None:
            self.owner.pnpDrawer = pnp_drw.pnp_qnp_draw()

        infer_frame = frame
        if opts.inference_source == GuiQueue.YoloInferenceSource.MARKUP:
            infer_frame = markup_frame

        output = self.owner.yoloSession.inferOnImage(infer_frame, False)

        algos = pnp_drw.twoToThreeSelectedAlgorithms()
        algos.use_pnp = opts.want_pnp
        algos.use_qnp = opts.want_qnp
        algos.use_wqnp_yolo = opts.want_wqnp_yolo
        algos.use_wqnp_kfest = opts.want_wqnp_kfest
        algos.display_feature_ids = self._parse_display_feature_ids(opts.display_feature_ids)
        want_weighted_kf = bool(opts.want_wqnp_yolo or opts.want_wqnp_kfest)

        prepared = self.owner.pnpDrawer.prepare_pose_inputs(
            image=markup_frame,
            output=output,
            markup_is_undistorted=ctx.undistorted.get_or(False),
            calibration=self.owner.calibration,
            conf=self.owner.camConfig.yolo_conf,
            iou=self.owner.camConfig.yolo_iou,
            yoloSize=self.owner.yoloSession.yoloSize,
            idsNamesLocs=self.owner.yoloSession.reader.idsNamesLocs,
            originalSize=(int(infer_frame.shape[0]), int(infer_frame.shape[1])),
        )
        sigma_2N_px = None
        feature_gate_covariances_px = None
        feature_gate_mahal_sq = None
        feature_kf_used = None
        kfest_position_covariances_px = None
        kfest_gate_covariances_px = None
        kfest_gate_mahal_sq = None
        kfest_kf_used = None
        if want_weighted_kf:
            self._set_feature_kf_sigma_proc(
                opts.sigma_proc,
                float(infer_frame.shape[1]),
                float(infer_frame.shape[0]),
            )
            frame_time_s = self._frame_time_s(ctx)
            sigma_2N_px, feature_gate_covariances_px, feature_gate_mahal_sq, feature_kf_used, kfest_object_points, kfest_image_points, kfest_class_ids, sigma_2N_kfest_px, kfest_position_covariances_px, kfest_gate_covariances_px, kfest_gate_mahal_sq, kfest_kf_used = self._build_feature_kf_metadata(
                prepared,
                frame_time_s,
                float(infer_frame.shape[1]),
                float(infer_frame.shape[0]),
                self.owner.yoloSession.reader.idsNamesLocs,
            )
            if prepared is not None:
                prepared.kfest_object_points = kfest_object_points
                prepared.kfest_image_points = kfest_image_points
                prepared.kfest_class_ids = kfest_class_ids
        else:
            self._disable_feature_kf_bank()
            sigma_2N_kfest_px = None

        pose_output = self.owner.pnpDrawer.markUpImage(
            image=markup_frame,
            output=output,
            markup_is_undistorted=ctx.undistorted.get_or(False),
            calibration=self.owner.calibration,
            conf=self.owner.camConfig.yolo_conf,
            iou=self.owner.camConfig.yolo_iou,
            yoloSize=self.owner.yoloSession.yoloSize,
            idsNamesLocs=self.owner.yoloSession.reader.idsNamesLocs,
            usedAlgos=algos,
            originalSize=(int(infer_frame.shape[0]), int(infer_frame.shape[1])),
            circles_not_features=opts.feature_circles,
            prepared=prepared,
            sigma_2N_px=sigma_2N_px,
            sigma_2N_kfest_px=sigma_2N_kfest_px,
            feature_gate_covariances_px=feature_gate_covariances_px,
            feature_gate_mahal_sq=feature_gate_mahal_sq,
            feature_kf_used=feature_kf_used,
            kfest_position_covariances_px=kfest_position_covariances_px,
            kfest_gate_covariances_px=kfest_gate_covariances_px,
            kfest_gate_mahal_sq=kfest_gate_mahal_sq,
            kfest_kf_used=kfest_kf_used,
        )

        if pose_output is not None:
            self.owner.pnpResult = {
                "rvec": pose_output.pnp_rvec,
                "tvec": pose_output.pnp_tvec,
                "object_points": pose_output.object_points,
                "image_points": pose_output.image_points,
                "class_ids": pose_output.class_ids,
            } if pose_output.pnp_rvec is not None and pose_output.pnp_tvec is not None else None

            self.owner.qnpResult = {
                "q": (
                    pose_output.wqnp_kfest_q
                    if pose_output.wqnp_kfest_q is not None
                    else pose_output.wqnp_yolo_q
                    if pose_output.wqnp_yolo_q is not None
                    else pose_output.qnp_q
                ),
                "tvec": (
                    pose_output.wqnp_kfest_tvec
                    if pose_output.wqnp_kfest_tvec is not None
                    else pose_output.wqnp_yolo_tvec
                    if pose_output.wqnp_yolo_tvec is not None
                    else pose_output.qnp_tvec
                ),
                "object_points": pose_output.object_points,
                "image_points": pose_output.image_points,
                "class_ids": pose_output.class_ids,
                "weighted_mode": (
                    "kfest"
                    if pose_output.wqnp_kfest_q is not None and pose_output.wqnp_kfest_tvec is not None
                    else "yolo"
                    if pose_output.wqnp_yolo_q is not None and pose_output.wqnp_yolo_tvec is not None
                    else None
                ),
                "feature_gate_covariances_px": pose_output.feature_gate_covariances_px,
                "feature_gate_mahal_sq": pose_output.feature_gate_mahal_sq,
                "feature_kf_used": pose_output.feature_kf_used,
            } if (
                (pose_output.wqnp_kfest_q is not None and pose_output.wqnp_kfest_tvec is not None)
                or (pose_output.wqnp_yolo_q is not None and pose_output.wqnp_yolo_tvec is not None)
                or (pose_output.qnp_q is not None and pose_output.qnp_tvec is not None)
            ) else None
        else:
            self.owner.pnpResult = None
            self.owner.qnpResult = None

        centers, boxes, scores, class_ids, img_time = output
        last_yolo_center = None
        last_bounding_box_size = None
        last_yolo_3d_estimate = None
        if len(centers) > 0 and self.owner.yoloSession.reader.numClasses == 1:
            best_idx = scores.index(max(scores))
            infer_h, infer_w = infer_frame.shape[:2]
            draw_h, draw_w = markup_frame.shape[:2]
            img_yolo_x_correction = infer_w / self.owner.yoloSession.reader.imageSize
            img_yolo_y_correction = infer_h / self.owner.yoloSession.reader.imageSize
            draw_sx = draw_w / float(infer_w) if infer_w > 0 else 1.0
            draw_sy = draw_h / float(infer_h) if infer_h > 0 else 1.0

            bbox_size_infer = (
                (boxes[best_idx][2] - boxes[best_idx][0]) * img_yolo_x_correction,
                (boxes[best_idx][3] - boxes[best_idx][1]) * img_yolo_y_correction,
            )
            center_infer = (
                int(centers[best_idx][0] * img_yolo_x_correction),
                int(centers[best_idx][1] * img_yolo_y_correction),
            )
            last_bounding_box_size = (
                bbox_size_infer[0] * draw_sx,
                bbox_size_infer[1] * draw_sy,
            )
            last_yolo_center = (
                int(round(center_infer[0] * draw_sx)),
                int(round(center_infer[1] * draw_sy)),
            )

            if self.check_above_horizon(last_yolo_center):
                estimate = estimate_single_feature_from_center_width(
                    center_px=(float(center_infer[0]), float(center_infer[1])),
                    bbox_w_px=float(bbox_size_infer[0]),
                    bbox_h_px=float(bbox_size_infer[1]),
                    K=camera_matrix_from_calibration(
                        self.owner.calibration,
                        image_size_px=(float(infer_w), float(infer_h)),
                        scale_to_image=True,
                    ),
                )
                if estimate is not None:
                    last_yolo_3d_estimate = np.asarray(estimate.xyz_cam_m, dtype=float)
                    bb_color = (50, 255, 255)
                    # self._draw_bottom_left_text(
                    #     markup_frame,
                    #     "BB-Width Solution",
                    #     row_idx=3,
                    #     color=bb_color,
                    # )
                    self._draw_bottom_left_text(
                        markup_frame,
                        (
                            f"BBS: {last_yolo_3d_estimate[0]:+6.3f}, "
                            f"{last_yolo_3d_estimate[1]:+6.3f}, "
                            f"{last_yolo_3d_estimate[2]:+6.3f} "
                            f"({np.linalg.norm(last_yolo_3d_estimate):6.3f})"
                        ),
                        row_idx=2,
                        color=bb_color,
                    )

        ctx.yolo.set(
            self.owner.yolo_output_type(
                last_bounding_box_size=last_bounding_box_size,
                last_yolo_center=last_yolo_center,
                last_yolo_3d_estimate=last_yolo_3d_estimate,
                pose=pose_output,
            )
        )

        if opts.factor_graph:
            self.factor_graph(frame, markup_frame, ctx, opts.hyper_focus)

    @staticmethod
    def draw_factor_graph_overlay(markup_frame: NDArray, fg_output, color) -> None:
        if fg_output is None or fg_output.curr_FG_pixel is None:
            return

        pixel = (int(fg_output.curr_FG_pixel[0]), int(fg_output.curr_FG_pixel[1]))

        h, w, _ = markup_frame.shape
        size = int(0.025 * h)

        text = (
            f"FG:   {fg_output.curr_r_T_d[0]:+6.3f}, {fg_output.curr_r_T_d[1]:+6.3f}, "
            f"{fg_output.curr_r_T_d[2]:+6.3f} ({np.linalg.norm(fg_output.curr_r_T_d):6.3f})"
        )
        (_txt_width, txt_height), _base = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            med_text(w),
            med_thick(h),
        )
        pad = int(0.3 * txt_height)
        txt_height_per_row = txt_height + pad
        loc = (pad, h - 3 * txt_height_per_row - pad)

        thickness = lrg_thick(h)
        cv2.circle(markup_frame, pixel, size, (0, 0, 0), thickness)
        cv2.line(markup_frame, [pixel[0] + size, pixel[1]], [pixel[0] - size, pixel[1]], (0, 0, 0), thickness)
        cv2.line(markup_frame, [pixel[0], pixel[1] + size], [pixel[0], pixel[1] - size], (0, 0, 0), thickness)

        thickness = med_thick(h)
        cv2.circle(markup_frame, pixel, size, color, thickness)
        cv2.line(markup_frame, [pixel[0] + size, pixel[1]], [pixel[0] - size, pixel[1]], color, thickness)
        cv2.line(markup_frame, [pixel[0], pixel[1] + size], [pixel[0], pixel[1] - size], color, thickness)

        cv2.putText(
            markup_frame,
            text,
            loc,
            cv2.FONT_HERSHEY_SIMPLEX,
            med_text(markup_frame.shape[0]),
            (0, 0, 0),
            lrg_thick(h),
        )
        cv2.putText(
            markup_frame,
            text,
            loc,
            cv2.FONT_HERSHEY_SIMPLEX,
            med_text(markup_frame.shape[0]),
            color,
            med_thick(h),
        )

    def factor_graph(
        self,
        frame: NDArray,
        markup_frame: NDArray,
        ctx: GuiQueue.FrameCtx,
        hyper_focus: bool,
    ) -> None:
        yolo = ctx.yolo.get_or()
        color = clr.YELLOWGREEN if yolo is not None else clr.RED

        R_wr = (
            self.owner.own_attitude.rotmat_wr()
            if self.owner.own_attitude is not None and self.owner.own_attitude.valid
            else None
        )

        self.owner.FG, self.owner.last_time_update, has_measurement, pred = run_factor_graph_step(
            fg=self.owner.FG,
            yolo=yolo,
            img_time=ctx.img_time,
            last_time_update=self.owner.last_time_update,
            R_wr=R_wr,
        )

        if not has_measurement:
            color = clr.RED

        if pred is not None and pred.r_T_d is not None:
            K = factor_graph_projection_matrix(
                calibration=self.owner.calibration,
                markup_frame=markup_frame,
                yolo=yolo,
            )

            fg_output = build_factor_graph_output(pred, K)
            if fg_output is not None:
                self.owner.last_fg_output = fg_output
                ctx.fg.set(fg_output)
                self.draw_factor_graph_overlay(markup_frame, fg_output, color)
        elif (
            self.owner.camConfig.imageSource == ImageSource.Stream_from_Folder
            and self.owner.playback_controller.pause
            and self.owner.last_fg_output is not None
        ):
            ctx.fg.set(self.owner.last_fg_output)
            self.draw_factor_graph_overlay(markup_frame, self.owner.last_fg_output, color)

        if hyper_focus:
            self.hyper_focus(markup_frame, ctx)

    @staticmethod
    def cv_pose_to_ours(R_cv: np.ndarray, t_cv: np.ndarray):
        S_MODEL = np.diag([1.0, -1.0, 1.0])
        C_OURS_TO_CV = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=float)
        C_CV_TO_OURS = C_OURS_TO_CV.T
        R_ours = C_CV_TO_OURS @ R_cv @ S_MODEL
        t_ours = C_CV_TO_OURS @ t_cv
        return mat2quat(R_ours.T), t_ours
