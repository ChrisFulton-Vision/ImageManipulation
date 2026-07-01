import numpy as np
from numpy.typing import NDArray
import cv2
from dataclasses import dataclass
from support.vision.calibration import Calibration
import support.viz.colors as clr
from support.viz.CVFontScaling import med_text, small_thick, med_thick, lrg_thick
from support.mathHelpers.twoD_to_threeD import solveQnP
from support.mathHelpers.quaternions import Quaternion as q


@dataclass(slots=True)
class PoseOutput:
    """Container for pose solutions generated from feature correspondences."""
    pnp_rvec: NDArray | None = None
    pnp_tvec: NDArray | None = None
    pnp_inlier_class_ids: list[int] | None = None
    pnp_outlier_class_ids: list[int] | None = None

    qnp_q: q = None  # Quaternion object from solveQnP
    qnp_tvec: NDArray | None = None
    wqnp_yolo_q: q = None
    wqnp_yolo_tvec: NDArray | None = None
    wqnp_kfest_q: q = None
    wqnp_kfest_tvec: NDArray | None = None

    object_points: NDArray | None = None
    image_points: NDArray | None = None
    class_ids: list[int] | None = None
    feature_gate_covariances_px: NDArray | None = None
    feature_gate_mahal_sq: NDArray | None = None
    feature_kf_used: NDArray | None = None
    kf_rejected_measurement_class_ids: list[int] | None = None
    kf_track_class_ids: list[int] | None = None
    kf_track_estimates_px: NDArray | None = None
    kf_track_position_covariances_px: NDArray | None = None


@dataclass(slots=True)
class PreparedPoseInputs:
    centers_for_draw: NDArray
    centers_for_pnp: NDArray
    boxes_for_draw: list
    class_ids: list[int]
    pose_class_ids: list[int]
    scores: list[float]
    object_points: NDArray
    image_points: NDArray
    kfest_object_points: NDArray | None = None
    kfest_image_points: NDArray | None = None
    kfest_class_ids: list[int] | None = None


class twoToThreeSelectedAlgorithms:
    def __init__(self):
        self.use_pnp = False
        self.use_qnp = False
        self.use_wqnp_yolo = False
        self.use_wqnp_kfest = False
        self.display_feature_ids: set[int] | None = None


class pnp_qnp_draw:
    def __init__(self):
        self.last_q_vec = None
        self.last_t_vec = None
        self.last_wq_vec = None
        self.last_wt_vec = None
        self.last_pnp_rvec = None
        self.last_pnp_tvec = None

        # -----------------------------
        # NEW: estimation-only helpers
        # -----------------------------

    @staticmethod
    def _scale_covariances_for_draw(
        covariances_px: NDArray | None,
        scale_x: float,
        scale_y: float,
    ) -> NDArray | None:
        if covariances_px is None:
            return None
        scale = np.diag([float(scale_x), float(scale_y)]).astype(np.float64)
        scaled_covariances: list[NDArray] = []
        for cov in np.asarray(covariances_px, dtype=np.float64):
            if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
                scaled_covariances.append(np.asarray(cov, dtype=np.float64))
                continue
            scaled_covariances.append(scale @ cov @ scale.T)
        return np.asarray(scaled_covariances, dtype=np.float64)

    @staticmethod
    def _scale_points_for_draw(
        points_px: NDArray | None,
        scale_x: float,
        scale_y: float,
    ) -> NDArray | None:
        if points_px is None:
            return None
        scaled_points = np.asarray(points_px, dtype=np.float64).copy()
        if scaled_points.ndim != 2 or scaled_points.shape[1] < 2:
            return scaled_points
        scaled_points[:, 0] *= float(scale_x)
        scaled_points[:, 1] *= float(scale_y)
        return scaled_points

    @staticmethod
    def _empty_prepared_inputs() -> PreparedPoseInputs:
        return PreparedPoseInputs(
            centers_for_draw=np.empty((0, 2), dtype=np.float64),
            centers_for_pnp=np.empty((0, 2), dtype=np.float64),
            boxes_for_draw=[],
            class_ids=[],
            pose_class_ids=[],
            scores=[],
            object_points=np.empty((0, 3), dtype=np.float64),
            image_points=np.empty((0, 2), dtype=np.float64),
            kfest_object_points=None,
            kfest_image_points=None,
            kfest_class_ids=[],
        )

    def _estimate_pnp(self,
                      object_points: NDArray,
                      image_points: NDArray,
                      calibration: Calibration):
        """
        Returns (rvec, tvec, inlier_indices) from solvePnPRansac, or None if it fails.
        """
        if calibration is None:
            return None
        if object_points is None or image_points is None:
            return None
        if len(object_points) < 6:
            return None

        camera_matrix = calibration.getCameraMatrix()
        dist_coeffs = np.zeros((5,))

        # if self.last_pnp_rvec is not None and self.last_pnp_tvec is not None:
        #
        #     ret, rvec, tvec = cv2.solvePnP(
        #         objectPoints=object_points,
        #         imagePoints=image_points,
        #         cameraMatrix=camera_matrix,
        #         distCoeffs=dist_coeffs,
        #         rvec=self.last_pnp_rvec,
        #         tvec=self.last_pnp_tvec,
        #         useExtrinsicGuess=True,
        #         flags=cv2.SOLVEPNP_ITERATIVE,
        #     )
        #     if ret:
        #         self.last_pnp_rvec = rvec
        #         self.last_pnp_tvec = tvec
        #         return rvec, tvec

        # fast_flag = cv2.SOLVEPNP_SQPNP if hasattr(cv2, "SOLVEPNP_SQPNP") else cv2.SOLVEPNP_EPNP
        # ret, rvec, tvec = cv2.solvePnP(
        #     objectPoints=object_points,
        #     imagePoints=image_points,
        #     cameraMatrix=camera_matrix,
        #     distCoeffs=dist_coeffs,
        #     flags=fast_flag,
        # )
        # if ret:
        #     self.last_pnp_rvec = rvec
        #     self.last_pnp_tvec = tvec
        #     return rvec, tvec

        rvec = self.last_pnp_rvec
        tvec = self.last_pnp_tvec

        ret, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=object_points,
            imagePoints=image_points,
            cameraMatrix=camera_matrix,
            distCoeffs=dist_coeffs,
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if ret:
            self.last_pnp_rvec = rvec
            self.last_pnp_tvec = tvec
            return rvec, tvec, inliers
        return None

    def _estimate_qnp(self,
                      object_points: NDArray,
                      image_points: NDArray,
                      calibration: Calibration,
                      seed_rvec=None,
                      seed_tvec=None,
                      sigma_2N=None,
                      weighted: bool = False):
        """
        Returns (q_rvec, q_tvec) from solveQnP, or None if it fails.

        seed_rvec/seed_tvec are intended to come from solvePnP.
        If your solveQnP expects quaternion seed instead of Rodrigues,
        convert seed_rvec -> quat before passing.
        """
        if calibration is None:
            return None
        if object_points is None or image_points is None:
            return None
        if len(object_points) < 6:
            return None

        # Prefer explicit seed (PnP), else fall back to last good QnP
        user_seed_q = None
        user_seed_t = None

        if seed_rvec is not None and seed_tvec is not None:
            # If solveQnP can take Rodrigues directly as its q seed, pass it through.
            # Otherwise, convert Rodrigues -> quat here and pass that.
            user_seed_q = q().from_rodrigues(seed_rvec)
            user_seed_t = np.squeeze(seed_tvec)
        else:
            prev_q = self.last_wq_vec if weighted else self.last_q_vec
            prev_t = self.last_wt_vec if weighted else self.last_t_vec
            if prev_q is not None and prev_t is not None:
                user_seed_q = prev_q
                user_seed_t = prev_t

        q_rvec, q_tvec = solveQnP(
            object_pts=object_points,
            img_pts=image_points,
            cal=calibration,
            sigma_2N=sigma_2N,
            user_seed_q=user_seed_q,
            user_seed_t=user_seed_t,
            # Turn this off because we are explicitly controlling the seed now.
            use_solvePnP_as_seed=False
        )

        # Persist last good solution
        if weighted:
            self.last_wq_vec, self.last_wt_vec = q_rvec, q_tvec
        else:
            self.last_q_vec, self.last_t_vec = q_rvec, q_tvec
        return q_rvec, q_tvec

    def prepare_pose_inputs(self,
                            image: NDArray,
                            output: tuple[list, list, list, list, float],
                            markup_is_undistorted: bool,
                            calibration: Calibration,
                            conf: float,
                            iou: float,
                            yoloSize: tuple[int, int],
                            idsNamesLocs,
                            originalSize: tuple[int, int]) -> PreparedPoseInputs | None:
        h, w, _ = image.shape
        h_ori, w_ori = originalSize

        centers_dist, boxes, scores, class_ids, _time = output
        if len(centers_dist) < 1:
            return self._empty_prepared_inputs()
        y_h, y_w = yoloSize
        sx = w_ori / float(y_w)
        sy = h_ori / float(y_h)
        draw_sx = w / float(w_ori) if w_ori > 0 else 1.0
        draw_sy = h / float(h_ori) if h_ori > 0 else 1.0

        centers_px = np.asarray(centers_dist, dtype=np.float64)
        centers_px[:, 0] *= sx
        centers_px[:, 1] *= sy

        boxes_for_draw = None
        if boxes is not None and len(boxes) > 0:
            b = np.asarray(boxes, dtype=np.float64)
            b[:, 0] *= sx
            b[:, 2] *= sx
            b[:, 1] *= sy
            b[:, 3] *= sy
            boxes_for_draw = b.tolist()

        centers_for_draw = centers_px.copy()
        centers_for_pnp = centers_px.copy()

        if boxes_for_draw is None:
            boxes_for_draw = []
        else:
            boxes_for_draw = [list(b) for b in boxes_for_draw]

        if markup_is_undistorted:
            pts = np.asarray(centers_for_draw, dtype=np.float32).reshape(-1, 1, 2)
            centers_for_draw = cv2.undistortPoints(
                pts,
                calibration.getCameraMatrix(),
                calibration.getDistortion(),
                P=calibration.remapK,
            ).reshape(-1, 2)

            undist_boxes = []
            for (x1, y1, x2, y2) in boxes_for_draw:
                corners = np.array([
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2],
                ], dtype=np.float32).reshape(-1, 1, 2)

                und_corners = cv2.undistortPoints(
                    corners,
                    calibration.getCameraMatrix(),
                    calibration.getDistortion(),
                    P=calibration.remapK,
                ).reshape(-1, 2)

                xmin = float(np.min(und_corners[:, 0]))
                xmax = float(np.max(und_corners[:, 0]))
                ymin = float(np.min(und_corners[:, 1]))
                ymax = float(np.max(und_corners[:, 1]))

                undist_boxes.append([xmin, ymin, xmax, ymax])

            boxes_for_draw = undist_boxes

        centers_for_draw[:, 0] *= draw_sx
        centers_for_draw[:, 1] *= draw_sy
        if len(boxes_for_draw) > 0:
            boxes_arr = np.asarray(boxes_for_draw, dtype=np.float64)
            boxes_arr[:, 0] *= draw_sx
            boxes_arr[:, 2] *= draw_sx
            boxes_arr[:, 1] *= draw_sy
            boxes_arr[:, 3] *= draw_sy
            boxes_for_draw = boxes_arr.tolist()

        if len(class_ids) == 0:
            return self._empty_prepared_inputs()

        boxes_for_nms = []
        for box in boxes_for_draw:
            x1, y1, x2, y2 = box
            boxes_for_nms.append([
                float(x1),
                float(y1),
                float(x2 - x1),
                float(y2 - y1),
            ])

        indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores, conf, iou)
        if len(indices) == 0:
            return self._empty_prepared_inputs()

        keep = np.array(
            [int(i[0]) if hasattr(i, "__len__") else int(i) for i in indices],
            dtype=np.int32
        )

        new_centers_for_draw = centers_for_draw[keep]
        new_centers_for_pnp = centers_for_pnp[keep]
        new_boxes = boxes_for_draw[keep] if isinstance(boxes_for_draw, np.ndarray) else [boxes_for_draw[i] for i in keep]
        new_class_ids = [class_ids[i] for i in keep]
        new_scores = [scores[i] for i in keep]

        object_points, image_points, pose_class_ids = self._collect_objPts_and_imgPts(
            new_class_ids,
            new_centers_for_pnp,
            idsNamesLocs,
        )

        return PreparedPoseInputs(
            centers_for_draw=new_centers_for_draw,
            centers_for_pnp=new_centers_for_pnp,
            boxes_for_draw=new_boxes,
            class_ids=list(new_class_ids),
            pose_class_ids=list(pose_class_ids),
            scores=list(new_scores),
            object_points=object_points,
            image_points=image_points,
            kfest_object_points=None,
            kfest_image_points=None,
            kfest_class_ids=[],
        )

    def markUpImage(self,
                    image: NDArray,
                    output: tuple[list, list, list, list, float],
                    markup_is_undistorted: bool,
                    calibration: Calibration,
                    conf: float,
                    iou: float,
                    yoloSize: tuple[int, int],
                    idsNamesLocs,
                    usedAlgos: twoToThreeSelectedAlgorithms,
                    originalSize: tuple[int, int],
                    circles_not_features: bool = False,
                    prepared: PreparedPoseInputs | None = None,
                    sigma_2N_px: NDArray | None = None,
                    sigma_2N_kfest_px: NDArray | None = None,
                    feature_gate_covariances_px: NDArray | None = None,
                    feature_gate_mahal_sq: NDArray | None = None,
                    feature_kf_used: NDArray | None = None,
                    kfest_gate_covariances_px: NDArray | None = None,
                    kfest_gate_mahal_sq: NDArray | None = None,
                    kfest_kf_used: NDArray | None = None,
                    kfest_position_covariances_px: NDArray | None = None) -> PoseOutput | None:

        h, w, _ = image.shape
        h_ori, w_ori = originalSize

        centers_dist, _boxes, _scores, _class_ids, time = output

        text = f'Inference time: {time:.3f}s'
        (txt_width, txt_height), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, med_text(w), 4)
        pad = int(0.3 * txt_height)
        cv2.putText(image, text,
                    (pad, pad + int(txt_height)),
                    cv2.FONT_HERSHEY_SIMPLEX, med_text(w), clr.BLACK, lrg_thick(h))
        cv2.putText(image, text,
                    (pad, pad + int(txt_height)),
                    cv2.FONT_HERSHEY_SIMPLEX, med_text(w), clr.LIGHTBLUE, med_thick(h))

        if calibration is not None:
            calibration.scaleCalibration(w_ori)  # K now matches 'image' pixel space, returns early if already correct

        if prepared is None:
            prepared = self.prepare_pose_inputs(
                image=image,
                output=output,
                markup_is_undistorted=markup_is_undistorted,
                calibration=calibration,
                conf=conf,
                iou=iou,
                yoloSize=yoloSize,
                idsNamesLocs=idsNamesLocs,
                originalSize=originalSize,
            )
            if prepared is None:
                return None

        newCentersForDraw = prepared.centers_for_draw
        newCentersForPnp = prepared.centers_for_pnp
        newBoxes = prepared.boxes_for_draw
        newClass_ids = prepared.class_ids
        newScores = prepared.scores
        draw_scale_x = (float(w) / float(w_ori)) if w_ori > 0 else 1.0
        draw_scale_y = (float(h) / float(h_ori)) if h_ori > 0 else 1.0
        feature_gate_covariances_draw_px = self._scale_covariances_for_draw(
            feature_gate_covariances_px,
            draw_scale_x,
            draw_scale_y,
        )
        kfest_image_points_draw = self._scale_points_for_draw(
            prepared.kfest_image_points,
            draw_scale_x,
            draw_scale_y,
        )
        kfest_gate_covariances_draw_px = self._scale_covariances_for_draw(
            kfest_gate_covariances_px,
            draw_scale_x,
            draw_scale_y,
        )

        self._drawBoxes(
            image,
            newCentersForDraw,
            newBoxes,
            newClass_ids,
            newScores,
            draw_as_circles=circles_not_features,
            feature_gate_covariances_px=feature_gate_covariances_draw_px,
            feature_gate_mahal_sq=feature_gate_mahal_sq,
            feature_kf_used=feature_kf_used,
            display_feature_ids=usedAlgos.display_feature_ids,
        )
        if (
            usedAlgos.use_wqnp_kfest
            and kfest_image_points_draw is not None
            and prepared.kfest_class_ids is not None
            and len(prepared.kfest_class_ids) > 0
        ):
            self._draw_kfest_estimates(
                image=image,
                kfest_image_points=kfest_image_points_draw,
                kfest_class_ids=prepared.kfest_class_ids,
                measured_class_ids=newClass_ids,
                draw_as_circles=circles_not_features,
                kfest_gate_covariances_px=kfest_gate_covariances_draw_px,
                kfest_gate_mahal_sq=kfest_gate_mahal_sq,
                kfest_kf_used=kfest_kf_used,
                display_feature_ids=usedAlgos.display_feature_ids,
            )
        object_points = prepared.object_points
        image_points = prepared.image_points

        have_pose_correspondences = len(object_points) >= 6
        pnp_inlier_class_ids: list[int] = []
        pnp_outlier_class_ids: list[int] = []
        kf_rejected_measurement_class_ids: list[int] = []
        if feature_kf_used is not None:
            for cid, used in zip(newClass_ids, feature_kf_used):
                if not bool(used):
                    kf_rejected_measurement_class_ids.append(int(cid))

        # 1) Estimate PnP (optional)
        pnp_pose = None
        if usedAlgos.use_pnp and have_pose_correspondences:
            pnp_pose = self._estimate_pnp(object_points,
                                          image_points,
                                          calibration)
            if pnp_pose is not None:
                _rvec_tmp, _tvec_tmp, inliers = pnp_pose
                inlier_idx_set: set[int] = set()
                if inliers is not None:
                    inlier_idx_set = {int(v) for v in np.asarray(inliers).reshape(-1)}
                pnp_inlier_class_ids = [
                    int(prepared.pose_class_ids[idx])
                    for idx in sorted(inlier_idx_set)
                    if 0 <= int(idx) < len(prepared.pose_class_ids)
                ]
                pnp_outlier_class_ids = [
                    int(cid)
                    for idx, cid in enumerate(prepared.pose_class_ids)
                    if idx not in inlier_idx_set
                ]

        # 2) Estimate QnP (optional) seeded by PnP if available
        qnp_pose = None
        if usedAlgos.use_qnp and have_pose_correspondences:
            if pnp_pose is not None:
                seed_rvec, seed_tvec = pnp_pose[0], pnp_pose[1]
            else:
                seed_rvec, seed_tvec = None, None
            qnp_pose = self._estimate_qnp(object_points,
                                          image_points,
                                          calibration,
                                          seed_rvec,
                                          seed_tvec)

        wqnp_yolo_pose = None
        if usedAlgos.use_wqnp_yolo and sigma_2N_px is not None and have_pose_correspondences:
            if pnp_pose is not None:
                seed_rvec, seed_tvec = pnp_pose[0], pnp_pose[1]
            else:
                seed_rvec, seed_tvec = None, None
            wqnp_yolo_pose = self._estimate_qnp(
                object_points,
                image_points,
                calibration,
                seed_rvec,
                seed_tvec,
                sigma_2N=sigma_2N_px,
                weighted=True,
            )

        wqnp_kfest_pose = None
        if (
            usedAlgos.use_wqnp_kfest
            and sigma_2N_kfest_px is not None
            and prepared.kfest_object_points is not None
            and prepared.kfest_image_points is not None
            and len(prepared.kfest_object_points) >= 6
            and np.all(np.isfinite(prepared.kfest_image_points))
        ):
            if pnp_pose is not None:
                seed_rvec, seed_tvec = pnp_pose[0], pnp_pose[1]
            else:
                seed_rvec, seed_tvec = None, None
            wqnp_kfest_pose = self._estimate_qnp(
                prepared.kfest_object_points,
                prepared.kfest_image_points,
                calibration,
                seed_rvec,
                seed_tvec,
                sigma_2N=sigma_2N_kfest_px,
                weighted=True,
            )

        # 3) Draw in desired order (match your idx stacking)
        idx = 0
        if usedAlgos.use_wqnp_kfest and wqnp_kfest_pose is not None:
            q_rvec, q_tvec = wqnp_kfest_pose
            self._drawQnP_from_pose(
                image=image,
                y_class_ids=prepared.kfest_class_ids or [],
                y_centers=prepared.kfest_image_points,
                object_points=prepared.kfest_object_points,
                q_rvec=q_rvec,
                q_tvec=q_tvec,
                markup_is_undistorted=markup_is_undistorted,
                calibration=calibration,
                yoloSize=yoloSize,
                idsNamesLocs=idsNamesLocs,
                originalSize=originalSize,
                idx=idx,
                txt_color=clr.YELLOWGREEN,
                title_prefix="WQNP_KF",
                draw_as_circles=circles_not_features,
                display_feature_ids=usedAlgos.display_feature_ids,
            )
            idx += 1

        if usedAlgos.use_wqnp_yolo and wqnp_yolo_pose is not None:
            q_rvec, q_tvec = wqnp_yolo_pose
            self._drawQnP_from_pose(
                image=image,
                y_class_ids=newClass_ids,
                y_centers=newCentersForPnp,
                object_points=object_points,
                q_rvec=q_rvec,
                q_tvec=q_tvec,
                markup_is_undistorted=markup_is_undistorted,
                calibration=calibration,
                yoloSize=yoloSize,
                idsNamesLocs=idsNamesLocs,
                originalSize=originalSize,
                idx=idx,
                txt_color=clr.BROWN,
                title_prefix="WQNP_YOLO",
                draw_as_circles=circles_not_features,
                display_feature_ids=usedAlgos.display_feature_ids,
            )
            idx += 1

        if usedAlgos.use_qnp and qnp_pose is not None:
            q_rvec, q_tvec = qnp_pose
            self._drawQnP_from_pose(
                image=image,
                y_class_ids=newClass_ids,
                y_centers=newCentersForPnp,
                object_points=object_points,
                q_rvec=q_rvec,
                q_tvec=q_tvec,
                markup_is_undistorted=markup_is_undistorted,
                calibration=calibration,
                yoloSize=yoloSize,
                idsNamesLocs=idsNamesLocs,
                originalSize=originalSize,
                idx=idx,
                draw_as_circles=circles_not_features,
                display_feature_ids=usedAlgos.display_feature_ids,
            )
            idx += 1

        if usedAlgos.use_pnp and pnp_pose is not None:
            rvec, tvec = pnp_pose[0], pnp_pose[1]
            self._drawPnP_from_pose(
                image=image,
                y_class_ids=newClass_ids,
                y_centers=newCentersForPnp,
                object_points=object_points,
                rvec=rvec,
                tvec=tvec,
                markup_is_undistorted=markup_is_undistorted,
                calibration=calibration,
                yoloSize=yoloSize,
                idsNamesLocs=idsNamesLocs,
                originalSize=originalSize,
                idx=idx,
                draw_as_circles=circles_not_features,
                display_feature_ids=usedAlgos.display_feature_ids,
            )
            idx += 1

        self._draw_overlay_legend(
            image=image,
            used_algos=usedAlgos,
        )

        return PoseOutput(
            pnp_rvec=pnp_pose[0] if pnp_pose is not None else None,
            pnp_tvec=pnp_pose[1] if pnp_pose is not None else None,
            pnp_inlier_class_ids=pnp_inlier_class_ids,
            pnp_outlier_class_ids=pnp_outlier_class_ids,
            qnp_q=qnp_pose[0] if qnp_pose is not None else None,
            qnp_tvec=qnp_pose[1] if qnp_pose is not None else None,
            wqnp_yolo_q=wqnp_yolo_pose[0] if wqnp_yolo_pose is not None else None,
            wqnp_yolo_tvec=wqnp_yolo_pose[1] if wqnp_yolo_pose is not None else None,
            wqnp_kfest_q=wqnp_kfest_pose[0] if wqnp_kfest_pose is not None else None,
            wqnp_kfest_tvec=wqnp_kfest_pose[1] if wqnp_kfest_pose is not None else None,
            object_points=prepared.kfest_object_points if wqnp_kfest_pose is not None else object_points,
            image_points=prepared.kfest_image_points if wqnp_kfest_pose is not None else image_points,
            class_ids=list(prepared.kfest_class_ids) if wqnp_kfest_pose is not None and prepared.kfest_class_ids is not None else list(newClass_ids),
            feature_gate_covariances_px=feature_gate_covariances_px,
            feature_gate_mahal_sq=feature_gate_mahal_sq,
            feature_kf_used=feature_kf_used,
            kf_rejected_measurement_class_ids=kf_rejected_measurement_class_ids,
            kf_track_class_ids=list(prepared.kfest_class_ids) if prepared.kfest_class_ids is not None else [],
            kf_track_estimates_px=prepared.kfest_image_points,
            kf_track_position_covariances_px=kfest_position_covariances_px,
        )

    @staticmethod
    def _drawBoxes(image: NDArray, newCenters: NDArray, newBoxes: NDArray,
                   newClass_ids: list, newScores: list,
                    draw_as_circles: bool = True,
                   circle_radius_px: int | None = None,
                   feature_gate_covariances_px: NDArray | None = None,
                   feature_gate_mahal_sq: NDArray | None = None,
                   feature_kf_used: NDArray | None = None,
                   display_feature_ids: set[int] | None = None,) -> None:
        '''
        Draws yolo boxes
        :param image: Original OpenCV image
        :param newCenters: center of bounding box
        :param newBoxes: onnxruntime box
        :param newClass_ids: onnxruntime id
        :param newScores: onnxruntime confidence
        :param color: color of box
        :return:
        '''
        h, w, _ = image.shape

        for idx, (centers, box, class_id, score) in enumerate(zip(newCenters, newBoxes, newClass_ids, newScores)):
            if display_feature_ids is not None and int(class_id) not in display_feature_ids:
                continue
            x, y = centers
            x1, y1, x2, y2 = box
            x = int(round(x))
            y = int(round(y))
            x1 = int(round(x1))
            y1 = int(round(y1))
            x2 = int(round(x2))
            y2 = int(round(y2))

            if draw_as_circles:
                r = int(circle_radius_px) if circle_radius_px is not None else max(2, int(round(0.002 * w)))
                cv2.circle(image, (x, y), r + 2, clr.BLACK, -1)
                cv2.circle(image, (x, y), r, clr.LIGHTBLUE, -1)
                if feature_gate_covariances_px is not None and idx < len(feature_gate_covariances_px):
                    cov = np.asarray(feature_gate_covariances_px[idx], dtype=np.float64)
                    if cov.shape == (2, 2) and np.all(np.isfinite(cov)):
                        eigvals, eigvecs = np.linalg.eigh(cov)
                        eigvals = np.maximum(eigvals, 0.0)
                        gate_msq = 1.0
                        if feature_gate_mahal_sq is not None and idx < len(feature_gate_mahal_sq):
                            gate_msq = max(float(feature_gate_mahal_sq[idx]), 0.0)
                        major = max(1, int(round(np.sqrt(gate_msq * eigvals[1]))))
                        minor = max(1, int(round(np.sqrt(gate_msq * eigvals[0]))))
                        angle = float(np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1])))
                        used = True if feature_kf_used is None else bool(feature_kf_used[idx])
                        color = clr.YELLOWGREEN if used else clr.ORANGE
                        cv2.ellipse(image, (x, y), (major, minor), angle, 0, 360, color, 1)
            else:
                label = f"{class_id}"
                cv2.rectangle(image, (x1, y1), (x2, y2), clr.LIGHTBLUE, small_thick(h))

                (txt_w, txt_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, med_text(w), 4)
                lowerLeftCorner = (int(x - txt_w / 2.0), int(y + txt_h / 2.0))

                cv2.putText(image,
                            label,
                            lowerLeftCorner,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            med_text(1.25 * h), clr.BLACK, lrg_thick(h))
                cv2.putText(image,
                            label,
                            lowerLeftCorner,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            med_text(1.25 * h), clr.LIGHTBLUE, med_thick(h))

        (txt_width, txt_height), base = cv2.getTextSize('I',
                                                        cv2.FONT_HERSHEY_SIMPLEX,
                                                        med_text(w), med_thick(h))
        pad = int(0.3 * txt_height)
        txt_height_perRow = txt_height + pad
        loc = (pad, h - 6 * txt_height_perRow - pad)

        cv2.putText(image, 'Direct Inference',
                    loc,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    med_text(w), clr.BLACK, lrg_thick(h))
        cv2.putText(image, 'Direct Inference',
                    loc,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    med_text(w), clr.LIGHTBLUE, med_thick(h))

    @staticmethod
    def _draw_kfest_estimates(
        image: NDArray,
        kfest_image_points: NDArray,
        kfest_class_ids: list[int],
        measured_class_ids: list[int],
        draw_as_circles: bool = True,
        kfest_gate_covariances_px: NDArray | None = None,
        kfest_gate_mahal_sq: NDArray | None = None,
        kfest_kf_used: NDArray | None = None,
        display_feature_ids: set[int] | None = None,
    ) -> None:
        h, w, _ = image.shape
        measured_ids = {int(cid) for cid in measured_class_ids}
        radius = max(3, int(round(0.0035 * w)))

        for idx, (center, class_id) in enumerate(zip(np.asarray(kfest_image_points, dtype=np.float64),
                                                     kfest_class_ids)):
            if display_feature_ids is not None and int(class_id) not in display_feature_ids:
                continue
            if len(center) < 2 or not np.all(np.isfinite(center[:2])):
                continue
            x = int(round(float(center[0])))
            y = int(round(float(center[1])))
            current_measurement = int(class_id) in measured_ids
            color = clr.GREEN if current_measurement else clr.PINK

            cv2.circle(image, (x, y), radius + 2, clr.BLACK, 1)
            cv2.circle(image, (x, y), radius, color, 1)
            if kfest_gate_covariances_px is not None and idx < len(kfest_gate_covariances_px):
                cov = np.asarray(kfest_gate_covariances_px[idx], dtype=np.float64)
                if cov.shape == (2, 2) and np.all(np.isfinite(cov)):
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    eigvals = np.maximum(eigvals, 0.0)
                    gate_msq = 1.0
                    if kfest_gate_mahal_sq is not None and idx < len(kfest_gate_mahal_sq):
                        gate_msq = max(float(kfest_gate_mahal_sq[idx]), 0.0)
                    major = max(1, int(round(np.sqrt(gate_msq * eigvals[1]))))
                    minor = max(1, int(round(np.sqrt(gate_msq * eigvals[0]))))
                    angle = float(np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1])))
                    used = True if kfest_kf_used is None else bool(kfest_kf_used[idx])
                    ellipse_color = clr.YELLOWGREEN if used else clr.ORANGE
                    cv2.ellipse(image, (x, y), (major, minor), angle, 0, 360, ellipse_color, 1)
            if not draw_as_circles:
                cv2.line(image, (x - radius, y), (x + radius, y), color, 1)
                cv2.line(image, (x, y - radius), (x, y + radius), color, 1)

        (_txt_width, txt_height), _base = cv2.getTextSize(
            'K',
            cv2.FONT_HERSHEY_SIMPLEX,
            med_text(w),
            med_thick(h),
        )
        pad = int(0.3 * txt_height)
        txt_height_per_row = txt_height + pad
        loc = (pad, h - 5 * txt_height_per_row - pad)

        cv2.putText(image, 'KF Estimate',
                    loc,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    med_text(w), clr.BLACK, lrg_thick(h))
        cv2.putText(image, 'KF Estimate',
                    loc,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    med_text(w), clr.GREEN, med_thick(h))

    @staticmethod
    def _collect_objPts_and_imgPts(y_class_ids, y_centers, idsNamesLocs):
        object_points = []
        image_points = []
        pose_class_ids = []
        for idx, cid in enumerate(y_class_ids):
            if cid < len(idsNamesLocs):
                x, y, z = idsNamesLocs[cid][2:]
                object_points.append([x, y, z])
                image_points.append(y_centers[idx])  # <-- must match scaled K pixel space
                pose_class_ids.append(int(cid))
        return (
            np.asarray(object_points, dtype=np.float64),
            np.asarray(image_points, dtype=np.float64),
            pose_class_ids,
        )

    @staticmethod
    def _draw_overlay_legend(
        image: NDArray,
        used_algos: twoToThreeSelectedAlgorithms,
    ) -> None:
        h, w, _ = image.shape
        x0 = int(0.84 * w)
        bottom_margin = int(0.12 * h)
        row_h = max(16, int(0.034 * h))
        icon_r = max(3, int(0.004 * w))
        font_scale = med_text(w)
        text_thick = med_thick(h)
        legend_rows = 2
        if used_algos.use_wqnp_yolo or used_algos.use_wqnp_kfest:
            legend_rows += 4
        y0 = h - bottom_margin - (legend_rows - 1) * row_h

        def draw_label(row_idx: int, text: str, color) -> None:
            y = y0 + row_idx * row_h
            cv2.putText(image, text, (x0, y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, clr.BLACK, lrg_thick(h))
            cv2.putText(image, text, (x0, y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, text_thick)

        def draw_circle_marker(row_idx: int, color, filled: bool, crosshair: bool = False) -> None:
            cy = y0 + row_idx * row_h - int(0.35 * row_h)
            cx = x0 - int(0.025 * w)
            cv2.circle(image, (cx, cy), icon_r + 2, clr.BLACK, 1)
            cv2.circle(image, (cx, cy), icon_r, color, -1 if filled else 1)
            if crosshair:
                cv2.line(image, (cx - icon_r, cy), (cx + icon_r, cy), color, 1)
                cv2.line(image, (cx, cy - icon_r), (cx, cy + icon_r), color, 1)

        def draw_ellipse_marker(row_idx: int, color) -> None:
            cy = y0 + row_idx * row_h - int(0.35 * row_h)
            cx = x0 - int(0.025 * w)
            cv2.ellipse(image, (cx, cy), (icon_r * 2, icon_r), 20.0, 0, 360, color, 1)

        row_idx = 0
        draw_label(row_idx, "Legend", clr.WHITE)
        row_idx += 1
        draw_circle_marker(row_idx, clr.LIGHTBLUE, filled=True)
        draw_label(row_idx, "YOLO", clr.LIGHTBLUE)
        row_idx += 1

        if used_algos.use_wqnp_yolo or used_algos.use_wqnp_kfest:
            draw_circle_marker(row_idx, clr.GREEN, filled=False)
            draw_label(row_idx, "mKFest", clr.GREEN)
            row_idx += 1
            draw_circle_marker(row_idx, clr.PINK, filled=False)
            draw_label(row_idx, "pKFest", clr.PINK)
            row_idx += 1
            draw_ellipse_marker(row_idx, clr.YELLOWGREEN)
            draw_label(row_idx, "mGate", clr.YELLOWGREEN)
            row_idx += 1
            draw_ellipse_marker(row_idx, clr.ORANGE)
            draw_label(row_idx, "pGate", clr.ORANGE)
            row_idx += 1

        if used_algos.display_feature_ids is None:
            filter_text = "Feats: all"
        else:
            filter_text = "Feats: " + ",".join(str(v) for v in sorted(used_algos.display_feature_ids))
        draw_label(row_idx, filter_text, clr.WHITE)

    def _drawPnP_from_pose(self,
                           image,
                           y_class_ids,
                           y_centers,
                           object_points,
                           rvec,
                           tvec,
                           markup_is_undistorted,
                           calibration,
                           yoloSize,
                           idsNamesLocs,
                           originalSize,
                           idx=0,
                           draw_as_circles=False,
                           display_feature_ids: set[int] | None = None):
        h, w, _ = image.shape

        if idx == 0:
            scale = 0.006
        elif idx == 1:
            scale = 0.010
        else:
            scale = 0.014

        self.draw_proj(
            image=image,
            y_class_ids=y_class_ids,
            y_centers=y_centers,
            object_points=object_points,
            rvec=rvec,
            tvec=tvec,
            markup_is_undistorted=markup_is_undistorted,
            calibration=calibration,
            yoloSize=yoloSize,
            idsNamesLocs=idsNamesLocs,
            originalSize=originalSize,
            title=f'PNP: {tvec[0, 0]:+6.3f}, {tvec[1, 0]:+6.3f}, {tvec[2, 0]:+6.3f} ({np.linalg.norm(tvec[:, 0]):6.3f})',
            rowIDX=idx,
            txt_scale=0.75,
            draw_as_circles=draw_as_circles,
            circle_radius_px=int(round(scale * w)),
            display_feature_ids=display_feature_ids,
        )

    def _drawQnP_from_pose(self,
                           image,
                           y_class_ids,
                           y_centers,
                           object_points,
                           q_rvec,
                           q_tvec,
                           markup_is_undistorted,
                           calibration,
                           yoloSize,
                           idsNamesLocs,
                           originalSize,
                           idx=0,
                           txt_color=clr.ORANGE,
                           title_prefix="QNP",
                           draw_as_circles=False,
                           display_feature_ids: set[int] | None = None):
        h, w, _ = image.shape

        if idx == 0:
            scale = 0.006
        elif idx == 1:
            scale = 0.010
        else:
            scale = 0.014

        self.draw_proj(
            image=image,
            y_class_ids=y_class_ids,
            y_centers=y_centers,
            object_points=object_points,
            rvec=q_rvec.to_rodrigues(),
            tvec=q_tvec,
            markup_is_undistorted=markup_is_undistorted,
            calibration=calibration,
            yoloSize=yoloSize,
            idsNamesLocs=idsNamesLocs,
            originalSize=originalSize,
            title=f'{title_prefix}: {q_tvec[0]:+6.3f}, {q_tvec[1]:+6.3f}, {q_tvec[2]:+6.3f} ({np.linalg.norm(q_tvec):6.3f})',
            rowIDX=idx,
            txt_color=txt_color,
            draw_as_circles=draw_as_circles,
            circle_radius_px=int(round(scale * w)),
            display_feature_ids=display_feature_ids,
        )

    @staticmethod
    def draw_proj(image: NDArray,
                  y_class_ids: list,
                  y_centers: list,
                  object_points: NDArray,
                  rvec: NDArray,
                  tvec: NDArray,
                  markup_is_undistorted: bool,
                  calibration: Calibration,
                  yoloSize,
                  idsNamesLocs,
                  originalSize: tuple[int, int],
                  title: str,
                  rowIDX: int,
                  txt_color=clr.YELLOW,
                  txt_scale=1.0,
                  draw_as_circles: bool = False,
                  circle_radius_px: int | None = None,
                  display_feature_ids: set[int] | None = None):

        h, w, _ = image.shape
        h_ori, w_ori = originalSize
        draw_sx = w / float(w_ori) if w_ori > 0 else 1.0
        draw_sy = h / float(h_ori) if h_ori > 0 else 1.0

        (width, height), base = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, med_text(w), 4)
        lower_left_corner = (int(0.01 * w), int(h - (0.01 * h * (rowIDX + 1)) - height * rowIDX))

        cv2.putText(image,
                    title,
                    lower_left_corner,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    med_text(w),
                    clr.BLACK, lrg_thick(h))
        cv2.putText(image,
                    title,
                    lower_left_corner,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    med_text(w), txt_color, med_thick(h))
        for y_class_id, y_center in zip(y_class_ids, y_centers):
            if display_feature_ids is not None and int(y_class_id) not in display_feature_ids:
                continue

            # for idNameLoc in idsNamesLocs:
            id_num = idsNamesLocs[y_class_id][0]
            # id = y_class_id
            xyz = np.array(idsNamesLocs[y_class_id][2:])

            projectedPixel, _ = cv2.projectPoints(xyz, rvec=rvec, tvec=tvec,
                                                  cameraMatrix=calibration.getCameraMatrix(),
                                                  distCoeffs=np.zeros((5,)))

            if markup_is_undistorted:
                projectedPixel = cv2.undistortPoints(projectedPixel,
                                                     calibration.getCameraMatrix(),
                                                     calibration.getDistortion(),
                                                     P=calibration.remapK,  # If undistorted, remap produces new K
                                                     )
            x, y = np.squeeze(projectedPixel)
            x *= draw_sx
            y *= draw_sy

            if np.isnan(x) or np.isnan(y):
                return

            if 0 < x < w and 0 < y < h:
                if draw_as_circles:
                    # radius scales gently with image size unless overridden
                    r = int(circle_radius_px) if circle_radius_px is not None else max(2, int(round(0.006 * w)))
                    cx, cy = int(round(x)), int(round(y))
                    # outline + fill for contrast
                    cv2.circle(image, (cx, cy), r + 3, clr.BLACK, 1)
                    cv2.circle(image, (cx, cy), r, txt_color, 2)
                else:
                    txt_size = med_text(txt_scale * h)

                    (txt_w, txt_h), base = cv2.getTextSize(str(id_num), cv2.FONT_HERSHEY_SIMPLEX, txt_size,
                                                           med_thick(h))
                    lowerLeftCorner = (int(x - txt_w / 2), int(y + txt_h / 2))

                    cv2.putText(image, str(id_num), lowerLeftCorner, cv2.FONT_HERSHEY_SIMPLEX,
                                txt_size, clr.BLACK, lrg_thick(h))
                    cv2.putText(image, str(id_num), lowerLeftCorner, cv2.FONT_HERSHEY_SIMPLEX,
                                txt_size, txt_color, med_thick(h))
