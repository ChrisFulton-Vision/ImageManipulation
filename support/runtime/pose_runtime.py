from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from support.core.enums import ImageSource
import support.gui.UserSelectQueue as GuiQueue
from support.mathHelpers.quaternions import Quaternion as q, mat2quat
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

    def run_yolo(self, frame: NDArray, markup_frame: NDArray, ctx: GuiQueue.FrameCtx, args) -> None:
        opts = self.owner.parse_args(args, GuiQueue.YoloOpts())

        from support.vision import yolo

        if self.owner.yoloSession is None:
            self.owner.yoloSession = yolo.YOLO()
            self.owner.yoloSession.setNewFolder(self.owner.camConfig.yoloFilepath)
            self.owner.yoloSession.set_calibration(self.owner.calibration)
            self.owner.yoloSession.iou = self.owner.camConfig.yolo_iou
            self.owner.yoloSession.conf = self.owner.camConfig.yolo_conf

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
        algos.use_wqnp = opts.want_wqnp

        scale = ctx.resize.get_or(1.0)

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
            img_scale=scale,
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
                "q": pose_output.qnp_q,
                "tvec": pose_output.qnp_tvec,
                "object_points": pose_output.object_points,
                "image_points": pose_output.image_points,
                "class_ids": pose_output.class_ids,
            } if pose_output.qnp_q is not None and pose_output.qnp_tvec is not None else None
        else:
            self.owner.pnpResult = None
            self.owner.qnpResult = None

        centers, boxes, scores, class_ids, img_time = output
        last_yolo_center = None
        last_bounding_box_size = None
        last_yolo_3d_estimate = None
        if len(centers) > 0 and self.owner.yoloSession.reader.numClasses == 1:
            best_idx = scores.index(max(scores))
            img_yolo_x_correction = markup_frame.shape[0] / self.owner.yoloSession.reader.imageSize
            img_yolo_y_correction = markup_frame.shape[1] / self.owner.yoloSession.reader.imageSize

            last_bounding_box_size = (
                (boxes[best_idx][2] - boxes[best_idx][0]) * img_yolo_x_correction,
                (boxes[best_idx][3] - boxes[best_idx][1]) * img_yolo_y_correction,
            )
            last_yolo_center = (
                int(centers[best_idx][0] * img_yolo_x_correction),
                int(centers[best_idx][1] * img_yolo_y_correction),
            )

            self.owner.calibration.scaleCalibration(markup_frame.shape[0])
            K = self.owner.calibration.getCameraMatrix()
            two_d_points = np.array([last_yolo_center[0], last_yolo_center[1], 1.0]) * scale
            dist_est = self.owner.calibration.fx * 4.07 / last_bounding_box_size[0]

            if self.check_above_horizon(last_yolo_center):
                last_yolo_3d_estimate = np.linalg.inv(K).dot(two_d_points) * dist_est
                w, h, _ = markup_frame.shape
                (_txt_width, txt_height), _base = cv2.getTextSize(
                    "I",
                    cv2.FONT_HERSHEY_SIMPLEX,
                    med_text(w),
                    med_thick(h),
                )
                pad = int(0.3 * txt_height)
                cv2.putText(
                    markup_frame,
                    "BB-Width Solution",
                    (pad, w - 2 * pad - txt_height),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    med_text(markup_frame.shape[0]),
                    (50, 255, 255),
                    med_thick(h),
                )
                cv2.putText(
                    markup_frame,
                    f"x:{last_yolo_3d_estimate[0]:+.3f}, y:{last_yolo_3d_estimate[1]:+.3f}, z:{last_yolo_3d_estimate[2]:+.3f}",
                    (pad, h - pad),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    med_text(markup_frame.shape[0]),
                    (50, 255, 255),
                    med_thick(h),
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
