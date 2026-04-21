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

    qnp_q: q = None  # Quaternion object from solveQnP
    qnp_tvec: NDArray | None = None

    object_points: NDArray | None = None
    image_points: NDArray | None = None
    class_ids: list[int] | None = None


class twoToThreeSelectedAlgorithms:
    def __init__(self):
        self.use_pnp = False
        self.use_qnp = False
        self.use_wqnp = False


class pnp_qnp_draw:
    def __init__(self):
        self.last_q_vec = None
        self.last_t_vec = None
        self.last_pnp_rvec = None
        self.last_pnp_tvec = None

        # -----------------------------
        # NEW: estimation-only helpers
        # -----------------------------

    def _estimate_pnp(self,
                      object_points: NDArray,
                      image_points: NDArray,
                      calibration: Calibration):
        """
        Returns (rvec, tvec) from solvePnPRansac, or None if it fails.
        """
        if calibration is None:
            return None
        if object_points is None or image_points is None:
            return None
        if len(object_points) < 6:
            return None

        camera_matrix = calibration.getCameraMatrix()
        dist_coeffs = np.zeros((5,))

        if self.last_pnp_rvec is not None and self.last_pnp_tvec is not None:
            ret, rvec, tvec = cv2.solvePnP(
                objectPoints=object_points,
                imagePoints=image_points,
                cameraMatrix=camera_matrix,
                distCoeffs=dist_coeffs,
                rvec=self.last_pnp_rvec,
                tvec=self.last_pnp_tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if ret:
                self.last_pnp_rvec = rvec
                self.last_pnp_tvec = tvec
                return rvec, tvec

        fast_flag = cv2.SOLVEPNP_SQPNP if hasattr(cv2, "SOLVEPNP_SQPNP") else cv2.SOLVEPNP_EPNP
        ret, rvec, tvec = cv2.solvePnP(
            objectPoints=object_points,
            imagePoints=image_points,
            cameraMatrix=camera_matrix,
            distCoeffs=dist_coeffs,
            flags=fast_flag,
        )
        if ret:
            self.last_pnp_rvec = rvec
            self.last_pnp_tvec = tvec
            return rvec, tvec

        ret, rvec, tvec, _inliers = cv2.solvePnPRansac(
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
            return rvec, tvec
        return None

    def _estimate_qnp(self,
                      object_points: NDArray,
                      image_points: NDArray,
                      calibration: Calibration,
                      seed_rvec=None,
                      seed_tvec=None):
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
        elif self.last_q_vec is not None and self.last_t_vec is not None:
            user_seed_q = self.last_q_vec
            user_seed_t = self.last_t_vec

        q_rvec, q_tvec = solveQnP(
            object_pts=object_points,
            img_pts=image_points,
            cal=calibration,
            user_seed_q=user_seed_q,
            user_seed_t=user_seed_t,
            # Turn this off because we are explicitly controlling the seed now.
            use_solvePnP_as_seed=False
        )

        # Persist last good solution
        self.last_q_vec, self.last_t_vec = q_rvec, q_tvec
        return q_rvec, q_tvec

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
                    img_scale: float = 1.0) -> PoseOutput | None:

        h, w, _ = image.shape
        h_ori, w_ori = originalSize

        centers_dist, boxes, scores, class_ids, time = output
        if len(centers_dist) < 1:
            return None
        y_h, y_w = yoloSize
        sx = w_ori / float(y_w)
        sy = h_ori / float(y_h)

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

                undist_boxes.append([
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                ])

            boxes_for_draw = undist_boxes

        if len(class_ids) == 0:
            return None

        boxes_for_draw = boxes_for_draw if boxes_for_draw is not None else boxes

        boxes_for_nms = []
        for box in boxes_for_draw:
            x1, y1, x2, y2 = box
            boxes_for_nms.append([
                float(x1),
                float(y1),
                float(x2 - x1),
                float(y2 - y1),
            ])

        if len(class_ids) == 0:
            return None

        indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores, conf, iou)

        if len(indices) == 0:
            return None

        keep = np.array(
            [int(i[0]) if hasattr(i, "__len__") else int(i) for i in indices],
            dtype=np.int32
        )

        newCentersForDraw = centers_for_draw[keep]
        newCentersForPnp = centers_for_pnp[keep]
        newBoxes = boxes_for_draw[keep] if isinstance(boxes_for_draw, np.ndarray) else [boxes_for_draw[i] for i in keep]
        newClass_ids = [class_ids[i] for i in keep]
        newScores = [scores[i] for i in keep]

        self._drawBoxes(
            image,
            newCentersForDraw,
            newBoxes,
            newClass_ids,
            newScores,
            draw_as_circles=circles_not_features,
        )

        if len(set(indices)) <= 5:
            return None

        object_points, image_points = self._collect_objPts_and_imgPts(newClass_ids,
                                                                      newCentersForPnp,
                                                                      idsNamesLocs)
        if len(object_points) < 6:
            return None

        # 1) Estimate PnP (optional)
        pnp_pose = None
        if usedAlgos.use_pnp:
            pnp_pose = self._estimate_pnp(object_points,
                                          image_points,
                                          calibration)

        # 2) Estimate QnP (optional) seeded by PnP if available
        qnp_pose = None
        if usedAlgos.use_qnp:
            if pnp_pose is not None:
                seed_rvec, seed_tvec = pnp_pose
            else:
                seed_rvec, seed_tvec = None, None
            qnp_pose = self._estimate_qnp(object_points,
                                          image_points,
                                          calibration,
                                          seed_rvec,
                                          seed_tvec)

        # 3) Draw in desired order (match your idx stacking)
        idx = 0
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
                idx=idx,
                draw_as_circles=circles_not_features,
                img_scale=img_scale
            )
            idx += 1

        if usedAlgos.use_pnp and pnp_pose is not None:
            rvec, tvec = pnp_pose
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
                idx=idx,
                draw_as_circles=circles_not_features,
                img_scale=img_scale
            )
            idx += 1

        return PoseOutput(
            pnp_rvec=pnp_pose[0] if pnp_pose is not None else None,
            pnp_tvec=pnp_pose[1] if pnp_pose is not None else None,
            qnp_q=qnp_pose[0] if qnp_pose is not None else None,
            qnp_tvec=qnp_pose[1] if qnp_pose is not None else None,
            object_points=object_points,
            image_points=image_points,
            class_ids=list(newClass_ids),
        )

    @staticmethod
    def _drawBoxes(image: NDArray, newCenters: NDArray, newBoxes: NDArray,
                   newClass_ids: list, newScores: list,
                   draw_as_circles: bool = True,
                   circle_radius_px: int | None = None,) -> None:
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

        for (centers, box, class_id, score) in zip(newCenters, newBoxes, newClass_ids, newScores):
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
        loc = (pad, h - 4 * txt_height_perRow - pad)

        cv2.putText(image, 'Direct Inference',
                    loc,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    med_text(w), clr.BLACK, lrg_thick(h))
        cv2.putText(image, 'Direct Inference',
                    loc,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    med_text(w), clr.LIGHTBLUE, med_thick(h))

    @staticmethod
    def _collect_objPts_and_imgPts(y_class_ids, y_centers, idsNamesLocs):
        object_points = []
        image_points = []
        for idx, cid in enumerate(y_class_ids):
            if cid < len(idsNamesLocs):
                x, y, z = idsNamesLocs[cid][2:]
                object_points.append([x, y, z])
                image_points.append(y_centers[idx])  # <-- must match scaled K pixel space
        return np.asarray(object_points, dtype=np.float64), np.asarray(image_points, dtype=np.float64)

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
                           idx=0,
                           draw_as_circles=False,
                           img_scale: float = 1.0):
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
            title=f'PNP: {tvec[0, 0]:+6.3f}, {tvec[1, 0]:+6.3f}, {tvec[2, 0]:+6.3f} ({np.linalg.norm(tvec[:, 0]):6.3f})',
            rowIDX=idx,
            txt_scale=0.75,
            draw_as_circles=draw_as_circles,
            circle_radius_px=int(round(scale * w)),
            img_scale=img_scale
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
                           idx=0,
                           draw_as_circles=False,
                           img_scale: float = 1.0):
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
            title=f'QNP: {q_tvec[0]:+6.3f}, {q_tvec[1]:+6.3f}, {q_tvec[2]:+6.3f} ({np.linalg.norm(q_tvec):6.3f})',
            rowIDX=idx,
            txt_color=clr.ORANGE,
            draw_as_circles=draw_as_circles,
            circle_radius_px=int(round(scale * w)),
            img_scale=img_scale
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
                  title: str,
                  rowIDX: int,
                  txt_color=clr.YELLOW,
                  txt_scale=1.0,
                  draw_as_circles: bool = False,
                  circle_radius_px: int | None = None,
                  img_scale: float = 1.0):

        h, w, _ = image.shape
        y_h, y_w = yoloSize

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

            # for idNameLoc in idsNamesLocs:
            id = idsNamesLocs[y_class_id][0]
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

            if np.isnan(x) or np.isnan(y):
                return

            if (0 < x < w and 0 < y < h):
                if draw_as_circles:
                    # radius scales gently with image size unless overridden
                    r = int(circle_radius_px) if circle_radius_px is not None else max(2, int(round(0.006 * w)))
                    cx, cy = int(round(img_scale * x)), int(round(img_scale * y))
                    # outline + fill for contrast
                    cv2.circle(image, (cx, cy), r + 3, clr.BLACK, 1)
                    cv2.circle(image, (cx, cy), r, txt_color, 2)
                else:
                    txt_size = med_text(txt_scale * h)

                    (txt_w, txt_h), base = cv2.getTextSize(str(id), cv2.FONT_HERSHEY_SIMPLEX, txt_size,
                                                           med_thick(h))
                    lowerLeftCorner = (int(img_scale * x - txt_w / 2), int(img_scale * y + txt_h / 2))

                    cv2.putText(image, str(id), lowerLeftCorner, cv2.FONT_HERSHEY_SIMPLEX,
                                txt_size, clr.BLACK, lrg_thick(h))
                    cv2.putText(image, str(id), lowerLeftCorner, cv2.FONT_HERSHEY_SIMPLEX,
                                txt_size, txt_color, med_thick(h))
