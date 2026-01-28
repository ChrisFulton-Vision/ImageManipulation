

import numpy as np
from numpy.typing import NDArray
import cv2
from support.vision.calibration import Calibration, undistort_points_px
import support.viz.colors as clr
from support.viz.CVFontScaling import med_text
from support.mathHelpers.twoD_to_threeD import solveQnP
from support.mathHelpers.quaternions import Quaternion as q

class twoToThreeSelectedAlgorithms:
    def __init__(self):
        self.use_pnp = False
        self.use_qnp = False
        self.use_wqnp = False


class pnp_qnp_draw:
    def __init__(self):
        self.last_q_vec = None
        self.last_t_vec = None

        # -----------------------------
        # NEW: estimation-only helpers
        # -----------------------------
    @staticmethod
    def _estimate_pnp(object_points: NDArray,
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

        ret, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=object_points,
            imagePoints=image_points,
            cameraMatrix=calibration.getCameraMatrix(),
            distCoeffs=np.zeros((5,)),
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ret:
            return None
        return rvec, tvec

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
                    circles_not_features: bool = False) -> None:

        h, w, _ = image.shape
        centers_dist, boxes, scores, class_ids, time = output

        y_h, y_w = yoloSize
        sx = w / float(y_w)
        sy = h / float(y_h)

        centers_px = np.asarray(centers_dist, dtype=np.float64)
        centers_px[:, 0] *= sx
        centers_px[:, 1] *= sy

        boxes_px = None
        if boxes is not None and len(boxes) > 0:
            b = np.asarray(boxes, dtype=np.float64)
            b[:, 0] *= sx
            b[:, 2] *= sx
            b[:, 1] *= sy
            b[:, 3] *= sy
            boxes_px = b


        text = f'Inference time: {time:.3f}s'
        (txt_width, txt_height), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, med_text(w), 4)
        cv2.putText(image, text, (10, 10 + int(txt_height)), cv2.FONT_HERSHEY_SIMPLEX, med_text(w), clr.BLACK, 4)
        cv2.putText(image, text, (10, 10 + int(txt_height)), cv2.FONT_HERSHEY_SIMPLEX, med_text(w), clr.LIGHTBLUE, 2)

        centers_und_px = None
        boxes_for_draw_px = boxes_px

        if calibration is not None:
            calibration.scaleCalibration(w)  # K now matches 'image' pixel space

            if centers_px is not None and len(centers_px) > 0:
                centers_und_px = undistort_points_px(calibration, centers_px, eps_px=1e-6)

            if markup_is_undistorted and boxes_px is not None and len(boxes_px) > 0:
                x1, y1, x2, y2 = boxes_px[:, 0], boxes_px[:, 1], boxes_px[:, 2], boxes_px[:, 3]
                corners_px = np.stack([
                    np.stack([x1, y1], axis=1),
                    np.stack([x2, y1], axis=1),
                    np.stack([x2, y2], axis=1),
                    np.stack([x1, y2], axis=1),
                ], axis=1).reshape(-1, 2)

                corners_und_px = undistort_points_px(calibration, corners_px, eps_px=1e-6).reshape(-1, 4, 2)

                x_min = np.min(corners_und_px[:, :, 0], axis=1)
                y_min = np.min(corners_und_px[:, :, 1], axis=1)
                x_max = np.max(corners_und_px[:, :, 0], axis=1)
                y_max = np.max(corners_und_px[:, :, 1], axis=1)

                x_min = np.clip(x_min, 0, w - 1)
                x_max = np.clip(x_max, 0, w - 1)
                y_min = np.clip(y_min, 0, h - 1)
                y_max = np.clip(y_max, 0, h - 1)

                boxes_for_draw_px = np.stack([x_min, y_min, x_max, y_max], axis=1)

        centers_for_draw = centers_px
        centers_for_pnp = centers_px

        if centers_und_px is not None:
            centers_for_pnp = centers_und_px
            if markup_is_undistorted:
                centers_for_draw = centers_und_px

        boxes_for_draw = boxes_for_draw_px.tolist() if boxes_for_draw_px is not None else boxes

        if len(class_ids) == 0:
            return

        indices = cv2.dnn.NMSBoxes(boxes_for_draw, scores, conf, iou)
        newCentersForDraw, newCentersForPnP, newBoxes, newClass_ids, newScores = [], [], [], [], []
        for i in indices:
            ii = int(i[0]) if hasattr(i, "__len__") else int(i)
            newCentersForDraw.append(centers_for_draw[ii])
            newCentersForPnP.append(centers_for_pnp[ii])
            newBoxes.append(boxes_for_draw[ii])
            newClass_ids.append(class_ids[ii])
            newScores.append(scores[ii])

        self._drawBoxes(
            image,
            newCentersForDraw,
            newBoxes,
            newClass_ids,
            newScores,
            yoloSize,
            draw_as_circles=circles_not_features
        )

        if len(set(indices)) <= 5:
            return

        object_points, image_points = self._collect_objPts_and_imgPts(newClass_ids, newCentersForPnP, idsNamesLocs)
        if len(object_points) < 6:
            return

        # 1) Estimate PnP (optional)
        pnp_pose = None
        if usedAlgos.use_pnp:
            pnp_pose = self._estimate_pnp(object_points, image_points, calibration)

        # 2) Estimate QnP (optional) seeded by PnP if available
        qnp_pose = None
        if usedAlgos.use_qnp:
            if pnp_pose is not None:
                seed_rvec, seed_tvec = pnp_pose
            else:
                seed_rvec, seed_tvec = None, None
            qnp_pose = self._estimate_qnp(object_points, image_points, calibration, seed_rvec, seed_tvec)

        # 3) Draw in desired order (match your idx stacking)
        idx = 0
        if usedAlgos.use_qnp and qnp_pose is not None:
            q_rvec, q_tvec = qnp_pose
            self._drawQnP_from_pose(
                image=image,
                y_class_ids=newClass_ids,
                y_centers=newCentersForPnP,
                object_points=object_points,
                q_rvec=q_rvec,
                q_tvec=q_tvec,
                markup_is_undistorted=markup_is_undistorted,
                calibration=calibration,
                yoloSize=yoloSize,
                idsNamesLocs=idsNamesLocs,
                idx=idx,
                draw_as_circles=circles_not_features
            )
            idx += 1

        if usedAlgos.use_pnp and pnp_pose is not None:
            rvec, tvec = pnp_pose
            self._drawPnP_from_pose(
                image=image,
                y_class_ids=newClass_ids,
                y_centers=newCentersForPnP,
                object_points=object_points,
                rvec=rvec,
                tvec=tvec,
                markup_is_undistorted=markup_is_undistorted,
                calibration=calibration,
                yoloSize=yoloSize,
                idsNamesLocs=idsNamesLocs,
                idx=idx,
                draw_as_circles=circles_not_features
            )
            idx += 1

    @staticmethod
    def _drawBoxes(image: NDArray, newCenters: list, newBoxes: list,
                  newClass_ids: list, newScores: list,
                  yoloSize: tuple[float, float],
                   draw_as_circles: bool = True,
                   circle_radius_px: int | None = None) -> NDArray:
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
        y_h, y_w = yoloSize

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
                cv2.rectangle(image, (x1, y1), (x2, y2), clr.LIGHTBLUE, 1)

                (txt_w, txt_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, med_text(w), 4)
                lowerLeftCorner = (int(x-txt_w/2.0), int(y+txt_h/2.0))

                cv2.putText(image, label, lowerLeftCorner, cv2.FONT_HERSHEY_SIMPLEX, med_text(w), clr.BLACK, 6)
                cv2.putText(image, label, lowerLeftCorner, cv2.FONT_HERSHEY_SIMPLEX, med_text(w), clr.LIGHTBLUE, 4)

        (txt_width, txt_height), base = cv2.getTextSize('I', cv2.FONT_HERSHEY_SIMPLEX, med_text(w), 4)
        txt_height_perRow = txt_height + 15
        cv2.putText(image, 'Direct Inference', (10, h - 2 * txt_height_perRow - 15), cv2.FONT_HERSHEY_SIMPLEX,
                med_text(w), clr.BLACK, 4)
        cv2.putText(image, 'Direct Inference', (10, h - 2 * txt_height_perRow - 15), cv2.FONT_HERSHEY_SIMPLEX,
                med_text(w), clr.LIGHTBLUE, 2)

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
                          draw_as_circles=False):
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
            title=f'PNP: {tvec[0,0]:+6.3f}, {tvec[1,0]:+6.3f}, {tvec[2,0]:+6.3f}',
            rowIDX=idx,
            txt_scale=0.75,
            draw_as_circles=draw_as_circles,
            circle_radius_px=int(round(scale * w))
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
                          draw_as_circles=False):
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
            title=f'QNP: {q_tvec[0]:+6.3f}, {q_tvec[1]:+6.3f}, {q_tvec[2]:+6.3f}',
            rowIDX=idx,
            txt_color=clr.ORANGE,
            draw_as_circles=draw_as_circles,
            circle_radius_px=int(round(scale * w))
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
                      txt_color = clr.YELLOW,
                      txt_scale = 1.0,
                      draw_as_circles: bool = True,
                      circle_radius_px: int | None = None):

        h, w, _ = image.shape
        y_h, y_w = yoloSize

        (width, height), base = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, med_text(w), 4)
        lower_left_corner = (int(0.01*w), int(h - (0.01 * h * (rowIDX + 1)) - height * rowIDX))

        cv2.putText(image, title, lower_left_corner, cv2.FONT_HERSHEY_SIMPLEX, med_text(w), clr.BLACK, 4)
        cv2.putText(image, title, lower_left_corner, cv2.FONT_HERSHEY_SIMPLEX, med_text(w), txt_color, 2)
        for y_class_id, y_center in zip(y_class_ids, y_centers):

            # for idNameLoc in idsNamesLocs:
            id = idsNamesLocs[y_class_id][0]
            # id = y_class_id
            xyz = np.array(idsNamesLocs[y_class_id][2:])

            if markup_is_undistorted:
                dist_coeffs = np.zeros((5,))
            else:
                dist_coeffs = calibration.getDistortion()

            projectedPixel, _ = cv2.projectPoints(xyz, rvec=rvec, tvec=tvec,
                                                  cameraMatrix=calibration.getCameraMatrix(),
                                                  distCoeffs=dist_coeffs)

            x, y = np.squeeze(projectedPixel)
            if np.isnan(x) or np.isnan(y):
                return

            # x = float(w / y_w * x)
            # y = float(h / y_h * y)

            if (0 < x < w and 0 < y < h):
                if draw_as_circles:
                    # radius scales gently with image size unless overridden
                    r = int(circle_radius_px) if circle_radius_px is not None else max(2, int(round(0.006 * w)))
                    cx, cy = int(round(x)), int(round(y))
                    # outline + fill for contrast
                    cv2.circle(image, (cx, cy), r + 3, clr.BLACK, 1)
                    cv2.circle(image, (cx, cy), r, txt_color, 2)
                else:
                    (txt_w, txt_h), base = cv2.getTextSize(str(id), cv2.FONT_HERSHEY_SIMPLEX, txt_scale * med_text(w),
                                                           2)
                    lowerLeftCorner = (int(x - txt_w / 2), int(y + txt_h / 2))

                    cv2.putText(image, str(id), lowerLeftCorner, cv2.FONT_HERSHEY_SIMPLEX,
                                txt_scale * med_text(w), clr.BLACK, 4)
                    cv2.putText(image, str(id), lowerLeftCorner, cv2.FONT_HERSHEY_SIMPLEX,
                                txt_scale * med_text(w), txt_color, 2)
