

import numpy as np
from numpy.typing import NDArray
import cv2
from support.vision.calibration import Calibration, undistort_points_px_numba
import support.viz.colors as clr
from support.viz.CVFontScaling import med_text
from support.mathHelpers.twoD_to_threeD import solveQnP


class twoToThreeSelectedAlgorithms:
    def __init__(self):
        self.use_pnp = False
        self.use_qnp = False
        self.use_wqnp = False


class pnp_qnp_draw:
    def __init__(self):
        self.last_q_vec = None
        self.last_t_vec = None

    def markUpImage(self,
                    image: NDArray,
                    output: tuple[list, list, list, list, float],
                    markup_is_undistorted: bool,
                    calibration: Calibration,
                    conf: float,
                    iou: float,
                    yoloSize: tuple[int, int],
                    idsNamesLocs,
                    usedAlgos:twoToThreeSelectedAlgorithms) -> None:
        '''
        Takes image and places bounding boxes on them. If there's more than 5 features, attempts to solvePnP and mark
        up the image with a PnP solution as well.
        :param image: Original OpenCV style np.array
        :param output: processed onnxruntime sessions
        :return: marked-up image
        '''
        h, w, _ = image.shape

        centers_dist, boxes, scores, class_ids, time = output

        text = f'Inference time: {time:.3f}s'
        (txt_width, txt_height), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, med_text(w), 4)
        cv2.putText(image, text, (10, 10 + int(txt_height)), cv2.FONT_HERSHEY_SIMPLEX, med_text(w), clr.BLACK, 4)
        cv2.putText(image, text, (10, 10 + int(txt_height)), cv2.FONT_HERSHEY_SIMPLEX, med_text(w), clr.LIGHTBLUE, 2)

        centers_und = None
        boxes_for_draw = boxes
        if calibration is not None and (markup_is_undistorted or len(set(class_ids)) > 5):
            # Ensure calibration matches YOLO coordinate system
            calibration.scaleCalibration(w)


            # Vectorized: distorted YOLO pixels -> undistorted YOLO pixels
            # (Function name may be cal.undistort_points_px or module-level undistort_points_px depending on your Calibration.py)
            numpy_centers = np.array(centers_dist, dtype=np.float64)

            if len(set(class_ids)) > 0:
                if not calibration.fisheye:
                    centers_und = undistort_points_px_numba(numpy_centers,
                                                    *calibration.iteratable_params,
                                                    calibration.has_tangential,
                                                    mode_opencv_5fp=False,
                                                    eps_px=1e-6)
                else:
                    numpy_centers = numpy_centers.reshape(-1, 1, 2)
                    centers_und = cv2.fisheye.undistortPoints(numpy_centers,
                                                              calibration.getCameraMatrix(),
                                                              calibration.getDistortion(),
                                                              P=calibration.getCameraMatrix()).reshape(-1,2)
                centers_und = centers_und.tolist()

            # ---- Undistort boxes too (undistort corners, then re-AABB) ----
            if markup_is_undistorted and boxes is not None and len(boxes) > 0:
                b = np.asarray(boxes, dtype=np.float64)  # (N,4) in YOLO pixel space: x1,y1,x2,y2

                # Build corner list: (x1,y1), (x2,y1), (x2,y2), (x1,y2) for each box
                x1 = b[:, 0]
                y1 = b[:, 1]
                x2 = b[:, 2]
                y2 = b[:, 3]
                corners = np.stack([
                    np.stack([x1, y1], axis=1),
                    np.stack([x2, y1], axis=1),
                    np.stack([x2, y2], axis=1),
                    np.stack([x1, y2], axis=1),
                ], axis=1).reshape(-1, 2)  # (4N,2)

                if not calibration.fisheye:
                    corners_und = undistort_points_px_numba(
                        corners,
                        *calibration.iteratable_params,
                        calibration.has_tangential,
                        mode_opencv_5fp=False,
                        eps_px=1e-6
                    ).reshape(-1, 4, 2)  # (N,4,2)
                else:
                    corners = corners.reshape(-1, 1, 2)
                    corners_und = cv2.fisheye.undistortPoints(
                        corners,
                        calibration.getCameraMatrix(),
                        calibration.getDistortion(),
                        P=calibration.getCameraMatrix()
                        ).reshape(-1, 4, 2)

                # Rebuild axis-aligned boxes in undistorted YOLO pixel space
                x_min = np.min(corners_und[:, :, 0], axis=1)
                y_min = np.min(corners_und[:, :, 1], axis=1)
                x_max = np.max(corners_und[:, :, 0], axis=1)
                y_max = np.max(corners_und[:, :, 1], axis=1)

                # Clamp to YOLO frame bounds (optional but helps avoid drawing weirdness)
                y_h, y_w = yoloSize
                x_min = np.clip(x_min, 0, y_w - 1)
                x_max = np.clip(x_max, 0, y_w - 1)
                y_min = np.clip(y_min, 0, y_h - 1)
                y_max = np.clip(y_max, 0, y_h - 1)

                boxes_for_draw = np.stack([x_min, y_min, x_max, y_max], axis=1).tolist()

        centers_for_draw = centers_dist
        if centers_und is not None:
            centers_for_pnp = centers_und
            if markup_is_undistorted:
                centers_for_draw = centers_und
        else:
            centers_for_pnp = centers_dist

        if len(class_ids) > 0:
            indices = cv2.dnn.NMSBoxes(boxes_for_draw,
                                       scores, conf, iou)
            newCentersForDraw, newCentersForPnP, newBoxes, newClass_ids, newScores = [], [], [], [], []
            for i in indices:
                ii = int(i[0]) if hasattr(i, "__len__") else int(i)

                newCentersForDraw.append(centers_for_draw[ii])
                newCentersForPnP.append(centers_for_pnp[ii])
                newBoxes.append(boxes_for_draw[ii])
                newClass_ids.append(class_ids[ii])
                newScores.append(scores[ii])
            self._drawBoxes(image,
                               newCentersForDraw,
                               newBoxes,
                               newClass_ids,
                               newScores,
                               yoloSize)

            if len(set(indices)) > 5:
                idx = 0
                if usedAlgos.use_pnp:
                    self._drawPnP(image,
                             newClass_ids,
                             newCentersForPnP,
                             markup_is_undistorted,
                             calibration,
                             yoloSize,
                             idsNamesLocs)
                    idx += 1
                if usedAlgos.use_qnp:
                    self._drawQnP(image,
                             newClass_ids,
                             newCentersForPnP,
                             markup_is_undistorted,
                             calibration,
                             yoloSize,
                             idsNamesLocs,
                             idx)

    @staticmethod
    def _drawBoxes(image: NDArray, newCenters: list, newBoxes: list,
                  newClass_ids: list, newScores: list,
                  yoloSize: tuple[float, float]) -> NDArray:
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
            x = int(w / y_w * x)
            y = int(h / y_h * y)
            x1, y1, x2, y2 = box
            x1 = int(w / y_w * x1)
            x2 = int(w / y_w * x2)
            y1 = int(h / y_h * y1)
            y2 = int(h / y_h * y2)


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
    def _collect_objPts_and_imgPts( y_class_ids, y_centers, idsNamesLocs):
        object_points = []
        image_points = []
        for idx, cid in enumerate(y_class_ids):
            if cid < len(idsNamesLocs):
                x, y, z = idsNamesLocs[cid][2:]
                object_points.append([x, y, z])
                image_points.append(y_centers[idx])  # <-- MUST be in same pixel space as scaled K
        return np.asarray(object_points, dtype=np.float64), np.asarray(image_points, dtype=np.float64)

    def _drawPnP(self,
                 image,
                 y_class_ids,
                 y_centers,
                 markup_is_undistorted,
                 calibration,
                 yoloSize,
                 idsNamesLocs):
        h, w, _ = image.shape

        if calibration is None:
            return

        # calibration.scaleCalibration(y_w)

        object_points, image_points = self._collect_objPts_and_imgPts(y_class_ids, y_centers, idsNamesLocs)
        if len(object_points) < 6:
            return

        ret, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=object_points,
            imagePoints=image_points,
            cameraMatrix=calibration.getCameraMatrix(),
            distCoeffs=np.zeros((5,)),
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ret:
            return

        R, _ = cv2.Rodrigues(rvec)
        P = np.hstack((R, tvec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(P)
        rpy = np.array([euler_angles[0,0], euler_angles[1,0], euler_angles[2,0]])

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
            title=f'PNP: {tvec[0,0]:+6.3f}, {tvec[1,0]:+6.3f}, {tvec[2,0]:+6.3f}', #, {rpy[0]:+4.1f}, {rpy[1]:+4.1f}, {rpy[2]:+4.1f}',
            rowIDX=0
        )

        return (rvec, tvec)

    def _drawQnP(self,
                 image,
                 y_class_ids,
                 y_centers,
                 markup_is_undistorted,
                 calibration,
                 yoloSize,
                 idsNamesLocs,
                 index_for_display):
        h, w, _ = image.shape

        if calibration is None:
            return

        # calibration.scaleCalibration(y_w)

        object_points, image_points = self._collect_objPts_and_imgPts(y_class_ids, y_centers, idsNamesLocs)
        if len(object_points) < 6:
            return

        q_rvec, q_tvec = solveQnP(
            object_pts=object_points,
            img_pts=image_points,
            cal=calibration,
            user_seed_q=self.last_q_vec,
            user_seed_t=self.last_t_vec
        )
        self.last_q_vec, self.last_t_vec = q_rvec, q_tvec

        rpy = q_rvec.eulerD()

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
            title=f'QNP: {q_tvec[0]:+6.3f}, {q_tvec[1]:+6.3f}, {q_tvec[2]:+6.3f}', #, {rpy[0]:+4.1f}, {rpy[1]:+4.1f}, {rpy[2]:+4.1f}',
            rowIDX=index_for_display,
            txt_color = clr.ORANGE,
            txt_scale = 0.75
        )
        return (q_rvec, q_tvec)

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
                      txt_scale = 1.0):

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

            x = float(w / y_w * x)
            y = float(h / y_h * y)

            if (0 < x < w and 0 < y < h):
                (txt_w, txt_h), base = cv2.getTextSize(str(id), cv2.FONT_HERSHEY_SIMPLEX, txt_scale * med_text(w), 2)
                lowerLeftCorner = (int(x-txt_w/2), int(y+txt_h/2))

                cv2.putText(image, str(id), lowerLeftCorner, cv2.FONT_HERSHEY_SIMPLEX,
                            txt_scale * med_text(w), clr.BLACK, 4)
                cv2.putText(image, str(id), lowerLeftCorner, cv2.FONT_HERSHEY_SIMPLEX,
                            txt_scale * med_text(w), txt_color, 2)