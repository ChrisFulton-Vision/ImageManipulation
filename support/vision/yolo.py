import copy
import datetime
import glob
import os
import re
import numpy as np
from numpy.typing import NDArray

from support.io.my_logging import LOG

import cv2
from cv2.dnn import NMSBoxes

from support.vision.calibration import Calibration, undistort_points_px_numba
from support.io.meta_yolo_reader import MetaYoloReader
import support.viz.colors as clr
from support.viz.CVFontScaling import small_text, med_text, lrg_text
from support.io.my_logging import LOG

CUDA_BIN  = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
CUDNN_BIN = r"C:\Program Files\NVIDIA\CUDNN\v9.4\bin\12.6"

if os.path.isdir(CUDA_BIN):
    os.add_dll_directory(CUDA_BIN)

if os.path.isdir(CUDNN_BIN):
    os.add_dll_directory(CUDNN_BIN)

for key in ("CUDA_PATH", "CUDNN_PATH"):
    p = os.environ.get(key)
    if p and os.path.isdir(p):
        os.add_dll_directory(p)

# Ensure CUDA_PATH is in environment: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin
# Ensure CUDNN_PATH is in environment: C:\Program Files\NVIDIA\CUDNN\v9.4\bin\12.6
import onnxruntime as ort

LOG.info(f'OnnxVersion: {ort.__version__}')
LOG.info(f'Onnx Providers: {ort.get_available_providers()}')

# ort.preload_dlls()
# ort.preload_dlls(cuda=False, cudnn=False, msvc=True, directory=None)
ort.preload_dlls(cuda=True, cudnn=True, msvc=True, directory=None)


class YOLO:
    '''
    This class will perform a YOLO inference on a provided image.
    '''

    def __init__(self, conf: float = 0.75, iou: float = 0.99, yoloSize=(864, 864),
                 model_path="YOLOModels/GIII_01172025_10_100M_MoreFeatures/",
                 numClasses: int = 94):
        self.conf = conf
        self.iou = iou
        self.pixel_buffer = 10

        self.modelPath = None
        self.reader = None
        self.provider = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.output = []
        self.boxes, self.scores, self.class_ids = [], [], []
        self.session = None
        self.calibration = None

        self.class_names = range(numClasses)
        self.yoloSize = yoloSize

        self.orig_tvec = []
        self.bias_tvec = []
        self.plotCount = 0

        self.biasTracker = {}

        self.bias_tracking_active = False

        self.setNewFolder(model_path)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def setNewFolder(self, directory: str) -> None:
        '''
        This function changes all the necessary settings for selecting a new YOLO folder. The folder should have
        ONE .onnx file and ONE .csv file. The onnx file should be the yolo model. The csv file should be the
        yolo meta_data.
        :param directory: As a string, the location of the intended directory.
        :return: Nothing
        '''
        if len(glob.glob(os.path.join(directory, f'*.onnx'))) > 0 and len(
                glob.glob(os.path.join(directory, f'*.csv'))) > 0:
            self.modelPath = glob.glob(os.path.join(directory, f'*.onnx'))[0]
            self.reader = MetaYoloReader(glob.glob(os.path.join(directory, f'*.csv'))[0])
            if isinstance(self.reader.imageSize, int):
                self.yoloSize = (self.reader.imageSize, self.reader.imageSize)
            else:
                self.yoloSize = self.reader.imageSize

            import ast, onnx

            self.class_names = range(self.reader.numClasses)

            model_proto = onnx.load(self.modelPath)
            meta = {p.key.lower(): p.value for p in model_proto.metadata_props}

            if "names" in meta:
                raw = meta["names"]
                try:
                    # Try JSON first
                    import json
                    names = json.loads(raw)
                except Exception:
                    # Fall back to literal_eval for Python-style dicts
                    names = ast.literal_eval(raw)

                # If it's a dict, convert to ordered list
                if isinstance(names, dict):
                    names = [names[k] for k in sorted(names.keys(), key=int)]

                if len(self.class_names) != len(names):
                    raise ImportError(
                        f"Onnx file and CSV disagree about number of classes! "
                        f"Onnx: {len(names)} vs CSV: {len(self.class_names)}. Aborting."
                    )

            self.reinitSession()

    def reinitSession(self) -> None:
        '''
        When yolo parameters change, this creates a new session with those parameters. Must be called when
        something changes.
        :return nothing:
        '''
        sess_options = ort.SessionOptions()
        # sess_options.intra_op_num_threads = 1
        # sess_options.inter_op_num_threads = 1
        # sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # sess_options.add_session_config_entry("session.intra_op.allow_spinning", "1")
        self.session = ort.InferenceSession(self.modelPath, sess_options=sess_options, providers=self.provider)

    def inferOnImage(self,
                     image: NDArray,
                     markup_image: NDArray,
                     markup_is_undistorted: bool = False,
                     bias_tracking: bool = False) -> tuple[NDArray, NDArray]:
        '''
        Runs the sub-methods necessary to process an image with YOLO
        :param image: np.array from OpenCV
        :return: Marked-up image post-yolo inference
        '''
        self.bias_tracking_active = bias_tracking
        yoloImage = self.preprocessImage(image)
        output = self.processImage(yoloImage)
        return self.markUpImage(markup_image, output, markup_is_undistorted), output

    def set_calibration(self, calibration: Calibration) -> None:
        self.calibration = copy.deepcopy(calibration)

    def preprocessImage(self, image: NDArray) -> NDArray:
        '''
        This preprocessing:
            Fixes the image to the YOLO network's size
            Transposes the image so that it matches onnxruntime's input format
            Adds a dimension to match onnxruntime's input format
        :param image: np.array from OpenCV
        :return: preprocessed image
        '''
        h, w, _ = image.shape
        if (h, w) != self.yoloSize:
            height, width = self.yoloSize
            image = cv2.resize(image, (width, height))
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32) / 255.0
        return image

    def processImage(self, yoloImage: NDArray) -> NDArray:
        '''
        Clears the 'cache' for previous YOLO solutions, then calls the yolo inference method
        :param yoloImage: np.array that has completed preprocessing
        :return: onnxruntime output
        '''
        self.boxes, self.scores, self.class_ids = [], [], []
        return self.runOneSession(yoloImage)

    def runOneSession(self, yoloImage: NDArray) -> NDArray:
        '''
        Records time before and after a yolo infernce for time differencing. Runs the YOLO session
        :param yoloImage: image that has been through preprocessImage
        :return: outputs from onnxruntime session. Labeled output for clarity.
        '''
        startTime = datetime.datetime.now()
        if self.session is not None:
            output = self.session.run(None, {self.session.get_inputs()[0].name: yoloImage})
        else:
            output = None
        endTime = datetime.datetime.now()
        centers, boxes, scores, class_ids = self.interpretOutput(output)
        return centers, boxes, scores, class_ids, (endTime - startTime).total_seconds()

    def interpretOutput(self, output: NDArray) -> (list, list, list, list):
        '''
        Takes outputs from onnxruntime and processes them
        Filters to retain only the highest-confidence detection for each class
        :param output:  onnxruntime session outputs
        :return: cleaner outputs for interpretation
        '''
        best_detections = {}

        if output is None:
            return [], [], [], []

        preds = np.squeeze(output[0])  # [N, 5+numClasses]
        if preds.ndim != 2 or preds.size == 0:
            return [], [], [], []

        xywhc = preds[:, :5]  # (x,y,w,h,conf)
        classp = preds[:, 5:]  # class probs

        conf_mask = xywhc[:, 4] > self.conf
        if not np.any(conf_mask):
            return [], [], [], []

        xywhc = xywhc[conf_mask]
        classp = classp[conf_mask]

        class_id = classp.argmax(axis=1)
        max_class = classp.max(axis=1)
        combined = xywhc[:, 4] * max_class

        x, y, w, h = xywhc[:, 0], xywhc[:, 1], xywhc[:, 2], xywhc[:, 3]
        x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2

        H, W = self.yoloSize
        buf = 10
        in_bounds = (x1 - buf >= 0) & (y1 - buf >= 0) & (x2 + buf <= W) & (y2 + buf <= H)

        x = x[in_bounds]
        y = y[in_bounds]
        x1 = x1[in_bounds]
        y1 = y1[in_bounds]
        x2 = x2[in_bounds]
        y2 = y2[in_bounds]
        class_id = class_id[in_bounds]
        combined = combined[in_bounds]
        score_obj = xywhc[in_bounds, 4]  # objectness as your "score"

        # keep best per class
        keep = {}
        for i, cid in enumerate(class_id):
            s = combined[i]
            if (cid not in keep) or (s > keep[cid][0]):
                keep[cid] = (s, [x[i], y[i]], [x1[i], y1[i], x2[i], y2[i]], float(score_obj[i]))

        centers = [v[1] for v in keep.values()]
        boxes = [v[2] for v in keep.values()]  # still x1,y1,x2,y2 as you expect
        scores = [v[3] for v in keep.values()]
        classes = [int(k) for k in keep.keys()]

        return centers, boxes, scores, classes

    def markUpImage(self, image: NDArray,
                    output: (list, list, list, list),
                    markup_is_undistorted: bool) -> tuple[NDArray, tuple[NDArray, NDArray]]:
        '''
        Takes image and places bounding boxes on them. If there's more than 5 features, attempts to solvePnP and mark
        up the image with a PnP solution as well.
        :param image: Original OpenCV style np.array
        :param output: processed onnxruntime sessions
        :return: marked-up image
        '''
        return image, (0.0, 0.0)
        h, w, _ = image.shape

        centers_dist, boxes, scores, class_ids, time = output

        text = f'Inference time: {time:.3f}s'
        (txt_width, txt_height), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, med_text(w), 4)
        cv2.putText(image, text, (10, 10 + int(txt_height)), cv2.FONT_HERSHEY_SIMPLEX, med_text(w), clr.BLACK, 4)
        cv2.putText(image, text, (10, 10 + int(txt_height)), cv2.FONT_HERSHEY_SIMPLEX, med_text(w), clr.LIGHTBLUE, 2)

        centers_und = None
        if self.calibration is not None and (markup_is_undistorted or len(set(class_ids)) > 5):
            # Ensure calibration matches YOLO coordinate system (you already do this in drawPnP)
            # Better: do it here once, before both PnP and draw
            y_h, y_w = self.yoloSize
            self.calibration.scaleCalibration(y_w)  # same logic you already use :contentReference[oaicite:2]{index=2}

            # Vectorized: distorted YOLO pixels -> undistorted YOLO pixels
            centers_und = undistort_points_px_numba(np.array(centers_dist, dtype=np.float64),
                                                    *self.calibration.iteratable_params,
                                                    self.calibration.has_tangential,
                                                    mode_opencv_5fp=False,
                                                    eps_px=1e-6)
            centers_und = centers_und.tolist()

        centers_for_draw = centers_dist
        if centers_und is not None:
            centers_for_pnp = centers_und
            if markup_is_undistorted:
                centers_for_draw = centers_und
        else:
            centers_for_pnp = centers_dist

        if len(class_ids) > 0:
            indices = NMSBoxes(boxes, scores, self.conf, self.iou)
            newCentersForDraw, newCentersForPnP, newBoxes, newClass_ids, newScores = [], [], [], [], []
            for i in indices:
                # for i in range(len(centers)):
                newCentersForDraw.append(centers_for_draw[i])
                newCentersForPnP.append(centers_for_pnp[i])
                newBoxes.append(boxes[i])
                newClass_ids.append(class_ids[i])
                newScores.append(scores[i])

            image = self.drawBoxes(image, newCentersForDraw, newBoxes, newClass_ids, newScores)
            if len(set(indices)) > 5:
                rvec_tvec = self.drawPnP(image,
                                         newClass_ids,
                                         newCentersForPnP,
                                         markup_is_undistorted)
                return image, rvec_tvec

        return image, None

    def drawBoxes(self, image: NDArray, newCenters: list, newBoxes: list,
                  newClass_ids: list, newScores: list) -> NDArray:
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
        y_h, y_w = self.yoloSize

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
        txt_height_perRow = txt_height + 10
        cv2.putText(image, 'Direct Inference', (10, h - 2 * txt_height_perRow - 10), cv2.FONT_HERSHEY_SIMPLEX,
                med_text(w), clr.BLACK, 4)
        cv2.putText(image, 'Direct Inference', (10, h - 2 * txt_height_perRow - 10), cv2.FONT_HERSHEY_SIMPLEX,
                med_text(w), clr.LIGHTBLUE, 2)

        return image

    def collect_objPts_and_imgPts(self, y_class_ids, y_centers):
        object_points = []
        image_points = []
        for idx, y_class_id in enumerate(y_class_ids):
            if y_class_id < len(self.reader.idsNamesLocs):
                x, y, z = self.reader.idsNamesLocs[y_class_id][2:]
                object_points.append([x, y, z])
                image_points.append(y_centers[idx])

        object_points = np.array(object_points)
        image_points = np.array(image_points)
        return object_points, image_points

    def drawPnP(self,
                image: NDArray,
                y_class_ids: list,
                y_centers: list,
                markup_is_undistorted: bool) -> tuple[NDArray, NDArray] | None:
        '''
        If enough features are detected, calculates the PnP solution for the image. Then, draws the reprojection
        onto the image. Note that the image is received by reference, and the image isn't needed to be returned because
        the original image is directly modified.
        :param image: OpenCV marked-up image
        :param y_class_ids: list of class ids for the solution
        :param y_centers: list of center pixels for the solution
        :return: Nothing
        '''
        h, w, _ = image.shape
        y_h, y_w = self.yoloSize

        if self.calibration is None:
            return

        self.calibration.scaleCalibration(y_w)

        object_points, image_points = self.collect_objPts_and_imgPts(y_class_ids, y_centers)

        if len(object_points) < 6:
            return

        ret, rvec, tvec, inliers = cv2.solvePnPRansac(objectPoints=object_points,
                                                      imagePoints=image_points,
                                                      cameraMatrix=self.calibration.getCameraMatrix(),
                                                      distCoeffs=np.zeros((5,)),
                                                      confidence=0.99,
                                                      flags=cv2.SOLVEPNP_ITERATIVE)

        dcm, jacob = cv2.Rodrigues(rvec)
        np.set_printoptions(suppress=True, precision=10)


        if not ret:
            return

        self.orig_tvec.append(tvec)
        vec_str = f'x:{tvec[0, 0]:+.3f}, y:{tvec[1, 0]:+.3f}, z:{tvec[2, 0]:+.3f}'
        txt = 'SolvePnP Solution'
        (vec_width, vec_height), base = cv2.getTextSize(vec_str, cv2.FONT_HERSHEY_SIMPLEX,
                                                          med_text(w), 4)

        cv2.putText(image, 'SolvePnP Solution', (10, h - vec_height - 20), cv2.FONT_HERSHEY_SIMPLEX,
                med_text(w), clr.BLACK, 4)
        cv2.putText(image, 'SolvePnP Solution', (10, h - vec_height - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    med_text(w), clr.YELLOW, 2)

        cv2.putText(image, vec_str, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, med_text(w), clr.BLACK, 4)
        cv2.putText(image, vec_str, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, med_text(w), clr.YELLOW, 2)

        self.draw_PnP_proj(image, y_class_ids, y_centers, object_points, rvec, tvec, markup_is_undistorted)

        return (rvec, tvec)

    def draw_PnP_proj(self, image: NDArray,
                      y_class_ids: list,
                      y_centers: list,
                      object_points: NDArray,
                      rvec: NDArray,
                      tvec: NDArray,
                      markup_is_undistorted: bool):

        h, w, _ = image.shape
        y_h, y_w = self.yoloSize

        for y_class_id, y_center in zip(y_class_ids, y_centers):
            if y_class_id <= len(self.reader.idsNamesLocs):

                # for idNameLoc in reader.idsNamesLocs:
                id = self.reader.idsNamesLocs[y_class_id][0]
                xyz = np.array(self.reader.idsNamesLocs[y_class_id][2:])

                if markup_is_undistorted:
                    dist_coeffs = np.zeros((5,))
                else:
                    dist_coeffs = self.calibration.getDistortion()

                projectedPixel, _ = cv2.projectPoints(xyz, rvec=rvec, tvec=tvec,
                                                      cameraMatrix=self.calibration.getCameraMatrix(),
                                                      distCoeffs=dist_coeffs)

                x, y = np.squeeze(projectedPixel)
                if np.isnan(x) or np.isnan(y):
                    return
                x = w / y_w * x
                y = h / y_h * y

                x_yolo, y_yolo = y_center

                # This section establishes a threshold for error estimates that are not outlier rejected
                if (x - x_yolo) ** 2.0 + (y - y_yolo) ** 2.0 < 40.0 ** 2:
                    if y_class_id in self.biasTracker:
                        num, x_bias, y_bias = self.biasTracker[y_class_id]
                        if num > 9:
                            num = 9
                        self.biasTracker[y_class_id] = [num + 1, (x_bias * num + x - x_yolo) / (num + 1),
                                                        (y_bias * num + y - y_yolo) / (num + 1)]
                    else:
                        self.biasTracker[y_class_id] = [1, x - x_yolo, y - y_yolo]

                (txt_w, txt_h), base = cv2.getTextSize(str(id), cv2.FONT_HERSHEY_SIMPLEX, med_text(w), 2)
                lowerLeftCorner = (int(x-txt_w/2), int(y+txt_h/2))

                cv2.putText(image, str(id), lowerLeftCorner, cv2.FONT_HERSHEY_SIMPLEX,
                            med_text(w), clr.BLACK, 4)
                cv2.putText(image, str(id), lowerLeftCorner, cv2.FONT_HERSHEY_SIMPLEX,
                            med_text(w), clr.YELLOW, 2)

                if self.bias_tracking_active and y_class_id in self.biasTracker:
                    lowerLeftCorner = (int(x_yolo + x_corr - txt_w / 2), int(y_yolo + y_corr + txt_h / 2))

                    num, x_corr, y_corr = self.biasTracker[y_class_id]
                    cv2.putText(image, str(id), (int(x_yolo + x_corr), int(y_yolo + y_corr)), cv2.FONT_HERSHEY_SIMPLEX,
                                med_text(w), clr.RED, 2)

        bias_image_points = []
        for idx, y_class_id in enumerate(y_class_ids):
            if y_class_id < len(self.reader.idsNamesLocs):
                if y_class_id in self.biasTracker:
                    num, x_corr, y_corr = self.biasTracker[y_class_id]
                    x_yolo, y_yolo = y_centers[idx]
                    bias_image_points.append((x_yolo + x_corr, y_yolo + y_corr))
                else:
                    bias_image_points.append(y_centers[idx])
        bias_image_points = np.array(bias_image_points)

        ret, bias_rvec, bias_tvec, inliers = cv2.solvePnPRansac(objectPoints=object_points,
                                                                imagePoints=bias_image_points,
                                                                cameraMatrix=self.calibration.getCameraMatrix(),
                                                                distCoeffs=np.zeros((5,)),
                                                                flags=cv2.SOLVEPNP_ITERATIVE)
        self.bias_tvec.append(bias_tvec)
        self.plotCount += 1
        # print(np.squeeze(np.array(self.orig_tvec)))
        tvecs = np.squeeze(np.array(self.orig_tvec))
        # print()
        # print(np.squeeze(np.array(self.bias_tvec)))
        bias_tvecs = np.squeeze(np.array(self.bias_tvec))
        # print('\n\n')

        # print(self.plotCount)
        # if len(tvecs.shape) > 1 and self.plotCount > 200:
        #     plt.title("Rigid vs. Semi-Rigid 3D Model Solve-PnP Solution")
        #     plt.xlabel("Frame Number")
        #     plt.ylabel("")
        #     plt.plot(tvecs[:, 2], label='Rigid Model', linewidth=2.0)
        #     plt.plot(bias_tvecs[:, 2], label='Semi-Rigid Model', linewidth=2.0)
        #     plt.legend()
        #     plt.tight_layout()
        #     plt.show()
        #     self.plotCount = 0
        if self.bias_tracking_active:
            (txt_width, txt_height), base = cv2.getTextSize("I", cv2.FONT_HERSHEY_SIMPLEX, med_text(w), 4)
            txt_height_perRow = txt_height + 10
            cv2.putText(image, f'x:{bias_tvec[0, 0]:+.3f}, y:{bias_tvec[1, 0]:+.3f}, z:{bias_tvec[2, 0]:+.3f}',
                        (10, h - 3 * txt_height_perRow - 10), cv2.FONT_HERSHEY_SIMPLEX, med_text(w), clr.BLACK, 2)
            cv2.putText(image, f'x:{bias_tvec[0, 0]:+.3f}, y:{bias_tvec[1, 0]:+.3f}, z:{bias_tvec[2, 0]:+.3f}',
                        (10, h - 3 * txt_height_perRow - 10), cv2.FONT_HERSHEY_SIMPLEX, med_text(w), clr.RED, 2)


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


if __name__ == '__main__':
    from cv2 import imshow, imread, waitKey

    yolo = YOLO(conf=0.75, iou=0.99, yoloSize=(864, 864),
                model_path="C:/repos/aburn/usr/hub/palindrome_playground/src/sn_UAS_Guidance/YOLO Models/Atterbury_Cub",
                numClasses=1)

    np.set_printoptions(suppress=True)

    # testImage = imread('BoundingBoxCandidates/13608.bmp')
    # testImage, sol = yolo.inferOnImage(testImage)

    allImages = glob.glob(
        os.path.join('C:/Users/fulto/Desktop/UAS Flight Test/25_Spring/__Flight 2_25_05_19', f'*.bmp'))

    allImages = natural_sort(allImages)

    for imgFP in allImages:
        (newImg, rvec_tvec), sol = yolo.inferOnImage(imread(imgFP))
        imshow('YOLO', newImg)
        # cv2.imwrite('BoundingBoxCandidates/SaveFiles/' + os.path.basename(imgFP), newImg)
        key = waitKey(0)
        if key == 121:
            print('you hit yes')
            with open("test.txt", "w") as f:
                f.write("string")
        if key == 110:
            print('you hit no')
            os.remove(imgFP)
        if key == 27:
            break
