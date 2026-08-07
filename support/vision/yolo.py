import copy
import datetime
import glob
import os

import cv2
import numpy as np
from numpy.typing import NDArray

from support.io.meta_yolo_reader import MetaYoloReader
from support.io.my_logging import LOG
from support.vision.calibration import Calibration

CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
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
    """
    This class will perform a YOLO inference on a provided image.
    """

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
        self.input_name = None
        self.calibration = None
        self._input_tensor = None
        self._resize_buffer = None

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
        """
        This function changes all the necessary settings for selecting a new YOLO folder. The folder should have
        ONE .onnx file and ONE .csv file. The onnx file should be the yolo model. The csv file should be the
        yolo meta_data.
        :param directory: As a string, the location of the intended directory.
        :return: Nothing
        """
        if len(glob.glob(os.path.join(directory, f'*.onnx'))) > 0 and len(
                glob.glob(os.path.join(directory, f'*.csv'))) > 0:
            self.modelPath = glob.glob(os.path.join(directory, f'*.onnx'))[0]
            self.reader = MetaYoloReader(glob.glob(os.path.join(directory, f'*.csv'))[0])
            if isinstance(self.reader.imageSize, int):
                self.yoloSize = (self.reader.imageSize, self.reader.imageSize)
            else:
                self.yoloSize = self.reader.imageSize

            import ast
            import onnx

            self.class_names = range(self.reader.numClasses)

            model_proto = onnx.load(self.modelPath)
            meta = {prop.key.lower(): prop.value for prop in model_proto.metadata_props}

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
        """
        When yolo parameters change, this creates a new session with those parameters. Must be called when
        something changes.
        :return nothing:
        """
        sess_options = ort.SessionOptions()
        # sess_options.intra_op_num_threads = 1
        # sess_options.inter_op_num_threads = 1
        # sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # sess_options.add_session_config_entry("session.intra_op.allow_spinning", "1")
        self.session = ort.InferenceSession(self.modelPath, sess_options=sess_options, providers=self.provider)
        self.input_name = self.session.get_inputs()[0].name

    def inferOnImage(self,
                     image: NDArray,
                     bias_tracking: bool = False) -> tuple[list, list, list, list, float]:
        """
        Runs the sub-methods necessary to process an image with YOLO
        :param bias_tracking:
        :param image: np.array from OpenCV
        :return: Marked-up image post-yolo inference
        """
        self.bias_tracking_active = bias_tracking
        yoloImage = self.preprocessImage(image)
        output = self.processImage(yoloImage)
        return output

    def inferClassCandidatesOnImage(self,
                                    image: NDArray,
                                    target_class: int,
                                    min_score: float = 0.01
                                    ) -> tuple[NDArray, NDArray, NDArray, NDArray, float]:
        """
        Runs YOLO and returns all raw candidate boxes for one class before the
        normal best-per-class reduction performed by interpretOutput().

        Returns:
            boxes
            objectness
            classness
            combined_score = objectness * classness
            inference_time
        """
        yoloImage = self.preprocessImage(image)

        startTime = datetime.datetime.now()
        if self.session is not None:
            output = self.session.run(None, {self.input_name: yoloImage})
        else:
            output = None
        endTime = datetime.datetime.now()

        boxes, objectness, classness, combined = self.interpretClassCandidates(
            output, target_class, min_score
        )

        return (
            boxes,
            objectness,
            classness,
            combined,
            (endTime - startTime).total_seconds()
        )

    def interpretClassCandidates(self,
                                 output: NDArray,
                                 target_class: int,
                                 min_score: float = 0.01
                                 ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Extract every raw proposal's score for one requested class.

        No argmax class assignment and no best-per-class reduction are applied.
        """

        if output is None:
            empty_boxes = np.empty((0, 4), dtype=np.float32)
            empty_scores = np.empty(0, dtype=np.float32)
            return empty_boxes, empty_scores, empty_scores, empty_scores

        preds = np.squeeze(output[0])

        if preds.ndim != 2 or preds.size == 0:
            empty_boxes = np.empty((0, 4), dtype=np.float32)
            empty_scores = np.empty(0, dtype=np.float32)
            return empty_boxes, empty_scores, empty_scores, empty_scores

        if target_class < 0 or target_class >= preds.shape[1] - 5:
            raise ValueError(
                f"target_class {target_class} is outside "
                f"[0, {preds.shape[1] - 6}]"
            )

        x = preds[:, 0]
        y = preds[:, 1]
        w = preds[:, 2]
        h = preds[:, 3]

        objectness = preds[:, 4]
        classness = preds[:, 5 + target_class]
        combined = objectness * classness

        # Much lower than normal YOLO acceptance threshold.
        # This only removes essentially meaningless proposals.
        keep = combined >= min_score

        if not np.any(keep):
            empty_boxes = np.empty((0, 4), dtype=np.float32)
            empty_scores = np.empty(0, dtype=np.float32)
            return empty_boxes, empty_scores, empty_scores, empty_scores

        x = x[keep]
        y = y[keep]
        w = w[keep]
        h = h[keep]

        objectness = objectness[keep]
        classness = classness[keep]
        combined = combined[keep]

        boxes = np.column_stack((
            x - w / 2,
            y - h / 2,
            x + w / 2,
            y + h / 2,
        ))

        return boxes, objectness, classness, combined

    def buildCandidateHeatmap(self,
                              boxes: NDArray,
                              weights: NDArray) -> NDArray:
        """
        Build a confidence heatmap by adding each candidate's weight
        to every pixel contained by its bounding box.
        """

        height, width = self.yoloSize

        heatmap = np.zeros(
            (height, width),
            dtype=np.float32
        )

        for box, weight in zip(boxes, weights):

            x1 = int(np.floor(np.clip(box[0], 0, width)))
            y1 = int(np.floor(np.clip(box[1], 0, height)))
            x2 = int(np.ceil(np.clip(box[2], 0, width)))
            y2 = int(np.ceil(np.clip(box[3], 0, height)))

            if x2 <= x1 or y2 <= y1:
                continue

            heatmap[y1:y2, x1:x2] += weight

        return heatmap

    def findHeatmapPeakCentroid(self,
                                heatmap: NDArray
                                ) -> tuple[float | None,
                                           float | None,
    float,
    int]:
        """
        Find the centroid of the largest connected region attaining
        the maximum heatmap value.

        Returns:
            centroid_x
            centroid_y
            peak_value
            peak_area
        """

        peak_value = float(np.max(heatmap))

        if peak_value <= 0.0:
            return None, None, 0.0, 0

        # The heatmap is piecewise constant, but isclose protects against
        # insignificant floating-point differences.
        peak_mask = np.isclose(
            heatmap,
            peak_value,
            rtol=1e-6,
            atol=1e-8
        ).astype(np.uint8)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            peak_mask,
            connectivity=8
        )

        # label 0 is the background.
        if num_labels <= 1:
            return None, None, peak_value, 0

        component_areas = stats[1:, cv2.CC_STAT_AREA]

        largest_label = 1 + int(np.argmax(component_areas))

        centroid_x, centroid_y = centroids[largest_label]

        peak_area = int(
            stats[largest_label, cv2.CC_STAT_AREA]
        )

        return (
            float(centroid_x),
            float(centroid_y),
            peak_value,
            peak_area
        )

    def set_calibration(self, calibration: Calibration) -> None:
        self.calibration = copy.deepcopy(calibration)

    def preprocessImage(self, image: NDArray) -> NDArray:
        """
        This preprocessing:
            Fixes the image to the YOLO network's size
            Transposes the image so that it matches onnxruntime's input format
            Adds a dimension to match onnxruntime's input format
        :param image: np.array from OpenCV
        :return: preprocessed image
        """
        h, w, _ = image.shape
        if (h, w) != self.yoloSize:
            height, width = self.yoloSize
            if (
                self._resize_buffer is None
                or self._resize_buffer.shape != (height, width, 3)
                or self._resize_buffer.dtype != image.dtype
            ):
                self._resize_buffer = np.empty((height, width, 3), dtype=image.dtype)
            cv2.resize(image, (width, height), dst=self._resize_buffer, interpolation=cv2.INTER_LINEAR)
            image_view = self._resize_buffer
        else:
            height, width = h, w
            image_view = image

        if self._input_tensor is None or self._input_tensor.shape != (1, 3, height, width):
            self._input_tensor = np.empty((1, 3, height, width), dtype=np.float32)

        np.multiply(
            image_view.transpose((2, 0, 1)),
            1.0 / 255.0,
            out=self._input_tensor[0],
            casting="unsafe",
        )
        return self._input_tensor

    def processImage(self, yoloImage: NDArray) -> tuple[list, list, list, list, float]:
        """
        Clears the 'cache' for previous YOLO solutions, then calls the yolo inference method
        :param yoloImage: np.array that has completed preprocessing
        :return: onnxruntime output
        """
        self.boxes, self.scores, self.class_ids = [], [], []
        return self.runOneSession(yoloImage)

    def runOneSession(self, yoloImage: NDArray) -> tuple[list, list, list, list, float]:
        """
        Records time before and after a yolo inference for time differencing. Runs the YOLO session
        :param yoloImage: image that has been through preprocessImage
        :return: outputs from onnxruntime session. Labeled output for clarity.
        """
        startTime = datetime.datetime.now()
        if self.session is not None:
            output = self.session.run(None, {self.input_name: yoloImage})
        else:
            output = None
        endTime = datetime.datetime.now()
        centers, boxes, scores, class_ids = self.interpretOutput(output)
        return centers, boxes, scores, class_ids, (endTime - startTime).total_seconds()

    def interpretOutput(self, output: NDArray):
        """
        Takes outputs from onnxruntime and processes them
        Filters to retain only the highest-confidence detection for each class
        :param output:  onnxruntime session outputs
        :return: cleaner outputs for interpretation
        """

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


if __name__ == '__main__':

    yolo = YOLO(
        conf=0.75,
        iou=0.99,
        yoloSize=(864, 864),
        model_path="C:/repos/jarvis_submodules/camera_calibration_python/YOLOModels/AtterburyFT_25/Cub_ANT/cub_solo_0924",
        numClasses=1
    )

    np.set_printoptions(suppress=True)

    target_class = 0

    # Separate from yolo.conf intentionally.
    # We want to see hypotheses far below the normal acceptance threshold.
    candidate_min_score = 0.45

    # The experiment we originally discussed.
    heatmap_weight = "classness"

    # Other useful comparisons:
    # heatmap_weight = "objectness"
    # heatmap_weight = "combined"

    allImages = glob.glob(
        os.path.join(
            'C:/repos/jarvis_submodules/camera_calibration_python/Images/DissertationDatasets/UAS/25SP/C1c/',
            '*.bmp'
        )
    )

    from support.io.data_processing import natural_sort

    allImages = natural_sort(allImages)

    for imgFP in allImages:

        img = cv2.imread(imgFP)

        if img is None:
            continue

        (
            boxes,
            objectness,
            classness,
            combined,
            infer_time
        ) = yolo.inferClassCandidatesOnImage(
            img,
            target_class=target_class,
            min_score=candidate_min_score
        )

        if heatmap_weight == "classness":
            weights = classness

        elif heatmap_weight == "objectness":
            weights = objectness

        else:
            weights = combined

        heatmap = yolo.buildCandidateHeatmap(
            boxes,
            weights
        )

        # ------------------------------------------------------------
        # Conventional candidate argmax
        # ------------------------------------------------------------

        if len(weights) > 0:

            candidate_argmax_idx = int(np.argmax(weights))

            candidate_argmax_box = boxes[candidate_argmax_idx]
            candidate_argmax_value = float(weights[candidate_argmax_idx])

        else:

            candidate_argmax_idx = None
            candidate_argmax_box = None
            candidate_argmax_value = 0.0

        # ------------------------------------------------------------
        # Heatmap peak-region centroid
        # ------------------------------------------------------------

        (
            heatmap_centroid_x,
            heatmap_centroid_y,
            heatmap_peak_value,
            heatmap_peak_area) = yolo.findHeatmapPeakCentroid(heatmap)

        height, width = yolo.yoloSize

        display_img = cv2.resize(
            img,
            (width, height),
            interpolation=cv2.INTER_LINEAR
        )

        candidate_view = display_img.copy()

        argmax_view = display_img.copy()

        # Show exactly which proposals contributed to the heatmap.
        for box in boxes:

            x1 = int(np.clip(
                np.floor(box[0]),
                0,
                width - 1
            ))

            y1 = int(np.clip(
                np.floor(box[1]),
                0,
                height - 1
            ))

            x2 = int(np.clip(
                np.ceil(box[2]),
                0,
                width - 1
            ))

            y2 = int(np.clip(
                np.ceil(box[3]),
                0,
                height - 1
            ))

            cv2.rectangle(
                candidate_view,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                1
            )

        if candidate_argmax_box is not None:
            x1 = int(np.clip(
                np.floor(candidate_argmax_box[0]),
                0,
                width - 1
            ))

            y1 = int(np.clip(
                np.floor(candidate_argmax_box[1]),
                0,
                height - 1
            ))

            x2 = int(np.clip(
                np.ceil(candidate_argmax_box[2]),
                0,
                width - 1
            ))

            y2 = int(np.clip(
                np.ceil(candidate_argmax_box[3]),
                0,
                height - 1
            ))

            cv2.rectangle(
                argmax_view,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            candidate_center_x = int(round((x1 + x2) / 2))
            candidate_center_y = int(round((y1 + y2) / 2))

            cv2.drawMarker(
                argmax_view,
                (candidate_center_x, candidate_center_y),
                (0, 255, 0),
                cv2.MARKER_CROSS,
                20,
                2
            )

            cv2.putText(
                argmax_view,
                f"candidate argmax = {candidate_argmax_value:.4f}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        # Convert the accumulated floating-point heatmap into something
        # OpenCV can colorize.
        if heatmap.max() > 0:

            heatmap_u8 = np.round(
                255.0 * heatmap / heatmap.max()
            ).astype(np.uint8)

            heatmap_color = cv2.applyColorMap(
                heatmap_u8,
                cv2.COLORMAP_TURBO
            )

            heatmap_view = display_img.copy()

            support_mask = heatmap > 0.0

            heatmap_view[support_mask] = cv2.addWeighted(
                display_img,
                0.55,
                heatmap_color,
                0.45,
                0.0
            )[support_mask]

        else:

            heatmap_view = display_img.copy()

        if heatmap_centroid_x is not None:
            centroid_x = int(round(heatmap_centroid_x))
            centroid_y = int(round(heatmap_centroid_y))

            cv2.drawMarker(
                heatmap_view,
                (centroid_x, centroid_y),
                (255, 255, 255),
                cv2.MARKER_CROSS,
                24,
                2
            )

            cv2.circle(
                heatmap_view,
                (centroid_x, centroid_y),
                8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                heatmap_view,
                f"peak={heatmap_peak_value:.4f}, area={heatmap_peak_area}",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        cv2.putText(
            candidate_view,
            f"class {target_class}: {len(boxes)} raw candidates",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            heatmap_view,
            f"weight={heatmap_weight}, max={heatmap.max():.3f}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow(
            'YOLO raw candidates',
            candidate_view
        )

        cv2.imshow(
            'YOLO candidate argmax',
            argmax_view
        )

        cv2.imshow(
            'YOLO feature heatmap',
            heatmap_view
        )

        centroid_str = (
            f"({heatmap_centroid_x:.1f}, {heatmap_centroid_y:.1f})"
            if heatmap_centroid_x is not None
            else "None"
        )

        print(
            f"{os.path.basename(imgFP)}: "
            f"{len(boxes)} candidates, "
            f"candidate argmax={candidate_argmax_value:.4f}, "
            f"heat peak={heatmap_peak_value:.4f}, "
            f"heat peak area={heatmap_peak_area}, "
            f"heat centroid={centroid_str}, "
            f"inference={infer_time * 1000.0:.2f} ms"
        )

        key = cv2.waitKey(0)

        if key == 121:
            print('you hit yes')
            with open("test.txt", "w") as f:
                f.write("string")

        if key == 110:
            print('you hit no')
            os.remove(imgFP)

        if key == 27:
            break

    cv2.destroyAllWindows()