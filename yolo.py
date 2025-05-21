import cv2
import numpy as np
import onnxruntime as ort
import os, glob, re, datetime
from metaYoloReader import MetaYoloReader

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

        self.class_names = range(numClasses)
        self.yoloSize = yoloSize

        self.setNewFolder(model_path)

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
        # sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # sess_options.add_session_config_entry("session.intra_op.allow_spinning", "1")
        self.session = ort.InferenceSession(self.modelPath, sess_options=sess_options, providers=self.provider)

    def inferOnImage(self, image: np.array, markup_image:np.array) -> (np.array, np.array):
        '''
        Runs the sub-methods necessary to process an image with YOLO
        :param image: np.array from OpenCV
        :return: Marked-up image post-yolo inference
        '''
        yoloImage = self.preprocessImage(image)
        output = self.processImage(yoloImage)
        return self.markUpImage(markup_image, output), output

    def preprocessImage(self, image: np.array) -> np.array:
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
            image = cv2.resize(image, self.yoloSize)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32) / 255.0
        return image

    def processImage(self, yoloImage: np.array) -> np.array:
        '''
        Clears the 'cache' for previous YOLO solutions, then calls the yolo inference method
        :param yoloImage: np.array that has completed preprocessing
        :return: onnxruntime output
        '''
        self.boxes, self.scores, self.class_ids = [], [], []
        return self.runOneSession(yoloImage)

    def runOneSession(self, yoloImage: np.array) -> np.array:
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

    def interpretOutput(self, output: np.array) -> (list, list, list, list):
        '''
        Takes outputs from onnxruntime and processes them
        :param output:  onnxruntime session outputs
        :return: cleaner outputs for interpretation
        '''

        centers, boxes, scores, class_ids = [], [], [], []
        if output is not None:
            predictions = np.squeeze(output[0])
        else:
            predictions = []

        for idx, detection in enumerate(predictions):
            x, y, w_box, h_box, confidence = detection[:5]
            class_probs = detection[5:]

            if confidence > self.conf:
                x1 = (x - w_box / 2)
                y1 = (y - h_box / 2)
                x2 = (x + w_box / 2)
                y2 = (y + h_box / 2)

                # if (x1 > self.pixel_buffer and x2 < self.yoloSize[0] - self.pixel_buffer
                #         and y1 > self.pixel_buffer and y2 < self.yoloSize[1] - self.pixel_buffer):

                centers.append([x, y])
                boxes.append([x1, y1, x2, y2])
                scores.append(float(confidence))
                class_ids.append(np.argmax(class_probs))

        return centers, boxes, scores, class_ids

    def markUpImage(self, image: np.array, output: (list, list, list, list)) -> np.array:
        '''
        Takes image and places bounding boxes on them. If there's more than 5 features, attempts to solvePnP and mark
        up the image with a PnP solution as well.
        :param image: Original OpenCV style np.array
        :param output: processed onnxruntime sessions
        :return: marked-up image
        '''
        h, w, _ = image.shape

        centers, boxes, scores, class_ids, time = output
        color = (255, 255, 0)

        text = f'Inference time: {time:.3f}s'
        cv2.putText(image, text, (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 3)

        if len(class_ids) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf, self.iou)
            newCenters, newBoxes, newClass_ids, newScores = [], [], [], []
            for i in indices:
            # for i in range(len(centers)):
                newCenters.append(centers[i])
                newBoxes.append(boxes[i])
                newClass_ids.append(class_ids[i])
                newScores.append(scores[i])

            image = self.drawBoxes(image, newCenters, newBoxes, newClass_ids, newScores, color)
            if len(set(indices)) > 5:
                self.drawPnP(image, newClass_ids, newCenters)

        return image

    def drawBoxes(self, image: np.array, newCenters: list, newBoxes: list,
                  newClass_ids: list, newScores: list, color: (int, int, int)) -> np.array:
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

            label = f"{class_id}: {score:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
            # cv2.putText(image, f"{score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            cv2.putText(image, f"{class_id}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            cv2.putText(image, 'Direct Inference', (25, w - 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, color, 1)

        return image

    def drawPnP(self, image:np.array, y_class_ids:list, y_centers:list)-> None:
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

        scale = 864.0 / 1424.0
        calibration = np.array(
            [[scale * 1548.72, 0, scale * (911.2923 + 0.5) - 0.5], [0.0, scale * 1550.61, scale * (828.34 + 0.5) - 0.5],
             [0.0, 0.0, 1.0]])
        np.set_printoptions(suppress=True, precision=4)
        object_points = []
        image_points = []
        badList = []
        for idx, y_class_id in enumerate(y_class_ids):
            if y_class_id not in badList and y_class_id < len(self.reader.idsNamesLocs):
                x, y, z = self.reader.idsNamesLocs[y_class_id][2:]
                object_points.append([x, y, z])
                image_points.append(y_centers[idx])
            else:
                print(f'Future Debug here:')

        object_points = np.array(object_points)
        image_points = np.array(image_points)

        if len(object_points) < 6:
            return

        ret, rvec, tvec, inliers = cv2.solvePnPRansac(objectPoints=object_points,
                                                      imagePoints=image_points,
                                                      cameraMatrix=calibration,
                                                      distCoeffs=np.zeros((5,)))
        # flags=cv2.SOLVEPNP_ITERATIVE)

        for y_class_id in y_class_ids:
            if y_class_id <= len(self.reader.idsNamesLocs):
                # for idNameLoc in reader.idsNamesLocs:
                id = self.reader.idsNamesLocs[y_class_id][0]
                xyz = np.array(self.reader.idsNamesLocs[y_class_id][2:])
                projectedPixel, _ = cv2.projectPoints(xyz, rvec=rvec, tvec=tvec,
                                                      cameraMatrix=calibration, distCoeffs=np.zeros((5,)))
                x, y = np.squeeze(projectedPixel)
                if np.isnan(x) or np.isnan(y):
                    return
                x = int(w / y_w * x)
                y = int(h / y_h * y)
                cv2.putText(image, str(id), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (50, 255, 255), 1)

                cv2.putText(image, 'SolvePnP Solution', (25, w - 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (50, 255, 255), 1)

                # image[0:self.pixel_buffer, :] = np.array([0, 0, 0.0])
                # image[h - self.pixel_buffer:h, :] = np.array([0, 0, 0.0])
                # image[:, 0:self.pixel_buffer] = np.array([0, 0, 0.0])
                # image[:, w - self.pixel_buffer:w] = np.array([0, 0, 0.0])


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

if __name__ == '__main__':
    yolo = YOLO(conf = 0.75, iou = 0.99, yoloSize=(864, 864),
                 model_path="C:/repos/aburn/usr/hub/palindrome_playground/src/sn_UAS_Guidance/YOLO Models/Atterbury_Cub",
                 numClasses = 1)

    np.set_printoptions(suppress=True)

    # testImage = cv2.imread('BoundingBoxCandidates/13608.bmp')
    # testImage, sol = yolo.inferOnImage(testImage)

    allImages = glob.glob(os.path.join('C:/Users/fulto/Desktop/UAS Flight Test/25_Spring/__Flight 2_25_05_19', f'*.bmp'))

    allImages = natural_sort(allImages)

    for imgFP in allImages:
        newImg, sol = yolo.inferOnImage(cv2.imread(imgFP))
        cv2.imshow('YOLO', newImg)
        # cv2.imwrite('BoundingBoxCandidates/SaveFiles/' + os.path.basename(imgFP), newImg)
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
