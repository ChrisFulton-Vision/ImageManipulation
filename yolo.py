import cv2
import numpy as np
# import torch
import onnxruntime as ort
import os
import glob
import datetime
import threading
import copy
import sys
from metaYoloReader import MetaYoloReader


# ort.preload_dlls()
ort.preload_dlls(cuda=True, cudnn=True, msvc=True, directory=None)


class YOLO:
    def __init__(self, conf: float = 0.65, iou: float = 0.99, yoloSize=(864, 864),
                 model_path="YOLOModels/PROBE_BothContext_01062024_0_3M/",
                              # "YOLOModels/PROBE_BothContext_01062024_0_3M/PROBEic.onnx",
                              # "YOLOModels/Aligned_Drogue_65_inandoutofcontext/AlignedDrogue_inandoutofcontext.onnx"],
                 # model_paths=['YOLOModels/cats_v_dogs.onnx'],
                 numClasses: int = 94):
        self.conf = conf
        self.iou = iou

        model = glob.glob(os.path.join(model_path, f'*.onnx'))
        metaYolo = glob.glob(os.path.join(model_path, f'*.csv'))
        self.modelPath = model[0]
        self.reader = MetaYoloReader(metaYolo[0])

        self.provider = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.output = []
        self.boxes, self.scores, self.class_ids = [], [], []
        self.session = None
        self.reinitSession()

        self.class_names = range(numClasses)
        self.yoloSize = yoloSize

    def updateModelPath(self, newPath: str):
        self.modelPath = newPath
        self.reinitSession()

    def reinitSession(self):
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        # sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.add_session_config_entry("session.intra_op.allow_spinning", "1")
        self.session = ort.InferenceSession(self.modelPath, sess_options=sess_options, providers=self.provider)

    def inferOnImage(self, image):
        yoloImage = self.preprocessImage(image)
        output = self.processImage(yoloImage)
        return self.markUpImage(image, output), output

    def preprocessImage(self, image):
        h, w, _ = image.shape
        if (h, w) != self.yoloSize:
            image = cv2.resize(image, self.yoloSize)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32) / 255.0
        return image

    def processImage(self, yoloImage):
        self.boxes, self.scores, self.class_ids = [], [], []
        output = self.runOneSession(yoloImage)
        return output

    def runOneSession(self, yoloImage):
        startTime = datetime.datetime.now()
        output = self.session.run(None, {self.session.get_inputs()[0].name: yoloImage})
        endTime = datetime.datetime.now()
        centers, boxes, scores, class_ids = self.interpretOutput(output)
        return centers, boxes, scores, class_ids, (endTime - startTime).total_seconds()

    def interpretOutput(self, output):

        centers, boxes, scores, class_ids = [], [], [], []
        predictions = np.squeeze(output[0])

        for idx, detection in enumerate(predictions):

            x, y, w_box, h_box, confidence = detection[:5]
            class_probs = detection[5:]

            if confidence > self.conf:
                x1 = int(x - w_box / 2)
                y1 = int(y - h_box / 2)
                x2 = int(x + w_box / 2)
                y2 = int(y + h_box / 2)

                if x1 > 10 and x2 < self.yoloSize[0] - 10 and y1 > 10 and y2 < self.yoloSize[1] - 10:
                    centers.append([x,y])
                    boxes.append([x1, y1, x2, y2])
                    scores.append(float(confidence))
                    class_ids.append(np.argmax(class_probs))

        return centers, boxes, scores, class_ids

    def markUpImage(self, image, output):
        h, w, _ = image.shape

        centers, boxes, scores, class_ids, time = output

        color = (255, 255, 0)

        text = f'Inference time: {time:.3f}s'
        cv2.putText(image, text, (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 3)

        if len(class_ids) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf, self.iou)
            newCenters, newBoxes, newClass_ids, newScores = [],[],[],[]
            for i in indices:
                newCenters.append(centers[i])
                newBoxes.append(boxes[i])
                newClass_ids.append(class_ids[i])
                newScores.append(scores[i])

            image = self.drawBoxes(image, newCenters, newBoxes, newClass_ids, newScores, color)
            if len(indices) > 5:
                self.drawPnP(image, newClass_ids, newCenters)

        return image

    def drawBoxes(self, image, newCenters, newBoxes, newClass_ids, newScores, color):
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
            # cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
            # cv2.putText(image, f"{score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            cv2.putText(image, f"{class_id}", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        return image

    def drawPnP(self, image, y_class_ids, y_centers):

        h, w, _ = image.shape
        y_h, y_w = self.yoloSize

        scale = 864.0 / 1424.0
        calibration = np.array(
            [[scale * 1548.72, 0, scale * (911.2923 + 0.5) - 0.5], [0.0, scale * 1550.61, scale * (828.34 + 0.5) - 0.5],
             [0.0, 0.0, 1.0]])
        np.set_printoptions(suppress=True, precision=4)
        object_points = []
        image_points = []
        badList = [63, 72, 19, 92, 78, 77, 12, 22]
        for idx, y_class_id in enumerate(y_class_ids):
            if y_class_id not in badList:
                x, y, z = self.reader.idsNamesLocs[y_class_id][2:]
                object_points.append([x, y, z])
                image_points.append(y_centers[idx])

        object_points = np.array(object_points)
        image_points = np.array(image_points)

        ret, rvec, tvec, inliers = cv2.solvePnPRansac(objectPoints=object_points,
                                       imagePoints=image_points,
                                       cameraMatrix=calibration,
                                       distCoeffs=np.zeros((5,)))
                                       # flags=cv2.SOLVEPNP_ITERATIVE)

        for y_class_id in y_class_ids:
            # for idNameLoc in reader.idsNamesLocs:
            id = self.reader.idsNamesLocs[y_class_id][0]
            xyz = np.array(self.reader.idsNamesLocs[y_class_id][2:])
            projectedPixel, _ = cv2.projectPoints(xyz, rvec=rvec, tvec=tvec,
                                                  cameraMatrix=calibration, distCoeffs=np.zeros((5,)))
            x, y = np.squeeze(projectedPixel)
            x = int(w / y_w * x)
            y = int(h / y_h * y)
            cv2.putText(image, str(id), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (50, 255, 255), 1)



if __name__ == '__main__':
    yolo = YOLO()

    scale = 864.0 / 1424.0
    calibration = np.array([[scale * 1548.7213762786, 0, scale * (911.2923662427 + 0.5) - 0.5],[0.0, scale * 1550.6128070942, scale * (828.3494658126 + 0.5) - 0.5],[0.0, 0.0, 1.0]])
    np.set_printoptions(suppress=True)

    # testImage = cv2.imread('BoundingBoxCandidates/13608.bmp')
    # testImage, sol = yolo.inferOnImage(testImage)

    allImages = glob.glob(os.path.join('BoundingBoxCandidates', f'*.bmp'))

    for imgFP in allImages:
        newImg, sol = yolo.inferOnImage(cv2.imread(imgFP))
        cv2.imshow('YOLO', newImg)
        cv2.imwrite('BoundingBoxCandidates/SaveFiles/' + os.path.basename(imgFP), newImg)
        cv2.waitKey(1)


    #
    # y_centers, boxes, scores, y_class_ids, time = sol
    # # print(reader.idsNamesLocs)
    # # t_class_ids, oid, xs, ys, zs = reader.idsNamesLocs
    # object_points = []
    # for y_class_id in y_class_ids:
    #     x,y,z = reader.idsNamesLocs[y_class_id][2:]
    #     object_points.append([x, y, z])
    # object_points = np.array(object_points)
    # y_centers = np.array(y_centers)
    #
    # ret, rvec, tvec = cv2.solvePnP(objectPoints=object_points,
    #                                imagePoints=y_centers,
    #                                cameraMatrix=calibration,
    #                                distCoeffs=np.zeros((5,)),
    #                                flags=cv2.SOLVEPNP_ITERATIVE)
    #
    # for y_class_id in y_class_ids:
    # # for idNameLoc in reader.idsNamesLocs:
    #     id = reader.idsNamesLocs[y_class_id][0]
    #     xyz = np.array(reader.idsNamesLocs[y_class_id][2:])
    #     projectedPixel, _ = cv2.projectPoints(xyz, rvec=rvec, tvec=tvec,
    #                                           cameraMatrix=calibration, distCoeffs=np.zeros((5,)))
    #     cv2.putText(testImage, str(id), tuple(np.squeeze(projectedPixel).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)

    # cv2.imshow('Test', testImage)
    # cv2.waitKey(0)
