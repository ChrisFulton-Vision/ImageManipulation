import os
import pickle, copy

import numpy as np
from pathlib import Path
from support.core.sensor_datum_mat4_bundle import parse_sensor_datum_mat4_bundle
from support.io.my_logging import LOG


class TruthPoints:
    def __init__(self):
        self.truthPoints = {}

        self.selectTruthPoints()

        self.saveToCache()

    def selectTruthPoints(self):
        # CALSPAN 2024 Dec AprilTags against LIDAR
        # self.truthPoints['0'] = np.array([5.25967, 2.52404, -1.01714])
        # self.truthPoints['1'] = np.array([2.90982, 2.45965, -0.60117])
        # self.truthPoints['4'] = np.array([1.02611, -0.09072, -1.04306])
        # self.truthPoints['5'] = np.array([1.71335, -0.48700, -0.66432])
        # self.truthPoints['6'] = np.array([0.62843, 1.60848, -0.65310])
        # self.truthPoints['7'] = np.array([3.51041, 3.23543, -0.80128])
        # self.truthPoints['8'] = np.array([4.11462, 3.41798, -0.24962])
        # self.truthPoints['9'] = np.array([-0.37882, -0.33134, -0.64973])
        # self.truthPoints['10'] = np.array([1.38713, 0.46928, -0.63064])
        # self.truthPoints['12'] = np.array([2.72211, 4.00315, -0.59416])
        # self.truthPoints['17'] = np.array([1.35135, -0.99257, -0.41729])
        # self.truthPoints['18'] = np.array([4.00388, 4.02418, 0.08610])
        # self.truthPoints['19'] = np.array([1.15202, 1.01983, -0.72926])
        # self.truthPoints['21'] = np.array([4.09984, 1.55906, -0.58504])
        # self.truthPoints['22'] = np.array([2.52925, 1.73592, -0.97918])
        # self.truthPoints['23'] = np.array([2.15135, 3.04855, -0.73227])

        # MOCAP 2025 Sept AprilTags
        # self.truthPoints['3'] = np.array([-0.12710, -1.21415, -1.23755])  # True location
        # # self.truthPoints['3'] = np.array([-0.12710, -1.21415, 3.0 - 1.23755])  # Modified location to test system
        # self.truthPoints['5'] = np.array([0.70693, -1.90905, -0.70546])
        # self.truthPoints['8'] = np.array([1.38641, 2.16063, -1.22338])
        # self.truthPoints['9'] = np.array([5.35155, -1.39477, 0.29776])
        # self.truthPoints['13'] = np.array([1.98385, -0.65895, -1.24558])
        # self.truthPoints['15'] = np.array([3.14318, 1.31491, -0.78264])
        # self.truthPoints['16'] = np.array([3.12187, 0.52254, -0.65637])
        # self.truthPoints['18'] = np.array([3.49844, 0.51306, -0.23403])
        # self.truthPoints['19'] = np.array([2.07009, -1.72274, -0.69668])
        # self.truthPoints['20'] = np.array([3.92248, -3.10489, 0.24576])
        # self.truthPoints['21'] = np.array([3.12467, -0.07760, -1.24881])
        # self.truthPoints['22'] = np.array([1.67668, -0.00762, -1.70942])
        # self.truthPoints['23'] = np.array([2.16908, -1.33895, -0.46741])

        # MOCAP LIDAR AprilTags
        # self.truthPoints['0'] = np.array([5.999722958, 0.210682318, 0.569373071])
        # self.truthPoints['3'] = np.array([4.268728733, -6.744913101, 0.986035943])
        # self.truthPoints['5'] = np.array([6.382734299, 0.639621198, 1.011341929])
        # self.truthPoints['8'] = np.array([6.157489300, 3.442303658, 0.590105295])
        # self.truthPoints['10'] = np.array([7.166899204, 0.034531891, 0.964832246])
        # self.truthPoints['11'] = np.array([5.398657322, -0.701054692, 1.037473083])
        # self.truthPoints['12'] = np.array([2.359554768, 3.344508171, 1.080431104])
        # self.truthPoints['18'] = np.array([4.460691452, 3.925699234, 0.932579398])
        # self.truthPoints['20'] = np.array([3.291348934, 3.324561834, 0.576795220])

        # MOCAP 2025 Nov AprilTags
        # self.truthPoints['0'] = np.array([7.333877, -4.724778, 0.563943])
        # self.truthPoints['1'] = np.array([7.575994, -2.204574, 1.515839])
        # self.truthPoints['2'] = np.array([7.244235, -1.982850, 1.091029])
        # self.truthPoints['3'] = np.array([4.328803, -1.043142, 0.993910])
        # self.truthPoints['3'] = np.array([4.328803, -1.043142, 10.993910])
        # self.truthPoints['4'] = np.array([6.590099, -5.168346, 0.585101])
        # self.truthPoints['5'] = np.array([1.034546, -2.530706, 0.581975])
        # self.truthPoints['6'] = np.array([5.609220, -1.167198, 0.569148])
        # self.truthPoints['7'] = np.array([7.273934, -2.772839, 0.553683])
        # self.truthPoints['8'] = np.array([3.281607, -1.631069, 0.598682])
        # self.truthPoints['10'] = np.array([3.242499, -5.955892, 0.578241])
        # self.truthPoints['11'] = np.array([1.318924, -6.154001, 1.047968])
        # self.truthPoints['12'] = np.array([2.104257, -1.842161, 1.059387])
        # self.truthPoints['13'] = np.array([4.598224, -5.790008, 0.828719])
        # self.truthPoints['15'] = np.array([])
        # self.truthPoints['16'] = np.array([])
        # self.truthPoints['17'] = np.array([6.844221, -1.158623, 0.970978])
        # self.truthPoints['18'] = np.array([5.479628, -5.628743, 0.920132])
        # self.truthPoints['19'] = np.array([7.138083, -5.202418, 1.440090])
        # self.truthPoints['20'] = np.array([4.234251, -2.057815, 0.576980])
        # self.truthPoints['22'] = np.array([])

        # Seery MOCAP Data
        self.truthPoints['0'] = np.array([2.213529348, -6.921269417, 0.581792355])
        self.truthPoints['1'] = np.array([-0.067431509, -7.154722214, 1.101872683])
        self.truthPoints['2'] = np.array([3.184869766, -7.632071495, 0.568810046])
        self.truthPoints['3'] = np.array([4.279306412, -6.735857964, 0.996919036])
        self.truthPoints['4'] = np.array([-1.547529221, -2.297834635, 0.637015581])
        self.truthPoints['6'] = np.array([5.042785168, -5.044515610, 0.582720816])
        self.truthPoints['7'] = np.array([1.149369478, -5.938117981, 0.583432853])
        self.truthPoints['8'] = np.array([-0.394698590, -6.333474159, 0.609945714])
        self.truthPoints['9'] = np.array([-1.033937335, -6.080481529, 0.602701604])
        self.truthPoints['10'] = np.array([1.423779011, -7.666602612, 0.993166387])
        self.truthPoints['11'] = np.array([4.289189816, -5.560795784, 1.043918014])
        self.truthPoints['12'] = np.array([2.024945736, -8.406939507, 1.078723192])
        self.truthPoints['13'] = np.array([5.471141338, -5.686547756, 0.831703246])
        self.truthPoints['14'] = np.array([6.128262997, -4.250380039, 0.939006746])
        self.truthPoints['15'] = np.array([6.322391510, -4.727521896, 0.614175320])
        self.truthPoints['16'] = np.array([5.656450272, -3.245581150, 0.839761913])
        self.truthPoints['17'] = np.array([5.444821835, -6.409629345, 0.993954897])
        self.truthPoints['18'] = np.array([0.887455463, -8.438985825, 0.944647372])
        self.truthPoints['19'] = np.array([4.030318260, -7.956672192, 0.939799428])
        self.truthPoints['20'] = np.array([0.458850652, -7.742451191, 0.588691354])
        self.truthPoints['23'] = np.array([6.223794937, -3.777098656, 0.596049547])


        # _FLU_TO_CV = np.array([
        #     [0., 1., 0.],
        #     [0., 0., -1.,],
        #     [-1., 0., 0.]
        # ], dtype=float)

        # for key, val in self.truthPoints.items():
        #     self.truthPoints[key] = _FLU_TO_CV @ val

    def getTruthPointsDict(self):
        return self.truthPoints

    def getTruthPointsNumpy(self):
        truthPointsArray = None

        for truthPoint in self.truthPoints.values():
            if truthPointsArray is None:
                truthPointsArray = truthPoint
            else:
                truthPointsArray = np.vstack((truthPointsArray, truthPoint))

        return truthPointsArray

    def saveToCache(self):
        # Store only the data dict, not the whole class instance
        with open('../../ThreeD_Truth_Points.pkl', 'wb') as f:
            pickle.dump(self.truthPoints, f)

    def copy(self, classToCopy):
        self.__dict__.update(copy.deepcopy(classToCopy.__dict__))

    def fromSensorDatumMat4Bundle(self, sensorDatumMat4Bundle: str | os.PathLike[str]):
        self.truthPoints = {}

        mat4_bundle = parse_sensor_datum_mat4_bundle(sensorDatumMat4Bundle).query("object_id.str.startswith('tag_')")

        for tag_id, tag_df in mat4_bundle.groupby('object_id'):
            self.truthPoints[tag_id] = tag_df[["x", "y", "z"]].mean().to_numpy()

    def try_load(self, lidar_path: Path):
        if not lidar_path.exists():
            LOG.error(
                f'Cached LiDAR file not found. Using defaults. Attempted filepath:\n{self.camConfig.lidarFilepath}'
            )
            return

        try:
            with lidar_path.open('rb') as f:
                obj = pickle.load(f)
        except AttributeError as e:
            # Old pickle referring to __main__.TruthPoints or otherwise broken:
            LOG.warning("Failed to unpickle LiDAR truth points (%s). Using defaults instead.", e)
            obj = TruthPoints()  # fall back to code-defined truth points
        except Exception as e:
            LOG.error("Error loading LiDAR truth points: %s", e)
            return

        # Accept either a TruthPoints instance or a raw dict
        if isinstance(obj, TruthPoints):
            self.copy(obj)
        elif isinstance(obj, dict):
            # Existing self.lidarTruthPoints is a TruthPoints()
            self.truthPoints = copy.deepcopy(obj)
        else:
            LOG.error("Unexpected LiDAR truth data type: %r", type(obj))


if __name__ == '__main__':
    TruthPoints()
