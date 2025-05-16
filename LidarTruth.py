import numpy as np
import pickle, copy

class TruthPoints:
    def __init__(self):
        self.truthPoints = {}

        self.selectTruthPoints()

        self.saveToCache()

    def selectTruthPoints(self):
        # 0
        self.truthPoints['0'] = np.array([5.25967, 2.52404, -1.01714])
        # 1
        self.truthPoints['1'] = np.array([2.90982, 2.45965, -0.60117])
        # 4
        self.truthPoints['4'] = np.array([1.02611, -0.09072, -1.04306])
        # 5
        self.truthPoints['5'] = np.array([1.71335, -0.48700, -0.66432])
        # 6
        self.truthPoints['6'] = np.array([0.62843, 1.60848, -0.65310])
        # 7
        self.truthPoints['7'] = np.array([3.51041, 3.23543, -0.80128])
        # 8
        self.truthPoints['8'] = np.array([4.11462, 3.41798, -0.24962])
        # 9
        self.truthPoints['9'] = np.array([-0.37882, -0.33134, -0.64973])
        # 10
        self.truthPoints['10'] = np.array([1.38713, 0.46928, -0.63064])
        # 12
        self.truthPoints['12'] = np.array([2.72211, 4.00315, -0.59416])
        # 17
        self.truthPoints['17'] = np.array([1.35135, -0.99257, -0.41729])
        # 18
        self.truthPoints['18'] = np.array([4.00388, 4.02418, 0.08610])
        # 19
        self.truthPoints['19'] = np.array([1.15202, 1.01983, -0.72926])
        # 21
        self.truthPoints['21'] = np.array([4.09984, 1.55906, -0.58504])
        # 22
        self.truthPoints['22'] = np.array([2.52925, 1.73592, -0.97918])
        # 23
        self.truthPoints['23'] = np.array([2.15135, 3.04855, -0.73227])

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
        with open('LIDAR_Truth_Points.pkl', 'wb') as f:
            pickle.dump(self, f)

    def copy(self, classToCopy):
        self.__dict__.update(copy.deepcopy(classToCopy.__dict__))