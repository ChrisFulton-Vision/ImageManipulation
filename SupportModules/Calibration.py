import os
import pickle as pkl
import re
from datetime import datetime
from os.path import join

import numpy as np


class Calibration:
    def __init__(self, filepath=None):
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.k1 = None
        self.k2 = None
        self.p1 = None
        self.p2 = None
        self.k3 = None
        self.k4 = None
        self.calTime = None
        self.numCBUsed = None
        self.rmsError = None
        self.width = None
        self.height = None
        self.hfov = None
        self.fisheye = None
        self.calDatetime = None

        self.scale = 1.0

        if filepath is not None:
            self.fromFile(filepath)

    def __str__(self):
        return self.calStr

    def setCameraMatrix(self, mtx=None, fx=None, fy=None, cx=None, cy=None):
        if mtx is not None:
            self.fx = mtx[0, 0]
            self.fy = mtx[1, 1]
            self.cx = mtx[0, 2]
            self.cy = mtx[1, 2]
        elif fx is not None and fy is not None and cx is not None and cy is not None:
            self.fx = fx
            self.fy = fy
            self.cx = cx
            self.cy = cy
        else:
            self.fx = None
            self.fy = None
            self.cx = None
            self.cy = None
        self.scale = 1.0

    @property
    def K(self):
        return self.getCameraMatrix()
    @property
    def inv(self):
        return np.array([[1.0/self.fx, 0.0, -self.cx / self.fx],
                         [0.0, 1.0/self.fy, -self.cy / self.fy],
                         [0.0, 0.0, 1.0]])

    def getCameraMatrix(self):
        # Included for backwards compatibility
        if not hasattr(self, "scale"):
            self.scale = 1.0

        if self.fx is not None and self.fy is not None and self.cx is not None and self.cy is not None and self.scale is not None:
            fx = self.scale * self.fx
            fy = self.scale * self.fy
            cx = self.scale * (self.cx + 0.5) - 0.5
            cy = self.scale * (self.cy + 0.5) - 0.5
            return np.array([[fx, 0.0, cx],
                             [0.0, fy, cy],
                             [0.0, 0.0, 1.0]])
        else:
            return None

    def setDistortion(self, dist=None, k1=None, k2=None, p1=None, p2=None, k3=None):
        if dist is not None and self.fisheye:
            self.k1 = dist[0]
            self.k2 = dist[1]
            self.k3 = dist[2]
            self.k4 = dist[3]
        elif dist is not None:
            self.k1 = dist[0]
            self.k2 = dist[1]
            self.p1 = dist[2]
            self.p2 = dist[3]
            self.k3 = dist[4]
        elif k1 is not None and k2 is not None and p1 is not None and p2 is not None and k3 is not None:
            self.k1 = k1
            self.k2 = k2
            self.p1 = p1
            self.p2 = p2
            self.k3 = k3
        else:
            self.k1 = None
            self.k2 = None
            self.p1 = None
            self.p2 = None
            self.k3 = None
            self.k4 = None

    def getDistortion(self):
        if self.fisheye:
            if self.k1 is not None and self.k2 is not None and self.k3 is not None and self.k4 is not None:
                return np.array([self.k1, self.k2, self.k3, self.k4]).flatten()
            else:
                return None

        if self.k1 is not None and self.k2 is not None and self.p1 is not None and self.p2 is not None and self.k3 is not None:
            return np.array([self.k1, self.k2, self.p1, self.p2, self.k3]).flatten()
        else:
            return None

    def setAccessories(self, calTime, numCBUsed, width, height, hfov, rms, timeOfCompute=None):

        self.calTime = calTime
        self.numCBUsed = numCBUsed
        self.width = width
        self.height = height
        self.hfov = hfov
        self.rmsError = rms
        self.calDatetime = timeOfCompute

    @property
    def calStr(self):
        mtx = self.getCameraMatrix()
        dist = self.getDistortion()

        calStr = ''
        if self.validCal:
            calStr += '# Camera matrix'
            if self.calDatetime is not None:
                calStr += ' computed at:\n#{}\n'.format(datetime.date(self.calDatetime))
            calStr += '# Original size: ' + str(int(self.width)) + 'x' + str(int(self.height)) + '\n'
            if self.fisheye:
                calStr += '#Fisheye Cal'
            calStr += '\nfx={:.{}f}'.format(mtx[0, 0], 10) + '\n'
            calStr += 'fy={:.{}f}'.format(mtx[1, 1], 10) + '\n'
            calStr += 'cx={:.{}f}'.format(mtx[0, 2], 10) + '\n'
            calStr += 'cy={:.{}f}'.format(mtx[1, 2], 10) + '\n\n'

            calStr += '#Distortion coefficients\n'
            if self.fisheye:
                calStr += 'k1={:.{}f}'.format(dist[0], 10) + '\n'
                calStr += 'k2={:.{}f}'.format(dist[1], 10) + '\n'
                calStr += 'k3={:.{}f}'.format(dist[2], 10) + '\n'
                calStr += 'k4={:.{}f}'.format(dist[3], 10) + '\n\n'
            else:
                calStr += 'k1={:.{}f}'.format(dist[0], 10) + '\n'
                calStr += 'k2={:.{}f}'.format(dist[1], 10) + '\n'
                calStr += 'p1={:.{}f}'.format(dist[2], 10) + '\n'
                calStr += 'p2={:.{}f}'.format(dist[3], 10) + '\n'
                calStr += 'k3={:.{}f}'.format(dist[4], 10) + '\n\n'

            calStr += '#Total cal time (sec)\n'
            calStr += 'ct={:.{}f}'.format(self.calTime, 10) + '\n\n'

            calStr += '#Chessboards used\n'
            calStr += 'total=' + str(self.numCBUsed) + '\n'
            calStr += 'valid=' + str(self.numCBUsed) + '\n'
            calStr += 'rmsErr=' + str(self.rmsError) + '\n\n'

            calStr += '#Other\n'
            calStr += 'resolution=' + str(int(self.scale * self.width)) + 'x' + str(int(self.scale*self.height)) + '\n'
            calStr += 'hfov=' + str(self.hfov) + "\n"
        else:
            calStr = 'Bad Cal'

        return calStr

    def copy(self, calToCopy):
        self.__init__()
        for obj in calToCopy.__dict__:
            try:
                self.__dict__[obj] = calToCopy.__dict__[obj]
            except KeyError:
                # Allows for versioning issues, changed naming conventions.
                print("Older version...")
                self.calDatetime = None

    def toBinFile(self, fileDirectory):
        with open(join(fileDirectory,'calibration.pkl'), 'wb') as file:
            pkl.dump(self, file)

    def fromBinFile(self, fileDirectory):
        if os.path.exists(join(fileDirectory, 'calibration.pkl')):
            with open(join(fileDirectory, 'calibration.pkl'), 'rb') as file:
                self.copy(pkl.load(file))
                return True
        if os.path.basename(fileDirectory) == 'calibration.pkl':
            with open(fileDirectory, 'rb') as file:
                self.copy(pkl.load(file))
                return True
        return False

    def toFile(self, fileDirectory):
        with open(join(fileDirectory, 'calibration.txt'), 'w') as file:
            file.write(self.calStr)

    def fromFile(self, fileDirectory):
        if os.path.exists(join(fileDirectory, 'calibration.txt')):
            filepath = join(fileDirectory, 'calibration.txt')
        elif os.path.exists(fileDirectory) and fileDirectory[-4:] == '.txt':
            filepath = fileDirectory
        else:
            return False

        self.__init__()

        if os.path.exists(filepath):
            with open(filepath, 'r') as file:
                for line in file:
                    match line[0:2]:
                        case '#2':
                            self.calDatetime = datetime.strptime(line[1:-1], '%Y-%m-%d')
                        case '#F':
                            self.fisheye = True
                        case 'fx':
                            self.fx = float(line[3:])
                        case 'fy':
                            self.fy = float(line[3:])
                        case 'cx':
                            self.cx = float(line[3:])
                        case 'cy':
                            self.cy = float(line[3:])
                        case 'k1':
                            self.k1 = float(line[3:])
                        case 'k2':
                            self.k2 = float(line[3:])
                        case 'k3':
                            self.k3 = float(line[3:])
                        case 'k4':
                            self.k4 = float(line[3:])
                        case 'p1':
                            self.p1 = float(line[3:])
                        case 'p2':
                            self.p2 = float(line[3:])
                        case 'ct':
                            self.calTime = float(line[3:])
                        case 'to':
                            self.numCBUsed = float(line[6:])
                        case 'rm':
                            self.rmsError = float(line[7:])
                        case 're':
                            intList = re.findall(r'\d+', line)
                            self.width = int(intList[0])
                            self.height = int(intList[1])
                        case 'hf':
                            self.hfov = float(line[7:])
        if self.validCal:
            return True
        else:
            return False

    @property
    def validCal(self):
        if self.fisheye and any([self.fx is None,
                self.fy is None,
                self.cx is None,
                self.cy is None,
                self.k1 is None,
                self.k2 is None,
                self.k3 is None,
                self.k4 is None,
                self.calTime is None,
                self.numCBUsed is None,
                self.rmsError is None,
                self.width is None,
                self.height is None,
                self.hfov is None]):
            return False
        elif not self.fisheye and any([self.fx is None,
                self.fy is None,
                self.cx is None,
                self.cy is None,
                self.k1 is None,
                self.k2 is None,
                self.p1 is None,
                self.p2 is None,
                self.k3 is None,
                self.calTime is None,
                self.numCBUsed is None,
                self.rmsError is None,
                self.width is None,
                self.height is None,
                self.hfov is None]):
            return False
        return True

    def scaleCalibration(self, newWidth: int):

        if not self.validCal:
            raise ValueError('Invalid Calibration. Missing necessary parameter.')

        self.scale = newWidth / self.width
