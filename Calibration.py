import os, re, copy
import numpy as np
import pickle as pkl

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
        self.calTime = None
        self.numCBUsed = None
        self.rmsError = None
        self.width = None
        self.height = None
        self.hfov = None

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

    def getCameraMatrix(self):
        if self.fx is not None and self.fy is not None and self.cx is not None and self.cy is not None:
            return np.array([[self.fx, 0.0, self.cx],
                             [0.0, self.fy, self.cy],
                             [0.0, 0.0, 1.0]])
        else:
            return None

    def setDistortion(self, dist=None, k1=None, k2=None, p1=None, p2=None, k3=None):
        if dist is not None:
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

    def getDistortion(self):
        if self.k1 is not None and self.k2 is not None and self.p1 is not None and self.p2 is not None and self.k3 is not None:
            return np.array([self.k1, self.k2, self.p1, self.p2, self.k3]).flatten()
        else:
            return None

    def setAccessories(self, calTime, numCBUsed, width, height, hfov, rms):

        self.calTime = calTime
        self.numCBUsed = numCBUsed
        self.width = width
        self.height = height
        self.hfov = hfov
        self.rmsError = rms

    @property
    def calStr(self):
        mtx = self.getCameraMatrix()
        dist = self.getDistortion()

        calStr = ''
        if mtx is not None and dist is not None and self.calTime is not None and self.numCBUsed is not None:
            calStr += '#Camera matrix\n'
            calStr += 'fx={:.{}f}'.format(mtx[0, 0], 10) + '\n'
            calStr += 'fy={:.{}f}'.format(mtx[1, 1], 10) + '\n'
            calStr += 'cx={:.{}f}'.format(mtx[0, 2], 10) + '\n'
            calStr += 'cy={:.{}f}'.format(mtx[1, 2], 10) + '\n\n'

            calStr += '#Distortion coefficients\n'
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
            calStr += 'resolution=' + str(self.width) + 'x' + str(self.height) + '\n'
            calStr += 'hfov=' + str(self.hfov) + "\n"
        else:
            calStr = 'Bad Cal'
        return calStr

    def copy(self, calToCopy):
        self.__dict__.update(copy.deepcopy(calToCopy.__dict__))

    def toBinFile(self, fileDirectory):
        with open(fileDirectory + '\\calibration.bin', 'wb') as file:
            pkl.dump(self, file)

    def fromBinFile(self, fileDirectory):
        if os.path.exists(os.path.join(fileDirectory, 'calibration.pkl')):
            with open(os.path.join(fileDirectory, 'calibration.bin'), 'rb') as file:
                self.copy(pkl.load(file))
                return True
        if os.path.basename(fileDirectory) == 'calibration.pkl':
            with open(fileDirectory, 'rb') as file:
                self.copy(pkl.load(file))
                return True
        return False

    def toFile(self, fileDirectory):
        with open(fileDirectory + '\\calibration.txt', 'w') as file:
            file.write(self.calStr)

    def fromFile(self, fileDirectory):
        if os.path.exists(os.path.join(fileDirectory, 'calibration.txt')):
            filepath = os.path.join(fileDirectory, 'calibration.txt')
        elif os.path.exists(fileDirectory) and fileDirectory[-4:] == '.txt':
            filepath = fileDirectory
        else:
            return False

        self.__init__()

        if os.path.exists(filepath):
            with open(filepath, 'r') as file:
                for line in file:
                    match line[0:2]:
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
        if any([self.fx is None,
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
        else:
            return True

    def scaleCalibration(self, newWidth:int):
        if not self.validCal:
            raise ValueError('Invalid Calibration. Missing necessary parameter.')

        scale = float(newWidth / self.width)
        self.fx *= scale
        self.fy *= scale
        self.cx = scale * (self.cx + 0.5) - 0.5
        self.cy = scale * (self.cy + 0.5) - 0.5


        self.height = int(self.height*scale)
        self.width = newWidth
