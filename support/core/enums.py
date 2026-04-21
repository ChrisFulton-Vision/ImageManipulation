from enum import Enum
import numpy as np
from itertools import cycle

class ExportQuality(Enum):
    low_quality = 'Low Quality'
    med_quality = 'Medium Quality'
    hgh_quality = 'High Quality'


class ImageKernel(Enum):
    Unfiltered = 'Unfiltered'  #None
    Sharpen = 'Sharpen'  #np.array([[0, -1, 0],[-1, 5, -1], [0, -1, 0]])
    GaussBlur = 'GaussBlur'  #np.array([[1, 4, 6, 4, 1],[4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]]) / 256.0
    EdgeDetect = 'EdgeDetect'  #np.array([[-1, -1, -1],[-1, 8, -1], [-1, -1, -1]])
    HorizontalEdgeDetect = 'HorizontalEdgeDetect'
    VerticalEdgeDetect = 'VerticalEdgeDetect'
    BoxBlur = 'BoxBlur'
    SobelEdgeDetectHorizontal = 'SobelEdgeDetectHorizontal'
    SobelEdgeDetectVertical = 'SobelEdgeDetectVertical'
    LaplaceEdgeDetect = 'LaplaceEdgeDetect'
    Gabor = 'Gabor'
    ScharrEdgeDetectHorizontal = 'ScharrEdgeDetectHorizontal'
    ScharrEdgeDetectVertical = 'ScharrEdgeDetectVertical'
    Unsharp = 'Unsharp'
    Invert = 'Invert'
    Gain = 'Gain'
    Brightness = 'Brightness'

    @staticmethod
    def get_convolution(imageKernel):
        kernel = None
        match imageKernel:
            case ImageKernel.Sharpen:
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            case ImageKernel.GaussBlur:
                kernel = np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4],
                                   [1, 4, 6, 4, 1]]) / 256.0
            case ImageKernel.EdgeDetect:
                kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            case ImageKernel.HorizontalEdgeDetect:
                kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
            case ImageKernel.VerticalEdgeDetect:
                kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            case ImageKernel.BoxBlur:
                kernel = np.ones((5, 5)) / 25.0
            case ImageKernel.SobelEdgeDetectHorizontal:
                kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            case ImageKernel.SobelEdgeDetectVertical:
                kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            case ImageKernel.LaplaceEdgeDetect:
                kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
            case ImageKernel.ScharrEdgeDetectHorizontal:
                kernel = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]])
            case ImageKernel.ScharrEdgeDetectVertical:
                kernel = np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]])
            case _:
                return None
        return kernel

class PlaybackSpeed(Enum):
    Fixed_fps = 'fixed_fps'
    Real_time = 'realtime'

    def next(self):
        iterator = cycle(self.__class__)
        for member in iterator:
            if member is self:
                return next(iterator)

class ImageSource(Enum):
    Camera_Stream = 'Camera Stream'
    Static_Image = 'Static Image'
    Stream_from_Folder = 'Stream from Folder'

class robust_cost(Enum):
    none = None,
    huber = 'huber',
    cauchy = 'cauchy',
    tukey = 'tukey'

    def next(self):
        iterator = cycle(self.__class__)
        for member in iterator:
            if member is self:
                return next(iterator)

    def val(self):
        if self == robust_cost.huber:
            return 1
        if self == robust_cost.cauchy:
            return 2
        if self == robust_cost.tukey:
            return 3
        return 0