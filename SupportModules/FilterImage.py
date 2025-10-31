from enum import Enum
import cv2
import numpy as np
from numpy import rad2deg
from numpy.typing import NDArray
from typing import Self

class Gabor:
    def __init__(self):
        self.ksize = (31, 31)
        self.sigma = 3.0
        self.theta = 0.0
        self.lambd = 10.0
        self.gamma = 0.63
        self.psi = 0.0

    def update_sigma(self, new_sigma: float):
        self.sigma = new_sigma

    def update_theta(self, new_theta: float):
        self.theta = new_theta

    def update_lambd(self, new_lambd: float):
        self.lambd = new_lambd

    def update_gamma(self, new_gamma: float):
        self.gamma = new_gamma

    def update_psi(self, new_psi: float):
        self.psi = new_psi

    def filter_kernel(self):
        return cv2.getGaborKernel(self.ksize,
                                  self.sigma,
                                  self.theta,
                                  self.lambd,
                                  self.gamma,
                                  self.psi)

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

    @staticmethod
    def get_convolution(imageKernel: Self):
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


def applyConvolutionFilter(img: NDArray, kernel: ImageKernel, gabor: None | Gabor = None) -> NDArray:
    if kernel == ImageKernel.Unsharp:
        gaussian_3 = cv2.GaussianBlur(img, (0, 0), 2.0)
        return cv2.addWeighted(img, 2.0, gaussian_3, -1.0, 0)

    if kernel == ImageKernel.Gabor:
        convolution = gabor.filter_kernel()
        return cv2.filter2D(img, -1, convolution)

    convolution = ImageKernel.get_convolution(kernel)
    return cv2.filter2D(img, -1, convolution)

