from __future__ import annotations

import cv2
from numpy import deg2rad
from numpy.typing import NDArray

from support.core.enums import ImageKernel


class Gabor:
    """GUI-independent Gabor kernel parameters.

    The application now supplies these values from the image-processing queue.
    This class remains available for non-GUI callers and compatibility with
    older code that retained a parameter object between frames.
    """

    def __init__(
        self,
        *,
        ksize: tuple[int, int] = (31, 31),
        sigma: float = 3.0,
        theta: float = 0.0,
        lambd: float = 10.0,
        gamma: float = 0.63,
        psi: float = 0.0,
    ) -> None:
        self.ksize = ksize
        self.sigma = float(sigma)
        self.theta = float(theta)
        self.lambd = float(lambd)
        self.gamma = float(gamma)
        self.psi = float(psi)

    def update_sigma(self, new_sigma: float) -> None:
        self.sigma = float(new_sigma)

    def update_theta(self, new_theta: float) -> None:
        self.theta = float(new_theta)

    def update_lambd(self, new_lambd: float) -> None:
        self.lambd = float(new_lambd)

    def update_gamma(self, new_gamma: float) -> None:
        self.gamma = float(new_gamma)

    def update_psi(self, new_psi: float) -> None:
        self.psi = float(new_psi)

    def filter_kernel(self):
        return cv2.getGaborKernel(
            self.ksize,
            self.sigma,
            self.theta,
            self.lambd,
            self.gamma,
            self.psi,
        )


class GaborGUI(Gabor):
    """Deprecated compatibility shim; Gabor controls now live in the queue."""

    pop_up = None

    def close(self) -> None:
        pass


def ensure_gabor_gui(gabor_gui: GaborGUI | None) -> GaborGUI:
    """Compatibility helper returning a parameter object with no popup."""
    return gabor_gui if gabor_gui is not None else GaborGUI()


def gabor_kernel(
    *,
    sigma: float = 3.0,
    theta_degrees: float = 0.0,
    wavelength: float = 10.0,
    gamma: float = 0.63,
    psi_degrees: float = 0.0,
    ksize: tuple[int, int] = (31, 31),
):
    """Build a Gabor kernel from the values exposed by the queue editor."""
    sigma = max(0.01, float(sigma))
    wavelength = max(0.01, float(wavelength))
    gamma = max(0.0, float(gamma))
    return cv2.getGaborKernel(
        ksize,
        sigma,
        float(deg2rad(theta_degrees)),
        wavelength,
        gamma,
        float(deg2rad(psi_degrees)),
    )


def _apply_convolution_filter(
    img: NDArray,
    kernel: ImageKernel,
    *,
    gabor: GaborGUI | Gabor | None = None,
    gain: float = 1.0,
    brightness: int = 0,
    gabor_sigma: float = 3.0,
    gabor_theta_deg: float = 0.0,
    gabor_lambda: float = 10.0,
    gabor_gamma: float = 0.63,
    gabor_psi_deg: float = 0.0,
) -> None:
    match kernel:
        case ImageKernel.Unfiltered:
            return

        case ImageKernel.Unsharp:
            gaussian_3 = cv2.GaussianBlur(img, (0, 0), 2.0)
            cv2.addWeighted(img, 2.0, gaussian_3, -1.0, 0, dst=img)

        case ImageKernel.Gabor:
            convolution = (
                gabor.filter_kernel()
                if gabor is not None
                else gabor_kernel(
                    sigma=gabor_sigma,
                    theta_degrees=gabor_theta_deg,
                    wavelength=gabor_lambda,
                    gamma=gabor_gamma,
                    psi_degrees=gabor_psi_deg,
                )
            )
            cv2.filter2D(img, -1, convolution, dst=img)

        case ImageKernel.Invert:
            cv2.bitwise_not(img, dst=img)

        case ImageKernel.Gain:
            cv2.convertScaleAbs(img, alpha=gain, beta=0.0, dst=img)

        case ImageKernel.Brightness:
            cv2.convertScaleAbs(img, alpha=1.0, beta=float(brightness), dst=img)

        case _:
            cv2.filter2D(img, -1, ImageKernel.get_convolution(kernel), dst=img)


def apply_filter(
    img: NDArray,
    kernel: ImageKernel,
    gabor_gui: GaborGUI | Gabor | None = None,
    gain: float = 1.0,
    brightness: int = 0,
    *,
    gabor_sigma: float = 3.0,
    gabor_theta_deg: float = 0.0,
    gabor_lambda: float = 10.0,
    gabor_gamma: float = 0.63,
    gabor_psi_deg: float = 0.0,
) -> GaborGUI | Gabor | None:
    """Apply an image filter using queue-provided Gabor parameters.

    ``gabor_gui`` is accepted only for compatibility with older callers.  It
    is a parameter object now and never creates a window.
    """
    if not isinstance(kernel, ImageKernel):
        raise ValueError(
            f"Image Kernel should be ImageKernel Enum class, but is instead {type(kernel)}"
        )

    if kernel == ImageKernel.Greyscale:
        if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR, dst=img)
        return None

    _apply_convolution_filter(
        img,
        kernel,
        gabor=gabor_gui,
        gain=gain,
        brightness=brightness,
        gabor_sigma=gabor_sigma,
        gabor_theta_deg=gabor_theta_deg,
        gabor_lambda=gabor_lambda,
        gabor_gamma=gabor_gamma,
        gabor_psi_deg=gabor_psi_deg,
    )
    return gabor_gui if kernel == ImageKernel.Gabor else None
