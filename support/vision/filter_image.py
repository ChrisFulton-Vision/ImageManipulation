import cv2
from numpy import rad2deg, deg2rad
from numpy.typing import NDArray
import customtkinter as ctk
from support.core.enums import ImageKernel


class GaborGUI:
    def __init__(self):
        self.pop_up = ctk.CTkToplevel()
        self.pop_up.title('Gabor Controls')
        self.pop_up.lift()
        self.gaborFilter = Gabor()

        self.sigma_label = None
        self.theta_label = None
        self.lambd_label = None
        self.gamma_label = None
        self.psi_label = None

        self.pop_up.geometry('200x350')
        self.pop_up.grid_columnconfigure([0, 1], weight=1)
        self.configure_pop_up()

    def configure_pop_up(self):
        rowID = 0
        self.sigma_label = ctk.CTkLabel(self.pop_up, text=f'Sigma: {self.gaborFilter.sigma:.2f}')
        self.sigma_label.grid(row=rowID, column=0, padx=5, pady=5, sticky='nsew')
        rowID += 1
        sigma_slider = ctk.CTkSlider(self.pop_up, from_=0.01, to=10.0, command=self.update_sigma)
        sigma_slider.set(self.gaborFilter.sigma)
        sigma_slider.grid(row=rowID, column=0, columnspan=2, padx=5, pady=5, sticky='nsew')
        rowID += 1

        self.theta_label = ctk.CTkLabel(self.pop_up, text=f'Theta: {rad2deg(self.gaborFilter.theta):.2f}')
        self.theta_label.grid(row=rowID, column=0, padx=5, pady=5, sticky='nsew')
        rowID += 1
        theta_slider = ctk.CTkSlider(self.pop_up, from_=0.0, to=360, command=self.update_theta)
        theta_slider.set(rad2deg(self.gaborFilter.theta))
        theta_slider.grid(row=rowID, column=0, columnspan=2, padx=5, pady=5, sticky='nsew')
        rowID += 1

        self.lambd_label = ctk.CTkLabel(self.pop_up, text=f'Lambda: {self.gaborFilter.lambd:.2f}')
        self.lambd_label.grid(row=rowID, column=0, padx=5, pady=5, sticky='nsew')
        rowID += 1
        lambd_slider = ctk.CTkSlider(self.pop_up, from_=0.0, to=10.0, command=self.update_lambd)
        lambd_slider.set(self.gaborFilter.lambd)
        lambd_slider.grid(row=rowID, column=0, columnspan=2, padx=5, pady=5, sticky='nsew')
        rowID += 1

        self.gamma_label = ctk.CTkLabel(self.pop_up, text=f'Gamma: {self.gaborFilter.gamma:.2f}')
        self.gamma_label.grid(row=rowID, column=0, padx=5, pady=5, sticky='nsew')
        rowID += 1
        gamma_slider = ctk.CTkSlider(self.pop_up, from_=0.0, to=1.0, command=self.update_gamma)
        gamma_slider.set(self.gaborFilter.gamma)
        gamma_slider.grid(row=rowID, column=0, columnspan=2, padx=5, pady=5, sticky='nsew')
        rowID += 1

        self.psi_label = ctk.CTkLabel(self.pop_up, text=f'Psi: {self.gaborFilter.psi:.2f}')
        self.psi_label.grid(row=rowID, column=0, padx=5, pady=5, sticky='nsew')
        rowID += 1
        psi_slider = ctk.CTkSlider(self.pop_up, from_=0.0, to=360, command=self.update_psi)
        psi_slider.set(rad2deg(self.gaborFilter.psi))
        psi_slider.grid(row=rowID, column=0, columnspan=2, padx=5, pady=5, sticky='nsew')

    def update_sigma(self, new_sigma: float):
        self.gaborFilter.sigma = new_sigma
        self.sigma_label.configure(text=f'Sigma: {self.gaborFilter.sigma:.2f}')

    def update_theta(self, new_thetaD: float):
        self.gaborFilter.theta = deg2rad(new_thetaD)
        self.theta_label.configure(text=f'Theta: {rad2deg(self.gaborFilter.theta):.2f}')

    def update_lambd(self, new_lambd: float):
        self.gaborFilter.lambd = new_lambd
        self.lambd_label.configure(text=f'Lambda: {self.gaborFilter.lambd:.2f}')

    def update_gamma(self, new_gamma: float):
        self.gaborFilter.gamma = new_gamma
        self.gamma_label.configure(text=f'Gamma: {self.gaborFilter.gamma:.2f}')

    def update_psi(self, new_psiD: float):
        self.gaborFilter.psi = deg2rad(new_psiD)
        self.psi_label.configure(text=f'Psi: {rad2deg(self.gaborFilter.psi):.2f}')

    def filter_kernel(self):
        if not self.pop_up.winfo_exists():
            self.pop_up = ctk.CTkToplevel()
            self.pop_up.title('Gabor Filter Controls')
            self.pop_up.focus_force()
            self.pop_up.geometry('200x500')
            self.pop_up.grid_columnconfigure([0, 1], weight=1)
            self.configure_pop_up()

        return cv2.getGaborKernel(self.gaborFilter.ksize,
                                  self.gaborFilter.sigma,
                                  self.gaborFilter.theta,
                                  self.gaborFilter.lambd,
                                  self.gaborFilter.gamma,
                                  self.gaborFilter.psi)

    def close(self):
        try:
            if self.pop_up is not None and self.pop_up.winfo_exists():
                self.pop_up.destroy()
                self.pop_up.update()
        except Exception:
            pass


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


def ensure_gabor_gui(gabor_gui: None | GaborGUI) -> GaborGUI:
    """
    Return a usable GaborGUI instance, creating or reviving one as needed.
    """
    if gabor_gui is None:
        return GaborGUI()

    try:
        if gabor_gui.pop_up is None or not gabor_gui.pop_up.winfo_exists():
            return GaborGUI()
    except Exception:
        return GaborGUI()

    return gabor_gui


def _apply_convolution_filter(img: NDArray,
                              kernel: ImageKernel,
                              gabor: None | GaborGUI = None,
                              gain: float = 1.0,
                              brightness: int = 0) -> None:
    match kernel:
        case ImageKernel.Unfiltered:
            return

        case ImageKernel.Unsharp:
            gaussian_3 = cv2.GaussianBlur(img, (0, 0), 2.0)
            cv2.addWeighted(img, 2.0, gaussian_3, -1.0, 0, dst=img)

        case ImageKernel.Gabor:
            if gabor is None:
                raise ValueError("Gabor filter requested but no GaborGUI was provided.")
            convolution = gabor.filter_kernel()
            cv2.filter2D(img, -1, convolution, dst=img)

        case ImageKernel.Invert:
            cv2.bitwise_not(img, dst=img)

        case ImageKernel.Gain:
            cv2.convertScaleAbs(img, alpha=gain, beta=0.0, dst=img)

        case ImageKernel.Brightness:
            cv2.convertScaleAbs(img, alpha=1.0, beta=float(brightness), dst=img)

        case _:
            convolution = ImageKernel.get_convolution(kernel)
            cv2.filter2D(img, -1, convolution, dst=img)


def apply_filter(img: NDArray,
                 kernel: ImageKernel,
                 gabor_gui: None | GaborGUI = None,
                 gain: float = 1.0,
                 brightness: int = 0) -> None | GaborGUI:
    """
    Apply the requested filter and manage Gabor GUI lifecycle policy.

    Returns:
        - GaborGUI instance when the Gabor filter is active
        - None when a non-Gabor filter is active
    """
    if not isinstance(kernel, ImageKernel):
        raise ValueError(f"Image Kernel should be ImageKernel Enum class, but is instead {type(kernel)}")

    if kernel == ImageKernel.Gabor:
        gabor_gui = ensure_gabor_gui(gabor_gui)
        _apply_convolution_filter(img, kernel, gabor_gui, gain, brightness)
        return gabor_gui

    if gabor_gui is not None:
        gabor_gui.close()
        gabor_gui = None

    _apply_convolution_filter(img, kernel, None, gain, brightness)
    return None
