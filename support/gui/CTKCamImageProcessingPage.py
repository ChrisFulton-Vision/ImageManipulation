# support/gui/CTKCamImageProcessingPage.py

from __future__ import annotations

from customtkinter import (
    CTkFrame, CTkLabel, CTkSlider, CTkCheckBox, CTkComboBox
)

from support.core.enums import ImageKernel


class ImageProcessing_page(CTkFrame):
    """
    GUI page for the 'Image Processing' tab.

    Assumes controller provides:
      - camConfig
      - calibration (with .validCal)
      - _flag_vars (dict[str, tkinter Variable])
      - confSlider(value)
      - iouSlider(value)
      - saveToCache(...)
      - GaborGUI (optional; has .close())
    """

    def __init__(self, master, controller, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.ctrl = controller

        # --- widgets (created once) ---
        self.conf_var = self.ctrl._flag_vars["yolo_conf"]
        self.confSliderLabel = CTkLabel(self, text=f"Conf: {self.conf_var.get():.2f}")
        self.confSliderBar = CTkSlider(
            self,
            variable=self.conf_var,  # <- key line
            from_=0.15,
            to=1.0,
        )

        self.iou_var = self.ctrl._flag_vars["yolo_iou"]
        self.iouSliderLabel = CTkLabel(self, text=f"IOU: {self.iou_var.get():.2f}")
        self.iouSliderBar = CTkSlider(
            self,
            variable=self.iou_var,  # <- key line
            from_=0.0,
            to=1.0,
        )

        self._trace_guard = False

        def _on_conf_var_changed(*_):
            if self._trace_guard:
                return
            v = float(self.conf_var.get())
            self.confSliderLabel.configure(text=f"Conf: {v:.2f}")

            # write-through to config (use your actual config fields)
            self.ctrl.camConfig.yolo_conf = v
            self.ctrl.saveToCache()

        def _on_iou_var_changed(*_):
            if self._trace_guard:
                return
            v = float(self.iou_var.get())
            self.iouSliderLabel.configure(text=f"IOU: {v:.2f}")

            # write-through to config (use your actual config fields)
            self.ctrl.camConfig.yolo_iou = v
            self.ctrl.saveToCache()

        self.conf_var.trace_add("write", _on_conf_var_changed)
        self.iou_var.trace_add("write", _on_iou_var_changed)

        self.drawChessboardButton = CTkCheckBox(
            self, text="Draw Chessboard", variable=self.ctrl._flag_vars["draw_chessboard"]
        )
        self.undistortCheckbox = CTkCheckBox(
            self, text="Undistort", variable=self.ctrl._flag_vars["undistort"]
        )

        self.detectAprilTagsCheckbox = CTkCheckBox(
            self, text="Detect April Tags", variable=self.ctrl._flag_vars["detectTags"]
        )
        self.hideAprilTagsCheckbox = CTkCheckBox(
            self, text="Hide April Tags", variable=self.ctrl._flag_vars["hideAprilTags"]
        )

        self.yoloInferenceCheckbox = CTkCheckBox(
            self, text="Run YOLO on image", variable=self.ctrl._flag_vars["yoloInference"]
        )
        self.yoloBiasCheckbox = CTkCheckBox(
            self, text="Run YOLO Bias Tracking", variable=self.ctrl._flag_vars["yoloBiasTracking"]
        )

        self.detectHorizonCheckbox = CTkCheckBox(
            self, text="Detect Horizon", variable=self.ctrl._flag_vars["detect_horizon"]
        )

        self.cubemapCheckbox = CTkCheckBox(
            self, text="Cubemap", variable=self.ctrl._flag_vars["cubemap"]
        )

        # Kernel selection
        self.imageProcessingKernelLabel = CTkLabel(self, text="Image Filter: ")
        self.imageProcessingKernelCombobox = CTkComboBox(
            self,
            values=list(ImageKernel.__members__.keys()),
            command=self.updateImageProcessingKernel,
        )

        # Build layout now (or you can delay and call setup() externally)
        self.setup()

    @staticmethod
    def grid_sideBySide(row, *args, col=0):
        for idx, item in enumerate(args):
            item.grid(row=row, column=col + idx, padx=5, pady=5, sticky="nsew")

    def setup(self):
        # Clear any prior layout (safe if re-called)
        for w in self.winfo_children():
            # keep widgets, just forget grid positions
            if w.grid_info():
                w.grid_forget()

        rowID = 0

        self.grid_sideBySide(rowID, self.confSliderLabel, self.confSliderBar)
        rowID += 1

        self.grid_sideBySide(rowID, self.iouSliderLabel, self.iouSliderBar)
        rowID += 1

        if not self.ctrl.calibration.validCal:
            self.undistortCheckbox.configure(state="disabled")

        self.grid_sideBySide(rowID, self.drawChessboardButton, self.undistortCheckbox)
        rowID += 1

        self.grid_sideBySide(rowID, self.detectAprilTagsCheckbox, self.hideAprilTagsCheckbox)
        rowID += 1

        pnp3DTruthPoints = CTkCheckBox(
            self, text="SolvePnP 3D Truth Into Image", variable=self.ctrl._flag_vars["pnp3DTruthPoints"]
        )
        qnp3DTruthPoints = CTkCheckBox(
            self, text="SolveQnP 3D Truth Into Image", variable=self.ctrl._flag_vars["qnp3DTruthPoints"]
        )
        self.grid_sideBySide(rowID, pnp3DTruthPoints, qnp3DTruthPoints)
        rowID += 1

        self.grid_sideBySide(rowID, self.yoloInferenceCheckbox, self.yoloBiasCheckbox)
        rowID += 1

        # --- Pose from YOLO centers (multi-feature) ---
        pnpYoloPoints = CTkCheckBox(
            self, text="SolvePnP from YOLO", variable=self.ctrl._flag_vars["pnpYoloPoints"]
        )
        qnpYoloPoints = CTkCheckBox(
            self, text="SolveQnP from YOLO", variable=self.ctrl._flag_vars["qnpYoloPoints"]
        )
        self.grid_sideBySide(rowID, pnpYoloPoints, qnpYoloPoints)
        rowID += 1

        qnpKFYoloPoints = CTkCheckBox(
            self, text="SolveWQnP from YOLO", variable=self.ctrl._flag_vars["qnpKFYoloPoints"]
        )
        qnpKFYoloPoints.grid(row=rowID, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        circleNotFeatureDraw = CTkCheckBox(
            self, text="Circles", variable=self.ctrl._flag_vars["circles_not_features"]
        )
        circleNotFeatureDraw.grid(row=rowID, column=1, columnspan=2, padx=5, pady=5, sticky="ew")
        rowID += 1

        detectCornersCheckbox = CTkCheckBox(
            self, text="Detect Corners", variable=self.ctrl._flag_vars["detect_corners"]
        )
        self.grid_sideBySide(rowID, detectCornersCheckbox, self.detectHorizonCheckbox)
        rowID += 1

        factorgraphCheckbox = CTkCheckBox(
            self, text="Factor Graph", variable=self.ctrl._flag_vars["factor_graph"]
        )
        hyperfocusCheckbox = CTkCheckBox(
            self, text="Hyper Focus", variable=self.ctrl._flag_vars["hyper_focus"]
        )
        self.grid_sideBySide(rowID, factorgraphCheckbox, hyperfocusCheckbox)
        rowID += 1

        phaseCorrelationCheckbox = CTkCheckBox(
            self, text="PhaseCorrelation", variable=self.ctrl._flag_vars["phase_correlation"]
        )
        crosshairsCheckbox = CTkCheckBox(
            self, text="Crosshairs", variable=self.ctrl._flag_vars["crosshairs"]
        )
        self.grid_sideBySide(rowID, phaseCorrelationCheckbox, crosshairsCheckbox)
        rowID += 1

        hudCheckbox = CTkCheckBox(self, text="HUD", variable=self.ctrl._flag_vars["hud"])
        self.grid_sideBySide(rowID, self.cubemapCheckbox, hudCheckbox)
        rowID += 1

        # Kernel drop-down
        self.imageProcessingKernelCombobox.set(self.ctrl.camConfig.processingKernel.name)
        self.updateImageProcessingKernel(self.ctrl.camConfig.processingKernel.name)
        self.grid_sideBySide(rowID, self.imageProcessingKernelLabel, self.imageProcessingKernelCombobox)


    def updateImageProcessingKernel(self, newValue):
        # Close Gabor GUI if switching away
        if self.ctrl.camConfig.processingKernel == ImageKernel.Gabor and self.ctrl.GaborGUI is not None:
            self.ctrl.GaborGUI.close()

        self.ctrl.camConfig.processingKernel = ImageKernel(newValue)

        if self.ctrl.camConfig.processingKernel == ImageKernel.Unfiltered:
            self.imageProcessingKernelCombobox.configure(fg_color="#343638", text_color="#DCE4EE")
        else:
            self.imageProcessingKernelCombobox.configure(fg_color="yellow", text_color="black")

        self.ctrl.saveToCache()
