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

        # Kernel drop-down
