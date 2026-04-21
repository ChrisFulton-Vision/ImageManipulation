from customtkinter import (
    CTkFrame, CTkLabel, CTkSlider, CTkCheckBox, CTkComboBox
)

class Hotkey_page(CTkFrame):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)

        title = 'Folder Replay Hotkeys'
        items = [
            ("Space", "Pause / resume"),
            ("f", "Toggle Fixed-FPS ↔ Real-time"),
            ("c / z", "Step forward / backward one frame"),
            ("d / a", "Speed up / slow down playback"),
            ("r", "Reverse direction"),
            ("w", "Toggle overlays"),
            ("s / e", "Mark export start / end"),
            ("[ / ] , { / }", "Adjust time offset (small / large)"),
            ("; / ' , : / \"", "Adjust time offset (fine)"),
            ("p", "Persist time offset"),
            ("Esc", "Exit player"),
        ]

        CTkLabel(self, text=title, font=("Segoe UI", 16, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=12, pady=(12, 8)
        )

        # headings
        CTkLabel(self, text="Key", font=("Segoe UI", 13, "bold")).grid(
            row=1, column=0, sticky="w", padx=12, pady=(6, 2)
        )
        CTkLabel(self, text="Action", font=("Segoe UI", 13, "bold")).grid(
            row=1, column=1, sticky="w", padx=12, pady=(6, 2)
        )

        # rows
        for i, (key, desc) in enumerate(items, start=2):
            CTkLabel(self, text=key).grid(row=i, column=0, sticky="w", padx=12, pady=2)
            CTkLabel(self, text=desc, justify="left", wraplength=520).grid(
                row=i, column=1, sticky="w", padx=12, pady=2
            )

        # let text column expand
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)