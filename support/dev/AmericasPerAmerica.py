import customtkinter as ctk
from datetime import datetime, timezone


# -----------------------------
# Silly-but-configurable constants
# -----------------------------

# "America" distance: approximate contiguous U.S. width
AMERICA_WIDTH_M = 2800 * 1609.344  # 2800 miles in meters

# "America" time: age of America since July 4, 1776
AMERICA_BIRTH = datetime(1776, 7, 4, tzinfo=timezone.utc)


UNIT_TO_MPS = {
    "mph": 0.44704,
    "m/s": 1.0,
    "knots": 1852.0 / 3600.0,
}


def america_age_seconds() -> float:
    """Return America's current age in seconds."""
    now = datetime.now(timezone.utc)
    return (now - AMERICA_BIRTH).total_seconds()


def mps_to_apa(speed_mps: float) -> float:
    """
    Convert meters per second to Americas per America.

    ApA = speed * America_age_seconds / America_width_meters
    """
    return speed_mps * america_age_seconds() / AMERICA_WIDTH_M


def unit_to_apa(speed: float, unit: str) -> float:
    """Convert an arbitrary supported speed unit to ApA."""
    speed_mps = speed * UNIT_TO_MPS[unit]
    return mps_to_apa(speed_mps)


class AmericaPerAmericaApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("America per America Converter")
        self.geometry("520x360")
        self.resizable(False, False)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.speed_var = ctk.StringVar(value="43")
        self.unit_var = ctk.StringVar(value="m/s")

        self._build_ui()
        self._update_values()

    def _build_ui(self):
        title = ctk.CTkLabel(
            self,
            text="America per America Converter",
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        title.pack(pady=(24, 6))

        subtitle = ctk.CTkLabel(
            self,
            text="Convert speed into freedom-normalized units.",
            font=ctk.CTkFont(size=14),
            text_color="gray75",
        )
        subtitle.pack(pady=(0, 20))

        input_frame = ctk.CTkFrame(self)
        input_frame.pack(padx=24, pady=8, fill="x")

        speed_label = ctk.CTkLabel(
            input_frame,
            text="Speed:",
            font=ctk.CTkFont(size=16),
        )
        speed_label.grid(row=0, column=0, padx=(16, 8), pady=18, sticky="w")

        self.speed_entry = ctk.CTkEntry(
            input_frame,
            textvariable=self.speed_var,
            width=160,
            font=ctk.CTkFont(size=16),
        )
        self.speed_entry.grid(row=0, column=1, padx=8, pady=18)

        self.unit_menu = ctk.CTkOptionMenu(
            input_frame,
            variable=self.unit_var,
            values=list(UNIT_TO_MPS.keys()),
            width=120,
            command=lambda _: self._update_values(),
        )
        self.unit_menu.grid(row=0, column=2, padx=(8, 16), pady=18)

        input_frame.grid_columnconfigure(1, weight=1)

        self.result_label = ctk.CTkLabel(
            self,
            text="",
            font=ctk.CTkFont(size=26, weight="bold"),
        )
        self.result_label.pack(pady=(24, 8))

        self.rate_label = ctk.CTkLabel(
            self,
            text="",
            font=ctk.CTkFont(size=14),
            text_color="gray80",
        )
        self.rate_label.pack(pady=4)

        self.age_label = ctk.CTkLabel(
            self,
            text="",
            font=ctk.CTkFont(size=14),
            text_color="gray80",
        )
        self.age_label.pack(pady=4)

        self.note_label = ctk.CTkLabel(
            self,
            text=f'1 America-distance = {AMERICA_WIDTH_M:,.1f} m',
            font=ctk.CTkFont(size=12),
            text_color="gray55",
        )
        self.note_label.pack(pady=(18, 0))

        self.speed_var.trace_add("write", lambda *_: self._update_values())

    def _update_values(self):
        age_s = america_age_seconds()
        age_years = age_s / (365.2425 * 24 * 3600)

        apa_per_mps = age_s / AMERICA_WIDTH_M

        self.rate_label.configure(
            text=f"Current rate: 1 m/s = {apa_per_mps:,.6f} ApA"
        )

        self.age_label.configure(
            text=f"America age: {age_years:,.9f} years"
        )

        try:
            speed = float(self.speed_var.get())
            unit = self.unit_var.get()

            apa = unit_to_apa(speed, unit)

            self.result_label.configure(
                text=f"{apa:,.6f} ApA",
                text_color=("black", "white"),
            )

        except ValueError:
            self.result_label.configure(
                text="Enter a valid number",
                text_color="tomato",
            )

        self.after(1000, self._update_values)


if __name__ == "__main__":
    app = AmericaPerAmericaApp()
    app.mainloop()