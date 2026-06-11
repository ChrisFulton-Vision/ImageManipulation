import customtkinter as ctk
from datetime import datetime, timezone
import math
import tkinter as tk


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

DISTANCE_UNIT_FOR_SPEED_UNIT = {
    "mph": ("miles", 1609.344),
    "m/s": ("m", 1.0),
    "knots": ("nmi", 1852.0),
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


def america_distance_for_unit(speed_unit: str) -> tuple[float, str]:
    """Return America width expressed in the distance unit matching the selected speed unit."""
    distance_unit, meters_per_unit = DISTANCE_UNIT_FOR_SPEED_UNIT[speed_unit]
    return AMERICA_WIDTH_M / meters_per_unit, distance_unit


class AmericaPerAmericaApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("America per America Converter")
        self.geometry("520x560")
        self.resizable(False, False)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.speed_var = ctk.StringVar(value="1")
        self.unit_var = ctk.StringVar(value="mph")
        self._update_job = None
        self._easter_egg_job = None
        self._easter_egg_active = False
        self._flash_phase = 0

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
            text="",
            font=ctk.CTkFont(size=12),
            text_color="gray55",
        )
        self.note_label.pack(pady=(18, 0))

        self.easter_egg_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.easter_egg_canvas = tk.Canvas(
            self.easter_egg_frame,
            width=460,
            height=180,
            highlightthickness=0,
            bg="#0b1020",
        )
        self.easter_egg_canvas.pack()

        self.speed_var.trace_add("write", lambda *_: self._update_values())

    def _update_values(self):
        if self._update_job is not None:
            self.after_cancel(self._update_job)
            self._update_job = None

        age_s = america_age_seconds()
        age_years = age_s / (365.2425 * 24 * 3600)
        unit = self.unit_var.get()
        apa_per_unit = unit_to_apa(1.0, unit)
        america_distance, distance_unit = america_distance_for_unit(unit)

        self.rate_label.configure(
            text=f"Current rate: 1 {unit} = {apa_per_unit:,.6f} ApA"
        )

        self.age_label.configure(
            text=f"America age: {age_years:,.9f} years"
        )

        self.note_label.configure(
            text=f"1 America-distance = {america_distance:,.1f} {distance_unit}"
        )

        try:
            speed = float(self.speed_var.get())

            apa = unit_to_apa(speed, unit)

            self.result_label.configure(
                text=f"{apa:,.6f} ApA",
                text_color=("black", "white"),
            )
            self._set_easter_egg_active(unit == "mph" and abs(speed - 1776.0) < 1e-9)

        except ValueError:
            self.result_label.configure(
                text="Enter a valid number",
                text_color="tomato",
            )
            self._set_easter_egg_active(False)

        self._update_job = self.after(1000, self._update_values)

    def _set_easter_egg_active(self, active: bool):
        if active == self._easter_egg_active:
            return

        self._easter_egg_active = active
        if active:
            self.easter_egg_frame.pack(pady=(18, 0))
            self._flash_phase = 0
            self._animate_easter_egg()
            return

        if self._easter_egg_job is not None:
            self.after_cancel(self._easter_egg_job)
            self._easter_egg_job = None
        self.easter_egg_canvas.delete("all")
        self.easter_egg_frame.pack_forget()

    def _animate_easter_egg(self):
        if not self._easter_egg_active:
            return

        self._flash_phase += 1
        self.easter_egg_canvas.delete("all")

        bg = "#08111f" if self._flash_phase % 2 == 0 else "#101b33"
        self.easter_egg_canvas.configure(bg=bg)
        self._draw_flag()
        self._draw_fireworks()
        self._draw_eagle()
        self.easter_egg_canvas.create_text(
            230, 162,
            text="1776 MPH: MAXIMUM FREEDOM",
            fill="#f5d76e",
            font=("Arial", 16, "bold"),
        )

        self._easter_egg_job = self.after(180, self._animate_easter_egg)

    def _star_points(self, cx, cy, r_outer, r_inner=None, rotation_deg=-90):
        if r_inner is None:
            r_inner = r_outer * 0.42

        pts = []
        start = math.radians(rotation_deg)
        for i in range(10):
            ang = start + i * math.pi / 5.0
            r = r_outer if i % 2 == 0 else r_inner
            pts.extend([cx + r * math.cos(ang), cy + r * math.sin(ang)])
        return pts

    def _draw_flag(self):

        c = self.easter_egg_canvas
        c.delete("flag")

        tag = "flag"

        # Bounding box
        x0, y0, x1, y1 = 18, 18, 208, 132
        w = x1 - x0
        h = y1 - y0

        # Colors
        red = "#b22234" if self._flash_phase % 2 == 0 else "#c81f34"
        white = "#ffffff"
        blue = "#3c3b6e"
        star_color = "#ffffff"

        # Optional tiny wave illusion
        wave = 1 if self._flash_phase % 2 == 0 else 0

        # --- Stripes ---
        stripe_h = h / 13.0
        for i in range(13):
            y_top = y0 + i * stripe_h
            y_bot = y0 + (i + 1) * stripe_h
            color = red if i % 2 == 0 else white
            c.create_rectangle(
                x0,
                y_top,
                x1,
                y_bot,
                fill=color,
                outline="",
                tags=tag,
            )

        # --- Canton ---
        # More realistic than 0.4*width; official-ish visual proportion
        canton_h = stripe_h * 7
        canton_w = h * 0.76
        canton_x1 = x0 + canton_w
        canton_y1 = y0 + canton_h

        c.create_rectangle(
            x0,
            y0,
            canton_x1,
            canton_y1,
            fill=blue,
            outline="",
            tags=tag,
        )

        # --- Stars: 9 rows alternating 6 / 5 / 6 / 5 ... = 50 total ---
        rows = 9
        stars_per_row = [6, 5, 6, 5, 6, 5, 6, 5, 6]

        # Margins inside canton
        mx = canton_w * 0.10
        my = canton_h * 0.10

        usable_w = canton_w - 2 * mx
        usable_h = canton_h - 2 * my

        row_gap = usable_h / (rows - 1)

        # Use smaller star size so they read as stars, not blobs
        star_r = min(usable_w / 18.0, usable_h / 18.0)

        for row in range(rows):
            n = stars_per_row[row]
            y = y0 + my + row * row_gap + wave * 0.15 * (row % 2)

            if n == 6:
                col_gap = usable_w / 5.0
                xs = [x0 + mx + i * col_gap for i in range(6)]
            else:
                col_gap = usable_w / 4.0
                xs = [x0 + mx + col_gap / 2.0 + i * col_gap for i in range(5)]

            for cx in xs:
                pts = self._star_points(cx, y, star_r, star_r * 0.45)
                c.create_polygon(
                    pts,
                    fill=star_color,
                    outline=star_color,
                    width=1,
                    tags=tag,
                )

        # Thin outline helps at tiny sizes
        c.create_rectangle(
            x0, y0, x1, y1,
            outline="#e6e6e6",
            width=1,
            tags=tag,
        )

    def _draw_fireworks(self):
        c = self.easter_egg_canvas
        centers = [(290, 52), (380, 42), (350, 92)]
        palettes = [
            ("#ff595e", "#ffca3a"),
            ("#8ac926", "#1982c4"),
            ("#ff924c", "#c1121f"),
        ]

        for idx, ((cx, cy), colors) in enumerate(zip(centers, palettes)):
            radius = 16 + ((self._flash_phase + idx) % 5) * 4
            for spoke in range(12):
                ang = 2.0 * math.pi * spoke / 12.0
                x2 = cx + radius * math.cos(ang)
                y2 = cy + radius * math.sin(ang)
                color = colors[(spoke + self._flash_phase) % len(colors)]
                c.create_line(cx, cy, x2, y2, fill=color, width=2)
                c.create_oval(x2 - 2, y2 - 2, x2 + 2, y2 + 2, fill=color, outline=color)

    def _draw_eagle(self):
        c = self.easter_egg_canvas
        c.delete("eagle")

        flap = 5 if self._flash_phase % 2 == 0 else -5
        bob = 1 if self._flash_phase % 2 == 0 else -1

        tag = "eagle"

        # Palette
        dark = "#2b1b13"
        body = "#5b3a29"
        body_mid = "#70452f"
        body_light = "#8a5a3c"
        outline = "#e6dcc7"
        white = "#f7f3e8"
        beak = "#f4c542"
        beak_shadow = "#c9921e"
        eye = "#111111"
        talon = "#f4c542"

        # --- Drop shadow ---
        c.create_polygon(
            282, 145 + bob,
            306, 128 + bob,
            342, 122 + bob,
            376, 126 + bob,
            405, 140 + bob,
            420, 151 + bob,
            388, 155 + bob,
            342, 155 + bob,
            302, 153 + bob,
            fill="#140f0c",
            outline="",
            smooth=True,
            tags=tag,
        )

        # --- Left swept wing ---
        left_wing = [
            282, 128 + bob,
            304, 108 + flap,
            332, 104 + flap,
            356, 117 + bob,
            344, 131 + bob,
            318, 128 + bob,
            300, 139 + bob,
        ]
        c.create_polygon(
            left_wing,
            fill=body,
            outline=outline,
            width=2,
            smooth=True,
            tags=tag,
        )

        # Left wing feather strokes
        feather_lines = [
            (302, 113 + flap, 283, 128 + bob),
            (316, 109 + flap, 296, 137 + bob),
            (332, 111 + flap, 314, 129 + bob),
            (346, 119 + bob, 330, 129 + bob),
        ]
        for x1, y1, x2, y2 in feather_lines:
            c.create_line(
                x1, y1, x2, y2,
                fill=body_light,
                width=2,
                capstyle="round",
                tags=tag,
            )

        # --- Right swept wing / raised shoulder ---
        right_wing = [
            350, 119 + bob,
            378, 102 - flap,
            405, 111 - flap,
            424, 131 + bob,
            406, 137 + bob,
            386, 130 + bob,
            364, 139 + bob,
        ]
        c.create_polygon(
            right_wing,
            fill=body_mid,
            outline=outline,
            width=2,
            smooth=True,
            tags=tag,
        )

        # Right wing feather strokes
        feather_lines = [
            (382, 108 - flap, 365, 135 + bob),
            (397, 113 - flap, 382, 130 + bob),
            (412, 126 + bob, 393, 132 + bob),
        ]
        for x1, y1, x2, y2 in feather_lines:
            c.create_line(
                x1, y1, x2, y2,
                fill=body_light,
                width=2,
                capstyle="round",
                tags=tag,
            )

        # --- Main body ---
        body_points = [
            298, 137 + bob,
            322, 127 + bob,
            354, 126 + bob,
            384, 132 + bob,
            407, 144 + bob,
            390, 153 + bob,
            350, 156 + bob,
            314, 153 + bob,
            289, 147 + bob,
        ]
        c.create_polygon(
            body_points,
            fill=body,
            outline=outline,
            width=2,
            smooth=True,
            tags=tag,
        )

        # Body highlight
        c.create_arc(
            306, 130 + bob,
            382, 158 + bob,
            start=190,
            extent=115,
            style="arc",
            outline=body_light,
            width=2,
            tags=tag,
        )

        # --- Tail feathers ---
        tail = [
            (287, 142 + bob, 269, 134 + bob, 284, 153 + bob),
            (292, 146 + bob, 270, 148 + bob, 292, 156 + bob),
            (298, 149 + bob, 279, 160 + bob, 307, 157 + bob),
        ]
        for pts in tail:
            c.create_polygon(
                pts,
                fill=dark,
                outline=outline,
                width=1,
                smooth=True,
                tags=tag,
            )

        # --- Neck transition ---
        c.create_polygon(
            389, 124 + bob,
            410, 119 + bob,
            421, 130 + bob,
            410, 142 + bob,
            391, 140 + bob,
            fill=body_mid,
            outline=outline,
            width=1,
            smooth=True,
            tags=tag,
        )

        # --- White head ---
        c.create_oval(
            407, 114 + bob,
            432, 136 + bob,
            fill=white,
            outline=outline,
            width=2,
            tags=tag,
        )

        # Brow / fierce eagle energy
        c.create_line(
            416, 121 + bob,
            426, 119 + bob,
            fill="#d8d0bf",
            width=2,
            capstyle="round",
            tags=tag,
        )

        # Eye
        c.create_oval(
            421, 122 + bob,
            424, 125 + bob,
            fill=eye,
            outline=eye,
            tags=tag,
        )

        # --- Beak ---
        c.create_polygon(
            430, 123 + bob,
            448, 118 + bob,
            435, 130 + bob,
            fill=beak,
            outline=beak,
            tags=tag,
        )
        c.create_polygon(
            433, 128 + bob,
            445, 119 + bob,
            438, 131 + bob,
            fill=beak_shadow,
            outline=beak_shadow,
            tags=tag,
        )

        # --- Tiny talons ---
        c.create_line(
            348, 155 + bob,
            344, 163 + bob,
            339, 160 + bob,
            fill=talon,
            width=2,
            capstyle="round",
            joinstyle="round",
            tags=tag,
        )
        c.create_line(
            362, 155 + bob,
            363, 163 + bob,
            370, 160 + bob,
            fill=talon,
            width=2,
            capstyle="round",
            joinstyle="round",
            tags=tag,
        )

        # Optional: keep eagle visually above older background fireworks
        c.tag_raise(tag)


if __name__ == "__main__":
    app = AmericaPerAmericaApp()
    app.mainloop()
