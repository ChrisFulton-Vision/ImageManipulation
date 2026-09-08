from __future__ import annotations

from datetime import datetime, timezone
import math
import sys

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import (
    QColor,
    QCloseEvent,
    QFont,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


# "America" distance: approximate contiguous U.S. width.
AMERICA_WIDTH_M = 2800 * 1609.344

# "America" time: age of America since July 4, 1776.
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
    return (datetime.now(timezone.utc) - AMERICA_BIRTH).total_seconds()


def mps_to_apa(speed_mps: float) -> float:
    """Convert meters per second to Americas per America."""
    return speed_mps * america_age_seconds() / AMERICA_WIDTH_M


def unit_to_apa(speed: float, unit: str) -> float:
    """Convert an arbitrary supported speed unit to ApA."""
    return speed * UNIT_TO_MPS[unit] * america_age_seconds() / AMERICA_WIDTH_M


def america_distance_for_unit(speed_unit: str) -> tuple[float, str]:
    """Return America width in the distance unit matching the speed unit."""
    distance_unit, meters_per_unit = DISTANCE_UNIT_FOR_SPEED_UNIT[speed_unit]
    return AMERICA_WIDTH_M / meters_per_unit, distance_unit


class FreedomCanvas(QWidget):
    """Animated native-Qt rendering of the 1776-mph easter egg."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._phase = 0
        self.setFixedSize(460, 180)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def set_phase(self, phase: int) -> None:
        self._phase = phase
        self.update()

    @staticmethod
    def _star_points(
        cx: float,
        cy: float,
        outer_radius: float,
        inner_radius: float | None = None,
        rotation_degrees: float = -90.0,
    ) -> QPolygonF:
        inner_radius = inner_radius or outer_radius * 0.42
        start = math.radians(rotation_degrees)
        points = []
        for index in range(10):
            angle = start + index * math.pi / 5.0
            radius = outer_radius if index % 2 == 0 else inner_radius
            points.append(
                QPointF(
                    cx + radius * math.cos(angle),
                    cy + radius * math.sin(angle),
                )
            )
        return QPolygonF(points)

    @staticmethod
    def _polygon(
        painter: QPainter,
        coordinates: list[tuple[float, float]],
        fill: str,
        outline: str | None = None,
        width: float = 1.0,
    ) -> None:
        polygon = QPolygonF([QPointF(x, y) for x, y in coordinates])
        painter.setBrush(QColor(fill))
        painter.setPen(
            QPen(QColor(outline), width)
            if outline
            else QPen(Qt.PenStyle.NoPen)
        )
        painter.drawPolygon(polygon)

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        background = "#08111f" if self._phase % 2 == 0 else "#101b33"
        painter.fillRect(self.rect(), QColor(background))
        self._draw_flag(painter)
        self._draw_fireworks(painter)
        self._draw_eagle(painter)

        painter.setPen(QColor("#f5d76e"))
        painter.setFont(QFont("Arial", 15, QFont.Weight.Bold))
        painter.drawText(
            QRectF(0, 148, self.width(), 28),
            Qt.AlignmentFlag.AlignCenter,
            "1776 MPH: MAXIMUM FREEDOM",
        )

    def _draw_flag(self, painter: QPainter) -> None:
        x0, y0, x1, y1 = 18.0, 18.0, 208.0, 132.0
        width = x1 - x0
        height = y1 - y0
        red = "#b22234" if self._phase % 2 == 0 else "#c81f34"
        stripe_height = height / 13.0

        painter.setPen(Qt.PenStyle.NoPen)
        for index in range(13):
            color = QColor(red if index % 2 == 0 else "#ffffff")
            painter.fillRect(
                QRectF(x0, y0 + index * stripe_height, width, stripe_height + 0.5),
                color,
            )

        canton_height = stripe_height * 7.0
        canton_width = height * 0.76
        painter.fillRect(
            QRectF(x0, y0, canton_width, canton_height),
            QColor("#3c3b6e"),
        )

        rows = (6, 5, 6, 5, 6, 5, 6, 5, 6)
        margin_x = canton_width * 0.10
        margin_y = canton_height * 0.10
        usable_width = canton_width - 2.0 * margin_x
        usable_height = canton_height - 2.0 * margin_y
        row_gap = usable_height / (len(rows) - 1)
        star_radius = min(usable_width / 18.0, usable_height / 18.0)
        wave = 1 if self._phase % 2 == 0 else 0

        painter.setBrush(QColor("#ffffff"))
        painter.setPen(QPen(QColor("#ffffff"), 0.7))
        for row, count in enumerate(rows):
            y = y0 + margin_y + row * row_gap + wave * 0.15 * (row % 2)
            if count == 6:
                gap = usable_width / 5.0
                xs = [x0 + margin_x + index * gap for index in range(6)]
            else:
                gap = usable_width / 4.0
                xs = [x0 + margin_x + gap / 2.0 + index * gap for index in range(5)]
            for x in xs:
                painter.drawPolygon(
                    self._star_points(x, y, star_radius, star_radius * 0.45)
                )

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#e6e6e6"), 1.0))
        painter.drawRect(QRectF(x0, y0, width, height))

    def _draw_fireworks(self, painter: QPainter) -> None:
        centers = ((290, 52), (380, 42), (350, 92))
        palettes = (
            ("#ff595e", "#ffca3a"),
            ("#8ac926", "#1982c4"),
            ("#ff924c", "#c1121f"),
        )
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for index, ((cx, cy), colors) in enumerate(zip(centers, palettes)):
            radius = 16 + ((self._phase + index) % 5) * 4
            for spoke in range(12):
                angle = 2.0 * math.pi * spoke / 12.0
                x2 = cx + radius * math.cos(angle)
                y2 = cy + radius * math.sin(angle)
                color = QColor(colors[(spoke + self._phase) % len(colors)])
                spoke_pen = QPen(color, 2.0)
                spoke_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(spoke_pen)
                painter.drawLine(QPointF(cx, cy), QPointF(x2, y2))
                painter.setBrush(color)
                painter.drawEllipse(QPointF(x2, y2), 2.0, 2.0)

    def _draw_eagle(self, painter: QPainter) -> None:
        flap = 5 if self._phase % 2 == 0 else -5
        bob = 1 if self._phase % 2 == 0 else -1
        dark = "#2b1b13"
        body = "#5b3a29"
        body_mid = "#70452f"
        body_light = "#8a5a3c"
        outline = "#e6dcc7"
        white = "#f7f3e8"
        beak = "#f4c542"
        beak_shadow = "#c9921e"

        self._polygon(
            painter,
            [(282, 145 + bob), (306, 128 + bob), (342, 122 + bob),
             (376, 126 + bob), (405, 140 + bob), (420, 151 + bob),
             (388, 155 + bob), (342, 155 + bob), (302, 153 + bob)],
            "#140f0c",
        )
        self._polygon(
            painter,
            [(282, 128 + bob), (304, 108 + flap), (332, 104 + flap),
             (356, 117 + bob), (344, 131 + bob), (318, 128 + bob),
             (300, 139 + bob)],
            body,
            outline,
            2.0,
        )
        self._polygon(
            painter,
            [(350, 119 + bob), (378, 102 - flap), (405, 111 - flap),
             (424, 131 + bob), (406, 137 + bob), (386, 130 + bob),
             (364, 139 + bob)],
            body_mid,
            outline,
            2.0,
        )

        feather_pen = QPen(QColor(body_light), 2.0)
        feather_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(feather_pen)
        for x1, y1, x2, y2 in (
            (302, 113 + flap, 283, 128 + bob),
            (316, 109 + flap, 296, 137 + bob),
            (332, 111 + flap, 314, 129 + bob),
            (346, 119 + bob, 330, 129 + bob),
            (382, 108 - flap, 365, 135 + bob),
            (397, 113 - flap, 382, 130 + bob),
            (412, 126 + bob, 393, 132 + bob),
        ):
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        self._polygon(
            painter,
            [(298, 137 + bob), (322, 127 + bob), (354, 126 + bob),
             (384, 132 + bob), (407, 144 + bob), (390, 153 + bob),
             (350, 156 + bob), (314, 153 + bob), (289, 147 + bob)],
            body,
            outline,
            2.0,
        )

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(body_light), 2.0))
        painter.drawArc(QRectF(306, 130 + bob, 76, 28), 190 * 16, 115 * 16)

        for points in (
            [(287, 142 + bob), (269, 134 + bob), (284, 153 + bob)],
            [(292, 146 + bob), (270, 148 + bob), (292, 156 + bob)],
            [(298, 149 + bob), (279, 160 + bob), (307, 157 + bob)],
        ):
            self._polygon(painter, points, dark, outline)

        self._polygon(
            painter,
            [(389, 124 + bob), (410, 119 + bob), (421, 130 + bob),
             (410, 142 + bob), (391, 140 + bob)],
            body_mid,
            outline,
        )

        painter.setBrush(QColor(white))
        painter.setPen(QPen(QColor(outline), 2.0))
        painter.drawEllipse(QRectF(407, 114 + bob, 25, 22))
        brow_pen = QPen(QColor("#d8d0bf"), 2.0)
        brow_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(brow_pen)
        painter.drawLine(QPointF(416, 121 + bob), QPointF(426, 119 + bob))
        painter.setBrush(QColor("#111111"))
        painter.setPen(QPen(QColor("#111111")))
        painter.drawEllipse(QRectF(421, 122 + bob, 3, 3))

        self._polygon(
            painter,
            [(430, 123 + bob), (448, 118 + bob), (435, 130 + bob)],
            beak,
            beak,
        )
        self._polygon(
            painter,
            [(433, 128 + bob), (445, 119 + bob), (438, 131 + bob)],
            beak_shadow,
            beak_shadow,
        )

        talon_pen = QPen(QColor(beak), 2.0)
        talon_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(talon_pen)
        talon_paths = (
            ((348, 155 + bob), (344, 163 + bob), (339, 160 + bob)),
            ((362, 155 + bob), (363, 163 + bob), (370, 160 + bob)),
        )
        for points in talon_paths:
            path = QPainterPath(QPointF(*points[0]))
            path.lineTo(QPointF(*points[1]))
            path.lineTo(QPointF(*points[2]))
            painter.drawPath(path)


class AmericaPerAmericaApp(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("America per America Converter")
        self.setFixedSize(520, 560)
        self._flash_phase = 0
        self._easter_egg_active = False

        self._build_ui()

        self._update_timer = QTimer(self)
        self._update_timer.setInterval(1000)
        self._update_timer.timeout.connect(self._update_values)
        self._update_timer.start()

        self._easter_egg_timer = QTimer(self)
        self._easter_egg_timer.setInterval(180)
        self._easter_egg_timer.timeout.connect(self._animate_easter_egg)

        self._update_values()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 18)
        root.setSpacing(8)

        title = QLabel("America per America Converter")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Segoe UI", 22, QFont.Weight.Bold))
        root.addWidget(title)

        subtitle = QLabel("Convert speed into freedom-normalized units.")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #b8bec9;")
        subtitle.setFont(QFont("Segoe UI", 11))
        root.addWidget(subtitle)
        root.addSpacing(10)

        input_frame = QFrame()
        input_frame.setObjectName("inputFrame")
        input_layout = QHBoxLayout(input_frame)
        input_layout.setContentsMargins(16, 16, 16, 16)
        input_layout.setSpacing(8)

        speed_label = QLabel("Speed:")
        speed_label.setFont(QFont("Segoe UI", 12))
        input_layout.addWidget(speed_label)

        self.speed_entry = QLineEdit("1")
        self.speed_entry.setFont(QFont("Segoe UI", 12))
        self.speed_entry.setAlignment(Qt.AlignmentFlag.AlignRight)
        input_layout.addWidget(self.speed_entry, 1)

        self.unit_menu = QComboBox()
        self.unit_menu.addItems(UNIT_TO_MPS)
        self.unit_menu.setCurrentText("mph")
        self.unit_menu.setMinimumWidth(110)
        input_layout.addWidget(self.unit_menu)
        root.addWidget(input_frame)

        root.addSpacing(12)
        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        root.addWidget(self.result_label)

        self.rate_label = QLabel()
        self.rate_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.rate_label.setStyleSheet("color: #c9ced8;")
        root.addWidget(self.rate_label)

        self.age_label = QLabel()
        self.age_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.age_label.setStyleSheet("color: #c9ced8;")
        root.addWidget(self.age_label)

        self.note_label = QLabel()
        self.note_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.note_label.setStyleSheet("color: #858d9c;")
        root.addWidget(self.note_label)

        root.addStretch(1)
        self.easter_egg_canvas = FreedomCanvas()
        self.easter_egg_canvas.hide()
        root.addWidget(
            self.easter_egg_canvas,
            alignment=Qt.AlignmentFlag.AlignHCenter,
        )

        self.speed_entry.textChanged.connect(self._update_values)
        self.unit_menu.currentTextChanged.connect(self._update_values)

        self.setStyleSheet(
            """
            AmericaPerAmericaApp {
                background-color: #111827;
                color: #f8fafc;
            }
            QFrame#inputFrame {
                background-color: #1f2937;
                border: 1px solid #374151;
                border-radius: 8px;
            }
            QLineEdit, QComboBox {
                background-color: #0f172a;
                color: #f8fafc;
                border: 1px solid #4b5563;
                border-radius: 5px;
                padding: 6px;
            }
            QComboBox QAbstractItemView {
                background-color: #1f2937;
                color: #f8fafc;
                selection-background-color: #2563eb;
            }
            """
        )

    def _update_values(self, *_args) -> None:
        age_seconds = america_age_seconds()
        age_years = age_seconds / (365.2425 * 24 * 3600)
        unit = self.unit_menu.currentText()
        apa_per_unit = UNIT_TO_MPS[unit] * age_seconds / AMERICA_WIDTH_M
        america_distance, distance_unit = america_distance_for_unit(unit)

        self.rate_label.setText(f"Current rate: 1 {unit} = {apa_per_unit:,.6f} ApA")
        self.age_label.setText(f"America age: {age_years:,.9f} years")
        self.note_label.setText(
            f"1 America-distance = {america_distance:,.1f} {distance_unit}"
        )

        try:
            speed = float(self.speed_entry.text())
        except ValueError:
            self.result_label.setText("Enter a valid number")
            self.result_label.setStyleSheet("color: #ff735f;")
            self._set_easter_egg_active(False)
            return

        self.result_label.setText(f"{speed * apa_per_unit:,.6f} ApA")
        self.result_label.setStyleSheet("color: #ffffff;")
        self._set_easter_egg_active(unit == "mph" and abs(speed - 1776.0) < 1e-9)

    def _set_easter_egg_active(self, active: bool) -> None:
        if active == self._easter_egg_active:
            return
        self._easter_egg_active = active
        if active:
            self._flash_phase = 0
            self.easter_egg_canvas.show()
            self._animate_easter_egg()
            self._easter_egg_timer.start()
        else:
            self._easter_egg_timer.stop()
            self.easter_egg_canvas.hide()

    def _animate_easter_egg(self) -> None:
        if not self._easter_egg_active:
            return
        self._flash_phase += 1
        self.easter_egg_canvas.set_phase(self._flash_phase)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        self._update_timer.stop()
        self._easter_egg_timer.stop()
        event.accept()


def _dark_palette() -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#111827"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#f8fafc"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#0f172a"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#1f2937"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#f8fafc"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#1f2937"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#f8fafc"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#2563eb"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    return palette


def run() -> int:
    application = QApplication.instance() or QApplication(sys.argv)
    application.setStyle("Fusion")
    application.setPalette(_dark_palette())
    window = AmericaPerAmericaApp()
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(run())