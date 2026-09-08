from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QLabel, QSizePolicy, QWidget


class HotkeyPage(QWidget):
    """Read-only folder replay hotkey reference."""

    ITEMS = (
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
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QGridLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setHorizontalSpacing(18)
        layout.setVerticalSpacing(4)

        title = QLabel("Folder Replay Hotkeys")
        title.setProperty("heading", True)
        title_font = title.font()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title, 0, 0, 1, 2)

        for column, text in enumerate(("Key", "Action")):
            heading = QLabel(text)
            font = heading.font()
            font.setBold(True)
            heading.setFont(font)
            layout.addWidget(heading, 1, column)

        for row, (key, description) in enumerate(self.ITEMS, start=2):
            key_label = QLabel(key)
            key_label.setAlignment(Qt.AlignmentFlag.AlignTop)
            description_label = QLabel(description)
            description_label.setWordWrap(True)
            description_label.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Preferred,
            )
            layout.addWidget(key_label, row, 0)
            layout.addWidget(description_label, row, 1)

        layout.setColumnStretch(1, 1)
        layout.setRowStretch(layout.rowCount(), 1)


# Transitional alias for import sites that still use the old class spelling.
Hotkey_page = HotkeyPage

