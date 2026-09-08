"""Native PySide6 application router for Camera Utilities.

Pages own their section widgets and expose a small lifecycle protocol.  The
router never measures child widgets or animates manual window geometry; Qt's
layouts and stacked widgets own sizing.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QCloseEvent, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QProxyStyle,
    QSizePolicy,
    QStackedWidget,
    QStyle,
    QVBoxLayout,
    QWidget,
)


class AccessibleCheckBoxStyle(QProxyStyle):
    """High-contrast checkbox indicators used throughout the application."""

    _INDICATOR_SIZE = 18

    def pixelMetric(self, metric, option=None, widget=None):  # noqa: N802 - Qt API
        if metric in (
            QStyle.PixelMetric.PM_IndicatorWidth,
            QStyle.PixelMetric.PM_IndicatorHeight,
        ):
            return self._INDICATOR_SIZE
        return super().pixelMetric(metric, option, widget)

    def drawPrimitive(self, element, option, painter, widget=None):  # noqa: N802 - Qt API
        if element != QStyle.PrimitiveElement.PE_IndicatorCheckBox:
            return super().drawPrimitive(element, option, painter, widget)

        state = option.state
        enabled = bool(state & QStyle.StateFlag.State_Enabled)
        checked = bool(state & QStyle.StateFlag.State_On)
        partial = bool(state & QStyle.StateFlag.State_NoChange)
        active = checked or partial
        rect = option.rect.adjusted(1, 1, -1, -1)

        if not enabled:
            fill = QColor("#E5E7EB")
            border = QColor("#9CA3AF")
            mark = QColor("#6B7280")
        elif active:
            fill = QColor("#D1FAE5")
            border = QColor("#15803D")
            mark = QColor("#064E3B")
        else:
            fill = option.palette.base().color()
            border = QColor("#6B7280")
            mark = QColor("#064E3B")

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(fill)
        painter.setPen(QPen(border, 1.8))
        painter.drawRoundedRect(rect, 3.0, 3.0)

        if active:
            pen = QPen(mark, 2.4)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            if partial:
                y = rect.center().y()
                painter.drawLine(
                    int(rect.left() + rect.width() * 0.24),
                    y,
                    int(rect.right() - rect.width() * 0.24),
                    y,
                )
            else:
                path = QPainterPath()
                path.moveTo(
                    rect.left() + rect.width() * 0.20,
                    rect.top() + rect.height() * 0.53,
                )
                path.lineTo(
                    rect.left() + rect.width() * 0.42,
                    rect.top() + rect.height() * 0.74,
                )
                path.lineTo(
                    rect.left() + rect.width() * 0.80,
                    rect.top() + rect.height() * 0.27,
                )
                painter.drawPath(path)

        painter.restore()


@dataclass(frozen=True)
class FooterAction:
    text: str
    callback: Callable[[], None]
    bind: Callable[[QPushButton], Callable[[], None] | None] | None = None


class SectionedPage(QWidget):
    """Base class for a main page with a page-owned submenu."""

    title = "Page"

    def __init__(
        self,
        sections: Mapping[str, QWidget] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.sections: dict[str, QWidget] = {}
        self.section_stack = QStackedWidget()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.section_stack)
        self._active_section_name = ""
        if sections:
            self.set_sections(sections)

    def set_sections(self, sections: Mapping[str, QWidget]) -> None:
        if not sections:
            raise ValueError("A SectionedPage needs at least one section.")
        while self.section_stack.count():
            self.section_stack.removeWidget(self.section_stack.widget(0))
        self.sections = dict(sections)
        for widget in self.sections.values():
            self.section_stack.addWidget(widget)
        self._active_section_name = next(iter(self.sections))
        self.show_section(self._active_section_name)

    @property
    def section_names(self) -> tuple[str, ...]:
        return tuple(self.sections)

    def show_section(self, name: str) -> None:
        if name not in self.sections:
            raise KeyError(name)
        previous = self._active_section_name
        if previous != name:
            self.on_section_hide(previous)
        self._active_section_name = name
        self.section_stack.setCurrentWidget(self.sections[name])
        self.on_section_show(name)

    def footer_action(self) -> FooterAction | None:
        return None

    def on_show(self) -> None:
        pass

    def on_hide(self) -> None:
        pass

    def on_section_show(self, _name: str) -> None:
        pass

    def on_section_hide(self, _name: str) -> None:
        pass

    def shutdown(self) -> None:
        pass


class CalibrationPage(SectionedPage):
    title = "Calibrate"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)
        from calibrateGUI_pyside import CalibrateGui

        self.calPage = CalibrateGui(self)
        self.set_sections({
            "Configuration": self.calPage.config_frame,
            "Images": self.calPage.image_frame,
            "Config": self.calPage.advanced_frame,
            "Cal Result": self.calPage.cal_frame,
        })

    def footer_action(self) -> FooterAction:
        def start_calibration() -> None:
            self.calPage.calibrate_buttonCallback()

        def bind(button: QPushButton) -> Callable[[], None]:
            def render(calculating: bool) -> None:
                button.setText("Calibrating…" if calculating else "Start Calibration")
                button.setEnabled(not calculating)

            self.calPage.calibrationStateChanged.connect(render)
            render(bool(self.calPage.calculating))

            def unbind() -> None:
                try:
                    self.calPage.calibrationStateChanged.disconnect(render)
                except (RuntimeError, TypeError):
                    pass

            return unbind

        return FooterAction("Start Calibration", start_calibration, bind)

    def on_show(self) -> None:
        self.calPage.set_ui_active(True)

    def on_hide(self) -> None:
        self.calPage.set_ui_active(False)

    def on_section_show(self, name: str) -> None:
        self.calPage.on_section_show(name)

    def on_section_hide(self, name: str) -> None:
        self.calPage.on_section_hide(name)

    def shutdown(self) -> None:
        self.calPage.on_app_close()


class CameraPage(SectionedPage):
    title = "Camera"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)
        from cameraGUI_pyside import CameraGui

        self.camGui = CameraGui(self)
        self.set_sections({
            "Filepaths": self.camGui.filepath_page,
            "Image Processing": self.camGui.image_processing_page,
            "Export": self.camGui.export_frame,
            "Playback": self.camGui.playback_frame,
            "Data Processing": self.camGui.data_frame,
            "Hotkeys": self.camGui.hotkey_page,
        })

    def footer_action(self) -> FooterAction:
        def toggle() -> None:
            running = self.camGui.startStreamToggle()
            self.camGui.filepath_page.update_buttonsForStream(running)

        def bind(button: QPushButton) -> Callable[[], None]:
            def render(value=None) -> None:
                running = bool(
                    self.camGui.stream_running_var.get()
                    if value is None else value
                )
                button.setText("Stop Camera" if running else "Start Camera")
                button.setProperty("running", running)
                self.camGui.filepath_page.update_buttonsForStream(running)

            self.camGui.stream_running_var.changed.connect(render)
            render()

            def unbind() -> None:
                try:
                    self.camGui.stream_running_var.changed.disconnect(render)
                except (RuntimeError, TypeError):
                    pass

            return unbind

        return FooterAction("Start Camera", toggle, bind)

    def on_show(self) -> None:
        self.camGui.shutting_down = False
        self.camGui.set_ui_active(True)

    def on_hide(self) -> None:
        self.camGui.recordOff()
        self.camGui.startStreamOffBool()
        self.camGui.set_ui_active(False)

    def on_section_show(self, name: str) -> None:
        self.camGui.on_section_show(name)

    def shutdown(self) -> None:
        self.camGui.on_app_close()


class NavigationPane(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 12, 10, 12)
        self.layout.setSpacing(6)
        self.layout.addStretch(1)

    def clear(self) -> None:
        # Footer actions are intentionally placed after the stretch so they sit
        # at the bottom of the navigation pane.  Consequently, the last layout
        # item is not guaranteed to be the stretch.  Remove everything and
        # rebuild that spacer instead of preserving the final item.
        while self.layout.count():
            item = self.layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.deleteLater()
        self.layout.addStretch(1)

    def add_heading(self, text: str) -> None:
        label = QLabel(text)
        font = label.font()
        font.setBold(True)
        font.setPointSize(12)
        label.setFont(font)
        self.layout.insertWidget(self.layout.count() - 1, label)

    def add_button(self, text: str, callback: Callable[[], None]) -> QPushButton:
        button = QPushButton(text)
        button.setCheckable(True)
        button.clicked.connect(callback)
        self.layout.insertWidget(self.layout.count() - 1, button)
        return button


class App(QMainWindow):
    def __init__(
        self,
        pages: Mapping[str, SectionedPage],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if not pages:
            raise ValueError("App needs at least one main page.")
        self.pages = dict(pages)
        self._active_page_name: str | None = None
        self._main_buttons: dict[str, QPushButton] = {}
        self._section_buttons: dict[str, QPushButton] = {}
        self._footer_unbind: Callable[[], None] | None = None

        self.setWindowTitle("Camera Utilities by Jarvis")
        self.resize(1280, 800)

        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.main_nav = NavigationPane()
        self.main_nav.setFixedWidth(150)
        self.main_nav.add_heading("Main Menu")
        self.sub_nav = NavigationPane()
        self.sub_nav.setMinimumWidth(180)
        self.sub_nav.setMaximumWidth(260)
        self.content = QStackedWidget()
        self.content.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        root.addWidget(self.main_nav)
        root.addWidget(self.sub_nav)
        root.addWidget(self.content, 1)
        self.setCentralWidget(central)

        for name, page in self.pages.items():
            self.content.addWidget(page)
            self._main_buttons[name] = self.main_nav.add_button(
                name,
                lambda _checked=False, page_name=name: self.show_page(page_name),
            )

        self.show_page(next(iter(self.pages)))

    def show_page(self, name: str) -> None:
        if name not in self.pages:
            raise KeyError(name)
        if self._active_page_name == name:
            return
        if self._active_page_name is not None:
            self._unbind_footer()
            self.pages[self._active_page_name].on_hide()

        self._active_page_name = name
        page = self.pages[name]
        self.content.setCurrentWidget(page)
        page.on_show()

        for page_name, button in self._main_buttons.items():
            button.setChecked(page_name == name)
        self._build_subnav(page)

    def _build_subnav(self, page: SectionedPage) -> None:
        self._unbind_footer()
        self.sub_nav.clear()
        self.sub_nav.add_heading(page.title)
        self._section_buttons = {}
        for section_name in page.section_names:
            button = self.sub_nav.add_button(
                section_name,
                lambda _checked=False, value=section_name: self._show_section(page, value),
            )
            self._section_buttons[section_name] = button

        action = page.footer_action()
        if action is not None:
            footer = QPushButton(action.text)
            footer.clicked.connect(action.callback)
            if action.bind is not None:
                self._footer_unbind = action.bind(footer)
            self.sub_nav.layout.addWidget(footer)

        self._show_section(page, page._active_section_name)

    def _show_section(self, page: SectionedPage, name: str) -> None:
        page.show_section(name)
        for section_name, button in self._section_buttons.items():
            button.setChecked(section_name == name)

    def _unbind_footer(self) -> None:
        unbind = self._footer_unbind
        self._footer_unbind = None
        if unbind is not None:
            unbind()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        self._unbind_footer()
        for page in self.pages.values():
            page.shutdown()
        event.accept()


def run(pages: Mapping[str, SectionedPage] | None = None) -> int:
    application = QApplication.instance() or QApplication(sys.argv)
    application.setStyle(AccessibleCheckBoxStyle("Fusion"))
    resolved_pages = (
        dict(pages)
        if pages is not None
        else {
            "Camera": CameraPage(),
            "Calibrate": CalibrationPage(),
        }
    )
    window = App(resolved_pages)
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(run())
