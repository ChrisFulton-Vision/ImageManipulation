import ipaddress
import json
import threading
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Qt, Signal, QMimeData
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QAbstractItemView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

REFRESH_MS = 2000
PING_TIMEOUT_MS = 800
CUSTOM_DEVICES_FILE = Path(__file__).with_name("IP_Monitor.custom_devices.json")
AIRCRAFT = ("Shadow", "Supersonic")
TIME_INPUT_HINT = "Use HH:MM, HH:MM:SS, or YYYY-MM-DD HH:MM"

DEVICES = [
    ("Microhard Antenna", "192.168.168.101"),
    ("SuperSonic Antenna", "192.168.168.104"),
    ("SuperSonic Thor", "192.168.168.114"),
    ("Shadow Antenna", "192.168.168.103"),
    ("Shadow Thor", "192.168.168.113"),
]


def parse_event_time(value):
    value = value.strip()
    if not value:
        return None

    now = datetime.now()
    formats = (
        "%H:%M",
        "%H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
    )
    for fmt in formats:
        try:
            parsed = datetime.strptime(value, fmt)
        except ValueError:
            continue

        if fmt.startswith("%H"):
            return parsed.replace(year=now.year, month=now.month, day=now.day)
        return parsed

    raise ValueError("Unsupported time format")


def format_elapsed(start_time, now):
    if start_time is None:
        return "--"

    elapsed_seconds = int((now - start_time).total_seconds())
    if elapsed_seconds < 0:
        return "starts in " + format_duration(abs(elapsed_seconds))

    return format_duration(elapsed_seconds)


def format_duration(total_seconds):
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def run_command(cmd):
    creationflags = 0
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        creationflags = subprocess.CREATE_NO_WINDOW

    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        creationflags=creationflags,
    )


def get_local_subnet_ip():
    try:
        result = run_command(["ipconfig"])
        match = re.search(r"192\.168\.168\.\d+", result.stdout)
        return match.group(0) if match else None
    except Exception:
        return None


def ping_host(ip):
    try:
        result = run_command(["ping", "-n", "1", "-w", str(PING_TIMEOUT_MS), ip])
        return result.returncode == 0
    except Exception:
        return False


class RefreshSignals(QObject):
    finished = Signal(object, object)


class RefreshWorker(QRunnable):
    def __init__(self, devices, shutdown_event):
        super().__init__()

        # Prevent QThreadPool from deleting the QRunnable underneath Python.
        self.setAutoDelete(False)

        self.devices = devices
        self.shutdown_event = shutdown_event
        self.signals = RefreshSignals()

    def run(self):
        if self.shutdown_event.is_set():
            return

        local_ip = get_local_subnet_ip()

        if self.shutdown_event.is_set():
            return

        results = []

        for device_id, name, ip in self.devices:
            if self.shutdown_event.is_set():
                return

            up = ping_host(ip)
            results.append((device_id, name, ip, up))

        if self.shutdown_event.is_set():
            return

        try:
            self.signals.finished.emit(local_ip, results)
        except RuntimeError:
            # Qt may already be tearing down its QObjects.
            if not self.shutdown_event.is_set():
                raise


class PingMonitorApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("192.168.168 Network Monitor")
        self.resize(620, 650)
        self.setMinimumSize(560, 460)

        self.running = True
        self.poll_in_progress = False
        self.refresh_anim_step = 0
        self.local_ip = None
        self.pending_form_status = ""

        self.next_device_id = 1
        self.devices = []
        self.device_rows = {}
        self.aircraft_stats = {}

        for name, ip in DEVICES:
            self.devices.append(self.make_device(name, ip, custom=False))
        self.load_custom_devices()

        self.shutdown_event = threading.Event()

        self.thread_pool = QThreadPool(self)

        # Strong Python references to every running QRunnable.
        self.active_workers = set()

        self.stats_timer = QTimer(self)
        self.stats_timer.setInterval(1000)
        self.stats_timer.timeout.connect(self.update_aircraft_stats)

        self.refresh_timer = QTimer(self)
        self.refresh_timer.setSingleShot(True)
        self.refresh_timer.timeout.connect(self.schedule_refresh)

        self.refresh_anim_timer = QTimer(self)
        self.refresh_anim_timer.setInterval(300)
        self.refresh_anim_timer.timeout.connect(self.animate_refresh_text)

        self.build_ui()
        self.stats_timer.start()
        self.update_aircraft_stats()
        self.schedule_refresh()

    def make_device(self, name, ip, custom):
        device = {
            "id": self.next_device_id,
            "name": name,
            "ip": ip,
            "custom": custom,
        }
        self.next_device_id += 1
        return device

    def build_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)

        main = QVBoxLayout(central)
        main.setContentsMargins(10, 10, 10, 10)
        main.setSpacing(7)

        header = QLabel("Device Status Monitor")
        header_font = QFont("Segoe UI", 14)
        header_font.setBold(True)
        header.setFont(header_font)
        main.addWidget(header)

        self.status_label = QLabel("Checking local subnet...")
        self.status_label.setFont(QFont("Segoe UI", 10))
        main.addWidget(self.status_label)

        self.last_update_label = QLabel("Last update: --")
        self.last_update_label.setFont(QFont("Segoe UI", 9))
        main.addWidget(self.last_update_label)

        main.addWidget(self.build_custom_ip_panel())
        main.addWidget(self.build_aircraft_stats_panel())

        self.device_table = QTableWidget(0, 4)
        self.device_table.setHorizontalHeaderLabels(("Device", "IP Address", "Status", "Action"))
        self.device_table.verticalHeader().setVisible(False)
        self.device_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.device_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.device_table.setAlternatingRowColors(True)
        self.device_table.setShowGrid(False)
        self.device_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        header_view = self.device_table.horizontalHeader()
        header_view.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header_view.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)

        main.addWidget(self.device_table, 1)
        self.render_device_table()

        button_row = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh Now")
        self.refresh_button.clicked.connect(self.manual_refresh)
        button_row.addWidget(self.refresh_button)
        button_row.addStretch(1)

        quit_button = QPushButton("Quit")
        quit_button.clicked.connect(self.close)
        button_row.addWidget(quit_button)
        main.addLayout(button_row)

        if self.pending_form_status:
            self.set_form_status(self.pending_form_status)

    def build_custom_ip_panel(self):
        group = QGroupBox("Add Custom IP")
        layout = QGridLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(4)
        layout.setColumnStretch(1, 1)

        self.custom_name_edit = QLineEdit()
        self.custom_ip_edit = QLineEdit()
        self.custom_name_edit.setPlaceholderText("Device name")
        self.custom_ip_edit.setPlaceholderText("192.168.168.x")

        add_button = QPushButton("Add")
        add_button.clicked.connect(self.add_custom_device)

        # Vertical form: Name and IP Address share the same label/input columns.
        layout.addWidget(QLabel("Name"), 0, 0)
        layout.addWidget(self.custom_name_edit, 0, 1)
        layout.addWidget(QLabel("IP Address"), 1, 0)
        layout.addWidget(self.custom_ip_edit, 1, 1)
        layout.addWidget(add_button, 0, 2, 2, 1)

        self.form_status_label = QLabel("")
        self.form_status_label.setStyleSheet("color: #aa0000;")
        layout.addWidget(self.form_status_label, 2, 0, 1, 3)

        self.custom_name_edit.returnPressed.connect(self.add_custom_device)
        self.custom_ip_edit.returnPressed.connect(self.add_custom_device)

        return group

    def build_aircraft_stats_panel(self):
        group = QGroupBox("Flight Test Stats")
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(8, 8, 8, 8)
        group_layout.setSpacing(5)

        control_row = QHBoxLayout()

        hint = QLabel(f"{TIME_INPUT_HINT}. Use Now buttons to stamp current local time.")
        hint.setFont(QFont("Segoe UI", 9))
        hint.setWordWrap(True)

        reset_button = QPushButton("Reset")
        reset_button.setMaximumWidth(65)
        reset_button.clicked.connect(self.reset_aircraft_stats)

        copy_button = QPushButton("Copy")
        copy_button.setMaximumWidth(65)
        copy_button.clicked.connect(self.copy_aircraft_stats)

        control_row.addWidget(hint, 1)
        control_row.addWidget(reset_button)
        control_row.addWidget(copy_button)

        group_layout.addLayout(control_row)

        for aircraft in AIRCRAFT:
            aircraft_frame = QFrame()
            aircraft_layout = QGridLayout(aircraft_frame)
            aircraft_layout.setContentsMargins(0, 2, 0, 2)
            aircraft_layout.setHorizontalSpacing(6)
            aircraft_layout.setVerticalSpacing(3)
            aircraft_layout.setColumnStretch(2, 1)

            name_label = QLabel(aircraft)
            name_font = QFont("Segoe UI", 10)
            name_font.setBold(True)
            name_label.setFont(name_font)
            name_label.setMinimumWidth(80)
            name_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
            aircraft_layout.addWidget(name_label, 0, 0, 4, 1)

            engine_edit = QLineEdit()
            engine_edit.setMaximumWidth(135)
            engine_now = QPushButton("Now")
            engine_now.setMaximumWidth(50)

            takeoff_edit = QLineEdit()
            takeoff_edit.setMaximumWidth(135)
            takeoff_now = QPushButton("Now")
            takeoff_now.setMaximumWidth(50)

            land_edit = QLineEdit()
            land_edit.setMaximumWidth(135)
            land_now = QPushButton("Now")
            land_now.setMaximumWidth(50)

            aircraft_layout.addWidget(QLabel("Engine Start"), 0, 1)
            aircraft_layout.addWidget(engine_edit, 0, 2)
            aircraft_layout.addWidget(engine_now, 0, 3)

            aircraft_layout.addWidget(QLabel("Take-off"), 1, 1)
            aircraft_layout.addWidget(takeoff_edit, 1, 2)
            aircraft_layout.addWidget(takeoff_now, 1, 3)

            aircraft_layout.addWidget(QLabel("Land"), 2, 1)
            aircraft_layout.addWidget(land_edit, 2, 2)
            aircraft_layout.addWidget(land_now, 2, 3)

            metric_row = QHBoxLayout()
            metric_row.setContentsMargins(0, 0, 0, 0)
            metric_row.setSpacing(5)

            engine_metric_title = QLabel("Engine Time")
            engine_metric_title.setFont(name_font)
            engine_time_label = QLabel("--")
            engine_time_label.setMinimumWidth(72)

            flight_metric_title = QLabel("Flight Time")
            flight_metric_title.setFont(name_font)
            flight_time_label = QLabel("--")
            flight_time_label.setMinimumWidth(72)

            status_label = QLabel("")
            status_label.setStyleSheet("color: #aa0000;")

            metric_row.addWidget(engine_metric_title)
            metric_row.addWidget(engine_time_label)
            metric_row.addSpacing(6)
            metric_row.addWidget(flight_metric_title)
            metric_row.addWidget(flight_time_label)
            metric_row.addSpacing(6)
            metric_row.addWidget(status_label, 1)

            aircraft_layout.addLayout(metric_row, 3, 1, 1, 3)
            group_layout.addWidget(aircraft_frame)

            self.aircraft_stats[aircraft] = {
                "engine_start_edit": engine_edit,
                "takeoff_edit": takeoff_edit,
                "land_edit": land_edit,
                "engine_time_label": engine_time_label,
                "flight_time_label": flight_time_label,
                "status_label": status_label,
            }

            engine_now.clicked.connect(
                lambda _checked=False, a=aircraft: self.stamp_aircraft_time(a, "engine_start_edit")
            )
            takeoff_now.clicked.connect(
                lambda _checked=False, a=aircraft: self.stamp_aircraft_time(a, "takeoff_edit")
            )
            land_now.clicked.connect(
                lambda _checked=False, a=aircraft: self.stamp_aircraft_time(a, "land_edit")
            )

            engine_edit.returnPressed.connect(self.update_aircraft_stats)
            takeoff_edit.returnPressed.connect(self.update_aircraft_stats)
            land_edit.returnPressed.connect(self.update_aircraft_stats)

        return group

    def render_device_table(self):
        self.device_table.setRowCount(len(self.devices))
        self.device_rows = {}

        for row, device in enumerate(self.devices):
            name_item = QTableWidgetItem(device["name"])
            ip_item = QTableWidgetItem(device["ip"])
            status_item = QTableWidgetItem("--")
            status_item.setForeground(QColor("black"))
            status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            self.device_table.setItem(row, 0, name_item)
            self.device_table.setItem(row, 1, ip_item)
            self.device_table.setItem(row, 2, status_item)

            if device["custom"]:
                action = QPushButton("Remove")
                action.clicked.connect(
                    lambda _checked=False, device_id=device["id"]: self.remove_custom_device(device_id)
                )
                self.device_table.setCellWidget(row, 3, action)
            else:
                built_in = QLabel("Built-in")
                built_in.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.device_table.setCellWidget(row, 3, built_in)

            self.device_rows[device["id"]] = status_item

        self.device_table.resizeRowsToContents()

    def set_device_status(self, device_id, text, color):
        item = self.device_rows.get(device_id)
        if item is None:
            return

        item.setText(text)
        item.setForeground(QColor(color))

    def stamp_aircraft_time(self, aircraft, field_name):
        self.aircraft_stats[aircraft][field_name].setText(datetime.now().strftime("%H:%M:%S"))
        self.update_aircraft_stats()

    def reset_aircraft_stats(self):
        for aircraft in AIRCRAFT:
            stats = self.aircraft_stats[aircraft]

            stats["engine_start_edit"].clear()
            stats["takeoff_edit"].clear()
            stats["land_edit"].clear()

        self.update_aircraft_stats()

    def copy_aircraft_stats(self):
        headers = [
            "Aircraft",
            "Engine Start",
            "Take-off",
            "Land",
            "Engine Time",
            "Flight Time",
        ]

        rows = []

        for aircraft in AIRCRAFT:
            stats = self.aircraft_stats[aircraft]

            rows.append([
                aircraft,
                stats["engine_start_edit"].text().strip(),
                stats["takeoff_edit"].text().strip(),
                stats["land_edit"].text().strip(),
                stats["engine_time_label"].text(),
                stats["flight_time_label"].text(),
            ])

        # Plain-text fallback: useful for Excel, Notepad, etc.
        text_rows = [headers] + rows
        plain_text = "\n".join(
            "\t".join(row)
            for row in text_rows
        )

        # Rich HTML version: OneNote should recognize this as a real table.
        html = """
        <table border="1" cellspacing="0" cellpadding="4">
            <thead>
                <tr>
        """

        for header in headers:
            html += f"<th>{header}</th>"

        html += """
                </tr>
            </thead>
            <tbody>
        """

        for row in rows:
            html += "<tr>"
            for value in row:
                html += f"<td>{value}</td>"
            html += "</tr>"

        html += """
            </tbody>
        </table>
        """

        mime_data = QMimeData()
        mime_data.setText(plain_text)
        mime_data.setHtml(html)

        QApplication.clipboard().setMimeData(mime_data)

    def update_aircraft_stats(self):
        now = datetime.now()

        for aircraft in AIRCRAFT:
            stats = self.aircraft_stats[aircraft]
            status_messages = []

            try:
                engine_start = parse_event_time(stats["engine_start_edit"].text())
            except ValueError:
                engine_start = None
                status_messages.append("Invalid engine start")

            try:
                takeoff = parse_event_time(stats["takeoff_edit"].text())
            except ValueError:
                takeoff = None
                status_messages.append("Invalid take-off")

            try:
                land = parse_event_time(stats["land_edit"].text())
            except ValueError:
                land = None
                status_messages.append("Invalid land")

            elapsed_end = land if land is not None else now

            stats["engine_time_label"].setText(
                "--" if engine_start is None else format_elapsed(engine_start, elapsed_end)
            )
            stats["flight_time_label"].setText(
                "--" if takeoff is None else format_elapsed(takeoff, elapsed_end)
            )

            if engine_start and takeoff and takeoff < engine_start:
                status_messages.append("Take-off before engine start")
            if engine_start and land and land < engine_start:
                status_messages.append("Land before engine start")
            if takeoff and land and land < takeoff:
                status_messages.append("Land before take-off")

            stats["status_label"].setText("; ".join(status_messages))

    def load_custom_devices(self):
        try:
            if not CUSTOM_DEVICES_FILE.exists():
                return

            with CUSTOM_DEVICES_FILE.open("r", encoding="utf-8") as fh:
                saved_devices = json.load(fh)

            for item in saved_devices:
                name = str(item["name"]).strip()
                ip = str(item["ip"]).strip()
                if not name:
                    continue

                ipaddress.ip_address(ip)
                if all(device["ip"] != ip for device in self.devices):
                    self.devices.append(self.make_device(name, ip, custom=True))
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            self.pending_form_status = "Could not load saved custom IPs."

    def save_custom_devices(self):
        saved_devices = [
            {"name": device["name"], "ip": device["ip"]}
            for device in self.devices
            if device["custom"]
        ]

        try:
            with CUSTOM_DEVICES_FILE.open("w", encoding="utf-8") as fh:
                json.dump(saved_devices, fh, indent=2)
        except OSError:
            self.set_form_status("Could not save custom IPs.")

    def set_form_status(self, text):
        self.pending_form_status = text
        if hasattr(self, "form_status_label"):
            self.form_status_label.setText(text)

    def add_custom_device(self):
        name = self.custom_name_edit.text().strip()
        ip = self.custom_ip_edit.text().strip()

        if not name:
            self.set_form_status("Enter a name for the custom device.")
            return

        try:
            ipaddress.ip_address(ip)
        except ValueError:
            self.set_form_status("Enter a valid IP address.")
            return

        for device in self.devices:
            if device["ip"] == ip:
                self.set_form_status("That IP is already being monitored.")
                return

        self.devices.append(self.make_device(name, ip, custom=True))
        self.save_custom_devices()
        self.set_form_status("")
        self.custom_name_edit.clear()
        self.custom_ip_edit.clear()
        self.render_device_table()

        if not self.poll_in_progress:
            self.schedule_refresh()

    def remove_custom_device(self, device_id):
        self.devices = [device for device in self.devices if device["id"] != device_id]
        self.save_custom_devices()
        self.set_form_status("")
        self.render_device_table()

    def manual_refresh(self):
        if not self.poll_in_progress:
            self.refresh_timer.stop()
            self.schedule_refresh()

    def start_refresh_animation(self):
        self.refresh_anim_step = 0
        self.animate_refresh_text()
        self.refresh_anim_timer.start()

    def animate_refresh_text(self):
        if not self.running or not self.poll_in_progress:
            return

        dots = "." * (self.refresh_anim_step % 4)
        if self.local_ip:
            self.status_label.setText(
                f"Detected local address on subnet: {self.local_ip}   Refreshing{dots}"
            )
        else:
            self.status_label.setText(f"Checking local subnet{dots}")

        self.refresh_anim_step += 1

    def stop_refresh_animation(self):
        self.refresh_anim_timer.stop()

    def schedule_refresh(self):
        if not self.running or self.poll_in_progress:
            return

        self.poll_in_progress = True
        self.refresh_button.setEnabled(False)
        self.start_refresh_animation()

        devices = [
            (device["id"], device["name"], device["ip"])
            for device in self.devices
        ]

        worker = RefreshWorker(devices, self.shutdown_event)

        self.active_workers.add(worker)

        worker.signals.finished.connect(self.finish_refresh)

        worker.signals.finished.connect(
            lambda _local_ip, _results, worker=worker:
            self.active_workers.discard(worker)
        )

        self.thread_pool.start(worker)

    def finish_refresh(self, local_ip, results):
        if not self.running:
            return

        self.stop_refresh_animation()
        self.local_ip = local_ip

        if local_ip:
            self.status_label.setText(f"Detected local address on subnet: {local_ip}")
        else:
            self.status_label.setText("No local 192.168.168.x address detected")

        for device_id, _, _, up in results:
            self.set_device_status(device_id, "UP" if up else "DOWN", "green" if up else "red")

        self.last_update_label.setText(
            f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        self.poll_in_progress = False
        self.refresh_button.setEnabled(True)
        self.refresh_timer.start(REFRESH_MS)

    def closeEvent(self, event):
        self.running = False
        self.shutdown_event.set()

        self.stats_timer.stop()
        self.refresh_timer.stop()
        self.refresh_anim_timer.stop()

        # Don't start anything else.
        self.thread_pool.clear()

        # Allow the currently executing ping to return before Qt starts
        # destroying the signal QObjects.
        self.thread_pool.waitForDone()

        self.active_workers.clear()

        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = PingMonitorApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()