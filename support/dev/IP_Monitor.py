import ipaddress
import json
from pathlib import Path
import re
import subprocess
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import ttk

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
        text = result.stdout
        match = re.search(r"192\.168\.168\.\d+", text)
        return match.group(0) if match else None
    except Exception:
        return None


def ping_host(ip):
    try:
        result = run_command(["ping", "-n", "1", "-w", str(PING_TIMEOUT_MS), ip])
        return result.returncode == 0
    except Exception:
        return False


class PingMonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("192.168.168 Network Monitor")
        self.root.geometry("760x620")
        self.root.minsize(700, 420)
        self.root.resizable(True, True)

        self.running = True
        self.poll_in_progress = False
        self.refresh_anim_job = None
        self.refresh_anim_step = 0
        self.local_ip = None

        self.status_var = tk.StringVar(value="Checking local subnet...")
        self.last_update_var = tk.StringVar(value="Last update: --")
        self.custom_name_var = tk.StringVar()
        self.custom_ip_var = tk.StringVar()
        self.form_status_var = tk.StringVar(value="")
        self.aircraft_stats = {}
        self.stats_update_job = None

        self.next_device_id = 1
        self.devices = []
        self.device_rows = {}

        for aircraft in AIRCRAFT:
            self.aircraft_stats[aircraft] = {
                "engine_start_var": tk.StringVar(),
                "takeoff_var": tk.StringVar(),
                "land_var": tk.StringVar(),
                "engine_time_var": tk.StringVar(value="--"),
                "flight_time_var": tk.StringVar(value="--"),
                "status_var": tk.StringVar(value=""),
            }

        for name, ip in DEVICES:
            self.devices.append(self.make_device(name, ip, custom=False))
        self.load_custom_devices()

        self.build_ui()
        self.schedule_stats_update()
        self.schedule_refresh(initial=True)

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
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)

        header = ttk.Label(
            main,
            text="Device Status Monitor",
            font=("Segoe UI", 14, "bold")
        )
        header.pack(anchor="w", pady=(0, 8))

        status_label = ttk.Label(main, textvariable=self.status_var, font=("Segoe UI", 10))
        status_label.pack(anchor="w", pady=(0, 4))

        last_update_label = ttk.Label(main, textvariable=self.last_update_var, font=("Segoe UI", 9))
        last_update_label.pack(anchor="w", pady=(0, 10))

        form = ttk.LabelFrame(main, text="Add Custom IP", padding=10)
        form.pack(fill="x", pady=(0, 10))

        ttk.Label(form, text="Name").grid(row=0, column=0, sticky="w", padx=(0, 8))
        name_entry = ttk.Entry(form, textvariable=self.custom_name_var, width=24)
        name_entry.grid(row=1, column=0, sticky="w", padx=(0, 12))

        ttk.Label(form, text="IP Address").grid(row=0, column=1, sticky="w", padx=(0, 8))
        ip_entry = ttk.Entry(form, textvariable=self.custom_ip_var, width=20)
        ip_entry.grid(row=1, column=1, sticky="w", padx=(0, 12))

        add_btn = ttk.Button(form, text="Add", command=self.add_custom_device)
        add_btn.grid(row=1, column=2, sticky="w", padx=(0, 12))

        ttk.Label(form, textvariable=self.form_status_var, foreground="#aa0000").grid(
            row=2, column=0, columnspan=3, sticky="w", pady=(8, 0)
        )

        name_entry.bind("<Return>", lambda _event: self.add_custom_device())
        ip_entry.bind("<Return>", lambda _event: self.add_custom_device())

        self.build_aircraft_stats_panel(main)

        button_row = ttk.Frame(main)
        button_row.pack(side="bottom", fill="x", pady=(16, 0))

        refresh_btn = ttk.Button(button_row, text="Refresh Now", command=self.manual_refresh)
        refresh_btn.pack(side="left")

        quit_btn = ttk.Button(button_row, text="Quit", command=self.close)
        quit_btn.pack(side="right")

        table_container = ttk.Frame(main)
        table_container.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(table_container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(table_container, orient="vertical", command=self.canvas.yview)
        self.table = ttk.Frame(self.canvas)

        self.table.bind(
            "<Configure>",
            lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.table_window = self.canvas.create_window((0, 0), window=self.table, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.bind(
            "<Configure>",
            lambda event: self.canvas.itemconfigure(self.table_window, width=event.width)
        )

        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.render_device_table()

    def build_aircraft_stats_panel(self, parent):
        stats_frame = ttk.LabelFrame(parent, text="Flight Test Stats", padding=10)
        stats_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(
            stats_frame,
            text=f"{TIME_INPUT_HINT}. Use Now buttons to stamp current local time.",
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(0, 8))

        for aircraft in AIRCRAFT:
            stats = self.aircraft_stats[aircraft]
            aircraft_frame = ttk.Frame(stats_frame)
            aircraft_frame.pack(fill="x", pady=3)
            aircraft_frame.columnconfigure(1, weight=1)

            ttk.Label(aircraft_frame, text=aircraft, font=("Segoe UI", 10, "bold"), width=12).grid(
                row=0, column=0, rowspan=2, sticky="nw", padx=(0, 10), pady=(2, 0)
            )

            input_row = ttk.Frame(aircraft_frame)
            input_row.grid(row=0, column=1, sticky="ew")

            ttk.Label(input_row, text="Engine Start").pack(side="left", padx=(0, 4))
            engine_entry = ttk.Entry(input_row, textvariable=stats["engine_start_var"], width=14)
            engine_entry.pack(side="left", padx=(0, 4))

            ttk.Button(
                input_row,
                text="Now",
                width=5,
                command=lambda aircraft=aircraft: self.stamp_aircraft_time(aircraft, "engine_start_var"),
            ).pack(side="left", padx=(0, 10))

            ttk.Label(input_row, text="Take-off").pack(side="left", padx=(0, 4))
            takeoff_entry = ttk.Entry(input_row, textvariable=stats["takeoff_var"], width=14)
            takeoff_entry.pack(side="left", padx=(0, 4))

            ttk.Button(
                input_row,
                text="Now",
                width=5,
                command=lambda aircraft=aircraft: self.stamp_aircraft_time(aircraft, "takeoff_var"),
            ).pack(side="left", padx=(0, 10))

            ttk.Label(input_row, text="Land").pack(side="left", padx=(0, 4))
            land_entry = ttk.Entry(input_row, textvariable=stats["land_var"], width=14)
            land_entry.pack(side="left", padx=(0, 4))

            ttk.Button(
                input_row,
                text="Now",
                width=5,
                command=lambda aircraft=aircraft: self.stamp_aircraft_time(aircraft, "land_var"),
            ).pack(side="left")

            metric_row = ttk.Frame(aircraft_frame)
            metric_row.grid(row=1, column=1, sticky="ew", pady=(4, 0))

            ttk.Label(metric_row, text="Engine Time", font=("Segoe UI", 10, "bold")).pack(side="left", padx=(0, 4))
            ttk.Label(metric_row, textvariable=stats["engine_time_var"], width=12).pack(
                side="left", padx=(0, 14)
            )
            ttk.Label(metric_row, text="Flight Time", font=("Segoe UI", 10, "bold")).pack(side="left", padx=(0, 4))
            ttk.Label(metric_row, textvariable=stats["flight_time_var"], width=12).pack(
                side="left", padx=(0, 14)
            )
            ttk.Label(metric_row, textvariable=stats["status_var"], foreground="#aa0000").pack(
                side="left", fill="x", expand=True
            )

            engine_entry.bind("<Return>", lambda _event: self.update_aircraft_stats())
            takeoff_entry.bind("<Return>", lambda _event: self.update_aircraft_stats())
            land_entry.bind("<Return>", lambda _event: self.update_aircraft_stats())

    def render_device_table(self):
        for child in self.table.winfo_children():
            child.destroy()

        self.device_rows = {}

        ttk.Label(self.table, text="Device", font=("Segoe UI", 10, "bold"), width=24).grid(row=0, column=0, sticky="w", padx=(0, 12), pady=(0, 6))
        ttk.Label(self.table, text="IP Address", font=("Segoe UI", 10, "bold"), width=18).grid(row=0, column=1, sticky="w", padx=(0, 12), pady=(0, 6))
        ttk.Label(self.table, text="Status", font=("Segoe UI", 10, "bold"), width=12).grid(row=0, column=2, sticky="w", padx=(0, 12), pady=(0, 6))
        ttk.Label(self.table, text="Action", font=("Segoe UI", 10, "bold"), width=10).grid(row=0, column=3, sticky="w", pady=(0, 6))

        for i, device in enumerate(self.devices, start=1):
            ttk.Label(self.table, text=device["name"], width=24).grid(row=i, column=0, sticky="w", padx=(0, 12), pady=3)
            ttk.Label(self.table, text=device["ip"], width=18).grid(row=i, column=1, sticky="w", padx=(0, 12), pady=3)

            status = tk.Label(
                self.table,
                text="--",
                width=12,
                anchor="w",
                fg="black"
            )
            status.grid(row=i, column=2, sticky="w", padx=(0, 12), pady=3)

            if device["custom"]:
                action = ttk.Button(
                    self.table,
                    text="Remove",
                    command=lambda device_id=device["id"]: self.remove_custom_device(device_id)
                )
            else:
                action = ttk.Label(self.table, text="Built-in", width=10)
            action.grid(row=i, column=3, sticky="w", pady=3)

            self.device_rows[device["id"]] = status

    def set_device_status(self, device_id, text, color):
        label = self.device_rows.get(device_id)
        if label is None:
            return
        label.config(text=text, fg=color)

    def stamp_aircraft_time(self, aircraft, field_name):
        self.aircraft_stats[aircraft][field_name].set(datetime.now().strftime("%H:%M:%S"))
        self.update_aircraft_stats()

    def schedule_stats_update(self):
        if not self.running:
            return

        self.update_aircraft_stats()
        self.stats_update_job = self.root.after(1000, self.schedule_stats_update)

    def update_aircraft_stats(self):
        now = datetime.now()

        for aircraft in AIRCRAFT:
            stats = self.aircraft_stats[aircraft]
            status_messages = []

            try:
                engine_start = parse_event_time(stats["engine_start_var"].get())
            except ValueError:
                engine_start = None
                status_messages.append("Invalid engine start")

            try:
                takeoff = parse_event_time(stats["takeoff_var"].get())
            except ValueError:
                takeoff = None
                status_messages.append("Invalid take-off")

            try:
                land = parse_event_time(stats["land_var"].get())
            except ValueError:
                land = None
                status_messages.append("Invalid land")

            elapsed_end = land if land is not None else now

            if engine_start is None:
                stats["engine_time_var"].set("--")
            else:
                stats["engine_time_var"].set(format_elapsed(engine_start, elapsed_end))

            if takeoff is None:
                stats["flight_time_var"].set("--")
            else:
                stats["flight_time_var"].set(format_elapsed(takeoff, elapsed_end))

            if engine_start and takeoff and takeoff < engine_start:
                status_messages.append("Take-off before engine start")
            if engine_start and land and land < engine_start:
                status_messages.append("Land before engine start")
            if takeoff and land and land < takeoff:
                status_messages.append("Land before take-off")

            stats["status_var"].set("; ".join(status_messages))

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
            self.form_status_var.set("Could not load saved custom IPs.")

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
            self.form_status_var.set("Could not save custom IPs.")

    def add_custom_device(self):
        name = self.custom_name_var.get().strip()
        ip = self.custom_ip_var.get().strip()

        if not name:
            self.form_status_var.set("Enter a name for the custom device.")
            return

        try:
            ipaddress.ip_address(ip)
        except ValueError:
            self.form_status_var.set("Enter a valid IP address.")
            return

        for device in self.devices:
            if device["ip"] == ip:
                self.form_status_var.set("That IP is already being monitored.")
                return

        self.devices.append(self.make_device(name, ip, custom=True))
        self.save_custom_devices()
        self.form_status_var.set("")
        self.custom_name_var.set("")
        self.custom_ip_var.set("")
        self.render_device_table()

        if not self.poll_in_progress:
            self.schedule_refresh(initial=False)

    def remove_custom_device(self, device_id):
        self.devices = [device for device in self.devices if device["id"] != device_id]
        self.save_custom_devices()
        self.form_status_var.set("")
        self.render_device_table()

    def manual_refresh(self):
        if not self.poll_in_progress:
            self.schedule_refresh(initial=False)

    def start_refresh_animation(self):
        self.refresh_anim_step = 0
        self.animate_refresh_text()

    def animate_refresh_text(self):
        if not self.running or not self.poll_in_progress:
            return

        dots = "." * (self.refresh_anim_step % 4)
        if self.local_ip:
            self.status_var.set(f"Detected local address on subnet: {self.local_ip}   Refreshing{dots}")
        else:
            self.status_var.set(f"Checking local subnet{dots}")

        self.refresh_anim_step += 1
        self.refresh_anim_job = self.root.after(300, self.animate_refresh_text)

    def stop_refresh_animation(self):
        if self.refresh_anim_job is not None:
            self.root.after_cancel(self.refresh_anim_job)
            self.refresh_anim_job = None

    def schedule_refresh(self, initial=False):
        if not self.running or self.poll_in_progress:
            return

        self.poll_in_progress = True
        self.start_refresh_animation()
        threading.Thread(target=self.refresh_worker, daemon=True).start()

    def refresh_worker(self):
        local_ip = get_local_subnet_ip()
        results = []

        devices = [(device["id"], device["name"], device["ip"]) for device in self.devices]

        for device_id, name, ip in devices:
            up = ping_host(ip)
            results.append((device_id, name, ip, up))

        self.root.after(0, lambda: self.finish_refresh(local_ip, results))

    def finish_refresh(self, local_ip, results):
        self.stop_refresh_animation()
        self.local_ip = local_ip
        if local_ip:
            self.status_var.set(f"Detected local address on subnet: {local_ip}")
        else:
            self.status_var.set("No local 192.168.168.x address detected")

        for device_id, _, _, up in results:
            if up:
                self.set_device_status(device_id, "UP", "green")
            else:
                self.set_device_status(device_id, "DOWN", "red")

        self.last_update_var.set(f"Last update: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        self.poll_in_progress = False
        if self.running:
            self.root.after(REFRESH_MS, self.schedule_refresh)

    def close(self):
        self.running = False
        self.stop_refresh_animation()
        if self.stats_update_job is not None:
            self.root.after_cancel(self.stats_update_job)
            self.stats_update_job = None
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = PingMonitorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()
