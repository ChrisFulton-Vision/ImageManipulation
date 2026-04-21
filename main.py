# Testing.py
# Minimal CustomTkinter "pages + router" example
# pip install customtkinter

from tkinter import TclError
import customtkinter as ctk
import superCalibrateCamera as cam
import superCalibrate as calibrate
from functools import partial
# from SupportModules.LidarTruth import TruthPoints
ctk.deactivate_automatic_dpi_awareness()
GREEN = '#2FA572'
DEFAULT_HOVER = ('#0C955A', '#106A43')

# ----- page-owned submenu -----
def build_submenu(master_frame, parent, on_click):
    ctk.CTkLabel(parent, text=master_frame.title, font=("Segoe UI", 16, "bold")).pack(
        padx=12, pady=(16, 8), anchor="w"
    )

    buttons = {}
    for name in master_frame.sections.keys():
        btn = ctk.CTkButton(parent, text=name, command=lambda n=name: on_click(n))
        btn.pack(fill="x", padx=12, pady=6)
        buttons[name] = btn
    return buttons


def _is_alive(widget) -> bool:
    try:
        return bool(widget) and widget.winfo_exists()
    except TclError:
        return False

# ----- section swapping -----
def show_section(master_frame, name):
    if hasattr(master_frame, "_active_section_name"):
        prev = master_frame._active_section_name
        if hasattr(master_frame, "on_section_hide"):
            try: master_frame.on_section_hide(prev)
            except Exception: pass

    if hasattr(master_frame, "_ensure_section"):
        master_frame._ensure_section(name)

    for f in master_frame.sections.values():
        if f is not None:
            f.grid_remove()

    frame = master_frame.sections[name]
    frame.grid(row=0, column=0, sticky="nsew")
    master_frame.grid_propagate(True)

    master_frame._active_section_name = name
    master_frame._active_section = frame

    if hasattr(master_frame, "on_section_show"):
        try: master_frame.on_section_show(name)
        except Exception: pass




# ------------ Pages ------------
class CalibratePage(ctk.CTkFrame):
    """This page exposes its own submenu (Sources / Parsing / Validation)."""
    def __init__(self, master):
        super().__init__(master)

        self.title = "Calibrate"
        # content area inside the page
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.calPage = calibrate.CalibrateGui(self)
        self.set_main_button_calculating = None

        # define "sections" as frames we can swap
        self.sections = {
            "Configuration": self._make_setupPage(),
            "Images": self._make_ImagePage(),
            "Config": self._make_configPage(),
            "Cal Result": self._make_calibrationPage(),
        }
        # show default section
        show_section(self, "Configuration")

    # ----- section UIs -----

    def _make_setupPage(self):
        return self.calPage.setup_configFrame(self)

    def _make_ImagePage(self):
        return self.calPage.setup_imageFrame(self)

    def _make_calibrationPage(self):
        return self.calPage.setup_CalFrame(self)

    def _make_configPage(self):
        return self.calPage.updateConfigWindow(self)

    def submenu_footer(self):
        def on_toggle(btn: ctk.CTkButton):
            # call your existing toggle
            if callable(self.set_main_button_calculating):
                self.set_main_button_calculating(True)
            running = self.calPage.calibrate_buttonCallback(self, btn)

            btn.configure(text="Calibrating", fg_color="royalblue4", hover_color="blue", state='disabled')
            # update UI to reflect state
            # if running:
            # else:
            #     btn.configure(text="Start Calibration", fg_color=GREEN, hover_color=DEFAULT_HOVER)

        return ("Start Calibration", on_toggle)

    def bind_main_button_state(self, callback):
        self.set_main_button_calculating = callback
        self.calPage.on_calibration_complete = lambda: callback(False)

    def on_show(self):
        if hasattr(self.calPage, "set_ui_active"):
            self.calPage.set_ui_active(True)

    def on_hide(self):
        if hasattr(self.calPage, "set_ui_active"):
            self.calPage.set_ui_active(False)

    def on_section_show(self, name: str):
        if name == "Images":
            self.calPage.updateImageFrame()
        if hasattr(self.calPage, "on_section_show"):
            self.calPage.on_section_show(name)

    def on_section_hide(self, name: str):
        if hasattr(self.calPage, "on_section_hide"):
            self.calPage.on_section_hide(name)


class CameraPage(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)

        self.title = "Camera"

        self.camGui = cam.CameraGui(self)

        self.sections = {
            "Filepaths": self._make_setupPage(),
            "Image Processing": self._make_ImgProcPage(),
            "Export": self._make_exportPage(),
            "Playback": self._make_playbackPage(),
            "Data Processing": self._make_dataPage(),
            "Hotkeys": self._make_hotkeyPage(),
        }
        show_section(self, "Filepaths")

    def _make_setupPage(self):
        return self.camGui.filepath_page
    def _make_ImgProcPage(self):
        return self.camGui.image_processing_page
    def _make_exportPage(self):
        return self.camGui.export_frame
    def _make_playbackPage(self):
        return self.camGui.playback_frame
    def _make_dataPage(self):
        return self.camGui.data_frame
    def _make_hotkeyPage(self):
        return self.camGui.hotkey_page

    def on_show(self):
        if hasattr(self.camGui, "set_ui_active"):
            self.camGui.set_ui_active(True)

    def on_hide(self):
        self.camGui.shutting_down = True
        self.camGui.recordOff()
        self.camGui.startStreamOffBool()
        if hasattr(self.camGui, "set_ui_active"):
            self.camGui.set_ui_active(False)

    def on_section_show(self, name: str):
        if hasattr(self.camGui, "on_section_show"):
            self.camGui.on_section_show(name)

    def on_section_hide(self, name: str):
        if hasattr(self.camGui, "on_section_hide"):
            self.camGui.on_section_hide(name)

    def submenu_footer(self):
        def render(btn: ctk.CTkButton):
            if getattr(btn, "_footer_owner", None) is not self:
                return

            running = bool(self.camGui.stream_running_var.get())
            if running:
                btn.configure(text="Stop Camera", fg_color="royalblue4", hover_color="blue")
            else:
                btn.configure(text="Start Camera", fg_color=GREEN, hover_color=DEFAULT_HOVER)

            self.camGui.filepath_page.update_buttonsForStream(running)

        def on_toggle(btn: ctk.CTkButton):
            self.camGui.startStreamToggle()
            # no UI updates here; trace will do it

        # IMPORTANT: this runs when the footer button gets created (it exists by then)
        def bind_footer(btn: ctk.CTkButton):
            render(btn)
            self.camGui.stream_running_var.trace_add("write", lambda *_: render(btn))

        # Return the handler *and* a binder
        return ("Start Camera", on_toggle, bind_footer)

# ------------ App / Router with two sidebars ------------

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")

        self.active_main_button = None
        self.active_sub_button = None
        self.sub_buttons = {}

        self.title("Camera Utilities by Jarvis")
        # self.geometry("1200x750")

        # Grid: [MainNav | SubNav | Content]
        self.columnconfigure(0, weight=0)  # main nav fixed
        self.columnconfigure(1, weight=0)  # sub nav fixed (shown/hidden)
        self.columnconfigure(2, weight=1)  # content expands
        self.rowconfigure(0, weight=1)

        # Main sidebar (global pages)
        self.mainnav = ctk.CTkFrame(self, fg_color=("gray12", "gray12"))
        self.mainnav.grid(row=0, column=0, sticky="ns")

        # Sub sidebar (page-provided submenu)
        self.subnav = ctk.CTkFrame(self, fg_color=("gray10", "gray10"))

        # after creating self.subnav ...
        self.subnav.rowconfigure(0, weight=1)  # top expands
        self.subnav.rowconfigure(1, weight=0)  # bottom stays fixed

        self.subnav_top = ctk.CTkFrame(self.subnav, fg_color="transparent")
        self.subnav_top.grid(row=0, column=0, sticky="nsew")

        self.subnav_bottom = ctk.CTkFrame(self.subnav, fg_color="transparent")
        self.subnav_bottom.grid(row=1, column=0, sticky="ew")

        # persistent button (shown/hidden or re-bound per page)
        self.subnav_footer_btn = ctk.CTkButton(self.subnav_bottom, text="Action", command=lambda: None)
        self.subnav_footer_btn.pack(fill="x", padx=12, pady=12)

        # initially hidden; will be shown when a page provides a submenu
        self.subnav.grid(row=0, column=1, sticky="ns")
        self.subnav.grid_remove()
        self.subnav.grid_propagate(True)

        # Content area
        self.content = ctk.CTkFrame(self)
        self.content.grid(row=0, column=2, sticky="nsew")
        self.content.rowconfigure(0, weight=1)
        self.content.columnconfigure(0, weight=1)

        # Pages
        self.pages = {
            "Calibrate": CalibratePage(self.content),
            "Camera": CameraPage(self.content),
        }
        for p in self.pages.values():
            p.grid(row=0, column=0, sticky="nsew")
            p.grid_remove()
            if hasattr(p, 'camGui'):
                p.camGui.func_to_refit(self.passToChild_fit_to_content(p))
            if hasattr(p, 'calPage'):
                p.calPage.func_to_refit(self.passToChild_fit_to_content(p))

        self.protocol("WM_DELETE_WINDOW", self.pages["Camera"].camGui.on_app_close)

        self._build_mainnav()
        self.current_page = None
        self.show_page("Camera")

        self.after_idle(self._initial_fit_once)

    def _initial_fit_once(self, tries: int = 0):
        # Let geometry settle
        self.update()
        page = self.pages.get(self.current_page)
        if not page:
            # first call happens right after show_page("Camera")
            self.after(16, self._initial_fit_once, tries + 1)
            return

        # Wait until widgets report non-trivial requested size
        if page.winfo_reqwidth() <= 1 or page.winfo_reqheight() <= 1:
            if tries < 30:  # ~0.5s max (30 * 16ms)
                self.after(16, self._initial_fit_once, tries + 1)
            return

        # Fit including subnav if present
        show_sub = bool(getattr(page, "sections", None))
        self._fit_to_content(page, show_subnav=show_sub)

    def _build_mainnav(self):
        ctk.CTkLabel(self.mainnav, text="Main Menu", font=("Segoe UI", 18, "bold")).pack(
            padx=12, pady=(16, 8), anchor="w"
        )

        self.main_buttons = {}
        for name in self.pages.keys():
            btn = ctk.CTkButton(
                self.mainnav,
                text=name,
                command=lambda n=name: self._on_mainnav_click(n)
            )
            btn.pack(fill="x", padx=12, pady=6)
            self.main_buttons[name] = btn

        calibrate_page = self.pages.get("Calibrate")
        if hasattr(calibrate_page, "bind_main_button_state"):
            calibrate_page.bind_main_button_state(self._set_calibrate_main_button_calculating)

    def _on_mainnav_click(self, name):
        self.show_page(name)

    def _set_calibrate_main_button_calculating(self, calculating: bool):
        btn = self.main_buttons.get("Calibrate")
        if not _is_alive(btn):
            return

        if calculating:
            btn.configure(state="disabled", fg_color="royalblue4", hover_color="blue")
        else:
            btn.configure(state="normal", fg_color=GREEN, hover_color=DEFAULT_HOVER)
            if self.current_page == "Calibrate":
                self._highlight_main_button("Calibrate")

    def _highlight_main_button(self, name):
        # safely reset the previously active main button
        if _is_alive(self.active_main_button):
            try:
                self.active_main_button.configure(fg_color=GREEN, hover_color=DEFAULT_HOVER)
            except TclError:
                pass

        new_btn = self.main_buttons.get(name)
        if _is_alive(new_btn):
            try:
                new_btn.configure(fg_color="royalblue4", hover_color='blue')
                self.active_main_button = new_btn
            except TclError:
                self.active_main_button = None
        else:
            self.active_main_button = None

    def _highlight_sub_button(self, name):
        if _is_alive(self.active_sub_button):
            self.active_sub_button.configure(fg_color=GREEN, hover_color=DEFAULT_HOVER)

        new_btn = self.sub_buttons.get(name)  # safe now
        if _is_alive(new_btn):
            new_btn.configure(fg_color="royalblue4", hover_color='blue')
            self.active_sub_button = new_btn
        else:
            self.active_sub_button = None

    # --- in App.show_page ---
    def show_page(self, name):
        self.update_idletasks()
        if self.current_page and hasattr(self.pages[self.current_page], "on_hide"):
            try:
                self.pages[self.current_page].on_hide()
            except Exception:
                pass

        if self.current_page:
            self.pages[self.current_page].grid_remove()

        page = self.pages[name]
        page.grid()
        self.current_page = name
        self._highlight_main_button(name)
        self._rebuild_subnav(page)

        if hasattr(page, "on_show"):
            try:
                page.on_show()
            except Exception:
                pass

    # in App
    def _rebuild_subnav(self, page):
        # drop stale selection (old buttons are about to be destroyed)
        self.active_sub_button = None

        for w in self.subnav_top.winfo_children():
            w.destroy()

        has_sub = False
        self.sub_buttons = {}  # reset mapping

        if hasattr(page, "sections"):
            # SAVE the dict returned by build_submenu
            self.sub_buttons = build_submenu(page, self.subnav_top,
                                             on_click=self._on_subnav_click(page)) or {}
            has_sub = bool(self.sub_buttons)

        # configure the persistent bottom button from the page (if provided)
        footer = getattr(page, "submenu_footer", None)
        if callable(footer):
            out = footer()
            if len(out) == 2:
                text, cmd = out
                binder = None
            else:
                text, cmd, binder = out

            self.subnav_footer_btn._footer_owner = page
            self.subnav_footer_btn.configure(
                text=text,
                command=lambda fn=cmd, btn=self.subnav_footer_btn: fn(btn)
            )
            if binder is not None:
                binder(self.subnav_footer_btn)

        if has_sub:
            self.subnav.grid()
            active_name = getattr(page, "_active_section_name", None) or next(iter(page.sections))
            if active_name in self.sub_buttons:
                self._highlight_sub_button(active_name)  # paint before sizing
        else:
            self.subnav.grid_remove()

        self.update_idletasks()
        self._fit_to_content(page, show_subnav=has_sub)

    def _on_subnav_click(self, page):
        def handler(name):
            show_section(page, name)
            self._highlight_sub_button(name)
            self.update_idletasks()
            self._fit_to_content(page, show_subnav=True)

        return handler

    def passToChild_fit_to_content(self, page):
        return partial(self._fit_to_content, page, True)

    def _fit_to_content(self, page, show_subnav: bool,
                        smooth_transition_tuple: tuple[list, tuple[int, int]] = (None, (None, None))):
        # prevent overlapping animations
        if getattr(self, "_resize_inflight", False):
            return
        self._resize_inflight = True

        try:
            self.update_idletasks()
            if not self.mainnav.winfo_exists():
                return

            # Sidebars
            main_w = self.mainnav.winfo_reqwidth()
            if show_subnav:
                mapped = [w for w in self.subnav.winfo_children() if w.winfo_ismapped()]
                sub_w = max((w.winfo_reqwidth() for w in mapped), default=0)
            else:
                sub_w = 0

            # Page width/height
            page_w = self._section_reqwidth(page)
            main_h = self.mainnav.winfo_reqheight()
            sub_h = self.subnav.winfo_reqheight() if show_subnav else 0
            page_h = self._section_layout_height(page)

            BORDER_W, BORDER_H = 12, 12
            target_w = main_w + sub_w + page_w + BORDER_W
            target_h = max(main_h, sub_h, page_h) + BORDER_H

            # current geom
            curr_w, curr_h = self.winfo_width(), self.winfo_height()
            dw, dh = target_w - curr_w, target_h - curr_h

            # If already close, snap once and bail
            if abs(dw) + abs(dh) <= 12:
                self.geometry(f"{target_w}x{target_h}")
                return

            # Build a tiny easing list based on distance (2–3 steps max)
            steps = 2 if (abs(dw) + abs(dh) < 600) else 3
            alphas = [i / float(steps) for i in range(1, steps + 1)]  # e.g., [0.5, 1.0] or [0.33, 0.66, 1.0]

            def tick(i: int):
                a = alphas[i]
                w = int(curr_w + a * dw)
                h = int(curr_h + a * dh)
                self.geometry(f"{w}x{h}")
                self.update_idletasks()  # cheaper than full update
                if i + 1 < len(alphas):
                    self.after(12, tick, i + 1)

            tick(0)
        finally:
            # let the last scheduled after run before clearing; small delay prevents re-entrancy thrash
            self.after(50, lambda: setattr(self, "_resize_inflight", False))

    def _active_section(self, page):
        # Prefer explicitly recorded active section (see show_section)
        sec = getattr(page, "_active_section", None)
        if sec and sec.winfo_exists():
            return sec
        # Fallback: any mapped section, skipping Nones
        mapped = [f for f in getattr(page, "sections", {}).values()
                  if f is not None and f.winfo_ismapped()]
        return mapped[0] if mapped else page

    # inside App
    def _section_reqwidth(self, page):
        """Return the natural width of the *visible* section, with scrollable frames handled."""
        target = self._active_section(page)

        # If it's a CTkScrollableFrame, measure the inner content frame (no manual summing)
        inner = getattr(target, "_scrollable_frame", target)

        # Force Tk to compute requested sizes after any layout change
        inner.update_idletasks()

        w = inner.winfo_reqwidth()
        if w <= 1:  # very defensive fallback
            w = target.winfo_reqwidth()
        # Account for a possible vertical scrollbar gutter if present (kept tiny)
        scrollbar = getattr(target, "_scrollbar", None)
        if scrollbar and scrollbar.winfo_ismapped():
            try:
                w += scrollbar.winfo_reqwidth()
            except Exception:
                w += 16  # safe default

        return int(w + 30)

    def _section_layout_height(self, page) -> int:
        """
        Height based on the currently visible section only.
        This prevents hidden/stale sections from inflating the window height.
        """
        """Return the natural width of the *visible* section, with scrollable frames handled."""
        target = self._active_section(page)

        # If it's a CTkScrollableFrame, measure the inner content frame (no manual summing)
        inner = getattr(target, "_scrollable_frame", target)

        # Force Tk to compute requested sizes after any layout change
        inner.update_idletasks()

        h = inner.winfo_reqheight()
        if h <= 1:  # very defensive fallback
            h = target.winfo_reqheight()
        # Account for a possible vertical scrollbar gutter if present (kept tiny)
        scrollbar = getattr(target, "_scrollbar", None)
        if scrollbar and scrollbar.winfo_ismapped():
            try:
                h += scrollbar.winfo_reqheight()
            except Exception:
                h += 16  # safe default

        return int(h)


if __name__ == "__main__":
    App().mainloop()
