# Testing.py
# Minimal CustomTkinter "pages + router" example
# pip install customtkinter

from tkinter import TclError
import customtkinter as ctk
import superCalibrateCamera as cam
import superCalibrate as calibrate

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

        # define "sections" as frames we can swap
        self.sections = {
            "Configuration": self._make_setupPage(),
            "Images": self._make_ImagePage(),
            "Cal Result": self._make_calibrationPage(),
        }
        # show default section
        show_section(self, "Configuration")

    # ----- section UIs -----
    def _make_ImagePage(self):
        return self.calPage.setup_imageFrame(self)

    def _make_setupPage(self):
        return self.calPage.setup_configFrame(self)

    def _make_calibrationPage(self):
        return self.calPage.setup_CalFrame(self)


class CameraPage(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)

        self.title = "Camera"

        self.camGui = cam.CameraGui(self)

        self.sections = {
            "Configuration": self._make_setupPage(),
            "Image Processing": self._make_ImgProcPage(),
            "Export": self._make_exportPage(),
            "Hotkeys": self._make_hotkeyPage(),
        }
        show_section(self, "Configuration")

    def _make_setupPage(self):
        return self.camGui.cam_frame
    def _make_ImgProcPage(self):
        return self.camGui.config_frame
    def _make_exportPage(self):
        return self.camGui.export_frame
    def _make_hotkeyPage(self):
        return self.camGui.hotkey_frame

    def submenu_footer(self):
        def on_toggle(btn: ctk.CTkButton):
            # call your existing toggle
            running = self.camGui.startStreamToggle()

            # update UI to reflect state
            if running:
                btn.configure(text="Stop Camera", fg_color="royalblue4", hover_color="blue")
            else:
                btn.configure(text="Start Camera", fg_color=GREEN, hover_color=DEFAULT_HOVER)

        return ("Start Camera", on_toggle)

# ------------ App / Router with two sidebars ------------

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.active_main_button = None
        self.active_sub_button = None
        self.sub_buttons = {}

        self.title("Two-level Sidebar Demo")
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

        self._build_mainnav()

        self.current_page = None
        self.show_page("Camera")

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

    def _on_mainnav_click(self, name):
        self.show_page(name)

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

    def show_page(self, name):
        # swap content page
        if self.current_page:
            self.pages[self.current_page].grid_remove()
        page = self.pages[name]
        page.grid()
        self.current_page = name

        # highlight first so it paints immediately
        self._highlight_main_button(name)

        # then rebuild submenu (which triggers sizing)
        self._rebuild_subnav(page)

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
            text, cmd = footer()
            # pass the button to the page's handler
            self.subnav_footer_btn.configure(
                text=text,
                command=lambda fn=cmd, btn=self.subnav_footer_btn: fn(btn)
            )
            self.subnav_bottom.grid()
        else:
            self.subnav_bottom.grid_remove()

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

    def _fit_to_content(self, page, show_subnav: bool):
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

        # Page: use only active section width (unchanged)
        page_w = self._section_reqwidth(page)

        # Heights: take max of columns to keep things simple
        main_h = self.mainnav.winfo_reqheight()
        sub_h = self.subnav.winfo_reqheight() if show_subnav else 0
        page_h = page.winfo_reqheight()

        # Modest window chrome allowance (keep small to avoid bloat)
        BORDER_W, BORDER_H = 12, 12

        total_w = main_w + sub_w + page_w + BORDER_W
        total_h = max(main_h, sub_h, page_h) + BORDER_H

        # Ensure parents propagate sizes
        self.content.grid_propagate(True)
        page.grid_propagate(True)

        curr_w, curr_h = self.winfo_width(), self.winfo_height()

        alpha = 0.5

        new_geom = f"{int(alpha*total_w + (1.0-alpha) * curr_w)}x{int(alpha * total_h + (1.0-alpha) * curr_h)}"
        self.geometry(new_geom)
        self.update()
        if abs(total_w - curr_w) > 10 or abs(total_h - curr_h) > 10:
            self.after(5, lambda: self._fit_to_content(page, show_subnav))

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

        return int(w)


if __name__ == "__main__":
    App().mainloop()