import cv2
import numpy as np
import time
from screeninfo import get_monitors
import support.viz.colors as clr

class Checkerboard:
    def __init__(self):
        # -------------------- Monitor detection --------------------

        # Default fallback
        self.screen_width = 1920
        self.screen_height = 1080

        self.xy_num_squares = [12, 9]

        self._is_fullScreen = True

        self._refresh_monitors()

        self.lp_freq_track = 20.0
        self.last_toggle = 0.0

    def _refresh_monitors(self):
        # Keep a stable list; fall back to a fake primary if screeninfo is odd.
        self.monitors = list(get_monitors()) or []
        if not self.monitors:
            class _M: pass
            m = _M()
            m.x, m.y, m.width, m.height, m.is_primary = 0, 0, self.screen_width, self.screen_height, True
            self.monitors = [m]

        # Choose primary if possible; otherwise 0
        self.monitor_idx = 0
        for i, m in enumerate(self.monitors):
            if getattr(m, "is_primary", False):
                self.monitor_idx = i
                break

        # Sync dimensions to current monitor
        m = self.monitors[self.monitor_idx]
        self.screen_width = int(m.width)
        self.screen_height = int(m.height)

    def _apply_monitor_geometry(self, win_name: str):
        """Move the OpenCV window to the current monitor and fullscreen it."""
        m = self.monitors[self.monitor_idx]
        x, y = int(getattr(m, "x", 0)), int(getattr(m, "y", 0))

        # Important: move/size while NOT fullscreen, then fullscreen.
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        cv2.moveWindow(win_name, x, y)
        cv2.resizeWindow(win_name, int(m.width), int(m.height))
        if self._is_fullScreen:
            cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Sync dims for buffer rebuilds
        self.screen_width = int(m.width)
        self.screen_height = int(m.height)

    def _next_monitor(self, win_name: str):
        """Advance monitor index, move window, and return new (w,h)."""
        if not hasattr(self, "monitors"):
            self._refresh_monitors()

        self.monitor_idx = (self.monitor_idx + 1) % len(self.monitors)
        self._apply_monitor_geometry(win_name)
        return self.screen_width, self.screen_height


    # -------------------- Pattern generation --------------------
    def make_checkerboard(self, width, height):
        """
        Create a checkerboard image of size (height, width).
        squares_x, squares_y: number of squares horizontally/vertically.
        """
        # Base pattern: 0/1 checkerboard at low resolution
        pattern = 1 - np.add.outer(np.arange(self.xy_num_squares[1]),
                                   np.arange(self.xy_num_squares[0])) % 2

        # Scale pattern so each square fills equal region in pixels
        tile_w = max(1, width  // self.xy_num_squares[0])
        tile_h = max(1, height // self.xy_num_squares[1])
        pattern = pattern.repeat(tile_h, axis=0).repeat(tile_w, axis=1)

        # Crop to exact size (in case width/height not divisible)
        pattern = pattern[:height, :width]

        img = (pattern * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    # -------------------- HUD helpers --------------------


    def build_timer_frames(self, checker_a, checker_b, flash_freq_hz, period, screen_width, screen_height):
        """
        Build HUD-annotated versions of checker_a and checker_b for temporary display.
        """
        timer_frame_a = checker_a.copy()
        timer_frame_b = checker_b.copy()

        org1 = (int(screen_width * 0.1), int(screen_height * 0.05))
        org2 = (int(screen_width * 0.1), int(screen_height * 0.10))

        freq_text = f'Freq(Hz): {flash_freq_hz:.2f} / ({self.lp_freq_track:.2f})'
        period_text = f'Period(s): {period:.4f}'

        # Draw on A
        cv2.putText(timer_frame_a, freq_text, org1,
                    cv2.FONT_HERSHEY_PLAIN, 2.0, clr.WHITE, 10)
        cv2.putText(timer_frame_a, freq_text, org1,
                    cv2.FONT_HERSHEY_PLAIN, 2.0, clr.BLACK, 6)
        cv2.putText(timer_frame_a, freq_text, org1,
                    cv2.FONT_HERSHEY_PLAIN, 2.0, clr.LIGHTBLUE, 2)
        cv2.putText(timer_frame_a, period_text, org2,
                    cv2.FONT_HERSHEY_PLAIN, 2.0, clr.WHITE, 10)
        cv2.putText(timer_frame_a, period_text, org2,
                    cv2.FONT_HERSHEY_PLAIN, 2.0, clr.BLACK, 6)
        cv2.putText(timer_frame_a, period_text, org2,
                    cv2.FONT_HERSHEY_PLAIN, 2.0, clr.LIGHTBLUE, 2)

        # Draw on B (same text)
        cv2.putText(timer_frame_b, freq_text, org1,
                    cv2.FONT_HERSHEY_PLAIN, 2.0, clr.WHITE, 10)
        cv2.putText(timer_frame_b, freq_text, org1,
                    cv2.FONT_HERSHEY_PLAIN, 2.0, clr.BLACK, 6)
        cv2.putText(timer_frame_b, freq_text, org1,
                    cv2.FONT_HERSHEY_PLAIN, 2.0, clr.LIGHTBLUE, 2)
        cv2.putText(timer_frame_b, period_text, org2,
                    cv2.FONT_HERSHEY_PLAIN, 2.0, clr.WHITE, 10)
        cv2.putText(timer_frame_b, period_text, org2,
                    cv2.FONT_HERSHEY_PLAIN, 2.0, clr.BLACK, 6)
        cv2.putText(timer_frame_b, period_text, org2,
                    cv2.FONT_HERSHEY_PLAIN, 2.0, clr.LIGHTBLUE, 2)

        return timer_frame_a, timer_frame_b

    def build_squares_frames(self,
                             checker_a,
                             checker_b,
                             screen_width,
                             screen_height,
                             now,
                             hud_duration = 5.0):
        timer_frame_a = checker_a.copy()
        timer_frame_b = checker_b.copy()

        org1 = (int(screen_width * 0.1), int(screen_height * 0.05))
        org2 = (int(screen_width * 0.1), int(screen_height * 0.10))

        instr_text_a = f'{self.xy_num_squares[0] - 1} inner row corners'
        instr_text_b = f'{self.xy_num_squares[1] - 1} inner col corners'

        # Draw on A
        cv2.putText(timer_frame_a, instr_text_a, org1,
                    cv2.FONT_HERSHEY_PLAIN, 2.0, clr.WHITE, 10)
        cv2.putText(timer_frame_a, instr_text_a, org1,
                    cv2.FONT_HERSHEY_PLAIN, 2.0, clr.BLACK, 6)
        cv2.putText(timer_frame_a, instr_text_a, org1,
                    cv2.FONT_HERSHEY_PLAIN, 2.0, clr.LIGHTBLUE, 2)
        cv2.putText(timer_frame_a, instr_text_b, org2,
                    cv2.FONT_HERSHEY_PLAIN, 2.0, clr.WHITE, 10)
        cv2.putText(timer_frame_a, instr_text_b, org2,
                    cv2.FONT_HERSHEY_PLAIN, 2.0, clr.BLACK, 6)
        cv2.putText(timer_frame_a, instr_text_b, org2,
                    cv2.FONT_HERSHEY_PLAIN, 2.0, clr.LIGHTBLUE, 2)

        # Draw on B (same text)
        cv2.putText(timer_frame_b, instr_text_a, org1,
                    cv2.FONT_HERSHEY_PLAIN, 2.0, clr.WHITE, 10)
        cv2.putText(timer_frame_b, instr_text_a, org1,
                    cv2.FONT_HERSHEY_PLAIN, 2.0, clr.BLACK, 6)
        cv2.putText(timer_frame_b, instr_text_a, org1,
                    cv2.FONT_HERSHEY_PLAIN, 2.0, clr.LIGHTBLUE, 2)
        cv2.putText(timer_frame_b, instr_text_b, org2,
                    cv2.FONT_HERSHEY_PLAIN, 2.0, clr.WHITE, 10)
        cv2.putText(timer_frame_b, instr_text_b, org2,
                    cv2.FONT_HERSHEY_PLAIN, 2.0, clr.BLACK, 6)
        cv2.putText(timer_frame_b, instr_text_b, org2,
                    cv2.FONT_HERSHEY_PLAIN, 2.0, clr.LIGHTBLUE, 2)

        display_until = now + hud_duration

        return display_until, timer_frame_a, timer_frame_b

    @staticmethod
    def build_instr_frames(checker_a,
                           checker_b,
                           screen_width,
                           screen_height,
                           now,
                           hud_duration = 5.0):
        """
        Build instruction-annotated versions of checker_a and checker_b for temporary display.
        """
        timer_frame_a = checker_a.copy()
        timer_frame_b = checker_b.copy()

        org1 = (int(screen_width * 0.1), int(screen_height * 0.05))
        org2 = (int(screen_width * 0.1), int(screen_height * 0.10))
        org3 = (int(screen_width * 0.1), int(screen_height * 0.15))

        instr_text_a = 'a/d: freq | n: next monitor | Esc/q: quit'
        instr_text_b = 'Numpad-Rows/Cols | f: fullscreen'
        instr_text_c = 'Space to flash (for EBS)'

        # Draw on A
        cv2.putText(timer_frame_a, instr_text_a, org1,
                    cv2.FONT_HERSHEY_PLAIN, 2.0, clr.WHITE, 10)
        cv2.putText(timer_frame_a, instr_text_a, org1,
                    cv2.FONT_HERSHEY_PLAIN, 2.0, clr.BLACK, 6)
        cv2.putText(timer_frame_a, instr_text_a, org1,
                    cv2.FONT_HERSHEY_PLAIN, 2.0, clr.LIGHTBLUE, 2)
        cv2.putText(timer_frame_a, instr_text_b, org2,
                    cv2.FONT_HERSHEY_PLAIN, 2.0, clr.WHITE, 10)
        cv2.putText(timer_frame_a, instr_text_b, org2,
                    cv2.FONT_HERSHEY_PLAIN, 2.0, clr.BLACK, 6)
        cv2.putText(timer_frame_a, instr_text_b, org2,
                    cv2.FONT_HERSHEY_PLAIN, 2.0, clr.LIGHTBLUE, 2)
        cv2.putText(timer_frame_a, instr_text_c, org3,
                    cv2.FONT_HERSHEY_PLAIN, 2.0, clr.WHITE, 10)
        cv2.putText(timer_frame_a, instr_text_c, org3,
                    cv2.FONT_HERSHEY_PLAIN, 2.0, clr.BLACK, 6)
        cv2.putText(timer_frame_a, instr_text_c, org3,
                    cv2.FONT_HERSHEY_PLAIN, 2.0, clr.LIGHTBLUE, 2)

        # Draw on B (same text)
        cv2.putText(timer_frame_b, instr_text_a, org1,
                    cv2.FONT_HERSHEY_PLAIN, 2.0, clr.WHITE, 10)
        cv2.putText(timer_frame_b, instr_text_a, org1,
                    cv2.FONT_HERSHEY_PLAIN, 2.0, clr.BLACK, 6)
        cv2.putText(timer_frame_b, instr_text_a, org1,
                    cv2.FONT_HERSHEY_PLAIN, 2.0, clr.LIGHTBLUE, 2)
        cv2.putText(timer_frame_b, instr_text_b, org2,
                    cv2.FONT_HERSHEY_PLAIN, 2.0, clr.WHITE, 10)
        cv2.putText(timer_frame_b, instr_text_b, org2,
                    cv2.FONT_HERSHEY_PLAIN, 2.0, clr.BLACK, 6)
        cv2.putText(timer_frame_b, instr_text_b, org2,
                    cv2.FONT_HERSHEY_PLAIN, 2.0, clr.LIGHTBLUE, 2)
        cv2.putText(timer_frame_b, instr_text_c, org3,
                    cv2.FONT_HERSHEY_PLAIN, 2.0, clr.WHITE, 10)
        cv2.putText(timer_frame_b, instr_text_c, org3,
                    cv2.FONT_HERSHEY_PLAIN, 2.0, clr.BLACK, 6)
        cv2.putText(timer_frame_b, instr_text_c, org3,
                    cv2.FONT_HERSHEY_PLAIN, 2.0, clr.LIGHTBLUE, 2)

        display_until = now + hud_duration

        return display_until, timer_frame_a, timer_frame_b

    def adjust_frequency_and_build_hud(self,
        flash_freq_hz,
        delta_hz,
        checker_a,
        checker_b,
        screen_width,
        screen_height,
        now,
        min_freq=1.0,
        hud_duration=5.0,
    ):
        """
        Adjust flash_freq_hz by delta_hz (clamped at min_freq),
        recompute period, rebuild HUD frames, and compute display_until.
        """
        flash_freq_hz = max(min_freq, flash_freq_hz + delta_hz)
        period = 1.0 / flash_freq_hz
        display_until = now + hud_duration

        timer_frame_a, timer_frame_b = self.build_timer_frames(
            checker_a, checker_b, flash_freq_hz, period, screen_width, screen_height
        )

        return flash_freq_hz, period, display_until, timer_frame_a, timer_frame_b

    # -------------------- Main loop --------------------

    def run_checkerboard(self, on_close):

        flash_freq_hz = 20.0
        period = 1.0 / flash_freq_hz

        # Build base patterns
        checker_a = self.make_checkerboard(self.screen_width,
                                           self.screen_height)
        # White frame for B (checkerboard vs full white)
        checker_b = np.ones_like(checker_a) * 255

        # Initial HUD frames (not shown until display_until is set)
        timer_frame_a, timer_frame_b = self.build_timer_frames(
            checker_a, checker_b, flash_freq_hz, period, self.screen_width, self.screen_height
        )
        display_until = None

        # Setup fullscreen window
        win_name = "Fast Checkerboard"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        self._apply_monitor_geometry(win_name)

        # Timing and state for A/B toggling
        start = time.perf_counter()
        last_time = start
        next_toggle = start + period
        use_a = True

        rebuild = False

        # Stats
        time_a = 0.0
        time_b = 0.0
        frames_a = 0
        frames_b = 0

        pause = True
        last_frame_id = None

        alpha = 0.50
        last_toggle_actual = None

        while True:
            did_toggle = False

            now = time.perf_counter()
            dt = now - last_time
            last_time = now

            # Accumulate time in the *current* state (this state was visible since last loop)
            if use_a:
                time_a += dt
            else:
                time_b += dt

            # Catch up on toggles if we fell behind
            while now >= next_toggle:
                use_a = not use_a
                did_toggle = True
                next_toggle += period

            if did_toggle:
                if last_toggle_actual is not None:
                    measured_period = now - last_toggle_actual
                    if measured_period > 1e-6:
                        inst_hz = 1.0 / measured_period
                        self.lp_freq_track = (1.0 - alpha) * self.lp_freq_track + alpha * inst_hz
                last_toggle_actual = now

            # Choose which frame to display (HUD vs plain)
            show_hud = (display_until is not None and now < display_until)
            if show_hud:
                frame = timer_frame_a if (pause or use_a) else timer_frame_b
            else:
                frame = checker_a if (pause or use_a) else checker_b



            # when choosing frame:
            frame_id = ("hud" if show_hud else "plain", "A" if (pause or use_a) else "B",
                        flash_freq_hz, tuple(self.xy_num_squares), self.monitor_idx)

            if frame_id != last_frame_id:
                cv2.imshow(win_name, frame)
                last_frame_id = frame_id

            # Frame counters
            if use_a:
                frames_a += 1
            else:
                frames_b += 1

            # How long until we *need* to flip A/B?
            dt_to_toggle = next_toggle - now
            # Keep UI responsive; don't block too long even if toggle is far away
            max_block_ms = 10
            block_ms = 1
            if dt_to_toggle > 0:
                block_ms = max(1, min(max_block_ms, int(dt_to_toggle * 1000)))

            key = cv2.waitKey(block_ms) & 0xFF

            # If we didn't block all the way to the toggle, optionally yield a little more
            # (waitKey already yields, so this can be tiny or omitted)
            if dt_to_toggle > (block_ms / 1000.0):
                time.sleep(0.0)

            if key == ord('a'):  # decrease frequency
                flash_freq_hz, period, display_until, timer_frame_a, timer_frame_b = self.adjust_frequency_and_build_hud(
                    flash_freq_hz,
                    -1.0,
                    checker_a,
                    checker_b,
                    self.screen_width,
                    self.screen_height,
                    now,
                )
                next_toggle = now + period  # reset schedule relative to "now"

            elif key == ord('d'):  # increase frequency
                flash_freq_hz, period, display_until, timer_frame_a, timer_frame_b = self.adjust_frequency_and_build_hud(
                    flash_freq_hz,
                    +1.0,
                    checker_a,
                    checker_b,
                    self.screen_width,
                    self.screen_height,
                    now,
                )
                next_toggle = now + period

            elif key in (27, ord('q')):  # ESC or 'q'
                break

            # ---- NEW: Arrow key handling for checker count ----
            # On most OpenCV builds:

            # --- NUMPAD-BASED CHECKER RESOLUTION CONTROL ---

            elif key == ord('4'):  # Numpad 4: fewer columns
                if self.xy_num_squares[0] > 3:
                    self.xy_num_squares[0] -= 1
                rebuild = True

            elif key == ord('6'):  # Numpad 6: more columns
                self.xy_num_squares[0] += 1
                rebuild = True

            elif key == ord('8'):  # Numpad 8: more rows
                self.xy_num_squares[1] += 1
                rebuild = True

            elif key == ord('2'):  # Numpad 2: fewer rows
                if self.xy_num_squares[1] > 3:
                    self.xy_num_squares[1] -= 1
                rebuild = True

            elif key == ord('n'):
                # Hop window to next monitor, then rebuild buffers to match new resolution
                self._next_monitor(win_name)

                checker_a = self.make_checkerboard(self.screen_width, self.screen_height)
                checker_b = np.ones_like(checker_a) * 255

                # Rebuild HUD frames too (use updated dims)
                display_until, timer_frame_a, timer_frame_b = self.build_instr_frames(
                    checker_a, checker_b, self.screen_width, self.screen_height, now)
                display_until = now + 2.0  # brief HUD flash so user knows it moved


            elif key == 32:
                pause = not pause

            elif key == ord('f'):
                self._is_fullScreen = not self._is_fullScreen
                self._apply_monitor_geometry(win_name)

            # Any other key (non-255): show instructions overlay for a bit
            elif key != 255:
                display_until, timer_frame_a, timer_frame_b = self.build_instr_frames(
                    checker_a, checker_b, self.screen_width, self.screen_height, now)

            if rebuild:
                # Clamp to something sane to avoid zero-sized tiles
                squares_x = max(2, min(self.xy_num_squares[0], self.screen_width))
                squares_y = max(2, min(self.xy_num_squares[1], self.screen_height))

                checker_a = self.make_checkerboard(self.screen_width, self.screen_height)
                checker_b = np.ones_like(checker_a) * 255

                # Refresh HUD variants too
                display_until, timer_frame_a, timer_frame_b = self.build_squares_frames(
                    checker_a, checker_b, self.screen_width, self.screen_height, now)
                rebuild = False
                last_frame_id = None

        cv2.destroyWindow(win_name)
        on_close()

if __name__ == "__main__":
    def close_lambda():
        print('Closed!')
    new_board = Checkerboard()
    new_board.run_checkerboard(close_lambda)
