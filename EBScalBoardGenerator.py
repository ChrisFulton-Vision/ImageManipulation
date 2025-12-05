import cv2
import numpy as np
import time
from screeninfo import get_monitors

# -------------------- Monitor detection --------------------

# Default fallback
screen_width = 1920
screen_height = 1080

# Try to use primary monitor if available
for m in get_monitors():
    if getattr(m, "is_primary", False):
        screen_width = m.width
        screen_height = m.height
        break

# -------------------- Pattern generation --------------------

def make_checkerboard(width, height, squares_x=10, squares_y=6):
    """
    Create a checkerboard image of size (height, width).
    squares_x, squares_y: number of squares horizontally/vertically.
    """
    # Base pattern: 0/1 checkerboard at low resolution
    pattern = np.add.outer(np.arange(squares_y), np.arange(squares_x)) % 2

    # Scale pattern so each square fills equal region in pixels
    tile_w = max(1, width  // squares_x)
    tile_h = max(1, height // squares_y)
    pattern = pattern.repeat(tile_h, axis=0).repeat(tile_w, axis=1)

    # Crop to exact size (in case width/height not divisible)
    pattern = pattern[:height, :width]

    img = (pattern * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

# -------------------- HUD helpers --------------------

def build_timer_frames(checker_a, checker_b, flash_freq_hz, period, screen_width, screen_height):
    """
    Build HUD-annotated versions of checker_a and checker_b for temporary display.
    """
    timer_frame_a = checker_a.copy()
    timer_frame_b = checker_b.copy()

    org1 = (int(screen_width * 0.1), int(screen_height * 0.05))
    org2 = (int(screen_width * 0.1), int(screen_height * 0.10))

    freq_text = f'Freq(Hz): {flash_freq_hz:.2f}'
    period_text = f'Period(s): {period:.4f}'

    # Draw on A
    cv2.putText(timer_frame_a, freq_text, org1,
                cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 0), 4)
    cv2.putText(timer_frame_a, freq_text, org1,
                cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 0), 1)
    cv2.putText(timer_frame_a, period_text, org2,
                cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 0), 4)
    cv2.putText(timer_frame_a, period_text, org2,
                cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 0), 1)

    # Draw on B (same text)
    cv2.putText(timer_frame_b, freq_text, org1,
                cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 0), 4)
    cv2.putText(timer_frame_b, freq_text, org1,
                cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 0), 1)
    cv2.putText(timer_frame_b, period_text, org2,
                cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 0), 4)
    cv2.putText(timer_frame_b, period_text, org2,
                cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 0), 1)

    return timer_frame_a, timer_frame_b

def build_squares_frames(checker_a,
                       checker_b,
                       screen_width,
                       screen_height,
                       now,
                       squares_x,
                       squares_y,
                       hud_duration = 5.0):
    timer_frame_a = checker_a.copy()
    timer_frame_b = checker_b.copy()

    org1 = (int(screen_width * 0.1), int(screen_height * 0.05))
    org2 = (int(screen_width * 0.1), int(screen_height * 0.10))

    instr_text_a = f'{squares_x - 1} inner row corners'
    instr_text_b = f'{squares_y - 1} inner col corners'

    # Draw on A
    cv2.putText(timer_frame_a, instr_text_a, org1,
                cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 0), 4)
    cv2.putText(timer_frame_a, instr_text_a, org1,
                cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 0), 1)
    cv2.putText(timer_frame_a, instr_text_b, org2,
                cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 0), 4)
    cv2.putText(timer_frame_a, instr_text_b, org2,
                cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 0), 1)

    # Draw on B (same text)
    cv2.putText(timer_frame_b, instr_text_a, org1,
                cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 0), 4)
    cv2.putText(timer_frame_b, instr_text_a, org1,
                cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 0), 1)
    cv2.putText(timer_frame_b, instr_text_b, org2,
                cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 0), 4)
    cv2.putText(timer_frame_b, instr_text_b, org2,
                cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 0), 1)

    display_until = now + hud_duration

    return display_until, timer_frame_a, timer_frame_b

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

    instr_text_a = 'a/d: freq | Esc/q: quit'
    instr_text_b = 'Numpad-Rows/Cols'

    # Draw on A
    cv2.putText(timer_frame_a, instr_text_a, org1,
                cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 0), 4)
    cv2.putText(timer_frame_a, instr_text_a, org1,
                cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 0), 1)
    cv2.putText(timer_frame_a, instr_text_b, org2,
                cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 0), 4)
    cv2.putText(timer_frame_a, instr_text_b, org2,
                cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 0), 1)

    # Draw on B (same text)
    cv2.putText(timer_frame_b, instr_text_a, org1,
                cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 0), 4)
    cv2.putText(timer_frame_b, instr_text_a, org1,
                cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 0), 1)
    cv2.putText(timer_frame_b, instr_text_b, org2,
                cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 0), 4)
    cv2.putText(timer_frame_b, instr_text_b, org2,
                cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 0), 1)

    display_until = now + hud_duration

    return display_until, timer_frame_a, timer_frame_b

def adjust_frequency_and_build_hud(
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

    timer_frame_a, timer_frame_b = build_timer_frames(
        checker_a, checker_b, flash_freq_hz, period, screen_width, screen_height
    )

    return flash_freq_hz, period, display_until, timer_frame_a, timer_frame_b

# -------------------- Main loop --------------------

def main():
    # Initial checkerboard resolution
    squares_x = 12
    squares_y = 7

    flash_freq_hz = 20.0
    period = 1.0 / flash_freq_hz

    # Build base patterns
    checker_a = make_checkerboard(screen_width, screen_height, squares_x, squares_y)
    # White frame for B (checkerboard vs full white)
    checker_b = np.ones_like(checker_a) * 255

    # Initial HUD frames (not shown until display_until is set)
    timer_frame_a, timer_frame_b = build_timer_frames(
        checker_a, checker_b, flash_freq_hz, period, screen_width, screen_height
    )
    display_until = None

    # Setup fullscreen window
    win_name = "Fast Checkerboard"
    cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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

    pause = False

    while True:
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
            next_toggle += period

        # Choose which frame to display (HUD vs plain)
        show_hud = (display_until is not None and now < display_until)
        if show_hud:
            frame = timer_frame_a if (pause or use_a) else timer_frame_b
        else:
            frame = checker_a if (pause or use_a) else checker_b

        cv2.imshow(win_name, frame)

        # Frame counters
        if use_a:
            frames_a += 1
        else:
            frames_b += 1

        key = cv2.waitKey(1) & 0xFF

        if key == ord('a'):  # decrease frequency
            flash_freq_hz, period, display_until, timer_frame_a, timer_frame_b = adjust_frequency_and_build_hud(
                flash_freq_hz,
                -1.0,
                checker_a,
                checker_b,
                screen_width,
                screen_height,
                now,
            )
            next_toggle = now + period  # reset schedule relative to "now"

        elif key == ord('d'):  # increase frequency
            flash_freq_hz, period, display_until, timer_frame_a, timer_frame_b = adjust_frequency_and_build_hud(
                flash_freq_hz,
                +1.0,
                checker_a,
                checker_b,
                screen_width,
                screen_height,
                now,
            )
            next_toggle = now + period

        elif key == ord('p'):
            total_t = time_a + time_b
            total_f = frames_a + frames_b
            if total_t > 0 and total_f > 0:
                print(
                    f"time_a={time_a:.3f}, time_b={time_b:.3f}, "
                    f"frac_a={time_a/total_t:.3f}, "
                    f"frames_a={frames_a}, frames_b={frames_b}, "
                    f"frame_frac_a={frames_a/total_f:.3f}, "
                    f"flash_freq_hz={flash_freq_hz:.2f}, "
                    f"squares_x={squares_x}, squares_y={squares_y}"
                )
            else:
                print("Not enough data yet.")

        elif key in (27, ord('q')):  # ESC or 'q'
            break

        # ---- NEW: Arrow key handling for checker count ----
        # On most OpenCV builds:

        # --- NUMPAD-BASED CHECKER RESOLUTION CONTROL ---

        elif key == ord('4'):  # Numpad 4: fewer columns
            if squares_x > 2:
                squares_x -= 1
            rebuild = True

        elif key == ord('6'):  # Numpad 6: more columns
            squares_x += 1
            rebuild = True

        elif key == ord('8'):  # Numpad 8: more rows
            squares_y += 1
            rebuild = True

        elif key == ord('2'):  # Numpad 2: fewer rows
            if squares_y > 2:
                squares_y -= 1
            rebuild = True

        elif key == 32:
            pause = not pause

        # Any other key (non-255): show instructions overlay for a bit
        elif key != 255:
            display_until, timer_frame_a, timer_frame_b = build_instr_frames(
                checker_a, checker_b, screen_width, screen_height, now)

        if rebuild:
            # Clamp to something sane to avoid zero-sized tiles
            squares_x = max(2, min(squares_x, screen_width))
            squares_y = max(2, min(squares_y, screen_height))

            checker_a = make_checkerboard(screen_width, screen_height, squares_x, squares_y)
            checker_b = np.ones_like(checker_a) * 255

            # Refresh HUD variants too
            display_until, timer_frame_a, timer_frame_b = build_squares_frames(
                checker_a, checker_b, screen_width, screen_height, now, squares_x, squares_y
            )
            rebuild = False




    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
