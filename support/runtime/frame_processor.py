import os
import time
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from support.core.enums import ImageSource
import support.gui.UserSelectQueue as GuiQueue
import support.viz.colors as clr
from support.viz.CVFontScaling import med_thick


class FrameProcessor:
    """Owns per-frame pipeline execution and display/export finishing."""

    def __init__(self, owner: Any):
        self.owner = owner

    def analyze_image(
        self,
        frame,
        img_time=None,
        name=None,
        display_in_realtime: bool = True,
        box_around: bool = False,
    ) -> NDArray | None:
        if frame is None or frame.size == 0:
            return None

        ctx = GuiQueue.FrameCtx(
            img_time=img_time,
            name=name,
            display_in_realtime=display_in_realtime,
        )

        self.owner.pnpResult = None
        self.owner.qnpResult = None
        self.owner.curr_frame_gray = None

        markup_frame = self.owner.markup_frame
        if markup_frame is None or markup_frame.shape != frame.shape:
            markup_frame = frame.copy()
            self.owner.markup_frame = markup_frame
        else:
            np.copyto(markup_frame, frame)

        for func, args in self.owner.list_of_image_process_functors:
            step_args = args if isinstance(args, dict) else {}
            if not bool(step_args.get("state", True)):
                continue
            func(frame, markup_frame, ctx, step_args)

        if box_around and not self.owner.screenshot_impending:
            self.draw_box_around(frame, markup_frame, ctx, ())

        if self.owner.camConfig.imageSource == ImageSource.Stream_from_Folder:
            self.draw_name(frame, markup_frame, ctx, ())
            if not self.owner.screenshot_impending and not self.owner.making_gifOrVid:
                self.owner.draw_playbackStats(frame, markup_frame, ctx, ())

        self.draw_time(frame, markup_frame, ctx, ())

        if time.monotonic() < float(getattr(self.owner, "_cb_status_until", 0.0)):
            self.owner._draw_chessboard_state(markup_frame)

        if display_in_realtime:
            self.cleanup(markup_frame)
            return None

        return np.ascontiguousarray(markup_frame).copy()

    def cleanup(self, markup_frame: NDArray, name=None) -> None:
        self.potential_resize(markup_frame)

        cv2.imshow(
            self.owner.windowName if name is None else name,
            cv2.resize(markup_frame, (self.owner.lastWidth, self.owner.lastHeight)),
        )

        if (
            (self.owner.recording and time.time() - self.owner.lastImageTime > self.owner.camConfig.secondsBetweenImages)
            or self.owner.screenshot_impending
        ):
            cv2.imwrite(
                os.path.join(self.owner.camConfig.saveFolder, str(self.owner.img_idx) + ".png"),
                markup_frame,
            )
            self.owner.img_idx += 1
            self.owner.lastImageTime = time.time()
            self.owner.recordButton.configure(text=f"Saving Imagery: #{self.owner.img_idx}")
            self.owner.screenshot_impending = False

    def potential_resize(self, markup_frame: NDArray) -> None:
        if not self.owner._window_is_open() or markup_frame.shape[0] == 0:
            return

        _x, _y, width, height = cv2.getWindowImageRect(self.owner.windowName)
        aspect_ratio = markup_frame.shape[1] / markup_frame.shape[0]

        if not self.owner._window_is_open():
            return

        if self.owner.lastHeight != height and height != 0:
            cv2.resizeWindow(self.owner.windowName, int(height * aspect_ratio), height)
            self.owner.lastHeight = height
            self.owner.lastWidth = int(height * aspect_ratio)
        elif self.owner.lastWidth != width and width != 0:
            cv2.resizeWindow(self.owner.windowName, width, int(width / aspect_ratio))
            self.owner.lastWidth = width
            self.owner.lastHeight = int(width / aspect_ratio)

    @staticmethod
    def draw_time(frame, markup_frame, ctx: GuiQueue.FrameCtx, args) -> None:
        if ctx.img_time is None or ctx.img_time > 1_000_000:
            return

        time_str = f"Flight Time: {ctx.img_time:.2f}"
        from support.viz.HUD_draw import draw_time_on_image

        draw_time_on_image(markup_frame, time_str)

    @staticmethod
    def draw_name(frame, markup_frame, ctx: GuiQueue.FrameCtx, args) -> None:
        from support.viz.HUD_draw import draw_name_on_image

        draw_name_on_image(os.path.basename(ctx.name), markup_frame)

    @staticmethod
    def draw_box_around(frame, markup_frame, ctx: GuiQueue.FrameCtx, args) -> None:
        h, w, _ = markup_frame.shape
        cv2.rectangle(markup_frame, (0, 0), (w - 1, h - 1), clr.HUD_YELLOW, med_thick(h))
