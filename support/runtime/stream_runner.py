import time
from typing import Any

import cv2

from support.core.enums import ImageSource
from support.io.my_logging import LOG
from support.vision.vimba_controller import HAVE_VMBPY, VmbTimeout


class StreamRunner:
    """Owns source dispatch and live/static frame acquisition loops."""

    def __init__(self, owner: Any):
        self.owner = owner

    def run(self) -> None:
        if self.owner.camConfig.imageSource == ImageSource.Camera_Stream:
            self.run_video_stream()
        elif self.owner.camConfig.imageSource == ImageSource.Stream_from_Folder:
            self.owner.run_folder_reader()
        elif self.owner.camConfig.imageSource == ImageSource.Static_Image:
            self.run_detect_single_image()

    def queue_live_vimba_update(self, settings: dict[str, Any]) -> bool:
        if not bool(getattr(self.owner.camConfig, "use_vimba", False)):
            return False
        if not bool(self.owner.stream_running_var.get()):
            return False

        self.owner.vimbaCam.live_update(settings)
        return True

    def run_detect_single_image(self) -> None:
        cv2.namedWindow(self.owner.windowName, cv2.WINDOW_NORMAL)
        frame = cv2.imread(str(self.owner.camConfig.imageFilepath))
        while (
            not self.owner.threadStopper.is_set()
            and self.owner._window_is_open()
            and self.owner.showWindow
        ):
            self.owner.analyze_image(frame)

            key = cv2.waitKey(1)
            if key == 27:
                self.owner.threadStopper.set()
                break

        try:
            cv2.destroyWindow(self.owner.windowName)
        except cv2.error:
            pass

        self.owner.after(0, self.owner.on_worker_exit)  # type: ignore[call-arg]

    def run_vimba_stream(self) -> None:
        if not HAVE_VMBPY:
            LOG.error("VmbPy is not installed or could not be imported.")
            self.owner.after(0, self.owner.on_worker_exit)  # type: ignore[call-arg]
            return

        def handler(camera, stream, img_frame):
            if self.owner.threadStopper.is_set() or not self.owner.showWindow:
                try:
                    camera.queue_frame(img_frame)
                except (AttributeError, RuntimeError, VmbTimeout):
                    pass
                return

            try:
                self.owner.vimbaCam.update_frame(
                    img_frame,
                    self.owner.camConfig.vimba_profile,
                )
            finally:
                try:
                    camera.queue_frame(img_frame)
                except (AttributeError, RuntimeError, VmbTimeout):
                    pass

        try:
            cv2.namedWindow(self.owner.windowName, cv2.WINDOW_NORMAL)

            with self.owner.vimbaCam.vmbSystem_getInstance() as vmb:
                cam = self.owner.vimbaCam.select_vimba_camera(
                    vmb,
                    self.owner.camConfig.vimba_camera_id,
                )

                with cam:
                    self.owner.vimbaCam.configure_stream(cam, self.owner.camConfig)

                    try:
                        self.owner.vimbaCam.start_stream(
                            handler,
                            self.owner.camConfig.vimba_profile,
                        )

                        while (
                            not self.owner.threadStopper.is_set()
                            and self.owner.showWindow
                            and not self.owner.making_gifOrVid
                        ):
                            frame, img_time = self.owner.vimbaCam.update_stream(self.owner.camConfig)

                            if frame is not None:
                                self.owner.curr_frame = frame
                                self.owner.analyze_image(frame, img_time=img_time)

                            key = cv2.waitKey(1)
                            if key == 27:
                                self.owner.threadStopper.set()
                                self.owner.showWindow = False
                                break

                            if not self.owner._window_is_open():
                                self.owner.threadStopper.set()
                                self.owner.showWindow = False
                                break
                    finally:
                        self.owner.vimbaCam.stop_stream()
        except Exception as e:
            LOG.exception(f"Vimba stream failed: {e}")
        finally:
            self.owner.vimbaCam.stop_lock()

            try:
                cv2.destroyWindow(self.owner.windowName)
            except cv2.error:
                pass

            self.owner.after(0, self.owner.on_worker_exit)  # type: ignore[call-arg]

    def run_video_stream(self) -> None:
        if bool(getattr(self.owner.camConfig, "use_vimba", False)):
            self.run_vimba_stream()
            return

        self.owner.vc = cv2.VideoCapture(self.owner.camConfig.cam_index, cv2.CAP_DSHOW)
        self.owner.vc.set(cv2.CAP_PROP_FPS, 60)

        cv2.namedWindow(self.owner.windowName, cv2.WINDOW_NORMAL)
        rval, self.owner.curr_frame = self.owner.vc.read()
        if rval:
            cv2.resizeWindow(
                self.owner.windowName,
                self.owner.curr_frame.shape[1],
                self.owner.curr_frame.shape[0],
            )
            self.owner.lastHeight = self.owner.curr_frame.shape[0]
            self.owner.lastWidth = self.owner.curr_frame.shape[1]

        while (
            rval
            and not self.owner.threadStopper.is_set()
            and self.owner.showWindow
            and not self.owner.making_gifOrVid
        ):
            rval, frame = self.owner.vc.read()

            self.owner.analyze_image(frame)
            key = cv2.waitKey(1)

            if key == 27:
                self.owner.after(0, self.owner.filepath_page.toggle_stream)  # type: ignore[call-arg]
                self.owner.threadStopper.set()
                break

            new_time = self.owner._handle_chessboard_hotkeys(key)
            if new_time is not None:
                self.owner._cb_status_until = new_time

            if not self.owner._window_is_open():
                self.owner.after(0, self.owner.filepath_page.toggle_stream)  # type: ignore[call-arg]
                self.owner.threadStopper.set()
                break

        if self.owner.vc is not None and self.owner.vc.isOpened():
            self.owner.vc.release()
            self.owner.vc = None

        try:
            cv2.destroyWindow(self.owner.windowName)
        except cv2.error:
            pass

        self.owner.after(0, self.owner.on_worker_exit)  # type: ignore[call-arg]
