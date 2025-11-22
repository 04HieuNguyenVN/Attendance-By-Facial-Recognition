"""Camera device management with reusable state."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Protocol

import cv2


logger = logging.getLogger(__name__)


class CameraError(RuntimeError):
    """Raised when camera operations fail."""


class CameraProvider(Protocol):
    """Abstraction for objects that can supply cv2.VideoCapture."""

    def open(self, index: int) -> cv2.VideoCapture:
        ...


class DefaultCameraProvider:
    """Real provider that uses OpenCV to create VideoCapture objects."""

    def open(self, index: int) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(index)
        if not capture or not capture.isOpened():
            raise CameraError(f"Cannot open camera index {index}")
        return capture


@dataclass
class CameraConfig:
    index: int = 0
    width: Optional[int] = None
    height: Optional[int] = None
    warmup_frames: int = 3
    buffer_size: Optional[int] = 2


@dataclass
class CameraState:
    """State container for camera device handles."""

    capture: Optional[cv2.VideoCapture] = None
    enabled: bool = True


class CameraManager:
    """Owns camera lifecycle and exposes safe read operations."""

    def __init__(
        self,
        index: int = 0,
        provider: Optional[CameraProvider] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        warmup_frames: int = 3,
        buffer_size: Optional[int] = 2,
    ):
        self.config = CameraConfig(
            index=index,
            width=width,
            height=height,
            warmup_frames=warmup_frames,
            buffer_size=buffer_size,
        )
        self.state = CameraState()
        self.provider = provider or DefaultCameraProvider()

    def start(self) -> cv2.VideoCapture:
        capture = self.state.capture
        if capture is not None and capture.isOpened():
            return capture

        capture = self.provider.open(self.config.index)
        self._configure_capture(capture)
        self.state.capture = capture
        return capture

    def _configure_capture(self, capture: cv2.VideoCapture) -> None:
        if capture is None:
            return
        try:
            if self.config.width:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            if self.config.height:
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            if self.config.buffer_size is not None and hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                capture.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)

            actual_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = capture.get(cv2.CAP_PROP_FPS)
            logger.info(
                "Camera ready: %sx%s @ %.2f fps",
                actual_w,
                actual_h,
                fps or 0,
            )

            warmup = max(0, self.config.warmup_frames)
            if warmup:
                logger.debug("Warming up camera (%s frames)", warmup)
                success = 0
                for _ in range(warmup):
                    ret, _frame = capture.read()
                    if ret:
                        success += 1
                    time.sleep(0.05)
                logger.debug("Warmup frames ok=%s/%s", success, warmup)
        except Exception as exc:
            logger.warning("Unable to configure camera: %s", exc)

    def stop(self) -> None:
        capture = self.state.capture
        self.state.capture = None
        if capture is not None:
            capture.release()

    def read(self):
        if not self.state.enabled:
            raise CameraError("Camera disabled")
        capture = self.start()
        ret, frame = capture.read()
        if not ret or frame is None:
            raise CameraError("Unable to read frame from camera")
        return frame

    def set_enabled(self, value: bool):
        self.state.enabled = bool(value)
        if not self.state.enabled:
            self.stop()

    def is_enabled(self) -> bool:
        return self.state.enabled

    def get_capture(self) -> Optional[cv2.VideoCapture]:
        return self.state.capture
