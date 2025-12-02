"""Wrapper trạng thái cho vòng đời camera và thu thập khung hình."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional

from .camera_manager import CameraManager, CameraError
from .pipeline import VisionPipeline, VisionFrame


@dataclass
class VisionStateConfig:
    index: int
    width: Optional[int] = None
    height: Optional[int] = None
    warmup_frames: int = 3
    buffer_size: Optional[int] = 2


class VisionPipelineState:
    """Quản lý CameraManager + VisionPipeline với các helper an toàn luồng."""

    def __init__(
        self,
        *,
        config: VisionStateConfig,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._config = config
        self._logger = logger or logging.getLogger(__name__)
        self._lock = threading.RLock()
        self._camera_manager: Optional[CameraManager] = None
        self._pipeline: Optional[VisionPipeline] = None
        self._enabled = True

    # ------------------------------------------------------------------
    # Các helper nội bộ
    def _get_or_create_manager(self) -> CameraManager:
        if self._camera_manager is None:
            self._camera_manager = CameraManager(
                index=self._config.index,
                width=self._config.width,
                height=self._config.height,
                warmup_frames=self._config.warmup_frames,
                buffer_size=self._config.buffer_size,
            )
        return self._camera_manager

    def _get_or_create_pipeline(self) -> VisionPipeline:
        manager = self._get_or_create_manager()
        if self._pipeline is None:
            self._pipeline = VisionPipeline(manager)
        return self._pipeline

    def _shutdown_locked(self) -> None:
        manager = self._camera_manager
        if manager is not None:
            try:
                manager.stop()
            except Exception as exc:  # pragma: no cover - defensive
                self._logger.debug("[Camera] stop() thất bại: %s", exc)
        self._pipeline = None
        self._camera_manager = None

    # ------------------------------------------------------------------
    # API công khai
    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._enabled = bool(enabled)
            if not self._enabled:
                self._shutdown_locked()

    def ensure_ready(self) -> Optional[VisionPipeline]:
        with self._lock:
            if not self._enabled:
                return None
            manager = self._get_or_create_manager()
            manager.set_enabled(True)
            manager.start()
            return self._get_or_create_pipeline()

    def next_frame(self) -> VisionFrame:
        pipeline = self.ensure_ready()
        if pipeline is None:
            raise CameraError("Camera bị vô hiệu hóa")
        return pipeline.next_frame()

    def stop(self) -> None:
        with self._lock:
            self._shutdown_locked()

    def status(self) -> dict:
        with self._lock:
            capture = None
            if self._camera_manager:
                capture = self._camera_manager.get_capture()
            opened = bool(capture and getattr(capture, "isOpened", lambda: False)())
            return {
                "enabled": self._enabled,
                "opened": opened,
            }

    def placeholder_frame(self, message: str = "Camera disabled") -> Optional[bytes]:
        return VisionPipeline.placeholder(message)

    def reset(self) -> None:
        with self._lock:
            self._shutdown_locked()
            if self._enabled:
                # Recreate manager lazily on next ensure_ready
                self._camera_manager = None
                self._pipeline = None
