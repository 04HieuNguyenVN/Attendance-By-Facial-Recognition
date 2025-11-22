"""Vision pipeline that normalizes frames before AI inference."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import cv2
import numpy as np

from .camera_manager import CameraManager, CameraError


@dataclass
class VisionFrame:
    frame_id: str
    timestamp: datetime
    bgr: np.ndarray
    rgb: np.ndarray
    metadata: Dict[str, Any]
    quality_score: float = 0.0


class VisionPipeline:
    """Combines CameraManager with preprocessing utilities."""

    def __init__(self, camera: CameraManager):
        self.camera = camera
        self.frame_counter = 0

    def _next_id(self) -> str:
        self.frame_counter += 1
        return f"frame-{self.frame_counter}"

    def _compute_quality(self, frame: np.ndarray) -> float:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            normalized = max(0.0, min(laplacian_var / 1500.0, 1.0))
            return float(normalized)
        except Exception:
            return 0.0

    def next_frame(self) -> VisionFrame:
        frame = self.camera.read()
        timestamp = datetime.utcnow()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        quality = self._compute_quality(frame)

        return VisionFrame(
            frame_id=self._next_id(),
            timestamp=timestamp,
            bgr=frame,
            rgb=rgb,
            metadata={
                "size": frame.shape,
                "source": "camera",
            },
            quality_score=quality,
        )

    def get_frame(self) -> VisionFrame:
        """Backward compatible alias."""
        return self.next_frame()

    @staticmethod
    def placeholder(message: str = "Camera disabled") -> Optional[bytes]:
        # Mirror existing helper but keep here for re-use
        h, w = 480, 640
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (30, 30, 30)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.8
        thickness = 2
        text_size, _ = cv2.getTextSize(message, font, scale, thickness)
        text_w, text_h = text_size
        x = max(10, (w - text_w) // 2)
        y = max(30, (h - text_h) // 2)
        cv2.putText(img, message, (x, y), font, scale, (200, 200, 200), thickness, cv2.LINE_AA)
        ret, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ret:
            return None
        return buf.tobytes()
