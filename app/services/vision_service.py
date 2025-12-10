"""
Vision Service - Quản lý camera pipeline và vision state
Tách từ app.py để module hóa
"""
import logging
from typing import Optional
from core.vision.state import VisionPipelineState, VisionStateConfig
from core.vision.camera_manager import CameraError


class VisionService:
    """Service quản lý vision pipeline và camera state."""
    
    def __init__(
        self,
        camera_index: int = 0,
        camera_width: Optional[int] = None,
        camera_height: Optional[int] = None,
        warmup_frames: int = 3,
        buffer_size: int = 2,
        logger: Optional[logging.Logger] = None
    ):
        self.config = VisionStateConfig(
            index=camera_index,
            width=camera_width,
            height=camera_height,
            warmup_frames=warmup_frames,
            buffer_size=buffer_size,
        )
        self.logger = logger or logging.getLogger(__name__)
        self._vision_state: Optional[VisionPipelineState] = None
        self._camera_enabled = True
    
    @property
    def camera_enabled(self) -> bool:
        """Trạng thái camera (bật/tắt)."""
        return self._camera_enabled
    
    @camera_enabled.setter
    def camera_enabled(self, value: bool):
        """Cập nhật trạng thái camera."""
        self._camera_enabled = value
    
    def get_or_create_vision_state(self) -> VisionPipelineState:
        """Tạo hoặc lấy vision state hiện tại."""
        if self._vision_state is None:
            self._vision_state = VisionPipelineState(
                config=self.config,
                logger=self.logger
            )
        return self._vision_state
    
    def ensure_camera_pipeline(self):
        """Khởi động camera pipeline nếu camera đang bật."""
        if not self._camera_enabled:
            return None
        
        state = self.get_or_create_vision_state()
        state.set_enabled(True)
        
        try:
            return state.ensure_ready()
        except CameraError as exc:
            self.logger.error("[Camera] ❌ Không thể khởi động camera: %s", exc)
            return None
    
    def release_camera_capture(self):
        """Giải phóng camera capture."""
        state = self._vision_state
        if state is None:
            return
        
        try:
            state.set_enabled(False)
            state.stop()
        except Exception as exc:
            self.logger.debug("[Camera] ⚠️ Không thể giải phóng camera: %s", exc)
    
    def get_camera_status(self) -> dict:
        """Lấy trạng thái camera hiện tại."""
        state = self._vision_state or self.get_or_create_vision_state()
        status = state.status() if state else {'opened': False}
        
        return {
            'enabled': self._camera_enabled,
            'opened': bool(status.get('opened'))
        }
    
    def toggle_camera(self) -> tuple[bool, Optional[str]]:
        """
        Bật/tắt camera.
        Returns: (success, error_message)
        """
        try:
            desired_state = not self._camera_enabled
            self._camera_enabled = desired_state
            
            if desired_state:
                pipeline = self.ensure_camera_pipeline()
                if pipeline is None:
                    self._camera_enabled = False
                    return False, 'Không thể khởi động camera'
            else:
                self.release_camera_capture()
            
            return True, None
            
        except Exception as exc:
            self.logger.error(f"Error toggling camera: {exc}")
            self._camera_enabled = False
            self.release_camera_capture()
            return False, str(exc)
