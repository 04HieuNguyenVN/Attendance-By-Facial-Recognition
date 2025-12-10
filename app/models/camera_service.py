"""
Camera Service - Quản lý camera và vision pipeline
Encapsulates camera management logic from app.py
"""
from typing import Optional
from pathlib import Path

from core.vision.camera_manager import CameraError
from core.vision.state import VisionPipelineState, VisionStateConfig


class CameraService:
    """Service quản lý camera và vision pipeline"""
    
    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        warmup_frames: int = 3,
        buffer_size: int = 2,
        logger=None
    ):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.warmup_frames = warmup_frames
        self.buffer_size = buffer_size
        self.logger = logger
        
        self.vision_state: Optional[VisionPipelineState] = None
        self.enabled = True
    
    def get_or_create_vision_state(self) -> VisionPipelineState:
        """Lấy hoặc tạo VisionPipelineState"""
        if self.vision_state is None:
            config = VisionStateConfig(
                index=self.camera_index,
                width=self.width,
                height=self.height,
                warmup_frames=self.warmup_frames,
                buffer_size=self.buffer_size,
            )
            self.vision_state = VisionPipelineState(config=config, logger=self.logger)
        return self.vision_state
    
    def ensure_camera_pipeline(self):
        """Đảm bảo camera pipeline sẵn sàng"""
        if not self.enabled:
            return None
        
        state = self.get_or_create_vision_state()
        state.set_enabled(True)
        
        try:
            return state.ensure_ready()
        except CameraError as exc:
            if self.logger:
                self.logger.error("[Camera] ❌ Không thể khởi động camera: %s", exc)
            return None
    
    def release_camera_capture(self):
        """Giải phóng camera capture"""
        if self.vision_state is None:
            return
        
        try:
            self.vision_state.set_enabled(False)
            self.vision_state.stop()
        except Exception as exc:
            if self.logger:
                self.logger.debug("[Camera] ⚠️ Không thể giải phóng camera: %s", exc)
    
    def toggle_camera(self) -> bool:
        """Bật/tắt camera"""
        self.enabled = not self.enabled
        
        if not self.enabled:
            self.release_camera_capture()
        else:
            self.ensure_camera_pipeline()
        
        return self.enabled
    
    def is_enabled(self) -> bool:
        """Kiểm tra camera có đang bật không"""
        return self.enabled
    
    def set_enabled(self, enabled: bool):
        """Đặt trạng thái camera"""
        if enabled != self.enabled:
            self.toggle_camera()
    
    def get_status(self) -> dict:
        """Lấy trạng thái camera"""
        state = self.vision_state
        
        return {
            'enabled': self.enabled,
            'initialized': state is not None,
            'ready': state.is_ready() if state else False,
            'camera_index': self.camera_index,
            'resolution': f"{self.width}x{self.height}"
        }
    
    def cleanup(self):
        """Dọn dẹp tài nguyên"""
        self.release_camera_capture()
        self.vision_state = None
