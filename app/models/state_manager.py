"""
State Manager - Quản lý trạng thái global của ứng dụng
Centralized state management for attendance tracking
"""
import threading
from datetime import datetime
from typing import Dict, Set, Optional, Any
from pathlib import Path


class StateManager:
    """Quản lý trạng thái global thay thế cho các biến global trong app.py"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Chỉ khởi tạo 1 lần
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        
        # Attendance tracking
        self.today_checked_in: Set[str] = set()
        self.today_checked_out: Set[str] = set()
        self.today_student_names: Dict[str, Dict[str, Any]] = {}
        self.today_recorded_lock = threading.Lock()
        
        # Last recognition tracking (cooldown)
        self.last_recognized: Dict[str, datetime] = {}
        self.last_recognized_lock = threading.Lock()
        
        # Presence tracking
        self.presence_tracking: Dict[str, Dict[str, Any]] = {}
        self.presence_tracking_lock = threading.Lock()
        
        # Attendance progress (for streaming)
        self.attendance_progress: Dict[str, Dict[str, Any]] = {}
        self.attendance_progress_lock = threading.Lock()
        
        # Credit session cache
        self.current_credit_session: Optional[Dict[str, Any]] = None
        self.current_session_lock = threading.Lock()
        
        # Face recognition data
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        self.known_face_embeddings = []
        
        # Camera state
        self.camera_enabled = True
        self.vision_state = None
        self.inference_engine = None
    
    def reset_attendance_tracking(self):
        """Reset tất cả tracking data"""
        with self.today_recorded_lock:
            self.today_checked_in.clear()
            self.today_checked_out.clear()
            self.today_student_names.clear()
        
        with self.presence_tracking_lock:
            self.presence_tracking.clear()
        
        with self.attendance_progress_lock:
            self.attendance_progress.clear()
    
    def add_checked_in(self, student_id: str, student_info: Optional[Dict] = None):
        """Thêm sinh viên vào danh sách đã check-in"""
        with self.today_recorded_lock:
            normalized_id = student_id.strip().upper()
            self.today_checked_in.add(normalized_id)
            if student_info:
                self.today_student_names[normalized_id] = student_info
    
    def add_checked_out(self, student_id: str):
        """Thêm sinh viên vào danh sách đã check-out"""
        with self.today_recorded_lock:
            normalized_id = student_id.strip().upper()
            self.today_checked_out.add(normalized_id)
    
    def is_checked_in(self, student_id: str) -> bool:
        """Kiểm tra sinh viên đã check-in chưa"""
        with self.today_recorded_lock:
            normalized_id = student_id.strip().upper()
            return normalized_id in self.today_checked_in
    
    def is_checked_out(self, student_id: str) -> bool:
        """Kiểm tra sinh viên đã check-out chưa"""
        with self.today_recorded_lock:
            normalized_id = student_id.strip().upper()
            return normalized_id in self.today_checked_out
    
    def can_recognize_again(self, student_id: str, cooldown_seconds: int = 30) -> bool:
        """Kiểm tra có thể nhận diện lại sinh viên không (cooldown check)"""
        with self.last_recognized_lock:
            normalized_id = student_id.strip().upper()
            last_time = self.last_recognized.get(normalized_id)
            
            if last_time is None:
                return True
            
            elapsed = (datetime.now() - last_time).total_seconds()
            return elapsed >= cooldown_seconds
    
    def update_last_recognized(self, student_id: str):
        """Cập nhật thời gian nhận diện gần nhất"""
        with self.last_recognized_lock:
            normalized_id = student_id.strip().upper()
            self.last_recognized[normalized_id] = datetime.now()
    
    def update_presence(self, student_id: str, additional_seconds: float = 0):
        """Cập nhật thời gian có mặt của sinh viên"""
        with self.presence_tracking_lock:
            normalized_id = student_id.strip().upper()
            
            if normalized_id not in self.presence_tracking:
                self.presence_tracking[normalized_id] = {
                    'last_seen': datetime.now(),
                    'total_time': 0
                }
            
            tracking = self.presence_tracking[normalized_id]
            tracking['last_seen'] = datetime.now()
            tracking['total_time'] = tracking.get('total_time', 0) + additional_seconds
    
    def get_presence_info(self, student_id: str) -> Optional[Dict[str, Any]]:
        """Lấy thông tin thời gian có mặt của sinh viên"""
        with self.presence_tracking_lock:
            normalized_id = student_id.strip().upper()
            return self.presence_tracking.get(normalized_id)
    
    def cleanup_stale_presence(self, timeout_seconds: int = 300):
        """Xóa tracking của sinh viên không xuất hiện trong thời gian dài"""
        with self.presence_tracking_lock:
            now = datetime.now()
            stale_ids = []
            
            for student_id, info in self.presence_tracking.items():
                last_seen = info.get('last_seen')
                if last_seen:
                    elapsed = (now - last_seen).total_seconds()
                    if elapsed >= timeout_seconds:
                        stale_ids.append(student_id)
            
            for student_id in stale_ids:
                del self.presence_tracking[student_id]
            
            return stale_ids
    
    def set_active_session(self, session_row: Optional[Dict]):
        """Cập nhật phiên điểm danh đang active"""
        with self.current_session_lock:
            self.current_credit_session = session_row
    
    def get_active_session(self) -> Optional[Dict]:
        """Lấy phiên điểm danh đang active"""
        with self.current_session_lock:
            return self.current_credit_session
    
    def update_attendance_progress(self, student_id: str, progress_info: Dict):
        """Cập nhật tiến độ xác nhận điểm danh"""
        with self.attendance_progress_lock:
            normalized_id = student_id.strip().upper()
            self.attendance_progress[normalized_id] = progress_info
    
    def get_attendance_progress(self, student_id: str) -> Optional[Dict]:
        """Lấy tiến độ xác nhận điểm danh"""
        with self.attendance_progress_lock:
            normalized_id = student_id.strip().upper()
            return self.attendance_progress.get(normalized_id)
    
    def clear_attendance_progress(self, student_id: str):
        """Xóa tiến độ xác nhận điểm danh"""
        with self.attendance_progress_lock:
            normalized_id = student_id.strip().upper()
            if normalized_id in self.attendance_progress:
                del self.attendance_progress[normalized_id]


# Singleton instance
state_manager = StateManager()
