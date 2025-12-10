"""
Attendance Tracker - Quản lý logic điểm danh
Business logic for attendance tracking and validation
"""
from datetime import datetime
from typing import Dict, Optional, Any
import threading


class AttendanceTracker:
    """Service quản lý logic điểm danh"""
    
    def __init__(self, database, state_manager, logger=None):
        self.db = database
        self.state = state_manager
        self.logger = logger
        self._lock = threading.Lock()
    
    def load_today_attendance(
        self,
        session_id: Optional[int] = None,
        credit_class_id: Optional[int] = None
    ):
        """Load danh sách đã điểm danh hôm nay từ database"""
        try:
            attendance_data = self.db.get_today_attendance(
                session_id=session_id,
                credit_class_id=credit_class_id,
            )
            
            # Reset state
            self.state.reset_attendance_tracking()
            
            # Load vào state
            for record in attendance_data:
                record_dict = dict(record) if not isinstance(record, dict) else record
                student_id = record_dict.get('student_id')
                
                if not student_id:
                    continue
                
                # Chuẩn bị thông tin sinh viên
                student_info = {
                    'name': record_dict.get('student_name') or record_dict.get('full_name') or student_id,
                    'class_name': record_dict.get('credit_class_name') or record_dict.get('class_name'),
                    'class_type': 'credit' if record_dict.get('credit_class_id') else 'administrative',
                    'credit_class_id': record_dict.get('credit_class_id')
                }
                
                # Cập nhật state
                if record_dict.get('check_in_time'):
                    self.state.add_checked_in(student_id, student_info)
                
                if record_dict.get('check_out_time'):
                    self.state.add_checked_out(student_id)
            
            if self.logger:
                self.logger.info(
                    f"[AttendanceTracker] Loaded {len(attendance_data)} records"
                )
        
        except Exception as e:
            if self.logger:
                self.logger.error(f"[AttendanceTracker] Error loading attendance: {e}")
    
    def can_mark_attendance(
        self,
        student_id: str,
        expected_student_id: Optional[str] = None,
        expected_credit_class_id: Optional[int] = None,
        cooldown_seconds: int = 30
    ) -> tuple[bool, Optional[str]]:
        """
        Kiểm tra có thể điểm danh không
        Returns: (can_mark: bool, reason: Optional[str])
        """
        normalized_id = student_id.strip().upper()
        
        # Kiểm tra expected student ID
        if expected_student_id:
            normalized_expected = expected_student_id.strip().upper()
            if normalized_id != normalized_expected:
                return False, f"Expected {normalized_expected} but got {normalized_id}"
        
        # Kiểm tra đã check-in chưa
        if self.state.is_checked_in(normalized_id):
            if not self.state.is_checked_out(normalized_id):
                return False, "Already checked in, not yet checked out"
        
        # Kiểm tra cooldown
        if not self.state.can_recognize_again(normalized_id, cooldown_seconds):
            return False, f"Recognition cooldown ({cooldown_seconds}s)"
        
        # Kiểm tra session/class match
        if expected_credit_class_id:
            session = self.state.get_active_session()
            if not session:
                return False, "No active session"
            
            session_class_id = session.get('credit_class_id')
            if session_class_id != expected_credit_class_id:
                return False, f"Session class mismatch"
        
        return True, None
    
    def mark_attendance(
        self,
        student_id: str,
        student_name: str,
        confidence_score: Optional[float] = None,
        expected_student_id: Optional[str] = None,
        expected_credit_class_id: Optional[int] = None,
        notes: Optional[str] = None
    ) -> bool:
        """
        Lưu điểm danh vào database
        Returns: True if successful, False otherwise
        """
        with self._lock:
            normalized_id = student_id.strip().upper()
            
            # Kiểm tra điều kiện
            can_mark, reason = self.can_mark_attendance(
                normalized_id,
                expected_student_id,
                expected_credit_class_id
            )
            
            if not can_mark:
                if self.logger:
                    self.logger.info(
                        f"[AttendanceTracker] Cannot mark attendance for {student_name}: {reason}"
                    )
                return False
            
            # Lấy session context
            session = self.state.get_active_session()
            credit_class_id = session.get('credit_class_id') if session else None
            session_id = session.get('id') if session else None
            
            # Lưu vào database
            success = self.db.mark_attendance(
                student_id=normalized_id,
                student_name=student_name,
                status='present',
                confidence_score=confidence_score,
                notes=notes,
                credit_class_id=credit_class_id,
                session_id=session_id
            )
            
            if success:
                # Cập nhật state
                student_info = {
                    'name': student_name,
                    'class_name': session.get('class_name') if session else None,
                    'class_type': 'credit' if credit_class_id else 'administrative',
                    'credit_class_id': credit_class_id
                }
                self.state.add_checked_in(normalized_id, student_info)
                self.state.update_last_recognized(normalized_id)
                
                # Auto-enroll vào lớp tín chỉ nếu cần
                if credit_class_id:
                    try:
                        self.db.enroll_student_if_not_exists(
                            credit_class_id=credit_class_id,
                            student_id=normalized_id
                        )
                    except Exception as e:
                        if self.logger:
                            self.logger.warning(
                                f"[AttendanceTracker] Auto-enroll failed: {e}"
                            )
                
                if self.logger:
                    self.logger.info(
                        f"[AttendanceTracker] ✅ Marked attendance: {student_name} ({normalized_id})"
                    )
            
            return success
    
    def mark_checkout(
        self,
        student_id: str,
        notes: Optional[str] = None
    ) -> bool:
        """Checkout cho sinh viên"""
        normalized_id = student_id.strip().upper()
        
        # Kiểm tra đã check-in chưa
        if not self.state.is_checked_in(normalized_id):
            if self.logger:
                self.logger.info(
                    f"[AttendanceTracker] Cannot checkout {normalized_id}: not checked in"
                )
            return False
        
        # Kiểm tra đã checkout chưa
        if self.state.is_checked_out(normalized_id):
            if self.logger:
                self.logger.info(
                    f"[AttendanceTracker] Cannot checkout {normalized_id}: already checked out"
                )
            return False
        
        # Get session
        session = self.state.get_active_session()
        
        # Update database
        success = self.db.update_checkout(
            student_id=normalized_id,
            session_id=session.get('id') if session else None,
            notes=notes
        )
        
        if success:
            self.state.add_checked_out(normalized_id)
            if self.logger:
                self.logger.info(
                    f"[AttendanceTracker] ✅ Checked out: {normalized_id}"
                )
        
        return success
    
    def get_attendance_stats(self, session_id: Optional[int] = None) -> Dict[str, Any]:
        """Lấy thống kê điểm danh"""
        session = self.state.get_active_session()
        if session_id is None and session:
            session_id = session.get('id')
        
        stats = {
            'checked_in': len(self.state.today_checked_in),
            'checked_out': len(self.state.today_checked_out),
            'active': len(self.state.today_checked_in - self.state.today_checked_out),
            'session_id': session_id,
            'session_info': session
        }
        
        return stats
    
    def reset_session_state(self, session_row: Optional[Dict] = None):
        """Reset state khi chuyển phiên"""
        session_id = session_row.get('id') if session_row else None
        credit_class_id = session_row.get('credit_class_id') if session_row else None
        
        # Update active session
        self.state.set_active_session(session_row)
        
        # Reload attendance data
        self.load_today_attendance(
            session_id=session_id,
            credit_class_id=credit_class_id
        )
        
        # Clear presence tracking
        self.state.presence_tracking.clear()
        self.state.attendance_progress.clear()
        
        if self.logger:
            self.logger.info(f"[AttendanceTracker] Reset state for session {session_id}")
