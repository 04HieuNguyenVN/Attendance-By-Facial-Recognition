"""
Event Broadcaster - Quản lý Server-Sent Events (SSE)
Handles real-time event broadcasting to connected clients
"""
import queue
import threading
import json
from typing import List, Dict, Any, Optional
from datetime import datetime


class EventBroadcaster:
    """Service quản lý SSE events cho thông báo real-time"""
    
    def __init__(self, logger=None):
        self.clients: List[queue.Queue] = []
        self.clients_lock = threading.Lock()
        self.logger = logger
    
    def add_client(self) -> queue.Queue:
        """Thêm client mới và trả về queue của client đó"""
        client_queue = queue.Queue(maxsize=50)
        
        with self.clients_lock:
            self.clients.append(client_queue)
        
        if self.logger:
            self.logger.info(f"[SSE] ✅ New client connected. Total: {len(self.clients)}")
        
        return client_queue
    
    def remove_client(self, client_queue: queue.Queue):
        """Xóa client khi disconnect"""
        with self.clients_lock:
            if client_queue in self.clients:
                self.clients.remove(client_queue)
                
                if self.logger:
                    self.logger.info(f"[SSE] Client disconnected. Remaining: {len(self.clients)}")
    
    def broadcast_event(self, event_data: Dict[str, Any]):
        """
        Broadcast event đến tất cả clients
        
        Args:
            event_data: Dictionary chứa event data
                - type: Loại event (vd: 'attendance_updated', 'session_updated')
                - data: Dữ liệu của event
                - timestamp: Thời gian (optional, sẽ tự động thêm nếu không có)
        """
        # Thêm timestamp nếu chưa có
        if 'timestamp' not in event_data:
            event_data['timestamp'] = datetime.now().isoformat()
        
        # Format SSE message
        message = self._format_sse_message(event_data)
        
        # Send to all clients
        disconnected_clients = []
        
        with self.clients_lock:
            for client_queue in self.clients:
                try:
                    client_queue.put_nowait(message)
                except queue.Full:
                    # Queue đầy, đánh dấu để remove
                    disconnected_clients.append(client_queue)
                    if self.logger:
                        self.logger.warning("[SSE] Client queue full, marking for removal")
        
        # Remove disconnected clients
        for client_queue in disconnected_clients:
            self.remove_client(client_queue)
        
        if self.logger and self.clients:
            self.logger.debug(
                f"[SSE] Broadcast {event_data.get('type', 'unknown')} to {len(self.clients)} clients"
            )
    
    def _format_sse_message(self, event_data: Dict[str, Any]) -> str:
        """Format data thành SSE message format"""
        event_type = event_data.get('type', 'message')
        
        # SSE format: event: type\ndata: json\n\n
        message_lines = [
            f"event: {event_type}",
            f"data: {json.dumps(event_data)}",
            "",  # Empty line để kết thúc message
        ]
        
        return "\n".join(message_lines)
    
    def broadcast_attendance_update(
        self,
        student_id: str,
        student_name: str,
        action: str = 'check_in',
        confidence: Optional[float] = None
    ):
        """Broadcast attendance update event"""
        event_data = {
            'type': 'attendance_updated',
            'data': {
                'student_id': student_id,
                'student_name': student_name,
                'action': action,  # 'check_in' or 'check_out'
                'confidence': confidence,
            }
        }
        self.broadcast_event(event_data)
    
    def broadcast_session_update(self, session_data: Optional[Dict]):
        """Broadcast session update event"""
        event_data = {
            'type': 'session_updated',
            'data': session_data
        }
        self.broadcast_event(event_data)
    
    def broadcast_recognition_event(
        self,
        student_id: str,
        student_name: str,
        confidence: float,
        status: str
    ):
        """Broadcast face recognition event"""
        event_data = {
            'type': 'face_recognized',
            'data': {
                'student_id': student_id,
                'student_name': student_name,
                'confidence': confidence,
                'status': status,
            }
        }
        self.broadcast_event(event_data)
    
    def broadcast_camera_status(self, enabled: bool, ready: bool = False):
        """Broadcast camera status change"""
        event_data = {
            'type': 'camera_status',
            'data': {
                'enabled': enabled,
                'ready': ready,
            }
        }
        self.broadcast_event(event_data)
    
    def broadcast_system_message(self, message: str, level: str = 'info'):
        """Broadcast system message"""
        event_data = {
            'type': 'system_message',
            'data': {
                'message': message,
                'level': level,  # 'info', 'warning', 'error', 'success'
            }
        }
        self.broadcast_event(event_data)
    
    def get_client_count(self) -> int:
        """Lấy số lượng clients đang kết nối"""
        with self.clients_lock:
            return len(self.clients)
    
    def cleanup(self):
        """Cleanup tất cả clients"""
        with self.clients_lock:
            self.clients.clear()
        
        if self.logger:
            self.logger.info("[SSE] All clients removed")


# Singleton instance
_broadcaster_instance = None
_broadcaster_lock = threading.Lock()


def get_event_broadcaster(logger=None) -> EventBroadcaster:
    """Get singleton instance of EventBroadcaster"""
    global _broadcaster_instance
    
    if _broadcaster_instance is None:
        with _broadcaster_lock:
            if _broadcaster_instance is None:
                _broadcaster_instance = EventBroadcaster(logger=logger)
    
    return _broadcaster_instance
