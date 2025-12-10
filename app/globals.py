"""
Global state module - REFACTORED
Sử dụng models mới thay vì raw global variables
"""
import threading
from app.models import (
    state_manager,
    get_event_broadcaster,
    CameraService,
    AttendanceTracker,
    FaceRecognitionManager,
)

# =============================================================================
# NEW ARCHITECTURE: Sử dụng model instances
# =============================================================================

# Singleton instances (sẽ được khởi tạo trong app/__init__.py)
camera_service = None
attendance_tracker = None
face_recognition_manager = None
event_broadcaster = None

# =============================================================================
# BACKWARD COMPATIBILITY: Direct references để code cũ vẫn hoạt động
# =============================================================================

# Face recognition data - proxied từ state_manager
known_face_encodings = state_manager.known_face_encodings
known_face_names = state_manager.known_face_names
known_face_ids = state_manager.known_face_ids
known_face_embeddings = state_manager.known_face_embeddings

# Attendance tracking - proxied từ state_manager
today_checked_in = state_manager.today_checked_in
today_checked_out = state_manager.today_checked_out
today_student_names = state_manager.today_student_names
today_recorded_lock = state_manager.today_recorded_lock

# Session cache - proxied từ state_manager
current_credit_session = state_manager.current_credit_session
current_session_lock = state_manager.current_session_lock

# Presence tracking - proxied từ state_manager
presence_tracking = state_manager.presence_tracking
presence_tracking_lock = state_manager.presence_tracking_lock
PRESENCE_TIMEOUT = 300  # 5 phút

# Attendance progress - proxied từ state_manager
attendance_progress = state_manager.attendance_progress
attendance_progress_lock = state_manager.attendance_progress_lock

# Recognition cooldown - proxied từ state_manager
last_recognized = state_manager.last_recognized
last_recognized_lock = state_manager.last_recognized_lock
RECOGNITION_COOLDOWN = 30  # Giây

# SSE clients - sẽ được init sau
sse_clients = []
sse_clients_lock = threading.Lock()

# Camera và vision state - sẽ được init sau
vision_state = None
camera_enabled = True
inference_engine = state_manager.inference_engine

# =============================================================================
# Helper functions để update references khi services được init
# =============================================================================

def init_service_references():
    """
    Update global references sau khi services được khởi tạo
    Gọi hàm này trong app/__init__.py sau khi init services
    """
    global sse_clients, sse_clients_lock, vision_state, camera_enabled
    
    # Update SSE references
    if event_broadcaster:
        sse_clients = event_broadcaster.clients
        sse_clients_lock = event_broadcaster.clients_lock
    
    # Update camera references
    if camera_service:
        vision_state = camera_service.vision_state
        camera_enabled = camera_service.enabled

# =============================================================================
# NOTE: Code mới nên dùng trực tiếp:
#   - state_manager: cho state management
#   - camera_service: cho camera operations
#   - attendance_tracker: cho attendance logic
#   - face_recognition_manager: cho face recognition
#   - event_broadcaster: cho SSE events
# =============================================================================
