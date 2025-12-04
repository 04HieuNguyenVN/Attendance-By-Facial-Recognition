# Thiết lập mã hóa UTF-8 cho đầu ra console
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Các import của Flask
from flask import (
    Flask,
    render_template,
    Response,
    redirect,
    url_for,
    request,
    jsonify,
    session,
    flash,
    g,
    abort,
    stream_with_context,
)

# Các import thư viện chuẩn
import os
import csv
import time
import random
import base64
import re
import shutil
from pathlib import Path
from datetime import datetime, date, timedelta
import threading
import hashlib
from functools import wraps
from typing import Any, Dict, Optional
# Lưu ý: sử dụng threading.Thread / threading.Lock qua module threading để tránh
# trùng lặp tên không sử dụng trong namespace của module.

# Các import bên thứ ba
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash, generate_password_hash
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Các import nội bộ
from database import db
from logging_config import setup_logging
from core.inference.engine import (
    DeepFaceStrategy,
    FaceNetStrategy,
    InferenceEngine,
    InferenceError,
)
from core.vision.camera_manager import CameraError
from core.vision.state import VisionPipelineState, VisionStateConfig

# Cố gắng tải dotenv, nhưng không báo lỗi nếu chưa cài đặt
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Using default configuration.")
    print("Install it with: pip install python-dotenv")

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Kích thước file tối đa 16MB

# Cấu hình upload ảnh
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
SUPPORTED_IMAGE_FORMATS = {'JPEG', 'PNG', 'WEBP'}
MIN_FILE_SIZE = 1024  # 1 KB - tối thiểu
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
MIN_FACE_SAMPLES_PER_STUDENT = max(3, int(os.getenv('MIN_FACE_SAMPLES', '3')))
MAX_FACE_SAMPLES_PER_REQUEST = max(
    MIN_FACE_SAMPLES_PER_STUDENT,
    int(os.getenv('MAX_FACE_SAMPLES', '12')),
)

# Thiết lập logging
setup_logging(app)

app.logger.info(f"[STARTUP] Working directory: {os.getcwd()}")
app.logger.info(f"[STARTUP] Database path: {os.path.abspath('attendance_system.db')}")

# Load attendance cache từ DB để đồng bộ trạng thái
try:
    attendance_records = db.get_today_attendance()
    for record in attendance_records:
        student_id = record['student_id'].strip().upper()
        check_out_time = record.get('check_out_time')
        if check_out_time:
            today_checked_out.add(student_id)
        else:
            today_checked_in.add(student_id)
        # Populate student names cache
        today_student_names[student_id] = {
            'name': record.get('student_name') or record.get('full_name') or student_id,
            'class_name': record.get('class_name'),
            'class_type': 'credit' if record.get('credit_class_id') else 'administrative',
            'credit_class_id': record.get('credit_class_id')
        }
    app.logger.info(f"[STARTUP] Loaded {len(attendance_records)} attendance records into cache")
except Exception as e:
    app.logger.warning(f"[STARTUP] Failed to load attendance cache: {e}")

# Dọn dẹp các bản ghi lớp học cũ bị lỗi nếu có
LEGACY_CLASS_NAMES = [
    'Công nghệ thông tin 01',
    'Công nghệ thông tin 01',
]
for legacy_name in LEGACY_CLASS_NAMES:
    removed = db.delete_class_by_name(legacy_name)
    if removed:
        app.logger.info("Removed legacy class entry: %s (records: %s)", legacy_name, removed)

# Cấu hình chế độ demo (sử dụng os.getenv sau khi load_dotenv)
DEMO_MODE = os.getenv('DEMO_MODE', '0') == '1'
USE_FACENET = os.getenv('USE_FACENET', '1') == '1'  # Sử dụng FaceNet thay vì face_recognition

# Import các dịch vụ nhận diện khuôn mặt nâng cao (dựa trên FaceNet)
face_service = None
antispoof_service = None
training_service = None
FACE_RECOGNITION_AVAILABLE = False

if USE_FACENET and not DEMO_MODE:
    try:
        from services.face_service import FaceRecognitionService
        from services.antispoof_service import AntiSpoofService
        from services.training_service import TrainingService

        face_service = FaceRecognitionService(
            confidence_threshold=float(os.getenv('FACENET_THRESHOLD', '0.85'))
        )

        antispoof_service = AntiSpoofService(
            device=os.getenv('ANTISPOOF_DEVICE', 'cpu'),
            spoof_threshold=float(os.getenv('ANTISPOOF_THRESHOLD', '0.5'))
        )

        app.logger.info("FaceNet services initialized successfully")
        FACE_RECOGNITION_AVAILABLE = True
    except Exception as e:
        app.logger.warning(f"Could not initialize FaceNet services: {e}")
        app.logger.info("Falling back to legacy face_recognition library")
        USE_FACENET = False

# Cố gắng import DeepFace và DeepFace DB helper
DEEPFACE_AVAILABLE = False
try:
    from deepface import DeepFace
    from services.deepface_db import build_db_from_data_dir, recognize_face as deepface_recognize
    DEEPFACE_AVAILABLE = True
    app.logger.info("DeepFace library available - using Facenet512 for face recognition")
except ImportError:
    app.logger.warning("DeepFace not available, will try face_recognition as fallback")

# Cố gắng import YOLOv8 để phát hiện khuôn mặt (tùy chọn nhưng khuyến nghị)
YOLO_AVAILABLE = False
yolo_face_model = None
try:
    from ultralytics import YOLO
    # Try to load YOLOv8 face detection model từ nhiều vị trí có thể
    possible_paths = [
        os.path.join(os.path.dirname(__file__), 'yolov8m-face.pt'),  # Thư mục gốc
        os.path.join(os.path.dirname(__file__), 'models', 'yolov8m-face.pt'),  # Thư mục models
        os.path.join(os.path.dirname(__file__), 'Cong-Nghe-Xu-Ly-Anh', 'yolov8m-face.pt'),  # Thư mục tham khảo (dự phòng)
    ]
    
    yolo_model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            yolo_model_path = path
            break
    
    if yolo_model_path:
        yolo_face_model = YOLO(yolo_model_path)
        YOLO_AVAILABLE = True
        app.logger.info(f"YOLOv8 face detection model loaded successfully from {yolo_model_path}")
    else:
        app.logger.warning("YOLOv8 model not found. Tried paths: " + ", ".join(possible_paths))
        app.logger.info("Hệ thống sẽ sử dụng face_recognition hoặc DeepFace face detection thay thế")
except ImportError:
    app.logger.warning("YOLOv8 (ultralytics) not available - install with: pip install ultralytics")
except Exception as e:
    app.logger.warning(f"Could not load YOLOv8 model: {e}")

# Dự phòng: Import thư viện face_recognition cũ
if not USE_FACENET and not DEEPFACE_AVAILABLE:
    try:
        import face_recognition
        FACE_RECOGNITION_AVAILABLE = True
        app.logger.info("Using legacy face_recognition library")
    except ImportError as e:
        FACE_RECOGNITION_AVAILABLE = False
        if not DEMO_MODE:
            # Nếu không có face_recognition và không phải demo mode, tự động chuyển sang demo mode
            print("Face recognition not available, switching to demo mode...")
            DEMO_MODE = True
            print("Demo mode: face_recognition not available, using simulation mode")
        else:
            print("Demo mode: face_recognition not available, using simulation mode")
else:
    # Nếu chúng ta có DeepFace hoặc FaceNet, chúng ta có thể thực hiện nhận diện khuôn mặt
    if not FACE_RECOGNITION_AVAILABLE:
        FACE_RECOGNITION_AVAILABLE = DEEPFACE_AVAILABLE or USE_FACENET

# Chỉ số thiết bị camera (sử dụng os.getenv sau khi load_dotenv)
CAMERA_INDEX = int(os.getenv('CAMERA_INDEX', '0'))
CAMERA_WIDTH = int(os.getenv('CAMERA_WIDTH', '640'))
CAMERA_HEIGHT = int(os.getenv('CAMERA_HEIGHT', '480'))
CAMERA_WARMUP_FRAMES = int(os.getenv('CAMERA_WARMUP_FRAMES', '3'))
CAMERA_BUFFER_SIZE = int(os.getenv('CAMERA_BUFFER_SIZE', '2'))

# Ngưỡng confidence cho face recognition
FACE_RECOGNITION_THRESHOLD = float(os.getenv('FACE_RECOGNITION_THRESHOLD', '0.6'))
FACE_DISTANCE_THRESHOLD = float(os.getenv('FACE_DISTANCE_THRESHOLD', '0.45'))  # Khoảng cách tối đa (càng nhỏ càng giống)
# Ngưỡng cho DeepFace cosine similarity (similarity >= threshold được chấp nhận)
DEEPFACE_SIMILARITY_THRESHOLD = float(os.getenv('DEEPFACE_SIMILARITY_THRESHOLD', '0.6'))  # Cosine similarity tối thiểu
# Xác nhận dựa trên thời gian + tư thế (yêu cầu nhìn thẳng trong N giây)
LOOK_STRAIGHT_SECONDS = float(os.getenv('LOOK_STRAIGHT_SECONDS', '10'))  # seconds
FRONTAL_YAW_RATIO_THRESHOLD = float(os.getenv('FRONTAL_YAW_RATIO_THRESHOLD', '0.15'))
FRONTAL_ROLL_DEG_THRESHOLD = float(os.getenv('FRONTAL_ROLL_DEG_THRESHOLD', '15'))
# Tối ưu hiệu năng phát hiện
YOLO_FRAME_SKIP = max(1, int(os.getenv('YOLO_FRAME_SKIP', '2')))  # Chỉ chạy YOLO mỗi N khung hình
YOLO_INFERENCE_WIDTH = int(os.getenv('YOLO_INFERENCE_WIDTH', '640'))  # Resize YOLO, 0 = giữ nguyên
SESSION_DURATION_MINUTES = max(1, int(os.getenv('SESSION_DURATION_MINUTES', '15')))

# Đường dẫn thư mục dữ liệu
DATA_DIR = Path('data')


def _normalize_student_dir_name(student_id: Optional[str]) -> str:
    """Sinh tên thư mục an toàn cho sinh viên, ưu tiên dùng mã số."""
    if not student_id:
        return 'student'
    normalized = secure_filename(str(student_id).strip()) or 'student'
    return normalized.lower()


def get_student_data_dir(student_id: Optional[str]) -> Path:
    """Trả về thư mục chứa ảnh của sinh viên trong DATA_DIR."""
    return DATA_DIR / _normalize_student_dir_name(student_id)


def ensure_student_data_dir(student_id: Optional[str]) -> Path:
    """Đảm bảo thư mục lưu ảnh của sinh viên tồn tại."""
    target_dir = get_student_data_dir(student_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def build_student_image_path(student_id: Optional[str], filename: str) -> Path:
    """Ghép đường dẫn file ảnh trong thư mục của sinh viên."""
    student_dir = ensure_student_data_dir(student_id)
    return student_dir / filename


RESERVED_DATA_SUBDIRS = {'training_samples', 'models', 'external_assets'}


def iter_student_face_image_files():
    """Duyệt qua tất cả ảnh mẫu sinh viên (bao gồm thư mục con)."""
    if not DATA_DIR.exists():
        return []

    allowed_suffixes = {f'.{ext.lower()}' for ext in ALLOWED_EXTENSIONS}
    files = []

    for entry in DATA_DIR.iterdir():
        if entry.is_file() and entry.suffix.lower() in allowed_suffixes:
            files.append(entry)
            continue

        if not entry.is_dir() or entry.name in RESERVED_DATA_SUBDIRS:
            continue

        for sub_path in entry.rglob('*'):
            if sub_path.is_file() and sub_path.suffix.lower() in allowed_suffixes:
                files.append(sub_path)

    return files


def _resolve_existing_image_path(path_str: Optional[str]) -> Optional[Path]:
    if not path_str:
        return None

    candidates = []
    raw_path = Path(path_str)
    candidates.append(raw_path)
    candidates.append(Path.cwd() / raw_path)

    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate
        except OSError:
            continue
    return None


def _move_image_into_student_dir(path_str: Optional[str], student_id: Optional[str]) -> Optional[str]:
    if not student_id:
        return None
    source_path = _resolve_existing_image_path(path_str)
    if source_path is None:
        return None

    target_dir = ensure_student_data_dir(student_id)
    try:
        relative_inside = source_path.resolve().relative_to(target_dir.resolve())
        normalized_path = target_dir / relative_inside
        return str(normalized_path)
    except Exception:
        pass

    target_path = target_dir / source_path.name
    counter = 1
    while target_path.exists():
        target_path = target_dir / f"{source_path.stem}_{counter:02d}{source_path.suffix}"
        counter += 1

    try:
        shutil.move(str(source_path), str(target_path))
    except Exception as exc:
        app.logger.warning("[DataReorg] Không thể di chuyển %s -> %s: %s", source_path, target_path, exc)
        return None

    return str(target_path)


def _infer_student_id_from_filename(filename: str) -> Optional[str]:
    if not filename:
        return None
    match = re.match(r'^([A-Za-z0-9]+)', filename)
    if match:
        return match.group(1)
    return None


def organize_student_data_directories():
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        students = db.get_all_students(active_only=False)
    except Exception as exc:
        app.logger.warning("[DataReorg] Không thể tải danh sách sinh viên: %s", exc)
        return

    moved_samples = 0
    updated_primary = 0

    for student in students:
        student_data = student if isinstance(student, dict) else dict(student)
        student_id = student_data.get('student_id')
        if not student_id:
            continue

        ensure_student_data_dir(student_id)

        primary_path = student_data.get('face_image_path')
        new_primary = _move_image_into_student_dir(primary_path, student_id)
        if new_primary and new_primary != primary_path:
            try:
                db.update_student(student_id, face_image_path=new_primary)
                updated_primary += 1
            except Exception as exc:
                app.logger.warning("[DataReorg] Không thể cập nhật avatar của %s: %s", student_id, exc)

        try:
            samples = db.get_face_samples(student_id)
        except Exception as exc:
            app.logger.warning("[DataReorg] Không thể tải ảnh mẫu cho %s: %s", student_id, exc)
            continue

        for sample in samples:
            sample_dict = sample if isinstance(sample, dict) else dict(sample)
            sample_id = sample_dict.get('id')
            sample_path = sample_dict.get('image_path')
            new_path = _move_image_into_student_dir(sample_path, student_id)
            if new_path and new_path != sample_path:
                if db.update_face_sample_path(sample_id, new_path):
                    moved_samples += 1

    loose_moves = 0
    allowed_suffixes = {f'.{ext.lower()}' for ext in ALLOWED_EXTENSIONS}
    for legacy_file in DATA_DIR.iterdir():
        if not legacy_file.is_file() or legacy_file.suffix.lower() not in allowed_suffixes:
            continue
        if legacy_file.parent != DATA_DIR:
            continue
        inferred_id = _infer_student_id_from_filename(legacy_file.stem)
        if not inferred_id:
            continue
        new_path = _move_image_into_student_dir(str(legacy_file), inferred_id)
        if new_path and new_path != str(legacy_file):
            loose_moves += 1

    if moved_samples or updated_primary or loose_moves:
        app.logger.info(
            "[DataReorg] Hoàn tất: di chuyển %d ảnh mẫu, cập nhật %d ảnh đại diện, sắp xếp %d ảnh lẻ",
            moved_samples,
            updated_primary,
            loose_moves,
        )


organize_student_data_directories()

# Hệ thống camera/vision
vision_state: Optional[VisionPipelineState] = None
camera_enabled = True  # Biến để bật/tắt camera
inference_engine: Optional[InferenceEngine] = None

# Khởi tạo biến global cho nhận diện khuôn mặt
known_face_encodings = []
known_face_names = []
known_face_ids = []
# DeepFace DB embeddings (Facenet512)
known_face_embeddings = []  # np.ndarray (N, D)
today_checked_in = set()  # student_ids đã check-in
today_checked_out = set()  # student_ids đã checkout
today_student_names = {}  # student_id -> name
today_recorded_lock = threading.Lock()

# Bộ nhớ đệm phiên điểm danh tín chỉ
current_credit_session = None
current_session_lock = threading.Lock()

# Theo dõi thời gian có mặt
presence_tracking = {}  # {student_id: {'last_seen': datetime, 'total_time': seconds}}
presence_tracking_lock = threading.Lock()
PRESENCE_TIMEOUT = 300  # 5 phút (300 giây) - nếu không thấy sẽ tự checkout

# Tiến độ xác nhận điểm danh cho mỗi sinh viên khi streaming
attendance_progress = {}
attendance_progress_lock = threading.Lock()

# Chống trùng lặp điểm danh (từ hệ thống mẫu Cong-Nghe-Xu-Ly-Anh)

# Chỉ cho phép điểm danh lại sau 30 giây (tránh điểm danh liên tục)
last_recognized = {}  # {student_id: datetime}
last_recognized_lock = threading.Lock()
RECOGNITION_COOLDOWN = 30  # Giây - thời gian chờ giữa các lần điểm danh

# Server-Sent Events cho thông báo thời gian thực
import queue
sse_clients = []  # Danh sách các hàng đợi cho mỗi client SSE đã kết nối
sse_clients_lock = threading.Lock()


def parse_datetime_safe(value):
    """Chuyển chuỗi datetime thành đối tượng datetime, trả về None nếu lỗi."""
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        try:
            return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None


def get_request_data():
    """Hợp nhất form/JSON payload thành một dict có thể thay đổi."""
    if request.is_json:
        return request.get_json() or {}
    if request.form:
        return request.form.to_dict()
    return request.get_json(silent=True) or {}


def parse_bool(value, default=None):
    """Chuyển đổi đầu vào string/int/bool thành giá trị boolean."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    value = str(value).strip().lower()
    if value in {'1', 'true', 'on', 'yes', 'y'}:
        return True
    if value in {'0', 'false', 'off', 'no', 'n'}:
        return False
    return default


def _session_deadline_raw(session_row):
    if not session_row:
        return None
    return session_row.get('checkin_deadline') or session_row.get('checkout_deadline')


def session_is_active(session_row):
    """Kiểm tra phiên điểm danh còn hiệu lực (status=open và chưa hết hạn)."""
    if not session_row or session_row.get('status') != 'open':
        return False
    expires_at = parse_datetime_safe(_session_deadline_raw(session_row))
    if expires_at and expires_at <= datetime.now():
        return False
    return True


def serialize_session_payload(session_row):
    """Chuyển phiên điểm danh thành payload JSON-friendly."""
    if not session_row:
        return None
    payload = {
        'id': session_row.get('id'),
        'credit_class_id': session_row.get('credit_class_id'),
        'class_name': session_row.get('credit_class_name'),
        'class_code': session_row.get('credit_code'),
        'status': session_row.get('status'),
        'opened_at': session_row.get('opened_at'),
        'session_date': session_row.get('session_date'),
        'checkin_deadline': session_row.get('checkin_deadline'),
        'checkout_deadline': session_row.get('checkout_deadline'),
        'notes': session_row.get('notes'),
    }
    expires_at = parse_datetime_safe(_session_deadline_raw(session_row))
    payload['expires_at'] = expires_at.isoformat() if expires_at else None
    if expires_at:
        payload['remaining_seconds'] = max(int((expires_at - datetime.now()).total_seconds()), 0)
    else:
        payload['remaining_seconds'] = None
    return payload


def row_to_dict(row):
    """Chuyển sqlite3.Row thành dict (nếu có thể)."""
    if row is None:
        return None
    if isinstance(row, dict):
        return row
    try:
        return dict(row)
    except Exception:
        return row


def get_current_role():
    user = getattr(g, 'user', None)
    if not user:
        return None
    return (user.get('role') or '').lower()


def resolve_teacher_context(teacher_id=None):
    """Xác định bản ghi giảng viên tương ứng với người dùng hiện tại."""
    user = getattr(g, 'user', None)
    if not user:
        return None
    role = get_current_role()
    if role == 'teacher':
        teacher = db.get_teacher_by_user(user['id'])
        if not teacher:
            teacher = db.ensure_teacher_profile(user)
        return row_to_dict(teacher)
    if role == 'admin' and teacher_id:
        return row_to_dict(db.get_teacher(teacher_id))
    return None


def resolve_student_context(student_identifier=None, auto_link=True):
    """Tìm sinh viên cho user hiện tại hoặc theo student_id được cung cấp."""
    user = getattr(g, 'user', None)
    role = get_current_role()

    if student_identifier:
        return row_to_dict(db.get_student(student_identifier))

    if not user:
        return None

    if role == 'student':
        student_row = db.get_student_by_user(user['id'])
        if student_row:
            return row_to_dict(student_row)

        username = (user.get('username') or '').strip()
        if username:
            student_row = db.get_student(username)
            if student_row and auto_link:
                try:
                    db.link_student_to_user(username, user['id'])
                except Exception as exc:
                    app.logger.debug("Không thể tự liên kết sinh viên %s với user %s: %s", username, user['id'], exc)
            return row_to_dict(student_row)

    if role == 'admin' and student_identifier:
        return row_to_dict(db.get_student(student_identifier))

    return None


def get_active_attendance_session(force_reload=False):
    """Trả về phiên điểm danh đang mở (và cập nhật cache khi cần)."""
    global current_credit_session
    with current_session_lock:
        if force_reload:
            current_credit_session = None
        else:
            if current_credit_session and not session_is_active(current_credit_session):
                current_credit_session = None

        try:
            db.expire_attendance_sessions()
        except Exception as exc:
            app.logger.debug("Không thể cập nhật trạng thái phiên: %s", exc)

        if current_credit_session is None:
            session_row = db.get_current_open_session()
            current_credit_session = session_row if session_row else None

        if current_credit_session and not session_is_active(current_credit_session):
            current_credit_session = None

        return current_credit_session


def set_active_session_cache(session_row):
    """Ghi đè cache phiên hiện tại."""
    global current_credit_session
    with current_session_lock:
        current_credit_session = session_row
        return current_credit_session


def broadcast_session_snapshot(force_reload=False):
    """Phát sự kiện SSE về trạng thái phiên điểm danh hiện tại."""
    payload = serialize_session_payload(get_active_attendance_session(force_reload=force_reload))
    broadcast_sse_event({'type': 'session_updated', 'data': payload})


PUBLIC_ENDPOINTS = {'login', 'logout', 'static'}


def sanitize_next_url(next_url):
    """Đảm bảo next_url luôn là đường dẫn nội bộ an toàn."""
    if not next_url:
        return None
    next_url = next_url.strip()
    if not next_url:
        return None
    if next_url.startswith(('http://', 'https://', '//')):
        return None
    if not next_url.startswith('/'):
        return None
    return next_url.rstrip('?') or '/'


def build_next_url():
    """Tạo giá trị next_url dựa trên request hiện tại."""
    if request.method == 'GET':
        candidate = request.full_path or request.path
    else:
        candidate = request.path
    return sanitize_next_url(candidate)


def is_api_request():
    """Kiểm tra request hiện tại có thuộc API không."""
    path = request.path or ''
    return path.startswith('/api/')


def is_public_endpoint(endpoint):
    """Xác định endpoint có được phép truy cập công khai hay không."""
    if not endpoint:
        return False
    if endpoint == 'static' or endpoint.startswith('static.'):
        return True
    return endpoint in PUBLIC_ENDPOINTS


def verify_user_password(user_record, candidate_password):
    """Kiểm tra mật khẩu người dùng (hỗ trợ hash legacy)."""
    if not user_record:
        return False
    stored_hash = user_record.get('password_hash') or ''
    if not stored_hash:
        return False

    if stored_hash.startswith(('pbkdf2:', 'scrypt:')):
        return check_password_hash(stored_hash, candidate_password)

    legacy_hash = hashlib.sha256(candidate_password.encode('utf-8')).hexdigest()
    if legacy_hash == stored_hash:
        try:
            new_hash = generate_password_hash(candidate_password)
            db.update_user_password(user_record['id'], new_hash)
            user_record['password_hash'] = new_hash
            app.logger.info("Đã nâng cấp hash mật khẩu cho người dùng %s", user_record.get('username'))
        except Exception as exc:
            app.logger.warning("Không thể nâng cấp hash mật khẩu: %s", exc)
        return True

    return False


def login_user(user_record):
    """Thiết lập session cho người dùng đã xác thực."""
    session.clear()
    session['user_id'] = user_record['id']
    session['user_role'] = user_record.get('role')
    session['user_name'] = user_record.get('full_name')
    session.permanent = True


def logout_current_user():
    """Đăng xuất người dùng hiện tại."""
    session.clear()


def role_required(*roles):
    """Decorator kiểm tra quyền truy cập dựa trên vai trò."""
    allowed_roles = {role.lower() for role in roles if role}

    def decorator(view_func):
        @wraps(view_func)
        def wrapper(*args, **kwargs):
            user = getattr(g, 'user', None)
            if not user:
                next_url = build_next_url()
                if is_api_request():
                    return jsonify({'success': False, 'message': 'Yêu cầu đăng nhập'}), 401
                if next_url:
                    return redirect(url_for('login', next=next_url))
                return redirect(url_for('login'))

            user_role = (user.get('role') or '').lower()
            if user_role != 'admin' and allowed_roles and user_role not in allowed_roles:
                app.logger.warning(
                    "User %s bị chặn truy cập %s (cần %s)",
                    user.get('username'),
                    request.path,
                    ','.join(allowed_roles) or 'any',
                )
                if is_api_request():
                    return jsonify({'success': False, 'message': 'Không có quyền truy cập'}), 403
                return abort(403)

            return view_func(*args, **kwargs)

        return wrapper

    return decorator


@app.before_request
def load_logged_in_user():
    """Nạp thông tin người dùng và bảo vệ các route yêu cầu đăng nhập."""
    user_id = session.get('user_id')
    g.user = db.get_user_by_id(user_id) if user_id else None

    if is_public_endpoint(request.endpoint):
        return

    if request.path.startswith('/static/'):
        return

    if g.user is None:
        if is_api_request():
            return jsonify({'success': False, 'message': 'Yêu cầu đăng nhập'}), 401
        next_url = build_next_url()
        if next_url:
            return redirect(url_for('login', next=next_url))
        return redirect(url_for('login'))


@app.context_processor
def inject_user_context():
    """Cung cấp user/role hiện tại cho tất cả các template."""
    user = getattr(g, 'user', None)
    role = user.get('role') if isinstance(user, dict) else None
    return {
        'current_user': user,
        'current_role': role,
    }


def safe_delete_file(path):
    """Cố gắng xóa một file mà không báo lỗi nếu thất bại."""
    if not path:
        return
    try:
        os.remove(path)
    except OSError:
        app.logger.debug("Could not remove file %s", path)


def _generate_face_image_filename(student_id, full_name, *, suffix=None, extension='jpg', timestamp=None):
    safe_base = secure_filename(f"{student_id}_{full_name}".strip()) or secure_filename(student_id) or 'student'
    timestamp = timestamp or datetime.now().strftime('%Y%m%d%H%M%S')
    suffix_part = f"_{suffix}" if suffix is not None else ''
    return f"{safe_base}_{timestamp}{suffix_part}.{extension}"


def save_uploaded_face_image(file_storage, student_id, full_name, *, suffix=None, timestamp=None):
    """Lưu ảnh khuôn mặt đã tải lên sau khi xác thực."""
    if not file_storage or not file_storage.filename:
        return None

    _, ext = os.path.splitext(file_storage.filename)
    ext = (ext or '').lower().lstrip('.')
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Định dạng file không hợp lệ. Chỉ cho phép: {', '.join(sorted(ALLOWED_EXTENSIONS))}")

    filename = _generate_face_image_filename(
        student_id,
        full_name,
        suffix=suffix,
        extension=ext,
        timestamp=timestamp,
    )
    file_path = build_student_image_path(student_id, filename)
    file_storage.save(str(file_path))

    success, error_msg, _ = validate_image_file(str(file_path), is_base64=False)
    if not success:
        safe_delete_file(str(file_path))
        raise ValueError(f"Ảnh không hợp lệ: {error_msg}")

    return str(file_path)


def save_base64_face_image(image_data, student_id, full_name, *, suffix=None, timestamp=None):
    """Giải mã ảnh base64 và lưu xuống đĩa sau khi xác thực."""
    if not image_data:
        raise ValueError('Thiếu dữ liệu ảnh')

    if ',' in image_data:
        image_data = image_data.split(',')[1]
    try:
        img_bytes = base64.b64decode(image_data)
    except Exception as exc:
        raise ValueError('Ảnh không hợp lệ: Không thể giải mã dữ liệu base64') from exc

    filename = _generate_face_image_filename(
        student_id,
        full_name,
        suffix=suffix,
        extension='jpg',
        timestamp=timestamp,
    )
    file_path = build_student_image_path(student_id, filename)
    with open(file_path, 'wb') as fp:
        fp.write(img_bytes)

    success, error_msg, _ = validate_image_file(str(file_path), is_base64=False)
    if not success:
        safe_delete_file(str(file_path))
        raise ValueError(f"Ảnh không hợp lệ: {error_msg}")

    return str(file_path)


def extract_face_encoding(image_path):
    """Tạo face encoding từ file ảnh đã lưu (trả về bytes hoặc None nếu thất bại)."""
    if not FACE_RECOGNITION_AVAILABLE or not image_path:
        return None
    try:
        if not os.path.exists(image_path):
            return None
        image = face_recognition.load_image_file(image_path)
        locations = face_recognition.face_locations(image)
        if not locations:
            return None
        encodings = face_recognition.face_encodings(image, known_face_locations=locations, num_jitters=1)
        if not encodings:
            return None
        return encodings[0].tobytes()
    except Exception as exc:
        app.logger.warning("Không thể tạo face encoding từ %s: %s", image_path, exc)
        return None


def serialize_student_record(student_row, class_map=None):
    """Chuyển đổi bản ghi sinh viên sqlite3.Row thành dict có thể serialize."""
    if not student_row:
        return None

    student = dict(student_row)
    class_id = student.get('class_id')
    class_name = None
    if class_id:
        if class_map is not None:
            class_name = class_map.get(class_id)
        else:
            class_info = db.get_class_by_id(class_id)
            class_name = class_info.get('class_name') if class_info else None

    return {
        'id': student.get('id'),
        'student_id': student.get('student_id'),
        'full_name': student.get('full_name'),
        'email': student.get('email'),
        'phone': student.get('phone'),
        'class_id': class_id,
        'class_name': class_name,
        'face_image_path': student.get('face_image_path'),
        'status': student.get('status'),
        'is_active': bool(student.get('is_active')),
        'created_at': student.get('created_at'),
        'updated_at': student.get('updated_at'),
    }


def serialize_credit_class_record(credit_row):
    """Normalize bản ghi lớp tín chỉ, bổ sung thông tin giảng viên."""
    if not credit_row:
        return None

    payload = dict(credit_row)
    if not payload.get('teacher_name') and payload.get('teacher_id'):
        teacher = db.get_teacher(payload['teacher_id'])
        if teacher:
            payload['teacher_name'] = teacher.get('full_name') or teacher.get('teacher_code')
            payload['teacher_code'] = teacher.get('teacher_code')
    return payload


def serialize_teacher_record(teacher_row):
    """Chuẩn hóa bản ghi giảng viên, kèm thông tin tài khoản người dùng."""
    if not teacher_row:
        return None

    teacher = dict(teacher_row)
    teacher['is_active'] = bool(teacher.get('is_active', 1))
    user_id = teacher.get('user_id')
    if user_id:
        user = db.get_user_by_id(user_id)
        if user:
            teacher['username'] = user.get('username')
            teacher['user_email'] = user.get('email')
            teacher['user_is_active'] = bool(user.get('is_active', 1))
            teacher['user_last_login'] = user.get('last_login')
    return teacher

def get_or_create_vision_state() -> VisionPipelineState:
    global vision_state
    if vision_state is None:
        config = VisionStateConfig(
            index=CAMERA_INDEX,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            warmup_frames=CAMERA_WARMUP_FRAMES,
            buffer_size=CAMERA_BUFFER_SIZE,
        )
        vision_state = VisionPipelineState(config=config, logger=app.logger)
    return vision_state


def ensure_camera_pipeline():
    if not camera_enabled:
        return None
    state = get_or_create_vision_state()
    state.set_enabled(True)
    try:
        return state.ensure_ready()
    except CameraError as exc:
        app.logger.error("[Camera] ❌ Không thể khởi động camera: %s", exc)
        return None


def release_camera_capture():
    state = vision_state
    if state is None:
        return
    try:
        state.set_enabled(False)
        state.stop()
    except Exception as exc:
        app.logger.debug("[Camera] ⚠️ Không thể giải phóng camera: %s", exc)


def lookup_student_name(student_id: Optional[str]) -> Optional[str]:
    if not student_id:
        return None
    try:
        student = db.get_student(student_id)
        if student:
            return student.get('full_name') or student.get('student_name') or student_id
    except Exception as exc:
        app.logger.debug("[Inference] Lookup failed cho %s: %s", student_id, exc)
    return None


def configure_inference_engine():
    """Khởi tạo inference engine với các chiến lược phù hợp."""
    global inference_engine
    try:
        inference_engine = InferenceEngine(logger=app.logger, demo_mode=DEMO_MODE)
    except Exception as exc:
        app.logger.warning("[Inference] Không thể khởi tạo InferenceEngine: %s", exc)
        inference_engine = None
        return

    if DEEPFACE_AVAILABLE:
        try:
            deepface_strategy = DeepFaceStrategy(
                data_dir=DATA_DIR,
                deepface_module=DeepFace,
                build_db_fn=build_db_from_data_dir,
                recognize_fn=deepface_recognize,
                similarity_threshold=DEEPFACE_SIMILARITY_THRESHOLD,
                enforce_detection=False,
                logger=app.logger,
            )
            inference_engine.add_strategy(deepface_strategy)
        except Exception as exc:
            app.logger.warning("[Inference] Không thể khởi tạo DeepFace strategy: %s", exc)

    if USE_FACENET and face_service is not None:
        try:
            facenet_strategy = FaceNetStrategy(
                service=face_service,
                label_lookup=lookup_student_name,
                logger=app.logger,
            )
            inference_engine.add_strategy(facenet_strategy)
        except Exception as exc:
            app.logger.warning("[Inference] Không thể khởi tạo FaceNet strategy: %s", exc)


configure_inference_engine()

# ============================================================================
# HỆ THỐNG NHẬN DIỆN VÀ ĐIỂM DANH - VIẾT LẠI DỰA TRÊN DỰ ÁN THAM KHẢO
# Logic từ: Cong-Nghe-Xu-Ly-Anh/diemdanh_deepface_gui.py
# ============================================================================

def load_known_faces(force_reload: bool = True):
    """Tải các khuôn mặt đã biết, ưu tiên inference engine nếu khả dụng."""
    global known_face_embeddings, known_face_names, known_face_ids

    app.logger.info(f"[LoadFaces] 🔄 Khởi động lại dữ liệu khuôn mặt từ {DATA_DIR}...")

    engine_ready = inference_engine is not None and inference_engine.has_strategies()
    if engine_ready:
        try:
            summary = (
                inference_engine.reload()
                if force_reload
                else inference_engine.warmup(force=False)
            )
            subjects = inference_engine.known_subjects(limit=10_000)
            known_face_embeddings = []
            known_face_ids = []
            known_face_names = []
            for student_id, name in subjects:
                normalized_id = (student_id or name or "UNKNOWN").strip()
                known_face_ids.append(normalized_id)
                known_face_names.append(name or normalized_id)
            app.logger.info(
                "[LoadFaces] ✅ Inference engine sẵn sàng với %d khuôn mặt",
                len(known_face_ids),
            )
            return summary
        except InferenceError as error:
            app.logger.warning(
                "[LoadFaces] ⚠️ Inference engine reload thất bại: %s. Fallback legacy.",
                error,
            )
        except Exception as exc:
            app.logger.error(
                "[LoadFaces] ⚠️ Không thể reload inference engine: %s. Fallback legacy.",
                exc,
                exc_info=True,
            )

    if not DEEPFACE_AVAILABLE:
        app.logger.error(
            "[LoadFaces] ❌ DeepFace không khả dụng. Vui lòng cài đặt: pip install deepface"
        )
        return

    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        app.logger.info(f"[LoadFaces] ✅ Đã tạo thư mục {DATA_DIR}")

    app.logger.info("[LoadFaces] [DeepFace] 🧠 Đang tải ảnh mẫu và tính embedding với Facenet512...")
    db_embeddings = []
    db_labels = []
    processed_count = 0
    failed_count = 0
    image_files = iter_student_face_image_files()
    app.logger.info(f"[LoadFaces] 📁 Tìm thấy {len(image_files)} file ảnh (gồm cả thư mục con)")

    for img_path in image_files:
        try:
            filename = img_path.stem
            import re
            student_id = None
            name = None

            try:
                relative_parts = img_path.relative_to(DATA_DIR).parts
            except ValueError:
                relative_parts = ()

            if len(relative_parts) > 1 and relative_parts[0] not in RESERVED_DATA_SUBDIRS:
                student_id = relative_parts[0]

            match = re.match(r'^(\d+)_([A-Za-z\s]+)', filename)
            if match:
                student_id = student_id or match.group(1)
                name = match.group(2).strip()
            else:
                parts = filename.split('_')
                if len(parts) >= 2:
                    student_id = student_id or parts[0]
                    name = '_'.join(parts[1:])
                else:
                    student_id = student_id or filename
                    name = filename

            app.logger.debug(
                f"[LoadFaces] Đang xử lý {img_path.name} -> {name} (ID: {student_id})..."
            )

            embedding = DeepFace.represent(
                img_path=str(img_path),
                model_name="Facenet512",
                enforce_detection=True,
            )[0]["embedding"]

            db_embeddings.append(embedding)
            db_labels.append((student_id, name))
            processed_count += 1
            app.logger.info(
                f"[LoadFaces] ✅ Đã tải khuôn mặt cho {name} (id={student_id}) từ {img_path.name}"
            )
        except Exception as e:
            failed_count += 1
            app.logger.error(
                f"[LoadFaces] ❌ Lỗi khi xử lý ảnh mẫu {img_path.name}: {e}",
                exc_info=True,
            )

    if len(db_embeddings) > 0:
        known_face_embeddings = np.array(db_embeddings)
        known_face_ids = [sid for sid, _ in db_labels]
        known_face_names = [name for _, name in db_labels]
        app.logger.info(
            f"[LoadFaces] ✅ Đã load {len(known_face_embeddings)} ảnh mẫu với Facenet512 embeddings"
        )
        app.logger.info(f"[LoadFaces] 📋 Known faces: {known_face_names}")
        app.logger.info(f"[LoadFaces] 📋 Known IDs: {known_face_ids}")
        app.logger.info(f"[LoadFaces] 📐 Embeddings shape: {known_face_embeddings.shape}")
        app.logger.info(
            f"[LoadFaces] 📊 Kết quả: {processed_count} thành công, {failed_count} thất bại"
        )
    else:
        app.logger.warning("[LoadFaces] ⚠️ Không load được ảnh nào!")


def ensure_legacy_embeddings(force_reload: bool = False) -> None:
    """Đảm bảo bộ embeddings DeepFace được build khi không có inference engine."""
    global known_face_embeddings
    if not DEEPFACE_AVAILABLE:
        return
    engine_ready = inference_engine is not None and inference_engine.has_strategies()
    if engine_ready:
        return  # ưu tiên inference engine
    needs_reload = force_reload or not known_face_embeddings or len(known_face_embeddings) == 0
    if not needs_reload:
        return
    try:
        load_known_faces(force_reload=force_reload)
    except Exception as exc:
        app.logger.warning("[LoadFaces] ⚠️ Không thể build legacy embeddings: %s", exc)

def validate_image_file(file_path, is_base64=False):
    """
    Validate ảnh trước khi lưu vào hệ thống
    
    Args:
        file_path: Đường dẫn file ảnh hoặc base64 data
        is_base64: True nếu file_path là base64 data
        
    Returns:
        tuple: (success: bool, error_message: str, face_count: int)
    """
    try:
        # Kiểm tra kích thước file
        if not is_base64:
            if not os.path.exists(file_path):
                return False, "File không tồn tại", 0
            
            file_size = os.path.getsize(file_path)
            if file_size < MIN_FILE_SIZE:
                return False, f"File quá nhỏ ({file_size} bytes). Tối thiểu {MIN_FILE_SIZE} bytes", 0
            
            if file_size > MAX_FILE_SIZE:
                return False, f"File quá lớn ({file_size / 1024 / 1024:.1f} MB). Tối đa {MAX_FILE_SIZE / 1024 / 1024} MB", 0
        
        # Kiểm tra định dạng ảnh với PIL
        if PIL_AVAILABLE:
            img = None
            img_stream = None
            try:
                if is_base64:
                    image_data = file_path
                    if ',' in image_data:
                        image_data = image_data.split(',')[1]
                    img_bytes = base64.b64decode(image_data)
                    img_stream = io.BytesIO(img_bytes)
                    img = Image.open(img_stream)
                else:
                    img = Image.open(file_path)

                detected_format = (img.format or '').upper()
                if detected_format == 'JPG':
                    detected_format = 'JPEG'

                if not detected_format or detected_format not in SUPPORTED_IMAGE_FORMATS:
                    message = (
                        "Không xác định được định dạng ảnh"
                        if not detected_format
                        else f"Định dạng không được hỗ trợ: {detected_format}. Chỉ chấp nhận JPG, JPEG, PNG"
                    )
                    return False, message, 0

                webp_detected = detected_format == 'WEBP'

                if img.mode not in ['RGB', 'L', 'RGBA']:
                    return False, f"Chế độ màu không được hỗ trợ: {img.mode}. Cần RGB hoặc Grayscale", 0

                if img.mode != 'RGB':
                    converted = img.convert('RGB')
                    img.close()
                    img = converted

                width, height = img.size
                if width < 100 or height < 100:
                    return False, f"Ảnh quá nhỏ ({width}x{height}). Tối thiểu 100x100 pixels", 0

                if webp_detected and not is_base64:
                    try:
                        img.save(file_path, 'JPEG', quality=95)
                    except Exception as exc:
                        return False, f"Không thể chuyển WEBP sang JPEG: {exc}", 0

            except Exception as e:
                return False, f"Lỗi đọc ảnh: {str(e)}", 0
            finally:
                if img is not None:
                    try:
                        img.close()
                    except Exception:
                        pass
                if img_stream is not None:
                    img_stream.close()
        
        # Kiểm tra phát hiện khuôn mặt với face_recognition
        # WORKAROUND: Bỏ qua face detection vì face_recognition v1.2.3 có bug
        # "Unsupported image type" ngay cả với ảnh RGB uint8 hợp lệ
        if FACE_RECOGNITION_AVAILABLE:
            try:
                # Load ảnh
                if is_base64:
                    # Decode base64
                    image_data = file_path
                    if ',' in image_data:
                        image_data = image_data.split(',')[1]
                    img_bytes = base64.b64decode(image_data)
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    # Convert BGR to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = face_recognition.load_image_file(file_path)
                
                # Phát hiện khuôn mặt
                face_locations = face_recognition.face_locations(image)
                
                if len(face_locations) == 0:
                    return False, "Không phát hiện khuôn mặt nào trong ảnh. Vui lòng chụp ảnh rõ mặt, đủ sáng", 0
                
                if len(face_locations) > 1:
                    return False, f"Phát hiện {len(face_locations)} khuôn mặt. Vui lòng chỉ chụp 1 người", len(face_locations)
                
                # Tạo encoding để đảm bảo khuôn mặt có thể encode được
                face_encodings = face_recognition.face_encodings(image, face_locations)
                
                if len(face_encodings) == 0:
                    return False, "Không thể tạo mã hóa khuôn mặt. Vui lòng chụp ảnh rõ hơn", 0
                
                # Kiểm tra chất lượng khuôn mặt (kích thước face trong ảnh)
                top, right, bottom, left = face_locations[0]
                face_width = right - left
                face_height = bottom - top
                
                if face_width < 50 or face_height < 50:
                    return False, f"Khuôn mặt quá nhỏ ({face_width}x{face_height}). Vui lòng chụp gần hơn", 1
                
                # Success!
                return True, "Ảnh hợp lệ", 1
                
            except RuntimeError as e:
                # WORKAROUND: face_recognition có bug "Unsupported image type"
                # Bỏ qua lỗi này và chấp nhận ảnh nếu định dạng cơ bản OK
                if "Unsupported image type" in str(e):
                    app.logger.warning(f"Face detection skipped due to library bug: {e}")
                    return True, "Ảnh hợp lệ (bỏ qua kiểm tra khuôn mặt do lỗi thư viện)", 0
                else:
                    return False, f"Lỗi xử lý khuôn mặt: {str(e)}", 0
            except Exception as e:
                # Các lỗi khác
                app.logger.warning(f"Face detection error: {e}")
                # Vẫn chấp nhận ảnh nếu định dạng cơ bản OK
                return True, f"Ảnh hợp lệ (bỏ qua kiểm tra khuôn mặt: {str(e)})", 0
        else:
            # Nếu không có face_recognition, chỉ kiểm tra định dạng
            return True, "Ảnh hợp lệ (chưa kiểm tra khuôn mặt)", 0
            
    except Exception as e:
        return False, f"Lỗi không xác định: {str(e)}", 0


def recognize_face_candidate(face_img) -> Dict[str, Any]:
    """Nhận diện khuôn mặt sử dụng inference engine hoặc fallback legacy."""
    result = {
        'student_id': 'UNKNOWN',
        'student_name': 'UNKNOWN',
        'confidence': 0.0,
        'strategy': 'none',
        'status': 'unknown',
    }
    engine_ready = inference_engine is not None and inference_engine.has_strategies()
    if engine_ready:
        try:
            inference_result = inference_engine.identify(face_img)
            sid = inference_result.student_id or 'UNKNOWN'
            name = inference_result.student_name or (sid if sid != 'UNKNOWN' else 'UNKNOWN')
            result.update({
                'student_id': sid,
                'student_name': name,
                'confidence': float(inference_result.confidence or 0.0),
                'strategy': inference_result.strategy or 'inference',
                'status': inference_result.status or ('match' if sid != 'UNKNOWN' else 'no_match'),
            })
            return result
        except InferenceError as error:
            app.logger.warning("[Inference] Nhận diện thất bại: %s", error)
        except Exception as exc:
            app.logger.error("[Inference] Lỗi nhận diện không xác định: %s", exc, exc_info=True)

    if DEEPFACE_AVAILABLE:
        ensure_legacy_embeddings(force_reload=False)

    if DEEPFACE_AVAILABLE and known_face_embeddings is not None and len(known_face_embeddings) > 0:
        try:
            legacy_embedding = DeepFace.represent(
                face_img,
                model_name="Facenet512",
                enforce_detection=False,
            )[0]["embedding"]
            db_labels = list(zip(known_face_ids, known_face_names))
            student_id, student_name, best_score = deepface_recognize(
                legacy_embedding,
                known_face_embeddings,
                db_labels,
                threshold=DEEPFACE_SIMILARITY_THRESHOLD,
            )
            sid = student_id or 'UNKNOWN'
            name = student_name or (sid if sid != 'UNKNOWN' else 'UNKNOWN')
            result.update({
                'student_id': sid,
                'student_name': name,
                'confidence': float(best_score or 0.0),
                'strategy': 'legacy-deepface',
                'status': 'match' if student_id else 'no_match',
            })
        except Exception as exc:
            app.logger.error("[Inference] ❌ Lỗi nhận diện legacy: %s", exc, exc_info=True)
    return result

# Load danh sách đã điểm danh hôm nay từ Database
def load_today_recorded(session_id=None, credit_class_id=None):
    global today_checked_in, today_checked_out, today_student_names
    today_checked_in = set()
    today_checked_out = set()
    today_student_names = {}

    session_filter = session_id
    class_filter = credit_class_id
    if session_filter is None:
        session_ctx = get_active_attendance_session()
        if session_ctx:
            session_filter = session_ctx.get('id')
            class_filter = class_filter or session_ctx.get('credit_class_id')

    try:
        attendance_data = db.get_today_attendance(
            session_id=session_filter,
            credit_class_id=class_filter,
        )
        for record in attendance_data:
            record_dict = dict(record) if not isinstance(record, dict) else record
            student_id = record_dict.get('student_id')
            name = record_dict.get('student_name') or record_dict.get('full_name')
            class_name = record_dict.get('credit_class_name') or record_dict.get('class_name')
            class_type = 'credit' if record_dict.get('credit_class_id') else 'administrative'
            if not student_id:
                continue
            today_student_names[student_id] = {
                'name': name or student_id,
                'class_name': class_name,
                'class_type': class_type,
                'credit_class_id': record_dict.get('credit_class_id')
            }
            if record_dict.get('check_in_time'):
                today_checked_in.add(student_id)
            if record_dict.get('check_out_time'):
                today_checked_out.add(student_id)
    except Exception as e:
        app.logger.error(f"Error loading today recorded: {e}")


def reset_session_runtime_state(session_row=None):
    """Đặt lại cache điểm danh & tracking khi chuyển phiên."""
    session_id = session_row.get('id') if isinstance(session_row, dict) else None
    credit_class_id = session_row.get('credit_class_id') if isinstance(session_row, dict) else None
    load_today_recorded(session_id=session_id, credit_class_id=credit_class_id)
    with presence_tracking_lock:
        presence_tracking.clear()
    with attendance_progress_lock:
        attendance_progress.clear()

# Lưu điểm danh vào Database
def mark_attendance(
    name: str,
    student_id: str = '',
    confidence_score: float = None,
    expected_student_id: str = None,
    expected_credit_class_id: int = None,
) -> bool:
    """Lưu điểm danh vào database với các ràng buộc tùy chọn."""
    normalized_student_id = (student_id or '').strip().upper()
    normalized_expected_id = (expected_student_id or '').strip().upper()
    if normalized_expected_id and normalized_student_id and normalized_student_id != normalized_expected_id:
        app.logger.info(
            "[Attendance] Rejecting check-in: recognized %s but expected %s",
            normalized_student_id,
            normalized_expected_id,
        )
        return False
    with today_recorded_lock:
        already_checked_in = normalized_student_id in today_checked_in
        already_checked_out = normalized_student_id in today_checked_out
        if already_checked_in and not already_checked_out:
            app.logger.info(f"Sinh vien {name} da check-in va chua checkout")
            return False
    
    session_ctx = get_active_attendance_session()
    credit_class_id = session_ctx.get('credit_class_id') if session_ctx else None
    session_id = session_ctx.get('id') if session_ctx else None

    if expected_credit_class_id is not None:
        if not session_ctx or int(credit_class_id or 0) != int(expected_credit_class_id):
            app.logger.info(
                "[Attendance] Rejecting check-in for %s: session mismatch (expected class %s, active %s)",
                normalized_student_id or name,
                expected_credit_class_id,
                credit_class_id,
            )
            return False

    success = db.mark_attendance(
        student_id=normalized_student_id or student_id,
        student_name=name,
        status='present',
        confidence_score=confidence_score,
        notes=None,
        credit_class_id=credit_class_id,
        session_id=session_id
    )
    app.logger.info(f"[DEBUG] Mark attendance success: {success}, session_id: {session_id}")
    
    # Tự động enroll sinh viên vào lớp tín chỉ nếu chưa enroll và điểm danh thành công
    if success and credit_class_id:
        try:
            # Kiểm tra xem đã enroll chưa
            student_db_id = db.get_student_id_by_student_id(normalized_student_id)
            if student_db_id:
                enrolled = db.check_student_enrolled_in_credit_class(student_db_id, credit_class_id)
                if not enrolled:
                    db.enroll_student_in_credit_class(student_db_id, credit_class_id)
                    app.logger.info(f"[Attendance] Auto-enrolled {normalized_student_id} into credit class {credit_class_id}")
        except Exception as e:
            app.logger.warning(f"[Attendance] Failed to auto-enroll {normalized_student_id}: {e}")
    
    if success:
        session_payload = serialize_session_payload(session_ctx)
        with today_recorded_lock:
            today_checked_in.add(normalized_student_id)
            today_checked_out.discard(normalized_student_id)
            existing_info = today_student_names.get(normalized_student_id)
            class_name = None
            class_type = None
            credit_ctx = credit_class_id
            if isinstance(existing_info, dict):
                class_name = existing_info.get('class_name')
                class_type = existing_info.get('class_type')
                credit_ctx = existing_info.get('credit_class_id', credit_ctx)
            if not class_name and session_payload:
                class_name = session_payload.get('class_name') or session_payload.get('class_code')
            if session_payload:
                class_type = 'credit'
                credit_ctx = session_payload.get('credit_class_id')
            today_student_names[normalized_student_id] = {
                'name': name,
                'class_name': class_name,
                'class_type': class_type or 'administrative',
                'credit_class_id': credit_ctx
            }
        # Khởi tạo presence tracking
        with presence_tracking_lock:
            presence_tracking[normalized_student_id] = {
                'last_seen': datetime.now(),
                'check_in_time': datetime.now(),
                'name': name
            }
        app.logger.info(
            f"Da danh dau diem danh: {name} (id={normalized_student_id or student_id}, confidence={confidence_score})"
        )
        
        broadcast_sse_event({
            'type': 'attendance_marked',
            'data': {
                'event': 'check_in',
                'student_id': normalized_student_id or student_id,
                'student_name': name,
                'confidence': confidence_score,
                'timestamp': datetime.now().isoformat(),
                'session': session_payload
            }
        })
    
    return success


def mark_student_checkout(
    student_id: str,
    student_name: str = '',
    reason: str = 'manual',
    confidence_score: float = None,
    expected_student_id: str = None,
    expected_credit_class_id: int = None,
) -> bool:
    """Đánh dấu checkout cho sinh viên với ràng buộc khuôn mặt/sessions tùy chọn."""
    print(f"[DEBUG] mark_student_checkout called for {student_id}, expected_credit_class_id={expected_credit_class_id}")
    normalized_student_id = (student_id or '').strip().upper()
    normalized_expected_id = (expected_student_id or '').strip()
    if normalized_expected_id and normalized_student_id and normalized_student_id != normalized_expected_id:
        app.logger.info(
            "[Attendance] Rejecting checkout: recognized %s but expected %s",
            normalized_student_id,
            normalized_expected_id,
        )
        return False
    with today_recorded_lock:
        already_checked_in = normalized_student_id in today_checked_in
        already_checked_out = normalized_student_id in today_checked_out
    print(f"[DEBUG] Checkout check: checked_in={already_checked_in}, checked_out={already_checked_out}")
    
    if not already_checked_in or already_checked_out:
        return False

    session_ctx = get_active_attendance_session()
    credit_class_id = session_ctx.get('credit_class_id') if session_ctx else None
    if expected_credit_class_id is not None:
        if not session_ctx or int(credit_class_id or 0) != int(expected_credit_class_id):
            app.logger.info(
                "[Attendance] Rejecting checkout for %s: session mismatch (expected class %s, active %s)",
                normalized_student_id or student_id,
                expected_credit_class_id,
                credit_class_id,
            )
            return False

    success = db.mark_checkout(
        normalized_student_id or student_id,
        session_id=session_ctx.get('id') if session_ctx else None,
    )
    if not success:
        return False
    
    existing_info = today_student_names.get(normalized_student_id)
    if isinstance(existing_info, dict):
        resolved_name = student_name or existing_info.get('name') or student_id
    else:
        resolved_name = student_name or existing_info or student_id
    with today_recorded_lock:
        today_checked_out.add(normalized_student_id)
        today_student_names[normalized_student_id] = {
            'name': resolved_name,
            'class_name': existing_info.get('class_name') if isinstance(existing_info, dict) else None
        }
    
    with presence_tracking_lock:
        presence_tracking.pop(normalized_student_id, None)
    
    broadcast_sse_event({
        'type': 'attendance_checkout',
        'data': {
            'event': 'check_out',
            'student_id': normalized_student_id or student_id,
            'student_name': resolved_name,
            'confidence': confidence_score,
            'reason': reason,
            'timestamp': datetime.now().isoformat(),
            'session': serialize_session_payload(get_active_attendance_session())
        }
    })
    
    app.logger.info(f"Da checkout: {resolved_name} (id={normalized_student_id or student_id}) - reason={reason}")
    return True

# Hàm nhận diện khuôn mặt (giống hệt hệ thống mẫu Cong-Nghe-Xu-Ly-Anh)
def recognize_face(embedding, db_embeddings, db_labels, threshold=0.4):
    """
    Nhận diện khuôn mặt bằng cosine similarity.
    Logic giống hệt Cong-Nghe-Xu-Ly-Anh/diemdanh_deepface_gui.py
    
    Args:
        embedding: Embedding vector của khuôn mặt cần nhận diện
        db_embeddings: Numpy array các embedding vectors từ database
        db_labels: List các tuple (student_id, name) tương ứng với db_embeddings
        threshold: Ngưỡng (0.4 = similarity > 0.6)
    
    Returns:
        (student_id, name) hoặc (None, None) nếu không nhận diện được
    """
    from numpy.linalg import norm
    
    if len(db_embeddings) == 0 or len(db_labels) == 0:
        return None, None
    
    def cosine_similarity(a, b):
        """Tính cosine similarity giữa 2 vectors (giống hệ thống mẫu)"""
        return np.dot(a, b) / (norm(a) * norm(b))
    
    # Tính similarity với tất cả embeddings trong database
    sims = [cosine_similarity(embedding, e) for e in db_embeddings]
    best_idx = np.argmax(sims)
    best_score = sims[best_idx]
    
    # Kiểm tra ngưỡng: similarity > (1 - threshold)
    # threshold=0.4 nghĩa là similarity > 0.6 (giống hệ thống mẫu)
    if best_score > (1 - threshold):
        return db_labels[best_idx]
    else:
        return None, None

def update_presence(student_id: str, name: str):
    """Cập nhật thời gian có mặt của sinh viên"""
    now = datetime.now()
    
    with presence_tracking_lock:
        if student_id in presence_tracking:
            # Cập nhật last_seen
            presence_tracking[student_id]['last_seen'] = now
            # Cập nhật database
            db.update_last_seen(student_id, name)
        else:
            # Nếu chưa có trong tracking nhưng đã điểm danh, thêm vào
            if student_id in today_checked_in:
                presence_tracking[student_id] = {
                    'last_seen': now,
                    'check_in_time': now,
                    'name': name
                }

def broadcast_sse_event(event_data):
    """Gửi sự kiện đến tất cả SSE clients"""
    import json
    with sse_clients_lock:
        dead_clients = []
        for client_queue in sse_clients:
            try:
                client_queue.put_nowait(event_data)
            except queue.Full:
                # Client queue is full, mark for removal
                dead_clients.append(client_queue)
            except Exception as e:
                app.logger.error(f"Error broadcasting SSE event: {e}")
                dead_clients.append(client_queue)
        
        # Remove dead clients
        for dead_client in dead_clients:
            try:
                sse_clients.remove(dead_client)
            except ValueError:
                pass


# --- Trình xem điểm danh bên ngoài (tích hợp an toàn) ----------------------
@app.route('/external-attendance', methods=['GET'])
def external_attendance():
    """Hiển thị chế độ xem chỉ đọc của các tệp CSV điểm danh từ dự án đính kèm.
    Điều này KHÔNG sửa đổi bất kỳ dữ liệu nào trong dự án chính; nó chỉ đọc các tệp CSV
    từ `external_projects/Cong-Nghe-Xu-Ly-Anh/attendance_logs` và hiển thị chúng
    bằng `templates/external_index.html`.
    """
    all_data = []
    headers = ["ID", "Name", "Time", "Status", "SourceFile"]
    external_dir = Path('Cong-Nghe-Xu-Ly-Anh') / 'attendance_logs'

    search_name = request.args.get('name', '').strip().lower()
    date_filter = request.args.get('date', '')

    if external_dir.exists():
        for filename in sorted(os.listdir(external_dir)):
            if filename.endswith('.csv'):
                file_path = external_dir / filename
                try:
                    with open(file_path, newline='', encoding='utf-8') as csvfile:
                        reader = csv.reader(csvfile)
                        rows = list(reader)
                        if len(rows) > 1 and rows[0][:4] == ["ID", "Name", "Time", "Status"]:
                            for row in rows[1:]:
                                row.append(filename)
                                all_data.append(row)
                except Exception as e:
                    app.logger.warning(f"Không thể đọc file external {filename}: {e}")

    # Lọc cục bộ (hành vi giống như template dự án bên ngoài mong đợi)
    if search_name:
        all_data = [row for row in all_data if search_name in str(row[1]).lower()]

    if date_filter:
        filtered = []
        for row in all_data:
            try:
                row_date = datetime.strptime(row[2], "%Y-%m-%d %H:%M:%S").date()
                if str(row_date) == date_filter:
                    filtered.append(row)
            except Exception:
                continue
        all_data = filtered

    return render_template('external_index.html', headers=headers, data=all_data,
                           search_name=search_name, date_filter=date_filter)


def check_presence_timeout():
    """Kiểm tra và tự động checkout những sinh viên không còn xuất hiện"""
    now = datetime.now()
    
    with presence_tracking_lock:
        timeout_students = []
        
        for student_id, data in presence_tracking.items():
            last_seen = data['last_seen']
            time_diff = (now - last_seen).total_seconds()
            
            # Nếu quá 5 phút không thấy, tự động checkout
            if time_diff > PRESENCE_TIMEOUT:
                timeout_students.append(student_id)
        
        # Checkout các sinh viên timeout
        for student_id in timeout_students:
            student_name = presence_tracking[student_id]['name']
            mark_student_checkout(student_id, student_name=student_name, reason='timeout')
            presence_tracking.pop(student_id, None)

# Đọc điểm danh hôm nay từ Database
def get_today_attendance(credit_class_id=None, session_id=None):
    """Lấy danh sách điểm danh hôm nay từ database với bộ lọc tùy chọn."""
    try:
        resolved_class_id = credit_class_id
        resolved_session_id = session_id

        if session_id:
            session_row = db.get_session_by_id(session_id)
            if session_row:
                resolved_class_id = resolved_class_id or session_row.get('credit_class_id')

        if resolved_class_id and not resolved_session_id:
            session_row = db.get_active_session_for_class(resolved_class_id)
            if session_row:
                resolved_session_id = session_row.get('id')

        if resolved_class_id is None and resolved_session_id is None:
            session_row = get_active_attendance_session()
            if session_row:
                resolved_session_id = session_row.get('id')
                resolved_class_id = session_row.get('credit_class_id')

        attendance_data = db.get_today_attendance(
            session_id=resolved_session_id,
            credit_class_id=resolved_class_id,
        )
        # Chuyển đổi đối tượng SQLite Row thành dict
        results = []
        now = datetime.now()

        for row in attendance_data:
            # Tính thời gian có mặt
            duration_minutes = 0
            status_text = "Đang có mặt"

            check_in = parse_datetime_safe(row['check_in_time'])
            check_out = parse_datetime_safe(row['check_out_time'])
            row_credit_class_id = row.get('credit_class_id')
            credit_class_name = row.get('credit_class_name')
            credit_class_code = row.get('credit_class_code')
            class_type = 'credit' if row_credit_class_id else 'administrative'
            base_class_name = row.get('class_name')
            class_display = credit_class_name or base_class_name
            if row_credit_class_id:
                label_parts = [credit_class_name, credit_class_code]
                class_display = ' · '.join([part for part in label_parts if part]) or class_display

            if check_in is None:
                app.logger.warning(
                    "Attendance row is missing check-in time", extra={"student_id": row['student_id']}
                )
                continue

            if check_out:
                # Đã checkout
                duration_seconds = max((check_out - check_in).total_seconds(), 0)
                status_text = "Đã rời"
            else:
                # Chưa checkout - tính từ check_in đến hiện tại
                duration_seconds = max((now - check_in).total_seconds(), 0)

                # Kiểm tra xem có đang được tracking không
                with presence_tracking_lock:
                    if row['student_id'] not in presence_tracking:
                        status_text = "Không còn phát hiện"

            duration_minutes = int(duration_seconds / 60)

            timestamp_value = check_in.isoformat()
            checkout_value = check_out.isoformat() if check_out else None

            results.append({
                'student_id': row['student_id'],
                'full_name': row['student_name'],
                'class_name': base_class_name,
                'class_display': class_display,
                'class_type': class_type,
                'credit_class_id': row_credit_class_id,
                'credit_class_code': credit_class_code,
                'credit_class_name': credit_class_name,
                'session_id': row.get('session_id'),
                'timestamp': timestamp_value,
                'checkout_time': checkout_value,
                'date': row['attendance_date'],
                'duration_minutes': duration_minutes,
                'status': status_text
            })
        return results
    except Exception as e:
        app.logger.error(f"Error getting today attendance: {e}")
        return []

# helper: tạo hình ảnh JPEG placeholder (numpy + OpenCV)
def make_placeholder_frame(message: str = "Camera không khả dụng"):
    # tạo hình ảnh 640x480 với nền tối và thông báo
    h, w = 480, 640
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # nền
    img[:] = (30, 30, 30)
    # đặt text thông báo
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    text_size, _ = cv2.getTextSize(message, font, scale, thickness)
    text_w, text_h = text_size
    x = max(10, (w - text_w) // 2)
    y = max(30, (h - text_h) // 2)
    cv2.putText(img, message, (x, y), font, scale, (200, 200, 200), thickness, cv2.LINE_AA)
    # mã hóa thành jpeg (chất lượng hơi thấp để tiết kiệm băng thông)
    ret, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ret:
        return None
    return buf.tobytes()

# Generator khung hình video
def generate_frames(
    expected_student_id: str = None,
    selected_action: str = 'checkin',
    enforce_student_match: bool = False,
    expected_credit_class_id: int = None,
):
    global camera_enabled
    
    app.logger.info("generate_frames() started")
    enforced_student_id = (expected_student_id or '').strip() if enforce_student_match else None
    requested_action = (selected_action or 'checkin').lower()
    if requested_action not in ('checkin', 'checkout'):
        requested_action = 'auto'
    
    # Nếu camera bị tắt, phát placeholder liên tục thay vì khởi tạo camera
    if not camera_enabled:
        placeholder = make_placeholder_frame("Camera đã tắt")
        if placeholder is None:
            return
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n')
        return

    pipeline = ensure_camera_pipeline()
    if pipeline is None:
        placeholder = make_placeholder_frame("Không thể khởi động camera")
        if placeholder is None:
            return
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n')
        return

    frame_count = 0
    detection_frame_counter = YOLO_FRAME_SKIP  # ép chạy YOLO ngay frame đầu tiên
    cached_face_data = []
    inference_warmed_up = False

    while True:
        # Kiểm tra xem camera có bị tắt không
        if not camera_enabled:
            app.logger.info("Camera da tat, dung stream")
            break
            
        try:
            vision_frame = pipeline.get_frame()
            frame = vision_frame.bgr
            frame_count += 1
            if frame_count % 30 == 0:
                app.logger.debug(f"[Camera] 📹 Đang đọc frame #{frame_count}...")
        except CameraError as exc:
            app.logger.warning("[Camera] ⚠️ Mất kết nối camera: %s", exc)
            release_camera_capture()
            time.sleep(0.2)
            pipeline = ensure_camera_pipeline()
            if pipeline is None:
                placeholder = make_placeholder_frame("Camera lỗi - đang thử lại")
                if placeholder is None:
                    break
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n')
                time.sleep(0.5)
            continue
        except Exception as exc:
            app.logger.error(f"[Camera] ❌ Lỗi đọc frame: {exc}", exc_info=True)
            time.sleep(0.2)
            continue

        # lấy kích thước khung hình
        frame_h, frame_w = frame.shape[:2]
        
        # Flip frame horizontally TRƯỚC (mirror effect - chế độ soi gương)
        # Làm này trước để text và bounding box không bị ngược
        frame = cv2.flip(frame, 1)

        # ========================================================================
        # HỆ THỐNG NHẬN DIỆN VÀ ĐIỂM DANH - LOGIC TỪ CONG-NGHE-XU-LY-ANH
        # ========================================================================
        face_data = []
        detection_frame_counter += 1
        should_run_detection = detection_frame_counter >= YOLO_FRAME_SKIP

        engine_ready = inference_engine is not None and inference_engine.has_strategies()
        if engine_ready and not inference_warmed_up:
            try:
                inference_engine.warmup(force=False)
                inference_warmed_up = True
            except Exception as exc:
                inference_warmed_up = False
                app.logger.warning("[Inference] Không thể warmup inference engine: %s", exc)
                engine_ready = False
        elif not engine_ready:
            inference_warmed_up = False

        detection_available = (not DEMO_MODE) and YOLO_AVAILABLE and yolo_face_model is not None

        if detection_available:
            if should_run_detection:
                detection_frame_counter = 0
                detection_frame = frame
                scale_x = scale_y = 1.0
                if YOLO_INFERENCE_WIDTH > 0 and frame_w > YOLO_INFERENCE_WIDTH:
                    detection_width = YOLO_INFERENCE_WIDTH
                    detection_height = int(frame_h * (detection_width / frame_w))
                    detection_frame = cv2.resize(
                        frame,
                        (detection_width, detection_height),
                        interpolation=cv2.INTER_LINEAR
                    )
                    scale_x = frame_w / detection_width
                    scale_y = frame_h / detection_height

                results = yolo_face_model(detection_frame, verbose=False)[0]
                boxes = results.boxes.xyxy.cpu().numpy()
                new_face_data = []

                for box in boxes:
                    xmin, ymin, xmax, ymax = map(int, box)
                    xmin = int(xmin * scale_x)
                    xmax = int(xmax * scale_x)
                    ymin = int(ymin * scale_y)
                    ymax = int(ymax * scale_y)

                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(frame_w, xmax)
                    ymax = min(frame_h, ymax)

                    face_img = frame[ymin:ymax, xmin:xmax]
                    if face_img.size == 0:
                        continue

                    recognition = recognize_face_candidate(face_img)
                    student_id = (recognition.get('student_id') or 'UNKNOWN').strip() or 'UNKNOWN'
                    name = recognition.get('student_name') or (student_id if student_id != 'UNKNOWN' else 'UNKNOWN')
                    confidence_score = float(recognition.get('confidence') or 0.0)
                    strategy = recognition.get('strategy', 'none')
                    recognition_status = recognition.get('status', 'unknown')

                    status = 'unknown'
                    now = datetime.now()

                    if student_id != 'UNKNOWN':
                        recognized_id_norm = (student_id or '').strip().upper()
                        checked_in = recognized_id_norm in today_checked_in
                        checked_out = recognized_id_norm in today_checked_out
                        with last_recognized_lock:
                            last_time = last_recognized.get(student_id)
                            cooldown_passed = not last_time or (now - last_time).total_seconds() > RECOGNITION_COOLDOWN

                        guard_student_id = enforced_student_id if enforce_student_match else None
                        guard_credit_class = expected_credit_class_id
                        recognized_id_norm = (student_id or '').strip().upper()
                        guard_id_norm = (
                            (guard_student_id or '').strip().upper()
                            if guard_student_id
                            else None
                        )
                        mismatch = guard_id_norm and recognized_id_norm != guard_id_norm

                        if mismatch:
                            status = 'mismatch'
                            app.logger.warning(
                                "[Guard] Student mismatch: recognized=%s (raw=%s) expected=%s action=%s class=%s",
                                recognized_id_norm or 'UNKNOWN',
                                student_id or 'UNKNOWN',
                                guard_id_norm,
                                requested_action,
                                guard_credit_class,
                            )
                        elif requested_action == 'checkout':
                            print(f"[DEBUG] Checkout attempt: checked_in={checked_in}, checked_out={checked_out}, cooldown_passed={cooldown_passed}")
                            if checked_in and not checked_out:
                                if mark_student_checkout(
                                    student_id,
                                    student_name=name,
                                    reason='auto',
                                    confidence_score=confidence_score,
                                    expected_student_id=guard_student_id,
                                    expected_credit_class_id=guard_credit_class,
                                ):
                                    status = 'checked_out'
                                    with last_recognized_lock:
                                        last_recognized[student_id] = now
                                else:
                                    status = 'already_marked'
                            elif not checked_in:
                                status = 'not_checked_in'
                            elif checked_out:
                                status = 'checked_out'
                            else:
                                status = 'cooldown'
                        else:
                            if not checked_in and cooldown_passed:
                                print(f"[DEBUG] Attempting mark_attendance for {student_id} ({name})")
                                try:
                                    success = mark_attendance(
                                        name,
                                        student_id=student_id,
                                        confidence_score=confidence_score,
                                        expected_student_id=guard_student_id,
                                        expected_credit_class_id=guard_credit_class,
                                    )
                                    print(f"[DEBUG] Mark attendance result: {success}")
                                    if success:
                                        status = 'checked_in'
                                        with last_recognized_lock:
                                            last_recognized[student_id] = now
                                    else:
                                        status = 'already_marked'
                                except Exception as e:
                                    app.logger.error(f"[System] Lỗi điểm danh: {e}")
                            elif (
                                requested_action == 'auto'
                                and checked_in
                                and not checked_out
                                and cooldown_passed
                            ):
                                if mark_student_checkout(
                                    student_id,
                                    student_name=name,
                                    reason='auto',
                                    confidence_score=confidence_score,
                                ):
                                    status = 'checked_out'
                                    with last_recognized_lock:
                                        last_recognized[student_id] = now
                                else:
                                    status = 'already_marked'
                            elif checked_in and not checked_out:
                                status = 'already_marked'
                            elif checked_out:
                                status = 'checked_out'
                            else:
                                status = 'cooldown' if not cooldown_passed else 'already_marked'
                    else:
                        status = recognition_status or 'unknown'

                    new_face_data.append({
                        'bbox': (xmin, ymin, xmax, ymax),
                        'name': name,
                        'student_id': student_id,
                        'confidence': confidence_score,
                        'status': status,
                        'strategy': strategy,
                    })

                cached_face_data = new_face_data
                face_data = new_face_data
            else:
                face_data = cached_face_data or []

        # Demo mode hoặc không có YOLOv8
        elif DEMO_MODE or not YOLO_AVAILABLE or yolo_face_model is None:
            # Tạo một số bounding box mô phỏng ở giữa màn hình
            face_data = []
            
            # Nếu có danh sách khuôn mặt đã load, hiển thị tên ngẫu nhiên
            if known_face_names:
                # Chọn ngẫu nhiên 1 người từ danh sách (giả lập nhận diện)
                idx = frame_count % len(known_face_names)  # Thay đổi theo frame
                demo_name = known_face_names[idx]
                demo_id = known_face_ids[idx] if idx < len(known_face_ids) else 'DEMO'
                demo_confidence = 0.85 + (random.random() * 0.15)  # 85-100%
                status = 'confirmed'
            else:
                # Chưa có khuôn mặt nào được đăng ký
                demo_name = 'Demo Mode - Đang chờ khuôn mặt'
                demo_id = 'DEMO'
                demo_confidence = 0.0
                status = 'waiting'
            
            # Chỉ hiển thị 1 khung giả lập ở giữa màn hình
            face_size_w = frame_w // 3  # Chiều rộng khuôn mặt
            face_size_h = int(face_size_w * 1.3)  # Chiều cao khuôn mặt (cao hơn rộng)
            
            # Vị trí ở giữa màn hình
            center_x = frame_w // 2
            center_y = frame_h // 2
            
            left = center_x - face_size_w // 2
            top = center_y - face_size_h // 2
            right = center_x + face_size_w // 2
            bottom = center_y + face_size_h // 2
            
            # Đảm bảo trong khung hình
            left = max(10, left)
            top = max(10, top)
            right = min(frame_w - 10, right)
            bottom = min(frame_h - 10, bottom)
            
            # Tạo thông tin khuôn mặt mô phỏng
            face_info = {
                'bbox': (left, top, right, bottom),
                'name': demo_name,
                'confidence': demo_confidence,
                'student_id': demo_id,
                'status': status
            }
            face_data.append(face_info)
            
            # Mô phỏng điểm danh (mỗi 30 frames ~ 1 giây)
            if status == 'confirmed' and frame_count % 30 == 0:
                try:
                    mark_attendance(demo_name, student_id=demo_id, confidence_score=demo_confidence)
                    # Cập nhật presence
                    update_presence(demo_id, demo_name)
                except Exception as e:
                    app.logger.error(f"Loi xac nhan diem danh cho {demo_name}: {e}")
            # Cập nhật presence mỗi 60 frames (2 giây)
            elif status == 'confirmed' and frame_count % 60 == 0:
                try:
                    update_presence(demo_id, demo_name)
                except Exception as e:
                    app.logger.error(f"Loi cap nhat presence cho {demo_name}: {e}")
        # Không có gì để xử lý - chỉ hiển thị frame
        else:
            face_data = []
        
        # Vẽ bounding boxes và labels (chỉ cho demo mode)
        for face_info in face_data:
            left, top, right, bottom = face_info['bbox']
            name = face_info.get('name', 'Unknown')
            confidence = face_info.get('confidence', 0.0)
            status = face_info.get('status', 'detected')
            progress = face_info.get('progress', 0.0)
            
            # Chọn màu dựa trên trạng thái
            if status == 'waiting':
                color = (255, 165, 0)  # Màu cam cho demo mode (đang chờ)
                thickness = 2
            elif status == 'already_marked':
                color = (128, 128, 128)  # Màu xám - đã điểm danh
                thickness = 2
            elif status == 'confirming':
                color = (0, 165, 255)  # Màu cam - đang xác nhận
                thickness = 3
                draw_progress_bar(frame, progress, left, top)
            elif status == 'confirmed' or status == 'checked_in':
                color = (0, 255, 0)  # Màu xanh lá - vừa điểm danh thành công
                thickness = 3
            elif status == 'checked_out':
                color = (0, 128, 255)  # Màu xanh dương nhạt cho checkout
                thickness = 3
            elif status == 'mismatch':
                color = (0, 0, 255)  # Màu đỏ cho sai tài khoản
                thickness = 2
            elif name == "Unknown" or status == 'unknown':
                color = (0, 0, 255)  # Màu đỏ cho Unknown (không nhận diện được)
                thickness = 2
            elif status == 'low_confidence':
                color = (0, 165, 255)  # Màu cam cho confidence thấp
                thickness = 2
            elif status == 'cooldown':
                color = (128, 128, 128)  # Màu xám - đang trong thời gian chờ
                thickness = 2
            else:
                color = (0, 165, 255)  # Màu cam cho nhận diện chưa chắc chắn
                thickness = 2
            
            # Vẽ bounding box với nét đậm hơn
            cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)
            
            # Vẽ nhãn với tên và độ tin cậy
            if status == 'already_marked':
                label = f"{name} - Da diem danh"
            elif status == 'confirming':
                label = f"{name} - Dang xac nhan..."
            elif status == 'confirmed' or status == 'checked_in':
                label = f"{name} - THANH CONG!"
            elif status == 'checked_out':
                label = f"{name} - Da ra ve"
            elif status == 'mismatch':
                label = f"{name} - Sai tai khoan"
            elif name == "Unknown":
                label = "Unknown - Chua dang ky"
            elif status == 'low_confidence':
                label = f"{name} (Confidence thap: {confidence*100:.1f}%)"
            elif status == 'cooldown':
                label = f"{name} - Vua diem danh (cho {RECOGNITION_COOLDOWN}s)"
            elif status == 'not_checked_in':
                label = f"{name} - Can check-in truoc"
            elif confidence > 0:
                label = f"{name} ({confidence*100:.1f}%)"
            else:
                label = name
            
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
            label_x = left
            label_y = top - 10 if top > 30 else bottom + 30
            
            # Vẽ nền nhãn (hiệu ứng bán trong suốt với padding)
            padding = 5
            cv2.rectangle(frame, 
                         (label_x - padding, label_y - label_size[1] - padding), 
                         (label_x + label_size[0] + padding, label_y + padding), 
                         color, -1)
            
            # Vẽ chữ nhãn màu đen để tương phản tốt hơn
            cv2.putText(frame, label, (label_x, label_y), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        
        # Kiểm tra frame có hợp lệ không
        if frame is None or frame.size == 0:
            continue

        # Tăng frame counter
        frame_count += 1

        # Kiểm tra timeout presence mỗi 100 frames (~3 giây)
        if frame_count % 100 == 0:
            try:
                check_presence_timeout()
            except Exception as e:
                app.logger.error(f"Loi kiem tra presence timeout: {e}")

        # Mã hóa frame với chất lượng giảm để giảm tải CPU và băng thông
        ret2, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ret2:
            continue
        frame_bytes = buf.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    release_camera_capture()

# Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    """Trang đăng nhập hệ thống."""
    next_url = sanitize_next_url(request.args.get('next') or request.form.get('next'))

    if g.get('user'):
        return redirect(next_url or url_for('index'))

    error = None
    if request.method == 'POST':
        username = (request.form.get('username') or '').strip()
        password = request.form.get('password') or ''

        if not username or not password:
            error = 'Vui lòng nhập tên đăng nhập và mật khẩu'
        else:
            user = db.get_user_by_username(username)
            if not user or not verify_user_password(user, password):
                error = 'Tên đăng nhập hoặc mật khẩu không đúng'
            else:
                login_user(user)
                db.update_last_login(user['id'])
                flash('Đăng nhập thành công', 'success')
                app.logger.info("User %s đã đăng nhập", user.get('username'))
                
                # Redirect based on user role
                if next_url:
                    return redirect(next_url)
                
                user_role = (user.get('role') or '').lower()
                if user_role == 'student':
                    return redirect(url_for('student_portal_page'))
                else:
                    return redirect(url_for('index'))

    return render_template('login.html', next_url=next_url, error=error, active_page='login')


@app.route('/logout')
def logout():
    """Đăng xuất người dùng hiện tại."""
    user = getattr(g, 'user', None)
    if user:
        app.logger.info("User %s đã đăng xuất", user.get('username'))
    logout_current_user()
    flash('Bạn đã đăng xuất khỏi hệ thống', 'info')
    return redirect(url_for('login'))


@app.route('/')
def index():
    """Trang chính - điểm danh"""
    attendance_data = get_today_attendance()
    checked_in = [row for row in attendance_data if not row.get('checkout_time')]
    checked_out = [row for row in attendance_data if row.get('checkout_time')]
    return render_template('index.html', attendance=attendance_data,
                           checked_in=checked_in, checked_out=checked_out)

@app.route('/video_feed')
@role_required('student')
def video_feed():
    """Video feed cho camera"""
    student = resolve_student_context()
    if not student:
        abort(403, description='Không tìm thấy hồ sơ sinh viên')

    selected_action = (request.args.get('action') or 'checkin').lower()
    if selected_action not in ('checkin', 'checkout'):
        abort(400, description='Hành động không hợp lệ')

    credit_class_id = request.args.get('credit_class_id', type=int)
    if not credit_class_id:
        abort(400, description='Thiếu lớp tín chỉ')

    session_row = get_active_attendance_session()
    active_class_id = session_row.get('credit_class_id') if session_row else None
    if not session_row or int(active_class_id or 0) != int(credit_class_id):
        abort(409, description='Lớp tín chỉ này chưa mở phiên điểm danh')

    def frame_stream():
        yield from generate_frames(
            expected_student_id=student.get('student_id'),
            selected_action=selected_action,
            enforce_student_match=True,
            expected_credit_class_id=credit_class_id,
        )

    return Response(
        stream_with_context(frame_stream()),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/camera/toggle', methods=['POST'])
@role_required('student')
def toggle_camera():
    """API bật/tắt camera pipeline"""
    global camera_enabled
    try:
        desired_state = not camera_enabled
        camera_enabled = desired_state

        if desired_state:
            pipeline = ensure_camera_pipeline()
            if pipeline is None:
                camera_enabled = False
                return jsonify({'success': False, 'enabled': camera_enabled, 'error': 'Không thể khởi động camera'}), 500
        else:
            release_camera_capture()

        return jsonify({'success': True, 'enabled': camera_enabled})

    except Exception as exc:
        app.logger.error(f"Error toggling camera: {exc}")
        camera_enabled = False
        release_camera_capture()
        return jsonify({'success': False, 'enabled': camera_enabled, 'error': str(exc)}), 500


@app.route('/api/camera/status', methods=['GET'])
@role_required('student')
def camera_status():
    """API kiểm tra trạng thái camera"""
    state = vision_state or get_or_create_vision_state()
    status = state.status() if state else {'opened': False}
    return jsonify({
        'enabled': camera_enabled,
        'opened': bool(status.get('opened'))
    })


@app.route('/api/camera/capture', methods=['POST'])
@role_required('student')
def capture_image():
    """API chụp ảnh từ camera"""
    try:
        if not camera_enabled:
            return jsonify({'error': 'Camera đang tắt'}), 400

        pipeline = ensure_camera_pipeline()
        if pipeline is None:
            return jsonify({'error': 'Không thể khởi động camera'}), 500

        try:
            frame = pipeline.get_frame().bgr
        except CameraError as exc:
            app.logger.error(f"Error capturing image: {exc}")
            return jsonify({'error': 'Không thể đọc frame từ camera'}), 400

        frame = cv2.flip(frame, 1)

        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ret:
            return jsonify({'error': 'Không thể mã hóa frame'}), 400

        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'success': True,
            'image': f'data:image/jpeg;base64,{frame_base64}'
        })

    except Exception as e:
        app.logger.error(f"Error capturing image: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/status')
def status():
    """API trạng thái hệ thống"""
    return jsonify({
        'demo_mode': DEMO_MODE,
        'face_recognition_available': FACE_RECOGNITION_AVAILABLE,
        'camera_enabled': camera_enabled,
        'known_faces_count': len(known_face_names) if 'known_face_names' in globals() else 0
    })

# API đăng ký nhanh khuôn mặt
@app.route('/api/quick-register', methods=['POST'])
def api_quick_register():
    """API đăng ký nhanh khuôn mặt"""
    try:
        data = request.form
        student_id = data.get('student_id', '').strip()
        full_name = data.get('full_name', '').strip()

        app.logger.info(f"Quick register request - ID: {student_id}, Name: {full_name}")
        app.logger.info(f"Form keys: {list(data.keys())}")
        app.logger.info(f"Files keys: {list(request.files.keys())}")

        if not all([student_id, full_name]):
            return jsonify({'error': 'Mã sinh viên và họ tên là bắt buộc'}), 400

        # Thu thập tất cả nguồn ảnh (webcam base64 + file upload)
        sample_sources = []
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

        base64_fields = sorted(key for key in data.keys() if key.startswith('image_data'))
        for key in base64_fields:
            image_payload = data.get(key)
            if image_payload:
                sample_sources.append(('base64', image_payload))

        file_candidates = []
        if request.files:
            file_candidates.extend(request.files.getlist('face_images'))
            file_candidates.extend(request.files.getlist('face_image'))

        seen_file_ids = set()
        for file_storage in file_candidates:
            if not file_storage or not file_storage.filename:
                continue
            marker = id(file_storage)
            if marker in seen_file_ids:
                continue
            seen_file_ids.add(marker)
            sample_sources.append(('upload', file_storage))

        if len(sample_sources) < MIN_FACE_SAMPLES_PER_STUDENT:
            return jsonify({'error': f'Cần tối thiểu {MIN_FACE_SAMPLES_PER_STUDENT} ảnh khuôn mặt (có thể chụp nhiều lần hoặc chọn nhiều file).'}), 400
        if len(sample_sources) > MAX_FACE_SAMPLES_PER_REQUEST:
            return jsonify({'error': f'Tối đa {MAX_FACE_SAMPLES_PER_REQUEST} ảnh khuôn mặt cho mỗi lần đăng ký.'}), 400

        saved_paths = []
        try:
            for idx, (source_type, payload) in enumerate(sample_sources):
                suffix = f"{idx:02d}"
                if source_type == 'base64':
                    path = save_base64_face_image(payload, student_id, full_name, suffix=suffix, timestamp=timestamp)
                else:
                    path = save_uploaded_face_image(payload, student_id, full_name, suffix=suffix, timestamp=timestamp)
                saved_paths.append(path)
        except ValueError as err:
            for path in saved_paths:
                safe_delete_file(path)
            return jsonify({'error': str(err)}), 400

        primary_face_path = saved_paths[0]
        face_encoding_blob = extract_face_encoding(primary_face_path)

        email = data.get('email', f'{student_id}@student.edu.vn')
        phone = data.get('phone', '')
        class_name = data.get('class_name', 'Chưa phân lớp')

        created, credentials = db.add_student(
            student_id,
            full_name,
            email,
            phone,
            class_name,
            primary_face_path,
            face_encoding=face_encoding_blob,
        )
        if not created:
            for path in saved_paths:
                safe_delete_file(path)
            return jsonify({'error': 'Mã sinh viên đã tồn tại'}), 400

        for idx, sample_path in enumerate(saved_paths):
            db.add_face_sample(student_id, sample_path, is_primary=(idx == 0))

        load_known_faces()

        payload = {
            'success': True,
            'message': f'Đăng ký thành công cho {full_name}! Đã lưu {len(saved_paths)} ảnh.',
            'samples': len(saved_paths),
        }
        if credentials:
            payload['credentials'] = credentials
        return jsonify(payload), 200

    except Exception as e:
        app.logger.error(f"Quick registration error: {e}")
        return jsonify({'error': f'Lỗi: {str(e)}'}), 500

# Các route trang
@app.route('/students')
@role_required('admin')
def students_page():
    """Trang quản lý sinh viên"""
    return render_template('students.html')

@app.route('/test-students')
def test_students_page():
    """Trang test API students"""
    # Template test chuyên dụng đã bị xóa trong quá trình dọn dẹp; sử dụng lại
    # trang quản lý sinh viên để route vẫn hoạt động.
    return render_template('students.html')

@app.route('/reports')
def reports_page():
    """Trang báo cáo"""
    return render_template('reports.html')

@app.route('/classes')
@role_required('admin')
def classes_page():
    """Trang quản lý lớp học"""
    try:
        admin_classes = db.get_all_classes()
        credit_classes = db.list_credit_classes_overview()
        teacher_options = db.get_all_teachers()
        classes = admin_classes + credit_classes

        total_admin = len(admin_classes)
        total_credit = len(credit_classes)
        total_classes = total_admin + total_credit

        # Sử dụng số lượng sinh viên của lớp hành chính; các lớp tín chỉ bao gồm số lượng riêng
        total_students = sum(cls.get('student_count', 0) for cls in admin_classes)
        active_classes = sum(1 for cls in admin_classes if cls.get('is_active', 1)) + sum(1 for cc in credit_classes if cc.get('is_active', 1))

        attendance_rates = []
        for cls in classes:
            try:
                stats = db.get_class_attendance_stats(cls['id'])
                attendance_rates.append(stats.get('attendance_rate', 0))
            except Exception as stats_error:
                app.logger.debug(
                    "Could not calculate attendance rate for class %s: %s",
                    cls.get('id'),
                    stats_error,
                )

        avg_attendance = round(sum(attendance_rates) / len(attendance_rates), 2) if attendance_rates else 0

        return render_template(
            'classes.html',
            admin_classes=admin_classes,
            credit_classes=credit_classes,
            teacher_options=teacher_options,
            total_classes=total_classes,
            total_admin=total_admin,
            total_credit=total_credit,
            total_students=total_students,
            active_classes=active_classes or total_classes,
            avg_attendance=avg_attendance,
        )
    except Exception as error:
        app.logger.error(f"Error loading classes page: {error}")
        return render_template(
            'classes.html',
            admin_classes=[],
            credit_classes=[],
            teacher_options=[],
            total_classes=0,
            total_admin=0,
            total_credit=0,
            total_students=0,
            active_classes=0,
            avg_attendance=0,
        )


@app.route('/teacher/credit-classes')
@role_required('teacher', 'admin')
def teacher_credit_classes_page():
    """Trang dành cho giảng viên theo dõi lớp tín chỉ của mình."""
    teacher_id = request.args.get('teacher_id', type=int) if get_current_role() == 'admin' else None
    teacher = resolve_teacher_context(teacher_id)
    if get_current_role() == 'teacher' and not teacher:
        flash('Không tìm thấy hồ sơ giảng viên cho tài khoản hiện tại.', 'warning')
    return render_template(
        'teacher_credit_classes.html',
        teacher=teacher,
        teacher_param=teacher_id,
        active_page='teacher-classes',
    )


@app.route('/student/portal')
@role_required('student', 'admin')
def student_portal_page():
    """Trang tổng quan cho sinh viên xem lịch học và lịch sử điểm danh."""
    student_id = request.args.get('student_id') if get_current_role() == 'admin' else None
    student = resolve_student_context(student_id)
    if (student_id or get_current_role() == 'student') and not student:
        flash('Không tìm thấy thông tin sinh viên phù hợp.', 'warning')
    return render_template(
        'student_portal.html',
        student=student,
        student_param=student_id,
        active_page='student-portal',
    )

@app.route('/api/students', methods=['GET'])
@role_required('admin')
def api_get_students():
    """API lấy danh sách sinh viên"""
    try:
        students = db.get_all_students(active_only=True)
        class_map = {
            cls['id']: cls['class_name']
            for cls in db.get_all_classes()
        }
        students_list = [
            serialize_student_record(student, class_map)
            for student in students
        ]
        return jsonify({'success': True, 'data': students_list})
    except Exception as e:
        app.logger.error(f"Error getting students: {e}")
        return jsonify({'success': False, 'data': [], 'error': str(e)}), 500


@app.route('/api/students', methods=['POST'])
@role_required('admin')
def api_create_student():
    """API tạo sinh viên mới"""
    data = get_request_data()
    student_id = (data.get('student_id') or '').strip()
    full_name = (data.get('full_name') or '').strip()

    if not student_id or not full_name:
        return jsonify({'success': False, 'message': 'Mã sinh viên và họ tên là bắt buộc'}), 400

    email = (data.get('email') or '').strip() or None
    phone = (data.get('phone') or '').strip() or None
    class_name = (data.get('class_name') or '').strip() or None
    face_files = []
    if request.files:
        face_files.extend(request.files.getlist('face_images'))
        fallback_file = request.files.get('face_image')
        if fallback_file:
            face_files.append(fallback_file)

    face_files = [f for f in face_files if f and f.filename]
    if len(face_files) < MIN_FACE_SAMPLES_PER_STUDENT:
        return jsonify({
            'success': False,
            'message': f'Cần tải lên tối thiểu {MIN_FACE_SAMPLES_PER_STUDENT} ảnh khuôn mặt rõ nét.',
        }), 400
    if len(face_files) > MAX_FACE_SAMPLES_PER_REQUEST:
        return jsonify({
            'success': False,
            'message': f'Tối đa {MAX_FACE_SAMPLES_PER_REQUEST} ảnh khuôn mặt mỗi lần thêm sinh viên.',
        }), 400

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    face_image_paths = []

    try:
        for idx, file in enumerate(face_files):
            path = save_uploaded_face_image(
                file,
                student_id,
                full_name,
                suffix=f"{idx:02d}",
                timestamp=timestamp,
            )
            face_image_paths.append(path)

        primary_face_path = face_image_paths[0] if face_image_paths else None
        face_encoding_blob = extract_face_encoding(primary_face_path) if primary_face_path else None

        created, credentials = db.add_student(
            student_id=student_id,
            full_name=full_name,
            email=email,
            phone=phone,
            class_name=class_name,
            face_image_path=primary_face_path,
            face_encoding=face_encoding_blob,
        )

        if not created:
            for path in face_image_paths:
                safe_delete_file(path)
            return jsonify({'success': False, 'message': 'Mã sinh viên đã tồn tại'}), 400

        for idx, sample_path in enumerate(face_image_paths):
            db.add_face_sample(student_id, sample_path, is_primary=(idx == 0))

        load_known_faces()

        student = db.get_student(student_id)
        response_payload = {
            'success': True,
            'data': serialize_student_record(student),
            'samples': len(face_image_paths),
        }
        if credentials:
            response_payload['credentials'] = credentials
        return jsonify(response_payload), 201
    except ValueError as err:
        for path in face_image_paths:
            safe_delete_file(path)
        return jsonify({'success': False, 'message': str(err)}), 400
    except Exception as err:
        app.logger.error(f"Error creating student {student_id}: {err}", exc_info=True)
        for path in face_image_paths:
            safe_delete_file(path)
        return jsonify({'success': False, 'message': 'Không thể tạo sinh viên'}), 500


@app.route('/api/students/<student_id>', methods=['GET', 'PUT', 'DELETE'])
@role_required('admin')
def api_student_detail(student_id):
    """API thao tác với sinh viên cụ thể"""
    student_id = (student_id or '').strip()
    if not student_id:
        return jsonify({'success': False, 'message': 'Mã sinh viên không hợp lệ'}), 400

    student = db.get_student(student_id)
    if not student:
        return jsonify({'success': False, 'message': 'Không tìm thấy sinh viên'}), 404

    if request.method == 'GET':
        return jsonify({'success': True, 'data': serialize_student_record(student)})

    if request.method == 'PUT':
        student_data = dict(student)
        data = get_request_data()
        updates = {}

        for field in ('full_name', 'email', 'phone'):
            if field in data:
                updates[field] = (data.get(field) or '').strip() or None

        if 'class_name' in data:
            updates['class_name'] = (data.get('class_name') or '').strip()

        if 'is_active' in data:
            bool_value = parse_bool(data.get('is_active'))
            if bool_value is not None:
                updates['is_active'] = bool_value

        face_files = []
        if request.files:
            face_files.extend(request.files.getlist('face_images'))
            fallback_file = request.files.get('face_image')
            if fallback_file:
                face_files.append(fallback_file)

        face_files = [f for f in face_files if f and f.filename]
        saved_face_paths = []
        if face_files:
            if len(face_files) < MIN_FACE_SAMPLES_PER_STUDENT:
                return jsonify({
                    'success': False,
                    'message': f'Cần tải lên tối thiểu {MIN_FACE_SAMPLES_PER_STUDENT} ảnh khuôn mặt rõ nét.',
                }), 400
            if len(face_files) > MAX_FACE_SAMPLES_PER_REQUEST:
                return jsonify({
                    'success': False,
                    'message': f'Tối đa {MAX_FACE_SAMPLES_PER_REQUEST} ảnh khuôn mặt mỗi lần cập nhật.',
                }), 400

            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            try:
                for idx, file in enumerate(face_files):
                    path = save_uploaded_face_image(
                        file,
                        student_id,
                        updates.get('full_name') or student_data.get('full_name'),
                        suffix=f"{idx:02d}",
                        timestamp=timestamp,
                    )
                    saved_face_paths.append(path)

                primary_face_path = saved_face_paths[0]
                updates['face_image_path'] = primary_face_path
                new_encoding = extract_face_encoding(primary_face_path)
                if new_encoding:
                    updates['face_encoding'] = new_encoding
            except ValueError as err:
                for path in saved_face_paths:
                    safe_delete_file(path)
                return jsonify({'success': False, 'message': str(err)}), 400

        if not updates:
            for path in saved_face_paths:
                safe_delete_file(path)
            return jsonify({'success': False, 'message': 'Không có dữ liệu để cập nhật'}), 400

        try:
            updated = db.update_student(student_id, **updates)
        except ValueError as err:
            for path in saved_face_paths:
                safe_delete_file(path)
            return jsonify({'success': False, 'message': str(err)}), 400

        if not updated:
            for path in saved_face_paths:
                safe_delete_file(path)
            return jsonify({'success': False, 'message': 'Không thể cập nhật sinh viên'}), 400

        if saved_face_paths:
            previous_face_path = student_data.get('face_image_path')
            if previous_face_path and previous_face_path not in saved_face_paths:
                safe_delete_file(previous_face_path)
            for idx, sample_path in enumerate(saved_face_paths):
                db.add_face_sample(student_id, sample_path, is_primary=(idx == 0))
            load_known_faces()

        student = db.get_student(student_id)
        return jsonify({'success': True, 'data': serialize_student_record(student)})

    # Phương thức DELETE
    permanent = parse_bool(request.args.get('permanent'), default=True)
    deleted = db.delete_student(student_id, permanent=permanent)
    if not deleted:
        return jsonify({'success': False, 'message': 'Không thể xóa sinh viên'}), 400
    if permanent:
        safe_delete_file(student.get('face_image_path'))
    return jsonify({'success': True})


@app.route('/api/classes', methods=['GET'])
@role_required('admin')
def api_get_classes():
    """API lấy danh sách lớp học"""
    try:
        classes = db.get_all_classes()
        for cls in classes:
            cls['class_type'] = 'administrative'
        return jsonify({'success': True, 'data': classes})
    except Exception as e:
        app.logger.error(f"Error getting classes: {e}")
        return jsonify({'success': False, 'data': [], 'message': str(e)}), 500


@app.route('/api/credit-classes', methods=['GET'])
def api_get_credit_classes():
    """API lấy danh sách lớp tín chỉ đang hoạt động."""
    teacher_only = parse_bool(request.args.get('mine'))
    teacher_id = None
    if teacher_only and getattr(g, 'user', None):
        teacher_row = db.get_teacher_by_user(g.user['id']) if get_current_role() == 'teacher' else None
        if teacher_row:
            teacher_id = teacher_row.get('id')
    try:
        credit_classes = db.list_credit_classes_overview(teacher_id=teacher_id)
        for cls in credit_classes:
            cls['class_type'] = 'credit'
            cls['display_name'] = ' · '.join(
                part for part in [cls.get('subject_name'), cls.get('credit_code')] if part
            ) or cls.get('subject_name') or cls.get('credit_code')
            cls['student_count'] = cls.get('student_count', 0) or 0
        return jsonify({'success': True, 'data': credit_classes})
    except Exception as err:
        app.logger.error(f"Error getting credit classes: {err}", exc_info=True)
        return jsonify({'success': False, 'data': [], 'message': 'Không thể tải lớp tín chỉ'}), 500


@app.route('/api/credit-classes', methods=['POST'])
@role_required('admin')
def api_create_credit_class():
    data = get_request_data()
    credit_code = (data.get('credit_code') or '').strip()
    subject_name = (data.get('subject_name') or '').strip()
    teacher_id = data.get('teacher_id')

    if not credit_code or not subject_name:
        return jsonify({'success': False, 'message': 'Mã lớp tín chỉ và tên môn học là bắt buộc'}), 400

    try:
        teacher_id = int(teacher_id)
    except (TypeError, ValueError):
        return jsonify({'success': False, 'message': 'Vui lòng chọn giảng viên phụ trách'}), 400

    teacher = db.get_teacher(teacher_id)
    if not teacher:
        return jsonify({'success': False, 'message': 'Giảng viên không tồn tại hoặc đã bị vô hiệu hóa'}), 400

    enrollment_limit = data.get('enrollment_limit')
    if enrollment_limit in (None, '', []):
        enrollment_limit = None
    else:
        try:
            enrollment_limit = int(enrollment_limit)
        except (TypeError, ValueError):
            return jsonify({'success': False, 'message': 'Sĩ số tối đa không hợp lệ'}), 400

    try:
        credit_class_id = db.create_credit_class(
            credit_code=credit_code,
            subject_name=subject_name,
            teacher_id=teacher_id,
            semester=data.get('semester'),
            academic_year=data.get('academic_year'),
            room=data.get('room'),
            schedule_info=data.get('schedule_info'),
            enrollment_limit=enrollment_limit,
            notes=data.get('notes'),
            status=data.get('status') or 'draft'
        )
        record = serialize_credit_class_record(db.get_credit_class(credit_class_id))
        return jsonify({'success': True, 'data': record}), 201
    except ValueError as exc:
        return jsonify({'success': False, 'message': str(exc)}), 400
    except Exception as exc:
        app.logger.error(f"Error creating credit class: {exc}", exc_info=True)
        return jsonify({'success': False, 'message': 'Không thể tạo lớp tín chỉ'}), 500


@app.route('/api/credit-classes/<int:credit_class_id>', methods=['GET', 'PUT', 'DELETE'])
@role_required('admin')
def api_credit_class_detail(credit_class_id):
    record = db.get_credit_class(credit_class_id)
    if not record:
        return jsonify({'success': False, 'message': 'Không tìm thấy lớp tín chỉ'}), 404

    if request.method == 'GET':
        return jsonify({'success': True, 'data': serialize_credit_class_record(record)})

    if request.method == 'PUT':
        data = get_request_data()
        updates = {}

        for field in ('credit_code', 'subject_name', 'semester', 'academic_year', 'room', 'schedule_info', 'status', 'notes'):
            if field in data:
                value = data.get(field)
                updates[field] = value.strip() if isinstance(value, str) else value

        if 'teacher_id' in data:
            teacher_id = data.get('teacher_id')
            try:
                teacher_id = int(teacher_id)
            except (TypeError, ValueError):
                return jsonify({'success': False, 'message': 'Giảng viên không hợp lệ'}), 400
            teacher = db.get_teacher(teacher_id)
            if not teacher:
                return jsonify({'success': False, 'message': 'Không tìm thấy giảng viên'}), 400
            updates['teacher_id'] = teacher_id

        if 'enrollment_limit' in data:
            limit_value = data.get('enrollment_limit')
            if limit_value in (None, '', []):
                updates['enrollment_limit'] = None
            else:
                try:
                    updates['enrollment_limit'] = int(limit_value)
                except (TypeError, ValueError):
                    return jsonify({'success': False, 'message': 'Sĩ số tối đa không hợp lệ'}), 400

        if not updates:
            return jsonify({'success': False, 'message': 'Không có dữ liệu để cập nhật'}), 400

        try:
            updated = db.update_credit_class(credit_class_id, **updates)
            if not updated:
                return jsonify({'success': False, 'message': 'Không thể cập nhật lớp tín chỉ'}), 400
            refreshed = serialize_credit_class_record(db.get_credit_class(credit_class_id))
            return jsonify({'success': True, 'data': refreshed})
        except Exception as exc:
            app.logger.error(f"Error updating credit class {credit_class_id}: {exc}", exc_info=True)
            return jsonify({'success': False, 'message': 'Lỗi khi cập nhật lớp tín chỉ'}), 500

    # DELETE
    try:
        deleted = db.delete_credit_class(credit_class_id)
        if not deleted:
            return jsonify({'success': False, 'message': 'Không thể xóa lớp tín chỉ'}), 400
        return jsonify({'success': True})
    except Exception as exc:
        app.logger.error(f"Error deleting credit class {credit_class_id}: {exc}", exc_info=True)
        return jsonify({'success': False, 'message': 'Lỗi khi xóa lớp tín chỉ'}), 500


@app.route('/api/teacher/credit-classes', methods=['GET'])
@role_required('teacher', 'admin')
def api_teacher_credit_classes():
    teacher_param = request.args.get('teacher_id', type=int) if get_current_role() == 'admin' else None
    teacher = resolve_teacher_context(teacher_param)
    if get_current_role() == 'teacher' and not teacher:
        return jsonify({'success': False, 'message': 'Không tìm thấy thông tin giảng viên'}), 404

    teacher_filter_id = teacher.get('id') if teacher else None
    try:
        credit_classes = db.list_credit_classes_overview(teacher_id=teacher_filter_id)
        results = []
        for cls in credit_classes:
            session_row = db.get_active_session_for_class(cls['id'])
            payload = dict(cls)
            payload['class_type'] = 'credit'
            payload['display_name'] = ' · '.join(
                part for part in [cls.get('subject_name'), cls.get('credit_code')] if part
            ) or cls.get('subject_name') or cls.get('credit_code')
            payload['student_count'] = cls.get('student_count', 0) or 0
            payload['active_session'] = serialize_session_payload(session_row)
            results.append(payload)

        return jsonify({
            'success': True,
            'data': results,
            'teacher': teacher,
        })
    except Exception as exc:
        app.logger.error("Error loading teacher credit classes: %s", exc, exc_info=True)
        return jsonify({'success': False, 'message': 'Không thể tải danh sách lớp'}), 500


@app.route('/api/teacher/credit-classes/<int:credit_class_id>/students', methods=['GET'])
@role_required('teacher', 'admin')
def api_teacher_credit_class_students(credit_class_id):
    teacher_param = request.args.get('teacher_id', type=int) if get_current_role() == 'admin' else None
    teacher = resolve_teacher_context(teacher_param)
    credit_class = db.get_credit_class(credit_class_id)
    if not credit_class:
        return jsonify({'success': False, 'message': 'Không tìm thấy lớp tín chỉ'}), 404
    if teacher and credit_class.get('teacher_id') and credit_class.get('teacher_id') != teacher.get('id'):
        return jsonify({'success': False, 'message': 'Không có quyền truy cập lớp này'}), 403

    try:
        roster = db.get_credit_class_students(credit_class_id) or []
        today_attendance = db.get_today_attendance(credit_class_id=credit_class_id) or []

        present_map = {}
        for att in today_attendance:
            sid = (att or {}).get('student_id')
            if not sid:
                continue
            present_map[sid] = {
                'attendance_id': att.get('attendance_id'),
                'check_in_time': att.get('check_in_time'),
                'checkout_time': att.get('checkout_time'),
                'checked_out': bool(att.get('checkout_time')),
            }

        serialized = []
        for student in roster:
            srec = dict(student)
            sid = srec.get('student_id')
            attendance_info = present_map.get(sid)
            if attendance_info:
                srec['is_present_today'] = True
                srec['checked_out'] = attendance_info.get('checked_out', False)
                srec['attendance_id'] = attendance_info.get('attendance_id')
                srec['check_in_time'] = attendance_info.get('check_in_time')
            else:
                srec['is_present_today'] = False
                srec['checked_out'] = False
                srec['attendance_id'] = None
                srec['check_in_time'] = None
            serialized.append(srec)

        return jsonify({
            'success': True,
            'credit_class': dict(credit_class),
            'students': serialized,
            'count': len(serialized),
        })
    except Exception as exc:
        app.logger.error("Error loading roster for credit class %s: %s", credit_class_id, exc, exc_info=True)
        return jsonify({'success': False, 'message': 'Không thể tải danh sách sinh viên'}), 500


@app.route('/api/teacher/credit-classes/<int:credit_class_id>/sessions', methods=['GET'])
@role_required('teacher', 'admin')
def api_teacher_credit_class_sessions(credit_class_id):
    teacher_param = request.args.get('teacher_id', type=int) if get_current_role() == 'admin' else None
    teacher = resolve_teacher_context(teacher_param)
    credit_class = db.get_credit_class(credit_class_id)
    if not credit_class:
        return jsonify({'success': False, 'message': 'Không tìm thấy lớp tín chỉ'}), 404
    if teacher and credit_class.get('teacher_id') and credit_class.get('teacher_id') != teacher.get('id'):
        return jsonify({'success': False, 'message': 'Không có quyền truy cập lớp này'}), 403

    limit = request.args.get('limit', 20, type=int) or 20
    limit = max(5, min(limit, 100))
    try:
        sessions = db.list_sessions_for_class(credit_class_id, limit=limit)
        serialized = [serialize_session_payload(row) for row in sessions]
        return jsonify({'success': True, 'credit_class': dict(credit_class), 'sessions': serialized})
    except Exception as exc:
        app.logger.error("Error loading sessions for credit class %s: %s", credit_class_id, exc, exc_info=True)
        return jsonify({'success': False, 'message': 'Không thể tải phiên điểm danh'}), 500


@app.route('/api/reports/credit-classes/<int:credit_class_id>/sessions', methods=['GET'])
def api_reports_credit_class_sessions(credit_class_id):
    """API công khai phục vụ bộ lọc báo cáo lấy danh sách phiên của lớp tín chỉ."""
    limit = request.args.get('limit', 25, type=int) or 25
    limit = max(5, min(limit, 100))

    try:
        credit_class = db.get_credit_class(credit_class_id)
        if not credit_class:
            return jsonify({'success': False, 'message': 'Không tìm thấy lớp tín chỉ'}), 404

        sessions = db.list_sessions_for_class(credit_class_id, limit=limit) or []
        payload = [serialize_session_payload(session) for session in sessions]
        return jsonify({
            'success': True,
            'credit_class': serialize_credit_class_record(credit_class),
            'sessions': payload,
        })
    except Exception as exc:
        app.logger.error(
            "Error loading report sessions for credit class %s: %s",
            credit_class_id,
            exc,
            exc_info=True,
        )
        return jsonify({'success': False, 'message': 'Không thể tải danh sách phiên'}), 500


@app.route('/api/student/credit-classes', methods=['GET'])
@role_required('student', 'admin')
def api_student_credit_classes():
    student_param = request.args.get('student_id') if get_current_role() == 'admin' else None
    student = resolve_student_context(student_param)
    if not student:
        return jsonify({'success': False, 'message': 'Không tìm thấy sinh viên'}), 404

    try:
        student_identifier = student.get('id') or student.get('student_id')
        classes = db.get_credit_classes_for_student(student_identifier)
        formatted = []
        active_sessions = 0
        known_class_ids = set()
        for cls in classes:
            session_row = db.get_active_session_for_class(cls['id'])
            if session_row:
                active_sessions += 1
            entry = dict(cls)
            entry['display_name'] = ' · '.join(
                part for part in [cls.get('subject_name'), cls.get('credit_code')] if part
            ) or cls.get('subject_name') or cls.get('credit_code')
            entry['active_session'] = serialize_session_payload(session_row)
            known_class_ids.add(cls.get('id'))
            formatted.append(entry)

        # Nếu sinh viên chưa đăng ký nhưng đang có phiên mở, hiển thị ở dạng session-only
        fallback_session = get_active_attendance_session()
        if fallback_session:
            fallback_class_id = fallback_session.get('credit_class_id')
            if fallback_class_id and fallback_class_id not in known_class_ids:
                credit_cls = db.get_credit_class(fallback_class_id)
                if credit_cls:
                    entry = dict(credit_cls)
                    entry['display_name'] = ' · '.join(
                        part for part in [credit_cls.get('subject_name'), credit_cls.get('credit_code')] if part
                    ) or credit_cls.get('subject_name') or credit_cls.get('credit_code')
                    entry['active_session'] = serialize_session_payload(fallback_session)
                    entry['is_session_only'] = True
                    if 'student_count' not in entry or entry.get('student_count') is None:
                        try:
                            roster = db.get_credit_class_students(fallback_class_id) or []
                            entry['student_count'] = len(roster)
                        except Exception:
                            entry['student_count'] = 0
                    formatted.insert(0, entry)
                    active_sessions += 1 if (fallback_session.get('status') == 'open') else 0

        summary = {
            'total_classes': len(formatted),
            'active_sessions': active_sessions,
        }

        return jsonify({
            'success': True,
            'student': {
                'student_id': student.get('student_id'),
                'full_name': student.get('full_name'),
                'email': student.get('email'),
                'class_id': student.get('class_id'),
            },
            'classes': formatted,
            'summary': summary,
        })
    except Exception as exc:
        app.logger.error("Error loading credit classes for student %s: %s", student.get('student_id'), exc, exc_info=True)
        return jsonify({'success': False, 'message': 'Không thể tải danh sách lớp tín chỉ'}), 500


@app.route('/api/classes', methods=['POST'])
@role_required('admin')
def api_create_class():
    """API tạo lớp học mới"""
    data = get_request_data()
    class_code = (data.get('class_code') or '').strip()
    class_name = (data.get('class_name') or '').strip()

    if not class_code or not class_name:
        return jsonify({'success': False, 'message': 'Mã lớp và tên lớp là bắt buộc'}), 400

    try:
        class_id = db.create_class(
            class_code=class_code,
            class_name=class_name,
            semester=data.get('semester'),
            academic_year=data.get('academic_year'),
            description=data.get('description'),
        )
        created_class = db.get_class_by_id(class_id)
        return jsonify({'success': True, 'data': created_class}), 201
    except ValueError as conflict:
        return jsonify({'success': False, 'message': str(conflict)}), 400
    except Exception as e:
        app.logger.error(f"Error creating class: {e}")
        return jsonify({'success': False, 'message': 'Không thể tạo lớp học'}), 500


@app.route('/api/classes/<int:class_id>', methods=['GET', 'PUT', 'DELETE'])
@role_required('admin')
def api_class_detail(class_id):
    """API lấy/cập nhật/xóa lớp học"""
    if request.method == 'GET':
        class_data = db.get_class_by_id(class_id)
        if not class_data:
            return jsonify({'success': False, 'message': 'Không tìm thấy lớp học'}), 404
        class_data = dict(class_data)
        students = db.get_students_by_class(class_id)
        class_data['student_count'] = len(students)
        return jsonify({'success': True, 'data': class_data})

    if request.method == 'PUT':
        data = get_request_data()
        updates = {k: v for k, v in data.items() if v not in (None, '')}
        try:
            updated = db.update_class(class_id, **updates)
            if not updated:
                return jsonify({'success': False, 'message': 'Không thể cập nhật lớp học'}), 400
            return jsonify({'success': True, 'data': db.get_class_by_id(class_id)})
        except Exception as e:
            app.logger.error(f"Error updating class {class_id}: {e}")
            return jsonify({'success': False, 'message': 'Lỗi khi cập nhật lớp học'}), 500

    try:
        deleted = db.delete_class(class_id)
        if not deleted:
            return jsonify({'success': False, 'message': 'Không thể xóa lớp học'}), 400
        return jsonify({'success': True})
    except Exception as e:
        app.logger.error(f"Error deleting class {class_id}: {e}")
        return jsonify({'success': False, 'message': 'Lỗi khi xóa lớp học'}), 500


@app.route('/api/classes/<int:class_id>/students', methods=['GET'])
@role_required('admin')
def api_class_students(class_id):
    """API danh sách sinh viên trong lớp"""
    class_data = db.get_class_by_id(class_id)
    if not class_data:
        return jsonify({'success': False, 'message': 'Không tìm thấy lớp học'}), 404

    try:
        students = db.get_students_by_class(class_id)
        serialized = [serialize_student_record(student) for student in students]
        return jsonify({
            'success': True,
            'class': class_data,
            'data': serialized,
        })
    except Exception as error:
        app.logger.error(f"Error getting students for class {class_id}: {error}")
        return jsonify({'success': False, 'message': 'Không thể tải danh sách sinh viên'}), 500


@app.route('/api/classes/<int:class_id>/stats', methods=['GET'])
@role_required('admin')
def api_class_stats(class_id):
    """API thống kê lớp học"""
    class_data = db.get_class_by_id(class_id)
    if not class_data:
        return jsonify({'success': False, 'message': 'Không tìm thấy lớp học'}), 404

    try:
        stats = db.get_class_attendance_stats(class_id)
        if not stats:
            stats = {}
        stats_payload = {
            'total_students': stats.get('total_students', 0),
            'attended_students': stats.get('attended_students', 0),
            'attendance_rate': stats.get('attendance_rate', 0),
            'daily_stats': stats.get('daily_stats', []),
            'period': stats.get('period'),
        }
        return jsonify({'success': True, 'class': class_data, 'stats': stats_payload})
    except Exception as error:
        app.logger.error(f"Error getting stats for class {class_id}: {error}")
        return jsonify({'success': False, 'message': 'Không thể tải thống kê lớp học'}), 500


@app.route('/api/attendance/session', methods=['GET'])
def api_get_attendance_session():
    """Trạng thái phiên điểm danh tín chỉ hiện tại."""
    session_row = get_active_attendance_session()
    return jsonify({
        'success': True,
        'session': serialize_session_payload(session_row),
        'default_duration': SESSION_DURATION_MINUTES,
    })


@app.route('/api/attendance/session/open', methods=['POST'])
@role_required('teacher')
def api_open_attendance_session():
    data = get_request_data()
    credit_class_id = data.get('credit_class_id')
    if not credit_class_id:
        return jsonify({'success': False, 'message': 'Vui lòng chọn lớp tín chỉ'}), 400

    try:
        credit_class_id = int(credit_class_id)
    except (ValueError, TypeError):
        return jsonify({'success': False, 'message': 'Mã lớp tín chỉ không hợp lệ'}), 400

    active_session = get_active_attendance_session()
    if active_session:
        active_class_id = active_session.get('credit_class_id')
        if active_class_id and int(active_class_id) != int(credit_class_id):
            return jsonify({'success': False, 'message': 'Đã có phiên điểm danh đang mở'}), 400
        try:
            db.close_attendance_session(active_session['id'], status='superseded')
            set_active_session_cache(None)
            broadcast_session_snapshot(force_reload=True)
        except Exception as exc:
            app.logger.warning(
                "Không thể tự động đóng phiên cũ %s: %s",
                active_session.get('id'),
                exc,
            )
            return jsonify({'success': False, 'message': 'Không thể đóng phiên cũ'}), 500

    teacher_ctx = resolve_teacher_context()
    if not teacher_ctx:
        return jsonify({'success': False, 'message': 'Không tìm thấy thông tin giảng viên'}), 403

    credit_class = db.get_credit_class(credit_class_id)
    if not credit_class or not credit_class.get('is_active', 1):
        return jsonify({'success': False, 'message': 'Không tìm thấy lớp tín chỉ'}), 404

    owner_id = credit_class.get('teacher_id')
    if not owner_id or int(owner_id) != int(teacher_ctx.get('id')):
        return jsonify({'success': False, 'message': 'Lớp tín chỉ không thuộc quyền quản lý của bạn'}), 403

    duration_minutes = data.get('duration_minutes')
    try:
        duration_minutes = int(duration_minutes) if duration_minutes is not None else SESSION_DURATION_MINUTES
    except (ValueError, TypeError):
        duration_minutes = SESSION_DURATION_MINUTES
    duration_minutes = max(1, min(duration_minutes, 90))

    now = datetime.now()
    deadline = (now + timedelta(minutes=duration_minutes)).isoformat()

    try:
        session_id = db.create_attendance_session(
            credit_class_id=credit_class_id,
            opened_by=g.user['id'] if getattr(g, 'user', None) else None,
            session_date=now.date().isoformat(),
            checkin_deadline=deadline,
            checkout_deadline=deadline,
            status='open',
            notes=data.get('notes')
        )
        session_row = db.get_session_by_id(session_id)
        set_active_session_cache(session_row)
        reset_session_runtime_state(session_row)
        payload = serialize_session_payload(session_row)
        broadcast_session_snapshot()
        return jsonify({'success': True, 'session': payload})
    except ValueError as err:
        return jsonify({'success': False, 'message': str(err)}), 400
    except Exception as exc:
        app.logger.error(f"Error opening attendance session: {exc}", exc_info=True)
        return jsonify({'success': False, 'message': 'Không thể mở phiên điểm danh'}), 500


@app.route('/api/attendance/session/close', methods=['POST'])
@role_required('admin', 'teacher')
def api_close_attendance_session():
    data = get_request_data()
    session_id = data.get('session_id')
    if session_id:
        try:
            session_id = int(session_id)
        except (ValueError, TypeError):
            return jsonify({'success': False, 'message': 'Mã phiên không hợp lệ'}), 400
    else:
        current_session = get_active_attendance_session()
        if not current_session:
            return jsonify({'success': False, 'message': 'Không có phiên nào đang mở'}), 400
        session_id = current_session.get('id')

    try:
        closed = db.close_attendance_session(session_id)
        if not closed:
            return jsonify({'success': False, 'message': 'Không thể đóng phiên'}), 400
        set_active_session_cache(None)
        broadcast_session_snapshot(force_reload=True)
        payload = serialize_session_payload(db.get_session_by_id(session_id))
        return jsonify({'success': True, 'session': payload})
    except Exception as exc:
        app.logger.error(f"Error closing attendance session {session_id}: {exc}", exc_info=True)
        return jsonify({'success': False, 'message': 'Không thể đóng phiên điểm danh'}), 500


@app.route('/api/attendance/session/<int:session_id>/mark', methods=['POST'])
@role_required('teacher', 'admin')
def api_mark_attendance_for_session(session_id):
    """Cho phép giảng viên (hoặc admin) điểm danh/checkout thủ công cho một phiên cụ thể."""
    data = get_request_data()
    student_code = data.get('student_id') or data.get('student_code')
    action = (data.get('action') or 'checkin').lower()

    if not student_code:
        return jsonify({'success': False, 'message': 'Missing student_id'}), 400

    session_row = db.get_session_by_id(session_id)
    if not session_row:
        return jsonify({'success': False, 'message': 'Session not found'}), 404

    credit_class = db.get_credit_class(session_row.get('credit_class_id'))
    if not credit_class:
        return jsonify({'success': False, 'message': 'Credit class not found'}), 404

    # Ủy quyền: giảng viên chỉ có thể điểm danh cho các lớp của họ
    if get_current_role() == 'teacher':
        teacher_ctx = resolve_teacher_context()
        if not teacher_ctx or int(credit_class.get('teacher_id') or 0) != int(teacher_ctx.get('id')):
            return jsonify({'success': False, 'message': 'Không có quyền trên lớp này'}), 403

    # Xác định sinh viên
    student_row = db.get_student(student_code)
    if not student_row:
        return jsonify({'success': False, 'message': 'Không tìm thấy sinh viên'}), 404
    student_name = student_row.get('full_name') or student_code

    try:
        if action in ('checkin', 'mark', 'present'):
            success = db.mark_attendance(
                student_id=student_code,
                student_name=student_name,
                status='present',
                confidence_score=None,
                notes='manual by teacher',
                credit_class_id=credit_class.get('id'),
                session_id=session_id,
            )
            if success:
                with today_recorded_lock:
                    today_checked_in.add(student_code)
                    today_checked_out.discard(student_code)
                    today_student_names[student_code] = {
                        'name': student_name,
                        'class_name': credit_class.get('subject_name') or credit_class.get('credit_code'),
                        'class_type': 'credit',
                        'credit_class_id': credit_class.get('id'),
                    }

                broadcast_sse_event({
                    'type': 'attendance_marked',
                    'data': {
                        'event': 'check_in',
                        'student_id': student_code,
                        'student_name': student_name,
                        'timestamp': datetime.now().isoformat(),
                        'session': serialize_session_payload(session_row),
                    },
                })
                return jsonify({'success': True})
            return jsonify({'success': False, 'message': 'Không thể điểm danh (có thể đã điểm danh trước đó)'}), 400

        elif action in ('checkout', 'check_out'):
            success = db.mark_checkout(student_code, session_id=session_id)
            if success:
                with today_recorded_lock:
                    today_checked_out.add(student_code)

                broadcast_sse_event({
                    'type': 'attendance_checkout',
                    'data': {
                        'event': 'check_out',
                        'student_id': student_code,
                        'student_name': student_name,
                        'timestamp': datetime.now().isoformat(),
                        'session': serialize_session_payload(session_row),
                    },
                })
                return jsonify({'success': True})
            return jsonify({'success': False, 'message': 'Không thể checkout hoặc chưa điểm danh'}), 400

        else:
            return jsonify({'success': False, 'message': 'Hành động không hợp lệ'}), 400

    except Exception as exc:
        app.logger.error(f"Error in manual mark endpoint: {exc}", exc_info=True)
        return jsonify({'success': False, 'message': 'Lỗi khi thực hiện điểm danh'}), 500


@app.route('/api/statistics')
def api_statistics():
    """API thống kê"""
    try:
        credit_class_id = request.args.get('credit_class_id', type=int)
        session_id = request.args.get('session_id', type=int)

        attendance_data = get_today_attendance(
            credit_class_id=credit_class_id,
            session_id=session_id,
        )

        total_students = len(known_face_names) if known_face_names else 0
        if credit_class_id:
            try:
                roster = db.get_credit_class_students(credit_class_id) or []
                total_students = len(roster)
            except Exception as roster_error:
                app.logger.warning(
                    "Không thể lấy sĩ số lớp tín chỉ %s: %s",
                    credit_class_id,
                    roster_error,
                )
        elif session_id:
            session_row = db.get_session_by_id(session_id)
            if session_row:
                resolved_class_id = session_row.get('credit_class_id')
                if resolved_class_id:
                    try:
                        roster = db.get_credit_class_students(resolved_class_id) or []
                        total_students = len(roster)
                    except Exception as roster_error:
                        app.logger.warning(
                            "Không thể lấy sĩ số cho phiên %s: %s",
                            session_id,
                            roster_error,
                        )

        attended_students = len(attendance_data)
        attendance_rate = (attended_students / total_students * 100) if total_students > 0 else 0
        
        # Tính tổng thời gian có mặt
        total_minutes = sum(item['duration_minutes'] for item in attendance_data if item['duration_minutes'])
        avg_duration = int(total_minutes / attended_students) if attended_students > 0 else 0

        return jsonify({
            'total_students': total_students,
            'attended_students': attended_students,
            'attendance_rate': round(attendance_rate, 2),
            'avg_duration_minutes': avg_duration,
            'total_duration_minutes': total_minutes
        })
    except Exception as e:
        app.logger.error(f"Error getting statistics: {e}")
        return jsonify({
            'total_students': 0, 
            'attended_students': 0, 
            'attendance_rate': 0,
            'avg_duration_minutes': 0,
            'total_duration_minutes': 0
        }), 500

@app.route('/api/presence/active', methods=['GET'])
def api_active_presence():
    """API lấy danh sách sinh viên đang có mặt (đang được tracking)"""
    try:
        with presence_tracking_lock:
            active_students = []
            now = datetime.now()
            
            for student_id, data in presence_tracking.items():
                check_in_time = data['check_in_time']
                last_seen = data['last_seen']
                duration_seconds = (now - check_in_time).total_seconds()
                time_since_seen = (now - last_seen).total_seconds()
                
                active_students.append({
                    'student_id': student_id,
                    'name': data['name'],
                    'check_in_time': check_in_time.isoformat(),
                    'last_seen': last_seen.isoformat(),
                    'duration_minutes': int(duration_seconds / 60),
                    'seconds_since_seen': int(time_since_seen),
                    'is_active': time_since_seen < 30  # Còn active nếu thấy trong 30s
                })
            
            return jsonify({
                'success': True,
                'count': len(active_students),
                'data': active_students
            })
    except Exception as e:
        app.logger.error(f"Error getting active presence: {e}")
        return jsonify({'success': False, 'data': [], 'error': str(e)}), 500

@app.route('/api/attendance/today', methods=['GET'])
def api_attendance_today():
    """API lấy điểm danh hôm nay"""
    try:
        credit_class_id = request.args.get('credit_class_id', type=int)
        session_id = request.args.get('session_id', type=int)

        attendance_data = get_today_attendance(
            credit_class_id=credit_class_id,
            session_id=session_id,
        )

        session_row = None
        if session_id:
            session_row = db.get_session_by_id(session_id)
        elif credit_class_id:
            session_row = db.get_active_session_for_class(credit_class_id)
        else:
            session_row = get_active_attendance_session()

        checked_in = []
        checked_out = []
        for item in attendance_data:
            if item.get('checkout_time'):
                checked_out.append(item)
            else:
                checked_in.append(item)
        return jsonify({
            'success': True,
            'data': attendance_data,
            'checked_in': checked_in,
            'checked_out': checked_out,
            'session': serialize_session_payload(session_row)
        })
    except Exception as e:
        app.logger.error(f"Error getting today's attendance: {e}")
        return jsonify({'success': False, 'data': [], 'error': str(e)}), 500


@app.route('/api/attendance/history/<student_id>', methods=['GET'])
def api_attendance_history(student_id):
    """API trả về lịch sử điểm danh gần đây của một sinh viên"""
    limit = request.args.get('limit', 10, type=int) or 10
    limit = max(1, min(limit, 50))

    try:
        history_rows = db.get_student_attendance_history(student_id, limit) or []
        student_info = db.get_student(student_id)

        if not history_rows and not student_info:
            return jsonify({'success': False, 'error': 'Không tìm thấy sinh viên'}), 404

        class_name = None
        student_name = None

        if student_info:
            student_info_dict = dict(student_info)
            student_name = student_info_dict.get('full_name') or student_info_dict.get('name')
            class_id = student_info_dict.get('class_id')
            if class_id:
                class_info = db.get_class_by_id(class_id)
                if class_info:
                    class_info_dict = dict(class_info)
                    class_name = class_info_dict.get('class_name') or class_info_dict.get('name')

        history = []
        now = datetime.now()
        status_class_map = {
            'present': 'bg-success',
            'late': 'bg-warning text-dark',
            'absent': 'bg-danger',
            'excused': 'bg-info text-dark'
        }

        summary = {
            'total_sessions': len(history_rows),
            'last_check_in': None,
            'last_check_out': None,
            'current_status': None,
            'status_class': 'bg-secondary'
        }

        for index, row in enumerate(history_rows):
            row_dict = dict(row)
            check_in = parse_datetime_safe(row_dict.get('check_in_time'))
            check_out = parse_datetime_safe(row_dict.get('check_out_time'))

            # Nếu bảng attendance có lưu tên lớp trực tiếp
            if not class_name and row_dict.get('class_name'):
                class_name = row_dict.get('class_name')

            if not student_name:
                student_name = row_dict.get('student_name') or row_dict.get('full_name')

            if check_in and check_out:
                duration_seconds = max((check_out - check_in).total_seconds(), 0)
            elif check_in:
                duration_seconds = max((now - check_in).total_seconds(), 0)
            else:
                duration_seconds = 0

            record = {
                'attendance_date': row_dict.get('attendance_date'),
                'check_in_time': row_dict.get('check_in_time'),
                'check_out_time': row_dict.get('check_out_time'),
                'status': row_dict.get('status'),
                'duration_minutes': int(duration_seconds / 60),
                'notes': row_dict.get('notes'),
                'class_type': 'credit' if row_dict.get('credit_class_id') else 'administrative',
                'class_display': row_dict.get('credit_class_name') or row_dict.get('class_name'),
                'credit_class_code': row_dict.get('credit_class_code'),
                'credit_class_id': row_dict.get('credit_class_id')
            }
            history.append(record)

            if index == 0:
                summary['last_check_in'] = row_dict.get('check_in_time')
                summary['last_check_out'] = row_dict.get('check_out_time')
                summary['current_status'] = row_dict.get('status')
                summary['status_class'] = status_class_map.get(row_dict.get('status'), 'bg-secondary')

        response_payload = {
            'success': True,
            'student_id': student_id,
            'student_name': student_name or student_id,
            'class_name': class_name,
            'summary': summary,
            'history': history
        }

        return jsonify(response_payload)
    except Exception as e:
        app.logger.error(f"Error getting attendance history for {student_id}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/events/stream')
def api_events_stream():
    """Server-Sent Events stream cho thông báo real-time"""
    import json
    
    def event_stream():
        # Tạo queue cho client này
        client_queue = queue.Queue(maxsize=10)
        
        # Thêm vào danh sách clients
        with sse_clients_lock:
            sse_clients.append(client_queue)

        initial_session = serialize_session_payload(get_active_attendance_session())
        if initial_session:
            try:
                client_queue.put_nowait({'type': 'session_updated', 'data': initial_session})
            except queue.Full:
                pass
        
        try:
            # Gửi event kết nối thành công
            yield f"data: {json.dumps({'type': 'connected'})}\n\n"
            
            # Lắng nghe events từ queue
            while True:
                try:
                    event_data = client_queue.get(timeout=30)
                    yield f"data: {json.dumps(event_data)}\n\n"
                except queue.Empty:
                    # Gửi heartbeat để giữ kết nối
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
        except GeneratorExit:
            # Client đã ngắt kết nối
            with sse_clients_lock:
                if client_queue in sse_clients:
                    sse_clients.remove(client_queue)
    
    return Response(event_stream(), mimetype='text/event-stream')


@app.route('/api/attendance/notifications', methods=['GET'])
def api_attendance_notifications():
    """API trả về các thông báo điểm danh / hệ thống để frontend hiển thị"""
    try:
        notifications = []

        # Lấy các bản ghi điểm danh gần đây (hôm nay) và chuyển thành thông báo
        try:
            session_row = get_active_attendance_session()
            session_id = session_row.get('id') if session_row else None
            credit_class_id = session_row.get('credit_class_id') if session_row else None
            recent_att = db.get_today_attendance(
                session_id=session_id,
                credit_class_id=credit_class_id,
            )
            # Chỉ lấy 5 bản ghi gần nhất
            for row in recent_att[:5]:
                msg = f"{row.get('student_name') or row.get('full_name')} đã điểm danh"
                ts = row.get('check_in_time')
                notifications.append({'message': msg, 'type': 'success', 'timestamp': ts})
        except Exception as e:
            app.logger.debug(f"Không lấy được attendance để làm notifications: {e}")

        # Lấy system logs gần nhất để hiển thị
        try:
            logs = db.get_system_logs(limit=10)
            for log in logs:
                level = (log.get('log_level') if isinstance(log, dict) else log['log_level'])
                message = (log.get('message') if isinstance(log, dict) else log['message'])
                ts = (log.get('created_at') if isinstance(log, dict) else log['created_at'])
                # Map level -> bootstrap alert type
                if level and level.upper() in ('ERROR', 'CRITICAL'):
                    ntype = 'danger'
                elif level and level.upper() in ('WARNING',):
                    ntype = 'warning'
                else:
                    ntype = 'info'

                notifications.append({'message': message, 'type': ntype, 'timestamp': ts})
        except Exception as e:
            app.logger.debug(f"Không lấy được system logs cho notifications: {e}")

        return jsonify({'notifications': notifications})
    except Exception as e:
        app.logger.error(f"Error building notifications: {e}")
        return jsonify({'notifications': []}), 500

@app.route('/update_faces', methods=['POST'])
def update_faces():
    """API cập nhật khuôn mặt"""
    try:
        load_known_faces()
        return 'Cap nhat thanh cong', 200
    except Exception as e:
        app.logger.error(f"Error updating faces: {e}")
        return f'Loi: {e}', 500

# ===== CÁC ROUTE HUẤN LUYỆN AI NÂNG CAO =====

@app.route('/api/train/start', methods=['POST'])
def api_train_start():
    """Bắt đầu training classifier với FaceNet embeddings"""
    if not USE_FACENET or face_service is None:
        return jsonify({'error': 'FaceNet service not available'}), 400
    
    try:
        # Khởi tạo training service nếu chưa có
        global training_service
        if training_service is None:
            from services.training_service import TrainingService
            training_service = TrainingService(face_service)
        
        # Kiểm tra trước khi huấn luyện: đảm bảo mỗi sinh viên có đủ mẫu theo cấu hình
        stats = training_service.get_training_stats()
        not_ready = [s for s in (stats.get('students') or []) if not s.get('ready')]
        if not_ready:
            return jsonify({
                'success': False,
                'error': 'Insufficient training data',
                'details': {
                    'min_required': stats.get('min_samples_required'),
                    'students': not_ready
                }
            }), 400

        # Train classifier
        success = training_service.train_classifier()

        if success:
            stats = training_service.get_training_stats()
            return jsonify({
                'success': True,
                'message': 'Training completed successfully',
                'stats': stats
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Training failed - see server logs'
            }), 500
    
    except Exception as e:
        app.logger.error(f"Training error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/train/status', methods=['GET'])
def api_train_status():
    """Lấy thông tin về training data"""
    if not USE_FACENET or face_service is None:
        return jsonify({'error': 'FaceNet service not available'}), 400
    
    try:
        global training_service
        if training_service is None:
            from services.training_service import TrainingService
            training_service = TrainingService(face_service)
        
        stats = training_service.get_training_stats()
        return jsonify({'success': True, 'stats': stats})
    
    except Exception as e:
        app.logger.error(f"Error getting training status: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/antispoof/check', methods=['POST'])
def api_antispoof_check():
    """Kiểm tra anti-spoofing cho frame hiện tại"""
    if not USE_FACENET or antispoof_service is None:
        return jsonify({'error': 'Anti-spoof service not available'}), 400
    
    try:
        # Lấy ảnh từ request (base64 hoặc tải lên file)
        if 'image_data' in request.form:
            # Ảnh Base64
            image_data = request.form['image_data']
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            img_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        elif 'image' in request.files:
            # Tải lên file
            file = request.files['image']
            img_bytes = file.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        # Kiểm tra giả mạo
        result = antispoof_service.check_frame(frame)
        
        return jsonify({
            'success': True,
            'is_real': result['is_real'],
            'confidence': result['confidence'],
            'message': result['message'],
            'bbox': result['bbox']
        })
    
    except Exception as e:
        app.logger.error(f"Anti-spoof check error: {e}")
        return jsonify({'error': str(e)}), 500


# Các hàm helper
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_face_image(image_path, student_id):
    """Xử lý ảnh khuôn mặt"""
    try:
        if FACE_RECOGNITION_AVAILABLE:
            img = face_recognition.load_image_file(str(image_path))
            encodings = face_recognition.face_encodings(img)
            if encodings:
                app.logger.info(f"Processed face image for student {student_id}")
            else:
                app.logger.warning(f"No face found in image for student {student_id}")
        else:
            app.logger.warning(f"Face recognition not available, skipping face processing for {student_id}")
    except Exception as e:
        app.logger.error(f"Error processing face image for {student_id}: {e}")


# ===== CÁC HÀM HELPER TỪ FACENET =====

def prewhiten_facenet(x):
    """
    FaceNet-style prewhitening để chuẩn hóa tốt hơn.
    Được điều chỉnh từ face_attendance/facenet.py
    """
    if isinstance(x, np.ndarray):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
        y = (x - mean) / std_adj
        return y
    return x


def estimate_head_pose(landmarks, frame_size):
    """
    Ước tính tư thế đầu đơn giản (yaw, pitch, roll) theo độ bằng solvePnP.
    landmarks: dictionary hoặc danh sách các điểm (x,y) cho các mốc quan trọng (chúng tôi mong đợi ít nhất
    left_eye, right_eye, nose, left_mouth, right_mouth) hoặc danh sách theo thứ tự trả về
    bởi dlib/face_recognition: chúng tôi sẽ cố gắng xử lý các định dạng phổ biến.
    Trả về (yaw_deg, pitch_deg, roll_deg) hoặc (None, None, None) nếu thất bại.
    """
    try:
        # Chuyển đổi landmarks thành mảng numpy các điểm 2D cần thiết
        lm = None
        if isinstance(landmarks, dict):
            # face_recognition đôi khi trả về 'left_eye','right_eye','nose_tip','mouth_left','mouth_right'
            keys = ['left_eye', 'right_eye', 'nose_tip', 'mouth_left', 'mouth_right']
            pts2 = []
            for k in keys:
                if k in landmarks:
                    pts2.append(landmarks[k])
            if len(pts2) < 4:
                # dự phòng: sử dụng tất cả các giá trị dict
                pts2 = list(landmarks.values())
        else:
            pts2 = list(landmarks)

        if len(pts2) < 4:
            return (None, None, None)

        # Chọn 4-5 điểm ổn định: mắt trái, mắt phải, mũi, khóe miệng trái, khóe miệng phải
        # Sử dụng các điểm mô hình 3D chung (xấp xỉ)
        model_points = np.array([ 
            ( -30.0,  30.0,  -30.0),   # left eye
            (  30.0,  30.0,  -30.0),   # right eye
            (   0.0,   0.0,    0.0),   # nose tip
            ( -25.0, -30.0,  -25.0),   # left mouth
            (  25.0, -30.0,  -25.0)    # right mouth
        ], dtype=np.float64)

        # Ánh xạ các điểm ảnh 2D từ landmarks (lấy 5 điểm đầu tiên)
        image_points = []
        for i in range(min(len(pts2), 5)):
            p = pts2[i]
            image_points.append((float(p[0]), float(p[1])))
        image_points = np.array(image_points, dtype=np.float64)

        # Nếu chúng ta có ít điểm hơn mô hình, giảm số điểm mô hình để khớp
        if image_points.shape[0] < model_points.shape[0]:
            model_points = model_points[:image_points.shape[0]]

        # Thông số nội tại của camera (xấp xỉ)
        size = frame_size
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4,1))  # giả sử không có biến dạng ống kính

        # solvePnP
        success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            return (None, None, None)

        # Chuyển đổi vector quay thành ma trận quay
        rmat, _ = cv2.Rodrigues(rotation_vector)
        # Tạo ma trận chiếu sau đó phân rã thành các góc Euler
        pose_mat = cv2.hconcat((rmat, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

        # euler_angles: [pitch, yaw, roll] theo độ (thứ tự OpenCV)
        pitch, yaw, roll = float(euler_angles[0]), float(euler_angles[1]), float(euler_angles[2])
        return (yaw, pitch, roll)
    except Exception as e:
        app.logger.debug(f"estimate_head_pose failed: {e}")
        return (None, None, None)



def draw_progress_bar(frame, progress, x, y, w=150, h=20):
    """
    Vẽ thanh tiến trình để xác nhận điểm danh.
    Được điều chỉnh từ face_attendance/reg.py
    
    Args:
        frame: Khung hình video
        progress: Giá trị tiến trình (0.0 đến 1.0)
        x, y: Tọa độ trên cùng bên trái
        w, h: Chiều rộng và chiều cao của thanh
    """
    bar_y = y - 30  # Phía trên hộp khuôn mặt
    
    # Nền (đen)
    cv2.rectangle(frame, (x, bar_y), (x + w, bar_y + h), (0, 0, 0), -1)
    
    # Tiến trình (xanh lá)
    filled_width = int(w * progress)
    if filled_width > 0:
        cv2.rectangle(frame, (x, bar_y), (x + filled_width, bar_y + h), (0, 255, 0), -1)
    
    # Viền
    cv2.rectangle(frame, (x, bar_y), (x + w, bar_y + h), (255, 255, 255), 1)
    
    # Văn bản phần trăm
    progress_text = f"{int(progress * 100)}%"
    text_size = cv2.getTextSize(progress_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
    text_x = x + (w - text_size[0]) // 2
    text_y = bar_y + (h + text_size[1]) // 2
    cv2.putText(frame, progress_text, (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)


def update_progress(student_id, name):
    """
    Cập nhật tiến trình xác nhận điểm danh.
    Hành vi mới: yêu cầu thời gian nhìn thẳng liên tục (LOOK_STRAIGHT_SECONDS) để xác nhận.
    Trả về: (elapsed_seconds, required_seconds, is_confirmed)
    """
    global attendance_progress
    now = datetime.now()
    with attendance_progress_lock:
        entry = attendance_progress.get(student_id)
        if entry is None:
            # Bắt đầu cửa sổ nhìn thẳng mới
            attendance_progress[student_id] = {
                'start_time': now,
                'last_seen': now,
                'name': name
            }
            elapsed = 0.0
        else:
            # Tiếp tục cửa sổ
            # Nếu có khoảng cách dài kể từ lần cuối nhìn thấy, khởi động lại cửa sổ
            last = entry.get('last_seen')
            gap = (now - last).total_seconds() if last else 9999
            if gap > 1.5:  # nếu mất tích > 1.5s, đặt lại bộ đếm thời gian nhìn thẳng
                attendance_progress[student_id] = {
                    'start_time': now,
                    'last_seen': now,
                    'name': name
                }
                elapsed = 0.0
            else:
                # Cập nhật lần cuối nhìn thấy và tính toán số giây nhìn thẳng liên tục đã trôi qua
                entry['last_seen'] = now
                elapsed = (now - entry['start_time']).total_seconds()

        is_confirmed = elapsed >= LOOK_STRAIGHT_SECONDS
        return elapsed, LOOK_STRAIGHT_SECONDS, is_confirmed


def reset_progress(student_id):
    """Đặt lại tiến trình cho một sinh viên."""
    global attendance_progress
    
    with attendance_progress_lock:
        if student_id in attendance_progress:
            del attendance_progress[student_id]


# Khởi tạo
if __name__ == '__main__':
    try:
        # Khởi tạo database
        db.init_database()
        
        # Đảm bảo các thư mục tồn tại
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Tải các khuôn mặt đã biết
        load_known_faces()
        
        # Tải tập hợp đã ghi hôm nay từ database
        load_today_recorded()
        
        # Ghi log khởi động hệ thống
        db.log_system_event('INFO', 'He thong diem danh khoi dong', 'app')
        app.logger.info("He thong diem danh da khoi dong thanh cong")
        
        app.run(host='0.0.0.0', port=5000, debug=True)
        
    except Exception as e:
        app.logger.error(f"Loi khoi dong he thong: {e}")

