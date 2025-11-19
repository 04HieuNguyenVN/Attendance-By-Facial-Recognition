# Set UTF-8 encoding for console output
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Flask imports
from flask import Flask, render_template, Response, redirect, url_for, request, jsonify

# Standard library imports
import os
import csv
import time
import random
import base64
from pathlib import Path
from datetime import datetime, date
import threading
# Note: use threading.Thread / threading.Lock via the threading module to avoid
# duplicate unused names in the module namespace.

# Third-party imports
import cv2
import numpy as np
from werkzeug.utils import secure_filename
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Local imports
from database import db
from logging_config import setup_logging

# Try to load dotenv, but don't fail if not installed
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
MIN_FILE_SIZE = 1024  # 1 KB - tối thiểu
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# Setup logging
setup_logging(app)

# Cấu hình chế độ demo (sử dụng os.getenv sau khi load_dotenv)
DEMO_MODE = os.getenv('DEMO_MODE', '0') == '1'
USE_FACENET = os.getenv('USE_FACENET', '1') == '1'  # Sử dụng FaceNet thay vì face_recognition

# Import advanced face recognition services (FaceNet-based)
face_service = None
antispoof_service = None
training_service = None
FACE_RECOGNITION_AVAILABLE = False

if USE_FACENET and not DEMO_MODE:
    try:
        from services.face_service import FaceRecognitionService
        from services.antispoof_service import AntiSpoofService
        from services.training_service import TrainingService
        
        # Initialize FaceNet service
        face_service = FaceRecognitionService(
            confidence_threshold=float(os.getenv('FACENET_THRESHOLD', '0.85'))
        )
        
        # Initialize anti-spoof service
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

# Try to import DeepFace and DeepFace DB helper (from Cong-Nghe-Xu-Ly-Anh system)
DEEPFACE_AVAILABLE = False
try:
    from deepface import DeepFace
    from services.deepface_db import build_db_from_data_dir, recognize_face as deepface_recognize
    DEEPFACE_AVAILABLE = True
    app.logger.info("DeepFace library available - using Facenet512 for face recognition")
except ImportError:
    app.logger.warning("DeepFace not available, will try face_recognition as fallback")

# Try to import YOLOv8 for face detection (optional but recommended)
YOLO_AVAILABLE = False
yolo_face_model = None
try:
    from ultralytics import YOLO
    # Try to load YOLOv8 face detection model từ nhiều vị trí có thể
    possible_paths = [
        os.path.join(os.path.dirname(__file__), 'yolov8m-face.pt'),  # Thư mục gốc
        os.path.join(os.path.dirname(__file__), 'models', 'yolov8m-face.pt'),  # Thư mục models
        os.path.join(os.path.dirname(__file__), 'Cong-Nghe-Xu-Ly-Anh', 'yolov8m-face.pt'),  # Thư mục tham khảo (fallback)
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

# Fallback: Import legacy face_recognition library
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
    # If we have DeepFace or FaceNet, we can do face recognition
    if not FACE_RECOGNITION_AVAILABLE:
        FACE_RECOGNITION_AVAILABLE = DEEPFACE_AVAILABLE or USE_FACENET

# Chỉ số thiết bị camera (sử dụng os.getenv sau khi load_dotenv)
CAMERA_INDEX = int(os.getenv('CAMERA_INDEX', '0'))

# Ngưỡng confidence cho face recognition
FACE_RECOGNITION_THRESHOLD = float(os.getenv('FACE_RECOGNITION_THRESHOLD', '0.6'))
FACE_DISTANCE_THRESHOLD = float(os.getenv('FACE_DISTANCE_THRESHOLD', '0.45'))  # Khoảng cách tối đa (càng nhỏ càng giống)
# Ngưỡng cho DeepFace cosine similarity (similarity >= threshold được chấp nhận)
DEEPFACE_SIMILARITY_THRESHOLD = float(os.getenv('DEEPFACE_SIMILARITY_THRESHOLD', '0.6'))  # Cosine similarity tối thiểu
# Temporal + pose-based confirmation (require looking straight for N seconds)
LOOK_STRAIGHT_SECONDS = float(os.getenv('LOOK_STRAIGHT_SECONDS', '10'))  # seconds
FRONTAL_YAW_RATIO_THRESHOLD = float(os.getenv('FRONTAL_YAW_RATIO_THRESHOLD', '0.15'))
FRONTAL_ROLL_DEG_THRESHOLD = float(os.getenv('FRONTAL_ROLL_DEG_THRESHOLD', '15'))
# Tối ưu hiệu năng phát hiện
YOLO_FRAME_SKIP = max(1, int(os.getenv('YOLO_FRAME_SKIP', '2')))  # Chỉ chạy YOLO mỗi N khung hình
YOLO_INFERENCE_WIDTH = int(os.getenv('YOLO_INFERENCE_WIDTH', '640'))  # Resize YOLO, 0 = giữ nguyên

# Đường dẫn thư mục dữ liệu
DATA_DIR = Path('data')

# Video capture và khóa
video_capture = None
video_lock = threading.Lock()
camera_enabled = True  # Biến để bật/tắt camera

# Khởi tạo biến global cho face recognition
known_face_encodings = []
known_face_names = []
known_face_ids = []
# DeepFace DB embeddings (Facenet512)
known_face_embeddings = []  # np.ndarray (N, D)
today_checked_in = set()  # student_ids đã check-in
today_checked_out = set()  # student_ids đã checkout
today_student_names = {}  # student_id -> name
today_recorded_lock = threading.Lock()

# Theo dõi thời gian có mặt
presence_tracking = {}  # {student_id: {'last_seen': datetime, 'total_time': seconds}}
presence_tracking_lock = threading.Lock()
PRESENCE_TIMEOUT = 300  # 5 phút (300 giây) - nếu không thấy sẽ tự checkout

# Chống trùng lặp điểm danh (từ hệ thống mẫu Cong-Nghe-Xu-Ly-Anh)
# Chỉ cho phép điểm danh lại sau 30 giây (tránh điểm danh liên tục)
last_recognized = {}  # {student_id: datetime}
last_recognized_lock = threading.Lock()
RECOGNITION_COOLDOWN = 30  # Giây - thời gian chờ giữa các lần điểm danh

# Progress tracking for attendance confirmation (inspired by reg.py)
# attendance_progress will track continuous frontal-looking time windows per student:
# {student_id: {'start_time': datetime, 'last_seen': datetime, 'name': str}}
attendance_progress = {}
attendance_progress_lock = threading.Lock()
REQUIRED_FRAMES = 30  # Legacy fallback - not used for time-based confirmation

# Server-Sent Events for real-time notifications
import queue
sse_clients = []  # List of queues for each connected SSE client
sse_clients_lock = threading.Lock()

# Khởi tạo camera đơn giản nhất có thể
def ensure_video_capture():
    global video_capture
    if video_capture is not None and getattr(video_capture, 'isOpened', lambda: False)():
        app.logger.debug("[Camera] Camera đã được khởi tạo và đang mở")
        return
    
    # Khởi tạo camera đơn giản nhất có thể
    app.logger.info(f"[Camera] 🎥 Đang khởi tạo camera với index={CAMERA_INDEX}...")
    try:
        video_capture = cv2.VideoCapture(CAMERA_INDEX)
        
        if not video_capture.isOpened():
            app.logger.error(f"[Camera] ❌ Không thể mở camera với index={CAMERA_INDEX}")
            video_capture = None
            return
        
        app.logger.info(f"[Camera] ✅ Camera đã mở thành công (index={CAMERA_INDEX})")
        
        # Set lower resolution by default to reduce CPU and network usage
        try:
            CAMERA_WIDTH = int(os.getenv('CAMERA_WIDTH', '640'))
            CAMERA_HEIGHT = int(os.getenv('CAMERA_HEIGHT', '480'))
            video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            
            # Lấy thông tin thực tế của camera
            actual_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            
            app.logger.info(f"[Camera] 📐 Độ phân giải: {actual_width}x{actual_height}, FPS: {fps:.2f}")
            
            # Try to set a small buffer if supported
            if hasattr(cv2, 'CAP_PROP_BUFFERSIZE'):
                try:
                    video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                    buffer_size = int(video_capture.get(cv2.CAP_PROP_BUFFERSIZE))
                    app.logger.debug(f"[Camera] Buffer size: {buffer_size}")
                except Exception:
                    pass
        except Exception as e:
            app.logger.warning(f"[Camera] ⚠️ Không thể thiết lập thông số camera: {e}")

        # Warm-up: read a few frames to clear initial buffer
        app.logger.debug("[Camera] 🔄 Đang warm-up camera (đọc 3 frame đầu)...")
        warmup_success = 0
        for i in range(3):
            try:
                ret, _ = video_capture.read()
                if ret:
                    warmup_success += 1
                else:
                    app.logger.warning(f"[Camera] ⚠️ Frame {i+1} warm-up không đọc được")
                    break
            except Exception as e:
                app.logger.warning(f"[Camera] ⚠️ Lỗi khi đọc frame {i+1} warm-up: {e}")
                break
        
        if warmup_success > 0:
            app.logger.info(f"[Camera] ✅ Warm-up thành công ({warmup_success}/3 frames)")
        else:
            app.logger.warning("[Camera] ⚠️ Warm-up không thành công, camera có thể có vấn đề")
        
        app.logger.info("[Camera] ✅ Camera đã sẵn sàng sử dụng")
    except Exception as e:
        app.logger.error(f"[Camera] ❌ Lỗi khởi tạo camera: {e}", exc_info=True)
        video_capture = None

# ============================================================================
# HỆ THỐNG NHẬN DIỆN VÀ ĐIỂM DANH - VIẾT LẠI DỰA TRÊN DỰ ÁN THAM KHẢO
# Logic từ: Cong-Nghe-Xu-Ly-Anh/diemdanh_deepface_gui.py
# ============================================================================

# Tải khuôn mặt đã biết từ DATA_DIR (giống hệt hệ thống mẫu)
def load_known_faces():
    """
    Load ảnh mẫu và tính embedding bằng DeepFace Facenet512.
    Logic giống hệt Cong-Nghe-Xu-Ly-Anh/diemdanh_deepface_gui.py
    """
    global known_face_embeddings, known_face_names, known_face_ids
    
    app.logger.info(f"[LoadFaces] 🔄 Bắt đầu load khuôn mặt từ {DATA_DIR}...")
    
    # Reset
    known_face_embeddings = []
    known_face_names = []
    known_face_ids = []
    
    if not DATA_DIR.exists():
        app.logger.warning(f"[LoadFaces] ⚠️ Thư mục {DATA_DIR} không tồn tại, đang tạo mới...")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        app.logger.info(f"[LoadFaces] ✅ Đã tạo thư mục {DATA_DIR}")
        return
    
    # Kiểm tra DeepFace
    if not DEEPFACE_AVAILABLE:
        app.logger.error("[LoadFaces] ❌ DeepFace không khả dụng. Vui lòng cài đặt: pip install deepface")
        return
    
    # Load ảnh mẫu và tính embedding (giống hệt hệ thống mẫu)
    app.logger.info("[LoadFaces] [DeepFace] 🧠 Đang tải ảnh mẫu và tính embedding với Facenet512...")
    
    db_embeddings = []
    db_labels = []
    processed_count = 0
    failed_count = 0
    
    # Lấy tất cả file ảnh
    image_files = list(DATA_DIR.glob('*.jpg')) + list(DATA_DIR.glob('*.jpeg')) + list(DATA_DIR.glob('*.png'))
    app.logger.info(f"[LoadFaces] 📁 Tìm thấy {len(image_files)} file ảnh")
    
    for img_path in image_files:
        try:
            # Parse student info from filename (format: ID_Name.jpg hoặc ID_Name1_Name2.jpg)
            filename = img_path.stem
            import re
            match = re.match(r'^(\d+)_([A-Za-z\s]+)', filename)
            if match:
                student_id = match.group(1)
                name = match.group(2).strip()
            else:
                # Fallback: tách bằng underscore
                parts = filename.split('_')
                if len(parts) >= 2:
                    student_id = parts[0]
                    name = '_'.join(parts[1:])
                else:
                    student_id = filename
                    name = filename
            
            app.logger.debug(f"[LoadFaces] Đang xử lý {img_path.name} -> {name} (ID: {student_id})...")
            
            # Tính embedding bằng DeepFace Facenet512 (giống hệt hệ thống mẫu)
            embedding = DeepFace.represent(
                img_path=str(img_path),
                model_name="Facenet512",
                enforce_detection=True
            )[0]["embedding"]
            
            db_embeddings.append(embedding)
            db_labels.append((student_id, name))
            processed_count += 1
            
            app.logger.info(f"[LoadFaces] ✅ Đã tải khuôn mặt cho {name} (id={student_id}) từ {img_path.name}")
            
        except Exception as e:
            failed_count += 1
            app.logger.error(f"[LoadFaces] ❌ Lỗi khi xử lý ảnh mẫu {img_path.name}: {e}", exc_info=True)
    
    # Convert sang numpy array (giống hệ thống mẫu)
    if len(db_embeddings) > 0:
        known_face_embeddings = np.array(db_embeddings)
        
        # Lưu labels
        for student_id, name in db_labels:
            known_face_names.append(name)
            known_face_ids.append(student_id)
        
        app.logger.info(f"[LoadFaces] ✅ Đã load {len(known_face_embeddings)} ảnh mẫu với Facenet512 embeddings")
        app.logger.info(f"[LoadFaces] 📋 Known faces: {known_face_names}")
        app.logger.info(f"[LoadFaces] 📋 Known IDs: {known_face_ids}")
        app.logger.info(f"[LoadFaces] 📐 Embeddings shape: {known_face_embeddings.shape}")
        app.logger.info(f"[LoadFaces] 📊 Kết quả: {processed_count} thành công, {failed_count} thất bại")
    else:
        app.logger.warning("[LoadFaces] ⚠️ Không load được ảnh nào!")

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
            try:
                if is_base64:
                    # Decode base64
                    image_data = file_path
                    if ',' in image_data:
                        image_data = image_data.split(',')[1]
                    img_bytes = base64.b64decode(image_data)
                    img = Image.open(io.BytesIO(img_bytes))
                else:
                    img = Image.open(file_path)
                
                # Kiểm tra format
                if img.format not in ['JPEG', 'PNG']:
                    return False, f"Định dạng không được hỗ trợ: {img.format}. Chỉ chấp nhận JPG, JPEG, PNG", 0
                
                # Kiểm tra mode (phải là RGB hoặc có thể convert sang RGB)
                if img.mode not in ['RGB', 'L', 'RGBA']:
                    return False, f"Chế độ màu không được hỗ trợ: {img.mode}. Cần RGB hoặc Grayscale", 0
                
                # Convert sang RGB nếu cần
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    
                # Kiểm tra kích thước ảnh (tối thiểu 100x100 pixels)
                width, height = img.size
                if width < 100 or height < 100:
                    return False, f"Ảnh quá nhỏ ({width}x{height}). Tối thiểu 100x100 pixels", 0
                
                # Lưu lại ảnh với định dạng đúng nếu cần
                if not is_base64 and img.mode != 'RGB':
                    img.save(file_path, 'JPEG', quality=95)
                    
            except Exception as e:
                return False, f"Lỗi đọc ảnh: {str(e)}", 0
        
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

# Load danh sách đã điểm danh hôm nay từ Database
def load_today_recorded():
    global today_checked_in, today_checked_out, today_student_names
    today_checked_in = set()
    today_checked_out = set()
    today_student_names = {}
    
    try:
        attendance_data = db.get_today_attendance()
        for record in attendance_data:
            student_id = record.get('student_id')
            name = record.get('student_name') or record.get('full_name')
            if not student_id:
                continue
            today_student_names[student_id] = name or student_id
            if record.get('check_in_time'):
                today_checked_in.add(student_id)
            if record.get('check_out_time'):
                today_checked_out.add(student_id)
    except Exception as e:
        app.logger.error(f"Error loading today recorded: {e}")

# Lưu điểm danh vào Database
def mark_attendance(name: str, student_id: str = '', confidence_score: float = None) -> bool:
    """Lưu điểm danh vào database"""
    with today_recorded_lock:
        already_checked_in = student_id in today_checked_in
        already_checked_out = student_id in today_checked_out
        if already_checked_in and not already_checked_out:
            app.logger.info(f"Sinh vien {name} da check-in va chua checkout")
            return False
    
    success = db.mark_attendance(
        student_id=student_id,
        student_name=name,
        status='present',
        confidence_score=confidence_score,
        notes=None
    )
    
    if success:
        with today_recorded_lock:
            today_checked_in.add(student_id)
            today_checked_out.discard(student_id)
            today_student_names[student_id] = name
        # Khởi tạo presence tracking
        with presence_tracking_lock:
            presence_tracking[student_id] = {
                'last_seen': datetime.now(),
                'check_in_time': datetime.now(),
                'name': name
            }
        app.logger.info(f"Da danh dau diem danh: {name} (id={student_id}, confidence={confidence_score})")
        
        broadcast_sse_event({
            'type': 'attendance_marked',
            'data': {
                'event': 'check_in',
                'student_id': student_id,
                'student_name': name,
                'confidence': confidence_score,
                'timestamp': datetime.now().isoformat()
            }
        })
    
    return success


def mark_student_checkout(student_id: str, student_name: str = '', reason: str = 'manual', confidence_score: float = None) -> bool:
    """Đánh dấu checkout cho sinh viên"""
    with today_recorded_lock:
        already_checked_in = student_id in today_checked_in
        already_checked_out = student_id in today_checked_out
    
    if not already_checked_in or already_checked_out:
        return False
    
    success = db.mark_checkout(student_id)
    if not success:
        return False
    
    resolved_name = student_name or today_student_names.get(student_id) or student_id
    with today_recorded_lock:
        today_checked_out.add(student_id)
        today_student_names[student_id] = resolved_name
    
    with presence_tracking_lock:
        presence_tracking.pop(student_id, None)
    
    broadcast_sse_event({
        'type': 'attendance_checkout',
        'data': {
            'event': 'check_out',
            'student_id': student_id,
            'student_name': resolved_name,
            'confidence': confidence_score,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }
    })
    
    app.logger.info(f"Da checkout: {resolved_name} (id={student_id}) - reason={reason}")
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


# --- External attendance viewer (safe integration) ----------------------
@app.route('/external-attendance', methods=['GET'])
def external_attendance():
    """Render a read-only view of attendance CSVs from the attached project.
    This does NOT modify any data in the main project; it only reads CSV files
    from `external_projects/Cong-Nghe-Xu-Ly-Anh/attendance_logs` and renders
    them using `templates/external_index.html`.
    """
    all_data = []
    headers = ["ID", "Name", "Time", "Status", "SourceFile"]
    external_dir = Path('external_projects') / 'Cong-Nghe-Xu-Ly-Anh' / 'attendance_logs'

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

    # Local filtering (same behavior as the external project template expects)
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
def get_today_attendance():
    """Lấy danh sách điểm danh hôm nay từ database"""
    try:
        attendance_data = db.get_today_attendance()
        # Convert SQLite Row objects to dict
        results = []
        for row in attendance_data:
            # Tính thời gian có mặt
            duration_minutes = None
            status_text = "Đang có mặt"
            
            if row['check_out_time']:
                # Đã checkout
                check_in = datetime.fromisoformat(row['check_in_time'])
                check_out = datetime.fromisoformat(row['check_out_time'])
                duration_seconds = (check_out - check_in).total_seconds()
                duration_minutes = int(duration_seconds / 60)
                status_text = "Đã rời"
            else:
                # Chưa checkout - tính từ check_in đến hiện tại
                check_in = datetime.fromisoformat(row['check_in_time'])
                duration_seconds = (datetime.now() - check_in).total_seconds()
                duration_minutes = int(duration_seconds / 60)
                
                # Kiểm tra xem có đang được tracking không
                with presence_tracking_lock:
                    if row['student_id'] not in presence_tracking:
                        status_text = "Không còn phát hiện"
            
            results.append({
                'student_id': row['student_id'],
                'full_name': row['student_name'],
                'timestamp': row['check_in_time'],
                'checkout_time': row['check_out_time'],
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
def generate_frames():
    global video_capture, camera_enabled
    
    app.logger.info("generate_frames() started")
    
    # Nếu camera bị tắt, yield placeholder và KHÔNG khởi tạo camera
    if not camera_enabled:
        placeholder = make_placeholder_frame("Camera đã tắt")
        if placeholder is None:
            return
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n')
        return

    # Khởi tạo camera nếu chưa có
    if video_capture is None or not getattr(video_capture, 'isOpened', lambda: False)():
        ensure_video_capture()

    # Nếu camera không thể mở sau khi khởi tạo, yield hình ảnh placeholder liên tục
    if video_capture is None or not getattr(video_capture, 'isOpened', lambda: False)():
        app.logger.error("Khong the mo video capture - phuc vu khung hinh placeholder")
        placeholder = make_placeholder_frame()
        if placeholder is None:
            return
        # yield placeholder liên tục để <img> hiển thị gì đó
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n')
        return

    frame_count = 0
    detection_frame_counter = YOLO_FRAME_SKIP  # ép chạy YOLO ngay frame đầu tiên
    cached_face_data = []

    while True:
        # Kiểm tra xem camera có bị tắt không
        if not camera_enabled:
            app.logger.info("Camera da tat, dung stream")
            break
            
        try:
            # Kiểm tra video_capture trước khi đọc
            if video_capture is None or not getattr(video_capture, 'isOpened', lambda: False)():
                app.logger.warning("[Camera] ⚠️ Video capture bị mất kết nối, thử khởi tạo lại...")
                if camera_enabled:  # Chỉ khởi tạo lại nếu camera đang bật
                    ensure_video_capture()
                if video_capture is None:
                    time.sleep(0.1)
                    continue
            
            ret, frame = video_capture.read()
            if not ret or frame is None:
                app.logger.debug("[Camera] ⚠️ Không đọc được frame (ret=False hoặc frame=None)")
                continue
            
            frame_count += 1
            if frame_count % 30 == 0:  # Log mỗi 30 frames (khoảng 1 giây nếu 30fps)
                app.logger.debug(f"[Camera] 📹 Đang đọc frame #{frame_count}...")
                
        except Exception as e:
            app.logger.error(f"[Camera] ❌ Lỗi đọc frame: {e}", exc_info=True)
            # Thử khởi tạo lại camera chỉ khi camera đang bật
            if camera_enabled:
                ensure_video_capture()
            time.sleep(0.1)
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
        
        if not DEMO_MODE and DEEPFACE_AVAILABLE and YOLO_AVAILABLE and yolo_face_model:
            global known_face_embeddings, known_face_ids, known_face_names
            # Kiểm tra xem đã build DB embeddings chưa
            if known_face_embeddings is None or len(known_face_embeddings) == 0:
                app.logger.warning("[System] Chưa có DeepFace DB, đang build từ thư mục data/...")
                embeddings, labels = build_db_from_data_dir(str(DATA_DIR))
                if embeddings is not None and len(embeddings) > 0:
                    import numpy as _np
                    # labels: list[(student_id, name)]
                    known_face_embeddings = _np.array(embeddings, dtype="float32")
                    known_face_ids = [sid for sid, _ in labels]
                    known_face_names = [name for _, name in labels]
                    app.logger.info(
                        "[System] ✅ Đã build DeepFace DB: %d embeddings, %d IDs",
                        known_face_embeddings.shape[0],
                        len(set(known_face_ids)),
                    )
                else:
                    app.logger.warning(
                        "[System] ⚠️ Không build được DB từ %s, bỏ qua nhận diện.", DATA_DIR
                    )
            
            if known_face_embeddings is not None and len(known_face_embeddings) > 0:
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

                        student_id = "UNKNOWN"
                        name = "UNKNOWN"
                        best_score = 0.0
                        try:
                            rep = DeepFace.represent(
                                face_img,
                                model_name="Facenet512",
                                enforce_detection=False,
                            )[0]["embedding"]

                            db_labels = list(zip(known_face_ids, known_face_names))
                            student_id, name, best_score = deepface_recognize(
                                rep,
                                known_face_embeddings,
                                db_labels,
                                threshold=DEEPFACE_SIMILARITY_THRESHOLD,
                            )

                            if student_id is None or name is None:
                                student_id = "UNKNOWN"
                                name = "UNKNOWN"
                        except Exception as e:
                            app.logger.error(f"[System] Lỗi nhận diện DeepFace: {e}", exc_info=True)

                        status = 'unknown'
                        confidence_score = float(best_score or 0.0)
                        now = datetime.now()

                        if student_id != "UNKNOWN":
                            checked_in = student_id in today_checked_in
                            checked_out = student_id in today_checked_out
                            with last_recognized_lock:
                                last_time = last_recognized.get(student_id)
                                cooldown_passed = not last_time or (now - last_time).total_seconds() > RECOGNITION_COOLDOWN

                            if not checked_in and cooldown_passed:
                                try:
                                    success = mark_attendance(name, student_id=student_id, confidence_score=confidence_score)
                                    if success:
                                        status = 'checked_in'
                                        with last_recognized_lock:
                                            last_recognized[student_id] = now
                                        app.logger.info(f"[+] {student_id} - {name} điểm danh lúc {now.strftime('%Y-%m-%d %H:%M:%S')}")
                                except Exception as e:
                                    status = 'error'
                                    app.logger.error(f"[System] Lỗi điểm danh: {e}")
                            elif checked_in and not checked_out and cooldown_passed:
                                if mark_student_checkout(student_id, student_name=name, reason='auto', confidence_score=confidence_score):
                                    status = 'checked_out'
                                    with last_recognized_lock:
                                        last_recognized[student_id] = now
                                else:
                                    status = 'already_marked'
                            elif checked_out:
                                status = 'checked_out'
                            else:
                                status = 'cooldown' if not cooldown_passed else 'already_marked'

                        new_face_data.append({
                            'bbox': (xmin, ymin, xmax, ymax),
                            'name': name,
                            'student_id': student_id,
                            'confidence': confidence_score,
                            'status': status
                        })

                    cached_face_data = new_face_data
                    face_data = new_face_data
                else:
                    face_data = cached_face_data or []
            else:
                cached_face_data = []
                app.logger.warning("[System] Không có embeddings để nhận diện. Vui lòng thêm ảnh vào thư mục data/")
        
        # Demo mode hoặc không có YOLOv8/DeepFace
        elif DEMO_MODE or not DEEPFACE_AVAILABLE or not YOLO_AVAILABLE:
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
        
        # Draw bounding boxes và labels (chỉ cho demo mode)
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
            
            # Draw bounding box with thicker lines
            cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)
            
            # Draw label with name and confidence
            if status == 'already_marked':
                label = f"{name} - Da diem danh"
            elif status == 'confirming':
                label = f"{name} - Dang xac nhan..."
            elif status == 'confirmed' or status == 'checked_in':
                label = f"{name} - THANH CONG!"
            elif status == 'checked_out':
                label = f"{name} - Da ra ve"
            elif name == "Unknown":
                label = "Unknown - Chua dang ky"
            elif status == 'low_confidence':
                label = f"{name} (Confidence thap: {confidence*100:.1f}%)"
            elif status == 'cooldown':
                label = f"{name} - Vua diem danh (cho {RECOGNITION_COOLDOWN}s)"
            elif confidence > 0:
                label = f"{name} ({confidence*100:.1f}%)"
            else:
                label = name
            
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
            label_x = left
            label_y = top - 10 if top > 30 else bottom + 30
            
            # Draw label background (semi-transparent effect with padding)
            padding = 5
            cv2.rectangle(frame, 
                         (label_x - padding, label_y - label_size[1] - padding), 
                         (label_x + label_size[0] + padding, label_y + padding), 
                         color, -1)
            
            # Draw label text in black for better contrast
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

        # Encode frame with reduced quality to lower CPU and bandwidth
        ret2, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ret2:
            continue
        frame_bytes = buf.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    # release
    with video_lock:
        cap = video_capture
        if cap is not None:
            cap.release()

# Routes
@app.route('/')
def index():
    """Trang chính - điểm danh"""
    attendance_data = get_today_attendance()
    checked_in = [row for row in attendance_data if not row.get('checkout_time')]
    checked_out = [row for row in attendance_data if row.get('checkout_time')]
    return render_template('index.html', attendance=attendance_data,
                           checked_in=checked_in, checked_out=checked_out)

@app.route('/video_feed')
def video_feed():
    """Video feed cho camera"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/camera/toggle', methods=['POST'])
def toggle_camera():
    """API bật/tắt camera"""
    global camera_enabled, video_capture
    try:
        # Toggle camera enabled state (bật/tắt)
        camera_enabled = not camera_enabled
        
        if not camera_enabled:
            # Tắt camera - giải phóng hoàn toàn
            app.logger.info("Turning OFF camera - releasing video capture")
            with video_lock:
                if video_capture is not None:
                    video_capture.release()
                    video_capture = None
            time.sleep(0.5)  # Đợi camera giải phóng hoàn toàn
        else:
            # Bật camera
            app.logger.info("Turning ON camera - initializing video capture")
            time.sleep(0.5)  # Đợi trước khi khởi tạo
            ensure_video_capture()
        
        return jsonify({'success': True, 'enabled': camera_enabled})
        
    except Exception as e:
        app.logger.error(f"Error toggling camera: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/camera/status', methods=['GET'])
def camera_status():
    """API kiểm tra trạng thái camera"""
    return jsonify({
        'enabled': camera_enabled,
        'opened': video_capture is not None and getattr(video_capture, 'isOpened', lambda: False)()
    })

@app.route('/api/camera/capture', methods=['POST'])
def capture_image():
    """API chụp ảnh từ camera"""
    global video_capture
    
    try:
        if video_capture is None or not getattr(video_capture, 'isOpened', lambda: False)():
            return jsonify({'error': 'Camera không khả dụng'}), 400
        
        ret, frame = video_capture.read()
        if not ret or frame is None:
            return jsonify({'error': 'Không thể đọc frame từ camera'}), 400
        
        # Flip frame horizontally (mirror effect)
        frame = cv2.flip(frame, 1)
        
        # Encode frame to base64 with reduced quality to save CPU/bandwidth
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

# Quick face registration API
@app.route('/api/quick-register', methods=['POST'])
def api_quick_register():
    """API đăng ký nhanh khuôn mặt"""
    try:
        data = request.form
        student_id = data.get('student_id', '').strip()
        full_name = data.get('full_name', '').strip()
        
        # Debug logging
        app.logger.info(f"Quick register request - ID: {student_id}, Name: {full_name}")
        app.logger.info(f"Form keys: {list(data.keys())}")
        app.logger.info(f"Files keys: {list(request.files.keys())}")
        
        if not all([student_id, full_name]):
            return jsonify({'error': 'Mã sinh viên và họ tên là bắt buộc'}), 400
        
        # Handle webcam capture or file upload
        face_image = None
        is_base64 = False
        
        # Check webcam capture first (has priority)
        if 'image_data' in request.form and request.form['image_data']:
            # Validate base64 image trước
            image_data = request.form['image_data']
            success, error_msg, face_count = validate_image_file(image_data, is_base64=True)
            
            if not success:
                app.logger.error(f"Image validation failed: {error_msg}")
                return jsonify({'error': f'Ảnh không hợp lệ: {error_msg}'}), 400
            
            # Handle base64 image from webcam
            app.logger.info(f"Image data length: {len(image_data)}")
            # Remove data:image/jpeg;base64, prefix
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            img_bytes = base64.b64decode(image_data)
            filename = f"{student_id}_{full_name}.jpg"
            file_path = DATA_DIR / filename
            DATA_DIR.mkdir(exist_ok=True)
            
            # Lưu ảnh đã validate
            with open(file_path, 'wb') as f:
                f.write(img_bytes)
            face_image = str(file_path)
            is_base64 = True
            app.logger.info(f"Saved webcam capture: {face_image} (faces: {face_count})")
            
        elif 'face_image' in request.files:
            file = request.files['face_image']
            if file and file.filename:
                # Kiểm tra extension
                ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
                if ext not in ALLOWED_EXTENSIONS:
                    return jsonify({'error': f'Định dạng file không hợp lệ. Chỉ chấp nhận: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
                
                # Save uploaded file tạm
                filename = secure_filename(f"{student_id}_{full_name}.jpg")
                file_path = DATA_DIR / filename
                DATA_DIR.mkdir(exist_ok=True)
                file.save(str(file_path))
                
                # Validate file đã lưu
                success, error_msg, face_count = validate_image_file(str(file_path), is_base64=False)
                
                if not success:
                    # Xóa file không hợp lệ
                    try:
                        os.remove(str(file_path))
                    except:
                        pass
                    app.logger.error(f"Image validation failed: {error_msg}")
                    return jsonify({'error': f'Ảnh không hợp lệ: {error_msg}'}), 400
                
                face_image = str(file_path)
                app.logger.info(f"Saved uploaded file: {face_image} (faces: {face_count})")
        
        if not face_image:
            app.logger.error("No face image provided")
            return jsonify({'error': 'Vui lòng chụp ảnh hoặc upload ảnh khuôn mặt'}), 400
        
        # Ảnh đã được validate, không cần kiểm tra lại
        # Face encoding sẽ được tạo khi load_known_faces()
        app.logger.info(f"Image validated successfully: {face_image}")
        
        # Add to database with face image path
        email = data.get('email', f'{student_id}@student.edu.vn')
        phone = data.get('phone', '')
        class_name = data.get('class_name', 'Chưa phân lớp')
        
        db.add_student(student_id, full_name, email, phone, class_name, face_image)
        
        # Reload known faces
        load_known_faces()
        
        return jsonify({'success': True, 'message': f'Đăng ký thành công cho {full_name}!'})
        
    except Exception as e:
        app.logger.error(f"Quick registration error: {e}")
        return jsonify({'error': f'Lỗi: {str(e)}'}), 500

# Page routes
@app.route('/students')
def students_page():
    """Trang quản lý sinh viên"""
    return render_template('students.html')

@app.route('/test-students')
def test_students_page():
    """Trang test API students"""
    # The dedicated test template was removed during cleanup; reuse the
    # students management page instead so the route remains functional.
    return render_template('students.html')

@app.route('/reports')
def reports_page():
    """Trang báo cáo"""
    return render_template('reports.html')

@app.route('/classes')
def classes_page():
    """Trang quản lý lớp học"""
    return render_template('classes.html')

@app.route('/api/students', methods=['GET'])
def api_get_students():
    """API lấy danh sách sinh viên"""
    try:
        students = db.get_all_students(active_only=True)
        students_list = []
        for student in students:
            # Lấy thông tin lớp học nếu có
            class_name = None
            if student['class_id']:
                class_info = db.get_class_by_id(student['class_id'])
                if class_info:
                    class_name = class_info.get('class_name')
            
            students_list.append({
                'id': student['id'],
                'student_id': student['student_id'],
                'full_name': student['full_name'],
                'email': student['email'],
                'phone': student['phone'],
                'class_id': student['class_id'],
                'class_name': class_name,
                'face_image_path': student['face_image_path'],
                'status': student['status'],
                'is_active': student['is_active'],
                'created_at': student['created_at']
            })
        return jsonify({'success': True, 'data': students_list})
    except Exception as e:
        app.logger.error(f"Error getting students: {e}")
        return jsonify({'success': False, 'data': [], 'error': str(e)}), 500

@app.route('/api/statistics')
def api_statistics():
    """API thống kê"""
    try:
        attendance_data = get_today_attendance()
        total_students = len(known_face_names) if known_face_names else 0
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
        attendance_data = get_today_attendance()
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
            'checked_out': checked_out
        })
    except Exception as e:
        app.logger.error(f"Error getting today's attendance: {e}")
        return jsonify({'success': False, 'data': [], 'error': str(e)}), 500


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
            recent_att = db.get_today_attendance()
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

# ===== ADVANCED AI TRAINING ROUTES =====

@app.route('/api/train/start', methods=['POST'])
def api_train_start():
    """Bắt đầu training classifier với FaceNet embeddings"""
    if not USE_FACENET or face_service is None:
        return jsonify({'error': 'FaceNet service not available'}), 400
    
    try:
        # Initialize training service if not yet
        global training_service
        if training_service is None:
            from services.training_service import TrainingService
            training_service = TrainingService(face_service)
        
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
                'error': 'Training failed - insufficient data'
            }), 400
    
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
        # Get image from request (base64 or file upload)
        if 'image_data' in request.form:
            # Base64 image
            image_data = request.form['image_data']
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            img_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        elif 'image' in request.files:
            # File upload
            file = request.files['image']
            img_bytes = file.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        # Check for spoofing
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


# Helper functions
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


# ===== HELPER FUNCTIONS FROM FACENET =====

def prewhiten_facenet(x):
    """
    FaceNet-style prewhitening for better normalization.
    Adapted from face_attendance/facenet.py
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
    Estimate simple head pose (yaw, pitch, roll) in degrees using solvePnP.
    landmarks: dictionary or list of (x,y) points for key landmarks (we expect at least
    left_eye, right_eye, nose, left_mouth, right_mouth) or list in the order returned
    by dlib/face_recognition: we will try to handle common formats.
    Returns (yaw_deg, pitch_deg, roll_deg) or (None, None, None) on failure.
    """
    try:
        # Convert landmarks to required numpy array of 2D points
        lm = None
        if isinstance(landmarks, dict):
            # face_recognition returns 'left_eye','right_eye','nose_tip','mouth_left','mouth_right' sometimes
            keys = ['left_eye', 'right_eye', 'nose_tip', 'mouth_left', 'mouth_right']
            pts2 = []
            for k in keys:
                if k in landmarks:
                    pts2.append(landmarks[k])
            if len(pts2) < 4:
                # fallback: use all dict values
                pts2 = list(landmarks.values())
        else:
            pts2 = list(landmarks)

        if len(pts2) < 4:
            return (None, None, None)

        # Pick 4-5 stable points: left eye, right eye, nose, left mouth corner, right mouth corner
        # Use a generic 3D model points (approximate)
        model_points = np.array([ 
            ( -30.0,  30.0,  -30.0),   # left eye
            (  30.0,  30.0,  -30.0),   # right eye
            (   0.0,   0.0,    0.0),   # nose tip
            ( -25.0, -30.0,  -25.0),   # left mouth
            (  25.0, -30.0,  -25.0)    # right mouth
        ], dtype=np.float64)

        # Map 2D image points from landmarks (take first 5)
        image_points = []
        for i in range(min(len(pts2), 5)):
            p = pts2[i]
            image_points.append((float(p[0]), float(p[1])))
        image_points = np.array(image_points, dtype=np.float64)

        # If we have fewer than model points, reduce model points to match
        if image_points.shape[0] < model_points.shape[0]:
            model_points = model_points[:image_points.shape[0]]

        # Camera internals (approximate)
        size = frame_size
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4,1))  # assume no lens distortion

        # solvePnP
        success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            return (None, None, None)

        # Convert rotation vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rotation_vector)
        # Compose projection matrix then decompose to Euler angles
        pose_mat = cv2.hconcat((rmat, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

        # euler_angles: [pitch, yaw, roll] in degrees (OpenCV ordering)
        pitch, yaw, roll = float(euler_angles[0]), float(euler_angles[1]), float(euler_angles[2])
        return (yaw, pitch, roll)
    except Exception as e:
        app.logger.debug(f"estimate_head_pose failed: {e}")
        return (None, None, None)



def draw_progress_bar(frame, progress, x, y, w=150, h=20):
    """
    Draw progress bar for attendance confirmation.
    Adapted from face_attendance/reg.py
    
    Args:
        frame: Video frame
        progress: Progress value (0.0 to 1.0)
        x, y: Top-left coordinates
        w, h: Width and height of bar
    """
    bar_y = y - 30  # Above the face box
    
    # Background (black)
    cv2.rectangle(frame, (x, bar_y), (x + w, bar_y + h), (0, 0, 0), -1)
    
    # Progress (green)
    filled_width = int(w * progress)
    if filled_width > 0:
        cv2.rectangle(frame, (x, bar_y), (x + filled_width, bar_y + h), (0, 255, 0), -1)
    
    # Border
    cv2.rectangle(frame, (x, bar_y), (x + w, bar_y + h), (255, 255, 255), 1)
    
    # Percentage text
    progress_text = f"{int(progress * 100)}%"
    text_size = cv2.getTextSize(progress_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
    text_x = x + (w - text_size[0]) // 2
    text_y = bar_y + (h + text_size[1]) // 2
    cv2.putText(frame, progress_text, (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)


def update_progress(student_id, name):
    """
    Update attendance confirmation progress.
    New behavior: require continuous frontal-looking time (LOOK_STRAIGHT_SECONDS) to confirm.
    Returns: (elapsed_seconds, required_seconds, is_confirmed)
    """
    global attendance_progress
    now = datetime.now()
    with attendance_progress_lock:
        entry = attendance_progress.get(student_id)
        if entry is None:
            # Start new frontal-looking window
            attendance_progress[student_id] = {
                'start_time': now,
                'last_seen': now,
                'name': name
            }
            elapsed = 0.0
        else:
            # Continue window
            # If there was a long gap since last seen, restart window
            last = entry.get('last_seen')
            gap = (now - last).total_seconds() if last else 9999
            if gap > 1.5:  # if missing for >1.5s, reset the frontal timer
                attendance_progress[student_id] = {
                    'start_time': now,
                    'last_seen': now,
                    'name': name
                }
                elapsed = 0.0
            else:
                # Update last seen and compute elapsed continuous frontal seconds
                entry['last_seen'] = now
                elapsed = (now - entry['start_time']).total_seconds()

        is_confirmed = elapsed >= LOOK_STRAIGHT_SECONDS
        return elapsed, LOOK_STRAIGHT_SECONDS, is_confirmed


def reset_progress(student_id):
    """Reset progress for a student."""
    global attendance_progress
    
    with attendance_progress_lock:
        if student_id in attendance_progress:
            del attendance_progress[student_id]


# Initialize
if __name__ == '__main__':
    try:
        # Initialize database
        db.init_database()
        
        # Ensure directories exist
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load known faces
        load_known_faces()
        
        # Load today's recorded set from database
        load_today_recorded()
        
        # Log system startup
        db.log_system_event('INFO', 'He thong diem danh khoi dong', 'app')
        app.logger.info("He thong diem danh da khoi dong thanh cong")
        
        app.run(host='0.0.0.0', port=5000, debug=True)
        
    except Exception as e:
        app.logger.error(f"Loi khoi dong he thong: {e}")

