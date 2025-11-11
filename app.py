# Set UTF-8 encoding for console output
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Flask imports
from flask import Flask, render_template, Response, redirect, url_for, request, jsonify

# Standard library imports
import os
import time
import random
import base64
from pathlib import Path
from datetime import datetime, date
import threading
from threading import Thread, Lock

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

# Import face_recognition với fallback cho demo mode
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError as e:
    FACE_RECOGNITION_AVAILABLE = False
    if not DEMO_MODE:
        # Nếu không có face_recognition và không phải demo mode, tự động chuyển sang demo mode
        print("Face recognition not available, switching to demo mode...")
        DEMO_MODE = True
        print("Demo mode: face_recognition not available, using simulation mode")
    else:
        print("Demo mode: face_recognition not available, using simulation mode")

# Chỉ số thiết bị camera (sử dụng os.getenv sau khi load_dotenv)
CAMERA_INDEX = int(os.getenv('CAMERA_INDEX', '0'))

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
today_recorded = set()
today_recorded_lock = threading.Lock()

# Theo dõi thời gian có mặt
presence_tracking = {}  # {student_id: {'last_seen': datetime, 'total_time': seconds}}
presence_tracking_lock = threading.Lock()
PRESENCE_TIMEOUT = 300  # 5 phút (300 giây) - nếu không thấy sẽ tự checkout

# Khởi tạo camera đơn giản nhất có thể
def ensure_video_capture():
    global video_capture
    if video_capture is not None and getattr(video_capture, 'isOpened', lambda: False)():
        return
    
    # Khởi tạo camera đơn giản nhất có thể
    try:
        video_capture = cv2.VideoCapture(CAMERA_INDEX)
        # Set lower resolution by default to reduce CPU and network usage
        try:
            CAMERA_WIDTH = int(os.getenv('CAMERA_WIDTH', '640'))
            CAMERA_HEIGHT = int(os.getenv('CAMERA_HEIGHT', '480'))
            video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            # Try to set a small buffer if supported
            if hasattr(cv2, 'CAP_PROP_BUFFERSIZE'):
                try:
                    video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                except Exception:
                    pass
        except Exception:
            pass

        # Warm-up: read a few frames to clear initial buffer
        for _ in range(3):
            try:
                ret, _ = video_capture.read()
                if not ret:
                    break
            except Exception:
                break

        app.logger.info("Camera khoi tao don gian (with resolution)")
    except Exception as e:
        app.logger.error(f"Loi khoi tao camera: {e}")
        video_capture = None

# Tải khuôn mặt đã biết từ DATA_DIR
def load_known_faces():
    global known_face_encodings, known_face_names, known_face_ids
    known_face_encodings = []
    known_face_names = []
    known_face_ids = []

    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Trong DEMO_MODE, vẫn load danh sách tên từ files để hiển thị
    if not FACE_RECOGNITION_AVAILABLE:
        app.logger.warning("Face recognition khong kha dung, chi tai danh sach ten (DEMO_MODE)")
        # Load danh sách tên từ files trong DATA_DIR
        for img_path in DATA_DIR.glob('*.jpg'):
            try:
                parts = img_path.stem.split('_')
                if len(parts) >= 2:
                    student_id = parts[0]
                    display_name = '_'.join(parts[1:])
                else:
                    student_id = img_path.stem
                    display_name = img_path.stem
                
                known_face_names.append(display_name)
                known_face_ids.append(student_id)
                app.logger.info(f"[DEMO] Da tai thong tin cho {display_name} (id={student_id})")
            except Exception as e:
                app.logger.error(f"Loi khi tai {img_path}: {e}")
        return

    for img_path in DATA_DIR.glob('*.jpg'):
        try:
            # Parse student info from filename
            parts = img_path.stem.split('_')
            if len(parts) >= 2:
                student_id = parts[0]
                display_name = '_'.join(parts[1:])
            else:
                student_id = img_path.stem
                display_name = img_path.stem

            img = face_recognition.load_image_file(str(img_path))
            encs = face_recognition.face_encodings(img)
            if encs:
                known_face_encodings.append(encs[0])
                known_face_names.append(display_name)
                known_face_ids.append(student_id)
                app.logger.info(f"Da tai khuon mat cho {display_name} (id={student_id}) tu {img_path.name}")
            else:
                app.logger.warning(f"Khong tim thay khuon mat trong {img_path}")
        except Exception as e:
            app.logger.error(f"Loi khi tai {img_path}: {e}")

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
    global today_recorded
    today_recorded = set()
    
    try:
        # Lấy danh sách điểm danh hôm nay từ database
        attendance_data = db.get_today_attendance()
        for record in attendance_data:
            today_recorded.add(record['student_name'])
    except Exception as e:
        app.logger.error(f"Error loading today recorded: {e}")

# Lưu điểm danh vào Database
def mark_attendance(name: str, student_id: str = '', confidence_score: float = None) -> bool:
    """Lưu điểm danh vào database"""
    with today_recorded_lock:
        # Kiểm tra xem đã điểm danh chưa
        if name in today_recorded:
            app.logger.info(f"Sinh vien {name} da diem danh roi")
            return False
        
        # Lưu vào database
        success = db.mark_attendance(
            student_id=student_id,
            student_name=name,
            status='present',
            confidence_score=confidence_score,
            notes=None
        )
        
        if success:
            today_recorded.add(name)
            # Khởi tạo presence tracking
            with presence_tracking_lock:
                presence_tracking[student_id] = {
                    'last_seen': datetime.now(),
                    'check_in_time': datetime.now(),
                    'name': name
                }
            app.logger.info(f"Da danh dau diem danh: {name} (id={student_id}, confidence={confidence_score})")
        
        return success

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
            if name in today_recorded:
                presence_tracking[student_id] = {
                    'last_seen': now,
                    'check_in_time': now,
                    'name': name
                }

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
            db.mark_checkout(student_id)
            app.logger.info(f"Auto checkout {student_name} ({student_id}) - timeout {PRESENCE_TIMEOUT}s")
            # Xóa khỏi tracking nhưng giữ trong today_recorded
            del presence_tracking[student_id]

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

    while True:
        # Kiểm tra xem camera có bị tắt không
        if not camera_enabled:
            app.logger.info("Camera da tat, dung stream")
            break
            
        try:
            # Kiểm tra video_capture trước khi đọc
            if video_capture is None or not getattr(video_capture, 'isOpened', lambda: False)():
                app.logger.warning("Video capture bi mat ket noi, thu khoi tao lai...")
                if camera_enabled:  # Chỉ khởi tạo lại nếu camera đang bật
                    ensure_video_capture()
                if video_capture is None:
                    time.sleep(0.1)
                    continue
            
            ret, frame = video_capture.read()
            if not ret or frame is None:
                continue
        except Exception as e:
            app.logger.error(f"Loi doc frame: {e}")
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

        # Demo mode: tạo bounding box mô phỏng
        if DEMO_MODE:
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
        else:
            # Real face recognition mode
            face_data = []
            if FACE_RECOGNITION_AVAILABLE and known_face_encodings:
                # Face detection và recognition
                face_locations = face_recognition.face_locations(frame)
                face_encodings = face_recognition.face_encodings(frame, face_locations)
                
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"
                    student_id = ""
                    confidence_score = 0.0
                    
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = known_face_names[first_match_index]
                        student_id = known_face_ids[first_match_index]
                        
                        # Tính confidence score
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        confidence_score = 1 - face_distances[first_match_index]
                        
                        # Điểm danh thực tế
                        if frame_count % 30 == 0:  # Mỗi 30 frames
                            try:
                                mark_attendance(name, student_id=student_id, confidence_score=confidence_score)
                                # Cập nhật presence
                                update_presence(student_id, name)
                            except Exception as e:
                                app.logger.error(f"Loi xac nhan diem danh cho {name}: {e}")
                        # Cập nhật presence mỗi 60 frames (2 giây) cho sinh viên đã điểm danh
                        elif frame_count % 60 == 0 and confidence_score > 0.6:
                            try:
                                update_presence(student_id, name)
                            except Exception as e:
                                app.logger.error(f"Loi cap nhat presence cho {name}: {e}")
                    
                    # Lưu thông tin khuôn mặt (tọa độ đã đúng vì frame đã flip)
                    face_info = {
                        'bbox': (left, top, right, bottom),
                        'name': name,
                        'confidence': confidence_score,
                        'student_id': student_id,
                        'status': 'confirmed' if confidence_score > 0.6 else 'detected'
                    }
                    face_data.append(face_info)
        
        # Draw green bounding boxes around detected faces
        for face_info in face_data:
            left, top, right, bottom = face_info['bbox']
            name = face_info.get('name', 'Unknown')
            confidence = face_info.get('confidence', 0.0)
            status = face_info.get('status', 'detected')
            
            # Chọn màu dựa trên trạng thái
            if status == 'waiting':
                color = (255, 165, 0)  # Màu cam cho demo mode (đang chờ)
                thickness = 2
            elif confidence > 0.6:
                color = (0, 255, 0)  # Màu xanh lá cho nhận diện thành công
                thickness = 3
            else:
                color = (0, 165, 255)  # Màu cam cho nhận diện chưa chắc chắn
                thickness = 2
            
            # Draw bounding box with thicker lines
            cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)
            
            # Draw label with name and confidence
            if confidence > 0:
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
    return render_template('index.html', attendance=attendance_data)

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
    return render_template('test_students.html')

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
        return jsonify({'success': True, 'data': attendance_data})
    except Exception as e:
        app.logger.error(f"Error getting today's attendance: {e}")
        return jsonify({'success': False, 'data': [], 'error': str(e)}), 500


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

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

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

