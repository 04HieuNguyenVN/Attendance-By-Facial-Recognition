# Set UTF-8 encoding for console output
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from flask import Flask, render_template, Response, redirect, url_for, request, jsonify, send_file, session
import cv2
import numpy as np
from pathlib import Path
import csv
from datetime import datetime, date
import threading
import pickle
import os
import time
import hashlib
import json
from werkzeug.utils import secure_filename
from functools import wraps
from database import db
from logging_config import setup_logging

# Dummy classes để thay thế optimization modules
class DummyOptimizer:
    def __init__(self, *args, **kwargs): 
        pass
    def __getattr__(self, name): 
        return lambda *args, **kwargs: None

# Initialize dummy optimizers
camera_optimizer = DummyOptimizer()
face_display_enhancer = DummyOptimizer()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ma-khoa-bi-mat-cua-ban'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Kích thước file tối đa 16MB

# Setup logging
setup_logging()

# Cấu hình chế độ demo
DEMO_MODE = os.environ.get('DEMO_MODE', '0') == '1'

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

# Chỉ số thiết bị camera (có thể ghi đè qua biến môi trường CAMERA_INDEX)
CAMERA_INDEX = int(os.environ.get('CAMERA_INDEX', '0'))

# Cấu hình xác nhận phát hiện
CONFIRM_SECONDS = 3  # Số giây để xác nhận phát hiện
PRESENCE_MAX_GAP = 5  # Khoảng cách tối đa giữa các lần phát hiện (giây)

# Đường dẫn thư mục dữ liệu
DATA_DIR = Path('data')
KNOWN_FACES_PKL = Path('known_faces.pkl')
ATTENDANCE_CSV = Path('attendance.csv')

# Video capture và khóa
video_capture = None
video_lock = threading.Lock()
camera_enabled = True  # Biến để bật/tắt camera

# Khởi tạo camera đơn giản nhất có thể
def ensure_video_capture():
    global video_capture
    if video_capture is not None and getattr(video_capture, 'isOpened', lambda: False)():
        return
    
    # Khởi tạo camera đơn giản nhất có thể
    try:
        video_capture = cv2.VideoCapture(CAMERA_INDEX)
        app.logger.info("Camera khoi tao don gian")
    except Exception as e:
        app.logger.error(f"Loi khoi tao camera: {e}")
        video_capture = None

# Tải khuôn mặt đã biết từ DATA_DIR
def load_known_faces():
    global known_face_encodings, known_face_names, known_face_ids
    known_face_encodings = []
    known_face_names = []
    known_face_ids = []

    if not FACE_RECOGNITION_AVAILABLE:
        app.logger.warning("Face recognition khong kha dung, bo qua tai khuon mat")
        return

    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
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

# Đảm bảo file CSV điểm danh tồn tại với header
def ensure_attendance_csv():
    # Sử dụng header chuẩn mới: name,timestamp,date
    header = ['name', 'timestamp', 'date']
    if not ATTENDANCE_CSV.exists():
        with ATTENDANCE_CSV.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)

# Load danh sách đã điểm danh hôm nay từ CSV
def load_today_recorded():
    global today_recorded
    today_recorded = set()
    today_str = date.today().isoformat()
    
    if not ATTENDANCE_CSV.exists():
        return
    
    try:
        with ATTENDANCE_CSV.open('r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Xác định các trường một cách robust
                name = row.get('name') or row.get('Name') or ''
                timestamp = row.get('timestamp') or row.get('Timestamp') or ''
                # ưu tiên cột 'date' rõ ràng nếu có
                date_col = row.get('date') or row.get('Date') or ''
                if not date_col and timestamp:
                    date_col = timestamp.split('T')[0]
                if date_col == today_str:
                    today_recorded.add(name)
    except Exception as e:
        app.logger.error(f"Error loading today recorded: {e}")

# Lưu điểm danh vào CSV với format chuẩn
def save_attendance(name: str) -> bool:
    timestamp = datetime.now().isoformat()
    today_str = date.today().isoformat()
    
    with ATTENDANCE_CSV.open('a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([name, timestamp, today_str])
    with today_recorded_lock:
        today_recorded.add(name)
    app.logger.info(f"Da luu diem danh cho {name} luc {timestamp}")
    return True

# Refactor: mark_attendance để sử dụng save_attendance cho format CSV nhất quán
def mark_attendance(name: str, student_id: str = '') -> bool:
    # giữ logging, nhưng ủy quyền lưu trữ cho save_attendance
    saved = save_attendance(name)
    if not saved:
        return False
    # Tùy chọn, bạn cũng có thể lưu student_id ở nơi khác hoặc mở rộng CSV; hiện tại chúng ta log nó
    app.logger.info(f"Da danh dau diem danh (qua mark_attendance): {name} (id={student_id})")
    return True

# Đọc điểm danh hôm nay từ CSV (robust với format cũ/mới)
def get_today_attendance():
    results = []
    if not ATTENDANCE_CSV.exists():
        return results
    today = date.today().isoformat()
    with ATTENDANCE_CSV.open('r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Xác định các trường một cách robust
            name = row.get('name') or row.get('Name') or ''
            timestamp = row.get('timestamp') or row.get('Timestamp') or ''
            # ưu tiên cột 'date' rõ ràng nếu có
            date_col = row.get('date') or row.get('Date') or ''
            if not date_col and timestamp:
                date_col = timestamp.split('T')[0]
            if date_col == today:
                results.append({
                    'name': name,
                    'timestamp': timestamp,
                    'date': date_col
                })
    return results

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
    # mã hóa thành jpeg
    ret, buf = cv2.imencode('.jpg', img)
    if not ret:
        return None
    return buf.tobytes()

# Generator khung hình video
def generate_frames():
    global video_capture, camera_enabled
    
    # Khởi tạo camera đơn giản
    ensure_video_capture()
    
    # Nếu camera bị tắt, yield placeholder
    if not camera_enabled:
        placeholder = make_placeholder_frame("Camera đã tắt")
        if placeholder is None:
            return
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n')
        return

    # Nếu camera không thể mở, yield hình ảnh placeholder liên tục
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
    detected_count = 0
    recognized_count = 0
    
    while True:
        try:
            ret, frame = video_capture.read()
            if not ret or frame is None:
                continue
        except Exception as e:
            app.logger.error(f"Loi doc frame: {e}")
            continue

        # lấy kích thước khung hình
        frame_h, frame_w = frame.shape[:2]
        
        # Demo mode: tạo bounding box mô phỏng
        if DEMO_MODE:
            # Tạo một số bounding box mô phỏng
            face_data = []
            for i in range(2):  # Hiển thị 2 khuôn mặt mô phỏng
                # Tạo vị trí ngẫu nhiên cho bounding box
                face_size = min(frame_w, frame_h) // 3
                left = max(10, min(frame_w - face_size - 10, i * (frame_w // 2)))
                top = max(10, min(frame_h - face_size - 10, 50 + i * 100))
                right = min(frame_w - 10, left + face_size)
                bottom = min(frame_h - 10, top + face_size)
                
                # Tạo thông tin khuôn mặt mô phỏng
                face_info = {
                    'bbox': (left, top, right, bottom),
                    'name': f'Demo Person {i+1}',
                    'confidence': 0.95,
                    'student_id': f'DEMO{i+1:03d}',
                    'status': 'confirmed' if i == 0 else 'detected'
                }
                face_data.append(face_info)
                
                # Mô phỏng điểm danh
                if i == 0 and frame_count % 30 == 0:  # Mỗi 30 frames
                    try:
                        mark_attendance(face_info['name'], student_id=face_info['student_id'], confidence_score=0.95)
                    except Exception as e:
                        app.logger.exception(f"Loi xac nhan diem danh cho {face_info['name']}: {e}")
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
                        
                        # Mô phỏng điểm danh
                        if frame_count % 30 == 0:  # Mỗi 30 frames
                            try:
                                mark_attendance(name, student_id=student_id, confidence_score=confidence_score)
                            except Exception as e:
                                app.logger.exception(f"Loi xac nhan diem danh cho {name}: {e}")
                    
                    face_info = {
                        'bbox': (left, top, right, bottom),
                        'name': name,
                        'confidence': confidence_score,
                        'student_id': student_id,
                        'status': 'confirmed' if confidence_score > 0.6 else 'detected'
                    }
                    face_data.append(face_info)
        
        # Draw all faces with enhanced display
        frame = face_display_enhancer.draw_multiple_faces(frame, face_data)
        
        # Flip frame horizontally (mirror effect) - chế độ soi gương
        frame = cv2.flip(frame, 1)
        
        # Kiểm tra frame có hợp lệ không
        if frame is None or frame.size == 0:
            continue
        
        # Add system status indicators AFTER flipping (để text hiển thị đúng)
        status_text = "DEMO MODE" if DEMO_MODE else "active"
        frame = face_display_enhancer.add_status_indicator(frame, status_text)
        frame = face_display_enhancer.add_detection_info(frame, detected_count, recognized_count)

        ret2, buf = cv2.imencode('.jpg', frame)
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

# Decorator để yêu cầu đăng nhập
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Routes
@app.route('/')
def index():
    """Trang chính - điểm danh"""
    attendance_data = get_today_attendance()
    return render_template('index.html', attendance=attendance_data)

@app.route('/mark')
def mark_route():
    """Route để đánh dấu điểm danh (legacy)"""
    name = request.args.get('name', 'Unknown')
    mark_attendance(name)
    return redirect(url_for('index'))

@app.route('/video_feed')
def video_feed():
    """Video feed cho camera"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/shutdown')
def shutdown():
    """Tắt server"""
    def _release():
        global video_capture
        with video_lock:
            if video_capture is not None:
                video_capture.release()
                video_capture = None
    
    threading.Thread(target=_release, daemon=True).start()
    return 'Server shutting down...'

@app.route('/reload_faces')
def reload_faces():
    """Tải lại danh sách khuôn mặt"""
    load_known_faces()
    return redirect(url_for('index'))

@app.route('/api/camera/toggle', methods=['POST'])
def toggle_camera():
    """API bật/tắt camera"""
    global camera_enabled, video_capture
    try:
        data = request.get_json()
        enabled = data.get('enabled', True)
        
        camera_enabled = enabled
        
        if not enabled:
            # Tắt camera
            with video_lock:
                if video_capture is not None:
                    video_capture.release()
                    video_capture = None
        else:
            # Bật camera
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

@app.route('/api/attendance/notifications', methods=['GET'])
def get_attendance_notifications():
    """API lấy thông báo điểm danh"""
    notifications = session.get('attendance_notifications', [])
    # Clear notifications after reading
    session['attendance_notifications'] = []
    return jsonify({'notifications': notifications})

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
        
        # Encode frame to base64
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            return jsonify({'error': 'Không thể mã hóa frame'}), 400
        
        import base64
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

# Login routes
@app.route('/login')
def login():
    """Trang đăng nhập"""
    return render_template('login.html')

@app.route('/api/login', methods=['POST'])
def api_login():
    """API đăng nhập"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        
        if not username or not password:
            return jsonify({'error': 'Tên đăng nhập và mật khẩu là bắt buộc'}), 400
        
        # Hash password
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        # Check credentials
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, username, full_name, role, email, is_active 
                FROM users 
                WHERE username = ? AND password_hash = ? AND is_active = 1
            ''', (username, password_hash))
            user = cursor.fetchone()
            
            if not user:
                return jsonify({'error': 'Tên đăng nhập hoặc mật khẩu không đúng'}), 401
            
            # Update last login
            cursor.execute('''
                UPDATE users SET last_login = ? WHERE id = ?
            ''', (datetime.now().isoformat(), user['id']))
            conn.commit()
            
            # Set session
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['full_name'] = user['full_name']
            session['role'] = user['role']
            
            return jsonify({
                'success': True,
                'user': {
                    'id': user['id'],
                    'username': user['username'],
                    'full_name': user['full_name'],
                    'role': user['role']
                }
            })
            
    except Exception as e:
        app.logger.error(f"Login error: {e}")
        return jsonify({'error': 'Lỗi hệ thống'}), 500

@app.route('/api/logout', methods=['POST'])
def api_logout():
    """API đăng xuất"""
    session.clear()
    return jsonify({'success': True, 'message': 'Đăng xuất thành công'})

@app.route('/logout')
def logout():
    """Đăng xuất"""
    session.clear()
    return redirect(url_for('index'))

# Register routes
@app.route('/register')
def register():
    """Trang đăng ký sinh viên"""
    return render_template('register.html')

@app.route('/api/register', methods=['POST'])
def api_register():
    """API đăng ký sinh viên"""
    try:
        data = request.form
        student_id = data.get('student_id', '').strip()
        full_name = data.get('full_name', '').strip()
        email = data.get('email', '').strip()
        phone = data.get('phone', '').strip()
        class_name = data.get('class_name', '').strip()
        
        if not all([student_id, full_name, email, phone, class_name]):
            return jsonify({'error': 'Tất cả các trường đều bắt buộc'}), 400
        
        # Handle image upload
        if 'face_image' in request.files:
            file = request.files['face_image']
            if file and file.filename:
                filename = secure_filename(file.filename)
                file_path = DATA_DIR / f"{student_id}_{full_name}.jpg"
                file.save(str(file_path))
                
                # Process face image
                process_face_image(file_path, student_id)
        
        # Add student to database
        db.add_student(student_id, full_name, email, phone, class_name)
        
        # Reload known faces
        load_known_faces()
        
        return jsonify({'success': True, 'message': 'Đăng ký thành công'})
        
    except Exception as e:
        app.logger.error(f"Registration error: {e}")
        return jsonify({'error': str(e)}), 500

# Management routes
@app.route('/management')
@login_required
def management():
    """Trang quản lý"""
    return render_template('management.html')

@app.route('/students')
@login_required
def students_page():
    """Trang quản lý sinh viên"""
    return render_template('students.html')

@app.route('/reports')
@login_required
def reports_page():
    """Trang báo cáo"""
    return render_template('reports.html')

@app.route('/classes')
@login_required
def classes_page():
    """Trang quản lý lớp học"""
    return render_template('classes.html')

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

def process_face_image(image_path, student_id):
    """Xử lý ảnh khuôn mặt"""
    try:
        if FACE_RECOGNITION_AVAILABLE:
            import face_recognition
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
        
        # Initialize CSV file
        ensure_attendance_csv()
        
        # Load known faces
        load_known_faces()
        
        # Load today's recorded set
        load_today_recorded()
        
        # Log system startup
        db.log_system_event('INFO', 'He thong diem danh khoi dong', 'app')
        app.logger.info("He thong diem danh da khoi dong thanh cong")
        
        app.run(host='0.0.0.0', port=5000, debug=True)
        
    except Exception as e:
        app.logger.error(f"Loi khoi dong he thong: {e}")

