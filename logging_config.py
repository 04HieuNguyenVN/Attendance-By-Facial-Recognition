"""
Cấu hình logging cho hệ thống điểm danh
"""

import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path

def setup_logging(app, log_level='INFO', max_log_size=10*1024*1024, backup_count=5):
    """
    Thiết lập logging cho ứng dụng Flask
    
    Args:
        app: Flask app instance
        log_level: Mức độ log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        max_log_size: Kích thước tối đa của file log (bytes)
        backup_count: Số lượng file log backup
    """
    
    # Tạo thư mục logs nếu chưa có
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Cấu hình logging
    log_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Formatter cho log
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler với rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / 'attendance_system.log',
        maxBytes=max_log_size,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    # Handler cho console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Handler cho file lỗi
    error_handler = logging.handlers.RotatingFileHandler(
        log_dir / 'errors.log',
        maxBytes=max_log_size,
        backupCount=backup_count,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    
    # Handler cho log bảo mật
    security_handler = logging.handlers.RotatingFileHandler(
        log_dir / 'security.log',
        maxBytes=max_log_size,
        backupCount=backup_count,
        encoding='utf-8'
    )
    security_handler.setLevel(logging.INFO)
    security_handler.setFormatter(formatter)
    
    # Cấu hình root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Xóa handlers cũ nếu có
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Thêm handlers mới
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(error_handler)
    
    # Logger riêng cho security
    security_logger = logging.getLogger('security')
    security_logger.addHandler(security_handler)
    security_logger.setLevel(logging.INFO)
    
    # Logger riêng cho face recognition
    face_recognition_logger = logging.getLogger('face_recognition')
    face_recognition_logger.setLevel(logging.INFO)
    
    # Logger riêng cho database
    database_logger = logging.getLogger('database')
    database_logger.setLevel(logging.INFO)
    
    # Logger riêng cho API
    api_logger = logging.getLogger('api')
    api_logger.setLevel(logging.INFO)
    
    # Cấu hình Flask logger
    app.logger.setLevel(log_level)
    
    # Ghi log khi khởi động
    app.logger.info("=" * 50)
    app.logger.info("ATTENDANCE SYSTEM STARTUP")
    app.logger.info(f"Timestamp: {datetime.now().isoformat()}")
    app.logger.info(f"Log Level: {log_level}")
    app.logger.info(f"Log Directory: {log_dir.absolute()}")
    app.logger.info("=" * 50)

class SecurityLogger:
    """Logger chuyên dụng cho các sự kiện bảo mật"""
    
    def __init__(self):
        self.logger = logging.getLogger('security')
    
    def log_login(self, username, ip_address, success=True):
        """Log sự kiện đăng nhập"""
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(f"LOGIN {status} - User: {username}, IP: {ip_address}")
    
    def log_logout(self, username, ip_address):
        """Log sự kiện đăng xuất"""
        self.logger.info(f"LOGOUT - User: {username}, IP: {ip_address}")
    
    def log_unauthorized_access(self, endpoint, ip_address, user_id=None):
        """Log truy cập trái phép"""
        user_info = f", User: {user_id}" if user_id else ""
        self.logger.warning(f"UNAUTHORIZED ACCESS - Endpoint: {endpoint}, IP: {ip_address}{user_info}")
    
    def log_admin_action(self, admin_user, action, details=None):
        """Log hành động của admin"""
        details_info = f", Details: {details}" if details else ""
        self.logger.info(f"ADMIN ACTION - User: {admin_user}, Action: {action}{details_info}")
    
    def log_data_access(self, user_id, data_type, action):
        """Log truy cập dữ liệu"""
        self.logger.info(f"DATA ACCESS - User: {user_id}, Type: {data_type}, Action: {action}")

class FaceRecognitionLogger:
    """Logger chuyên dụng cho face recognition"""
    
    def __init__(self):
        self.logger = logging.getLogger('face_recognition')
    
    def log_face_detected(self, face_count, frame_size):
        """Log phát hiện khuôn mặt"""
        self.logger.debug(f"Face detected - Count: {face_count}, Frame: {frame_size}")
    
    def log_face_recognized(self, name, confidence, student_id=None):
        """Log nhận diện khuôn mặt"""
        student_info = f", Student ID: {student_id}" if student_id else ""
        self.logger.info(f"Face recognized - Name: {name}, Confidence: {confidence:.3f}{student_info}")
    
    def log_attendance_marked(self, name, student_id, confidence=None):
        """Log điểm danh"""
        confidence_info = f", Confidence: {confidence:.3f}" if confidence else ""
        self.logger.info(f"Attendance marked - Name: {name}, Student ID: {student_id}{confidence_info}")
    
    def log_recognition_error(self, error_message):
        """Log lỗi nhận diện"""
        self.logger.error(f"Recognition error - {error_message}")

class DatabaseLogger:
    """Logger chuyên dụng cho database operations"""
    
    def __init__(self):
        self.logger = logging.getLogger('database')
    
    def log_query(self, query_type, table, duration=None):
        """Log truy vấn database"""
        duration_info = f", Duration: {duration:.3f}s" if duration else ""
        self.logger.debug(f"DB Query - Type: {query_type}, Table: {table}{duration_info}")
    
    def log_error(self, operation, error_message):
        """Log lỗi database"""
        self.logger.error(f"DB Error - Operation: {operation}, Error: {error_message}")
    
    def log_backup(self, backup_type, success=True):
        """Log sao lưu database"""
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(f"DB Backup - Type: {backup_type}, Status: {status}")

class APILogger:
    """Logger chuyên dụng cho API calls"""
    
    def __init__(self):
        self.logger = logging.getLogger('api')
    
    def log_request(self, method, endpoint, user_id=None, ip_address=None):
        """Log yêu cầu API"""
        user_info = f", User: {user_id}" if user_id else ""
        ip_info = f", IP: {ip_address}" if ip_address else ""
        self.logger.info(f"API Request - {method} {endpoint}{user_info}{ip_info}")
    
    def log_response(self, endpoint, status_code, duration=None):
        """Log phản hồi API"""
        duration_info = f", Duration: {duration:.3f}s" if duration else ""
        self.logger.info(f"API Response - {endpoint}, Status: {status_code}{duration_info}")
    
    def log_error(self, endpoint, error_message, status_code=500):
        """Log lỗi API"""
        self.logger.error(f"API Error - {endpoint}, Status: {status_code}, Error: {error_message}")

# Các instance logger toàn cục
security_logger = SecurityLogger()
face_recognition_logger = FaceRecognitionLogger()
database_logger = DatabaseLogger()
api_logger = APILogger()

def get_client_ip(request):
    """Lấy IP address của client"""
    if request.headers.get('X-Forwarded-For'):
        return request.headers.get('X-Forwarded-For').split(',')[0].strip()
    elif request.headers.get('X-Real-IP'):
        return request.headers.get('X-Real-IP')
    else:
        return request.remote_addr

def log_request_info(request, user_id=None):
    """Log thông tin request"""
    ip_address = get_client_ip(request)
    api_logger.log_request(
        request.method,
        request.endpoint,
        user_id=user_id,
        ip_address=ip_address
    )
    return ip_address
