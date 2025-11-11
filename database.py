"""
Database module for Attendance System
Quản lý cơ sở dữ liệu SQLite cho hệ thống điểm danh
"""

import sqlite3
import os
from datetime import datetime, date
from pathlib import Path
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path="attendance_system.db"):
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        """Tạo kết nối database"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Cho phép truy cập theo tên cột
        return conn
    
    def init_database(self):
        """Khởi tạo database và các bảng"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Bảng lớp học
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS classes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    class_code VARCHAR(20) UNIQUE NOT NULL,
                    class_name VARCHAR(100) NOT NULL,
                    semester VARCHAR(20),
                    academic_year VARCHAR(20),
                    teacher_name VARCHAR(100),
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
            ''')
            
            # Bảng sinh viên (cải thiện)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS students (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id VARCHAR(20) UNIQUE NOT NULL,
                    full_name VARCHAR(100) NOT NULL,
                    email VARCHAR(100),
                    phone VARCHAR(20),
                    class_id INTEGER,
                    face_encoding BLOB,
                    face_image_path VARCHAR(200),
                    enrollment_date DATE,
                    status VARCHAR(20) DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (class_id) REFERENCES classes(id)
                )
            ''')
            
            # Bảng điểm danh (cải thiện)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id VARCHAR(20) NOT NULL,
                    student_name VARCHAR(100) NOT NULL,
                    class_id INTEGER,
                    attendance_date DATE NOT NULL,
                    check_in_time TIMESTAMP,
                    check_out_time TIMESTAMP,
                    status VARCHAR(20) DEFAULT 'present',
                    confidence_score REAL,
                    face_image_path VARCHAR(200),
                    notes TEXT,
                    session_type VARCHAR(20) DEFAULT 'lecture',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (student_id) REFERENCES students(student_id),
                    FOREIGN KEY (class_id) REFERENCES classes(id)
                )
            ''')
            
            # Bảng cài đặt hệ thống
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS settings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    setting_key VARCHAR(50) UNIQUE NOT NULL,
                    setting_value TEXT,
                    description TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Bảng người dùng (admin/giáo viên)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    full_name VARCHAR(100) NOT NULL,
                    role VARCHAR(20) DEFAULT 'teacher',
                    email VARCHAR(100),
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP
                )
            ''')
            
            # Bảng logs hệ thống
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    log_level VARCHAR(20) NOT NULL,
                    message TEXT NOT NULL,
                    module VARCHAR(50),
                    user_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')
            
            # Bảng tokens cho mobile app
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_tokens (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    token VARCHAR(255) UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at REAL NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')
            
            # Tạo indexes để tối ưu hiệu suất
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(attendance_date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_attendance_student ON attendance(student_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_students_active ON students(is_active)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_created ON system_logs(created_at)')
            
            # Thêm cài đặt mặc định
            self._insert_default_settings(cursor)
            
            # Tạo admin mặc định
            self._create_default_admin(cursor)
            
            # Tạo teacher mặc định
            self._create_default_teacher(cursor)
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    def _insert_default_settings(self, cursor):
        """Thêm các cài đặt mặc định"""
        default_settings = [
            ('face_recognition_threshold', '0.5', 'Ngưỡng nhận diện khuôn mặt (0.0-1.0)'),
            ('attendance_timeout', '5.0', 'Thời gian chờ xác nhận điểm danh (giây)'),
            ('camera_index', '0', 'Chỉ số camera mặc định'),
            ('min_face_ratio', '0.15', 'Tỷ lệ khuôn mặt tối thiểu trong khung hình'),
            ('process_every_frames', '4', 'Xử lý mỗi N khung hình'),
            ('system_name', 'Hệ thống điểm danh thông minh', 'Tên hệ thống'),
            ('auto_checkout_time', '17:00', 'Thời gian tự động checkout'),
            ('max_attendance_per_day', '2', 'Số lần điểm danh tối đa mỗi ngày'),
        ]
        
        for key, value, desc in default_settings:
            cursor.execute('''
                INSERT OR IGNORE INTO settings (setting_key, setting_value, description)
                VALUES (?, ?, ?)
            ''', (key, value, desc))
    
    def _create_default_admin(self, cursor):
        """Tạo tài khoản admin mặc định"""
        import hashlib
        default_password = hashlib.sha256('admin123'.encode()).hexdigest()
        
        cursor.execute('''
            INSERT OR IGNORE INTO users (username, password_hash, full_name, role, email)
            VALUES (?, ?, ?, ?, ?)
        ''', ('admin', default_password, 'Administrator', 'admin', 'admin@example.com'))
    
    def _create_default_teacher(self, cursor):
        """Tạo tài khoản teacher mặc định"""
        import hashlib
        default_password = hashlib.sha256('teacher123'.encode()).hexdigest()
        
        cursor.execute('''
            INSERT OR IGNORE INTO users (username, password_hash, full_name, role, email)
            VALUES (?, ?, ?, ?, ?)
        ''', ('teacher', default_password, 'Giáo viên', 'teacher', 'teacher@example.com'))
    
    # === QUẢN LÝ SINH VIÊN ===
    def add_student(self, student_id, full_name, email=None, phone=None, class_name=None, face_image_path=None):
        """Thêm sinh viên mới"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                # Tìm hoặc tạo class_id từ class_name
                class_id = None
                if class_name:
                    cursor.execute('SELECT id FROM classes WHERE class_name = ?', (class_name,))
                    result = cursor.fetchone()
                    if result:
                        class_id = result[0]
                    else:
                        # Tạo lớp mới nếu chưa có
                        # Generate class_code from class_name (remove spaces, uppercase)
                        class_code = class_name.replace(' ', '').upper()[:20]
                        cursor.execute('INSERT INTO classes (class_code, class_name) VALUES (?, ?)', 
                                     (class_code, class_name))
                        class_id = cursor.lastrowid
                        logger.info(f"Created new class: {class_name} (ID: {class_id}, Code: {class_code})")
                
                cursor.execute('''
                    INSERT INTO students (student_id, full_name, email, phone, class_id, face_image_path)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (student_id, full_name, email, phone, class_id, face_image_path))
                conn.commit()
                logger.info(f"Added student: {full_name} ({student_id})")
                return True
            except sqlite3.IntegrityError as e:
                logger.error(f"Student ID {student_id} already exists: {e}")
                return False
    
    def get_student(self, student_id):
        """Lấy thông tin sinh viên"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM students WHERE student_id = ?', (student_id,))
            return cursor.fetchone()
    
    def get_all_students(self, active_only=True):
        """Lấy danh sách tất cả sinh viên"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if active_only:
                cursor.execute('SELECT * FROM students WHERE is_active = 1 ORDER BY full_name')
            else:
                cursor.execute('SELECT * FROM students ORDER BY full_name')
            return cursor.fetchall()
    
    def update_student(self, student_id, **kwargs):
        """Cập nhật thông tin sinh viên"""
        allowed_fields = ['full_name', 'email', 'phone', 'class_name', 'face_image_path', 'is_active']
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        
        if not updates:
            return False
        
        updates['updated_at'] = datetime.now().isoformat()
        set_clause = ', '.join([f"{k} = ?" for k in updates.keys()])
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
                UPDATE students SET {set_clause} WHERE student_id = ?
            ''', list(updates.values()) + [student_id])
            conn.commit()
            return cursor.rowcount > 0
    
    def delete_student(self, student_id):
        """Xóa sinh viên (soft delete)"""
        return self.update_student(student_id, is_active=False)
    
    # === QUẢN LÝ ĐIỂM DANH ===
    def mark_attendance(self, student_id, student_name, status='present', confidence_score=None, notes=None):
        """Điểm danh sinh viên"""
        today = date.today().isoformat()
        now = datetime.now().isoformat()
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Kiểm tra xem đã điểm danh chưa
            cursor.execute('''
                SELECT id FROM attendance 
                WHERE student_id = ? AND attendance_date = ? AND status = 'present'
            ''', (student_id, today))
            
            if cursor.fetchone():
                logger.info(f"Student {student_name} already marked present today")
                return False
            
            # Thêm điểm danh
            cursor.execute('''
                INSERT INTO attendance (student_id, student_name, attendance_date, check_in_time, status, confidence_score, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (student_id, student_name, today, now, status, confidence_score, notes))
            
            conn.commit()
            logger.info(f"Marked attendance for {student_name} ({student_id})")
            return True
    
    def update_last_seen(self, student_id, student_name):
        """Cập nhật lần cuối thấy sinh viên (để tracking presence)"""
        today = date.today().isoformat()
        now = datetime.now().isoformat()
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Tìm bản ghi điểm danh hôm nay
            cursor.execute('''
                SELECT id, check_in_time, check_out_time FROM attendance 
                WHERE student_id = ? AND attendance_date = ? AND status = 'present'
                ORDER BY check_in_time DESC LIMIT 1
            ''', (student_id, today))
            
            record = cursor.fetchone()
            if record:
                # Cập nhật notes với timestamp cuối cùng
                cursor.execute('''
                    UPDATE attendance 
                    SET notes = ?
                    WHERE id = ?
                ''', (f"Last seen: {now}", record['id']))
                conn.commit()
                return True
            return False
    
    def mark_checkout(self, student_id):
        """Đánh dấu checkout cho sinh viên"""
        today = date.today().isoformat()
        now = datetime.now().isoformat()
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Tìm bản ghi điểm danh hôm nay chưa checkout
            cursor.execute('''
                SELECT id, check_in_time FROM attendance 
                WHERE student_id = ? AND attendance_date = ? AND check_out_time IS NULL
                ORDER BY check_in_time DESC LIMIT 1
            ''', (student_id, today))
            
            record = cursor.fetchone()
            if record:
                cursor.execute('''
                    UPDATE attendance 
                    SET check_out_time = ?
                    WHERE id = ?
                ''', (now, record['id']))
                conn.commit()
                logger.info(f"Marked checkout for student {student_id}")
                return True
            return False
    
    def get_attendance_with_duration(self, attendance_date=None):
        """Lấy danh sách điểm danh với thời gian có mặt"""
        if not attendance_date:
            attendance_date = date.today().isoformat()
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT 
                    a.*,
                    s.full_name, 
                    s.email, 
                    s.phone, 
                    c.class_name,
                    CASE 
                        WHEN a.check_out_time IS NOT NULL THEN
                            CAST((julianday(a.check_out_time) - julianday(a.check_in_time)) * 24 * 60 AS INTEGER)
                        ELSE
                            CAST((julianday('now', 'localtime') - julianday(a.check_in_time)) * 24 * 60 AS INTEGER)
                    END as duration_minutes
                FROM attendance a
                LEFT JOIN students s ON a.student_id = s.student_id
                LEFT JOIN classes c ON s.class_id = c.id
                WHERE a.attendance_date = ?
                ORDER BY a.check_in_time DESC
            ''', (attendance_date,))
            return cursor.fetchall()
    
    def get_today_attendance(self):
        """Lấy danh sách điểm danh hôm nay"""
        today = date.today().isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT a.*, s.full_name, s.email, s.phone, c.class_name
                FROM attendance a
                LEFT JOIN students s ON a.student_id = s.student_id
                LEFT JOIN classes c ON s.class_id = c.id
                WHERE a.attendance_date = ?
                ORDER BY a.check_in_time DESC
            ''', (today,))
            return cursor.fetchall()
    
    def get_attendance_by_date_range(self, start_date, end_date):
        """Lấy điểm danh theo khoảng thời gian"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT a.*, s.email, s.phone, c.class_name
                FROM attendance a
                LEFT JOIN students s ON a.student_id = s.student_id
                LEFT JOIN classes c ON s.class_id = c.id
                WHERE a.attendance_date BETWEEN ? AND ?
                ORDER BY a.attendance_date DESC, a.check_in_time DESC
            ''', (start_date, end_date))
            return cursor.fetchall()
    
    def get_student_attendance_history(self, student_id, limit=30):
        """Lấy lịch sử điểm danh của sinh viên"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM attendance 
                WHERE student_id = ?
                ORDER BY attendance_date DESC, check_in_time DESC
                LIMIT ?
            ''', (student_id, limit))
            return cursor.fetchall()
    
    # === QUẢN LÝ CÀI ĐẶT ===
    def get_setting(self, key, default=None):
        """Lấy giá trị cài đặt"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT setting_value FROM settings WHERE setting_key = ?', (key,))
            result = cursor.fetchone()
            return result['setting_value'] if result else default
    
    def set_setting(self, key, value):
        """Cập nhật cài đặt"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO settings (setting_key, setting_value, updated_at)
                VALUES (?, ?, ?)
            ''', (key, value, datetime.now().isoformat()))
            conn.commit()
    
    def get_all_settings(self):
        """Lấy tất cả cài đặt"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM settings ORDER BY setting_key')
            return cursor.fetchall()
    
    # === THỐNG KÊ ===
    def get_attendance_stats(self, start_date=None, end_date=None):
        """Lấy thống kê điểm danh"""
        if not start_date:
            start_date = date.today().isoformat()
        if not end_date:
            end_date = start_date
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Tổng số sinh viên
            cursor.execute('SELECT COUNT(*) FROM students WHERE is_active = 1')
            total_students = cursor.fetchone()[0]
            
            # Số sinh viên đã điểm danh
            cursor.execute('''
                SELECT COUNT(DISTINCT student_id) FROM attendance 
                WHERE attendance_date BETWEEN ? AND ?
            ''', (start_date, end_date))
            attended_students = cursor.fetchone()[0]
            
            # Tỷ lệ điểm danh
            attendance_rate = (attended_students / total_students * 100) if total_students > 0 else 0
            
            return {
                'total_students': total_students,
                'attended_students': attended_students,
                'attendance_rate': round(attendance_rate, 2),
                'period': f"{start_date} to {end_date}"
            }
    
    def get_daily_attendance_stats(self, days=30):
        """Lấy thống kê điểm danh theo ngày"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT 
                    attendance_date,
                    COUNT(*) as total_attendance,
                    COUNT(DISTINCT student_id) as unique_students
                FROM attendance 
                WHERE attendance_date >= date('now', '-{} days')
                GROUP BY attendance_date
                ORDER BY attendance_date DESC
            '''.format(days))
            return cursor.fetchall()
    
    # === LOGGING ===
    def log_system_event(self, level, message, module=None, user_id=None):
        """Ghi log hệ thống"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO system_logs (log_level, message, module, user_id)
                VALUES (?, ?, ?, ?)
            ''', (level, message, module, user_id))
            conn.commit()
    
    def get_system_logs(self, limit=100, level=None):
        """Lấy logs hệ thống"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if level:
                cursor.execute('''
                    SELECT sl.*, u.username 
                    FROM system_logs sl
                    LEFT JOIN users u ON sl.user_id = u.id
                    WHERE sl.log_level = ?
                    ORDER BY sl.created_at DESC
                    LIMIT ?
                ''', (level, limit))
            else:
                cursor.execute('''
                    SELECT sl.*, u.username 
                    FROM system_logs sl
                    LEFT JOIN users u ON sl.user_id = u.id
                    ORDER BY sl.created_at DESC
                    LIMIT ?
                ''', (limit,))
            return cursor.fetchall()
    
    # === QUẢN LÝ LỚP HỌC ===
    
    def create_class(self, class_code, class_name, semester=None, academic_year=None, teacher_name=None, description=None):
        """Tạo lớp học mới"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute('''
                    INSERT INTO classes (class_code, class_name, semester, academic_year, teacher_name, description)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (class_code, class_name, semester, academic_year, teacher_name, description))
                conn.commit()
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                raise ValueError(f"Lớp học với mã {class_code} đã tồn tại")
    
    def get_all_classes(self):
        """Lấy danh sách tất cả lớp học"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT c.*, COUNT(s.id) as student_count
                FROM classes c
                LEFT JOIN students s ON c.id = s.class_id AND s.is_active = 1
                WHERE c.is_active = 1
                GROUP BY c.id
                ORDER BY c.class_name
            ''')
            return [dict(row) for row in cursor.fetchall()]
    
    def get_class_by_id(self, class_id):
        """Lấy thông tin lớp học theo ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM classes WHERE id = ? AND is_active = 1', (class_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def update_class(self, class_id, **kwargs):
        """Cập nhật thông tin lớp học"""
        if not kwargs:
            return False
        
        valid_fields = ['class_code', 'class_name', 'semester', 'academic_year', 'teacher_name', 'description']
        updates = []
        values = []
        
        for field, value in kwargs.items():
            if field in valid_fields:
                updates.append(f"{field} = ?")
                values.append(value)
        
        if not updates:
            return False
        
        updates.append("updated_at = CURRENT_TIMESTAMP")
        values.append(class_id)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
                UPDATE classes 
                SET {', '.join(updates)}
                WHERE id = ? AND is_active = 1
            ''', values)
            conn.commit()
            return cursor.rowcount > 0
    
    def delete_class(self, class_id):
        """Xóa lớp học (soft delete)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE classes 
                SET is_active = 0, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (class_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    def get_students_by_class(self, class_id):
        """Lấy danh sách sinh viên trong lớp"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT s.*, c.class_name, c.class_code
                FROM students s
                LEFT JOIN classes c ON s.class_id = c.id
                WHERE s.class_id = ? AND s.is_active = 1
                ORDER BY s.student_id
            ''', (class_id,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_class_attendance_stats(self, class_id, start_date=None, end_date=None):
        """Lấy thống kê điểm danh của lớp"""
        if not start_date:
            start_date = date.today().strftime('%Y-%m-%d')
        if not end_date:
            end_date = date.today().strftime('%Y-%m-%d')
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Tổng số sinh viên trong lớp
            cursor.execute('''
                SELECT COUNT(*) FROM students 
                WHERE class_id = ? AND is_active = 1
            ''', (class_id,))
            total_students = cursor.fetchone()[0]
            
            # Số sinh viên đã điểm danh
            cursor.execute('''
                SELECT COUNT(DISTINCT student_id) FROM attendance 
                WHERE class_id = ? AND attendance_date BETWEEN ? AND ?
            ''', (class_id, start_date, end_date))
            attended_students = cursor.fetchone()[0]
            
            # Chi tiết điểm danh theo ngày
            cursor.execute('''
                SELECT 
                    attendance_date,
                    COUNT(*) as total_attendance,
                    COUNT(DISTINCT student_id) as unique_students
                FROM attendance 
                WHERE class_id = ? AND attendance_date BETWEEN ? AND ?
                GROUP BY attendance_date
                ORDER BY attendance_date DESC
            ''', (class_id, start_date, end_date))
            daily_stats = [dict(row) for row in cursor.fetchall()]
            
            attendance_rate = (attended_students / total_students * 100) if total_students > 0 else 0
            
            return {
                'total_students': total_students,
                'attended_students': attended_students,
                'attendance_rate': round(attendance_rate, 2),
                'daily_stats': daily_stats,
                'period': f"{start_date} to {end_date}"
            }

# Khởi tạo instance global
db = DatabaseManager()
