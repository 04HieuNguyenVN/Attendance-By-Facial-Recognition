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

            # Bổ sung khóa liên kết user -> student (không áp UNIQUE trực tiếp để tránh lỗi ALTER)
            self._ensure_column(cursor, 'students', 'user_id', 'INTEGER')
            
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

            # Bổ sung cột mới (nếu thiếu) để liên kết lớp tín chỉ và phiên điểm danh
            self._ensure_column(cursor, 'attendance', 'credit_class_id', 'INTEGER')
            self._ensure_column(cursor, 'attendance', 'session_id', 'INTEGER')
            
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

            # Bảng thông tin giảng viên
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS teachers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    teacher_code VARCHAR(20) UNIQUE NOT NULL,
                    full_name VARCHAR(100) NOT NULL,
                    email VARCHAR(100),
                    phone VARCHAR(20),
                    department VARCHAR(100),
                    user_id INTEGER UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users(id)
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
            
            # Bảng lưu nhiều ảnh mẫu của sinh viên (cho AI training)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS student_face_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id VARCHAR(20) NOT NULL,
                    image_path VARCHAR(200) NOT NULL,
                    embedding BLOB,
                    quality_score REAL,
                    is_primary BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE
                )
            ''')

            # Bảng lớp tín chỉ (dành cho giảng viên)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS credit_classes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    credit_code VARCHAR(30) UNIQUE NOT NULL,
                    subject_name VARCHAR(150) NOT NULL,
                    teacher_id INTEGER,
                    semester VARCHAR(20),
                    academic_year VARCHAR(20),
                    room VARCHAR(50),
                    schedule_info TEXT,
                    status VARCHAR(20) DEFAULT 'draft',
                    enrollment_limit INTEGER,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (teacher_id) REFERENCES teachers(id)
                )
            ''')

            # Bảng liên kết sinh viên với lớp tín chỉ
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS credit_class_students (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    credit_class_id INTEGER NOT NULL,
                    student_id INTEGER NOT NULL,
                    enrollment_status VARCHAR(20) DEFAULT 'active',
                    joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (credit_class_id, student_id),
                    FOREIGN KEY (credit_class_id) REFERENCES credit_classes(id),
                    FOREIGN KEY (student_id) REFERENCES students(id)
                )
            ''')

            # Bảng phiên điểm danh cho lớp tín chỉ
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS attendance_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    credit_class_id INTEGER NOT NULL,
                    opened_by INTEGER,
                    session_date DATE NOT NULL,
                    opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    checkin_deadline TIMESTAMP,
                    checkout_deadline TIMESTAMP,
                    closed_at TIMESTAMP,
                    status VARCHAR(20) DEFAULT 'scheduled',
                    notes TEXT,
                    FOREIGN KEY (credit_class_id) REFERENCES credit_classes(id),
                    FOREIGN KEY (opened_by) REFERENCES users(id)
                )
            ''')
            
            # Tạo indexes để tối ưu hiệu suất
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(attendance_date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_attendance_student ON attendance(student_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_students_active ON students(is_active)')
            try:
                cursor.execute(
                    'CREATE UNIQUE INDEX IF NOT EXISTS idx_students_user '
                    'ON students(user_id) WHERE user_id IS NOT NULL'
                )
            except sqlite3.OperationalError as exc:
                logger.warning("Không thể tạo unique index cho students.user_id: %s", exc)
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_created ON system_logs(created_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_face_samples_student ON student_face_samples(student_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_teachers_code ON teachers(teacher_code)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_credit_classes_teacher ON credit_classes(teacher_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_credit_class_students_cc ON credit_class_students(credit_class_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_attendance_sessions_class ON attendance_sessions(credit_class_id)')
            
            # Thêm cài đặt mặc định
            self._insert_default_settings(cursor)
            
            # Tạo admin mặc định
            self._create_default_admin(cursor)
            
            # Tạo teacher mặc định
            self._create_default_teacher(cursor)

            # Tạo student mặc định (để test frontend)
            self._create_default_student_user(cursor)
            
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

    def _create_default_student_user(self, cursor):
        """Tạo tài khoản sinh viên mặc định để thử nghiệm UI."""
        import hashlib
        cursor.execute('SELECT id FROM users WHERE username = ? AND is_active = 1', ('student',))
        if cursor.fetchone():
            return

        password_hash = hashlib.sha256('student123'.encode()).hexdigest()
        cursor.execute('''
            INSERT INTO users (username, password_hash, full_name, role, email)
            VALUES (?, ?, ?, ?, ?)
        ''', ('student', password_hash, 'Sinh viên mặc định', 'student', 'student@example.com'))
        student_user_id = cursor.lastrowid

        # Liên kết user này với sinh viên đầu tiên (nếu có sẵn dữ liệu mẫu)
        cursor.execute('SELECT id FROM students WHERE is_active = 1 ORDER BY id LIMIT 1')
        row = cursor.fetchone()
        if row:
            student_db_id = row['id'] if isinstance(row, sqlite3.Row) else row[0]
            cursor.execute('''
                UPDATE students
                SET user_id = ?
                WHERE id = ? AND (user_id IS NULL OR user_id = ?)
            ''', (student_user_id, student_db_id, student_user_id))

    # === QUẢN LÝ NGƯỜI DÙNG ===

    def get_user_by_id(self, user_id):
        """Lấy thông tin người dùng theo ID."""
        if not user_id:
            return None
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE id = ? AND is_active = 1', (user_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_user_by_username(self, username):
        """Lấy thông tin người dùng theo username."""
        if not username:
            return None
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE username = ? AND is_active = 1', (username,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def create_user(self, username, password_hash, full_name, role='student', email=None, is_active=True):
        """Tạo tài khoản người dùng mới và trả về ID."""
        if not username or not password_hash or not full_name:
            raise ValueError("Thiếu thông tin tạo tài khoản người dùng")

        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute('''
                    INSERT INTO users (username, password_hash, full_name, role, email, is_active)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (username, password_hash, full_name, role or 'student', email, 1 if is_active else 0))
                conn.commit()
                return cursor.lastrowid
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"Tài khoản {username} đã tồn tại") from exc

    def update_user_password(self, user_id, password_hash):
        """Cập nhật mật khẩu người dùng."""
        if not user_id or not password_hash:
            return False
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE users SET password_hash = ? WHERE id = ? AND is_active = 1',
                (password_hash, user_id)
            )
            conn.commit()
            return cursor.rowcount > 0

    def update_last_login(self, user_id):
        """Cập nhật thời gian đăng nhập cuối cùng."""
        if not user_id:
            return False
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE users SET last_login = ? WHERE id = ? AND is_active = 1',
                (datetime.now().isoformat(), user_id)
            )
            conn.commit()
            return cursor.rowcount > 0

    def get_students_missing_user(self, active_only=True):
        """Trả về danh sách sinh viên chưa được gán tài khoản người dùng."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = [
                'SELECT * FROM students',
                'WHERE (user_id IS NULL OR user_id = "" OR user_id = 0)'
            ]
            if active_only:
                query.append('AND is_active = 1')
            query.append('ORDER BY full_name')
            cursor.execute(' '.join(query))
            return cursor.fetchall()

    def _ensure_column(self, cursor, table_name, column_name, column_def):
        """Thêm cột mới nếu chưa tồn tại (dùng cho nâng cấp DB)."""
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in cursor.fetchall()]
        if column_name in columns:
            return
        ddl = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_def}".strip()
        try:
            cursor.execute(ddl)
        except sqlite3.OperationalError as exc:
            logger.warning(
                "Không thể thêm cột %s.%s (%s): %s",
                table_name,
                column_name,
                column_def,
                exc,
            )

    def _generate_class_code(self, class_name, attempt=0):
        """Generate a normalized class code with optional suffix for uniqueness."""
        normalized = ''.join(ch for ch in class_name if ch.isalnum()) or 'CLASS'
        normalized = normalized.upper()
        if attempt <= 0:
            return normalized[:20]
        suffix = str(attempt)
        prefix_length = max(1, 20 - len(suffix))
        return (normalized[:prefix_length] + suffix)[:20]

    def _get_or_create_class(self, cursor, class_name):
        """Find existing class_id by name or create a new class entry."""
        if class_name is None:
            return None
        trimmed = class_name.strip()
        if not trimmed:
            return None

        cursor.execute('SELECT id FROM classes WHERE class_name = ?', (trimmed,))
        row = cursor.fetchone()
        if row:
            return row[0]

        attempt = 0
        while attempt < 10:
            class_code = self._generate_class_code(trimmed, attempt)
            try:
                cursor.execute('''
                    INSERT INTO classes (class_code, class_name)
                    VALUES (?, ?)
                ''', (class_code, trimmed))
                logger.info(f"Created new class: {trimmed} (ID: {cursor.lastrowid}, Code: {class_code})")
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                attempt += 1

        raise ValueError(f"Không thể tạo lớp học mới cho {trimmed}")
    
    # === QUẢN LÝ SINH VIÊN ===
    def add_student(self, student_id, full_name, email=None, phone=None, class_name=None, face_image_path=None):
        """Thêm sinh viên mới"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                # Tìm hoặc tạo class_id từ class_name
                class_id = None
                if class_name:
                    class_id = self._get_or_create_class(cursor, class_name)
                
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

    def get_student_by_user(self, user_id):
        """Lấy sinh viên dựa trên user_id đã liên kết."""
        if not user_id:
            return None
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM students WHERE user_id = ? AND is_active = 1', (user_id,))
            return cursor.fetchone()

    def link_student_to_user(self, student_identifier, user_id):
        """Gán user_id cho sinh viên (nếu chưa được gán)."""
        if not student_identifier or not user_id:
            return False
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                student_db_id = self._resolve_student_db_id(cursor, student_identifier)
            except ValueError:
                return False

            cursor.execute('''
                UPDATE students
                SET user_id = ?
                WHERE id = ? AND (user_id IS NULL OR user_id = ?)
            ''', (user_id, student_db_id, user_id))
            conn.commit()
            return cursor.rowcount > 0
    
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
        if not kwargs:
            return False

        allowed_fields = ['full_name', 'email', 'phone', 'face_image_path', 'is_active', 'class_id']
        updates = {}
        class_name = kwargs.pop('class_name', None)

        with self.get_connection() as conn:
            cursor = conn.cursor()

            if class_name is not None:
                class_id = None
                if isinstance(class_name, str) and class_name.strip():
                    class_id = self._get_or_create_class(cursor, class_name)
                updates['class_id'] = class_id

            for field in allowed_fields:
                if field in kwargs and kwargs[field] is not None:
                    value = kwargs[field]
                    if field == 'is_active':
                        if isinstance(value, str):
                            value = 1 if value.strip().lower() in ('1', 'true', 'on', 'yes') else 0
                        else:
                            value = 1 if bool(value) else 0
                    updates[field] = value

            if not updates:
                return False

            updates['updated_at'] = datetime.now().isoformat()
            set_clause = ', '.join([f"{k} = ?" for k in updates.keys()])
            cursor.execute(
                f'''UPDATE students SET {set_clause} WHERE student_id = ?''',
                list(updates.values()) + [student_id]
            )
            conn.commit()
            return cursor.rowcount > 0
    
    def delete_student(self, student_id):
        """Xóa sinh viên (soft delete)"""
        return self.update_student(student_id, is_active=False)
    
    # === QUẢN LÝ ẢNH MẪU SINH VIÊN (CHO AI TRAINING) ===
    
    def add_face_sample(self, student_id, image_path, embedding=None, quality_score=None, is_primary=False):
        """Thêm ảnh mẫu khuôn mặt cho sinh viên"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute('''
                    INSERT INTO student_face_samples (student_id, image_path, embedding, quality_score, is_primary)
                    VALUES (?, ?, ?, ?, ?)
                ''', (student_id, image_path, embedding, quality_score, is_primary))
                conn.commit()
                logger.info(f"Added face sample for student {student_id}: {image_path}")
                return cursor.lastrowid
            except Exception as e:
                logger.error(f"Failed to add face sample: {e}")
                return None
    
    def get_face_samples(self, student_id):
        """Lấy tất cả ảnh mẫu của sinh viên"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM student_face_samples 
                WHERE student_id = ?
                ORDER BY is_primary DESC, created_at DESC
            ''', (student_id,))
            return cursor.fetchall()
    
    def get_primary_face_sample(self, student_id):
        """Lấy ảnh đại diện chính của sinh viên"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM student_face_samples 
                WHERE student_id = ? AND is_primary = 1
                LIMIT 1
            ''', (student_id,))
            return cursor.fetchone()
    
    def set_primary_face_sample(self, sample_id):
        """Đặt một ảnh làm ảnh đại diện chính"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Lấy student_id từ sample_id
            cursor.execute('SELECT student_id FROM student_face_samples WHERE id = ?', (sample_id,))
            result = cursor.fetchone()
            if not result:
                return False
            
            student_id = result['student_id']
            
            # Bỏ primary của tất cả ảnh khác
            cursor.execute('''
                UPDATE student_face_samples 
                SET is_primary = 0 
                WHERE student_id = ?
            ''', (student_id,))
            
            # Đặt ảnh này làm primary
            cursor.execute('''
                UPDATE student_face_samples 
                SET is_primary = 1 
                WHERE id = ?
            ''', (sample_id,))
            
            conn.commit()
            return True
    
    def delete_face_sample(self, sample_id):
        """Xóa ảnh mẫu"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM student_face_samples WHERE id = ?', (sample_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    def get_all_student_embeddings(self):
        """Lấy tất cả embeddings của sinh viên (cho training)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT sfs.student_id, s.full_name, sfs.embedding, sfs.image_path
                FROM student_face_samples sfs
                JOIN students s ON sfs.student_id = s.student_id
                WHERE s.is_active = 1 AND sfs.embedding IS NOT NULL
                ORDER BY sfs.student_id, sfs.quality_score DESC
            ''')
            return cursor.fetchall()
    
    def count_face_samples(self, student_id):
        """Đếm số lượng ảnh mẫu của sinh viên"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM student_face_samples WHERE student_id = ?', (student_id,))
            return cursor.fetchone()[0]
    
    # === QUẢN LÝ ĐIỂM DANH ===
    def mark_attendance(self, student_id, student_name, status='present', confidence_score=None,
                        notes=None, credit_class_id=None, session_id=None):
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
                INSERT INTO attendance (
                    student_id, student_name, attendance_date, check_in_time,
                    status, confidence_score, notes, credit_class_id, session_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (student_id, student_name, today, now, status, confidence_score, notes,
                  credit_class_id, session_id))
            
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
                    cc.subject_name AS credit_class_name,
                    cc.credit_code AS credit_class_code,
                    CASE 
                        WHEN a.check_out_time IS NOT NULL THEN
                            CAST((julianday(a.check_out_time) - julianday(a.check_in_time)) * 24 * 60 AS INTEGER)
                        ELSE
                            CAST((julianday('now', 'localtime') - julianday(a.check_in_time)) * 24 * 60 AS INTEGER)
                    END as duration_minutes
                FROM attendance a
                LEFT JOIN students s ON a.student_id = s.student_id
                LEFT JOIN classes c ON s.class_id = c.id
                LEFT JOIN credit_classes cc ON a.credit_class_id = cc.id
                WHERE a.attendance_date = ?
                ORDER BY a.check_in_time DESC
            ''', (attendance_date,))
            return [dict(r) for r in cursor.fetchall()]
    
    def get_today_attendance(self):
        """Lấy danh sách điểm danh hôm nay"""
        today = date.today().isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT 
                    a.*, 
                    s.full_name, s.email, s.phone, c.class_name,
                    cc.subject_name AS credit_class_name,
                    cc.credit_code AS credit_class_code
                FROM attendance a
                LEFT JOIN students s ON a.student_id = s.student_id
                LEFT JOIN classes c ON s.class_id = c.id
                LEFT JOIN credit_classes cc ON a.credit_class_id = cc.id
                WHERE a.attendance_date = ?
                ORDER BY a.check_in_time DESC
            ''', (today,))
            return [dict(r) for r in cursor.fetchall()]
    
    def get_attendance_by_date_range(self, start_date, end_date):
        """Lấy điểm danh theo khoảng thời gian"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT 
                    a.*, 
                    s.email, s.phone, c.class_name,
                    cc.subject_name AS credit_class_name,
                    cc.credit_code AS credit_class_code
                FROM attendance a
                LEFT JOIN students s ON a.student_id = s.student_id
                LEFT JOIN classes c ON s.class_id = c.id
                LEFT JOIN credit_classes cc ON a.credit_class_id = cc.id
                WHERE a.attendance_date BETWEEN ? AND ?
                ORDER BY a.attendance_date DESC, a.check_in_time DESC
            ''', (start_date, end_date))
            return [dict(r) for r in cursor.fetchall()]
    
    def get_student_attendance_history(self, student_id, limit=30):
        """Lấy lịch sử điểm danh của sinh viên"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT 
                    a.*, s.full_name, s.class_id, c.class_name,
                    cc.subject_name AS credit_class_name,
                    cc.credit_code AS credit_class_code
                FROM attendance a
                LEFT JOIN students s ON a.student_id = s.student_id
                LEFT JOIN classes c ON s.class_id = c.id
                LEFT JOIN credit_classes cc ON a.credit_class_id = cc.id
                WHERE a.student_id = ?
                ORDER BY a.attendance_date DESC, a.check_in_time DESC
                LIMIT ?
            ''', (student_id, limit))
            return [dict(r) for r in cursor.fetchall()]

    # === PHIÊN ĐIỂM DANH ===

    def expire_attendance_sessions(self):
        """Đánh dấu các phiên quá hạn là expired."""
        now = datetime.now().isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE attendance_sessions
                SET status = 'expired', closed_at = COALESCE(closed_at, CURRENT_TIMESTAMP)
                WHERE status = 'open'
                  AND checkin_deadline IS NOT NULL
                  AND checkin_deadline <= ?
            ''', (now,))
            conn.commit()
            return cursor.rowcount

    def get_current_open_session(self):
        """Lấy phiên điểm danh đang mở (chưa hết hạn)."""
        now = datetime.now().isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT ast.*, cc.subject_name AS credit_class_name, cc.credit_code
                FROM attendance_sessions ast
                LEFT JOIN credit_classes cc ON ast.credit_class_id = cc.id
                WHERE ast.status = 'open'
                  AND ast.closed_at IS NULL
                  AND (ast.checkin_deadline IS NULL OR ast.checkin_deadline > ?)
                ORDER BY ast.opened_at DESC
                LIMIT 1
            ''', (now,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_session_by_id(self, session_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT ast.*, cc.subject_name AS credit_class_name, cc.credit_code
                FROM attendance_sessions ast
                LEFT JOIN credit_classes cc ON ast.credit_class_id = cc.id
                WHERE ast.id = ?
            ''', (session_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
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

    def delete_class_by_name(self, class_name):
        """Xóa lớp theo tên (soft delete). Trả về số bản ghi bị ảnh hưởng."""
        if not class_name:
            return 0
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE classes
                SET is_active = 0, updated_at = CURRENT_TIMESTAMP
                WHERE class_name = ?
            ''', (class_name,))
            conn.commit()
            return cursor.rowcount
    
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

    # === QUẢN LÝ GIẢNG VIÊN ===

    def create_teacher(self, teacher_code, full_name, email=None, phone=None, department=None, user_id=None):
        """Tạo giảng viên mới."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO teachers (teacher_code, full_name, email, phone, department, user_id)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (teacher_code, full_name, email, phone, department, user_id))
            conn.commit()
            return cursor.lastrowid

    def get_teacher_by_code(self, teacher_code):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM teachers WHERE teacher_code = ? AND is_active = 1', (teacher_code,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_teacher(self, teacher_id):
        """Lấy thông tin giảng viên theo ID."""
        if not teacher_id:
            return None
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM teachers WHERE id = ? AND is_active = 1', (teacher_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_teacher_by_user(self, user_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM teachers WHERE user_id = ? AND is_active = 1', (user_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all_teachers(self, active_only=True):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if active_only:
                cursor.execute('SELECT * FROM teachers WHERE is_active = 1 ORDER BY full_name')
            else:
                cursor.execute('SELECT * FROM teachers ORDER BY full_name')
            return [dict(row) for row in cursor.fetchall()]

    def update_teacher(self, teacher_id, **kwargs):
        valid_fields = ['teacher_code', 'full_name', 'email', 'phone', 'department', 'user_id', 'is_active']
        updates = {k: v for k, v in kwargs.items() if k in valid_fields}
        if not updates:
            return False
        updates['updated_at'] = datetime.now().isoformat()
        set_clause = ', '.join(f"{k} = ?" for k in updates.keys())
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f'''UPDATE teachers SET {set_clause} WHERE id = ?''',
                list(updates.values()) + [teacher_id]
            )
            conn.commit()
            return cursor.rowcount > 0

    # === QUẢN LÝ LỚP TÍN CHỈ ===

    def create_credit_class(self, credit_code, subject_name, teacher_id=None, semester=None,
                             academic_year=None, room=None, schedule_info=None,
                             enrollment_limit=None, notes=None, status='draft'):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO credit_classes (
                    credit_code, subject_name, teacher_id, semester, academic_year, room,
                    schedule_info, status, enrollment_limit, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (credit_code, subject_name, teacher_id, semester, academic_year, room,
                  schedule_info, status, enrollment_limit, notes))
            conn.commit()
            return cursor.lastrowid

    def get_credit_class(self, credit_class_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM credit_classes WHERE id = ? AND is_active = 1', (credit_class_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_credit_class_by_code(self, credit_code):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM credit_classes WHERE credit_code = ? AND is_active = 1', (credit_code,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def list_credit_classes(self, teacher_id=None, active_only=True):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = 'SELECT * FROM credit_classes WHERE 1=1'
            params = []
            if active_only:
                query += ' AND is_active = 1'
            if teacher_id:
                query += ' AND teacher_id = ?'
                params.append(teacher_id)
            query += ' ORDER BY academic_year DESC, semester DESC, subject_name'
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def list_credit_classes_overview(self, teacher_id=None, active_only=True):
        """Trả về danh sách lớp tín chỉ kèm số sinh viên đã liên kết."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            base_query = [
                'SELECT cc.*,',
                '   (',
                '       SELECT COUNT(*)',
                '       FROM credit_class_students ccs',
                '       JOIN students s ON ccs.student_id = s.id',
                '       WHERE ccs.credit_class_id = cc.id',
                '         AND s.is_active = 1',
                '   ) AS student_count',
                'FROM credit_classes cc',
                'WHERE 1=1'
            ]
            params = []
            if active_only:
                base_query.append('AND cc.is_active = 1')
            if teacher_id:
                base_query.append('AND cc.teacher_id = ?')
                params.append(teacher_id)
            base_query.append('ORDER BY cc.academic_year DESC, cc.semester DESC, cc.subject_name')
            cursor.execute('\n'.join(base_query), params)
            return [dict(row) for row in cursor.fetchall()]

    def get_credit_classes_for_student(self, student_identifier, active_only=True):
        """Lấy danh sách lớp tín chỉ mà sinh viên đã đăng ký."""
        if not student_identifier:
            return []
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if isinstance(student_identifier, int):
                student_db_id = student_identifier
            else:
                cursor.execute('SELECT id FROM students WHERE student_id = ?', (student_identifier,))
                row = cursor.fetchone()
                if not row:
                    return []
                student_db_id = row[0] if not isinstance(row, sqlite3.Row) else row['id']

            query = [
                'SELECT cc.*,',
                '       ccs.enrollment_status,',
                '       ccs.joined_at,',
                '       (',
                '           SELECT COUNT(*)',
                '           FROM credit_class_students sub',
                '           JOIN students sub_s ON sub.student_id = sub_s.id',
                '           WHERE sub.credit_class_id = cc.id AND sub_s.is_active = 1',
                '       ) AS student_count',
                'FROM credit_class_students ccs',
                'JOIN credit_classes cc ON cc.id = ccs.credit_class_id',
                'WHERE ccs.student_id = ?'
            ]
            params = [student_db_id]
            if active_only:
                query.append('AND cc.is_active = 1')
            query.append('ORDER BY cc.academic_year DESC, cc.semester DESC, cc.subject_name')
            cursor.execute('\n'.join(query), params)
            return [dict(row) for row in cursor.fetchall()]

    def update_credit_class(self, credit_class_id, **kwargs):
        valid_fields = ['credit_code', 'subject_name', 'teacher_id', 'semester', 'academic_year',
                        'room', 'schedule_info', 'status', 'enrollment_limit', 'notes', 'is_active']
        updates = {k: v for k, v in kwargs.items() if k in valid_fields}
        if not updates:
            return False
        updates['updated_at'] = datetime.now().isoformat()
        set_clause = ', '.join(f"{k} = ?" for k in updates.keys())
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f'''UPDATE credit_classes SET {set_clause} WHERE id = ?''',
                list(updates.values()) + [credit_class_id]
            )
            conn.commit()
            return cursor.rowcount > 0

    def delete_credit_class(self, credit_class_id):
        return self.update_credit_class(credit_class_id, is_active=0)

    def _resolve_student_db_id(self, cursor, student_identifier):
        if isinstance(student_identifier, int):
            return student_identifier
        cursor.execute('SELECT id FROM students WHERE student_id = ?', (student_identifier,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Không tìm thấy sinh viên với mã {student_identifier}")
        return row['id'] if isinstance(row, sqlite3.Row) else row[0]

    def enroll_student_to_credit_class(self, credit_class_id, student_identifier):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            student_db_id = self._resolve_student_db_id(cursor, student_identifier)
            cursor.execute('''
                INSERT OR IGNORE INTO credit_class_students (credit_class_id, student_id)
                VALUES (?, ?)
            ''', (credit_class_id, student_db_id))
            conn.commit()
            return cursor.rowcount > 0

    def remove_student_from_credit_class(self, credit_class_id, student_identifier):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            student_db_id = self._resolve_student_db_id(cursor, student_identifier)
            cursor.execute('''
                DELETE FROM credit_class_students WHERE credit_class_id = ? AND student_id = ?
            ''', (credit_class_id, student_db_id))
            conn.commit()
            return cursor.rowcount > 0

    def clear_attendance_records(self):
        """Xóa toàn bộ bản ghi điểm danh để tránh xung đột dữ liệu cũ."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            table_names = self._table_names(cursor)
            cursor.execute('DELETE FROM attendance_records')
            cursor.execute('DELETE FROM attendance_sessions')
            if 'attendance_history' in table_names:
                cursor.execute('DELETE FROM attendance_history')
            conn.commit()

    def _table_names(self, cursor):
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return {row['name'] if isinstance(row, sqlite3.Row) else row[0] for row in cursor.fetchall()}

    def seed_sample_credit_classes(self):
        """Tạo dữ liệu mẫu cho lớp tín chỉ và ghi danh sinh viên thử nghiệm."""
        sample_classes = [
            {
                'credit_code': 'ICT1001',
                'subject_name': 'Nhập môn Lập trình',
                'semester': 'HK1',
                'academic_year': '2025-2026',
                'room': 'Lab A1.03',
                'schedule_info': 'Thứ 2 · Tiết 1-3',
                'notes': 'Lớp thực hành Python'
            },
            {
                'credit_code': 'BUS2003',
                'subject_name': 'Kỹ năng mềm cho kỹ sư',
                'semester': 'HK1',
                'academic_year': '2025-2026',
                'room': 'B203',
                'schedule_info': 'Thứ 4 · Tiết 4-6'
            },
            {
                'credit_code': 'AI3002',
                'subject_name': 'Thị giác máy tính ứng dụng',
                'semester': 'HK2',
                'academic_year': '2025-2026',
                'room': 'Lab AI',
                'schedule_info': 'Thứ 6 · Tiết 1-3',
                'notes': 'Điểm danh bằng camera DeepFace'
            },
        ]

        sample_students = [
            {
                'student_id': 'SV1001',
                'full_name': 'Nguyễn Minh An',
                'email': 'an.nguyen@example.com',
                'phone': '0901001001',
                'class_name': 'KTPM20A'
            },
            {
                'student_id': 'SV1002',
                'full_name': 'Trần Thị Bảo',
                'email': 'bao.tran@example.com',
                'phone': '0902002002',
                'class_name': 'KTPM20A'
            },
            {
                'student_id': 'SV1003',
                'full_name': 'Phạm Quốc Cường',
                'email': 'cuong.pham@example.com',
                'phone': '0903003003',
                'class_name': 'KTPM20B'
            },
        ]

        enrollments = {
            'ICT1001': ['SV1001', 'SV1002'],
            'BUS2003': ['SV1001', 'SV1003'],
            'AI3002': ['SV1002', 'SV1003'],
        }

        summary = {
            'classes_created': 0,
            'students_created': 0,
            'enrollments_created': 0,
        }

        teacher_id = None
        try:
            teacher_user = self.get_user_by_username('teacher')
            if teacher_user:
                teacher = self.get_teacher_by_user(teacher_user['id'])
                if teacher:
                    teacher_id = teacher.get('id')
                else:
                    teacher_id = self.create_teacher(
                        teacher_code='GV-DEMO',
                        full_name=teacher_user.get('full_name') or 'Giảng viên demo',
                        email=teacher_user.get('email'),
                        phone=teacher_user.get('phone'),
                        department='Demo',
                        user_id=teacher_user['id']
                    )
        except Exception as exc:
            logger.warning("Không thể tạo hồ sơ giảng viên mẫu: %s", exc)

        class_ids = {}
        for sample in sample_classes:
            existing = self.get_credit_class_by_code(sample['credit_code'])
            if existing:
                class_ids[sample['credit_code']] = existing['id']
                continue
            class_id = self.create_credit_class(
                credit_code=sample['credit_code'],
                subject_name=sample['subject_name'],
                teacher_id=teacher_id,
                semester=sample.get('semester'),
                academic_year=sample.get('academic_year'),
                room=sample.get('room'),
                schedule_info=sample.get('schedule_info'),
                enrollment_limit=sample.get('enrollment_limit'),
                notes=sample.get('notes'),
                status='published'
            )
            class_ids[sample['credit_code']] = class_id
            summary['classes_created'] += 1

        for student in sample_students:
            existing = self.get_student(student['student_id'])
            if existing:
                continue
            created = self.add_student(
                student['student_id'],
                student['full_name'],
                email=student.get('email'),
                phone=student.get('phone'),
                class_name=student.get('class_name')
            )
            if created:
                summary['students_created'] += 1

        for credit_code, student_ids in enrollments.items():
            credit_class_id = class_ids.get(credit_code)
            if not credit_class_id:
                continue
            for student_id in student_ids:
                try:
                    if self.enroll_student_to_credit_class(credit_class_id, student_id):
                        summary['enrollments_created'] += 1
                except ValueError as exc:
                    logger.debug("Không thể ghi danh %s vào %s: %s", student_id, credit_code, exc)

        logger.info(
            "Seeded credit classes: %(classes_created)s lớp, %(students_created)s sinh viên, %(enrollments_created)s ghi danh",
            summary
        )
        return summary

    def get_credit_class_students(self, credit_class_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT s.*, ccs.enrollment_status, ccs.joined_at
                FROM credit_class_students ccs
                JOIN students s ON ccs.student_id = s.id
                WHERE ccs.credit_class_id = ? AND s.is_active = 1
                ORDER BY s.full_name
            ''', (credit_class_id,))
            return [dict(row) for row in cursor.fetchall()]

    # === PHIÊN ĐIỂM DANH LỚP TÍN CHỈ ===

    def create_attendance_session(self, credit_class_id, opened_by, session_date=None,
                                  checkin_deadline=None, checkout_deadline=None, status='open', notes=None):
        session_date = session_date or date.today().isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Không cho mở phiên mới nếu đã có phiên đang mở
            cursor.execute('''
                SELECT id FROM attendance_sessions
                WHERE credit_class_id = ? AND status IN ('open', 'scheduled') AND closed_at IS NULL
            ''', (credit_class_id,))
            if cursor.fetchone():
                raise ValueError('Lớp đang có phiên điểm danh mở, vui lòng đóng trước khi tạo phiên mới')

            cursor.execute('''
                INSERT INTO attendance_sessions (
                    credit_class_id, opened_by, session_date, checkin_deadline, checkout_deadline, status, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (credit_class_id, opened_by, session_date, checkin_deadline, checkout_deadline, status, notes))
            conn.commit()
            return cursor.lastrowid

    def close_attendance_session(self, session_id, status='closed'):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE attendance_sessions
                SET status = ?, closed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (status, session_id))
            conn.commit()
            return cursor.rowcount > 0

    def get_active_session_for_class(self, credit_class_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM attendance_sessions
                WHERE credit_class_id = ? AND status IN ('open', 'scheduled') AND closed_at IS NULL
                ORDER BY opened_at DESC
                LIMIT 1
            ''', (credit_class_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def list_sessions_for_class(self, credit_class_id, limit=20):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM attendance_sessions
                WHERE credit_class_id = ?
                ORDER BY opened_at DESC
                LIMIT ?
            ''', (credit_class_id, limit))
            return [dict(row) for row in cursor.fetchall()]

# Khởi tạo instance global
db = DatabaseManager()
