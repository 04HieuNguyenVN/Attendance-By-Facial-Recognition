# utils.py - Utility functions for the attendance system

import os
import re
import shutil
import base64
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
from werkzeug.utils import secure_filename
from flask import request

from config import DATA_DIR, ALLOWED_EXTENSIONS


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


def iter_student_face_image_files():
    """Duyệt qua tất cả ảnh mẫu sinh viên (bao gồm thư mục con)."""
    from config import RESERVED_DATA_SUBDIRS

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
        print(f"[DataReorg] Không thể di chuyển {source_path} -> {target_path}: {exc}")
        return None

    return str(target_path)


def _infer_student_id_from_filename(filename: str) -> Optional[str]:
    if not filename:
        return None
    match = re.match(r'^([A-Za-z0-9]+)', filename)
    if match:
        return match.group(1)
    return None


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
        payload['time_remaining'] = max(0, int((expires_at - datetime.now()).total_seconds()))
    else:
        payload['time_remaining'] = None
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
        return None


def get_current_role():
    from flask import g
    user = getattr(g, 'user', None)
    if not user:
        return None
    return (user.get('role') or '').lower()


def resolve_teacher_context(teacher_id=None):
    """Xác định bản ghi giảng viên tương ứng với người dùng hiện tại."""
    from flask import g
    from database import db

    user = getattr(g, 'user', None)
    if not user:
        return None
    role = get_current_role()
    if role == 'teacher':
        teacher_id = teacher_id or user.get('teacher_id')
        if teacher_id:
            return db.get_teacher(teacher_id)
    if role == 'admin' and teacher_id:
        return db.get_teacher(teacher_id)
    return None


def resolve_student_context(student_identifier=None, auto_link=True):
    """Tìm sinh viên cho user hiện tại hoặc theo student_id được cung cấp."""
    from flask import g
    from database import db

    user = getattr(g, 'user', None)
    role = get_current_role()

    if student_identifier:
        student = db.get_student_by_id(student_identifier)
        if student:
            return student

    if not user:
        return None

    if role == 'student':
        student_id = user.get('student_id')
        if student_id:
            student = db.get_student_by_id(student_id)
            if student:
                return student

        # Nếu chưa link, thử tự động link bằng email
        if auto_link and user.get('email'):
            students = db.get_students_by_email(user['email'])
            if len(students) == 1:
                student = students[0]
                # Tự động link tài khoản
                db.link_user_to_student(user['id'], student['id'])
                return student

    if role == 'admin' and student_identifier:
        return db.get_student_by_id(student_identifier)

    return None


def sanitize_next_url(next_url):
    """Đảm bảo next_url luôn là đường dẫn nội bộ an toàn."""
    if not next_url:
        return '/'
    next_url = next_url.strip()
    if not next_url:
        return '/'
    if next_url.startswith(('http://', 'https://', '//')):
        return '/'
    if not next_url.startswith('/'):
        next_url = '/' + next_url
    return next_url.rstrip('?') or '/'


def build_next_url():
    """Tạo giá trị next_url dựa trên request hiện tại."""
    from flask import request

    if request.method == 'GET':
        return sanitize_next_url(request.args.get('next'))
    else:
        return sanitize_next_url(request.form.get('next'))


def is_api_request():
    """Kiểm tra request hiện tại có thuộc API không."""
    from flask import request

    path = request.path or ''
    return path.startswith('/api/')


def is_public_endpoint(endpoint):
    """Xác định endpoint có được phép truy cập công khai hay không."""
    from config import PUBLIC_ENDPOINTS

    if not endpoint:
        return False
    if endpoint == 'static' or endpoint.startswith('static.'):
        return True
    return endpoint in PUBLIC_ENDPOINTS


def verify_user_password(user_record, candidate_password):
    """Kiểm tra mật khẩu người dùng (hỗ trợ hash legacy)."""
    import hashlib
    from werkzeug.security import check_password_hash

    if not user_record:
        return False
    stored_hash = user_record.get('password_hash') or ''
    if not stored_hash:
        return False

    if stored_hash.startswith(('pbkdf2:', 'scrypt:')):
        return check_password_hash(stored_hash, candidate_password)

    legacy_hash = hashlib.sha256(candidate_password.encode('utf-8')).hexdigest()
    if legacy_hash == stored_hash:
        # Nâng cấp hash
        from werkzeug.security import generate_password_hash
        new_hash = generate_password_hash(candidate_password)
        from database import db
        db.update_user_password(user_record['id'], new_hash)
        return True

    return False


def login_user(user_record):
    """Thiết lập session cho người dùng đã xác thực."""
    from flask import session

    session.clear()
    session['user_id'] = user_record['id']
    session['user_role'] = user_record.get('role')
    session['user_name'] = user_record.get('full_name')
    session.permanent = True


def logout_current_user():
    """Đăng xuất người dùng hiện tại."""
    from flask import session

    session.clear()


def role_required(*roles):
    """Decorator kiểm tra quyền truy cập dựa trên vai trò."""
    from flask import g, abort, request, redirect, url_for, flash
    from functools import wraps

    allowed_roles = {role.lower() for role in roles if role}

    def decorator(view_func):
        @wraps(view_func)
        def wrapper(*args, **kwargs):
            user = getattr(g, 'user', None)
            if not user:
                if is_api_request():
                    abort(401)
                next_url = build_next_url()
                return redirect(url_for('login', next=next_url))

            user_role = (user.get('role') or '').lower()
            if user_role not in allowed_roles:
                if is_api_request():
                    abort(403)
                flash('Bạn không có quyền truy cập trang này.', 'error')
                return redirect(url_for('index'))

            return view_func(*args, **kwargs)
        return wrapper
    return decorator


def safe_delete_file(path):
    """Cố gắng xóa một file mà không báo lỗi nếu thất bại."""
    if not path:
        return
    try:
        os.remove(path)
    except OSError:
        pass


def _generate_face_image_filename(student_id, full_name, *, suffix=None, extension='jpg', timestamp=None):
    safe_base = secure_filename(f"{student_id}_{full_name}".strip()) or secure_filename(student_id) or 'student'
    timestamp = timestamp or datetime.now().strftime('%Y%m%d%H%M%S')
    suffix_part = f"_{suffix}" if suffix is not None else ''
    return f"{safe_base}_{timestamp}{suffix_part}.{extension}"


def save_uploaded_face_image(file_storage, student_id, full_name, *, suffix=None, timestamp=None):
    """Lưu ảnh khuôn mặt đã tải lên sau khi xác thực."""
    from config import ALLOWED_EXTENSIONS

    if not file_storage or not file_storage.filename:
        raise ValueError("No file provided")

    _, ext = os.path.splitext(file_storage.filename)
    ext = (ext or '').lower().lstrip('.')
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {ext}")

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
        raise ValueError(error_msg)

    return str(file_path)


def save_base64_face_image(image_data, student_id, full_name, *, suffix=None, timestamp=None):
    """Giải mã ảnh base64 và lưu xuống đĩa sau khi xác thực."""
    if not image_data:
        raise ValueError("No image data provided")

    if ',' in image_data:
        image_data = image_data.split(',')[1]
    try:
        image_bytes = base64.b64decode(image_data)
    except Exception as exc:
        raise ValueError(f"Invalid base64 data: {exc}")

    filename = _generate_face_image_filename(
        student_id,
        full_name,
        suffix=suffix,
        extension='jpg',
        timestamp=timestamp,
    )
    file_path = build_student_image_path(student_id, filename)
    with open(file_path, 'wb') as fp:
        fp.write(image_bytes)

    success, error_msg, _ = validate_image_file(str(file_path), is_base64=False)
    if not success:
        safe_delete_file(str(file_path))
        raise ValueError(error_msg)

    return str(file_path)


def validate_image_file(file_path, is_base64=False):
    """
    Validate ảnh trước khi lưu vào hệ thống

    Args:
        file_path: Đường dẫn file ảnh hoặc base64 data
        is_base64: True nếu file_path là base64 data

    Returns:
        tuple: (success: bool, error_message: str, face_count: int)
    """
    from config import MIN_FILE_SIZE, MAX_FILE_SIZE, SUPPORTED_IMAGE_FORMATS
    import cv2

    try:
        if is_base64:
            if ',' in file_path:
                file_path = file_path.split(',')[1]
            image_bytes = base64.b64decode(file_path)
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            if not os.path.exists(file_path):
                return False, "File does not exist", 0

            file_size = os.path.getsize(file_path)
            if file_size < MIN_FILE_SIZE:
                return False, f"File too small: {file_size} bytes (minimum {MIN_FILE_SIZE})", 0
            if file_size > MAX_FILE_SIZE:
                return False, f"File too large: {file_size} bytes (maximum {MAX_FILE_SIZE})", 0

            img = cv2.imread(file_path)
            if img is None:
                return False, "Could not read image file", 0

        # Kiểm tra format
        if hasattr(img, 'mode'):
            # PIL Image
            if img.mode not in SUPPORTED_IMAGE_FORMATS:
                return False, f"Unsupported image format: {img.mode}", 0
        else:
            # NumPy array (OpenCV)
            if len(img.shape) != 3 or img.shape[2] != 3:
                return False, "Image must be RGB format", 0

        # Phát hiện khuôn mặt
        face_count = 0
        try:
            # Thử dùng YOLO nếu có
            global yolo_face_model, YOLO_AVAILABLE
            if YOLO_AVAILABLE and yolo_face_model:
                results = yolo_face_model(img, conf=0.3, verbose=False)
                face_count = len(results[0].boxes) if results and len(results) > 0 else 0
            else:
                # Fallback to OpenCV Haar cascades
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                face_count = len(faces)
        except Exception as e:
            return False, f"Face detection failed: {str(e)}", 0

        if face_count == 0:
            return False, "No face detected in image", 0
        elif face_count > 1:
            return False, f"Multiple faces detected: {face_count} (only single face allowed)", 0

        return True, "", face_count

    except Exception as e:
        return False, f"Image validation error: {str(e)}", 0


def serialize_student_record(student_row, class_map=None):
    """Chuyển đổi bản ghi sinh viên sqlite3.Row thành dict có thể serialize."""
    if not student_row:
        return None

    student = dict(student_row)
    class_id = student.get('class_id')
    class_name = None
    if class_id:
        from database import db
        class_info = db.get_class(class_id)
        if class_info:
            class_name = class_info.get('name')

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
        from database import db
        teacher = db.get_teacher(payload['teacher_id'])
        if teacher:
            payload['teacher_name'] = teacher.get('full_name')
    return payload


def serialize_teacher_record(teacher_row):
    """Chuẩn hóa bản ghi giảng viên, kèm thông tin tài khoản người dùng."""
    if not teacher_row:
        return None

    teacher = dict(teacher_row)
    teacher['is_active'] = bool(teacher.get('is_active', 1))
    user_id = teacher.get('user_id')
    if user_id:
        from database import db
        user = db.get_user(user_id)
        if user:
            teacher['user_email'] = user.get('email')
            teacher['user_role'] = user.get('role')
    return teacher


def lookup_student_name(student_id: Optional[str]) -> Optional[str]:
    if not student_id:
        return None
    try:
        from database import db
        student = db.get_student_by_id(student_id)
        return student.get('full_name') if student else None
    except Exception as exc:
        return None


def allowed_file(filename):
    from config import ALLOWED_EXTENSIONS
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS