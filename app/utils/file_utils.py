"""
File utilities
Xử lý file upload, validation và storage
"""
import os
import base64
from datetime import datetime
from pathlib import Path
from werkzeug.utils import secure_filename
from flask import current_app as app
from app.config import ALLOWED_EXTENSIONS, MIN_FILE_SIZE, MAX_FILE_SIZE

# Try to import face_recognition library for encoding extraction
# Use sys.modules to avoid name collision with local face_recognition.py
try:
    import sys
    import importlib
    
    # Temporarily remove current directory from module search to avoid local file collision
    original_path = sys.path.copy()
    workspace_root = str(Path(__file__).parent.parent.parent)
    if workspace_root in sys.path:
        sys.path.remove(workspace_root)
    
    face_recognition = importlib.import_module('face_recognition')
    
    # Restore original path
    sys.path = original_path
    
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    face_recognition = None


def safe_delete_file(path):
    """Cố gắng xóa một file mà không báo lỗi nếu thất bại."""
    if not path:
        return
    try:
        os.remove(path)
    except OSError:
        app.logger.debug("Could not remove file %s", path)


def _generate_face_image_filename(student_id, full_name, *, suffix=None, extension='jpg', timestamp=None):
    """Tạo tên file ảnh khuôn mặt an toàn."""
    safe_base = secure_filename(f"{student_id}_{full_name}".strip()) or secure_filename(student_id) or 'student'
    timestamp = timestamp or datetime.now().strftime('%Y%m%d%H%M%S')
    suffix_part = f"_{suffix}" if suffix is not None else ''
    return f"{safe_base}_{timestamp}{suffix_part}.{extension}"


def build_student_image_path(student_id, filename):
    """Tạo đường dẫn đầy đủ cho file ảnh sinh viên."""
    from app.config import FACE_DATA_DIR
    student_dir = os.path.join(FACE_DATA_DIR, str(student_id))
    os.makedirs(student_dir, exist_ok=True)
    return os.path.join(student_dir, filename)


def validate_image_file(file_path, is_base64=False):
    """
    Xác thực file ảnh.
    Returns: (success: bool, error_message: str, image_data: bytes or None)
    """
    try:
        if not os.path.exists(file_path):
            return False, "File không tồn tại", None
        
        file_size = os.path.getsize(file_path)
        if file_size < MIN_FILE_SIZE:
            return False, f"File quá nhỏ (tối thiểu {MIN_FILE_SIZE} bytes)", None
        if file_size > MAX_FILE_SIZE:
            return False, f"File quá lớn (tối đa {MAX_FILE_SIZE} bytes)", None
        
        # Kiểm tra định dạng ảnh
        try:
            from PIL import Image
            with Image.open(file_path) as img:
                img.verify()
            return True, "", None
        except Exception as e:
            return False, f"Ảnh không hợp lệ: {str(e)}", None
            
    except Exception as e:
        return False, f"Lỗi xác thực file: {str(e)}", None


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
