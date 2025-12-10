"""
Data utilities
Helper functions cho data transformation và validation
"""
from datetime import datetime
from flask import request


def row_to_dict(row):
    """Chuyển đổi sqlite3.Row thành dict."""
    if row is None:
        return None
    return dict(row)


def parse_datetime_safe(value):
    """
    Phân tích chuỗi datetime thành đối tượng datetime hoặc trả về None.
    """
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d']:
        try:
            return datetime.strptime(value, fmt)
        except (ValueError, TypeError):
            continue
    return None


def get_request_data():
    """Lấy request data từ JSON hoặc form."""
    if request.is_json:
        return request.get_json() or {}
    return request.form.to_dict()


def parse_bool(value, default=None):
    """
    Phân tích giá trị boolean từ string, int, hoặc bool.
    Returns: True, False, hoặc default nếu không xác định được.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in ('true', '1', 'yes', 'on'):
            return True
        if lower in ('false', '0', 'no', 'off'):
            return False
    return default


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
            from database import db
            class_obj = db.get_class_by_id(class_id)
            class_name = class_obj.get('class_name') if class_obj else None
    
    student['class_name'] = class_name
    return student


def get_current_role():
    """Lấy role của user hiện tại."""
    from flask import g
    user = getattr(g, 'user', None)
    if not user:
        return None
    return (user.get('role') or '').lower()
