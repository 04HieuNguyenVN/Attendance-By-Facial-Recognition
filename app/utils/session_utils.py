"""
Session utilities
Các hàm tiện ích cho quản lý phiên điểm danh
"""
from datetime import datetime
from flask import g
from database import db
from app import globals as app_globals


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
    from flask import current_app
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
                    current_app.logger.debug("Không thể tự liên kết sinh viên %s với user %s: %s", username, user['id'], exc)
            return row_to_dict(student_row)

    if role == 'admin' and student_identifier:
        return row_to_dict(db.get_student(student_identifier))

    return None


def get_active_attendance_session(force_reload=False):
    """Trả về phiên điểm danh đang mở (và cập nhật cache khi cần)."""
    with app_globals.current_session_lock:
        if force_reload:
            app_globals.current_credit_session = None
        else:
            if app_globals.current_credit_session and not session_is_active(app_globals.current_credit_session):
                app_globals.current_credit_session = None

        try:
            db.expire_attendance_sessions()
        except Exception:
            pass

        if app_globals.current_credit_session is None:
            session_row = db.get_current_open_session()
            app_globals.current_credit_session = session_row if session_row else None

        if app_globals.current_credit_session and not session_is_active(app_globals.current_credit_session):
            app_globals.current_credit_session = None

        return app_globals.current_credit_session


def set_active_session_cache(session_row):
    """Ghi đè cache phiên hiện tại."""
    with app_globals.current_session_lock:
        app_globals.current_credit_session = session_row
        return app_globals.current_credit_session


def broadcast_session_snapshot(force_reload=False):
    """Phát sự kiện SSE về trạng thái phiên điểm danh hiện tại."""
    from app.utils.sse_utils import broadcast_sse_event
    payload = serialize_session_payload(get_active_attendance_session(force_reload=force_reload))
    broadcast_sse_event({'type': 'session_updated', 'data': payload})


def load_today_recorded(session_id=None, credit_class_id=None):
    app_globals.today_checked_in = set()
    app_globals.today_checked_out = set()
    app_globals.today_student_names = {}

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
            app_globals.today_student_names[student_id] = {
                'name': name or student_id,
                'class_name': class_name,
                'class_type': class_type,
                'credit_class_id': record_dict.get('credit_class_id')
            }
            if record_dict.get('check_in_time'):
                app_globals.today_checked_in.add(student_id)
            if record_dict.get('check_out_time'):
                app_globals.today_checked_out.add(student_id)
    except Exception:
        pass


def reset_session_runtime_state(session_row=None):
    """Đặt lại cache điểm danh & tracking khi chuyển phiên."""
    session_id = session_row.get('id') if isinstance(session_row, dict) else None
    credit_class_id = session_row.get('credit_class_id') if isinstance(session_row, dict) else None
    load_today_recorded(session_id=session_id, credit_class_id=credit_class_id)
    with app_globals.presence_tracking_lock:
        app_globals.presence_tracking.clear()
    with app_globals.attendance_progress_lock:
        app_globals.attendance_progress.clear()


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
