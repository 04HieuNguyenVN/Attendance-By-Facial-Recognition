# attendance.py - Attendance marking and management logic

import threading
from datetime import datetime
from typing import Optional

from config import PRESENCE_TIMEOUT, RECOGNITION_COOLDOWN
from utils import serialize_session_payload, parse_datetime_safe
from database import db

# Global variables for attendance tracking
today_checked_in = set()
today_checked_out = set()
today_student_names = {}
today_recorded_lock = threading.Lock()

# Presence tracking
presence_tracking = {}
presence_tracking_lock = threading.Lock()

# Recognition cooldown
last_recognized = {}
last_recognized_lock = threading.Lock()

# Attendance progress for streaming
attendance_progress = {}
attendance_progress_lock = threading.Lock()


def load_today_recorded(session_id=None, credit_class_id=None, app_logger=None):
    """Load danh sách đã điểm danh hôm nay từ Database"""
    global today_checked_in, today_checked_out, today_student_names
    today_checked_in = set()
    today_checked_out = set()
    today_student_names = {}

    session_filter = session_id
    class_filter = credit_class_id
    if session_filter is None:
        from app import get_active_attendance_session
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
            today_student_names[student_id] = {
                'name': name or student_id,
                'class_name': class_name,
                'class_type': class_type,
                'credit_class_id': record_dict.get('credit_class_id')
            }
            if record_dict.get('check_in_time'):
                today_checked_in.add(student_id)
            if record_dict.get('check_out_time'):
                today_checked_out.add(student_id)
    except Exception as e:
        if app_logger:
            app_logger.error(f"Error loading today recorded: {e}")


def reset_session_runtime_state(session_row=None, app_logger=None):
    """Đặt lại cache điểm danh & tracking khi chuyển phiên."""
    session_id = session_row.get('id') if isinstance(session_row, dict) else None
    credit_class_id = session_row.get('credit_class_id') if isinstance(session_row, dict) else None
    load_today_recorded(session_id=session_id, credit_class_id=credit_class_id, app_logger=app_logger)
    with presence_tracking_lock:
        presence_tracking.clear()
    with attendance_progress_lock:
        attendance_progress.clear()


def mark_attendance(
    name: str,
    student_id: str = '',
    confidence_score: float = None,
    expected_student_id: str = None,
    expected_credit_class_id: int = None,
    app_logger=None,
    broadcast_func=None,
) -> bool:
    """Lưu điểm danh vào database với các ràng buộc tùy chọn."""
    from app import get_active_attendance_session

    normalized_student_id = (student_id or '').strip().upper()
    normalized_expected_id = (expected_student_id or '').strip().upper()
    if normalized_expected_id and normalized_student_id and normalized_student_id != normalized_expected_id:
        if app_logger:
            app_logger.info(
                "[Attendance] Rejecting check-in: recognized %s but expected %s",
                normalized_student_id,
                normalized_expected_id,
            )
        return False
    with today_recorded_lock:
        already_checked_in = normalized_student_id in today_checked_in
        already_checked_out = normalized_student_id in today_checked_out
        if already_checked_in and not already_checked_out:
            if app_logger:
                app_logger.info(f"Sinh vien {name} da check-in va chua checkout")
            return False

    session_ctx = get_active_attendance_session()
    credit_class_id = session_ctx.get('credit_class_id') if session_ctx else None
    session_id = session_ctx.get('id') if session_ctx else None

    if expected_credit_class_id is not None:
        if not session_ctx or int(credit_class_id or 0) != int(expected_credit_class_id):
            if app_logger:
                app_logger.info(
                    "[Attendance] Rejecting check-in for %s: session mismatch (expected class %s, active %s)",
                    normalized_student_id or name,
                    expected_credit_class_id,
                    credit_class_id,
                )
            return False

    success = db.mark_attendance(
        student_id=normalized_student_id or student_id,
        student_name=name,
        status='present',
        confidence_score=confidence_score,
        notes=None,
        credit_class_id=credit_class_id,
        session_id=session_id
    )
    if app_logger:
        app_logger.info(f"[DEBUG] Mark attendance success: {success}, session_id: {session_id}")

    # Tự động enroll sinh viên vào lớp tín chỉ nếu chưa enroll và điểm danh thành công
    if success and credit_class_id:
        try:
            # Kiểm tra xem đã enroll chưa
            student_db_id = db.get_student_id_by_student_id(normalized_student_id)
            if student_db_id:
                enrolled = db.check_student_enrolled_in_credit_class(student_db_id, credit_class_id)
                if not enrolled:
                    db.enroll_student_in_credit_class(student_db_id, credit_class_id)
                    if app_logger:
                        app_logger.info(f"[Attendance] Auto-enrolled {normalized_student_id} into credit class {credit_class_id}")
        except Exception as e:
            if app_logger:
                app_logger.warning(f"[Attendance] Failed to auto-enroll {normalized_student_id}: {e}")

    if success:
        session_payload = serialize_session_payload(session_ctx)
        with today_recorded_lock:
            today_checked_in.add(normalized_student_id)
            today_checked_out.discard(normalized_student_id)
            existing_info = today_student_names.get(normalized_student_id)
            class_name = None
            class_type = None
            credit_ctx = credit_class_id
            if isinstance(existing_info, dict):
                class_name = existing_info.get('class_name')
                class_type = existing_info.get('class_type')
                credit_ctx = existing_info.get('credit_class_id', credit_ctx)
            if not class_name and session_payload:
                class_name = session_payload.get('class_name') or session_payload.get('class_code')
            if session_payload:
                class_type = 'credit'
                credit_ctx = session_payload.get('credit_class_id')
            today_student_names[normalized_student_id] = {
                'name': name,
                'class_name': class_name,
                'class_type': class_type or 'administrative',
                'credit_class_id': credit_ctx
            }
        # Khởi tạo presence tracking
        with presence_tracking_lock:
            presence_tracking[normalized_student_id] = {
                'last_seen': datetime.now(),
                'check_in_time': datetime.now(),
                'name': name
            }
        if app_logger:
            app_logger.info(
                f"Da danh dau diem danh: {name} (id={normalized_student_id or student_id}, confidence={confidence_score})"
            )

        if broadcast_func:
            broadcast_func({
                'type': 'attendance_marked',
                'data': {
                    'event': 'check_in',
                    'student_id': normalized_student_id or student_id,
                    'student_name': name,
                    'confidence': confidence_score,
                    'timestamp': datetime.now().isoformat(),
                    'session': session_payload
                }
            })

    return success


def mark_student_checkout(
    student_id: str,
    student_name: str = '',
    reason: str = 'manual',
    confidence_score: float = None,
    expected_student_id: str = None,
    expected_credit_class_id: int = None,
    app_logger=None,
    broadcast_func=None,
) -> bool:
    """Đánh dấu checkout cho sinh viên với ràng buộc khuôn mặt/sessions tùy chọn."""
    from app import get_active_attendance_session

    normalized_student_id = (student_id or '').strip()
    normalized_expected_id = (expected_student_id or '').strip()
    if normalized_expected_id and normalized_student_id and normalized_student_id != normalized_expected_id:
        if app_logger:
            app_logger.info(
                "[Attendance] Rejecting checkout: recognized %s but expected %s",
                normalized_student_id,
                normalized_expected_id,
            )
        return False
    with today_recorded_lock:
        already_checked_in = normalized_student_id in today_checked_in
        already_checked_out = normalized_student_id in today_checked_out

    if not already_checked_in or already_checked_out:
        return False

    session_ctx = get_active_attendance_session()
    credit_class_id = session_ctx.get('credit_class_id') if session_ctx else None
    if expected_credit_class_id is not None:
        if not session_ctx or int(credit_class_id or 0) != int(expected_credit_class_id):
            if app_logger:
                app_logger.info(
                    "[Attendance] Rejecting checkout for %s: session mismatch (expected class %s, active %s)",
                    normalized_student_id or student_id,
                    expected_credit_class_id,
                    credit_class_id,
                )
            return False

    success = db.mark_checkout(
        normalized_student_id or student_id,
        session_id=session_ctx.get('id') if session_ctx else None,
    )
    if not success:
        return False

    existing_info = today_student_names.get(normalized_student_id)
    if isinstance(existing_info, dict):
        resolved_name = student_name or existing_info.get('name') or student_id
    else:
        resolved_name = student_name or existing_info or student_id
    with today_recorded_lock:
        today_checked_out.add(normalized_student_id)
        today_student_names[normalized_student_id] = {
            'name': resolved_name,
            'class_name': existing_info.get('class_name') if isinstance(existing_info, dict) else None
        }

    with presence_tracking_lock:
        presence_tracking.pop(normalized_student_id, None)

    if broadcast_func:
        broadcast_func({
            'type': 'attendance_checkout',
            'data': {
                'event': 'check_out',
                'student_id': normalized_student_id or student_id,
                'student_name': resolved_name,
                'confidence': confidence_score,
                'reason': reason,
                'timestamp': datetime.now().isoformat(),
                'session': serialize_session_payload(get_active_attendance_session())
            }
        })

    if app_logger:
        app_logger.info(f"Da checkout: {resolved_name} (id={normalized_student_id or student_id}) - reason={reason}")
    return True


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


def check_presence_timeout(app_logger=None, checkout_func=None):
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
            if checkout_func:
                checkout_func(student_id, student_name=student_name, reason='timeout')
            presence_tracking.pop(student_id, None)


def get_today_attendance(credit_class_id=None, session_id=None, app_logger=None):
    """Lấy danh sách điểm danh hôm nay từ database với bộ lọc tùy chọn."""
    try:
        from app import get_active_attendance_session

        resolved_class_id = credit_class_id
        resolved_session_id = session_id

        if session_id:
            session_row = db.get_session_by_id(session_id)
            if session_row:
                resolved_class_id = resolved_class_id or session_row.get('credit_class_id')

        if resolved_class_id and not resolved_session_id:
            session_row = db.get_active_session_for_class(resolved_class_id)
            if session_row:
                resolved_session_id = session_row.get('id')

        if resolved_class_id is None and resolved_session_id is None:
            session_row = get_active_attendance_session()
            if session_row:
                resolved_session_id = session_row.get('id')
                resolved_class_id = session_row.get('credit_class_id')

        attendance_data = db.get_today_attendance(
            session_id=resolved_session_id,
            credit_class_id=resolved_class_id,
        )
        # Chuyển đổi đối tượng SQLite Row thành dict
        results = []
        now = datetime.now()

        for row in attendance_data:
            # Tính thời gian có mặt
            duration_minutes = 0
            status_text = "Đang có mặt"

            check_in = parse_datetime_safe(row['check_in_time'])
            check_out = parse_datetime_safe(row['check_out_time'])
            row_credit_class_id = row.get('credit_class_id')
            credit_class_name = row.get('credit_class_name')
            credit_class_code = row.get('credit_class_code')
            class_type = 'credit' if row_credit_class_id else 'administrative'
            base_class_name = row.get('class_name')
            class_display = credit_class_name or base_class_name
            if row_credit_class_id:
                label_parts = [credit_class_name, credit_class_code]
                class_display = ' · '.join([part for part in label_parts if part]) or class_display

            if check_in is None:
                if app_logger:
                    app_logger.warning(
                        "Attendance row is missing check-in time", extra={"student_id": row['student_id']}
                    )
                continue

            if check_out:
                # Đã checkout
                duration_seconds = max((check_out - check_in).total_seconds(), 0)
                status_text = "Đã rời"
            else:
                # Chưa checkout - tính từ check_in đến hiện tại
                duration_seconds = max((now - check_in).total_seconds(), 0)

                # Kiểm tra xem có đang được tracking không
                with presence_tracking_lock:
                    if row['student_id'] not in presence_tracking:
                        status_text = "Không còn phát hiện"

            duration_minutes = int(duration_seconds / 60)

            timestamp_value = check_in.isoformat()
            checkout_value = check_out.isoformat() if check_out else None

            results.append({
                'student_id': row['student_id'],
                'full_name': row['student_name'],
                'class_name': base_class_name,
                'class_display': class_display,
                'class_type': class_type,
                'credit_class_id': row_credit_class_id,
                'credit_class_code': credit_class_code,
                'credit_class_name': credit_class_name,
                'session_id': row.get('session_id'),
                'timestamp': timestamp_value,
                'checkout_time': checkout_value,
                'date': row['attendance_date'],
                'duration_minutes': duration_minutes,
                'status': status_text
            })
        return results
    except Exception as e:
        if app_logger:
            app_logger.error(f"Error getting today attendance: {e}")
        return []


def update_progress(student_id, name, app_logger=None):
    """Cập nhật tiến độ xác nhận điểm danh"""
    with attendance_progress_lock:
        if student_id not in attendance_progress:
            attendance_progress[student_id] = {
                'name': name,
                'progress': 0,
                'last_update': datetime.now(),
                'status': 'starting'
            }

        progress_data = attendance_progress[student_id]
        progress_data['progress'] = min(100, progress_data['progress'] + 20)
        progress_data['last_update'] = datetime.now()

        if progress_data['progress'] >= 100:
            progress_data['status'] = 'complete'
        elif progress_data['progress'] >= 50:
            progress_data['status'] = 'confirming'
        else:
            progress_data['status'] = 'detecting'


def reset_progress(student_id):
    """Đặt lại tiến độ cho sinh viên"""
    with attendance_progress_lock:
        attendance_progress.pop(student_id, None)


def draw_progress_bar(frame, progress, x, y, w=150, h=20):
    """Vẽ thanh tiến độ trên frame"""
    import cv2

    # Vẽ nền thanh tiến độ
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), -1)
    cv2.rectangle(frame, (x + 2, y + 2), (x + w - 2, y + h - 2), (255, 255, 255), 1)

    # Vẽ phần đã hoàn thành
    progress_width = int((w - 4) * (progress / 100))
    if progress_width > 0:
        cv2.rectangle(frame, (x + 2, y + 2), (x + 2 + progress_width, y + h - 2), (0, 255, 0), -1)

    # Vẽ text phần trăm
    text = f"{progress}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 0.5, 1)[0]
    text_x = x + (w - text_size[0]) // 2
    text_y = y + (h + text_size[1]) // 2
    cv2.putText(frame, text, (text_x, text_y), font, 0.5, (255, 255, 255), 1)