"""
Attendance utilities
Các hàm tiện ích cho điểm danh
"""
from datetime import datetime
from flask import current_app
from database import db
from app.utils.session_utils import (
    parse_datetime_safe, 
    get_active_attendance_session,
    serialize_session_payload
)
from app import globals as app_globals


def get_today_attendance(credit_class_id=None, session_id=None):
    """Lấy danh sách điểm danh hôm nay từ database với bộ lọc tùy chọn."""
    try:
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
                current_app.logger.warning(
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
                with app_globals.presence_tracking_lock:
                    if row['student_id'] not in app_globals.presence_tracking:
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
        current_app.logger.error(f"Error getting today attendance: {e}")
        return []
