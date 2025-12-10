"""
API routes for attendance
Các API endpoint cho quản lý điểm danh
"""
from flask import Blueprint, jsonify, request, g, current_app
from datetime import datetime, timedelta
from database import db
from app.middleware.auth import role_required
from app.utils import get_request_data
from app.utils.session_utils import (
    serialize_session_payload,
    get_active_attendance_session,
    set_active_session_cache,
    reset_session_runtime_state,
    broadcast_session_snapshot,
    resolve_teacher_context,
    get_current_role
)
from app.utils.sse_utils import broadcast_sse_event
from app.utils.attendance_utils import get_today_attendance
from app import globals as app_globals
import os

attendance_api_bp = Blueprint('attendance_api', __name__, url_prefix='/api/attendance')

# Lấy SESSION_DURATION_MINUTES từ env
SESSION_DURATION_MINUTES = max(1, int(os.getenv('SESSION_DURATION_MINUTES', '15')))


@attendance_api_bp.route('/session', methods=['GET'])
def api_get_attendance_session():
    """Trạng thái phiên điểm danh tín chỉ hiện tại."""
    session_row = get_active_attendance_session()
    return jsonify({
        'success': True,
        'session': serialize_session_payload(session_row),
        'default_duration': SESSION_DURATION_MINUTES,
    })


@attendance_api_bp.route('/session/open', methods=['POST'])
@role_required('teacher')
def api_open_attendance_session():
    data = get_request_data()
    credit_class_id = data.get('credit_class_id')
    if not credit_class_id:
        return jsonify({'success': False, 'message': 'Vui lòng chọn lớp tín chỉ'}), 400

    try:
        credit_class_id = int(credit_class_id)
    except (ValueError, TypeError):
        return jsonify({'success': False, 'message': 'Mã lớp tín chỉ không hợp lệ'}), 400

    active_session = get_active_attendance_session()
    if active_session:
        active_class_id = active_session.get('credit_class_id')
        if active_class_id and int(active_class_id) != int(credit_class_id):
            return jsonify({'success': False, 'message': 'Đã có phiên điểm danh đang mở'}), 400
        try:
            db.close_attendance_session(active_session['id'], status='superseded')
            set_active_session_cache(None)
            broadcast_session_snapshot(force_reload=True)
        except Exception as exc:
            current_app.logger.warning(
                "Không thể tự động đóng phiên cũ %s: %s",
                active_session.get('id'),
                exc,
            )
            return jsonify({'success': False, 'message': 'Không thể đóng phiên cũ'}), 500

    teacher_ctx = resolve_teacher_context()
    if not teacher_ctx:
        return jsonify({'success': False, 'message': 'Không tìm thấy thông tin giảng viên'}), 403

    credit_class = db.get_credit_class(credit_class_id)
    if not credit_class or not credit_class.get('is_active', 1):
        return jsonify({'success': False, 'message': 'Không tìm thấy lớp tín chỉ'}), 404

    owner_id = credit_class.get('teacher_id')
    if not owner_id or int(owner_id) != int(teacher_ctx.get('id')):
        return jsonify({'success': False, 'message': 'Lớp tín chỉ không thuộc quyền quản lý của bạn'}), 403

    duration_minutes = data.get('duration_minutes')
    try:
        duration_minutes = int(duration_minutes) if duration_minutes is not None else SESSION_DURATION_MINUTES
    except (ValueError, TypeError):
        duration_minutes = SESSION_DURATION_MINUTES
    duration_minutes = max(1, min(duration_minutes, 90))

    now = datetime.now()
    deadline = (now + timedelta(minutes=duration_minutes)).isoformat()

    try:
        session_id = db.create_attendance_session(
            credit_class_id=credit_class_id,
            opened_by=g.user['id'] if getattr(g, 'user', None) else None,
            session_date=now.date().isoformat(),
            checkin_deadline=deadline,
            checkout_deadline=deadline,
            status='open',
            notes=data.get('notes')
        )
        session_row = db.get_session_by_id(session_id)
        set_active_session_cache(session_row)
        reset_session_runtime_state(session_row)
        payload = serialize_session_payload(session_row)
        broadcast_session_snapshot()
        return jsonify({'success': True, 'session': payload})
    except ValueError as err:
        return jsonify({'success': False, 'message': str(err)}), 400
    except Exception as exc:
        current_app.logger.error(f"Error opening attendance session: {exc}", exc_info=True)
        return jsonify({'success': False, 'message': 'Không thể mở phiên điểm danh'}), 500


@attendance_api_bp.route('/session/close', methods=['POST'])
@role_required('admin', 'teacher')
def api_close_attendance_session():
    data = get_request_data()
    session_id = data.get('session_id')
    if session_id:
        try:
            session_id = int(session_id)
        except (ValueError, TypeError):
            return jsonify({'success': False, 'message': 'Mã phiên không hợp lệ'}), 400
    else:
        current_session = get_active_attendance_session()
        if not current_session:
            return jsonify({'success': False, 'message': 'Không có phiên nào đang mở'}), 400
        session_id = current_session.get('id')

    try:
        closed = db.close_attendance_session(session_id)
        if not closed:
            return jsonify({'success': False, 'message': 'Không thể đóng phiên'}), 400
        set_active_session_cache(None)
        broadcast_session_snapshot(force_reload=True)
        payload = serialize_session_payload(db.get_session_by_id(session_id))
        return jsonify({'success': True, 'session': payload})
    except Exception as exc:
        current_app.logger.error(f"Error closing attendance session {session_id}: {exc}", exc_info=True)
        return jsonify({'success': False, 'message': 'Không thể đóng phiên điểm danh'}), 500


@attendance_api_bp.route('/session/<int:session_id>/mark', methods=['POST'])
@role_required('teacher', 'admin')
def api_mark_attendance_for_session(session_id):
    """Cho phép giảng viên (hoặc admin) điểm danh/checkout thủ công cho một phiên cụ thể."""
    data = get_request_data()
    student_code = data.get('student_id') or data.get('student_code')
    action = (data.get('action') or 'checkin').lower()

    if not student_code:
        return jsonify({'success': False, 'message': 'Missing student_id'}), 400

    session_row = db.get_session_by_id(session_id)
    if not session_row:
        return jsonify({'success': False, 'message': 'Session not found'}), 404

    credit_class = db.get_credit_class(session_row.get('credit_class_id'))
    if not credit_class:
        return jsonify({'success': False, 'message': 'Credit class not found'}), 404

    # Ủy quyền: giảng viên chỉ có thể điểm danh cho các lớp của họ
    if get_current_role() == 'teacher':
        teacher_ctx = resolve_teacher_context()
        if not teacher_ctx or int(credit_class.get('teacher_id') or 0) != int(teacher_ctx.get('id')):
            return jsonify({'success': False, 'message': 'Không có quyền trên lớp này'}), 403

    # Xác định sinh viên
    student_row = db.get_student(student_code)
    if not student_row:
        return jsonify({'success': False, 'message': 'Không tìm thấy sinh viên'}), 404
    student_name = student_row.get('full_name') or student_code

    try:
        if action in ('checkin', 'mark', 'present'):
            success = db.mark_attendance(
                student_id=student_code,
                student_name=student_name,
                status='present',
                confidence_score=None,
                notes='manual by teacher',
                credit_class_id=credit_class.get('id'),
                session_id=session_id,
            )
            if success:
                with app_globals.today_recorded_lock:
                    app_globals.today_checked_in.add(student_code)
                    app_globals.today_checked_out.discard(student_code)
                    app_globals.today_student_names[student_code] = {
                        'name': student_name,
                        'class_name': credit_class.get('subject_name') or credit_class.get('credit_code'),
                        'class_type': 'credit',
                        'credit_class_id': credit_class.get('id'),
                    }

                broadcast_sse_event({
                    'type': 'attendance_marked',
                    'data': {
                        'event': 'check_in',
                        'student_id': student_code,
                        'student_name': student_name,
                        'timestamp': datetime.now().isoformat(),
                        'session': serialize_session_payload(session_row),
                    },
                })
                return jsonify({'success': True})
            return jsonify({'success': False, 'message': 'Không thể điểm danh (có thể đã điểm danh trước đó)'}), 400

        elif action in ('checkout', 'check_out'):
            success = db.mark_checkout(student_code, session_id=session_id)
            if success:
                with app_globals.today_recorded_lock:
                    app_globals.today_checked_out.add(student_code)

                broadcast_sse_event({
                    'type': 'attendance_checkout',
                    'data': {
                        'event': 'check_out',
                        'student_id': student_code,
                        'student_name': student_name,
                        'timestamp': datetime.now().isoformat(),
                        'session': serialize_session_payload(session_row),
                    },
                })
                return jsonify({'success': True})
            return jsonify({'success': False, 'message': 'Không thể checkout hoặc chưa điểm danh'}), 400

        else:
            return jsonify({'success': False, 'message': 'Hành động không hợp lệ'}), 400

    except Exception as exc:
        current_app.logger.error(f"Error in manual mark endpoint: {exc}", exc_info=True)
        return jsonify({'success': False, 'message': 'Lỗi khi thực hiện điểm danh'}), 500


@attendance_api_bp.route('/today', methods=['GET'])
def api_attendance_today():
    """API lấy điểm danh hôm nay"""
    try:
        credit_class_id = request.args.get('credit_class_id', type=int)
        session_id = request.args.get('session_id', type=int)

        attendance_data = get_today_attendance(
            credit_class_id=credit_class_id,
            session_id=session_id,
        )

        session_row = None
        if session_id:
            session_row = db.get_session_by_id(session_id)
        elif credit_class_id:
            session_row = db.get_active_session_for_class(credit_class_id)
        else:
            session_row = get_active_attendance_session()

        checked_in = []
        checked_out = []
        for item in attendance_data:
            if item.get('checkout_time'):
                checked_out.append(item)
            else:
                checked_in.append(item)
        return jsonify({
            'success': True,
            'data': attendance_data,
            'checked_in': checked_in,
            'checked_out': checked_out,
            'session': serialize_session_payload(session_row)
        })
    except Exception as e:
        current_app.logger.error(f"Error getting today's attendance: {e}")
        return jsonify({'success': False, 'data': [], 'error': str(e)}), 500


@attendance_api_bp.route('/history/<student_id>', methods=['GET'])
def api_attendance_history(student_id):
    """API trả về lịch sử điểm danh gần đây của một sinh viên"""
    limit = request.args.get('limit', 10, type=int) or 10
    limit = max(1, min(limit, 50))
    
    try:
        history = db.get_student_attendance_history(student_id, limit=limit)
        return jsonify({'success': True, 'data': history or []})
    except Exception as e:
        current_app.logger.error(f"Error getting attendance history: {e}")
        return jsonify({'success': False, 'data': [], 'error': str(e)}), 500


@attendance_api_bp.route('/notifications', methods=['GET'])
def api_attendance_notifications():
    """API trả về các thông báo điểm danh / hệ thống để frontend hiển thị"""
    try:
        notifications = []

        # Lấy các bản ghi điểm danh gần đây (hôm nay) và chuyển thành thông báo
        try:
            session_row = get_active_attendance_session()
            session_id = session_row.get('id') if session_row else None
            credit_class_id = session_row.get('credit_class_id') if session_row else None
            recent_att = get_today_attendance(
                session_id=session_id,
                credit_class_id=credit_class_id,
            )
            # Chỉ lấy 5 bản ghi gần nhất
            for row in recent_att[:5]:
                notifications.append({
                    'type': 'attendance',
                    'student_id': row.get('student_id'),
                    'student_name': row.get('full_name'),
                    'timestamp': row.get('timestamp'),
                    'message': f"{row.get('full_name')} đã điểm danh"
                })
        except Exception:
            pass

        return jsonify({'success': True, 'notifications': notifications})
    except Exception as e:
        current_app.logger.error(f"Error getting notifications: {e}")
        return jsonify({'success': False, 'notifications': []}), 500
