"""
API routes for credit classes
Các API endpoint cho quản lý lớp tín chỉ
"""
from flask import Blueprint, jsonify, request, current_app
from database import db
from app.middleware.auth import role_required
from app.utils import get_request_data, parse_bool
from app.utils.session_utils import (
    serialize_credit_class_record,
    serialize_session_payload,
    resolve_teacher_context,
    resolve_student_context,
    get_current_role,
    get_active_attendance_session
)

credit_classes_api_bp = Blueprint('credit_classes_api', __name__, url_prefix='/api')


@credit_classes_api_bp.route('/credit-classes', methods=['GET'])
def api_get_credit_classes():
    """API lấy danh sách lớp tín chỉ đang hoạt động."""
    teacher_only = parse_bool(request.args.get('mine'))
    teacher_id = None
    if teacher_only:
        from flask import g
        if getattr(g, 'user', None):
            teacher_row = db.get_teacher_by_user(g.user['id']) if get_current_role() == 'teacher' else None
            if teacher_row:
                teacher_id = teacher_row.get('id')
    try:
        credit_classes = db.list_credit_classes_overview(teacher_id=teacher_id)
        for cls in credit_classes:
            cls['class_type'] = 'credit'
            cls['display_name'] = ' · '.join(
                part for part in [cls.get('subject_name'), cls.get('credit_code')] if part
            ) or cls.get('subject_name') or cls.get('credit_code')
            cls['student_count'] = cls.get('student_count', 0) or 0
        return jsonify({'success': True, 'data': credit_classes})
    except Exception as err:
        current_app.logger.error(f"Error getting credit classes: {err}", exc_info=True)
        return jsonify({'success': False, 'data': [], 'message': 'Không thể tải lớp tín chỉ'}), 500


@credit_classes_api_bp.route('/credit-classes', methods=['POST'])
@role_required('admin')
def api_create_credit_class():
    data = get_request_data()
    credit_code = (data.get('credit_code') or '').strip()
    subject_name = (data.get('subject_name') or '').strip()
    teacher_id = data.get('teacher_id')

    if not credit_code or not subject_name:
        return jsonify({'success': False, 'message': 'Mã lớp tín chỉ và tên môn học là bắt buộc'}), 400

    try:
        teacher_id = int(teacher_id)
    except (TypeError, ValueError):
        return jsonify({'success': False, 'message': 'Vui lòng chọn giảng viên phụ trách'}), 400

    teacher = db.get_teacher(teacher_id)
    if not teacher:
        return jsonify({'success': False, 'message': 'Giảng viên không tồn tại hoặc đã bị vô hiệu hóa'}), 400

    enrollment_limit = data.get('enrollment_limit')
    if enrollment_limit in (None, '', []):
        enrollment_limit = None
    else:
        try:
            enrollment_limit = int(enrollment_limit)
        except (TypeError, ValueError):
            return jsonify({'success': False, 'message': 'Sĩ số tối đa không hợp lệ'}), 400

    try:
        credit_class_id = db.create_credit_class(
            credit_code=credit_code,
            subject_name=subject_name,
            teacher_id=teacher_id,
            semester=data.get('semester'),
            academic_year=data.get('academic_year'),
            room=data.get('room'),
            schedule_info=data.get('schedule_info'),
            enrollment_limit=enrollment_limit,
            notes=data.get('notes'),
            status=data.get('status') or 'draft'
        )
        record = serialize_credit_class_record(db.get_credit_class(credit_class_id))
        return jsonify({'success': True, 'data': record}), 201
    except ValueError as exc:
        return jsonify({'success': False, 'message': str(exc)}), 400
    except Exception as exc:
        current_app.logger.error(f"Error creating credit class: {exc}", exc_info=True)
        return jsonify({'success': False, 'message': 'Không thể tạo lớp tín chỉ'}), 500


@credit_classes_api_bp.route('/credit-classes/<int:credit_class_id>', methods=['GET', 'PUT', 'DELETE'])
@role_required('admin')
def api_credit_class_detail(credit_class_id):
    record = db.get_credit_class(credit_class_id)
    if not record:
        return jsonify({'success': False, 'message': 'Không tìm thấy lớp tín chỉ'}), 404

    if request.method == 'GET':
        return jsonify({'success': True, 'data': serialize_credit_class_record(record)})

    if request.method == 'PUT':
        data = get_request_data()
        updates = {}

        for field in ('credit_code', 'subject_name', 'semester', 'academic_year', 'room', 'schedule_info', 'status', 'notes'):
            if field in data:
                value = data.get(field)
                updates[field] = value.strip() if isinstance(value, str) else value

        if 'teacher_id' in data:
            teacher_id = data.get('teacher_id')
            try:
                teacher_id = int(teacher_id)
            except (TypeError, ValueError):
                return jsonify({'success': False, 'message': 'Giảng viên không hợp lệ'}), 400
            teacher = db.get_teacher(teacher_id)
            if not teacher:
                return jsonify({'success': False, 'message': 'Không tìm thấy giảng viên'}), 400
            updates['teacher_id'] = teacher_id

        if 'enrollment_limit' in data:
            limit_value = data.get('enrollment_limit')
            if limit_value in (None, '', []):
                updates['enrollment_limit'] = None
            else:
                try:
                    updates['enrollment_limit'] = int(limit_value)
                except (TypeError, ValueError):
                    return jsonify({'success': False, 'message': 'Sĩ số tối đa không hợp lệ'}), 400

        if not updates:
            return jsonify({'success': False, 'message': 'Không có dữ liệu để cập nhật'}), 400

        try:
            updated = db.update_credit_class(credit_class_id, **updates)
            if not updated:
                return jsonify({'success': False, 'message': 'Không thể cập nhật lớp tín chỉ'}), 400
            refreshed = serialize_credit_class_record(db.get_credit_class(credit_class_id))
            return jsonify({'success': True, 'data': refreshed})
        except Exception as exc:
            current_app.logger.error(f"Error updating credit class {credit_class_id}: {exc}", exc_info=True)
            return jsonify({'success': False, 'message': 'Lỗi khi cập nhật lớp tín chỉ'}), 500

    # DELETE
    try:
        deleted = db.delete_credit_class(credit_class_id)
        if not deleted:
            return jsonify({'success': False, 'message': 'Không thể xóa lớp tín chỉ'}), 400
        return jsonify({'success': True})
    except Exception as exc:
        current_app.logger.error(f"Error deleting credit class {credit_class_id}: {exc}", exc_info=True)
        return jsonify({'success': False, 'message': 'Lỗi khi xóa lớp tín chỉ'}), 500


@credit_classes_api_bp.route('/teacher/credit-classes', methods=['GET'])
@role_required('teacher', 'admin')
def api_teacher_credit_classes():
    teacher_param = request.args.get('teacher_id', type=int) if get_current_role() == 'admin' else None
    teacher = resolve_teacher_context(teacher_param)
    if get_current_role() == 'teacher' and not teacher:
        return jsonify({'success': False, 'message': 'Không tìm thấy thông tin giảng viên'}), 404

    teacher_filter_id = teacher.get('id') if teacher else None
    try:
        credit_classes = db.list_credit_classes_overview(teacher_id=teacher_filter_id)
        results = []
        for cls in credit_classes:
            session_row = db.get_active_session_for_class(cls['id'])
            payload = dict(cls)
            payload['class_type'] = 'credit'
            payload['display_name'] = ' · '.join(
                part for part in [cls.get('subject_name'), cls.get('credit_code')] if part
            ) or cls.get('subject_name') or cls.get('credit_code')
            payload['student_count'] = cls.get('student_count', 0) or 0
            payload['active_session'] = serialize_session_payload(session_row)
            results.append(payload)

        return jsonify({
            'success': True,
            'data': results,
            'teacher': teacher,
        })
    except Exception as exc:
        current_app.logger.error("Error loading teacher credit classes: %s", exc, exc_info=True)
        return jsonify({'success': False, 'message': 'Không thể tải danh sách lớp'}), 500


@credit_classes_api_bp.route('/teacher/credit-classes/<int:credit_class_id>/students', methods=['GET'])
@role_required('teacher', 'admin')
def api_teacher_credit_class_students(credit_class_id):
    teacher_param = request.args.get('teacher_id', type=int) if get_current_role() == 'admin' else None
    teacher = resolve_teacher_context(teacher_param)
    credit_class = db.get_credit_class(credit_class_id)
    if not credit_class:
        return jsonify({'success': False, 'message': 'Không tìm thấy lớp tín chỉ'}), 404
    if teacher and credit_class.get('teacher_id') and credit_class.get('teacher_id') != teacher.get('id'):
        return jsonify({'success': False, 'message': 'Không có quyền truy cập lớp này'}), 403

    try:
        roster = db.get_credit_class_students(credit_class_id) or []
        today_attendance = db.get_today_attendance(credit_class_id=credit_class_id) or []

        present_map = {}
        for att in today_attendance:
            sid = (att or {}).get('student_id')
            if not sid:
                continue
            present_map[sid] = {
                'attendance_id': att.get('attendance_id'),
                'check_in_time': att.get('check_in_time'),
                'checkout_time': att.get('checkout_time'),
                'checked_out': bool(att.get('checkout_time')),
            }

        serialized = []
        for student in roster:
            srec = dict(student)
            sid = srec.get('student_id')
            attendance_info = present_map.get(sid)
            if attendance_info:
                srec['is_present_today'] = True
                srec['checked_out'] = attendance_info.get('checked_out', False)
                srec['attendance_id'] = attendance_info.get('attendance_id')
                srec['check_in_time'] = attendance_info.get('check_in_time')
            else:
                srec['is_present_today'] = False
                srec['checked_out'] = False
                srec['attendance_id'] = None
                srec['check_in_time'] = None
            serialized.append(srec)

        return jsonify({
            'success': True,
            'credit_class': dict(credit_class),
            'students': serialized,
            'count': len(serialized),
        })
    except Exception as exc:
        current_app.logger.error("Error loading roster for credit class %s: %s", credit_class_id, exc, exc_info=True)
        return jsonify({'success': False, 'message': 'Không thể tải danh sách sinh viên'}), 500


@credit_classes_api_bp.route('/teacher/credit-classes/<int:credit_class_id>/sessions', methods=['GET'])
@role_required('teacher', 'admin')
def api_teacher_credit_class_sessions(credit_class_id):
    teacher_param = request.args.get('teacher_id', type=int) if get_current_role() == 'admin' else None
    teacher = resolve_teacher_context(teacher_param)
    credit_class = db.get_credit_class(credit_class_id)
    if not credit_class:
        return jsonify({'success': False, 'message': 'Không tìm thấy lớp tín chỉ'}), 404
    if teacher and credit_class.get('teacher_id') and credit_class.get('teacher_id') != teacher.get('id'):
        return jsonify({'success': False, 'message': 'Không có quyền truy cập lớp này'}), 403

    limit = request.args.get('limit', 20, type=int) or 20
    limit = max(5, min(limit, 100))
    try:
        sessions = db.list_sessions_for_class(credit_class_id, limit=limit)
        return jsonify({
            'success': True,
            'credit_class': dict(credit_class),
            'sessions': [dict(s) for s in sessions] if sessions else [],
            'count': len(sessions) if sessions else 0,
        })
    except Exception as exc:
        current_app.logger.error("Error loading sessions for credit class %s: %s", credit_class_id, exc, exc_info=True)
        return jsonify({'success': False, 'message': 'Không thể tải danh sách phiên'}), 500


@credit_classes_api_bp.route('/student/credit-classes', methods=['GET'])
@role_required('student', 'admin')
def api_student_credit_classes():
    """API lấy danh sách lớp tín chỉ cho sinh viên."""
    student_param = request.args.get('student_id') if get_current_role() == 'admin' else None
    student = resolve_student_context(student_param)
    if not student:
        return jsonify({'success': False, 'message': 'Không tìm thấy sinh viên'}), 404

    try:
        student_identifier = student.get('id') or student.get('student_id')
        classes = db.get_credit_classes_for_student(student_identifier)
        formatted = []
        active_sessions = 0
        known_class_ids = set()
        for cls in classes:
            session_row = db.get_active_session_for_class(cls['id'])
            if session_row:
                active_sessions += 1
            entry = dict(cls)
            entry['display_name'] = ' · '.join(
                part for part in [cls.get('subject_name'), cls.get('credit_code')] if part
            ) or cls.get('subject_name') or cls.get('credit_code')
            entry['active_session'] = serialize_session_payload(session_row)
            known_class_ids.add(cls.get('id'))
            formatted.append(entry)

        # Nếu sinh viên chưa đăng ký nhưng đang có phiên mở, hiển thị ở dạng session-only
        fallback_session = get_active_attendance_session()
        if fallback_session:
            fallback_class_id = fallback_session.get('credit_class_id')
            if fallback_class_id and fallback_class_id not in known_class_ids:
                credit_cls = db.get_credit_class(fallback_class_id)
                if credit_cls:
                    entry = dict(credit_cls)
                    entry['display_name'] = ' · '.join(
                        part for part in [credit_cls.get('subject_name'), credit_cls.get('credit_code')] if part
                    ) or credit_cls.get('subject_name') or credit_cls.get('credit_code')
                    entry['active_session'] = serialize_session_payload(fallback_session)
                    entry['is_session_only'] = True
                    if 'student_count' not in entry or entry.get('student_count') is None:
                        try:
                            roster = db.get_credit_class_students(fallback_class_id) or []
                            entry['student_count'] = len(roster)
                        except Exception:
                            entry['student_count'] = 0
                    formatted.insert(0, entry)
                    active_sessions += 1 if (fallback_session.get('status') == 'open') else 0

        summary = {
            'total_classes': len(formatted),
            'active_sessions': active_sessions,
        }

        return jsonify({
            'success': True,
            'student': {
                'student_id': student.get('student_id'),
                'full_name': student.get('full_name'),
                'email': student.get('email'),
                'class_id': student.get('class_id'),
            },
            'classes': formatted,
            'summary': summary,
        })
    except Exception as exc:
        current_app.logger.error("Error loading credit classes for student %s: %s", student.get('student_id'), exc, exc_info=True)
        return jsonify({'success': False, 'message': 'Không thể tải danh sách lớp tín chỉ'}), 500
