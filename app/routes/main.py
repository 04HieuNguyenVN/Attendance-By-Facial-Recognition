"""
Main routes
Trang chủ và các trang cơ bản
"""
from flask import Blueprint, render_template, Response, current_app, request, flash
from app.middleware.auth import role_required
from app.utils.session_utils import resolve_student_context, get_current_role
from database import db
from datetime import datetime

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
@role_required('admin', 'teacher', 'student')
def index():
    """Trang chủ - điểm danh."""
    try:
        # Lấy dữ liệu điểm danh hôm nay từ DB
        attendance_records = db.get_today_attendance()
        
        # Chuyển đổi sang dict và phân loại
        attendance_data = []
        checked_in = []
        checked_out = []
        
        for row in attendance_records:
            item = {
                'student_id': row['student_id'],
                'full_name': row.get('student_name'),
                'class_name': row.get('class_name'),
                'timestamp': row.get('check_in_time'),
                'checkout_time': row.get('check_out_time'),
                'date': row.get('attendance_date'),
            }
            attendance_data.append(item)
            
            if row.get('check_out_time'):
                checked_out.append(item)
            else:
                checked_in.append(item)
        
        return render_template('index.html', 
                             attendance=attendance_data,
                             checked_in=checked_in, 
                             checked_out=checked_out)
    except Exception as e:
        current_app.logger.error(f"Error loading index page: {e}")
        return render_template('index.html', 
                             attendance=[], 
                             checked_in=[], 
                             checked_out=[])


@main_bp.route('/students')
@role_required('admin', 'teacher')
def students():
    """Trang quản lý sinh viên."""
    return render_template('students.html')


@main_bp.route('/classes')
@role_required('admin', 'teacher')
def classes():
    """Trang quản lý lớp học."""
    try:
        admin_classes = db.get_all_classes()
        credit_classes = db.list_credit_classes_overview()
        teacher_options = db.get_all_teachers()
        classes_list = admin_classes + credit_classes

        total_admin = len(admin_classes)
        total_credit = len(credit_classes)
        total_classes = total_admin + total_credit

        # Sử dụng số lượng sinh viên của lớp hành chính; các lớp tín chỉ bao gồm số lượng riêng
        total_students = sum(cls.get('student_count', 0) for cls in admin_classes)
        active_classes = sum(1 for cls in admin_classes if cls.get('is_active', 1)) + sum(1 for cc in credit_classes if cc.get('is_active', 1))

        attendance_rates = []
        for cls in classes_list:
            try:
                stats = db.get_class_attendance_stats(cls['id'])
                attendance_rates.append(stats.get('attendance_rate', 0))
            except Exception as stats_error:
                current_app.logger.debug(
                    "Could not calculate attendance rate for class %s: %s",
                    cls.get('id'),
                    stats_error,
                )

        avg_attendance = round(sum(attendance_rates) / len(attendance_rates), 2) if attendance_rates else 0

        return render_template(
            'classes.html',
            admin_classes=admin_classes,
            credit_classes=credit_classes,
            teacher_options=teacher_options,
            total_classes=total_classes,
            total_admin=total_admin,
            total_credit=total_credit,
            total_students=total_students,
            active_classes=active_classes or total_classes,
            avg_attendance=avg_attendance,
        )
    except Exception as error:
        current_app.logger.error(f"Error loading classes page: {error}")
        return render_template(
            'classes.html',
            admin_classes=[],
            credit_classes=[],
            teacher_options=[],
            total_classes=0,
            total_admin=0,
            total_credit=0,
            total_students=0,
            active_classes=0,
            avg_attendance=0,
        )


@main_bp.route('/reports')
@role_required('admin', 'teacher')
def reports():
    """Trang báo cáo."""
    return render_template('reports.html')


@main_bp.route('/teacher/credit-classes')
@role_required('teacher')
def teacher_credit_classes():
    """Trang quản lý lớp tín chỉ của giáo viên."""
    return render_template('teacher_credit_classes.html')


@main_bp.route('/student/portal')
@role_required('student', 'admin')
def student_portal():
    """Trang tổng quan cho sinh viên xem lịch học và lịch sử điểm danh."""
    student_id = request.args.get('student_id') if get_current_role() == 'admin' else None
    student = resolve_student_context(student_id)
    if (student_id or get_current_role() == 'student') and not student:
        flash('Không tìm thấy thông tin sinh viên cho tài khoản hiện tại. Vui lòng liên hệ quản trị viên để được cấp quyền.', 'warning')
    return render_template(
        'student_portal.html',
        student=student,
        student_param=student_id,
        active_page='student-portal',
    )


@main_bp.route('/status')
@role_required('admin', 'teacher', 'student')
def status():
    """Trang trạng thái hệ thống."""
    return render_template('status.html')
