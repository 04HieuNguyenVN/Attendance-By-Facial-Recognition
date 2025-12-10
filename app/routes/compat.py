"""
Compatibility layer
Giúp templates và JavaScript tìm đúng endpoint sau khi refactor
"""
from flask import Blueprint, redirect, url_for, render_template, request, current_app
import csv
import os
from pathlib import Path

compat_bp = Blueprint('compat', __name__)


# Redirect old endpoints to new blueprint endpoints
@compat_bp.route('/index')
def index():
    """Redirect /index to main.index"""
    return redirect(url_for('main.index'))


@compat_bp.route('/login_redirect')
def login_redirect():
    """Redirect /login_redirect to auth.login"""
    return redirect(url_for('auth.login'))


@compat_bp.route('/external-attendance', methods=['GET'])
def external_attendance():
    """Hiển thị chế độ xem chỉ đọc của các tệp CSV điểm danh từ dự án đính kèm.
    Điều này KHÔNG sửa đổi bất kỳ dữ liệu nào trong dự án chính; nó chỉ đọc các tệp CSV
    từ `external_projects/Cong-Nghe-Xu-Ly-Anh/attendance_logs` và hiển thị chúng
    bằng `templates/external_index.html`.
    """
    all_data = []
    headers = ["ID", "Name", "Time", "Status", "SourceFile"]
    external_dir = Path('Cong-Nghe-Xu-Ly-Anh') / 'attendance_logs'

    search_name = request.args.get('name', '').strip().lower()
    date_filter = request.args.get('date', '')

    if external_dir.exists():
        for filename in sorted(os.listdir(external_dir)):
            if filename.endswith('.csv'):
                file_path = external_dir / filename
                try:
                    with open(file_path, newline='', encoding='utf-8') as csvfile:
                        reader = csv.reader(csvfile)
                        rows = list(reader)
                        if len(rows) > 1 and rows[0][:4] == ["ID", "Name", "Time", "Status"]:
                            for row in rows[1:]:
                                row.append(filename)
                                all_data.append(row)
                except Exception as e:
                    current_app.logger.warning(f"Không thể đọc file external {filename}: {e}")

    # Lọc cục bộ
    if search_name:
        all_data = [row for row in all_data if search_name in str(row[1]).lower()]

    if date_filter:
        filtered = []
        for row in all_data:
            try:
                row_date_str = row[2].split()[0] if row[2] else ''
                if date_filter in row_date_str:
                    filtered.append(row)
            except Exception:
                continue
        all_data = filtered

    return render_template('external_index.html', all_data=all_data, headers=headers)


# Helper function to inject into Jinja2 context
def get_url_for_compat():
    """
    Provide backward-compatible url_for helper
    Maps old endpoint names to new blueprint.endpoint names
    """
    endpoint_map = {
        'index': 'main.index',
        'login': 'auth.login',
        'logout': 'auth.logout',
        'students': 'main.students',
        'classes': 'main.classes',
        'reports': 'main.reports',
        'teacher_credit_classes': 'main.teacher_credit_classes',
        'student_portal': 'main.student_portal',
        'student_portal_page': 'main.student_portal',
        'status': 'main.status',
        'video_feed': 'camera_api.video_feed',
    }
    return endpoint_map


def register_compat_helpers(app):
    """Register compatibility helpers with Flask app"""
    
    @app.context_processor
    def inject_endpoint_map():
        """Make endpoint map available in templates"""
        return {
            'endpoint_map': get_url_for_compat()
        }
