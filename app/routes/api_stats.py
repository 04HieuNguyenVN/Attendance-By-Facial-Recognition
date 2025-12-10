"""
API routes for statistics and presence tracking
Các API endpoint cho thống kê và theo dõi có mặt
"""
from flask import Blueprint, jsonify, request, current_app
from datetime import datetime
from database import db
from app.utils.attendance_utils import get_today_attendance
from app import globals as app_globals

stats_api_bp = Blueprint('stats_api', __name__, url_prefix='/api')


@stats_api_bp.route('/statistics')
def api_statistics():
    """API thống kê"""
    try:
        credit_class_id = request.args.get('credit_class_id', type=int)
        session_id = request.args.get('session_id', type=int)

        attendance_data = get_today_attendance(
            credit_class_id=credit_class_id,
            session_id=session_id,
        )

        total_students = len(app_globals.known_face_names) if app_globals.known_face_names else 0
        if credit_class_id:
            try:
                roster = db.get_credit_class_students(credit_class_id) or []
                total_students = len(roster)
            except Exception as roster_error:
                current_app.logger.warning(
                    "Không thể lấy sĩ số lớp tín chỉ %s: %s",
                    credit_class_id,
                    roster_error,
                )
        elif session_id:
            session_row = db.get_session_by_id(session_id)
            if session_row:
                resolved_class_id = session_row.get('credit_class_id')
                if resolved_class_id:
                    try:
                        roster = db.get_credit_class_students(resolved_class_id) or []
                        total_students = len(roster)
                    except Exception as roster_error:
                        current_app.logger.warning(
                            "Không thể lấy sĩ số cho phiên %s: %s",
                            session_id,
                            roster_error,
                        )

        attended_students = len(attendance_data)
        attendance_rate = (attended_students / total_students * 100) if total_students > 0 else 0
        
        # Tính tổng thời gian có mặt
        total_minutes = sum(item['duration_minutes'] for item in attendance_data if item.get('duration_minutes'))
        avg_duration = int(total_minutes / attended_students) if attended_students > 0 else 0

        return jsonify({
            'total_students': total_students,
            'attended_students': attended_students,
            'attendance_rate': round(attendance_rate, 2),
            'avg_duration_minutes': avg_duration,
            'total_duration_minutes': total_minutes
        })
    except Exception as e:
        current_app.logger.error(f"Error getting statistics: {e}")
        return jsonify({
            'total_students': 0, 
            'attended_students': 0, 
            'attendance_rate': 0,
            'avg_duration_minutes': 0,
            'total_duration_minutes': 0
        }), 500


@stats_api_bp.route('/presence/active', methods=['GET'])
def api_active_presence():
    """API lấy danh sách sinh viên đang có mặt (đang được tracking)"""
    try:
        with app_globals.presence_tracking_lock:
            active_students = []
            now = datetime.now()
            
            for student_id, data in app_globals.presence_tracking.items():
                check_in_time = data.get('check_in_time')
                last_seen = data.get('last_seen')
                
                if not check_in_time or not last_seen:
                    continue
                    
                duration_seconds = (now - check_in_time).total_seconds()
                time_since_seen = (now - last_seen).total_seconds()
                
                active_students.append({
                    'student_id': student_id,
                    'name': data.get('name'),
                    'check_in_time': check_in_time.isoformat(),
                    'last_seen': last_seen.isoformat(),
                    'duration_minutes': int(duration_seconds / 60),
                    'seconds_since_seen': int(time_since_seen),
                    'is_active': time_since_seen < 30  # Còn active nếu thấy trong 30s
                })
            
            return jsonify({
                'success': True,
                'count': len(active_students),
                'data': active_students
            })
    except Exception as e:
        current_app.logger.error(f"Error getting active presence: {e}")
        return jsonify({'success': False, 'data': [], 'error': str(e)}), 500
