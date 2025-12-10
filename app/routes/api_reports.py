"""
API routes for reports
Các API endpoint cho báo cáo
"""
from flask import Blueprint, jsonify, request, current_app
from database import db
from app.utils.session_utils import (
    serialize_session_payload,
    serialize_credit_class_record
)

reports_api_bp = Blueprint('reports_api', __name__, url_prefix='/api/reports')


@reports_api_bp.route('/credit-classes/<int:credit_class_id>/sessions', methods=['GET'])
def api_reports_credit_class_sessions(credit_class_id):
    """API công khai phục vụ bộ lọc báo cáo lấy danh sách phiên của lớp tín chỉ."""
    limit = request.args.get('limit', 25, type=int) or 25
    limit = max(5, min(limit, 100))

    try:
        credit_class = db.get_credit_class(credit_class_id)
        if not credit_class:
            return jsonify({'success': False, 'message': 'Không tìm thấy lớp tín chỉ'}), 404

        sessions = db.list_sessions_for_class(credit_class_id, limit=limit) or []
        payload = [serialize_session_payload(session) for session in sessions]
        return jsonify({
            'success': True,
            'credit_class': serialize_credit_class_record(credit_class),
            'sessions': payload,
        })
    except Exception as exc:
        current_app.logger.error(
            "Error loading report sessions for credit class %s: %s",
            credit_class_id,
            exc,
            exc_info=True,
        )
        return jsonify({'success': False, 'message': 'Không thể tải danh sách phiên'}), 500
