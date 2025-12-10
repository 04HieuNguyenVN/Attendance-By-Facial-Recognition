"""
API routes for system status
Các API cho trạng thái hệ thống
"""
from flask import Blueprint, jsonify
from app import globals as app_globals
import os

system_api_bp = Blueprint('system_api', __name__)

# Configuration
DEMO_MODE = os.getenv('DEMO_MODE', 'false').lower() == 'true'
FACE_RECOGNITION_AVAILABLE = True  # Will be updated based on actual availability


@system_api_bp.route('/status')
def api_system_status():
    """API trạng thái hệ thống"""
    return jsonify({
        'demo_mode': DEMO_MODE,
        'face_recognition_available': FACE_RECOGNITION_AVAILABLE,
        'camera_enabled': getattr(app_globals, 'camera_enabled', False),
        'known_faces_count': len(app_globals.known_face_names) if hasattr(app_globals, 'known_face_names') else 0
    })
