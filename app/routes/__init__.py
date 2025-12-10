"""
Routes package
Đăng ký tất cả các blueprints
"""
from .auth import auth_bp
from .main import main_bp
from .api_students import student_api_bp
from .api_classes import class_api_bp
from .api_camera import camera_api_bp
from .api_attendance import attendance_api_bp
from .api_credit_classes import credit_classes_api_bp
from .api_stats import stats_api_bp
from .api_events import events_api_bp
from .api_register import quick_register_api_bp
from .api_reports import reports_api_bp
from .api_system import system_api_bp
from .api_training import training_api_bp
from .compat import compat_bp, register_compat_helpers


def register_blueprints(app):
    """Đăng ký tất cả các blueprints với Flask app."""
    # Authentication routes
    app.register_blueprint(auth_bp)
    
    # Main routes
    app.register_blueprint(main_bp)
    
    # API routes
    app.register_blueprint(student_api_bp)
    app.register_blueprint(class_api_bp)
    # NOTE: camera_api_bp tạm thời bị vô hiệu hóa vì chưa migrate đầy đủ
    # app.register_blueprint(camera_api_bp)  # Commented out - dùng routes trong app.py gốc
    app.register_blueprint(attendance_api_bp)
    app.register_blueprint(credit_classes_api_bp)
    app.register_blueprint(stats_api_bp)
    app.register_blueprint(events_api_bp)
    app.register_blueprint(quick_register_api_bp)
    app.register_blueprint(reports_api_bp)
    app.register_blueprint(system_api_bp)
    app.register_blueprint(training_api_bp)
    
    # Compatibility routes (for old endpoints)
    app.register_blueprint(compat_bp)
    register_compat_helpers(app)
    
    app.logger.info("✅ Đã đăng ký tất cả blueprints")
