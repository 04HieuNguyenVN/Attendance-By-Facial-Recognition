"""
App package initialization
Khởi tạo Flask application và cấu hình
"""
from flask import Flask
import os
from pathlib import Path
from logging_config import setup_logging
from database import db
from app import globals as app_globals
from app import config
from app.models import (
    state_manager,
    CameraService,
    AttendanceTracker,
    FaceRecognitionManager,
    get_event_broadcaster,
)
from i18n import init_i18n  # Quốc tế hóa (i18n)


def _init_face_recognition_services(app):
    """Khởi tạo các dịch vụ nhận diện khuôn mặt"""
    face_service = None
    antispoof_service = None
    deepface_module = None
    deepface_available = False
    yolo_model = None
    yolo_available = False
    
    # Try to load FaceNet services
    if config.USE_FACENET and not config.DEMO_MODE:
        try:
            from app.services.face_service import FaceRecognitionService
            from app.services.antispoof_service import AntiSpoofService
            
            face_service = FaceRecognitionService(
                confidence_threshold=float(os.getenv('FACENET_THRESHOLD', '0.85'))
            )
            antispoof_service = AntiSpoofService(
                device=os.getenv('ANTISPOOF_DEVICE', 'cpu'),
                spoof_threshold=float(os.getenv('ANTISPOOF_THRESHOLD', '0.5'))
            )
            app.logger.info("[STARTUP] ✅ FaceNet services initialized")
        except Exception as e:
            app.logger.warning(f"[STARTUP] ⚠️ Could not initialize FaceNet: {e}")
    
    # Try to load DeepFace
    try:
        from deepface import DeepFace
        deepface_module = DeepFace
        deepface_available = True
        app.logger.info("[STARTUP] ✅ DeepFace available")
    except Exception as e:
        app.logger.warning(f"[STARTUP] ⚠️ DeepFace not available: {e}")
        deepface_available = False
    
    # Try to load YOLO
    try:
        from ultralytics import YOLO
        possible_paths = [
            'yolov8m-face.pt',
            'models/yolov8m-face.pt',
        ]
        for path in possible_paths:
            if os.path.exists(path):
                yolo_model = YOLO(path)
                yolo_available = True
                app.logger.info(f"[STARTUP] ✅ YOLOv8 loaded from {path}")
                break
        if not yolo_available:
            app.logger.warning("[STARTUP] ⚠️ YOLOv8 model not found")
    except Exception as e:
        app.logger.warning(f"[STARTUP] ⚠️ Could not load YOLO: {e}")
    
    return {
        'face_service': face_service,
        'antispoof_service': antispoof_service,
        'deepface_module': deepface_module,
        'deepface_available': deepface_available,
        'yolo_model': yolo_model,
        'yolo_available': yolo_available,
    }


def _init_inference_engine(app, services):
    """Khởi tạo inference engine"""
    from core.inference.engine import InferenceEngine, DeepFaceStrategy, FaceNetStrategy
    
    try:
        inference_engine = InferenceEngine(logger=app.logger, demo_mode=config.DEMO_MODE)
        
        # Add DeepFace strategy
        if services['deepface_available']:
            try:
                from app.services.deepface_db import build_db_from_data_dir, recognize_face as deepface_recognize
                deepface_strategy = DeepFaceStrategy(
                    data_dir=config.DATA_DIR,
                    deepface_module=services['deepface_module'],
                    build_db_fn=build_db_from_data_dir,
                    recognize_fn=deepface_recognize,
                    similarity_threshold=config.DEEPFACE_SIMILARITY_THRESHOLD,
                    enforce_detection=False,
                    logger=app.logger,
                )
                inference_engine.add_strategy(deepface_strategy)
                app.logger.info("[STARTUP] ✅ DeepFace strategy added to inference engine")
            except Exception as e:
                app.logger.warning(f"[STARTUP] ⚠️ Could not add DeepFace strategy: {e}")
        
        # Add FaceNet strategy
        if config.USE_FACENET and services['face_service']:
            try:
                def lookup_student_name(student_id):
                    try:
                        student = db.get_student(student_id)
                        if student:
                            return student.get('full_name') or student.get('student_name') or student_id
                    except Exception:
                        pass
                    return None
                
                facenet_strategy = FaceNetStrategy(
                    service=services['face_service'],
                    label_lookup=lookup_student_name,
                    logger=app.logger,
                )
                inference_engine.add_strategy(facenet_strategy)
                app.logger.info("[STARTUP] ✅ FaceNet strategy added to inference engine")
            except Exception as e:
                app.logger.warning(f"[STARTUP] ⚠️ Could not add FaceNet strategy: {e}")
        
        state_manager.inference_engine = inference_engine
        app.logger.info("[STARTUP] ✅ Inference engine initialized")
        return inference_engine
    
    except Exception as e:
        app.logger.error(f"[STARTUP] ❌ Could not initialize inference engine: {e}")
        return None


def create_app():
    """Factory function để tạo Flask application"""
    app = Flask(__name__, 
                template_folder='../templates',
                static_folder='../static')
    
    # Cấu hình cơ bản
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
    
    # Thiết lập logging
    setup_logging(app)
    
    # Khởi tạo i18n (đa ngôn ngữ)
    init_i18n(app)
    
    app.logger.info(f"[STARTUP] Working directory: {os.getcwd()}")
    app.logger.info(f"[STARTUP] Database path: {os.path.abspath('attendance_system.db')}")
    
    # =============================================================================
    # INITIALIZE SERVICES - NEW ARCHITECTURE
    # =============================================================================
    
    # 1. Initialize face recognition services
    fr_services = _init_face_recognition_services(app)
    
    # 2. Initialize inference engine
    inference_engine = _init_inference_engine(app, fr_services)
    
    # 3. Initialize CameraService
    app_globals.camera_service = CameraService(
        camera_index=config.CAMERA_INDEX,
        width=config.CAMERA_WIDTH,
        height=config.CAMERA_HEIGHT,
        warmup_frames=config.CAMERA_WARMUP_FRAMES,
        buffer_size=config.CAMERA_BUFFER_SIZE,
        logger=app.logger
    )
    app.logger.info("[STARTUP] ✅ CameraService initialized")
    
    # 4. Initialize AttendanceTracker
    app_globals.attendance_tracker = AttendanceTracker(
        database=db,
        state_manager=state_manager,
        logger=app.logger
    )
    app.logger.info("[STARTUP] ✅ AttendanceTracker initialized")
    
    # 5. Initialize FaceRecognitionManager
    app_globals.face_recognition_manager = FaceRecognitionManager(
        data_dir=config.DATA_DIR,
        inference_engine=inference_engine,
        deepface_module=fr_services['deepface_module'],
        deepface_available=fr_services['deepface_available'],
        yolo_model=fr_services['yolo_model'],
        yolo_available=fr_services['yolo_available'],
        similarity_threshold=config.DEEPFACE_SIMILARITY_THRESHOLD,
        logger=app.logger
    )
    app.logger.info("[STARTUP] ✅ FaceRecognitionManager initialized")
    
    # 6. Initialize EventBroadcaster
    app_globals.event_broadcaster = get_event_broadcaster(logger=app.logger)
    app.logger.info("[STARTUP] ✅ EventBroadcaster initialized")
    
    # 7. Update global references for backward compatibility
    app_globals.init_service_references()
    app.logger.info("[STARTUP] ✅ All services initialized successfully")
    
    # =============================================================================
    # LOAD DATA - Use new AttendanceTracker
    # =============================================================================
    
    try:
        app_globals.attendance_tracker.load_today_attendance()
        stats = app_globals.attendance_tracker.get_attendance_stats()
        app.logger.info(f"[STARTUP] Loaded attendance: {stats['checked_in']} checked in, {stats['checked_out']} checked out")
    except Exception as e:
        app.logger.warning(f"[STARTUP] Failed to load attendance cache: {e}")
    
    # Dọn dẹp các bản ghi lớp học cũ bị lỗi nếu có
    LEGACY_CLASS_NAMES = [
        'Công nghệ thông tin 01',
        'Công nghệ thông tin 01',
    ]
    for legacy_name in LEGACY_CLASS_NAMES:
        removed = db.delete_class_by_name(legacy_name)
        if removed:
            app.logger.info("Removed legacy class entry: %s (records: %s)", legacy_name, removed)
    
    # Đăng ký middleware
    from app.middleware.auth import register_auth_middleware
    register_auth_middleware(app)
    
    # Đăng ký blueprints
    from app.routes import register_blueprints
    register_blueprints(app)
    
    # TEMPORARY: Đăng ký camera routes từ app.py gốc (file ở root)
    # Các routes này chưa được migrate đầy đủ do phụ thuộc nhiều global state
    # Sẽ được refactor sau khi tách vision service hoàn chỉnh
    try:
        import sys
        from pathlib import Path
        # Thêm thư mục root vào path để import app.py gốc
        root_dir = Path(__file__).parent.parent
        if str(root_dir) not in sys.path:
            sys.path.insert(0, str(root_dir))
        
        # Import app.py gốc (không phải package app)
        import importlib.util
        spec = importlib.util.spec_from_file_location("legacy_app", root_dir / "app.py")
        if spec and spec.loader:
            legacy_app = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(legacy_app)
            
            # Đăng ký các route camera
            app.add_url_rule('/video_feed', 'legacy_video_feed', 
                            legacy_app.video_feed, methods=['GET'])
            app.add_url_rule('/api/camera/toggle', 'legacy_camera_toggle',
                            legacy_app.toggle_camera, methods=['POST'])
            app.add_url_rule('/api/camera/status', 'legacy_camera_status',
                            legacy_app.camera_status, methods=['GET'])
            app.add_url_rule('/api/camera/capture', 'legacy_camera_capture',
                            legacy_app.capture_image, methods=['POST'])
            app.logger.info("[STARTUP] ✅ Đăng ký camera routes từ app.py gốc")
    except Exception as e:
        app.logger.warning(f"[STARTUP] ⚠️ Không thể đăng ký camera routes từ app.py: {e}")
        import traceback
        app.logger.debug(traceback.format_exc())
    
    return app
