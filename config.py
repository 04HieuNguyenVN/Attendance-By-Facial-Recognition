# config.py - Configuration and constants for the attendance system

import os
from pathlib import Path
from typing import Optional

# Cố gắng tải dotenv, nhưng không báo lỗi nếu chưa cài đặt
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Using default configuration.")
    print("Install it with: pip install python-dotenv")

# Flask app configuration
SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

# Upload configuration
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
SUPPORTED_IMAGE_FORMATS = {'JPEG', 'PNG', 'WEBP'}
MIN_FILE_SIZE = 1024  # 1 KB
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
MIN_FACE_SAMPLES_PER_STUDENT = max(3, int(os.getenv('MIN_FACE_SAMPLES', '3')))
MAX_FACE_SAMPLES_PER_REQUEST = max(
    MIN_FACE_SAMPLES_PER_STUDENT,
    int(os.getenv('MAX_FACE_SAMPLES', '12')),
)

# Demo mode configuration
DEMO_MODE = os.getenv('DEMO_MODE', '0') == '1'
USE_FACENET = os.getenv('USE_FACENET', '1') == '1'

# Camera configuration
CAMERA_INDEX = int(os.getenv('CAMERA_INDEX', '0'))
CAMERA_WIDTH = int(os.getenv('CAMERA_WIDTH', '640'))
CAMERA_HEIGHT = int(os.getenv('CAMERA_HEIGHT', '480'))
CAMERA_WARMUP_FRAMES = int(os.getenv('CAMERA_WARMUP_FRAMES', '3'))
CAMERA_BUFFER_SIZE = int(os.getenv('CAMERA_BUFFER_SIZE', '2'))

# Face recognition thresholds
FACE_RECOGNITION_THRESHOLD = float(os.getenv('FACE_RECOGNITION_THRESHOLD', '0.6'))
FACE_DISTANCE_THRESHOLD = float(os.getenv('FACE_DISTANCE_THRESHOLD', '0.45'))
DEEPFACE_SIMILARITY_THRESHOLD = float(os.getenv('DEEPFACE_SIMILARITY_THRESHOLD', '0.6'))

# Pose verification thresholds
LOOK_STRAIGHT_SECONDS = float(os.getenv('LOOK_STRAIGHT_SECONDS', '10'))
FRONTAL_YAW_RATIO_THRESHOLD = float(os.getenv('FRONTAL_YAW_RATIO_THRESHOLD', '0.15'))
FRONTAL_ROLL_DEG_THRESHOLD = float(os.getenv('FRONTAL_ROLL_DEG_THRESHOLD', '15'))

# Performance optimization
YOLO_FRAME_SKIP = max(1, int(os.getenv('YOLO_FRAME_SKIP', '2')))
YOLO_INFERENCE_WIDTH = int(os.getenv('YOLO_INFERENCE_WIDTH', '640'))
SESSION_DURATION_MINUTES = max(1, int(os.getenv('SESSION_DURATION_MINUTES', '15')))

# FaceNet service thresholds
FACENET_THRESHOLD = float(os.getenv('FACENET_THRESHOLD', '0.85'))
ANTISPOOF_DEVICE = os.getenv('ANTISPOOF_DEVICE', 'cpu')
ANTISPOOF_THRESHOLD = float(os.getenv('ANTISPOOF_THRESHOLD', '0.5'))

# Data directory
DATA_DIR = Path('data')
RESERVED_DATA_SUBDIRS = {'training_samples', 'models', 'external_assets'}

# Legacy class names to clean up
LEGACY_CLASS_NAMES = [
    'Công nghệ thông tin 01',
    'Công nghệ thông tin 01',
]

# Public endpoints
PUBLIC_ENDPOINTS = {'login', 'logout', 'static'}

# Presence tracking timeout
PRESENCE_TIMEOUT = 300  # 5 minutes

# Recognition cooldown
RECOGNITION_COOLDOWN = 30  # seconds