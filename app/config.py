"""
Configuration constants và settings
"""
import os
from pathlib import Path

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

# Directory paths
UPLOAD_FOLDER = 'uploads'
DATA_FOLDER = 'data'
DATA_DIR = Path(DATA_FOLDER)
FACE_DATA_DIR = os.path.join(DATA_FOLDER, 'faces')

# Camera configuration
CAMERA_INDEX = int(os.getenv('CAMERA_INDEX', '0'))
CAMERA_WIDTH = int(os.getenv('CAMERA_WIDTH', '640'))
CAMERA_HEIGHT = int(os.getenv('CAMERA_HEIGHT', '480'))
CAMERA_WARMUP_FRAMES = int(os.getenv('CAMERA_WARMUP_FRAMES', '3'))
CAMERA_BUFFER_SIZE = int(os.getenv('CAMERA_BUFFER_SIZE', '2'))

# Face recognition configuration
FACE_RECOGNITION_THRESHOLD = float(os.getenv('FACE_RECOGNITION_THRESHOLD', '0.6'))
FACE_DISTANCE_THRESHOLD = float(os.getenv('FACE_DISTANCE_THRESHOLD', '0.45'))
DEEPFACE_SIMILARITY_THRESHOLD = float(os.getenv('DEEPFACE_SIMILARITY_THRESHOLD', '0.6'))

# Attendance configuration
PRESENCE_TIMEOUT = 300  # 5 phút (300 giây)
RECOGNITION_COOLDOWN = 30  # 30 giây

# Demo mode
DEMO_MODE = os.getenv('DEMO_MODE', '0') == '1'
USE_FACENET = os.getenv('USE_FACENET', '1') == '1'

# Session configuration
SESSION_TIMEOUT = 3600  # 1 hour
SESSION_DURATION_MINUTES = max(1, int(os.getenv('SESSION_DURATION_MINUTES', '15')))

# YOLO configuration
YOLO_FRAME_SKIP = max(1, int(os.getenv('YOLO_FRAME_SKIP', '2')))
YOLO_INFERENCE_WIDTH = int(os.getenv('YOLO_INFERENCE_WIDTH', '640'))

