"""
Advanced face recognition services for attendance system.

This package provides:
- FaceNet-based embeddings (more accurate than face_recognition library)
- MTCNN face detection and alignment
- Anti-spoofing detection to prevent photo/video attacks
- Training pipeline for registering new students
"""

__all__ = [
	'FaceRecognitionService',
	'AntiSpoofService',
	'TrainingService',
]

# Tránh import nặng (TensorFlow, Torch) ngay khi package được import.
# Các module cụ thể sẽ được import tường minh ở nơi cần dùng.
