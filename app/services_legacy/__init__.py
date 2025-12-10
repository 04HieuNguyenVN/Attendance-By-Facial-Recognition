"""
Các dịch vụ nhận diện khuôn mặt nâng cao cho hệ thống điểm danh.

Gói này cung cấp:
- Embeddings dựa trên FaceNet (chính xác hơn thư viện face_recognition)
- Phát hiện và căn chỉnh khuôn mặt bằng MTCNN
- Phát hiện chống giả mạo để ngăn chặn các cuộc tấn công bằng ảnh/video
- Quy trình huấn luyện để đăng ký sinh viên mới
"""

__all__ = [
	'FaceRecognitionService',
	'AntiSpoofService',
	'TrainingService',
]

# Tránh import nặng (TensorFlow, Torch) ngay khi package được import.
# Các module cụ thể sẽ được import tường minh ở nơi cần dùng.
