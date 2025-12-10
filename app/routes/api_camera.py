"""
API routes for camera and video feed
Camera management và video streaming  
TEMPORARY: Proxy to app.py routes until full migration
"""
from flask import Blueprint

camera_api_bp = Blueprint('camera_api', __name__)

# Import các route từ app.py gốc để tái sử dụng
# NOTE: Đây là giải pháp tạm thời - các routes này vẫn hoạt động tốt trong app.py
# Migration đầy đủ sẽ được thực hiện sau khi có thời gian refactor toàn bộ global state

# Camera routes sẽ được đăng ký trực tiếp từ app.py cho đến khi migration hoàn tất
# Hiện tại app.py có:
# - /video_feed (with generate_frames logic)
# - /api/camera/toggle
# - /api/camera/status  
# - /api/camera/capture
# Tất cả đã có phân quyền đúng @role_required('student')

