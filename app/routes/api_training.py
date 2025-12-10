"""
API routes for training and model management
Các API cho huấn luyện AI và quản lý model
"""
from flask import Blueprint, jsonify, request, current_app
import base64
import cv2
import numpy as np
from app import globals as app_globals

training_api_bp = Blueprint('training_api', __name__, url_prefix='/api')

# Will be initialized in app context
face_service = None
training_service = None
antispoof_service = None
USE_FACENET = False


@training_api_bp.route('/update_faces', methods=['POST'])
def update_faces():
    """API cập nhật khuôn mặt (reload cache)"""
    try:
        # Trigger reload known faces
        # This will be implemented via a service function
        current_app.logger.info("Reloading known faces cache...")
        
        # For now, just return success
        # TODO: Implement actual reload logic
        return 'Cap nhat thanh cong', 200
    except Exception as e:
        current_app.logger.error(f"Error updating faces: {e}", exc_info=True)
        return f'Loi: {e}', 500


@training_api_bp.route('/train/start', methods=['POST'])
def api_train_start():
    """Bắt đầu training classifier với FaceNet embeddings"""
    global training_service, face_service, USE_FACENET
    
    if not USE_FACENET or face_service is None:
        return jsonify({'error': 'FaceNet service not available'}), 400
    
    try:
        # Khởi tạo training service nếu chưa có
        if training_service is None:
            from services.training_service import TrainingService
            training_service = TrainingService(face_service)
        
        # Kiểm tra trước khi huấn luyện: đảm bảo mỗi sinh viên có đủ mẫu theo cấu hình
        stats = training_service.get_training_stats()
        not_ready = [s for s in (stats.get('students') or []) if not s.get('ready')]
        if not_ready:
            return jsonify({
                'success': False,
                'error': 'Insufficient training data',
                'details': {
                    'min_required': stats.get('min_samples_required'),
                    'students': not_ready
                }
            }), 400

        # Train classifier
        success = training_service.train_classifier()

        if success:
            stats = training_service.get_training_stats()
            return jsonify({
                'success': True,
                'message': 'Training completed successfully',
                'stats': stats
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Training failed - see server logs'
            }), 500
    
    except Exception as e:
        current_app.logger.error(f"Training error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@training_api_bp.route('/train/status', methods=['GET'])
def api_train_status():
    """Lấy thông tin về training data"""
    global training_service, face_service, USE_FACENET
    
    if not USE_FACENET or face_service is None:
        return jsonify({'error': 'FaceNet service not available'}), 400
    
    try:
        if training_service is None:
            from services.training_service import TrainingService
            training_service = TrainingService(face_service)
        
        stats = training_service.get_training_stats()
        return jsonify({'success': True, 'stats': stats})
    
    except Exception as e:
        current_app.logger.error(f"Error getting training status: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@training_api_bp.route('/antispoof/check', methods=['POST'])
def api_antispoof_check():
    """Kiểm tra anti-spoofing cho frame hiện tại"""
    global antispoof_service, USE_FACENET
    
    if not USE_FACENET or antispoof_service is None:
        return jsonify({'error': 'Anti-spoof service not available'}), 400
    
    try:
        # Lấy ảnh từ request (base64 hoặc tải lên file)
        if 'image_data' in request.form:
            # Ảnh Base64
            image_data = request.form['image_data']
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            img_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        elif 'image' in request.files:
            # Tải lên file
            file = request.files['image']
            img_bytes = file.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        # Kiểm tra giả mạo
        result = antispoof_service.check_frame(frame)
        
        return jsonify({
            'success': True,
            'is_real': result['is_real'],
            'confidence': result['confidence'],
            'message': result['message'],
            'bbox': result['bbox']
        })
    
    except Exception as e:
        current_app.logger.error(f"Anti-spoof check error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500
