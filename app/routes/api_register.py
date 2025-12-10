"""
API routes for quick registration
Đăng ký nhanh sinh viên qua webcam + upload ảnh
"""
from flask import Blueprint, jsonify, request, current_app
from datetime import datetime
from database import db
from app.utils.file_utils import (
    save_base64_face_image,
    save_uploaded_face_image,
    safe_delete_file,
    extract_face_encoding
)
import os

quick_register_api_bp = Blueprint('quick_register_api', __name__, url_prefix='/api')

# Configuration
MIN_FACE_SAMPLES_PER_STUDENT = max(3, int(os.getenv('MIN_FACE_SAMPLES', '3')))
MAX_FACE_SAMPLES_PER_REQUEST = max(
    MIN_FACE_SAMPLES_PER_STUDENT,
    int(os.getenv('MAX_FACE_SAMPLES', '12')),
)


@quick_register_api_bp.route('/quick-register', methods=['POST'])
def api_quick_register():
    """API đăng ký nhanh khuôn mặt (từ webcam + file upload)"""
    try:
        data = request.form
        student_id = data.get('student_id', '').strip()
        full_name = data.get('full_name', '').strip()

        current_app.logger.info(f"Quick register request - ID: {student_id}, Name: {full_name}")
        current_app.logger.info(f"Form keys: {list(data.keys())}")
        current_app.logger.info(f"Files keys: {list(request.files.keys())}")

        if not all([student_id, full_name]):
            return jsonify({'error': 'Mã sinh viên và họ tên là bắt buộc'}), 400

        # Thu thập tất cả nguồn ảnh (webcam base64 + file upload)
        sample_sources = []
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

        # Base64 images từ webcam
        base64_fields = sorted(key for key in data.keys() if key.startswith('image_data'))
        for key in base64_fields:
            image_payload = data.get(key)
            if image_payload:
                sample_sources.append(('base64', image_payload))

        # Upload files
        file_candidates = []
        if request.files:
            file_candidates.extend(request.files.getlist('face_images'))
            file_candidates.extend(request.files.getlist('face_image'))

        seen_file_ids = set()
        for file_storage in file_candidates:
            if not file_storage or not file_storage.filename:
                continue
            marker = id(file_storage)
            if marker in seen_file_ids:
                continue
            seen_file_ids.add(marker)
            sample_sources.append(('upload', file_storage))

        # Validate số lượng ảnh
        if len(sample_sources) < MIN_FACE_SAMPLES_PER_STUDENT:
            return jsonify({'error': f'Cần tối thiểu {MIN_FACE_SAMPLES_PER_STUDENT} ảnh khuôn mặt (có thể chụp nhiều lần hoặc chọn nhiều file).'}), 400
        if len(sample_sources) > MAX_FACE_SAMPLES_PER_REQUEST:
            return jsonify({'error': f'Tối đa {MAX_FACE_SAMPLES_PER_REQUEST} ảnh khuôn mặt cho mỗi lần đăng ký.'}), 400

        # Lưu ảnh
        saved_paths = []
        try:
            for idx, (source_type, payload) in enumerate(sample_sources):
                suffix = f"{idx:02d}"
                if source_type == 'base64':
                    path = save_base64_face_image(payload, student_id, full_name, suffix=suffix, timestamp=timestamp)
                else:
                    path = save_uploaded_face_image(payload, student_id, full_name, suffix=suffix, timestamp=timestamp)
                saved_paths.append(path)
        except ValueError as err:
            for path in saved_paths:
                safe_delete_file(path)
            return jsonify({'error': str(err)}), 400

        # Extract face encoding từ ảnh đầu tiên
        primary_face_path = saved_paths[0]
        face_encoding_blob = extract_face_encoding(primary_face_path)

        # Thông tin bổ sung
        email = data.get('email', f'{student_id}@student.edu.vn')
        phone = data.get('phone', '')
        class_name = data.get('class_name', 'Chưa phân lớp')

        # Tạo student record
        created, credentials = db.add_student(
            student_id,
            full_name,
            email,
            phone,
            class_name,
            primary_face_path,
            face_encoding=face_encoding_blob,
        )
        if not created:
            for path in saved_paths:
                safe_delete_file(path)
            return jsonify({'error': 'Mã sinh viên đã tồn tại'}), 400

        # Lưu tất cả face samples
        for idx, sample_path in enumerate(saved_paths):
            db.add_face_sample(student_id, sample_path, is_primary=(idx == 0))

        # Reload known faces cache (cần import từ app context)
        from app import globals as app_globals
        try:
            # Trigger reload - sẽ được implement sau
            current_app.logger.info(f"Face samples saved for {student_id}, reloading cache...")
        except Exception as e:
            current_app.logger.warning(f"Could not reload faces cache: {e}")

        payload = {
            'success': True,
            'message': f'Đăng ký thành công cho {full_name}! Đã lưu {len(saved_paths)} ảnh.',
            'samples': len(saved_paths),
        }
        if credentials:
            payload['credentials'] = credentials
        return jsonify(payload), 200

    except Exception as e:
        current_app.logger.error(f"Quick registration error: {e}", exc_info=True)
        return jsonify({'error': 'Lỗi khi đăng ký: ' + str(e)}), 500
