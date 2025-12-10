"""
API routes for students
Các API endpoint cho quản lý sinh viên
"""
from flask import Blueprint, jsonify, request
from database import db
from app.middleware.auth import role_required
from app.utils import (
    get_request_data,
    serialize_student_record,
    save_uploaded_face_image,
    save_base64_face_image,
    safe_delete_file
)

student_api_bp = Blueprint('student_api', __name__, url_prefix='/api/students')


@student_api_bp.route('', methods=['GET'])
@role_required('admin', 'teacher')
def get_students():
    """Lấy danh sách sinh viên."""
    try:
        class_id = request.args.get('class_id', type=int)
        students = db.get_students(class_id=class_id) if class_id else db.get_all_students()
        
        # Serialize students
        class_map = {c['id']: c['class_name'] for c in db.get_all_classes()}
        result = [serialize_student_record(s, class_map) for s in students]
        
        return jsonify({'success': True, 'data': result})  # Changed 'students' to 'data'
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@student_api_bp.route('', methods=['POST'])
@role_required('admin')
def create_student():
    """Tạo sinh viên mới."""
    try:
        data = get_request_data()
        student_id = data.get('student_id', '').strip()
        full_name = data.get('full_name', '').strip()
        class_id = data.get('class_id')
        
        if not student_id or not full_name:
            return jsonify({'success': False, 'message': 'Thiếu thông tin bắt buộc'}), 400
        
        # Kiểm tra trùng lặp
        existing = db.get_student(student_id)
        if existing:
            return jsonify({'success': False, 'message': f'Mã sinh viên {student_id} đã tồn tại'}), 400
        
        # Xử lý ảnh nếu có
        face_images = request.files.getlist('face_images')
        saved_paths = []
        
        try:
            for img_file in face_images:
                if img_file and img_file.filename:
                    path = save_uploaded_face_image(img_file, student_id, full_name)
                    saved_paths.append(path)
            
            # Tạo sinh viên
            db.add_student(student_id, full_name, class_id)
            
            return jsonify({
                'success': True,
                'message': 'Đã thêm sinh viên thành công',
                'student_id': student_id,
                'images_saved': len(saved_paths)
            })
            
        except Exception as img_error:
            # Xóa ảnh đã lưu nếu có lỗi
            for path in saved_paths:
                safe_delete_file(path)
            raise img_error
            
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@student_api_bp.route('/<student_id>', methods=['GET', 'PUT', 'DELETE'])
@role_required('admin')
def manage_student(student_id):
    """Quản lý một sinh viên cụ thể (GET/PUT/DELETE)."""
    try:
        student = db.get_student(student_id)
        if not student:
            return jsonify({'success': False, 'message': 'Không tìm thấy sinh viên'}), 404
        
        if request.method == 'GET':
            return jsonify({'success': True, 'student': serialize_student_record(student)})
        
        elif request.method == 'PUT':
            data = get_request_data()
            full_name = data.get('full_name', '').strip()
            class_id = data.get('class_id')
            
            if not full_name:
                return jsonify({'success': False, 'message': 'Tên sinh viên không được để trống'}), 400
            
            db.update_student(student_id, full_name, class_id)
            return jsonify({'success': True, 'message': 'Đã cập nhật sinh viên thành công'})
        
        elif request.method == 'DELETE':
            db.delete_student(student_id)
            return jsonify({'success': True, 'message': 'Đã xóa sinh viên thành công'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500
