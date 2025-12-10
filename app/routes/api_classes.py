"""
API routes for classes
Các API endpoint cho quản lý lớp học
"""
from flask import Blueprint, jsonify, request
from database import db
from app.middleware.auth import role_required
from app.utils import get_request_data

class_api_bp = Blueprint('class_api', __name__, url_prefix='/api/classes')


@class_api_bp.route('', methods=['GET'])
@role_required('admin', 'teacher')
def get_classes():
    """Lấy danh sách lớp học."""
    try:
        classes = db.get_all_classes()
        return jsonify({'success': True, 'classes': [dict(c) for c in classes]})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@class_api_bp.route('', methods=['POST'])
@role_required('admin')
def create_class():
    """Tạo lớp học mới."""
    try:
        data = get_request_data()
        class_name = data.get('class_name', '').strip()
        
        if not class_name:
            return jsonify({'success': False, 'message': 'Tên lớp không được để trống'}), 400
        
        # Kiểm tra trùng lặp
        existing = db.get_class_by_name(class_name)
        if existing:
            return jsonify({'success': False, 'message': f'Lớp {class_name} đã tồn tại'}), 400
        
        class_id = db.add_class(class_name)
        return jsonify({
            'success': True,
            'message': 'Đã thêm lớp học thành công',
            'class_id': class_id
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@class_api_bp.route('/<int:class_id>', methods=['GET', 'PUT', 'DELETE'])
@role_required('admin')
def manage_class(class_id):
    """Quản lý một lớp học cụ thể (GET/PUT/DELETE)."""
    try:
        class_obj = db.get_class_by_id(class_id)
        if not class_obj:
            return jsonify({'success': False, 'message': 'Không tìm thấy lớp học'}), 404
        
        if request.method == 'GET':
            return jsonify({'success': True, 'class': dict(class_obj)})
        
        elif request.method == 'PUT':
            data = get_request_data()
            class_name = data.get('class_name', '').strip()
            
            if not class_name:
                return jsonify({'success': False, 'message': 'Tên lớp không được để trống'}), 400
            
            db.update_class(class_id, class_name)
            return jsonify({'success': True, 'message': 'Đã cập nhật lớp học thành công'})
        
        elif request.method == 'DELETE':
            # Kiểm tra xem có sinh viên nào trong lớp không
            students = db.get_students(class_id=class_id)
            if students:
                return jsonify({
                    'success': False,
                    'message': f'Không thể xóa lớp vì còn {len(students)} sinh viên'
                }), 400
            
            db.delete_class(class_id)
            return jsonify({'success': True, 'message': 'Đã xóa lớp học thành công'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@class_api_bp.route('/<int:class_id>/students', methods=['GET'])
@role_required('admin', 'teacher')
def get_class_students(class_id):
    """Lấy danh sách sinh viên trong lớp."""
    try:
        class_obj = db.get_class_by_id(class_id)
        if not class_obj:
            return jsonify({'success': False, 'message': 'Không tìm thấy lớp học'}), 404
        
        students = db.get_students(class_id=class_id)
        return jsonify({
            'success': True,
            'class': dict(class_obj),
            'students': [dict(s) for s in students]
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500
