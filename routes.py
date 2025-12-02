# routes.py - All Flask routes and API endpoints (functions only, no decorators)

# Import necessary modules
from flask import Flask, render_template, Response, redirect, url_for, request, jsonify, session, flash, g, abort, stream_with_context
import os
import csv
import time
import random
import base64
import re
import shutil
from pathlib import Path
from datetime import datetime, date, timedelta
import threading
import hashlib
from functools import wraps
from typing import Any, Dict, Optional
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash, generate_password_hash
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from config import *
from utils import *
from face_recognition import *
from attendance import *
from sse import *
from camera import *

# Import database module
import database as db

# ===== PAGE ROUTES =====

def index():
    """Trang chủ"""
    return render_template('index.html')

def login_page():
    """Trang đăng nhập"""
    return render_template('login.html')

def students_page():
    """Trang quản lý sinh viên"""
    return render_template('students.html')

def test_students_page():
    """Trang test sinh viên"""
    return render_template('test-students.html')

def reports_page():
    """Trang báo cáo"""
    return render_template('reports.html')

def classes_page():
    """Trang quản lý lớp học"""
    try:
        classes = db.get_all_classes()
        return render_template('classes.html', classes=classes)
    except Exception as error:
        print(f"Error loading classes page: {error}")
        flash('Có lỗi khi tải trang lớp học', 'error')
        return render_template('classes.html', classes=[])

def teacher_credit_classes_page():
    """Trang lớp tín chỉ cho giáo viên"""
    return render_template('teacher_credit_classes.html')

def student_portal_page():
    """Trang cổng thông tin sinh viên"""
    return render_template('student_portal.html')

# ===== API ROUTES =====

def api_get_students():
    """API lấy danh sách sinh viên"""
    try:
        students = db.get_all_students()
        return jsonify([serialize_student_record(s) for s in students])
    except Exception as e:
        print(f"Error getting students: {e}")
        return jsonify({'error': 'Không thể lấy danh sách sinh viên'}), 500

def api_create_student():
    """API tạo sinh viên mới"""
    try:
        data = get_request_data()
        if not data:
            return jsonify({'error': 'Dữ liệu không hợp lệ'}), 400

        student_id = data.get('student_id', '').strip().upper()
        full_name = data.get('full_name', '').strip()
        class_name = data.get('class_name', '').strip()
        email = data.get('email', '').strip()
        phone = data.get('phone', '').strip()

        if not all([student_id, full_name, class_name]):
            return jsonify({'error': 'Thiếu thông tin bắt buộc'}), 400

        # Validate email format
        if email and not re.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$', email):
            return jsonify({'error': 'Email không hợp lệ'}), 400

        # Check if student already exists
        existing = db.get_student_by_id(student_id)
        if existing:
            return jsonify({'error': f'Sinh viên {student_id} đã tồn tại'}), 409

        # Create student record
        student_data = {
            'student_id': student_id,
            'full_name': full_name,
            'class_name': class_name,
            'email': email,
            'phone': phone,
            'created_at': datetime.now()
        }

        success = db.create_student(student_data)
        if success:
            # Reload known faces to include new student
            load_known_faces()
            return jsonify({'message': f'Đã tạo sinh viên {student_id}', 'student': student_data}), 201
        else:
            return jsonify({'error': 'Không thể tạo sinh viên'}), 500

    except Exception as err:
        print(f"Error creating student {student_id}: {err}", exc_info=True)
        return jsonify({'error': 'Lỗi hệ thống'}), 500

def api_get_student(student_id):
    """API lấy thông tin sinh viên theo ID"""
    try:
        student = db.get_student_by_id(student_id.upper())
        if student:
            return jsonify(serialize_student_record(student))
        else:
            return jsonify({'error': 'Sinh viên không tồn tại'}), 404
    except Exception as e:
        print(f"Error getting student {student_id}: {e}")
        return jsonify({'error': 'Không thể lấy thông tin sinh viên'}), 500

def api_update_student(student_id):
    """API cập nhật thông tin sinh viên"""
    try:
        data = get_request_data()
        if not data:
            return jsonify({'error': 'Dữ liệu không hợp lệ'}), 400

        student_id = student_id.upper()
        existing = db.get_student_by_id(student_id)
        if not existing:
            return jsonify({'error': 'Sinh viên không tồn tại'}), 404

        # Update allowed fields
        update_data = {}
        if 'full_name' in data:
            update_data['full_name'] = data['full_name'].strip()
        if 'class_name' in data:
            update_data['class_name'] = data['class_name'].strip()
        if 'email' in data:
            email = data['email'].strip()
            if email and not re.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$', email):
                return jsonify({'error': 'Email không hợp lệ'}), 400
            update_data['email'] = email
        if 'phone' in data:
            update_data['phone'] = data['phone'].strip()

        success = db.update_student(student_id, update_data)
        if success:
            return jsonify({'message': f'Đã cập nhật sinh viên {student_id}'})
        else:
            return jsonify({'error': 'Không thể cập nhật sinh viên'}), 500

    except Exception as e:
        print(f"Error updating student {student_id}: {e}")
        return jsonify({'error': 'Lỗi hệ thống'}), 500

def api_delete_student(student_id):
    """API xóa sinh viên"""
    try:
        student_id = student_id.upper()
        existing = db.get_student_by_id(student_id)
        if not existing:
            return jsonify({'error': 'Sinh viên không tồn tại'}), 404

        success = db.delete_student(student_id)
        if success:
            # Reload known faces
            load_known_faces()
            return jsonify({'message': f'Đã xóa sinh viên {student_id}'})
        else:
            return jsonify({'error': 'Không thể xóa sinh viên'}), 500

    except Exception as e:
        print(f"Error deleting student {student_id}: {e}")
        return jsonify({'error': 'Lỗi hệ thống'}), 500

def api_get_classes():
    """API lấy danh sách lớp học"""
    try:
        classes = db.get_all_classes()
        return jsonify(classes)
    except Exception as e:
        print(f"Error getting classes: {e}")
        return jsonify({'error': 'Không thể lấy danh sách lớp học'}), 500

def api_create_class():
    """API tạo lớp học mới"""
    try:
        data = get_request_data()
        if not data:
            return jsonify({'error': 'Dữ liệu không hợp lệ'}), 400

        class_name = data.get('class_name', '').strip()
        class_type = data.get('class_type', 'administrative')

        if not class_name:
            return jsonify({'error': 'Tên lớp không được để trống'}), 400

        if class_type not in ['administrative', 'credit']:
            return jsonify({'error': 'Loại lớp không hợp lệ'}), 400

        # Check if class already exists
        existing = db.get_class_by_name(class_name)
        if existing:
            return jsonify({'error': f'Lớp {class_name} đã tồn tại'}), 409

        success = db.create_class(class_name, class_type)
        if success:
            return jsonify({'message': f'Đã tạo lớp {class_name}', 'class_name': class_name, 'class_type': class_type}), 201
        else:
            return jsonify({'error': 'Không thể tạo lớp học'}), 500

    except Exception as e:
        print(f"Error creating class: {e}")
        return jsonify({'error': 'Lỗi hệ thống'}), 500

# ===== UTILITY FUNCTIONS =====

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_face_image(image_data, student_id):
    """Process uploaded face image"""
    # Implementation here
    pass

def prewhiten_facenet(x):
    """Prewhiten image for FaceNet"""
    # Implementation here
    pass

def estimate_head_pose(image, landmarks):
    """Estimate head pose from facial landmarks"""
    # Implementation here
    pass

def draw_progress_bar(frame, progress, position=(10, 10), size=(200, 20)):
    """Draw progress bar on frame"""
    # Implementation here
    pass

def update_progress(student_id, progress):
    """Update attendance progress for student"""
    # Implementation here
    pass

def reset_progress(student_id=None):
    """Reset attendance progress"""
    # Implementation here
    pass