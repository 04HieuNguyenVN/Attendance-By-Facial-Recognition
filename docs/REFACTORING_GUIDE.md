> # Cấu Trúc Dự Án Mới - Refactored

## 📋 Tổng Quan

Dự án đã được tổ chức lại theo mô hình **MVC (Model-View-Controller)** để dễ bảo trì và mở rộng hơn.

### ⚠️ Quan Trọng

- File `app.py` cũ (3745 dòng) đã được tách thành nhiều module nhỏ
- File mới để chạy ứng dụng: **`run.py`**
- Cấu trúc mới nằm trong thư mục **`app/`**

## 📁 Cấu Trúc Thư Mục Mới

```
project/
├── run.py                          # ⭐ File chính để chạy ứng dụng
├── app/                            # Package chính của ứng dụng
│   ├── __init__.py                 # Factory function create_app()
│   ├── config.py                   # Cấu hình constants
│   │
│   ├── routes/                     # 🛣️ Routes/Controllers
│   │   ├── __init__.py            # Đăng ký blueprints
│   │   ├── auth.py                # Login/Logout routes
│   │   ├── main.py                # Trang chủ và views cơ bản
│   │   ├── api_students.py        # API quản lý sinh viên
│   │   ├── api_classes.py         # API quản lý lớp học
│   │   ├── api_attendance.py      # TODO: API điểm danh
│   │   ├── api_camera.py          # TODO: API camera/video
│   │   └── api_statistics.py     # TODO: API thống kê
│   │
│   ├── middleware/                 # 🔒 Authentication & Authorization
│   │   ├── __init__.py
│   │   └── auth.py                # Auth middleware, decorators
│   │
│   ├── services/                   # 🧠 Business Logic
│   │   ├── __init__.py
│   │   ├── face_recognition.py   # TODO: Face recognition service
│   │   ├── attendance_service.py  # TODO: Attendance logic
│   │   └── camera_service.py      # TODO: Camera management
│   │
│   ├── utils/                      # 🔧 Helper Functions
│   │   ├── __init__.py
│   │   ├── file_utils.py          # File upload, validation
│   │   └── data_utils.py          # Data transformation
│   │
│   └── models/                     # 📊 Data Models (future use)
│       └── __init__.py
│
├── database.py                     # Database layer (giữ nguyên)
├── logging_config.py               # Logging config (giữ nguyên)
├── templates/                      # Jinja2 templates
├── static/                         # CSS, JS, images
└── data/                          # Data files
```

## 🚀 Cách Chạy Ứng Dụng

### Phương Pháp 1: Sử dụng run.py (Khuyến nghị)

```bash
python run.py
```

### Phương Pháp 2: Flask CLI

```bash
set FLASK_APP=run.py
flask run
```

### Phương Pháp 3: Với environment variables

```bash
set FLASK_HOST=0.0.0.0
set FLASK_PORT=5000
set FLASK_DEBUG=True
python run.py
```

## 📝 Chi Tiết Các Module

### 1. `app/__init__.py` - Application Factory

```python
from app import create_app
app = create_app()
```

- Tạo và cấu hình Flask application
- Đăng ký middleware và blueprints
- Setup logging

### 2. `app/config.py` - Configuration

- Chứa tất cả constants và settings
- Upload configuration
- File size limits
- Directory paths

### 3. `app/routes/` - Route Blueprints

#### `auth.py` - Authentication Routes

```
/login   (GET, POST) - Đăng nhập
/logout  (GET)       - Đăng xuất
```

#### `main.py` - Main Pages

```
/                    - Trang chủ
/students            - Quản lý sinh viên
/classes             - Quản lý lớp học
/reports             - Báo cáo
/teacher/credit-classes - Lớp tín chỉ (giáo viên)
/student/portal      - Portal sinh viên
/status              - Trạng thái hệ thống
```

#### `api_students.py` - Student API

```
GET    /api/students          - Lấy danh sách sinh viên
POST   /api/students          - Tạo sinh viên mới
GET    /api/students/<id>     - Lấy thông tin sinh viên
PUT    /api/students/<id>     - Cập nhật sinh viên
DELETE /api/students/<id>     - Xóa sinh viên
```

#### `api_classes.py` - Class API

```
GET    /api/classes           - Lấy danh sách lớp
POST   /api/classes           - Tạo lớp mới
GET    /api/classes/<id>      - Lấy thông tin lớp
PUT    /api/classes/<id>      - Cập nhật lớp
DELETE /api/classes/<id>      - Xóa lớp
GET    /api/classes/<id>/students - Sinh viên trong lớp
```

### 4. `app/middleware/auth.py` - Authentication Middleware

- `load_logged_in_user()` - Load user từ session
- `role_required()` - Decorator kiểm tra quyền
- `login_user()` - Tạo session
- `logout_current_user()` - Xóa session
- `verify_user_password()` - Xác thực mật khẩu

### 5. `app/utils/` - Utility Functions

#### `file_utils.py`

- `save_uploaded_face_image()` - Lưu ảnh upload
- `save_base64_face_image()` - Lưu ảnh base64
- `validate_image_file()` - Validate ảnh
- `safe_delete_file()` - Xóa file an toàn

#### `data_utils.py`

- `row_to_dict()` - Convert SQLite row to dict
- `parse_datetime_safe()` - Parse datetime string
- `get_request_data()` - Get JSON/form data
- `serialize_student_record()` - Serialize student data

## 🔄 Migration từ app.py cũ

### Đã Hoàn Thành ✅

- [x] Tách authentication logic → `app/middleware/auth.py`
- [x] Tách file utilities → `app/utils/file_utils.py`
- [x] Tách data utilities → `app/utils/data_utils.py`
- [x] Tách login/logout routes → `app/routes/auth.py`
- [x] Tách main pages → `app/routes/main.py`
- [x] Tách student API → `app/routes/api_students.py`
- [x] Tách class API → `app/routes/api_classes.py`
- [x] Tạo configuration → `app/config.py`
- [x] Tạo application factory → `app/__init__.py`
- [x] Tạo entry point mới → `run.py`

### Còn Lại (TODO) ⏳

- [ ] Tách attendance API → `app/routes/api_attendance.py`
- [ ] Tách camera/video API → `app/routes/api_camera.py`
- [ ] Tách statistics API → `app/routes/api_statistics.py`
- [ ] Tách credit class API → `app/routes/api_credit_classes.py`
- [ ] Tách reports API → `app/routes/api_reports.py`
- [ ] Tách face recognition service → `app/services/face_recognition.py`
- [ ] Tách attendance service → `app/services/attendance_service.py`
- [ ] Tách camera service → `app/services/camera_service.py`

## 🎯 Lợi Ích Của Cấu Trúc Mới

### 1. **Dễ Bảo Trì**

- Mỗi file có trách nhiệm rõ ràng
- Dễ tìm và fix bug
- Code ngắn gọn hơn (mỗi file ~100-300 dòng)

### 2. **Dễ Mở Rộng**

- Thêm feature mới = Thêm file mới
- Không ảnh hưởng code cũ
- Blueprint system cho phép module hóa

### 3. **Dễ Test**

- Mỗi module có thể test riêng
- Mock dependencies dễ dàng
- Unit test cho từng function

### 4. **Dễ Làm Việc Nhóm**

- Nhiều người code cùng lúc không conflict
- Mỗi người phụ trách một module
- Code review dễ hơn

### 5. **Reusable Code**

- Utils có thể dùng ở nhiều nơi
- Services có thể inject vào routes
- Middleware áp dụng toàn app

## 📊 So Sánh

| Aspect           | app.py Cũ          | Cấu Trúc Mới       |
| ---------------- | ------------------ | ------------------ |
| **Số dòng/file** | 3745 dòng          | ~100-300 dòng/file |
| **Số files**     | 1 file             | 15+ files          |
| **Tổ chức**      | Monolithic         | Modular            |
| **Tìm bug**      | Khó (scroll nhiều) | Dễ (file cụ thể)   |
| **Thêm feature** | Append vào cuối    | Tạo file mới       |
| **Test**         | Khó                | Dễ                 |
| **Team work**    | Conflict nhiều     | Conflict ít        |

## 🔧 Development Workflow

### Thêm một route mới:

1. Tạo function trong file blueprint phù hợp (`app/routes/`)
2. Hoặc tạo blueprint mới nếu cần
3. Đăng ký blueprint trong `app/routes/__init__.py`

### Thêm một utility function:

1. Thêm vào `app/utils/file_utils.py` hoặc `data_utils.py`
2. Export trong `app/utils/__init__.py`
3. Import và sử dụng ở routes

### Thêm một service:

1. Tạo file trong `app/services/`
2. Implement business logic
3. Inject vào routes cần sử dụng

## 🐛 Troubleshooting

### Lỗi: `ModuleNotFoundError: No module named 'app'`

**Giải pháp**: Đảm bảo chạy từ thư mục gốc của project

```bash
cd "g:\Python\Attendance by facial recognition"
python run.py
```

### Lỗi: Template not found

**Giải pháp**: Kiểm tra path trong `app/__init__.py`

```python
app = Flask(__name__,
            template_folder='../templates',  # Đúng path
            static_folder='../static')
```

### Lỗi: Import errors

**Giải pháp**: Cài đặt lại dependencies

```bash
pip install -r requirements.txt
```

## 📚 Best Practices

### 1. Import Order

```python
# Standard library
import os
import sys

# Third-party
from flask import Flask, render_template

# Local application
from app.middleware.auth import role_required
from app.utils import get_request_data
```

### 2. Blueprint Naming

- File: `api_students.py`
- Blueprint name: `student_api_bp`
- URL prefix: `/api/students`

### 3. Error Handling

```python
try:
    # Your logic
    return jsonify({'success': True, 'data': data})
except Exception as e:
    app.logger.error(f"Error: {str(e)}")
    return jsonify({'success': False, 'message': str(e)}), 500
```

## 🚀 Next Steps

1. **Tiếp tục refactoring**: Tách các route còn lại từ `app.py`
2. **Tạo services layer**: Di chuyển business logic vào `app/services/`
3. **Thêm tests**: Tạo `tests/` folder với unit tests
4. **Documentation**: Thêm docstrings cho tất cả functions
5. **Type hints**: Thêm type annotations cho better IDE support

## 📖 Tài Liệu Tham Khảo

- [Flask Blueprints](https://flask.palletsprojects.com/en/2.3.x/blueprints/)
- [Application Factory Pattern](https://flask.palletsprojects.com/en/2.3.x/patterns/appfactories/)
- [Flask Project Structure](https://flask.palletsprojects.com/en/2.3.x/tutorial/layout/)

---

**Version**: 1.0  
**Date**: 2024-12-04  
**Status**: ✅ Partial Refactoring Complete (Authentication, Students, Classes done)
