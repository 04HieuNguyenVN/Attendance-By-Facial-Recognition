# 🎯 Quick Start - Cấu Trúc Mới

## ⚡ Khởi Chạy Nhanh

```bash
# Phương pháp 1: Sử dụng file batch (Windows)
start.bat

# Phương pháp 2: Chạy trực tiếp
python run.py

# Phương pháp 3: Flask CLI
set FLASK_APP=run.py
flask run
```

## 📁 Cấu Trúc Quan Trọng

```
project/
├── run.py           ← ⭐ FILE MỚI để chạy app
├── app/             ← 📦 Package chính
│   ├── routes/      ← 🛣️ Tất cả routes/controllers
│   ├── middleware/  ← 🔒 Authentication
│   ├── utils/       ← 🔧 Helper functions
│   └── services/    ← 🧠 Business logic (TODO)
├── app.py           ← ⚠️ FILE CŨ (3745 dòng) - Giữ lại cho reference
└── templates/       ← HTML templates
```

## 🔄 Thay Đổi Chính

| Trước              | Sau                           |
| ------------------ | ----------------------------- |
| `python app.py`    | `python run.py`               |
| 1 file (3745 dòng) | 15+ files (100-300 dòng/file) |
| Khó tìm bug        | Dễ tìm bug (file rõ ràng)     |
| Khó thêm feature   | Dễ thêm (tạo file mới)        |

## ✅ Đã Tách Xong

- ✅ Authentication (login/logout)
- ✅ Student Management API
- ✅ Class Management API
- ✅ File utilities
- ✅ Data utilities
- ✅ Auth middleware

## ⏳ Cần Tách Tiếp

- ⏳ Attendance API
- ⏳ Camera/Video Feed
- ⏳ Statistics API
- ⏳ Face Recognition Service

## 📖 Chi Tiết

Xem file **`REFACTORING_GUIDE.md`** để biết thêm chi tiết về:

- Cấu trúc đầy đủ
- Cách thêm routes mới
- Best practices
- Troubleshooting

## 🎓 Ví Dụ: Thêm Route Mới

```python
# File: app/routes/api_example.py
from flask import Blueprint, jsonify
from app.middleware.auth import role_required

example_bp = Blueprint('example', __name__, url_prefix='/api/example')

@example_bp.route('', methods=['GET'])
@role_required('admin')
def get_example():
    return jsonify({'message': 'Hello World'})
```

Sau đó đăng ký trong `app/routes/__init__.py`:

```python
from .api_example import example_bp

def register_blueprints(app):
    # ... existing code
    app.register_blueprint(example_bp)
```

## 🐛 Lỗi Thường Gặp

**Q: Module not found?**  
A: Đảm bảo chạy từ thư mục gốc project

**Q: Template not found?**  
A: Kiểm tra path trong `app/__init__.py`

**Q: Import error?**  
A: Chạy `pip install -r requirements.txt`

---

**Lưu ý**: File `app.py` cũ vẫn được giữ lại để tham khảo, nhưng không dùng nữa!
