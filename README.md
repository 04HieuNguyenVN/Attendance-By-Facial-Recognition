# Hệ thống điểm danh bằng nhận diện khuôn mặt

## 🚀 Tính năng chính

- **Nhận diện khuôn mặt real-time** với OpenCV và face_recognition
- **Quản lý sinh viên và lớp học** đầy đủ qua giao diện web
- **Báo cáo và thống kê** chi tiết với biểu đồ
- **Database SQLite** lưu trữ dữ liệu điểm danh
- **Giao diện web responsive** với Bootstrap 5
- **Demo mode** khi không cài được face_recognition

## � Yêu cầu hệ thống

- Python 3.8 trở lên
- Webcam (hoặc sử dụng DEMO_MODE)
- Windows/Linux/MacOS

## 🛠️ Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/04HieuNguyenVN/Attendance-By-Facial-Recognition.git
cd "Attendance by facial recognition"
```

### 2. Tạo môi trường ảo

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# hoặc CMD
python -m venv .venv
.venv\Scripts\activate.bat
```

```bash
# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Cài đặt dependencies

#### Cách 1: Cài đặt đầy đủ (với face recognition)

**Lưu ý**: `dlib` và `face-recognition` yêu cầu Visual C++ Build Tools trên Windows

```powershell
# Cài đặt tất cả dependencies
pip install -r requirements.txt
```

#### Cách 2: Cài đặt cho DEMO MODE (không cần face_recognition)

```powershell
# Chỉ cài các package cơ bản (bỏ qua face-recognition và dlib)
pip install Flask opencv-python numpy pandas python-dotenv werkzeug openpyxl reportlab
```

Sau đó set biến môi trường: `DEMO_MODE=1`

### 4. Cấu hình

Tạo file `.env` từ template:

```powershell
Copy-Item .env.example .env
```

Sửa file `.env` và thay đổi `SECRET_KEY`:

```env
SECRET_KEY=your-random-secret-key-here-change-this
DEMO_MODE=0
CAMERA_INDEX=0
```

### 5. Khởi tạo database

```powershell
# Database sẽ tự động tạo khi chạy app lần đầu
# Hoặc có thể test trước:
.\.venv\Scripts\python.exe -c "from database import db; db.init_database(); print('Database initialized')"
```

## 🚀 Chạy ứng dụng

### Cách 1: Sử dụng script

```powershell
# Windows
.\start.bat
```

### Cách 2: Chạy trực tiếp

```powershell
# Windows
.\.venv\Scripts\python.exe app.py
```

```bash
# Linux/Mac
./venv/bin/python app.py
```

### Cách 3: Chạy DEMO MODE (không cần camera/face_recognition)

```powershell
$env:DEMO_MODE="1"
.\.venv\Scripts\python.exe app.py
```

## 🌐 Truy cập ứng dụng

Mở trình duyệt và truy cập: **http://localhost:5000**

### Tài khoản mặc định

- **Admin**
  - Username: `admin`
  - Password: `admin123`
- **Teacher**
  - Username: `teacher`
  - Password: `teacher123`

**⚠️ Quan trọng**: Đổi mật khẩu ngay sau lần đăng nhập đầu tiên!

## 📁 Cấu trúc dự án

```
├── app.py                      # Ứng dụng Flask chính
├── database.py                 # Quản lý SQLite database
├── logging_config.py           # Cấu hình logging
├── requirements.txt            # Python dependencies
├── .env.example                # Template cho environment variables
├── start.bat                   # Script khởi động (Windows)
├── README.md                   # Tài liệu này
├── data/                       # Ảnh khuôn mặt sinh viên (*.jpg)
├── logs/                       # Log files
├── uploads/                    # File uploads
├── templates/                  # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── students.html
│   ├── classes.html
│   ├── reports.html
│   └── ...
└── static/                     # CSS, JS, images
    ├── css/
    │   └── main.css
    ├── js/
    │   └── main.js
    └── img/
```

## � Sử dụng

### 1. Đăng ký sinh viên

1. Truy cập **Quản lý sinh viên**
2. Click **Thêm sinh viên**
3. Điền thông tin và upload ảnh khuôn mặt
4. Hệ thống sẽ tự động xử lý và lưu

### 2. Điểm danh

1. Mở trang chủ
2. Camera sẽ tự động bật
3. Sinh viên đứng trước camera
4. Hệ thống tự động nhận diện và điểm danh

### 3. Xem báo cáo

1. Truy cập **Báo cáo**
2. Chọn khoảng thời gian
3. Xem thống kê và biểu đồ
4. Xuất file Excel/PDF nếu cần

## 🔧 Cấu hình nâng cao

### Environment Variables (.env)

```env
# Flask
SECRET_KEY=your-secret-key
HOST=0.0.0.0
PORT=5000
DEBUG=True

# Camera
CAMERA_INDEX=0
DEMO_MODE=0

# Face Recognition
FACE_RECOGNITION_THRESHOLD=0.6
MIN_FACE_RATIO=0.15
PROCESS_EVERY_FRAMES=4

# Attendance
CONFIRM_SECONDS=3
PRESENCE_MAX_GAP=5
```

### Demo Mode

Khi không có camera hoặc không thể cài `face-recognition`:

```powershell
$env:DEMO_MODE="1"
.\.venv\Scripts\python.exe app.py
```

Demo mode sẽ:

- Tạo khuôn mặt mô phỏng
- Tự động "điểm danh" các sinh viên ảo
- Cho phép test giao diện mà không cần camera

## 🐛 Troubleshooting

### Lỗi: `dlib` không cài được trên Windows

**Giải pháp 1**: Cài Visual C++ Build Tools

- Tải từ: https://visualstudio.microsoft.com/visual-cpp-build-tools/
- Chọn "Desktop development with C++"

**Giải pháp 2**: Sử dụng wheel file

```powershell
pip install https://github.com/z-mahmud22/Dlib_Windows_Python3.x/raw/main/dlib-19.22.99-cp310-cp310-win_amd64.whl
```

**Giải pháp 3**: Sử dụng DEMO_MODE

```powershell
$env:DEMO_MODE="1"
```

### Lỗi: Camera không hoạt động

1. Kiểm tra `CAMERA_INDEX` trong `.env` (thử 0, 1, 2...)
2. Đảm bảo camera không bị app khác sử dụng
3. Kiểm tra quyền truy cập camera

### Lỗi: Import dotenv failed

```powershell
pip install python-dotenv
```

## 📊 API Endpoints

- `GET /` - Trang chủ điểm danh
- `GET /video_feed` - Stream video từ camera
- `POST /api/camera/toggle` - Bật/tắt camera
- `GET /api/camera/status` - Trạng thái camera
- `GET /api/attendance/today` - Điểm danh hôm nay
- `GET /api/statistics` - Thống kê
- `POST /api/register` - Đăng ký sinh viên mới
- `POST /api/login` - Đăng nhập
- `POST /api/logout` - Đăng xuất

## 🔐 Bảo mật

- Mật khẩu được hash bằng Werkzeug
- Session được mã hóa
- File upload được validate
- SQL injection prevention
- CSRF protection (Flask built-in)

## 📝 License

MIT License - Sử dụng tự do cho mục đích giáo dục và thương mại.

## 👨‍💻 Tác giả

**04HieuNguyenVN**

- GitHub: [@04HieuNguyenVN](https://github.com/04HieuNguyenVN)
- Repository: [Attendance-By-Facial-Recognition](https://github.com/04HieuNguyenVN/Attendance-By-Facial-Recognition)

## 🙏 Đóng góp

Mọi đóng góp đều được hoan nghênh! Vui lòng:

1. Fork project
2. Tạo branch mới (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

## � Hỗ trợ

Nếu gặp vấn đề, vui lòng:

1. Kiểm tra [Issues](https://github.com/04HieuNguyenVN/Attendance-By-Facial-Recognition/issues)
2. Tạo issue mới với mô tả chi tiết
3. Đính kèm log files từ `logs/` nếu có

---

**Lưu ý**: Dự án này được phát triển cho mục đích giáo dục. Sử dụng thực tế cần cân nhắc về privacy và GDPR compliance.
