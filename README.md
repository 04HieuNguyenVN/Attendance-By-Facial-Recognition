# Hệ thống điểm danh bằng nhận diện khuôn mặt

## 🚀 Tính năng chính

- **Nhận diện khuôn mặt real-time** với bounding box cải tiến
- **Quản lý sinh viên và lớp học** đầy đủ
- **Báo cáo và thống kê** chi tiết
- **Tối ưu hóa hiệu suất** với caching và parallel processing
- **Giao diện web responsive** với Bootstrap 5
- **API RESTful** cho mobile app

## 📁 Cấu trúc dự án

```
├── app.py                          # Ứng dụng chính
├── database.py                     # Quản lý database
├── logging_config.py               # Cấu hình logging
├── core/                          # Modules core
│   ├── face_recognition_optimizer.py
│   ├── camera_optimizer.py
│   └── face_display_enhancer.py
├── templates/                     # HTML templates
│   ├── base.html                  # Template cơ sở
│   ├── index.html                 # Trang chủ
│   ├── students.html              # Quản lý sinh viên
│   ├── classes.html               # Quản lý lớp học
│   ├── reports.html               # Báo cáo
│   ├── performance.html           # Hiệu suất
│   └── settings.html              # Cài đặt
├── static/                        # Static files
│   ├── css/
│   │   ├── main.css               # CSS chính
│   │   └── components.css          # CSS components
│   └── js/
│       ├── main.js                # JavaScript chính
│       └── features.js            # JavaScript features
├── data/                          # Ảnh khuôn mặt sinh viên
├── uploads/                        # Files upload
├── requirements.txt               # Dependencies
├── run.bat                        # Script chạy hệ thống
└── demo.bat                       # Script demo
```

## 🛠️ Cài đặt và chạy

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Chạy hệ thống

```bash
# Windows
run.bat

# Hoặc trực tiếp
python app.py
```

### 3. Chạy demo mode

```bash
# Windows
demo.bat
```

### 4. Truy cập ứng dụng

- **Web**: http://localhost:5000
- **Demo**: http://localhost:5000 (chạy demo.bat)

## 🎯 Sử dụng

1. **Thêm sinh viên**: Upload ảnh khuôn mặt trong thư mục `data/`
2. **Điểm danh**: Mở camera và nhận diện tự động
3. **Xem báo cáo**: Truy cập trang Reports
4. **Quản lý lớp**: Sử dụng trang Classes
5. **Theo dõi hiệu suất**: Xem trang Performance

## 🔧 Cấu hình

- **Camera**: Thay đổi `CAMERA_INDEX` trong app.py
- **Face Recognition**: Điều chỉnh `MATCH_THRESHOLD` và `PROCESS_EVERY`
- **Database**: SQLite tự động tạo trong `attendance_system.db`
- **Demo Mode**: Set biến môi trường `DEMO_MODE=1`

## 📊 API Endpoints

- `GET /api/students` - Danh sách sinh viên
- `POST /api/students` - Thêm sinh viên
- `GET /api/attendance/history` - Lịch sử điểm danh
- `GET /api/performance/stats` - Thống kê hiệu suất
- `POST /api/performance/optimize` - Tối ưu hóa

## 🎨 Tính năng nâng cao

- **Enhanced Face Display**: Bounding box với góc nổi bật
- **Adaptive Threshold**: Tự động điều chỉnh ngưỡng nhận diện
- **Parallel Processing**: Xử lý nhiều khuôn mặt song song
- **Performance Monitoring**: Theo dõi hiệu suất real-time
- **Camera Optimization**: Tối ưu hóa camera settings
- **Modular Architecture**: Cấu trúc module rõ ràng
- **Responsive Design**: Giao diện thích ứng mọi thiết bị

## 🎨 Frontend Architecture

### CSS Structure

- **main.css**: Styles chính, variables, global styles
- **components.css**: Styles cho các component cụ thể

### JavaScript Structure

- **main.js**: Core functionality, utilities, API calls
- **features.js**: Feature-specific functionality (students, classes, reports)

### Template Structure

- **base.html**: Template cơ sở với navigation và layout
- **Individual templates**: Kế thừa từ base.html

## 🐛 Troubleshooting

- **Camera không hoạt động**: Kiểm tra `CAMERA_INDEX`
- **Face recognition chậm**: Giảm `PROCESS_EVERY`
- **Database lỗi**: Xóa `attendance_system.db` để tạo lại
- **Import lỗi**: Chạy `pip install -r requirements.txt`
- **CSS/JS không load**: Kiểm tra đường dẫn static files

## 📝 License

MIT License - Sử dụng tự do cho mục đích giáo dục và thương mại.
