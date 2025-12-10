# 📋 Phân Chia Các Đầu Công Việc - Attendance By Facial Recognition

Tài liệu này phân chia các module và công việc chính trong dự án hệ thống điểm danh bằng nhận diện khuôn mặt.

---

## 1. 🖥️ Backend / Flask Application

| File | Công việc | Mô tả |
|------|-----------|-------|
| `app.py` | Flask Routes & API | Định tuyến, xử lý request/response |
| `app.py` | Session Management | Quản lý phiên đăng nhập, authentication |
| `app.py` | Video Streaming | Stream video từ camera qua HTTP |
| `app.py` | Attendance Logic | Logic điểm danh, progress confirmation (30 frames) |
| `sse.py` | SSE Events | Server-Sent Events cho thông báo real-time |
| `run.py` | Entry Point | Điểm khởi động ứng dụng |

---

## 2. 🗄️ Database Management

| File | Công việc | Mô tả |
|------|-----------|-------|
| `database.py` | Quản lý sinh viên | CRUD sinh viên, thông tin cá nhân |
| `database.py` | Quản lý lớp học | Lớp tín chỉ, ghi danh, lịch học |
| `database.py` | Quản lý điểm danh | Bản ghi check-in/check-out, phiên điểm danh |
| `database.py` | Quản lý người dùng | Tài khoản admin/teacher/student |
| `database.py` | Quản lý mẫu khuôn mặt | `student_face_samples`, embeddings |

### Bảng dữ liệu chính:
- `users` - Tài khoản đăng nhập
- `students` - Thông tin sinh viên
- `classes` - Lớp học
- `credit_classes` - Lớp tín chỉ
- `attendance` - Bản ghi điểm danh
- `attendance_sessions` - Phiên điểm danh
- `student_face_samples` - Mẫu ảnh khuôn mặt

---

## 3. 👁️ Computer Vision / Xử Lý Ảnh

### 3.1 Core Vision (`core/vision/`)

| File | Công việc | Mô tả |
|------|-----------|-------|
| `camera_manager.py` | Camera Connection | Kết nối camera, cấu hình resolution, buffer |
| `camera_manager.py` | Warmup Frames | "Làm nóng" camera để ổn định ánh sáng |
| `pipeline.py` | Quality Assessment | Đánh giá chất lượng ảnh (Laplacian variance) |
| `pipeline.py` | Color Conversion | Chuyển đổi BGR ↔ RGB |
| `state.py` | Pipeline State | Quản lý trạng thái xử lý video |

### 3.2 Image Processing Operations

| Operation | OpenCV Function | File |
|-----------|-----------------|------|
| Chuyển xám | `cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)` | `pipeline.py` |
| Đánh giá blur | `cv2.Laplacian(gray, cv2.CV_64F).var()` | `pipeline.py` |
| Resize ảnh | `cv2.resize()` với `INTER_CUBIC/INTER_LINEAR` | `face_service.py`, `camera.py` |
| Flip ảnh | `cv2.flip(frame, 1)` | `camera.py` |
| Vẽ rectangle | `cv2.rectangle()` | `camera.py` |
| Vẽ text | `cv2.putText()` | `camera.py` |
| Encode JPEG | `cv2.imencode('.jpg', frame)` | `camera.py` |

---

## 4. 🤖 AI / Face Recognition

### 4.1 Face Detection

| Phương pháp | File | Mô tả |
|-------------|------|-------|
| **YOLOv8** | `camera.py` | Phát hiện nhanh, chính xác (`yolov8m-face.pt`) |
| **RetinaFace** | `antispoof_service.py` | Dùng OpenCV DNN (`cv2.dnn.readNetFromCaffe`) |
| **Haar Cascade** | `face_service.py` | Fallback với `haarcascade_frontalface_default.xml` |

### 4.2 Face Recognition / Embedding

| File | Công việc | Mô tả |
|------|-----------|-------|
| `face_service.py` | Preprocess Face | Resize 160x160, Whitening (mean=0, std=1) |
| `face_service.py` | Get Embedding | Trích xuất vector 128/512 chiều |
| `core/inference/engine.py` | FaceNet Strategy | Inference với FaceNet model |
| `core/inference/engine.py` | DeepFace Strategy | Inference với DeepFace (Facenet512) |
| `services/deepface_db.py` | DeepFace DB | Quản lý database embedding với DeepFace |

### 4.3 Anti-Spoofing

| File | Công việc | Mô tả |
|------|-----------|-------|
| `antispoof_service.py` | Liveness Detection | Phát hiện ảnh giả (in, màn hình) |
| `antispoof_service.py` | MiniFASNet | Model anti-spoof PyTorch |

### 4.4 Training & Classification

| File | Công việc | Mô tả |
|------|-----------|-------|
| `training_service.py` | Train SVM | Huấn luyện SVM classifier trên embeddings |
| `training_service.py` | Save Model | Lưu `facemodel.pkl` |
| `training_service.py` | Capture Samples | Thu thập ảnh mẫu cho training |

---

## 5. 🎨 Frontend / UI

### 5.1 Templates (`templates/`)

| File | Công việc | Mô tả |
|------|-----------|-------|
| `base.html` | Layout chung | Header, navigation, footer |
| `index.html` | Trang điểm danh | Video stream, danh sách điểm danh |
| `students.html` | Quản lý sinh viên | CRUD sinh viên, upload ảnh |
| `classes.html` | Quản lý lớp | CRUD lớp học, ghi danh |
| `reports.html` | Báo cáo | Thống kê, biểu đồ, xuất file |
| `student_portal.html` | Portal sinh viên | Xem lịch sử điểm danh cá nhân |
| `login.html` | Đăng nhập | Form authentication |
| `components/` | UI Components | 20 component tái sử dụng |

### 5.2 Static Files (`static/`)

| Folder | Công việc | Mô tả |
|--------|-----------|-------|
| `css/main.css` | Styling | CSS chính, Dark Mode |
| `js/main.js` | Frontend Logic | SSE client, AJAX, DOM manipulation |
| `img/` | Images | Logo, icons |

---

## 6. ⚙️ Configuration & Utilities

| File | Công việc | Mô tả |
|------|-----------|-------|
| `config.py` | App Config | Cấu hình ứng dụng |
| `logging_config.py` | Logging | Setup logging, format, handlers |
| `utils.py` | Utilities | Các hàm tiện ích dùng chung |
| `.env` | Environment | SECRET_KEY, CAMERA_INDEX, thresholds |
| `requirements.txt` | Dependencies | Package Python cần thiết |

---

## 7. 🛠️ Tools & Scripts (`tools/`)

| Script | Công việc | Mô tả |
|--------|-----------|-------|
| `seed_credit_classes.py` | Seed Data | Tạo dữ liệu mẫu lớp tín chỉ |
| `reset_attendance_records.py` | Reset Data | Xóa sạch bản ghi điểm danh |
| Các script khác | Utilities | Testing, migration, maintenance |

---

## 8. 📚 Documentation

| File | Nội dung |
|------|----------|
| `README.md` | Hướng dẫn sử dụng tổng quan |
| `vision_code_details.md` | Chi tiết xử lý ảnh OpenCV |
| `DARK_MODE_*.md` | Hướng dẫn Dark Mode |
| `REFACTORING_*.md` | Hướng dẫn tái cấu trúc code |
| `UI_UX_*.md` | Cải tiến giao diện |

---

## 🏗️ Kiến Trúc Hệ Thống

```
┌────────────────────────────────────────────────────────────────────┐
│                         FRONTEND LAYER                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │
│  │  Templates   │  │  Static CSS  │  │     JavaScript (SSE)     │ │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘ │
├────────────────────────────────────────────────────────────────────┤
│                         FLASK APP LAYER                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │
│  │    Routes    │  │     API      │  │     Video Stream         │ │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘ │
├────────────────────────────────────────────────────────────────────┤
│                         SERVICES LAYER                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │
│  │ FaceService  │  │  Antispoof   │  │    TrainingService       │ │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘ │
├────────────────────────────────────────────────────────────────────┤
│                          CORE LAYER                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │
│  │CameraManager │  │   Pipeline   │  │    InferenceEngine       │ │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘ │
├────────────────────────────────────────────────────────────────────┤
│                        DATABASE LAYER                              │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                    SQLite (database.py)                      │  │
│  └─────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Thống Kê Dự Án

| Metric | Giá trị |
|--------|---------|
| Tổng số file Python | ~15 files |
| Tổng số templates | 9 + 20 components |
| Lines of Code (app.py) | ~4,275 lines |
| Lines of Code (database.py) | ~1,857 lines |
| Model AI | YOLOv8, FaceNet, DeepFace, MiniFASNet |
| Database | SQLite |
| Framework | Flask |

---

## ✅ Checklist Phát Triển

### Backend
- [ ] API endpoints đầy đủ
- [ ] Session security
- [ ] Error handling
- [ ] Logging

### AI/CV
- [ ] Face detection accuracy
- [ ] Recognition threshold tuning
- [ ] Anti-spoof integration
- [ ] Performance optimization

### Frontend
- [ ] Responsive design
- [ ] Dark mode
- [ ] Real-time updates (SSE)
- [ ] Error messages

### Testing
- [ ] Unit tests
- [ ] Integration tests
- [ ] Performance tests

### Documentation
- [ ] API documentation
- [ ] User guide
- [ ] Developer guide

---

*Tài liệu được tạo: 2025-12-05*
