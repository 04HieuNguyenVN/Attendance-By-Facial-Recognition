# Attendance By Facial Recognition

README này cung cấp hướng dẫn và mô tả chi tiết cho dự án "Attendance By Facial
Recognition" — một ứng dụng web để quản lý và điểm danh bằng nhận diện khuôn
mặt. Nội dung trình bày bằng tiếng Việt, bao gồm chức năng, kiến trúc,
hướng dẫn cài đặt, cách chạy và ghi chú về chế độ AI / demo.

## Mục tiêu dự án

- Tạo một hệ thống điểm danh dễ triển khai cho trường học / lớp học.
- Cho phép thử nghiệm nhanh (demo mode) mà không cần cài đặt các thư viện AI nặng.
- Cung cấp thư mục tham khảo `face_attendance/` chứa pipeline FaceNet / MTCNN
  / anti-spoof để phát triển nâng cao.

## Tính năng chính

- Quản lý học sinh, lớp học và thông tin cơ bản.
- Upload ảnh khuôn mặt và lưu nhiều mẫu cho mỗi học sinh (`student_face_samples`).
- Điểm danh tự động từ camera (real-time) hoặc từ hình ảnh.
- Giảm false-positive bằng cơ chế so sánh embedding + progress confirmation (nhiều frame liên tiếp).
- Thông báo thời gian thực trên giao diện bằng SSE (Server-Sent Events).
- Chế độ DEMO cho phép chạy ứng dụng mà không cần face_recognition / dlib.

## Kiến trúc tổng quan

- `app.py`: ứng dụng Flask chính, route xử lý, SSE, API.
- `database.py`: helper và schema SQLite, các hàm CRUD cho `students`, `attendance`, `student_face_samples`.
- `templates/` và `static/`: giao diện người dùng (HTML/CSS/JS).
- `uploads/` và `data/`: nơi lưu ảnh upload và dữ liệu mẫu.
- `face_attendance/`: mã tham khảo cho pipeline AI (không bắt buộc để chạy demo).

## Cài đặt nhanh

1. Clone repo và chuyển vào thư mục dự án:

```powershell
git clone https://github.com/04HieuNguyenVN/Attendance-By-Facial-Recognition.git
cd "Attendance by facial recognition"
```

2. Tạo virtualenv và kích hoạt (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Cài package cơ bản:

```powershell
pip install -r requirements.txt
```

Ghi chú: nếu bạn muốn bật tính năng AI đầy đủ (FaceNet / MTCNN / anti-spoof),
chuẩn bị một môi trường riêng (virtualenv/conda) và cài các dependency nâng cao
(xem lịch sử `requirements_advanced.txt` hoặc tài liệu trong `face_attendance/`).

## Cấu hình

Tạo file `.env` từ `./.env.example` và chỉnh các giá trị cần thiết:

```powershell
Copy-Item .env.example .env
```

Thiết lập tối thiểu:

```env
SECRET_KEY=your-random-secret-key
DEMO_MODE=0
CAMERA_INDEX=0
```

DEMO_MODE=1 sẽ bật chế độ giả lập (không cần camera hoặc thư viện dlib/face_recognition).

## Chạy ứng dụng

Chạy trong PowerShell:

```powershell
# nếu dùng virtualenv
.\.venv\Scripts\python.exe app.py
# hoặc dùng start.bat
.\start.bat
```

Truy cập giao diện tại `http://127.0.0.1:5000`.

## Chế độ AI vs Demo

- Demo: phù hợp để kiểm tra giao diện, thao tác quản lý học sinh, báo cáo mà không
  cần cài `dlib` hay `face_recognition`.
- Full AI: khi muốn chạy pipeline nhận diện thực sự, cài thêm các thư viện AI
  (TensorFlow, dlib, face-recognition, PyTorch nếu dùng anti-spoof). Khuyến nghị
  dùng môi trường riêng và kiểm tra phiên bản tương thích.

## Cấu trúc thư mục (tóm tắt)

```
├── app.py
├── database.py
├── logging_config.py
├── requirements.txt
├── .env.example
├── start.bat
├── README.md
├── data/                 # ảnh mẫu đã tiền xử lý
├── uploads/              # file do người dùng upload
├── logs/
├── templates/
└── static/
```

## Chi tiết kỹ thuật đáng chú ý

- Bảng `student_face_samples` trong DB lưu nhiều mẫu ảnh/embedding cho mỗi học sinh.
- Logic nhận diện dùng embedding comparison: tính khoảng cách embedding và
  so sánh với ngưỡng `FACE_DISTANCE_THRESHOLD` + `FACE_RECOGNITION_THRESHOLD`.
- Progress confirmation: hệ thống đếm số frame liên tiếp nhận diện cùng một
  người (ví dụ `REQUIRED_FRAMES = 30`) trước khi ghi điểm danh, giảm sai dương tính.
- SSE: endpoint `/api/events/stream` phát event khi có bản ghi mới;
  frontend sử dụng `EventSource` để hiển thị thông báo và tự động làm mới danh sách.

## Sơ đồ pipeline xử lý (mô tả ngắn)

Dưới đây là sơ đồ luồng xử lý một frame (hoặc một ảnh) trong hệ thống — từ
việc lấy ảnh đến khi ghi nhận điểm danh và cập nhật giao diện:

```
Camera (OpenCV VideoCapture)
    |
    v
  Chụp khung hình (BGR)
    |
    v
  Chuyển BGR -> RGB
    |
    v
  Phát hiện khuôn mặt: MTCNN (`face_attendance/align`)  HOẶC  Cascade OpenCV
    |
    v
  Với mỗi khuôn mặt tìm được:
    - Căn chỉnh / crop vùng mặt (padding, resize)
    - Tiền xử lý (resize theo kích thước FaceNet, prewhiten / chuẩn hoá)
    - (Tuỳ chọn) Kiểm tra anti-spoof (MiniFASNet, PyTorch)
    - Trích xuất embedding (FaceNet TensorFlow `.pb`  HOẶC  `face_recognition` / dlib)
    - Nhận dạng / đối sánh: classifier (SVM trên embedding) hoặc so sánh khoảng cách
    - Cập nhật bộ đếm progress theo thời gian (REQUIRED_FRAMES)
    |
    v
  Nếu xác nhận đủ progress -> ghi điểm danh (ghi vào SQLite)
    |
    v
  Phát SSE event -> frontend hiển thị thông báo + refresh danh sách điểm danh
    |
    v
  Vẽ overlay lên khung video (bbox, tên, progress) và stream về client
```

Phân tích tóm tắt (mỗi bước liên kết tới các file):

- Capture frame: `app.py` (`ensure_video_capture`, video loop)
- Detection: `face_attendance/align/detect_face.py` (MTCNN) hoặc OpenCV cascade fallback trong `services/face_service.py`.
- Align / Crop / Preprocess: `face_attendance/facenet.py` (prewhiten) và `services/face_service.py` (`preprocess_face`).
- Anti-spoof: `face_attendance/src/anti_spoof_predict.py` + `face_attendance/src/model_lib/*`.
- Embedding: `services/face_service.py` (`get_embedding`) sử dụng FaceNet `.pb` (or `face_recognition` fallback).
- Classifier / Matching: `services/training_service.py` (train SVM), classifier load in `services/face_service.py` (`facemodel.pkl`) or simple distance compare.
- Progress / Temporal confirm: logic in `app.py` (`attendance_progress`, `REQUIRED_FRAMES`).
- SSE + UI update: `app.py` (`/api/events/stream`) và frontend `templates/index.html`, `static/js/main.js`.

Nếu bạn muốn, tôi có thể vẽ phiên bản mermaid (flowchart) để hiển thị trên GitHub nếu repo hỗ trợ mermaid rendering — hoặc tạo sơ đồ PNG/SVG và thêm vào `static/img/`.

## Công nghệ & Thư viện

Dưới đây liệt kê các thư viện/technology chính được sử dụng, chia thành 2 nhóm:

1. Nhóm xử lý ảnh / Thị giác máy tính (CV)

- `OpenCV` (`opencv-python`): xử lý ảnh/video, đọc camera, crop/resize hình, vẽ bounding box và overlay progress. Ứng dụng chính: `app.py` (video stream, preview, crop mặt), `services/training_service.py` (ghi ảnh mẫu). Cài: `pip install opencv-python`.
- `face-recognition` (dựa trên `dlib`): phát hiện khuôn mặt và trích xuất embedding nhanh (thực tế là wrapper tiện lợi). Ứng dụng: nhận diện nhẹ trong chế độ non-FaceNet. Thư viện **tùy chọn** (khó cài trên Windows). Cài (nếu cần): `pip install face-recognition` (yêu cầu `dlib`).
- `dlib`: thư viện nền tảng cho face-recognition (HOG/NN), dùng cho phát hiện face/landmark. Ứng dụng: phát hiện khuôn mặt, landmark. Cài đặc biệt trên Windows (cần Visual C++ Build Tools) hoặc dùng wheel prebuilt.
- `Pillow` (`PIL`): thao tác ảnh phụ trợ (resize/convert) khi lưu/hiển thị. Cài: `pip install Pillow`.

2. Nhóm AI / Deep Learning

- `TensorFlow` / FaceNet model: dùng để tính embedding chất lượng cao (file mẫu `face_attendance/Models/20180402-114759.pb`). Ứng dụng: `face_attendance/facenet.py` và pipeline tham khảo trong `face_attendance/`. Đây là phần **tùy chọn**. Cài: `pip install tensorflow` hoặc `pip install tensorflow-cpu`.
- `PyTorch`: dùng cho các mô hình anti-spoofing / MultiFTNet trong `face_attendance/src` (training và inference). Ứng dụng: `face_attendance/src/train_main.py` và anti-spoof inference. Cài: `pip install torch torchvision` (chọn phiên bản phù hợp với CUDA nếu cần).
- `scikit-learn` (`sklearn`): dùng để huấn luyện SVM/classifier trên embedding (TrainingService lưu/huấn luyện `facemodel.pkl`). Ứng dụng: `services/training_service.py`. Cài: `pip install scikit-learn`.
- `numpy`: xử lý mảng/embedding, bắt buộc cho hầu hết luồng numeric. Cài: `pip install numpy`.
- `tensorboardX` / `tensorboard`: ghi logs khi train (xem `face_attendance/src/train_main.py`). Cài: `pip install tensorboardX` hoặc `pip install tensorboard`.

Ghi chú cài đặt / vận hành

- Một số thư viện AI (TensorFlow, PyTorch, dlib) **nặng** và có yêu cầu nền tảng (Visual Studio build tools, CUDA). Khuyến nghị tạo môi trường ảo riêng (ví dụ `.venv-ai`) để cài đặt khi cần.
- `requirements.txt` chứa các package cơ bản để chạy ứng dụng ở chế độ demo. Các dependency nâng cao đã được lưu/archived trong lịch sử (`requirements_advanced.txt`) — khôi phục khi chuẩn bị môi trường AI.
- Để chạy pipeline huấn luyện PyTorch (`face_attendance/src/train_main.py`), chuẩn bị dataset theo cấu hình `face_attendance/src/default_config.py` (mặc định `./datasets/rgb_image`) và cài các package trong nhóm AI.

## Gỡ rối (Troubleshooting)

- Lỗi khi cài `dlib` trên Windows: cài Visual C++ Build Tools hoặc dùng wheel
  prebuilt. Hoặc chạy trong DEMO_MODE.
- Nếu camera không hoạt động: kiểm tra `CAMERA_INDEX` trong `.env` và
  đảm bảo camera không bị ứng dụng khác chiếm dụng.
- Nếu nhận diện sai nhiều: thử điều chỉnh `FACE_DISTANCE_THRESHOLD` và
  `REQUIRED_FRAMES` trong `app.py`.

## Lưu ý bảo mật & pháp lý

- Ứng dụng mang tính minh họa/giáo dục. Khi triển khai thực tế cần xem xét
  chính sách bảo mật, quyền riêng tư và quy định pháp lý (GDPR / luật
  địa phương) liên quan tới xử lý dữ liệu sinh trắc học.

## Đóng góp

Nếu bạn muốn đóng góp:

1. Fork repository
2. Tạo branch mới: `git checkout -b feature/your-feature`
3. Commit, push và tạo Pull Request

Các thay đổi lớn liên quan tới AI nên tách branch riêng và kèm hướng dẫn
triển khai môi trường (requirements, model weights, notes).

## Tác giả

- `04HieuNguyenVN` (xem repo trên GitHub)

---

Phiên bản README: cập nhật bởi trợ lý (ngày 2025-11-15). Nếu bạn muốn bổ
thêm phần tiếng Anh, hướng dẫn CI/CD, hoặc README rút gọn cho người dùng,
hãy cho tôi biết để tôi cập nhật tiếp.

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

## 🪟 Win Console Demo Script

`win_console.py` cung cấp một giao diện Tkinter đơn giản để chạy nhận diện khuôn mặt trực tiếp trên Windows mà không cần mở trang web.

1. Kích hoạt virtualenv và đảm bảo đã cài các thư viện tối thiểu: `pip install -r requirements.txt` (nếu muốn nhận diện thật cần thêm `face-recognition` + `dlib`).
2. Chuẩn bị ảnh mẫu trong thư mục `data/` theo định dạng `MSSV_HoTen.jpg` để script tự nạp.
3. Chạy script:

```powershell
.\.venv\Scripts\python.exe win_console.py
```

4. Nhấn **Start** để bật camera, script sẽ hiển thị các lần nhận diện thành công trong danh sách sự kiện.

> Lưu ý: nếu chưa cài `face_recognition`, script vẫn chạy ở chế độ demo và chỉ hiển thị dấu thời gian.

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
