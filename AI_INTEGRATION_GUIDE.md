<!-- REMOVED: This file was archived/removed to avoid conflicts. -->

The original `AI_INTEGRATION_GUIDE.md` has been archived. If you need the detailed integration guide again, it is preserved in the repository history or can be restored from backups. For safety we removed active content to avoid stale instructions conflicting with the running code.

---

## 🎯 Kiến trúc hệ thống

```
┌─────────────────┐
│   Camera Feed   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│  Anti-Spoof     │────▶│  FaceNet Service │
│  Detection      │     │  (128-dim embed) │
└─────────────────┘     └────────┬─────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ SVM Classifier  │
                        │ (Trained Model) │
                        └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │   Database      │
                        │  (Attendance)   │
                        └─────────────────┘
```

---

## 📦 Cài đặt

### 1. Cài đặt Dependencies

```bash
# Cài đặt advanced dependencies
pip install -r requirements_advanced.txt

# Hoặc cài riêng từng phần
pip install tensorflow==2.13.0 torch==2.0.1 scikit-learn==1.3.0
```

### 2. Download Models

**FaceNet Model (20180402-114759.pb)**

- Đã có trong `face_attendance/Models/`
- Size: ~90MB
- [Download backup](https://drive.google.com/uc?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-)

**Anti-Spoof Models (MiniFASNet)**

- Đã có trong `face_attendance/resources/anti_spoof_models/`
- 2 models: 80x80 MiniFASNetV2, MiniFASNetV1SE
- [Download backup](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)

**MTCNN Models**

- Đã có trong `face_attendance/align/`
- det1.npy, det2.npy, det3.npy

### 3. Cấu hình Environment

Thêm vào `.env`:

```bash
# Bật FaceNet mode (mặc định: on)
USE_FACENET=1

# FaceNet settings
FACENET_THRESHOLD=0.85  # Ngưỡng nhận diện (0.0-1.0)

# Anti-spoofing settings
ANTISPOOF_DEVICE=cpu  # hoặc 'cuda' nếu có GPU
ANTISPOOF_THRESHOLD=0.5  # Ngưỡng phát hiện giả mạo

# Demo mode (tắt AI, dùng simulation)
DEMO_MODE=0
```

---

## 🚀 Sử dụng

### A. Đăng ký sinh viên mới (Training)

#### Bước 1: Thu thập ảnh khuôn mặt

**Cách 1: Upload ảnh**

```python
# Upload ảnh vào data/<student_id>_<student_name>.jpg
# Ví dụ: data/SV0001_Nguyen Van A.jpg
```

**Cách 2: Capture từ webcam** (khuyến nghị)

```javascript
// Frontend: Chụp 20 ảnh với góc độ khác nhau
fetch("/api/quick-register", {
  method: "POST",
  body: formData, // Gồm student_id, full_name, image_data (base64)
});
```

#### Bước 2: Training Classifier

**API Training:**

```bash
curl -X POST http://localhost:5000/api/train/start
```

**Response:**

```json
{
  "success": true,
  "message": "Training completed successfully",
  "stats": {
    "total_samples": 150,
    "num_students": 10,
    "students": [
      {
        "student_id": "SV0001",
        "sample_count": 15,
        "ready": true
      }
    ]
  }
}
```

#### Bước 3: Kiểm tra Training Status

```bash
curl http://localhost:5000/api/train/status
```

**Lưu ý:**

- Cần tối thiểu **10 ảnh** cho mỗi sinh viên
- Khuyến nghị: **15-20 ảnh** với góc độ đa dạng
- Training tự động save model vào `data/models/facemodel.pkl`

---

### B. Anti-Spoofing Check

#### API Endpoint

```bash
POST /api/antispoof/check
Content-Type: multipart/form-data

# Gửi ảnh dạng base64 hoặc file
{
    "image_data": "data:image/jpeg;base64,/9j/4AAQ..."
}
```

**Response:**

```json
{
  "success": true,
  "is_real": true,
  "confidence": 0.92,
  "message": "Real face",
  "bbox": [120, 80, 200, 250]
}
```

#### Tích hợp vào workflow

```python
# Trong generate_frames() hoặc xử lý attendance
if USE_FACENET and antispoof_service:
    spoof_result = antispoof_service.check_frame(frame)

    if not spoof_result['is_real']:
        # Từ chối: Phát hiện giả mạo!
        cv2.putText(frame, "SPOOF DETECTED!", ...)
        continue

    # OK: Tiếp tục nhận diện
    face_results = face_service.process_frame(frame)
```

---

### C. Live Attendance Recognition

#### Flow hoàn chỉnh

```python
# 1. Detect face và anti-spoof
spoof_check = antispoof_service.check_frame(frame)
if not spoof_check['is_real']:
    return  # Bỏ qua frame này

# 2. Nhận diện khuôn mặt
face_results = face_service.process_frame(frame)

for face_data in face_results:
    student_id = face_data['student_id']
    confidence = face_data['confidence']
    bbox = face_data['bbox']

    # 3. Progress tracking (30 frames liên tiếp)
    if student_id != "UNKNOWN":
        progress_count[student_id] += 1

        if progress_count[student_id] >= 30:
            # 4. Mark attendance
            mark_attendance(student_id, confidence)
            progress_count[student_id] = 0
```

---

## 🔄 So sánh: FaceNet vs face_recognition

| Feature            | face_recognition (dlib) | FaceNet (TensorFlow) |
| ------------------ | ----------------------- | -------------------- |
| **Độ chính xác**   | 99.38% (LFW)            | 99.63% (LFW)         |
| **Embedding size** | 128-dim                 | 128-dim              |
| **Model size**     | ~100 MB                 | ~90 MB               |
| **Tốc độ (CPU)**   | ~0.3s/face              | ~0.5s/face           |
| **Tốc độ (GPU)**   | N/A                     | ~0.05s/face          |
| **Anti-spoof**     | ❌ Không có             | ✅ Có (MiniFASNet)   |
| **Training mới**   | ⚠️ Khó                  | ✅ Dễ (SVM)          |
| **Dependencies**   | dlib (khó cài)          | TensorFlow           |

**Khi nào dùng FaceNet?**

- Cần độ chính xác cao
- Có GPU (tăng tốc 10x)
- Cần anti-spoofing
- Thường xuyên đăng ký sinh viên mới

**Khi nào dùng face_recognition?**

- Môi trường đơn giản
- Không có GPU
- Không cần anti-spoof
- Ít thay đổi database

---

## 🛠️ Advanced Configuration

### 1. GPU Acceleration

```bash
# Cài TensorFlow GPU
pip uninstall tensorflow
pip install tensorflow-gpu==2.13.0

# Kiểm tra GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Cập nhật .env
ANTISPOOF_DEVICE=cuda
```

### 2. Fine-tuning Thresholds

**FaceNet Recognition Threshold:**

```python
# Thấp (0.7-0.8): Dễ nhận diện nhưng nhiễu
# Trung bình (0.8-0.85): Cân bằng (khuyến nghị)
# Cao (0.9+): Chính xác nhưng khó nhận diện
FACENET_THRESHOLD=0.85
```

**Anti-Spoof Threshold:**

```python
# Thấp (0.3-0.4): Ít false positives (khuyến nghị production)
# Trung bình (0.5): Cân bằng
# Cao (0.6+): Strict, nhiều false alarms
ANTISPOOF_THRESHOLD=0.5
```

### 3. Training Tips

**Thu thập ảnh tốt:**

- ✅ Góc độ đa dạng: chính diện, nghiêng 15°, 30°
- ✅ Ánh sáng khác nhau: sáng, tối, backlight
- ✅ Biểu cảm: mặt thường, cười, nghiêm túc
- ❌ Tránh: mờ, che mặt, quá xa/gần

**Tăng độ chính xác:**

```python
# 1. Tăng số lượng samples
min_samples_per_person = 20  # Thay vì 10

# 2. Data augmentation (flip, rotate, brightness)
# 3. Retrain định kỳ khi có sinh viên mới

# 4. Sử dụng kernel RBF thay vì linear
training_service.train_classifier(kernel='rbf')
```

---

## 📊 Monitoring & Debugging

### 1. Kiểm tra Model Status

```bash
# Training stats
curl http://localhost:5000/api/train/status

# System status
curl http://localhost:5000/status
```

### 2. Logs

```bash
# Main logs
tail -f logs/attendance_system.log

# Errors
tail -f logs/errors.log

# Tìm lỗi FaceNet
grep "FaceNet" logs/attendance_system.log
```

### 3. Common Issues

**Issue: "FaceNet service not available"**

```bash
# Kiểm tra TensorFlow
python -c "import tensorflow as tf; print(tf.__version__)"

# Kiểm tra model file
ls -lh face_attendance/Models/20180402-114759.pb
```

**Issue: "Training failed - insufficient data"**

```bash
# Kiểm tra số lượng ảnh
ls -l data/*.jpg | wc -l
ls -l data/training_samples/*/

# Cần ít nhất 10 ảnh/sinh viên
```

**Issue: Anti-spoof luôn báo "Spoof detected"**

```bash
# Giảm threshold
ANTISPOOF_THRESHOLD=0.3

# Hoặc tắt tạm
# Comment out anti-spoof check trong code
```

---

## 🔐 Security Best Practices

1. **Anti-Spoofing là bắt buộc trong production**

   - Ngăn chặn tấn công bằng ảnh in
   - Phát hiện video replay

2. **Lưu embeddings, không lưu ảnh gốc**

   ```python
   # Sau khi training, có thể xóa ảnh gốc
   # Chỉ giữ facemodel.pkl
   ```

3. **Rate limiting cho API training**

   ```python
   # Giới hạn số lần training/ngày
   # Tránh DoS attacks
   ```

4. **Encrypt trained model**
   ```python
   # Mã hóa facemodel.pkl trong production
   ```

---

## 📚 API Reference

### Training Service

```python
from services.training_service import TrainingService

# Initialize
training_service = TrainingService(face_service)

# Train classifier
success = training_service.train_classifier(kernel='linear')

# Get stats
stats = training_service.get_training_stats()

# Remove student
training_service.remove_student('SV0001')
```

### Face Service

```python
from services.face_service import FaceRecognitionService

# Initialize
face_service = FaceRecognitionService()
face_service.load_model()

# Process frame
results = face_service.process_frame(frame)
# Returns: [{'bbox': (x,y,w,h), 'student_id': 'SV001', 'confidence': 0.95, ...}]

# Close
face_service.close()
```

### Anti-Spoof Service

```python
from services.antispoof_service import AntiSpoofService

# Initialize
antispoof = AntiSpoofService(device='cpu')

# Check frame
result = antispoof.check_frame(frame)
# Returns: {'is_real': True, 'confidence': 0.92, 'bbox': (...), 'message': '...'}
```

---

## 🎓 Training & Fine-tuning

### Retrain khi có sinh viên mới

```bash
# 1. Thêm ảnh sinh viên vào data/
cp student_photos/* data/

# 2. Retrain
curl -X POST http://localhost:5000/api/train/start

# 3. Reload app (hoặc reload classifier trong runtime)
curl -X POST http://localhost:5000/update_faces
```

### Transfer Learning (Advanced)

Nếu muốn fine-tune FaceNet model:

```python
# Không khuyến nghị cho use case này
# FaceNet đã được pretrain trên 200M+ ảnh
# Chỉ cần train SVM classifier là đủ
```

---

## 📞 Hỗ trợ

- GitHub Issues: [Attendance-By-Facial-Recognition/issues](https://github.com/04HieuNguyenVN/Attendance-By-Facial-Recognition/issues)
- Email: support@example.com
- Documentation: Xem README.md và code comments

---

## 📝 Changelog

### Version 2.0 (November 2025)

- ✅ Thêm FaceNet-based recognition
- ✅ Thêm anti-spoofing detection
- ✅ Training service cho sinh viên mới
- ✅ API endpoints cho AI features
- ✅ Fallback to legacy face_recognition
- ✅ GPU acceleration support

---

**Happy Coding! 🚀**
