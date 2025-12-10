# Chi Tiết Mã Nguồn Xử Lý Ảnh & Thị Giác Máy Tính

Tài liệu này tổng hợp các đoạn code quan trọng liên quan đến xử lý ảnh, thị giác máy tính và sử dụng thư viện OpenCV trong dự án.

## 1. Quản Lý Camera & Thu Nhận Hình Ảnh

### File: [core/vision/camera_manager.py](file:///g:/Python/Attendance%20by%20facial%20recognition/core/vision/camera_manager.py)

Đoạn code này chịu trách nhiệm mở kết nối với camera và cấu hình các thông số kỹ thuật.

```python
class CameraManager:
    # ... (các phần khác của class)

    def _configure_capture(self, capture: cv2.VideoCapture) -> None:
        """
        Cấu hình các thông số cho camera sử dụng OpenCV.
        """
        if capture is None:
            return
        try:
            # Thiết lập chiều rộng khung hình (Width)
            if self.config.width:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            
            # Thiết lập chiều cao khung hình (Height)
            if self.config.height:
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            
            # Thiết lập kích thước bộ đệm (Buffer Size) để giảm độ trễ video
            if self.config.buffer_size is not None and hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                capture.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)

            # Đọc lại thông số thực tế từ camera để log
            actual_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = capture.get(cv2.CAP_PROP_FPS)
            
            # ... (Log thông tin)

            # "Làm nóng" (Warmup) camera: Đọc bỏ một số frame đầu để camera ổn định ánh sáng
            warmup = max(0, self.config.warmup_frames)
            if warmup:
                for _ in range(warmup):
                    ret, _frame = capture.read() # Đọc frame
                    time.sleep(0.05)
        except Exception as exc:
            logger.warning("Không thể cấu hình camera: %s", exc)
```

## 2. Tiền Xử Lý Ảnh (Preprocessing)

### File: [core/vision/pipeline.py](file:///g:/Python/Attendance%20by%20facial%20recognition/core/vision/pipeline.py)

Xử lý frame thô từ camera trước khi đưa vào các thuật toán AI.

```python
    def _compute_quality(self, frame: np.ndarray) -> float:
        """
        Đánh giá chất lượng ảnh dựa trên độ sắc nét (độ mờ).
        Sử dụng toán tử Laplacian của OpenCV.
        """
        try:
            # Chuyển ảnh sang ảnh xám (Grayscale) vì Laplacian hoạt động trên 1 kênh màu
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Tính phương sai của Laplacian. Giá trị càng cao -> ảnh càng sắc nét.
            # Giá trị thấp -> ảnh bị mờ (blur).
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Chuẩn hóa về thang điểm 0.0 - 1.0
            normalized = max(0.0, min(laplacian_var / 1500.0, 1.0))
            return float(normalized)
        except Exception:
            return 0.0

    def next_frame(self) -> VisionFrame:
        """
        Lấy frame tiếp theo và chuyển đổi không gian màu.
        """
        frame = self.camera.read() # Đọc ảnh BGR từ OpenCV
        
        # Chuyển đổi từ BGR (mặc định của OpenCV) sang RGB (chuẩn của các model AI)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        quality = self._compute_quality(frame)
        
        # ... (Trả về đối tượng VisionFrame)
```

### File: [services/face_service.py](file:///g:/Python/Attendance%20by%20facial%20recognition/services/face_service.py)

Chuẩn bị ảnh khuôn mặt cho mô hình FaceNet.

```python
    def preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """
        Tiền xử lý khuôn mặt: Resize và chuẩn hóa (Whitening).
        """
        # Resize ảnh về kích thước đầu vào của FaceNet (thường là 160x160)
        # Sử dụng nội suy INTER_CUBIC cho chất lượng tốt hơn
        resized = cv2.resize(face_image, (self.image_size, self.image_size),
                           interpolation=cv2.INTER_CUBIC)
        
        # Làm trắng (Whitening): Chuẩn hóa để mean=0 và std=1
        # Giúp model hoạt động tốt hơn với các điều kiện ánh sáng khác nhau
        mean = np.mean(resized)
        std = np.std(resized)
        std_adj = np.maximum(std, 1.0 / np.sqrt(resized.size))
        whitened = (resized - mean) / std_adj
        
        return whitened
```

## 3. Phát Hiện Khuôn Mặt (Face Detection)

### File: [camera.py](file:///g:/Python/Attendance%20by%20facial%20recognition/camera.py) (Sử dụng YOLOv8)

Đây là phương pháp phát hiện chính, nhanh và chính xác.

```python
        # ... (Trong vòng lặp generate_frames)
        
        # Resize ảnh để tăng tốc độ suy luận YOLO nếu ảnh quá lớn
        if YOLO_INFERENCE_WIDTH > 0 and frame_w > YOLO_INFERENCE_WIDTH:
            detection_width = YOLO_INFERENCE_WIDTH
            detection_height = int(frame_h * (detection_width / frame_w))
            
            # Resize ảnh
            detection_frame = cv2.resize(
                frame,
                (detection_width, detection_height),
                interpolation=cv2.INTER_LINEAR
            )
            # Tính tỷ lệ để scale lại tọa độ bbox sau này
            scale_x = frame_w / detection_width
            scale_y = frame_h / detection_height

        # Chạy model YOLO
        results = yolo_face_model(detection_frame, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy()

        for box in boxes:
            # Lấy tọa độ và scale về kích thước gốc
            xmin, ymin, xmax, ymax = map(int, box)
            xmin = int(xmin * scale_x)
            # ... (Xử lý tọa độ)
            
            # Cắt vùng ảnh khuôn mặt từ frame gốc
            face_img = frame[ymin:ymax, xmin:xmax]
```

### File: [services/antispoof_service.py](file:///g:/Python/Attendance%20by%20facial%20recognition/services/antispoof_service.py) (Sử dụng RetinaFace/OpenCV DNN)

Sử dụng module DNN của OpenCV để chạy mô hình Caffe (RetinaFace).

```python
    def _init_detector(self):
        """Khởi tạo bộ phát hiện RetinaFace từ file model Caffe."""
        # ...
        # Load model từ file prototxt và caffemodel
        self.face_detector = cv2.dnn.readNetFromCaffe(deploy, caffemodel)

    def get_bbox(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Phát hiện khuôn mặt dùng OpenCV DNN."""
        # ...
        # Tạo blob từ ảnh để đưa vào mạng neural
        # Mean subtraction: (104, 117, 123)
        blob = cv2.dnn.blobFromImage(resized, 1.0, mean=(104, 117, 123))
        
        self.face_detector.setInput(blob, 'data')
        # Chạy forward pass để lấy kết quả
        detections = self.face_detector.forward('detection_out').squeeze()
        
        # ... (Xử lý kết quả detections để lấy tọa độ)
```

## 4. Nhận Diện Khuôn Mặt (Face Recognition)

### File: [services/face_service.py](file:///g:/Python/Attendance%20by%20facial%20recognition/services/face_service.py) (FaceNet)

```python
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Phát hiện khuôn mặt dùng Haar Cascade (Fallback khi không dùng MTCNN/YOLO).
        """
        # Chuyển sang ảnh xám
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Load bộ phân loại Haar Cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Phát hiện khuôn mặt đa tỉ lệ
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        return [(x, y, w, h) for (x, y, w, h) in faces]
```

## 5. Ước Tính Tư Thế Đầu (Head Pose Estimation)

### File: [face_recognition.py](file:///g:/Python/Attendance%20by%20facial%20recognition/face_recognition.py)

Tính toán góc quay của đầu trong không gian 3D.

```python
    def estimate_head_pose(landmarks, frame_size):
        """
        Ước tính góc Yaw, Pitch, Roll sử dụng thuật toán PnP.
        """
        # ... (Định nghĩa model_points 3D và image_points 2D)

        # Giải bài toán Perspective-n-Point (PnP)
        # Tìm vector quay (rotation) và tịnh tiến (translation) của vật thể 3D
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        # Chuyển đổi vector quay thành ma trận quay
        rmat, _ = cv2.Rodrigues(rotation_vector)
        
        # ... (Tính toán góc Euler từ ma trận quay)
```

## 6. Hậu Xử Lý & Hiển Thị (Post-processing)

### File: [camera.py](file:///g:/Python/Attendance%20by%20facial%20recognition/camera.py)

Vẽ các thông tin lên frame hình ảnh trước khi hiển thị cho người dùng.

```python
        # ... (Trong vòng lặp generate_frames)

        # Lật ảnh theo trục dọc (tạo hiệu ứng gương)
        frame = cv2.flip(frame, 1)

        # ... (Logic xử lý)

        # Vẽ hình chữ nhật quanh khuôn mặt
        # color: màu sắc (B, G, R), thickness: độ dày nét
        cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)

        # Tính kích thước văn bản để vẽ nền
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
        
        # Vẽ hình chữ nhật nền cho chữ (để chữ dễ đọc hơn)
        cv2.rectangle(frame,
                     (label_x - padding, label_y - label_size[1] - padding),
                     (label_x + label_size[0] + padding, label_y + padding),
                     color, -1) # -1 nghĩa là tô kín (fill)

        # Viết tên và trạng thái lên ảnh
        cv2.putText(frame, label, (label_x, label_y),
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        # ...

        # Mã hóa ảnh thành định dạng JPEG để stream qua mạng
        # Giảm chất lượng xuống 75% để tối ưu băng thông
        ret2, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
```
