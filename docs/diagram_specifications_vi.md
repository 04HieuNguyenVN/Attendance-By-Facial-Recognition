# Đặc Tả Biểu Đồ Hoạt Động (Chi Tiết & Chính Xác)

Tài liệu này cung cấp mô tả chi tiết nhất về luồng hoạt động của hệ thống để sử dụng cho việc tạo biểu đồ hoạt động (Activity Diagram).

## Lưu ý chung cho tất cả biểu đồ
*   **Loại biểu đồ:** Activity Diagram (Biểu đồ hoạt động).
*   **Phong cách:** Swimlane (Phân làn).
*   **Các làn (Swimlanes):** Thường chia làm 2 làn chính: **Người dùng (User/Actor)** và **Hệ thống (System/Backend)**. Có thể có thêm làn phụ như **Database** hoặc **Hardware** nếu cần thiết.
*   **Ký hiệu:** Sử dụng các ký hiệu chuẩn UML: Start Node, End Node, Action Node, Decision Node (Hình thoi), Fork/Join (Thanh ngang).

---

## 1. Biểu đồ: Luồng Xử lý Nhận diện Khuôn mặt (AI Inference Flow)
**Mục tiêu:** Mô tả chi tiết cách hệ thống xử lý một khung hình video để nhận diện sinh viên.

*   **Tác nhân (Actors):**
    *   **Camera:** Thiết bị phần cứng.
    *   **Inference Engine:** Hệ thống AI xử lý trung tâm.
*   **Quy trình (Steps):**
    1.  **[Camera]**: Bắt hình ảnh (Capture Frame) từ luồng video thời gian thực.
    2.  **[System-InferenceEngine]**: Nhận frame hình ảnh.
    3.  **[System-InferenceEngine]**: **Phát hiện khuôn mặt (Face Detection)**: Sử dụng thuật toán (YOLO/MTCNN) để tìm vị trí các khuôn mặt trong ảnh.
    4.  **[Decision]**: Có tìm thấy khuôn mặt không?
        *   *Không:* Quay lại bước 1 (Bắt frame tiếp theo).
        *   *Có:* Tiếp tục bước 5.
    5.  **[System-InferenceEngine]**: **Trích xuất khuôn mặt (Face Extraction)**: Cắt vùng ảnh khuôn mặt (ROI) & Căn chỉnh (Alignment) để khuôn mặt thẳng góc.
    6.  **[System-InferenceEngine]**: **Tạo Embedding (Feature Extraction)**: Đưa ảnh khuôn mặt đã căn chỉnh qua model Deep Learning (FaceNet/DeepFace) để tạo vector đặc trưng (128-D hoặc 512-D embedding).
    7.  **[System-InferenceEngine]**: **Phân loại (Classification)**: Đưa vector embedding vào bộ phân loại SVM (`svc.predict_proba`).
    8.  **[Decision]**: Kiểm tra độ tin cậy (Confidence Score)?
        *   *Nếu Confidence < Threshold (ví dụ 0.6):* Đánh dấu là "Unknown" (Người lạ).
        *   *Nếu Confidence >= Threshold:* Xác định được danh tính (Student ID).
    9.  **[System-Output]**: Trả về kết quả cuối cùng: {Mã SV, Họ Tên, Độ tin cậy, Trạng thái Check-in}.
    10. **[End]**: Kết thúc xử lý frame hiện tại, hiển thị kết quả lên màn hình.

---

## 2. Biểu đồ: Luồng Quản lý Phiên điểm danh (Session Management Flow)
**Mục tiêu:** Mô tả vòng đời của một buổi học.

*   **Tác nhân (Actors):**
    *   **Giảng viên (Lecturer):** Người quản lý lớp.
    *   **Hệ thống (System):** Xử lý logic nghiệp vụ & Database.
*   **Quy trình (Steps):**
    1.  **[Lecturer]**: Truy cập danh sách lớp tín chỉ & Chọn "Bắt đầu điểm danh" (`api_create_session`).
    2.  **[System]**: Tạo bản ghi phiên mới trong Database với trạng thái `status = 'open'`.
    3.  **[System]**: Khởi tạo thời gian bắt đầu (`opened_at`) và thời gian hết hạn (`checkin_deadline`).
    4.  **[System-Loop]**: **Vòng lặp giám sát phiên (Session Monitoring)**:
        *   Hệ thống liên tục nhận diện sinh viên (kết nối với Luồng 1).
        *   Nếu sinh viên được nhận diện & chưa điểm danh -> Tạo bản ghi `Attendance` (Check-in).
    5.  **[System-Timer]**: Kiểm tra thời gian định kỳ.
    6.  **[Decision]**: Đã hết thời gian điểm danh chưa? (Hiện tại > `checkin_deadline`?)
        *   *Chưa:* Tiếp tục cho phép điểm danh.
        *   *Rồi:* Chuyển sang bước 7.
    7.  **[System]**: **Tự động đóng phiên (Auto-close)**: Cập nhật trạng thái phiên thành `closed`.
    8.  **[Lecturer]**: (Tùy chọn) Có thể bấm nút "Kết thúc sớm" bất kỳ lúc nào -> Kích hoạt bước 7 ngay lập tức.
    9.  **[System]**: Tổng hợp danh sách: Số lượng có mặt, vắng mặt.
    10. **[End]**: Lưu báo cáo phiên vào Database.

---

## 3. Biểu đồ: Luồng Đồng bộ Dữ liệu (Data Synchronization Flow)
**Mục tiêu:** Đảm bảo dữ liệu nhận diện (RAM) khớp với dữ liệu lưu trữ (Disk/DB).

*   **Tác nhân (Actors):**
    *   **Background Service:** Tiến trình chạy ngầm của hệ thống.
    *   **Disk Storage:** Ổ cứng chứa ảnh và file model.
*   **Quy trình (Steps):**
    1.  **[Event]**: Kích hoạt khi (A) Khởi động ứng dụng hoặc (B) Có thay đổi dữ liệu (Thêm/Sửa/Xóa sinh viên).
    2.  **[System-Sync]**: Thực hiện hàm `load_all_embeddings()`.
    3.  **[System-Disk]**: Quét toàn bộ thư mục `data/` để tìm các thư mục con (Mã SV) và file ảnh.
    4.  **[System-Process]**: Tải các file ảnh `.jpg/.png` và file `.npy` (numpy array) lên bộ nhớ.
    5.  **[System-Process]**: Tạo/Cập nhật danh sách `Input Embeddings` (X) và `Labels` (y).
    6.  **[System-Training]**: (Nếu cần) Gọi `TrainingService.train_classifier()` để huấn luyện lại SVM nhanh (Incremental Learning) nếu file model cũ không khớp.
    7.  **[System-RAM]**: Cập nhật biến toàn cục `known_face_encodings` và `classifier` trong bộ nhớ RAM.
    8.  **[End]**: Hệ thống ở trạng thái "Ready" (Sẵn sàng nhận diện).

---

## 4. Biểu đồ: Quy trình Đăng ký Khuôn mặt (Face Registration Flow)
**Mục tiêu:** Thêm sinh viên mới vào hệ thống.

*   **Tác nhân (Actors):**
    *   **Admin/Teacher:** Người thực hiện đăng ký.
    *   **API:** `api_quick_register`.
*   **Quy trình (Steps):**
    1.  **[User]**: Nhập thông tin sinh viên (Mã SV, Tên, Lớp) trên Form.
    2.  **[User]**: Cung cấp ảnh khuôn mặt (2 cách):
        *   *Cách A:* Upload file ảnh từ máy tính.
        *   *Cách B:* Chụp ảnh trực tiếp từ Webcam (Web browser).
    3.  **[User]**: Gửi yêu cầu (Submit Form) đến API.
    4.  **[System-API]**: **Validate dữ liệu**:
        *   Kiểm tra Mã SV đã tồn tại chưa?
        *   Kiểm tra số lượng ảnh (Tối thiểu 3 ảnh).
    5.  **[Decision]**: Dữ liệu hợp lệ?
        *   *Không:* Trả về lỗi, yêu cầu nhập lại.
        *   *Có:* Tiếp tục bước 6.
    6.  **[System-Disk]**: Tạo thư mục mới `data/{student_id}`.
    7.  **[System-Process]**: Chạy Face Detection trên từng ảnh gửi lên.
        *   Nếu ảnh không có mặt hoặc quá mờ -> Loại bỏ.
        *   Lưu các ảnh hợp lệ vào thư mục đã tạo.
    8.  **[System-DB]**: Thêm bản ghi vào bảng `students`.
    9.  **[System-Sync]**: Kích hoạt sự kiện "Đồng bộ dữ liệu" (Gọi Luồng 3) để hệ thống AI học khuôn mặt mới ngay lập tức.
    10. **[End]**: Thông báo "Đăng ký thành công".

---

## 5. Biểu đồ: Quy trình Huấn luyện Mô hình (Model Training Flow)
**Mục tiêu:** Cập nhật trí tuệ của AI (SVM Classifier).

*   **Tác nhân (Actors):**
    *   **Admin:** Người quản trị.
    *   **Training Service:** Module huấn luyện.
*   **Quy trình (Steps):**
    1.  **[Admin]**: Truy cập trang Quản lý AI -> Bấm "Huấn luyện lại Model" (Retrain Model).
    2.  **[System]**: Khóa tạm thời tiến trình nhận diện (để tránh xung đột bộ nhớ).
    3.  **[System-Loader]**: Tải toàn bộ Embeddings từ Disk (`load_all_embeddings`).
    4.  **[System-Process]**: Sử dụng `LabelEncoder` để mã hóa nhãn (Tên sinh viên -> Số nguyên).
    5.  **[System-ML]**: Khởi tạo thuật toán SVM (`SVC(kernel='linear', probability=True)`).
    6.  **[System-ML]**: Thực hiện `fit(X, y)`: Huấn luyện mô hình với dữ liệu hiện có.
    7.  **[System-Disk]**: Lưu model đã huấn luyện xuống file `data/models/facemodel.pkl` (Sử dụng `pickle`).
    8.  **[System-Live]**: Thực hiện **Hot-swap**: Thay thế biến `classifier` trong RAM bằng model mới vừa tạo.
    9.  **[System]**: Mở lại tiến trình nhận diện.
    10. **[End]**: Hiển thị thông báo "Huấn luyện hoàn tất - Độ chính xác X%".

---

## 6. Biểu đồ: Quy trình Xác thực Người dùng (Authentication Flow)
**Mục tiêu:** Bảo mật truy cập vào dashboard quản trị.

*   **Tác nhân (Actors):**
    *   **Người dùng:** Admin hoặc Giảng viên.
    *   **Auth System:** Module xác thực.
*   **Quy trình (Steps):**
    1.  **[User]**: Truy cập đường dẫn `/login`.
    2.  **[System-UI]**: Hiển thị Form đăng nhập.
    3.  **[User]**: Nhập Username và Password -> Bấm Login.
    4.  **[System-DB]**: Truy vấn tìm User trong bảng `users` theo Username.
    5.  **[Decision]**: User có tồn tại không?
        *   *Không:* Báo lỗi "Tài khoản không tồn tại".
    6.  **[System-Security]**: Lấy `password_hash` từ Database.
    7.  **[Decision]**: Kiểm tra mật khẩu (Verify Password)?
        *   *Logic:* Sử dụng `check_password_hash(hash, input_password)`.
        *   *Sai:* Báo lỗi "Sai mật khẩu".
        *   *Đúng:* Tiếp tục bước 8.
    8.  **[Decision]**: Kiểm tra chuẩn mã hóa (Check Legacy)?
        *   *Nếu là Hash cũ (SHA256):* Tự động tạo Hash mới (PBKDF2) và cập nhật lại vào DB (Auto-upgrade security).
    9.  **[System-Session]**: Tạo Session đăng nhập (Lưu `user_id`, `role` vào Cookie).
    10. **[System]**: Chuyển hướng (Redirect) vào Trang chủ Dashboard.
    11. **[End]**: Đăng nhập thành công.
