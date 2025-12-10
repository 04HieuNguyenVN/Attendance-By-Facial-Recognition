# TỔNG HỢP BIỂU ĐỒ LỚP (CLASS DIAGRAMS)

Tài liệu này bao gồm:
1.  **Biểu đồ Lớp Hệ thống (System Class Diagram):** Cái nhìn tổng quan về cấu trúc tĩnh của toàn bộ hệ thống.
2.  **Biểu đồ Lớp theo Ca sử dụng (Use Case Class Diagrams):** Chi tiết các lớp tham gia (VOPC) cho 6 luồng nghiệp vụ chính.

---

# PHẦN 1: Biểu đồ Lớp Hệ thống (System Class Diagram)

Biểu đồ này mô hình hóa cấu trúc tĩnh của hệ thống, bao gồm các lớp xử lý chính, thực thể dữ liệu (Entities) và mối quan hệ giữa chúng.

> [!NOTE]
> Biểu đồ tập trung vào các lớp cốt lõi tham gia vào các ca sử dụng chính: Nhận diện, Điểm danh, Đồng bộ và Quản lý.

```mermaid
classDiagram
    %% --- Core Logic ---
    class InferenceEngine {
        -strategies: List~RecognitionStrategy~
        -demo_mode: bool
        +add_strategy(strategy)
        +identify(face_image) InferenceResult
        +warmup()
        +reload()
        +known_subjects()
    }

    class RecognitionStrategy {
        <<Interface>>
        +warmup()
        +identify(face_image)
        +known_subjects()
        +is_ready()
    }

    class DeepFaceStrategy {
        -model_name: str
        -threshold: float
        +identify(face_image)
    }

    class CameraManager {
        -config: CameraConfig
        -state: CameraState
        +start() VideoCapture
        +read() Frame
        +stop()
        +set_enabled(bool)
    }

    class AttendanceStateManager {
        -db: DatabaseManager
        -checked_in: Set~str~
        -presence: Dict
        -progress: Dict
        +mark_check_in(student_id, name)
        +mark_check_out(student_id)
        +track_confirmation(student_id)
        +prune_presence()
        +load_today_records()
    }

    class TrainingService {
        -face_service: FaceService
        -data_dir: Path
        +collect_samples(student_id)
        +train_classifier()
        +load_all_embeddings()
        +add_student_samples()
    }

    %% --- Data Access & Entities ---
    class DatabaseManager {
        -db_path: str
        +get_connection()
        +add_student()
        +get_student()
        +mark_attendance()
        +create_credit_class()
        +get_active_session_for_class()
    }

    class Student {
        +id: int
        +student_id: str
        +full_name: str
        +face_encoding: Blob
        +class_id: int
        +user_id: int
    }

    class User {
        +id: int
        +username: str
        +password_hash: str
        +role: str
    }

    class CreditClass {
        +id: int
        +credit_code: str
        +subject_name: str
        +teacher_id: int
        +status: str
    }

    class AttendanceSession {
        +id: int
        +credit_class_id: int
        +opened_at: timestamp
        +checkin_deadline: timestamp
        +status: str
    }

    class AttendanceRecord {
        +id: int
        +student_id: str
        +check_in_time: timestamp
        +status: str
        +confidence_score: float
    }

    %% --- Relationships ---
    InferenceEngine *-- RecognitionStrategy
    RecognitionStrategy <|-- DeepFaceStrategy

    AttendanceStateManager --> DatabaseManager : uses
    TrainingService --> DatabaseManager : reads/writes
    
    DatabaseManager ..> Student : manages
    DatabaseManager ..> AttendanceRecord : manages
    DatabaseManager ..> AttendanceSession : manages

    Student "1" -- "0..1" User : linked to
    CreditClass "1" -- "*" Student : contains
    CreditClass "1" -- "*" AttendanceSession : has
    AttendanceSession "1" -- "*" AttendanceRecord : records
    Student "1" -- "*" AttendanceRecord : has

    %% Controller / Application linkages
    class Application {
        +inference_engine: InferenceEngine
        +camera_manager: CameraManager
        +state_manager: AttendanceStateManager
    }

    Application --> InferenceEngine
    Application --> CameraManager
    Application --> AttendanceStateManager
```

## Giải thích chi tiết

### Core Logic Layers
*   **InferenceEngine**: Đóng vai trò là "bộ não" AI, sử dụng mẫu thiết kế Strategy để hỗ trợ nhiều thuật toán nhận diện (DeepFace, FaceNet).
*   **CameraManager**: Trừu tượng hóa việc điều khiển phần cứng camera, xử lý warmup và buffer.
*   **AttendanceStateManager**: Quản lý logic nghiệp vụ điểm danh thời gian thực, bao gồm việc theo dõi "sự hiện diện" (presence) để tránh spam check-in và xử lý time-out.

### Service Layer
*   **TrainingService**: Chịu trách nhiệm về quy trình Machine Learning (thu thập ảnh, trích xuất đặc trưng, huấn luyện SVM).

### Data Entities (Mô hình dữ liệu)
*   **Student**: Đại diện cho sinh viên, chứa thông tin định danh và vector khuôn mặt (face_encoding).
*   **CreditClass & AttendanceSession**: Quản lý cấu trúc lớp học tín chỉ và từng buổi học cụ thể.
*   **AttendanceRecord**: Lưu lịch sử điểm danh của từng sinh viên trong mỗi phiên.

---

# PHẦN 2: Biểu đồ Lớp theo Ca sử dụng (Use Case Class Diagrams)

Phần này cung cấp các biểu đồ lớp chi tiết (View of Participating Classes - VOPC) cho 6 ca sử dụng chính, chỉ hiển thị các lớp tham gia trực tiếp.

## 1. Ca sử dụng: Nhận diện Khuôn mặt (AI Inference)
Các lớp tham gia vào quá trình xử lý luồng video và nhận diện.

```mermaid
classDiagram
    direction TB
    class CameraManager {
        +read() Frame
    }
    class InferenceEngine {
        +identify(face_image) Result
    }
    class DeepFaceStrategy {
        +identify(face_image)
    }
    class InferenceResult {
        +student_id
        +confidence
    }
    
    CameraManager --> InferenceEngine : cung cấp frame
    InferenceEngine --> DeepFaceStrategy : ủy quyền xử lý
    DeepFaceStrategy ..> InferenceResult : trả về
```

## 2. Ca sử dụng: Quản lý Phiên Điểm danh (Session Management)
Các lớp tham gia vào việc tạo và quản lý vòng đời phiên học.

```mermaid
classDiagram
    direction TB
    class AttendanceStateManager {
        +mark_check_in()
        +mark_check_out()
        +prune_presence()
    }
    class DatabaseManager {
        +get_active_session_for_class()
        +mark_attendance()
        +expire_attendance_sessions()
    }
    class AttendanceSession {
        +id
        +status
        +checkin_deadline
    }
    class AttendanceRecord {
        +student_id
        +check_in_time
        +status
    }

    AttendanceStateManager --> DatabaseManager : gọi hàm DB
    DatabaseManager ..> AttendanceSession : đọc/ghi
    DatabaseManager ..> AttendanceRecord : tạo mới
```

## 3. Ca sử dụng: Đồng bộ Dữ liệu (Data Synchronization)
Các lớp tham gia vào quá trình nạp dữ liệu từ Disk lên RAM.

```mermaid
classDiagram
    direction TB
    class TrainingService {
        +load_all_embeddings()
        +train_classifier()
    }
    class InferenceEngine {
        +reload()
        +warmup()
    }
    class DatabaseManager {
        +get_all_student_embeddings()
    }
    class DiskStorage {
        <<System>>
        +scan_data_dir()
        +load_pkl_model()
    }

    TrainingService --> DatabaseManager : lấy dữ liệu
    TrainingService --> DiskStorage : quét file ảnh
    InferenceEngine --> TrainingService : sử dụng để warmup
```

## 4. Ca sử dụng: Đăng ký Khuôn mặt (Face Registration)
Các lớp xử lý việc thêm mới sinh viên và mẫu khuôn mặt.

```mermaid
classDiagram
    direction TB
    class RegisterAPI {
        <<Controller>>
        +api_quick_register()
    }
    class DatabaseManager {
        +add_student()
        +add_face_sample()
    }
    class Student {
        +student_id
        +full_name
        +face_encoding
    }
    class FaceSample {
        +image_path
        +embedding
        +is_primary
    }

    RegisterAPI --> DatabaseManager : lưu thông tin
    DatabaseManager ..> Student : tạo
    DatabaseManager ..> FaceSample : tạo
```

## 5. Ca sử dụng: Huấn luyện Mô hình (Model Training)
Các lớp tham gia vào việc huấn luyện lại bộ phân loại SVM.

```mermaid
classDiagram
    direction TB
    class TrainingService {
        +train_classifier()
        +collect_samples()
    }
    class FaceService {
        +process_frame()
        +get_embedding()
    }
    class SVMClassifier {
        <<Scikit-Learn>>
        +fit(X, y)
        +predict_proba(X)
    }

    TrainingService --> FaceService : trích xuất đặc trưng
    TrainingService --> SVMClassifier : huấn luyện & lưu
```

## 6. Ca sử dụng: Xác thực Người dùng (Authentication)
Các lớp tham gia vào quá trình đăng nhập và bảo mật.

```mermaid
classDiagram
    direction TB
    class AuthController {
        <<Controller>>
        +login()
        +logout()
        +load_user()
    }
    class DatabaseManager {
        +get_user_by_username()
        +update_user_password()
    }
    class User {
        +username
        +password_hash
        +role
    }
    class SecurityUtils {
        <<Helper>>
        +check_password_hash()
        +generate_password_hash()
    }

    AuthController --> DatabaseManager : tìm user
    AuthController --> SecurityUtils : kiểm tra pass
    DatabaseManager ..> User : trả về
```
