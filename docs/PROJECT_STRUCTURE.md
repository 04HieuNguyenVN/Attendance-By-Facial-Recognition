# 📁 Cấu Trúc Dự Án Theo Layer Architecture

Đã tái cấu trúc project theo **TASK_BREAKDOWN.md** với kiến trúc phân lớp rõ ràng.

## 🏗️ Kiến Trúc Hệ Thống

```
attendance-by-facial-recognition/
├── app/                          # APPLICATION LAYER
│   ├── __init__.py              # Flask app factory
│   ├── config.py                # Centralized config
│   ├── globals.py               # Backward compatibility
│   │
│   ├── models/                  # BUSINESS LOGIC MODELS
│   │   ├── state_manager.py         # State management
│   │   ├── camera_service.py        # Camera operations
│   │   ├── attendance_tracker.py    # Attendance logic
│   │   ├── face_recognition_manager.py  # Face recognition
│   │   └── event_broadcaster.py     # SSE events
│   │
│   ├── routes/                  # FLASK ROUTES (API)
│   │   ├── main.py                  # Main web pages
│   │   ├── auth.py                  # Authentication
│   │   ├── api_attendance.py        # Attendance APIs
│   │   ├── api_students.py          # Student management
│   │   ├── api_classes.py           # Class management
│   │   ├── api_credit_classes.py    # Credit class APIs
│   │   ├── api_training.py          # Training APIs
│   │   ├── api_reports.py           # Reports & analytics
│   │   ├── api_camera.py            # Camera control
│   │   ├── api_register.py          # Registration
│   │   ├── api_events.py            # SSE events
│   │   ├── api_stats.py             # Statistics
│   │   └── api_system.py            # System APIs
│   │
│   ├── services/                # SERVICES LAYER (AI/CV)
│   │   ├── face_service.py          # FaceNet + face detection
│   │   ├── antispoof_service.py     # Liveness detection
│   │   ├── training_service.py      # SVM training
│   │   ├── deepface_db.py           # DeepFace database
│   │   └── presence_tracker.py      # Presence tracking
│   │
│   ├── database/                # DATABASE LAYER
│   │   └── database.py              # SQLite operations
│   │
│   ├── middleware/              # MIDDLEWARE
│   │   └── auth_middleware.py       # Auth checks
│   │
│   └── utils/                   # UTILITIES
│       └── helpers.py               # Helper functions
│
├── core/                        # CORE LAYER (CV/AI)
│   ├── vision/                  # Computer Vision
│   │   ├── camera_manager.py        # Camera connection
│   │   ├── pipeline.py              # Image processing
│   │   └── state.py                 # Pipeline state
│   │
│   ├── inference/               # AI Inference
│   │   ├── engine.py                # Multi-strategy engine
│   │   ├── base_strategy.py         # Strategy interface
│   │   ├── deepface_strategy.py     # DeepFace strategy
│   │   └── facenet_strategy.py      # FaceNet strategy
│   │
│   └── attendance/              # Attendance Logic
│       └── manager.py               # Attendance management
│
├── templates/                   # FRONTEND - Templates
│   ├── base.html
│   ├── index.html
│   ├── students.html
│   ├── classes.html
│   └── components/              # UI Components (20 files)
│
├── static/                      # FRONTEND - Assets
│   ├── css/
│   │   └── main.css
│   ├── js/
│   │   └── main.js
│   └── img/
│
├── tools/                       # DEVELOPMENT TOOLS
│   ├── ml/                      # ML utilities
│   └── diagnostics/             # Diagnostic scripts
│
├── data/                        # DATA DIRECTORY
│   └── models/                  # Trained models
│
├── logs/                        # LOGS
├── uploads/                     # UPLOADED FILES
│
├── run.py                       # ENTRY POINT
├── database.py                  # Legacy database (deprecated)
├── app.py                       # Legacy app (deprecated)
├── requirements.txt             # Dependencies
└── README.md                    # Documentation

app/services_legacy/             # Old services (to be removed)
```

## 📊 Layer Architecture

### 1️⃣ Frontend Layer

- **Templates**: HTML templates với Jinja2
- **Static**: CSS, JavaScript, images
- **UI Components**: 20 reusable components

### 2️⃣ Application Layer (`app/`)

- **Routes**: Flask blueprints cho API endpoints
- **Models**: Business logic (StateManager, CameraService, AttendanceTracker, etc.)
- **Middleware**: Authentication, authorization
- **Utils**: Helper functions

### 3️⃣ Services Layer (`app/services/`)

- **FaceRecognitionService**: FaceNet embeddings
- **AntiSpoofService**: Liveness detection
- **TrainingService**: SVM classifier training
- **DeepFaceDB**: DeepFace database operations
- **PresenceTracker**: Student presence tracking

### 4️⃣ Core Layer (`core/`)

- **Vision**: Camera management, image processing
- **Inference**: Multi-strategy recognition engine
- **Attendance**: Core attendance logic

### 5️⃣ Database Layer (`app/database/`)

- **database.py**: SQLite operations
- Tables: users, students, classes, credit_classes, attendance, attendance_sessions

## 🔄 Service Initialization Flow

```
run.py
  → app/__init__.py.create_app()
     → Load config (app/config.py)
     → Initialize services:
        1. FaceNet services (app/services/face_service.py)
        2. AntiSpoof service (app/services/antispoof_service.py)
        3. DeepFace module (optional)
        4. YOLOv8 model (yolov8m-face.pt)
        5. Inference Engine (core/inference/engine.py)
           - DeepFace strategy (if available)
           - FaceNet strategy
        6. CameraService (app/models/camera_service.py)
        7. AttendanceTracker (app/models/attendance_tracker.py)
        8. FaceRecognitionManager (app/models/face_recognition_manager.py)
        9. EventBroadcaster (app/models/event_broadcaster.py)
     → Load today's attendance
     → Register blueprints (app/routes/)
     → Register middleware
     → Register legacy camera routes (app.py - deprecated)
  → Start Flask server (0.0.0.0:5000)
```

## 📦 Import Paths

### ✅ New (Correct)

```python
# Services
from app.services.face_service import FaceRecognitionService
from app.services.antispoof_service import AntiSpoofService
from app.services.training_service import TrainingService
from app.services.deepface_db import DeepFaceDB
from app.services.presence_tracker import PresenceTracker

# Models
from app.models.state_manager import StateManager
from app.models.camera_service import CameraService
from app.models.attendance_tracker import AttendanceTracker
from app.models.face_recognition_manager import FaceRecognitionManager
from app.models.event_broadcaster import EventBroadcaster

# Core
from core.vision.camera_manager import CameraManager
from core.inference.engine import InferenceEngine
from core.inference.deepface_strategy import DeepFaceStrategy
from core.inference.facenet_strategy import FaceNetStrategy

# Database
from app.database.database import Database, get_db, init_db
```

### ❌ Old (Deprecated)

```python
# NO LONGER WORKS
from services.face_service import FaceRecognitionService  # ❌
from services.antispoof_service import AntiSpoofService    # ❌
```

## 🔧 Thay Đổi Chính

### ✅ Completed

1. **Di chuyển services/** → **app/services/**

   - Tất cả AI/CV services giờ trong `app/services/`
   - Updated imports trong tất cả files

2. **Tạo app/database/**

   - Database layer riêng biệt
   - `database.py` được copy vào `app/database/`

3. **Tạo app/models/**

   - 5 model classes: StateManager, CameraService, AttendanceTracker, FaceRecognitionManager, EventBroadcaster
   - Extracted ~800+ lines từ app.py

4. **Tạo app/routes/**

   - 15 blueprint files cho API endpoints
   - Phân tách routes theo chức năng

5. **Centralized Config**
   - `app/config.py` chứa tất cả constants
   - Dễ maintenance và testing

### 🗑️ Deprecated Files

- `app.py` - Legacy file (chỉ còn camera routes)
- `database.py` - Moved to `app/database/database.py`
- `app/services_legacy/` - Old services directory

## 📈 Metrics

| Metric            | Before      | After                    |
| ----------------- | ----------- | ------------------------ |
| app.py lines      | 2658        | 2450 (legacy only)       |
| Services location | `services/` | `app/services/`          |
| Database location | Root        | `app/database/`          |
| Models            | Inline      | `app/models/` (5 files)  |
| Routes            | Mixed       | `app/routes/` (15 files) |
| Config            | Scattered   | `app/config.py`          |

## ✨ Benefits

1. **Rõ ràng hơn**: Mỗi layer có trách nhiệm cụ thể
2. **Dễ maintain**: Code được tổ chức theo chức năng
3. **Dễ test**: Mỗi layer có thể test độc lập
4. **Scalable**: Dễ dàng thêm features mới
5. **Professional**: Follow best practices

## 🚀 Next Steps

1. ✅ Test app với structure mới
2. ⏳ Migrate camera routes từ app.py → `app/routes/api_camera.py`
3. ⏳ Remove `app.py` hoàn toàn
4. ⏳ Remove `app/services_legacy/` directory
5. ⏳ Update documentation

---

**Status**: ✅ App chạy thành công với architecture mới
**Last Updated**: 2025-12-05
**Test Command**: `python run.py`
