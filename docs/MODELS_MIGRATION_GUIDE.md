# Models Migration Guide

## 📋 Tổng quan

Đã tách ~800+ dòng business logic từ `app.py` thành 5 model classes trong `app/models/`:

### ✅ Models đã tạo:

1. **StateManager** - Quản lý global state
2. **CameraService** - Quản lý camera và vision pipeline
3. **AttendanceTracker** - Business logic điểm danh
4. **FaceRecognitionManager** - Nhận diện khuôn mặt
5. **EventBroadcaster** - Server-Sent Events (SSE)

---

## 🔄 Migration Strategy

### Backward Compatibility

File `app/globals.py` đã được refactor để:

- ✅ Code cũ vẫn hoạt động (proxied qua state_manager)
- ✅ Code mới có thể dùng models trực tiếp
- ✅ Không cần rewrite tất cả ngay lập tức

### Architecture

```
run.py
  └── app/__init__.py (create_app)
      ├── app/globals.py (backward compat layer)
      ├── app/models/ (NEW - business logic)
      │   ├── state_manager.py
      │   ├── camera_service.py
      │   ├── attendance_tracker.py
      │   ├── face_recognition_manager.py
      │   └── event_broadcaster.py
      ├── app/routes/ (blueprints)
      └── app/services/ (existing services)
```

---

## 📝 Cách sử dụng Models

### 1. StateManager

**OLD WAY (vẫn hoạt động):**

```python
from app import globals as app_globals

# Check if student checked in
if student_id in app_globals.today_checked_in:
    print("Already checked in")

# Update last recognized time
with app_globals.last_recognized_lock:
    app_globals.last_recognized[student_id] = datetime.now()
```

**NEW WAY (recommended):**

```python
from app.models import state_manager

# Check if student checked in
if state_manager.is_checked_in(student_id):
    print("Already checked in")

# Update last recognized time (thread-safe)
state_manager.update_last_recognized(student_id)
```

**Benefits:**

- Thread-safe methods
- Cleaner API
- Testable
- Type hints

---

### 2. CameraService

**OLD WAY:**

```python
from app import globals as app_globals
from core.vision.state import VisionPipelineState

# Toggle camera
app_globals.camera_enabled = not app_globals.camera_enabled
if app_globals.camera_enabled:
    # Complex initialization logic...
    pass
```

**NEW WAY:**

```python
from app import globals as app_globals

# Toggle camera (encapsulated logic)
enabled = app_globals.camera_service.toggle_camera()

# Check status
status = app_globals.camera_service.get_status()
print(f"Camera: {status['enabled']}, Ready: {status['ready']}")
```

**Benefits:**

- Encapsulated camera management
- Consistent state handling
- Easy to mock for testing

---

### 3. AttendanceTracker

**OLD WAY:**

```python
from database import db
from app import globals as app_globals

# Complex validation logic scattered everywhere
if student_id in app_globals.today_checked_in:
    if student_id not in app_globals.today_checked_out:
        return False

# Manual cooldown check
last_time = app_globals.last_recognized.get(student_id)
if last_time and (datetime.now() - last_time).total_seconds() < 30:
    return False

# Save to database
db.mark_attendance(student_id, name, ...)
```

**NEW WAY:**

```python
from app import globals as app_globals

# All validation + business logic in one place
success = app_globals.attendance_tracker.mark_attendance(
    student_id=student_id,
    student_name=name,
    confidence_score=0.95,
    expected_credit_class_id=class_id
)

if not success:
    print("Cannot mark attendance (validation failed)")
```

**Benefits:**

- Centralized business rules
- Automatic validation (cooldown, duplicates, session matching)
- Auto-enrollment trong credit classes
- Easier to maintain & test

---

### 4. FaceRecognitionManager

**OLD WAY:**

```python
# Scattered logic in app.py
if USE_FACENET and inference_engine:
    # Try inference engine
    pass
elif DEEPFACE_AVAILABLE:
    # Try DeepFace
    pass
else:
    # Fallback
    pass
```

**NEW WAY:**

```python
from app import globals as app_globals

# Load faces
summary = app_globals.face_recognition_manager.load_known_faces(force_reload=True)

# Recognize face
result = app_globals.face_recognition_manager.recognize_face(face_img)
print(f"Student: {result['student_name']}, Confidence: {result['confidence']}")
```

**Benefits:**

- Strategy pattern (inference engine → DeepFace fallback)
- Simplified API
- Better error handling

---

### 5. EventBroadcaster

**OLD WAY:**

```python
from app import globals as app_globals
import json

# Manual SSE broadcasting
with app_globals.sse_clients_lock:
    for client_queue in app_globals.sse_clients:
        try:
            message = f"data: {json.dumps(event_data)}\n\n"
            client_queue.put_nowait(message)
        except queue.Full:
            pass
```

**NEW WAY:**

```python
from app import globals as app_globals

# Simple broadcast
app_globals.event_broadcaster.broadcast_attendance_update(
    student_id=student_id,
    student_name=name,
    action='check_in',
    confidence=0.95
)

# Or generic event
app_globals.event_broadcaster.broadcast_event({
    'type': 'custom_event',
    'data': {'key': 'value'}
})
```

**Benefits:**

- Automatic formatting (SSE protocol)
- Client lifecycle management
- Type-specific broadcast methods

---

## 🚀 Next Steps

### Để hoàn tất migration:

1. **Khởi tạo services trong `app/__init__.py`:**

   ```python
   from app import globals as app_globals
   from app.models import CameraService, AttendanceTracker, FaceRecognitionManager

   # Initialize services
   app_globals.camera_service = CameraService(
       camera_index=CAMERA_INDEX,
       width=CAMERA_WIDTH,
       height=CAMERA_HEIGHT,
       logger=app.logger
   )

   app_globals.attendance_tracker = AttendanceTracker(
       database=db,
       state_manager=state_manager,
       logger=app.logger
   )

   # ... etc

   # Update global references
   app_globals.init_service_references()
   ```

2. **Dần dần refactor blueprints:**

   - Bắt đầu với routes đơn giản
   - Thay thế direct global access bằng service methods
   - Test từng route sau khi refactor

3. **Update `app.py` (legacy camera routes):**

   - Khi sẵn sàng, migrate camera routes sang blueprint
   - Dùng `camera_service` và `face_recognition_manager`
   - Remove dynamic import hack trong `app/__init__.py`

4. **Benefits sau migration:**
   - ✅ Code dễ test (mock services)
   - ✅ Separation of concerns rõ ràng
   - ✅ Thread-safe by default
   - ✅ Easier to add features
   - ✅ Better error handling
   - ✅ Type hints support

---

## 📊 Impact Assessment

### Files Modified:

- ✅ `app/globals.py` - Refactored với backward compatibility
- ✅ Created `app/models/*.py` - 5 new model classes
- ✅ Created `app/models/__init__.py` - Package exports

### Files to Update (next phase):

- ⏳ `app/__init__.py` - Initialize services
- ⏳ `app/routes/*.py` - Gradually migrate to use services
- ⏳ `app.py` - Migrate camera routes to blueprint

### Backward Compatibility:

- ✅ 100% - Code cũ vẫn hoạt động
- ✅ No breaking changes
- ✅ Gradual migration possible

---

## 🐛 Testing Strategy

### Unit Tests (recommended):

```python
# test_state_manager.py
def test_checked_in():
    from app.models import StateManager
    manager = StateManager()
    manager.add_checked_in("SV001", {"name": "Test"})
    assert manager.is_checked_in("SV001")
    assert manager.is_checked_in("sv001")  # Case insensitive

# test_attendance_tracker.py
def test_mark_attendance_with_cooldown():
    tracker = AttendanceTracker(mock_db, state_manager, logger)
    assert tracker.mark_attendance("SV001", "Test Student")
    assert not tracker.mark_attendance("SV001", "Test Student")  # Cooldown
```

### Integration Tests:

```python
# test_camera_service.py
def test_camera_toggle():
    service = CameraService(camera_index=0, logger=app.logger)
    assert service.is_enabled() == True
    service.toggle_camera()
    assert service.is_enabled() == False
```

---

## 📚 Additional Notes

### Thread Safety:

- ✅ Tất cả models đều thread-safe
- ✅ Internal locking trong StateManager
- ✅ Không cần manual lock trong application code

### Performance:

- ✅ Singleton pattern - no overhead
- ✅ Lazy initialization where appropriate
- ✅ No additional memory overhead

### Maintenance:

- ✅ Clear responsibilities
- ✅ Easy to locate bugs
- ✅ Simple to add logging/monitoring
- ✅ Better code organization

---

**Status:** ✅ Models created, ready for integration  
**Next:** Initialize services in `app/__init__.py`  
**Goal:** Clean, maintainable, testable codebase
