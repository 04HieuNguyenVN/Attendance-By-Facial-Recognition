# 🔧 Migration Progress & Next Steps

## ✅ Completed (Đã Hoàn Thành)

### 1. Core Structure

- [x] Application factory pattern (`app/__init__.py`)
- [x] Configuration management (`app/config.py`)
- [x] New entry point (`run.py`)
- [x] Updated start script (`start.bat`)

### 2. Middleware

- [x] Authentication middleware (`app/middleware/auth.py`)
- [x] Authorization decorators (`@role_required`)
- [x] Session management

### 3. Utilities

- [x] File utilities (`app/utils/file_utils.py`)
- [x] Data utilities (`app/utils/data_utils.py`)

### 4. Routes (Blueprints)

- [x] Authentication routes (`app/routes/auth.py`)
  - `/login` (GET, POST)
  - `/logout` (GET)
- [x] Main page routes (`app/routes/main.py`)
  - `/` (index/home)
  - `/students`
  - `/classes`
  - `/reports`
  - `/teacher/credit-classes`
  - `/student/portal`
  - `/status`
- [x] Student API (`app/routes/api_students.py`)
  - `GET /api/students` - List students
  - `POST /api/students` - Create student
  - `GET /api/students/<id>` - Get student
  - `PUT /api/students/<id>` - Update student
  - `DELETE /api/students/<id>` - Delete student
- [x] Class API (`app/routes/api_classes.py`)
  - `GET /api/classes` - List classes
  - `POST /api/classes` - Create class
  - `GET /api/classes/<id>` - Get class
  - `PUT /api/classes/<id>` - Update class
  - `DELETE /api/classes/<id>` - Delete class
  - `GET /api/classes/<id>/students` - Get class students
- [x] Camera API (placeholder) (`app/routes/api_camera.py`)
  - `GET /video_feed` - Video streaming
  - `GET /api/camera/status` - Camera status
  - `POST /api/camera/toggle` - Toggle camera
  - `POST /api/camera/capture` - Capture image
- [x] Compatibility layer (`app/routes/compat.py`)
  - Backward-compatible endpoint mapping

### 5. Templates

- [x] Fixed `url_for()` calls in templates
  - `url_for('index')` → `url_for('main.index')`
  - `url_for('login')` → `url_for('auth.login')`
  - `url_for('student_portal_page')` → `url_for('main.student_portal')`

## ⏳ TODO (Cần Làm Tiếp)

### High Priority (Ưu Tiên Cao)

#### 1. Attendance API Routes

**File**: `app/routes/api_attendance.py`

Routes cần migrate từ `app.py`:

- `GET /api/attendance/session` - Get active session
- `POST /api/attendance/session/open` - Open attendance session
- `POST /api/attendance/session/close` - Close attendance session
- `POST /api/attendance/session/<id>/mark` - Mark attendance
- `GET /api/attendance/today` - Today's attendance
- `GET /api/attendance/history/<student_id>` - Student attendance history
- `GET /api/attendance/notifications` - Get notifications

#### 2. Statistics API Routes

**File**: `app/routes/api_statistics.py`

Routes cần migrate:

- `GET /api/statistics` - System statistics
- `GET /api/presence/active` - Active presence list

#### 3. Credit Classes API Routes

**File**: `app/routes/api_credit_classes.py`

Routes cần migrate:

- `GET /api/credit-classes` - List credit classes
- `POST /api/credit-classes` - Create credit class
- `GET /api/credit-classes/<id>` - Get credit class
- `PUT /api/credit-classes/<id>` - Update credit class
- `DELETE /api/credit-classes/<id>` - Delete credit class
- `GET /api/teacher/credit-classes` - Teacher's credit classes
- `GET /api/teacher/credit-classes/<id>/students` - Class students
- `GET /api/teacher/credit-classes/<id>/sessions` - Class sessions
- `GET /api/student/credit-classes` - Student's enrolled classes

#### 4. Reports API Routes

**File**: `app/routes/api_reports.py`

Routes cần migrate:

- `GET /api/reports/credit-classes/<id>/sessions` - Session reports

#### 5. Training & Model Routes

**File**: `app/routes/api_training.py`

Routes cần migrate:

- `POST /update_faces` - Update face database
- `POST /api/train/start` - Start training
- `GET /api/train/status` - Training status
- `POST /api/antispoof/check` - Anti-spoof check

#### 6. Quick Register Route

**File**: `app/routes/api_register.py`

Routes cần migrate:

- `POST /api/quick-register` - Quick student registration

#### 7. SSE (Server-Sent Events) Route

**File**: `app/routes/api_events.py`

Routes cần migrate:

- `GET /api/events/stream` - Event streaming

### Medium Priority (Ưu Tiên Trung Bình)

#### 8. Camera/Video Service

**File**: `app/services/camera_service.py`

Logic cần migrate:

- Camera management (CameraManager)
- Video streaming
- Frame processing
- Camera state management

#### 9. Face Recognition Service

**File**: `app/services/face_recognition.py`

Logic cần migrate:

- Face detection
- Face encoding
- Face matching
- DeepFace integration
- FaceNet integration
- Inference engine management

#### 10. Attendance Service

**File**: `app/services/attendance_service.py`

Business logic:

- Session management
- Attendance marking
- Check-in/check-out logic
- Attendance validation
- Statistics calculation

### Low Priority (Ưu Tiên Thấp)

#### 11. Models Layer

**File**: `app/models/*.py`

Tạo data models nếu cần:

- Student model
- Class model
- Attendance model
- Session model

#### 12. Tests

**Directory**: `tests/`

Tạo unit tests:

- Test authentication
- Test API endpoints
- Test services
- Test utilities

## 📝 Migration Steps (Các Bước Thực Hiện)

### Để Migrate Một Route Mới:

1. **Xác định route trong app.py**

   ```bash
   # Tìm route definition
   grep -n "@app.route('/your-route')" app.py
   ```

2. **Tạo hoặc mở file blueprint tương ứng**

   - Authentication → `app/routes/auth.py`
   - Main pages → `app/routes/main.py`
   - APIs → `app/routes/api_*.py`

3. **Copy route function**

   - Copy function decorator và function body
   - Thay `@app.route` → `@blueprint_name.route`
   - Import các dependencies cần thiết

4. **Update imports**

   - Import từ `app.middleware.auth` thay vì trực tiếp
   - Import từ `app.utils` thay vì local functions
   - Import từ `database` và các modules khác

5. **Đăng ký blueprint**

   - Thêm vào `app/routes/__init__.py`
   - Import blueprint
   - Gọi `app.register_blueprint()`

6. **Test route**
   - Chạy `python test_app_structure.py`
   - Kiểm tra route có trong danh sách
   - Test trên browser/Postman

### Để Migrate Business Logic Sang Service:

1. **Tạo service file**

   ```python
   # app/services/my_service.py
   class MyService:
       def __init__(self, logger=None):
           self.logger = logger

       def do_something(self):
           # Business logic here
           pass
   ```

2. **Import và sử dụng trong route**

   ```python
   from app.services.my_service import MyService

   @bp.route('/endpoint')
   def my_route():
       service = MyService(app.logger)
       result = service.do_something()
       return jsonify(result)
   ```

## 🔍 How to Find What to Migrate Next

### 1. Check API calls in JavaScript

```bash
grep -r "/api/" static/js/
```

### 2. Check routes in app.py

```bash
grep -n "@app.route" app.py | wc -l  # Count remaining routes
```

### 3. Check template url_for calls

```bash
grep -r "url_for(" templates/
```

### 4. Run test to see errors

```bash
python run.py
# Try accessing different pages
# Check console for 404/500 errors
```

## 🐛 Common Issues & Solutions

### Issue: `BuildError: Could not build url for endpoint 'xxx'`

**Solution**: Update template to use blueprint name

```html
<!-- Old -->
{{ url_for('login') }}

<!-- New -->
{{ url_for('auth.login') }}
```

### Issue: `404 Not Found` on API endpoint

**Solution**:

1. Check if blueprint is registered in `app/routes/__init__.py`
2. Check URL prefix matches
3. Verify route decorator path

### Issue: `ImportError: cannot import name 'xxx'`

**Solution**:

1. Check if module exists in new structure
2. Update import path from old to new location
3. Check circular imports

### Issue: Global variables not accessible

**Solution**:

1. Move to `app.config` for constants
2. Use `current_app` for app context
3. Create service classes for stateful logic

## 📊 Progress Tracking

- **Total Routes in app.py**: ~47 routes
- **Migrated**: 22 routes (47%)
- **Remaining**: ~25 routes (53%)

### By Category:

- ✅ Auth: 100% (2/2)
- ✅ Main Pages: 100% (7/7)
- ✅ Student API: 100% (5/5)
- ✅ Class API: 100% (6/6)
- ⏳ Attendance API: 0% (0/8)
- ⏳ Credit Class API: 0% (0/9)
- ⏳ Statistics API: 0% (0/2)
- ⏳ Training API: 0% (0/4)
- ⏳ Camera/Video: 20% (1/5) - placeholder only
- ⏳ Other: 0% (0/6)

## 🎯 Next Immediate Steps

1. **Test current implementation**

   ```bash
   python run.py
   # Access http://localhost:5000
   # Try login, view pages
   ```

2. **Fix any template issues**

   - Check all `url_for()` calls
   - Update to use blueprint names

3. **Migrate Attendance API** (Most important)

   - Create `app/routes/api_attendance.py`
   - Move session management logic
   - Move marking logic

4. **Create Attendance Service**

   - Extract business logic from routes
   - Move to `app/services/attendance_service.py`

5. **Migrate Camera functionality**
   - Complete `app/routes/api_camera.py`
   - Create `app/services/camera_service.py`
   - Move video streaming logic

## 📚 References

- Old codebase: `app.py` (keep as reference)
- New structure: `REFACTORING_GUIDE.md`
- Quick start: `REFACTORING_QUICKSTART.md`
- Test script: `test_app_structure.py`

---

**Last Updated**: 2024-12-04
**Status**: 🚧 In Progress - Core structure complete, APIs partially migrated
