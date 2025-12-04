# face_recognition.py - Logic nhận diện khuôn mặt và suy luận

import os
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path

from config import (
    DATA_DIR, USE_FACENET, RESERVED_DATA_SUBDIRS,
    DEEPFACE_SIMILARITY_THRESHOLD
)
from utils import iter_student_face_image_files

# Biến toàn cục cho nhận diện khuôn mặt
known_face_embeddings = []  # np.ndarray (N, D)
known_face_names = []
known_face_ids = []

# Nhập các dịch vụ nếu có sẵn
face_service = None
antispoof_service = None
training_service = None
FACE_RECOGNITION_AVAILABLE = False

if USE_FACENET:
    try:
        from services.face_service import FaceRecognitionService
        from services.antispoof_service import AntiSpoofService
        from services.training_service import TrainingService
        from config import FACENET_THRESHOLD, ANTISPOOF_DEVICE, ANTISPOOF_THRESHOLD

        face_service = FaceRecognitionService(confidence_threshold=FACENET_THRESHOLD)
        antispoof_service = AntiSpoofService(
            device=ANTISPOOF_DEVICE,
            spoof_threshold=ANTISPOOF_THRESHOLD
        )
        FACE_RECOGNITION_AVAILABLE = True
    except Exception as e:
        print(f"Could not initialize FaceNet services: {e}")

# Nhập DeepFace nếu có sẵn
try:
    from deepface import DeepFace
    from services.deepface_db import build_db_from_data_dir, recognize_face as deepface_recognize
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

# Nhập các thành phần của inference engine
try:
    from core.inference.engine import (
        DeepFaceStrategy,
        FaceNetStrategy,
        InferenceEngine,
        InferenceError,
    )
    inference_engine = None
except ImportError:
    inference_engine = None


def configure_inference_engine(app_logger):
    """Khởi tạo inference engine với các chiến lược phù hợp."""
    global inference_engine
    try:
        from config import DEMO_MODE
        inference_engine = InferenceEngine(logger=app_logger, demo_mode=DEMO_MODE)
    except Exception as exc:
        app_logger.warning(f"[Inference] Không thể khởi tạo InferenceEngine: {exc}")
        inference_engine = None
        return

    if DEEPFACE_AVAILABLE:
        try:
            deepface_strategy = DeepFaceStrategy(
                data_dir=DATA_DIR,
                deepface_module=DeepFace,
                build_db_fn=build_db_from_data_dir,
                recognize_fn=deepface_recognize,
                similarity_threshold=DEEPFACE_SIMILARITY_THRESHOLD,
                enforce_detection=False,
                logger=app_logger,
            )
            inference_engine.add_strategy(deepface_strategy)
        except Exception as exc:
            app_logger.warning(f"[Inference] Không thể khởi tạo DeepFace strategy: {exc}")

    if USE_FACENET and face_service is not None:
        try:
            facenet_strategy = FaceNetStrategy(
                service=face_service,
                label_lookup=lambda sid: lookup_student_name(sid, app_logger),
                logger=app_logger,
            )
            inference_engine.add_strategy(facenet_strategy)
        except Exception as exc:
            app_logger.warning(f"[Inference] Không thể khởi tạo FaceNet strategy: {exc}")


def lookup_student_name(student_id: Optional[str], app_logger=None) -> Optional[str]:
    if not student_id:
        return None
    try:
        from database import db
        student = db.get_student(student_id)
        if student:
            return student.get('full_name') or student.get('student_name') or student_id
    except Exception as exc:
        if app_logger:
            app_logger.debug(f"[Inference] Lookup failed cho {student_id}: {exc}")
    return None


def load_known_faces(force_reload: bool = True, app_logger=None):
    """Tải các khuôn mặt đã biết, ưu tiên inference engine nếu khả dụng."""
    global known_face_embeddings, known_face_names, known_face_ids

    if app_logger:
        app_logger.info(f"[LoadFaces] 🔄 Khởi động lại dữ liệu khuôn mặt từ {DATA_DIR}...")

    engine_ready = inference_engine is not None and inference_engine.has_strategies()
    if engine_ready:
        try:
            summary = (
                inference_engine.reload()
                if force_reload
                else inference_engine.warmup(force=False)
            )
            subjects = inference_engine.known_subjects(limit=10_000)
            known_face_embeddings = []
            known_face_ids = []
            known_face_names = []
            for student_id, name in subjects:
                normalized_id = (student_id or name or "UNKNOWN").strip()
                known_face_ids.append(normalized_id)
                known_face_names.append(name or normalized_id)
            if app_logger:
                app_logger.info(
                    "[LoadFaces] ✅ Inference engine sẵn sàng với %d khuôn mặt",
                    len(known_face_ids),
                )
            return summary
        except InferenceError as error:
            if app_logger:
                app_logger.warning(
                    "[LoadFaces] ⚠️ Inference engine reload thất bại: %s. Fallback legacy.",
                    error,
                )
        except Exception as exc:
            if app_logger:
                app_logger.error(
                    "[LoadFaces] ⚠️ Không thể reload inference engine: %s. Fallback legacy.",
                    exc,
                    exc_info=True,
                )

    if not DEEPFACE_AVAILABLE:
        if app_logger:
            app_logger.error(
                "[LoadFaces] ❌ DeepFace không khả dụng. Vui lòng cài đặt: pip install deepface"
            )
        return

    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if app_logger:
            app_logger.info(f"[LoadFaces] ✅ Đã tạo thư mục {DATA_DIR}")

    if app_logger:
        app_logger.info("[LoadFaces] [DeepFace] 🧠 Đang tải ảnh mẫu và tính embedding với Facenet512...")
    db_embeddings = []
    db_labels = []
    processed_count = 0
    failed_count = 0
    image_files = iter_student_face_image_files()
    if app_logger:
        app_logger.info(f"[LoadFaces] 📁 Tìm thấy {len(image_files)} file ảnh (gồm cả thư mục con)")

    for img_path in image_files:
        try:
            filename = img_path.stem
            import re
            student_id = None
            name = None

            try:
                relative_parts = img_path.relative_to(DATA_DIR).parts
            except ValueError:
                relative_parts = ()

            if len(relative_parts) > 1 and relative_parts[0] not in RESERVED_DATA_SUBDIRS:
                student_id = relative_parts[0]

            match = re.match(r'^(\d+)_([A-Za-z\s]+)', filename)
            if match:
                student_id = student_id or match.group(1)
                name = match.group(2).strip()
            else:
                parts = filename.split('_')
                if len(parts) >= 2:
                    student_id = student_id or parts[0]
                    name = '_'.join(parts[1:])
                else:
                    student_id = student_id or filename
                    name = filename

            if app_logger:
                app_logger.debug(
                    f"[LoadFaces] Đang xử lý {img_path.name} -> {name} (ID: {student_id})..."
                )

            embedding = DeepFace.represent(
                img_path=str(img_path),
                model_name="Facenet512",
                enforce_detection=True,
            )[0]["embedding"]

            db_embeddings.append(embedding)
            db_labels.append((student_id, name))
            processed_count += 1
            if app_logger:
                app_logger.info(
                    f"[LoadFaces] ✅ Đã tải khuôn mặt cho {name} (id={student_id}) từ {img_path.name}"
                )
        except Exception as e:
            failed_count += 1
            if app_logger:
                app_logger.error(
                    f"[LoadFaces] ❌ Lỗi khi xử lý ảnh mẫu {img_path.name}: {e}",
                    exc_info=True,
                )

    if len(db_embeddings) > 0:
        known_face_embeddings = np.array(db_embeddings)
        known_face_ids = [sid for sid, _ in db_labels]
        known_face_names = [name for _, name in db_labels]
        if app_logger:
            app_logger.info(
                f"[LoadFaces] ✅ Đã load {len(known_face_embeddings)} ảnh mẫu với Facenet512 embeddings"
            )
            app_logger.info(f"[LoadFaces] 📋 Known faces: {known_face_names}")
            app_logger.info(f"[LoadFaces] 📋 Known IDs: {known_face_ids}")
            app_logger.info(f"[LoadFaces] 📐 Embeddings shape: {known_face_embeddings.shape}")
            app_logger.info(
                f"[LoadFaces] 📊 Kết quả: {processed_count} thành công, {failed_count} thất bại"
            )
    else:
        if app_logger:
            app_logger.warning("[LoadFaces] ⚠️ Không load được ảnh nào!")


def ensure_legacy_embeddings(force_reload: bool = False, app_logger=None) -> None:
    """Đảm bảo bộ embeddings DeepFace được build khi không có inference engine."""
    global known_face_embeddings
    if not DEEPFACE_AVAILABLE:
        return
    engine_ready = inference_engine is not None and inference_engine.has_strategies()
    if engine_ready:
        return  # ưu tiên inference engine
    needs_reload = force_reload or not known_face_embeddings or len(known_face_embeddings) == 0
    if not needs_reload:
        return
    try:
        load_known_faces(force_reload=force_reload, app_logger=app_logger)
    except Exception as exc:
        if app_logger:
            app_logger.warning(f"[LoadFaces] ⚠️ Không thể build legacy embeddings: {exc}")


def recognize_face_candidate(face_img, app_logger=None) -> Dict[str, Any]:
    """Nhận diện khuôn mặt sử dụng inference engine hoặc fallback legacy."""
    result = {
        'student_id': 'UNKNOWN',
        'student_name': 'UNKNOWN',
        'confidence': 0.0,
        'strategy': 'none',
        'status': 'unknown',
    }
    engine_ready = inference_engine is not None and inference_engine.has_strategies()
    if engine_ready:
        try:
            inference_result = inference_engine.identify(face_img)
            sid = inference_result.student_id or 'UNKNOWN'
            name = inference_result.student_name or (sid if sid != 'UNKNOWN' else 'UNKNOWN')
            result.update({
                'student_id': sid,
                'student_name': name,
                'confidence': float(inference_result.confidence or 0.0),
                'strategy': inference_result.strategy or 'inference',
                'status': inference_result.status or ('match' if sid != 'UNKNOWN' else 'no_match'),
            })
            return result
        except InferenceError as error:
            if app_logger:
                app_logger.warning(f"[Inference] Nhận diện thất bại: {error}")
        except Exception as exc:
            if app_logger:
                app_logger.error(f"[Inference] Lỗi nhận diện không xác định: {exc}", exc_info=True)

    if DEEPFACE_AVAILABLE:
        ensure_legacy_embeddings(force_reload=False, app_logger=app_logger)

    if DEEPFACE_AVAILABLE and known_face_embeddings is not None and len(known_face_embeddings) > 0:
        try:
            legacy_embedding = DeepFace.represent(
                face_img,
                model_name="Facenet512",
                enforce_detection=False,
            )[0]["embedding"]
            db_labels = list(zip(known_face_ids, known_face_names))
            student_id, student_name, best_score = deepface_recognize(
                legacy_embedding,
                known_face_embeddings,
                db_labels,
                threshold=DEEPFACE_SIMILARITY_THRESHOLD,
            )
            sid = student_id or 'UNKNOWN'
            name = student_name or (sid if sid != 'UNKNOWN' else 'UNKNOWN')
            result.update({
                'student_id': sid,
                'student_name': name,
                'confidence': float(best_score or 0.0),
                'strategy': 'legacy-deepface',
                'status': 'match' if student_id else 'no_match',
            })
        except Exception as exc:
            if app_logger:
                app_logger.error(f"[Inference] ❌ Lỗi nhận diện legacy: {exc}", exc_info=True)
    return result


def prewhiten_facenet(x):
    """
    FaceNet-style prewhitening để chuẩn hóa tốt hơn.
    Được điều chỉnh từ face_attendance/facenet.py
    """
    if isinstance(x, np.ndarray):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1 / std_adj)
        return y
    return x


def estimate_head_pose(landmarks, frame_size):
    """
    Ước tính tư thế đầu đơn giản (yaw, pitch, roll) theo độ bằng solvePnP.
    landmarks: dictionary hoặc danh sách các điểm (x,y) cho các mốc quan trọng (chúng tôi mong đợi ít nhất
    left_eye, right_eye, nose, left_mouth, right_mouth) hoặc danh sách theo thứ tự trả về
    bởi dlib/face_recognition: chúng tôi sẽ cố gắng xử lý các định dạng phổ biến.
    Trả về (yaw_deg, pitch_deg, roll_deg) hoặc (None, None, None) nếu thất bại.
    """
    try:
        import cv2
        import math

        # Điểm mô hình cho khuôn mặt trung bình (đơn vị: mm)
        model_points = np.array([
                (0.0, 0.0, 0.0),          # Đỉnh mũi
                (0.0, -330.0, -65.0),     # Cằm
                (-225.0, 170.0, -135.0),  # Góc ngoài mắt trái
                (225.0, 170.0, -135.0),   # Góc ngoài mắt phải
                (-150.0, -150.0, -125.0), # Góc miệng trái
                (150.0, -150.0, -125.0)   # Góc miệng phải
        ])

        # Xử lý landmarks
        if isinstance(landmarks, dict):
            # Dictionary format
            nose = landmarks.get('nose')
            left_eye = landmarks.get('left_eye')
            right_eye = landmarks.get('right_eye')
            left_mouth = landmarks.get('left_mouth')
            right_mouth = landmarks.get('right_mouth')
            if not all([nose, left_eye, right_eye, left_mouth, right_mouth]):
                return None, None, None
            image_points = np.array([
                nose,
                (nose[0], nose[1] + 100),  # Approximate chin
                left_eye,
                right_eye,
                left_mouth,
                right_mouth
            ], dtype=np.float32)
        elif isinstance(landmarks, list) and len(landmarks) >= 68:
            # Dlib 68-point format
            image_points = np.array([
                landmarks[30],  # Nose
                landmarks[8],   # Chin
                landmarks[36],  # Left eye left
                landmarks[45],  # Right eye right
                landmarks[48],  # Left mouth
                landmarks[54]   # Right mouth
            ], dtype=np.float32)
        else:
            return None, None, None

        # Camera matrix
        focal_length = frame_size[1]
        center = (frame_size[1] / 2, frame_size[0] / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)

        dist_coeffs = np.zeros((4, 1))  # Giả sử không có sai lệch ống kính

        # Giải bài toán PnP (Perspective-n-Point)
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None, None, None

        # Chuyển vector quay sang góc Euler
        rmat, _ = cv2.Rodrigues(rotation_vector)
        sy = math.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
        singular = sy < 1e-6

        if not singular:
            yaw = math.atan2(rmat[1, 0], rmat[0, 0])
            pitch = math.atan2(-rmat[2, 0], sy)
            roll = math.atan2(rmat[2, 1], rmat[2, 2])
        else:
            yaw = math.atan2(-rmat[0, 1], rmat[1, 1])
            pitch = math.atan2(-rmat[2, 0], sy)
            roll = 0

        # Convert to degrees
        yaw_deg = math.degrees(yaw)
        pitch_deg = math.degrees(pitch)
        roll_deg = math.degrees(roll)

        return yaw_deg, pitch_deg, roll_deg

    except Exception as e:
        return None, None, None