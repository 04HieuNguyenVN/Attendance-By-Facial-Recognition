# camera.py - Camera handling and video streaming

import cv2
import numpy as np
import random
import time
import os
from datetime import datetime
from flask import Response, stream_with_context

from config import (
    DEMO_MODE, YOLO_FRAME_SKIP, YOLO_INFERENCE_WIDTH,
    RECOGNITION_COOLDOWN, CAMERA_WIDTH, CAMERA_HEIGHT
)
from face_recognition import recognize_face_candidate
from attendance import (
    mark_attendance, mark_student_checkout, update_presence,
    check_presence_timeout, update_progress, draw_progress_bar,
    today_checked_in, today_checked_out, last_recognized, last_recognized_lock
)
from utils import role_required

# Global camera state
vision_state = None
camera_enabled = True

# Import camera manager
try:
    from core.vision.camera_manager import CameraError
    from core.vision.state import VisionPipelineState, VisionStateConfig
except ImportError:
    VisionPipelineState = None
    VisionStateConfig = None
    CameraError = Exception


def get_or_create_vision_state():
    """Get or create vision pipeline state"""
    global vision_state
    if vision_state is None and VisionStateConfig:
        config = VisionStateConfig(
            index=0,  # CAMERA_INDEX from config
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            warmup_frames=3,  # CAMERA_WARMUP_FRAMES
            buffer_size=2,  # CAMERA_BUFFER_SIZE
        )
        vision_state = VisionPipelineState(config=config)
    return vision_state


def ensure_camera_pipeline():
    """Ensure camera pipeline is ready"""
    if not camera_enabled:
        return None
    state = get_or_create_vision_state()
    if state:
        try:
            return state.ensure_ready()
        except CameraError as exc:
            print(f"[Camera] Không thể khởi động camera: {exc}")
    return None


def release_camera_capture():
    """Release camera capture"""
    state = vision_state
    if state is None:
        return
    try:
        state.set_enabled(False)
        state.stop()
    except Exception as exc:
        print(f"[Camera] Không thể giải phóng camera: {exc}")


def make_placeholder_frame(message: str = "Camera không khả dụng"):
    """Create placeholder frame"""
    h, w = 480, 640
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    text_size, _ = cv2.getTextSize(message, font, scale, thickness)
    text_w, text_h = text_size
    x = max(10, (w - text_w) // 2)
    y = max(30, (h - text_h) // 2)
    cv2.putText(img, message, (x, y), font, scale, (200, 200, 200), thickness, cv2.LINE_AA)
    ret, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ret:
        return None
    return buf.tobytes()


def generate_frames(
    expected_student_id: str = None,
    selected_action: str = 'checkin',
    enforce_student_match: bool = False,
    expected_credit_class_id: int = None,
    app_logger=None,
):
    """Generator for video frames"""
    global camera_enabled

    if app_logger:
        app_logger.info("generate_frames() started")

    enforced_student_id = (expected_student_id or '').strip() if enforce_student_match else None
    requested_action = (selected_action or 'checkin').lower()
    if requested_action not in ('checkin', 'checkout', 'auto'):
        requested_action = 'auto'

    # If camera is disabled, send placeholder continuously
    if not camera_enabled:
        placeholder = make_placeholder_frame("Camera đã tắt")
        if placeholder is None:
            return
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n')
        return

    pipeline = ensure_camera_pipeline()
    if pipeline is None:
        placeholder = make_placeholder_frame("Không thể khởi động camera")
        if placeholder is None:
            return
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n')
        return

    frame_count = 0
    detection_frame_counter = YOLO_FRAME_SKIP  # Force YOLO run on first frame
    cached_face_data = []
    inference_warmed_up = False

    # Import YOLO if available
    yolo_face_model = None
    try:
        from ultralytics import YOLO
        possible_paths = [
            'yolov8m-face.pt',
            os.path.join('models', 'yolov8m-face.pt'),
            os.path.join('Cong-Nghe-Xu-Ly-Anh', 'yolov8m-face.pt'),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                yolo_face_model = YOLO(path)
                break
    except ImportError:
        pass

    while True:
        # Check if camera is disabled
        if not camera_enabled:
            if app_logger:
                app_logger.info("Camera đã tắt, dừng stream")
            break

        try:
            vision_frame = pipeline.get_frame()
            frame = vision_frame.bgr
            frame_count += 1
            if frame_count % 30 == 0 and app_logger:
                app_logger.debug(f"[Camera] Đang đọc frame #{frame_count}...")
        except CameraError as exc:
            if app_logger:
                app_logger.warning(f"[Camera] Mất kết nối camera: {exc}")
            release_camera_capture()
            time.sleep(0.2)
            pipeline = ensure_camera_pipeline()
            if pipeline is None:
                placeholder = make_placeholder_frame("Camera lỗi - đang thử lại")
                if placeholder is None:
                    break
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n')
                time.sleep(0.5)
            continue
        except Exception as exc:
            if app_logger:
                app_logger.error(f"[Camera] Lỗi đọc frame: {exc}", exc_info=True)
            time.sleep(0.2)
            continue

        # Get frame dimensions
        frame_h, frame_w = frame.shape[:2]

        # Flip frame horizontally (mirror effect)
        frame = cv2.flip(frame, 1)

        # Face detection and recognition
        face_data = []
        detection_frame_counter += 1
        should_run_detection = detection_frame_counter >= YOLO_FRAME_SKIP

        if yolo_face_model and should_run_detection:
            detection_frame_counter = 0
            detection_frame = frame
            scale_x = scale_y = 1.0
            if YOLO_INFERENCE_WIDTH > 0 and frame_w > YOLO_INFERENCE_WIDTH:
                detection_width = YOLO_INFERENCE_WIDTH
                detection_height = int(frame_h * (detection_width / frame_w))
                detection_frame = cv2.resize(
                    frame,
                    (detection_width, detection_height),
                    interpolation=cv2.INTER_LINEAR
                )
                scale_x = frame_w / detection_width
                scale_y = frame_h / detection_height

            results = yolo_face_model(detection_frame, verbose=False)[0]
            boxes = results.boxes.xyxy.cpu().numpy()
            new_face_data = []

            for box in boxes:
                xmin, ymin, xmax, ymax = map(int, box)
                xmin = int(xmin * scale_x)
                xmax = int(xmax * scale_x)
                ymin = int(ymin * scale_y)
                ymax = int(ymax * scale_y)

                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(frame_w, xmax)
                ymax = min(frame_h, ymax)

                face_img = frame[ymin:ymax, xmin:xmax]
                if face_img.size == 0:
                    continue

                recognition = recognize_face_candidate(face_img)
                student_id = (recognition.get('student_id') or 'UNKNOWN').strip() or 'UNKNOWN'
                name = recognition.get('student_name') or (student_id if student_id != 'UNKNOWN' else 'UNKNOWN')
                confidence_score = float(recognition.get('confidence') or 0.0)
                strategy = recognition.get('strategy', 'none')
                recognition_status = recognition.get('status', 'unknown')

                status = 'unknown'
                now = datetime.now()

                if student_id != 'UNKNOWN':
                    recognized_id_norm = (student_id or '').strip().upper()
                    checked_in = recognized_id_norm in today_checked_in
                    checked_out = recognized_id_norm in today_checked_out
                    with last_recognized_lock:
                        last_time = last_recognized.get(student_id)
                        cooldown_passed = not last_time or (now - last_time).total_seconds() > RECOGNITION_COOLDOWN

                    guard_student_id = enforced_student_id if enforce_student_match else None
                    guard_credit_class = expected_credit_class_id
                    recognized_id_norm = (student_id or '').strip().upper()
                    guard_id_norm = (
                        (guard_student_id or '').strip().upper()
                        if guard_student_id
                        else None
                    )
                    mismatch = guard_id_norm and recognized_id_norm != guard_id_norm

                    if mismatch:
                        status = 'mismatch'
                        if app_logger:
                            app_logger.warning(
                                "[Guard] Student mismatch: recognized=%s expected=%s action=%s class=%s",
                                recognized_id_norm, guard_id_norm, requested_action, guard_credit_class,
                            )
                    elif requested_action == 'checkout':
                        if checked_in and not checked_out:
                            if mark_student_checkout(
                                student_id,
                                student_name=name,
                                reason='auto',
                                confidence_score=confidence_score,
                                expected_student_id=guard_student_id,
                                expected_credit_class_id=guard_credit_class,
                            ):
                                status = 'checked_out'
                                with last_recognized_lock:
                                    last_recognized[student_id] = now
                            else:
                                status = 'already_marked'
                        elif not checked_in:
                            status = 'not_checked_in'
                        elif checked_out:
                            status = 'checked_out'
                        else:
                            status = 'cooldown'
                    else:
                        if not checked_in and cooldown_passed:
                            try:
                                success = mark_attendance(
                                    name,
                                    student_id=student_id,
                                    confidence_score=confidence_score,
                                    expected_student_id=guard_student_id,
                                    expected_credit_class_id=guard_credit_class,
                                )
                                if success:
                                    status = 'checked_in'
                                    with last_recognized_lock:
                                        last_recognized[student_id] = now
                                else:
                                    status = 'already_marked'
                            except Exception as e:
                                if app_logger:
                                    app_logger.error(f"[System] Lỗi điểm danh: {e}")
                        elif (
                            requested_action == 'auto'
                            and checked_in
                            and not checked_out
                            and cooldown_passed
                        ):
                            if mark_student_checkout(
                                student_id,
                                student_name=name,
                                reason='auto',
                                confidence_score=confidence_score,
                            ):
                                status = 'checked_out'
                                with last_recognized_lock:
                                    last_recognized[student_id] = now
                            else:
                                status = 'already_marked'
                        elif checked_in and not checked_out:
                            status = 'already_marked'
                        elif checked_out:
                            status = 'checked_out'
                        else:
                            status = 'cooldown' if not cooldown_passed else 'already_marked'
                else:
                    status = recognition_status or 'unknown'

                new_face_data.append({
                    'bbox': (xmin, ymin, xmax, ymax),
                    'name': name,
                    'student_id': student_id,
                    'confidence': confidence_score,
                    'status': status,
                    'strategy': strategy,
                })

            cached_face_data = new_face_data
            face_data = new_face_data
        elif DEMO_MODE:
            # Demo mode - simulate faces
            face_data = []

            # Import known faces
            from face_recognition import known_face_names, known_face_ids

            if known_face_names:
                idx = frame_count % len(known_face_names)
                demo_name = known_face_names[idx]
                demo_id = known_face_ids[idx] if idx < len(known_face_ids) else 'DEMO'
                demo_confidence = 0.85 + (random.random() * 0.15)
                status = 'confirmed'
            else:
                demo_name = 'Demo Mode - Đang chờ khuôn mặt'
                demo_id = 'DEMO'
                demo_confidence = 0.0
                status = 'waiting'

            face_size_w = frame_w // 3
            face_size_h = int(face_size_w * 1.3)
            center_x = frame_w // 2
            center_y = frame_h // 2
            left = center_x - face_size_w // 2
            top = center_y - face_size_h // 2
            right = center_x + face_size_w // 2
            bottom = center_y + face_size_h // 2

            left = max(10, left)
            top = max(10, top)
            right = min(frame_w - 10, right)
            bottom = min(frame_h - 10, bottom)

            face_info = {
                'bbox': (left, top, right, bottom),
                'name': demo_name,
                'confidence': demo_confidence,
                'student_id': demo_id,
                'status': status
            }
            face_data.append(face_info)

            if status == 'confirmed' and frame_count % 30 == 0:
                try:
                    mark_attendance(demo_name, student_id=demo_id, confidence_score=demo_confidence)
                    update_presence(demo_id, demo_name)
                except Exception as e:
                    if app_logger:
                        app_logger.error(f"Lỗi xác nhận điểm danh cho {demo_name}: {e}")
            elif status == 'confirmed' and frame_count % 60 == 0:
                try:
                    update_presence(demo_id, demo_name)
                except Exception as e:
                    if app_logger:
                        app_logger.error(f"Lỗi cập nhật presence cho {demo_name}: {e}")
        else:
            face_data = cached_face_data or []

        # Draw bounding boxes and labels
        for face_info in face_data:
            left, top, right, bottom = face_info['bbox']
            name = face_info.get('name', 'Unknown')
            confidence = face_info.get('confidence', 0.0)
            status = face_info.get('status', 'detected')

            # Choose color based on status
            if status == 'waiting':
                color = (255, 165, 0)
                thickness = 2
            elif status == 'already_marked':
                color = (128, 128, 128)
                thickness = 2
            elif status == 'confirming':
                color = (0, 165, 255)
                thickness = 3
                progress = face_info.get('progress', 0.0)
                draw_progress_bar(frame, progress, left, top)
            elif status in ('confirmed', 'checked_in'):
                color = (0, 255, 0)
                thickness = 3
            elif status == 'checked_out':
                color = (0, 128, 255)
                thickness = 3
            elif status == 'mismatch':
                color = (0, 0, 255)
                thickness = 2
            elif name == "Unknown" or status == 'unknown':
                color = (0, 0, 255)
                thickness = 2
            elif status == 'low_confidence':
                color = (0, 165, 255)
                thickness = 2
            elif status == 'cooldown':
                color = (128, 128, 128)
                thickness = 2
            else:
                color = (0, 165, 255)
                thickness = 2

            cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)

            # Create label
            if status == 'already_marked':
                label = f"{name} - Đã điểm danh"
            elif status == 'confirming':
                label = f"{name} - Đang xác nhận..."
            elif status in ('confirmed', 'checked_in'):
                label = f"{name} - THÀNH CÔNG!"
            elif status == 'checked_out':
                label = f"{name} - Đã ra về"
            elif status == 'mismatch':
                label = f"{name} - Sai tài khoản"
            elif name == "Unknown":
                label = "Unknown - Chưa đăng ký"
            elif status == 'low_confidence':
                label = f"{name} (Confidence thấp: {confidence*100:.1f}%)"
            elif status == 'cooldown':
                label = f"{name} - Vừa điểm danh (chờ {RECOGNITION_COOLDOWN}s)"
            elif status == 'not_checked_in':
                label = f"{name} - Cần check-in trước"
            elif confidence > 0:
                label = f"{name} ({confidence*100:.1f}%)"
            else:
                label = name

            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
            label_x = left
            label_y = top - 10 if top > 30 else bottom + 30

            # Draw label background
            padding = 5
            cv2.rectangle(frame,
                         (label_x - padding, label_y - label_size[1] - padding),
                         (label_x + label_size[0] + padding, label_y + padding),
                         color, -1)

            # Draw label text
            cv2.putText(frame, label, (label_x, label_y),
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        # Check if frame is valid
        if frame is None or frame.size == 0:
            continue

        frame_count += 1

        # Check presence timeout every 100 frames (~3 seconds)
        if frame_count % 100 == 0:
            try:
                check_presence_timeout()
            except Exception as e:
                if app_logger:
                    app_logger.error(f"Lỗi kiểm tra presence timeout: {e}")

        # Encode frame with reduced quality
        ret2, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ret2:
            continue
        frame_bytes = buf.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    release_camera_capture()