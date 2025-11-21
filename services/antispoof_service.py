"""
Anti-spoofing service to detect face presentation attacks.

Uses MiniFASNet models to detect:
- Printed photos
- Video replay attacks
- 3D masks (partially)
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AntiSpoofService:
    """Detect face spoofing attacks using MiniFASNet ensemble."""
    
    def __init__(self, 
                 models_dir: str = 'face_attendance/resources/anti_spoof_models',
                 device: str = 'cpu',
                 spoof_threshold: float = 0.5):
        """
        Initialize anti-spoofing service.
        
        Args:
            models_dir: Directory containing MiniFASNet models
            device: 'cpu' or 'cuda'
            spoof_threshold: Threshold for real vs spoof (higher = stricter)
        """
        self.models_dir = Path(models_dir)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.spoof_threshold = spoof_threshold
        
        # Face detector for getting bbox
        self.face_detector = None
        self._init_detector()
        
        logger.info(f"AntiSpoofService đã khởi tạo trên {self.device}")
    
    def _init_detector(self):
        """Initialize RetinaFace detector for anti-spoof."""
        try:
            caffemodel = 'face_attendance/resources/detection_model/Widerface-RetinaFace.caffemodel'
            deploy = 'face_attendance/resources/detection_model/deploy.prototxt'
            
            if not Path(caffemodel).exists() or not Path(deploy).exists():
                logger.warning("Không tìm thấy mô hình RetinaFace, phát hiện chống giả mạo bị vô hiệu")
                return
            
            self.face_detector = cv2.dnn.readNetFromCaffe(deploy, caffemodel)
            logger.info("Bộ phát hiện RetinaFace đã được tải cho chức năng chống giả mạo")
        except Exception as e:
            logger.error(f"Tải bộ phát hiện khuôn mặt thất bại: {e}")
            self.face_detector = None
    
    def get_bbox(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect face bounding box using RetinaFace.
        
        Args:
            image: BGR image
            
        Returns:
            (x, y, w, h) or None if no face detected
        """
        if self.face_detector is None:
            # Fallback to OpenCV
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) > 0:
                return tuple(faces[0])
            return None
        
        height, width = image.shape[:2]
        aspect_ratio = width / height
        
        # Resize for faster detection
        if width * height >= 192 * 192:
            import math
            resized = cv2.resize(image, 
                               (int(192 * math.sqrt(aspect_ratio)),
                                int(192 / math.sqrt(aspect_ratio))),
                               interpolation=cv2.INTER_LINEAR)
        else:
            resized = image
        
        # Prepare blob
        blob = cv2.dnn.blobFromImage(resized, 1.0, mean=(104, 117, 123))
        self.face_detector.setInput(blob, 'data')
        detections = self.face_detector.forward('detection_out').squeeze()
        
        if len(detections.shape) == 1:
            return None
        
        # Get highest confidence detection
        max_conf_idx = np.argmax(detections[:, 2])
        confidence = detections[max_conf_idx, 2]
        
        if confidence < 0.5:
            return None
        
        # Scale back to original size
        left = int(detections[max_conf_idx, 3] * width)
        top = int(detections[max_conf_idx, 4] * height)
        right = int(detections[max_conf_idx, 5] * width)
        bottom = int(detections[max_conf_idx, 6] * height)
        
        bbox = [left, top, right - left + 1, bottom - top + 1]
        return tuple(bbox)
    
    def crop_face(self, image: np.ndarray, bbox: Tuple[int, int, int, int],
                  scale: float, out_w: int, out_h: int) -> np.ndarray:
        """
        Crop and resize face region for anti-spoof model.
        
        Args:
            image: Original image
            bbox: (x, y, w, h)
            scale: Crop scale factor
            out_w, out_h: Output size
            
        Returns:
            Cropped and resized face patch
        """
        x, y, w, h = bbox
        height, width = image.shape[:2]
        
        # Calculate crop box with margin
        x1 = max(0, int(x - w * (scale - 1) / 2))
        y1 = max(0, int(y - h * (scale - 1) / 2))
        x2 = min(width, int(x + w * (1 + (scale - 1) / 2)))
        y2 = min(height, int(y + h * (1 + (scale - 1) / 2)))
        
        # Crop
        cropped = image[y1:y2, x1:x2]
        
        # Resize
        resized = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        
        return resized
    
    def _load_model(self, model_path: str):
        """
        Load a MiniFASNet model.
        
        Args:
            model_path: Path to .pth file
            
        Returns:
            Loaded model
        """
        try:
            from face_attendance.src.model_lib.MiniFASNet import (
                MiniFASNetV1, MiniFASNetV2, MiniFASNetV1SE, MiniFASNetV2SE
            )
            from face_attendance.src.utility import parse_model_name, get_kernel
        except ImportError:
            logger.error("Không có mô hình MiniFASNet")
            return None
        
        model_name = os.path.basename(model_path)
        h_input, w_input, model_type, _ = parse_model_name(model_name)
        kernel_size = get_kernel(h_input, w_input)
        
        # Map model type to class
        MODEL_MAP = {
            'MiniFASNetV1': MiniFASNetV1,
            'MiniFASNetV2': MiniFASNetV2,
            'MiniFASNetV1SE': MiniFASNetV1SE,
            'MiniFASNetV2SE': MiniFASNetV2SE
        }
        
        model_class = MODEL_MAP.get(model_type)
        if model_class is None:
            logger.error(f"Loại mô hình không xác định: {model_type}")
            return None
        
        model = model_class(conv6_kernel=kernel_size).to(self.device)
        
        # Load weights
        state_dict = torch.load(model_path, map_location=self.device)
        
        # Handle 'module.' prefix from DataParallel
        if list(state_dict.keys())[0].startswith('module.'):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k[7:]] = v
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
        model.eval()
        
        return model, h_input, w_input
    
    def predict_single_model(self, image: np.ndarray, model_path: str) -> np.ndarray:
        """
        Get anti-spoof prediction from a single model.
        
        Args:
            image: Face patch (BGR)
            model_path: Path to model weights
            
        Returns:
            Prediction array [spoof_score, real_score, ???]
        """
        try:
            from face_attendance.src.data_io import transform as trans
        except ImportError:
            logger.error("Module transform không khả dụng")
            return np.array([0.5, 0.5, 0.0])
        
        # Load model
        model_info = self._load_model(model_path)
        if model_info is None:
            return np.array([0.5, 0.5, 0.0])
        
        model, _, _ = model_info
        
        # Preprocess
        test_transform = trans.Compose([trans.ToTensor()])
        img_tensor = test_transform(image)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            result = model(img_tensor)
            result = F.softmax(result, dim=1).cpu().numpy()
        
        return result[0]
    
    def is_real_face(self, image: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> Tuple[bool, float]:
        """
        Check if face is real (not spoofed).
        
        Args:
            image: BGR image
            bbox: Optional pre-computed bbox. If None, will detect.
            
        Returns:
            (is_real, confidence) tuple
        """
        if not self.models_dir.exists():
            logger.warning("Không tìm thấy mô hình chống giả mạo, bỏ qua kiểm tra")
            return (True, 1.0)  # Default to real if no models
        
        # Get bbox if not provided
        if bbox is None:
            bbox = self.get_bbox(image)
            if bbox is None:
                logger.warning("Không phát hiện khuôn mặt cho kiểm tra chống giả mạo")
                return (False, 0.0)
        
        # Ensemble prediction from all models
        prediction_sum = np.zeros(3)
        model_count = 0
        
        try:
            from face_attendance.src.utility import parse_model_name
            
            for model_name in os.listdir(self.models_dir):
                if not model_name.endswith('.pth'):
                    continue
                
                model_path = self.models_dir / model_name
                h_input, w_input, model_type, scale = parse_model_name(model_name)
                
                # Crop face for this model
                if scale is not None:
                    face_patch = self.crop_face(image, bbox, scale, w_input, h_input)
                else:
                    # No crop, just resize
                    x, y, w, h = bbox
                    face_patch = image[y:y+h, x:x+w]
                    face_patch = cv2.resize(face_patch, (w_input, h_input))
                
                # Get prediction
                pred = self.predict_single_model(face_patch, str(model_path))
                prediction_sum += pred
                model_count += 1
        
        except Exception as e:
            logger.error(f"Dự đoán chống giả mạo thất bại: {e}")
            return (True, 0.5)  # Default to uncertain
        
        if model_count == 0:
            return (True, 0.5)
        
        # Average predictions
        avg_prediction = prediction_sum / model_count
        
        # Class 1 = real, class 0 = spoof
        label = np.argmax(avg_prediction)
        confidence = avg_prediction[label]
        
        is_real = (label == 1 and confidence >= self.spoof_threshold)
        
        return (is_real, float(confidence))
    
    def check_frame(self, frame: np.ndarray) -> dict:
        """
        Check a video frame for spoof attacks.
        
        Args:
            frame: BGR image from camera
            
        Returns:
            Dict with keys: is_real, confidence, bbox, message
        """
        bbox = self.get_bbox(frame)
        
        if bbox is None:
            return {
                'is_real': False,
                'confidence': 0.0,
                'bbox': None,
                'message': 'No face detected'
            }
        
        is_real, confidence = self.is_real_face(frame, bbox)
        
        message = 'Real face' if is_real else 'Spoof detected!'
        
        return {
            'is_real': is_real,
            'confidence': confidence,
            'bbox': bbox,
            'message': message
        }
