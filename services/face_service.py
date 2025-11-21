"""
FaceNet-based face recognition service.

Uses FaceNet (Inception-ResNet) to generate 128-dimensional embeddings
and MTCNN for face detection/alignment. More accurate than dlib's
face_recognition but requires TensorFlow.
"""

import os
import pickle
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class FaceRecognitionService:
    """Advanced face recognition using FaceNet embeddings."""
    
    def __init__(self, 
                 facenet_model_path: str = 'face_attendance/Models/20180402-114759.pb',
                 classifier_path: str = 'data/models/facemodel.pkl',
                 embedding_size: int = 128,
                 image_size: int = 160,
                 confidence_threshold: float = 0.85):
        """
        Initialize FaceNet service.
        
        Args:
            facenet_model_path: Path to FaceNet .pb model
            classifier_path: Path to trained classifier (SVM/softmax)
            embedding_size: Size of face embeddings (default 128)
            image_size: Input size for FaceNet (default 160x160)
            confidence_threshold: Minimum confidence for recognition
        """
        self.facenet_model_path = Path(facenet_model_path)
        self.classifier_path = Path(classifier_path)
        self.embedding_size = embedding_size
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold
        
        # Lazy loading
        self.graph = None
        self.sess = None
        self.images_placeholder = None
        self.embeddings_tensor = None
        self.phase_train_placeholder = None
        self.classifier = None
        self.class_names = []
        
        # MTCNN detector (optional - can fallback to OpenCV)
        self.mtcnn_detector = None
        self._init_mtcnn()
    
    def _init_mtcnn(self):
        """Initialize MTCNN face detector if available."""
        try:
            from face_attendance.align import detect_face
            
            # Load MTCNN models
            mtcnn_path = Path('face_attendance/align')
            if not all((mtcnn_path / f'det{i}.npy').exists() for i in [1, 2, 3]):
                logger.warning("Không tìm thấy mô hình MTCNN, sẽ dùng bộ phát hiện OpenCV thay thế")
                return
            
            with tf.Graph().as_default():
                gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
                sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
                    gpu_options=gpu_options, log_device_placement=False))
                with sess.as_default():
                    self.mtcnn_pnet, self.mtcnn_rnet, self.mtcnn_onet = detect_face.create_mtcnn(
                        sess, str(mtcnn_path))
            
            logger.info("Bộ phát hiện MTCNN đã khởi tạo")
        except Exception as e:
            logger.warning(f"Không thể khởi tạo MTCNN: {e}. Sẽ dùng cascade của OpenCV.")
            self.mtcnn_detector = None
    
    def load_model(self):
        """Load FaceNet model and trained classifier."""
        if self.sess is not None:
            logger.info("Mô hình FaceNet đã được tải trước đó")
            return
        
        try:
            # Load FaceNet graph
            logger.info(f"Đang tải mô hình FaceNet từ {self.facenet_model_path}")
            
            self.graph = tf.Graph()
            with self.graph.as_default():
                with tf.io.gfile.GFile(str(self.facenet_model_path), 'rb') as f:
                    graph_def = tf.compat.v1.GraphDef()
                    graph_def.ParseFromString(f.read())
                    tf.import_graph_def(graph_def, name='')
                
                # Get input/output tensors
                self.images_placeholder = self.graph.get_tensor_by_name("input:0")
                self.embeddings_tensor = self.graph.get_tensor_by_name("embeddings:0")
                self.phase_train_placeholder = self.graph.get_tensor_by_name("phase_train:0")
            
            # Create session
            gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
            self.sess = tf.compat.v1.Session(
                graph=self.graph,
                config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
            )
            
            logger.info("Mô hình FaceNet đã được tải thành công")
            
            # Load classifier if exists
            if self.classifier_path.exists():
                with open(self.classifier_path, 'rb') as f:
                    self.classifier, self.class_names = pickle.load(f)
                logger.info(f"Đã tải bộ phân loại với {len(self.class_names)} lớp")
            else:
                logger.warning(f"Không tìm thấy bộ phân loại tại {self.classifier_path}")
                self.classifier = None
                self.class_names = []
        
        except Exception as e:
            logger.error(f"Tải mô hình FaceNet thất bại: {e}")
            raise
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in image using MTCNN or OpenCV.
        
        Args:
            image: RGB image array
            
        Returns:
            List of bounding boxes (x, y, w, h)
        """
        if self.mtcnn_detector is not None:
            try:
                from face_attendance.align import detect_face
                
                # MTCNN expects RGB
                minsize = 20
                threshold = [0.6, 0.7, 0.7]
                factor = 0.709
                
                bboxes, _ = detect_face.detect_face(
                    image, minsize, self.mtcnn_pnet, self.mtcnn_rnet, 
                    self.mtcnn_onet, threshold, factor)
                
                faces = []
                for bbox in bboxes:
                    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
                    faces.append((x1, y1, x2 - x1, y2 - y1))
                
                return faces
            except Exception as e:
                logger.warning(f"Phát hiện bằng MTCNN thất bại: {e}, chuyển sang OpenCV")
        
        # Fallback to OpenCV Haar cascade
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        return [(x, y, w, h) for (x, y, w, h) in faces]
    
    def preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess face for FaceNet (resize + whitening).
        
        Args:
            face_image: Face crop (RGB)
            
        Returns:
            Preprocessed image ready for FaceNet
        """
        # Resize to FaceNet input size
        resized = cv2.resize(face_image, (self.image_size, self.image_size),
                           interpolation=cv2.INTER_CUBIC)
        
        # Whitening (mean=0, std=1)
        mean = np.mean(resized)
        std = np.std(resized)
        std_adj = np.maximum(std, 1.0 / np.sqrt(resized.size))
        whitened = (resized - mean) / std_adj
        
        return whitened
    
    def get_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract 128-dim embedding from face image.
        
        Args:
            face_image: Preprocessed face image
            
        Returns:
            128-dimensional embedding vector
        """
        if self.sess is None:
            self.load_model()
        
        # Add batch dimension
        face_batch = np.expand_dims(face_image, axis=0)
        
        # Run inference
        feed_dict = {
            self.images_placeholder: face_batch,
            self.phase_train_placeholder: False
        }
        embedding = self.sess.run(self.embeddings_tensor, feed_dict=feed_dict)
        
        return embedding[0]  # Return first (and only) embedding
    
    def recognize_face(self, face_image: np.ndarray) -> Tuple[str, float]:
        """
        Recognize a face using trained classifier.
        
        Args:
            face_image: Preprocessed face image
            
        Returns:
            (student_id, confidence) tuple
        """
        if self.classifier is None:
            return ("UNKNOWN", 0.0)
        
        # Get embedding
        embedding = self.get_embedding(face_image)
        embedding = embedding.reshape(1, -1)
        
        # Predict
        predictions = self.classifier.predict_proba(embedding)
        best_class_idx = np.argmax(predictions, axis=1)[0]
        best_prob = predictions[0, best_class_idx]
        
        if best_prob >= self.confidence_threshold:
            student_id = self.class_names[best_class_idx]
            return (student_id, float(best_prob))
        else:
            return ("UNKNOWN", float(best_prob))
    
    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Process a video frame and return all detected faces with identities.
        
        Args:
            frame: BGR image from camera
            
        Returns:
            List of dicts with keys: bbox, student_id, confidence, embedding
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = self.detect_faces(rgb_frame)
        
        results = []
        for (x, y, w, h) in faces:
            # Extract and preprocess face
            face_crop = rgb_frame[y:y+h, x:x+w]
            if face_crop.size == 0:
                continue
            
            preprocessed = self.preprocess_face(face_crop)
            
            # Recognize
            student_id, confidence = self.recognize_face(preprocessed)
            
            # Get embedding for verification/training
            embedding = self.get_embedding(preprocessed)
            
            results.append({
                'bbox': (x, y, w, h),
                'student_id': student_id,
                'confidence': confidence,
                'embedding': embedding
            })
        
        return results
    
    def close(self):
        """Release resources."""
        if self.sess is not None:
            self.sess.close()
            self.sess = None
        logger.info("Dịch vụ FaceNet đã đóng")
