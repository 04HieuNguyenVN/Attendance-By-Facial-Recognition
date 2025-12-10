"""
Dịch vụ nhận diện khuôn mặt dựa trên FaceNet.

Sử dụng FaceNet (Inception-ResNet) để tạo ra các embedding 128 chiều
và MTCNN để phát hiện/căn chỉnh khuôn mặt. Chính xác hơn dlib's
face_recognition nhưng yêu cầu TensorFlow.
"""

import os
import pickle
import shutil
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class FaceRecognitionService:
    """Nhận diện khuôn mặt nâng cao sử dụng FaceNet embeddings."""
    
    def __init__(self, 
                 facenet_model_path: str = 'face_attendance/Models/20180402-114759.pb',
                 classifier_path: str = 'data/models/facemodel.pkl',
                 embedding_size: int = 128,
                 image_size: int = 160,
                 confidence_threshold: float = 0.85):
        """
        Khởi tạo dịch vụ FaceNet.
        
        Args:
            facenet_model_path: Đường dẫn đến mô hình FaceNet .pb
            classifier_path: Đường dẫn đến bộ phân loại đã huấn luyện (SVM/softmax)
            embedding_size: Kích thước của face embeddings (mặc định 128)
            image_size: Kích thước đầu vào cho FaceNet (mặc định 160x160)
            confidence_threshold: Độ tin cậy tối thiểu để nhận diện
        """
        self.facenet_model_path = Path(facenet_model_path)
        self.classifier_path = Path(classifier_path)
        self.embedding_size = embedding_size
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold
        
        # Tải lười (Lazy loading)
        self.graph = None
        self.sess = None
        self.images_placeholder = None
        self.embeddings_tensor = None
        self.phase_train_placeholder = None
        self.classifier = None
        self.class_names = []
        
        # Bộ phát hiện MTCNN (tùy chọn - có thể dự phòng sang OpenCV)
        self.mtcnn_detector = None
        self._ensure_facenet_model_available()
        self._init_mtcnn()

    def _ensure_facenet_model_available(self):
        """Đảm bảo file mô hình FaceNet tồn tại, tự động phục hồi từ bản lưu nếu có."""
        if self.facenet_model_path.exists():
            return

        # Các vị trí dự phòng chứa mô hình (ví dụ thư mục archive)
        candidate_paths = [
            Path('archive/legacy/projects/face_attendance/Models/20180402-114759.pb'),
            Path('archive/legacy/projects/pipeline/Models/20180402-114759.pb'),
            Path('data/models/20180402-114759.pb'),
        ]

        for backup_path in candidate_paths:
            if not backup_path.exists():
                continue

            self.facenet_model_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(backup_path, self.facenet_model_path)
                logger.info(
                    "Đã tự động sao chép FaceNet model từ %s tới %s",
                    backup_path,
                    self.facenet_model_path,
                )
                return
            except Exception as copy_err:
                logger.warning("Không thể sao chép mô hình FaceNet từ %s: %s", backup_path, copy_err)
                continue

        logger.error(
            "Không tìm thấy file FaceNet (.pb) tại %s và không có bản sao dự phòng."
            " Sao chép 20180402-114759.pb vào vị trí này hoặc tắt USE_FACENET.",
            self.facenet_model_path,
        )
    
    def _init_mtcnn(self):
        """Khởi tạo bộ phát hiện khuôn mặt MTCNN nếu có sẵn."""
        try:
            from face_attendance.align import detect_face
            
            # Tải các mô hình MTCNN
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
        """Tải mô hình FaceNet và bộ phân loại đã huấn luyện."""
        if self.sess is not None:
            logger.info("Mô hình FaceNet đã được tải trước đó")
            return
        
        try:
            # Đảm bảo mô hình tồn tại trước khi tải
            self._ensure_facenet_model_available()
            if not self.facenet_model_path.exists():
                raise FileNotFoundError(
                    f"FaceNet model not found at {self.facenet_model_path}."
                )
            # Tải đồ thị FaceNet
            logger.info(f"Đang tải mô hình FaceNet từ {self.facenet_model_path}")
            
            self.graph = tf.Graph()
            with self.graph.as_default():
                with tf.io.gfile.GFile(str(self.facenet_model_path), 'rb') as f:
                    graph_def = tf.compat.v1.GraphDef()
                    graph_def.ParseFromString(f.read())
                    tf.import_graph_def(graph_def, name='')
                
                # Lấy các tensor đầu vào/đầu ra
                self.images_placeholder = self.graph.get_tensor_by_name("input:0")
                self.embeddings_tensor = self.graph.get_tensor_by_name("embeddings:0")
                self.phase_train_placeholder = self.graph.get_tensor_by_name("phase_train:0")
            
            # Tạo session
            gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
            self.sess = tf.compat.v1.Session(
                graph=self.graph,
                config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
            )
            
            logger.info("Mô hình FaceNet đã được tải thành công")
            
            # Tải bộ phân loại nếu tồn tại
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
        Phát hiện khuôn mặt trong ảnh sử dụng MTCNN hoặc OpenCV.
        
        Args:
            image: Mảng ảnh RGB
            
        Returns:
            Danh sách các bounding box (x, y, w, h)
        """
        if self.mtcnn_detector is not None:
            try:
                from face_attendance.align import detect_face
                
                # MTCNN yêu cầu RGB
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
        
        # Dự phòng sang OpenCV Haar cascade
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        return [(x, y, w, h) for (x, y, w, h) in faces]
    
    def preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """
        Tiền xử lý khuôn mặt cho FaceNet (thay đổi kích thước + làm trắng).
        
        Args:
            face_image: Vùng cắt khuôn mặt (RGB)
            
        Returns:
            Ảnh đã tiền xử lý sẵn sàng cho FaceNet
        """
        # Thay đổi kích thước về kích thước đầu vào FaceNet
        resized = cv2.resize(face_image, (self.image_size, self.image_size),
                           interpolation=cv2.INTER_CUBIC)
        
        # Làm trắng (mean=0, std=1)
        mean = np.mean(resized)
        std = np.std(resized)
        std_adj = np.maximum(std, 1.0 / np.sqrt(resized.size))
        whitened = (resized - mean) / std_adj
        
        return whitened
    
    def get_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Trích xuất embedding 128 chiều từ ảnh khuôn mặt.
        
        Args:
            face_image: Ảnh khuôn mặt đã tiền xử lý
            
        Returns:
            Vector embedding 128 chiều
        """
        if self.sess is None:
            self.load_model()
        
        # Thêm chiều batch
        face_batch = np.expand_dims(face_image, axis=0)
        
        # Chạy suy luận
        feed_dict = {
            self.images_placeholder: face_batch,
            self.phase_train_placeholder: False
        }
        embedding = self.sess.run(self.embeddings_tensor, feed_dict=feed_dict)
        
        return embedding[0]  # Trả về embedding đầu tiên (và duy nhất)
    
    def recognize_face(self, face_image: np.ndarray) -> Tuple[str, float]:
        """
        Nhận diện một khuôn mặt sử dụng bộ phân loại đã huấn luyện.
        
        Args:
            face_image: Ảnh khuôn mặt đã tiền xử lý
            
        Returns:
            Tuple (student_id, confidence)
        """
        if self.classifier is None:
            return ("UNKNOWN", 0.0)
        
        # Lấy embedding
        embedding = self.get_embedding(face_image)
        embedding = embedding.reshape(1, -1)
        
        # Dự đoán
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
        Xử lý một khung hình video và trả về tất cả các khuôn mặt đã phát hiện cùng với danh tính.
        
        Args:
            frame: Ảnh BGR từ camera
            
        Returns:
            Danh sách các dict với các key: bbox, student_id, confidence, embedding
        """
        # Chuyển đổi BGR sang RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Phát hiện khuôn mặt
        faces = self.detect_faces(rgb_frame)
        
        results = []
        for (x, y, w, h) in faces:
            # Trích xuất và tiền xử lý khuôn mặt
            face_crop = rgb_frame[y:y+h, x:x+w]
            if face_crop.size == 0:
                continue
            
            preprocessed = self.preprocess_face(face_crop)
            
            # Nhận diện
            student_id, confidence = self.recognize_face(preprocessed)
            
            # Lấy embedding để xác minh/huấn luyện
            embedding = self.get_embedding(preprocessed)
            
            results.append({
                'bbox': (x, y, w, h),
                'student_id': student_id,
                'confidence': confidence,
                'embedding': embedding
            })
        
        return results
    
    def close(self):
        """Giải phóng tài nguyên."""
        if self.sess is not None:
            self.sess.close()
            self.sess = None
        logger.info("Dịch vụ FaceNet đã đóng")
