"""
Dịch vụ huấn luyện để đăng ký sinh viên mới và cập nhật bộ phân loại.

Chức năng:
- Thu thập mẫu khuôn mặt từ webcam
- Tạo embedding bằng FaceNet
- Huấn luyện/cập nhật bộ phân loại SVM
- Lưu mô hình đã huấn luyện
"""

import os
import pickle
import numpy as np
from pathlib import Path
from typing import List, Tuple
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)


class TrainingService:
    """Dịch vụ huấn luyện bộ phân loại nhận diện khuôn mặt."""
    
    def __init__(self,
                 face_service,
                 data_dir: str = 'data',
                 models_dir: str = 'data/models',
                 min_samples_per_person: int = 10):
        """
        Khởi tạo dịch vụ huấn luyện.

        Tham số:
            face_service: thể hiện của FaceRecognitionService
            data_dir: thư mục chứa ảnh dữ liệu
            models_dir: thư mục để lưu mô hình đã huấn luyện
            min_samples_per_person: số mẫu tối thiểu để huấn luyện
        """
        self.face_service = face_service
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.min_samples_per_person = min_samples_per_person
        
        # Đảm bảo các thư mục tồn tại
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_samples(self, student_id: str, student_name: str, 
                       num_samples: int = 20) -> List[np.ndarray]:
        """
        Thu thập mẫu khuôn mặt từ webcam để huấn luyện.

        Đây là hàm đồng bộ, nên được gọi từ một route trả về frame
        (để hiển thị xem trước thời gian thực và thanh tiến trình).

        Tham số:
            student_id: mã sinh viên
            student_name: họ tên sinh viên
            num_samples: số mẫu cần thu thập
        Trả về:
            Danh sách embedding của khuôn mặt
        """
        import cv2
        
        embeddings = []
        samples_dir = self.data_dir / 'training_samples' / student_id
        samples_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(0)
        frame_count = 0
        sample_count = 0
        
        logger.info(f"Đang thu thập {num_samples} mẫu cho {student_name} ({student_id})")
        
        try:
            while sample_count < num_samples:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Xử lý mỗi khung thứ 5 để người dùng có thời gian chỉnh tư thế
                frame_count += 1
                if frame_count % 5 != 0:
                    continue
                
                # Phát hiện và nhận diện
                results = self.face_service.process_frame(frame)
                
                if len(results) == 0:
                    continue
                
                # Lấy khuôn mặt đầu tiên được phát hiện
                face_data = results[0]
                bbox = face_data['bbox']
                embedding = face_data['embedding']
                
                # Lưu ảnh mẫu
                x, y, w, h = bbox
                face_crop = frame[y:y+h, x:x+w]
                sample_path = samples_dir / f'sample_{sample_count:03d}.jpg'
                cv2.imwrite(str(sample_path), face_crop)
                
                # Lưu embedding
                embeddings.append(embedding)
                sample_count += 1
                
                logger.info(f"Đã thu mẫu {sample_count}/{num_samples}")
        
        finally:
            cap.release()
        
        logger.info(f"Đã thu {len(embeddings)} mẫu cho {student_id}")
        return embeddings
    
    def load_all_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tải tất cả embedding khuôn mặt từ thư mục dữ liệu.

        Kỳ vọng cấu trúc: data/<student_id>_<student_name>.jpg
        Hoặc: data/training_samples/<student_id>/sample_*.jpg

        Trả về:
            (mảng_embeddings, mảng_nhãn)
        """
        embeddings_list = []
        labels_list = []
        
        # Tải từ các ảnh mặt cá nhân trong thư mục data
        for img_path in self.data_dir.glob('*.jpg'):
            if img_path.stem == '.gitkeep':
                continue
            
            try:
                # Phân tích mã sinh viên từ tên file
                parts = img_path.stem.split('_')
                if len(parts) >= 2:
                    student_id = parts[0]
                else:
                    student_id = img_path.stem
                
                # Đọc ảnh và lấy embedding
                import cv2
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # Phát hiện khuôn mặt
                results = self.face_service.process_frame(img)
                if len(results) == 0:
                    logger.warning(f"Không tìm thấy khuôn mặt trong {img_path.name}")
                    continue
                
                embedding = results[0]['embedding']
                embeddings_list.append(embedding)
                labels_list.append(student_id)
                
                logger.debug(f"Đã tải embedding cho {student_id} từ {img_path.name}")
            
            except Exception as e:
                logger.error(f"Lỗi khi tải {img_path.name}: {e}")
                continue
        
        # Tải từ các thư mục training_samples
        samples_dir = self.data_dir / 'training_samples'
        if samples_dir.exists():
            for student_dir in samples_dir.iterdir():
                if not student_dir.is_dir():
                    continue
                
                student_id = student_dir.name
                
                for sample_path in student_dir.glob('*.jpg'):
                    try:
                        import cv2
                        img = cv2.imread(str(sample_path))
                        if img is None:
                            continue
                        
                        results = self.face_service.process_frame(img)
                        if len(results) == 0:
                            continue
                        
                        embedding = results[0]['embedding']
                        embeddings_list.append(embedding)
                        labels_list.append(student_id)
                    
                    except Exception as e:
                        logger.error(f"Lỗi khi tải {sample_path}: {e}")
                        continue
        
        if len(embeddings_list) == 0:
            logger.warning("Không có embedding nào được tải lên!")
            return np.array([]), np.array([])
        
        embeddings = np.array(embeddings_list)
        labels = np.array(labels_list)
        
        logger.info(f"Đã tải {len(embeddings)} embedding cho {len(np.unique(labels))} sinh viên")
        
        return embeddings, labels
    
    def train_classifier(self, kernel: str = 'linear', 
                        probability: bool = True) -> bool:
        """
        Huấn luyện bộ phân loại SVM trên embedding đã thu thập.

        Tham số:
            kernel: kernel của SVM ('linear', 'rbf', 'poly')
            probability: bật ước lượng xác suất

        Trả về:
            True nếu huấn luyện thành công
        """
        logger.info("Bắt đầu huấn luyện bộ phân loại...")
        
        # Tải embedding
        embeddings, labels = self.load_all_embeddings()
        
        if len(embeddings) == 0:
            logger.error("Không tìm thấy dữ liệu huấn luyện!")
            return False
        
        # Kiểm tra số mẫu tối thiểu
        unique_labels, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            if count < self.min_samples_per_person:
                logger.warning(f"Student {label} has only {count} samples "
                             f"(minimum: {self.min_samples_per_person})")
        
        # Mã hóa nhãn
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        
        # Huấn luyện SVM
        logger.info(f"Huấn luyện SVM với {len(embeddings)} mẫu, {len(unique_labels)} lớp")
        
        classifier = SVC(kernel=kernel, probability=probability, gamma='scale')
        classifier.fit(embeddings, encoded_labels)
        
        # Lưu mô hình
        model_path = self.models_dir / 'facemodel.pkl'
        class_names = label_encoder.classes_.tolist()
        
        with open(model_path, 'wb') as f:
            pickle.dump((classifier, class_names), f)
        
        logger.info(f"Đã lưu bộ phân loại vào {model_path}")
        logger.info(f"Các lớp: {class_names}")
        
        # Cập nhật bộ phân loại trong face_service
        self.face_service.classifier = classifier
        self.face_service.class_names = class_names
        self.face_service.classifier_path = model_path
        
        return True
    
    def add_student_samples(self, student_id: str, embeddings: List[np.ndarray]) -> bool:
        """
        Thêm mẫu cho sinh viên mới và huấn luyện lại bộ phân loại.

        Tham số:
            student_id: mã sinh viên
            embeddings: danh sách embedding khuôn mặt

        Trả về:
            True nếu thành công
        """
        if len(embeddings) < self.min_samples_per_person:
            logger.warning(f"Chỉ có {len(embeddings)} mẫu được cung cấp, "
                         f"tối thiểu yêu cầu là {self.min_samples_per_person}")
            return False
        
        # Lưu embedding ra đĩa
        samples_dir = self.data_dir / 'training_samples' / student_id
        samples_dir.mkdir(parents=True, exist_ok=True)
        
        embeddings_file = samples_dir / 'embeddings.npy'
        np.save(embeddings_file, np.array(embeddings))
        
        logger.info(f"Đã lưu {len(embeddings)} embedding cho {student_id}")
        
        # Huấn luyện lại bộ phân loại
        success = self.train_classifier()
        
        return success
    
    def remove_student(self, student_id: str) -> bool:
        """
        Xóa sinh viên khỏi dữ liệu huấn luyện và huấn luyện lại.

        Tham số:
            student_id: mã sinh viên cần xóa

        Trả về:
            True nếu thành công
        """
        # Xóa thư mục mẫu huấn luyện
        samples_dir = self.data_dir / 'training_samples' / student_id
        if samples_dir.exists():
            import shutil
            shutil.rmtree(samples_dir)
            logger.info(f"Đã xóa mẫu huấn luyện cho {student_id}")
        
        # Xóa các ảnh cá nhân tương ứng
        for img_path in self.data_dir.glob(f'{student_id}_*.jpg'):
            img_path.unlink()
            logger.info(f"Đã xóa {img_path.name}")
        
        # Huấn luyện lại
        success = self.train_classifier()
        
        return success
    
    def get_training_stats(self) -> dict:
        """
        Lấy thống kê về dữ liệu huấn luyện.

        Trả về:
            Dict chứa số liệu thống kê huấn luyện
        """
        embeddings, labels = self.load_all_embeddings()
        
        if len(embeddings) == 0:
            return {
                'total_samples': 0,
                'num_students': 0,
                'students': []
            }
        
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        students = []
        for label, count in zip(unique_labels, counts):
            students.append({
                'student_id': label,
                'sample_count': int(count),
                'ready': count >= self.min_samples_per_person
            })
        
        return {
            'total_samples': len(embeddings),
            'num_students': len(unique_labels),
            'students': students,
            'min_samples_required': self.min_samples_per_person
        }
