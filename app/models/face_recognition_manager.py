"""
Face Recognition Manager - Quản lý nhận diện khuôn mặt
Encapsulates face recognition logic using inference engine and legacy methods
"""
import re
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from core.inference.engine import InferenceEngine, InferenceError


class FaceRecognitionManager:
    """Service quản lý nhận diện khuôn mặt"""
    
    def __init__(
        self,
        data_dir: Path,
        inference_engine: Optional[InferenceEngine] = None,
        deepface_module=None,
        deepface_available: bool = False,
        yolo_model=None,
        yolo_available: bool = False,
        similarity_threshold: float = 0.6,
        logger=None
    ):
        self.data_dir = Path(data_dir)
        self.inference_engine = inference_engine
        self.deepface_module = deepface_module
        self.deepface_available = deepface_available
        self.yolo_model = yolo_model
        self.yolo_available = yolo_available
        self.similarity_threshold = similarity_threshold
        self.logger = logger
        
        # Legacy face data
        self.known_face_embeddings = []
        self.known_face_ids = []
        self.known_face_names = []
        
        # Reserved subdirectories
        self.reserved_subdirs = {'training_samples', 'models', 'external_assets'}
    
    def load_known_faces(self, force_reload: bool = True) -> Optional[Dict]:
        """Tải các khuôn mặt đã biết"""
        if self.logger:
            self.logger.info(f"[FaceRecognition] 🔄 Loading faces from {self.data_dir}...")
        
        # Ưu tiên inference engine
        if self.inference_engine and self.inference_engine.has_strategies():
            return self._load_from_inference_engine(force_reload)
        
        # Fallback to legacy DeepFace
        if self.deepface_available:
            return self._load_legacy_deepface()
        
        if self.logger:
            self.logger.warning("[FaceRecognition] ⚠️ No recognition method available")
        return None
    
    def _load_from_inference_engine(self, force_reload: bool) -> Dict:
        """Load từ inference engine"""
        try:
            summary = (
                self.inference_engine.reload()
                if force_reload
                else self.inference_engine.warmup(force=False)
            )
            
            subjects = self.inference_engine.known_subjects(limit=10_000)
            
            self.known_face_embeddings = []
            self.known_face_ids = []
            self.known_face_names = []
            
            for student_id, name in subjects:
                normalized_id = (student_id or name or "UNKNOWN").strip()
                self.known_face_ids.append(normalized_id)
                self.known_face_names.append(name or normalized_id)
            
            if self.logger:
                self.logger.info(
                    f"[FaceRecognition] ✅ Loaded {len(self.known_face_ids)} faces via inference engine"
                )
            
            return summary
        
        except InferenceError as error:
            if self.logger:
                self.logger.warning(
                    f"[FaceRecognition] ⚠️ Inference engine failed: {error}"
                )
        except Exception as exc:
            if self.logger:
                self.logger.error(
                    f"[FaceRecognition] ⚠️ Error loading faces: {exc}",
                    exc_info=True
                )
        
        return None
    
    def _load_legacy_deepface(self) -> Dict:
        """Load từ DeepFace (legacy)"""
        if not self.deepface_available or not self.deepface_module:
            return None
        
        if self.logger:
            self.logger.info("[FaceRecognition] 🧠 Loading with DeepFace Facenet512...")
        
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
        
        db_embeddings = []
        db_labels = []
        processed_count = 0
        failed_count = 0
        
        image_files = self._iter_student_face_images()
        
        if self.logger:
            self.logger.info(f"[FaceRecognition] 📁 Found {len(image_files)} image files")
        
        for img_path in image_files:
            try:
                student_id, name = self._parse_image_filename(img_path)
                
                if self.logger:
                    self.logger.debug(
                        f"[FaceRecognition] Processing {img_path.name} -> {name} (ID: {student_id})"
                    )
                
                embedding = self.deepface_module.represent(
                    img_path=str(img_path),
                    model_name="Facenet512",
                    enforce_detection=True,
                )[0]["embedding"]
                
                db_embeddings.append(embedding)
                db_labels.append((student_id, name))
                processed_count += 1
                
                if self.logger:
                    self.logger.info(
                        f"[FaceRecognition] ✅ Loaded {name} (ID: {student_id})"
                    )
            
            except Exception as e:
                failed_count += 1
                if self.logger:
                    self.logger.error(
                        f"[FaceRecognition] ❌ Error processing {img_path.name}: {e}"
                    )
        
        if db_embeddings:
            self.known_face_embeddings = np.array(db_embeddings)
            self.known_face_ids = [sid for sid, _ in db_labels]
            self.known_face_names = [name for _, name in db_labels]
            
            if self.logger:
                self.logger.info(
                    f"[FaceRecognition] ✅ Loaded {len(self.known_face_embeddings)} faces"
                )
                self.logger.info(f"[FaceRecognition] 📊 Success: {processed_count}, Failed: {failed_count}")
        else:
            if self.logger:
                self.logger.warning("[FaceRecognition] ⚠️ No faces loaded!")
        
        return {
            'processed': processed_count,
            'failed': failed_count,
            'total': len(self.known_face_embeddings)
        }
    
    def _iter_student_face_images(self) -> List[Path]:
        """Duyệt qua tất cả ảnh mẫu sinh viên"""
        if not self.data_dir.exists():
            return []
        
        allowed_suffixes = {'.jpg', '.jpeg', '.png'}
        files = []
        
        for entry in self.data_dir.iterdir():
            if entry.is_file() and entry.suffix.lower() in allowed_suffixes:
                files.append(entry)
                continue
            
            if not entry.is_dir() or entry.name in self.reserved_subdirs:
                continue
            
            for sub_path in entry.rglob('*'):
                if sub_path.is_file() and sub_path.suffix.lower() in allowed_suffixes:
                    files.append(sub_path)
        
        return files
    
    def _parse_image_filename(self, img_path: Path) -> Tuple[str, str]:
        """Parse filename để lấy student_id và name"""
        filename = img_path.stem
        student_id = None
        name = None
        
        # Try to get from directory structure
        try:
            relative_parts = img_path.relative_to(self.data_dir).parts
        except ValueError:
            relative_parts = ()
        
        if len(relative_parts) > 1 and relative_parts[0] not in self.reserved_subdirs:
            student_id = relative_parts[0]
        
        # Parse filename pattern: ID_Name
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
        
        return student_id, name
    
    def recognize_face(self, face_img) -> Dict[str, Any]:
        """Nhận diện khuôn mặt"""
        result = {
            'student_id': 'UNKNOWN',
            'student_name': 'UNKNOWN',
            'confidence': 0.0,
            'strategy': 'none',
            'status': 'unknown',
        }
        
        # Try inference engine first
        if self.inference_engine and self.inference_engine.has_strategies():
            result = self._recognize_via_inference_engine(face_img)
            if result['student_id'] != 'UNKNOWN':
                return result
        
        # Fallback to legacy DeepFace
        if self.deepface_available and len(self.known_face_embeddings) > 0:
            result = self._recognize_via_legacy_deepface(face_img)
        
        return result
    
    def _recognize_via_inference_engine(self, face_img) -> Dict[str, Any]:
        """Nhận diện qua inference engine"""
        try:
            inference_result = self.inference_engine.identify(face_img)
            
            sid = inference_result.student_id or 'UNKNOWN'
            name = inference_result.student_name or (sid if sid != 'UNKNOWN' else 'UNKNOWN')
            
            return {
                'student_id': sid,
                'student_name': name,
                'confidence': float(inference_result.confidence or 0.0),
                'strategy': inference_result.strategy or 'inference',
                'status': inference_result.status or ('match' if sid != 'UNKNOWN' else 'no_match'),
            }
        
        except InferenceError as error:
            if self.logger:
                self.logger.warning(f"[FaceRecognition] Inference failed: {error}")
        except Exception as exc:
            if self.logger:
                self.logger.error(f"[FaceRecognition] Error: {exc}", exc_info=True)
        
        return {
            'student_id': 'UNKNOWN',
            'student_name': 'UNKNOWN',
            'confidence': 0.0,
            'strategy': 'none',
            'status': 'error',
        }
    
    def _recognize_via_legacy_deepface(self, face_img) -> Dict[str, Any]:
        """Nhận diện qua DeepFace legacy"""
        try:
            from services.deepface_db import recognize_face as deepface_recognize
            
            legacy_embedding = self.deepface_module.represent(
                face_img,
                model_name="Facenet512",
                enforce_detection=False,
            )[0]["embedding"]
            
            db_labels = list(zip(self.known_face_ids, self.known_face_names))
            student_id, student_name, best_score = deepface_recognize(
                legacy_embedding,
                self.known_face_embeddings,
                db_labels,
                threshold=self.similarity_threshold,
            )
            
            sid = student_id or 'UNKNOWN'
            name = student_name or (sid if sid != 'UNKNOWN' else 'UNKNOWN')
            
            return {
                'student_id': sid,
                'student_name': name,
                'confidence': float(best_score or 0.0),
                'strategy': 'legacy-deepface',
                'status': 'match' if student_id else 'no_match',
            }
        
        except Exception as exc:
            if self.logger:
                self.logger.error(f"[FaceRecognition] Legacy error: {exc}", exc_info=True)
            
            return {
                'student_id': 'UNKNOWN',
                'student_name': 'UNKNOWN',
                'confidence': 0.0,
                'strategy': 'legacy-deepface',
                'status': 'error',
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Lấy thống kê"""
        return {
            'known_faces': len(self.known_face_ids),
            'known_ids': self.known_face_ids[:10] if self.known_face_ids else [],
            'known_names': self.known_face_names[:10] if self.known_face_names else [],
            'has_inference_engine': self.inference_engine is not None,
            'deepface_available': self.deepface_available,
            'yolo_available': self.yolo_available,
        }
