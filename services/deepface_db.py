"""DeepFace-based face database and recognition utilities.

This module is adapted from Cong-Nghe-Xu-Ly-Anh/diemdanh_deepface_gui.py
but integrated for the main Flask application.
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    logger.warning("DeepFace không khả dụng trong môi trường hiện tại.")


def build_db_from_data_dir(
    data_dir: str = "data",
    model_name: str = "Facenet512",
    enforce_detection: bool = True,
) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    """Load all sample images from data_dir and compute embeddings.

    Ảnh mẫu nên đặt trong thư mục `data/` với tên dạng:
        <ID>_<Name>.jpg  (ví dụ: 1912345_NguyenVanA.jpg)

    Trả về:
        embeddings: np.ndarray shape (N, D)
        labels: list[(student_id, name)]
    """
    embeddings: List[np.ndarray] = []
    labels: List[Tuple[str, str]] = []

    data_path = Path(data_dir)
    if not data_path.exists():
        logger.warning("[DeepFaceDB] Thư mục %s không tồn tại.", data_dir)
        return np.array([]), []

    if not DEEPFACE_AVAILABLE:
        logger.error("[DeepFaceDB] DeepFace không khả dụng, không thể build DB.")
        return np.array([]), []

    image_files = list(data_path.glob("*.jpg")) + list(data_path.glob("*.jpeg")) + list(data_path.glob("*.png"))
    logger.info("[DeepFaceDB] Bắt đầu build DB từ %s, tìm thấy %d file ảnh.", data_dir, len(image_files))

    for img_path in image_files:
        try:
            filename = img_path.stem
            # Ưu tiên pattern: ID_Name...
            match = re.match(r"^(\d+)_([A-Za-zÀ-ỹA-Z\s_]+)", filename)
            if match:
                student_id = match.group(1)
                name = match.group(2).replace("_", " ").strip()
            else:
                # Fallback: tách bằng underscore, phần đầu là ID
                parts = filename.split("_")
                if len(parts) >= 2:
                    student_id = parts[0]
                    name = " ".join(parts[1:])
                else:
                    student_id = filename
                    name = filename

            logger.debug(
                "[DeepFaceDB] Đang tính embedding cho %s -> %s (ID=%s)",
                img_path.name,
                name,
                student_id,
            )

            rep = DeepFace.represent(
                img_path=str(img_path),
                model_name=model_name,
                enforce_detection=enforce_detection,
            )[0]["embedding"]

            embeddings.append(np.array(rep, dtype="float32"))
            labels.append((student_id, name))

            logger.info(
                "[DeepFaceDB] ✅ Đã thêm mẫu: %s (ID=%s) từ %s",
                name,
                student_id,
                img_path.name,
            )
        except Exception as e:
            logger.error(
                "[DeepFaceDB] ❌ Lỗi khi xử lý ảnh mẫu %s: %s",
                img_path.name,
                e,
                exc_info=True,
            )

    if not embeddings:
        logger.warning("[DeepFaceDB] Không có embedding nào được tạo từ thư mục %s", data_dir)
        return np.array([]), []

    emb_array = np.vstack(embeddings)
    logger.info(
        "[DeepFaceDB] Hoàn thành build DB: %d embeddings, %d nhãn (distinct IDs=%d)",
        emb_array.shape[0],
        len(labels),
        len(set(sid for sid, _ in labels)),
    )
    return emb_array, labels


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Tính cosine similarity giữa 2 vector."""
    from numpy.linalg import norm

    denom = norm(a) * norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def recognize_face(
    embedding: np.ndarray,
    db_embeddings: np.ndarray,
    db_labels: List[Tuple[str, str]],
    threshold: float = 0.6,
) -> Tuple[Optional[str], Optional[str], float]:
    """Nhận diện khuôn mặt bằng cosine similarity trên DB.

    Args:
        embedding: vector embedding của khuôn mặt cần nhận diện.
        db_embeddings: ma trận embeddings trong DB.
        db_labels: list[(student_id, name)] tương ứng với từng dòng trong db_embeddings.
        threshold: ngưỡng cosine similarity tối thiểu (ví dụ 0.6).

    Returns:
        (student_id, name, best_score) hoặc (None, None, 0.0) nếu không đạt ngưỡng.
    """
    if db_embeddings is None or len(db_embeddings) == 0:
        logger.warning("[DeepFaceDB] DB embeddings rỗng, không thể nhận diện.")
        return None, None, 0.0

    # Đảm bảo embedding là 1D
    emb = np.array(embedding, dtype="float32").ravel()

    sims = [cosine_similarity(emb, e) for e in db_embeddings]
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])

    student_id, name = db_labels[best_idx]

    logger.debug(
        "[DeepFaceDB] Kết quả so khớp tốt nhất: %s (%s) với sim=%.4f, threshold=%.3f",
        name,
        student_id,
        best_score,
        threshold,
    )

    if best_score >= threshold:
        logger.info(
            "[DeepFaceDB] ✅ Nhận diện: %s (ID=%s), sim=%.4f >= threshold=%.3f",
            name,
            student_id,
            best_score,
            threshold,
        )
        return student_id, name, best_score

    logger.info(
        "[DeepFaceDB] ❌ Từ chối: best sim=%.4f < threshold=%.3f (ID=%s, name=%s)",
        best_score,
        threshold,
        student_id,
        name,
    )
    return None, None, best_score
