"""Reusable AI inference engine abstractions.

This module centralizes DeepFace/FaceNet/fallback recognition logic so the Flask
app can treat AI inference as a stateful service instead of scattering global
variables across routes.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # Optional dependency for FaceNet strategy
    import cv2
except ImportError:  # pragma: no cover - OpenCV always available in runtime
    cv2 = None  # type: ignore

EmbeddingArray = Optional[np.ndarray]
LabelList = List[Tuple[str, str]]


class InferenceError(RuntimeError):
    """Raised when an inference strategy cannot complete."""


@dataclass
class InferenceResult:
    student_id: Optional[str]
    student_name: Optional[str]
    confidence: float = 0.0
    embedding: Optional[np.ndarray] = None
    strategy: str = ""
    status: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


class EmbeddingStore:
    """Thread-safe holder for embeddings and labels."""

    def __init__(self) -> None:
        self._embeddings: EmbeddingArray = None
        self._labels: LabelList = []
        self._lock = threading.RLock()
        self._version = 0
        self._last_loaded: Optional[datetime] = None

    def ready(self) -> bool:
        with self._lock:
            return bool(self._embeddings is not None and self._labels)

    def count(self) -> int:
        with self._lock:
            return len(self._labels)

    def snapshot(self) -> Tuple[EmbeddingArray, LabelList]:
        with self._lock:
            embeddings = self._embeddings.copy() if isinstance(self._embeddings, np.ndarray) else self._embeddings
            labels = list(self._labels)
        return embeddings, labels

    def update(self, embeddings: EmbeddingArray, labels: Sequence[Tuple[str, str]]) -> None:
        with self._lock:
            self._embeddings = embeddings
            self._labels = list(labels)
            self._version += 1
            self._last_loaded = datetime.utcnow()

    def describe(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "count": len(self._labels),
                "version": self._version,
                "last_loaded": self._last_loaded.isoformat() if self._last_loaded else None,
            }


class RecognitionStrategy:
    """Protocol-ish base class for duck-typed strategies."""

    name: str = "strategy"

    def warmup(self, force: bool = False) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def identify(self, face_image: np.ndarray) -> InferenceResult:  # pragma: no cover - interface
        raise NotImplementedError

    def known_subjects(self) -> LabelList:
        return []

    def is_ready(self) -> bool:
        return True


class DeepFaceStrategy(RecognitionStrategy):
    name = "deepface"

    def __init__(
        self,
        *,
        data_dir: Path,
        deepface_module: Any,
        build_db_fn: Callable[[str], Tuple[np.ndarray, LabelList]],
        recognize_fn: Callable[[np.ndarray, np.ndarray, LabelList, float], Tuple[Optional[str], Optional[str], float]],
        similarity_threshold: float,
        model_name: str = "Facenet512",
        enforce_detection: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._deepface = deepface_module
        self._build_db = build_db_fn
        self._recognize = recognize_fn
        self._threshold = similarity_threshold
        self._model_name = model_name
        self._enforce_detection = enforce_detection
        self._store = EmbeddingStore()
        self._lock = threading.RLock()
        self._logger = logger or logging.getLogger(__name__)

    def warmup(self, force: bool = False) -> None:
        if self._deepface is None or self._build_db is None:
            raise InferenceError("DeepFace dependencies are missing")
        if not force and self._store.ready():
            return
        embeddings, labels = self._build_db(str(self._data_dir))
        if embeddings is None or len(embeddings) == 0:
            raise InferenceError("No DeepFace embeddings available")
        self._store.update(np.array(embeddings, dtype="float32"), labels)
        self._logger.info(
            "[Inference] DeepFace store ready: %d embeddings", self._store.count()
        )

    def identify(self, face_image: np.ndarray) -> InferenceResult:
        if not self._store.ready():
            self.warmup()
        embeddings, labels = self._store.snapshot()
        if embeddings is None or not labels:
            raise InferenceError("DeepFace store is empty")
        try:
            representation = self._deepface.represent(  # type: ignore[attr-defined]
                face_image,
                model_name=self._model_name,
                enforce_detection=self._enforce_detection,
            )[0]["embedding"]
        except Exception as exc:  # pragma: no cover - DeepFace errors runtime dependent
            raise InferenceError(f"DeepFace failed to create embedding: {exc}") from exc
        embedding_vec = np.array(representation, dtype="float32")
        student_id, student_name, score = self._recognize(
            embedding_vec,
            embeddings,
            labels,
            threshold=self._threshold,
        )
        status = "match" if student_id else "no_match"
        return InferenceResult(
            student_id=student_id,
            student_name=student_name,
            confidence=float(score or 0.0),
            embedding=embedding_vec,
            strategy=self.name,
            status=status,
        )

    def known_subjects(self) -> LabelList:
        _, labels = self._store.snapshot()
        return labels

    def is_ready(self) -> bool:
        return self._store.ready()


class FaceNetStrategy(RecognitionStrategy):
    name = "facenet"

    def __init__(
        self,
        *,
        service: Any,
        label_lookup: Optional[Callable[[str], Optional[str]]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._service = service
        self._label_lookup = label_lookup
        self._logger = logger or logging.getLogger(__name__)

    def warmup(self, force: bool = False) -> None:
        if self._service is None:
            raise InferenceError("FaceNet service missing")
        if hasattr(self._service, "load_model"):
            self._service.load_model()

    def identify(self, face_image: np.ndarray) -> InferenceResult:
        if self._service is None:
            raise InferenceError("FaceNet service missing")
        if cv2 is None:
            raise InferenceError("OpenCV not available for FaceNet preprocessing")
        self.warmup()
        rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        preprocessed = self._service.preprocess_face(rgb)
        student_id, confidence = self._service.recognize_face(preprocessed)
        embedding = self._service.get_embedding(preprocessed)
        status = "match" if student_id and student_id != "UNKNOWN" else "no_match"
        name = None
        if status == "match" and self._label_lookup:
            try:
                name = self._label_lookup(student_id)
            except Exception as exc:  # pragma: no cover - DB errors depend on runtime
                self._logger.debug("Label lookup failed for %s: %s", student_id, exc)
        return InferenceResult(
            student_id=student_id if status == "match" else None,
            student_name=name,
            confidence=float(confidence or 0.0),
            embedding=embedding,
            strategy=self.name,
            status=status,
        )

    def known_subjects(self) -> LabelList:
        class_names = getattr(self._service, "class_names", []) or []
        return [(name, name) for name in class_names]


class InferenceEngine:
    """High-level orchestrator picking the first successful strategy."""

    def __init__(self, *, logger: Optional[logging.Logger] = None, demo_mode: bool = False) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._strategies: List[RecognitionStrategy] = []
        self._demo_mode = demo_mode
        self._demo_subjects: LabelList = [("DEMO", "Demo User")]
        self._lock = threading.RLock()

    def add_strategy(self, strategy: Optional[RecognitionStrategy]) -> None:
        if strategy is None:
            return
        self._logger.info("[Inference] Added strategy %s", strategy.name)
        self._strategies.append(strategy)

    def ready(self) -> bool:
        return any(getattr(strategy, "is_ready", lambda: True)() for strategy in self._strategies)

    def has_strategies(self) -> bool:
        return bool(self._strategies)

    def identify(self, face_image: np.ndarray) -> InferenceResult:
        last_error: Optional[Exception] = None
        for strategy in self._strategies:
            try:
                result = strategy.identify(face_image)
                if result:
                    result.strategy = result.strategy or strategy.name
                    return result
            except InferenceError as exc:
                last_error = exc
                self._logger.debug("Strategy %s failed: %s", strategy.name, exc)
            except Exception as exc:  # pragma: no cover - depends on runtime libs
                last_error = exc
                self._logger.exception("Strategy %s crashed", strategy.name)
        if self._demo_mode:
            sid, name = self._demo_subjects[0]
            return InferenceResult(
                student_id=sid,
                student_name=name,
                confidence=0.0,
                strategy="demo",
                status="demo",
            )
        raise InferenceError(str(last_error) if last_error else "No inference strategy succeeded")

    def warmup(self, force: bool = False) -> None:
        for strategy in self._strategies:
            try:
                strategy.warmup(force=force)
            except InferenceError as exc:
                self._logger.warning("Strategy %s warmup error: %s", strategy.name, exc)
            except Exception as exc:  # pragma: no cover - depends on runtime libs
                self._logger.exception("Strategy %s warmup crashed", strategy.name)

    def reload(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {"strategies": []}
        for strategy in self._strategies:
            try:
                strategy.warmup(force=True)
                summary["strategies"].append({
                    "name": strategy.name,
                    "ready": getattr(strategy, "is_ready", lambda: True)(),
                    "subjects": len(strategy.known_subjects()),
                })
            except Exception as exc:  # pragma: no cover - runtime specific
                summary["strategies"].append({
                    "name": strategy.name,
                    "ready": False,
                    "error": str(exc),
                })
        return summary

    def known_subjects(self, limit: int = 50) -> LabelList:
        subjects: LabelList = []
        for strategy in self._strategies:
            try:
                subjects.extend(strategy.known_subjects())
            except Exception:
                continue
        if not subjects and self._demo_mode:
            return self._demo_subjects[:limit]
        return subjects[:limit]

    def subject_count(self) -> int:
        return len(self.known_subjects(limit=10_000))

    def set_demo_subjects(self, subjects: LabelList) -> None:
        if subjects:
            self._demo_subjects = subjects


__all__ = [
    "InferenceEngine",
    "InferenceError",
    "InferenceResult",
    "DeepFaceStrategy",
    "FaceNetStrategy",
]
