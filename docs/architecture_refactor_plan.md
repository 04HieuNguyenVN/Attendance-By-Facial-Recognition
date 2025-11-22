# Vision & AI Refactor Roadmap

This document captures the target architecture for separating **image/vision handling**, **AI inference**, and **attendance workflow state**. The plan aligns with the five implementation phases requested by the user.

---

## 1. Pipeline Overview

```
Camera Input -> Vision Pipeline -> AI Inference Engine -> Attendance State Manager
                           |                         |
                      (preprocessing)           (embeddings, models)
```

### Core principles

1. **Single-responsibility states**

   - Every long-lived component keeps its own state object (camera device handles, frame buffers, loaded models, attendance context).
   - Flask routes request work from these states instead of storing globals.

2. **Pluggable boundaries**

   - Vision layer produces normalized `VisionFrame` objects (RGB array + metadata).
   - AI layer consumes `VisionFrame` and returns `InferenceResult` (identity, confidence, embeddings, explanations).
   - Attendance workflow consumes `InferenceResult` plus HTTP context to update check-in/out state machines.

3. **Thread-safe resource managers**
   - Each state exposes `start()`, `stop()`, `acquire()` patterns with locks.
   - Background tasks (camera loop, SSE broadcast) subscribe through event emitters.

---

## 2. Proposed Modules

| Module                               | Responsibility                                                                               | Key Types / Interfaces                                         |
| ------------------------------------ | -------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| `core/vision/camera_manager.py`      | Owns camera devices, frame capture, health checks.                                           | `CameraManager`, `CameraState`                                 |
| `core/vision/pipeline.py`            | Applies preprocessing (flip, resize, denoise, face detection ROI) and yields `VisionFrame`.  | `VisionPipeline`, `VisionFrame`                                |
| `core/inference/engine.py`           | Loads DeepFace/FaceNet/legacy recognizers, caches embeddings, dispatches inference requests. | `InferenceEngine`, `InferenceResult`, `EmbeddingStore`         |
| `core/workflows/attendance_state.py` | Tracks sessions, student progress (5-second confirm), duplicates, SSE events.                | `AttendanceStateManager`, `AttendanceEvent`, `ProgressTracker` |
| `core/events/bus.py`                 | Minimal pub/sub so layers can signal without tight coupling.                                 | `EventBus`, `EventListener`                                    |

Existing files to be refactored to use the modules above:

- `app.py` (routes become thin controllers).
- `services/face_service.py`, `services/training_service.py`, `services/antispoof_service.py` (logic moves/rewrapped).
- `win_console*.py`, `tools/train_classifier.py` (import new modules instead of duplicating logic).

---

## 3. State Definitions (Draft)

### `VisionFrame`

```python
class VisionFrame(TypedDict):
    frame_id: str
    timestamp: datetime
    rgb: np.ndarray  # normalized, post-processed
    bgr: np.ndarray  # original for legacy consumers
    detections: list[FaceDetection]
    metadata: dict[str, Any]
```

### `InferenceResult`

```python
class InferenceResult(TypedDict):
    student_id: str | None
    confidence: float
    embedding: np.ndarray | None
    detector_score: float
    explanation: dict[str, Any]
```

### `AttendanceState`

```python
@dataclass
class AttendanceState:
    active_session: Optional[SessionInfo]
    progress: dict[str, ProgressWindow]
    last_events: deque[AttendanceEvent]
```

Each state object exposes methods to mutate/query data while encapsulating locks and persistence hooks.

---

## 4. Phase Breakdown

| Phase             | Scope                  | Deliverables                                                                                        |
| ----------------- | ---------------------- | --------------------------------------------------------------------------------------------------- |
| 1 (this document) | Architecture plan      | **Done** – shared here.                                                                             |
| 2                 | Vision refactor        | `core/vision/*`, migrate camera routes to use `VisionPipeline`.                                     |
| 3                 | AI inference refactor  | `core/inference/*`, unify DeepFace/FaceNet fallback, expose gRPC-friendly API for future expansion. |
| 4                 | Attendance workflow    | `core/workflows/attendance_state.py`, update Flask routes + SSE to call through state manager.      |
| 5                 | Frontend/API alignment | Adjust REST/SSE payloads, document new contract, ensure JS uses updated fields.                     |

Each phase should include unit-like smoke tests (where practical) and manual validation notes.

---

## 5. Migration Notes

1. **Incremental adoption**: Temporarily keep adapters so legacy functions call into new states; remove adapters once routes are fully migrated.
2. **Configuration**: Consolidate ENV handling into a `settings.py` module consumed by all states (avoid scattering `os.getenv`).
3. **Error handling**: States return structured errors; Flask translates to HTTP responses. This keeps UI consistent even when internals change.
4. **Testing hooks**: Provide fake implementations (`FakeCameraManager`, `FakeInferenceEngine`) for CLI scripts and unit tests.

---

With this plan approved, the next step is Phase 2 (vision layer extraction). Let me know if any adjustments are needed before coding.
