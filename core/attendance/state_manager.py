"""Centralized attendance state machine for check-in/check-out workflows."""
from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple


SessionProvider = Callable[[], Optional[Dict[str, Any]]]
SessionSerializer = Callable[[Optional[Dict[str, Any]]], Optional[Dict[str, Any]]]
EventBroadcaster = Callable[[Dict[str, Any]], None]


class AttendanceStateManager:
    """Thread-safe state manager for attendance lifecycle."""

    def __init__(
        self,
        *,
        db: Any,
        session_provider: SessionProvider,
        session_serializer: SessionSerializer,
        broadcaster: Optional[EventBroadcaster] = None,
        confirm_seconds: float = 5.0,
        presence_timeout: float = 300.0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._db = db
        self._session_provider = session_provider
        self._session_serializer = session_serializer
        self._broadcast = broadcaster
        self._confirm_seconds = max(confirm_seconds, 0.0)
        self._presence_timeout = max(presence_timeout, 0.0)
        self._logger = logger or logging.getLogger(__name__)

        self._checked_in: set[str] = set()
        self._checked_out: set[str] = set()
        self._students: Dict[str, Dict[str, Any]] = {}
        self._presence: Dict[str, Dict[str, Any]] = {}
        self._progress: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------
    def load_today_records(self) -> None:
        try:
            records = self._db.get_today_attendance() or []
        except Exception as exc:  # pragma: no cover - DB errors runtime dependent
            self._logger.error("[Attendance] Không thể tải điểm danh hôm nay: %s", exc)
            return

        with self._lock:
            self._checked_in.clear()
            self._checked_out.clear()
            self._students.clear()
            for row in records:
                row_dict = dict(row) if not isinstance(row, dict) else row
                student_id = (row_dict.get("student_id") or "").strip()
                if not student_id:
                    continue
                if row_dict.get("check_in_time"):
                    self._checked_in.add(student_id)
                if row_dict.get("check_out_time"):
                    self._checked_out.add(student_id)
                self._students[student_id] = {
                    "name": row_dict.get("student_name") or student_id,
                    "class_name": row_dict.get("class_name") or row_dict.get("credit_class_name"),
                    "class_type": "credit" if row_dict.get("credit_class_id") else "administrative",
                    "credit_class_id": row_dict.get("credit_class_id"),
                }

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def is_checked_in(self, student_id: str) -> bool:
        with self._lock:
            return student_id in self._checked_in

    def is_checked_out(self, student_id: str) -> bool:
        with self._lock:
            return student_id in self._checked_out

    def get_student_meta(self, student_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            meta = self._students.get(student_id)
            return dict(meta) if meta else None

    def is_actively_tracked(self, student_id: str) -> bool:
        with self._lock:
            return student_id in self._presence

    # ------------------------------------------------------------------
    # Check-in/out operations
    # ------------------------------------------------------------------
    def mark_check_in(
        self,
        *,
        name: str,
        student_id: str,
        confidence_score: Optional[float] = None,
        expected_student_id: Optional[str] = None,
        expected_credit_class_id: Optional[int] = None,
    ) -> bool:
        normalized_id = (student_id or "").strip()
        expected_id = (expected_student_id or "").strip()
        if expected_id and normalized_id and normalized_id != expected_id:
            self._logger.info(
                "[Attendance] Rejecting check-in %s (expected %s)", normalized_id, expected_id
            )
            return False

        session_ctx = self._session_provider()
        credit_class_id = session_ctx.get("credit_class_id") if session_ctx else None
        if (
            expected_credit_class_id is not None
            and int(credit_class_id or 0) != int(expected_credit_class_id)
        ):
            self._logger.info(
                "[Attendance] Rejecting check-in for %s: session mismatch (expected %s, active %s)",
                normalized_id or student_id,
                expected_credit_class_id,
                credit_class_id,
            )
            return False

        success = self._db.mark_attendance(
            student_id=normalized_id or student_id,
            student_name=name,
            status="present",
            confidence_score=confidence_score,
            notes=None,
            credit_class_id=credit_class_id,
            session_id=session_ctx.get("id") if session_ctx else None,
        )
        if not success:
            return False

        session_payload = self._session_serializer(session_ctx) if self._session_serializer else None
        now = datetime.now()
        with self._lock:
            self._checked_in.add(normalized_id)
            self._checked_out.discard(normalized_id)
            class_name = None
            class_type = None
            credit_ctx = credit_class_id
            if session_payload:
                class_name = session_payload.get("class_name") or session_payload.get("class_code")
                class_type = "credit"
                credit_ctx = session_payload.get("credit_class_id", credit_ctx)
            self._students[normalized_id] = {
                "name": name,
                "class_name": class_name,
                "class_type": class_type or "administrative",
                "credit_class_id": credit_ctx,
            }
            self._presence[normalized_id] = {
                "last_seen": now,
                "check_in_time": now,
                "name": name,
            }
            self._progress.pop(normalized_id, None)

        if self._broadcast:
            self._broadcast(
                {
                    "type": "attendance_marked",
                    "data": {
                        "event": "check_in",
                        "student_id": normalized_id or student_id,
                        "student_name": name,
                        "confidence": confidence_score,
                        "timestamp": now.isoformat(),
                        "session": session_payload,
                    },
                }
            )
        self._logger.info("[Attendance] Check-in success for %s", normalized_id)
        return True

    def mark_check_out(
        self,
        *,
        student_id: str,
        student_name: str = "",
        reason: str = "manual",
        confidence_score: Optional[float] = None,
        expected_student_id: Optional[str] = None,
        expected_credit_class_id: Optional[int] = None,
    ) -> bool:
        normalized_id = (student_id or "").strip()
        expected_id = (expected_student_id or "").strip()
        if expected_id and normalized_id and normalized_id != expected_id:
            self._logger.info(
                "[Attendance] Rejecting checkout %s (expected %s)", normalized_id, expected_id
            )
            return False

        session_ctx = self._session_provider()
        credit_class_id = session_ctx.get("credit_class_id") if session_ctx else None
        if (
            expected_credit_class_id is not None
            and int(credit_class_id or 0) != int(expected_credit_class_id)
        ):
            self._logger.info(
                "[Attendance] Rejecting checkout for %s: session mismatch (expected %s, active %s)",
                normalized_id or student_id,
                expected_credit_class_id,
                credit_class_id,
            )
            return False

        success = self._db.mark_checkout(normalized_id or student_id)
        if not success:
            return False

        session_payload = self._session_serializer(session_ctx) if self._session_serializer else None
        with self._lock:
            self._checked_out.add(normalized_id)
            student_meta = self._students.get(normalized_id)
            resolved_name = student_name or (
                student_meta.get("name") if isinstance(student_meta, dict) else normalized_id
            )
            self._students[normalized_id] = {
                "name": resolved_name,
                "class_name": student_meta.get("class_name") if isinstance(student_meta, dict) else None,
            }
            self._presence.pop(normalized_id, None)
            self._progress.pop(normalized_id, None)

        if self._broadcast:
            self._broadcast(
                {
                    "type": "attendance_checkout",
                    "data": {
                        "event": "check_out",
                        "student_id": normalized_id or student_id,
                        "student_name": student_name or resolved_name,
                        "confidence": confidence_score,
                        "reason": reason,
                        "timestamp": datetime.now().isoformat(),
                        "session": session_payload,
                    },
                }
            )
        self._logger.info("[Attendance] Checkout success for %s (%s)", normalized_id, reason)
        return True

    # ------------------------------------------------------------------
    # Presence tracking
    # ------------------------------------------------------------------
    def update_presence(self, student_id: str, name: str) -> None:
        now = datetime.now()
        with self._lock:
            if student_id in self._presence:
                self._presence[student_id]["last_seen"] = now
            elif student_id in self._checked_in:
                self._presence[student_id] = {
                    "last_seen": now,
                    "check_in_time": now,
                    "name": name,
                }
        try:
            self._db.update_last_seen(student_id, name)
        except Exception as exc:  # pragma: no cover - DB call best effort
            self._logger.debug("[Attendance] update_last_seen failed: %s", exc)

    def prune_presence(self) -> None:
        now = datetime.now()
        timed_out: List[Tuple[str, str]] = []
        with self._lock:
            for student_id, data in list(self._presence.items()):
                last_seen = data.get("last_seen") or now
                if (now - last_seen).total_seconds() > self._presence_timeout:
                    timed_out.append((student_id, data.get("name") or student_id))
        for student_id, name in timed_out:
            self.mark_check_out(student_id=student_id, student_name=name, reason="timeout")

    def presence_snapshot(self) -> List[Dict[str, Any]]:
        snapshot: List[Dict[str, Any]] = []
        now = datetime.now()
        with self._lock:
            for student_id, data in self._presence.items():
                check_in_time = data.get("check_in_time") or now
                last_seen = data.get("last_seen") or now
                snapshot.append(
                    {
                        "student_id": student_id,
                        "name": data.get("name") or student_id,
                        "check_in_time": check_in_time,
                        "last_seen": last_seen,
                        "duration_minutes": int((now - check_in_time).total_seconds() / 60),
                        "seconds_since_seen": int((now - last_seen).total_seconds()),
                        "is_active": (now - last_seen).total_seconds() < 30,
                    }
                )
        return snapshot

    # ------------------------------------------------------------------
    # Confirmation tracking
    # ------------------------------------------------------------------
    def track_confirmation(self, student_id: str, name: str) -> Tuple[float, float, bool, float]:
        if self._confirm_seconds == 0:
            return (0.0, 0.0, True, 1.0)

        now = datetime.now()
        with self._lock:
            entry = self._progress.get(student_id)
            if entry is not None:
                gap = (now - entry.get("last_seen", now)).total_seconds()
                if gap > 1.5:
                    entry = None
            if entry is None:
                entry = {"start": now, "last_seen": now, "name": name}
            else:
                entry["last_seen"] = now
            self._progress[student_id] = entry
            elapsed = (now - entry["start"]).total_seconds()
            confirmed = elapsed >= self._confirm_seconds
            if confirmed:
                self._progress.pop(student_id, None)
        ratio = min(max(elapsed / self._confirm_seconds, 0.0), 1.0)
        return (elapsed, self._confirm_seconds, confirmed, ratio)

    def reset_confirmation(self, student_id: str) -> None:
        with self._lock:
            self._progress.pop(student_id, None)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @property
    def confirm_seconds(self) -> float:
        return self._confirm_seconds

    @property
    def presence_timeout(self) -> float:
        return self._presence_timeout

    def checked_in_ids(self) -> List[str]:
        with self._lock:
            return list(self._checked_in)

    def checked_out_ids(self) -> List[str]:
        with self._lock:
            return list(self._checked_out)
```}