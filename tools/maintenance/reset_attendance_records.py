"""Utility script to purge attendance-related records.

Usage:
    python tools/reset_attendance_records.py
"""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from database import db  # noqa: E402


def main():
    db.clear_attendance_records()
    print("Đã xóa toàn bộ dữ liệu điểm danh, phiên và lịch sử liên quan.")


if __name__ == "__main__":
    main()
