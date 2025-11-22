"""Seed sample credit classes and enroll demo students."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from database import db


def main():
    summary = db.seed_sample_credit_classes()
    if not summary:
        print("Không có dữ liệu nào được thêm.")
        return
    print("Hoàn tất khởi tạo lớp tín chỉ mẫu:")
    print(f"  - Lớp tín chỉ mới: {summary.get('classes_created', 0)}")
    print(f"  - Sinh viên mới: {summary.get('students_created', 0)}")
    print(f"  - Ghi danh mới: {summary.get('enrollments_created', 0)}")


if __name__ == "__main__":
    main()
