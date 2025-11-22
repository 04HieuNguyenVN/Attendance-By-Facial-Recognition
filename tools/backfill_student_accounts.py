"""Generate login accounts for students missing user credentials."""
import csv
import secrets
import string
from datetime import datetime
from pathlib import Path

from werkzeug.security import generate_password_hash

from database import db


def generate_password(length=10):
    alphabet = string.ascii_letters + string.digits
    rng = secrets.SystemRandom()
    return ''.join(rng.choice(alphabet) for _ in range(max(6, length)))


def main():
    students = db.get_students_missing_user(active_only=False)
    if not students:
        print("All students already have linked user accounts.")
        return

    credentials = []
    for row in students:
        student_id = row["student_id"].strip() if row["student_id"] else None
        full_name = row["full_name"] or ""
        email = row.get("email") if isinstance(row, dict) else row["email"]

        if not student_id:
            print("Skipping record without a valid student_id.")
            continue

        plain_password = generate_password()
        password_hash = generate_password_hash(plain_password)

        try:
            user_id = db.create_user(
                username=student_id,
                password_hash=password_hash,
                full_name=full_name or student_id,
                role='student',
                email=email,
            )
        except ValueError as exc:
            print(f"Skip {student_id}: {exc}")
            continue

        if not db.link_student_to_user(student_id, user_id):
            print(f"Warning: unable to link account for {student_id}")

        credentials.append({
            'student_id': student_id,
            'full_name': full_name,
            'username': student_id,
            'password': plain_password,
        })

    if not credentials:
        print("No accounts were generated.")
        return

    output_dir = Path('data')
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = output_dir / f'generated_credentials_{timestamp}.csv'

    with output_path.open('w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['student_id', 'full_name', 'username', 'password'])
        writer.writeheader()
        writer.writerows(credentials)

    print(f"Created {len(credentials)} accounts. Details saved to {output_path}.")


if __name__ == '__main__':
    main()
