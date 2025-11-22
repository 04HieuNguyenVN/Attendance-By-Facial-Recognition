"""Preprocess collected face images.

Converts images under `data/<id_name>/` into aligned/resized faces saved in
`pipeline/processed/<id_name>/` ready for embeddings extraction.

Usage:
  python pipeline/preprocess.py --sourcedir data --outdir pipeline/processed

This script tries to use `face_recognition` for detection; falls back to OpenCV cascade.
"""
import argparse
from pathlib import Path
import cv2
import os

try:
    import face_recognition
    _HAS_FR = True
except Exception:
    _HAS_FR = False


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def detect_and_save(face_img_path, out_path, cascade):
    img = cv2.imread(str(face_img_path))
    if img is None:
        return False
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    boxes = []
    if _HAS_FR:
        boxes = face_recognition.face_locations(rgb)
    else:
        rects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        for (x, y, w, h) in rects:
            boxes.append((y, x + w, y + h, x))

    if not boxes:
        return False

    top, right, bottom, left = boxes[0]
    h, w = img.shape[:2]
    top = max(0, top)
    left = max(0, left)
    bottom = min(h, bottom)
    right = min(w, right)
    face = img[top:bottom, left:right]
    if face.size == 0:
        return False
    face = cv2.resize(face, (160, 160))
    cv2.imwrite(str(out_path), face)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sourcedir", default="data")
    parser.add_argument("--outdir", default="pipeline/processed")
    args = parser.parse_args()

    src = Path(args.sourcedir)
    out = Path(args.outdir)
    ensure_dir(out)

    cascade = None
    if not _HAS_FR:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    for person_dir in src.iterdir():
        if not person_dir.is_dir():
            continue
        label = person_dir.name
        out_label_dir = out / label
        ensure_dir(out_label_dir)
        for img_path in person_dir.glob("*.jpg"):
            out_path = out_label_dir / img_path.name
            ok = detect_and_save(img_path, out_path, cascade)
            if not ok:
                # try next; we don't abort the whole run
                continue

    print("Preprocessing finished. Processed images are in:", out)


if __name__ == "__main__":
    from pathlib import Path
    main()
