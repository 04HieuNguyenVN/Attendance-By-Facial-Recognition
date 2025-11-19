"""Collect webcam face samples for a student.

Saves images to `data/<id>_<name>/` as JPEG files.

Usage:
  python "pipeline/collect_samples.py" --id SV0001 --name "Nguyen Van A" --count 30

Dependencies: opencv-python, optionally face_recognition for detection.
If face_recognition is not available, falls back to OpenCV Haar cascade.
"""
import argparse
import time
from pathlib import Path

try:
    import face_recognition
    _HAS_FR = True
except Exception:
    _HAS_FR = False

import cv2


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def detect_faces_fr(rgb):
    boxes = face_recognition.face_locations(rgb)
    return boxes


def detect_faces_cv(gray, cascade):
    rects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    boxes = []
    for (x, y, w, h) in rects:
        boxes.append((y, x + w, y + h, x))
    return boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--outdir", default="data")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    outdir = Path(args.outdir) / f"{args.id}_{args.name}"
    ensure_dir(outdir)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    cascade = None
    if not _HAS_FR:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    collected = 0
    print(f"Starting capture for {args.id}_{args.name}, target {args.count} images. Press q to quit.")
    while collected < args.count:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        boxes = []
        if _HAS_FR:
            boxes = detect_faces_fr(rgb)
        else:
            boxes = detect_faces_cv(gray, cascade)

        for (top, right, bottom, left) in boxes:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.putText(frame, f"Collected {collected}/{args.count}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Collect Samples", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break

        if boxes:
            top, right, bottom, left = boxes[0]
            top = max(0, top)
            left = max(0, left)
            bottom = min(frame.shape[0], bottom)
            right = min(frame.shape[1], right)
            face = frame[top:bottom, left:right]
            if face.size == 0:
                continue
            fname = outdir / f"img_{int(time.time()*1000)}.jpg"
            cv2.imwrite(str(fname), face)
            collected += 1
            time.sleep(0.12)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Finished. Saved {collected} images to {outdir}")


if __name__ == "__main__":
    main()
