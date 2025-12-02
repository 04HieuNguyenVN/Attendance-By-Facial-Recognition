"""Simple console presence demo.

This script runs a webcam loop, recognizes faces using the trained SVM
classifier in `face_attendance/Models/facemodel.pkl` (preferred) or falls back
to average encodings from `data/*` directories using `face_recognition`.

Confirmed recognition updates the `PresenceTracker` which can be saved to CSV.

Usage:
  python win_console_presence.py

Dependencies: face_recognition, opencv-python, scikit-learn
"""
import os
import time
from pathlib import Path
import argparse
from datetime import datetime

try:
    import cv2
except Exception:
    raise RuntimeError('opencv-python is required')

try:
    import face_recognition
    _HAS_FR = True
except Exception:
    _HAS_FR = False

import pickle
import numpy as np

from services.presence_tracker import PresenceTracker


REQUIRED_FRAMES = 8
DIST_THRESHOLD = 0.5


def load_classifier(model_path: Path, label_path: Path):
    if model_path.exists() and label_path.exists():
        with open(model_path, 'rb') as f:
            clf = pickle.load(f)
        with open(label_path, 'rb') as f:
            le = pickle.load(f)
        return clf, le
    return None, None


RESERVED_DATA_SUBDIRS = {'training_samples', 'models', 'external_assets'}


def build_known_encodings_from_data(data_dir: Path):
    known = {}
    if not _HAS_FR:
        return known
    for person in data_dir.iterdir():
        if not person.is_dir() or person.name in RESERVED_DATA_SUBDIRS:
            continue
        encs = []
        for img in person.glob('*.jpg'):
            try:
                img_arr = face_recognition.load_image_file(str(img))
                e = face_recognition.face_encodings(img_arr)
                if e:
                    encs.append(e[0])
            except Exception:
                continue
        if encs:
            known[person.name] = np.mean(encs, axis=0)
    return known


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0)
    args = parser.parse_args()

    model_path = Path('face_attendance/Models/facemodel.pkl')
    label_path = Path('face_attendance/Models/label_encoder.pkl')

    clf, le = load_classifier(model_path, label_path)
    if clf is not None:
        print('Loaded classifier model')
    else:
        print('Classifier not found, using fallback average encodings from data/')

    known = build_known_encodings_from_data(Path('data'))
    tracker = PresenceTracker()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print('Cannot open camera')
        return

    frame_count = {}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if not _HAS_FR:
                time.sleep(0.05)
                continue
            face_locs = face_recognition.face_locations(rgb)
            encs = face_recognition.face_encodings(rgb, face_locs)
            names_in_frame = []
            for emb in encs:
                name = 'Unknown'
                if clf is not None:
                    probas = clf.predict_proba([emb])[0]
                    idx = np.argmax(probas)
                    conf = probas[idx]
                    if conf > 0.6:
                        name = le.inverse_transform([idx])[0]
                else:
                    # fallback nearest neighbor to known
                    best = None
                    best_d = 1e9
                    for k, v in known.items():
                        d = np.linalg.norm(v - emb)
                        if d < best_d:
                            best_d = d
                            best = k
                    if best is not None and best_d < DIST_THRESHOLD:
                        name = best

                # stability over frames
                cnt = frame_count.get(name, 0) + 1
                frame_count[name] = cnt
                if cnt >= REQUIRED_FRAMES:
                    # confirmed
                    tracker.update(name, datetime.now())
                    frame_count[name] = 0
                    print(f"[{datetime.now().isoformat()}] Confirmed: {name}")

            # periodically print summary
            if int(time.time()) % 10 == 0:
                s = tracker.get_summary()
                if s:
                    print('--- Presence summary ---')
                    for r in s:
                        print(f"{r['name']:30} last:{r['last_seen'].strftime('%H:%M:%S')}  mins:{r['minutes']}")
                    print('------------------------')

            # basic throttle
            time.sleep(0.08)
    finally:
        # save session CSV
        fname = tracker.session_filename()
        tracker.save_csv(fname)
        print('Saved session to', fname)
        cap.release()


if __name__ == '__main__':
    main()
