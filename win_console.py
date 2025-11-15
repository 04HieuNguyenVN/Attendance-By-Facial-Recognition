"""
Win-console recognition UI (Tkinter)

This script provides a simple Windows-style window (Tkinter) that captures
camera frames, runs face recognition (using `face_recognition` when available)
and displays confirmed attendance events in a listbox.

Notes:
- It's a lightweight helper/demo. It does not modify the main app database by
  default (to avoid side effects). If you want it to write attendance into the
  app DB, we can add a call to `database.db.mark_attendance()`.
- Designed to run in DEMO or with `face_recognition` installed.

Run:
  - Create and activate virtualenv
  - pip install -r requirements.txt (or at least: opencv-python, face-recognition, Pillow)
  - python win_console.py

"""
import threading
import time
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk

try:
    import cv2
except Exception:
    raise RuntimeError('OpenCV (cv2) is required: pip install opencv-python')

try:
    import face_recognition
    FACE_LIB = 'face_recognition'
except Exception:
    face_recognition = None
    FACE_LIB = None

import numpy as np

# Basic configuration
DATA_DIR = Path('data')
CAMERA_INDEX = int(os.getenv('CAMERA_INDEX', '0'))
REQUIRED_FRAMES = int(os.getenv('REQUIRED_FRAMES', '15'))  # shorter for demo
DISTANCE_THRESHOLD = float(os.getenv('FACE_DISTANCE_THRESHOLD', '0.45'))


def load_known_faces():
    """Load images from `data/` folder and compute encodings (if possible).
    Returns: list of (student_id, display_name, encoding or None)
    """
    known = []
    if not DATA_DIR.exists():
        return known

    for img_path in DATA_DIR.glob('*.jpg'):
        parts = img_path.stem.split('_')
        if len(parts) >= 2:
            student_id = parts[0]
            display_name = ' '.join(parts[1:])
        else:
            student_id = img_path.stem
            display_name = img_path.stem

        encoding = None
        if face_recognition is not None:
            try:
                img = face_recognition.load_image_file(str(img_path))
                encs = face_recognition.face_encodings(img)
                if encs:
                    encoding = encs[0]
            except Exception:
                encoding = None

        known.append({'id': student_id, 'name': display_name, 'encoding': encoding})
    return known


class RecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Attendance - Console UI')
        self.frame = ttk.Frame(root, padding=8)
        self.frame.grid(sticky='nsew')

        ttk.Label(self.frame, text='Recognition events').grid(row=0, column=0, sticky='w')
        self.listbox = tk.Listbox(self.frame, width=60, height=15)
        self.listbox.grid(row=1, column=0, pady=6)

        # Controls
        ctrl = ttk.Frame(self.frame)
        ctrl.grid(row=2, column=0, sticky='w')
        self.btn_start = ttk.Button(ctrl, text='Start', command=self.start)
        self.btn_start.grid(row=0, column=0, padx=4)
        self.btn_stop = ttk.Button(ctrl, text='Stop', command=self.stop, state='disabled')
        self.btn_stop.grid(row=0, column=1, padx=4)

        # State
        self.capture = None
        self.running = False
        self.thread = None
        self.known = load_known_faces()

        # Map id -> consecutive frames seen
        self.progress = {k['id']: 0 for k in self.known}

        # If no face_recognition encodings available, show names only
        if not self.known:
            self.listbox.insert(tk.END, 'No known faces found in data/ (place JPG files named <id>_<name>.jpg)')

    def start(self):
        if self.running:
            return
        try:
            self.capture = cv2.VideoCapture(CAMERA_INDEX)
            self.running = True
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()
            self.btn_start.config(state='disabled')
            self.btn_stop.config(state='normal')
            self.listbox.insert(tk.END, f'Started camera index={CAMERA_INDEX}')
        except Exception as e:
            self.listbox.insert(tk.END, f'Error starting camera: {e}')

    def stop(self):
        self.running = False
        self.btn_start.config(state='normal')
        self.btn_stop.config(state='disabled')
        if self.capture is not None:
            try:
                self.capture.release()
            except Exception:
                pass
        self.listbox.insert(tk.END, 'Stopped')

    def _loop(self):
        while self.running:
            ret, frame = self.capture.read()
            if not ret:
                time.sleep(0.1)
                continue

            # Resize small for speed
            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            if face_recognition is None:
                # No recognition library: skip detection, just show timestamp
                ts = time.strftime('%Y-%m-%d %H:%M:%S')
                self._push_event(f'[DEMO] {ts}')
                time.sleep(1.0)
                continue

            # Detect faces and encodings
            try:
                face_locs = face_recognition.face_locations(rgb)
                encs = face_recognition.face_encodings(rgb, face_locs)
            except Exception as e:
                self._push_event(f'Face lib error: {e}')
                time.sleep(0.5)
                continue

            # For each detected face, compare to known
            for enc in encs:
                best_name = None
                best_id = None
                best_dist = 1.0
                for k in self.known:
                    if k['encoding'] is None:
                        continue
                    dist = np.linalg.norm(k['encoding'] - enc)
                    if dist < best_dist:
                        best_dist = dist
                        best_name = k['name']
                        best_id = k['id']

                if best_name is not None and best_dist <= DISTANCE_THRESHOLD:
                    # increment progress for this id, reset others
                    for pid in list(self.progress.keys()):
                        if pid == best_id:
                            self.progress[pid] = self.progress.get(pid, 0) + 1
                        else:
                            self.progress[pid] = 0

                    if self.progress.get(best_id, 0) >= REQUIRED_FRAMES:
                        # Confirmed attendance
                        self._push_event(f'[{time.strftime("%H:%M:%S")}] {best_name} (id={best_id}) - confirmed')
                        # Reset to avoid duplicate immediate events
                        self.progress[best_id] = 0
                else:
                    # no match: optionally reset all
                    for pid in list(self.progress.keys()):
                        self.progress[pid] = 0

            # small sleep to reduce CPU
            time.sleep(0.05)

        # end loop

    def _push_event(self, text):
        # Schedule UI update on main thread
        def _add():
            self.listbox.insert(0, text)
            # keep listbox to reasonable length
            if self.listbox.size() > 200:
                self.listbox.delete(200, tk.END)

        try:
            self.root.after(0, _add)
        except Exception:
            pass


def main():
    root = tk.Tk()
    app = RecognitionApp(root)
    root.protocol('WM_DELETE_WINDOW', lambda: (app.stop(), root.destroy()))
    root.mainloop()


if __name__ == '__main__':
    main()
