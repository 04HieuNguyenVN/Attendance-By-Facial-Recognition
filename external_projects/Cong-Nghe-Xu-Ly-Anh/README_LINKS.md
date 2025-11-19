This folder references the attached project located at:

`g:\Python\Attendance by facial recognition\Cong-Nghe-Xu-Ly-Anh`

Integration suggestions

- Option A (Safe, recommended): Keep this project as a subproject. Import or call specific modules from it when needed.

  - Use scripts directly from the attached folder path.
  - Merge only what you need (e.g., `yolov8m-face.pt`, `diemdanh_deepface_gui.py`, or selected templates).

- Option B (Merge into main project root): Carefully merge the following files into the main project, resolving conflicts:

  - `app.py` (compare to existing `app.py`) — merge handlers/features you want.
  - `requirements1.txt` — merge required packages into main `requirements.txt`.
  - `yolov8m-face.pt` — place in `models/` or `weights/` and update code to use it.
  - `templates/` — copy templates that you want to reuse; avoid name collisions.
  - `faces/`, `images/` — copy only necessary sample images/data.

- Option C (Extract specific functionality): If you only need the YOLO face model or the DeepFace GUI script, copy those files into `services/` or `tools/` and adapt imports.

Next steps you can pick:

1. Keep as subproject (no further action required). I will leave a pointer here.
2. I can merge selected files now — tell me which files to merge into the root or whether to overwrite `app.py`.
3. I can prepare a safe merge plan (list of files with conflict-risk and exact steps to merge).

Tell me which option you prefer and I'll proceed with the chosen integration (I can also run diffs and create merge patches).
