Pipeline for data collection, preprocessing, embeddings and training
===============================================================

This folder contains helper scripts to run the full offline pipeline:

- `collect_samples.py`: capture face images from webcam per student.
- `preprocess.py`: detect, crop and resize faces to 160x160.
- `extract_embeddings.py`: extract embeddings (FaceNet if available, fallback to `face_recognition`).
- `train_classifier.py`: train an SVM classifier and save the model.

Quick start (recommended):

1) Create a Python venv and install minimal pipeline dependencies (see `requirements_pipeline.txt`).

2) Collect samples for each student:

```powershell
python "pipeline/collect_samples.py" --id SV0001 --name "Nguyen Van A" --count 30
```

3) Preprocess collected images:

```powershell
python pipeline/preprocess.py --sourcedir data --outdir pipeline/processed
```

4) Extract embeddings:

```powershell
python pipeline/extract_embeddings.py --processed pipeline/processed --outdir pipeline/embeddings
```

5) Train classifier:

```powershell
python pipeline/train_classifier.py --emb pipeline/embeddings --out face_attendance/Models
```

After training, `face_attendance/Models/facemodel.pkl` and
`face_attendance/Models/label_encoder.pkl` will be created and can be used by
the attendance application / demo.

Notes:
- TensorFlow + FaceNet gives higher-quality embeddings but requires GPU/CPU heavy deps.
- `face_recognition` fallback works well for small deployments and is much easier to install on Windows.
