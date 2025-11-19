"""Train a classifier (SVM) on embeddings and save model.

Input: `pipeline/embeddings/embeddings.npy` and `pipeline/embeddings/labels.npy`
Output: `face_attendance/Models/facemodel.pkl` and `face_attendance/Models/label_encoder.pkl`

Usage:
  python pipeline/train_classifier.py --emb pipeline/embeddings --out face_attendance/Models
"""
import argparse
from pathlib import Path
import numpy as np
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", default="pipeline/embeddings")
    parser.add_argument("--out", default="face_attendance/Models")
    args = parser.parse_args()

    emb_dir = Path(args.emb)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_path = emb_dir / "embeddings.npy"
    labels_path = emb_dir / "labels.npy"
    if not emb_path.exists() or not labels_path.exists():
        print("Embeddings not found. Run pipeline/extract_embeddings.py first.")
        return

    X = np.load(emb_path)
    y = np.load(labels_path)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    clf = SVC(C=1.0, kernel='linear', probability=True)
    clf.fit(X, y_enc)

    with open(out_dir / "facemodel.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open(out_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    print(f"Saved classifier and label encoder to {out_dir}")


if __name__ == "__main__":
    main()
