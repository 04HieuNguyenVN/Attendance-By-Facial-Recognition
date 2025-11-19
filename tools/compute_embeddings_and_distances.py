"""
Small diagnostic script:
- Loads either FaceNet service (if available) or face_recognition
- Iterates images in `data/` folder named as `studentid_name.jpg` (existing convention)
- Computes embeddings for each image
- Prints per-image embedding norm and pairwise distances grouped by student_id

Run locally: `python tools/compute_embeddings_and_distances.py`

This helps identify whether embeddings for different students are distinct.
"""

import os
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'

# Try to import FaceRecognitionService
face_service = None
USE_FACENET = False
try:
    sys.path.insert(0, str(ROOT))
    from services.face_service import FaceRecognitionService
    face_service = FaceRecognitionService()
    face_service.load_model()
    USE_FACENET = True
    print('Using FaceNet service for embeddings')
except Exception as e:
    print('FaceNet service not available or failed to load:', e)
    print('Falling back to face_recognition if installed')
    try:
        import face_recognition
        print('Using face_recognition for embeddings')
    except Exception as e2:
        print('face_recognition not available:', e2)
        print('Cannot compute embeddings. Install FaceNet or face_recognition.')
        sys.exit(1)


def load_images(data_dir):
    imgs = []
    for p in sorted(data_dir.glob('*.jpg')):
        name = p.stem
        # expect studentid_name.jpg or similar
        parts = name.split('_')
        student_id = parts[0] if len(parts) > 1 else name
        imgs.append({'path': str(p), 'student_id': student_id, 'filename': p.name})
    return imgs


def get_embedding_face_recognition(path):
    import face_recognition
    img = face_recognition.load_image_file(path)
    encs = face_recognition.face_encodings(img)
    if not encs:
        return None
    return encs[0]


def get_embedding_facenet(path):
    # load image, detect largest face, preprocess using service
    import cv2
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        return None
    # convert to RGB
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # try detect faces using service
    faces = face_service.detect_faces(img)
    if not faces:
        return None
    # pick largest
    faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
    x, y, w, h = faces[0]
    crop = img[y:y+h, x:x+w]
    pre = face_service.preprocess_face(crop)
    emb = face_service.get_embedding(pre)
    return emb


def main():
    images = load_images(DATA_DIR)
    if not images:
        print('No images found in data/ - add sample images named as <studentid>_<name>.jpg')
        return

    embeddings = []
    for info in images:
        path = info['path']
        try:
            if USE_FACENET:
                emb = get_embedding_facenet(path)
            else:
                emb = get_embedding_face_recognition(path)
            if emb is None:
                print(f"WARNING: no embedding for {info['filename']}")
                continue
            embeddings.append({'info': info, 'emb': np.array(emb, dtype=float)})
            print(f"Loaded embedding for {info['filename']} (student_id={info['student_id']}) norm={np.linalg.norm(emb):.3f}")
        except Exception as e:
            print(f"Error processing {info['filename']}: {e}")

    # Compute pairwise distances
    n = len(embeddings)
    if n < 2:
        print('Need at least 2 embeddings to compute distances')
        return

    # build matrix
    mats = np.array([e['emb'] for e in embeddings])
    # normalize if using face_recognition (they are already normalized usually)
    from scipy.spatial.distance import cdist
    dists = cdist(mats, mats, metric='euclidean')

    print('\nPairwise distance matrix (euclidean):')
    # print header
    fnames = [e['info']['filename'] for e in embeddings]
    header = ' \t' + '\t'.join(fnames)
    print(header)
    for i, row in enumerate(dists):
        line = fnames[i] + '\t' + '\t'.join(f"{v:.3f}" for v in row)
        print(line)

    # Summarize intra-class vs inter-class
    same = []
    diff = []
    for i in range(n):
        for j in range(i+1, n):
            if embeddings[i]['info']['student_id'] == embeddings[j]['info']['student_id']:
                same.append(dists[i, j])
            else:
                diff.append(dists[i, j])

    if same:
        print(f"\nAverage intra-class distance: {np.mean(same):.4f} (count={len(same)})")
    if diff:
        print(f"Average inter-class distance: {np.mean(diff):.4f} (count={len(diff)})")

    if same and diff:
        print('\nInterpretation: intra-class should be much smaller than inter-class. Typical FaceNet thresholds: 0.6-1.0 for similar vs different (euclidean distance).')

if __name__ == '__main__':
    main()
