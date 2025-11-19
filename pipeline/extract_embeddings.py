"""Extract embeddings from processed face images.

This will try to use FaceNet frozen graph if available at
`face_attendance/Models/20180402-114759.pb` with TensorFlow. If TF or the
model is not available, falls back to `face_recognition` encodings (128-d).

Output: saves NumPy arrays `embeddings.npy` and `labels.npy` under
`pipeline/embeddings/`.
"""
import argparse
from pathlib import Path
import numpy as np
import os

try:
    import tensorflow as tf
    _HAS_TF = True
except Exception:
    _HAS_TF = False

try:
    import face_recognition
    _HAS_FR = True
except Exception:
    _HAS_FR = False

from PIL import Image


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def face_recognition_embeddings(image_path):
    img = face_recognition.load_image_file(str(image_path))
    encs = face_recognition.face_encodings(img)
    if not encs:
        return None
    return encs[0]


def load_facenet_graph(pb_path):
    with tf.io.gfile.GFile(str(pb_path), 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name='')
    return graph


def tf_facenet_embeddings(graph, image_paths):
    emb_list = []
    input_tensor = graph.get_tensor_by_name('input:0')
    phase_train_tensor = graph.get_tensor_by_name('phase_train:0')
    embeddings_tensor = graph.get_tensor_by_name('embeddings:0')
    sess = tf.compat.v1.Session(graph=graph)
    for p in image_paths:
        img = Image.open(p).convert('RGB').resize((160, 160))
        arr = np.asarray(img, dtype=np.float32)
        # prewhiten
        mean = np.mean(arr)
        std = np.std(arr)
        std_adj = np.maximum(std, 1.0/np.sqrt(arr.size))
        img_data = (arr - mean) / std_adj
        img_data = np.expand_dims(img_data, 0)
        feed = {input_tensor: img_data, phase_train_tensor: False}
        emb = sess.run(embeddings_tensor, feed_dict=feed)
        emb_list.append(emb[0])
    sess.close()
    return emb_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", default="pipeline/processed")
    parser.add_argument("--outdir", default="pipeline/embeddings")
    args = parser.parse_args()

    processed = Path(args.processed)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    model_pb = Path("face_attendance/Models/20180402-114759.pb")

    image_paths = []
    labels = []
    for label_dir in processed.iterdir():
        if not label_dir.is_dir():
            continue
        for img_path in label_dir.glob("*.jpg"):
            image_paths.append(img_path)
            labels.append(label_dir.name)

    if not image_paths:
        print("No processed images found. Run pipeline/preprocess.py first.")
        return

    embeddings = []
    if _HAS_TF and model_pb.exists():
        print("Using TensorFlow FaceNet model for embeddings (may be slow).")
        graph = load_facenet_graph(model_pb)
        embeddings = tf_facenet_embeddings(graph, image_paths)
    elif _HAS_FR:
        print("Using face_recognition fallback for embeddings.")
        for p in image_paths:
            emb = face_recognition_embeddings(p)
            if emb is None:
                emb = np.zeros(128, dtype=float)
            embeddings.append(emb)
    else:
        raise RuntimeError("No embedding backend available: install tensorflow or face_recognition")

    embeddings = np.array(embeddings)
    labels = np.array(labels)
    np.save(outdir / "embeddings.npy", embeddings)
    np.save(outdir / "labels.npy", labels)
    print(f"Saved embeddings ({embeddings.shape}) and labels to {outdir}")


if __name__ == "__main__":
    main()
