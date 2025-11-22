"""
Train classifier CLI script.

Usage (PowerShell):
    python .\tools\train_classifier.py

This script will:
- Import `services.face_service.FaceRecognitionService`
- Initialize `TrainingService` with that service
- Run `train_classifier()` and print results/stats

Note: FaceNet must be available (TensorFlow model path and MTCNN), otherwise this script will fail and instruct how to proceed.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    try:
        from services.face_service import FaceRecognitionService
        from services.training_service import TrainingService
    except Exception as e:
        print('Could not import services. Ensure you run this from project root and dependencies are installed.')
        print('Import error:', e)
        return

    try:
        print('Initializing FaceRecognitionService (FaceNet)...')
        face_service = FaceRecognitionService()
        face_service.load_model()
    except Exception as e:
        print('Failed to initialize FaceNet service. Error:')
        print(e)
        print('\nIf you do not have FaceNet/TensorFlow available, you must install them or use a precomputed classifier.')
        return

    try:
        trainer = TrainingService(face_service)
        print('Loaded training service. Gathering training stats...')
        stats = trainer.get_training_stats()
        print('Training data summary:')
        print(stats)

        confirm = input('Continue to train classifier now? (y/N): ').strip().lower()
        if confirm != 'y':
            print('Aborted by user.')
            return

        print('Starting training...')
        success = trainer.train_classifier()
        if success:
            print('Training completed successfully.')
            stats = trainer.get_training_stats()
            print('Updated training stats:')
            print(stats)
        else:
            print('Training failed. Check logs for details.')

    except Exception as e:
        print('Error during training:', e)

if __name__ == '__main__':
    main()
