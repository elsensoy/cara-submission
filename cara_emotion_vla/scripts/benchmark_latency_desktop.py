# scripts/benchmark_latency_desktop.py
import time
from pathlib import Path
import cv2
import numpy as np
import torch

from models.emotion_model import EmotionModel

SAMPLE_FACE_DIR = Path(__file__).resolve().parents[1] / "data" / "faces"


def get_sample_face():
    # Just grab any one face for timing
    for emotion_dir in SAMPLE_FACE_DIR.iterdir():
        if emotion_dir.is_dir():
            imgs = list(emotion_dir.glob("*.png"))
            if imgs:
                img = cv2.imread(str(imgs[0]))
                return img
    return None


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from models import pretrained
    model = pretrained.load_pretrained_model()
    emo_model = EmotionModel(model, device=device)

    sample_face = get_sample_face()
    if sample_face is None:
        print("No sample face found in data/faces/. Run data prep first.")
        return

    N_WARMUP = 20
    N_RUNS = 200

    # Warmup
    for _ in range(N_WARMUP):
        emo_model.predict(sample_face)

    start = time.time()
    for _ in range(N_RUNS):
        emo_model.predict(sample_face)
    end = time.time()

    total = end - start
    avg = total / N_RUNS
    fps = 1.0 / avg

    print(f"Desktop benchmark: {avg * 1000:.2f} ms per frame, {fps:.2f} FPS")


if __name__ == "__main__":
    main()
