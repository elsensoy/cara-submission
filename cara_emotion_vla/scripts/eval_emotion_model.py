# scripts/eval_emotion_model.py
from pathlib import Path
import cv2
import json

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import yaml
import torch

from models.emotion_model import EmotionModel

DATA_FACES_DIR = Path(__file__).resolve().parents[1] / "data" / "faces"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "emotions.yaml"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_emotions():
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg["emotions"]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # TODO: replace this with actual model loading code
    # e.g., model = load_pretrained_vit_fer()
    # For now, assume you wrote load_pretrained_model() that returns nn.Module
    from models import pretrained  # you'll create this
    model = pretrained.load_pretrained_model()

    emo_model = EmotionModel(model, device=device)
    label_list = load_emotions()

    y_true = []
    y_pred = []

    for label in label_list:
        class_dir = DATA_FACES_DIR / label
        if not class_dir.exists():
            print(f"Skipping {label}, no directory.")
            continue

        img_paths = list(class_dir.glob("*.png"))
        for img_path in img_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            result = emo_model.predict(img)
            y_true.append(label)
            y_pred.append(result["emotion"])

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=label_list)

    print("Accuracy:", acc)
    print("Macro F1:", macro_f1)
    print("Per-class report:")
    print(classification_report(y_true, y_pred))

    # Save to json
    results = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "classification_report": report,
        "labels": label_list,
        "confusion_matrix": cm.tolist(),
    }
    with open(RESULTS_DIR / "emotion_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {RESULTS_DIR / 'emotion_eval_results.json'}")


if __name__ == "__main__":
    main()
