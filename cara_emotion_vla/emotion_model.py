# models/emotion_model.py
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "emotions.yaml"


class EmotionModel:
    def __init__(self, model, device="cuda"):
        """
        model: a torch.nn.Module that takes a (B, C, H, W) tensor and outputs logits (B, num_classes)
        device: "cuda" or "cpu"
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()

        # Load emotion labels from config
        with open(CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f)
        self.idx_to_label = cfg["emotions"]

        # Basic preprocessing; adjust size/mean/std for your chosen model
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # common ViT input size
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

    @torch.no_grad()
    def predict(self, face_image_np):
        """
        face_image_np: H x W x 3 numpy array (RGB or BGR depending on how you pass it)
        Returns: dict with emotion, confidence, probs (per class)
        """
        # If from OpenCV, it's BGR; convert to RGB
        if face_image_np.shape[2] == 3:
            # We assume it's BGR (OpenCV) → convert to RGB
            face_image_np = face_image_np[:, :, ::-1]

        pil_img = Image.fromarray(face_image_np)
        x = self.transform(pil_img).unsqueeze(0).to(self.device)  # (1, C, H, W)

        logits = self.model(x)  # (1, num_classes)
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        idx = probs.argmax()
        label = self.idx_to_label[idx]
        confidence = float(probs[idx])

        return {
            "emotion": label,
            "confidence": confidence,
            "probs": {self.idx_to_label[i]: float(p) for i, p in enumerate(probs)}
        }
