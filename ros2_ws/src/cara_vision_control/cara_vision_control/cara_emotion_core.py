import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import ViTModel, ViTImageProcessor
import numpy as np
import cv2
from pathlib import Path
import json
from datetime import datetime
import glob

# --- 1. THE DATASET HELPER ---
class PersonalEmotionDataset(Dataset):
    def __init__(self, samples, emotion_names):
        self.samples = samples
        self.emotion_names = emotion_names
        self.processor = ViTImageProcessor.from_pretrained('WinKawaks/vit-tiny-patch16-224')
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = sample['frame']
        if image.shape[2] == 3: # BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)
        
        # Find index of the string label (e.g., "happy" -> 0)
        emotion_idx = self.emotion_names.index(sample['true_emotion'])
        
        return {
            'pixel_values': pixel_values,
            'emotion_labels': torch.tensor(emotion_idx, dtype=torch.long)
        }

# --- 2. THE MODEL ---
class PersonalizedEmotionViT(nn.Module):
    def __init__(self, num_emotions=7, freeze_base=True):
        super().__init__()
        self.base_vit = ViTModel.from_pretrained('WinKawaks/vit-tiny-patch16-224', add_pooling_layer=False)
        hidden_size = 192 
        
        if freeze_base:
            for param in self.base_vit.parameters():
                param.requires_grad = False
        
        self.personal_adapter = nn.ModuleDict({
            'adapter': nn.Sequential(
                nn.Linear(hidden_size, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(0.1),
            ),
            'emotion_head': nn.Linear(128, num_emotions),
            'intensity_head': nn.Linear(128, 1),
            'valence_arousal_head': nn.Linear(128, 2),
        })
        
        self.emotion_names = ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral']
        
    def forward(self, pixel_values):
        outputs = self.base_vit(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0]
        adapted = self.personal_adapter['adapter'](cls_token)
        
        return {
            'emotion_logits': self.personal_adapter['emotion_head'](adapted),
            'intensity': torch.sigmoid(self.personal_adapter['intensity_head'](adapted)),
            'valence_arousal': torch.tanh(self.personal_adapter['valence_arousal_head'](adapted)),
        }

    def predict(self, image):
        self.eval()
        with torch.no_grad():
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image).float()
            if image.dim() == 3:
                image = image.unsqueeze(0)
            
            outputs = self.forward(image)
            probs = F.softmax(outputs['emotion_logits'], dim=-1)
            primary_idx = probs.argmax(dim=-1).item()
            
            return {
                'primary_emotion': self.emotion_names[primary_idx],
                'confidence': probs[0, primary_idx].item(),
                'intensity': outputs['intensity'][0, 0].item(),
                'valence': outputs['valence_arousal'][0, 0].item(),
                'arousal': outputs['valence_arousal'][0, 1].item(),
            }

#  3. THE LEARNING SYSTEM   
class InteractiveLearningSystem:
    def __init__(self, model, storage_path="./memory/cara_personal_data"):
        self.model = model
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.processor = ViTImageProcessor.from_pretrained('WinKawaks/vit-tiny-patch16-224')

    def preprocess(self, frame):
        if frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.processor(images=frame, return_tensors="pt")['pixel_values']

    def save_labeled_sample(self, frame, label):
        timestamp = datetime.now()
        filename = timestamp.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(str(self.storage_path / f"{filename}.jpg"), frame)
        meta = {'true_emotion': label, 'timestamp': timestamp.isoformat()}
        with open(self.storage_path / f"{filename}.json", 'w') as f:
            json.dump(meta, f)
        print(f"Saved sample: {label}")

    def load_samples(self):
        samples = []
        json_files = list(self.storage_path.glob("*.json"))
        print(f"Loading {len(json_files)} samples from disk...")
        for j_path in json_files:
            try:
                with open(j_path, 'r') as f:
                    meta = json.load(f)
                img_path = j_path.with_suffix('.jpg')
                if img_path.exists():
                    frame = cv2.imread(str(img_path))
                    samples.append({'frame': frame, 'true_emotion': meta['true_emotion']})
            except Exception as e:
                print(f"Skipping corrupt sample {j_path}: {e}")
        return samples

    def update_model(self, epochs=5, batch_size=4, lr=1e-4):
        # 1. Load data
        data = self.load_samples()
        if len(data) < 2:
            print("Need at least 2 samples to train!")
            return False
            
        dataset = PersonalEmotionDataset(data, self.model.emotion_names)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 2. Setup Optimizer
        optimizer = torch.optim.AdamW(self.model.personal_adapter.parameters(), lr=lr)
        device = next(self.model.parameters()).device
        
        print(f"Starting training on {len(data)} samples...")
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['emotion_labels'].to(device)
                
                outputs = self.model(pixel_values)
                loss = F.cross_entropy(outputs['emotion_logits'], labels)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss/len(dataloader):.4f}")
            
        self.model.eval()
        print("Training complete!")
        return True
