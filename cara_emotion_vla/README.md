# Cara Emotion VLA (Vision-Language-Action) Perception Module

##  Project Overview
This repository houses the visual emotion recognition subsystem for **Cara**, an embodied companion robot. It implements a Vision Transformer (ViT) pipeline to detect and classify human facial expressions in real-time, serving as the sensory input for Cara's emotional state machine.

**Current Architecture:**
* **Backbone:** ViT-Base (Patch 16, 224x224)
* **Pretraining:** ImageNet-21k -> FER Fine-tuning
* **Inference Target:** NVIDIA Jetson Xavier / Orin

##  Structure
* `data/`: Hierarchical storage for raw video, extracted frames, and cropped face datasets.
* `models/`: PyTorch definitions and HuggingFace wrappers for the ViT backbone.
* `vision/`: Hardware abstractions for camera streaming and face detection (Haar/YuNet).
* `scripts/`: Utilities for the full data lifecycle (Recording -> Labelling -> Benchmarking).

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt