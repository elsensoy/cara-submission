This report is structured to serve as technical documentation for your Senior Project or a research paper. It bridges the gap between high-level robotics concepts and the deep learning theory "under the hood."

You can include this in your documentation or use it as a script foundation when presenting Cara's architecture.

***

# Project Cara: Adaptive Personalized Emotion Recognition using Vision Transformers
**Technical Report & Architecture Overview**
**Author:** Elida | **Platform:** NVIDIA Jetson Orin / ROS 2 Humble

---

## 1. Abstract
Standard facial emotion recognition models are trained on generic datasets (like FER-2013), often failing to capture the nuances of a specific user’s facial micro-expressions. Project Cara introduces a **Personalized Adaptive Vision System**. By utilizing a pre-trained Vision Transformer (ViT-Tiny) and implementing a lightweight, trainable adapter layer, the system performs real-time, on-device learning. This allows the robot to "learn" its owner's specific emotional cues via human-in-the-loop feedback, bridging the gap between raw perception and empathetic Human-Robot Interaction (HRI).

---

## 2. System Architecture
The pipeline operates within a distributed ROS 2 environment, ensuring modularity and low-latency performance suitable for edge computing.



### 2.1 The Perception Stage
1.  **Input:** 640x480 video stream at 30 FPS.
2.  **Detection (Face-YuNet):** A lightweight CNN-based detector locates the face.
3.  **Preprocessing:** The face is cropped, resized to 224x224, and normalized. This region of interest (ROI) is passed to the inference engine.

### 2.2 The Inference Engine (ViT)
We utilize `vit-tiny-patch16-224` for its balance of speed and accuracy. Unlike Convolutional Neural Networks (CNNs) that process pixels in local windows, the ViT processes the image as a sequence of patches, allowing it to understand global context (e.g., how the eyes relate to the mouth) immediately.

---

## 3. Under the Hood: The Vision Transformer (ViT)
To understand how Cara "sees," we must look at the Transformer architecture.



[Image of Vision Transformer architecture diagram]


### 3.1 Patch Embeddings
The Transformer cannot process a raw grid of pixels. Instead, the 224x224 face image is sliced into **196 fixed-size patches** (each 16x16 pixels).
* **Analogy:** Imagine cutting a photograph into puzzle pieces.
* **Linear Projection:** Each patch is flattened into a vector and mapped to a specific dimension (embedding size).

### 3.2 Positional Embeddings
Because the Transformer processes these patches in parallel (not sequentially), it loses the concept of "up," "down," "left," or "right."
* **The Fix:** We add **Positional Embeddings** to the patch vectors. This effectively "tags" each puzzle piece with its coordinate, so the model knows the forehead patch is above the eye patch.

### 3.3 The Self-Attention Mechanism
This is the core engine. The mechanism, mathematically represented as:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

* **Query (Q):** What the model is looking for (e.g., "Are there crinkles?").
* **Key (K):** What the patch contains (e.g., "I am an eye corner").
* **Value (V):** The actual information content.

**How it works for Cara:**
When Cara analyzes a "smile," the attention mechanism allows the mouth patches to "attend to" the eye patches. It recognizes that a smile is not just a curved mouth, but a curved mouth *plus* crinkled eyes (Duchenne smile).

### 3.4 The CLS Token
A special "Classification Token" is added to the start of the sequence. As the data moves through the Transformer layers, this token aggregates information from all other patches. By the final layer, this single token holds the semantic representation of the entire face.

---

## 4. Personalized Adaptation Strategy
Training a Transformer from scratch requires millions of images and massive compute. To run this on a Jetson Orin, we use **Transfer Learning with Parameter-Efficient Fine-Tuning (PEFT)**.

### 4.1 Frozen Backbone
We freeze the weights of the base ViT model. This ensures Cara retains the general knowledge of what a "face" looks like (edges, textures, shapes) without catastrophic forgetting.

### 4.2 The Trainable Adapter Head
We attach a custom Multi-Layer Perceptron (MLP) head to the output of the CLS token.
* **Input:** 192-dimensional feature vector from ViT.
* **Hidden Layers:** Linear -> LayerNorm -> GELU -> Dropout.
* **Output:** 7 probabilities (Happy, Sad, Angry, etc.).

**Why this matters:** When we run the "Train" command, we *only* update the weights of this small adapter head. This reduces the trainable parameters from millions to just a few thousand, enabling training to finish in seconds on the robot.

---

## 5. The Interactive Learning Loop
This feature allows Cara to evolve from a generic detector to a personal companion.



### 5.1 Data Collection (The Trigger)
When the user triggers `/cara/feedback` (e.g., "I am happy"), the system performs **Latent Latching**:
1.  It retrieves the current frame from the buffer.
2.  It assigns the user-provided label as the "Ground Truth."
3.  It saves the pair to the local dataset.

### 5.2 The Optimization Step
When `/cara/train` is triggered:
1.  **Batching:** The system loads the saved personal images.
2.  **Forward Pass:** Images run through the frozen ViT to get features.
3.  **Loss Calculation:** We use **Cross-Entropy Loss** to measure the difference between Cara's prediction and the user's label.
4.  **Backpropagation:** The error is propagated back, adjusting *only* the Adapter Head weights.

---

## 6. Future Integration: Vision-Language-Action (VLA)
The ultimate goal is to convert this perception into empathetic behavior. This moves the system from a standard classifier to a **Vision-Language-Action** model.

### 6.1 Emotion as Contextual Prompting
Currently, Large Language Models (LLMs) like Llama-3 (running locally) lack visual context. We bridge this by injecting the emotion confidence score into the system prompt.

**Logic Flow:**
1.  **ViT Output:** `Primary: Sad, Confidence: 0.85`.
2.  **Prompt Injection:**
    > *System: The user appears significantly SAD (85%). Adjust response tone to be comforting, lower pitch, and slower.*
3.  **LLM Generation:** "I notice you seem a bit down. Do you want to talk about it?"

### 6.2 Servo-Reflex Action
Simultaneously, the emotion state drives the `cara_head_controller`:
* **If Happy:** Increase head tracking speed (excitement).
* **If Sad:** Tilt head 15 degrees (empathy) and reduce movement speed.
### 6.3 Blinking
Cara has an already implemented blinking system. With its eyelids moving, she can easily create engaged conversations with the user.
---

## 7. Conclusion
Project Cara demonstrates that complex, Transformer-based architectures can be adapted for personalized, edge-based robotics. By decoupling the feature extraction (Frozen ViT) from the classification (Personal Adapter), we achieve a system that is both computationally efficient and highly adaptive to the specific emotional expressions of its human companion.
