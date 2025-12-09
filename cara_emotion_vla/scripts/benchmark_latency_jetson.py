import time
import numpy as np
import cv2
from pathlib import Path

# Import your optimized TRT wrapper
from models.trt_model import TRTEmotionWrapper

# Constants
ENGINE_PATH = Path(__file__).parents[1] / "models" / "cara_emotion_fp16.engine"
FACES_DIR = Path(__file__).parents[1] / "data" / "faces"
N_WARMUP = 50
N_RUNS = 1000

def get_random_face():
    """Grabs a real face from your dataset to ensure realistic input processing."""
    all_faces = list(FACES_DIR.glob("**/*.png"))
    if not all_faces:
        print("No faces found in data/faces/! Using dummy noise data.")
        return np.random.rand(1, 3, 224, 224).astype(np.float32)
    
    # Pick random face
    path = np.random.choice(all_faces)
    img = cv2.imread(str(path))
    
    # Preprocess match EmotionModel logic: Resize -> BGR2RGB -> Normalize -> Transpose
    img = cv2.resize(img, (224, 224))
    img = img[:, :, ::-1] # BGR to RGB
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = img.transpose(2, 0, 1) # HWC -> CHW
    img = np.expand_dims(img, axis=0) # Add batch dimension -> (1, 3, 224, 224)
    return np.ascontiguousarray(img)

def main():
    if not ENGINE_PATH.exists():
        print(f" Engine not found at {ENGINE_PATH}")
        print("   Did you run scripts/compile_engine.sh?")
        return

    print(f" Loading TensorRT Engine: {ENGINE_PATH.name}...")
    model = TRTEmotionWrapper(str(ENGINE_PATH))

    # Prepare input
    img = get_random_face()
    print(f"   Input shape: {img.shape}")

    # --- PHASE 1: WARMUP ---
    # The Jetson GPU starts at a low clock speed. We spam it with requests
    # to trigger the 'boost' clocks before we start timing.
    print(f" Warming up GPU ({N_WARMUP} iters)...")
    for _ in range(N_WARMUP):
        _ = model.forward(img)

    # --- PHASE 2: BENCHMARK ---
    print(f"⏱️  Running benchmark ({N_RUNS} iters)...")
    latencies = []

    for _ in range(N_RUNS):
        start = time.time()
        
        # This call is blocking (synchronous) because of the wrapper's internals
        _ = model.forward(img)
        
        end = time.time()
        latencies.append((end - start) * 1000) # Convert to ms

    # phase 3: reporting
    latencies = np.array(latencies)
    avg_lat = np.mean(latencies)
    p99_lat = np.percentile(latencies, 99)
    fps = 1000.0 / avg_lat

    print("\n" + "="*40)
    print(f"  JETSON BENCHMARK RESULTS ({ENGINE_PATH.name})")
    print("="*40)
    print(f"  Avg Latency  : {avg_lat:.2f} ms")
    print(f"  P99 Latency  : {p99_lat:.2f} ms (Worst case)")
    print(f"  Throughput   : {fps:.2f} FPS")
    print("="*40)

    if fps > 30:
        print(" Status: REAL-TIME READY (Runs faster than standard webcam 30fps)")
    else:
        print("Status: LAGGY (Optimization needed)")

if __name__ == "__main__":
    main()