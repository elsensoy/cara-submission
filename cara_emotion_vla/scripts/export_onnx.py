# scripts/export_onnx.py
from optimum.exporters.onnx import main_export
from pathlib import Path

# The specific model we chose
MODEL_ID = "dima806/facial_emotions_image_detection"
OUTPUT_DIR = Path(__file__).parents[1] / "models" / "onnx"

def export():
    print(f"Exporting {MODEL_ID} to ONNX...")
    
    # This automatically traces the ViT graph and handles input shapes
    main_export(
        model_name_or_path=MODEL_ID,
        output=OUTPUT_DIR,
        task="image-classification",
        opset=12,  # Good compatibility for Jetson
        no_post_process=True # We want raw logits, not softmaxed dictionaries
    )
    print(f" Export complete! Model saved to: {OUTPUT_DIR / 'model.onnx'}")

if __name__ == "__main__":
    export()