# scripts/crop_faces.py
import cv2
from pathlib import Path
from vision.face_detector import FaceDetector

RAW_FRAMES_DIR = Path(__file__).resolve().parents[1] / "data" / "raw_frames"
FACES_DIR = Path(__file__).resolve().parents[1] / "data" / "faces"

MIN_FACE_SIZE = 80  # minimum width/height to consider valid


def crop_faces_for_emotion(emotion: str, detector: FaceDetector):
    src_dir = RAW_FRAMES_DIR / emotion
    dst_dir = FACES_DIR / emotion
    dst_dir.mkdir(parents=True, exist_ok=True)

    img_paths = list(src_dir.glob("*.png"))
    print(f"Cropping faces for emotion={emotion}, images={len(img_paths)}")

    saved_count = 0
    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        faces = detector.detect_faces(img)
        if len(faces) == 0:
            continue

        # pick the largest face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]

        if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
            continue

        face_crop = img[y:y+h, x:x+w]
        out_path = dst_dir / f"{img_path.stem}_face.png"
        cv2.imwrite(str(out_path), face_crop)
        saved_count += 1

    print(f"Saved {saved_count} face crops for emotion={emotion}")


if __name__ == "__main__":
    detector = FaceDetector()
    emotions = [d.name for d in RAW_FRAMES_DIR.iterdir() if d.is_dir()]
    for emo in emotions:
        crop_faces_for_emotion(emo, detector)
