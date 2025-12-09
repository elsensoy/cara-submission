# scripts/extract_frames.py
import cv2
from pathlib import Path

RAW_VIDEO_DIR = Path(__file__).resolve().parents[1] / "data" / "raw_videos"
RAW_FRAMES_DIR = Path(__file__).resolve().parents[1] / "data" / "raw_frames"

FRAME_STEP = 5  # save every 5th frame


def extract_frames_for_emotion(emotion: str):
    src_dir = RAW_VIDEO_DIR / emotion
    dst_dir = RAW_FRAMES_DIR / emotion
    dst_dir.mkdir(parents=True, exist_ok=True)

    video_paths = list(src_dir.glob("*.mp4"))

    print(f"Extracting frames for emotion={emotion}, videos={len(video_paths)}")

    for vid_idx, vid_path in enumerate(video_paths):
        cap = cv2.VideoCapture(str(vid_path))
        frame_idx = 0
        saved_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % FRAME_STEP == 0:
                out_path = dst_dir / f"{vid_path.stem}_frame{saved_idx:04d}.png"
                cv2.imwrite(str(out_path), frame)
                saved_idx += 1

            frame_idx += 1

        cap.release()

    print("Done.")


if __name__ == "__main__":
    emotions = [d.name for d in RAW_VIDEO_DIR.iterdir() if d.is_dir()]
    print("Found emotions:", emotions)
    for emo in emotions:
        extract_frames_for_emotion(emo)
