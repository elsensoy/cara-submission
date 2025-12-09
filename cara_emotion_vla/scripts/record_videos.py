# scripts/record_videos.py
import cv2
from pathlib import Path
import time

EMOTIONS = ["neutral", "happy", "sad", "angry", "surprise", "confused"]
BASE_DIR = Path(__file__).resolve().parents[1] / "data" / "raw_videos"


def record_for_emotion(emotion: str, camera_id: int = 0, num_clips: int = 5, seconds_per_clip: int = 5):
    emotion_dir = BASE_DIR / emotion
    emotion_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"Recording {num_clips} clips for emotion: {emotion}")

    for clip_idx in range(num_clips):
        filename = emotion_dir / f"{emotion}_{clip_idx:02d}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(filename), fourcc, 20.0, (640, 480))

        print(f"--> Clip {clip_idx+1}/{num_clips}. Press 'q' to stop early.")
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            out.write(frame)
            cv2.imshow("Recording - press q to stop", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if time.time() - start_time >= seconds_per_clip:
                break

        out.release()

    cap.release()
    cv2.destroyAllWindows()
    print("Done recording.")


if __name__ == "__main__":
    print("Available emotions:", EMOTIONS)
    emotion = input("Enter emotion to record: ").strip().lower()
    if emotion not in EMOTIONS:
        print("Invalid emotion.")
    else:
        record_for_emotion(emotion)
