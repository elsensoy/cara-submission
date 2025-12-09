# vision/face_detector.py
import cv2
import numpy as np
from typing import List, Tuple


class FaceDetector:
    def __init__(self, model_path=None):
        """
        Stub: implement with YuNet, MediaPipe, or another detector.
        For now, we can use OpenCV's Haar cascade to get going.
        """
        if model_path is None:
            self.detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
        else:
            # TODO: load a different detector
            self.detector = cv2.CascadeClassifier(model_path)

    def detect_faces(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Returns list of (x, y, w, h) in pixel coords.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )
        return faces
