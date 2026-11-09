"""Webcam capture and face-crop helpers."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


class CameraCapture:
    """Thin wrapper around ``cv2.VideoCapture``."""

    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera index {self.camera_index}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            raise RuntimeError("Camera is not open; call open() first")
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return False, None
        return True, frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "CameraCapture":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def crop_face(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    margin: float = 0.2,
    size: int = 224,
) -> np.ndarray:
    """Crop a square-ish face region from ``frame`` and resize to ``size``."""
    h, w = frame.shape[:2]
    x, y, bw, bh = bbox
    mx = int(bw * margin)
    my = int(bh * margin)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(w, x + bw + mx)
    y2 = min(h, y + bh + my)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
