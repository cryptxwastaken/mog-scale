"""Face detection and landmark extraction via OpenCV YuNet (FaceDetectorYN)."""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
YUNET_NAME = "face_detection_yunet_2023mar.onnx"
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

# Landmark indices in the parsed (5, 2) array: RE, LE, nose, RM, LM.
LM_RIGHT_EYE = 0
LM_LEFT_EYE = 1
LM_NOSE = 2
LM_RIGHT_MOUTH = 3
LM_LEFT_MOUTH = 4


@dataclass
class FaceResult:
    """Primary face detection with landmarks in image coordinates."""

    bbox: Tuple[int, int, int, int]  # x, y, w, h
    score: float
    landmarks: np.ndarray  # (5, 2) float32: RE, LE, nose, RM, LM


def ensure_yunet_model(models_dir: Path = MODELS_DIR) -> Path:
    """Download YuNet ONNX weights if missing; return local path."""
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / YUNET_NAME
    if path.exists() and path.stat().st_size > 0:
        return path
    print(f"Downloading YuNet model to {path} ...")
    urllib.request.urlretrieve(YUNET_URL, path)
    return path


def parse_face_row(row: np.ndarray) -> FaceResult:
    """
    Parse one FaceDetectorYN detection row into bbox / landmarks / score.

    OpenCV 5 layout (current): x, y, w, h, then 5 landmarks, then score at [14].
    Older OpenCV 4 layout:     x, y, w, h, score at [4], then 5 landmarks.
    """
    row = np.asarray(row, dtype=np.float32).reshape(-1)
    if row.size < 15:
        raise ValueError(f"FaceDetectorYN row expected 15 values, got {row.size}")

    x, y, bw, bh = [int(round(v)) for v in row[:4]]

    # Prefer OpenCV 5+: confidence lives at the end and is typically in (0, 1].
    score_tail = float(row[14])
    score_mid = float(row[4])
    if 0.0 < score_tail <= 1.0 and score_mid > 1.0:
        landmarks = row[4:14].reshape(5, 2).copy()
        score = score_tail
    elif 0.0 < score_mid <= 1.0:
        landmarks = row[5:15].reshape(5, 2).copy()
        score = score_mid
    else:
        # Ambiguous (e.g. very confident OpenCV 4 score==1.0 next to small coords):
        # default to OpenCV 5 layout used by this project's pinned runtime.
        landmarks = row[4:14].reshape(5, 2).copy()
        score = score_tail

    return FaceResult(bbox=(x, y, bw, bh), score=score, landmarks=landmarks)


class FaceLandmarkDetector:
    """Detect the largest face and its 5 YuNet landmarks."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        path = Path(model_path) if model_path else ensure_yunet_model()
        self._model_path = str(path)
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self._detector: Optional[cv2.FaceDetectorYN] = None
        self._input_size: Optional[Tuple[int, int]] = None

    def _ensure_detector(self, width: int, height: int) -> cv2.FaceDetectorYN:
        size = (width, height)
        if self._detector is None or self._input_size != size:
            self._detector = cv2.FaceDetectorYN.create(
                self._model_path,
                "",
                size,
                self.score_threshold,
                self.nms_threshold,
                self.top_k,
            )
            self._input_size = size
        else:
            self._detector.setInputSize(size)
        return self._detector

    def detect(self, frame: np.ndarray) -> Optional[FaceResult]:
        """Return the highest-scoring face, or ``None`` if none found."""
        faces = self.detect_all(frame)
        if not faces:
            return None
        return max(faces, key=lambda f: f.score)

    def detect_all(self, frame: np.ndarray) -> List[FaceResult]:
        h, w = frame.shape[:2]
        detector = self._ensure_detector(w, h)
        _, faces = detector.detect(frame)
        if faces is None or len(faces) == 0:
            return []
        return [parse_face_row(row) for row in faces]
