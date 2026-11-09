"""OpenCV overlay helpers for landmarks, scores, and FPS."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from mog_scale.landmarks import FaceResult
from mog_scale.scoring import ScoreBreakdown


def draw_face(
    frame: np.ndarray,
    face: FaceResult,
    score: Optional[ScoreBreakdown] = None,
    fps: Optional[float] = None,
    logging_on: bool = False,
) -> np.ndarray:
    """Draw bbox, landmarks, and score HUD onto a copy of ``frame``."""
    out = frame.copy()
    x, y, w, h = face.bbox
    cv2.rectangle(out, (x, y), (x + w, y + h), (40, 200, 80), 2)

    for i, (px, py) in enumerate(face.landmarks):
        color = (0, 220, 255) if i < 2 else (255, 180, 0) if i == 2 else (200, 80, 255)
        cv2.circle(out, (int(px), int(py)), 3, color, -1)

    # Eye / mouth midlines for symmetry cue.
    re, le = face.landmarks[0], face.landmarks[1]
    rm, lm = face.landmarks[3], face.landmarks[4]
    mid_x = int(0.5 * (re[0] + le[0]))
    cv2.line(out, (mid_x, y), (mid_x, y + h), (180, 180, 180), 1)
    cv2.line(
        out,
        (int(re[0]), int(re[1])),
        (int(le[0]), int(le[1])),
        (0, 200, 255),
        1,
    )
    cv2.line(
        out,
        (int(rm[0]), int(rm[1])),
        (int(lm[0]), int(lm[1])),
        (200, 80, 255),
        1,
    )

    _draw_hud(out, score, fps, logging_on)
    return out


def draw_no_face(
    frame: np.ndarray,
    fps: Optional[float] = None,
    logging_on: bool = False,
) -> np.ndarray:
    out = frame.copy()
    cv2.putText(
        out,
        "No face detected",
        (16, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (40, 40, 220),
        2,
        cv2.LINE_AA,
    )
    _draw_hud(out, None, fps, logging_on)
    return out


def _draw_hud(
    out: np.ndarray,
    score: Optional[ScoreBreakdown],
    fps: Optional[float],
    logging_on: bool,
) -> None:
    h, w = out.shape[:2]
    panel_w = 240
    overlay = out.copy()
    cv2.rectangle(overlay, (w - panel_w - 8, 8), (w - 8, 170), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

    x0 = w - panel_w
    y = 32
    if score is not None:
        cv2.putText(
            out,
            f"Score {score.total:5.1f}",
            (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (40, 255, 160),
            2,
            cv2.LINE_AA,
        )
        y += 28
        _bar(out, x0, y, "Sym", score.symmetry)
        y += 22
        _bar(out, x0, y, "3rds", score.thirds)
        y += 22
        _bar(out, x0, y, "Prop", score.proportions)
        y += 28
    else:
        y += 20

    if fps is not None:
        cv2.putText(
            out,
            f"FPS {fps:4.1f}",
            (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        y += 20
    log_txt = "LOG ON" if logging_on else "LOG OFF"
    log_col = (40, 220, 40) if logging_on else (120, 120, 120)
    cv2.putText(
        out,
        log_txt,
        (x0, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        log_col,
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        out,
        "q quit | s log | c crop",
        (12, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )


def _bar(img: np.ndarray, x: int, y: int, label: str, value: float, width: int = 140) -> None:
    cv2.putText(
        img,
        label,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    bx = x + 42
    by = y - 8
    cv2.rectangle(img, (bx, by), (bx + width, by + 10), (60, 60, 60), -1)
    fill = int(width * max(0.0, min(1.0, value / 100.0)))
    color = (40, 200, 80) if value >= 70 else (0, 180, 255) if value >= 45 else (40, 40, 220)
    cv2.rectangle(img, (bx, by), (bx + fill, by + 10), color, -1)
