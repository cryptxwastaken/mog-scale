"""NumPy geometric feature vectors from facial landmarks."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from mog_scale.landmarks import (
    LM_LEFT_EYE,
    LM_LEFT_MOUTH,
    LM_NOSE,
    LM_RIGHT_EYE,
    LM_RIGHT_MOUTH,
    FaceResult,
)

# Stable column order for logging / model geo branch.
FEATURE_NAMES: Tuple[str, ...] = (
    "symmetry_error",
    "eye_height_asym",
    "mouth_height_asym",
    "nose_midline_offset",
    "thirds_upper",
    "thirds_mid",
    "thirds_lower",
    "thirds_balance",
    "interocular_face_ratio",
    "eye_span_face_ratio",
    "mouth_face_ratio",
    "nose_face_y_ratio",
    "face_aspect",
)


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    if abs(den) < 1e-6:
        return default
    return float(num / den)


def extract_geometry(face: FaceResult) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Build a fixed-length, scale-invariant geometry vector.

    Returns
    -------
    vector : np.ndarray, shape (len(FEATURE_NAMES),), dtype float32
    features : dict mapping FEATURE_NAMES → float
    """
    lm = face.landmarks.astype(np.float64)
    x, y, bw, bh = face.bbox
    face_w = max(float(bw), 1.0)
    face_h = max(float(bh), 1.0)

    re, le = lm[LM_RIGHT_EYE], lm[LM_LEFT_EYE]
    nose = lm[LM_NOSE]
    rm, lm_pt = lm[LM_RIGHT_MOUTH], lm[LM_LEFT_MOUTH]

    # Midline from eye midpoint projected vertically (image x).
    eye_mid_x = 0.5 * (re[0] + le[0])
    eye_mid_y = 0.5 * (re[1] + le[1])
    mouth_mid_x = 0.5 * (rm[0] + lm_pt[0])
    mouth_mid_y = 0.5 * (rm[1] + lm_pt[1])

    # Bilateral distances to midline (normalized by face width).
    re_dx = abs(re[0] - eye_mid_x) / face_w
    le_dx = abs(le[0] - eye_mid_x) / face_w
    rm_dx = abs(rm[0] - mouth_mid_x) / face_w
    lm_dx = abs(lm_pt[0] - mouth_mid_x) / face_w
    symmetry_error = abs(re_dx - le_dx) + abs(rm_dx - lm_dx)

    eye_height_asym = abs(re[1] - le[1]) / face_h
    mouth_height_asym = abs(rm[1] - lm_pt[1]) / face_h
    nose_midline_offset = abs(nose[0] - eye_mid_x) / face_w

    # Facial thirds proxies: brow≈eye line, mid=nose, lower=chin≈bbox bottom.
    top_y = float(y)
    brow_y = eye_mid_y
    mid_y = float(nose[1])
    chin_y = float(y + bh)
    total = max(chin_y - top_y, 1.0)
    thirds_upper = _safe_div(brow_y - top_y, total)
    thirds_mid = _safe_div(mid_y - brow_y, total)
    thirds_lower = _safe_div(chin_y - mid_y, total)
    # Distance from equal thirds (1/3 each); lower is better when converted in scoring.
    target = 1.0 / 3.0
    thirds_balance = (
        abs(thirds_upper - target)
        + abs(thirds_mid - target)
        + abs(thirds_lower - target)
    )

    interocular = float(np.linalg.norm(le - re))
    eye_span = abs(le[0] - re[0])
    mouth_w = float(np.linalg.norm(lm_pt - rm))

    features: Dict[str, float] = {
        "symmetry_error": float(symmetry_error),
        "eye_height_asym": float(eye_height_asym),
        "mouth_height_asym": float(mouth_height_asym),
        "nose_midline_offset": float(nose_midline_offset),
        "thirds_upper": float(thirds_upper),
        "thirds_mid": float(thirds_mid),
        "thirds_lower": float(thirds_lower),
        "thirds_balance": float(thirds_balance),
        "interocular_face_ratio": _safe_div(interocular, face_w),
        "eye_span_face_ratio": _safe_div(eye_span, face_w),
        "mouth_face_ratio": _safe_div(mouth_w, face_w),
        "nose_face_y_ratio": _safe_div(nose[1] - top_y, face_h),
        "face_aspect": _safe_div(face_h, face_w),
    }

    vector = np.array([features[n] for n in FEATURE_NAMES], dtype=np.float32)
    return vector, features
