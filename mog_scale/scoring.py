"""Rule-based geometric facial symmetry / proportion score (0–100)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

# Canonical targets (heuristic, not clinical).
TARGET_INTEROCULAR = 0.36
TARGET_MOUTH_RATIO = 0.40
TARGET_EYE_SPAN = 0.38
TARGET_NOSE_Y = 0.55
TARGET_ASPECT = 1.30


@dataclass
class ScoreBreakdown:
    total: float
    symmetry: float
    thirds: float
    proportions: float
    components: Dict[str, float]


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return float(max(lo, min(hi, v)))


def _ratio_score(value: float, target: float, tolerance: float) -> float:
    """100 when value==target; falls toward 0 as error exceeds tolerance."""
    err = abs(value - target)
    return _clamp(100.0 * (1.0 - err / max(tolerance, 1e-6)))


def score_geometry(features: Dict[str, float]) -> ScoreBreakdown:
    """Map geometry features to a 0–100 rule score with component breakdown."""
    # Symmetry: lower error → higher score.
    sym_pen = (
        120.0 * features.get("symmetry_error", 0.0)
        + 80.0 * features.get("eye_height_asym", 0.0)
        + 80.0 * features.get("mouth_height_asym", 0.0)
        + 100.0 * features.get("nose_midline_offset", 0.0)
    )
    symmetry = _clamp(100.0 - sym_pen)

    # Equal thirds: thirds_balance is sum of abs deviations from 1/3 (max ~2).
    thirds = _clamp(100.0 - 80.0 * features.get("thirds_balance", 0.0))

    # Proportions vs soft targets.
    prop_scores = {
        "interocular": _ratio_score(
            features.get("interocular_face_ratio", TARGET_INTEROCULAR),
            TARGET_INTEROCULAR,
            0.20,
        ),
        "eye_span": _ratio_score(
            features.get("eye_span_face_ratio", TARGET_EYE_SPAN),
            TARGET_EYE_SPAN,
            0.20,
        ),
        "mouth": _ratio_score(
            features.get("mouth_face_ratio", TARGET_MOUTH_RATIO),
            TARGET_MOUTH_RATIO,
            0.25,
        ),
        "nose_y": _ratio_score(
            features.get("nose_face_y_ratio", TARGET_NOSE_Y),
            TARGET_NOSE_Y,
            0.25,
        ),
        "aspect": _ratio_score(
            features.get("face_aspect", TARGET_ASPECT),
            TARGET_ASPECT,
            0.50,
        ),
    }
    proportions = float(np.mean(list(prop_scores.values())))

    # Weighted blend.
    total = _clamp(0.45 * symmetry + 0.25 * thirds + 0.30 * proportions)

    components = {
        "symmetry": symmetry,
        "thirds": thirds,
        "proportions": proportions,
        **{f"prop_{k}": v for k, v in prop_scores.items()},
    }
    return ScoreBreakdown(
        total=total,
        symmetry=symmetry,
        thirds=thirds,
        proportions=proportions,
        components=components,
    )
