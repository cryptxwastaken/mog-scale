"""Pandas CSV (+ optional crop) session logging."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import cv2
import pandas as pd

from mog_scale.geometry import FEATURE_NAMES

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_DIR = ROOT / "data" / "logs"
DEFAULT_CROP_DIR = ROOT / "data" / "crops"


class FeatureLogger:
    """Append geometry features and scores to a session CSV."""

    def __init__(
        self,
        log_dir: Path = DEFAULT_LOG_DIR,
        crop_dir: Path = DEFAULT_CROP_DIR,
        enabled: bool = False,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.crop_dir = Path(crop_dir)
        self.enabled = enabled
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.crop_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.log_dir / f"session_{stamp}.csv"
        self._rows: list[Dict] = []
        self._frame_id = 0

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        return self.enabled

    def log(
        self,
        features: Dict[str, float],
        rule_score: float,
        components: Dict[str, float],
        face_score: float,
        crop_bgr=None,
        save_crop: bool = False,
    ) -> Optional[str]:
        """Append one row; optionally write a face crop. Returns crop path if saved."""
        if not self.enabled and not save_crop:
            return None

        self._frame_id += 1
        crop_path: Optional[str] = None
        if save_crop and crop_bgr is not None:
            name = f"crop_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._frame_id}.jpg"
            out = self.crop_dir / name
            cv2.imwrite(str(out), crop_bgr)
            crop_path = str(out)

        if not self.enabled:
            return crop_path

        row: Dict = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "frame_id": self._frame_id,
            "face_det_score": face_score,
            "rule_score": rule_score,
            "symmetry": components.get("symmetry"),
            "thirds": components.get("thirds"),
            "proportions": components.get("proportions"),
            "crop_path": crop_path or "",
        }
        for name in FEATURE_NAMES:
            row[name] = features.get(name)
        self._rows.append(row)
        self._flush()
        return crop_path

    def _flush(self) -> None:
        if not self._rows:
            return
        df = pd.DataFrame(self._rows)
        df.to_csv(self.csv_path, index=False)

    def close(self) -> None:
        self._flush()
