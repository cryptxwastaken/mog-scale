"""Image I/O helpers that work with non-ASCII paths on Windows."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np


def imread_bgr(path: Union[str, Path]) -> Optional[np.ndarray]:
    """
    Read an image as BGR uint8.

    Uses ``np.fromfile`` + ``cv2.imdecode`` so Unicode filenames work on Windows,
    where ``cv2.imread`` often fails.
    """
    path = Path(path)
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)
