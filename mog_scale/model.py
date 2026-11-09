"""Hybrid fusion network: CNN image branch + geometry MLP."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from mog_scale.geometry import FEATURE_NAMES

GEO_DIM = len(FEATURE_NAMES)
IMAGE_SIZE = 224


class HybridFusionNet(nn.Module):
    """
    Multi-input network:

    - Image → small CNN → embedding
    - Geometry vector → MLP → embedding
    - Concat → MLP → scalar score (trained on 0–100 labels)
    """

    def __init__(
        self,
        geo_dim: int = GEO_DIM,
        img_embed: int = 64,
        geo_embed: int = 32,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        self.geo_dim = geo_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, img_embed),
            nn.ReLU(inplace=True),
        )

        self.geo_mlp = nn.Sequential(
            nn.Linear(geo_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, geo_embed),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Linear(img_embed + geo_embed, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, image: torch.Tensor, geo: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        image : (B, 3, H, W) float tensor in [0, 1]
        geo : (B, geo_dim) float tensor

        Returns
        -------
        (B, 1) predicted score
        """
        img_emb = self.cnn(image)
        geo_emb = self.geo_mlp(geo)
        fused = torch.cat([img_emb, geo_emb], dim=1)
        return self.head(fused)


def preprocess_face_bgr(
    crop_bgr: np.ndarray,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """BGR uint8 HxWx3 crop → float tensor (1, 3, H, W) RGB in [0, 1]."""
    import cv2

    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[0] != IMAGE_SIZE or rgb.shape[1] != IMAGE_SIZE:
        rgb = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
    t = t.unsqueeze(0)
    if device is not None:
        t = t.to(device)
    return t


def geo_to_tensor(
    geo_vector: np.ndarray,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    arr = np.asarray(geo_vector, dtype=np.float32).reshape(1, -1)
    t = torch.from_numpy(arr)
    if device is not None:
        t = t.to(device)
    return t


def clamp_score(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return float(max(lo, min(hi, value)))


def load_fusion_checkpoint(
    path: Union[str, Path],
    device: Optional[torch.device] = None,
) -> Tuple[HybridFusionNet, dict]:
    """Load model weights; returns (model.eval(), checkpoint meta dict)."""
    device = device or torch.device("cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    geo_dim = int(ckpt.get("geo_dim", GEO_DIM)) if isinstance(ckpt, dict) else GEO_DIM
    model = HybridFusionNet(geo_dim=geo_dim)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    meta = ckpt if isinstance(ckpt, dict) else {}
    return model, meta


def save_fusion_checkpoint(
    model: HybridFusionNet,
    path: Union[str, Path],
    **extra,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "geo_dim": model.geo_dim,
        "feature_names": list(FEATURE_NAMES),
        **extra,
    }
    torch.save(payload, path)
