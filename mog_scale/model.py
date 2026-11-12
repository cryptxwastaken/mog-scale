"""Hybrid fusion network: pretrained ResNet18 image branch + geometry MLP."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

from mog_scale.geometry import FEATURE_NAMES

GEO_DIM = len(FEATURE_NAMES)
IMAGE_SIZE = 224
IMG_EMBED = 512
GEO_EMBED = 32
HIDDEN = 64

# ImageNet normalization (applied inside forward on [0, 1] tensors).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class HybridFusionNet(nn.Module):
    """
    Multi-input network:

    - Image → pretrained ResNet18 → 512-D embedding
    - Geometry vector (z-scored) → MLP → embedding
    - Concat → MLP → scalar score (trained on 0–100 labels)
    """

    def __init__(
        self,
        geo_dim: int = GEO_DIM,
        pretrained: bool = True,
        geo_embed: int = GEO_EMBED,
        hidden: int = HIDDEN,
    ) -> None:
        super().__init__()
        self.geo_dim = geo_dim

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # → (B, 512, 1, 1)
        self.img_dropout = nn.Dropout(0.3)

        self.geo_mlp = nn.Sequential(
            nn.Linear(geo_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, geo_embed),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Linear(IMG_EMBED + geo_embed, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )

        mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("img_mean", mean)
        self.register_buffer("img_std", std)
        self.register_buffer("geo_mean", torch.zeros(geo_dim))
        self.register_buffer("geo_std", torch.ones(geo_dim))

    def set_geo_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        """Set train-split geometry mean/std for z-score standardization."""
        mean_t = torch.as_tensor(mean, dtype=torch.float32).reshape(-1)
        std_t = torch.as_tensor(std, dtype=torch.float32).reshape(-1)
        if mean_t.numel() != self.geo_dim or std_t.numel() != self.geo_dim:
            raise ValueError(
                f"geo stats must have length {self.geo_dim}, "
                f"got mean={mean_t.numel()} std={std_t.numel()}"
            )
        std_t = torch.clamp(std_t, min=1e-6)
        self.geo_mean.copy_(mean_t)
        self.geo_std.copy_(std_t)

    def forward(self, image: torch.Tensor, geo: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        image : (B, 3, H, W) float tensor in [0, 1]
        geo : (B, geo_dim) float tensor (raw geometry; standardized inside)

        Returns
        -------
        (B, 1) predicted score
        """
        x = (image - self.img_mean) / self.img_std
        img_emb = self.backbone(x).flatten(1)
        img_emb = self.img_dropout(img_emb)

        geo_z = (geo - self.geo_mean) / self.geo_std
        geo_emb = self.geo_mlp(geo_z)
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
    # pretrained=False: do not re-download ImageNet weights; load checkpoint state.
    model = HybridFusionNet(geo_dim=geo_dim, pretrained=False)
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
