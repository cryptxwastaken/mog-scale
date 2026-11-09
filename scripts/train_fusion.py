"""
Train HybridFusionNet on prepared MEBeauty CSVs (image + geometry -> label).

Research / non-commercial use of MEBeauty labels only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mog_scale.geometry import FEATURE_NAMES
from mog_scale.image_io import imread_bgr
from mog_scale.model import HybridFusionNet, IMAGE_SIZE, save_fusion_checkpoint


class FusionDataset(Dataset):
    def __init__(self, csv_path: Path) -> None:
        self.df = pd.read_csv(csv_path)
        if "label" not in self.df.columns:
            raise ValueError(f"{csv_path} missing 'label' column")
        for name in FEATURE_NAMES:
            if name not in self.df.columns:
                raise ValueError(f"{csv_path} missing geometry column '{name}'")
        if "crop_path" not in self.df.columns:
            raise ValueError(f"{csv_path} missing 'crop_path' column")
        # Drop rows with missing images up front.
        exists = self.df["crop_path"].map(lambda p: Path(str(p)).exists())
        missing = int((~exists).sum())
        if missing:
            print(f"Warning: dropping {missing} rows with missing crops in {csv_path.name}")
        self.df = self.df[exists].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No usable rows in {csv_path}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        bgr = imread_bgr(str(row["crop_path"]))
        if bgr is None:
            bgr = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[0] != IMAGE_SIZE or rgb.shape[1] != IMAGE_SIZE:
            rgb = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        image = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        geo = np.array([float(row[n]) for n in FEATURE_NAMES], dtype=np.float32)
        geo_t = torch.from_numpy(geo)
        label_t = torch.tensor([float(row["label"])], dtype=torch.float32)
        return image, geo_t, label_t


def run_epoch(
    model: HybridFusionNet,
    loader: DataLoader,
    loss_fn: nn.Module,
    opt: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    train = opt is not None
    model.train(train)
    total = 0.0
    n = 0
    for images, geos, labels in loader:
        images = images.to(device)
        geos = geos.to(device)
        labels = labels.to(device)
        if train:
            opt.zero_grad()
        pred = model(images, geos)
        loss = loss_fn(pred, labels)
        if train:
            loss.backward()
            opt.step()
        total += loss.item() * images.size(0)
        n += images.size(0)
    return total / max(n, 1)


def train(args: argparse.Namespace) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Device: {device}")

    train_csv = Path(args.train_csv)
    val_csv = Path(args.val_csv)
    if not train_csv.exists():
        print(f"Missing train CSV: {train_csv}", file=sys.stderr)
        print("Run scripts/prepare_mebeauty.py first.", file=sys.stderr)
        return 1
    if not val_csv.exists():
        print(f"Missing val CSV: {val_csv}", file=sys.stderr)
        return 1

    train_ds = FusionDataset(train_csv)
    val_ds = FusionDataset(val_csv)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    print(f"Train {len(train_ds)} | Val {len(val_ds)}")

    model = HybridFusionNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    out = Path(args.out)
    for epoch in range(1, args.epochs + 1):
        train_mse = run_epoch(model, train_loader, loss_fn, opt, device)
        val_mse = run_epoch(model, val_loader, loss_fn, None, device)
        print(f"epoch {epoch}/{args.epochs}  train_mse={train_mse:.4f}  val_mse={val_mse:.4f}")
        if val_mse < best_val:
            best_val = val_mse
            save_fusion_checkpoint(
                model,
                out,
                label_col="label",
                epochs=epoch,
                best_val_mse=best_val,
                note="HybridFusionNet trained on MEBeauty (research/non-commercial).",
            )
            print(f"  saved best -> {out}")

    print(f"Done. Best val MSE={best_val:.4f}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Train HybridFusionNet on MEBeauty CSVs")
    p.add_argument(
        "--train-csv",
        default=str(ROOT / "data" / "mebeauty" / "train.csv"),
    )
    p.add_argument(
        "--val-csv",
        default=str(ROOT / "data" / "mebeauty" / "val.csv"),
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--out",
        default=str(ROOT / "checkpoints" / "fusion.pt"),
    )
    p.add_argument("--cuda", action="store_true")
    return train(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
