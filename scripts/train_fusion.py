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
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mog_scale.geometry import FEATURE_NAMES
from mog_scale.image_io import imread_bgr
from mog_scale.model import HybridFusionNet, IMAGE_SIZE, load_fusion_checkpoint, save_fusion_checkpoint


class FusionDataset(Dataset):
    def __init__(self, csv_path: Path, augment: bool = False) -> None:
        self.df = pd.read_csv(csv_path)
        if "label" not in self.df.columns:
            raise ValueError(f"{csv_path} missing 'label' column")
        for name in FEATURE_NAMES:
            if name not in self.df.columns:
                raise ValueError(f"{csv_path} missing geometry column '{name}'")
        if "crop_path" not in self.df.columns:
            raise ValueError(f"{csv_path} missing 'crop_path' column")
        exists = self.df["crop_path"].map(lambda p: Path(str(p)).exists())
        missing = int((~exists).sum())
        if missing:
            print(f"Warning: dropping {missing} rows with missing crops in {csv_path.name}")
        self.df = self.df[exists].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No usable rows in {csv_path}")

        if augment:
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.15, contrast=0.15),
                    transforms.ToTensor(),  # → [0, 1]
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                ]
            )

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
        image = self.transform(rgb)
        geo = np.array([float(row[n]) for n in FEATURE_NAMES], dtype=np.float32)
        geo_t = torch.from_numpy(geo)
        label_t = torch.tensor([float(row["label"])], dtype=torch.float32)
        return image, geo_t, label_t


def compute_geo_stats(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    vals = df[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    mean = vals.mean(axis=0)
    std = vals.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def run_epoch(
    model: HybridFusionNet,
    loader: DataLoader,
    loss_fn: nn.Module,
    opt: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float, float]:
    """Returns (mse, rmse, pearson). Pearson is 0.0 if undefined."""
    train = opt is not None
    model.train(train)
    total_loss = 0.0
    n = 0
    preds: list[float] = []
    targets: list[float] = []
    for images, geos, labels in loader:
        images = images.to(device)
        geos = geos.to(device)
        labels = labels.to(device)
        if train:
            assert opt is not None
            opt.zero_grad()
        pred = model(images, geos)
        loss = loss_fn(pred, labels)
        if train:
            loss.backward()
            opt.step()
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
        preds.extend(pred.detach().cpu().view(-1).tolist())
        targets.extend(labels.detach().cpu().view(-1).tolist())

    mse = total_loss / max(n, 1)
    rmse = float(np.sqrt(mse))
    if n >= 2:
        p = np.corrcoef(np.asarray(preds), np.asarray(targets))[0, 1]
        pearson = float(p) if np.isfinite(p) else 0.0
    else:
        pearson = 0.0
    return mse, rmse, pearson


def mean_guess_mse(csv_path: Path, mean_label: float) -> float:
    df = pd.read_csv(csv_path)
    labels = df["label"].to_numpy(dtype=np.float64)
    return float(np.mean((labels - mean_label) ** 2))


def evaluate(
    model: HybridFusionNet,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    with torch.no_grad():
        return run_epoch(model, loader, loss_fn, None, device)


def train(args: argparse.Namespace) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Device: {device}")

    train_csv = Path(args.train_csv)
    val_csv = Path(args.val_csv)
    test_csv = Path(args.test_csv)
    if not train_csv.exists():
        print(f"Missing train CSV: {train_csv}", file=sys.stderr)
        print("Run scripts/prepare_mebeauty.py first.", file=sys.stderr)
        return 1
    if not val_csv.exists():
        print(f"Missing val CSV: {val_csv}", file=sys.stderr)
        return 1

    train_ds = FusionDataset(train_csv, augment=True)
    val_ds = FusionDataset(val_csv, augment=False)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    print(f"Train {len(train_ds)} | Val {len(val_ds)}")

    train_label_mean = float(pd.read_csv(train_csv)["label"].mean())
    baseline_val = mean_guess_mse(val_csv, train_label_mean)
    print(f"Mean-guess baseline val MSE={baseline_val:.4f} (predict {train_label_mean:.2f})")

    geo_mean, geo_std = compute_geo_stats(train_csv)
    model = HybridFusionNet(pretrained=True).to(device)
    model.set_geo_stats(geo_mean, geo_std)

    backbone_params = list(model.backbone.parameters())
    head_params = (
        list(model.geo_mlp.parameters())
        + list(model.head.parameters())
        + list(model.img_dropout.parameters())
    )
    opt = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.backbone_lr},
            {"params": head_params, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    epochs_no_improve = 0
    out = Path(args.out)

    for epoch in range(1, args.epochs + 1):
        train_mse, train_rmse, train_r = run_epoch(
            model, train_loader, loss_fn, opt, device
        )
        val_mse, val_rmse, val_r = evaluate(model, val_loader, loss_fn, device)
        scheduler.step()
        print(
            f"epoch {epoch}/{args.epochs}  "
            f"train_mse={train_mse:.4f}  val_mse={val_mse:.4f}  "
            f"val_rmse={val_rmse:.4f}  val_r={val_r:.3f}  "
            f"baseline={baseline_val:.4f}"
        )
        if val_mse < best_val - 1e-4:
            best_val = val_mse
            epochs_no_improve = 0
            save_fusion_checkpoint(
                model,
                out,
                label_col="label",
                epochs=epoch,
                best_val_mse=best_val,
                baseline_val_mse=baseline_val,
                val_rmse=val_rmse,
                val_pearson=val_r,
                note="HybridFusionNet (ResNet18) trained on MEBeauty (research/non-commercial).",
            )
            print(f"  saved best -> {out}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stop after {args.patience} epochs without val improvement.")
                break

    print(f"Done. Best val MSE={best_val:.4f} (baseline {baseline_val:.4f})")

    if test_csv.exists():
        model, _ = load_fusion_checkpoint(out, device=device)
        test_ds = FusionDataset(test_csv, augment=False)
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
        )
        baseline_test = mean_guess_mse(test_csv, train_label_mean)
        test_mse, test_rmse, test_r = evaluate(model, test_loader, loss_fn, device)
        print(
            f"Test MSE={test_mse:.4f}  RMSE={test_rmse:.4f}  r={test_r:.3f}  "
            f"baseline={baseline_test:.4f}"
        )
    else:
        print(f"No test CSV at {test_csv}; skipping test eval.")

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
    p.add_argument(
        "--test-csv",
        default=str(ROOT / "data" / "mebeauty" / "test.csv"),
    )
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=5, help="Early-stop patience")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3, help="LR for geo MLP + head")
    p.add_argument(
        "--backbone-lr",
        type=float,
        default=1e-4,
        help="LR for ResNet18 backbone",
    )
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument(
        "--out",
        default=str(ROOT / "checkpoints" / "fusion.pt"),
    )
    p.add_argument("--cuda", action="store_true")
    return train(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
