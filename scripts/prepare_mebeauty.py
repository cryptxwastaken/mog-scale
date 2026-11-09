"""
Build train/val/test CSVs from MEBeauty score lists + YuNet geometry.

MEBeauty is non-commercial research only:
https://github.com/fbplab/MEBeauty-database

Labels are scaled score * 10 into 0-100 to match the live HUD.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mog_scale.geometry import FEATURE_NAMES, extract_geometry
from mog_scale.image_io import imread_bgr
from mog_scale.landmarks import FaceLandmarkDetector

SPLITS = {
    "train": "scores/train_2022.txt",
    "val": "scores/val_2022.txt",
    "test": "scores/test_2022.txt",
}


def parse_score_file(path: Path) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        # "relative/path.jpg 4.56" — path may contain spaces rarely; split from right.
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            continue
        rel, score_s = parts
        try:
            score = float(score_s)
        except ValueError:
            continue
        rows.append((rel.lstrip("./"), score))
    return rows


def resolve_image(mebeauty_root: Path, rel: str) -> Path | None:
    cand = mebeauty_root / rel
    if cand.exists():
        return cand
    # Some checkouts omit leading dirs; try basename search under cropped_images.
    name = Path(rel).name
    cropped = mebeauty_root / "cropped_images"
    if cropped.exists():
        hits = list(cropped.rglob(name))
        if hits:
            return hits[0]
    return None


def prepare_split(
    mebeauty_root: Path,
    split: str,
    score_rel: str,
    out_csv: Path,
    detector: FaceLandmarkDetector,
) -> tuple[int, int]:
    score_path = mebeauty_root / score_rel
    if not score_path.exists():
        raise FileNotFoundError(f"Missing score file: {score_path}")

    entries = parse_score_file(score_path)
    rows: list[dict] = []
    skipped = 0

    for i, (rel, score) in enumerate(entries, start=1):
        img_path = resolve_image(mebeauty_root, rel)
        if img_path is None:
            skipped += 1
            continue
        bgr = imread_bgr(img_path)
        if bgr is None:
            skipped += 1
            continue
        face = detector.detect(bgr)
        if face is None:
            skipped += 1
            continue
        geo_vec, _features = extract_geometry(face)
        label = float(max(0.0, min(100.0, score * 10.0)))
        row = {
            "crop_path": str(img_path.resolve()),
            "mebeauty_rel": rel,
            "raw_score": score,
            "label": label,
            "face_det_score": face.score,
        }
        for name, val in zip(FEATURE_NAMES, geo_vec.tolist()):
            row[name] = val
        rows.append(row)

        if i % 100 == 0:
            print(f"  [{split}] {i}/{len(entries)} processed ({len(rows)} ok, {skipped} skipped)")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(rows)} rows, {skipped} skipped)")
    return len(rows), skipped


def main() -> int:
    p = argparse.ArgumentParser(description="Prepare MEBeauty CSVs with geometry features")
    p.add_argument(
        "--mebeauty-root",
        type=Path,
        required=True,
        help="Path to cloned MEBeauty-database repo",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "mebeauty",
        help="Directory for train.csv / val.csv / test.csv",
    )
    p.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="YuNet detection score threshold",
    )
    args = p.parse_args()

    root = args.mebeauty_root.resolve()
    if not root.exists():
        print(f"MEBeauty root not found: {root}", file=sys.stderr)
        return 1

    detector = FaceLandmarkDetector(score_threshold=args.score_threshold)
    total_ok = 0
    total_skip = 0
    for split, score_rel in SPLITS.items():
        ok, skip = prepare_split(
            root,
            split,
            score_rel,
            args.out_dir / f"{split}.csv",
            detector,
        )
        total_ok += ok
        total_skip += skip

    print(f"Done. {total_ok} samples, {total_skip} skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
