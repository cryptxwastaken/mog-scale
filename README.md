# mog-scale

Real-time facial feature and symmetry analyzer from a webcam feed:

- **OpenCV** — camera capture, face detection, landmarks
- **NumPy** — geometric symmetry / ratio feature vectors
- **Pandas** — session feature logging
- **PyTorch** — optional hybrid fusion (CNN pixels + geometry), trained on MEBeauty

**Score** on the HUD is always rule-based geometry (0–100). **Mog Score** appears only when you load a trained checkpoint.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

YuNet ONNX weights download automatically into `models/` on first run.

## Live camera

```bash
python main.py live
python main.py live --camera 0 --log
python main.py live --checkpoint checkpoints/fusion.pt
```

Keys while the window is open:

| Key | Action              |
|-----|---------------------|
| `q` | Quit                |
| `s` | Toggle CSV logging  |
| `c` | Save a face crop    |

## Train hybrid fusion (MEBeauty)

[MEBeauty](https://github.com/fbplab/MEBeauty-database) is **non-commercial research only**. Do not redistribute faces or use this for commercial beauty scoring.

```bash
# once, outside this repo
cd ..
git clone https://github.com/fbplab/MEBeauty-database.git

# back in mog-scale
python scripts/prepare_mebeauty.py --mebeauty-root ..\MEBeauty-database
python scripts/train_fusion.py
python main.py live --checkpoint checkpoints/fusion.pt
```

`prepare_mebeauty.py` runs YuNet + geometry on each scored crop and writes `data/mebeauty/{train,val,test}.csv` with labels scaled to 0–100 (`raw_score * 10`).  
`train_fusion.py` fine-tunes `HybridFusionNet` (pretrained ResNet18 + geometry MLP) with augmentation, cosine LR, and early stopping. The best val checkpoint is saved to `checkpoints/fusion.pt`.

### How to read the metrics

- **baseline** — MSE if you always predict the train-set mean rating. Your model should beat this.
- **val_mse / val_rmse** — error on held-out validation (lower is better). RMSE is on the same 0–100 scale as Mog Score.
- **val_r** — Pearson correlation between predictions and true labels (−1…1). Higher is better; this is the main quality signal.
- After training, a one-shot **test** line reports the same metrics on `test.csv`.

## Layout

```
main.py                      CLI entry
mog_scale/                   capture, landmarks, geometry, scoring, model, logging, visualize
scripts/prepare_mebeauty.py  Build geo+label CSVs from MEBeauty
scripts/train_fusion.py      Train HybridFusionNet
data/logs/                   Session CSV feature logs
data/crops/                  Optional face crops
data/mebeauty/               Prepared train/val/test CSVs (gitignored)
checkpoints/                 fusion.pt (gitignored)
models/                      YuNet ONNX (auto-downloaded)
```
