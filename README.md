# mog-scale

Real-time facial feature and symmetry analyzer from a webcam feed:

- **OpenCV** — camera capture, face detection, landmarks
- **NumPy** — geometric symmetry / ratio feature vectors
- **Pandas** — session feature logging for later analysis

Scores faces with transparent **rule-based geometric heuristics** (0–100): symmetry, facial thirds, and proportions.

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
```

Keys while the window is open:

| Key | Action              |
|-----|---------------------|
| `q` | Quit                |
| `s` | Toggle CSV logging  |
| `c` | Save a face crop    |

## Layout

```
main.py                 CLI entry
mog_scale/              capture, landmarks, geometry, scoring, logging, visualize
data/logs/              CSV feature logs
data/crops/             Optional face crops
models/                 YuNet ONNX (auto-downloaded)
```
