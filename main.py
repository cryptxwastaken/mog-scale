"""CLI entry for mog-scale live facial symmetry analyzer."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import torch

from mog_scale.capture import CameraCapture, crop_face
from mog_scale.geometry import extract_geometry
from mog_scale.landmarks import FaceLandmarkDetector
from mog_scale.logging_io import FeatureLogger
from mog_scale.model import (
    HybridFusionNet,
    clamp_score,
    geo_to_tensor,
    load_fusion_checkpoint,
    preprocess_face_bgr,
)
from mog_scale.scoring import score_geometry
from mog_scale.visualize import draw_face, draw_no_face


def run_live(args: argparse.Namespace) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    fusion: HybridFusionNet | None = None
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
            return 1
        fusion, _ = load_fusion_checkpoint(ckpt_path, device=device)
        print(f"Loaded fusion checkpoint: {ckpt_path} ({device})")

    detector = FaceLandmarkDetector()
    logger = FeatureLogger(enabled=args.log)
    if args.log:
        print(f"Logging to {logger.csv_path}")

    window = "mog-scale"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    fps = 0.0
    t_prev = time.perf_counter()
    frames = 0

    try:
        with CameraCapture(camera_index=args.camera, width=args.width, height=args.height) as cam:
            while True:
                ok, frame = cam.read()
                if not ok or frame is None:
                    print("Failed to read frame", file=sys.stderr)
                    break

                face = detector.detect(frame)
                fusion_score = None
                pending = None

                if face is None:
                    display = draw_no_face(
                        frame, fps=fps, logging_on=logger.enabled, fusion_score=None
                    )
                else:
                    geo_vec, features = extract_geometry(face)
                    breakdown = score_geometry(features)
                    face_crop = crop_face(frame, face.bbox)

                    if fusion is not None:
                        with torch.no_grad():
                            img_t = preprocess_face_bgr(face_crop, device=device)
                            geo_t = geo_to_tensor(geo_vec, device=device)
                            pred = fusion(img_t, geo_t).item()
                            fusion_score = clamp_score(float(pred))

                    display = draw_face(
                        frame,
                        face,
                        score=breakdown,
                        fusion_score=fusion_score,
                        fps=fps,
                        logging_on=logger.enabled,
                    )
                    pending = {
                        "features": features,
                        "breakdown": breakdown,
                        "face": face,
                        "crop": face_crop,
                    }

                frames += 1
                now = time.perf_counter()
                dt = now - t_prev
                if dt >= 0.5:
                    fps = frames / dt
                    frames = 0
                    t_prev = now

                cv2.imshow(window, display)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                if key == ord("s"):
                    on = logger.toggle()
                    print(f"Logging {'ON -> ' + str(logger.csv_path) if on else 'OFF'}")
                if key == ord("c") and pending is not None:
                    path = logger.log(
                        features=pending["features"],
                        rule_score=pending["breakdown"].total,
                        components=pending["breakdown"].components,
                        face_score=pending["face"].score,
                        crop_bgr=pending["crop"],
                        save_crop=True,
                    )
                    print(f"Saved crop: {path}")
                elif pending is not None and logger.enabled:
                    logger.log(
                        features=pending["features"],
                        rule_score=pending["breakdown"].total,
                        components=pending["breakdown"].components,
                        face_score=pending["face"].score,
                        crop_bgr=pending["crop"],
                        save_crop=args.save_crops,
                    )
    finally:
        logger.close()
        cv2.destroyAllWindows()

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="mog-scale facial symmetry analyzer",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    live = sub.add_parser("live", help="Open webcam and score faces in real time")
    live.add_argument("--camera", type=int, default=0, help="Camera index")
    live.add_argument("--width", type=int, default=640)
    live.add_argument("--height", type=int, default=480)
    live.add_argument("--log", action="store_true", help="Start with CSV logging enabled")
    live.add_argument(
        "--save-crops",
        action="store_true",
        help="When logging, also write face crops each frame",
    )
    live.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="HybridFusionNet checkpoint (shows Mog Score on HUD)",
    )
    live.add_argument(
        "--cuda",
        action="store_true",
        help="Run fusion inference on CUDA if available",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "live":
        return run_live(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
