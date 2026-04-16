#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_demo.scripts._common import ensure_workdirs, write_rgb_video, WORK_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract teacher foreground with simple white-background thresholding.")
    parser.add_argument("--input", default="teacher_demo/work/wanmove_outputs/teacher_motion.mp4")
    parser.add_argument("--frames-dir", default="teacher_demo/work/foreground/frames_rgba")
    parser.add_argument("--preview-video", default="teacher_demo/work/foreground/foreground_preview.mp4")
    parser.add_argument("--white-threshold", type=int, default=245)
    parser.add_argument("--sat-threshold", type=int, default=25)
    parser.add_argument("--blur", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_workdirs()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Generated teacher video not found: {input_path}")

    frames_dir = Path(args.frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    preview_frames: list[np.ndarray] = []

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
    frame_idx = 0

    print(f"[04] Extracting foreground from {input_path}")
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        value = hsv[..., 2]
        sat = hsv[..., 1]
        near_white = (value >= args.white_threshold) & (sat <= args.sat_threshold)
        alpha = np.where(near_white, 0, 255).astype(np.uint8)

        kernel = np.ones((3, 3), np.uint8)
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel)
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)
        if args.blur > 0:
            blur_size = args.blur if args.blur % 2 == 1 else args.blur + 1
            alpha = cv2.GaussianBlur(alpha, (blur_size, blur_size), 0)

        rgba = np.dstack([frame_rgb, alpha])
        Image.fromarray(rgba, mode="RGBA").save(frames_dir / f"frame_{frame_idx:04d}.png")

        checker = np.full_like(frame_rgb, 220)
        checker[::2, ::2] = 245
        checker[1::2, 1::2] = 245
        alpha_f = alpha.astype(np.float32)[..., None] / 255.0
        preview = np.clip(frame_rgb.astype(np.float32) * alpha_f + checker.astype(np.float32) * (1.0 - alpha_f), 0, 255).astype(np.uint8)
        preview_frames.append(preview)
        frame_idx += 1

    cap.release()
    if not preview_frames:
        raise RuntimeError("No frames were decoded from the generated teacher video.")

    write_rgb_video(args.preview_video, preview_frames, fps)
    print(f"[04] Saved {frame_idx} RGBA frames to {frames_dir}")
    print(f"[04] Saved preview video to {args.preview_video}")


if __name__ == "__main__":
    main()

