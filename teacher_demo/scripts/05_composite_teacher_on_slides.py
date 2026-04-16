#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_demo.scripts._common import alpha_blend, ensure_workdirs, ffmpeg_available, load_config, repo_path, WORK_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Composite RGBA teacher frames onto the slides video.")
    parser.add_argument("--config", default="teacher_demo/configs/demo_config.yaml")
    parser.add_argument("--foreground-dir", default="teacher_demo/work/foreground/frames_rgba")
    parser.add_argument("--output", default="teacher_demo/work/composite/final_demo.mp4")
    parser.add_argument("--mode", choices=["loop", "stretch"], default="loop")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_workdirs()
    config = load_config(args.config)
    bg_video = repo_path(config["input_video"])
    if not bg_video.exists():
        raise FileNotFoundError(f"Background lecture video not found: {bg_video}")

    foreground_dir = Path(args.foreground_dir)
    frame_paths = sorted(foreground_dir.glob("frame_*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"No RGBA frames found in {foreground_dir}")

    cap = cv2.VideoCapture(str(bg_video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {bg_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    overlay_cfg = config["overlay"]
    target_height = max(1, int(height * float(overlay_cfg["height_ratio"])))
    margin = int(height * float(overlay_cfg["margin_ratio"]))
    position = overlay_cfg["position"]

    teacher_frames = []
    for frame_path in frame_paths:
        rgba = np.array(Image.open(frame_path).convert("RGBA"))
        scale = target_height / rgba.shape[0]
        target_width = max(1, int(round(rgba.shape[1] * scale)))
        resized = cv2.resize(rgba, (target_width, target_height), interpolation=cv2.INTER_AREA)
        teacher_frames.append(resized)

    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_video = output_dir / "final_demo_no_audio.mp4"
    writer = cv2.VideoWriter(
        str(temp_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open output writer: {temp_video}")

    print(f"[05] Compositing {len(teacher_frames)} teacher frames onto {bg_video}")
    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        bg_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if args.mode == "stretch":
            teacher_idx = min(len(teacher_frames) - 1, int(frame_idx / max(frame_count - 1, 1) * (len(teacher_frames) - 1)))
        else:
            teacher_idx = frame_idx % len(teacher_frames)
        teacher_rgba = teacher_frames[teacher_idx]

        if position == "bottom_left":
            x = margin
        else:
            x = width - teacher_rgba.shape[1] - margin
        y = height - teacher_rgba.shape[0] - margin

        composited = alpha_blend(bg_rgb, teacher_rgba, x, y)
        writer.write(cv2.cvtColor(composited, cv2.COLOR_RGB2BGR))
        frame_idx += 1

    cap.release()
    writer.release()

    final_output = Path(args.output)
    if ffmpeg_available():
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(temp_video),
            "-i",
            str(bg_video),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(final_output),
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print("[05] ffmpeg audio remux failed; keeping silent video.")
            shutil.move(str(temp_video), str(final_output))
        else:
            temp_video.unlink(missing_ok=True)
    else:
        print("[05] ffmpeg not available; saving video without audio.")
        shutil.move(str(temp_video), str(final_output))

    print(f"[05] Saved final composite to {final_output}")


if __name__ == "__main__":
    main()

