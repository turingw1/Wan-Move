#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_demo.scripts._common import ensure_workdirs, extract_video_triptych_frames, load_config, repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a quick preview grid for the MVP pipeline.")
    parser.add_argument("--config", default="teacher_demo/configs/demo_config.yaml")
    parser.add_argument("--track-preview", default="teacher_demo/work/previews/track_preview.png")
    parser.add_argument("--teacher-video", default="teacher_demo/work/wanmove_outputs/teacher_motion.mp4")
    parser.add_argument("--composite-video", default="teacher_demo/work/composite/final_demo.mp4")
    parser.add_argument("--output", default="teacher_demo/work/previews/preview_grid.png")
    return parser.parse_args()


def labeled_tile(image: np.ndarray, label: str, tile_size: tuple[int, int]) -> Image.Image:
    tile_w, tile_h = tile_size
    tile = Image.new("RGB", (tile_w, tile_h), (248, 248, 248))
    draw = ImageDraw.Draw(tile)
    frame = Image.fromarray(image).convert("RGB")
    frame.thumbnail((tile_w - 24, tile_h - 48), Image.Resampling.LANCZOS)
    tile.paste(frame, ((tile_w - frame.width) // 2, 10))
    draw.text((12, tile_h - 28), label, fill=(30, 30, 30), font=ImageFont.load_default())
    return tile


def main() -> None:
    args = parse_args()
    ensure_workdirs()
    config = load_config(args.config)

    teacher_image = np.array(Image.open(repo_path(config["input_image"])).convert("RGB"))
    track_preview = np.array(Image.open(args.track_preview).convert("RGB"))
    teacher_frames = extract_video_triptych_frames(args.teacher_video)
    composite_frames = extract_video_triptych_frames(args.composite_video)

    images = [teacher_image, track_preview, *teacher_frames, *composite_frames]
    labels = [
        "input teacher image",
        "track preview",
        "teacher first",
        "teacher middle",
        "teacher last",
        "composite first",
        "composite middle",
        "composite last",
    ]

    tile_size = (420, 280)
    cols = 3
    rows = 3
    canvas = Image.new("RGB", (cols * tile_size[0], rows * tile_size[1]), (255, 255, 255))
    for idx, (image, label) in enumerate(zip(images, labels)):
        tile = labeled_tile(image, label, tile_size)
        canvas.paste(tile, ((idx % cols) * tile_size[0], (idx // cols) * tile_size[1]))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"[06] Saved preview grid to {output_path}")


if __name__ == "__main__":
    main()

