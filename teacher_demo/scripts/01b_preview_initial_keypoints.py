#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_demo.scripts._common import ensure_workdirs, load_config, load_json, repo_path, save_json


COLOR_MAP = {
    "stick_tip_initial": (255, 80, 80),
    "stick_mid_upper": (255, 200, 0),
    "stick_mid_lower": (100, 220, 100),
    "stick_handle": (60, 170, 255),
    "right_hand": (220, 120, 255),
    "right_elbow_anchor": (255, 140, 0),
    "body_anchor": (120, 120, 120),
}

SKELETON = [
    ("stick_tip_initial", "stick_mid_upper"),
    ("stick_mid_upper", "stick_mid_lower"),
    ("stick_mid_lower", "stick_handle"),
    ("stick_handle", "right_hand"),
    ("right_hand", "right_elbow_anchor"),
    ("right_elbow_anchor", "body_anchor"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview teacher keypoints on the original image and export an editable JSON.")
    parser.add_argument("--config", default="teacher_demo/configs/demo_config.yaml")
    parser.add_argument("--output-image", default="teacher_demo/work/previews/initial_keypoints_overlay.png")
    parser.add_argument("--output-json", default="teacher_demo/work/tracks/manual_keypoints.json")
    parser.add_argument("--force", action="store_true", help="Overwrite the editable JSON template if it already exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_workdirs()
    config = load_config(args.config)
    image_path = repo_path(config["input_image"])
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    json_path = Path(args.output_json)
    if json_path.exists() and not args.force:
        payload = load_json(json_path)
    else:
        payload = {
            "source_config": args.config,
            "note": "Edit coordinates here before running 02_generate_wanmove_tracks.py with --keypoint-json.",
            "teacher_keypoints": config["teacher_keypoints"],
        }
        save_json(json_path, payload)

    keypoints = payload["teacher_keypoints"]
    image = Image.open(image_path).convert("RGB")
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for a_name, b_name in SKELETON:
        draw.line([tuple(keypoints[a_name]), tuple(keypoints[b_name])], fill=(40, 180, 255), width=3)

    for name, xy in keypoints.items():
        x, y = xy
        color = COLOR_MAP.get(name, (255, 0, 0))
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=color, outline=(0, 0, 0), width=2)
        draw.text((x + 12, y - 10), name, fill=color, font=font)

    output_image = Path(args.output_image)
    output_image.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_image)

    print(f"[01b] Saved initial keypoint overlay to {output_image}")
    print(f"[01b] Saved editable keypoint JSON to {json_path}")


if __name__ == "__main__":
    main()

