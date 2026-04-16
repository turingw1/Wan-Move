#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_demo.scripts._common import (
    check_video_write_capability,
    ensure_workdirs,
    interpolate_segment,
    load_config,
    load_json,
    repo_path,
    resample_points,
    save_json,
    write_rgb_video,
)


COLOR_MAP = [
    (102, 153, 255),
    (0, 255, 255),
    (255, 255, 0),
    (255, 102, 204),
    (0, 255, 0),
    (255, 0, 0),
    (128, 0, 128),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Wan-Move tracks.npy and visibility.npy with previews.")
    parser.add_argument("--config", default="teacher_demo/configs/demo_config.yaml")
    parser.add_argument("--targets", default="teacher_demo/work/targets/targets.json")
    parser.add_argument("--output-dir", default="teacher_demo/work/tracks")
    parser.add_argument("--preview-dir", default="teacher_demo/work/previews")
    parser.add_argument("--max-targets", type=int, default=5)
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--motion-region", nargs=4, type=float, default=[650, 840, 170, 360], metavar=("X0", "X1", "Y0", "Y1"))
    return parser.parse_args()


def smooth_sequence(initial_tip: np.ndarray, targets: list[np.ndarray], frame_num: int, motion_cfg: dict[str, float]) -> np.ndarray:
    hold = max(1, int(motion_cfg["hold_frames"]))
    move = max(2, int(motion_cfg["move_frames"]))
    emphasize = max(1, int(motion_cfg["emphasize_frames"]))
    settle = max(1, int(motion_cfg["settle_frames"]))
    jitter = float(motion_cfg["jitter_px"])

    tip_frames = []
    current = initial_tip.astype(np.float32)
    for target in targets:
        tip_frames.extend([current.copy()] * hold)
        move_path = interpolate_segment(current, target, move)
        tip_frames.extend(move_path.tolist())

        angle = math.atan2(float(target[1] - current[1]), float(target[0] - current[0] + 1e-6))
        perp = np.array([-math.sin(angle), math.cos(angle)], dtype=np.float32)
        phase = np.linspace(0.0, 1.0, emphasize, dtype=np.float32)
        for p in phase:
            mag = jitter * 0.6 * math.sin(math.pi * p)
            offset = perp * mag + np.array([math.cos(2 * math.pi * p), math.sin(2 * math.pi * p)], dtype=np.float32) * (jitter * 0.15)
            tip_frames.append((target + offset).astype(np.float32))
        settle_path = interpolate_segment(np.array(tip_frames[-1], dtype=np.float32), target, settle)
        tip_frames.extend(settle_path.tolist())
        current = target.astype(np.float32)

    tip_dense = np.array(tip_frames, dtype=np.float32)
    if len(tip_dense) < 2:
        tip_dense = np.repeat(initial_tip[None, :], frame_num, axis=0)
    return resample_points(tip_dense, frame_num)


def draw_track_preview(base_image: np.ndarray, tracks: np.ndarray, visibility: np.ndarray, output_png: Path, output_mp4: Path) -> dict[str, object]:
    frame_num = tracks.shape[0]
    preview_frames: list[np.ndarray] = []
    composite = Image.fromarray(base_image.copy()).convert("RGB")
    draw_static = ImageDraw.Draw(composite)
    for track_idx in range(tracks.shape[1]):
        coords = [tuple(map(float, pt)) for pt in tracks[:, track_idx]]
        draw_static.line(coords, fill=COLOR_MAP[track_idx % len(COLOR_MAP)], width=3)
        x, y = coords[-1]
        draw_static.ellipse((x - 5, y - 5, x + 5, y + 5), fill=COLOR_MAP[track_idx % len(COLOR_MAP)])
    composite.save(output_png)

    for frame_idx in range(frame_num):
        frame_img = Image.fromarray(base_image.copy()).convert("RGB")
        draw = ImageDraw.Draw(frame_img)
        for track_idx in range(tracks.shape[1]):
            valid = visibility[frame_idx, track_idx]
            if not valid:
                continue
            trail = [tuple(map(float, pt)) for pt in tracks[: frame_idx + 1, track_idx]]
            if len(trail) > 1:
                draw.line(trail, fill=COLOR_MAP[track_idx % len(COLOR_MAP)], width=4)
            x, y = trail[-1]
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=COLOR_MAP[track_idx % len(COLOR_MAP)])
        preview_frames.append(np.array(frame_img))
    return write_rgb_video(output_mp4, preview_frames, fps=16.0)


def main() -> None:
    args = parse_args()
    ensure_workdirs()
    config = load_config(args.config)
    image_path = repo_path(config["input_image"])
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    targets_payload = load_json(args.targets) if Path(args.targets).exists() else {"targets": []}
    targets_data = targets_payload.get("targets", [])
    frame_num = int(config["frame_num"])
    motion_cfg = config["motion"]
    keypoints = config["teacher_keypoints"]

    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    width, height = image.size

    x0, x1, y0, y1 = args.motion_region
    motion_region = {
        "x": [max(0.0, x0), min(float(width - 1), x1)],
        "y": [max(0.0, y0), min(float(height - 1), y1)],
    }

    selected = targets_data[: max(1, args.max_targets)]
    if not selected:
        selected = [{"target_norm": [0.5, 0.5], "source": "fallback"}]

    teacher_targets = []
    for item in selected:
        tx_norm, ty_norm = item["target_norm"]
        target = np.array(
            [
                motion_region["x"][0] + tx_norm * (motion_region["x"][1] - motion_region["x"][0]),
                motion_region["y"][0] + ty_norm * (motion_region["y"][1] - motion_region["y"][0]),
            ],
            dtype=np.float32,
        )
        initial_tip = np.array(keypoints["stick_tip_initial"], dtype=np.float32)
        delta = target - initial_tip
        delta_norm = float(np.linalg.norm(delta))
        max_shift = float(motion_cfg["max_tip_shift_px"])
        if delta_norm > max_shift > 0:
            target = initial_tip + delta / delta_norm * max_shift
        teacher_targets.append(target)

    initial_points = {
        "stick_tip": np.array(keypoints["stick_tip_initial"], dtype=np.float32),
        "stick_mid_upper": np.array(keypoints["stick_mid_upper"], dtype=np.float32),
        "stick_mid_lower": np.array(keypoints["stick_mid_lower"], dtype=np.float32),
        "stick_handle": np.array(keypoints["stick_handle"], dtype=np.float32),
        "right_hand": np.array(keypoints["right_hand"], dtype=np.float32),
        "right_elbow_anchor": np.array(keypoints["right_elbow_anchor"], dtype=np.float32),
        "body_anchor": np.array(keypoints["body_anchor"], dtype=np.float32),
    }

    tip_path = smooth_sequence(initial_points["stick_tip"], teacher_targets, frame_num, motion_cfg)
    handle0 = initial_points["stick_handle"]
    tip0 = initial_points["stick_tip"]
    hand0 = initial_points["right_hand"]
    elbow0 = initial_points["right_elbow_anchor"]
    body0 = initial_points["body_anchor"]
    handle_to_tip_len = float(np.linalg.norm(tip0 - handle0))
    hand_follow = float(motion_cfg["hand_follow_ratio"])
    elbow_follow = float(motion_cfg["elbow_follow_ratio"])

    ratios = {}
    for name in ("stick_mid_upper", "stick_mid_lower"):
        point = initial_points[name]
        ratios[name] = float(np.linalg.norm(point - handle0) / max(handle_to_tip_len, 1e-6))

    tracks = np.zeros((frame_num, 7, 2), dtype=np.float32)
    visibility = np.ones((frame_num, 7), dtype=np.bool_)

    for idx in range(frame_num):
        target_tip = tip_path[idx]
        base_handle = handle0 + hand_follow * (target_tip - tip0)
        direction = target_tip - base_handle
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm < 1e-5:
            direction = tip0 - handle0
            direction_norm = float(np.linalg.norm(direction))
        direction = direction / max(direction_norm, 1e-6)
        handle = target_tip - direction * handle_to_tip_len
        tip = target_tip

        tracks[idx, 0] = tip
        tracks[idx, 1] = handle + direction * handle_to_tip_len * ratios["stick_mid_upper"]
        tracks[idx, 2] = handle + direction * handle_to_tip_len * ratios["stick_mid_lower"]
        tracks[idx, 3] = handle

        handle_delta = handle - handle0
        tip_delta = tip - tip0
        hand = hand0 + handle_delta * 0.85 + tip_delta * 0.05
        elbow = elbow0 + handle_delta * elbow_follow
        breathe = math.sin(2.0 * math.pi * idx / max(frame_num - 1, 1))
        body = body0 + np.array([0.5 * breathe, -1.5 * breathe], dtype=np.float32)

        tracks[idx, 4] = hand
        tracks[idx, 5] = elbow
        tracks[idx, 6] = body

    tracks[..., 0] = np.clip(tracks[..., 0], 0, width - 1)
    tracks[..., 1] = np.clip(tracks[..., 1], 0, height - 1)

    output_dir = Path(args.output_dir)
    preview_dir = Path(args.preview_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    tracks_batched = tracks[None, ...].astype(np.float32)
    visibility_batched = visibility[None, ...]
    np.save(output_dir / "tracks.npy", tracks_batched)
    np.save(output_dir / "visibility.npy", visibility_batched)

    preview_png = preview_dir / "track_preview.png"
    preview_mp4 = preview_dir / "track_preview.mp4"
    preview_diag = {"ok": True}
    try:
        writer_result = draw_track_preview(image_np, tracks, visibility, preview_png, preview_mp4)
        preview_diag.update(writer_result)
    except Exception as exc:
        preview_diag = {
            "ok": False,
            "error": str(exc),
            "writer_probe": check_video_write_capability(preview_mp4, (height, width), fps=16.0),
        }

    save_json(preview_dir / "track_preview_diagnostics.json", preview_diag)

    print(f"[02] Saved tracks to {output_dir / 'tracks.npy'}")
    print(f"[02] Saved visibility to {output_dir / 'visibility.npy'}")
    print(f"[02] Saved preview image to {preview_png}")
    if preview_diag.get("ok"):
        print(f"[02] Saved preview video to {preview_mp4}")
    else:
        print(f"[02] Preview video generation failed; see {preview_dir / 'track_preview_diagnostics.json'}")
    if args.preview_only:
        print("[02] --preview_only enabled; Wan-Move inference is not run by this script.")


if __name__ == "__main__":
    main()
