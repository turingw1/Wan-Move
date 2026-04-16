#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_demo.scripts._common import (
    check_video_write_capability,
    ensure_workdirs,
    load_config,
    repo_path,
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

TRACK_NAMES = [
    "stick_tip",
    "stick_mid_upper",
    "stick_mid_lower",
    "stick_handle",
    "right_hand",
    "right_elbow_anchor",
    "body_anchor",
]


@dataclass
class MotionCase:
    category: str
    case_id: str
    title: str
    target_offset: tuple[float, float]
    hold_frames: int
    move_frames: int
    settle_frames: int
    tap_count: int
    tap_forward_px: float
    tap_hold_frames: int
    tap_move_frames: int
    hand_follow_ratio: float
    elbow_follow_ratio: float
    body_follow_ratio: float
    body_breath_amp: float
    body_sway_amp: tuple[float, float]
    body_sway_cycles: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-generate categorized teacher motion tests and optional Wan-Move white-background videos.")
    parser.add_argument("--config", default="teacher_demo/configs/demo_config.yaml")
    parser.add_argument("--output-root", default="/temp/Zhengwei/Wan-move")
    parser.add_argument("--run-inference", action="store_true", help="Run Wan-Move for every generated motion case.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing per-case outputs.")
    parser.add_argument("--dtype", default="bf16")
    return parser.parse_args()


def minimum_jerk(t: np.ndarray) -> np.ndarray:
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def interpolate_segment(start: np.ndarray, end: np.ndarray, frames: int) -> np.ndarray:
    if frames <= 1:
        return end[None, :].astype(np.float32)
    time = np.linspace(0.0, 1.0, frames, dtype=np.float32)
    weight = minimum_jerk(time)[:, None]
    return (start[None, :] * (1.0 - weight) + end[None, :] * weight).astype(np.float32)


def resample_points(points: np.ndarray, frame_num: int) -> np.ndarray:
    if len(points) == frame_num:
        return points.astype(np.float32)
    source_t = np.linspace(0.0, 1.0, len(points), dtype=np.float32)
    target_t = np.linspace(0.0, 1.0, frame_num, dtype=np.float32)
    x = np.interp(target_t, source_t, points[:, 0])
    y = np.interp(target_t, source_t, points[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)


def build_motion_cases() -> list[MotionCase]:
    return [
        MotionCase("small_slow", "ss_center_tap2", "small slow center reach + two taps", (45, -18), 14, 42, 16, 2, 9, 4, 4, 0.22, 0.08, 0.01, 1.5, (0.0, 0.0), 0.0),
        MotionCase("small_slow", "ss_left_gentle", "small slow left drift", (-55, 8), 12, 44, 18, 1, 6, 4, 5, 0.22, 0.08, 0.02, 1.2, (-1.0, 0.8), 0.5),
        MotionCase("small_fast", "sf_center_fasttap", "small fast jab + two taps", (38, -22), 8, 22, 12, 2, 12, 3, 3, 0.26, 0.10, 0.02, 1.6, (0.0, 0.0), 0.0),
        MotionCase("small_fast", "sf_right_nod", "small fast right point", (70, -10), 8, 20, 10, 1, 10, 3, 3, 0.25, 0.09, 0.03, 1.4, (2.0, -1.5), 1.0),
        MotionCase("large_slow", "ls_up_right_sweep", "large slow up-right sweep", (125, -78), 12, 48, 18, 2, 10, 4, 5, 0.28, 0.10, 0.05, 2.0, (4.0, -2.0), 1.0),
        MotionCase("large_slow", "ls_left_board", "large slow left board reach", (-120, -35), 14, 50, 18, 1, 8, 4, 5, 0.27, 0.10, 0.05, 2.0, (-3.5, 1.5), 1.0),
        MotionCase("large_fast", "lf_up_jab", "large fast upward jab", (105, -95), 8, 24, 12, 2, 14, 3, 3, 0.30, 0.11, 0.06, 2.2, (4.0, -2.0), 1.2),
        MotionCase("large_fast", "lf_right_snap", "large fast right snap", (150, -18), 6, 18, 10, 2, 14, 3, 3, 0.31, 0.11, 0.07, 2.0, (5.0, -1.0), 1.3),
        MotionCase("body_motion", "bm_lean_right", "body follow right lean", (90, -28), 10, 34, 14, 1, 8, 4, 4, 0.28, 0.12, 0.10, 2.5, (10.0, -4.0), 1.0),
        MotionCase("body_motion", "bm_lean_left", "body follow left lean", (-95, -14), 10, 34, 14, 1, 8, 4, 4, 0.28, 0.12, 0.10, 2.5, (-10.0, -3.0), 1.0),
        MotionCase("emphasis", "em_doubletap_board", "board point with double emphasis", (80, -55), 12, 32, 16, 2, 15, 4, 4, 0.27, 0.10, 0.04, 1.8, (0.0, 0.0), 0.0),
        MotionCase("emphasis", "em_tripletap_center", "center point with triple emphasis", (50, -24), 12, 28, 14, 3, 11, 3, 3, 0.26, 0.10, 0.03, 1.8, (1.0, -1.0), 0.8),
    ]


def straight_line_with_taps(initial_tip: np.ndarray, target_tip: np.ndarray, frame_num: int, case: MotionCase) -> np.ndarray:
    direction = target_tip - initial_tip
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        direction = np.array([1.0, 0.0], dtype=np.float32)
    else:
        direction = direction / norm

    frames: list[np.ndarray] = []
    frames.extend([initial_tip.copy()] * case.hold_frames)
    frames.extend(interpolate_segment(initial_tip, target_tip, case.move_frames).tolist())
    frames.extend([target_tip.copy()] * case.tap_hold_frames)
    for _ in range(case.tap_count):
        emphasize_tip = target_tip + direction * case.tap_forward_px
        frames.extend(interpolate_segment(target_tip, emphasize_tip, case.tap_move_frames).tolist())
        frames.extend(interpolate_segment(emphasize_tip, target_tip, case.tap_move_frames).tolist())
        frames.extend([target_tip.copy()] * case.tap_hold_frames)
    frames.extend(interpolate_segment(np.array(frames[-1], dtype=np.float32), target_tip, case.settle_frames).tolist())
    return resample_points(np.array(frames, dtype=np.float32), frame_num)


def draw_preview(base_image: np.ndarray, tracks: np.ndarray, output_png: Path, output_mp4: Path) -> dict[str, object]:
    canvas = Image.fromarray(base_image.copy()).convert("RGB")
    draw_static = ImageDraw.Draw(canvas)
    for idx in range(tracks.shape[1]):
        coords = [tuple(map(float, pt)) for pt in tracks[:, idx]]
        draw_static.line(coords, fill=COLOR_MAP[idx % len(COLOR_MAP)], width=3)
        x, y = coords[-1]
        draw_static.ellipse((x - 5, y - 5, x + 5, y + 5), fill=COLOR_MAP[idx % len(COLOR_MAP)])
    output_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_png)

    frames = []
    for frame_idx in range(tracks.shape[0]):
        frame = Image.fromarray(base_image.copy()).convert("RGB")
        draw = ImageDraw.Draw(frame)
        for idx in range(tracks.shape[1]):
            coords = [tuple(map(float, pt)) for pt in tracks[: frame_idx + 1, idx]]
            if len(coords) > 1:
                draw.line(coords, fill=COLOR_MAP[idx % len(COLOR_MAP)], width=4)
            x, y = coords[-1]
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=COLOR_MAP[idx % len(COLOR_MAP)])
        frames.append(np.array(frame))
    return write_rgb_video(output_mp4, frames, fps=16.0)


def draw_schematic(base_image: np.ndarray, tracks: np.ndarray, output_path: Path) -> None:
    canvas = Image.fromarray(base_image.copy()).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for idx, name in enumerate(TRACK_NAMES[: tracks.shape[1]]):
        color = COLOR_MAP[idx % len(COLOR_MAP)]
        coords = [tuple(map(float, pt)) for pt in tracks[:, idx]]
        if len(coords) > 1:
            draw.line(coords, fill=color, width=3)
        sx, sy = coords[0]
        ex, ey = coords[-1]
        draw.ellipse((sx - 6, sy - 6, sx + 6, sy + 6), fill=color, outline=(0, 0, 0), width=2)
        draw.rectangle((ex - 6, ey - 6, ex + 6, ey + 6), outline=color, width=2)
        draw.text((sx + 10, sy - 10), f"{name}:start", fill=color, font=font)
        draw.text((ex + 10, ey + 4), f"{name}:end", fill=color, font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def build_tracks(keypoints: dict[str, list[int]], frame_num: int, case: MotionCase, image_size: tuple[int, int], max_tip_shift_px: float) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    width, height = image_size
    initial_points = {
        "stick_tip": np.array(keypoints["stick_tip_initial"], dtype=np.float32),
        "stick_mid_upper": np.array(keypoints["stick_mid_upper"], dtype=np.float32),
        "stick_mid_lower": np.array(keypoints["stick_mid_lower"], dtype=np.float32),
        "stick_handle": np.array(keypoints["stick_handle"], dtype=np.float32),
        "right_hand": np.array(keypoints["right_hand"], dtype=np.float32),
        "right_elbow_anchor": np.array(keypoints["right_elbow_anchor"], dtype=np.float32),
        "body_anchor": np.array(keypoints["body_anchor"], dtype=np.float32),
    }
    initial_tip = initial_points["stick_tip"]
    target_tip = initial_tip + np.array(case.target_offset, dtype=np.float32)
    delta = target_tip - initial_tip
    delta_norm = float(np.linalg.norm(delta))
    if delta_norm > max_tip_shift_px > 0:
        target_tip = initial_tip + delta / delta_norm * max_tip_shift_px
    target_tip[0] = np.clip(target_tip[0], 0, width - 1)
    target_tip[1] = np.clip(target_tip[1], 0, height - 1)

    tip_path = straight_line_with_taps(initial_tip, target_tip, frame_num, case)

    handle0 = initial_points["stick_handle"]
    tip0 = initial_points["stick_tip"]
    hand0 = initial_points["right_hand"]
    elbow0 = initial_points["right_elbow_anchor"]
    body0 = initial_points["body_anchor"]
    handle_to_tip_len = float(np.linalg.norm(tip0 - handle0))

    ratios = {}
    for name in ("stick_mid_upper", "stick_mid_lower"):
        point = initial_points[name]
        ratios[name] = float(np.linalg.norm(point - handle0) / max(handle_to_tip_len, 1e-6))

    tracks = np.zeros((frame_num, 7, 2), dtype=np.float32)
    visibility = np.ones((frame_num, 7), dtype=np.bool_)
    time = np.linspace(0.0, 1.0, frame_num, dtype=np.float32)
    sway = np.stack(
        [
            np.sin(2.0 * math.pi * case.body_sway_cycles * time) * case.body_sway_amp[0],
            np.cos(2.0 * math.pi * case.body_sway_cycles * time) * case.body_sway_amp[1],
        ],
        axis=1,
    ).astype(np.float32)
    breathe = np.sin(2.0 * math.pi * time).astype(np.float32) * case.body_breath_amp

    for idx in range(frame_num):
        tip = tip_path[idx]
        base_handle = handle0 + case.hand_follow_ratio * (tip - tip0)
        direction = tip - base_handle
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm < 1e-5:
            direction = tip0 - handle0
            direction_norm = float(np.linalg.norm(direction))
        direction = direction / max(direction_norm, 1e-6)
        handle = tip - direction * handle_to_tip_len

        tracks[idx, 0] = tip
        tracks[idx, 1] = handle + direction * handle_to_tip_len * ratios["stick_mid_upper"]
        tracks[idx, 2] = handle + direction * handle_to_tip_len * ratios["stick_mid_lower"]
        tracks[idx, 3] = handle

        handle_delta = handle - handle0
        tip_delta = tip - tip0
        hand = hand0 + handle_delta * 0.85 + tip_delta * 0.05
        elbow = elbow0 + handle_delta * case.elbow_follow_ratio
        body = body0 + handle_delta * case.body_follow_ratio + sway[idx] + np.array([0.5 * breathe[idx], -1.2 * breathe[idx]], dtype=np.float32)

        tracks[idx, 4] = hand
        tracks[idx, 5] = elbow
        tracks[idx, 6] = body

    tracks[..., 0] = np.clip(tracks[..., 0], 0, width - 1)
    tracks[..., 1] = np.clip(tracks[..., 1], 0, height - 1)

    summary = {
        "target_tip": target_tip.tolist(),
        "tip_offset": list(case.target_offset),
        "hand_follow_ratio": case.hand_follow_ratio,
        "elbow_follow_ratio": case.elbow_follow_ratio,
        "body_follow_ratio": case.body_follow_ratio,
        "body_sway_amp": list(case.body_sway_amp),
        "body_breath_amp": case.body_breath_amp,
    }
    return tracks, visibility, summary


def write_case_run_script(case_dir: Path, config: dict[str, object], image_path: Path, dtype: str) -> None:
    script_path = case_dir / "run_wanmove.sh"
    script = f"""#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

REPO_ROOT={shlex.quote(str(REPO_ROOT))}
cd "$REPO_ROOT"

INPUT_IMAGE={shlex.quote(str(image_path))}
CKPT_DIR={shlex.quote(str(config["ckpt_dir"]))}
OUTPUT_SIZE={shlex.quote(str(config["output_size"]))}
FRAME_NUM={shlex.quote(str(config["frame_num"]))}
PROMPT={shlex.quote(str(config["prompt"]))}
TRACK_PATH={shlex.quote(str(case_dir / "tracks.npy"))}
VIS_PATH={shlex.quote(str(case_dir / "visibility.npy"))}
OUTPUT_PATH={shlex.quote(str(case_dir / "teacher_motion.mp4"))}

python generate.py \\
  --task wan-move-i2v \\
  --size "$OUTPUT_SIZE" \\
  --frame_num "$FRAME_NUM" \\
  --ckpt_dir "$CKPT_DIR" \\
  --image "$INPUT_IMAGE" \\
  --track "$TRACK_PATH" \\
  --track_visibility "$VIS_PATH" \\
  --prompt "$PROMPT" \\
  --save_file "$OUTPUT_PATH" \\
  --dtype {shlex.quote(dtype)} \\
  --offload_model False
"""
    script_path.write_text(script, encoding="utf-8")
    script_path.chmod(0o755)


def maybe_run_case(case_dir: Path) -> dict[str, object]:
    cmd = ["bash", str(case_dir / "run_wanmove.sh")]
    log_path = case_dir / "wanmove_run.log"
    lines: list[str] = []
    process = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert process.stdout is not None
    case_name = case_dir.name
    with log_path.open("w", encoding="utf-8") as log_file:
        for line in process.stdout:
            lines.append(line)
            log_file.write(line)
            print(f"[08][{case_name}] {line.rstrip()}", flush=True)
    returncode = process.wait()
    return {
        "returncode": returncode,
        "log_path": str(log_path),
        "video_exists": (case_dir / "teacher_motion.mp4").exists(),
    }


def main() -> None:
    args = parse_args()
    ensure_workdirs()
    config = load_config(args.config)
    image_path = repo_path(config["input_image"])
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    width, height = image.size
    frame_num = int(config["frame_num"])
    max_tip_shift_px = float(config["motion"]["max_tip_shift_px"])
    keypoints = config["teacher_keypoints"]
    keypoint_payload = {
        "source_config": args.config,
        "note": "Batch motion sweep uses these teacher keypoints as the motion base.",
        "teacher_keypoints": keypoints,
    }

    cases = build_motion_cases()
    manifest = {
        "output_root": str(output_root),
        "case_count": len(cases),
        "categories": {},
        "cases": [],
    }

    video_probe = check_video_write_capability(output_root / "codec_probe.mp4", (height, width), fps=16.0)
    manifest["preview_writer_probe"] = video_probe

    run_all_lines = ["#!/usr/bin/env bash", "set -euo pipefail", f"cd {shlex.quote(str(REPO_ROOT))}"]

    total_cases = len(cases)
    for idx, case in enumerate(cases, start=1):
        category_dir = output_root / case.category
        case_dir = category_dir / case.case_id
        if case_dir.exists() and not args.overwrite:
            raise FileExistsError(f"Case directory already exists: {case_dir}. Use --overwrite to replace.")
        case_dir.mkdir(parents=True, exist_ok=True)
        save_json(case_dir / "manual_keypoints.json", keypoint_payload)

        tracks, visibility, motion_summary = build_tracks(keypoints, frame_num, case, (width, height), max_tip_shift_px)
        tracks_batched = tracks[None, ...].astype(np.float32)
        visibility_batched = visibility[None, ...]
        np.save(case_dir / "tracks.npy", tracks_batched)
        np.save(case_dir / "visibility.npy", visibility_batched)

        preview_result = draw_preview(image_np, tracks, case_dir / "track_preview.png", case_dir / "track_preview.mp4")
        draw_schematic(image_np, tracks, case_dir / "track_schematic.png")

        case_config = {
            "category": case.category,
            "case_id": case.case_id,
            "title": case.title,
            "frame_num": frame_num,
            "motion_case": case.__dict__,
            "motion_summary": motion_summary,
            "output_tracks_shape": list(tracks_batched.shape),
            "output_visibility_shape": list(visibility_batched.shape),
            "preview_writer": preview_result,
        }
        save_json(case_dir / "case_config.json", case_config)
        write_case_run_script(case_dir, config, image_path, args.dtype)
        run_all_lines.append(f"bash {shlex.quote(str(case_dir / 'run_wanmove.sh'))}")

        case_info = {
            "category": case.category,
            "case_id": case.case_id,
            "title": case.title,
            "case_dir": str(case_dir),
            "tracks": str(case_dir / "tracks.npy"),
            "visibility": str(case_dir / "visibility.npy"),
            "preview_png": str(case_dir / "track_preview.png"),
            "preview_mp4": str(case_dir / "track_preview.mp4"),
            "schematic_png": str(case_dir / "track_schematic.png"),
            "run_script": str(case_dir / "run_wanmove.sh"),
        }

        if args.run_inference:
            print(f"[08] Running inference for case {idx}/{total_cases}: {case.category}/{case.case_id}", flush=True)
            case_info["inference"] = maybe_run_case(case_dir)
            print(
                f"[08] Finished case {idx}/{total_cases}: {case.category}/{case.case_id} "
                f"(returncode={case_info['inference']['returncode']}, "
                f"video_exists={case_info['inference']['video_exists']})",
                flush=True,
            )

        manifest["cases"].append(case_info)
        manifest["categories"].setdefault(case.category, []).append(case.case_id)
        print(f"[08] Prepared {case.category}/{case.case_id}", flush=True)

    run_all_path = output_root / "run_all_wanmove_cases.sh"
    run_all_path.write_text("\n".join(run_all_lines) + "\n", encoding="utf-8")
    run_all_path.chmod(0o755)
    save_json(output_root / "batch_manifest.json", manifest)

    print(f"[08] Saved batch manifest to {output_root / 'batch_manifest.json'}", flush=True)
    print(f"[08] Saved batch runner to {run_all_path}", flush=True)
    print(f"[08] Output root: {output_root}", flush=True)
    if args.run_inference:
        print("[08] Inference was executed for all prepared cases.", flush=True)
    else:
        print("[08] Track/previews only. Add --run-inference on the server to generate white-background teacher videos.", flush=True)


if __name__ == "__main__":
    main()
