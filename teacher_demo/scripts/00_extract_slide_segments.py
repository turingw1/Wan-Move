#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_demo.scripts._common import ensure_workdirs, load_config, repo_path, save_json, WORK_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract slide transition segments from intro.mp4.")
    parser.add_argument("--config", default="teacher_demo/configs/demo_config.yaml")
    parser.add_argument("--sample-interval", type=int, default=6, help="Frame interval for transition analysis.")
    parser.add_argument("--min-segment-frames", type=int, default=24, help="Minimum frames between boundaries.")
    parser.add_argument("--diff-threshold", type=float, default=2.5, help="Z-score threshold for transitions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_workdirs()
    config = load_config(args.config)
    video_path = repo_path(config["input_video"])
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    slide_dir = WORK_ROOT / "slides"

    sampled_frames: list[int] = []
    sampled_scores: list[float] = []
    stored_frames: dict[int, np.ndarray] = {}
    prev_small = None
    prev_hist = None
    frame_idx = 0

    print(f"[00] Reading {video_path} ({frame_count} frames @ {fps:.3f} fps)")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % args.sample_interval != 0:
            frame_idx += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stored_frames[frame_idx] = rgb
        small = cv2.resize(rgb, (160, 90), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        score = 0.0
        if prev_small is not None and prev_hist is not None:
            frame_diff = float(np.mean(np.abs(small.astype(np.float32) - prev_small.astype(np.float32))))
            hist_diff = float(cv2.compareHist(prev_hist.astype(np.float32), hist.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA))
            score = frame_diff * 0.8 + hist_diff * 255.0 * 0.2

        sampled_frames.append(frame_idx)
        sampled_scores.append(score)
        prev_small = small
        prev_hist = hist
        frame_idx += 1

    cap.release()
    if not sampled_frames:
        raise RuntimeError("No frames were sampled from the input video.")

    scores = np.array(sampled_scores, dtype=np.float32)
    mean = float(scores.mean())
    std = float(scores.std() + 1e-6)
    zscores = (scores - mean) / std

    boundaries = [0]
    last_boundary = 0
    for frame_num, z in zip(sampled_frames, zscores):
        if frame_num == 0:
            continue
        if z >= args.diff_threshold and frame_num - last_boundary >= args.min_segment_frames:
            boundaries.append(frame_num)
            last_boundary = frame_num
    if boundaries[-1] != frame_count:
        boundaries.append(frame_count)

    segments = []
    for idx, (start_frame, end_frame) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        if end_frame <= start_frame:
            continue
        rep_frame = min(start_frame + max((end_frame - start_frame) // 2, 0), frame_count - 1)
        candidate_keys = np.array(sorted(stored_frames.keys()), dtype=np.int32)
        nearest_idx = int(candidate_keys[np.argmin(np.abs(candidate_keys - rep_frame))])
        rep_rgb = stored_frames[nearest_idx]
        slide_path = slide_dir / f"slide_{idx:03d}.png"
        cv2.imwrite(str(slide_path), cv2.cvtColor(rep_rgb, cv2.COLOR_RGB2BGR))
        segments.append(
            {
                "segment_id": idx,
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "start_time": float(start_frame / fps),
                "end_time": float(end_frame / fps),
                "representative_frame": int(nearest_idx),
                "slide_path": str(slide_path.relative_to(REPO_ROOT)),
            }
        )

    save_json(WORK_ROOT / "segments.json", {"fps": fps, "frame_count": frame_count, "segments": segments})
    save_json(
        WORK_ROOT / "slides" / "transition_debug.json",
        {
            "sample_interval": args.sample_interval,
            "mean_score": mean,
            "std_score": std,
            "sampled_frames": sampled_frames,
            "scores": [float(x) for x in sampled_scores],
            "zscores": [float(x) for x in zscores.tolist()],
            "boundaries": boundaries,
        },
    )
    print(f"[00] Saved {len(segments)} segments to {WORK_ROOT / 'segments.json'}")


if __name__ == "__main__":
    main()
