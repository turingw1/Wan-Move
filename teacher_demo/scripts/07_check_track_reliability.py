#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_demo.scripts._common import check_video_write_capability, ensure_workdirs, load_config, repo_path, save_json


TRACK_NAMES = [
    "stick_tip",
    "stick_mid_upper",
    "stick_mid_lower",
    "stick_handle",
    "right_hand",
    "right_elbow_anchor",
    "body_anchor",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Wan-Move track legality and reliability for fast iteration.")
    parser.add_argument("--config", default="teacher_demo/configs/demo_config.yaml")
    parser.add_argument("--tracks", default="teacher_demo/work/tracks/tracks.npy")
    parser.add_argument("--visibility", default="teacher_demo/work/tracks/visibility.npy")
    parser.add_argument("--output", default="teacher_demo/work/tracks/track_reliability_report.json")
    parser.add_argument("--preview-path", default="teacher_demo/work/previews/track_preview.mp4")
    return parser.parse_args()


def summarize_vector(points: np.ndarray) -> dict[str, float]:
    deltas = np.diff(points, axis=0)
    speeds = np.linalg.norm(deltas, axis=1) if len(deltas) else np.array([0.0], dtype=np.float32)
    accels = np.linalg.norm(np.diff(deltas, axis=0), axis=1) if len(deltas) > 1 else np.array([0.0], dtype=np.float32)
    displacement = float(np.linalg.norm(points[-1] - points[0]))
    path_length = float(np.linalg.norm(deltas, axis=1).sum()) if len(deltas) else 0.0
    bbox = {
        "min_x": float(points[:, 0].min()),
        "max_x": float(points[:, 0].max()),
        "min_y": float(points[:, 1].min()),
        "max_y": float(points[:, 1].max()),
    }
    return {
        "displacement_px": displacement,
        "path_length_px": path_length,
        "max_speed_px_per_frame": float(speeds.max()),
        "mean_speed_px_per_frame": float(speeds.mean()),
        "max_accel_px_per_frame2": float(accels.max()),
        **bbox,
    }


def main() -> None:
    args = parse_args()
    ensure_workdirs()
    config = load_config(args.config)

    image_path = repo_path(config["input_image"])
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    tracks_path = Path(args.tracks)
    vis_path = Path(args.visibility)
    if not tracks_path.exists():
        raise FileNotFoundError(f"Track file not found: {tracks_path}")
    if not vis_path.exists():
        raise FileNotFoundError(f"Visibility file not found: {vis_path}")

    tracks = np.load(tracks_path)
    visibility = np.load(vis_path)

    report: dict[str, object] = {
        "config_path": args.config,
        "tracks_path": str(tracks_path),
        "visibility_path": str(vis_path),
        "image_size": {"width": width, "height": height},
        "input_shapes": {
            "tracks": list(tracks.shape),
            "visibility": list(visibility.shape),
        },
        "input_dtypes": {
            "tracks": str(tracks.dtype),
            "visibility": str(visibility.dtype),
        },
        "motion_config": config.get("motion", {}),
    }

    warnings: list[str] = []
    suggestions: list[str] = []

    if tracks.ndim == 4:
        tracks_eval = tracks[0]
    elif tracks.ndim == 3:
        tracks_eval = tracks
    else:
        raise ValueError(f"Invalid tracks shape {tracks.shape}; expected [F, N, 2] or [1, F, N, 2].")

    if visibility.ndim == 3:
        visibility_eval = visibility[0]
    elif visibility.ndim == 2:
        visibility_eval = visibility
    else:
        raise ValueError(f"Invalid visibility shape {visibility.shape}; expected [F, N] or [1, F, N].")

    frame_num, track_num, coord_dim = tracks_eval.shape
    report["normalized_shape"] = {"frame_num": int(frame_num), "track_num": int(track_num), "coord_dim": int(coord_dim)}

    if coord_dim != 2:
        warnings.append(f"Coordinate dimension is {coord_dim}, expected 2.")
    if frame_num != int(config["frame_num"]):
        warnings.append(f"frame_num mismatch: tracks have {frame_num}, config expects {config['frame_num']}.")
    if track_num < 6:
        warnings.append(f"Only {track_num} tracks found; MVP expects at least 6.")
    if not np.isfinite(tracks_eval).all():
        warnings.append("tracks contain NaN or inf values.")

    in_bounds = (
        (tracks_eval[..., 0] >= 0)
        & (tracks_eval[..., 0] <= width - 1)
        & (tracks_eval[..., 1] >= 0)
        & (tracks_eval[..., 1] <= height - 1)
    )
    in_bounds_ratio = float(in_bounds.mean())
    report["in_bounds_ratio"] = in_bounds_ratio
    if in_bounds_ratio < 1.0:
        warnings.append(f"Some coordinates are outside image bounds. In-bounds ratio={in_bounds_ratio:.4f}.")

    visibility_ratio = float(np.mean(visibility_eval.astype(np.float32)))
    report["visibility_ratio"] = visibility_ratio
    if visibility_ratio < 0.98:
        warnings.append(f"Visibility ratio is low ({visibility_ratio:.4f}); unexpected for the current synthetic track generator.")

    per_track = {}
    for idx in range(track_num):
        name = TRACK_NAMES[idx] if idx < len(TRACK_NAMES) else f"track_{idx}"
        per_track[name] = summarize_vector(tracks_eval[:, idx])
    report["per_track_stats"] = per_track

    rigid_pairs = [
        ("stick_tip", "stick_handle"),
        ("stick_tip", "stick_mid_upper"),
        ("stick_tip", "stick_mid_lower"),
        ("stick_handle", "right_hand"),
    ]
    rigid_stats = {}
    for a_name, b_name in rigid_pairs:
        if a_name not in per_track or b_name not in per_track:
            continue
        a_idx = TRACK_NAMES.index(a_name)
        b_idx = TRACK_NAMES.index(b_name)
        distances = np.linalg.norm(tracks_eval[:, a_idx] - tracks_eval[:, b_idx], axis=1)
        rigid_stats[f"{a_name}__{b_name}"] = {
            "mean_distance_px": float(distances.mean()),
            "std_distance_px": float(distances.std()),
            "cv": float(distances.std() / max(distances.mean(), 1e-6)),
        }
    report["rigidity_stats"] = rigid_stats

    tip_stats = per_track.get("stick_tip", {})
    handle_stats = per_track.get("stick_handle", {})
    body_stats = per_track.get("body_anchor", {})

    motion_cfg = config["motion"]
    max_tip_shift_cfg = float(motion_cfg["max_tip_shift_px"])
    if tip_stats.get("displacement_px", 0.0) > max_tip_shift_cfg * 1.05:
        warnings.append("Tip displacement exceeds configured max_tip_shift_px.")
        suggestions.append(f"Reduce motion.max_tip_shift_px below {tip_stats['displacement_px']:.1f}.")

    if tip_stats.get("max_speed_px_per_frame", 0.0) > 18.0:
        warnings.append("Tip speed is aggressive for an MVP motion path.")
        suggestions.append("Increase motion.move_frames or reduce motion.max_tip_shift_px.")

    tip_handle_cv = rigid_stats.get("stick_tip__stick_handle", {}).get("cv", 0.0)
    if tip_handle_cv > 0.08:
        warnings.append(f"Stick rigidity varies too much (tip-handle cv={tip_handle_cv:.4f}).")
        suggestions.append("Reduce motion.jitter_px or reduce hand/elbow follow motion.")

    if body_stats.get("path_length_px", 0.0) > 12.0:
        warnings.append("Body anchor moves more than expected for a mostly stable body.")
        suggestions.append("Reduce body breathing amplitude in 02_generate_wanmove_tracks.py.")

    hand_speed = per_track.get("right_hand", {}).get("max_speed_px_per_frame", 0.0)
    elbow_speed = per_track.get("right_elbow_anchor", {}).get("max_speed_px_per_frame", 0.0)
    if elbow_speed > hand_speed:
        warnings.append("Elbow is moving faster than hand; follow ratios may be inverted or too strong.")
        suggestions.append("Lower motion.elbow_follow_ratio.")

    preview_probe = check_video_write_capability(args.preview_path, (height, width), fps=16.0)
    report["preview_writer_probe"] = preview_probe
    if not preview_probe.get("ok", False):
        warnings.append("Preview video writer probe failed.")
        suggestions.append("Install/enable OpenCV video codecs or use ffmpeg/imageio fallback on the server.")

    score = 100
    score -= int((1.0 - in_bounds_ratio) * 50)
    score -= 15 if tip_handle_cv > 0.08 else 0
    score -= 15 if tip_stats.get("max_speed_px_per_frame", 0.0) > 18.0 else 0
    score -= 10 if body_stats.get("path_length_px", 0.0) > 12.0 else 0
    score -= 20 if not preview_probe.get("ok", False) else 0
    report["reliability_score"] = max(0, score)
    report["warnings"] = warnings
    report["suggestions"] = suggestions
    report["key_iteration_params"] = {
        "frame_num": config["frame_num"],
        "output_size": config["output_size"],
        "hold_frames": motion_cfg["hold_frames"],
        "move_frames": motion_cfg["move_frames"],
        "emphasize_frames": motion_cfg["emphasize_frames"],
        "settle_frames": motion_cfg["settle_frames"],
        "max_tip_shift_px": motion_cfg["max_tip_shift_px"],
        "hand_follow_ratio": motion_cfg["hand_follow_ratio"],
        "elbow_follow_ratio": motion_cfg["elbow_follow_ratio"],
        "jitter_px": motion_cfg["jitter_px"],
    }

    save_json(args.output, report)

    print(f"[07] Saved track reliability report to {args.output}")
    print(f"[07] reliability_score={report['reliability_score']}")
    if warnings:
        print("[07] warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("[07] No critical warnings.")
    if suggestions:
        print("[07] suggestions:")
        for suggestion in suggestions:
            print(f"  - {suggestion}")


if __name__ == "__main__":
    main()
