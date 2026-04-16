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

from teacher_demo.scripts._common import ensure_workdirs, load_json, save_json, WORK_ROOT


FALLBACK_TARGETS = [
    [0.25, 0.30],
    [0.70, 0.30],
    [0.50, 0.50],
    [0.30, 0.70],
    [0.70, 0.65],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one target point for each extracted slide image.")
    parser.add_argument("--segments", default="teacher_demo/work/segments.json")
    parser.add_argument("--output", default="teacher_demo/work/targets/targets.json")
    parser.add_argument("--white-threshold", type=int, default=242)
    parser.add_argument("--min-area-ratio", type=float, default=0.002)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_workdirs()
    segments_payload = load_json(args.segments)
    segments = segments_payload.get("segments", [])
    if not segments:
        raise RuntimeError("segments.json does not contain any segments.")

    targets = []
    fallback_index = 0
    for segment in segments:
        slide_path = Path(segment["slide_path"])
        if not slide_path.exists():
            raise FileNotFoundError(f"Slide image missing: {slide_path}")

        bgr = cv2.imread(str(slide_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read slide image: {slide_path}")
        h, w = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        non_white = (gray < args.white_threshold).astype(np.uint8) * 255

        margin_x = int(w * 0.03)
        margin_y = int(h * 0.03)
        non_white[:margin_y] = 0
        non_white[h - margin_y :] = 0
        non_white[:, :margin_x] = 0
        non_white[:, w - margin_x :] = 0

        kernel = np.ones((5, 5), np.uint8)
        clean = cv2.morphologyEx(non_white, cv2.MORPH_OPEN, kernel)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean)
        best = None
        best_score = -1.0
        min_area = max(20, int(args.min_area_ratio * h * w))

        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x, y, box_w, box_h, _ = stats[label_id]
            cx, cy = centroids[label_id]
            upper_middle_pref = 1.0 - abs((cy / h) - 0.42)
            center_pref = 1.0 - abs((cx / w) - 0.50) * 0.4
            aspect_bonus = min(box_w, box_h) / max(box_w, box_h, 1)
            score = area * max(upper_middle_pref, 0.1) * max(center_pref, 0.1) * (0.6 + 0.4 * aspect_bonus)
            if score > best_score:
                best_score = score
                best = (float(cx), float(cy))

        if best is None:
            norm_x, norm_y = FALLBACK_TARGETS[fallback_index % len(FALLBACK_TARGETS)]
            fallback_index += 1
            target_abs = [int(round(norm_x * w)), int(round(norm_y * h))]
            source = "fallback"
        else:
            target_abs = [int(round(best[0])), int(round(best[1]))]
            norm_x = float(np.clip(target_abs[0] / max(w - 1, 1), 0.0, 1.0))
            norm_y = float(np.clip(target_abs[1] / max(h - 1, 1), 0.0, 1.0))
            source = "heuristic"

        targets.append(
            {
                "segment_id": segment["segment_id"],
                "slide_path": segment["slide_path"],
                "target_norm": [round(norm_x, 4), round(norm_y, 4)],
                "target_abs": target_abs,
                "source": source,
            }
        )

    save_json(args.output, {"targets": targets, "fallback_targets": FALLBACK_TARGETS})
    print(f"[01] Saved {len(targets)} target points to {args.output}")


if __name__ == "__main__":
    main()
