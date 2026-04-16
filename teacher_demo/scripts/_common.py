from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
TEACHER_DEMO_ROOT = REPO_ROOT / "teacher_demo"
WORK_ROOT = TEACHER_DEMO_ROOT / "work"

WORK_DIRS = [
    WORK_ROOT / "slides",
    WORK_ROOT / "targets",
    WORK_ROOT / "tracks",
    WORK_ROOT / "previews",
    WORK_ROOT / "wanmove_outputs",
    WORK_ROOT / "foreground",
    WORK_ROOT / "foreground" / "frames_rgba",
    WORK_ROOT / "composite",
]


def ensure_workdirs() -> None:
    for path in WORK_DIRS:
        path.mkdir(parents=True, exist_ok=True)


def expand_config_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: expand_config_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_config_value(v) for v in value]
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    return value


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return expand_config_value(yaml.safe_load(handle))


def repo_path(value: str | Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(value))))
    return path if path.is_absolute() else REPO_ROOT / path


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def video_meta(video_path: str | Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0.0
    cap.release()
    return {
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "duration": duration,
    }


def read_video_frame(video_path: str | Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def write_rgb_video(path: str | Path, frames: list[np.ndarray], fps: float) -> dict[str, Any]:
    if not frames:
        raise ValueError("No frames to write.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    attempted: list[dict[str, Any]] = []

    for codec in ("mp4v", "avc1", "MJPG"):
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        opened = writer.isOpened()
        attempted.append({"backend": "opencv", "codec": codec, "opened": bool(opened)})
        if not opened:
            writer.release()
            continue
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        if path.exists() and path.stat().st_size > 0:
            return {"backend": "opencv", "codec": codec, "path": str(path), "attempted": attempted}

    try:
        import imageio.v2 as imageio

        imageio.mimsave(str(path), frames, fps=fps)
        if path.exists() and path.stat().st_size > 0:
            attempted.append({"backend": "imageio", "codec": "default", "opened": True})
            return {"backend": "imageio", "codec": "default", "path": str(path), "attempted": attempted}
    except Exception as exc:  # pragma: no cover - diagnostic path
        attempted.append({"backend": "imageio", "codec": "default", "opened": False, "error": str(exc)})

    raise RuntimeError(f"Failed to create video writer for {path}. Attempts: {attempted}")


def check_video_write_capability(output_path: str | Path, frame_shape: tuple[int, int], fps: float = 16.0) -> dict[str, Any]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame_shape
    frames = [
        np.full((height, width, 3), 245, dtype=np.uint8),
        np.full((height, width, 3), 235, dtype=np.uint8),
    ]
    temp_path = output_path.parent / f"{output_path.stem}_codec_probe.mp4"
    try:
        result = write_rgb_video(temp_path, frames, fps=fps)
        return {
            "ok": True,
            "writer_backend": result.get("backend"),
            "codec": result.get("codec"),
            "path": str(temp_path),
            "attempted": result.get("attempted", []),
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
        }
    finally:
        temp_path.unlink(missing_ok=True)


def alpha_blend(base_rgb: np.ndarray, overlay_rgba: np.ndarray, x: int, y: int) -> np.ndarray:
    out = base_rgb.copy()
    h, w = overlay_rgba.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(base_rgb.shape[1], x + w)
    y1 = min(base_rgb.shape[0], y + h)
    if x0 >= x1 or y0 >= y1:
        return out

    overlay_crop = overlay_rgba[y0 - y : y1 - y, x0 - x : x1 - x]
    alpha = overlay_crop[..., 3:4].astype(np.float32) / 255.0
    color = overlay_crop[..., :3].astype(np.float32)
    base_slice = out[y0:y1, x0:x1].astype(np.float32)
    out[y0:y1, x0:x1] = np.clip(color * alpha + base_slice * (1.0 - alpha), 0, 255).astype(np.uint8)
    return out


def extract_video_triptych_frames(video_path: str | Path) -> list[np.ndarray]:
    meta = video_meta(video_path)
    indices = [0, max(0, meta["frame_count"] // 2), max(0, meta["frame_count"] - 1)]
    return [read_video_frame(video_path, idx) for idx in indices]


def make_canvas(images: list[np.ndarray], labels: list[str], tile_size: tuple[int, int]) -> np.ndarray:
    tile_w, tile_h = tile_size
    cols = 3
    rows = math.ceil(len(images) / cols)
    canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), (250, 250, 250))
    for idx, image in enumerate(images):
        tile = Image.fromarray(image).convert("RGB")
        tile.thumbnail((tile_w - 20, tile_h - 50), Image.Resampling.LANCZOS)
        x = (idx % cols) * tile_w + (tile_w - tile.width) // 2
        y = (idx // cols) * tile_h + 10
        canvas.paste(tile, (x, y))
        label = labels[idx]
        draw = Image.fromarray(np.array(canvas))
        canvas = Image.fromarray(np.array(draw))
    return np.array(canvas)


def ffmpeg_available() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return True
    except FileNotFoundError:
        return False
