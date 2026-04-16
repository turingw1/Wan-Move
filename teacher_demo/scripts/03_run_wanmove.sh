#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-teacher_demo/configs/demo_config.yaml}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[03] Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

eval "$(python - "${CONFIG_PATH}" <<'PY'
import os
import shlex
import sys
from pathlib import Path
import yaml

config_path = Path(sys.argv[1])
with config_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

def expand(value: str) -> str:
    return os.path.expanduser(os.path.expandvars(value))

for key, value in {
    "INPUT_IMAGE": cfg["input_image"],
    "CKPT_DIR": cfg["ckpt_dir"],
    "OUTPUT_SIZE": cfg["output_size"],
    "FRAME_NUM": str(cfg["frame_num"]),
    "PROMPT": cfg["prompt"],
}.items():
    print(f"{key}={shlex.quote(expand(str(value)))}")
PY
)"

TRACK_PATH="teacher_demo/work/tracks/tracks.npy"
VIS_PATH="teacher_demo/work/tracks/visibility.npy"
OUTPUT_PATH="teacher_demo/work/wanmove_outputs/teacher_motion.mp4"

mkdir -p teacher_demo/work/wanmove_outputs

if [[ ! -f "${INPUT_IMAGE}" ]]; then
  echo "[03] Input image missing: ${INPUT_IMAGE}" >&2
  exit 1
fi
if [[ ! -d "${CKPT_DIR}" ]]; then
  echo "[03] Checkpoint directory missing: ${CKPT_DIR}" >&2
  echo "[03] Expected server layout: export WAN_MOVE_CACHE=/cache/\$NAME/Wan-Move" >&2
  exit 1
fi
if [[ ! -f "${TRACK_PATH}" ]]; then
  echo "[03] Track file missing: ${TRACK_PATH}" >&2
  exit 1
fi
if [[ ! -f "${VIS_PATH}" ]]; then
  echo "[03] Visibility file missing: ${VIS_PATH}" >&2
  exit 1
fi

echo "[03] Running Wan-Move inference..."
python generate.py \
  --task wan-move-i2v \
  --size "${OUTPUT_SIZE}" \
  --frame_num "${FRAME_NUM}" \
  --ckpt_dir "${CKPT_DIR}" \
  --image "${INPUT_IMAGE}" \
  --track "${TRACK_PATH}" \
  --track_visibility "${VIS_PATH}" \
  --prompt "${PROMPT}" \
  --save_file "${OUTPUT_PATH}" \
  --dtype bf16 \
  --offload_model False

echo "[03] Saved Wan-Move output to ${OUTPUT_PATH}"
