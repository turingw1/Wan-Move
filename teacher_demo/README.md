# Teacher Demo MVP

This directory contains a minimal end-to-end demo pipeline that:

1. extracts slide segments from `intro.mp4`
2. generates heuristic slide target points
3. converts them into Wan-Move-compatible pointer trajectories for `teacher_with_stick.png`
4. runs Wan-Move inference
5. removes the white background from the generated teacher video
6. composites the teacher onto the original lecture video

Run every command from the repository root.

## Server Layout Assumption

This MVP follows the same storage convention as the rest of the deployment docs:

- workspace repo: `~/workspace/$NAME/Wan-Move`
- cache root: `/cache/$NAME/Wan-Move`

Before running the demo, export at least:

```bash
export NAME="Zhengwei"   # replace with your own identifier
export WAN_MOVE_ROOT="$HOME/workspace/$NAME/Wan-Move"
export WAN_MOVE_CACHE="/cache/$NAME/Wan-Move"
```

The default demo config uses:

```text
${WAN_MOVE_CACHE}/models/Wan-Move-14B-480P
```

for `ckpt_dir`, so `WAN_MOVE_CACHE` must be set in the shell before running the scripts.

## Batch Motion Sweep

For teacher-motion-only evaluation, use the batch sweep script:

```bash
python teacher_demo/scripts/08_batch_motion_sweep.py \
  --config teacher_demo/configs/demo_config.yaml \
  --output-root /temp/Zhengwei/Wan-move
```

This prepares categorized motion cases such as:

- `small_slow`
- `small_fast`
- `large_slow`
- `large_fast`
- `body_motion`
- `emphasis`

Each case gets its own folder under `/temp/Zhengwei/Wan-move/<category>/<case_id>/` with:

- `tracks.npy`
- `visibility.npy`
- `track_preview.png`
- `track_preview.mp4`
- `track_schematic.png`
- `case_config.json`
- `run_wanmove.sh`

To also run Wan-Move and generate the white-background teacher videos automatically:

```bash
python teacher_demo/scripts/08_batch_motion_sweep.py \
  --config teacher_demo/configs/demo_config.yaml \
  --output-root /temp/Zhengwei/Wan-move \
  --run-inference \
  --overwrite
```

This will also create:

- `/temp/Zhengwei/Wan-move/batch_manifest.json`
- `/temp/Zhengwei/Wan-move/run_all_wanmove_cases.sh`

## Expected Inputs

- `teacher_with_stick.png`
- `intro.mp4`

Both files are expected at the repository root.

## Config

Default config:

```text
teacher_demo/configs/demo_config.yaml
```

## Full MVP Commands

0. Preview and manually verify the initial keypoints before generating tracks:

```bash
python teacher_demo/scripts/01b_preview_initial_keypoints.py \
  --config teacher_demo/configs/demo_config.yaml
```

This writes:

```text
teacher_demo/work/previews/initial_keypoints_overlay.png
teacher_demo/work/tracks/manual_keypoints.json
```

If any point is not exactly on the stick / hand / elbow / body anchor, edit
`teacher_demo/work/tracks/manual_keypoints.json` first.

1. Extract slide segments:

```bash
python teacher_demo/scripts/00_extract_slide_segments.py \
  --config teacher_demo/configs/demo_config.yaml
```

2. Generate slide targets:

```bash
python teacher_demo/scripts/01_generate_targets.py
```

3. Generate Wan-Move tracks and preview.

For the current stick-motion debug pass, use the straight-line emphasis demo first:

```bash
python teacher_demo/scripts/02_generate_wanmove_tracks.py \
  --config teacher_demo/configs/demo_config.yaml \
  --keypoint-json teacher_demo/work/tracks/manual_keypoints.json \
  --mode straight_emphasis \
  --preview_only
```

Track preview diagnostics will be written to:

```text
teacher_demo/work/previews/track_preview_diagnostics.json
teacher_demo/work/previews/track_schematic.png
teacher_demo/work/previews/track_generation_summary.json
```

If you want to use slide targets later:

```bash
python teacher_demo/scripts/02_generate_wanmove_tracks.py \
  --config teacher_demo/configs/demo_config.yaml \
  --keypoint-json teacher_demo/work/tracks/manual_keypoints.json \
  --targets teacher_demo/work/targets/targets.json \
  --mode video_targets \
  --preview_only
```

3.1 Check track legality and reliability:

```bash
python teacher_demo/scripts/07_check_track_reliability.py \
  --config teacher_demo/configs/demo_config.yaml \
  --tracks teacher_demo/work/tracks/tracks.npy \
  --visibility teacher_demo/work/tracks/visibility.npy \
  --output teacher_demo/work/tracks/track_reliability_report.json
```

4. Run Wan-Move:

```bash
bash teacher_demo/scripts/03_run_wanmove.sh \
  teacher_demo/configs/demo_config.yaml
```

5. Extract foreground:

```bash
python teacher_demo/scripts/04_extract_foreground.py \
  --input teacher_demo/work/wanmove_outputs/teacher_motion.mp4
```

6. Composite the final video:

```bash
python teacher_demo/scripts/05_composite_teacher_on_slides.py \
  --config teacher_demo/configs/demo_config.yaml \
  --foreground-dir teacher_demo/work/foreground/frames_rgba \
  --output teacher_demo/work/composite/final_demo.mp4
```

7. Create a preview grid:

```bash
python teacher_demo/scripts/06_make_preview_grid.py \
  --config teacher_demo/configs/demo_config.yaml
```

## Outputs

- `teacher_demo/work/segments.json`
- `teacher_demo/work/targets/targets.json`
- `teacher_demo/work/tracks/tracks.npy`
- `teacher_demo/work/tracks/visibility.npy`
- `teacher_demo/work/previews/initial_keypoints_overlay.png`
- `teacher_demo/work/previews/track_preview.png`
- `teacher_demo/work/previews/track_preview.mp4`
- `teacher_demo/work/previews/track_schematic.png`
- `teacher_demo/work/previews/track_preview_diagnostics.json`
- `teacher_demo/work/previews/track_generation_summary.json`
- `teacher_demo/work/tracks/track_reliability_report.json`
- `teacher_demo/work/wanmove_outputs/teacher_motion.mp4`
- `teacher_demo/work/foreground/frames_rgba/*.png`
- `teacher_demo/work/composite/final_demo.mp4`
- `teacher_demo/work/previews/preview_grid.png`
