# Teacher Demo MVP

This directory contains a minimal end-to-end demo pipeline that:

1. extracts slide segments from `intro.mp4`
2. generates heuristic slide target points
3. converts them into Wan-Move-compatible pointer trajectories for `teacher_with_stick.png`
4. runs Wan-Move inference
5. removes the white background from the generated teacher video
6. composites the teacher onto the original lecture video

Run every command from the repository root.

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

1. Extract slide segments:

```bash
python teacher_demo/scripts/00_extract_slide_segments.py \
  --config teacher_demo/configs/demo_config.yaml
```

2. Generate slide targets:

```bash
python teacher_demo/scripts/01_generate_targets.py
```

3. Generate Wan-Move tracks and preview:

```bash
python teacher_demo/scripts/02_generate_wanmove_tracks.py \
  --config teacher_demo/configs/demo_config.yaml \
  --targets teacher_demo/work/targets/targets.json
```

If you only want the tracks and preview artifacts:

```bash
python teacher_demo/scripts/02_generate_wanmove_tracks.py \
  --config teacher_demo/configs/demo_config.yaml \
  --targets teacher_demo/work/targets/targets.json \
  --preview_only
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
- `teacher_demo/work/previews/track_preview.png`
- `teacher_demo/work/previews/track_preview.mp4`
- `teacher_demo/work/wanmove_outputs/teacher_motion.mp4`
- `teacher_demo/work/foreground/frames_rgba/*.png`
- `teacher_demo/work/composite/final_demo.mp4`
- `teacher_demo/work/previews/preview_grid.png`
