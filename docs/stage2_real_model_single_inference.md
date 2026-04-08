# Stage 2 Real Model Single Inference

## Scope

Goal: add a real (non-dummy) single-sample inference path that takes:
- screenshot
- natural-language instruction

and outputs structured action prediction plus qualitative visualization artifact.

Validation date: 2026-03-31

## What Was Implemented

1. Real Qwen2-VL integration path (implemented in code)
- Added real `VLMBackbone` loading/generation in `src/gui_grounding/models/vlm_backbone.py`.
- Added `Qwen2VLGroundingModel` in `src/gui_grounding/models/qwen2_vl_grounding.py`.
- Added structured JSON prompting, parsing, coordinate clamping, and `PredictionResult` conversion.

2. Runnable single-inference entry point
- Added `scripts/run_single_inference.py`.
- Supports:
  - dataset path (`Mind2WebDataset`) mode
  - direct local sample override mode (`--image-path` + `--instruction`) for robust replay
  - selectable backend (`qwen2vl` or `clip_grid`)

3. Real fallback backend to guarantee runnable inference now
- Added `CLIPGridGroundingModel` in `src/gui_grounding/models/clip_grid_grounding.py`.
- Uses real `openai/clip-vit-base-patch32` embeddings:
  - slices screenshot into a grid
  - ranks patches by instruction-image similarity
  - returns bbox/click/action/confidence in project schema

4. Config support
- Added `configs/demo/single_inference.yaml`.
- Includes backend selection and runtime options.

## Actual Model Used (Successful Run)

- Backend: `clip_grid`
- Model identifier: `openai/clip-vit-base-patch32`
- Device used: `cuda` (RTX 3090)

> Note: Qwen2-VL backend is implemented but not fully runnable on this machine yet due model shard download bottleneck (see blockers).

## Environment Used

- Python: `3.13.5`
- Torch: `2.10.0+cu128`
- Transformers: `5.3.0`
- GPU: `NVIDIA GeForce RTX 3090 (24GB)`

## Commands Actually Run

```bash
python scripts/run_single_inference.py --config configs/demo/single_inference.yaml --dry-run

python scripts/run_single_inference.py \
  --config configs/demo/single_inference.yaml \
  --backend clip_grid \
  --model-id openai/clip-vit-base-patch32 \
  --sample-id mind2web_train_6c7a7082-2897-41c7-9688-4b0f3d778cdb \
  --image-path "data/processed/mind2web_screenshots/train/6c7a7082-2897-41c7-9688-4b0f3d778cdb.jpg" \
  --instruction "rent a car in Brooklyn - Central, NY on from April 9 to April 15."
```

## Sample Source

- Real Mind2Web sample id: `mind2web_train_6c7a7082-2897-41c7-9688-4b0f3d778cdb`
- Screenshot path:
  - `data/processed/mind2web_screenshots/train/6c7a7082-2897-41c7-9688-4b0f3d778cdb.jpg`
- Instruction:
  - `rent a car in Brooklyn - Central, NY on from April 9 to April 15.`

## Output Schema Produced

Saved JSON follows the repository’s `PredictionResult` structure:
- `predicted_action_type`
- `predicted_bbox`
- `predicted_click_point`
- `predicted_element_id`
- `confidence`

Additionally includes:
- `raw_model_response`
- `parsed_model_payload`
- `artifact_path`

## Artifact Paths

- Structured output:
  - `outputs/single_inference/mind2web_train_6c7a7082-2897-41c7-9688-4b0f3d778cdb_20260331_185315.json`
- Qualitative visualization:
  - `outputs/single_inference/mind2web_train_6c7a7082-2897-41c7-9688-4b0f3d778cdb_20260331_185315.png`

## Files Changed

- `src/gui_grounding/models/vlm_backbone.py`
- `src/gui_grounding/models/qwen2_vl_grounding.py`
- `src/gui_grounding/models/clip_grid_grounding.py`
- `src/gui_grounding/models/__init__.py`
- `scripts/run_single_inference.py`
- `configs/demo/single_inference.yaml`
- `docs/stage2_real_model_single_inference.md`

## Remaining Blockers for Stage A SFT

1. Qwen2-VL full weights are not yet locally available
- Code path exists, but first-time download of `Qwen/Qwen2-VL-2B-Instruct` weight shards is currently too slow/unstable in this environment (large shard transfer stalls).

2. Current successful run uses CLIP grid matching fallback
- This is real inference, but it is weaker than instruction-following VLM grounding and does not predict DOM element ids directly.

3. Structured action quality ceiling
- CLIP grid approach provides coarse region grounding (grid-cell bbox), not token-level UI element localization.

4. Stage A SFT readiness dependency
- Need a fully downloaded and runnable Qwen2-VL backend (or an equivalent compact instruction-following VLM) for collecting high-quality supervised trajectories/candidates.
