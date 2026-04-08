# Stage 3 SFT Baseline

## Scope

Goal: implement a **real Stage A supervised fine-tuning baseline** that actually trains, validates, saves checkpoints, and logs metrics, while staying aligned with the currently executable backend in this environment.

Validation date: 2026-03-31

## Practical Backend Decision

- **Intended long-term backbone**: Qwen2-VL family.
- **Current executable baseline backend**: CLIP-grid (`openai/clip-vit-base-patch32` + trainable heads).
- Reason: Qwen2-VL full weight download/cache remains slow/unstable on this machine; CLIP backend is currently the most reliable runnable path for real train/val execution.

This is explicit and temporary. The code keeps a clean upgrade path to Qwen2-VL later.

## Target Formulation (Stage A)

Supervised targets from real Mind2Web samples:

1. **Action type classification** (`click/type/select/hover`)
2. **Grid-cell classification** (4x6 grid by default)
   - Convert GT bbox center to a grid id.
   - Predicted grid id maps back to bbox/click-point for downstream reuse.

Loss:

- `L = w_action * CE(action_logits, action_label) + w_grid * CE(grid_logits, grid_label)`
- default `w_action=1.0`, `w_grid=1.0`

Validation metrics:

- `val_loss`
- `val_action_acc`
- `val_grid_acc`
- `val_point_acc` (predicted grid center inside GT bbox)
- `val_mean_iou` (predicted grid box vs GT bbox)

## What Was Implemented

1. Real trainable model for Stage A:
- `src/gui_grounding/models/sft_clip_grid_model.py`
- CLIP embeddings + fusion MLP + action/grid heads.

2. Real SFT trainer:
- `src/gui_grounding/training/trainer_sft.py`
- Includes:
  - torch dataset + dataloader + collator
  - forward/loss/backprop
  - optimizer + scheduler
  - periodic validation
  - checkpoint save (`model.pt`, `optimizer.pt`, `scheduler.pt`, `trainer_state.pt`, `meta.json`)
  - eval history and training summary artifacts
  - wandb logging when available, graceful local fallback when unavailable

3. Training entrypoint repaired:
- `scripts/run_train_sft.py`
- Loads env robustly, loads Mind2Web data, splits train/val holdout, builds model/trainer, runs training.
- Added safe post-run hard exit to avoid Python 3.13 finalization crash after successful completion.

4. New configs:
- Debug/smoke: `configs/train/sft_clip_grid_debug.yaml`
- Baseline: `configs/train/sft_clip_grid_baseline.yaml`

## Commands Actually Run

```bash
python -m py_compile scripts/run_train_sft.py src/gui_grounding/training/trainer_sft.py src/gui_grounding/models/sft_clip_grid_model.py

python scripts/run_train_sft.py --config configs/train/sft_clip_grid_debug.yaml --dry-run

python scripts/run_train_sft.py --config configs/train/sft_clip_grid_debug.yaml

python scripts/run_train_sft.py --config configs/train/sft_clip_grid_debug.yaml \
  data.train.max_samples=40 data.eval.max_samples=40 \
  training.output_dir=outputs/sft_clip_grid_debug_rerun
```

Notes:
- First full debug run completed training/validation/checkpoint but hit Python 3.13 finalization crash at process exit.
- After adding safe hard-exit, rerun finished with exit code 0.

## Environment Used

- Python: `3.13.5`
- Torch: `2.10.0+cu128`
- Transformers: `5.3.0`
- GPU: RTX 3090 (`cuda` used)
- HF_TOKEN: not present in process env during run (fallback to HF session/cache)
- WANDB_API_KEY: not present in process env during run (wandb disabled gracefully; local logging used)

## Dataset Slice / Split Used

Successful debug rerun:

- Source split: `train` (Mind2Web real data)
- Supervised filter: samples with both `action_type` and `target_bbox`
- `max_samples=40`
- Holdout split mode: `single_split_holdout`, `val_ratio=0.2`
- Resulting supervised sets: `train=32`, `eval=8`

## Losses and Metrics Logged (Debug Rerun)

From `outputs/sft_clip_grid_debug_rerun/eval_history.json`:

- `val_loss`: `4.4274`
- `val_action_loss`: `1.3090`
- `val_grid_loss`: `3.1184`
- `val_action_acc`: `0.75`
- `val_grid_acc`: `0.50`
- `val_point_acc`: `0.00`
- `val_mean_iou`: `0.0071`

## Checkpoint Paths

- `outputs/sft_clip_grid_debug_rerun/checkpoint-best/`
- `outputs/sft_clip_grid_debug_rerun/checkpoint-latest/`

Each checkpoint directory contains:

- `model.pt`
- `optimizer.pt`
- `scheduler.pt`
- `trainer_state.pt`
- `meta.json`

## Artifact Paths

- Train summary: `outputs/sft_clip_grid_debug_rerun/train_summary.json`
- Eval history: `outputs/sft_clip_grid_debug_rerun/eval_history.json`

## Files Changed

- `scripts/run_train_sft.py`
- `src/gui_grounding/training/trainer_sft.py`
- `src/gui_grounding/models/sft_clip_grid_model.py`
- `src/gui_grounding/models/__init__.py`
- `configs/train/sft_clip_grid_debug.yaml`
- `configs/train/sft_clip_grid_baseline.yaml`
- `docs/stage3_sft_baseline.md`

## What Remains Before Stage B Candidate Generation

1. Replace coarse grid target with stronger proposal mechanism
- current Stage A predicts coarse region id + action type; this is enough for a real baseline but not ideal candidate diversity/precision.

2. Add candidate decoding interface from trained Stage A model
- expose top-k grid/action candidates with scores for downstream reranking.

3. Upgrade backbone path to Qwen2-VL once cache/download is stable
- keep Stage A pipeline structure, swap model backend and supervision head as needed.

4. Optional: enable wandb online tracking in training environment
- set `WANDB_API_KEY` and rerun baseline config for experiment registry.
