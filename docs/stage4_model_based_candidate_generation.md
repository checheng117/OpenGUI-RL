# Stage 4 Model-Based Candidate Generation

## Scope

Goal: replace scaffold dummy candidate generation with a real model-based export path using the trained Stage-A checkpoint.

This step is explicitly **CLIP-grid-based model candidate generation**, not Qwen2-VL candidate generation.

## What Was Implemented

1. Reworked candidate export entrypoint:
- `scripts/run_generate_candidates.py`
- Now loads:
  - real Mind2Web samples
  - Stage-A trained CLIP-grid checkpoint
  - CLIP processor + model for inference

2. Real top-k model hypotheses:
- Computes `action_logits` + `grid_logits`.
- Builds top-k candidate combinations from top action/grid logits.
- Produces candidate fields:
  - `action_type`
  - `grid_id`
  - `score`
  - `confidence`
  - `bbox`
  - `click_point`
  - `rank`

3. Deterministic reward integration:
- Uses `VerifiableRewardCalculator` for every candidate.
- Attaches:
  - `reward.total_reward`
  - `reward.is_valid_format`
  - `reward.components.*`

4. JSONL artifacts for reranker construction:
- Sample-level JSONL (`candidates` list per sample).
- Flat candidate-level JSONL (one candidate per line).

## Config Added

- `configs/train/generate_candidates_clip_grid.yaml`

Key settings:
- backend: `clip_grid_stagea`
- checkpoint: `outputs/sft_clip_grid_debug_rerun/checkpoint-best`
- top_k: `6` (greater than 1)
- sample count: `12`

## Commands Actually Run

```bash
python -m py_compile scripts/run_generate_candidates.py

python scripts/run_generate_candidates.py \
  --config configs/train/generate_candidates_clip_grid.yaml \
  --dry-run

python scripts/run_generate_candidates.py \
  --config configs/train/generate_candidates_clip_grid.yaml
```

## Environment Used

- Python: 3.13.5
- Torch: 2.10.0+cu128
- Transformers: 5.3.0
- GPU used: yes (`cuda`, RTX 3090)
- HF_TOKEN in process env: not present (fallback to local HF cache/session)

## Checkpoint Used

- `outputs/sft_clip_grid_debug_rerun/checkpoint-best/model.pt`

## Run Summary

- Backend: `clip_grid_stagea`
- Split: `train`
- Sample count: `12`
- Top-k: `6`
- Total exported candidates: `72`

From summary artifact:
- `avg_reward`: `0.15138687074377577`
- `avg_best_reward_per_sample`: `0.15654532566637872`

## Exported Schema Example

Sample-level JSONL line includes:
- sample metadata (`sample_id`, `instruction`, `image_path`, targets)
- `candidates` list

Each candidate includes:
- `candidate_id`
- `rank`
- `action_type`
- `grid_id`
- `score`
- `confidence`
- `joint_log_prob`
- `action_log_prob`
- `grid_log_prob`
- `bbox`
- `click_point`
- `source`
- `reward.total_reward`
- `reward.is_valid_format`
- `reward.components`

## Artifact Paths

- Sample-level candidates:
  - `outputs/candidate_generation_clip_grid/candidates_train.jsonl`
- Flat candidates:
  - `outputs/candidate_generation_clip_grid/candidates_train_flat.jsonl`
- Summary:
  - `outputs/candidate_generation_clip_grid/summary_train.json`

## Files Changed

- `scripts/run_generate_candidates.py`
- `configs/train/generate_candidates_clip_grid.yaml`
- `docs/stage4_model_based_candidate_generation.md`

## Current Weaknesses / Limitations

1. Backend limitation:
- Candidate generation currently depends on CLIP-grid Stage-A model; not yet upgraded to Qwen2-VL candidate backbone.

2. Candidate diversity:
- Current checkpoint often prefers one action type (`click`) and relies mainly on grid variation.

3. Coarse localization:
- Grid-based candidates are coarse compared with element-level grounding.

## What Remains Before Stage 5 Learned Reranker Training

1. Connect reranker trainer input directly to exported JSONL:
- Use `candidates_train.jsonl` (sample-level) or `candidates_train_flat.jsonl` (flat) as reranker supervision source.

2. Build reranker dataset adapter:
- Parse candidate list + reward fields into pairwise/listwise training examples.

3. Add train/val split handling for candidate sets:
- Ensure reranker sees disjoint sample splits and can report reward gain metrics.
