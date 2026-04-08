# Stage 5.5 Candidate Pool Expansion and Diagnostics

## Scope

Goal of this step:

1. Expand CLIP-grid model-based candidate pool scale.
2. Increase candidate diversity without fabricating candidates.
3. Diagnose rerankability headroom before retrying learned reranker training.

This step remains fully **CLIP-grid-based**, not Qwen2-VL.

## What Changed

### 1) Candidate generation scale + diversity

Updated `scripts/run_generate_candidates.py`:

- Added configurable candidate selection strategy:
  - `score_only` (previous behavior)
  - `diverse` (new)
- `diverse` strategy is still model-driven:
  - candidates still come from model logits (`action_logits`, `grid_logits`)
  - no heuristic-only or fabricated candidates
  - selection now encourages action coverage + grid coverage while preserving score ranking
- Added config fields:
  - `selection_strategy`
  - `max_per_action`

Added expanded config:

- `configs/train/generate_candidates_clip_grid_expanded.yaml`
  - `max_samples: 120`
  - `top_k: 12`
  - `top_action_k: 4`
  - `top_grid_k: 24`
  - `selection_strategy: diverse`
  - `max_per_action: 4`

### 2) Candidate pool diagnostics

Added `scripts/analyze_candidate_pool.py` to compute rerankability diagnostics from exported grouped JSONL.

Computed metrics include:

- number of pools
- candidates per pool
- reward max/min/mean and spread per pool
- fraction first-choice already reward-best
- fraction with positive headroom
- average oracle gain (best reward - first reward)
- action diversity / grid diversity
- duplicate ratio
- focused subset stats for pools where first-choice is not best

Artifacts include:

- summary JSON
- markdown table
- headroom bar figure
- oracle gain histogram figure

## Commands Actually Run

```bash
python -m py_compile scripts/run_generate_candidates.py scripts/analyze_candidate_pool.py

python scripts/run_generate_candidates.py \
  --config configs/train/generate_candidates_clip_grid_expanded.yaml \
  --dry-run

python scripts/run_generate_candidates.py \
  --config configs/train/generate_candidates_clip_grid_expanded.yaml

python scripts/analyze_candidate_pool.py \
  --input outputs/candidate_generation_clip_grid_expanded/candidates_train_expanded.jsonl \
  --output-dir outputs/candidate_generation_clip_grid_expanded/diagnostics
```

## Environment

- Python: 3.13.5
- Torch: 2.10.0+cu128
- Device used: `cuda` (RTX 3090)
- Checkpoint used:
  - `outputs/sft_clip_grid_debug_rerun/checkpoint-best/model.pt`

## Expanded Export Result

From `outputs/candidate_generation_clip_grid_expanded/summary_train_expanded.json`:

- pools: `118`
- top_k: `12`
- total candidates: `1416`
- selection strategy: `diverse`

Compared with previous run:

- previous pools: `12` -> now `118` (**9.83x larger**)
- previous top_k: `6` -> now `12` (**2x per pool**)
- previous total candidates: `72` -> now `1416` (**19.67x larger**)

## Key Diagnostic Findings

From `outputs/candidate_generation_clip_grid_expanded/diagnostics/candidate_pool_diagnostics_summary.json`:

- `frac_first_choice_best`: `0.6695`
- `frac_positive_headroom`: `0.3305`
- `avg_oracle_gain`: `0.0271`
- `subset_not_best_count`: `39` pools
- `subset_not_best_avg_oracle_gain`: `0.0819`
- `avg_action_diversity_per_pool`: `4.0`
- `avg_grid_diversity_per_pool`: `9.0`
- `avg_duplicate_ratio_per_pool`: `0.0`

## Direct Answers to Required Questions

### Q1: How often is first-choice already reward-best?

- About **66.95%** of pools.

### Q2: How often is there actual improvement headroom?

- About **33.05%** of pools have positive headroom (`best_reward > first_reward`).

### Q3: What is the oracle upper bound over current pool?

- Mean achievable oracle gain over all pools: **+0.0271 reward**.
- On the headroom subset only, mean oracle gain: **+0.0819 reward**.

Interpretation:

- The pool now contains non-trivial rerankable signal.
- Headroom exists but is concentrated in a subset, so reranker improvement is possible but not guaranteed to be large.

## Artifact Paths

Expanded candidate export:

- `outputs/candidate_generation_clip_grid_expanded/candidates_train_expanded.jsonl`
- `outputs/candidate_generation_clip_grid_expanded/candidates_train_expanded_flat.jsonl`
- `outputs/candidate_generation_clip_grid_expanded/summary_train_expanded.json`

Diagnostics:

- `outputs/candidate_generation_clip_grid_expanded/diagnostics/candidate_pool_diagnostics_summary.json`
- `outputs/candidate_generation_clip_grid_expanded/diagnostics/candidate_pool_diagnostics_table.md`
- `outputs/candidate_generation_clip_grid_expanded/diagnostics/candidate_pool_headroom.png`
- `outputs/candidate_generation_clip_grid_expanded/diagnostics/candidate_pool_oracle_gain_hist.png`

## Files Changed

- `scripts/run_generate_candidates.py`
- `scripts/analyze_candidate_pool.py`
- `configs/train/generate_candidates_clip_grid_expanded.yaml`
- `docs/stage5_5_candidate_pool_expansion_and_diagnostics.md`

## Recommended Next Step (Before Reranker Retry)

Retry learned reranker training using the expanded grouped JSONL:

- Input:
  - `outputs/candidate_generation_clip_grid_expanded/candidates_train_expanded.jsonl`
- Keep the same evaluation protocol:
  - first-choice vs reranked on the same candidate pools
- Report:
  - reward gain
  - win rate
  - subset-specific gain on pools with headroom

If gain is still near zero, likely next bottleneck is scorer feature expressiveness rather than pool size.
