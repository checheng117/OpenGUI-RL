# Stage 5b Reranker Retry on Expanded Pool

## Scope

Goal: retrain and reevaluate the learned reranker on the expanded CLIP-grid candidate pool, and report:

- full-pool first-choice vs reranked
- headroom-subset first-choice vs reranked
- oracle reference over the same pool

This step remains a **CLIP-grid candidate reranker** (not Qwen2-VL reranker).

## What Changed

1. Added expanded reranker config:
- `configs/train/reranker_clip_grid_expanded_retry.yaml`

2. Extended reranker evaluation in `trainer_reranker.py`:
- full-pool metrics
- headroom-subset metrics (oracle > first-choice)
- oracle mean reward reference
- reward-best recovery rates
- baseline-to-oracle and reranked-to-oracle gaps

3. Updated comparison artifacts:
- table now includes full pool + headroom subset + oracle columns
- figure now shows baseline/reranked/oracle for both full and headroom scopes

No DPO/GRPO changes were made.

## Input Candidate Artifact Used

- `outputs/candidate_generation_clip_grid_expanded/candidates_train_expanded.jsonl`

## Reranker Formulation and Pair Construction

- Formulation: pairwise preference learning
- Supervision: deterministic `reward.total_reward`
- Pair construction:
  - compare candidates **only within the same original sample pool**
  - higher reward candidate = preferred
  - near ties filtered with `min_reward_diff=1e-4`

## Commands Actually Run

```bash
python -m py_compile src/gui_grounding/training/trainer_reranker.py scripts/run_train_reranker.py

python scripts/run_train_reranker.py \
  --config configs/train/reranker_clip_grid_expanded_retry.yaml \
  --dry-run

python scripts/run_train_reranker.py \
  --config configs/train/reranker_clip_grid_expanded_retry.yaml
```

## Environment / Device

- Python: 3.13.5
- Torch: 2.10.0+cu128
- Device used: `cuda` (RTX 3090)

## Training Sample Counts

From run outputs:

- total pools: `118`
- train pools: `89`
- eval pools: `29`
- pairwise train examples: `3582`

## Evaluation Results

From `outputs/reranker_clip_grid_expanded_retry/evaluation_summary.json`:

### Full pool (29 eval pools)

- baseline mean reward: `0.1678`
- reranked mean reward: `0.1678`
- reward gain: `+0.0000`
- rerank win rate: `0.0000`
- oracle mean reward: `0.2027`
- oracle upper-bound gain over baseline: `+0.0349`
- baseline reward-best recovery: `0.6897`
- reranked reward-best recovery: `0.6897`

### Headroom subset (9 pools where first-choice is not best)

- baseline mean reward: `0.0920`
- reranked mean reward: `0.0920`
- reward gain: `+0.0000`
- rerank win rate: `0.0000`
- oracle mean reward: `0.2046`
- oracle upper-bound gain over baseline: `+0.1126`
- baseline reward-best recovery: `0.0000`
- reranked reward-best recovery: `0.0000`

## Did Reranking Help?

No. On this retry:

- full pool: no measurable improvement
- headroom subset: still no measurable improvement

This is a real measured result, not a pipeline failure.

## Artifacts

- checkpoint:
  - `outputs/reranker_clip_grid_expanded_retry/checkpoint-best/model.pt`
  - `outputs/reranker_clip_grid_expanded_retry/checkpoint-best/meta.json`
- evaluation summary:
  - `outputs/reranker_clip_grid_expanded_retry/evaluation_summary.json`
- training history:
  - `outputs/reranker_clip_grid_expanded_retry/training_history.json`
- comparison table:
  - `outputs/reranker_clip_grid_expanded_retry/comparison_table.md`
- comparison figure:
  - `outputs/reranker_clip_grid_expanded_retry/comparison_reward_full_vs_headroom.png`

## Files Changed

- `src/gui_grounding/training/trainer_reranker.py`
- `configs/train/reranker_clip_grid_expanded_retry.yaml`
- `docs/stage5b_reranker_retry_expanded_pool.md`

## Interpretation and Next Step

### Is reranker useful now?

Not yet, based on measured gain (0 on both full and headroom subset).

### If no, what is likely bottleneck?

Given Step 5.5 already showed non-trivial headroom, the likely next bottleneck is now **reranker representation/features**, not candidate pool size alone.

### Exact next step

Before Step 6A (DPO-style), run one focused reranker feature redesign pass:

- add stronger per-candidate features (e.g., richer geometry + action/grid calibration + score calibration),
- optionally switch to listwise objective on the same pools,
- keep the same evaluation protocol (full + headroom + oracle reference).

If that still shows near-zero gain, proceed to Step 6A only with clear caveats.
