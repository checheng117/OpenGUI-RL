# Stage 5 Learned Reranker Results

## Scope

Goal: train a learned reranker on real model-based candidates and compare:

- first-choice baseline (raw Stage-A candidate rank 1)
- reranked top-1 (learned scorer over the same candidate pool)

This step is fully based on **CLIP-grid-generated candidates**, not Qwen2-VL candidates.

## What Was Implemented

1. Real learned candidate scorer
- `src/gui_grounding/models/candidate_scorer.py`
- Upgraded from scaffold to trainable MLP (`CandidateScorer` as `nn.Module`).

2. Real reranker trainer
- `src/gui_grounding/training/trainer_reranker.py`
- Implemented:
  - loading grouped candidate JSONL
  - pairwise preference construction from reward labels
  - **pairwise pairs only within the same original sample**
  - training loop with pairwise logistic loss
  - evaluation on the same candidate pool: first-choice vs reranked
  - checkpoint saving
  - comparison table + figure artifact export

3. Real training entrypoint wiring
- `scripts/run_train_reranker.py`
- Loads config, builds scorer/trainer, trains and evaluates.
- Includes clean hard-exit workaround for Python 3.13 finalization issue.

4. New config
- `configs/train/reranker_clip_grid.yaml`

## Reranker Formulation

- Formulation: **pairwise preference learning**
- Supervision source: deterministic `reward.total_reward`
- Pair rule:
  - For two candidates from the same sample, higher reward is preferred.
  - Pairs with near-equal rewards (`|diff| <= min_reward_diff`) are skipped.
- Loss:
  - `-log(sigmoid(s_pos - s_neg - margin))`

## Data Source Used

- Input candidate file:
  - `outputs/candidate_generation_clip_grid/candidates_train.jsonl`
- This file is model-based CLIP-grid top-k output from Step 4.

## Commands Actually Run

```bash
python -m py_compile scripts/run_train_reranker.py src/gui_grounding/training/trainer_reranker.py src/gui_grounding/models/candidate_scorer.py

python scripts/run_train_reranker.py --config configs/train/reranker_clip_grid.yaml --dry-run

python scripts/run_train_reranker.py --config configs/train/reranker_clip_grid.yaml
```

## Environment / Device

- Python: 3.13.5
- Torch: 2.10.0+cu128
- Device used: `cuda` (RTX 3090)

## Training / Eval Counts

- Total candidate pools (samples): `12`
- Train pools: `9`
- Eval pools: `3`
- Pairwise train examples: `47`
- Epochs: `20`

## Baseline vs Reranked (Same Candidate Pool)

From `outputs/reranker_clip_grid/evaluation_summary.json`:

- Baseline mean reward: `0.2017`
- Reranked mean reward: `0.2017`
- Reward gain: `+0.0000`
- Baseline mean IoU: `0.0035`
- Reranked mean IoU: `0.0035`
- Baseline action_type_correct: `1.0000`
- Reranked action_type_correct: `1.0000`
- Baseline click_inside_target: `0.0000`
- Reranked click_inside_target: `0.0000`
- Rerank win rate: `0.0000`

## Did Reranking Help?

On this run: **No measurable improvement**.

This is reported honestly. The current candidate pool appears low-diversity (many samples already have similar top candidate reward patterns), limiting gain for learned reranking on this small subset.

## Artifacts

- Checkpoint:
  - `outputs/reranker_clip_grid/checkpoint-best/model.pt`
  - `outputs/reranker_clip_grid/checkpoint-best/meta.json`
- Evaluation summary:
  - `outputs/reranker_clip_grid/evaluation_summary.json`
- Training history:
  - `outputs/reranker_clip_grid/training_history.json`
- Comparison table:
  - `outputs/reranker_clip_grid/comparison_table.md`
- Figure:
  - `outputs/reranker_clip_grid/comparison_reward_bar.png`

## Files Changed

- `src/gui_grounding/models/candidate_scorer.py`
- `src/gui_grounding/training/trainer_reranker.py`
- `scripts/run_train_reranker.py`
- `configs/train/reranker_clip_grid.yaml`
- `docs/stage5_learned_reranker_results.md`

## Current Weaknesses / Limitations

1. Very small candidate dataset (12 pools) for learned reranking.
2. CLIP-grid candidate diversity is limited; action logits are often peaked similarly.
3. Evaluation set is tiny (`3` pools), so signal is noisy and weak.

## What Remains Before Step 6

1. Scale candidate export size (more samples, possibly larger top-k) to increase reranking signal.
2. Re-train reranker on larger pool and re-check baseline vs reranked gain.
3. Once pairwise signal is sufficiently rich, proceed to Step 6 (DPO-style preference optimization or GRPO-light) using the same preference structure.
