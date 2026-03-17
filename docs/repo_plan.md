# Repository Architecture Plan

## Design Philosophy

This repository follows a **layered, stage-gated** design:

1. **Data layer** — Canonical schemas, dataset adapters, preprocessing
2. **Model layer** — VLM backbone, task-specific heads, scorer/adapter interfaces
3. **Reward layer** — Deterministic, verifiable reward computation
4. **Training layer** — Four trainer variants (SFT → Reranker → Preference → GRPO)
5. **Evaluation layer** — Metrics, evaluators, error analysis

Each layer depends only on the layers below it, ensuring testability and clean separation of concerns.

## Why This Structure?

| Decision | Rationale |
|----------|-----------|
| `src/` layout | Standard Python packaging; avoids import conflicts |
| Pydantic schemas | Validated data contracts between modules; self-documenting |
| Separate reward module | Reward logic is deterministic and testable independently of models |
| Four trainer variants | Maps directly to the four experimental stages in the proposal |
| YAML configs | Reproducible experiments without code changes |
| OmegaConf | Supports dot-list overrides from CLI, merge, interpolation |

## Module Responsibilities

### `data/`
- `schemas.py` — Canonical `GroundingSample`, `CandidateAction`, `PredictionResult`
- `*_dataset.py` — Convert raw dataset files into `GroundingSample` lists
- `collators.py` — Batch construction for training
- `preprocessors.py` — Image loading, normalization, prompt formatting

### `models/`
- `vlm_backbone.py` — Wraps any VLM (Qwen2-VL, etc.) behind a common API
- `grounding_head.py` / `action_head.py` — Task-specific output layers
- `candidate_scorer.py` — Ranks candidate actions (used in Stage B/C)
- `policy_adapter.py` — Adapts the model for different policy-improvement methods

### `reward/`
- `verifiable_reward.py` — Composite reward r = λ1·elem + λ2·IoU + λ3·click + λ4·act − λ5·penalty
- `candidate_generator.py` — Produces top-K candidates per sample

### `training/`
- `trainer_sft.py` — Stage A: standard supervised fine-tuning
- `trainer_reranker.py` — Stage B: train scorer from reward labels
- `trainer_pairwise.py` — Stage C-1: DPO-style preference optimization
- `trainer_grpo_light.py` — Stage C-2: Lightweight GRPO / contextual bandit

### `evaluation/`
- `metrics.py` — All standard metrics (element acc, point acc, IoU, action acc, reranking gain)
- `evaluator_grounding.py` — Single-split evaluation pipeline
- `evaluator_transfer.py` — Multi-split generalization study
- `error_analysis.py` — Error categorization and summaries

## Development Order

| Priority | Task | Dependencies |
|----------|------|-------------|
| **P0** | Download Mind2Web + ScreenSpot-v2 data | None |
| **P0** | Verify dataset adapters load real data | Data download |
| **P1** | Integrate Qwen2-VL backbone loading | HuggingFace access |
| **P1** | Implement SFT training loop | Backbone + data |
| **P2** | Model-based candidate generation | Trained SFT model |
| **P2** | Train reranker from reward labels | Candidates + rewards |
| **P3** | Pairwise preference optimization | Preference pairs |
| **P3** | GRPO-light training | Reward + candidate pipeline |
| **P4** | Cross-website transfer evaluation | All splits + model |
| **P4** | VisualWebBench + ScreenSpot-Pro eval | Optional datasets |
| **P5** | Polish demo, figures, report | All experiments |
