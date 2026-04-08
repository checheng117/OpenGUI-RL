# Stage 6A.5 Preference Target Redesign

## Goal

在不改候选池、不改 Step5c 特征表示的前提下，做一次小而聚焦的偏好目标/配对策略重设计，验证能否超过当前最强基线（Step 5c），尤其在 headroom subset 上。

## What Was Inspected First

已先检查以下真实产物与代码：

- `outputs/reranker_clip_grid_step6a_dpo_style/preference_pairs_train.jsonl`
- `outputs/reranker_clip_grid_step6a_dpo_style/preference_pairs_eval.jsonl`
- `outputs/reranker_clip_grid_step6a_dpo_style/evaluation_summary.json`
- `outputs/reranker_clip_grid_expanded_feature_upgrade/evaluation_summary.json`
- `src/gui_grounding/training/trainer_reranker.py`
- `configs/train/reranker_clip_grid_step6a_dpo_style.yaml`

## Step 6A Pair Construction Diagnosis (Code/Schema Grounded)

从 Step 6A 偏好对可见：

- 大量全量两两配对（同池内全部组合）；
- 包含很多很小 reward gap（如 `0.004~0.01`）；
- 这些小 gap 对“纠正 baseline 错误选择”帮助有限，可能主要增加噪声；
- 与此同时，Step 6A policy/reference 都从 Step5c 同 checkpoint 初始化，易收敛到与 Step5c 等价解。

因此本步优先提升“偏好信号质量”而非数量。

## Preference Redesign Chosen (Only 2 Focused Changes)

### 1) Headroom + Hard-Negative Pair Construction

新增 `pair_construction_mode=headroom_hard_negative`：

- 只在有 headroom 的池子上构建监督（`oracle_reward > first_choice_reward`）；
- 只构建“能纠正 baseline(first-choice)错误”的对：
  - preferred = reward 更高候选
  - rejected = baseline(first-choice)候选

即：直接对齐“我们真正想改正的决策错误”。

### 2) Gap-Aware Filtering + Weighting

- `pair_reward_gap_threshold=0.01`：过滤极小 gap 偏好对
- `pair_weight_mode=reward_gap`：按 reward_diff 加权（线性上升+cap）

目标：让大 gap、高置信偏好对贡献更大。

## Implementation Changes

- `src/gui_grounding/training/trainer_reranker.py`
  - 新增 pair 构造模式、pair 权重模式、gap 过滤与权重超参
  - loss 支持样本权重（pairwise 和 dpo_style 均支持）
  - 偏好对导出新增 `pair_weight` 与构造元信息
  - 新增 Step5c / Step6A / Step6A.5 / Oracle 四方对比表与图
- `scripts/run_train_reranker.py`
  - 透传上述新配置项
- 新配置：
  - `configs/train/reranker_clip_grid_step6a5_preference_redesign.yaml`

## Commands Actually Run

```bash
python -m py_compile src/gui_grounding/training/trainer_reranker.py scripts/run_train_reranker.py

python scripts/run_train_reranker.py \
  --config configs/train/reranker_clip_grid_step6a5_preference_redesign.yaml \
  --dry-run

python scripts/run_train_reranker.py \
  --config configs/train/reranker_clip_grid_step6a5_preference_redesign.yaml
```

## Environment / Device

- Python: `3.13.5`
- Torch: `2.10.0+cu128`
- Device: `cuda` (`NVIDIA GeForce RTX 3090`)

## Data / Pair Counts

- Candidate source:
  - `outputs/candidate_generation_clip_grid_expanded/candidates_train_expanded.jsonl`
- 本次 pair 构造后：
  - train pairs: `44`
  - eval pairs: `17`

（相比 Step6A 的 3582/1156，数量显著减少，信号更聚焦）

## Evaluation (Protocol Kept Comparable)

### Step 6A.5 absolute metrics

- full-pool reranked mean reward: `0.1679575959`
- headroom-subset reranked mean reward: `0.0925695906`
- full-pool rerank win rate: `0.0689655172`
- headroom-subset rerank win rate: `0.2222222222`
- full-pool best recovery: `0.7586206897`
- headroom-subset best recovery: `0.2222222222`

### Step 6A.5 vs Step 5c

- full-pool mean reward delta: `0.0`
- headroom-subset mean reward delta: `0.0`
- full-pool best recovery delta: `0.0`
- headroom-subset best recovery delta: `0.0`

### Step 6A.5 vs Step 6A

- full/headroom mean reward delta: `0.0 / 0.0`
- full/headroom win rate delta: `0.0 / 0.0`
- full/headroom best recovery delta: `0.0 / 0.0`

结论：Step 6A.5 没有超过 Step 5c，也没有超过 Step 6A（最终最佳点与二者持平）。

## Artifacts

- redesigned preference pairs:
  - `outputs/reranker_clip_grid_step6a5_preference_redesign/preference_pairs_train.jsonl`
  - `outputs/reranker_clip_grid_step6a5_preference_redesign/preference_pairs_eval.jsonl`
- checkpoint:
  - `outputs/reranker_clip_grid_step6a5_preference_redesign/checkpoint-best/model.pt`
- summaries:
  - `outputs/reranker_clip_grid_step6a5_preference_redesign/evaluation_summary.json`
  - `outputs/reranker_clip_grid_step6a5_preference_redesign/training_history.json`
- comparisons:
  - `outputs/reranker_clip_grid_step6a5_preference_redesign/comparison_table.md`
  - `outputs/reranker_clip_grid_step6a5_preference_redesign/comparison_step5c_step6a_step6a5.md`
- figures:
  - `outputs/reranker_clip_grid_step6a5_preference_redesign/comparison_reward_full_vs_headroom.png`
  - `outputs/reranker_clip_grid_step6a5_preference_redesign/comparison_step5c_step6a_step6a5_oracle.png`

## Why This Redesign Still Didn't Beat Step 5c

本次重设计确实改变了监督结构，但从结果看：

- 信号更聚焦也更稀疏（pair 数从 3582 降到 44），
- 训练后最优点仍回到与 Step5c 等价性能，
- 说明当前“偏好对构造调整”仍不足以带来可泛化增益。

## Exact Next Decision

按硬规则：

- Step 6A.5 未超过 Step 5c（尤其 headroom subset 未提升），
- **不建议进入 Step 6B / GRPO-light**。

下一步应继续停留在偏好目标设计域，但保持小步：

- 优先尝试“同样 headroom-focused，但不过度稀疏”的折中配对策略（例如 headroom pools 内 top-m hard negatives，而非仅 baseline 对）；
- 或者尝试更稳定的 listwise 目标（仍保持同评估协议）。
