# Stage 6A DPO-Style Preference Optimization

## Goal

在保持 Step 5c 特征与评估协议可比的前提下，完成一次轻量级 DPO-style 偏好优化实验，并与 Step 5c 最强基线直接对比。

## Baseline Inspected First

已先检查 `outputs/reranker_clip_grid_expanded_feature_upgrade/evaluation_summary.json`，Step 5c 当前最强结果为：

- full-pool reranked mean reward: `0.1679575959`
- headroom-subset reranked mean reward: `0.0925695906`
- full-pool best recovery: `0.7586206897`
- headroom-subset best recovery: `0.2222222222`

## Data Source and Schema

- Candidate artifact:
  - `outputs/candidate_generation_clip_grid_expanded/candidates_train_expanded.jsonl`
- 每个 pool 内候选字段含：
  - `candidate_id`, `action_type`, `grid_id`, `score`, `confidence`
  - `action_log_prob`, `grid_log_prob`, `joint_log_prob`
  - `bbox`, `click_point`
  - `reward.total_reward` 与 `reward.components`

## Preference Pair Construction (Auditable)

仅在**同一原始 sample/pool 内**构建偏好对：

- preferred: `reward.total_reward` 更高候选
- rejected: `reward.total_reward` 更低候选
- tie 过滤：`abs(diff) <= min_reward_diff` 时跳过（本次 `1e-4`）
- 不做跨 sample 比较

保存了可追踪偏好对工件（包含 `sample_id`, `pool_id`, `preferred/rejected candidate_id`, `reward_diff` 等）：

- `outputs/reranker_clip_grid_step6a_dpo_style/preference_pairs_train.jsonl`
- `outputs/reranker_clip_grid_step6a_dpo_style/preference_pairs_eval.jsonl`

## Implemented Optimization Formulation

本次实现是**轻量 DPO-style**（不是完整语言模型 canonical DPO）：

- 复用 Step 5c 的 candidate scorer + 特征表示
- policy scorer 从 Step 5c checkpoint 初始化
- reference scorer 使用冻结的 Step 5c checkpoint
- 对每个偏好对最小化：
  - `-log(sigmoid(beta * ((s_pos - s_neg) - (s_ref_pos - s_ref_neg))))`

其中 `s_*` 为 scorer 输出。  
这本质是对“相对参考模型优势”的 pairwise 偏好优化，适配当前候选打分场景。

## What Changed

1. `src/gui_grounding/training/trainer_reranker.py`
   - 增加 `optimization_mode` (`pairwise` / `dpo_style`)
   - 增加 DPO-style loss（参考模型冻结）
   - 增加 checkpoint 初始化与 reference checkpoint 支持
   - 增加偏好对导出（train/eval）
   - 增加 Step5c vs Step6A 对比表/图产出
2. `scripts/run_train_reranker.py`
   - 透传新增训练参数（mode、beta、checkpoint、baseline summary、导出开关）
3. 新增配置
   - `configs/train/reranker_clip_grid_step6a_dpo_style.yaml`

## Commands Actually Run

```bash
python -m py_compile \
  src/gui_grounding/training/trainer_reranker.py \
  src/gui_grounding/models/candidate_scorer.py \
  scripts/run_train_reranker.py

python scripts/run_train_reranker.py \
  --config configs/train/reranker_clip_grid_step6a_dpo_style.yaml \
  --dry-run

python scripts/run_train_reranker.py \
  --config configs/train/reranker_clip_grid_step6a_dpo_style.yaml
```

## Environment / Device

- Python: `3.13.5`
- PyTorch: `2.10.0+cu128`
- Device: `cuda` (`NVIDIA GeForce RTX 3090`)

## Training Counts

- total pools: `118`
- train pools: `89`
- eval pools: `29`
- train preference pairs: `3582`
- eval preference pairs: `1156`

## Evaluation (Same Protocol as Step 5c)

### Step 6A absolute metrics

- full-pool:
  - baseline(first-choice) mean reward: `0.1677918612`
  - Step6A mean reward: `0.1679575959`
  - oracle mean reward: `0.2027272483`
  - reward gain vs first-choice: `+0.0001657347`
  - rerank win rate: `0.0689655172`
  - best recovery: `0.7586206897`
- headroom-subset:
  - baseline(first-choice) mean reward: `0.0920355566`
  - Step6A mean reward: `0.0925695906`
  - oracle mean reward: `0.2046051372`
  - reward gain vs first-choice: `+0.0005340340`
  - rerank win rate: `0.2222222222`
  - best recovery: `0.2222222222`

### Step 6A vs Step 5c (main comparison)

对比 `outputs/reranker_clip_grid_expanded_feature_upgrade/evaluation_summary.json`：

- full-pool Step6A mean reward - Step5c mean reward: `0.0`
- headroom-subset Step6A mean reward - Step5c mean reward: `0.0`
- full-pool best recovery delta: `0.0`
- headroom-subset best recovery delta: `0.0`

即：Step 6A **未超过** Step 5c。

## Artifacts

- preference pairs:
  - `outputs/reranker_clip_grid_step6a_dpo_style/preference_pairs_train.jsonl`
  - `outputs/reranker_clip_grid_step6a_dpo_style/preference_pairs_eval.jsonl`
- checkpoint:
  - `outputs/reranker_clip_grid_step6a_dpo_style/checkpoint-best/model.pt`
  - `outputs/reranker_clip_grid_step6a_dpo_style/checkpoint-best/meta.json`
- summaries:
  - `outputs/reranker_clip_grid_step6a_dpo_style/evaluation_summary.json`
  - `outputs/reranker_clip_grid_step6a_dpo_style/training_history.json`
- tables:
  - `outputs/reranker_clip_grid_step6a_dpo_style/comparison_table.md`
  - `outputs/reranker_clip_grid_step6a_dpo_style/comparison_step5c_vs_step6a.md`
- figures:
  - `outputs/reranker_clip_grid_step6a_dpo_style/comparison_reward_full_vs_headroom.png`
  - `outputs/reranker_clip_grid_step6a_dpo_style/comparison_step5c_vs_step6a_oracle.png`

## Did Step 6A Improve over Step 5c?

没有。  
Step 6A 在 full pool 与 headroom subset 上均与 Step 5c 持平（delta = 0）。

## Exact Next Decision

按硬规则：

- Step 6A **没有**在 headroom subset 上超过 Step 5c，
- 因此当前**不建议直接进入 Step 6B / GRPO-light**。

下一步应留在 preference/reranker 设计域，优先做：

- reference/policy 差异更强的偏好目标（避免收敛到与 Step5c 等价解），或
- listwise / 温度校准 / pair weighting（按 reward_diff）的小规模针对性改造，
- 并继续使用同一 full/headroom/oracle 协议验收。
