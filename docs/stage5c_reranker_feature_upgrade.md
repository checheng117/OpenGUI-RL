# Stage 5c Reranker Feature Upgrade

## Goal

在不改候选池生成的前提下，对 reranker 做一次单点特征/表征升级，验证是否能在同一 expanded pool 上取得可测提升（尤其 headroom subset）。

## Inputs and Baseline Context

- Candidate pool artifact:
  - `outputs/candidate_generation_clip_grid_expanded/candidates_train_expanded.jsonl`
- Step 5b baseline (same pool, same protocol):
  - full-pool gain = `0`
  - headroom-subset gain = `0`

## Representation Diagnosis (Grounded in Current Code/Schema)

基于 `trainer_reranker.py` 和候选 schema 的实查：

- 旧特征仅 17 维，主要是候选绝对值（score/conf/log-prob + 少量几何）。
- 缺少 pool 内相对信息（如相对 pool 最优、z-score、分位、action 内局部排序）。
- expanded schema中有可用字段（`score`/`confidence`/`joint_log_prob`/`action_log_prob`/`grid_log_prob`/`rank`/`action_type`/`grid_id`/`bbox`/`click_point`），但旧特征没有充分利用“同池相对结构”。

这与 Step 5b 的“排序几乎不变”现象一致，是信息瓶颈的高概率来源。

## Focused Upgrade Implemented

### 1) Feature Builder Upgrade (single focused pass)

在 `src/gui_grounding/training/trainer_reranker.py` 中，将候选特征从旧 17 维扩展到 36 维，新增两类 inference-time 特征（不使用 reward 标签）：

- 几何增强：
  - `x1n,y1n,x2n,y2n`
  - `aspect_ratio`
  - `center_dist`
- pool-relative 增强：
  - `score - max/mean`, `score z-score`, `score percentile`
  - `jlp - max/mean`, `jlp z-score`
  - `alp - max`, `glp - max`
  - `conf - max`, `conf/max`, `conf z-score`
  - `action_local_rank_norm`（同 action 组内按 joint log-prob 排序）
  - `grid_id_norm`
  - `rank_norm - score_percentile`

### 2) Scorer Input Compatibility

在 `src/gui_grounding/models/candidate_scorer.py` 中将首层改为 `nn.LazyLinear`，让输入维度可随特征升级自动适配，并保持旧配置可运行（同时保留 `input_dim` 元数据）。

### 3) New Config

- `configs/train/reranker_clip_grid_expanded_feature_upgrade.yaml`
  - data path 保持 expanded pool 不变
  - `input_dim: 36`
  - 其余训练参数与 Step 5b 保持同量级（无大 sweep）

## Commands Actually Run

```bash
python -m py_compile src/gui_grounding/models/candidate_scorer.py \
  src/gui_grounding/training/trainer_reranker.py \
  scripts/run_train_reranker.py

python scripts/run_train_reranker.py \
  --config configs/train/reranker_clip_grid_expanded_feature_upgrade.yaml \
  --dry-run

python scripts/run_train_reranker.py \
  --config configs/train/reranker_clip_grid_expanded_feature_upgrade.yaml
```

## Environment and Device

- Python: `3.13.5`
- PyTorch: `2.10.0+cu128`
- Device: `cuda` (`NVIDIA GeForce RTX 3090`)

## Training Counts

来自本次实际运行：

- total pools: `118`
- train pools: `89`
- eval pools: `29`
- pairwise train examples: `3582`

## Comparable Evaluation Results (Same Protocol as Step 5b)

### Full Pool

- baseline mean reward: `0.1677918612`
- reranked mean reward: `0.1679575959`
- oracle mean reward: `0.2027272483`
- reward gain: `+0.0001657347`
- rerank win rate: `0.0689655172`
- reward-best recovery:
  - baseline: `0.6896551724`
  - reranked: `0.7586206897`
- oracle gap:
  - baseline to oracle: `0.0349353871`
  - reranked to oracle: `0.0347696524`

### Headroom Subset

- subset size: `9`
- baseline mean reward: `0.0920355566`
- reranked mean reward: `0.0925695906`
- oracle mean reward: `0.2046051372`
- reward gain: `+0.0005340340`
- rerank win rate: `0.2222222222`
- reward-best recovery:
  - baseline: `0.0`
  - reranked: `0.2222222222`
- oracle gap:
  - baseline to oracle: `0.1125695806`
  - reranked to oracle: `0.1120355466`

## Is Upgraded Reranker Useful?

是，**但幅度很小**：

- full-pool gain 从 0 变为正值（`+0.0001657`）
- headroom-subset gain 从 0 变为正值（`+0.0005340`）
- headroom subset 的 reward-best recovery 从 `0` 提升到 `0.2222`

这说明表征升级确实带来了可测改进，但距离 oracle 仍很远。

## Artifacts

- checkpoint:
  - `outputs/reranker_clip_grid_expanded_feature_upgrade/checkpoint-best/model.pt`
  - `outputs/reranker_clip_grid_expanded_feature_upgrade/checkpoint-best/meta.json`
- evaluation summary:
  - `outputs/reranker_clip_grid_expanded_feature_upgrade/evaluation_summary.json`
- training history:
  - `outputs/reranker_clip_grid_expanded_feature_upgrade/training_history.json`
- comparison table:
  - `outputs/reranker_clip_grid_expanded_feature_upgrade/comparison_table.md`
- figure:
  - `outputs/reranker_clip_grid_expanded_feature_upgrade/comparison_reward_full_vs_headroom.png`

## Files Changed

- `src/gui_grounding/models/candidate_scorer.py`
- `src/gui_grounding/training/trainer_reranker.py`
- `configs/train/reranker_clip_grid_expanded_feature_upgrade.yaml`
- `docs/stage5c_reranker_feature_upgrade.md`

## Exact Next Decision

根据硬规则（headroom-subset gain > 0）：

- 本次已出现 **非零正增益**，因此 **Step 6A（DPO-style）可以进入**。
- 但建议带着 caveat 进入：当前增益较小，Step 6A 前后都应保留同一 full/headroom/oracle 评估协议，持续监控是否有实质提升。
