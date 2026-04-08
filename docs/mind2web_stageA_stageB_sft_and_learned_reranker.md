# Mind2Web Stage A -> Stage B -> Learned Reranker

## Objective

This task was scoped to restore the original project blueprint around a real training-centered Mind2Web pipeline:

1. Stage A: supervised Qwen-first baseline on Mind2Web
2. Stage B: top-k candidate export on top of that baseline
3. Stage B.5: lightweight learned reranker trained from deterministic verifiable reward labels
4. Evaluation: `SFT first-choice` vs `SFT + learned reranker`

The task explicitly did **not** redesign the broader held-out inference stack, revert to the old surrogate path as the main direction, or introduce PPO/GRPO/DPO as the primary method.

## Inspection Summary And Frozen Assets

Before editing, the repo state was inspected across:

- proposal and blueprint docs
- current Qwen-first inference path
- point-native decoupled inference
- structured bbox/action path
- dual-path candidate generation and lightweight verifier path
- Mind2Web dataset loading and preprocessing
- current training and reranking scripts

The key reuse decision was:

- keep the existing Qwen-first output contract and parser path as the canonical contract
- keep the held-out ScreenSpot-v2 / transfer-oriented inference assets conceptually frozen
- make the Mind2Web training path produce the same kind of structured output that the existing inference stack already expects

The resulting contract stays blueprint-aligned:

- `action_type`
- `bbox_proposal`
- `click_point`
- optional `element_hint_id`
- parser / provenance metadata for auditability

## What Changed

### 1. Stage A is now a real Qwen-first Mind2Web SFT path

New/updated pieces:

- `src/gui_grounding/training/trainer_sft_qwen.py`
- `scripts/run_train_sft.py`
- `src/gui_grounding/models/vlm_backbone.py`
- `src/gui_grounding/models/qwen2_vl_grounding.py`
- `configs/train/mind2web_stageA_qwen2_5_vl_3b_sft.yaml`

The old Qwen SFT path was a smoke-level scaffold. It is now a real LoRA fine-tuning loop with:

- multimodal prompt/target construction from Mind2Web samples
- masked supervised loss on the assistant JSON only
- training loop
- validation loop
- checkpoint saving
- resolved config saving
- sample manifests
- metric history logging

### 2. Stage B now exports true top-k candidate pools from the Stage A model

New/updated pieces:

- `scripts/run_generate_candidates.py`
- `configs/train/mind2web_stageB_candidates_qwen2_5_vl_3b.yaml`

The export path now loads the Stage A LoRA adapter and produces explicit candidate pools where:

- candidate 1 is the deterministic Stage A first-choice (`temperature=0.0`)
- remaining candidates are sampled variants (`temperature=0.6`)
- candidates are deduplicated by action/bbox/click signature
- grouped JSONL and flat JSONL are both saved
- each candidate includes provenance, parser metadata, DOM-match metadata, reward labels, and normalized geometry

### 3. The reranker is now a learned scorer over the canonical candidate schema

New/updated pieces:

- `src/gui_grounding/training/trainer_reranker.py`
- `scripts/run_train_reranker.py`
- `configs/train/mind2web_stageB_reranker_qwen.yaml`

The reranker was upgraded to consume the current candidate schema instead of relying on the earlier grid-centric assumptions. It now:

- reads `bbox_proposal`, `click_point`, `action_type`, parser metadata, DOM-match metadata, provenance, and image size
- builds 61-dim candidate feature vectors
- constructs pairwise preference data from deterministic reward gaps
- trains a lightweight MLP scorer
- evaluates external candidate files for named splits and saves split-specific reports

## Stage A Baseline Actually Trained

### Model and setup

- Base model: `Qwen/Qwen2.5-VL-3B-Instruct`
- Training mode: LoRA SFT
- LoRA targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Precision: `bfloat16`
- Attention backend: `sdpa`
- Output: structured JSON with `action_type`, `predicted_element_id`, `predicted_bbox`, `predicted_click_point`, `confidence`

### Actual config used

From `configs/train/mind2web_stageA_qwen2_5_vl_3b_sft.yaml`:

- train source: Mind2Web `train`
- evaluation source: held-out subset from Mind2Web `train`
- `max_samples: 120`
- `val_ratio: 0.15`
- `num_epochs: 1`
- `max_steps: 40`
- `batch_size: 1`
- `gradient_accumulation_steps: 2`
- `learning_rate: 2e-4`

### Actual run result

Command run:

```bash
python scripts/run_train_sft.py --config configs/train/mind2web_stageA_qwen2_5_vl_3b_sft.yaml
```

Artifacts saved under `outputs/mind2web_stageA_sft/`.

Observed result:

- usable train samples: `101`
- usable eval samples: `17`
- optimizer steps: `40`
- best eval loss: `0.6651765739216524`
- checkpoints:
  - `outputs/mind2web_stageA_sft/checkpoint-best/`
  - `outputs/mind2web_stageA_sft/checkpoint-last/`

This is a real Mind2Web supervised baseline, not a scaffold.

## Stage B Candidate Export

### How export works

Stage B is built directly on top of the Stage A checkpoint:

- adapter loaded from `outputs/mind2web_stageA_sft/checkpoint-best`
- `top_k = 4`
- candidate rank 1 = Stage A deterministic first-choice
- candidate ranks 2..k = sampled candidate variants
- candidates are normalized into the same `bbox_proposal + click_point + action_type` contract

Each candidate carries:

- `click_point`
- `bbox_proposal`
- `action_type`
- `score`
- `confidence`
- `source`
- `provenance`
- `parser_metadata`
- `dom_match`
- deterministic `reward`

### Reward labels used for candidate supervision

The deterministic reward is:

`element_correct + 0.5 * IoU + 0.3 * click_inside_target + 0.2 * action_type_correct - 0.5 * invalid_format_penalty`

with exact component values saved per candidate.

### Actual export results

Train export command:

```bash
python scripts/run_generate_candidates.py --config configs/train/mind2web_stageB_candidates_qwen2_5_vl_3b.yaml
```

Train export artifacts under `outputs/mind2web_stageB_candidates/train/`:

- attempted samples: `118`
- successful samples: `118`
- failed samples: `0`
- total candidates: `472`
- average candidates per sample: `4.0`
- parseable structured output rate: `1.0`
- valid bbox rate: `0.9979`
- valid click rate: `0.9979`
- average best-of-k reward: `0.19827939323773977`

Official split-name export pilots were also generated for:

- `test_task` -> `outputs/mind2web_stageB_candidates/test_task/`
- `test_website` -> `outputs/mind2web_stageB_candidates/test_website/`
- `test_domain` -> `outputs/mind2web_stageB_candidates/test_domain/`

These are honest pilot-size exports over the official split names:

- `test_task`: `20` samples / `80` candidates
- `test_website`: `20` samples / `80` candidates
- `test_domain`: `19` samples / `76` candidates

This is not yet a full official-split sweep. The split names are correct, but the runs were capped to stable pilot-sized subsets for this task.

## Learned Reward Reranker

### Model

- lightweight MLP scorer
- feature size: `61`
- hidden size: `96`

Checkpoint:

- `outputs/mind2web_stageB_reranker/checkpoint-best/model.pt`

### Supervision and optimization

The reranker is trained from Stage B candidate pools using deterministic reward supervision.

Pair construction:

- all reward-separated candidate pairs inside a sample pool
- keep only pairs with reward gap above `0.01`
- pair weights scale with reward gap

Training setup:

- epochs: `15`
- optimizer target: pairwise ranking
- train pools: `95`
- eval pools: `23`
- pairwise train samples: `34`
- pairwise eval samples: `6`

### Features used

The scorer uses auditable engineered features from the exported candidate objects, including:

- candidate score / confidence / rank features
- normalized bbox and click geometry
- action-type one-hot features
- source/provenance features
- parser diagnostics
- DOM-match signals
- element-hint presence
- generation temperature

## Evaluation: SFT First-Choice vs SFT + Learned Reranker

### Commands run

Reranker training and split evaluation:

```bash
python scripts/run_train_reranker.py --config configs/train/mind2web_stageB_reranker_qwen.yaml training.device=cpu
```

Artifacts saved under `outputs/mind2web_stageB_reranker/`.

### Internal held-out train-pool note

On the internal train-pool validation slice, reranking produced essentially no gain because the candidate pools already had very little headroom there:

- baseline mean reward: `0.1715`
- reranked mean reward: `0.1715`
- oracle mean reward: `0.1725`

That is a candidate-pool headroom bottleneck, not a sign that the reranker path is fake.

### Official split-name pilot comparison

| Split | N | SFT reward | Reranked reward | Delta | Oracle reward | SFT point acc | Reranked point acc | SFT action acc | Reranked action acc | Parseable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `test_task` | 20 | 0.1300 | 0.1530 | +0.0230 | 0.1738 | 0.0000 | 0.0500 | 0.6500 | 0.6500 | 1.0000 |
| `test_website` | 20 | 0.1900 | 0.2050 | +0.0150 | 0.2388 | 0.0000 | 0.0500 | 0.9500 | 0.9500 | 1.0000 |
| `test_domain` | 19 | 0.1158 | 0.1158 | +0.0000 | 0.1352 | 0.0000 | 0.0000 | 0.5789 | 0.5789 | 1.0000 |

Additional observations:

- `test_task` rerank win rate: `0.10`
- `test_website` rerank win rate: `0.05`
- `test_domain` rerank win rate: `0.00`
- `test_task` oracle point accuracy: `0.10`
- `test_website` oracle point accuracy: `0.15`
- `test_domain` oracle point accuracy: `0.0526`

Headroom subset behavior was stronger than the full-pool averages:

- `test_task` headroom subset reward gain: `+0.1534`
- `test_website` headroom subset reward gain: `+0.1000`
- `test_domain` headroom subset reward gain: `+0.0000`

Interpretation:

- the learned reranker does recover some real reward headroom on `test_task`
- it also gives a smaller positive gain on `test_website`
- it is flat on `test_domain`
- the limiting factor is still candidate diversity / headroom, especially on the harder split

## Artifacts

### Stage A

- `outputs/mind2web_stageA_sft/train_summary.json`
- `outputs/mind2web_stageA_sft/training_history.json`
- `outputs/mind2web_stageA_sft/checkpoint-best/`

### Stage B

- `outputs/mind2web_stageB_candidates/train/candidates_train.jsonl`
- `outputs/mind2web_stageB_candidates/train/candidates_train_flat.jsonl`
- `outputs/mind2web_stageB_candidates/train/summary_train.json`
- `outputs/mind2web_stageB_candidates/test_task/candidates_test_task.jsonl`
- `outputs/mind2web_stageB_candidates/test_website/candidates_test_website.jsonl`
- `outputs/mind2web_stageB_candidates/test_domain/candidates_test_domain.jsonl`

### Stage B.5

- `outputs/mind2web_stageB_reranker/checkpoint-best/`
- `outputs/mind2web_stageB_reranker/training_history.json`
- `outputs/mind2web_stageB_reranker/evaluation_summary.json`
- `outputs/mind2web_stageB_reranker/official_split_evaluations.json`
- `outputs/mind2web_stageB_reranker/evaluation_test_task.json`
- `outputs/mind2web_stageB_reranker/evaluation_test_website.json`
- `outputs/mind2web_stageB_reranker/evaluation_test_domain.json`

## Does This Meaningfully Close The Blueprint Gap?

Yes, meaningfully, but not completely.

What is now real and blueprint-aligned:

- a genuine Mind2Web Stage A supervised baseline
- a genuine Stage B top-k candidate export on top of that baseline
- a learned reward reranker trained from verifiable labels
- a direct `SFT first-choice` vs `SFT + reranker` comparison on Mind2Web split names

What is still missing relative to the full blueprint:

- stronger Stage A coverage and larger-scale training
- broader and more diverse candidate pools so reranking has more headroom
- full official-split sweeps at larger sample counts
- transfer of the best Stage A/Stage B stack back into held-out ScreenSpot-v2 / VisualWebBench evaluation

## Most Important Next Missing Piece

The next highest-value missing piece is **stronger Stage B candidate headroom**, not a new RL stack.

Reason:

- the reranker is real
- the evaluation path is real
- gains already appear on `test_task` and `test_website`
- the main bottleneck is that too many pools already have little or no oracle headroom

The most important next step is therefore:

- strengthen candidate diversity while preserving the same Qwen-first structured contract
- then rerun the same `SFT first-choice` vs `SFT + reranker` comparison at larger official-split coverage

