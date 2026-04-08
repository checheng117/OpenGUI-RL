# Mind2Web Stage A Stronger Supervised Baseline

## Scope

This task was scoped only to Stage A:

- keep the Qwen-first Mind2Web LoRA SFT path
- strengthen it from pilot scale into a more credible supervised baseline
- preserve downstream Stage B compatibility
- avoid redesigning candidate generation, reranking, DPO, or GRPO

## What Was Preserved

The following parts of the working Stage A path were kept intact:

- base model family: `Qwen/Qwen2.5-VL-3B-Instruct`
- Stage A training mode: LoRA SFT
- structured output contract used by the Qwen path:
  - `action_type`
  - `predicted_bbox`
  - `predicted_click_point`
  - `predicted_element_id`
  - `confidence`
- downstream compatibility assumption:
  - Stage B still consumes the same Qwen structured output and normalizes it into canonical `bbox_proposal + click_point + action_type`

No Stage B or reranker logic was changed in this task.

## Why The Old Stage A Was Still Pilot-Like

The previous real Qwen Stage A run under `outputs/mind2web_stageA_sft/` was important as proof of pipeline, but it was still too small to serve as the main supervised baseline:

- train samples: `101`
- eval samples: `17`
- optimizer steps: `40`
- validation setup: random holdout from the same small slice
- reporting: loss-focused, no post-train generative Stage A readout

That was enough to show the path worked, but not enough to call it the main baseline from the original proposal.

## What Changed

### 1. Train data loading now scales from local cached Mind2Web shards

Updated file:

- `src/gui_grounding/data/mind2web_dataset.py`

Change:

- the dataset loader now prefers local cached Mind2Web parquet shards for `train`
- Hugging Face streaming remains the fallback path

Why this matters:

- the stronger Stage A run no longer depends on fragile network streaming for the main train split
- larger Mind2Web Stage A runs are now reproducible on this machine from local cache

Observed local cache used for this run:

- `8` cached train parquet shards
- `2304` raw rows available in those local shards
- `2178` valid action+bbox rows available in those local shards

### 2. Subset selection and validation split are now reproducible and more principled

Updated file:

- `scripts/run_train_sft.py`

Changes:

- added deterministic website-aware subset selection for Stage A train data
- added deterministic website-aware train/val splitting
- added split statistics to `train_summary.json`

For the stronger run:

- loaded `960` local train rows
- kept `640` valid supervised rows after deterministic website-aware subset selection
- split to:
  - `560` train
  - `80` internal validation

Coverage of the stronger internal split:

- train websites: `43`
- eval websites: `43`
- train domains: `3`
- eval domains: `3`

### 3. Stage A training is now materially larger than the pilot

Updated files:

- `configs/train/mind2web_stageA_qwen2_5_vl_3b_sft_stronger.yaml`
- `src/gui_grounding/training/trainer_sft_qwen.py`

Changes:

- stronger dedicated config under a new output root:
  - `outputs/mind2web_stageA_sft_stronger/`
- deterministic dataloader seed
- `checkpoint-best`
- `checkpoint-latest`
- richer `train_summary.json`

Actual stronger config used:

- base model: `Qwen/Qwen2.5-VL-3B-Instruct`
- LoRA targets:
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`
  - `gate_proj`
  - `up_proj`
  - `down_proj`
- dtype: `bfloat16`
- attention backend: `sdpa`
- train source: Mind2Web `train` local cached shards
- loaded rows: `960`
- selected supervised rows: `640`
- split:
  - train `560`
  - val `80`
- epochs: `2`
- optimizer steps: `224`
- batch size: `1`
- gradient accumulation: `4`
- effective batch size: `4`
- learning rate: `1e-4`
- warmup steps: `11`

### 4. Stage A now has a real post-train generative evaluation readout

Updated file:

- `scripts/run_train_sft.py`

Added outputs:

- `outputs/mind2web_stageA_sft_stronger/eval_summary.json`
- `outputs/mind2web_stageA_sft_stronger/eval_predictions_internal_val.jsonl`
- `outputs/mind2web_stageA_sft_stronger/eval_predictions_test_task.jsonl`
- `outputs/mind2web_stageA_sft_stronger/eval_predictions_test_website.jsonl`
- `outputs/mind2web_stageA_sft_stronger/eval_predictions_test_domain.jsonl`

The post-train eval reports:

- parseable output rate
- valid bbox rate
- valid click rate
- valid action rate
- element accuracy
- point accuracy
- mean IoU
- IoU@0.5
- IoU@0.75
- action-type accuracy

## Pilot vs Stronger Run

| Item | Pilot Stage A | Stronger Stage A |
|---|---:|---:|
| Selected supervised rows | `118` total | `640` total |
| Train samples | `101` | `560` |
| Eval samples | `17` | `80` |
| Optimizer steps | `40` | `224` |
| Epoch budget | `1` | `2` |
| Effective batch size | `2` | `4` |
| Best validation loss | `0.6652` | `0.5900` |
| Checkpoints | best + last | best + latest |
| Post-train Stage A metrics | none | yes |

Relative to the old pilot:

- train sample count increased by about `5.5x`
- eval sample count increased by about `4.7x`
- optimizer steps increased by about `5.6x`
- best validation loss improved by about `0.0752`

## Actual Stronger Run Results

Artifacts:

- `outputs/mind2web_stageA_sft_stronger/checkpoint-best/`
- `outputs/mind2web_stageA_sft_stronger/checkpoint-latest/`
- `outputs/mind2web_stageA_sft_stronger/train_summary.json`
- `outputs/mind2web_stageA_sft_stronger/eval_summary.json`
- `outputs/mind2web_stageA_sft_stronger/training_history.json`

### Internal validation

From `outputs/mind2web_stageA_sft_stronger/eval_summary.json`:

- samples: `80`
- parseable output rate: `1.0000`
- valid bbox rate: `1.0000`
- valid click rate: `1.0000`
- valid action rate: `1.0000`
- action-type accuracy: `0.8750`
- point accuracy: `0.0000`
- mean IoU: `0.0000`
- IoU@0.5: `0.0000`
- IoU@0.75: `0.0000`

### Cached official split-name subsets

These are the existing cached official split-name subsets already present in the repo, not full official sweeps:

- `test_task`: `20`
- `test_website`: `20`
- `test_domain`: `19`

Results:

| Split | N | Parseable | Point Acc | Mean IoU | Action Acc |
|---|---:|---:|---:|---:|---:|
| `test_task` | `20` | `1.0000` | `0.0000` | `0.0000` | `0.7000` |
| `test_website` | `20` | `1.0000` | `0.0000` | `0.0000` | `0.9500` |
| `test_domain` | `19` | `1.0000` | `0.0000` | `0.0000` | `0.5789` |

## Interpretation

### What is now clearly better

The upgraded Stage A is materially stronger than the old pilot in engineering terms:

- much larger train/eval scale
- deterministic subset and split handling
- clean stronger config
- `checkpoint-best` and `checkpoint-latest`
- real train/eval summaries
- auditable per-sample prediction exports
- cheap official split-name readout on cached subsets

This means Stage A is no longer a smoke run or proof-of-pipeline only.

### What is still not good enough

The run is stronger, but it is still not strong enough empirically to claim that the Stage A gap in the original proposal is fully closed.

Reason:

- the model learned to emit valid structured JSON reliably
- action prediction improved to a useful level
- but grounding quality is still effectively collapsed on the evaluated subsets:
  - internal val point accuracy: `0.0000`
  - internal val mean IoU: `0.0000`
  - official cached subset point accuracy: `0.0000` on all three split names

Observed failure mode in the saved predictions:

- outputs are parseable
- outputs often reuse generic low-information boxes such as `[100,100,120,120]`
- this indicates format learning and partial action learning without real element localization

So:

- this run is a **stronger and credible Stage A engineering baseline**
- it is **not yet a strong supervised grounding baseline** in the sense originally intended by the proposal

## Does This Count As The Main Stage A Baseline?

Answer: **partially**.

It now counts as the main Stage A baseline in terms of:

- reproducibility
- actual training scale
- checkpointing
- logging
- auditable evaluation artifacts

It does **not** yet count as a strong final supervised grounding baseline in terms of actual GUI grounding quality.

## Most Important Remaining Stage-A Gap

The next most important Stage-A-specific gap is:

**break the coordinate-template collapse so Stage A learns real grounding rather than mostly structured action formatting.**

That should stay a Stage A task and happen before spending more effort on later reward-based improvement, because otherwise Stage B is improving on top of a base model whose geometry is still weak.

Concretely, the next Stage-A-focused step should target one or more of:

- tighter geometry supervision / target normalization
- prompt and decoding alignment specifically for point/box grounding
- stronger localization-oriented validation during training, not loss-only checkpointing
- a follow-up stronger Stage A rerun once the geometry collapse is addressed
