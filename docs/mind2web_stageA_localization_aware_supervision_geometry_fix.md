# Mind2Web Stage A Localization-Aware Supervision Geometry Fix

## Scope

This work stayed inside Stage A only.

- kept the Qwen-first path
- kept Mind2Web as the Stage A supervision source
- did not modify Stage B candidate expansion or the reranker
- did not work on ScreenSpot-v2
- did not do a scale-only rerun

## Diagnosis

The stronger Stage A baseline in `outputs/mind2web_stageA_sft_stronger` learned the JSON format and action typing, but it did not learn real grounding.

Observed collapse on internal validation:

- `point_accuracy = 0.0000`
- `iou@0.5 = 0.0000`
- `mean_iou = 0.0000`
- parseable / valid action / valid bbox / valid click all stayed at `1.0000`

The dominant failure mode was geometry collapse into repeated generic coordinate templates rather than sample-grounded localization.

Evidence from the saved collapsed run:

- dominant bbox template: `[100, 100, 120, 120]` on `40 / 80 = 50.0%`
- dominant click template: `[110, 110]` on `40 / 80 = 50.0%`
- clicks within the same `20px` band around that dominant point: `55.0%`
- only `21` unique bbox outputs across `80` samples
- coarse normalized click bucket `(0.1, 0.0)` held `51 / 80` predictions

The main cause was not just a bad parser setting. It was the supervision itself.

1. Coordinate-frame / numeric-target mismatch

- Qwen encodes a resized model-view image, but Stage A supervision asked it to emit raw original-image absolute pixels.
- On the Stage A train set, the mean resize ratio was:
  - `scale_x = 0.3079`
  - `scale_y = 0.3139`
- The resize ratio ranged from about `0.09` to `0.61`, so the model had to infer a different absolute-pixel scale per sample from a resized visual frame.

2. Localization tokens were too easy to collapse

- Absolute pixel targets varied widely across screenshots.
- `predicted_click_point` was not primary in the serialized target.
- `predicted_element_id` was supervised even though it was not helping grounding and added noisy digit loss.

3. The failure was not only eval-side frame decoding

I reinterpreted the saved collapsed baseline outputs as if they were emitted in the model-resized frame.

- `point_accuracy: 0.0000 -> 0.0250`
- `mean_iou: 0.0000 -> 0.0074`
- `iou@0.5` stayed `0.0000`

That probe confirms the old run was not merely decoded in the wrong frame. Learning itself had collapsed.

## Fixes Applied

I kept the fix narrow and Stage-A-local.

1. Normalized localization supervision

- Stage A training targets now emit normalized `[0, 1]` coordinates instead of raw absolute pixels.
- Evaluation prompts were aligned to the same normalized coordinate format.
- This removes the biggest numeric burden from the model and makes targets compatible with the resized visual frame.

2. Point-first target serialization

- Stage A training now serializes `predicted_click_point` before `predicted_bbox`.
- The prompt explicitly says to choose the click point first and then make the bbox enclose it.
- `predicted_element_id` supervision was disabled and serialized as `null` to reduce noisy token pressure on localization.

3. Localization-aware validation and checkpoint selection

- Added collapse diagnostics to Stage A eval output.
- Added internal-val checkpoint selection using:
  - `point_accuracy`
  - `iou@0.5`
  - `mean_iou`
  - `action_type_accuracy`
  - `parseable_output_rate`
  - `-eval_loss`

## Code Changes

- `src/gui_grounding/training/trainer_sft_qwen.py`
  - normalized / point-first Stage A target formatting
  - optional element-id de-emphasis
- `src/gui_grounding/models/qwen2_vl_grounding.py`
  - normalized-coordinate prompting support
  - prompt/schema alignment with point-first Stage A output
- `src/gui_grounding/evaluation/collapse_diagnostics.py`
  - repeated-template and tiny-box collapse diagnostics
- `scripts/run_train_sft.py`
  - localization-aware eval summaries
  - collapse diagnostics export
  - localization-aware checkpoint selection
- `configs/train/mind2web_stageA_qwen2_5_vl_3b_sft_localization_fixed.yaml`
  - fixed Stage A run config

## Rerun

Run directory:

- `outputs/mind2web_stageA_sft_localization_fixed`

Actual run:

- train samples: `560`
- eval samples: `80`
- optimizer steps: `224`
- epochs: `2`
- best validation loss: `0.4695`
- selected checkpoint: `checkpoint-best`

## Before / After

### Internal validation

| Metric | Collapsed baseline | Localization-fixed run |
|---|---:|---:|
| best val loss | 0.5900 | 0.4695 |
| point accuracy | 0.0000 | 0.0375 |
| IoU@0.5 | 0.0000 | 0.0000 |
| mean IoU | 0.0000 | 0.0061 |
| action accuracy | 0.8750 | 0.9000 |
| parseable output rate | 1.0000 | 1.0000 |
| valid bbox rate | 1.0000 | 1.0000 |
| valid click rate | 1.0000 | 1.0000 |
| valid action rate | 1.0000 | 1.0000 |

### Collapse diagnostics on internal validation

| Diagnostic | Collapsed baseline | Localization-fixed run |
|---|---:|---:|
| dominant bbox fraction | 0.5000 | 0.0125 |
| dominant click fraction | 0.5000 | 0.0125 |
| unique bbox count | 21 | 80 |
| unique click count | 21 | 80 |
| collapse score | 0.5000 | 0.0125 |

Coarse normalized click clustering also relaxed materially:

- old top click bucket: `(0.1, 0.0)` with `51 / 80`
- new top click bucket: `(0.4, 0.0)` with `13 / 80`

### Official cached subsets

| Split | Point Acc Before | Point Acc After | Mean IoU Before | Mean IoU After | Action Acc Before | Action Acc After |
|---|---:|---:|---:|---:|---:|---:|
| `test_task` | 0.0000 | 0.0000 | 0.0000 | 0.0057 | 0.7000 | 0.7000 |
| `test_website` | 0.0000 | 0.0000 | 0.0000 | 0.0031 | 0.9500 | 0.9500 |
| `test_domain` | 0.0000 | 0.0526 | 0.0000 | 0.0198 | 0.5789 | 0.7368 |

## Verdict

### What got fixed

Yes: the geometry-collapse failure mode itself was fixed.

- outputs no longer collapse into a repeated generic bbox/click template
- predictions are now spatially diverse and normalized-coordinate aware
- point accuracy and mean IoU moved off zero

### What did not get fixed

No: Stage A is still not a credible supervised grounding baseline yet.

Why:

- internal-val `point_accuracy` is only `0.0375` (`3 / 80`)
- internal-val `iou@0.5` is still `0.0000`
- internal-val `mean_iou` is still only `0.0061`

So the run now shows weak non-zero grounding behavior, but not strong enough grounding quality.

## Proposal Alignment

This materially closes the specific Stage A collapse bug:

- Stage A is no longer just learning output syntax plus generic boxes.

But it does **not** materially close the full proposal-level Stage A gap:

- the model is still far below what should count as a strong supervised grounding baseline for later reward-based improvements.

## Most Important Remaining Stage-A Gap

The next most important Stage-A-only gap is stronger point-centric supervision beyond plain text-token SFT over bbox JSON.

Concretely:

- Stage A still learns localization through text generation only
- Mind2Web only provides bbox supervision, so click supervision is still a bbox-center proxy
- the model likely needs a more point-native Stage A objective / serialization where the click point is the true primary target and bbox is secondary

In short:

- collapse is fixed
- non-zero grounding appears
- Stage A still needs stronger point-centric supervision before it qualifies as the proposal’s intended supervised grounding baseline

## Saved Artifacts

- `outputs/mind2web_stageA_sft_localization_fixed/checkpoint-best`
- `outputs/mind2web_stageA_sft_localization_fixed/checkpoint-latest`
- `outputs/mind2web_stageA_sft_localization_fixed/train_summary.json`
- `outputs/mind2web_stageA_sft_localization_fixed/eval_summary.json`
- `outputs/mind2web_stageA_sft_localization_fixed/collapse_diagnostics.json`
