# Mind2Web Stage A Point-Native Supervision Redesign

## Scope

This task stayed inside Stage A only.

- kept the Qwen-first path
- kept Mind2Web as the Stage A supervision source
- did not modify Stage B candidate generation or the learned reranker
- did not work on ScreenSpot-v2
- did not do a scale-only rerun

## Starting Point

The previous localization-aware Stage A fix in `outputs/mind2web_stageA_sft_localization_fixed` had already removed the old geometry-collapse bug:

- normalized `[0, 1]` coordinates fixed the raw-pixel/frame mismatch
- point-first JSON ordering reduced dominant-template collapse
- parseability / valid action / valid bbox / valid click all stayed at `1.0000`

But grounding was still weak:

- internal-val `point_accuracy = 0.0375`
- internal-val `iou@0.5 = 0.0000`
- internal-val `mean_iou = 0.0061`

So the remaining Stage-A-only bottleneck was no longer repeated-template collapse. It was that click localization was still being learned through a coupled, free-form structured JSON target that remained too indirect for the model.

## Remaining Bottleneck Inspected Before Editing

The current Stage A path still had four point-learning problems:

1. `predicted_click_point` was still learned inside one coupled JSON object together with bbox/action/confidence tokens.
2. Coordinates were still emitted as free-form decimal text, which is brittle for a language model and spreads learning across many numeric token variants.
3. Bbox/action tokens still occupied a large share of the supervised response even though the real Stage A question is first “where should the click go?”
4. Validation still centered mainly on point hit and IoU, but lacked a finer click-distance signal for checkpoint selection once outputs were no longer collapsing.

Evidence from the saved localization-fixed predictions supported that diagnosis:

- parseability was perfect, so the main problem was not syntax
- collapse score was already low (`0.0125`)
- point predictions were still inaccurate despite diverse outputs
- internal-val mean normalized click L1, recomputed from saved predictions, was `0.1485`

## Point-Native Redesign Applied

I implemented one focused Stage-A-only redesign:

### 1. Point-primary mixed supervision

Instead of supervising one coupled structured response per sample, Stage A now trains on two serialization types:

- primary point example for **every** sample
- secondary bbox-support example for a deterministic subset

The primary example teaches only the click point plus action type:

```json
{"point_bin":[x_bin,y_bin],"action_type":"click|type|select|hover"}
```

The secondary example teaches only the supporting bbox given a fixed click:

```json
{"bbox_bin":[x1_bin,y1_bin,x2_bin,y2_bin]}
```

That makes click-point prediction the true first-class learned target and keeps bbox explicitly secondary.

### 2. Quantized coordinate serialization

I replaced free-form decimal coordinates with integer bins:

- `point_bin`
- `bbox_bin`
- integer range `[0, 999]`

These bins are normalized relative to screenshot width/height and mapped back after parsing.

The intent was to make point prediction easier for the language model than emitting many 4-decimal float variants.

### 3. Matching decoupled eval path

The eval model was updated to understand the same bin-based serialization:

- primary point-native pass predicts `point_bin`
- secondary support pass predicts `bbox_bin`
- best-checkpoint selection became more point-aware with:
  - `point_accuracy`
  - `-mean_normalized_click_l1`
  - `iou@0.5`
  - `mean_iou`
  - `action_type_accuracy`
  - `parseable_output_rate`
  - `-eval_loss`

## Code Changes

- `src/gui_grounding/training/trainer_sft_qwen.py`
  - added point-primary mixed Stage A supervision mode
  - added quantized `point_bin` / `bbox_bin` target serialization
  - added deterministic bbox-support subset generation
- `src/gui_grounding/models/qwen2_vl_grounding.py`
  - added quantized point/bbox parsing
  - added quantized point-native prompts and bbox-only support prompt
- `src/gui_grounding/evaluation/metrics.py`
  - added `mean_normalized_click_l1`
- `scripts/run_train_sft.py`
  - passed through new Stage A serialization config
  - added click-distance metric to eval summaries and checkpoint selection
- `configs/train/mind2web_stageA_qwen2_5_vl_3b_sft_point_native_redesign.yaml`
  - new Stage A point-native run config

## Rerun

Run directory:

- `outputs/mind2web_stageA_sft_point_native_redesign`

Actual run:

- original train samples: `560`
- original eval samples: `80`
- train supervision examples: `896`
  - point-primary: `560`
  - bbox-support: `336`
- eval supervision examples: `128`
  - point-primary: `80`
  - bbox-support: `48`
- optimizer steps: `224`
- epochs: `1`
- best val loss: `0.6272`
- selected checkpoint: `checkpoint-best`

This kept the optimizer-step budget equal to the previous localization-fixed run (`224`) while redirecting the supervised response budget toward point-native serialization.

## Before / After

### Internal validation

Note: `best val loss` is not perfectly apples-to-apples across the two runs because the supervised response format changed from one coupled structured target to a mixed point-primary / bbox-support target. The grounding metrics below are the decisive comparison.

| Metric | Localization-fixed Stage A | Point-native redesign |
| --- | ---: | ---: |
| best val loss | `0.4695` | `0.6272` |
| point accuracy | `0.0375` | `0.0125` |
| IoU@0.5 | `0.0000` | `0.0000` |
| mean IoU | `0.0061` | `0.0037` |
| action accuracy | `0.9000` | `0.8750` |
| parseable output rate | `1.0000` | `1.0000` |
| valid bbox rate | `1.0000` | `1.0000` |
| valid click rate | `1.0000` | `1.0000` |
| valid action rate | `1.0000` | `1.0000` |
| mean normalized click L1 | `0.1485` | `0.1685` |

### Collapse diagnostics on internal validation

| Diagnostic | Localization-fixed Stage A | Point-native redesign |
| --- | ---: | ---: |
| dominant bbox fraction | `0.0125` | `0.0250` |
| dominant click fraction | `0.0125` | `0.0250` |
| unique bbox count | `80` | `79` |
| unique click count | `80` | `79` |
| collapse score | `0.0125` | `0.0250` |

Interpretation:

- the redesign did **not** reintroduce hard collapse
- outputs stayed parseable and mostly diverse
- but click localization quality got worse anyway

### Official cached subsets

| Split | Point Acc Before | Point Acc After | Mean IoU Before | Mean IoU After | Action Acc Before | Action Acc After |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `test_task` | `0.0000` | `0.0000` | `0.0057` | `0.0016` | `0.7000` | `0.7000` |
| `test_website` | `0.0000` | `0.0500` | `0.0031` | `0.0022` | `0.9500` | `0.8000` |
| `test_domain` | `0.0526` | `0.0000` | `0.0198` | `0.0001` | `0.7368` | `0.5789` |

Normalized click distance also worsened on the most important splits:

- internal val: `0.1485 -> 0.1685`
- `test_task`: `0.1501 -> 0.1709`
- `test_domain`: `0.1393 -> 0.2360`

## Honest Interpretation

### What improved

The redesign did achieve the intended serialization change:

- click became the real primary learned target
- bbox became an explicitly secondary support target
- free-form float coordinates were replaced with discrete bins
- validation became more localization-aware

### What did not improve

Grounding quality did **not** improve.

On the main internal validation readout, the redesign regressed:

- point accuracy: `0.0375 -> 0.0125`
- mean IoU: `0.0061 -> 0.0037`
- normalized click distance: `0.1485 -> 0.1685` (worse)

Official cached subsets also did not show a consistent win:

- `test_website` moved slightly off zero on point accuracy
- `test_domain` fell back to zero
- IoU stayed extremely weak or worsened

## Why This Likely Happened

The result suggests that Stage A’s remaining weakness is **not solved by making the text format more point-native alone**.

Most likely:

1. Mind2Web still provides only bbox supervision, so the click target is a bbox-center proxy rather than a real human click distribution.
2. Fully decoupling point prediction from bbox support removed too much of the geometric structure that the model was still using.
3. Quantized point bins made the format easier to emit, but they did not make the supervision itself more informative.

In short:

- collapse was already fixed before this task
- this redesign changed serialization successfully
- but it did not make the supervision materially stronger

## Verdict

### Did point grounding improve materially?

No.

It regressed on the main internal Stage A grounding metrics and did not produce a consistent held-out improvement.

### Is Stage A now materially closer to a strong supervised grounding baseline?

No.

This run is **not** materially closer to the proposal’s intended strong supervised Stage A baseline.

If anything, it is slightly farther away than the previous localization-fixed Stage A run.

### Does this count as a credible supervised grounding baseline?

No.

The best current Stage A result remains the previous localization-fixed run, not this point-native redesign.

## Most Important Remaining Stage-A Gap

The next most important Stage-A-related gap is:

**stronger point supervision quality, not just stronger point serialization.**

This result suggests the next Stage A attempt should keep click primary while restoring tighter point-to-box coupling instead of fully decoupling them, for example through a point-primary joint target or explicit point-token weighting, because Mind2Web’s point signal is still only a bbox-center proxy.

## Saved Artifacts

- `outputs/mind2web_stageA_sft_point_native_redesign/checkpoint-best`
- `outputs/mind2web_stageA_sft_point_native_redesign/checkpoint-latest`
- `outputs/mind2web_stageA_sft_point_native_redesign/train_summary.json`
- `outputs/mind2web_stageA_sft_point_native_redesign/eval_summary.json`
- `outputs/mind2web_stageA_sft_point_native_redesign/collapse_diagnostics.json`
