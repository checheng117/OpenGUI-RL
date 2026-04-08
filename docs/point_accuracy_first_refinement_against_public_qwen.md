# Point-Accuracy-First Refinement Against Public Qwen

## Scope

- Goal: improve `point_accuracy` against the reproduced plain public `Qwen/Qwen2.5-VL-3B-Instruct` baseline.
- Constraints respected:
  - Qwen-first path only
  - no new training
  - no reranker / DPO / GRPO work
  - no pipeline redesign
  - keep structured output advantages where possible

## Inputs Inspected Before Editing

- `docs/public_baseline_reproduction_and_same_protocol_comparison.md`
- `configs/eval/screenspot_v2_public_qwen2_5_vl_3b_point_baseline.yaml`
- `configs/eval/screenspot_v2_qwen2_5_vl_3b_model_resized_full.yaml`
- `src/gui_grounding/models/qwen2_vl_public_point_baseline.py`
- `src/gui_grounding/models/qwen2_vl_grounding.py`
- `scripts/run_eval_screenspot_v2.py`
- `outputs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized/predictions.jsonl`
- `outputs/screenspot_v2_public_qwen2_5_vl_3b_point_baseline/predictions.jsonl`

## Smallest Likely Bottleneck

The sample-level comparison pointed to a concrete failure mode in the structured path:

- plain public Qwen is point-native and only derives a tiny bbox after the fact
- our structured path jointly predicts bbox and click
- many structured predictions place `predicted_click_point` on the bbox corner or top-left edge instead of the actionable interior

Measured on the saved current structured run:

- `270 / 1272 = 21.23%` of predictions had click exactly at `bbox.x1, bbox.y1`
- those exact-top-left cases only hit at `0.5444`
- clicks near the bbox center band hit at `0.7843`

This grounded the hypothesis that structured bbox rigidity was still dragging down point hit rate.

## Minimal Refinements Implemented

Two small changes only:

1. Point-first prompt nudge in `src/gui_grounding/models/qwen2_vl_grounding.py`
   - explicitly prioritizes the click point
   - tells Qwen the click must be inside the actionable interior, preferably near the center
   - tells Qwen not to place the click on a bbox corner/border unless the target is tiny
   - tells Qwen to choose the click first and make the bbox enclose it

2. Lightweight point decode correction in `src/gui_grounding/models/qwen2_vl_grounding.py`
   - if the parsed click falls in the top-left fringe of its own predicted bbox, move it inward to a safer interior point
   - used `edge_click_interior_threshold = 0.2`
   - used `edge_click_interior_position = 0.45`
   - bbox and action schema remain unchanged

Config used:

- `configs/eval/screenspot_v2_qwen2_5_vl_3b_point_accuracy_first.yaml`

Evaluation runner wiring:

- `scripts/run_eval_screenspot_v2.py`

Unit test added:

- `tests/test_qwen_point_refinement.py`

## Same-Protocol Evaluation

### Reproduced Public Baseline

- artifact root: `outputs/screenspot_v2_public_qwen2_5_vl_3b_point_baseline`
- evaluated samples: `1272`
- point accuracy: `0.7563`
- desktop point accuracy: `0.7275`
- web point accuracy: `0.7346`
- mobile point accuracy: `0.7944`
- IoU@0.5: `0.0519`
- mean IoU: `0.1327`
- action-type validity: `0.2980`
- parseable output rate: `0.9937`

### Previous Structured Method

- artifact root: `outputs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized`
- evaluated samples: `1272`
- point accuracy: `0.7099`
- desktop point accuracy: `0.6976`
- web point accuracy: `0.6796`
- mobile point accuracy: `0.7445`
- IoU@0.5: `0.1682`
- mean IoU: `0.2404`
- action-type validity: `1.0000`
- parseable output rate: `1.0000`

### Updated Point-Accuracy-First Method

- artifact root: `outputs/screenspot_v2_eval_qwen2_5_vl_3b_point_accuracy_first`
- evaluated samples: `1272`
- point accuracy: `0.7296`
- desktop point accuracy: `0.7305`
- web point accuracy: `0.6751`
- mobile point accuracy: `0.7764`
- IoU@0.5: `0.1635`
- mean IoU: `0.2365`
- action-type validity: `1.0000`
- parseable output rate: `1.0000`

## Direct Comparison

### Updated Method vs Previous Structured Method

- overall point accuracy: `0.7099 -> 0.7296` (`+0.0197`)
- desktop point accuracy: `0.6976 -> 0.7305` (`+0.0329`)
- web point accuracy: `0.6796 -> 0.6751` (`-0.0046`)
- mobile point accuracy: `0.7445 -> 0.7764` (`+0.0319`)
- IoU@0.5: `0.1682 -> 0.1635` (`-0.0047`)
- mean IoU: `0.2404 -> 0.2365` (`-0.0039`)
- action-type validity stayed `1.0000`
- parseable output rate stayed `1.0000`

### Updated Method vs Reproduced Public Baseline

- overall point accuracy: `0.7563 -> 0.7296` (`-0.0267`)
- desktop point accuracy: `0.7275 -> 0.7305` (`+0.0030`)
- web point accuracy: `0.7346 -> 0.6751` (`-0.0595`)
- mobile point accuracy: `0.7944 -> 0.7764` (`-0.0180`)

Gap accounting:

- previous gap to public baseline: `-0.0464`
- updated gap to public baseline: `-0.0267`
- gap closed: `+0.0197`

Verdict: the gap **partially closed**, but did **not** fully close.

## Structured-Output Stability

The update kept the main structured-output strengths:

- strict action-type validity remained `1.0000`
- parseable output rate remained `1.0000`
- bbox quality stayed well above the public point baseline:
  - updated IoU@0.5 `0.1635` vs public baseline `0.0519`
  - updated mean IoU `0.2365` vs public baseline `0.1327`

The new click refinement was actually applied on:

- `316 / 1272 = 24.84%` of full-run predictions

## Remaining Likely Bottleneck

The updated method now slightly exceeds the public baseline on `desktop`, and it narrows the `mobile` gap, but `web` remains the main bottleneck.

Most likely remaining issue:

- even after the point-first nudge, the structured path is still more bbox-coupled than the plain point-native public Qwen behavior
- that coupling still hurts web click placement more than desktop/mobile, especially when the best click is a free interior point that should not be inferred from a region proposal

## Saved Artifacts

- updated full predictions:
  - `outputs/screenspot_v2_eval_qwen2_5_vl_3b_point_accuracy_first/predictions.jsonl`
- updated summary:
  - `outputs/screenspot_v2_eval_qwen2_5_vl_3b_point_accuracy_first/evaluation_summary.json`
- updated subgroup metrics:
  - `outputs/screenspot_v2_eval_qwen2_5_vl_3b_point_accuracy_first/subgroup_metrics.json`
- comparison vs previous structured method:
  - `outputs/screenspot_v2_eval_qwen2_5_vl_3b_point_accuracy_first/comparison_vs_previous_structured.json`
  - `outputs/screenspot_v2_eval_qwen2_5_vl_3b_point_accuracy_first/comparison_vs_previous_structured.md`
- comparison vs reproduced public baseline:
  - `outputs/screenspot_v2_eval_qwen2_5_vl_3b_point_accuracy_first/comparison_vs_public_baseline.json`
  - `outputs/screenspot_v2_eval_qwen2_5_vl_3b_point_accuracy_first/comparison_vs_public_baseline.md`

## Final Answer

- Did the refinement improve point accuracy: **yes**
- By how much overall: `+0.0197`
- Did it narrow the gap to the reproduced public baseline: **yes**
- Did it close the gap: **no**
- Are we above or below the reproduced public baseline now: **still below**
