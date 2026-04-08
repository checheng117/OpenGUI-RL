# ScreenSpot-v2 Failure Analysis And Prompt Refinement

## Scope

- Keep the path Qwen-first.
- No new training.
- No reranker / DPO / GRPO.
- No surrogate path.
- One focused coordinate-grounding refinement only.

## Inputs Inspected

- `docs/screenspot_v2_clean_heldout_eval.md`
- `outputs/screenspot_v2_eval_qwen2_5_vl_3b/evaluation_summary.json`
- `outputs/screenspot_v2_eval_qwen2_5_vl_3b/subgroup_metrics.json`
- `outputs/screenspot_v2_eval_qwen2_5_vl_3b/predictions.jsonl`
- `outputs/screenspot_v2_eval_qwen2_5_vl_3b/summary_table.md`
- `src/gui_grounding/models/qwen2_vl_grounding.py`
- `scripts/run_eval_screenspot_v2.py`

## Dominant Failure Taxonomy

Saved baseline failure-analysis artifacts:

- `outputs/screenspot_v2_eval_qwen2_5_vl_3b_analysis/failure_taxonomy.json`
- `outputs/screenspot_v2_eval_qwen2_5_vl_3b_analysis/failure_taxonomy.md`

Measured baseline findings from the full `1272` saved predictions:

- Parse failures: `1 / 1272`
- BBox-from-click fallbacks: `79`
- Malformed bbox payloads: `19`
- Parseable-but-spatially-wrong outputs: `1163 / 1272`
- `click` action overuse: `1093 / 1272 = 0.8593`

Key interpretation:

- The baseline problem is not primarily JSON formatting.
- The baseline problem is that predictions are usually spatially wrong even when parseable.

### Failure Mode 1: Coordinate-System Mismatch Dominates

The strongest measured pattern is a consistent coordinate shrinkage relative to ground truth:

- `mobile`
  - median click x ratio: `0.3761`
  - median click y ratio: `0.4094`
  - median bbox width ratio: `0.1894`
  - median bbox height ratio: `0.1712`
- `web`
  - median click x ratio: `0.3668`
  - median click y ratio: `0.3662`
  - median bbox width ratio: `0.2895`
  - median bbox height ratio: `0.2531`
- `desktop`
  - median click x ratio: `0.5836`
  - median click y ratio: `0.6578`
  - median bbox width ratio: `0.3631`
  - median bbox height ratio: `0.3478`

This matches the Qwen image-preprocessing resize path much better than the original screenshot coordinate system.

Counterfactual diagnostic on the saved baseline predictions:

- If the saved baseline predictions are interpreted in the Qwen resized-image frame and mapped back to original image size, overall point accuracy rises from `0.0849` to `0.6863`.
- Under the same diagnostic, overall IoU@0.5 rises from `0.0102` to `0.1321`.

That is too large to ignore. It identifies the dominant failure mode.

### Failure Mode 2: Web/Mobile Are Mostly Coordinate-Collapsed, Not Unparseable

Baseline full held-out pattern:

- `desktop` point accuracy: `0.3084`
- `mobile` point accuracy: `0.0060`
- `web` point accuracy: `0.0046`

The platform split is consistent with the resize mismatch:

- Desktop screenshots are often smaller, so the coordinate mismatch is less destructive.
- Web/mobile screenshots are larger or taller, so resized-frame coordinates land far from the original-frame target.

### Failure Mode 3: Action-Type Validity Is Already Strong

Baseline full held-out action/parse quality:

- action-type validity: `0.9992`
- parseable output rate: `0.9992`

This means prompt work focused on “better JSON only” was unlikely to be the main lever.

## Single Refinement Chosen

Implemented one focused refinement:

- Explicit `model_resized` coordinate-frame mode for Qwen prompting/parsing.

What it does:

1. The prompt now tells Qwen to emit bbox/click coordinates in the resized model-view image frame rather than the original screenshot frame.
2. The parser rescales those coordinates back into the original screenshot frame before scoring/export.

Why this one:

- It directly matches the dominant measured failure mode.
- It leaves the main candidate semantics unchanged:
  - `bbox_proposal`
  - `click_point`
  - `action_type`
- It is small, auditable, and does not redesign the pipeline.

Files changed for the refinement:

- `src/gui_grounding/models/qwen2_vl_grounding.py`
- `scripts/run_eval_screenspot_v2.py`

Supporting analysis tooling:

- `scripts/analyze_screenspot_v2_predictions.py`

## Exact Commands Run

Failure analysis:

```bash
python scripts/analyze_screenspot_v2_predictions.py
```

Balanced `180`-sample reevaluation with the refinement:

```bash
env HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  /home/cc/.conda/envs/gui-grounding-py310/bin/python \
  scripts/run_eval_screenspot_v2.py \
  --indices-json outputs/screenspot_v2_eval_qwen2_5_vl_3b_analysis/diagnostic_subset_balanced_indices.json \
  --coordinate-frame model_resized \
  --output-dir outputs/screenspot_v2_eval_qwen2_5_vl_3b_coordfix_balanced180 \
  --log-every 30
```

Broader balanced `360`-sample reevaluation:

```bash
python scripts/analyze_screenspot_v2_predictions.py \
  --output-dir outputs/screenspot_v2_eval_qwen2_5_vl_3b_analysis_balanced360 \
  --subset-per-platform 120
```

```bash
env HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  /home/cc/.conda/envs/gui-grounding-py310/bin/python \
  scripts/run_eval_screenspot_v2.py \
  --indices-json outputs/screenspot_v2_eval_qwen2_5_vl_3b_analysis_balanced360/diagnostic_subset_balanced_indices.json \
  --coordinate-frame model_resized \
  --output-dir outputs/screenspot_v2_eval_qwen2_5_vl_3b_coordfix_balanced360 \
  --log-every 60
```

## Reevaluation Results

### First Diagnostic Check: Balanced 180 Samples

Artifacts:

- `outputs/screenspot_v2_eval_qwen2_5_vl_3b_coordfix_balanced180/evaluation_summary.json`
- `outputs/screenspot_v2_eval_qwen2_5_vl_3b_coordfix_balanced180/subgroup_metrics.json`
- `outputs/screenspot_v2_eval_qwen2_5_vl_3b_coordfix_balanced180/comparison_vs_baseline.json`

Baseline on the same `180` samples:

- point accuracy: `0.1222`
- IoU@0.5: `0.0167`
- mean IoU: `0.0327`
- action-type validity: `1.0000`
- parseable output rate: `1.0000`

After refinement on the same `180` samples:

- point accuracy: `0.6833`
- IoU@0.5: `0.1778`
- mean IoU: `0.2259`
- action-type validity: `1.0000`
- parseable output rate: `1.0000`

Platform before/after on the same `180` samples:

- `desktop`
  - point accuracy: `0.3667 -> 0.6500`
  - IoU@0.5: `0.0500 -> 0.1500`
- `web`
  - point accuracy: `0.0000 -> 0.7000`
  - IoU@0.5: `0.0000 -> 0.3000`
- `mobile`
  - point accuracy: `0.0000 -> 0.7000`
  - IoU@0.5: `0.0000 -> 0.0833`

### Broader Confirmation: Balanced 360 Samples

Artifacts:

- `outputs/screenspot_v2_eval_qwen2_5_vl_3b_coordfix_balanced360/evaluation_summary.json`
- `outputs/screenspot_v2_eval_qwen2_5_vl_3b_coordfix_balanced360/subgroup_metrics.json`
- `outputs/screenspot_v2_eval_qwen2_5_vl_3b_coordfix_balanced360/comparison_vs_baseline.json`

Baseline on the same `360` samples:

- point accuracy: `0.1194`
- IoU@0.5: `0.0139`
- mean IoU: `0.0298`
- action-type validity: `0.9972`
- parseable output rate: `0.9972`

After refinement on the same `360` samples:

- point accuracy: `0.6722`
- IoU@0.5: `0.1722`
- mean IoU: `0.2281`
- action-type validity: `1.0000`
- parseable output rate: `1.0000`

Platform before/after on the same `360` samples:

- `desktop`
  - point accuracy: `0.3500 -> 0.6667`
  - IoU@0.5: `0.0417 -> 0.1167`
  - mean IoU: `0.0889 -> 0.1850`
- `web`
  - point accuracy: `0.0083 -> 0.6000`
  - IoU@0.5: `0.0000 -> 0.2833`
  - mean IoU: `0.0004 -> 0.3045`
- `mobile`
  - point accuracy: `0.0000 -> 0.7500`
  - IoU@0.5: `0.0000 -> 0.1167`
  - mean IoU: `0.0000 -> 0.1947`

## Decision

Did the single refinement help?

- Yes, clearly and materially on targeted held-out reevaluation.

What is now established:

- The dominant baseline failure was coordinate-frame mismatch, not general parse instability.
- A small coordinate-grounding refinement fixes a large share of the observed held-out error on balanced subsets.

What is not yet claimed:

- I did not rerun the full `1272` benchmark after the refinement in this step.
- So the new official full-held-out benchmark number is not claimed yet.

## Recommended Next Step

Run the refined `model_resized` coordinate-frame mode on the full ScreenSpot-v2 held-out benchmark, using the existing shard-safe evaluation path, so the repo can replace the weak baseline with a new official full held-out result.
