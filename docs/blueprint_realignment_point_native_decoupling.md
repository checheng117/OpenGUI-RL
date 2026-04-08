# Blueprint Realignment: Point-Native Decoupling

## Goal

Return the repo toward the original blueprint:

- make `click_point` the first-class prediction target
- keep `bbox_proposal` and `action_type` in the structured contract
- stay Qwen-first
- do one clean method change under the same ScreenSpot-v2 protocol
- answer directly whether this closes or beats the reproduced public Qwen baseline

This task explicitly did **not** introduce new training, reranking, DPO, GRPO, or another prompt-only refinement round.

## What Was Inspected First

- Project proposal:
  - `docs/gui_grounding_project_proposal.docx`
- Same-protocol public-baseline comparison:
  - `docs/public_baseline_reproduction_and_same_protocol_comparison.md`
- Best prior point-first refinement:
  - `docs/point_accuracy_first_refinement_against_public_qwen.md`
- Later prompt-only web/mobile refinement:
  - `docs/web_mobile_same_protocol_refinement_round.md`
- Current Qwen decoding path:
  - `src/gui_grounding/models/qwen2_vl_grounding.py`
  - `src/gui_grounding/models/qwen2_vl_public_point_baseline.py`
- Current ScreenSpot-v2 evaluation runner:
  - `scripts/run_eval_screenspot_v2.py`

## Where The Old Path Was Still Bbox-Coupled

The remaining coupling was structural, not just prompt wording:

1. The structured adapter asked Qwen for `predicted_bbox` and `predicted_click_point` in the same response.
2. The parser still treated bbox and click as a joint object:
   - missing click fell back to `bbox.center`
   - missing bbox fell back to a small click-derived bbox
3. The best prior point-first path still depended on the same one-shot structured JSON and then applied a small post-parse click correction.

In other words, `click_point` was nudged, but not actually promoted to a separate primary prediction object.

That was misaligned with the proposal wording:

- action `a = (e, b, t)`, where `b` is a **bounding box or click point**
- Stage B is candidate generation + reranking over action objects

The cleaner blueprint-aligned interpretation is:

- predict the best click point first
- attach supporting bbox/action second
- keep provenance so later reward-based candidate generation can score the object cleanly

## One Clean Method Change

Implemented one new decode path in `src/gui_grounding/models/qwen2_vl_grounding.py`:

### 1. Primary point-native pass

- Uses a plain Qwen-style `point_2d` JSON prompt, close to the reproduced public baseline behavior.
- This pass predicts the primary click point first.
- No bbox is requested in this pass.

### 2. Secondary structured support pass

- Runs a second prompt after the click is chosen.
- The prompt is anchored on the fixed primary click point.
- It predicts the supporting `predicted_bbox` and `action_type`.
- The secondary pass is not allowed to rewrite the primary click.

### 3. Safe structured fallback

- If the point-native pass fails to produce a usable point, the code falls back to the old structured single-pass path.
- This keeps the same output contract robust.

### 4. Blueprint-aligned metadata

The new candidate schema recorded by the evaluator is:

- `version: bbox_click_action_v2_point_primary`
- `primary_fields: ["bbox_proposal", "click_point", "action_type"]`
- `primary_prediction_field: "click_point"`

Added secondary metadata for later Stage-B-style scoring:

- `click_point_provenance`
- `bbox_provenance`
- `action_type_provenance`
- `point_pass_confidence`
- `structure_pass_confidence`

### 5. Eval/config/test support

Added:

- `configs/eval/screenspot_v2_qwen2_5_vl_3b_point_native_decoupled.yaml`
- `--decoupled-point-native-decode` wiring in `scripts/run_eval_screenspot_v2.py`
- pure-function coverage in `tests/test_qwen_point_refinement.py`

## Full Same-Protocol Evaluation

Run:

```bash
HF_ENDPOINT=https://hf-mirror.com conda run -n gui-grounding-py310 \
  python scripts/run_eval_screenspot_v2.py \
  --config configs/eval/screenspot_v2_qwen2_5_vl_3b_point_native_decoupled.yaml
```

Primary artifact root:

- `outputs/screenspot_v2_eval_qwen2_5_vl_3b_point_native_decoupled`

### New Decoupled Result

Artifacts:

- `outputs/screenspot_v2_eval_qwen2_5_vl_3b_point_native_decoupled/evaluation_summary.json`
- `outputs/screenspot_v2_eval_qwen2_5_vl_3b_point_native_decoupled/subgroup_metrics.json`
- `outputs/screenspot_v2_eval_qwen2_5_vl_3b_point_native_decoupled/predictions.jsonl`

Metrics:

| Metric | Point-Native Decoupled |
| --- | ---: |
| Evaluated samples | 1272 |
| Point accuracy | 0.7736 |
| Desktop point accuracy | 0.7365 |
| Mobile point accuracy | 0.8084 |
| Web point accuracy | 0.7620 |
| IoU@0.5 | 0.0967 |
| Mean IoU | 0.1912 |
| Action-type validity | 1.0000 |
| Parseable output rate | 1.0000 |
| Valid bbox rate | 0.9969 |
| Valid click point rate | 0.9969 |
| Icon point accuracy | 0.6625 |
| Text point accuracy | 0.8593 |

## Direct Comparison

### Versus Previous Structured Method (`0.7099`)

- overall point accuracy: `0.7099 -> 0.7736` (`+0.0637`)
- desktop point accuracy: `0.6976 -> 0.7365` (`+0.0389`)
- mobile point accuracy: `0.7445 -> 0.8084` (`+0.0639`)
- web point accuracy: `0.6796 -> 0.7620` (`+0.0824`)
- IoU@0.5: `0.1682 -> 0.0967` (`-0.0715`)
- mean IoU: `0.2404 -> 0.1912` (`-0.0492`)

### Versus Best Point-First Refinement (`0.7296`)

- overall point accuracy: `0.7296 -> 0.7736` (`+0.0440`)
- desktop point accuracy: `0.7305 -> 0.7365` (`+0.0060`)
- mobile point accuracy: `0.7764 -> 0.8084` (`+0.0319`)
- web point accuracy: `0.6751 -> 0.7620` (`+0.0870`)
- IoU@0.5: `0.1635 -> 0.0967` (`-0.0668`)
- mean IoU: `0.2365 -> 0.1912` (`-0.0452`)
- icon point accuracy: `0.5794 -> 0.6625` (`+0.0830`)
- text point accuracy: `0.8454 -> 0.8593` (`+0.0139`)

### Versus Reproduced Public Baseline (`0.7563`)

- overall point accuracy: `0.7563 -> 0.7736` (`+0.0173`)
- desktop point accuracy: `0.7275 -> 0.7365` (`+0.0090`)
- mobile point accuracy: `0.7944 -> 0.8084` (`+0.0140`)
- web point accuracy: `0.7346 -> 0.7620` (`+0.0275`)
- icon point accuracy: `0.6300 -> 0.6625` (`+0.0325`)
- text point accuracy: `0.8538 -> 0.8593` (`+0.0056`)
- IoU@0.5 remains above the public point baseline:
  - `0.0967` vs `0.0519`
- mean IoU remains above the public point baseline:
  - `0.1912` vs `0.1327`

## Decoupling Quality / Provenance

Measured from saved predictions:

- primary click provenance:
  - `point_native_primary_pass`: `1249 / 1272`
  - `structured_single_pass_fallback`: `23 / 1272`
- bbox provenance:
  - `structured_secondary_pass`: `1209 / 1272`
  - `derived_from_primary_click`: `40 / 1272`
  - `structured_single_pass_fallback`: `23 / 1272`
- action provenance:
  - `structured_secondary_pass`: `1249 / 1272`
  - `structured_single_pass_fallback`: `23 / 1272`

Interpretation:

- The new path is genuinely decoupled on the large majority of cases.
- The click is usually produced by the point-native pass, not inferred from the bbox.
- The bbox/action fields remain available as structured outputs for later candidate scoring work.

## Honest Interpretation

### Did decoupling point from bbox/action help?

Yes.

This is not a tiny prompt tweak effect:

- it changed the decoding structure
- it made click point the primary object
- it produced a large same-protocol gain over both prior structured runs

### Did it beat the best prior repo point-first result?

Yes.

- `0.7296 -> 0.7736`
- absolute gain: `+0.0440`

### Did it reach or exceed the reproduced public Qwen baseline?

Yes, under the repo’s same evaluation protocol.

- reproduced public baseline: `0.7563`
- new decoupled result: `0.7736`
- absolute delta: `+0.0173`

This remains a **same-protocol repo comparison**, not a claim about the official external GUI-Actor evaluation script.

### What tradeoff did this introduce?

Point accuracy improved substantially, but bbox metrics dropped versus the bbox-heavier structured path:

- point accuracy improved
- IoU metrics fell from the previous structured / point-first runs

This is consistent with the intended realignment:

- `click_point` is now primary
- `bbox_proposal` is supporting structure

### What is the most likely remaining bottleneck now?

The main remaining bottleneck is no longer click placement versus public plain Qwen.

The next bottleneck is likely:

- improving secondary bbox quality without dragging the point back into a bbox-first coupling regime
- reducing the fallback count and click-derived bbox count
- turning the new point-primary action object into a cleaner Stage-B candidate pool for reward-based reranking

## Why This Is More Blueprint-Aligned

This change moves the repo closer to the original proposal in three ways:

1. It treats click point as a valid primary form of `b` in the action object.
2. It keeps the structured action contract alive:
   - `bbox_proposal`
   - `click_point`
   - `action_type`
3. It adds provenance and schema metadata that make later reward-scored candidate generation cleaner rather than more entangled.

## Saved Comparison Artifacts

- vs previous structured:
  - `outputs/screenspot_v2_eval_qwen2_5_vl_3b_point_native_decoupled/comparison_vs_previous_structured.json`
  - `outputs/screenspot_v2_eval_qwen2_5_vl_3b_point_native_decoupled/comparison_vs_previous_structured.md`
- vs best prior point-first:
  - `outputs/screenspot_v2_eval_qwen2_5_vl_3b_point_native_decoupled/comparison_vs_point_first.json`
  - `outputs/screenspot_v2_eval_qwen2_5_vl_3b_point_native_decoupled/comparison_vs_point_first.md`
- vs reproduced public baseline:
  - `outputs/screenspot_v2_eval_qwen2_5_vl_3b_point_native_decoupled/comparison_vs_public_baseline.json`
  - `outputs/screenspot_v2_eval_qwen2_5_vl_3b_point_native_decoupled/comparison_vs_public_baseline.md`

## Direct Answer

- Did the blueprint-aligned move help?
  - **Yes**
- Did overall point accuracy improve beyond `0.7296`?
  - **Yes**
- Did it reach or exceed `0.7563`?
  - **Yes**, under the same repo protocol
- Is the repo now better aligned with the original blueprint?
  - **Yes**

