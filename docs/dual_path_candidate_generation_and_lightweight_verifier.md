# Dual-Path Candidate Generation and Lightweight Verifier

## Goal

Move the repo one step closer to the original proposal blueprint without redesigning the whole pipeline:

- keep the current Qwen-first setup
- keep both existing strong paths
- generate a small candidate set from them
- add a cheap, interpretable verifier on top
- test whether candidate selection beats the reproduced public Qwen baseline and the current point-native method on the same ScreenSpot-v2 protocol

This task did **not** introduce:

- large new training
- a new surrogate path
- broad prompt sweeps
- DPO / GRPO / PPO

## Inputs Inspected Before Editing

- proposal:
  - `docs/gui_grounding_project_proposal.docx`
- blueprint/system design:
  - `docs/system_design.md`
- reproduced public baseline comparison:
  - `docs/public_baseline_reproduction_and_same_protocol_comparison.md`
- current point-native strongest path:
  - `docs/blueprint_realignment_point_native_decoupling.md`
  - `configs/eval/screenspot_v2_qwen2_5_vl_3b_point_native_decoupled.yaml`
  - `outputs/screenspot_v2_eval_qwen2_5_vl_3b_point_native_decoupled/*`
- current structured bbox-heavy path:
  - `configs/eval/screenspot_v2_qwen2_5_vl_3b_model_resized_full.yaml`
  - `outputs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized/*`
- current evaluation/output contract:
  - `scripts/run_eval_screenspot_v2.py`
  - `src/gui_grounding/models/qwen2_vl_grounding.py`
  - `src/gui_grounding/reward/verifiable_reward.py`

## What Changed

### 1. Dual-path candidate builder

Added:

- `src/gui_grounding/reward/lightweight_verifier.py`

This builds a per-sample candidate pool directly from the saved same-protocol prediction artifacts of:

- `point_native_primary`
- `structured_single_pass`
- `hybrid_point_structured`

The hybrid candidate is the practical bridge between the two paths:

- click point from the stronger point-native path
- bbox/action from the stronger structured path
- bbox is expanded only if needed to contain the selected point-native click

Candidate schema:

- `bbox_click_action_v3_dual_path_candidates`

Each candidate exposes:

- `click_point`
- `bbox_proposal`
- `action_type`
- `source_path`
- `element_hint_id`
- `confidence`
- `click_provenance`
- `bbox_provenance`
- `action_provenance`
- `parser_metadata`
- `source_artifacts`

### 2. Lightweight verifier

Added:

- `scripts/run_eval_dual_path_verifier.py`
- `configs/eval/screenspot_v2_qwen2_5_vl_3b_dual_path_verifier.yaml`

The verifier is intentionally cheap and interpretable:

- default preference: `hybrid_point_structured`
- structured override only when:
  - the two paths disagree enough in click location
  - the structured click lies inside the point-native bbox
  - the point-native click does **not** lie inside the structured bbox
  - structured confidence is high enough

Score components are explicit:

- source prior
- format validity
- action validity
- click-inside-own-bbox consistency
- support from the other path’s bbox
- confidence
- structured override bonus

Verifier schema:

- `lightweight_dual_path_verifier_v1`

### 3. Same-protocol artifact generation

The new evaluator writes a selected `predictions.jsonl` under the same contract as the existing ScreenSpot-v2 runner:

- `bbox_proposal`
- `click_point`
- `action_type`

It also saves:

- `candidate_artifacts.jsonl`
- `verifier_outputs.jsonl`
- `evaluation_summary.json`
- `subgroup_metrics.json`
- `comparison_vs_public_baseline.{json,md}`
- `comparison_vs_point_native.{json,md}`
- `comparison_vs_structured.{json,md}`
- `core_method_comparison.{json,md}`

Primary output root:

- `outputs/screenspot_v2_eval_qwen2_5_vl_3b_dual_path_verifier`

## Same-Protocol Results

### Core comparison

| Method | Point Acc | Desktop | Web | Mobile | IoU@0.5 | Mean IoU | Action Valid | Parseable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reproduced public baseline | 0.7563 | 0.7275 | 0.7346 | 0.7944 | 0.0519 | 0.1327 | 0.2980 | 0.9937 |
| Structured only | 0.7099 | 0.6976 | 0.6796 | 0.7445 | 0.1682 | 0.2404 | 1.0000 | 1.0000 |
| Point-native only | 0.7736 | 0.7365 | 0.7620 | 0.8084 | 0.0967 | 0.1912 | 1.0000 | 1.0000 |
| Dual-path + verifier | 0.7791 | 0.7485 | 0.7643 | 0.8124 | 0.1722 | 0.2520 | 1.0000 | 1.0000 |

### Direct deltas

Versus point-native only:

- overall point accuracy: `0.7736 -> 0.7791` (`+0.0055`)
- desktop point accuracy: `0.7365 -> 0.7485` (`+0.0120`)
- web point accuracy: `0.7620 -> 0.7643` (`+0.0023`)
- mobile point accuracy: `0.8084 -> 0.8124` (`+0.0040`)
- IoU@0.5: `0.0967 -> 0.1722` (`+0.0755`)
- mean IoU: `0.1912 -> 0.2520` (`+0.0607`)

Versus reproduced public baseline:

- overall point accuracy: `0.7563 -> 0.7791` (`+0.0228`)
- desktop point accuracy: `0.7275 -> 0.7485` (`+0.0210`)
- web point accuracy: `0.7346 -> 0.7643` (`+0.0297`)
- mobile point accuracy: `0.7944 -> 0.8124` (`+0.0180`)

Versus structured only:

- overall point accuracy: `0.7099 -> 0.7791` (`+0.0692`)
- IoU@0.5: `0.1682 -> 0.1722` (`+0.0039`)
- mean IoU: `0.2404 -> 0.2520` (`+0.0115`)

### Oracle headroom and actual selection behavior

- oracle best-of-two point accuracy from point-native vs structured: `0.8160`
- actual dual-path verifier point accuracy: `0.7791`
- remaining gap to oracle: `0.0369`

Selected source distribution:

- `hybrid_point_structured`: `1183 / 1272`
- `structured_single_pass`: `89 / 1272`

Per platform:

- desktop: `312` hybrid, `22` structured
- mobile: `449` hybrid, `52` structured
- web: `422` hybrid, `15` structured

Interpretation:

- most of the gain comes from using the hybrid candidate as the stable default
- a smaller but real part comes from allowing a structured-click override on disagreement cases
- the verifier only recovers part of the available oracle headroom, so there is still room for a stronger Stage-B-style selector later

## Honest Interpretation

### Does dual-path candidate generation help?

Yes.

It is stronger than either single path alone:

- better point accuracy than the current point-native method
- far better point accuracy than structured only
- stronger bbox quality than point-native only

### Does the lightweight verifier improve over point-native alone?

Yes, modestly but clearly under the same protocol.

- point accuracy gain: `+0.0055`

This is not a bug-fix recovery claim, because the coordinate-frame issue was already fixed before this task. The gain comes from actual candidate-set construction plus selection.

### Does it improve over the reproduced public baseline?

Yes.

- reproduced public baseline: `0.7563`
- dual-path + verifier: `0.7791`
- gain: `+0.0228`

### Does it preserve useful bbox/structured quality?

Yes, and it actually improves structured quality relative to both major single-path references:

- vs point-native only:
  - `IoU@0.5`: `+0.0755`
  - `mean IoU`: `+0.0607`
- vs structured only:
  - `IoU@0.5`: `+0.0039`
  - `mean IoU`: `+0.0115`

### Is this a meaningful return to the original blueprint?

Yes.

This method now matches the proposal structure more directly:

- screenshot + instruction
- multiple candidate actions from different generators
- verifiable / interpretable scoring signals
- lightweight reranking / selection

At the same time, the improvement is still modest relative to oracle best-of-two headroom, so it should be described as a solid Stage-B-style baseline improvement, not a solved reranking problem.

## Final Answer

- Dual-path candidate generation helps.
- The lightweight verifier does improve over point-native alone.
- It is above both the reproduced public Qwen baseline and the current point-native method.
- It preserves and strengthens useful bbox/structured quality.
- This is a real blueprint-aligned method improvement, not just another recovery from an evaluation bug.
