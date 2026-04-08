# Qwen Medium Candidate Export And Quality Report

## Scope

- Goal completed: real Qwen-first medium-scale candidate export on 50 real Mind2Web train samples.
- Out of scope and unchanged: surrogate-path optimization, reranker/DPO/GRPO work, full-pipeline redesign, CLIP-grid as main path.
- Frozen safe baseline preserved unchanged:
  - `configs/train/generate_candidates_qwen2_5_vl_3b_safe_baseline_locked.yaml`

## Inputs Inspected Before Editing

- Export runner:
  - `scripts/run_generate_candidates.py`
- Qwen export backend:
  - `src/gui_grounding/models/qwen2_vl_grounding.py`
  - `src/gui_grounding/models/vlm_backbone.py`
- Existing configs:
  - `configs/train/generate_candidates_qwen2_5_vl_3b_safe_baseline_locked.yaml`
  - `configs/train/generate_candidates_qwen2_5_vl_3b_safe.yaml`
  - `configs/train/generate_candidates_qwen2_5_vl_3b.yaml`
- Existing Qwen reports/logs:
  - `docs/qwen_candidate_export_safe_run.md`
  - `docs/qwen_bridge_to_original_crash.md`
  - `outputs/logs/qwen_candidate_export_qwen2_5_safe.log`
  - `outputs/logs/qwen_candidate_export_qwen2_5.log`
  - `outputs/qwen_bridge_repro/run_artifacts/phase6_final_full_repro_target/summary_train.json`

## Minimal Changes Made

1. Kept the export path Qwen-first and kept CLIP-grid explicitly legacy-only.
2. Added canonical schema metadata so exported candidates are explicitly centered on:
   - `bbox_proposal`
   - `click_point`
   - `action_type`
3. Added Qwen alias support in parsing so canonical keys are accepted without breaking legacy `predicted_*` payloads:
   - `bbox_proposal` or `predicted_bbox`
   - `click_point` or `predicted_click_point`
   - `element_hint_id` or `predicted_element_id`
4. Added per-candidate structured-output diagnostics to the export artifacts.
5. Added machine-readable and human-readable structured quality summaries from the real export run.
6. Added a dedicated explicit medium-run config instead of changing the frozen safe baseline.

## Candidate Semantics Status

Current exported candidate objects are now explicitly documented with:

- `candidate_schema.version = bbox_click_action_v1`
- `candidate_schema.primary_fields = ["bbox_proposal", "click_point", "action_type"]`
- `candidate_schema.legacy_fields = ["legacy_metadata.grid_id"]`

Important nuance from the measured run:

- The **exported artifacts** are canonical bbox/click/action objects.
- The **raw Qwen model payloads** still mostly used legacy key names (`predicted_bbox`, `predicted_click_point`) during this run.
- This is handled cleanly by the parser, and `grid_id` does not remain a primary exported semantic field.

## Dedicated Medium Config

- Added:
  - `configs/train/generate_candidates_qwen2_5_vl_3b_medium_50.yaml`

This config explicitly uses the validated higher-load operating point from the bridge study:

- backbone: `qwen2_5_vl_3b`
- dtype: `bfloat16`
- attention backend: `sdpa`
- `top_k: 4`
- `max_new_tokens: 256`
- `max_samples: 50`
- incremental save: enabled
- per-sample fault isolation: enabled

The safe baseline default was not overwritten.

## Exact Commands Actually Run

```bash
python -m py_compile scripts/run_generate_candidates.py src/gui_grounding/models/qwen2_vl_grounding.py
```

```bash
conda run -n gui-grounding-py310 env HF_ENDPOINT=https://hf-mirror.com python scripts/run_generate_candidates.py --config configs/train/generate_candidates_qwen2_5_vl_3b_medium_50.yaml --num-samples 2 > outputs/logs/qwen_candidate_export_medium50_smoke.log 2>&1
```

```bash
stdbuf -oL -eL conda run -n gui-grounding-py310 env HF_ENDPOINT=https://hf-mirror.com python scripts/run_generate_candidates.py --config configs/train/generate_candidates_qwen2_5_vl_3b_medium_50.yaml 2>&1 | tee outputs/logs/qwen_candidate_export_medium50.log
```

## Environment Used

- Conda env: `gui-grounding-py310`
- Python:
  - `/home/cc/.conda/envs/gui-grounding-py310/bin/python`
- HF mirror endpoint used: **yes**
  - `HF_ENDPOINT=https://hf-mirror.com`
- Model used:
  - `Qwen/Qwen2.5-VL-3B-Instruct`
- Runtime used for final 50-sample run:
  - `torch_dtype=torch.bfloat16`
  - `attn_implementation=sdpa`
  - `gpu_memory_utilization=0.7`
  - `max_new_tokens=256`
  - `temperature=0.2`
  - `min_pixels=65536`
  - `max_pixels=524288`

## Real 50-Sample Export Result

- Config:
  - `configs/train/generate_candidates_qwen2_5_vl_3b_medium_50.yaml`
- Split:
  - `train`
- Real samples attempted:
  - `50`
- Successful end-to-end runs:
  - `50`
- Failed runs:
  - `0`
- Total exported candidates:
  - `200`
- Average candidates per successful sample:
  - `4.0`

Run timing from log:

- Start:
  - `2026-04-02 16:38:26`
- Finish:
  - `2026-04-02 16:43:35`

## Output Artifacts

- Grouped candidates:
  - `outputs/candidate_generation_qwen2_5_vl_3b_medium_50/candidates_train_medium50.jsonl`
- Flat candidates:
  - `outputs/candidate_generation_qwen2_5_vl_3b_medium_50/candidates_train_medium50_flat.jsonl`
- Run summary:
  - `outputs/candidate_generation_qwen2_5_vl_3b_medium_50/summary_train_medium50.json`
- Failures:
  - `outputs/candidate_generation_qwen2_5_vl_3b_medium_50/failures_train_medium50.json`
- Runtime details:
  - `outputs/candidate_generation_qwen2_5_vl_3b_medium_50/runtime_train_medium50.json`
- Structured quality summary:
  - `outputs/candidate_generation_qwen2_5_vl_3b_medium_50/structured_quality_summary.json`
- Structured quality summary (human-readable):
  - `outputs/candidate_generation_qwen2_5_vl_3b_medium_50/structured_quality_summary.md`
- Action histogram:
  - `outputs/candidate_generation_qwen2_5_vl_3b_medium_50/action_type_histogram.json`
- Full execution log:
  - `outputs/logs/qwen_candidate_export_medium50.log`

## Structured Output Quality Statistics

Measured from the final 50-sample export artifacts:

- attempted samples: `50`
- successful end-to-end runs: `50`
- failed runs: `0`
- parseable structured outputs: `200 / 200 = 1.0000`
- valid `bbox_proposal`: `199 / 200 = 0.9950`
- valid `click_point`: `199 / 200 = 0.9950`
- valid `action_type`: `200 / 200 = 1.0000`
- action type distribution:
  - `click: 161`
  - `type: 39`
- average candidates per sample: `4.0`
- empty-candidate samples: `0 / 50 = 0.0000`
- malformed JSON: `0 / 200 = 0.0000`
- malformed coordinate outputs: `67 / 200 = 0.3350`
- bbox clipping corrections applied: `0`
- click clipping corrections applied: `0`
- bbox axis reorder corrections applied: `0`

Common parse/structure issue categories:

- `malformed_bbox: 67`
- `bbox_from_click_fallback: 85`
- `invalid_action_type: 7`
- `missing_bbox: 1`
- `malformed_click_point: 1`
- `missing_click_point: 1`

Additional measured notes:

- `bbox_field_histogram`:
  - `predicted_bbox: 181`
- `click_field_histogram`:
  - `predicted_click_point: 200`
- `confidence_source_histogram`:
  - `model_json: 194`
  - `uniform_fallback: 6`

## Interpretation

- Runtime stability for the Qwen-first medium export is strong in the current environment:
  - 50/50 sample-level completion
  - incremental persistence survived throughout the run
  - no sample-level crashes
- Structured parseability is also strong:
  - 200/200 candidate outputs were parseable as structured JSON
  - 200/200 action types were valid after parser normalization
- The main remaining quality issue is coordinate cleanliness:
  - bbox validity was high at `99.5%`
  - but raw bbox formatting issues were still common (`67` malformed bbox payloads)
  - `85` candidates required bbox synthesis from click-point fallback

This does **not** block held-out evaluation, but it is the main residual quality risk likely to affect held-out grounding quality.

## Is The Repo Ready For Held-Out Evaluation?

**Yes, from a runtime/export-contract standpoint.**

Why:

- Qwen-first export is stable on a real 50-sample run.
- Canonical exported semantics are bbox/click/action-centered.
- Failures are counted honestly.
- Output artifacts are saved in a reusable form under `outputs/`.
- The repo is now set up so a ScreenSpot-v2 adapter/eval can consume the same candidate schema next.

## Current Blockers Before Clean Held-Out Evaluation

No hard blocker remains for starting held-out evaluation.

Residual risks to carry into the next step:

1. Raw model bbox formatting is still noisy, even though exported bbox validity is high after fallback handling.
2. The current measured action distribution over this 50-sample slice is narrow (`click`/`type` only), so broader held-out coverage should be checked explicitly.
3. The canonical export schema is now clean, but the raw model payload is still mostly using legacy `predicted_*` keys rather than canonical `bbox_proposal` / `click_point`.

## Exact Recommended Next Step

Implement a clean held-out evaluation adapter for ScreenSpot-v2 (or equivalent held-out split) that consumes:

- `bbox_proposal`
- `click_point`
- `action_type`

from the current export schema without changing the generation pipeline again.

Concretely:

1. Reuse `configs/train/generate_candidates_qwen2_5_vl_3b_medium_50.yaml` as the explicit medium operating point when needed.
2. Keep `configs/train/generate_candidates_qwen2_5_vl_3b_safe_baseline_locked.yaml` as the unchanged fallback.
3. Build the ScreenSpot-v2 adapter/eval around the current canonical candidate schema and carry forward the same structured quality summary fields for held-out reporting.

## Files Changed

- `src/gui_grounding/models/qwen2_vl_grounding.py`
- `scripts/run_generate_candidates.py`
- `configs/train/generate_candidates_qwen2_5_vl_3b_medium_50.yaml`
- `docs/qwen_medium_candidate_export_and_quality_report.md`
