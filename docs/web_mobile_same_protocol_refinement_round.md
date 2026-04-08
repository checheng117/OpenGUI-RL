# Web/Mobile Same-Protocol Refinement Round

## Scope

- Goal: make one very small same-protocol refinement aimed at web/mobile point accuracy without redesigning the pipeline.
- Constraints kept:
  - Qwen-first path only
  - no new training
  - no reranker / DPO / GRPO work
  - no bbox-heavy redesign
  - same point-first structured path retained
  - same parser/export contract retained: `bbox_proposal`, `click_point`, `action_type`

## Artifacts Inspected Before Editing

- `docs/public_baseline_reproduction_and_same_protocol_comparison.md`
- `docs/point_accuracy_first_refinement_against_public_qwen.md`
- `configs/eval/screenspot_v2_qwen2_5_vl_3b_point_accuracy_first.yaml`
- `src/gui_grounding/models/qwen2_vl_grounding.py`
- `scripts/run_eval_screenspot_v2.py`
- `outputs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized/predictions.jsonl`
- `outputs/screenspot_v2_eval_qwen2_5_vl_3b_point_accuracy_first/predictions.jsonl`
- `outputs/screenspot_v2_public_qwen2_5_vl_3b_point_baseline/predictions.jsonl`

## Concise Pre-Edit Failure Pattern

Current point-first baseline before this round:

- overall point accuracy: `0.7296`
- desktop point accuracy: `0.7305`
- web point accuracy: `0.6751`
- mobile point accuracy: `0.7764`

Grounded observations from the saved current point-first predictions:

- The old top-left-corner issue was no longer the main web/mobile failure pattern.
- In current web misses, `88 / 142` were `icon`; in current mobile misses, `81 / 112` were `icon`.
- Versus the previous structured run, current point-first helped `web` text slightly (`0.7607 -> 0.7692`) but hurt `web` icon (`0.5862 -> 0.5665`).
- On web cases where the reproduced public baseline hit and current point-first missed, `28 / 51` were `icon`.
- Those web public-win cases were still often border-biased inside the model’s own predicted bbox:
  - nearest-edge distance `<= 0.2` for `45.1%`
  - nearest-edge distance `<= 0.1` for `29.4%`
  - top-left-band concentration was `0.0%`

Interpretation:

- remaining web/mobile misses were more consistent with general edge-biased hotspot selection and icon sensitivity than with the old top-left bug
- that supported a very small prompt-only refinement centered on actionable hotspot center wording for web/mobile

## Tiny Refinement Used

Only one refinement family was implemented:

- web/mobile-specific hotspot-center prompt wording in `src/gui_grounding/models/qwen2_vl_grounding.py`

What changed:

- added a gated `web_mobile_hotspot_prompt` flag
- when enabled for `web` or `mobile`, the prompt now says to click the center of the actual clickable/tappable hotspot
- explicitly discourages left/top edge clicks, borders, row/container edges, and nearby whitespace
- adds light metadata-aware wording:
  - for `text`: click the middle of the label/text region, not the first character or left edge
  - for `icon`: click the visual center of the icon/button glyph, not surrounding padding or card area

What did **not** change:

- parser logic
- output export fields
- bbox/click/action downstream format
- point-first decode path
- edge-click refinement logic

Supporting additions:

- new eval config:
  - `configs/eval/screenspot_v2_qwen2_5_vl_3b_web_mobile_hotspot_refinement.yaml`
- prompt-unit coverage:
  - `tests/test_qwen_point_refinement.py`

## Same-Protocol Re-Evaluation

Artifact root:

- `outputs/screenspot_v2_eval_qwen2_5_vl_3b_web_mobile_hotspot_refinement`

Metrics:

- evaluated samples: `1272`
- overall point accuracy: `0.7264`
- desktop point accuracy: `0.7305`
- web point accuracy: `0.6796`
- mobile point accuracy: `0.7645`
- parseable output rate: `0.9992`
- action-type validity: `0.9992`
- IoU@0.5: `0.1588`
- mean IoU: `0.2286`

Useful web/mobile text-vs-icon detail against the current point-first baseline:

- web icon point accuracy: `0.5665 -> 0.5665` (`+0.0000`)
- web text point accuracy: `0.7692 -> 0.7778` (`+0.0085`)
- mobile icon point accuracy: `0.6161 -> 0.5924` (`-0.0237`)
- mobile text point accuracy: `0.8931 -> 0.8897` (`-0.0034`)

## Direct Comparison

### Versus Previous Structured Method (`0.7099`)

- overall point accuracy: `0.7099 -> 0.7264` (`+0.0165`)
- desktop point accuracy: `0.6976 -> 0.7305` (`+0.0329`)
- web point accuracy: `0.6796 -> 0.6796` (`+0.0000`)
- mobile point accuracy: `0.7445 -> 0.7645` (`+0.0200`)

### Versus Current Point-First Baseline (`0.7296`)

- overall point accuracy: `0.7296 -> 0.7264` (`-0.0031`)
- desktop point accuracy: `0.7305 -> 0.7305` (`+0.0000`)
- web point accuracy: `0.6751 -> 0.6796` (`+0.0046`)
- mobile point accuracy: `0.7764 -> 0.7645` (`-0.0120`)
- parseable output rate: `1.0000 -> 0.9992` (`-0.0008`)
- action-type validity: `1.0000 -> 0.9992` (`-0.0008`)

### Versus Reproduced Public Baseline (`0.7563`)

- reproduced public baseline point accuracy: `0.7563`
- overall point accuracy: `0.7563 -> 0.7264` (`-0.0299`)
- desktop point accuracy: `0.7275 -> 0.7305` (`+0.0030`)
- web point accuracy: `0.7346 -> 0.6796` (`-0.0549`)
- mobile point accuracy: `0.7944 -> 0.7645` (`-0.0299`)

Gap accounting:

- previous gap from current point-first baseline to public baseline: `0.7563 - 0.7296 = 0.0267`
- new gap after this refinement: `0.7563 - 0.7264 = 0.0299`
- result: the gap **widened** by `0.0031`

## Direct Answer

- Did web/mobile improve?
  - `web`: **yes, slightly** (`+0.0046`)
  - `mobile`: **no, regressed** (`-0.0120`)
- Did overall point accuracy improve versus the current point-first baseline?
  - **no** (`0.7296 -> 0.7264`, `-0.0031`)
- Did desktop stay stable enough?
  - **yes**, desktop stayed exactly at `0.7305`
- Did we close the gap to `0.7563`?
  - **no**
- Are we above or below the reproduced public baseline?
  - **still below**

Most likely remaining bottleneck in one sentence:

- web/mobile icon targeting is still more bbox-coupled and edge-sensitive than the reproduced plain point-native Qwen behavior.

## Saved Artifacts

- merged evaluation summary:
  - `outputs/screenspot_v2_eval_qwen2_5_vl_3b_web_mobile_hotspot_refinement/evaluation_summary.json`
- merged subgroup metrics:
  - `outputs/screenspot_v2_eval_qwen2_5_vl_3b_web_mobile_hotspot_refinement/subgroup_metrics.json`
- comparison vs previous structured:
  - `outputs/screenspot_v2_eval_qwen2_5_vl_3b_web_mobile_hotspot_refinement/comparison_vs_previous_structured.json`
  - `outputs/screenspot_v2_eval_qwen2_5_vl_3b_web_mobile_hotspot_refinement/comparison_vs_previous_structured.md`
- comparison vs current point-first baseline:
  - `outputs/screenspot_v2_eval_qwen2_5_vl_3b_web_mobile_hotspot_refinement/comparison_vs_current_point_first_baseline.json`
  - `outputs/screenspot_v2_eval_qwen2_5_vl_3b_web_mobile_hotspot_refinement/comparison_vs_current_point_first_baseline.md`
- comparison vs reproduced public baseline:
  - `outputs/screenspot_v2_eval_qwen2_5_vl_3b_web_mobile_hotspot_refinement/comparison_vs_public_baseline.json`
  - `outputs/screenspot_v2_eval_qwen2_5_vl_3b_web_mobile_hotspot_refinement/comparison_vs_public_baseline.md`
