# Mind2Web Stage-B Conditional Singleton Recovery Supervision

## Objective

Keep the existing Qwen-first Stage A -> Stage B -> learned reranker pipeline intact.

Do not add candidate sources.
Do not expand the candidate pools again.
Do not replace the reranker family.

This step only targets the remaining singleton recovery bottlenecks:

- rare singleton point-first recovery on `test_domain`
- structured singleton disambiguation on `test_task`

The benchmark to beat remained:

- `outputs/mind2web_stageB_reranker_rare_recovery_targeted`

## Inputs Inspected Before Editing

- `docs/mind2web_stageB_rare_recovery_targeted_supervision.md`
- `docs/mind2web_stageB_rare_point_first_disagreement_support_supervision.md`
- `outputs/mind2web_stageB_reranker_rare_recovery_targeted/official_split_evaluations.json`
- `outputs/mind2web_stageB_reranker_rare_point_first_disagreement_support_v2/official_split_evaluations.json`
- `outputs/mind2web_stageB_reranker_rare_recovery_targeted/preference_pairs_train.jsonl`
- `outputs/mind2web_stageB_reranker_rare_point_first_disagreement_support_v2/preference_pairs_train.jsonl`
- `outputs/mind2web_stageB_candidates_headroom_expanded/*/candidates_*.jsonl`
- `src/gui_grounding/training/trainer_reranker.py`
- `scripts/run_train_reranker.py`

## Singleton Diagnosis

Saved compact singleton analysis:

- `outputs/mind2web_stageB_conditional_singleton_analysis_current/conditional_singleton_analysis.json`
- `outputs/mind2web_stageB_conditional_singleton_analysis_current/conditional_singleton_analysis.md`

Key findings:

1. The remaining singleton domain miss is still:
   - `test_domain` `mind2web_test_domain_981e4bf0-9049-4eab-b157-105473f53a97`
   - oracle source: `point_first_sampled_t0p7`
   - oracle gap: `0.3563`
   - current best reranker still selects `structured_sampled_t0p6`

2. The remaining structured singleton task miss is still:
   - `test_task` `mind2web_test_task_ea2865e4-2858-478c-bf83-93d576cad774`
   - oracle source: `structured_sampled_t0p6`
   - oracle gap: `0.4409`
   - current best reranker still selects `hybrid_point_structured`

3. The current best already recovers the other large structured singleton:
   - `test_task` `mind2web_test_task_3e671043-cab2-4e44-a1ce-3ed9de91d16b`
   - oracle gap: `0.4435`
   - current best reranker hits the oracle structured candidate

4. The failed rare point-first run was too broad.
   - it added full-strength point-first-targeted supervision to all three train singleton point-first positives
   - one of those train pools had only a tiny oracle gap `0.0100`
   - this did not recover the domain miss and it knocked out the already-recovered `test_task` singleton

5. The conditional signals are real and narrow.
   - isolated train point-first singleton positives above the chosen signal floor: `2`
   - isolated train point-first negatives above the same signal floor: `0`
   - exact train structured singleton analogue for the `test_task` rank-3/4 miss family: `1`

Chosen singleton conditions:

- point-first singleton:
  - source `point_first_sampled_t0p7`
  - rank `7+`
  - at least `3` structured decoys
  - zero bbox overlap to the pool
  - min structured click distance `>= 80`
  - positive signal `click_inside_target + iou >= 0.05`
  - oracle gap `>= 0.02`

- structured singleton:
  - source `structured_sampled_t0p6`
  - singleton positive at rank `3/4`
  - at least `2` same-source structured decoys
  - max same-source IoU `<= 0.05`
  - min same-source click distance `>= 80`
  - positive signal `click_inside_target + iou >= 1.20`
  - oracle gap `>= 0.35`

## What Changed

Updated:

- `src/gui_grounding/training/trainer_reranker.py`
- `scripts/run_train_reranker.py`

Added:

- `scripts/analyze_mind2web_stageb_conditional_singletons.py`
- `configs/train/mind2web_stageB_reranker_qwen_conditional_singleton_recovery.yaml`

### 1. Benchmark-Aligned Feature Gate

The current trainer code had drifted past the confirmed best reranker:

- current code path emitted the rejected structured-relative support feature extension
- the confirmed best reranker checkpoint was still on the earlier 84-dim feature path

So this step added a small explicit gate:

- `feature_include_structured_relative_support: false`

This keeps the new run aligned with the actual current-best reranker representation instead of silently reusing the rejected v3 feature variant.

### 2. Conditional Singleton Pair Redesign

New pair mode:

- `pair_construction_mode=conditional_singleton_recovery`

It preserves the current rare-recovery backbone:

- `recovery_vs_first_choice`
- `best_vs_other_positive`
- generic wrong-source/same-source decoys outside the singleton-targeted pools

But it changes singleton supervision in two targeted ways.

Structured singleton pools:

- `structured_singleton_best_vs_structured_decoy`
- `structured_singleton_support_anchor`

Point-first singleton pools:

- `point_first_best_vs_structured_disagreement`
- `point_first_support_anchor`

Two deliberate restrictions relative to the failed point-first run:

- only the strong point-first singleton train pools are targeted
- the tiny-gap point-first singleton train pool falls back to generic supervision instead of receiving heavy singleton bonuses

### 3. Conditional Weighting Instead Of Broad Boosting

The new run keeps `pair_weight_mode=rare_recovery_targeted`, but adds explicit conditional singleton controls:

- `pair_conditional_singleton_bonus: 0.75`
- `pair_point_first_signal_threshold: 0.05`
- `pair_point_first_gap_threshold: 0.02`
- `pair_structured_singleton_signal_threshold: 1.20`
- `pair_structured_singleton_gap_threshold: 0.35`
- `pair_structured_singleton_decoy_weight: 2.1`
- `pair_structured_singleton_support_anchor_weight: 1.8`

This narrows the strong singleton supervision to the cases where the reward-side signal is clearly separated from the train false-alarm prior.

### 4. Final Supervision Mix

Saved supervision summary:

- `outputs/mind2web_stageB_reranker_conditional_singleton_recovery/supervision_summary.json`
- `outputs/mind2web_stageB_reranker_conditional_singleton_recovery/supervision_summary.md`

Train-pair mix:

- total train pairs: `73`
- `recovery_vs_first_choice`: `18`
- `best_vs_other_positive`: `4`
- `best_vs_best_wrong_source`: `32`
- `best_vs_point_first_false_alarm`: `3`
- `point_first_best_vs_structured_disagreement`: `2`
- `point_first_support_anchor`: `4`
- `best_vs_same_source_decoy`: `5`
- `structured_singleton_best_vs_structured_decoy`: `2`
- `structured_singleton_support_anchor`: `3`

So the singleton supervision is now explicit and interpretable, but still small.

## Final Run

Final reranker:

- `outputs/mind2web_stageB_reranker_conditional_singleton_recovery`

Saved artifacts:

- `outputs/mind2web_stageB_reranker_conditional_singleton_recovery/official_split_evaluations.json`
- `outputs/mind2web_stageB_reranker_conditional_singleton_recovery/preference_pairs_train.jsonl`
- `outputs/mind2web_stageB_reranker_conditional_singleton_recovery/training_history.json`
- `outputs/mind2web_stageB_reranker_conditional_singleton_recovery/supervision_summary.json`
- `outputs/mind2web_stageB_reranker_conditional_singleton_recovery/supervision_summary.md`

Final singleton analysis:

- `outputs/mind2web_stageB_conditional_singleton_analysis_final/conditional_singleton_analysis.json`
- `outputs/mind2web_stageB_conditional_singleton_analysis_final/conditional_singleton_analysis.md`

Direct comparison:

- `outputs/mind2web_stageB_conditional_singleton_comparison/official_split_comparison.json`
- `outputs/mind2web_stageB_conditional_singleton_comparison/official_split_comparison.md`

## Official Split Results

Main comparison:

| Split | SFT reward | SFT point | Current best reward | Current best point | New reward | New point | Parseable | Oracle | Current best recovery | New recovery |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `test_task` | `0.1300` | `0.0000` | `0.1576` | `0.0500` | `0.1576` | `0.0500` | `1.0000` | `0.1986` | `0.4031` | `0.4031` |
| `test_website` | `0.1900` | `0.0000` | `0.2111` | `0.0500` | `0.2111` | `0.0500` | `1.0000` | `0.2117` | `0.9745` | `0.9745` |
| `test_domain` | `0.1158` | `0.0000` | `0.1359` | `0.0526` | `0.1359` | `0.0526` | `1.0000` | `0.1547` | `0.5181` | `0.5181` |

## Sample-Level Effect

The meaningful headroom decisions did not improve beyond the current best reranker.

Important singleton cases:

- `test_domain` `mind2web_test_domain_981e4bf0-9049-4eab-b157-105473f53a97`
  - still missed
  - current best: `structured_sampled_t0p6`
  - new run: `structured_sampled_t0p6`

- `test_task` `mind2web_test_task_ea2865e4-2858-478c-bf83-93d576cad774`
  - still missed
  - current best: `hybrid_point_structured`
  - new run: `hybrid_point_structured`

- `test_task` `mind2web_test_task_3e671043-cab2-4e44-a1ce-3ed9de91d16b`
  - still recovered correctly
  - current best: `structured_sampled_t0p6`
  - new run: `structured_sampled_t0p6`

Per-sample differences versus the current best were only tie-swaps on non-headroom pools:

- one `test_task` zero-reward tie
- one `test_website` equal-reward tie
- no `test_domain` differences at all

## Direct Answer

Did conditional singleton supervision beat the current best reranker?

- No.

Did the singleton domain case improve?

- No.
- `mind2web_test_domain_981e4bf0-9049-4eab-b157-105473f53a97` is still missed.

Did `test_task` stay strong?

- Yes.
- It exactly matched the current best: `0.1576 / 0.0500`.

Did `test_website` stay strong?

- Yes.
- It exactly matched the current best: `0.2111 / 0.0500`.

Did `test_domain` improve beyond `0.1359 / 0.0526`?

- No.
- It exactly tied the current best.

Does this meaningfully advance the blueprint?

- Analytically yes:
  - the singleton conditions are now isolated cleanly
  - the rejected v3 feature drift is now explicitly gated
  - singleton supervision is now small, interpretable, and thresholded instead of broad
- Benchmark-wise no:
  - it ties the current best instead of beating it

Are singleton cases still the dominant bottleneck after this step?

- Yes, together with the remaining structured multi-positive task disambiguation miss.

## Next Missing Piece

The next most important missing piece is not more pool expansion and not a new RL stack.

It is a stronger inference-time-discriminative ranking signal for the unresolved singleton top-choice conflicts that pair redesign alone did not move:

- the singleton `point_first_sampled_t0p7` domain miss versus its structured decoys
- the singleton structured `test_task` miss versus the hybrid / point-native anchors

In other words:

- singleton supervision is now cleanly targeted
- but the current lightweight scorer representation still does not separate the last unresolved singleton choices well enough to beat the benchmark
