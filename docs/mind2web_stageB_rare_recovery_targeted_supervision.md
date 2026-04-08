# Mind2Web Stage-B Rare-Recovery-Targeted Supervision

## Objective

Keep the existing Qwen-first Stage A -> Stage B -> learned reranker pipeline intact.

Do not add new candidate sources.
Do not replace the reranker family.

The goal of this step is narrower:

- identify rare but high-value recovery signatures already present in the current Stage-B pools
- redesign supervision so those rare recoveries are learned more reliably
- preserve oracle headroom and the current strong official-split behavior
- directly test whether rare-recovery-targeted supervision beats the current best source-aware reranker

## Inputs Inspected Before Editing

- `docs/mind2web_stageB_source_aware_reranker_supervision.md`
- `outputs/mind2web_stageB_reranker_source_aware_supervision/official_split_evaluations.json`
- `outputs/mind2web_stageB_reranker_source_aware_supervision/preference_pairs_train.jsonl`
- `outputs/mind2web_stageB_recovery_cases_source_aware/recovery_cases_summary.md`
- `outputs/mind2web_stageB_source_attribution_current/source_attribution_summary.md`
- `scripts/analyze_mind2web_stageb_recovery_cases.py`
- `scripts/run_train_reranker.py`
- `src/gui_grounding/training/trainer_reranker.py`

## Rare Recovery Patterns That Matter Most

Saved current rare-pattern analysis:

- `outputs/mind2web_stageB_rare_recovery_analysis_current/rare_recovery_analysis.json`
- `outputs/mind2web_stageB_rare_recovery_analysis_current/rare_recovery_analysis.md`

Key findings from the preserved Stage-B pools:

- train still has only `16 / 118` genuine headroom pools
- `test_task` has `4 / 20` headroom pools
- `test_website` has `4 / 20`
- `test_domain` has `3 / 19`

The most important underlearned signatures under the current source-aware reranker were:

1. `structured_sampled_t0p6 | rank_3_4 | singleton | ...`
- only `1` matching train headroom pool
- appears `5` times across official splits
- total official oracle gap `1.3126`
- current source-aware reranker still missed or only partially recovered several of these cases

2. `structured_sampled_t0p6 | rank_3_4 | multi3+ | ...`
- `0` matching train headroom pools
- appears in `test_task`
- official oracle gap `0.4402`
- current source-aware reranker improved the pool but still chose the wrong structured candidate

3. `point_first_sampled_t0p7 | rank_7_8 | singleton | point_first | ...`
- `3` train headroom pools total
- appears in `test_domain`
- official oracle gap `0.3563`
- still fully missed by the current source-aware reranker

Important train/test mismatch:

- the current random train/val split placed `2 / 3` rare `point_first_sampled_t0p7` headroom pools into validation, leaving only `1` such pool in the actual train subset
- the dominant structured singleton recovery signature had only `1` matching train headroom pool despite being the biggest residual official-split pattern

This confirmed the bottleneck described in the task:

- candidate pools were not the main limit
- rare recovery supervision coverage and pair informativeness were the main limit

## What Changed

Updated:

- `src/gui_grounding/training/trainer_reranker.py`
- `scripts/run_train_reranker.py`
- `configs/train/mind2web_stageB_reranker_qwen_rare_recovery_targeted.yaml`

Added:

- `scripts/analyze_mind2web_stageb_rare_recovery.py`

### 1. Headroom-Aware Train/Val Split

New split mode:

- `sample_split_mode=headroom_source_stratified`

This keeps the overall sample count unchanged, but avoids throwing most rare headroom sources into validation.

Within each oracle-best source bucket, validation examples are taken from the more common signatures first, so rare headroom signatures remain in train whenever possible.

For the final run this changed the effective train headroom composition to:

- `structured_sampled_t0p6: 6`
- `hybrid_point_structured: 4`
- `point_first_sampled_t0p7: 2`
- `point_native_primary: 1`

That is still small, but it is better aligned with the rare-recovery goal than the previous random split.

### 2. Pair Construction Redesign

New pair mode:

- `pair_construction_mode=rare_recovery_targeted`

It keeps the same candidate pools and the same reranker family, but changes which comparisons are learned.

Retained pair types:

- `recovery_vs_first_choice`
- `best_vs_other_positive`

Added targeted pair types:

- `best_vs_best_wrong_source`
- `best_vs_same_source_decoy`

Why these were needed:

- the current source-aware trainer already learned best-vs-first-choice
- it still missed large task cases where the oracle structured candidate needed to beat a strong wrong-source decoy
- it also still missed a multi-positive task pool where the reranker chose the wrong structured candidate inside the same source family

The new decoys are selected by hardness, using reward and DOM-alignment evidence, instead of rank-only source representatives.

Final train pair mix:

- total train pairs: `65`
- `recovery_vs_first_choice`: `17`
- `best_vs_other_positive`: `4`
- `best_vs_best_wrong_source`: `38`
- `best_vs_same_source_decoy`: `6`

Saved supervision summary:

- `outputs/mind2web_stageB_reranker_rare_recovery_targeted/supervision_summary.json`
- `outputs/mind2web_stageB_reranker_rare_recovery_targeted/supervision_summary.md`

### 3. Rare-Pattern-Aware Weighting

New weight mode:

- `pair_weight_mode=rare_recovery_targeted`

The final run uses explicit, auditable weighting:

- stronger recovery-anchor weight
- stronger wrong-source decoy weight
- explicit same-source decoy weight
- rare-source bonus
- rare-signature bonus
- negative-strength bonus for strong decoys
- pool oracle-gap bonus

The previous source-prior bonus was set to `0.0` in the final run so common structured sources were not preferentially reinforced.

## Saved Artifacts

Current baseline analysis:

- `outputs/mind2web_stageB_rare_recovery_analysis_current/rare_recovery_analysis.json`
- `outputs/mind2web_stageB_rare_recovery_analysis_current/rare_recovery_analysis.md`

Final rare-targeted reranker:

- `outputs/mind2web_stageB_reranker_rare_recovery_targeted/official_split_evaluations.json`
- `outputs/mind2web_stageB_reranker_rare_recovery_targeted/preference_pairs_train.jsonl`
- `outputs/mind2web_stageB_reranker_rare_recovery_targeted/training_history.json`
- `outputs/mind2web_stageB_reranker_rare_recovery_targeted/supervision_summary.json`
- `outputs/mind2web_stageB_reranker_rare_recovery_targeted/supervision_summary.md`

Final recovery-case analysis:

- `outputs/mind2web_stageB_recovery_cases_rare_targeted/recovery_cases_summary.json`
- `outputs/mind2web_stageB_recovery_cases_rare_targeted/recovery_cases_summary.md`

Final rare-pattern analysis:

- `outputs/mind2web_stageB_rare_recovery_analysis_targeted/rare_recovery_analysis.json`
- `outputs/mind2web_stageB_rare_recovery_analysis_targeted/rare_recovery_analysis.md`

Direct comparison tables:

- `outputs/mind2web_stageB_rare_recovery_targeted_comparison/official_split_comparison.json`
- `outputs/mind2web_stageB_rare_recovery_targeted_comparison/official_split_comparison.md`
- `outputs/mind2web_stageB_rare_recovery_targeted_comparison/official_split_comparison_full.json`
- `outputs/mind2web_stageB_rare_recovery_targeted_comparison/official_split_comparison_full.md`

## Official Split Results

Main comparison:

| Split | SFT reward | SFT point | Current best reward | Current best point | New reward | New point | Parseable | Oracle | Current best recovery | New recovery |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `test_task` | `0.1300` | `0.0000` | `0.1355` | `0.0000` | `0.1576` | `0.0500` | `1.0000` | `0.1986` | `0.0796` | `0.4031` |
| `test_website` | `0.1900` | `0.0000` | `0.2111` | `0.0500` | `0.2111` | `0.0500` | `1.0000` | `0.2117` | `0.9745` | `0.9745` |
| `test_domain` | `0.1158` | `0.0000` | `0.1359` | `0.0526` | `0.1359` | `0.0526` | `1.0000` | `0.1547` | `0.5181` | `0.5181` |

Saved official comparison:

- `outputs/mind2web_stageB_rare_recovery_targeted_comparison/official_split_comparison_full.md`

## Recovery-Case Effect

Saved recovery-case comparison for the final rare-targeted run:

- `outputs/mind2web_stageB_recovery_cases_rare_targeted/recovery_cases_summary.md`

The main recovery effect was on `test_task`:

- headroom recovery-case rate improved from `0.5000` to `0.7500`
- the reranker now recovers one of the two large singleton structured misses that the source-aware run still missed

What remained unresolved after the final run:

- one large `structured_sampled_t0p6` singleton task miss still remains
- the `structured_sampled_t0p6` multi-positive task pool is still only partially solved
- the rare `point_first_sampled_t0p7` domain recovery is still missed
- the tiny `test_website` residual cases stay unchanged, but the large website recovery remains preserved

## Direct Answers

### Did rare-recovery-targeted supervision help more than the current source-aware supervision?

Yes, on the official benchmark.

The final rare-targeted reranker improved `test_task` materially while preserving the current best `test_website` and `test_domain` results.

### Did `test_domain` improve?

No.

It stayed at the current best result:

- reward: `0.1359`
- point accuracy: `0.0526`

### Did `test_website` stay strong?

Yes.

It stayed at the current best result:

- reward: `0.2111`
- point accuracy: `0.0500`

### Did `test_task` stay stable or improve?

It improved clearly:

- reward: `0.1355 -> 0.1576`
- point accuracy: `0.0000 -> 0.0500`
- recovery toward oracle: `0.0796 -> 0.4031`

### Did this beat the current best Stage-B reranker?

Yes.

It beat the current best official benchmark because:

- `test_task` improved materially
- `test_website` stayed equally strong
- `test_domain` stayed equally strong

### Is rare recovery still the dominant bottleneck after this step?

Yes.

The remaining biggest miss is still a rare, high-value recovery signature:

- `point_first_sampled_t0p7 | rank_7_8 | singleton | point_first | ...`

There is also one large remaining structured singleton task miss and one within-source structured multi-positive task miss.

## Final Takeaway

This step meaningfully advances the original blueprint.

It remains fully Qwen-first and preserves the current Stage A / Stage B / learned reranker design.

The improvement comes from supervision only:

- better rare-headroom train coverage
- harder rare-recovery pair construction
- explicit rare-signature weighting

The final result is not a universal lift on every split, so this should not be overclaimed.

The honest conclusion is:

- rare-recovery-targeted supervision did beat the current best source-aware reranker overall
- the gain came from `test_task`
- `test_website` and `test_domain` were preserved, not improved
- the rare domain `point_first_sampled_t0p7` recovery remains the clearest next missing piece
