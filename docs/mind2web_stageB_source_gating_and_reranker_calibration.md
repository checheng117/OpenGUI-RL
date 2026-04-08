# Mind2Web Stage-B Source Gating and Reranker Calibration

## Objective

This step stays inside the existing Qwen-first Stage A -> Stage B -> learned reranker pipeline.

The goal is not more candidates. The goal is:

- identify which Stage-B candidate sources help or hurt each official Mind2Web split
- add interpretable source gating
- add lightweight reranker calibration
- rerun the official splits and check whether reranker gains become more robust across `test_task`, `test_website`, and `test_domain`

## Inputs Inspected Before Editing

- `docs/mind2web_stageB_candidate_diversity_headroom_expansion.md`
- `outputs/mind2web_stageB_headroom_expansion/headroom_comparison.md`
- `outputs/mind2web_stageB_headroom_expansion/reranker_comparison.md`
- `outputs/mind2web_stageB_candidates_headroom_expanded/`
- `outputs/mind2web_stageB_reranker_headroom_expanded_hardneg/`

## Code Changes

Updated:

- `scripts/run_generate_candidates.py`
- `scripts/run_train_reranker.py`
- `src/gui_grounding/training/trainer_reranker.py`

Added:

- `scripts/analyze_mind2web_stageb_sources.py`
- `scripts/apply_stageb_source_gating.py`
- `scripts/compare_mind2web_stageb_runs.py`
- `configs/train/mind2web_stageB_candidates_qwen2_5_vl_3b_source_gated.yaml`
- `configs/train/mind2web_stageB_reranker_qwen_source_gated_calibrated.yaml`
- `configs/train/mind2web_stageB_reranker_qwen_source_gated_posthoc_calibrated.yaml`
- `configs/train/mind2web_stageB_reranker_qwen_source_selection_gated_calibrated.yaml`

The changes are intentionally narrow.

- keep the same Qwen-first Stage A -> Stage B -> learned reranker structure
- keep the same Stage-B candidate schema and provenance fields
- add source-quality control instead of adding new model families
- add source-aware calibration features instead of replacing the reranker

## Saved Artifacts

Per-source attribution:

- current expanded pools:
  - `outputs/mind2web_stageB_source_attribution_current/source_attribution_summary.json`
  - `outputs/mind2web_stageB_source_attribution_current/source_attribution_summary.md`
- conservative gated pools:
  - `outputs/mind2web_stageB_source_attribution_posthoc/source_attribution_summary.json`
  - `outputs/mind2web_stageB_source_attribution_posthoc/source_attribution_summary.md`

Gated candidate pools:

- aggressive export-time gating:
  - `outputs/mind2web_stageB_candidates_source_gated/`
- conservative post-hoc gating:
  - `outputs/mind2web_stageB_candidates_source_gated_posthoc/`

Official split reranker runs:

- current expanded-pool baseline:
  - `outputs/mind2web_stageB_reranker_headroom_expanded_hardneg/official_split_evaluations.json`
- aggressive export-time gating + calibrated reranker:
  - `outputs/mind2web_stageB_reranker_source_gated_calibrated/official_split_evaluations.json`
- conservative post-hoc gating + calibrated reranker:
  - `outputs/mind2web_stageB_reranker_source_gated_posthoc_calibrated/official_split_evaluations.json`
- selection-time source masking on original expanded pools:
  - `outputs/mind2web_stageB_reranker_source_selection_gated_calibrated/official_split_evaluations.json`

Comparison tables:

- aggressive gating comparison:
  - `outputs/mind2web_stageB_source_gating_comparison/official_split_comparison.md`
  - `outputs/mind2web_stageB_source_gating_comparison/pool_quality_comparison.md`
- conservative gating comparison:
  - `outputs/mind2web_stageB_source_gating_posthoc_comparison/official_split_comparison.md`
  - `outputs/mind2web_stageB_source_gating_posthoc_comparison/pool_quality_comparison.md`
- final combined table:
  - `outputs/mind2web_stageB_source_gating_final/final_comparison.json`
  - `outputs/mind2web_stageB_source_gating_final/final_comparison.md`

## Per-Source Attribution

Pre-gating attribution from `outputs/mind2web_stageB_source_attribution_current/` shows a stable pattern.

### test_task

- `structured_sampled_t0p6` is the main useful source:
  - oracle contribution rate: `0.95`
  - better-than-first rate: `0.15`
  - source-best gain vs first: `+0.0662`
- `hybrid_point_structured` provides some extra task-specific headroom:
  - better-than-first rate: `0.10`
  - source-best gain vs first: `+0.0177`
- `point_native_primary` has only minor upside:
  - better-than-first rate: `0.05`
- `point_first_structured` is redundant:
  - better-than-first rate: `0.00`
  - near-duplicate-to-first rate: `0.1765`

### test_website

- `structured_sampled_t0p6` is again the only clearly useful source:
  - oracle contribution rate: `0.95`
  - better-than-first rate: `0.15`
  - source-best gain vs first: `+0.0214`
- `hybrid_point_structured` is marginal:
  - better-than-first rate: `0.05`
  - source-best gain vs first: `+0.0003`
- `point_first_sampled_t0p7`, `point_first_structured`, and `point_native_primary` add no measured headroom:
  - better-than-first rate: `0.00` for all three
- `point_first_structured` is especially redundant:
  - near-duplicate-to-first rate: `0.5556`

### test_domain

- `structured_sampled_t0p6` still carries most of the recoverable headroom:
  - oracle contribution rate: `0.9474`
  - better-than-first rate: `0.1053`
  - source-best gain vs first: `+0.0202`
- `point_first_sampled_t0p7` is the only extra point-first source with real value here:
  - better-than-first rate: `0.0526`
  - selected-source reward in the original reranker run: `0.1854`
  - source-best gain vs first: `+0.0188`
- `point_native_primary` is harmful as a selected source:
  - better-than-first rate: `0.0000`
  - reranker-selected rate: `0.3158`
  - selected-source reward: `0.0667`
- `point_first_structured` is also non-contributing:
  - better-than-first rate: `0.0000`
  - selected-source reward: `0.0667`
- `hybrid_point_structured` adds no measured headroom on this split

### Split-by-Split Help vs Hurt

Most helpful:

- `test_task`: `structured_sampled_t0p6`, then `hybrid_point_structured`
- `test_website`: `structured_sampled_t0p6` by a wide margin
- `test_domain`: `structured_sampled_t0p6`, then `point_first_sampled_t0p7`

Most harmful or dilutive:

- `test_task`: `point_first_structured` is redundant; `point_native_primary` adds little relative to its calibration cost
- `test_website`: `point_first_structured`, `point_native_primary`, and `point_first_sampled_t0p7` all behave as noise on the measured pools
- `test_domain`: `point_native_primary` is the clearest harmful source; `point_first_structured` is also noise

## Source Gating Design

Two gating variants were implemented and tested.

### 1. Aggressive export-time gating

Implemented in `scripts/run_generate_candidates.py` and used by `configs/train/mind2web_stageB_candidates_qwen2_5_vl_3b_source_gated.yaml`.

Rules:

- drop `point_first_structured`
- drop standalone `point_native_primary` while still allowing it to donate the click for `hybrid_point_structured`
- require disagreement for:
  - `point_first_sampled_t0p7`
  - `hybrid_point_structured`
- prune near-duplicates
- cap per-source counts
- refill freed slots through the existing structured fallback path

This was interpretable, but too aggressive.

### 2. Conservative post-hoc gating

Implemented in `scripts/apply_stageb_source_gating.py` and used by `outputs/mind2web_stageB_candidates_source_gated_posthoc/`.

Rules:

- remove `point_first_structured`
- remove `point_native_primary`
- leave all other candidates unchanged
- preserve all candidate metadata fields and provenance

This conservative gate preserved the original oracle headroom exactly, so it is the safer gating variant.

## Lightweight Reranker Calibration

The reranker remains the same lightweight MLP scorer with the same pairwise training setup.

Calibration changes are feature-level:

- exact normalized source one-hot, not only coarse source families
- source-local count within the pool
- source-local rank within the pool
- click-distance relative to the first-choice candidate
- bbox IoU relative to the first-choice candidate
- action agreement relative to the first-choice candidate
- DOM-match delta relative to the first-choice candidate
- support-consistency features:
  - same-action support count
  - distinct supporting-source count
  - max same-action bbox overlap
  - min same-action click distance
- optional selection-time source masking for clearly noisy sources

This keeps the reranker lightweight and auditable.

## Pool Quality After Gating

### Aggressive export-time gating

`outputs/mind2web_stageB_source_gating_comparison/pool_quality_comparison.md`

| Split | Avg cand old | Avg cand new | Oracle reward old | Oracle reward new | Oracle point old | Oracle point new |
|---|---:|---:|---:|---:|---:|---:|
| test_task | 8.00 | 8.00 | 0.1986 | 0.1955 | 0.1500 | 0.1500 |
| test_website | 8.00 | 8.00 | 0.2117 | 0.1909 | 0.0500 | 0.0000 |
| test_domain | 8.00 | 8.00 | 0.1547 | 0.1344 | 0.1053 | 0.0556 |

Interpretation:

- this gate improved neither oracle reward nor oracle point accuracy
- it materially damaged `test_website`
- it also damaged `test_domain`

### Conservative post-hoc gating

`outputs/mind2web_stageB_source_gating_posthoc_comparison/pool_quality_comparison.md`

| Split | Avg cand posthoc | Oracle reward posthoc | Oracle point posthoc |
|---|---:|---:|---:|
| test_task | 6.15 | 0.1986 | 0.1500 |
| test_website | 6.15 | 0.2117 | 0.0500 |
| test_domain | 6.16 | 0.1547 | 0.1053 |

Interpretation:

- this gate removed obvious noise while preserving oracle headroom
- it improved auditability and provenance cleanliness
- it did not raise oracle reward beyond the original expanded pools

## Official Split Results

Final combined comparison artifact:

- `outputs/mind2web_stageB_source_gating_final/final_comparison.md`

### Current expanded-pool reranker baseline

This is the confirmed pre-edit reference from `outputs/mind2web_stageB_reranker_headroom_expanded_hardneg/official_split_evaluations.json`.

| Split | SFT reward | Reranker reward | Oracle reward | Gain | Point acc | Parseable | Recovery toward oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| test_task | 0.1300 | 0.1307 | 0.1986 | +0.0007 | 0.0000 | 1.0000 | 0.0102 |
| test_website | 0.1900 | 0.1902 | 0.2117 | +0.0002 | 0.0000 | 1.0000 | 0.0076 |
| test_domain | 0.1158 | 0.1345 | 0.1547 | +0.0188 | 0.0526 | 1.0000 | 0.4819 |

### Aggressive export-time gating + calibrated reranker

From `outputs/mind2web_stageB_reranker_source_gated_calibrated/official_split_evaluations.json`.

| Split | SFT reward | Reranker reward | Oracle reward | Gain | Point acc | Parseable | Recovery toward oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| test_task | 0.1300 | 0.1477 | 0.1955 | +0.0177 | 0.0500 | 1.0000 | 0.2704 |
| test_website | 0.1900 | 0.1902 | 0.1909 | +0.0002 | 0.0000 | 1.0000 | 0.2684 |
| test_domain | 0.1111 | 0.1111 | 0.1344 | +0.0000 | 0.0000 | 1.0000 | 0.0000 |

Notes:

- `test_task` improved materially
- `test_website` did not meaningfully improve; the apparent recovery ratio is inflated because the gate destroyed most of the oracle gap
- `test_domain` lost the earlier gain completely
- this rerun produced `18` valid `test_domain` samples instead of `19`
- the missing sample had flat `0.2` reward across the original pool, so it does not explain the domain regression

### Conservative post-hoc gating + calibrated reranker

From `outputs/mind2web_stageB_reranker_source_gated_posthoc_calibrated/official_split_evaluations.json`.

| Split | SFT reward | Reranker reward | Oracle reward | Gain | Point acc | Parseable | Recovery toward oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| test_task | 0.1300 | 0.1355 | 0.1986 | +0.0055 | 0.0000 | 1.0000 | 0.0796 |
| test_website | 0.1900 | 0.1900 | 0.2117 | +0.0000 | 0.0000 | 1.0000 | 0.0000 |
| test_domain | 0.1158 | 0.1158 | 0.1547 | +0.0000 | 0.0000 | 1.0000 | 0.0000 |

Selection-time source masking on the original expanded pools matched this same result:

- `test_task`: `0.1355`
- `test_website`: `0.1900`
- `test_domain`: `0.1158`

## What Changed

- added per-source attribution analysis by official split
- added interpretable source gating in two forms:
  - aggressive export-time gating
  - conservative post-hoc pruning and selection-time source masking
- added lightweight reranker calibration features that expose source identity, pool-relative rank, first-choice-relative geometry, and source-consistency support
- reran official split evaluation after gating and calibration

## Direct Answers

### 1. Which sources helped vs hurt by split?

Helped most:

- `test_task`: `structured_sampled_t0p6`, then `hybrid_point_structured`
- `test_website`: `structured_sampled_t0p6`
- `test_domain`: `structured_sampled_t0p6`, then `point_first_sampled_t0p7`

Hurt or diluted most:

- `test_task`: `point_first_structured`
- `test_website`: `point_first_structured`, `point_native_primary`, `point_first_sampled_t0p7`
- `test_domain`: `point_native_primary`, then `point_first_structured`

### 2. Did source gating improve effective pool quality?

Not overall.

- aggressive export-time gating made pool quality worse on `test_website` and `test_domain`
- conservative post-hoc gating preserved oracle headroom exactly, but did not raise oracle reward or oracle point accuracy above the original expanded pools

### 3. Did reranker calibration improve split-robust gains?

Not consistently across the three official splits.

- aggressive gating plus calibration improved `test_task`
- conservative gating plus calibration gave a smaller `test_task` gain
- neither gated variant preserved the original `test_domain` gain
- neither gated variant fixed the `test_website` flatness

### 4. Did `test_website` recover?

No.

- the safest gated run remained flat at `0.1900 -> 0.1900`
- the aggressive run stayed effectively flat at `0.1900 -> 0.1902`, but only after damaging oracle headroom from `0.2117` to `0.1909`

### 5. Did `test_domain` stay improved?

No for the new gated variants.

- the original expanded-pool reranker still has the best `test_domain` result: `0.1158 -> 0.1345`, point accuracy `0.0000 -> 0.0526`
- aggressive export-time gating lost that gain
- conservative post-hoc gating also lost that gain

### 6. Does this meaningfully advance the original blueprint?

Yes in diagnosis and control, but not yet in robust end performance.

- the step clearly identified which sources help and which ones hurt by split
- it added auditable source gating and source-aware reranker calibration without leaving the blueprint
- it did not yet deliver robust reranker gains across `test_task`, `test_website`, and `test_domain`

### 7. Is Stage B robust enough now?

No.

The candidate pools are better understood and easier to audit, but split-robust reranker selection is still not solved.

### 8. What is the dominant bottleneck after this step?

The dominant bottleneck is still reranker conversion of preserved headroom into correct selections.

More concretely:

- source diversity is no longer the main limit
- even after removing clearly noisy sources, the reranker still fails to consistently exploit the remaining headroom
- the next highest-signal step is more targeted source-aware reranker supervision on the few genuine recovery cases, not more source families

## Final Takeaway

This task produced a useful Stage-B diagnosis and added interpretable source gating and calibration machinery without redesigning the pipeline.

The strongest source-level conclusion is:

- keep `structured_sampled_t0p6` as the main headroom engine
- keep `point_first_sampled_t0p7` only as a domain-sensitive disagreement source
- treat `point_first_structured` as removable noise
- treat standalone `point_native_primary` as calibration-toxic unless it is used inside `hybrid_point_structured`

The strongest evaluation conclusion is:

- no tested gating + calibration variant produced robust gains across all three official splits
- `test_website` did not recover
- the previous `test_domain` gain was not preserved under the new gated variants

So this step advances the blueprint diagnostically, but the next missing piece is still a better source-aware reranker training signal, not a larger or redesigned Stage-B pool.
