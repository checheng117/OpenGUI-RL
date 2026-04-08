# Mind2Web Stage-B Source-Aware Reranker Supervision

## Objective

This step keeps the existing Qwen-first Stage A -> Stage B -> learned reranker pipeline intact.

The goal is not more candidates and not more source families.

The goal is:

- identify the genuine recovery pools where `oracle > first-choice`
- redesign reranker supervision so it learns more from those pools
- keep the current expanded-pool oracle headroom unchanged
- improve reranker robustness across `test_task`, `test_website`, and `test_domain`

## Inputs Inspected Before Editing

- `docs/mind2web_stageB_candidate_diversity_headroom_expansion.md`
- `docs/mind2web_stageB_source_gating_and_reranker_calibration.md`
- `outputs/mind2web_stageB_source_attribution_current/source_attribution_summary.md`
- `outputs/mind2web_stageB_reranker_headroom_expanded_hardneg/official_split_evaluations.json`
- `outputs/mind2web_stageB_reranker_headroom_expanded_hardneg/preference_pairs_train.jsonl`
- `outputs/mind2web_stageB_reranker_headroom_expanded_hardneg/training_history.json`
- `scripts/run_train_reranker.py`
- `src/gui_grounding/training/trainer_reranker.py`

## What Changed

Updated:

- `src/gui_grounding/training/trainer_reranker.py`
- `scripts/run_train_reranker.py`

Added:

- `scripts/analyze_mind2web_stageb_recovery_cases.py`
- `configs/train/mind2web_stageB_reranker_qwen_source_aware_supervision.yaml`

Saved new artifacts:

- recovery-case analysis on the current best baseline:
  - `outputs/mind2web_stageB_recovery_cases_current/recovery_cases_summary.json`
  - `outputs/mind2web_stageB_recovery_cases_current/recovery_cases_summary.md`
- recovery-case analysis on the redesigned reranker:
  - `outputs/mind2web_stageB_recovery_cases_source_aware/recovery_cases_summary.json`
  - `outputs/mind2web_stageB_recovery_cases_source_aware/recovery_cases_summary.md`
- new supervision summary:
  - `outputs/mind2web_stageB_reranker_source_aware_supervision/supervision_summary.json`
  - `outputs/mind2web_stageB_reranker_source_aware_supervision/supervision_summary.md`
- official split evaluation for the redesigned reranker:
  - `outputs/mind2web_stageB_reranker_source_aware_supervision/official_split_evaluations.json`
- direct comparison table:
  - `outputs/mind2web_stageB_source_aware_supervision_comparison/official_split_comparison.json`
  - `outputs/mind2web_stageB_source_aware_supervision_comparison/official_split_comparison.md`

## Genuine Recovery Cases

Current recovery-case summary:

- `outputs/mind2web_stageB_recovery_cases_current/recovery_cases_summary.md`

Key facts from the unchanged expanded pools:

- `train` has only `16 / 118` genuine headroom pools
- `test_task` has `4 / 20`
- `test_website` has `4 / 20`
- `test_domain` has `3 / 19`

This confirms the core problem: the reranker is trained on a very small number of real recovery situations.

### Source patterns inside recovery pools

Train oracle-best sources:

- `structured_sampled_t0p6`: `7`
- `hybrid_point_structured`: `5`
- `point_first_sampled_t0p7`: `3`
- `point_native_primary`: `1`

Official split oracle-best sources:

- `test_task`: `structured_sampled_t0p6` dominates, with one `hybrid_point_structured` recovery
- `test_website`: `structured_sampled_t0p6` dominates, with one tiny `hybrid_point_structured` recovery
- `test_domain`: two `structured_sampled_t0p6` recoveries and one `point_first_sampled_t0p7` recovery

### Why the old supervision was too weak

The current best pre-edit reranker used `headroom_hard_negative` supervision and exported only:

- `17` train pairs
- `2` eval pairs

That is too little signal for the ranking problem actually faced at inference time.

The old trainer mainly learned:

- better candidate vs first-choice

It barely learned:

- best recovery candidate vs weaker recovery candidate
- best recovery candidate vs strong same-pool decoys from other sources
- source-aware disagreements inside the small set of recovery pools

### Current reranker miss pattern

Old reranker headroom recovery rate:

- `test_task`: `0.2500`
- `test_website`: `0.2500`
- `test_domain`: `0.3333`

Important misses from the old best reranker:

- `test_task` missed two very large `structured_sampled_t0p6` recoveries with gaps `0.4435` and `0.4409`
- `test_website` missed the largest website recovery, a `structured_sampled_t0p6` candidate with gap `0.4220`
- `test_domain` missed one large `structured_sampled_t0p6` recovery with gap `0.3824`

This is the key justification for supervision redesign.

## Pair Construction Redesign

The new supervision stays on the same expanded pools.

No candidate-source count was increased and oracle headroom was preserved exactly.

New pair construction mode:

- `recovery_source_aware`

It builds three auditable pair types inside headroom pools:

1. `recovery_vs_first_choice`
   - every candidate that beats first-choice is paired against first-choice
2. `best_vs_other_positive`
   - when multiple candidates beat first-choice, the best recovery candidate is paired against weaker positive candidates
3. `best_vs_source_decoy`
   - the best recovery candidate is paired against the strongest non-positive decoy from each competing source

This changed the train signal from:

- old best reranker: `17` train pairs

to:

- redesigned reranker: `62` train pairs on the same train split

The new pair mix from `outputs/mind2web_stageB_reranker_source_aware_supervision/supervision_summary.md` is:

- `recovery_vs_first_choice`: `18`
- `best_vs_source_decoy`: `39`
- `best_vs_other_positive`: `5`

This is the main supervision change.

## Calibration and Weighting Redesign

The reranker model family is unchanged.

The changes are in weighting and model selection:

- `pair_weight_mode=source_aware_recovery`
- `checkpoint_selection_mode=headroom_then_full`

Pair weighting now emphasizes:

- larger reward gaps
- larger pool-level oracle gaps
- recovery-anchor pairs against first-choice
- cross-source disagreements
- best-vs-weaker-positive calibration pairs

The weighting stays lightweight and explicit:

- reward-gap scaling
- pair-type multipliers
- cross-source bonus
- pool-gap bonus

Checkpoint selection also changed.

Instead of choosing the checkpoint only by full-pool validation gain, the trainer now selects lexicographically by:

1. `headroom_subset_reward_gain`
2. `headroom_subset_reranked_best_recovery_rate`
3. `full_pool_reward_gain`

This matters because the reranker objective should be tuned on the exact subset where recovery is possible.

## Official Split Results

Current best pre-edit baseline:

- `outputs/mind2web_stageB_reranker_headroom_expanded_hardneg/official_split_evaluations.json`

New source-aware supervision run:

- `outputs/mind2web_stageB_reranker_source_aware_supervision/official_split_evaluations.json`

Direct comparison table:

- `outputs/mind2web_stageB_source_aware_supervision_comparison/official_split_comparison.md`

### SFT First-Choice vs Reranker

| Split | SFT reward | Current best reranker | Source-aware reranker | Oracle reward | Current best gain | Source-aware gain | Current best point acc | Source-aware point acc | Parseable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `test_task` | `0.1300` | `0.1307` | `0.1355` | `0.1986` | `+0.0007` | `+0.0055` | `0.0000` | `0.0000` | `1.0000` |
| `test_website` | `0.1900` | `0.1902` | `0.2111` | `0.2117` | `+0.0002` | `+0.0211` | `0.0000` | `0.0500` | `1.0000` |
| `test_domain` | `0.1158` | `0.1345` | `0.1359` | `0.1547` | `+0.0188` | `+0.0202` | `0.0526` | `0.0526` | `1.0000` |

### Recovery Toward Oracle

| Split | Current best recovery | Source-aware recovery |
|---|---:|---:|
| `test_task` | `0.0102` | `0.0796` |
| `test_website` | `0.0076` | `0.9745` |
| `test_domain` | `0.4819` | `0.5181` |

### Recovery-Case Hit Rate

Using the saved recovery-case analysis:

| Split | Old reranker recovery-case rate | New reranker recovery-case rate |
|---|---:|---:|
| `test_task` | `0.2500` | `0.5000` |
| `test_website` | `0.2500` | `0.2500` |
| `test_domain` | `0.3333` | `0.6667` |

Important note:

- `test_website` did not improve because it recovered more cases
- it improved because it recovered the largest website recovery case instead of a tiny one

## Comparison Against Gating Runs

From `outputs/mind2web_stageB_source_aware_supervision_comparison/official_split_comparison.md`:

- aggressive gating improved `test_task` but damaged oracle quality and lost `test_domain`
- conservative gating preserved oracle headroom but stayed flat on `test_website` and `test_domain`
- source-aware supervision improved end reranking on all three official splits while keeping the same preserved oracle headroom

This is the key result of the task.

## Direct Answers

### Did source-aware supervision help more than source gating?

Yes.

Source gating either:

- damaged oracle headroom, or
- preserved headroom but stayed flat

The source-aware supervision redesign improved actual reranking on all three official splits without shrinking the pools.

### Did `test_domain` keep or improve its gain?

Yes.

- current best pre-edit: `0.1158 -> 0.1345`
- source-aware supervision: `0.1158 -> 0.1359`

Point accuracy stayed at `0.0526`.

### Did `test_task` and/or `test_website` finally improve meaningfully?

Yes.

`test_task` improved modestly but clearly:

- `0.1300 -> 0.1355`

`test_website` improved substantially:

- `0.1900 -> 0.2111`
- point accuracy `0.0000 -> 0.0500`

### Did this beat the current best pre-edit expanded-pool reranker?

Yes.

It beat the current best on all three official splits:

- `test_task`: `0.1307 -> 0.1355`
- `test_website`: `0.1902 -> 0.2111`
- `test_domain`: `0.1345 -> 0.1359`

### Is reranker supervision now good enough?

Improved clearly, but not fully solved.

The reranker still misses:

- two large `structured_sampled_t0p6` task recoveries
- the rare `point_first_sampled_t0p7` domain recovery
- several tiny website headroom cases

So supervision redesign helped materially, but there is still a remaining bottleneck.

### What is the dominant bottleneck after this step?

The next dominant bottleneck is low-shot recovery generalization across rare recovery patterns.

Concretely:

- train still has only `16` genuine headroom pools
- rare but important sources like `point_first_sampled_t0p7` still have very little supervision
- the reranker now exploits the common structured-sampled recovery pattern much better than before
- it still does not generalize reliably to every rare recovery source/path pattern

## Final Takeaway

This step meaningfully advances the blueprint.

It stays entirely inside the current Qwen-first Stage A / Stage B / learned reranker pipeline and moves the main improvement to supervision quality instead of candidate-count changes.

The strongest conclusion is:

- better recovery-focused supervision helped more than source gating
- preserving headroom and learning better from the few true recovery pools was the right move
- the current best reranker benchmark is now beaten on `test_task`, `test_website`, and `test_domain`

The remaining bottleneck is not more sources.

It is still sparse supervision on rare recovery patterns, especially rare domain-specific recoveries such as `point_first_sampled_t0p7`.
