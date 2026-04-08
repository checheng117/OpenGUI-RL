# Mind2Web Stage-B Rare Point-First Disagreement/Support Supervision

## Objective

Keep the existing Qwen-first Stage A -> Stage B -> learned reranker pipeline intact.

Do not add candidate sources.
Do not expand the candidate pools again.
Do not replace the reranker family.

This step only targets the remaining rare point-first disagreement/support recovery failures.

## Inputs Inspected Before Editing

- `docs/mind2web_stageB_source_aware_reranker_supervision.md`
- `docs/mind2web_stageB_rare_recovery_targeted_supervision.md`
- `outputs/mind2web_stageB_reranker_rare_recovery_targeted/official_split_evaluations.json`
- `outputs/mind2web_stageB_reranker_rare_recovery_targeted/preference_pairs_train.jsonl`
- `outputs/mind2web_stageB_recovery_cases_rare_targeted/recovery_cases_summary.md`
- `outputs/mind2web_stageB_rare_recovery_analysis_targeted/rare_recovery_analysis.md`
- `outputs/mind2web_stageB_candidates_headroom_expanded/*/candidates_*.jsonl`
- `scripts/run_train_reranker.py`
- `src/gui_grounding/training/trainer_reranker.py`

## Rare Point-First Patterns That Still Matter

Saved compact analysis:

- `outputs/mind2web_stageB_point_first_disagreement_support_analysis_current/point_first_disagreement_support_analysis.json`
- `outputs/mind2web_stageB_point_first_disagreement_support_analysis_current/point_first_disagreement_support_analysis.md`

Key findings:

1. The remaining official point-first bottleneck is still the singleton `point_first_sampled_t0p7` recovery on `test_domain`.
   - sample: `mind2web_test_domain_981e4bf0-9049-4eab-b157-105473f53a97`
   - oracle gap: `0.3563`
   - current best reranker still selected `structured_sampled_t0p6`

2. The shared train/test pattern is narrow and real:
   - rank-7 singleton `point_first_sampled_t0p7`
   - three structured decoys present
   - zero bbox overlap to all other candidates
   - large click disagreement against the structured path

3. But isolated point-first geometry alone is not sufficient.
   - isolated point-first candidates across preserved pools: `85`
   - isolated positives: `4`
   - isolated negatives: `81`
   - so a generic point-first prior would be wrong

4. The supervision gap in the current best reranker is concrete.
   - current best pair export only had `9` preferred point-first pairs and `7` rejected point-first pairs
   - there were no explicit point-first-targeted disagreement/support pair types
   - one train point-first recovery pool was still misranked by the learned scorer even after training

5. The remaining oracle gap is not only point-first.
   - `test_task` still has two large structured singleton misses with gaps `0.4435` and `0.4409`
   - those structured within-pool disambiguation misses remain the biggest non-point-first residual

## What Changed

Updated:

- `src/gui_grounding/training/trainer_reranker.py`
- `scripts/run_train_reranker.py`

Added:

- `scripts/analyze_mind2web_stageb_point_first_disagreement_support.py`
- `configs/train/mind2web_stageB_reranker_qwen_rare_point_first_disagreement_support.yaml`
- `configs/train/mind2web_stageB_reranker_qwen_rare_point_first_disagreement_support_v2.yaml`
- `configs/train/mind2web_stageB_reranker_qwen_rare_point_first_disagreement_support_v3.yaml`

### 1. Protected Rare Point-First Train Coverage

The reranker split now supports protected headroom sources.

For the point-first-targeted runs, `point_first_sampled_t0p7` was kept in train so the rare singleton point-first recovery signature was not partially spent on validation again.

### 2. Point-First Disagreement/Support Pair Redesign

The chosen new reranker is:

- `outputs/mind2web_stageB_reranker_rare_point_first_disagreement_support_v2`

It keeps the same pools and the same reranker family, but changes supervision inside rare recovery pools.

Added pair types:

- `point_first_best_vs_structured_disagreement`
- `point_first_support_anchor`
- `best_vs_point_first_false_alarm`

What these do:

- compare the point-first oracle against all structured disagreement decoys in the rare singleton point-first pools
- compare the point-first oracle against hybrid / point-native / point-first-structured anchors from the same pool
- add a small number of counterexample pairs where the best non-point-first candidate must beat an isolated point-first false alarm

This was the critical redesign:

- boost rare point-first recoveries
- but avoid teaching a broad “isolated point-first is good” shortcut

### 3. Weighting Redesign

The chosen run uses explicit, auditable bonuses for:

- rare point-first pools
- disagreement magnitude
- positive reward-signal margin inside the pair
- point-first support-anchor comparisons

Saved supervision summaries:

- `outputs/mind2web_stageB_reranker_rare_point_first_disagreement_support/supervision_summary.md`
- `outputs/mind2web_stageB_reranker_rare_point_first_disagreement_support_v2/supervision_summary.md`

### 4. Feature Probe

A further lightweight feature extension was tested in:

- `outputs/mind2web_stageB_reranker_rare_point_first_disagreement_support_v3`

It added structured-relative support/disagreement features, but it degraded official-split performance and was not selected.

## Official Split Results

Chosen comparison:

- SFT first-choice
- current best rare-recovery reranker:
  - `outputs/mind2web_stageB_reranker_rare_recovery_targeted`
- best new rare point-first disagreement/support reranker:
  - `outputs/mind2web_stageB_reranker_rare_point_first_disagreement_support_v2`

Saved comparison:

- `outputs/mind2web_stageB_rare_point_first_disagreement_support_comparison/official_split_comparison.json`
- `outputs/mind2web_stageB_rare_point_first_disagreement_support_comparison/official_split_comparison.md`
- `outputs/mind2web_stageB_rare_point_first_disagreement_support_comparison/official_split_comparison_full.json`
- `outputs/mind2web_stageB_rare_point_first_disagreement_support_comparison/official_split_comparison_full.md`

### SFT vs Current Best vs New

| Split | SFT reward | SFT point | Current best reward | Current best point | New reward | New point | Parseable | Oracle | Current best recovery | New recovery |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `test_task` | `0.1300` | `0.0000` | `0.1576` | `0.0500` | `0.1355` | `0.0000` | `1.0000` | `0.1986` | `0.4031` | `0.0796` |
| `test_website` | `0.1900` | `0.0000` | `0.2111` | `0.0500` | `0.2111` | `0.0500` | `1.0000` | `0.2117` | `0.9745` | `0.9745` |
| `test_domain` | `0.1158` | `0.0000` | `0.1359` | `0.0526` | `0.1359` | `0.0526` | `1.0000` | `0.1547` | `0.5181` | `0.5181` |

## Recovery-Case Effect

Saved recovery-case summary for the chosen new run:

- `outputs/mind2web_stageB_recovery_cases_rare_point_first_disagreement_support/recovery_cases_summary.json`
- `outputs/mind2web_stageB_recovery_cases_rare_point_first_disagreement_support/recovery_cases_summary.md`

The rare point-first miss did not recover:

- `test_domain` still misses `mind2web_test_domain_981e4bf0-9049-4eab-b157-105473f53a97`

The structured task side also remained underresolved:

- `test_task` recovery-case hit rate fell back to `0.5000`
- both large structured singleton misses remained

## Direct Answer

Did rare point-first-targeted supervision help more than the current rare-recovery supervision?

- No.

Did `test_domain` improve beyond `0.1359 / 0.0526`?

- No. It tied the current best and did not recover the remaining singleton point-first domain miss.

Did `test_website` stay strong?

- Yes. It stayed at the current-best `0.2111 / 0.0500`.

Did `test_task` stay strong?

- No. The best new rerun regressed to `0.1355 / 0.0000`, below the current best `0.1576 / 0.0500`.

Are rare point-first disagreement/support cases still the dominant bottleneck after this step?

- Yes for the remaining domain miss.
- But the largest residual official-split loss is now shared with the unresolved structured singleton task misses.

Does this meaningfully advance the blueprint?

- Analytically yes:
  - the rare point-first signature is now isolated cleanly
  - the false-positive trap is explicit
  - the pair construction now has auditable point-first disagreement/support pair types
- Benchmark-wise no:
  - it did not beat the current best Stage-B reranker

## Saved Artifacts

- point-first analysis, current:
  - `outputs/mind2web_stageB_point_first_disagreement_support_analysis_current/point_first_disagreement_support_analysis.md`
- point-first analysis, chosen new run:
  - `outputs/mind2web_stageB_point_first_disagreement_support_analysis_v2/point_first_disagreement_support_analysis.md`
- chosen new reranker:
  - `outputs/mind2web_stageB_reranker_rare_point_first_disagreement_support_v2/official_split_evaluations.json`
  - `outputs/mind2web_stageB_reranker_rare_point_first_disagreement_support_v2/preference_pairs_train.jsonl`
  - `outputs/mind2web_stageB_reranker_rare_point_first_disagreement_support_v2/supervision_summary.md`
- recovery-case analysis:
  - `outputs/mind2web_stageB_recovery_cases_rare_point_first_disagreement_support/recovery_cases_summary.md`
- direct comparison:
  - `outputs/mind2web_stageB_rare_point_first_disagreement_support_comparison/official_split_comparison_full.md`

## Next Missing Piece

The next most important missing piece is not more pool expansion.

It is a tighter conditional ranking signal for:

- singleton point-first recoveries that disagree sharply with structured decoys
- without collapsing into a generic isolated-point-first preference
- plus a separate within-source structured disambiguation fix for the two remaining large `test_task` structured singleton misses
