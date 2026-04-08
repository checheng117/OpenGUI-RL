# Mind2Web Stage B Candidate Diversity / Headroom Expansion

## Objective

This task focused only on Stage B candidate-pool quality for the existing Qwen-first Mind2Web pipeline:

1. measure current headroom on the official Mind2Web split names
2. expand candidate diversity without replacing Stage A or the learned reranker
3. rerun the reranker on the improved pools
4. determine whether better Stage B headroom actually improves reranking, especially on `test_domain`

This was **not** a new model stack or RL-method task.

## Inputs Inspected Before Editing

- current Stage A report:
  - `docs/mind2web_stageA_stageB_sft_and_learned_reranker.md`
- current Stage B config / export:
  - `configs/train/mind2web_stageB_candidates_qwen2_5_vl_3b.yaml`
  - `scripts/run_generate_candidates.py`
- current reranker:
  - `configs/train/mind2web_stageB_reranker_qwen.yaml`
  - `src/gui_grounding/training/trainer_reranker.py`
  - `outputs/mind2web_stageB_reranker/official_split_evaluations.json`
- current Stage B artifacts:
  - `outputs/mind2web_stageB_candidates/train/`
  - `outputs/mind2web_stageB_candidates/test_task/`
  - `outputs/mind2web_stageB_candidates/test_website/`
  - `outputs/mind2web_stageB_candidates/test_domain/`

## What Changed

### Stage B export changes

Updated:

- `scripts/run_generate_candidates.py`
- `src/gui_grounding/models/qwen2_vl_grounding.py`

Added:

- `configs/train/mind2web_stageB_candidates_qwen2_5_vl_3b_headroom_expanded.yaml`

The export path now supports explicit multi-source Qwen candidate generation while preserving the canonical candidate schema:

- `click_point`
- `bbox_proposal`
- `action_type`
- `score` / `confidence`
- `source`
- `provenance`
- `parser_metadata`
- `reward`

New candidate-source support:

- `stagea_first_choice`
- repeated structured sampled variants
- `point_first_structured`
- `point_native_primary`
- `point_first_sampled_t0p7`
- `hybrid_point_structured`

The important design constraint was:

- preserve the old useful structured sampled pool as a subset
- then add new point-first / point-native / hybrid candidates on top

This matters because an earlier expansion attempt increased diversity but accidentally reduced oracle headroom by replacing too much of the old structured sampling. The final export recipe explicitly avoids that failure mode.

### Small reranker-side change

The unchanged reranker training objective (`all_pairs`) did not exploit the larger pools well enough. A narrow follow-up adjustment was therefore used:

- `pair_construction_mode=headroom_hard_negative`

This is still the same lightweight learned reranker. The change only narrows supervision toward the exact decision of interest:

- baseline first-choice
- versus better alternatives in the same pool

Artifacts for both reranker runs were kept:

- unchanged all-pairs reranker:
  - `outputs/mind2web_stageB_reranker_headroom_expanded/`
- headroom-focused reranker:
  - `outputs/mind2web_stageB_reranker_headroom_expanded_hardneg/`

## Final Expanded Stage B Recipe

Final config:

- `configs/train/mind2web_stageB_candidates_qwen2_5_vl_3b_headroom_expanded.yaml`

Final pool shape:

- `top_k = 8`

Per-sample export plan:

1. `stagea_first_choice`
2. `structured_sampled_t0p6` repeated 3 times
3. `point_first_structured`
4. `point_native_primary`
5. `point_first_sampled_t0p7`
6. `hybrid_point_structured`

Why this final recipe:

- it preserves the old Stage B sampled-structured backbone
- it adds explicit point-path diversity
- it adds a hybrid candidate without changing the output contract
- it remains fully auditable through `source` and `provenance`

## Current vs Expanded Headroom

These measurements use the same official split names as the current repo pipeline:

- `train`
- `test_task`
- `test_website`
- `test_domain`

The current exports remain pilot-sized for the official test splits:

- `train`: 118 usable pools
- `test_task`: 20 pools
- `test_website`: 20 pools
- `test_domain`: 19 pools

Saved comparison artifacts:

- `outputs/mind2web_stageB_headroom_expansion/headroom_comparison.json`
- `outputs/mind2web_stageB_headroom_expansion/headroom_comparison.md`

### Pool-quality comparison

| Split | Candidates/sample | Source div/sample | Oracle reward | Oracle point acc | Headroom rate |
|---|---:|---:|---:|---:|---:|
| `train` old -> new | `4.00 -> 8.00` | `2.00 -> 6.00` | `0.1983 -> 0.2084` | `0.0508 -> 0.0678` | `0.0678 -> 0.1271` |
| `test_task` old -> new | `4.00 -> 8.00` | `2.00 -> 6.00` | `0.1738 -> 0.1986` | `0.1000 -> 0.1500` | `0.1500 -> 0.2000` |
| `test_website` old -> new | `4.00 -> 8.00` | `2.00 -> 6.00` | `0.2388 -> 0.2117` | `0.1500 -> 0.0500` | `0.1500 -> 0.0500` |
| `test_domain` old -> new | `4.00 -> 8.00` | `2.00 -> 6.00` | `0.1352 -> 0.1547` | `0.0526 -> 0.1053` | `0.1053 -> 0.1053` |

### Interpretation

What improved:

- `train` headroom improved
- `test_task` headroom improved
- `test_domain` headroom improved substantially
- source-path diversity increased strongly on every split
- point and bbox diversity increased strongly on every split

What did **not** improve:

- `test_website` oracle headroom got worse

So the result is not “more candidates always helps.” It is:

- the new source mix is helpful on `train`, `test_task`, and especially `test_domain`
- but it is not uniformly beneficial across all split regimes

## Reranker Reruns

### 1. Expanded pool + unchanged reranker objective

Command:

```bash
python scripts/run_train_reranker.py \
  --config configs/train/mind2web_stageB_reranker_qwen_headroom_expanded.yaml
```

Result:

- the larger pools did **not** automatically produce better reranking gains

This run actually underused the extra headroom, especially on `test_task` and `test_domain`.

### 2. Expanded pool + headroom-focused pair construction

Command:

```bash
python scripts/run_train_reranker.py \
  --config configs/train/mind2web_stageB_reranker_qwen_headroom_expanded.yaml \
  training.output_dir=outputs/mind2web_stageB_reranker_headroom_expanded_hardneg \
  training.pair_construction_mode=headroom_hard_negative
```

This is the final reranker result used for analysis because it is the smallest adjustment that actually aligns supervision with the new Stage B objective.

Saved artifacts:

- `outputs/mind2web_stageB_reranker_headroom_expanded_hardneg/official_split_evaluations.json`
- `outputs/mind2web_stageB_headroom_expansion/reranker_comparison.json`
- `outputs/mind2web_stageB_headroom_expansion/reranker_comparison.md`

## Final SFT First-Choice vs Reranker Results

Final setting reported below:

- expanded Stage B candidate pools
- lightweight reranker
- `headroom_hard_negative` pair construction

### Official split-name pilot results

| Split | SFT reward | Reranked reward | Delta | Oracle reward | SFT point acc | Reranked point acc | Parseable |
|---|---:|---:|---:|---:|---:|---:|---:|
| `test_task` | 0.1300 | 0.1307 | +0.0007 | 0.1986 | 0.0000 | 0.0000 | 1.0000 |
| `test_website` | 0.1900 | 0.1902 | +0.0002 | 0.2117 | 0.0000 | 0.0000 | 1.0000 |
| `test_domain` | 0.1158 | 0.1345 | +0.0188 | 0.1547 | 0.0000 | 0.0526 | 1.0000 |

### Recovery toward oracle

`test_task`:

- baseline-to-oracle gap: `0.0686`
- reranked-to-oracle gap: `0.0679`

`test_website`:

- baseline-to-oracle gap: `0.0217`
- reranked-to-oracle gap: `0.0215`

`test_domain`:

- baseline-to-oracle gap: `0.0389`
- reranked-to-oracle gap: `0.0202`

### Comparison against the previous pool + reranker

Old pool + old reranker:

- `test_task`: `+0.0230`
- `test_website`: `+0.0150`
- `test_domain`: `+0.0000`

Expanded pool + hard-negative reranker:

- `test_task`: `+0.0007`
- `test_website`: `+0.0002`
- `test_domain`: `+0.0188`

This means:

- the new Stage B headroom helped the hardest split, `test_domain`
- but the improvement did **not** generalize cleanly to `test_task` and `test_website`

## Direct Answers

### Did Stage B candidate diversity/headroom improve?

Yes, partially.

Strong improvement:

- `train`
- `test_task`
- `test_domain`

Regression:

- `test_website`

### Did oracle best-of-k improve?

Yes on:

- `train`
- `test_task`
- `test_domain`

No on:

- `test_website`

### Did the learned reranker benefit from the larger/better headroom?

Not automatically.

With the unchanged reranker objective, the answer was basically no.

With the small `headroom_hard_negative` adjustment, the answer became:

- yes on `test_domain`
- nearly flat on `test_task`
- nearly flat on `test_website`

### Did `test_domain` improve?

Yes.

This is the clearest positive result from this task:

- oracle reward improved: `0.1352 -> 0.1547`
- reranked reward improved: `0.1158 -> 0.1345`
- reranked point accuracy improved: `0.0000 -> 0.0526`

So the hardest split remains hard, but it is no longer completely flat under reranking.

## Blueprint Interpretation

This task does meaningfully advance the original blueprint, but only in a narrow and honest way.

What is now better aligned:

- Stage B is no longer just “more sampled variants from one path”
- candidate pools now include explicit multi-source Qwen variants and a hybrid candidate
- official-split headroom is measured explicitly
- `test_domain` now shows real reranking benefit once supervision is aligned to headroom cases

What is still missing:

- a split-robust candidate recipe that improves headroom without hurting `test_website`
- a reranker that can exploit larger pools consistently across all official split types

So the bottleneck has shifted.

It is no longer only:

- “we do not have enough candidate diversity”

It is now more specifically:

- “we need better source-quality control and reranker calibration so the added diversity stays useful across split regimes”

## Most Important Next Missing Piece

The next most important missing piece is **split-robust source gating / pool construction**, not a new RL stack.

Concretely:

- preserve the new `test_domain` gains
- stop the `test_website` headroom regression
- keep the same canonical candidate schema and the same lightweight reranker framing

The most likely high-value follow-up is:

- candidate-source ablations and gating rules that decide when to keep or suppress point-native / hybrid variants per sample or per split regime

