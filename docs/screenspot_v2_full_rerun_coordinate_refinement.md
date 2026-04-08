# ScreenSpot-v2 Full Rerun With Coordinate-Frame Refinement

## Scope

- Keep the evaluation path Qwen-first.
- Do not recompute a new baseline method.
- Reuse the saved full held-out baseline as the official `before`.
- Run the full `1272 / 1272` ScreenSpot-v2 test benchmark with the coordinate-frame refinement enabled.
- Keep canonical prediction semantics centered on:
  - `bbox_proposal`
  - `click_point`
  - `action_type`

## Inputs Inspected First

- `docs/screenspot_v2_clean_heldout_eval.md`
- `outputs/screenspot_v2_eval_qwen2_5_vl_3b/evaluation_summary.json`
- `outputs/screenspot_v2_eval_qwen2_5_vl_3b/subgroup_metrics.json`
- `outputs/screenspot_v2_eval_qwen2_5_vl_3b/predictions.jsonl`
- `docs/screenspot_v2_failure_analysis_and_prompt_refinement.md`
- `src/gui_grounding/models/qwen2_vl_grounding.py`
- `scripts/run_eval_screenspot_v2.py`
- `scripts/merge_screenspot_v2_eval_shards.py`
- Existing refinement artifacts:
  - `outputs/screenspot_v2_eval_qwen2_5_vl_3b_coordfix_balanced180/*`
  - `outputs/screenspot_v2_eval_qwen2_5_vl_3b_coordfix_balanced360/*`

## Confirmed Active Refinement Path

Grounded in `src/gui_grounding/models/qwen2_vl_grounding.py`:

1. When `coordinate_frame="model_resized"`, the prompt explicitly instructs Qwen to emit bbox and click coordinates in the resized model-view frame.
2. `_get_model_coordinate_size(...)` computes the Qwen resize target using `smart_resize(...)`.
3. `_parse_prediction(...)` first parses/clamps bbox and click in that resized frame.
4. The parsed bbox and click are then scaled back into the original screenshot frame before `PredictionResult` is returned.

Grounded in `scripts/run_eval_screenspot_v2.py`:

- The active toggle is `--coordinate-frame model_resized`.
- The evaluation record schema written to `predictions.jsonl` remains canonical:
  - `bbox_proposal`
  - `click_point`
  - `action_type`
- The saved `candidate_schema.primary_fields` remain:
  - `bbox_proposal`
  - `click_point`
  - `action_type`

Confirmed from a smoke run:

- Parsed payload still contains model-output keys such as `predicted_bbox` and `predicted_click_point`.
- Saved evaluation records still export canonical fields `bbox_proposal`, `click_point`, and `action_type`.
- Exported `bbox_proposal` / `click_point` are in original screenshot coordinates after refinement scaling.

## Minimal Code Changes For This Step

- Added config support to `scripts/run_eval_screenspot_v2.py` while preserving existing CLI overrides.
- Added named full-rerun config:
  - `configs/eval/screenspot_v2_qwen2_5_vl_3b_model_resized_full.yaml`
- Updated `scripts/merge_screenspot_v2_eval_shards.py` so merged full outputs retain config/runtime metadata from shard runs, including `coordinate_frame: model_resized`.
- Added comparison helper:
  - `scripts/compare_screenspot_v2_runs.py`

## Named Full-Rerun Config

New explicit run config:

- `configs/eval/screenspot_v2_qwen2_5_vl_3b_model_resized_full.yaml`

Key settings:

- dataset source: `lscpku/ScreenSpot-v2`
- split: `test`
- backbone: `qwen2_5_vl_3b`
- coordinate frame: `model_resized`
- device: `cuda`
- `torch_dtype=bfloat16`
- `attn_implementation=sdpa`
- `gpu_memory_utilization=0.7`
- `max_new_tokens=256`
- `temperature=0.2`
- `min_pixels=65536`
- `max_pixels=524288`

## Execution Method

I did not use a single monolithic run for the new official full rerun.

Reason:

- The previously documented full baseline monolithic run had already shown long-run instability (`COMMAND_EXIT_CODE="132"` after `601` predictions).
- Because the goal here was an authoritative full rerun rather than another stability experiment, I used shard-safe execution directly.

Full rerun method actually used:

- `4` contiguous shards of `318` samples each
- merged into one final authoritative output directory

## Commands Actually Run

Smoke validation of the new config path:

```bash
env HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  /home/cc/.conda/envs/gui-grounding-py310/bin/python \
  scripts/run_eval_screenspot_v2.py \
  --config configs/eval/screenspot_v2_qwen2_5_vl_3b_model_resized_full.yaml \
  --end-index 1 \
  --output-dir outputs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized_smoke \
  --log-every 1
```

Full shard runs:

```bash
script -q -c "env HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  /home/cc/.conda/envs/gui-grounding-py310/bin/python \
  scripts/run_eval_screenspot_v2.py \
  --config configs/eval/screenspot_v2_qwen2_5_vl_3b_model_resized_full.yaml \
  --start-index 0 --end-index 318 \
  --output-dir outputs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized_shard_0000_0318 \
  --log-every 60" \
  outputs/logs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized_shard_0000_0318.log
```

```bash
script -q -c "env HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  /home/cc/.conda/envs/gui-grounding-py310/bin/python \
  scripts/run_eval_screenspot_v2.py \
  --config configs/eval/screenspot_v2_qwen2_5_vl_3b_model_resized_full.yaml \
  --start-index 318 --end-index 636 \
  --output-dir outputs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized_shard_0318_0636 \
  --log-every 60" \
  outputs/logs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized_shard_0318_0636.log
```

```bash
script -q -c "env HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  /home/cc/.conda/envs/gui-grounding-py310/bin/python \
  scripts/run_eval_screenspot_v2.py \
  --config configs/eval/screenspot_v2_qwen2_5_vl_3b_model_resized_full.yaml \
  --start-index 636 --end-index 954 \
  --output-dir outputs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized_shard_0636_0954 \
  --log-every 60" \
  outputs/logs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized_shard_0636_0954.log
```

```bash
script -q -c "env HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  /home/cc/.conda/envs/gui-grounding-py310/bin/python \
  scripts/run_eval_screenspot_v2.py \
  --config configs/eval/screenspot_v2_qwen2_5_vl_3b_model_resized_full.yaml \
  --start-index 954 --end-index 1272 \
  --output-dir outputs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized_shard_0954_1272 \
  --log-every 60" \
  outputs/logs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized_shard_0954_1272.log
```

Final merge:

```bash
python scripts/merge_screenspot_v2_eval_shards.py \
  --predictions \
    outputs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized_shard_0000_0318/predictions.jsonl \
    outputs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized_shard_0318_0636/predictions.jsonl \
    outputs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized_shard_0636_0954/predictions.jsonl \
    outputs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized_shard_0954_1272/predictions.jsonl \
  --expected-samples 1272 \
  --output-dir outputs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized \
  --dataset-source lscpku/ScreenSpot-v2 \
  --dataset-split test \
  --model-backbone qwen2_5_vl_3b
```

Direct before/after comparison:

```bash
python scripts/compare_screenspot_v2_runs.py \
  --before-summary outputs/screenspot_v2_eval_qwen2_5_vl_3b/evaluation_summary.json \
  --before-subgroups outputs/screenspot_v2_eval_qwen2_5_vl_3b/subgroup_metrics.json \
  --after-summary outputs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized/evaluation_summary.json \
  --after-subgroups outputs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized/subgroup_metrics.json \
  --output-json outputs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized/comparison_vs_baseline.json \
  --output-md outputs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized/comparison_vs_baseline.md \
  --before-label baseline_full \
  --after-label model_resized_full
```

## Full Benchmark Completion

- Final evaluated samples: `1272 / 1272`
- Final merged prediction lines: `1272`
- Final successful runs: `1272`
- Final failed runs: `0`
- Final merged coordinate frame recorded in summary: `model_resized`
- Full rerun status: complete and authoritative

## Overall Before / After

`Before` = saved official baseline from `outputs/screenspot_v2_eval_qwen2_5_vl_3b/*`  
`After` = full rerun with `coordinate_frame=model_resized`

| Metric | Before | After | Delta |
| --- | ---: | ---: | ---: |
| Evaluated samples | 1272 | 1272 | 0 |
| Point accuracy | 0.0849 | 0.7099 | +0.6250 |
| IoU@0.5 | 0.0102 | 0.1682 | +0.1580 |
| Mean IoU | 0.0196 | 0.2404 | +0.2209 |
| Action-type validity | 0.9992 | 1.0000 | +0.0008 |
| Parseable output rate | 0.9992 | 1.0000 | +0.0008 |
| Valid bbox rate | 0.9992 | 1.0000 | +0.0008 |
| Valid click_point rate | 0.9992 | 1.0000 | +0.0008 |

## Platform Before / After

| Platform | Count | Point Acc Before | Point Acc After | IoU@0.5 Before | IoU@0.5 After | Mean IoU Before | Mean IoU After |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| desktop | 334 | 0.3084 | 0.6976 | 0.0389 | 0.1407 | 0.0738 | 0.2168 |
| mobile | 501 | 0.0060 | 0.7445 | 0.0000 | 0.0898 | 0.0002 | 0.1849 |
| web | 437 | 0.0046 | 0.6796 | 0.0000 | 0.2792 | 0.0004 | 0.3221 |

## Element-Type Before / After

| Element Type | Count | Point Acc Before | Point Acc After | IoU@0.5 Before | IoU@0.5 After | Mean IoU Before | Mean IoU After |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| icon | 554 | 0.0722 | 0.5614 | 0.0108 | 0.1534 | 0.0172 | 0.2075 |
| text | 718 | 0.0947 | 0.8245 | 0.0097 | 0.1797 | 0.0214 | 0.2658 |

## Key Source Splits Before / After

| Source | Count | Point Acc Before | Point Acc After | IoU@0.5 Before | IoU@0.5 After | Mean IoU Before | Mean IoU After |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| windows | 159 | 0.6289 | 0.7547 | 0.0818 | 0.1321 | 0.1492 | 0.2139 |
| macos | 147 | 0.0204 | 0.6939 | 0.0000 | 0.1633 | 0.0062 | 0.2354 |
| ios | 238 | 0.0042 | 0.7353 | 0.0000 | 0.1134 | 0.0002 | 0.2103 |
| android | 211 | 0.0047 | 0.8057 | 0.0000 | 0.0474 | 0.0001 | 0.1505 |
| shop | 245 | 0.0041 | 0.6449 | 0.0000 | 0.2327 | 0.0000 | 0.2876 |
| tool | 120 | 0.0000 | 0.6500 | 0.0000 | 0.3583 | 0.0003 | 0.3752 |
| forum | 79 | 0.0127 | 0.7089 | 0.0000 | 0.2532 | 0.0006 | 0.2994 |
| gitlab | 73 | 0.0137 | 0.6027 | 0.0000 | 0.1644 | 0.0009 | 0.2226 |

## Shard Outcomes

| Shard | Range | Samples | Point Acc | IoU@0.5 | Mean IoU | Failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `0:318` | 318 | 0.7830 | 0.0943 | 0.1951 | 0 |
| 2 | `318:636` | 318 | 0.6824 | 0.1321 | 0.2129 | 0 |
| 3 | `636:954` | 318 | 0.6761 | 0.3050 | 0.3362 | 0 |
| 4 | `954:1272` | 318 | 0.6981 | 0.1415 | 0.2176 | 0 |

## Interpretation

This full rerun supports the same conclusion suggested by the balanced held-out reevaluations, but now on the entire official benchmark:

- The coordinate-frame refinement materially improves full held-out grounding quality.
- The gain is not a small regression-resistant bump; it is a large benchmark-level shift.
- Improvement is broad, not just concentrated on the already-strong `windows` slice.
- The biggest validation upgrade is that `mobile` and `web` move from near-zero point accuracy to strong held-out performance.
- Structured output quality was already strong and remains strong, so the measured gain is truly from grounding quality rather than format cleanup.

## Held-Out Validation Claim

With this full `1272 / 1272` rerun complete, the repo can now make a stronger clean held-out validation claim for the Qwen-first path:

- `Qwen/Qwen2.5-VL-3B-Instruct` with the model-resized coordinate-frame refinement is credibly and materially stronger than the saved official baseline on ScreenSpot-v2 held-out evaluation.

## Final Artifacts

Merged final artifacts:

- `outputs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized/predictions.jsonl`
- `outputs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized/evaluation_summary.json`
- `outputs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized/subgroup_metrics.json`
- `outputs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized/summary_table.md`
- `outputs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized/comparison_vs_baseline.json`
- `outputs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized/comparison_vs_baseline.md`
- `outputs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized/merge_manifest.json`
- `outputs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized/failures.json`

Shard artifacts:

- `outputs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized_shard_0000_0318/*`
- `outputs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized_shard_0318_0636/*`
- `outputs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized_shard_0636_0954/*`
- `outputs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized_shard_0954_1272/*`

Logs:

- `outputs/logs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized_shard_0000_0318.log`
- `outputs/logs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized_shard_0318_0636.log`
- `outputs/logs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized_shard_0636_0954.log`
- `outputs/logs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized_shard_0954_1272.log`
