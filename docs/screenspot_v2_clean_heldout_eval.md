# ScreenSpot-v2 Clean Held-Out Evaluation

## Summary

- Evaluation target: `ScreenSpot-v2`
- Dataset source used: `lscpku/ScreenSpot-v2` on Hugging Face
- Primary model: `Qwen/Qwen2.5-VL-3B-Instruct`
- Candidate semantics kept canonical:
  - `bbox_proposal`
  - `click_point`
  - `action_type`
- Evaluated held-out samples: `1272 / 1272`
- Final evaluation mode: shard-completed merge across three prediction sets

## What Changed

- Added a real Hugging Face-backed `ScreenSpotV2Dataset` adapter in `src/gui_grounding/data/screenspot_dataset.py`.
- Exported the adapter in `src/gui_grounding/data/__init__.py`.
- Updated `configs/data/screenspot_v2.yaml` to reflect the actual HF source and bbox format (`xywh_absolute`).
- Added `scripts/run_eval_screenspot_v2.py` for Qwen-first held-out evaluation with canonical `bbox_proposal` / `click_point` / `action_type` outputs.
- Added cached-local-backbone preference in `src/gui_grounding/models/vlm_backbone.py` so model startup uses the already-downloaded Qwen snapshot instead of depending on live Hub fetches.
- Added `scripts/merge_screenspot_v2_eval_shards.py` to merge shard predictions into a single authoritative held-out summary.
- Added shard support to `scripts/run_eval_screenspot_v2.py` via `--start-index` / `--end-index`.

## Dataset And Runtime

- Dataset source: `lscpku/ScreenSpot-v2`
- Dataset split used: `test`
- Total held-out samples: `1272`
- Platform breakdown:
  - `mobile`: `501`
  - `web`: `437`
  - `desktop`: `334`
- Element-type breakdown:
  - `icon`: `554`
  - `text`: `718`
- Bounding-box source format: absolute `xywh`
- Ground-truth bbox clipping corrections applied at load time: `1`

Runtime used:

- Python env: `/home/cc/.conda/envs/gui-grounding-py310/bin/python`
- Device: `cuda`
- GPU: `NVIDIA GeForce RTX 3090`
- `torch_dtype=bfloat16`
- `attn_implementation=sdpa`
- `gpu_memory_utilization=0.7`
- `max_new_tokens=256`
- `temperature=0.2`
- `min_pixels=65536`
- `max_pixels=524288`
- HF mirror endpoint used for dataset access: `https://hf-mirror.com`
- Qwen model weights loaded from cached local snapshot:
  - `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3`

## Commands Actually Run

Smoke validation:

```bash
env HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  /home/cc/.conda/envs/gui-grounding-py310/bin/python \
  scripts/run_eval_screenspot_v2.py --max-samples 5 --log-every 1
```

Monolithic full-run attempt:

```bash
script -q -c "env HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  /home/cc/.conda/envs/gui-grounding-py310/bin/python \
  scripts/run_eval_screenspot_v2.py --log-every 50" \
  outputs/logs/screenspot_v2_eval_qwen2_5_vl_3b.log
```

Observed monolithic outcome:

- Process reached `601` incremental predictions.
- Transcript ended with `COMMAND_EXIT_CODE="132"`.
- Incremental predictions up to dataset index `600` were preserved and reused.

Boundary resume validation:

```bash
env HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  /home/cc/.conda/envs/gui-grounding-py310/bin/python \
  scripts/run_eval_screenspot_v2.py \
  --start-index 600 --end-index 601 \
  --output-dir outputs/screenspot_v2_eval_qwen2_5_vl_3b_resume_test \
  --log-every 1
```

Shard completion runs:

```bash
script -q -c "env HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  /home/cc/.conda/envs/gui-grounding-py310/bin/python \
  scripts/run_eval_screenspot_v2.py \
  --start-index 601 --end-index 936 \
  --output-dir outputs/screenspot_v2_eval_qwen2_5_vl_3b_shard_0601_0936 \
  --log-every 50" \
  outputs/logs/screenspot_v2_eval_qwen2_5_vl_3b_shard_0601_0936.log
```

```bash
script -q -c "env HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  /home/cc/.conda/envs/gui-grounding-py310/bin/python \
  scripts/run_eval_screenspot_v2.py \
  --start-index 936 --end-index 1272 \
  --output-dir outputs/screenspot_v2_eval_qwen2_5_vl_3b_shard_0936_1272 \
  --log-every 50" \
  outputs/logs/screenspot_v2_eval_qwen2_5_vl_3b_shard_0936_1272.log
```

Final merge:

```bash
python scripts/merge_screenspot_v2_eval_shards.py \
  --predictions \
    outputs/screenspot_v2_eval_qwen2_5_vl_3b_shard_0000_0601/predictions.jsonl \
    outputs/screenspot_v2_eval_qwen2_5_vl_3b_shard_0601_0936/predictions.jsonl \
    outputs/screenspot_v2_eval_qwen2_5_vl_3b_shard_0936_1272/predictions.jsonl \
  --expected-samples 1272 \
  --output-dir outputs/screenspot_v2_eval_qwen2_5_vl_3b
```

## Final Held-Out Metrics

Overall:

| Metric | Value |
| --- | ---: |
| Evaluated samples | 1272 |
| Successful runs | 1272 |
| Failed runs | 0 |
| Point-in-box accuracy | 0.0849 |
| Point-in-box hits | 108 |
| BBox IoU@0.5 accuracy | 0.0102 |
| BBox IoU@0.5 hits | 13 |
| Mean IoU | 0.0196 |
| Action-type validity rate | 0.9992 |
| Parseable structured-output rate | 0.9992 |
| Valid bbox rate | 0.9992 |
| Valid click-point rate | 0.9992 |

Action-type distribution:

- `click`: `1093`
- `type`: `164`
- `select`: `14`
- `invalid_or_missing`: `1`

## Subgroup Metrics

Platform:

| Group | Count | Point Acc | IoU@0.5 | Mean IoU |
| --- | ---: | ---: | ---: | ---: |
| `desktop` | 334 | 0.3084 | 0.0389 | 0.0738 |
| `mobile` | 501 | 0.0060 | 0.0000 | 0.0002 |
| `web` | 437 | 0.0046 | 0.0000 | 0.0004 |

Element type:

| Group | Count | Point Acc | IoU@0.5 | Mean IoU |
| --- | ---: | ---: | ---: | ---: |
| `icon` | 554 | 0.0722 | 0.0108 | 0.0172 |
| `text` | 718 | 0.0947 | 0.0097 | 0.0214 |

Most important data-source breakdown:

| Group | Count | Point Acc | IoU@0.5 | Mean IoU |
| --- | ---: | ---: | ---: | ---: |
| `windows` | 159 | 0.6289 | 0.0818 | 0.1492 |
| `macos` | 147 | 0.0204 | 0.0000 | 0.0062 |
| `ios` | 238 | 0.0042 | 0.0000 | 0.0002 |
| `android` | 211 | 0.0047 | 0.0000 | 0.0001 |
| `shop` | 245 | 0.0041 | 0.0000 | 0.0000 |

## Structured Output Quality

- Sample-level inference exceptions in merged evaluation: `0`
- Non-parseable / invalid structured outputs: `1 / 1272`
- Invalid example:
  - dataset index: `651`
  - sample id: `screenspot_v2_test_00651_web_4af68da5-d7bb-4910-b434-1301097cda1f`
  - platform: `web`
  - data source: `forum`
  - instruction: `subscribe the second book`
  - raw model response: free-form refusal/explanation instead of JSON

Interpretation:

- The Qwen-first path is operationally stable enough to produce nearly fully parseable structured outputs on held-out data.
- Held-out grounding quality is weak overall.
- Performance is heavily concentrated in `desktop` / especially `windows`, while `web` and `mobile` are near-zero on the current prompt/runtime path.

## Output Artifacts

Final merged artifacts:

- `outputs/screenspot_v2_eval_qwen2_5_vl_3b/predictions.jsonl`
- `outputs/screenspot_v2_eval_qwen2_5_vl_3b/evaluation_summary.json`
- `outputs/screenspot_v2_eval_qwen2_5_vl_3b/subgroup_metrics.json`
- `outputs/screenspot_v2_eval_qwen2_5_vl_3b/summary_table.md`
- `outputs/screenspot_v2_eval_qwen2_5_vl_3b/merge_manifest.json`
- `outputs/screenspot_v2_eval_qwen2_5_vl_3b/failures.json`

Shard/runtime artifacts:

- `outputs/logs/screenspot_v2_eval_qwen2_5_vl_3b.log`
- `outputs/logs/screenspot_v2_eval_qwen2_5_vl_3b_shard_0601_0936.log`
- `outputs/logs/screenspot_v2_eval_qwen2_5_vl_3b_shard_0936_1272.log`
- `outputs/screenspot_v2_eval_qwen2_5_vl_3b_shard_0000_0601/predictions.jsonl`
- `outputs/screenspot_v2_eval_qwen2_5_vl_3b_shard_0601_0936/predictions.jsonl`
- `outputs/screenspot_v2_eval_qwen2_5_vl_3b_shard_0936_1272/predictions.jsonl`

## Files Changed

- `src/gui_grounding/data/screenspot_dataset.py`
- `src/gui_grounding/data/__init__.py`
- `configs/data/screenspot_v2.yaml`
- `src/gui_grounding/models/vlm_backbone.py`
- `scripts/run_eval_screenspot_v2.py`
- `scripts/merge_screenspot_v2_eval_shards.py`
- `docs/screenspot_v2_clean_heldout_eval.md`

## Current Blockers Before Presentation-Grade Validation

- Clean held-out execution works, but monolithic long-run stability is not yet perfect in this environment because the single-process full run exited with code `132` after `601` predictions.
- Grounding accuracy is not strong enough to claim broad clean held-out validation.
- The current prompt/path appears to transfer much better to desktop Windows UI than to web/mobile screenshots.

## Exact Recommended Next Step

Run targeted follow-up analysis on the saved predictions, especially:

1. Compare desktop Windows hits against mobile/web misses to isolate the main transfer failure mode.
2. Add presentation-ready breakdown plots from the saved subgroup metrics.
3. Only after that, iterate on the Qwen prompt / coordinate grounding behavior for ScreenSpot-style held-out transfer.

Bottom line:

- `bbox_proposal` / `click_point` / `action_type` are now clearly the primary evaluation semantics.
- The Qwen-first path is validated as a clean held-out evaluation pipeline.
- The current Qwen-first grounding quality is **not** yet credibly validated as strong held-out GUI grounding performance.
