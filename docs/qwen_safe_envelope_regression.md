# Qwen Safe Envelope Regression (RTX 3090 24G)

## Task scope and constraints
- Goal: controlled runtime regression only, no method/pipeline redesign.
- Path remains Qwen-first (`backend=qwen_vl`, `backbone=qwen2_5_vl_3b`).
- Single-variable discipline enforced per run.

## Frozen baseline
- Locked baseline config:
  - `configs/train/generate_candidates_qwen2_5_vl_3b_safe_baseline_locked.yaml`
- Frozen copy used by this batch:
  - `outputs/qwen_safe_regression/frozen_safe_baseline.yaml`
- Baseline signature:
  - `backend=qwen_vl|dtype=float16|attention_backend=eager|top_k=2|max_new_tokens=96|num_samples=6`

## Tooling added for controlled study
- Regression runner:
  - `scripts/run_qwen_safe_regression.py`
- Enhanced per-run tracking in exporter:
  - `scripts/run_generate_candidates.py`
  - Added `run_id`, `config_signature`, `peak_gpu_memory`, `last_processed_sample_id`, `last_failed_sample_id`, `final_failure_type`.
- Structured outputs:
  - `outputs/qwen_safe_regression/run_logs/`
  - `outputs/qwen_safe_regression/run_summaries/`
  - `outputs/qwen_safe_regression/regression_table.csv`
  - `outputs/qwen_safe_regression/unsafe_configs.json`
  - `outputs/qwen_safe_regression/safe_envelope_summary.json`

## Commands actually run
```bash
python -m py_compile scripts/run_generate_candidates.py scripts/run_qwen_safe_regression.py
```

```bash
conda run -n gui-grounding-py310 env HF_ENDPOINT=https://hf-mirror.com python scripts/run_qwen_safe_regression.py | tee outputs/logs/qwen_safe_regression_runner.log
```

## Experiment plan executed (A -> B -> C)
- Control baseline:
  - `baseline_fp16_eager_tk2_nt96_ns6`
- Phase A (load only, one variable per run):
  - `max_new_tokens`: 128 -> 160 -> 192
  - `top_k`: 3 -> 4
  - `num_samples`: 8 -> 10
- Phase B (attention backend only; load/dtype fixed):
  - `eager`
  - `sdpa`
- Phase C (dtype only; load/backend fixed):
  - `float16`
  - `bfloat16`

## Results (measured evidence)
- Total runs: 12
- Success: 12
- Failure: 0
- Unsafe configs captured: none (`unsafe_configs.json` is empty)
- Regression table:
  - `outputs/qwen_safe_regression/regression_table.csv`
- Runner log:
  - `outputs/logs/qwen_safe_regression_runner.log`

## Observed stable operating envelope (within tested points)
- Stable attention backend(s):
  - `eager`, `sdpa`
- Stable dtype(s):
  - `float16`, `bfloat16`
- Stable load range observed:
  - `max_new_tokens <= 192`
  - `top_k <= 4`
  - `num_samples <= 10` (serial execution in safe mode)
- Note:
  - This is an observed-stable envelope for tested points, not a theoretical upper bound.

## Unsafe combinations found
- None reproduced in this controlled batch.

## Likely root cause assessment (after this batch)
- In this regression window, no instability was reproduced.
- Therefore current evidence does **not** isolate a single primary trigger among load/backend/dtype.
- Most likely remaining explanation:
  - instability is interaction-sensitive (driver/runtime timing, environment/network jitter, or untested higher-load points), not triggered by the tested envelope itself.

## Recommended default config for future Qwen export
- Keep default runtime conservative even though wider range passed:
  - `torch_dtype: float16`
  - `attn_implementation: eager`
  - `max_new_tokens: 96`
  - `top_k: 2`
  - `max_samples: 6` (or up to 10 when needed)
  - `safe_run.enabled: true`
  - `safe_run.force_serial_qwen: true`
  - `safe_run.incremental_save: true`
- Use:
  - `configs/train/generate_candidates_qwen2_5_vl_3b_safe_baseline_locked.yaml`

## Next exact step
- Expand envelope with one-variable escalation beyond current tested ceiling:
  - `max_new_tokens: 224 -> 256`
  - then `top_k: 5 -> 6`
  - then `num_samples: 12 -> 16`
- Keep all other variables fixed while escalating each axis.
