# Qwen Bridge-to-Original-Crash Study (RTX 3090 24G)

## Scope
- Task type: targeted crash-reproduction bridge study.
- No method redesign, no pipeline redesign, no CLIP-grid fallback.
- Qwen-first path preserved throughout.

## Inputs inspected before edits
- Frozen baseline:
  - `configs/train/generate_candidates_qwen2_5_vl_3b_safe_baseline_locked.yaml`
- Existing regression tooling:
  - `scripts/run_qwen_safe_regression.py`
  - `scripts/run_generate_candidates.py`
- Prior study outputs:
  - `docs/qwen_safe_envelope_regression.md`
  - `outputs/qwen_safe_regression/regression_table.csv`
  - `outputs/qwen_safe_regression/safe_envelope_summary.json`
- Original crash evidence:
  - `outputs/logs/qwen_candidate_export_qwen2_5.log`

## Frozen safe baseline status
- Baseline file was preserved unchanged.
- Phase 0 re-verification run succeeded before any bridge escalation.

## Tooling added
- Dedicated bridge runner:
  - `scripts/run_qwen_bridge_repro.py`
- This runner:
  - enforces Phase 0 -> 6 execution order
  - applies one primary variable change per phase
  - writes per-run config/log/summary artifacts
  - records last stable and first failure checkpoints
  - stops escalation on first failure

## Commands actually run
```bash
python -m py_compile scripts/run_qwen_bridge_repro.py scripts/run_generate_candidates.py
```

```bash
conda run -n gui-grounding-py310 env HF_ENDPOINT=https://hf-mirror.com python scripts/run_qwen_bridge_repro.py | tee outputs/logs/qwen_bridge_repro_runner.log
```

## Environment used
- Python: `/home/cc/.conda/envs/gui-grounding-py310/bin/python`
- HF endpoint: `https://hf-mirror.com`
- HF mirror used: yes

## Required bridge order executed
- Phase 0: baseline verify
- Phase 1 (`max_new_tokens` only): 96 -> 128 -> 160 -> 192 -> 224 -> 256
- Phase 2 (`top_k` only): 2 -> 3 -> 4
- Phase 3 (`max_samples` only): 6 -> 8 -> 10
- Phase 4 (`attention backend` only): eager -> sdpa
- Phase 5 (`dtype` only): float16 -> bfloat16
- Phase 6 (final combined target):
  - dtype=bfloat16, attention=sdpa, max_new_tokens=256, top_k=4, max_samples=10

## Tested configs and outcomes
- Full structured table:
  - `outputs/qwen_bridge_repro/bridge_table.csv`
- Run summaries:
  - `outputs/qwen_bridge_repro/run_summaries/`
- Run logs:
  - `outputs/qwen_bridge_repro/run_logs/`

Observed results:
- Total runs: 13
- Successful runs: 13
- Failed runs: 0

## Last stable and first failing records
- Last stable:
  - `outputs/qwen_bridge_repro/last_stable.json`
  - last stable run: `phase6_final_full_repro_target`
  - signature: `backend=qwen_vl|dtype=bfloat16|attention_backend=sdpa|top_k=4|max_new_tokens=256|num_samples=10`
- First failing:
  - `outputs/qwen_bridge_repro/first_failure.json`
  - no failing run observed

## Is original crash reproducible now?
- Result: **No (not reproducible in this bridge study).**
- Final combined target (matching original risky tuple) completed successfully with:
  - exit_code=0
  - attempted_samples=10
  - successful_samples=10
  - failed_samples=0

## Root-cause class interpretation
- Based on this bridge evidence:
  - **currently non-reproducible under current environment/path**
- This does not prove the old crash is impossible; it indicates no reproduction under controlled Phase 0->6 runs on current runtime state.

## Artifacts
- `outputs/qwen_bridge_repro/run_logs/`
- `outputs/qwen_bridge_repro/run_summaries/`
- `outputs/qwen_bridge_repro/bridge_table.csv`
- `outputs/qwen_bridge_repro/first_failure.json`
- `outputs/qwen_bridge_repro/last_stable.json`
- `outputs/qwen_bridge_repro/reproduction_summary.json`
- `outputs/logs/qwen_bridge_repro_runner.log`

## Files changed
- `scripts/run_qwen_bridge_repro.py`
- `docs/qwen_bridge_to_original_crash.md`

## Recommended default runtime config after bridge study
- Keep default as conservative safe baseline despite successful bridge:
  - `configs/train/generate_candidates_qwen2_5_vl_3b_safe_baseline_locked.yaml`
  - `torch_dtype: float16`
  - `attn_implementation: eager`
  - `max_new_tokens: 96`
  - `top_k: 2`
  - `max_samples: 6`
  - safe-run serial + incremental save enabled
