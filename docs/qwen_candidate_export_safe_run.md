# Qwen Candidate Export Safe-Run Report

## Scope
- Task type: runtime stability hardening (not method redesign)
- Path kept Qwen-first (`backend=qwen_vl`, `backbone=qwen2_5_vl_3b`)
- CLIP-grid was not used as the solution path

## Grounded findings from failing path
- Failing config: `configs/train/generate_candidates_qwen2_5_vl_3b.yaml`
- Failing log: `outputs/logs/qwen_candidate_export_qwen2_5.log`
- Confirmed runtime path before changes:
  - attention backend: `sdpa` (from config and model init)
  - dtype path: `torch_dtype=auto -> torch.bfloat16` on CUDA (from `VLMBackbone._resolve_torch_dtype`)
  - image/token path: Qwen processor + `process_vision_info(...)` + multimodal `generate(...)`
  - execution strategy: per-sample loop, and per-candidate sequential calls for Qwen path
- Confirmed crash location: Qwen2.5-VL vision attention forward in HF `modeling_qwen2_5_vl.py`, raising `RuntimeError: CUDA error: unspecified launch failure`

## Implemented safe-run hardening
- Added dedicated safe-run config:
  - `configs/train/generate_candidates_qwen2_5_vl_3b_safe.yaml`
- Added explicit safe-run controls in export script:
  - `candidate_generation.safe_run.enabled`
  - `force_serial_qwen`, `incremental_save`, `clear_cuda_cache_on_error`
  - `log_cuda_memory_per_sample`, `log_qwen_generate_diagnostics`
  - `cuda_launch_blocking` compatibility switch
- Added per-sample failure isolation:
  - sample-level `try/except` with `sample_id`, traceback, CUDA memory snapshot
  - on failure: optional `torch.cuda.empty_cache()` + `torch.cuda.ipc_collect()`
  - continue to next sample instead of killing full export
- Added incremental persistence:
  - append sample rows and flat rows immediately per sample
  - outputs survive later failures
- Added runtime diagnostics and summary artifacts:
  - `summary_train.json`, `failures_train.json`, `runtime_train.json`
  - includes attempted/success/failed counts and exact runtime settings
- Added safer Qwen backbone controls:
  - conservative dtype/logging surfaced (`resolved_torch_dtype`)
  - attention backend logging
  - optional `min_pixels` / `max_pixels` passthrough to processor
  - per-call image/visual token diagnostics (`image_grid_thw`, `pixel_values_shape`)

## Commands actually run
```bash
python -m py_compile scripts/run_generate_candidates.py src/gui_grounding/models/vlm_backbone.py src/gui_grounding/models/qwen2_vl_grounding.py
```

```bash
conda run -n gui-grounding-py310 env HF_ENDPOINT=https://hf-mirror.com python scripts/run_generate_candidates.py --config configs/train/generate_candidates_qwen2_5_vl_3b_safe.yaml --num-samples 6 | tee outputs/logs/qwen_candidate_export_qwen2_5_safe.log
```

```bash
python -m py_compile scripts/run_generate_candidates.py
```

```bash
conda run -n gui-grounding-py310 env HF_ENDPOINT=https://hf-mirror.com python scripts/run_generate_candidates.py --config configs/train/generate_candidates_qwen2_5_vl_3b_safe.yaml --num-samples 6 | tee outputs/logs/qwen_candidate_export_qwen2_5_safe.log
```

```bash
wc -l outputs/candidate_generation_qwen2_5_vl_3b_safe/candidates_train.jsonl outputs/candidate_generation_qwen2_5_vl_3b_safe/candidates_train_flat.jsonl
```

## Config used for successful run
- `configs/train/generate_candidates_qwen2_5_vl_3b_safe.yaml`
- Key conservative settings:
  - `torch_dtype: float16`
  - `attn_implementation: eager`
  - `gpu_memory_utilization: 0.7`
  - `max_new_tokens: 96`
  - `temperature: 0.2`
  - `min_pixels: 65536`
  - `max_pixels: 524288`
  - `top_k: 2`
  - `safe_run.enabled: true`

## Stable run result (real run evidence)
- Whole run survived: **yes**
- Attempted samples: **6**
- Successful samples: **6**
- Failed samples: **0**
- Dtype used: `torch.float16`
- Attention backend used: `eager`
- Output counts:
  - sample JSONL lines: 6
  - flat JSONL lines: 12

## Artifacts
- Run log: `outputs/logs/qwen_candidate_export_qwen2_5_safe.log`
- Candidates: `outputs/candidate_generation_qwen2_5_vl_3b_safe/candidates_train.jsonl`
- Flat candidates: `outputs/candidate_generation_qwen2_5_vl_3b_safe/candidates_train_flat.jsonl`
- Summary: `outputs/candidate_generation_qwen2_5_vl_3b_safe/summary_train.json`
- Failures: `outputs/candidate_generation_qwen2_5_vl_3b_safe/failures_train.json`
- Runtime details: `outputs/candidate_generation_qwen2_5_vl_3b_safe/runtime_train.json`

## Files changed
- `scripts/run_generate_candidates.py`
- `src/gui_grounding/models/vlm_backbone.py`
- `src/gui_grounding/models/qwen2_vl_grounding.py`
- `configs/train/generate_candidates_qwen2_5_vl_3b_safe.yaml`
- `docs/qwen_candidate_export_safe_run.md`

## Most likely remaining root cause (if original failures reappear)
- Most likely suspect remains CUDA kernel/driver instability in Qwen2.5-VL visual attention path under more aggressive settings (especially `bfloat16 + sdpa + higher generation/throughput load`) on RTX 3090.

## Exact recommended next step
- Keep safe profile as the baseline and run a controlled ablation one variable at a time:
  1) keep `float16 + eager`, raise `num_samples` and `top_k` gradually;
  2) then only switch `attn_implementation` from `eager -> sdpa`;
  3) then only switch dtype `float16 -> bfloat16`;
  4) if crash returns, the first changed variable is the primary suspect.
