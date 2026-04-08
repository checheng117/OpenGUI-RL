# Qwen Runtime Unblock Summary

## Outcome

- HF runtime path is unblocked via mirror endpoint (`HF_ENDPOINT=https://hf-mirror.com`).
- CUDA runtime is now matched for RTX 3090 (`torch 2.5.1+cu124`).
- Real Qwen2.5-VL-3B single-sample inference completed and artifacts were saved.

## Evidence

- CUDA check:
  - `torch.__version__ == 2.5.1+cu124`
  - `torch.cuda.is_available() == True`
  - GPU name: `NVIDIA GeForce RTX 3090`
- Real inference log:
  - `outputs/logs/qwen2_5_vl_3b_unblock.log`
  - includes successful model load on `device=cuda` and completion.
- Artifacts:
  - `outputs/single_inference_qwen_unblock/unblock_qwen2_5_20260401_161151.json`
  - `outputs/single_inference_qwen_unblock/unblock_qwen2_5_20260401_161151.png`

## Minimal changes made

- `scripts/setup_env.sh`: pinned reproducible GPU stack to cu124 wheels.
- `scripts/run_single_inference.py`: added env bootstrap + HF runtime diagnostics.
- `src/gui_grounding/models/vlm_backbone.py`: added HF token fallback and endpoint retry fallback.
- `.env.example`: documented optional mirror endpoint.

## Remaining caveat

- Direct access to `huggingface.co` remains unstable/reset in this environment.
- The repo runtime path is now robust by supporting fallback endpoint behavior without changing the Qwen-first architecture.
