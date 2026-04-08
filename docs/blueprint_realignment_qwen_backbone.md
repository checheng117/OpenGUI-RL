# Blueprint Realignment Report: Qwen Backbone First

## Goal

Realign the repository from the CLIP-grid surrogate-centered path back toward the original blueprint:

1. Qwen-VL real multimodal backbone as primary path
2. Candidate representation moving from grid semantic ID toward bbox/click/action semantic object
3. Stage-A and single inference interfaces pointing to Qwen first
4. Python 3.10-oriented setup and isolation of Python 3.13 shutdown workaround logic

## Environment Used

- OS: Linux
- Shell: bash
- Python environment target: `conda env gui-grounding-py310` with `Python 3.10.20`
- Setup script path: `scripts/setup_env.sh`

## Exact Commands Actually Run

1. Syntax and lint checks
   - `python -m compileall scripts/run_single_inference.py scripts/run_train_sft.py scripts/run_generate_candidates.py src/gui_grounding/models/vlm_backbone.py src/gui_grounding/models/qwen2_vl_grounding.py src/gui_grounding/training/trainer_sft_qwen.py`
   - `python -m compileall scripts/run_single_inference.py`
2. Environment setup (real Py3.10 path)
   - `USE_CONDA=1 bash scripts/setup_env.sh`
3. Single inference dry-run (Qwen primary config)
   - `set -a && source .env && set +a && conda run -n gui-grounding-py310 python scripts/run_single_inference.py --config configs/demo/single_inference.yaml --dry-run`
4. Stage-A dry-run (Qwen primary config)
   - `set -a && source .env && set +a && conda run -n gui-grounding-py310 python scripts/run_train_sft.py --config configs/train/sft_qwen2_5_vl_3b_stagea.yaml --dry-run`
5. Candidate generation dry-run after schema shift
   - `conda run -n gui-grounding-py310 python scripts/run_generate_candidates.py --config configs/train/generate_candidates_clip_grid.yaml --dry-run`
6. Real Qwen single-sample smoke attempts
   - `conda run -n gui-grounding-py310 python -c "from PIL import Image; Image.new('RGB',(1280,720),(240,240,240)).save('outputs/qwen_smoke_input.png')"`
   - `set -a && source .env && set +a && conda run -n gui-grounding-py310 python scripts/run_single_inference.py --config configs/demo/single_inference.yaml --backend qwen2_5_vl_3b --image-path outputs/qwen_smoke_input.png --instruction "Click the search button." --sample-id smoke_qwen2_5 --output-dir outputs/single_inference_qwen_smoke`
   - `set -a && source .env && set +a && conda run -n gui-grounding-py310 python scripts/run_single_inference.py --config configs/demo/single_inference.yaml --backend qwen3_vl_2b --image-path outputs/qwen_smoke_input.png --instruction "Click the search button." --sample-id smoke_qwen3 --output-dir outputs/single_inference_qwen3_smoke`
   - Captured log artifact:
     - `outputs/logs/qwen2_5_vl_3b_smoke.log`

## Primary vs Secondary Backbone

- **Primary path (now configured):** `qwen2_5_vl_3b` => `Qwen/Qwen2.5-VL-3B-Instruct`
- **Secondary compatibility path:** `qwen3_vl_2b` => `Qwen/Qwen3-VL-2B-Instruct`
- 7B class was not introduced as first realignment target.

## What Was Changed

### 1) Python 3.10 setup realignment

- Rewrote `scripts/setup_env.sh` to be Py3.10-first and reproducible:
  - default target env: `gui-grounding-py310`
  - conda/venv path selection
  - explicit Qwen runtime deps install:
    - `transformers`, `accelerate`, `safetensors`, `huggingface_hub`, `pillow`, `torchvision`, `qwen-vl-utils`, `python-dotenv`
- Fixed editable install failure by modernizing build backend:
  - `pyproject.toml`: `setuptools.backends._legacy:_Backend` -> `setuptools.build_meta`

### 2) Real Qwen backbone integration

- `src/gui_grounding/models/vlm_backbone.py`
  - Added explicit registry with identifiers:
    - `qwen2_5_vl_3b`
    - `qwen3_vl_2b`
    - `qwen2_vl_2b_legacy`
  - Integrated real Qwen message/image processing:
    - `AutoProcessor` + `AutoModelForImageTextToText`
    - `qwen_vl_utils.process_vision_info(...)`
  - Added practical options:
    - `attn_implementation`
    - `gpu_memory_utilization`
  - Kept `trust_remote_code=True` for Qwen model/processor compatibility.

- `src/gui_grounding/models/qwen2_vl_grounding.py`
  - Promoted unified class `QwenVLGroundingModel`
  - Default model switched to `qwen2_5_vl_3b`
  - Kept `Qwen2VLGroundingModel` alias for backward compatibility

- `src/gui_grounding/models/__init__.py`
  - Exported `QwenVLGroundingModel` explicitly

### 3) Step2/Step3 path reconnection to Qwen

- `scripts/run_single_inference.py`
  - Backend choices now explicit:
    - `qwen2_5_vl_3b`
    - `qwen3_vl_2b`
    - `clip_grid_legacy`
  - Qwen paths instantiate real model and processor via `QwenVLGroundingModel`
  - Added memory/runtime knobs pass-through (`attn_implementation`, `gpu_memory_utilization`)
  - Kept visualization + structured payload export logic
  - Legacy hard-exit switched to opt-in config key:
    - `legacy_hard_exit_on_success`

- `configs/demo/single_inference.yaml`
  - Default backend now `qwen2_5_vl_3b`
  - `clip_grid` is marked as `clip_grid_legacy`

- `scripts/run_train_sft.py`
  - Stage-A dispatch now Qwen-first:
    - Qwen backends: `qwen2_5_vl_3b` / `qwen3_vl_2b`
    - legacy fallback: `clip_grid_legacy`
  - Added opt-in legacy hard exit via env var:
    - `GUI_GROUNDING_LEGACY_HARD_EXIT=1`

- Added lightweight real Qwen Stage-A training interface:
  - `src/gui_grounding/training/trainer_sft_qwen.py`
  - Includes:
    - multimodal chat-style supervised batch builder
    - one/few-step smoke training loop
    - output summary artifact
  - This provides a real Qwen Stage-A interface even when full-scale SFT is not run in this step.

- Added Qwen-first Stage-A configs:
  - `configs/train/sft_qwen2_5_vl_3b_stagea.yaml` (primary)
  - `configs/train/sft_qwen3_vl_2b_stagea.yaml` (secondary)

### 4) Candidate granularity first move (away from grid semantic object)

- `scripts/run_generate_candidates.py`
  - Candidate export now centers on:
    - `bbox_proposal`
    - `click_point`
    - `action_type`
  - `grid_id` demoted into `legacy_metadata`
  - Added minimal hooks for future element-level alignment:
    - `candidate_semantics: "bbox_proposal"`
    - `ocr_text_hint`
    - `dom_candidates_available`
    - `dom_candidates_count`
  - `pred_element_id` for reward call is no longer fabricated from grid id.

## Python 3.13 Legacy Logic Removed / Isolated

The old hard-exit workaround (`os._exit`) is no longer unconditional:

- `scripts/run_train_sft.py`
- `scripts/run_generate_candidates.py`
- `scripts/run_mind2web_sanity_pipeline.py`
- `scripts/inspect_mind2web_samples.py`
- `scripts/run_train_reranker.py`
- `scripts/run_single_inference.py` (config gate)

Now legacy path is opt-in only:
- env flag: `GUI_GROUNDING_LEGACY_HARD_EXIT=1`
- or config flag where applicable (`legacy_hard_exit_on_success`)

## Did Qwen2.5-VL-3B-Instruct Run?

- **Real load path was executed and reached model loading stage.**
- `run_single_inference.py` logs confirm:
  - backend resolved to `qwen2_5_vl_3b`
  - `VLMBackbone` attempted loading `Qwen/Qwen2.5-VL-3B-Instruct`

## Was Qwen3-VL-2B-Instruct Needed?

- Attempted as secondary compatibility path.
- It hit the same external connectivity blocker before full processor/model fetch completed.

## Current Blockers (Honest Status)

1. **Download/Network blocker (primary blocker)**
   - `httpx.ProxyError: 403 Forbidden` while requesting Hugging Face model files
   - See `outputs/logs/qwen2_5_vl_3b_smoke.log`
2. **Runtime compatibility warning**
   - Installed torch build reports NVIDIA driver too old for CUDA path in this environment
   - This currently forces/leans to CPU path for model load, which is impractical for full Qwen-VL inference speed

## What Now Works

- Py3.10 setup script is realigned and executable (validated with real run).
- Single-inference routing is Qwen-first with explicit model family identifiers.
- Stage-A entrypoint structure is Qwen-first with real Qwen training interface (`trainer_sft_qwen.py`) and configs.
- CLIP-grid path is retained but labeled legacy/surrogate fallback.
- Candidate export semantics now prioritize bbox/click/action, with grid id demoted.

## What Is Still Blocked

- End-to-end successful Qwen generation output and saved prediction visualization artifact from real Qwen inference are blocked by HF download/proxy failure.
- Full Qwen Stage-A training run is not completed in this step due the same upstream model-access/runtime constraints.

## Files Changed (for this realignment step)

- `scripts/setup_env.sh`
- `pyproject.toml`
- `requirements.txt`
- `.gitignore`
- `src/gui_grounding/models/vlm_backbone.py`
- `src/gui_grounding/models/qwen2_vl_grounding.py`
- `src/gui_grounding/models/__init__.py`
- `scripts/run_single_inference.py`
- `configs/demo/single_inference.yaml`
- `scripts/run_train_sft.py`
- `src/gui_grounding/training/trainer_sft_qwen.py` (new)
- `configs/train/sft_qwen2_5_vl_3b_stagea.yaml` (new)
- `configs/train/sft_qwen3_vl_2b_stagea.yaml` (new)
- `configs/model/qwen2_5_vl_3b.yaml` (new)
- `configs/model/qwen3_vl_2b.yaml` (new)
- `configs/train/sft_clip_grid_baseline.yaml`
- `scripts/run_generate_candidates.py`
- `scripts/run_mind2web_sanity_pipeline.py`
- `scripts/inspect_mind2web_samples.py`
- `scripts/run_train_reranker.py`

## Remaining Work for bbox / element-level candidate realignment

1. Replace CLIP grid proposal source with true region/element proposal modules:
   - OCR text regions
   - DOM element bboxes
   - multimodal detector proposals
2. Change Stage-B model scoring input from `(action, grid_id)` to `(action, bbox_proposal, element_metadata)`
3. Add candidate schema versioning for transition (`v1_grid` -> `v2_bbox_element`)
4. Add held-out split strictness checks at data loader/trainer level

## Exact Next Step After This Realignment

1. Fix model fetch path first:
   - resolve Hugging Face proxy/network 403 for model artifacts
2. Pin PyTorch/torchvision to a CUDA build compatible with local driver (or update driver)
3. Re-run:
   - `run_single_inference.py` with `qwen2_5_vl_3b` and save full JSON+PNG artifact
4. Then run `run_train_sft.py` with `configs/train/sft_qwen2_5_vl_3b_stagea.yaml` for >1 step smoke training
5. Start Stage-B replacement of grid proposal generator with bbox/element proposal generator

---

## Runtime Unblock Update (2026-04-01)

This section records the follow-up unblock task focused only on runtime.

### Newly executed commands

1. HF/proxy diagnostics
   - `conda run -n gui-grounding-py310 python -c "import os,json; ..."` (proxy/HF env inspection)
   - `conda run -n gui-grounding-py310 python -c "import httpx; httpx.get('https://huggingface.co')"` (direct HF check)
   - `conda run -n gui-grounding-py310 python -c "import httpx; httpx.get('https://hf-mirror.com')"` (mirror check)
   - `conda run -n gui-grounding-py310 env HF_ENDPOINT=https://hf-mirror.com python -c "from transformers import AutoProcessor; ..."` (processor load check)
2. CUDA/torch diagnostics and fix
   - `nvidia-smi`
   - `conda run -n gui-grounding-py310 python -c "import torch; ..."`
   - `conda run -n gui-grounding-py310 python -m pip install --upgrade --force-reinstall torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124`
3. Real Qwen inference rerun
   - `conda run -n gui-grounding-py310 env HF_ENDPOINT=https://hf-mirror.com python scripts/run_single_inference.py --config configs/demo/single_inference.yaml --backend qwen2_5_vl_3b --image-path outputs/qwen_smoke_input.png --instruction "Click the search button." --sample-id unblock_qwen2_5 --output-dir outputs/single_inference_qwen_unblock`

### Runtime diagnosis result

1. HF direct endpoint status
   - Direct `https://huggingface.co` requests in this environment fail with connection reset.
   - Proxy environment variables were not the active blocker in-process.
2. HF mirror status
   - `https://hf-mirror.com` is reachable and returns HTTP 200.
   - Qwen model metadata and processor loading work via `HF_ENDPOINT=https://hf-mirror.com`.
3. CUDA stack status before fix
   - Torch: `2.11.0+cu130`
   - Warning: local driver seen as too old for that CUDA build; `torch.cuda.is_available() == False`.
4. CUDA stack status after fix
   - Torch: `2.5.1+cu124`
   - `torch.cuda.is_available() == True`
   - GPU detected: `NVIDIA GeForce RTX 3090`

### Code/path updates in unblock task

- `scripts/setup_env.sh`
  - now installs a reproducible `cu124` PyTorch stack (`torch==2.5.1`, `torchvision==0.20.1`, `torchaudio==2.5.1`) when GPU runtime is present.
- `scripts/run_single_inference.py`
  - added explicit `.env` bootstrap for this entrypoint.
  - added HF runtime diagnostics logging.
- `src/gui_grounding/models/vlm_backbone.py`
  - added HF token fallback from local hub cache.
  - added retry logic: on network/proxy-style failure, retry once with fallback endpoint (`GUI_GROUNDING_HF_FALLBACK_ENDPOINT`, default `https://hf-mirror.com`).
- `.env.example`
  - documented optional `HF_ENDPOINT=https://hf-mirror.com`.

### End-to-end run result (primary model)

- Primary model used: `Qwen/Qwen2.5-VL-3B-Instruct`
- Real loading completed on GPU:
  - log confirms `device=cuda dtype=torch.bfloat16`
- Real single-sample inference completed successfully.
- Artifacts generated:
  - JSON: `outputs/single_inference_qwen_unblock/unblock_qwen2_5_20260401_161151.json`
  - PNG: `outputs/single_inference_qwen_unblock/unblock_qwen2_5_20260401_161151.png`
  - Log: `outputs/logs/qwen2_5_vl_3b_unblock.log`

### Current status after unblock

- HF access: **fixed for runtime via mirror endpoint path**; direct huggingface.co remains network-restricted in this environment.
- CUDA compatibility: **fixed** for RTX 3090 in `gui-grounding-py310`.
- Qwen2.5-VL-3B execution: **successfully ran end-to-end once** with saved JSON+PNG outputs.
