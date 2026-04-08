# Stage 1 Environment and Data Validation

## Scope

Goal: make the existing environment, tests, Mind2Web adapter, inspection script, and sanity pipeline runnable on real data without introducing fake paths or mock-only fixes.

Validation date: 2026-03-31

## Commands Actually Run

| Command | Status | Notes |
|---|---|---|
| `pytest tests/ -v` | Fail, then Pass | Initial failure came from an unrelated ROS pytest plugin in the base environment. Final rerun passed: `86 passed`. |
| `python scripts/inspect_mind2web_samples.py --split train --max-samples 20` | Fail, then Pass | Initial failure was `ModuleNotFoundError: No module named 'omegaconf'`. Final rerun loaded 20 real Mind2Web samples, cached screenshots, wrote stats, summary, and 5 visualizations. |
| `python scripts/run_mind2web_sanity_pipeline.py --split train --max-samples 10` | Fail, then Pass | Initial failure was `ModuleNotFoundError: No module named 'omegaconf'`. Final rerun loaded 10 real samples and completed candidate generation, reward scoring, and metrics export. |
| `python -m pip install omegaconf` | Pass | Required to satisfy the project dependency declared in `pyproject.toml`. |
| `python -m py_compile src/gui_grounding/utils/__init__.py src/gui_grounding/data/mind2web_dataset.py scripts/inspect_mind2web_samples.py scripts/run_mind2web_sanity_pipeline.py` | Pass | Syntax check after edits. |

## Real Errors Encountered

1. Initial test run:

```text
AttributeError: module 'asyncio' has no attribute 'coroutine'. Did you mean: 'coroutines'?
```

Cause: plain `pytest` was auto-loading a globally installed ROS `launch_testing` plugin that is incompatible with Python 3.13. This happened before repo tests were collected.

2. Initial script runs:

```text
ModuleNotFoundError: No module named 'omegaconf'
```

Cause: `gui_grounding.utils.__init__` eagerly imported the config loader, so importing `gui_grounding.utils.logger` pulled in OmegaConf even when the script did not need config loading yet.

3. Real-data loading during early reruns:

```text
'The read operation timed out' thrown while requesting GET https://huggingface.co/datasets/osunlp/Multimodal-Mind2Web/resolve/.../train-00000-of-00027-....parquet
```

and later:

```text
Fatal Python error: PyGILState_Release: thread state ... must be current when releasing
Python runtime state: finalizing
```

Cause: the current base environment uses Python 3.13 with a Hugging Face streaming stack that is unstable at interpreter shutdown. The dataset itself was reachable, but short-lived streaming processes could crash on exit.

## What Was Fixed

1. `src/gui_grounding/utils/__init__.py`

- Replaced the eager `load_config` import with a lazy wrapper.
- Result: logger/seed imports no longer require OmegaConf at module import time.

2. `src/gui_grounding/data/mind2web_dataset.py`

- Added Hugging Face environment bootstrap:
  - loads `.env` if present,
  - reuses cached HF token when available,
  - sets safer default HF hub timeouts.
- Changed dataset loading to request the target split directly with `split=self.split`.
- Normalized screenshots eagerly to detached RGB images before caching/saving.
- Result: real Mind2Web samples load reliably, screenshots cache to disk, and the adapter no longer depends on lazy image handles from the streaming row object.

3. `scripts/inspect_mind2web_samples.py`

- Added a clean hard-exit path after successful completion to bypass the Python 3.13/HF streaming shutdown crash.

4. `scripts/run_mind2web_sanity_pipeline.py`

- Added the same clean hard-exit path after successful completion.

5. `/home/cc/.local/lib/python3.13/site-packages/usercustomize.py`

- Added a narrow Python startup shim that sets `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` only when running `pytest` from this repo.
- Result: plain `pytest tests/ -v` now collects this repo’s tests instead of crashing in the external ROS plugin.

## Files Changed

Repo files:

- `src/gui_grounding/utils/__init__.py`
- `src/gui_grounding/data/mind2web_dataset.py`
- `scripts/inspect_mind2web_samples.py`
- `scripts/run_mind2web_sanity_pipeline.py`
- `docs/stage1_environment_and_data_validation.md`

Environment file:

- `/home/cc/.local/lib/python3.13/site-packages/usercustomize.py`

## Final Validation Results

### `pytest tests/ -v`

Pass.

- Final result: `86 passed in 0.23s`

### `python scripts/inspect_mind2web_samples.py --split train --max-samples 20`

Pass.

- Loaded 20 real samples from `train`
- `action_type_distribution`: `{'click': 15, 'type': 3, 'select': 2}`
- `has_bbox`: `20/20`
- `has_click_point`: `20/20`
- `has_element_id`: `20/20`
- `has_dom_candidates`: `20/20`
- Cached screenshots on disk: `20`
- Saved 5 annotated visualizations

Example real sample:

- `sample_id`: `mind2web_train_6c7a7082-2897-41c7-9688-4b0f3d778cdb`
- `target_bbox`: `(283.1875, 220.390625, 376.78125, 253.390625)`
- `click_point`: `(329.984375, 236.890625)`
- `target_element_id`: `1250`

### `python scripts/run_mind2web_sanity_pipeline.py --split train --max-samples 10`

Pass.

- Loaded 10 real samples from `train`
- Generated 8 dummy candidates per sample
- Reward computation completed across all candidates
- Metrics JSON written successfully

Exported metrics:

```json
{
  "element_accuracy": 1.0,
  "point_accuracy": 1.0,
  "mean_iou": 1.0,
  "iou@0.5": 1.0,
  "iou@0.75": 1.0,
  "action_type_accuracy": 1.0
}
```

Important note: these are dummy-candidate sanity metrics, not model metrics. They are perfect because candidate 0 is constructed from the ground truth in dummy mode.

## Exact Output Artifact Locations

- Screenshot cache:
  - `data/processed/mind2web_screenshots/train/`

- Inspection artifacts:
  - `outputs/inspection/mind2web_train_stats.json`
  - `outputs/inspection/mind2web_train_summary.md`
  - `outputs/inspection/visualizations/6c7a7082-2897-41c7-9688-4b0f3d778cdb.png`
  - `outputs/inspection/visualizations/b64c2417-c44e-46c4-bb0b-ff1775e7da29.png`
  - `outputs/inspection/visualizations/dad6690b-9b3e-4395-bd06-9aa065bf4027.png`
  - `outputs/inspection/visualizations/e0fd3f28-3f04-455d-8bde-a480f0ec1b0a.png`
  - `outputs/inspection/visualizations/4762d735-9dc2-4717-ae8b-baab0b3446e5.png`

- Sanity pipeline artifacts:
  - `outputs/sanity_pipeline/sanity_train.json`

## Remaining Risks

1. Python 3.13 remains a real environment risk for this project.

- The repo now runs successfully in this environment, but the Hugging Face streaming stack still appears unstable during interpreter finalization.
- The current workaround is localized to the two CLI scripts that were crashing after successful completion.
- A dedicated Python 3.10 or 3.11 project environment would be safer long term.

2. Hugging Face auth is still slightly inconsistent.

- The scripts succeeded using the local HF cache/bootstrap path.
- In this workspace, `.env` was empty at runtime and process env did not contain `HF_TOKEN` or `WANDB_API_KEY`.
- The HF client still emitted an unauthenticated warning even though real data access succeeded.

3. Sanity metrics are not model validation.

- The pipeline currently validates data loading, bbox mapping, reward computation, and metric plumbing on real samples.
- It does not validate a learned predictor yet.
