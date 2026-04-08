# Cross-Website GUI Grounding with Verifiable Reward Optimization

A multimodal system that takes a **screenshot** and a **natural-language instruction**, locates the correct **UI element**, predicts the **next action**, and improves decision quality using **verifiable rewards** — without relying on pure supervised learning alone.

> **CSC6129 Reinforcement Learning — Course Project**

---

## Motivation

Recent multimodal agents are increasingly expected to interact with real software interfaces. A practical system must map an instruction like *"click the login button"* to a specific UI element on a screenshot and choose the correct action type. This capability — **GUI grounding** — is the core perception layer behind web agents and computer-use assistants.

This project formulates GUI grounding as a **contextual bandit** problem: the context is the screenshot and instruction, the action is the selected UI element and operation, and the reward is computed from element correctness, click-point inclusion, and bounding-box overlap. The project therefore remains genuinely tied to reinforcement learning rather than being a generic VLM fine-tuning exercise.

## Scope

This project focuses on **single-step, instruction-conditioned GUI grounding and action prediction**. It is **not** a complete browser agent or long-horizon web navigation system.

Specifically, this project studies:
1. Supervised baseline for grounding (Stage A)
2. Candidate generation and reward-based reranking (Stage B)
3. Pairwise preference optimization / Lightweight GRPO (Stage C)
4. Cross-website generalization evaluation

## Task Definition

| Component | Description |
|-----------|-------------|
| **Input** | Screenshot image `I`, instruction `u`, optional DOM/OCR cues `c` |
| **Output** | Target element `e`, bounding box `b`, action type `t ∈ {click, type, select, hover}` |
| **Reward** | `r = λ1·element_correct + λ2·IoU(b, b*) + λ3·click_inside_target + λ4·action_type_correct − λ5·invalid_format` |

## Repository Structure

```
.
├── README.md                       # This file
├── pyproject.toml                  # Project metadata and tool configuration
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variable template
│
├── configs/                        # YAML configuration files
│   ├── data/                       #   Dataset configs (Mind2Web, ScreenSpot, etc.)
│   ├── model/                      #   Model configs (Qwen2-VL 2B/7B)
│   ├── train/                      #   Training configs (SFT, reranker, DPO, GRPO)
│   ├── eval/                       #   Evaluation configs
│   └── demo/                       #   Demo configuration
│
├── data/                           # Data directory (gitignored, populated by scripts)
│   ├── raw/                        #   Original downloaded data
│   ├── interim/                    #   Intermediate artifacts
│   ├── processed/                  #   Model-ready data
│   └── manifests/                  #   Split manifests
│
├── docs/                           # Documentation
│   ├── gui_grounding_project_proposal.docx
│   ├── repo_plan.md                #   Architecture decisions and development order
│   ├── system_design.md            #   Pipeline design with diagrams
│   ├── experiment_plan.md          #   Planned experiments and ablation templates
│   └── dataset_notes.md            #   Dataset roles and preparation checklist
│
├── notebooks/                      # Exploratory notebooks
│   └── exploratory_sanity_checks.ipynb
│
├── scripts/                        # Runnable entry points
│   ├── setup_env.sh                #   Environment setup
│   ├── prepare_mind2web.py         #   Data preparation
│   ├── prepare_screenspot_v2.py    #   Data preparation
│   ├── run_train_sft.py            #   Stage A: SFT training
│   ├── run_generate_candidates.py  #   Stage B: Candidate generation
│   ├── run_train_reranker.py       #   Stage B: Reranker training
│   ├── run_eval_grounding.py       #   Evaluation
│   ├── run_eval_transfer.py        #   Cross-website evaluation
│   └── run_demo.py                 #   Gradio demo
│
├── src/gui_grounding/              # Main Python package
│   ├── constants.py                #   Project-wide constants
│   ├── utils/                      #   Logging, config, seeding, I/O, visualization
│   ├── data/                       #   Schemas, dataset adapters, collators
│   ├── models/                     #   VLM backbone, heads, scorer, policy adapter
│   ├── training/                   #   Losses, 4 trainer variants
│   ├── reward/                     #   Verifiable reward, candidate generator
│   ├── evaluation/                 #   Metrics, evaluators, error analysis
│   └── demo/                       #   Gradio application
│
├── tests/                          # Unit tests
│   ├── test_reward.py              #   Reward function tests
│   ├── test_metrics.py             #   Evaluation metrics tests
│   ├── test_config_loading.py      #   Config system tests
│   └── test_dataset_schema.py      #   Data schema tests
│
└── outputs/                        # Training outputs, checkpoints (gitignored)
```

## Quick Start

### 1. Create Environment

```bash
# Clone and enter the repo
cd Cross-Website-GUI-Grounding-with-Verifiable-Reward-Optimization

# Create a conda environment
conda create -n gui-grounding python=3.10 -y
conda activate gui-grounding

# Or use the setup script
bash scripts/setup_env.sh
```

### 2. Install Dependencies

```bash
pip install -e ".[dev]"
# Or: pip install -r requirements.txt
```

### 3. Verify Installation

```bash
# Run tests (no GPU or data required)
pytest tests/ -v

# Smoke test: load a config and run candidate generation + reward scoring
python scripts/run_generate_candidates.py --config configs/train/rerank_reward.yaml

# Smoke test: run evaluation with dummy data
python scripts/run_eval_grounding.py --config configs/eval/grounding_eval.yaml
```

### 4. Load Mind2Web Data and Run Sanity Check

The adapter loads directly from HuggingFace (streaming, no bulk download needed):

```bash
# Inspect real data: load 20 samples, print stats, save visualizations
python scripts/inspect_mind2web_samples.py --split train --max-samples 20

# Run the full sanity pipeline: data → candidates → reward → metrics
python scripts/run_mind2web_sanity_pipeline.py --split train --max-samples 10
```

Screenshots are cached to `data/processed/mind2web_screenshots/` on first load.

### 5. Run the Demo

```bash
python scripts/run_demo.py --config configs/demo/demo.yaml
# Opens at http://localhost:7860
```

## Current Progress and Results

This repository is no longer just a scaffold. The Qwen-first evaluation path is live, and the main comparable ScreenSpot-v2 metric is now tracked under one consistent repo protocol.

### What Has Been Achieved

- Integrated a real Qwen-first inference path for GUI grounding.
- Built a clean ScreenSpot-v2 evaluation runner with canonical `bbox_proposal` / `click_point` / `action_type` outputs.
- Reproduced a plain public `Qwen/Qwen2.5-VL-3B-Instruct` same-protocol baseline inside this repo.
- Fixed the major coordinate-frame mismatch that had previously dominated underperformance.
- Iterated from bbox-heavier structured decoding to point-first decoding, then to a blueprint-aligned point-native decoupled decode path.
- Preserved structured outputs while improving the main comparable metric, `point_accuracy`.

### Same-Protocol ScreenSpot-v2 Snapshot

| Method | Point Acc | Desktop | Web | Mobile | IoU@0.5 | Mean IoU | Action Valid | Parseable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reproduced public Qwen baseline | 0.7563 | 0.7275 | 0.7346 | 0.7944 | 0.0519 | 0.1327 | 0.2980 | 0.9937 |
| Structured refined method | 0.7099 | 0.6976 | 0.6796 | 0.7445 | 0.1682 | 0.2404 | 1.0000 | 1.0000 |
| Best prior point-first refinement | 0.7296 | 0.7305 | 0.6751 | 0.7764 | 0.1635 | 0.2365 | 1.0000 | 1.0000 |
| Later web/mobile hotspot prompt tweak | 0.7264 | 0.7305 | 0.6796 | 0.7645 | 0.1588 | 0.2286 | 0.9992 | 0.9992 |
| **Point-native decoupled structured path** | **0.7736** | **0.7365** | **0.7620** | **0.8084** | **0.0967** | **0.1912** | **1.0000** | **1.0000** |

### Current Best Method

The current best same-protocol result is the blueprint-aligned point-native decoupled path:

- `click_point` is predicted first in a point-native Qwen pass
- `bbox_proposal` and `action_type` are attached in a secondary structured pass
- the output contract remains compatible with later candidate export and reward-based scoring
- same-protocol result is now above the reproduced public plain-Qwen baseline on the main comparable metric

Latest best-run artifacts:

- `outputs/screenspot_v2_eval_qwen2_5_vl_3b_point_native_decoupled/evaluation_summary.json`
- `outputs/screenspot_v2_eval_qwen2_5_vl_3b_point_native_decoupled/subgroup_metrics.json`
- `outputs/screenspot_v2_eval_qwen2_5_vl_3b_point_native_decoupled/comparison_vs_previous_structured.md`
- `outputs/screenspot_v2_eval_qwen2_5_vl_3b_point_native_decoupled/comparison_vs_point_first.md`
- `outputs/screenspot_v2_eval_qwen2_5_vl_3b_point_native_decoupled/comparison_vs_public_baseline.md`

Key writeups:

- [`docs/public_baseline_reproduction_and_same_protocol_comparison.md`](docs/public_baseline_reproduction_and_same_protocol_comparison.md)
- [`docs/point_accuracy_first_refinement_against_public_qwen.md`](docs/point_accuracy_first_refinement_against_public_qwen.md)
- [`docs/web_mobile_same_protocol_refinement_round.md`](docs/web_mobile_same_protocol_refinement_round.md)
- [`docs/blueprint_realignment_point_native_decoupling.md`](docs/blueprint_realignment_point_native_decoupling.md)

## Current Development Stage

### Completed

- [x] Full project scaffold with clean module boundaries
- [x] YAML configuration system (OmegaConf) with all experiment configs
- [x] Canonical data schemas (Pydantic) for grounding samples and candidates
- [x] **Real Mind2Web data pipeline** — HF streaming → `GroundingSample` with verified field mapping
- [x] **Fully functional verifiable reward calculator** (element, IoU, click, action, format penalty)
- [x] **Fully functional evaluation metrics** (element acc, point acc, IoU@k, action acc, reranking gain)
- [x] Candidate generator (dummy + heuristic modes)
- [x] End-to-end sanity pipeline: real data → candidate generation → reward scoring → metrics
- [x] Data inspection script with bbox visualization on real screenshots
- [x] Training loop skeletons for all 4 stages (SFT, Reranker, DPO, GRPO-light)
- [x] Model interface layer (VLM backbone, grounding head, action head, scorer, policy adapter)
- [x] Gradio demo application (scaffold mode)
- [x] Unit tests for reward, metrics, config, data schemas, and Mind2Web adapter
- [x] Error analysis utilities
- [x] Comprehensive documentation
- [x] Real Qwen backbone integration for single-step GUI grounding
- [x] Clean ScreenSpot-v2 held-out evaluation under one repo protocol
- [x] Same-protocol reproduction of the plain public Qwen baseline
- [x] Coordinate-frame repair and same-protocol comparison reporting
- [x] Blueprint-aligned point-native decoupled decode path with structured outputs retained
- [x] Full ScreenSpot-v2 re-evaluation for the new point-native decoupled method

### In Progress / Not Yet Implemented

- [~] Formal top-k candidate generation built around the new point-primary action object
  - Implemented top-k export + diversity/gating utilities: `scripts/run_generate_candidates.py`
  - Still missing a fully unified point-primary action-object candidate API (see scaffold TODOs): `src/gui_grounding/reward/candidate_generator.py`
- [x] Lightweight verifier / reward scorer for candidate selection
  - Dual-path candidate builder + scoring: `src/gui_grounding/reward/lightweight_verifier.py`
  - End-to-end evaluation script (ScreenSpot-v2): `scripts/run_eval_dual_path_verifier.py`
- [ ] Coarse-to-fine local crop refinement for small targets and dense interfaces
- [ ] Text-target vs icon-target specialized decoding / refinement logic
- [ ] Preference optimization or larger training loops on top of a stronger candidate-and-verifier pipeline
- [ ] Broader transfer evaluation on additional benchmarks after the method stack stabilizes

## Near-Term Technical Roadmap

The next goal is **method improvement**, not more bug recovery and not another round of tiny prompt wording changes.

### Near-Term Priorities

1. **Formalize the dual-path decode design**
   - Treat the point-native path as the primary click predictor.
   - Treat the bbox/action path as supporting structure.
   - Export candidates with `click_point_primary`, `bbox_support`, `action_type`, confidence, and provenance-style metadata.

2. **Build top-k candidates + a lightweight verifier**
   - Let the main model propose multiple candidates.
   - Use a reward-aligned verifier or scorer to choose top-1.
   - This matches the original Stage B direction more closely than immediately retraining a larger backbone.

3. **Add coarse-to-fine local refinement**
   - Use the first-stage point to crop or zoom into a local region.
   - Refine small targets, icon-only targets, dense layouts, and bbox quality without pulling the primary click back into bbox-first coupling.

4. **Split text and icon handling more explicitly**
   - Use separate lightweight logic for text targets vs icon/widget targets.
   - Keep this as a method-level distinction rather than another single shared prompt tweak.

5. **Only then move to heavier training**
   - Revisit reranker training, preference optimization, or other larger optimization stages after the candidate-and-verifier path is strong enough.

### What Is De-Prioritized For Now

- More pure prompt micro-tuning as the main strategy
- Jumping straight into large retraining before the candidate/verifier path is in place
- Going back to a heavy RL shell before reward-based candidate selection is stable

## Method Status Summary

The project has already moved through three distinct phases:

1. **Error recovery**
   - repairing coordinate mismatch and rebuilding a fair same-protocol baseline
2. **Decode realignment**
   - moving from bbox-heavier structured decoding toward point-first behavior
3. **Blueprint-aligned method improvement**
   - promoting click point to the primary prediction object while retaining structured outputs

The next stage should therefore focus on:

- candidate-layer improvements
- verifier / reward-based selection
- localized refinement

rather than more bug-fix-style gains.

## Datasets

| Dataset | Role | Reference |
|---------|------|-----------|
| Multimodal-Mind2Web | Train + eval | Deng et al., NeurIPS 2023 |
| ScreenSpot-v2 | Primary eval | 2025 release |
| VisualWebBench | Supplementary eval | Wang et al., 2024 |
| ScreenSpot-Pro | Optional hard eval | Li et al., 2025 |

See [`docs/dataset_notes.md`](docs/dataset_notes.md) for detailed dataset documentation.

## Status Note

This repository now contains real same-protocol evaluation artifacts and documented benchmark results for the Qwen-first grounding path. The current strongest result in this repo is the point-native decoupled structured method on ScreenSpot-v2.

What remains unfinished is not the basic evaluation pipeline, but the next layer of method work:

- candidate generation around the point-primary action object
- verifier / reward-based selection
- stronger secondary localization quality

## License

MIT

## References

1. Deng et al., "Mind2Web: Towards a Generalist Agent for the Web", NeurIPS 2023
2. Zheng et al., "GPT-4V(ision) is a Generalist Web Agent, if Grounded", 2024
3. Cheng et al., "SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents", 2024
4. ScreenSpot-v2 dataset release, 2025
5. Wang et al., "VisualWebBench", 2024
6. Li et al., "ScreenSpot-Pro", 2025
