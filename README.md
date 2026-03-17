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

### Not Yet Implemented

- [ ] Real model loading and inference (Qwen2-VL integration)
- [ ] SFT training loop with real data
- [ ] Model-based candidate generation
- [ ] Learned reranker training
- [ ] DPO / GRPO training with real reward signals
- [ ] Full benchmark evaluation on all splits
- [ ] ScreenSpot-v2 / VisualWebBench adapter with real data

## Future Plan

| Stage | Task | Status |
|-------|------|--------|
| **A** | Supervised fine-tuning on Mind2Web | Scaffold ready |
| **B** | Candidate generation + reward reranking | Scaffold ready |
| **C-1** | Pairwise preference optimization (DPO) | Scaffold ready |
| **C-2** | Lightweight GRPO / contextual bandit | Scaffold ready |
| **Eval** | Cross-website/domain generalization study | Scaffold ready |
| **Demo** | Qualitative demo with real model | Scaffold ready |

## Datasets

| Dataset | Role | Reference |
|---------|------|-----------|
| Multimodal-Mind2Web | Train + eval | Deng et al., NeurIPS 2023 |
| ScreenSpot-v2 | Primary eval | 2025 release |
| VisualWebBench | Supplementary eval | Wang et al., 2024 |
| ScreenSpot-Pro | Optional hard eval | Li et al., 2025 |

See [`docs/dataset_notes.md`](docs/dataset_notes.md) for detailed dataset documentation.

## Disclaimer

This repository is a **scaffold and starter implementation** for a course project. It provides a complete engineering skeleton with functional reward computation, evaluation metrics, and runnable entry points in scaffold mode. **It does not contain trained models, benchmark results, or claims of experimental performance.** All experiment result placeholders in the documentation are explicitly marked as templates to be filled after real experiments are conducted.

## License

MIT

## References

1. Deng et al., "Mind2Web: Towards a Generalist Agent for the Web", NeurIPS 2023
2. Zheng et al., "GPT-4V(ision) is a Generalist Web Agent, if Grounded", 2024
3. Cheng et al., "SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents", 2024
4. ScreenSpot-v2 dataset release, 2025
5. Wang et al., "VisualWebBench", 2024
6. Li et al., "ScreenSpot-Pro", 2025
