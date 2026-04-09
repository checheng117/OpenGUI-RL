# Cross-Website GUI Grounding with Verifiable Reward Optimization

This repository studies instruction-conditioned GUI grounding across websites and interface styles. The core problem is: given a screenshot and a natural-language instruction, predict the correct UI target and action in a form that can support later verification and selection.

The project focuses on a research question rather than a full browser agent: how far can a strong supervised multimodal grounding baseline go, and when does verifiable reward based reranking still add value once supervision becomes strong?

## Current Status / Main Findings / Next Steps

### Current Status

- The primary benchmark story is complete on **ScreenSpot-v2** and **Mind2Web**.
- The repository now contains a mature **Stage A -> Stage B** pipeline and benchmark evidence for its main scientific claims.
- The next step is no longer core pipeline redesign. The next step is **supplementary benchmark evaluation**.

### Main Findings

1. **Hybrid OCR/DOM candidate augmentation is critical for a strong supervised GUI grounding baseline.**
   Pure screenshot-only Stage A was weak on Mind2Web. Adding screenshot plus OCR/DOM-style candidate anchors produced the first credible strong supervised baseline in this repo.
2. **Reward-based reranking helps when the supervised baseline is weak or imperfect.**
   Earlier Mind2Web Stage B runs showed that candidate generation plus deterministic reward plus learned reranking can recover meaningful oracle headroom.
3. **Once strong hybrid supervision saturates most headroom, reranking gains diminish sharply.**
   Rebuilding Stage B on top of the strong hybrid Stage A baseline shrank official-split headroom from `11/59` pools to `5/59`, and reranking stopped being a robust default improvement.
4. **On ScreenSpot-v2, the current Qwen-first method is already above the reproduced public plain-Qwen baseline under the repo protocol.**
   Reproduced public baseline point accuracy is `0.7563`, the point-native decoupled method reaches `0.7736`, and the dual-path candidate generation plus lightweight verifier reaches `0.7791`.

### Next Steps

- **Next immediate step:** VisualWebBench supplementary benchmark evaluation and transfer analysis.
- **Optional later:** ScreenSpot-Pro hard benchmark evaluation.

## Project Overview

### Problem Statement

The task is single-step GUI grounding:

- input: screenshot, instruction, and optional candidate cues
- output: click point, supporting bounding box, and action type
- objective: generalize across websites and interface layouts while keeping outputs verifiable and auditable

The project is intentionally scoped to the perception-and-selection layer behind GUI agents. It is not a long-horizon browser automation system.

### Research Motivation

Two practical issues motivate this work:

1. Pure supervised grounding can be brittle under layout shifts, dense interfaces, and website changes.
2. Reward-based selection is only useful if the supervised model leaves enough recoverable headroom in the candidate pool.

This repository therefore studies both sides of the problem:

- building a strong supervised grounding baseline
- testing whether verifiable reward optimization still matters once that baseline becomes strong

## Pipeline Overview

### Stage A: Strong Supervised Baseline

Stage A is a Qwen-first supervised grounding pipeline that predicts:

- `click_point`
- `bbox_proposal`
- `action_type`

The repository now contains both:

- a **pure-visual baseline**
- a **hybrid screenshot + OCR/DOM candidate-aware baseline**

The hybrid Stage A path uses compact candidate anchors derived from Mind2Web DOM/OCR-style cues and became the first strong supervised baseline in this repo.

Key implementation path:

- [`scripts/run_train_sft.py`](scripts/run_train_sft.py)
- [`configs/train/mind2web_stageA_qwen2_5_vl_3b_sft_hybrid_candidates.yaml`](configs/train/mind2web_stageA_qwen2_5_vl_3b_sft_hybrid_candidates.yaml)

### Stage B: Candidate Generation, Verifiable Reward, and Reranking

Stage B exports small candidate pools, assigns deterministic verifiable rewards, and trains a lightweight learned reranker.

The core Stage B components are:

- candidate generation from the Stage A model
- deterministic reward from localization and action correctness signals
- learned reranking over compact engineered candidate features

The most important current Stage B result is the rebuild on top of the strong hybrid Stage A baseline: once Stage A becomes strong, candidate-pool headroom shrinks and reranking stops giving reliable overall gains.

Key implementation path:

- [`scripts/run_generate_candidates.py`](scripts/run_generate_candidates.py)
- [`scripts/run_train_reranker.py`](scripts/run_train_reranker.py)
- [`configs/train/mind2web_stageB_candidates_qwen2_5_vl_3b_hybrid_stagea.yaml`](configs/train/mind2web_stageB_candidates_qwen2_5_vl_3b_hybrid_stagea.yaml)
- [`configs/train/mind2web_stageB_reranker_qwen_hybrid_stagea.yaml`](configs/train/mind2web_stageB_reranker_qwen_hybrid_stagea.yaml)

### Benchmarking and Evaluation

The repo currently supports two main benchmark tracks:

- **ScreenSpot-v2** for held-out GUI grounding evaluation under a consistent repo protocol
- **Mind2Web** for supervised grounding, candidate generation, deterministic reward analysis, and reranking

The current benchmark story is complete for these primary tracks. The next benchmark step is supplementary evaluation on **VisualWebBench**.

## Main Experimental Conclusions

### ScreenSpot-v2

ScreenSpot-v2 is the clean held-out benchmark in this repo. The main verified milestones are:

1. same-protocol reproduction of a public plain `Qwen/Qwen2.5-VL-3B-Instruct` baseline
2. coordinate-frame refinement that fixed a major evaluation mismatch
3. point-native decoupled Qwen-first evaluation that exceeded the reproduced public baseline
4. dual-path candidate generation plus lightweight verifier that improved further

Current same-protocol snapshot:

| Method | Point Accuracy | Mean IoU | Notes |
|---|---:|---:|---|
| Reproduced public plain-Qwen baseline | `0.7563` | `0.1327` | same repo protocol baseline |
| Point-native decoupled Qwen-first path | `0.7736` | `0.1912` | strong point-first primary method |
| Dual-path candidate generation + lightweight verifier | `0.7791` | `0.2520` | current strongest ScreenSpot-v2 result in repo |

Key reports:

- [`docs/public_baseline_reproduction_and_same_protocol_comparison.md`](docs/public_baseline_reproduction_and_same_protocol_comparison.md)
- [`docs/blueprint_realignment_point_native_decoupling.md`](docs/blueprint_realignment_point_native_decoupling.md)
- [`docs/dual_path_candidate_generation_and_lightweight_verifier.md`](docs/dual_path_candidate_generation_and_lightweight_verifier.md)

### Mind2Web

Mind2Web is the main blueprint benchmark for the supervised-plus-reward story.

The verified story so far is:

1. the pure-visual Stage A baseline was established and diagnosed
2. the hybrid screenshot + OCR/DOM candidate representation was built
3. the hybrid Stage A baseline became the first credible strong supervised baseline
4. Stage B candidate generation, deterministic reward, and learned reranking were completed
5. Stage B was rebuilt on top of the strong hybrid Stage A baseline
6. the rebuild showed that strong hybrid supervision sharply reduces remaining reranking headroom

Representative Mind2Web findings:

| Finding | Evidence |
|---|---|
| Pure visual Stage A is not enough | internal validation point accuracy `0.0375`, mean IoU `0.0061` |
| Hybrid Stage A is strong | internal validation point accuracy `0.7875`, IoU@0.5 `0.7250`, mean IoU `0.7314` |
| Hybrid augmentation is the dominant Stage A improvement | official subset readouts jump to `0.9500 / 0.8500 / 0.8947` point accuracy on `test_task / test_website / test_domain` |
| Stage B headroom shrinks on strong Stage A | official-split headroom pools drop from `11/59` to `5/59` |
| Reranking no longer adds robust overall gains | rebuilt hybrid reranker helps only narrowly on `test_domain` and lowers average official-split reward overall |

Important scope note:

- the current Stage B official-split comparisons use the repo's established pilot-size official split pools for `test_task`, `test_website`, and `test_domain`
- the conclusion is still clear: once strong hybrid supervision saturates most recoverable mistakes, reward reranking becomes sparse rather than broadly beneficial

Key reports:

- [`docs/mind2web_stageA_hybrid_representation_strong_baseline.md`](docs/mind2web_stageA_hybrid_representation_strong_baseline.md)
- [`docs/mind2web_stageB_source_aware_reranker_supervision.md`](docs/mind2web_stageB_source_aware_reranker_supervision.md)
- [`docs/mind2web_stageB_rebuild_on_hybrid_stageA.md`](docs/mind2web_stageB_rebuild_on_hybrid_stageA.md)

## Current Benchmark Coverage

| Benchmark | Role | Status | Current Scope |
|---|---|---|---|
| **Mind2Web** | Primary | **Completed for main project story** | Stage A pure-visual baseline, Stage A hybrid baseline, Stage B candidate/reward/reranker, rebuild on hybrid Stage A |
| **ScreenSpot-v2** | Primary | **Completed for main project story** | full held-out evaluation, public baseline reproduction, point-native decoupling, dual-path verifier |
| **VisualWebBench** | Supplementary | **Next immediate step** | supplementary benchmark evaluation and transfer analysis |
| **ScreenSpot-Pro** | Supplementary | Optional later | harder follow-up benchmark if time permits |

## Repository Structure

The repository contains both the current main pipeline and a number of historical experiment artifacts. The most important directories are:

```text
.
├── configs/
│   ├── data/                  # dataset configs
│   ├── eval/                  # benchmark evaluation configs
│   ├── train/                 # Stage A / Stage B training configs
│   └── demo/                  # demo config
├── data/
│   ├── raw/                   # raw benchmark data
│   ├── processed/             # cached screenshots / processed assets
│   ├── interim/               # intermediate files
│   └── manifests/             # split manifests
├── docs/                      # experiment reports and project documentation
├── outputs/                   # checkpoints, benchmark outputs, analysis artifacts
├── scripts/                   # main runnable entry points
├── src/gui_grounding/
│   ├── data/                  # dataset adapters and schemas
│   ├── models/                # Qwen adapters and scorers
│   ├── reward/                # verifiable reward and lightweight verifier logic
│   ├── training/              # SFT and reranker training
│   ├── evaluation/            # metrics and evaluators
│   ├── utils/                 # config, IO, logging, visualization
│   └── demo/                  # demo app
└── tests/                     # unit tests
```

## Main Entry Points and Reproduction

### Installation

```bash
conda create -n gui-grounding python=3.10 -y
conda activate gui-grounding
pip install -e ".[dev]"
```

### Data Preparation

```bash
python scripts/prepare_mind2web.py
python scripts/prepare_screenspot_v2.py
```

### ScreenSpot-v2

Run the strongest single-path Qwen-first evaluation:

```bash
python scripts/run_eval_screenspot_v2.py \
  --config configs/eval/screenspot_v2_qwen2_5_vl_3b_point_native_decoupled.yaml
```

Run the dual-path candidate generation plus lightweight verifier:

```bash
python scripts/run_eval_dual_path_verifier.py \
  --config configs/eval/screenspot_v2_qwen2_5_vl_3b_dual_path_verifier.yaml
```

Reference public baseline reproduction:

```bash
python scripts/run_eval_screenspot_v2.py \
  --config configs/eval/screenspot_v2_public_qwen2_5_vl_3b_point_baseline.yaml
```

### Mind2Web Stage A

Train the strong hybrid supervised baseline:

```bash
python scripts/run_train_sft.py \
  --config configs/train/mind2web_stageA_qwen2_5_vl_3b_sft_hybrid_candidates.yaml
```

### Mind2Web Stage B

Export rebuilt Stage B candidate pools on top of the hybrid Stage A baseline:

```bash
python scripts/run_generate_candidates.py \
  --config configs/train/mind2web_stageB_candidates_qwen2_5_vl_3b_hybrid_stagea.yaml
```

Train and evaluate the rebuilt hybrid-stage-A reranker:

```bash
python scripts/run_train_reranker.py \
  --config configs/train/mind2web_stageB_reranker_qwen_hybrid_stagea.yaml
```

### Demo

```bash
python scripts/run_demo.py --config configs/demo/demo.yaml
```

## Key Outputs and Reports

Current best benchmark artifacts:

- ScreenSpot-v2 point-native decoupled:
  - `outputs/screenspot_v2_eval_qwen2_5_vl_3b_point_native_decoupled/`
- ScreenSpot-v2 dual-path verifier:
  - `outputs/screenspot_v2_eval_qwen2_5_vl_3b_dual_path_verifier/`
- Mind2Web hybrid Stage A:
  - `outputs/mind2web_stageA_sft_hybrid_candidates/`
- Mind2Web rebuilt Stage B on hybrid Stage A:
  - `outputs/mind2web_stageB_candidates_hybrid_stagea/`
  - `outputs/mind2web_stageB_reranker_hybrid_stagea/`
  - `outputs/mind2web_stageB_rebuild_hybrid_stagea/`

Useful documentation entry points:

- [`docs/final_project_report.md`](docs/final_project_report.md)
- [`docs/final_artifact_index.md`](docs/final_artifact_index.md)
- [`docs/public_baseline_reproduction_and_same_protocol_comparison.md`](docs/public_baseline_reproduction_and_same_protocol_comparison.md)
- [`docs/dual_path_candidate_generation_and_lightweight_verifier.md`](docs/dual_path_candidate_generation_and_lightweight_verifier.md)
- [`docs/mind2web_stageA_hybrid_representation_strong_baseline.md`](docs/mind2web_stageA_hybrid_representation_strong_baseline.md)
- [`docs/mind2web_stageB_rebuild_on_hybrid_stageA.md`](docs/mind2web_stageB_rebuild_on_hybrid_stageA.md)

## Current Project Status

The main primary-benchmark research story is complete:

- ScreenSpot-v2 held-out evaluation is complete
- Mind2Web Stage A supervised baseline work is complete
- Mind2Web Stage B reward-pipeline work is complete
- the key scientific conclusion about diminishing reranking returns on top of strong hybrid supervision is now established

That means the repository is no longer in a pipeline-construction phase. It is in a **supplementary benchmark and transfer-analysis phase**.

## Roadmap / Next Steps

### Next Immediate Step

**VisualWebBench supplementary benchmark evaluation and transfer analysis**

The immediate goal is to test whether the current conclusions transfer beyond the primary benchmarks:

- how the current Qwen-first grounding stack transfers to a supplementary benchmark
- whether the ScreenSpot-v2 and Mind2Web story remains consistent
- whether the sparse residual value of Stage B becomes more visible on harder or more transfer-heavy settings

### Optional Later Step

**ScreenSpot-Pro hard benchmark**

This is a later follow-up benchmark rather than the next required milestone.

## Notes

- This repository contains historical intermediate experiments and exploratory artifacts. The README reflects the **current verified project story**, not every earlier branch of exploration.
- The current README intentionally avoids overstating Stage B: the evidence now supports a nuanced conclusion, not a blanket claim that reward reranking always improves a strong supervised GUI grounding model.
