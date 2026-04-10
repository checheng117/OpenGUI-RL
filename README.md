# Cross-Website GUI Grounding with Verifiable Reward Optimization

This repository packages a completed final-project study of single-step GUI grounding: given a screenshot and a natural-language instruction, predict the UI target and action in a format that can be verified and, if needed, reranked. The project is intentionally scoped to the perception-and-selection layer behind GUI agents rather than full browser automation.

## Final Thesis

- Hybrid screenshot + OCR/DOM candidate augmentation is critical when the benchmark exposes semantically meaningful candidate structure.
- Point-first grounding is a strong transferable inference strategy on held-out GUI benchmarks.
- Reward-based reranking helps when candidate-pool headroom is real, but its value shrinks sharply once Stage A supervision becomes strong.

## Exact Results Snapshot

All numbers in this section were recomputed from saved repository artifacts with:

```bash
python scripts/run_quantitative_metrics_suite.py
```

The canonical machine-readable summary is:

- [`outputs/quantitative_metrics_suite/summary.json`](outputs/quantitative_metrics_suite/summary.json)
- [`outputs/quantitative_metrics_suite/summary.md`](outputs/quantitative_metrics_suite/summary.md)
- [`docs/standalone_quantitative_evaluation_section.md`](docs/standalone_quantitative_evaluation_section.md)

### Mind2Web: Strong Supervised Baseline

Current strongest Mind2Web Stage A result is the hybrid screenshot + OCR/DOM candidate-aware model.

| Split | Element Acc | Click-Point Acc | IoU@0.5 | Action-Type Acc | Invalid Format |
| --- | ---: | ---: | ---: | ---: | ---: |
| `test_task` | `95.00%` | `95.00%` | `95.00%` | `90.00%` | `0.00%` |
| `test_website` | `80.00%` | `85.00%` | `85.00%` | `95.00%` | `0.00%` |
| `test_domain` | `89.47%` | `89.47%` | `89.47%` | `94.74%` | `0.00%` |

Pure-visual Stage A remained weak even after the geometry-collapse fix:

- internal point accuracy: `0.0375`
- internal mean IoU: `0.0061`

Hybrid candidate-aware Stage A was the main breakthrough:

- internal point accuracy: `0.7875`
- internal mean IoU: `0.7314`
- official-subset average gain over pure visual:
  - element accuracy: `+88.16 pts`
  - click-point accuracy: `+88.07 pts`
  - IoU@0.5: `+89.82 pts`
  - action-type accuracy: `+13.68 pts`

### Mind2Web Stage B: Headroom Before and After Saturation

The cleanest Stage B story is not “reranking always improves results.” It is “candidate-pool headroom exists before saturation, then shrinks.”

Oracle best-of-k headroom from saved candidate pools:

| Candidate Pool | Point Gain vs First Choice | Reward Gain vs First Choice |
| --- | ---: | ---: |
| historical `k=4` | `+10.17 pts` | `0.0376` |
| historical `k=8 expanded` | `+10.17 pts` | `0.0431` |
| final hybrid rebuild `k=4` | `+5.08 pts` | `0.0983` |

The key qualitative takeaway is:

- earlier weaker Stage A left real recoverable headroom
- rebuilding Stage B on the strong hybrid Stage A checkpoint shrank that headroom sharply
- reranking stopped being a robust default improvement

### ScreenSpot-v2: Held-Out Transfer Benchmark

ScreenSpot-v2 is the cleanest held-out benchmark in the repository.

| Method | Point Acc | IoU@0.5 | Mean IoU |
| --- | ---: | ---: | ---: |
| reproduced public plain-Qwen baseline | `75.63%` | `5.19%` | `13.27%` |
| point-native decoupled | `77.36%` | `9.67%` | `19.12%` |
| dual-path verifier | `77.91%` | `17.22%` | `25.20%` |

Additional held-out findings:

- text elements outperform icons by `+19.70 pts` point accuracy
- desktop is `-1.58 pts` vs web
- mobile is `+4.81 pts` vs web on the current held-out split

Runtime from saved full-run artifacts:

- point-native first choice: `2.1185 s / image`
- structured support path: `1.3599 s / image`
- verifier overhead only: `0.000167 s / image`
- dual-path end-to-end: `3.4786 s / image`

### VisualWebBench: Supplementary Transfer Check

| Method | Official Choice Acc | Point Acc | Mean IoU |
| --- | ---: | ---: | ---: |
| structured screenshot-only | `78.88%` | `64.53%` | `32.06%` |
| point-native decoupled | `87.21%` | `79.46%` | `34.35%` |
| dual-path verifier | `86.82%` | `79.65%` | `33.12%` |
| Mind2Web hybrid zero-shot transfer | `23.84%` | `23.84%` | `23.84%` |

This sharpened the final claim:

- point-first transfer is strong
- dual-path gains saturate once point-native is already strong
- candidate-aware transfer does not survive a mismatched anonymous-box protocol

## What This Repository Is and Is Not

- It is a final-project repository about single-step multimodal GUI grounding.
- It is not a long-horizon web agent or a full browser automation system.
- It contains both the current final story and many historical intermediate experiments.
- The final claim is intentionally scoped. The evidence supports a nuanced conclusion, not a blanket “RL always helps” narrative.

## Start Here

If you are using this repository for report writing, presentation building, or project handoff, read these files in order:

1. [`docs/final_handoff_brief.md`](docs/final_handoff_brief.md)
2. [`docs/standalone_quantitative_evaluation_section.md`](docs/standalone_quantitative_evaluation_section.md)
3. [`docs/final_presentation_outline.md`](docs/final_presentation_outline.md)
4. [`docs/final_artifact_index.md`](docs/final_artifact_index.md)
5. [`docs/final_project_report.md`](docs/final_project_report.md)

Use these files for exact numbers:

- [`outputs/quantitative_metrics_suite/summary.json`](outputs/quantitative_metrics_suite/summary.json)
- [`outputs/quantitative_metrics_suite/summary.md`](outputs/quantitative_metrics_suite/summary.md)

Use these files for visual assets:

- [`outputs/final_packaging/figures/`](outputs/final_packaging/figures/)
- [`outputs/final_packaging/tables/`](outputs/final_packaging/tables/)

Important handoff note:

- treat `outputs/quantitative_metrics_suite/*` and [`docs/standalone_quantitative_evaluation_section.md`](docs/standalone_quantitative_evaluation_section.md) as the source of truth for exact numbers
- use `outputs/final_packaging/*` primarily for presentation-ready figures and tables

## Recommended Claim Language

Safe final-project wording:

1. Hybrid OCR/DOM candidate augmentation is critical for strong supervised GUI grounding when candidate structure is semantically informative.
2. Point-first grounding is a strong transferable inference strategy across held-out GUI benchmarks.
3. Reward-based reranking helps when the baseline is weak enough to leave real recoverable headroom.
4. Once strong supervision saturates most headroom, reranking gains become sparse and inconsistent.
5. Candidate-aware methods do not transfer unchanged across mismatched benchmark protocols.

## Key Artifacts

Main benchmark output paths:

- Mind2Web pure-visual Stage A:
  - `outputs/mind2web_stageA_sft_localization_fixed/`
- Mind2Web hybrid Stage A:
  - `outputs/mind2web_stageA_sft_hybrid_candidates/`
- Mind2Web rebuilt Stage B candidate pools:
  - `outputs/mind2web_stageB_candidates_hybrid_stagea/`
- Mind2Web rebuilt Stage B reranker:
  - `outputs/mind2web_stageB_reranker_hybrid_stagea/`
- Mind2Web rebuild comparison:
  - `outputs/mind2web_stageB_rebuild_hybrid_stagea/`
- ScreenSpot-v2 public baseline reproduction:
  - `outputs/screenspot_v2_public_qwen2_5_vl_3b_point_baseline/`
- ScreenSpot-v2 point-native:
  - `outputs/screenspot_v2_eval_qwen2_5_vl_3b_point_native_decoupled/`
- ScreenSpot-v2 dual-path verifier:
  - `outputs/screenspot_v2_eval_qwen2_5_vl_3b_dual_path_verifier/`
- VisualWebBench structured:
  - `outputs/visualwebbench_eval_qwen2_5_vl_3b_model_resized/`
- VisualWebBench point-native:
  - `outputs/visualwebbench_eval_qwen2_5_vl_3b_point_native_decoupled/`
- VisualWebBench dual-path verifier:
  - `outputs/visualwebbench_eval_qwen2_5_vl_3b_dual_path_verifier/`
- VisualWebBench Mind2Web hybrid transfer:
  - `outputs/visualwebbench_eval_mind2web_stageA_hybrid_candidates/`

Most useful documentation entry points:

- [`docs/final_handoff_brief.md`](docs/final_handoff_brief.md)
- [`docs/standalone_quantitative_evaluation_section.md`](docs/standalone_quantitative_evaluation_section.md)
- [`docs/final_project_report.md`](docs/final_project_report.md)
- [`docs/final_presentation_outline.md`](docs/final_presentation_outline.md)
- [`docs/final_artifact_index.md`](docs/final_artifact_index.md)
- [`docs/visualwebbench_supplementary_benchmark_analysis.md`](docs/visualwebbench_supplementary_benchmark_analysis.md)

## Minimal Commands

### Installation

```bash
conda create -n gui-grounding python=3.10 -y
conda activate gui-grounding
pip install -e ".[dev]"
```

### Prepare Datasets

```bash
python scripts/prepare_mind2web.py
python scripts/prepare_screenspot_v2.py
```

### Recompute Final Numbers From Saved Artifacts

```bash
python scripts/run_quantitative_metrics_suite.py
```

### Re-run Main Held-Out Evaluations

ScreenSpot-v2 point-native:

```bash
python scripts/run_eval_screenspot_v2.py \
  --config configs/eval/screenspot_v2_qwen2_5_vl_3b_point_native_decoupled.yaml
```

ScreenSpot-v2 dual-path verifier:

```bash
python scripts/run_eval_dual_path_verifier.py \
  --config configs/eval/screenspot_v2_qwen2_5_vl_3b_dual_path_verifier.yaml
```

VisualWebBench point-native:

```bash
python scripts/run_eval_visualwebbench.py \
  --config configs/eval/visualwebbench_qwen2_5_vl_3b_point_native_decoupled.yaml
```

Mind2Web hybrid Stage A training:

```bash
python scripts/run_train_sft.py \
  --config configs/train/mind2web_stageA_qwen2_5_vl_3b_sft_hybrid_candidates.yaml
```

Mind2Web Stage B candidate export on hybrid Stage A:

```bash
python scripts/run_generate_candidates.py \
  --config configs/train/mind2web_stageB_candidates_qwen2_5_vl_3b_hybrid_stagea.yaml
```

Mind2Web rebuilt reranker:

```bash
python scripts/run_train_reranker.py \
  --config configs/train/mind2web_stageB_reranker_qwen_hybrid_stagea.yaml
```

## Repository Layout

```text
.
├── configs/                  # train / eval / data / demo configs
├── data/                     # raw and processed benchmark assets
├── docs/                     # experiment reports and handoff docs
├── outputs/                  # checkpoints, predictions, comparisons, final packaging
├── scripts/                  # runnable entry points and summary utilities
├── src/gui_grounding/        # data, models, reward, training, evaluation, utils
└── tests/                    # unit tests
```

## Status

The benchmark story is complete across Mind2Web, ScreenSpot-v2, and the supplementary VisualWebBench transfer check. The repository is no longer in a benchmark-building phase. It is in a final packaging and handoff phase for report writing, presentation reuse, and optional demo polish.
