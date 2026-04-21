# Reproduction Guide

This guide describes the public reproduction path for OpenGUI-RL. It is intentionally honest about what the repository can reproduce by itself and what requires external benchmark access.

## What Is Reproducible From This Repository Alone

Without external datasets or checkpoints, users can:

- inspect the public README, docs, and summary artifacts,
- run lightweight unit tests,
- inspect the reward schema, candidate representation, and dual-path verifier code,
- use synthetic records in `data_examples/` to understand the expected data interface,
- inspect public summary artifacts in `artifacts/`.

## What Requires External Access

The main reported experiments require:

- Mind2Web access for Stage A and Stage B training / evaluation,
- ScreenSpot-v2 access for held-out point-native and dual-path evaluation,
- VisualWebBench access for supplementary transfer analysis,
- Qwen2.5-VL model access and GPU memory suitable for VLM inference or LoRA training.

The repository does not ship model checkpoints, raw screenshots, or benchmark payloads.

## Setup

```bash
conda create -n opengui-rl python=3.10 -y
conda activate opengui-rl
pip install -e ".[dev]"
```

Optional local environment:

```bash
cp .env.example .env
```

Fill in tokens only on your machine. Do not commit `.env`.

## Stage A: Candidate-Aware Supervised Grounding

Stage A trains a Qwen2.5-VL LoRA policy to emit structured GUI actions. The final project setting uses screenshot + instruction + OCR/DOM-style candidate cues.

```bash
python scripts/run_train_sft.py \
  --config configs/train/mind2web_stageA_qwen2_5_vl_3b_sft_hybrid_candidates.yaml
```

Expected role:

- tests whether a faithful candidate-aware interface can build a strong supervised grounding baseline,
- produces the first-stage policy used by Stage B candidate export,
- depends on external Mind2Web access and local compute.

## Stage B: Reward-Labeled Candidate Generation

Stage B exports a small candidate pool per example and labels each candidate with deterministic verifiable reward.

```bash
python scripts/run_generate_candidates.py \
  --config configs/train/mind2web_stageB_candidates_qwen2_5_vl_3b_hybrid_stagea.yaml
```

The exported pool supports:

- first-choice evaluation,
- oracle best-of-\(k\) headroom,
- pairwise preference construction,
- learned reward-based reranking.

## Stage B: Lightweight Reranker

```bash
python scripts/run_train_reranker.py \
  --config configs/train/mind2web_stageB_reranker_qwen_hybrid_stagea.yaml
```

The reranker is intentionally small and auditable. It is used to test whether reward-labeled candidate selection adds value after Stage A, not to hide the grounding problem inside another large model.

## Held-Out Point-Native / Dual-Path Inference

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

VisualWebBench dual-path verifier:

```bash
python scripts/run_eval_visualwebbench_dual_path_verifier.py \
  --config configs/eval/visualwebbench_qwen2_5_vl_3b_dual_path_verifier.yaml
```

## Quantitative Summary

When local saved artifacts are available, recompute the summary:

```bash
python scripts/run_quantitative_metrics_suite.py
```

The public release includes a frozen copy of the reported summary in `artifacts/metrics/`. That copy is for inspection; it is not a substitute for rerunning the benchmark pipeline with external data access.

## Interpreting Results

Do not read the reported numbers as "RL always beats supervision." The intended conclusion is narrower:

- representation quality comes first,
- reward-based reranking is useful when candidate-pool headroom remains,
- point-native inference transfers well,
- candidate-aware transfer depends on semantically meaningful candidate protocols.
