# OpenGUI-RL

**Cross-Website GUI Grounding with Verifiable Reward Optimization**

OpenGUI-RL is a research-engineering project for instruction-conditioned GUI grounding in computer-use agents, framed as a single-step contextual-bandit decision with verifiable reward.

<p align="center">
  <img src="assets/figures/opengui_rl_pipeline.png" alt="OpenGUI-RL pipeline" width="96%">
</p>

<p align="center"><em>A wrong-click failure is isolated as one-step GUI grounding, then studied through candidate-aware supervision, reward-based reranking, and point-native / dual-path transfer inference.</em></p>

## Why This Matters

Computer-use assistants can understand an instruction and still fail at the first executable step: clicking the wrong UI element. OpenGUI-RL studies that bottleneck directly by isolating the perception-to-action grounding layer rather than claiming to solve full browser automation.

Given a screenshot and a natural-language instruction, the system predicts a structured GUI action: target element, click point, bounding box, and action type. A deterministic verifier scores the action against the annotated target.

## What Is Implemented

- **Stage A: candidate-aware supervised grounding** with screenshot, instruction, and OCR/DOM-style candidate cues.
- **Stage B: reward-labeled candidate generation and reranking** with first-choice, oracle best-of-\(k\), and learned reranker analysis.
- **Held-out transfer inference** with point-native decoding and a lightweight dual-path verifier.
- **Benchmark analysis** across Mind2Web, ScreenSpot-v2, and VisualWebBench, focused on headroom and protocol-sensitive transfer.

## Main Findings

- **Representation first.** On Mind2Web, semantically meaningful OCR/DOM candidate cues dominate pure screenshot-only supervision.
- **Reward-based reranking is conditional.** It helps when the candidate pool still contains recoverable alternatives, but gains shrink once Stage A grounding is strong.
- **Point-native inference transfers well.** On held-out GUI grounding benchmarks, making click prediction primary transfers more reliably than box-first decoding.
- **Transfer is protocol-sensitive.** Candidate-aware methods depend on meaningful candidate protocols and do not transfer unchanged to anonymous option-box settings such as VisualWebBench.

## What Is Not In Scope

- This is not a full browser automation agent.
- This is not a long-horizon planning, memory, recovery, or execution-loop system.
- This is not an online PPO browser-training implementation.
- This repository does not redistribute benchmark datasets, raw benchmark screenshots, or large model checkpoints.
- The final claim is not that reinforcement learning always beats supervision.

## RL Framing

The project uses a contextual-bandit-style formulation:

- **Context / state:** screenshot, instruction, and optional OCR/DOM candidate cues.
- **Action:** structured GUI action with selected element, click point, bounding box, and action type.
- **Reward:** deterministic verifiable score from element match, click-inside-target, IoU, action-type match, and invalid-format penalties.
- **Transition:** single-step episode; evaluation stops after scoring the prediction.
- **Objective:** improve expected one-step reward under the scoped grounding decision layer.

This preserves the RL ingredients that matter for this project while avoiding confounds from full browser dynamics. Reward is used to label candidates, build preference pairs, estimate best-of-\(k\) headroom, and test whether reward-based selection improves the first-stage policy output.

## Repository Structure

```text
.
├── assets/                  # public figures used by README and documentation
├── artifacts/               # small releasable metrics, plots, and tables
├── configs/                 # data, model, training, evaluation, and demo configs
├── data/                    # placeholder directories only; datasets are not redistributed
├── data_examples/           # tiny synthetic examples of expected record formats
├── docs/                    # external-facing scope, data, and reproduction notes
├── notebooks/               # lightweight output-cleared sanity-check notebooks
├── scripts/                 # command-line entry points for preparation, training, eval
├── src/gui_grounding/       # package code: data, models, reward, training, evaluation
└── tests/                   # lightweight unit tests
```

The Python package name is currently `gui_grounding`; the repository and release name are **OpenGUI-RL**.

## Setup

Python 3.10 is recommended. GPU hardware is needed for Qwen2.5-VL training and full evaluation; CPU is sufficient for lightweight tests and most metadata checks.

```bash
conda create -n opengui-rl python=3.10 -y
conda activate opengui-rl
pip install -e ".[dev]"
```

Optional environment variables can be copied from `.env.example`. Do not commit real tokens.

```bash
cp .env.example .env
```

## Reproduction Path

Full reproduction requires external access to the relevant datasets and enough compute for Qwen2.5-VL LoRA training or inference. The public repository provides code, configs, small summary artifacts, and synthetic data-format examples, but not benchmark payloads or checkpoints.

1. Prepare dataset interfaces after obtaining benchmark access:

```bash
python scripts/prepare_mind2web.py
python scripts/prepare_screenspot_v2.py
```

2. Train Stage A candidate-aware supervised grounding:

```bash
python scripts/run_train_sft.py \
  --config configs/train/mind2web_stageA_qwen2_5_vl_3b_sft_hybrid_candidates.yaml
```

3. Export Stage B reward-labeled candidate pools:

```bash
python scripts/run_generate_candidates.py \
  --config configs/train/mind2web_stageB_candidates_qwen2_5_vl_3b_hybrid_stagea.yaml
```

4. Train the lightweight reward-based reranker:

```bash
python scripts/run_train_reranker.py \
  --config configs/train/mind2web_stageB_reranker_qwen_hybrid_stagea.yaml
```

5. Run held-out point-native and dual-path inference:

```bash
python scripts/run_eval_screenspot_v2.py \
  --config configs/eval/screenspot_v2_qwen2_5_vl_3b_point_native_decoupled.yaml

python scripts/run_eval_dual_path_verifier.py \
  --config configs/eval/screenspot_v2_qwen2_5_vl_3b_dual_path_verifier.yaml

python scripts/run_eval_visualwebbench.py \
  --config configs/eval/visualwebbench_qwen2_5_vl_3b_point_native_decoupled.yaml
```

6. Recompute the final quantitative summary from saved artifacts, when available locally:

```bash
python scripts/run_quantitative_metrics_suite.py
```

For details, see [docs/REPRODUCTION.md](docs/REPRODUCTION.md) and [docs/DATA.md](docs/DATA.md).

## Results Summary

The compact summary below mirrors the final project findings. It should be read as scoped benchmark evidence, not as a universal claim about RL dominance.

| Benchmark slice | Key comparison | Reported result | Takeaway |
| --- | --- | --- | --- |
| Mind2Web Stage A | Pure visual vs hybrid candidate-aware | Internal point accuracy `0.0375 -> 0.7875`; cached official split point accuracy `0.0000/0.0000/0.0526 -> 0.9500/0.8500/0.8947` | OCR/DOM candidate semantics are decisive when available. |
| Mind2Web Stage B | Historical pools vs hybrid rebuild | Best-of-\(k\) point headroom shrinks from `+10.17 pts` to `+5.08 pts` after stronger Stage A | Reward-based selection depends on recoverable headroom. |
| ScreenSpot-v2 | Public baseline, point-native, dual-path | Point accuracy `75.63% -> 77.36% -> 77.91%` | Point-native transfer is strong; dual-path adds a modest gain. |
| VisualWebBench | Structured, point-native, dual-path, hybrid transfer | Choice accuracy `78.88% -> 87.21%`; dual-path `86.82%`; Mind2Web hybrid transfer `23.84%` | Point-native transfer generalizes; candidate-aware transfer is protocol-sensitive. |

Small public artifacts are available in:

- [artifacts/metrics/quantitative_summary.md](artifacts/metrics/quantitative_summary.md)
- [artifacts/tables/](artifacts/tables/)
- [assets/figures/](assets/figures/)

## Limitations

- The project studies single-step grounding, not full browser automation.
- Sequential planning, execution feedback, and recovery from wrong clicks are outside the final scope.
- Mind2Web official split results are cached subset readouts rather than full large-scale benchmark sweeps.
- Main training runs use a single seed due to compute limits, so small deltas should be interpreted cautiously.
- ScreenSpot-Pro was proposed as an optional stress test but was not completed.
- Candidate-aware grounding depends on meaningful OCR/DOM candidate structure and does not transfer unchanged to anonymous candidate protocols.

## Acknowledgments

This project builds on public research infrastructure and benchmarks including Qwen2.5-VL, Mind2Web, ScreenSpot-v2, VisualWebBench, WebArena, OSWorld, SeeAct, CogAgent, and SeeClick. Dataset access and redistribution are governed by the respective upstream licenses and terms.

## License

Code in this repository is released under the MIT License. Benchmark datasets, screenshots, and model weights are not redistributed and remain subject to their original licenses.
