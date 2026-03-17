# Experiment Plan

> **Status**: This document describes planned experiments. No results have been produced yet.

## Baselines

| ID | Method | Description |
|----|--------|-------------|
| B0 | Zero-shot VLM | Prompt-only, no fine-tuning |
| B1 | SFT (screenshot only) | Supervised fine-tuning on Mind2Web, visual input only |
| B2 | SFT (screenshot + OCR/DOM) | SFT with auxiliary text cues |
| B3 | SFT + Reward Reranking | B1/B2 + top-K reranking with verifiable reward |
| B4 | SFT + Pairwise Preference | B1/B2 + DPO-style optimization |
| B5 | SFT + GRPO-Light | B1/B2 + group reward policy optimization |

## Metrics

| Metric | Description | Applies To |
|--------|-------------|------------|
| Element Accuracy | Exact match of predicted element | Mind2Web |
| Point Accuracy | Click point inside GT box | ScreenSpot-v2 |
| Mean IoU | Average bounding-box overlap | All |
| IoU@0.5 | Fraction with IoU ≥ 0.5 | All |
| Action-Type Accuracy | Correct action classification | Mind2Web |
| Best-of-K Improvement | Gain from oracle reranking | Reranking experiments |
| Reranked Gain | Reward improvement from learned reranker | Reranking experiments |

## Main Comparison Table (Template)

> **Note**: All values below are placeholders. Fill in after running experiments.

| Method | Elem Acc | Point Acc | IoU@0.5 | Act Acc | Split |
|--------|----------|-----------|---------|---------|-------|
| B0: Zero-shot | — | — | — | — | test_task |
| B1: SFT (vis) | — | — | — | — | test_task |
| B2: SFT (vis+text) | — | — | — | — | test_task |
| B3: + Reranking | — | — | — | — | test_task |
| B4: + Pairwise | — | — | — | — | test_task |
| B5: + GRPO | — | — | — | — | test_task |

## Generalization Table (Template)

| Method | test_task | test_website | test_domain | ScreenSpot-v2 |
|--------|-----------|-------------|-------------|---------------|
| B1: SFT | — | — | — | — |
| B3: + Reranking | — | — | — | — |
| Best | — | — | — | — |

## Planned Ablation Studies

### A1: Input Representation
- Pure screenshot vs. screenshot + OCR vs. screenshot + DOM candidates
- Expected: DOM candidates help on text-heavy elements; visual-only is more general

### A2: Candidate Generation Strategy
- Single-candidate decoding (greedy) vs. multi-candidate (temperature sampling, K=4,8,16)
- Expected: More candidates → better best-of-K, but diminishing returns

### A3: Text vs. Icon Elements
- Separate accuracy on text elements vs. icon/widget elements
- Expected: Text elements easier due to OCR alignment; icons require stronger visual grounding

### A4: Seen vs. Unseen
- Performance on seen websites vs. unseen websites vs. unseen domains
- Expected: Progressive degradation; reward optimization may help with unseen websites

### A5: Reward Weight Sensitivity
- Vary λ1…λ5 and measure downstream performance
- Expected: Element-correct weight (λ1) most impactful; format penalty (λ5) important for stability

### A6: Model Size
- Qwen2-VL 2B vs. 7B
- Expected: 7B stronger baseline, but 2B may benefit more from reward optimization

## Train / Eval Split Strategy

- **Training**: Mind2Web train split (7,775 actions from 1,009 tasks)
- **In-domain eval**: Mind2Web test_task
- **Cross-website eval**: Mind2Web test_website
- **Cross-domain eval**: Mind2Web test_domain
- **External eval**: ScreenSpot-v2 test (desktop + mobile + web)
- **Supplementary**: VisualWebBench (element grounding, action grounding, action prediction)
- **Stress test** (optional): ScreenSpot-Pro
