# Final Presentation Outline

Recommended length: 10 to 12 slides  
Recommended style: thesis first, evidence second, honest about negative findings

## Source of Truth

Use these files when writing slide numbers:

- [`outputs/quantitative_metrics_suite/summary.md`](../outputs/quantitative_metrics_suite/summary.md)
- [`outputs/quantitative_metrics_suite/summary.json`](../outputs/quantitative_metrics_suite/summary.json)
- [`docs/standalone_quantitative_evaluation_section.md`](standalone_quantitative_evaluation_section.md)

Use these for visuals:

- [`outputs/final_packaging/figures/`](../outputs/final_packaging/figures/)

## Slide 1 - Title and Final Thesis

**Title**

Cross-Website GUI Grounding with Verifiable Reward Optimization

**What to say**

- Goal: map `screenshot + instruction` to `element / bbox / click point / action type`
- Final thesis:
  - hybrid OCR/DOM candidate augmentation is critical when candidate structure is meaningful
  - point-first grounding transfers well on held-out GUI benchmarks
  - reward-based reranking helps when headroom exists, then saturates

**Recommended visual**

- [`cross_benchmark_summary.md`](../outputs/final_packaging/tables/cross_benchmark_summary.md)

## Slide 2 - Problem Framing and RL Connection

**What to say**

- Scope is single-step GUI grounding, not full browser automation
- Context = screenshot + instruction
- Action = `click_point`, `bbox_proposal`, `action_type`
- Reward = element correctness + click validity + IoU + action-type correctness - invalid format
- The project stayed inside this RL framing, but the strongest gains came from representation and inference structure

**Recommended visual**

- Simple contextual-bandit diagram or a hand-built pipeline figure

## Slide 3 - Benchmarks and Why Each One Matters

**What to say**

- Mind2Web:
  - main benchmark for supervision, candidate structure, and Stage B reranking
- ScreenSpot-v2:
  - clean held-out benchmark for transfer-ready grounding
- VisualWebBench:
  - supplementary transfer test under a mismatched anonymous-box protocol

**Recommended visual**

- one 3-column benchmark role table

## Slide 4 - Mind2Web Negative Result: Pure Visual Was Too Weak

**What to say**

- Geometry-collapse bug was fixed first
- Even after that, pure-visual Stage A stayed weak:
  - internal point accuracy `0.0375`
  - internal mean IoU `0.0061`
- This ruled out the easy claim that screenshot-only supervision was already enough

**Recommended visual**

- left panel from [`mind2web_stageA_pure_vs_hybrid.png`](../outputs/final_packaging/figures/mind2web_stageA_pure_vs_hybrid.png)

## Slide 5 - Mind2Web Positive Result: Hybrid Candidate-Aware Stage A

**What to say**

- Added screenshot + OCR/DOM-style candidate anchors
- Added candidate-slot supervision and candidate-anchored grounding
- Internal validation jumped to:
  - point accuracy `0.7875`
  - mean IoU `0.7314`
- Official cached subset metrics:

| Split | Element | Point | IoU@0.5 | Action |
| --- | ---: | ---: | ---: | ---: |
| `test_task` | `95.00%` | `95.00%` | `95.00%` | `90.00%` |
| `test_website` | `80.00%` | `85.00%` | `85.00%` | `95.00%` |
| `test_domain` | `89.47%` | `89.47%` | `89.47%` | `94.74%` |

**Recommended visual**

- [`mind2web_stageA_pure_vs_hybrid.png`](../outputs/final_packaging/figures/mind2web_stageA_pure_vs_hybrid.png)

## Slide 6 - Mind2Web Stage B: Headroom Exists Before Saturation

**What to say**

- Earlier weaker Stage A left real recoverable headroom in the candidate pool
- Oracle best-of-k headroom from saved candidate artifacts:
  - historical `k=4`: `+10.17 pts`
  - historical `k=8 expanded`: `+10.17 pts`
  - final hybrid rebuild `k=4`: `+5.08 pts`
- Interpretation:
  - reward-based reranking is meaningful when baseline errors are still recoverable
  - strong Stage A shrinks the remaining headroom

**Recommended visual**

- [`mind2web_stageB_headroom_old_vs_hybrid_rebuild.png`](../outputs/final_packaging/figures/mind2web_stageB_headroom_old_vs_hybrid_rebuild.png)

## Slide 7 - ScreenSpot-v2 Turning Point: Coordinate-Frame Mismatch

**What to say**

- Initial localization looked catastrophically bad
- Failure analysis showed a frame mismatch between model-resized coordinates and original screenshot coordinates
- After refinement, the held-out benchmark became meaningful
- This debugging step matters because it turned an invalid benchmark loop into a reliable one

**Recommended visual**

- a coordinate-frame diagram, or the before/after ScreenSpot figure

## Slide 8 - ScreenSpot-v2 Final Method Ladder

**What to say**

| Method | Point Acc | IoU@0.5 | Mean IoU |
| --- | ---: | ---: | ---: |
| public plain-Qwen baseline | `75.63%` | `5.19%` | `13.27%` |
| point-native decoupled | `77.36%` | `9.67%` | `19.12%` |
| dual-path verifier | `77.91%` | `17.22%` | `25.20%` |

- Point-native beat the reproduced public baseline
- Dual-path improved slightly further and recovered stronger box quality
- Subgroup finding:
  - text vs icon point-accuracy gap = `+19.70 pts`

**Recommended visual**

- [`screenspot_v2_before_after_and_method_comparison.png`](../outputs/final_packaging/figures/screenspot_v2_before_after_and_method_comparison.png)

## Slide 9 - VisualWebBench Refined the Transfer Claim

**What to say**

| Method | Official Choice Acc | Point Acc | Mean IoU |
| --- | ---: | ---: | ---: |
| structured screenshot-only | `78.88%` | `64.53%` | `32.06%` |
| point-native decoupled | `87.21%` | `79.46%` | `34.35%` |
| dual-path verifier | `86.82%` | `79.65%` | `33.12%` |
| Mind2Web hybrid transfer | `23.84%` | `23.84%` | `23.84%` |

- Point-first transfer remained strong
- Dual-path no longer beat point-native overall
- Mind2Web hybrid transfer failed because candidate semantics did not transfer

**Recommended visual**

- [`visualwebbench_method_comparison.png`](../outputs/final_packaging/figures/visualwebbench_method_comparison.png)

## Slide 10 - Final Takeaways

**What to say**

- Strong positive result:
  - hybrid OCR/DOM candidate augmentation is critical when candidate structure is meaningful
- Strong transferable result:
  - point-first grounding is robust across held-out GUI benchmarks
- Scoped RL result:
  - reranking helps when headroom exists
  - reranking saturates once strong supervision already solves most recoverable cases
- Negative but important result:
  - candidate-aware transfer fails under mismatched anonymous-box protocols

**Recommended visual**

- [`cross_benchmark_summary.md`](../outputs/final_packaging/tables/cross_benchmark_summary.md)

## Optional Slide 11 - Runtime and Deployment Practicality

**What to say**

- ScreenSpot-v2 point-native first choice: `2.1185 s / image`
- structured support path: `1.3599 s / image`
- verifier overhead only: `0.000167 s / image`
- dual-path end-to-end: `3.4786 s / image`
- Main point:
  - the final system is a practical perception-and-selection stack, not a huge multi-stage agent loop

## Optional Slide 12 - Limits and Honest Scope

**What to say**

- Mind2Web official split results are pilot-sized cached subset readouts
- The project studies single-step grounding, not complete browser automation
- Stage B should not be oversold as a universal reranking win
- The correct final claim is narrower and stronger because it is evidence-backed
