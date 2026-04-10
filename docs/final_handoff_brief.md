# Final Handoff Brief

This is the fastest way for another person to understand the repository well enough to write the final report or presentation.

## One-Paragraph Project Summary

This project studies single-step multimodal GUI grounding rather than full browser automation. The core task is: given a screenshot and a natural-language instruction, predict the correct UI target and action in a verifiable format. The final story is not that reward-based reranking always wins. It is that the project became strongest when it combined three evidence-backed ideas: semantically informative OCR/DOM candidate augmentation on Mind2Web, point-first Qwen inference on held-out benchmarks, and careful measurement of when reranking still has recoverable headroom. The strongest supervised baseline came from hybrid candidate-aware grounding, the strongest held-out transfer results came from point-native and dual-path inference on ScreenSpot-v2, and the final RL conclusion is conditional: reranking helps when headroom exists, then saturates once Stage A becomes strong.

## Source of Truth

For exact numbers, use:

- [`outputs/quantitative_metrics_suite/summary.json`](../outputs/quantitative_metrics_suite/summary.json)
- [`outputs/quantitative_metrics_suite/summary.md`](../outputs/quantitative_metrics_suite/summary.md)
- [`standalone_quantitative_evaluation_section.md`](standalone_quantitative_evaluation_section.md)

For presentation visuals, use:

- [`outputs/final_packaging/figures/`](../outputs/final_packaging/figures/)

## Final Claims to Repeat

1. Hybrid OCR/DOM candidate augmentation is critical when candidate structure is semantically informative.
2. Point-first grounding is a strong transferable inference strategy across held-out GUI benchmarks.
3. Reward-based reranking helps when the baseline is weak enough to leave real recoverable headroom.
4. Once strong supervision saturates most recoverable headroom, reranking gains shrink sharply.
5. Candidate-aware methods do not transfer unchanged across mismatched benchmark protocols.

## Exact Numbers Worth Citing

### Mind2Web Hybrid Stage A Official Cached Subset

| Split | Element | Point | IoU@0.5 | Action |
| --- | ---: | ---: | ---: | ---: |
| `test_task` | `95.00%` | `95.00%` | `95.00%` | `90.00%` |
| `test_website` | `80.00%` | `85.00%` | `85.00%` | `95.00%` |
| `test_domain` | `89.47%` | `89.47%` | `89.47%` | `94.74%` |

### Pure Visual vs Hybrid

- pure-visual internal point accuracy: `0.0375`
- pure-visual internal mean IoU: `0.0061`
- hybrid internal point accuracy: `0.7875`
- hybrid internal mean IoU: `0.7314`
- official-subset average hybrid minus pure-visual gains:
  - element: `+88.16 pts`
  - point: `+88.07 pts`
  - IoU@0.5: `+89.82 pts`
  - action type: `+13.68 pts`

### Mind2Web Stage B Headroom

- historical `k=4` oracle point gain: `+10.17 pts`
- historical `k=8 expanded` oracle point gain: `+10.17 pts`
- final hybrid rebuild `k=4` oracle point gain: `+5.08 pts`

### ScreenSpot-v2

- public plain-Qwen baseline point accuracy: `75.63%`
- point-native point accuracy: `77.36%`
- dual-path verifier point accuracy: `77.91%`
- dual-path verifier IoU@0.5: `17.22%`
- text minus icon point-accuracy gap: `+19.70 pts`

### VisualWebBench

- structured official choice accuracy: `78.88%`
- point-native official choice accuracy: `87.21%`
- dual-path official choice accuracy: `86.82%`
- Mind2Web hybrid transfer official choice accuracy: `23.84%`

## Recommended Report Structure

1. Introduce GUI grounding as the perception layer behind browser/computer-use agents.
2. Frame the task as a single-step contextual-bandit-like decision problem with verifiable reward.
3. Explain the three benchmarks and their different roles.
4. Present Mind2Web Stage A as the main supervised-learning result:
   - pure visual failed
   - hybrid OCR/DOM candidate-aware grounding succeeded
5. Present Stage B carefully:
   - headroom existed on weaker baselines
   - headroom shrank after rebuilding on strong hybrid Stage A
6. Present ScreenSpot-v2 as the strongest held-out transfer result.
7. Present VisualWebBench as the benchmark that sharpened the transfer claim.
8. Conclude with the scoped claim, not with an oversold RL claim.

## Recommended Presentation Structure

Use [`final_presentation_outline.md`](final_presentation_outline.md). The shortest good version is:

1. title + thesis
2. task + RL framing
3. benchmarks and why they exist
4. Mind2Web pure visual failure
5. Mind2Web hybrid breakthrough
6. Stage B headroom before and after saturation
7. ScreenSpot-v2 debugging and final method ladder
8. VisualWebBench transfer refinement
9. final takeaways

## What to Avoid

- Do not say “RL was the main reason the project succeeded.”
- Do not say “the reranker improved the final strong baseline everywhere.”
- Do not say “candidate slots generalize by themselves.”
- Do not say “this is a complete browser agent.”

## Best Files for Each Need

- Need the exact numbers:
  - [`summary.md`](../outputs/quantitative_metrics_suite/summary.md)
- Need polished wording for the quantitative section:
  - [`standalone_quantitative_evaluation_section.md`](standalone_quantitative_evaluation_section.md)
- Need a slide plan:
  - [`final_presentation_outline.md`](final_presentation_outline.md)
- Need a broad narrative draft:
  - [`final_project_report.md`](final_project_report.md)
- Need figure and artifact paths:
  - [`final_artifact_index.md`](final_artifact_index.md)
