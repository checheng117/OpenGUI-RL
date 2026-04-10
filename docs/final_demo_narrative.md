# Lightweight Demo Narrative

This project does not need a heavy live demo. The best demo format is a short qualitative walkthrough followed immediately by the benchmark evidence.

## Recommended Demo Goal

Use the demo to help the audience understand the action format:

- input: screenshot + instruction
- output: `click_point`, `bbox_proposal`, `action_type`

Do not use the demo as the main proof. The proof is the benchmark suite.

## Suggested 2-Minute Flow

1. Open with one sentence:
   - “This project focuses on the single-step GUI grounding layer behind browser agents.”
2. Show one saved qualitative example.
3. Show a second example from a different page layout.
4. Transition immediately to the benchmark takeaway:
   - Mind2Web hybrid candidate-aware grounding is the strongest supervised result
   - ScreenSpot-v2 dual-path verifier is the strongest held-out result
   - VisualWebBench confirms the transfer claim is about point-first grounding, not arbitrary candidate slots

## Example 1

- Artifact:
  - `outputs/qwen_multisample_validation_qwen2_5/visualizations/mind2web_train_6c7a7082-2897-41c7-9688-4b0f3d778cdb.png`
- Why it is useful:
  - shows a realistic webpage screenshot
  - makes the structured action output easy to explain

## Example 2

- Artifact:
  - `outputs/qwen_multisample_validation_qwen2_5/visualizations/mind2web_train_c7548fe6-29eb-4ffb-a431-24ad7f535f5c.png`
- Why it is useful:
  - shows a very different page geometry
  - reinforces the cross-website nature of the task

## Benchmark Anchor Lines

If you want one sentence after the qualitative examples, use this:

- “The qualitative demo shows the output format, but the real evidence is the benchmark suite: Mind2Web hybrid Stage A reaches `95.00 / 80.00 / 89.47` element accuracy on the official cached subsets, ScreenSpot-v2 dual-path verifier reaches `77.91%` point accuracy, and VisualWebBench point-native reaches `87.21%` official choice accuracy.”

## Best Supporting Slides

- quantitative summary:
  - [`outputs/quantitative_metrics_suite/summary.md`](../outputs/quantitative_metrics_suite/summary.md)
- ScreenSpot-v2 comparison:
  - [`outputs/final_packaging/figures/screenspot_v2_before_after_and_method_comparison.png`](../outputs/final_packaging/figures/screenspot_v2_before_after_and_method_comparison.png)
- Mind2Web Stage A comparison:
  - [`outputs/final_packaging/figures/mind2web_stageA_pure_vs_hybrid.png`](../outputs/final_packaging/figures/mind2web_stageA_pure_vs_hybrid.png)
