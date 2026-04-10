# Final Artifact Index

This file is the handoff map for report writing, presentation building, and artifact lookup.

## Read These First

1. [`README.md`](../README.md)
2. [`final_handoff_brief.md`](final_handoff_brief.md)
3. [`standalone_quantitative_evaluation_section.md`](standalone_quantitative_evaluation_section.md)
4. [`final_presentation_outline.md`](final_presentation_outline.md)
5. [`final_project_report.md`](final_project_report.md)

## Source of Truth for Exact Numbers

Use these files for any metric you quote in the final report or slides:

- [`summary.json`](../outputs/quantitative_metrics_suite/summary.json)
- [`summary.md`](../outputs/quantitative_metrics_suite/summary.md)
- [`standalone_quantitative_evaluation_section.md`](standalone_quantitative_evaluation_section.md)

Important rule:

- use `outputs/quantitative_metrics_suite/*` for exact numbers
- use `outputs/final_packaging/*` for presentation-ready visuals

## Most Useful Visual Assets

- ScreenSpot-v2 method comparison:
  - [`screenspot_v2_before_after_and_method_comparison.png`](../outputs/final_packaging/figures/screenspot_v2_before_after_and_method_comparison.png)
- Mind2Web Stage A pure vs hybrid:
  - [`mind2web_stageA_pure_vs_hybrid.png`](../outputs/final_packaging/figures/mind2web_stageA_pure_vs_hybrid.png)
- Mind2Web Stage B old vs rebuild headroom:
  - [`mind2web_stageB_headroom_old_vs_hybrid_rebuild.png`](../outputs/final_packaging/figures/mind2web_stageB_headroom_old_vs_hybrid_rebuild.png)
- VisualWebBench method comparison:
  - [`visualwebbench_method_comparison.png`](../outputs/final_packaging/figures/visualwebbench_method_comparison.png)

## Benchmark Output Directories

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
- Earlier weaker-Stage-A Stage B evidence:
  - `outputs/mind2web_stageB_candidates/`
  - `outputs/mind2web_stageB_candidates_headroom_expanded/`
  - `outputs/mind2web_stageB_reranker_source_aware_supervision/`
- ScreenSpot-v2 public baseline:
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

## Best Methods to Present

- Mind2Web default supervised baseline:
  - hybrid screenshot + OCR/DOM candidate-aware Stage A
- Mind2Web Stage B final interpretation:
  - earlier weaker Stage A had real headroom
  - rebuilt hybrid-stage-A reranking is not a robust default win
- ScreenSpot-v2 strongest final method:
  - dual-path candidate generation + lightweight verifier
- VisualWebBench strongest default transfer method:
  - point-native decoupled inference path

## Safe Final Claims

Use language close to the following:

1. Hybrid OCR/DOM candidate augmentation is critical for strong supervised GUI grounding when the benchmark exposes semantically informative candidate structure.
2. Point-first grounding is a strong transferable inference strategy across held-out GUI benchmarks.
3. Reward-based reranking helps when the baseline is weak enough to leave real recoverable headroom.
4. Once strong supervision saturates most recoverable headroom, reranking gains become sparse and inconsistent.
5. Candidate-aware methods do not transfer unchanged across mismatched benchmark protocols.

## Claims to Avoid

- Do not say pure screenshot supervision was already strong.
- Do not say reranking universally improved the final strong baseline.
- Do not say Mind2Web hybrid transfer generalizes unchanged to any GUI benchmark.
- Do not frame the project as a full browser-agent system.

## Primary Supporting Reports

- Final report draft:
  - [`final_project_report.md`](final_project_report.md)
- Presentation outline:
  - [`final_presentation_outline.md`](final_presentation_outline.md)
- Quantitative standalone section:
  - [`standalone_quantitative_evaluation_section.md`](standalone_quantitative_evaluation_section.md)
- Mind2Web Stage A report:
  - [`mind2web_stageA_hybrid_representation_strong_baseline.md`](mind2web_stageA_hybrid_representation_strong_baseline.md)
- Mind2Web Stage B rebuild report:
  - [`mind2web_stageB_rebuild_on_hybrid_stageA.md`](mind2web_stageB_rebuild_on_hybrid_stageA.md)
- ScreenSpot-v2 public baseline report:
  - [`public_baseline_reproduction_and_same_protocol_comparison.md`](public_baseline_reproduction_and_same_protocol_comparison.md)
- ScreenSpot-v2 dual-path verifier report:
  - [`dual_path_candidate_generation_and_lightweight_verifier.md`](dual_path_candidate_generation_and_lightweight_verifier.md)
- VisualWebBench report:
  - [`visualwebbench_supplementary_benchmark_analysis.md`](visualwebbench_supplementary_benchmark_analysis.md)
