# Final Artifact Index

## One-Paragraph Project Summary

This project packages **Cross-Website GUI Grounding with Verifiable Reward Optimization** as a multimodal LLM engineering study of screenshot + instruction to grounded UI action prediction. The final report keeps the RL relevance explicit through a contextual-bandit formulation and verifiable reward, but the strongest final empirical result is not reward optimization alone. Early surrogate/reranker/preference experiments produced useful infrastructure and negative findings, while the decisive held-out improvement came from the Qwen-first path after identifying and fixing a coordinate-frame mismatch between the resized model-view image and the original screenshot. On the full ScreenSpot-v2 clean held-out benchmark, this refinement improved point accuracy from `0.0849` to `0.7099` on `1272/1272` samples.

## Main Deliverables

- Final report:
  - `docs/final_project_report.md`
- Final presentation outline:
  - `docs/final_presentation_outline.md`
- Final packaging figures:
  - `outputs/final_packaging/figures/`
- Final packaging tables:
  - `outputs/final_packaging/tables/`
- Optional demo narrative:
  - `docs/final_demo_narrative.md`

## Best Method and Best Config

- Best method:
  - `Qwen/Qwen2.5-VL-3B-Instruct` with `coordinate_frame=model_resized`
- Named config:
  - `configs/eval/screenspot_v2_qwen2_5_vl_3b_model_resized_full.yaml`

## Authoritative Evaluation Artifacts

- Baseline full held-out summary:
  - `outputs/screenspot_v2_eval_qwen2_5_vl_3b/evaluation_summary.json`
- Baseline subgroup metrics:
  - `outputs/screenspot_v2_eval_qwen2_5_vl_3b/subgroup_metrics.json`
- Final full rerun summary:
  - `outputs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized/evaluation_summary.json`
- Final subgroup metrics:
  - `outputs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized/subgroup_metrics.json`
- Before/after comparison:
  - `outputs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized/comparison_vs_baseline.json`
  - `outputs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized/comparison_vs_baseline.md`

## Final Figures

- Overall before/after chart:
  - `outputs/final_packaging/figures/screenspot_v2_overall_before_after.png`
- Platform before/after chart:
  - `outputs/final_packaging/figures/screenspot_v2_platform_point_accuracy_before_after.png`
- Element-type before/after chart:
  - `outputs/final_packaging/figures/screenspot_v2_element_type_point_accuracy_before_after.png`
- Source-split before/after chart:
  - `outputs/final_packaging/figures/screenspot_v2_source_split_point_accuracy_before_after.png`

## Final Tables

- Core metrics summary table:
  - `outputs/final_packaging/tables/screenspot_v2_core_metrics_summary.md`
  - `outputs/final_packaging/tables/screenspot_v2_core_metrics_summary.csv`
- Point-accuracy breakdown table:
  - `outputs/final_packaging/tables/screenspot_v2_point_accuracy_breakdowns.md`
  - `outputs/final_packaging/tables/screenspot_v2_point_accuracy_breakdowns.csv`

## Key Historical Reports Used In The Final Story

- Proposal:
  - `docs/gui_grounding_project_proposal.docx`
- Qwen realignment:
  - `docs/blueprint_realignment_qwen_backbone.md`
- Qwen medium export quality:
  - `docs/qwen_medium_candidate_export_and_quality_report.md`
- Initial clean held-out eval:
  - `docs/screenspot_v2_clean_heldout_eval.md`
- Failure analysis:
  - `docs/screenspot_v2_failure_analysis_and_prompt_refinement.md`
- Full coordinate-refined rerun:
  - `docs/screenspot_v2_full_rerun_coordinate_refinement.md`
- Reward/reranker exploration:
  - `docs/stage5_learned_reranker_results.md`
  - `docs/stage5c_reranker_feature_upgrade.md`
  - `docs/stage6a_dpo_style_preference_optimization.md`
  - `docs/stage6a_5_preference_target_redesign.md`

## Packaging Manifest

- Generated packaging manifest:
  - `outputs/final_packaging/packaging_manifest.json`
