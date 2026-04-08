# Cross-Website GUI Grounding with Verifiable Reward Optimization

CSC6129 Reinforcement Learning Final Project Report Draft  
Student: Che Cheng

## Abstract

This project studies instruction-conditioned GUI grounding as a reinforcement-learning-relevant multimodal decision problem: given a screenshot and a natural-language instruction, predict a UI action and localize the target element. The original plan emphasized supervised training plus verifiable-reward-driven reranking and preference optimization. That early path produced useful engineering artifacts and clarified the reward formulation, but it did not deliver the strongest empirical result. The project was later realigned to a Qwen-first multimodal backbone, with evaluation centered on clean held-out GUI grounding rather than legacy surrogate optimization. The decisive breakthrough came from failure analysis on ScreenSpot-v2: the model was already producing mostly parseable structured outputs, but many predictions were scored in the wrong coordinate frame. After explicitly predicting in the resized model-view frame and mapping coordinates back to the original screenshot, full held-out ScreenSpot-v2 point accuracy improved from `0.0849` to `0.7099` on `1272/1272` samples, with broad gains across desktop, web, and mobile. The final contribution is therefore an honest multimodal LLM engineering story: reward optimization remains part of the project framing and exploration, but the strongest final evidence is benchmark-driven debugging and coordinate-frame refinement for Qwen-first GUI grounding.

## 1. Introduction and Motivation

Modern computer-use agents depend on a perception layer that can interpret screenshots, understand user intent, and ground the next action to the correct UI element. This project focuses on that single-step GUI grounding problem rather than long-horizon browser-agent execution. The goal is practical and measurable: take a screenshot plus instruction and predict a structured action object centered on `bbox_proposal`, `click_point`, and `action_type`.

This is a strong reinforcement learning course project because the task admits a clean verifiable reward. If the model predicts the correct element, lands the click inside the target box, and outputs a valid action type, reward can be computed automatically. That makes the problem fit a contextual-bandit view of RL: the context is the screenshot and instruction, the action is the grounded UI action, and reward is derived from correctness and localization quality.

The final report is intentionally honest about the project arc. Early surrogate and reward-optimization stages existed and were useful, but the final strongest result came from a Qwen-first held-out evaluation pipeline and a debugging-led refinement to the coordinate system used for prediction and scoring.

## 2. Problem Formulation and RL Framing

### 2.1 Single-step decision problem

For each example:

- Context `x`: screenshot image `I` and instruction `u`
- Action `a`: structured GUI action with `bbox_proposal`, `click_point`, and `action_type`
- Reward `r`: deterministic function of spatial correctness and output validity

This is naturally framed as a contextual bandit rather than a full sequential RL task. The project does not claim to solve long-horizon web navigation, memory, recovery from mistakes, or multi-step environment interaction. It isolates the grounding/action-prediction step that upstream browser agents rely on.

### 2.2 Verifiable reward

The project proposal defined reward using components such as:

- element correctness
- click-point inclusion
- IoU overlap with the target box
- action-type correctness
- penalties for malformed outputs

This reward formulation motivated the early reranking and preference-learning stages. Even though those stages were not the final main result, they remain important because they define the RL relevance of the project and shaped the evaluation mindset used later in held-out analysis.

## 3. Datasets and Benchmarks

### 3.1 Training and development data

The broader project used Mind2Web-style screenshot-instruction-action data for early supervised and candidate-generation work. This supported:

- Stage A supervised baseline experiments
- candidate generation
- reward-scored reranking experiments
- Qwen structured-output validation on real screenshot tasks

### 3.2 Primary final benchmark

The final primary benchmark is **ScreenSpot-v2 clean held-out evaluation**. This became the authoritative benchmark for the final package because it is a clean, explicit grounding benchmark with held-out desktop, web, and mobile examples. The final report therefore treats ScreenSpot-v2 as the strongest evidence source for the final claim.

Dataset facts used in the final benchmark:

- dataset source: `lscpku/ScreenSpot-v2`
- split: `test`
- total evaluated samples: `1272`
- platform counts:
  - desktop: `334`
  - mobile: `501`
  - web: `437`
- element types:
  - text: `718`
  - icon: `554`

## 4. Method Overview

### 4.1 Final primary method

The final primary method is:

**Qwen-first GUI grounding with coordinate-frame refinement**

Core prediction semantics:

- `bbox_proposal`
- `click_point`
- `action_type`

The Qwen backbone receives a screenshot and instruction and emits structured JSON-like predictions. The critical refinement is that the model predicts coordinates in the resized model-view frame used internally by the vision-language model, and those coordinates are then mapped back to the original screenshot frame before evaluation.

### 4.2 Why this refinement matters

If the model internally reasons over a resized image but outputs coordinates interpreted as if they belonged to the original screenshot, predictions can appear catastrophically wrong despite being semantically sensible relative to the model-view image. This project discovered that this mismatch, not generic output instability, was the dominant reason for weak initial held-out results.

## 5. Early Stages and Negative Findings

The early project stages are important historical context and should not be erased.

### 5.1 CLIP-grid supervised baseline

The first executable baseline used a CLIP-grid surrogate path because Qwen runtime was not yet stable in the local environment. This baseline established a runnable training/evaluation stack, but the representation was coarse and not adequate as a final grounding solution.

Representative debug metrics from the saved Stage 3 report:

- `val_action_acc = 0.75`
- `val_grid_acc = 0.50`
- `val_point_acc = 0.00`
- `val_mean_iou = 0.0071`

These numbers were useful for pipeline validation but not persuasive as a final project result.

### 5.2 Reward-based reranking exploration

The project then explored verifiable-reward-driven candidate reranking and lightweight preference optimization on CLIP-grid-generated candidate pools.

What the saved artifacts show:

- Stage 5 learned reranker on a small pool produced **no measurable improvement**
- Stage 5c feature upgrades yielded only **very small positive gains**
  - full-pool mean reward gain: `+0.0001657`
  - headroom-subset mean reward gain: `+0.0005340`
- Step 6A DPO-style preference optimization did **not** beat Step 5c
- Step 6A.5 preference-target redesign also did **not** beat Step 5c

Interpretation:

- reward optimization was a valid and RL-relevant exploration path
- the reranking experiments were informative about data quality and headroom
- but they did not become the strongest empirical claim of the final project

This is why the final package does not overstate reward optimization success.

## 6. Qwen-First Realignment

The project was later realigned to the original multimodal objective: use a real Qwen vision-language model as the primary path rather than keeping the surrogate backbone as the centerpiece.

Key milestones from the saved repository evidence:

- Qwen-first runtime path was unblocked using the local cached model plus mirror-backed runtime support
- real Qwen single-sample inference completed successfully
- a medium-scale Qwen candidate export on `50` real Mind2Web train samples completed with:
  - `50/50` successful sample-level runs
  - `200/200` parseable structured outputs
  - `199/200` valid bbox outputs
  - `199/200` valid click-point outputs

This was important because it showed the Qwen-first path was operationally stable enough to support held-out evaluation.

## 7. Held-Out Failure Analysis

### 7.1 Initial full ScreenSpot-v2 held-out result

The first authoritative full ScreenSpot-v2 held-out run used Qwen-first evaluation but interpreted outputs directly in the original screenshot frame.

Overall baseline result:

| Metric | Value |
| --- | ---: |
| Evaluated samples | 1272 |
| Point accuracy | 0.0849 |
| IoU@0.5 | 0.0102 |
| Mean IoU | 0.0196 |
| Action-type validity | 0.9992 |
| Parseable output rate | 0.9992 |
| Valid bbox rate | 0.9992 |
| Valid click-point rate | 0.9992 |

This result is crucial because it revealed a mismatch between **structural output quality** and **grounding quality**. The model was almost always producing parseable outputs, but the spatial predictions were usually wrong.

### 7.2 Why the baseline looked bad

The failure analysis showed:

- only `1/1272` prediction was non-parseable
- most errors were parseable but spatially wrong
- desktop was much better than web/mobile:
  - desktop point accuracy: `0.3084`
  - mobile point accuracy: `0.0060`
  - web point accuracy: `0.0046`

This asymmetry strongly suggested a coordinate-scale issue rather than total model failure. The saved failure analysis then quantified systematic coordinate shrinkage:

- mobile median click ratios around `0.38-0.41`
- web median click ratios around `0.37`
- desktop ratios larger but still compressed

The decisive counterfactual test was even stronger:

- if the saved baseline predictions were treated as belonging to the Qwen resized-image frame and mapped back to the original image size, overall point accuracy would rise from `0.0849` to `0.6863`
- under the same reinterpretation, IoU@0.5 would rise from `0.0102` to `0.1321`

That analysis identified the dominant failure mode.

## 8. Coordinate-Frame Refinement

The final refinement was intentionally narrow:

1. tell Qwen to emit coordinates in the **resized model-view frame**
2. parse and clamp coordinates in that frame
3. map them back to the **original screenshot frame** before scoring/export

This change preserved the rest of the pipeline:

- no new model training
- no reranker redesign
- no switch away from Qwen
- no pipeline replacement

### 8.1 Diagnostic reevaluation before the full rerun

Balanced `180`-sample reevaluation:

- point accuracy: `0.1222 -> 0.6833`
- IoU@0.5: `0.0167 -> 0.1778`
- mean IoU: `0.0327 -> 0.2259`

Balanced `360`-sample reevaluation:

- point accuracy: `0.1194 -> 0.6722`
- IoU@0.5: `0.0139 -> 0.1722`
- mean IoU: `0.0298 -> 0.2281`

Those results justified a full benchmark rerun.

## 9. Final Results

The strongest final result is the full ScreenSpot-v2 rerun with coordinate-frame refinement enabled via [`configs/eval/screenspot_v2_qwen2_5_vl_3b_model_resized_full.yaml`](../configs/eval/screenspot_v2_qwen2_5_vl_3b_model_resized_full.yaml).

### 9.1 Overall benchmark result

| Metric | Before | After | Delta |
| --- | ---: | ---: | ---: |
| Evaluated samples | 1272 | 1272 | 0 |
| Point accuracy | 0.0849 | 0.7099 | +0.6250 |
| IoU@0.5 | 0.0102 | 0.1682 | +0.1580 |
| Mean IoU | 0.0196 | 0.2404 | +0.2209 |
| Action-type validity | 0.9992 | 1.0000 | +0.0008 |
| Parseable output rate | 0.9992 | 1.0000 | +0.0008 |
| Valid bbox rate | 0.9992 | 1.0000 | +0.0008 |
| Valid click-point rate | 0.9992 | 1.0000 | +0.0008 |

![Overall before/after metrics](../outputs/final_packaging/figures/screenspot_v2_overall_before_after.png)

### 9.2 Platform breakdown

| Platform | Point Acc Before | Point Acc After |
| --- | ---: | ---: |
| desktop | 0.3084 | 0.6976 |
| mobile | 0.0060 | 0.7445 |
| web | 0.0046 | 0.6796 |

![Platform point accuracy](../outputs/final_packaging/figures/screenspot_v2_platform_point_accuracy_before_after.png)

### 9.3 Element-type breakdown

| Element type | Point Acc Before | Point Acc After |
| --- | ---: | ---: |
| text | 0.0947 | 0.8245 |
| icon | 0.0722 | 0.5614 |

![Element-type point accuracy](../outputs/final_packaging/figures/screenspot_v2_element_type_point_accuracy_before_after.png)

### 9.4 Key source splits

| Source | Point Acc Before | Point Acc After |
| --- | ---: | ---: |
| windows | 0.6289 | 0.7547 |
| macos | 0.0204 | 0.6939 |
| ios | 0.0042 | 0.7353 |
| android | 0.0047 | 0.8057 |

![Key source-split point accuracy](../outputs/final_packaging/figures/screenspot_v2_source_split_point_accuracy_before_after.png)

### 9.5 Main interpretation

The final results support a precise conclusion:

- the weak initial held-out performance was caused mainly by **coordinate-frame mismatch**
- the model was already structurally stable enough for evaluation
- once coordinates were emitted and interpreted in the correct frame, held-out grounding quality improved dramatically

This is a stronger and more defensible project claim than saying reward optimization alone solved the task.

## 10. Discussion and Limitations

### 10.1 What the project does show

The final package shows that:

- a Qwen-first multimodal grounding system can be packaged into a clean held-out benchmark pipeline
- structured GUI action outputs can be made highly reliable
- careful failure analysis can uncover a dominant bug-like failure mode with benchmark-scale consequences
- debugging the coordinate frame can matter more than adding extra optimization complexity

### 10.2 What the project does not show

The final package does **not** claim:

- full long-horizon web-agent competence
- success on sequential browser execution
- that reward optimization became the strongest final empirical contributor
- that the coordinate refinement has been validated across every possible GUI benchmark

### 10.3 Why reward optimization still belongs in the story

Reward optimization remains part of the story for three reasons:

1. it grounds the project in RL-relevant verifiable-reward reasoning
2. it shaped the candidate/action schema and evaluation discipline
3. it provided negative results that clarified where empirical leverage was and was not found

That is valuable in a course project. Honest negative findings are part of the final result.

## 11. Conclusion

This project began as a multimodal GUI grounding system with verifiable-reward-driven improvement. Early surrogate and reranking stages established the RL framing and produced useful engineering infrastructure, but they did not yield the strongest empirical gain. The project then returned to the intended Qwen-first path, built a stable held-out evaluation pipeline, and discovered that the main obstacle on ScreenSpot-v2 was coordinate-frame mismatch rather than general output instability. Fixing that mismatch transformed full held-out performance from weak to strong: point accuracy improved from `0.0849` to `0.7099` on `1272` clean held-out samples, with especially large gains on web and mobile. The final project contribution is therefore a benchmark-driven multimodal engineering result with explicit RL relevance, honest negative findings, and a clear debugging-led breakthrough.

## References and Key Artifacts

Primary repo artifacts used for this final report:

- [`docs/gui_grounding_project_proposal.docx`](./gui_grounding_project_proposal.docx)
- [`docs/stage5_learned_reranker_results.md`](./stage5_learned_reranker_results.md)
- [`docs/stage5c_reranker_feature_upgrade.md`](./stage5c_reranker_feature_upgrade.md)
- [`docs/stage6a_dpo_style_preference_optimization.md`](./stage6a_dpo_style_preference_optimization.md)
- [`docs/stage6a_5_preference_target_redesign.md`](./stage6a_5_preference_target_redesign.md)
- [`docs/qwen_medium_candidate_export_and_quality_report.md`](./qwen_medium_candidate_export_and_quality_report.md)
- [`docs/screenspot_v2_clean_heldout_eval.md`](./screenspot_v2_clean_heldout_eval.md)
- [`docs/screenspot_v2_failure_analysis_and_prompt_refinement.md`](./screenspot_v2_failure_analysis_and_prompt_refinement.md)
- [`docs/screenspot_v2_full_rerun_coordinate_refinement.md`](./screenspot_v2_full_rerun_coordinate_refinement.md)

Authoritative final benchmark artifacts:

- [`outputs/screenspot_v2_eval_qwen2_5_vl_3b/evaluation_summary.json`](../outputs/screenspot_v2_eval_qwen2_5_vl_3b/evaluation_summary.json)
- [`outputs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized/evaluation_summary.json`](../outputs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized/evaluation_summary.json)
- [`outputs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized/comparison_vs_baseline.json`](../outputs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized/comparison_vs_baseline.json)
