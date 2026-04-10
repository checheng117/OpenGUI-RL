# Cross-Website GUI Grounding with Verifiable Reward Optimization

CSC6129 Reinforcement Learning Final Project Report Draft  
Student: Che Cheng

Accuracy note for handoff: if any exact number in this draft conflicts with the later handoff materials, prefer [`docs/standalone_quantitative_evaluation_section.md`](standalone_quantitative_evaluation_section.md) and [`outputs/quantitative_metrics_suite/summary.json`](../outputs/quantitative_metrics_suite/summary.json), which were recomputed from saved evaluation artifacts on `2026-04-10`.

## Abstract

This project studies instruction-conditioned GUI grounding as a single-step multimodal decision problem: given a screenshot and a natural-language instruction, predict the target UI element, click location, and action type. The original proposal emphasized a supervised baseline plus verifiable-reward-driven candidate reranking. That framing remained useful throughout the project, but the final evidence is more nuanced than a simple "RL always helps" story.

Across the completed benchmarks, three findings emerged. First, on Mind2Web, pure screenshot-only supervision was not enough: after fixing a geometry-collapse bug, Stage A still reached only `0.0375` internal-validation point accuracy and `0.0061` mean IoU. Adding semantically informative OCR/DOM-style candidate augmentation transformed Stage A into a strong supervised baseline, reaching `0.7875` point accuracy and `0.7314` mean IoU internally, with cached official-subset point accuracy of `0.9500 / 0.8500 / 0.8947` on `test_task / test_website / test_domain`. Second, on ScreenSpot-v2, failure analysis revealed a coordinate-frame mismatch between the model-resized image and the original screenshot. Fixing that mismatch produced the first credible held-out baseline, after which a point-native decoupled inference path exceeded the reproduced public plain-Qwen baseline (`0.7736` vs `0.7563` point accuracy) and a dual-path verifier improved slightly further to `0.7791`. Third, supplementary transfer on VisualWebBench showed that point-first grounding transfers well (`0.8721` official choice accuracy vs `0.7888` for structured screenshot-only), but Mind2Web-style hybrid candidate-aware transfer does not transfer unchanged to an anonymous 8-box protocol (`0.2384` official choice accuracy).

The final conclusion is therefore scoped and evidence-based: semantically informative candidate augmentation is critical when the benchmark exposes meaningful candidate structure; point-first grounding is a strong transferable inference strategy; reward-based reranking helps when the baseline leaves real candidate-pool headroom; but once strong supervision saturates most recoverable cases, reranking gains diminish sharply and cease to be a robust default improvement.

## 1. Introduction and Motivation

Modern browser and computer-use agents depend on a perception layer that can read a screenshot, understand an instruction, and ground the next action to the correct UI target. This project isolates that layer as a single-step GUI grounding problem rather than attempting full long-horizon web navigation. The core output contract is a structured action object containing `click_point`, `bbox_proposal`, and `action_type`.

This is a good reinforcement learning project because the task admits a clean verifiable reward. Once a target element and action annotation are available, correctness can be checked automatically through element match, click-inside-target, bounding-box overlap, action-type correctness, and formatting validity. That makes the task naturally fit a contextual-bandit-like formulation: context is the screenshot and instruction, action is the grounded UI action, and reward is deterministic.

The original proposal asked two linked questions:

1. How strong can a supervised multimodal GUI grounding baseline become?
2. When does verifiable-reward-driven reranking still add value on top of that baseline?

The finished project answers both questions, but not symmetrically. The strongest final positive result is not heavy RL or preference optimization. It is a combination of better representation, benchmark debugging, and stronger inference structure. Reward-based reranking remains scientifically important, but its benefits are conditional on residual candidate-pool headroom.

## 2. Problem Formulation and RL Framing

### 2.1 Single-step decision problem

For each example:

- Context `x`: screenshot `I`, instruction `u`, and optional candidate cues `c`
- Action `a`: grounded GUI action with `click_point`, `bbox_proposal`, and `action_type`
- Reward `r`: deterministic score derived from spatial correctness and output validity

This project intentionally stops at the single-step grounding layer. It does not claim to solve long-horizon planning, recovery from execution errors, or full browser-agent interaction loops.

### 2.2 Verifiable reward

The reward framing from the proposal remained stable throughout the project:

- element correctness
- click-point inclusion
- IoU overlap
- action-type correctness
- penalties for malformed outputs

That reward was used directly in Mind2Web Stage B candidate labeling and reranker training. It also shaped the project's evaluation style more broadly: results were assessed not just by parseability, but by whether outputs were spatially correct and recoverable.

### 2.3 Why heavy RL was not the final strongest result

The final evidence does not support a claim that stronger RL machinery was the main source of performance. Two reasons are clear from the completed experiments:

1. The largest gains came from fixing upstream representation and inference problems: Mind2Web hybrid augmentation and ScreenSpot-v2 coordinate-frame correction plus point-native decoding.
2. Reranking helps only when there is enough headroom in the candidate pool. Once Stage A becomes strong enough to solve most recoverable cases directly, the reward model has too few true recovery opportunities to remain a reliable default improvement.

So the RL framing is honest and relevant, but the finished story is about when verifiable reward matters, not about claiming that more RL always wins.

## 3. Datasets and Benchmarks

| Benchmark | Role in final story | What it tested |
| --- | --- | --- |
| Mind2Web | Primary blueprint benchmark | Stage A supervision, hybrid candidate augmentation, Stage B candidate generation, deterministic reward, learned reranking, and headroom analysis across `test_task`, `test_website`, and `test_domain` |
| ScreenSpot-v2 | Primary held-out benchmark | Cross-platform GUI grounding under a clean same-protocol evaluation, including baseline reproduction, coordinate-frame debugging, point-native inference, and dual-path verification |
| VisualWebBench | Supplementary transfer benchmark | Whether the main Mind2Web and ScreenSpot-v2 findings transfer to a broader grounding-compatible benchmark with an anonymous 8-box protocol |

Mind2Web anchored the proposal-facing supervised-plus-reward story. ScreenSpot-v2 provided the cleanest held-out evaluation of transfer-ready grounding behavior. VisualWebBench served as the supplementary check on what transfers, what saturates, and what fails under a protocol mismatch.

## 4. Methods

### 4.1 Stage A pure-visual supervised baseline

The first serious Mind2Web Stage A path used screenshot-only supervision. A geometry-collapse issue was identified and fixed, so the final pure-visual comparison is not against a broken baseline. Even after that fix, however, Stage A remained weak on actual localization. This mattered because it established that pure screenshot supervision was not enough for the project’s original claim.

### 4.2 Stage A hybrid candidate-aware baseline

The decisive Mind2Web Stage A improvement was to augment the screenshot with compact OCR/DOM-style candidate anchors and supervise an auxiliary `candidate_slot` target. The important detail is not merely "more tokens" or "more slots." The important detail is that the candidate structure carried semantically meaningful evidence:

- DOM bounding boxes
- compact attribute text
- recovered cleaned-HTML node text
- deterministic heuristic ranking of candidate anchors

At evaluation time, a valid predicted candidate slot anchored the grounded element, box, and click point. This created the first strong supervised grounding baseline in the repository.

### 4.3 Stage B candidate generation, deterministic reward, and reranking

Stage B kept the same structured action contract and added:

1. small top-k candidate pools exported from the Stage A model
2. deterministic verifiable reward labels per candidate
3. a lightweight learned reranker over auditable engineered candidate features

This stage was first run on weaker Stage A foundations and later rebuilt on top of the strong hybrid Stage A baseline. That rebuild is crucial because it directly tests whether reward-based reranking still helps after strong supervision closes most of the gap.

### 4.4 ScreenSpot-v2 Qwen-first inference path

The ScreenSpot-v2 line evolved in three steps:

1. coordinate-frame refinement
2. point-native decoupled decoding
3. dual-path candidate generation plus lightweight verifier

The coordinate-frame fix was not a new model. It corrected a mismatch between the resized image frame used by Qwen internally and the original screenshot frame used for scoring. Once that issue was fixed, the project could meaningfully compare stronger inference strategies on held-out data. The point-native path then made click prediction primary rather than treating it as a byproduct of box decoding, and the dual-path verifier combined structured and point-native candidates under a lightweight selection rule.

## 5. Main Experimental Results

### 5.1 Mind2Web Stage A: pure visual was insufficient, hybrid was the breakthrough

The finished Stage A comparison is the clearest positive result on Mind2Web.

| Setting | Pure visual point acc | Hybrid point acc | Pure visual mean IoU | Hybrid mean IoU |
| --- | ---: | ---: | ---: | ---: |
| Internal validation | `0.0375` | `0.7875` | `0.0061` | `0.7314` |
| `test_task` cached subset | `0.0000` | `0.9500` | `0.0057` | `0.9500` |
| `test_website` cached subset | `0.0000` | `0.8500` | `0.0031` | `0.8275` |
| `test_domain` cached subset | `0.0526` | `0.8947` | `0.0198` | `0.8947` |

This is not a marginal improvement. It is the transition from an inadequate supervised baseline to a credible strong baseline.

![Mind2Web Stage A comparison](../outputs/final_packaging/figures/mind2web_stageA_pure_vs_hybrid.png)

### 5.2 Mind2Web Stage B on weaker Stage A: reranking helped when headroom existed

Before rebuilding Stage B on top of the hybrid Stage A baseline, the best source-aware reranker on expanded pools produced consistent positive reward gains on the official split pools:

- `test_task`: `+0.0055`
- `test_website`: `+0.0211`
- `test_domain`: `+0.0202`

Those gains were not huge, but they were real. They established the core conditional claim that reward-based reranking can help when the supervised baseline is weak enough to leave meaningful recoverable headroom.

### 5.3 Mind2Web Stage B rebuild on hybrid Stage A: headroom shrank and default gains disappeared

Rebuilding Stage B on top of the strong hybrid Stage A changed the picture sharply.

| Split | Old headroom frac | Rebuild headroom frac | Old reward gain | Rebuild reward gain |
| --- | ---: | ---: | ---: | ---: |
| `test_task` | `0.2000` | `0.0500` | `+0.0055` | `-0.0400` |
| `test_website` | `0.2000` | `0.1500` | `+0.0211` | `-0.2588` |
| `test_domain` | `0.1579` | `0.0526` | `+0.0202` | `+0.0737` |

Across the official split pools together:

- headroom pools dropped from `11 / 59` to `5 / 59`
- mean split reward moved from `1.7496` for hybrid first-choice to `1.6745` after reranking

The rebuild therefore answers the proposal question directly: reranking still has residual value in narrow cases, but it is no longer a robust default improvement once Stage A already resolves most recoverable cases.

![Mind2Web Stage B comparison](../outputs/final_packaging/figures/mind2web_stageB_headroom_old_vs_hybrid_rebuild.png)

### 5.4 ScreenSpot-v2: coordinate fix, point-native inference, and a modest dual-path gain

ScreenSpot-v2 produced the strongest held-out inference story in the project.

The turning point was the coordinate-frame mismatch discovery. The same model that initially achieved only `0.0849` point accuracy rose to `0.7099` after the coordinate-frame refinement. That created a credible structured baseline. From there, stronger inference structure improved further:

| Method | Point accuracy | IoU@0.5 | Mean IoU |
| --- | ---: | ---: | ---: |
| Reproduced public plain-Qwen baseline | `0.7563` | `0.0519` | `0.1327` |
| Structured coordinate-refined path | `0.7099` | `0.1682` | `0.2404` |
| Point-native decoupled path | `0.7736` | `0.0967` | `0.1912` |
| Dual-path verifier | `0.7791` | `0.1722` | `0.2520` |

The right comparison here is not just "before vs after bug fix." The stronger final claim is that point-native grounding exceeded the reproduced public same-protocol baseline, and dual-path verification improved slightly further.

![ScreenSpot-v2 comparison](../outputs/final_packaging/figures/screenspot_v2_before_after_and_method_comparison.png)

### 5.5 VisualWebBench: point-first transfer held, hybrid candidate transfer did not

VisualWebBench was intentionally supplementary and evaluation-only. No benchmark-specific tuning was added. The goal was to test transfer and scope.

| Method | Official choice acc | Point acc | Mean IoU |
| --- | ---: | ---: | ---: |
| Structured screenshot-only | `0.7888` | `0.6453` | `0.3206` |
| Point-native decoupled | `0.8721` | `0.7946` | `0.3435` |
| Dual-path verifier | `0.8682` | `0.7965` | `0.3312` |
| Mind2Web Stage A hybrid transfer | `0.2384` | `0.2384` | `0.2384` |

Three important things happened here:

1. Point-native transfer remained strong.
2. Dual-path no longer beat point-native overall; the remaining combination headroom was tiny.
3. The Mind2Web hybrid checkpoint transferred poorly because VisualWebBench exposes eight anonymous option boxes rather than semantically informative OCR/DOM candidates.

![VisualWebBench comparison](../outputs/final_packaging/figures/visualwebbench_method_comparison.png)

## 6. Cross-Benchmark Synthesis

### 6.1 Strong positive findings

**Hybrid OCR/DOM candidate augmentation works on Mind2Web when the candidate structure is informative.**  
This is one of the strongest results in the repository. The hybrid Stage A baseline is not merely better than pure visual; it changes the scientific conclusion from "Stage A is weak" to "Stage A can be strong when the representation exposes semantically useful candidate structure."

**Point-first grounding is a strong transferable inference strategy.**  
ScreenSpot-v2 and VisualWebBench both support this. On ScreenSpot-v2, point-native decoding improved from the structured coordinate-refined path to `0.7736` point accuracy and exceeded the reproduced public baseline. On VisualWebBench, point-native raised official choice accuracy from `0.7888` to `0.8721`.

### 6.2 Scoped or conditional positive findings

**Reward-based reranking helps when headroom exists.**  
Earlier Mind2Web Stage B runs on weaker Stage A foundations produced real gains across all three official split pools. That result should be kept, but it should be described conditionally rather than universally.

**Dual-path combination still helps relative to weaker baselines.**  
On ScreenSpot-v2, dual-path verification improved over both the structured and point-native single paths, but only modestly over point-native (`0.7736 -> 0.7791`). On VisualWebBench, dual-path improved clearly over structured screenshot-only, but no longer exceeded point-native overall. This is exactly what a saturated-headroom story predicts.

### 6.3 Important negative findings

**Pure visual Stage A was not enough on Mind2Web.**  
This matters because it falsified the easy version of the proposal. A screenshot-only supervised pipeline did not become strong simply by training longer or fixing geometry.

**Reranking gains diminish sharply once Stage A becomes strong.**  
The rebuild on hybrid Stage A is the key evidence: official headroom shrank from `11/59` pools to `5/59`, and reranking lowered average official-split reward overall.

**Candidate-aware methods do not transfer unchanged across mismatched benchmark protocols.**  
Mind2Web-style hybrid transfer failed on VisualWebBench because the underlying candidate semantics changed. The lesson is not that candidate augmentation is unhelpful; it is that the gain depends on semantically informative candidate structure rather than arbitrary slot prompting.

### 6.4 Final claim language

The safest final wording is:

1. Hybrid OCR/DOM candidate augmentation is critical for strong supervised GUI grounding when the benchmark exposes semantically informative candidate structure.
2. Point-first grounding is a strong and transferable inference strategy across held-out GUI benchmarks.
3. Reward-based reranking helps substantially when the baseline is weak or imperfect enough to leave real candidate-pool headroom.
4. Once strong supervision saturates most recoverable headroom, reranking gains become sparse and inconsistent.
5. Candidate-aware methods do not transfer unchanged across benchmarks with mismatched candidate protocols.

## 7. Discussion, Negative Findings, and Limits

### 7.1 What the project did not show

The finished project does not support the following claims:

- that pure screenshot supervision alone is a strong general solution
- that reranking gives universal gains on top of a strong baseline
- that Mind2Web-style candidate-aware checkpoints transfer unchanged to any GUI benchmark
- that the project solves long-horizon browser automation

### 7.2 Why the project still aligns with the original proposal

The proposal asked for a supervised baseline, a verifiable reward, a reranking mechanism, and cross-benchmark evaluation. All of those were completed. What changed is the conclusion:

- the strongest baseline required hybrid candidate-aware supervision on Mind2Web
- the strongest held-out inference results came from coordinate-aware and point-first Qwen inference on ScreenSpot-v2
- reranking helped in the earlier, weaker-baseline regime but saturated after strong supervision

That is still a faithful realization of the proposal. It is simply an honest one.

### 7.3 Scope limits

- The Mind2Web official split comparisons are pilot-sized pools and cached subset readouts rather than a full-scale paper-style benchmark sweep.
- The project studies single-step grounding, not complete agent execution.
- VisualWebBench was used as a supplementary transfer analysis, not as a new training target.

None of these limits invalidate the conclusions above, but they do define their scope.

## 8. Conclusion

This project began as a proposal about multimodal GUI grounding with verifiable reward optimization and ended with a sharper, more defensible story. On Mind2Web, semantically informative OCR/DOM candidate augmentation was the critical ingredient for a strong supervised baseline. On ScreenSpot-v2, benchmark-driven debugging and point-first decoding produced the strongest held-out transfer results, culminating in a dual-path verifier at `0.7791` point accuracy. On VisualWebBench, point-first transfer held up, reranking-style gains saturated quickly, and Mind2Web-style candidate-aware transfer failed under an anonymous 8-box protocol.

The final synthesis is therefore not "reward optimization wins everywhere." It is more useful than that: strong supervision and strong inference structure do most of the heavy lifting, verifiable reward helps when headroom is real, and candidate-aware methods only transfer when their candidate semantics transfer too.
