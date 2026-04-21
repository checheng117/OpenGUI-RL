# Speaking Script — Cross-Website GUI Grounding
10 minutes · estimated time per section in brackets

---

## Slide 1 · Title + Core Thesis (~1 min)

Hi everyone. My project is called Cross-Website GUI Grounding with Verifiable Reward Optimization — this is my final project for CSC6129 Reinforcement Learning.

The core question is: can we train a model to look at a webpage screenshot, read a natural-language instruction, and accurately locate and interact with the right UI element?

We ran experiments across three benchmarks and came away with three main findings. First, hybrid OCR/DOM candidate augmentation is critical for strong performance on Mind2Web — it's not a marginal gain, it's what makes the system work at all. Second, point-first grounding transfers well across held-out benchmarks without needing matched candidate structure. Third, reward-based reranking is a conditional gain — it helps when the baseline is weak enough to leave recoverable headroom, but the benefit shrinks sharply once Stage A becomes strong.

---

## Slide 2 · Problem Framing (~1 min)

Let me define the task clearly.

The input is a screenshot plus a natural-language instruction — something like "click the search button." The output is a predicted click point, a bounding box, and an action type: click, type, or select.

An important scope note: we study single-step grounding only. This is not a full browser agent — there's no multi-step planning, no execution loop, no recovery from errors. We isolate the perception-and-selection layer.

This task is a natural fit for reinforcement learning because the reward is fully verifiable. We can check automatically: did the predicted click point land inside the target element? Does the bounding box overlap with ground truth at IoU ≥ 0.5? Is the action type correct? These are all deterministic checks. So we frame it as a contextual bandit — the screenshot and instruction are the context, the grounded action is the action, and the reward is computed from these rules.

---

## Slide 3 · System Architecture (~1 min)

The system has two stages.

Stage A is the grounding model, built on Qwen2.5-VL 3B. It takes a screenshot and instruction as input, and predicts the click point and bounding box. In hybrid mode, it also receives compact OCR/DOM-style candidate anchors and is supervised on an auxiliary candidate slot target alongside the grounding output.

Stage B is a reranker. Stage A generates k equals 4 candidates, and a learned reward model selects the best one.

The project answers two design questions: how strong can Stage A become, and when does Stage B reranking still add meaningful value on top of that?

---

## Slide 4 · Three Benchmarks (~1 min)

We use three benchmarks, each serving a different role.

Mind2Web is our primary supervised benchmark. We train Stage A here, build the hybrid augmentation, run Stage B candidate generation and reranker training, and analyze recoverable headroom. It has three test splits: test_task tests new tasks on seen websites, test_website tests on unseen websites, and test_domain tests on unseen domains. These three splits let us measure how quickly generalization degrades in open-web settings.

ScreenSpot-v2 is our primary held-out benchmark. It doesn't touch training at all — it's purely for evaluating transfer.

VisualWebBench is a supplementary transfer check. We use it to test whether our conclusions hold under a different benchmark protocol.

---

## Slide 5 · Mind2Web Hybrid Breakthrough (~1.5 min)

The most important result on Mind2Web is the hybrid candidate augmentation breakthrough.

We started with a pure visual Stage A. We fixed a geometry-collapse bug first, so the comparison is fair. Even after that fix, pure visual Stage A only reached an internal point accuracy of 0.04 and a mean IoU of 0.006 — essentially no effective localization. Screenshot-only supervision couldn't give the model enough structured evidence to ground UI elements reliably.

After adding OCR/DOM candidate anchors, the result changed dramatically. Internal point accuracy jumped from 0.04 to 0.79, and mean IoU went from 0.006 to 0.73.

On the official cached subset, the results across three splits are: 95% click-point accuracy on test_task, 85% on test_website, and 89.5% on test_domain. The average gain over pure visual is plus 88 percentage points on element accuracy, click-point accuracy, and IoU@0.5.

This isn't a fine-tuning margin — it's the difference between a system that can and cannot ground UI elements at all. The candidate structure carries semantic evidence that the model simply cannot recover from the screenshot alone.

---

## Slide 6 · Stage B Reranking (~1 min)

The Stage B conclusion is more nuanced and worth being careful about.

We measure oracle best-of-k headroom — that's the theoretical upper bound on what reranking can recover from a candidate pool, assuming a perfect selector.

On the historical weak Stage A, oracle point gain was plus 10 points for both k equals 4 and k equals 8. There was real recoverable headroom in the pool.

After rebuilding the candidate pool on the strong hybrid Stage A, the same k equals 4 oracle gain shrank to plus 5 points — cut in half.

The takeaway is: the stronger Stage A becomes, the fewer recoverable errors remain in the pool, and the less reranking can do. Investing in Stage A representation delivers more reliable gains than scaling up Stage B candidate diversity. Reranking is not a universal win — it's a second-stage mechanism that depends on the baseline leaving room to recover.

---

## Slide 7 · ScreenSpot-v2 (~1.5 min)

ScreenSpot-v2 is our cleanest held-out benchmark, but we hit a serious problem early on — initial results were near zero.

After investigation, we found a coordinate-frame mismatch. The model was predicting coordinates in resized-image space, but the evaluation was scoring against original-image coordinates. The two systems were not aligned. Fixing this mismatch was what made the held-out benchmark meaningful.

After the fix, the method ladder looks like this. Our reproduced plain-Qwen public baseline reached 75.6% point accuracy. Point-native decoupled inference improved that to 77.4%. Adding the dual-path verifier brought it to 77.9% point accuracy and 17.2% IoU@0.5 — a plus 12 point IoU improvement over the baseline.

The verifier overhead is only 0.000167 seconds per image, so it's essentially free in terms of runtime.

One interesting subgroup finding: text elements outperform icon elements by nearly 20 percentage points in point accuracy, which suggests localization on icon-based targets still has room to improve.

---

## Slide 8 · VisualWebBench (~1 min)

VisualWebBench sharpened our transfer claim by revealing where it breaks.

This benchmark uses an anonymous 8-box protocol — no semantic candidate labels, just numbered boxes. That's a different protocol from Mind2Web.

Point-native inference still transferred well here: 87.2% official choice accuracy, about 9 points above structured screenshot-only. Dual-path verifier matched that roughly.

But when we took the Mind2Web hybrid candidate-aware model and applied it directly, performance collapsed to 23.8%. The candidate slot semantics simply don't exist in an anonymous-box protocol — the model received structurally mismatched inputs and failed.

So the refined claim is: point-first inference is robust across protocols because it doesn't rely on semantic candidate labels. Candidate-aware methods require protocol compatibility to transfer.

---

## Slide 9 · Final Takeaways (~1 min)

Let me close with the key lessons.

Three things worked reliably. Hybrid OCR/DOM candidate augmentation delivered plus 88 points on Mind2Web — that's a decisive result, not a marginal one. Point-first grounding transferred well to both ScreenSpot-v2 and VisualWebBench without needing matched candidate structure. The dual-path verifier gave consistent IoU improvement at negligible runtime cost.

Two things were conditional or failed. Reward-based reranking is valuable when the baseline is weak, but saturates after strong Stage A supervision. Candidate-aware methods fail under mismatched benchmark protocols.

The one-sentence summary of this project: strong representation first, reward as a conditional second-stage gain. The project doesn't claim RL always wins — it claims that verifiable reward adds measurable value under specific conditions, and we've been honest about where those conditions stop holding.

Thank you. Happy to take questions.

---

*Total ~950 words · approximately 9–10 minutes at a comfortable presentation pace.*
