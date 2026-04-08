# Final Presentation Outline

Suggested length: 12 slides  
Talk style: evidence-first, visually driven, honest about pivots

## Slide 1 - Title and One-Sentence Claim

**Slide title**  
Cross-Website GUI Grounding with Verifiable Reward Optimization

**Bullets**

- Goal: map screenshot + instruction to grounded UI action
- Final main result: fixing coordinate-frame mismatch transformed held-out performance on ScreenSpot-v2
- Final method: Qwen-first grounding with model-resized coordinate refinement

**Recommended visual**

- One large benchmark headline panel with:
  - point accuracy `0.0849 -> 0.7099`
  - ScreenSpot-v2 `1272/1272`

**Speaker note / intent**

- Open with the final answer, not the chronology. The audience should immediately know what worked.

## Slide 2 - Why GUI Grounding Matters

**Bullets**

- Browser/computer-use agents need reliable UI localization before they can act
- The project focuses on single-step grounding/action prediction, not full long-horizon execution
- This isolates a core bottleneck that can be measured cleanly

**Recommended visual**

- Two example screenshots with a highlighted target element
- Optional use one existing qualitative visualization from `outputs/qwen_multisample_validation_qwen2_5/visualizations/`

**Speaker note / intent**

- Establish practical importance and keep scope disciplined.

## Slide 3 - Why This Is RL-Relevant

**Bullets**

- Contextual-bandit framing:
  - state/context = screenshot + instruction
  - action = `bbox_proposal`, `click_point`, `action_type`
  - reward = correctness, IoU, click-inside-target, format validity
- Verifiable reward makes the problem RL-relevant without requiring full environment rollouts
- Early project stages used this reward for reranking and preference optimization

**Recommended visual**

- Simple pipeline diagram: input -> action prediction -> verifiable reward

**Speaker note / intent**

- Make the RL connection explicit early so the later engineering results still fit the course.

## Slide 4 - Original Plan vs Actual Project Arc

**Bullets**

- Initial plan:
  - supervised baseline
  - candidate generation
  - reward reranking / DPO / lightweight policy improvement
- What actually happened:
  - surrogate path helped build infrastructure
  - reward optimization gave limited gains
  - Qwen-first held-out debugging produced the main breakthrough

**Recommended visual**

- Simple 3-phase timeline:
  - surrogate/reward exploration
  - Qwen runtime unblock and realignment
  - ScreenSpot-v2 debugging and full rerun

**Speaker note / intent**

- Frame the pivot as disciplined project management, not loss of direction.

## Slide 5 - Early Exploration: What Did Not Become the Final Result

**Bullets**

- CLIP-grid Stage A baseline was runnable but too coarse
- Learned reranker on small candidate pools showed no measurable gain
- Feature-upgraded reranker produced only tiny gains:
  - full-pool reward gain `+0.0001657`
  - headroom-subset reward gain `+0.0005340`
- DPO-style and preference-redesign stages did not beat the best reranker baseline

**Recommended visual**

- Small summary table of Stage 5 / 5c / 6A / 6A.5 outcomes

**Speaker note / intent**

- Be direct: reward optimization was a real exploration path, but not the headline empirical success.

## Slide 6 - Qwen-First Realignment

**Bullets**

- Repository was realigned back to a real multimodal backbone
- Qwen runtime path was unblocked and stable enough for evaluation
- Medium-scale Qwen export evidence:
  - `50/50` successful samples
  - `200/200` parseable outputs
  - `199/200` valid bbox and click-point outputs

**Recommended visual**

- Short evidence box with the three numbers above
- Optional artifact screenshot from `outputs/single_inference_qwen_unblock/`

**Speaker note / intent**

- Show that by the time held-out evaluation started, runtime stability was largely solved.

## Slide 7 - Initial ScreenSpot-v2 Held-Out Result

**Bullets**

- First full clean held-out run completed on `1272/1272` samples
- Output structure was already highly stable:
  - parseable rate `0.9992`
  - valid bbox rate `0.9992`
  - valid click-point rate `0.9992`
- But grounding quality was weak:
  - point accuracy `0.0849`
  - web `0.0046`
  - mobile `0.0060`

**Recommended visual**

- Overall before/after figure, but initially highlight only the baseline bars
- Or show the core metrics table from `outputs/final_packaging/tables/screenspot_v2_core_metrics_summary.md`

**Speaker note / intent**

- The key setup for the debugging story is that structure was fine while localization was bad.

## Slide 8 - Failure Analysis: Coordinate Mismatch

**Bullets**

- Dominant failure mode was spatial, not formatting-related
- Predictions were systematically shrunk relative to the original screenshot
- Counterfactual diagnostic was decisive:
  - reinterpreting saved baseline outputs in resized-image coordinates would raise point accuracy to `0.6863`
  - IoU@0.5 would rise to `0.1321`

**Recommended visual**

- Diagram showing original screenshot frame vs model-resized frame
- A short callout box with the counterfactual numbers

**Speaker note / intent**

- This is the centerpiece of the talk. Spend time here.

## Slide 9 - The Fix

**Bullets**

- One focused refinement only:
  - prompt Qwen to emit coordinates in the resized model-view frame
  - scale predictions back to the original screenshot before evaluation
- No new training
- No reranker redesign
- No pipeline replacement

**Recommended visual**

- Before/after coordinate transform diagram
- Three-step arrow: predict in model frame -> clamp -> map back

**Speaker note / intent**

- Emphasize how small the code change was relative to the benchmark impact.

## Slide 10 - Diagnostic Reevaluations Before the Full Rerun

**Bullets**

- Balanced `180` samples:
  - point accuracy `0.1222 -> 0.6833`
- Balanced `360` samples:
  - point accuracy `0.1194 -> 0.6722`
- These subset checks justified running the full official rerun

**Recommended visual**

- Small two-row table for the 180 and 360 subset checks

**Speaker note / intent**

- Show that the final rerun was evidence-driven, not a blind full rerun.

## Slide 11 - Final Full Benchmark Result

**Bullets**

- Official full rerun: `1272/1272` samples
- Overall:
  - point accuracy `0.0849 -> 0.7099`
  - IoU@0.5 `0.0102 -> 0.1682`
  - mean IoU `0.0196 -> 0.2404`
- Broad gains across platforms:
  - desktop `0.3084 -> 0.6976`
  - mobile `0.0060 -> 0.7445`
  - web `0.0046 -> 0.6796`

**Recommended visual**

- `outputs/final_packaging/figures/screenspot_v2_overall_before_after.png`
- `outputs/final_packaging/figures/screenspot_v2_platform_point_accuracy_before_after.png`

**Speaker note / intent**

- This is the empirical climax. Keep it clean and numerical.

## Slide 12 - Takeaways and Honest Limits

**Bullets**

- Final main claim:
  - the strongest project result is Qwen-first held-out grounding improvement via coordinate-frame refinement
- RL relevance remains real through contextual-bandit framing and verifiable reward
- Reward optimization was part of the story, but not the strongest final empirical result
- Limits:
  - single-step grounding only
  - not full browser-agent execution
  - final claim is benchmarked on ScreenSpot-v2, not every GUI benchmark

**Recommended visual**

- Final takeaway box plus one compact artifact path list

**Speaker note / intent**

- End with a claim the evidence fully supports and nothing stronger.

## Optional Backup Slide - Artifact Map

**Bullets**

- Final report
- Benchmark figures and tables
- Final evaluation summaries
- Best config and key output directories

**Recommended visual**

- One-column file path list drawn from `docs/final_artifact_index.md`

**Speaker note / intent**

- Useful if asked how to navigate the repository quickly.
