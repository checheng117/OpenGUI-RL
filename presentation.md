---
marp: true
theme: default
paginate: true
style: |
  section {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 21px;
    padding: 36px 50px;
  }
  h1 { font-size: 34px; color: #1a1a2e; }
  h2 { font-size: 26px; color: #16213e; border-bottom: 2px solid #0f3460; padding-bottom: 6px; margin-bottom: 14px; }
  table { font-size: 17px; width: 100%; border-collapse: collapse; }
  th { background: #0f3460; color: white; padding: 6px 10px; }
  td { padding: 5px 10px; border-bottom: 1px solid #ddd; }
  .highlight { background: #e8f4f8; border-left: 4px solid #0f3460; padding: 8px 14px; border-radius: 4px; margin-top: 10px; }
  .positive { background: #eafaf1; border-left: 4px solid #27ae60; padding: 8px 14px; border-radius: 4px; margin-top: 10px; }
  .warn { background: #fef9e7; border-left: 4px solid #f39c12; padding: 8px 14px; border-radius: 4px; margin-top: 10px; }
  code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-size: 17px; }
---

<!-- Slide 1: Title -->
# Cross-Website GUI Grounding
## with Verifiable Reward Optimization

**CSC6129 Reinforcement Learning — Final Project**
Che Cheng · 2026

<br>

**Three core findings:**
1. Hybrid OCR/DOM candidate augmentation is **critical** when candidate structure is semantically informative
2. Point-first grounding **transfers well** across held-out GUI benchmarks
3. Reward-based reranking **helps conditionally** — gains saturate once supervision is already strong

---

<!-- Slide 2: Problem Framing -->
## What is GUI Grounding?

**Task:** Given a screenshot + natural-language instruction → predict the correct UI action

```
Input:  [screenshot]  +  "Click the search button"
Output: click_point: (0.52, 0.13)   bbox: [0.45, 0.10, 0.60, 0.17]   action_type: click
```

**Scope:** Single-step grounding only — not full browser automation or long-horizon planning.

**Why it's an RL problem** — reward is **verifiable and deterministic:**

| Signal | Meaning |
|---|---|
| Element match | Predicted element = annotated element |
| Click inside target | Click point falls within bounding box |
| IoU ≥ 0.5 | Predicted box overlaps ground truth |
| Action-type correct | `click` / `type` / `select` matches label |
| Invalid format penalty | Penalizes malformed coordinate output |

→ Natural fit for a **contextual bandit**: context = screenshot + instruction, action = grounded UI action

---

<!-- Slide 3: System Architecture -->
## System Architecture

```
  Screenshot + Instruction  (+  OCR/DOM candidate anchors  ←  hybrid mode)
               │
               ▼
   ┌─────────────────────────┐
   │   Stage A  ·  Grounding │   Qwen2.5-VL 3B   SFT on Mind2Web
   │   click_point + bbox    │   candidate_slot supervision (hybrid)
   └────────────┬────────────┘
                │   k = 4 candidates
                ▼
   ┌─────────────────────────┐
   │   Stage B  ·  Reranker  │   Verifiable reward scoring
   │   best-of-k selection   │   Learned reward model
   └────────────┬────────────┘
                │
                ▼
         Final Action Output
```

**Two research questions:**
1. How strong can Stage A become?
2. When does Stage B reranking still add value on top?

---

<!-- Slide 4: Benchmarks -->
## Three Benchmarks, Three Roles

| Benchmark | Role | Key Purpose |
|---|---|---|
| **Mind2Web** | Primary supervised | Stage A SFT, hybrid augmentation, Stage B reranking, headroom analysis |
| **ScreenSpot-v2** | Primary held-out | Clean transfer evaluation: point-native & dual-path verifier |
| **VisualWebBench** | Supplementary transfer | What transfers vs. what breaks under protocol mismatch |

<br>

**Mind2Web generalization splits** — three levels of difficulty:

| Split | Description |
|---|---|
| `test_task` | Same website, new task instructions |
| `test_website` | Held-out websites (harder) |
| `test_domain` | Held-out domains (hardest) |

---

<!-- Slide 5: Mind2Web Hybrid Stage A -->
## Mind2Web Stage A: Hybrid Candidate Augmentation

**Key design:** Augment screenshot with compact OCR/DOM-style candidate anchors + supervise auxiliary `candidate_slot` target alongside grounding output.

Pure visual Stage A was insufficient (point acc ≈ 0.04 even after bug fix) → structured candidate evidence was essential.

<div class="positive">

**Hybrid Stage A — internal validation jump:**
Point accuracy **0.0375 → 0.7875** &nbsp;·&nbsp; Mean IoU **0.0061 → 0.7314**

</div>

**Official cached subset results:**

| Split | Element Acc | Click-Point Acc | IoU@0.5 | Action Acc |
|---|---:|---:|---:|---:|
| `test_task` | **95.00%** | **95.00%** | **95.00%** | 90.00% |
| `test_website` | 80.00% | 85.00% | 85.00% | **95.00%** |
| `test_domain` | 89.47% | 89.47% | 89.47% | 94.74% |

Average gain over pure visual: **+88 pts** on element / point / IoU@0.5

---

<!-- Slide 6: Stage B Reranking -->
## Stage B: When Does Reranking Help?

**Setup:** Stage A generates k = 4 candidates → learned reward model selects the best one.

**Oracle best-of-k headroom** (upper bound on what reranking can recover):

| Candidate Pool | Oracle Point Gain | Reward Gain |
|---|---:|---:|
| Historical k=4 &nbsp;(weak Stage A) | **+10.17 pts** | 0.0376 |
| Historical k=8 expanded | **+10.17 pts** | 0.0431 |
| **Final hybrid rebuild k=4** | **+5.08 pts** | 0.0983 |

<div class="highlight">

**Key insight:** Weak Stage A left ~10 pts of recoverable headroom in the candidate pool. Strong hybrid Stage A cut that headroom to ~5 pts. **Reranking is a conditional gain, not a universal win.**

</div>

**Design trade-off:** Investing in better Stage A representation delivers more reliable gains than scaling up Stage B reranking.

---

<!-- Slide 7: ScreenSpot-v2 -->
## ScreenSpot-v2: Held-Out Transfer Benchmark

**Debugging step that unblocked evaluation:** Initial results were near-zero — traced to a coordinate-frame mismatch (model predicted in resized-image space; evaluation scored against original-image coordinates). Fixing this produced the first credible held-out baseline.

**Final method comparison:**

| Method | Point Acc | IoU@0.5 | Mean IoU |
|---|---:|---:|---:|
| Reproduced public Qwen baseline | 75.63% | 5.19% | 13.27% |
| Point-native decoupled | 77.36% | 9.67% | 19.12% |
| **Dual-path verifier** | **77.91%** | **17.22%** | **25.20%** |

<div class="positive">

Point-native inference beat the reproduced baseline. Dual-path verifier improved IoU@0.5 by **+12 pts** with only **0.000167 s/image** overhead.

</div>

**Subgroup finding:** Text elements outperform icon elements by **+19.70 pts** point accuracy.

---

<!-- Slide 8: VisualWebBench -->
## VisualWebBench: Sharpening the Transfer Claim

**Protocol difference from Mind2Web:** Anonymous 8-box layout — no semantic candidate labels attached to boxes.

| Method | Official Choice Acc | Point Acc | Mean IoU |
|---|---:|---:|---:|
| Structured screenshot-only | 78.88% | 64.53% | 32.06% |
| **Point-native decoupled** | **87.21%** | **79.46%** | 34.35% |
| Dual-path verifier | 86.82% | 79.65% | 33.12% |
| Mind2Web hybrid transfer | 23.84% | 23.84% | 23.84% |

<div class="warn">

**Mind2Web hybrid transfer collapsed to 23.84%** — candidate slot semantics do not survive an anonymous-box protocol.

</div>

**Refined transfer claim:** Point-first grounding is robust across protocols. Candidate-aware methods require matching benchmark semantics to transfer.

---

<!-- Slide 9: Final Takeaways -->
## Final Takeaways

**What worked well:**

✅ Hybrid OCR/DOM candidate augmentation → **+88 pts** on Mind2Web — decisive, not marginal

✅ Point-first grounding → strong transfer on both ScreenSpot-v2 and VisualWebBench

✅ Dual-path verifier → consistent IoU gain with negligible runtime overhead

<br>

**Key design lessons:**

⚠️ Reward-based reranking is **conditional** — valuable when the baseline leaves recoverable headroom; saturates once Stage A is strong

⚠️ Candidate-aware transfer requires **protocol compatibility** — fails on anonymous-box benchmarks

<br>

<div class="highlight">

**Honest scope:** This is a single-step perception layer, not a complete browser agent.
Correct claim: *"strong representation first, reward as conditional second-stage gain"*

</div>
