# Project Scope

OpenGUI-RL studies instruction-conditioned GUI grounding as a scoped decision layer for computer-use agents.

## What Problem Is Isolated?

A real computer-use assistant can understand a user instruction and still fail by clicking the wrong UI element. This project isolates that failure mode:

```text
screenshot + instruction -> target element / click point / box / action type
```

The project therefore focuses on the perception-to-action grounding step, not on the rest of the browser-agent stack.

## Why Single-Step?

Full browser automation is sequential. A wrong click changes future state and can require memory, recovery, replanning, and execution feedback. Those are important problems, but they would obscure the question this project studies:

> When GUI grounding is treated as a verifiable decision layer, when does reward-based selection improve beyond strong supervision?

The single-step formulation makes each example auditable:

- the context is observable,
- the action is structured,
- the reward is deterministic,
- the episode terminates after scoring.

This is why the project is framed as a contextual-bandit-style reinforcement learning problem rather than a full MDP.

## What Counts As RL Here?

The RL component is the verifiable decision layer:

- candidate actions are sampled or generated,
- each candidate receives a deterministic reward,
- best-of-\(k\) headroom measures recoverable alternatives,
- preference pairs and reranker labels are built from reward,
- reward-based selection acts as a lightweight policy-improvement mechanism.

This is not online PPO training and not a deployed browser loop.

## Final Scope Boundary

In scope:

- candidate-aware supervised grounding,
- verifiable reward design,
- reward-labeled candidate generation,
- lightweight reranking,
- point-native and dual-path held-out inference,
- benchmark analysis across Mind2Web, ScreenSpot-v2, and VisualWebBench.

Out of scope:

- long-horizon web task completion,
- browser state transitions after execution,
- recovery from wrong clicks,
- full agent memory and planning,
- online RL in a live browser environment,
- redistribution of benchmark datasets or screenshots.

## Core Lesson

The project's main lesson is practical rather than maximalist:

1. Build a faithful representation and inference contract first.
2. Measure whether the candidate pool still contains recoverable alternatives.
3. Apply verifiable reward where that headroom exists.
4. Treat transfer as protocol-sensitive, especially when candidate semantics change.
