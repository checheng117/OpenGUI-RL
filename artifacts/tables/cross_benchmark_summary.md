# Cross-Benchmark Summary

Compact synthesis of the completed primary and supplementary benchmark stories.

| Benchmark slice | Comparison | Key measured result | Final takeaway |
| --- | --- | --- | --- |
| Mind2Web Stage A | Pure visual vs hybrid candidate-aware Stage A | Internal point acc 0.0375 -> 0.7875; cached official split point acc 0.0000/0.0000/0.0526 -> 0.9500/0.8500/0.8947 | Hybrid OCR/DOM candidate augmentation is critical when semantically informative candidate structure is available. |
| Mind2Web Stage B (earlier) | Old Stage A + expanded pools + source-aware reranker | Reward gain +0.0055 / +0.0211 / +0.0202 on test_task / test_website / test_domain | Reward-based reranking helps when the baseline is weak enough to leave recoverable headroom. |
| Mind2Web Stage B (rebuild) | Rebuild on hybrid Stage A | Official headroom pools 11/59 -> 5/59; mean official-split reward 1.7496 -> 1.6745 after reranking | Once Stage A is strong, reranking gains become sparse and inconsistent rather than robust defaults. |
| ScreenSpot-v2 | Structured coord-fix, point-native, dual-path | Public baseline 0.7563; point-native 0.7736; dual-path 0.7791 point accuracy | Point-first grounding transfers strongly; lightweight verification adds only modest extra gain after the point-native path is already strong. |
| VisualWebBench | Structured, point-native, dual-path, hybrid transfer | Choice acc 0.7888 -> 0.8721 (point-native); dual-path 0.8682; hybrid transfer 0.2384 | Point-first transfer generalizes, dual-path saturates quickly, and Mind2Web-style candidate-aware transfer fails when the benchmark exposes only anonymous option boxes. |
