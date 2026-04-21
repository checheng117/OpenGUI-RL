# Quantitative Metrics Suite

- Generated at: `2026-04-10T10:55:47.647530+00:00`

## Mind2Web Hybrid Stage A Official Cached Subset

| Split | Element Acc | Click-Point Acc | IoU@0.5 | Action-Type Acc | Invalid Format |
| --- | ---: | ---: | ---: | ---: | ---: |
| test_task | 95.00% | 95.00% | 95.00% | 90.00% | 0.00% |
| test_website | 80.00% | 85.00% | 85.00% | 95.00% | 0.00% |
| test_domain | 89.47% | 89.47% | 89.47% | 94.74% | 0.00% |

## OCR/DOM Hybrid Minus Pure Visual

- Official-split average element gain: `88.16 pts`
- Official-split average click-point gain: `88.07 pts`
- Official-split average IoU@0.5 gain: `89.82 pts`
- Official-split average action-type gain: `13.68 pts`

## Mind2Web Stage B Oracle Best-of-k Headroom

- Historical k=4 pool: point gain `10.17 pts`, reward gain `0.0376`
- Historical k=8 expanded pool: point gain `10.17 pts`, reward gain `0.0431`
- Final hybrid-rebuild k=4 pool: point gain `5.08 pts`, reward gain `0.0983`

## ScreenSpot-v2

- Point-native: point acc `77.36%`, IoU@0.5 `9.67%`, mean IoU `19.12%`
- Dual-path verifier: point acc `77.91%`, IoU@0.5 `17.22%`, mean IoU `25.20%`
- Text minus icon point-accuracy gap: `19.70 pts`
- Desktop minus web point-accuracy gap: `-1.58 pts`
- Mobile minus web point-accuracy gap: `4.81 pts`

## Runtime

- Point-native first choice: `2.1185 s / image`
- Structured support path: `1.3599 s / image`
- Dual-path verifier only: `0.000167 s / image`
- Dual-path end-to-end: `3.4786 s / image`

## VisualWebBench

- Point-native official choice accuracy: `87.21%`
- Dual-path official choice accuracy: `86.82%`