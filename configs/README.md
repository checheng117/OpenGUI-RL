# Configs

Configs are grouped by workflow:

- `data/`: dataset source and local path settings.
- `model/`: VLM backbone settings.
- `train/`: Stage A SFT, Stage B candidate generation, and reranker configs.
- `eval/`: held-out evaluation and transfer configs.
- `demo/`: lightweight demo and single-inference configs.

The final report's main public workflow uses:

- `train/mind2web_stageA_qwen2_5_vl_3b_sft_hybrid_candidates.yaml`
- `train/mind2web_stageB_candidates_qwen2_5_vl_3b_hybrid_stagea.yaml`
- `train/mind2web_stageB_reranker_qwen_hybrid_stagea.yaml`
- `eval/screenspot_v2_qwen2_5_vl_3b_point_native_decoupled.yaml`
- `eval/screenspot_v2_qwen2_5_vl_3b_dual_path_verifier.yaml`
- `eval/visualwebbench_qwen2_5_vl_3b_point_native_decoupled.yaml`
- `eval/visualwebbench_qwen2_5_vl_3b_dual_path_verifier.yaml`

Some older configs are retained for traceability. They are not all part of the final reported result.
