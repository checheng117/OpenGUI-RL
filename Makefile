.PHONY: install test lint metrics stage-a stage-b-candidates stage-b-reranker eval-screenspot-point eval-screenspot-dual eval-visualwebbench-point

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src scripts tests

metrics:
	python scripts/run_quantitative_metrics_suite.py

stage-a:
	python scripts/run_train_sft.py --config configs/train/mind2web_stageA_qwen2_5_vl_3b_sft_hybrid_candidates.yaml

stage-b-candidates:
	python scripts/run_generate_candidates.py --config configs/train/mind2web_stageB_candidates_qwen2_5_vl_3b_hybrid_stagea.yaml

stage-b-reranker:
	python scripts/run_train_reranker.py --config configs/train/mind2web_stageB_reranker_qwen_hybrid_stagea.yaml

eval-screenspot-point:
	python scripts/run_eval_screenspot_v2.py --config configs/eval/screenspot_v2_qwen2_5_vl_3b_point_native_decoupled.yaml

eval-screenspot-dual:
	python scripts/run_eval_dual_path_verifier.py --config configs/eval/screenspot_v2_qwen2_5_vl_3b_dual_path_verifier.yaml

eval-visualwebbench-point:
	python scripts/run_eval_visualwebbench.py --config configs/eval/visualwebbench_qwen2_5_vl_3b_point_native_decoupled.yaml
