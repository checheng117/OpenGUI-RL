# Scripts

The main public entry points are:

| Workflow | Script |
| --- | --- |
| Prepare Mind2Web interface | `prepare_mind2web.py` |
| Prepare ScreenSpot-v2 interface | `prepare_screenspot_v2.py` |
| Train Stage A SFT | `run_train_sft.py` |
| Export reward-labeled candidates | `run_generate_candidates.py` |
| Train Stage B reranker | `run_train_reranker.py` |
| Run ScreenSpot-v2 inference | `run_eval_screenspot_v2.py` |
| Run dual-path verifier | `run_eval_dual_path_verifier.py` |
| Run VisualWebBench inference | `run_eval_visualwebbench.py` |
| Run quantitative summary | `run_quantitative_metrics_suite.py` |

For common commands, see the top-level `Makefile` and `docs/REPRODUCTION.md`.

Several analysis scripts are retained for transparency but are not required for the minimal reproduction path.
