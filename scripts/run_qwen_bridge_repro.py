#!/usr/bin/env python3
"""Bridge from safe baseline to original crash config, one variable at a time."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG = PROJECT_ROOT / "configs/train/generate_candidates_qwen2_5_vl_3b_safe_baseline_locked.yaml"
MAIN_SCRIPT = PROJECT_ROOT / "scripts/run_generate_candidates.py"

OUT_ROOT = PROJECT_ROOT / "outputs/qwen_bridge_repro"
RUN_LOG_DIR = OUT_ROOT / "run_logs"
RUN_SUMMARY_DIR = OUT_ROOT / "run_summaries"
RUN_CONFIG_DIR = OUT_ROOT / "run_configs"
RUN_ARTIFACT_DIR = OUT_ROOT / "run_artifacts"
BRIDGE_TABLE_CSV = OUT_ROOT / "bridge_table.csv"
FIRST_FAILURE_JSON = OUT_ROOT / "first_failure.json"
LAST_STABLE_JSON = OUT_ROOT / "last_stable.json"
REPRO_SUMMARY_JSON = OUT_ROOT / "reproduction_summary.json"
FROZEN_BASELINE_COPY = OUT_ROOT / "frozen_safe_baseline.yaml"


@dataclass
class RunSpec:
    run_id: str
    phase: str
    variable: str
    value: Any
    notes: str
    updates: dict[str, Any]


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _set_nested(d: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur = d
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def _safe_get(d: dict[str, Any], *keys: str, default=None):
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _build_signature(cfg: dict[str, Any]) -> str:
    return (
        f"backend={_safe_get(cfg, 'candidate_generation', 'backend', default='qwen_vl')}|"
        f"dtype={_safe_get(cfg, 'model', 'torch_dtype', default='auto')}|"
        f"attention_backend={_safe_get(cfg, 'model', 'attn_implementation', default='sdpa')}|"
        f"top_k={_safe_get(cfg, 'candidate_generation', 'top_k', default='?')}|"
        f"max_new_tokens={_safe_get(cfg, 'model', 'max_new_tokens', default='?')}|"
        f"num_samples={_safe_get(cfg, 'data', 'train', 'max_samples', default='?')}"
    )


def _run_once(spec: RunSpec, baseline_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = deepcopy(baseline_cfg)
    for k, v in spec.updates.items():
        _set_nested(cfg, k, v)
    _set_nested(cfg, "experiment.run_id", spec.run_id)
    _set_nested(cfg, "experiment.name", f"qwen_bridge_repro_{spec.run_id}")
    run_output_dir = RUN_ARTIFACT_DIR / spec.run_id
    _set_nested(cfg, "output.output_dir", str(run_output_dir.relative_to(PROJECT_ROOT)))

    run_cfg_path = RUN_CONFIG_DIR / f"{spec.run_id}.yaml"
    _write_yaml(run_cfg_path, cfg)
    log_path = RUN_LOG_DIR / f"{spec.run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    cmd = [sys.executable, str(MAIN_SCRIPT), "--config", str(run_cfg_path)]
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=False,
        )

    summary_path = run_output_dir / "summary_train.json"
    failure_path = run_output_dir / "failures_train.json"
    runtime_path = run_output_dir / "runtime_train.json"
    summary = _read_json(summary_path) if summary_path.exists() else {}
    failures = _read_json(failure_path) if failure_path.exists() else []

    run_success = bool(proc.returncode == 0 and summary.get("run_survived", False))
    final_failure_type = summary.get("final_failure_type")
    if final_failure_type is None and failures:
        final_failure_type = failures[-1].get("error_type")

    record = {
        "run_id": spec.run_id,
        "phase": spec.phase,
        "variable": spec.variable,
        "value": spec.value,
        "notes": spec.notes,
        "config_signature": _build_signature(cfg),
        "exit_code": int(proc.returncode),
        "run_success": run_success,
        "attempted_samples": int(summary.get("attempted_samples", 0)) if summary else 0,
        "successful_samples": int(summary.get("successful_samples", 0)) if summary else 0,
        "failed_samples": int(summary.get("failed_samples", 0)) if summary else 0,
        "peak_gpu_memory": summary.get("peak_gpu_memory", {}),
        "peak_gpu_reserved_bytes": _safe_get(summary, "peak_gpu_memory", "peak_max_reserved_bytes", default=0),
        "final_failure_type": final_failure_type,
        "last_processed_sample_id": summary.get("last_processed_sample_id"),
        "last_failed_sample_id": summary.get("last_failed_sample_id"),
        "log_path": str(log_path.relative_to(PROJECT_ROOT)),
        "summary_path": str(summary_path.relative_to(PROJECT_ROOT)) if summary_path.exists() else None,
        "failure_path": str(failure_path.relative_to(PROJECT_ROOT)) if failure_path.exists() else None,
        "runtime_path": str(runtime_path.relative_to(PROJECT_ROOT)) if runtime_path.exists() else None,
    }
    _write_json(RUN_SUMMARY_DIR / f"{spec.run_id}.json", record)
    return record


def _write_bridge_table(rows: list[dict[str, Any]]) -> None:
    columns = [
        "run_id",
        "phase",
        "variable",
        "value",
        "config_signature",
        "exit_code",
        "run_success",
        "attempted_samples",
        "successful_samples",
        "failed_samples",
        "peak_gpu_reserved_bytes",
        "final_failure_type",
        "last_processed_sample_id",
        "last_failed_sample_id",
        "log_path",
        "summary_path",
        "failure_path",
        "runtime_path",
        "notes",
    ]
    BRIDGE_TABLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(BRIDGE_TABLE_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in columns})


def _effective_values(updates_so_far: dict[str, Any], baseline_cfg: dict[str, Any]) -> dict[str, Any]:
    def val(key: str, *base_keys: str):
        return updates_so_far.get(key, _safe_get(baseline_cfg, *base_keys))

    return {
        "backend": _safe_get(baseline_cfg, "candidate_generation", "backend", default="qwen_vl"),
        "dtype": val("model.torch_dtype", "model", "torch_dtype"),
        "attention_backend": val("model.attn_implementation", "model", "attn_implementation"),
        "top_k": val("candidate_generation.top_k", "candidate_generation", "top_k"),
        "max_new_tokens": val("model.max_new_tokens", "model", "max_new_tokens"),
        "max_samples": val("data.train.max_samples", "data", "train", "max_samples"),
    }


def main() -> None:
    baseline_cfg = _read_yaml(BASELINE_CONFIG)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for p in (RUN_LOG_DIR, RUN_SUMMARY_DIR, RUN_CONFIG_DIR, RUN_ARTIFACT_DIR):
        p.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(BASELINE_CONFIG, FROZEN_BASELINE_COPY)

    all_rows: list[dict[str, Any]] = []
    first_failure: dict[str, Any] | None = None
    updates_so_far: dict[str, Any] = {}
    last_stable_effective = _effective_values(updates_so_far, baseline_cfg)
    phase_stop: str | None = None

    bridge_plan: list[RunSpec] = [
        # Phase 0
        RunSpec("phase0_safe_baseline_verify", "0", "none", "baseline_verify", "Freeze baseline and verify once.", {}),
        # Phase 1 max_new_tokens only
        RunSpec("phase1_max_new_tokens_128", "1", "max_new_tokens", 128, "Only max_new_tokens changed.", {"model.max_new_tokens": 128}),
        RunSpec("phase1_max_new_tokens_160", "1", "max_new_tokens", 160, "Only max_new_tokens changed.", {"model.max_new_tokens": 160}),
        RunSpec("phase1_max_new_tokens_192", "1", "max_new_tokens", 192, "Only max_new_tokens changed.", {"model.max_new_tokens": 192}),
        RunSpec("phase1_max_new_tokens_224", "1", "max_new_tokens", 224, "Only max_new_tokens changed.", {"model.max_new_tokens": 224}),
        RunSpec("phase1_max_new_tokens_256", "1", "max_new_tokens", 256, "Only max_new_tokens changed.", {"model.max_new_tokens": 256}),
        # Phase 2 top_k only
        RunSpec("phase2_top_k_3", "2", "top_k", 3, "Only top_k changed.", {"candidate_generation.top_k": 3}),
        RunSpec("phase2_top_k_4", "2", "top_k", 4, "Only top_k changed.", {"candidate_generation.top_k": 4}),
        # Phase 3 max_samples only
        RunSpec("phase3_max_samples_8", "3", "max_samples", 8, "Only max_samples changed.", {"data.train.max_samples": 8}),
        RunSpec("phase3_max_samples_10", "3", "max_samples", 10, "Only max_samples changed.", {"data.train.max_samples": 10}),
        # Phase 4 backend only
        RunSpec("phase4_attention_sdpa", "4", "attention_backend", "sdpa", "Only attention backend changed.", {"model.attn_implementation": "sdpa"}),
        # Phase 5 dtype only
        RunSpec("phase5_dtype_bfloat16", "5", "dtype", "bfloat16", "Only dtype changed.", {"model.torch_dtype": "bfloat16"}),
        # Phase 6 final combined target (only if no failures before)
        RunSpec("phase6_final_full_repro_target", "6", "final_combined", "target", "Final combined reproduction attempt.", {}),
    ]

    for spec in bridge_plan:
        if phase_stop is not None:
            break

        # Compose updates for this run: all previously stabilized updates + this run's single change.
        current_updates = deepcopy(updates_so_far)
        current_updates.update(spec.updates)

        # Final phase explicitly sets full target tuple.
        if spec.phase == "6":
            current_updates.update(
                {
                    "model.torch_dtype": "bfloat16",
                    "model.attn_implementation": "sdpa",
                    "model.max_new_tokens": 256,
                    "candidate_generation.top_k": 4,
                    "data.train.max_samples": 10,
                }
            )

        rec = _run_once(
            RunSpec(
                run_id=spec.run_id,
                phase=spec.phase,
                variable=spec.variable,
                value=spec.value,
                notes=spec.notes,
                updates=current_updates,
            ),
            baseline_cfg,
        )
        all_rows.append(rec)
        _write_bridge_table(all_rows)

        if rec["run_success"]:
            updates_so_far = current_updates
            last_stable_effective = _effective_values(updates_so_far, baseline_cfg)
            _write_json(
                LAST_STABLE_JSON,
                {
                    "last_stable_run_id": rec["run_id"],
                    "last_stable_config": last_stable_effective,
                    "last_stable_signature": rec["config_signature"],
                    "record": rec,
                },
            )
        else:
            first_failure = {
                "first_failing_run_id": rec["run_id"],
                "phase": rec["phase"],
                "first_failing_signature": rec["config_signature"],
                "record": rec,
            }
            _write_json(FIRST_FAILURE_JSON, first_failure)
            phase_stop = rec["phase"]

    if first_failure is None and not FIRST_FAILURE_JSON.exists():
        _write_json(FIRST_FAILURE_JSON, {"first_failing_run_id": None, "message": "No failing run observed."})
    if not LAST_STABLE_JSON.exists():
        _write_json(
            LAST_STABLE_JSON,
            {
                "last_stable_run_id": None,
                "last_stable_config": _effective_values({}, baseline_cfg),
                "message": "No successful run recorded.",
            },
        )

    original_reproducible = bool(first_failure is not None and first_failure.get("phase") == "6")
    if first_failure is None:
        root_cause_class = "currently_non_reproducible_under_bridge_path"
    elif first_failure["phase"] in {"1", "2", "3", "4", "5"}:
        root_cause_class = "single_variable_sensitive"
    else:
        root_cause_class = "multi_factor_interaction_sensitive"

    summary = {
        "environment": {
            "python": sys.executable,
            "hf_endpoint": os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
            "hf_mirror_used": True,
        },
        "phase_order": ["0", "1", "2", "3", "4", "5", "6"],
        "tested_runs": len(all_rows),
        "successful_runs": sum(1 for r in all_rows if r["run_success"]),
        "failed_runs": sum(1 for r in all_rows if not r["run_success"]),
        "last_stable": _read_json(LAST_STABLE_JSON),
        "first_failure": _read_json(FIRST_FAILURE_JSON),
        "original_crash_reproducible_now": original_reproducible,
        "root_cause_class": root_cause_class,
        "artifacts": {
            "bridge_table_csv": str(BRIDGE_TABLE_CSV.relative_to(PROJECT_ROOT)),
            "first_failure_json": str(FIRST_FAILURE_JSON.relative_to(PROJECT_ROOT)),
            "last_stable_json": str(LAST_STABLE_JSON.relative_to(PROJECT_ROOT)),
            "run_logs_dir": str(RUN_LOG_DIR.relative_to(PROJECT_ROOT)),
            "run_summaries_dir": str(RUN_SUMMARY_DIR.relative_to(PROJECT_ROOT)),
        },
    }
    _write_json(REPRO_SUMMARY_JSON, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
