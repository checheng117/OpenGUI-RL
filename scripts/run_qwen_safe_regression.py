#!/usr/bin/env python3
"""Controlled one-variable-at-a-time regression runner for Qwen safe envelope."""

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
REG_ROOT = PROJECT_ROOT / "outputs/qwen_safe_regression"
RUN_LOG_DIR = REG_ROOT / "run_logs"
RUN_SUMMARY_DIR = REG_ROOT / "run_summaries"
RUN_CONFIG_DIR = REG_ROOT / "run_configs"
RUN_ARTIFACT_DIR = REG_ROOT / "run_artifacts"
REG_TABLE_CSV = REG_ROOT / "regression_table.csv"
UNSAFE_JSON = REG_ROOT / "unsafe_configs.json"
SAFE_ENVELOPE_JSON = REG_ROOT / "safe_envelope_summary.json"


@dataclass
class RunSpec:
    run_id: str
    phase: str
    variable: str
    value: Any
    notes: str
    updates: dict[str, Any]


def _set_nested(d: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur = d
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _read_json(path: Path) -> dict[str, Any] | list[Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_get(d: dict[str, Any], *keys: str, default=None):
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _build_signature(cfg: dict[str, Any]) -> str:
    backend = _safe_get(cfg, "candidate_generation", "backend", default="qwen_vl")
    dtype = _safe_get(cfg, "model", "torch_dtype", default="auto")
    attn = _safe_get(cfg, "model", "attn_implementation", default="sdpa")
    top_k = _safe_get(cfg, "candidate_generation", "top_k", default="?")
    max_new_tokens = _safe_get(cfg, "model", "max_new_tokens", default="?")
    num_samples = _safe_get(cfg, "data", "train", "max_samples", default="?")
    return (
        f"backend={backend}|dtype={dtype}|attention_backend={attn}|"
        f"top_k={top_k}|max_new_tokens={max_new_tokens}|num_samples={num_samples}"
    )


def _run_once(spec: RunSpec, baseline_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = deepcopy(baseline_cfg)
    for key, value in spec.updates.items():
        _set_nested(cfg, key, value)
    _set_nested(cfg, "experiment.run_id", spec.run_id)
    _set_nested(cfg, "experiment.name", f"qwen_safe_regression_{spec.run_id}")
    run_output_dir = RUN_ARTIFACT_DIR / spec.run_id
    _set_nested(cfg, "output.output_dir", str(run_output_dir.relative_to(PROJECT_ROOT)))

    config_signature = _build_signature(cfg)
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
    failures: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    if summary_path.exists():
        summary = _read_json(summary_path)  # type: ignore[assignment]
    if failure_path.exists():
        failures = _read_json(failure_path)  # type: ignore[assignment]

    run_success = bool(proc.returncode == 0 and summary.get("run_survived", False))
    final_failure_type = summary.get("final_failure_type")
    if final_failure_type is None and failures:
        final_failure_type = failures[-1].get("error_type")

    peak_reserved = _safe_get(summary, "peak_gpu_memory", "peak_max_reserved_bytes", default=0)
    attempted = int(summary.get("attempted_samples", 0)) if summary else 0
    succeeded = int(summary.get("successful_samples", 0)) if summary else 0
    failed = int(summary.get("failed_samples", 0)) if summary else 0
    last_sample = summary.get("last_processed_sample_id")
    last_failed_sample = summary.get("last_failed_sample_id")

    run_record = {
        "run_id": spec.run_id,
        "phase": spec.phase,
        "variable": spec.variable,
        "value": spec.value,
        "notes": spec.notes,
        "config_signature": config_signature,
        "exit_code": proc.returncode,
        "run_success": run_success,
        "attempted_samples": attempted,
        "successful_samples": succeeded,
        "failed_samples": failed,
        "peak_gpu_reserved_bytes": peak_reserved,
        "final_failure_type": final_failure_type,
        "last_processed_sample_id": last_sample,
        "last_failed_sample_id": last_failed_sample,
        "log_path": str(log_path.relative_to(PROJECT_ROOT)),
        "summary_path": str(summary_path.relative_to(PROJECT_ROOT)) if summary_path.exists() else None,
        "failure_path": str(failure_path.relative_to(PROJECT_ROOT)) if failure_path.exists() else None,
        "runtime_path": str(runtime_path.relative_to(PROJECT_ROOT)) if runtime_path.exists() else None,
    }
    RUN_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUN_SUMMARY_DIR / f"{spec.run_id}.json", "w", encoding="utf-8") as f:
        json.dump(run_record, f, indent=2, ensure_ascii=False)
    return run_record


def _write_regression_table(rows: list[dict[str, Any]]) -> None:
    REG_TABLE_CSV.parent.mkdir(parents=True, exist_ok=True)
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
    with open(REG_TABLE_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in columns})


def _infer_root_cause(rows: list[dict[str, Any]]) -> str:
    failed = [r for r in rows if not r["run_success"]]
    if not failed:
        return "no_failure_observed_under_tested_range"
    fail_vars = {r["variable"] for r in failed}
    if fail_vars == {"max_new_tokens"} or fail_vars == {"top_k"} or fail_vars == {"num_samples"}:
        return "load_generation_burden_primary"
    if "attn_implementation" in fail_vars and "torch_dtype" not in fail_vars:
        return "attention_backend_primary"
    if "torch_dtype" in fail_vars and "attn_implementation" not in fail_vars:
        return "dtype_primary"
    return "driver_or_runtime_interaction_likely"


def main() -> None:
    baseline_cfg = _read_yaml(BASELINE_CONFIG)
    REG_ROOT.mkdir(parents=True, exist_ok=True)
    for d in (RUN_LOG_DIR, RUN_SUMMARY_DIR, RUN_CONFIG_DIR, RUN_ARTIFACT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # Keep a literal frozen baseline copy used in this regression batch.
    frozen_copy = REG_ROOT / "frozen_safe_baseline.yaml"
    shutil.copyfile(BASELINE_CONFIG, frozen_copy)

    all_rows: list[dict[str, Any]] = []

    # Baseline control run.
    baseline_run = RunSpec(
        run_id="baseline_fp16_eager_tk2_nt96_ns6",
        phase="control",
        variable="none",
        value="baseline",
        notes="Frozen safe baseline control run.",
        updates={},
    )
    all_rows.append(_run_once(baseline_run, baseline_cfg))

    # Phase A: Load scaling only (one variable at a time, monotonic increments, stop on failure).
    load_sweeps: list[tuple[str, list[Any], str]] = [
        ("model.max_new_tokens", [128, 160, 192], "max_new_tokens"),
        ("candidate_generation.top_k", [3, 4], "top_k"),
        ("data.train.max_samples", [8, 10], "num_samples"),
    ]
    stable_load = {
        "model.max_new_tokens": int(_safe_get(baseline_cfg, "model", "max_new_tokens", default=96)),
        "candidate_generation.top_k": int(_safe_get(baseline_cfg, "candidate_generation", "top_k", default=2)),
        "data.train.max_samples": int(_safe_get(baseline_cfg, "data", "train", "max_samples", default=6)),
    }
    for dotted_key, values, variable in load_sweeps:
        for value in values:
            run_id = f"phaseA_{variable}_{value}"
            rec = _run_once(
                RunSpec(
                    run_id=run_id,
                    phase="A",
                    variable=variable,
                    value=value,
                    notes="Phase A load scaling single-variable run.",
                    updates={dotted_key: value},
                ),
                baseline_cfg,
            )
            all_rows.append(rec)
            if rec["run_success"]:
                stable_load[dotted_key] = value
            else:
                break

    # Phase B: Attention backend only, keep load + dtype fixed.
    phase_b_load_updates = {
        "model.max_new_tokens": stable_load["model.max_new_tokens"],
        "candidate_generation.top_k": stable_load["candidate_generation.top_k"],
        "data.train.max_samples": stable_load["data.train.max_samples"],
        "model.torch_dtype": "float16",
    }
    phase_b_results: dict[str, bool] = {}
    for backend in ["eager", "sdpa"]:
        run_id = f"phaseB_attn_{backend}"
        updates = dict(phase_b_load_updates)
        updates["model.attn_implementation"] = backend
        rec = _run_once(
            RunSpec(
                run_id=run_id,
                phase="B",
                variable="attn_implementation",
                value=backend,
                notes="Phase B backend-only run.",
                updates=updates,
            ),
            baseline_cfg,
        )
        all_rows.append(rec)
        phase_b_results[backend] = bool(rec["run_success"])

    stable_backend = "sdpa" if phase_b_results.get("sdpa") else "eager"

    # Phase C: Dtype only, keep load + backend fixed.
    phase_c_fixed = {
        "model.max_new_tokens": stable_load["model.max_new_tokens"],
        "candidate_generation.top_k": stable_load["candidate_generation.top_k"],
        "data.train.max_samples": stable_load["data.train.max_samples"],
        "model.attn_implementation": stable_backend,
    }
    for dtype in ["float16", "bfloat16"]:
        run_id = f"phaseC_dtype_{dtype}_attn_{stable_backend}"
        updates = dict(phase_c_fixed)
        updates["model.torch_dtype"] = dtype
        rec = _run_once(
            RunSpec(
                run_id=run_id,
                phase="C",
                variable="torch_dtype",
                value=dtype,
                notes="Phase C dtype-only run.",
                updates=updates,
            ),
            baseline_cfg,
        )
        all_rows.append(rec)

    _write_regression_table(all_rows)
    unsafe_rows = [r for r in all_rows if not r["run_success"]]
    with open(UNSAFE_JSON, "w", encoding="utf-8") as f:
        json.dump(unsafe_rows, f, indent=2, ensure_ascii=False)

    safe_rows = [r for r in all_rows if r["run_success"]]
    max_nt = max((int(r["value"]) for r in safe_rows if r["variable"] == "max_new_tokens"), default=96)
    max_tk = max((int(r["value"]) for r in safe_rows if r["variable"] == "top_k"), default=2)
    max_ns = max((int(r["value"]) for r in safe_rows if r["variable"] == "num_samples"), default=6)
    stable_backends = sorted({str(r["value"]) for r in safe_rows if r["variable"] == "attn_implementation"})
    stable_dtypes = sorted({str(r["value"]) for r in safe_rows if r["variable"] == "torch_dtype"})

    envelope = {
        "frozen_baseline_config": str(BASELINE_CONFIG.relative_to(PROJECT_ROOT)),
        "tested_runs": len(all_rows),
        "successful_runs": len(safe_rows),
        "failed_runs": len(unsafe_rows),
        "stable_load_range_observed": {
            "max_new_tokens_lte": max_nt,
            "top_k_lte": max_tk,
            "num_samples_lte": max_ns,
            "note": "Observed-stable range within tested points; not a formal upper bound.",
        },
        "stable_backends_observed": stable_backends,
        "stable_dtypes_observed": stable_dtypes,
        "unsafe_configs": unsafe_rows,
        "most_likely_cause": _infer_root_cause(all_rows),
    }
    with open(SAFE_ENVELOPE_JSON, "w", encoding="utf-8") as f:
        json.dump(envelope, f, indent=2, ensure_ascii=False)

    print(json.dumps(envelope, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
