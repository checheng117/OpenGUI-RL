#!/usr/bin/env python3
"""Stage A: Supervised fine-tuning for GUI grounding (Qwen-primary).

Usage:
    python scripts/run_train_sft.py --config configs/train/sft_baseline.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, NoReturn

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gui_grounding.data.mind2web_dataset import Mind2WebDataset
from gui_grounding.data.schemas import BBox, GroundingSample
from gui_grounding.evaluation.collapse_diagnostics import compute_prediction_collapse_diagnostics
from gui_grounding.evaluation.metrics import compute_all_metrics, mean_normalized_click_l1
from gui_grounding.models.sft_clip_grid_model import SFTCLIPGridModel
from gui_grounding.models.qwen2_vl_grounding import QwenVLGroundingModel
from gui_grounding.reward.verifiable_reward import bbox_iou
from gui_grounding.training.trainer_sft import SFTTrainer
from gui_grounding.training.trainer_sft_qwen import QwenSFTTrainer
from gui_grounding.utils.config import load_config
from gui_grounding.utils.io import save_json, save_jsonl
from gui_grounding.utils.logger import get_logger
from gui_grounding.utils.seed import set_seed

logger = get_logger("run_train_sft")
ACTION_TYPES = {"click", "type", "select", "hover"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT training for GUI grounding")
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Load config and exit without training")
    parser.add_argument("overrides", nargs="*", help="Config overrides (dot-list format)")
    return parser.parse_args()


def _bootstrap_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except Exception:
        pass

    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("GUI_GROUNDING_HF_FALLBACK_ENDPOINT", "https://hf-mirror.com")
    if not os.getenv("HF_TOKEN") and not os.getenv("HUGGINGFACE_HUB_TOKEN"):
        logger.warning("HF token not found in process env; relying on HF local cache/session if available.")
    if not os.getenv("WANDB_API_KEY"):
        logger.warning("WANDB_API_KEY not found. wandb logging will be disabled gracefully.")


def _stable_sample_order(samples: list[GroundingSample], seed: int) -> list[GroundingSample]:
    def _key(sample: GroundingSample) -> str:
        payload = f"{seed}:{sample.sample_id}".encode("utf-8")
        return hashlib.sha1(payload).hexdigest()

    return sorted(samples, key=_key)


def _group_key(sample: GroundingSample) -> str:
    website = str(sample.website or "").strip().lower()
    if website:
        return website
    domain = str(sample.domain or "").strip().lower()
    if domain:
        return f"domain::{domain}"
    return "unknown"


def _allocate_group_counts(
    group_sizes: dict[str, int],
    target_total: int,
    *,
    min_per_group: int = 0,
    max_caps: dict[str, int] | None = None,
) -> dict[str, int]:
    if target_total <= 0 or not group_sizes:
        return {key: 0 for key in group_sizes}

    allocations = {key: 0 for key in group_sizes}
    max_caps = max_caps or {}
    capped_sizes = {key: min(size, max_caps.get(key, size)) for key, size in group_sizes.items()}
    eligible = sorted(key for key, size in capped_sizes.items() if size > 0)
    if not eligible:
        return allocations
    target_total = min(int(target_total), sum(capped_sizes.values()))

    if min_per_group > 0:
        for key in eligible:
            if target_total <= 0:
                break
            allocations[key] = min(min_per_group, capped_sizes[key])
            target_total -= allocations[key]

    if target_total <= 0:
        return allocations

    remaining_keys = [key for key in eligible if allocations[key] < capped_sizes[key]]
    total_size = sum(group_sizes[key] for key in remaining_keys)
    remainders: list[tuple[float, str]] = []
    added_after_floor = 0
    for key in remaining_keys:
        remaining_cap = capped_sizes[key] - allocations[key]
        if remaining_cap <= 0:
            continue
        raw = (group_sizes[key] / max(total_size, 1)) * target_total
        add = min(int(raw), remaining_cap)
        allocations[key] += add
        added_after_floor += add
        remainders.append((raw - add, key))

    slots_left = target_total - added_after_floor
    for _, key in sorted(remainders, reverse=True):
        if slots_left <= 0:
            break
        remaining_cap = capped_sizes[key] - allocations[key]
        if remaining_cap <= 0:
            continue
        allocations[key] += 1
        slots_left -= 1

    while slots_left > 0:
        advanced = False
        for key in sorted(remaining_keys, key=lambda item: group_sizes[item], reverse=True):
            remaining_cap = capped_sizes[key] - allocations[key]
            if remaining_cap <= 0:
                continue
            allocations[key] += 1
            slots_left -= 1
            advanced = True
            if slots_left <= 0:
                break
        if not advanced:
            break

    return allocations


def _select_subset(
    samples: list[GroundingSample],
    max_samples: int | None,
    seed: int,
    strategy: str = "stratified_by_website",
) -> list[GroundingSample]:
    if max_samples is None or max_samples <= 0 or len(samples) <= max_samples:
        return list(samples)
    if strategy == "source_order":
        return list(samples[:max_samples])
    if strategy != "stratified_by_website":
        raise ValueError(f"Unsupported subset selection strategy: {strategy}")

    grouped: dict[str, list[GroundingSample]] = defaultdict(list)
    for sample in samples:
        grouped[_group_key(sample)].append(sample)
    ordered_groups = {key: _stable_sample_order(rows, seed=seed) for key, rows in grouped.items()}
    group_sizes = {key: len(rows) for key, rows in ordered_groups.items()}
    allocations = _allocate_group_counts(group_sizes, max_samples, min_per_group=1)

    selected: list[GroundingSample] = []
    for key in sorted(ordered_groups):
        take_n = min(allocations.get(key, 0), len(ordered_groups[key]))
        selected.extend(ordered_groups[key][:take_n])

    if len(selected) < max_samples:
        leftovers: list[GroundingSample] = []
        for key in sorted(ordered_groups):
            leftovers.extend(ordered_groups[key][allocations.get(key, 0) :])
        selected.extend(_stable_sample_order(leftovers, seed=seed + 17)[: max_samples - len(selected)])

    return _stable_sample_order(selected[:max_samples], seed=seed + 29)


def _split_train_eval(
    all_samples: list[GroundingSample],
    val_ratio: float,
    seed: int,
) -> tuple[list[GroundingSample], list[GroundingSample]]:
    if not all_samples:
        return [], []
    if len(all_samples) == 1:
        return list(all_samples), list(all_samples)

    target_eval = max(1, int(round(len(all_samples) * val_ratio)))
    grouped: dict[str, list[GroundingSample]] = defaultdict(list)
    for sample in all_samples:
        grouped[_group_key(sample)].append(sample)
    ordered_groups = {key: _stable_sample_order(rows, seed=seed) for key, rows in grouped.items()}
    group_sizes = {key: len(rows) for key, rows in ordered_groups.items()}
    max_caps = {key: max(0, size - 1) for key, size in group_sizes.items()}
    eval_allocations = _allocate_group_counts(group_sizes, target_eval, min_per_group=1, max_caps=max_caps)

    train_samples: list[GroundingSample] = []
    eval_samples: list[GroundingSample] = []
    for key in sorted(ordered_groups):
        rows = ordered_groups[key]
        eval_n = min(eval_allocations.get(key, 0), max(0, len(rows) - 1))
        eval_samples.extend(rows[:eval_n])
        train_samples.extend(rows[eval_n:])

    if not eval_samples:
        ordered = _stable_sample_order(all_samples, seed=seed)
        eval_samples = [ordered[0]]
        train_samples = ordered[1:] or ordered

    return _stable_sample_order(train_samples, seed=seed + 101), _stable_sample_order(eval_samples, seed=seed + 202)


def _summarize_samples(samples: list[GroundingSample]) -> dict[str, Any]:
    websites = Counter(str(sample.website or "unknown") for sample in samples)
    domains = Counter(str(sample.domain or "unknown") for sample in samples)
    actions = Counter(str(sample.action_type or "unknown") for sample in samples)
    return {
        "count": len(samples),
        "unique_websites": len(websites),
        "unique_domains": len(domains),
        "website_histogram_top10": dict(websites.most_common(10)),
        "domain_histogram": dict(domains),
        "action_histogram": dict(actions),
    }


def _build_sample_from_manifest_row(row: dict[str, Any]) -> GroundingSample:
    target_bbox = row.get("target_bbox")
    return GroundingSample(
        sample_id=row["sample_id"],
        dataset_name=row.get("dataset_name", "mind2web"),
        split=row.get("split", "unknown"),
        image_path=row["image_path"],
        instruction=row["instruction"],
        action_type=row.get("target_action_type"),
        target_element_id=row.get("target_element_id"),
        target_bbox=BBox(x1=target_bbox[0], y1=target_bbox[1], x2=target_bbox[2], y2=target_bbox[3])
        if isinstance(target_bbox, list) and len(target_bbox) == 4
        else None,
        click_point=tuple(row["target_click_point"]) if isinstance(row.get("target_click_point"), list) else None,
        website=row.get("website"),
        domain=row.get("domain"),
        platform=row.get("platform", "web"),
        metadata={"source_manifest": row.get("_source_manifest")},
    )


def _load_manifest_eval_samples(path: str | Path) -> tuple[str, list[GroundingSample]]:
    path = Path(path)
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_source_manifest"] = str(path)
            rows.append(row)
    split_name = rows[0].get("split", path.stem) if rows else path.stem
    samples = [_build_sample_from_manifest_row(row) for row in rows]
    return split_name, samples


def _load_checkpoint_eval_loss(checkpoint_dir: str | Path) -> float | None:
    checkpoint_dir = Path(checkpoint_dir)
    state_path = checkpoint_dir / "trainer_state.json"
    if not state_path.exists():
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    raw_eval_loss = state.get("eval_loss")
    if raw_eval_loss is None:
        return None
    try:
        value = float(raw_eval_loss)
    except (TypeError, ValueError):
        return None
    if value != value:  # NaN check
        return None
    return value


def _localization_selection_key(
    result: dict[str, Any],
) -> tuple[float, float, float, float, float, float, float]:
    eval_loss = result.get("eval_loss")
    if eval_loss is None:
        eval_loss = float("inf")
    return (
        float(result.get("point_accuracy", 0.0)),
        -float(result.get("mean_normalized_click_l1", float("inf"))),
        float(result.get("iou@0.5", 0.0)),
        float(result.get("mean_iou", 0.0)),
        float(result.get("action_type_accuracy", 0.0)),
        float(result.get("parseable_output_rate", 0.0)),
        -float(eval_loss),
    )


def _prediction_to_record(
    sample: GroundingSample,
    pred,
    raw_text: str,
    parsed: dict[str, Any],
) -> dict[str, Any]:
    pred_bbox = pred.predicted_bbox.as_tuple() if pred.predicted_bbox is not None else None
    gt_bbox = sample.target_bbox.as_tuple() if sample.target_bbox is not None else None
    pred_click = pred.predicted_click_point
    parseable = bool(parsed)
    point_in_box = False
    if pred_click is not None and gt_bbox is not None:
        point_in_box = gt_bbox[0] <= pred_click[0] <= gt_bbox[2] and gt_bbox[1] <= pred_click[1] <= gt_bbox[3]
    return {
        "sample_id": sample.sample_id,
        "split": sample.split,
        "image_path": sample.image_path,
        "instruction": sample.instruction,
        "target_action_type": str(sample.action_type) if sample.action_type is not None else None,
        "target_bbox": list(gt_bbox) if gt_bbox is not None else None,
        "target_click_point": list(sample.click_point) if sample.click_point is not None else None,
        "predicted_action_type": pred.predicted_action_type,
        "predicted_bbox": list(pred_bbox) if pred_bbox is not None else None,
        "predicted_click_point": list(pred_click) if pred_click is not None else None,
        "predicted_element_id": pred.predicted_element_id,
        "confidence": pred.confidence,
        "parseable_output": parseable,
        "response_nonempty": bool(raw_text.strip()),
        "valid_action_type": pred.predicted_action_type in ACTION_TYPES,
        "valid_bbox": pred_bbox is not None,
        "valid_click_point": pred_click is not None,
        "point_in_target": point_in_box,
        "iou": bbox_iou(pred_bbox, gt_bbox) if pred_bbox is not None and gt_bbox is not None else None,
        "raw_response": raw_text,
        "parsed_payload": parsed,
    }


def _build_qwen_eval_model(
    *,
    checkpoint_dir: str | Path,
    model_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    eval_cfg: dict[str, Any],
) -> QwenVLGroundingModel:
    return QwenVLGroundingModel(
        model_name=model_cfg.get("backbone", model_cfg.get("backend", "qwen2_5_vl_3b")),
        device=training_cfg.get("device", "auto"),
        torch_dtype=model_cfg.get("torch_dtype", "auto"),
        max_new_tokens=int(eval_cfg.get("max_new_tokens", 192)),
        temperature=float(eval_cfg.get("temperature", 0.0)),
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
        gpu_memory_utilization=float(model_cfg.get("gpu_memory_utilization", 0.9)),
        min_pixels=model_cfg.get("min_pixels"),
        max_pixels=model_cfg.get("max_pixels"),
        coordinate_frame=eval_cfg.get("coordinate_frame", "original"),
        coordinate_format=eval_cfg.get("coordinate_format", "absolute"),
        point_first_prompt=bool(eval_cfg.get("point_first_prompt", False)),
        web_mobile_hotspot_prompt=bool(eval_cfg.get("web_mobile_hotspot_prompt", False)),
        decoupled_point_native_decode=bool(eval_cfg.get("decoupled_point_native_decode", False)),
        coordinate_quantization_bins=eval_cfg.get("coordinate_quantization_bins"),
        point_native_secondary_bbox_only=bool(eval_cfg.get("point_native_secondary_bbox_only", False)),
        edge_click_interior_threshold=float(eval_cfg.get("edge_click_interior_threshold", 0.0)),
        edge_click_interior_position=float(eval_cfg.get("edge_click_interior_position", 0.45)),
        adapter_path=str(checkpoint_dir),
    )


def _evaluate_qwen_model(
    *,
    model: QwenVLGroundingModel,
    split_name: str,
    samples: list[GroundingSample],
    checkpoint_dir: str | Path,
    output_dir: Path,
    prediction_filename: str,
) -> dict[str, Any]:
    logger.info("Running Stage-A generative eval: split=%s n=%d", split_name, len(samples))
    records: list[dict[str, Any]] = []
    pred_element_ids: list[str | None] = []
    gt_element_ids: list[str | None] = []
    pred_bboxes: list[tuple[float, float, float, float] | None] = []
    gt_bboxes: list[tuple[float, float, float, float] | None] = []
    pred_points: list[tuple[float, float] | None] = []
    gt_points: list[tuple[float, float] | None] = []
    image_sizes: list[tuple[int, int] | None] = []
    pred_actions: list[str | None] = []
    gt_actions: list[str | None] = []

    for idx, sample in enumerate(samples, start=1):
        pred, raw_text, parsed = model.predict_with_details(sample)
        records.append(_prediction_to_record(sample, pred, raw_text, parsed))
        pred_element_ids.append(pred.predicted_element_id)
        gt_element_ids.append(sample.target_element_id)
        pred_bboxes.append(pred.predicted_bbox.as_tuple() if pred.predicted_bbox is not None else None)
        gt_bboxes.append(sample.target_bbox.as_tuple() if sample.target_bbox is not None else None)
        pred_points.append(pred.predicted_click_point)
        gt_points.append(
            sample.click_point if sample.click_point is not None else (sample.target_bbox.center if sample.target_bbox is not None else None)
        )
        image_size: tuple[int, int] | None = None
        image_file = Path(sample.image_path)
        if image_file.exists():
            with Image.open(image_file) as image:
                image_size = image.size
        image_sizes.append(image_size)
        pred_actions.append(pred.predicted_action_type)
        gt_actions.append(str(sample.action_type) if sample.action_type is not None else None)
        if idx % 10 == 0 or idx == len(samples):
            logger.info("Stage-A eval progress split=%s %d/%d", split_name, idx, len(samples))

    prediction_path = output_dir / prediction_filename
    save_jsonl(records, prediction_path)

    metrics = compute_all_metrics(
        pred_element_ids=pred_element_ids,
        gt_element_ids=gt_element_ids,
        pred_bboxes=pred_bboxes,
        gt_bboxes=gt_bboxes,
        pred_points=pred_points,
        pred_actions=pred_actions,
        gt_actions=gt_actions,
    )
    metrics["mean_normalized_click_l1"] = mean_normalized_click_l1(
        pred_points=pred_points,
        gt_points=gt_points,
        image_sizes=image_sizes,
    )
    parseable = sum(1 for row in records if row["parseable_output"])
    valid_bbox = sum(1 for row in records if row["valid_bbox"])
    valid_click = sum(1 for row in records if row["valid_click_point"])
    valid_action = sum(1 for row in records if row["valid_action_type"])
    response_nonempty = sum(1 for row in records if row["response_nonempty"])
    avg_confidence = sum(float(row["confidence"]) for row in records if row["confidence"] is not None) / max(
        1,
        sum(1 for row in records if row["confidence"] is not None),
    )
    collapse_diagnostics = compute_prediction_collapse_diagnostics(records)
    return {
        "split": split_name,
        "num_samples": len(samples),
        "checkpoint_dir": str(checkpoint_dir),
        "prediction_path": str(prediction_path),
        "parseable_output_rate": parseable / max(len(records), 1),
        "response_nonempty_rate": response_nonempty / max(len(records), 1),
        "valid_bbox_rate": valid_bbox / max(len(records), 1),
        "valid_click_point_rate": valid_click / max(len(records), 1),
        "valid_action_type_rate": valid_action / max(len(records), 1),
        "avg_confidence": avg_confidence,
        "collapse_diagnostics": collapse_diagnostics,
        **metrics,
    }


def main() -> None:
    args = parse_args()

    _bootstrap_env()
    cfg = load_config(args.config, overrides=args.overrides if args.overrides else None)
    logger.info("Loaded config: %s", args.config)
    logger.info("Experiment: %s", cfg.get("experiment", {}).get("name", "unknown"))

    seed = cfg.get("training", {}).get("seed", 42)
    set_seed(seed)
    logger.info("Random seed set to %d", seed)

    if args.dry_run:
        logger.info("Dry run — config loaded successfully. Exiting.")
        return

    logger.info("=" * 60)
    logger.info("SFT Training Pipeline")
    logger.info("=" * 60)

    # --- Data loading ---
    data_cfg = cfg.get("data", {})
    train_cfg = data_cfg.get("train", {})
    eval_cfg = data_cfg.get("eval", {})
    split_mode = data_cfg.get("split_mode", "single_split_holdout")
    logger.info(
        "Data mode=%s train_split=%s eval_split=%s",
        split_mode,
        train_cfg.get("split", "train"),
        eval_cfg.get("split", "test_task"),
    )

    if split_mode == "single_split_holdout":
        load_max_samples = train_cfg.get("load_max_samples", train_cfg.get("max_samples"))
        ds = Mind2WebDataset(
            split=train_cfg.get("split", "train"),
            max_samples=load_max_samples,
            cache_screenshots=train_cfg.get("cache_screenshots", True),
        )
        all_samples = [s for s in ds if s.target_bbox is not None and s.action_type is not None]
        all_samples = _select_subset(
            samples=all_samples,
            max_samples=train_cfg.get("max_samples", None),
            seed=seed,
            strategy=train_cfg.get("selection_strategy", "stratified_by_website"),
        )
        train_samples, eval_samples = _split_train_eval(
            all_samples=all_samples,
            val_ratio=float(data_cfg.get("val_ratio", 0.1)),
            seed=seed,
        )
    elif split_mode == "explicit_eval_split":
        train_ds = Mind2WebDataset(
            split=train_cfg.get("split", "train"),
            max_samples=train_cfg.get("max_samples", None),
            cache_screenshots=train_cfg.get("cache_screenshots", True),
        )
        eval_ds = Mind2WebDataset(
            split=eval_cfg.get("split", "test_task"),
            max_samples=eval_cfg.get("max_samples", None),
            cache_screenshots=eval_cfg.get("cache_screenshots", True),
        )
        train_samples = [s for s in train_ds if s.target_bbox is not None and s.action_type is not None]
        eval_samples = [s for s in eval_ds if s.target_bbox is not None and s.action_type is not None]
    else:
        raise ValueError(f"Unsupported data.split_mode: {split_mode}")

    if not train_samples:
        raise RuntimeError("No valid train samples with action+bbox supervision.")
    if not eval_samples:
        raise RuntimeError("No valid eval samples with action+bbox supervision.")
    logger.info("Loaded supervised samples: train=%d eval=%d", len(train_samples), len(eval_samples))
    train_stats = _summarize_samples(train_samples)
    eval_stats = _summarize_samples(eval_samples)
    logger.info("Train stats: %s", train_stats)
    logger.info("Eval stats: %s", eval_stats)

    # --- Model loading ---
    model_cfg = cfg.get("model", {})
    backend = model_cfg.get("backend", "qwen2_5_vl_3b")
    backbone = model_cfg.get("backbone", "qwen2_5_vl_3b")
    logger.info("Model backend=%s backbone=%s", backend, backbone)
    training_cfg = cfg.get("training", {})
    wandb_cfg = cfg.get("wandb", {})
    if backend in {"qwen2_5_vl_3b", "qwen3_vl_2b"}:
        output_dir = Path(training_cfg.get("output_dir", "outputs/sft_qwen_stagea"))
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            from omegaconf import OmegaConf

            resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
        except Exception:
            resolved_cfg = cfg
        save_json(resolved_cfg, output_dir / "resolved_config.json")
        trainer = QwenSFTTrainer(
            model_name=backend if backbone == backend else backbone,
            train_samples=train_samples,
            eval_samples=eval_samples,
            output_dir=training_cfg.get("output_dir", "outputs/sft_qwen_stagea"),
            learning_rate=float(training_cfg.get("learning_rate", 1e-5)),
            weight_decay=float(training_cfg.get("weight_decay", 0.0)),
            batch_size=int(training_cfg.get("batch_size", 1)),
            gradient_accumulation_steps=int(training_cfg.get("gradient_accumulation_steps", 1)),
            num_epochs=int(training_cfg.get("num_epochs", 1)),
            max_steps=training_cfg.get("max_steps", None),
            warmup_ratio=float(training_cfg.get("warmup_ratio", 0.05)),
            max_grad_norm=float(training_cfg.get("max_grad_norm", 1.0)),
            logging_steps=int(training_cfg.get("logging_steps", 10)),
            eval_steps=int(training_cfg.get("eval_steps", 50)),
            save_steps=int(training_cfg.get("save_steps", 200)),
            torch_dtype=model_cfg.get("torch_dtype", "auto"),
            device=training_cfg.get("device", "auto"),
            attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
            gpu_memory_utilization=float(model_cfg.get("gpu_memory_utilization", 0.9)),
            min_pixels=model_cfg.get("min_pixels"),
            max_pixels=model_cfg.get("max_pixels"),
            num_workers=int(training_cfg.get("num_workers", 0)),
            lora_r=int(training_cfg.get("lora_r", 16)),
            lora_alpha=int(training_cfg.get("lora_alpha", 32)),
            lora_dropout=float(training_cfg.get("lora_dropout", 0.05)),
            lora_target_modules=training_cfg.get("lora_target_modules", None),
            gradient_checkpointing=bool(training_cfg.get("gradient_checkpointing", True)),
            target_coordinate_frame=str(training_cfg.get("target_coordinate_frame", "original")),
            target_coordinate_format=str(training_cfg.get("target_coordinate_format", "absolute")),
            point_first_target=bool(training_cfg.get("point_first_target", False)),
            supervise_element_id=bool(training_cfg.get("supervise_element_id", True)),
            supervision_mode=str(training_cfg.get("supervision_mode", "structured")),
            coordinate_quantization_bins=training_cfg.get("coordinate_quantization_bins"),
            bbox_support_fraction=float(training_cfg.get("bbox_support_fraction", 0.0)),
            seed=seed,
        )
    elif backend == "clip_grid_legacy":
        logger.warning("Using legacy surrogate Stage-A backend: clip_grid_legacy")
        from transformers import CLIPProcessor

        processor = CLIPProcessor.from_pretrained(backbone)
        model = SFTCLIPGridModel(
            model_name=backbone,
            num_actions=4,
            grid_rows=int(model_cfg.get("grid_rows", 4)),
            grid_cols=int(model_cfg.get("grid_cols", 6)),
            hidden_dim=int(model_cfg.get("hidden_dim", 512)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            freeze_backbone=bool(model_cfg.get("freeze_backbone", True)),
        )
        trainer = SFTTrainer(
            model=model,
            processor=processor,
            train_samples=train_samples,
            eval_samples=eval_samples,
            output_dir=training_cfg.get("output_dir", "outputs/sft_clip_grid_legacy"),
            learning_rate=training_cfg.get("learning_rate", 2e-5),
            weight_decay=training_cfg.get("weight_decay", 0.01),
            num_epochs=training_cfg.get("num_epochs", 3),
            batch_size=training_cfg.get("batch_size", 4),
            gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 1),
            warmup_ratio=training_cfg.get("warmup_ratio", 0.1),
            max_grad_norm=training_cfg.get("max_grad_norm", 1.0),
            save_steps=training_cfg.get("save_steps", 100),
            eval_steps=training_cfg.get("eval_steps", 50),
            logging_steps=training_cfg.get("logging_steps", 10),
            seed=seed,
            use_wandb=bool(wandb_cfg.get("enabled", False)),
            wandb_project=wandb_cfg.get("project", "gui-grounding"),
            wandb_run_name=wandb_cfg.get("run_name", "sft-clip-grid-legacy"),
            grid_rows=int(model_cfg.get("grid_rows", 4)),
            grid_cols=int(model_cfg.get("grid_cols", 6)),
            loss_weights=training_cfg.get("loss_weights", {"action": 1.0, "grid": 1.0}),
            num_workers=int(training_cfg.get("num_workers", 0)),
            device=(
                None
                if training_cfg.get("device", "auto") == "auto"
                else training_cfg.get("device")
            ),
        )
    else:
        raise ValueError(f"Unsupported model.backend={backend}")

    result = trainer.train()
    if backend in {"qwen2_5_vl_3b", "qwen3_vl_2b"}:
        eval_cfg = cfg.get("evaluation", {})
        best_checkpoint_dir = result.get("best_checkpoint_dir")
        latest_checkpoint_dir = result.get("latest_checkpoint_dir") or result.get("last_checkpoint_dir")
        if eval_cfg.get("enabled", True) and (best_checkpoint_dir or latest_checkpoint_dir):
            output_dir = Path(training_cfg.get("output_dir", "outputs/sft_qwen_stagea"))
            checkpoint_candidates: list[dict[str, Any]] = []
            seen_checkpoint_dirs: set[str] = set()
            checkpoint_selection_rule = (
                "lexicographic(point_accuracy, -mean_normalized_click_l1, iou@0.5, mean_iou, action_type_accuracy, parseable_output_rate, -eval_loss)"
            )
            for checkpoint_dir in [best_checkpoint_dir, latest_checkpoint_dir]:
                if not checkpoint_dir:
                    continue
                checkpoint_dir = str(checkpoint_dir)
                if checkpoint_dir in seen_checkpoint_dirs:
                    continue
                checkpoint_path = Path(checkpoint_dir)
                if not checkpoint_path.exists():
                    logger.warning("Skipping missing checkpoint candidate: %s", checkpoint_path)
                    continue
                seen_checkpoint_dirs.add(checkpoint_dir)
                candidate_model = _build_qwen_eval_model(
                    checkpoint_dir=checkpoint_dir,
                    model_cfg=model_cfg,
                    training_cfg=training_cfg,
                    eval_cfg=eval_cfg,
                )
                internal_eval = _evaluate_qwen_model(
                    model=candidate_model,
                    split_name="internal_val",
                    samples=eval_samples,
                    checkpoint_dir=checkpoint_dir,
                    output_dir=output_dir,
                    prediction_filename=f"eval_predictions_internal_val_{checkpoint_path.name}.jsonl",
                )
                del candidate_model
                eval_loss = _load_checkpoint_eval_loss(checkpoint_dir)
                selection_key = _localization_selection_key(
                    {
                        **internal_eval,
                        "eval_loss": eval_loss,
                    }
                )
                checkpoint_candidates.append(
                    {
                        "checkpoint_dir": checkpoint_dir,
                        "checkpoint_name": checkpoint_path.name,
                        "eval_loss": eval_loss,
                        "selection_key": list(selection_key),
                        "internal_eval": internal_eval,
                    }
                )

            selected_checkpoint_dir = best_checkpoint_dir
            if checkpoint_candidates:
                selected_checkpoint_dir = max(
                    checkpoint_candidates,
                    key=lambda item: tuple(float(v) for v in item["selection_key"]),
                )["checkpoint_dir"]

            eval_summary: dict[str, Any] = {
                "status": "ok",
                "model_name": backend,
                "best_checkpoint_dir": best_checkpoint_dir,
                "latest_checkpoint_dir": latest_checkpoint_dir,
                "selected_checkpoint_dir": selected_checkpoint_dir,
                "checkpoint_selection_rule": checkpoint_selection_rule,
                "checkpoint_candidates": checkpoint_candidates,
                "internal_eval": None,
                "official_cached_subset": {},
            }
            eval_model = _build_qwen_eval_model(
                checkpoint_dir=selected_checkpoint_dir,
                model_cfg=model_cfg,
                training_cfg=training_cfg,
                eval_cfg=eval_cfg,
            )
            eval_summary["internal_eval"] = _evaluate_qwen_model(
                model=eval_model,
                split_name="internal_val",
                samples=eval_samples,
                checkpoint_dir=selected_checkpoint_dir,
                output_dir=output_dir,
                prediction_filename="eval_predictions_internal_val.jsonl",
            )

            for manifest_path in eval_cfg.get("official_eval_manifests", []):
                manifest_file = Path(manifest_path)
                if not manifest_file.exists():
                    logger.warning("Skipping missing official eval manifest: %s", manifest_file)
                    continue
                split_name, manifest_samples = _load_manifest_eval_samples(manifest_file)
                eval_summary["official_cached_subset"][split_name] = _evaluate_qwen_model(
                    model=eval_model,
                    split_name=split_name,
                    samples=manifest_samples,
                    checkpoint_dir=selected_checkpoint_dir,
                    output_dir=output_dir,
                    prediction_filename=f"eval_predictions_{split_name}.jsonl",
                )
            del eval_model

            eval_summary_path = Path(training_cfg.get("output_dir", "outputs/sft_qwen_stagea")) / "eval_summary.json"
            collapse_diagnostics_path = (
                Path(training_cfg.get("output_dir", "outputs/sft_qwen_stagea")) / "collapse_diagnostics.json"
            )
            collapse_summary = {
                "status": "ok",
                "checkpoint_selection_rule": checkpoint_selection_rule,
                "selected_checkpoint_dir": selected_checkpoint_dir,
                "checkpoint_candidates": checkpoint_candidates,
                "selected_internal_eval": eval_summary["internal_eval"],
            }
            save_json(eval_summary, eval_summary_path)
            save_json(collapse_summary, collapse_diagnostics_path)

            augmented_summary = dict(result)
            augmented_summary.update(
                {
                    "data_split_mode": split_mode,
                    "selection_strategy": train_cfg.get("selection_strategy", "stratified_by_website"),
                    "load_max_samples": train_cfg.get("load_max_samples", train_cfg.get("max_samples")),
                    "selected_max_samples": train_cfg.get("max_samples"),
                    "val_ratio": float(data_cfg.get("val_ratio", 0.1)),
                    "train_stats": train_stats,
                    "eval_stats": eval_stats,
                    "selected_checkpoint_dir": selected_checkpoint_dir,
                    "checkpoint_selection_rule": checkpoint_selection_rule,
                    "eval_summary_path": str(eval_summary_path),
                    "collapse_diagnostics_path": str(collapse_diagnostics_path),
                }
            )
            save_json(augmented_summary, Path(training_cfg.get("output_dir", "outputs/sft_qwen_stagea")) / "train_summary.json")
            result = augmented_summary
    logger.info("Training result: %s", result)
    logger.info("Done.")


def _exit_cleanly(exit_code: int = 0) -> NoReturn:
    """Legacy workaround for Python 3.13 finalization crashes."""
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


if __name__ == "__main__":
    main()
    if os.getenv("GUI_GROUNDING_LEGACY_HARD_EXIT", "0") == "1":
        _exit_cleanly(0)
