"""Real Stage-A SFT trainer for CLIP-grid baseline."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor, get_scheduler

from gui_grounding.data.schemas import GroundingSample
from gui_grounding.reward.verifiable_reward import bbox_iou
from gui_grounding.utils.logger import get_logger

logger = get_logger(__name__)

ACTION_TO_ID = {"click": 0, "type": 1, "select": 2, "hover": 3}
ID_TO_ACTION = {v: k for k, v in ACTION_TO_ID.items()}


def _safe_action_to_id(action: Any) -> int:
    if action is None:
        return -100
    text = str(action).lower()
    return ACTION_TO_ID.get(text, -100)


def _bbox_center_to_grid(
    bbox_xyxy: tuple[float, float, float, float],
    image_size: tuple[int, int],
    grid_rows: int,
    grid_cols: int,
) -> int:
    x1, y1, x2, y2 = bbox_xyxy
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    width, height = image_size
    col = int(min(grid_cols - 1, max(0, math.floor((cx / max(width, 1)) * grid_cols))))
    row = int(min(grid_rows - 1, max(0, math.floor((cy / max(height, 1)) * grid_rows))))
    return row * grid_cols + col


def _grid_to_bbox(
    grid_id: int,
    image_size: tuple[int, int],
    grid_rows: int,
    grid_cols: int,
) -> tuple[float, float, float, float]:
    width, height = image_size
    row = grid_id // grid_cols
    col = grid_id % grid_cols
    cell_w = width / grid_cols
    cell_h = height / grid_rows
    x1 = col * cell_w
    y1 = row * cell_h
    x2 = (col + 1) * cell_w
    y2 = (row + 1) * cell_h
    return (x1, y1, x2, y2)


class SFTGridDataset(Dataset):
    """Torch dataset for Stage-A CLIP-grid supervision."""

    def __init__(self, samples: list[GroundingSample], grid_rows: int, grid_cols: int) -> None:
        self.samples = samples
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        path = Path(sample.image_path)
        image = Image.open(path).convert("RGB")
        action_id = _safe_action_to_id(sample.action_type)
        grid_id = -100
        target_bbox = None
        if sample.target_bbox is not None:
            target_bbox = sample.target_bbox.as_tuple()
            grid_id = _bbox_center_to_grid(
                bbox_xyxy=target_bbox,
                image_size=image.size,
                grid_rows=self.grid_rows,
                grid_cols=self.grid_cols,
            )
        return {
            "sample_id": sample.sample_id,
            "image": image,
            "instruction": sample.instruction,
            "action_id": action_id,
            "grid_id": grid_id,
            "target_bbox": target_bbox,
            "image_size": image.size,
        }


@dataclass
class CLIPGridSFTCollator:
    """Collator using CLIPProcessor for image-text tensorization."""

    processor: CLIPProcessor

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        images = [item["image"] for item in batch]
        texts = [item["instruction"] for item in batch]
        model_inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        action_labels = torch.tensor([item["action_id"] for item in batch], dtype=torch.long)
        grid_labels = torch.tensor([item["grid_id"] for item in batch], dtype=torch.long)
        gt_bboxes = []
        for item in batch:
            if item["target_bbox"] is None:
                gt_bboxes.append([float("nan")] * 4)
            else:
                gt_bboxes.append(list(item["target_bbox"]))
        gt_bbox_tensor = torch.tensor(gt_bboxes, dtype=torch.float32)
        return {
            **model_inputs,
            "action_labels": action_labels,
            "grid_labels": grid_labels,
            "gt_bboxes": gt_bbox_tensor,
            "sample_ids": [item["sample_id"] for item in batch],
            "image_sizes": [item["image_size"] for item in batch],
        }


class SFTTrainer:
    """Train/val/checkpoint trainer for Stage-A SFT baseline."""

    def __init__(
        self,
        model: Any,
        processor: CLIPProcessor,
        train_samples: list[GroundingSample],
        eval_samples: list[GroundingSample],
        output_dir: str | Path = "outputs/sft",
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        num_epochs: int = 3,
        batch_size: int = 8,
        gradient_accumulation_steps: int = 1,
        warmup_ratio: float = 0.1,
        max_grad_norm: float = 1.0,
        save_steps: int = 100,
        eval_steps: int = 50,
        logging_steps: int = 10,
        seed: int = 42,
        use_wandb: bool = False,
        wandb_project: str = "gui-grounding",
        wandb_run_name: str = "sft-baseline",
        grid_rows: int = 4,
        grid_cols: int = 6,
        loss_weights: Optional[dict[str, float]] = None,
        num_workers: int = 0,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.processor = processor
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_ratio = warmup_ratio
        self.max_grad_norm = max_grad_norm
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps
        self.seed = seed
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.num_workers = num_workers
        self.loss_weights = loss_weights or {"action": 1.0, "grid": 1.0}

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = self.model.to(self.device)

        self.train_dataset = SFTGridDataset(train_samples, grid_rows=grid_rows, grid_cols=grid_cols)
        self.eval_dataset = SFTGridDataset(eval_samples, grid_rows=grid_rows, grid_cols=grid_cols)
        self.collator = CLIPGridSFTCollator(processor=processor)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collator,
        )
        self.eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collator,
        )

        self.optimizer = AdamW(
            params=[p for p in self.model.parameters() if p.requires_grad],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        total_train_steps = max(
            1,
            math.ceil(len(self.train_loader) / max(self.gradient_accumulation_steps, 1)) * self.num_epochs,
        )
        warmup_steps = int(total_train_steps * self.warmup_ratio)
        self.scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_train_steps,
        )

        self._global_step = 0
        self._best_val_loss = float("inf")
        self.eval_history: list[dict[str, float]] = []
        self.wandb_run = self._maybe_init_wandb()
        logger.info(
            "SFTTrainer initialized: device=%s train=%d eval=%d output=%s",
            self.device,
            len(self.train_dataset),
            len(self.eval_dataset),
            self.output_dir,
        )

    def _maybe_init_wandb(self):
        if not self.use_wandb:
            return None
        api_key = os.getenv("WANDB_API_KEY")
        if not api_key:
            logger.warning("WANDB_API_KEY not found in environment. Falling back to local logging.")
            return None
        try:
            import wandb

            return wandb.init(project=self.wandb_project, name=self.wandb_run_name)
        except Exception as exc:
            logger.warning("wandb init failed: %s. Falling back to local logging.", exc)
            return None

    def _to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        tensor_keys = ["input_ids", "attention_mask", "pixel_values", "action_labels", "grid_labels", "gt_bboxes"]
        result = dict(batch)
        for key in tensor_keys:
            result[key] = result[key].to(self.device)
        return result

    def _compute_loss_and_metrics(self, batch: dict[str, Any], train_mode: bool) -> dict[str, Any]:
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
        )
        action_logits = outputs["action_logits"]
        grid_logits = outputs["grid_logits"]
        action_labels = batch["action_labels"]
        grid_labels = batch["grid_labels"]

        action_loss = F.cross_entropy(action_logits, action_labels, ignore_index=-100)
        grid_loss = F.cross_entropy(grid_logits, grid_labels, ignore_index=-100)
        loss = self.loss_weights["action"] * action_loss + self.loss_weights["grid"] * grid_loss

        with torch.no_grad():
            action_pred = torch.argmax(action_logits, dim=-1)
            grid_pred = torch.argmax(grid_logits, dim=-1)

            action_mask = action_labels != -100
            grid_mask = grid_labels != -100

            action_acc = (
                (action_pred[action_mask] == action_labels[action_mask]).float().mean().item()
                if action_mask.any()
                else 0.0
            )
            grid_acc = (
                (grid_pred[grid_mask] == grid_labels[grid_mask]).float().mean().item()
                if grid_mask.any()
                else 0.0
            )

            click_hits = 0.0
            iou_values: list[float] = []
            for idx, valid in enumerate(grid_mask.tolist()):
                if not valid:
                    continue
                image_size = batch["image_sizes"][idx]
                pred_bbox = _grid_to_bbox(
                    grid_id=int(grid_pred[idx].item()),
                    image_size=image_size,
                    grid_rows=self.grid_rows,
                    grid_cols=self.grid_cols,
                )
                gt_bbox_arr = batch["gt_bboxes"][idx].detach().cpu().tolist()
                if any(math.isnan(v) for v in gt_bbox_arr):
                    continue
                gt_bbox = tuple(float(v) for v in gt_bbox_arr)
                px = (pred_bbox[0] + pred_bbox[2]) / 2
                py = (pred_bbox[1] + pred_bbox[3]) / 2
                if gt_bbox[0] <= px <= gt_bbox[2] and gt_bbox[1] <= py <= gt_bbox[3]:
                    click_hits += 1.0
                iou_values.append(bbox_iou(pred_bbox, gt_bbox))

            point_acc = click_hits / max(int(grid_mask.sum().item()), 1)
            mean_iou = sum(iou_values) / max(len(iou_values), 1)

        metrics = {
            "loss": float(loss.item()),
            "action_loss": float(action_loss.item()),
            "grid_loss": float(grid_loss.item()),
            "action_acc": float(action_acc),
            "grid_acc": float(grid_acc),
            "point_acc": float(point_acc),
            "mean_iou": float(mean_iou),
        }
        if train_mode:
            metrics["lr"] = float(self.optimizer.param_groups[0]["lr"])
        return {"loss_tensor": loss, "metrics": metrics}

    def _log(self, metrics: dict[str, float], step: int, prefix: str) -> None:
        formatted = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float)])
        logger.info("[%s step=%d] %s", prefix, step, formatted)
        if self.wandb_run is not None:
            payload = {f"{prefix}/{k}": v for k, v in metrics.items()}
            payload["global_step"] = step
            self.wandb_run.log(payload)

    def save_checkpoint(self, tag: str = "latest") -> Path:
        ckpt_dir = self.output_dir / f"checkpoint-{tag}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), ckpt_dir / "model.pt")
        torch.save(self.optimizer.state_dict(), ckpt_dir / "optimizer.pt")
        torch.save(self.scheduler.state_dict(), ckpt_dir / "scheduler.pt")
        torch.save(
            {"global_step": self._global_step, "best_val_loss": self._best_val_loss},
            ckpt_dir / "trainer_state.pt",
        )
        with open(ckpt_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "global_step": self._global_step,
                    "best_val_loss": self._best_val_loss,
                    "grid_rows": self.grid_rows,
                    "grid_cols": self.grid_cols,
                    "num_train_samples": len(self.train_dataset),
                    "num_eval_samples": len(self.eval_dataset),
                },
                f,
                indent=2,
            )
        logger.info("Saved checkpoint: %s", ckpt_dir)
        return ckpt_dir

    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        aggregate = {
            "loss": 0.0,
            "action_loss": 0.0,
            "grid_loss": 0.0,
            "action_acc": 0.0,
            "grid_acc": 0.0,
            "point_acc": 0.0,
            "mean_iou": 0.0,
        }
        num_batches = 0
        with torch.inference_mode():
            for batch in self.eval_loader:
                batch = self._to_device(batch)
                result = self._compute_loss_and_metrics(batch, train_mode=False)
                for key in aggregate:
                    aggregate[key] += result["metrics"][key]
                num_batches += 1

        metrics = {f"val_{k}": (v / max(num_batches, 1)) for k, v in aggregate.items()}
        return metrics

    def _save_eval_history(self) -> None:
        path = self.output_dir / "eval_history.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.eval_history, f, indent=2)
        logger.info("Saved eval history: %s", path)

    def train(self) -> dict[str, Any]:
        logger.info("=" * 60)
        logger.info("Stage-A SFT Training (CLIP-grid baseline)")
        logger.info("=" * 60)
        logger.info(
            "Config: epochs=%d batch=%d lr=%.2e grad_acc=%d",
            self.num_epochs,
            self.batch_size,
            self.learning_rate,
            self.gradient_accumulation_steps,
        )

        self.model.train()
        running = {
            "loss": 0.0,
            "action_loss": 0.0,
            "grid_loss": 0.0,
            "action_acc": 0.0,
            "grid_acc": 0.0,
            "point_acc": 0.0,
            "mean_iou": 0.0,
        }
        running_steps = 0

        for epoch in range(self.num_epochs):
            logger.info("Epoch %d/%d", epoch + 1, self.num_epochs)
            for step_idx, batch in enumerate(self.train_loader, start=1):
                batch = self._to_device(batch)
                result = self._compute_loss_and_metrics(batch, train_mode=True)
                loss = result["loss_tensor"] / max(self.gradient_accumulation_steps, 1)
                loss.backward()

                for key in running:
                    running[key] += result["metrics"][key]
                running_steps += 1

                if step_idx % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self._global_step += 1

                    if self._global_step % self.logging_steps == 0:
                        avg_metrics = {k: (v / max(running_steps, 1)) for k, v in running.items()}
                        avg_metrics["lr"] = float(self.optimizer.param_groups[0]["lr"])
                        self._log(avg_metrics, step=self._global_step, prefix="train")
                        for key in running:
                            running[key] = 0.0
                        running_steps = 0

                    if self._global_step % self.eval_steps == 0:
                        val_metrics = self.evaluate()
                        self.eval_history.append({"step": self._global_step, **val_metrics})
                        self._log(val_metrics, step=self._global_step, prefix="eval")
                        self.model.train()
                        if val_metrics["val_loss"] < self._best_val_loss:
                            self._best_val_loss = val_metrics["val_loss"]
                            self.save_checkpoint(tag="best")

                    if self._global_step % self.save_steps == 0:
                        self.save_checkpoint(tag=f"step-{self._global_step}")

            # End-of-epoch validation
            val_metrics = self.evaluate()
            self.eval_history.append({"step": self._global_step, "epoch": epoch + 1, **val_metrics})
            self._log(val_metrics, step=self._global_step, prefix=f"eval_epoch{epoch + 1}")
            self.model.train()
            if val_metrics["val_loss"] < self._best_val_loss:
                self._best_val_loss = val_metrics["val_loss"]
                self.save_checkpoint(tag="best")

        final_ckpt = self.save_checkpoint(tag="latest")
        self._save_eval_history()
        summary = {
            "status": "ok",
            "global_step": self._global_step,
            "num_train_samples": len(self.train_dataset),
            "num_eval_samples": len(self.eval_dataset),
            "best_val_loss": self._best_val_loss,
            "latest_checkpoint": str(final_ckpt),
            "eval_history_path": str(self.output_dir / "eval_history.json"),
        }
        with open(self.output_dir / "train_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("Training finished. Summary saved: %s", self.output_dir / "train_summary.json")
        if self.wandb_run is not None:
            self.wandb_run.finish()
        return summary
