"""Model-agnostic training loop.

Because every `BaseDetector` subclass returns `{"loss": tensor, ...}` in
train mode and a prediction list in eval mode, this trainer never needs to
know which model family (YOLO-style / Faster R-CNN / RetinaNet) it is
driving — swapping models is purely a config change.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from object_detection_suite.entity.config_entity import ModelTrainSpec
from object_detection_suite.eval.evaluator import Evaluator
from object_detection_suite.models.base import BaseDetector
from object_detection_suite.train.checkpointing import checkpoint_path, save_checkpoint
from object_detection_suite.utils.common import save_json

logger = logging.getLogger(__name__)


def _build_optimizer(model: torch.nn.Module, spec: ModelTrainSpec) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if spec.optimizer.lower() == "sgd":
        return torch.optim.SGD(params, lr=spec.learning_rate, momentum=0.9, weight_decay=spec.weight_decay)
    return torch.optim.AdamW(params, lr=spec.learning_rate, weight_decay=spec.weight_decay)


def _build_scheduler(optimizer: torch.optim.Optimizer, spec: ModelTrainSpec):
    if spec.scheduler.lower() == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(spec.epochs, 1))
    if spec.scheduler.lower() == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(spec.epochs // 3, 1), gamma=0.1)
    return None


class Trainer:
    def __init__(
        self,
        model: BaseDetector,
        train_loader: DataLoader,
        val_loader: DataLoader,
        spec: ModelTrainSpec,
        num_classes: int,
        device: str,
        checkpoints_dir: Path,
        log_every_n_steps: int = 20,
        iou_thresholds: list[float] | None = None,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.spec = spec
        self.device = device
        self.checkpoints_dir = Path(checkpoints_dir)
        self.log_every_n_steps = log_every_n_steps

        self.optimizer = _build_optimizer(self.model, spec)
        self.scheduler = _build_scheduler(self.optimizer, spec)
        self.scaler = torch.amp.GradScaler(enabled=spec.use_amp and device == "cuda")

        self.evaluator = Evaluator(
            model=self.model,
            dataloader=self.val_loader,
            num_classes=num_classes,
            device=device,
            iou_thresholds=iou_thresholds,
        )

        self.history: list[dict] = []
        self.best_metric = -1.0
        self.best_epoch = -1
        self.epochs_since_improvement = 0

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        running_loss = 0.0
        n_batches = len(self.train_loader)

        for step, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda" if self.device == "cuda" else "cpu", enabled=self.scaler.is_enabled()):
                loss_dict = self.model(images, targets)
                loss = loss_dict["loss"]

            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            running_loss += float(loss.item())
            if step % self.log_every_n_steps == 0:
                logger.info(
                    "[%s] epoch %d/%d step %d/%d loss=%.4f",
                    self.model.name, epoch, self.spec.epochs, step, n_batches, float(loss.item()),
                )

        return running_loss / max(n_batches, 1)

    def fit(self) -> dict:
        logger.info(
            "Starting training for '%s': %d epochs, device=%s, params=%d",
            self.model.name, self.spec.epochs, self.device, self.model.count_parameters(),
        )

        if self.spec.freeze_backbone_epochs > 0:
            self.model.freeze_backbone()
            logger.info("Backbone frozen for the first %d epoch(s)", self.spec.freeze_backbone_epochs)

        for epoch in range(1, self.spec.epochs + 1):
            if self.spec.freeze_backbone_epochs > 0 and epoch == self.spec.freeze_backbone_epochs + 1:
                self.model.unfreeze_backbone()
                logger.info("Backbone unfrozen at epoch %d", epoch)

            start = time.time()
            train_loss = self._train_one_epoch(epoch)
            val_metrics = self.evaluator.evaluate()
            if self.scheduler is not None:
                self.scheduler.step()
            epoch_time = time.time() - start

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "epoch_time_sec": epoch_time,
                **{f"val_{k}": v for k, v in val_metrics.items() if not isinstance(v, dict)},
            }
            self.history.append(record)
            logger.info(
                "[%s] epoch %d done in %.1fs | train_loss=%.4f val_mAP@0.5=%.4f",
                self.model.name, epoch, epoch_time, train_loss, val_metrics["mAP_50"],
            )

            monitored = val_metrics["mAP_50"]
            if monitored > self.best_metric:
                self.best_metric = monitored
                self.best_epoch = epoch
                self.epochs_since_improvement = 0
                save_checkpoint(
                    checkpoint_path(self.checkpoints_dir, self.model.name, "best"),
                    self.model,
                    self.optimizer,
                    epoch,
                    val_metrics,
                )
            else:
                self.epochs_since_improvement += 1

            save_checkpoint(
                checkpoint_path(self.checkpoints_dir, self.model.name, "last"),
                self.model,
                self.optimizer,
                epoch,
                val_metrics,
            )

            if self.epochs_since_improvement >= self.spec.early_stopping_patience:
                logger.info(
                    "Early stopping at epoch %d (no improvement for %d epochs)",
                    epoch, self.epochs_since_improvement,
                )
                break

        save_json(self.checkpoints_dir / f"{self.model.name}_history.json", {"history": self.history})
        return {"best_metric": self.best_metric, "best_epoch": self.best_epoch, "history": self.history}
