"""Runs a trained detector over a dataloader and reports the full metric
suite: mAP@0.5, mAP@0.5:0.95, precision, recall, mean IoU, and inference
latency/FPS."""
from __future__ import annotations

import logging

import torch
from torch.utils.data import DataLoader

from object_detection_suite.eval.metrics import MeanAveragePrecisionCalculator
from object_detection_suite.models.base import BaseDetector
from object_detection_suite.utils.common import Timer

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(
        self,
        model: BaseDetector,
        dataloader: DataLoader,
        num_classes: int,
        device: str,
        iou_thresholds: list[float] | None = None,
    ) -> None:
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.metric_calc = MeanAveragePrecisionCalculator(
            num_classes=num_classes + 1,  # +1 background, matches BaseDetector convention
            iou_thresholds=iou_thresholds or [0.5],
        )

    @torch.no_grad()
    def evaluate(self, max_batches: int | None = None) -> dict:
        self.model.eval()
        self.metric_calc.reset()

        total_images, total_time = 0, 0.0
        for batch_idx, (images, targets) in enumerate(self.dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = images.to(self.device)

            with Timer() as timer:
                predictions = self.model(images)
                if self.device == "cuda":
                    torch.cuda.synchronize()
            total_time += timer.elapsed
            total_images += images.shape[0]

            self.metric_calc.update(predictions, targets)

        metrics = self.metric_calc.compute()
        avg_latency_ms = (total_time / max(total_images, 1)) * 1000.0
        fps = total_images / total_time if total_time > 0 else 0.0
        metrics.update({
            "avg_latency_ms": avg_latency_ms,
            "fps": fps,
            "num_images_evaluated": total_images,
        })
        logger.info(
            "Eval[%s]: mAP@0.5=%.4f mAP@0.5:0.95=%.4f precision=%.4f recall=%.4f "
            "mean_iou=%.4f latency=%.2fms fps=%.1f",
            getattr(self.model, "name", "model"),
            metrics["mAP_50"],
            metrics["mAP_50_95"],
            metrics["precision"],
            metrics["recall"],
            metrics["mean_iou"],
            avg_latency_ms,
            fps,
        )
        return metrics
