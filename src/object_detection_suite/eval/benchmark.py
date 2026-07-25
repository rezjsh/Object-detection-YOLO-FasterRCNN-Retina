"""Loads every trained model's best checkpoint, evaluates it on the test
split, and writes a comparison table (CSV + markdown + PNG chart) to
`artifacts/benchmarks/`."""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from object_detection_suite.data.dataset_loader import detection_collate_fn
from object_detection_suite.data.dataset_registry import get_dataset
from object_detection_suite.data.augmentations import get_val_transforms
from object_detection_suite.entity.config_entity import DataConfig, EvalConfig
from object_detection_suite.eval.evaluator import Evaluator
from object_detection_suite.models.model_factory import ModelFactory
from object_detection_suite.train.checkpointing import checkpoint_path, load_checkpoint
from object_detection_suite.utils.common import create_directories
from object_detection_suite.visualize.plots import plot_benchmark_comparison

logger = logging.getLogger(__name__)


def run_benchmark(
    model_names: list[str],
    data_cfg: DataConfig,
    eval_cfg: EvalConfig,
    checkpoints_dir: Path,
    device: str,
    split: str = "test",
) -> list[dict]:
    test_dataset = get_dataset(
        "yolo_v1",
        split_dir=data_cfg.processed_dir / split,
        img_size=data_cfg.img_size,
        transforms=get_val_transforms(data_cfg.img_size),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        collate_fn=detection_collate_fn,
    )

    rows = []
    for model_name in model_names:
        ckpt_path = checkpoint_path(checkpoints_dir, model_name, "best")
        if not ckpt_path.exists():
            logger.warning("No checkpoint found for '%s' at %s, skipping", model_name, ckpt_path)
            continue

        model = ModelFactory.create(model_name, num_classes=data_cfg.num_classes, pretrained_backbone=False)
        load_checkpoint(ckpt_path, model, map_location=device)

        evaluator = Evaluator(
            model=model,
            dataloader=test_loader,
            num_classes=data_cfg.num_classes,
            device=device,
            iou_thresholds=eval_cfg.iou_thresholds,
        )
        metrics = evaluator.evaluate()
        rows.append({
            "model": model_name,
            "mAP_50": round(metrics["mAP_50"], 4),
            "mAP_50_95": round(metrics["mAP_50_95"], 4),
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "mean_iou": round(metrics["mean_iou"], 4),
            "avg_latency_ms": round(metrics["avg_latency_ms"], 2),
            "fps": round(metrics["fps"], 2),
            "params": model.count_parameters(),
        })

    _save_results(rows, eval_cfg.benchmarks_dir)
    return rows


def _save_results(rows: list[dict], benchmarks_dir: Path) -> None:
    create_directories([benchmarks_dir])
    if not rows:
        logger.warning("No benchmark rows to save (no checkpoints found).")
        return

    csv_path = Path(benchmarks_dir) / "benchmark_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved benchmark CSV to %s", csv_path)

    md_path = Path(benchmarks_dir) / "benchmark_results.md"
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved benchmark markdown table to %s", md_path)

    try:
        plot_benchmark_comparison(rows, Path(benchmarks_dir) / "benchmark_comparison.png")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not render benchmark chart: %s", exc)
