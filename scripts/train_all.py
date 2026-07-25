"""Trains every model listed in `configs/train.yaml` (or a single model via
--model) on the same dataset/split, saving per-model checkpoints and
training-curve plots.

Usage:
    uv run python scripts/train_all.py
    uv run python scripts/train_all.py --model yolo_style
"""
from __future__ import annotations

import argparse
import logging

from torch.utils.data import DataLoader

from object_detection_suite.config.configuration import ConfigurationManager
from object_detection_suite.data.augmentations import get_train_transforms, get_val_transforms
from object_detection_suite.data.dataset_loader import detection_collate_fn
from object_detection_suite.data.dataset_registry import get_dataset
from object_detection_suite.models.model_factory import ModelFactory
from object_detection_suite.train.trainer import Trainer
from object_detection_suite.utils.common import create_directories, seed_everything, setup_logging
from object_detection_suite.visualize.plots import plot_training_curves

logger = logging.getLogger(__name__)


def build_dataloaders(data_cfg, img_size: int, batch_size: int, num_workers: int):
    train_ds = get_dataset(
        "yolo_v1",
        split_dir=data_cfg.processed_dir / "train",
        img_size=img_size,
        transforms=get_train_transforms(img_size),
    )
    val_ds = get_dataset(
        "yolo_v1",
        split_dir=data_cfg.processed_dir / "valid",
        img_size=img_size,
        transforms=get_val_transforms(img_size),
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=detection_collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=detection_collate_fn,
    )
    return train_loader, val_loader


def train_single_model(model_name: str, project_cfg, data_cfg, train_cfg) -> dict:
    logger.info("=" * 70)
    logger.info("Training model: %s", model_name)
    logger.info("=" * 70)

    spec = train_cfg.model_specs[model_name]
    train_loader, val_loader = build_dataloaders(
        data_cfg, data_cfg.img_size, data_cfg.batch_size, data_cfg.num_workers
    )

    model = ModelFactory.create(model_name, num_classes=data_cfg.num_classes, img_size=data_cfg.img_size)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        spec=spec,
        num_classes=data_cfg.num_classes,
        device=project_cfg.device,
        checkpoints_dir=train_cfg.checkpoints_dir,
        log_every_n_steps=train_cfg.log_every_n_steps,
    )
    result = trainer.fit()

    create_directories(["artifacts/plots"])
    plot_training_curves(result["history"], f"artifacts/plots/{model_name}_training_curves.png", model_name)
    logger.info("Finished training '%s': best mAP@0.5=%.4f at epoch %d", model_name, result["best_metric"], result["best_epoch"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one or all configured detection models")
    parser.add_argument("--model", type=str, default=None, help="Train only this model (default: all models in train.yaml)")
    args = parser.parse_args()

    setup_logging()
    cfg_manager = ConfigurationManager()
    project_cfg = cfg_manager.get_project_config()
    data_cfg = cfg_manager.get_data_config()
    train_cfg = cfg_manager.get_train_config()

    seed_everything(project_cfg.seed)
    create_directories([train_cfg.checkpoints_dir])

    models = [args.model] if args.model else train_cfg.models_to_train
    results = {}
    for model_name in models:
        results[model_name] = train_single_model(model_name, project_cfg, data_cfg, train_cfg)

    logger.info("All requested models trained: %s", list(results.keys()))


if __name__ == "__main__":
    main()
