"""Evaluates every trained model on the test split and writes the
comparison table + chart to `artifacts/benchmarks/`.

Usage:
    uv run python scripts/evaluate_all.py
    uv run python scripts/evaluate_all.py --models yolo_style faster_rcnn
"""
from __future__ import annotations

import argparse
import logging

from object_detection_suite.config.configuration import ConfigurationManager
from object_detection_suite.eval.benchmark import run_benchmark
from object_detection_suite.utils.common import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and benchmark trained models")
    parser.add_argument("--models", nargs="+", default=None, help="Subset of model names to evaluate")
    parser.add_argument("--split", type=str, default="test", choices=["valid", "test"])
    args = parser.parse_args()

    setup_logging()
    cfg_manager = ConfigurationManager()
    project_cfg = cfg_manager.get_project_config()
    data_cfg = cfg_manager.get_data_config()
    eval_cfg = cfg_manager.get_eval_config()
    train_cfg = cfg_manager.get_train_config()

    model_names = args.models or train_cfg.models_to_train

    rows = run_benchmark(
        model_names=model_names,
        data_cfg=data_cfg,
        eval_cfg=eval_cfg,
        checkpoints_dir=train_cfg.checkpoints_dir,
        device=project_cfg.device,
        split=args.split,
    )

    if not rows:
        logger.error("No models were evaluated — did you train any checkpoints yet?")
        return

    logger.info("Benchmark results:")
    for row in rows:
        logger.info(row)


if __name__ == "__main__":
    main()
