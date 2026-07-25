"""Exports a trained model's best checkpoint to TorchScript.

Note: Faster R-CNN / RetinaNet (torchvision detection models) are traced via
`torch.jit.script` where possible; the YOLO-style model supports both trace
and script cleanly since it's a plain conv stack.

Usage:
    uv run python scripts/export_model.py --model yolo_style
"""
from __future__ import annotations

import argparse
import logging

import torch

from object_detection_suite.config.configuration import ConfigurationManager
from object_detection_suite.models.model_factory import ModelFactory
from object_detection_suite.train.checkpointing import checkpoint_path, load_checkpoint
from object_detection_suite.utils.common import create_directories, setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a trained model checkpoint")
    parser.add_argument("--model", type=str, required=True, choices=ModelFactory.available_models())
    parser.add_argument("--format", type=str, default="torchscript", choices=["torchscript", "state_dict"])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    setup_logging()
    cfg_manager = ConfigurationManager()
    data_cfg = cfg_manager.get_data_config()
    train_cfg = cfg_manager.get_train_config()

    model = ModelFactory.create(args.model, num_classes=data_cfg.num_classes, pretrained_backbone=False)
    ckpt_path = checkpoint_path(train_cfg.checkpoints_dir, args.model, "best")
    load_checkpoint(ckpt_path, model, map_location="cpu")
    model.eval()

    out_dir = "artifacts/exported"
    create_directories([out_dir])
    output_path = args.output or f"{out_dir}/{args.model}_exported.{'pt' if args.format == 'torchscript' else 'pth'}"

    if args.format == "state_dict":
        torch.save(model.state_dict(), output_path)
    else:
        try:
            scripted = torch.jit.script(model)
            scripted.save(output_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("torch.jit.script failed (%s); falling back to state_dict export", exc)
            output_path = output_path.replace(".pt", ".pth")
            torch.save(model.state_dict(), output_path)

    logger.info("Exported '%s' to %s", args.model, output_path)


if __name__ == "__main__":
    main()
