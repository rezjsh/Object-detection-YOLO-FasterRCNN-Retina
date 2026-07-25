"""Single entrypoint for the whole pipeline.

Usage:
    uv run python main.py download
    uv run python main.py train [--model NAME]
    uv run python main.py evaluate [--models NAME ...] [--split test|valid]
    uv run python main.py infer-image --input path/to/img.jpg [--model NAME]
    uv run python main.py infer-folder --input dir/ --output dir/ [--model NAME]
    uv run python main.py infer-video --input path/to/video.mp4 --output out.mp4 [--model NAME]
    uv run python main.py app
"""
from __future__ import annotations

import argparse
import logging
import sys

from object_detection_suite.config.configuration import ConfigurationManager
from object_detection_suite.models.model_factory import ModelFactory
from object_detection_suite.train.checkpointing import checkpoint_path, load_checkpoint
from object_detection_suite.utils.common import setup_logging

logger = logging.getLogger(__name__)


def _load_predictor(model_name: str):
    from object_detection_suite.infer.predictor import Predictor

    cfg_manager = ConfigurationManager()
    project_cfg = cfg_manager.get_project_config()
    data_cfg = cfg_manager.get_data_config()
    train_cfg = cfg_manager.get_train_config()
    inference_cfg = cfg_manager.get_inference_config()

    model = ModelFactory.create(model_name, num_classes=data_cfg.num_classes, pretrained_backbone=False)
    ckpt_path = checkpoint_path(train_cfg.checkpoints_dir, model_name, "best")
    load_checkpoint(ckpt_path, model, map_location=project_cfg.device)

    return Predictor(
        model=model,
        class_names=data_cfg.class_names,
        device=project_cfg.device,
        img_size=data_cfg.img_size,
        score_threshold=inference_cfg.score_threshold,
        nms_iou_threshold=inference_cfg.nms_iou_threshold,
    )


def cmd_download(args: argparse.Namespace) -> None:
    from scripts.download_dataset import main as download_main

    sys.argv = ["download_dataset.py"] + (["--max-images", str(args.max_images)] if args.max_images else [])
    download_main()


def cmd_train(args: argparse.Namespace) -> None:
    from scripts.train_all import main as train_main

    sys.argv = ["train_all.py"] + (["--model", args.model] if args.model else [])
    train_main()


def cmd_evaluate(args: argparse.Namespace) -> None:
    from scripts.evaluate_all import main as evaluate_main

    sys.argv = ["evaluate_all.py", "--split", args.split] + (["--models", *args.models] if args.models else [])
    evaluate_main()


def cmd_infer_image(args: argparse.Namespace) -> None:
    cfg_manager = ConfigurationManager()
    model_name = args.model or cfg_manager.get_inference_config().default_model
    predictor = _load_predictor(model_name)
    output = args.output or f"artifacts/predictions/{model_name}_output.jpg"
    result = predictor.predict_image(args.input, output)
    n = len(result["predictions"]["boxes"])
    logger.info("%d detection(s) found. Saved to %s", n, output)


def cmd_infer_folder(args: argparse.Namespace) -> None:
    cfg_manager = ConfigurationManager()
    model_name = args.model or cfg_manager.get_inference_config().default_model
    predictor = _load_predictor(model_name)
    results = predictor.predict_folder(args.input, args.output)
    logger.info("Processed %d images -> %s", len(results), args.output)


def cmd_infer_video(args: argparse.Namespace) -> None:
    from object_detection_suite.infer.video_inference import run_video_inference

    cfg_manager = ConfigurationManager()
    model_name = args.model or cfg_manager.get_inference_config().default_model
    predictor = _load_predictor(model_name)
    result = run_video_inference(predictor, args.input, args.output, frame_stride=args.frame_stride)
    logger.info("Video inference done: %s", result)


def cmd_app(args: argparse.Namespace) -> None:
    import subprocess

    subprocess.run(["streamlit", "run", "app/app.py"], check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Object Detection Suite CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("download").add_argument("--max-images", type=int, default=None)

    p_train = sub.add_parser("train")
    p_train.add_argument("--model", type=str, default=None)

    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--models", nargs="+", default=None)
    p_eval.add_argument("--split", type=str, default="test", choices=["valid", "test"])

    p_img = sub.add_parser("infer-image")
    p_img.add_argument("--input", type=str, required=True)
    p_img.add_argument("--output", type=str, default=None)
    p_img.add_argument("--model", type=str, default=None, choices=ModelFactory.available_models())

    p_folder = sub.add_parser("infer-folder")
    p_folder.add_argument("--input", type=str, required=True)
    p_folder.add_argument("--output", type=str, required=True)
    p_folder.add_argument("--model", type=str, default=None, choices=ModelFactory.available_models())

    p_video = sub.add_parser("infer-video")
    p_video.add_argument("--input", type=str, required=True)
    p_video.add_argument("--output", type=str, required=True)
    p_video.add_argument("--model", type=str, default=None, choices=ModelFactory.available_models())
    p_video.add_argument("--frame-stride", type=int, default=1)

    sub.add_parser("app")
    return parser


def main() -> None:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "download": cmd_download,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "infer-image": cmd_infer_image,
        "infer-folder": cmd_infer_folder,
        "infer-video": cmd_infer_video,
        "app": cmd_app,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
