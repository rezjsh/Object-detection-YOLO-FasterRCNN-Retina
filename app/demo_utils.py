"""Helper functions for the Streamlit demo app: cached config/model loading
so switching the selected model in the UI doesn't reload everything from
scratch on every interaction."""
from __future__ import annotations

import streamlit as st

from object_detection_suite.config.configuration import ConfigurationManager
from object_detection_suite.infer.predictor import Predictor
from object_detection_suite.models.model_factory import ModelFactory
from object_detection_suite.train.checkpointing import checkpoint_path, load_checkpoint


@st.cache_resource(show_spinner=False)
def get_config_manager() -> ConfigurationManager:
    return ConfigurationManager()


@st.cache_resource(show_spinner="Loading model...")
def load_predictor(model_name: str, score_threshold: float, nms_iou_threshold: float) -> Predictor:
    cfg_manager = get_config_manager()
    project_cfg = cfg_manager.get_project_config()
    data_cfg = cfg_manager.get_data_config()
    train_cfg = cfg_manager.get_train_config()

    model = ModelFactory.create(model_name, num_classes=data_cfg.num_classes, pretrained_backbone=False)
    ckpt_path = checkpoint_path(train_cfg.checkpoints_dir, model_name, "best")
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found for '{model_name}' at {ckpt_path}. Train it first with:\n"
            f"  uv run python main.py train --model {model_name}"
        )
    load_checkpoint(ckpt_path, model, map_location=project_cfg.device)

    return Predictor(
        model=model,
        class_names=data_cfg.class_names,
        device=project_cfg.device,
        img_size=data_cfg.img_size,
        score_threshold=score_threshold,
        nms_iou_threshold=nms_iou_threshold,
    )


def available_models_with_checkpoints() -> list[str]:
    cfg_manager = get_config_manager()
    train_cfg = cfg_manager.get_train_config()
    available = []
    for name in ModelFactory.available_models():
        if checkpoint_path(train_cfg.checkpoints_dir, name, "best").exists():
            available.append(name)
    return available
