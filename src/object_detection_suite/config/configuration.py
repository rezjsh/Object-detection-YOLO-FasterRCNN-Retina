"""Central configuration manager.

Every other module gets its typed config objects from here instead of
reading YAML directly. This is the single seam you touch if the on-disk
config format ever changes.
"""
from __future__ import annotations

from pathlib import Path

from object_detection_suite.constants import constants as C
from object_detection_suite.entity.config_entity import (
    DataConfig,
    EvalConfig,
    InferenceConfig,
    ModelTrainSpec,
    ProjectConfig,
    RoboflowConfig,
    TrainConfig,
)
from object_detection_suite.utils.common import get_device, read_yaml


class ConfigurationManager:
    """Loads YAML files under `configs/` and exposes typed config objects."""

    def __init__(
        self,
        project_config_path: Path | str = C.PROJECT_CONFIG_FILE,
        data_config_path: Path | str = C.DATA_CONFIG_FILE,
        train_config_path: Path | str = C.TRAIN_CONFIG_FILE,
        eval_config_path: Path | str = C.EVAL_CONFIG_FILE,
        inference_config_path: Path | str = C.INFERENCE_CONFIG_FILE,
    ) -> None:
        self._project_raw = read_yaml(project_config_path)
        self._data_raw = read_yaml(data_config_path)
        self._train_raw = read_yaml(train_config_path)
        self._eval_raw = read_yaml(eval_config_path)
        self._inference_raw = read_yaml(inference_config_path)

    def get_project_config(self) -> ProjectConfig:
        raw = self._project_raw
        device = raw.get("device", "auto")
        if device == "auto":
            device = get_device()
        return ProjectConfig(
            project_name=raw["project_name"],
            seed=int(raw.get("seed", C.DEFAULT_SEED)),
            device=device,
            artifacts_dir=C.ROOT_DIR / raw.get("artifacts_dir", "artifacts"),
            checkpoints_dir=C.ROOT_DIR / raw.get("checkpoints_dir", "artifacts/checkpoints"),
            logs_dir=C.ROOT_DIR / raw.get("logs_dir", "logs"),
        )

    def get_data_config(self) -> DataConfig:
        raw = self._data_raw
        rf = raw["roboflow"]
        return DataConfig(
            dataset_name=raw["dataset_name"],
            roboflow=RoboflowConfig(
                workspace=rf["workspace"],
                project=rf["project"],
                version=int(rf["version"]),
                format=rf.get("format", "yolov8"),
            ),
            raw_dir=C.ROOT_DIR / raw.get("raw_dir", "data/raw"),
            processed_dir=C.ROOT_DIR / raw.get("processed_dir", "data/processed"),
            class_names=list(raw["class_names"]),
            num_classes=int(raw["num_classes"]),
            img_size=int(raw.get("img_size", C.DEFAULT_IMG_SIZE)),
            batch_size=int(raw.get("batch_size", 8)),
            num_workers=int(raw.get("num_workers", C.DEFAULT_NUM_WORKERS)),
            val_split=float(raw.get("val_split", 0.15)),
            test_split=float(raw.get("test_split", 0.10)),
        )

    def get_train_config(self) -> TrainConfig:
        raw = self._train_raw
        specs = {
            name: ModelTrainSpec(
                name=name,
                epochs=int(spec["epochs"]),
                learning_rate=float(spec["learning_rate"]),
                weight_decay=float(spec["weight_decay"]),
                optimizer=spec.get("optimizer", "adamw"),
                scheduler=spec.get("scheduler", "cosine"),
                freeze_backbone_epochs=int(spec.get("freeze_backbone_epochs", 0)),
                use_amp=bool(spec.get("use_amp", True)),
                early_stopping_patience=int(spec.get("early_stopping_patience", 8)),
            )
            for name, spec in raw["model_specs"].items()
        }
        return TrainConfig(
            models_to_train=list(raw.get("models_to_train", list(specs.keys()))),
            model_specs=specs,
            checkpoints_dir=C.ROOT_DIR / raw.get("checkpoints_dir", "artifacts/checkpoints"),
            log_every_n_steps=int(raw.get("log_every_n_steps", 20)),
            use_mlflow=bool(raw.get("use_mlflow", False)),
            mlflow_experiment_name=raw.get("mlflow_experiment_name", "object_detection_suite"),
        )

    def get_eval_config(self) -> EvalConfig:
        raw = self._eval_raw
        return EvalConfig(
            iou_thresholds=[float(x) for x in raw.get("iou_thresholds", [0.5])],
            score_threshold=float(raw.get("score_threshold", 0.05)),
            nms_iou_threshold=float(raw.get("nms_iou_threshold", 0.5)),
            batch_size=int(raw.get("batch_size", 8)),
            benchmarks_dir=C.ROOT_DIR / raw.get("benchmarks_dir", "artifacts/benchmarks"),
        )

    def get_inference_config(self) -> InferenceConfig:
        raw = self._inference_raw
        return InferenceConfig(
            score_threshold=float(raw.get("score_threshold", 0.35)),
            nms_iou_threshold=float(raw.get("nms_iou_threshold", 0.45)),
            output_dir=C.ROOT_DIR / raw.get("output_dir", "artifacts/predictions"),
            default_model=raw.get("default_model", "yolo_style"),
        )
