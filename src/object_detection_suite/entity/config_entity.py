"""Typed configuration objects (dataclasses).

These are the plain, strongly-typed structures that the rest of the codebase
consumes. They are produced by `ConfigurationManager` from the YAML files in
`configs/`, so no other module needs to touch raw dicts/YAML directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    project_name: str
    seed: int
    device: str  # "cuda" | "cpu" | "mps" | "auto"
    artifacts_dir: Path
    checkpoints_dir: Path
    logs_dir: Path


@dataclass(frozen=True)
class RoboflowConfig:
    workspace: str
    project: str
    version: int
    format: str  # e.g. "yolov8"


@dataclass(frozen=True)
class DataConfig:
    dataset_name: str
    roboflow: RoboflowConfig
    raw_dir: Path
    processed_dir: Path
    class_names: list[str]
    num_classes: int
    img_size: int
    batch_size: int
    num_workers: int
    val_split: float
    test_split: float


@dataclass(frozen=True)
class ModelTrainSpec:
    """Per-model hyperparameters for a single training run."""

    name: str
    epochs: int
    learning_rate: float
    weight_decay: float
    optimizer: str  # "adamw" | "sgd"
    scheduler: str  # "cosine" | "step" | "none"
    freeze_backbone_epochs: int
    use_amp: bool
    early_stopping_patience: int


@dataclass(frozen=True)
class TrainConfig:
    models_to_train: list[str]
    model_specs: dict[str, ModelTrainSpec]
    checkpoints_dir: Path
    log_every_n_steps: int
    use_mlflow: bool
    mlflow_experiment_name: str


@dataclass(frozen=True)
class EvalConfig:
    iou_thresholds: list[float]
    score_threshold: float
    nms_iou_threshold: float
    batch_size: int
    benchmarks_dir: Path


@dataclass(frozen=True)
class InferenceConfig:
    score_threshold: float
    nms_iou_threshold: float
    output_dir: Path
    default_model: str
