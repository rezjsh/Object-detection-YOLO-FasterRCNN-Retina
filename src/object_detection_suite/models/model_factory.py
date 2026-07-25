"""Factory pattern for detector construction.

`Trainer`, `Evaluator`, and the inference scripts all create models through
`ModelFactory.create(name, ...)` so swapping model families never requires
touching training/eval/inference code — only this registry.
"""
from __future__ import annotations

from typing import Callable

from object_detection_suite.constants.constants import MODEL_FASTER_RCNN, MODEL_RETINANET, MODEL_YOLO
from object_detection_suite.models.base import BaseDetector
from object_detection_suite.models.faster_rcnn_model import FasterRCNNModel
from object_detection_suite.models.retinanet_model import RetinaNetModel
from object_detection_suite.models.yolo_model import YoloStyleDetector

ModelBuilder = Callable[..., BaseDetector]

_MODEL_REGISTRY: dict[str, ModelBuilder] = {
    MODEL_YOLO: YoloStyleDetector,
    MODEL_FASTER_RCNN: FasterRCNNModel,
    MODEL_RETINANET: RetinaNetModel,
}


class ModelFactory:
    @staticmethod
    def available_models() -> list[str]:
        return list(_MODEL_REGISTRY.keys())

    @staticmethod
    def create(name: str, num_classes: int, **kwargs) -> BaseDetector:
        if name not in _MODEL_REGISTRY:
            raise KeyError(
                f"Unknown model '{name}'. Available models: {ModelFactory.available_models()}"
            )
        builder = _MODEL_REGISTRY[name]
        
        # Safely extract img_size if the model type doesn't need it
        if name in [MODEL_FASTER_RCNN, MODEL_RETINANET]:
            kwargs.pop("img_size", None)
            
        model = builder(num_classes=num_classes, **kwargs)
        model.name = name
        return model

    @staticmethod
    def register(name: str, builder: ModelBuilder) -> None:
        """Allows extending the registry at runtime (e.g. from a notebook)
        without editing this file."""
        _MODEL_REGISTRY[name] = builder
