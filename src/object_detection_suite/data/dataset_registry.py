"""Strategy-pattern registry for dataset loaders.

Today there is a single concrete strategy (`YOLODataset` reading a
Roboflow-exported YOLOv8 split), but new dataset formats (e.g. a raw
PASCAL VOC XML strategy) can be added by registering another factory
function here without touching the training/eval code.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from torch.utils.data import Dataset

from object_detection_suite.constants.constants import IMAGES_DIR_NAME, LABELS_DIR_NAME
from object_detection_suite.data.dataset_loader import YOLODataset

DatasetFactory = Callable[..., Dataset]

_DATASET_REGISTRY: dict[str, DatasetFactory] = {}


def register_dataset(name: str) -> Callable[[DatasetFactory], DatasetFactory]:
    def decorator(factory: DatasetFactory) -> DatasetFactory:
        _DATASET_REGISTRY[name] = factory
        return factory

    return decorator


@register_dataset("yolo_v1")
def _build_yolo_dataset(split_dir: Path | str, img_size: int, transforms=None) -> Dataset:
    split_dir = Path(split_dir)
    return YOLODataset(
        images_dir=split_dir / IMAGES_DIR_NAME,
        labels_dir=split_dir / LABELS_DIR_NAME,
        img_size=img_size,
        transforms=transforms,
    )


def get_dataset(name: str, **kwargs) -> Dataset:
    if name not in _DATASET_REGISTRY:
        raise KeyError(
            f"Unknown dataset strategy '{name}'. Available: {list(_DATASET_REGISTRY)}"
        )
    return _DATASET_REGISTRY[name](**kwargs)


def list_dataset_strategies() -> list[str]:
    return list(_DATASET_REGISTRY.keys())
