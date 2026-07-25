"""Dataset loader for YOLO-format object detection data.

The on-disk layout (as exported by Roboflow in YOLOv8 format) is:

    <split>/images/<name>.jpg
    <split>/labels/<name>.txt   # one line per box: "class cx cy w h" (normalized)

`YOLODataset` reads that layout and returns, per sample:
    image  : FloatTensor [3, H, W]           (normalized, padded to img_size)
    target : {
        "boxes": FloatTensor [N, 4] in absolute xyxy pixel coords,
        "labels": Int64Tensor [N] (1-indexed; 0 is reserved for background,
                  matching torchvision's convention, and the YOLO-style head
                  simply treats 0 as "no object" too),
        "image_id": Int64Tensor [1],
    }

This single representation is consumed by all three model families via
per-model adapters in `models/`, so the data pipeline never has to know
which detector it is feeding.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from object_detection_suite.constants.constants import IMAGE_EXTENSIONS, IMAGES_DIR_NAME, LABELS_DIR_NAME

logger = logging.getLogger(__name__)


def _find_images(images_dir: Path) -> list[Path]:
    return sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _read_yolo_label_file(label_path: Path) -> tuple[list[list[float]], list[int]]:
    """Returns (boxes_yolo_normalized, class_ids). Empty lists if no file/boxes."""
    if not label_path.exists():
        return [], []
    boxes, class_ids = [], []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, cx, cy, w, h = parts
            boxes.append([float(cx), float(cy), float(w), float(h)])
            class_ids.append(int(cls))
    return boxes, class_ids


class YOLODataset(Dataset):
    """Reads a single split (train/valid/test) of a YOLO-format dataset."""

    def __init__(
        self,
        images_dir: Path | str,
        labels_dir: Path | str,
        img_size: int = 640,
        transforms: Callable | None = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.img_size = img_size
        self.transforms = transforms

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        self.image_paths = _find_images(self.images_dir)
        if not self.image_paths:
            raise RuntimeError(f"No images found in {self.images_dir}")
        logger.info("Loaded %d images from %s", len(self.image_paths), self.images_dir)

    def __len__(self) -> int:
        return len(self.image_paths)

    def _label_path_for(self, image_path: Path) -> Path:
        return self.labels_dir / f"{image_path.stem}.txt"

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        image_path = self.image_paths[idx]
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes_yolo, class_ids = _read_yolo_label_file(self._label_path_for(image_path))

        if self.transforms is not None:
            transformed = self.transforms(image=image, bboxes=boxes_yolo, class_labels=class_ids)
            image_t = transformed["image"]
            boxes_yolo = transformed["bboxes"]
            class_ids = transformed["class_labels"]
            h, w = image_t.shape[1], image_t.shape[2]
        else:
            image_t = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            h, w = image.shape[0], image.shape[1]

        boxes_xyxy = _yolo_to_xyxy_abs(boxes_yolo, img_w=w, img_h=h)

        target = {
            # +1 shifts YOLO's 0-indexed classes to 1-indexed, keeping 0 free
            # for "background" the way torchvision's detection models expect.
            "boxes": torch.as_tensor(boxes_xyxy, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor([c + 1 for c in class_ids], dtype=torch.int64),
            "image_id": torch.as_tensor([idx], dtype=torch.int64),
        }
        return image_t, target

    def get_image_path(self, idx: int) -> Path:
        return self.image_paths[idx]


def _yolo_to_xyxy_abs(boxes_yolo: list[list[float]], img_w: int, img_h: int) -> list[list[float]]:
    out = []
    for cx, cy, w, h in boxes_yolo:
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        out.append([x1, y1, x2, y2])
    return out


def detection_collate_fn(batch: list[tuple[torch.Tensor, dict]]) -> tuple[torch.Tensor, list[dict]]:
    """Stacks images into a single batch tensor (they are already padded to a
    fixed img_size by the transforms) and keeps targets as a list of dicts,
    since box counts vary per image."""
    images = torch.stack([b[0] for b in batch], dim=0)
    targets = [b[1] for b in batch]
    return images, targets
