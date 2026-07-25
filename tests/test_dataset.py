"""Tests for the data pipeline: YOLODataset reading, augmentation shapes,
and the dataset validator's error detection."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from object_detection_suite.data.augmentations import get_train_transforms, get_val_transforms
from object_detection_suite.data.dataset_loader import YOLODataset, detection_collate_fn
from object_detection_suite.data.dataset_validator import validate_split


@pytest.fixture()
def tmp_yolo_split():
    """Builds a tiny synthetic YOLO-format split: 3 images, 3 label files,
    2 classes, one image intentionally left unlabeled."""
    root = Path(tempfile.mkdtemp())
    images_dir, labels_dir = root / "images", root / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    for i in range(3):
        img = Image.fromarray((np.random.rand(120, 160, 3) * 255).astype(np.uint8))
        img.save(images_dir / f"img_{i}.jpg")

    (labels_dir / "img_0.txt").write_text("0 0.5 0.5 0.2 0.3\n1 0.2 0.2 0.1 0.1\n")
    (labels_dir / "img_1.txt").write_text("1 0.5 0.5 0.4 0.4\n")
    # img_2 intentionally has no label file (unlabeled/background image)

    yield images_dir, labels_dir
    shutil.rmtree(root)


def test_yolo_dataset_length_and_item_shape(tmp_yolo_split):
    images_dir, labels_dir = tmp_yolo_split
    ds = YOLODataset(images_dir, labels_dir, img_size=64, transforms=get_val_transforms(64))
    assert len(ds) == 3

    image, target = ds[0]
    assert image.shape == (3, 64, 64)
    assert target["boxes"].shape[1] == 4
    assert target["boxes"].shape[0] == target["labels"].shape[0]
    # labels are 1-indexed (class 0 -> label 1, class 1 -> label 2)
    assert set(target["labels"].tolist()) <= {1, 2}


def test_yolo_dataset_boxes_are_within_image_bounds(tmp_yolo_split):
    images_dir, labels_dir = tmp_yolo_split
    ds = YOLODataset(images_dir, labels_dir, img_size=64, transforms=get_val_transforms(64))
    for idx in range(len(ds)):
        _, target = ds[idx]
        if target["boxes"].numel() == 0:
            continue
        assert (target["boxes"][:, 0] >= -1).all() and (target["boxes"][:, 2] <= 65).all()
        assert (target["boxes"][:, 1] >= -1).all() and (target["boxes"][:, 3] <= 65).all()


def test_unlabeled_image_returns_empty_target(tmp_yolo_split):
    images_dir, labels_dir = tmp_yolo_split
    ds = YOLODataset(images_dir, labels_dir, img_size=64, transforms=get_val_transforms(64))
    _, target = ds[2]  # img_2 has no label file
    assert target["boxes"].shape == (0, 4)
    assert target["labels"].shape == (0,)


def test_train_transforms_produce_correct_shape(tmp_yolo_split):
    images_dir, labels_dir = tmp_yolo_split
    ds = YOLODataset(images_dir, labels_dir, img_size=96, transforms=get_train_transforms(96))
    image, _ = ds[0]
    assert image.shape == (3, 96, 96)


def test_collate_fn_stacks_variable_length_targets(tmp_yolo_split):
    images_dir, labels_dir = tmp_yolo_split
    ds = YOLODataset(images_dir, labels_dir, img_size=64, transforms=get_val_transforms(64))
    batch = [ds[i] for i in range(3)]
    images, targets = detection_collate_fn(batch)
    assert images.shape == (3, 3, 64, 64)
    assert len(targets) == 3


def test_validate_split_reports_stats(tmp_yolo_split):
    images_dir, labels_dir = tmp_yolo_split
    report = validate_split(images_dir, labels_dir, num_classes=2, split_name="unit_test")
    assert report.num_images == 3
    assert report.num_labeled_images == 2
    assert report.num_unlabeled_images == 1
    assert report.num_boxes == 3
    assert report.is_valid


def test_validate_split_flags_out_of_range_class(tmp_yolo_split):
    images_dir, labels_dir = tmp_yolo_split
    (labels_dir / "img_0.txt").write_text("5 0.5 0.5 0.2 0.3\n")  # class 5 invalid for num_classes=2
    report = validate_split(images_dir, labels_dir, num_classes=2, split_name="unit_test")
    assert len(report.errors) >= 1
    assert not report.is_valid
