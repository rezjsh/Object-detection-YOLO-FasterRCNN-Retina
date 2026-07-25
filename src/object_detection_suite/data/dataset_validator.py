"""Validates a YOLO-format dataset split before training ever starts.

Catches the usual suspects: missing/mismatched image-label pairs, label
files with out-of-range class indices, malformed lines, unreadable images,
and empty splits. Also reports basic class-distribution stats so imbalance
is visible up front.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from object_detection_suite.constants.constants import IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    split: str
    num_images: int = 0
    num_labeled_images: int = 0
    num_unlabeled_images: int = 0
    num_boxes: int = 0
    class_counts: Counter = field(default_factory=Counter)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0 and self.num_images > 0

    def summary(self) -> str:
        lines = [
            f"Split '{self.split}': {self.num_images} images, "
            f"{self.num_labeled_images} labeled, {self.num_unlabeled_images} unlabeled, "
            f"{self.num_boxes} total boxes",
            f"  class distribution: {dict(self.class_counts)}",
        ]
        if self.warnings:
            lines.append(f"  {len(self.warnings)} warning(s), e.g. {self.warnings[:3]}")
        if self.errors:
            lines.append(f"  {len(self.errors)} ERROR(s), e.g. {self.errors[:3]}")
        return "\n".join(lines)


def validate_split(
    images_dir: Path | str,
    labels_dir: Path | str,
    num_classes: int,
    split_name: str = "split",
    check_image_integrity: bool = True,
) -> ValidationReport:
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    report = ValidationReport(split=split_name)

    if not images_dir.exists():
        report.errors.append(f"Images dir does not exist: {images_dir}")
        return report

    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    report.num_images = len(image_paths)
    if report.num_images == 0:
        report.errors.append(f"No images found in {images_dir}")
        return report

    for img_path in image_paths:
        if check_image_integrity:
            img = cv2.imread(str(img_path))
            if img is None:
                report.errors.append(f"Unreadable/corrupt image: {img_path}")
                continue

        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            report.num_unlabeled_images += 1
            continue

        n_boxes_this_file = 0
        with open(label_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    report.errors.append(
                        f"{label_path.name}:{line_no} malformed line (expected 5 fields, got {len(parts)})"
                    )
                    continue
                try:
                    cls_id = int(parts[0])
                    cx, cy, w, h = (float(x) for x in parts[1:])
                except ValueError:
                    report.errors.append(f"{label_path.name}:{line_no} non-numeric fields")
                    continue

                if not (0 <= cls_id < num_classes):
                    report.errors.append(
                        f"{label_path.name}:{line_no} class id {cls_id} out of range [0, {num_classes})"
                    )
                    continue
                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    report.warnings.append(
                        f"{label_path.name}:{line_no} suspicious normalized box values"
                    )

                report.class_counts[cls_id] += 1
                n_boxes_this_file += 1
                report.num_boxes += 1

        if n_boxes_this_file > 0:
            report.num_labeled_images += 1
        else:
            report.num_unlabeled_images += 1

    if report.num_unlabeled_images / max(report.num_images, 1) > 0.5:
        report.warnings.append(
            f"More than half of images in '{split_name}' have no boxes; "
            "double-check the export/split."
        )

    return report


def validate_dataset(processed_dir: Path | str, num_classes: int, splits: tuple[str, ...] = ("train", "valid", "test")) -> dict[str, ValidationReport]:
    processed_dir = Path(processed_dir)
    reports = {}
    for split in splits:
        split_dir = processed_dir / split
        if not split_dir.exists():
            logger.warning("Split '%s' not found under %s, skipping", split, processed_dir)
            continue
        report = validate_split(
            images_dir=split_dir / "images",
            labels_dir=split_dir / "labels",
            num_classes=num_classes,
            split_name=split,
        )
        logger.info(report.summary())
        reports[split] = report
    return reports
