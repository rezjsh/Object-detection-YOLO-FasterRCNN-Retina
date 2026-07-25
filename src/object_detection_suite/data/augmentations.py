"""Albumentations pipelines shared by every model family.

We standardize on Albumentations' native 'yolo' bbox format
(normalized [x_center, y_center, w, h]) inside the pipeline itself, since
that is exactly how the labels are stored on disk. The dataset loader
converts the augmented boxes to absolute-pixel xyxy afterwards, which is
what both the custom YOLO-style head and the torchvision detectors expect.
"""
from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_size: int = 640) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=0,
                fill=(114, 114, 114),
            ),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=25, val_shift_limit=15, p=0.3),
            A.Affine(
                scale=(0.85, 1.15),
                translate_percent=(-0.05, 0.05),
                rotate=(-5, 5),
                p=0.4,
            ),
            A.GaussNoise(p=0.1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.2,
            clip=True,
        ),
    )


def get_val_transforms(img_size: int = 640) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=0,
                fill=(114, 114, 114),
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.0,
            clip=True,
        ),
    )


def get_inference_transforms(img_size: int = 640) -> A.Compose:
    """Same geometry as val transforms but with no bbox_params, since
    inference-time images have no ground-truth boxes to carry through."""
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=0,
                fill=(114, 114, 114),
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
