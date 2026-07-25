"""OpenCV-based annotation helpers: draws predicted or ground-truth boxes
with class labels and confidence scores onto an image."""
from __future__ import annotations

import cv2
import numpy as np

_PALETTE = [
    (66, 135, 245), (245, 132, 66), (66, 245, 141), (245, 66, 200),
    (245, 218, 66), (147, 66, 245), (66, 245, 236), (245, 66, 66),
]


def _color_for_class(class_id: int) -> tuple[int, int, int]:
    return _PALETTE[class_id % len(_PALETTE)]


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray | None = None,
    class_names: list[str] | None = None,
    thickness: int = 2,
) -> np.ndarray:
    """`image` is expected in BGR (OpenCV-native) format. `labels` are
    1-indexed (0 reserved for background) to match the dataset convention;
    `class_names` is indexed 0..num_classes-1, so we subtract 1 when looking
    the name up."""
    out = image.copy()
    for i in range(len(boxes)):
        x1, y1, x2, y2 = (int(round(v)) for v in boxes[i])
        cls_id = int(labels[i])
        color = _color_for_class(cls_id)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        name = "unknown"
        if class_names is not None and 0 <= cls_id - 1 < len(class_names):
            name = class_names[cls_id - 1]
        text = name if scores is None else f"{name} {scores[i]:.2f}"

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            out, text, (x1 + 2, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return out
