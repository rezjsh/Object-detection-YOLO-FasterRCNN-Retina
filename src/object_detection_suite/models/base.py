"""Common interface every detector in this project implements.

Standardizing on this interface is what lets `Trainer`, `Evaluator` and the
inference/predictor code stay completely model-agnostic:

    - In train mode, `forward(images, targets)` returns a dict containing at
      least a scalar `"loss"` tensor (plus optional named loss components for
      logging).
    - In eval mode, `forward(images)` (targets optional/ignored) returns a
      list of per-image prediction dicts:
          {"boxes": FloatTensor[N,4] xyxy abs pixels,
           "scores": FloatTensor[N],
           "labels": Int64Tensor[N]}

This mirrors how torchvision's detection models already behave, so the
torchvision-backed wrappers (Faster R-CNN, RetinaNet) are thin, and the
custom YOLO-style model simply implements the same contract by hand.
"""
from __future__ import annotations

import abc

import torch
import torch.nn as nn


class BaseDetector(nn.Module, abc.ABC):
    name: str = "base"

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        # +1 for the implicit background class (label 0), matching the
        # dataset loader's 1-indexed label convention.
        self.num_classes = num_classes + 1

    @abc.abstractmethod
    def forward(
        self,
        images: torch.Tensor,
        targets: list[dict] | None = None,
    ) -> dict | list[dict]:
        """See module docstring for the train/eval mode contract."""
        raise NotImplementedError

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> list[dict]:
        was_training = self.training
        self.eval()
        try:
            outputs = self.forward(images)
        finally:
            self.train(was_training)
        return outputs

    def freeze_backbone(self) -> None:
        """Override in subclasses that have a distinct backbone module."""
        pass

    def unfreeze_backbone(self) -> None:
        """Override in subclasses that have a distinct backbone module."""
        pass

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
