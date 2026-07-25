"""Faster R-CNN (two-stage) detector, thinly wrapped to satisfy `BaseDetector`.

torchvision's detection models already implement almost exactly the
train/eval contract this project standardizes on (loss dict when
`self.training` and targets are given, prediction list otherwise), so this
wrapper is mostly plumbing: swap in the right number of classes, unbind the
stacked image batch into the list-of-tensors form torchvision expects, and
normalize the loss dict to always expose a `"loss"` key.
"""
from __future__ import annotations

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from object_detection_suite.models.base import BaseDetector


class FasterRCNNModel(BaseDetector):
    name = "faster_rcnn"

    def __init__(self, num_classes: int, pretrained_backbone: bool = True) -> None:
        super().__init__(num_classes=num_classes)  # sets self.num_classes = num_classes + 1 (bg)

        # NOTE: torchvision's `weights_backbone` defaults to ImageNet weights
        # *independently* of `weights`, so it must be set explicitly to None
        # here or a "no pretrained weights" request would silently still
        # download a pretrained ResNet-50 backbone.
        if pretrained_backbone:
            weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            weights_backbone = torchvision.models.ResNet50_Weights.DEFAULT
        else:
            weights = None
            weights_backbone = None
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=weights, weights_backbone=weights_backbone
        )

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

    def freeze_backbone(self) -> None:
        for p in self.model.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.model.backbone.parameters():
            p.requires_grad = True

    def forward(self, images: torch.Tensor, targets: list[dict] | None = None):
        image_list = list(images.unbind(0))

        if self.training:
            if targets is None:
                raise ValueError("targets must be provided in training mode")
            # torchvision requires non-empty boxes; images with zero GT boxes
            # are represented with an empty tensor, which it handles natively.
            loss_dict = self.model(image_list, targets)
            total = sum(loss_dict.values())
            out = {"loss": total}
            out.update({k: v.detach() for k, v in loss_dict.items()})
            return out

        predictions = self.model(image_list)
        return [
            {"boxes": p["boxes"], "scores": p["scores"], "labels": p["labels"]}
            for p in predictions
        ]
