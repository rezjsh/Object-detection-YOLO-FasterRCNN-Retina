"""RetinaNet (single-stage, focal loss) detector wrapped to satisfy `BaseDetector`.

Unlike Faster R-CNN, torchvision's RetinaNet has no explicit background
class: it does one-vs-all sigmoid classification over exactly `num_classes`
foreground classes, so labels passed to the internal model must be
0-indexed. The dataset/loader convention elsewhere in this project is
1-indexed (0 reserved for background), so this wrapper shifts labels down by
one on the way in and back up by one on the way out, keeping every other
module oblivious to the difference.
"""
from __future__ import annotations

import torch
import torchvision
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

from object_detection_suite.models.base import BaseDetector


class RetinaNetModel(BaseDetector):
    name = "retinanet"

    def __init__(self, num_classes: int, pretrained_backbone: bool = True) -> None:
        super().__init__(num_classes=num_classes)  # self.num_classes = num_classes + 1 (bg, unused here)
        num_fg_classes = num_classes  # RetinaNet has no background class

        # See the matching note in faster_rcnn_model.py: `weights_backbone`
        # must be set explicitly or it defaults to pretrained ImageNet
        # weights regardless of `weights`.
        if pretrained_backbone:
            weights = torchvision.models.detection.RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
            weights_backbone = torchvision.models.ResNet50_Weights.DEFAULT
        else:
            weights = None
            weights_backbone = None
        self.model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
            weights=weights, weights_backbone=weights_backbone
        )

        num_anchors = self.model.head.classification_head.num_anchors
        in_channels = self.model.backbone.out_channels
        self.model.head.classification_head = RetinaNetClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_fg_classes,
            norm_layer=torch.nn.BatchNorm2d,
        )

    def freeze_backbone(self) -> None:
        for p in self.model.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.model.backbone.parameters():
            p.requires_grad = True

    @staticmethod
    def _shift_targets_down(targets: list[dict]) -> list[dict]:
        return [
            {**t, "labels": (t["labels"] - 1).clamp(min=0)}
            for t in targets
        ]

    @staticmethod
    def _shift_predictions_up(predictions: list[dict]) -> list[dict]:
        for p in predictions:
            p["labels"] = p["labels"] + 1
        return predictions

    def forward(self, images: torch.Tensor, targets: list[dict] | None = None):
        image_list = list(images.unbind(0))

        if self.training:
            if targets is None:
                raise ValueError("targets must be provided in training mode")
            shifted = self._shift_targets_down(targets)
            loss_dict = self.model(image_list, shifted)
            total = sum(loss_dict.values())
            out = {"loss": total}
            out.update({k: v.detach() for k, v in loss_dict.items()})
            return out

        predictions = self.model(image_list)
        predictions = self._shift_predictions_up(
            [{"boxes": p["boxes"], "scores": p["scores"], "labels": p["labels"]} for p in predictions]
        )
        return predictions
