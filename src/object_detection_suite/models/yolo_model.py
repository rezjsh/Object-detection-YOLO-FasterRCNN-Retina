"""A compact, from-scratch YOLO-style single-stage detector.

Architecture: ImageNet-pretrained ResNet-18 truncated after layer4 (stride
32) as the backbone, followed by a small conv head that predicts, per grid
cell and per anchor, (tx, ty, tw, th, objectness, class_logits...).

This intentionally mirrors classic YOLOv2-style single-scale detection
rather than reproducing a specific modern YOLO release (which would pull in
the `ultralytics` package and its own training loop, defeating the point of
a shared, swappable training pipeline). It still satisfies the "YOLO-style
detector at 640x640" requirement: grid + anchors + objectness + NMS.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision
from torchvision.ops import nms

from object_detection_suite.models.base import BaseDetector
from object_detection_suite.train.losses import YoloLoss

# Anchor boxes in grid-cell units (width, height), chosen to roughly span
# small (helmet/head) to large (person) objects at a 32px stride / 640 input.
DEFAULT_ANCHORS = [(1.0, 1.0), (2.5, 3.5), (5.0, 7.0)]


class YoloStyleDetector(BaseDetector):
    name = "yolo_style"

    def __init__(
        self,
        num_classes: int,
        img_size: int = 640,
        anchors: list[tuple[float, float]] | None = None,
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__(num_classes=num_classes)
        self.img_size = img_size
        self.stride = 32
        self.grid_size = img_size // self.stride
        self.anchors = anchors or DEFAULT_ANCHORS
        self.num_anchors = len(self.anchors)
        num_fg_classes = self.num_classes - 1  # exclude background

        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        resnet = torchvision.models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # drop avgpool + fc -> [B, 512, S, S]
        backbone_out_channels = 512

        head_out_channels = self.num_anchors * (5 + num_fg_classes)
        self.head = nn.Sequential(
            nn.Conv2d(backbone_out_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, head_out_channels, kernel_size=1),
        )

        self.loss_fn = YoloLoss(
            num_fg_classes=num_fg_classes,
            anchors=self.anchors,
            grid_size=self.grid_size,
            stride=self.stride,
        )

        self._score_threshold = 0.35
        self._nms_iou_threshold = 0.45
        self._max_detections = 300

    def set_inference_thresholds(self, score_threshold: float, nms_iou_threshold: float) -> None:
        self._score_threshold = score_threshold
        self._nms_iou_threshold = nms_iou_threshold

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True

    def _raw_predictions(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(images)  # [B, 512, S, S]
        out = self.head(feats)  # [B, A*(5+C), S, S]
        B, _, S, _ = out.shape
        num_fg_classes = self.num_classes - 1
        out = out.view(B, self.num_anchors, 5 + num_fg_classes, S, S)
        out = out.permute(0, 3, 4, 1, 2).contiguous()  # [B, S, S, A, 5+C]
        return out

    def forward(self, images: torch.Tensor, targets: list[dict] | None = None):
        preds = self._raw_predictions(images)
        if self.training:
            if targets is None:
                raise ValueError("targets must be provided in training mode")
            return self.loss_fn(preds, targets)
        return self._decode(preds)

    @torch.no_grad()
    def _decode(self, preds: torch.Tensor) -> list[dict]:
        device = preds.device
        B, S, _, A, _ = preds.shape
        anchors = torch.tensor(self.anchors, device=device)  # [A, 2]

        grid_y, grid_x = torch.meshgrid(
            torch.arange(S, device=device), torch.arange(S, device=device), indexing="ij"
        )
        grid_x = grid_x.unsqueeze(-1).expand(S, S, A).float()
        grid_y = grid_y.unsqueeze(-1).expand(S, S, A).float()
        anchor_w = anchors[:, 0].view(1, 1, A)
        anchor_h = anchors[:, 1].view(1, 1, A)

        results = []
        for b in range(B):
            p = preds[b]  # [S, S, A, 5+C]
            cx = (torch.sigmoid(p[..., 0]) + grid_x) * self.stride
            cy = (torch.sigmoid(p[..., 1]) + grid_y) * self.stride
            w = torch.exp(p[..., 2].clamp(max=10)) * anchor_w * self.stride
            h = torch.exp(p[..., 3].clamp(max=10)) * anchor_h * self.stride
            obj = torch.sigmoid(p[..., 4])
            cls_probs = torch.softmax(p[..., 5:], dim=-1)
            cls_scores, cls_idx = cls_probs.max(dim=-1)
            scores = obj * cls_scores

            x1 = (cx - w / 2).reshape(-1)
            y1 = (cy - h / 2).reshape(-1)
            x2 = (cx + w / 2).reshape(-1)
            y2 = (cy + h / 2).reshape(-1)
            boxes = torch.stack([x1, y1, x2, y2], dim=-1)
            scores = scores.reshape(-1)
            labels = (cls_idx.reshape(-1) + 1)  # back to 1-indexed to match dataset convention

            keep = scores > self._score_threshold
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            if boxes.numel() > 0:
                keep_idx = nms(boxes, scores, self._nms_iou_threshold)
                keep_idx = keep_idx[: self._max_detections]
                boxes, scores, labels = boxes[keep_idx], scores[keep_idx], labels[keep_idx]

            results.append({"boxes": boxes, "scores": scores, "labels": labels})
        return results
