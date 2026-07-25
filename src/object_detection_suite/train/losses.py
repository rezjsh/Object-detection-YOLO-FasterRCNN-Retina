"""Loss function for the custom YOLO-style single-stage detector.

This is a deliberately simplified YOLOv2-style loss (grid + anchor boxes,
per-cell objectness/box/class targets) rather than a full reimplementation
of a specific YOLO version. It is self-contained (no ultralytics dependency)
and keeps the "swap models without touching training code" contract, since
`YoloStyleDetector.forward` calls this internally and returns the same
`{"loss": ...}` dict that the torchvision-backed models produce natively.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _wh_iou(wh1: torch.Tensor, wh2: torch.Tensor) -> torch.Tensor:
    """IoU between two sets of (w, h) pairs, ignoring position (both boxes
    centered at the origin). wh1: [N, 2], wh2: [M, 2] -> [N, M]."""
    wh1 = wh1[:, None, :]  # [N, 1, 2]
    wh2 = wh2[None, :, :]  # [1, M, 2]
    inter = torch.min(wh1, wh2).prod(dim=2)
    union = wh1.prod(dim=2) + wh2.prod(dim=2) - inter
    return inter / union.clamp(min=1e-9)


class YoloLoss(nn.Module):
    def __init__(
        self,
        num_fg_classes: int,
        anchors: list[tuple[float, float]],
        grid_size: int,
        stride: int,
        lambda_coord: float = 5.0,
        lambda_obj: float = 1.0,
        lambda_noobj: float = 0.5,
        lambda_cls: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_fg_classes = num_fg_classes
        self.register_buffer("anchors", torch.tensor(anchors, dtype=torch.float32))  # [A, 2] in grid-cell units
        self.num_anchors = len(anchors)
        self.grid_size = grid_size
        self.stride = stride
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_cls = lambda_cls

    def build_targets(self, targets: list[dict], device: torch.device) -> dict[str, torch.Tensor]:
        B, S, A, C = len(targets), self.grid_size, self.num_anchors, self.num_fg_classes

        obj_mask = torch.zeros(B, S, S, A, dtype=torch.bool, device=device)
        noobj_mask = torch.ones(B, S, S, A, dtype=torch.bool, device=device)
        tx = torch.zeros(B, S, S, A, device=device)
        ty = torch.zeros(B, S, S, A, device=device)
        tw = torch.zeros(B, S, S, A, device=device)
        th = torch.zeros(B, S, S, A, device=device)
        tcls = torch.zeros(B, S, S, A, dtype=torch.long, device=device)

        for b, tgt in enumerate(targets):
            boxes, labels = tgt["boxes"], tgt["labels"]
            if boxes.numel() == 0:
                continue
            for box, label in zip(boxes, labels):
                if label.item() == 0:
                    continue  # background/padding, shouldn't normally appear here
                x1, y1, x2, y2 = box.tolist()
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    continue

                grid_x, grid_y = cx / self.stride, cy / self.stride
                i, j = int(grid_x), int(grid_y)
                if not (0 <= i < S and 0 <= j < S):
                    continue

                gt_wh_cells = torch.tensor([[w / self.stride, h / self.stride]], device=device)
                ious = _wh_iou(gt_wh_cells, self.anchors)[0]  # [A]
                anchor_idx = int(torch.argmax(ious).item())

                obj_mask[b, j, i, anchor_idx] = True
                noobj_mask[b, j, i, anchor_idx] = False
                tx[b, j, i, anchor_idx] = grid_x - i
                ty[b, j, i, anchor_idx] = grid_y - j
                aw, ah = self.anchors[anchor_idx].tolist()
                tw[b, j, i, anchor_idx] = math.log(max(w / self.stride, 1e-9) / aw)
                th[b, j, i, anchor_idx] = math.log(max(h / self.stride, 1e-9) / ah)
                tcls[b, j, i, anchor_idx] = label.item() - 1  # back to 0-indexed fg class

        return {
            "obj_mask": obj_mask,
            "noobj_mask": noobj_mask,
            "tx": tx,
            "ty": ty,
            "tw": tw,
            "th": th,
            "tcls": tcls,
        }

    def forward(self, preds: torch.Tensor, targets: list[dict]) -> dict[str, torch.Tensor]:
        """preds: raw [B, S, S, A, 5 + C] (not yet activated)."""
        device = preds.device
        t = self.build_targets(targets, device)
        obj_mask, noobj_mask = t["obj_mask"], t["noobj_mask"]

        pred_tx = preds[..., 0]
        pred_ty = preds[..., 1]
        pred_tw = preds[..., 2]
        pred_th = preds[..., 3]
        pred_obj = preds[..., 4]
        pred_cls = preds[..., 5:]

        n_pos = obj_mask.sum().clamp(min=1)

        loss_box = (
            F.mse_loss(torch.sigmoid(pred_tx)[obj_mask], t["tx"][obj_mask], reduction="sum")
            + F.mse_loss(torch.sigmoid(pred_ty)[obj_mask], t["ty"][obj_mask], reduction="sum")
            + F.mse_loss(pred_tw[obj_mask], t["tw"][obj_mask], reduction="sum")
            + F.mse_loss(pred_th[obj_mask], t["th"][obj_mask], reduction="sum")
        ) / n_pos

        loss_obj = F.binary_cross_entropy_with_logits(
            pred_obj[obj_mask], torch.ones_like(pred_obj[obj_mask])
        ) if obj_mask.any() else torch.tensor(0.0, device=device)

        loss_noobj = F.binary_cross_entropy_with_logits(
            pred_obj[noobj_mask], torch.zeros_like(pred_obj[noobj_mask])
        ) if noobj_mask.any() else torch.tensor(0.0, device=device)

        if obj_mask.any():
            loss_cls = F.cross_entropy(pred_cls[obj_mask], t["tcls"][obj_mask])
        else:
            loss_cls = torch.tensor(0.0, device=device)

        total = (
            self.lambda_coord * loss_box
            + self.lambda_obj * loss_obj
            + self.lambda_noobj * loss_noobj
            + self.lambda_cls * loss_cls
        )

        return {
            "loss": total,
            "loss_box": loss_box.detach(),
            "loss_obj": loss_obj.detach(),
            "loss_noobj": loss_noobj.detach(),
            "loss_cls": loss_cls.detach(),
        }
