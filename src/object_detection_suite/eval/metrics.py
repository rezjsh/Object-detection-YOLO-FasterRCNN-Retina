"""Self-contained detection metrics: mAP@0.5, mAP@0.5:0.95, precision,
recall, and mean IoU of matched detections.

Implemented from scratch (no pycocotools) to keep the dependency footprint
light. Algorithm is the standard greedy-matching + all-point interpolated
average precision used by PASCAL VOC / COCO.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torchvision.ops import box_iou


def _average_precision_from_curve(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """All-point interpolated AP (COCO/VOC-2010 style)."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap


class MeanAveragePrecisionCalculator:
    """Accumulates predictions/ground-truths across batches, then computes
    the full metric suite in one pass over the accumulated data."""

    def __init__(self, num_classes: int, iou_thresholds: list[float]):
        self.num_classes = num_classes  # includes background at index 0
        self.iou_thresholds = sorted(iou_thresholds)
        self._preds: list[dict[str, torch.Tensor]] = []
        self._gts: list[dict[str, torch.Tensor]] = []

    def reset(self) -> None:
        self._preds.clear()
        self._gts.clear()

    def update(self, preds: list[dict], targets: list[dict]) -> None:
        for p, t in zip(preds, targets):
            self._preds.append({k: v.detach().cpu() for k, v in p.items()})
            self._gts.append({k: v.detach().cpu() for k, v in t.items()})

    def compute(self) -> dict:
        fg_classes = list(range(1, self.num_classes))
        ap_table: dict[int, dict[float, float]] = {c: {} for c in fg_classes}
        all_ious: list[float] = []

        for cls_id in fg_classes:
            # 1. Pre-compute IoU matrices and sort detections ONCE per class
            detections = []  # Stores: (score, img_idx, pred_idx_in_image)
            iou_matrices: dict[int, torch.Tensor] = {}
            num_gt_total = 0

            for img_idx, (pred, gt) in enumerate(zip(self._preds, self._gts)):
                gt_mask = gt["labels"] == cls_id
                gt_boxes = gt["boxes"][gt_mask]
                num_gt_total += gt_boxes.shape[0]

                pred_mask = pred["labels"] == cls_id
                pred_boxes = pred["boxes"][pred_mask]
                pred_scores = pred["scores"][pred_mask]
                
                if pred_boxes.shape[0] > 0 and gt_boxes.shape[0] > 0:
                    iou_matrices[img_idx] = box_iou(pred_boxes, gt_boxes)
                
                for pred_idx, score in enumerate(pred_scores):
                    detections.append((float(score), img_idx, pred_idx))

            if num_gt_total == 0:
                for iou_thr in self.iou_thresholds:
                    ap_table[cls_id][iou_thr] = 0.0
                continue

            # Global sort by confidence (done once per class, not per threshold)
            detections.sort(key=lambda d: d[0], reverse=True)

            # 2. Evaluate across each IoU threshold using the pre-computed structures
            for iou_thr in self.iou_thresholds:
                matched_gt: dict[int, set[int]] = defaultdict(set)
                tp = np.zeros(len(detections))
                fp = np.zeros(len(detections))
                matched_ious_at_thr: list[float] = []

                for i, (_, img_idx, pred_idx) in enumerate(detections):
                    if img_idx not in iou_matrices:
                        fp[i] = 1
                        continue
                        
                    ious = iou_matrices[img_idx][pred_idx]
                    best_iou, best_j = torch.max(ious, dim=0)
                    best_iou, best_j = float(best_iou), int(best_j)
                    
                    if best_iou >= iou_thr and best_j not in matched_gt[img_idx]:
                        tp[i] = 1
                        matched_gt[img_idx].add(best_j)
                        matched_ious_at_thr.append(best_iou)
                    else:
                        fp[i] = 1

                tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
                recalls = tp_cum / max(num_gt_total, 1)
                precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
                
                ap = _average_precision_from_curve(recalls, precisions) if recalls.size else 0.0
                ap_table[cls_id][iou_thr] = ap
                
                if abs(iou_thr - 0.5) < 1e-6:
                    all_ious.extend(matched_ious_at_thr)

        iou_50 = min(self.iou_thresholds, key=lambda x: abs(x - 0.5))
        map_50 = float(np.mean([ap_table[c][iou_50] for c in fg_classes])) if fg_classes else 0.0
        map_50_95 = float(
            np.mean([np.mean(list(ap_table[c].values())) for c in fg_classes])
        ) if fg_classes else 0.0

        precision_50, recall_50 = self._precision_recall_at(iou_50)
        mean_iou = float(np.mean(all_ious)) if all_ious else 0.0

        return {
            "mAP_50": map_50,
            "mAP_50_95": map_50_95,
            "precision": precision_50,
            "recall": recall_50,
            "mean_iou": mean_iou,
            "per_class_ap50": {c: ap_table[c][iou_50] for c in fg_classes},
        }

    def _precision_recall_at(self, iou_thr: float, score_threshold: float = 0.5) -> tuple[float, float]:
        """Macro-averaged precision/recall across classes at a fixed
        confidence threshold — a single operating-point summary, distinct
        from the full-curve integral used for AP."""
        precisions, recalls = [], []
        for cls_id in range(1, self.num_classes):
            tp = fp = fn = 0
            for pred, gt in zip(self._preds, self._gts):
                gt_mask = gt["labels"] == cls_id
                gt_boxes = gt["boxes"][gt_mask]
                
                pred_mask = (pred["labels"] == cls_id) & (pred["scores"] >= score_threshold)
                pred_scores = pred["scores"][pred_mask]
                
                sort_idx = torch.argsort(pred_scores, descending=True)
                pred_boxes = pred["boxes"][pred_mask][sort_idx]

                matched = set()
                if pred_boxes.shape[0] and gt_boxes.shape[0]:
                    ious = box_iou(pred_boxes, gt_boxes)
                    for i in range(pred_boxes.shape[0]):
                        best_iou, best_j = torch.max(ious[i], dim=0)
                        best_j = int(best_j)
                        if float(best_iou) >= iou_thr and best_j not in matched:
                            tp += 1
                            matched.add(best_j)
                        else:
                            fp += 1
                else:
                    fp += pred_boxes.shape[0]
                fn += gt_boxes.shape[0] - len(matched)

            if tp + fp > 0:
                precisions.append(tp / (tp + fp))
            if tp + fn > 0:
                recalls.append(tp / (tp + fn))

        precision = float(np.mean(precisions)) if precisions else 0.0
        recall = float(np.mean(recalls)) if recalls else 0.0
        return precision, recall