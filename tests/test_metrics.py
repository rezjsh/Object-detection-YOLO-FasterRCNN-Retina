"""Tests for `eval/metrics.py` against hand-computed expected values on a
tiny, fully-deterministic synthetic example (no randomness), so the
algorithm's correctness is pinned down exactly."""
from __future__ import annotations

import numpy as np
import torch

from object_detection_suite.eval.metrics import MeanAveragePrecisionCalculator, _average_precision_from_curve


def test_average_precision_perfect_detector():
    # Perfect precision at every recall level -> AP == 1.0
    recalls = np.array([0.5, 1.0])
    precisions = np.array([1.0, 1.0])
    ap = _average_precision_from_curve(recalls, precisions)
    assert abs(ap - 1.0) < 1e-6


def test_average_precision_known_value():
    # One TP then one FP: recall=[1,1], precision=[1, 0.5] -> AP should be 1.0
    # (the FP after full recall is already achieved doesn't hurt AP under
    # all-point interpolation, since precision is monotonically maxed from the right).
    recalls = np.array([1.0, 1.0])
    precisions = np.array([1.0, 0.5])
    ap = _average_precision_from_curve(recalls, precisions)
    assert abs(ap - 1.0) < 1e-6


def _one_tp_one_fp_scenario():
    """Image 0: one GT box perfectly matched by one high-confidence prediction.
    Image 1: no GT boxes, one lower-confidence false-positive prediction."""
    calc = MeanAveragePrecisionCalculator(num_classes=2, iou_thresholds=[0.5])  # 1 bg + 1 fg class

    preds = [
        {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "scores": torch.tensor([0.9]), "labels": torch.tensor([1])},
        {"boxes": torch.tensor([[50.0, 50.0, 60.0, 60.0]]), "scores": torch.tensor([0.8]), "labels": torch.tensor([1])},
    ]
    targets = [
        {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([1])},
        {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.int64)},
    ]
    calc.update(preds, targets)
    return calc


def test_map_calculator_perfect_and_false_positive_scenario():
    calc = _one_tp_one_fp_scenario()
    metrics = calc.compute()

    assert abs(metrics["mAP_50"] - 1.0) < 1e-6, "single-class AP should be 1.0: the FP comes after full recall"
    assert abs(metrics["mean_iou"] - 1.0) < 1e-6, "the one matched detection has IoU 1.0 (identical box)"
    # precision/recall are computed at a fixed 0.5 score threshold (different
    # from the full-curve AP integral): both the TP and the FP clear 0.5.
    assert abs(metrics["precision"] - 0.5) < 1e-6
    assert abs(metrics["recall"] - 1.0) < 1e-6


def test_map_calculator_no_detections_gives_zero():
    calc = MeanAveragePrecisionCalculator(num_classes=2, iou_thresholds=[0.5])
    preds = [{"boxes": torch.zeros((0, 4)), "scores": torch.zeros((0,)), "labels": torch.zeros((0,), dtype=torch.int64)}]
    targets = [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([1])}]
    calc.update(preds, targets)
    metrics = calc.compute()
    assert metrics["mAP_50"] == 0.0
    assert metrics["recall"] == 0.0


def test_map_calculator_no_ground_truth_no_predictions():
    calc = MeanAveragePrecisionCalculator(num_classes=2, iou_thresholds=[0.5])
    preds = [{"boxes": torch.zeros((0, 4)), "scores": torch.zeros((0,)), "labels": torch.zeros((0,), dtype=torch.int64)}]
    targets = [{"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.int64)}]
    calc.update(preds, targets)
    metrics = calc.compute()
    # No ground truth at all for the only class -> AP defined as 0.0 by convention here
    assert metrics["mAP_50"] == 0.0
