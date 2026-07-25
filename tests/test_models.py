"""Tests for the model factory and each detector's forward-pass contract:
train mode returns a loss dict, eval mode returns a prediction list with the
expected keys/shapes. Uses `pretrained_backbone=False` everywhere so tests
run fully offline (no ImageNet weight downloads)."""
from __future__ import annotations

import torch

from object_detection_suite.models.faster_rcnn_model import FasterRCNNModel
from object_detection_suite.models.model_factory import ModelFactory
from object_detection_suite.models.retinanet_model import RetinaNetModel
from object_detection_suite.models.yolo_model import YoloStyleDetector

NUM_CLASSES = 3  # foreground classes (helmet, head, person)
IMG_SIZE = 128
BATCH_SIZE = 2

# Faster R-CNN / RetinaNet use ResNet-50 + FPN, which is considerably more
# memory-hungry on CPU than the lightweight custom YOLO-style model even at
# tiny resolutions — use a smaller batch/image size for those two so the
# suite stays fast and CI-friendly without touching the models' actual logic.
TORCHVISION_IMG_SIZE = 96
TORCHVISION_BATCH_SIZE = 1


def _dummy_batch(batch_size: int = BATCH_SIZE, box_scale: float = 1.0):
    images = torch.rand(batch_size, 3, IMG_SIZE, IMG_SIZE)
    all_targets = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 60.0, 80.0], [30.0, 20.0, 90.0, 100.0]]),
            "labels": torch.tensor([1, 2], dtype=torch.int64),
        },
        {
            "boxes": torch.tensor([[5.0, 5.0, 40.0, 40.0]]),
            "labels": torch.tensor([3], dtype=torch.int64),
        },
    ]
    return images, all_targets[:batch_size]


def _small_dummy_batch():
    images = torch.rand(TORCHVISION_BATCH_SIZE, 3, TORCHVISION_IMG_SIZE, TORCHVISION_IMG_SIZE)
    targets = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 60.0, 80.0]]),
            "labels": torch.tensor([1], dtype=torch.int64),
        }
    ][:TORCHVISION_BATCH_SIZE]
    return images, targets


def test_model_factory_lists_expected_models():
    available = ModelFactory.available_models()
    assert set(available) == {"yolo_style", "faster_rcnn", "retinanet"}


def test_model_factory_creates_each_model():
    for name in ModelFactory.available_models():
        model = ModelFactory.create(name, num_classes=NUM_CLASSES, pretrained_backbone=False)
        assert model.name == name
        assert model.count_parameters() > 0


def _assert_loss_dict(output):
    assert "loss" in output
    assert output["loss"].dim() == 0  # scalar
    assert torch.isfinite(output["loss"])


def _assert_predictions(output, batch_size):
    assert isinstance(output, list)
    assert len(output) == batch_size
    for pred in output:
        assert set(pred.keys()) == {"boxes", "scores", "labels"}
        assert pred["boxes"].shape[-1] == 4
        assert pred["boxes"].shape[0] == pred["scores"].shape[0] == pred["labels"].shape[0]


def test_yolo_style_train_and_eval_forward():
    model = YoloStyleDetector(num_classes=NUM_CLASSES, img_size=IMG_SIZE, pretrained_backbone=False)
    images, targets = _dummy_batch()

    model.train()
    out = model(images, targets)
    _assert_loss_dict(out)
    out["loss"].backward()  # verify gradients flow

    model.eval()
    preds = model(images)
    _assert_predictions(preds, BATCH_SIZE)


def test_faster_rcnn_train_and_eval_forward():
    model = FasterRCNNModel(num_classes=NUM_CLASSES, pretrained_backbone=False)
    images, targets = _small_dummy_batch()

    model.train()
    out = model(images, targets)
    _assert_loss_dict(out)

    model.eval()
    preds = model(images)
    _assert_predictions(preds, TORCHVISION_BATCH_SIZE)


def test_retinanet_train_and_eval_forward():
    model = RetinaNetModel(num_classes=NUM_CLASSES, pretrained_backbone=False)
    images, targets = _small_dummy_batch()

    model.train()
    out = model(images, targets)
    _assert_loss_dict(out)

    model.eval()
    preds = model(images)
    _assert_predictions(preds, TORCHVISION_BATCH_SIZE)


def test_freeze_and_unfreeze_backbone_toggles_requires_grad():
    model = FasterRCNNModel(num_classes=NUM_CLASSES, pretrained_backbone=False)
    model.freeze_backbone()
    assert all(not p.requires_grad for p in model.model.backbone.parameters())
    model.unfreeze_backbone()
    assert all(p.requires_grad for p in model.model.backbone.parameters())
