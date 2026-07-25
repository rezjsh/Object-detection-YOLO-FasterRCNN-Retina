"""Inference on single images or whole folders.

Loads images with OpenCV, runs them through whichever `BaseDetector` is
handed in, draws the results, and writes annotated copies to disk. Model-
agnostic by construction since it only relies on the `BaseDetector.predict`
contract.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import torch

from object_detection_suite.constants.constants import IMAGE_EXTENSIONS
from object_detection_suite.data.augmentations import get_inference_transforms
from object_detection_suite.models.base import BaseDetector
from object_detection_suite.utils.common import create_directories
from object_detection_suite.visualize.annotate import draw_detections

logger = logging.getLogger(__name__)


class Predictor:
    def __init__(
        self,
        model: BaseDetector,
        class_names: list[str],
        device: str,
        img_size: int = 640,
        score_threshold: float = 0.35,
        nms_iou_threshold: float = 0.45,
    ) -> None:
        self.model = model.to(device).eval()
        self.class_names = class_names
        self.device = device
        self.img_size = img_size
        self.transforms = get_inference_transforms(img_size)

        if hasattr(self.model, "set_inference_thresholds"):
            self.model.set_inference_thresholds(score_threshold, nms_iou_threshold)
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold

    def _preprocess(self, image_rgb) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
        original_h, original_w = image_rgb.shape[:2]
        transformed = self.transforms(image=image_rgb)
        tensor = transformed["image"].unsqueeze(0).to(self.device)
        padded_h, padded_w = tensor.shape[2], tensor.shape[3]
        return tensor, (original_h, original_w), (padded_h, padded_w)

    @staticmethod
    def _rescale_boxes(boxes, original_hw, padded_hw):
        """Undo LongestMaxSize + PadIfNeeded so boxes map back to the
        original image resolution."""
        if boxes.numel() == 0:
            return boxes
        orig_h, orig_w = original_hw
        pad_h, pad_w = padded_hw
        scale = min(pad_h / orig_h, pad_w / orig_w)
        new_h, new_w = orig_h * scale, orig_w * scale
        pad_top = (pad_h - new_h) / 2
        pad_left = (pad_w - new_w) / 2

        boxes = boxes.clone()
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top) / scale
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, orig_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, orig_h)
        return boxes

    @torch.no_grad()
    def predict_array(self, image_rgb) -> dict:
        """Runs inference on an in-memory RGB numpy array. Returns a dict
        with boxes rescaled to the original image resolution."""
        tensor, original_hw, padded_hw = self._preprocess(image_rgb)
        predictions = self.model.predict(tensor)[0]
        predictions["boxes"] = self._rescale_boxes(predictions["boxes"], original_hw, padded_hw)
        return predictions

    def predict_image(self, image_path: Path | str, output_path: Path | str | None = None) -> dict:
        image_path = Path(image_path)
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        predictions = self.predict_array(image_rgb)

        annotated = draw_detections(
            image_bgr,
            predictions["boxes"].cpu().numpy(),
            predictions["labels"].cpu().numpy(),
            predictions["scores"].cpu().numpy(),
            class_names=self.class_names,
        )
        if output_path is not None:
            output_path = Path(output_path)
            create_directories([output_path.parent])
            cv2.imwrite(str(output_path), annotated)
            logger.info("Saved annotated image to %s", output_path)

        return {"predictions": predictions, "annotated_image": annotated}

    def predict_folder(self, input_dir: Path | str, output_dir: Path | str) -> list[dict]:
        input_dir, output_dir = Path(input_dir), Path(output_dir)
        create_directories([output_dir])
        image_paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not image_paths:
            logger.warning("No images found in %s", input_dir)

        results = []
        for image_path in image_paths:
            out_path = output_dir / image_path.name
            result = self.predict_image(image_path, out_path)
            results.append({"image": str(image_path), "num_detections": len(result["predictions"]["boxes"])})
            logger.info("%s -> %d detections", image_path.name, results[-1]["num_detections"])
        return results
