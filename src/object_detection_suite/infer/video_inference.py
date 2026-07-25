"""Runs detection over every frame of a video and writes an annotated copy."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2

from object_detection_suite.infer.predictor import Predictor
from object_detection_suite.utils.common import create_directories

logger = logging.getLogger(__name__)


def run_video_inference(
    predictor: Predictor,
    input_video_path: Path | str,
    output_video_path: Path | str,
    frame_stride: int = 1,
    max_frames: int | None = None,
) -> dict:
    """`frame_stride` > 1 skips frames (re-using the last annotation) to
    speed up processing of long videos; detection still runs every
    `frame_stride`-th frame."""
    input_video_path = Path(input_video_path)
    output_video_path = Path(output_video_path)
    create_directories([output_video_path.parent])

    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    frame_idx = 0
    processed_frames = 0
    last_annotated = None
    start_time = time.time()

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break

            if frame_idx % frame_stride == 0:
                image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                predictions = predictor.predict_array(image_rgb)
                from object_detection_suite.visualize.annotate import draw_detections

                last_annotated = draw_detections(
                    frame_bgr,
                    predictions["boxes"].cpu().numpy(),
                    predictions["labels"].cpu().numpy(),
                    predictions["scores"].cpu().numpy(),
                    class_names=predictor.class_names,
                )
                processed_frames += 1
            annotated_frame = last_annotated if last_annotated is not None else frame_bgr
            writer.write(annotated_frame)
            frame_idx += 1

            if frame_idx % 50 == 0:
                logger.info("Processed %d/%d frames", frame_idx, total_frames)
    finally:
        cap.release()
        writer.release()

    elapsed = time.time() - start_time
    logger.info(
        "Video inference done: %d frames (%d detection passes) in %.1fs -> %s",
        frame_idx, processed_frames, elapsed, output_video_path,
    )
    return {
        "total_frames": frame_idx,
        "processed_frames": processed_frames,
        "elapsed_sec": elapsed,
        "output_path": str(output_video_path),
    }
