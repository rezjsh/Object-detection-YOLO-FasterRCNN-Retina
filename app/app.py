"""Streamlit demo for the object detection suite.

Run with:
    uv run streamlit run app/app.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from demo_utils import available_models_with_checkpoints, load_predictor  # noqa: E402

st.set_page_config(page_title="Object Detection Suite", layout="wide")
st.title("🔎 Object Detection Suite — Live Demo")
st.caption("Compare YOLO-style, Faster R-CNN, and RetinaNet detectors on your own images or video.")

available_models = available_models_with_checkpoints()
if not available_models:
    st.warning(
        "No trained checkpoints found yet. Train at least one model first, e.g.:\n\n"
        "`uv run python main.py train --model yolo_style`"
    )
    st.stop()

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Model", available_models)
    score_threshold = st.slider("Score threshold", 0.0, 1.0, 0.35, 0.05)
    nms_iou_threshold = st.slider("NMS IoU threshold", 0.0, 1.0, 0.45, 0.05)

predictor = load_predictor(model_name, score_threshold, nms_iou_threshold)

tab_image, tab_video = st.tabs(["📷 Image", "🎥 Video"])

with tab_image:
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded_image is not None:
        file_bytes = np.frombuffer(uploaded_image.read(), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        predictions = predictor.predict_array(image_rgb)
        from object_detection_suite.visualize.annotate import draw_detections

        annotated = draw_detections(
            image_bgr,
            predictions["boxes"].cpu().numpy(),
            predictions["labels"].cpu().numpy(),
            predictions["scores"].cpu().numpy(),
            class_names=predictor.class_names,
        )
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        col1.image(image_rgb, caption="Original", use_container_width=True)
        col2.image(annotated_rgb, caption=f"Detections ({model_name})", use_container_width=True)
        st.write(f"**{len(predictions['boxes'])} detection(s)** at score >= {score_threshold}")

with tab_video:
    uploaded_video = st.file_uploader("Upload a short video", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
            tmp_in.write(uploaded_video.read())
            input_path = tmp_in.name
        output_path = str(Path(tempfile.gettempdir()) / "annotated_output.mp4")

        if st.button("Run detection on video"):
            from object_detection_suite.infer.video_inference import run_video_inference

            with st.spinner("Running inference on video frames..."):
                result = run_video_inference(predictor, input_path, output_path, frame_stride=2)
            st.success(f"Processed {result['total_frames']} frames in {result['elapsed_sec']:.1f}s")
            st.video(output_path)
            
            # --- NEW CODE: Add a download button ---
            with open(output_path, "rb") as video_file:
                st.download_button(
                    label="💾 Download Annotated Video",
                    data=video_file,
                    file_name="my_annotated_video.mp4",
                    mime="video/mp4"
                )
