import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ----------------------------------------------
# 🧠 App Configuration
# ----------------------------------------------
st.set_page_config(page_title="YOLO Object Detection", layout="wide")
st.title("🧠 YOLO Object Detection App")
st.markdown("Upload an image, video, or use webcam for real-time object detection using YOLO.")

# ----------------------------------------------
# ⚙️ Sidebar Configuration
# ----------------------------------------------
st.sidebar.header("⚙️ Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

# ----------------------------------------------
# 🧩 Load Model (single model for all modes)
# ----------------------------------------------
@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # 🔹 Replace with your trained YOLO model
    return model

model = load_model()

# -----------------------
# Helper: safe VideoWriter factory
# -----------------------
def create_video_writer(path, fps, width, height):
    """Try common codecs and return a cv2.VideoWriter or raise."""
    # Try H.264 (avc1) first (best browser compatibility)
    fourccs = ["avc1", "mp4v", "X264", "H264"]
    for code in fourccs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*code)
            writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
            if writer.isOpened():
                return writer, code
            else:
                writer.release()
        except Exception:
            continue
    raise RuntimeError("No suitable video codec available for VideoWriter.")

# ----------------------------------------------
# 🎛️ Mode Selection
# ----------------------------------------------
st.sidebar.subheader("📷 Select Input Source")
mode = st.sidebar.radio("Choose Input Type", ["Image", "Video", "Webcam"])

# ----------------------------------------------
# 📸 Image Mode
# ----------------------------------------------
if mode == "Image":
    uploaded_image = st.file_uploader("📂 Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        img_np = np.array(image)

        # Inference
        results = model(img_np, conf=confidence)
        annotated = results[0].plot()

        st.image(annotated, caption="🔍 Detection Result", use_column_width=True)

# ----------------------------------------------
# 🎥 Video Mode
# ----------------------------------------------
elif mode == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    if uploaded_video:
        st.info("Saving uploaded file...")
        input_path = "input_video.mp4"
        output_path = "output_video.mp4"

        # Save uploaded file to stable filename
        try:
            with open(input_path, "wb") as f:
                f.write(uploaded_video.read())
        except Exception as e:
            st.error("Failed to save uploaded file.")
            st.exception(e)
            st.stop()

        # Process
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            st.error("Unable to open uploaded video.")
            st.stop()

        # Gather properties with safe fallbacks
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        st.write(f"Video properties — fps: {fps}, resolution: {width}x{height}, frames: {total_frames}")

        try:
            out, used_fourcc = create_video_writer(output_path, fps, width, height)
            st.write(f"Using codec: {used_fourcc}")
        except Exception as e:
            st.error("Failed to create VideoWriter (codec issue).")
            st.exception(e)
            cap.release()
            st.stop()

        progress = st.progress(0)
        frame_i = 0
        try:
            # Reduce OpenCV threads (cloud CPU stability)
            cv2.setNumThreads(1)
        except Exception:
            pass

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Optionally resize large frames to speed up inference
                # frame = cv2.resize(frame, (640, 480))

                # Inference (model accepts BGR numpy)
                results = model(frame, conf=confidence)
                annotated = results[0].plot()  # BGR

                # Write annotated frame to output file
                out.write(annotated)

                frame_i += 1
                if total_frames:
                    progress.progress(min(frame_i / total_frames, 1.0))
        except Exception as e:
            st.error("Error during processing loop.")
            st.exception(e)
        finally:
            # Always release handles
            cap.release()
            out.release()

        # Confirm file exists & size
        if not os.path.exists(output_path):
            st.error("Processed output file not found.")
            st.stop()

        size_kb = os.path.getsize(output_path) / 1024
        st.success(f"Processing finished — output file size: {size_kb:.1f} KB")

        # Serve video via bytes (more reliable on some hosting)
        try:
            with open(output_path, "rb") as f:
                video_bytes = f.read()
            st.video(video_bytes)
            st.download_button("Download processed video", data=video_bytes, file_name="detections.mp4")
        except Exception as e:
            st.error("Failed to read/play the output file.")
            st.exception(e)
# ----------------------------------------------
# 🧍 Webcam Mode (works online)
# ----------------------------------------------
elif mode == "Webcam":
    st.markdown("🎥 **Webcam mode active — works online via browser.**")

    class YOLOVideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = model(img, conf=confidence)
            annotated_frame = results[0].plot()
            return annotated_frame

    webrtc_streamer(
        key="yolo-webcam",
        video_transformer_factory=YOLOVideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
    )
