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
st.set_page_config(page_title="YOLO Object Detection + Tracking", layout="wide")
st.title("🧠 YOLO Object Detection & Tracking App")
st.markdown("Upload an image, video, or use webcam for real-time **object detection and tracking** using YOLO.")

# ----------------------------------------------
# ⚙️ Sidebar Configuration
# ----------------------------------------------
st.sidebar.header("⚙️ Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
tracker_choice = st.sidebar.radio("Select Tracker", ["BoT-SORT (Default)", "ByteTrack"])
tracker_yaml = "botsort.yaml" if "BoT" in tracker_choice else "bytetrack.yaml"

# ----------------------------------------------
# 🧩 Load Model
# ----------------------------------------------
@st.cache_resource
def load_model():
    model = YOLO("best.onnx")  # Replace with your custom YOLO model
    return model

model = load_model()

# -----------------------
# Helper: safe VideoWriter factory
# -----------------------
def create_video_writer(path, fps, width, height):
    """Try common codecs and return a cv2.VideoWriter or raise."""
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
        results = model(img_np, conf=confidence)
        annotated = results[0].plot()
        st.image(annotated, caption="🔍 Detection Result", use_column_width=True)

# ----------------------------------------------
# 🎥 Video Mode (Detection + Tracking)
# ----------------------------------------------
elif mode == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    if uploaded_video:
        st.info("Saving uploaded file...")
        input_path = "input_video.mp4"
        output_path = "output_video.mp4"

        # Save uploaded video
        with open(input_path, "wb") as f:
            f.write(uploaded_video.read())

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            st.error("Unable to open video.")
            st.stop()

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        st.write(f"🎞️ Video properties — {width}x{height}, {fps:.1f} FPS, {total_frames} frames")

        try:
            out, used_fourcc = create_video_writer(output_path, fps, width, height)
            st.write(f"Using codec: {used_fourcc}")
        except Exception as e:
            st.error("Failed to create VideoWriter.")
            st.exception(e)
            st.stop()

        progress = st.progress(0)
        frame_i = 0

        try:
            cv2.setNumThreads(1)
        except Exception:
            pass

        try:
            # Tracking loop
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Use model.track instead of model()
                results = model.track(frame, conf=confidence, persist=True, tracker=tracker_yaml)
                annotated = results[0].plot()

                out.write(annotated)
                frame_i += 1
                if total_frames:
                    progress.progress(min(frame_i / total_frames, 1.0))
        except Exception as e:
            st.error("Error during tracking loop.")
            st.exception(e)
        finally:
            cap.release()
            out.release()

        if not os.path.exists(output_path):
            st.error("Processed output not found.")
            st.stop()

        st.success("✅ Tracking complete!")
        with open(output_path, "rb") as f:
            video_bytes = f.read()
        st.video(video_bytes)
        st.download_button("⬇️ Download Tracked Video", data=video_bytes, file_name="tracked_video.mp4")

# ----------------------------------------------
# 🧍 Webcam Mode (Real-Time Tracking)
# ----------------------------------------------
elif mode == "Webcam":
    st.markdown("🎥 **Webcam mode active — with YOLO tracking. Press Stop to end.**")

    class YOLOTrackerTransformer(VideoTransformerBase):
        def __init__(self):
            self.tracker_yaml = tracker_yaml

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = model.track(img, conf=confidence, persist=True, tracker=self.tracker_yaml)
            return results[0].plot()

    webrtc_streamer(
        key="yolo-webcam-tracker",
        video_transformer_factory=YOLOTrackerTransformer,
        media_stream_constraints={"video": True, "audio": False},
    )
