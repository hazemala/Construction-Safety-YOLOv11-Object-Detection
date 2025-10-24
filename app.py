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
st.title("🎯 YOLO Object Detection + Tracking App")
st.markdown("Upload an image or video for detection and tracking using YOLO.")

# ----------------------------------------------
# ⚙️ Sidebar Configuration
# ----------------------------------------------
st.sidebar.header("⚙️ Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

# ----------------------------------------------
# 🧩 Load Model
# ----------------------------------------------
@st.cache_resource
def load_model():
    model = YOLO("best.onnx")  # replace with your trained model
    return model

model = load_model()

# ----------------------------------------------
# 🎛️ Mode Selection
# ----------------------------------------------
st.sidebar.subheader("📷 Select Input Source")
mode = st.sidebar.radio("Choose Input Type", ["Image", "Video"])

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
# 🎥 Video Mode (tracking, silent processing)
# ----------------------------------------------
elif mode == "Video":
    uploaded_video = st.file_uploader("📹 Upload a video", type=["mp4", "mov", "avi", "mkv"])

    if uploaded_video:
        st.info("📥 Saving and processing your video... Please wait.")
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)

        if not cap.isOpened():
            st.error("❌ Unable to open the uploaded video.")
            st.stop()

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_path = "output_tracked.mp4"
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        progress = st.progress(0)
        frame_i = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 🔹 Run detection + tracking (default tracker)
            results = model.track(frame, persist=True, conf=confidence)
            annotated_frame = results[0].plot()

            out.write(annotated_frame)

            frame_i += 1
            if total_frames > 0:
                progress.progress(min(frame_i / total_frames, 1.0))

        cap.release()
        out.release()

        st.success("✅ Video processed successfully!")

        # ----------------------------------------------
        # 💾 Download Processed Video
        # ----------------------------------------------
        with open(output_path, "rb") as f:
            video_bytes = f.read()

        st.download_button(
            label="⬇️ Download Tracked Video",
            data=video_bytes,
            file_name="tracked_output.mp4",
            mime="video/mp4"
        )

# ----------------------------------------------
# 🧍 Webcam Mode (Real-Time Tracking)
# ----------------------------------------------
elif mode == "Webcam":
    st.markdown("🎥 **Webcam mode active — with YOLO tracking. Press Stop to end.**")

    class YOLOTrackerTransformer(VideoTransformerBase):

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = model.track(img, conf=confidence, persist=True)
            return results[0].plot()

    webrtc_streamer(
        key="yolo-webcam-tracker",
        video_transformer_factory=YOLOTrackerTransformer,
        media_stream_constraints={"video": True, "audio": False},
    )
