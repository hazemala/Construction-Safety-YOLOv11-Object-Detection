import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
from PIL import Image
import time

# ----------------------------------------------
# 🧠 App Configuration
# ----------------------------------------------
st.set_page_config(page_title="YOLO Object Detection", layout="wide")
st.title("🧠 YOLO Object Detection App")
st.markdown("Upload an image, video, or use your webcam for real-time YOLO detection.")

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
    model = YOLO('best.onnx')
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

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
# 🎥 Video Mode
# ----------------------------------------------
elif mode == "Video":
    uploaded_video = st.file_uploader("📂 Upload a video", type=["mp4", "mov", "avi"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        progress = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_count = 0
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=confidence)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)
            stframe.image(annotated_frame, channels="BGR", use_column_width=True)

            frame_count += 1
            progress.progress(min(frame_count / total_frames, 1.0))

        cap.release()
        out.release()
        st.success("✅ Video Processing Complete!")
        st.video(out_path)

# ----------------------------------------------
# 🧍 Webcam Mode (Image or Video)
# ----------------------------------------------
elif mode == "Webcam":
    st.subheader("🎥 Webcam Detection")

    webcam_mode = st.radio("Select Webcam Mode", ["Live Stream", "Capture Image"])
    run = st.button("▶️ Start Webcam")

    FRAME_WINDOW = st.image([])
    stop_button = st.button("⏹ Stop")

    cap = None

    if run:
        cap = cv2.VideoCapture(0)
        st.info("Webcam started. Press '⏹ Stop' to end.")
        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("No webcam detected.")
                break

            results = model(frame, conf=confidence)
            annotated_frame = results[0].plot()

            if webcam_mode == "Capture Image":
                FRAME_WINDOW.image(annotated_frame, channels="BGR", use_column_width=True)
                time.sleep(0.5)
                break
            else:
                FRAME_WINDOW.image(annotated_frame, channels="BGR", use_column_width=True)

            if stop_button:
                st.warning("⏹ Webcam stopped.")
                break

        cap.release()
    else:
        st.info("Activate the webcam by checking the '▶️ Start Webcam' box.")
