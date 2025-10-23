import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
from PIL import Image
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

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=confidence)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)

            frame_count += 1
            progress.progress(min(frame_count / total_frames, 1.0))

        cap.release()
        out.release()
        st.success("✅ Video Processing Complete!")
        st.video(out_path)

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
