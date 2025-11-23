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
st.set_page_config(page_title="Construction Safety – YOLOv11", layout="wide")
st.title("🏗️ Construction Safety Detection System")
st.markdown("Real-time PPE detection using YOLOv11.")

# ----------------------------------------------
# 🧭 Sidebar Navigation
# ----------------------------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Home", "🔍 Image", "🎥 Video", "📷 Webcam"])

st.sidebar.header("⚙️ Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

# ----------------------------------------------
# 🧩 Load Model
# ----------------------------------------------
@st.cache_resource
def load_model():
    model = YOLO("best.onnx", task='detect')  # replace with your trained model
    return model

model = load_model()

# ----------------------------------------------
# 🏠 Home Page
# ----------------------------------------------
if page == "🏠 Home":

    st.markdown("""
    ## 🔍 **Project Summary**
    This project improves safety compliance at construction sites by automatically detecting workers who are not wearing required Personal Protective Equipment (PPE).  
    It uses a YOLOv11m object detection model trained on 2.2k images (Roboflow dataset) with default YOLO augmentations.

    ---

    ## 🎯 **Objectives**
    - Ensure PPE compliance (helmet & vest detection)
    - Provide real-time alerts
    - Build a deployable, lightweight computer vision solution
    - Support real-time, image, and video detection

    ---

    ## 🧠 **Model & Dataset**
    **Model:** YOLOv11m  
    **Classes Detected:**
    - Helmet  
    - Vest  
    - No-Helmet  
    - No-Vest  

    **Training Dataset:**  
    - 2.2k images  
    - Roboflow preprocessing  
    - YOLO default augmentation enabled  

    ---

    ## 📊 **Performance Metrics**
    - **mAP50:** 93%  
    - **mAP50-95:** 61%  
    - **Precision:** 93%  
    - **Recall:** 92%  

    These results show strong detection accuracy suitable for real-time deployment.

    ---

    ## 💻 **Application Features**
    - Real-time webcam detection  
    - Image detection  
    - Video detection + processed video download   

    ---

    ## 🧩 **Tech Stack**
    - Python  
    - Streamlit  
    - YOLOv11  
    - OpenCV  
    - NumPy  
    - Roboflow  

    ---

    ## 🚧 **Use Cases**
    - Construction site safety monitoring  
    - Automated PPE compliance  
    - Reducing workplace accidents  
    - CCTV-based live detection systems  

    ---

    ### ✅ Get started using the sidebar to run detection.
    """)

# ----------------------------------------------
# 📸 IMAGE PAGE
# ----------------------------------------------
elif page == "🔍 Image":

    st.header("📸 Image Detection")

    uploaded_image = st.file_uploader("📂 Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        img_np = np.array(image)

        results = model(img_np, conf=confidence)
        annotated = results[0].plot()

        st.image(annotated, caption="🔍 Detection Result", use_column_width=True)

# ----------------------------------------------
# 🎥 VIDEO PAGE
# ----------------------------------------------
elif page == "🎥 Video":

    st.header("🎥 Video Detection")

    uploaded_video = st.file_uploader("📹 Upload a video", type=["mp4", "mov", "avi", "mkv"])

    if uploaded_video:
        st.info("📥 Processing your video... Please wait.")
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

        output_path = "output_detected.mp4"
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        progress = st.progress(0)
        frame_i = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=confidence)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)

            frame_i += 1
            progress.progress(frame_i / total_frames)

        cap.release()
        out.release()

        st.success("✅ Video processed successfully!")

        with open(output_path, "rb") as f:
            video_bytes = f.read()

        st.download_button(
            label="⬇️ Download Detected Video",
            data=video_bytes,
            file_name="detected_output.mp4",
            mime="video/mp4"
        )

# ----------------------------------------------
# 📷 WEBCAM PAGE
# ----------------------------------------------
elif page == "📷 Webcam":

    st.header("📷 Real-Time Webcam Detection")

    class YOLODetectionTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = model(img, conf=confidence)
            return results[0].plot()

    webrtc_streamer(
        key="yolo-webcam-detector",
        video_transformer_factory=YOLODetectionTransformer,
        media_stream_constraints={"video": True, "audio": False},
    )
