import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
from PIL import Image
import os

# ---------------- APP HEADER ---------------- #
st.set_page_config(
    page_title="PPE Detection System",
    page_icon="🦺",
    layout="centered"
)

# Title and Subtitle
st.markdown("""
<div style="text-align: center; padding: 10px; border-bottom: 2px solid #f0f0f0;">
    <h1 style="color:#00BFFF;">🦺 PPE Detection System</h1>
    <h4 style="color:#FFFF00;">Helmet & Reflective Vest Detection using <strong>YOLOv8s</strong></h4>
    <p style="color:#FFFF00; font-size:14px;">Developed by <strong>Rayyan Ahmed</strong></p>
</div>
""", unsafe_allow_html=True)

# ---------------- Background ----------------
st.markdown("""
<style>
.stApp {
    background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)),
                      url("https://www.shutterstock.com/image-photo/construction-new-residential-complex-many-600nw-2487410463.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    color: white;
}
h1 { color: #FFD700; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ---------------- Sidebar Styling ----------------
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: rgba(0, 0, 0.7, 0.5);
    color: white;
}
[data-testid="stSidebar"] h1, h2, h3 { color: #00BFFF; }
::-webkit-scrollbar-thumb { background: #FFD700; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ---------------- Sidebar Info ----------------
with st.sidebar.expander("📌 Project Intro"):
    st.markdown("""
    - Detect **helmets and reflective vests** in images, videos, or uploaded files  
    - Apply **bounding boxes and labels** for detected PPE  
    - Compare **original vs detected video** side-by-side for portfolio demonstration  
    """)

with st.sidebar.expander("👨‍💻 Developer's Intro"):
    st.markdown("""
    - **Hi, I'm Rayyan Ahmed**
    - Google Certified **AI Prompt Specialist**  
    - IBM Certified **Advanced LLM FineTuner**  
    - Hugging Face Certified: **Fundamentalist of LLMs**  
    - Expert in **EDA, ML, RL, ANN, CNN, CV, RNN, NLP, LLMs**  
    [💼 Visit LinkedIn](https://www.linkedin.com/in/rayyan-ahmed-504725321/)
    """)

with st.sidebar.expander("🛠️ Tech Stack Used"):
    st.markdown("""
- 🧠 **Ultralytics YOLOv8** → Object detection for PPE  
- 🎥 **OpenCV** → Frame reading, video handling, drawing bounding boxes  
- ⚙️ **NumPy** → Array and pixel operations  
- 🌐 **Streamlit** → Interactive web interface  
- 🖥️ **imageio / imageio-ffmpeg** → Save detected video as MP4  
""")


# Load model
@st.cache_resource
def load_model():
    return YOLO("saved_trained_model.pt")

model = load_model()

# Sidebar
st.sidebar.title("Upload Options")
option = st.sidebar.radio("Choose input type:", ["Image", "Video"])

from PIL import Image
import streamlit as st
import io
import base64
import cv2
import numpy as np

# ---------------- IMAGE DETECTION ---------------- #
if option == "Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        image = Image.open(uploaded_image)

        # Convert image to bytes for HTML embedding
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes = img_bytes.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode()

        # Display original image via HTML to control width
        st.markdown("### Orignal Image")
        st.markdown(f"""
        <div style="width:600px">
            <img src="data:image/png;base64,{img_b64}" style="width:100%;">
        </div>
        """, unsafe_allow_html=True)

        if st.button("Run Detection"):
            results = model.predict(image)
            result_img = results[0].plot()

            # Convert result to bytes for HTML embedding
            _, buffer = cv2.imencode(".png", result_img)
            result_bytes = buffer.tobytes()
            result_b64 = base64.b64encode(result_bytes).decode()

            # Display detected image via HTML to control width
            st.markdown("### Detected Image")
            st.markdown(f"""
            <div style="width:600px">
                <img src="data:image/png;base64,{result_b64}" style="width:100%;">
            </div>
            """, unsafe_allow_html=True)


# ---------------- VIDEO DETECTION (SIMPLE & MP4 ONLY) ---------------- #

else:
    uploaded_video = st.file_uploader("Upload MP4 video", type=["mp4"])

    if uploaded_video:
        import tempfile
        import cv2
        import os
        import imageio
        import base64

        # Save uploaded video temporarily
        input_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        input_video.write(uploaded_video.read())
        input_video.close()

        # ---------------- Original video preview via HTML ---------------- #
        st.markdown("### Original Video")
        with open(input_video.name, "rb") as f:
            video_bytes = f.read()
        video_b64 = base64.b64encode(video_bytes).decode()
        st.markdown(f"""
        <video width="600" controls>
            <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
        </video>
        """, unsafe_allow_html=True)

        x = 1

        if st.button("Run Detection"):
            cap = cv2.VideoCapture(input_video.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Output MP4 path
            output_path = os.path.abspath(f"ppe_detected_output_{x}.mp4")
            x += 1

            writer = imageio.get_writer(output_path, fps=fps, codec='libx264')

            # Initialize progress bar
            progress_text = st.empty()
            progress_bar = st.progress(0)

            current_frame = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.predict(frame, conf=0.4, verbose=False)
                annotated = results[0].plot()
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                writer.append_data(annotated_rgb)

                # Update progress
                current_frame += 1
                progress_text.text(f"Processing frames: {current_frame}/{total_frames}")
                progress_bar.progress(current_frame / total_frames)

            cap.release()
            writer.close()

            st.success("Video detection completed")
            st.write(f"📁 Saved video path: `{output_path}`")

            # ---------------- Detected video preview via HTML ---------------- #
            st.markdown("### Detected Video")
            if os.path.exists(output_path):
                with open(output_path, "rb") as f:
                    video_bytes = f.read()
                video_b64 = base64.b64encode(video_bytes).decode()
                st.markdown(f"""
                <video width="600" controls>
                    <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
                </video>
                """, unsafe_allow_html=True)
                st.success(f"Playing video: {os.path.basename(output_path)}")
            else:
                st.error("Detected video file not found.")








            





