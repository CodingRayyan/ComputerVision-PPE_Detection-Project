import streamlit as st
import os

video_path = r"C:\Users\HP\Downloads\Computer Vision\Object Detection - PPE Detection\ppe_detected_output_1.mp4"

if os.path.exists(video_path):
    with open(video_path, "rb") as video_file:
        st.video(video_file.read())
    st.success("Playing video: ppe_detected_output_1.mp4")
else:
    st.error("Video file not found.")
