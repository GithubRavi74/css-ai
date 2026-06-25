import streamlit as st
import cv2
import numpy as np
from detector import PPEDetector

st.set_page_config(page_title="PPE Safety System", layout="wide")

st.title("🦺 Real-Time PPE Detection System (ONNX)")

detector = PPEDetector("models/best.onnx")

run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Camera not accessible")
        break

    # inference
    outputs = detector.predict(frame)

    # NOTE: simplified visualization placeholder
    # (we can refine bounding box decoding later)
    frame = cv2.putText(frame, "PPE Detection Running...", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()
