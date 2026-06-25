import sys
try:
    import cv2
except ImportError:
    # If standard cv2 fails, this forces the environment to look for headless
    import os
    os.system("pip install opencv-python-headless")

import streamlit as st
import os
from ultralytics import YOLO

# ... rest of your code below



import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image

st.title("PPE Detection - ONNX Runtime")

# 1. Load the ONNX session safely using caching
@st.cache_resource
def load_onnx_model():
    # Make sure your exported model is inside a 'models' folder or update this path
    model_path = "models/best.onnx" 
    # Force the CPU provider since Streamlit Cloud does not provide a GPU
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    return session

try:
    session = load_onnx_model()
    st.success("ONNX Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load ONNX model: {e}")

# 2. Image Uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # 3. Preprocessing image for ONNX format
    # (YOLO ONNX expects scaled float32 arrays, usually structured as [1, 3, 640, 640])
    img_array = np.array(image.convert("RGB"))
    img_resized = cv2.resize(img_array, (640, 640))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_transpose = np.transpose(img_normalized, (2, 0, 1)) # HWC to CHW
    img_input = np.expand_dims(img_transpose, axis=0)       # Add batch dimension [1, 3, 640, 640]

    # 4. Run Inference
    if st.button("Detect PPE"):
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: img_input})
        
        st.write("Inference complete! Raw Output Shape:", outputs[0].shape)
        # Add your bounding box parsing logic here based on the output shape
