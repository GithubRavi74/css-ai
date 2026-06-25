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
        # 1. Get raw output matrix and squeeze the batch dimension
        # Output shape changes from (1, 14, 8400) -> (14, 8400)
        output = np.squeeze(outputs[0]) 
        
        # Transpose so rows are boxes: (14, 8400) -> (8400, 14)
        output = output.T 

        boxes = []
        confidences = []
        class_ids = []
        
        # Define your 10 custom PPE class names in their EXACT training order
        #classes = ["Helmet", "Vest", "Goggles", "Gloves", "Boots", "Class6", "Class7", "Class8", "Class9", "Class10"]
        # Replace the old placeholder classes list with this exact one:
        classes = [
            'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 
            'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle'
        ]

        # 2. Filter boxes by a confidence threshold
        CONF_THRESHOLD = 0.4
        
        for row in output:
            scores = row[4:] # The 10 class scores
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > CONF_THRESHOLD:
                # YOLO ONNX coordinates are scaled to 640x640
                xc, yc, w, h = row[0:4]
                
                # Convert center coordinates to top-left corner coordinates for OpenCV
                x1 = int((xc - w/2))
                y1 = int((yc - h/2))
                width = int(w)
                height = int(h)
                
                boxes.append([x1, y1, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

        # 3. Apply Non-Maximum Suppression to remove overlapping duplicates
        # (cv2.dnn.NMSBoxes requires standard Python floats/ints)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, 0.45)

        # 4. Draw boxes back onto your original image
        # Convert PIL image back to CV2 format for drawing
        orig_img = np.array(image.convert("RGB"))
        orig_h, orig_w, _ = orig_img.shape
        
        # Scaling factors (since model processed 640x640, scale boxes back to original image size)
        x_scale = orig_w / 640
        y_scale = orig_h / 640

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                
                # Scale coordinates up to original size
                x = int(x * x_scale)
                y = int(y * y_scale)
                w = int(w * x_scale)
                h = int(h * y_scale)
                
                # Draw the box rectangle
                cv2.rectangle(orig_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                
                # Prepare and draw label text
                label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
                cv2.putText(orig_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 5. Display the final annotated image in Streamlit
        st.image(orig_img, caption="Processed Image with Detections", use_container_width=True)
        
        st.write("Inference complete! Raw Output Shape:", outputs[0].shape)
        # Add your bounding box parsing logic here based on the output shape
