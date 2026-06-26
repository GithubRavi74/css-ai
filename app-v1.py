import sys
try:
    import cv2
except ImportError:
    # If standard cv2 fails, this forces the environment to look for headless
    import os
    os.system("pip install opencv-python-headless")

import streamlit as st
import os
import onnxruntime as ort
import numpy as np
from PIL import Image
from collections import Counter

st.set_page_config(page_title="AI Safety Auditor", layout="wide")
st.title("👷‍♂️ PPE Detection - AI Site Safety Auditor")

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
    st.success("ONNX Model engine loaded successfully!")
except Exception as e:
    st.error(f"Failed to load ONNX model: {e}")

# 2. Image Uploader
uploaded_file = st.file_uploader("Upload a site snapshot...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Original Image", use_container_width=True)
    
    # 3. Preprocessing image for ONNX format
    img_array = np.array(image.convert("RGB"))
    img_resized = cv2.resize(img_array, (640, 640))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_transpose = np.transpose(img_normalized, (2, 0, 1)) # HWC to CHW
    img_input = np.expand_dims(img_transpose, axis=0)       # Add batch dimension [1, 3, 640, 640]

    # 4. Run Inference
    if st.button("Analyze Site Compliance"):
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: img_input})
        
        # Squeeze the batch dimension: (1, 14, 8400) -> (14, 8400)
        output = np.squeeze(outputs[0]) 
        # Transpose so rows are individual boxes: (8400, 14)
        output = output.T 

        boxes = []
        confidences = []
        class_ids = []
        
        # Explicit model index dictionary mapping from training
        classes = [
            'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 
            'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle'
        ]

        # Filter boxes by a confidence threshold
        CONF_THRESHOLD = 0.4
        
        for row in output:
            scores = row[4:] # The 10 class scores
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > CONF_THRESHOLD:
                xc, yc, w, h = row[0:4]
                
                # Convert center coordinates to top-left corner coordinates for OpenCV
                x1 = int((xc - w/2))
                y1 = int((yc - h/2))
                width = int(w)
                height = int(h)
                
                boxes.append([x1, y1, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

        # Apply Non-Maximum Suppression to remove overlapping duplicates
        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, 0.45)

        # Setup base structures for drawing and counting
        orig_img = np.array(image.convert("RGB"))
        orig_h, orig_w, _ = orig_img.shape
        
        x_scale = orig_w / 640
        y_scale = orig_h / 640
        
        final_detections = []

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                current_class = classes[class_ids[i]]
                final_detections.append(current_class)
                
                # Scale coordinates up to original size
                x = int(x * x_scale)
                y = int(y * y_scale)
                w = int(w * x_scale)
                h = int(h * y_scale)
                
                # Dynamic bounding box coloring (Red for missing PPE infractions, Green for assets/safe gear)
                violators = ["NO-Hardhat", "NO-Mask", "NO-Safety Vest"]
                box_color = (220, 53, 69) if current_class in violators else (40, 167, 69)
                
                # Draw the box rectangle
                cv2.rectangle(orig_img, (x, y), (x + w, y + h), box_color, 3)
                
                # Draw label text
                label = f"{current_class}: {confidences[i]:.2f}"
                cv2.putText(orig_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 5. Display the final annotated image in Streamlit
        st.image(orig_img, caption="Processed Image with Detections", use_container_width=True)
        
        # ----------------------------------------------------
        # CUSTOMER-COMPATIBLE ANALYTICS DASHBOARD
        # ----------------------------------------------------
        st.write("---")
        st.subheader("📊 Executive Safety Analytics Summary")
        
        if len(final_detections) > 0:
            counts = Counter(final_detections)
            violators = ["NO-Hardhat", "NO-Mask", "NO-Safety Vest"]
            has_violation = any(v in counts for v in violators)
            
            # 1. Executive Status Header Alert
            if has_violation:
                st.error("⚠️ **Safety Compliance Alert:** The model detected missing or inadequate PPE on site personnel.")
            else:
                st.success("✅ **Compliance Passed:** All detected site personnel are properly equipped with standard safety gear.")
                
            # 2. Scorecard KPIs
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Total Objects Spotted", value=len(final_detections))
            with col2:
                st.metric(label="Workers On-Site", value=counts.get("Person", 0))
            with col3:
                # Calculate a dynamic compliance percentage
                total_violations = sum(counts[v] for v in violators if v in counts)
                safety_score = max(0, 100 - (total_violations * 25))
                st.metric(label="Site Compliance Score", value=f"{safety_score}%")

            # 3. Formatted Business Inventory Breakdowns
            st.markdown("### 📋 Detailed Inspection Ledger")
            report_data = []
            for item, count in counts.items():
                if item in violators:
                    status = "🔴 Violation / Risk Factor"
                elif item in ["Hardhat", "Mask", "Safety Vest"]:
                    status = "🟢 Compliant Protection"
                else:
                    status = "🔵 Registered Asset"
                    
                report_data.append({"Identified Object": item, "Quantity Detected": count, "Operational Status": status})
                
            st.table(report_data)
        else:
            st.info("Scan clear. No personnel, vehicles, or safety equipment were registered in this view frame.")
