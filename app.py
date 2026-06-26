# app.py
import sys
try:
    import cv2
except ImportError:
    import os
    os.system("pip install opencv-python-headless")

import streamlit as st
import numpy as np
from PIL import Image
from collections import Counter
from detector import PPEModel  # Import the newly refactored engine wrapper

st.set_page_config(page_title="AI Safety Auditor", layout="wide")
st.title("👷‍♂️ PPE Detection - AI Site Safety Auditor")

# 1. Load the ONNX model wrapper safely inside Streamlit's resource cache
@st.cache_resource
def load_cached_detector():
    return PPEModel(model_path="models/best.onnx")

try:
    detector = load_cached_detector()
    st.success("ONNX Model engine loaded cleanly into application cache!")
except Exception as e:
    st.error(f"Failed to initialize model wrapper engine: {e}")

# 2. Image Upload Component
uploaded_file = st.file_uploader("Upload a site snapshot...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image and preserve the original layout to pass to OpenCV backend
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Original Image", use_container_width=True)
    
    # 3. Process Button Trigger
    if st.button("Analyze Site Compliance"):
        # Convert PIL Image to standard OpenCV BGR image matrix numpy array format
        img_array = np.array(image.convert("RGB"))
        cv2_img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Call the standalone processing engine
        final_detections, annotated_bgr = detector.process_frame(cv2_img_bgr)
        
        # Convert frame back to standard RGB visualization for standard Streamlit rendering
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, caption="Processed Image with Detections", use_container_width=True)
        
        # ----------------------------------------------------
        # CUSTOMER-COMPATIBLE ANALYTICS DASHBOARD
        # ----------------------------------------------------
        st.write("---")
        st.subheader("📊 Executive Safety Analytics Summary")
        
        if len(final_detections) > 0:
            counts = Counter(final_detections)
            violators = detector.violators
            has_violation = any(v in counts for v in violators)
            
            # Executive Status Header Alert
            if has_violation:
                st.error("⚠️ **Safety Compliance Alert:** The model detected missing or inadequate PPE on site personnel.")
            else:
                st.success("✅ **Compliance Passed:** All detected site personnel are properly equipped with standard safety gear.")
                
            # Scorecard KPIs
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

            # Formatted Business Inventory Breakdowns
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
