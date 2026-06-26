# app.py
import sys
try:
    import cv2
except ImportError:
    import os
    os.system("pip install opencv-python-headless")

import streamlit as st
import numpy as np
import tempfile
from PIL import Image
from collections import Counter
from detector import PPEModel  # Import your backend engine wrapper

st.set_page_config(page_title="AI Safety Auditor", layout="wide")
st.title("👷‍♂️ PPE Detection - AI Site Safety Auditor")

# --- INITIALIZE SESSION STATE ---
# This prevents Streamlit from wiping out your results when the video loop ends
if "video_detections" not in st.session_state:
    st.session_state.video_detections = None

# Helper function to display the analytics dashboard cleanly
def show_dashboard(final_detections, violators):
    st.write("---")
    st.subheader("📊 Executive Safety Analytics Summary")
    
    if len(final_detections) > 0:
        counts = Counter(final_detections)
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
            st.metric(label="Workers Spotted (Total Instances)", value=counts.get("Person", 0))
        with col3:
            # Calculate a dynamic compliance percentage
            total_violations = sum(counts[v] for v in violators if v in counts)
            safety_score = max(0, 100 - (total_violations * 5)) 
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
                
            report_data.append({"Identified Object": item, "Total Instances Seen": count, "Operational Status": status})
            
        st.table(report_data)
    else:
        st.info("Scan clear. No personnel or assets were registered in this file.")


# 1. Load the ONNX model wrapper safely inside Streamlit's resource cache
@st.cache_resource
def load_cached_detector():
    return PPEModel(model_path="models/best.onnx")

try:
    detector = load_cached_detector()
    st.success("ONNX Model engine loaded cleanly into application cache!")
except Exception as e:
    st.error(f"Failed to initialize model wrapper engine: {e}")

# 2. File Uploader to support both media types
uploaded_file = st.file_uploader(
    "Upload a site snapshot or video clip...", 
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

if uploaded_file is not None:
    # Check file type
    file_extension = uploaded_file.name.split(".")[-1].lower()
    is_video = file_extension in ["mp4", "avi", "mov"]

    if not is_video:
        # --- IMAGE PROCESSING PIPELINE ---
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Original Image", use_container_width=True)
        
        if st.button("Analyze Image Compliance"):
            img_array = np.array(image.convert("RGB"))
            cv2_img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            final_detections, annotated_bgr = detector.process_frame(cv2_img_bgr)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption="Processed Image with Detections", use_container_width=True)
            
            # Show dashboard report for image
            show_dashboard(final_detections, detector.violators)
            
    else:
        # --- VIDEO PROCESSING PIPELINE ---
        st.info("🎥 Video file detected. Click the button below to start frame-by-frame analysis.")
        
        if st.button("Run Video AI Analysis"):
            # Clear previous runs from session state
            st.session_state.video_detections = None
            
            # Streamlit uploads files to memory. Save temporarily locally for OpenCV link path
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            cap = cv2.VideoCapture(tfile.name)
            
            # Create a live placeholder in the UI where frames render sequentially
            video_frame_placeholder = st.empty()
            
            all_video_detections = []
            
            # Process frame by frame
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break  # Video ends
                
                # Use engine to process the frame
                frame_detections, annotated_frame_bgr = detector.process_frame(frame)
                all_video_detections.extend(frame_detections)
                
                # Convert back to RGB format for Streamlit
                annotated_frame_rgb = cv2.cvtColor(annotated_frame_bgr, cv2.COLOR_BGR2RGB)
                
                # Render live video playback frames
                video_frame_placeholder.image(annotated_frame_rgb, channels="RGB", use_container_width=True)
            
            cap.release()
            
            # Save the compiled results into the Session State memory bank!
            st.session_state.video_detections = all_video_detections
            st.rerun()  # Forces a clean page refresh to display the dashboard permanently

        # If a video run has completed successfully in this session, render the dashboard
        if st.session_state.video_detections is not None:
            st.success("🎉 Video Analysis Complete!")
            show_dashboard(st.session_state.video_detections, detector.violators)
