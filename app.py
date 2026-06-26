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
st.title("静态/视频 👷‍♂️ PPE Detection - AI Site Safety Auditor")

# --- INITIALIZE SESSION STATE ---
if "video_detections" not in st.session_state:
    st.session_state.video_detections = None
if "total_frames" not in st.session_state:
    st.session_state.total_frames = 0
if "last_frame" not in st.session_state:
    st.session_state.last_frame = None

# Helper function to display the analytics dashboard cleanly
def show_dashboard(final_detections, violators, total_frames):
    st.write("---")
    st.subheader("📊 Executive Safety Analytics Summary")
    
    if len(final_detections) > 0 and total_frames > 0:
        raw_counts = Counter(final_detections)
        
        # Calculate the AVERAGE number of objects seen per frame (rounded to nearest whole number)
        # This converts "total instances across time" into "actual people on screen"
        avg_counts = {item: max(1, round(count / total_frames)) for item, count in raw_counts.items()}
        
        has_violation = any(v in avg_counts for v in violators)
        
        # 1. Executive Status Header Alert
        if has_violation:
            st.error("⚠️ **Safety Compliance Alert:** The model detected ongoing missing or inadequate PPE on site personnel.")
        else:
            st.success("✅ **Compliance Passed:** Site personnel are consistently equipped with standard safety gear.")
            
        # 2. Scorecard KPIs (Using Normalized Averages)
        col1, col2, col3 = st.columns(3)
        with col1:
            total_avg_objects = sum(avg_counts.values())
            st.metric(label="Avg Objects Spotted/Frame", value=total_avg_objects)
        with col2:
            st.metric(label="Estimated Workers on Site", value=avg_counts.get("Person", 0))
        with col3:
            # Calculate safety score based on averaged violations
            total_violations = sum(avg_counts[v] for v in violators if v in avg_counts)
            safety_score = max(0, 100 - (total_violations * 25)) 
            st.metric(label="Site Compliance Score", value=f"{safety_score}%")

        # 3. Formatted Business Inventory Breakdowns
        st.markdown("### 📋 Detailed Inspection Ledger (Averaged Stream Data)")
        report_data = []
        for item, count in avg_counts.items():
            if item in violators:
                status = "🔴 Violation / Risk Factor"
            elif item in ["Hardhat", "Mask", "Safety Vest"]:
                status = "🟢 Compliant Protection"
            else:
                status = "🔵 Registered Asset"
                
            report_data.append({"Identified Object": item, "Avg Quantity On-Screen": count, "Operational Status": status})
            
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
            
            # Show dashboard report for image (treated as a 1-frame video)
            show_dashboard(final_detections, detector.violators, total_frames=1)
            
    else:
        # --- VIDEO PROCESSING PIPELINE ---
        st.info("🎥 Video file detected. Click the button below to start frame-by-frame analysis.")
        
        if st.button("Run Video AI Analysis"):
            # Reset session state on new button clicks
            st.session_state.video_detections = None
            st.session_state.total_frames = 0
            st.session_state.last_frame = None
            
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            cap = cv2.VideoCapture(tfile.name)
            video_frame_placeholder = st.empty()
            
            all_video_detections = []
            frame_count = 0
            last_processed_rgb = None
            
            # Process frame by frame
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break  
                
                frame_count += 1
                
                # Run ONNX engine frame processing
                frame_detections, annotated_frame_bgr = detector.process_frame(frame)
                all_video_detections.extend(frame_detections)
                
                # Convert back to RGB for web browser playback rendering
                last_processed_rgb = cv2.cvtColor(annotated_frame_bgr, cv2.COLOR_BGR2RGB)
                video_frame_placeholder.image(last_processed_rgb, channels="RGB", use_container_width=True)
            
            cap.release()
            
            # Save data AND the final frame image to state memory so it stays on screen
            st.session_state.video_detections = all_video_detections
            st.session_state.total_frames = frame_count
            st.session_state.last_frame = last_processed_rgb
            st.rerun()  

        # Render step after rerun loop finishes
        if st.session_state.video_detections is not None:
            st.success("🎉 Video Analysis Complete!")
            
            # Keep the final frozen annotated snapshot visible on screen
            if st.session_state.last_frame is not None:
                st.image(st.session_state.last_frame, caption="Final Video Frame Analysis Snapshot", use_container_width=True)
                
            # Render the mathematically normalized dashboard
            show_dashboard(
                st.session_state.video_detections, 
                detector.violators, 
                st.session_state.total_frames
            )
