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

# --- BRANDED LOGO HEADER INTEGRATION ---
# --- BRANDED LOGO HEADER INTEGRATION ---
try:
    logo_img = Image.open("IDEA LOGIC Logo.jpg")
    
    # 1. Render the logo first (Adjust width to make it as big as you like)
    st.image(logo_img, width=350)
    
    # 2. Render the title directly below it
    st.title("AI Construction Site Safety Detection")
    st.write("") # Adds a tiny bit of clean vertical spacing below the title

except FileNotFoundError:
    st.title("👷‍♂️ Construction Site Safety Detection - AI Site Safety Auditor")

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
        avg_counts = {item: max(1, round(count / total_frames)) for item, count in raw_counts.items()}
        has_violation = any(v in avg_counts for v in violators)
        
        if has_violation:
            st.error("⚠️ **Safety Compliance Alert:** The model detected ongoing missing or inadequate PPE on site personnel.")
        else:
            st.success("✅ **Compliance Passed:** Site personnel are consistently equipped with standard safety gear.")
            
        col1, col2, col3 = st.columns(3)
        with col1:
            total_avg_objects = sum(avg_counts.values())
            st.metric(label="Avg Objects Spotted/Frame", value=total_avg_objects)
        with col2:
            st.metric(label="Estimated Workers on Site", value=avg_counts.get("Person", 0))
        with col3:
            total_violations = sum(avg_counts[v] for v in violators if v in avg_counts)
            safety_score = max(0, 100 - (total_violations * 25)) 
            st.metric(label="Site Compliance Score", value=f"{safety_score}%")

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
    st.success("AI successfully loaded")
except Exception as e:
    st.error(f"Failed to initialize model wrapper engine: {e}")

# 2. File Uploader to support both media types
uploaded_file = st.file_uploader(
    "Upload a site snapshot or video clip...", 
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

# Set a compact image rendering width standard
COMPACT_WIDTH = 420

if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1].lower()
    is_video = file_extension in ["mp4", "avi", "mov"]

    if not is_video:
        # --- IMAGE PROCESSING PIPELINE ---
        image = Image.open(uploaded_file)
        
        # Split layout into smaller structured display nodes
        img_col1, img_col2 = st.columns(2)
        
        with img_col1:
            st.subheader("📸 Original View")
            st.image(image, use_container_width=False, width=COMPACT_WIDTH)
            analyze_clicked = st.button("Analyze Image Compliance", type="primary")
        
        if analyze_clicked:
            img_array = np.array(image.convert("RGB"))
            cv2_img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            final_detections, annotated_bgr = detector.process_frame(cv2_img_bgr)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            
            with img_col2:
                st.subheader("🔍 AI Detections")
                st.image(annotated_rgb, use_container_width=False, width=COMPACT_WIDTH)
            
            show_dashboard(final_detections, detector.violators, total_frames=1)
            
    else:
        # --- VIDEO PROCESSING PIPELINE ---
        st.info("🎥 Video file detected. Click the button below to start frame-by-frame analysis.")
        run_analysis = st.button("Run Video AI Analysis", type="primary")
        
        vid_col1, vid_col2 = st.columns(2)
        
        if run_analysis:
            st.session_state.video_detections = None
            st.session_state.total_frames = 0
            st.session_state.last_frame = None
            
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            cap = cv2.VideoCapture(tfile.name)
            
            with vid_col1:
                st.subheader("🎞️ Processing Stream")
                video_frame_placeholder = st.empty()
            
            all_video_detections = []
            frame_count = 0
            last_processed_rgb = None
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break  
                
                frame_count += 1
                
                frame_detections, annotated_frame_bgr = detector.process_frame(frame)
                all_video_detections.extend(frame_detections)
                
                last_processed_rgb = cv2.cvtColor(annotated_frame_bgr, cv2.COLOR_BGR2RGB)
                video_frame_placeholder.image(last_processed_rgb, channels="RGB", width=COMPACT_WIDTH)
            
            cap.release()
            
            st.session_state.video_detections = all_video_detections
            st.session_state.total_frames = frame_count
            st.session_state.last_frame = last_processed_rgb
            st.rerun()  

        if st.session_state.video_detections is not None:
            st.success("🎉 Video Analysis Complete!")
            
            res_col1, res_col2 = st.columns(2)
            if st.session_state.last_frame is not None:
                with res_col1:
                    st.subheader("🎯 Final Stream Frame")
                    st.image(st.session_state.last_frame, use_container_width=False, width=COMPACT_WIDTH)
                
            show_dashboard(
                st.session_state.video_detections, 
                detector.violators, 
                st.session_state.total_frames
            )
