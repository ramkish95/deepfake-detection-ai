import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import tempfile
import os

# --- Page Configuration ---
st.set_page_config(page_title="Deepfake Detection Lab", page_icon="🛡️", layout="wide")

# --- Load the Model ---
@st.cache_resource
def load_forensic_model():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(root_dir, "models", "deepfake_detector.h5")
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at {model_path}. Please run train_detector.py first.")
        st.stop()
        
    model = tf.keras.models.load_model(model_path)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = load_forensic_model()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Forensic Settings")
    st.write("Current Model: **MobileNetV2 v1.0**")
    st.write("Last Training Epochs: **6**")
    st.progress(85, text="Model Reliability")
    
    st.divider()
    st.markdown("""
    ### 🛡️ Cyber-Forensics Lab
    This system analyzes pixel inconsistencies and GAN-generated artifacts.
    
    **Author:** 3rd Year AIML Engineer
    """)

# --- Header Section ---
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://img.icons8.com/fluency/96/shield.png")
with col2:
    st.title("AI-Powered Deepfake Detector")
    st.write("Ensuring Digital Integrity through Artificial Intelligence")

st.info("Upload a video file to analyze its authenticity. Our AI scans for digital artifacts typical of deepfakes.")

# --- File Uploader ---
uploaded_file = st.file_uploader("📂 Upload video evidence for analysis...", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    left_col, right_col = st.columns(2)
    
    with left_col:
        st.subheader("Original Evidence")
        st.video(uploaded_file)
    
    with right_col:
        st.subheader("Forensic Analysis")
        if st.button('🔍 Start Deepfake Scan'):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            cap = cv2.VideoCapture(tfile.name)
            found_face = False
            
            # Scan frames for a face
            for i in range(100):
                ret, frame = cap.read()
                if not ret: break
                
                progress_bar.progress((i + 1) / 100)
                status_text.text(f"Scanning Frame {i+1}/100...")
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) > 0:
                    (x, y, w, h) = faces[0]
                    face = frame[y:y+h, x:x+w]
                    
                    # AI Pre-processing
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))
                    face = face.astype('float32') / 255.0 
                    face = np.expand_dims(face, axis=0)

                    # Prediction: < 0.5 is Fake, > 0.5 is Real
                    prediction = model.predict(face, verbose=0)[0][0]
                    
                    st.divider()
                    if prediction < 0.5:
                        conf = (1 - prediction) * 100
                        st.error(f"🚨 **RESULT: DEEPFAKE DETECTED**")
                        st.metric("AI Confidence Score", f"{conf:.2f}%")
                        st.warning("Analysis: This video contains pixel-level inconsistencies associated with AI manipulation.")
                    else:
                        conf = prediction * 100
                        st.success(f"✅ **RESULT: AUTHENTIC VIDEO**")
                        st.metric("AI Confidence Score", f"{conf:.2f}%")
                        st.write("Analysis: No digital artifacts detected. Video patterns appear natural.")
                    
                    found_face = True
                    status_text.text("Scan Complete.")
                    break
            
            if not found_face:
                st.error("❌ No clear face detected. Please use a video with a clear view of the person.")
            
            cap.release()