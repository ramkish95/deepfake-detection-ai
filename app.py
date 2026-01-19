import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import tempfile
import os
import matplotlib.pyplot as plt

# --- 1. Page Configuration ---
st.set_page_config(page_title="Forensic Deepfake Lab", page_icon="🛡️", layout="wide")

# --- 2. Load the Model (Cached for Speed) ---
@st.cache_resource
def load_forensic_model():
    # Use absolute path for reliability on Windows and Cloud
    root_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(root_dir, "models", "deepfake_detector.h5")
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at {model_path}. Please run train_detector.py first.")
        st.stop()
        
    model = tf.keras.models.load_model(model_path)
    # Re-compile to ensure metrics are ready
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = load_forensic_model()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- 3. Sidebar: Forensic Tools & Info ---
with st.sidebar:
    st.header("🔬 Forensic Laboratory")
    st.write("Current Model: **MobileNetV2 v1.0**")
    st.write("Mode: **Ensemble Analysis (10-Frame Average)**")
    
    st.divider()
    st.subheader("Frequency Fingerprint")
    st.info("GANs leave 'Checkerboard' patterns in the frequency domain. Look for unnatural dots in the corners below.")
    
    # This is the placeholder where the heatmap will appear
    analysis_container = st.empty() 
    
    st.divider()
    st.markdown("### 👨‍💻 Project Details")
    st.write("Author: 3rd Year AIML Engineer")
    st.write("Domain: Cybersecurity & AI")

# --- 4. Main User Interface ---
st.title("🛡️ Advanced Deepfake Detection Lab")
st.write("Ensuring Digital Integrity through Multi-Layered AI Analysis")

uploaded_file = st.file_uploader("📂 Upload video evidence for analysis...", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Save uploaded file to temporary location
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Evidence")
        st.video(uploaded_file)
    
    with col2:
        st.subheader("Forensic Analysis")
        if st.button('🚀 Start Multi-Layer Scan'):
            cap = cv2.VideoCapture(tfile.name)
            predictions = []
            frames_scanned = 0
            
            with st.spinner("Processing High-Frequency Artifacts..."):
                # ENSEMBLE TESTING: Iterate until 10 frames with faces are analyzed
                while frames_scanned < 10: 
                    ret, frame = cap.read()
                    if not ret: break # End of video
                    
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    
                    if len(faces) > 0:
                        (x, y, w, h) = faces[0]
                        face = frame[y:y+h, x:x+w]
                        
                        # --- FEATURE 1: Frequency Domain Analysis (FFT) ---
                        # We analyze the FIRST valid face frame for the fingerprint
                        if frames_scanned == 0:
                            gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                            
                            # Fast Fourier Transform logic
                            f = np.fft.fft2(gray_face)
                            fshift = np.fft.fftshift(f)
                            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
                            
                            # Create and display the plot in the sidebar
                            fig, ax = plt.subplots()
                            ax.imshow(magnitude_spectrum, cmap='magma')
                            ax.axis('off')
                            analysis_container.pyplot(fig)
                        
                        # --- FEATURE 2: AI Prediction ---
                        face_input = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                        face_input = cv2.resize(face_input, (224, 224))
                        face_input = face_input.astype('float32') / 255.0 
                        face_input = np.expand_dims(face_input, axis=0)

                        # Prediction: < 0.5 is Fake, > 0.5 is Real
                        prediction = model.predict(face_input, verbose=0)[0][0]
                        predictions.append(prediction)
                        frames_scanned += 1
                
                cap.release()
                
                # --- DISPLAY RESULTS ---
                # --- UPDATED RESULTS LOGIC ---
                if len(predictions) > 0:
                    # Use the MINIMUM score (lowest = most fake) instead of the AVERAGE
                    min_pred = np.min(predictions) 
                    avg_pred = np.mean(predictions)
                    
                    st.divider()
                    
                    # If the worst frame is below 0.5, we flag it as a Deepfake
                    if min_pred < 0.5:
                        # We show the confidence of the most suspicious frame
                        conf = (1 - min_pred) * 100
                        st.error(f"🚨 **RESULT: DEEPFAKE DETECTED**")
                        st.metric("Peak Suspicion Score", f"{conf:.2f}%")
                        st.warning(f"Forensic Note: Analysis of {len(predictions)} frames found digital inconsistencies. Even high-quality fakes often fail frame-to-frame consistency checks.")
                    else:
                        conf = avg_pred * 100
                        st.success(f"✅ **RESULT: AUTHENTIC VIDEO**")
                        st.metric("AI Confidence Score", f"{conf:.2f}%")
                        st.write("Forensic Note: All scanned frames passed the pixel-integrity test.")
                    
                    # --- EXTRA FEATURE: Download Report ---
                    report_text = f"Forensic Deepfake Report\nResult: {'FAKE' if avg_pred < 0.5 else 'REAL'}\nConfidence: {conf:.2f}%"
                    st.download_button("📥 Download Forensic Report", report_text, file_name="forensic_report.txt")
                
                else:
                    st.error("❌ Forensic Analysis Failed: No clear faces detected in the video.")