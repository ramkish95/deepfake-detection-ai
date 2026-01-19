import cv2
import numpy as np
import tensorflow as tf
import os

# 1. LOAD THE TRAINED MODEL
# In Cybersecurity, we treat this as our 'Forensic Signature Database'
model_path = '../models/deepfake_detector.h5'

if not os.path.exists(model_path):
    print(f"ERROR: Model file not found at {model_path}. Please train the model first.")
    exit()

model = tf.keras.models.load_model(model_path)

# Re-compiling ensures all metrics are ready for evaluation
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 2. INITIALIZE DETECTOR
# Using the Haar Cascade sensor to find the 'Suspect' (the face)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def scan_video(video_path):
    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    found_face = False

    # CYBERSECURITY STRATEGY: 
    # We scan the first 60 frames (approx 2 seconds) to find a clear face.
    # This prevents 'False Negatives' caused by blurry opening frames.
    for _ in range(60):
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            # Isolate the primary face
            (x, y, w, h) = faces[0]
            face = frame[y:y+h, x:x+w]
            
            # AI PRE-PROCESSING:
            # 1. Convert BGR (OpenCV default) to RGB (AI standard)
            # 2. Resize to 224x224 (MobileNetV2 standard)
            # 3. Normalize pixels to 0-1 range
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = face.astype('float32') / 255.0 
            face = np.expand_dims(face, axis=0) # Add batch dimension

            # THE INFERENCE PHASE
            prediction = model.predict(face, verbose=0)[0][0]
            
            print(f"\n--- SCAN REPORT: {os.path.basename(video_path)} ---")
            
            # MAPPING LOGIC (Keras Alphabetical Rule):
            # Folder 'fake' = Class 0
            # Folder 'real' = Class 1
            if prediction < 0.5:
                # Value near 0 means AI identified 'fake' characteristics
                confidence = (1 - prediction) * 100
                print(f"RESULT: [!] FAKE / MANIPULATED")
                print(f"AI ANALYSIS: Digital artifacts detected (Confidence: {confidence:.2f}%)")
            else:
                # Value near 1 means AI identified 'real' characteristics
                confidence = prediction * 100
                print(f"RESULT: [✓] AUTHENTIC / REAL")
                print(f"AI ANALYSIS: Natural pixel patterns found (Confidence: {confidence:.2f}%)")
            
            print("-------------------------------------------\n")
            found_face = True
            break 

    if not found_face:
        print(f"ALERT: No human face detected in {os.path.basename(video_path)}. Forensic scan failed.")
    
    cap.release()

# 3. EXECUTE FORENSIC SCAN
# Replace these paths with your actual video locations
base_path = "C:/Projects/deepfake-detection-ai/data"

print("Initializing Forensic Scanner...")
scan_video(f"{base_path}/fake/aagfhgtpmv.mp4")
scan_video(f"{base_path}/real/abarnvbtwb.mp4")