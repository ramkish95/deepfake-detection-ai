import cv2
import numpy as np
import tensorflow as tf
import os

# 1. Load the new 'Advanced' Brain
model_path = '../models/deepfake_detector.h5'
model = tf.keras.models.load_model(model_path)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def scan_video(video_path):
    cap = cv2.VideoCapture(video_path)
    found_face = False

    # We scan up to 100 frames to find the clearest face
    for _ in range(100):
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face = frame[y:y+h, x:x+w]
            
            # Forensic Pre-processing
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = face.astype('float32') / 255.0 
            face = np.expand_dims(face, axis=0)

            prediction = model.predict(face, verbose=0)[0][0]
            
            print(f"\n--- FORENSIC REPORT: {os.path.basename(video_path)} ---")
            
            # Label 0 = Fake, Label 1 = Real (Alphabetical)
            if prediction < 0.5:
                confidence = (1 - prediction) * 100
                print(f"STATUS: [!] DEEPFAKE DETECTED")
                print(f"CYBER ANALYSIS: Artifacts found with {confidence:.2f}% confidence.")
            else:
                confidence = prediction * 100
                print(f"STATUS: [✓] AUTHENTIC VIDEO")
                print(f"CYBER ANALYSIS: Natural patterns verified with {confidence:.2f}% confidence.")
            
            print("-------------------------------------------\n")
            found_face = True
            break 

    if not found_face:
        print(f"ALERT: Scan failed for {os.path.basename(video_path)} - No face detected.")
    cap.release()

# TEST PATHS
base_path = "C:/Projects/deepfake-detection-ai/data"
scan_video(f"{base_path}/fake/aettqgevhz.mp4") # New Fake from your list
scan_video(f"{base_path}/real/asaxgevnnp.mp4") # New Real from your list