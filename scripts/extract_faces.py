import cv2
import os

# 1. HARDCODED ABSOLUTE PATHS (No more guessing)
BASE_DIR = "C:/Projects/data"
REAL_VIDEO = "C:/Projects/deepfake-detection-ai/data/real/abarnvbtwb.mp4"
FAKE_VIDEO = "C:/Projects/deepfake-detection-ai/data/fake/aagfhgtpmv.mp4"

# Create output folders
os.makedirs(f"{BASE_DIR}/processed_faces/real", exist_ok=True)
os.makedirs(f"{BASE_DIR}/processed_faces/fake", exist_ok=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract(v_path, out_path):
    print(f"Opening: {v_path}")
    cap = cv2.VideoCapture(v_path)
    
    if not cap.isOpened():
        print(f"FAILED to open: {v_path}")
        return

    count = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if count % 10 == 0: # Process every 10th frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                face = cv2.resize(face, (224, 224))
                
                # We use the full path to save
                file_name = f"face_{count}.jpg"
                full_save_path = os.path.join(out_path, file_name)
                
                success = cv2.imwrite(full_save_path, face)
                if success:
                    saved += 1
                else:
                    print(f"CRITICAL: Could not write to {full_save_path}")
        count += 1
    
    cap.release()
    print(f"Successfully saved {saved} images to {out_path}")

# Run it
extract(REAL_VIDEO, f"{BASE_DIR}/processed_faces/real")
extract(FAKE_VIDEO, f"{BASE_DIR}/processed_faces/fake")