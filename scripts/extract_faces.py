import cv2
import os
import glob

# Correct folder path
BASE_DIR = "C:/Projects/deepfake-detection-ai/data"
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def process_folder(label):
    video_files = glob.glob(f"{BASE_DIR}/{label}/*.mp4")
    output_path = f"{BASE_DIR}/processed_faces/{label}"
    os.makedirs(output_path, exist_ok=True)

    print(f"Starting extraction for: {label}")
    for v_path in video_files:
        cap = cv2.VideoCapture(v_path)
        count = 0
        saved_per_video = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Process every 20th frame (to avoid having too many identical images)
            if count % 20 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    face = cv2.resize(frame[y:y+h, x:x+w], (224, 224))
                    # Save with unique name: [video_name]_f[frame_number].jpg
                    file_name = f"{os.path.basename(v_path)}_f{count}.jpg"
                    cv2.imwrite(os.path.join(output_path, file_name), face)
                    saved_per_video += 1
            count += 1
        cap.release()
        print(f"Processed {os.path.basename(v_path)}: Saved {saved_per_video} faces.")

# Clean old data first so we don't mix old/new
import shutil
if os.path.exists(f"{BASE_DIR}/processed_faces"):
    shutil.rmtree(f"{BASE_DIR}/processed_faces")

process_folder("real")
process_folder("fake")