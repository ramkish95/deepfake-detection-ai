# deepfake-detection-ai
An AI-powered digital forensics tool to detect manipulated videos using Computer Vision.
# AI-Powered Deepfake Detector (Digital Forensics)

## 📌 Project Overview
This project is a 3rd-year AIML engineering experiment focused on **Cybersecurity** and **Digital Forensics**. It uses Computer Vision to detect facial "artifacts" (digital inconsistencies) in videos to determine if they are authentic or AI-generated deepfakes.

## 🛡️ Cybersecurity Concepts Applied
* **Digital Forensics:** Isolating "suspect" data (faces) from a crime scene (video) for analysis.
* **Artifact Detection:** Identifying pixel-level glitches left behind by GANs (Generative Adversarial Networks).
* **Integrity Verification:** Using AI to verify the authenticity of digital media.
* **False Negatives/Positives:** Tuning the model to reduce the risk of letting a fake video pass as real.

## 🤖 AI Architecture
* **Backbone:** MobileNetV2 (Transfer Learning)
* **Detection Sensor:** OpenCV Haar Cascades for real-time face tracking.
* **Preprocessing:** BGR-to-RGB conversion, Feature Scaling (1./255), and Image Normalization.

## 📂 Project Structure
```text
deepfake-detection-ai/
├── data/
│   ├── real/             # Authentic video samples
│   ├── fake/             # Deepfake video samples
│   └── processed_faces/  # Cropped facial evidence
├── models/
│   └── deepfake_detector.h5  # Trained AI brain
├── scripts/
│   ├── extract_faces.py  # Forensic triage & face cropping
│   ├── train_detector.py # AI model training
│   └── scan_video.py     # Inference & scan reporting
└── README.md