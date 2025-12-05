# Palm-Sense: AI-Powered Multimodal Assistive System for the Visually Impaired

Palm-Sense is a real-time intelligent assistive system designed to support visually impaired individuals by combining Computer Vision, Face Recognition, Object Detection, Distance Estimation, Vibration Feedback, and Voice Interaction into a single integrated platform.

This project uses OpenCV, YOLOv8 face_recognition, pyttsx3, and speech recognition to analyze the surrounding environment and verbally assist the user while also triggering tactile feedback.

---

## Features

### 1. Face Recognition
- Detects and identifies known individuals in real-time.
- Supports dynamic dataset expansion by registering new users instantly.
- Improves accuracy via YOLO-based face extraction during training.

### 2. Object Detection
- Real-time detection of 80+ object categories using YOLOv8.
- Announces detected objects through speech output.
- Uses a cooldown handler to avoid repeated announcements.

### 3. Distance Estimation
- Estimates how far the detected object or person is from the user.
- Converts distance → step count guidance for mobility assistance.
- Example: “A table is around 2.8 steps ahead.”

### 4. Voice Assistant
- Supports simple voice queries such as:
  - “Hello”
  - “What is nearby?”
  - “Stop”
-Responds naturally using text-to-speech synthesis.

### 5. Text-to-Speech Queue System
- Ensures non-overlapping and smooth voice output.
- Asynchronous processing using worker threads.

### 6. Real-Time Environment Interaction
- Simultaneously detects:
  - Objects  
  - Known faces  
  - Unknown faces  
- Fuses detections to deliver **context-aware guidance**.

### 7. Data Capture & Auto-Face Enrollment
- Automatically captures and stores faces of new users:
  - Press Y
  - Enter the name of the new user
  - The system saves image samples & encodes for recognition

### 8. Model Accuracy Evaluation
- Generates:
  - Confusion matrix
  - Classification report
  - Recognition accuracy analytics

