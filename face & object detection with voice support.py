import cv2
import pyttsx3
import threading
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import speech_recognition as sr
import face_recognition
from ultralytics import YOLO
import numpy as np
import os
import time

true_labels = []
pred_labels = []

# -------------------------------
# 1. Initialize Text-to-Speech
# -------------------------------
engine = pyttsx3.init(driverName='sapi5')
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

tts_lock = threading.Lock()
def speak(text):
    with tts_lock:
        engine.say(text)
        engine.runAndWait()

def speak_async(text):
    threading.Thread(target=speak, args=(text,), daemon=True).start()

# -------------------------------
# 2. Voice Command Recognition
# -------------------------------
def listen_command():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source, phrase_time_limit=5)
    try:
        command = r.recognize_google(audio).lower()
        print("You said:", command)
        return command
    except sr.UnknownValueError:
        speak_async("Sorry, I didn’t catch that.")
    except sr.RequestError:
        speak_async("Network error.")
    return ""

# -------------------------------
# 3. Load Known Faces
# -------------------------------
def load_known_faces(dataset_path=r"D:\Face_Re\known_faces"):
    known_encodings = []
    known_names = []
    
    for person in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person)
        if not os.path.isdir(person_path):
            continue
        
        for file in os.listdir(person_path):
            img_path = os.path.join(person_path, file)
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(person)
    
    return known_encodings, known_names

# -------------------------------
# 4. Distance Estimation
# -------------------------------
FOCAL_LENGTH = 700
KNOWN_WIDTHS = {
    "person": 40,
    "chair": 50,
    "bottle": 7,
    "cup": 8,
    "tv": 90,
    "cell phone": 8,
    "book": 20
}

def estimate_distance(label, box_width):
    if label not in KNOWN_WIDTHS:
        return None
    known_width = KNOWN_WIDTHS[label]
    distance_cm = (known_width * FOCAL_LENGTH) / box_width
    steps = distance_cm / 50
    return round(steps, 1)

# -------------------------------
# 5. Face + Object Detection
# -------------------------------
def recognize_faces_and_objects(known_encodings, known_names):

    # ----- FIX Camera Problem -----
    for cam_index in [0, 1, 2]:
        cap = cv2.VideoCapture(cam_index)
        if cap.isOpened():
            print(f"Camera opened on index {cam_index}")
            break
    else:
        print("❌ No camera detected!")
        return

    model = YOLO("yolov8n.pt")

    last_face_spoken = ""
    last_face_time = 0
    last_obj_spoken = ""
    last_obj_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera frame not captured.")
            break

        frame_messages = []

        # ----- FACE RECOGNITION -----
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, faces)

        for face_encoding, face_location in zip(encodings, faces):

            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                best_match_index = np.argmin(face_recognition.face_distance(known_encodings, face_encoding))
                name = known_names[best_match_index]

                if name != last_face_spoken or time.time() - last_face_time > 3:
                    frame_messages.append(f"{name} is in front of you")
                    last_face_spoken = name
                    last_face_time = time.time()

            # Draw box
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # For accuracy calculation
            pred_labels.append(name)
            true_labels.append("Shraddha")

        # ----- OBJECT DETECTION -----
        results = model.predict(frame, verbose=False)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = box.conf[0]
                width = x2 - x1
                label = model.names[cls]

                distance_steps = estimate_distance(label, width)

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                text = f"{label} {conf:.2f}"
                if distance_steps:
                    text += f" ({distance_steps} steps)"
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                if label != last_obj_spoken or time.time() - last_obj_time > 3:
                    if distance_steps:
                        frame_messages.append(f"I see a {label} about {distance_steps} steps ahead")
                    else:
                        frame_messages.append(f"I see a {label}")
                    last_obj_time = time.time()
                    last_obj_spoken = label

        if frame_messages:
            speak_async(". ".join(frame_messages))

        cv2.imshow("Face + Object + Distance Detection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('v'):
            command = listen_command()
            if "your name" in command:
                speak_async("I am your assistant Shraddha.")
            elif "hello" in command:
                speak_async("Hello, how can I help you?")
            elif "stop" in command:
                speak_async("Goodbye")
                break

        if key == ord('q'):
            speak_async("Camera closed")
            break

    cap.release()
    cv2.destroyAllWindows()

# -------------------------------
# 6. Start
# -------------------------------
speak_async("Hi Shraddha, I am your assistant.")
known_encodings, known_names = load_known_faces()
recognize_faces_and_objects(known_encodings, known_names)

# -------------------------------
# 7. Final Accuracy Report
# -------------------------------
if len(true_labels) == 0 or len(pred_labels) == 0:
    print("⚠ No face recognition data collected. Cannot compute accuracy.")
else:
    accuracy = accuracy_score(true_labels, pred_labels)
    cm = confusion_matrix(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels)

    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)
