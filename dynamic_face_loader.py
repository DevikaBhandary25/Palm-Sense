#unknown                                                                                                                               import cv2
import pyttsx3
import threading
import face_recognition
from ultralytics import YOLO
import numpy as np
import os
import time
import queue

# --------------------------
# 1. TEXT TO SPEECH THREAD
# --------------------------

speech_queue = queue.Queue()

def tts_worker():
    engine = pyttsx3.init("sapi5")
    engine.setProperty("rate", 150)

    while True:
        text = speech_queue.get()
        if text == "SHUTDOWN":
            break

        print("Speaking:", text)
        engine.say(text)
        engine.runAndWait()

tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

def speak(text):
    speech_queue.put(text)


# --------------------------
# 2. LOAD KNOWN FACES (SAFE USING YOLO)
# --------------------------

known_faces = []
known_names = []

KNOWN_DIR = "known_faces"

yolo_loader = YOLO("yolov8n.pt")   # using YOLO for face detection in stored images

print("\nLoading known faces...\n")

for name in os.listdir(KNOWN_DIR):
    person_dir = os.path.join(KNOWN_DIR, name)
    if not os.path.isdir(person_dir):
        continue

    for filename in os.listdir(person_dir):
        path = os.path.join(person_dir, filename)
        img = cv2.imread(path)

        if img is None:
            print(f"⚠ Failed to load: {path}")
            continue

        results = yolo_loader(img, stream=False)

        face_found = False

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])

                # YOLO class 0 = person
                if cls == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    face = img[y1:y2, x1:x2]

                    if face.size == 0:
                        continue

                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    encs = face_recognition.face_encodings(face_rgb)

                    if len(encs) > 0:
                        known_faces.append(encs[0])
                        known_names.append(name)
                        face_found = True
                        print(f"Loaded: {name} from {filename}")
                        break

            if face_found:
                break

        if not face_found:
            print(f"⚠ No face detected in {path}, skipping...")

print("\nKnown faces loaded.\n")


# --------------------------
# 3. LOAD YOLO MODEL FOR CAMERA
# --------------------------

model = YOLO("yolov8n.pt")

# --------------------------
# 4. CAMERA LOOP
# --------------------------

video = cv2.VideoCapture(0)
last_spoken = ""
last_time = 0

while True:
    ret, frame = video.read()
    if not ret:
        break

    results = model(frame, stream=True)

    face_locations = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_locations.append((y1, x2, y2, x1))

    if len(face_locations) > 0:

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(rgb, face_locations)

        for enc, loc in zip(encodes, face_locations):
            matches = face_recognition.compare_faces(known_faces, enc)
            name = "Unknown"

            if True in matches:
                idx = matches.index(True)
                name = known_names[idx]

            # Speak if new name OR 3 seconds passed
            if name != last_spoken or time.time() - last_time > 3:

                if name == "Unknown":
                    speak("Unknown person detected. Press Y to add this person.")
                    print("\nUnknown person detected!")
                    print("Press Y to ADD")
                    print("Press N to IGNORE")

                    key = cv2.waitKey(5000) & 0xFF

                    if key == ord('y'):
                        speak("Please enter the name to save this person.")
                        new_name = input("Enter name for this person: ")

                        save_dir = os.path.join(KNOWN_DIR, new_name)
                        os.makedirs(save_dir, exist_ok=True)

                        filename = f"{int(time.time())}.jpg"
                        filepath = os.path.join(save_dir, filename)
                        cv2.imwrite(filepath, frame)

                        known_faces.append(enc)
                        known_names.append(new_name)

                        speak(f"{new_name} has been added successfully.")
                        print(f"{new_name} added!")

                        # Force next loop to speak this new name
                        last_spoken = ""
                        last_time = 0

                else:
                    speak(f"{name} is in front of you.")

                last_spoken = name
                last_time = time.time()

    # Draw rectangles
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# --------------------------
# 5. CLEAN EXIT
# --------------------------

speech_queue.put("SHUTDOWN")
video.release()
cv2.destroyAllWindows()
