import cv2
import numpy as np
import os
import re
import torch
import torchvision.models as models
import torchvision.transforms as transforms

# === Create folders ===
save_folder = "captured_images"
os.makedirs(save_folder, exist_ok=True)

# === Helper: Get next capture count ===
def get_next_capture_count(folder):
    existing_files = os.listdir(folder)
    numbers = []
    for f in existing_files:
        match = re.search(r'capture_(\d+)', f)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1

capture_count = get_next_capture_count(save_folder)

# === Edge detection ===
def detect_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return gray, edges

# === CNN Feature Extractor (VGG16) ===
cnn_model = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features.eval()

transform_pipeline = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def extract_cnn_features(image):
    img_tensor = transform_pipeline(image).unsqueeze(0)
    with torch.no_grad():
        features = cnn_model(img_tensor).squeeze().numpy()
    return features

# === Process Image ===
def process_image(image, save=False, capture_number=None):
    gray, edges = detect_edges(image)

    if save and capture_number is not None:
        # Save original
        cv2.imwrite(os.path.join(save_folder, f"capture_{capture_number}.png"), image)
        # Save grayscale
        cv2.imwrite(os.path.join(save_folder, f"capture_{capture_number}_grayscale.png"), gray)
        # Save edges
        cv2.imwrite(os.path.join(save_folder, f"capture_{capture_number}_edges.png"), edges)

        # Save CNN features
        features = extract_cnn_features(image)
        np.save(os.path.join(save_folder, f"capture_{capture_number}_cnn_features.npy"), features)

        print(f"✅ Saved capture {capture_number}")

# === CAMERA LOOP ===
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Force camera resolution to 1280x720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Create resizable window (preserves aspect ratio better)
cv2.namedWindow('Live Camera - SPACE: capture | L: load | ESC: exit', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Live Camera - SPACE: capture | L: load | ESC: exit', 1280, 720)

print("Press SPACE to capture and process.")
print("Press L to load and process a static image.")
print("Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow('Live Camera - SPACE: capture | L: load | ESC: exit', frame)
    key = cv2.waitKey(1)

    if key == 27:  # ESC
        print("Exiting...")
        break
    elif key == 32:  # SPACE
        print(f"📸 Image {capture_count} captured! Processing...")
        process_image(frame, save=True, capture_number=capture_count)
        capture_count += 1
    elif key in [ord('l'), ord('L')]:
        image_path = input("Enter path of image file: ").strip()
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            if image is not None:
                print("📂 Image loaded! Processing...")
                process_image(image, save=True, capture_number=capture_count)
                capture_count += 1
            else:
                print("Failed to load image!")
        else:
            print("File does not exist!")

cap.release()
cv2.destroyAllWindows()
