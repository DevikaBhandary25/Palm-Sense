# vibration_code_fixed.py
import cv2
import numpy as np
import socket
import os
import re
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import time

# ====== CONFIG ======
ESP_IP = "10.162.124.105"   # <- put your ESP32 IP here (from Serial Monitor)
ESP_PORT = 8080
SAVE_FOLDER = "captured_images"
os.makedirs(SAVE_FOLDER, exist_ok=True)

# Debug visualization (set False to disable extra windows)
SHOW_DEBUG_WINDOWS = True

# ====== NETWORK: send to ESP32 via TCP ======
def send_pattern(pattern):
    """Send vibration pattern (9 chars '0'/'1') to ESP32"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2.0)
        s.connect((ESP_IP, ESP_PORT))
        s.sendall((pattern + "\n").encode())
        s.close()
        print(f"✅ Sent pattern: {pattern}")
    except Exception as e:
        print(f"⚠️ Failed to send pattern: {e}")

# ====== file helper ======
def get_next_capture_count(folder):
    existing_files = os.listdir(folder)
    numbers = []
    for f in existing_files:
        match = re.search(r'capture_(\d+)', f)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1

capture_count = get_next_capture_count(SAVE_FOLDER)

# ====== edge detection and helpers ======
def detect_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Optional - smooth slightly to reduce noise
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    edges = cv2.Canny(gray, 50, 150)
    return gray, edges

def crop_to_main_shape(edges):
    """Return cropped edges image containing the largest external contour.
       If none found, return original edges image."""
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return edges
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    # Some padding (keep it within image bounds)
    pad = max(2, int(0.05 * max(w, h)))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(edges.shape[1], x + w + pad)
    y1 = min(edges.shape[0], y + h + pad)
    cropped = edges[y0:y1, x0:x1]
    return cropped

def edges_to_pattern(edges, downsize=(3,3), debug=False):
    """Convert edges image (focus on the main shape) -> binary downsample -> '010101010' string"""
    # Dilate slightly so thin strokes survive resizing
    kernel = np.ones((3,3), np.uint8)
    edges_d = cv2.dilate(edges, kernel, iterations=1)

    h, w = edges_d.shape
    if h == 0 or w == 0:
        # fallback
        small = np.zeros(downsize, dtype=np.uint8)
    else:
        # Resize content to downsize using INTER_AREA (better for shrinking)
        small = cv2.resize(edges_d, downsize, interpolation=cv2.INTER_AREA)

    # dynamic threshold: pick a threshold relative to small mean
    thr = max(10, int(np.mean(small) * 0.6))
    _, binary = cv2.threshold(small, thr, 1, cv2.THRESH_BINARY)

    # Flatten row-major and create string
    pattern = ''.join(str(int(x)) for x in binary.flatten())

    if debug:
        print("small (resized):")
        print(small)
        print("binary (3x3):")
        print(binary)
        print(f"threshold used: {thr}")
        print(f"pattern -> {pattern}")

    return pattern, binary

# ====== optional CNN features function (kept for analysis, not used for mapping) ======
cnn_model = None
try:
    cnn_model = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features.eval()
    transform_pipeline = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
except Exception as e:
    print("Warning: torchvision model load failed or not needed:", e)
    cnn_model = None

def extract_cnn_features(image):
    if cnn_model is None:
        return None
    img_tensor = transform_pipeline(image).unsqueeze(0)
    with torch.no_grad():
        features = cnn_model(img_tensor).squeeze().numpy()
    return features

# ====== Process image: crop, edge -> pattern -> send ======
def process_image(image, save=False, capture_number=None, visualize_debug=False):
    gray, edges = detect_edges(image)

    # Crop to main shape before downsampling
    cropped = crop_to_main_shape(edges)

    # If cropped region is tiny, fall back to resizing original edges
    if cropped.size == 0 or cropped.shape[0] < 3 or cropped.shape[1] < 3:
        cropped = edges

    pattern, binary = edges_to_pattern(cropped, downsize=(3,3), debug=visualize_debug)

    # Save files if requested
    if save and capture_number is not None:
        cv2.imwrite(os.path.join(SAVE_FOLDER, f"capture_{capture_number}.png"), image)
        cv2.imwrite(os.path.join(SAVE_FOLDER, f"capture_{capture_number}_grayscale.png"), gray)
        cv2.imwrite(os.path.join(SAVE_FOLDER, f"capture_{capture_number}_edges.png"), edges)
        cv2.imwrite(os.path.join(SAVE_FOLDER, f"capture_{capture_number}_cropped_edges.png"), cropped)
        # save 3x3 visual for quick glance (upsampled)
        up = cv2.resize((binary*255).astype(np.uint8), (300,300), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(SAVE_FOLDER, f"capture_{capture_number}_pattern_visual.png"), up)
        if cnn_model is not None:
            features = extract_cnn_features(image)
            np.save(os.path.join(SAVE_FOLDER, f"capture_{capture_number}_cnn_features.npy"), features)
        print(f"✅ Saved capture {capture_number}")

    # Visual debug windows (optional)
    if SHOW_DEBUG_WINDOWS or visualize_debug:
        cv2.imshow("Gray", gray)
        cv2.imshow("Edges (full)", edges)
        cv2.imshow("Cropped edges", cropped)
        cv2.imshow("3x3 pattern (upsampled)", cv2.resize((binary*255).astype(np.uint8), (300,300), interpolation=cv2.INTER_NEAREST))
        cv2.waitKey(1)

    # Make sure pattern is exactly 9 characters (3x3)
    if len(pattern) != 9:
        print("⚠️ Unexpected pattern length:", len(pattern))
        # pad or trim to 9
        if len(pattern) < 9:
            pattern = pattern.ljust(9, "0")
        else:
            pattern = pattern[:9]

    # Send to ESP32
    send_pattern(pattern)

# ====== CAMERA LOOP ======
def main_loop():
    global capture_count
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # Set recommended resolution (not required)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cv2.namedWindow('Live', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Live', 800, 450)

    print("Press SPACE to capture and process (live).")
    print("Press L to load and process a static image.")
    print("Press ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        cv2.imshow('Live', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            print("Exiting...")
            break
        elif key == 32:  # SPACE
            print(f"📸 Image {capture_count} captured! Processing...")
            process_image(frame, save=True, capture_number=capture_count, visualize_debug=False)
            capture_count += 1
        elif key in [ord('l'), ord('L')]:
            image_path = input("Enter path of image file: ").strip().strip('"')
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                if image is not None:
                    print("📂 Image loaded! Processing...")
                    process_image(image, save=True, capture_number=capture_count, visualize_debug=True)
                    capture_count += 1
                else:
                    print("Failed to load image!")
            else:
                print("File does not exist!")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_loop()

