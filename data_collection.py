import cv2
import os
import mediapipe as mp
import time

# Set up MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Labels in Nepali
labels = [
    "क", "ख", "ग", "घ", "ङ", "च", "छ", "ज", "झ", "ञ",
    "ट", "ठ", "ड", "ढ", "ण", "त", "थ", "द", "ध", "न",
    "प", "फ", "ब", "भ", "म", "य", "र", "ल", "व", "श",
    "ष", "स", "ह", "क्ष", "त्र", "ज्ञ",
    "आइतबार", "सोमबार", "मङ्गलबार", "बुधबार", "बिहिबार", "शुक्रबार", "शनिबार"
]

# Data directory
base_path = 'C:/Users/Prajwol/Desktop/Sign-Language-detection-main/Sign-Language-detection-main/data'

# Create directories
if not os.path.exists(base_path):
    os.makedirs(base_path)

for label in labels:
    label_path = os.path.join(base_path, label)
    if not os.path.exists(label_path):
        os.makedirs(label_path)

# Function to capture images
def capture_images(label, num_images=300):
    cap = cv2.VideoCapture(0)
    count = 0

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect hands
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]

            # Get the bounding box coordinates for the hands
            h, w, _ = frame.shape
            x_min = min([landmark.x for landmark in hand_landmarks.landmark]) * w
            x_max = max([landmark.x for landmark in hand_landmarks.landmark]) * w
            y_min = min([landmark.y for landmark in hand_landmarks.landmark]) * h
            y_max = max([landmark.y for landmark in hand_landmarks.landmark]) * h

            # Expand the bounding box slightly to capture the whole hand
            margin = 10
            x_min = int(max(x_min - margin, 0))
            x_max = int(min(x_max + margin, w))
            y_min = int(max(y_min - margin, 0))
            y_max = int(min(y_max + margin, h))

            # Crop the frame to the bounding box
            cropped_frame = frame[y_min:y_max, x_min:x_max]

            # Save the cropped frame as an image file
            img_name = os.path.join(base_path, label, f"{label}_{count:04}.png")
            cv2.imwrite(img_name, cropped_frame)

            count += 1

        # Wait for 1 second before capturing the next frame
        time.sleep(1)

    cap.release()
    cv2.destroyAllWindows()

# Capture images for each label sequentially
for label in labels:
    print(f"Capturing images for {label}")
    capture_images(label)
    print(f"Completed capturing images for {label}")

print("Data collection complete!")
