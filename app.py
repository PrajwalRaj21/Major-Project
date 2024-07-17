import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import tensorflow as tf
from transformers import ViTFeatureExtractor, TFViTForImageClassification
from PIL import Image, ImageDraw, ImageFont

# Load your ViT model
model_path = r"C:\Users\Prajwol\Desktop\Sign-Language-detection-main\Sign-Language-detection-main\sign_language_vit_model"
model = TFViTForImageClassification.from_pretrained(model_path)

# Initialize OpenCV variables
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)  # Instantiate HandDetector
offset = 20
imgSize = 224  # Adjusted image size for ViT model input
labels = ["क", "ख", "ग", "घ", "ङ"]

# Load the Devanagari font
font_path = r"C:\Users\Prajwol\Downloads\Noto_Sans_Devanagari\NotoSansDevanagari-VariableFont_wdth,wght.ttf"
font = ImageFont.truetype(font_path, 32)

# Initialize the feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

def preprocess_image(img):
    """Preprocess the image for ViT model."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    img_resized = img_pil.resize((imgSize, imgSize))
    img_array = np.array(img_resized) / 255.0  # Scale pixel values between 0 and 1
    return img_array

def put_text_pil(image, text, position, font, color):
    """Draw text on an image using PIL."""
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

while True:
    success, img = cap.read()
    imgOutput = img.copy()

    # Detect hands using cvzone's HandDetector
    hands, img = detector.findHands(img)  # Use detector here
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]
        imgCrop = cv2.resize(imgCrop, (imgSize, imgSize))  # Resize image for ViT input

        # Preprocess image
        img_preprocessed = preprocess_image(imgCrop)
        img_preprocessed = feature_extractor(images=img_preprocessed, return_tensors="tf")['pixel_values']

        # Predict with your ViT model
        predictions = model(img_preprocessed)
        logits = predictions.logits
        index = tf.argmax(logits, axis=-1).numpy()[0]  # Get the index of the highest probability class

        # Display prediction result on the output image using PIL
        imgOutput = put_text_pil(imgOutput, labels[index], (x, y-30), font, (0, 0, 0))

        # Draw bounding box
        cv2.rectangle(imgOutput, (x-offset, y-offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)   

    cv2.imshow('Image', imgOutput)
    cv2.waitKey(1)
