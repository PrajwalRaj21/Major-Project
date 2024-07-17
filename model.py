import os
import numpy as np
import tensorflow as tf
from transformers import TFViTForImageClassification, ViTFeatureExtractor, create_optimizer
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# Directory containing your preprocessed data
data_dir = r"C:\Users\Prajwol\Downloads\Sign-Language-detection-main\Sign-Language-detection-main\Data"

# Parameters
img_height, img_width = 224, 224  # Dimensions for ViT input
batch_size = 16  # Increased batch size for better GPU utilization
epochs = 20  # Increased epochs for better training
validation_split = 0.1

# Load and preprocess data
def load_data(data_dir, img_height, img_width):
    X = []
    y = []
    labels = os.listdir(data_dir)
    label_map = {label: num for num, label in enumerate(labels)}

    for label in labels:
        class_dir = os.path.join(data_dir, label)
        if os.path.isdir(class_dir):
            frames = sorted(os.listdir(class_dir))
            for frame in frames:
                img_path = os.path.join(class_dir, frame)
                img = load_img(img_path, target_size=(img_height, img_width))
                img_array = img_to_array(img)
                X.append(img_array)
                y.append(label_map[label])
        else:
            print(f"Skipping {class_dir}, not a directory.")

    X = np.array(X)
    num_classes = len(labels)
    y = tf.keras.utils.to_categorical(y, num_classes=num_classes)
    
    print(f"Loaded {len(X)} samples from {num_classes} classes.")
    return X, y, num_classes

X, y, num_classes = load_data(data_dir, img_height, img_width)

# Check if data is loaded
if len(X) == 0 or len(y) == 0:
    raise ValueError("No data loaded. Please check the data directory and preprocessing steps.")

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42)

# Normalize the data
X_train = X_train / 255.0
X_val = X_val / 255.0

# Load ViT feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
feature_extractor.do_rescale = False  # Avoid rescaling already rescaled images

# Preprocess the data for ViT
def preprocess_images(images, feature_extractor):
    images = [image for image in images]
    encodings = feature_extractor(images=images, return_tensors='tf')
    return encodings['pixel_values']

X_train_encodings = preprocess_images(X_train, feature_extractor)
X_val_encodings = preprocess_images(X_val, feature_extractor)

# Define the ViT model
vit_model = TFViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=num_classes)

# Create optimizer
optimizer, lr_schedule = create_optimizer(
    init_lr=0.001, 
    num_train_steps=len(X_train_encodings) // batch_size * epochs, 
    num_warmup_steps=0
)

# Compile the model
vit_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
vit_model.fit(X_train_encodings, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val_encodings, y_val))

# Save the model
vit_model.save_pretrained(r"C:\Users\Prajwol\Desktop\Sign-Language-detection-main\Sign-Language-detection-main\sign_language_vit_model")
