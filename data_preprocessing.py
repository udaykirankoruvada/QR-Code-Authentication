import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Set image size & dataset path
IMG_SIZE = (128, 128)
DATASET_PATH = "F://QR-Code-Authentication//Dataset"  # Update the correct path

# Function to load images
def load_images_and_labels(dataset_path):
    images, labels = [], []
    label_map = {}  # To map class names to integers
    
    # Get all class folders
    for idx, label in enumerate(sorted(os.listdir(dataset_path))):
        class_path = os.path.join(dataset_path, label)
        if not os.path.isdir(class_path):
            continue

        label_map[label] = idx  # Assign an index to each label
        
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                continue
            
            img = cv2.resize(img, IMG_SIZE)  # Resize
            img = img / 255.0  # Normalize to [0,1]
            
            images.append(img)
            labels.append(idx)  # Store label index
    
    return np.array(images), np.array(labels), label_map

# Load dataset
X, y, label_map = load_images_and_labels(DATASET_PATH)

# Reshape X for CNN (Adding channel dimension)
X = X.reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1)  # (num_samples, height, width, channels)

# Convert labels to categorical (one-hot encoding)
y = to_categorical(y, num_classes=len(label_map))

# Split dataset (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print dataset summary
print(f"Dataset Loaded! {X_train.shape[0]} training images & {X_test.shape[0]} testing images.")
print(f"Image Shape: {X_train.shape[1:]} | Number of Classes: {len(label_map)}")
print("Label Mapping:", label_map)
