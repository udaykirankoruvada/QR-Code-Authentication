import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import cv2
import joblib

# Load & preprocess images
def load_images(image_dir, img_size=(128, 128)):
    images, labels = [], []
    
    for label in os.listdir(image_dir):
        class_path = os.path.join(image_dir, label)
        if not os.path.isdir(class_path):
            continue
        
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            img = cv2.resize(img, img_size)
            img = img / 255.0  # Normalize to [0,1]
            images.append(img)
            labels.append(label)
    
    return np.array(images), np.array(labels)

# Load dataset
data_dir = "F://QR-Code-Authentication//Dataset"  
X, y = load_images(data_dir)
X = X.reshape(-1, 128, 128, 1)  

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data Augmentation
datagen = ImageDataGenerator(rotation_range=20, zoom_range=0.2, horizontal_flip=True)

# Define CNN Model
def create_cnn_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.4),  
        layers.Dense(len(np.unique(y)), activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train CNN
model = create_cnn_model()
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    validation_data=(X_test, y_test), epochs=100, verbose=1)

# Ensure model directory exists
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)

# Save model & encoder
model.save(os.path.join(model_dir, "cnn_qr_model.keras"))
joblib.dump(label_encoder, os.path.join(model_dir, "label_encoder.pkl"))

print("Model and label encoder saved!")

# Evaluate model
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.4f}")

# Predictions & Classification Report
y_pred = np.argmax(model.predict(X_test), axis=1)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
