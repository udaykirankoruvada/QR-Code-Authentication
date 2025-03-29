import os
import numpy as np
import cv2
import tensorflow as tf
import joblib

# Load trained model and label encoder
model_dir = "model"
model_path = os.path.join(model_dir, "cnn_qr_model.keras")
encoder_path = os.path.join(model_dir, "label_encoder.pkl")

model = tf.keras.models.load_model(model_path)
label_encoder = joblib.load(encoder_path)

# Function to preprocess a single image
def preprocess_image(img_path, img_size=(128, 128)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Error: Cannot load image {img_path}")
    
    img = cv2.resize(img, img_size)
    img = img / 255.0  
    img = img.reshape(1, img_size[0], img_size[1], 1)
    return img

# Function to predict class of an image
def predict_image(img_path):
    img = preprocess_image(img_path)
    prediction = model.predict(img)
    class_idx = np.argmax(prediction)
    class_label = label_encoder.inverse_transform([class_idx])[0]
    
    print(f"Predicted Class: {class_label}")
    return class_label


image_path = "F://QR-Code-Authentication//Dataset//First Print//input_image_adjust (4).png"  
predict_image(image_path)
