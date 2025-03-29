import cv2
import numpy as np
import joblib
from skimage.feature import hog

# Load the trained model and scaler
svm_model = joblib.load("svm_qr_model.pkl")
scaler = joblib.load("scaler.pkl")

# Function to preprocess a test image
def preprocess_image(image_path, expected_size=(128, 128)):  
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")
    
    # Resize image to (128, 128) to match training data
    image = cv2.resize(image, expected_size)
    
    # Apply Canny Edge Detection
    edges = cv2.Canny(image, threshold1=100, threshold2=200).flatten()
    
    # Extract HOG Features
    hog_features = hog(image, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), feature_vector=True)
    
    # Combine Features
    combined_features = np.hstack((edges, hog_features))

    # Ensure the feature vector matches the trained model
    if combined_features.shape[0] != scaler.n_features_in_:
        raise ValueError(f"Feature size mismatch! Expected {scaler.n_features_in_}, but got {combined_features.shape[0]}.")

    # Normalize Features
    combined_features = scaler.transform([combined_features])
    
    return combined_features

# Function to predict the QR code class
def predict_qr_code(image_path):
    try:
        features = preprocess_image(image_path)
        prediction = svm_model.predict(features)
        print(f"Predicted QR Code Class: {prediction[0]}")
    except Exception as e:
        print(f"Error: {e}")

# Test Image Path 

test_image = "F://QR-Code-Authentication//Dataset//Second Print//input_image_assume (2).png"  
predict_qr_code(test_image)
