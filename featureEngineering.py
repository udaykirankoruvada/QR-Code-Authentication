import cv2
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_images(image_dir, img_size=(128, 128)):
    features = []
    labels = []
    
    for label in os.listdir(image_dir):
        class_path = os.path.join(image_dir, label)
        if not os.path.isdir(class_path):
            continue
        
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            
            # Load Image in Grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            # Resize Image
            img = cv2.resize(img, img_size)
            
            # Apply Canny Edge Detection
            edges = cv2.Canny(img, threshold1=100, threshold2=200)
            
            # Flatten Edge Image into a Feature Vector
            features.append(edges.flatten())
            labels.append(label)
    
    
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    return np.array(features), np.array(labels)


image_directory = "F://QR-Code-Authentication//Dataset"  
X, y = load_and_preprocess_images(image_directory)
print("Feature Shape:", X.shape, "Labels Shape:", y.shape)
