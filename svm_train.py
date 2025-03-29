import cv2
import numpy as np
import os
import joblib
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import collections

# Load & preprocess images
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
                print(f"Warning: Could not read {img_path}")
                continue
            
            # Resize Image
            img = cv2.resize(img, img_size)
            
            # Apply Canny Edge Detection
            edges = cv2.Canny(img, threshold1=100, threshold2=200).flatten()
            
            # Extract HOG Features
            hog_features = hog(img, orientations=9, pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2), feature_vector=True)
            
            # Combine Features
            combined_features = np.hstack((edges, hog_features))
            
            features.append(combined_features)
            labels.append(label)
    
    return np.array(features), np.array(labels)


image_directory = "F://QR-Code-Authentication//Dataset"  
X, y = load_and_preprocess_images(image_directory)

print("Data Loaded Successfully!")
print("Feature Shape:", X.shape, "Labels Shape:", y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Checking for Duplicate Samples Between Train and Test

train_set = {tuple(x) for x in X_train}
test_set = {tuple(x) for x in X_test}

duplicates = train_set.intersection(test_set)
if duplicates:
    print(f"{len(duplicates)} duplicate samples found in both train & test sets! Possible data leakage.")
else:
    print("No duplicate samples found in train and test sets.")

# Check Class Distribution
train_dist = collections.Counter(y_train)
test_dist = collections.Counter(y_test)

print(f"Train Label Distribution: {train_dist}")
print(f"Test Label Distribution: {test_dist}")


min_class_samples = min(train_dist.values())
if min_class_samples < 5:
    print("Too few samples for some classes! Reducing cv to 3.")
    cv_folds = 3
else:
    cv_folds = 5

# Normalize Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  
X_test = scaler.transform(X_test) 

# Hyperparameter Tuning using GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.01, 0.001]
}

grid_search = GridSearchCV(SVC(), param_grid, cv=cv_folds)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Best Parameters Found:", grid_search.best_params_)

# Stratified K-Fold Cross-Validation
kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X_train, y_train, cv=kf, scoring='accuracy')

print(f"Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# Train SVM Classifier with Best Parameters
svm_model = best_model
svm_model.fit(X_train, y_train)

# Save the model and scaler
joblib.dump(svm_model, "svm_qr_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model and Scaler saved successfully!")

# Evaluate Model on Test Data
y_pred = svm_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Classification Report & Confusion Matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=set(y_test), yticklabels=set(y_test))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Identify Misclassified Samples
misclassified_indices = np.where(y_pred != y_test)[0]
print(f"Number of Misclassified Samples: {len(misclassified_indices)}")

for i in misclassified_indices[:5]:  # Show first 5 misclassifications
    print(f"Misclassified Image: {i}, True Label: {y_test[i]}, Predicted: {y_pred[i]}")
