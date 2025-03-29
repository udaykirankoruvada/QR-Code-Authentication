# QR Code Authentication

This repository contains a QR code authentication system that utilizes both CNN and SVM models for classification. The project includes data preprocessing, feature engineering, model training, and prediction.

## Folder Structure
```
📂 QR-Code-Authentication
│-- 📂 model
│   │-- cnn_qr_model.keras      # Trained CNN model
│   │-- label_encoder.pkl       # Label encoder used for training
│
│-- 📝 data_Exploration.ipynb   # Jupyter notebook for data exploration
│-- 🏗️ data_preprocessing.py    # Script for preprocessing raw data
│-- 🛠️ featureEnginnering.py     # Feature extraction and engineering methods
│
│-- 🤖 cnn_train.py             # Trains the CNN model
│-- 🔍 cnn_predict.py           # Predicts using the CNN model
│
│-- 🏆 svm_train.py             # Trains the SVM model
│-- 🔍 svm_predict.py           # Predicts using the SVM model
│
│-- 📊 scaler.pkl               # Scaler used for normalizing features
│-- 🎯 svm_qr_model.pkl         # Trained SVM model
│
│-- 📄 documentation.docx       # Detailed project documentation
```

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/udaykirankoruvada/QR-Code-Authentication.git
   cd QR-Code-Authentication
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Train the CNN model:
   ```bash
   python cnn_train.py
   ```

4. Train the SVM model:
   ```bash
   python svm_train.py
   ```

5. Run predictions:
   ```bash
   python cnn_predict.py
   python svm_predict.py
   ```

6. Check documentation for more details:
   ```
   Open documentation.docx
   ```

---


## 📄 Project Documentation
[📑 View Documentation](./document_pdf.pdf)


