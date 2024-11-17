# Basic Libraries
import subprocess

# Uninstall problematic packages
subprocess.run(["pip", "uninstall", "-y", "jax", "jaxlib"], check=True)

import numpy as np
import pandas as pd
import os
import requests

# Image Processing and Visualization
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import random
# Deep Learning Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Metrics and Evaluation
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score


from tensorflow.keras.preprocessing.image import ImageDataGenerator


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image

import streamlit as st

import os
import subprocess

def uninstall_packages(packages):
    for package in packages:
        try:
            subprocess.run(
                ["pip", "uninstall", package, "-y"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            print(f"Uninstalled {package}")
        except subprocess.CalledProcessError as e:
            print(f"Error uninstalling {package}: {e}")

# List of packages to uninstall
packages_to_remove = ["jax", "jaxlib"]

if __name__ == "__main__":
    uninstall_packages(packages_to_remove)


# Load the pre-trained model
@st.cache_resource  # Cache the model for faster access

@st.cache_resource
def load_model():
    model_url = st.secrets["model"]["url"]
    local_model_path = "pneumonia_detection_model.keras"

    # Download the model if not already present
    if not os.path.exists(local_model_path):
        response = requests.get(model_url)
        with open(local_model_path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded model saved to {local_model_path}")
    
    print(f"Model path exists: {os.path.exists(local_model_path)}")
    print(f"Downloaded file size: {os.path.getsize(local_model_path)} bytes")
    
    # Load the model
    return tf.keras.models.load_model(local_model_path)
model = load_model()

# Function to preprocess and predict the uploaded image
def predict_image(img):
    # Ensure the image is in RGB format
    img = img.convert("RGB")
    img = img.resize((224, 224))  # Resize to match model input size
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make a prediction
    prediction = model.predict(img_array)
    class_label = "PNEUMONIA" if prediction[0][0] > 0.5 else "NORMAL"
    confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
    return class_label, confidence

# Streamlit interface
st.title("Pneumonia Detection from Chest X-rays")
st.write("Upload a chest X-ray image to classify it as NORMAL or PNEUMONIA.")

# Upload image
uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Predict and display result
    with st.spinner("Classifying..."):
        class_label, confidence = predict_image(img)
    st.success(f"Prediction: {class_label}")
    st.write(f"Confidence: {confidence * 100:.2f}%")
