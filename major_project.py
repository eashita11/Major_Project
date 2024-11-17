import os
import kagglehub
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

# Load the pre-trained model from KaggleHub
@st.cache_resource
def load_model_from_kagglehub():
    # Download model using KaggleHub
    path = kagglehub.model_download("eashitadhillon/pneumonia_detection_model/keras/default")
    st.write(f"Model downloaded to: {path}")

    # Find the .h5 file in the downloaded directory
    model_file = None
    for file in os.listdir(path):
        if file.endswith(".h5"):  # Look for .h5 file
            model_file = os.path.join(path, file)
            break

    if model_file is None:
        st.error("No .h5 file found in the downloaded path.")
        raise FileNotFoundError("No .h5 model file found.")

    # Load the .h5 model
    try:
        model = load_model(model_file)
        st.write("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        raise

# Function to preprocess and predict the uploaded image
def predict_image(img, model):
    img = img.convert("RGB")  # Ensure the image is in RGB format
    img = img.resize((224, 224))  # Resize to match model input size
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make a prediction
    prediction = model.predict(img_array)
    class_label = "PNEUMONIA" if prediction[0][0] > 0.5 else "NORMAL"
    confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
    return class_label, confidence

# Main Streamlit app
st.title("Pneumonia Detection from Chest X-rays")
st.write("Upload a chest X-ray image to classify it as NORMAL or PNEUMONIA.")

# Load the model
model = load_model_from_kagglehub()

# Upload image
uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Predict and display result
    with st.spinner("Classifying..."):
        try:
            class_label, confidence = predict_image(img, model)
            st.success(f"Prediction: {class_label}")
            st.write(f"Confidence: {confidence * 100:.2f}%")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
else:
    st.info("Please upload an image to classify.")
