import numpy as np
from PIL import Image
import tensorflow as tf
import kagglehub
import streamlit as st
import os 
from keras.layers import TFSMLayer

# Load the pre-trained model using kagglehub
@st.cache_resource
def load_model():
    path = kagglehub.model_download("eashitadhillon/pneumonia_detection_model/keras/default")
    st.write(f"Model downloaded to: {path}")
    
    try:
        # Use TFSMLayer for inference
        model = TFSMLayer(path, call_endpoint='serving_default')  # Adjust `call_endpoint` if necessary
        st.write("Model loaded using TFSMLayer.")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        raise
    return model

# Function to preprocess and predict the uploaded image
def predict_image(img):
    img = img.convert("RGB")  # Ensure the image is in RGB format
    img = img.resize((224, 224))  # Resize to match model input size
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make a prediction using TFSMLayer or SavedModel
    prediction = model(img_array)  # Call the model directly
    prediction_value = prediction.numpy()[0][0]  # Convert prediction to numpy

    class_label = "PNEUMONIA" if prediction_value > 0.5 else "NORMAL"
    confidence = prediction_value if prediction_value > 0.5 else 1 - prediction_value
    return class_label, confidence


# Load the model
model = load_model()

# Streamlit interface
st.title("Pneumonia Detection from Chest X-rays")
st.write("Upload a chest X-ray image to classify it as NORMAL or PNEUMONIA.")

# Upload image
uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Predict and display result
    with st.spinner("Classifying..."):
        try:
            class_label, confidence = predict_image(img)
            st.success(f"Prediction: {class_label}")
            st.write(f"Confidence: {confidence * 100:.2f}%")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
else:
    st.info("Please upload an image to classify.")
