import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image

# Define the CancerNet model
class CancerNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = tf.keras.models.Sequential()
        shape = (height, width, depth)
        channelDim = -1  

        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=shape))
        model.add(tf.keras.layers.BatchNormalization(axis=channelDim))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2))
        model.add(tf.keras.layers.Dropout(0.25))

        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
        model.add(tf.keras.layers.BatchNormalization(axis=channelDim))
        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
        model.add(tf.keras.layers.BatchNormalization(axis=channelDim))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2))
        model.add(tf.keras.layers.Dropout(0.25))

        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
        model.add(tf.keras.layers.BatchNormalization(axis=channelDim))
        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
        model.add(tf.keras.layers.BatchNormalization(axis=channelDim))
        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
        model.add(tf.keras.layers.BatchNormalization(axis=channelDim))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2))
        model.add(tf.keras.layers.Dropout(0.25))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=256, activation='relu'))
        model.add(tf.keras.layers.BatchNormalization(axis=channelDim))
        model.add(tf.keras.layers.Dropout(0.5))

        model.add(tf.keras.layers.Dense(units=classes, activation='softmax'))
        return model

# Load the trained model
@st.cache_resource
def load_model():
    model = CancerNet.build(width=48, height=48, depth=3, classes=2)
    
    model_path = "breastcancer.h5"

    
    if os.path.exists(model_path):
        model.load_weights(model_path)
    else:
        st.error(f"Error: Model file NOT found at {model_path}. Please check the file path.")

    return model

model = load_model()

# Function to preprocess image
def preprocess_image(image):
    image = np.array(image)
    image = cv2.resize(image, (48, 48))  
    image = image.astype("float") / 255.0  
    image = np.expand_dims(image, axis=0)  
    return image

# Streamlit UI
st.title("Breast Cancer Prediction")
st.write("Upload an image to classify it as **Malignant (Cancerous)** or **Benign (Non-Cancerous)**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)  # Updated parameter

    # Preprocess and predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    
    result = "Malignant (Cancerous)" if prediction[0][0] > 0.5 else "Benign (Non-Cancerous)"
    confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]

    st.write(f"### Prediction: **{result}**")
    st.write(f"### Confidence: **{confidence:.2f}**")
