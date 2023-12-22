import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the model
model_path = ".\MNIST_Model.h5"
model = load_model(model_path)

# Function to preprocess the image
def preprocess_image(image):
    image = image.convert("L")
    image = image.resize((28, 28), Image.ANTIALIAS)
    image_array = np.array(image)
    image_array = image_array.reshape(1, 28, 28, 1) / 255.0
    return image_array

# Streamlit app
def main():
    st.title("Handwritten Digits Recognition")

    # File uploader for image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True, width=80 )
        # st.markdown("<style>img{max-height: 100px;}</style>", unsafe_allow_html=True)
        st.markdown("<style>img{max-height: 150px; max-width: 200px;}</style>", unsafe_allow_html=True)
        # Preprocess and make prediction
        image_array = preprocess_image(image)

        # Make prediction using TensorFlow
        prediction = model.predict(image_array)
        result = np.argmax(prediction)

        st.write(f"Predicted digit probably be : {result}")

if __name__ == "__main__":
    main()
