import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

# -------------------------------------------------
# Page setup
# -------------------------------------------------
st.set_page_config(page_title="MNIST Digit Classifier", page_icon="✏️")
st.title("✏️ MNIST Digit Classifier")
st.write("Upload an image of a handwritten digit (0-9) and the model will predict it.")

# -------------------------------------------------
# Load the trained model (cached so it only loads once)
# -------------------------------------------------
@st.cache_resource
def load_model():
    # Make sure this filename matches the model file you upload to GitHub
    model = tf.keras.models.load_model("mnist_model.h5")
    return model

model = load_model()

# -------------------------------------------------
# Helper: preprocess any image into MNIST format
# 28x28, grayscale, white digit on black background, normalized 0-1
# -------------------------------------------------
def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("L")                 # grayscale
    img = ImageOps.invert(img)             # MNIST digits are white-on-black
    img = img.resize((28, 28))
    arr = np.array(img).astype("float32") / 255.0
    arr = arr.reshape(1, 784)              # flatten to match model's (None, 784) input
    return arr

# -------------------------------------------------
# Upload an image
# -------------------------------------------------
uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

image_to_predict = None
if uploaded_file is not None:
    image_to_predict = Image.open(uploaded_file)
    st.image(image_to_predict, caption="Uploaded image", width=150)

# -------------------------------------------------
# Predict
# -------------------------------------------------
if st.button("Predict"):
    if image_to_predict is None:
        st.warning("Please upload a digit image first.")
    else:
        processed = preprocess_image(image_to_predict)
        prediction = model.predict(processed)
        predicted_digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction)) * 100

        st.success(f"### Predicted digit: **{predicted_digit}**")
        st.write(f"Confidence: {confidence:.2f}%")
        st.bar_chart(prediction[0])
