import streamlit as st
import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

# 1. Load Model
@st.cache_resource
def load_model():
    try:
        # Try loading from models directory
        return joblib.load('models/mnist_model.pkl')
    except FileNotFoundError:
        # Fallback if running from a different context or file moved differently
        try:
            return joblib.load('mnist_model.pkl')
        except FileNotFoundError:
            return None

model = load_model()

# 2. UI Setup
st.set_page_config(page_title="MNIST Digit Recognizer", page_icon="🔢", layout="centered")

st.title("🔢 MNIST Digit Recognizer")
st.markdown("Upload an image of a handwritten digit (0-9) to predict what it is.")

if model is None:
    st.error("Model not found! Please run `python src/mnist_train.py` to generate `models/mnist_model.pkl`.")
    st.stop()

# 3. Image Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 4. Preprocessing
    image = Image.open(uploaded_file)
    
    st.image(image, caption='Uploaded Image', use_column_width=False, width=200)
    
    # Checkbox to invert image
    invert = st.checkbox("Invert Image (Check this if image is Black digit on White background)", value=True)
    
    if invert:
        image = ImageOps.invert(image.convert('RGB'))
    
    # Convert to grayscale
    img_gray = image.convert('L')
    
    # Resize to 28x28
    img_resized = img_gray.resize((28, 28))
    
    # Display processed image
    st.write("Processed Image (28x28 Grayscale):")
    st.image(img_resized, width=100)
    
    # Flatten and Normalize
    img_array = np.array(img_resized)
    img_flat = img_array.flatten()
    img_normalized = img_flat / 255.0
    
    # Reshape for prediction (1, 784)
    img_input = img_normalized.reshape(1, -1)
    
    # 5. Prediction
    if st.button("Predict"):
        prediction = model.predict(img_input)
        prediction_proba = model.predict_proba(img_input)
        
        predicted_digit = prediction[0]
        confidence = np.max(prediction_proba)
        
        st.success(f"Prediction: **{predicted_digit}**")
        st.info(f"Confidence: **{confidence:.2%}**")
        
        # Plot probability distribution
        st.subheader("Prediction Probabilities")
        proba_df = pd.DataFrame(prediction_proba, columns=[str(i) for i in range(10)])
        st.bar_chart(proba_df.T)


