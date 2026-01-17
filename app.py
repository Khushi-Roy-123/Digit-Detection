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
    invert = st.checkbox("Invert Image (Select this if image is Black digit on White background)", value=True)
    
    def preprocess_image(img, invert_colors):
        # 1. Convert to grayscale
        img_gray = img.convert('L')
        
        # 2. Invert if requested (MNIST is white on black)
        # If user uploaded black on white, we MUST invert.
        if invert_colors:
            img_gray = ImageOps.invert(img_gray)
            
        # 3. Thresholding to clear up noise and facilitate cropping
        # Convert to numpy array
        img_arr = np.array(img_gray)
        
        # Simple threshold: everything below 50 becomes 0, else 255.
        # This helps in removing noise and finding the bounding box of the digit.
        # Ensure we don't accidentally kill soft edges too much, but for bounding box it's good.
        # For the final image, we might want to keep the original grayscale values but cropped.
        
        # Let's find the bounding box using a binary version
        binary_arr = np.where(img_arr > 50, 255, 0).astype(np.uint8)
        coords = np.argwhere(binary_arr > 0)
        
        # Determine bounding box
        if coords.size == 0:
            # Empty image
            return img_gray.resize((28, 28))
            
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # Crop the ORIGINAL grayscale image (not the binary one)
        cropped = img_gray.crop((x_min, y_min, x_max+1, y_max+1))
        
        # 4. Resize with aspect ratio preservation to fit in 20x20 box
        # MNIST digits are roughly 20x20 inside a 28x28 box
        output_size = 20
        w, h = cropped.size
        scale = output_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # 5. Paste into 28x28 black canvas (centered)
        final_img = Image.new('L', (28, 28), 0)
        
        # Calculate paste position (center)
        paste_x = (28 - new_w) // 2
        paste_y = (28 - new_h) // 2
        
        final_img.paste(resized, (paste_x, paste_y))
        
        return final_img

    # Perform preprocessing
    img_processed = preprocess_image(image, invert)
    
    # Display processed image
    st.write("Processed Image (Input to Model):")
    st.image(img_processed, width=100, caption="28x28 Centered")
    
    # Flatten and Normalize
    img_array = np.array(img_processed)
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


