import base64
import streamlit as st
from PIL import ImageOps, Image
import numpy as np

def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

# Example usage in Streamlit app
# Assume 'model', 'lab', and 'image_file' are defined appropriately before this point

# Load model architecture from JSON file
with open('model.json', 'r') as json_file:
    loaded_model_json = json_file.read()

# Close the JSON file
model = model_from_json(loaded_model_json)

# Load weights into the new model
model.load_weights("model.h5")

# Load label mapping from pickle file
with open('lab.pickle', 'rb') as f:
    lab = pickle.load(f)

# Load background image from the "bgs" folder
set_background("bgs/bg5.png")

# Upload image in Streamlit
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Read image from the uploaded file
    image = Image.open(uploaded_file)

    # Classify the image using the model
    predicted_diagnosis = classify(image, model, lab)

    # Display the result
    st.write(f"Predicted Diagnosis: {predicted_diagnosis}")
