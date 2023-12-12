import streamlit as st
from keras.models import model_from_json
from PIL import Image
from util import classify, set_background
import pickle

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

# Load background image
set_background("bg5.png")

# Set title
st.title('Pneumonia classification')

# Set header
st.header('Please upload a chest X-ray image')

# Upload file
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# Display image
if uploaded_file is not None:
    # Read image from the uploaded file
    image = Image.open(uploaded_file)

    # Classify the image using the model
    predicted_diagnosis = classify(image, model, lab)

    # Display the result
    st.write(f"Predicted Diagnosis: {predicted_diagnosis}")
