import base64
import streamlit as st
from PIL import ImageOps, Image
import numpy as np
!pip install mahotas
import mahotas as mh
from keras.models import model_from_json
import pickle

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


def classify(image, model, lab):
    """
    This function takes an image, a model, and a label mapping and returns the predicted diagnosis.

    Parameters:
        image (PIL.Image.Image): An image to be classified.
        model (keras.Model): A trained machine learning model for diagnosis prediction.
        lab (dict): A mapping of diagnosis labels to integer values.

    Returns:
        The predicted diagnosis.
    """
    # convert image to (224, 224)
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    # convert image to numpy array
    image_array = np.asarray(image)

    # normalize image
    normalized_image_array = image_array / 255.0

    # reshape input images
    image_input = normalized_image_array.reshape(-1, 224, 224, 1)

    # make prediction
    predicted_probabilities = model.predict(image_input)
    diagnosis_index = np.argmax(predicted_probabilities, axis=-1)
    predicted_diagnosis = list(lab.keys())[list(lab.values()).index(diagnosis_index[0])]

    return predicted_diagnosis


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

# Load background image
set_background("background_image.png")

# Upload image in Streamlit
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Read image from the uploaded file
    image = Image.open(uploaded_file)

    # Classify the image using the model
    predicted_diagnosis = classify(image, model, lab)

    # Display the result
    st.write(f"Predicted Diagnosis: {predicted_diagnosis}")
