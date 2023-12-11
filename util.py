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


def diagnosis(file):
    IMM_SIZE = 224  # Replace with your desired size

    # Download image
    ## YOUR CODE GOES HERE ##

    # Initialize image variable
    image = None

    try:
        # Attempt to read the image from the file
        image = mh.imread(file)
    except Exception as e:
        # Print an error message if the image cannot be read
        print(f"Error reading image from {file}: {e}")

    # Check if image is None (i.e., an error occurred during image reading)
    if image is None:
        # Handle the error or return an appropriate value
        return None

    # Prepare image for classification
    ## YOUR CODE GOES HERE ##

    # Check if the image has more than 2 dimensions (i.e., it's RGB or has an alpha channel)
    if len(image.shape) > 2:
        # Resize RGB and PNG images
        image = mh.resize_to(image, [IMM_SIZE, IMM_SIZE, image.shape[2]])
    else:
        # Resize grayscale images
        image = mh.resize_to(image, [IMM_SIZE, IMM_SIZE])

    # Check if the image has more than 2 dimensions (i.e., it's RGB)
    if len(image.shape) > 2:
        # Convert RGB to grayscale and remove alpha channel
        image = mh.colors.rgb2grey(image[:, :, :3], dtype=np.uint8)

    # Load model
    ## YOUR CODE GOES HERE ##

    # Load model architecture from JSON file
    with open('model.json', 'r') as json_file:
        loaded_model_json = json_file.read()

    # Close the JSON file
    model = model_from_json(loaded_model_json)

    # Load weights into the new model
    model.load_weights("model.h5")

    # Load history and lab from pickle files
    with open('history.pickle', 'rb') as f:
        history = pickle.load(f)

    with open('lab.pickle', 'rb') as f:
        lab = pickle.load(f)

    # Normalize the data
    ## YOUR CODE GOES HERE ##
    image = np.array(image) / 255

    # Reshape input images
    ## YOUR CODE GOES HERE ##
    image = image.reshape(-1, IMM_SIZE, IMM_SIZE, 1)

    # Predict the diagnosis
    ## YOUR CODE GOES HERE ##
    predicted_probabilities = model.predict(image)
    diag = np.argmax(predicted_probabilities, axis=-1)

    # Find the name of the diagnosis
    ## YOUR CODE GOES HERE ##
    diag = list(lab.keys())[list(lab.values()).index(diag[0])]

    return diag
