import sys
import os

# Add the directory containing util.py to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from keras.models import load_model
from PIL import Image
from util import classify, set_background

# Set background
set_background('./bgs/bg5.png')

# Set title
st.title('Pneumonia classification')

# Set header
st.header('Please upload a chest X-ray image')

# Upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# Load classifier
model = load_model('./model/pmacp05.h5')

# Load class names
with open('./model/labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]

# Display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # Classify image
    class_name, conf_score = classify(image, model, class_names)

    # Write classification
    st.write("## {}".format(class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))
