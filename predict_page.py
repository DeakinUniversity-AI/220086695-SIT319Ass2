import streamlit as st
import pickle
#load libraries/modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
import os
import cv2  #if not being installed yet, run: pip install opencv-python
import random
from keras_preprocessing.image import ImageDataGenerator  #for input pipeline

from datetime import datetime
from packaging import version

import os
import time

from IPython import display
import streamlit.components.v1 as stc

from PIL import Image 

@st.cache(allow_output_mutation=True)
def load_model():
    with open('saved_model.pkl', 'rb') as file:
        pickle_data = pickle.load(file)
    return pickle_data

data = load_model()
network = data['model']
target_classes = data['target_classes']

IMAGE_SIZE = 150

def load_image(image):
    return Image.open(image)

results = []

def show_predict_page():
    image = st.file_uploader("Please upload your household item to identify whether it s recyclable or not?",type=['png','jpeg','jpg'])
    new_result = []

    if image is not None:
        # file_details = {"FileName": image.name,
        #                 "FileType": image.type, 
        #                 "FileSize": image.size}
        # st.write(file_details)
        img = load_image(image)
        st.image(img, width=300)

        #write the image file to tmp folder
        with open(os.path.join("tmp", image.name), "wb") as f:
         f.write(image.getvalue())

        #preprocess the image using OpenCV
        image_array = cv2.imread(os.path.join("tmp", image.name), cv2.IMREAD_GRAYSCALE)
        image_array = cv2.resize(image_array, (IMAGE_SIZE, IMAGE_SIZE))

        x_test = []
        x_test.append(image_array)

        x_test[0].reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)

        #normalisation
        x_test = np.asarray(x_test).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)/255.0

        y_predicted = network.predict(x_test)
        value = np.round(y_predicted[0])
        st.write("""#### Predicted Result: """)
        if value.sum() < 1:
            st.write("Undertermined Item")
            new_result = [img, 'Undertermined Item']
        else:
            for index in np.arange(10):
                if value[index].any() == 1:
                    st.write(target_classes[index])
                    new_result = [img, target_classes[index]]

        

        st.write('Is the predicted result correct or wrong?')
        correct = st.button('Correct')
        wrong = st.button('Wrong')

        if correct:
            new_result.append('Correct')
            results.append(new_result)
        elif wrong:
            new_result.append('Wrong')
            results.append(new_result)

    if len(results) > 0:
        st.sidebar.write("""### Predicted history:""")
        for result in reversed(results):
            img, name, correct_or_wrong = result
            st.sidebar.image(img, width=200, caption=name + ' - ' + correct_or_wrong)
    else:
        st.sidebar.title('Recyclable Item Prediction')
        st.sidebar.image("https://www.whitehorse.vic.gov.au/sites/whitehorse.vic.gov.au/files/assets/images/Recycle%20Right%20A3%20Poster.jpg")

