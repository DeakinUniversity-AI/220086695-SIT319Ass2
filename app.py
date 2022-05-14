

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

import streamlit as st


from predict_page import show_predict_page
from about_page import show_about_page

about = st.sidebar.button('About the project')

if about:
    show_about_page()
else:
    show_predict_page()