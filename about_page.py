import streamlit as st
from predict_page import show_predict_page

def show_about_page():
    predict = st.sidebar.button('Predict Item')
    if predict:
        show_predict_page()

    st.title('A Deep Learning solution to reduce recycling contamination')
    st.write('*1. The image dataset comes the web resources below:*')
    st.write('https://www.kaggle.com/datasets/hseyinsaidkoca/recyclable-solid-waste-dataset-on-5-background-co')
    st.write('https://www.kaggle.com/code/tszheilau/0-9102-efficientnet-b0-svm-transfer-learning/data')
    st.write('https://www.kaggle.com/datasets/ashwinshrivastav/most-common-recyclable-and-nonrecyclable-objects')
    st.write('https://www.bing.com/images/trending')

    st.write('There are 1125 images in total collected across 10 different classes:')
    st.write('''
    0 : Recyclable Paper\n
    1 : Recyclable Cardboard\n
    2 : Recyclable Steel\n
    3 : Recyclable Glass\n
    4 : Recyclable Hard Plastic\n
    5 : Unrecyclable Takeaway Item\n
    6 : Unrecyclable Soft Plastic\n
    7 : Unrecyclable Tissue\n
    8 : Unrecyclable Disposable Item\n
    9 : Unrecyclable Polystyrene\n
    ''')

    st.write('*2. Convolutional Neural Network*')
    st.write('An input pipeline applied to the CNN model, the images were resized and transformed into certain size.')
    st.write('Then the sample images was split into training and test based on a ratio of 80:20, fitting and evaluating')
    st.write('the model.')

    st.write('*3. Performance of the model*')
    st.write('After improvements on the model, the validation accuray was as much as 82.47%.')

    st.write('\n\n')
    st.write('Code can be found at:')
    st.write('')