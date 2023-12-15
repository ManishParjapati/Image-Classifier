import streamlit as st
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import pickle
from PIL import Image

st.title('Image Classifier')

model = pickle.load(open('image_model.p', 'rb'))

upload_file = st.file_uploader("Choose an image", type="jpg")

if upload_file is not None:
    img = Image.open(upload_file)
    st.image(img, caption='uploaded image')

if st.button('Predict'):
    Categories = ['Bike', 'Car', 'Dog']
    st.write("Result...")
    flat_data = []
    img = np.array(img)
    img_resized = resize(img, (150, 150, 3))
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)
    y_out = model.predict(flat_data)
    y_out = Categories[y_out[0]]
    st.title(f"Predicted Output: {y_out}")


