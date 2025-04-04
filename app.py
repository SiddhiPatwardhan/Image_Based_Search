import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from sklearn.neighbors import NearestNeighbors
import os
from numpy.linalg import norm
import streamlit as st
from io import BytesIO
from PIL import Image

# --- Streamlit UI Header ---
st.title("👗 Image based Fashion Recommendation System")

# --- Load Precomputed Features & Filenames ---
IMAGE_DIR = r"C:\Users\HP\OneDrive\Desktop\images"  # Corrected image path
Image_features = pkl.load(open('Images_features.pkl', 'rb'))
filenames = pkl.load(open('filenames.pkl', 'rb'))

# Fix filenames to match local paths
filenames = [os.path.join(IMAGE_DIR, os.path.basename(f)) for f in filenames]

# --- Load Pretrained ResNet50 Model ---
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.models.Sequential([model, GlobalMaxPool2D()])


# --- Function to Extract Features from Uploaded Image ---
def extract_features_from_images(uploaded_image, model):
    img = Image.open(uploaded_image).convert("RGB")  # Open and Convert image
    img = img.resize((224, 224))  # Resize for model compatibility
    img_array = np.array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = preprocess_input(img_expand_dim)

    result = model.predict(img_preprocess).flatten()
    norm_result = result / norm(result)
    return norm_result


# --- KNN Model for Finding Similar Images ---
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(Image_features)

# --- Upload & Process Image ---
upload_file = st.file_uploader("📤 Upload an Image for Recommendation", type=["jpg", "png", "jpeg"])

if upload_file is not None:
    # Display Uploaded Image
    st.subheader("📷 Uploaded Image")
    st.image(upload_file, width=200)

    # Extract Features & Find Nearest Neighbors
    input_img_features = extract_features_from_images(upload_file, model)
    distance, indices = neighbors.kneighbors([input_img_features])

    # Display Recommendations
    st.subheader("✨ Recommended Images")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.image(filenames[indices[0][1]], width=150)
    with col2:
        st.image(filenames[indices[0][2]], width=150)
    with col3:
        st.image(filenames[indices[0][3]], width=150)
    with col4:
        st.image(filenames[indices[0][4]], width=150)
    with col5:
        st.image(filenames[indices[0][5]], width=150)