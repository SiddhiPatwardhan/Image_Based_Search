from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from sklearn.neighbors import NearestNeighbors
import os
from numpy.linalg import norm
from PIL import Image
import base64
from io import BytesIO

app = Flask(__name__)

# --- Load Precomputed Features & Filenames ---
IMAGE_DIR = r"C:\Users\HP\OneDrive\Desktop\images"
Image_features = pkl.load(open('Images_features.pkl', 'rb'))
filenames = pkl.load(open('filenames.pkl', 'rb'))

# Fix filenames to match local paths
filenames = [os.path.join(IMAGE_DIR, os.path.basename(f)) for f in filenames]

# --- Load Pretrained ResNet50 Model ---
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.models.Sequential([model, GlobalMaxPool2D()])

# --- KNN Model for Finding Similar Images ---
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(Image_features)

def extract_features_from_images(img_data, model):
    img = Image.open(BytesIO(img_data)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = preprocess_input(img_expand_dim)

    result = model.predict(img_preprocess).flatten()
    norm_result = result / norm(result)
    return norm_result

def get_image_base64(image_path):
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Process the image
    img_data = file.read()
    input_img_features = extract_features_from_images(img_data, model)
    distance, indices = neighbors.kneighbors([input_img_features])

    # Get base64 encoded images for recommendations
    recommendations = []
    for idx in indices[0][1:6]:
        img_path = filenames[idx]
        img_base64 = get_image_base64(img_path)
        recommendations.append(img_base64)

    return jsonify({
        'recommendations': recommendations
    })

if __name__ == '__main__':
    app.run(debug=True) 