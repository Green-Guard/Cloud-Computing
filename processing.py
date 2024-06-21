from flask import Flask, request, jsonify
from google.cloud import storage
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os
import cv2
import numpy as np

app = Flask(__name__)

# Initialize Google Cloud Storage client
storage_client = storage.Client()

# Environment variables
image_bucket_name = 'greenguard1'
model_bucket_name = 'greenguard_ml'
model_file_name = 'model_train.h5'

# Adjust the model blob path to point directly to the file in the bucket
model_blob_path = model_file_name

tmp_dir = '/tmp'
os.makedirs(tmp_dir, exist_ok=True)
model_local_path = os.path.join(tmp_dir, "model_train.h5")

def download_model():
  # Construct the full path of the model file in Cloud Storage
  model_blob_path = os.path.join(model_bucket_name, model_file_name)

  # Download the model file
  blob = storage_client.bucket(model_bucket_name).blob(model_file_name)
  blob.download_to_filename(model_local_path)

  print(f"Model downloaded from {model_blob_path} to {model_local_path}")

# Download the ML model once when the app starts
download_model()

# Define your Keras model
def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(len(disease_classes), activation='softmax'))  # Adjust the number of neurons
    return model

# Define your disease classes
disease_classes = [
    "Strawberry___healthy",
    "Grape___Black_rot",
    "Potato___Early_blight",
    "Blueberry___healthy",
    "Corn_(maize)_healthy",
    "Tomato___Target_Spot",
    "Peach___healthy",
    "Potato___Late_blight",
    "Tomato___Late_blight",
    "Tomato___Tomato_mosaic_virus",
    "Pepper,bell__healthy",
    "Orange__Haunglongbing(Citrus_greening)",
    "Tomato___Leaf_Mold",
    "Grape__Leaf_blight(Isariopsis_Leaf_Spot)",
    "Cherry_(including_sour)_Powdery_mildew",
    "Apple___Cedar_apple_rust",
    "Tomato___Bacterial_spot",
    "Grape___healthy",
    "Tomato___Early_blight",
    "Corn_(maize)Common_rust",
    "Grape__Esca(Black_Measles)",
    "Raspberry___healthy",
    "Tomato___healthy",
    "Cherry_(including_sour)_healthy",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Apple___Apple_scab",
    "Corn_(maize)_Northern_Leaf_Blight",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Peach___Bacterial_spot",
    "Pepper,bell__Bacterial_spot",
    "Tomato___Septoria_leaf_spot",
    "Squash___Powdery_mildew",
    "Corn_(maize)_Cercospora_leaf_spot Gray_leaf_spot",
    "Apple___Black_rot",
    "Apple___healthy",
    "Strawberry___Leaf_scorch",
    "Potato___healthy",
    "Soybean___healthy",
]

# Load the ML model globally
try:
    ml_model = tf.keras.models.load_model(model_local_path)
    print(f"Model loaded from {model_local_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    data = request.files
    file_name = data['image'].filename  # Get the filename from the file object

    # Download image from GCS
    image_blob = storage_client.bucket(image_bucket_name).blob(file_name)
    image_path = f'/tmp/{file_name}'
    image_blob.download_to_filename(image_path)
    print(f"Image downloaded to {image_path}")

    # Load and preprocess the image
    try:
        image_data = cv2.imread(image_path)
        if image_data is None:
            return jsonify({"message": "Failed to load image"}), 400

        resized_image = cv2.resize(image_data, (128, 128))  # Resize to match model input shape
        preprocessed_image = np.expand_dims(resized_image, axis=0)  # Add batch dimension

        # Run your ML model on the preprocessed image
        predictions = ml_model.predict(preprocessed_image)

        # Process the predictions (e.g., get the class with the highest probability)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = predictions[0][predicted_class]

        # Get the disease name from the dictionary
        disease_name = disease_classes[predicted_class]  # Use the disease_classes list

        result = {
        "predicted_class": int(predicted_class),  # Rename "class" to "predicted_class"
        "confidence": float(confidence),
        "disease": disease_name
    }

        return jsonify({"message": "Image analyzed successfully", "result": result}), 200
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({"message": "Error processing image", "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)
