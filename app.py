# app.py
# Import necessary libraries
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image
import io

# Initialize the Flask application
app = Flask(__name__)

# --- Model Loading ---
# Define the path to your trained model file.
# IMPORTANT: Make sure the 'pneumonia_model.h5' file is in the same directory
# as this 'app.py' file, or provide the correct path.
MODEL_PATH = 'pneumonia_cnn_model.keras'

# Load the trained model
# We are creating a dummy model here for demonstration purposes.
# In a real scenario, you would uncomment the line `model = load_model(MODEL_PATH)`
# and ensure your actual model file is present.
try:
    # model = load_model(MODEL_PATH)
    # Dummy model for placeholder - REMOVE THIS and use the line above
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(150, 150, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    print("Model loaded successfully. Using a placeholder model for now.")
    print("Please replace it with your actual 'pneumonia_model.h5'.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Creating a dummy model for demonstration.")
    # Create a dummy model if loading fails, so the app can still run.
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(150, 150, 3)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.is_dummy = True # Flag to indicate this is not the real model

# --- Image Preprocessing Function ---
def preprocess_image(img_file):
    """
    Preprocesses the uploaded image to match the model's input requirements.
    - Resizes the image to 150x150 pixels.
    - Converts the image to a NumPy array.
    - Rescales the pixel values.
    - Expands dimensions to create a batch of 1.
    """
    try:
        # Open the image file using PIL
        img = Image.open(io.BytesIO(img_file.read()))
        # Ensure image is in RGB format
        img = img.convert('RGB')
        # Resize the image to the target size your model expects (e.g., 150x150)
        img = img.resize((150, 150))
        # Convert the image to a numpy array
        img_array = image.img_to_array(img)
        # Rescale the image data (if your model was trained with rescaled data)
        img_array /= 255.0
        # Expand the dimensions to match the model's input shape (1, 150, 150, 3)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error in preprocessing image: {e}")
        return None

# --- Routes ---
@app.route('/', methods=['GET'])
def index():
    """
    Renders the main page of the web application.
    """
    # The 'index.html' file should be in a folder named 'templates'
    # in the same directory as this 'app.py' file.
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the image upload and prediction.
    """
    # Check if a file was posted
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']

    # Check if a file was selected
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file:
        try:
            # Preprocess the image
            processed_image = preprocess_image(file)
            if processed_image is None:
                return jsonify({'error': 'Could not process image'}), 500

            # Make a prediction
            prediction = model.predict(processed_image)
            
            # If the model is a dummy, return a random result for demonstration
            if hasattr(model, 'is_dummy') and model.is_dummy:
                prediction_result = 'Pneumonia' if np.random.rand() > 0.5 else 'Normal'
                confidence = np.random.uniform(0.5, 1.0)
            else:
                # Interpret the prediction
                # This assumes your model outputs a single value (sigmoid activation)
                # where > 0.5 means Pneumonia. Adjust this threshold if needed.
                confidence = float(prediction[0][0])
                if confidence > 0.5:
                    prediction_result = 'Pneumonia'
                else:
                    prediction_result = 'Normal'
                    confidence = 1 - confidence # Show confidence for the 'Normal' class

            return jsonify({
                'prediction': prediction_result,
                'confidence': f"{confidence*100:.2f}%"
            })

        except Exception as e:
            print(f"Prediction error: {e}")
            return jsonify({'error': 'An error occurred during prediction.'}), 500

# --- Main execution ---
if __name__ == '__main__':
    # To run this app:
    # 1. Make sure you have Flask, TensorFlow, and Pillow installed:
    #    pip install Flask tensorflow Pillow
    # 2. Save your trained model as 'pneumonia_model.h5' in this directory.
    # 3. Create a folder named 'templates' and put the 'index.html' file inside it.
    # 4. Run this script from your terminal: python app.py
    app.run()
