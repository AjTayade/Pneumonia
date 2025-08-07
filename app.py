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
MODEL_PATH = 'pneumonia_cnn_model.keras'

# Load your actual model
try:
    model = load_model(MODEL_PATH)
    print(f"Successfully loaded model from: {MODEL_PATH}")
except Exception as e:
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"!!! CRITICAL ERROR: COULD NOT LOAD MODEL from {MODEL_PATH}")
    print(f"!!! Error: {e}")
    print("!!! The app will not work without the model.")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    model = None

# --- Image Preprocessing Function ---
def preprocess_image(img_file):
    """
    Preprocesses the uploaded image to match the model's input requirements.
    - Converts to Grayscale.
    - Resizes the image to 150x150 pixels.
    - Converts the image to a NumPy array.
    - Rescales the pixel values.
    - Expands dimensions to create a batch of 1.
    """
    try:
        # Open the image file using PIL
        img = Image.open(io.BytesIO(img_file.read()))
        
        # --- THIS IS THE FIX ---
        # Convert image to grayscale ('L' mode) to match the model's expected input
        img = img.convert('L') 
        
        # Resize the image to the target size your model expects (e.g., 150x150)
        img = img.resize((150, 150))
        
        # Convert the image to a numpy array
        img_array = image.img_to_array(img)
        
        # Rescale the image data (if your model was trained with rescaled data)
        img_array /= 255.0
        
        # Expand the dimensions to match the model's input shape (1, 150, 150, 1)
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
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the image upload and prediction.
    """
    # Check if the model was loaded successfully
    if model is None:
        return jsonify({'error': 'Model is not loaded, check server logs.'}), 500

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
            
            # Interpret the prediction
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
    # This block is for deployment platforms like Render, Heroku, etc.
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
