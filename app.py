# app.py
# Import necessary libraries
from flask import Flask, request, jsonify, render_template
# Use the lightweight tflite-runtime interpreter
from tflite_runtime.interpreter import Interpreter 
import numpy as np
import os
from PIL import Image
import io

# Initialize the Flask application
app = Flask(__name__)


# --- Model Loading (TFLite Runtime) ---
# Get the absolute path of the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Join this path with the model filename to create a robust, absolute path
MODEL_PATH = os.path.join(BASE_DIR, 'Pneumonia_CNN_model.tflite')
# --- Model Loading (TFLite Runtime) ---
#MODEL_PATH = 'Pneumonia_CNN_model.tflite'
interpreter = None
input_details = None
output_details = None

try:
    # Load the TFLite model and allocate tensors.
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    # Get input and output tensor details.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"Successfully loaded TFLite model from: {MODEL_PATH}")
except Exception as e:
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"!!! CRITICAL ERROR: COULD NOT LOAD MODEL from {MODEL_PATH}")
    print(f"!!! Error: {e}")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

# --- Image Preprocessing Function ---
def preprocess_image(img_file):
    """
    Preprocesses the uploaded image to match the model's input requirements.
    """
    try:
        img = Image.open(io.BytesIO(img_file.read()))
        img = img.convert('L') # Convert to grayscale
        img = img.resize((100, 100))
        img_array = np.array(img, dtype=np.float32)
        img_array /= 255.0
        img_array = np.expand_dims(img_array, axis=-1)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error in preprocessing image: {e}")
        return None

# --- Routes ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the image upload and prediction using the TFLite interpreter.
    """
    if interpreter is None:
        return jsonify({'error': 'Model is not loaded, check server logs.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file:
        try:
            processed_image = preprocess_image(file)
            if processed_image is None:
                return jsonify({'error': 'Could not process image'}), 500

            # --- TFLite Prediction ---
            interpreter.set_tensor(input_details[0]['index'], processed_image)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])
            
            confidence = float(prediction[0][0])
            if confidence > 0.5:
                prediction_result = 'Pneumonia'
            else:
                prediction_result = 'Normal'
                confidence = 1 - confidence

            return jsonify({
                'prediction': prediction_result,
                'confidence': f"{confidence*100:.2f}%"
            })

        except Exception as e:
            print(f"Prediction error: {e}")
            return jsonify({'error': 'An error occurred during prediction.'}), 500
