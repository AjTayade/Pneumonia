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
# Define the path to your 2-class TFLite model file.
# Make sure this file is in the same directory as app.py
MODEL_PATH = 'Pneumonia_CNN_model (1).tflite'

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
        # Convert image to grayscale ('L' mode) as the model expects
        img = img.convert('L') 
        # Resize the image to the model's expected input size (150x150)
        img = img.resize((100, 100))
        # Convert to a NumPy array with the correct data type
        img_array = np.array(img, dtype=np.float32)
        # Rescale pixel values from [0, 255] to [0, 1]
        img_array /= 255.0
        # Add a channel dimension
        img_array = np.expand_dims(img_array, axis=-1)
        # Add a batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error in preprocessing image: {e}")
        return None

# --- Routes ---
@app.route('/', methods=['GET'])
def index():
    """ Renders the main page of the web application. """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """ Handles the image upload and prediction for the 2-class model. """
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
            
            # --- Logic for 2 Classes (Binary Classification) ---
            confidence = float(prediction[0][0])
            if confidence > 0.5:
                prediction_result = 'Pneumonia'
            else:
                prediction_result = 'Normal'
                # Invert confidence for the 'Normal' class to show certainty
                confidence = 1 - confidence

            return jsonify({
                'prediction': prediction_result,
                'confidence': f"{confidence*100:.2f}%"
            })

        except Exception as e:
            print(f"Prediction error: {e}")
            return jsonify({'error': 'An error occurred during prediction.'}), 500

'''# app.py
# Import necessary libraries
from flask import Flask, request, jsonify, render_template
# Use the lightweight tflAite-runtime interpreter
from tflite_runtime.interpreter import Interpreter
import numpy as np
import os
from PIL import Image
import io

# Initialize the Flask application
app = Flask(__name__)

# --- Define Class Names (in alphabetical order, as Keras does) ---
# This order is confirmed from your training notebook's `class_indices`
CLASS_NAMES = ['COVID', 'NORMAL', 'PNEUMONIA']

# --- Model Loading (TFLite Runtime) ---
# Get the absolute path of the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Join this path with the model filename to create a robust, absolute path
# Make sure your .tflite file is in the same directory as app.py
MODEL_PATH = os.path.join(BASE_DIR, '3Class_pneumonia_model.tflite')

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
    Handles the image upload and prediction for the 3-class model.
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
            
            # The output is now an array of 3 probabilities
            prediction = interpreter.get_tensor(output_details[0]['index'])
            
            # --- NEW LOGIC FOR 3 CLASSES ---
            # Find the index of the highest probability
            predicted_class_index = np.argmax(prediction[0])
            
            # Get the name of the predicted class using the CLASS_NAMES list
            prediction_result = CLASS_NAMES[predicted_class_index]
            
            # Get the confidence score of the highest probability
            confidence = float(prediction[0][predicted_class_index])

            return jsonify({
                'prediction': prediction_result,
                'confidence': f"{confidence*100:.2f}%"
            })

        except Exception as e:
            print(f"Prediction error: {e}")
            return jsonify({'error': 'An error occurred during prediction.'}), 500

if __name__ == '__main__':
    # Use 0.0.0.0 to make it accessible from outside a container
    app.run(host='0.0.0.0', port=5000)
'''

