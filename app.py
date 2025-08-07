# app.py
# Import necessary libraries
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import io

# Initialize the Flask application
app = Flask(__name__)

# --- Model Loading (TensorFlow Lite) ---
# Define the path to your trained .tflite model file.
MODEL_PATH = 'pneumonia_cnn_model_float16.tflite'

# Load the TFLite model and allocate tensors.
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    # Get input and output tensor details.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"Successfully loaded TFLite model from: {MODEL_PATH}")
except Exception as e:
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"!!! CRITICAL ERROR: COULD NOT LOAD TFLITE MODEL from {MODEL_PATH}")
    print(f"!!! Error: {e}")
    print("!!! The app will not work without the model.")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    interpreter = None

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
        
        # Convert image to grayscale ('L' mode)
        img = img.convert('L') 
        
        # Resize the image to the target size your model expects (e.g., 150x150)
        img = img.resize((150, 150))
        
        # Convert the image to a numpy array
        img_array = np.array(img, dtype=np.float32)
        
        # Rescale the image data
        img_array /= 255.0
        
        # Expand the dimensions to match the model's input shape (1, 150, 150, 1)
        img_array = np.expand_dims(img_array, axis=-1) # Add channel dimension
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
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
            # Preprocess the image
            processed_image = preprocess_image(file)
            if processed_image is None:
                return jsonify({'error': 'Could not process image'}), 500

            # --- TFLite Prediction ---
            # Set the value of the input tensor
            interpreter.set_tensor(input_details[0]['index'], processed_image)
            
            # Run the inference
            interpreter.invoke()
            
            # Extract the output data from the output tensor
            prediction = interpreter.get_tensor(output_details[0]['index'])
            
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
