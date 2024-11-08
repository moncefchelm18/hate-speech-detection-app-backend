from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
import re
import pickle

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the pre-trained model from the 'hate_speech_model' directory
print("Loading model...")
model = tf.keras.models.load_model('hate_speech_model')
print("Model loaded successfully.")

# model.save('hate_speech_model.h5')

# Load tokenizer
print("Loading tokenizer...")
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
print("Tokenizer loaded successfully.")

# Route for the homepage (GET request)
@app.route('/', methods=['GET'])
def home():
    print("Received a GET request on the home route.")
    return jsonify({"message": "Welcome to the Hate Speech Detection API!"})

# Route for prediction (POST request)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Received a POST request on the /predict route.")
        # Get the text from the request
        data = request.get_json(force=True)  # Added force=True to handle different content types
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided in the request body'}), 400
            
        text = data['text']
        print(f"Text received for prediction: {text}")

        # Preprocess the input text
        cleaned_text = clean_content(text)
        print(f"Cleaned text: {cleaned_text}")

        # Convert text to sequence and pad it
        seq = tokenizer.texts_to_sequences([cleaned_text])
        padded_seq = pad_sequences(seq, maxlen=100)  # Adjust maxlen to match your training
        print(f"Padded sequence shape: {padded_seq.shape}")

        # Get the prediction from the model
        prediction = model.predict(padded_seq, verbose=0)  # Added verbose=0 to reduce output
        print(f"Prediction result: {prediction}")

        # Return the prediction as a JSON response
        result = {
            'prediction': 'Hate Speech Detected' if prediction[0] > 0.5 else 'No Hate Speech Detected',
            'confidence': float(prediction[0])  # Convert numpy float to Python float
        }
        print(f"Returning result: {result}")
        return jsonify(result)

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({'error': str(e)}), 500

def clean_content(text):
    print("Cleaning the text...")
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^A-Za-z0-9\s\']', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    print(f"Cleaned text: {text}")
    return text

if __name__ == '__main__':
    print("Starting Flask app...")
    # Add host='0.0.0.0' to make it accessible from other machines
    app.run(host='0.0.0.0', port=5000, debug=True)
    print("Flask app is running on http://127.0.0.1:5000/")