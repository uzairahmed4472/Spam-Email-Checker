# app.py
# Flask API for spam email detection
# This is the main web service that provides REST API endpoints for spam detection

# Import required libraries for web service and machine learning
import os          # For file path operations
import sys         # For system-specific parameters and functions
import json        # For JSON data handling
import pickle      # For loading the trained tokenizer
from flask import Flask, request, jsonify  # Flask web framework components

# Set up file paths for imports and model artifacts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get current directory (where app.py is located)
PROJ_ROOT = os.path.dirname(BASE_DIR)                  # Get parent directory (project root)
sys.path.append(BASE_DIR)                              # Add current directory to Python path for imports

# Import our custom text cleaning function
from utils_text import basic_clean

# Import TensorFlow and Keras for machine learning
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define paths to saved model artifacts (created during training)
ART_DIR = os.path.join(BASE_DIR, "artifacts")                     # Directory containing model files (in current directory)
MODEL_PATH = os.path.join(ART_DIR, "spam_model.keras")           # Path to trained neural network model
TOK_PATH = os.path.join(ART_DIR, "tokenizer.pkl")                # Path to text tokenizer
CFG_PATH = os.path.join(ART_DIR, "config.json")                  # Path to configuration file

# Load the trained model and preprocessing components at startup
# This happens once when the server starts, not for each prediction
model = tf.keras.models.load_model(MODEL_PATH)                   # Load the trained LSTM model
with open(TOK_PATH, "rb") as f:                                  # Open tokenizer file in binary read mode
    tokenizer = pickle.load(f)                                   # Load the tokenizer object
with open(CFG_PATH, "r") as f:                                   # Open config file in read mode
    cfg = json.load(f)                                           # Load configuration as JSON

# Get the maximum sequence length from config (how many words to use per email)
MAX_LEN = cfg.get("max_len", 100)                               # Default to 100 if not found

# Create Flask application instance
app = Flask(__name__)

@app.route("/", methods=["GET"])
def health():
    """
    Health check endpoint
    Returns a simple status message to verify the service is running
    This is useful for monitoring and load balancers
    """
    return jsonify({"status":"ok", "service":"spam-detector"})

@app.route("/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint for spam detection
    Expects JSON input with 'text' field containing the email content
    Returns prediction results including label and confidence score
    """
    # Extract JSON data from the HTTP request
    # force=True: parse JSON even if Content-Type is not application/json
    # silent=True: return None instead of raising exception on parse error
    data = request.get_json(force=True, silent=True) or {}
    
    # Get the email text from the request, default to empty string if not provided
    text = data.get("text", "")
    
    # Clean the input text using the same preprocessing as training data
    # This ensures consistency between training and prediction
    cleaned = basic_clean(text)
    
    # Convert cleaned text to sequence of numbers (tokenization)
    # The tokenizer converts words to numbers based on the vocabulary learned during training
    seq = tokenizer.texts_to_sequences([cleaned])
    
    # Pad or truncate the sequence to fixed length (MAX_LEN)
    # padding="post": add zeros at the end if sequence is too short
    # truncating="post": remove words from the end if sequence is too long
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    
    # Make prediction using the trained model
    # verbose=0: suppress TensorFlow output during prediction
    # [0][0]: get the first (and only) prediction value from the batch
    proba = float(model.predict(pad, verbose=0)[0][0])
    
    # Convert probability to binary label
    # If probability >= 0.5, classify as spam; otherwise, classify as ham (legitimate)
    label = "Spam" if proba >= 0.5 else "Ham"
    
    # Return prediction results as JSON response
    return jsonify({
        "input": text,                    # Original input text
        "cleaned": cleaned,               # Preprocessed text used for prediction
        "label": label,                   # Final classification (Spam or Ham)
        "confidence": round(proba, 4)     # Confidence score (0-1, rounded to 4 decimal places)
    })

if __name__ == "__main__":
    """
    Main execution block - runs when script is executed directly
    Sets up the web server for production use
    """
    # Import waitress - a production WSGI server (better than Flask's built-in server)
    from waitress import serve
    
    # Get port number from environment variable, default to 8000
    port = int(os.environ.get("PORT", 8000))
    
    # Print startup message
    print(f"Starting on port {port} ...")
    
    # Start the web server
    # host="0.0.0.0": listen on all network interfaces (accessible from outside)
    # port: the port number to listen on
    serve(app, host="0.0.0.0", port=port)
