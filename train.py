# train.py
# Machine learning training script for spam email detection
# This script builds, trains, and evaluates an LSTM neural network model

# Import required libraries for data processing and machine learning
import os          # For file and directory operations
import sys         # For system-specific parameters and functions
import json        # For saving configuration files
import pickle      # For saving the tokenizer object
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation

# Import scikit-learn for data splitting and evaluation metrics
from sklearn.model_selection import train_test_split  # For splitting data into train/validation
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score  # For model evaluation

# Import TensorFlow and Keras for deep learning
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer  # For converting text to numbers
from tensorflow.keras.preprocessing.sequence import pad_sequences  # For making sequences same length

# Import Keras callbacks for training optimization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Set up file paths for the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get current directory
sys.path.append(BASE_DIR)                              # Add to Python path for imports

PROJ_ROOT = os.path.dirname(BASE_DIR)                  # Get project root directory
DATA_PATH = os.path.join(BASE_DIR, "data", "processed.csv")  # Path to processed data (in current directory)
ART_DIR = os.path.join(BASE_DIR, "artifacts")          # Directory for saving model artifacts (in current directory)
os.makedirs(ART_DIR, exist_ok=True)                    # Create artifacts directory if it doesn't exist

# Define paths for saving trained model components
MODEL_PATH = os.path.join(ART_DIR, "spam_model.keras")  # Path to save the trained model
TOK_PATH = os.path.join(ART_DIR, "tokenizer.pkl")       # Path to save the tokenizer
CFG_PATH = os.path.join(ART_DIR, "config.json")         # Path to save configuration

# Define hyperparameters for the model
MAX_LEN = 100           # Maximum number of words per email (sequences longer will be truncated)
VOCAB_SIZE = 20000      # Maximum vocabulary size (most frequent words to keep)
OOV_TOKEN = "<OOV>"     # Token for out-of-vocabulary words (words not seen during training)

def load_data():
    """
    Load the processed dataset and prepare it for training
    Converts text labels to binary format (spam=1, ham=0)
    """
    # Load the processed CSV file
    df = pd.read_csv(DATA_PATH)
    
    # Remove rows with missing text or label data
    df = df.dropna(subset=["text","label"])
    
    # Convert labels to binary format: spam=1, ham=0
    # First convert to string, then to lowercase, then check if equal to "spam"
    df["label"] = (df["label"].astype(str).str.lower() == "spam").astype(int)
    
    # Extract text and labels as separate arrays
    X = df["text"].astype(str).tolist()  # Convert text to list of strings
    y = df["label"].values               # Get labels as numpy array
    
    return X, y

def fit_tokenizer(X_train, X_val):
    """
    Create and fit a tokenizer on training data, then convert text to sequences
    Tokenization converts words to numbers that the neural network can understand
    """
    # Create a tokenizer with specified vocabulary size and OOV token
    tok = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
    
    # Fit the tokenizer on training text to build vocabulary
    # This learns which words are most common and assigns them numbers
    tok.fit_on_texts(X_train)

    # Convert text to sequences of numbers
    # Each word becomes a number based on the learned vocabulary
    seq_train = tok.texts_to_sequences(X_train)  # Convert training text
    seq_val = tok.texts_to_sequences(X_val)      # Convert validation text

    # Pad sequences to make them all the same length (MAX_LEN)
    # padding="post": add zeros at the end if sequence is too short
    # truncating="post": remove words from the end if sequence is too long
    seq_train = pad_sequences(seq_train, maxlen=MAX_LEN, padding="post", truncating="post")
    seq_val = pad_sequences(seq_val, maxlen=MAX_LEN, padding="post", truncating="post")
    
    return tok, seq_train, seq_val

def build_model(vocab_size):
    """
    Build the LSTM neural network model for spam detection
    The model uses embeddings, LSTM, and dense layers for classification
    """
    # Create a sequential model (layers stacked one after another)
    model = tf.keras.Sequential([
        # Embedding layer: converts word numbers to dense vectors
        # input_dim: vocabulary size, output_dim: embedding dimension, input_length: sequence length
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=32, input_length=MAX_LEN),
        
        # LSTM layer: processes sequences and learns long-term dependencies
        # 16: number of LSTM units (hidden state size)
        tf.keras.layers.LSTM(16),
        
        # Dense layer: fully connected layer with ReLU activation
        # 32: number of neurons, relu: activation function
        tf.keras.layers.Dense(32, activation="relu"),
        
        # Output layer: single neuron with sigmoid activation for binary classification
        # sigmoid outputs values between 0 and 1 (probability of spam)
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    
    # Compile the model with optimizer, loss function, and metrics
    model.compile(
        loss="binary_crossentropy",  # Loss function for binary classification
        optimizer="adam",            # Optimizer for updating weights
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]  # Metrics to track during training
    )
    
    return model

def main():
    """
    Main function that orchestrates the entire training pipeline
    """
    # Load the processed dataset
    X, y = load_data()

    # Split data into training and validation sets
    # test_size=0.2: use 20% for validation, 80% for training
    # stratify=y: ensure both classes are represented in both sets
    # random_state=42: ensures reproducible results
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create tokenizer and convert text to sequences
    tok, Xtr, Xva = fit_tokenizer(X_train, X_val)

    # Build the neural network model
    # Use actual vocabulary size (minimum of VOCAB_SIZE and actual vocabulary)
    model = build_model(vocab_size=min(VOCAB_SIZE, len(tok.word_index)+1))

    # Set up training callbacks for optimization
    # Early stopping: stop training if validation accuracy doesn't improve for 3 epochs
    es = EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)
    
    # Learning rate reduction: reduce learning rate if validation loss doesn't improve
    rl = ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5)

    # Train the model
    history = model.fit(
        Xtr, y_train,                    # Training data
        validation_data=(Xva, y_val),    # Validation data
        epochs=12,                       # Maximum number of training epochs
        batch_size=32,                   # Number of samples per gradient update
        callbacks=[es, rl],              # Training callbacks
        verbose=1                        # Show training progress
    )

    # Make predictions on validation set
    y_proba = model.predict(Xva).ravel()  # Get probability scores
    y_pred = (y_proba >= 0.5).astype(int)  # Convert to binary predictions (threshold=0.5)

    # Print evaluation metrics
    print("\nClassification Report:\n", classification_report(y_val, y_pred, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
    
    # Calculate and print ROC AUC score (measure of classification performance)
    try:
        print("ROC AUC:", round(roc_auc_score(y_val, y_proba),4))
    except Exception:
        pass  # Skip if there's an error calculating AUC

    # Save the trained model and preprocessing components
    model.save(MODEL_PATH)  # Save the trained model
    
    # Save the tokenizer for use during prediction
    with open(TOK_PATH, "wb") as f:
        pickle.dump(tok, f)
    
    # Save configuration for use during prediction
    with open(CFG_PATH, "w") as f:
        json.dump({"max_len": MAX_LEN}, f)

    # Print confirmation of saved files
    print("Saved model ->", MODEL_PATH)
    print("Saved tokenizer ->", TOK_PATH)
    print("Saved config ->", CFG_PATH)

# Run the main function if this script is executed directly
if __name__ == "__main__":
    main()
