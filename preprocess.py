# preprocess.py
# Data preprocessing script for spam email detection
# This script loads raw email data, cleans the text, and balances the dataset

# Import required libraries for data processing and text analysis
import os          # For file and directory operations
import re          # For regular expressions (text pattern matching)
import string      # For string manipulation and punctuation handling
import nltk        # Natural Language Toolkit for text processing
import pandas as pd  # For data manipulation and analysis
from nltk.corpus import stopwords  # For removing common English words

# Download NLTK stopwords if not already available
# Stopwords are common words like 'the', 'is', 'and' that don't help in classification
try:
    _ = stopwords.words('english')  # Try to access English stopwords
except LookupError:  # If stopwords are not downloaded
    nltk.download('stopwords')      # Download them automatically

# Create a translation table for efficient punctuation removal
# This is faster than using regex for simple punctuation removal
PUNCT_TABLE = str.maketrans('', '', string.punctuation)

# Get English stopwords as a set for faster lookup during text processing
STOPWORDS = set(stopwords.words('english'))

# ------------------------
# Text Cleaning Functions
# ------------------------

def remove_punct(text: str) -> str:
    """
    Remove all punctuation marks from text using translation table
    This is more efficient than regex for simple punctuation removal
    """
    return text.translate(PUNCT_TABLE)

def remove_stopwords(text: str) -> str:
    """
    Remove common English stopwords that don't help in spam detection
    Stopwords like 'the', 'is', 'and' appear frequently but don't indicate spam
    """
    # Split text into words, convert to lowercase, filter out stopwords
    tokens = [w.lower() for w in text.split() if w.lower() not in STOPWORDS]
    # Join the remaining words back into a sentence
    return " ".join(tokens)

def basic_clean(text: str) -> str:
    """
    Comprehensive text cleaning function for email preprocessing
    This function prepares raw email text for machine learning by:
    1. Removing email headers and metadata
    2. Removing URLs and email addresses
    3. Removing numbers and normalizing text
    4. Removing punctuation and stopwords
    """
    # Convert input to string (handle any non-string inputs like None or numbers)
    text = str(text)
    
    # Remove "Subject:" headers that are common in email datasets
    # flags=re.IGNORECASE makes the search case-insensitive
    text = re.sub(r'\bSubject\b', '', text, flags=re.IGNORECASE)
    
    # Remove URLs and web addresses (http://, https://, www.)
    # \S+ matches any non-whitespace characters
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    
    # Remove email addresses (pattern: word@word.word)
    text = re.sub(r'\S+@\S+', ' ', text)
    
    # Remove standalone numbers (spam often contains many numbers)
    # \b ensures we match whole words only
    text = re.sub(r'\b\d+\b', ' ', text)
    
    # Normalize multiple spaces, tabs, and newlines to single spaces
    # \s+ matches one or more whitespace characters
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove punctuation marks using the translation table
    text = remove_punct(text)
    
    # Remove common stopwords
    text = remove_stopwords(text)
    
    return text

# ------------------------
# Dataset Functions
# ------------------------

# Set up file paths for data processing
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get current directory
DATA_DIR = os.path.join(BASE_DIR, "data")              # Path to data folder
RAW_PATH = os.path.join(DATA_DIR, "Emails.csv")        # Path to raw email data
PROCESSED_CSV = os.path.join(DATA_DIR, "processed.csv")  # Path to processed data

def load_dataset():
    """
    Load the raw email dataset and validate its structure
    Ensures the dataset has the required 'text' and 'label' columns
    """
    # Check if the raw data file exists
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError("Emails.csv not found in data/ folder")
    
    # Load the CSV file with latin-1 encoding (common for email datasets)
    df = pd.read_csv(RAW_PATH, encoding="latin-1")
    
    # Validate that required columns exist
    if "text" in df.columns and "label" in df.columns:
        # Return only the required columns
        return df[["text", "label"]]
    else:
        # Raise error with helpful message showing what columns were found
        raise ValueError(f"Emails.csv must contain 'text' and 'label' columns, found: {df.columns}")

def balance_downsample(df: pd.DataFrame) -> pd.DataFrame:
    """
    Balance the dataset by downsampling the majority class (ham) to match spam count
    This prevents the model from being biased toward the majority class
    """
    # Normalize label values: convert to string, strip whitespace, convert to lowercase
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    
    # Filter to only include valid labels (spam, ham, 0, 1)
    df = df[df["label"].isin(["spam", "ham", "0", "1"])]
    
    # Standardize label values: convert 0->ham, 1->spam
    df["label"] = df["label"].replace({"0": "ham", "1": "spam"})

    # Separate spam and ham emails
    ham = df[df["label"] == "ham"]    # Legitimate emails
    spam = df[df["label"] == "spam"]  # Spam emails
    
    # Ensure we have both classes in the dataset
    if len(spam) == 0 or len(ham) == 0:
        raise ValueError("Dataset must contain both spam and ham")

    # Downsample ham emails to match the number of spam emails
    # random_state=42 ensures reproducible results
    ham_bal = ham.sample(n=len(spam), random_state=42)
    
    # Combine balanced ham and all spam, then shuffle the entire dataset
    # frac=1.0 means shuffle all rows, reset_index creates new sequential index
    balanced = pd.concat([ham_bal, spam]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    return balanced

# ------------------------
# Main Processing Function
# ------------------------

def main():
    """
    Main function that orchestrates the entire preprocessing pipeline
    """
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Load the raw dataset
    df = load_dataset()
    print("Raw shape:", df.shape)  # Print original dataset size

    # Ensure text column is string type and clean the text
    df["text"] = df["text"].astype(str)                    # Convert to string
    df["text_clean"] = df["text"].apply(basic_clean)       # Apply cleaning function

    # Balance the dataset and rename columns for consistency
    balanced = balance_downsample(df[["text_clean", "label"]].rename(columns={"text_clean": "text"}))
    print("Balanced shape:", balanced.shape)  # Print balanced dataset size

    # Save the processed dataset to CSV
    balanced.to_csv(PROCESSED_CSV, index=False)  # index=False prevents saving row numbers
    print("Saved processed:", PROCESSED_CSV)

# Run the main function if this script is executed directly
if __name__ == "__main__":
    main()
