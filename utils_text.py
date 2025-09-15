# utils_text.py
# Text processing utilities for spam detection
# This file contains the basic_clean function used by the Flask app

import re
import string
import nltk
from nltk.corpus import stopwords

# Download stopwords if missing (needed for text cleaning)
try:
    _ = stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# Create a translation table to remove punctuation efficiently
PUNCT_TABLE = str.maketrans('', '', string.punctuation)

# Get English stopwords set for faster lookup
STOPWORDS = set(stopwords.words('english'))

def remove_punct(text: str) -> str:
    """
    Remove all punctuation marks from text using translation table
    This is faster than regex for simple punctuation removal
    """
    return text.translate(PUNCT_TABLE)

def remove_stopwords(text: str) -> str:
    """
    Remove common English stopwords like 'the', 'is', 'and', etc.
    These words don't help in spam detection and can add noise
    """
    # Split text into words, convert to lowercase, filter out stopwords
    tokens = [w.lower() for w in text.split() if w.lower() not in STOPWORDS]
    # Join the remaining words back into a sentence
    return " ".join(tokens)

def basic_clean(text: str) -> str:
    """
    Comprehensive text cleaning function for spam detection
    This function prepares raw email text for machine learning processing
    
    Steps:
    1. Convert to string (handle any non-string inputs)
    2. Remove email subject headers
    3. Remove URLs and web addresses
    4. Remove email addresses
    5. Remove standalone numbers
    6. Normalize whitespace
    7. Remove punctuation
    8. Remove stopwords
    """
    # Ensure input is a string (handle None, numbers, etc.)
    text = str(text)
    
    # Remove "Subject:" headers (common in email data)
    text = re.sub(r'\bSubject\b', '', text, flags=re.IGNORECASE)
    
    # Remove URLs and web addresses (http://, https://, www.)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    
    # Remove email addresses (pattern: word@word.word)
    text = re.sub(r'\S+@\S+', ' ', text)
    
    # Remove standalone numbers (spam often contains many numbers)
    text = re.sub(r'\b\d+\b', ' ', text)
    
    # Normalize multiple spaces/tabs/newlines to single spaces and trim
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove punctuation marks
    text = remove_punct(text)
    
    # Remove common stopwords
    text = remove_stopwords(text)
    
    return text
