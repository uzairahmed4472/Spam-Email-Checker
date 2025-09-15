# preprocess.py
# Load Emails.csv, clean text, balance spam/ham

import os
import re
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords

# download stopwords if missing
try:
    _ = stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

PUNCT_TABLE = str.maketrans('', '', string.punctuation)
STOPWORDS = set(stopwords.words('english'))

# ------------------------
# Text Cleaning Functions
# ------------------------

def remove_punct(text: str) -> str:
    """Remove punctuation"""
    return text.translate(PUNCT_TABLE)

def remove_stopwords(text: str) -> str:
    """Remove common stopwords like 'the', 'is'"""
    tokens = [w.lower() for w in text.split() if w.lower() not in STOPWORDS]
    return " ".join(tokens)

def basic_clean(text: str) -> str:
    """Basic text cleaning"""
    text = str(text)
    text = re.sub(r'\bSubject\b', '', text, flags=re.IGNORECASE)  # remove "Subject"
    text = re.sub(r'http\S+|www\.\S+', ' ', text)                 # remove URLs
    text = re.sub(r'\S+@\S+', ' ', text)                          # remove emails
    text = re.sub(r'\b\d+\b', ' ', text)                          # remove numbers
    text = re.sub(r'\s+', ' ', text).strip()                      # normalize spaces
    text = remove_punct(text)                                     # remove punctuation
    text = remove_stopwords(text)                                 # remove stopwords
    return text

# ------------------------
# Dataset Functions
# ------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_PATH = os.path.join(DATA_DIR, "Emails.csv")
PROCESSED_CSV = os.path.join(DATA_DIR, "processed.csv")

def load_dataset():
    """Load Emails.csv and ensure columns are (text, label)"""
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError("Emails.csv not found in data/ folder")
    
    df = pd.read_csv(RAW_PATH, encoding="latin-1")
    if "text" in df.columns and "label" in df.columns:
        return df[["text", "label"]]
    else:
        raise ValueError(f"Emails.csv must contain 'text' and 'label' columns, found: {df.columns}")

def balance_downsample(df: pd.DataFrame) -> pd.DataFrame:
    """Downsample ham to match spam count"""
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(["spam", "ham", "0", "1"])]
    df["label"] = df["label"].replace({"0": "ham", "1": "spam"})

    ham = df[df["label"] == "ham"]
    spam = df[df["label"] == "spam"]
    if len(spam) == 0 or len(ham) == 0:
        raise ValueError("Dataset must contain both spam and ham")

    ham_bal = ham.sample(n=len(spam), random_state=42)
    balanced = pd.concat([ham_bal, spam]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    return balanced

# ------------------------
# Main
# ------------------------

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    df = load_dataset()
    print("Raw shape:", df.shape)

    df["text"] = df["text"].astype(str)
    df["text_clean"] = df["text"].apply(basic_clean)

    balanced = balance_downsample(df[["text_clean", "label"]].rename(columns={"text_clean": "text"}))
    print("Balanced shape:", balanced.shape)

    balanced.to_csv(PROCESSED_CSV, index=False)
    print("Saved processed:", PROCESSED_CSV)

if __name__ == "__main__":
    main()
