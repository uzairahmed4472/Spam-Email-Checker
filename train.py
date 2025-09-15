# train.py
# Tokenize, pad, build LSTM model, train, evaluate, save artifacts
# short Roman Urdu comments har logic block ke sath

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# callbacks
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# base paths set kar lo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

PROJ_ROOT = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(PROJ_ROOT, "data", "processed.csv")
ART_DIR = os.path.join(PROJ_ROOT, "artifacts")
os.makedirs(ART_DIR, exist_ok=True)

# artifacts ke paths
MODEL_PATH = os.path.join(ART_DIR, "spam_model.keras")
TOK_PATH = os.path.join(ART_DIR, "tokenizer.pkl")
CFG_PATH = os.path.join(ART_DIR, "config.json")

# hyperparams
MAX_LEN = 100           # max sequence length
VOCAB_SIZE = 20000      # vocab limit
OOV_TOKEN = "<OOV>"     # unknown words ka token

def load_data():
    # processed csv load karo, labels ko binary (spam=1, ham=0)
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text","label"])
    df["label"] = (df["label"].astype(str).str.lower() == "spam").astype(int)
    X = df["text"].astype(str).tolist()
    y = df["label"].values
    return X, y

def fit_tokenizer(X_train, X_val):
    # tokenizer banaye aur train text pe fit karo
    tok = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
    tok.fit_on_texts(X_train)

    # train/val ko sequences me convert karo
    seq_train = tok.texts_to_sequences(X_train)
    seq_val = tok.texts_to_sequences(X_val)

    # sequences ko pad karo taake fixed length ho
    seq_train = pad_sequences(seq_train, maxlen=MAX_LEN, padding="post", truncating="post")
    seq_val = pad_sequences(seq_val, maxlen=MAX_LEN, padding="post", truncating="post")
    return tok, seq_train, seq_val

def build_model(vocab_size):
    # simple LSTM based model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=32, input_length=MAX_LEN),  # embedding
        tf.keras.layers.LSTM(16),                                                             # LSTM layer
        tf.keras.layers.Dense(32, activation="relu"),                                         # hidden dense
        tf.keras.layers.Dense(1, activation="sigmoid")                                        # output binary
    ])
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model

def main():
    # data load karo
    X, y = load_data()

    # train/val split (stratified)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # tokenizer aur sequences banao
    tok, Xtr, Xva = fit_tokenizer(X_train, X_val)

    # model banao
    model = build_model(vocab_size=min(VOCAB_SIZE, len(tok.word_index)+1))

    # callbacks: early stopping & learning rate reduce
    es = EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)
    rl = ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5)

    # model train karo
    history = model.fit(
        Xtr, y_train,
        validation_data=(Xva, y_val),
        epochs=12,
        batch_size=32,
        callbacks=[es, rl],
        verbose=1
    )

    # validation pe predict karo
    y_proba = model.predict(Xva).ravel()
    y_pred = (y_proba >= 0.5).astype(int)

    # evaluation metrics print karo
    print("\nClassification Report:\n", classification_report(y_val, y_pred, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
    try:
        print("ROC AUC:", round(roc_auc_score(y_val, y_proba),4))
    except Exception:
        pass

    # artifacts save karo (model, tokenizer, config)
    model.save(MODEL_PATH)
    with open(TOK_PATH, "wb") as f:
        pickle.dump(tok, f)
    with open(CFG_PATH, "w") as f:
        json.dump({"max_len": MAX_LEN}, f)

    print("Saved model ->", MODEL_PATH)
    print("Saved tokenizer ->", TOK_PATH)
    print("Saved config ->", CFG_PATH)

if __name__ == "__main__":
    main()
