# app.py
# Simple Flask API for predictions (Roman Urdu comments)

import os
import sys
import json
import pickle
from flask import Flask, request, jsonify

# path setup for imports/artifacts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

from utils_text import basic_clean
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

ART_DIR = os.path.join(PROJ_ROOT, "artifacts")
MODEL_PATH = os.path.join(ART_DIR, "spam_model.keras")
TOK_PATH = os.path.join(ART_DIR, "tokenizer.pkl")
CFG_PATH = os.path.join(ART_DIR, "config.json")

# load model + tokenizer + config once at startup
model = tf.keras.models.load_model(MODEL_PATH)
with open(TOK_PATH, "rb") as f:
    tokenizer = pickle.load(f)
with open(CFG_PATH, "r") as f:
    cfg = json.load(f)

MAX_LEN = cfg.get("max_len", 100)

app = Flask(__name__)

@app.route("/", methods=["GET"])
def health():
    # health check route
    return jsonify({"status":"ok", "service":"spam-detector"})

@app.route("/predict", methods=["POST"])
def predict():
    # expect JSON: {"text": "your email body"}
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "")
    # clean text same tarike se
    cleaned = basic_clean(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    proba = float(model.predict(pad, verbose=0)[0][0])
    label = "Spam" if proba >= 0.5 else "Ham"
    return jsonify({
        "input": text,
        "cleaned": cleaned,
        "label": label,
        "confidence": round(proba, 4)
    })

if __name__ == "__main__":
    # for local simple serving use waitress (production friendly)
    from waitress import serve
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting on port {port} ...")
    serve(app, host="0.0.0.0", port=port)
