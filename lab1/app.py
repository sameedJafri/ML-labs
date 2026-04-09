"""
Flask + Waitress inference server for the sentiment analysis model.

Secrets loaded from .env via python-dotenv (never hard-coded).

Endpoints:
    POST /predict  -> JSON body {"text": "..."}
                  <- {"prediction": [...]}
"""

import os

# Prevent PyTorch/MKL deadlocks before loading transformers
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from flask import Flask, request, jsonify
from transformers import pipeline
from waitress import serve
from dotenv import load_dotenv

load_dotenv()

# Secrets / config
PORT = int(os.getenv("PORT", 5000))
MODEL_NAME = os.getenv("MODEL_NAME", "distilbert-base-uncased-finetuned-sst-2-english")
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")  # optional: needed for gated/private models

app = Flask(__name__)

print(f"Loading model '{MODEL_NAME}'...")
classifier = pipeline(
    "sentiment-analysis",
    model=MODEL_NAME,
    token=HF_TOKEN or None,
    device="cpu",  # force CPU - avoids WSL2 ghost-GPU issues
)
print("Model loaded successfully.")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "No 'text' field provided"}), 400

        result = classifier(data["text"])
        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print(f"Starting server on port {PORT}...")
    serve(app, host="0.0.0.0", port=PORT, threads=4)
