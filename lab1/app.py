import os
# Force PyTorch/MKL to use a single thread to prevent deadlocks BEFORE loading transformers
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from flask import Flask, request, jsonify
from transformers import pipeline
from waitress import serve

app = Flask(__name__)

# Load the pretrained model
print("Loading model...")
# The device="cpu" flag forces PyTorch to ignore WSL2's ghost GPU
classifier = pipeline(
    "sentiment-analysis", 
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device="cpu" 
)
print("Model loaded successfully.")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text field provided'}), 400
        
        result = classifier(data['text'])
        return jsonify({'prediction': result})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting server on port {port}...")
    serve(app, host='0.0.0.0', port=port)