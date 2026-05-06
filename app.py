"""
Flask REST API for Fake News Detection.
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.predict import FakeNewsDetector

app = Flask(__name__)
CORS(app)
detector = FakeNewsDetector()

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Fake News Detector API', 'version': '1.0.0', 'status': 'running'})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Please provide a "text" field'}), 400
    result = detector.predict(data['text'])
    return jsonify(result)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
