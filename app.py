# app.py
"""
Flask API for Startup Investment Model Deployment
"""
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import traceback
import json

# Load model and metadata
deployment_meta = json.load(open('investment_deployment_metadata.json'))
model = joblib.load(deployment_meta['model_file'])
features = deployment_meta['features']

app = Flask(__name__)

@app.route('/')
def home():
    return "<h2>Startup Investment Model API is running.</h2>"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        # Ensure all required features are present
        missing = [f for f in features if f not in data]
        if missing:
            return jsonify({'error': f'Missing features: {missing}'}), 400
        X = pd.DataFrame([data], columns=features)
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0, 1]
        return jsonify({
            'prediction': int(pred),
            'decision': 'Invest' if pred == 1 else "Don't Invest",
            'confidence': round(float(proba), 4)
        })
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
