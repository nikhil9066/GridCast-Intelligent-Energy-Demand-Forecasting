
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

app = Flask(__name__)

# Load model at startup
try:
    model = joblib.load('models/xgboost_model.pkl')
    feature_cols = joblib.load('models/feature_columns.pkl')
    print("Model loaded successfully!")
except:
    model = None
    print("Model loading failed!")

@app.route('/predict', methods=['GET'])
def predict():
    try:
        hours = int(request.args.get('hours', 24))

        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        # Simplified prediction (in practice, load recent data)
        predictions = np.random.normal(30000, 5000, hours).tolist()

        # Generate timestamps
        start_time = datetime.now()
        timestamps = [(start_time + timedelta(hours=i)).isoformat() 
                     for i in range(hours)]

        return jsonify({
            'predictions': predictions,
            'timestamps': timestamps,
            'hours_ahead': hours
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
