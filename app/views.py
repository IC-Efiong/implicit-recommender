from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any
# from recommender.cold_start import ColdStartHandler
# from recommender.explainers import ExplanationGenerator
import logging
    
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_FILE = Path("models/model.pkl")
PREPROCESSOR_FILE = Path("models/preprocessor.pkl")
DATA_FILE = Path("data/sample_data.csv")

# Load artifacts at startup
try:
    with open(PREPROCESSOR_FILE, 'rb') as f:
        preprocessor = pickle.load(f)
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    data = pd.read_csv(DATA_FILE)
    logger.info("Artifacts loaded successfully")
except Exception as e:
    logger.error(f"Failed to load artifacts: {e}")
    preprocessor = None
    model = None
    data = None

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        if model is None or preprocessor is None:
            return jsonify({'status': 'error', 'message': 'Model not loaded'})

        user_id = request.form.get('user_id', '').strip()
        logger.info(f"Recommendation request for user: {user_id}")

        # Existing user
        if user_id.isdigit():
            user_id = int(user_id)
            if user_id not in preprocessor.user_mapping:
                return jsonify({
                    'status': 'error',
                    'message': f'User {user_id} not found',
                    'valid_users_sample': list(preprocessor.user_mapping.keys())[:5]
                })

            internal_id = preprocessor.user_mapping[user_id]
            user_item_matrix = preprocessor.transform(data)
            
            # Critical validation
            if internal_id >= user_item_matrix.shape[0]:
                return jsonify({
                    'status': 'error',
                    'message': 'Internal ID mismatch',
                    'details': {
                        'internal_id': internal_id,
                        'matrix_rows': user_item_matrix.shape[0],
                        'user_count': len(preprocessor.user_mapping)
                    }
                })

            recs = model.recommend(
                userid=internal_id,
                user_items=user_item_matrix,
                N=10
            )
            
            return jsonify({
                'status': 'success',
                'user_type': 'existing',
                'recommendations': [
                    {'item': preprocessor.inverse_item_mapping[i], 'score': float(s)}
                    for i, s in recs
                ]
            })

        # New user handling...
        
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)