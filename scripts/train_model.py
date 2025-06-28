"""Train and save recommendation model.
Run with: python -m scripts.train_model
"""

import pandas as pd
import pickle
from pathlib import Path
from recommender.preprocessing import DataPreprocessor
from recommender.models import ImplicitRecommender
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_FILE = Path(__file__).parent.parent / "data" / "sample_data.csv"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_FILE = MODEL_DIR / "model.pkl"
PREPROCESSOR_FILE = MODEL_DIR / "preprocessor.pkl"

def train_model():
    try:
        logger.info("Loading data...")
        df = pd.read_csv(DATA_FILE)
        
        logger.info("Preprocessing data...")
        preprocessor = DataPreprocessor(min_user_interactions=3)
        user_item_matrix = preprocessor.fit_transform(df)
        
        logger.info(f"Training matrix shape: {user_item_matrix.shape}")
        
        logger.info("Training ALS model...")
        model = ImplicitRecommender(model_type='als', factors=64)
        model.fit(user_item_matrix)
        
        logger.info("Saving artifacts...")
        MODEL_DIR.mkdir(exist_ok=True)
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(model, f)
        with open(PREPROCESSOR_FILE, 'wb') as f:
            pickle.dump(preprocessor, f)
            
        logger.info("Training complete!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    train_model()