"""Train and save recommendation model.
Run with: python -m scripts.train_model
"""

import pandas as pd
import pickle
from pathlib import Path
from recommender.preprocessing import DataPreprocessor
from recommender.models import ImplicitRecommender
import logging
import numpy as np


DATA_FILE = Path("data/sample_data.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_FILE = MODEL_DIR / "model.pkl"
PREPROCESSOR_FILE = MODEL_DIR / "preprocessor.pkl"
# Model parameters
MODEL_TYPE = 'als'
FACTORS = 64
REGULARIZATION = 0.01
ITERATIONS = 15

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model():
    try:
        logger.info("Loading data...")
        df = pd.read_csv(DATA_FILE)
        logger.info(f"Loaded {len(df)} interactions")

        logger.info("Preprocessing data...")
        preprocessor = DataPreprocessor(min_user_interactions=1)
        user_item_matrix = preprocessor.fit_transform(df)
        
        logger.info(f"Matrix shape: {user_item_matrix.shape}")
        logger.info(f"Non-zero entries: {user_item_matrix.nnz}")

        # Verify all users are represented
        present_users = set(np.unique(user_item_matrix.nonzero()[0]))
        expected_users = set(range(user_item_matrix.shape[0]))
        missing = expected_users - present_users
        if missing:
            logger.warning(f"{len(missing)} users have no interactions")

        logger.info("Training model...")
        model = ImplicitRecommender(model_type='als', factors=64)
        model.fit(user_item_matrix)

        logger.info("Saving artifacts...")
        with open(PREPROCESSOR_FILE, 'wb') as f:
            pickle.dump(preprocessor, f)
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(model, f)
            
        logger.info("Training complete!")
        return True

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    train_model()