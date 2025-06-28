"""Evaluate recommendation model performance.
Run with: python -m scripts.evaluate_model
"""

import pickle
import logging
from recommender.evaluation import Evaluator
from pathlib import Path
import pandas as pd

MODEL_FILE = Path(__file__).parent.parent / 'models' / 'model.pkl'
PREPROCESSOR_FILE = Path(__file__).parent.parent / 'models' / 'preprocessor.pkl'
DATA_FILE = Path(__file__).parent.parent / 'data' / 'sample_data.csv'
K_VALUES = [5, 10, 20]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate():
    """Evaluate model performance."""
    try:
        logger.info("Starting model evaluation...")
        
        # Load model and preprocessor
        logger.info(f"Loading model from {MODEL_FILE}")
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Loading preprocessor from {PREPROCESSOR_FILE}")
        with open(PREPROCESSOR_FILE, 'rb') as f:
            preprocessor = pickle.load(f)
        
        # Load data
        logger.info(f"Loading data from {DATA_FILE}")
        data = pd.read_csv(DATA_FILE)
        
        # Transform data
        logger.info("Transforming data...")
        user_item_matrix = preprocessor.transform(data)
        
        # Evaluate
        logger.info("Evaluating model...")
        evaluator = Evaluator(user_item_matrix)
        results = evaluator.evaluate(model)
        
        # Print results
        logger.info("\nEvaluation Results:")
        for metric, value in results.items():
            logger.info(f"{metric}: {value:.4f}")
        
        return results
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    evaluate()