"""Generate synthetic implicit feedback data.
Run with: python -m data.generate_data
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_FILE = Path(__file__).parent.parent / "data" / "sample_data.csv"

def generate_data(num_users=1000, num_items=500):
    """Generate realistic implicit feedback data"""
    np.random.seed(42)
    
    # User segments
    user_segments = np.random.choice(
        ['casual', 'regular', 'power'],
        size=num_users,
        p=[0.6, 0.3, 0.1]
    )
    
    # Item categories
    item_categories = np.random.choice(
        ['electronics', 'clothing', 'home', 'books'],
        size=num_items,
        p=[0.3, 0.3, 0.2, 0.2]
    )
    
    interactions = []
    for user_id in range(num_users):
        segment = user_segments[user_id]
        num_interactions = np.random.poisson(
            3 if segment == 'casual' else 
            10 if segment == 'regular' else 
            30
        )
        
        for _ in range(num_interactions):
            item_id = np.random.randint(num_items)
            feedback = np.random.choice(
                ['view', 'click', 'purchase', 'wishlist'],
                p=[0.6, 0.3, 0.08, 0.02]
            )
            confidence = {
                'view': 1,
                'click': 2,
                'wishlist': 3,
                'purchase': 5
            }[feedback] * np.random.uniform(0.8, 1.2)
            
            interactions.append({
                'user_id': user_id,
                'item_id': item_id,
                'feedback_type': feedback,
                'confidence': confidence,
                'user_segment': segment,
                'item_category': item_categories[item_id]
            })
    
    return pd.DataFrame(interactions)

if __name__ == "__main__":
    logger.info("Generating sample data...")
    df = generate_data()
    DATA_FILE.parent.mkdir(exist_ok=True)
    df.to_csv(DATA_FILE, index=False)
    logger.info(f"Saved {len(df)} interactions to {DATA_FILE}")