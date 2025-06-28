import numpy as np
import pandas as pd
from pathlib import Path

# Configuration moved here from config.py
DATA_FILE = Path(__file__).parent.parent / 'data' / 'sample_data.csv'

def generate_implicit_data(num_users=1000, num_items=500, sparsity=0.95):
    """Generate synthetic implicit feedback data with realistic patterns."""
    np.random.seed(42)
    
    # Create user segments
    user_segments = np.random.choice(['casual', 'regular', 'power'], 
                                   size=num_users, 
                                   p=[0.6, 0.3, 0.1])
    
    # Create item categories
    item_categories = np.random.choice(['electronics', 'clothing', 'home', 'books'],
                                     size=num_items,
                                     p=[0.3, 0.3, 0.2, 0.2])
    
    # Generate interactions based on segments and categories
    interactions = []
    for user_id in range(num_users):
        segment = user_segments[user_id]
        
        # Determine base number of interactions
        if segment == 'casual':
            num_interactions = np.random.poisson(3)
        elif segment == 'regular':
            num_interactions = np.random.poisson(10)
        else:  # power user
            num_interactions = np.random.poisson(30)
            
        # Generate interactions
        for _ in range(num_interactions):
            # Bias items based on user segment
            if segment == 'casual':
                item_probs = [0.4, 0.3, 0.2, 0.1]
            elif segment == 'regular':
                item_probs = [0.3, 0.3, 0.2, 0.2]
            else:
                item_probs = [0.2, 0.2, 0.3, 0.3]
                
            category = np.random.choice(['electronics', 'clothing', 'home', 'books'], p=item_probs)
            item_mask = (item_categories == category)
            available_items = np.where(item_mask)[0]
            
            if len(available_items) > 0:
                item_id = np.random.choice(available_items)
                
                # Determine feedback type
                feedback_types = ['view', 'click', 'purchase', 'wishlist']
                weights = [0.6, 0.3, 0.08, 0.02]
                feedback = np.random.choice(feedback_types, p=weights)
                
                # Set confidence based on feedback
                confidence_map = {'view': 1, 'click': 2, 'wishlist': 3, 'purchase': 5}
                confidence = confidence_map[feedback] * np.random.uniform(0.8, 1.2)
                
                interactions.append({
                    'user_id': user_id,
                    'item_id': item_id,
                    'feedback_type': feedback,
                    'confidence': confidence,
                    'user_segment': segment,
                    'item_category': category
                })
    
    return pd.DataFrame(interactions)

def main():
    """Generate and save sample data."""
    print("Generating synthetic implicit feedback data...")
    data = generate_implicit_data()
    
    # Ensure data directory exists
    DATA_FILE.parent.mkdir(exist_ok=True)
    data.to_csv(DATA_FILE, index=False)
    print(f"Data generated with {len(data)} interactions. Saved to {DATA_FILE}")

if __name__ == "__main__":
    main()