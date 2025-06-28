from typing import List, Optional
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


class ColdStartHandler:
    """Handles recommendation scenarios for new users and items."""
    
    def __init__(self, item_features: Optional[np.ndarray] = None):
        self.item_features = item_features
        self.item_knn: Optional[NearestNeighbors] = None
        self.popular_items: Optional[np.ndarray] = None
        
    def fit(self, item_factors: np.ndarray, user_item_matrix: csr_matrix) -> None:
        """Initialize cold start handler with item factors and popularity."""
        # Fit KNN model for item similarity
        self.item_knn = NearestNeighbors(
            n_neighbors=20,
            metric='cosine',
            algorithm='brute')
        self.item_knn.fit(item_factors)
        
        # Calculate item popularity
        self.popular_items = np.argsort(-np.array(user_item_matrix.sum(axis=0)).flatten())
    
    def recommend_for_new_user(self, initial_interactions: csr_matrix, 
                             item_factors: np.ndarray, N: int = 10) -> List[int]:
        """Recommend items for a new user based on initial interactions."""
        if self.item_knn is None:
            raise ValueError("ColdStartHandler must be fitted first")
            
        if initial_interactions.sum() == 0:
            # No interactions - return popular items
            return self.popular_items[:N].tolist()
            
        # Get interacted items
        interacted_indices = initial_interactions.indices
        if len(interacted_indices) == 0:
            return self.popular_items[:N].tolist()
            
        # Average the item factors of interacted items
        user_vector = np.mean(item_factors[interacted_indices], axis=0)
        
        # Find similar items
        _, indices = self.item_knn.kneighbors(
            user_vector.reshape(1, -1),
            n_neighbors=N)
        
        return indices[0].tolist()
    
    def recommend_for_new_item(self, item_vector: np.ndarray, 
                             user_factors: np.ndarray, N: int = 10) -> List[int]:
        """Recommend users for a new item."""
        if self.item_knn is None:
            raise ValueError("ColdStartHandler must be fitted first")
            
        # Find most similar users to the item vector
        _, indices = self.item_knn.kneighbors(
            item_vector.reshape(1, -1),
            n_neighbors=N)
        
        return indices[0].tolist()
    
    def content_based_recommend(self, item_id: int, N: int = 10) -> List[int]:
        """Content-based recommendations if item features are available."""
        if self.item_features is None:
            raise ValueError("Item features not provided")
            
        target_features = self.item_features[item_id]
        similarities = self.item_features.dot(target_features.T)
        similar_items = np.argsort(-similarities.flatten())[1:N+1]
        return similar_items.tolist()