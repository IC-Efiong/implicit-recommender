"""Recommendation models.
Run tests with: pytest tests/test_models.py
"""

import implicit
import numpy as np
from scipy.sparse import csr_matrix
from implicit.nearest_neighbours import bm25_weight
import logging

logger = logging.getLogger(__name__)

class ImplicitRecommender:
    def __init__(self, model_type='als', factors=64, regularization=0.01, iterations=15):
        self.model_type = model_type.lower()
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.model = None
        self.user_factors = None
        self.item_factors = None

    def fit(self, user_item_matrix: csr_matrix) -> None:
        """Train the recommendation model"""
        weighted_matrix = bm25_weight(user_item_matrix, K1=100, B=0.8)
        
        if self.model_type == 'als':
            self.model = implicit.als.AlternatingLeastSquares(
                factors=self.factors,
                regularization=self.regularization,
                iterations=self.iterations,
                random_state=42
            )
        elif self.model_type == 'bpr':
            self.model = implicit.bpr.BayesianPersonalizedRanking(
                factors=self.factors,
                regularization=self.regularization,
                iterations=self.iterations,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        self.model.fit(weighted_matrix)
        
        if hasattr(self.model, 'user_factors'):
            self.user_factors = self.model.user_factors
            self.item_factors = self.model.item_factors

    def recommend(self, user_id: int, user_items: csr_matrix, N=10) -> list:
        """Generate recommendations for a user"""
        if self.model is None:
            raise ValueError("Model not trained")
        if user_id >= user_items.shape[0]:
            raise ValueError(f"User ID {user_id} out of bounds (max: {user_items.shape[0]-1})")
            
        # Corrected recommendation call - using positional arguments
        return self.model.recommend(
            user_id,  # Positional argument
            user_items,  # Positional argument
            N=N,
            filter_already_liked_items=True
        )

    def similar_items(self, item_id: int, N=10) -> list:
        """Find similar items"""
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model.similar_items(item_id, N=N)