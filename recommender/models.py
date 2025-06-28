from typing import List, Tuple, Optional
import implicit
import numpy as np
from scipy.sparse import csr_matrix
from implicit.nearest_neighbours import bm25_weight

MODEL_TYPE = 'als'
FACTORS = 64
REGULARIZATION = 0.01
ITERATIONS = 15

class ImplicitRecommender:
    """Implicit feedback recommendation system with multiple algorithm support."""
    
    def __init__(self, model_type = MODEL_TYPE, factors = FACTORS, 
                 regularization = REGULARIZATION, iterations = ITERATIONS):
        self.model_type = model_type or MODEL_TYPE
        self.factors = factors or FACTORS
        self.regularization = regularization or REGULARIZATION
        self.iterations = iterations or ITERATIONS
        
        self.model = None
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        
    def fit(self, user_item_matrix: csr_matrix) -> None:
        """Train the recommendation model."""
        # Apply BM25 weighting to reduce popularity bias
        weighted_matrix = bm25_weight(user_item_matrix, K1=100, B=0.8)
        
        if self.model_type == 'als':
            self.model = implicit.als.AlternatingLeastSquares(
                factors=self.factors,
                regularization=self.regularization,
                iterations=self.iterations,
                random_state=42)
        elif self.model_type == 'bpr':
            self.model = implicit.bpr.BayesianPersonalizedRanking(
                factors=self.factors,
                regularization=self.regularization,
                iterations=self.iterations,
                random_state=42)
        elif self.model_type == 'lmf':
            self.model = implicit.lmf.LogisticMatrixFactorization(
                factors=self.factors,
                regularization=self.regularization,
                iterations=self.iterations,
                random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        self.model.fit(weighted_matrix)
        
        # Store factors for explanation and cold start
        if hasattr(self.model, 'user_factors'):
            self.user_factors = self.model.user_factors
            self.item_factors = self.model.item_factors
    
    def recommend(self, user_id, user_item_matrix, N=10, filter_already_liked=True):
        """Generate recommendations with additional validation"""
        if self.model is None:
            raise ValueError("Model must be fitted before making recommendations")
        
        if user_id >= user_item_matrix.shape[0]:
            raise ValueError(
                f"User ID {user_id} is out of bounds for matrix with {user_item_matrix.shape[0]} users")
        
        return self.model.recommend(
            user_id, 
            user_item_matrix, 
            N=N,
            filter_already_liked_items=filter_already_liked)
    
    def similar_items(self, item_id: int, N: int = 10) -> List[Tuple[int, float]]:
        """Find similar items to the given item."""
        if self.model is None:
            raise ValueError("Model must be fitted before finding similar items")
        return self.model.similar_items(item_id, N=N)
    
    def explain(self, user_id: int, item_id: int, 
               user_item_matrix: csr_matrix) -> List[Tuple[int, float]]:
        """Explain why an item was recommended to a user."""
        if not hasattr(self.model, 'explain'):
            raise NotImplementedError(f"Explanation not supported for {self.model_type}")
        
        _, contributions = self.model.explain(
            user_id,
            user_item_matrix,
            itemid=item_id)
        return contributions