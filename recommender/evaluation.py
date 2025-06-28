from typing import Dict, List, Tuple
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Evaluation parameters
TEST_SIZE = 0.2
K_VALUES = [5, 10, 20]

class Evaluator:
    """Evaluates recommendation system performance using various metrics."""
    
    def __init__(self, user_item_matrix: csr_matrix):
        self.user_item_matrix = user_item_matrix
        self.train_matrix, self.test_matrix = self._train_test_split()
        
    def _train_test_split(self) -> Tuple[csr_matrix, csr_matrix]:
        """Split data into train and test sets while preserving user interactions."""
        train = self.user_item_matrix.copy().tocoo()
        test = self.user_item_matrix.copy().tocoo()
        
        # For each user, split their interactions
        user_interactions = defaultdict(list)
        for u, i, v in zip(train.row, train.col, train.data):
            user_interactions[u].append((i, v))
            
        train_data = []
        test_data = []
        
        for u, interactions in user_interactions.items():
            if len(interactions) > 1:
                items, values = zip(*interactions)
                train_idx, test_idx = train_test_split(
                    range(len(items)),
                    test_size=TEST_SIZE,
                    random_state=42)
                
                # Add to train data
                for idx in train_idx:
                    train_data.append((u, items[idx], values[idx]))
                
                # Add to test data
                for idx in test_idx:
                    test_data.append((u, items[idx], values[idx]))
            else:
                # For users with only one interaction, keep in train
                train_data.append((u, interactions[0][0], interactions[0][1]))
                
        # Create new sparse matrices
        train_rows, train_cols, train_vals = zip(*train_data) if train_data else ([], [], [])
        test_rows, test_cols, test_vals = zip(*test_data) if test_data else ([], [], [])
        
        train_matrix = csr_matrix(
            (train_vals, (train_rows, train_cols)),
            shape=self.user_item_matrix.shape)
            
        test_matrix = csr_matrix(
            (test_vals, (test_rows, test_cols)),
            shape=self.user_item_matrix.shape)
            
        return train_matrix, test_matrix
    
    def precision_at_k(self, model, K: int = 10) -> float:
        """Calculate precision@K."""
        precisions = []
        
        for u in range(self.train_matrix.shape[0]):
            if self.test_matrix[u].sum() == 0:
                continue  # Skip users with no test items
                
            recommended = model.recommend(u, self.train_matrix, N=K)
            recommended_items = {i for i, _ in recommended}
            
            actual_items = set(self.test_matrix[u].indices)
            
            hits = len(recommended_items & actual_items)
            precisions.append(hits / K)
            
        return np.mean(precisions) if precisions else 0.0
    
    def recall_at_k(self, model, K: int = 10) -> float:
        """Calculate recall@K."""
        recalls = []
        
        for u in range(self.train_matrix.shape[0]):
            if self.test_matrix[u].sum() == 0:
                continue
                
            recommended = model.recommend(u, self.train_matrix, N=K)
            recommended_items = {i for i, _ in recommended}
            
            actual_items = set(self.test_matrix[u].indices)
            
            hits = len(recommended_items & actual_items)
            recalls.append(hits / len(actual_items))
            
        return np.mean(recalls) if recalls else 0.0
    
    def mean_average_precision(self, model, K: int = 10) -> float:
        """Calculate MAP@K."""
        average_precisions = []
        
        for u in range(self.train_matrix.shape[0]):
            if self.test_matrix[u].sum() == 0:
                continue
                
            recommended = model.recommend(u, self.train_matrix, N=K)
            recommended_items = [i for i, _ in recommended]
            actual_items = set(self.test_matrix[u].indices)
            
            hits = 0
            sum_precisions = 0.0
            
            for n, item in enumerate(recommended_items, 1):
                if item in actual_items:
                    hits += 1
                    sum_precisions += hits / n
                    
            if hits > 0:
                avg_prec = sum_precisions / min(len(actual_items), K)
                average_precisions.append(avg_prec)
                
        return np.mean(average_precisions) if average_precisions else 0.0
    
    def coverage(self, model, K: int = 10) -> float:
        """Calculate catalog coverage@K."""
        all_items = set(range(self.train_matrix.shape[1]))
        recommended_items = set()
        
        for u in range(self.train_matrix.shape[0]):
            recommended = model.recommend(u, self.train_matrix, N=K)
            recommended_items.update({i for i, _ in recommended})
            
        return len(recommended_items) / len(all_items)
    
    def evaluate(self, model, K_values: List[int] = None) -> Dict[str, float]:
        """Run full evaluation with multiple K values."""
        K_values = K_values or K_VALUES
        results = {}
        
        for K in K_values:
            results[f'precision@{K}'] = self.precision_at_k(model, K)
            results[f'recall@{K}'] = self.recall_at_k(model, K)
            results[f'map@{K}'] = self.mean_average_precision(model, K)
            results[f'coverage@{K}'] = self.coverage(model, K)
            
        return results