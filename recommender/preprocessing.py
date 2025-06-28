"""Data preprocessing pipeline.
Run tests with: pytest tests/test_preprocessing.py
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from typing import Dict, Tuple, Union
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, min_user_interactions=1, min_item_interactions=1):
        self.min_user_interactions = min_user_interactions
        self.min_item_interactions = min_item_interactions
        self.user_mapping: Dict[Union[int, str], int] = {}
        self.item_mapping: Dict[Union[int, str], int] = {}
        self.inverse_user_mapping: Dict[int, Union[int, str]] = {}
        self.inverse_item_mapping: Dict[int, Union[int, str]] = {}

    def fit(self, df: pd.DataFrame) -> None:
        """Learn user and item mappings"""
        required_cols = {'user_id', 'item_id', 'confidence'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Missing required columns: {required_cols - set(df.columns)}")
        
        user_counts = df['user_id'].value_counts()
        item_counts = df['item_id'].value_counts()
        
        valid_users = user_counts[user_counts >= self.min_user_interactions].index.sort_values()
        valid_items = item_counts[item_counts >= self.min_item_interactions].index.sort_values()
        
        self.user_mapping = {u: i for i, u in enumerate(valid_users)}
        self.item_mapping = {i: j for j, i in enumerate(valid_items)}
        
        self.inverse_user_mapping = {v: k for k, v in self.user_mapping.items()}
        self.inverse_item_mapping = {v: k for k, v in self.item_mapping.items()}

    def transform(self, df: pd.DataFrame) -> csr_matrix:
        """Transform data ensuring all users are represented"""
        if not self.user_mapping:
            raise ValueError("Preprocessor must be fitted first")

        # Create empty matrix with correct dimensions
        matrix = csr_matrix(
            (len(self.user_mapping), len(self.item_mapping)),
            dtype=np.float32
        )

        # Only fill in interactions for valid user-item pairs
        valid_mask = (
            df['user_id'].isin(self.user_mapping) & 
            df['item_id'].isin(self.item_mapping)
        )
        filtered_df = df[valid_mask]
        
        if not filtered_df.empty:
            rows = [self.user_mapping[u] for u in filtered_df['user_id']]
            cols = [self.item_mapping[i] for i in filtered_df['item_id']]
            values = filtered_df['confidence'].values
            
            # Add interactions to the matrix
            matrix += csr_matrix(
                (values, (rows, cols)),
                shape=matrix.shape
            )

        # Verify all users are present (even with zero interactions)
        assert matrix.shape[0] == len(self.user_mapping), \
            f"Matrix has {matrix.shape[0]} rows but expected {len(self.user_mapping)} users"
        
        return matrix