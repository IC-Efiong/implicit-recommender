import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from typing import Dict, Tuple, Union
import logging

#MIN_USER_INTERACTIONS = 1
#MIN_ITEM_INTERACTIONS = 1

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
        """Enhanced fit with consistent user ordering"""
        logger.info("Fitting preprocessor...")
        
        # Ensure consistent ordering
        user_counts = df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= self.min_user_interactions].index.sort_values()
        
        item_counts = df['item_id'].value_counts()
        valid_items = item_counts[item_counts >= self.min_item_interactions].index.sort_values()

        self.user_mapping = {u: i for i, u in enumerate(valid_users)}
        self.item_mapping = {i: j for j, i in enumerate(valid_items)}
        
        self.inverse_user_mapping = {v: k for k, v in self.user_mapping.items()}
        self.inverse_item_mapping = {v: k for k, v in self.item_mapping.items()}
        
        logger.info(f"Fitted {len(self.user_mapping)} users and {len(self.item_mapping)} items")

    def transform(self, df: pd.DataFrame) -> csr_matrix:
        """Guaranteed to maintain all fitted users"""
        if not self.user_mapping:
            raise ValueError("Preprocessor must be fitted first")
            
        # Filter to known users/items
        valid_mask = (df['user_id'].isin(self.user_mapping)) & (df['item_id'].isin(self.item_mapping))
        filtered_df = df[valid_mask].copy()
        
        # Create sparse matrix with ALL users (even if no interactions)
        rows = [self.user_mapping[u] for u in filtered_df['user_id']]
        cols = [self.item_mapping[i] for i in filtered_df['item_id']]
        values = filtered_df['confidence'].values
        
        matrix = csr_matrix(
            (values, (rows, cols)),
            shape=(len(self.user_mapping), len(self.item_mapping)),
            dtype=np.float32
        )
        
        # Verify all users are represented
        present_users = set(np.unique(matrix.nonzero()[0]))
        expected_users = set(range(len(self.user_mapping)))
        if len(expected_users - present_users) > 0:
            logger.warning(f"{len(expected_users - present_users)} users have no interactions")
        
        return matrix
    
    def fit_transform(self, df: pd.DataFrame) -> csr_matrix:
        """Combined fit and transform operation"""
        self.fit(df)
        return self.transform(df)

    def transform_new_user(self, interactions: pd.DataFrame) -> csr_matrix:
        """Transform for cold start users"""
        if not self.item_mapping:
            raise ValueError("Preprocessor must be fitted first")
            
        valid_items = interactions['item_id'].isin(self.item_mapping)
        filtered = interactions[valid_items]
        
        cols = [self.item_mapping[i] for i in filtered['item_id']]
        values = filtered['confidence'].values
        
        return csr_matrix(
            (values, ([0]*len(cols), cols)),
            shape=(1, len(self.item_mapping))
        )