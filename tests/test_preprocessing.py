import pytest
import numpy as np
from scipy.sparse import csr_matrix
from recommender.preprocessing import DataPreprocessor
import pandas as pd

def test_preprocessor_fit(sample_data):
    """Test preprocessor fitting."""
    preprocessor = DataPreprocessor(min_user_interactions=1, min_item_interactions=1)
    preprocessor.fit(sample_data)
    
    assert len(preprocessor.user_mapping) == 4  # 4 users
    assert len(preprocessor.item_mapping) == 5  # 5 items

def test_preprocessor_transform(preprocessor, sample_data):
    """Test data transformation."""
    matrix = preprocessor.transform(sample_data)
    
    assert isinstance(matrix, csr_matrix)
    assert matrix.shape == (4, 5)  # 4 users, 5 items
    assert matrix.sum() == pytest.approx(28.0)  # Sum of all confidences

def test_new_user_transform(preprocessor):
    """Test transformation of new user data."""
    new_user_data = pd.DataFrame({
        'item_id': [101, 103],
        'confidence': [2.0, 1.0]
    })
    
    transformed = preprocessor.transform_new_user(new_user_data)
    
    assert transformed.shape == (1, 5)
    assert transformed.sum() == pytest.approx(3.0)