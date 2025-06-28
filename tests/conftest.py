import pytest
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from recommender.preprocessing import DataPreprocessor
from recommender.models import ImplicitRecommender

@pytest.fixture
def sample_data():
    """Sample implicit feedback data."""
    return pd.DataFrame({
        'user_id': [1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4],
        'item_id': [101, 102, 101, 103, 102, 103, 104, 101, 102, 103, 105],
        'feedback_type': ['click', 'view', 'purchase', 'view', 'click', 'view', 'purchase', 
                        'click', 'view', 'purchase', 'wishlist'],
        'confidence': [2.0, 1.0, 5.0, 1.0, 2.0, 1.0, 5.0, 2.0, 1.0, 5.0, 3.0]
    })

@pytest.fixture
def preprocessor(sample_data):
    """Preprocessor fitted on sample data."""
    preprocessor = DataPreprocessor(min_user_interactions=1, min_item_interactions=1)
    preprocessor.fit(sample_data)
    return preprocessor

@pytest.fixture
def user_item_matrix(preprocessor, sample_data):
    """User-item matrix from sample data."""
    return preprocessor.transform(sample_data)

@pytest.fixture
def trained_model(user_item_matrix):
    """Trained recommendation model."""
    model = ImplicitRecommender(model_type='als', factors=2, iterations=5)
    model.fit(user_item_matrix)
    return model