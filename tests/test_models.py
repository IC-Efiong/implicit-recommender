import pytest
import numpy as np
from recommender.models import ImplicitRecommender

def test_model_initialization():
    """Test model initialization with different types."""
    for model_type in ['als', 'bpr', 'lmf']:
        model = ImplicitRecommender(model_type=model_type)
        assert model.model_type == model_type
        assert model.model is None

def test_model_fitting(trained_model, user_item_matrix):
    """Test model training."""
    assert trained_model.model is not None
    assert trained_model.user_factors.shape[0] == user_item_matrix.shape[0]
    assert trained_model.item_factors.shape[0] == user_item_matrix.shape[1]

def test_recommendations(trained_model, user_item_matrix):
    """Test recommendation generation."""
    recommendations = trained_model.recommend(0, user_item_matrix, N=2)
    assert len(recommendations) == 2
    assert all(isinstance(i, tuple) and len(i) == 2 for i in recommendations)

def test_similar_items(trained_model):
    """Test similar items retrieval."""
    similar_items = trained_model.similar_items(0, N=2)
    assert len(similar_items) == 2
    assert all(isinstance(i, tuple) and len(i) == 2 for i in similar_items)