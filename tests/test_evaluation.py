import pytest
from recommender.evaluation import Evaluator

def test_evaluator_initialization(user_item_matrix):
    """Test evaluator initialization."""
    evaluator = Evaluator(user_item_matrix)
    
    assert evaluator.train_matrix.shape == user_item_matrix.shape
    assert evaluator.test_matrix.shape == user_item_matrix.shape
    assert evaluator.train_matrix.sum() < user_item_matrix.sum()
    assert evaluator.test_matrix.sum() > 0

def test_precision_at_k(evaluator, trained_model):
    """Test precision@k calculation."""
    precision = evaluator.precision_at_k(trained_model, K=2)
    assert 0 <= precision <= 1

def test_recall_at_k(evaluator, trained_model):
    """Test recall@k calculation."""
    recall = evaluator.recall_at_k(trained_model, K=2)
    assert 0 <= recall <= 1

def test_map_at_k(evaluator, trained_model):
    """Test MAP@k calculation."""
    map_score = evaluator.mean_average_precision(trained_model, K=2)
    assert 0 <= map_score <= 1

def test_coverage(evaluator, trained_model):
    """Test coverage calculation."""
    coverage = evaluator.coverage(trained_model, K=2)
    assert 0 < coverage <= 1