from typing import Dict, List, Optional
import numpy as np
from scipy.sparse import csr_matrix
from collections import defaultdict

class ExplanationGenerator:
    """Generates explanations for recommendations."""
    
    def __init__(self, user_factors: np.ndarray, item_factors: np.ndarray, 
                 user_item_matrix: csr_matrix):
        self.user_factors = user_factors
        self.item_factors = item_factors
        self.user_item_matrix = user_item_matrix
        
    def generate_explanation(self, user_id: int, item_id: int) -> Dict[str, str]:
        """Generate explanation for why an item was recommended to a user."""
        # Get user's previously interacted items
        interacted_items = self.user_item_matrix[user_id].indices
        
        if len(interacted_items) == 0:
            return {
                'type': 'popular_item',
                'reason': "This is a popular item among all users"
            }
        
        # Calculate similarity between recommended item and user's history
        item_vector = self.item_factors[item_id]
        history_vectors = self.item_factors[interacted_items]
        
        similarities = history_vectors.dot(item_vector)
        most_similar_idx = np.argmax(similarities)
        most_similar_item = interacted_items[most_similar_idx]
        
        return {
            'type': 'similar_to_history',
            'reason': f"Similar to item {most_similar_item} you interacted with",
            'similarity_score': float(similarities[most_similar_idx]),
            'most_similar_item': int(most_similar_item)
        }
    
    def generate_diversity_explanation(self, recommendations: List[int]) -> Dict[str, str]:
        """Explain diversity in recommendations."""
        if len(recommendations) < 2:
            return {
                'type': 'single_recommendation',
                'reason': "This is your top recommendation"
            }
            
        # Calculate pairwise similarity between recommended items
        rec_vectors = self.item_factors[recommendations]
        similarity_matrix = rec_vectors.dot(rec_vectors.T)
        np.fill_diagonal(similarity_matrix, 0)
        avg_similarity = np.mean(similarity_matrix)
        
        if avg_similarity < 0.3:
            return {
                'type': 'diverse_recommendations',
                'reason': "We've included a diverse set of recommendations to explore",
                'average_similarity': float(avg_similarity)
            }
        else:
            return {
                'type': 'focused_recommendations',
                'reason': "These recommendations are closely related to your interests",
                'average_similarity': float(avg_similarity)
            }
    
    def generate_user_profile_explanation(self, user_id: int) -> Dict[str, str]:
        """Generate explanation based on user's profile vector."""
        if self.user_factors is None:
            return {
                'type': 'generic_explanation',
                'reason': "Recommendations based on your activity"
            }
            
        user_vector = self.user_factors[user_id]
        strongest_dimensions = np.argsort(-np.abs(user_vector))[:3]
        
        dimension_descriptions = {
            0: "preference for popular items",
            1: "interest in new releases",
            2: "tendency toward discounted items",
            3: "preference for premium items",
            4: "interest in seasonal products"
        }
        
        traits = []
        for dim in strongest_dimensions:
            if user_vector[dim] > 0:
                trait = f"strong {dimension_descriptions.get(dim, 'interest')}"
            else:
                trait = f"avoidance of {dimension_descriptions.get(dim, 'certain items')}"
            traits.append(trait)
            
        return {
            'type': 'profile_based',
            'reason': "Recommendations based on your: " + ", ".join(traits),
            'strong_dimensions': [int(d) for d in strongest_dimensions]
        }