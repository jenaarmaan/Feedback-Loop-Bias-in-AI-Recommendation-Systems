import numpy as np
import pandas as pd

class Item:
    """Represents a content item (e.g., video, article, product)."""
    def __init__(self, item_id, intrinsic_quality):
        self.id = item_id
        self.quality = intrinsic_quality  # Intrinsic value/utility
        self.exposure = 0  # How many times it has been recommended
        self.clicks = 0    # How many times it was actually selected
        self.category = np.random.choice(['A', 'B', 'C', 'D']) # Simple categories for diversity testing

    def __repr__(self):
        return f"Item(id={self.id}, q={self.quality:.2f}, exp={self.exposure}, clk={self.clicks})"

class User:
    """Represents a user with specific preferences."""
    def __init__(self, user_id):
        self.id = user_id
        # In this simple model, users prefer quality but are also influenced by exposure (social proof/availability)
        # We can also add category affinity if needed.
        self.preference_weights = np.random.dirichlet(np.ones(4), size=1)[0] # Affinity for categories A, B, C, D

    def decide_click(self, recommended_items, feedback_loop_strength=0.5):
        """
        User decides which item to click from the recommended list.
        Utility = (Quality * (1 - strength)) + (Normalized_Exposure * strength)
        """
        if not recommended_items:
            return None
        
        utilities = []
        for item in recommended_items:
            # Simple utility model: Quality + Popularity Bias
            # normalized_exposure = item.exposure / (max([i.exposure for i in recommended_items]) + 1)
            utility = item.quality * (1 - feedback_loop_strength) + (item.exposure * 0.01 * feedback_loop_strength)
            utilities.append(max(0, utility))
            
        prob = np.array(utilities)
        if prob.sum() == 0:
            return np.random.choice(recommended_items)
            
        prob /= prob.sum()
        return np.random.choice(recommended_items, p=prob)

def calculate_gini(exposure_counts):
    """Calculates Gini Coefficient (0 = perfect equality, 1 = absolute inequality)."""
    n = len(exposure_counts)
    if n == 0: return 0
    sorted_counts = np.sort(exposure_counts)
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * sorted_counts)) / (n * np.sum(sorted_counts))

def calculate_diversity(item_stats):
    """Calculates Aggregate Diversity (Number of unique items exposed at least once)."""
    exposed_count = sum(1 for _, exposure, _, _ in item_stats if exposure > 0)
    return exposed_count

class TopKRecommender:
    """A simple recommender that selects the top-K items based on current popularity (clicks) or exposure."""
    def __init__(self, k=5, use_exploration=False, epsilon=0.1):
        self.k = k
        self.use_exploration = use_exploration
        self.epsilon = epsilon

    def recommend(self, items):
        """Recommends top-K items."""
        if self.use_exploration and np.random.random() < self.epsilon:
            # Epsilon-Greedy Exploration: Randomly pick items
            return list(np.random.choice(items, size=self.k, replace=False))
        
        # Naive approach: Sort by clicks (popularity)
        # In early stages where clicks might be 0, we can use quality as a slight prior or just random
        sorted_items = sorted(items, key=lambda x: (x.clicks, x.quality), reverse=True)
        return sorted_items[:self.k]

def run_simulation(num_items=100, num_users=50, cycles=20, k=5, feedback_loop_strength=0.5, mitigation=False):
    """Runs the simulation and returns historical data."""
    # Initialize items
    items = [Item(i, np.random.random()) for i in range(num_items)]
    # Initialize users
    users = [User(i) for i in range(num_users)]
    # Initialize recommender
    recommender = TopKRecommender(k=k, use_exploration=mitigation)
    
    history = []

    for cycle in range(cycles):
        cycle_interactions = []
        for user in users:
            # 1. Get Recommendation
            rec_list = recommender.recommend(items)
            
            # 2. Record Exposure
            for item in rec_list:
                item.exposure += 1
            
            # 3. User Interaction
            clicked_item = user.decide_click(rec_list, feedback_loop_strength=feedback_loop_strength)
            if clicked_item:
                clicked_item.clicks += 1
                cycle_interactions.append((user.id, clicked_item.id))
        
        # Record snapshot for metrics later
        snapshot = {
            'cycle': cycle,
            'item_stats': [(i.id, i.exposure, i.clicks, i.quality) for i in items]
        }
        history.append(snapshot)
        
    return history, items

if __name__ == "__main__":
    # Test simulation loop
    history, final_items = run_simulation(num_items=20, num_users=10, cycles=5)
    print(f"Simulation finished with {len(history)} cycles.")
    print(f"Top item after sim: {sorted(final_items, key=lambda x: x.clicks, reverse=True)[0]}")
