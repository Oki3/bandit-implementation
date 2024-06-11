import numpy as np
from Policies.BasePolicy import BasePolicy

class EXP3(BasePolicy):
    def __init__(self, n_arms, gamma=0.1):
        self.n_arms = n_arms
        self.gamma = gamma
        self.log_weights = np.zeros(n_arms)  # Initialize log weights to zero
    
    def select_arm(self, contexts):
        # Normalize log weights to prevent overflow
        max_log_weight = np.max(self.log_weights)
        normalized_log_weights = self.log_weights - max_log_weight
        weights = np.exp(normalized_log_weights)  # Convert log weights to weights
        total_weight = np.sum(weights)
        probs = (1 - self.gamma) * (weights / total_weight) + (self.gamma / self.n_arms)
        return np.random.choice(np.arange(self.n_arms), p=probs)
    
    def update(self, chosen_arm, reward, context):
        # Normalize log weights to prevent overflow
        max_log_weight = np.max(self.log_weights)
        normalized_log_weights = self.log_weights - max_log_weight
        weights = np.exp(normalized_log_weights)  # Convert log weights to weights
        total_weight = np.sum(weights)
        probs = (1 - self.gamma) * (weights / total_weight) + (self.gamma / self.n_arms)
        estimated_reward = reward / probs[chosen_arm]
        self.log_weights[chosen_arm] += self.gamma * estimated_reward / self.n_arms  # Update log weight
