import numpy as np

class EXP3:
    def __init__(self, n_arms, gamma=0.1):
        self.n_arms = n_arms
        self.gamma = gamma
        self.weights = np.ones(n_arms)
    
    def select_arm(self, contexts):
        total_weight = np.sum(self.weights)
        probs = (1 - self.gamma) * (self.weights / total_weight) + (self.gamma / self.n_arms)
        return np.random.choice(np.arange(self.n_arms), p=probs)
    
    def update(self, chosen_arm, reward, context):
        total_weight = np.sum(self.weights)
        probs = (1 - self.gamma) * (self.weights / total_weight) + (self.gamma / self.n_arms)
        estimated_reward = reward / probs[chosen_arm]
        self.weights[chosen_arm] *= np.exp(self.gamma * estimated_reward / self.n_arms)