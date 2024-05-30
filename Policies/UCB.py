import numpy as np

class UCB:
    def __init__(self, n_arms, alpha=1.0):
        self.n_arms = n_arms
        self.alpha = alpha
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
    
    def select_arm(self):
        total_counts = np.sum(self.counts)
        if total_counts < self.n_arms:
            return int(total_counts)
        ucb_values = self.values + self.alpha * np.sqrt(np.log(total_counts + 1) / (self.counts + 1e-5))
        return int(np.argmax(ucb_values))
    
    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[chosen_arm] = new_value