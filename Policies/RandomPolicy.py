import numpy as np
from BasePolicy import BasePolicy

class RandomPolicy(BasePolicy):
    def __init__(self, n_arms, d):
        super().__init__(n_arms)
        self.A = [np.identity(d) for _ in range(n_arms)]
        self.b = [np.zeros(d) for _ in range(n_arms)]
    
    def select_arm(self, contexts):
        return np.random.randint(self.n_arms)
    
    def update(self, chosen_arm, reward, context):
        self.A[chosen_arm] += np.outer(context, context)
        self.b[chosen_arm] += reward * context
