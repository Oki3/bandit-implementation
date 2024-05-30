import numpy as np

class LinUCB:
    def __init__(self, n_arms, d, alpha=0.1):
        self.n_arms = n_arms
        self.alpha = alpha
        self.A = [np.identity(d) for _ in range(n_arms)]
        self.b = [np.zeros(d) for _ in range(n_arms)]
    
    def select_arm(self, contexts):
        p = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[arm])
            theta = np.dot(A_inv, self.b[arm])
            p[arm] = np.dot(theta, contexts[arm]) + self.alpha * np.sqrt(np.dot(contexts[arm], np.dot(A_inv, contexts[arm])))
        return int(np.argmax(p))
    
    def update(self, chosen_arm, reward, context):
        self.A[chosen_arm] += np.outer(context, context)
        self.b[chosen_arm] += reward * context