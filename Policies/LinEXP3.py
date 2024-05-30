import numpy as np
import numpy.random as rn
from Policies.BasePolicy import BasePolicy

ETA = 0.1
GAMMA = 0.1
BETA = 0.5
M = 1200

class LinEXP3(BasePolicy):
    """
    The linEXP3 contextual bandit policy with a sophisticated estimator.
    """

    def __init__(self, n_arms, dimension, m=M, beta=BETA, eta=ETA, gamma=GAMMA, lower=0., amplitude=1.):
        super().__init__(n_arms)
        assert eta > 0, "Error: the 'eta' parameter for the LinEXP3 class must be greater than 0"
        assert 0 < gamma < 1, "Error: the 'gamma' parameter must be in the range (0, 1)"
        assert 0 < beta < 1, "Error: the 'beta' parameter must be in the range (0, 1)"
        assert dimension > 0, "Error: the 'dimension' parameter for the LinEXP3 class must be greater than 0"

        self.M = m
        self.beta = beta
        self.eta = eta
        self.gamma = gamma
        self.dimension = dimension
        self.k = n_arms
        self.cumulative_theta_hats = np.zeros((n_arms, dimension))

    def __str__(self):
        return r"linEXP3($\eta: {:.3g}$, $\gamma: {:.3g}$)".format(self.eta, self.gamma)

    def update(self, arm, reward, context):
        """Update the parameter estimates for the chosen arm."""
        Sigma_plus_t_a = self.matrix_geometric_resampling(context)
        theta_hats = np.dot(Sigma_plus_t_a, context[arm]) * reward
        self.cumulative_theta_hats[arm] += theta_hats

    def select_arm(self, contexts):
        """Choose an arm based on the LINEXP3 policy."""
        weights = np.zeros(self.k)
        for a in range(self.k):
            weights[a] = np.exp(self.eta * np.dot(contexts[a], self.cumulative_theta_hats[a]))

        # Compute probabilities
        sum_weights = np.sum(weights)
        probabilities = (1 - self.gamma) * (weights / sum_weights) + self.gamma / self.k

        return rn.choice(self.k, p=probabilities)
    
    def matrix_geometric_resampling(self, context):
        """Perform MGR procedure"""
        A = [np.eye(self.dimension) for _ in range(self.k)] 
        for _ in range(self.M):
            r = np.random.randint(0, self.k)
            context_drawn = context[r]
            B_k = np.outer(context_drawn, context_drawn)
            A[r] = np.dot(A[r], (np.eye(self.dimension) - self.beta * B_k))

        Sigma_plus_t_a = self.beta * np.eye(self.dimension) + self.beta * sum(A)
        return Sigma_plus_t_a

