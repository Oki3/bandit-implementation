import numpy as np
import numpy.random as rn
from Policies.BasePolicy import BasePolicy

ETA = 0.1
GAMMA = 0.1
BETA = 0.5
M = 31*4*3
log_weight_enabled = True

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
        if not log_weight_enabled:
            weights = np.zeros(self.k)
            for a in range(self.k):
                weights[a] = self.eta * np.dot(contexts[a], self.cumulative_theta_hats[a])

            sum_weights = np.sum(weights)
            probabilities = (1 - self.gamma) * (weights / sum_weights) + self.gamma / self.k
            return rn.choice(self.k, p=probabilities)
        else:
            log_weights = np.zeros(self.k)
            for a in range(self.k):
                log_weights[a] = self.eta * np.dot(contexts[a], self.cumulative_theta_hats[a])
            
            # Use log-sum-exp trick for numerical stability
            max_log_weight = np.max(log_weights)
            log_weights -= max_log_weight  # Prevent overflow
            weights = np.exp(log_weights)
            
            # Compute probabilities
            sum_weights = np.sum(weights)
            if sum_weights == 0 or np.isnan(sum_weights) or np.isinf(sum_weights):
                print("Sum of weights is zero, NaN, or inf. Weights:", weights)
                raise ValueError("Sum of weights is zero, NaN, or inf")

            probabilities = (1 - self.gamma) * (weights / sum_weights) + self.gamma / self.k

            # Check for NaN values in probabilities
            if np.any(np.isnan(probabilities)):
                print("NaN values detected in probabilities:")
                print("Log Weights:", log_weights)
                print("Weights:", weights)
                print("Sum of weights:", sum_weights)
                print("Probabilities:", probabilities)
                raise ValueError("Probabilities contain NaN values")

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
