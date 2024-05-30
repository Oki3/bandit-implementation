# policies/base_policy.py
class BasePolicy:
    def __init__(self, n_arms):
        self.n_arms = n_arms

    def select_arm(self, context_vectors):
        raise NotImplementedError("This method should be overridden by subclasses")

    def update(self, chosen_arm, reward, context):
        raise NotImplementedError("This method should be overridden by subclasses")
