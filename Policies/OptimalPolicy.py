import numpy as np

from BasePolicy import BasePolicy

class OptimalPolicy(BasePolicy):
    def __init__(self, n_arms, reward_function):
        super().__init__(n_arms)
        self.reward_function = reward_function

    def select_arm(self, t, contexts):
        rewards = [self.reward_function(t, context) for context in contexts]
        return int(np.argmax(rewards))

    def update(self, chosen_arm, reward, context):
        pass