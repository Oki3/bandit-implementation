import numpy as np
from BasePolicy import BasePolicy

class UCB(BasePolicy):
    def __init__(self, n_arms, alpha=1.0):
        super().__init__(n_arms)
        self.alpha = alpha
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.choices = []
        self.total_counts = 0
    
    def select_arm(self, contexts):
        if self.total_counts < self.n_arms:
            return int(self.total_counts)
        ucb_values = self.values + self.alpha * np.sqrt((np.log(self.total_counts)) / (self.counts))
        choice = int(np.argmax(ucb_values))
        self.choices.append(choice)
        return choice
    
    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        self.total_counts += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[chosen_arm] = new_value

    def printChosenPerArm(self, interval=90):
        total_intervals = int(np.ceil(self.total_counts / interval))
        
        for i in range(total_intervals):
            start = i * interval
            end = min((i + 1) * interval, self.total_counts)
            if end < self.total_counts:
                interval_counts = np.zeros(self.n_arms)
                
                for j in range(start, end):
                    arm = self.choices[j]
                    interval_counts[arm] += 1
                    
                print(f"UCB: From count {start} to {end}:")
                for arm in range(self.n_arms):
                    print(f"  Arm {arm} was chosen {int(interval_counts[arm])} times.")
