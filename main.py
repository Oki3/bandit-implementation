import numpy as np
import pandas as pd

from SMPyBandits.Arms import Gaussian
from SMPyBandits.Policies import UCB, Exp3
# from custom_policies.LinUCB import LinUCB

num_arms = 5
time_steps = 1000

# 10M is too much, it is expected to split the dataset into training/validation/test set to 70%/15%/15% respectively.
chunk_size = 1000

# Process the file in chunks
# for chunk in pd.read_csv('pcb_dataset_final.tsv', delimiter='\t', chunksize=chunk_size):
#     print(chunk.head())


#########################
# Algorithm logic below #
#########################
np.random.seed(0)

# Initialize the arms
arms = [Gaussian(np.random.uniform(0.1, 0.9), 0.1) for _ in range(num_arms)]

# Define a function to run the simulation for a given policy
def run_simulation(policy, arms, time_steps):
    history = []
    for t in range(time_steps):
        try:
            # Stochastic bandit environment
            chosen_arm = policy.choice()
            reward = arms[chosen_arm].draw()
            policy.getReward(chosen_arm, reward)

            # Record history (optional)
            history.append((chosen_arm, reward))
        except Exception as e:
            print(f"Error on iteration {t}: {e}")
            break
    
    return history

# Initialize the policies in a dictionary
policies = {
    'UCB': UCB(nbArms=num_arms),
    'EXP3': Exp3(nbArms=num_arms, gamma=0.1),
}

# Run simulations for all policies
results = {}
for policy_name, policy in policies.items():
    results[policy_name] = run_simulation(policy, arms, time_steps)

# Display results
for policy_name, result in results.items():
    print(f"Results for {policy_name}: {result[:10]}...")  # Print the first 10 results for brevity
