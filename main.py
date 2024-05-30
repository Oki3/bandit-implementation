import numpy as np
import matplotlib.pyplot as plt

from Policies.UCB import UCB
from Policies.EXP3 import EXP3
from Policies.LinUCB import LinUCB

# Given parameters
w = np.array([0.44, 0.6, 0.2])
phi = np.array([0.2, 0.1, 0.45])
context_vectors = [
    np.array([0.2, 0.1, 0.3]),
    np.array([0.5, 0.3, 0.4]),
    np.array([0.4, 0.3, 0.7]),
    np.array([0.5, 0.5, 0.35])
]

# Function to calculate reward vector
def calculate_reward_vector(t):
    return np.sin(w * t + phi)

# Function to calculate reward
def calculate_reward(t, context):
    reward_vector = calculate_reward_vector(t)
    return np.dot(reward_vector, context)

# Simulation parameters
num_iterations = 100000
num_repetitions = 1

# Initialize reward arrays
ucb_rewards = np.zeros(num_iterations)
exp3_rewards = np.zeros(num_iterations)
linucb_rewards = np.zeros(num_iterations)

# Run simulations
for _ in range(num_repetitions):
    ucb = UCB(n_arms=len(context_vectors))
    exp3 = EXP3(n_arms=len(context_vectors))
    linucb = LinUCB(n_arms=len(context_vectors), d=len(context_vectors[0]))

    ucb_cumulative_rewards = []
    exp3_cumulative_rewards = []
    linucb_cumulative_rewards = []

    for t in range(1, num_iterations + 1):
        # UCB
        ucb_arm = ucb.select_arm()
        ucb_context = context_vectors[ucb_arm]
        ucb_reward = calculate_reward(t, ucb_context)
        ucb.update(ucb_arm, ucb_reward)
        if t == 1:
            ucb_cumulative_rewards.append(ucb_reward)
        else:
            ucb_cumulative_rewards.append(ucb_cumulative_rewards[-1] + ucb_reward)

        # EXP3
        exp3_arm = exp3.select_arm()
        exp3_context = context_vectors[exp3_arm]
        exp3_reward = calculate_reward(t, exp3_context)
        exp3.update(exp3_arm, exp3_reward, exp3_context)
        if t == 1:
            exp3_cumulative_rewards.append(exp3_reward)
        else:
            exp3_cumulative_rewards.append(exp3_cumulative_rewards[-1] + exp3_reward)

        # LinUCB
        linucb_arm = linucb.select_arm(context_vectors)
        linucb_context = context_vectors[linucb_arm]
        linucb_reward = calculate_reward(t, linucb_context)
        linucb.update(linucb_arm, linucb_reward, linucb_context)
        if t == 1:
            linucb_cumulative_rewards.append(linucb_reward)
        else:
            linucb_cumulative_rewards.append(linucb_cumulative_rewards[-1] + linucb_reward)

    # Accumulate rewards for averaging
    ucb_rewards += np.array(ucb_cumulative_rewards)
    exp3_rewards += np.array(exp3_cumulative_rewards)
    linucb_rewards += np.array(linucb_cumulative_rewards)

# Average rewards over repetitions
ucb_rewards /= num_repetitions
exp3_rewards /= num_repetitions
linucb_rewards /= num_repetitions

# Plot results
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['lines.markersize'] = 0
plt.figure(figsize=(12, 8))
plt.plot(ucb_rewards, label='UCB')
plt.plot(exp3_rewards, label='EXP3')
plt.plot(linucb_rewards, label='LinUCB')
plt.xlabel('Iteration')
plt.ylabel('Cumulative Reward')
plt.title('Average Cumulative Reward Over Time')
plt.legend()
plt.show()

ucb_rewards[-1], exp3_rewards[-1], linucb_rewards[-1]
