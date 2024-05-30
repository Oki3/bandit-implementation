import numpy as np
import matplotlib.pyplot as plt

from Policies import UCB, EXP3, LinUCB

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

# Simulation
num_iterations = 10000
ucb_rewards = []
exp3_rewards = []
linucb_rewards = []

ucb = UCB(n_arms=len(context_vectors))
exp3 = EXP3(n_arms=len(context_vectors))
linucb = LinUCB(n_arms=len(context_vectors), d=len(context_vectors[0]))

for t in range(1, num_iterations + 1):
    # UCB
    ucb_arm = ucb.select_arm()
    ucb_context = context_vectors[ucb_arm]
    ucb_reward = calculate_reward(t, ucb_context)
    ucb.update(ucb_arm, ucb_reward)
    if t == 1:
        ucb_rewards.append(ucb_reward)
    else:
        ucb_rewards.append(ucb_rewards[-1] + ucb_reward)

    # EXP3
    exp3_arm = exp3.select_arm()
    exp3_context = context_vectors[exp3_arm]
    exp3_reward = calculate_reward(t, exp3_context)
    exp3.update(exp3_arm, exp3_reward, exp3_context)
    if t == 1:
        exp3_rewards.append(exp3_reward)
    else:
        exp3_rewards.append(exp3_rewards[-1] + exp3_reward)

    # LinUCB
    linucb_arm = linucb.select_arm(context_vectors)
    linucb_context = context_vectors[linucb_arm]
    linucb_reward = calculate_reward(t, linucb_context)
    linucb.update(linucb_arm, linucb_reward, linucb_context)
    if t == 1:
        linucb_rewards.append(linucb_reward)
    else:
        linucb_rewards.append(linucb_rewards[-1] + linucb_reward)

# Plot results
plt.figure(figsize=(12, 8))
plt.plot(ucb_rewards, label='UCB')
plt.plot(exp3_rewards, label='EXP3')
plt.plot(linucb_rewards, label='LinUCB')
plt.xlabel('Iteration')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Reward Over Time')
plt.legend()
plt.show()

ucb_rewards[-1], exp3_rewards[-1], linucb_rewards[-1]
