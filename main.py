import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from Policies.UCB import UCB
from Policies.EXP3 import EXP3
from Policies.LinUCB import LinUCB
from Policies.LinEXP3 import LinEXP3
from Policies.RandomPolicy import RandomPolicy
from Policies.OptimalPolicy import OptimalPolicy

w = np.array([0.04, 0.06, 0.02])
phi = np.array([0.2, 0.1, 0.45])*0.1
context_vectors = [
    np.array([0.9, 0.1, 0.2]),   # High variance in the first dimension
    np.array([0.1, 0.9, 0.1]),   # High values in the second dimension
    np.array([0.4, 0.4, 0.4]),   # Balanced values across all dimensions
    np.array([0.1, 0.1, 0.1]),   # Low values across all dimensions
    np.array([0.1, 0.2, 0.9]),   # High values in the third dimension
    np.array([0.3, 0.5, 0.6]),   # Moderate values with slight bias towards the second and third dimensions
    np.array([0.6, 0.2, 0.8]),   # Random values with a mix of high and low in different dimensions
    np.array([0.5, 0.5, 0.5]),   # Values centered around mid-range
    np.array([0.3, 0.4, 0.9]),   # High variance in the third dimension
    np.array([0.8, 0.1, 0.1])    # Very low values except for the first dimension
]

# context_vectors = [
#     np.array([0.9, 0.1, 0.2]),   # High variance in the first dimension
#     np.array([0.1, 0.9, 0.1]),
# ]

# Function to calculate reward vector
def calculate_reward_vector(t):
    # sin_values = np.abs(np.sin(w * t + phi))
    # ones_array = np.zeros(3)
    # elementwise_max = np.maximum(sin_values, ones_array)
    # return elementwise_max
    # return np.abs(np.sin(w * t + phi))
    # return np.cos(w*t+phi)
    # return np.log(t)*(w-phi)
    return 10*(w*t + phi)
    # return np.sqrt(t)*(w+phi)
    # return w-phi

# Function to calculate reward
def calculate_reward(t, context):
    reward_vector = calculate_reward_vector(t)
    return np.dot(reward_vector, context)

# Simulation parameters
num_iterations = 5000
num_repetitions = 1

# Initialize reward and regret arrays
reward_records = {policy_name: np.zeros(num_iterations) for policy_name in ["UCB", "EXP3", "LinUCB", "LinEXP3", "Random", "Optimal"]}
regret_records = {policy_name: np.zeros(num_iterations) for policy_name in ["UCB", "EXP3", "LinUCB", "LinEXP3", "Random"]}
arm_selection_counts = {policy_name: np.zeros(len(context_vectors)) for policy_name in ["UCB", "EXP3", "LinUCB", "LinEXP3", "Random", "Optimal"]}
all_cumulative_regrets = {policy_name: np.zeros((num_repetitions, num_iterations)) for policy_name in ["UCB", "EXP3", "LinUCB", "LinEXP3", "Random"]}

# Run simulations
for rep in range(num_repetitions):
    print(f"Turn {rep}...")
    policies = {
        "UCB": UCB(n_arms=len(context_vectors)),
        "EXP3": EXP3(n_arms=len(context_vectors), gamma=0.1),
        "LinUCB": LinUCB(n_arms=len(context_vectors), d=len(context_vectors[0]), alpha=0.5),
        "LinEXP3": LinEXP3(n_arms=len(context_vectors), dimension=len(context_vectors[0]), eta=0.55, gamma=0.2),
        "Random": RandomPolicy(n_arms=len(context_vectors), d=len(context_vectors[0])),
        "Optimal": OptimalPolicy(n_arms=len(context_vectors), reward_function=calculate_reward)
    }

    cumulative_rewards = {policy_name: [] for policy_name in policies.keys()}

    for t in range(1, num_iterations + 1):
        for name, policy in policies.items():
            if name == "Optimal" or name == "LinEXP3":
                arm = policy.select_arm(t, context_vectors)
            else:
                arm = policy.select_arm(context_vectors)
            context = context_vectors[arm]
            reward = calculate_reward(t, context)
            if name == "Optimal":
                policy.update(arm, reward, context)
            elif name == "LinEXP3":
                policy.update(arm, reward, context_vectors, t)
            else:
                policy.update(arm, reward, context)

            arm_selection_counts[name][arm] += 1

            if t == 1:
                cumulative_rewards[name].append(reward)
            else:
                cumulative_rewards[name].append(cumulative_rewards[name][-1] + reward)

    for name in reward_records.keys():
        reward_records[name] += np.array(cumulative_rewards[name])

    for name in regret_records.keys():
        regret_records[name] += np.array(cumulative_rewards["Optimal"]) - np.array(cumulative_rewards[name])

    for name in all_cumulative_regrets.keys():
        all_cumulative_regrets[name][rep, :] = np.array(cumulative_rewards["Optimal"]) - np.array(cumulative_rewards[name])

# Average rewards and regrets over repetitions
for name in reward_records.keys():
    reward_records[name] /= num_repetitions

for name in regret_records.keys():
    regret_records[name] /= num_repetitions

# Calculate final cumulative regret
final_cumulative_regret = {name: regrets[-1] for name, regrets in regret_records.items()}

# Sort policies by their final cumulative regret
sorted_policies = sorted(final_cumulative_regret.items(), key=lambda x: x[1])

# Print the final ranking with detailed arm pull counts
print("Final Ranking of Policies by Cumulative Regret:")
for rank, (name, regret) in enumerate(sorted_policies, 1):
    arm_pulls = ', '.join([f"Arm {i}: {int(count)}" for i, count in enumerate(arm_selection_counts[name])])
    print(f"{rank}. {name}: {regret}, Arm Pulls: [{arm_pulls}]")

# Print arm selections for Optimal policy
optimal_arm_pulls = ', '.join([f"Arm {i}: {int(count)}" for i, count in enumerate(arm_selection_counts["Optimal"])])
print(f"\nOptimal Policy Arm Pulls: [{optimal_arm_pulls}]")

def plot_cumulative_rewards():
    plt.figure(figsize=(12, 8))
    for name, rewards in reward_records.items():
        if name != "Optimal":
            plt.plot(rewards, label=name)
    plt.xlabel('Iteration')
    plt.ylabel('Cumulative Reward')
    plt.title('Average Cumulative Reward Over Time')
    plt.legend()
    plt.show()

def plot_cumulative_regrets():
    plt.figure(figsize=(12, 8))
    for name, regrets in regret_records.items():
        plt.plot(regrets, label=name)
    plt.xlabel('Iteration')
    plt.ylabel('Cumulative Regret')
    plt.title('Cumulative Regret Over Time')
    plt.legend()
    plt.show()

def plot_normal_distributions_of_cumulative_regret():
    plt.figure(figsize=(12, 8))
    x = np.linspace(-100, 1000, 1000)

    for name, regrets in all_cumulative_regrets.items():
        final_regrets = regrets[:, -1]
        mean, std_dev = np.mean(final_regrets), np.std(final_regrets)
        p = stats.norm.pdf(x, mean, std_dev)
        plt.plot(x, p, label=f"{name} (mean={mean:.2f}, std={std_dev:.2f})")

    plt.xlabel('Cumulative Regret')
    plt.ylabel('Density')
    plt.title('Normal Distribution of Cumulative Regret for 20 Realizations')
    plt.legend()
    plt.show()

def plot_cumulative_regrets_with_confidence_intervals():
    plt.figure(figsize=(12, 8))
    for name, regrets in regret_records.items():
        mean_regrets = np.mean(regrets, axis=0)
        std_dev = np.std(regrets, axis=0)
        confidence_interval = 1.96 * std_dev / np.sqrt(regrets.shape[0])  # 95% confidence interval

        plt.plot(mean_regrets, label=name)
        plt.fill_between(range(len(mean_regrets)), 
                            mean_regrets - confidence_interval, 
                            mean_regrets + confidence_interval, 
                            alpha=0.2)

    plt.xlabel('Iteration')
    plt.ylabel('Cumulative Regret')
    plt.title('Cumulative Regret Over Time with Confidence Intervals')
    plt.legend()
    plt.show()

plot_cumulative_rewards()
plot_cumulative_regrets()
plot_normal_distributions_of_cumulative_regret()
plot_cumulative_regrets_with_confidence_intervals()