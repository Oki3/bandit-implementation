import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from Policies.UCB import UCB
from Policies.EXP3 import EXP3
from Policies.LinUCB import LinUCB
from Policies.LinEXP3 import LinEXP3
from Policies.RandomPolicy import RandomPolicy
from Policies.OptimalPolicy import OptimalPolicy

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
    return np.sin(w * t*100 + phi)
    # return np.cos(w*t+phi)
    # return np.sqrt(t)*np.dot(w, phi)

# Function to calculate reward
def calculate_reward(t, context):
    reward_vector = calculate_reward_vector(t)
    return np.dot(reward_vector, context)

# Simulation parameters
num_iterations = 1000
num_repetitions = 20

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
        "EXP3": EXP3(n_arms=len(context_vectors)),
        "LinUCB": LinUCB(n_arms=len(context_vectors), d=len(context_vectors[0])),
        "LinEXP3": LinEXP3(n_arms=len(context_vectors), dimension=len(context_vectors[0]), eta=0.5, gamma=0.2),
        "Random": RandomPolicy(n_arms=len(context_vectors), d=len(context_vectors[0])),
        "Optimal": OptimalPolicy(n_arms=len(context_vectors), reward_function=calculate_reward)
    }

    cumulative_rewards = {policy_name: [] for policy_name in policies.keys()}

    for t in range(1, num_iterations + 1):
        for name, policy in policies.items():
            if name == "Optimal":
                arm = policy.select_arm(t, context_vectors)
            else:
                arm = policy.select_arm(context_vectors)
            context = context_vectors[arm]
            reward = calculate_reward(t, context)
            if name == "Optimal":
                policy.update(arm, reward, context)
            elif name == "LinEXP3":
                policy.update(arm, reward, context_vectors)
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

# Plot cumulative rewards
# plot_cumulative_rewards()

# Plot cumulative regrets
plot_cumulative_regrets()

# Plot normal distributions of cumulative regrets for 20 realizations
def plot_normal_distributions_of_cumulative_regret():
    plt.figure(figsize=(12, 8))
    x = np.linspace(-100, 1000, 1000)  # Define the range for the x-axis

    for name, regrets in all_cumulative_regrets.items():
        final_regrets = regrets[:, -1]  # Use the final cumulative regret
        mean, std_dev = np.mean(final_regrets), np.std(final_regrets)
        p = stats.norm.pdf(x, mean, std_dev)
        plt.plot(x, p, label=f"{name} (mean={mean:.2f}, std={std_dev:.2f})")

    plt.xlabel('Cumulative Regret')
    plt.ylabel('Density')
    plt.title('Normal Distribution of Cumulative Regret for 20 Realizations')
    plt.legend()
    plt.show()

# Plot normal distributions of cumulative regrets
plot_normal_distributions_of_cumulative_regret()
