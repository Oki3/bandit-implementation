import numpy as np
import pandas as pd

from SMPyBandits.Arms import Gaussian
from SMPyBandits.Policies import UCB, Exp3
from SMPyBandits.Environment import Evaluator, tqdm

import random


policies = [
    {"archtype": UCB, "params": {}},
    {"archtype": Exp3, "params": {"gamma": 0.01}},
]

environments = [ 
        {
            "arm_type": Gaussian,
            "params": [(np.random.uniform(0.1, 0.9), 0.1), 
                       (np.random.uniform(0.1, 0.9), 0.1), 
                       (np.random.uniform(0.1, 0.9), 0.1),
                       (np.random.uniform(0.1, 0.9), 0.1),
                       (np.random.uniform(0.1, 0.9), 0.1)]
        }
    ]

config = {
    "horizon": 1000,
    "repetitions": 10,
    "n_jobs": 4,
    "verbosity": 6,
    "environment": environments,
    "policies": policies,
}

evaluator = Evaluator(config)

# Example of setting a seed with an integer
seed_value = 123  # Ensure this is an integer, or cast it
random.seed(seed_value)  # This should work without error

def safe_set_seed(seed_value):
    """ Safely set the seed for random number generation. """
    try:
        # Attempt to convert to int if not None and not already an int
        if seed_value is not None and not isinstance(seed_value, int):
            seed_value = int(seed_value)
    except (TypeError, ValueError):
        # If the seed is not convertible to int, log it and use a default seed
        print(f"Provided seed ({seed_value}) is invalid. Using a default seed.")
        seed_value = 42  # Default seed
    
    # Set the seed
    random.seed(seed_value)

# Fetch the seed from the configuration, or use None if not set
config_seed = config.get('seed', None)

# Call the safe_set_seed function with the fetched or default seed
safe_set_seed(config_seed)
random.seed(seed_value)

evaluator.startAllEnv()

def plot_env(evaluation, environment_id):
    evaluation.printFinalRanking(environment_id)
    evaluation.plotRegrets(environment_id)
    evaluation.plotRegrets(environment_id, semilogx=True)
    evaluation.plotRegrets(environment_id, meanReward=True)
    # evaluation.plotBestArmPulls(environment_id)


for env_id in range(len(environments)):
    plot_env(evaluator, env_id)