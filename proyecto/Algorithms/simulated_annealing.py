import numpy as np
import math
import time
from Algorithms.common_function_algorithms import read_json, generate_permutation

def two_opt(permutation, idx1, idx2):
    new_permutation = permutation.copy()
    new_permutation[idx1:idx2] = np.flip(permutation[idx1:idx2])
    return new_permutation

def two_swap(permutation, idx1, idx2):
    new_permutation = permutation.copy()
    new_permutation[idx1], new_permutation[idx2] = new_permutation[idx2], new_permutation[idx1]
    return new_permutation

def simulated_annealing(tsp_instance, number_cities, max_time):
    # Initial solution (random permutation of cities)
    current_permutation =generate_permutation(number_cities)
    current_cost = tsp_instance.cost(current_permutation)

    # Initial temperature and cooling rate
    hyperparameters = read_json()
    initial_temperature = hyperparameters['initial_temperature_SA']
    cooling_rate = hyperparameters['cooling_rate_SA']

    current_costs = []  # To store costs at each iteration
    evals = 0
    start_time = time.time()

    while time.time() - start_time < max_time:
        temperature = initial_temperature * math.pow(cooling_rate, max_time)

        # Randomly choose a neighborhood operator
        if np.random.rand() < 0.5:
            idx1, idx2 = np.random.choice(number_cities, size=2, replace=False)
            new_permutation = two_opt(current_permutation, idx1, idx2)
        else:
            idx1, idx2 = np.random.choice(number_cities, size=2, replace=False)
            new_permutation = two_swap(current_permutation, idx1, idx2)

        new_cost = tsp_instance.cost(new_permutation)
        cost_diff = new_cost - current_cost

        # Acceptance criterion
        if cost_diff < 0 or np.random.rand() < math.exp(-cost_diff / temperature):
            current_permutation = new_permutation
            current_cost = new_cost

        current_costs.append(current_cost)
        tsp_instance.evaluate_and_resgister(new_permutation, 'SimAn')
        evals += 1
    return current_costs, current_permutation, evals