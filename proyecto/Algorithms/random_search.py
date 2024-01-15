import numpy as np
import time
from Algorithms.common_function_algorithms import generate_permutation

def random_search(tsp_instance, number_cities, time_limit):
    current_costs = []
    best_cost = np.inf
    best_permutation = None
    start_time = time.time()
    evals = 0
    while time.time() - start_time < time_limit:
        current_permutation = generate_permutation(number_cities)
        current_costs.append(tsp_instance.evaluate_and_resgister(current_permutation, 'RS'))
        if current_costs[-1] < best_cost:
            best_cost = current_costs[-1]
            best_permutation = current_permutation
        evals += 1

    return current_costs, best_permutation, evals