import numpy as np
import time
from Algorithms.common_function_algorithms import read_json

def calculate_attraction(beta0, gamma, distance):
    return beta0 * np.exp(-gamma * distance)

def generate_random_permutation(number_cities):
    return np.random.permutation(number_cities)

def ensure_no_duplicates(arr):
    unique_values = np.unique(arr)
    while len(unique_values) < len(arr):
        for i in range(len(arr)):
            if np.count_nonzero(arr == arr[i]) > 1:
                available_values = np.setdiff1d(np.arange(len(arr)), unique_values)
                if len(available_values) > 0:
                    arr[i] = available_values[0]
                    unique_values = np.unique(arr)
                else:
                    break
    return arr


def firefly_algorithm(tsp_instance, number_cities, max_time_seconds):
    start_time = time.time()

    # Initialize fireflies with random permutations without repetitions
    fireflies = [generate_random_permutation(number_cities) for _ in range(20000)]
    current_costs = [tsp_instance.cost(firefly) for firefly in fireflies]

    hyperparameters = read_json()
    alpha = hyperparameters['alphaFA']
    beta0 = hyperparameters['betaFA']
    gamma = hyperparameters['gammaFA']
    sigma = hyperparameters['sigmaFA']

    eval_count = 0
    count = 0
    while True:
        current_permutation = fireflies[eval_count].copy()  # Copy to avoid modifying the array in-place
        current_cost = current_costs[eval_count]

        if current_costs[eval_count] < current_cost:  # Move toward brighter fireflies
            distance = np.linalg.norm(current_permutation - fireflies[eval_count])
            attraction = calculate_attraction(beta0, gamma, distance)
            direction = (fireflies[eval_count] - current_permutation) * attraction
            step = sigma * np.random.uniform(-1, 1, size=number_cities)
            new_position = current_permutation + alpha * direction + step
            new_position = np.clip(new_position, 0, number_cities - 1).astype(int)  # Convert to integers
            new_position = ensure_no_duplicates(new_position)  # Ensure no duplicates
            new_cost = tsp_instance.cost(new_position)

            if new_cost < current_cost:
                current_permutation = new_position
                current_cost = new_cost

        tsp_instance.evaluate_and_resgister(current_permutation, 'FA')

        fireflies[eval_count] = current_permutation
        current_costs[eval_count] = current_cost

        eval_count += 1
        count += 1
        if eval_count >= len(fireflies):
            eval_count = 0  # Reset eval_count when it reaches the end of the population

        if time.time() - start_time >= max_time_seconds:
            break

    best_idx = np.argmin(current_costs)
    best_permutation = fireflies[best_idx]
    return current_costs, best_permutation, eval_count
