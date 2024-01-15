import numpy as np
from Algorithms.common_function_algorithms import generate_permutation

def two_opt(tour, i, j):
    new_tour = tour.copy()
    new_tour[i:j + 1] = np.flip(tour[i:j + 1])
    return new_tour

def two_swap(tour, i, j):
    new_tour = tour.copy()
    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return new_tour

def water_cycle_algorithm(tsp_instance, number_cities, max_evals, results_list):
    current_permutation = generate_permutation(number_cities)
    current_cost = tsp_instance.cost(current_permutation)
    current_costs = []

    best_permutation = current_permutation  # Initialize best_permutation
    best_cost = current_cost

    evals = 0
    while evals < max_evals:
        evals += 1

        if evals % number_cities == 0:
            results_list.append(best_cost)

        if np.random.rand() < 0.5:
            idx1, idx2 = np.random.choice(number_cities, 2, replace=False)
            new_permutation = two_opt(current_permutation, idx1, idx2)
        else:
            idx1, idx2 = np.random.choice(number_cities, 2, replace=False)
            new_permutation = two_swap(current_permutation, idx1, idx2)

        new_cost = tsp_instance.cost(new_permutation)

        if new_cost < current_cost:
            current_permutation = new_permutation
            current_cost = new_cost

            if new_cost < best_cost:  # Update best_permutation and best_cost
                best_permutation = new_permutation
                best_cost = new_cost

        current_costs.append(current_cost)  # Moved this line

    return current_costs, best_permutation
