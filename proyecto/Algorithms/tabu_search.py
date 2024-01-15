import numpy as np
import time
from collections import deque
from Algorithms.common_function_algorithms import generate_permutation, read_json

def two_opt(solution, i, j):
    new_solution = solution.copy()
    new_solution[i:j + 1] = np.flip(solution[i:j + 1])
    return new_solution

def two_swap(solution, i, j):
    new_solution = solution.copy()
    new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
    return new_solution

def tabu_search(tsp_instance, number_cities, max_time):
    hyperparameters = read_json()
    tabu_list_size = hyperparameters['tabu_list_size']

    def evaluate_solution(solution):
        return tsp_instance.cost(solution)

    current_solution = generate_permutation(number_cities)
    current_cost = evaluate_solution(current_solution)
    best_solution = current_solution
    best_cost = current_cost
    current_costs = []
    tabu_list = deque(maxlen=tabu_list_size)
    evals = 0
    start_time = time.time()

    while time.time() - start_time < max_time:

        neighbors = []

        # Generate neighbors using 2-opt and 2-swap operators
        for i in range(number_cities):
            for j in range(i + 1, number_cities):
                neighbors.append(two_opt(current_solution, i, j))
                neighbors.append(two_swap(current_solution, i, j))

        # Remove neighbors that are in the tabu list
        non_tabu_neighbors = [neighbor for neighbor in neighbors if tuple(neighbor) not in tabu_list]

        neighbor_costs = []

        # Evaluate the non-tabu neighbors
        for i in range(len(non_tabu_neighbors)):
            tsp_instance.evaluate_and_resgister(non_tabu_neighbors[i], 'TS')
            current_costs.append(current_cost)
            neighbor_costs.append(evaluate_solution(non_tabu_neighbors[i]))
            evals += 1
            if time.time() - start_time > max_time:
                break

        # Find the best non-tabu neighbor
        best_non_tabu_index = np.argmin(neighbor_costs)
        best_non_tabu_neighbor = non_tabu_neighbors[best_non_tabu_index]
        best_non_tabu_cost = neighbor_costs[best_non_tabu_index]

        # Update the current solution with the best non-tabu neighbor
        current_solution = best_non_tabu_neighbor
        current_cost = best_non_tabu_cost

        # Update the tabu list
        tabu_list.append(tuple(best_non_tabu_neighbor))

        # Update the best solution if necessary
        if current_cost < best_cost:
            best_solution = current_solution
            best_cost = current_cost

    return current_costs, best_solution, evals