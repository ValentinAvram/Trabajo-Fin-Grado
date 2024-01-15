import time
import numpy as np

from Algorithms.common_function_algorithms import generate_permutation

def two_opt(solution, i, j):
    new_solution = solution.copy()
    new_solution[i:j + 1] = np.flip(solution[i:j + 1])
    return new_solution


def two_swap(solution, i, j):
    new_solution = solution.copy()
    new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
    return new_solution

def first_local_search(tsp_instance, number_cities, max_time):
    current_solution = generate_permutation(number_cities)
    current_cost = tsp_instance.cost(current_solution)

    best_solution = current_solution
    best_cost = current_cost

    current_costs = []
    no_improvement_count = 0
    max_no_improvement = 200
    evals = 0
    start_time = time.time()
    while time.time() - start_time < max_time:
        neighbors = []

        # Generate neighbors using 2-opt and 2-swap operators
        for i in range(number_cities):
            for j in range(i + 1, number_cities):
                neighbors.append(two_opt(current_solution, i, j))
                neighbors.append(two_swap(current_solution, i, j))

        # Evaluate neighbors and select the best improvement
        for neighbor in neighbors:
            neighbor_cost = tsp_instance.cost(neighbor)

            if neighbor_cost < current_cost:
                current_solution = neighbor
                current_cost = neighbor_cost

            tsp_instance.evaluate_and_resgister(current_solution, 'FLS')
            current_costs.append(current_cost)
            evals += 1
            if time.time() - start_time > max_time:
                break

        # Update the best solution if necessary
        if current_cost < best_cost:
            best_solution = current_solution
            best_cost = current_cost
            no_improvement_count = 0
        else:
            no_improvement_count += 1


        # Break if there's no improvement for a certain number of iterations
        if no_improvement_count >= max_no_improvement:
            break

        # Store current cost for each iteration

    return current_costs, best_solution, evals