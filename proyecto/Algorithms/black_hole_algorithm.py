import numpy as np
import time
from Algorithms.common_function_algorithms import generate_permutation


def two_opt(solution, i, j):
    new_solution = solution.copy()
    new_solution[i:j + 1] = np.flip(solution[i:j + 1])
    return new_solution

def two_swap(solution, i, j):
    new_solution = solution.copy()
    new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
    return new_solution

def calculate_attraction(agent_cost, black_hole_cost, max_cost):
    return np.exp(-agent_cost / max_cost) / (np.exp(-black_hole_cost / max_cost) + 1e-10)


def black_hole_algorithm(tsp_instance, number_cities, max_time):
    current_solutions = [generate_permutation(number_cities) for _ in range(number_cities)]
    current_costs = [tsp_instance.cost(solution) for solution in current_solutions]
    max_cost = max(current_costs)
    results_list = []
    evals = 0
    start_time = time.time()
    while time.time() - start_time < max_time:
        black_hole_index = np.argmin(current_costs)
        black_hole_cost = current_costs[black_hole_index]

        for agent_index in range(number_cities):

            agent_solution = current_solutions[agent_index]
            agent_cost = current_costs[agent_index]

            attraction = calculate_attraction(agent_cost, black_hole_cost, max_cost)

            if np.random.rand() < attraction:
                idx1, idx2 = np.random.choice(number_cities, size=2, replace=False)

                if np.random.rand() < 0.5:
                    agent_solution = two_opt(agent_solution, idx1, idx2)
                else:
                    agent_solution = two_swap(agent_solution, idx1, idx2)

                agent_cost = tsp_instance.cost(agent_solution)
                current_solutions[agent_index] = agent_solution
                current_costs[agent_index] = agent_cost
                max_cost = max(max_cost, agent_cost)

            tsp_instance.evaluate_and_resgister(current_solutions[agent_index], 'BHA')
            results_list.append(agent_cost)
            evals += 1
            if time.time() - start_time > max_time:
                break

    best_solution_index = np.argmin(current_costs)
    best_solution = current_solutions[best_solution_index]

    return results_list, best_solution, evals
