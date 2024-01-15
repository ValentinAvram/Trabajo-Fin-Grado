import random
import time

import numpy as np

from Algorithms.common_function_algorithms import generate_permutation

def two_opt(solution, i, j):
    new_solution = solution.copy()
    new_solution[i:j + 1] = np.flip(solution[i:j + 1])
    return new_solution

def evaluate_solution(solution, tsp_instance):
    return tsp_instance.cost(solution)

def bee_colony_optimization(tsp_instance, number_cities, max_time):
    current_solution = generate_permutation(number_cities)
    current_cost = evaluate_solution(current_solution, tsp_instance)
    current_costs = []
    evals = 0
    start_time = time.time()
    while time.time() - start_time < max_time:
        # Employed Bees Phase
        new_solution = two_opt(current_solution, random.randint(0, number_cities - 1),
                               random.randint(0, number_cities - 1))
        new_cost = evaluate_solution(new_solution, tsp_instance)

        if new_cost < current_cost:
            current_solution = new_solution
            current_cost = new_cost

        # Onlooker Bees Phase
        for _ in range(5):
            onlooker_solution = two_opt(current_solution, random.randint(0, number_cities - 1),
                                        random.randint(0, number_cities - 1))
            onlooker_cost = evaluate_solution(onlooker_solution, tsp_instance)

            current_costs.append(current_cost)
            tsp_instance.evaluate_and_resgister(current_solution, 'BCO')
            evals += 1
            if onlooker_cost < current_cost:
                current_solution = onlooker_solution
                current_cost = onlooker_cost

        # Scout Bees Phase
        if random.random() < 0.1:
            current_solution = generate_permutation(number_cities)
            current_cost = evaluate_solution(current_solution, tsp_instance)

    return current_costs, current_solution, evals
