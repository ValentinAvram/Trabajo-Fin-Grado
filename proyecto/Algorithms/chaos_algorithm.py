import random
import time
from Algorithms.common_function_algorithms import generate_permutation

def chaos_algorithm_optimization(tsp_instance, number_cities, max_time):
    current_permutation = generate_permutation(number_cities)
    current_cost = tsp_instance.cost(current_permutation)
    current_costs = []
    start_time = time.time()
    evals = 0
    while time.time() - start_time < max_time:
        new_permutation = current_permutation.copy()

        # Introduce caos: Realiza una perturbación aleatoria en la permutación actual
        i, j = random.sample(range(number_cities), 2)
        new_permutation[i], new_permutation[j] = new_permutation[j], new_permutation[i]

        new_cost = tsp_instance.cost(new_permutation)

        # Aplicar criterio de aceptación: Si la nueva solución es mejor o se cumple una probabilidad de aceptación
        if new_cost < current_cost or random.random() < 0.01:
            current_permutation = new_permutation
            current_cost = new_cost

        current_costs.append(current_cost)
        tsp_instance.evaluate_and_resgister(new_permutation, 'CAO')
        evals += 1
    return current_costs, current_permutation, evals