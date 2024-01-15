import random
import time
import numpy as np

from Algorithms.common_function_algorithms import generate_permutation

def harmony_search(tsp_instance, number_cities, max_time):
    # Inicializar una matriz de armonías aleatorias sin repetición
    harmonies = [generate_permutation(number_cities) for _ in range(10)]

    # Inicializar la mejor solución encontrada
    best_permutation = harmonies[0]
    best_cost = tsp_instance.cost(best_permutation)
    evals = 0
    current_costs = []
    start_time = time.time()
    while time.time() - start_time < max_time:
        # Calcular el costo de la armonía actual
        current_permutation = harmonies[0]
        current_cost = tsp_instance.cost(current_permutation)

        tsp_instance.evaluate_and_resgister(current_permutation, 'HS')
        evals += 1
        # Actualizar la mejor solución si es necesario
        if current_cost < best_cost:
            best_permutation = current_permutation
            best_cost = current_cost

        # Actualizar la lista de costos
        current_costs.append(current_cost)

        # Generar una nueva armonía aleatoria sin repetir números
        new_permutation = random.sample(range(0, number_cities), number_cities)

        # Reemplazar la armonía actual si la nueva armonía es mejor
        if tsp_instance.cost(new_permutation) < current_cost:
            harmonies[0] = new_permutation

    return current_costs, best_permutation, evals