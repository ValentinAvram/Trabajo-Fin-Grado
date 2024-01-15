import time
from Algorithms.common_function_algorithms import generate_permutation

def guided_local_search(tsp_instance, number_cities, max_time):
    current_permutation = generate_permutation(number_cities)  # Permutación inicial
    best_permutation = current_permutation.copy()  # Mejor permutación hasta el momento
    best_cost = tsp_instance.cost(current_permutation)  # Costo de la mejor solución
    current_costs = []  # Lista para almacenar los costos en cada iteración

    evals = 0

    def calculate_delta_cost(neighbor_permutation):
        return tsp_instance.cost(neighbor_permutation) - tsp_instance.cost(current_permutation)

    def select_best_neighbor(current_permutation, evals):
        neighbors = []
        for i in range(number_cities):
            evals += 1
            for j in range(i + 1, number_cities):
                neighbor_permutation = current_permutation[:]
                neighbor_permutation[i], neighbor_permutation[j] = neighbor_permutation[j], neighbor_permutation[i]
                neighbors.append((neighbor_permutation, calculate_delta_cost(neighbor_permutation)))
            neighbors.sort(key=lambda x: x[1])
            current_costs.append(tsp_instance.cost(neighbors[0][0]))
            tsp_instance.evaluate_and_resgister(neighbors[0][0], 'GLS')
        return neighbors[0][0], neighbors[0][1], evals

    start_time = time.time()
    while time.time() - start_time < max_time:
        best_neighbor, delta_cost, evals = select_best_neighbor(current_permutation, evals) # Seleccionar el mejor vecino

        # Actualizar la solución actual si el vecino es mejor
        if delta_cost < 0:
            current_permutation = best_neighbor[:]
            current_cost = tsp_instance.cost(current_permutation)
            if current_cost < best_cost:
                best_permutation = current_permutation[:]
                best_cost = current_cost

    return current_costs, best_permutation, evals