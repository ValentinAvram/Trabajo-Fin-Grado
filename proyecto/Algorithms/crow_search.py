import numpy as np
import time


def crow_search_algorithm(tsp_instance, number_cities, max_time):
    # Parámetros del algoritmo CSA
    num_crows = 10  # Número de cuervos (ajusta según tus necesidades)
    max_evals = 2000  # Número máximo de evaluaciones de la función objetivo

    # Generar la población inicial
    population = [list(np.random.permutation(number_cities)) for _ in range(num_crows)]
    current_costs = []
    evals = 0
    start_time = time.time()
    while time.time() - start_time < max_time:
        for crow in population:
            # Evaluar el costo de la solución actual
            cost = tsp_instance.cost(crow)
            # Realizar búsqueda local en la vecindad
            new_crow, evals = local_search(crow, tsp_instance, start_time, current_costs, max_time, evals)
            new_cost = tsp_instance.cost(new_crow)

            # Actualizar la solución si es mejor
            if new_cost < cost:
                crow[:] = new_crow

    # Encontrar la mejor solución en la población
    best_crow = min(population, key=lambda crow: tsp_instance.cost(crow))

    return current_costs, best_crow, evals


def local_search(solution, tsp_instance, start_time, current_costs, max_time, evals):
    best_solution = solution[:]
    best_cost = tsp_instance.cost(solution)

    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            # Intercambiar dos ciudades
            solution[i], solution[j] = solution[j], solution[i]
            cost = tsp_instance.cost(solution)

            if cost < best_cost:
                best_solution = solution[:]
                best_cost = cost

            # Deshacer el intercambio
            solution[i], solution[j] = solution[j], solution[i]


        tsp_instance.evaluate_and_resgister(best_solution, 'CSA')
        evals += 1
        if time.time() - start_time > max_time:
            break
        current_costs.append(best_cost)
    current_costs.append(best_cost)
    return best_solution, evals
