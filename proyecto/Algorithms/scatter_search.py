import numpy as np
import random
import time
from Algorithms.common_function_algorithms import generate_permutation

def scatter_search(tsp_instance, number_cities, max_time):
    current_solution = generate_permutation(number_cities)
    best_solution = current_solution.copy()
    current_costs = []
    evals = 0
    start_time = time.time()
    while time.time() - start_time < max_time:
        # Generar una nueva solución a partir de la perturbación de la mejor solución
        perturbed_solution = perturb(best_solution)
        # Realizar una búsqueda local en la solución perturbada
        local_solution, evals = local_search(tsp_instance, perturbed_solution, current_costs, start_time, max_time, evals)
        # Actualizar la mejor solución si es necesario
        if tsp_instance.cost(local_solution) < tsp_instance.cost(best_solution):
            best_solution = local_solution.copy()
        current_solution = select_best_solutions(tsp_instance ,current_solution, local_solution)

    return current_costs, best_solution, evals

def perturb(solution):
    # Aplicar una perturbación simple intercambiando dos elementos aleatorios
    perturbed_solution = solution.copy()
    i, j = random.sample(range(len(solution)), 2)
    perturbed_solution[i], perturbed_solution[j] = perturbed_solution[j], perturbed_solution[i]
    return perturbed_solution

def local_search(tsp_instance, solution, current_costs, start_time, max_time, evals):
    # Implementa una búsqueda local simple intercambiando pares de elementos para mejorar la solución
    current_cost = tsp_instance.cost(solution)
    improved = True
    while improved:
        improved = False
        for i in range(len(solution)):
            for j in range(i+1, len(solution)):
                new_solution = solution.copy()
                new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
                new_cost = tsp_instance.cost(new_solution)
                if new_cost < current_cost:
                    solution = new_solution.copy()
                    current_cost = new_cost
                    improved = True

            if time.time() - start_time > max_time:
                break

            current_costs.append(tsp_instance.cost(solution))
            tsp_instance.evaluate_and_resgister(solution, 'SS')
            evals += 1
    return solution, evals

def select_best_solutions(tsp_instance, current_solution, local_solution):
    # Actualizar la lista de soluciones actuales seleccionando la mejor
    solutions = [current_solution, local_solution]
    costs = [tsp_instance.cost(solution) for solution in solutions]
    best_index = np.argmin(costs)
    return solutions[best_index]
