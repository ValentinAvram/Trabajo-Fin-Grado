import numpy as np
import time
from Algorithms.common_function_algorithms import read_json

def two_swap(solution, idx1, idx2):
    new_solution = solution[:]
    new_solution[idx1], new_solution[idx2] = new_solution[idx2], new_solution[idx1]
    return new_solution


def individual_learning(solution, number_cities):
    idx1, idx2 = np.random.choice(number_cities, 2, replace=False)
    return two_swap(solution, idx1, idx2)


def social_learning(tsp_instance, solution, population, number_cities):
    idx1, idx2 = np.random.choice(len(population), 2, replace=False)
    other_solution = population[idx1]
    other_solution_cost = tsp_instance.cost(other_solution)
    current_cost = tsp_instance.cost(solution)

    if other_solution_cost < current_cost:
        return individual_learning(solution, number_cities)
    else:
        return solution


def cultural_algorithm(tsp_instance, number_cities, max_time):
    # Inicialización
    hyperparameters = read_json()
    population_size = hyperparameters['population_size_CA']
    elite_size = hyperparameters['elite_size_CA']

    evals = 0
    best_solution = None
    best_cost = float('inf')
    current_costs = []

    # Generación de la población inicial
    population = [np.random.permutation(number_cities) for _ in range(population_size)]

    start_time = time.time()
    while time.time() - start_time < max_time:
        # Evaluación de la población
        costs = [tsp_instance.cost(solution) for solution in population]

        # Actualizar la mejor solución
        min_cost_idx = np.argmin(costs)
        if costs[min_cost_idx] < best_cost:
            best_solution = population[min_cost_idx]
            best_cost = costs[min_cost_idx]

        # Selección de elite
        elite_indices = np.argsort(costs)[:elite_size]
        elite_population = [population[i] for i in elite_indices]

        # Aplicar operadores de vecindario y aprendizaje individual
        new_population = []
        for solution in elite_population:
            new_population.append(solution)
            new_population.append(individual_learning(solution, number_cities))
            current_costs.append(tsp_instance.cost(solution))
            tsp_instance.evaluate_and_resgister(solution, 'CA')
            evals += 1

            if time.time() - start_time > max_time:
                break

        # Aplicar aprendizaje social
        for solution in elite_population:
            new_population.append(social_learning(tsp_instance, solution, elite_population, number_cities))

        # Llenar la población con mutaciones y soluciones aleatorias
        while len(new_population) < population_size:
            random_solution = np.random.permutation(number_cities)
            new_population.append(random_solution)

        # Reemplazar la población anterior
        population = new_population

    return current_costs, best_solution, evals