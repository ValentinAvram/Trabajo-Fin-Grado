import numpy as np
import random
import time
from Algorithms.common_function_algorithms import read_json

def bat_algorithm(tsp_instance, number_cities, max_time):
    hyperparameters = read_json()
    population_size = hyperparameters["population_size_BA"]
    A = hyperparameters["A"]
    r_min = hyperparameters["r_min"]
    r_max = hyperparameters["r_max"]
    alpha = hyperparameters["alpha_BA"]
    gamma = hyperparameters["gamma_BA"]

    # Generar una población inicial de soluciones aleatorias
    population = [list(np.random.permutation(number_cities)) for _ in range(population_size)]

    # Inicializar la lista de costos actuales
    current_costs = []
    evals = 0
    # Iterar hasta alcanzar el número máximo de evaluaciones

    start_time = time.time()
    while time.time() - start_time < max_time:

        for i in range(population_size):
            evals += 1
            # Generar una nueva solución aleatoria
            new_solution = list(np.random.permutation(number_cities))

            # Aplicar una perturbación según la frecuencia de los murciélagos
            if random.random() > r_min:
                for j in range(number_cities):
                    if random.random() < A:
                        perturbation = int(alpha * (random.randint(-1, 1)))
                        new_solution[j] = (new_solution[j] + perturbation) % number_cities

            # Evaluar la nueva solución
            cost_new = tsp_instance.cost(new_solution)
            cost_current = tsp_instance.cost(population[i])

            # Si la nueva solución es mejor o es aceptada según una probabilidad
            if cost_new < cost_current or random.random() < gamma:
                population[i] = new_solution

            current_costs.append(tsp_instance.cost(population[i]))
            tsp_instance.evaluate_and_resgister(population[i], 'BA')
            evals += 1
            if time.time() - start_time > max_time:
                break

        # Actualizar la frecuencia mínima y máxima
        r_min = r_min + (r_max - r_min) * (1 - np.exp(-alpha * evals))

    # Encontrar la mejor solución en la población
    best_solution = min(population, key=lambda x: tsp_instance.cost(x))

    return current_costs, best_solution, evals
