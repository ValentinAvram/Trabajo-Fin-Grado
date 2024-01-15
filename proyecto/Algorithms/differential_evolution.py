import random
import time
from Algorithms.common_function_algorithms import generate_permutation, read_json

def differential_evolution(tsp_instance, number_cities, max_time):
    # Inicialización
    hyperparameters = read_json()
    population_size = hyperparameters['population_size_DE']
    mutation_factor = hyperparameters['mutation_factor_DE']
    crossover_prob = hyperparameters['crossover_prob_DE']

    population = [generate_permutation(number_cities) for _ in range(population_size)]
    best_solution = None
    best_cost = float('inf')

    # Función para calcular el costo de una permutación
    def calculate_cost(perm):
        return tsp_instance.cost(perm)

    # Evaluación inicial de la población
    #current_costs = [calculate_cost(perm) for perm in population]
    current_costs = []
    evals = 0
    start_time = time.time()
    while time.time() - start_time < max_time:
        for i in range(population_size):

            # Selección de tres individuos aleatorios distintos
            a, b, c = random.sample(range(population_size), 3)
            target_vector = list(population[i])

            # Mutación (Diferencial)
            if random.random() < mutation_factor:
                for j in range(number_cities):
                    perturbation = (population[a][j] +
                                    mutation_factor * (population[b][j] - population[c][j]))
                    target_vector[j] = int(perturbation) % number_cities

                    # Asegurarse de que no haya elementos duplicados
                    used_elements = []
                    for k in range(number_cities):
                        if target_vector[k] not in used_elements:
                            used_elements.append(target_vector[k])
                        else:
                            available_values = [v for v in range(number_cities) if v not in used_elements]
                            target_vector[k] = random.choice(available_values)
                            used_elements.append(target_vector[k])

            # Cruce (Binomial)
            if random.random() < crossover_prob:
                trial_vector = list(population[i])
                for j in range(number_cities):
                    trial_vector[j] = target_vector[j]

                    # Asegurarse de que no haya elementos duplicados
                    used_elements = []
                    for k in range(number_cities):
                        if trial_vector[k] not in used_elements:
                            used_elements.append(trial_vector[k])
                        else:
                            available_values = [v for v in range(number_cities) if v not in used_elements]
                            trial_vector[k] = random.choice(available_values)
                            used_elements.append(trial_vector[k])

                # Evaluación del trial vector
                new_costs = [calculate_cost(perm) for perm in population]
                trial_cost = calculate_cost(trial_vector)

                # Reemplazo si es mejor que el actual
                if trial_cost < new_costs[i]:
                    #current_costs[i] = trial_cost
                    population[i] = trial_vector

                    # Actualizar mejor solución global
                    if trial_cost < best_cost:
                        best_cost = trial_cost
                        best_solution = trial_vector

            current_costs.append(tsp_instance.cost(population[i]))
            tsp_instance.evaluate_and_resgister(population[i], 'DE')
            evals += 1
            if time.time() - start_time > max_time:
                break

    return current_costs, best_solution, evals
